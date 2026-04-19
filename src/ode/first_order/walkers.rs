//! Arc<Expr>-native walkers supporting the first-order ODE solvers.
//!
//! All helpers operate on canonical `Arc<Expr>` and `SymbolId`. Callers that
//! still hold `Expression`-typed inputs compile at the boundary. Signatures
//! avoid redundant compile/decompile round-trips for `ode.rhs_arc()` /
//! `ode.forcing_arc()` consumers (milestone `Expr-migration` task 10.4).

use std::sync::Arc;

use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::normalize;
use crate::numeric::substitute::substitute as arc_substitute;
use crate::numeric::{MulNode, SymbolId};
use crate::solver::helpers::contains_symbol;
use num::traits::One;

/// Attempt to separate `dy/dx = f(x, y)` into `g(x) · h(y)`.
///
/// Pure cases (one variable absent) return immediately. Otherwise the
/// expression must be a canonical product; each factor is assigned to
/// `g(x)` (x-only or constant) or `h(y)` (y-only). Any factor that
/// mentions both variables makes the ODE non-separable.
pub(crate) fn try_separate(
    rhs: &Arc<Expr>,
    x: SymbolId,
    y: SymbolId,
) -> Option<(Arc<Expr>, Arc<Expr>)> {
    let has_x = contains_symbol(rhs, x);
    let has_y = contains_symbol(rhs, y);

    // Pure-x (or constant): h(y) = 1.
    if !has_y {
        return Some((rhs.clone(), Expr::int(1)));
    }
    // Pure-y: g(x) = 1.
    if !has_x {
        return Some((Expr::int(1), rhs.clone()));
    }

    // Mixed → must be a product; iterate the canonical factor map.
    let Expr::Mul(node) = rhs.as_ref() else {
        return None;
    };

    let mut g = MulNode::from_coeff(node.coeff.clone());
    let mut h = MulNode::one();

    for (base, exp) in &node.factors {
        let factor_has_x = contains_symbol(base, x) || contains_symbol(exp, x);
        let factor_has_y = contains_symbol(base, y) || contains_symbol(exp, y);
        if factor_has_x && factor_has_y {
            return None; // factor entangles x and y
        }
        if factor_has_y {
            h.add_factor(base.clone(), exp.clone());
        } else {
            g.add_factor(base.clone(), exp.clone());
        }
    }

    let g_arc: Arc<Expr> = if g.is_one() {
        Expr::int(1)
    } else {
        Arc::new(Expr::Mul(g))
    };
    let h_arc: Arc<Expr> = if h.is_one() {
        Expr::int(1)
    } else {
        Arc::new(Expr::Mul(h))
    };
    Some((g_arc, h_arc))
}

/// Extract P(x) and Q(x) from a linear ODE in form dy/dx = -P(x)·y + Q(x).
///
/// The RHS must be linear in `y`: every AddNode term either is `y`-free or
/// has the form `coeff · y` (possibly with y-free multiplicative factors).
/// Returns `(P(x), Q(x))` where `rhs = -P·y + Q`, or `None` if rhs is
/// non-linear in `y`.
pub(crate) fn extract_linear_coefficients(
    rhs: &Arc<Expr>,
    y: SymbolId,
) -> Option<(Arc<Expr>, Arc<Expr>)> {
    // Pure y-free: P = 0, Q = rhs.
    if !contains_symbol(rhs, y) {
        return Some((Expr::int(0), rhs.clone()));
    }

    // Sum form: walk AddNode terms.
    if let Expr::Add(node) = rhs.as_ref() {
        let mut y_coeff: Arc<Expr> = Expr::int(0);
        let mut q: Arc<Expr> = Arc::new(Expr::Rational(node.constant.clone()));
        for (term, coeff) in &node.terms {
            let (y_c, const_c) = decompose_linear_y_term(term, coeff, y)?;
            y_coeff = normalize::add(y_coeff, y_c);
            q = normalize::add(q, const_c);
        }
        let p = normalize::neg(y_coeff);
        if contains_symbol(&p, y) {
            return None;
        }
        return Some((p, q));
    }

    // Single term (Mul / Symbol / Pow / Func).
    let one = crate::numeric::BigRational::from_integer(crate::numeric::SmallInt::from(1i64));
    let (y_c, const_c) = decompose_linear_y_term(rhs, &one, y)?;
    let p = normalize::neg(y_c);
    if contains_symbol(&p, y) {
        return None;
    }
    Some((p, const_c))
}

/// Decompose a single AddNode term `coeff · term` into (y-contribution,
/// y-free-contribution). Returns `None` when the term is non-linear in
/// `y`, contains `y` inside a function, or raises `y` to any power other
/// than 1.
fn decompose_linear_y_term(
    term: &Arc<Expr>,
    coeff: &crate::numeric::BigRational,
    y: SymbolId,
) -> Option<(Arc<Expr>, Arc<Expr>)> {
    let coeff_expr: Arc<Expr> = Arc::new(Expr::Rational(coeff.clone()));

    // y-free term: all contribution flows to Q.
    if !contains_symbol(term, y) {
        let contrib = normalize::mul(coeff_expr, term.clone());
        return Some((Expr::int(0), contrib));
    }

    // Bare `y`: coefficient becomes the y-coefficient.
    if matches!(term.as_ref(), Expr::Symbol(s) if *s == y) {
        return Some((coeff_expr, Expr::int(0)));
    }

    // Product with `y` as a direct factor of exponent 1.
    if let Expr::Mul(node) = term.as_ref() {
        let mut y_seen = false;
        let mut others = MulNode::from_coeff(node.coeff.clone());
        for (base, exp) in &node.factors {
            let base_is_y = matches!(base.as_ref(), Expr::Symbol(s) if *s == y);
            if base_is_y {
                let exp_is_one = matches!(exp.as_ref(), Expr::Integer(n) if n.to_i64() == Some(1));
                if !exp_is_one || y_seen {
                    return None; // y^n, n != 1, or multiple y factors
                }
                y_seen = true;
                continue;
            }
            if contains_symbol(base, y) || contains_symbol(exp, y) {
                return None; // y buried inside a non-direct factor
            }
            others.add_factor(base.clone(), exp.clone());
        }
        if !y_seen {
            return None;
        }
        let others_arc: Arc<Expr> = if others.is_one() {
            Expr::int(1)
        } else {
            Arc::new(Expr::Mul(others))
        };
        let y_contrib = normalize::mul(coeff_expr, others_arc);
        return Some((y_contrib, Expr::int(0)));
    }

    // `y^n` (n != 1 handled above as Integer/any non-1) or y inside Func → non-linear.
    None
}

/// Substitute `var` with `replacement` in `expr`.
///
/// Thin delegation to [`crate::numeric::substitute::substitute`]; exported
/// under the ODE `walkers` namespace for callsite discoverability.
pub(crate) fn substitute_var(
    expr: &Arc<Expr>,
    var: SymbolId,
    replacement: &Arc<Expr>,
) -> Arc<Expr> {
    arc_substitute(expr, var, replacement)
}

/// Try to solve an implicit relation for `y` explicitly.
///
/// Pattern-matches on the canonical [`Expr`] form. Handles:
///
/// - `y = right` → `right`
/// - `ln(y) = right` or `ln(|y|) = right` → `exp(right)` (positive branch)
/// - `y^n = right` with n y-free → `right^(1/n)`
/// - `1/y = right` → `1/right`
pub(crate) fn try_solve_implicit_for_y(
    left: &Arc<Expr>,
    right: &Arc<Expr>,
    y: SymbolId,
) -> Option<Arc<Expr>> {
    // Case 1: left is just y → y = right
    if matches!(left.as_ref(), Expr::Symbol(s) if *s == y) {
        return Some(right.clone());
    }

    // Case 2: ln(y) = right → y = exp(right); also ln(|y|) = right → y = exp(right)
    if let Expr::Func(FuncId::Ln, args) = left.as_ref() {
        if args.len() == 1 {
            let inner = &args[0];
            let inner_is_y = matches!(inner.as_ref(), Expr::Symbol(s) if *s == y);
            let inner_is_abs_y = matches!(
                inner.as_ref(),
                Expr::Func(FuncId::Abs, abs_args)
                    if abs_args.len() == 1
                        && matches!(abs_args[0].as_ref(), Expr::Symbol(s) if *s == y)
            );
            if inner_is_y || inner_is_abs_y {
                return Some(Expr::func(FuncId::Exp, vec![right.clone()]));
            }
        }
    }

    // Case 3: y^n = right with n y-free → y = right^(1/n)
    if let Expr::Pow(base, exp) = left.as_ref() {
        if matches!(base.as_ref(), Expr::Symbol(s) if *s == y) && !contains_symbol(exp, y) {
            let one_over_n = normalize::div(Expr::int(1), exp.clone());
            return Some(normalize::pow(right.clone(), one_over_n));
        }
    }

    // Case 4: 1/y = right → y = 1/right.
    // `1/y` compiles to MulNode { coeff = 1, factors = { y: -1 } }; match that shape.
    if let Expr::Mul(node) = left.as_ref() {
        if node.coeff.is_one() && node.factors.len() == 1 {
            let (base, exp) = node.factors.iter().next().unwrap();
            let base_is_y = matches!(base.as_ref(), Expr::Symbol(s) if *s == y);
            let exp_is_neg_one = matches!(
                exp.as_ref(),
                Expr::Integer(n) if n.to_i64() == Some(-1)
            );
            if base_is_y && exp_is_neg_one {
                return Some(normalize::div(Expr::int(1), right.clone()));
            }
        }
    }

    None
}

/// Try to solve an equation for a constant (typically `C`).
///
/// The equation is treated as `expr = 0`. Handles:
///
/// - Direct isolation: `C`, `C + rest = 0`, `C - rest = 0`, `rest - C = 0` etc.
/// - Exponential form: `exp(C) - value = 0` → `C = ln(value)`.
/// - Linear-in-C: any sum that is linear in `C`, solved via the canonical
///   `AddNode` coefficient map.
///
/// Returns `None` when the equation is non-linear in `C` or has multiple
/// `C`-bearing terms.
pub(crate) fn solve_for_constant(equation: &Arc<Expr>, c: SymbolId) -> Option<Arc<Expr>> {
    // `C = 0` — equation is literally Symbol(c).
    if matches!(equation.as_ref(), Expr::Symbol(s) if *s == c) {
        return Some(Expr::int(0));
    }

    let Expr::Add(node) = equation.as_ref() else {
        // Fallback: not in AddNode form → unsupported shape.
        // (A single `Mul`/`Pow`/`Func` containing `C` is non-linear.)
        return None;
    };

    // Locate the single `C`-bearing term.
    let mut c_term: Option<(Arc<Expr>, crate::numeric::BigRational)> = None;
    for (term, coeff) in &node.terms {
        if contains_symbol(term, c) {
            if c_term.is_some() {
                return None; // multiple C-bearing terms → give up
            }
            c_term = Some((term.clone(), coeff.clone()));
        }
    }
    let (term, coeff) = c_term?;

    // Determine C-bearing term shape.
    let is_direct = matches!(term.as_ref(), Expr::Symbol(s) if *s == c);
    let is_exp_c = matches!(
        term.as_ref(),
        Expr::Func(FuncId::Exp, args)
            if args.len() == 1
                && matches!(args[0].as_ref(), Expr::Symbol(s) if *s == c)
    );
    if !is_direct && !is_exp_c {
        return None;
    }

    // Build `rest = equation - coeff * term` by reassembling without the C term.
    let mut rest = crate::numeric::AddNode::from_constant(node.constant.clone());
    for (t, co) in &node.terms {
        if Arc::ptr_eq(t, &term) {
            continue;
        }
        rest.add_term(t.clone(), co.clone());
    }
    let rest_arc: Arc<Expr> = if rest.is_zero() {
        Expr::int(0)
    } else {
        Arc::new(Expr::Add(rest))
    };

    // solved = -rest / coeff
    let neg_rest = normalize::neg(rest_arc);
    let coeff_arc: Arc<Expr> = Arc::new(Expr::Rational(coeff));
    let solved = normalize::div(neg_rest, coeff_arc);

    if is_direct {
        Some(solved)
    } else {
        // C = ln(solved)
        Some(Expr::func(FuncId::Ln, vec![solved]))
    }
}
