//! Expression-AST walkers supporting the first-order ODE solvers.
//!
//! These helpers inspect or manipulate the legacy `Expression` tree at the
//! migration boundary — they are called after `FirstOrderODE::rhs_expr()` has
//! decompiled the canonical `Arc<Expr>` storage back to `Expression`. Porting
//! them to `Arc<Expr>` walkers is tracked as follow-up work under milestone
//! `Expr-migration` (task 10.4).

use std::sync::Arc;

use crate::ast::{BinaryOp, Expression, Function, UnaryOp};
use crate::numeric::compile::{compile, decompile};
use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::normalize;
use crate::numeric::substitute::substitute as arc_substitute;
use crate::numeric::{MulNode, SymbolId};
use crate::solver::helpers::contains_symbol;
use num::traits::One;

/// Attempt to separate `dy/dx = f(x, y)` into `g(x) · h(y)`.
///
/// Returns `(g(x), h(y))` if separable, `None` otherwise. Delegates to the
/// `Arc<Expr>`-native worker [`try_separate_expr`] which iterates the
/// canonical `MulNode` factors, so it naturally handles division
/// (`g(x) / h(y)` stores as `Mul { g(x), Pow(h(y), -1) }`) and arbitrary
/// factor counts.
pub(crate) fn try_separate(
    expr: &Expression,
    x_var: &str,
    y_var: &str,
) -> Option<(Expression, Expression)> {
    let expr_arc = compile(expr);
    let x_id = SymbolId::intern(x_var);
    let y_id = SymbolId::intern(y_var);
    let (g, h) = try_separate_expr(&expr_arc, x_id, y_id)?;
    Some((decompile(&g), decompile(&h)))
}

/// Arc<Expr>-native worker for [`try_separate`].
///
/// Pure cases (one variable absent) return immediately. Otherwise the
/// expression must be a canonical product; each factor is assigned to
/// `g(x)` (x-only or constant) or `h(y)` (y-only). Any factor that
/// mentions both variables makes the ODE non-separable.
fn try_separate_expr(rhs: &Arc<Expr>, x: SymbolId, y: SymbolId) -> Option<(Arc<Expr>, Arc<Expr>)> {
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

/// Extract P(x) and Q(x) from a linear ODE in form dy/dx = -P(x)*y + Q(x).
///
/// Returns (P(x), Q(x)) if linear, None otherwise.
pub(crate) fn extract_linear_coefficients(
    rhs: &Expression,
    _x_var: &str,
    y_var: &str,
) -> Option<(Expression, Expression)> {
    // The RHS should be of form: terms with y (linear in y) + terms without y
    // dy/dx = a*y + b  where a might depend on x, b might depend on x
    // Standard form: dy/dx + P*y = Q means rhs = -P*y + Q

    // Collect terms with y and without y
    let mut y_coefficient = Expression::Integer(0);
    let mut constant_terms = Expression::Integer(0);

    fn collect_terms(
        expr: &Expression,
        y_var: &str,
        y_coeff: &mut Expression,
        const_terms: &mut Expression,
    ) -> bool {
        match expr {
            // Simple y term
            Expression::Variable(v) if v.name == y_var => {
                *y_coeff = Expression::Binary(
                    BinaryOp::Add,
                    Box::new(y_coeff.clone()),
                    Box::new(Expression::Integer(1)),
                );
                true
            }
            // Sum: recurse into both sides
            Expression::Binary(BinaryOp::Add, left, right) => {
                collect_terms(left, y_var, y_coeff, const_terms)
                    && collect_terms(right, y_var, y_coeff, const_terms)
            }
            // Difference: handle subtraction
            Expression::Binary(BinaryOp::Sub, left, right) => {
                let mut neg_y_coeff = Expression::Integer(0);
                let mut neg_const = Expression::Integer(0);
                if !collect_terms(left, y_var, y_coeff, const_terms) {
                    return false;
                }
                if !collect_terms(right, y_var, &mut neg_y_coeff, &mut neg_const) {
                    return false;
                }
                // Subtract the right side contributions
                *y_coeff = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(y_coeff.clone()),
                    Box::new(neg_y_coeff),
                );
                *const_terms = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(const_terms.clone()),
                    Box::new(neg_const),
                );
                true
            }
            // Product: check if linear in y
            Expression::Binary(BinaryOp::Mul, left, right) => {
                let left_has_y = left.contains_variable(y_var);
                let right_has_y = right.contains_variable(y_var);

                if left_has_y && right_has_y {
                    // y^2 or similar - not linear
                    return false;
                }
                if !left_has_y && !right_has_y {
                    // No y - this is a constant term
                    *const_terms = Expression::Binary(
                        BinaryOp::Add,
                        Box::new(const_terms.clone()),
                        Box::new(expr.clone()),
                    );
                    return true;
                }
                // One factor is y (or contains y linearly), other is coefficient
                if left_has_y {
                    // left is y or y-term, right is coefficient
                    if matches!(left.as_ref(), Expression::Variable(v) if v.name == y_var) {
                        *y_coeff = Expression::Binary(
                            BinaryOp::Add,
                            Box::new(y_coeff.clone()),
                            right.clone(),
                        );
                        return true;
                    }
                } else {
                    // right is y or y-term, left is coefficient
                    if matches!(right.as_ref(), Expression::Variable(v) if v.name == y_var) {
                        *y_coeff = Expression::Binary(
                            BinaryOp::Add,
                            Box::new(y_coeff.clone()),
                            left.clone(),
                        );
                        return true;
                    }
                }
                false
            }
            // Negation
            Expression::Unary(UnaryOp::Neg, inner) => {
                let mut neg_y_coeff = Expression::Integer(0);
                let mut neg_const = Expression::Integer(0);
                if !collect_terms(inner, y_var, &mut neg_y_coeff, &mut neg_const) {
                    return false;
                }
                *y_coeff = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(y_coeff.clone()),
                    Box::new(neg_y_coeff),
                );
                *const_terms = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(const_terms.clone()),
                    Box::new(neg_const),
                );
                true
            }
            // Any expression without y is a constant term
            _ if !expr.contains_variable(y_var) => {
                *const_terms = Expression::Binary(
                    BinaryOp::Add,
                    Box::new(const_terms.clone()),
                    Box::new(expr.clone()),
                );
                true
            }
            // Anything else with y that doesn't fit above is not linear
            _ => false,
        }
    }

    if !collect_terms(rhs, y_var, &mut y_coefficient, &mut constant_terms) {
        return None;
    }

    // Simplify collected terms
    let y_coeff = y_coefficient.simplify();
    let q_x = constant_terms.simplify();

    // P(x) is the negative of the y coefficient (since rhs = -P*y + Q)
    let p_x = Expression::Unary(UnaryOp::Neg, Box::new(y_coeff)).simplify();

    // Check that P(x) doesn't contain y
    if p_x.contains_variable(y_var) {
        return None;
    }

    Some((p_x, q_x))
}

/// Substitute a variable with an expression.
///
/// Delegates to the `Arc<Expr>` walker [`crate::numeric::substitute::substitute`]
/// via compile/decompile at the boundary. Signature is preserved so callers
/// keep operating on `Expression` until they migrate.
pub(crate) fn substitute_var(expr: &Expression, var: &str, replacement: &Expression) -> Expression {
    let expr_arc = compile(expr);
    let replacement_arc = compile(replacement);
    let var_id = SymbolId::intern(var);
    let result = arc_substitute(&expr_arc, var_id, &replacement_arc);
    decompile(&result)
}

/// Try to solve an implicit relation for y explicitly.
///
/// Delegates to the `Arc<Expr>`-native helper [`try_solve_implicit_for_y_expr`]
/// via compile/decompile at the boundary. Handles:
///
/// - `y = right` → `right`
/// - `ln(y) = right` or `ln(|y|) = right` → `exp(right)` (positive branch)
/// - `y^n = right` with n y-free → `right^(1/n)`
/// - `1/y = right` → `1/right`
pub(crate) fn try_solve_implicit_for_y(
    left: &Expression,
    right: &Expression,
    y_var: &str,
) -> Option<Expression> {
    let left_arc = compile(left);
    let right_arc = compile(right);
    let y_id = SymbolId::intern(y_var);
    let result = try_solve_implicit_for_y_expr(&left_arc, &right_arc, y_id)?;
    Some(decompile(&result))
}

/// Arc<Expr>-native worker for [`try_solve_implicit_for_y`].
///
/// Pattern-matches on the canonical [`Expr`] form. Callers should compile
/// their `Expression` inputs first; results are normalized through the
/// smart constructors in [`crate::numeric::normalize`].
fn try_solve_implicit_for_y_expr(
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
/// - Exponential form: `exp(C) - value = 0` → `C = ln(value)` (and symmetric).
/// - Linear-in-C: any sum that is linear in `C`, solved via the canonical
///   `AddNode` coefficient map.
///
/// Delegates to the `Arc<Expr>`-native worker [`solve_for_constant_expr`].
pub(crate) fn solve_for_constant(equation: &Expression, const_name: &str) -> Option<Expression> {
    let eq_arc = compile(equation);
    let c_id = SymbolId::intern(const_name);
    let result = solve_for_constant_expr(&eq_arc, c_id)?;
    Some(decompile(&result))
}

/// Arc<Expr>-native worker for [`solve_for_constant`].
///
/// Handles `C = 0`, direct isolation via `AddNode`, and the `exp(C) - value`
/// shorthand. Returns `None` when the equation is non-linear in `C` or has
/// multiple `C`-bearing terms.
fn solve_for_constant_expr(equation: &Arc<Expr>, c: SymbolId) -> Option<Arc<Expr>> {
    // `C = 0` — equation is literally Symbol(c).
    if matches!(equation.as_ref(), Expr::Symbol(s) if *s == c) {
        return Some(Expr::int(0));
    }

    // `exp(C) = 0` has no real solution; skip.
    // `exp(C)` alone without additive residual cannot be solved either.

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
