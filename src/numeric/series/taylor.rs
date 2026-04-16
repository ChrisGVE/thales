//! Taylor series computation via repeated symbolic differentiation.
//!
//! The `n`-th coefficient is `f^(n)(center) / n!`, computed by:
//!
//! 1. Differentiating `f` symbolically `n` times using [`diff_arc`].
//! 2. Substituting the expansion `center` for the variable.
//! 3. Dividing by `n!`.
//!
//! Substitution is performed by a recursive tree walk that replaces every
//! [`Expr::Symbol`] matching `var` with `center` and rebuilds the expression
//! through the normalizing smart constructors, so constant folding happens
//! automatically.

use std::sync::Arc;

use num::traits::{One, Zero};

use super::super::{
    differentiation::diff_arc,
    expr::{Expr, FuncId},
    normalize,
    ring::Ring,
    SymbolId,
};
use super::TaylorSeries;

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute the Taylor series of `expr` around `center` up to `order`.
///
/// Returns a [`TaylorSeries`] with `order + 1` coefficients
/// `[a_0, a_1, …, a_order]` where `a_n = f^(n)(center) / n!`.
///
/// When `center` is a numeric constant, all coefficients are evaluated to
/// numeric expressions.  Symbolic centers produce symbolic coefficients.
///
/// # Arguments
///
/// * `expr`   — expression to expand.
/// * `var`    — the expansion variable.
/// * `center` — point around which to expand (e.g. `Expr::int(0)` for Maclaurin).
/// * `order`  — truncation order (number of terms = `order + 1`).
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{Expr, SymbolId, FuncId};
/// use thales::numeric::series::taylor;
///
/// let x_id = SymbolId::intern("taylor_exp");
/// let x    = Expr::symbol("taylor_exp");
/// let expr = Expr::func(FuncId::Exp, vec![x]);
/// let ts   = taylor(&expr, x_id, &Expr::int(0), 3);
/// // a_0 = 1, a_1 = 1, a_2 = 1/2, a_3 = 1/6
/// assert!(ts.coeff(0).is_one());
/// assert!(ts.coeff(1).is_one());
/// ```
pub fn taylor(expr: &Arc<Expr>, var: SymbolId, center: &Arc<Expr>, order: usize) -> TaylorSeries {
    let mut coefficients = Vec::with_capacity(order + 1);
    let mut current = expr.clone();

    for n in 0..=order {
        // a_n = current(center) / n!
        let at_center = substitute(&current, var, center);
        let factorial_n = factorial(n);
        let coeff = normalize::div(at_center, Expr::int(factorial_n));
        coefficients.push(coeff);

        // Differentiate for the next iteration (skip on last pass)
        if n < order {
            current = diff_arc(&current, var);
        }
    }

    TaylorSeries::from_coefficients(center.clone(), var, coefficients)
}

// ── Substitution ─────────────────────────────────────────────────────────────

/// Substitute every occurrence of `var` in `expr` with `value`.
///
/// Rebuilds the expression through normalizing smart constructors so that
/// constant folding and identity removal apply automatically.
pub(crate) fn substitute(expr: &Arc<Expr>, var: SymbolId, value: &Arc<Expr>) -> Arc<Expr> {
    match expr.as_ref() {
        // Numeric leaves are unchanged.
        Expr::Integer(_) | Expr::Rational(_) | Expr::Float(_) => expr.clone(),

        // Symbol: replace if it matches, otherwise keep.
        Expr::Symbol(s) => {
            if *s == var {
                value.clone()
            } else {
                expr.clone()
            }
        }

        // AddNode: substitute into each term and rebuild.
        Expr::Add(node) => {
            let constant_expr: Arc<Expr> = if node.constant.is_zero() {
                Expr::int(0)
            } else {
                Arc::new(Expr::Rational(node.constant.clone()))
            };

            let mut acc = constant_expr;
            for (term, coeff) in &node.terms {
                let new_term = substitute(term, var, value);
                let coeff_expr = rational_to_arc(coeff);
                let scaled = normalize::mul(coeff_expr, new_term);
                acc = normalize::add(acc, scaled);
            }
            acc
        }

        // MulNode: substitute into each factor and rebuild.
        Expr::Mul(node) => {
            let coeff_expr = rational_to_arc(&node.coeff);
            let mut acc = coeff_expr;
            for (base, exp) in &node.factors {
                let new_base = substitute(base, var, value);
                let new_exp = substitute(exp, var, value);
                let term = normalize::pow(new_base, new_exp);
                acc = normalize::mul(acc, term);
            }
            acc
        }

        // Pow: substitute into base and exponent.
        Expr::Pow(base, exp) => {
            let new_base = substitute(base, var, value);
            let new_exp = substitute(exp, var, value);
            normalize::pow(new_base, new_exp)
        }

        // Func: substitute into each argument.
        Expr::Func(id, args) => {
            let new_args: Vec<Arc<Expr>> = args.iter().map(|a| substitute(a, var, value)).collect();
            rebuild_func(*id, new_args)
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Compute `n!` as an `i64`. Saturates at `i64::MAX` for n > 20 (not needed
/// for series orders a CAS typically handles, but safe for compilation).
fn factorial(n: usize) -> i64 {
    let mut acc: i64 = 1;
    for k in 2..=(n as i64) {
        acc = acc.saturating_mul(k);
    }
    acc
}

/// Convert a `&BigRational` coefficient to an `Arc<Expr>`.
fn rational_to_arc(r: &super::super::BigRational) -> Arc<Expr> {
    use num::traits::{One, Zero};
    if r.denom().is_one() {
        if let Some(n) = r.numer().to_i64() {
            return Expr::int(n);
        }
    }
    Arc::new(Expr::Rational(r.clone()))
}

/// Rebuild a function application, allowing normalization of known unary
/// functions when their argument is a numeric constant.
fn rebuild_func(id: FuncId, args: Vec<Arc<Expr>>) -> Arc<Expr> {
    // For built-in single-argument functions with numeric argument, evaluate.
    if args.len() == 1 {
        if let Some(result) = try_eval_func(id, &args[0]) {
            return result;
        }
    }
    Arc::new(Expr::Func(id, args))
}

/// Attempt to evaluate a built-in unary function at a numeric constant.
///
/// Returns `None` for symbolic arguments or unsupported functions.
fn try_eval_func(id: FuncId, arg: &Arc<Expr>) -> Option<Arc<Expr>> {
    let v = numeric_f64(arg)?;
    let result = match id {
        FuncId::Sin => v.sin(),
        FuncId::Cos => v.cos(),
        FuncId::Tan => v.tan(),
        FuncId::Ln => {
            if v <= 0.0 {
                return None;
            }
            v.ln()
        }
        FuncId::Exp => v.exp(),
        FuncId::Sqrt => {
            if v < 0.0 {
                return None;
            }
            v.sqrt()
        }
        FuncId::Abs => v.abs(),
        FuncId::Other(_) => return None,
    };

    // Represent exact zeros/ones as integers; everything else as Float.
    if result == 0.0 {
        Some(Expr::int(0))
    } else if result == 1.0 {
        Some(Expr::int(1))
    } else {
        Some(Expr::float(result))
    }
}

/// Extract an `f64` from a numeric `Expr`, returning `None` for non-numeric.
fn numeric_f64(expr: &Arc<Expr>) -> Option<f64> {
    match expr.as_ref() {
        Expr::Integer(n) => n.to_i64().map(|v| v as f64),
        Expr::Rational(r) => Some(r.to_f64()),
        Expr::Float(f) => Some(*f),
        _ => None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{normalize, Expr, FuncId, SymbolId};

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }

    // ── substitute ───────────────────────────────────────────────────────

    #[test]
    fn test_substitute_var() {
        let x_id = SymbolId::intern("sub_x");
        let x = sym("sub_x");
        // x -> 3; result should be 3
        let result = substitute(&x, x_id, &Expr::int(3));
        assert_eq!(*result, Expr::Integer(crate::numeric::SmallInt::from(3i64)));
    }

    #[test]
    fn test_substitute_other_symbol_unchanged() {
        let x_id = SymbolId::intern("sub_ox");
        let y = sym("sub_oy");
        let result = substitute(&y, x_id, &Expr::int(99));
        assert_eq!(*result, *y);
    }

    #[test]
    fn test_substitute_integer_unchanged() {
        let x_id = SymbolId::intern("sub_int_x");
        let c = Expr::int(42);
        let result = substitute(&c, x_id, &Expr::int(0));
        assert_eq!(*result, *c);
    }

    #[test]
    fn test_substitute_in_pow() {
        // x^2 at x=3 → 9
        let x_id = SymbolId::intern("sub_pow_x");
        let x = sym("sub_pow_x");
        let expr = normalize::pow(x, Expr::int(2));
        let result = substitute(&expr, x_id, &Expr::int(3));
        assert_eq!(*result, Expr::Integer(crate::numeric::SmallInt::from(9i64)));
    }

    #[test]
    fn test_substitute_in_add() {
        // x + 1 at x=4 → 5
        let x_id = SymbolId::intern("sub_add_x");
        let x = sym("sub_add_x");
        let expr = normalize::add(x, Expr::int(1));
        let result = substitute(&expr, x_id, &Expr::int(4));
        assert_eq!(*result, Expr::Integer(crate::numeric::SmallInt::from(5i64)));
    }

    #[test]
    fn test_substitute_in_mul() {
        // 3*x at x=2 → 6
        let x_id = SymbolId::intern("sub_mul_x");
        let x = sym("sub_mul_x");
        let expr = normalize::mul(Expr::int(3), x);
        let result = substitute(&expr, x_id, &Expr::int(2));
        assert_eq!(*result, Expr::Integer(crate::numeric::SmallInt::from(6i64)));
    }

    #[test]
    fn test_substitute_sin_at_zero() {
        // sin(x) at x=0 → 0
        let x_id = SymbolId::intern("sub_sin_x");
        let x = sym("sub_sin_x");
        let expr = Expr::func(FuncId::Sin, vec![x]);
        let result = substitute(&expr, x_id, &Expr::int(0));
        assert!(result.is_zero());
    }

    #[test]
    fn test_substitute_cos_at_zero() {
        // cos(x) at x=0 → 1
        let x_id = SymbolId::intern("sub_cos_x");
        let x = sym("sub_cos_x");
        let expr = Expr::func(FuncId::Cos, vec![x]);
        let result = substitute(&expr, x_id, &Expr::int(0));
        assert!(result.is_one());
    }

    #[test]
    fn test_substitute_exp_at_zero() {
        // exp(x) at x=0 → 1
        let x_id = SymbolId::intern("sub_exp_x");
        let x = sym("sub_exp_x");
        let expr = Expr::func(FuncId::Exp, vec![x]);
        let result = substitute(&expr, x_id, &Expr::int(0));
        assert!(result.is_one());
    }

    // ── factorial ────────────────────────────────────────────────────────

    #[test]
    fn test_factorial_base_cases() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
    }

    #[test]
    fn test_factorial_small() {
        assert_eq!(factorial(4), 24);
        assert_eq!(factorial(6), 720);
    }

    // ── taylor ───────────────────────────────────────────────────────────

    #[test]
    fn test_taylor_constant_expr() {
        // Taylor of constant 5 at any order: a_0=5, rest=0
        let x_id = SymbolId::intern("tay_const_x");
        let expr = Expr::int(5);
        let ts = taylor(&expr, x_id, &Expr::int(0), 3);
        assert_eq!(ts.order, 3);
        assert_eq!(
            *ts.coeff(0),
            Expr::Integer(crate::numeric::SmallInt::from(5i64))
        );
        assert!(ts.coeff(1).is_zero());
        assert!(ts.coeff(2).is_zero());
    }

    #[test]
    fn test_taylor_identity() {
        // Taylor of x at 0, order 2: a_0=0, a_1=1, a_2=0
        let x_id = SymbolId::intern("tay_id_x");
        let x = sym("tay_id_x");
        let ts = taylor(&x, x_id, &Expr::int(0), 2);
        assert!(ts.coeff(0).is_zero());
        assert!(ts.coeff(1).is_one());
        assert!(ts.coeff(2).is_zero());
    }

    #[test]
    fn test_taylor_exp_at_zero() {
        // exp(x) at 0: a_n = 1/n!
        let x_id = SymbolId::intern("tay_exp_x");
        let x = sym("tay_exp_x");
        let expr = Expr::func(FuncId::Exp, vec![x]);
        let ts = taylor(&expr, x_id, &Expr::int(0), 4);
        assert!(ts.coeff(0).is_one(), "a_0 = 1");
        assert!(ts.coeff(1).is_one(), "a_1 = 1");
        // a_2 = 1/2
        match ts.coeff(2).as_ref() {
            Expr::Rational(r) => {
                assert_eq!(r.to_f64(), 0.5, "a_2 = 1/2");
            }
            _ => panic!("expected rational for a_2, got {}", ts.coeff(2)),
        }
        // a_3 = 1/6
        match ts.coeff(3).as_ref() {
            Expr::Rational(r) => {
                let v = r.to_f64();
                assert!((v - 1.0 / 6.0).abs() < 1e-12, "a_3 = 1/6");
            }
            _ => panic!("expected rational for a_3"),
        }
    }

    #[test]
    fn test_taylor_sin_at_zero() {
        // sin(x) at 0: a_0=0, a_1=1, a_2=0, a_3=-1/6
        let x_id = SymbolId::intern("tay_sin_x");
        let x = sym("tay_sin_x");
        let expr = Expr::func(FuncId::Sin, vec![x]);
        let ts = taylor(&expr, x_id, &Expr::int(0), 5);
        assert!(ts.coeff(0).is_zero(), "sin: a_0 = 0");
        assert!(ts.coeff(1).is_one(), "sin: a_1 = 1");
        assert!(ts.coeff(2).is_zero(), "sin: a_2 = 0");
        // a_3 = -1/6
        let a3 = ts.coeff(3);
        match a3.as_ref() {
            Expr::Rational(r) => {
                let v = r.to_f64();
                assert!((v + 1.0 / 6.0).abs() < 1e-12, "a_3 = -1/6, got {v}");
            }
            _ => panic!("expected rational for sin a_3, got {a3}"),
        }
    }

    #[test]
    fn test_taylor_cos_at_zero() {
        // cos(x) at 0: a_0=1, a_1=0, a_2=-1/2
        let x_id = SymbolId::intern("tay_cos_x");
        let x = sym("tay_cos_x");
        let expr = Expr::func(FuncId::Cos, vec![x]);
        let ts = taylor(&expr, x_id, &Expr::int(0), 4);
        assert!(ts.coeff(0).is_one(), "cos: a_0 = 1");
        assert!(ts.coeff(1).is_zero(), "cos: a_1 = 0");
        // a_2 = -1/2
        let a2 = ts.coeff(2);
        match a2.as_ref() {
            Expr::Rational(r) => {
                let v = r.to_f64();
                assert!((v + 0.5).abs() < 1e-12, "a_2 = -1/2, got {v}");
            }
            _ => panic!("expected rational for cos a_2, got {a2}"),
        }
    }

    #[test]
    fn test_taylor_geometric_at_zero() {
        // 1/(1-x) = (1-x)^(-1); but we build it as Pow(1-x, -1)
        // a_n should all be 1 (geometric series)
        let x_id = SymbolId::intern("tay_geo_x");
        let x = sym("tay_geo_x");
        let one_minus_x = normalize::sub(Expr::int(1), x);
        let expr = normalize::pow(one_minus_x, Expr::int(-1));
        let ts = taylor(&expr, x_id, &Expr::int(0), 4);
        // All coefficients should be 1
        for n in 0..=4 {
            let c = ts.coeff(n);
            assert!(c.is_one(), "1/(1-x) coeff a_{n} should be 1, got {c}");
        }
    }

    #[test]
    fn test_taylor_ln_1_plus_x_at_zero() {
        // ln(1+x) at 0: a_0=0, a_1=1, a_2=-1/2, a_3=1/3
        let x_id = SymbolId::intern("tay_ln_x");
        let x = sym("tay_ln_x");
        let one_plus_x = normalize::add(Expr::int(1), x);
        let expr = Expr::func(FuncId::Ln, vec![one_plus_x]);
        let ts = taylor(&expr, x_id, &Expr::int(0), 3);
        assert!(ts.coeff(0).is_zero(), "ln(1+x): a_0=0");
        assert!(ts.coeff(1).is_one(), "ln(1+x): a_1=1");
        let a2 = ts.coeff(2);
        match a2.as_ref() {
            Expr::Rational(r) => {
                let v = r.to_f64();
                assert!((v + 0.5).abs() < 1e-12, "a_2 = -1/2, got {v}");
            }
            _ => panic!("expected rational for ln a_2, got {a2}"),
        }
        let a3 = ts.coeff(3);
        match a3.as_ref() {
            Expr::Rational(r) => {
                let v = r.to_f64();
                assert!((v - 1.0 / 3.0).abs() < 1e-12, "a_3 = 1/3, got {v}");
            }
            _ => panic!("expected rational for ln a_3, got {a3}"),
        }
    }
}
