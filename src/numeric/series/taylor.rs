//! Taylor series computation via repeated symbolic differentiation.
//!
//! The `n`-th coefficient is `f^(n)(center) / n!`, computed by:
//!
//! 1. Differentiating `f` symbolically `n` times using [`diff_arc`].
//! 2. Substituting the expansion `center` for the variable.
//! 3. Dividing by `n!`.
//!
//! Substitution is performed by [`crate::numeric::substitute::substitute`],
//! which rebuilds through normalizing smart constructors and folds built-in
//! unary functions applied to numeric arguments.

use std::sync::Arc;

use super::super::{
    big_rational::BigRational, differentiation::diff_arc, expr::Expr, normalize,
    small_int::SmallInt, substitute::substitute, SymbolId,
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
        let at_center = substitute(&current, var, center);
        let coeff = divide_by_factorial(at_center, n);
        coefficients.push(coeff);

        if n < order {
            current = diff_arc(&current, var);
        }
    }

    TaylorSeries::from_coefficients(center.clone(), var, coefficients)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Divide `expr` by `n!`, preserving exact rational arithmetic for all `n`.
fn divide_by_factorial(expr: Arc<Expr>, n: usize) -> Arc<Expr> {
    if n <= 1 {
        return expr;
    }
    if n <= 20 {
        let mut acc: i64 = 1;
        for k in 2..=(n as i64) {
            acc *= k;
        }
        return normalize::div(expr, Expr::int(acc));
    }
    let mut denom = SmallInt::from(1i64);
    for k in 2..=(n as u64) {
        denom = &denom * &SmallInt::from(k as i64);
    }
    let one_over_nfact = BigRational::new(SmallInt::from(1i64), denom);
    normalize::mul(expr, Arc::new(Expr::Rational(one_over_nfact)))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{normalize, Expr, FuncId, SymbolId};

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }

    // ── divide_by_factorial ────────────────────────────────────────────

    #[test]
    fn test_divide_by_factorial_identity() {
        let five = Expr::int(5);
        let result = divide_by_factorial(five.clone(), 0);
        assert!(Arc::ptr_eq(&result, &five));
        let result1 = divide_by_factorial(five.clone(), 1);
        assert!(Arc::ptr_eq(&result1, &five));
    }

    #[test]
    fn test_divide_by_factorial_small() {
        let sixty = Expr::int(120);
        let result = divide_by_factorial(sixty, 5);
        assert!(result.is_one(), "120/5! = 120/120 = 1");
    }

    #[test]
    fn test_taylor_exp_order_21_not_saturated() {
        let x_id = SymbolId::intern("tay_exp21_x");
        let x = sym("tay_exp21_x");
        let expr = Expr::func(FuncId::Exp, vec![x]);
        let ts = taylor(&expr, x_id, &Expr::int(0), 21);
        let a21 = ts.coeff(21);
        match a21.as_ref() {
            Expr::Rational(r) => {
                let v = r.to_f64();
                let expected = 1.0 / 51090942171709440000.0_f64;
                let rel_err = ((v - expected) / expected).abs();
                assert!(
                    rel_err < 1e-10,
                    "a_21 should be 1/21!, got {v:.6e}, rel_err={rel_err:.2e}"
                );
            }
            other => panic!("expected rational for a_21, got {:?}", other),
        }
    }

    // ── taylor ───────────────────────────────────────────────────────────

    #[test]
    fn test_taylor_constant_expr() {
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
        let x_id = SymbolId::intern("tay_id_x");
        let x = sym("tay_id_x");
        let ts = taylor(&x, x_id, &Expr::int(0), 2);
        assert!(ts.coeff(0).is_zero());
        assert!(ts.coeff(1).is_one());
        assert!(ts.coeff(2).is_zero());
    }

    #[test]
    fn test_taylor_exp_at_zero() {
        let x_id = SymbolId::intern("tay_exp_x");
        let x = sym("tay_exp_x");
        let expr = Expr::func(FuncId::Exp, vec![x]);
        let ts = taylor(&expr, x_id, &Expr::int(0), 4);
        assert!(ts.coeff(0).is_one(), "a_0 = 1");
        assert!(ts.coeff(1).is_one(), "a_1 = 1");
        match ts.coeff(2).as_ref() {
            Expr::Rational(r) => {
                assert_eq!(r.to_f64(), 0.5, "a_2 = 1/2");
            }
            _ => panic!("expected rational for a_2, got {}", ts.coeff(2)),
        }
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
        let x_id = SymbolId::intern("tay_sin_x");
        let x = sym("tay_sin_x");
        let expr = Expr::func(FuncId::Sin, vec![x]);
        let ts = taylor(&expr, x_id, &Expr::int(0), 5);
        assert!(ts.coeff(0).is_zero(), "sin: a_0 = 0");
        assert!(ts.coeff(1).is_one(), "sin: a_1 = 1");
        assert!(ts.coeff(2).is_zero(), "sin: a_2 = 0");
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
        let x_id = SymbolId::intern("tay_cos_x");
        let x = sym("tay_cos_x");
        let expr = Expr::func(FuncId::Cos, vec![x]);
        let ts = taylor(&expr, x_id, &Expr::int(0), 4);
        assert!(ts.coeff(0).is_one(), "cos: a_0 = 1");
        assert!(ts.coeff(1).is_zero(), "cos: a_1 = 0");
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
        let x_id = SymbolId::intern("tay_geo_x");
        let x = sym("tay_geo_x");
        let one_minus_x = normalize::sub(Expr::int(1), x);
        let expr = normalize::pow(one_minus_x, Expr::int(-1));
        let ts = taylor(&expr, x_id, &Expr::int(0), 4);
        for n in 0..=4 {
            let c = ts.coeff(n);
            assert!(c.is_one(), "1/(1-x) coeff a_{n} should be 1, got {c}");
        }
    }

    #[test]
    fn test_taylor_ln_1_plus_x_at_zero() {
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
