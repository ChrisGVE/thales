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
    differentiation::diff_arc, expr::Expr, normalize, substitute::substitute, SymbolId,
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
        let factorial_n = factorial(n);
        let coeff = normalize::div(at_center, Expr::int(factorial_n));
        coefficients.push(coeff);

        if n < order {
            current = diff_arc(&current, var);
        }
    }

    TaylorSeries::from_coefficients(center.clone(), var, coefficients)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Compute `n!` as an `i64`. Saturates at `i64::MAX` for `n > 20`.
fn factorial(n: usize) -> i64 {
    let mut acc: i64 = 1;
    for k in 2..=(n as i64) {
        acc = acc.saturating_mul(k);
    }
    acc
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{normalize, Expr, FuncId, SymbolId};

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
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
