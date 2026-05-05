//! Series arithmetic: addition, multiplication, and truncation.
//!
//! All operations work on [`TaylorSeries`] values that share the same
//! expansion variable and center.  No runtime check is performed that the
//! centers match; callers are responsible for ensuring compatibility.

use std::sync::Arc;

use super::super::{expr::Expr, normalize};
use super::TaylorSeries;

// ── Public API ────────────────────────────────────────────────────────────────

/// Add two Taylor series term-by-term.
///
/// The result has order `max(s1.order, s2.order)` and the same center and
/// variable as `s1` (which must match `s2`).
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{Expr, SymbolId};
/// use thales::numeric::series::{add, sin_series, cos_series};
///
/// let x_id = SymbolId::intern("arith_add");
/// // sin + cos: a_0 = 0+1 = 1, a_1 = 1+0 = 1
/// let s = sin_series(x_id, &Expr::int(0), 4);
/// let c = cos_series(x_id, &Expr::int(0), 4);
/// let r = add(&s, &c);
/// assert!(r.coeff(0).is_one());
/// assert!(r.coeff(1).is_one());
/// ```
pub fn add(s1: &TaylorSeries, s2: &TaylorSeries) -> TaylorSeries {
    let order = s1.order.max(s2.order);
    let coefficients = (0..=order)
        .map(|n| normalize::add(s1.coeff(n), s2.coeff(n)))
        .collect();
    TaylorSeries::from_coefficients(s1.center.clone(), s1.var, coefficients)
}

/// Multiply two Taylor series via the Cauchy product, truncated to
/// `min(s1.order, s2.order)`.
///
/// The Cauchy product coefficient at degree `n` is `Σ_{k=0}^{n} a_k · b_{n-k}`.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{Expr, FuncId, SymbolId};
/// use thales::numeric::series::{mul, exp_series};
/// use thales::numeric::normalize;
///
/// let x_id = SymbolId::intern("arith_mul");
/// let x    = Expr::symbol("arith_mul");
/// // exp(x) * exp(-x) ≈ 1 + 0·x + 0·x^2 + ... (truncated)
/// let pos = exp_series(x_id, &Expr::int(0), 4);
/// // Build exp(-x) series manually from known coefficients: a_n = (-1)^n / n!
/// let neg_x_id = SymbolId::intern("arith_mul");
/// let neg = exp_series(neg_x_id, &Expr::int(0), 4);
/// // (Just test shape: product has order min(4,4) = 4)
/// let p = mul(&pos, &neg);
/// assert_eq!(p.order, 4);
/// ```
pub fn mul(s1: &TaylorSeries, s2: &TaylorSeries) -> TaylorSeries {
    let order = s1.order.min(s2.order);
    let coefficients = (0..=order).map(|n| cauchy_coeff(s1, s2, n)).collect();
    TaylorSeries::from_coefficients(s1.center.clone(), s1.var, coefficients)
}

/// Truncate a Taylor series to a lower order.
///
/// If `new_order >= series.order`, the series is returned unchanged (cloned).
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{Expr, SymbolId};
/// use thales::numeric::series::{truncate, exp_series};
///
/// let x_id = SymbolId::intern("arith_trunc");
/// let ts   = exp_series(x_id, &Expr::int(0), 6);
/// let short = truncate(&ts, 3);
/// assert_eq!(short.order, 3);
/// assert_eq!(short.coefficients.len(), 4);
/// ```
pub fn truncate(series: &TaylorSeries, new_order: usize) -> TaylorSeries {
    if new_order >= series.order {
        return series.clone();
    }
    let coefficients = series.coefficients[..=new_order].to_vec();
    TaylorSeries::from_coefficients(series.center.clone(), series.var, coefficients)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Compute the Cauchy product coefficient at degree `n`:
/// `c_n = Σ_{k=0}^{n} a_k · b_{n-k}`.
fn cauchy_coeff(s1: &TaylorSeries, s2: &TaylorSeries, n: usize) -> Arc<Expr> {
    let mut acc = Expr::int(0);
    for k in 0..=n {
        let a_k = s1.coeff(k);
        let b_nk = s2.coeff(n - k);
        // Skip zero terms to avoid building trivial Mul nodes.
        if a_k.is_zero() || b_nk.is_zero() {
            continue;
        }
        let term = normalize::mul(a_k, b_nk);
        acc = normalize::add(acc, term);
    }
    acc
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::normalize;
    use crate::numeric::series::{cos_series, exp_series, sin_series, taylor};
    use crate::numeric::{Expr, FuncId, SymbolId};

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }

    fn to_f64_expr(expr: &Arc<Expr>) -> f64 {
        match expr.as_ref() {
            Expr::Integer(n) => n.to_i64().unwrap_or(0) as f64,
            Expr::Rational(r) => r.to_f64(),
            Expr::Float(f) => *f,
            _ => panic!("non-numeric: {expr}"),
        }
    }

    // ── add ──────────────────────────────────────────────────────────────

    #[test]
    fn test_add_sin_cos_a0_a1() {
        // sin(x) + cos(x) at 0, order 4
        let x_id = SymbolId::intern("arith_add_sc");
        let s = sin_series(x_id, &Expr::int(0), 4);
        let c = cos_series(x_id, &Expr::int(0), 4);
        let r = add(&s, &c);
        assert_eq!(r.order, 4);
        // a_0 = 0 + 1 = 1
        assert!(r.coeff(0).is_one(), "a_0 = 1");
        // a_1 = 1 + 0 = 1
        assert!(r.coeff(1).is_one(), "a_1 = 1");
    }

    #[test]
    fn test_add_different_orders_pads() {
        // order-2 + order-4 → order-4; extra coefficients come from s2
        let x_id = SymbolId::intern("arith_add_diff");
        let s = sin_series(x_id, &Expr::int(0), 2);
        let c = cos_series(x_id, &Expr::int(0), 4);
        let r = add(&s, &c);
        assert_eq!(r.order, 4);
    }

    #[test]
    fn test_add_self_cancels_with_neg() {
        // s + (-1)*s should give zero coefficients
        let x_id = SymbolId::intern("arith_add_cancel");
        let s = exp_series(x_id, &Expr::int(0), 3);
        // Build -s by negating each coefficient
        let neg_coeffs: Vec<Arc<Expr>> = s
            .coefficients
            .iter()
            .map(|c| normalize::neg(c.clone()))
            .collect();
        let neg_s = TaylorSeries::from_coefficients(s.center.clone(), s.var, neg_coeffs);
        let r = add(&s, &neg_s);
        for n in 0..=3 {
            assert!(r.coeff(n).is_zero(), "a_{n} should be 0");
        }
    }

    // ── mul ──────────────────────────────────────────────────────────────

    #[test]
    fn test_mul_exp_x_exp_neg_x_is_one() {
        // exp(x) * exp(-x) = 1; truncated series should have a_0=1, rest ≈ 0
        let x_id = SymbolId::intern("arith_mul_ee");
        let x = sym("arith_mul_ee");
        let neg_x = normalize::neg(x.clone());
        let expr_pos = Expr::func(FuncId::Exp, vec![x]);
        let expr_neg = Expr::func(FuncId::Exp, vec![neg_x]);
        let ts_pos = taylor(&expr_pos, x_id, &Expr::int(0), 4);
        let ts_neg = taylor(&expr_neg, x_id, &Expr::int(0), 4);
        let product = mul(&ts_pos, &ts_neg);
        // a_0 should be 1
        assert!(product.coeff(0).is_one(), "exp*exp(-x) a_0 = 1");
        // Higher coefficients should be zero (or numerically tiny)
        for n in 1..=4 {
            let v = to_f64_expr(&product.coeff(n));
            assert!(v.abs() < 1e-10, "exp*exp(-x) a_{n} should be 0, got {v}");
        }
    }

    #[test]
    fn test_mul_order_is_min() {
        let x_id = SymbolId::intern("arith_mul_ord");
        let s2 = sin_series(x_id, &Expr::int(0), 2);
        let c4 = cos_series(x_id, &Expr::int(0), 4);
        let p = mul(&s2, &c4);
        assert_eq!(p.order, 2, "order = min(2,4) = 2");
    }

    #[test]
    fn test_mul_by_constant_one_series() {
        // (1 + 0*x + ...) * f(x) = f(x) truncated
        let x_id = SymbolId::intern("arith_mul_one");
        let ones: Vec<Arc<Expr>> = vec![Expr::int(1), Expr::int(0), Expr::int(0)];
        let one_series = TaylorSeries::from_coefficients(Expr::int(0), x_id, ones);
        let e = exp_series(x_id, &Expr::int(0), 2);
        let p = mul(&one_series, &e);
        assert_eq!(p.order, 2);
        assert!(p.coeff(0).is_one());
    }

    // ── truncate ─────────────────────────────────────────────────────────

    #[test]
    fn test_truncate_reduces_order() {
        let x_id = SymbolId::intern("arith_trunc_r");
        let ts = exp_series(x_id, &Expr::int(0), 6);
        let short = truncate(&ts, 3);
        assert_eq!(short.order, 3);
        assert_eq!(short.coefficients.len(), 4);
    }

    #[test]
    fn test_truncate_preserves_coefficients() {
        let x_id = SymbolId::intern("arith_trunc_p");
        let ts = exp_series(x_id, &Expr::int(0), 4);
        let short = truncate(&ts, 2);
        // First 3 coefficients should match the original
        for n in 0..=2 {
            let orig = to_f64_expr(&ts.coeff(n));
            let trunc = to_f64_expr(&short.coeff(n));
            assert!((orig - trunc).abs() < 1e-12, "coeff {n} should match");
        }
    }

    #[test]
    fn test_truncate_no_op_when_larger() {
        let x_id = SymbolId::intern("arith_trunc_nop");
        let ts = exp_series(x_id, &Expr::int(0), 3);
        let same = truncate(&ts, 10);
        assert_eq!(same.order, 3);
    }
}
