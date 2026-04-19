//! Series composition and Lagrange reversion on `TaylorSeries`.
//!
//! - [`compose`] computes `outer ∘ inner` via repeated Cauchy-product
//!   convolution (precomputing `inner^k` and accumulating `a_k · inner^k`).
//! - [`revert`] computes the compositional inverse of a series with `a_0 = 0`
//!   and `a_1 ≠ 0`, using the Lagrange-style recurrence derived from
//!   `S(T(y)) = y`.
//!
//! Both engines emit [`TechniqueTag`] traces when a `&mut Trace` is supplied.
//! Coefficient arithmetic stays in `Arc<Expr>` throughout — the output
//! coefficients are whatever `normalize::{add, mul, div, neg}` produce from
//! the inputs, so exact rational or symbolic series round-trip cleanly.

use std::sync::Arc;

use super::super::{
    expr::Expr,
    normalize,
    trace::{record, Step, TechniqueTag, Trace},
};
use super::TaylorSeries;

// ── Public API ───────────────────────────────────────────────────────────────

/// Compose `outer(inner(x))`.
///
/// Requires:
/// - outer and inner share the same variable and center,
/// - `inner.coefficients[0]` is numerically zero.
///
/// The result is truncated to `min(outer.order, inner.order)`.
pub fn compose(
    outer: &TaylorSeries,
    inner: &TaylorSeries,
    mut trace: Option<&mut Trace>,
) -> Option<TaylorSeries> {
    if outer.var != inner.var {
        return None;
    }
    if outer.center != inner.center {
        return None;
    }
    if !inner.coefficients.first()?.is_zero() {
        return None;
    }

    let order = outer.order.min(inner.order);

    record(
        trace.as_deref_mut(),
        Step::new(TechniqueTag::SeriesComposition, format!("order {order}")),
    );

    // Pad inner coefficients to `order + 1`.
    let inner_coeffs = pad_coeffs(&inner.coefficients, order);

    // Accumulator `result[m]` is the coefficient of x^m after summing
    // `a_k · inner^k` for k = 0, 1, ..., order.
    let mut result: Vec<Arc<Expr>> = zeros(order + 1);

    // k = 0: `inner^0 = 1`, contributing `outer[0]` to x^0.
    if let Some(c0) = outer.coefficients.first() {
        result[0] = c0.clone();
    }

    // Powers of `inner`, truncated to `order` each step.
    let mut inner_power: Vec<Arc<Expr>> = {
        let mut base = zeros(order + 1);
        base[0] = Expr::int(1);
        base
    };

    for k in 1..=order {
        inner_power = convolve(&inner_power, &inner_coeffs, order);
        if k >= outer.coefficients.len() {
            continue;
        }
        let a_k = outer.coefficients[k].clone();
        if a_k.is_zero() {
            continue;
        }
        for m in 0..=order {
            if inner_power[m].is_zero() {
                continue;
            }
            let contrib = normalize::mul(a_k.clone(), inner_power[m].clone());
            result[m] = normalize::add(result[m].clone(), contrib);
        }
    }

    Some(TaylorSeries::from_coefficients(
        outer.center.clone(),
        outer.var,
        result,
    ))
}

/// Compositional inverse of `series` (Lagrange reversion).
///
/// Given `S(x) = Σ a_k x^k` with `a_0 = 0` and `a_1 ≠ 0`, produce `T(y)` so
/// that `S(T(y)) = y` up to the requested truncation order.
///
/// Returns `None` when `a_0 ≠ 0` or `a_1` is numerically zero.
pub fn revert(series: &TaylorSeries, mut trace: Option<&mut Trace>) -> Option<TaylorSeries> {
    let a = &series.coefficients;
    if a.is_empty() {
        return None;
    }
    if !a[0].is_zero() {
        return None;
    }
    if a.len() < 2 || a[1].is_zero() {
        return None;
    }

    let order = series.order;

    record(
        trace.as_deref_mut(),
        Step::new(TechniqueTag::LagrangeReversion, format!("order {order}")),
    );

    let inv_a1 = normalize::div(Expr::int(1), a[1].clone());

    // b[j] holds coefficient of y^j in the inverse. b[0] = 0.
    let mut b: Vec<Arc<Expr>> = zeros(order + 1);
    if order >= 1 {
        b[1] = inv_a1.clone();
    }

    for n in 2..=order {
        // Sum_{k=2}^n a_k · [y^n] T(y)^k, where T uses b[1..n] (b[n] not yet set).
        let mut sum: Arc<Expr> = Expr::int(0);
        for k in 2..=n {
            if k >= a.len() || a[k].is_zero() {
                continue;
            }
            let p_k_n = power_coeff(&b, k, n, order);
            if p_k_n.is_zero() {
                continue;
            }
            let term = normalize::mul(a[k].clone(), p_k_n);
            sum = normalize::add(sum, term);
        }
        // b_n = -(sum) / a_1
        let neg_sum = normalize::neg(sum);
        b[n] = normalize::mul(inv_a1.clone(), neg_sum);
    }

    Some(TaylorSeries::from_coefficients(
        series.center.clone(),
        series.var,
        b,
    ))
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Coefficient of `y^n` in `(Σ_{j=1}^{order} b_j y^j)^k`.
///
/// Recursive convolution bounded by `max_order` so intermediate expansions
/// never overshoot the truncation. Relies on `b[0] = 0`.
fn power_coeff(b: &[Arc<Expr>], k: usize, n: usize, max_order: usize) -> Arc<Expr> {
    if n > max_order {
        return Expr::int(0);
    }
    match k {
        0 => {
            if n == 0 {
                Expr::int(1)
            } else {
                Expr::int(0)
            }
        }
        1 => b.get(n).cloned().unwrap_or_else(|| Expr::int(0)),
        _ => {
            // [y^n] T^k = sum_{j=1}^{n} b[j] * [y^(n-j)] T^(k-1)
            let mut sum: Arc<Expr> = Expr::int(0);
            for j in 1..=n {
                if j >= b.len() || b[j].is_zero() {
                    continue;
                }
                let inner = power_coeff(b, k - 1, n - j, max_order);
                if inner.is_zero() {
                    continue;
                }
                let term = normalize::mul(b[j].clone(), inner);
                sum = normalize::add(sum, term);
            }
            sum
        }
    }
}

/// Cauchy-product convolution truncated to `order`.
fn convolve(a: &[Arc<Expr>], b: &[Arc<Expr>], order: usize) -> Vec<Arc<Expr>> {
    let mut out: Vec<Arc<Expr>> = zeros(order + 1);
    let a_last = a.len().saturating_sub(1).min(order);
    let b_last = b.len().saturating_sub(1).min(order);
    for i in 0..=a_last {
        if a[i].is_zero() {
            continue;
        }
        let max_j = (order - i).min(b_last);
        for j in 0..=max_j {
            if b[j].is_zero() {
                continue;
            }
            let prod = normalize::mul(a[i].clone(), b[j].clone());
            out[i + j] = normalize::add(out[i + j].clone(), prod);
        }
    }
    out
}

fn zeros(n: usize) -> Vec<Arc<Expr>> {
    (0..n).map(|_| Expr::int(0)).collect()
}

fn pad_coeffs(src: &[Arc<Expr>], order: usize) -> Vec<Arc<Expr>> {
    let mut out = zeros(order + 1);
    for (i, c) in src.iter().take(order + 1).enumerate() {
        out[i] = c.clone();
    }
    out
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{BigRational, SmallInt, SymbolId};

    fn taylor_from(raw: &[i64]) -> TaylorSeries {
        let var = SymbolId::intern("comp_x");
        let center = Expr::int(0);
        let coeffs: Vec<Arc<Expr>> = raw.iter().map(|&c| Expr::int(c)).collect();
        TaylorSeries::from_coefficients(center, var, coeffs)
    }

    fn rational_coeff(ts: &TaylorSeries, n: usize) -> Option<BigRational> {
        match ts.coeff(n).as_ref() {
            Expr::Integer(i) => Some(BigRational::from_integer(i.clone())),
            Expr::Rational(r) => Some(r.clone()),
            _ => None,
        }
    }

    fn to_f64(expr: &Arc<Expr>) -> Option<f64> {
        match expr.as_ref() {
            Expr::Integer(n) => n.to_i64().map(|v| v as f64),
            Expr::Rational(r) => Some(r.to_f64()),
            Expr::Float(f) => Some(*f),
            _ => None,
        }
    }

    // ── compose ──

    #[test]
    fn compose_identity_is_inner() {
        // outer(y) = y  →  outer ∘ inner = inner.
        let outer = taylor_from(&[0, 1, 0, 0]);
        let inner = taylor_from(&[0, 2, 3, 5]);
        let got = compose(&outer, &inner, None).expect("compose");
        for n in 0..=3 {
            assert_eq!(
                to_f64(&got.coeff(n)).unwrap() as i64,
                inner.coeff(n).as_ref().clone().to_i64_expect(),
                "coeff {n}"
            );
        }
    }

    #[test]
    fn compose_square_of_x() {
        // outer(y) = y^2, inner(x) = x → composition = x^2.
        let outer = taylor_from(&[0, 0, 1, 0]);
        let inner = taylor_from(&[0, 1, 0, 0]);
        let got = compose(&outer, &inner, None).expect("compose");
        assert!(got.coeff(0).is_zero());
        assert!(got.coeff(1).is_zero());
        assert!(got.coeff(2).is_one());
        assert!(got.coeff(3).is_zero());
    }

    #[test]
    fn compose_rejects_nonzero_inner_constant() {
        let outer = taylor_from(&[0, 1, 0, 0]);
        let inner = taylor_from(&[1, 1, 0, 0]); // constant term = 1
        assert!(compose(&outer, &inner, None).is_none());
    }

    #[test]
    fn compose_rejects_mismatched_var() {
        let o_center = Expr::int(0);
        let i_center = Expr::int(0);
        let outer = TaylorSeries::from_coefficients(
            o_center,
            SymbolId::intern("comp_mvar_a"),
            vec![Expr::int(0), Expr::int(1)],
        );
        let inner = TaylorSeries::from_coefficients(
            i_center,
            SymbolId::intern("comp_mvar_b"),
            vec![Expr::int(0), Expr::int(1)],
        );
        assert!(compose(&outer, &inner, None).is_none());
    }

    #[test]
    fn compose_linear_polynomial() {
        // outer = 1 + 2y + 3y^2, inner = x (a_0=0, a_1=1).
        // result = 1 + 2x + 3x^2.
        let outer = taylor_from(&[1, 2, 3, 0]);
        let inner = taylor_from(&[0, 1, 0, 0]);
        let got = compose(&outer, &inner, None).expect("compose");
        assert_eq!(to_f64(&got.coeff(0)).unwrap() as i64, 1);
        assert_eq!(to_f64(&got.coeff(1)).unwrap() as i64, 2);
        assert_eq!(to_f64(&got.coeff(2)).unwrap() as i64, 3);
    }

    #[test]
    fn compose_with_scaled_inner() {
        // outer = y + y^2, inner = 2x → result = 2x + 4x^2.
        let outer = taylor_from(&[0, 1, 1, 0]);
        let inner = taylor_from(&[0, 2, 0, 0]);
        let got = compose(&outer, &inner, None).expect("compose");
        assert_eq!(to_f64(&got.coeff(0)).unwrap() as i64, 0);
        assert_eq!(to_f64(&got.coeff(1)).unwrap() as i64, 2);
        assert_eq!(to_f64(&got.coeff(2)).unwrap() as i64, 4);
    }

    // ── revert ──

    #[test]
    fn revert_rejects_nonzero_constant() {
        // a_0 = 1 → no inverse.
        let series = taylor_from(&[1, 1, 0, 0]);
        assert!(revert(&series, None).is_none());
    }

    #[test]
    fn revert_rejects_zero_linear() {
        // a_0 = 0, a_1 = 0 → no inverse.
        let series = taylor_from(&[0, 0, 1, 0]);
        assert!(revert(&series, None).is_none());
    }

    #[test]
    fn revert_of_x_is_x() {
        // S(x) = x  →  T(y) = y.
        let series = taylor_from(&[0, 1, 0, 0]);
        let got = revert(&series, None).expect("revert");
        assert!(got.coeff(0).is_zero());
        assert!(got.coeff(1).is_one());
        assert!(got.coeff(2).is_zero());
    }

    #[test]
    fn revert_of_two_x_is_half_y() {
        // S(x) = 2x  →  T(y) = y/2.
        let series = taylor_from(&[0, 2, 0, 0]);
        let got = revert(&series, None).expect("revert");
        assert!(got.coeff(0).is_zero());
        let r = rational_coeff(&got, 1).expect("rational a_1");
        assert!((r.to_f64() - 0.5).abs() < 1e-15);
        assert!(got.coeff(2).is_zero());
    }

    #[test]
    fn revert_of_x_plus_x_squared() {
        // S(x) = x + x^2.  Known inverse: T(y) = y - y^2 + 2y^3 - 5y^4 + ...
        // Check b_1..b_4 = 1, -1, 2, -5.
        let series = taylor_from(&[0, 1, 1, 0, 0]);
        let got = revert(&series, None).expect("revert");
        let c1 = to_f64(&got.coeff(1)).unwrap();
        let c2 = to_f64(&got.coeff(2)).unwrap();
        let c3 = to_f64(&got.coeff(3)).unwrap();
        let c4 = to_f64(&got.coeff(4)).unwrap();
        assert!((c1 - 1.0).abs() < 1e-12, "b_1 = 1");
        assert!((c2 + 1.0).abs() < 1e-12, "b_2 = -1, got {c2}");
        assert!((c3 - 2.0).abs() < 1e-12, "b_3 = 2, got {c3}");
        assert!((c4 + 5.0).abs() < 1e-12, "b_4 = -5, got {c4}");
    }

    // ── Trace plumbing ──

    #[test]
    fn compose_records_trace_step() {
        let outer = taylor_from(&[0, 1, 0, 0]);
        let inner = taylor_from(&[0, 1, 0, 0]);
        let mut trace = Trace::new();
        let _ = compose(&outer, &inner, Some(&mut trace)).unwrap();
        assert_eq!(trace.steps()[0].tag, TechniqueTag::SeriesComposition);
    }

    #[test]
    fn revert_records_trace_step() {
        let series = taylor_from(&[0, 1, 0, 0]);
        let mut trace = Trace::new();
        let _ = revert(&series, Some(&mut trace)).unwrap();
        assert_eq!(trace.steps()[0].tag, TechniqueTag::LagrangeReversion);
    }

    // ── SmallInt → i64 helper used by the compose_identity_is_inner test ──

    // Tiny compatibility wrapper: treat `Expr::Integer(_)` as i64 so tests
    // assert on structural equality via a round-trip.
    trait ExprToI64 {
        fn to_i64_expect(self) -> i64;
    }
    impl ExprToI64 for Expr {
        fn to_i64_expect(self) -> i64 {
            match self {
                Expr::Integer(n) => n.to_i64().expect("fits in i64"),
                other => panic!("expected integer, got {other}"),
            }
        }
    }

    // SmallInt is imported for `BigRational::from_integer`.
    #[allow(dead_code)]
    fn _use_smallint(s: SmallInt) -> SmallInt {
        s
    }
}
