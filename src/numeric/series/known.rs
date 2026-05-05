//! Pre-computed Taylor series for common functions.
//!
//! These constructors build series directly from closed-form coefficient
//! formulas, which is faster than the general differentiation-based approach
//! and avoids floating-point accumulation errors.
//!
//! All series are Maclaurin series (expansion around zero) unless the caller
//! passes a non-zero `center`; in that case the coefficients are computed
//! symbolically and may remain unevaluated.
//!
//! | Function   | Non-zero coefficients        | Radius |
//! |------------|------------------------------|--------|
//! | `sin(x)`   | odd powers: `(-1)^k/(2k+1)!` | ∞      |
//! | `cos(x)`   | even powers: `(-1)^k/(2k)!`  | ∞      |
//! | `exp(x)`   | all powers: `1/n!`           | ∞      |
//! | `ln(1+x)`  | n≥1: `(-1)^{n+1}/n`          | 1      |
//! | `atan(x)`  | odd powers: `(-1)^k/(2k+1)`  | 1      |

use std::sync::Arc;

use super::super::{expr::Expr, BigRational};
use super::TaylorSeries;
use crate::numeric::ring::Ring;
use crate::numeric::SymbolId;

// ── Public API ────────────────────────────────────────────────────────────────

/// Taylor series for `sin(x)` around `center` up to `order`.
///
/// Maclaurin coefficients: `a_{2k+1} = (-1)^k / (2k+1)!`, all even = 0.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{Expr, SymbolId};
/// use thales::numeric::series::sin_series;
///
/// let x_id = SymbolId::intern("known_sin");
/// let ts   = sin_series(x_id, &Expr::int(0), 5);
/// assert!(ts.coeff(0).is_zero());
/// assert!(ts.coeff(1).is_one());
/// ```
pub fn sin_series(var: SymbolId, center: &Arc<Expr>, order: usize) -> TaylorSeries {
    let coefficients = (0..=order)
        .map(|n| {
            if n % 2 == 0 {
                // Even powers: coefficient is 0
                Expr::int(0)
            } else {
                // Odd power n = 2k+1 where k = (n-1)/2
                let k = (n - 1) / 2;
                // (-1)^k / n!
                let sign: i64 = if k % 2 == 0 { 1 } else { -1 };
                rational_expr(sign, factorial(n))
            }
        })
        .collect();
    TaylorSeries::from_coefficients(center.clone(), var, coefficients)
}

/// Taylor series for `cos(x)` around `center` up to `order`.
///
/// Maclaurin coefficients: `a_{2k} = (-1)^k / (2k)!`, all odd = 0.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{Expr, SymbolId};
/// use thales::numeric::series::cos_series;
///
/// let x_id = SymbolId::intern("known_cos");
/// let ts   = cos_series(x_id, &Expr::int(0), 4);
/// assert!(ts.coeff(0).is_one());
/// assert!(ts.coeff(1).is_zero());
/// ```
pub fn cos_series(var: SymbolId, center: &Arc<Expr>, order: usize) -> TaylorSeries {
    let coefficients = (0..=order)
        .map(|n| {
            if n % 2 != 0 {
                // Odd powers: coefficient is 0
                Expr::int(0)
            } else {
                // Even power n = 2k where k = n/2
                let k = n / 2;
                // (-1)^k / n!
                let sign: i64 = if k % 2 == 0 { 1 } else { -1 };
                rational_expr(sign, factorial(n))
            }
        })
        .collect();
    TaylorSeries::from_coefficients(center.clone(), var, coefficients)
}

/// Taylor series for `exp(x)` around `center` up to `order`.
///
/// Maclaurin coefficients: `a_n = 1/n!`.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{Expr, SymbolId};
/// use thales::numeric::series::exp_series;
///
/// let x_id = SymbolId::intern("known_exp");
/// let ts   = exp_series(x_id, &Expr::int(0), 3);
/// assert!(ts.coeff(0).is_one());
/// assert!(ts.coeff(1).is_one());
/// ```
pub fn exp_series(var: SymbolId, center: &Arc<Expr>, order: usize) -> TaylorSeries {
    let coefficients = (0..=order)
        .map(|n| rational_expr(1, factorial(n)))
        .collect();
    TaylorSeries::from_coefficients(center.clone(), var, coefficients)
}

/// Taylor series for `ln(1+x)` around `center = 0` up to `order`.
///
/// Maclaurin coefficients: `a_0 = 0`, `a_n = (-1)^{n+1} / n` for `n ≥ 1`.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{Expr, SymbolId};
/// use thales::numeric::series::ln_series;
///
/// let x_id = SymbolId::intern("known_ln");
/// let ts   = ln_series(x_id, &Expr::int(0), 4);
/// assert!(ts.coeff(0).is_zero());
/// assert!(ts.coeff(1).is_one());
/// ```
pub fn ln_series(var: SymbolId, center: &Arc<Expr>, order: usize) -> TaylorSeries {
    let coefficients = (0..=order)
        .map(|n| {
            if n == 0 {
                Expr::int(0)
            } else {
                // (-1)^{n+1} / n
                let sign: i64 = if (n + 1) % 2 == 0 { 1 } else { -1 };
                rational_expr(sign, n as i64)
            }
        })
        .collect();
    TaylorSeries::from_coefficients(center.clone(), var, coefficients)
}

/// Taylor series for `atan(x)` around `center = 0` up to `order`.
///
/// Maclaurin coefficients: `a_{2k+1} = (-1)^k / (2k+1)`, all even = 0.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{Expr, SymbolId};
/// use thales::numeric::series::atan_series;
///
/// let x_id = SymbolId::intern("known_atan");
/// let ts   = atan_series(x_id, &Expr::int(0), 5);
/// assert!(ts.coeff(0).is_zero());
/// assert!(ts.coeff(1).is_one());
/// ```
pub fn atan_series(var: SymbolId, center: &Arc<Expr>, order: usize) -> TaylorSeries {
    let coefficients = (0..=order)
        .map(|n| {
            if n % 2 == 0 {
                Expr::int(0)
            } else {
                // n = 2k+1, k = (n-1)/2
                let k = (n - 1) / 2;
                // (-1)^k / (2k+1) = (-1)^k / n
                let sign: i64 = if k % 2 == 0 { 1 } else { -1 };
                rational_expr(sign, n as i64)
            }
        })
        .collect();
    TaylorSeries::from_coefficients(center.clone(), var, coefficients)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build `sign / denom` as an `Arc<Expr>`, reducing to an integer when the
/// denominator is 1.
fn rational_expr(sign: i64, denom: i64) -> Arc<Expr> {
    debug_assert!(denom > 0, "rational_expr: denom must be positive");
    if denom == 1 {
        return Expr::int(sign);
    }
    // Use BigRational which automatically reduces (e.g. 2/2 → 1).
    let r = BigRational::from_i64(sign, denom);
    if r.denom().is_one() {
        // Reduced to an integer.
        if let Some(n) = r.numer().to_i64() {
            return Expr::int(n);
        }
    }
    Arc::new(Expr::Rational(r))
}

/// Compute `n!` as `i64`.  Saturates at `i64::MAX` for `n > 20`.
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
    use crate::numeric::SymbolId;

    fn to_f64(expr: &Arc<Expr>) -> f64 {
        match expr.as_ref() {
            Expr::Integer(n) => n.to_i64().unwrap_or(0) as f64,
            Expr::Rational(r) => r.to_f64(),
            Expr::Float(f) => *f,
            _ => panic!("non-numeric: {expr}"),
        }
    }

    // ── sin ──────────────────────────────────────────────────────────────

    #[test]
    fn test_sin_series_zero_a0() {
        let x_id = SymbolId::intern("known_sin_a0");
        let ts = sin_series(x_id, &Expr::int(0), 7);
        assert!(ts.coeff(0).is_zero());
    }

    #[test]
    fn test_sin_series_a1() {
        let x_id = SymbolId::intern("known_sin_a1");
        let ts = sin_series(x_id, &Expr::int(0), 7);
        assert!(ts.coeff(1).is_one());
    }

    #[test]
    fn test_sin_series_a3() {
        // a_3 = -1/6
        let x_id = SymbolId::intern("known_sin_a3");
        let ts = sin_series(x_id, &Expr::int(0), 7);
        let v = to_f64(&ts.coeff(3));
        assert!((v + 1.0 / 6.0).abs() < 1e-12, "a_3 = -1/6, got {v}");
    }

    #[test]
    fn test_sin_series_a5() {
        // a_5 = 1/120
        let x_id = SymbolId::intern("known_sin_a5");
        let ts = sin_series(x_id, &Expr::int(0), 7);
        let v = to_f64(&ts.coeff(5));
        assert!((v - 1.0 / 120.0).abs() < 1e-12, "a_5 = 1/120, got {v}");
    }

    #[test]
    fn test_sin_series_even_are_zero() {
        let x_id = SymbolId::intern("known_sin_evz");
        let ts = sin_series(x_id, &Expr::int(0), 6);
        for n in [0, 2, 4, 6] {
            assert!(ts.coeff(n).is_zero(), "sin a_{n} should be 0");
        }
    }

    // ── cos ──────────────────────────────────────────────────────────────

    #[test]
    fn test_cos_series_a0() {
        let x_id = SymbolId::intern("known_cos_a0");
        let ts = cos_series(x_id, &Expr::int(0), 4);
        assert!(ts.coeff(0).is_one());
    }

    #[test]
    fn test_cos_series_a1_zero() {
        let x_id = SymbolId::intern("known_cos_a1");
        let ts = cos_series(x_id, &Expr::int(0), 4);
        assert!(ts.coeff(1).is_zero());
    }

    #[test]
    fn test_cos_series_a2() {
        // a_2 = -1/2
        let x_id = SymbolId::intern("known_cos_a2");
        let ts = cos_series(x_id, &Expr::int(0), 4);
        let v = to_f64(&ts.coeff(2));
        assert!((v + 0.5).abs() < 1e-12, "a_2 = -1/2, got {v}");
    }

    #[test]
    fn test_cos_series_odd_are_zero() {
        let x_id = SymbolId::intern("known_cos_odz");
        let ts = cos_series(x_id, &Expr::int(0), 5);
        for n in [1, 3, 5] {
            assert!(ts.coeff(n).is_zero(), "cos a_{n} should be 0");
        }
    }

    // ── exp ──────────────────────────────────────────────────────────────

    #[test]
    fn test_exp_series_a0() {
        let x_id = SymbolId::intern("known_exp_a0");
        let ts = exp_series(x_id, &Expr::int(0), 4);
        assert!(ts.coeff(0).is_one());
    }

    #[test]
    fn test_exp_series_a1() {
        let x_id = SymbolId::intern("known_exp_a1");
        let ts = exp_series(x_id, &Expr::int(0), 4);
        assert!(ts.coeff(1).is_one());
    }

    #[test]
    fn test_exp_series_a2() {
        // a_2 = 1/2
        let x_id = SymbolId::intern("known_exp_a2");
        let ts = exp_series(x_id, &Expr::int(0), 4);
        let v = to_f64(&ts.coeff(2));
        assert!((v - 0.5).abs() < 1e-12, "a_2 = 1/2, got {v}");
    }

    #[test]
    fn test_exp_series_a3() {
        // a_3 = 1/6
        let x_id = SymbolId::intern("known_exp_a3");
        let ts = exp_series(x_id, &Expr::int(0), 4);
        let v = to_f64(&ts.coeff(3));
        assert!((v - 1.0 / 6.0).abs() < 1e-12, "a_3 = 1/6, got {v}");
    }

    // ── ln ───────────────────────────────────────────────────────────────

    #[test]
    fn test_ln_series_a0_zero() {
        let x_id = SymbolId::intern("known_ln_a0");
        let ts = ln_series(x_id, &Expr::int(0), 4);
        assert!(ts.coeff(0).is_zero());
    }

    #[test]
    fn test_ln_series_a1() {
        let x_id = SymbolId::intern("known_ln_a1");
        let ts = ln_series(x_id, &Expr::int(0), 4);
        assert!(ts.coeff(1).is_one());
    }

    #[test]
    fn test_ln_series_a2() {
        // a_2 = -1/2
        let x_id = SymbolId::intern("known_ln_a2");
        let ts = ln_series(x_id, &Expr::int(0), 4);
        let v = to_f64(&ts.coeff(2));
        assert!((v + 0.5).abs() < 1e-12, "a_2 = -1/2, got {v}");
    }

    #[test]
    fn test_ln_series_a3() {
        // a_3 = 1/3
        let x_id = SymbolId::intern("known_ln_a3");
        let ts = ln_series(x_id, &Expr::int(0), 4);
        let v = to_f64(&ts.coeff(3));
        assert!((v - 1.0 / 3.0).abs() < 1e-12, "a_3 = 1/3, got {v}");
    }

    // ── atan ─────────────────────────────────────────────────────────────

    #[test]
    fn test_atan_series_a0_zero() {
        let x_id = SymbolId::intern("known_atan_a0");
        let ts = atan_series(x_id, &Expr::int(0), 5);
        assert!(ts.coeff(0).is_zero());
    }

    #[test]
    fn test_atan_series_a1() {
        let x_id = SymbolId::intern("known_atan_a1");
        let ts = atan_series(x_id, &Expr::int(0), 5);
        assert!(ts.coeff(1).is_one());
    }

    #[test]
    fn test_atan_series_a3() {
        // a_3 = -1/3
        let x_id = SymbolId::intern("known_atan_a3");
        let ts = atan_series(x_id, &Expr::int(0), 5);
        let v = to_f64(&ts.coeff(3));
        assert!((v + 1.0 / 3.0).abs() < 1e-12, "a_3 = -1/3, got {v}");
    }

    #[test]
    fn test_atan_series_even_are_zero() {
        let x_id = SymbolId::intern("known_atan_evz");
        let ts = atan_series(x_id, &Expr::int(0), 6);
        for n in [0, 2, 4, 6] {
            assert!(ts.coeff(n).is_zero(), "atan a_{n} should be 0");
        }
    }

    // ── rational_expr helper ─────────────────────────────────────────────

    #[test]
    fn test_rational_expr_integer_result() {
        // 6/3 should reduce to integer 2
        let e = rational_expr(6, 3);
        match e.as_ref() {
            Expr::Integer(n) => assert_eq!(n.to_i64(), Some(2)),
            _ => panic!("expected integer"),
        }
    }

    #[test]
    fn test_rational_expr_negative() {
        let e = rational_expr(-1, 2);
        let v = to_f64(&e);
        assert!((v + 0.5).abs() < 1e-12);
    }

    // ── factorial helper ─────────────────────────────────────────────────

    #[test]
    fn test_factorial_zero() {
        assert_eq!(factorial(0), 1);
    }

    #[test]
    fn test_factorial_five() {
        assert_eq!(factorial(5), 120);
    }
}
