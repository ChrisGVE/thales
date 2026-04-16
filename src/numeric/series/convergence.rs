//! Convergence radius estimation for Taylor series.
//!
//! Uses the ratio test: `R = lim_{n→∞} |a_n / a_{n+1}|`.
//!
//! The estimate is computed from the last two non-zero numerical coefficients
//! of the series.  When coefficients are symbolic (non-numeric), `None` is
//! returned because the limit cannot be evaluated.

use std::sync::Arc;

use super::super::expr::Expr;
use super::TaylorSeries;

// ── Public API ────────────────────────────────────────────────────────────────

/// Estimate the convergence radius of a Taylor series via the ratio test.
///
/// The ratio test states `R = lim_{n→∞} |a_n / a_{n+1}|`.  This function
/// estimates `R` from the last consecutive pair of non-zero numeric
/// coefficients.
///
/// Returns:
/// - `Some(r)` — a numeric [`Expr`] representing the estimated radius.
/// - `None` — when no two consecutive numeric non-zero coefficients are
///   available, or when all high-order coefficients are zero (suggesting
///   an infinite radius for the *computed* truncation).
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{Expr, SymbolId};
/// use thales::numeric::series::{taylor, convergence_radius};
/// use thales::numeric::normalize;
///
/// // 1/(1-x): all coefficients are 1 → ratio = 1 → R = 1
/// let x_id = SymbolId::intern("conv_geo");
/// let x    = Expr::symbol("conv_geo");
/// let expr = normalize::pow(normalize::sub(Expr::int(1), x), Expr::int(-1));
/// let ts   = taylor(&expr, x_id, &Expr::int(0), 5);
/// let r    = convergence_radius(&ts).expect("should have numeric radius");
/// match r.as_ref() {
///     Expr::Integer(n) => assert_eq!(n.to_i64(), Some(1)),
///     Expr::Float(f)   => assert!((*f - 1.0).abs() < 1e-9),
///     other => panic!("unexpected: {other}"),
/// }
/// ```
pub fn convergence_radius(series: &TaylorSeries) -> Option<Arc<Expr>> {
    // Collect (index, f64 value) for non-zero numeric coefficients.
    let numeric: Vec<(usize, f64)> = series
        .coefficients
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let v = to_f64(c)?;
            if v == 0.0 {
                None
            } else {
                Some((i, v))
            }
        })
        .collect();

    if numeric.len() < 2 {
        // Not enough data to form a ratio.
        return None;
    }

    // Use the last two consecutive non-zero coefficients for the ratio test.
    // "Consecutive" means consecutive non-zero entries (by index difference 1).
    // Fall back to the last two non-zero entries regardless.
    let last_pair = find_last_consecutive(&numeric).or_else(|| {
        let n = numeric.len();
        Some((numeric[n - 2], numeric[n - 1]))
    })?;

    let ((i, a_i), (j, a_j)) = last_pair;

    // R ≈ |a_i / a_j| adjusted for the power gap (j - i)
    // For a true ratio test between consecutive powers the gap is 1.
    // When gap > 1, the series has zero coefficients in between; the ratio
    // test still applies but with a gap correction: R = |a_i/a_j|^(1/(j-i)).
    let gap = (j - i) as f64;
    let raw_ratio = (a_i / a_j).abs();
    let radius = raw_ratio.powf(1.0 / gap);

    // Represent the result as a clean integer when it rounds to one.
    let rounded = radius.round();
    if (radius - rounded).abs() < 1e-9 && rounded >= 0.0 {
        Some(Expr::int(rounded as i64))
    } else {
        Some(Expr::float(radius))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Find the last pair of numerically consecutive (index gap = 1) non-zero
/// coefficient entries.
fn find_last_consecutive(numeric: &[(usize, f64)]) -> Option<((usize, f64), (usize, f64))> {
    // Scan from the end.
    for window in numeric.windows(2).rev() {
        let (i, a_i) = window[0];
        let (j, a_j) = window[1];
        if j - i == 1 {
            return Some(((i, a_i), (j, a_j)));
        }
    }
    None
}

/// Extract a finite `f64` from a numeric expression.
fn to_f64(expr: &Arc<Expr>) -> Option<f64> {
    match expr.as_ref() {
        Expr::Integer(n) => n.to_i64().map(|v| v as f64),
        Expr::Rational(r) => Some(r.to_f64()),
        Expr::Float(f) if f.is_finite() => Some(*f),
        _ => None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::series::taylor;
    use crate::numeric::{normalize, Expr, FuncId, SymbolId};

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }

    #[test]
    fn test_convergence_geometric_series() {
        // 1/(1-x) at 0: all a_n = 1 → R = 1
        let x_id = SymbolId::intern("conv_geo_t");
        let x = sym("conv_geo_t");
        let expr = normalize::pow(normalize::sub(Expr::int(1), x), Expr::int(-1));
        let ts = taylor(&expr, x_id, &Expr::int(0), 6);
        let r = convergence_radius(&ts).expect("should return radius");
        let v = to_f64_expr(&r);
        assert!((v - 1.0).abs() < 1e-9, "R should be 1, got {v}");
    }

    #[test]
    fn test_convergence_exp_series_infinite() {
        // exp(x): a_n = 1/n! → ratio test: a_n/a_{n+1} = (n+1)! / n! = n+1 → ∞
        // With finite truncation at order 5, we get an estimate > 1 but not ∞.
        // The test just checks we get Some(r) with r > 1.
        let x_id = SymbolId::intern("conv_exp_t");
        let x = sym("conv_exp_t");
        let expr = Expr::func(FuncId::Exp, vec![x]);
        let ts = taylor(&expr, x_id, &Expr::int(0), 6);
        let r = convergence_radius(&ts).expect("should return a radius");
        let v = to_f64_expr(&r);
        assert!(v > 1.0, "exp series radius should be > 1, got {v}");
    }

    #[test]
    fn test_convergence_constant_series_returns_none() {
        // Taylor of 5 (constant): only a_0 is non-zero → can't form a ratio.
        let x_id = SymbolId::intern("conv_const_t");
        let ts = taylor(&Expr::int(5), x_id, &Expr::int(0), 4);
        assert!(
            convergence_radius(&ts).is_none(),
            "constant series should return None"
        );
    }

    #[test]
    fn test_convergence_sin_series() {
        // sin(x): non-zero only at odd powers.  R should be estimated as > 1.
        let x_id = SymbolId::intern("conv_sin_t");
        let x = sym("conv_sin_t");
        let expr = Expr::func(FuncId::Sin, vec![x]);
        let ts = taylor(&expr, x_id, &Expr::int(0), 7);
        let r = convergence_radius(&ts).expect("sin should have a radius");
        let v = to_f64_expr(&r);
        assert!(v > 1.0, "sin radius should be large, got {v}");
    }

    /// Helper: pull an f64 out of a numeric Expr.
    fn to_f64_expr(expr: &Arc<Expr>) -> f64 {
        match expr.as_ref() {
            Expr::Integer(n) => n.to_i64().unwrap_or(0) as f64,
            Expr::Rational(r) => r.to_f64(),
            Expr::Float(f) => *f,
            _ => panic!("non-numeric radius: {expr}"),
        }
    }
}
