//! Small helpers shared across the limits submodules.

use std::sync::Arc;

use super::super::expr::Expr;
use super::super::BigRational;

/// Maximum number of L'Hôpital iterations before giving up.
pub(crate) const LHOPITAL_MAX_ITER: usize = 5;

/// Taylor series order used when falling back to series analysis.
pub(crate) const SERIES_ORDER: usize = 6;

/// Convert a literal numeric `Expr` to an `f64`. Returns `None` for anything
/// that isn't a plain Integer/Rational/Float literal.
pub(crate) fn expr_to_f64(expr: &Arc<Expr>) -> Option<f64> {
    match expr.as_ref() {
        Expr::Integer(n) => n.to_i64().map(|v| v as f64),
        Expr::Rational(r) => Some(r.to_f64()),
        Expr::Float(f) => Some(*f),
        _ => None,
    }
}

/// Convert a `BigRational` to `f64`.
pub(crate) fn rational_to_f64_opt(r: &BigRational) -> Option<f64> {
    Some(r.to_f64())
}

/// Convert a float limit value to the tightest exact `Expr` form available.
/// Near-integer results promote to [`Expr::int`]; other values stay as
/// [`Expr::float`]. The tolerance is loose enough to absorb the 1e-7-scale
/// noise from our one-sided numerical probe, tight enough that genuinely
/// non-integer limits (e.g. `0.5`, `e`, `π`) stay as floats.
pub(crate) fn float_to_exact(v: f64) -> Arc<Expr> {
    if v.is_finite() {
        let rounded = v.round();
        if (v - rounded).abs() < 1e-6 && rounded.abs() < i64::MAX as f64 {
            return Expr::int(rounded as i64);
        }
    }
    Expr::float(v)
}
