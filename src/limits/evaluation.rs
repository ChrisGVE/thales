//! Limit evaluation via delegation to the `Arc<Expr>`-native engine.
//!
//! Thin Expression-typed facade over [`crate::numeric::limits::limit`]. The
//! legacy public API is retained: `LimitPoint::Value(f64)` is the entry type,
//! `LimitResult::Value(f64)` is the exit type, and L'Hôpital / series / float
//! probes all run inside the unified `numeric::limits` engine.
//!
//! `limit_left` and `limit_right` now route through
//! `LimitPoint::FromLeft(_)` / `FromRight(_)` rather than the previous
//! `point ± ε` numerical probe — the numeric engine performs that probe (and
//! more) internally.

use std::sync::Arc;

use crate::ast::Expression;
use crate::numeric::compile::compile;
use crate::numeric::expr::Expr;
use crate::numeric::limits::{
    limit as numeric_limit, LimitPoint as NumLimitPoint, LimitResult as NumLimitResult,
};
use crate::numeric::SymbolId;

use super::types::{try_expr_to_f64, IndeterminateForm, LimitError, LimitPoint, LimitResult};

/// Compute `lim_{var → approaches} expr`.
///
/// Delegates to the numeric engine. Direct substitution, L'Hôpital's rule,
/// series analysis, and float-probe disambiguation all run inside
/// `numeric::limits::limit`.
pub fn limit(
    expr: &Expression,
    var: &str,
    approaches: LimitPoint,
) -> Result<LimitResult, LimitError> {
    let var_id = SymbolId::intern(var);
    let expr_arc = compile(expr);
    let num_point = legacy_point_to_numeric(&approaches);
    let result = numeric_limit(&expr_arc, var_id, &num_point);
    convert_numeric_result(&result, &approaches)
}

/// Compute `lim_{x → a⁻} expr`.
#[must_use = "computing limits returns a result that should be used"]
pub fn limit_left(
    expr: &Expression,
    var: &str,
    approaches: f64,
) -> Result<LimitResult, LimitError> {
    let var_id = SymbolId::intern(var);
    let expr_arc = compile(expr);
    let point_arc = float_to_expr_arc(approaches);
    let num_point = NumLimitPoint::FromLeft(point_arc);
    let result = numeric_limit(&expr_arc, var_id, &num_point);
    convert_numeric_result(&result, &LimitPoint::Value(approaches))
}

/// Compute `lim_{x → a⁺} expr`.
#[must_use = "computing limits returns a result that should be used"]
pub fn limit_right(
    expr: &Expression,
    var: &str,
    approaches: f64,
) -> Result<LimitResult, LimitError> {
    let var_id = SymbolId::intern(var);
    let expr_arc = compile(expr);
    let point_arc = float_to_expr_arc(approaches);
    let num_point = NumLimitPoint::FromRight(point_arc);
    let result = numeric_limit(&expr_arc, var_id, &num_point);
    convert_numeric_result(&result, &LimitPoint::Value(approaches))
}

/// Compute a limit, applying L'Hôpital's rule on indeterminate `0/0` / `∞/∞`
/// forms. Since the unified engine already applies L'Hôpital internally, this
/// is equivalent to [`limit`] — kept for API compatibility.
#[must_use = "computing limits returns a result that should be used"]
pub fn limit_with_lhopital(
    expr: &Expression,
    var: &str,
    approaches: LimitPoint,
) -> Result<LimitResult, LimitError> {
    limit(expr, var, approaches)
}

// ── Conversion helpers ────────────────────────────────────────────────────────

fn legacy_point_to_numeric(point: &LimitPoint) -> NumLimitPoint {
    match point {
        LimitPoint::Value(v) => NumLimitPoint::Value(float_to_expr_arc(*v)),
        LimitPoint::PositiveInfinity => NumLimitPoint::PosInfinity,
        LimitPoint::NegativeInfinity => NumLimitPoint::NegInfinity,
    }
}

/// Convert an f64 to the tightest-fitting `Arc<Expr>` literal (integer if the
/// float is exactly integral, otherwise `Float`).
fn float_to_expr_arc(v: f64) -> Arc<Expr> {
    if v.is_finite() && v.fract() == 0.0 && v.abs() < i64::MAX as f64 {
        Expr::int(v as i64)
    } else {
        Arc::new(Expr::Float(v))
    }
}

fn convert_numeric_result(
    result: &NumLimitResult,
    original: &LimitPoint,
) -> Result<LimitResult, LimitError> {
    match result {
        NumLimitResult::Value(arc) => {
            let expr = crate::numeric::compile::decompile(arc);
            if let Some(f) = try_expr_to_f64(&expr) {
                Ok(LimitResult::Value(f))
            } else {
                Ok(LimitResult::Expression(expr))
            }
        }
        NumLimitResult::PosInfinity => Ok(LimitResult::PositiveInfinity),
        NumLimitResult::NegInfinity => Ok(LimitResult::NegativeInfinity),
        NumLimitResult::DoesNotExist => Err(LimitError::DoesNotExist(format!(
            "Limit does not exist at {:?}",
            original
        ))),
        NumLimitResult::Indeterminate => {
            Err(LimitError::Indeterminate(IndeterminateForm::ZeroOverZero))
        }
    }
}
