//! FFI implementation functions for precision evaluation and manual-computation optimization.

use crate::parser::parse_expression;

/// Parse a mode string into `PrecisionMode`, folding `precision` into the variant.
fn parse_precision_mode(
    mode: &str,
    precision: u32,
) -> Result<crate::precision::PrecisionMode, String> {
    use crate::precision::PrecisionMode;
    match mode {
        "fixed" => Ok(PrecisionMode::FixedDecimal(precision)),
        "significant" => Ok(PrecisionMode::SignificantFigures(precision)),
        "arbitrary" => Ok(PrecisionMode::Arbitrary),
        "full" => Ok(PrecisionMode::Full),
        other => Err(format!(
            "Unknown precision mode '{other}': expected fixed, significant, arbitrary, or full"
        )),
    }
}

/// Parse a rounding string into `RoundingMode`.
fn parse_rounding_mode(rounding: &str) -> Result<crate::precision::RoundingMode, String> {
    use crate::precision::RoundingMode;
    match rounding {
        "half_up" | "up" => Ok(RoundingMode::HalfUp),
        "half_even" | "even" | "banker" => Ok(RoundingMode::HalfEven),
        "truncate" | "trunc" => Ok(RoundingMode::Truncate),
        "ceiling" | "ceil" => Ok(RoundingMode::Ceiling),
        "floor" => Ok(RoundingMode::Floor),
        "" => Ok(RoundingMode::default()),
        other => Err(format!(
            "Unknown rounding mode '{other}': expected half_up, half_even, truncate, ceiling, or floor"
        )),
    }
}

/// Evaluate an expression with configurable precision and rounding.
pub(super) fn evaluate_with_precision_ffi(
    expression: &str,
    values_json: &str,
    mode: &str,
    precision: u32,
    rounding: &str,
) -> Result<super::ffi::PrecisionEvaluationResultFFI, String> {
    use crate::precision::EvalContext;
    use std::collections::HashMap;

    let precision_mode = parse_precision_mode(mode, precision)?;
    let rounding_mode = parse_rounding_mode(rounding)?;

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let values: HashMap<String, f64> = if values_json.is_empty() || values_json == "{}" {
        HashMap::new()
    } else {
        serde_json::from_str(values_json)
            .map_err(|e| format!("Failed to parse values JSON: {}", e))?
    };

    let mut ctx = EvalContext::new(precision_mode).with_rounding(rounding_mode);
    for (name, val) in values {
        ctx.set_f64(&name, val);
    }

    let precision_mode_str = format!("{:?}", precision_mode);
    let rounding_mode_str = format!("{:?}", rounding_mode);

    match ctx.evaluate(&expr) {
        Ok(value) => {
            let numeric = value.as_f64();
            let value_string = format!("{}", value);
            Ok(super::ffi::PrecisionEvaluationResultFFI {
                original: expression.to_string(),
                value: numeric,
                value_string,
                precision_mode: precision_mode_str,
                rounding_mode: rounding_mode_str,
                success: true,
                error_message: String::new(),
            })
        }
        Err(e) => Ok(super::ffi::PrecisionEvaluationResultFFI {
            original: expression.to_string(),
            value: f64::NAN,
            value_string: String::new(),
            precision_mode: precision_mode_str,
            rounding_mode: rounding_mode_str,
            success: false,
            error_message: format!("{}", e),
        }),
    }
}

/// Analyze an expression and return optimized manual-computation steps as JSON.
///
/// Returns a JSON object with fields `steps` (array of step descriptions) and
/// `chains` (multiplicative chains suitable for slide-rule computation).
pub(super) fn optimize_for_manual_computation_ffi(expression: &str) -> Result<String, String> {
    use crate::optimization::{
        analyze_expression, find_multiplicative_chains, optimize_computation_order,
        to_manual_steps, OperationConfig,
    };

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;
    let config = OperationConfig::default();

    let raw_steps = analyze_expression(&expr);
    let optimized = optimize_computation_order(&raw_steps, &config);
    let manual = to_manual_steps(&optimized, &config);
    let chains = find_multiplicative_chains(&expr);

    let steps_json: Vec<serde_json::Value> = manual
        .iter()
        .map(|s| {
            serde_json::json!({
                "instruction": s.instruction,
                "precision": s.precision
            })
        })
        .collect();

    let chains_json: Vec<serde_json::Value> = chains
        .iter()
        .map(|c| {
            let numerator: Vec<String> = c
                .numerator_factors
                .iter()
                .map(|e| format!("{}", e))
                .collect();
            let denominator: Vec<String> = c
                .denominator_factors
                .iter()
                .map(|e| format!("{}", e))
                .collect();
            serde_json::json!({
                "numerator": numerator,
                "denominator": denominator
            })
        })
        .collect();

    let result = serde_json::json!({
        "original": expression,
        "steps": steps_json,
        "multiplicative_chains": chains_json,
        "step_count": manual.len()
    });

    serde_json::to_string(&result).map_err(|e| format!("Failed to serialize result: {}", e))
}

/// Apply small-angle approximations to trigonometric functions in an expression.
pub(super) fn small_angle_approximation_ffi(
    expression: &str,
    variable: &str,
    threshold: f64,
) -> Result<String, String> {
    use crate::approximations::apply_small_angle_approx;
    use crate::ast::Variable;

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;
    let var = Variable::new(variable);

    match apply_small_angle_approx(&expr, &var, threshold) {
        Some(approx) => {
            let result = serde_json::json!({
                "original": expression,
                "approximation": format!("{}", approx.approximation),
                "approximation_latex": approx.approximation.to_latex(),
                "formula_used": approx.formula_used,
                "error_bound": approx.error_bound,
                "valid_range": {
                    "lower": approx.valid_range.0,
                    "upper": approx.valid_range.1
                }
            });
            serde_json::to_string(&result).map_err(|e| format!("Failed to serialize result: {}", e))
        }
        None => {
            let result = serde_json::json!({
                "original": expression,
                "approximation": expression,
                "approximation_latex": expr.to_latex(),
                "formula_used": "no approximation applied",
                "error_bound": 0.0,
                "valid_range": { "lower": -threshold, "upper": threshold }
            });
            serde_json::to_string(&result).map_err(|e| format!("Failed to serialize result: {}", e))
        }
    }
}
