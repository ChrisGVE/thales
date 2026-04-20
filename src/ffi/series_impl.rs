//! FFI implementation functions for series expansions, special functions, and Fourier series.

use crate::parser::parse_expression;

// =============================================================================
// Series expansion operations
// =============================================================================

/// Compute Taylor series expansion around a given center point.
pub(super) fn taylor_series_ffi(
    expression: &str,
    variable: &str,
    center: f64,
    order: u32,
) -> Result<super::ffi::TaylorSeriesResultFFI, String> {
    use crate::numeric::compile::{compile, decompile};
    use crate::numeric::expr::Expr;
    use crate::numeric::series::taylor;
    use crate::numeric::SymbolId;

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;
    let arc_expr = compile(&expr);
    let var_id = SymbolId::intern(variable);
    let center_arc = Expr::float(center);

    let ts = taylor(&arc_expr, var_id, &center_arc, order as usize);
    let series_arc = ts.to_expr();
    let series_expr = decompile(&series_arc);

    Ok(super::ffi::TaylorSeriesResultFFI {
        original: expression.to_string(),
        variable: variable.to_string(),
        center,
        order,
        series: format!("{}", series_expr),
        series_latex: series_expr.to_latex(),
        success: true,
        error_message: String::new(),
    })
}

/// Compute Maclaurin series expansion (Taylor series centered at 0).
pub(super) fn maclaurin_series_ffi(
    expression: &str,
    variable: &str,
    order: u32,
) -> Result<super::ffi::TaylorSeriesResultFFI, String> {
    use crate::numeric::compile::{compile, decompile};
    use crate::numeric::expr::Expr;
    use crate::numeric::series::taylor;
    use crate::numeric::SymbolId;

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;
    let arc_expr = compile(&expr);
    let var_id = SymbolId::intern(variable);

    let ts = taylor(&arc_expr, var_id, &Expr::int(0), order as usize);
    let series_arc = ts.to_expr();
    let series_expr = decompile(&series_arc);

    Ok(super::ffi::TaylorSeriesResultFFI {
        original: expression.to_string(),
        variable: variable.to_string(),
        center: 0.0,
        order,
        series: format!("{}", series_expr),
        series_latex: series_expr.to_latex(),
        success: true,
        error_message: String::new(),
    })
}

/// Compute Laurent series expansion around a given center point.
pub(super) fn laurent_series_ffi(
    expression: &str,
    variable: &str,
    center: f64,
    neg_order: u32,
    pos_order: u32,
) -> Result<super::ffi::LaurentSeriesResultFFI, String> {
    use crate::numeric::compile::{compile, decompile};
    use crate::numeric::expr::Expr;
    use crate::numeric::series::laurent_expand;
    use crate::numeric::SymbolId;

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;
    let arc_expr = compile(&expr);
    let var_id = SymbolId::intern(variable);
    let center_arc = center_to_expr(center);

    match laurent_expand(&arc_expr, var_id, &center_arc, neg_order, pos_order, None) {
        Some(series) => {
            let series_arc = series.to_expr();
            let series_expr = decompile(&series_arc);
            Ok(super::ffi::LaurentSeriesResultFFI {
                original: expression.to_string(),
                variable: variable.to_string(),
                center,
                neg_order,
                pos_order,
                series: format!("{}", series_expr),
                series_latex: series_expr.to_latex(),
                success: true,
                error_message: String::new(),
            })
        }
        None => Ok(super::ffi::LaurentSeriesResultFFI {
            original: expression.to_string(),
            variable: variable.to_string(),
            center,
            neg_order,
            pos_order,
            series: String::new(),
            series_latex: String::new(),
            success: false,
            error_message:
                "Cannot expand: structural shift exceeds the Laurent engine's supported range"
                    .to_string(),
        }),
    }
}

/// Convert a caller-supplied f64 center into a canonical `Arc<Expr>`, using
/// an integer when the value is integer-valued so the structural matchers in
/// the Laurent engine recognize `(x − center)` shifts.
fn center_to_expr(center: f64) -> std::sync::Arc<crate::numeric::expr::Expr> {
    use crate::numeric::expr::Expr;
    if center == 0.0 {
        return Expr::int(0);
    }
    if center.is_finite() && center.fract() == 0.0 && center.abs() < (i64::MAX as f64) {
        return Expr::int(center as i64);
    }
    Expr::float(center)
}

/// Compute asymptotic series expansion of an expression.
///
/// The `direction` parameter must be one of: `"pos_infinity"`, `"neg_infinity"`, `"zero"`.
pub(super) fn asymptotic_series_ffi(
    expression: &str,
    variable: &str,
    direction: &str,
    num_terms: u32,
) -> Result<super::ffi::AsymptoticSeriesResultFFI, String> {
    use crate::numeric::compile::{compile, decompile};
    use crate::numeric::series::{asymptotic, AsymptoticDirection};
    use crate::numeric::SymbolId;

    let dir = match direction {
        "pos_infinity" => AsymptoticDirection::PosInfinity,
        "neg_infinity" => AsymptoticDirection::NegInfinity,
        "zero" => AsymptoticDirection::Zero,
        other => {
            return Err(format!(
                "Unknown direction '{other}': expected pos_infinity, neg_infinity, or zero"
            ))
        }
    };

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;
    let arc_expr = compile(&expr);
    let var_id = SymbolId::intern(variable);

    match asymptotic(&arc_expr, var_id, dir, num_terms as usize, None) {
        Some(series) => {
            let series_arc = series.to_expr();
            let series_expr = decompile(&series_arc);
            Ok(super::ffi::AsymptoticSeriesResultFFI {
                original: expression.to_string(),
                variable: variable.to_string(),
                direction: direction.to_string(),
                num_terms,
                series: format!("{}", series_expr),
                series_latex: series_expr.to_latex(),
                success: true,
                error_message: String::new(),
            })
        }
        None => Ok(super::ffi::AsymptoticSeriesResultFFI {
            original: expression.to_string(),
            variable: variable.to_string(),
            direction: direction.to_string(),
            num_terms,
            series: String::new(),
            series_latex: String::new(),
            success: false,
            error_message: "Cannot expand: expression is not a Laurent polynomial in the variable"
                .to_string(),
        }),
    }
}

/// Compose two power series: compute outer(inner(x)).
///
/// Both expressions are first expanded as Maclaurin series to the given order,
/// then composed. The inner series must have a zero constant term.
pub(super) fn compose_series_ffi(
    outer: &str,
    inner: &str,
    variable: &str,
    order: u32,
) -> Result<super::ffi::TaylorSeriesResultFFI, String> {
    use crate::numeric::compile::{compile, decompile};
    use crate::numeric::expr::Expr;
    use crate::numeric::series::{compose, taylor};
    use crate::numeric::SymbolId;

    let outer_expr =
        parse_expression(outer).map_err(|e| format!("Parse error in outer: {:?}", e))?;
    let inner_expr =
        parse_expression(inner).map_err(|e| format!("Parse error in inner: {:?}", e))?;
    let outer_arc = compile(&outer_expr);
    let inner_arc = compile(&inner_expr);
    let var_id = SymbolId::intern(variable);
    let zero = Expr::int(0);

    let outer_series = taylor(&outer_arc, var_id, &zero, order as usize);
    let inner_series = taylor(&inner_arc, var_id, &zero, order as usize);

    let original_label = format!("({}) \u{2218} ({})", outer, inner);

    match compose(&outer_series, &inner_series, None) {
        Some(composed) => {
            let expr_arc = composed.to_expr();
            let series_expr = decompile(&expr_arc);
            Ok(super::ffi::TaylorSeriesResultFFI {
                original: original_label,
                variable: variable.to_string(),
                center: 0.0,
                order,
                series: format!("{}", series_expr),
                series_latex: series_expr.to_latex(),
                success: true,
                error_message: String::new(),
            })
        }
        None => Ok(super::ffi::TaylorSeriesResultFFI {
            original: original_label,
            variable: variable.to_string(),
            center: 0.0,
            order,
            series: String::new(),
            series_latex: String::new(),
            success: false,
            error_message:
                "Cannot compose: variables/centers differ or inner constant term is non-zero"
                    .to_string(),
        }),
    }
}

/// Compute the compositional inverse (reversion) of a power series.
///
/// The expression is expanded as a Maclaurin series, then reverted.
/// The series must have a zero constant term and nonzero linear coefficient.
pub(super) fn reversion_series_ffi(
    expression: &str,
    variable: &str,
    order: u32,
) -> Result<super::ffi::TaylorSeriesResultFFI, String> {
    use crate::numeric::compile::{compile, decompile};
    use crate::numeric::expr::Expr;
    use crate::numeric::series::{revert, taylor};
    use crate::numeric::SymbolId;

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;
    let arc_expr = compile(&expr);
    let var_id = SymbolId::intern(variable);

    let series = taylor(&arc_expr, var_id, &Expr::int(0), order as usize);

    match revert(&series, None) {
        Some(reverted) => {
            let rev_arc = reverted.to_expr();
            let rev_expr = decompile(&rev_arc);
            Ok(super::ffi::TaylorSeriesResultFFI {
                original: expression.to_string(),
                variable: variable.to_string(),
                center: 0.0,
                order,
                series: format!("{}", rev_expr),
                series_latex: rev_expr.to_latex(),
                success: true,
                error_message: String::new(),
            })
        }
        None => Ok(super::ffi::TaylorSeriesResultFFI {
            original: expression.to_string(),
            variable: variable.to_string(),
            center: 0.0,
            order,
            series: String::new(),
            series_latex: String::new(),
            success: false,
            error_message:
                "Cannot revert: constant term must be zero and linear coefficient non-zero"
                    .to_string(),
        }),
    }
}

// =============================================================================
// Special functions
// =============================================================================

/// Compute the Gamma function with derivation steps.
pub(super) fn gamma_ffi(x: f64) -> Result<super::ffi::SpecialFunctionResultFFI, String> {
    use crate::special::gamma;

    let x_expr = crate::ast::Expression::Float(x);

    let result = gamma(&x_expr);

    match result {
        Ok(gamma_result) => {
            let steps_json = serde_json::to_string(&gamma_result.derivation_steps)
                .map_err(|e| format!("Failed to serialize derivation steps: {}", e))?;

            Ok(super::ffi::SpecialFunctionResultFFI {
                value: format!("{}", gamma_result.value),
                value_latex: gamma_result.value.to_latex(),
                numeric_value: gamma_result.numeric_value.unwrap_or(f64::NAN),
                derivation_steps: steps_json,
                success: true,
                error_message: String::new(),
            })
        }
        Err(e) => Ok(super::ffi::SpecialFunctionResultFFI {
            value: String::new(),
            value_latex: String::new(),
            numeric_value: f64::NAN,
            derivation_steps: String::new(),
            success: false,
            error_message: format!("{}", e),
        }),
    }
}

/// Compute the error function with derivation steps.
pub(super) fn erf_ffi(x: f64) -> Result<super::ffi::SpecialFunctionResultFFI, String> {
    use crate::special::erf;

    let x_expr = crate::ast::Expression::Float(x);

    let result = erf(&x_expr);

    match result {
        Ok(erf_result) => {
            let steps_json = serde_json::to_string(&erf_result.derivation_steps)
                .map_err(|e| format!("Failed to serialize derivation steps: {}", e))?;

            Ok(super::ffi::SpecialFunctionResultFFI {
                value: format!("{}", erf_result.value),
                value_latex: erf_result.value.to_latex(),
                numeric_value: erf_result.numeric_value.unwrap_or(f64::NAN),
                derivation_steps: steps_json,
                success: true,
                error_message: String::new(),
            })
        }
        Err(e) => Ok(super::ffi::SpecialFunctionResultFFI {
            value: String::new(),
            value_latex: String::new(),
            numeric_value: f64::NAN,
            derivation_steps: String::new(),
            success: false,
            error_message: format!("{}", e),
        }),
    }
}

/// Compute the Beta function B(a, b) = Gamma(a)*Gamma(b) / Gamma(a+b) with derivation steps.
pub(super) fn beta_ffi(a: f64, b: f64) -> Result<super::ffi::SpecialFunctionResultFFI, String> {
    use crate::special::beta;

    let a_expr = crate::ast::Expression::Float(a);
    let b_expr = crate::ast::Expression::Float(b);

    match beta(&a_expr, &b_expr) {
        Ok(beta_result) => {
            let steps_json = serde_json::to_string(&beta_result.derivation_steps)
                .map_err(|e| format!("Failed to serialize derivation steps: {}", e))?;

            Ok(super::ffi::SpecialFunctionResultFFI {
                value: format!("{}", beta_result.value),
                value_latex: beta_result.value.to_latex(),
                numeric_value: beta_result.numeric_value.unwrap_or(f64::NAN),
                derivation_steps: steps_json,
                success: true,
                error_message: String::new(),
            })
        }
        Err(e) => Ok(super::ffi::SpecialFunctionResultFFI {
            value: String::new(),
            value_latex: String::new(),
            numeric_value: f64::NAN,
            derivation_steps: String::new(),
            success: false,
            error_message: format!("{}", e),
        }),
    }
}

/// Compute the complementary error function erfc(x) = 1 - erf(x) with derivation steps.
pub(super) fn erfc_ffi(x: f64) -> Result<super::ffi::SpecialFunctionResultFFI, String> {
    use crate::special::erfc;

    let x_expr = crate::ast::Expression::Float(x);

    match erfc(&x_expr) {
        Ok(erfc_result) => {
            let steps_json = serde_json::to_string(&erfc_result.derivation_steps)
                .map_err(|e| format!("Failed to serialize derivation steps: {}", e))?;

            Ok(super::ffi::SpecialFunctionResultFFI {
                value: format!("{}", erfc_result.value),
                value_latex: erfc_result.value.to_latex(),
                numeric_value: erfc_result.numeric_value.unwrap_or(f64::NAN),
                derivation_steps: steps_json,
                success: true,
                error_message: String::new(),
            })
        }
        Err(e) => Ok(super::ffi::SpecialFunctionResultFFI {
            value: String::new(),
            value_latex: String::new(),
            numeric_value: f64::NAN,
            derivation_steps: String::new(),
            success: false,
            error_message: format!("{}", e),
        }),
    }
}

// =============================================================================
// Fourier series operations
// =============================================================================

/// Compute the Fourier series of an expression.
///
/// Pass `period = 0.0` to use the default period of 2pi.
pub(super) fn fourier_series_ffi(
    expression: &str,
    variable: &str,
    num_terms: u32,
    period: f64,
) -> Result<super::ffi::FourierSeriesResultFFI, String> {
    use crate::ast::Variable;
    use crate::fourier::fourier_series;

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;
    let var = Variable::new(variable);
    let opt_period = if period <= 0.0 { None } else { Some(period) };

    let result = fourier_series(&expr, &var, num_terms as usize, opt_period);

    match result {
        Ok(series) => {
            let a_json = serde_json::to_string(&series.a_coefficients)
                .map_err(|e| format!("Failed to serialize a_coefficients: {}", e))?;
            let b_json = serde_json::to_string(&series.b_coefficients)
                .map_err(|e| format!("Failed to serialize b_coefficients: {}", e))?;
            Ok(super::ffi::FourierSeriesResultFFI {
                original: expression.to_string(),
                variable: variable.to_string(),
                num_terms,
                period: series.period,
                a_coefficients_json: a_json,
                b_coefficients_json: b_json,
                series: series.to_display_string(),
                series_latex: series.to_latex(),
                success: true,
                error_message: String::new(),
            })
        }
        Err(e) => Ok(super::ffi::FourierSeriesResultFFI {
            original: expression.to_string(),
            variable: variable.to_string(),
            num_terms,
            period: if period <= 0.0 {
                std::f64::consts::TAU
            } else {
                period
            },
            a_coefficients_json: String::new(),
            b_coefficients_json: String::new(),
            series: String::new(),
            series_latex: String::new(),
            success: false,
            error_message: format!("{}", e),
        }),
    }
}
