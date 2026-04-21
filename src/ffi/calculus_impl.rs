//! FFI implementation functions for calculus operations (differentiation, integration, limits, ODEs).

use crate::parser::parse_expression;

// =============================================================================
// Calculus operations
// =============================================================================

/// Differentiate an expression with respect to a variable.
pub(super) fn differentiate_ffi(
    expression: &str,
    variable: &str,
) -> Result<super::ffi::DifferentiationResultFFI, String> {
    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let derivative = expr.differentiate(variable);
    let simplified = derivative.simplify();

    Ok(super::ffi::DifferentiationResultFFI {
        original: expression.to_string(),
        variable: variable.to_string(),
        derivative: format!("{}", simplified),
        derivative_latex: simplified.to_latex(),
    })
}

/// Differentiate an expression n times.
pub(super) fn differentiate_n_ffi(
    expression: &str,
    variable: &str,
    n: u32,
) -> Result<super::ffi::DifferentiationResultFFI, String> {
    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let mut result = expr;
    for _ in 0..n {
        result = result.differentiate(variable).simplify();
    }

    Ok(super::ffi::DifferentiationResultFFI {
        original: expression.to_string(),
        variable: variable.to_string(),
        derivative: format!("{}", result),
        derivative_latex: result.to_latex(),
    })
}

/// Compute the gradient of an expression with respect to multiple variables.
pub(super) fn gradient_ffi(expression: &str, variables_json: &str) -> Result<String, String> {
    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let variables: Vec<String> = serde_json::from_str(variables_json)
        .map_err(|e| format!("Failed to parse variables JSON: {}", e))?;

    let gradient: Vec<serde_json::Value> = variables
        .iter()
        .map(|var| {
            let deriv = expr.differentiate(var).simplify();
            serde_json::json!({
                "variable": var,
                "partial_derivative": format!("{}", deriv),
                "latex": deriv.to_latex()
            })
        })
        .collect();

    serde_json::to_string(&gradient).map_err(|e| format!("Failed to serialize gradient: {}", e))
}

/// Integrate an expression with respect to a variable (indefinite integral).
pub(super) fn integrate_ffi(
    expression: &str,
    variable: &str,
) -> Result<super::ffi::IntegrationResultFFI, String> {
    use crate::integration::integrate;

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let result = integrate(&expr, variable);

    match result {
        Ok(integral) => {
            let simplified = integral.simplify();
            Ok(super::ffi::IntegrationResultFFI {
                original: expression.to_string(),
                variable: variable.to_string(),
                integral: format!("{}", simplified),
                integral_latex: simplified.to_latex(),
                success: true,
                error_message: String::new(),
            })
        }
        Err(e) => Ok(super::ffi::IntegrationResultFFI {
            original: expression.to_string(),
            variable: variable.to_string(),
            integral: String::new(),
            integral_latex: String::new(),
            success: false,
            error_message: format!("{:?}", e),
        }),
    }
}

/// Compute a definite integral.
pub(super) fn definite_integral_ffi(
    expression: &str,
    variable: &str,
    lower: f64,
    upper: f64,
) -> Result<super::ffi::DefiniteIntegralResultFFI, String> {
    use crate::ast::Expression;
    use crate::integration::definite_integral;

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let lower_expr = Expression::Float(lower);
    let upper_expr = Expression::Float(upper);

    let result = definite_integral(&expr, variable, &lower_expr, &upper_expr);

    match result {
        Ok(value) => Ok(super::ffi::DefiniteIntegralResultFFI {
            original: expression.to_string(),
            variable: variable.to_string(),
            lower_bound: lower,
            upper_bound: upper,
            value: format!("{}", value),
            value_latex: value.to_latex(),
            numeric_value: evaluate_to_f64(&value),
            success: true,
            error_message: String::new(),
        }),
        Err(e) => Ok(super::ffi::DefiniteIntegralResultFFI {
            original: expression.to_string(),
            variable: variable.to_string(),
            lower_bound: lower,
            upper_bound: upper,
            value: String::new(),
            value_latex: String::new(),
            numeric_value: f64::NAN,
            success: false,
            error_message: format!("{:?}", e),
        }),
    }
}

/// Helper to evaluate expression to f64 if possible.
pub(super) fn evaluate_to_f64(expr: &crate::ast::Expression) -> f64 {
    use crate::ast::Expression;
    match expr {
        Expression::Integer(n) => *n as f64,
        Expression::Float(f) => *f,
        Expression::Rational(r) => *r.numer() as f64 / *r.denom() as f64,
        _ => f64::NAN,
    }
}

/// Evaluate a limit.
pub(super) fn limit_ffi(
    expression: &str,
    variable: &str,
    approaches: f64,
) -> Result<super::ffi::LimitResultFFI, String> {
    use crate::limits::{limit, LimitPoint};

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let result = limit(&expr, variable, LimitPoint::Value(approaches));

    match result {
        Ok(lim_result) => {
            let (value_str, value_latex, numeric) = format_limit_result(&lim_result);
            Ok(super::ffi::LimitResultFFI {
                original: expression.to_string(),
                variable: variable.to_string(),
                approaches: format!("{}", approaches),
                value: value_str,
                value_latex,
                numeric_value: numeric,
                success: true,
                error_message: String::new(),
            })
        }
        Err(e) => Ok(super::ffi::LimitResultFFI {
            original: expression.to_string(),
            variable: variable.to_string(),
            approaches: format!("{}", approaches),
            value: String::new(),
            value_latex: String::new(),
            numeric_value: f64::NAN,
            success: false,
            error_message: format!("{:?}", e),
        }),
    }
}

/// Evaluate a limit at positive infinity.
pub(super) fn limit_infinity_ffi(
    expression: &str,
    variable: &str,
) -> Result<super::ffi::LimitResultFFI, String> {
    use crate::limits::{limit, LimitPoint};

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let result = limit(&expr, variable, LimitPoint::PositiveInfinity);

    match result {
        Ok(lim_result) => {
            let (value_str, value_latex, numeric) = format_limit_result(&lim_result);
            Ok(super::ffi::LimitResultFFI {
                original: expression.to_string(),
                variable: variable.to_string(),
                approaches: "\u{221e}".to_string(),
                value: value_str,
                value_latex,
                numeric_value: numeric,
                success: true,
                error_message: String::new(),
            })
        }
        Err(e) => Ok(super::ffi::LimitResultFFI {
            original: expression.to_string(),
            variable: variable.to_string(),
            approaches: "\u{221e}".to_string(),
            value: String::new(),
            value_latex: String::new(),
            numeric_value: f64::NAN,
            success: false,
            error_message: format!("{:?}", e),
        }),
    }
}

/// Helper to format limit result.
fn format_limit_result(result: &crate::limits::LimitResult) -> (String, String, f64) {
    use crate::limits::LimitResult;
    match result {
        LimitResult::Value(v) => (format!("{}", v), format!("{}", v), *v),
        LimitResult::PositiveInfinity => {
            ("\u{221e}".to_string(), "\\infty".to_string(), f64::INFINITY)
        }
        LimitResult::NegativeInfinity => (
            "-\u{221e}".to_string(),
            "-\\infty".to_string(),
            f64::NEG_INFINITY,
        ),
        LimitResult::Expression(expr) => (format!("{}", expr), expr.to_latex(), f64::NAN),
    }
}

// =============================================================================
// ODE solving functions
// =============================================================================

/// Build an `ODEResultFFI` success value from an `ODESolution`.
fn ode_success(
    equation: &str,
    solution: &crate::ode::ODESolution,
    ode_type: &str,
) -> super::ffi::ODEResultFFI {
    let simplified = crate::numeric::compile::decompile(&solution.general_solution).simplify();
    super::ffi::ODEResultFFI {
        equation: equation.to_string(),
        solution: format!("{}", simplified),
        solution_latex: simplified.to_latex(),
        ode_type: ode_type.to_string(),
        method_used: solution.method.clone(),
        success: true,
        error_message: String::new(),
    }
}

/// Build an `ODEResultFFI` error value.
fn ode_error(equation: &str, error: &str) -> super::ffi::ODEResultFFI {
    super::ffi::ODEResultFFI {
        equation: equation.to_string(),
        solution: String::new(),
        solution_latex: String::new(),
        ode_type: String::new(),
        method_used: String::new(),
        success: false,
        error_message: error.to_string(),
    }
}

/// Classify and solve a first-order ODE given its RHS expression string.
///
/// The `equation` parameter is the right-hand side expression of
/// `d(dependent_var)/d(independent_var) = equation`.
pub(super) fn solve_ode_ffi(
    equation: &str,
    dependent_var: &str,
    independent_var: &str,
) -> super::ffi::ODEResultFFI {
    use crate::ode::{solve_linear, solve_separable, FirstOrderODE};

    let rhs = match parse_expression(equation) {
        Ok(expr) => expr,
        Err(e) => return ode_error(equation, &format!("Parse error: {:?}", e)),
    };

    let ode = FirstOrderODE::new(dependent_var, independent_var, rhs);

    if ode.is_separable() {
        match solve_separable(&ode) {
            Ok(sol) => ode_success(equation, &sol, "separable"),
            Err(e) => ode_error(equation, &format!("Separable solve failed: {}", e)),
        }
    } else if ode.is_linear() {
        match solve_linear(&ode) {
            Ok(sol) => ode_success(equation, &sol, "linear"),
            Err(e) => ode_error(equation, &format!("Linear solve failed: {}", e)),
        }
    } else {
        ode_error(equation, "ODE is neither separable nor first-order linear")
    }
}

/// Solve a first-order ODE initial value problem.
///
/// `initial_conditions_json` must be a JSON object with numeric keys `"x0"` and `"y0"`,
/// e.g. `{"x0": 0.0, "y0": 1.0}`.
pub(super) fn solve_ode_ivp_ffi(
    equation: &str,
    dependent_var: &str,
    independent_var: &str,
    initial_conditions_json: &str,
) -> super::ffi::ODEResultFFI {
    use crate::ast::Expression;
    use crate::ode::{solve_ivp, FirstOrderODE};

    let rhs = match parse_expression(equation) {
        Ok(expr) => expr,
        Err(e) => return ode_error(equation, &format!("Parse error: {:?}", e)),
    };

    #[derive(serde::Deserialize)]
    struct Ivp {
        x0: f64,
        y0: f64,
    }

    let ivp: Ivp = match serde_json::from_str(initial_conditions_json) {
        Ok(v) => v,
        Err(e) => return ode_error(equation, &format!("Invalid initial conditions JSON: {}", e)),
    };

    let ode = FirstOrderODE::new(dependent_var, independent_var, rhs);
    let x0 = Expression::Float(ivp.x0);
    let y0 = Expression::Float(ivp.y0);

    match solve_ivp(&ode, &x0, &y0) {
        Ok(sol) => ode_success(equation, &sol, "ivp"),
        Err(e) => ode_error(equation, &format!("IVP solve failed: {}", e)),
    }
}

/// Solve a second-order constant-coefficient ODE: `a*y'' + b*y' + c*y = f(x)`.
///
/// `coefficients_json` must be a JSON array `[a, b, c]`.
/// `forcing_fn` is an optional expression string; use `""` for the homogeneous case.
pub(super) fn solve_second_order_ode_ffi(
    coefficients_json: &str,
    forcing_fn: &str,
) -> Result<super::ffi::ODEResultFFI, String> {
    use crate::ast::Expression;
    use crate::ode::{solve_second_order_homogeneous, SecondOrderODE};

    let coeffs: Vec<f64> = serde_json::from_str(coefficients_json)
        .map_err(|e| format!("Invalid coefficients JSON: {}", e))?;
    if coeffs.len() != 3 {
        return Err(format!(
            "Expected 3 coefficients [a, b, c], got {}",
            coeffs.len()
        ));
    }
    let (a, b, c) = (coeffs[0], coeffs[1], coeffs[2]);

    let forcing = if forcing_fn.is_empty() {
        Expression::Integer(0)
    } else {
        parse_expression(forcing_fn)
            .map_err(|e| format!("Parse error in forcing function: {:?}", e))?
    };

    let ode = SecondOrderODE::new("y", "x", a, b, c, forcing);
    match solve_second_order_homogeneous(&ode) {
        Ok(sol) => {
            let simplified = crate::numeric::compile::decompile(&sol.general_solution).simplify();
            Ok(super::ffi::ODEResultFFI {
                equation: coefficients_json.to_string(),
                solution: format!("{}", simplified),
                solution_latex: simplified.to_latex(),
                ode_type: "second_order".to_string(),
                method_used: sol.method.clone(),
                success: true,
                error_message: String::new(),
            })
        }
        Err(e) => Ok(ode_error(coefficients_json, &format!("{}", e))),
    }
}

/// Solve an n-th order constant-coefficient homogeneous ODE.
///
/// `coefficients_json` must be a JSON array of at least 2 floats, ordered from
/// the highest-order coefficient down to the zero-th order term.
pub(super) fn solve_higher_order_ode_ffi(
    coefficients_json: &str,
) -> Result<super::ffi::ODEResultFFI, String> {
    use crate::ode_higher::{solve_higher_order_homogeneous, HigherOrderODE};

    let coeffs: Vec<f64> = serde_json::from_str(coefficients_json)
        .map_err(|e| format!("Invalid coefficients JSON: {}", e))?;
    if coeffs.len() < 2 {
        return Err("Need at least 2 coefficients to define an ODE".to_string());
    }

    let ode = HigherOrderODE::new("y", "x", coeffs);
    match solve_higher_order_homogeneous(&ode) {
        Ok(sol) => {
            let simplified = crate::numeric::compile::decompile(&sol.general_solution).simplify();
            Ok(super::ffi::ODEResultFFI {
                equation: coefficients_json.to_string(),
                solution: format!("{}", simplified),
                solution_latex: simplified.to_latex(),
                ode_type: "higher_order".to_string(),
                method_used: sol.method.clone(),
                success: true,
                error_message: String::new(),
            })
        }
        Err(e) => Ok(ode_error(coefficients_json, &format!("{}", e))),
    }
}

/// Numerically integrate a scalar first-order ODE y' = f(x, y) using RK4.
///
/// `equation` is the RHS expression (e.g. `"y"` for y' = y).
/// `variable` is the dependent variable name.
/// Returns a JSON string containing the trajectory as an array of `[x, y]` pairs.
pub(super) fn rk4_solve_ffi(
    equation: &str,
    variable: &str,
    x0: f64,
    y0: f64,
    x_end: f64,
    steps: u32,
) -> Result<String, String> {
    use crate::runge_kutta::{rk4_solve, Rk4Config};
    use std::collections::HashMap;

    let expr = parse_expression(equation).map_err(|e| format!("Parse error: {:?}", e))?;
    let var = variable.to_string();

    let f = move |x: f64, y: f64| -> f64 {
        let mut vars = HashMap::new();
        vars.insert(var.clone(), y);
        vars.insert("x".to_string(), x);
        expr.evaluate(&vars).unwrap_or(f64::NAN)
    };

    let config = Rk4Config::new(x0, y0, x_end, steps as usize);
    let sol = rk4_solve(f, config).map_err(|e| format!("RK4 error: {}", e))?;

    serde_json::to_string(&sol.trajectory).map_err(|e| format!("Serialization error: {}", e))
}
