//! FFI exports for Swift interoperability using swift-bridge.
//!
//! This module provides C-compatible bindings for use from Swift/iOS applications
//! via the [swift-bridge](https://github.com/chinedufn/swift-bridge) framework.
//! All functions and types in this module are automatically exposed to Swift code.
//!
//! # Architecture
//!
//! The FFI layer follows these principles:
//! - **Type Safety**: Uses `swift_bridge::bridge` macro for automatic type conversion
//! - **Error Handling**: Returns `Result<T, String>` for fallible operations
//! - **Memory Safety**: All string data is copied; no raw pointers exposed
//! - **JSON Serialization**: Complex data structures passed as JSON strings
//!
//! # Swift Integration
//!
//! After building the Rust library, swift-bridge generates Swift wrapper code that
//! can be imported directly:
//!
//! ```swift
//! import Thales
//!
//! // Parse and solve an equation
//! do {
//!     let solution = try solve_equation_ffi("2*x + 5 = 15", "x")
//!     print("Solution: \(solution)")
//! } catch {
//!     print("Error: \(error)")
//! }
//! ```

// Note: Doc comments inside #[swift_bridge::bridge] modules are not supported
// by swift-bridge 0.1.x. See src/bridge.rs for the interface documentation.

#[swift_bridge::bridge]
mod ffi {
    #[swift_bridge(swift_repr = "struct")]
    pub struct ResolutionPathFFI {
        pub initial_expr: String,
        pub steps_json: String,
        pub result_expr: String,
        pub success: bool,
    }

    #[swift_bridge(swift_repr = "struct")]
    pub struct CartesianCoords2D {
        pub x: f64,
        pub y: f64,
    }

    #[swift_bridge(swift_repr = "struct")]
    pub struct CartesianCoords3D {
        pub x: f64,
        pub y: f64,
        pub z: f64,
    }

    #[swift_bridge(swift_repr = "struct")]
    pub struct PolarCoords {
        pub r: f64,
        pub theta: f64,
    }

    #[swift_bridge(swift_repr = "struct")]
    pub struct SphericalCoords {
        pub r: f64,
        pub theta: f64,
        pub phi: f64,
    }

    #[swift_bridge(swift_repr = "struct")]
    pub struct ComplexNumber {
        pub real: f64,
        pub imaginary: f64,
    }

    #[swift_bridge(swift_repr = "struct")]
    pub struct DifferentiationResultFFI {
        pub original: String,
        pub variable: String,
        pub derivative: String,
        pub derivative_latex: String,
    }

    #[swift_bridge(swift_repr = "struct")]
    pub struct IntegrationResultFFI {
        pub original: String,
        pub variable: String,
        pub integral: String,
        pub integral_latex: String,
        pub success: bool,
        pub error_message: String,
    }

    #[swift_bridge(swift_repr = "struct")]
    pub struct DefiniteIntegralResultFFI {
        pub original: String,
        pub variable: String,
        pub lower_bound: f64,
        pub upper_bound: f64,
        pub value: String,
        pub value_latex: String,
        pub numeric_value: f64,
        pub success: bool,
        pub error_message: String,
    }

    #[swift_bridge(swift_repr = "struct")]
    pub struct LimitResultFFI {
        pub original: String,
        pub variable: String,
        pub approaches: String,
        pub value: String,
        pub value_latex: String,
        pub numeric_value: f64,
        pub success: bool,
        pub error_message: String,
    }

    #[swift_bridge(swift_repr = "struct")]
    pub struct EvaluationResultFFI {
        pub original: String,
        pub value: f64,
        pub success: bool,
        pub error_message: String,
    }

    #[swift_bridge(swift_repr = "struct")]
    pub struct SimplificationResultFFI {
        pub original: String,
        pub simplified: String,
        pub simplified_latex: String,
    }

    #[swift_bridge(swift_repr = "struct")]
    pub struct TaylorSeriesResultFFI {
        pub original: String,
        pub variable: String,
        pub center: f64,
        pub order: u32,
        pub series: String,
        pub series_latex: String,
        pub success: bool,
        pub error_message: String,
    }

    #[swift_bridge(swift_repr = "struct")]
    pub struct LaurentSeriesResultFFI {
        pub original: String,
        pub variable: String,
        pub center: f64,
        pub neg_order: u32,
        pub pos_order: u32,
        pub series: String,
        pub series_latex: String,
        pub success: bool,
        pub error_message: String,
    }

    #[swift_bridge(swift_repr = "struct")]
    pub struct AsymptoticSeriesResultFFI {
        pub original: String,
        pub variable: String,
        pub direction: String,
        pub num_terms: u32,
        pub series: String,
        pub series_latex: String,
        pub success: bool,
        pub error_message: String,
    }

    #[swift_bridge(swift_repr = "struct")]
    pub struct SpecialFunctionResultFFI {
        pub value: String,
        pub value_latex: String,
        pub numeric_value: f64,
        pub derivation_steps: String,
        pub success: bool,
        pub error_message: String,
    }

    #[swift_bridge(swift_repr = "struct")]
    pub struct ODEResultFFI {
        pub equation: String,
        pub solution: String,
        pub solution_latex: String,
        pub ode_type: String,
        pub method_used: String,
        pub success: bool,
        pub error_message: String,
    }

    #[swift_bridge(swift_repr = "struct")]
    pub struct PrecisionEvaluationResultFFI {
        pub original: String,
        pub value: f64,
        pub value_string: String,
        pub precision_mode: String,
        pub rounding_mode: String,
        pub success: bool,
        pub error_message: String,
    }

    #[swift_bridge(swift_repr = "struct")]
    pub struct FourierSeriesResultFFI {
        pub original: String,
        pub variable: String,
        pub num_terms: u32,
        pub period: f64,
        pub a_coefficients_json: String,
        pub b_coefficients_json: String,
        pub series: String,
        pub series_latex: String,
        pub success: bool,
        pub error_message: String,
    }

    extern "Rust" {
        fn parse_equation_ffi(input: &str) -> Result<String, String>;
        fn parse_expression_ffi(input: &str) -> Result<String, String>;
    }

    extern "Rust" {
        fn solve_ode_ffi(
            equation: &str,
            dependent_var: &str,
            independent_var: &str,
        ) -> ODEResultFFI;
        fn solve_ode_ivp_ffi(
            equation: &str,
            dependent_var: &str,
            independent_var: &str,
            initial_conditions_json: &str,
        ) -> ODEResultFFI;
        fn solve_second_order_ode_ffi(
            coefficients_json: &str,
            forcing_fn: &str,
        ) -> Result<ODEResultFFI, String>;
        fn solve_higher_order_ode_ffi(coefficients_json: &str) -> Result<ODEResultFFI, String>;
        fn rk4_solve_ffi(
            equation: &str,
            variable: &str,
            x0: f64,
            y0: f64,
            x_end: f64,
            steps: u32,
        ) -> Result<String, String>;
    }

    extern "Rust" {
        fn solve_equation_ffi(equation: &str, variable: &str) -> Result<String, String>;
        fn solve_with_values_ffi(
            equation: &str,
            variable: &str,
            known_values_json: &str,
        ) -> Result<ResolutionPathFFI, String>;
        fn solve_numerically_ffi(
            equation: &str,
            variable: &str,
            initial_guess: f64,
        ) -> Result<f64, String>;
    }

    extern "Rust" {
        fn cartesian_to_polar_ffi(x: f64, y: f64) -> PolarCoords;
        fn polar_to_cartesian_ffi(r: f64, theta: f64) -> CartesianCoords2D;
        fn cartesian_to_spherical_ffi(x: f64, y: f64, z: f64) -> SphericalCoords;
        fn spherical_to_cartesian_ffi(r: f64, theta: f64, phi: f64) -> CartesianCoords3D;
    }

    extern "Rust" {
        fn complex_add_ffi(a_re: f64, a_im: f64, b_re: f64, b_im: f64) -> ComplexNumber;
        fn complex_multiply_ffi(a_re: f64, a_im: f64, b_re: f64, b_im: f64) -> ComplexNumber;
        fn complex_to_polar_ffi(re: f64, im: f64) -> PolarCoords;
        fn complex_power_ffi(re: f64, im: f64, n: f64) -> ComplexNumber;
    }

    extern "Rust" {
        fn parse_latex_ffi(input: &str) -> Result<String, String>;
        fn parse_latex_to_latex_ffi(input: &str) -> Result<String, String>;
        fn to_latex_ffi(expression: &str) -> Result<String, String>;
    }

    extern "Rust" {
        fn differentiate_ffi(
            expression: &str,
            variable: &str,
        ) -> Result<DifferentiationResultFFI, String>;
        fn differentiate_n_ffi(
            expression: &str,
            variable: &str,
            n: u32,
        ) -> Result<DifferentiationResultFFI, String>;
        fn gradient_ffi(expression: &str, variables_json: &str) -> Result<String, String>;
        fn integrate_ffi(expression: &str, variable: &str) -> Result<IntegrationResultFFI, String>;
        fn definite_integral_ffi(
            expression: &str,
            variable: &str,
            lower: f64,
            upper: f64,
        ) -> Result<DefiniteIntegralResultFFI, String>;
        fn limit_ffi(
            expression: &str,
            variable: &str,
            approaches: f64,
        ) -> Result<LimitResultFFI, String>;
        fn limit_infinity_ffi(expression: &str, variable: &str) -> Result<LimitResultFFI, String>;
    }

    extern "Rust" {
        fn evaluate_ffi(expression: &str, values_json: &str)
            -> Result<EvaluationResultFFI, String>;
        fn simplify_ffi(expression: &str) -> Result<SimplificationResultFFI, String>;
        fn simplify_trig_ffi(expression: &str) -> Result<SimplificationResultFFI, String>;
        fn simplify_trig_with_steps_ffi(expression: &str) -> Result<String, String>;
    }

    extern "Rust" {
        fn solve_system_ffi(equations_json: &str) -> Result<String, String>;
        fn solve_inequality_ffi(inequality: &str, variable: &str) -> Result<String, String>;
        fn partial_fractions_ffi(
            numerator: &str,
            denominator: &str,
            variable: &str,
        ) -> Result<String, String>;
        fn solve_equation_system_ffi(
            equations_json: &str,
            known_values_json: &str,
            targets_json: &str,
        ) -> Result<String, String>;
    }

    extern "Rust" {
        fn taylor_series_ffi(
            expression: &str,
            variable: &str,
            center: f64,
            order: u32,
        ) -> Result<TaylorSeriesResultFFI, String>;
        fn maclaurin_series_ffi(
            expression: &str,
            variable: &str,
            order: u32,
        ) -> Result<TaylorSeriesResultFFI, String>;
        fn laurent_series_ffi(
            expression: &str,
            variable: &str,
            center: f64,
            neg_order: u32,
            pos_order: u32,
        ) -> Result<LaurentSeriesResultFFI, String>;
        fn asymptotic_series_ffi(
            expression: &str,
            variable: &str,
            direction: &str,
            num_terms: u32,
        ) -> Result<AsymptoticSeriesResultFFI, String>;
        fn gamma_ffi(x: f64) -> Result<SpecialFunctionResultFFI, String>;
        fn erf_ffi(x: f64) -> Result<SpecialFunctionResultFFI, String>;
        fn beta_ffi(a: f64, b: f64) -> Result<SpecialFunctionResultFFI, String>;
        fn erfc_ffi(x: f64) -> Result<SpecialFunctionResultFFI, String>;
    }

    extern "Rust" {
        fn fourier_series_ffi(
            expression: &str,
            variable: &str,
            num_terms: u32,
            period: f64,
        ) -> Result<FourierSeriesResultFFI, String>;
    }

    extern "Rust" {
        fn evaluate_with_precision_ffi(
            expression: &str,
            values_json: &str,
            mode: &str,
            precision: u32,
            rounding: &str,
        ) -> Result<PrecisionEvaluationResultFFI, String>;
        fn optimize_for_manual_computation_ffi(expression: &str) -> Result<String, String>;
        fn small_angle_approximation_ffi(
            expression: &str,
            variable: &str,
            threshold: f64,
        ) -> Result<String, String>;
    }
}

// =============================================================================
// Implementation of FFI functions
// =============================================================================

use crate::parser::{parse_equation, parse_expression};
use crate::transforms::{Cartesian2D, Cartesian3D, ComplexOps, Polar, Spherical};
use num_complex::Complex64;

/// Parse equation and return string representation.
fn parse_equation_ffi(input: &str) -> Result<String, String> {
    parse_equation(input)
        .map(|eq| format!("{}", eq))
        .map_err(|e| format!("Parse error: {:?}", e))
}

/// Parse expression and return string representation.
fn parse_expression_ffi(input: &str) -> Result<String, String> {
    parse_expression(input)
        .map(|expr| format!("{}", expr))
        .map_err(|e| format!("Parse error: {:?}", e))
}

/// Solve equation symbolically.
fn solve_equation_ffi(equation: &str, variable: &str) -> Result<String, String> {
    use crate::solver::solve_for;
    use std::collections::HashMap;

    let parsed_equation =
        parse_equation(equation).map_err(|e| format!("Failed to parse equation: {:?}", e))?;

    let known_values = HashMap::new();
    let resolution_path = solve_for(&parsed_equation, variable, &known_values)
        .map_err(|e| format!("Failed to solve equation: {:?}", e))?;

    Ok(format!("{:?}", resolution_path.result))
}

/// Solve equation numerically.
fn solve_numerically_ffi(
    equation: &str,
    variable: &str,
    initial_guess: f64,
) -> Result<f64, String> {
    use crate::ast::Variable;
    use crate::numerical::{NumericalConfig, SmartNumericalSolver};

    let parsed_equation =
        parse_equation(equation).map_err(|e| format!("Failed to parse equation: {:?}", e))?;

    let mut config = NumericalConfig::default();
    config.initial_guess = Some(initial_guess);
    let solver = SmartNumericalSolver::new(config);

    let target_var = Variable::new(variable);
    let (solution, _path) = solver
        .solve(&parsed_equation, &target_var)
        .map_err(|e| format!("Numerical solving failed: {:?}", e))?;

    if !solution.converged {
        return Err(format!(
            "Did not converge after {} iterations (residual: {})",
            solution.iterations, solution.residual
        ));
    }

    Ok(solution.value)
}

/// Convert Cartesian to polar coordinates.
fn cartesian_to_polar_ffi(x: f64, y: f64) -> ffi::PolarCoords {
    let cart = Cartesian2D::new(x, y);
    let polar = cart.to_polar();
    ffi::PolarCoords {
        r: polar.r,
        theta: polar.theta,
    }
}

/// Convert polar to Cartesian coordinates.
fn polar_to_cartesian_ffi(r: f64, theta: f64) -> ffi::CartesianCoords2D {
    let polar = Polar::new(r, theta);
    let cart = polar.to_cartesian();
    ffi::CartesianCoords2D {
        x: cart.x,
        y: cart.y,
    }
}

/// Convert 3D Cartesian to spherical coordinates.
fn cartesian_to_spherical_ffi(x: f64, y: f64, z: f64) -> ffi::SphericalCoords {
    let cart = Cartesian3D::new(x, y, z);
    let spherical = cart.to_spherical();
    ffi::SphericalCoords {
        r: spherical.r,
        theta: spherical.theta,
        phi: spherical.phi,
    }
}

/// Convert spherical to 3D Cartesian coordinates.
fn spherical_to_cartesian_ffi(r: f64, theta: f64, phi: f64) -> ffi::CartesianCoords3D {
    let spherical = Spherical::new(r, theta, phi);
    let cart = spherical.to_cartesian();
    ffi::CartesianCoords3D {
        x: cart.x,
        y: cart.y,
        z: cart.z,
    }
}

/// Add two complex numbers.
fn complex_add_ffi(a_re: f64, a_im: f64, b_re: f64, b_im: f64) -> ffi::ComplexNumber {
    let a = Complex64::new(a_re, a_im);
    let b = Complex64::new(b_re, b_im);
    let result = a + b;
    ffi::ComplexNumber {
        real: result.re,
        imaginary: result.im,
    }
}

/// Multiply two complex numbers.
fn complex_multiply_ffi(a_re: f64, a_im: f64, b_re: f64, b_im: f64) -> ffi::ComplexNumber {
    let a = Complex64::new(a_re, a_im);
    let b = Complex64::new(b_re, b_im);
    let result = a * b;
    ffi::ComplexNumber {
        real: result.re,
        imaginary: result.im,
    }
}

/// Convert complex number to polar form.
fn complex_to_polar_ffi(re: f64, im: f64) -> ffi::PolarCoords {
    let c = Complex64::new(re, im);
    let polar = ComplexOps::to_polar(c);
    ffi::PolarCoords {
        r: polar.r,
        theta: polar.theta,
    }
}

/// Raise complex number to a power using De Moivre's theorem.
fn complex_power_ffi(re: f64, im: f64, n: f64) -> ffi::ComplexNumber {
    let c = Complex64::new(re, im);
    let result = ComplexOps::de_moivre(c, n);
    ffi::ComplexNumber {
        real: result.re,
        imaginary: result.im,
    }
}

/// Solve equation with known values and return full resolution path.
fn solve_with_values_ffi(
    equation: &str,
    variable: &str,
    known_values_json: &str,
) -> Result<ffi::ResolutionPathFFI, String> {
    use crate::solver::solve_for;
    use std::collections::HashMap;

    let parsed_equation =
        parse_equation(equation).map_err(|e| format!("Failed to parse equation: {:?}", e))?;

    let known_values: HashMap<String, f64> = if known_values_json.is_empty() {
        HashMap::new()
    } else {
        serde_json::from_str(known_values_json)
            .map_err(|e| format!("Failed to parse known values JSON: {}", e))?
    };

    let resolution_path = solve_for(&parsed_equation, variable, &known_values)
        .map_err(|e| format!("Failed to solve equation: {:?}", e))?;

    let steps: Vec<serde_json::Value> = resolution_path
        .steps
        .iter()
        .map(|step| {
            serde_json::json!({
                "operation": step.operation.describe(),
                "explanation": step.explanation,
                "result": format!("{:?}", step.result)
            })
        })
        .collect();

    let steps_json =
        serde_json::to_string(&steps).map_err(|e| format!("Failed to serialize steps: {}", e))?;

    Ok(ffi::ResolutionPathFFI {
        initial_expr: format!("{:?}", resolution_path.initial),
        steps_json,
        result_expr: format!("{:?}", resolution_path.result),
        success: true,
    })
}

// =============================================================================
// LaTeX functions
// =============================================================================

/// Parse LaTeX expression and return human-readable string representation.
fn parse_latex_ffi(input: &str) -> Result<String, String> {
    use crate::latex::parse_latex;
    parse_latex(input)
        .map(|expr| format!("{}", expr))
        .map_err(|e| format!("LaTeX parse error: {:?}", e))
}

/// Parse LaTeX expression and return as LaTeX output.
fn parse_latex_to_latex_ffi(input: &str) -> Result<String, String> {
    use crate::latex::parse_latex;
    parse_latex(input)
        .map(|expr| expr.to_latex())
        .map_err(|e| format!("LaTeX parse error: {:?}", e))
}

/// Get the LaTeX representation of an expression.
fn to_latex_ffi(expression: &str) -> Result<String, String> {
    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;
    Ok(expr.to_latex())
}

// =============================================================================
// Calculus operations
// =============================================================================

/// Differentiate an expression with respect to a variable.
fn differentiate_ffi(
    expression: &str,
    variable: &str,
) -> Result<ffi::DifferentiationResultFFI, String> {
    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let derivative = expr.differentiate(variable);
    let simplified = derivative.simplify();

    Ok(ffi::DifferentiationResultFFI {
        original: expression.to_string(),
        variable: variable.to_string(),
        derivative: format!("{}", simplified),
        derivative_latex: simplified.to_latex(),
    })
}

/// Differentiate an expression n times.
fn differentiate_n_ffi(
    expression: &str,
    variable: &str,
    n: u32,
) -> Result<ffi::DifferentiationResultFFI, String> {
    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let mut result = expr;
    for _ in 0..n {
        result = result.differentiate(variable).simplify();
    }

    Ok(ffi::DifferentiationResultFFI {
        original: expression.to_string(),
        variable: variable.to_string(),
        derivative: format!("{}", result),
        derivative_latex: result.to_latex(),
    })
}

/// Compute the gradient of an expression with respect to multiple variables.
fn gradient_ffi(expression: &str, variables_json: &str) -> Result<String, String> {
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
fn integrate_ffi(expression: &str, variable: &str) -> Result<ffi::IntegrationResultFFI, String> {
    use crate::integration::integrate;

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let result = integrate(&expr, variable);

    match result {
        Ok(integral) => {
            let simplified = integral.simplify();
            Ok(ffi::IntegrationResultFFI {
                original: expression.to_string(),
                variable: variable.to_string(),
                integral: format!("{}", simplified),
                integral_latex: simplified.to_latex(),
                success: true,
                error_message: String::new(),
            })
        }
        Err(e) => Ok(ffi::IntegrationResultFFI {
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
fn definite_integral_ffi(
    expression: &str,
    variable: &str,
    lower: f64,
    upper: f64,
) -> Result<ffi::DefiniteIntegralResultFFI, String> {
    use crate::ast::Expression;
    use crate::integration::definite_integral;

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let lower_expr = Expression::Float(lower);
    let upper_expr = Expression::Float(upper);

    let result = definite_integral(&expr, variable, &lower_expr, &upper_expr);

    match result {
        Ok(value) => Ok(ffi::DefiniteIntegralResultFFI {
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
        Err(e) => Ok(ffi::DefiniteIntegralResultFFI {
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
fn evaluate_to_f64(expr: &crate::ast::Expression) -> f64 {
    use crate::ast::Expression;
    match expr {
        Expression::Integer(n) => *n as f64,
        Expression::Float(f) => *f,
        Expression::Rational(r) => *r.numer() as f64 / *r.denom() as f64,
        _ => f64::NAN,
    }
}

/// Evaluate a limit.
fn limit_ffi(
    expression: &str,
    variable: &str,
    approaches: f64,
) -> Result<ffi::LimitResultFFI, String> {
    use crate::limits::{limit, LimitPoint};

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let result = limit(&expr, variable, LimitPoint::Value(approaches));

    match result {
        Ok(lim_result) => {
            let (value_str, value_latex, numeric) = format_limit_result(&lim_result);
            Ok(ffi::LimitResultFFI {
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
        Err(e) => Ok(ffi::LimitResultFFI {
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
fn limit_infinity_ffi(expression: &str, variable: &str) -> Result<ffi::LimitResultFFI, String> {
    use crate::limits::{limit, LimitPoint};

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let result = limit(&expr, variable, LimitPoint::PositiveInfinity);

    match result {
        Ok(lim_result) => {
            let (value_str, value_latex, numeric) = format_limit_result(&lim_result);
            Ok(ffi::LimitResultFFI {
                original: expression.to_string(),
                variable: variable.to_string(),
                approaches: "∞".to_string(),
                value: value_str,
                value_latex,
                numeric_value: numeric,
                success: true,
                error_message: String::new(),
            })
        }
        Err(e) => Ok(ffi::LimitResultFFI {
            original: expression.to_string(),
            variable: variable.to_string(),
            approaches: "∞".to_string(),
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
        LimitResult::PositiveInfinity => ("∞".to_string(), "\\infty".to_string(), f64::INFINITY),
        LimitResult::NegativeInfinity => {
            ("-∞".to_string(), "-\\infty".to_string(), f64::NEG_INFINITY)
        }
        LimitResult::Expression(expr) => (format!("{}", expr), expr.to_latex(), f64::NAN),
    }
}

// =============================================================================
// Expression evaluation and simplification
// =============================================================================

/// Evaluate an expression with given variable values.
///
/// Delegates directly to [`Expression::evaluate`] from the core AST, ensuring
/// full parity with the core evaluator (Log2, Log10, Cbrt, Atan2, Sign, Min,
/// Max, Pow, and correct domain handling for Ln/Log).
fn evaluate_ffi(expression: &str, values_json: &str) -> Result<ffi::EvaluationResultFFI, String> {
    use std::collections::HashMap;

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let values: HashMap<String, f64> = serde_json::from_str(values_json)
        .map_err(|e| format!("Failed to parse values JSON: {}", e))?;

    match expr.evaluate(&values) {
        Some(value) => Ok(ffi::EvaluationResultFFI {
            original: expression.to_string(),
            value,
            success: true,
            error_message: String::new(),
        }),
        None => Ok(ffi::EvaluationResultFFI {
            original: expression.to_string(),
            value: f64::NAN,
            success: false,
            error_message:
                "Cannot evaluate expression (may contain undefined variables or operations)"
                    .to_string(),
        }),
    }
}

/// Simplify an expression.
fn simplify_ffi(expression: &str) -> Result<ffi::SimplificationResultFFI, String> {
    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let simplified = expr.simplify();

    Ok(ffi::SimplificationResultFFI {
        original: expression.to_string(),
        simplified: format!("{}", simplified),
        simplified_latex: simplified.to_latex(),
    })
}

/// Simplify trigonometric expression.
fn simplify_trig_ffi(expression: &str) -> Result<ffi::SimplificationResultFFI, String> {
    use crate::trigonometric::simplify_trig;

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let simplified = simplify_trig(&expr);

    Ok(ffi::SimplificationResultFFI {
        original: expression.to_string(),
        simplified: format!("{}", simplified),
        simplified_latex: simplified.to_latex(),
    })
}

/// Simplify trigonometric expression with steps.
fn simplify_trig_with_steps_ffi(expression: &str) -> Result<String, String> {
    use crate::trigonometric::simplify_trig_with_steps;

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let (simplified, steps) = simplify_trig_with_steps(&expr);

    let result = serde_json::json!({
        "original": expression,
        "simplified": format!("{}", simplified),
        "simplified_latex": simplified.to_latex(),
        "steps": steps
    });

    serde_json::to_string(&result).map_err(|e| format!("Failed to serialize result: {}", e))
}

// =============================================================================
// Advanced solving operations
// =============================================================================

/// Solve a system of linear equations.
fn solve_system_ffi(equations_json: &str) -> Result<String, String> {
    use crate::ast::Variable;
    use crate::solver::SystemSolver;

    let equations: Vec<String> = serde_json::from_str(equations_json)
        .map_err(|e| format!("Failed to parse equations JSON: {}", e))?;

    let mut parsed_equations = Vec::new();
    for eq_str in &equations {
        let eq = parse_equation(eq_str)
            .map_err(|e| format!("Failed to parse equation '{}': {:?}", eq_str, e))?;
        parsed_equations.push(eq);
    }

    // Extract variables from equations
    let mut vars = std::collections::HashSet::new();
    for eq in &parsed_equations {
        collect_variables(&eq.left, &mut vars);
        collect_variables(&eq.right, &mut vars);
    }
    let variables: Vec<Variable> = vars.into_iter().map(|s| Variable::new(&s)).collect();

    let solver = SystemSolver::new();

    match solver.solve_system(&parsed_equations, &variables) {
        Ok(solutions) => {
            let result: std::collections::HashMap<String, String> = solutions
                .iter()
                .map(|(var, sol)| (var.name.clone(), format!("{:?}", sol)))
                .collect();
            serde_json::to_string(&result).map_err(|e| format!("Failed to serialize result: {}", e))
        }
        Err(e) => Err(format!("Failed to solve system: {:?}", e)),
    }
}

/// Helper to collect variable names from expression.
fn collect_variables(expr: &crate::ast::Expression, vars: &mut std::collections::HashSet<String>) {
    use crate::ast::Expression;
    match expr {
        Expression::Variable(v) => {
            vars.insert(v.name.clone());
        }
        Expression::Unary(_, inner) => collect_variables(inner, vars),
        Expression::Binary(_, left, right) => {
            collect_variables(left, vars);
            collect_variables(right, vars);
        }
        Expression::Power(base, exp) => {
            collect_variables(base, vars);
            collect_variables(exp, vars);
        }
        Expression::Function(_, args) => {
            for arg in args {
                collect_variables(arg, vars);
            }
        }
        _ => {}
    }
}

/// Solve an inequality.
fn solve_inequality_ffi(inequality: &str, variable: &str) -> Result<String, String> {
    use crate::inequality::{solve_inequality, Inequality};

    // Parse inequality (expects format like "expr < value" or "expr > value")
    let parts: Vec<&str> = if inequality.contains("<=") {
        vec![
            &inequality[..inequality.find("<=").unwrap()],
            &inequality[inequality.find("<=").unwrap() + 2..],
            "<=",
        ]
    } else if inequality.contains(">=") {
        vec![
            &inequality[..inequality.find(">=").unwrap()],
            &inequality[inequality.find(">=").unwrap() + 2..],
            ">=",
        ]
    } else if inequality.contains('<') {
        vec![
            &inequality[..inequality.find('<').unwrap()],
            &inequality[inequality.find('<').unwrap() + 1..],
            "<",
        ]
    } else if inequality.contains('>') {
        vec![
            &inequality[..inequality.find('>').unwrap()],
            &inequality[inequality.find('>').unwrap() + 1..],
            ">",
        ]
    } else {
        return Err("Invalid inequality format. Use <, >, <=, or >=".to_string());
    };

    let left = parse_expression(parts[0].trim())
        .map_err(|e| format!("Parse error in left side: {:?}", e))?;
    let right = parse_expression(parts[1].trim())
        .map_err(|e| format!("Parse error in right side: {:?}", e))?;

    let ineq = match parts[2] {
        "<" => Inequality::LessThan(left, right),
        ">" => Inequality::GreaterThan(left, right),
        "<=" => Inequality::LessEqual(left, right),
        ">=" => Inequality::GreaterEqual(left, right),
        _ => return Err("Invalid operator".to_string()),
    };

    match solve_inequality(&ineq, variable) {
        Ok(solution) => Ok(format!("{:?}", solution)),
        Err(e) => Err(format!("Failed to solve inequality: {:?}", e)),
    }
}

/// Partial fraction decomposition.
fn partial_fractions_ffi(
    numerator: &str,
    denominator: &str,
    variable: &str,
) -> Result<String, String> {
    use crate::ast::Variable;
    use crate::partial_fractions::decompose;

    let num =
        parse_expression(numerator).map_err(|e| format!("Parse error in numerator: {:?}", e))?;
    let denom = parse_expression(denominator)
        .map_err(|e| format!("Parse error in denominator: {:?}", e))?;

    match decompose(&num, &denom, &Variable::new(variable)) {
        Ok(result) => {
            let expr = result.to_expression();
            let output = serde_json::json!({
                "original_numerator": numerator,
                "original_denominator": denominator,
                "decomposition": format!("{}", expr),
                "decomposition_latex": expr.to_latex(),
                "terms_count": result.terms.len(),
                "steps": result.steps
            });
            serde_json::to_string(&output).map_err(|e| format!("Failed to serialize result: {}", e))
        }
        Err(e) => Err(format!("Decomposition failed: {:?}", e)),
    }
}

/// Solve a multi-equation system.
fn solve_equation_system_ffi(
    equations_json: &str,
    known_values_json: &str,
    targets_json: &str,
) -> Result<String, String> {
    use crate::equation_system::{EquationSystem, MultiEquationSolver, SystemContext};
    use std::collections::HashMap;

    // Parse equations JSON: {"id": "equation_str", ...}
    let equations_map: HashMap<String, String> = serde_json::from_str(equations_json)
        .map_err(|e| format!("Failed to parse equations JSON: {}", e))?;

    // Parse known values JSON: {"var": value, ...}
    let known_values: HashMap<String, f64> = serde_json::from_str(known_values_json)
        .map_err(|e| format!("Failed to parse known values JSON: {}", e))?;

    // Parse targets JSON: ["var1", "var2", ...]
    let targets: Vec<String> = serde_json::from_str(targets_json)
        .map_err(|e| format!("Failed to parse targets JSON: {}", e))?;

    // Build the equation system
    let mut system = EquationSystem::new();
    for (id, eq_str) in equations_map {
        let equation = parse_equation(&eq_str)
            .map_err(|e| format!("Failed to parse equation '{}': {:?}", id, e))?;
        system.add_equation(id, equation);
    }

    // Build the context
    let mut context = SystemContext::new();
    for (var, val) in known_values {
        context = context.with_known_value(var, val);
    }
    for target in targets {
        context = context.with_target(target);
    }

    // Solve the system
    let solver = MultiEquationSolver::new();
    let solution = solver
        .solve(&system, &context)
        .map_err(|e| format!("Failed to solve system: {}", e))?;

    // Build the result JSON
    let mut solutions_map: HashMap<String, serde_json::Value> = HashMap::new();
    for (var, val) in &solution.solutions {
        if let Some(num) = val.as_numeric() {
            solutions_map.insert(var.clone(), serde_json::json!(num));
        } else {
            solutions_map.insert(
                var.clone(),
                serde_json::json!(format!("{}", val.to_expression())),
            );
        }
    }

    // Build step descriptions
    let steps: Vec<serde_json::Value> = solution
        .resolution_path
        .steps
        .iter()
        .map(|step| {
            serde_json::json!({
                "step_number": step.step_number,
                "equation_id": step.equation_id,
                "operation": format!("{}", step.operation),
                "explanation": step.explanation
            })
        })
        .collect();

    let output = serde_json::json!({
        "solutions": solutions_map,
        "steps": steps,
        "unsolved": solution.unsolved,
        "warnings": solution.warnings,
        "is_complete": solution.is_complete()
    });

    serde_json::to_string(&output).map_err(|e| format!("Failed to serialize result: {}", e))
}

// =============================================================================
// Series expansion operations
// =============================================================================

/// Compute Taylor series expansion around a given center point.
fn taylor_series_ffi(
    expression: &str,
    variable: &str,
    center: f64,
    order: u32,
) -> Result<ffi::TaylorSeriesResultFFI, String> {
    use crate::ast::Variable;
    use crate::series::taylor;

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let var = Variable::new(variable);
    let center_expr = crate::ast::Expression::Float(center);

    let result = taylor(&expr, &var, &center_expr, order);

    match result {
        Ok(series) => {
            let series_expr = series.to_expression();
            Ok(ffi::TaylorSeriesResultFFI {
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
        Err(e) => Ok(ffi::TaylorSeriesResultFFI {
            original: expression.to_string(),
            variable: variable.to_string(),
            center,
            order,
            series: String::new(),
            series_latex: String::new(),
            success: false,
            error_message: format!("{}", e),
        }),
    }
}

/// Compute Maclaurin series expansion (Taylor series centered at 0).
fn maclaurin_series_ffi(
    expression: &str,
    variable: &str,
    order: u32,
) -> Result<ffi::TaylorSeriesResultFFI, String> {
    use crate::ast::Variable;
    use crate::series::maclaurin;

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let var = Variable::new(variable);

    let result = maclaurin(&expr, &var, order);

    match result {
        Ok(series) => {
            let series_expr = series.to_expression();
            Ok(ffi::TaylorSeriesResultFFI {
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
        Err(e) => Ok(ffi::TaylorSeriesResultFFI {
            original: expression.to_string(),
            variable: variable.to_string(),
            center: 0.0,
            order,
            series: String::new(),
            series_latex: String::new(),
            success: false,
            error_message: format!("{}", e),
        }),
    }
}

/// Compute Laurent series expansion around a given center point.
fn laurent_series_ffi(
    expression: &str,
    variable: &str,
    center: f64,
    neg_order: u32,
    pos_order: u32,
) -> Result<ffi::LaurentSeriesResultFFI, String> {
    use crate::ast::Variable;
    use crate::series::laurent;

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;
    let var = Variable::new(variable);
    let center_expr = crate::ast::Expression::Float(center);

    let result = laurent(&expr, &var, &center_expr, neg_order, pos_order);

    match result {
        Ok(series) => Ok(ffi::LaurentSeriesResultFFI {
            original: expression.to_string(),
            variable: variable.to_string(),
            center,
            neg_order,
            pos_order,
            series: format!("{}", series),
            series_latex: series.to_latex(),
            success: true,
            error_message: String::new(),
        }),
        Err(e) => Ok(ffi::LaurentSeriesResultFFI {
            original: expression.to_string(),
            variable: variable.to_string(),
            center,
            neg_order,
            pos_order,
            series: String::new(),
            series_latex: String::new(),
            success: false,
            error_message: format!("{}", e),
        }),
    }
}

/// Compute asymptotic series expansion of an expression.
///
/// The `direction` parameter must be one of: `"pos_infinity"`, `"neg_infinity"`, `"zero"`.
fn asymptotic_series_ffi(
    expression: &str,
    variable: &str,
    direction: &str,
    num_terms: u32,
) -> Result<ffi::AsymptoticSeriesResultFFI, String> {
    use crate::series::{asymptotic, AsymptoticDirection};

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

    let result = asymptotic(&expr, variable, dir, num_terms);

    match result {
        Ok(series) => {
            let series_expr = series.to_expression();
            Ok(ffi::AsymptoticSeriesResultFFI {
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
        Err(e) => Ok(ffi::AsymptoticSeriesResultFFI {
            original: expression.to_string(),
            variable: variable.to_string(),
            direction: direction.to_string(),
            num_terms,
            series: String::new(),
            series_latex: String::new(),
            success: false,
            error_message: format!("{}", e),
        }),
    }
}

// =============================================================================
// Special functions
// =============================================================================

/// Compute the Gamma function with derivation steps.
fn gamma_ffi(x: f64) -> Result<ffi::SpecialFunctionResultFFI, String> {
    use crate::special::gamma;

    let x_expr = crate::ast::Expression::Float(x);

    let result = gamma(&x_expr);

    match result {
        Ok(gamma_result) => {
            let steps_json = serde_json::to_string(&gamma_result.derivation_steps)
                .map_err(|e| format!("Failed to serialize derivation steps: {}", e))?;

            Ok(ffi::SpecialFunctionResultFFI {
                value: format!("{}", gamma_result.value),
                value_latex: gamma_result.value.to_latex(),
                numeric_value: gamma_result.numeric_value.unwrap_or(f64::NAN),
                derivation_steps: steps_json,
                success: true,
                error_message: String::new(),
            })
        }
        Err(e) => Ok(ffi::SpecialFunctionResultFFI {
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
fn erf_ffi(x: f64) -> Result<ffi::SpecialFunctionResultFFI, String> {
    use crate::special::erf;

    let x_expr = crate::ast::Expression::Float(x);

    let result = erf(&x_expr);

    match result {
        Ok(erf_result) => {
            let steps_json = serde_json::to_string(&erf_result.derivation_steps)
                .map_err(|e| format!("Failed to serialize derivation steps: {}", e))?;

            Ok(ffi::SpecialFunctionResultFFI {
                value: format!("{}", erf_result.value),
                value_latex: erf_result.value.to_latex(),
                numeric_value: erf_result.numeric_value.unwrap_or(f64::NAN),
                derivation_steps: steps_json,
                success: true,
                error_message: String::new(),
            })
        }
        Err(e) => Ok(ffi::SpecialFunctionResultFFI {
            value: String::new(),
            value_latex: String::new(),
            numeric_value: f64::NAN,
            derivation_steps: String::new(),
            success: false,
            error_message: format!("{}", e),
        }),
    }
}

/// Compute the Beta function B(a, b) = Γ(a)·Γ(b) / Γ(a+b) with derivation steps.
fn beta_ffi(a: f64, b: f64) -> Result<ffi::SpecialFunctionResultFFI, String> {
    use crate::special::beta;

    let a_expr = crate::ast::Expression::Float(a);
    let b_expr = crate::ast::Expression::Float(b);

    match beta(&a_expr, &b_expr) {
        Ok(beta_result) => {
            let steps_json = serde_json::to_string(&beta_result.derivation_steps)
                .map_err(|e| format!("Failed to serialize derivation steps: {}", e))?;

            Ok(ffi::SpecialFunctionResultFFI {
                value: format!("{}", beta_result.value),
                value_latex: beta_result.value.to_latex(),
                numeric_value: beta_result.numeric_value.unwrap_or(f64::NAN),
                derivation_steps: steps_json,
                success: true,
                error_message: String::new(),
            })
        }
        Err(e) => Ok(ffi::SpecialFunctionResultFFI {
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
fn erfc_ffi(x: f64) -> Result<ffi::SpecialFunctionResultFFI, String> {
    use crate::special::erfc;

    let x_expr = crate::ast::Expression::Float(x);

    match erfc(&x_expr) {
        Ok(erfc_result) => {
            let steps_json = serde_json::to_string(&erfc_result.derivation_steps)
                .map_err(|e| format!("Failed to serialize derivation steps: {}", e))?;

            Ok(ffi::SpecialFunctionResultFFI {
                value: format!("{}", erfc_result.value),
                value_latex: erfc_result.value.to_latex(),
                numeric_value: erfc_result.numeric_value.unwrap_or(f64::NAN),
                derivation_steps: steps_json,
                success: true,
                error_message: String::new(),
            })
        }
        Err(e) => Ok(ffi::SpecialFunctionResultFFI {
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
// ODE solving functions
// =============================================================================

/// Build an `ODEResultFFI` success value from an `ODESolution`.
fn ode_success(
    equation: &str,
    solution: &crate::ode::ODESolution,
    ode_type: &str,
) -> ffi::ODEResultFFI {
    let simplified = solution.general_solution.clone().simplify();
    ffi::ODEResultFFI {
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
fn ode_error(equation: &str, error: &str) -> ffi::ODEResultFFI {
    ffi::ODEResultFFI {
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
fn solve_ode_ffi(equation: &str, dependent_var: &str, independent_var: &str) -> ffi::ODEResultFFI {
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
fn solve_ode_ivp_ffi(
    equation: &str,
    dependent_var: &str,
    independent_var: &str,
    initial_conditions_json: &str,
) -> ffi::ODEResultFFI {
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
fn solve_second_order_ode_ffi(
    coefficients_json: &str,
    forcing_fn: &str,
) -> Result<ffi::ODEResultFFI, String> {
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
            let simplified = sol.general_solution.clone().simplify();
            Ok(ffi::ODEResultFFI {
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
fn solve_higher_order_ode_ffi(coefficients_json: &str) -> Result<ffi::ODEResultFFI, String> {
    use crate::ode_higher::{solve_higher_order_homogeneous, HigherOrderODE};

    let coeffs: Vec<f64> = serde_json::from_str(coefficients_json)
        .map_err(|e| format!("Invalid coefficients JSON: {}", e))?;
    if coeffs.len() < 2 {
        return Err("Need at least 2 coefficients to define an ODE".to_string());
    }

    let ode = HigherOrderODE::new("y", "x", coeffs);
    match solve_higher_order_homogeneous(&ode) {
        Ok(sol) => {
            let simplified = sol.general_solution.clone().simplify();
            Ok(ffi::ODEResultFFI {
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
fn rk4_solve_ffi(
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

// =============================================================================
// Fourier series operations
// =============================================================================

/// Compute the Fourier series of an expression.
///
/// Pass `period = 0.0` to use the default period of 2π.
fn fourier_series_ffi(
    expression: &str,
    variable: &str,
    num_terms: u32,
    period: f64,
) -> Result<ffi::FourierSeriesResultFFI, String> {
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
            Ok(ffi::FourierSeriesResultFFI {
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
        Err(e) => Ok(ffi::FourierSeriesResultFFI {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fourier_series_ffi_sin_x() {
        let result = fourier_series_ffi("sin(x)", "x", 3, 0.0).unwrap();
        assert!(
            result.success,
            "Expected success, got: {}",
            result.error_message
        );
        assert_eq!(result.variable, "x");
        assert_eq!(result.num_terms, 3);
        assert!((result.period - std::f64::consts::TAU).abs() < 1e-10);
        let b: Vec<f64> = serde_json::from_str(&result.b_coefficients_json).unwrap();
        assert!((b[1] - 1.0).abs() < 1e-4, "b_1 should be ~1, got {}", b[1]);
        assert!(!result.series.is_empty());
        assert!(!result.series_latex.is_empty());
    }

    #[test]
    fn test_fourier_series_ffi_cos_x() {
        let result = fourier_series_ffi("cos(x)", "x", 3, 0.0).unwrap();
        assert!(result.success);
        let a: Vec<f64> = serde_json::from_str(&result.a_coefficients_json).unwrap();
        assert!((a[1] - 1.0).abs() < 1e-4, "a_1 should be ~1, got {}", a[1]);
    }

    #[test]
    fn test_fourier_series_ffi_custom_period() {
        let result = fourier_series_ffi("cos(x)", "x", 3, std::f64::consts::TAU).unwrap();
        assert!(result.success);
        assert!((result.period - std::f64::consts::TAU).abs() < 1e-10);
    }

    #[test]
    fn test_fourier_series_ffi_invalid_expression() {
        let result = fourier_series_ffi("@@@", "x", 3, 0.0).unwrap();
        assert!(!result.success);
        assert!(!result.error_message.is_empty());
    }

    #[test]
    fn test_fourier_series_ffi_zero_terms_error() {
        let result = fourier_series_ffi("sin(x)", "x", 0, 0.0).unwrap();
        assert!(!result.success);
        assert!(result.error_message.contains("at least 1"));
    }
}

// =============================================================================
// Precision evaluation and manual-computation optimization
// =============================================================================

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
fn evaluate_with_precision_ffi(
    expression: &str,
    values_json: &str,
    mode: &str,
    precision: u32,
    rounding: &str,
) -> Result<ffi::PrecisionEvaluationResultFFI, String> {
    use crate::precision::{EvalContext, Value};
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
            Ok(ffi::PrecisionEvaluationResultFFI {
                original: expression.to_string(),
                value: numeric,
                value_string,
                precision_mode: precision_mode_str,
                rounding_mode: rounding_mode_str,
                success: true,
                error_message: String::new(),
            })
        }
        Err(e) => Ok(ffi::PrecisionEvaluationResultFFI {
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
fn optimize_for_manual_computation_ffi(expression: &str) -> Result<String, String> {
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
fn small_angle_approximation_ffi(
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

#[cfg(test)]
mod precision_tests {
    use super::*;

    #[test]
    fn test_evaluate_with_precision_fixed_decimal() {
        let result = evaluate_with_precision_ffi("1/3", "{}", "fixed", 4, "half_even").unwrap();
        assert!(result.success);
        assert!((result.value - 0.3333).abs() < 1e-3);
        assert!(result.precision_mode.contains("FixedDecimal"));
    }

    #[test]
    fn test_evaluate_with_precision_significant_figures() {
        let result =
            evaluate_with_precision_ffi("355/113", "{}", "significant", 6, "half_up").unwrap();
        assert!(result.success);
        assert!((result.value - std::f64::consts::PI).abs() < 1e-4);
        assert!(result.precision_mode.contains("SignificantFigures"));
    }

    #[test]
    fn test_evaluate_with_precision_with_variables() {
        let result =
            evaluate_with_precision_ffi("x * y", r#"{"x": 3.0, "y": 4.0}"#, "full", 0, "").unwrap();
        assert!(result.success);
        assert!((result.value - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_with_precision_unknown_mode() {
        let err = evaluate_with_precision_ffi("x", "{}", "bogus", 3, "").unwrap_err();
        assert!(err.contains("Unknown precision mode"));
    }

    #[test]
    fn test_evaluate_with_precision_unknown_rounding() {
        let err = evaluate_with_precision_ffi("1", "{}", "fixed", 2, "bogus").unwrap_err();
        assert!(err.contains("Unknown rounding mode"));
    }

    #[test]
    fn test_evaluate_with_precision_division_by_zero() {
        let result = evaluate_with_precision_ffi("1/0", "{}", "full", 0, "").unwrap();
        assert!(!result.success);
        assert!(!result.error_message.is_empty());
    }

    #[test]
    fn test_optimize_for_manual_computation_basic() {
        let json = optimize_for_manual_computation_ffi("a * b + c").unwrap();
        let val: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(val["steps"].is_array());
        assert!(val["step_count"].as_u64().unwrap() > 0);
        assert!(val["multiplicative_chains"].is_array());
    }

    #[test]
    fn test_optimize_for_manual_computation_parse_error() {
        let err = optimize_for_manual_computation_ffi("@@@invalid@@@").unwrap_err();
        assert!(err.contains("Parse error"));
    }

    #[test]
    fn test_small_angle_approximation_sin() {
        let json = small_angle_approximation_ffi("sin(x)", "x", 0.1).unwrap();
        let val: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(val["approximation"].as_str().unwrap(), "x");
        assert!(val["formula_used"].as_str().unwrap().contains("sin"));
    }

    #[test]
    fn test_small_angle_approximation_no_match() {
        let json = small_angle_approximation_ffi("x^2 + 1", "x", 0.1).unwrap();
        let val: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            val["formula_used"].as_str().unwrap(),
            "no approximation applied"
        );
    }

    #[test]
    fn test_small_angle_approximation_parse_error() {
        let err = small_angle_approximation_ffi("@@@", "x", 0.1).unwrap_err();
        assert!(err.contains("Parse error"));
    }
}
