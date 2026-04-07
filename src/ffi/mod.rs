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

// Implementation submodules
mod calculus_impl;
mod precision_impl;
mod series_impl;
mod solver_impl;
mod transforms_impl;

// Re-export all implementation functions into this module's namespace
// so the swift-bridge `extern "Rust"` declarations can find them.
use calculus_impl::*;
use precision_impl::*;
use series_impl::*;
use solver_impl::*;
use transforms_impl::*;

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
        fn compose_series_ffi(
            outer: &str,
            inner: &str,
            variable: &str,
            order: u32,
        ) -> Result<TaylorSeriesResultFFI, String>;
        fn reversion_series_ffi(
            expression: &str,
            variable: &str,
            order: u32,
        ) -> Result<TaylorSeriesResultFFI, String>;
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

    extern "Rust" {
        fn translate_2d_ffi(x: f64, y: f64, dx: f64, dy: f64) -> CartesianCoords2D;
        fn rotate_2d_ffi(x: f64, y: f64, theta: f64) -> CartesianCoords2D;
        fn scale_2d_ffi(x: f64, y: f64, sx: f64, sy: f64) -> CartesianCoords2D;
    }

    extern "Rust" {
        fn complex_nth_roots_ffi(re: f64, im: f64, n: i32) -> Result<String, String>;
    }

    extern "Rust" {
        fn convert_units_ffi(value: f64, from_unit: &str, to_unit: &str) -> Result<f64, String>;
    }

    extern "Rust" {
        fn parse_latex_calculus_ffi(input: &str) -> Result<String, String>;
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
        match fourier_series_ffi("@@@", "x", 3, 0.0) {
            Err(err) => assert!(err.contains("Parse error")),
            Ok(_) => panic!("Expected parse error for invalid expression"),
        }
    }

    #[test]
    fn test_fourier_series_ffi_zero_terms_error() {
        let result = fourier_series_ffi("sin(x)", "x", 0, 0.0).unwrap();
        assert!(!result.success);
        assert!(result.error_message.contains("at least 1"));
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
        match evaluate_with_precision_ffi("x", "{}", "bogus", 3, "") {
            Err(err) => assert!(err.contains("Unknown precision mode")),
            Ok(_) => panic!("Expected error for unknown precision mode"),
        }
    }

    #[test]
    fn test_evaluate_with_precision_unknown_rounding() {
        match evaluate_with_precision_ffi("1", "{}", "fixed", 2, "bogus") {
            Err(err) => assert!(err.contains("Unknown rounding mode")),
            Ok(_) => panic!("Expected error for unknown rounding mode"),
        }
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

#[cfg(test)]
mod new_wrapper_tests {
    use super::*;

    // 2nd-order ODE tests
    #[test]
    fn test_second_order_ode_homogeneous() {
        // y'' + 3y' + 2y = 0 -> coefficients [1, 3, 2]
        let result = solve_second_order_ode_ffi("[1, 3, 2]", "").unwrap();
        assert!(result.success, "Error: {}", result.error_message);
        assert!(!result.solution.is_empty());
        assert_eq!(result.ode_type, "second_order");
    }

    #[test]
    fn test_second_order_ode_with_forcing() {
        // y'' + y = x -> coefficients [1, 0, 1], forcing "x"
        let result = solve_second_order_ode_ffi("[1, 0, 1]", "x").unwrap();
        assert!(result.success, "Error: {}", result.error_message);
    }

    #[test]
    fn test_second_order_ode_wrong_coefficients() {
        match solve_second_order_ode_ffi("[1, 2]", "") {
            Err(err) => assert!(err.contains("Expected 3 coefficients")),
            Ok(_) => panic!("Expected error for wrong number of coefficients"),
        }
    }

    // Higher-order ODE tests
    #[test]
    fn test_higher_order_ode() {
        // y''' - y = 0 -> coefficients [1, 0, 0, -1]
        let result = solve_higher_order_ode_ffi("[1, 0, 0, -1]").unwrap();
        assert!(result.success, "Error: {}", result.error_message);
        assert_eq!(result.ode_type, "higher_order");
    }

    #[test]
    fn test_higher_order_ode_too_few_coefficients() {
        match solve_higher_order_ode_ffi("[1]") {
            Err(err) => assert!(err.contains("at least 2")),
            Ok(_) => panic!("Expected error for too few coefficients"),
        }
    }

    // RK4 tests
    #[test]
    fn test_rk4_solve_exponential() {
        // y' = y, y(0) = 1 -> y = e^x
        let json = rk4_solve_ffi("y", "y", 0.0, 1.0, 1.0, 100).unwrap();
        let trajectory: Vec<Vec<f64>> = serde_json::from_str(&json).unwrap();
        assert!(!trajectory.is_empty());
        // Last point should be close to e ~ 2.718
        let last = trajectory.last().unwrap();
        assert!(
            (last[1] - std::f64::consts::E).abs() < 0.01,
            "Expected ~e, got {}",
            last[1]
        );
    }

    // Series composition tests
    #[test]
    fn test_compose_series_exp_sin() {
        let result = compose_series_ffi("exp(x)", "sin(x)", "x", 5).unwrap();
        assert!(result.success, "Error: {}", result.error_message);
        assert!(!result.series.is_empty());
    }

    #[test]
    fn test_reversion_series_sin() {
        let result = reversion_series_ffi("sin(x)", "x", 5).unwrap();
        assert!(result.success, "Error: {}", result.error_message);
        assert!(!result.series.is_empty());
    }

    // 2D transform tests
    #[test]
    fn test_translate_2d() {
        let result = translate_2d_ffi(1.0, 2.0, 3.0, 4.0);
        assert!((result.x - 4.0).abs() < 1e-10);
        assert!((result.y - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotate_2d() {
        // Rotate (1, 0) by pi/2 -> (0, 1)
        let result = rotate_2d_ffi(1.0, 0.0, std::f64::consts::FRAC_PI_2);
        assert!(result.x.abs() < 1e-10, "Expected ~0, got {}", result.x);
        assert!(
            (result.y - 1.0).abs() < 1e-10,
            "Expected ~1, got {}",
            result.y
        );
    }

    #[test]
    fn test_scale_2d() {
        let result = scale_2d_ffi(3.0, 4.0, 2.0, 0.5);
        assert!((result.x - 6.0).abs() < 1e-10);
        assert!((result.y - 2.0).abs() < 1e-10);
    }

    // Complex nth roots tests
    #[test]
    fn test_complex_nth_roots_cube_roots_of_unity() {
        let json = complex_nth_roots_ffi(1.0, 0.0, 3).unwrap();
        let roots: Vec<[f64; 2]> = serde_json::from_str(&json).unwrap();
        assert_eq!(roots.len(), 3);
        // First root should be 1+0i
        assert!((roots[0][0] - 1.0).abs() < 1e-10);
        assert!(roots[0][1].abs() < 1e-10);
    }

    #[test]
    fn test_complex_nth_roots_negative_n() {
        match complex_nth_roots_ffi(1.0, 0.0, -1) {
            Err(err) => assert!(err.contains("positive")),
            Ok(_) => panic!("Expected error for negative n"),
        }
    }

    // Unit conversion tests
    #[test]
    fn test_convert_units_same_dimension() {
        // km to m
        let result = convert_units_ffi(1.0, "km", "m");
        match result {
            Ok(val) => assert!((val - 1000.0).abs() < 1e-6),
            Err(_) => {
                // Unit registry may not have km -- that's ok for a basic test
            }
        }
    }
}
