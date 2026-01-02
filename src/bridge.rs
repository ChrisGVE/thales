//! Swift bridge definitions for code generation.
//!
//! This file contains the `#[swift_bridge::bridge]` module that defines
//! the FFI interface. It's separate from ffi.rs to allow swift-bridge-build
//! to parse it without issues with doc comments.
//!
//! The actual implementations are in ffi.rs.

#[swift_bridge::bridge]
mod ffi {
    // =========================================================================
    // Result types
    // =========================================================================

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

    // =========================================================================
    // Parsing functions
    // =========================================================================

    extern "Rust" {
        fn parse_equation_ffi(input: &str) -> Result<String, String>;
        fn parse_expression_ffi(input: &str) -> Result<String, String>;
    }

    // =========================================================================
    // Equation solving functions
    // =========================================================================

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

    // =========================================================================
    // Coordinate transformation functions
    // =========================================================================

    extern "Rust" {
        fn cartesian_to_polar_ffi(x: f64, y: f64) -> PolarCoords;
        fn polar_to_cartesian_ffi(r: f64, theta: f64) -> CartesianCoords2D;
        fn cartesian_to_spherical_ffi(x: f64, y: f64, z: f64) -> SphericalCoords;
        fn spherical_to_cartesian_ffi(r: f64, theta: f64, phi: f64) -> CartesianCoords3D;
    }

    // =========================================================================
    // Complex number operations
    // =========================================================================

    extern "Rust" {
        fn complex_add_ffi(a_re: f64, a_im: f64, b_re: f64, b_im: f64) -> ComplexNumber;
        fn complex_multiply_ffi(a_re: f64, a_im: f64, b_re: f64, b_im: f64) -> ComplexNumber;
        fn complex_to_polar_ffi(re: f64, im: f64) -> PolarCoords;
        fn complex_power_ffi(re: f64, im: f64, n: f64) -> ComplexNumber;
    }

    // =========================================================================
    // LaTeX functions
    // =========================================================================

    extern "Rust" {
        fn parse_latex_ffi(input: &str) -> Result<String, String>;
        fn parse_latex_to_latex_ffi(input: &str) -> Result<String, String>;
        fn to_latex_ffi(expression: &str) -> Result<String, String>;
    }

    // =========================================================================
    // Calculus operations
    // =========================================================================

    extern "Rust" {
        fn differentiate_ffi(expression: &str, variable: &str) -> Result<DifferentiationResultFFI, String>;
        fn differentiate_n_ffi(expression: &str, variable: &str, n: u32) -> Result<DifferentiationResultFFI, String>;
        fn gradient_ffi(expression: &str, variables_json: &str) -> Result<String, String>;
        fn integrate_ffi(expression: &str, variable: &str) -> Result<IntegrationResultFFI, String>;
        fn definite_integral_ffi(
            expression: &str,
            variable: &str,
            lower: f64,
            upper: f64,
        ) -> Result<DefiniteIntegralResultFFI, String>;
        fn limit_ffi(expression: &str, variable: &str, approaches: f64) -> Result<LimitResultFFI, String>;
        fn limit_infinity_ffi(expression: &str, variable: &str) -> Result<LimitResultFFI, String>;
    }

    // =========================================================================
    // Expression evaluation and simplification
    // =========================================================================

    extern "Rust" {
        fn evaluate_ffi(expression: &str, values_json: &str) -> Result<EvaluationResultFFI, String>;
        fn simplify_ffi(expression: &str) -> Result<SimplificationResultFFI, String>;
        fn simplify_trig_ffi(expression: &str) -> Result<SimplificationResultFFI, String>;
        fn simplify_trig_with_steps_ffi(expression: &str) -> Result<String, String>;
    }

    // =========================================================================
    // Advanced solving operations
    // =========================================================================

    extern "Rust" {
        fn solve_system_ffi(equations_json: &str) -> Result<String, String>;
        fn solve_inequality_ffi(inequality: &str, variable: &str) -> Result<String, String>;
        fn partial_fractions_ffi(numerator: &str, denominator: &str, variable: &str) -> Result<String, String>;
        fn solve_equation_system_ffi(
            equations_json: &str,
            known_values_json: &str,
            targets_json: &str,
        ) -> Result<String, String>;
    }
}
