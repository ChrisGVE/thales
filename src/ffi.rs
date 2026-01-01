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
//! import MathSolverCore
//!
//! // Parse and solve an equation
//! do {
//!     let solution = try solve_equation_ffi("2*x + 5 = 15", "x")
//!     print("Solution: \(solution)")
//! } catch {
//!     print("Error: \(error)")
//! }
//! ```
//!
//! # Categories
//!
//! The FFI is organized into these functional areas:
//! - **Parsing**: `parse_equation_ffi`, `parse_expression_ffi`
//! - **Symbolic Solving**: `solve_equation_ffi`, `solve_with_values_ffi`
//! - **Numerical Solving**: `solve_numerically_ffi`
//! - **Coordinate Transforms**: Cartesian ↔ Polar, Cartesian ↔ Spherical
//! - **Complex Numbers**: Arithmetic and conversions

#[swift_bridge::bridge]
mod ffi {
    /// Parsing functions for equations and expressions.
    ///
    /// These functions parse mathematical text into internal AST representations.
    extern "Rust" {
        /// Parse an equation string into its AST representation.
        ///
        /// # Arguments
        /// * `input` - Equation string (e.g., "2*x + 5 = 15")
        ///
        /// # Returns
        /// * `Ok(String)` - Debug representation of parsed equation
        /// * `Err(String)` - Parse error description
        ///
        /// # Example (Swift)
        /// ```swift
        /// do {
        ///     let parsed = try parse_equation_ffi("a^2 + b^2 = c^2")
        ///     print("Parsed: \(parsed)")
        /// } catch {
        ///     print("Parse error: \(error)")
        /// }
        /// ```
        #[swift_bridge(return_with = Result)]
        fn parse_equation_ffi(input: &str) -> Result<String, String>;

        /// Parse an expression string into its AST representation.
        ///
        /// # Arguments
        /// * `input` - Expression string (e.g., "2*x + 5")
        ///
        /// # Returns
        /// * `Ok(String)` - Debug representation of parsed expression
        /// * `Err(String)` - Parse error description
        ///
        /// # Example (Swift)
        /// ```swift
        /// do {
        ///     let parsed = try parse_expression_ffi("sin(x) + cos(y)")
        ///     print("Parsed: \(parsed)")
        /// } catch {
        ///     print("Parse error: \(error)")
        /// }
        /// ```
        #[swift_bridge(return_with = Result)]
        fn parse_expression_ffi(input: &str) -> Result<String, String>;
    }

    /// Equation solving functions (symbolic and numerical).
    ///
    /// Provides multiple solving strategies for different use cases.
    extern "Rust" {
        /// Solve an equation symbolically for a variable.
        ///
        /// Returns a symbolic solution without substituting numerical values.
        ///
        /// # Arguments
        /// * `equation` - Equation string (e.g., "2*x + 5 = 15")
        /// * `variable` - Variable to solve for (e.g., "x")
        ///
        /// # Returns
        /// * `Ok(String)` - Symbolic solution expression
        /// * `Err(String)` - Solving error description
        ///
        /// # Example (Swift)
        /// ```swift
        /// do {
        ///     let solution = try solve_equation_ffi("2*x + 5 = 15", "x")
        ///     print("x = \(solution)")  // Output: x = (15 - 5) / 2
        /// } catch {
        ///     print("Solving failed: \(error)")
        /// }
        /// ```
        #[swift_bridge(return_with = Result)]
        fn solve_equation_ffi(equation: &str, variable: &str) -> Result<String, String>;

        /// Solve an equation with known variable values and return full resolution path.
        ///
        /// Substitutes known values and provides step-by-step solution path.
        ///
        /// # Arguments
        /// * `equation` - Equation string (e.g., "F = m * a")
        /// * `variable` - Variable to solve for (e.g., "a")
        /// * `known_values_json` - JSON object of known values (e.g., `{"F": 100.0, "m": 20.0}`)
        ///
        /// # Returns
        /// * `Ok(ResolutionPathFFI)` - Full solution path with steps
        /// * `Err(String)` - Solving error description
        ///
        /// # JSON Format
        /// The `known_values_json` parameter expects a JSON object mapping variable names
        /// to numerical values:
        /// ```json
        /// {
        ///   "variable_name": numeric_value,
        ///   "another_var": numeric_value
        /// }
        /// ```
        ///
        /// # Example (Swift)
        /// ```swift
        /// let knownValues = """
        /// {
        ///   "F": 100.0,
        ///   "m": 20.0
        /// }
        /// """
        ///
        /// do {
        ///     let path = try solve_with_values_ffi("F = m * a", "a", knownValues)
        ///     print("Initial: \(path.initial_expr)")
        ///     print("Result: \(path.result_expr)")
        ///
        ///     // Parse steps JSON
        ///     if let data = path.steps_json.data(using: .utf8),
        ///        let steps = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
        ///         for step in steps {
        ///             print("Step: \(step["operation"] ?? "")")
        ///             print("  -> \(step["result"] ?? "")")
        ///         }
        ///     }
        /// } catch {
        ///     print("Solving failed: \(error)")
        /// }
        /// ```
        #[swift_bridge(return_with = Result)]
        fn solve_with_values_ffi(
            equation: &str,
            variable: &str,
            known_values_json: &str,
        ) -> Result<ResolutionPathFFI, String>;

        /// Solve an equation numerically using Newton-Raphson method.
        ///
        /// Uses numerical methods when symbolic solving is impractical or impossible.
        ///
        /// # Arguments
        /// * `equation` - Equation string (e.g., "x^3 - 2*x - 5 = 0")
        /// * `variable` - Variable to solve for (e.g., "x")
        /// * `initial_guess` - Starting value for numerical iteration
        ///
        /// # Returns
        /// * `Ok(f64)` - Numerical solution
        /// * `Err(String)` - Error if solving failed or did not converge
        ///
        /// # Example (Swift)
        /// ```swift
        /// do {
        ///     let solution = try solve_numerically_ffi("x^3 - 2*x - 5 = 0", "x", 2.0)
        ///     print("x ≈ \(solution)")  // Output: x ≈ 2.094551...
        /// } catch {
        ///     print("Numerical solving failed: \(error)")
        /// }
        /// ```
        #[swift_bridge(return_with = Result)]
        fn solve_numerically_ffi(
            equation: &str,
            variable: &str,
            initial_guess: f64,
        ) -> Result<f64, String>;
    }

    /// Resolution path containing step-by-step solution details.
    ///
    /// Returned by `solve_with_values_ffi` to provide complete solution information
    /// including intermediate steps for educational purposes.
    ///
    /// # Fields
    /// * `initial_expr` - The original equation after parsing
    /// * `steps_json` - JSON array of solution steps (see format below)
    /// * `result_expr` - Final result expression
    /// * `success` - Whether solving succeeded
    ///
    /// # Steps JSON Format
    /// The `steps_json` field contains a JSON array of step objects:
    /// ```json
    /// [
    ///   {
    ///     "operation": "IsolateTerm",
    ///     "explanation": "Subtract 5 from both sides",
    ///     "result": "2*x = 10"
    ///   },
    ///   {
    ///     "operation": "DivideBothSides",
    ///     "explanation": "Divide both sides by 2",
    ///     "result": "x = 5"
    ///   }
    /// ]
    /// ```
    ///
    /// # Example (Swift)
    /// ```swift
    /// let path = try solve_with_values_ffi("2*x + 5 = 15", "x", "{}")
    /// print("Initial: \(path.initial_expr)")
    /// print("Success: \(path.success)")
    /// print("Result: \(path.result_expr)")
    ///
    /// // Decode steps
    /// let decoder = JSONDecoder()
    /// if let data = path.steps_json.data(using: .utf8),
    ///    let steps = try? decoder.decode([[String: String]].self, from: data) {
    ///     for (i, step) in steps.enumerated() {
    ///         print("Step \(i + 1): \(step["explanation"] ?? "")")
    ///     }
    /// }
    /// ```
    #[swift_bridge(swift_repr = "struct")]
    pub struct ResolutionPathFFI {
        pub initial_expr: String,
        pub steps_json: String,
        pub result_expr: String,
        pub success: bool,
    }

    /// Coordinate transformation functions.
    ///
    /// Convert between Cartesian, polar, and spherical coordinate systems.
    extern "Rust" {
        /// Convert 2D Cartesian coordinates to polar coordinates.
        ///
        /// # Arguments
        /// * `x` - X coordinate
        /// * `y` - Y coordinate
        ///
        /// # Returns
        /// Polar coordinates with radius `r` and angle `theta` (in radians)
        ///
        /// # Example (Swift)
        /// ```swift
        /// let polar = cartesian_to_polar_ffi(3.0, 4.0)
        /// print("r = \(polar.r), θ = \(polar.theta) rad")  // r = 5.0, θ = 0.927...
        /// ```
        fn cartesian_to_polar_ffi(x: f64, y: f64) -> PolarCoords;

        /// Convert polar coordinates to 2D Cartesian coordinates.
        ///
        /// # Arguments
        /// * `r` - Radius
        /// * `theta` - Angle in radians
        ///
        /// # Returns
        /// Cartesian coordinates (x, y)
        ///
        /// # Example (Swift)
        /// ```swift
        /// let cart = polar_to_cartesian_ffi(5.0, 0.927)
        /// print("x = \(cart.x), y = \(cart.y)")  // x ≈ 3.0, y ≈ 4.0
        /// ```
        fn polar_to_cartesian_ffi(r: f64, theta: f64) -> CartesianCoords2D;

        /// Convert 3D Cartesian coordinates to spherical coordinates.
        ///
        /// # Arguments
        /// * `x` - X coordinate
        /// * `y` - Y coordinate
        /// * `z` - Z coordinate
        ///
        /// # Returns
        /// Spherical coordinates with:
        /// - `r` - Radial distance
        /// - `theta` - Azimuthal angle (in radians)
        /// - `phi` - Polar angle from z-axis (in radians)
        ///
        /// # Example (Swift)
        /// ```swift
        /// let spherical = cartesian_to_spherical_ffi(1.0, 1.0, 1.0)
        /// print("r = \(spherical.r), θ = \(spherical.theta), φ = \(spherical.phi)")
        /// ```
        fn cartesian_to_spherical_ffi(x: f64, y: f64, z: f64) -> SphericalCoords;

        /// Convert spherical coordinates to 3D Cartesian coordinates.
        ///
        /// # Arguments
        /// * `r` - Radial distance
        /// * `theta` - Azimuthal angle (in radians)
        /// * `phi` - Polar angle from z-axis (in radians)
        ///
        /// # Returns
        /// Cartesian coordinates (x, y, z)
        ///
        /// # Example (Swift)
        /// ```swift
        /// let cart = spherical_to_cartesian_ffi(1.732, 0.785, 0.955)
        /// print("x = \(cart.x), y = \(cart.y), z = \(cart.z)")
        /// ```
        fn spherical_to_cartesian_ffi(r: f64, theta: f64, phi: f64) -> CartesianCoords3D;
    }

    /// Complex number operations.
    ///
    /// Perform arithmetic and conversions with complex numbers.
    extern "Rust" {
        /// Add two complex numbers.
        ///
        /// # Arguments
        /// * `a_re` - Real part of first complex number
        /// * `a_im` - Imaginary part of first complex number
        /// * `b_re` - Real part of second complex number
        /// * `b_im` - Imaginary part of second complex number
        ///
        /// # Returns
        /// Sum as a ComplexNumber
        ///
        /// # Example (Swift)
        /// ```swift
        /// let sum = complex_add_ffi(3.0, 4.0, 1.0, 2.0)
        /// print("\(sum.real) + \(sum.imaginary)i")  // 4.0 + 6.0i
        /// ```
        fn complex_add_ffi(a_re: f64, a_im: f64, b_re: f64, b_im: f64) -> ComplexNumber;

        /// Multiply two complex numbers.
        ///
        /// # Arguments
        /// * `a_re` - Real part of first complex number
        /// * `a_im` - Imaginary part of first complex number
        /// * `b_re` - Real part of second complex number
        /// * `b_im` - Imaginary part of second complex number
        ///
        /// # Returns
        /// Product as a ComplexNumber
        ///
        /// # Example (Swift)
        /// ```swift
        /// let product = complex_multiply_ffi(3.0, 4.0, 1.0, 2.0)
        /// print("\(product.real) + \(product.imaginary)i")  // -5.0 + 10.0i
        /// ```
        fn complex_multiply_ffi(a_re: f64, a_im: f64, b_re: f64, b_im: f64) -> ComplexNumber;

        /// Convert complex number to polar form.
        ///
        /// # Arguments
        /// * `re` - Real part
        /// * `im` - Imaginary part
        ///
        /// # Returns
        /// Polar coordinates where `r` is magnitude and `theta` is phase angle (radians)
        ///
        /// # Example (Swift)
        /// ```swift
        /// let polar = complex_to_polar_ffi(3.0, 4.0)
        /// print("Magnitude: \(polar.r), Phase: \(polar.theta)")  // |z| = 5.0, arg(z) = 0.927
        /// ```
        fn complex_to_polar_ffi(re: f64, im: f64) -> PolarCoords;

        /// Raise complex number to a power using De Moivre's theorem.
        ///
        /// # Arguments
        /// * `re` - Real part of base
        /// * `im` - Imaginary part of base
        /// * `n` - Exponent (can be fractional)
        ///
        /// # Returns
        /// Result as a ComplexNumber
        ///
        /// # Example (Swift)
        /// ```swift
        /// let result = complex_power_ffi(1.0, 1.0, 2.0)
        /// print("\(result.real) + \(result.imaginary)i")  // (1+i)^2 = 0.0 + 2.0i
        /// ```
        fn complex_power_ffi(re: f64, im: f64, n: f64) -> ComplexNumber;
    }

    /// 2D Cartesian coordinate representation.
    ///
    /// Used as return type for polar-to-Cartesian conversions.
    ///
    /// # Fields
    /// * `x` - X coordinate
    /// * `y` - Y coordinate
    #[swift_bridge(swift_repr = "struct")]
    pub struct CartesianCoords2D {
        pub x: f64,
        pub y: f64,
    }

    /// 3D Cartesian coordinate representation.
    ///
    /// Used as return type for spherical-to-Cartesian conversions.
    ///
    /// # Fields
    /// * `x` - X coordinate
    /// * `y` - Y coordinate
    /// * `z` - Z coordinate
    #[swift_bridge(swift_repr = "struct")]
    pub struct CartesianCoords3D {
        pub x: f64,
        pub y: f64,
        pub z: f64,
    }

    /// Polar coordinate representation.
    ///
    /// Used for 2D polar coordinates and complex number polar form.
    ///
    /// # Fields
    /// * `r` - Radial distance (magnitude)
    /// * `theta` - Angle in radians (azimuthal angle or phase)
    #[swift_bridge(swift_repr = "struct")]
    pub struct PolarCoords {
        pub r: f64,
        pub theta: f64,
    }

    /// Spherical coordinate representation.
    ///
    /// Uses physics convention (ISO 80000-2):
    /// - `r`: radial distance
    /// - `theta`: azimuthal angle (angle in x-y plane from x-axis)
    /// - `phi`: polar angle (angle from positive z-axis)
    ///
    /// # Fields
    /// * `r` - Radial distance
    /// * `theta` - Azimuthal angle in radians (0 to 2π)
    /// * `phi` - Polar angle in radians (0 to π)
    #[swift_bridge(swift_repr = "struct")]
    pub struct SphericalCoords {
        pub r: f64,
        pub theta: f64,
        pub phi: f64,
    }

    /// Complex number representation.
    ///
    /// Represents numbers in the form `a + bi` where `i² = -1`.
    ///
    /// # Fields
    /// * `real` - Real part (a)
    /// * `imaginary` - Imaginary part (b)
    ///
    /// # Example (Swift)
    /// ```swift
    /// let z = ComplexNumber(real: 3.0, imaginary: 4.0)
    /// let polar = complex_to_polar_ffi(z.real, z.imaginary)
    /// print("Magnitude: \(polar.r)")  // |3+4i| = 5.0
    /// ```
    #[swift_bridge(swift_repr = "struct")]
    pub struct ComplexNumber {
        pub real: f64,
        pub imaginary: f64,
    }

    // =========================================================================
    // New CAS Operation Types and Functions
    // =========================================================================

    /// Result of symbolic differentiation.
    ///
    /// # Fields
    /// * `original` - The original expression
    /// * `variable` - Variable with respect to which we differentiated
    /// * `derivative` - The derivative expression as a string
    /// * `derivative_latex` - The derivative in LaTeX format
    #[swift_bridge(swift_repr = "struct")]
    pub struct DifferentiationResultFFI {
        pub original: String,
        pub variable: String,
        pub derivative: String,
        pub derivative_latex: String,
    }

    /// Result of symbolic integration.
    ///
    /// # Fields
    /// * `original` - The original expression
    /// * `variable` - Variable of integration
    /// * `integral` - The integral expression as a string (empty if failed)
    /// * `integral_latex` - The integral in LaTeX format
    /// * `success` - Whether integration succeeded
    /// * `error_message` - Error description if failed
    #[swift_bridge(swift_repr = "struct")]
    pub struct IntegrationResultFFI {
        pub original: String,
        pub variable: String,
        pub integral: String,
        pub integral_latex: String,
        pub success: bool,
        pub error_message: String,
    }

    /// Result of definite integral evaluation.
    ///
    /// # Fields
    /// * `original` - The original expression
    /// * `variable` - Variable of integration
    /// * `lower_bound` - Lower limit of integration
    /// * `upper_bound` - Upper limit of integration
    /// * `value` - The result expression as a string
    /// * `value_latex` - The result in LaTeX format
    /// * `numeric_value` - Numerical value (NaN if not evaluable)
    /// * `success` - Whether evaluation succeeded
    /// * `error_message` - Error description if failed
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

    /// Result of limit evaluation.
    ///
    /// # Fields
    /// * `original` - The original expression
    /// * `variable` - The variable approaching the limit
    /// * `approaches` - The value being approached
    /// * `value` - The limit value as a string
    /// * `value_latex` - The limit in LaTeX format
    /// * `numeric_value` - Numerical value (NaN if symbolic)
    /// * `success` - Whether evaluation succeeded
    /// * `error_message` - Error description if failed
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

    /// Result of expression evaluation.
    ///
    /// # Fields
    /// * `original` - The original expression
    /// * `value` - The numerical result
    /// * `success` - Whether evaluation succeeded
    /// * `error_message` - Error description if failed
    #[swift_bridge(swift_repr = "struct")]
    pub struct EvaluationResultFFI {
        pub original: String,
        pub value: f64,
        pub success: bool,
        pub error_message: String,
    }

    /// Result of expression simplification.
    ///
    /// # Fields
    /// * `original` - The original expression
    /// * `simplified` - The simplified expression as a string
    /// * `simplified_latex` - The simplified expression in LaTeX format
    #[swift_bridge(swift_repr = "struct")]
    pub struct SimplificationResultFFI {
        pub original: String,
        pub simplified: String,
        pub simplified_latex: String,
    }

    /// LaTeX parsing and rendering functions.
    extern "Rust" {
        /// Parse a LaTeX expression and return its AST representation.
        ///
        /// # Arguments
        /// * `input` - LaTeX expression string (e.g., "\\frac{1}{2}")
        ///
        /// # Returns
        /// * `Ok(String)` - Debug representation of parsed expression
        /// * `Err(String)` - Parse error description
        #[swift_bridge(return_with = Result)]
        fn parse_latex_ffi(input: &str) -> Result<String, String>;

        /// Parse a LaTeX expression and return it as normalized LaTeX.
        ///
        /// # Arguments
        /// * `input` - LaTeX expression string
        ///
        /// # Returns
        /// * `Ok(String)` - Normalized LaTeX output
        /// * `Err(String)` - Parse error description
        #[swift_bridge(return_with = Result)]
        fn parse_latex_to_latex_ffi(input: &str) -> Result<String, String>;

        /// Convert a mathematical expression to LaTeX format.
        ///
        /// # Arguments
        /// * `expression` - Expression string (e.g., "x^2 + 2*x + 1")
        ///
        /// # Returns
        /// * `Ok(String)` - LaTeX representation
        /// * `Err(String)` - Parse error description
        #[swift_bridge(return_with = Result)]
        fn to_latex_ffi(expression: &str) -> Result<String, String>;
    }

    /// Calculus operations: differentiation, integration, limits.
    extern "Rust" {
        /// Compute the symbolic derivative of an expression.
        ///
        /// # Arguments
        /// * `expression` - Expression to differentiate (e.g., "x^2 + 2*x")
        /// * `variable` - Variable to differentiate with respect to (e.g., "x")
        ///
        /// # Returns
        /// * `Ok(DifferentiationResultFFI)` - The derivative with metadata
        /// * `Err(String)` - Error description
        ///
        /// # Example (Swift)
        /// ```swift
        /// let result = try differentiate_ffi("x^3 + 2*x", "x")
        /// print("d/dx = \(result.derivative)")  // "3*x^2 + 2"
        /// print("LaTeX: \(result.derivative_latex)")
        /// ```
        #[swift_bridge(return_with = Result)]
        fn differentiate_ffi(expression: &str, variable: &str) -> Result<DifferentiationResultFFI, String>;

        /// Compute the nth derivative of an expression.
        ///
        /// # Arguments
        /// * `expression` - Expression to differentiate
        /// * `variable` - Variable to differentiate with respect to
        /// * `n` - Number of times to differentiate
        ///
        /// # Returns
        /// * `Ok(DifferentiationResultFFI)` - The nth derivative
        /// * `Err(String)` - Error description
        #[swift_bridge(return_with = Result)]
        fn differentiate_n_ffi(expression: &str, variable: &str, n: u32) -> Result<DifferentiationResultFFI, String>;

        /// Compute the gradient of an expression.
        ///
        /// # Arguments
        /// * `expression` - Expression to compute gradient for
        /// * `variables_json` - JSON array of variable names: ["x", "y", "z"]
        ///
        /// # Returns
        /// * `Ok(String)` - JSON array of partial derivatives
        /// * `Err(String)` - Error description
        #[swift_bridge(return_with = Result)]
        fn gradient_ffi(expression: &str, variables_json: &str) -> Result<String, String>;

        /// Compute the indefinite integral of an expression.
        ///
        /// # Arguments
        /// * `expression` - Expression to integrate
        /// * `variable` - Variable of integration
        ///
        /// # Returns
        /// * `Ok(IntegrationResultFFI)` - The integral with metadata
        /// * `Err(String)` - Error description
        ///
        /// # Example (Swift)
        /// ```swift
        /// let result = try integrate_ffi("x^2", "x")
        /// if result.success {
        ///     print("∫ = \(result.integral)")  // "x^3/3"
        /// }
        /// ```
        #[swift_bridge(return_with = Result)]
        fn integrate_ffi(expression: &str, variable: &str) -> Result<IntegrationResultFFI, String>;

        /// Compute a definite integral.
        ///
        /// # Arguments
        /// * `expression` - Expression to integrate
        /// * `variable` - Variable of integration
        /// * `lower` - Lower bound
        /// * `upper` - Upper bound
        ///
        /// # Returns
        /// * `Ok(DefiniteIntegralResultFFI)` - The definite integral value
        /// * `Err(String)` - Error description
        #[swift_bridge(return_with = Result)]
        fn definite_integral_ffi(expression: &str, variable: &str, lower: f64, upper: f64) -> Result<DefiniteIntegralResultFFI, String>;

        /// Evaluate a limit.
        ///
        /// # Arguments
        /// * `expression` - Expression to evaluate limit of
        /// * `variable` - Variable approaching the limit
        /// * `approaches` - Value the variable approaches
        ///
        /// # Returns
        /// * `Ok(LimitResultFFI)` - The limit value
        /// * `Err(String)` - Error description
        ///
        /// # Example (Swift)
        /// ```swift
        /// let result = try limit_ffi("sin(x)/x", "x", 0.0)
        /// print("lim = \(result.value)")  // "1"
        /// ```
        #[swift_bridge(return_with = Result)]
        fn limit_ffi(expression: &str, variable: &str, approaches: f64) -> Result<LimitResultFFI, String>;

        /// Evaluate a limit at positive infinity.
        ///
        /// # Arguments
        /// * `expression` - Expression to evaluate limit of
        /// * `variable` - Variable approaching infinity
        ///
        /// # Returns
        /// * `Ok(LimitResultFFI)` - The limit value
        /// * `Err(String)` - Error description
        #[swift_bridge(return_with = Result)]
        fn limit_infinity_ffi(expression: &str, variable: &str) -> Result<LimitResultFFI, String>;
    }

    /// Expression evaluation and simplification.
    extern "Rust" {
        /// Evaluate an expression numerically with given variable values.
        ///
        /// # Arguments
        /// * `expression` - Expression to evaluate
        /// * `values_json` - JSON object mapping variable names to values: {"x": 2.0, "y": 3.0}
        ///
        /// # Returns
        /// * `Ok(EvaluationResultFFI)` - The numerical result
        /// * `Err(String)` - Error description
        ///
        /// # Example (Swift)
        /// ```swift
        /// let result = try evaluate_ffi("x^2 + y", "{\"x\": 3.0, \"y\": 1.0}")
        /// print("value = \(result.value)")  // 10.0
        /// ```
        #[swift_bridge(return_with = Result)]
        fn evaluate_ffi(expression: &str, values_json: &str) -> Result<EvaluationResultFFI, String>;

        /// Simplify an algebraic expression.
        ///
        /// # Arguments
        /// * `expression` - Expression to simplify
        ///
        /// # Returns
        /// * `Ok(SimplificationResultFFI)` - The simplified expression
        /// * `Err(String)` - Error description
        #[swift_bridge(return_with = Result)]
        fn simplify_ffi(expression: &str) -> Result<SimplificationResultFFI, String>;

        /// Simplify a trigonometric expression.
        ///
        /// # Arguments
        /// * `expression` - Expression to simplify
        ///
        /// # Returns
        /// * `Ok(SimplificationResultFFI)` - The simplified expression
        /// * `Err(String)` - Error description
        #[swift_bridge(return_with = Result)]
        fn simplify_trig_ffi(expression: &str) -> Result<SimplificationResultFFI, String>;

        /// Simplify a trigonometric expression with step-by-step output.
        ///
        /// # Arguments
        /// * `expression` - Expression to simplify
        ///
        /// # Returns
        /// * `Ok(String)` - JSON with simplified expression and steps
        /// * `Err(String)` - Error description
        #[swift_bridge(return_with = Result)]
        fn simplify_trig_with_steps_ffi(expression: &str) -> Result<String, String>;
    }

    /// Advanced solving operations.
    extern "Rust" {
        /// Solve a system of linear equations.
        ///
        /// # Arguments
        /// * `equations_json` - JSON array of equation strings: ["2*x + y = 5", "x - y = 1"]
        ///
        /// # Returns
        /// * `Ok(String)` - JSON object mapping variables to solutions
        /// * `Err(String)` - Error description
        #[swift_bridge(return_with = Result)]
        fn solve_system_ffi(equations_json: &str) -> Result<String, String>;

        /// Solve an inequality.
        ///
        /// # Arguments
        /// * `inequality` - Inequality string (e.g., "x^2 - 4 < 0")
        /// * `variable` - Variable to solve for
        ///
        /// # Returns
        /// * `Ok(String)` - Solution interval description
        /// * `Err(String)` - Error description
        #[swift_bridge(return_with = Result)]
        fn solve_inequality_ffi(inequality: &str, variable: &str) -> Result<String, String>;

        /// Perform partial fraction decomposition.
        ///
        /// # Arguments
        /// * `numerator` - Numerator polynomial
        /// * `denominator` - Denominator polynomial
        /// * `variable` - Variable name
        ///
        /// # Returns
        /// * `Ok(String)` - JSON with decomposition details
        /// * `Err(String)` - Error description
        #[swift_bridge(return_with = Result)]
        fn partial_fractions_ffi(numerator: &str, denominator: &str, variable: &str) -> Result<String, String>;

        /// Solve a multi-equation system.
        ///
        /// Takes multiple equations of any type (algebraic, ODE, differential, etc.),
        /// known values, and target variables. Automatically determines solving order
        /// using dependency analysis and chains solutions through equations.
        ///
        /// # Arguments
        /// * `equations_json` - JSON object: {"eq1": "F = m * a", "eq2": "v = u + a * t"}
        /// * `known_values_json` - JSON object: {"F": 100.0, "m": 20.0, "u": 0.0, "t": 5.0}
        /// * `targets_json` - JSON array: ["a", "v"]
        ///
        /// # Returns
        /// * `Ok(String)` - JSON with solutions and step-by-step resolution path
        /// * `Err(String)` - Error description
        ///
        /// # Example (Swift)
        /// ```swift
        /// let equations = """
        ///     {"newton": "F = m * a", "kinematic": "v = u + a * t"}
        /// """
        /// let known = """
        ///     {"F": 100.0, "m": 20.0, "u": 0.0, "t": 5.0}
        /// """
        /// let targets = """
        ///     ["a", "v"]
        /// """
        /// let result = try solve_equation_system_ffi(equations, known, targets)
        /// // Returns: {"solutions": {"a": 5.0, "v": 25.0}, "steps": [...]}
        /// ```
        #[swift_bridge(return_with = Result)]
        fn solve_equation_system_ffi(
            equations_json: &str,
            known_values_json: &str,
            targets_json: &str,
        ) -> Result<String, String>;
    }
}

// Implementation of FFI functions
use crate::parser::{parse_equation, parse_expression};
use crate::transforms::{Cartesian2D, Cartesian3D, ComplexOps, Polar, Spherical};
use num_complex::Complex64;

/// Parse equation and return string representation.
fn parse_equation_ffi(input: &str) -> Result<String, String> {
    parse_equation(input)
        .map(|eq| format!("{:?}", eq))
        .map_err(|e| format!("Parse error: {:?}", e))
}

/// Parse expression and return string representation.
fn parse_expression_ffi(input: &str) -> Result<String, String> {
    parse_expression(input)
        .map(|expr| format!("{:?}", expr))
        .map_err(|e| format!("Parse error: {:?}", e))
}

/// Solve equation symbolically.
fn solve_equation_ffi(equation: &str, variable: &str) -> Result<String, String> {
    use crate::parser::parse_equation;
    use crate::solver::solve_for;
    use std::collections::HashMap;

    // Parse the equation string
    let parsed_equation =
        parse_equation(equation).map_err(|e| format!("Failed to parse equation: {:?}", e))?;

    // Solve for the variable with no known values
    let known_values = HashMap::new();
    let resolution_path = solve_for(&parsed_equation, variable, &known_values)
        .map_err(|e| format!("Failed to solve equation: {:?}", e))?;

    // Format the result as a string
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
    use crate::parser::parse_equation;

    // Parse the equation string
    let parsed_equation =
        parse_equation(equation).map_err(|e| format!("Failed to parse equation: {:?}", e))?;

    // Create solver with initial guess
    let mut config = NumericalConfig::default();
    config.initial_guess = Some(initial_guess);
    let solver = SmartNumericalSolver::new(config);

    // Solve numerically
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
    use crate::parser::parse_equation;
    use crate::solver::solve_for;
    use std::collections::HashMap;

    // Parse the equation string
    let parsed_equation =
        parse_equation(equation).map_err(|e| format!("Failed to parse equation: {:?}", e))?;

    // Parse known values from JSON
    let known_values: HashMap<String, f64> = if known_values_json.is_empty() {
        HashMap::new()
    } else {
        serde_json::from_str(known_values_json)
            .map_err(|e| format!("Failed to parse known values JSON: {}", e))?
    };

    // Solve for the variable
    let resolution_path = solve_for(&parsed_equation, variable, &known_values)
        .map_err(|e| format!("Failed to solve equation: {:?}", e))?;

    // Convert steps to JSON manually (using simple format)
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
// New CAS Operations FFI Functions
// =============================================================================

/// Parse LaTeX expression and return string representation.
fn parse_latex_ffi(input: &str) -> Result<String, String> {
    use crate::latex::parse_latex;
    parse_latex(input)
        .map(|expr| format!("{:?}", expr))
        .map_err(|e| format!("LaTeX parse error: {:?}", e))
}

/// Parse LaTeX expression and return as LaTeX output.
fn parse_latex_to_latex_ffi(input: &str) -> Result<String, String> {
    use crate::latex::parse_latex;
    parse_latex(input)
        .map(|expr| expr.to_latex())
        .map_err(|e| format!("LaTeX parse error: {:?}", e))
}

/// Differentiate an expression with respect to a variable.
fn differentiate_ffi(expression: &str, variable: &str) -> Result<ffi::DifferentiationResultFFI, String> {
    use crate::parser::parse_expression;

    let expr = parse_expression(expression)
        .map_err(|e| format!("Parse error: {:?}", e))?;

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
fn differentiate_n_ffi(expression: &str, variable: &str, n: u32) -> Result<ffi::DifferentiationResultFFI, String> {
    use crate::parser::parse_expression;

    let expr = parse_expression(expression)
        .map_err(|e| format!("Parse error: {:?}", e))?;

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
    use crate::parser::parse_expression;

    let expr = parse_expression(expression)
        .map_err(|e| format!("Parse error: {:?}", e))?;

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

    serde_json::to_string(&gradient)
        .map_err(|e| format!("Failed to serialize gradient: {}", e))
}

/// Integrate an expression with respect to a variable (indefinite integral).
fn integrate_ffi(expression: &str, variable: &str) -> Result<ffi::IntegrationResultFFI, String> {
    use crate::integration::integrate;
    use crate::parser::parse_expression;

    let expr = parse_expression(expression)
        .map_err(|e| format!("Parse error: {:?}", e))?;

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
    use crate::parser::parse_expression;

    let expr = parse_expression(expression)
        .map_err(|e| format!("Parse error: {:?}", e))?;

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
    use crate::parser::parse_expression;

    let expr = parse_expression(expression)
        .map_err(|e| format!("Parse error: {:?}", e))?;

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
    use crate::parser::parse_expression;

    let expr = parse_expression(expression)
        .map_err(|e| format!("Parse error: {:?}", e))?;

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
        LimitResult::NegativeInfinity => ("-∞".to_string(), "-\\infty".to_string(), f64::NEG_INFINITY),
        LimitResult::Expression(expr) => (format!("{}", expr), expr.to_latex(), f64::NAN),
    }
}

/// Evaluate an expression with given variable values.
fn evaluate_ffi(expression: &str, values_json: &str) -> Result<ffi::EvaluationResultFFI, String> {
    use crate::parser::parse_expression;
    use std::collections::HashMap;

    let expr = parse_expression(expression)
        .map_err(|e| format!("Parse error: {:?}", e))?;

    let values: HashMap<String, f64> = serde_json::from_str(values_json)
        .map_err(|e| format!("Failed to parse values JSON: {}", e))?;

    let result = evaluate_expression(&expr, &values);

    match result {
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
            error_message: "Cannot evaluate expression (may contain undefined variables or operations)".to_string(),
        }),
    }
}

/// Helper to evaluate expression with variable substitution.
fn evaluate_expression(expr: &crate::ast::Expression, values: &std::collections::HashMap<String, f64>) -> Option<f64> {
    use crate::ast::{BinaryOp, Expression, Function, UnaryOp};

    match expr {
        Expression::Integer(n) => Some(*n as f64),
        Expression::Float(f) => Some(*f),
        Expression::Rational(r) => Some(*r.numer() as f64 / *r.denom() as f64),
        Expression::Variable(v) => values.get(&v.name).copied(),
        Expression::Constant(c) => {
            use crate::ast::SymbolicConstant;
            match c {
                SymbolicConstant::Pi => Some(std::f64::consts::PI),
                SymbolicConstant::E => Some(std::f64::consts::E),
                SymbolicConstant::I => None, // Complex not supported in f64
            }
        }
        Expression::Unary(op, inner) => {
            let v = evaluate_expression(inner, values)?;
            match op {
                UnaryOp::Neg => Some(-v),
                UnaryOp::Pos => Some(v),
                UnaryOp::Factorial => {
                    if v >= 0.0 && v == v.floor() {
                        Some((1..=(v as u64)).product::<u64>() as f64)
                    } else {
                        None
                    }
                }
            }
        }
        Expression::Binary(op, left, right) => {
            let l = evaluate_expression(left, values)?;
            let r = evaluate_expression(right, values)?;
            match op {
                BinaryOp::Add => Some(l + r),
                BinaryOp::Sub => Some(l - r),
                BinaryOp::Mul => Some(l * r),
                BinaryOp::Div => if r != 0.0 { Some(l / r) } else { None },
                BinaryOp::Mod => if r != 0.0 { Some(l % r) } else { None },
            }
        }
        Expression::Power(base, exp) => {
            let b = evaluate_expression(base, values)?;
            let e = evaluate_expression(exp, values)?;
            Some(b.powf(e))
        }
        Expression::Function(func, args) => {
            let arg_values: Option<Vec<f64>> = args.iter()
                .map(|a| evaluate_expression(a, values))
                .collect();
            let arg_values = arg_values?;

            match func {
                Function::Sin => Some(arg_values[0].sin()),
                Function::Cos => Some(arg_values[0].cos()),
                Function::Tan => Some(arg_values[0].tan()),
                Function::Asin => Some(arg_values[0].asin()),
                Function::Acos => Some(arg_values[0].acos()),
                Function::Atan => Some(arg_values[0].atan()),
                Function::Sinh => Some(arg_values[0].sinh()),
                Function::Cosh => Some(arg_values[0].cosh()),
                Function::Tanh => Some(arg_values[0].tanh()),
                Function::Exp => Some(arg_values[0].exp()),
                Function::Ln => if arg_values[0] > 0.0 { Some(arg_values[0].ln()) } else { None },
                Function::Log => {
                    if arg_values.len() == 2 && arg_values[0] > 0.0 && arg_values[1] > 0.0 {
                        Some(arg_values[1].log(arg_values[0]))
                    } else if arg_values.len() == 1 && arg_values[0] > 0.0 {
                        Some(arg_values[0].log10())
                    } else {
                        None
                    }
                }
                Function::Sqrt => if arg_values[0] >= 0.0 { Some(arg_values[0].sqrt()) } else { None },
                Function::Abs => Some(arg_values[0].abs()),
                Function::Floor => Some(arg_values[0].floor()),
                Function::Ceil => Some(arg_values[0].ceil()),
                Function::Round => Some(arg_values[0].round()),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Simplify an expression.
fn simplify_ffi(expression: &str) -> Result<ffi::SimplificationResultFFI, String> {
    use crate::parser::parse_expression;

    let expr = parse_expression(expression)
        .map_err(|e| format!("Parse error: {:?}", e))?;

    let simplified = expr.simplify();

    Ok(ffi::SimplificationResultFFI {
        original: expression.to_string(),
        simplified: format!("{}", simplified),
        simplified_latex: simplified.to_latex(),
    })
}

/// Simplify trigonometric expression.
fn simplify_trig_ffi(expression: &str) -> Result<ffi::SimplificationResultFFI, String> {
    use crate::parser::parse_expression;
    use crate::trigonometric::simplify_trig;

    let expr = parse_expression(expression)
        .map_err(|e| format!("Parse error: {:?}", e))?;

    let simplified = simplify_trig(&expr);

    Ok(ffi::SimplificationResultFFI {
        original: expression.to_string(),
        simplified: format!("{}", simplified),
        simplified_latex: simplified.to_latex(),
    })
}

/// Simplify trigonometric expression with steps.
fn simplify_trig_with_steps_ffi(expression: &str) -> Result<String, String> {
    use crate::parser::parse_expression;
    use crate::trigonometric::simplify_trig_with_steps;

    let expr = parse_expression(expression)
        .map_err(|e| format!("Parse error: {:?}", e))?;

    let (simplified, steps) = simplify_trig_with_steps(&expr);

    let result = serde_json::json!({
        "original": expression,
        "simplified": format!("{}", simplified),
        "simplified_latex": simplified.to_latex(),
        "steps": steps
    });

    serde_json::to_string(&result)
        .map_err(|e| format!("Failed to serialize result: {}", e))
}

/// Solve a system of linear equations.
fn solve_system_ffi(equations_json: &str) -> Result<String, String> {
    use crate::ast::Variable;
    use crate::solver::{LinearSystem, SystemSolver};

    let equations: Vec<String> = serde_json::from_str(equations_json)
        .map_err(|e| format!("Failed to parse equations JSON: {}", e))?;

    let mut parsed_equations = Vec::new();
    for eq_str in &equations {
        let eq = crate::parser::parse_equation(eq_str)
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

    let system = LinearSystem::new(parsed_equations, variables);
    let solver = SystemSolver::new();

    match solver.solve(&system) {
        Ok(solution) => {
            let result: std::collections::HashMap<String, String> = solution
                .solutions
                .iter()
                .map(|(var, expr)| (var.name.clone(), format!("{}", expr)))
                .collect();
            serde_json::to_string(&result)
                .map_err(|e| format!("Failed to serialize result: {}", e))
        }
        Err(e) => Err(format!("Failed to solve system: {:?}", e)),
    }
}

/// Helper to collect variable names from expression.
fn collect_variables(expr: &crate::ast::Expression, vars: &mut std::collections::HashSet<String>) {
    use crate::ast::Expression;
    match expr {
        Expression::Variable(v) => { vars.insert(v.name.clone()); }
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
    use crate::ast::Variable;
    use crate::inequality::{solve_inequality, Inequality};
    use crate::parser::parse_expression;

    // Parse inequality (expects format like "expr < value" or "expr > value")
    let parts: Vec<&str> = if inequality.contains("<=") {
        vec![&inequality[..inequality.find("<=").unwrap()], &inequality[inequality.find("<=").unwrap()+2..], "<="]
    } else if inequality.contains(">=") {
        vec![&inequality[..inequality.find(">=").unwrap()], &inequality[inequality.find(">=").unwrap()+2..], ">="]
    } else if inequality.contains("<") {
        vec![&inequality[..inequality.find('<').unwrap()], &inequality[inequality.find('<').unwrap()+1..], "<"]
    } else if inequality.contains(">") {
        vec![&inequality[..inequality.find('>').unwrap()], &inequality[inequality.find('>').unwrap()+1..], ">"]
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

    match solve_inequality(&ineq, &Variable::new(variable)) {
        Ok(solution) => Ok(format!("{:?}", solution)),
        Err(e) => Err(format!("Failed to solve inequality: {:?}", e)),
    }
}

/// Get the LaTeX representation of an expression.
fn to_latex_ffi(expression: &str) -> Result<String, String> {
    use crate::parser::parse_expression;

    let expr = parse_expression(expression)
        .map_err(|e| format!("Parse error: {:?}", e))?;

    Ok(expr.to_latex())
}

/// Partial fraction decomposition.
fn partial_fractions_ffi(numerator: &str, denominator: &str, variable: &str) -> Result<String, String> {
    use crate::ast::Variable;
    use crate::parser::parse_expression;
    use crate::partial_fractions::decompose;

    let num = parse_expression(numerator)
        .map_err(|e| format!("Parse error in numerator: {:?}", e))?;
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
            serde_json::to_string(&output)
                .map_err(|e| format!("Failed to serialize result: {}", e))
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
    use crate::parser::parse_equation;
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
    let solution = solver.solve(&system, &context)
        .map_err(|e| format!("Failed to solve system: {}", e))?;

    // Build the result JSON
    let mut solutions_map: HashMap<String, serde_json::Value> = HashMap::new();
    for (var, val) in &solution.solutions {
        if let Some(num) = val.as_numeric() {
            solutions_map.insert(var.clone(), serde_json::json!(num));
        } else {
            solutions_map.insert(var.clone(), serde_json::json!(format!("{}", val.to_expression())));
        }
    }

    // Build step descriptions
    let steps: Vec<serde_json::Value> = solution.resolution_path.steps.iter().map(|step| {
        serde_json::json!({
            "step_number": step.step_number,
            "equation_id": step.equation_id,
            "operation": format!("{}", step.operation),
            "explanation": step.explanation
        })
    }).collect();

    let output = serde_json::json!({
        "solutions": solutions_map,
        "steps": steps,
        "unsolved": solution.unsolved,
        "warnings": solution.warnings,
        "is_complete": solution.is_complete()
    });

    serde_json::to_string(&output)
        .map_err(|e| format!("Failed to serialize result: {}", e))
}

// TODO: Add async FFI support for long-running operations
// TODO: Add callback support for progress updates
// TODO: Add memory-safe buffer passing for large data
