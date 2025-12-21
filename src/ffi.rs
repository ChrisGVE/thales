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

// TODO: Add FFI for unit conversions
// TODO: Add FFI for expression evaluation
// TODO: Add FFI for expression simplification
// TODO: Add async FFI support for long-running operations
// TODO: Add callback support for progress updates
// TODO: Add memory-safe buffer passing for large data
