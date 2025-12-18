//! FFI exports for Swift interoperability using swift-bridge.
//!
//! Provides C-compatible bindings for use from Swift via swift-bridge.

#[swift_bridge::bridge]
mod ffi {
    /// Parse an equation from string.
    extern "Rust" {
        #[swift_bridge(return_with = Result)]
        fn parse_equation_ffi(input: &str) -> Result<String, String>;

        #[swift_bridge(return_with = Result)]
        fn parse_expression_ffi(input: &str) -> Result<String, String>;
    }

    /// Solve an equation for a variable.
    extern "Rust" {
        #[swift_bridge(return_with = Result)]
        fn solve_equation_ffi(
            equation: &str,
            variable: &str,
        ) -> Result<String, String>;

        #[swift_bridge(return_with = Result)]
        fn solve_with_values_ffi(
            equation: &str,
            variable: &str,
            known_values_json: &str,
        ) -> Result<ResolutionPathFFI, String>;

        #[swift_bridge(return_with = Result)]
        fn solve_numerically_ffi(
            equation: &str,
            variable: &str,
            initial_guess: f64,
        ) -> Result<f64, String>;
    }

    /// Resolution path with steps (for Swift).
    #[swift_bridge(swift_repr = "struct")]
    pub struct ResolutionPathFFI {
        pub initial_expr: String,
        pub steps_json: String,
        pub result_expr: String,
        pub success: bool,
    }

    /// Coordinate transformations.
    extern "Rust" {
        fn cartesian_to_polar_ffi(x: f64, y: f64) -> PolarCoords;
        fn polar_to_cartesian_ffi(r: f64, theta: f64) -> CartesianCoords2D;

        fn cartesian_to_spherical_ffi(x: f64, y: f64, z: f64) -> SphericalCoords;
        fn spherical_to_cartesian_ffi(r: f64, theta: f64, phi: f64) -> CartesianCoords3D;
    }

    /// Complex number operations.
    extern "Rust" {
        fn complex_add_ffi(a_re: f64, a_im: f64, b_re: f64, b_im: f64) -> ComplexNumber;
        fn complex_multiply_ffi(a_re: f64, a_im: f64, b_re: f64, b_im: f64) -> ComplexNumber;
        fn complex_to_polar_ffi(re: f64, im: f64) -> PolarCoords;
        fn complex_power_ffi(re: f64, im: f64, n: f64) -> ComplexNumber;
    }

    /// Swift-visible types for coordinate results.
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
    let parsed_equation = parse_equation(equation)
        .map_err(|e| format!("Failed to parse equation: {:?}", e))?;

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
    use crate::parser::parse_equation;
    use crate::numerical::{NumericalConfig, SmartNumericalSolver};
    use crate::ast::Variable;

    // Parse the equation string
    let parsed_equation = parse_equation(equation)
        .map_err(|e| format!("Failed to parse equation: {:?}", e))?;

    // Create solver with initial guess
    let mut config = NumericalConfig::default();
    config.initial_guess = Some(initial_guess);
    let solver = SmartNumericalSolver::new(config);

    // Solve numerically
    let target_var = Variable::new(variable);
    let (solution, _path) = solver.solve(&parsed_equation, &target_var)
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
    let parsed_equation = parse_equation(equation)
        .map_err(|e| format!("Failed to parse equation: {:?}", e))?;

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

    let steps_json = serde_json::to_string(&steps)
        .map_err(|e| format!("Failed to serialize steps: {}", e))?;

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
