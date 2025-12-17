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
        fn solve_numerically_ffi(
            equation: &str,
            variable: &str,
            initial_guess: f64,
        ) -> Result<f64, String>;
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
    // TODO: Implement equation solving pipeline
    // 1. Parse equation
    // 2. Parse variable
    // 3. Solve using SmartSolver
    // 4. Format solution as string
    Err("Not yet implemented".to_string())
}

/// Solve equation numerically.
fn solve_numerically_ffi(
    equation: &str,
    variable: &str,
    initial_guess: f64,
) -> Result<f64, String> {
    // TODO: Implement numerical solving pipeline
    // 1. Parse equation
    // 2. Parse variable
    // 3. Solve using SmartNumericalSolver with initial guess
    // 4. Return solution value
    Err("Not yet implemented".to_string())
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

// TODO: Add FFI for unit conversions
// TODO: Add FFI for expression evaluation
// TODO: Add FFI for expression simplification
// TODO: Add FFI for getting resolution path steps
// TODO: Add error types that map cleanly to Swift
// TODO: Add async FFI support for long-running operations
// TODO: Add callback support for progress updates
// TODO: Add memory-safe buffer passing for large data
