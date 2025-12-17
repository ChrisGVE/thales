//! MathSolver Core - Mathematical equation parsing, solving, and transformations.
//!
//! A Rust library providing comprehensive mathematical capabilities including:
//! - Expression and equation parsing with full operator precedence
//! - Symbolic equation solving (linear, quadratic, polynomial, transcendental)
//! - Numerical approximation methods (Newton-Raphson, Brent's, etc.)
//! - Coordinate system transformations (Cartesian, Polar, Spherical, Complex)
//! - Unit and dimension handling with automatic conversion
//! - Resolution path tracking for step-by-step solutions
//! - FFI bindings for Swift interoperability
//!
//! # Examples
//!
//! ```ignore
//! use mathsolver_core::parser::parse_equation;
//! use mathsolver_core::solver::{SmartSolver, Solver};
//! use mathsolver_core::ast::Variable;
//!
//! // Parse and solve an equation
//! let equation = parse_equation("2*x + 5 = 13").unwrap();
//! let solver = SmartSolver::new();
//! let (solution, path) = solver.solve(&equation, &Variable::new("x")).unwrap();
//! ```
//!
//! # Features
//!
//! ## Parsing
//! - Mathematical expressions with full operator precedence
//! - Functions (trigonometric, logarithmic, exponential)
//! - Complex numbers and rational numbers
//! - Variables with optional dimensions
//!
//! ## Solving
//! - Symbolic solving for linear, quadratic, and polynomial equations
//! - Numerical methods for transcendental equations
//! - System of equations support
//! - Step-by-step solution paths
//!
//! ## Transformations
//! - 2D: Cartesian ↔ Polar
//! - 3D: Cartesian ↔ Spherical ↔ Cylindrical
//! - Complex number operations with polar form
//! - Rotation matrices and homogeneous transformations
//!
//! ## Units and Dimensions
//! - SI base and derived units
//! - Automatic unit conversion
//! - Dimensional analysis validation
//! - Physical quantity operations
//!
//! # Architecture
//!
//! The library is organized into modules by functionality:
//!
//! - `ast`: Abstract syntax tree definitions
//! - `parser`: Expression and equation parsing
//! - `solver`: Symbolic equation solving
//! - `numerical`: Numerical approximation methods
//! - `resolution_path`: Solution step tracking
//! - `dimensions`: Unit and dimension handling
//! - `transforms`: Coordinate transformations
//! - `ffi`: Foreign function interface for Swift
//!
//! # Platform Support
//!
//! Designed for cross-platform use with specific support for iOS targets:
//! - `aarch64-apple-ios` (iOS devices)
//! - `x86_64-apple-ios` (iOS simulator on Intel)
//! - `aarch64-apple-ios-sim` (iOS simulator on ARM)

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

// Public module exports
pub mod ast;
pub mod dimensions;
pub mod numerical;
pub mod parser;
pub mod resolution_path;
pub mod solver;
pub mod transforms;

// FFI module (conditionally compiled for FFI builds)
#[cfg(feature = "ffi")]
pub mod ffi;

// Re-export commonly used types at crate root for convenience
pub use ast::{BinaryOp, Equation, Expression, Function, UnaryOp, Variable};
pub use dimensions::{Dimension, Quantity, Unit, UnitRegistry};
pub use numerical::{NumericalConfig, NumericalSolution, SmartNumericalSolver};
pub use parser::{parse_equation, parse_expression};
pub use resolution_path::{Operation, ResolutionPath, ResolutionStep};
pub use solver::{Solver, SmartSolver, Solution};
pub use transforms::{
    Cartesian2D, Cartesian3D, ComplexOps, Cylindrical, Polar, Spherical, Transform2D,
};

/// Library version information.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library name.
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Get library version string.
pub fn version() -> &'static str {
    VERSION
}

/// Check if library is compiled with FFI support.
pub fn has_ffi_support() -> bool {
    cfg!(feature = "ffi")
}

// TODO: Add prelude module with commonly used imports
// TODO: Add error types module with unified error handling
// TODO: Add traits module with common trait definitions
// TODO: Add macro module for expression DSL
// TODO: Add serde support for serialization
// TODO: Add wasm support for web usage
// TODO: Add Python bindings via PyO3
// TODO: Add comprehensive integration tests
// TODO: Add performance benchmarks
// TODO: Add documentation examples that compile and run
