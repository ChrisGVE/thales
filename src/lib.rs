//! MathSolver Core - Mathematical equation parsing, solving, and transformations.
//!
//! A high-performance Rust library providing comprehensive mathematical capabilities for
//! equation solving, coordinate transformations, and numerical analysis. Designed for
//! cross-platform use with first-class iOS support via FFI bindings.
//!
//! # Executive Summary
//!
//! **Primary Use Cases:**
//! - Parse and solve algebraic equations symbolically
//! - Perform coordinate system transformations (2D/3D)
//! - Numerical root-finding for transcendental equations
//! - Track step-by-step solution paths for educational applications
//! - Dimensional analysis and unit conversion for physics calculations
//! - Swift interoperability for iOS/macOS applications
//!
//! **Key Strengths:**
//! - Zero-cost abstractions with compile-time guarantees
//! - Memory-safe implementation (no unsafe code except FFI boundary)
//! - Comprehensive test coverage including property-based tests
//! - Optimized for embedded and mobile targets (iOS)
//! - Clear separation between symbolic and numerical methods
//!
//! # Quick Start
//!
//! ## Example 1: Coordinate Transformations
//!
//! Convert between Cartesian and polar coordinate systems:
//!
//! ```rust
//! use mathsolver_core::{Cartesian2D, Polar};
//!
//! // 2D Cartesian to Polar
//! let cartesian = Cartesian2D::new(3.0, 4.0);
//! let polar = cartesian.to_polar();
//! assert!((polar.r - 5.0).abs() < 1e-10);
//! assert!((polar.theta - 0.927295218).abs() < 1e-6);
//!
//! // Polar to Cartesian round-trip
//! let back = polar.to_cartesian();
//! assert!((back.x - 3.0).abs() < 1e-10);
//! assert!((back.y - 4.0).abs() < 1e-10);
//! ```
//!
//! ## Example 2: 3D Coordinate Transformations
//!
//! Convert between Cartesian and spherical coordinates:
//!
//! ```rust
//! use mathsolver_core::{Cartesian3D, Spherical};
//!
//! // Cartesian to Spherical
//! let cart3d = Cartesian3D::new(1.0, 1.0, 1.0);
//! let spherical = cart3d.to_spherical();
//! assert!((spherical.r - 1.732050808).abs() < 1e-6);
//!
//! // Spherical to Cartesian round-trip
//! let back = spherical.to_cartesian();
//! assert!((back.x - 1.0).abs() < 1e-10);
//! assert!((back.y - 1.0).abs() < 1e-10);
//! assert!((back.z - 1.0).abs() < 1e-10);
//! ```
//!
//! ## Example 3: Complex Number Operations
//!
//! Work with complex numbers and polar form:
//!
//! ```rust
//! use mathsolver_core::ComplexOps;
//! use num_complex::Complex64;
//!
//! // De Moivre's theorem: (r∠θ)^n = r^n∠(nθ)
//! let z = Complex64::new(1.0, 1.0);
//! let result = ComplexOps::de_moivre(z, 2.0);
//!
//! // Complex conjugate (using num_complex methods)
//! let conj = z.conj();
//! assert_eq!(conj.re, z.re);
//! assert_eq!(conj.im, -z.im);
//!
//! // Modulus (magnitude) of complex number
//! let modulus = z.norm();
//! assert!((modulus - 1.4142135623730951).abs() < 1e-10);
//! ```
//!
//! ## Example 4: Expression and Variable Basics
//!
//! Build mathematical expressions using the AST:
//!
//! ```rust
//! use mathsolver_core::{Expression, Variable, BinaryOp};
//!
//! // Create expression: 2*x + 5
//! let x = Variable::new("x");
//! let two_x = Expression::Binary(
//!     BinaryOp::Mul,
//!     Box::new(Expression::Integer(2)),
//!     Box::new(Expression::Variable(x.clone()))
//! );
//! let expr = Expression::Binary(
//!     BinaryOp::Add,
//!     Box::new(two_x),
//!     Box::new(Expression::Integer(5))
//! );
//!
//! // Check variable containment
//! assert!(expr.contains_variable("x"));
//! assert!(!expr.contains_variable("y"));
//! ```
//!
//! # Feature Matrix
//!
//! | Feature Category | Component | Status | Notes |
//! |-----------------|-----------|--------|-------|
//! | **Parsing** | Expression parser | In Progress | Using chumsky parser combinators |
//! | | Equation parser | In Progress | Full operator precedence support |
//! | | Function parsing | Planned | Trig, log, exp functions |
//! | **Symbolic Solving** | Linear equations | Implemented | ax + b = c form |
//! | | Quadratic equations | Implemented | ax² + bx + c = 0 with discriminant |
//! | | Polynomial equations | In Progress | General polynomial solving |
//! | | Transcendental | In Progress | Equations with trig/exp functions |
//! | **Numerical Methods** | Newton-Raphson | In Progress | With symbolic differentiation |
//! | | Bisection | Planned | Guaranteed convergence |
//! | | Brent's method | Planned | Hybrid root-finding |
//! | | Smart solver | Implemented | Automatic method selection |
//! | **Transformations** | 2D Cartesian ↔ Polar | **Complete** | Fully tested |
//! | | 3D Cartesian ↔ Spherical | **Complete** | Fully tested |
//! | | 3D Cartesian ↔ Cylindrical | **Complete** | Fully tested |
//! | | Complex polar form | **Complete** | De Moivre, conjugate, modulus |
//! | | Homogeneous transforms | Stubbed | 2D transformation matrices |
//! | **Dimensions** | SI base units | Stubbed | Length, mass, time, etc. |
//! | | Derived units | Stubbed | Velocity, force, energy |
//! | | Unit conversion | Planned | Automatic conversion |
//! | | Dimensional analysis | Planned | Type-safe dimension checking |
//! | **Resolution Paths** | Step tracking | In Progress | Educational solution paths |
//! | | Operation recording | In Progress | For UI display |
//! | **FFI** | Swift bindings | Implemented | via swift-bridge |
//! | | iOS targets | **Complete** | aarch64/x86_64 simulator + device |
//!
//! # Architecture Overview
//!
//! The library follows a modular design with clear separation of concerns:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    Public API Layer                     │
//! │  (parse_equation, SmartSolver, Cartesian2D, etc.)      │
//! └─────────────────────────────────────────────────────────┘
//!                            │
//!           ┌────────────────┼────────────────┐
//!           ▼                ▼                ▼
//!    ┌──────────┐     ┌──────────┐    ┌──────────┐
//!    │  Parser  │     │  Solver  │    │Transform │
//!    │ (chumsky)│     │ (symbolic)│    │(nalgebra)│
//!    └──────────┘     └──────────┘    └──────────┘
//!           │                │                │
//!           └────────────────┼────────────────┘
//!                            ▼
//!                     ┌──────────┐
//!                     │   AST    │
//!                     │(equation,│
//!                     │  expr,   │
//!                     │variable) │
//!                     └──────────┘
//!                            │
//!           ┌────────────────┼────────────────┐
//!           ▼                ▼                ▼
//!    ┌──────────┐     ┌──────────┐    ┌──────────┐
//!    │Numerical │     │Resolution│    │Dimensions│
//!    │ (argmin) │     │   Path   │    │  (units) │
//!    └──────────┘     └──────────┘    └──────────┘
//!                            │
//!                            ▼
//!                     ┌──────────┐
//!                     │   FFI    │
//!                     │  (Swift) │
//!                     └──────────┘
//! ```
//!
//! ## Module Responsibilities
//!
//! - [`ast`]: Core data structures for mathematical expressions, equations, variables,
//!   operators, and functions. All other modules build upon these types.
//!
//! - [`parser`]: String → AST conversion using the chumsky parser combinator library.
//!   Handles operator precedence, function calls, and complex number literals.
//!
//! - [`solver`]: Symbolic equation solving using algebraic manipulation. Includes
//!   specialized solvers for linear, quadratic, polynomial, and transcendental equations.
//!   The [`SmartSolver`] automatically dispatches to the appropriate solver.
//!
//! - [`numerical`]: Numerical root-finding methods for equations that cannot be solved
//!   symbolically. Integrates with symbolic differentiation from the AST module.
//!
//! - [`resolution_path`]: Records the step-by-step solution process for educational
//!   applications. Each transformation is recorded with its operation type.
//!
//! - [`dimensions`]: Dimensional analysis and unit conversion. Ensures physical
//!   equations maintain dimensional consistency.
//!
//! - [`transforms`]: Coordinate system conversions (Cartesian, Polar, Spherical,
//!   Cylindrical) and complex number operations. Built on nalgebra for linear algebra.
//!
//! - `ffi`: Foreign function interface for Swift via swift-bridge. Provides
//!   C-compatible bindings for iOS/macOS integration. Enabled with the `ffi` feature flag.
//!
//! # Safety Guarantees
//!
//! This library adheres to strict memory safety principles:
//!
//! - **No unsafe code in core logic**: All mathematical operations, parsing, solving,
//!   and transformations use only safe Rust.
//!
//! - **FFI boundary isolation**: The only `unsafe` code appears in the `ffi` module
//!   for C interoperability, which is:
//!   - Isolated behind the `ffi` feature flag
//!   - Managed by the swift-bridge library
//!   - Validated at the FFI boundary with explicit error handling
//!
//! - **Ownership guarantees**: The Rust type system prevents:
//!   - Use-after-free bugs
//!   - Data races in concurrent access
//!   - Null pointer dereferences
//!   - Buffer overflows
//!
//! - **Integer overflow protection**: All arithmetic operations use checked or saturating
//!   semantics where appropriate.
//!
//! - **Thread safety**: All public types are `Send + Sync` where semantically appropriate,
//!   with compile-time verification.
//!
//! # Performance Characteristics
//!
//! ## Time Complexity Guarantees
//!
//! | Operation | Complexity | Notes |
//! |-----------|-----------|-------|
//! | Parse expression | O(n) | Linear in input string length |
//! | Variable lookup | O(1) | HashMap-based symbol table |
//! | Coordinate transform | O(1) | Fixed number of trig operations |
//! | Linear solve | O(1) | Constant number of operations |
//! | Quadratic solve | O(1) | Discriminant calculation + sqrt |
//! | Polynomial solve (degree d) | O(d²) | Companion matrix method |
//! | Numerical solve (Newton) | O(k) | k iterations to convergence |
//!
//! ## Space Complexity
//!
//! - **AST storage**: O(n) where n is the number of expression nodes
//! - **Resolution path**: O(k) where k is the number of solution steps
//! - **Parser stack**: O(d) where d is maximum nesting depth
//!
//! ## Optimization Features
//!
//! Build configuration in release mode enables:
//!
//! - **Link-Time Optimization (LTO)**: Cross-module inlining and dead code elimination
//! - **Single codegen unit**: Maximum optimization at cost of compile time
//! - **opt-level=3**: Aggressive compiler optimizations
//! - **Zero-cost abstractions**: Generic functions specialized at compile time
//! - **SIMD auto-vectorization**: Compiler-generated vectorized code where applicable
//!
//! ```toml
//! [profile.release]
//! opt-level = 3
//! lto = true
//! codegen-units = 1
//! ```
//!
//! Benchmark your performance-critical paths with:
//!
//! ```bash
//! cargo bench
//! ```
//!
//! # Platform Support
//!
//! ## Tier 1: Fully Supported
//!
//! - **iOS Devices** (`aarch64-apple-ios`): Native ARM64 execution on iPhone/iPad
//! - **iOS Simulator on ARM** (`aarch64-apple-ios-sim`): M1/M2/M3 Mac simulator
//! - **iOS Simulator on Intel** (`x86_64-apple-ios`): Intel Mac simulator
//!
//! Build for all iOS targets:
//!
//! ```bash
//! # Add targets (one-time setup)
//! rustup target add aarch64-apple-ios
//! rustup target add aarch64-apple-ios-sim
//! rustup target add x86_64-apple-ios
//!
//! # Build for device
//! cargo build --release --target aarch64-apple-ios
//!
//! # Build for simulator (ARM)
//! cargo build --release --target aarch64-apple-ios-sim
//!
//! # Build universal library with lipo
//! lipo -create \
//!   target/aarch64-apple-ios-sim/release/libmathsolver_core.a \
//!   target/x86_64-apple-ios/release/libmathsolver_core.a \
//!   -output libmathsolver_core_universal.a
//! ```
//!
//! ## Tier 2: Standard Rust Targets
//!
//! The library uses only stable Rust features and should compile on any tier 1 Rust platform:
//! - Linux (x86_64, aarch64)
//! - macOS (x86_64, aarch64)
//! - Windows (x86_64)
//!
//! ## FFI Integration
//!
//! Enable Swift bindings with the `ffi` feature:
//!
//! ```bash
//! cargo build --release --features ffi --target aarch64-apple-ios
//! ```
//!
//! This generates Swift bridge code and C headers for Xcode integration.
//!
//! # Version History
//!
//! ## 0.1.0 (2025-12-17) - Initial Release
//!
//! **Implemented:**
//! - Core AST definitions (Expression, Equation, Variable)
//! - Coordinate transformations (2D/3D) with full test coverage
//! - Complex number operations (De Moivre, conjugate, modulus)
//! - Linear and quadratic equation solvers
//! - Smart solver with automatic method dispatch
//! - iOS cross-compilation support
//! - FFI bindings infrastructure
//! - Test and benchmark framework
//!
//! **In Progress:**
//! - Expression parser using chumsky
//! - Polynomial and transcendental solvers
//! - Numerical methods with symbolic differentiation
//! - Resolution path generation
//! - Unit and dimension system
//!
//! See [CHANGELOG.md](https://github.com/cberclaz/mathsolver-core/blob/main/CHANGELOG.md)
//! for detailed version history.
//!
//! # Module Documentation
//!
//! Explore detailed documentation for each module:
//!
//! - [`ast`] - Abstract syntax tree types
//! - [`parser`] - String parsing to AST
//! - [`solver`] - Symbolic equation solving
//! - [`numerical`] - Numerical approximation methods
//! - [`resolution_path`] - Solution step tracking
//! - [`dimensions`] - Units and dimensional analysis
//! - [`transforms`] - Coordinate system conversions
//! - `ffi` - Swift interoperability (requires `ffi` feature)
//!
//! # Development Status
//!
//! This is version 0.1.0 with foundational infrastructure in place. The coordinate
//! transformation and complex number modules are production-ready. Parser, solver,
//! and numerical modules are under active development with many features stubbed
//! with TODO comments.
//!
//! Run the test suite to see current implementation status:
//!
//! ```bash
//! cargo test
//! cargo test --doc        # Documentation examples
//! cargo test -- --ignored # Requires full implementation
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

// Public module exports
pub mod ast;
pub mod dimensions;
pub mod latex;
pub mod limits;
pub mod numerical;
pub mod parser;
pub mod pattern;
pub mod resolution_path;
pub mod solver;
pub mod transforms;
pub mod matrix;
pub mod precision;
pub mod integration;
pub mod inequality;

// FFI module (conditionally compiled for FFI builds)
#[cfg(feature = "ffi")]
pub mod ffi;

// Re-export commonly used types at crate root for convenience
pub use ast::{BinaryOp, Equation, Expression, Function, UnaryOp, Variable};
pub use dimensions::{Dimension, Quantity, Unit, UnitRegistry};
pub use numerical::{NumericalConfig, NumericalSolution, SmartNumericalSolver};
pub use latex::{parse_latex, parse_latex_equation};
pub use parser::{parse_equation, parse_expression};
pub use resolution_path::{Operation, ResolutionPath, ResolutionStep};
pub use solver::{SmartSolver, Solution, Solver};
pub use transforms::{
    Cartesian2D, Cartesian3D, ComplexOps, Cylindrical, Polar, Spherical, Transform2D,
};
pub use matrix::{MatrixExpr, MatrixError, BracketStyle};
pub use precision::{EvalContext, EvalError, PrecisionMode, RoundingMode, Value};
pub use integration::{
    integrate, integrate_by_substitution, integrate_with_substitution, IntegrationError,
};
pub use inequality::{
    solve_inequality, solve_system, Bound, Inequality, InequalityError, IntervalSolution,
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
