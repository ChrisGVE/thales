//! # Thales - Computer Algebra System
//!
//! A comprehensive Computer Algebra System (CAS) library for symbolic mathematics,
//! equation solving, calculus, and numerical methods. Named after
//! [Thales of Miletus](https://en.wikipedia.org/wiki/Thales_of_Miletus),
//! the first mathematician in the Greek tradition.
//!
//! ## Overview
//!
//! Thales provides:
//! - **Symbolic equation solving** - Linear, quadratic, polynomial, transcendental, and systems
//! - **Calculus** - Differentiation, integration, limits, Taylor series, first and second-order ODEs
//! - **Numerical methods** - Newton-Raphson, bisection, Brent's method, Levenberg-Marquardt
//! - **Coordinate systems** - 2D/3D transformations, complex numbers, De Moivre's theorem
//! - **Units & dimensions** - Dimensional analysis and unit conversion
//! - **Step-by-step solutions** - Resolution paths for educational applications
//! - **iOS/macOS support** - FFI bindings via swift-bridge
//!
//! ## Key Features
//!
//! - Zero-cost abstractions with compile-time guarantees
//! - Memory-safe implementation (no unsafe code except FFI boundary)
//! - 970+ tests including property-based tests with proptest
//! - Optimized for mobile targets (iOS)
//! - Clear separation between symbolic and numerical methods
//!
//! # Quick Start
//!
//! ## Example 1: Coordinate Transformations
//!
//! Convert between Cartesian and polar coordinate systems:
//!
//! ```rust
//! use thales::{Cartesian2D, Polar};
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
//! use thales::{Cartesian3D, Spherical};
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
//! use thales::ComplexOps;
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
//! use thales::{Expression, Variable, BinaryOp};
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
//! ## User Guides
//!
//! For detailed tutorials and workflows, see the [`guides`] module:
//!
//! - [`guides::solving_equations`] - Linear, quadratic, polynomial, and systems
//! - [`guides::calculus_operations`] - Derivatives, integrals, limits, ODEs
//! - [`guides::series_expansions`] - Taylor, Maclaurin, Laurent, asymptotic
//! - [`guides::coordinate_systems`] - 2D/3D transforms, complex numbers
//! - [`guides::numerical_methods`] - Root-finding when symbolic fails
//! - [`guides::working_with_units`] - Dimensional analysis
//! - [`guides::error_handling`] - Working with [`ThalesError`]
//!
//! ## Feature Summary
//!
//! | Category | Features |
//! |----------|----------|
//! | **Parsing** | Expression parser, equation parser, LaTeX input/output |
//! | **Solving** | Linear, quadratic, polynomial, transcendental, multi-equation systems |
//! | **Calculus** | Differentiation, integration (by parts, substitution), limits, L'Hôpital |
//! | **ODEs** | First-order (separable, linear), second-order (constant coefficient) |
//! | **Series** | Taylor, Maclaurin, Laurent, asymptotic expansions, Big-O |
//! | **Special** | Gamma, beta, error functions with derivation steps |
//! | **Numerical** | Newton-Raphson, bisection, Brent's, secant, Levenberg-Marquardt |
//! | **Transforms** | Cartesian ↔ Polar ↔ Spherical ↔ Cylindrical, complex operations |
//! | **Units** | SI base/derived units, dimensional analysis, conversions |
//! | **FFI** | Swift bindings via swift-bridge, iOS device + simulator |
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
//!   target/aarch64-apple-ios-sim/release/libthales.a \
//!   target/x86_64-apple-ios/release/libthales.a \
//!   -output libthales_universal.a
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
//! ## Version History
//!
//! **Current: v0.3.0** - Advanced Calculus & API Stabilization
//!
//! - Second-order ODEs with characteristic equation method
//! - Nonlinear system solver (Newton-Raphson for systems)
//! - Taylor, Maclaurin, Laurent series expansions
//! - Asymptotic expansions with Big-O notation
//! - Special functions (gamma, beta, erf, erfc)
//! - Small angle approximations with error bounds
//! - Unified [`ThalesError`] type
//! - 970+ tests including property-based tests
//!
//! See [CHANGELOG.md](https://github.com/ChrisGVE/thales/blob/main/CHANGELOG.md)
//! for complete version history.
//!
//! ## Module Reference
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`ast`] | Abstract syntax tree types for expressions and equations |
//! | [`parser`] | String → AST conversion with chumsky |
//! | [`solver`] | Symbolic equation solving |
//! | [`equation_system`] | Multi-equation system solver |
//! | [`numerical`] | Numerical root-finding methods |
//! | [`integration`] | Symbolic integration |
//! | [`limits`] | Limit evaluation with L'Hôpital's rule |
//! | [`ode`] | First and second-order ODE solving |
//! | [`series`] | Taylor, Laurent, asymptotic series |
//! | [`special`] | Gamma, beta, error functions |
//! | [`approximations`] | Small angle and scaled approximations |
//! | [`transforms`] | Coordinate system conversions |
//! | [`dimensions`] | Units and dimensional analysis |
//! | [`pattern`] | Rule-based expression rewriting |
//! | [`latex`] | LaTeX parsing and rendering |
//! | [`resolution_path`] | Step-by-step solution tracking |
//! | [`guides`] | User tutorials and workflows |
//!
//! ## Running Tests
//!
//! ```bash
//! cargo test              # All tests
//! cargo test --doc        # Documentation examples only
//! cargo test --release    # Optimized build
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
pub mod trigonometric;
pub mod ode;
pub mod partial_fractions;
pub mod equation_system;
pub mod series;
pub mod optimization;
pub mod special;
pub mod approximations;

// User guides for common workflows
pub mod guides;

// FFI module (conditionally compiled for FFI builds)
#[cfg(feature = "ffi")]
pub mod ffi;

// Re-export commonly used types at crate root for convenience
pub use ast::{BinaryOp, Equation, Expression, Function, UnaryOp, Variable};
pub use dimensions::{Dimension, Quantity, Unit, UnitRegistry};
pub use numerical::{NumericalConfig, NumericalSolution, SmartNumericalSolver};
pub use latex::{parse_latex, parse_latex_equation};
pub use parser::{parse_equation, parse_expression};
pub use resolution_path::{
    Operation, OperationCounts, PathStatistics, ResolutionPath, ResolutionPathBuilder,
    ResolutionStep, Verbosity,
};
pub use solver::{
    LinearSystem, SmartSolver, Solution, Solver, SystemSolution, SystemSolver,
};
pub use transforms::{
    Cartesian2D, Cartesian3D, ComplexOps, Cylindrical, Polar, Spherical, Transform2D,
};
pub use matrix::{MatrixExpr, MatrixError, BracketStyle};
pub use precision::{EvalContext, EvalError, PrecisionMode, RoundingMode, Value};
pub use integration::{
    definite_integral, definite_integral_with_fallback, definite_integral_with_steps,
    improper_integral_to_infinity, integrate, integrate_by_parts, integrate_by_parts_with_steps,
    integrate_by_substitution, integrate_with_substitution, numerical_integrate,
    tabular_integration, IntegrationError,
};
pub use inequality::{
    solve_inequality, solve_system, Bound, Inequality, InequalityError, IntervalSolution,
};
pub use trigonometric::{
    all_trig_rules, double_angle_rules, inverse_rules, parity_rules, product_to_sum_rules,
    pythagorean_rules, quotient_rules, simplify_double_angle, simplify_pythagorean,
    simplify_quotient, simplify_trig, simplify_trig_step, simplify_trig_with_steps,
    special_value_rules,
};
pub use ode::{
    solve_ivp, solve_linear, solve_separable, solve_second_order_homogeneous,
    solve_second_order_ivp, solve_characteristic_equation, FirstOrderODE, ODEError,
    ODESolution, SecondOrderODE, SecondOrderSolution, CharacteristicRoots, RootType,
};
pub use partial_fractions::{
    decompose, is_polynomial, is_rational_function, DecomposeError, PartialFractionResult,
    PartialFractionTerm,
};
pub use equation_system::{
    Constraint, DependencyGraph, EquationSystem, EquationType, IntegralInfo,
    MultiEquationSolver, MultiEquationSolution, NamedEquation, ODEInfo, SolutionStrategy,
    SolutionValue, SolveMethod, SolveStep, SolverConfig, SystemContext, SystemError,
    SystemOperation, SystemResolutionPath, SystemStep, StepResult,
    // Nonlinear system solver
    NonlinearSystem, NonlinearSystemConfig, NonlinearSystemSolverError, NonlinearSystemSolverResult,
    NonlinearSystemSolver, NewtonRaphsonSolver, BroydenSolver, FixedPointSolver, SmartNonlinearSystemSolver,
    ConvergenceBehavior, ConvergenceDiagnostics, newton_raphson_system, broyden_system, fixed_point_system,
    residual_norm, solve_linear_system_lu, validate_jacobian,
};
pub use series::{
    arctan_series, binomial_series, cos_series, exp_series, ln_1_plus_x_series,
    maclaurin, sin_series, taylor, factorial, factorial_expr, compute_nth_derivative,
    evaluate_at, RemainderTerm, Series, SeriesError, SeriesResult, SeriesTerm,
    // Laurent series support
    LaurentSeries, Singularity, SingularityType, laurent, residue, pole_order, find_singularities,
    // Series arithmetic (composition and reversion)
    compose_series, reversion,
    // Asymptotic expansions
    AsymptoticDirection, AsymptoticSeries, AsymptoticTerm, BigO, asymptotic, limit_via_asymptotic,
};
pub use optimization::{
    OperationConfig, OperationType, ComputationStep, StepOperand, MultiplicativeChain,
    PrecisionReport, ManualStep, find_multiplicative_chains, track_precision,
    optimize_computation_order, to_manual_steps, analyze_expression,
};
pub use special::{
    gamma, beta, erf, erfc, SpecialFunctionError, SpecialFunctionResult,
};
pub use approximations::{
    ApproxResult, ApproxType, ScaledExpForm, apply_small_angle_approx,
    compute_approximation_error, select_exp_scaling, is_approximation_valid,
    generate_approximation_step, optimize_pythagorean,
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

/// Unified error type for the thales library.
///
/// This enum provides a single error type that encompasses all possible errors
/// that can occur within the library. It wraps error types from individual modules,
/// allowing for consistent error handling across the entire library.
///
/// # Design
///
/// The `#[non_exhaustive]` attribute allows future versions to add new error variants
/// without breaking existing code. Users should always include a wildcard match arm
/// when matching on this type.
///
/// # Examples
///
/// ```rust
/// use thales::{ThalesError, parse_expression};
///
/// match parse_expression("2 + x") {
///     Ok(expr) => println!("Parsed: {:?}", expr),
///     Err(errors) => {
///         // parse_expression returns Vec<ParseError>, not ThalesError
///         println!("Parse errors: {:?}", errors);
///     }
/// }
/// ```
///
/// Future usage with unified error handling:
///
/// ```rust,ignore
/// use thales::ThalesError;
///
/// fn process() -> Result<(), ThalesError> {
///     // Future API will return ThalesError
///     Ok(())
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum ThalesError {
    /// Error from the parser module.
    Parse(parser::ParseError),
    /// Error from the solver module.
    Solver(solver::SolverError),
    /// Error from the series module.
    Series(series::SeriesError),
    /// Error from the matrix module.
    Matrix(matrix::MatrixError),
    /// Error from the integration module.
    Integration(integration::IntegrationError),
    /// Error from the numerical module.
    Numerical(numerical::NumericalError),
    /// Error from the limits module.
    Limits(limits::LimitError),
    /// Error from the ODE module.
    ODE(ode::ODEError),
    /// Error from the special functions module.
    SpecialFunction(special::SpecialFunctionError),
    /// Error from the inequality module.
    Inequality(inequality::InequalityError),
    /// Error from the precision module.
    Evaluation(precision::EvalError),
    /// Error from the partial fractions module.
    PartialFractions(partial_fractions::DecomposeError),
    /// Error from the LaTeX parser module.
    LaTeXParse(latex::LaTeXParseError),
    /// Error from the equation system module.
    System(equation_system::SystemError),
    /// Error from the nonlinear system solver.
    NonlinearSystem(equation_system::NonlinearSystemSolverError),
}

impl std::fmt::Display for ThalesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThalesError::Parse(e) => write!(f, "Parse error: {}", e),
            ThalesError::Solver(e) => write!(f, "Solver error: {:?}", e),
            ThalesError::Series(e) => write!(f, "Series error: {}", e),
            ThalesError::Matrix(e) => write!(f, "Matrix error: {}", e),
            ThalesError::Integration(e) => write!(f, "Integration error: {}", e),
            ThalesError::Numerical(e) => write!(f, "Numerical error: {:?}", e),
            ThalesError::Limits(e) => write!(f, "Limits error: {}", e),
            ThalesError::ODE(e) => write!(f, "ODE error: {}", e),
            ThalesError::SpecialFunction(e) => write!(f, "Special function error: {}", e),
            ThalesError::Inequality(e) => write!(f, "Inequality error: {}", e),
            ThalesError::Evaluation(e) => write!(f, "Evaluation error: {}", e),
            ThalesError::PartialFractions(e) => write!(f, "Partial fractions error: {}", e),
            ThalesError::LaTeXParse(e) => write!(f, "LaTeX parse error: {}", e),
            ThalesError::System(e) => write!(f, "System error: {:?}", e),
            ThalesError::NonlinearSystem(e) => write!(f, "Nonlinear system error: {:?}", e),
        }
    }
}

impl std::error::Error for ThalesError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ThalesError::Parse(e) => Some(e),
            ThalesError::Series(e) => Some(e),
            ThalesError::Matrix(e) => Some(e),
            ThalesError::Integration(e) => Some(e),
            ThalesError::Limits(e) => Some(e),
            ThalesError::ODE(e) => Some(e),
            ThalesError::SpecialFunction(e) => Some(e),
            ThalesError::Inequality(e) => Some(e),
            ThalesError::Evaluation(e) => Some(e),
            ThalesError::PartialFractions(e) => Some(e),
            ThalesError::LaTeXParse(e) => Some(e),
            // These error types don't implement std::error::Error
            ThalesError::Solver(_) => None,
            ThalesError::Numerical(_) => None,
            ThalesError::System(_) => None,
            ThalesError::NonlinearSystem(_) => None,
        }
    }
}

// Implement From conversions for each error type
impl From<parser::ParseError> for ThalesError {
    fn from(e: parser::ParseError) -> Self {
        ThalesError::Parse(e)
    }
}

impl From<solver::SolverError> for ThalesError {
    fn from(e: solver::SolverError) -> Self {
        ThalesError::Solver(e)
    }
}

impl From<series::SeriesError> for ThalesError {
    fn from(e: series::SeriesError) -> Self {
        ThalesError::Series(e)
    }
}

impl From<matrix::MatrixError> for ThalesError {
    fn from(e: matrix::MatrixError) -> Self {
        ThalesError::Matrix(e)
    }
}

impl From<integration::IntegrationError> for ThalesError {
    fn from(e: integration::IntegrationError) -> Self {
        ThalesError::Integration(e)
    }
}

impl From<numerical::NumericalError> for ThalesError {
    fn from(e: numerical::NumericalError) -> Self {
        ThalesError::Numerical(e)
    }
}

impl From<limits::LimitError> for ThalesError {
    fn from(e: limits::LimitError) -> Self {
        ThalesError::Limits(e)
    }
}

impl From<ode::ODEError> for ThalesError {
    fn from(e: ode::ODEError) -> Self {
        ThalesError::ODE(e)
    }
}

impl From<special::SpecialFunctionError> for ThalesError {
    fn from(e: special::SpecialFunctionError) -> Self {
        ThalesError::SpecialFunction(e)
    }
}

impl From<inequality::InequalityError> for ThalesError {
    fn from(e: inequality::InequalityError) -> Self {
        ThalesError::Inequality(e)
    }
}

impl From<precision::EvalError> for ThalesError {
    fn from(e: precision::EvalError) -> Self {
        ThalesError::Evaluation(e)
    }
}

impl From<partial_fractions::DecomposeError> for ThalesError {
    fn from(e: partial_fractions::DecomposeError) -> Self {
        ThalesError::PartialFractions(e)
    }
}

impl From<latex::LaTeXParseError> for ThalesError {
    fn from(e: latex::LaTeXParseError) -> Self {
        ThalesError::LaTeXParse(e)
    }
}

impl From<equation_system::SystemError> for ThalesError {
    fn from(e: equation_system::SystemError) -> Self {
        ThalesError::System(e)
    }
}

impl From<equation_system::NonlinearSystemSolverError> for ThalesError {
    fn from(e: equation_system::NonlinearSystemSolverError) -> Self {
        ThalesError::NonlinearSystem(e)
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thales_error_from_parse_error() {
        let parse_err = parser::ParseError::UnexpectedCharacter { pos: 0, found: 'x' };
        let thales_err: ThalesError = parse_err.clone().into();

        match thales_err {
            ThalesError::Parse(e) => assert_eq!(e, parse_err),
            _ => panic!("Expected ThalesError::Parse"),
        }
    }

    #[test]
    fn test_thales_error_from_solver_error() {
        let solver_err = solver::SolverError::NoSolution;
        let thales_err: ThalesError = solver_err.clone().into();

        match thales_err {
            ThalesError::Solver(e) => assert_eq!(e, solver_err),
            _ => panic!("Expected ThalesError::Solver"),
        }
    }

    #[test]
    fn test_thales_error_from_numerical_error() {
        let num_err = numerical::NumericalError::NoConvergence;
        let thales_err: ThalesError = num_err.clone().into();

        match thales_err {
            ThalesError::Numerical(e) => assert_eq!(e, num_err),
            _ => panic!("Expected ThalesError::Numerical"),
        }
    }

    #[test]
    fn test_thales_error_display() {
        let solver_err = solver::SolverError::NoSolution;
        let thales_err: ThalesError = solver_err.into();
        let display_str = format!("{}", thales_err);

        assert!(display_str.contains("Solver error"));
        assert!(display_str.contains("NoSolution"));
    }

    #[test]
    fn test_thales_error_source() {
        use std::error::Error;

        let parse_err = parser::ParseError::UnexpectedCharacter { pos: 5, found: '!' };
        let thales_err: ThalesError = parse_err.into();

        assert!(thales_err.source().is_some());
    }

    #[test]
    fn test_thales_error_from_integration_error() {
        let int_err = integration::IntegrationError::DivisionByZero;
        let thales_err: ThalesError = int_err.clone().into();

        match thales_err {
            ThalesError::Integration(e) => assert_eq!(e, int_err),
            _ => panic!("Expected ThalesError::Integration"),
        }
    }

    #[test]
    fn test_thales_error_from_matrix_error() {
        let matrix_err = matrix::MatrixError::EmptyMatrix;
        let thales_err: ThalesError = matrix_err.clone().into();

        match thales_err {
            ThalesError::Matrix(e) => assert_eq!(e, matrix_err),
            _ => panic!("Expected ThalesError::Matrix"),
        }
    }
}
