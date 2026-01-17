# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.3] - 2026-01-17

### Fixed

- Release workflow: add `--allow-dirty` flag for cargo publish
- Swift XCFramework workflow: add permissions for release uploads

## [0.3.2] - 2026-01-17

### Added

- GitHub Actions CI workflow with build status badge
- Automated release workflow for crates.io publishing
- Swift XCFramework build workflow for iOS/macOS distribution
- Swift Package Index configuration for DocC documentation hosting
- DocC documentation catalog for Swift package
- CONTRIBUTING.md with development and release guidelines

### Fixed

- Resolved compiler warnings (unused imports, variables, unreachable patterns)
- Fixed version test to use dynamic version from Cargo.toml
- Applied cargo fmt formatting fixes across codebase

## [0.3.1] - 2026-01-12

### Fixed

- Documentation: replaced incorrect "LaTeX rendering" terminology with "LaTeX generation/output" to accurately describe the library's functionality (generates LaTeX strings, not visual output)

## [0.3.0] - 2026-01-12 - Advanced Calculus & API Stabilization

### Added

- **Second-Order ODE Solver** (`second_order_ode` module)
  - Constant coefficient homogeneous equations
  - Characteristic equation method
  - Real distinct, repeated, and complex conjugate roots
  - Initial value problems with boundary conditions

- **Nonlinear System Solver** (`nonlinear_system` module)
  - Newton-Raphson for systems of nonlinear equations
  - Jacobian matrix computation
  - Convergence control and iteration limits
  - Multiple solution detection

- **Taylor Series Expansions** (`series` module enhancements)
  - Taylor series expansion around arbitrary points
  - Maclaurin series (expansion around zero)
  - Laurent series for functions with poles
  - Configurable expansion order
  - Coefficient extraction and manipulation

- **Asymptotic Expansions** (`series` module)
  - Big-O notation support
  - Asymptotic series for large arguments
  - Direction-aware expansions (infinity, zero, custom)
  - Remainder term estimation

- **Special Mathematical Functions** (`special` module)
  - Gamma function with reflection formula
  - Beta function via gamma
  - Error function (erf) with series expansion
  - Complementary error function (erfc)
  - Step-by-step derivation tracking

- **Small Angle Approximations** (`approximations` module)
  - sin(x) ≈ x with error bounds
  - cos(x) ≈ 1 - x²/2 with error bounds
  - tan(x) ≈ x with error bounds
  - Automatic threshold detection
  - Scaled exponential forms

- **Operation Ordering Optimizer** (`optimization` module)
  - Optimize expression evaluation order for manual calculation
  - Minimize intermediate result magnitudes
  - Slide rule-friendly operation sequencing
  - Multiplicative chain optimization

- **Unified Error Type** (`ThalesError`)
  - Single error type wrapping all module errors
  - Implements `std::error::Error` trait
  - Source error chaining where available
  - `#[non_exhaustive]` for forward compatibility

- **Property-Based Tests** (`tests/property_tests.rs`)
  - 23 proptest-based tests
  - Coordinate transformation round-trips
  - Numerical method convergence
  - Parser/formatter consistency

- **Extended FFI Bindings** (`ffi` module)
  - Second-order ODE solving
  - Nonlinear system solving
  - Series expansion functions
  - Special function access

### Changed

- **API Stabilization**
  - Added `#[non_exhaustive]` to public enums
  - Added `#[must_use]` to Result-returning functions
  - Improved error messages with context

### Fixed

- Error type Display implementations for types without std::error::Error

### Tests

- 971 total tests (292 doctests, 351 unit tests, 23 property tests, others)
- All tests passing

## [0.2.0] - 2026-01-01 - Multi-Equation System Solver & Calculus

### Added

- **Multi-Equation System Solver** (`equation_system` module)
  - Solve systems of arbitrary equations (algebraic, ODE, differential, integral)
  - Automatic dependency graph construction
  - Topological sorting for optimal solving order
  - Chained solution propagation between equations
  - Unified `SystemResolutionPath` for step-by-step tracking
  - FFI bindings via `solve_equation_system_ffi()`

- **Limits with L'Hôpital's Rule** (`limits` module)
  - Direct substitution for continuous functions
  - Limits at positive/negative infinity
  - One-sided limits (left and right)
  - Automatic L'Hôpital's rule for 0/0 and ∞/∞ indeterminate forms
  - Detection of all indeterminate forms (0·∞, ∞-∞, 0⁰, 1^∞, ∞⁰)
  - Special limits (sin(x)/x, tan(x)/x, (1-cos(x))/x²)

- **Partial Fraction Decomposition** (`partial_fractions` module)
  - Decompose rational functions into partial fractions
  - Support for linear and repeated linear factors
  - Symbolic integration of decomposed forms

- **Pattern Matching** (`pattern` module)
  - Rule-based expression rewriting
  - Wildcard patterns with binding
  - Commutativity-aware matching for + and *
  - Common algebraic rules (identity, zero, double negation, etc.)
  - Apply rules recursively to fixpoint

- **LaTeX Support** (`latex` module)
  - Parse LaTeX mathematical notation
  - Support for `\frac`, `\sqrt`, `\sin`, `\cos`, Greek letters, etc.
  - Render expressions to LaTeX output
  - Display and inline math modes

- **Integration Enhancements** (`integration` module)
  - Integration by parts with step tracking
  - Integration by substitution
  - Tabular integration method
  - Improper integrals to infinity

- **ODE Solving** (`ode` module)
  - First-order separable ODEs
  - First-order linear ODEs
  - Initial value problems

- **Trigonometric Simplification** (`trigonometric` module)
  - Pythagorean identities
  - Double angle formulas
  - Product-to-sum rules
  - Quotient identities
  - Step-by-step simplification

- **Inequality Solving** (`inequality` module)
  - Linear and polynomial inequalities
  - Interval solution representation
  - Systems of inequalities

- **Matrix Expressions** (`matrix` module)
  - Matrix AST representation
  - LaTeX output with bracket styles

- **Precision Control** (`precision` module)
  - Configurable evaluation context
  - Multiple rounding modes
  - High-precision computation

### Changed

- Extended FFI bindings for all new features
- Enhanced resolution path tracking

### Tests

- 276+ unit tests with comprehensive coverage

## [0.1.0] - 2025-12-17 - Initial Release

### Added

- **Core AST** (`ast` module)
  - `Expression`, `Equation`, `Variable`, `BinaryOp`, `UnaryOp`, `Function` types
  - Full operator precedence support

- **Coordinate Transformations** (`transforms` module)
  - 2D: Cartesian ↔ Polar
  - 3D: Cartesian ↔ Spherical ↔ Cylindrical
  - Full round-trip verification

- **Complex Number Operations**
  - De Moivre's theorem implementation
  - Polar form conversion
  - Conjugate and modulus operations

- **Expression Parser** (`parser` module)
  - Chumsky-based parser with full operator precedence
  - Support for variables, functions, and nested expressions

- **Equation Solvers** (`solver` module)
  - Linear equation solver (ax + b = c)
  - Quadratic equation solver (discriminant method)
  - Polynomial solver (companion matrix)
  - Transcendental equation solver (trig/exp/log)
  - Smart solver with automatic method dispatch

- **Numerical Methods** (`numerical` module)
  - Newton-Raphson with symbolic differentiation
  - Bisection method
  - Brent's hybrid method
  - Secant method
  - Levenberg-Marquardt

- **Calculus** (`integration` module)
  - Symbolic differentiation engine
  - Basic symbolic integration
  - Resolution path generation

- **iOS Support**
  - Cross-compilation for aarch64-apple-ios
  - Simulator support (ARM and Intel)
  - Universal library creation
  - FFI bindings with swift-bridge

- **Infrastructure**
  - Test framework with proptest integration
  - Benchmark infrastructure with criterion
  - Build optimization (LTO, single codegen unit)

[0.3.0]: https://github.com/ChrisGVE/thales/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ChrisGVE/thales/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ChrisGVE/thales/releases/tag/v0.1.0
