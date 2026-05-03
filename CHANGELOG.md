# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-05-02

### Breaking Changes

- **JSON wire format**: Expression fields serialize as structured JSON objects
  (`{"kind": "Binary", "value": {...}}`) instead of display strings (`"x + 1"`).
  Callers must update request payloads and response parsers.
- **FFI surface collapsed**: All ~57 per-operation FFI functions removed.
  `execute_json_ffi(json)` is the sole cross-language entry point.
  Swift package requires rebuild against new bindings.

### Added

- **Series expansions**: Puiseux, Frobenius, Padé approximant, WKB approximation
- **Vector calculus**: Nabla operator (`Grad`, `Div`, `Curl`, `Laplacian`,
  `DivOfCurl`, `CurlOfGrad`, `DivOfGrad`) with golden identity tests
- **ODE systems**: `OdeSystem` command with matrix extraction, eigenvalue solver,
  and RK4 system integration; `Pde` command stub
- **Integral transforms**: Laplace, inverse Laplace, Fourier, inverse Fourier,
  Z-transform, inverse Z-transform, Mellin, inverse Mellin
- **Special functions**: 18 new variants — Gamma, Beta, Erf, Erfc, LnGamma,
  Digamma, BesselJ/Y/I/K, AiryAi/Bi, Zeta, Si, Ci, Ei, Heaviside, DiracDelta
- **Multi-variable integration**: `MultiIntegrate`, `ChangeCoords`,
  `PathIntegral`, `SurfaceIntegral` commands with coordinate system support
- **Algebra promotions**: `Conjugate`, `InverseFn`, `ApplyIdentity` commands
- **Higher-dim calculus**: `TotalDiff`, `Divergence`, `Curl`, `Laplacian`,
  `Jacobian`, `Hessian`, `DirectionalDiff` commands
- **Optimization**: `Optimize` and `LagrangeMult` commands with constraint support
- **Structured response model**: `StructuredResult` envelope with typed result
  shapes (Scalar, Labeled, Decomposition, CoefficientArray, Branches, Shaped,
  TransformResult)
- **Narrative resolution**: dispatch exit renders narratives into localized text

### Fixed

- Complex roots in quadratic/cubic/quartic solvers use `Expr::complex()` natively
  instead of building symbolic `re + im*i` tree — fixes decompile producing
  `2*i` instead of `Complex{re:0, im:2}`
- Eigenvalue dispatch returns `Expression::Complex` for complex eigenvalues
  instead of silently dropping imaginary parts
- Stress tests: ~30 impossible-to-solve tests converted from `#[ignore]` to
  proper `assert_solve_fails` assertions

### Changed

- Migrated to mathlex v0.4.0 ExprKind/Expression API
- `dispatch.rs` split into command-family submodules
- `compile.rs` (1859 lines) → `compile/` submodule directory
- `mathlex_bridge.rs` (1267 lines) → `mathlex_bridge/` submodule directory
- `lib.rs` error types extracted to `error.rs`
- `simplify()` (263 lines) → 7 per-pass helper functions
- `polynomial.rs` (706 lines) → `polynomial/` with cubic, quartic, numerical submodules
- `to_latex_inner()` (220 lines) → 7 per-variant helper functions
- Annotations RFC deferred to v0.10.0

## [0.4.2] - 2026-04-07

### Fixed

- Split SPM package into ThalesBridge (C) and Thales (Swift) targets so swift-bridge C types (RustStr, RustString) are visible when consumed as an SPM dependency
- Fix ToRustStr protocol to use rethrows for throwing closure support
- Add @retroactive conformance annotations for RustStr extensions

## [0.4.1] - 2026-04-07

### Added

- **Swift wrappers for 2nd-order ODE**: `solveODE2ndOrder` and `solveODE2ndOrderIVP` for constant-coefficient ODEs with optional forcing function
- **Swift wrappers for higher-order ODE**: `solveHigherOrderODE` for nth-order constant-coefficient homogeneous ODEs
- **Swift wrapper for RK4 numerical ODE integration**: `solveODENumerical` returns trajectory as array of (x, y) pairs
- **Swift wrapper for precision-controlled evaluation**: `evaluate(_:with:precision:decimalPlaces:rounding:)` with `PrecisionMode` and `RoundingMode` enums
- **Swift wrapper for Fourier series**: `fourierSeries` with `FourierSeriesResult` containing parsed coefficients
- **Series composition and reversion**: `composeSeries` for f(g(x)) and `reversionSeries` for functional inverse, with new FFI bindings
- **2D coordinate transform wrappers**: `translate2D`, `rotate2D`, `scale2D` via FFI
- **Complex nth roots wrapper**: `complexNthRoots` returns all n roots of a complex number
- **Unit conversion wrapper**: `convertUnits` using the built-in dimensional analysis registry
- **LaTeX calculus notation parsing wrapper**: `parseLatexCalculus` for integral, limit, and sum notations

### Fixed

- Sync bridge.rs with ffi.rs: add missing result type structs for Taylor, Laurent, Asymptotic, and SpecialFunction series
- Regenerate Swift bridge bindings to include all FFI functions added since v0.3.0
- Fix ComplexNumber field references in Swift wrapper (re/im to real/imaginary)
- Fix Fourier series FFI test that expected Ok for parse errors (now correctly expects Err)
- Fix precision evaluation tests to avoid Debug trait requirement on bridge types

## [0.4.0] - 2026-04-06

### Added

- **Parser migration**: migrate parsing to the mathlex shared library for cross-project consistency
- **LAPACK support**: optional LAPACK backend for matrix operations (accelerate, netlib, openblas)
- **Numerical optimizers**: gradient descent and Levenberg-Marquardt nonlinear least squares
- **Brent's method**: root-finding algorithm for robust numerical solutions
- **ODE/integration solvers**: wire ODE and numerical integration into the equation system solver
- **LaTeX parsing**: support for `\int`, `\lim`, `\sum`, and `\log_{b}` expressions
- **Equation system parsing**: `parse_equation_system` for semicolon-separated equations
- **FFI expansion**: ODE solver types, Laurent/asymptotic series, beta/erfc special functions
- **Swift wrappers**: series and special function wrappers for the Swift bridge
- **Transforms**: complex nth roots via De Moivre's theorem, 2D translation/rotation/scaling
- **Dimensions**: dimension arithmetic and display formatting
- **Simplification**: wire pattern matching rules into `Expression::simplify`
- **DocC documentation**: expanded coverage for ODE, series, parsing, and result types

### Fixed

- Normalize log argument order to `log(value, base)` convention
- Fix FFI parse error formatting (use Debug for vectors, Display for output)
- Fix numerical integration aliasing bug
- Fix Runge-Kutta doc test missing unwrap on Result
- Replace string-based equality with structural expression comparison in integration
- Remove dead code (unused functions in solver, series modules)
- Correct documentation claims to match implemented reality
- Update test expectations for implicit multiplication changes

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
