# mathsolver-core

[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](https://github.com/ChrisGVE/mathsolver-core)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-iOS%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](https://www.rust-lang.org/what/embedded)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org)

A high-performance Rust library for mathematical equation parsing, symbolic solving, numerical approximation, and coordinate transformations. Designed for cross-platform use with first-class iOS support via FFI bindings.

## Features

### Parsing & Expression Handling
- **Expression Parsing**: Parse mathematical expressions and equations with full operator precedence using chumsky parser combinators
- **LaTeX Parsing**: Parse LaTeX mathematical notation (`\frac`, `\sqrt`, Greek letters, etc.)
- **LaTeX Rendering**: Convert expressions to LaTeX output for display
- **Pattern Matching**: Rule-based expression rewriting with wildcard matching and commutativity awareness

### Equation Solving
- **Symbolic Solving**: Solve linear, quadratic, polynomial, and transcendental equations using algebraic manipulation
- **Multi-Equation Systems**: Solve systems of arbitrary equations with automatic dependency analysis and solving order determination
- **Inequality Solving**: Solve inequalities and systems of inequalities with interval solutions
- **Numerical Methods**: Newton-Raphson, secant method, bisection, Brent's method, Levenberg-Marquardt for equations that cannot be solved symbolically

### Calculus
- **Differentiation**: Symbolic differentiation with support for all elementary functions
- **Integration**: Indefinite and definite integration, integration by parts, substitution, tabular method
- **Limits**: Limit evaluation with L'Hôpital's rule for indeterminate forms (0/0, ∞/∞)
- **ODEs**: First-order ODE solving (separable and linear equations)

### Advanced Features
- **Partial Fractions**: Decompose rational functions into partial fractions
- **Trigonometric Simplification**: Apply Pythagorean identities, double angle formulas, and more
- **Matrix Operations**: Matrix expressions with LaTeX rendering

### Coordinate Systems
- **Coordinate Transformations**: Convert between Cartesian, polar, spherical, and cylindrical coordinate systems with full precision
- **Complex Numbers**: Complete support for complex arithmetic, polar form operations, and De Moivre's theorem

### Infrastructure
- **Resolution Paths**: Track step-by-step solution processes for educational applications
- **Unit System**: Dimensional analysis and automatic unit conversion for physics calculations
- **FFI Support**: Swift bindings for seamless iOS/macOS integration via swift-bridge
- **Memory Safety**: Zero unsafe code in core logic, all abstractions are zero-cost
- **Performance**: Link-time optimization, single codegen unit, aggressive compiler optimizations

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Building from Source](#building-from-source)
- [Testing](#testing)
- [iOS Cross-Compilation](#ios-cross-compilation)
- [Contributing](#contributing)
- [License](#license)
- [Dependencies](#dependencies)
- [Version History](#version-history)
- [References](#references)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
mathsolver-core = "0.1.0"
```

Or install from GitHub:

```toml
[dependencies]
mathsolver-core = { git = "https://github.com/ChrisGVE/mathsolver-core", branch = "main" }
```

## Quick Start

### Example 1: Coordinate Transformations

Convert between 2D Cartesian and polar coordinate systems:

```rust
use mathsolver_core::{Cartesian2D, Polar};

fn main() {
    // Create a 2D Cartesian point
    let cartesian = Cartesian2D::new(3.0, 4.0);

    // Convert to polar coordinates
    let polar = cartesian.to_polar();
    assert!((polar.r - 5.0).abs() < 1e-10);
    assert!((polar.theta - 0.927295218).abs() < 1e-6);

    // Convert back to Cartesian (round-trip verification)
    let back = polar.to_cartesian();
    assert!((back.x - 3.0).abs() < 1e-10);
    assert!((back.y - 4.0).abs() < 1e-10);
}
```

### Example 2: 3D Coordinate Transformations

Convert between Cartesian and spherical coordinates:

```rust
use mathsolver_core::{Cartesian3D, Spherical};

fn main() {
    // Create a 3D Cartesian point
    let cart3d = Cartesian3D::new(1.0, 1.0, 1.0);

    // Convert to spherical coordinates
    let spherical = cart3d.to_spherical();
    assert!((spherical.r - 1.732050808).abs() < 1e-6);

    // Convert back to Cartesian
    let back = spherical.to_cartesian();
    assert!((back.x - 1.0).abs() < 1e-10);
    assert!((back.y - 1.0).abs() < 1e-10);
    assert!((back.z - 1.0).abs() < 1e-10);
}
```

### Example 3: Complex Number Operations

Work with complex numbers using polar form and De Moivre's theorem:

```rust
use mathsolver_core::ComplexOps;
use num_complex::Complex64;

fn main() {
    // Create a complex number (1 + i)
    let z = Complex64::new(1.0, 1.0);

    // Apply De Moivre's theorem: (r∠θ)^n = r^n∠(nθ)
    let result = ComplexOps::de_moivre(z, 2.0);

    // Complex conjugate
    let conj = z.conj();
    assert_eq!(conj.re, z.re);
    assert_eq!(conj.im, -z.im);

    // Modulus (magnitude) of complex number
    let modulus = z.norm();
    assert!((modulus - 1.4142135623730951).abs() < 1e-10);
}
```

### Example 4: Expression Parsing and Solving (In Progress)

Build mathematical expressions using the AST:

```rust
use mathsolver_core::{Expression, Variable, BinaryOp};

fn main() {
    // Create expression: 2*x + 5
    let x = Variable::new("x");
    let two_x = Expression::Binary(
        BinaryOp::Mul,
        Box::new(Expression::Integer(2)),
        Box::new(Expression::Variable(x.clone()))
    );
    let expr = Expression::Binary(
        BinaryOp::Add,
        Box::new(two_x),
        Box::new(Expression::Integer(5))
    );

    // Check variable containment
    assert!(expr.contains_variable("x"));
    assert!(!expr.contains_variable("y"));
}
```

For complete parsing and solving capabilities (parser implementation in progress):

```rust
use mathsolver_core::{parse_equation, SmartSolver, Solver, Variable};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse equation from string
    let equation = parse_equation("2*x + 5 = 13")?;

    // Solve for x using smart solver
    let solver = SmartSolver::new();
    let x = Variable::new("x");
    let (solution, path) = solver.solve(&equation, &x)?;

    // Display step-by-step solution
    for step in path.steps() {
        println!("{}", step);
    }

    Ok(())
}
```

### Example 5: Cylindrical Coordinates

Convert between Cartesian and cylindrical coordinates:

```rust
use mathsolver_core::{Cartesian3D, Cylindrical};

fn main() {
    // Create Cartesian point
    let cart = Cartesian3D::new(3.0, 4.0, 5.0);

    // Convert to cylindrical (rho, phi, z)
    let cyl = cart.to_cylindrical();
    assert!((cyl.rho - 5.0).abs() < 1e-10);
    assert!((cyl.z - 5.0).abs() < 1e-10);

    // Round-trip verification
    let back = cyl.to_cartesian();
    assert!((back.x - 3.0).abs() < 1e-10);
    assert!((back.y - 4.0).abs() < 1e-10);
}
```

### Example 6: Multi-Equation System Solver

Solve systems of equations with automatic dependency analysis:

```rust
use mathsolver_core::{
    EquationSystem, SystemContext, MultiEquationSolver, parse_equation
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a system of physics equations
    let mut system = EquationSystem::new();
    system.add_equation("newton", parse_equation("F = m * a")?);
    system.add_equation("kinematic", parse_equation("v = u + a * t")?);

    // Set up known values and targets
    let context = SystemContext::new()
        .with_known_value("F", 100.0)   // Force = 100 N
        .with_known_value("m", 20.0)    // Mass = 20 kg
        .with_known_value("u", 0.0)     // Initial velocity = 0 m/s
        .with_known_value("t", 5.0)     // Time = 5 s
        .with_target("a")               // Find acceleration
        .with_target("v");              // Find final velocity

    // Solve the system
    let solver = MultiEquationSolver::new();
    let solution = solver.solve(&system, &context)?;

    // Get results
    // From F = m*a: a = F/m = 100/20 = 5 m/s²
    let a = solution.get_numeric("a").unwrap();
    assert!((a - 5.0).abs() < 1e-10);

    // From v = u + a*t: v = 0 + 5*5 = 25 m/s
    let v = solution.get_numeric("v").unwrap();
    assert!((v - 25.0).abs() < 1e-10);

    // Access step-by-step resolution
    println!("{}", solution.resolution_path.format_text());

    Ok(())
}
```

### Example 7: Calculus Operations

Differentiation, integration, and limits:

```rust
use mathsolver_core::{parse_expression, integrate, limit};
use mathsolver_core::ast::Variable;
use mathsolver_core::limits::{compute_limit, LimitPoint};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Differentiation
    let expr = parse_expression("x^3 + 2*x^2 - 5*x + 3")?;
    let derivative = expr.differentiate(&Variable::new("x"));
    // Result: 3*x^2 + 4*x - 5

    // Integration
    let integral = integrate(&expr, &Variable::new("x"))?;
    // Result: (1/4)*x^4 + (2/3)*x^3 - (5/2)*x^2 + 3*x + C

    // Limits with L'Hôpital's rule
    let sin_x_over_x = parse_expression("sin(x) / x")?;
    let result = compute_limit(&sin_x_over_x, &Variable::new("x"), LimitPoint::Value(0.0))?;
    // Result: 1.0 (using L'Hôpital's rule for 0/0 form)

    Ok(())
}
```

## API Reference

The library is organized into focused modules. See [full documentation on docs.rs](https://docs.rs/mathsolver-core) or build locally:

```bash
cargo doc --open
```

### Core Modules

- **`ast`** - Abstract syntax tree definitions for mathematical expressions, equations, variables, operators, and functions
- **`parser`** - String to AST conversion using chumsky parser combinator library
- **`latex`** - LaTeX parsing and rendering for mathematical expressions
- **`solver`** - Symbolic equation solving with specialized solvers for linear, quadratic, polynomial, and transcendental equations
- **`equation_system`** - Multi-equation system solver with dependency analysis and automatic solving order
- **`numerical`** - Numerical root-finding methods (Newton-Raphson, bisection, Brent's method) with symbolic differentiation
- **`limits`** - Limit evaluation with L'Hôpital's rule for indeterminate forms
- **`integration`** - Symbolic integration with multiple methods (by parts, substitution, tabular)
- **`ode`** - Ordinary differential equation solving (separable and linear first-order)
- **`inequality`** - Inequality solving with interval solutions
- **`partial_fractions`** - Rational function decomposition into partial fractions
- **`pattern`** - Pattern matching and rule-based expression rewriting
- **`trigonometric`** - Trigonometric identity simplification
- **`resolution_path`** - Solution step tracking for educational applications
- **`precision`** - High-precision evaluation with configurable rounding modes
- **`matrix`** - Matrix expressions with LaTeX rendering
- **`dimensions`** - Dimensional analysis and unit conversion system
- **`transforms`** - Coordinate system conversions (Cartesian, Polar, Spherical, Cylindrical) and complex number operations
- **`ffi`** - Foreign function interface for Swift via swift-bridge (requires `ffi` feature flag)

### Key Types

Commonly used types re-exported at crate root:

```rust
use mathsolver_core::{
    // AST types
    Expression, Equation, Variable, BinaryOp, UnaryOp, Function,

    // Equation System Solving
    EquationSystem, SystemContext, MultiEquationSolver, MultiEquationSolution,
    NamedEquation, EquationType, SolutionValue, DependencyGraph,

    // Coordinate systems
    Cartesian2D, Cartesian3D, Polar, Spherical, Cylindrical,
    ComplexOps,

    // Single-equation Solvers
    SmartSolver, Solver, Solution,
    SmartNumericalSolver, NumericalSolution, NumericalConfig,

    // Calculus
    integrate, definite_integral, integrate_by_parts,
    solve_separable, solve_linear, FirstOrderODE, ODESolution,

    // Limits
    // (use mathsolver_core::limits::{compute_limit, LimitPoint, ...})

    // Inequalities
    solve_inequality, Inequality, IntervalSolution,

    // Partial Fractions
    decompose, PartialFractionResult, PartialFractionTerm,

    // Pattern Matching
    // (use mathsolver_core::pattern::{Pattern, match_pattern, ...})

    // Trigonometric
    simplify_trig, simplify_trig_with_steps,

    // LaTeX
    parse_latex, parse_latex_equation,

    // Units and dimensions
    Unit, Dimension, Quantity, UnitRegistry,

    // Resolution tracking
    ResolutionPath, ResolutionStep, Operation,
};
```

### Feature Flags

- **`default`** - Standard features (currently none)
- **`ffi`** - Enable Swift bindings for iOS/macOS integration

Enable FFI in your `Cargo.toml`:

```toml
[dependencies]
mathsolver-core = { version = "0.1.0", features = ["ffi"] }
```

## Building from Source

### Standard Build

```bash
# Clone repository
git clone https://github.com/ChrisGVE/mathsolver-core.git
cd mathsolver-core

# Build library
cargo build --release

# Build with FFI support
cargo build --release --features ffi

# Run tests
cargo test

# Generate documentation
cargo doc --open

# Run benchmarks
cargo bench
```

### Build Profiles

The project uses aggressive optimization in release mode:

```toml
[profile.release]
opt-level = 3           # Maximum optimization
lto = true              # Link-time optimization
codegen-units = 1       # Single codegen unit for maximum optimization
```

### Clean Build

```bash
# Remove all build artifacts
cargo clean

# Rebuild from scratch
cargo build --release
```

## Testing

Comprehensive test suite with unit tests, integration tests, property-based tests, and numerical accuracy validation.

### Run All Tests

```bash
# Run standard test suite
cargo test

# Run with verbose output
cargo test -- --nocapture

# Run documentation tests
cargo test --doc

# Run specific test
cargo test test_cartesian_to_polar

# Run ignored tests (requires full implementation)
cargo test -- --ignored
```

### Property-Based Testing

The library uses [proptest](https://github.com/proptest-rs/proptest) for property-based testing:

```bash
# Run property tests with verbose output
cargo test --release -- --nocapture proptest
```

### Test Organization

- **Unit tests**: Embedded in source files (`#[cfg(test)]` modules)
- **Integration tests**: Located in `tests/` directory
- **Documentation tests**: Code examples in doc comments
- **Property tests**: Randomized test generation with proptest

### Test Coverage

Current implementation status (verify with tests):

```bash
# Run all tests and display pass/fail status
cargo test --release

# Run only implemented features
cargo test --release --lib
```

## iOS Cross-Compilation

Build for iOS devices and simulators with full ARM64 and x86_64 support.

### Prerequisites

Install iOS targets (one-time setup):

```bash
rustup target add aarch64-apple-ios          # iOS devices (iPhone, iPad)
rustup target add aarch64-apple-ios-sim      # iOS simulator (Apple Silicon)
rustup target add x86_64-apple-ios           # iOS simulator (Intel)
```

### Build for iOS Device

```bash
cargo build --release --target aarch64-apple-ios
```

Output: `target/aarch64-apple-ios/release/libmathsolver_core.a`

### Build for iOS Simulator

Build for both Apple Silicon and Intel simulators:

```bash
# iOS simulator (Apple Silicon)
cargo build --release --target aarch64-apple-ios-sim

# iOS simulator (Intel)
cargo build --release --target x86_64-apple-ios
```

### Create Universal Simulator Library

Combine Intel and ARM simulator builds into a universal library:

```bash
lipo -create \
  target/aarch64-apple-ios-sim/release/libmathsolver_core.a \
  target/x86_64-apple-ios/release/libmathsolver_core.a \
  -output target/libmathsolver_core_sim.a
```

### Verify Library Architectures

```bash
# Check device library (should show arm64)
lipo -info target/aarch64-apple-ios/release/libmathsolver_core.a

# Check simulator library (should show x86_64 and arm64)
lipo -info target/libmathsolver_core_sim.a
```

### Automated Build Script

Use the provided build script for convenience:

```bash
./build_ios.sh
```

This script:
1. Builds for all three iOS targets
2. Creates universal simulator library
3. Verifies architectures
4. Copies generated Swift files to Xcode project

See [IOS_BUILD.md](IOS_BUILD.md) for complete iOS integration guide including:
- Swift-Bridge code generation
- Xcode project configuration
- Library search paths
- Bridging header setup
- Xcode build phase integration

### Swift-Bridge FFI

Build with FFI support for Swift bindings:

```bash
cargo build --release --features ffi --target aarch64-apple-ios
```

Generated files:
- `target/SwiftBridgeCore.swift` - Core Swift bridge code
- `target/mathsolver_core.swift` - Generated Swift API
- `target/mathsolver-core-Bridging-Header.h` - Objective-C bridging header

## Contributing

This is a private project for use with the SlipStick iOS app. Contributions are welcome via pull requests.

### Code Standards

All contributions must adhere to these standards:

1. **Rust Idioms**: Follow Rust 2021 edition idioms and best practices
2. **Zero Unsafe Code**: No `unsafe` blocks outside FFI boundary (core abstractions must be safe)
3. **Clippy Compliance**: Pass `clippy::pedantic` lint checks:
   ```bash
   cargo clippy --all-targets --all-features -- -D warnings
   ```
4. **Documentation**: Comprehensive documentation with examples for all public APIs
5. **Test Coverage**: Complete test coverage including:
   - Unit tests for all functions
   - Integration tests for complete workflows
   - Documentation tests (code examples in doc comments)
   - Property-based tests for numerical functions
6. **Performance**: Benchmark performance-critical code paths:
   ```bash
   cargo bench
   ```
7. **Formatting**: Use `rustfmt` for consistent code style:
   ```bash
   cargo fmt --all
   ```

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes following code standards
4. Run tests: `cargo test`
5. Run clippy: `cargo clippy -- -D warnings`
6. Format code: `cargo fmt --all`
7. Commit with conventional format (see below)
8. Push and create pull request

### Commit Format

Use conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation only
- `style` - Code formatting (no logic change)
- `refactor` - Code change (neither fix nor feature)
- `test` - Adding or updating tests
- `chore` - Auxiliary tools/libraries
- `perf` - Performance improvement
- `ci` - CI configuration changes
- `build` - Build system/script changes
- `revert` - Reverts previous commit

**Scope:** Optional, location of change (e.g., `parser`, `solver`, `transforms`)

**Breaking Changes:** Add `!` after type/scope and include `BREAKING CHANGE:` in footer

**Examples:**

```
feat(transforms): add homogeneous transformation matrices
```

```
fix(parser): handle negative exponents correctly

Previous implementation failed to parse expressions like 10^-3
due to precedence issues with unary minus.
```

```
feat(solver)!: change solution return type to Result

BREAKING CHANGE: solve() now returns Result<Solution, Error>
instead of Option<Solution> for better error reporting.
```

### Semantic Versioning

This project follows [Semantic Versioning](https://semver.org/):

- **Breaking changes** → increment major version (1.0.0 → 2.0.0)
- **New features** (`feat`) → increment minor version (1.0.0 → 1.1.0)
- **Bug fixes** (`fix`) → increment patch version (1.0.0 → 1.0.1)

### Testing Requirements

Before submitting a pull request:

```bash
# Run full test suite
cargo test --all-features

# Run clippy
cargo clippy --all-targets --all-features -- -D warnings

# Format code
cargo fmt --all

# Build documentation
cargo doc --no-deps

# Run benchmarks (if touching performance-critical code)
cargo bench
```

### Code Review Process

All pull requests require:
1. Passing CI checks (tests, clippy, formatting)
2. Code review approval
3. Updated documentation
4. Updated CHANGELOG.md

## License

MIT License

Copyright (c) 2025 Christian C. Berclaz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Dependencies

### Core Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| **chumsky** | 1.0.0-alpha.8 | Parser combinator library for expression and equation parsing with full operator precedence |
| **num** | 0.4 | Generic numeric types providing unified interface for all numeric operations |
| **num-bigint** | 0.4 | Arbitrary-precision integer arithmetic for exact symbolic computation |
| **num-complex** | 0.4 | Complex number types and operations (Complex64, polar form) |
| **num-rational** | 0.4 | Rational number types for exact fraction arithmetic |
| **argmin** | 0.10 | Numerical optimization framework for root-finding and minimization algorithms |
| **nalgebra** | 0.33 | Linear algebra library for matrix operations and coordinate transformations |
| **fasteval** | 0.2 | Fast expression evaluation engine for numerical computation |
| **swift-bridge** | 0.1 | Swift FFI bindings generator for iOS/macOS integration |
| **serde** | 1.0 | Serialization framework for data persistence and interchange |
| **serde_json** | 1.0 | JSON serialization support for equation and expression data |

### Development Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| **proptest** | 1.9 | Property-based testing framework for randomized test generation |
| **criterion** | 0.5 | Benchmarking framework with statistical analysis and regression detection |

### Dependency Rationale

- **chumsky**: Modern parser combinator with excellent error reporting and composability
- **num ecosystem**: Industry-standard numeric types with comprehensive trait implementations
- **argmin**: Mature optimization library with multiple numerical methods
- **nalgebra**: High-performance linear algebra with extensive coordinate system support
- **swift-bridge**: Zero-overhead Swift FFI with automatic binding generation
- **proptest**: Gold standard for property-based testing in Rust
- **criterion**: Statistical benchmarking with visualization and regression detection

## Version History

### 0.2.0 (2026-01-01) - Multi-Equation System Solver & Calculus

**New Features:**
- **Multi-Equation System Solver** (`equation_system` module):
  - Solve systems of arbitrary equations (algebraic, ODE, differential, integral)
  - Automatic dependency graph construction
  - Topological sorting for optimal solving order
  - Chained solution propagation between equations
  - Unified `SystemResolutionPath` for step-by-step tracking
  - FFI bindings via `solve_equation_system_ffi()`

- **Limits with L'Hôpital's Rule** (`limits` module):
  - Direct substitution for continuous functions
  - Limits at positive/negative infinity
  - One-sided limits (left and right)
  - Automatic L'Hôpital's rule for 0/0 and ∞/∞ indeterminate forms
  - Detection of all indeterminate forms (0·∞, ∞-∞, 0⁰, 1^∞, ∞⁰)
  - Special limits (sin(x)/x, tan(x)/x, (1-cos(x))/x²)

- **Partial Fraction Decomposition** (`partial_fractions` module):
  - Decompose rational functions into partial fractions
  - Support for linear and repeated linear factors
  - Symbolic integration of decomposed forms

- **Pattern Matching** (`pattern` module):
  - Rule-based expression rewriting
  - Wildcard patterns with binding
  - Commutativity-aware matching for + and *
  - Common algebraic rules (identity, zero, double negation, etc.)
  - Apply rules recursively to fixpoint

- **LaTeX Support** (`latex` module):
  - Parse LaTeX mathematical notation
  - Support for `\frac`, `\sqrt`, `\sin`, `\cos`, Greek letters, etc.
  - Render expressions to LaTeX output
  - Display and inline math modes

- **Integration Enhancements** (`integration` module):
  - Integration by parts with step tracking
  - Integration by substitution
  - Tabular integration method
  - Improper integrals to infinity

- **ODE Solving** (`ode` module):
  - First-order separable ODEs
  - First-order linear ODEs
  - Initial value problems

- **Trigonometric Simplification** (`trigonometric` module):
  - Pythagorean identities
  - Double angle formulas
  - Product-to-sum rules
  - Quotient identities
  - Step-by-step simplification

- **Inequality Solving** (`inequality` module):
  - Linear and polynomial inequalities
  - Interval solution representation
  - Systems of inequalities

- **Matrix Expressions** (`matrix` module):
  - Matrix AST representation
  - LaTeX rendering with bracket styles

- **Precision Control** (`precision` module):
  - Configurable evaluation context
  - Multiple rounding modes
  - High-precision computation

**Improvements:**
- Extended FFI bindings for all new features
- 276+ unit tests with comprehensive coverage
- Enhanced resolution path tracking

### 0.1.0 (2025-12-17) - Initial Release

**Implemented:**
- Core AST definitions (`Expression`, `Equation`, `Variable`, `BinaryOp`, `UnaryOp`, `Function`)
- Complete coordinate transformation system:
  - 2D: Cartesian ↔ Polar
  - 3D: Cartesian ↔ Spherical
  - 3D: Cartesian ↔ Cylindrical
  - Full test coverage with round-trip verification
- Complex number operations:
  - De Moivre's theorem implementation
  - Polar form conversion
  - Conjugate and modulus operations
- Expression parser using chumsky with full operator precedence
- Linear equation solver (ax + b = c form)
- Quadratic equation solver (ax² + bx + c = 0 with discriminant)
- Polynomial solver (companion matrix method)
- Transcendental equation solver (trig/exp/log functions)
- Smart solver with automatic method dispatch
- Numerical methods:
  - Newton-Raphson with symbolic differentiation
  - Bisection method
  - Brent's hybrid method
  - Secant method
  - Levenberg-Marquardt
- Resolution path generation for step-by-step solutions
- Symbolic differentiation engine
- Basic symbolic integration
- iOS cross-compilation support:
  - aarch64-apple-ios (device)
  - aarch64-apple-ios-sim (ARM simulator)
  - x86_64-apple-ios (Intel simulator)
  - Universal library creation
- FFI bindings infrastructure with swift-bridge
- Test framework with proptest integration
- Benchmark infrastructure with criterion
- Documentation with comprehensive examples
- Build optimization (LTO, single codegen unit)

### Future Roadmap

**0.3.0** - Advanced Calculus
- Higher-order ODEs
- Systems of ODEs
- Partial derivatives and gradients
- Series expansions (Taylor, Fourier)

**0.4.0** - Units and Dimensions
- Complete unit system
- Automatic conversions
- Dimensional analysis

**1.0.0** - Stable Release
- API stabilization
- Performance optimization
- Comprehensive documentation
- Production-ready for iOS integration

## References

### Rust Language and Ecosystem

- [The Rust Programming Language](https://doc.rust-lang.org/book/) - Official Rust book
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/) - Learn Rust through examples
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/) - API design best practices
- [Cargo Book](https://doc.rust-lang.org/cargo/) - Cargo package manager documentation

### Core Libraries

- [Chumsky Parser Tutorial](https://github.com/zesterer/chumsky) - Parser combinator library
- [Argmin Documentation](https://argmin-rs.org/) - Numerical optimization framework
- [Nalgebra User Guide](https://nalgebra.org/) - Linear algebra library
- [num Crate Documentation](https://docs.rs/num/) - Generic numeric types

### FFI and Mobile

- [Swift-Bridge Guide](https://github.com/chinedufn/swift-bridge) - Swift FFI bindings
- [Rust on iOS - Mozilla Blog](https://blog.mozilla.org/data/2022/01/31/this-week-in-glean-building-and-deploying-a-rust-library-on-ios/) - iOS integration guide
- [Apple Developer: Using Swift with C and Objective-C](https://developer.apple.com/documentation/swift/imported-c-and-objective-c-apis) - Bridging documentation

### Testing and Benchmarking

- [Proptest Book](https://proptest-rs.github.io/proptest/) - Property-based testing
- [Criterion.rs Guide](https://bheisler.github.io/criterion.rs/book/) - Benchmarking framework
- [Rust Testing Guide](https://doc.rust-lang.org/book/ch11-00-testing.html) - Official testing documentation

### Mathematical Algorithms

- [Numerical Recipes](http://numerical.recipes/) - Numerical algorithms reference
- [Wolfram MathWorld](https://mathworld.wolfram.com/) - Mathematical encyclopedia
- [Wikipedia: Coordinate Systems](https://en.wikipedia.org/wiki/List_of_common_coordinate_transformations) - Transformation formulas

### Performance Optimization

- [The Rust Performance Book](https://nnethercote.github.io/perf-book/) - Performance optimization guide
- [Rust Compiler Optimization](https://doc.rust-lang.org/rustc/codegen-options/) - Compiler optimization flags
- [Profile-Guided Optimization](https://doc.rust-lang.org/rustc/profile-guided-optimization.html) - PGO guide

---

**Repository**: [github.com/ChrisGVE/mathsolver-core](https://github.com/ChrisGVE/mathsolver-core)
**Issues**: [Report bugs and request features](https://github.com/ChrisGVE/mathsolver-core/issues)
**Author**: Christian C. Berclaz
**Status**: Active Development (v0.2.0)
