# mathsolver-core

A high-performance Rust library for mathematical equation parsing, symbolic solving, numerical approximation, and coordinate transformations.

## Features

- **Expression Parsing**: Parse mathematical expressions and equations with full operator precedence
- **Symbolic Solving**: Solve linear, quadratic, polynomial, and transcendental equations
- **Numerical Methods**: Newton-Raphson, secant method, bisection, Brent's method, and more
- **Coordinate Transformations**: Convert between Cartesian, polar, spherical, and cylindrical coordinates
- **Complex Numbers**: Full support for complex arithmetic and polar form operations
- **Unit System**: Dimensional analysis and automatic unit conversion
- **Resolution Paths**: Track step-by-step solution processes
- **FFI Support**: Swift bindings for iOS integration via swift-bridge

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
mathsolver-core = "0.1.0"
```

## Usage

### Parsing Equations

```rust
use mathsolver_core::parse_equation;

let equation = parse_equation("2*x + 5 = 13")?;
```

### Solving Equations

```rust
use mathsolver_core::{SmartSolver, Solver, Variable};

let solver = SmartSolver::new();
let (solution, path) = solver.solve(&equation, &Variable::new("x"))?;
```

### Coordinate Transformations

```rust
use mathsolver_core::{Cartesian2D, Polar};

let cart = Cartesian2D::new(3.0, 4.0);
let polar = cart.to_polar();
assert_eq!(polar.r, 5.0);
```

### Complex Number Operations

```rust
use mathsolver_core::ComplexOps;
use num_complex::Complex64;

let c = Complex64::new(1.0, 1.0);
let result = ComplexOps::de_moivre(c, 2.0);
```

## iOS Integration

Build for iOS targets:

```bash
# Add iOS targets
rustup target add aarch64-apple-ios
rustup target add x86_64-apple-ios
rustup target add aarch64-apple-ios-sim

# Build for iOS device
cargo build --release --target aarch64-apple-ios

# Build for iOS simulator (Intel)
cargo build --release --target x86_64-apple-ios

# Build for iOS simulator (ARM)
cargo build --release --target aarch64-apple-ios-sim
```

Enable FFI feature for Swift bindings:

```bash
cargo build --release --features ffi --target aarch64-apple-ios
```

## Architecture

The library is organized into focused modules:

- `ast` - Abstract syntax tree definitions
- `parser` - Expression and equation parsing with chumsky
- `solver` - Symbolic equation solving algorithms
- `numerical` - Numerical approximation methods with argmin
- `resolution_path` - Solution step tracking
- `dimensions` - Unit and dimension handling
- `transforms` - Coordinate system transformations with nalgebra
- `ffi` - Foreign function interface for Swift

## Performance

Optimized for performance with:

- Zero-cost abstractions
- Link-time optimization (LTO)
- Single codegen unit for maximum optimization
- Minimal allocations
- Cache-efficient algorithms

Run benchmarks:

```bash
cargo bench
```

## Testing

Comprehensive test suite with:

- Unit tests for all modules
- Integration tests for complete workflows
- Property-based tests with proptest
- Numerical accuracy validation

Run tests:

```bash
cargo test
```

Run tests with ignored tests (requires full implementation):

```bash
cargo test -- --ignored
```

## Development Status

Current version: 0.1.0

This is an initial release with core structure in place. Many features are stubbed out with TODO comments indicating future implementation. The coordinate transformation and complex number modules are functional.

### Implemented

- Core AST definitions
- Coordinate transformations (2D/3D)
- Complex number operations
- Project structure and module organization
- Test and benchmark infrastructure

### In Progress

- Expression parser with chumsky
- Symbolic equation solving
- Numerical approximation methods
- Unit and dimension system
- Resolution path generation

## Contributing

This is a private project for use with SlipStick iOS app. Contributions are welcome via pull requests.

### Code Standards

- Follow Rust 2021 idioms
- Zero unsafe code outside core abstractions
- `clippy::pedantic` compliance
- Comprehensive documentation with examples
- Complete test coverage including doctests
- Benchmark performance-critical code

### Commit Format

Use conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: feat, fix, docs, style, refactor, test, chore, perf, ci, build

## License

MIT License - Copyright (c) 2025 Christian C. Berclaz

## Dependencies

### Core Dependencies

- **chumsky** - Parser combinator library
- **num** - Generic numeric types
- **argmin** - Numerical optimization
- **nalgebra** - Linear algebra
- **fasteval** - Fast expression evaluation
- **swift-bridge** - Swift FFI bindings

### Development Dependencies

- **proptest** - Property-based testing
- **criterion** - Benchmarking framework

## References

- [Rust Book](https://doc.rust-lang.org/book/)
- [Chumsky Parser Tutorial](https://github.com/zesterer/chumsky)
- [Argmin Documentation](https://argmin-rs.org/)
- [Nalgebra User Guide](https://nalgebra.org/)
- [Swift-Bridge Guide](https://github.com/chinedufn/swift-bridge)

## Version History

### 0.1.0 (2025-12-17)

- Initial project structure
- Core module definitions
- Coordinate transformation implementation
- Complex number operations
- Test and benchmark infrastructure
- iOS cross-compilation support
