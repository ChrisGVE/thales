# thales

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/ChrisGVE/thales)](https://github.com/ChrisGVE/thales/releases)
[![CI](https://github.com/ChrisGVE/thales/actions/workflows/ci.yml/badge.svg)](https://github.com/ChrisGVE/thales/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/thales.svg)](https://crates.io/crates/thales)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org)
[![Documentation](https://docs.rs/thales/badge.svg)](https://docs.rs/thales)
[![Swift Versions](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FChrisGVE%2Fthales%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/ChrisGVE/thales)
[![Platforms](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FChrisGVE%2Fthales%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/ChrisGVE/thales)
[![Swift Documentation](https://img.shields.io/badge/Swift-Documentation-orange)](https://swiftpackageindex.com/ChrisGVE/thales/documentation)

A comprehensive Computer Algebra System (CAS) library for symbolic mathematics, equation solving, calculus, and numerical methods. Named after [Thales of Miletus](https://en.wikipedia.org/wiki/Thales_of_Miletus), the first mathematician in the Greek tradition.

**[Full Documentation on docs.rs](https://docs.rs/thales)**

## Features

- **Expression Parsing** - Parse mathematical expressions with full operator precedence
- **Equation Solving** - Linear, quadratic, polynomial, transcendental, and systems of equations
- **Calculus** - Differentiation, integration, limits, Taylor series, ODEs
- **Numerical Methods** - Newton-Raphson, bisection, Brent's method when symbolic fails
- **Coordinate Systems** - 2D/3D transformations, complex numbers, De Moivre's theorem
- **Units & Dimensions** - Dimensional analysis and unit conversion
- **iOS Support** - FFI bindings for Swift via swift-bridge

## Installation

```toml
[dependencies]
thales = "0.3.3"
```

## Quick Start

### Solve an Equation

```rust
use thales::{parse_equation, SmartSolver, Solver, Variable};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let equation = parse_equation("2*x + 5 = 13")?;
    let solver = SmartSolver::new();
    let x = Variable::new("x");
    let (solution, path) = solver.solve(&equation, &x)?;

    // View step-by-step solution
    for step in path.steps() {
        println!("{}", step);
    }
    Ok(())
}
```

### Coordinate Transformations

```rust
use thales::{Cartesian2D, Polar};

fn main() {
    let point = Cartesian2D::new(3.0, 4.0);
    let polar = point.to_polar();

    assert!((polar.r - 5.0).abs() < 1e-10);

    // Round-trip conversion
    let back = polar.to_cartesian();
    assert!((back.x - 3.0).abs() < 1e-10);
}
```

## Documentation

The full documentation is available on **[docs.rs/thales](https://docs.rs/thales)**, including:

- **[User Guides](https://docs.rs/thales/latest/thales/guides/)** - Step-by-step tutorials
- **[API Reference](https://docs.rs/thales/latest/thales/)** - Complete type and function documentation
- **[Examples](https://docs.rs/thales/latest/thales/#quick-start)** - Working code examples

### Guides

| Guide | Description |
|-------|-------------|
| [Solving Equations](https://docs.rs/thales/latest/thales/guides/solving_equations/) | Linear, quadratic, polynomial, and systems |
| [Calculus Operations](https://docs.rs/thales/latest/thales/guides/calculus_operations/) | Derivatives, integrals, limits, ODEs |
| [Series Expansions](https://docs.rs/thales/latest/thales/guides/series_expansions/) | Taylor, Maclaurin, Laurent, asymptotic |
| [Coordinate Systems](https://docs.rs/thales/latest/thales/guides/coordinate_systems/) | 2D/3D transforms, complex numbers |
| [Numerical Methods](https://docs.rs/thales/latest/thales/guides/numerical_methods/) | Root-finding algorithms |
| [Working with Units](https://docs.rs/thales/latest/thales/guides/working_with_units/) | Dimensional analysis |
| [Error Handling](https://docs.rs/thales/latest/thales/guides/error_handling/) | ThalesError patterns |

## LaTeX Support

Thales can parse LaTeX mathematical notation into its internal expression tree via `parse_latex`.

### Supported constructs

| Category | LaTeX syntax | Examples |
|----------|-------------|---------|
| Fractions | `\frac{num}{denom}` | `\frac{1}{2}`, `\frac{x+1}{y}` |
| Square root | `\sqrt{x}` | `\sqrt{2}`, `\sqrt{x+1}` |
| nth root | `\sqrt[n]{x}` | `\sqrt[3]{8}`, `\sqrt[n]{x}` |
| Superscripts | `x^{n}` or `x^n` | `x^{2}`, `e^{-x}` |
| Subscripts | `x_{n}` or `x_n` | `x_{1}`, `x_{12}` |
| Greek letters | `\alpha`, `\beta`, `\pi`, etc. | `\alpha`, `\theta`, `\pi` |
| Trig functions | `\sin`, `\cos`, `\tan`, etc. | `\sin{x}`, `\cos(\theta)` |
| Logarithms / exp | `\ln`, `\log`, `\exp` | `\ln{x}`, `\log_{10}{x}`, `\log_{2}{8}` |
| Integrals | `\int_{a}^{b} expr \, dx` | `\int_{0}^{1} x \, dx`, `\int x dx` |
| Limits | `\lim_{x \to a}` | `\lim_{x \to 0} x`, `\lim_{x \to \infty} x` |
| Sums | `\sum_{i=a}^{b}` | `\sum_{i=1}^{10} i` |
| Operators | `\cdot`, `\times`, `\div`, `\pm` | `a \cdot b`, `2 \times 3` |

### Not yet supported

The following constructs are not currently parsed and will return an error:

- Double/contour integrals: `\iint`, `\oint`
- Products: `\prod`
- Partial derivatives: `\partial`
- Matrix environments: `\begin{matrix}`, `\begin{pmatrix}`, `\begin{bmatrix}`, etc.

## Optional LAPACK Acceleration

Enable hardware-accelerated matrix operations (eigenvalues, eigenvectors, QR decomposition, linear system solving) by selecting a LAPACK backend:

```toml
# macOS / iOS — links against Apple Accelerate.framework
thales = { version = "0.3", features = ["lapack-accelerate"] }

# Linux — uses reference LAPACK (requires liblapack-dev / gfortran)
thales = { version = "0.3", features = ["lapack-netlib"] }

# Linux — uses OpenBLAS (requires libopenblas-dev)
thales = { version = "0.3", features = ["lapack-openblas"] }
```

The `lapack` feature is an alias for `lapack-accelerate`. Without any LAPACK feature, thales uses pure-Rust implementations for all numerical linear algebra.

## iOS Cross-Compilation

Build for iOS with FFI support:

```bash
# Add iOS targets
rustup target add aarch64-apple-ios aarch64-apple-ios-sim

# Build for device
cargo build --release --features ffi --target aarch64-apple-ios
```

See [IOS_BUILD.md](IOS_BUILD.md) for complete iOS integration instructions.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- **Crate**: [crates.io/crates/thales](https://crates.io/crates/thales)
- **Documentation**: [docs.rs/thales](https://docs.rs/thales)
- **Repository**: [github.com/ChrisGVE/thales](https://github.com/ChrisGVE/thales)
- **Issues**: [Report bugs](https://github.com/ChrisGVE/thales/issues)
