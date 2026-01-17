# Thales Swift Package

Swift bindings for the [thales](https://crates.io/crates/thales) Computer Algebra System library.

## Requirements

- iOS 14+ / macOS 11+
- Xcode 14+
- Pre-built `libthales.a` static library or XCFramework

## Installation

### Option 1: Swift Package Manager with Pre-built Library (Recommended)

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/ChrisGVE/thales.git", from: "0.3.0")
]
```

Or in Xcode: File > Add Package Dependencies > Enter the repository URL.

**Important**: You must also configure your project to link against the pre-built Rust library. See [Configuration](#configuration) below.

### Option 2: Using XCFramework from Releases

Starting with version 0.4.0, pre-built XCFrameworks are available as GitHub release assets:

1. Download `Thales.xcframework.zip` from the [latest release](https://github.com/ChrisGVE/thales/releases)
2. Extract and add the XCFramework to your Xcode project
3. The Package.swift will automatically use the binary target

### Option 3: Manual Installation

1. Copy the Swift files from `swift/Sources/Thales/` to your project
2. Copy the headers from `swift/Sources/Thales/include/` to your project
3. Build the Rust library (see below)
4. Configure your bridging header

## Building the Rust Library

Before using this package, you must build the Rust library:

```bash
# Clone the repository
git clone https://github.com/ChrisGVE/thales.git
cd thales

# Install iOS targets (one-time)
rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios

# Build for iOS device
cargo build --release --features ffi --target aarch64-apple-ios

# Build for iOS simulator (Apple Silicon)
cargo build --release --features ffi --target aarch64-apple-ios-sim

# Build for iOS simulator (Intel)
cargo build --release --features ffi --target x86_64-apple-ios

# Create universal simulator library
lipo -create \
    target/aarch64-apple-ios-sim/release/libthales.a \
    target/x86_64-apple-ios/release/libthales.a \
    -output target/libthales_sim_universal.a
```

## Configuration

### Xcode Project Settings

1. **Library Search Paths** (Build Settings):
   ```
   $(PROJECT_DIR)/path/to/thales/target/aarch64-apple-ios/release
   $(PROJECT_DIR)/path/to/thales/target
   ```

2. **Other Linker Flags** (Build Settings):
   ```
   -lthales -lresolv
   ```

3. **Bridging Header** (if not using SPM):
   Create or update your bridging header:
   ```c
   #import "SwiftBridgeCore.h"
   #import "thales.h"
   ```

## Usage

Thales provides a clean Swift API through the `Thales` namespace:

```swift
import Thales

// Solve an equation
let solution = try Thales.solve("2*x + 5 = 13", for: "x")
print(solution) // "x = 4"

// Differentiate an expression
let derivative = try Thales.differentiate("x^3 + 2*x", withRespectTo: "x")
print(derivative.derivative) // "3*x^2 + 2"

// Integrate an expression
let integral = try Thales.integrate("3*x^2", withRespectTo: "x")
print(integral.integral) // "x^3 + C"

// Coordinate transformations
let point = Point2D(x: 3.0, y: 4.0)
let polar = point.toPolar()
print("r = \(polar.r), theta = \(polar.theta)")
// r = 5.0, theta = 0.927...

// Complex numbers
let z = Complex(real: 1.0, imaginary: 1.0)
let squared = z.power(2)
print("z^2 = \(squared.real) + \(squared.imaginary)i")
```

## API Reference

See the [thales documentation](https://docs.rs/thales) for complete API reference.

### Equation Solving
- `Thales.solve(_:for:)` - Solve an equation for a variable
- `Thales.solve(_:for:knownValues:)` - Solve with known values
- `Thales.solveNumerically(_:for:initialGuess:)` - Numerical root finding
- `Thales.solveSystem(equations:)` - Solve equation systems
- `Thales.solveEquationSystem(equations:knownValues:targets:)` - Advanced system solver
- `Thales.solveInequality(_:for:)` - Solve inequalities

### Calculus
- `Thales.differentiate(_:withRespectTo:)` - Symbolic differentiation
- `Thales.nthDerivative(_:withRespectTo:order:)` - Higher-order derivatives
- `Thales.gradient(_:variables:)` - Gradient of multivariable functions
- `Thales.integrate(_:withRespectTo:)` - Indefinite integration
- `Thales.definiteIntegral(_:withRespectTo:from:to:)` - Definite integrals
- `Thales.limit(_:as:approaches:)` - Limit computation
- `Thales.limitToInfinity(_:as:)` - Limits at infinity

### Coordinate Systems
- `Point2D` - 2D Cartesian coordinates with `toPolar()`
- `Point3D` - 3D Cartesian coordinates with `toSpherical()`
- `PolarPoint` - 2D polar coordinates with `toCartesian()`
- `SphericalPoint` - 3D spherical coordinates with `toCartesian()`

### Complex Numbers
- `Complex` - Complex number type with arithmetic operators
- `Complex.power(_:)` - De Moivre's theorem
- `Complex.conjugate` - Complex conjugate
- `Complex.modulus` - Magnitude
- `Complex.toPolar()` - Convert to polar form

### Simplification
- `Thales.simplify(_:)` - Algebraic simplification
- `Thales.simplifyTrig(_:)` - Trigonometric simplification
- `Thales.partialFractions(numerator:denominator:variable:)` - Partial fraction decomposition

### LaTeX
- `Thales.parseLatex(_:)` - Parse LaTeX notation
- `Thales.toLatex(_:)` - Convert to LaTeX

### Evaluation
- `Thales.evaluate(_:with:)` - Evaluate expression with values

### Parsing
- `Thales.parseEquation(_:)` - Parse equation string
- `Thales.parseExpression(_:)` - Parse expression string

## Documentation

Documentation is automatically generated and hosted on [Swift Package Index](https://swiftpackageindex.com/ChrisGVE/thales/documentation/thales).

## License

MIT License - see [LICENSE](../LICENSE)
