# Thales Swift Package

Swift bindings for the [thales](https://crates.io/crates/thales) Computer Algebra System library.

## Requirements

- iOS 14+ / macOS 11+
- Xcode 14+
- Pre-built `libthales.a` static library (see Building the Rust Library below)

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/ChrisGVE/thales.git", from: "0.3.0")
]
```

Or in Xcode: File > Add Package Dependencies > Enter the repository URL.

**Important**: You must also link against the pre-built Rust library. See Configuration below.

### Manual Installation

1. Copy the Swift files from `Sources/Thales/` to your project
2. Copy the headers from `Sources/Thales/include/` to your project
3. Configure your bridging header (see Configuration)

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

```swift
import Thales

// Parse and solve an equation
let equation = parse_equation_ffi("2*x + 5 = 13")
let solution = solve_equation_ffi(equation, "x")

// Coordinate transformations
let cartesian = Cartesian2D(x: 3.0, y: 4.0)
let polar = cartesian_to_polar_ffi(cartesian)
// polar.r == 5.0, polar.theta == 0.927...

// Complex numbers
let z = Complex64(re: 1.0, im: 1.0)
let result = de_moivre_ffi(z, 2.0)
```

## Available Functions

See the [thales documentation](https://docs.rs/thales) for complete API reference.

### Equation Solving
- `parse_equation_ffi(_:)` - Parse equation string
- `parse_expression_ffi(_:)` - Parse expression string
- `solve_equation_ffi(_:_:)` - Solve for variable
- `solve_equation_system_ffi(_:_:_:)` - Solve equation systems

### Coordinate Transformations
- `cartesian_to_polar_ffi(_:)` / `polar_to_cartesian_ffi(_:)`
- `cartesian_to_spherical_ffi(_:)` / `spherical_to_cartesian_ffi(_:)`
- `cartesian_to_cylindrical_ffi(_:)` / `cylindrical_to_cartesian_ffi(_:)`

### Complex Numbers
- `de_moivre_ffi(_:_:)` - De Moivre's theorem
- `complex_conjugate_ffi(_:)` - Complex conjugate
- `complex_modulus_ffi(_:)` - Modulus/magnitude

### Calculus
- `differentiate_ffi(_:_:)` - Symbolic differentiation
- `integrate_ffi(_:_:)` - Symbolic integration

### Series
- `taylor_series_ffi(_:_:_:_:)` - Taylor expansion
- `maclaurin_series_ffi(_:_:_:)` - Maclaurin expansion

## License

MIT License - see [LICENSE](../LICENSE)
