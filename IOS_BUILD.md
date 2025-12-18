# iOS Build Integration Guide

This document describes how to build the mathsolver-core Rust library for iOS and integrate it with the SlipStick Xcode project.

## Prerequisites

- Xcode 14+ with command line tools
- Rust toolchain (install via rustup)
- iOS targets for Rust

### Install iOS Targets

```bash
rustup target add aarch64-apple-ios          # iOS devices (iPhone, iPad)
rustup target add aarch64-apple-ios-sim      # iOS simulator (Apple Silicon Macs)
rustup target add x86_64-apple-ios           # iOS simulator (Intel Macs)
```

## Building for iOS

### 1. Build for Each Target

From the `mathsolver-core` directory:

```bash
# iOS device (arm64)
cargo build --release --target aarch64-apple-ios

# iOS simulator (arm64 - Apple Silicon)
cargo build --release --target aarch64-apple-ios-sim

# iOS simulator (x86_64 - Intel)
cargo build --release --target x86_64-apple-ios
```

### 2. Create Universal Simulator Library

The iOS simulator needs a universal library that works on both Intel and Apple Silicon Macs:

```bash
lipo -create \
  target/aarch64-apple-ios-sim/release/libmathsolver_core.a \
  target/x86_64-apple-ios/release/libmathsolver_core.a \
  -output target/libmathsolver_core_sim.a
```

### 3. Verify Library Architectures

```bash
# Check device library
lipo -info target/aarch64-apple-ios/release/libmathsolver_core.a
# Expected: "Non-fat file ... is architecture: arm64"

# Check simulator library
lipo -info target/libmathsolver_core_sim.a
# Expected: "Architectures in the fat file ... are: x86_64 arm64"
```

## Swift-Bridge Code Generation

The swift-bridge crate automatically generates Swift bindings during the build process.

### Generated Files Location

After building, swift-bridge creates:
- `target/SwiftBridgeCore.swift` - Core Swift bridge code
- `target/mathsolver-core-Bridging-Header.h` - Objective-C bridging header
- `target/mathsolver_core.swift` - Generated Swift API

### Copy Generated Files to Xcode Project

```bash
# From mathsolver-core directory
cp target/SwiftBridgeCore.swift ../SlipStick/Core/MathSolver/
cp target/mathsolver_core.swift ../SlipStick/Core/MathSolver/
cp target/mathsolver-core-Bridging-Header.h ../SlipStick/
```

## Xcode Project Configuration

### 1. Add Static Libraries

1. In Xcode, select the SlipStick project
2. Go to the SlipStick target > Build Phases > Link Binary With Libraries
3. Click + and "Add Other..." > "Add Files..."
4. Add both:
   - `mathsolver-core/target/aarch64-apple-ios/release/libmathsolver_core.a`
   - `mathsolver-core/target/libmathsolver_core_sim.a`

### 2. Configure Library Search Paths

1. Go to Build Settings
2. Search for "Library Search Paths"
3. Add for Debug and Release:
   - `$(PROJECT_DIR)/../mathsolver-core/target/aarch64-apple-ios/release`
   - `$(PROJECT_DIR)/../mathsolver-core/target`

### 3. Configure Header Search Paths

1. In Build Settings, search for "Header Search Paths"
2. Add:
   - `$(PROJECT_DIR)/../mathsolver-core/target`

### 4. Configure Bridging Header

1. In Build Settings, search for "Objective-C Bridging Header"
2. Set to: `SlipStick/mathsolver-core-Bridging-Header.h`

### 5. Add Swift Files to Project

1. In Xcode navigator, right-click SlipStick/Core/MathSolver
2. Choose "Add Files to SlipStick..."
3. Add:
   - `SwiftBridgeCore.swift`
   - `mathsolver_core.swift`
   - `MathSolverBridge.swift` (wrapper)

## Build Configuration by Platform

Xcode automatically selects the correct library based on the build destination:

- **iOS Device**: Uses `libmathsolver_core.a` (arm64)
- **iOS Simulator**: Uses `libmathsolver_core_sim.a` (arm64 + x86_64)

## Automated Build Script

To automate the build process, create a build script in `mathsolver-core/build_ios.sh`:

```bash
#!/bin/bash
set -e

echo "Building mathsolver-core for iOS..."

# Clean previous builds
cargo clean

# Build for all iOS targets
echo "Building for iOS device (arm64)..."
cargo build --release --target aarch64-apple-ios

echo "Building for iOS simulator (arm64)..."
cargo build --release --target aarch64-apple-ios-sim

echo "Building for iOS simulator (x86_64)..."
cargo build --release --target x86_64-apple-ios

# Create universal simulator library
echo "Creating universal simulator library..."
lipo -create \
  target/aarch64-apple-ios-sim/release/libmathsolver_core.a \
  target/x86_64-apple-ios/release/libmathsolver_core.a \
  -output target/libmathsolver_core_sim.a

echo "Verifying libraries..."
lipo -info target/aarch64-apple-ios/release/libmathsolver_core.a
lipo -info target/libmathsolver_core_sim.a

# Copy generated Swift files
echo "Copying generated Swift files..."
mkdir -p ../SlipStick/Core/MathSolver
cp target/SwiftBridgeCore.swift ../SlipStick/Core/MathSolver/ 2>/dev/null || true
cp target/mathsolver_core.swift ../SlipStick/Core/MathSolver/ 2>/dev/null || true
cp target/mathsolver-core-Bridging-Header.h ../SlipStick/ 2>/dev/null || true

echo "iOS build complete!"
echo "Device library: target/aarch64-apple-ios/release/libmathsolver_core.a"
echo "Simulator library: target/libmathsolver_core_sim.a"
```

Make it executable:

```bash
chmod +x mathsolver-core/build_ios.sh
```

## Xcode Build Phase Integration

To rebuild the Rust library automatically when building in Xcode:

1. In Xcode, select SlipStick target
2. Go to Build Phases
3. Click + > New Run Script Phase
4. Name it "Build Rust Library"
5. Add script:

```bash
cd "${PROJECT_DIR}/../mathsolver-core"
./build_ios.sh
```

6. Drag this phase to run before "Compile Sources"

## Troubleshooting

### Library Not Found

If Xcode shows "library not found for -lmathsolver_core":
1. Verify library search paths are correct
2. Check that libraries exist in specified locations
3. Clean build folder (Cmd+Shift+K) and rebuild

### Swift-Bridge Errors

If you get "Use of unresolved identifier" for FFI functions:
1. Ensure bridging header is correctly set
2. Verify generated Swift files are in project
3. Check that swift-bridge version matches between Rust and Swift

### Architecture Mismatch

If you get "architecture arm64 not found":
1. Verify you built for the correct target
2. Check lipo output shows expected architectures
3. Ensure universal library was created successfully

### Build Script Permissions

If build_ios.sh won't execute:
```bash
chmod +x mathsolver-core/build_ios.sh
```

## Testing the Integration

Create a simple test in Swift to verify the integration:

```swift
import XCTest

class MathSolverBridgeTests: XCTestCase {
    func testCoordinateTransform() {
        let solver = MathSolverBridge()
        let (r, theta) = solver.cartesianToPolar(x: 3.0, y: 4.0)

        XCTAssertEqual(r, 5.0, accuracy: 0.001)
        XCTAssertEqual(theta, atan2(4.0, 3.0), accuracy: 0.001)
    }

    func testComplexMultiplication() {
        let solver = MathSolverBridge()
        let a = (real: 2.0, imaginary: 3.0)
        let b = (real: 4.0, imaginary: 5.0)
        let result = solver.complexMultiply(a: a, b: b)

        // (2+3i)(4+5i) = 8+10i+12i+15i² = 8+22i-15 = -7+22i
        XCTAssertEqual(result.real, -7.0, accuracy: 0.001)
        XCTAssertEqual(result.imaginary, 22.0, accuracy: 0.001)
    }
}
```

## Performance Considerations

- Static libraries increase app size (~few MB depending on features)
- Release builds use full optimizations (LTO enabled in Cargo.toml)
- Consider using dynamic library for development, static for production
- Profile hot paths and benchmark if performance is critical

## References

- [swift-bridge Documentation](https://chinedufn.github.io/swift-bridge/)
- [Rust on iOS - Mozilla Blog](https://blog.mozilla.org/data/2022/01/31/this-week-in-glean-building-and-deploying-a-rust-library-on-ios/)
- [Apple Developer: Using Swift with C and Objective-C](https://developer.apple.com/documentation/swift/imported-c-and-objective-c-apis)
