# iOS Build Integration Guide

This document describes how to build the thales Rust library for iOS and integrate it with your Xcode project.

## Prerequisites

- Xcode 14+ with command line tools
- Rust toolchain (install via [rustup](https://rustup.rs))
- iOS targets for Rust

### Install iOS Targets

```bash
rustup target add aarch64-apple-ios          # iOS devices (iPhone, iPad)
rustup target add aarch64-apple-ios-sim      # iOS simulator (Apple Silicon Macs)
rustup target add x86_64-apple-ios           # iOS simulator (Intel Macs)
```

## Building for iOS

### 1. Build for Each Target

From the `thales` directory:

```bash
# iOS device (arm64)
cargo build --release --features ffi --target aarch64-apple-ios

# iOS simulator (arm64 - Apple Silicon)
cargo build --release --features ffi --target aarch64-apple-ios-sim

# iOS simulator (x86_64 - Intel)
cargo build --release --features ffi --target x86_64-apple-ios
```

### 2. Create Universal Simulator Library

The iOS simulator needs a universal library that works on both Intel and Apple Silicon Macs:

```bash
lipo -create \
  target/aarch64-apple-ios-sim/release/libthales.a \
  target/x86_64-apple-ios/release/libthales.a \
  -output target/libthales_sim_universal.a
```

### 3. Verify Library Architectures

```bash
# Check device library
lipo -info target/aarch64-apple-ios/release/libthales.a
# Expected: "Non-fat file ... is architecture: arm64"

# Check simulator library
lipo -info target/libthales_sim_universal.a
# Expected: "Architectures in the fat file ... are: x86_64 arm64"
```

## Swift Package Manager Integration

The easiest way to integrate thales is via Swift Package Manager:

1. Add the package to your Xcode project:
   - File > Add Package Dependencies
   - Enter: `https://github.com/ChrisGVE/thales.git`
   - Select version `0.3.0` or later

2. Configure Library Search Paths (Build Settings):
   ```
   $(PROJECT_DIR)/path/to/thales/target/aarch64-apple-ios/release
   $(PROJECT_DIR)/path/to/thales/target
   ```

3. The Swift bindings are automatically included via the package.

## Manual Integration

### Swift-Bridge Generated Files

The swift-bridge crate generates Swift bindings during the build process. Pre-generated files are included in the `swift/` directory:

- `swift/Sources/Thales/thales.swift` - Generated Swift API
- `swift/Sources/Thales/SwiftBridgeCore.swift` - Core Swift bridge code
- `swift/Sources/Thales/include/thales.h` - C header
- `swift/Sources/Thales/include/SwiftBridgeCore.h` - Bridge header

### Copy Files to Your Project

```bash
# Copy Swift files
cp swift/Sources/Thales/*.swift YourProject/Sources/

# Copy headers
cp swift/Sources/Thales/include/*.h YourProject/Headers/
```

### Regenerating Swift Files (Optional)

If you modify the FFI interface, regenerate the Swift files:

```bash
cargo build --features ffi
# Files are generated in target/debug/build/thales-*/out/
```

## Xcode Project Configuration

### 1. Add Static Libraries

1. In Xcode, select your project in the navigator
2. Select your target > Build Phases > Link Binary With Libraries
3. Click + and "Add Other..." > "Add Files..."
4. Add:
   - For device: `thales/target/aarch64-apple-ios/release/libthales.a`
   - For simulator: `thales/target/libthales_sim_universal.a`

### 2. Configure Library Search Paths

In Build Settings > Library Search Paths:

**Debug** (simulator):
```
$(PROJECT_DIR)/../thales/target
```

**Release** (device):
```
$(PROJECT_DIR)/../thales/target/aarch64-apple-ios/release
```

Or use conditional settings:
```
$(PROJECT_DIR)/../thales/target/$(PLATFORM_NAME)/release
$(PROJECT_DIR)/../thales/target
```

### 3. Configure Header Search Paths

In Build Settings > Header Search Paths:
```
$(PROJECT_DIR)/../thales/swift/Sources/Thales/include
```

### 4. Set Up Bridging Header

If not using SPM, create `YourProject-Bridging-Header.h`:

```c
#ifndef YourProject_Bridging_Header_h
#define YourProject_Bridging_Header_h

#import "SwiftBridgeCore.h"
#import "thales.h"

#endif
```

In Build Settings > Objective-C Bridging Header:
```
$(PROJECT_DIR)/YourProject/YourProject-Bridging-Header.h
```

### 5. Other Linker Flags

In Build Settings > Other Linker Flags:
```
-lresolv
```

This is required by swift-bridge for DNS resolution support.

## Usage in Swift

```swift
import Thales  // If using SPM

// Parse and solve an equation
let equation = parse_equation_ffi("2*x + 5 = 13")
let solution = solve_equation_ffi(equation, "x")

// Coordinate transformations
let point = Cartesian2D(x: 3.0, y: 4.0)
let polar = cartesian_to_polar_ffi(point)
print("r = \(polar.r), θ = \(polar.theta)")

// Complex number operations
let z = Complex64(re: 1.0, im: 1.0)
let squared = de_moivre_ffi(z, 2.0)

// Differentiation
let expr = parse_expression_ffi("x^3 + 2*x")
let derivative = differentiate_ffi(expr, "x")

// Taylor series
let series = taylor_series_ffi("sin(x)", "x", 0.0, 5)
```

## Build Script (Optional)

Create a `build_ios.sh` script for convenience:

```bash
#!/bin/bash
set -e

echo "Building thales for iOS..."

# Build all targets
cargo build --release --features ffi --target aarch64-apple-ios
cargo build --release --features ffi --target aarch64-apple-ios-sim
cargo build --release --features ffi --target x86_64-apple-ios

# Create universal simulator library
echo "Creating universal simulator library..."
lipo -create \
    target/aarch64-apple-ios-sim/release/libthales.a \
    target/x86_64-apple-ios/release/libthales.a \
    -output target/libthales_sim_universal.a

# Verify
echo "Verifying architectures..."
lipo -info target/aarch64-apple-ios/release/libthales.a
lipo -info target/libthales_sim_universal.a

echo "Done! Libraries are ready in target/"
```

Make it executable:
```bash
chmod +x build_ios.sh
./build_ios.sh
```

## Troubleshooting

### "Library not found for -lthales"

Ensure Library Search Paths includes the directory containing `libthales.a`.

### "Symbol not found" at runtime

Make sure you're linking the correct architecture:
- Simulator: `libthales_sim_universal.a`
- Device: `libthales.a` from `aarch64-apple-ios`

### Swift bridging header errors

1. Verify header search paths include the `include/` directory
2. Check that both `SwiftBridgeCore.h` and `thales.h` are accessible
3. Clean build folder (Cmd+Shift+K) and rebuild

### FFI function not found

Ensure the `ffi` feature is enabled when building:
```bash
cargo build --release --features ffi --target aarch64-apple-ios
```

## Resources

- [thales documentation](https://docs.rs/thales)
- [swift-bridge documentation](https://github.com/chinedufn/swift-bridge)
- [Rust on iOS guide](https://mozilla.github.io/firefox-browser-architecture/experiments/2017-09-06-rust-on-ios.html)
