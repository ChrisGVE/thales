#!/bin/bash
set -e

echo "Building mathsolver-core for iOS..."

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

# Copy generated Swift files if they exist
echo "Copying generated Swift files..."
mkdir -p ../SlipStick/Core/MathSolver
if [ -f "target/SwiftBridgeCore.swift" ]; then
    cp target/SwiftBridgeCore.swift ../SlipStick/Core/MathSolver/
fi
if [ -f "target/mathsolver_core.swift" ]; then
    cp target/mathsolver_core.swift ../SlipStick/Core/MathSolver/
fi
if [ -f "target/mathsolver-core-Bridging-Header.h" ]; then
    cp target/mathsolver-core-Bridging-Header.h ../SlipStick/
fi

echo "iOS build complete!"
echo "Device library: target/aarch64-apple-ios/release/libmathsolver_core.a"
echo "Simulator library: target/libmathsolver_core_sim.a"
