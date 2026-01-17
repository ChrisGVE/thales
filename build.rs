//! Build script for generating Swift bridge code.
//!
//! This script uses swift-bridge-build to generate Swift and C header files
//! from the FFI definitions in src/bridge.rs.
//!
//! Generated files are written to:
//! - `$OUT_DIR/thales/thales.swift` - Swift bindings
//! - `$OUT_DIR/thales/thales.h` - C header for Swift interop
//! - `$OUT_DIR/SwiftBridgeCore.swift` - Swift runtime support
//! - `$OUT_DIR/SwiftBridgeCore.h` - C runtime header
//!
//! Note: src/bridge.rs contains a simplified copy of the FFI interface
//! from src/ffi.rs without doc comments, as swift-bridge-build has
//! parsing issues with doc comments inside the bridge module.

use std::path::PathBuf;

fn main() {
    // Rerun if either the bridge definitions or FFI implementations change
    println!("cargo::rerun-if-changed=src/ffi.rs");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let bridges = vec!["src/bridge.rs"];

    for path in &bridges {
        println!("cargo::rerun-if-changed={}", path);
    }

    swift_bridge_build::parse_bridges(bridges).write_all_concatenated(&out_dir, "thales");

    // Print the output directory for debugging
    println!("cargo::warning=Swift bridge output: {}", out_dir.display());
}
