//! Swift bridge definitions for code generation.
//!
//! This file is parsed by swift-bridge-build to generate Swift and C header
//! files. It must stay in sync with src/ffi/mod.rs.

#[swift_bridge::bridge]
mod ffi {
    extern "Rust" {
        fn execute_json_ffi(request_json: &str) -> Result<String, String>;
    }
}
