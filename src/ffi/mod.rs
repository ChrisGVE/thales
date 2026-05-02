//! FFI exports for Swift interoperability using swift-bridge.
//!
//! v0.9.0: All operations are accessible through the single JSON entry
//! point [`execute_json_ffi`]. Legacy per-operation wrappers have been
//! removed.

mod json_impl;
use json_impl::*;

#[swift_bridge::bridge]
mod ffi {
    extern "Rust" {
        fn execute_json_ffi(request_json: &str) -> Result<String, String>;
    }
}
