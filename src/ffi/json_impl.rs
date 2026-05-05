//! FFI implementation for the unified JSON entry point.

/// Canonical cross-language entry point.
///
/// Accepts a JSON-encoded request and returns a JSON-encoded response.
/// All new operations are accessible through this function.
///
/// See [`crate::api::json::execute_ffi`] for the wire format.
pub(super) fn execute_json_ffi(request_json: &str) -> Result<String, String> {
    crate::api::json::execute_ffi(request_json)
}
