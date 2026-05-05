//! JSON wire protocol for [`super::execute`].
//!
//! v0.9.0 ships a derive-based serde schema: each JSON request carries a
//! `command.type` tag plus native [`mathlex::Expression`] JSON objects for all
//! expression fields. The dispatcher converts mathlex expressions to the
//! internal thales representation via [`crate::mathlex_bridge`].
//! Responses serialise `Expression` values back to structured mathlex JSON
//! using the canonical display string as an intermediate parse step.
//!
//! Module layout:
//! - [`request`]: `JsonCommand`, `JsonRequest`, `JsonPrecision`, `JsonBudget`,
//!   and `request_from_json` parsing.
//! - [`response`]: `response_to_json` and supporting serialisers.
//! - [`tests`]: `#[cfg(test)]` integration tests.

mod request;
mod response;
#[cfg(test)]
mod tests;

use crate::ThalesError;

/// FFI-shaped entry point: JSON request → JSON response, with errors
/// stringified for cross-language transport.
pub fn execute_ffi(request_json: &str) -> Result<String, String> {
    let request_val: serde_json::Value =
        serde_json::from_str(request_json).map_err(|e| format!("invalid JSON request: {}", e))?;
    let request = request::request_from_json(&request_val)?;
    let response = super::dispatch::execute(request)
        .map_err(|e: ThalesError| format!("dispatch error: {}", e))?;
    let response_val = response::response_to_json(&response);
    serde_json::to_string(&response_val).map_err(|e| format!("failed to serialise response: {}", e))
}
