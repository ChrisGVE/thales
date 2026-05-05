//! Request parsing: derive-based [`JsonCommand`] mirror schema,
//! [`JsonRequest`], [`JsonPrecision`], [`JsonBudget`], and their conversions
//! into the internal [`Request`] / [`Command`] types.
//!
//! Submodule layout:
//! - [`schema`]: serde-derived JSON types (mirror schema).
//! - [`parsers`]: string → domain / enum parsers.
//! - [`convert`]: `JsonCommand` → `Command` and `JsonRequest` → `Request`.

pub(super) mod convert;
pub(super) mod parsers;
pub(super) mod schema;

// Re-export the entry point used by mod.rs.
pub(super) use convert::request_from_json;
