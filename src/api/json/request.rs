//! Request parsing: derive-based [`JsonCommand`] mirror schema,
//! [`JsonRequest`], [`JsonPrecision`], [`JsonBudget`], and their conversions
//! into the internal [`Request`] / [`Command`] types.
//!
//! Submodule layout:
//! - [`schema`]: serde-derived JSON types (mirror schema).
//! - [`parsers`]: string → domain / enum parsers and expression helpers.
//! - [`convert`]: `JsonCommand` → `Command` and `JsonRequest` → `Request`.

pub(super) mod convert;
pub(super) mod parsers;
pub(super) mod schema;

// Re-export the two functions used by mod.rs and tests.rs.
pub(super) use convert::request_from_json;
pub(super) use parsers::parse_expr_str;
