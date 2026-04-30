//! Public API surface for thales — single entry point and single output point.
//!
//! This module implements architecture rules 3 and 4:
//!
//! - **Rule 3 — single entry point**: [`execute`] is the one function callers
//!   invoke. It accepts a [`Request`] packaging a [`Command`] with its
//!   `Expression` inputs and returns a [`Response`].
//!
//! - **Rule 4 — single output point**: [`Response`] carries a map of
//!   [`ResultKey`] → [`ResultEntry`]. Each entry packages the result value
//!   (symbolic, numeric, or hybrid), shape, optional unit, narrated steps with
//!   technique tag / difficulty / Markdown narrative / positional path,
//!   alternative equivalent forms, and engine provenance. The response also
//!   carries diagnostics, assumptions, and execution metadata.
//!
//! FFI JSON transport lives in [`json`] and calls the same dispatcher as this
//! public Rust entry point.

pub mod command;
pub mod condition;
pub mod diagnostic;
pub mod dispatch;
pub mod domain;
pub mod json;
pub mod narrative;
pub mod narratives;
pub mod path;
pub mod render;
pub mod request;
pub mod response;

pub use command::Command;
pub use condition::{Bound, CompoundCondition, Condition};
pub use diagnostic::{Assumption, Diagnostic, DiagnosticCode, Severity};
pub use domain::{BaseDomain, Domain, DomainExpr, DomainPolicy, Inclusion, Qualifier};
pub use narrative::{Narrative, NarrativeValue, TheoremId};
pub use path::{ExprPath, PathSegment};
pub use render::render_response;
pub use request::{Budget, Precision, Request, SolveMode, UnitSystem};
pub use response::{
    EngineId, ExecutionMeta, NarratedStep, NumericMethod, Response, ResultEntry, ResultKey,
    ResultShape, ResultValue,
};

use crate::ThalesError;

/// Single entry point for thales.
///
/// Compiles the `Expression` inputs carried by [`Request::command`] into the
/// internal `Arc<Expr>` representation, dispatches to the engine matching the
/// command variant, and assembles the [`Response`] by decompiling results back
/// to `Expression` at this exit point.
///
pub fn execute(request: Request) -> Result<Response, ThalesError> {
    dispatch::execute(request)
}
