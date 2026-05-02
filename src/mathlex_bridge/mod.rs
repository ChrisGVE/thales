//! Conversion layer between mathlex AST types and thales AST types.
//!
//! mathlex provides the parsing infrastructure; thales retains its own Expression
//! type for computation (evaluation, differentiation, simplification, solving).
//! This module bridges the two representations.

mod convert;
mod helpers;
mod ode;

pub use convert::{convert_equation, convert_expression};
pub use ode::{try_extract_ode, ExtractedODE};

#[cfg(test)]
mod tests;
