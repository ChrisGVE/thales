//! Compiler from the legacy `ast::Expression` tree to the CAS `numeric::Expr`.
//!
//! [`compile`] performs a structural translation, mapping each `Expression`
//! variant to its `Expr` equivalent and using the smart constructors in
//! [`normalize`] so the output is already in canonical form.
//!
//! [`decompile`] performs the inverse translation, mapping each `Expr` variant
//! back to its `Expression` equivalent. Because `compile` normalizes expressions,
//! the round-trip is semantically equivalent but may differ structurally.

pub mod compile_expr;
pub mod decompile;

pub use compile_expr::{compile, map_func_id};
pub use decompile::{decompile, reverse_map_func_id};

#[cfg(test)]
mod tests;
