//! Exact numeric types for computer algebra.
//!
//! Provides overflow-safe integer and rational types used throughout
//! the expression tree and polynomial arithmetic.
//!
//! - [`SmallInt`] — tagged union: inline `i64` or heap-allocated `BigInt`
//! - [`BigRational`] — exact rational with `SmallInt` components

mod add_node;
mod big_rational;
mod mul_node;
pub mod ring;
mod small_int;
mod symbol;

pub use add_node::AddNode;
pub use big_rational::BigRational;
pub use mul_node::MulNode;
pub use small_int::SmallInt;
pub use symbol::SymbolId;
