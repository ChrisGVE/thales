//! Exact numeric types for computer algebra.
//!
//! Provides overflow-safe integer and rational types used throughout
//! the expression tree and polynomial arithmetic.
//!
//! - [`SmallInt`] — tagged union: inline `i64` or heap-allocated `BigInt`
//! - [`BigRational`] — exact rational with `SmallInt` components

mod big_rational;
mod small_int;
mod symbol;

pub use big_rational::BigRational;
pub use small_int::SmallInt;
pub use symbol::SymbolId;
