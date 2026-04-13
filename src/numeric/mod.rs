//! Exact numeric types for computer algebra.
//!
//! Provides overflow-safe integer and rational types used throughout
//! the expression tree and polynomial arithmetic.
//!
//! - [`SmallInt`] — tagged union: inline `i64` or heap-allocated `BigInt`
//! - [`BigRational`] — exact rational with `SmallInt` components (future)

mod small_int;

pub use small_int::SmallInt;
