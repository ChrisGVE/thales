//! Exact numeric types for computer algebra.
//!
//! Provides overflow-safe integer and rational types used throughout
//! the expression tree and polynomial arithmetic.
//!
//! - [`SmallInt`] — tagged union: inline `i64` or heap-allocated `BigInt`
//! - [`BigRational`] — exact rational with `SmallInt` components
//! - [`SymbolId`] — interned variable name (4-byte `Copy` handle)
//! - [`Expr`] — Arc-based expression with structural sharing
//! - [`AddNode`] — canonical n-ary sum
//! - [`MulNode`] — canonical n-ary product
//! - [`ExprPool`] — hash-consing pool for common sub-expression elimination

mod big_rational;
pub mod expr;
pub mod ring;
mod small_int;
mod symbol;

mod add_node;
mod dense_poly;
mod mul_node;
mod poly_ops;
mod sparse_poly;

pub use add_node::AddNode;
pub use big_rational::BigRational;
pub use dense_poly::DensePolynomial;
pub use expr::{Expr, ExprPool};
pub use mul_node::MulNode;
pub use small_int::SmallInt;
pub use sparse_poly::SparsePolynomial;
pub use symbol::SymbolId;
