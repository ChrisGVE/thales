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
mod compute_ctx;
mod dense_poly;
mod hermite;
mod mul_node;
mod multivariate_poly;
pub mod normalize;
mod number_theory;
mod poly_factoring;
mod poly_ops;
mod rational_fn;
mod solution_set;
mod sparse_poly;
mod term_order;

pub use add_node::AddNode;
pub use big_rational::BigRational;
pub use compute_ctx::{CancelHandle, ComputeContext, ComputeError, ComputeResult, FeatureFlags};
pub use dense_poly::DensePolynomial;
pub use expr::{Expr, ExprPool};
pub use hermite::{hermite_reduce, HermiteReduction};
pub use mul_node::MulNode;
pub use multivariate_poly::{Monomial, MultivariatePolynomial};
pub use number_theory::{crt, ext_gcd, mod_inverse, mod_pow, ExtGcdResult};
pub use poly_factoring::SqfFactor;
pub use rational_fn::RationalFunction;
pub use small_int::SmallInt;
pub use solution_set::{Constraint, IntervalBound, SolutionSet};
pub use sparse_poly::SparsePolynomial;
pub use symbol::SymbolId;
pub use term_order::{DegLex, GrevLex, Lex, MonomialOrder, OrderedMonomial};
