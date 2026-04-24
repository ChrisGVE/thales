//! Calculus operations on [`Arc<Expr>`][std::sync::Arc] trees.
//!
//! This module provides symbolic calculus primitives that operate natively on
//! the `Arc<Expr>` internal representation.  All computation delegates to the
//! single-variable differentiation engine in
//! [`crate::numeric::differentiation`] and composes results without
//! duplicating any symbolic logic.
//!
//! ## Sub-modules
//!
//! - [`multivar`] — partial derivatives, gradient, Jacobian, Hessian,
//!   directional derivative, and total derivative for multivariate scalar and
//!   vector-valued functions.

pub mod multivar;
