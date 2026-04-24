//! Differential geometry — k-forms, wedge product, exterior derivative.
//!
//! This module provides the foundational machinery for differential forms on
//! a smooth manifold represented symbolically. All coefficients are `Arc<Expr>`.
//!
//! ## Scope
//!
//! - `DifferentialForm`: a degree-k differential form as a sum of `FormTerm`s.
//! - `wedge`: antisymmetric tensor product of two forms.
//! - `exterior_derivative`: maps a k-form to a (k+1)-form.
//! - Convenience constructors `zero` and `one_form`.
//!
//! Manifold/chart/atlas machinery is deferred to the `thales-GR` sister crate.

pub mod forms;

pub use forms::{exterior_derivative, one_form, wedge, zero, DifferentialForm, FormTerm};
