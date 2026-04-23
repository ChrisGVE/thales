//! Shared utility functions for equation solvers.
//!
//! Split into focused submodules:
//! - [`detection`] — Arc<Expr> predicates (`contains_symbol`, `is_polynomial_expr`, …)
//! - [`coefficients`] — Arc<Expr> coefficient extraction and boundary helpers
//! - [`substitution`] — Expression value substitution (used by `solve_for` boundary)
//!
//! All internal solver logic uses the Arc<Expr>-based functions. The two
//! Expression-based helpers (`evaluate_constants`, `simplify_numeric_expression`)
//! remain solely to build `Expression` values for public `Solution` output.

pub(crate) mod coefficients;
pub(crate) mod detection;
pub(crate) mod substitution;

pub(crate) use coefficients::{
    evaluate_constants, extract_quadratic_coefficients_expr, get_polynomial_degree_expr,
    simplify_numeric_expression,
};
pub(crate) use detection::{
    contains_symbol, has_any_symbol, has_any_variable, has_obvious_nonlinearity_expr,
    is_linear_in_variable_expr, is_linear_system_expr, is_polynomial_expr,
};
pub(crate) use substitution::substitute_values;
