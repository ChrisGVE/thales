//! Shared utility functions for equation solvers.
//!
//! Split into focused submodules:
//! - [`detection`] — predicates like `contains_variable`, `is_linear_in_variable`
//! - [`coefficients`] — coefficient extraction and numeric evaluation helpers
//! - [`isolation`] — the variable-isolation engine used by the linear solver
//! - [`substitution`] — value substitution into expressions
//!
//! The public surface re-exports the original flat API so existing callers in
//! other solver modules need no changes while the Expr migration is in
//! progress. Expr-based variants (`contains_symbol`, …) live alongside the
//! Expression-based ones and will be introduced subtask by subtask.

pub(crate) mod coefficients;
pub(crate) mod detection;
pub(crate) mod isolation;
pub(crate) mod substitution;

pub(crate) use coefficients::{
    evaluate_constants, extract_coefficient, extract_polynomial_coefficients,
    extract_quadratic_coefficients, get_polynomial_degree, simplify_numeric_expression,
};
pub(crate) use detection::{
    contains_symbol, contains_variable, has_any_variable, has_obvious_nonlinearity,
    is_linear_in_variable, is_polynomial_expression,
};
pub(crate) use isolation::isolate_variable;
pub(crate) use substitution::substitute_values;
