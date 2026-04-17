//! Transcendental-function detection + domain validation.
//!
//! Operates on `Arc<Expr>` canonical form. Used by
//! [`super::TranscendentalSolver::can_solve`] to decide whether this
//! solver should be dispatched, and by the trig family to validate the
//! domain of `Asin`/`Acos`.

use std::sync::Arc;

use crate::numeric::{Expr, FuncId};

use super::super::helpers::has_any_symbol;
use super::super::types::SolverError;

/// `true` when the expression contains at least one transcendental
/// function node (trig, inverse trig, hyperbolic, exp, log), or a
/// `Pow(base, exp)` whose exponent contains any symbol (which makes the
/// whole expression exponential in that symbol).
pub(super) fn has_transcendental_function(expr: &Expr) -> bool {
    match expr {
        Expr::Func(id, args) => {
            matches!(
                id,
                FuncId::Sin
                    | FuncId::Cos
                    | FuncId::Tan
                    | FuncId::Asin
                    | FuncId::Acos
                    | FuncId::Atan
                    | FuncId::Sinh
                    | FuncId::Cosh
                    | FuncId::Tanh
                    | FuncId::Exp
                    | FuncId::Ln
                    | FuncId::Log
                    | FuncId::Log2
                    | FuncId::Log10
            ) || args.iter().any(|a| has_transcendental_function(a))
        }
        Expr::Pow(base, exp) => {
            // Variable in exponent ⇒ exponential form.
            has_any_symbol(exp)
                || has_transcendental_function(base)
                || has_transcendental_function(exp)
        }
        Expr::Add(node) => node.terms.keys().any(|t| has_transcendental_function(t)),
        Expr::Mul(node) => node
            .factors
            .iter()
            .any(|(b, e)| has_transcendental_function(b) || has_transcendental_function(e)),
        _ => false,
    }
}

/// Domain check for inverse trig input values.
pub(super) fn validate_trig_domain(value: f64, func: FuncId) -> Result<(), SolverError> {
    if matches!(func, FuncId::Asin | FuncId::Acos) && value.abs() > 1.0 {
        return Err(SolverError::Other(format!(
            "Domain error: {:?} requires |value| ≤ 1, got {}",
            func, value
        )));
    }
    Ok(())
}

/// Try to evaluate `expr` under empty bindings to an `f64`. Returns
/// `None` if the expression is not a pure constant.
pub(super) fn eval_constant(expr: &Arc<Expr>) -> Option<f64> {
    crate::numeric::evaluation::evaluate(expr, &std::collections::HashMap::new())
}
