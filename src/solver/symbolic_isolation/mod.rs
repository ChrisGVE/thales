//! Symbolic isolation engine for equation rearrangement.
//!
//! Takes pre-compiled `Arc<Expr>` equation sides, runs an Expr-native
//! unwrap engine that takes advantage of canonical `AddNode`/`MulNode`
//! forms, and decompiles the result back to `Expression` only at the
//! resolution-path boundary.

mod calculus;
mod linear;
mod rational;
mod unwrap;

use std::sync::Arc;

use crate::ast::{Expression, Variable};
use crate::numeric::compile::decompile;
use crate::numeric::{normalize, Expr, SymbolId};
use crate::resolution_path::ResolutionPathBuilder;

use super::helpers::contains_symbol;
use super::types::SolverError;

use unwrap::unwrap_variable;

/// Attempt to symbolically isolate the target variable in the equation.
///
/// Callers pass already-compiled `Arc<Expr>` sides. Returns the
/// expression that the variable equals (decompiled to `Expression` for
/// resolution-path consumption), plus the updated path builder.
pub fn symbolic_isolate(
    lhs: &Arc<Expr>,
    rhs: &Arc<Expr>,
    variable: &Variable,
    path: ResolutionPathBuilder,
) -> Result<(Expression, ResolutionPathBuilder), SolverError> {
    let var = SymbolId::intern(&variable.name);

    let left_has = contains_symbol(lhs, var);
    let right_has = contains_symbol(rhs, var);

    if !left_has && !right_has {
        return Err(SolverError::CannotSolve(format!(
            "Variable '{}' not found in equation",
            variable.name
        )));
    }

    let (var_side, other_side) = if left_has && !right_has {
        (Arc::clone(lhs), Arc::clone(rhs))
    } else if right_has && !left_has {
        (Arc::clone(rhs), Arc::clone(lhs))
    } else {
        (
            normalize::sub(Arc::clone(lhs), Arc::clone(rhs)),
            Expr::int(0),
        )
    };

    let (result_expr, final_path) = unwrap_variable(&var_side, &other_side, var, path)?;
    Ok((decompile(&result_expr), final_path))
}

#[cfg(test)]
mod tests;
