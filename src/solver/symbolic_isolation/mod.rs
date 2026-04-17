//! Symbolic isolation engine for equation rearrangement.
//!
//! The public surface preserves the legacy `Expression`-based signature so
//! callers (equation_system, solver facade) need no changes. Internals
//! compile to `Arc<Expr>`, run an Expr-native unwrap engine that takes
//! advantage of canonical `AddNode`/`MulNode` forms, and decompile the
//! result back to `Expression` at the boundary.

mod calculus;
mod linear;
mod rational;
mod unwrap;

use crate::ast::{Expression, Variable};
use crate::numeric::compile::{compile, decompile};
use crate::numeric::{normalize, Expr, SymbolId};
use crate::resolution_path::ResolutionPathBuilder;

use super::helpers::contains_symbol;
use super::types::SolverError;

use unwrap::unwrap_variable;

/// Attempt to symbolically isolate the target variable in the equation.
///
/// Returns the expression that the variable equals, plus the updated path
/// builder. Works by recursively peeling off operations wrapping the
/// variable and applying their inverses to the other side.
pub fn symbolic_isolate(
    lhs: &Expression,
    rhs: &Expression,
    variable: &Variable,
    path: ResolutionPathBuilder,
) -> Result<(Expression, ResolutionPathBuilder), SolverError> {
    let var = SymbolId::intern(&variable.name);
    let lhs_arc = compile(lhs);
    let rhs_arc = compile(rhs);

    let left_has = contains_symbol(&lhs_arc, var);
    let right_has = contains_symbol(&rhs_arc, var);

    if !left_has && !right_has {
        return Err(SolverError::CannotSolve(format!(
            "Variable '{}' not found in equation",
            variable.name
        )));
    }

    let (var_side, other_side) = if left_has && !right_has {
        (lhs_arc, rhs_arc)
    } else if right_has && !left_has {
        (rhs_arc, lhs_arc)
    } else {
        (normalize::sub(lhs_arc, rhs_arc), Expr::int(0))
    };

    let (result_expr, final_path) = unwrap_variable(&var_side, &other_side, var, path)?;
    Ok((decompile(&result_expr), final_path))
}

#[cfg(test)]
mod tests;
