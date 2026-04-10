//! General symbolic isolation engine for equation rearrangement.
//!
//! Implements a recursive inverse-unwrapping algorithm that can isolate any
//! variable appearing linearly (exactly once) in an equation, handling
//! arithmetic, powers, and invertible functions.

mod calculus;
mod linear;
mod rational;
mod unwrap;

use crate::ast::{BinaryOp, Expression, Variable};
use crate::resolution_path::ResolutionPathBuilder;

use super::helpers::contains_variable;
use super::types::SolverError;
use unwrap::unwrap_variable;

/// Attempt to symbolically isolate the target variable in the equation.
///
/// Returns the expression that the variable equals, plus the updated path
/// builder. Works by recursively "peeling off" operations wrapping the
/// variable and applying their inverses to the other side.
pub fn symbolic_isolate(
    lhs: &Expression,
    rhs: &Expression,
    variable: &Variable,
    path: ResolutionPathBuilder,
) -> Result<(Expression, ResolutionPathBuilder), SolverError> {
    let var = &variable.name;
    let left_has = contains_variable(lhs, var);
    let right_has = contains_variable(rhs, var);

    if !left_has && !right_has {
        return Err(SolverError::CannotSolve(format!(
            "Variable '{}' not found in equation",
            var
        )));
    }

    // Determine which side contains the variable
    let (var_side, other_side) = if left_has && !right_has {
        (lhs.clone(), rhs.clone())
    } else if right_has && !left_has {
        (rhs.clone(), lhs.clone())
    } else {
        // Variable on both sides: move everything to the left
        let combined =
            Expression::Binary(BinaryOp::Sub, Box::new(lhs.clone()), Box::new(rhs.clone()))
                .simplify();
        (combined, Expression::Integer(0))
    };

    unwrap_variable(&var_side, &other_side, var, path)
}

#[cfg(test)]
mod tests;
