//! Rational equation solving via denominator clearing.
//!
//! When the target variable appears in a denominator, this module attempts to
//! clear denominators by multiplying through, converting the equation into a
//! polynomial form that can be solved by the standard isolation engine.

use crate::ast::{BinaryOp, Expression};
use crate::resolution_path::{Operation, ResolutionPathBuilder, StepAnnotation};

use super::super::helpers::contains_variable;
use super::super::types::SolverError;
use super::unwrap::unwrap_variable;

/// Check whether a variable appears in any denominator within an expression.
fn has_var_in_denominator(expr: &Expression, var: &str) -> bool {
    match expr {
        Expression::Binary(BinaryOp::Div, _, right) => {
            if contains_variable(right, var) {
                return true;
            }
            // Also check recursively within both children
            has_var_in_denominator_recursive(expr, var)
        }
        _ => has_var_in_denominator_recursive(expr, var),
    }
}

/// Recursively check all sub-expressions for variable-in-denominator.
fn has_var_in_denominator_recursive(expr: &Expression, var: &str) -> bool {
    match expr {
        Expression::Variable(_)
        | Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Rational(_)
        | Expression::Complex(_)
        | Expression::Constant(_) => false,
        Expression::Unary(_, inner) => has_var_in_denominator(inner, var),
        Expression::Binary(BinaryOp::Div, left, right) => {
            contains_variable(right, var)
                || has_var_in_denominator(left, var)
                || has_var_in_denominator(right, var)
        }
        Expression::Binary(_, left, right) => {
            has_var_in_denominator(left, var) || has_var_in_denominator(right, var)
        }
        Expression::Power(base, exp) => {
            has_var_in_denominator(base, var) || has_var_in_denominator(exp, var)
        }
        Expression::Function(_, args) => args.iter().any(|a| has_var_in_denominator(a, var)),
    }
}

/// Extract the denominator expression containing the variable from a
/// division node. Returns the first denominator found that contains the
/// variable.
fn extract_var_denominator(expr: &Expression, var: &str) -> Option<Expression> {
    match expr {
        Expression::Binary(BinaryOp::Div, left, right) => {
            if contains_variable(right, var) {
                Some(right.as_ref().clone())
            } else {
                // Search deeper in numerator
                extract_var_denominator(left, var)
            }
        }
        Expression::Binary(_, left, right) => {
            extract_var_denominator(left, var).or_else(|| extract_var_denominator(right, var))
        }
        Expression::Unary(_, inner) => extract_var_denominator(inner, var),
        Expression::Power(base, _) => extract_var_denominator(base, var),
        Expression::Function(_, args) => args.iter().find_map(|a| extract_var_denominator(a, var)),
        _ => None,
    }
}

/// Try to clear denominators when the variable appears in a denominator.
///
/// For `a*var - b/(c*var) = other`, multiply everything by `c*var`:
/// `a*var*(c*var) - b = other*(c*var)` → polynomial in var.
///
/// Returns `None` if no denominator clearing is applicable. Returns
/// `Some(result)` with the isolation result if clearing succeeds.
pub(super) fn try_clear_denominators(
    op: BinaryOp,
    left: &Expression,
    right: &Expression,
    other: &Expression,
    var: &str,
    path: &ResolutionPathBuilder,
) -> Option<Result<(Expression, ResolutionPathBuilder), SolverError>> {
    // Check if either child has the variable in a denominator
    let left_denom = has_var_in_denominator(left, var);
    let right_denom = has_var_in_denominator(right, var);

    if !left_denom && !right_denom {
        return None;
    }

    // Extract the denominator to multiply through
    let denom = if left_denom {
        extract_var_denominator(left, var)
    } else {
        extract_var_denominator(right, var)
    };

    let denom = match denom {
        Some(d) => d,
        None => return None,
    };

    // Build the cleared equation: (left op right) * denom = other * denom
    // We multiply each part individually to help simplification
    let new_left = Expression::Binary(
        BinaryOp::Mul,
        Box::new(left.clone()),
        Box::new(denom.clone()),
    )
    .simplify();

    let new_right = Expression::Binary(
        BinaryOp::Mul,
        Box::new(right.clone()),
        Box::new(denom.clone()),
    )
    .simplify();

    let new_other = Expression::Binary(
        BinaryOp::Mul,
        Box::new(other.clone()),
        Box::new(denom.clone()),
    )
    .simplify();

    // Guard against infinite recursion: if the cleared expression still has
    // the variable in a denominator, bail out
    let var_side =
        Expression::Binary(op, Box::new(new_left.clone()), Box::new(new_right.clone())).simplify();

    if has_var_in_denominator(&var_side, var) || has_var_in_denominator(&new_other, var) {
        return None;
    }

    let p = path.clone().annotated_step(
        Operation::MultiplyBothSides(denom.clone()),
        format!("Multiply both sides by {} to clear denominator", denom),
        new_other.clone(),
        StepAnnotation::elementary(),
    );

    Some(unwrap_variable(&var_side, &new_other, var, p))
}
