//! Linear term collection for symbolic isolation.
//!
//! Handles cases where the target variable appears on both sides of a binary
//! operation, collecting linear coefficients to combine terms.

use crate::ast::{BinaryOp, Expression, UnaryOp};
use crate::resolution_path::{Operation, ResolutionPathBuilder, StepAnnotation};

use super::super::helpers::contains_variable;
use super::super::types::SolverError;

/// Count how many times a variable appears in an expression.
pub(super) fn count_variable(expr: &Expression, var: &str) -> usize {
    match expr {
        Expression::Variable(v) if v.name == var => 1,
        Expression::Variable(_)
        | Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Rational(_)
        | Expression::Complex(_)
        | Expression::Constant(_) => 0,
        Expression::Unary(_, inner) => count_variable(inner, var),
        Expression::Binary(_, left, right) => {
            count_variable(left, var) + count_variable(right, var)
        }
        Expression::Power(base, exp) => count_variable(base, var) + count_variable(exp, var),
        Expression::Function(_, args) => args.iter().map(|a| count_variable(a, var)).sum(),
    }
}

/// Attempt to collect linear terms when the variable appears on both sides
/// of a binary operation.
///
/// Handles patterns like `a*v + b*v = (a+b)*v` and `a*v - b*v = (a-b)*v`.
pub(super) fn collect_linear_terms(
    op: BinaryOp,
    left: &Expression,
    right: &Expression,
    other: &Expression,
    var: &str,
    path: ResolutionPathBuilder,
) -> Result<(Expression, ResolutionPathBuilder), SolverError> {
    // Only addition and subtraction can have linear collection
    if op != BinaryOp::Add && op != BinaryOp::Sub {
        return Err(SolverError::CannotSolve(format!(
            "Cannot isolate '{}': variable appears in both operands of {:?}",
            var, op
        )));
    }

    // Try to extract coefficients: left = coeff_l * var, right = coeff_r * var
    let coeff_l = try_extract_linear_coeff(left, var);
    let coeff_r = try_extract_linear_coeff(right, var);

    match (coeff_l, coeff_r) {
        (Some(cl), Some(cr)) => {
            // Combine: (cl +/- cr) * var = other
            let combined_coeff = match op {
                BinaryOp::Add => {
                    Expression::Binary(BinaryOp::Add, Box::new(cl), Box::new(cr)).simplify()
                }
                BinaryOp::Sub => {
                    Expression::Binary(BinaryOp::Sub, Box::new(cl), Box::new(cr)).simplify()
                }
                _ => unreachable!(),
            };
            let new_other = Expression::Binary(
                BinaryOp::Div,
                Box::new(other.clone()),
                Box::new(combined_coeff.clone()),
            )
            .simplify();
            let p = path.annotated_step(
                Operation::DivideBothSides(combined_coeff),
                format!("Collect terms and divide to isolate {}", var),
                new_other.clone(),
                StepAnnotation::elementary(),
            );
            Ok((new_other, p))
        }
        _ => {
            // Linear extraction failed — try clearing denominators
            if let Some(result) =
                super::rational::try_clear_denominators(op, left, right, other, var, &path)
            {
                return result;
            }
            Err(SolverError::CannotSolve(format!(
                "Cannot isolate '{}': variable appears non-linearly in both operands",
                var
            )))
        }
    }
}

/// Try to extract the coefficient of a variable from an expression that is
/// a linear multiple of the variable.
///
/// Returns `Some(coeff)` if `expr = coeff * var`, `None` otherwise.
fn try_extract_linear_coeff(expr: &Expression, var: &str) -> Option<Expression> {
    if count_variable(expr, var) != 1 {
        return None;
    }

    match expr {
        Expression::Variable(v) if v.name == var => Some(Expression::Integer(1)),

        Expression::Unary(UnaryOp::Neg, inner) => try_extract_linear_coeff(inner, var)
            .map(|c| Expression::Unary(UnaryOp::Neg, Box::new(c)).simplify()),

        Expression::Binary(BinaryOp::Mul, left, right) => {
            let left_has = contains_variable(left, var);
            let right_has = contains_variable(right, var);
            if left_has && !right_has {
                try_extract_linear_coeff(left, var).map(|c| {
                    Expression::Binary(BinaryOp::Mul, Box::new(c), Box::new(right.as_ref().clone()))
                        .simplify()
                })
            } else if right_has && !left_has {
                try_extract_linear_coeff(right, var).map(|c| {
                    Expression::Binary(BinaryOp::Mul, Box::new(left.as_ref().clone()), Box::new(c))
                        .simplify()
                })
            } else {
                None
            }
        }

        Expression::Binary(BinaryOp::Div, left, right) => {
            if contains_variable(right, var) {
                None // var in denominator is not linear
            } else {
                try_extract_linear_coeff(left, var).map(|c| {
                    Expression::Binary(BinaryOp::Div, Box::new(c), Box::new(right.as_ref().clone()))
                        .simplify()
                })
            }
        }

        _ => None,
    }
}
