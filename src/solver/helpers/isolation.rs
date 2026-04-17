//! Variable-isolation engine for the linear solver.
//!
//! Rearranges an equation so the target variable stands alone. Handles:
//! - already-isolated forms (`x = expr`, `expr = x`)
//! - pure coefficient forms (`a*x = c`)
//! - addition forms (`x + b = c`)
//! - combined forms (`a*x + b = c`)
//!
//! More complex shapes fall back to the symbolic-isolation engine.

use crate::ast::{BinaryOp, Equation, Expression};
use crate::resolution_path::ResolutionPathBuilder;

use super::super::SolverError;
use super::coefficients::{evaluate_constants, extract_coefficient};
use super::detection::contains_variable;

/// Isolate a variable in an equation.
pub(crate) fn isolate_variable(
    equation: &Equation,
    var: &str,
    _path: &mut ResolutionPathBuilder,
) -> Result<Expression, SolverError> {
    let left = &equation.left;
    let right = &equation.right;

    if !contains_variable(left, var) && !contains_variable(right, var) {
        return Err(SolverError::CannotSolve(format!(
            "Variable '{}' not found in equation",
            var
        )));
    }

    // Special case: variable already isolated.
    if let Expression::Variable(v) = left {
        if v.name == var && !contains_variable(right, var) {
            return Ok(right.clone());
        }
    }
    if let Expression::Variable(v) = right {
        if v.name == var && !contains_variable(left, var) {
            return Ok(left.clone());
        }
    }

    // Pattern: a * x = c  =>  x = c / a
    if let Some(coeff) = extract_coefficient(left, var) {
        if !contains_variable(right, var) {
            let result = Expression::Binary(
                BinaryOp::Div,
                Box::new(right.clone()),
                Box::new(coeff.clone()),
            )
            .simplify();
            return Ok(evaluate_constants(&result));
        }
    }

    // Pattern: c = a * x  =>  x = c / a
    if let Some(coeff) = extract_coefficient(right, var) {
        if !contains_variable(left, var) {
            let result = Expression::Binary(
                BinaryOp::Div,
                Box::new(left.clone()),
                Box::new(coeff.clone()),
            )
            .simplify();
            return Ok(evaluate_constants(&result));
        }
    }

    // Pattern: x + b = c  =>  x = c - b
    if let Expression::Binary(BinaryOp::Add, l, r) = left {
        if let Expression::Variable(v) = l.as_ref() {
            if v.name == var && !contains_variable(r, var) && !contains_variable(right, var) {
                let result = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(right.clone()),
                    Box::new(r.as_ref().clone()),
                )
                .simplify();
                return Ok(evaluate_constants(&result));
            }
        }
        if let Expression::Variable(v) = r.as_ref() {
            if v.name == var && !contains_variable(l, var) && !contains_variable(right, var) {
                let result = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(right.clone()),
                    Box::new(l.as_ref().clone()),
                )
                .simplify();
                return Ok(evaluate_constants(&result));
            }
        }
    }

    // Pattern: c = x + b  =>  x = c - b
    if let Expression::Binary(BinaryOp::Add, l, r) = right {
        if let Expression::Variable(v) = l.as_ref() {
            if v.name == var && !contains_variable(r, var) && !contains_variable(left, var) {
                let result = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(left.clone()),
                    Box::new(r.as_ref().clone()),
                )
                .simplify();
                return Ok(evaluate_constants(&result));
            }
        }
        if let Expression::Variable(v) = r.as_ref() {
            if v.name == var && !contains_variable(l, var) && !contains_variable(left, var) {
                let result = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(left.clone()),
                    Box::new(l.as_ref().clone()),
                )
                .simplify();
                return Ok(evaluate_constants(&result));
            }
        }
    }

    // Pattern: a * x + b = c  =>  x = (c - b) / a
    if let Expression::Binary(BinaryOp::Add, l, r) = left {
        if let Some(coeff) = extract_coefficient(l, var) {
            if !contains_variable(r, var) && !contains_variable(right, var) {
                let numerator = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(right.clone()),
                    Box::new(r.as_ref().clone()),
                );
                let result =
                    Expression::Binary(BinaryOp::Div, Box::new(numerator), Box::new(coeff))
                        .simplify();
                return Ok(evaluate_constants(&result));
            }
        }
        if let Some(coeff) = extract_coefficient(r, var) {
            if !contains_variable(l, var) && !contains_variable(right, var) {
                let numerator = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(right.clone()),
                    Box::new(l.as_ref().clone()),
                );
                let result =
                    Expression::Binary(BinaryOp::Div, Box::new(numerator), Box::new(coeff))
                        .simplify();
                return Ok(evaluate_constants(&result));
            }
        }
    }

    Err(SolverError::CannotSolve(
        "Equation pattern not yet supported for Phase 1".to_string(),
    ))
}
