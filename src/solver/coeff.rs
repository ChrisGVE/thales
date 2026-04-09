//! Coefficient extraction from linear expressions.

use std::collections::HashMap;

use crate::ast::{BinaryOp, Expression, UnaryOp, Variable};

use super::types::{SolverError, SolverResult};

/// Extract all terms connected by `+` or `-` from `expr`.
pub(super) fn collect_additive_terms(expr: &Expression) -> Vec<Expression> {
    match expr {
        Expression::Binary(BinaryOp::Add, left, right) => {
            let mut terms = collect_additive_terms(left);
            terms.extend(collect_additive_terms(right));
            terms
        }
        Expression::Binary(BinaryOp::Sub, left, right) => {
            let mut terms = collect_additive_terms(left);
            for term in collect_additive_terms(right) {
                terms.push(Expression::Unary(UnaryOp::Neg, Box::new(term)));
            }
            terms
        }
        _ => vec![expr.clone()],
    }
}

/// Extract the numeric coefficient of `var` from a single multiplicative `term`.
pub(super) fn extract_coefficient(term: &Expression, var: &Variable) -> SolverResult<f64> {
    match term {
        Expression::Variable(v) if v.name == var.name => Ok(1.0),

        Expression::Unary(UnaryOp::Neg, inner) => Ok(-extract_coefficient(inner, var)?),

        Expression::Binary(BinaryOp::Mul, left, right) => {
            let left_has_var = left.contains_variable(&var.name);
            let right_has_var = right.contains_variable(&var.name);

            if left_has_var && right_has_var {
                return Err(SolverError::Other(format!(
                    "Non-linear term: {} * {} both contain {}",
                    left, right, var.name
                )));
            }

            let empty: HashMap<String, f64> = HashMap::new();
            if left_has_var {
                let coeff = right.evaluate(&empty).ok_or_else(|| {
                    SolverError::Other(format!("Cannot evaluate coefficient: {}", right))
                })?;
                Ok(coeff * extract_coefficient(left, var)?)
            } else {
                let coeff = left.evaluate(&empty).ok_or_else(|| {
                    SolverError::Other(format!("Cannot evaluate coefficient: {}", left))
                })?;
                Ok(coeff * extract_coefficient(right, var)?)
            }
        }

        Expression::Binary(BinaryOp::Div, left, right) => {
            if right.contains_variable(&var.name) {
                return Err(SolverError::Other(format!(
                    "Non-linear: variable {} in denominator",
                    var.name
                )));
            }
            let empty: HashMap<String, f64> = HashMap::new();
            let divisor = right
                .evaluate(&empty)
                .ok_or_else(|| SolverError::Other(format!("Cannot evaluate divisor: {}", right)))?;
            if divisor.abs() < 1e-15 {
                return Err(SolverError::DivisionByZero);
            }
            Ok(extract_coefficient(left, var)? / divisor)
        }

        _ => {
            if term.contains_variable(&var.name) {
                Err(SolverError::Other(format!(
                    "Cannot extract coefficient from: {}",
                    term
                )))
            } else {
                Ok(0.0)
            }
        }
    }
}

/// Extract (variable_coefficients, constant_term) from a linear expression.
pub(super) fn extract_linear_coefficients(
    expr: &Expression,
    variables: &[Variable],
) -> SolverResult<(Vec<f64>, f64)> {
    let mut coeffs = vec![0.0_f64; variables.len()];
    let mut constant = 0.0_f64;

    for term in collect_additive_terms(expr) {
        let mut found_var = false;
        for (i, var) in variables.iter().enumerate() {
            if term.contains_variable(&var.name) {
                coeffs[i] += extract_coefficient(&term, var)?;
                found_var = true;
                break;
            }
        }

        if !found_var {
            let empty: HashMap<String, f64> = HashMap::new();
            match term.evaluate(&empty) {
                Some(val) => constant += val,
                None => {
                    return Err(SolverError::Other(format!(
                        "Cannot evaluate constant term: {}",
                        term
                    )));
                }
            }
        }
    }

    Ok((coeffs, constant))
}
