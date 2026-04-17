//! Coefficient extraction, polynomial-degree inspection, and small numeric
//! utilities on `Expression`.
//!
//! These helpers operate on the legacy `Expression` form because their
//! callers (linear, quadratic, polynomial, transcendental) are still
//! Expression-based. They will be superseded by Expr-native equivalents as
//! each solver migrates.

use std::collections::HashMap;

use crate::ast::{BinaryOp, Expression, UnaryOp};

use super::detection::{contains_variable, has_any_variable};

/// Extract the coefficient of a variable from a single term.
///
/// Recognises `x`, `a * x`, and `x * a` where `a` is any sub-expression
/// not containing the variable. Returns `None` when the variable is absent
/// or appears in a non-linear position.
pub(crate) fn extract_coefficient(expr: &Expression, var: &str) -> Option<Expression> {
    match expr {
        Expression::Variable(v) if v.name == var => Some(Expression::Integer(1)),

        Expression::Binary(BinaryOp::Mul, left, right) => {
            if let Expression::Variable(v) = left.as_ref() {
                if v.name == var && !contains_variable(right, var) {
                    return Some(right.as_ref().clone());
                }
            }
            if let Expression::Variable(v) = right.as_ref() {
                if v.name == var && !contains_variable(left, var) {
                    return Some(left.as_ref().clone());
                }
            }
            None
        }

        _ => None,
    }
}

/// Evaluate constant expressions to their numeric values.
///
/// Simplifies first; if nothing depends on a variable, evaluates to a
/// literal. Near-integer results collapse to `Expression::Integer`.
pub(crate) fn evaluate_constants(expr: &Expression) -> Expression {
    let simplified = expr.simplify();

    if !has_any_variable(&simplified) {
        if let Some(value) = simplified.evaluate(&HashMap::new()) {
            if value.fract().abs() < 1e-10 {
                return Expression::Integer(value.round() as i64);
            } else {
                return Expression::Float(value);
            }
        }
    }

    simplified
}

/// Simplify a numeric value to the best Expression representation.
pub(crate) fn simplify_numeric_expression(val: f64) -> Expression {
    let rounded = val.round();
    if (val - rounded).abs() < 1e-10 && rounded.abs() < i64::MAX as f64 {
        Expression::Integer(rounded as i64)
    } else {
        Expression::Float(val)
    }
}

/// Extract coefficients `(a, b, c)` from a quadratic expression
/// `a*x² + b*x + c`.
pub(crate) fn extract_quadratic_coefficients(expr: &Expression, var: &str) -> (f64, f64, f64) {
    let mut a = 0.0;
    let mut b = 0.0;
    let mut c = 0.0;

    extract_poly_coefficients_recursive(expr, var, 1.0, &mut a, &mut b, &mut c);
    (a, b, c)
}

fn extract_poly_coefficients_recursive(
    expr: &Expression,
    var: &str,
    multiplier: f64,
    a: &mut f64,
    b: &mut f64,
    c: &mut f64,
) {
    match expr {
        Expression::Integer(n) => *c += (*n as f64) * multiplier,
        Expression::Float(f) => *c += f * multiplier,
        Expression::Rational(r) => *c += (*r.numer() as f64 / *r.denom() as f64) * multiplier,
        Expression::Variable(v) if v.name == var => *b += multiplier,
        Expression::Variable(_) | Expression::Constant(_) => {
            if let Some(val) = expr.evaluate(&HashMap::new()) {
                *c += val * multiplier;
            }
        }
        Expression::Unary(UnaryOp::Neg, inner) => {
            extract_poly_coefficients_recursive(inner, var, -multiplier, a, b, c);
        }
        Expression::Binary(BinaryOp::Add, left, right) => {
            extract_poly_coefficients_recursive(left, var, multiplier, a, b, c);
            extract_poly_coefficients_recursive(right, var, multiplier, a, b, c);
        }
        Expression::Binary(BinaryOp::Sub, left, right) => {
            extract_poly_coefficients_recursive(left, var, multiplier, a, b, c);
            extract_poly_coefficients_recursive(right, var, -multiplier, a, b, c);
        }
        Expression::Binary(BinaryOp::Mul, left, right) => {
            let left_val = left.evaluate(&HashMap::new());
            let right_val = right.evaluate(&HashMap::new());

            match (left_val, right_val) {
                (Some(lv), None) => {
                    extract_poly_coefficients_recursive(right, var, multiplier * lv, a, b, c);
                }
                (None, Some(rv)) => {
                    extract_poly_coefficients_recursive(left, var, multiplier * rv, a, b, c);
                }
                (Some(lv), Some(rv)) => {
                    *c += lv * rv * multiplier;
                }
                (None, None) => {
                    if matches!(&**left, Expression::Variable(v) if v.name == var)
                        && matches!(&**right, Expression::Variable(v) if v.name == var)
                    {
                        *a += multiplier;
                    } else if matches!(&**left, Expression::Variable(v) if v.name == var) {
                        if let Some(rv) = right.evaluate(&HashMap::new()) {
                            *b += multiplier * rv;
                        }
                    } else if matches!(&**right, Expression::Variable(v) if v.name == var) {
                        if let Some(lv) = left.evaluate(&HashMap::new()) {
                            *b += multiplier * lv;
                        }
                    }
                }
            }
        }
        Expression::Power(base, exp) => {
            if matches!(&**base, Expression::Variable(v) if v.name == var) {
                if let Some(exp_val) = exp.evaluate(&HashMap::new()) {
                    if (exp_val - 2.0).abs() < 1e-10 {
                        *a += multiplier;
                    } else if (exp_val - 1.0).abs() < 1e-10 {
                        *b += multiplier;
                    } else if exp_val.abs() < 1e-10 {
                        *c += multiplier;
                    }
                }
            }
        }
        _ => {
            if let Some(val) = expr.evaluate(&HashMap::new()) {
                *c += val * multiplier;
            }
        }
    }
}

/// Extract coefficients for a general polynomial.
///
/// Returns a vector `[a0, a1, …, a_max_degree]` where `a_i` is the
/// coefficient of `x^i`.
pub(crate) fn extract_polynomial_coefficients(
    expr: &Expression,
    var: &str,
    max_degree: usize,
) -> Vec<f64> {
    let mut coeffs = vec![0.0; max_degree + 1];
    extract_general_poly_coefficients(expr, var, 1.0, &mut coeffs);
    coeffs
}

fn extract_general_poly_coefficients(
    expr: &Expression,
    var: &str,
    multiplier: f64,
    coeffs: &mut [f64],
) {
    match expr {
        Expression::Integer(n) => coeffs[0] += (*n as f64) * multiplier,
        Expression::Float(f) => coeffs[0] += f * multiplier,
        Expression::Rational(r) => {
            coeffs[0] += (*r.numer() as f64 / *r.denom() as f64) * multiplier
        }
        Expression::Variable(v) if v.name == var => {
            if coeffs.len() > 1 {
                coeffs[1] += multiplier;
            }
        }
        Expression::Variable(_) | Expression::Constant(_) => {
            if let Some(val) = expr.evaluate(&HashMap::new()) {
                coeffs[0] += val * multiplier;
            }
        }
        Expression::Unary(UnaryOp::Neg, inner) => {
            extract_general_poly_coefficients(inner, var, -multiplier, coeffs);
        }
        Expression::Binary(BinaryOp::Add, left, right) => {
            extract_general_poly_coefficients(left, var, multiplier, coeffs);
            extract_general_poly_coefficients(right, var, multiplier, coeffs);
        }
        Expression::Binary(BinaryOp::Sub, left, right) => {
            extract_general_poly_coefficients(left, var, multiplier, coeffs);
            extract_general_poly_coefficients(right, var, -multiplier, coeffs);
        }
        Expression::Binary(BinaryOp::Mul, left, right) => {
            let left_val = left.evaluate(&HashMap::new());
            let right_val = right.evaluate(&HashMap::new());

            match (left_val, right_val) {
                (Some(lv), None) => {
                    extract_general_poly_coefficients(right, var, multiplier * lv, coeffs);
                }
                (None, Some(rv)) => {
                    extract_general_poly_coefficients(left, var, multiplier * rv, coeffs);
                }
                (Some(lv), Some(rv)) => {
                    coeffs[0] += lv * rv * multiplier;
                }
                (None, None) => {
                    if matches!(&**left, Expression::Variable(v) if v.name == var)
                        && matches!(&**right, Expression::Variable(v) if v.name == var)
                    {
                        if coeffs.len() > 2 {
                            coeffs[2] += multiplier;
                        }
                    }
                }
            }
        }
        Expression::Power(base, exp) => {
            if matches!(&**base, Expression::Variable(v) if v.name == var) {
                if let Some(exp_val) = exp.evaluate(&HashMap::new()) {
                    let degree = exp_val.round() as usize;
                    if degree < coeffs.len() {
                        coeffs[degree] += multiplier;
                    }
                }
            }
        }
        _ => {
            if let Some(val) = expr.evaluate(&HashMap::new()) {
                coeffs[0] += val * multiplier;
            }
        }
    }
}

/// Get the polynomial degree of `expr` with respect to `var`.
pub(crate) fn get_polynomial_degree(expr: &Expression, var: &str) -> usize {
    match expr {
        Expression::Integer(_)
        | Expression::Rational(_)
        | Expression::Float(_)
        | Expression::Complex(_)
        | Expression::Constant(_) => 0,

        Expression::Variable(v) if v.name == var => 1,
        Expression::Variable(_) => 0,

        Expression::Unary(UnaryOp::Neg, inner) => get_polynomial_degree(inner, var),

        Expression::Binary(BinaryOp::Add | BinaryOp::Sub, left, right) => {
            get_polynomial_degree(left, var).max(get_polynomial_degree(right, var))
        }

        Expression::Binary(BinaryOp::Mul, left, right) => {
            get_polynomial_degree(left, var) + get_polynomial_degree(right, var)
        }

        Expression::Binary(BinaryOp::Div, left, right) => {
            if contains_variable(right, var) {
                0
            } else {
                get_polynomial_degree(left, var)
            }
        }

        Expression::Power(base, exp) => {
            if let Expression::Variable(v) = base.as_ref() {
                if v.name == var {
                    if let Some(exp_val) = exp.evaluate(&HashMap::new()) {
                        if exp_val >= 0.0 && (exp_val - exp_val.round()).abs() < 1e-10 {
                            return exp_val.round() as usize;
                        }
                    }
                }
            }
            let base_deg = get_polynomial_degree(base, var);
            if base_deg == 0 {
                0
            } else if let Some(exp_val) = exp.evaluate(&HashMap::new()) {
                if exp_val >= 0.0 && (exp_val - exp_val.round()).abs() < 1e-10 {
                    base_deg * (exp_val.round() as usize)
                } else {
                    0
                }
            } else {
                0
            }
        }

        _ => 0,
    }
}
