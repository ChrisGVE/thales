//! Polynomial analysis and root-finding utilities.

use crate::ast::{BinaryOp, Expression, SymbolicConstant, UnaryOp, Variable};
use std::collections::HashMap;

pub fn is_rational_function(expr: &Expression, var: &str) -> bool {
    match expr {
        // Division is the main case we check
        Expression::Binary(BinaryOp::Div, num, denom) => {
            is_polynomial(num, var) && is_polynomial(denom, var)
        }
        // A polynomial by itself is a rational function (denominator = 1)
        _ => is_polynomial(expr, var),
    }
}

/// Check if an expression is a polynomial in the given variable.
pub fn is_polynomial(expr: &Expression, var: &str) -> bool {
    match expr {
        Expression::Integer(_) | Expression::Float(_) | Expression::Rational(_) => true,
        Expression::Variable(_) => true, // Any variable is a polynomial
        Expression::Constant(_) => true,
        Expression::Unary(UnaryOp::Neg, inner) => is_polynomial(inner, var),
        Expression::Binary(op, left, right) => match op {
            BinaryOp::Add | BinaryOp::Sub => is_polynomial(left, var) && is_polynomial(right, var),
            BinaryOp::Mul => is_polynomial(left, var) && is_polynomial(right, var),
            BinaryOp::Div => {
                // Division is only polynomial if denominator doesn't contain the variable
                is_polynomial(left, var) && !contains_variable(right, var)
            }
            _ => false,
        },
        Expression::Power(base, exp) => {
            if let Expression::Integer(n) = exp.as_ref() {
                *n >= 0 && is_polynomial(base, var)
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Check if an expression contains the given variable.
pub(super) fn contains_variable(expr: &Expression, var: &str) -> bool {
    match expr {
        Expression::Variable(v) => v.name == var,
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Rational(_)
        | Expression::Constant(_) => false,
        Expression::Unary(_, inner) => contains_variable(inner, var),
        Expression::Binary(_, left, right) => {
            contains_variable(left, var) || contains_variable(right, var)
        }
        Expression::Power(base, exp) => contains_variable(base, var) || contains_variable(exp, var),
        Expression::Function(_, args) => args.iter().any(|arg| contains_variable(arg, var)),
        _ => false,
    }
}

/// Get the polynomial degree of an expression.
pub fn get_polynomial_degree(expr: &Expression, var: &str) -> Option<i32> {
    match expr {
        Expression::Integer(_) | Expression::Float(_) | Expression::Rational(_) => Some(0),
        Expression::Constant(_) => Some(0),
        Expression::Variable(v) => {
            if v.name == var {
                Some(1)
            } else {
                Some(0)
            }
        }
        Expression::Unary(UnaryOp::Neg, inner) => get_polynomial_degree(inner, var),
        Expression::Binary(op, left, right) => match op {
            BinaryOp::Add | BinaryOp::Sub => {
                let ld = get_polynomial_degree(left, var)?;
                let rd = get_polynomial_degree(right, var)?;
                Some(ld.max(rd))
            }
            BinaryOp::Mul => {
                let ld = get_polynomial_degree(left, var)?;
                let rd = get_polynomial_degree(right, var)?;
                Some(ld + rd)
            }
            BinaryOp::Div => {
                if !contains_variable(right, var) {
                    get_polynomial_degree(left, var)
                } else {
                    None
                }
            }
            _ => None,
        },
        Expression::Power(base, exp) => {
            if let Expression::Integer(n) = exp.as_ref() {
                if *n >= 0 {
                    let base_deg = get_polynomial_degree(base, var)?;
                    Some(base_deg * (*n as i32))
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Extract polynomial coefficients from an expression.
///
/// Returns a map from power to coefficient, e.g., for 3x² + 2x + 1:
/// {0: 1.0, 1: 2.0, 2: 3.0}
pub fn extract_coefficients(expr: &Expression, var: &str) -> Option<HashMap<i32, f64>> {
    let mut coeffs = HashMap::new();
    if extract_coefficients_impl(expr, var, 1.0, &mut coeffs) {
        Some(coeffs)
    } else {
        None
    }
}

pub(super) fn extract_coefficients_impl(
    expr: &Expression,
    var: &str,
    multiplier: f64,
    coeffs: &mut HashMap<i32, f64>,
) -> bool {
    match expr {
        Expression::Integer(n) => {
            *coeffs.entry(0).or_insert(0.0) += (*n as f64) * multiplier;
            true
        }
        Expression::Float(f) => {
            *coeffs.entry(0).or_insert(0.0) += f * multiplier;
            true
        }
        Expression::Rational(r) => {
            let val = *r.numer() as f64 / *r.denom() as f64;
            *coeffs.entry(0).or_insert(0.0) += val * multiplier;
            true
        }
        Expression::Constant(c) => {
            let val = match c {
                SymbolicConstant::Pi => std::f64::consts::PI,
                SymbolicConstant::E => std::f64::consts::E,
                SymbolicConstant::I => return false, // Can't handle imaginary
            };
            *coeffs.entry(0).or_insert(0.0) += val * multiplier;
            true
        }
        Expression::Variable(v) => {
            if v.name == var {
                *coeffs.entry(1).or_insert(0.0) += multiplier;
            } else {
                // Treat other variables as constants
                // For now, this is not fully supported
                *coeffs.entry(0).or_insert(0.0) += multiplier;
            }
            true
        }
        Expression::Unary(UnaryOp::Neg, inner) => {
            extract_coefficients_impl(inner, var, -multiplier, coeffs)
        }
        Expression::Binary(op, left, right) => match op {
            BinaryOp::Add => {
                extract_coefficients_impl(left, var, multiplier, coeffs)
                    && extract_coefficients_impl(right, var, multiplier, coeffs)
            }
            BinaryOp::Sub => {
                extract_coefficients_impl(left, var, multiplier, coeffs)
                    && extract_coefficients_impl(right, var, -multiplier, coeffs)
            }
            BinaryOp::Mul => {
                // Try to handle coefficient * variable
                if let Some(val) = evaluate_constant(left) {
                    extract_coefficients_impl(right, var, multiplier * val, coeffs)
                } else if let Some(val) = evaluate_constant(right) {
                    extract_coefficients_impl(left, var, multiplier * val, coeffs)
                } else if !contains_variable(left, var) {
                    if let Some(val) = evaluate_constant(left) {
                        extract_coefficients_impl(right, var, multiplier * val, coeffs)
                    } else {
                        false
                    }
                } else if !contains_variable(right, var) {
                    if let Some(val) = evaluate_constant(right) {
                        extract_coefficients_impl(left, var, multiplier * val, coeffs)
                    } else {
                        false
                    }
                } else {
                    // Both sides contain variable - need to expand
                    // For now, handle simple cases like x * x
                    if let (Expression::Variable(v1), Expression::Variable(v2)) =
                        (left.as_ref(), right.as_ref())
                    {
                        if v1.name == var && v2.name == var {
                            *coeffs.entry(2).or_insert(0.0) += multiplier;
                            return true;
                        }
                    }
                    false
                }
            }
            BinaryOp::Div => {
                if !contains_variable(right, var) {
                    if let Some(val) = evaluate_constant(right) {
                        if val.abs() < 1e-15 {
                            return false;
                        }
                        extract_coefficients_impl(left, var, multiplier / val, coeffs)
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            _ => false,
        },
        Expression::Power(base, exp) => {
            if let Expression::Integer(n) = exp.as_ref() {
                if *n >= 0 {
                    if let Expression::Variable(v) = base.as_ref() {
                        if v.name == var {
                            *coeffs.entry(*n as i32).or_insert(0.0) += multiplier;
                            return true;
                        }
                    }
                }
            }
            false
        }
        _ => false,
    }
}

/// Evaluate a constant expression to a float.
pub(super) fn evaluate_constant(expr: &Expression) -> Option<f64> {
    match expr {
        Expression::Integer(n) => Some(*n as f64),
        Expression::Float(f) => Some(*f),
        Expression::Rational(r) => Some(*r.numer() as f64 / *r.denom() as f64),
        Expression::Constant(c) => match c {
            SymbolicConstant::Pi => Some(std::f64::consts::PI),
            SymbolicConstant::E => Some(std::f64::consts::E),
            SymbolicConstant::I => None,
        },
        Expression::Unary(UnaryOp::Neg, inner) => evaluate_constant(inner).map(|v| -v),
        Expression::Binary(op, left, right) => {
            let lv = evaluate_constant(left)?;
            let rv = evaluate_constant(right)?;
            match op {
                BinaryOp::Add => Some(lv + rv),
                BinaryOp::Sub => Some(lv - rv),
                BinaryOp::Mul => Some(lv * rv),
                BinaryOp::Div => {
                    if rv.abs() < 1e-15 {
                        None
                    } else {
                        Some(lv / rv)
                    }
                }
                _ => None,
            }
        }
        Expression::Power(base, exp) => {
            let bv = evaluate_constant(base)?;
            let ev = evaluate_constant(exp)?;
            Some(bv.powf(ev))
        }
        _ => None,
    }
}

/// Find the real roots of a polynomial given its coefficients.
///
/// # Arguments
///
/// * `coeffs` - Map from power to coefficient
///
/// # Returns
///
/// A vector of (root, multiplicity) pairs for real roots.
pub(super) fn find_polynomial_roots(coeffs: &HashMap<i32, f64>) -> Vec<(f64, u32)> {
    let max_degree = coeffs.keys().copied().max().unwrap_or(0);

    if max_degree == 0 {
        return vec![];
    }

    if max_degree == 1 {
        // Linear: ax + b = 0 => x = -b/a
        let a = *coeffs.get(&1).unwrap_or(&0.0);
        let b = *coeffs.get(&0).unwrap_or(&0.0);
        if a.abs() < 1e-15 {
            return vec![];
        }
        return vec![(-b / a, 1)];
    }

    if max_degree == 2 {
        // Quadratic: ax² + bx + c = 0
        let a = *coeffs.get(&2).unwrap_or(&0.0);
        let b = *coeffs.get(&1).unwrap_or(&0.0);
        let c = *coeffs.get(&0).unwrap_or(&0.0);

        if a.abs() < 1e-15 {
            // Actually linear
            if b.abs() < 1e-15 {
                return vec![];
            }
            return vec![(-c / b, 1)];
        }

        let discriminant = b * b - 4.0 * a * c;
        if discriminant < -1e-15 {
            // No real roots (complex roots)
            return vec![];
        } else if discriminant.abs() < 1e-15 {
            // One repeated root
            let root = -b / (2.0 * a);
            return vec![(root, 2)];
        } else {
            // Two distinct roots
            let sqrt_disc = discriminant.sqrt();
            let r1 = (-b + sqrt_disc) / (2.0 * a);
            let r2 = (-b - sqrt_disc) / (2.0 * a);
            return vec![(r1, 1), (r2, 1)];
        }
    }

    // For higher degrees, use numerical methods
    // Try common roots first
    let mut roots = vec![];

    // Check integer roots from -10 to 10
    for i in -10..=10 {
        let x = i as f64;
        if evaluate_polynomial(coeffs, x).abs() < 1e-10 {
            roots.push((x, 1));
        }
    }

    // Try to find more roots using Newton-Raphson
    for start in &[-5.0, -1.0, 0.0, 1.0, 5.0] {
        if let Some(root) = newton_raphson(coeffs, *start) {
            // Check if we already have this root
            let already_found = roots.iter().any(|(r, _)| (r - root).abs() < 1e-10);
            if !already_found {
                roots.push((root, 1));
            }
        }
    }

    // Determine multiplicities by factoring out each root and checking again
    // This is a simplified approach
    roots
}

/// Evaluate a polynomial at a given point.
pub(super) fn evaluate_polynomial(coeffs: &HashMap<i32, f64>, x: f64) -> f64 {
    coeffs.iter().map(|(pow, coeff)| coeff * x.powi(*pow)).sum()
}

/// Evaluate the derivative of a polynomial at a given point.
pub(super) fn evaluate_polynomial_derivative(coeffs: &HashMap<i32, f64>, x: f64) -> f64 {
    coeffs
        .iter()
        .filter(|(pow, _)| **pow > 0)
        .map(|(pow, coeff)| (*pow as f64) * coeff * x.powi(*pow - 1))
        .sum()
}

/// Newton-Raphson method to find a root.
pub(super) fn newton_raphson(coeffs: &HashMap<i32, f64>, start: f64) -> Option<f64> {
    let mut x = start;
    for _ in 0..100 {
        let f = evaluate_polynomial(coeffs, x);
        let df = evaluate_polynomial_derivative(coeffs, x);
        if df.abs() < 1e-15 {
            return None;
        }
        let new_x = x - f / df;
        if (new_x - x).abs() < 1e-12 {
            if evaluate_polynomial(coeffs, new_x).abs() < 1e-10 {
                return Some(new_x);
            } else {
                return None;
            }
        }
        x = new_x;
    }
    None
}

/// Check if a quadratic is irreducible (no real roots).
pub(super) fn is_irreducible_quadratic(p: f64, q: f64) -> bool {
    // x² + px + q has discriminant p² - 4q
    let discriminant = p * p - 4.0 * q;
    discriminant < 0.0
}
