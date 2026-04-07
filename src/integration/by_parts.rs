//! Integration by parts techniques.
//!
//! This module provides integration by parts, including the tabular method
//! and step-by-step explanations.

use crate::ast::{BinaryOp, Expression, UnaryOp};

use super::helpers::{
    combine_factors, expressions_equivalent, extract_factors, is_polynomial_like,
};
use super::substitution::{integrate_by_substitution, liate_priority};
use super::{IntegrationError, IntegrationResult};

/// Attempt to integrate using integration by parts: ∫u dv = uv - ∫v du
///
/// # Arguments
///
/// * `expr` - The expression to integrate (should be a product)
/// * `var` - The variable of integration
///
/// # Returns
///
/// The antiderivative if integration by parts succeeds, or an error.
///
/// # Example
///
/// ```
/// use thales::integration::integrate_by_parts;
/// use thales::ast::{Expression, Variable, BinaryOp};
///
/// // ∫x * e^x dx
/// let x = Expression::Variable(Variable::new("x"));
/// let e_x = Expression::Function(thales::ast::Function::Exp, vec![x.clone()]);
/// let expr = Expression::Binary(BinaryOp::Mul, Box::new(x), Box::new(e_x));
///
/// let result = integrate_by_parts(&expr, "x");
/// assert!(result.is_ok());
/// ```
pub fn integrate_by_parts(expr: &Expression, var: &str) -> IntegrationResult {
    integrate_by_parts_impl(expr, var, 0)
}

/// Maximum depth for integration by parts to prevent infinite loops.
const MAX_PARTS_DEPTH: usize = 10;

/// Internal implementation with depth tracking.
pub(super) fn integrate_by_parts_impl(
    expr: &Expression,
    var: &str,
    depth: usize,
) -> IntegrationResult {
    if depth > MAX_PARTS_DEPTH {
        return Err(IntegrationError::CannotIntegrate(
            "Integration by parts exceeded maximum depth".to_string(),
        ));
    }

    // First try regular integration
    if let Ok(result) = super::integrate_impl(expr, var) {
        return Ok(result);
    }

    // Try u-substitution
    if let Ok(result) = integrate_by_substitution(expr, var) {
        return Ok(result);
    }

    // Extract factors for integration by parts
    let factors = extract_factors(expr);

    if factors.len() < 2 {
        return Err(IntegrationError::CannotIntegrate(
            "Integration by parts requires a product".to_string(),
        ));
    }

    // Use LIATE to choose u (highest priority) and dv (rest)
    let (u, dv) = choose_u_and_dv(&factors, var);

    // Compute du = d(u)/dx
    let du = u.differentiate(var).simplify();

    // Compute v = ∫dv dx
    let v = match super::integrate_impl(&dv, var) {
        Ok(v) => v.simplify(),
        Err(_) => {
            // If we can't integrate dv directly, try recursively
            match integrate_by_parts_impl(&dv, var, depth + 1) {
                Ok(v) => v.simplify(),
                Err(e) => return Err(e),
            }
        }
    };

    // Now compute: uv - ∫v·du dx
    let uv = Expression::Binary(BinaryOp::Mul, Box::new(u.clone()), Box::new(v.clone())).simplify();

    let v_du =
        Expression::Binary(BinaryOp::Mul, Box::new(v.clone()), Box::new(du.clone())).simplify();

    // Check for recurring integral (integral of v*du equals original or a multiple)
    if let Some(result) = try_solve_recurring_integral(expr, &v_du, &uv, var, depth) {
        return Ok(result);
    }

    // Compute ∫v·du dx
    let integral_v_du = match super::integrate_impl(&v_du, var) {
        Ok(result) => result.simplify(),
        Err(_) => {
            // Try recursively with parts
            match integrate_by_parts_impl(&v_du, var, depth + 1) {
                Ok(result) => result.simplify(),
                Err(e) => return Err(e),
            }
        }
    };

    // Result: uv - ∫v·du
    let result =
        Expression::Binary(BinaryOp::Sub, Box::new(uv), Box::new(integral_v_du)).simplify();

    Ok(result)
}

/// Choose u and dv from factors using LIATE heuristic.
pub(super) fn choose_u_and_dv(factors: &[Expression], var: &str) -> (Expression, Expression) {
    // Find the factor with highest LIATE priority for u
    let mut best_u_idx = 0;
    let mut best_priority = 0;

    for (i, factor) in factors.iter().enumerate() {
        let priority = liate_priority(factor, var);
        if priority > best_priority {
            best_priority = priority;
            best_u_idx = i;
        }
    }

    let u = factors[best_u_idx].clone();
    let dv_factors: Vec<_> = factors
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != best_u_idx)
        .map(|(_, f)| f.clone())
        .collect();
    let dv = combine_factors(&dv_factors);

    (u, dv)
}

/// Try to solve a recurring integral.
///
/// When ∫f dx appears as part of the result, we can solve algebraically.
/// Example: ∫e^x·sin(x) dx = e^x·sin(x) - ∫e^x·cos(x) dx
///        = e^x·sin(x) - e^x·cos(x) + ∫e^x·sin(x) dx
/// So: 2∫e^x·sin(x) dx = e^x·sin(x) - e^x·cos(x)
///     ∫e^x·sin(x) dx = (e^x·sin(x) - e^x·cos(x))/2
fn try_solve_recurring_integral(
    original: &Expression,
    v_du: &Expression,
    uv: &Expression,
    var: &str,
    depth: usize,
) -> Option<Expression> {
    if depth >= 2 {
        // Only check for recurring at depth >= 2 to avoid false positives
        return None;
    }

    // Check if v_du is structurally similar to original
    // This is a simplified check - we compare simplified forms
    let original_simplified = original.simplify();
    let v_du_simplified = v_du.simplify();

    // Check if they're the same (up to constant multiple)
    if let Some(coefficient) =
        check_same_up_to_constant(&original_simplified, &v_du_simplified, var)
    {
        // ∫f dx = uv - c·∫f dx
        // (1 + c)∫f dx = uv
        // ∫f dx = uv / (1 + c)

        let one_plus_c = Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Integer(1)),
            Box::new(coefficient),
        )
        .simplify();

        // Check for 0 (would mean 0 = uv, which is a contradiction or identity)
        if matches!(one_plus_c, Expression::Integer(0)) {
            return None; // Can't divide by zero
        }

        let result = Expression::Binary(BinaryOp::Div, Box::new(uv.clone()), Box::new(one_plus_c))
            .simplify();

        return Some(result);
    }

    // Also check if we're in the case where applying parts twice gives back original
    // This requires tracking through two applications
    None
}

/// Check if two expressions are the same up to a constant multiple.
/// Returns the constant if expr2 = c * expr1.
pub(super) fn check_same_up_to_constant(
    expr1: &Expression,
    expr2: &Expression,
    var: &str,
) -> Option<Expression> {
    // Simple case: same expression means c = 1
    if expressions_equivalent(expr1, expr2) {
        return Some(Expression::Integer(1));
    }

    // Check if expr2 = -expr1
    if let Expression::Unary(UnaryOp::Neg, inner) = expr2 {
        if expressions_equivalent(expr1, inner) {
            return Some(Expression::Integer(-1));
        }
    }

    // Check if expr2 = c * expr1 where c is a constant
    if let Expression::Binary(BinaryOp::Mul, left, right) = expr2 {
        if !left.contains_variable(var) && expressions_equivalent(expr1, right) {
            return Some(left.as_ref().clone());
        }
        if !right.contains_variable(var) && expressions_equivalent(expr1, left) {
            return Some(right.as_ref().clone());
        }
    }

    None
}

/// Integration by parts with detailed steps.
///
/// Returns the result along with a step-by-step explanation.
pub fn integrate_by_parts_with_steps(
    expr: &Expression,
    var: &str,
) -> Result<(Expression, Vec<String>), IntegrationError> {
    let mut steps = Vec::new();
    steps.push(format!("∫{} d{}", expr, var));
    steps.push("Using integration by parts: ∫u dv = uv - ∫v du".to_string());

    // Extract factors
    let factors = extract_factors(expr);
    if factors.len() < 2 {
        return Err(IntegrationError::CannotIntegrate(
            "Integration by parts requires a product".to_string(),
        ));
    }

    // Choose u and dv
    let (u, dv) = choose_u_and_dv(&factors, var);
    steps.push(format!("Let u = {}", u));
    steps.push(format!("Let dv = {} d{}", dv, var));

    // Compute du
    let du = u.differentiate(var).simplify();
    steps.push(format!("Then du = {} d{}", du, var));

    // Compute v
    let v = match super::integrate_impl(&dv, var) {
        Ok(v) => v.simplify(),
        Err(_) => {
            return Err(IntegrationError::CannotIntegrate(format!(
                "Cannot integrate dv = {}",
                dv
            )))
        }
    };
    steps.push(format!("And v = ∫{} d{} = {}", dv, var, v));

    // Compute uv
    let uv = Expression::Binary(BinaryOp::Mul, Box::new(u.clone()), Box::new(v.clone())).simplify();
    steps.push(format!("uv = {}", uv));

    // Compute v·du
    let v_du =
        Expression::Binary(BinaryOp::Mul, Box::new(v.clone()), Box::new(du.clone())).simplify();
    steps.push(format!("v·du = {}", v_du));

    // Compute ∫v·du
    let integral_v_du = match integrate_by_parts_impl(&v_du, var, 1) {
        Ok(result) => result.simplify(),
        Err(e) => return Err(e),
    };
    steps.push(format!("∫v du = {}", integral_v_du));

    // Final result
    let result =
        Expression::Binary(BinaryOp::Sub, Box::new(uv), Box::new(integral_v_du)).simplify();
    steps.push(format!("Result: {}", result));

    Ok((result, steps))
}

/// Tabular method for integration by parts.
///
/// This method is efficient for integrals of the form ∫P(x)·f(x) dx
/// where P(x) is a polynomial and f(x) has an easy antiderivative (like e^x, sin(x), cos(x)).
///
/// The method repeatedly differentiates P(x) (until 0) and integrates f(x),
/// alternating signs: + - + - ...
///
/// # Example
///
/// ∫x²·e^x dx:
/// | Derivatives of x² | Integrals of e^x | Sign |
/// |-------------------|------------------|------|
/// | x²                | e^x              | +    |
/// | 2x                | e^x              | -    |
/// | 2                 | e^x              | +    |
/// | 0                 | e^x              | -    |
///
/// Result: x²·e^x - 2x·e^x + 2·e^x = (x² - 2x + 2)·e^x
pub fn tabular_integration(
    polynomial: &Expression,
    integrable: &Expression,
    var: &str,
) -> IntegrationResult {
    // Verify polynomial is actually polynomial-like
    if !is_polynomial_like(polynomial, var) {
        return Err(IntegrationError::CannotIntegrate(
            "First argument must be a polynomial".to_string(),
        ));
    }

    // Build the table
    let mut derivatives = Vec::new();
    let mut integrals = Vec::new();

    // Compute derivatives until we get 0
    let mut current_deriv = polynomial.clone();
    derivatives.push(current_deriv.clone());

    loop {
        current_deriv = current_deriv.differentiate(var).simplify();
        derivatives.push(current_deriv.clone());

        // Check if derivative is 0
        if matches!(current_deriv, Expression::Integer(0)) {
            break;
        }

        // Safety check to prevent infinite loops
        if derivatives.len() > 50 {
            return Err(IntegrationError::CannotIntegrate(
                "Polynomial degree too high for tabular method".to_string(),
            ));
        }
    }

    // Compute integrals
    let mut current_integral = integrable.clone();
    for _ in 0..derivatives.len() {
        integrals.push(current_integral.clone());
        current_integral = match super::integrate_impl(&current_integral, var) {
            Ok(i) => i.simplify(),
            Err(e) => return Err(e),
        };
    }

    // Combine with alternating signs: d[0]·i[0] - d[1]·i[1] + d[2]·i[2] - ...
    let mut result = Expression::Integer(0);
    let mut positive = true;

    for (d, i) in derivatives.iter().zip(integrals.iter()) {
        if matches!(d, Expression::Integer(0)) {
            break;
        }

        let term = Expression::Binary(BinaryOp::Mul, Box::new(d.clone()), Box::new(i.clone()));

        if positive {
            result = Expression::Binary(BinaryOp::Add, Box::new(result), Box::new(term));
        } else {
            result = Expression::Binary(BinaryOp::Sub, Box::new(result), Box::new(term));
        }

        positive = !positive;
    }

    Ok(result.simplify())
}
