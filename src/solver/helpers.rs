//! Shared utility functions for equation solvers.

use crate::ast::{BinaryOp, Equation, Expression, UnaryOp};
use crate::resolution_path::ResolutionPathBuilder;
use std::collections::HashMap;

use super::SolverError;

/// Check if expression contains the given variable.
///
/// This is a convenience wrapper around [`Expression::contains_variable`].
pub(crate) fn contains_variable(expr: &Expression, var: &str) -> bool {
    expr.contains_variable(var)
}

/// Extract the coefficient of a variable from an expression.
///
/// Recognizes patterns like:
/// - `x` → coefficient is 1
/// - `3 * x` → coefficient is 3
/// - `x * 3` → coefficient is 3
/// - `a * x` → coefficient is a (where a doesn't contain x)
///
/// Returns `None` if the variable is not found or appears in a non-linear way.
pub(crate) fn extract_coefficient(expr: &Expression, var: &str) -> Option<Expression> {
    match expr {
        // x -> coefficient is 1
        Expression::Variable(v) if v.name == var => Some(Expression::Integer(1)),

        // a * x or x * a
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
/// If the expression contains only constants, evaluate it completely.
pub(crate) fn evaluate_constants(expr: &Expression) -> Expression {
    // First simplify
    let simplified = expr.simplify();

    // Try to evaluate if it's all constants
    if !has_any_variable(&simplified) {
        if let Some(value) = simplified.evaluate(&HashMap::new()) {
            // Check if it's an integer value
            if value.fract().abs() < 1e-10 {
                return Expression::Integer(value.round() as i64);
            } else {
                return Expression::Float(value);
            }
        }
    }

    simplified
}

/// Isolate a variable in an equation.
///
/// Rearranges the equation to solve for the target variable, returning the
/// expression that equals the variable. This is the core solving logic for
/// linear equations.
///
/// # Algorithm
///
/// Recognizes and solves several linear patterns:
/// 1. Variable already isolated: `x = expr` or `expr = x`
/// 2. Coefficient pattern: `a*x = c` → `x = c/a`
/// 3. Addition pattern: `x + b = c` → `x = c - b`
/// 4. Combined pattern: `a*x + b = c` → `x = (c - b)/a`
///
/// All patterns are checked in both left-to-right and right-to-left orientations.
pub(crate) fn isolate_variable(
    equation: &Equation,
    var: &str,
    _path: &mut ResolutionPathBuilder,
) -> Result<Expression, SolverError> {
    let left = &equation.left;
    let right = &equation.right;

    // Check if variable exists in equation
    if !contains_variable(left, var) && !contains_variable(right, var) {
        return Err(SolverError::CannotSolve(format!(
            "Variable '{}' not found in equation",
            var
        )));
    }

    // Special case: variable already isolated (x = expr or expr = x)
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

    // Try to solve simple patterns

    // Pattern: a * x = c  =>  x = c / a
    if let Some(coeff) = extract_coefficient(left, var) {
        if !contains_variable(right, var) {
            let result = Expression::Binary(
                BinaryOp::Div,
                Box::new(right.clone()),
                Box::new(coeff.clone()),
            )
            .simplify();
            let evaluated = evaluate_constants(&result);
            return Ok(evaluated);
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
            let evaluated = evaluate_constants(&result);
            return Ok(evaluated);
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
                let evaluated = evaluate_constants(&result);
                return Ok(evaluated);
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
                let evaluated = evaluate_constants(&result);
                return Ok(evaluated);
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
                let evaluated = evaluate_constants(&result);
                return Ok(evaluated);
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
                let evaluated = evaluate_constants(&result);
                return Ok(evaluated);
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
                let evaluated = evaluate_constants(&result);
                return Ok(evaluated);
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
                let evaluated = evaluate_constants(&result);
                return Ok(evaluated);
            }
        }
    }

    // More complex cases not yet supported
    Err(SolverError::CannotSolve(
        "Equation pattern not yet supported for Phase 1".to_string(),
    ))
}

/// Check if an expression has obvious non-linear features like x^2.
pub(crate) fn has_obvious_nonlinearity(expr: &Expression) -> bool {
    match expr {
        Expression::Power(base, exp) => {
            // x^2 or any variable raised to power > 1
            if has_any_variable(base) {
                // Check if exponent is > 1
                if let Some(exp_val) = exp.evaluate(&HashMap::new()) {
                    if exp_val > 1.0 {
                        return true;
                    }
                }
            }
            has_obvious_nonlinearity(base) || has_obvious_nonlinearity(exp)
        }
        Expression::Unary(_, inner) => has_obvious_nonlinearity(inner),
        Expression::Binary(_, left, right) => {
            has_obvious_nonlinearity(left) || has_obvious_nonlinearity(right)
        }
        Expression::Function(_, args) => args.iter().any(|arg| has_obvious_nonlinearity(arg)),
        _ => false,
    }
}

/// Check if expression contains any variables.
pub(crate) fn has_any_variable(expr: &Expression) -> bool {
    match expr {
        Expression::Variable(_) => true,
        Expression::Unary(_, inner) => has_any_variable(inner),
        Expression::Binary(_, left, right) => has_any_variable(left) || has_any_variable(right),
        Expression::Function(_, args) => args.iter().any(has_any_variable),
        Expression::Power(base, exp) => has_any_variable(base) || has_any_variable(exp),
        _ => false,
    }
}

/// Extract coefficients (a, b, c) from a quadratic expression ax² + bx + c.
pub(crate) fn extract_quadratic_coefficients(expr: &Expression, var: &str) -> (f64, f64, f64) {
    let mut a = 0.0;
    let mut b = 0.0;
    let mut c = 0.0;

    extract_poly_coefficients_recursive(expr, var, 1.0, &mut a, &mut b, &mut c);
    (a, b, c)
}

/// Recursively extract polynomial coefficients.
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
            // Other variable or constant treated as part of constant term
            if let Some(val) = expr.evaluate(&std::collections::HashMap::new()) {
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
            // Check for coefficient * variable or coefficient * x^2
            let left_val = left.evaluate(&std::collections::HashMap::new());
            let right_val = right.evaluate(&std::collections::HashMap::new());

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
                    // Could be x * x = x^2 or other variable products
                    if matches!(&**left, Expression::Variable(v) if v.name == var)
                        && matches!(&**right, Expression::Variable(v) if v.name == var)
                    {
                        *a += multiplier;
                    } else if matches!(&**left, Expression::Variable(v) if v.name == var) {
                        if let Some(rv) = right.evaluate(&std::collections::HashMap::new()) {
                            *b += multiplier * rv;
                        }
                    } else if matches!(&**right, Expression::Variable(v) if v.name == var) {
                        if let Some(lv) = left.evaluate(&std::collections::HashMap::new()) {
                            *b += multiplier * lv;
                        }
                    }
                }
            }
        }
        Expression::Power(base, exp) => {
            // Check for x^2 or x^n
            if matches!(&**base, Expression::Variable(v) if v.name == var) {
                if let Some(exp_val) = exp.evaluate(&std::collections::HashMap::new()) {
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
            // For other expressions, try to evaluate as constant
            if let Some(val) = expr.evaluate(&std::collections::HashMap::new()) {
                *c += val * multiplier;
            }
        }
    }
}

/// Simplify a numeric value to the best Expression representation.
pub(crate) fn simplify_numeric_expression(val: f64) -> Expression {
    // Check if it's close to an integer
    let rounded = val.round();
    if (val - rounded).abs() < 1e-10 && rounded.abs() < i64::MAX as f64 {
        Expression::Integer(rounded as i64)
    } else {
        Expression::Float(val)
    }
}

/// Extract coefficients for a general polynomial.
/// Returns vector of coefficients [a0, a1, a2, ..., an] for a0 + a1*x + a2*x^2 + ...
pub(crate) fn extract_polynomial_coefficients(
    expr: &Expression,
    var: &str,
    max_degree: usize,
) -> Vec<f64> {
    let mut coeffs = vec![0.0; max_degree + 1];
    extract_general_poly_coefficients(expr, var, 1.0, &mut coeffs);
    coeffs
}

/// Recursively extract general polynomial coefficients.
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
            if let Some(val) = expr.evaluate(&std::collections::HashMap::new()) {
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
            let left_val = left.evaluate(&std::collections::HashMap::new());
            let right_val = right.evaluate(&std::collections::HashMap::new());

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
                    // Variable * variable case
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
                if let Some(exp_val) = exp.evaluate(&std::collections::HashMap::new()) {
                    let degree = exp_val.round() as usize;
                    if degree < coeffs.len() {
                        coeffs[degree] += multiplier;
                    }
                }
            }
        }
        _ => {
            if let Some(val) = expr.evaluate(&std::collections::HashMap::new()) {
                coeffs[0] += val * multiplier;
            }
        }
    }
}

/// Get the degree of a polynomial expression with respect to a variable.
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
            // Division by variable increases complexity, treat as special case
            if contains_variable(right, var) {
                0 // Not a polynomial
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
            // For complex powers, multiply base degree by power
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

/// Check if an expression is polynomial (contains no transcendental functions).
pub(crate) fn is_polynomial_expression(expr: &Expression) -> bool {
    match expr {
        Expression::Integer(_)
        | Expression::Rational(_)
        | Expression::Float(_)
        | Expression::Complex(_)
        | Expression::Constant(_)
        | Expression::Variable(_) => true,

        Expression::Unary(_, inner) => is_polynomial_expression(inner),

        Expression::Binary(_, left, right) => {
            is_polynomial_expression(left) && is_polynomial_expression(right)
        }

        Expression::Power(base, exp) => {
            // Power is polynomial if base is polynomial and exponent is a non-negative integer
            if !is_polynomial_expression(base) {
                return false;
            }
            if let Some(exp_val) = exp.evaluate(&HashMap::new()) {
                exp_val >= 0.0 && (exp_val - exp_val.round()).abs() < 1e-10
            } else {
                // If exponent contains variables, check if it's polynomial
                is_polynomial_expression(exp)
            }
        }

        Expression::Function(_, _) => false, // Functions are transcendental
    }
}

/// Check if an expression is linear with respect to a specific variable.
/// An expression is linear in variable x if:
/// - x appears to at most power 1
/// - x does not appear in denominators
/// - x does not appear multiplied by itself
/// - x does not appear in functions
pub(crate) fn is_linear_in_variable(expr: &Expression, var: &str) -> bool {
    match expr {
        Expression::Integer(_)
        | Expression::Rational(_)
        | Expression::Float(_)
        | Expression::Complex(_)
        | Expression::Constant(_) => true,

        Expression::Variable(_v) => {
            // The target variable itself is linear
            true
        }

        Expression::Unary(_, inner) => is_linear_in_variable(inner, var),

        Expression::Binary(op, left, right) => {
            let left_has_var = contains_variable(left, var);
            let right_has_var = contains_variable(right, var);

            match op {
                BinaryOp::Add | BinaryOp::Sub => {
                    // x + y and x - y are linear if both sides are linear
                    is_linear_in_variable(left, var) && is_linear_in_variable(right, var)
                }
                BinaryOp::Mul => {
                    // For multiplication to be linear in x, at most one side can contain x
                    if left_has_var && right_has_var {
                        // x * x or x * f(x) is not linear
                        false
                    } else {
                        // a * x is linear
                        is_linear_in_variable(left, var) && is_linear_in_variable(right, var)
                    }
                }
                BinaryOp::Div => {
                    // x / a is linear, but a / x is not
                    if right_has_var {
                        false // Variable in denominator makes it non-linear
                    } else {
                        is_linear_in_variable(left, var)
                    }
                }
                _ => false,
            }
        }

        Expression::Power(base, exp) => {
            // x^2 is not linear, but a^x could be (though we don't handle that in Phase 1)
            // For Phase 1, we only allow constant powers where base doesn't have the variable
            !contains_variable(base, var) && is_linear_in_variable(exp, var)
        }

        Expression::Function(_, _) => {
            // For Phase 1, functions are not supported
            false
        }
    }
}

/// Substitute known variable values into an expression.
pub(crate) fn substitute_values(expr: &Expression, values: &HashMap<String, f64>) -> Expression {
    match expr {
        Expression::Variable(v) => {
            if let Some(&value) = values.get(&v.name) {
                Expression::Float(value)
            } else {
                expr.clone()
            }
        }
        Expression::Unary(op, inner) => {
            Expression::Unary(*op, Box::new(substitute_values(inner, values)))
        }
        Expression::Binary(op, left, right) => Expression::Binary(
            *op,
            Box::new(substitute_values(left, values)),
            Box::new(substitute_values(right, values)),
        ),
        Expression::Function(func, args) => Expression::Function(
            func.clone(),
            args.iter()
                .map(|arg| substitute_values(arg, values))
                .collect(),
        ),
        Expression::Power(base, exp) => Expression::Power(
            Box::new(substitute_values(base, values)),
            Box::new(substitute_values(exp, values)),
        ),
        _ => expr.clone(),
    }
}
