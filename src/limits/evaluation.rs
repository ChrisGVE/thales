//! Limit evaluation algorithms.

use crate::ast::{BinaryOp, Expression, Function, SymbolicConstant, UnaryOp, Variable};
use std::collections::HashMap;

use super::helpers::{
    check_infinity_direction, check_special_limits, detect_indeterminate_form,
    evaluate_with_values, get_leading_coefficient, get_leading_coefficient_sign,
    get_polynomial_degree,
};
use super::types::{
    try_expr_to_f64, IndeterminateForm, LimitError, LimitPoint, LimitResult,
    MAX_LHOPITAL_ITERATIONS,
};

pub fn limit(
    expr: &Expression,
    var: &str,
    approaches: LimitPoint,
) -> Result<LimitResult, LimitError> {
    // First, try direct substitution
    match &approaches {
        LimitPoint::Value(val) => direct_substitution_limit(expr, var, *val),
        LimitPoint::PositiveInfinity => limit_at_infinity(expr, var, true),
        LimitPoint::NegativeInfinity => limit_at_infinity(expr, var, false),
    }
}

/// Evaluate limit from the left (approaching from below).
///
/// Computes `lim_{x -> a^-} f(x)`.
#[must_use = "computing limits returns a result that should be used"]
pub fn limit_left(
    expr: &Expression,
    var: &str,
    approaches: f64,
) -> Result<LimitResult, LimitError> {
    // Approach from slightly below
    let epsilon = 1e-10;
    let test_value = approaches - epsilon;

    // Try to evaluate at the test point
    let result = evaluate_with_values(expr, var, test_value)?;

    // Check for division by zero or other issues
    if result.is_nan() {
        return Err(LimitError::Undefined(
            "Left-hand limit undefined".to_string(),
        ));
    }

    if result.is_infinite() {
        if result > 0.0 {
            return Ok(LimitResult::PositiveInfinity);
        } else {
            return Ok(LimitResult::NegativeInfinity);
        }
    }

    // For continuous functions, the left limit equals the value
    Ok(LimitResult::Value(result))
}

/// Evaluate limit from the right (approaching from above).
///
/// Computes `lim_{x -> a^+} f(x)`.
#[must_use = "computing limits returns a result that should be used"]
pub fn limit_right(
    expr: &Expression,
    var: &str,
    approaches: f64,
) -> Result<LimitResult, LimitError> {
    // Approach from slightly above with progressively smaller epsilon
    let epsilons = [1e-3, 1e-6, 1e-9, 1e-12];

    let mut last_result = f64::NAN;
    for &epsilon in &epsilons {
        let test_value = approaches + epsilon;
        let result = evaluate_with_values(expr, var, test_value)?;

        if result.is_nan() {
            return Err(LimitError::Undefined(
                "Right-hand limit undefined".to_string(),
            ));
        }

        if result.is_infinite() {
            if result > 0.0 {
                return Ok(LimitResult::PositiveInfinity);
            } else {
                return Ok(LimitResult::NegativeInfinity);
            }
        }

        // Check if values are growing without bound
        if !last_result.is_nan() && result.abs() > last_result.abs() * 10.0 && result.abs() > 1e6 {
            // Values are growing rapidly - likely going to infinity
            if result > 0.0 {
                return Ok(LimitResult::PositiveInfinity);
            } else {
                return Ok(LimitResult::NegativeInfinity);
            }
        }

        last_result = result;
    }

    Ok(LimitResult::Value(last_result))
}

/// Evaluate the limit using L'Hôpital's rule when needed.
///
/// This function extends the basic `limit` function by automatically applying
/// L'Hôpital's rule when an indeterminate form (0/0 or ∞/∞) is detected.
///
/// # L'Hôpital's Rule
///
/// For limits of the form `lim_{x->a} f(x)/g(x)` where direct substitution
/// yields an indeterminate form, L'Hôpital's rule states:
///
/// ```text
/// lim_{x->a} f(x)/g(x) = lim_{x->a} f'(x)/g'(x)
/// ```
///
/// provided the limit on the right exists.
///
/// # Examples
///
/// ```
/// use thales::limits::{limit_with_lhopital, LimitPoint, LimitResult};
/// use thales::parser::parse_expression;
///
/// // lim_{x->0} sin(x)/x = 1 (using L'Hôpital: cos(x)/1 at x=0 = 1)
/// let expr = parse_expression("sin(x)/x").unwrap();
/// let result = limit_with_lhopital(&expr, "x", LimitPoint::Value(0.0)).unwrap();
/// if let LimitResult::Value(v) = result {
///     assert!((v - 1.0).abs() < 1e-10);
/// }
/// ```
///
/// # Arguments
///
/// * `expr` - The expression to evaluate
/// * `var` - The variable approaching the limit point
/// * `approaches` - The value the variable approaches
///
/// # Returns
///
/// * `Ok(LimitResult)` - The computed limit value
/// * `Err(LimitError)` - If the limit cannot be computed
#[must_use = "computing limits returns a result that should be used"]
pub fn limit_with_lhopital(
    expr: &Expression,
    var: &str,
    approaches: LimitPoint,
) -> Result<LimitResult, LimitError> {
    // First try regular limit
    match limit(expr, var, approaches.clone()) {
        Ok(result) => Ok(result),
        Err(LimitError::Indeterminate(IndeterminateForm::ZeroOverZero))
        | Err(LimitError::Indeterminate(IndeterminateForm::InfinityOverInfinity)) => {
            // Apply L'Hôpital's rule
            apply_lhopital_rule(expr, var, &approaches, 0)
        }
        Err(LimitError::Indeterminate(IndeterminateForm::ZeroTimesInfinity)) => {
            // Transform 0 * ∞ to 0/(1/f) or f/(1/g) form
            if let Some(result) = try_transform_zero_times_infinity(expr, var, &approaches) {
                result
            } else {
                Err(LimitError::Indeterminate(
                    IndeterminateForm::ZeroTimesInfinity,
                ))
            }
        }
        Err(e) => Err(e),
    }
}

/// Apply L'Hôpital's rule to a fraction.
///
/// Differentiates the numerator and denominator, then re-evaluates the limit.
fn apply_lhopital_rule(
    expr: &Expression,
    var: &str,
    approaches: &LimitPoint,
    depth: u32,
) -> Result<LimitResult, LimitError> {
    if depth >= MAX_LHOPITAL_ITERATIONS {
        return Err(LimitError::MaxIterationsExceeded);
    }

    // Expression must be a fraction
    if let Expression::Binary(BinaryOp::Div, num, denom) = expr {
        // Differentiate both numerator and denominator
        let num_derivative = num.differentiate(var);
        let denom_derivative = denom.differentiate(var);

        // Create the new fraction f'(x)/g'(x)
        let new_expr = Expression::Binary(
            BinaryOp::Div,
            Box::new(num_derivative.simplify()),
            Box::new(denom_derivative.simplify()),
        );

        // Try to evaluate the new limit
        match approaches {
            LimitPoint::Value(val) => {
                // Check for special limits first
                if let Some(result) = check_special_limits(&new_expr, var, *val) {
                    return Ok(result);
                }

                // Try direct evaluation
                match evaluate_with_values(&new_expr, var, *val) {
                    Ok(result) => {
                        if result.is_nan() {
                            // Still indeterminate - need to check form
                            let form = detect_indeterminate_form_type(&new_expr, var, *val);
                            match form {
                                Some(IndeterminateForm::ZeroOverZero)
                                | Some(IndeterminateForm::InfinityOverInfinity) => {
                                    // Apply L'Hôpital again
                                    apply_lhopital_rule(&new_expr, var, approaches, depth + 1)
                                }
                                Some(other) => Err(LimitError::Indeterminate(other)),
                                None => Ok(LimitResult::Value(result)),
                            }
                        } else if result.is_infinite() {
                            if result > 0.0 {
                                Ok(LimitResult::PositiveInfinity)
                            } else {
                                Ok(LimitResult::NegativeInfinity)
                            }
                        } else {
                            Ok(LimitResult::Value(result))
                        }
                    }
                    Err(_) => {
                        // Check if still indeterminate and apply again
                        let form = detect_indeterminate_form_type(&new_expr, var, *val);
                        match form {
                            Some(IndeterminateForm::ZeroOverZero)
                            | Some(IndeterminateForm::InfinityOverInfinity) => {
                                apply_lhopital_rule(&new_expr, var, approaches, depth + 1)
                            }
                            Some(other) => Err(LimitError::Indeterminate(other)),
                            None => Err(LimitError::EvaluationError(
                                "Could not evaluate after L'Hôpital's rule".to_string(),
                            )),
                        }
                    }
                }
            }
            LimitPoint::PositiveInfinity | LimitPoint::NegativeInfinity => {
                // For infinity limits, recursively apply limit_with_lhopital
                limit_with_lhopital(&new_expr, var, approaches.clone())
            }
        }
    } else {
        Err(LimitError::EvaluationError(
            "L'Hôpital's rule requires a fraction".to_string(),
        ))
    }
}

/// Detect what type of indeterminate form we have, if any.
fn detect_indeterminate_form_type(
    expr: &Expression,
    var: &str,
    value: f64,
) -> Option<IndeterminateForm> {
    match expr {
        Expression::Binary(BinaryOp::Div, num, denom) => {
            let num_val = evaluate_with_values(num, var, value).unwrap_or(f64::NAN);
            let denom_val = evaluate_with_values(denom, var, value).unwrap_or(f64::NAN);

            if num_val.abs() < 1e-15 && denom_val.abs() < 1e-15 {
                Some(IndeterminateForm::ZeroOverZero)
            } else if num_val.is_infinite() && denom_val.is_infinite() {
                Some(IndeterminateForm::InfinityOverInfinity)
            } else {
                None
            }
        }
        Expression::Binary(BinaryOp::Mul, left, right) => {
            let left_val = evaluate_with_values(left, var, value).unwrap_or(f64::NAN);
            let right_val = evaluate_with_values(right, var, value).unwrap_or(f64::NAN);

            if (left_val.abs() < 1e-15 && right_val.is_infinite())
                || (left_val.is_infinite() && right_val.abs() < 1e-15)
            {
                Some(IndeterminateForm::ZeroTimesInfinity)
            } else {
                None
            }
        }
        Expression::Power(base, exp) => {
            let base_val = evaluate_with_values(base, var, value).unwrap_or(f64::NAN);
            let exp_val = evaluate_with_values(exp, var, value).unwrap_or(f64::NAN);

            if base_val.abs() < 1e-15 && exp_val.abs() < 1e-15 {
                Some(IndeterminateForm::ZeroToZero)
            } else if (base_val - 1.0).abs() < 1e-15 && exp_val.is_infinite() {
                Some(IndeterminateForm::OneToInfinity)
            } else if base_val.is_infinite() && exp_val.abs() < 1e-15 {
                Some(IndeterminateForm::InfinityToZero)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Try to transform 0 * ∞ form to a fraction that L'Hôpital can handle.
fn try_transform_zero_times_infinity(
    expr: &Expression,
    var: &str,
    approaches: &LimitPoint,
) -> Option<Result<LimitResult, LimitError>> {
    if let Expression::Binary(BinaryOp::Mul, left, right) = expr {
        if let LimitPoint::Value(val) = approaches {
            let left_val = evaluate_with_values(left, var, *val).ok()?;
            let right_val = evaluate_with_values(right, var, *val).ok()?;

            if left_val.abs() < 1e-15 && right_val.is_infinite() {
                // Transform f * g (where f->0, g->∞) to f / (1/g)
                let new_denom = Expression::Binary(
                    BinaryOp::Div,
                    Box::new(Expression::Integer(1)),
                    right.clone(),
                );
                let new_expr = Expression::Binary(BinaryOp::Div, left.clone(), Box::new(new_denom));
                return Some(apply_lhopital_rule(&new_expr, var, approaches, 0));
            } else if left_val.is_infinite() && right_val.abs() < 1e-15 {
                // Transform f * g (where f->∞, g->0) to g / (1/f)
                let new_denom = Expression::Binary(
                    BinaryOp::Div,
                    Box::new(Expression::Integer(1)),
                    left.clone(),
                );
                let new_expr =
                    Expression::Binary(BinaryOp::Div, right.clone(), Box::new(new_denom));
                return Some(apply_lhopital_rule(&new_expr, var, approaches, 0));
            }
        }
    }
    None
}

/// Try direct substitution for computing the limit.
fn direct_substitution_limit(
    expr: &Expression,
    var: &str,
    value: f64,
) -> Result<LimitResult, LimitError> {
    // Check for special limits first
    if let Some(result) = check_special_limits(expr, var, value) {
        return Ok(result);
    }

    // Try to evaluate directly
    let result = evaluate_with_values(expr, var, value);

    match result {
        Ok(val) => {
            if val.is_nan() {
                // Need to check for indeterminate forms
                detect_indeterminate_form(expr, var, value)
            } else if val.is_infinite() {
                if val > 0.0 {
                    Ok(LimitResult::PositiveInfinity)
                } else {
                    Ok(LimitResult::NegativeInfinity)
                }
            } else {
                Ok(LimitResult::Value(val))
            }
        }
        Err(e) => {
            // Try to detect if it's an indeterminate form
            if let Err(LimitError::DivisionByZero) = detect_indeterminate_form(expr, var, value) {
                // It's a genuine division by zero, not 0/0
                // Check if it goes to +∞ or -∞
                check_infinity_direction(expr, var, value)
            } else {
                Err(e)
            }
        }
    }
}

/// Evaluate the limit as variable approaches infinity.
fn limit_at_infinity(
    expr: &Expression,
    var: &str,
    positive: bool,
) -> Result<LimitResult, LimitError> {
    // For polynomials and rational functions, analyze leading terms
    match expr {
        Expression::Variable(v) if v.name == var => {
            if positive {
                Ok(LimitResult::PositiveInfinity)
            } else {
                Ok(LimitResult::NegativeInfinity)
            }
        }
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Rational(_)
        | Expression::Constant(_) => {
            // Constants stay constant
            let val = try_expr_to_f64(expr).unwrap_or(0.0);
            Ok(LimitResult::Value(val))
        }
        Expression::Binary(BinaryOp::Div, num, denom) => {
            // For rational functions, compare polynomial degrees
            let num_degree = get_polynomial_degree(num, var);
            let denom_degree = get_polynomial_degree(denom, var);

            if num_degree > denom_degree {
                // Numerator dominates -> infinity
                let sign = get_leading_coefficient_sign(num, var)
                    * get_leading_coefficient_sign(denom, var);
                if (sign > 0.0) == positive {
                    Ok(LimitResult::PositiveInfinity)
                } else {
                    Ok(LimitResult::NegativeInfinity)
                }
            } else if num_degree < denom_degree {
                // Denominator dominates -> 0
                Ok(LimitResult::Value(0.0))
            } else {
                // Same degree -> ratio of leading coefficients
                let num_coef = get_leading_coefficient(num, var);
                let denom_coef = get_leading_coefficient(denom, var);
                Ok(LimitResult::Value(num_coef / denom_coef))
            }
        }
        Expression::Power(base, exp) => {
            // Check if base is the variable
            if matches!(**base, Expression::Variable(ref v) if v.name == var) {
                // Check for negative exponent: x^(-n) -> 0 as x -> ∞
                if let Expression::Unary(UnaryOp::Neg, _) = exp.as_ref() {
                    return Ok(LimitResult::Value(0.0));
                }
                // Check for positive integer exponent: x^n -> ∞ as x -> ∞
                if let Some(exp_val) = try_expr_to_f64(exp) {
                    if exp_val > 0.0 {
                        // x^n for positive n goes to +∞ (if n is even or x -> +∞)
                        // or -∞ if n is odd and x -> -∞
                        if positive {
                            return Ok(LimitResult::PositiveInfinity);
                        } else {
                            // Negative infinity: x^n where n is the exponent
                            let n = exp_val as i64;
                            if n % 2 == 0 {
                                return Ok(LimitResult::PositiveInfinity);
                            } else {
                                return Ok(LimitResult::NegativeInfinity);
                            }
                        }
                    } else if exp_val < 0.0 {
                        return Ok(LimitResult::Value(0.0));
                    }
                }
            }

            // Use numerical approximation for other cases
            let test_val = if positive { 1e10 } else { -1e10 };
            let result = evaluate_with_values(expr, var, test_val)?;

            if result.is_infinite() {
                if result > 0.0 {
                    Ok(LimitResult::PositiveInfinity)
                } else {
                    Ok(LimitResult::NegativeInfinity)
                }
            } else if result.abs() > 1e100 {
                // Very large values are effectively infinity
                if result > 0.0 {
                    Ok(LimitResult::PositiveInfinity)
                } else {
                    Ok(LimitResult::NegativeInfinity)
                }
            } else {
                Ok(LimitResult::Value(result))
            }
        }
        _ => {
            // Use numerical approximation for large values
            let test_val = if positive { 1e10 } else { -1e10 };
            match evaluate_with_values(expr, var, test_val) {
                Ok(result) => {
                    if result.is_infinite() {
                        if result > 0.0 {
                            Ok(LimitResult::PositiveInfinity)
                        } else {
                            Ok(LimitResult::NegativeInfinity)
                        }
                    } else if result.abs() < 1e-15 {
                        Ok(LimitResult::Value(0.0))
                    } else {
                        Ok(LimitResult::Value(result))
                    }
                }
                Err(e) => Err(e),
            }
        }
    }
}
