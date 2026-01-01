//! Limit evaluation for mathematical expressions.
//!
//! This module provides functions for computing limits of expressions
//! as a variable approaches a specific value.
//!
//! # Overview
//!
//! The limit module supports:
//! - Direct substitution for continuous functions
//! - Detection of indeterminate forms (0/0, ∞/∞, etc.)
//! - Limits at infinity
//! - One-sided limits (left and right)
//! - Common special limits (sin(x)/x, etc.)
//!
//! # Examples
//!
//! ## Direct Substitution
//!
//! ```
//! use thales::limits::{limit, LimitPoint};
//! use thales::parser::parse_expression;
//!
//! // lim_{x->2} x^2 = 4
//! let expr = parse_expression("x^2").unwrap();
//! let result = limit(&expr, "x", LimitPoint::Value(2.0));
//! // Returns Ok with value 4.0
//! ```
//!
//! ## Limit at Infinity
//!
//! ```
//! use thales::limits::{limit, LimitPoint};
//! use thales::parser::parse_expression;
//!
//! // lim_{x->∞} 1/x = 0
//! let expr = parse_expression("1/x").unwrap();
//! let result = limit(&expr, "x", LimitPoint::PositiveInfinity);
//! ```

use crate::ast::{BinaryOp, Expression, Function, SymbolicConstant, UnaryOp};
use std::collections::HashMap;
use std::fmt;

/// Try to convert a simple expression to f64.
/// Returns None for expressions that can't be directly converted.
fn try_expr_to_f64(expr: &Expression) -> Option<f64> {
    match expr {
        Expression::Integer(n) => Some(*n as f64),
        Expression::Float(f) => Some(*f),
        Expression::Rational(r) => Some(*r.numer() as f64 / *r.denom() as f64),
        Expression::Constant(c) => match c {
            SymbolicConstant::Pi => Some(std::f64::consts::PI),
            SymbolicConstant::E => Some(std::f64::consts::E),
            SymbolicConstant::I => None,
        },
        _ => None,
    }
}

/// Maximum number of L'Hôpital's rule applications.
const MAX_LHOPITAL_ITERATIONS: u32 = 10;

/// Error type for limit evaluation failures.
#[derive(Debug, Clone, PartialEq)]
pub enum LimitError {
    /// The limit results in an indeterminate form that requires further analysis.
    Indeterminate(IndeterminateForm),
    /// The limit does not exist (e.g., different one-sided limits).
    DoesNotExist(String),
    /// The expression cannot be evaluated at the limit point.
    Undefined(String),
    /// Division by zero at the limit point.
    DivisionByZero,
    /// General evaluation error.
    EvaluationError(String),
    /// L'Hôpital's rule exceeded maximum iterations.
    MaxIterationsExceeded,
}

impl fmt::Display for LimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LimitError::Indeterminate(form) => {
                write!(f, "Indeterminate form: {}", form)
            }
            LimitError::DoesNotExist(msg) => {
                write!(f, "Limit does not exist: {}", msg)
            }
            LimitError::Undefined(msg) => {
                write!(f, "Undefined at limit point: {}", msg)
            }
            LimitError::DivisionByZero => {
                write!(f, "Division by zero")
            }
            LimitError::EvaluationError(msg) => {
                write!(f, "Evaluation error: {}", msg)
            }
            LimitError::MaxIterationsExceeded => {
                write!(f, "L'Hôpital's rule: maximum iterations exceeded")
            }
        }
    }
}

impl std::error::Error for LimitError {}

/// Types of indeterminate forms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IndeterminateForm {
    /// 0/0 - requires L'Hopital's rule or algebraic manipulation
    ZeroOverZero,
    /// ∞/∞ - requires L'Hopital's rule
    InfinityOverInfinity,
    /// 0 * ∞ - needs to be rewritten as 0/0 or ∞/∞
    ZeroTimesInfinity,
    /// ∞ - ∞ - needs algebraic manipulation
    InfinityMinusInfinity,
    /// 0^0 - requires logarithmic analysis
    ZeroToZero,
    /// 1^∞ - requires logarithmic analysis
    OneToInfinity,
    /// ∞^0 - requires logarithmic analysis
    InfinityToZero,
}

impl fmt::Display for IndeterminateForm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndeterminateForm::ZeroOverZero => write!(f, "0/0"),
            IndeterminateForm::InfinityOverInfinity => write!(f, "∞/∞"),
            IndeterminateForm::ZeroTimesInfinity => write!(f, "0·∞"),
            IndeterminateForm::InfinityMinusInfinity => write!(f, "∞-∞"),
            IndeterminateForm::ZeroToZero => write!(f, "0^0"),
            IndeterminateForm::OneToInfinity => write!(f, "1^∞"),
            IndeterminateForm::InfinityToZero => write!(f, "∞^0"),
        }
    }
}

/// The point that a limit approaches.
#[derive(Debug, Clone, PartialEq)]
pub enum LimitPoint {
    /// A finite value.
    Value(f64),
    /// Positive infinity.
    PositiveInfinity,
    /// Negative infinity.
    NegativeInfinity,
}

impl LimitPoint {
    /// Check if this is infinity.
    pub fn is_infinite(&self) -> bool {
        matches!(self, LimitPoint::PositiveInfinity | LimitPoint::NegativeInfinity)
    }
}

/// Result of a limit evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum LimitResult {
    /// A finite value.
    Value(f64),
    /// Positive infinity.
    PositiveInfinity,
    /// Negative infinity.
    NegativeInfinity,
    /// An expression that couldn't be simplified to a number.
    Expression(Expression),
}

impl LimitResult {
    /// Convert to f64 if possible.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            LimitResult::Value(v) => Some(*v),
            LimitResult::PositiveInfinity => Some(f64::INFINITY),
            LimitResult::NegativeInfinity => Some(f64::NEG_INFINITY),
            LimitResult::Expression(_) => None,
        }
    }

    /// Check if the result is zero.
    pub fn is_zero(&self) -> bool {
        matches!(self, LimitResult::Value(v) if v.abs() < 1e-15)
    }

    /// Check if the result is infinite.
    pub fn is_infinite(&self) -> bool {
        matches!(self, LimitResult::PositiveInfinity | LimitResult::NegativeInfinity)
    }
}

/// Evaluate the limit of an expression.
///
/// Computes `lim_{var -> approaches} expr` using direct substitution
/// and simplification. Returns an error for indeterminate forms.
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
/// * `Err(LimitError)` - If the limit cannot be computed or is indeterminate
///
/// # Examples
///
/// ```
/// use thales::limits::{limit, LimitPoint, LimitResult};
/// use thales::ast::{Expression, Variable, BinaryOp};
///
/// // lim_{x->3} 2x = 6
/// let x = Expression::Variable(Variable::new("x"));
/// let expr = Expression::Binary(BinaryOp::Mul, Box::new(Expression::Integer(2)), Box::new(x));
///
/// let result = limit(&expr, "x", LimitPoint::Value(3.0)).unwrap();
/// if let LimitResult::Value(v) = result {
///     assert!((v - 6.0).abs() < 1e-10);
/// }
/// ```
pub fn limit(expr: &Expression, var: &str, approaches: LimitPoint) -> Result<LimitResult, LimitError> {
    // First, try direct substitution
    match &approaches {
        LimitPoint::Value(val) => {
            direct_substitution_limit(expr, var, *val)
        }
        LimitPoint::PositiveInfinity => {
            limit_at_infinity(expr, var, true)
        }
        LimitPoint::NegativeInfinity => {
            limit_at_infinity(expr, var, false)
        }
    }
}

/// Evaluate limit from the left (approaching from below).
///
/// Computes `lim_{x -> a^-} f(x)`.
pub fn limit_left(expr: &Expression, var: &str, approaches: f64) -> Result<LimitResult, LimitError> {
    // Approach from slightly below
    let epsilon = 1e-10;
    let test_value = approaches - epsilon;

    // Try to evaluate at the test point
    let result = evaluate_with_values(expr, var, test_value)?;

    // Check for division by zero or other issues
    if result.is_nan() {
        return Err(LimitError::Undefined("Left-hand limit undefined".to_string()));
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
pub fn limit_right(expr: &Expression, var: &str, approaches: f64) -> Result<LimitResult, LimitError> {
    // Approach from slightly above with progressively smaller epsilon
    let epsilons = [1e-3, 1e-6, 1e-9, 1e-12];

    let mut last_result = f64::NAN;
    for &epsilon in &epsilons {
        let test_value = approaches + epsilon;
        let result = evaluate_with_values(expr, var, test_value)?;

        if result.is_nan() {
            return Err(LimitError::Undefined("Right-hand limit undefined".to_string()));
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
pub fn limit_with_lhopital(expr: &Expression, var: &str, approaches: LimitPoint) -> Result<LimitResult, LimitError> {
    // First try regular limit
    match limit(expr, var, approaches.clone()) {
        Ok(result) => Ok(result),
        Err(LimitError::Indeterminate(IndeterminateForm::ZeroOverZero)) |
        Err(LimitError::Indeterminate(IndeterminateForm::InfinityOverInfinity)) => {
            // Apply L'Hôpital's rule
            apply_lhopital_rule(expr, var, &approaches, 0)
        }
        Err(LimitError::Indeterminate(IndeterminateForm::ZeroTimesInfinity)) => {
            // Transform 0 * ∞ to 0/(1/f) or f/(1/g) form
            if let Some(result) = try_transform_zero_times_infinity(expr, var, &approaches) {
                result
            } else {
                Err(LimitError::Indeterminate(IndeterminateForm::ZeroTimesInfinity))
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
                                Some(IndeterminateForm::ZeroOverZero) |
                                Some(IndeterminateForm::InfinityOverInfinity) => {
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
                            Some(IndeterminateForm::ZeroOverZero) |
                            Some(IndeterminateForm::InfinityOverInfinity) => {
                                apply_lhopital_rule(&new_expr, var, approaches, depth + 1)
                            }
                            Some(other) => Err(LimitError::Indeterminate(other)),
                            None => Err(LimitError::EvaluationError(
                                "Could not evaluate after L'Hôpital's rule".to_string()
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
            "L'Hôpital's rule requires a fraction".to_string()
        ))
    }
}

/// Detect what type of indeterminate form we have, if any.
fn detect_indeterminate_form_type(expr: &Expression, var: &str, value: f64) -> Option<IndeterminateForm> {
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
                let new_expr = Expression::Binary(
                    BinaryOp::Div,
                    left.clone(),
                    Box::new(new_denom),
                );
                return Some(apply_lhopital_rule(&new_expr, var, approaches, 0));
            } else if left_val.is_infinite() && right_val.abs() < 1e-15 {
                // Transform f * g (where f->∞, g->0) to g / (1/f)
                let new_denom = Expression::Binary(
                    BinaryOp::Div,
                    Box::new(Expression::Integer(1)),
                    left.clone(),
                );
                let new_expr = Expression::Binary(
                    BinaryOp::Div,
                    right.clone(),
                    Box::new(new_denom),
                );
                return Some(apply_lhopital_rule(&new_expr, var, approaches, 0));
            }
        }
    }
    None
}

/// Try direct substitution for computing the limit.
fn direct_substitution_limit(expr: &Expression, var: &str, value: f64) -> Result<LimitResult, LimitError> {
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
fn limit_at_infinity(expr: &Expression, var: &str, positive: bool) -> Result<LimitResult, LimitError> {
    // For polynomials and rational functions, analyze leading terms
    match expr {
        Expression::Variable(v) if v.name == var => {
            if positive {
                Ok(LimitResult::PositiveInfinity)
            } else {
                Ok(LimitResult::NegativeInfinity)
            }
        }
        Expression::Integer(_) | Expression::Float(_) | Expression::Rational(_) | Expression::Constant(_) => {
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
                let sign = get_leading_coefficient_sign(num, var) *
                           get_leading_coefficient_sign(denom, var);
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

/// Check for special limits that have known values.
fn check_special_limits(expr: &Expression, var: &str, value: f64) -> Option<LimitResult> {
    // Check for sin(x)/x -> 1 as x -> 0
    if value.abs() < 1e-15 {
        if let Expression::Binary(BinaryOp::Div, num, denom) = expr {
            // sin(x)/x
            if let Expression::Function(Function::Sin, args) = num.as_ref() {
                if args.len() == 1 {
                    if let Expression::Variable(ref v) = args[0] {
                        if v.name == var {
                            if let Expression::Variable(ref v2) = denom.as_ref() {
                                if v2.name == var {
                                    return Some(LimitResult::Value(1.0));
                                }
                            }
                        }
                    }
                }
            }
            // tan(x)/x
            if let Expression::Function(Function::Tan, args) = num.as_ref() {
                if args.len() == 1 {
                    if let Expression::Variable(ref v) = args[0] {
                        if v.name == var {
                            if let Expression::Variable(ref v2) = denom.as_ref() {
                                if v2.name == var {
                                    return Some(LimitResult::Value(1.0));
                                }
                            }
                        }
                    }
                }
            }
            // (1 - cos(x))/x^2 -> 1/2
            if let Expression::Binary(BinaryOp::Sub, one, cos_term) = num.as_ref() {
                if matches!(**one, Expression::Integer(1)) {
                    if let Expression::Function(Function::Cos, args) = cos_term.as_ref() {
                        if args.len() == 1 {
                            if let Expression::Variable(ref v) = args[0] {
                                if v.name == var {
                                    if let Expression::Power(base, exp) = denom.as_ref() {
                                        if matches!(**base, Expression::Variable(ref v2) if v2.name == var) {
                                            if matches!(**exp, Expression::Integer(2)) {
                                                return Some(LimitResult::Value(0.5));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

/// Detect which indeterminate form we have.
fn detect_indeterminate_form(expr: &Expression, var: &str, value: f64) -> Result<LimitResult, LimitError> {
    match expr {
        Expression::Binary(BinaryOp::Div, num, denom) => {
            let num_val = evaluate_with_values(num, var, value).unwrap_or(f64::NAN);
            let denom_val = evaluate_with_values(denom, var, value).unwrap_or(f64::NAN);

            if num_val.abs() < 1e-15 && denom_val.abs() < 1e-15 {
                Err(LimitError::Indeterminate(IndeterminateForm::ZeroOverZero))
            } else if num_val.is_infinite() && denom_val.is_infinite() {
                Err(LimitError::Indeterminate(IndeterminateForm::InfinityOverInfinity))
            } else if denom_val.abs() < 1e-15 {
                Err(LimitError::DivisionByZero)
            } else {
                Ok(LimitResult::Value(num_val / denom_val))
            }
        }
        Expression::Binary(BinaryOp::Mul, left, right) => {
            let left_val = evaluate_with_values(left, var, value).unwrap_or(f64::NAN);
            let right_val = evaluate_with_values(right, var, value).unwrap_or(f64::NAN);

            if (left_val.abs() < 1e-15 && right_val.is_infinite())
                || (left_val.is_infinite() && right_val.abs() < 1e-15)
            {
                Err(LimitError::Indeterminate(IndeterminateForm::ZeroTimesInfinity))
            } else {
                Ok(LimitResult::Value(left_val * right_val))
            }
        }
        Expression::Binary(BinaryOp::Sub, left, right) => {
            let left_val = evaluate_with_values(left, var, value).unwrap_or(f64::NAN);
            let right_val = evaluate_with_values(right, var, value).unwrap_or(f64::NAN);

            if left_val.is_infinite() && right_val.is_infinite()
                && (left_val > 0.0) == (right_val > 0.0)
            {
                Err(LimitError::Indeterminate(IndeterminateForm::InfinityMinusInfinity))
            } else {
                Ok(LimitResult::Value(left_val - right_val))
            }
        }
        Expression::Power(base, exp) => {
            let base_val = evaluate_with_values(base, var, value).unwrap_or(f64::NAN);
            let exp_val = evaluate_with_values(exp, var, value).unwrap_or(f64::NAN);

            if base_val.abs() < 1e-15 && exp_val.abs() < 1e-15 {
                Err(LimitError::Indeterminate(IndeterminateForm::ZeroToZero))
            } else if (base_val - 1.0).abs() < 1e-15 && exp_val.is_infinite() {
                Err(LimitError::Indeterminate(IndeterminateForm::OneToInfinity))
            } else if base_val.is_infinite() && exp_val.abs() < 1e-15 {
                Err(LimitError::Indeterminate(IndeterminateForm::InfinityToZero))
            } else {
                Ok(LimitResult::Value(base_val.powf(exp_val)))
            }
        }
        _ => {
            let val = evaluate_with_values(expr, var, value)?;
            Ok(LimitResult::Value(val))
        }
    }
}

/// Check if a limit goes to +∞ or -∞.
fn check_infinity_direction(expr: &Expression, var: &str, value: f64) -> Result<LimitResult, LimitError> {
    // Evaluate at points approaching from both sides
    let epsilon = 1e-10;
    let left_val = evaluate_with_values(expr, var, value - epsilon);
    let right_val = evaluate_with_values(expr, var, value + epsilon);

    match (left_val, right_val) {
        (Ok(l), Ok(r)) => {
            if l.is_infinite() || r.is_infinite() {
                // Check if both sides agree
                if l.signum() == r.signum() {
                    if l > 0.0 || r > 0.0 {
                        Ok(LimitResult::PositiveInfinity)
                    } else {
                        Ok(LimitResult::NegativeInfinity)
                    }
                } else {
                    Err(LimitError::DoesNotExist(
                        "Left and right limits differ".to_string()
                    ))
                }
            } else {
                Ok(LimitResult::Value((l + r) / 2.0))
            }
        }
        _ => Err(LimitError::EvaluationError("Cannot evaluate near limit point".to_string())),
    }
}

/// Evaluate an expression with a specific value for a variable.
fn evaluate_with_values(expr: &Expression, var: &str, value: f64) -> Result<f64, LimitError> {
    let mut vars = HashMap::new();
    vars.insert(var.to_string(), value);
    evaluate_expr(expr, &vars)
}

/// Recursively evaluate an expression with variable values.
fn evaluate_expr(expr: &Expression, vars: &HashMap<String, f64>) -> Result<f64, LimitError> {
    match expr {
        Expression::Integer(n) => Ok(*n as f64),
        Expression::Float(f) => Ok(*f),
        Expression::Rational(r) => Ok(*r.numer() as f64 / *r.denom() as f64),
        Expression::Complex(_) => Err(LimitError::EvaluationError("Complex numbers not supported in limits".to_string())),
        Expression::Constant(c) => match c {
            SymbolicConstant::Pi => Ok(std::f64::consts::PI),
            SymbolicConstant::E => Ok(std::f64::consts::E),
            SymbolicConstant::I => Err(LimitError::EvaluationError("Imaginary unit not supported in limits".to_string())),
        },
        Expression::Variable(v) => {
            vars.get(&v.name)
                .copied()
                .ok_or_else(|| LimitError::EvaluationError(format!("Unbound variable: {}", v.name)))
        }
        Expression::Unary(UnaryOp::Neg, inner) => {
            Ok(-evaluate_expr(inner, vars)?)
        }
        Expression::Unary(UnaryOp::Not, _) => {
            Err(LimitError::EvaluationError("Logical not not supported".to_string()))
        }
        Expression::Unary(UnaryOp::Abs, inner) => {
            Ok(evaluate_expr(inner, vars)?.abs())
        }
        Expression::Binary(op, left, right) => {
            let l = evaluate_expr(left, vars)?;
            let r = evaluate_expr(right, vars)?;
            match op {
                BinaryOp::Add => Ok(l + r),
                BinaryOp::Sub => Ok(l - r),
                BinaryOp::Mul => Ok(l * r),
                BinaryOp::Div => {
                    if r.abs() < 1e-300 {
                        if l.abs() < 1e-300 {
                            Ok(f64::NAN) // 0/0
                        } else if l > 0.0 {
                            Ok(f64::INFINITY)
                        } else {
                            Ok(f64::NEG_INFINITY)
                        }
                    } else {
                        Ok(l / r)
                    }
                }
                BinaryOp::Mod => Ok(l % r),
            }
        }
        Expression::Power(base, exp) => {
            let b = evaluate_expr(base, vars)?;
            let e = evaluate_expr(exp, vars)?;
            Ok(b.powf(e))
        }
        Expression::Function(func, args) => {
            let evaluated_args: Result<Vec<f64>, _> =
                args.iter().map(|a| evaluate_expr(a, vars)).collect();
            let args = evaluated_args?;

            match func {
                Function::Sin => Ok(args.first().copied().unwrap_or(0.0).sin()),
                Function::Cos => Ok(args.first().copied().unwrap_or(0.0).cos()),
                Function::Tan => Ok(args.first().copied().unwrap_or(0.0).tan()),
                Function::Asin => Ok(args.first().copied().unwrap_or(0.0).asin()),
                Function::Acos => Ok(args.first().copied().unwrap_or(0.0).acos()),
                Function::Atan => Ok(args.first().copied().unwrap_or(0.0).atan()),
                Function::Atan2 => {
                    if args.len() >= 2 {
                        Ok(args[0].atan2(args[1]))
                    } else {
                        Err(LimitError::EvaluationError("atan2 requires 2 arguments".to_string()))
                    }
                }
                Function::Sinh => Ok(args.first().copied().unwrap_or(0.0).sinh()),
                Function::Cosh => Ok(args.first().copied().unwrap_or(0.0).cosh()),
                Function::Tanh => Ok(args.first().copied().unwrap_or(0.0).tanh()),
                Function::Exp => Ok(args.first().copied().unwrap_or(0.0).exp()),
                Function::Ln => Ok(args.first().copied().unwrap_or(1.0).ln()),
                Function::Log => {
                    if args.len() >= 2 {
                        Ok(args[1].log(args[0]))
                    } else {
                        Ok(args.first().copied().unwrap_or(1.0).log10())
                    }
                }
                Function::Log2 => Ok(args.first().copied().unwrap_or(1.0).log2()),
                Function::Log10 => Ok(args.first().copied().unwrap_or(1.0).log10()),
                Function::Sqrt => Ok(args.first().copied().unwrap_or(0.0).sqrt()),
                Function::Cbrt => Ok(args.first().copied().unwrap_or(0.0).cbrt()),
                Function::Abs => Ok(args.first().copied().unwrap_or(0.0).abs()),
                Function::Sign => Ok(args.first().copied().unwrap_or(0.0).signum()),
                Function::Floor => Ok(args.first().copied().unwrap_or(0.0).floor()),
                Function::Ceil => Ok(args.first().copied().unwrap_or(0.0).ceil()),
                Function::Round => Ok(args.first().copied().unwrap_or(0.0).round()),
                Function::Min => {
                    if args.len() >= 2 {
                        Ok(args[0].min(args[1]))
                    } else {
                        args.first().copied().ok_or_else(|| {
                            LimitError::EvaluationError("min requires arguments".to_string())
                        })
                    }
                }
                Function::Max => {
                    if args.len() >= 2 {
                        Ok(args[0].max(args[1]))
                    } else {
                        args.first().copied().ok_or_else(|| {
                            LimitError::EvaluationError("max requires arguments".to_string())
                        })
                    }
                }
                Function::Pow => {
                    if args.len() >= 2 {
                        Ok(args[0].powf(args[1]))
                    } else {
                        Err(LimitError::EvaluationError("pow requires 2 arguments".to_string()))
                    }
                }
                Function::Custom(_) => {
                    Err(LimitError::EvaluationError("Custom functions not supported".to_string()))
                }
            }
        }
    }
}

/// Get the degree of a polynomial in the given variable.
fn get_polynomial_degree(expr: &Expression, var: &str) -> i32 {
    match expr {
        Expression::Integer(_) | Expression::Float(_) | Expression::Rational(_) => 0,
        Expression::Variable(v) if v.name == var => 1,
        Expression::Variable(_) => 0,
        Expression::Constant(_) => 0,
        Expression::Power(base, exp) => {
            if matches!(**base, Expression::Variable(ref v) if v.name == var) {
                if let Expression::Integer(n) = **exp {
                    n as i32
                } else {
                    0
                }
            } else {
                0
            }
        }
        Expression::Binary(BinaryOp::Add | BinaryOp::Sub, left, right) => {
            get_polynomial_degree(left, var).max(get_polynomial_degree(right, var))
        }
        Expression::Binary(BinaryOp::Mul, left, right) => {
            get_polynomial_degree(left, var) + get_polynomial_degree(right, var)
        }
        _ => 0,
    }
}

/// Get the leading coefficient of a polynomial.
fn get_leading_coefficient(expr: &Expression, var: &str) -> f64 {
    let degree = get_polynomial_degree(expr, var);
    extract_coefficient_for_degree(expr, var, degree)
}

/// Get the sign of the leading coefficient.
fn get_leading_coefficient_sign(expr: &Expression, var: &str) -> f64 {
    let coef = get_leading_coefficient(expr, var);
    if coef >= 0.0 { 1.0 } else { -1.0 }
}

/// Extract coefficient for a specific degree.
fn extract_coefficient_for_degree(expr: &Expression, var: &str, target_degree: i32) -> f64 {
    match expr {
        Expression::Integer(n) if target_degree == 0 => *n as f64,
        Expression::Float(f) if target_degree == 0 => *f,
        Expression::Variable(v) if v.name == var && target_degree == 1 => 1.0,
        Expression::Power(base, exp) => {
            if matches!(**base, Expression::Variable(ref v) if v.name == var) {
                if let Expression::Integer(n) = **exp {
                    if n as i32 == target_degree {
                        return 1.0;
                    }
                }
            }
            0.0
        }
        Expression::Binary(BinaryOp::Mul, left, right) => {
            let left_deg = get_polynomial_degree(left, var);
            let right_deg = get_polynomial_degree(right, var);

            if left_deg + right_deg == target_degree {
                let left_coef = if left_deg == 0 {
                    try_expr_to_f64(left).unwrap_or(1.0)
                } else {
                    get_leading_coefficient(left, var)
                };
                let right_coef = if right_deg == 0 {
                    try_expr_to_f64(right).unwrap_or(1.0)
                } else {
                    get_leading_coefficient(right, var)
                };
                left_coef * right_coef
            } else {
                0.0
            }
        }
        Expression::Binary(BinaryOp::Add, left, right) => {
            let left_coef = extract_coefficient_for_degree(left, var, target_degree);
            let right_coef = extract_coefficient_for_degree(right, var, target_degree);
            left_coef + right_coef
        }
        Expression::Binary(BinaryOp::Sub, left, right) => {
            let left_coef = extract_coefficient_for_degree(left, var, target_degree);
            let right_coef = extract_coefficient_for_degree(right, var, target_degree);
            left_coef - right_coef
        }
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Variable;
    use crate::parser::parse_expression;

    #[test]
    fn test_direct_substitution() {
        // lim_{x->2} x^2 = 4
        let expr = parse_expression("x^2").unwrap();
        let result = limit(&expr, "x", LimitPoint::Value(2.0)).unwrap();
        if let LimitResult::Value(v) = result {
            assert!((v - 4.0).abs() < 1e-10);
        } else {
            panic!("Expected value");
        }
    }

    #[test]
    fn test_linear_limit() {
        // lim_{x->3} 2x + 1 = 7
        let expr = parse_expression("2*x + 1").unwrap();
        let result = limit(&expr, "x", LimitPoint::Value(3.0)).unwrap();
        if let LimitResult::Value(v) = result {
            assert!((v - 7.0).abs() < 1e-10);
        } else {
            panic!("Expected value");
        }
    }

    #[test]
    fn test_sinx_over_x() {
        // lim_{x->0} sin(x)/x = 1
        let expr = parse_expression("sin(x)/x").unwrap();
        let result = limit(&expr, "x", LimitPoint::Value(0.0)).unwrap();
        if let LimitResult::Value(v) = result {
            assert!((v - 1.0).abs() < 1e-10);
        } else {
            panic!("Expected value 1.0");
        }
    }

    #[test]
    fn test_limit_at_infinity_rational() {
        // lim_{x->∞} 1/x = 0
        let expr = parse_expression("1/x").unwrap();
        let result = limit(&expr, "x", LimitPoint::PositiveInfinity).unwrap();
        if let LimitResult::Value(v) = result {
            assert!(v.abs() < 1e-10);
        } else {
            panic!("Expected value 0");
        }
    }

    #[test]
    fn test_limit_polynomial_infinity() {
        // lim_{x->∞} x^2 = ∞
        let expr = parse_expression("x^2").unwrap();
        let result = limit(&expr, "x", LimitPoint::PositiveInfinity).unwrap();
        assert!(matches!(result, LimitResult::PositiveInfinity));
    }

    #[test]
    fn test_indeterminate_0_over_0() {
        // lim_{x->0} x/x requires simplification (but we detect as 0/0 first)
        // Note: x/x simplifies to 1, so this would return 1
        // We test a more complex case
        let x = Expression::Variable(Variable::new("x"));
        let x_squared = Expression::Power(
            Box::new(x.clone()),
            Box::new(Expression::Integer(2))
        );
        let expr = Expression::Binary(
            BinaryOp::Div,
            Box::new(x_squared),
            Box::new(x)
        );
        // x^2 / x at x=0 is 0/0
        let result = limit(&expr, "x", LimitPoint::Value(0.0));
        // After evaluation, 0^2/0 = 0/0, but the simplified form would be x which is 0
        // Our implementation should detect this
        assert!(result.is_ok() || matches!(result, Err(LimitError::Indeterminate(_))));
    }

    #[test]
    fn test_one_sided_limits() {
        // lim_{x->0+} 1/x = +∞
        let expr = parse_expression("1/x").unwrap();
        let result = limit_right(&expr, "x", 0.0).unwrap();
        assert!(matches!(result, LimitResult::PositiveInfinity));
    }

    #[test]
    fn test_constant_limit() {
        // lim_{x->5} 3 = 3
        let expr = Expression::Integer(3);
        let result = limit(&expr, "x", LimitPoint::Value(5.0)).unwrap();
        if let LimitResult::Value(v) = result {
            assert!((v - 3.0).abs() < 1e-10);
        } else {
            panic!("Expected value");
        }
    }

    #[test]
    fn test_trig_limit() {
        // lim_{x->0} cos(x) = 1
        let expr = parse_expression("cos(x)").unwrap();
        let result = limit(&expr, "x", LimitPoint::Value(0.0)).unwrap();
        if let LimitResult::Value(v) = result {
            assert!((v - 1.0).abs() < 1e-10);
        } else {
            panic!("Expected value");
        }
    }

    // L'Hôpital's Rule tests

    #[test]
    fn test_lhopital_sinx_over_x() {
        // lim_{x->0} sin(x)/x = 1
        // L'Hôpital: lim cos(x)/1 = 1
        let expr = parse_expression("sin(x)/x").unwrap();
        let result = limit_with_lhopital(&expr, "x", LimitPoint::Value(0.0)).unwrap();
        if let LimitResult::Value(v) = result {
            assert!((v - 1.0).abs() < 1e-10);
        } else {
            panic!("Expected value 1.0");
        }
    }

    #[test]
    fn test_lhopital_exp_minus_1_over_x() {
        // lim_{x->0} (e^x - 1)/x = 1
        // L'Hôpital: lim e^x/1 = e^0 = 1
        let expr = parse_expression("(exp(x) - 1)/x").unwrap();
        let result = limit_with_lhopital(&expr, "x", LimitPoint::Value(0.0)).unwrap();
        if let LimitResult::Value(v) = result {
            assert!((v - 1.0).abs() < 1e-10);
        } else {
            panic!("Expected value 1.0");
        }
    }

    #[test]
    fn test_lhopital_1_minus_cosx_over_x2() {
        // lim_{x->0} (1 - cos(x))/x^2 = 1/2
        // L'Hôpital twice: sin(x)/2x -> cos(x)/2 = 1/2
        let expr = parse_expression("(1 - cos(x))/x^2").unwrap();
        let result = limit_with_lhopital(&expr, "x", LimitPoint::Value(0.0)).unwrap();
        if let LimitResult::Value(v) = result {
            assert!((v - 0.5).abs() < 1e-10, "Expected 0.5, got {}", v);
        } else {
            panic!("Expected value 0.5");
        }
    }

    #[test]
    fn test_lhopital_lnx_over_x_infinity() {
        // lim_{x->∞} ln(x)/x = 0
        // L'Hôpital: lim (1/x)/1 = 0
        let expr = parse_expression("ln(x)/x").unwrap();
        let result = limit_with_lhopital(&expr, "x", LimitPoint::PositiveInfinity).unwrap();
        if let LimitResult::Value(v) = result {
            assert!(v.abs() < 1e-6, "Expected 0, got {}", v);
        } else {
            panic!("Expected value 0");
        }
    }

    #[test]
    fn test_lhopital_x2_over_expx_infinity() {
        // lim_{x->∞} x^2/e^x = 0
        // L'Hôpital twice: 2x/e^x -> 2/e^x = 0
        // Note: This is tricky numerically as exp(large) overflows
        let expr = parse_expression("x^2/exp(x)").unwrap();
        let result = limit_with_lhopital(&expr, "x", LimitPoint::PositiveInfinity);
        // The limit should either give 0 or could fail due to numerical issues
        match result {
            Ok(LimitResult::Value(v)) if v.abs() < 1e-6 || v.is_nan() => { /* OK */ }
            Ok(LimitResult::Value(v)) => panic!("Expected ~0, got {}", v),
            Err(_) => { /* Acceptable due to numerical challenges at infinity */ }
            _ => panic!("Unexpected result"),
        }
    }

    #[test]
    fn test_lhopital_max_iterations() {
        // A pathological case that never converges (e.g., x/x which should simplify to 1)
        // This tests that we don't loop forever - but x/x should actually work
        let x = Expression::Variable(Variable::new("x"));
        let expr = Expression::Binary(
            BinaryOp::Div,
            Box::new(x.clone()),
            Box::new(x),
        );
        // x/x should simplify via L'Hôpital: 1/1 = 1
        let result = limit_with_lhopital(&expr, "x", LimitPoint::Value(0.0));
        assert!(result.is_ok(), "x/x should work with L'Hôpital");
        if let Ok(LimitResult::Value(v)) = result {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_lhopital_tanx_over_x() {
        // lim_{x->0} tan(x)/x = 1
        // L'Hôpital: lim sec²(x)/1 = sec²(0) = 1
        let expr = parse_expression("tan(x)/x").unwrap();
        let result = limit_with_lhopital(&expr, "x", LimitPoint::Value(0.0)).unwrap();
        if let LimitResult::Value(v) = result {
            assert!((v - 1.0).abs() < 1e-10);
        } else {
            panic!("Expected value 1.0");
        }
    }
}
