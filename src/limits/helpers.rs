//! Helper functions for limit computation.

use crate::ast::{BinaryOp, Expression, Function, SymbolicConstant, UnaryOp, Variable};
use std::collections::HashMap;

use super::types::{try_expr_to_f64, IndeterminateForm, LimitError, LimitPoint, LimitResult};

pub(super) fn check_special_limits(
    expr: &Expression,
    var: &str,
    value: f64,
) -> Option<LimitResult> {
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
                                        if matches!(**base, Expression::Variable(ref v2) if v2.name == var)
                                        {
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
pub(super) fn detect_indeterminate_form(
    expr: &Expression,
    var: &str,
    value: f64,
) -> Result<LimitResult, LimitError> {
    match expr {
        Expression::Binary(BinaryOp::Div, num, denom) => {
            let num_val = evaluate_with_values(num, var, value).unwrap_or(f64::NAN);
            let denom_val = evaluate_with_values(denom, var, value).unwrap_or(f64::NAN);

            if num_val.abs() < 1e-15 && denom_val.abs() < 1e-15 {
                Err(LimitError::Indeterminate(IndeterminateForm::ZeroOverZero))
            } else if num_val.is_infinite() && denom_val.is_infinite() {
                Err(LimitError::Indeterminate(
                    IndeterminateForm::InfinityOverInfinity,
                ))
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
                Err(LimitError::Indeterminate(
                    IndeterminateForm::ZeroTimesInfinity,
                ))
            } else {
                Ok(LimitResult::Value(left_val * right_val))
            }
        }
        Expression::Binary(BinaryOp::Sub, left, right) => {
            let left_val = evaluate_with_values(left, var, value).unwrap_or(f64::NAN);
            let right_val = evaluate_with_values(right, var, value).unwrap_or(f64::NAN);

            if left_val.is_infinite()
                && right_val.is_infinite()
                && (left_val > 0.0) == (right_val > 0.0)
            {
                Err(LimitError::Indeterminate(
                    IndeterminateForm::InfinityMinusInfinity,
                ))
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
pub(super) fn check_infinity_direction(
    expr: &Expression,
    var: &str,
    value: f64,
) -> Result<LimitResult, LimitError> {
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
                        "Left and right limits differ".to_string(),
                    ))
                }
            } else {
                Ok(LimitResult::Value((l + r) / 2.0))
            }
        }
        _ => Err(LimitError::EvaluationError(
            "Cannot evaluate near limit point".to_string(),
        )),
    }
}

/// Evaluate an expression with a specific value for a variable.
pub(super) fn evaluate_with_values(
    expr: &Expression,
    var: &str,
    value: f64,
) -> Result<f64, LimitError> {
    let mut vars = HashMap::new();
    vars.insert(var.to_string(), value);
    evaluate_expr(expr, &vars)
}

/// Recursively evaluate an expression with variable values.
pub(super) fn evaluate_expr(
    expr: &Expression,
    vars: &HashMap<String, f64>,
) -> Result<f64, LimitError> {
    match expr {
        Expression::Integer(n) => Ok(*n as f64),
        Expression::Float(f) => Ok(*f),
        Expression::Rational(r) => Ok(*r.numer() as f64 / *r.denom() as f64),
        Expression::Complex(_) => Err(LimitError::EvaluationError(
            "Complex numbers not supported in limits".to_string(),
        )),
        Expression::Constant(c) => match c {
            SymbolicConstant::Pi => Ok(std::f64::consts::PI),
            SymbolicConstant::E => Ok(std::f64::consts::E),
            SymbolicConstant::I => Err(LimitError::EvaluationError(
                "Imaginary unit not supported in limits".to_string(),
            )),
        },
        Expression::Variable(v) => vars
            .get(&v.name)
            .copied()
            .ok_or_else(|| LimitError::EvaluationError(format!("Unbound variable: {}", v.name))),
        Expression::Unary(UnaryOp::Neg, inner) => Ok(-evaluate_expr(inner, vars)?),
        Expression::Unary(UnaryOp::Not, _) => Err(LimitError::EvaluationError(
            "Logical not not supported".to_string(),
        )),
        Expression::Unary(UnaryOp::Abs, inner) => Ok(evaluate_expr(inner, vars)?.abs()),
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
                        Err(LimitError::EvaluationError(
                            "atan2 requires 2 arguments".to_string(),
                        ))
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
                        Err(LimitError::EvaluationError(
                            "pow requires 2 arguments".to_string(),
                        ))
                    }
                }
                Function::Custom(_) => Err(LimitError::EvaluationError(
                    "Custom functions not supported".to_string(),
                )),
            }
        }
    }
}

/// Get the degree of a polynomial in the given variable.
pub(super) fn get_polynomial_degree(expr: &Expression, var: &str) -> i32 {
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
pub(super) fn get_leading_coefficient(expr: &Expression, var: &str) -> f64 {
    let degree = get_polynomial_degree(expr, var);
    extract_coefficient_for_degree(expr, var, degree)
}

/// Get the sign of the leading coefficient.
pub(super) fn get_leading_coefficient_sign(expr: &Expression, var: &str) -> f64 {
    let coef = get_leading_coefficient(expr, var);
    if coef >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Extract coefficient for a specific degree.
pub(super) fn extract_coefficient_for_degree(
    expr: &Expression,
    var: &str,
    target_degree: i32,
) -> f64 {
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
