//! Shared helper functions for the series module.

use crate::ast::{BinaryOp, Expression, Variable};
use std::collections::HashMap;

use super::{SeriesError, SeriesResult};

/// Try to convert a simple expression to f64.
/// Returns None for expressions that can't be directly converted.
pub(crate) fn try_expr_to_f64(expr: &Expression) -> Option<f64> {
    use crate::ast::{SymbolicConstant, UnaryOp};
    match expr {
        Expression::Integer(n) => Some(*n as f64),
        Expression::Float(f) => Some(*f),
        Expression::Rational(r) => Some(*r.numer() as f64 / *r.denom() as f64),
        Expression::Constant(c) => match c {
            SymbolicConstant::Pi => Some(std::f64::consts::PI),
            SymbolicConstant::E => Some(std::f64::consts::E),
            SymbolicConstant::I => None,
        },
        Expression::Unary(op, inner) => {
            let val = try_expr_to_f64(inner)?;
            match op {
                UnaryOp::Neg => Some(-val),
                UnaryOp::Abs => Some(val.abs()),
                _ => None,
            }
        }
        Expression::Binary(op, left, right) => {
            let l = try_expr_to_f64(left)?;
            let r = try_expr_to_f64(right)?;
            match op {
                BinaryOp::Add => Some(l + r),
                BinaryOp::Sub => Some(l - r),
                BinaryOp::Mul => Some(l * r),
                BinaryOp::Div if r.abs() > 1e-15 => Some(l / r),
                _ => None,
            }
        }
        Expression::Power(base, exp) => {
            let b = try_expr_to_f64(base)?;
            let e = try_expr_to_f64(exp)?;
            Some(b.powf(e))
        }
        _ => None,
    }
}

/// Format a coefficient for LaTeX output.
pub(crate) fn format_coefficient_latex(expr: &Expression) -> String {
    match expr {
        Expression::Integer(n) => n.to_string(),
        Expression::Float(x) => {
            if (x - x.round()).abs() < 1e-10 {
                format!("{}", x.round() as i64)
            } else {
                format!("{:.6}", x)
            }
        }
        Expression::Rational(r) => {
            format!("\\frac{{{}}}{{{}}}", r.numer(), r.denom())
        }
        _ => format!("{}", expr),
    }
}

/// Compute n! (factorial).
pub fn factorial(n: u32) -> u64 {
    if n <= 1 {
        1
    } else {
        (2..=n as u64).product()
    }
}

/// Compute n! as an Expression.
pub fn factorial_expr(n: u32) -> Expression {
    Expression::Integer(factorial(n) as i64)
}

/// Evaluate an expression at a specific value of a variable.
pub fn evaluate_at(
    expr: &Expression,
    var: &Variable,
    value: &Expression,
) -> SeriesResult<Expression> {
    // Create substitution
    let substituted = substitute(expr, var, value);

    // Try to simplify to a constant
    let simplified = substituted.simplify();

    // Check if it's a numeric result
    if let Some(val) = simplified.evaluate(&HashMap::new()) {
        if val.is_nan() {
            return Err(SeriesError::EvaluationFailed(format!(
                "Expression evaluates to NaN at {} = {}",
                var.name, value
            )));
        }
        if val.is_infinite() {
            return Err(SeriesError::EvaluationFailed(format!(
                "Expression evaluates to infinity at {} = {}",
                var.name, value
            )));
        }
        return Ok(Expression::Float(val));
    }

    Ok(simplified)
}

/// Substitute a variable with an expression.
fn substitute(expr: &Expression, var: &Variable, value: &Expression) -> Expression {
    match expr {
        Expression::Variable(v) if v.name == var.name => value.clone(),
        Expression::Binary(op, left, right) => Expression::Binary(
            *op,
            Box::new(substitute(left, var, value)),
            Box::new(substitute(right, var, value)),
        ),
        Expression::Unary(op, inner) => {
            Expression::Unary(*op, Box::new(substitute(inner, var, value)))
        }
        Expression::Function(func, args) => Expression::Function(
            func.clone(),
            args.iter().map(|a| substitute(a, var, value)).collect(),
        ),
        Expression::Power(base, exp) => Expression::Power(
            Box::new(substitute(base, var, value)),
            Box::new(substitute(exp, var, value)),
        ),
        _ => expr.clone(),
    }
}

/// Compute the nth derivative of an expression.
pub fn compute_nth_derivative(
    expr: &Expression,
    var: &Variable,
    n: u32,
) -> SeriesResult<Expression> {
    let mut result = expr.clone();
    for _ in 0..n {
        result = result.differentiate(&var.name);
        result = result.simplify();
    }
    Ok(result)
}

// Shared helper functions for Laurent series and other submodules

pub(crate) fn is_power_of_var(expr: &Expression, var: &Variable, center: &Expression) -> bool {
    match expr {
        Expression::Variable(v) if v.name == var.name => {
            matches!(center, Expression::Integer(0))
        }
        Expression::Power(base, _) => is_var_minus_center(base, var, center),
        Expression::Binary(BinaryOp::Sub, left, right) => {
            matches!(left.as_ref(), Expression::Variable(v) if v.name == var.name)
                && expressions_equal(right, center)
        }
        _ => false,
    }
}

pub(crate) fn is_var_minus_center(expr: &Expression, var: &Variable, center: &Expression) -> bool {
    match expr {
        Expression::Variable(v) if v.name == var.name => {
            matches!(center, Expression::Integer(0))
        }
        Expression::Binary(BinaryOp::Sub, left, right) => {
            matches!(left.as_ref(), Expression::Variable(v) if v.name == var.name)
                && expressions_equal(right, center)
        }
        _ => false,
    }
}

pub(crate) fn extract_integer(expr: &Expression) -> Option<i64> {
    match expr {
        Expression::Integer(n) => Some(*n),
        Expression::Float(f) if f.fract() == 0.0 => Some(*f as i64),
        _ => None,
    }
}

pub(crate) fn extract_positive_integer(expr: &Expression) -> Option<u32> {
    extract_integer(expr).and_then(|n| if n > 0 { Some(n as u32) } else { None })
}

pub(crate) fn expressions_equal(a: &Expression, b: &Expression) -> bool {
    // Simple equality check - could be improved with simplification
    match (a, b) {
        (Expression::Integer(x), Expression::Integer(y)) => x == y,
        (Expression::Float(x), Expression::Float(y)) => (x - y).abs() < 1e-15,
        (Expression::Integer(x), Expression::Float(y))
        | (Expression::Float(y), Expression::Integer(x)) => (*x as f64 - y).abs() < 1e-15,
        (Expression::Variable(v1), Expression::Variable(v2)) => v1.name == v2.name,
        _ => false,
    }
}
