//! Taylor and Maclaurin series expansion module.
//!
//! This module provides symbolic Taylor and Maclaurin series expansion capabilities
//! for mathematical expressions. It supports:
//!
//! - Taylor series expansion around arbitrary center points
//! - Maclaurin series (Taylor series centered at 0)
//! - Built-in series for common functions (exp, sin, cos, ln, arctan)
//! - Polynomial output and LaTeX rendering
//! - Numerical evaluation of series approximations
//!
//! # Examples
//!
//! ```rust
//! use thales::series::{taylor, maclaurin};
//! use thales::ast::{Expression, Variable};
//!
//! // Maclaurin series of e^x to 4th order
//! let x = Variable::new("x");
//! let expr = Expression::Function(thales::ast::Function::Exp, vec![Expression::Variable(x.clone())]);
//! let series = maclaurin(&expr, &x, 4).unwrap();
//! // Result: 1 + x + x²/2 + x³/6 + x⁴/24 + O(x⁵)
//! ```

use crate::ast::{BinaryOp, Expression, Function, Variable};
use std::collections::HashMap;
use std::fmt;

/// Error types for series expansion operations.
#[derive(Debug, Clone, PartialEq)]
pub enum SeriesError {
    /// The expression cannot be expanded as a series.
    CannotExpand(String),
    /// The center point is invalid for this expansion.
    InvalidCenter(String),
    /// Division by zero encountered during expansion.
    DivisionByZero,
    /// Differentiation failed during coefficient computation.
    DerivativeFailed(String),
    /// Evaluation at the center point failed.
    EvaluationFailed(String),
    /// Order must be non-negative.
    InvalidOrder(String),
}

impl fmt::Display for SeriesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeriesError::CannotExpand(msg) => write!(f, "Cannot expand expression: {}", msg),
            SeriesError::InvalidCenter(msg) => write!(f, "Invalid center point: {}", msg),
            SeriesError::DivisionByZero => write!(f, "Division by zero in series expansion"),
            SeriesError::DerivativeFailed(msg) => write!(f, "Differentiation failed: {}", msg),
            SeriesError::EvaluationFailed(msg) => write!(f, "Evaluation at center failed: {}", msg),
            SeriesError::InvalidOrder(msg) => write!(f, "Invalid order: {}", msg),
        }
    }
}

impl std::error::Error for SeriesError {}

/// Result type for series operations.
pub type SeriesResult<T> = Result<T, SeriesError>;

/// A single term in a power series: coefficient × (x - center)^power.
#[derive(Debug, Clone, PartialEq)]
pub struct SeriesTerm {
    /// The coefficient of this term (can be symbolic).
    pub coefficient: Expression,
    /// The power of (x - center) for this term.
    pub power: u32,
}

impl SeriesTerm {
    /// Create a new series term.
    pub fn new(coefficient: Expression, power: u32) -> Self {
        SeriesTerm { coefficient, power }
    }

    /// Check if this term has a zero coefficient.
    pub fn is_zero(&self) -> bool {
        matches!(&self.coefficient, Expression::Integer(0))
            || matches!(&self.coefficient, Expression::Float(x) if x.abs() < 1e-15)
    }
}

impl fmt::Display for SeriesTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.power == 0 {
            write!(f, "{}", self.coefficient)
        } else if self.power == 1 {
            write!(f, "{}·x", self.coefficient)
        } else {
            write!(f, "{}·x^{}", self.coefficient, self.power)
        }
    }
}

/// Remainder term representation for truncated series.
#[derive(Debug, Clone, PartialEq)]
pub enum RemainderTerm {
    /// Lagrange remainder with explicit error bound.
    Lagrange {
        /// Upper bound on the remainder.
        bound: Expression,
        /// Order of the remainder term.
        order: u32,
    },
    /// Big-O notation for asymptotic remainder.
    BigO {
        /// Order of the remainder (e.g., O(x^n) has order n).
        order: u32,
    },
}

impl RemainderTerm {
    /// Get the order of the remainder term.
    pub fn order(&self) -> u32 {
        match self {
            RemainderTerm::Lagrange { order, .. } => *order,
            RemainderTerm::BigO { order } => *order,
        }
    }

    /// Convert to LaTeX representation.
    pub fn to_latex(&self) -> String {
        match self {
            RemainderTerm::Lagrange { bound, order } => {
                format!("R_{{{}}}(x) \\leq {}", order, bound)
            }
            RemainderTerm::BigO { order } => {
                format!("O(x^{{{}}})", order)
            }
        }
    }
}

impl fmt::Display for RemainderTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RemainderTerm::Lagrange { order, .. } => write!(f, "R_{}(x)", order),
            RemainderTerm::BigO { order } => write!(f, "O(x^{})", order),
        }
    }
}

/// A complete power series representation.
#[derive(Debug, Clone, PartialEq)]
pub struct Series {
    /// The terms of the series, in ascending order of power.
    pub terms: Vec<SeriesTerm>,
    /// The center point of the expansion.
    pub center: Expression,
    /// The variable of expansion.
    pub variable: Variable,
    /// The highest order computed.
    pub order: u32,
    /// Optional remainder term.
    pub remainder: Option<RemainderTerm>,
}

impl Series {
    /// Create a new empty series.
    pub fn new(variable: Variable, center: Expression, order: u32) -> Self {
        Series {
            terms: Vec::new(),
            center,
            variable,
            order,
            remainder: None,
        }
    }

    /// Add a term to the series.
    pub fn add_term(&mut self, term: SeriesTerm) {
        if !term.is_zero() {
            self.terms.push(term);
        }
    }

    /// Set the remainder term.
    pub fn set_remainder(&mut self, remainder: RemainderTerm) {
        self.remainder = Some(remainder);
    }

    /// Get the number of non-zero terms.
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    /// Get a specific term by power.
    pub fn get_term(&self, power: u32) -> Option<&SeriesTerm> {
        self.terms.iter().find(|t| t.power == power)
    }

    /// Convert the series to a polynomial Expression.
    pub fn to_expression(&self) -> Expression {
        if self.terms.is_empty() {
            return Expression::Integer(0);
        }

        let var_expr = Expression::Variable(self.variable.clone());
        let is_centered_at_zero = matches!(&self.center, Expression::Integer(0))
            || matches!(&self.center, Expression::Float(x) if x.abs() < 1e-15);

        let mut result: Option<Expression> = None;

        for term in &self.terms {
            // Build (x - center)^power
            let power_base = if is_centered_at_zero {
                var_expr.clone()
            } else {
                Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(var_expr.clone()),
                    Box::new(self.center.clone()),
                )
            };

            let term_expr = if term.power == 0 {
                term.coefficient.clone()
            } else if term.power == 1 {
                Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(term.coefficient.clone()),
                    Box::new(power_base),
                )
            } else {
                Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(term.coefficient.clone()),
                    Box::new(Expression::Power(
                        Box::new(power_base),
                        Box::new(Expression::Integer(term.power as i64)),
                    )),
                )
            };

            result = Some(match result {
                None => term_expr,
                Some(acc) => Expression::Binary(BinaryOp::Add, Box::new(acc), Box::new(term_expr)),
            });
        }

        result.unwrap_or(Expression::Integer(0)).simplify()
    }

    /// Convert the series to LaTeX representation.
    pub fn to_latex(&self) -> String {
        if self.terms.is_empty() {
            return "0".to_string();
        }

        let is_centered_at_zero = matches!(&self.center, Expression::Integer(0));
        let var_name = &self.variable.name;

        let mut parts = Vec::new();
        for (i, term) in self.terms.iter().enumerate() {
            let coeff_str = format_coefficient_latex(&term.coefficient);

            let term_str = if term.power == 0 {
                coeff_str
            } else {
                let var_part = if is_centered_at_zero {
                    if term.power == 1 {
                        var_name.clone()
                    } else {
                        format!("{}^{{{}}}", var_name, term.power)
                    }
                } else {
                    if term.power == 1 {
                        format!("({} - {})", var_name, self.center)
                    } else {
                        format!("({} - {})^{{{}}}", var_name, self.center, term.power)
                    }
                };

                if coeff_str == "1" {
                    var_part
                } else if coeff_str == "-1" {
                    format!("-{}", var_part)
                } else {
                    format!("{} {}", coeff_str, var_part)
                }
            };

            if i == 0 {
                parts.push(term_str);
            } else {
                // Handle sign for subsequent terms
                if term_str.starts_with('-') {
                    parts.push(format!(" - {}", &term_str[1..]));
                } else {
                    parts.push(format!(" + {}", term_str));
                }
            }
        }

        let mut result = parts.join("");

        if let Some(ref remainder) = self.remainder {
            result.push_str(&format!(" + {}", remainder.to_latex()));
        }

        result
    }

    /// Numerically evaluate the series at a point.
    pub fn evaluate(&self, x: f64) -> Option<f64> {
        let center_val = self.center.evaluate(&HashMap::new())?;
        let dx = x - center_val;

        let mut sum = 0.0;
        for term in &self.terms {
            let coeff = term.coefficient.evaluate(&HashMap::new())?;
            sum += coeff * dx.powi(term.power as i32);
        }

        Some(sum)
    }
}

/// Format a coefficient for LaTeX output.
fn format_coefficient_latex(expr: &Expression) -> String {
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
pub fn evaluate_at(expr: &Expression, var: &Variable, value: &Expression) -> SeriesResult<Expression> {
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
        Expression::Unary(op, inner) => Expression::Unary(
            *op,
            Box::new(substitute(inner, var, value)),
        ),
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
pub fn compute_nth_derivative(expr: &Expression, var: &Variable, n: u32) -> SeriesResult<Expression> {
    let mut result = expr.clone();
    for _ in 0..n {
        result = result.differentiate(&var.name);
        result = result.simplify();
    }
    Ok(result)
}

/// Compute the Taylor series of an expression around a center point.
///
/// # Arguments
/// * `expr` - The expression to expand
/// * `var` - The variable of expansion
/// * `center` - The center point of expansion
/// * `order` - The number of terms to compute (0 through order)
///
/// # Returns
/// A `Series` containing the Taylor expansion terms and remainder.
pub fn taylor(
    expr: &Expression,
    var: &Variable,
    center: &Expression,
    order: u32,
) -> SeriesResult<Series> {
    // First, check if we can use a known series
    if let Some(series) = try_known_series(expr, var, center, order) {
        return Ok(series);
    }

    let mut series = Series::new(var.clone(), center.clone(), order);

    for n in 0..=order {
        // Compute nth derivative
        let nth_deriv = compute_nth_derivative(expr, var, n)?;

        // Evaluate at center
        let deriv_at_center = evaluate_at(&nth_deriv, var, center)?;

        // Compute coefficient: f^(n)(center) / n!
        let n_fact = factorial(n) as i64;
        let coefficient = if n_fact == 1 {
            deriv_at_center
        } else {
            Expression::Binary(
                BinaryOp::Div,
                Box::new(deriv_at_center),
                Box::new(Expression::Integer(n_fact)),
            ).simplify()
        };

        // Add term if non-zero
        let term = SeriesTerm::new(coefficient, n);
        series.add_term(term);
    }

    // Set remainder
    series.set_remainder(RemainderTerm::BigO { order: order + 1 });

    Ok(series)
}

/// Compute the Maclaurin series (Taylor series centered at 0).
pub fn maclaurin(
    expr: &Expression,
    var: &Variable,
    order: u32,
) -> SeriesResult<Series> {
    taylor(expr, var, &Expression::Integer(0), order)
}

/// Try to match the expression to a known series for efficiency.
fn try_known_series(
    expr: &Expression,
    var: &Variable,
    center: &Expression,
    order: u32,
) -> Option<Series> {
    // Only handle Maclaurin (center = 0) for built-in series
    if !matches!(center, Expression::Integer(0)) {
        return None;
    }

    match expr {
        Expression::Function(Function::Exp, args) if args.len() == 1 => {
            if matches!(&args[0], Expression::Variable(v) if v.name == var.name) {
                return Some(exp_series(var, order));
            }
        }
        Expression::Function(Function::Sin, args) if args.len() == 1 => {
            if matches!(&args[0], Expression::Variable(v) if v.name == var.name) {
                return Some(sin_series(var, order));
            }
        }
        Expression::Function(Function::Cos, args) if args.len() == 1 => {
            if matches!(&args[0], Expression::Variable(v) if v.name == var.name) {
                return Some(cos_series(var, order));
            }
        }
        Expression::Function(Function::Ln, args) if args.len() == 1 => {
            // Check for ln(1 + x)
            if let Expression::Binary(BinaryOp::Add, left, right) = &args[0] {
                if matches!(**left, Expression::Integer(1))
                    && matches!(**right, Expression::Variable(ref v) if v.name == var.name)
                {
                    return Some(ln_1_plus_x_series(var, order));
                }
            }
        }
        Expression::Function(Function::Atan, args) if args.len() == 1 => {
            if matches!(&args[0], Expression::Variable(v) if v.name == var.name) {
                return Some(arctan_series(var, order));
            }
        }
        _ => {}
    }

    None
}

/// Maclaurin series for e^x: sum(x^n / n!) for n = 0 to order.
pub fn exp_series(var: &Variable, order: u32) -> Series {
    let mut series = Series::new(var.clone(), Expression::Integer(0), order);

    for n in 0..=order {
        let coeff = if n == 0 {
            Expression::Integer(1)
        } else {
            let n_fact = factorial(n);
            Expression::Rational(num_rational::Ratio::new(1, n_fact as i64))
        };
        series.add_term(SeriesTerm::new(coeff, n));
    }

    series.set_remainder(RemainderTerm::BigO { order: order + 1 });
    series
}

/// Maclaurin series for sin(x): x - x³/3! + x⁵/5! - ...
pub fn sin_series(var: &Variable, order: u32) -> Series {
    let mut series = Series::new(var.clone(), Expression::Integer(0), order);

    let mut n = 0u32;
    while 2 * n + 1 <= order {
        let power = 2 * n + 1;
        let sign = if n % 2 == 0 { 1i64 } else { -1i64 };
        let fact = factorial(power) as i64;
        let coeff = Expression::Rational(num_rational::Ratio::new(sign, fact));
        series.add_term(SeriesTerm::new(coeff, power));
        n += 1;
    }

    series.set_remainder(RemainderTerm::BigO { order: order + 1 });
    series
}

/// Maclaurin series for cos(x): 1 - x²/2! + x⁴/4! - ...
pub fn cos_series(var: &Variable, order: u32) -> Series {
    let mut series = Series::new(var.clone(), Expression::Integer(0), order);

    let mut n = 0u32;
    while 2 * n <= order {
        let power = 2 * n;
        let sign = if n % 2 == 0 { 1i64 } else { -1i64 };
        let fact = if power == 0 { 1 } else { factorial(power) as i64 };
        let coeff = Expression::Rational(num_rational::Ratio::new(sign, fact));
        series.add_term(SeriesTerm::new(coeff, power));
        n += 1;
    }

    series.set_remainder(RemainderTerm::BigO { order: order + 1 });
    series
}

/// Maclaurin series for ln(1+x): x - x²/2 + x³/3 - x⁴/4 + ...
pub fn ln_1_plus_x_series(var: &Variable, order: u32) -> Series {
    let mut series = Series::new(var.clone(), Expression::Integer(0), order);

    for n in 1..=order {
        let sign = if n % 2 == 1 { 1i64 } else { -1i64 };
        let coeff = Expression::Rational(num_rational::Ratio::new(sign, n as i64));
        series.add_term(SeriesTerm::new(coeff, n));
    }

    series.set_remainder(RemainderTerm::BigO { order: order + 1 });
    series
}

/// Maclaurin series for arctan(x): x - x³/3 + x⁵/5 - x⁷/7 + ...
pub fn arctan_series(var: &Variable, order: u32) -> Series {
    let mut series = Series::new(var.clone(), Expression::Integer(0), order);

    let mut n = 0u32;
    while 2 * n + 1 <= order {
        let power = 2 * n + 1;
        let sign = if n % 2 == 0 { 1i64 } else { -1i64 };
        let coeff = Expression::Rational(num_rational::Ratio::new(sign, power as i64));
        series.add_term(SeriesTerm::new(coeff, power));
        n += 1;
    }

    series.set_remainder(RemainderTerm::BigO { order: order + 1 });
    series
}

/// Binomial series for (1+x)^a: sum(C(a,n) * x^n).
/// Only works for symbolic or numeric exponents.
pub fn binomial_series(exponent: &Expression, var: &Variable, order: u32) -> SeriesResult<Series> {
    let mut series = Series::new(var.clone(), Expression::Integer(0), order);

    // For now, only handle numeric exponents
    let a = match exponent.evaluate(&HashMap::new()) {
        Some(val) => val,
        None => return Err(SeriesError::CannotExpand(
            "Binomial series requires numeric exponent".to_string()
        )),
    };

    // Compute binomial coefficients C(a, n) = a*(a-1)*...*(a-n+1) / n!
    let mut binom_coeff = 1.0;
    for n in 0..=order {
        if n > 0 {
            binom_coeff *= (a - (n as f64 - 1.0)) / (n as f64);
        }

        if binom_coeff.abs() > 1e-15 {
            let coeff = if (binom_coeff - binom_coeff.round()).abs() < 1e-10 {
                Expression::Integer(binom_coeff.round() as i64)
            } else {
                Expression::Float(binom_coeff)
            };
            series.add_term(SeriesTerm::new(coeff, n));
        }

        // For positive integer a, series terminates
        if a.fract() == 0.0 && a >= 0.0 && n as f64 >= a {
            break;
        }
    }

    // Only add remainder if series doesn't terminate
    if a.fract() != 0.0 || a < 0.0 {
        series.set_remainder(RemainderTerm::BigO { order: order + 1 });
    }

    Ok(series)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(5), 120);
        assert_eq!(factorial(10), 3628800);
    }

    #[test]
    fn test_exp_series() {
        let x = Variable::new("x");
        let series = exp_series(&x, 4);

        assert_eq!(series.term_count(), 5);
        assert_eq!(series.order, 4);

        // Check coefficients: 1, 1, 1/2, 1/6, 1/24
        let term0 = series.get_term(0).unwrap();
        assert!(matches!(&term0.coefficient, Expression::Integer(1)));

        let term2 = series.get_term(2).unwrap();
        if let Expression::Rational(r) = &term2.coefficient {
            assert_eq!(*r.numer(), 1);
            assert_eq!(*r.denom(), 2);
        } else {
            panic!("Expected rational coefficient");
        }
    }

    #[test]
    fn test_sin_series() {
        let x = Variable::new("x");
        let series = sin_series(&x, 7);

        // Should have terms at powers 1, 3, 5, 7
        assert!(series.get_term(0).is_none());
        assert!(series.get_term(1).is_some());
        assert!(series.get_term(2).is_none());
        assert!(series.get_term(3).is_some());

        // First term should be x (coefficient 1)
        let term1 = series.get_term(1).unwrap();
        if let Expression::Rational(r) = &term1.coefficient {
            assert_eq!(*r.numer(), 1);
            assert_eq!(*r.denom(), 1);
        }

        // x^3 coefficient should be -1/6
        let term3 = series.get_term(3).unwrap();
        if let Expression::Rational(r) = &term3.coefficient {
            assert_eq!(*r.numer(), -1);
            assert_eq!(*r.denom(), 6);
        }
    }

    #[test]
    fn test_cos_series() {
        let x = Variable::new("x");
        let series = cos_series(&x, 6);

        // Should have terms at powers 0, 2, 4, 6
        assert!(series.get_term(0).is_some());
        assert!(series.get_term(1).is_none());
        assert!(series.get_term(2).is_some());

        // First term should be 1
        let term0 = series.get_term(0).unwrap();
        if let Expression::Rational(r) = &term0.coefficient {
            assert_eq!(*r.numer(), 1);
        }

        // x^2 coefficient should be -1/2
        let term2 = series.get_term(2).unwrap();
        if let Expression::Rational(r) = &term2.coefficient {
            assert_eq!(*r.numer(), -1);
            assert_eq!(*r.denom(), 2);
        }
    }

    #[test]
    fn test_series_evaluate() {
        let x = Variable::new("x");
        let series = exp_series(&x, 10);

        // e^0.1 ≈ 1.10517...
        let result = series.evaluate(0.1).unwrap();
        let expected = 0.1_f64.exp();
        assert!((result - expected).abs() < 1e-8);
    }

    #[test]
    fn test_sin_series_evaluate() {
        let x = Variable::new("x");
        let series = sin_series(&x, 11);

        // sin(0.5) ≈ 0.4794...
        let result = series.evaluate(0.5).unwrap();
        let expected = 0.5_f64.sin();
        assert!((result - expected).abs() < 1e-8);
    }

    #[test]
    fn test_series_to_expression() {
        let x = Variable::new("x");
        let series = exp_series(&x, 3);
        let expr = series.to_expression();

        // Should be able to evaluate the expression
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 0.1);
        let result = expr.evaluate(&vars).unwrap();
        let expected = 1.0 + 0.1 + 0.01/2.0 + 0.001/6.0;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_series_to_latex() {
        let x = Variable::new("x");
        let series = sin_series(&x, 5);
        let latex = series.to_latex();

        // Should contain x and x^3 terms
        assert!(latex.contains("x"));
        assert!(latex.contains("x^{3}") || latex.contains("x³"));
    }

    #[test]
    fn test_ln_1_plus_x_series() {
        let x = Variable::new("x");
        let series = ln_1_plus_x_series(&x, 5);

        // Should have terms 1, 2, 3, 4, 5 (no constant term)
        assert!(series.get_term(0).is_none());
        assert!(series.get_term(1).is_some());

        // x coefficient should be 1
        let term1 = series.get_term(1).unwrap();
        if let Expression::Rational(r) = &term1.coefficient {
            assert_eq!(*r.numer(), 1);
            assert_eq!(*r.denom(), 1);
        }

        // x^2 coefficient should be -1/2
        let term2 = series.get_term(2).unwrap();
        if let Expression::Rational(r) = &term2.coefficient {
            assert_eq!(*r.numer(), -1);
            assert_eq!(*r.denom(), 2);
        }
    }

    #[test]
    fn test_arctan_series() {
        let x = Variable::new("x");
        let series = arctan_series(&x, 11);

        // arctan(0.3) ≈ 0.2915...
        let result = series.evaluate(0.3).unwrap();
        let expected = 0.3_f64.atan();
        assert!((result - expected).abs() < 1e-4);
    }

    #[test]
    fn test_binomial_series() {
        let x = Variable::new("x");

        // (1+x)^2 should give 1 + 2x + x^2 exactly
        let series = binomial_series(&Expression::Integer(2), &x, 5).unwrap();
        assert_eq!(series.term_count(), 3);

        // (1+x)^0.5 should give sqrt(1+x) approximation
        let series = binomial_series(&Expression::Float(0.5), &x, 5).unwrap();
        let result = series.evaluate(0.21).unwrap();  // sqrt(1.21) = 1.1
        assert!((result - 1.1).abs() < 0.01);
    }

    #[test]
    fn test_taylor_polynomial() {
        let x = Variable::new("x");
        // x^2 Taylor series around 0 should just be x^2
        let expr = Expression::Power(
            Box::new(Expression::Variable(x.clone())),
            Box::new(Expression::Integer(2)),
        );

        let series = taylor(&expr, &x, &Expression::Integer(0), 3).unwrap();

        // Should have just the x^2 term
        let term2 = series.get_term(2);
        assert!(term2.is_some());
    }

    #[test]
    fn test_maclaurin_exp() {
        let x = Variable::new("x");
        let expr = Expression::Function(Function::Exp, vec![Expression::Variable(x.clone())]);

        let series = maclaurin(&expr, &x, 5).unwrap();

        // Should match exp_series
        let expected = exp_series(&x, 5);
        assert_eq!(series.term_count(), expected.term_count());
    }
}
