//! Taylor and Maclaurin series expansion module.
//!
//! This module provides symbolic Taylor and Maclaurin series expansion capabilities
//! for mathematical expressions. It supports:
//!
//! - Taylor series expansion around arbitrary center points
//! - Maclaurin series (Taylor series centered at 0)
//! - Built-in series for common functions (exp, sin, cos, ln, arctan)
//! - Polynomial output and LaTeX generation
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

pub mod asymptotic;
pub mod composition;
pub mod known_series;
pub mod laurent;
pub mod taylor;

// Re-export all public items for backward compatibility
pub use asymptotic::{
    asymptotic, limit_via_asymptotic, AsymptoticDirection, AsymptoticSeries, AsymptoticTerm, BigO,
};
pub use composition::{compose_series, reversion};
pub use known_series::{
    arctan_series, binomial_series, cos_series, exp_series, ln_1_plus_x_series, sin_series,
};
pub use laurent::{
    find_singularities, laurent, pole_order, residue, LaurentSeries, Singularity, SingularityType,
};
pub use taylor::{maclaurin, taylor};

use crate::ast::{BinaryOp, Expression, Variable};
use std::collections::HashMap;
use std::fmt;

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

/// Error types for series expansion operations.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
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

    /// Get coefficient for a given power as f64, or 0 if not present.
    pub(crate) fn coeff_f64(&self, power: u32) -> f64 {
        self.get_term(power)
            .and_then(|t| try_expr_to_f64(&t.coefficient))
            .unwrap_or(0.0)
    }

    /// Compute the reciprocal of this series (1/S).
    /// Requires a_0 ≠ 0.
    pub fn reciprocal(&self) -> SeriesResult<Series> {
        // Get a_0
        let a0 = self.coeff_f64(0);
        if a0.abs() < 1e-15 {
            return Err(SeriesError::CannotExpand(
                "Cannot compute reciprocal: constant term is zero".into(),
            ));
        }

        let mut result = Series::new(self.variable.clone(), self.center.clone(), self.order);

        // b_0 = 1/a_0
        result.add_term(SeriesTerm::new(Expression::Float(1.0 / a0), 0));

        // b_n = -(1/a_0) * sum_{k=1}^n a_k * b_{n-k}
        for n in 1..=self.order {
            let mut sum = 0.0;
            for k in 1..=n {
                let a_k = self.coeff_f64(k);
                let b_n_k = result.coeff_f64(n - k);
                sum += a_k * b_n_k;
            }
            let b_n = -sum / a0;
            if b_n.abs() > 1e-15 {
                result.add_term(SeriesTerm::new(Expression::Float(b_n), n));
            }
        }

        Ok(result)
    }

    /// Term-by-term differentiation of the series.
    /// d/dx[sum a_n * (x-c)^n] = sum n * a_n * (x-c)^{n-1}
    pub fn differentiate(&self) -> Series {
        let new_order = if self.order > 0 { self.order - 1 } else { 0 };
        let mut result = Series::new(self.variable.clone(), self.center.clone(), new_order);

        for term in &self.terms {
            if term.power > 0 {
                // n * a_n -> coefficient of (x-c)^{n-1}
                let new_coeff = Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(Expression::Integer(term.power as i64)),
                    Box::new(term.coefficient.clone()),
                )
                .simplify();
                result.add_term(SeriesTerm::new(new_coeff, term.power - 1));
            }
        }

        result
    }

    /// Term-by-term integration of the series.
    /// integral[sum a_n * (x-c)^n] = C + sum a_n * (x-c)^{n+1} / (n+1)
    pub fn integrate(&self, constant: Expression) -> Series {
        let mut result = Series::new(self.variable.clone(), self.center.clone(), self.order + 1);

        // Add the integration constant as the x^0 term
        result.add_term(SeriesTerm::new(constant, 0));

        for term in &self.terms {
            // a_n / (n+1) -> coefficient of (x-c)^{n+1}
            let new_coeff = Expression::Binary(
                BinaryOp::Div,
                Box::new(term.coefficient.clone()),
                Box::new(Expression::Integer((term.power + 1) as i64)),
            )
            .simplify();
            result.add_term(SeriesTerm::new(new_coeff, term.power + 1));
        }

        result
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

// Helper functions shared across submodules

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Function;

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
        let expected = 1.0 + 0.1 + 0.01 / 2.0 + 0.001 / 6.0;
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
        let result = series.evaluate(0.21).unwrap(); // sqrt(1.21) = 1.1
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

    // Laurent Series Tests

    #[test]
    fn test_singularity_type_display() {
        let pole = SingularityType::Pole(2);
        assert_eq!(format!("{}", pole), "pole of order 2");

        let removable = SingularityType::Removable;
        assert_eq!(format!("{}", removable), "removable singularity");

        let essential = SingularityType::Essential;
        assert_eq!(format!("{}", essential), "essential singularity");
    }

    #[test]
    fn test_laurent_series_creation() {
        let z = Variable::new("z");
        let center = Expression::Integer(0);

        // Create a simple Laurent series with positive and negative terms
        let mut laurent = LaurentSeries::new(z.clone(), center, 1, 1);

        // Add positive terms
        laurent.add_positive_term(SeriesTerm::new(Expression::Integer(1), 0));
        laurent.add_positive_term(SeriesTerm::new(Expression::Integer(2), 1));

        // Add negative term (1/z)
        laurent.add_negative_term(Expression::Integer(3), 1);

        assert_eq!(laurent.principal_part_order, 1);
        assert_eq!(laurent.analytic_part_order, 1);
        assert_eq!(laurent.positive_terms.len(), 2);
        assert_eq!(laurent.negative_terms.len(), 1);
    }

    #[test]
    fn test_laurent_series_residue() {
        let z = Variable::new("z");

        // Laurent series: 1/z + 2 + 3z (residue should be 1)
        let mut laurent = LaurentSeries::new(z.clone(), Expression::Integer(0), 1, 1);
        laurent.add_positive_term(SeriesTerm::new(Expression::Integer(2), 0));
        laurent.add_positive_term(SeriesTerm::new(Expression::Integer(3), 1));
        laurent.add_negative_term(Expression::Integer(1), 1);

        let res = laurent.residue();
        if let Expression::Integer(n) = res {
            assert_eq!(n, 1);
        } else {
            panic!("Expected integer residue, got {:?}", res);
        }
    }

    #[test]
    fn test_laurent_series_pole_order() {
        let z = Variable::new("z");

        // Laurent series: 1/z^3 + 2/z + 1 (pole of order 3)
        let mut laurent = LaurentSeries::new(z.clone(), Expression::Integer(0), 3, 0);
        laurent.add_positive_term(SeriesTerm::new(Expression::Integer(1), 0));
        laurent.add_negative_term(Expression::Integer(1), 3);
        laurent.add_negative_term(Expression::Integer(2), 1);

        // The principal_part_order stores the pole order
        assert_eq!(laurent.principal_part_order, 3);
    }

    #[test]
    fn test_laurent_series_principal_part() {
        let z = Variable::new("z");

        // Laurent series: 2/z^2 + 3/z + 1 + z
        let mut laurent = LaurentSeries::new(z.clone(), Expression::Integer(0), 2, 1);
        laurent.add_positive_term(SeriesTerm::new(Expression::Integer(1), 0));
        laurent.add_positive_term(SeriesTerm::new(Expression::Integer(1), 1));
        laurent.add_negative_term(Expression::Integer(2), 2);
        laurent.add_negative_term(Expression::Integer(3), 1);

        let principal = laurent.principal_part();
        assert_eq!(principal.negative_terms.len(), 2);
    }

    #[test]
    fn test_laurent_series_analytic_part() {
        let z = Variable::new("z");

        // Laurent series: 1/z + 2 + 3z + 4z^2
        let mut laurent = LaurentSeries::new(z.clone(), Expression::Integer(0), 1, 2);
        laurent.add_positive_term(SeriesTerm::new(Expression::Integer(2), 0));
        laurent.add_positive_term(SeriesTerm::new(Expression::Integer(3), 1));
        laurent.add_positive_term(SeriesTerm::new(Expression::Integer(4), 2));
        laurent.add_negative_term(Expression::Integer(1), 1);

        let analytic = laurent.analytic_part();
        assert_eq!(analytic.term_count(), 3);
    }

    #[test]
    fn test_laurent_series_evaluate() {
        let z = Variable::new("z");

        // Laurent series: 1/z + 1 + z centered at 0
        // At z = 2: 1/2 + 1 + 2 = 3.5
        let mut laurent = LaurentSeries::new(z.clone(), Expression::Integer(0), 1, 1);
        laurent.add_positive_term(SeriesTerm::new(Expression::Integer(1), 0));
        laurent.add_positive_term(SeriesTerm::new(Expression::Integer(1), 1));
        laurent.add_negative_term(Expression::Integer(1), 1);

        let result = laurent.evaluate(2.0);
        assert!(result.is_some());
        assert!((result.unwrap() - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_laurent_series_evaluate_at_singularity() {
        let z = Variable::new("z");

        // Laurent series: 1/z centered at 0
        let mut laurent = LaurentSeries::new(z.clone(), Expression::Integer(0), 1, 0);
        laurent.add_negative_term(Expression::Integer(1), 1);

        // Should return None at the singularity
        let result = laurent.evaluate(0.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_laurent_series_to_latex() {
        let z = Variable::new("z");

        // Laurent series: 2/z + 1 + 3z centered at 0
        let mut laurent = LaurentSeries::new(z.clone(), Expression::Integer(0), 1, 1);
        laurent.add_positive_term(SeriesTerm::new(Expression::Integer(1), 0));
        laurent.add_positive_term(SeriesTerm::new(Expression::Integer(3), 1));
        laurent.add_negative_term(Expression::Integer(2), 1);

        let latex = laurent.to_latex();
        // Should contain z^{-1} for negative power
        assert!(latex.contains("z^{-1}"));
    }

    #[test]
    fn test_laurent_is_taylor() {
        let z = Variable::new("z");

        // A Taylor series has no negative powers
        let mut taylor_like = LaurentSeries::new(z.clone(), Expression::Integer(0), 0, 2);
        taylor_like.add_positive_term(SeriesTerm::new(Expression::Integer(1), 0));
        taylor_like.add_positive_term(SeriesTerm::new(Expression::Integer(2), 1));

        assert!(taylor_like.is_taylor());

        // A true Laurent series has negative powers
        let mut laurent = LaurentSeries::new(z.clone(), Expression::Integer(0), 1, 2);
        laurent.add_positive_term(SeriesTerm::new(Expression::Integer(1), 0));
        laurent.add_negative_term(Expression::Integer(1), 1);

        assert!(!laurent.is_taylor());
    }

    #[test]
    fn test_singularity_creation() {
        let location = Expression::Integer(0);
        let singularity = Singularity {
            location: location.clone(),
            singularity_type: SingularityType::Pole(2),
        };

        assert!(matches!(
            singularity.singularity_type,
            SingularityType::Pole(2)
        ));
    }

    #[test]
    fn test_find_singularities_simple_pole() {
        let z = Variable::new("z");

        // f(z) = 1/z has a simple pole at z = 0
        let expr = Expression::Binary(
            BinaryOp::Div,
            Box::new(Expression::Integer(1)),
            Box::new(Expression::Variable(z.clone())),
        );

        let singularities = find_singularities(&expr, &z);

        // Should find at least one singularity at z = 0
        assert!(!singularities.is_empty());
    }

    #[test]
    fn test_pole_order_simple() {
        let z = Variable::new("z");

        // 1/z has a simple pole (order 1)
        let expr = Expression::Binary(
            BinaryOp::Div,
            Box::new(Expression::Integer(1)),
            Box::new(Expression::Variable(z.clone())),
        );

        let order = pole_order(&expr, &z, &Expression::Integer(0));
        assert!(order.is_ok());
        assert_eq!(order.unwrap(), 1);
    }

    #[test]
    fn test_pole_order_double() {
        let z = Variable::new("z");

        // 1/z^2 has a double pole (order 2)
        let z_squared = Expression::Power(
            Box::new(Expression::Variable(z.clone())),
            Box::new(Expression::Integer(2)),
        );
        let expr = Expression::Binary(
            BinaryOp::Div,
            Box::new(Expression::Integer(1)),
            Box::new(z_squared),
        );

        let order = pole_order(&expr, &z, &Expression::Integer(0));
        assert!(order.is_ok());
        assert_eq!(order.unwrap(), 2);
    }

    #[test]
    fn test_residue_simple_pole() {
        let z = Variable::new("z");

        // f(z) = 1/z has residue 1 at z = 0
        let expr = Expression::Binary(
            BinaryOp::Div,
            Box::new(Expression::Integer(1)),
            Box::new(Expression::Variable(z.clone())),
        );

        let res = residue(&expr, &z, &Expression::Integer(0));
        assert!(res.is_ok());
        // Simplified residue should be 1
        let res_val = res.unwrap().simplify();
        if let Expression::Integer(n) = res_val {
            assert_eq!(n, 1);
        } else if let Expression::Float(f) = res_val {
            assert!((f - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_laurent_function_simple() {
        let z = Variable::new("z");

        // 1/z Laurent expansion around z=0 should give the series 1/z
        let expr = Expression::Binary(
            BinaryOp::Div,
            Box::new(Expression::Integer(1)),
            Box::new(Expression::Variable(z.clone())),
        );

        // Request expansion with neg_order=1 for a simple pole
        let result = laurent(&expr, &z, &Expression::Integer(0), 1, 3);
        assert!(result.is_ok());

        let laurent_series = result.unwrap();
        // Should have one negative term (1/z)
        assert!(!laurent_series.negative_terms.is_empty());
        // principal_part_order matches requested neg_order
        assert_eq!(laurent_series.principal_part_order, 1);
    }

    #[test]
    fn test_laurent_to_taylor() {
        let z = Variable::new("z");

        // A Laurent series without negative powers can convert to Taylor
        let mut laurent = LaurentSeries::new(z.clone(), Expression::Integer(0), 0, 2);
        laurent.add_positive_term(SeriesTerm::new(Expression::Integer(1), 0));
        laurent.add_positive_term(SeriesTerm::new(Expression::Integer(2), 1));

        let taylor_opt = laurent.to_taylor();
        assert!(taylor_opt.is_some());

        let taylor = taylor_opt.unwrap();
        assert_eq!(taylor.term_count(), 2);
    }

    // Series Arithmetic Tests

    #[test]
    fn test_series_addition() {
        let x = Variable::new("x");

        // Series 1: 1 + x + x^2
        let mut s1 = Series::new(x.clone(), Expression::Integer(0), 3);
        s1.add_term(SeriesTerm::new(Expression::Integer(1), 0));
        s1.add_term(SeriesTerm::new(Expression::Integer(1), 1));
        s1.add_term(SeriesTerm::new(Expression::Integer(1), 2));

        // Series 2: 2 + 3x + x^2
        let mut s2 = Series::new(x.clone(), Expression::Integer(0), 3);
        s2.add_term(SeriesTerm::new(Expression::Integer(2), 0));
        s2.add_term(SeriesTerm::new(Expression::Integer(3), 1));
        s2.add_term(SeriesTerm::new(Expression::Integer(1), 2));

        // Sum should be: 3 + 4x + 2x^2
        let sum = (s1 + s2).unwrap();

        // Evaluate at x = 0.5: 3 + 4*0.5 + 2*0.25 = 3 + 2 + 0.5 = 5.5
        let result = sum.evaluate(0.5).unwrap();
        assert!((result - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_series_subtraction() {
        let x = Variable::new("x");

        // Series 1: 3 + 2x + x^2
        let mut s1 = Series::new(x.clone(), Expression::Integer(0), 2);
        s1.add_term(SeriesTerm::new(Expression::Integer(3), 0));
        s1.add_term(SeriesTerm::new(Expression::Integer(2), 1));
        s1.add_term(SeriesTerm::new(Expression::Integer(1), 2));

        // Series 2: 1 + x + x^2
        let mut s2 = Series::new(x.clone(), Expression::Integer(0), 2);
        s2.add_term(SeriesTerm::new(Expression::Integer(1), 0));
        s2.add_term(SeriesTerm::new(Expression::Integer(1), 1));
        s2.add_term(SeriesTerm::new(Expression::Integer(1), 2));

        // Difference should be: 2 + x
        let diff = (s1 - s2).unwrap();

        // Evaluate at x = 1: 2 + 1 = 3
        let result = diff.evaluate(1.0).unwrap();
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_series_multiplication() {
        let x = Variable::new("x");

        // (1 + x) * (1 - x) = 1 - x^2
        let mut s1 = Series::new(x.clone(), Expression::Integer(0), 4);
        s1.add_term(SeriesTerm::new(Expression::Integer(1), 0));
        s1.add_term(SeriesTerm::new(Expression::Integer(1), 1));

        let mut s2 = Series::new(x.clone(), Expression::Integer(0), 4);
        s2.add_term(SeriesTerm::new(Expression::Integer(1), 0));
        s2.add_term(SeriesTerm::new(Expression::Integer(-1), 1));

        let product = (s1 * s2).unwrap();

        // Evaluate at x = 0.5: 1 - 0.25 = 0.75
        let result = product.evaluate(0.5).unwrap();
        assert!((result - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_series_reciprocal() {
        let x = Variable::new("x");

        // 1/(1-x) = 1 + x + x^2 + x^3 + ... (geometric series)
        let mut s = Series::new(x.clone(), Expression::Integer(0), 5);
        s.add_term(SeriesTerm::new(Expression::Integer(1), 0));
        s.add_term(SeriesTerm::new(Expression::Integer(-1), 1));

        let recip = s.reciprocal().unwrap();

        // Evaluate at x = 0.5: 1/(1-0.5) = 2
        // Series approximation: 1 + 0.5 + 0.25 + 0.125 + 0.0625 + 0.03125 ≈ 1.96875
        let result = recip.evaluate(0.5).unwrap();
        assert!((result - 1.96875).abs() < 1e-10);
    }

    #[test]
    fn test_series_division() {
        let x = Variable::new("x");

        // (1 - x^2) / (1 - x) = 1 + x
        let mut num = Series::new(x.clone(), Expression::Integer(0), 4);
        num.add_term(SeriesTerm::new(Expression::Integer(1), 0));
        num.add_term(SeriesTerm::new(Expression::Integer(-1), 2));

        let mut denom = Series::new(x.clone(), Expression::Integer(0), 4);
        denom.add_term(SeriesTerm::new(Expression::Integer(1), 0));
        denom.add_term(SeriesTerm::new(Expression::Integer(-1), 1));

        let quotient = (num / denom).unwrap();

        // Evaluate at x = 0.5: 1 + 0.5 = 1.5
        let result = quotient.evaluate(0.5).unwrap();
        assert!((result - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_series_differentiate() {
        let x = Variable::new("x");

        // d/dx[1 + x + x^2/2 + x^3/6] = 1 + x + x^2/2 (e^x derivative is e^x)
        let exp_s = exp_series(&x, 4);
        let deriv = exp_s.differentiate();

        // d/dx[e^x] at x=0.1 should ≈ e^{0.1} ≈ 1.10517
        let result = deriv.evaluate(0.1).unwrap();
        let expected = 0.1_f64.exp();
        assert!((result - expected).abs() < 0.01);
    }

    #[test]
    fn test_series_integrate() {
        let x = Variable::new("x");

        // integral[1 + x + x^2/2!] dx with C=0 should give x + x^2/2 + x^3/6
        // This is integrating the first few terms of e^x
        let exp_s = exp_series(&x, 3);
        let integrated = exp_s.integrate(Expression::Integer(0));

        // At x=0.1: 0.1 + 0.01/2 + 0.001/6 ≈ 0.10517
        let result = integrated.evaluate(0.1).unwrap();
        let expected = 0.1_f64.exp() - 1.0; // integral of e^x from 0 to 0.1 is e^0.1 - 1
        assert!((result - expected).abs() < 0.01);
    }

    #[test]
    fn test_differentiate_then_integrate() {
        let x = Variable::new("x");

        // Differentiate sin(x), then integrate should give back sin(x) (up to constant)
        let sin_s = sin_series(&x, 7);
        let deriv = sin_s.differentiate(); // Should be cos(x) series
        let integrated = deriv.integrate(Expression::Integer(0)); // Back to sin(x)

        // At x=0.5: sin(0.5) ≈ 0.4794
        let result = integrated.evaluate(0.5).unwrap();
        let expected = 0.5_f64.sin();
        assert!((result - expected).abs() < 0.01);
    }

    #[test]
    fn test_exp_times_neg_exp() {
        let x = Variable::new("x");

        // e^x * e^{-x} = 1 (should cancel to 1)
        let exp_pos = exp_series(&x, 5);

        // Build e^{-x} = 1 - x + x^2/2 - x^3/6 + ...
        let mut exp_neg = Series::new(x.clone(), Expression::Integer(0), 5);
        for n in 0..=5 {
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            let coeff = sign / factorial(n) as f64;
            exp_neg.add_term(SeriesTerm::new(Expression::Float(coeff), n));
        }

        let product = (exp_pos * exp_neg).unwrap();

        // At any x, e^x * e^{-x} = 1
        let result = product.evaluate(0.5).unwrap();
        assert!((result - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compose_series_exp_sin() {
        let x = Variable::new("x");

        // e^{sin(x)} composition
        let exp_s = exp_series(&x, 5);
        let sin_s = sin_series(&x, 5);

        // sin(x) has no constant term, so composition is valid
        let composed = compose_series(&exp_s, &sin_s).unwrap();

        // e^{sin(0.1)} = e^{0.0998...} ≈ 1.1049
        let result = composed.evaluate(0.1).unwrap();
        let expected = (0.1_f64.sin()).exp();
        assert!((result - expected).abs() < 0.01);
    }

    #[test]
    fn test_series_reversion() {
        let x = Variable::new("x");

        // sin(x) = x - x^3/6 + ...
        // arcsin(x) is the reversion of sin(x)
        let sin_s = sin_series(&x, 7);
        let arcsin_s = reversion(&sin_s).unwrap();

        // arcsin(0.5) ≈ 0.5236 (π/6)
        let result = arcsin_s.evaluate(0.5).unwrap();
        let expected = 0.5_f64.asin();
        assert!((result - expected).abs() < 0.05);
    }

    // Asymptotic Expansion Tests

    #[test]
    fn test_asymptotic_direction_display() {
        assert_eq!(format!("{}", AsymptoticDirection::PosInfinity), "x→+∞");
        assert_eq!(format!("{}", AsymptoticDirection::NegInfinity), "x→-∞");
        assert_eq!(format!("{}", AsymptoticDirection::Zero), "x→0");
    }

    #[test]
    fn test_asymptotic_term_creation() {
        let term = AsymptoticTerm::new(Expression::Integer(2), Expression::Integer(-1));
        assert_eq!(term.coefficient, Expression::Integer(2));
        assert_eq!(term.exponent, Expression::Integer(-1));
        assert!(!term.is_zero());

        let zero_term = AsymptoticTerm::new(Expression::Integer(0), Expression::Integer(1));
        assert!(zero_term.is_zero());
    }

    #[test]
    fn test_asymptotic_term_evaluate() {
        let x = Variable::new("x");

        // 2*x^(-1) at x=4 should be 2/4 = 0.5
        let term = AsymptoticTerm::new(Expression::Integer(2), Expression::Integer(-1));
        let result = term.evaluate(&x, 4.0).unwrap();
        assert!((result - 0.5).abs() < 1e-10);

        // 3*x^2 at x=2 should be 3*4 = 12
        let term2 = AsymptoticTerm::new(Expression::Integer(3), Expression::Integer(2));
        let result2 = term2.evaluate(&x, 2.0).unwrap();
        assert!((result2 - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_asymptotic_term_display() {
        let term1 = AsymptoticTerm::new(Expression::Integer(2), Expression::Integer(0));
        assert_eq!(format!("{}", term1), "2");

        let term2 = AsymptoticTerm::new(Expression::Integer(3), Expression::Integer(1));
        assert_eq!(format!("{}", term2), "3·x");

        let term3 = AsymptoticTerm::new(Expression::Integer(1), Expression::Integer(-2));
        assert_eq!(format!("{}", term3), "1/x^2");
    }

    #[test]
    fn test_big_o_creation() {
        let x = Variable::new("x");
        let order = Expression::Power(
            Box::new(Expression::Variable(x.clone())),
            Box::new(Expression::Integer(2)),
        );
        let big_o = BigO::new(order.clone(), x.clone());

        assert_eq!(big_o.order, order);
        assert_eq!(big_o.variable, x);
    }

    #[test]
    fn test_big_o_is_same_order() {
        let x = Variable::new("x");
        let order1 = Expression::Power(
            Box::new(Expression::Variable(x.clone())),
            Box::new(Expression::Integer(2)),
        );
        let order2 = Expression::Power(
            Box::new(Expression::Variable(x.clone())),
            Box::new(Expression::Integer(2)),
        );

        let big_o1 = BigO::new(order1, x.clone());
        let big_o2 = BigO::new(order2, x.clone());

        assert!(big_o1.is_same_order(&big_o2));
    }

    #[test]
    fn test_big_o_display() {
        let x = Variable::new("x");
        let order = Expression::Power(
            Box::new(Expression::Variable(x.clone())),
            Box::new(Expression::Integer(3)),
        );
        let big_o = BigO::new(order, x);

        assert!(format!("{}", big_o).contains("O("));
    }

    #[test]
    fn test_asymptotic_series_creation() {
        let x = Variable::new("x");
        let series = AsymptoticSeries::new(x.clone(), AsymptoticDirection::PosInfinity);

        assert_eq!(series.variable, x);
        assert_eq!(series.direction, AsymptoticDirection::PosInfinity);
        assert_eq!(series.terms.len(), 0);
    }

    #[test]
    fn test_asymptotic_series_add_term() {
        let x = Variable::new("x");
        let mut series = AsymptoticSeries::new(x.clone(), AsymptoticDirection::PosInfinity);

        series.add_term(AsymptoticTerm::new(
            Expression::Integer(1),
            Expression::Integer(-1),
        ));
        series.add_term(AsymptoticTerm::new(
            Expression::Integer(1),
            Expression::Integer(-2),
        ));

        assert_eq!(series.terms.len(), 2);
    }

    #[test]
    fn test_asymptotic_series_dominant_term() {
        let x = Variable::new("x");
        let mut series = AsymptoticSeries::new(x.clone(), AsymptoticDirection::PosInfinity);

        series.add_term(AsymptoticTerm::new(
            Expression::Integer(1),
            Expression::Integer(-1),
        ));
        series.add_term(AsymptoticTerm::new(
            Expression::Integer(1),
            Expression::Integer(-2),
        ));

        let dominant = series.dominant_term().unwrap();
        assert_eq!(dominant.exponent, Expression::Integer(-1));
    }

    #[test]
    fn test_asymptotic_series_order_of_magnitude() {
        let x = Variable::new("x");
        let mut series = AsymptoticSeries::new(x.clone(), AsymptoticDirection::PosInfinity);

        series.add_term(AsymptoticTerm::new(
            Expression::Integer(2),
            Expression::Integer(-1),
        ));

        let order = series.order_of_magnitude().unwrap();
        assert_eq!(order, Expression::Integer(-1));
    }

    #[test]
    fn test_asymptotic_series_with_error_term() {
        let x = Variable::new("x");
        let mut series = AsymptoticSeries::new(x.clone(), AsymptoticDirection::PosInfinity);

        series.add_term(AsymptoticTerm::new(
            Expression::Integer(1),
            Expression::Integer(-1),
        ));
        series.add_term(AsymptoticTerm::new(
            Expression::Integer(1),
            Expression::Integer(-2),
        ));

        let (_, big_o) = series.with_error_term();
        // Error term should be O(x^(-3)) for x→∞ when last term is x^(-2)
        assert_eq!(big_o.variable, x);
    }

    #[test]
    fn test_asymptotic_series_evaluate() {
        let x = Variable::new("x");
        let mut series = AsymptoticSeries::new(x.clone(), AsymptoticDirection::PosInfinity);

        // 1/x + 1/x^2 at x=2 should be 0.5 + 0.25 = 0.75
        series.add_term(AsymptoticTerm::new(
            Expression::Integer(1),
            Expression::Integer(-1),
        ));
        series.add_term(AsymptoticTerm::new(
            Expression::Integer(1),
            Expression::Integer(-2),
        ));

        let result = series.evaluate(2.0).unwrap();
        assert!((result - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_asymptotic_1_over_x() {
        use crate::parser::parse_expression;

        // 1/x as x→∞ should give [1/x]
        let expr = parse_expression("1/x").unwrap();
        let series = asymptotic(&expr, "x", AsymptoticDirection::PosInfinity, 3).unwrap();

        assert_eq!(series.terms.len(), 1);
        assert_eq!(series.terms[0].coefficient, Expression::Integer(1));
        assert_eq!(series.terms[0].exponent, Expression::Integer(-1));
    }

    #[test]
    fn test_asymptotic_1_over_x_plus_1_over_x2() {
        use crate::parser::parse_expression;

        // 1/x + 1/x^2 as x→∞
        let expr = parse_expression("1/x + 1/x^2").unwrap();
        let series = asymptotic(&expr, "x", AsymptoticDirection::PosInfinity, 3).unwrap();

        assert_eq!(series.terms.len(), 2);
        // Dominant term should be 1/x (exponent -1)
        assert_eq!(series.terms[0].exponent, Expression::Integer(-1));

        // Next term should be 1/x^2 (exponent -2)
        // The exponent might be Unary(Neg, Float(2.0)) or Integer(-2) depending on simplification
        let exp1 = &series.terms[1].exponent;
        let exp1_val = try_expr_to_f64(exp1).unwrap();
        assert!((exp1_val - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_asymptotic_x_squared_plus_x() {
        use crate::parser::parse_expression;

        // x^2 + x as x→∞, dominant term is x^2
        let expr = parse_expression("x^2 + x").unwrap();
        let series = asymptotic(&expr, "x", AsymptoticDirection::PosInfinity, 3).unwrap();

        let dominant = series.dominant_term().unwrap();
        // Exponent might be Integer(2) or Float(2.0) depending on parser
        let exp_val = try_expr_to_f64(&dominant.exponent).unwrap();
        assert!((exp_val - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_asymptotic_evaluate_at_point() {
        use crate::parser::parse_expression;

        // 1/x + 1/x^2 at x=10 should be 0.1 + 0.01 = 0.11
        let expr = parse_expression("1/x + 1/x^2").unwrap();
        let series = asymptotic(&expr, "x", AsymptoticDirection::PosInfinity, 3).unwrap();

        let result = series.evaluate(10.0).unwrap();
        assert!((result - 0.11).abs() < 1e-10);
    }

    #[test]
    fn test_sort_by_dominance_pos_infinity() {
        use super::asymptotic::sort_by_dominance;

        let _x = Variable::new("x");
        let mut terms = vec![
            AsymptoticTerm::new(Expression::Integer(1), Expression::Integer(-2)),
            AsymptoticTerm::new(Expression::Integer(1), Expression::Integer(-1)),
            AsymptoticTerm::new(Expression::Integer(1), Expression::Integer(0)),
        ];

        sort_by_dominance(&mut terms, AsymptoticDirection::PosInfinity);

        // For x→∞, constant (0) > 1/x (-1) > 1/x^2 (-2)
        assert_eq!(terms[0].exponent, Expression::Integer(0));
        assert_eq!(terms[1].exponent, Expression::Integer(-1));
        assert_eq!(terms[2].exponent, Expression::Integer(-2));
    }

    #[test]
    fn test_sort_by_dominance_zero() {
        use super::asymptotic::sort_by_dominance;

        let _x = Variable::new("x");
        let mut terms = vec![
            AsymptoticTerm::new(Expression::Integer(1), Expression::Integer(2)),
            AsymptoticTerm::new(Expression::Integer(1), Expression::Integer(1)),
            AsymptoticTerm::new(Expression::Integer(1), Expression::Integer(0)),
        ];

        sort_by_dominance(&mut terms, AsymptoticDirection::Zero);

        // For x→0, constant (0) > x (1) > x^2 (2)
        assert_eq!(terms[0].exponent, Expression::Integer(0));
        assert_eq!(terms[1].exponent, Expression::Integer(1));
        assert_eq!(terms[2].exponent, Expression::Integer(2));
    }

    #[test]
    fn test_limit_via_asymptotic_to_zero() {
        use crate::limits::LimitResult;
        use crate::parser::parse_expression;

        // lim_{x→∞} 1/x = 0
        let expr = parse_expression("1/x").unwrap();
        let result = limit_via_asymptotic(&expr, "x", AsymptoticDirection::PosInfinity).unwrap();

        assert_eq!(result, LimitResult::Value(0.0));
    }

    #[test]
    fn test_limit_via_asymptotic_to_infinity() {
        use crate::limits::LimitResult;
        use crate::parser::parse_expression;

        // lim_{x→∞} x^2 = ∞
        let expr = parse_expression("x^2").unwrap();
        let result = limit_via_asymptotic(&expr, "x", AsymptoticDirection::PosInfinity).unwrap();

        assert_eq!(result, LimitResult::PositiveInfinity);
    }

    #[test]
    fn test_limit_via_asymptotic_constant() {
        use crate::limits::LimitResult;

        // lim_{x→∞} 5 = 5
        let expr = Expression::Integer(5);
        let result = limit_via_asymptotic(&expr, "x", AsymptoticDirection::PosInfinity).unwrap();

        assert_eq!(result, LimitResult::Value(5.0));
    }

    #[test]
    fn test_asymptotic_series_to_expression() {
        let x = Variable::new("x");
        let mut series = AsymptoticSeries::new(x.clone(), AsymptoticDirection::PosInfinity);

        series.add_term(AsymptoticTerm::new(
            Expression::Integer(1),
            Expression::Integer(-1),
        ));
        series.add_term(AsymptoticTerm::new(
            Expression::Integer(2),
            Expression::Integer(-2),
        ));

        let expr = series.to_expression();
        // Should be simplifiable to some form
        assert!(!matches!(expr, Expression::Integer(0)));
    }

    #[test]
    fn test_asymptotic_series_display() {
        let x = Variable::new("x");
        let mut series = AsymptoticSeries::new(x.clone(), AsymptoticDirection::PosInfinity);

        series.add_term(AsymptoticTerm::new(
            Expression::Integer(1),
            Expression::Integer(-1),
        ));

        let display_str = format!("{}", series);
        assert!(display_str.contains("x→+∞"));
    }
}
