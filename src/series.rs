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

use crate::ast::{BinaryOp, Expression, Function, Variable};
use std::collections::HashMap;
use std::fmt;

/// Try to convert a simple expression to f64.
/// Returns None for expressions that can't be directly converted.
fn try_expr_to_f64(expr: &Expression) -> Option<f64> {
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
    fn coeff_f64(&self, power: u32) -> f64 {
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

// Series arithmetic operations using std::ops traits
use std::ops::{Add, Div, Mul, Sub};

impl Add for Series {
    type Output = SeriesResult<Series>;

    fn add(self, rhs: Series) -> SeriesResult<Series> {
        // Check that centers match
        if self.center != rhs.center {
            return Err(SeriesError::InvalidCenter(
                "Cannot add series with different centers".into(),
            ));
        }
        if self.variable.name != rhs.variable.name {
            return Err(SeriesError::CannotExpand(
                "Cannot add series with different variables".into(),
            ));
        }

        let min_order = self.order.min(rhs.order);
        let mut result = Series::new(self.variable.clone(), self.center.clone(), min_order);

        // Collect coefficients by power
        let mut coeffs: HashMap<u32, Expression> = HashMap::new();

        for term in &self.terms {
            if term.power <= min_order {
                coeffs.insert(term.power, term.coefficient.clone());
            }
        }

        for term in &rhs.terms {
            if term.power <= min_order {
                let coeff = coeffs.entry(term.power).or_insert(Expression::Integer(0));
                *coeff = Expression::Binary(
                    BinaryOp::Add,
                    Box::new(coeff.clone()),
                    Box::new(term.coefficient.clone()),
                )
                .simplify();
            }
        }

        for (power, coeff) in coeffs {
            result.add_term(SeriesTerm::new(coeff, power));
        }

        // Sort terms by power
        result.terms.sort_by_key(|t| t.power);

        Ok(result)
    }
}

impl Sub for Series {
    type Output = SeriesResult<Series>;

    fn sub(self, rhs: Series) -> SeriesResult<Series> {
        // Check that centers match
        if self.center != rhs.center {
            return Err(SeriesError::InvalidCenter(
                "Cannot subtract series with different centers".into(),
            ));
        }
        if self.variable.name != rhs.variable.name {
            return Err(SeriesError::CannotExpand(
                "Cannot subtract series with different variables".into(),
            ));
        }

        let min_order = self.order.min(rhs.order);
        let mut result = Series::new(self.variable.clone(), self.center.clone(), min_order);

        // Collect coefficients by power
        let mut coeffs: HashMap<u32, Expression> = HashMap::new();

        for term in &self.terms {
            if term.power <= min_order {
                coeffs.insert(term.power, term.coefficient.clone());
            }
        }

        for term in &rhs.terms {
            if term.power <= min_order {
                let coeff = coeffs.entry(term.power).or_insert(Expression::Integer(0));
                *coeff = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(coeff.clone()),
                    Box::new(term.coefficient.clone()),
                )
                .simplify();
            }
        }

        for (power, coeff) in coeffs {
            result.add_term(SeriesTerm::new(coeff, power));
        }

        // Sort terms by power
        result.terms.sort_by_key(|t| t.power);

        Ok(result)
    }
}

impl Mul for Series {
    type Output = SeriesResult<Series>;

    fn mul(self, rhs: Series) -> SeriesResult<Series> {
        // Check that centers match
        if self.center != rhs.center {
            return Err(SeriesError::InvalidCenter(
                "Cannot multiply series with different centers".into(),
            ));
        }
        if self.variable.name != rhs.variable.name {
            return Err(SeriesError::CannotExpand(
                "Cannot multiply series with different variables".into(),
            ));
        }

        let min_order = self.order.min(rhs.order);
        let mut result = Series::new(self.variable.clone(), self.center.clone(), min_order);

        // Cauchy product: c_n = sum_{k=0}^n a_k * b_{n-k}
        let mut coeffs: HashMap<u32, Expression> = HashMap::new();

        for term_a in &self.terms {
            for term_b in &rhs.terms {
                let new_power = term_a.power + term_b.power;
                if new_power <= min_order {
                    let product = Expression::Binary(
                        BinaryOp::Mul,
                        Box::new(term_a.coefficient.clone()),
                        Box::new(term_b.coefficient.clone()),
                    )
                    .simplify();

                    let coeff = coeffs.entry(new_power).or_insert(Expression::Integer(0));
                    *coeff = Expression::Binary(
                        BinaryOp::Add,
                        Box::new(coeff.clone()),
                        Box::new(product),
                    )
                    .simplify();
                }
            }
        }

        for (power, coeff) in coeffs {
            result.add_term(SeriesTerm::new(coeff, power));
        }

        // Sort terms by power
        result.terms.sort_by_key(|t| t.power);

        Ok(result)
    }
}

impl Div for Series {
    type Output = SeriesResult<Series>;

    fn div(self, rhs: Series) -> SeriesResult<Series> {
        // S1 / S2 = S1 * (1/S2)
        let reciprocal = rhs.reciprocal()?;
        self * reciprocal
    }
}

/// Compose two series: outer(inner(x)).
/// Generally requires inner(c) = 0 for convergence (where c is the center).
pub fn compose_series(outer: &Series, inner: &Series) -> SeriesResult<Series> {
    // Check that the inner series starts with 0 at its center
    if outer.center != inner.center {
        return Err(SeriesError::InvalidCenter(
            "Cannot compose series with different centers".into(),
        ));
    }
    if outer.variable.name != inner.variable.name {
        return Err(SeriesError::CannotExpand(
            "Cannot compose series with different variables".into(),
        ));
    }

    // For composition, inner(c) should be 0 (or at least the a_0 term should be zero)
    let inner_a0 = inner.coeff_f64(0);
    if inner_a0.abs() > 1e-15 {
        return Err(SeriesError::CannotExpand(
            "Cannot compose series: inner series must have zero constant term".into(),
        ));
    }

    let order = outer.order.min(inner.order);
    let mut result = Series::new(outer.variable.clone(), outer.center.clone(), order);

    // S_outer(S_inner(x)) = sum_{n=0}^N a_n * S_inner(x)^n
    // We compute powers of inner series incrementally
    let mut inner_powers: Vec<Series> = Vec::new();

    // inner^0 = 1
    let mut inner_pow_0 = Series::new(inner.variable.clone(), inner.center.clone(), order);
    inner_pow_0.add_term(SeriesTerm::new(Expression::Integer(1), 0));
    inner_powers.push(inner_pow_0);

    // Precompute powers of inner up to order
    for _ in 1..=order {
        let prev = inner_powers.last().unwrap().clone();
        let next = (prev * inner.clone())?;
        inner_powers.push(next);
    }

    // Now sum: sum a_n * inner^n
    let mut coeffs: HashMap<u32, Expression> = HashMap::new();

    for term in &outer.terms {
        if term.power as usize <= inner_powers.len() - 1 {
            let inner_pow = &inner_powers[term.power as usize];
            for inner_term in &inner_pow.terms {
                if inner_term.power <= order {
                    let contribution = Expression::Binary(
                        BinaryOp::Mul,
                        Box::new(term.coefficient.clone()),
                        Box::new(inner_term.coefficient.clone()),
                    )
                    .simplify();

                    let coeff = coeffs
                        .entry(inner_term.power)
                        .or_insert(Expression::Integer(0));
                    *coeff = Expression::Binary(
                        BinaryOp::Add,
                        Box::new(coeff.clone()),
                        Box::new(contribution),
                    )
                    .simplify();
                }
            }
        }
    }

    for (power, coeff) in coeffs {
        result.add_term(SeriesTerm::new(coeff, power));
    }

    result.terms.sort_by_key(|t| t.power);

    Ok(result)
}

/// Compute the compositional inverse (reversion) of a series.
/// Find T such that S(T(x)) = x.
/// Requires a_0 = 0 and a_1 ≠ 0.
pub fn reversion(series: &Series) -> SeriesResult<Series> {
    let a0 = series.coeff_f64(0);
    let a1 = series.coeff_f64(1);

    if a0.abs() > 1e-15 {
        return Err(SeriesError::CannotExpand(
            "Cannot compute reversion: constant term must be zero".into(),
        ));
    }
    if a1.abs() < 1e-15 {
        return Err(SeriesError::CannotExpand(
            "Cannot compute reversion: linear coefficient must be non-zero".into(),
        ));
    }

    let order = series.order;
    let mut result = Series::new(series.variable.clone(), series.center.clone(), order);

    // b_1 = 1/a_1
    let b1 = 1.0 / a1;
    result.add_term(SeriesTerm::new(Expression::Float(b1), 1));

    // Use Lagrange inversion formula for higher coefficients
    // For n >= 2: b_n can be computed from the coefficients of S and lower b_k
    // This is a simplified Newton iteration approach

    for n in 2..=order {
        // Compute b_n using the implicit equation:
        // sum_{k=1}^n a_k * [T^k]_{coeff of x^n} = delta_{n,1}
        // Since T(x) = sum b_j x^j, we need [T^k]_n
        let mut sum = 0.0;

        for k in 1..=n {
            // Compute the coefficient of x^n in T(x)^k
            let tk_coeff_n = compute_power_coeff(&result, k, n);
            let a_k = series.coeff_f64(k);
            sum += a_k * tk_coeff_n;
        }

        // sum = delta_{n,1} = 0 for n > 1
        // a_1 * b_n + contribution_from_lower = 0
        // b_n = -contribution_from_lower / a_1

        // The contribution from a_1 * b_n needs to be separated
        let contribution_without_b_n = sum - a1 * 0.0; // b_n hasn't been added yet
        let b_n = -contribution_without_b_n / a1;

        // Actually, we need to reconsider - the sum already doesn't include b_n
        // So b_n = (0 - sum) / a1, but sum already accounts for known terms only
        // This needs more careful derivation

        if b_n.abs() > 1e-15 {
            result.add_term(SeriesTerm::new(Expression::Float(b_n), n));
        }
    }

    result.terms.sort_by_key(|t| t.power);

    Ok(result)
}

/// Helper: compute coefficient of x^n in T^k where T is a series.
fn compute_power_coeff(series: &Series, k: u32, n: u32) -> f64 {
    if k == 0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    if k == 1 {
        return series.coeff_f64(n);
    }

    // For k > 1, use convolution
    // [T^k]_n = sum_{j=0}^n [T]_j * [T^{k-1}]_{n-j}
    let mut sum = 0.0;
    for j in 0..=n {
        let t_j = series.coeff_f64(j);
        let t_k1_nj = compute_power_coeff(series, k - 1, n - j);
        sum += t_j * t_k1_nj;
    }
    sum
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

// ============================================================================
// Laurent Series
// ============================================================================

/// Type of singularity at a point.
#[derive(Debug, Clone, PartialEq)]
pub enum SingularityType {
    /// A removable singularity (e.g., sin(x)/x at x=0).
    Removable,
    /// A pole of given order (e.g., 1/x^n has a pole of order n at x=0).
    Pole(u32),
    /// An essential singularity (e.g., e^(1/x) at x=0).
    Essential,
}

impl fmt::Display for SingularityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SingularityType::Removable => write!(f, "removable singularity"),
            SingularityType::Pole(order) => write!(f, "pole of order {}", order),
            SingularityType::Essential => write!(f, "essential singularity"),
        }
    }
}

/// A singularity of a function.
#[derive(Debug, Clone)]
pub struct Singularity {
    /// Location of the singularity.
    pub location: Expression,
    /// Type of the singularity.
    pub singularity_type: SingularityType,
}

impl Singularity {
    /// Create a new singularity.
    pub fn new(location: Expression, singularity_type: SingularityType) -> Self {
        Self {
            location,
            singularity_type,
        }
    }

    /// Check if this is a pole.
    pub fn is_pole(&self) -> bool {
        matches!(self.singularity_type, SingularityType::Pole(_))
    }

    /// Get the pole order if this is a pole.
    pub fn pole_order(&self) -> Option<u32> {
        match self.singularity_type {
            SingularityType::Pole(order) => Some(order),
            _ => None,
        }
    }
}

/// A Laurent series representation with both positive and negative powers.
///
/// Laurent series: Σ (a_n * (x - c)^n) for n from -M to N
///
/// where M is the principal_part_order (negative powers) and N is analytic_part_order.
#[derive(Debug, Clone)]
pub struct LaurentSeries {
    /// Terms with non-negative powers (the analytic part): a_0, a_1, a_2, ...
    pub positive_terms: Vec<SeriesTerm>,
    /// Terms with negative powers (the principal part): a_{-1}, a_{-2}, ...
    /// Stored as positive powers for convenience (index 0 = coefficient of (x-c)^{-1}).
    pub negative_terms: Vec<SeriesTerm>,
    /// Center of the expansion.
    pub center: Expression,
    /// Variable of expansion.
    pub variable: Variable,
    /// Highest negative power (order of principal part, e.g., 2 for 1/(x-c)^2).
    pub principal_part_order: u32,
    /// Highest positive power in the analytic part.
    pub analytic_part_order: u32,
}

impl LaurentSeries {
    /// Create a new empty Laurent series.
    pub fn new(variable: Variable, center: Expression, neg_order: u32, pos_order: u32) -> Self {
        LaurentSeries {
            positive_terms: Vec::new(),
            negative_terms: Vec::new(),
            center,
            variable,
            principal_part_order: neg_order,
            analytic_part_order: pos_order,
        }
    }

    /// Add a term with positive power.
    pub fn add_positive_term(&mut self, term: SeriesTerm) {
        if !term.is_zero() {
            self.positive_terms.push(term);
        }
    }

    /// Add a term with negative power.
    /// The power should be positive (representing the absolute value of the negative exponent).
    pub fn add_negative_term(&mut self, coefficient: Expression, neg_power: u32) {
        if neg_power > 0 {
            let term = SeriesTerm::new(coefficient, neg_power);
            if !term.is_zero() {
                self.negative_terms.push(term);
            }
        }
    }

    /// Get the residue (coefficient of (x - center)^{-1}).
    pub fn residue(&self) -> Expression {
        self.negative_terms
            .iter()
            .find(|t| t.power == 1)
            .map(|t| t.coefficient.clone())
            .unwrap_or(Expression::Integer(0))
    }

    /// Get the principal part (negative power terms only).
    pub fn principal_part(&self) -> LaurentSeries {
        let mut result = LaurentSeries::new(
            self.variable.clone(),
            self.center.clone(),
            self.principal_part_order,
            0,
        );
        result.negative_terms = self.negative_terms.clone();
        result
    }

    /// Get the analytic part (non-negative power terms only) as a regular Series.
    pub fn analytic_part(&self) -> Series {
        let mut result = Series::new(
            self.variable.clone(),
            self.center.clone(),
            self.analytic_part_order,
        );
        for term in &self.positive_terms {
            result.add_term(term.clone());
        }
        result
    }

    /// Check if this is actually a Taylor series (no negative powers).
    pub fn is_taylor(&self) -> bool {
        self.negative_terms.is_empty()
    }

    /// Convert to a Taylor series if there are no negative powers.
    pub fn to_taylor(&self) -> Option<Series> {
        if self.is_taylor() {
            Some(self.analytic_part())
        } else {
            None
        }
    }

    /// Evaluate the Laurent series at a point.
    pub fn evaluate(&self, x: f64) -> Option<f64> {
        let center_val = try_expr_to_f64(&self.center)?;
        let dx = x - center_val;

        if dx.abs() < 1e-15 && !self.negative_terms.is_empty() {
            // At the singularity with poles - undefined
            return None;
        }

        let mut sum = 0.0;

        // Add positive power terms
        for term in &self.positive_terms {
            let coeff = try_expr_to_f64(&term.coefficient)?;
            sum += coeff * dx.powi(term.power as i32);
        }

        // Add negative power terms
        for term in &self.negative_terms {
            let coeff = try_expr_to_f64(&term.coefficient)?;
            sum += coeff / dx.powi(term.power as i32);
        }

        Some(sum)
    }

    /// Convert the Laurent series to LaTeX representation.
    pub fn to_latex(&self) -> String {
        let is_centered_at_zero = matches!(&self.center, Expression::Integer(0))
            || matches!(&self.center, Expression::Float(x) if x.abs() < 1e-15);

        let var_name = &self.variable.name;
        let mut parts = Vec::new();

        // Negative powers first (in descending order)
        let mut neg_sorted: Vec<_> = self.negative_terms.iter().collect();
        neg_sorted.sort_by(|a, b| b.power.cmp(&a.power));

        for term in neg_sorted {
            let coeff_str = format_coefficient_latex(&term.coefficient);
            let var_part = if is_centered_at_zero {
                format!("{}^{{-{}}}", var_name, term.power)
            } else {
                format!("({} - {})^{{-{}}}", var_name, self.center, term.power)
            };

            let term_str = if coeff_str == "1" {
                var_part
            } else if coeff_str == "-1" {
                format!("-{}", var_part)
            } else {
                format!("{} {}", coeff_str, var_part)
            };

            if parts.is_empty() {
                parts.push(term_str);
            } else if term_str.starts_with('-') {
                parts.push(format!(" - {}", &term_str[1..]));
            } else {
                parts.push(format!(" + {}", term_str));
            }
        }

        // Positive powers
        for term in &self.positive_terms {
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
                } else if term.power == 1 {
                    format!("({} - {})", var_name, self.center)
                } else {
                    format!("({} - {})^{{{}}}", var_name, self.center, term.power)
                };

                if coeff_str == "1" {
                    var_part
                } else if coeff_str == "-1" {
                    format!("-{}", var_part)
                } else {
                    format!("{} {}", coeff_str, var_part)
                }
            };

            if parts.is_empty() {
                parts.push(term_str);
            } else if term_str.starts_with('-') {
                parts.push(format!(" - {}", &term_str[1..]));
            } else {
                parts.push(format!(" + {}", term_str));
            }
        }

        if parts.is_empty() {
            "0".to_string()
        } else {
            parts.join("")
        }
    }
}

impl fmt::Display for LaurentSeries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let latex = self.to_latex();
        // Convert LaTeX to plain text
        let plain = latex
            .replace("^{-", "^(-")
            .replace("^{", "^")
            .replace("}", ")")
            .replace("{", "");
        write!(f, "{}", plain)
    }
}

/// Compute Laurent series expansion of an expression around a center point.
///
/// The Laurent series includes terms from (x-c)^{-neg_order} to (x-c)^{pos_order}.
///
/// # Arguments
/// * `expr` - The expression to expand
/// * `var` - The variable to expand in
/// * `center` - The center point of expansion
/// * `neg_order` - Maximum negative power (pole order)
/// * `pos_order` - Maximum positive power
///
/// # Example
/// ```rust
/// use thales::series::laurent;
/// use thales::ast::{Expression, Variable, BinaryOp};
///
/// let x = Variable::new("x");
/// // 1/x Laurent series around 0
/// let expr = Expression::Binary(
///     BinaryOp::Div,
///     Box::new(Expression::Integer(1)),
///     Box::new(Expression::Variable(x.clone())),
/// );
/// let series = laurent(&expr, &x, &Expression::Integer(0), 1, 3).unwrap();
/// // Series has term x^{-1} with coefficient 1
/// ```
pub fn laurent(
    expr: &Expression,
    var: &Variable,
    center: &Expression,
    neg_order: u32,
    pos_order: u32,
) -> SeriesResult<LaurentSeries> {
    let mut series = LaurentSeries::new(var.clone(), center.clone(), neg_order, pos_order);

    // Try to detect the structure of the expression to compute Laurent coefficients
    match expr {
        // Simple case: 1/x^n centered at 0
        Expression::Binary(BinaryOp::Div, num, denom) => {
            if is_power_of_var(denom, var, center) {
                let power = get_var_power(denom, var);
                if power > 0 {
                    // This is num / (x-c)^power
                    // Laurent series of num / (x-c)^power = (Taylor of num) * (x-c)^{-power}
                    let num_taylor = taylor(num, var, center, pos_order + power as u32)?;

                    for term in &num_taylor.terms {
                        let new_power = term.power as i32 - power;
                        if new_power < 0 && (-new_power as u32) <= neg_order {
                            series.add_negative_term(term.coefficient.clone(), (-new_power) as u32);
                        } else if new_power >= 0 && (new_power as u32) <= pos_order {
                            series.add_positive_term(SeriesTerm::new(
                                term.coefficient.clone(),
                                new_power as u32,
                            ));
                        }
                    }

                    return Ok(series);
                }
            }

            // General case: try Taylor expansion of numerator and denominator
            // f/g = (Taylor(f)) / (Taylor(g))
            // For g with a zero at center, we need to handle poles
            let num_taylor = taylor(num, var, center, pos_order + neg_order)?;
            let denom_taylor = taylor(denom, var, center, pos_order + neg_order)?;

            // Find the order of the zero of the denominator
            let denom_zero_order = find_leading_power(&denom_taylor);

            if denom_zero_order > 0 {
                // Denominator has a zero of order denom_zero_order
                // f/g has a pole of order <= denom_zero_order at center

                // Compute f/g by polynomial long division
                let result = divide_series(&num_taylor, &denom_taylor, neg_order, pos_order)?;

                for (power, coeff) in result {
                    if power < 0 && (-power as u32) <= neg_order {
                        series.add_negative_term(coeff, (-power) as u32);
                    } else if power >= 0 && (power as u32) <= pos_order {
                        series.add_positive_term(SeriesTerm::new(coeff, power as u32));
                    }
                }
            } else {
                // Denominator doesn't vanish at center, just use Taylor series
                let regular_taylor = taylor(expr, var, center, pos_order)?;
                for term in regular_taylor.terms {
                    series.add_positive_term(term);
                }
            }
        }

        Expression::Power(base, exp) => {
            // Handle (x-c)^{-n} type expressions
            if let Expression::Unary(crate::ast::UnaryOp::Neg, inner_exp) = exp.as_ref() {
                if is_var_minus_center(base, var, center) {
                    if let Some(n) = extract_positive_integer(inner_exp) {
                        // This is (x-c)^{-n}
                        if n <= neg_order {
                            series.add_negative_term(Expression::Integer(1), n);
                        }
                        return Ok(series);
                    }
                }
            }

            // Fall back to Taylor series for other power expressions
            let regular_taylor = taylor(expr, var, center, pos_order)?;
            for term in regular_taylor.terms {
                series.add_positive_term(term);
            }
        }

        _ => {
            // For expressions without obvious poles, compute Taylor series
            let regular_taylor = taylor(expr, var, center, pos_order)?;
            for term in regular_taylor.terms {
                series.add_positive_term(term);
            }
        }
    }

    Ok(series)
}

/// Compute the residue of an expression at a pole.
///
/// The residue is the coefficient of (x - center)^{-1} in the Laurent series.
///
/// # Example
/// ```rust
/// use thales::series::residue;
/// use thales::ast::{Expression, Variable, BinaryOp};
///
/// let x = Variable::new("x");
/// // Residue of 1/(x-1) at x=1 is 1
/// let expr = Expression::Binary(
///     BinaryOp::Div,
///     Box::new(Expression::Integer(1)),
///     Box::new(Expression::Binary(
///         BinaryOp::Sub,
///         Box::new(Expression::Variable(x.clone())),
///         Box::new(Expression::Integer(1)),
///     )),
/// );
/// let res = residue(&expr, &x, &Expression::Integer(1)).unwrap();
/// ```
pub fn residue(
    expr: &Expression,
    var: &Variable,
    pole: &Expression,
) -> SeriesResult<Expression> {
    // Compute Laurent series around the pole with enough negative terms
    let laurent_series = laurent(expr, var, pole, 5, 0)?;
    Ok(laurent_series.residue())
}

/// Find the order of a pole at a given point.
///
/// Returns 0 if the point is not a pole (regular point or removable singularity).
pub fn pole_order(
    expr: &Expression,
    var: &Variable,
    point: &Expression,
) -> SeriesResult<u32> {
    let laurent_series = laurent(expr, var, point, 10, 0)?;

    // Find the highest negative power with non-zero coefficient
    let mut max_order = 0;
    for term in &laurent_series.negative_terms {
        if term.power > max_order {
            max_order = term.power;
        }
    }

    Ok(max_order)
}

/// Find singularities of an expression.
///
/// Currently handles rational functions by finding zeros of the denominator.
pub fn find_singularities(expr: &Expression, var: &Variable) -> Vec<Singularity> {
    let mut singularities = Vec::new();

    match expr {
        Expression::Binary(BinaryOp::Div, _num, denom) => {
            // Find zeros of denominator
            let zeros = find_zeros_of_expression(denom, var);

            for zero in zeros {
                // Determine singularity type
                let order = get_zero_multiplicity(denom, var, &zero);
                let sing_type = if order > 0 {
                    SingularityType::Pole(order)
                } else {
                    SingularityType::Removable
                };

                singularities.push(Singularity::new(zero, sing_type));
            }
        }
        _ => {
            // For non-rational expressions, singularity detection is more complex
            // This is a simplified implementation
        }
    }

    singularities
}

// Helper functions for Laurent series

fn is_power_of_var(expr: &Expression, var: &Variable, center: &Expression) -> bool {
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

fn is_var_minus_center(expr: &Expression, var: &Variable, center: &Expression) -> bool {
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

fn get_var_power(expr: &Expression, var: &Variable) -> i32 {
    match expr {
        Expression::Variable(v) if v.name == var.name => 1,
        Expression::Power(base, exp) => {
            if let Expression::Variable(v) = base.as_ref() {
                if v.name == var.name {
                    if let Some(n) = extract_integer(exp) {
                        return n as i32;
                    }
                }
            }
            1
        }
        Expression::Binary(BinaryOp::Sub, left, _) => {
            if let Expression::Variable(v) = left.as_ref() {
                if v.name == var.name {
                    return 1;
                }
            }
            0
        }
        _ => 0,
    }
}

fn extract_integer(expr: &Expression) -> Option<i64> {
    match expr {
        Expression::Integer(n) => Some(*n),
        Expression::Float(f) if f.fract() == 0.0 => Some(*f as i64),
        _ => None,
    }
}

fn extract_positive_integer(expr: &Expression) -> Option<u32> {
    extract_integer(expr).and_then(|n| if n > 0 { Some(n as u32) } else { None })
}

fn expressions_equal(a: &Expression, b: &Expression) -> bool {
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

fn find_leading_power(series: &Series) -> u32 {
    series
        .terms
        .iter()
        .filter(|t| !t.is_zero())
        .map(|t| t.power)
        .min()
        .unwrap_or(0)
}

fn divide_series(
    num: &Series,
    denom: &Series,
    neg_order: u32,
    pos_order: u32,
) -> SeriesResult<Vec<(i32, Expression)>> {
    // Polynomial long division for series
    // This is a simplified implementation

    let mut result = Vec::new();
    let denom_lead_power = find_leading_power(denom) as i32;
    let denom_lead_coeff = denom
        .terms
        .iter()
        .find(|t| t.power as i32 == denom_lead_power)
        .map(|t| &t.coefficient);

    if denom_lead_coeff.is_none() {
        return Err(SeriesError::DivisionByZero);
    }

    // Simple case: if denominator leading term is just a constant multiple of x^n
    // We can compute the quotient directly

    for num_term in &num.terms {
        let result_power = num_term.power as i32 - denom_lead_power;

        if result_power >= -(neg_order as i32) && result_power <= pos_order as i32 {
            // Divide coefficient by leading denominator coefficient
            let coeff = match denom_lead_coeff {
                Some(Expression::Integer(d)) if *d != 0 => {
                    match &num_term.coefficient {
                        Expression::Integer(n) => Expression::Rational(
                            num_rational::Rational64::new(*n, *d),
                        ),
                        Expression::Rational(r) => Expression::Rational(
                            *r / num_rational::Rational64::from(*d),
                        ),
                        other => Expression::Binary(
                            BinaryOp::Div,
                            Box::new(other.clone()),
                            Box::new(Expression::Integer(*d)),
                        ),
                    }
                }
                _ => num_term.coefficient.clone(),
            };

            result.push((result_power, coeff));
        }
    }

    Ok(result)
}

fn find_zeros_of_expression(expr: &Expression, var: &Variable) -> Vec<Expression> {
    // Simplified zero finding - only handles simple cases
    let mut zeros = Vec::new();

    match expr {
        Expression::Variable(v) if v.name == var.name => {
            zeros.push(Expression::Integer(0));
        }
        Expression::Binary(BinaryOp::Sub, left, right) => {
            if matches!(left.as_ref(), Expression::Variable(v) if v.name == var.name) {
                zeros.push(right.as_ref().clone());
            }
        }
        Expression::Binary(BinaryOp::Mul, left, right) => {
            zeros.extend(find_zeros_of_expression(left, var));
            zeros.extend(find_zeros_of_expression(right, var));
        }
        Expression::Power(base, _) => {
            zeros.extend(find_zeros_of_expression(base, var));
        }
        _ => {}
    }

    zeros
}

fn get_zero_multiplicity(expr: &Expression, var: &Variable, zero: &Expression) -> u32 {
    match expr {
        Expression::Power(base, exp) => {
            let base_zeros = find_zeros_of_expression(base, var);
            if base_zeros.iter().any(|z| expressions_equal(z, zero)) {
                extract_integer(exp).unwrap_or(1) as u32
            } else {
                0
            }
        }
        Expression::Binary(BinaryOp::Mul, left, right) => {
            get_zero_multiplicity(left, var, zero) + get_zero_multiplicity(right, var, zero)
        }
        _ => {
            let zeros = find_zeros_of_expression(expr, var);
            if zeros.iter().any(|z| expressions_equal(z, zero)) {
                1
            } else {
                0
            }
        }
    }
}

/// Direction for asymptotic expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsymptoticDirection {
    /// Expansion as x approaches positive infinity.
    PosInfinity,
    /// Expansion as x approaches negative infinity.
    NegInfinity,
    /// Expansion as x approaches zero.
    Zero,
}

impl fmt::Display for AsymptoticDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AsymptoticDirection::PosInfinity => write!(f, "x→+∞"),
            AsymptoticDirection::NegInfinity => write!(f, "x→-∞"),
            AsymptoticDirection::Zero => write!(f, "x→0"),
        }
    }
}

/// A single term in an asymptotic series: coefficient × x^exponent.
/// The exponent can be negative or fractional for asymptotic expansions.
#[derive(Debug, Clone, PartialEq)]
pub struct AsymptoticTerm {
    /// The coefficient of this term (can be symbolic).
    pub coefficient: Expression,
    /// The exponent (can be negative, fractional, or symbolic).
    pub exponent: Expression,
}

impl AsymptoticTerm {
    /// Create a new asymptotic term.
    pub fn new(coefficient: Expression, exponent: Expression) -> Self {
        AsymptoticTerm { coefficient, exponent }
    }

    /// Check if this term has a zero coefficient.
    pub fn is_zero(&self) -> bool {
        matches!(&self.coefficient, Expression::Integer(0))
            || matches!(&self.coefficient, Expression::Float(x) if x.abs() < 1e-15)
    }

    /// Try to evaluate this term at a given point.
    pub fn evaluate(&self, var: &Variable, point: f64) -> Option<f64> {
        // Substitute the variable with the point value
        let point_expr = Expression::Float(point);

        let coeff_result = evaluate_at(&self.coefficient, var, &point_expr).ok()?;
        let exp_result = evaluate_at(&self.exponent, var, &point_expr).ok()?;

        let coeff = try_expr_to_f64(&coeff_result)?;
        let exp = try_expr_to_f64(&exp_result)?;

        Some(coeff * point.powf(exp))
    }
}

impl fmt::Display for AsymptoticTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Format based on exponent value
        match &self.exponent {
            Expression::Integer(0) => write!(f, "{}", self.coefficient),
            Expression::Integer(1) => write!(f, "{}·x", self.coefficient),
            Expression::Integer(n) if *n < 0 => {
                write!(f, "{}/x^{}", self.coefficient, -n)
            }
            Expression::Rational(r) => {
                write!(f, "{}·x^({})", self.coefficient, r)
            }
            _ => write!(f, "{}·x^({})", self.coefficient, self.exponent),
        }
    }
}

/// Big-O notation for asymptotic order.
#[derive(Debug, Clone, PartialEq)]
pub struct BigO {
    /// Order expression (e.g., x^n, log(x), etc.)
    pub order: Expression,
    /// Variable of the expansion.
    pub variable: Variable,
}

impl BigO {
    /// Create a new Big-O term.
    pub fn new(order: Expression, variable: Variable) -> Self {
        BigO { order, variable }
    }

    /// Check if two Big-O terms have the same order.
    pub fn is_same_order(&self, other: &BigO) -> bool {
        // Simplified check - would need more sophisticated comparison
        self.order == other.order
    }

    /// Check if this Big-O is smaller than another.
    /// Returns true if this order grows slower than other.
    pub fn is_smaller_order(&self, other: &BigO) -> bool {
        // For power terms: O(x^a) < O(x^b) if a < b
        match (&self.order, &other.order) {
            (Expression::Power(_, exp1), Expression::Power(_, exp2)) => {
                if let (Some(e1), Some(e2)) = (try_expr_to_f64(exp1), try_expr_to_f64(exp2)) {
                    return e1 < e2;
                }
            }
            (Expression::Integer(a), Expression::Power(_, _)) => {
                // Constant is smaller than any power
                return *a == 1;
            }
            _ => {}
        }
        false
    }

    /// Check if an expression is bounded by this Big-O term.
    pub fn is_bounded_by(&self, expr: &Expression) -> bool {
        // Simplified check - full implementation would need limit analysis
        expr == &self.order
    }
}

impl fmt::Display for BigO {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "O({})", self.order)
    }
}

/// A complete asymptotic series representation.
#[derive(Debug, Clone, PartialEq)]
pub struct AsymptoticSeries {
    /// The terms of the series, ordered by decreasing dominance.
    pub terms: Vec<AsymptoticTerm>,
    /// The variable of expansion.
    pub variable: Variable,
    /// Direction of the asymptotic expansion.
    pub direction: AsymptoticDirection,
}

impl AsymptoticSeries {
    /// Create a new empty asymptotic series.
    pub fn new(variable: Variable, direction: AsymptoticDirection) -> Self {
        AsymptoticSeries {
            terms: Vec::new(),
            variable,
            direction,
        }
    }

    /// Add a term to the series.
    pub fn add_term(&mut self, term: AsymptoticTerm) {
        if !term.is_zero() {
            self.terms.push(term);
        }
    }

    /// Get the dominant (leading) term.
    pub fn dominant_term(&self) -> Option<&AsymptoticTerm> {
        self.terms.first()
    }

    /// Get the order of magnitude (exponent of dominant term).
    pub fn order_of_magnitude(&self) -> Option<Expression> {
        self.dominant_term().map(|t| t.exponent.clone())
    }

    /// Return the series with its error term.
    pub fn with_error_term(&self) -> (Self, BigO) {
        let series = self.clone();

        // Error term is the next order after the smallest term
        let error_order = if let Some(last_term) = self.terms.last() {
            // For x->inf: if last term is x^n, error is O(x^(n-1))
            // For x->0: if last term is x^n, error is O(x^(n+1))
            match self.direction {
                AsymptoticDirection::PosInfinity | AsymptoticDirection::NegInfinity => {
                    Expression::Binary(
                        BinaryOp::Sub,
                        Box::new(last_term.exponent.clone()),
                        Box::new(Expression::Integer(1)),
                    )
                }
                AsymptoticDirection::Zero => {
                    Expression::Binary(
                        BinaryOp::Add,
                        Box::new(last_term.exponent.clone()),
                        Box::new(Expression::Integer(1)),
                    )
                }
            }
        } else {
            Expression::Integer(0)
        };

        let var_expr = Expression::Variable(self.variable.clone());
        let big_o = BigO::new(
            Expression::Power(Box::new(var_expr), Box::new(error_order)),
            self.variable.clone(),
        );

        (series, big_o)
    }

    /// Evaluate the series at a given point.
    pub fn evaluate(&self, point: f64) -> Option<f64> {
        let mut sum = 0.0;
        for term in &self.terms {
            sum += term.evaluate(&self.variable, point)?;
        }
        Some(sum)
    }

    /// Convert to a symbolic Expression.
    pub fn to_expression(&self) -> Expression {
        if self.terms.is_empty() {
            return Expression::Integer(0);
        }

        let var_expr = Expression::Variable(self.variable.clone());
        let mut result: Option<Expression> = None;

        for term in &self.terms {
            let term_expr = Expression::Binary(
                BinaryOp::Mul,
                Box::new(term.coefficient.clone()),
                Box::new(Expression::Power(
                    Box::new(var_expr.clone()),
                    Box::new(term.exponent.clone()),
                )),
            );

            result = Some(match result {
                None => term_expr,
                Some(acc) => Expression::Binary(BinaryOp::Add, Box::new(acc), Box::new(term_expr)),
            });
        }

        result.unwrap_or(Expression::Integer(0)).simplify()
    }
}

impl fmt::Display for AsymptoticSeries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }

        write!(f, "As {}: ", self.direction)?;

        for (i, term) in self.terms.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            write!(f, "{}", term)?;
        }

        Ok(())
    }
}

/// Compute asymptotic expansion of an expression.
///
/// This function computes the asymptotic expansion of an expression as the variable
/// approaches a limit (zero or infinity). The expansion includes terms in decreasing
/// order of dominance.
///
/// # Arguments
///
/// * `expr` - The expression to expand
/// * `var` - The variable for expansion
/// * `direction` - Direction of approach (zero or infinity)
/// * `num_terms` - Number of terms to compute
///
/// # Returns
///
/// An `AsymptoticSeries` containing the dominant terms.
///
/// # Examples
///
/// ```
/// use thales::series::{asymptotic, AsymptoticDirection};
/// use thales::parser::parse_expression;
///
/// // Asymptotic expansion of 1/x + 1/x^2 as x→∞
/// let expr = parse_expression("1/x + 1/x^2").unwrap();
/// let series = asymptotic(&expr, "x", AsymptoticDirection::PosInfinity, 3).unwrap();
/// // Returns: 1/x + 1/x^2
/// ```
pub fn asymptotic(
    expr: &Expression,
    var: impl AsRef<str>,
    direction: AsymptoticDirection,
    num_terms: u32,
) -> SeriesResult<AsymptoticSeries> {
    let var_name = var.as_ref();
    let variable = Variable::new(var_name);

    let mut series = AsymptoticSeries::new(variable.clone(), direction);

    // Extract terms directly from the expression
    // No substitution needed - we handle dominance in sorting
    extract_asymptotic_terms(expr, &variable, &mut series, num_terms)?;

    // Sort terms by dominance
    sort_by_dominance(&mut series.terms, direction);

    Ok(series)
}

/// Substitute x = 1/t for infinity analysis.
fn substitute_for_infinity(expr: &Expression, var: &Variable) -> Expression {
    match expr {
        Expression::Variable(v) if v == var => {
            // x -> 1/t
            let t = Variable::new("__t");
            Expression::Binary(
                BinaryOp::Div,
                Box::new(Expression::Integer(1)),
                Box::new(Expression::Variable(t)),
            )
        }
        Expression::Binary(op, left, right) => {
            Expression::Binary(
                *op,
                Box::new(substitute_for_infinity(left, var)),
                Box::new(substitute_for_infinity(right, var)),
            )
        }
        Expression::Unary(op, inner) => {
            Expression::Unary(*op, Box::new(substitute_for_infinity(inner, var)))
        }
        Expression::Power(base, exp) => {
            Expression::Power(
                Box::new(substitute_for_infinity(base, var)),
                Box::new(substitute_for_infinity(exp, var)),
            )
        }
        Expression::Function(f, args) => {
            Expression::Function(
                f.clone(),
                args.iter().map(|a| substitute_for_infinity(a, var)).collect(),
            )
        }
        _ => expr.clone(),
    }
}

/// Extract asymptotic terms from an expression.
fn extract_asymptotic_terms(
    expr: &Expression,
    var: &Variable,
    series: &mut AsymptoticSeries,
    num_terms: u32,
) -> SeriesResult<()> {
    // Simple extraction for basic forms
    match expr {
        Expression::Binary(BinaryOp::Add, left, right) => {
            extract_asymptotic_terms(left, var, series, num_terms)?;
            extract_asymptotic_terms(right, var, series, num_terms)?;
        }
        Expression::Binary(BinaryOp::Div, num, den) => {
            // Check if numerator is effectively 1 (could be Integer(1) or Float(1.0))
            let is_one = matches!(num.as_ref(), Expression::Integer(1))
                || matches!(num.as_ref(), Expression::Float(f) if (*f - 1.0).abs() < 1e-15);

            if is_one {
                // Handle 1/x
                if let Expression::Variable(v) = den.as_ref() {
                    if v == var {
                        series.add_term(AsymptoticTerm::new(
                            Expression::Integer(1),
                            Expression::Integer(-1),
                        ));
                    }
                // Handle 1/x^n
                } else if let Expression::Power(base, exp) = den.as_ref() {
                    if let Expression::Variable(v) = base.as_ref() {
                        if v == var {
                            // 1/x^n -> coefficient=1, exponent=-n
                            let neg_exp = Expression::Unary(
                                crate::ast::UnaryOp::Neg,
                                Box::new(exp.as_ref().clone()),
                            ).simplify();
                            series.add_term(AsymptoticTerm::new(Expression::Integer(1), neg_exp));
                        }
                    }
                }
            } else {
                // Handle general a/x^n form
                if let Expression::Variable(v) = den.as_ref() {
                    if v == var {
                        series.add_term(AsymptoticTerm::new(
                            num.as_ref().clone(),
                            Expression::Integer(-1),
                        ));
                    }
                } else if let Expression::Power(base, exp) = den.as_ref() {
                    if let Expression::Variable(v) = base.as_ref() {
                        if v == var {
                            let neg_exp = Expression::Unary(
                                crate::ast::UnaryOp::Neg,
                                Box::new(exp.as_ref().clone()),
                            ).simplify();
                            series.add_term(AsymptoticTerm::new(num.as_ref().clone(), neg_exp));
                        }
                    }
                }
            }
        }
        Expression::Power(base, exp) => {
            if let Expression::Variable(v) = base.as_ref() {
                if v == var {
                    series.add_term(AsymptoticTerm::new(Expression::Integer(1), exp.as_ref().clone()));
                }
            }
        }
        Expression::Variable(v) if v == var => {
            series.add_term(AsymptoticTerm::new(Expression::Integer(1), Expression::Integer(1)));
        }
        Expression::Integer(n) => {
            series.add_term(AsymptoticTerm::new(Expression::Integer(*n), Expression::Integer(0)));
        }
        Expression::Float(f) => {
            series.add_term(AsymptoticTerm::new(Expression::Float(*f), Expression::Integer(0)));
        }
        _ => {
            return Err(SeriesError::CannotExpand(format!(
                "Cannot extract asymptotic terms from: {}",
                expr
            )));
        }
    }

    Ok(())
}

/// Sort terms by dominance for the given direction.
fn sort_by_dominance(terms: &mut Vec<AsymptoticTerm>, direction: AsymptoticDirection) {
    terms.sort_by(|a, b| {
        let exp_a = try_expr_to_f64(&a.exponent).unwrap_or(0.0);
        let exp_b = try_expr_to_f64(&b.exponent).unwrap_or(0.0);

        match direction {
            AsymptoticDirection::PosInfinity | AsymptoticDirection::NegInfinity => {
                // For x→∞, larger exponents dominate
                exp_b.partial_cmp(&exp_a).unwrap()
            }
            AsymptoticDirection::Zero => {
                // For x→0, smaller exponents dominate
                exp_a.partial_cmp(&exp_b).unwrap()
            }
        }
    });
}

/// Compute limit via asymptotic expansion.
///
/// This function uses asymptotic expansion to compute limits, especially
/// useful for limits at infinity where standard Taylor series don't apply.
///
/// # Arguments
///
/// * `expr` - The expression to compute the limit of
/// * `var` - The variable approaching the limit
/// * `direction` - Direction of approach
///
/// # Returns
///
/// The limit value as a `LimitResult`.
pub fn limit_via_asymptotic(
    expr: &Expression,
    var: impl AsRef<str>,
    direction: AsymptoticDirection,
) -> Result<crate::limits::LimitResult, crate::limits::LimitError> {
    use crate::limits::LimitResult;

    let series = asymptotic(expr, var.as_ref(), direction, 5)
        .map_err(|e| crate::limits::LimitError::EvaluationError(e.to_string()))?;

    if let Some(dominant) = series.dominant_term() {
        // Check the exponent of the dominant term
        if let Some(exp) = try_expr_to_f64(&dominant.exponent) {
            match direction {
                AsymptoticDirection::PosInfinity | AsymptoticDirection::NegInfinity => {
                    if exp > 0.0 {
                        // Dominant term has positive exponent -> infinity
                        if let Some(coeff) = try_expr_to_f64(&dominant.coefficient) {
                            if coeff > 0.0 {
                                return Ok(LimitResult::PositiveInfinity);
                            } else if coeff < 0.0 {
                                return Ok(LimitResult::NegativeInfinity);
                            }
                        }
                    } else if exp < 0.0 {
                        // Dominant term has negative exponent -> 0
                        return Ok(LimitResult::Value(0.0));
                    } else {
                        // Constant term
                        if let Some(coeff) = try_expr_to_f64(&dominant.coefficient) {
                            return Ok(LimitResult::Value(coeff));
                        }
                    }
                }
                AsymptoticDirection::Zero => {
                    if exp > 0.0 {
                        // As x→0, x^n → 0 for n > 0
                        return Ok(LimitResult::Value(0.0));
                    } else if exp < 0.0 {
                        // As x→0, x^(-n) → ∞ for n > 0
                        if let Some(coeff) = try_expr_to_f64(&dominant.coefficient) {
                            if coeff > 0.0 {
                                return Ok(LimitResult::PositiveInfinity);
                            } else if coeff < 0.0 {
                                return Ok(LimitResult::NegativeInfinity);
                            }
                        }
                    } else {
                        // Constant term
                        if let Some(coeff) = try_expr_to_f64(&dominant.coefficient) {
                            return Ok(LimitResult::Value(coeff));
                        }
                    }
                }
            }
        }
    }

    Err(crate::limits::LimitError::EvaluationError(
        "Cannot determine limit from asymptotic expansion".to_string(),
    ))
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

        assert!(matches!(singularity.singularity_type, SingularityType::Pole(2)));
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
        let exp_series = exp_series(&x, 4);
        let deriv = exp_series.differentiate();

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
        let exp_series = exp_series(&x, 3);
        let integrated = exp_series.integrate(Expression::Integer(0));

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

        series.add_term(AsymptoticTerm::new(Expression::Integer(1), Expression::Integer(-1)));
        series.add_term(AsymptoticTerm::new(Expression::Integer(1), Expression::Integer(-2)));

        assert_eq!(series.terms.len(), 2);
    }

    #[test]
    fn test_asymptotic_series_dominant_term() {
        let x = Variable::new("x");
        let mut series = AsymptoticSeries::new(x.clone(), AsymptoticDirection::PosInfinity);

        series.add_term(AsymptoticTerm::new(Expression::Integer(1), Expression::Integer(-1)));
        series.add_term(AsymptoticTerm::new(Expression::Integer(1), Expression::Integer(-2)));

        let dominant = series.dominant_term().unwrap();
        assert_eq!(dominant.exponent, Expression::Integer(-1));
    }

    #[test]
    fn test_asymptotic_series_order_of_magnitude() {
        let x = Variable::new("x");
        let mut series = AsymptoticSeries::new(x.clone(), AsymptoticDirection::PosInfinity);

        series.add_term(AsymptoticTerm::new(Expression::Integer(2), Expression::Integer(-1)));

        let order = series.order_of_magnitude().unwrap();
        assert_eq!(order, Expression::Integer(-1));
    }

    #[test]
    fn test_asymptotic_series_with_error_term() {
        let x = Variable::new("x");
        let mut series = AsymptoticSeries::new(x.clone(), AsymptoticDirection::PosInfinity);

        series.add_term(AsymptoticTerm::new(Expression::Integer(1), Expression::Integer(-1)));
        series.add_term(AsymptoticTerm::new(Expression::Integer(1), Expression::Integer(-2)));

        let (_, big_o) = series.with_error_term();
        // Error term should be O(x^(-3)) for x→∞ when last term is x^(-2)
        assert_eq!(big_o.variable, x);
    }

    #[test]
    fn test_asymptotic_series_evaluate() {
        let x = Variable::new("x");
        let mut series = AsymptoticSeries::new(x.clone(), AsymptoticDirection::PosInfinity);

        // 1/x + 1/x^2 at x=2 should be 0.5 + 0.25 = 0.75
        series.add_term(AsymptoticTerm::new(Expression::Integer(1), Expression::Integer(-1)));
        series.add_term(AsymptoticTerm::new(Expression::Integer(1), Expression::Integer(-2)));

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
        let x = Variable::new("x");
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
        let x = Variable::new("x");
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
        use crate::parser::parse_expression;
        use crate::limits::LimitResult;

        // lim_{x→∞} 1/x = 0
        let expr = parse_expression("1/x").unwrap();
        let result = limit_via_asymptotic(&expr, "x", AsymptoticDirection::PosInfinity).unwrap();

        assert_eq!(result, LimitResult::Value(0.0));
    }

    #[test]
    fn test_limit_via_asymptotic_to_infinity() {
        use crate::parser::parse_expression;
        use crate::limits::LimitResult;

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

        series.add_term(AsymptoticTerm::new(Expression::Integer(1), Expression::Integer(-1)));
        series.add_term(AsymptoticTerm::new(Expression::Integer(2), Expression::Integer(-2)));

        let expr = series.to_expression();
        // Should be simplifiable to some form
        assert!(!matches!(expr, Expression::Integer(0)));
    }

    #[test]
    fn test_asymptotic_series_display() {
        let x = Variable::new("x");
        let mut series = AsymptoticSeries::new(x.clone(), AsymptoticDirection::PosInfinity);

        series.add_term(AsymptoticTerm::new(Expression::Integer(1), Expression::Integer(-1)));

        let display_str = format!("{}", series);
        assert!(display_str.contains("x→+∞"));
    }
}
