//! Laurent series, singularity analysis, pole order, and residue computation.

use crate::ast::{BinaryOp, Expression, Variable};
use std::fmt;

use super::taylor::taylor;
use super::{
    expressions_equal, extract_integer, extract_positive_integer, format_coefficient_latex,
    is_power_of_var, is_var_minus_center, try_expr_to_f64, Series, SeriesError, SeriesResult,
    SeriesTerm,
};

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
pub fn residue(expr: &Expression, var: &Variable, pole: &Expression) -> SeriesResult<Expression> {
    // Compute Laurent series around the pole with enough negative terms
    let laurent_series = laurent(expr, var, pole, 5, 0)?;
    Ok(laurent_series.residue())
}

/// Find the order of a pole at a given point.
///
/// Returns 0 if the point is not a pole (regular point or removable singularity).
pub fn pole_order(expr: &Expression, var: &Variable, point: &Expression) -> SeriesResult<u32> {
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
                Some(Expression::Integer(d)) if *d != 0 => match &num_term.coefficient {
                    Expression::Integer(n) => {
                        Expression::Rational(num_rational::Rational64::new(*n, *d))
                    }
                    Expression::Rational(r) => {
                        Expression::Rational(*r / num_rational::Rational64::from(*d))
                    }
                    other => Expression::Binary(
                        BinaryOp::Div,
                        Box::new(other.clone()),
                        Box::new(Expression::Integer(*d)),
                    ),
                },
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
