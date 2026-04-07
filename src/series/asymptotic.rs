//! Asymptotic expansions, AsymptoticSeries, AsymptoticTerm, BigO, AsymptoticDirection.

use crate::ast::{BinaryOp, Expression, Variable};
use std::fmt;

use super::{evaluate_at, try_expr_to_f64, SeriesError, SeriesResult};

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
        AsymptoticTerm {
            coefficient,
            exponent,
        }
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
                AsymptoticDirection::Zero => Expression::Binary(
                    BinaryOp::Add,
                    Box::new(last_term.exponent.clone()),
                    Box::new(Expression::Integer(1)),
                ),
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
                            )
                            .simplify();
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
                            )
                            .simplify();
                            series.add_term(AsymptoticTerm::new(num.as_ref().clone(), neg_exp));
                        }
                    }
                }
            }
        }
        Expression::Power(base, exp) => {
            if let Expression::Variable(v) = base.as_ref() {
                if v == var {
                    series.add_term(AsymptoticTerm::new(
                        Expression::Integer(1),
                        exp.as_ref().clone(),
                    ));
                }
            }
        }
        Expression::Variable(v) if v == var => {
            series.add_term(AsymptoticTerm::new(
                Expression::Integer(1),
                Expression::Integer(1),
            ));
        }
        Expression::Integer(n) => {
            series.add_term(AsymptoticTerm::new(
                Expression::Integer(*n),
                Expression::Integer(0),
            ));
        }
        Expression::Float(f) => {
            series.add_term(AsymptoticTerm::new(
                Expression::Float(*f),
                Expression::Integer(0),
            ));
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
pub(crate) fn sort_by_dominance(terms: &mut Vec<AsymptoticTerm>, direction: AsymptoticDirection) {
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
