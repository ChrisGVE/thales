//! Inequality solving module for linear and quadratic inequalities.
//!
//! This module provides functionality for solving various types of inequalities:
//! - Linear inequalities (e.g., 2x + 3 > 7)
//! - Quadratic inequalities (e.g., x² - 4 < 0)
//! - Systems of inequalities (conjunction of multiple inequalities)
//!
//! # Example
//!
//! ```
//! use thales::inequality::{Inequality, solve_inequality};
//! use thales::ast::{Expression, Variable};
//!
//! // Solve 2x + 3 > 7
//! let x = Expression::Variable(Variable::new("x"));
//! let lhs = Expression::Binary(
//!     thales::ast::BinaryOp::Add,
//!     Box::new(Expression::Binary(
//!         thales::ast::BinaryOp::Mul,
//!         Box::new(Expression::Integer(2)),
//!         Box::new(x.clone()),
//!     )),
//!     Box::new(Expression::Integer(3)),
//! );
//! let rhs = Expression::Integer(7);
//!
//! let ineq = Inequality::GreaterThan(lhs, rhs);
//! let solution = solve_inequality(&ineq, "x").unwrap();
//! // Solution: x > 2
//! ```

use crate::ast::{BinaryOp, Expression, UnaryOp, Variable};
use std::collections::HashMap;
use std::fmt;

/// Represents an inequality relation between two expressions.
#[derive(Debug, Clone, PartialEq)]
pub enum Inequality {
    /// Strict less than: left < right
    LessThan(Expression, Expression),
    /// Less than or equal: left ≤ right
    LessEqual(Expression, Expression),
    /// Strict greater than: left > right
    GreaterThan(Expression, Expression),
    /// Greater than or equal: left ≥ right
    GreaterEqual(Expression, Expression),
}

impl fmt::Display for Inequality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Inequality::LessThan(l, r) => write!(f, "{} < {}", l, r),
            Inequality::LessEqual(l, r) => write!(f, "{} ≤ {}", l, r),
            Inequality::GreaterThan(l, r) => write!(f, "{} > {}", l, r),
            Inequality::GreaterEqual(l, r) => write!(f, "{} ≥ {}", l, r),
        }
    }
}

/// Represents a bound on a real number line.
#[derive(Debug, Clone, PartialEq)]
pub enum Bound {
    /// Negative infinity (-∞)
    NegativeInfinity,
    /// Positive infinity (+∞)
    PositiveInfinity,
    /// A specific value (may be inclusive or exclusive)
    Value(Expression),
}

impl fmt::Display for Bound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Bound::NegativeInfinity => write!(f, "-∞"),
            Bound::PositiveInfinity => write!(f, "+∞"),
            Bound::Value(e) => write!(f, "{}", e),
        }
    }
}

/// Represents the solution set of an inequality.
#[derive(Debug, Clone, PartialEq)]
pub enum IntervalSolution {
    /// A single interval [lower, upper] or (lower, upper) etc.
    Interval {
        /// Lower bound
        lower: Bound,
        /// Whether lower bound is inclusive
        lower_inclusive: bool,
        /// Upper bound
        upper: Bound,
        /// Whether upper bound is inclusive
        upper_inclusive: bool,
    },
    /// Union of multiple intervals
    Union(Vec<IntervalSolution>),
    /// Empty set (no solutions)
    Empty,
    /// All real numbers (-∞, +∞)
    AllReals,
}

impl fmt::Display for IntervalSolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntervalSolution::Interval {
                lower,
                lower_inclusive,
                upper,
                upper_inclusive,
            } => {
                let left_bracket = if *lower_inclusive { "[" } else { "(" };
                let right_bracket = if *upper_inclusive { "]" } else { ")" };
                write!(f, "{}{}, {}{}", left_bracket, lower, upper, right_bracket)
            }
            IntervalSolution::Union(intervals) => {
                let parts: Vec<String> = intervals.iter().map(|i| format!("{}", i)).collect();
                write!(f, "{}", parts.join(" ∪ "))
            }
            IntervalSolution::Empty => write!(f, "∅"),
            IntervalSolution::AllReals => write!(f, "(-∞, +∞)"),
        }
    }
}

impl IntervalSolution {
    /// Create an interval for x > a
    pub fn greater_than(a: Expression) -> Self {
        IntervalSolution::Interval {
            lower: Bound::Value(a),
            lower_inclusive: false,
            upper: Bound::PositiveInfinity,
            upper_inclusive: false,
        }
    }

    /// Create an interval for x ≥ a
    pub fn greater_equal(a: Expression) -> Self {
        IntervalSolution::Interval {
            lower: Bound::Value(a),
            lower_inclusive: true,
            upper: Bound::PositiveInfinity,
            upper_inclusive: false,
        }
    }

    /// Create an interval for x < a
    pub fn less_than(a: Expression) -> Self {
        IntervalSolution::Interval {
            lower: Bound::NegativeInfinity,
            lower_inclusive: false,
            upper: Bound::Value(a),
            upper_inclusive: false,
        }
    }

    /// Create an interval for x ≤ a
    pub fn less_equal(a: Expression) -> Self {
        IntervalSolution::Interval {
            lower: Bound::NegativeInfinity,
            lower_inclusive: false,
            upper: Bound::Value(a),
            upper_inclusive: true,
        }
    }

    /// Create an interval for a < x < b
    pub fn open_interval(a: Expression, b: Expression) -> Self {
        IntervalSolution::Interval {
            lower: Bound::Value(a),
            lower_inclusive: false,
            upper: Bound::Value(b),
            upper_inclusive: false,
        }
    }

    /// Create an interval for a ≤ x ≤ b
    pub fn closed_interval(a: Expression, b: Expression) -> Self {
        IntervalSolution::Interval {
            lower: Bound::Value(a),
            lower_inclusive: true,
            upper: Bound::Value(b),
            upper_inclusive: true,
        }
    }
}

/// Error types for inequality solving.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum InequalityError {
    /// Cannot solve this type of inequality
    CannotSolve(String),
    /// Variable not found in inequality
    VariableNotFound(String),
    /// Inequality is not linear or quadratic
    NonPolynomial(String),
}

impl fmt::Display for InequalityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InequalityError::CannotSolve(msg) => write!(f, "Cannot solve: {}", msg),
            InequalityError::VariableNotFound(var) => {
                write!(f, "Variable '{}' not found in inequality", var)
            }
            InequalityError::NonPolynomial(msg) => write!(f, "Non-polynomial: {}", msg),
        }
    }
}

impl std::error::Error for InequalityError {}

/// Result type for inequality solving.
pub type InequalityResult = Result<IntervalSolution, InequalityError>;

/// Solve an inequality for a given variable.
///
/// # Arguments
///
/// * `inequality` - The inequality to solve
/// * `var` - The variable to solve for
///
/// # Returns
///
/// The solution set as an interval or union of intervals.
pub fn solve_inequality(inequality: &Inequality, var: &str) -> InequalityResult {
    // Move everything to one side: f(x) <op> 0
    let (expr, is_strict, is_greater) = normalize_inequality(inequality);

    // Check if the variable appears in the expression
    if !expr.contains_variable(var) {
        // Constant inequality - evaluate
        return solve_constant_inequality(&expr, is_strict, is_greater);
    }

    // Try to determine polynomial degree
    let degree = estimate_polynomial_degree(&expr, var);

    match degree {
        Some(1) => solve_linear_inequality(&expr, var, is_strict, is_greater),
        Some(2) => solve_quadratic_inequality(&expr, var, is_strict, is_greater),
        Some(n) if n > 2 => {
            // Higher degree - try to factor or use numerical methods
            Err(InequalityError::CannotSolve(format!(
                "Polynomial degree {} is too high",
                n
            )))
        }
        _ => Err(InequalityError::NonPolynomial(
            "Cannot determine polynomial degree".to_string(),
        )),
    }
}

/// Normalize inequality to f(x) <op> 0 form.
/// Returns (expression, is_strict, is_greater_than_zero)
fn normalize_inequality(ineq: &Inequality) -> (Expression, bool, bool) {
    match ineq {
        // left < right => left - right < 0
        Inequality::LessThan(left, right) => {
            let diff = Expression::Binary(
                BinaryOp::Sub,
                Box::new(left.clone()),
                Box::new(right.clone()),
            );
            (diff, true, false)
        }
        // left ≤ right => left - right ≤ 0
        Inequality::LessEqual(left, right) => {
            let diff = Expression::Binary(
                BinaryOp::Sub,
                Box::new(left.clone()),
                Box::new(right.clone()),
            );
            (diff, false, false)
        }
        // left > right => left - right > 0
        Inequality::GreaterThan(left, right) => {
            let diff = Expression::Binary(
                BinaryOp::Sub,
                Box::new(left.clone()),
                Box::new(right.clone()),
            );
            (diff, true, true)
        }
        // left ≥ right => left - right ≥ 0
        Inequality::GreaterEqual(left, right) => {
            let diff = Expression::Binary(
                BinaryOp::Sub,
                Box::new(left.clone()),
                Box::new(right.clone()),
            );
            (diff, false, true)
        }
    }
}

/// Solve a constant inequality (no variables).
fn solve_constant_inequality(
    expr: &Expression,
    is_strict: bool,
    is_greater: bool,
) -> InequalityResult {
    // Try to evaluate the expression with no variables
    let vars: HashMap<String, f64> = HashMap::new();
    let val = expr.evaluate(&vars);

    match val {
        Some(v) => {
            let is_positive: bool = v > 0.0;
            let is_zero: bool = v.abs() < 1e-15;

            let satisfied = if is_greater {
                if is_strict {
                    is_positive && !is_zero
                } else {
                    is_positive || is_zero
                }
            } else {
                if is_strict {
                    !is_positive && !is_zero
                } else {
                    !is_positive || is_zero
                }
            };

            if satisfied {
                Ok(IntervalSolution::AllReals)
            } else {
                Ok(IntervalSolution::Empty)
            }
        }
        None => Err(InequalityError::CannotSolve(
            "Cannot evaluate constant expression".to_string(),
        )),
    }
}

/// Estimate the polynomial degree of an expression in a variable.
fn estimate_polynomial_degree(expr: &Expression, var: &str) -> Option<u32> {
    match expr {
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Rational(_)
        | Expression::Complex(_)
        | Expression::Constant(_) => Some(0),

        Expression::Variable(v) => {
            if v.name == var {
                Some(1)
            } else {
                Some(0)
            }
        }

        Expression::Unary(UnaryOp::Neg, inner) => estimate_polynomial_degree(inner, var),

        Expression::Binary(op, left, right) => {
            let left_deg = estimate_polynomial_degree(left, var)?;
            let right_deg = estimate_polynomial_degree(right, var)?;

            match op {
                BinaryOp::Add | BinaryOp::Sub => Some(left_deg.max(right_deg)),
                BinaryOp::Mul => Some(left_deg + right_deg),
                BinaryOp::Div => {
                    // f(x)/c is still polynomial if c doesn't contain x
                    if !right.contains_variable(var) {
                        Some(left_deg)
                    } else {
                        None // Rational function
                    }
                }
                BinaryOp::Mod => None,
            }
        }

        Expression::Power(base, exp) => {
            if !base.contains_variable(var) {
                Some(0)
            } else if !exp.contains_variable(var) {
                // x^n where n is constant
                let empty_vars: HashMap<String, f64> = HashMap::new();
                if let Some(n) = exp.evaluate(&empty_vars) {
                    if n >= 0.0 && (n - n.floor()).abs() < 1e-10 {
                        let base_deg = estimate_polynomial_degree(base, var)?;
                        Some(base_deg * (n as u32))
                    } else {
                        None // Non-integer exponent
                    }
                } else {
                    None
                }
            } else {
                None // Variable in exponent
            }
        }

        Expression::Function(_, _) => None, // Transcendental
        _ => None,
    }
}

/// Solve a linear inequality ax + b <op> 0.
fn solve_linear_inequality(
    expr: &Expression,
    var: &str,
    is_strict: bool,
    is_greater: bool,
) -> InequalityResult {
    // Extract coefficients a and b from ax + b
    let (a, b) = extract_linear_coefficients(expr, var)?;

    // ax + b <op> 0
    // x <op> -b/a (flipping if a < 0)

    let neg_b_over_a = if a.abs() < 1e-15 {
        // Degenerate case: b <op> 0
        if is_greater {
            if is_strict {
                return if b > 0.0 {
                    Ok(IntervalSolution::AllReals)
                } else {
                    Ok(IntervalSolution::Empty)
                };
            } else {
                return if b >= 0.0 {
                    Ok(IntervalSolution::AllReals)
                } else {
                    Ok(IntervalSolution::Empty)
                };
            }
        } else {
            if is_strict {
                return if b < 0.0 {
                    Ok(IntervalSolution::AllReals)
                } else {
                    Ok(IntervalSolution::Empty)
                };
            } else {
                return if b <= 0.0 {
                    Ok(IntervalSolution::AllReals)
                } else {
                    Ok(IntervalSolution::Empty)
                };
            }
        }
    } else {
        -b / a
    };

    let threshold = Expression::Float(neg_b_over_a);

    // Flip inequality direction if a < 0
    let flip = a < 0.0;
    let effective_greater = if flip { !is_greater } else { is_greater };

    if effective_greater {
        if is_strict {
            Ok(IntervalSolution::greater_than(threshold))
        } else {
            Ok(IntervalSolution::greater_equal(threshold))
        }
    } else {
        if is_strict {
            Ok(IntervalSolution::less_than(threshold))
        } else {
            Ok(IntervalSolution::less_equal(threshold))
        }
    }
}

/// Extract linear coefficients (a, b) from expression ax + b.
fn extract_linear_coefficients(
    expr: &Expression,
    var: &str,
) -> Result<(f64, f64), InequalityError> {
    // Simplified implementation - evaluate at x=0 to get b, at x=1 to get a+b
    let mut vars = HashMap::new();

    vars.insert(var.to_string(), 0.0);
    let b = expr
        .evaluate(&vars)
        .ok_or_else(|| InequalityError::CannotSolve("Cannot evaluate at x=0".to_string()))?;

    vars.insert(var.to_string(), 1.0);
    let a_plus_b = expr
        .evaluate(&vars)
        .ok_or_else(|| InequalityError::CannotSolve("Cannot evaluate at x=1".to_string()))?;

    Ok((a_plus_b - b, b))
}

/// Solve a quadratic inequality ax² + bx + c <op> 0.
fn solve_quadratic_inequality(
    expr: &Expression,
    var: &str,
    is_strict: bool,
    is_greater: bool,
) -> InequalityResult {
    // Extract coefficients
    let (a, b, c) = extract_quadratic_coefficients(expr, var)?;

    if a.abs() < 1e-15 {
        // Actually linear
        let linear_expr = Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Float(b)),
                Box::new(Expression::Variable(Variable::new(var))),
            )),
            Box::new(Expression::Float(c)),
        );
        return solve_linear_inequality(&linear_expr, var, is_strict, is_greater);
    }

    let discriminant = b * b - 4.0 * a * c;

    if discriminant < -1e-15 {
        // No real roots
        // ax² + bx + c has constant sign (same as a)
        let parabola_positive = a > 0.0;

        if is_greater {
            if parabola_positive {
                Ok(IntervalSolution::AllReals)
            } else {
                Ok(IntervalSolution::Empty)
            }
        } else {
            if parabola_positive {
                Ok(IntervalSolution::Empty)
            } else {
                Ok(IntervalSolution::AllReals)
            }
        }
    } else if discriminant.abs() < 1e-15 {
        // One double root
        let root = -b / (2.0 * a);
        let root_expr = Expression::Float(root);
        let parabola_positive = a > 0.0;

        // Parabola touches x-axis at root
        if is_greater {
            if is_strict {
                // > 0: all x except the root
                if parabola_positive {
                    Ok(IntervalSolution::Union(vec![
                        IntervalSolution::less_than(root_expr.clone()),
                        IntervalSolution::greater_than(root_expr),
                    ]))
                } else {
                    Ok(IntervalSolution::Empty)
                }
            } else {
                // >= 0
                if parabola_positive {
                    Ok(IntervalSolution::AllReals)
                } else {
                    // Only at the single point x = root
                    Ok(IntervalSolution::closed_interval(
                        root_expr.clone(),
                        root_expr,
                    ))
                }
            }
        } else {
            if is_strict {
                // < 0
                if parabola_positive {
                    Ok(IntervalSolution::Empty)
                } else {
                    Ok(IntervalSolution::Union(vec![
                        IntervalSolution::less_than(root_expr.clone()),
                        IntervalSolution::greater_than(root_expr),
                    ]))
                }
            } else {
                // <= 0
                if parabola_positive {
                    Ok(IntervalSolution::closed_interval(
                        root_expr.clone(),
                        root_expr,
                    ))
                } else {
                    Ok(IntervalSolution::AllReals)
                }
            }
        }
    } else {
        // Two distinct real roots
        let sqrt_disc = discriminant.sqrt();
        let r1 = (-b - sqrt_disc) / (2.0 * a);
        let r2 = (-b + sqrt_disc) / (2.0 * a);

        // Ensure r1 < r2
        let (root1, root2) = if r1 < r2 { (r1, r2) } else { (r2, r1) };
        let root1_expr = Expression::Float(root1);
        let root2_expr = Expression::Float(root2);

        let parabola_positive_outside = a > 0.0;

        // Test intervals: (-∞, r1), (r1, r2), (r2, +∞)
        // For a > 0: positive outside roots, negative between
        // For a < 0: negative outside roots, positive between

        if is_greater {
            if parabola_positive_outside {
                // > 0 outside roots
                if is_strict {
                    Ok(IntervalSolution::Union(vec![
                        IntervalSolution::less_than(root1_expr),
                        IntervalSolution::greater_than(root2_expr),
                    ]))
                } else {
                    Ok(IntervalSolution::Union(vec![
                        IntervalSolution::less_equal(root1_expr),
                        IntervalSolution::greater_equal(root2_expr),
                    ]))
                }
            } else {
                // > 0 between roots
                if is_strict {
                    Ok(IntervalSolution::open_interval(root1_expr, root2_expr))
                } else {
                    Ok(IntervalSolution::closed_interval(root1_expr, root2_expr))
                }
            }
        } else {
            if parabola_positive_outside {
                // < 0 between roots
                if is_strict {
                    Ok(IntervalSolution::open_interval(root1_expr, root2_expr))
                } else {
                    Ok(IntervalSolution::closed_interval(root1_expr, root2_expr))
                }
            } else {
                // < 0 outside roots
                if is_strict {
                    Ok(IntervalSolution::Union(vec![
                        IntervalSolution::less_than(root1_expr),
                        IntervalSolution::greater_than(root2_expr),
                    ]))
                } else {
                    Ok(IntervalSolution::Union(vec![
                        IntervalSolution::less_equal(root1_expr),
                        IntervalSolution::greater_equal(root2_expr),
                    ]))
                }
            }
        }
    }
}

/// Extract quadratic coefficients (a, b, c) from ax² + bx + c.
fn extract_quadratic_coefficients(
    expr: &Expression,
    var: &str,
) -> Result<(f64, f64, f64), InequalityError> {
    // Evaluate at three points to determine coefficients
    let mut vars = HashMap::new();

    vars.insert(var.to_string(), 0.0);
    let f0 = expr
        .evaluate(&vars)
        .ok_or_else(|| InequalityError::CannotSolve("Cannot evaluate at x=0".to_string()))?;

    vars.insert(var.to_string(), 1.0);
    let f1 = expr
        .evaluate(&vars)
        .ok_or_else(|| InequalityError::CannotSolve("Cannot evaluate at x=1".to_string()))?;

    vars.insert(var.to_string(), -1.0);
    let f_1 = expr
        .evaluate(&vars)
        .ok_or_else(|| InequalityError::CannotSolve("Cannot evaluate at x=-1".to_string()))?;

    // f(0) = c
    // f(1) = a + b + c
    // f(-1) = a - b + c
    // So: c = f(0)
    //     a + b = f(1) - c
    //     a - b = f(-1) - c
    //     a = ((f(1) - c) + (f(-1) - c)) / 2
    //     b = ((f(1) - c) - (f(-1) - c)) / 2

    let c = f0;
    let a = ((f1 - c) + (f_1 - c)) / 2.0;
    let b = ((f1 - c) - (f_1 - c)) / 2.0;

    Ok((a, b, c))
}

/// Solve a system of inequalities (conjunction).
///
/// Returns the intersection of all solution sets.
pub fn solve_system(inequalities: &[Inequality], var: &str) -> InequalityResult {
    if inequalities.is_empty() {
        return Ok(IntervalSolution::AllReals);
    }

    // Solve each inequality
    let mut solutions = Vec::new();
    for ineq in inequalities {
        solutions.push(solve_inequality(ineq, var)?);
    }

    // Intersect all solutions
    let mut result = solutions[0].clone();
    for sol in &solutions[1..] {
        result = intersect_intervals(&result, sol);
    }

    Ok(result)
}

/// Intersect two interval solutions.
fn intersect_intervals(a: &IntervalSolution, b: &IntervalSolution) -> IntervalSolution {
    match (a, b) {
        (IntervalSolution::Empty, _) | (_, IntervalSolution::Empty) => IntervalSolution::Empty,
        (IntervalSolution::AllReals, other) | (other, IntervalSolution::AllReals) => other.clone(),
        (IntervalSolution::Union(intervals_a), _) => {
            // Intersect each interval in a with b
            let mut results = Vec::new();
            for int_a in intervals_a {
                let intersection = intersect_intervals(int_a, b);
                if !matches!(intersection, IntervalSolution::Empty) {
                    results.push(intersection);
                }
            }
            if results.is_empty() {
                IntervalSolution::Empty
            } else if results.len() == 1 {
                results.pop().unwrap()
            } else {
                IntervalSolution::Union(results)
            }
        }
        (_, IntervalSolution::Union(intervals_b)) => {
            // Intersect a with each interval in b
            let mut results = Vec::new();
            for int_b in intervals_b {
                let intersection = intersect_intervals(a, int_b);
                if !matches!(intersection, IntervalSolution::Empty) {
                    results.push(intersection);
                }
            }
            if results.is_empty() {
                IntervalSolution::Empty
            } else if results.len() == 1 {
                results.pop().unwrap()
            } else {
                IntervalSolution::Union(results)
            }
        }
        (
            IntervalSolution::Interval {
                lower: l1,
                lower_inclusive: li1,
                upper: u1,
                upper_inclusive: ui1,
            },
            IntervalSolution::Interval {
                lower: l2,
                lower_inclusive: li2,
                upper: u2,
                upper_inclusive: ui2,
            },
        ) => {
            // Find the intersection of two single intervals
            // This is a simplified implementation for common cases
            let (new_lower, new_li) = max_bound(l1, *li1, l2, *li2);
            let (new_upper, new_ui) = min_bound(u1, *ui1, u2, *ui2);

            // Check if valid interval
            if is_valid_interval(&new_lower, &new_upper) {
                IntervalSolution::Interval {
                    lower: new_lower,
                    lower_inclusive: new_li,
                    upper: new_upper,
                    upper_inclusive: new_ui,
                }
            } else {
                IntervalSolution::Empty
            }
        }
    }
}

/// Evaluate an expression with no variables.
fn eval_constant(expr: &Expression) -> Option<f64> {
    let empty: HashMap<String, f64> = HashMap::new();
    expr.evaluate(&empty)
}

/// Get the maximum of two lower bounds.
fn max_bound(b1: &Bound, inc1: bool, b2: &Bound, inc2: bool) -> (Bound, bool) {
    match (b1, b2) {
        (Bound::NegativeInfinity, _) => (b2.clone(), inc2),
        (_, Bound::NegativeInfinity) => (b1.clone(), inc1),
        (Bound::PositiveInfinity, _) | (_, Bound::PositiveInfinity) => {
            (Bound::PositiveInfinity, false)
        }
        (Bound::Value(e1), Bound::Value(e2)) => {
            // Try to compare numerically
            match (eval_constant(e1), eval_constant(e2)) {
                (Some(v1), Some(v2)) => {
                    if v1 > v2 {
                        (b1.clone(), inc1)
                    } else if v2 > v1 {
                        (b2.clone(), inc2)
                    } else {
                        // Equal - take less inclusive
                        (b1.clone(), inc1 && inc2)
                    }
                }
                _ => (b1.clone(), inc1), // Fallback
            }
        }
    }
}

/// Get the minimum of two upper bounds.
fn min_bound(b1: &Bound, inc1: bool, b2: &Bound, inc2: bool) -> (Bound, bool) {
    match (b1, b2) {
        (Bound::PositiveInfinity, _) => (b2.clone(), inc2),
        (_, Bound::PositiveInfinity) => (b1.clone(), inc1),
        (Bound::NegativeInfinity, _) | (_, Bound::NegativeInfinity) => {
            (Bound::NegativeInfinity, false)
        }
        (Bound::Value(e1), Bound::Value(e2)) => match (eval_constant(e1), eval_constant(e2)) {
            (Some(v1), Some(v2)) => {
                if v1 < v2 {
                    (b1.clone(), inc1)
                } else if v2 < v1 {
                    (b2.clone(), inc2)
                } else {
                    (b1.clone(), inc1 && inc2)
                }
            }
            _ => (b1.clone(), inc1),
        },
    }
}

/// Check if an interval [lower, upper] is valid (lower < upper).
fn is_valid_interval(lower: &Bound, upper: &Bound) -> bool {
    match (lower, upper) {
        (Bound::NegativeInfinity, _) => true,
        (_, Bound::PositiveInfinity) => true,
        (Bound::PositiveInfinity, _) | (_, Bound::NegativeInfinity) => false,
        (Bound::Value(l), Bound::Value(u)) => {
            match (eval_constant(l), eval_constant(u)) {
                (Some(vl), Some(vu)) => vl <= vu,
                _ => true, // Assume valid if can't compare
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn add(l: Expression, r: Expression) -> Expression {
        Expression::Binary(BinaryOp::Add, Box::new(l), Box::new(r))
    }

    fn sub(l: Expression, r: Expression) -> Expression {
        Expression::Binary(BinaryOp::Sub, Box::new(l), Box::new(r))
    }

    fn mul(l: Expression, r: Expression) -> Expression {
        Expression::Binary(BinaryOp::Mul, Box::new(l), Box::new(r))
    }

    fn pow(base: Expression, exp: Expression) -> Expression {
        Expression::Power(Box::new(base), Box::new(exp))
    }

    #[test]
    fn test_linear_greater_than() {
        // 2x + 3 > 7  =>  2x > 4  =>  x > 2
        let lhs = add(mul(int(2), var("x")), int(3));
        let ineq = Inequality::GreaterThan(lhs, int(7));

        let solution = solve_inequality(&ineq, "x").unwrap();

        // Should be (2, +∞)
        if let IntervalSolution::Interval {
            lower,
            lower_inclusive,
            upper,
            upper_inclusive,
        } = solution
        {
            assert!(!lower_inclusive);
            assert!(!upper_inclusive);
            assert!(matches!(upper, Bound::PositiveInfinity));
            if let Bound::Value(e) = lower {
                let val = eval_constant(&e).unwrap();
                assert!((val - 2.0).abs() < 1e-10);
            } else {
                panic!("Expected Value bound");
            }
        } else {
            panic!("Expected Interval solution");
        }
    }

    #[test]
    fn test_quadratic_less_than() {
        // x^2 - 4 < 0  =>  -2 < x < 2
        let x_sq = pow(var("x"), int(2));
        let lhs = sub(x_sq, int(4));
        let ineq = Inequality::LessThan(lhs, int(0));

        let solution = solve_inequality(&ineq, "x").unwrap();

        // Should be (-2, 2)
        if let IntervalSolution::Interval {
            lower,
            lower_inclusive,
            upper,
            upper_inclusive,
        } = solution
        {
            assert!(!lower_inclusive);
            assert!(!upper_inclusive);
            if let (Bound::Value(l), Bound::Value(u)) = (lower, upper) {
                let vl = eval_constant(&l).unwrap();
                let vu = eval_constant(&u).unwrap();
                assert!((vl - (-2.0)).abs() < 1e-10);
                assert!((vu - 2.0).abs() < 1e-10);
            }
        } else {
            panic!("Expected Interval solution");
        }
    }

    #[test]
    fn test_quadratic_greater_equal() {
        // x^2 - 4 >= 0  =>  x <= -2 OR x >= 2
        let x_sq = pow(var("x"), int(2));
        let lhs = sub(x_sq, int(4));
        let ineq = Inequality::GreaterEqual(lhs, int(0));

        let solution = solve_inequality(&ineq, "x").unwrap();

        // Should be union: (-∞, -2] ∪ [2, +∞)
        assert!(matches!(solution, IntervalSolution::Union(_)));
    }

    #[test]
    fn test_linear_flip_sign() {
        // -x + 3 > 0  =>  x < 3
        let lhs = add(Expression::Unary(UnaryOp::Neg, Box::new(var("x"))), int(3));
        let ineq = Inequality::GreaterThan(lhs, int(0));

        let solution = solve_inequality(&ineq, "x").unwrap();

        // Should be (-∞, 3)
        if let IntervalSolution::Interval {
            lower,
            upper,
            lower_inclusive,
            upper_inclusive,
        } = solution
        {
            assert!(matches!(lower, Bound::NegativeInfinity));
            assert!(!lower_inclusive);
            assert!(!upper_inclusive);
            if let Bound::Value(u) = upper {
                let vu = eval_constant(&u).unwrap();
                assert!((vu - 3.0).abs() < 1e-10);
            }
        } else {
            panic!("Expected Interval solution");
        }
    }

    #[test]
    fn test_system_intersection() {
        // x > 0 AND x < 5  =>  (0, 5)
        let ineq1 = Inequality::GreaterThan(var("x"), int(0));
        let ineq2 = Inequality::LessThan(var("x"), int(5));

        let solution = solve_system(&[ineq1, ineq2], "x").unwrap();

        // Should be (0, 5)
        if let IntervalSolution::Interval {
            lower,
            upper,
            lower_inclusive,
            upper_inclusive,
        } = solution
        {
            assert!(!lower_inclusive);
            assert!(!upper_inclusive);
            if let (Bound::Value(l), Bound::Value(u)) = (lower, upper) {
                let vl = eval_constant(&l).unwrap();
                let vu = eval_constant(&u).unwrap();
                assert!((vl - 0.0).abs() < 1e-10);
                assert!((vu - 5.0).abs() < 1e-10);
            }
        } else {
            panic!("Expected Interval solution: {:?}", solution);
        }
    }

    #[test]
    fn test_no_solution() {
        // x^2 + 1 < 0 has no real solutions
        let x_sq = pow(var("x"), int(2));
        let lhs = add(x_sq, int(1));
        let ineq = Inequality::LessThan(lhs, int(0));

        let solution = solve_inequality(&ineq, "x").unwrap();
        assert!(matches!(solution, IntervalSolution::Empty));
    }

    #[test]
    fn test_all_reals_solution() {
        // x^2 + 1 > 0 is always true
        let x_sq = pow(var("x"), int(2));
        let lhs = add(x_sq, int(1));
        let ineq = Inequality::GreaterThan(lhs, int(0));

        let solution = solve_inequality(&ineq, "x").unwrap();
        assert!(matches!(solution, IntervalSolution::AllReals));
    }
}
