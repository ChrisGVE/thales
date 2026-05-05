//! Types for inequality solving.

use crate::ast::Expression;
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
