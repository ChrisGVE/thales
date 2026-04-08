//! Types for limit computation.

use crate::ast::{Expression, SymbolicConstant};
use std::fmt;

pub(super) fn try_expr_to_f64(expr: &Expression) -> Option<f64> {
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
pub(super) const MAX_LHOPITAL_ITERATIONS: u32 = 10;

/// Error type for limit evaluation failures.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
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
        matches!(
            self,
            LimitPoint::PositiveInfinity | LimitPoint::NegativeInfinity
        )
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
        matches!(
            self,
            LimitResult::PositiveInfinity | LimitResult::NegativeInfinity
        )
    }
}
