//! Types for precision-aware evaluation.

use crate::ast::Expression;
use num_rational::Rational64;
use std::fmt;

/// Error types for precision evaluation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum EvalError {
    /// Division by zero.
    DivisionByZero,
    /// Undefined variable.
    UndefinedVariable(String),
    /// Domain error (e.g., sqrt of negative).
    DomainError(String),
    /// Overflow in computation.
    Overflow,
    /// Cannot evaluate expression (e.g., contains unevaluable parts).
    CannotEvaluate(String),
    /// Invalid operation.
    InvalidOperation(String),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::DivisionByZero => write!(f, "Division by zero"),
            EvalError::UndefinedVariable(name) => write!(f, "Undefined variable: {}", name),
            EvalError::DomainError(msg) => write!(f, "Domain error: {}", msg),
            EvalError::Overflow => write!(f, "Overflow in computation"),
            EvalError::CannotEvaluate(msg) => write!(f, "Cannot evaluate: {}", msg),
            EvalError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
        }
    }
}

impl std::error::Error for EvalError {}

/// Result type for precision evaluation.
pub type EvalResult<T> = Result<T, EvalError>;

/// Precision mode for numerical evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrecisionMode {
    /// Fixed number of decimal places (e.g., 6 decimal places).
    FixedDecimal(u32),
    /// Fixed number of significant figures (e.g., 10 significant figures).
    SignificantFigures(u32),
    /// Arbitrary precision using exact rationals where possible.
    Arbitrary,
    /// Full floating-point precision (default f64).
    Full,
}

impl Default for PrecisionMode {
    fn default() -> Self {
        PrecisionMode::Full
    }
}

/// Rounding mode for precision operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoundingMode {
    /// Round half up (0.5 → 1, -0.5 → 0).
    HalfUp,
    /// Round half even (banker's rounding: 0.5 → 0, 1.5 → 2).
    HalfEven,
    /// Truncate toward zero.
    Truncate,
    /// Round toward positive infinity (ceiling).
    Ceiling,
    /// Round toward negative infinity (floor).
    Floor,
}

impl Default for RoundingMode {
    fn default() -> Self {
        RoundingMode::HalfEven
    }
}

/// Value representation with precision information.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Exact integer value.
    Integer(i64),
    /// Exact rational value (fraction).
    Rational(Rational64),
    /// Floating-point value.
    Float(f64),
    /// Complex value (real, imaginary).
    Complex(f64, f64),
    /// Positive infinity.
    PositiveInfinity,
    /// Negative infinity.
    NegativeInfinity,
    /// Not a number.
    NaN,
}

impl Value {
    /// Convert value to f64, if possible.
    pub fn as_f64(&self) -> f64 {
        match self {
            Value::Integer(n) => *n as f64,
            Value::Rational(r) => *r.numer() as f64 / *r.denom() as f64,
            Value::Float(f) => *f,
            Value::Complex(re, _) => *re, // Real part only
            Value::PositiveInfinity => f64::INFINITY,
            Value::NegativeInfinity => f64::NEG_INFINITY,
            Value::NaN => f64::NAN,
        }
    }

    /// Check if value is finite.
    pub fn is_finite(&self) -> bool {
        match self {
            Value::Integer(_) | Value::Rational(_) => true,
            Value::Float(f) => f.is_finite(),
            Value::Complex(re, im) => re.is_finite() && im.is_finite(),
            _ => false,
        }
    }

    /// Check if value is real (not complex with imaginary part).
    pub fn is_real(&self) -> bool {
        match self {
            Value::Complex(_, im) => im.abs() < 1e-15,
            _ => true,
        }
    }

    /// Check if value is NaN.
    pub fn is_nan(&self) -> bool {
        matches!(self, Value::NaN) || matches!(self, Value::Float(f) if f.is_nan())
    }

    /// Check if value is zero.
    pub fn is_zero(&self) -> bool {
        match self {
            Value::Integer(0) => true,
            Value::Rational(r) => *r.numer() == 0,
            Value::Float(f) => f.abs() < 1e-15,
            Value::Complex(re, im) => re.abs() < 1e-15 && im.abs() < 1e-15,
            _ => false,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Integer(n) => write!(f, "{}", n),
            Value::Rational(r) => write!(f, "{}/{}", r.numer(), r.denom()),
            Value::Float(x) => write!(f, "{}", x),
            Value::Complex(re, im) => {
                if im.abs() < 1e-15 {
                    write!(f, "{}", re)
                } else if re.abs() < 1e-15 {
                    write!(f, "{}i", im)
                } else if *im >= 0.0 {
                    write!(f, "{}+{}i", re, im)
                } else {
                    write!(f, "{}{}i", re, im)
                }
            }
            Value::PositiveInfinity => write!(f, "∞"),
            Value::NegativeInfinity => write!(f, "-∞"),
            Value::NaN => write!(f, "NaN"),
        }
    }
}
