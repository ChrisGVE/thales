//! Types for partial fraction decomposition.

use crate::ast::Expression;
use std::fmt;

/// Error types for partial fraction decomposition.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DecomposeError {
    /// The expression is not a rational function.
    NotRational(String),
    /// Cannot factor the denominator.
    CannotFactor(String),
    /// Cannot solve for coefficients.
    CoefficientError(String),
    /// The denominator has degree less than numerator (need polynomial division first).
    ImproperFraction(String),
    /// Division by zero would occur.
    DivisionByZero,
}

impl fmt::Display for DecomposeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecomposeError::NotRational(msg) => write!(f, "Not a rational function: {}", msg),
            DecomposeError::CannotFactor(msg) => write!(f, "Cannot factor denominator: {}", msg),
            DecomposeError::CoefficientError(msg) => {
                write!(f, "Cannot solve for coefficients: {}", msg)
            }
            DecomposeError::ImproperFraction(msg) => write!(f, "Improper fraction: {}", msg),
            DecomposeError::DivisionByZero => write!(f, "Division by zero"),
        }
    }
}

impl std::error::Error for DecomposeError {}

/// A term in a partial fraction decomposition.
#[derive(Debug, Clone, PartialEq)]
pub enum PartialFractionTerm {
    /// A/(x-a)^n - linear factor term
    Linear {
        /// Coefficient A
        coefficient: f64,
        /// Root a (where x-a is the factor)
        root: f64,
        /// Power n
        power: u32,
    },
    /// (Ax+B)/(x²+px+q)^n - irreducible quadratic term
    Quadratic {
        /// Coefficient A (for x term)
        a_coeff: f64,
        /// Coefficient B (constant term)
        b_coeff: f64,
        /// p coefficient in x²+px+q
        p: f64,
        /// q coefficient in x²+px+q
        q: f64,
        /// Power n
        power: u32,
    },
    /// A polynomial term (when numerator degree >= denominator degree)
    Polynomial(Expression),
}

/// Result of partial fraction decomposition.
#[derive(Debug, Clone)]
pub struct PartialFractionResult {
    /// The decomposed terms.
    pub terms: Vec<PartialFractionTerm>,
    /// The variable of decomposition.
    pub variable: String,
    /// Steps taken during decomposition (for resolution path).
    pub steps: Vec<String>,
}
