//! Types for ODE solving.

use crate::ast::Expression;
use crate::integration::IntegrationError;
use crate::resolution_path::ResolutionPath;

use super::first_order::{extract_linear_coefficients, try_separate};

/// Error types for ODE solving
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ODEError {
    /// The equation is not in the expected form
    NotInExpectedForm(String),
    /// Cannot solve this type of ODE
    CannotSolve(String),
    /// Integration failed during solving
    IntegrationFailed(IntegrationError),
    /// Initial condition cannot be applied
    InitialConditionError(String),
    /// The ODE is not separable
    NotSeparable,
    /// The ODE is not linear
    NotLinear,
    /// Characteristic equation solving failed
    CharacteristicEquationError(String),
    /// Coefficients are not constant (depend on independent variable)
    NonConstantCoefficients(String),
    /// Boundary value problem error
    BoundaryValueError(String),
    /// Resonance detected in particular solution
    ResonanceDetected(String),
}

impl From<IntegrationError> for ODEError {
    fn from(e: IntegrationError) -> Self {
        ODEError::IntegrationFailed(e)
    }
}

impl std::fmt::Display for ODEError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ODEError::NotInExpectedForm(msg) => write!(f, "ODE not in expected form: {}", msg),
            ODEError::CannotSolve(msg) => write!(f, "Cannot solve ODE: {}", msg),
            ODEError::IntegrationFailed(e) => write!(f, "Integration failed: {}", e),
            ODEError::InitialConditionError(msg) => {
                write!(f, "Initial condition error: {}", msg)
            }
            ODEError::NotSeparable => write!(f, "ODE is not separable"),
            ODEError::NotLinear => write!(f, "ODE is not first-order linear"),
            ODEError::CharacteristicEquationError(msg) => {
                write!(f, "Characteristic equation error: {}", msg)
            }
            ODEError::NonConstantCoefficients(msg) => {
                write!(f, "Non-constant coefficients: {}", msg)
            }
            ODEError::BoundaryValueError(msg) => write!(f, "Boundary value error: {}", msg),
            ODEError::ResonanceDetected(msg) => write!(f, "Resonance detected: {}", msg),
        }
    }
}

impl std::error::Error for ODEError {}

/// Represents a first-order ordinary differential equation: dy/dx = f(x, y)
#[derive(Debug, Clone)]
pub struct FirstOrderODE {
    /// The dependent variable (e.g., "y")
    pub dependent: String,
    /// The independent variable (e.g., "x")
    pub independent: String,
    /// The right-hand side expression f(x, y) where dy/dx = f(x, y)
    pub rhs: Expression,
}

impl FirstOrderODE {
    /// Create a new first-order ODE.
    ///
    /// # Arguments
    ///
    /// * `dependent` - The dependent variable name (e.g., "y")
    /// * `independent` - The independent variable name (e.g., "x")
    /// * `rhs` - The expression f(x, y) such that dy/dx = f(x, y)
    pub fn new(dependent: &str, independent: &str, rhs: Expression) -> Self {
        Self {
            dependent: dependent.to_string(),
            independent: independent.to_string(),
            rhs,
        }
    }

    /// Check if this ODE is separable (can be written as g(x) * h(y)).
    pub fn is_separable(&self) -> bool {
        try_separate(&self.rhs, &self.independent, &self.dependent).is_some()
    }

    /// Check if this ODE is first-order linear (dy/dx + P(x)*y = Q(x)).
    pub fn is_linear(&self) -> bool {
        extract_linear_coefficients(&self.rhs, &self.independent, &self.dependent).is_some()
    }
}

/// Result of solving an ODE
#[derive(Debug, Clone)]
pub struct ODESolution {
    /// The general solution expression (may contain constant C)
    pub general_solution: Expression,
    /// Description of the solution method used
    pub method: String,
    /// Solution steps for educational output
    pub steps: Vec<String>,
}
