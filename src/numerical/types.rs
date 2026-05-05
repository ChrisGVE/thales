//! Types and configuration for numerical solvers.

use crate::ast::{Expression, Variable};
use std::collections::HashMap;

/// Error types for numerical solving.
///
/// Represents the various failure modes that can occur during numerical
/// root-finding and optimization.
///
/// # Variants
///
/// * `NoConvergence` - The algorithm did not converge within the maximum number
///   of iterations. This can happen if the initial guess is too far from the root,
///   the function is poorly behaved, or the tolerance is too tight.
///
/// * `Unstable` - Numerical instability was detected, such as division by zero
///   (when derivative is zero in Newton-Raphson), NaN values, or infinite values.
///
/// * `InvalidInitialGuess` - The provided initial guess is invalid (e.g., outside
///   the valid domain of the function).
///
/// * `EvaluationFailed` - Function or derivative evaluation failed at a specific
///   point. The string contains details about the failure.
///
/// * `Other` - Any other error with a descriptive message.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum NumericalError {
    /// Failed to converge within iteration limit
    NoConvergence,
    /// Numerical instability detected (zero derivative, NaN, infinity)
    Unstable,
    /// Invalid initial guess
    InvalidInitialGuess,
    /// Function evaluation failed
    EvaluationFailed(String),
    /// Other error
    Other(String),
}

impl std::fmt::Display for NumericalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NumericalError::NoConvergence => write!(f, "Failed to converge within iteration limit"),
            NumericalError::Unstable => write!(f, "Numerical instability detected"),
            NumericalError::InvalidInitialGuess => write!(f, "Invalid initial guess"),
            NumericalError::EvaluationFailed(msg) => {
                write!(f, "Function evaluation failed: {}", msg)
            }
            NumericalError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for NumericalError {}

/// Result type for numerical operations.
pub type NumericalResult<T> = Result<T, NumericalError>;

/// Configuration for numerical solvers.
///
/// Controls the behavior and termination criteria for numerical root-finding
/// algorithms.
///
/// # Fields
///
/// * `max_iterations` - Maximum number of iterations before giving up.
///   Default: 1000
///
/// * `tolerance` - Convergence tolerance. The algorithm stops when either:
///   - The residual |f(x)| < tolerance, or
///   - The step size |Δx| < tolerance
///   Default: 1e-10
///
/// * `initial_guess` - Starting point for the algorithm. If `None`, the solver
///   will attempt to estimate a reasonable starting point. For Newton-Raphson,
///   a good initial guess close to the actual root leads to faster convergence.
///   Default: None (will use 1.0)
///
/// * `step_size` - Step size for finite difference derivative approximation.
///   This is only used as a fallback; the Newton-Raphson solver primarily uses
///   symbolic differentiation for exact derivatives.
///   Default: 1e-6
///
/// # Example
///
/// ```
/// use thales::numerical::NumericalConfig;
///
/// // Use default configuration
/// let config = NumericalConfig::default();
///
/// // Custom configuration for high precision
/// let precise_config = NumericalConfig {
///     max_iterations: 10000,
///     tolerance: 1e-15,
///     initial_guess: Some(2.0),
///     step_size: 1e-8,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct NumericalConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Tolerance for convergence
    pub tolerance: f64,
    /// Initial guess (if not provided, will be estimated)
    pub initial_guess: Option<f64>,
    /// Step size for derivative approximation
    pub step_size: f64,
}

impl Default for NumericalConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-10,
            initial_guess: None,
            step_size: 1e-6,
        }
    }
}

/// Numerical solution with convergence information.
///
/// Contains the result of a numerical root-finding operation along with
/// diagnostic information about the convergence process.
///
/// # Fields
///
/// * `value` - The approximate solution (root) found by the algorithm
///
/// * `iterations` - Number of iterations performed before convergence or
///   termination
///
/// * `residual` - Final residual value |f(x)|. For a perfect solution, this
///   would be 0.0. In practice, it should be smaller than the configured
///   tolerance.
///
/// * `converged` - Whether the algorithm successfully converged to a solution
///   within the tolerance and iteration limits
///
/// # Example
///
/// ```
/// use thales::numerical::{NewtonRaphson, NumericalConfig};
/// use thales::ast::{Equation, Expression, Variable};
///
/// // Solve x^2 = 5
/// let equation = Equation::new(
///     "find_sqrt5",
///     Expression::Power(
///         Box::new(Expression::Variable(Variable::new("x"))),
///         Box::new(Expression::Integer(2))
///     ),
///     Expression::Integer(5)
/// );
///
/// let solver = NewtonRaphson::with_default_config();
/// let (solution, _path) = solver.solve(&equation, &Variable::new("x")).unwrap();
///
/// assert!(solution.converged);
/// assert!((solution.value - 2.236067977).abs() < 1e-6); // √5 ≈ 2.236
/// assert!(solution.residual < 1e-10);
/// println!("Solution found in {} iterations", solution.iterations);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct NumericalSolution {
    /// The approximate solution value
    pub value: f64,
    /// Number of iterations taken
    pub iterations: usize,
    /// Final residual (how close to zero)
    pub residual: f64,
    /// Whether convergence was achieved
    pub converged: bool,
}

/// Expression evaluator with variable substitution.
pub struct Evaluator {
    variables: HashMap<Variable, f64>,
}

impl Evaluator {
    /// Creates a new evaluator with no variables defined.
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    /// Creates a new evaluator with predefined variable values.
    ///
    /// # Arguments
    ///
    /// * `variables` - Map of variables to their numeric values
    pub fn with_variables(variables: HashMap<Variable, f64>) -> Self {
        Self { variables }
    }

    /// Sets or updates the value of a variable.
    ///
    /// # Arguments
    ///
    /// * `var` - Variable to set
    /// * `value` - Numeric value to assign
    pub fn set_variable(&mut self, var: Variable, value: f64) {
        self.variables.insert(var, value);
    }

    /// Evaluate an expression to a floating point value.
    pub fn evaluate(&self, _expression: &Expression) -> Result<f64, String> {
        // TODO: Implement expression evaluation
        // TODO: Handle all expression types (binary ops, functions, etc.)
        // TODO: Substitute variables from the variable map
        // TODO: Use fasteval for efficient evaluation
        Err("Not yet implemented".to_string())
    }
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new()
    }
}
