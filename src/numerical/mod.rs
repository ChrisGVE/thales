//! Numerical approximation methods for equations.
//!
//! Provides numerical solvers for equations that cannot be solved symbolically,
//! using root-finding and optimization algorithms.
//!
//! # Methods
//!
//! ## Newton-Raphson Method
//! Fast convergence for smooth functions with good initial guesses.
//! Uses **symbolic differentiation** from the AST module for exact derivatives.
//!
//! ## Bisection Method
//! Guaranteed convergence for continuous functions when root is bracketed.
//! More robust but slower than Newton-Raphson.
//!
//! ## Smart Solver
//! Automatically selects the best method based on the problem characteristics.
//!
//! # Integration with Symbolic Differentiation
//!
//! The Newton-Raphson solver integrates with the symbolic differentiation
//! capability from `ast::Expression::differentiate()` (Task 188). This provides:
//!
//! - **Exact derivatives** instead of finite difference approximations
//! - **Faster convergence** due to precise derivative calculations
//! - **Better numerical stability** by avoiding finite difference errors
//! - **Clear resolution paths** showing the derivative expressions used
//!
//! # Example
//!
//! ```ignore
//! use thales::numerical::{NewtonRaphson, NumericalConfig};
//! use thales::ast::{Equation, Expression, Variable};
//!
//! // Solve x^2 = 5
//! let equation = Equation::new("quad",
//!     Expression::power(Expression::var("x"), 2),
//!     Expression::Integer(5));
//!
//! let solver = NewtonRaphson::with_default_config();
//! let (solution, path) = solver.solve(&equation, &Variable::new("x")).unwrap();
//! println!("x = {}", solution.value); // x ≈ 2.236
//! ```

mod bisection;
/// Bordered Hessian classification for constrained optimization.
pub mod bordered_hessian;
mod brent;
/// Lagrangian constrained optimization solver.
pub mod lagrangian;
mod least_squares;
mod newton;
/// Penalty method for constrained optimization.
pub mod penalty;
mod secant;
mod smart;
mod types;

// Re-export all public types
pub use bisection::BisectionMethod;
pub use brent::BrentsMethod;
pub use least_squares::{GradientDescent, LevenbergMarquardt};
pub use newton::NewtonRaphson;
pub use secant::SecantMethod;
pub use smart::SmartNumericalSolver;
pub use types::{Evaluator, NumericalConfig, NumericalError, NumericalResult, NumericalSolution};

use crate::ast::{Expression, Variable};
use std::collections::HashMap;

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute derivative using finite differences (central difference method).
///
/// f'(x) ≈ [f(x + h) - f(x - h)] / (2h)
///
/// Used by [`GradientDescent::minimize`] once that method is implemented (task 24).
#[allow(dead_code)]
fn compute_derivative_fd(
    expr: &Expression,
    variable: &Variable,
    x: f64,
    h: f64,
) -> NumericalResult<f64> {
    let mut vars = HashMap::new();

    // Evaluate f(x + h)
    vars.insert(variable.name.clone(), x + h);
    let f_plus = expr.evaluate(&vars).ok_or_else(|| {
        NumericalError::EvaluationFailed(format!("Failed to evaluate function at x = {}", x + h))
    })?;

    // Evaluate f(x - h)
    vars.insert(variable.name.clone(), x - h);
    let f_minus = expr.evaluate(&vars).ok_or_else(|| {
        NumericalError::EvaluationFailed(format!("Failed to evaluate function at x = {}", x - h))
    })?;

    // Central difference approximation
    let derivative = (f_plus - f_minus) / (2.0 * h);

    if !derivative.is_finite() {
        return Err(NumericalError::Unstable);
    }

    Ok(derivative)
}

/// Find a suitable initial interval for root finding by bracketing.
///
/// Searches for an interval [a, b] where the function changes sign, which
/// guarantees the existence of at least one root (by the Intermediate Value
/// Theorem for continuous functions).
///
/// # Algorithm
///
/// The function uses an expanding search strategy:
/// 1. Tries intervals of increasing size: 1, 10, 100, 1000 (up to max_range)
/// 2. For each size, tries different offsets from the center point
/// 3. Returns the first interval where f(a) and f(b) have opposite signs
///
/// This approach balances thoroughness with efficiency, checking common
/// ranges first before expanding to larger intervals.
///
/// # Arguments
///
/// * `expr` - The function expression to evaluate (should be in f(x) = 0 form)
/// * `variable` - The variable to solve for
/// * `center` - The center point around which to search for a bracket
/// * `max_range` - Maximum distance from center to search (prevents unbounded searches)
///
/// # Returns
///
/// * `Some((a, b))` - An interval where f(a) and f(b) have opposite signs
/// * `None` - No bracketing interval found within the search range
///
/// # Usage
///
/// This helper function is used internally by `SmartNumericalSolver` to
/// automatically find suitable intervals for bisection when the user hasn't
/// provided an initial guess or interval. It's particularly useful for:
/// - Unknown function behavior
/// - Automated solving without manual interval specification
/// - Fallback when Newton-Raphson fails
///
/// # Example Context
///
/// ```ignore
/// // Used internally by SmartNumericalSolver
/// let f = Expression::Binary(BinaryOp::Sub, lhs, rhs); // f(x) = 0 form
/// if let Some((a, b)) = bracket_root(&f, &var, 1.0, 10000.0) {
///     // Found bracketing interval, can use bisection method
///     let bisection = BisectionMethod::new(config);
///     bisection.solve(equation, variable, (a, b))
/// }
/// ```
pub(crate) fn bracket_root(
    expr: &Expression,
    variable: &Variable,
    center: f64,
    max_range: f64,
) -> Option<(f64, f64)> {
    let mut vars = HashMap::new();

    // Try expanding intervals around the center
    for scale in [1.0_f64, 10.0, 100.0, 1000.0] {
        let range = scale.min(max_range);

        for offset in [0.0, range / 4.0, range / 2.0, 3.0 * range / 4.0] {
            let a = center - range + offset;
            let b = center + range - offset;

            vars.insert(variable.name.clone(), a);
            let fa = expr.evaluate(&vars)?;

            vars.insert(variable.name.clone(), b);
            let fb = expr.evaluate(&vars)?;

            if fa * fb < 0.0 {
                return Some((a, b));
            }
        }
    }

    None
}

// TODO: Add support for interval arithmetic
// TODO: Add automatic differentiation
// TODO: Add support for complex-valued functions
// TODO: Add parallel evaluation for gradient computation
// TODO: Add sensitivity analysis
// TODO: Add uncertainty propagation
