//! ODE solver adapter implementing the [`Solver`] trait for differential equations.
//!
//! This module bridges the ODE-specific solving functions in [`crate::ode`] with
//! the generic [`Solver`] trait used throughout this crate.  It also exposes
//! convenience functions for callers that already hold a typed ODE struct and do
//! not need to go through the [`SmartSolver`](crate::solver::SmartSolver) dispatch
//! path.
//!
//! # Limitations
//!
//! The AST does not have an `Expression::Derivative` variant, so ODEs cannot be
//! parsed from a plain [`Equation`].  Consequently:
//!
//! - [`OdeSolver::can_solve`] always returns `false`.
//! - [`OdeSolver::solve`] always returns
//!   [`SolverError::UnsupportedEquationType`].
//!
//! Use [`solve_ode_first_order`] or [`solve_ode_second_order`] when you already
//! have an [`FirstOrderODE`] or [`SecondOrderODE`] struct.
//!
//! # Examples
//!
//! ```rust
//! use thales::ast::{BinaryOp, Expression, Variable};
//! use thales::ode::FirstOrderODE;
//! use thales::solver::ode_solver::solve_ode_first_order;
//! use thales::solver::Solution;
//!
//! // dy/dx = y  →  y = C·eˣ
//! let ode = FirstOrderODE::new(
//!     "y", "x",
//!     Expression::Variable(Variable::new("y")),
//! );
//! let (solution, path) = solve_ode_first_order(&ode).unwrap();
//! assert!(matches!(solution, Solution::Unique(_)));
//! assert!(!path.steps.is_empty());
//! ```

use crate::ast::{Equation, Variable};
use crate::ode::{
    solve_linear, solve_second_order_homogeneous, solve_separable, FirstOrderODE, SecondOrderODE,
};
use crate::resolution_path::{Operation, ResolutionPath, ResolutionStep};
use crate::solver::ode_classifier::{classify_first_order, ODEType};
use crate::solver::types::{Solution, SolverError, SolverResult};
use crate::solver::Solver;

// ---------------------------------------------------------------------------
// OdeSolver struct
// ---------------------------------------------------------------------------

/// Solver adapter for ordinary differential equations.
///
/// This struct implements the [`Solver`] trait but cannot accept plain
/// [`Equation`] values because the AST has no derivative node.  It exists
/// primarily as a marker for the solver hierarchy; actual ODE solving is done
/// through [`solve_ode_first_order`] and [`solve_ode_second_order`].
#[derive(Debug, Default)]
pub struct OdeSolver;

impl OdeSolver {
    /// Create a new [`OdeSolver`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Solver for OdeSolver {
    /// Always returns `false` — ODEs cannot be detected from a plain
    /// [`Equation`] without a derivative AST node.
    fn can_solve(&self, _equation: &Equation) -> bool {
        false
    }

    /// Always returns [`SolverError::UnsupportedEquationType`].
    ///
    /// Use [`solve_ode_first_order`] or [`solve_ode_second_order`] instead.
    fn solve(
        &self,
        _equation: &Equation,
        _variable: &Variable,
    ) -> SolverResult<(Solution, ResolutionPath)> {
        Err(SolverError::UnsupportedEquationType)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a [`ResolutionPath`] from an ordered list of textual ODE steps.
fn build_ode_path(steps: &[String], solution_expr: &crate::ast::Expression) -> ResolutionPath {
    let mut path = ResolutionPath::new(solution_expr.clone());
    for step_desc in steps {
        path.add_step(ResolutionStep::new(
            Operation::SolveODE {
                method: step_desc.clone(),
            },
            step_desc.clone(),
            solution_expr.clone(),
        ));
    }
    path.set_result(solution_expr.clone());
    path
}

// ---------------------------------------------------------------------------
// Public convenience functions
// ---------------------------------------------------------------------------

/// Solve a first-order ODE directly, returning a [`Solution`] and a
/// [`ResolutionPath`] with the step-by-step explanation.
///
/// The ODE type is classified first (separable vs. linear) and the matching
/// solver is applied.  If the ODE does not fit either supported type the
/// function returns [`SolverError::CannotSolve`].
///
/// # Errors
///
/// Returns [`SolverError::CannotSolve`] when the ODE type is not yet
/// supported or when the underlying solver fails.
///
/// # Examples
///
/// ```rust
/// use thales::ast::{Expression, Variable};
/// use thales::ode::FirstOrderODE;
/// use thales::solver::ode_solver::solve_ode_first_order;
/// use thales::solver::Solution;
///
/// // dy/dx = y  (separable)
/// let ode = FirstOrderODE::new("y", "x", Expression::Variable(Variable::new("y")));
/// let (solution, _path) = solve_ode_first_order(&ode).unwrap();
/// assert!(matches!(solution, Solution::Unique(_)));
/// ```
pub fn solve_ode_first_order(ode: &FirstOrderODE) -> SolverResult<(Solution, ResolutionPath)> {
    let classification = classify_first_order(ode);

    // For ODEs classified as Separable but also linear, try separable first and
    // fall back to the integrating-factor method when integration fails.
    let ode_result = match classification.ode_type {
        ODEType::Separable => solve_separable(ode)
            .or_else(|_| solve_linear(ode))
            .map_err(|e| SolverError::CannotSolve(e.to_string())),
        ODEType::Linear => solve_linear(ode).map_err(|e| SolverError::CannotSolve(e.to_string())),
        _ => Err(SolverError::CannotSolve(
            "ODE type not yet supported".to_string(),
        )),
    }?;

    let path = build_ode_path(&ode_result.steps, &ode_result.general_solution);
    Ok((Solution::Unique(ode_result.general_solution), path))
}

/// Solve a second-order homogeneous ODE with constant coefficients directly.
///
/// Uses the characteristic equation method.  Non-homogeneous forcing terms are
/// not yet supported and will return [`SolverError::CannotSolve`].
///
/// # Errors
///
/// Returns [`SolverError::CannotSolve`] when:
/// - The ODE is non-homogeneous (forcing term ≠ 0).
/// - The characteristic equation solver fails.
///
/// # Examples
///
/// ```rust
/// use thales::ode::SecondOrderODE;
/// use thales::solver::ode_solver::solve_ode_second_order;
/// use thales::solver::Solution;
///
/// // y'' - y = 0  →  y = C₁·eˣ + C₂·e^(−x)
/// let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 0.0, -1.0);
/// let (solution, _path) = solve_ode_second_order(&ode).unwrap();
/// assert!(matches!(solution, Solution::Unique(_)));
/// ```
pub fn solve_ode_second_order(ode: &SecondOrderODE) -> SolverResult<(Solution, ResolutionPath)> {
    if !ode.is_homogeneous() {
        return Err(SolverError::CannotSolve(
            "Non-homogeneous second-order ODEs are not yet supported".to_string(),
        ));
    }

    let result =
        solve_second_order_homogeneous(ode).map_err(|e| SolverError::CannotSolve(e.to_string()))?;

    let path = build_ode_path(&result.steps, &result.general_solution);
    Ok((Solution::Unique(result.general_solution), path))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Expression, Variable};
    use crate::ode::{FirstOrderODE, SecondOrderODE};

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn neg(expr: Expression) -> Expression {
        use crate::ast::UnaryOp;
        Expression::Unary(UnaryOp::Neg, Box::new(expr))
    }

    // ------------------------------------------------------------------
    // OdeSolver trait tests
    // ------------------------------------------------------------------

    #[test]
    fn can_solve_always_false() {
        use crate::ast::Equation;
        let solver = OdeSolver::new();
        let eq = Equation::new("test", Expression::Integer(0), Expression::Integer(0));
        assert!(!solver.can_solve(&eq));
    }

    #[test]
    fn solve_trait_returns_unsupported() {
        use crate::ast::Equation;
        let solver = OdeSolver::new();
        let eq = Equation::new("test", Expression::Integer(0), Expression::Integer(0));
        let result = solver.solve(&eq, &Variable::new("x"));
        assert!(matches!(result, Err(SolverError::UnsupportedEquationType)));
    }

    // ------------------------------------------------------------------
    // First-order ODE convenience function tests
    // ------------------------------------------------------------------

    #[test]
    fn solve_first_order_separable_dy_equals_y() {
        // dy/dx = y  →  separable, general solution y = C·eˣ
        let ode = FirstOrderODE::new("y", "x", var("y"));
        let result = solve_ode_first_order(&ode);
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
        let (solution, path) = result.unwrap();
        assert!(matches!(solution, Solution::Unique(_)));
        assert!(!path.steps.is_empty());
    }

    #[test]
    fn solve_first_order_linear_dy_equals_minus_y() {
        // dy/dx = -y  →  linear (P = 1, Q = 0), solution y = C·e^(−x)
        let ode = FirstOrderODE::new("y", "x", neg(var("y")));
        let result = solve_ode_first_order(&ode);
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
        let (solution, path) = result.unwrap();
        assert!(matches!(solution, Solution::Unique(_)));
        assert!(!path.steps.is_empty());
    }

    // ------------------------------------------------------------------
    // Second-order ODE convenience function tests
    // ------------------------------------------------------------------

    #[test]
    fn solve_second_order_homogeneous_y_pp_minus_y() {
        // y'' - y = 0  →  y = C₁·eˣ + C₂·e^(−x)
        let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 0.0, -1.0);
        let result = solve_ode_second_order(&ode);
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
        let (solution, path) = result.unwrap();
        assert!(matches!(solution, Solution::Unique(_)));
        assert!(!path.steps.is_empty());
    }

    #[test]
    fn solve_second_order_non_homogeneous_returns_error() {
        // y'' + y = x  →  non-homogeneous, not yet supported
        let ode = SecondOrderODE::new("y", "x", 1.0, 0.0, 1.0, var("x"));
        let result = solve_ode_second_order(&ode);
        assert!(
            matches!(result, Err(SolverError::CannotSolve(_))),
            "Expected CannotSolve, got {result:?}"
        );
    }
}
