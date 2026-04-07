//! Linear equation solver for equations of the form `ax + b = c`.

use crate::ast::{BinaryOp, Equation, Expression, Variable};
use crate::resolution_path::{Operation, ResolutionPath, ResolutionPathBuilder};

use super::helpers::{
    contains_variable, has_obvious_nonlinearity, is_linear_in_variable, isolate_variable,
};
use super::types::{Solution, SolverError, SolverResult};
use super::Solver;

/// Linear equation solver for equations of the form `ax + b = c`.
///
/// Solves first-degree polynomial equations in one variable by pattern matching
/// and algebraic manipulation. Handles various linear patterns including:
/// - Simple variable: `x = 5`
/// - Multiplication: `3x = 12`
/// - Addition: `x + 7 = 10`
/// - Combined: `2x + 3 = 11`
///
/// # Mathematical Foundation
///
/// A linear equation in one variable has the general form:
/// ```text
/// ax + b = c
/// ```
///
/// The solution is obtained by:
/// 1. Subtracting `b` from both sides: `ax = c - b`
/// 2. Dividing both sides by `a`: `x = (c - b) / a`
///
/// The solver recognizes these patterns automatically and applies the
/// appropriate transformations.
///
/// # Limitations
///
/// - Only handles linear equations (degree 1)
/// - Cannot solve equations with the variable in denominators (e.g., `1/x = 2`)
/// - Cannot solve equations with the variable in exponents (e.g., `2^x = 8`)
/// - Cannot handle products of variables (e.g., `x*y = 5`)
///
/// For more complex equations, use [`super::TranscendentalSolver`] or [`super::SmartSolver`].
///
/// # Examples
///
/// ## Simple Linear Equation
///
/// ```
/// use thales::solver::{LinearSolver, Solver};
/// use thales::ast::{Equation, Expression, Variable, BinaryOp};
///
/// // Solve: 2x + 3 = 11
/// let x = Expression::Variable(Variable::new("x"));
/// let two_x = Expression::Binary(
///     BinaryOp::Mul,
///     Box::new(Expression::Integer(2)),
///     Box::new(x),
/// );
/// let left = Expression::Binary(
///     BinaryOp::Add,
///     Box::new(two_x),
///     Box::new(Expression::Integer(3)),
/// );
/// let equation = Equation::new("linear", left, Expression::Integer(11));
///
/// let solver = LinearSolver::new();
/// let (solution, path) = solver.solve(&equation, &Variable::new("x")).unwrap();
///
/// // Verify solution: x = 4
/// # use thales::solver::Solution;
/// # use std::collections::HashMap;
/// # match solution {
/// #     Solution::Unique(expr) => {
/// #         assert_eq!(expr.evaluate(&HashMap::new()), Some(4.0));
/// #     }
/// #     _ => panic!("Expected unique solution"),
/// # }
/// ```
///
/// ## Equation with Parametric Coefficients
///
/// ```
/// use thales::solver::{LinearSolver, Solver};
/// use thales::ast::{Equation, Expression, Variable, BinaryOp};
/// use std::collections::HashMap;
///
/// // Solve: ax = b for x (symbolic)
/// let a = Expression::Variable(Variable::new("a"));
/// let x = Expression::Variable(Variable::new("x"));
/// let b = Expression::Variable(Variable::new("b"));
///
/// let left = Expression::Binary(BinaryOp::Mul, Box::new(a), Box::new(x));
/// let equation = Equation::new("parametric", left, b);
///
/// let solver = LinearSolver::new();
/// let (solution, _path) = solver.solve(&equation, &Variable::new("x")).unwrap();
///
/// // Solution is symbolic: x = b/a
/// # use thales::solver::Solution;
/// # match solution {
/// #     Solution::Unique(expr) => {
/// #         // Can substitute values later
/// #         let mut values = HashMap::new();
/// #         values.insert("a".to_string(), 3.0);
/// #         values.insert("b".to_string(), 12.0);
/// #         // Result would be 4.0
/// #     }
/// #     _ => panic!("Expected unique solution"),
/// # }
/// ```
///
/// ## Checking Solver Applicability
///
/// ```
/// use thales::solver::{LinearSolver, Solver};
/// use thales::ast::{Equation, Expression, Variable};
///
/// let solver = LinearSolver::new();
///
/// // Can solve linear equation
/// let linear = Equation::new(
///     "linear",
///     Expression::Variable(Variable::new("x")),
///     Expression::Integer(5),
/// );
/// assert!(solver.can_solve(&linear));
///
/// // Cannot solve quadratic equation
/// let x = Expression::Variable(Variable::new("x"));
/// let x_squared = Expression::Power(
///     Box::new(x),
///     Box::new(Expression::Integer(2)),
/// );
/// let quadratic = Equation::new("quadratic", x_squared, Expression::Integer(4));
/// assert!(!solver.can_solve(&quadratic));
/// ```
///
/// # See Also
///
/// - [`super::SmartSolver`]: Automatically selects LinearSolver for linear equations
/// - [`super::TranscendentalSolver`]: For equations with sin, cos, exp, ln, etc.
/// - [`super::solve_for`]: High-level API that uses SmartSolver and substitutes known values
#[derive(Debug, Default)]
pub struct LinearSolver;

impl LinearSolver {
    /// Create a new linear equation solver.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::solver::LinearSolver;
    ///
    /// let solver = LinearSolver::new();
    /// ```
    pub fn new() -> Self {
        Self
    }
}

impl Solver for LinearSolver {
    fn solve(
        &self,
        equation: &Equation,
        variable: &Variable,
    ) -> SolverResult<(Solution, ResolutionPath)> {
        let var_name = &variable.name;

        // Initialize resolution path
        let initial_expr = Expression::Binary(
            BinaryOp::Sub,
            Box::new(equation.left.clone()),
            Box::new(equation.right.clone()),
        );
        let mut path = ResolutionPathBuilder::new(initial_expr.clone());

        // Check if variable appears in equation
        let left_has_var = contains_variable(&equation.left, var_name);
        let right_has_var = contains_variable(&equation.right, var_name);

        if !left_has_var && !right_has_var {
            return Err(SolverError::CannotSolve(format!(
                "Variable '{}' not found in equation",
                var_name
            )));
        }

        // Check if equation is linear in the target variable
        if !is_linear_in_variable(&equation.left, var_name)
            || !is_linear_in_variable(&equation.right, var_name)
        {
            return Err(SolverError::UnsupportedEquationType);
        }

        // Isolate the variable
        let result_expr = isolate_variable(equation, var_name, &mut path)?;

        // Add isolation step
        path = path.step(
            Operation::Isolate(variable.clone()),
            format!("Isolate {} on one side", variable),
            result_expr.clone(),
        );

        // Build final resolution path
        let resolution_path = path.finish(result_expr.clone());

        Ok((Solution::Unique(result_expr), resolution_path))
    }

    fn can_solve(&self, equation: &Equation) -> bool {
        // Check if equation has obvious non-linear features (powers > 1 with variables)
        // We're more permissive here since we don't know the target variable yet,
        // but we can still reject clearly quadratic/polynomial equations.
        !has_obvious_nonlinearity(&equation.left) && !has_obvious_nonlinearity(&equation.right)
    }
}
