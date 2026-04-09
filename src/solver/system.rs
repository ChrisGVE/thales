//! Linear system of equations solver.

use std::collections::HashMap;

use crate::ast::{Equation, Expression, Variable};

use super::linear_system::LinearSystem;
use super::types::{Solution, SolverError, SolverResult};

/// Result type for system solutions.
#[derive(Debug, Clone)]
pub enum SystemSolution {
    /// Unique solution: each variable has exactly one value.
    Unique(HashMap<Variable, Expression>),
    /// Infinite solutions: variables are expressed in terms of free parameters.
    Infinite {
        /// Variables that have specific values.
        bound: HashMap<Variable, Expression>,
        /// Variables that are free parameters (can take any value).
        free: Vec<Variable>,
    },
    /// No solution: the system is inconsistent.
    NoSolution,
}

/// System of equations solver.
#[derive(Debug, Default)]
pub struct SystemSolver;

impl SystemSolver {
    /// Creates a new system of equations solver.
    pub fn new() -> Self {
        Self
    }

    /// Solve a system of linear equations for multiple variables.
    ///
    /// Uses Gaussian elimination with partial pivoting for general systems.
    /// For 2x2 and 3x3 systems, Cramer's rule is also available.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::solver::{SystemSolver, SystemSolution};
    /// use thales::ast::{Equation, Expression, Variable, BinaryOp};
    ///
    /// let solver = SystemSolver::new();
    ///
    /// // Solve: x + y = 5, x - y = 1
    /// let x = Variable::new("x");
    /// let y = Variable::new("y");
    ///
    /// let eq1 = Equation::new(
    ///     "eq1",
    ///     Expression::Binary(
    ///         BinaryOp::Add,
    ///         Box::new(Expression::Variable(x.clone())),
    ///         Box::new(Expression::Variable(y.clone())),
    ///     ),
    ///     Expression::Integer(5),
    /// );
    ///
    /// let eq2 = Equation::new(
    ///     "eq2",
    ///     Expression::Binary(
    ///         BinaryOp::Sub,
    ///         Box::new(Expression::Variable(x.clone())),
    ///         Box::new(Expression::Variable(y.clone())),
    ///     ),
    ///     Expression::Integer(1),
    /// );
    ///
    /// let result = solver.solve_linear_system(&[eq1, eq2], &[x.clone(), y.clone()]).unwrap();
    /// match result {
    ///     SystemSolution::Unique(sol) => {
    ///         // x = 3, y = 2
    ///         assert!(sol.contains_key(&x));
    ///         assert!(sol.contains_key(&y));
    ///     }
    ///     _ => panic!("Expected unique solution"),
    /// }
    /// ```
    pub fn solve_linear_system(
        &self,
        equations: &[Equation],
        variables: &[Variable],
    ) -> SolverResult<SystemSolution> {
        let system = LinearSystem::from_equations(equations, variables)?;
        system.solve()
    }

    /// Solve using Cramer's rule (2x2 and 3x3 systems only).
    pub fn solve_cramers(
        &self,
        equations: &[Equation],
        variables: &[Variable],
    ) -> SolverResult<SystemSolution> {
        let system = LinearSystem::from_equations(equations, variables)?;
        system.solve_cramers()
    }

    /// Solve a system of equations for multiple variables.
    ///
    /// This is a legacy method that delegates to solve_linear_system.
    pub fn solve_system(
        &self,
        equations: &[Equation],
        variables: &[Variable],
    ) -> SolverResult<HashMap<Variable, Solution>> {
        let result = self.solve_linear_system(equations, variables)?;

        match result {
            SystemSolution::Unique(sol) => {
                let mut out = HashMap::new();
                for (var, expr) in sol {
                    out.insert(var, Solution::Unique(expr));
                }
                Ok(out)
            }
            SystemSolution::Infinite { bound, free: _ } => {
                let mut out = HashMap::new();
                for (var, expr) in bound {
                    out.insert(
                        var,
                        Solution::Parametric {
                            expression: expr,
                            constraints: vec![],
                        },
                    );
                }
                Ok(out)
            }
            SystemSolution::NoSolution => Err(SolverError::NoSolution),
        }
    }
}
