//! System of equations solver — linear and polynomial.
//!
//! Linear systems are handled by the internal [`LinearSystem`] machinery
//! (exact Gaussian / Cramer / LU on `Arc<Expr>`). Polynomial (non-linear)
//! systems are delegated to [`crate::numeric::system_solver`], which
//! computes a Groebner basis under lexicographic elimination order and
//! back-substitutes to recover every rational solution point.
//!
//! [`SystemSolver::solve`] auto-dispatches: it inspects each equation's
//! `(lhs − rhs)` in canonical `Arc<Expr>` form and routes linear systems
//! through the linear path and polynomial (or mixed linear/polynomial)
//! systems through the Groebner path.

use std::collections::HashMap;
use std::sync::Arc;

use crate::ast::{Equation, Expression, Variable};
use crate::numeric::compile::{compile, decompile};
use crate::numeric::system_solver::solve_system_expr;
use crate::numeric::trace::{Step, TechniqueTag, Trace};
use crate::numeric::{Expr, SymbolId};

use super::helpers::is_linear_system_expr;
use super::linear_system::LinearSystem;
use super::types::{Solution, SolverError, SolverResult};

/// Result type for system solutions.
#[derive(Debug, Clone)]
pub enum SystemSolution {
    /// Unique solution: each variable has exactly one value.
    Unique(HashMap<Variable, Expression>),
    /// Multiple discrete solutions (typical of non-linear polynomial systems
    /// solved via Groebner basis elimination). Each element is a full
    /// solution point assigning a value to every system variable.
    Multiple(Vec<HashMap<Variable, Expression>>),
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
            SystemSolution::Multiple(points) => {
                // Aggregate each variable's values across all solution
                // points into a `Solution::Multiple`. Pairing between
                // variables is lost here; callers that need joint points
                // should consume `SystemSolution::Multiple` directly.
                let mut by_var: HashMap<Variable, Vec<Expression>> = HashMap::new();
                for point in points {
                    for (var, expr) in point {
                        by_var.entry(var).or_default().push(expr);
                    }
                }
                let mut out = HashMap::new();
                for (var, values) in by_var {
                    let sol = if values.len() == 1 {
                        Solution::Unique(values.into_iter().next().unwrap())
                    } else {
                        Solution::Multiple(values)
                    };
                    out.insert(var, sol);
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

    /// Solve using matrix inversion: x = A⁻¹ b.
    ///
    /// This method computes the explicit inverse of the coefficient matrix and
    /// multiplies it by the constant vector.  It is primarily useful for small
    /// systems or educational purposes; for larger systems prefer
    /// [`solve_linear_system`] (Gaussian elimination) or
    /// [`solve_via_lu`](super::linear_system::LinearSystem::solve_via_lu).
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CannotSolve`] when the matrix is singular, and
    /// [`SolverError::Other`] for any other matrix or dimension error.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::solver::{SystemSolver, SystemSolution};
    /// use thales::ast::{Equation, Expression, Variable, BinaryOp};
    ///
    /// let x = Variable::new("x");
    /// let y = Variable::new("y");
    ///
    /// // x + y = 5,  x − y = 1  =>  x = 3, y = 2
    /// let eq1 = Equation::new(
    ///     "eq1",
    ///     Expression::Binary(
    ///         BinaryOp::Add,
    ///         Box::new(Expression::Variable(x.clone())),
    ///         Box::new(Expression::Variable(y.clone())),
    ///     ),
    ///     Expression::Integer(5),
    /// );
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
    /// let solver = SystemSolver::new();
    /// let sol = solver
    ///     .solve_matrix_inverse(&[eq1, eq2], &[x.clone(), y.clone()])
    ///     .unwrap();
    /// match sol {
    ///     SystemSolution::Unique(map) => {
    ///         assert!(map.contains_key(&x));
    ///         assert!(map.contains_key(&y));
    ///     }
    ///     _ => panic!("expected unique solution"),
    /// }
    /// ```
    pub fn solve_matrix_inverse(
        &self,
        equations: &[Equation],
        variables: &[Variable],
    ) -> SolverResult<SystemSolution> {
        self.solve_matrix_inverse_with_path(equations, variables)
            .map(|(sol, _path)| sol)
    }

    /// Solve using matrix inversion and return both the solution and the
    /// [`Trace`] of techniques applied.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CannotSolve`] when the matrix is singular, and
    /// [`SolverError::Other`] for any other matrix or dimension error.
    pub fn solve_matrix_inverse_with_path(
        &self,
        equations: &[Equation],
        variables: &[Variable],
    ) -> SolverResult<(SystemSolution, Trace)> {
        let system = LinearSystem::from_equations(equations, variables)?;
        let mut trace = Trace::new();

        trace.push(Step::new(
            TechniqueTag::MatrixInverse,
            "Compute x = A⁻¹ b via exact LU decomposition".to_string(),
        ));

        let sol = system.solve_via_lu()?;

        for var in &system.variables {
            trace.push(Step::new(
                TechniqueTag::Custom("BackSubstitute"),
                format!(
                    "variable={}; Compute value of {} from x = A⁻¹b",
                    var.name, var.name
                ),
            ));
        }

        Ok((sol, trace))
    }

    /// Solve using the best available method.
    ///
    /// Tries LU decomposition first (fast, numerically stable).  If the system
    /// is not square or LU fails, falls back to Gaussian elimination.  The
    /// matrix-inverse path is not attempted here because it is more expensive
    /// and carries no advantage over LU for well-conditioned systems.
    ///
    /// Returns the first successful `SystemSolution` or the last error
    /// encountered.
    pub fn solve_best_effort(
        &self,
        equations: &[Equation],
        variables: &[Variable],
    ) -> SolverResult<SystemSolution> {
        let system = LinearSystem::from_equations(equations, variables)?;
        if system.num_equations() == system.num_variables() {
            if let Ok(sol) = system.solve_via_lu() {
                return Ok(sol);
            }
        }
        system.solve()
    }

    /// Solve a non-linear polynomial system via Groebner basis elimination.
    ///
    /// Each equation `lhs = rhs` is compiled to canonical `Arc<Expr>` form
    /// and treated as `lhs − rhs = 0`. The unknowns are listed in
    /// `variables`; the last variable is eliminated first under lex order.
    ///
    /// Only rational real solution points are returned — complex roots
    /// and irrational roots outside the quadratic-surd envelope are
    /// dropped by the underlying numeric solver.
    ///
    /// Returns:
    /// - [`SystemSolution::Unique`] when exactly one rational point is
    ///   recovered,
    /// - [`SystemSolution::Multiple`] for several rational points,
    /// - [`SystemSolution::NoSolution`] when the Groebner basis has no
    ///   rational solutions.
    ///
    /// Underdetermined systems (fewer independent polynomial relations
    /// than unknowns) currently return `NoSolution` because the numeric
    /// back-substitution cannot enumerate a parametric family without a
    /// univariate base element — this is a limitation of the numeric
    /// layer, not a semantic claim.
    pub fn solve_polynomial_system(
        &self,
        equations: &[Equation],
        variables: &[Variable],
    ) -> SolverResult<SystemSolution> {
        if equations.is_empty() || variables.is_empty() {
            return Err(SolverError::Other("Empty system".to_string()));
        }

        let eq_pairs: Vec<(Arc<Expr>, Arc<Expr>)> = equations
            .iter()
            .map(|eq| (compile(&eq.left), compile(&eq.right)))
            .collect();

        let var_ids: Vec<SymbolId> = variables
            .iter()
            .map(|v| SymbolId::intern(&v.name))
            .collect();

        let points = solve_system_expr(&eq_pairs, &var_ids);

        if points.is_empty() {
            return Ok(SystemSolution::NoSolution);
        }

        let mut maps: Vec<HashMap<Variable, Expression>> = Vec::with_capacity(points.len());
        for point in points {
            let mut map: HashMap<Variable, Expression> = HashMap::new();
            for (sid, arc) in point {
                if let Some(var) = variables.iter().find(|v| SymbolId::intern(&v.name) == sid) {
                    map.insert(var.clone(), decompile(&arc));
                }
            }
            maps.push(map);
        }

        if maps.len() == 1 {
            Ok(SystemSolution::Unique(maps.into_iter().next().unwrap()))
        } else {
            Ok(SystemSolution::Multiple(maps))
        }
    }

    /// Auto-dispatching solver: chooses linear or polynomial path based
    /// on equation structure.
    ///
    /// Compiles every `(lhs − rhs)` to canonical `Arc<Expr>` and checks
    /// joint linearity with respect to `variables`. If every combined
    /// residual is linear, the linear path (Gauss) is used; otherwise
    /// the polynomial path (Groebner) is invoked.
    pub fn solve(
        &self,
        equations: &[Equation],
        variables: &[Variable],
    ) -> SolverResult<SystemSolution> {
        if equations.is_empty() || variables.is_empty() {
            return Err(SolverError::Other("Empty system".to_string()));
        }

        let var_ids: Vec<SymbolId> = variables
            .iter()
            .map(|v| SymbolId::intern(&v.name))
            .collect();

        let is_linear = equations.iter().all(|eq| {
            let combined = Expression::Binary(
                crate::ast::BinaryOp::Sub,
                Box::new(eq.left.clone()),
                Box::new(eq.right.clone()),
            );
            let compiled = compile(&combined);
            is_linear_system_expr(&compiled, &var_ids)
        });

        if is_linear {
            self.solve_linear_system(equations, variables)
        } else {
            self.solve_polynomial_system(equations, variables)
        }
    }
}

#[cfg(test)]
mod tests;
