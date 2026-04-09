//! Linear system of equations solver.

use std::collections::HashMap;

use crate::ast::{Equation, Expression, Variable};
use crate::resolution_path::{Operation, ResolutionPath, ResolutionStep};

use super::linear_system::LinearSystem;
use super::types::{Solution, SolverError, SolverResult};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Convert a numeric column `MatrixExpr` result into a `SystemSolution`.
///
/// `x` must be an n×1 matrix whose elements evaluate to concrete numbers.
fn matrix_to_solution(
    x: &crate::matrix::MatrixExpr,
    variables: &[Variable],
) -> SolverResult<SystemSolution> {
    let empty: HashMap<String, f64> = HashMap::new();
    let mut result = HashMap::new();
    for (i, var) in variables.iter().enumerate() {
        let val = x
            .get(i, 0)
            .map_err(|_| SolverError::Other(format!("index error for variable '{}'", var.name)))
            .and_then(|e| {
                e.evaluate(&empty).ok_or_else(|| {
                    SolverError::Other(format!("failed to evaluate solution for '{}'", var.name))
                })
            })?;
        let expr = if (val - val.round()).abs() < 1e-10 {
            Expression::Integer(val.round() as i64)
        } else {
            Expression::Float(val)
        };
        result.insert(var.clone(), expr);
    }
    Ok(SystemSolution::Unique(result))
}

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
    /// resolution path recording all steps.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CannotSolve`] when the matrix is singular, and
    /// [`SolverError::Other`] for any other matrix or dimension error.
    pub fn solve_matrix_inverse_with_path(
        &self,
        equations: &[Equation],
        variables: &[Variable],
    ) -> SolverResult<(SystemSolution, ResolutionPath)> {
        let system = LinearSystem::from_equations(equations, variables)?;
        let initial = Expression::Integer(equations.len() as i64);
        let mut path = ResolutionPath::new(initial);

        path.add_step(ResolutionStep::new(
            Operation::MatrixInverse,
            "Compute A⁻¹ from the coefficient matrix".to_string(),
            Expression::Integer(0),
        ));

        let inv_a = system
            .matrix_a
            .inverse()
            .map_err(|e| SolverError::CannotSolve(e.to_string()))?;

        let x = inv_a
            .mul(&system.vector_b)
            .map_err(|e| SolverError::CannotSolve(e.to_string()))?;

        for var in &system.variables {
            path.add_step(ResolutionStep::new(
                Operation::BackSubstitute {
                    variable: var.name.clone(),
                },
                format!("Compute value of {} from x = A⁻¹b", var.name),
                Expression::Integer(0),
            ));
        }

        let sol = matrix_to_solution(&x, &system.variables)?;
        let result_expr = Expression::Integer(system.variables.len() as i64);
        path.set_result(result_expr);

        Ok((sol, path))
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
        if system.matrix_a.rows() == system.variables.len() {
            if let Ok(sol) = system.solve_via_lu() {
                return Ok(sol);
            }
        }
        system.solve()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Equation, Expression, Variable};

    // ── helpers ───────────────────────────────────────────────────────────────

    fn var_expr(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn add(l: Expression, r: Expression) -> Expression {
        Expression::Binary(BinaryOp::Add, Box::new(l), Box::new(r))
    }

    fn sub(l: Expression, r: Expression) -> Expression {
        Expression::Binary(BinaryOp::Sub, Box::new(l), Box::new(r))
    }

    fn mul_expr(l: Expression, r: Expression) -> Expression {
        Expression::Binary(BinaryOp::Mul, Box::new(l), Box::new(r))
    }

    fn eval(expr: &Expression) -> f64 {
        let empty: HashMap<String, f64> = HashMap::new();
        expr.evaluate(&empty).expect("evaluate")
    }

    fn make_2x2_system() -> ([Equation; 2], [Variable; 2]) {
        // x + y = 5,  x − y = 1  =>  x = 3, y = 2
        let x = Variable::new("x");
        let y = Variable::new("y");
        let eq1 = Equation::new(
            "eq1",
            add(var_expr("x"), var_expr("y")),
            Expression::Integer(5),
        );
        let eq2 = Equation::new(
            "eq2",
            sub(var_expr("x"), var_expr("y")),
            Expression::Integer(1),
        );
        ([eq1, eq2], [x, y])
    }

    fn make_3x3_system() -> ([Equation; 3], [Variable; 3]) {
        // x + y + z = 6,  2x + 5y = 0,  2x + 3z = 10  =>  x=5, y=-2, z=0
        // Use a cleaner 3×3: 2x+y-z=8, -3x-y+2z=-11, -2x+y+2z=-3 => x=2,y=3,z=-1
        let x = Variable::new("x");
        let y = Variable::new("y");
        let z = Variable::new("z");
        // 2x + y - z = 8
        let lhs1 = sub(
            add(
                mul_expr(Expression::Integer(2), var_expr("x")),
                var_expr("y"),
            ),
            var_expr("z"),
        );
        // -3x - y + 2z = -11
        let lhs2 = add(
            sub(
                mul_expr(Expression::Integer(-3), var_expr("x")),
                var_expr("y"),
            ),
            mul_expr(Expression::Integer(2), var_expr("z")),
        );
        // -2x + y + 2z = -3
        let lhs3 = add(
            add(
                mul_expr(Expression::Integer(-2), var_expr("x")),
                var_expr("y"),
            ),
            mul_expr(Expression::Integer(2), var_expr("z")),
        );
        let eq1 = Equation::new("eq1", lhs1, Expression::Integer(8));
        let eq2 = Equation::new("eq2", lhs2, Expression::Integer(-11));
        let eq3 = Equation::new("eq3", lhs3, Expression::Integer(-3));
        ([eq1, eq2, eq3], [x, y, z])
    }

    // ── solve_matrix_inverse: 2×2 ─────────────────────────────────────────────

    #[test]
    fn test_inverse_2x2_correct_solution() {
        let ([eq1, eq2], [x, y]) = make_2x2_system();
        let solver = SystemSolver::new();
        let sol = solver
            .solve_matrix_inverse(&[eq1, eq2], &[x.clone(), y.clone()])
            .unwrap();
        match sol {
            SystemSolution::Unique(map) => {
                let xv = eval(map.get(&x).unwrap());
                let yv = eval(map.get(&y).unwrap());
                assert!((xv - 3.0).abs() < 1e-9, "x={xv}");
                assert!((yv - 2.0).abs() < 1e-9, "y={yv}");
            }
            _ => panic!("expected unique solution"),
        }
    }

    // ── solve_matrix_inverse: 3×3 ─────────────────────────────────────────────

    #[test]
    fn test_inverse_3x3_correct_solution() {
        let ([eq1, eq2, eq3], [x, y, z]) = make_3x3_system();
        let solver = SystemSolver::new();
        let sol = solver
            .solve_matrix_inverse(&[eq1, eq2, eq3], &[x.clone(), y.clone(), z.clone()])
            .unwrap();
        match sol {
            SystemSolution::Unique(map) => {
                let xv = eval(map.get(&x).unwrap());
                let yv = eval(map.get(&y).unwrap());
                let zv = eval(map.get(&z).unwrap());
                assert!((xv - 2.0).abs() < 1e-9, "x={xv}");
                assert!((yv - 3.0).abs() < 1e-9, "y={yv}");
                assert!((zv - (-1.0)).abs() < 1e-9, "z={zv}");
            }
            _ => panic!("expected unique solution"),
        }
    }

    // ── singular system returns error ─────────────────────────────────────────

    #[test]
    fn test_inverse_singular_system_returns_error() {
        // x + y = 3,  2x + 2y = 6  (rows proportional => singular)
        let x = Variable::new("x");
        let y = Variable::new("y");
        let eq1 = Equation::new(
            "eq1",
            add(var_expr("x"), var_expr("y")),
            Expression::Integer(3),
        );
        let eq2 = Equation::new(
            "eq2",
            add(
                mul_expr(Expression::Integer(2), var_expr("x")),
                mul_expr(Expression::Integer(2), var_expr("y")),
            ),
            Expression::Integer(6),
        );
        let solver = SystemSolver::new();
        let result = solver.solve_matrix_inverse(&[eq1, eq2], &[x, y]);
        assert!(result.is_err(), "expected error for singular matrix");
    }

    // ── inverse result matches LU / Gaussian ─────────────────────────────────

    #[test]
    fn test_inverse_matches_gaussian_2x2() {
        let ([eq1, eq2], [x, y]) = make_2x2_system();
        let solver = SystemSolver::new();

        let inv_sol = solver
            .solve_matrix_inverse(&[eq1.clone(), eq2.clone()], &[x.clone(), y.clone()])
            .unwrap();
        let gauss_sol = solver
            .solve_linear_system(&[eq1, eq2], &[x.clone(), y.clone()])
            .unwrap();

        match (inv_sol, gauss_sol) {
            (SystemSolution::Unique(inv_map), SystemSolution::Unique(g_map)) => {
                for var in &[x, y] {
                    let iv = eval(inv_map.get(var).unwrap());
                    let gv = eval(g_map.get(var).unwrap());
                    assert!(
                        (iv - gv).abs() < 1e-9,
                        "mismatch for {}: inv={iv}, gauss={gv}",
                        var.name
                    );
                }
            }
            _ => panic!("both should be unique"),
        }
    }

    // ── solve_matrix_inverse_with_path ───────────────────────────────────────

    #[test]
    fn test_inverse_with_path_contains_matrix_inverse_op() {
        use crate::resolution_path::Operation;

        let ([eq1, eq2], [x, y]) = make_2x2_system();
        let solver = SystemSolver::new();
        let (_sol, path) = solver
            .solve_matrix_inverse_with_path(&[eq1, eq2], &[x.clone(), y.clone()])
            .unwrap();

        let has_matrix_inverse = path
            .steps
            .iter()
            .any(|s| matches!(s.operation, Operation::MatrixInverse));
        assert!(has_matrix_inverse, "path must contain MatrixInverse step");
    }

    #[test]
    fn test_inverse_with_path_contains_back_substitute_steps() {
        use crate::resolution_path::Operation;

        let ([eq1, eq2], [x, y]) = make_2x2_system();
        let solver = SystemSolver::new();
        let (_sol, path) = solver
            .solve_matrix_inverse_with_path(&[eq1, eq2], &[x.clone(), y.clone()])
            .unwrap();

        let back_subs: Vec<_> = path
            .steps
            .iter()
            .filter(|s| matches!(s.operation, Operation::BackSubstitute { .. }))
            .collect();
        assert_eq!(
            back_subs.len(),
            2,
            "expected one BackSubstitute per variable"
        );
    }

    #[test]
    fn test_inverse_with_path_difficulty_is_advanced() {
        use crate::resolution_path::TechniqueDifficulty;

        let ([eq1, eq2], [x, y]) = make_2x2_system();
        let solver = SystemSolver::new();
        let (_sol, path) = solver
            .solve_matrix_inverse_with_path(&[eq1, eq2], &[x.clone(), y.clone()])
            .unwrap();

        assert_eq!(path.max_difficulty(), TechniqueDifficulty::Advanced);
    }

    // ── solve_best_effort: prefers LU, falls back correctly ───────────────────

    #[test]
    fn test_best_effort_returns_unique_for_2x2() {
        let ([eq1, eq2], [x, y]) = make_2x2_system();
        let solver = SystemSolver::new();
        let sol = solver
            .solve_best_effort(&[eq1, eq2], &[x.clone(), y.clone()])
            .unwrap();
        match sol {
            SystemSolution::Unique(map) => {
                let xv = eval(map.get(&x).unwrap());
                let yv = eval(map.get(&y).unwrap());
                assert!((xv - 3.0).abs() < 1e-9);
                assert!((yv - 2.0).abs() < 1e-9);
            }
            _ => panic!("expected unique solution"),
        }
    }
}
