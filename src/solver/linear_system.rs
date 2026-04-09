//! [`LinearSystem`]: matrix representation of a linear system Ax = b.

use std::collections::HashMap;

use crate::ast::{BinaryOp, Equation, Expression, Variable};
use crate::matrix::{MatrixError, MatrixExpr};

use super::coeff::extract_linear_coefficients;
use super::gauss::{det_2x2, det_3x3, f64_to_expr, solve_gaussian};
use super::system::SystemSolution;
use super::types::{SolverError, SolverResult};

// ── Error conversion ──────────────────────────────────────────────────────────

fn matrix_err_to_solver(e: MatrixError) -> SolverError {
    match e {
        MatrixError::InvalidOperation(msg) if msg.contains("singular") => {
            SolverError::Other(format!("Singular matrix: {}", msg))
        }
        other => SolverError::Other(other.to_string()),
    }
}

// ── LinearSystem ──────────────────────────────────────────────────────────────

/// A linear system of equations in matrix form Ax = b.
#[derive(Debug, Clone)]
pub struct LinearSystem {
    /// Coefficient matrix A stored as `MatrixExpr`.
    pub(crate) matrix_a: MatrixExpr,
    /// Constant vector b stored as a column `MatrixExpr`.
    pub(crate) vector_b: MatrixExpr,
    /// Variable names corresponding to columns.
    pub(crate) variables: Vec<Variable>,
}

impl LinearSystem {
    /// Create a `LinearSystem` directly from `MatrixExpr` data.
    ///
    /// `matrix_a` must be an m×n matrix and `vector_b` a column vector with m rows.
    ///
    /// # Errors
    ///
    /// Returns an error if `vector_b` is not a single-column matrix, or if the
    /// number of rows in `matrix_a` does not match `vector_b`.
    pub fn from_matrix(
        matrix_a: MatrixExpr,
        vector_b: MatrixExpr,
        variables: Vec<Variable>,
    ) -> SolverResult<Self> {
        if vector_b.cols() != 1 {
            return Err(SolverError::Other(
                "vector_b must be a column vector (single column)".to_string(),
            ));
        }
        if matrix_a.rows() != vector_b.rows() {
            return Err(SolverError::Other(format!(
                "matrix_a has {} rows but vector_b has {} rows",
                matrix_a.rows(),
                vector_b.rows()
            )));
        }
        if matrix_a.cols() != variables.len() {
            return Err(SolverError::Other(format!(
                "matrix_a has {} columns but {} variables provided",
                matrix_a.cols(),
                variables.len()
            )));
        }
        Ok(Self {
            matrix_a,
            vector_b,
            variables,
        })
    }

    /// Create a linear system from equations and variables.
    ///
    /// Extracts coefficients from linear equations of the form:
    /// a₁x₁ + a₂x₂ + ... + aₙxₙ = b
    ///
    /// # Errors
    ///
    /// Returns an error if any equation is not linear in the given variables.
    pub fn from_equations(equations: &[Equation], variables: &[Variable]) -> SolverResult<Self> {
        let n_eqs = equations.len();
        let n_vars = variables.len();

        if n_eqs == 0 || n_vars == 0 {
            return Err(SolverError::Other("Empty system".to_string()));
        }

        let mut coeff_rows: Vec<Vec<Expression>> = Vec::with_capacity(n_eqs);
        let mut const_rows: Vec<Vec<Expression>> = Vec::with_capacity(n_eqs);

        for eq in equations {
            let combined = Expression::Binary(
                BinaryOp::Sub,
                Box::new(eq.left.clone()),
                Box::new(eq.right.clone()),
            )
            .simplify();

            let (row, constant) = extract_linear_coefficients(&combined, variables)?;
            coeff_rows.push(row.into_iter().map(Expression::Float).collect());
            const_rows.push(vec![Expression::Float(-constant)]);
        }

        let matrix_a =
            MatrixExpr::from_elements(coeff_rows).map_err(|e| SolverError::Other(e.to_string()))?;
        let vector_b =
            MatrixExpr::from_elements(const_rows).map_err(|e| SolverError::Other(e.to_string()))?;

        Ok(Self {
            matrix_a,
            vector_b,
            variables: variables.to_vec(),
        })
    }

    // ── numeric helpers for legacy methods ────────────────────────────────────

    /// Extract the raw coefficient grid from `matrix_a`.
    fn coefficients(&self) -> Vec<Vec<f64>> {
        let empty: HashMap<String, f64> = HashMap::new();
        self.matrix_a.evaluate(&empty).unwrap_or_default()
    }

    /// Extract the raw constant vector from `vector_b`.
    fn constants(&self) -> Vec<f64> {
        let empty: HashMap<String, f64> = HashMap::new();
        self.vector_b
            .evaluate(&empty)
            .unwrap_or_default()
            .into_iter()
            .map(|row| row.into_iter().next().unwrap_or(0.0))
            .collect()
    }

    // ── solvers ───────────────────────────────────────────────────────────────

    /// Solve the linear system using LU decomposition via [`MatrixExpr::solve_system`].
    ///
    /// This method requires a square, non-singular coefficient matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The system is not square.
    /// - The matrix is singular or near-singular.
    /// - Any element contains unresolved symbolic variables.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::ast::{Expression, Variable};
    /// use thales::solver::{LinearSystem, SystemSolution};
    ///
    /// let a = MatrixExpr::from_elements(vec![
    ///     vec![Expression::Integer(2), Expression::Integer(1)],
    ///     vec![Expression::Integer(1), Expression::Integer(3)],
    /// ]).unwrap();
    /// let b = MatrixExpr::from_elements(vec![
    ///     vec![Expression::Integer(5)],
    ///     vec![Expression::Integer(10)],
    /// ]).unwrap();
    /// let vars = vec![Variable::new("x"), Variable::new("y")];
    /// let system = LinearSystem::from_matrix(a, b, vars).unwrap();
    /// let solution = system.solve_via_lu().unwrap();
    /// match solution {
    ///     SystemSolution::Unique(sol) => assert_eq!(sol.len(), 2),
    ///     _ => panic!("expected unique solution"),
    /// }
    /// ```
    pub fn solve_via_lu(&self) -> SolverResult<SystemSolution> {
        let n_vars = self.variables.len();
        let n_eqs = self.matrix_a.rows();

        if n_eqs != n_vars {
            return Err(SolverError::Other(format!(
                "solve_via_lu requires a square system ({} eqs, {} vars)",
                n_eqs, n_vars
            )));
        }

        let x = self
            .matrix_a
            .solve_system(&self.vector_b)
            .map_err(matrix_err_to_solver)?;

        let empty: HashMap<String, f64> = HashMap::new();
        let mut result = HashMap::new();
        for (i, var) in self.variables.iter().enumerate() {
            let val = x
                .get(i, 0)
                .map_err(|_| {
                    SolverError::Other(format!("Failed to get solution index for {}", var.name))
                })
                .and_then(|e| {
                    e.evaluate(&empty).ok_or_else(|| {
                        SolverError::Other(format!("Failed to evaluate solution for {}", var.name))
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

    /// Solve the linear system using Gaussian elimination with partial pivoting.
    pub fn solve(&self) -> SolverResult<SystemSolution> {
        let coefficients = self.coefficients();
        let constants = self.constants();
        solve_gaussian(&coefficients, &constants, &self.variables)
    }

    /// Solve using Cramer's rule (2×2 and 3×3 systems only).
    pub fn solve_cramers(&self) -> SolverResult<SystemSolution> {
        let coefficients = self.coefficients();
        let constants = self.constants();
        let n = self.variables.len();

        if coefficients.len() != n {
            return Err(SolverError::Other(
                "Cramer's rule requires square system".to_string(),
            ));
        }
        if n != 2 && n != 3 {
            return Err(SolverError::Other(
                "Cramer's rule only implemented for 2x2 and 3x3 systems".to_string(),
            ));
        }

        let det_a = if n == 2 {
            det_2x2(&coefficients)
        } else {
            det_3x3(&coefficients)
        };

        if det_a.abs() < 1e-15 {
            return solve_gaussian(&coefficients, &constants, &self.variables);
        }

        let mut result = HashMap::new();
        for i in 0..n {
            let mut modified = coefficients.clone();
            for (row, &c) in constants.iter().enumerate() {
                modified[row][i] = c;
            }
            let det_i = if n == 2 {
                det_2x2(&modified)
            } else {
                det_3x3(&modified)
            };
            let val = det_i / det_a;
            let expr = if (val - val.round()).abs() < 1e-10 {
                Expression::Integer(val.round() as i64)
            } else {
                Expression::Float(val)
            };
            result.insert(self.variables[i].clone(), expr);
        }

        Ok(SystemSolution::Unique(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Equation, Expression, Variable};
    use crate::matrix::MatrixExpr;
    use std::collections::HashMap;

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn add(l: Expression, r: Expression) -> Expression {
        Expression::Binary(BinaryOp::Add, Box::new(l), Box::new(r))
    }

    fn sub(l: Expression, r: Expression) -> Expression {
        Expression::Binary(BinaryOp::Sub, Box::new(l), Box::new(r))
    }

    fn mul(l: Expression, r: Expression) -> Expression {
        Expression::Binary(BinaryOp::Mul, Box::new(l), Box::new(r))
    }

    fn make_vars(names: &[&str]) -> Vec<Variable> {
        names.iter().map(|n| Variable::new(*n)).collect()
    }

    fn eval_empty(expr: &Expression) -> f64 {
        let empty: HashMap<String, f64> = HashMap::new();
        expr.evaluate(&empty).expect("evaluate should succeed")
    }

    // ── from_matrix constructor ───────────────────────────────────────────────

    #[test]
    fn test_from_matrix_valid() {
        let a =
            MatrixExpr::from_elements(vec![vec![int(2), int(1)], vec![int(1), int(3)]]).unwrap();
        let b = MatrixExpr::from_elements(vec![vec![int(5)], vec![int(10)]]).unwrap();
        let vars = make_vars(&["x", "y"]);
        let sys = LinearSystem::from_matrix(a, b, vars).unwrap();
        assert_eq!(sys.matrix_a.rows(), 2);
        assert_eq!(sys.matrix_a.cols(), 2);
        assert_eq!(sys.vector_b.rows(), 2);
        assert_eq!(sys.vector_b.cols(), 1);
    }

    #[test]
    fn test_from_matrix_rejects_non_column_b() {
        let a =
            MatrixExpr::from_elements(vec![vec![int(1), int(0)], vec![int(0), int(1)]]).unwrap();
        let b =
            MatrixExpr::from_elements(vec![vec![int(1), int(2)], vec![int(3), int(4)]]).unwrap();
        let vars = make_vars(&["x", "y"]);
        assert!(LinearSystem::from_matrix(a, b, vars).is_err());
    }

    #[test]
    fn test_from_matrix_rejects_dimension_mismatch() {
        let a =
            MatrixExpr::from_elements(vec![vec![int(1), int(0)], vec![int(0), int(1)]]).unwrap();
        let b = MatrixExpr::from_elements(vec![vec![int(1)]]).unwrap();
        let vars = make_vars(&["x", "y"]);
        assert!(LinearSystem::from_matrix(a, b, vars).is_err());
    }

    // ── solve_via_lu: 2x2 ─────────────────────────────────────────────────────

    #[test]
    fn test_solve_via_lu_2x2() {
        // [[2,1],[1,3]] x = [5, 10]  =>  x=1, y=3
        let a =
            MatrixExpr::from_elements(vec![vec![int(2), int(1)], vec![int(1), int(3)]]).unwrap();
        let b = MatrixExpr::from_elements(vec![vec![int(5)], vec![int(10)]]).unwrap();
        let vars = make_vars(&["x", "y"]);
        let sys = LinearSystem::from_matrix(a, b, vars.clone()).unwrap();
        let sol = sys.solve_via_lu().unwrap();

        match sol {
            SystemSolution::Unique(map) => {
                let x_val = eval_empty(map.get(&vars[0]).unwrap());
                let y_val = eval_empty(map.get(&vars[1]).unwrap());
                assert!((x_val - 1.0).abs() < 1e-9, "x={}", x_val);
                assert!((y_val - 3.0).abs() < 1e-9, "y={}", y_val);
            }
            _ => panic!("expected unique solution"),
        }
    }

    // ── solve_via_lu: 3x3 ─────────────────────────────────────────────────────

    #[test]
    fn test_solve_via_lu_3x3() {
        // x + y + z = 6
        // 2y + 5z = -4
        // 2x + 5y - z = 27
        // Solution: x=5, y=3, z=-2
        let a = MatrixExpr::from_elements(vec![
            vec![int(1), int(1), int(1)],
            vec![int(0), int(2), int(5)],
            vec![int(2), int(5), int(-1)],
        ])
        .unwrap();
        let b =
            MatrixExpr::from_elements(vec![vec![int(6)], vec![int(-4)], vec![int(27)]]).unwrap();
        let vars = make_vars(&["x", "y", "z"]);
        let sys = LinearSystem::from_matrix(a, b, vars.clone()).unwrap();
        let sol = sys.solve_via_lu().unwrap();

        match sol {
            SystemSolution::Unique(map) => {
                let x = eval_empty(map.get(&vars[0]).unwrap());
                let y = eval_empty(map.get(&vars[1]).unwrap());
                let z = eval_empty(map.get(&vars[2]).unwrap());
                assert!((x - 5.0).abs() < 1e-9, "x={}", x);
                assert!((y - 3.0).abs() < 1e-9, "y={}", y);
                assert!((z - (-2.0)).abs() < 1e-9, "z={}", z);
            }
            _ => panic!("expected unique solution"),
        }
    }

    // ── singular system detection ─────────────────────────────────────────────

    #[test]
    fn test_solve_via_lu_singular_system() {
        // [[1, 2], [2, 4]] is singular (rows are proportional)
        let a =
            MatrixExpr::from_elements(vec![vec![int(1), int(2)], vec![int(2), int(4)]]).unwrap();
        let b = MatrixExpr::from_elements(vec![vec![int(3)], vec![int(6)]]).unwrap();
        let vars = make_vars(&["x", "y"]);
        let sys = LinearSystem::from_matrix(a, b, vars).unwrap();
        let result = sys.solve_via_lu();
        assert!(result.is_err(), "expected error for singular matrix");
    }

    // ── solve_via_lu result matches solve() ───────────────────────────────────

    #[test]
    fn test_solve_via_lu_matches_gaussian() {
        let x = Variable::new("x");
        let y = Variable::new("y");
        let eq1 = Equation::new("eq1", add(var("x"), var("y")), int(5));
        let eq2 = Equation::new("eq2", sub(var("x"), var("y")), int(1));
        let sys = LinearSystem::from_equations(&[eq1, eq2], &[x.clone(), y.clone()]).unwrap();

        let lu_sol = sys.solve_via_lu().unwrap();
        let gauss_sol = sys.solve().unwrap();

        let empty: HashMap<String, f64> = HashMap::new();
        match (lu_sol, gauss_sol) {
            (SystemSolution::Unique(lu_map), SystemSolution::Unique(g_map)) => {
                for (var, lu_expr) in &lu_map {
                    let g_expr = g_map.get(var).unwrap();
                    let lu_val = lu_expr.evaluate(&empty).unwrap();
                    let g_val = g_expr.evaluate(&empty).unwrap();
                    assert!((lu_val - g_val).abs() < 1e-9);
                }
            }
            _ => panic!("both should be unique"),
        }
    }
}
