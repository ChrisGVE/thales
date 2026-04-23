//! [`LinearSystem`]: canonical `Arc<Expr>` representation of `A x = b`.
//!
//! Internal representation is fully exact: coefficient rows and the
//! constant vector are stored as `Vec<Vec<Arc<Expr>>>` / `Vec<Arc<Expr>>`.
//! The numeric `MatrixExpr` entry point ([`from_matrix`](LinearSystem::from_matrix))
//! compiles each cell through [`crate::numeric::compile::compile`] and
//! rejects cells that depend on any system variable (the unknowns being
//! solved for) — such coefficients are algebraically incoherent.
//! Symbolic parameters independent of the unknowns are retained and
//! handled symbolically by the solvers.

use std::sync::Arc;

use crate::ast::{Equation, Expression, Variable};
use crate::matrix::MatrixExpr;
use crate::numeric::compile::compile;
use crate::numeric::{normalize, BigRational, Expr, SymbolId};

use super::coeff::extract_linear_coefficients;
use super::cramer::solve_cramer;
use super::gauss::solve_gaussian;
use super::helpers::detection::contains_symbol;
use super::lu_exact::solve_via_lu_exact;
use super::system::SystemSolution;
use super::types::{SolverError, SolverResult};

// ── Conversions ───────────────────────────────────────────────────────────────

fn bigrational_to_arc(r: BigRational) -> Arc<Expr> {
    if r.is_integer() {
        if let Some(n) = r.numer().to_i64() {
            return Expr::int(n);
        }
    }
    Arc::new(Expr::Rational(r))
}

// ── LinearSystem ──────────────────────────────────────────────────────────────

/// A linear system `A x = b` stored as canonical `Arc<Expr>` matrices.
#[derive(Debug, Clone)]
pub struct LinearSystem {
    /// Coefficient matrix `A` (m × n).
    pub(crate) coeffs: Vec<Vec<Arc<Expr>>>,
    /// Constant vector `b` (length m).
    pub(crate) constants: Vec<Arc<Expr>>,
    /// Ordered unknowns (length n).
    pub(crate) variables: Vec<Variable>,
}

impl LinearSystem {
    /// Create a `LinearSystem` from user-supplied matrices.
    ///
    /// Each cell of `matrix_a` and `vector_b` is compiled to canonical
    /// `Arc<Expr>` form. Symbolic cells are accepted **only if they do not
    /// depend on any system variable** — a coefficient cell cannot be a
    /// function of one of the unknowns.
    ///
    /// # Errors
    ///
    /// - `vector_b` is not a single-column matrix.
    /// - Row count mismatch between `matrix_a` and `vector_b`.
    /// - Column count of `matrix_a` does not equal `variables.len()`.
    /// - Any cell in `matrix_a` or `vector_b` contains one of the system
    ///   variables.
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

        let var_ids: Vec<SymbolId> = variables
            .iter()
            .map(|v| SymbolId::intern(&v.name))
            .collect();

        let rows = matrix_a.rows();
        let cols = matrix_a.cols();
        let mut coeffs: Vec<Vec<Arc<Expr>>> = Vec::with_capacity(rows);
        for i in 0..rows {
            let mut row: Vec<Arc<Expr>> = Vec::with_capacity(cols);
            for j in 0..cols {
                let cell = matrix_a
                    .get(i, j)
                    .map_err(|e| SolverError::Other(e.to_string()))?;
                for (k, &sid) in var_ids.iter().enumerate() {
                    if contains_symbol(cell, sid) {
                        return Err(SolverError::Other(format!(
                            "matrix_a cell ({}, {}) depends on system variable '{}'; \
                             coefficients must be independent of the unknowns",
                            i, j, variables[k].name
                        )));
                    }
                }
                row.push(cell.clone());
            }
            coeffs.push(row);
        }

        let mut constants: Vec<Arc<Expr>> = Vec::with_capacity(rows);
        for i in 0..rows {
            let cell = vector_b
                .get(i, 0)
                .map_err(|e| SolverError::Other(e.to_string()))?;
            for (k, &sid) in var_ids.iter().enumerate() {
                if contains_symbol(cell, sid) {
                    return Err(SolverError::Other(format!(
                        "vector_b cell ({}) depends on system variable '{}'",
                        i, variables[k].name
                    )));
                }
            }
            constants.push(cell.clone());
        }

        Ok(Self {
            coeffs,
            constants,
            variables,
        })
    }

    /// Build a system from linear equations over the given unknowns.
    ///
    /// Each equation `lhs = rhs` is reduced to `(lhs - rhs)`, compiled, and
    /// passed to [`extract_linear_coefficients`]. The returned exact
    /// rationals become the row coefficients; the residual constant is
    /// negated to produce the right-hand side (so that `A x = b`).
    ///
    /// # Errors
    ///
    /// Propagates any error from coefficient extraction (e.g. non-linear
    /// terms).
    pub fn from_equations(equations: &[Equation], variables: &[Variable]) -> SolverResult<Self> {
        let n_eqs = equations.len();
        let n_vars = variables.len();

        if n_eqs == 0 || n_vars == 0 {
            return Err(SolverError::Other("Empty system".to_string()));
        }

        let mut coeffs: Vec<Vec<Arc<Expr>>> = Vec::with_capacity(n_eqs);
        let mut constants: Vec<Arc<Expr>> = Vec::with_capacity(n_eqs);

        for eq in equations {
            let lhs_arc = compile(&eq.left);
            let rhs_arc = compile(&eq.right);
            let combined_expr = normalize::sub(lhs_arc, rhs_arc);
            let (row_rat, constant_rat) = extract_linear_coefficients(&combined_expr, variables)?;
            let row: Vec<Arc<Expr>> = row_rat.into_iter().map(bigrational_to_arc).collect();
            coeffs.push(row);
            constants.push(bigrational_to_arc(-constant_rat));
        }

        Ok(Self {
            coeffs,
            constants,
            variables: variables.to_vec(),
        })
    }

    /// Number of equations (rows of `A`, length of `b`).
    pub fn num_equations(&self) -> usize {
        self.coeffs.len()
    }

    /// Number of unknowns (columns of `A`).
    pub fn num_variables(&self) -> usize {
        self.variables.len()
    }

    /// Solve the linear system using exact LU decomposition with row
    /// pivoting. Requires a square, non-singular system.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The system is not square.
    /// - The matrix is singular (an exact zero pivot is encountered).
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::ast::{Expression, Variable};
    /// use thales::solver::{LinearSystem, SystemSolution};
    ///
    /// let a = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expression::Integer(2), Expression::Integer(1)],
    ///     vec![Expression::Integer(1), Expression::Integer(3)],
    /// ]).unwrap();
    /// let b = MatrixExpr::from_expr_elements(vec![
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
        solve_via_lu_exact(&self.coeffs, &self.constants, &self.variables)
    }

    /// Solve the linear system using Gaussian elimination with
    /// first-nonzero pivoting.
    pub fn solve(&self) -> SolverResult<SystemSolution> {
        solve_gaussian(&self.coeffs, &self.constants, &self.variables)
    }

    /// Solve using Cramer's rule (2×2 and 3×3 systems only). Falls back
    /// to [`solve`](Self::solve) if the coefficient determinant is zero.
    pub fn solve_cramers(&self) -> SolverResult<SystemSolution> {
        solve_cramer(&self.coeffs, &self.constants, &self.variables)
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

    fn var_expr(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    // Arc<Expr> helpers used inside MatrixExpr::from_expr_elements calls.
    fn aint(n: i64) -> Arc<Expr> {
        Expr::int(n)
    }

    fn avar(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }

    fn add(l: Expression, r: Expression) -> Expression {
        Expression::Binary(BinaryOp::Add, Box::new(l), Box::new(r))
    }

    fn sub(l: Expression, r: Expression) -> Expression {
        Expression::Binary(BinaryOp::Sub, Box::new(l), Box::new(r))
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
            MatrixExpr::from_expr_elements(vec![vec![aint(2), aint(1)], vec![aint(1), aint(3)]])
                .unwrap();
        let b = MatrixExpr::from_expr_elements(vec![vec![aint(5)], vec![aint(10)]]).unwrap();
        let vars = make_vars(&["x", "y"]);
        let sys = LinearSystem::from_matrix(a, b, vars).unwrap();
        assert_eq!(sys.num_equations(), 2);
        assert_eq!(sys.num_variables(), 2);
    }

    #[test]
    fn test_from_matrix_rejects_non_column_b() {
        let a =
            MatrixExpr::from_expr_elements(vec![vec![aint(1), aint(0)], vec![aint(0), aint(1)]])
                .unwrap();
        let b =
            MatrixExpr::from_expr_elements(vec![vec![aint(1), aint(2)], vec![aint(3), aint(4)]])
                .unwrap();
        let vars = make_vars(&["x", "y"]);
        assert!(LinearSystem::from_matrix(a, b, vars).is_err());
    }

    #[test]
    fn test_from_matrix_rejects_dimension_mismatch() {
        let a =
            MatrixExpr::from_expr_elements(vec![vec![aint(1), aint(0)], vec![aint(0), aint(1)]])
                .unwrap();
        let b = MatrixExpr::from_expr_elements(vec![vec![aint(1)]]).unwrap();
        let vars = make_vars(&["x", "y"]);
        assert!(LinearSystem::from_matrix(a, b, vars).is_err());
    }

    #[test]
    fn test_from_matrix_rejects_system_variable_in_cell() {
        // Coefficient cell contains `x`, which is a system variable —
        // algebraically incoherent.
        let a =
            MatrixExpr::from_expr_elements(vec![vec![avar("x"), aint(1)], vec![aint(0), aint(1)]])
                .unwrap();
        let b = MatrixExpr::from_expr_elements(vec![vec![aint(1)], vec![aint(1)]]).unwrap();
        let vars = make_vars(&["x", "y"]);
        let err = LinearSystem::from_matrix(a, b, vars).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("system variable"), "unexpected error: {}", msg);
    }

    #[test]
    fn test_from_matrix_accepts_symbolic_parameter() {
        // `a` is a symbolic parameter, independent of the unknowns.
        let a_mat =
            MatrixExpr::from_expr_elements(vec![vec![avar("a"), aint(1)], vec![aint(0), aint(1)]])
                .unwrap();
        let b_mat = MatrixExpr::from_expr_elements(vec![vec![aint(1)], vec![aint(1)]]).unwrap();
        let vars = make_vars(&["x", "y"]);
        assert!(LinearSystem::from_matrix(a_mat, b_mat, vars).is_ok());
    }

    // ── solve_via_lu: 2x2 ─────────────────────────────────────────────────────

    #[test]
    fn test_solve_via_lu_2x2() {
        let a =
            MatrixExpr::from_expr_elements(vec![vec![aint(2), aint(1)], vec![aint(1), aint(3)]])
                .unwrap();
        let b = MatrixExpr::from_expr_elements(vec![vec![aint(5)], vec![aint(10)]]).unwrap();
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
        let a = MatrixExpr::from_expr_elements(vec![
            vec![aint(1), aint(1), aint(1)],
            vec![aint(0), aint(2), aint(5)],
            vec![aint(2), aint(5), aint(-1)],
        ])
        .unwrap();
        let b = MatrixExpr::from_expr_elements(vec![vec![aint(6)], vec![aint(-4)], vec![aint(27)]])
            .unwrap();
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
        let a =
            MatrixExpr::from_expr_elements(vec![vec![aint(1), aint(2)], vec![aint(2), aint(4)]])
                .unwrap();
        let b = MatrixExpr::from_expr_elements(vec![vec![aint(3)], vec![aint(6)]]).unwrap();
        let vars = make_vars(&["x", "y"]);
        let sys = LinearSystem::from_matrix(a, b, vars).unwrap();
        let result = sys.solve_via_lu();
        assert!(result.is_err(), "expected error for singular matrix");
    }

    // ── LU and Gaussian agree on non-singular systems ─────────────────────────

    #[test]
    fn test_solve_via_lu_matches_gaussian() {
        let x = Variable::new("x");
        let y = Variable::new("y");
        let eq1 = Equation::new("eq1", add(var_expr("x"), var_expr("y")), int(5));
        let eq2 = Equation::new("eq2", sub(var_expr("x"), var_expr("y")), int(1));
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
