//! [`LinearSystem`]: matrix representation of a linear system Ax = b.

use std::collections::HashMap;

use crate::ast::{BinaryOp, Equation, Expression, UnaryOp, Variable};
use crate::matrix::{MatrixError, MatrixExpr};

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

            let (row, constant) = Self::extract_linear_coefficients(&combined, variables)?;
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

    // ── coefficient extraction ─────────────────────────────────────────────────

    fn extract_linear_coefficients(
        expr: &Expression,
        variables: &[Variable],
    ) -> SolverResult<(Vec<f64>, f64)> {
        let mut coeffs = vec![0.0; variables.len()];
        let mut constant = 0.0;

        let terms = Self::collect_additive_terms(expr);

        for term in terms {
            let mut found_var = false;
            for (i, var) in variables.iter().enumerate() {
                if term.contains_variable(&var.name) {
                    let coeff = Self::extract_coefficient(&term, var)?;
                    coeffs[i] += coeff;
                    found_var = true;
                    break;
                }
            }

            if !found_var {
                let empty: HashMap<String, f64> = HashMap::new();
                match term.evaluate(&empty) {
                    Some(val) => constant += val,
                    None => {
                        return Err(SolverError::Other(format!(
                            "Cannot evaluate constant term: {}",
                            term
                        )));
                    }
                }
            }
        }

        Ok((coeffs, constant))
    }

    fn collect_additive_terms(expr: &Expression) -> Vec<Expression> {
        match expr {
            Expression::Binary(BinaryOp::Add, left, right) => {
                let mut terms = Self::collect_additive_terms(left);
                terms.extend(Self::collect_additive_terms(right));
                terms
            }
            Expression::Binary(BinaryOp::Sub, left, right) => {
                let mut terms = Self::collect_additive_terms(left);
                for term in Self::collect_additive_terms(right) {
                    terms.push(Expression::Unary(UnaryOp::Neg, Box::new(term)));
                }
                terms
            }
            _ => vec![expr.clone()],
        }
    }

    fn extract_coefficient(term: &Expression, var: &Variable) -> SolverResult<f64> {
        match term {
            Expression::Variable(v) if v.name == var.name => Ok(1.0),

            Expression::Unary(UnaryOp::Neg, inner) => Ok(-Self::extract_coefficient(inner, var)?),

            Expression::Binary(BinaryOp::Mul, left, right) => {
                let left_has_var = left.contains_variable(&var.name);
                let right_has_var = right.contains_variable(&var.name);

                if left_has_var && right_has_var {
                    return Err(SolverError::Other(format!(
                        "Non-linear term: {} * {} both contain {}",
                        left, right, var.name
                    )));
                }

                let empty: HashMap<String, f64> = HashMap::new();
                if left_has_var {
                    let coeff = right.evaluate(&empty).ok_or_else(|| {
                        SolverError::Other(format!("Cannot evaluate coefficient: {}", right))
                    })?;
                    Ok(coeff * Self::extract_coefficient(left, var)?)
                } else {
                    let coeff = left.evaluate(&empty).ok_or_else(|| {
                        SolverError::Other(format!("Cannot evaluate coefficient: {}", left))
                    })?;
                    Ok(coeff * Self::extract_coefficient(right, var)?)
                }
            }

            Expression::Binary(BinaryOp::Div, left, right) => {
                if right.contains_variable(&var.name) {
                    return Err(SolverError::Other(format!(
                        "Non-linear: variable {} in denominator",
                        var.name
                    )));
                }
                let empty: HashMap<String, f64> = HashMap::new();
                let divisor = right.evaluate(&empty).ok_or_else(|| {
                    SolverError::Other(format!("Cannot evaluate divisor: {}", right))
                })?;
                if divisor.abs() < 1e-15 {
                    return Err(SolverError::DivisionByZero);
                }
                Ok(Self::extract_coefficient(left, var)? / divisor)
            }

            _ => {
                if term.contains_variable(&var.name) {
                    Err(SolverError::Other(format!(
                        "Cannot extract coefficient from: {}",
                        term
                    )))
                } else {
                    Ok(0.0)
                }
            }
        }
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

// ── free numeric helpers ──────────────────────────────────────────────────────

fn det_2x2(m: &[Vec<f64>]) -> f64 {
    m[0][0] * m[1][1] - m[0][1] * m[1][0]
}

fn det_3x3(m: &[Vec<f64>]) -> f64 {
    let minor1 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let minor2 = m[1][0] * m[2][2] - m[1][2] * m[2][0];
    let minor3 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    m[0][0] * minor1 - m[0][1] * minor2 + m[0][2] * minor3
}

/// Gaussian elimination with partial pivoting.
///
/// Returns a `SystemSolution` from coefficient rows, constant vector and
/// variable list.
fn solve_gaussian(
    coefficients: &[Vec<f64>],
    constants: &[f64],
    variables: &[Variable],
) -> SolverResult<SystemSolution> {
    let n_eqs = coefficients.len();
    let n_vars = variables.len();

    let mut augmented: Vec<Vec<f64>> = coefficients
        .iter()
        .zip(constants.iter())
        .map(|(row, &c)| {
            let mut new_row = row.clone();
            new_row.push(c);
            new_row
        })
        .collect();

    let mut pivot_row = 0;
    let mut pivot_cols: Vec<usize> = Vec::new();

    for col in 0..n_vars {
        if pivot_row >= n_eqs {
            break;
        }
        let (max_row, max_val) = find_pivot(&augmented, pivot_row, n_eqs, col);
        if max_val < 1e-15 {
            continue;
        }
        if max_row != pivot_row {
            augmented.swap(pivot_row, max_row);
        }
        pivot_cols.push(col);
        eliminate_below(&mut augmented, pivot_row, col, n_eqs, n_vars);
        pivot_row += 1;
    }

    let rank = pivot_cols.len();

    for row in rank..n_eqs {
        let rhs = augmented[row][n_vars];
        if augmented[row][0..n_vars].iter().all(|&x| x.abs() < 1e-15) && rhs.abs() > 1e-15 {
            return Ok(SystemSolution::NoSolution);
        }
    }

    if rank == n_vars {
        Ok(back_substitute_unique(&augmented, &pivot_cols, variables))
    } else {
        Ok(build_infinite_solution(
            &augmented,
            &pivot_cols,
            n_vars,
            variables,
        ))
    }
}

fn find_pivot(augmented: &[Vec<f64>], start_row: usize, n_eqs: usize, col: usize) -> (usize, f64) {
    let mut max_row = start_row;
    let mut max_val = augmented[start_row][col].abs();
    for row in (start_row + 1)..n_eqs {
        if augmented[row][col].abs() > max_val {
            max_val = augmented[row][col].abs();
            max_row = row;
        }
    }
    (max_row, max_val)
}

fn eliminate_below(
    augmented: &mut Vec<Vec<f64>>,
    pivot_row: usize,
    col: usize,
    n_eqs: usize,
    n_vars: usize,
) {
    let pivot_val = augmented[pivot_row][col];
    for row in (pivot_row + 1)..n_eqs {
        let factor = augmented[row][col] / pivot_val;
        augmented[row][col] = 0.0;
        for c in (col + 1)..=n_vars {
            augmented[row][c] -= factor * augmented[pivot_row][c];
        }
    }
}

fn f64_to_expr(val: f64) -> Expression {
    if (val - val.round()).abs() < 1e-10 {
        Expression::Integer(val.round() as i64)
    } else {
        Expression::Float(val)
    }
}

fn back_substitute_unique(
    augmented: &[Vec<f64>],
    pivot_cols: &[usize],
    variables: &[Variable],
) -> SystemSolution {
    let rank = pivot_cols.len();
    let n_vars = variables.len();
    let mut solution_values = vec![0.0_f64; n_vars];

    for i in (0..rank).rev() {
        let col = pivot_cols[i];
        let mut sum = augmented[i][n_vars];
        for j in (col + 1)..n_vars {
            sum -= augmented[i][j] * solution_values[j];
        }
        solution_values[col] = sum / augmented[i][col];
    }

    let mut result = HashMap::new();
    for (i, var) in variables.iter().enumerate() {
        result.insert(var.clone(), f64_to_expr(solution_values[i]));
    }
    SystemSolution::Unique(result)
}

fn build_infinite_solution(
    augmented: &[Vec<f64>],
    pivot_cols: &[usize],
    n_vars: usize,
    variables: &[Variable],
) -> SystemSolution {
    let rank = pivot_cols.len();
    let pivot_set: std::collections::HashSet<_> = pivot_cols.iter().cloned().collect();
    let free_cols: Vec<_> = (0..n_vars).filter(|c| !pivot_set.contains(c)).collect();
    let free_vars: Vec<_> = free_cols.iter().map(|&c| variables[c].clone()).collect();

    let mut bound = HashMap::new();

    for i in (0..rank).rev() {
        let col = pivot_cols[i];
        let rhs = augmented[i][n_vars];
        let pivot_coeff = augmented[i][col];

        let mut terms: Vec<Expression> = Vec::new();
        if rhs.abs() > 1e-15 {
            terms.push(f64_to_expr(rhs));
        }

        for &free_col in &free_cols {
            let coeff = -augmented[i][free_col] / pivot_coeff;
            if coeff.abs() > 1e-15 {
                let free_var = Expression::Variable(variables[free_col].clone());
                terms.push(build_coeff_term(coeff, free_var));
            }
        }

        let expr = combine_terms(terms);
        let final_expr = if (pivot_coeff - 1.0).abs() < 1e-15 {
            expr
        } else {
            Expression::Binary(
                BinaryOp::Div,
                Box::new(expr),
                Box::new(f64_to_expr(pivot_coeff)),
            )
        };
        bound.insert(variables[col].clone(), final_expr);
    }

    SystemSolution::Infinite {
        bound,
        free: free_vars,
    }
}

fn build_coeff_term(coeff: f64, free_var: Expression) -> Expression {
    if (coeff - coeff.round()).abs() < 1e-10 {
        let int_coeff = coeff.round() as i64;
        match int_coeff {
            1 => free_var,
            -1 => Expression::Unary(UnaryOp::Neg, Box::new(free_var)),
            _ => Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Integer(int_coeff)),
                Box::new(free_var),
            ),
        }
    } else {
        Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Float(coeff)),
            Box::new(free_var),
        )
    }
}

fn combine_terms(mut terms: Vec<Expression>) -> Expression {
    if terms.is_empty() {
        return Expression::Integer(0);
    }
    if terms.len() == 1 {
        return terms.remove(0);
    }
    let mut result = terms.remove(0);
    for term in terms {
        result = Expression::Binary(BinaryOp::Add, Box::new(result), Box::new(term));
    }
    result
}
