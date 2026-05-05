//! LU decomposition with partial pivoting for [`MatrixExpr`].

use std::sync::Arc;

use crate::numeric::expr::Expr;

use super::{MatrixError, MatrixExpr, MatrixResult};

// ── private numeric helpers ───────────────────────────────────────────────────

/// Forward substitution: solve L·y = b where L is unit lower-triangular.
fn forward_substitute(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let sum: f64 = (0..i).map(|k| l[i][k] * y[k]).sum();
        y[i] = b[i] - sum; // L[i][i] == 1 for unit lower triangular
    }
    y
}

/// Back substitution: solve U·x = y where U is upper-triangular.
fn back_substitute(u: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let sum: f64 = (i + 1..n).map(|k| u[i][k] * x[k]).sum();
        x[i] = (y[i] - sum) / u[i][i];
    }
    x
}

/// Evaluate the matrix numerically using an empty variable map.
///
/// Returns an error when any element contains unresolved symbolic variables.
fn to_numeric(m: &MatrixExpr) -> MatrixResult<Vec<Vec<f64>>> {
    let empty: std::collections::HashMap<crate::numeric::SymbolId, f64> =
        std::collections::HashMap::new();
    m.evaluate(&empty).ok_or_else(|| {
        MatrixError::InvalidOperation(
            "Matrix contains unresolved symbolic variables; \
             LU decomposition requires a numeric matrix"
                .to_string(),
        )
    })
}

/// Wrap a 2-D `f64` grid into a `MatrixExpr` of `Expr::Float` values.
fn from_numeric(data: Vec<Vec<f64>>) -> MatrixExpr {
    let rows = data.len();
    let cols = if rows == 0 { 0 } else { data[0].len() };
    let elements: Vec<Vec<Arc<Expr>>> = data
        .into_iter()
        .map(|row| row.into_iter().map(Expr::float).collect())
        .collect();
    MatrixExpr::from_expr_elements_unchecked(rows, cols, elements)
}

// ── LU decomposition core (Doolittle with partial pivoting) ──────────────────

/// Perform Doolittle LU factorisation with partial pivoting on a numeric matrix.
///
/// Returns `(l_data, u_data, perm)` where `perm[i]` is the source row for row `i`.
fn lu_numeric(a: Vec<Vec<f64>>) -> MatrixResult<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<usize>)> {
    let n = a.len();
    let mut lu = a; // work in-place
    let mut perm: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Find pivot row: row with max absolute value in column k from row k onward.
        let pivot_row = (k..n)
            .max_by(|&i, &j| lu[i][k].abs().partial_cmp(&lu[j][k].abs()).unwrap())
            .unwrap();

        if lu[pivot_row][k].abs() < 1e-12 {
            return Err(MatrixError::InvalidOperation(
                "Matrix is singular or near-singular; LU decomposition failed".to_string(),
            ));
        }

        // Swap rows k and pivot_row.
        if pivot_row != k {
            lu.swap(k, pivot_row);
            perm.swap(k, pivot_row);
        }

        // Eliminate entries below pivot.
        for i in (k + 1)..n {
            lu[i][k] /= lu[k][k];
            for j in (k + 1)..n {
                let factor = lu[i][k];
                lu[i][j] -= factor * lu[k][j];
            }
        }
    }

    // Split the combined lu matrix into L and U.
    let mut l_data = vec![vec![0.0_f64; n]; n];
    let mut u_data = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i > j {
                l_data[i][j] = lu[i][j];
            } else if i == j {
                l_data[i][j] = 1.0; // unit diagonal
                u_data[i][j] = lu[i][j];
            } else {
                u_data[i][j] = lu[i][j];
            }
        }
    }

    Ok((l_data, u_data, perm))
}

// ── public impl block ─────────────────────────────────────────────────────────

impl MatrixExpr {
    /// Compute the LU decomposition with partial pivoting.
    ///
    /// Since pivoting is a numeric operation, all matrix elements must evaluate
    /// to concrete `f64` values (no unresolved symbolic variables).
    ///
    /// Returns `(L, U, perm)` such that `P·A = L·U`, where:
    /// - `L` is unit lower-triangular,
    /// - `U` is upper-triangular,
    /// - `perm[i]` gives the original row index that was permuted to row `i`.
    ///
    /// # Errors
    ///
    /// - `InvalidOperation` if the matrix is not square.
    /// - `InvalidOperation` if any element contains unresolved variables.
    /// - `InvalidOperation` if the matrix is singular or near-singular.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::numeric::expr::Expr;
    ///
    /// let a = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(4), Expr::int(3)],
    ///     vec![Expr::int(6), Expr::int(3)],
    /// ]).unwrap();
    ///
    /// let (l, u, perm) = a.lu_decompose().unwrap();
    /// assert_eq!(l.rows(), 2);
    /// assert_eq!(u.rows(), 2);
    /// assert_eq!(perm.len(), 2);
    /// ```
    pub fn lu_decompose(&self) -> MatrixResult<(MatrixExpr, MatrixExpr, Vec<usize>)> {
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "LU decomposition requires a square matrix".to_string(),
            ));
        }

        let numeric = to_numeric(self)?;
        let (l_data, u_data, perm) = lu_numeric(numeric)?;

        Ok((from_numeric(l_data), from_numeric(u_data), perm))
    }

    /// Solve the linear system `A·x = b` using LU decomposition.
    ///
    /// `b` must be a column vector (single-column [`MatrixExpr`]) with the same
    /// number of rows as `self`.
    ///
    /// Returns `x` as a column [`MatrixExpr`].
    ///
    /// # Errors
    ///
    /// - `InvalidOperation` if `self` is not square.
    /// - `DimensionMismatch` if `b` does not have matching row count or is not a column vector.
    /// - `InvalidOperation` if any element contains unresolved variables.
    /// - `InvalidOperation` if the matrix is singular or near-singular.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::numeric::expr::Expr;
    /// use thales::numeric::evaluation::evaluate;
    /// use thales::numeric::SymbolId;
    /// use std::collections::HashMap;
    ///
    /// // Solve [[2, 1], [1, 3]] x = [[3], [4]]
    /// let a = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(2), Expr::int(1)],
    ///     vec![Expr::int(1), Expr::int(3)],
    /// ]).unwrap();
    ///
    /// let b = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(3)],
    ///     vec![Expr::int(4)],
    /// ]).unwrap();
    ///
    /// let x = a.solve_system(&b).unwrap();
    /// let empty: HashMap<SymbolId, f64> = HashMap::new();
    /// // Solution: x = [1, 1]
    /// assert!((evaluate(x.get(0, 0).unwrap(), &empty).unwrap() - 1.0).abs() < 1e-10);
    /// assert!((evaluate(x.get(1, 0).unwrap(), &empty).unwrap() - 1.0).abs() < 1e-10);
    /// ```
    pub fn solve_system(&self, b: &MatrixExpr) -> MatrixResult<MatrixExpr> {
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "solve_system requires a square coefficient matrix".to_string(),
            ));
        }
        if b.cols() != 1 {
            return Err(MatrixError::InvalidOperation(
                "solve_system requires b to be a column vector (cols == 1)".to_string(),
            ));
        }
        if b.rows() != self.rows() {
            return Err(MatrixError::DimensionMismatch {
                operation: "solve_system".to_string(),
                expected: (self.rows(), 1),
                got: (b.rows(), 1),
            });
        }

        let a_num = to_numeric(self)?;
        let b_num = to_numeric(b)?;

        let (l_data, u_data, perm) = lu_numeric(a_num)?;

        // Apply permutation to b.
        let b_vec: Vec<f64> = perm.iter().map(|&i| b_num[i][0]).collect();

        let y = forward_substitute(&l_data, &b_vec);
        let x = back_substitute(&u_data, &y);

        let elements: Vec<Vec<Arc<Expr>>> = x.into_iter().map(|v| vec![Expr::float(v)]).collect();
        let n = elements.len();
        Ok(MatrixExpr::from_expr_elements_unchecked(n, 1, elements))
    }
}
