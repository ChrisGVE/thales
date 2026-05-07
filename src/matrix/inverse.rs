//! Matrix inverse, adjugate, cofactor matrix, and singularity check.

use std::sync::Arc;

use crate::numeric::evaluation::evaluate;
use crate::numeric::expr::Expr;
use crate::numeric::normalize;
use crate::numeric::SymbolId;

use super::{MatrixError, MatrixExpr, MatrixResult};

impl MatrixExpr {
    /// Compute the cofactor matrix (matrix of all cofactors).
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square or is 1x1.
    pub fn cofactor_matrix(&self) -> MatrixResult<MatrixExpr> {
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "Cofactor matrix requires a square matrix".to_string(),
            ));
        }
        if self.rows == 1 {
            return Err(MatrixError::InvalidOperation(
                "Cofactor matrix not defined for 1x1 matrix".to_string(),
            ));
        }

        let mut elements = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut row = Vec::with_capacity(self.cols);
            for j in 0..self.cols {
                row.push(self.cofactor(i, j)?);
            }
            elements.push(row);
        }

        MatrixExpr::from_expr_elements(elements)
    }

    /// Compute the adjugate (classical adjoint) matrix.
    ///
    /// The adjugate is the transpose of the cofactor matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::numeric::expr::Expr;
    ///
    /// let m = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(1), Expr::int(2)],
    ///     vec![Expr::int(3), Expr::int(4)],
    /// ]).unwrap();
    ///
    /// let adj = m.adjugate().unwrap();
    /// // adj = [[4, -2], [-3, 1]]
    /// ```
    pub fn adjugate(&self) -> MatrixResult<MatrixExpr> {
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "Adjugate requires a square matrix".to_string(),
            ));
        }

        // Special case for 1x1 matrix
        if self.rows == 1 {
            return Ok(MatrixExpr::from_expr_elements(vec![vec![Expr::int(1)]]).unwrap());
        }

        let cofactor_mat = self.cofactor_matrix()?;
        Ok(cofactor_mat.transpose())
    }

    /// Compute the inverse using the adjugate formula: A^(-1) = adj(A) / det(A).
    ///
    /// Used for matrices up to 3x3.
    pub(crate) fn inverse_adjugate(&self) -> MatrixResult<MatrixExpr> {
        let det = self.determinant()?;

        let is_zero = if det.is_zero() {
            true
        } else {
            let empty: std::collections::HashMap<SymbolId, f64> = std::collections::HashMap::new();
            evaluate(&det, &empty).map_or(false, |v| v.abs() < 1e-10)
        };

        if is_zero {
            return Err(MatrixError::InvalidOperation(
                "Matrix is singular (determinant is zero)".to_string(),
            ));
        }

        // 1x1 case
        if self.rows == 1 {
            let inv_element = normalize::div(Expr::int(1), self.elements[0][0].clone());
            return MatrixExpr::from_expr_elements(vec![vec![inv_element]]);
        }

        let adj = self.adjugate()?;
        let inv_det = normalize::div(Expr::int(1), det);
        Ok(adj.scalar_mul(&inv_det))
    }

    /// Compute the inverse using Gauss-Jordan elimination on the augmented matrix [A | I].
    ///
    /// Partial pivoting (row swaps) is used to find a non-zero pivot in each column.
    /// Each pivot row is scaled to a leading 1, then elimination clears all other
    /// entries in that column (both above and below). After full reduction the right
    /// half of the augmented matrix is the inverse.
    ///
    /// # Errors
    ///
    /// Returns `InvalidOperation` if a zero column is encountered (singular matrix).
    pub(crate) fn inverse_gauss_jordan(&self) -> MatrixResult<MatrixExpr> {
        let n = self.rows;

        // Build augmented matrix [A | I] with 2n columns.
        let mut aug: Vec<Vec<Arc<Expr>>> = (0..n)
            .map(|i| {
                let mut row: Vec<Arc<Expr>> = self.elements[i].clone();
                for j in 0..n {
                    row.push(if i == j { Expr::int(1) } else { Expr::int(0) });
                }
                row
            })
            .collect();

        let ncols = 2 * n;

        for col in 0..n {
            // Find a non-zero pivot at or below `col`.
            let pivot_row = (col..n).find(|&r| !aug[r][col].is_zero());

            let pivot_row = match pivot_row {
                Some(r) => r,
                None => {
                    return Err(MatrixError::InvalidOperation("Singular matrix".to_string()));
                }
            };

            if pivot_row != col {
                aug.swap(col, pivot_row);
            }

            // Scale pivot row so aug[col][col] == 1.
            let pivot_val = aug[col][col].clone();
            let scaled: Vec<Arc<Expr>> = (0..ncols)
                .map(|j| normalize::div(aug[col][j].clone(), pivot_val.clone()))
                .collect();
            aug[col] = scaled;

            // Eliminate all other rows in this column.
            for r in 0..n {
                if r == col {
                    continue;
                }
                let factor = aug[r][col].clone();
                if factor.is_zero() {
                    continue;
                }
                let new_row: Vec<Arc<Expr>> = (0..ncols)
                    .map(|j| {
                        let sub = normalize::mul(factor.clone(), aug[col][j].clone());
                        normalize::sub(aug[r][j].clone(), sub)
                    })
                    .collect();
                aug[r] = new_row;
            }
        }

        // Extract the right half (columns n..2n) as the inverse.
        let inv_elements: Vec<Vec<Arc<Expr>>> = aug
            .into_iter()
            .map(|row| row.into_iter().skip(n).collect())
            .collect();

        MatrixExpr::from_expr_elements(inv_elements)
    }

    /// Compute the inverse of the matrix.
    ///
    /// Dispatches to the adjugate method for n <= 3 and Gauss-Jordan elimination
    /// for n > 3. Both paths detect singular matrices and return an error.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The matrix is not square
    /// - The matrix is singular (determinant is zero / zero pivot column)
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
    /// let m = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(4), Expr::int(7)],
    ///     vec![Expr::int(2), Expr::int(6)],
    /// ]).unwrap();
    ///
    /// let inv = m.inverse().unwrap();
    /// // Verify A * A^(-1) = I
    /// let product = m.mul(&inv).unwrap();
    /// let empty: HashMap<SymbolId, f64> = HashMap::new();
    /// let result = product.evaluate(&empty).unwrap();
    /// assert!((result[0][0] - 1.0).abs() < 1e-10);
    /// assert!((result[1][1] - 1.0).abs() < 1e-10);
    /// ```
    pub fn inverse(&self) -> MatrixResult<MatrixExpr> {
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "Inverse requires a square matrix".to_string(),
            ));
        }

        if self.rows <= 3 {
            self.inverse_adjugate()
        } else {
            self.inverse_gauss_jordan()
        }
    }

    /// Check if the matrix is singular (determinant is zero when evaluated numerically).
    ///
    /// Returns `None` if the determinant cannot be evaluated numerically.
    pub fn is_singular(&self, vars: &std::collections::HashMap<SymbolId, f64>) -> Option<bool> {
        let det = self.determinant().ok()?;
        let det_value = evaluate(&det, vars)?;
        Some(det_value.abs() < 1e-10)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::numeric::{evaluation::evaluate, SymbolId};

    use super::*;

    fn mat(rows: Vec<Vec<i64>>) -> MatrixExpr {
        let elems = rows
            .into_iter()
            .map(|row| row.into_iter().map(Expr::int).collect())
            .collect();
        MatrixExpr::from_expr_elements(elems).unwrap()
    }

    fn to_f64(m: &MatrixExpr) -> Vec<Vec<f64>> {
        let empty: HashMap<SymbolId, f64> = HashMap::new();
        m.evaluate(&empty).expect("matrix must be fully numeric")
    }

    fn approx_eq(a: f64, b: f64) {
        assert!(
            (a - b).abs() < 1e-9,
            "expected {b:.12}, got {a:.12}, diff = {:.3e}",
            (a - b).abs()
        );
    }

    fn assert_identity(m: &MatrixExpr) {
        let data = to_f64(m);
        let n = m.rows();
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                approx_eq(data[i][j], expected);
            }
        }
    }

    #[test]
    fn fast_test_gj_inverse_matches_adjugate_2x2() {
        let m = mat(vec![vec![4, 7], vec![2, 6]]);
        let adj_inv = m.inverse_adjugate().unwrap();
        let gj_inv = m.inverse_gauss_jordan().unwrap();
        let adj_data = to_f64(&adj_inv);
        let gj_data = to_f64(&gj_inv);
        for i in 0..2 {
            for j in 0..2 {
                approx_eq(adj_data[i][j], gj_data[i][j]);
            }
        }
    }

    #[test]
    fn fast_test_gj_inverse_matches_adjugate_3x3() {
        let m = mat(vec![vec![1, 2, 3], vec![0, 1, 4], vec![5, 6, 0]]);
        let adj_inv = m.inverse_adjugate().unwrap();
        let gj_inv = m.inverse_gauss_jordan().unwrap();
        let adj_data = to_f64(&adj_inv);
        let gj_data = to_f64(&gj_inv);
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(adj_data[i][j], gj_data[i][j]);
            }
        }
    }

    #[test]
    fn fast_test_gj_inverse_4x4() {
        // Upper triangular with easy inverse.
        let m = mat(vec![
            vec![1, 2, 3, 4],
            vec![0, 2, 3, 4],
            vec![0, 0, 3, 4],
            vec![0, 0, 0, 4],
        ]);
        let inv = m.inverse().unwrap();
        // Verify A * A^(-1) = I
        let product = m.mul(&inv).unwrap();
        assert_identity(&product);
    }

    #[test]
    fn fast_test_gj_inverse_5x5() {
        // Diagonal matrix: inverse is diagonal with reciprocals.
        let mut rows = vec![vec![0i64; 5]; 5];
        for i in 0..5 {
            rows[i][i] = (i as i64) + 1;
        }
        let m = mat(rows);
        let inv = m.inverse().unwrap();
        let product = m.mul(&inv).unwrap();
        assert_identity(&product);
    }

    #[test]
    fn fast_test_gj_inverse_singular() {
        // Row 1 is 2 * row 0 — singular.
        let m = mat(vec![
            vec![1, 2, 3, 4],
            vec![2, 4, 6, 8],
            vec![0, 1, 0, 0],
            vec![0, 0, 1, 0],
        ]);
        let result = m.inverse();
        assert!(result.is_err(), "singular matrix must return an error");
        match result.unwrap_err() {
            MatrixError::InvalidOperation(msg) => {
                assert!(
                    msg.contains("ingular") || msg.contains("zero"),
                    "unexpected error message: {msg}"
                );
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn fast_test_gj_inverse_dispatch_uses_adjugate_for_small() {
        // Verify the public inverse() dispatches to adjugate for n <= 3.
        let m2 = mat(vec![vec![1, 2], vec![3, 4]]);
        let m3 = mat(vec![vec![1, 2, 3], vec![0, 1, 4], vec![5, 6, 0]]);
        // Both must succeed and give A * A^(-1) = I.
        let product2 = m2.mul(&m2.inverse().unwrap()).unwrap();
        let product3 = m3.mul(&m3.inverse().unwrap()).unwrap();
        assert_identity(&product2);
        assert_identity(&product3);
    }

    #[test]
    fn fast_test_gj_inverse_with_row_swap() {
        // First column has a zero in the first row, forcing a pivot swap.
        let m = mat(vec![
            vec![0, 1, 0, 0],
            vec![1, 0, 0, 0],
            vec![0, 0, 2, 1],
            vec![0, 0, 3, 2],
        ]);
        let inv = m.inverse().unwrap();
        let product = m.mul(&inv).unwrap();
        assert_identity(&product);
    }

    #[test]
    fn fast_test_inverse_existing_2x2() {
        // Regression: existing test from doc-example still passes.
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(4), Expr::int(7)],
            vec![Expr::int(2), Expr::int(6)],
        ])
        .unwrap();
        let inv = m.inverse().unwrap();
        let product = m.mul(&inv).unwrap();
        assert_identity(&product);
    }

    #[test]
    fn fast_test_gj_evaluate_inverse_entries_4x4() {
        // Explicitly check numerical values of a known 4x4 inverse.
        // A = diag(2, 3, 4, 5): A^-1 = diag(0.5, 1/3, 0.25, 0.2)
        let mut rows = vec![vec![0i64; 4]; 4];
        for i in 0..4 {
            rows[i][i] = (i as i64) + 2;
        }
        let m = mat(rows);
        let inv = m.inverse().unwrap();
        let data = to_f64(&inv);
        let expected_diag = [0.5, 1.0 / 3.0, 0.25, 0.2];
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    approx_eq(data[i][j], expected_diag[i]);
                } else {
                    approx_eq(data[i][j], 0.0);
                }
            }
        }
    }

    #[test]
    fn fast_test_gj_inverse_non_square_error() {
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(2), Expr::int(3)],
            vec![Expr::int(4), Expr::int(5), Expr::int(6)],
        ])
        .unwrap();
        assert!(m.inverse().is_err());
    }
}
