//! Null space (kernel) and column space basis extraction for [`MatrixExpr`].
//!
//! Both operations rely on the RREF from [`row_echelon`](super::row_echelon).
//! Pivot columns identify the rank structure; free columns drive null-space
//! parametrisation; the column space uses original (pre-RREF) columns.

use std::sync::Arc;

use crate::numeric::{expr::Expr, normalize};

use super::{MatrixError, MatrixExpr, MatrixResult};

impl MatrixExpr {
    /// Compute a basis for the null space (kernel) of `self`.
    ///
    /// Returns a `Vec` of `n×1` column vectors spanning `{x : Ax = 0}`.
    /// An empty `Vec` means the null space is trivial (full column rank).
    ///
    /// The algorithm computes RREF and reads off the free-variable columns.
    /// For each free variable `j`, a basis vector is built by setting `x_j = 1`,
    /// all other free variables to `0`, and back-solving each pivot row for the
    /// corresponding pivot variable: `x_pivot = −rref[pivot_row][j]`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidOperation` if the matrix is empty (0 rows or 0 cols).
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::numeric::expr::Expr;
    ///
    /// // [[1, 2, 3]] has a 2-dimensional null space.
    /// let m = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(1), Expr::int(2), Expr::int(3)],
    /// ]).unwrap();
    ///
    /// let basis = m.kernel().unwrap();
    /// assert_eq!(basis.len(), 2);
    /// ```
    pub fn kernel(&self) -> MatrixResult<Vec<MatrixExpr>> {
        if self.rows == 0 || self.cols == 0 {
            return Err(MatrixError::InvalidOperation(
                "kernel requires a non-empty matrix".to_string(),
            ));
        }

        let (rref, pivot_cols) = self.rref()?;
        let n = self.cols;

        // Identify free variable columns (all columns that are not pivot columns).
        let free_cols: Vec<usize> = (0..n).filter(|c| !pivot_cols.contains(c)).collect();

        // Build a map: pivot_col → row index in RREF.
        // pivot_cols[row] = col means row `row` has its pivot in column `col`.
        // We invert it: pivot_col_to_row[col] = row.
        let pivot_col_to_row: std::collections::HashMap<usize, usize> = pivot_cols
            .iter()
            .enumerate()
            .map(|(row, &col)| (col, row))
            .collect();

        let mut basis: Vec<MatrixExpr> = Vec::with_capacity(free_cols.len());

        for &free_j in &free_cols {
            // Allocate the basis vector (n×1): default all entries to 0.
            let mut vec_entries: Vec<Arc<Expr>> = (0..n).map(|_| Expr::int(0)).collect();

            // Free variable gets value 1.
            vec_entries[free_j] = Expr::int(1);

            // For each pivot row, back-solve: x_pivot = −rref[pivot_row][free_j].
            for (&pcol, &prow) in &pivot_col_to_row {
                let coeff = rref.elements[prow][free_j].clone();
                vec_entries[pcol] = normalize::neg(coeff);
            }

            // Wrap as n×1 column matrix (each row is a single-element Vec).
            let col_elements: Vec<Vec<Arc<Expr>>> =
                vec_entries.into_iter().map(|e| vec![e]).collect();
            basis.push(MatrixExpr::from_expr_elements(col_elements)?);
        }

        Ok(basis)
    }

    /// Compute a basis for the column space of `self`.
    ///
    /// Returns the pivot columns of `self` (not of the RREF) as `m×1` column
    /// vectors.  The number of returned vectors equals the rank of `self`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidOperation` if the matrix is empty (0 rows or 0 cols).
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::numeric::expr::Expr;
    ///
    /// let m = MatrixExpr::identity(3);
    /// let basis = m.column_space().unwrap();
    /// assert_eq!(basis.len(), 3);
    /// ```
    pub fn column_space(&self) -> MatrixResult<Vec<MatrixExpr>> {
        if self.rows == 0 || self.cols == 0 {
            return Err(MatrixError::InvalidOperation(
                "column_space requires a non-empty matrix".to_string(),
            ));
        }

        let (_, pivot_cols) = self.rref()?;

        pivot_cols
            .iter()
            .map(|&col| {
                // Extract column `col` from the original matrix as an m×1 matrix.
                let col_elements: Vec<Vec<Arc<Expr>>> = (0..self.rows)
                    .map(|row| vec![self.elements[row][col].clone()])
                    .collect();
                MatrixExpr::from_expr_elements(col_elements)
            })
            .collect()
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::matrix::MatrixExpr;
    use crate::numeric::{evaluation::evaluate, expr::Expr, SymbolId};

    fn eval_f(e: &std::sync::Arc<Expr>) -> f64 {
        let empty: HashMap<SymbolId, f64> = HashMap::new();
        evaluate(e, &empty).expect("expression must be fully numeric")
    }

    fn approx(a: f64, b: f64) {
        assert!(
            (a - b).abs() < 1e-9,
            "expected {b:.12}, got {a:.12}, diff = {:.3e}",
            (a - b).abs()
        );
    }

    // ── kernel tests ─────────────────────────────────────────────────────────

    #[test]
    fn fast_test_kernel_identity() {
        // Identity has full column rank — trivial null space.
        let basis = MatrixExpr::identity(3).kernel().unwrap();
        assert!(basis.is_empty(), "identity has no null vectors");
    }

    #[test]
    fn fast_test_kernel_zero_matrix() {
        // Zero 3×3: every column is free — kernel dimension = 3.
        let basis = MatrixExpr::zero(3, 3).kernel().unwrap();
        assert_eq!(basis.len(), 3, "zero 3×3 kernel must have dimension 3");
        for v in &basis {
            assert_eq!(v.rows(), 3);
            assert_eq!(v.cols(), 1);
        }
    }

    #[test]
    fn fast_test_kernel_rank_1_row() {
        // [[1, 2, 3]]: rank 1 → kernel dimension 2.
        let m =
            MatrixExpr::from_expr_elements(vec![vec![Expr::int(1), Expr::int(2), Expr::int(3)]])
                .unwrap();
        let basis = m.kernel().unwrap();
        assert_eq!(basis.len(), 2, "1×3 rank-1 matrix has 2-dim kernel");
        for v in &basis {
            assert_eq!(v.rows(), 3);
            assert_eq!(v.cols(), 1);
        }
    }

    #[test]
    fn fast_test_kernel_rank_deficient_3x3() {
        // [[1,2,3],[4,5,6],[7,8,9]]: rank 2 → kernel dimension 1.
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(2), Expr::int(3)],
            vec![Expr::int(4), Expr::int(5), Expr::int(6)],
            vec![Expr::int(7), Expr::int(8), Expr::int(9)],
        ])
        .unwrap();
        let basis = m.kernel().unwrap();
        assert_eq!(basis.len(), 1, "rank-2 matrix has 1-dim kernel");
        assert_eq!(basis[0].rows(), 3);
        assert_eq!(basis[0].cols(), 1);
    }

    #[test]
    fn fast_test_kernel_verify_av_zero() {
        // [[1,2,3],[4,5,6],[7,8,9]]: verify Av = 0 for each kernel vector.
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(2), Expr::int(3)],
            vec![Expr::int(4), Expr::int(5), Expr::int(6)],
            vec![Expr::int(7), Expr::int(8), Expr::int(9)],
        ])
        .unwrap();
        let basis = m.kernel().unwrap();
        for v in &basis {
            // A * v should be the zero vector (m×1).
            let av = m.mul(v).unwrap();
            assert_eq!(av.rows(), m.rows());
            assert_eq!(av.cols(), 1);
            for row in 0..av.rows() {
                approx(eval_f(av.get(row, 0).unwrap()), 0.0);
            }
        }
    }

    // ── column_space tests ───────────────────────────────────────────────────

    #[test]
    fn fast_test_colspace_identity() {
        // Identity 3×3: all 3 columns are pivot columns.
        let basis = MatrixExpr::identity(3).column_space().unwrap();
        assert_eq!(basis.len(), 3, "identity(3) column space has 3 vectors");
        for v in &basis {
            assert_eq!(v.rows(), 3);
            assert_eq!(v.cols(), 1);
        }
    }

    #[test]
    fn fast_test_colspace_dimension() {
        // [[1,2,3],[4,5,6],[7,8,9]]: rank 2 → 2 column-space vectors.
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(2), Expr::int(3)],
            vec![Expr::int(4), Expr::int(5), Expr::int(6)],
            vec![Expr::int(7), Expr::int(8), Expr::int(9)],
        ])
        .unwrap();
        let basis = m.column_space().unwrap();
        assert_eq!(basis.len(), m.rank().unwrap());
    }

    #[test]
    fn fast_test_colspace_rectangular() {
        // 3×4 matrix with rank 3: should produce 3 column vectors.
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(2), Expr::int(3), Expr::int(4)],
            vec![Expr::int(0), Expr::int(1), Expr::int(0), Expr::int(2)],
            vec![Expr::int(0), Expr::int(0), Expr::int(1), Expr::int(3)],
        ])
        .unwrap();
        let basis = m.column_space().unwrap();
        assert_eq!(basis.len(), 3);
        for v in &basis {
            assert_eq!(v.rows(), 3);
            assert_eq!(v.cols(), 1);
        }
    }

    #[test]
    fn fast_test_colspace_from_original() {
        // Verify the returned vectors come from the original matrix (not RREF).
        // Use a rank-1 matrix where RREF col-0 is [1,0]ᵀ but original col-0 is [2,4]ᵀ.
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(2), Expr::int(6)],
            vec![Expr::int(4), Expr::int(12)],
        ])
        .unwrap();
        let basis = m.column_space().unwrap();
        assert_eq!(basis.len(), 1, "rank-1 matrix has 1-dim column space");
        // The single basis vector should be the first original column [2, 4]ᵀ.
        approx(eval_f(basis[0].get(0, 0).unwrap()), 2.0);
        approx(eval_f(basis[0].get(1, 0).unwrap()), 4.0);
    }
}
