//! Row echelon form (REF) and reduced row echelon form (RREF) for [`MatrixExpr`].
//!
//! Implements Gaussian elimination with symbolic pivot selection. Both forms
//! return the transformed matrix together with a `Vec<usize>` of pivot column
//! indices that downstream tasks (rank, null space, column space) consume.

use std::sync::Arc;

use crate::numeric::{expr::Expr, normalize};

use super::{MatrixError, MatrixExpr, MatrixResult};

// ── private helpers ───────────────────────────────────────────────────────────

/// Return `true` when the expression is structurally the integer/float zero.
///
/// This is intentionally conservative: a symbolic expression that evaluates
/// to zero (e.g. `x - x`) is NOT considered zero because we have no general
/// algebraic simplifier that guarantees structural collapse.  Downstream code
/// that needs a stronger check must normalise the expression first.
fn is_structural_zero(e: &Arc<Expr>) -> bool {
    e.is_zero()
}

/// Search column `col` of `rows` for the first non-zero entry at or below
/// `from_row`.  Returns `Some(row_index)` or `None` if all entries are zero.
fn find_pivot_row(rows: &[Vec<Arc<Expr>>], from_row: usize, col: usize) -> Option<usize> {
    rows[from_row..]
        .iter()
        .position(|row| !is_structural_zero(&row[col]))
        .map(|offset| from_row + offset)
}

/// Eliminate all entries in column `col` for rows other than `pivot_row`.
///
/// For each target row `r != pivot_row`:
///   row[r] = row[r] * pivot - row[r][col] * row[pivot_row]
///
/// The multiply-across form avoids introducing `div` into off-pivot rows,
/// keeping expressions smaller when working symbolically.  Callers that want
/// a unit pivot should normalise the pivot row before calling this helper.
fn eliminate_col(rows: &mut Vec<Vec<Arc<Expr>>>, pivot_row: usize, col: usize, only_below: bool) {
    let ncols = rows[0].len();
    let nrows = rows.len();
    let pivot_val = rows[pivot_row][col].clone();

    for r in 0..nrows {
        if r == pivot_row {
            continue;
        }
        if only_below && r < pivot_row {
            continue;
        }
        let factor = rows[r][col].clone();
        if is_structural_zero(&factor) {
            continue;
        }
        // new_row[j] = factor_row[j] * pivot_val - factor * pivot_row[j]
        let new_row: Vec<Arc<Expr>> = (0..ncols)
            .map(|j| {
                let lhs = normalize::mul(rows[r][j].clone(), pivot_val.clone());
                let rhs = normalize::mul(factor.clone(), rows[pivot_row][j].clone());
                normalize::sub(lhs, rhs)
            })
            .collect();
        rows[r] = new_row;
    }
}

// ── public impl ──────────────────────────────────────────────────────────────

impl MatrixExpr {
    /// Compute the row echelon form (REF) via Gaussian elimination.
    ///
    /// Returns `(ref_matrix, pivot_cols)` where `pivot_cols[i]` is the column
    /// index of the leading non-zero entry in row `i`.  Rows that are entirely
    /// zero have no pivot entry and are moved to the bottom.
    ///
    /// The algorithm uses a multiply-across elimination step that avoids
    /// symbolic division during forward elimination, keeping intermediate
    /// expressions well-formed even when pivots are symbolic.
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
    /// let m = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(1), Expr::int(2)],
    ///     vec![Expr::int(3), Expr::int(4)],
    /// ]).unwrap();
    ///
    /// let (ref_m, pivots) = m.ref_form().unwrap();
    /// assert_eq!(pivots.len(), 2);
    /// ```
    pub fn ref_form(&self) -> MatrixResult<(MatrixExpr, Vec<usize>)> {
        if self.rows == 0 || self.cols == 0 {
            return Err(MatrixError::InvalidOperation(
                "REF requires a non-empty matrix".to_string(),
            ));
        }

        let mut rows: Vec<Vec<Arc<Expr>>> = self.elements.clone();
        let mut pivot_cols: Vec<usize> = Vec::new();
        let mut current_row = 0;

        for col in 0..self.cols {
            if current_row >= self.rows {
                break;
            }

            match find_pivot_row(&rows, current_row, col) {
                None => continue, // entire column below current_row is zero
                Some(pivot_row) => {
                    if pivot_row != current_row {
                        rows.swap(current_row, pivot_row);
                    }
                    eliminate_col(&mut rows, current_row, col, true);
                    pivot_cols.push(col);
                    current_row += 1;
                }
            }
        }

        Ok((
            MatrixExpr::from_expr_elements_unchecked(self.rows, self.cols, rows),
            pivot_cols,
        ))
    }

    /// Compute the reduced row echelon form (RREF).
    ///
    /// Builds on [`ref_form`]: after forward elimination each pivot row is
    /// normalised to have a leading 1 (via `pivot / pivot`), then back-
    /// substitution eliminates entries above each pivot.
    ///
    /// Returns `(rref_matrix, pivot_cols)` with the same pivot semantics as
    /// [`ref_form`].
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
    /// let m = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(1), Expr::int(2)],
    ///     vec![Expr::int(3), Expr::int(4)],
    /// ]).unwrap();
    ///
    /// let (rref_m, pivots) = m.rref().unwrap();
    /// assert_eq!(pivots.len(), 2);
    /// // Leading entry of each pivot row is 1.
    /// assert!(rref_m.get(0, pivots[0]).unwrap().is_one());
    /// assert!(rref_m.get(1, pivots[1]).unwrap().is_one());
    /// ```
    pub fn rref(&self) -> MatrixResult<(MatrixExpr, Vec<usize>)> {
        let (ref_mat, pivot_cols) = self.ref_form()?;
        let mut rows: Vec<Vec<Arc<Expr>>> = ref_mat.elements;

        // Normalise each pivot row so the leading entry becomes 1,
        // then back-substitute to clear entries above the pivot.
        for (pivot_row_idx, &col) in pivot_cols.iter().enumerate() {
            let pivot_val = rows[pivot_row_idx][col].clone();
            if is_structural_zero(&pivot_val) {
                continue;
            }

            // Divide every element of the pivot row by the pivot value.
            let ncols = rows[0].len();
            let normalised: Vec<Arc<Expr>> = (0..ncols)
                .map(|j| normalize::div(rows[pivot_row_idx][j].clone(), pivot_val.clone()))
                .collect();
            rows[pivot_row_idx] = normalised;

            // Back-substitute: eliminate entries above this pivot column.
            for r in 0..pivot_row_idx {
                let factor = rows[r][col].clone();
                if is_structural_zero(&factor) {
                    continue;
                }
                let new_row: Vec<Arc<Expr>> = (0..ncols)
                    .map(|j| {
                        let sub_val =
                            normalize::mul(factor.clone(), rows[pivot_row_idx][j].clone());
                        normalize::sub(rows[r][j].clone(), sub_val)
                    })
                    .collect();
                rows[r] = new_row;
            }
        }

        Ok((
            MatrixExpr::from_expr_elements_unchecked(self.rows, self.cols, rows),
            pivot_cols,
        ))
    }

    /// Compute the rank of the matrix.
    ///
    /// The rank equals the number of pivot columns in the RREF, which is the
    /// dimension of the row space (equivalently, the column space).
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
    /// assert_eq!(m.rank().unwrap(), 3);
    ///
    /// let z = MatrixExpr::zero(2, 3);
    /// assert_eq!(z.rank().unwrap(), 0);
    /// ```
    pub fn rank(&self) -> MatrixResult<usize> {
        let (_, pivots) = self.rref()?;
        Ok(pivots.len())
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::numeric::{evaluation::evaluate, SymbolId};

    use super::*;

    /// Evaluate a single `Arc<Expr>` to f64, panicking on failure.
    fn eval(e: &Arc<Expr>) -> f64 {
        let empty: HashMap<SymbolId, f64> = HashMap::new();
        evaluate(e, &empty).expect("expression must be fully numeric")
    }

    /// Evaluate a MatrixExpr to Vec<Vec<f64>>, panicking on failure.
    fn to_f64(m: &MatrixExpr) -> Vec<Vec<f64>> {
        let empty: HashMap<SymbolId, f64> = HashMap::new();
        m.evaluate(&empty).expect("matrix must be fully numeric")
    }

    /// Assert two f64 values are close to machine precision.
    fn approx_eq(a: f64, b: f64) {
        assert!(
            (a - b).abs() < 1e-9,
            "expected {b:.12}, got {a:.12}, diff = {:.3e}",
            (a - b).abs()
        );
    }

    /// Verify that every entry below the diagonal is zero in a matrix.
    fn assert_upper_triangular(m: &MatrixExpr) {
        let data = to_f64(m);
        for i in 0..m.rows() {
            for j in 0..i.min(m.cols()) {
                approx_eq(data[i][j], 0.0);
            }
        }
    }

    #[test]
    fn fast_ref_2x2() {
        // [[1, 2], [3, 4]]
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(2)],
            vec![Expr::int(3), Expr::int(4)],
        ])
        .unwrap();

        let (ref_m, pivots) = m.ref_form().unwrap();
        assert_eq!(pivots, vec![0, 1]);
        assert_upper_triangular(&ref_m);
        // The (1,0) entry must be zero.
        approx_eq(eval(ref_m.get(1, 0).unwrap()), 0.0);
    }

    #[test]
    fn fast_ref_3x3() {
        // [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(2), Expr::int(1), Expr::int(-1)],
            vec![Expr::int(-3), Expr::int(-1), Expr::int(2)],
            vec![Expr::int(-2), Expr::int(1), Expr::int(2)],
        ])
        .unwrap();

        let (ref_m, pivots) = m.ref_form().unwrap();
        // Full-rank 3x3 should have 3 pivots.
        assert_eq!(pivots.len(), 3);
        assert_upper_triangular(&ref_m);
    }

    #[test]
    fn fast_ref_rectangular_3x4() {
        // 3×4 matrix — should produce 3 pivots (full row rank).
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(2), Expr::int(3), Expr::int(4)],
            vec![Expr::int(2), Expr::int(4), Expr::int(7), Expr::int(8)],
            vec![Expr::int(3), Expr::int(6), Expr::int(10), Expr::int(14)],
        ])
        .unwrap();

        let (ref_m, pivots) = m.ref_form().unwrap();
        assert_eq!(ref_m.rows(), 3);
        assert_eq!(ref_m.cols(), 4);
        // Entries below each pivot must be zero.
        for (row, &col) in pivots.iter().enumerate() {
            for r in (row + 1)..3 {
                approx_eq(eval(ref_m.get(r, col).unwrap()), 0.0);
            }
        }
    }

    #[test]
    fn fast_ref_zero_matrix() {
        // All-zero 2×3 — no pivots.
        let m = MatrixExpr::zero(2, 3);
        let (ref_m, pivots) = m.ref_form().unwrap();
        assert!(pivots.is_empty(), "zero matrix has no pivots");
        // All entries still zero.
        let data = to_f64(&ref_m);
        for row in &data {
            for &v in row {
                approx_eq(v, 0.0);
            }
        }
    }

    #[test]
    fn fast_ref_identity() {
        let m = MatrixExpr::identity(3);
        let (ref_m, pivots) = m.ref_form().unwrap();
        assert_eq!(pivots, vec![0, 1, 2]);
        let data = to_f64(&ref_m);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                approx_eq(data[i][j], expected);
            }
        }
    }

    #[test]
    fn fast_rref_2x2() {
        // [[1, 2], [3, 4]] — RREF should be [[1, 0], [0, 1]]
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(2)],
            vec![Expr::int(3), Expr::int(4)],
        ])
        .unwrap();

        let (rref_m, pivots) = m.rref().unwrap();
        assert_eq!(pivots, vec![0, 1]);

        let data = to_f64(&rref_m);
        approx_eq(data[0][0], 1.0);
        approx_eq(data[0][1], 0.0);
        approx_eq(data[1][0], 0.0);
        approx_eq(data[1][1], 1.0);
    }

    #[test]
    fn fast_rref_3x3() {
        // [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]] with solution [8, -11, -3]
        // RREF of augmented not tested here — just verify RREF properties.
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(2), Expr::int(1), Expr::int(-1)],
            vec![Expr::int(-3), Expr::int(-1), Expr::int(2)],
            vec![Expr::int(-2), Expr::int(1), Expr::int(2)],
        ])
        .unwrap();

        let (rref_m, pivots) = m.rref().unwrap();
        assert_eq!(pivots.len(), 3);

        // Each pivot entry must equal 1.
        for (row, &col) in pivots.iter().enumerate() {
            approx_eq(eval(rref_m.get(row, col).unwrap()), 1.0);
        }

        // All entries in a pivot column other than the pivot row must be zero.
        for (row, &col) in pivots.iter().enumerate() {
            for r in 0..3 {
                if r != row {
                    approx_eq(eval(rref_m.get(r, col).unwrap()), 0.0);
                }
            }
        }
    }

    #[test]
    fn fast_rref_symbolic() {
        // Matrix with a symbolic entry: [[x, 1], [0, 2]]
        // REF/RREF should still work — the x stays symbolic.
        let x = Expr::symbol("x");
        let m = MatrixExpr::from_expr_elements(vec![
            vec![x.clone(), Expr::int(1)],
            vec![Expr::int(0), Expr::int(2)],
        ])
        .unwrap();

        let (_rref_m, pivots) = m.rref().unwrap();
        // Row 0 pivot is col 0 (the x column), row 1 pivot is col 1.
        assert_eq!(pivots, vec![0, 1]);
    }

    #[test]
    fn fast_pivot_count_rank_deficient() {
        // [[1, 2, 3], [4, 5, 6], [7, 8, 9]] — rank 2 (rows are linearly dep.)
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(2), Expr::int(3)],
            vec![Expr::int(4), Expr::int(5), Expr::int(6)],
            vec![Expr::int(7), Expr::int(8), Expr::int(9)],
        ])
        .unwrap();

        let (_rref_m, pivots) = m.rref().unwrap();
        assert_eq!(
            pivots.len(),
            2,
            "rank-deficient 3x3 should have exactly 2 pivots"
        );
    }

    #[test]
    fn fast_test_rank_identity() {
        assert_eq!(MatrixExpr::identity(3).rank().unwrap(), 3);
    }

    #[test]
    fn fast_test_rank_zero() {
        assert_eq!(MatrixExpr::zero(3, 3).rank().unwrap(), 0);
    }

    #[test]
    fn fast_test_rank_deficient() {
        // [[1,2,3],[4,5,6],[7,8,9]] — rank 2 (third row = sum of first two rows'
        // linear combination).
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(2), Expr::int(3)],
            vec![Expr::int(4), Expr::int(5), Expr::int(6)],
            vec![Expr::int(7), Expr::int(8), Expr::int(9)],
        ])
        .unwrap();
        assert_eq!(m.rank().unwrap(), 2);
    }

    #[test]
    fn fast_test_rank_full_row() {
        // 2×3 full row-rank matrix — rank must equal number of rows (2).
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(0), Expr::int(0)],
            vec![Expr::int(0), Expr::int(1), Expr::int(0)],
        ])
        .unwrap();
        assert_eq!(m.rank().unwrap(), 2);
    }

    #[test]
    fn fast_test_rank_1x1() {
        let nonzero = MatrixExpr::from_expr_elements(vec![vec![Expr::int(5)]]).unwrap();
        assert_eq!(nonzero.rank().unwrap(), 1);

        let zero = MatrixExpr::from_expr_elements(vec![vec![Expr::int(0)]]).unwrap();
        assert_eq!(zero.rank().unwrap(), 0);
    }
}
