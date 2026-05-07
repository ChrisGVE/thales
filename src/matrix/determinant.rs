//! Matrix determinant, minor, and cofactor computations.

use std::sync::Arc;

use crate::numeric::expr::Expr;
use crate::numeric::normalize;

use super::{MatrixError, MatrixExpr, MatrixResult};

impl MatrixExpr {
    /// Get the submatrix by removing row `row_idx` and column `col_idx`.
    ///
    /// This is used for computing minors and cofactors.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is 1x1 or smaller.
    pub fn submatrix(&self, row_idx: usize, col_idx: usize) -> MatrixResult<MatrixExpr> {
        if self.rows <= 1 || self.cols <= 1 {
            return Err(MatrixError::InvalidOperation(
                "Cannot compute submatrix of 1x1 or smaller matrix".to_string(),
            ));
        }

        let elements: Vec<Vec<Arc<Expr>>> = self
            .elements
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != row_idx)
            .map(|(_, row)| {
                row.iter()
                    .enumerate()
                    .filter(|(j, _)| *j != col_idx)
                    .map(|(_, elem)| elem.clone())
                    .collect()
            })
            .collect();

        MatrixExpr::from_expr_elements(elements)
    }

    /// Compute the minor M(i, j) — the determinant of the submatrix excluding row i and column j.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square or is 1x1.
    pub fn minor(&self, row: usize, col: usize) -> MatrixResult<Arc<Expr>> {
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "Minor requires a square matrix".to_string(),
            ));
        }
        let sub = self.submatrix(row, col)?;
        sub.determinant()
    }

    /// Compute the cofactor C(i, j) = (-1)^(i+j) * M(i, j).
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square or is 1x1.
    pub fn cofactor(&self, row: usize, col: usize) -> MatrixResult<Arc<Expr>> {
        let minor = self.minor(row, col)?;
        if (row + col) % 2 == 0 {
            Ok(minor)
        } else {
            Ok(normalize::neg(minor))
        }
    }

    /// Compute the determinant of the matrix.
    ///
    /// Uses the following algorithms:
    /// - 1x1: Returns the single element
    /// - 2x2: Uses ad - bc formula
    /// - 3x3, 4x4: Uses cofactor expansion (Laplace) along the first row
    /// - 5x5 and larger: Uses the Bareiss fraction-free algorithm
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
    /// use thales::numeric::evaluation::evaluate;
    /// use thales::numeric::SymbolId;
    /// use std::collections::HashMap;
    ///
    /// // 2x2 matrix: [[1, 2], [3, 4]]
    /// let m = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(1), Expr::int(2)],
    ///     vec![Expr::int(3), Expr::int(4)],
    /// ]).unwrap();
    ///
    /// let det = m.determinant().unwrap();
    /// // det = 1*4 - 2*3 = -2
    /// let empty: HashMap<SymbolId, f64> = HashMap::new();
    /// assert_eq!(evaluate(&det, &empty), Some(-2.0));
    /// ```
    pub fn determinant(&self) -> MatrixResult<Arc<Expr>> {
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "Determinant requires a square matrix".to_string(),
            ));
        }

        match self.rows {
            0 => Ok(Expr::int(1)),
            1..=4 => self.determinant_laplace(),
            _ => self.determinant_bareiss(),
        }
    }

    /// Compute the determinant using cofactor (Laplace) expansion along the first row.
    ///
    /// Used for matrices up to 4x4 where the recursion depth is bounded.
    pub(crate) fn determinant_laplace(&self) -> MatrixResult<Arc<Expr>> {
        match self.rows {
            1 => Ok(self.elements[0][0].clone()),
            2 => {
                let a = &self.elements[0][0];
                let b = &self.elements[0][1];
                let c = &self.elements[1][0];
                let d = &self.elements[1][1];

                let ad = normalize::mul(a.clone(), d.clone());
                let bc = normalize::mul(b.clone(), c.clone());
                Ok(normalize::sub(ad, bc))
            }
            _ => {
                // Cofactor expansion along first row
                let mut det = Expr::int(0);
                for j in 0..self.cols {
                    let cofactor = self.cofactor(0, j)?;
                    let term = normalize::mul(self.elements[0][j].clone(), cofactor);
                    det = normalize::add(det, term);
                }
                Ok(det)
            }
        }
    }

    /// Compute the determinant using the Bareiss fraction-free algorithm.
    ///
    /// This algorithm avoids intermediate fractions by maintaining the invariant
    /// that all divisions performed at step k are exact (divisible by the previous
    /// pivot). This makes it suitable for integer and rational matrices where
    /// cofactor expansion would produce exponentially large intermediate values.
    ///
    /// The recurrence at step k is:
    ///   `a[i][j] = (a[i][j] * a[k][k] - a[i][k] * a[k][j]) / prev_pivot`
    ///
    /// More efficient than Laplace expansion for n > 4.
    pub(crate) fn determinant_bareiss(&self) -> MatrixResult<Arc<Expr>> {
        let n = self.rows;

        // Clone all elements into a mutable working matrix.
        let mut mat: Vec<Vec<Arc<Expr>>> = self.elements.iter().map(|row| row.clone()).collect();

        let mut sign = Expr::int(1);
        let mut prev_pivot = Expr::int(1);

        for k in 0..n {
            // Find a non-zero pivot in column k at or below row k.
            let pivot_row = (k..n).find(|&i| !mat[i][k].is_zero());

            let pivot_row = match pivot_row {
                Some(r) => r,
                None => {
                    // Column is entirely zero — determinant is 0.
                    return Ok(Expr::int(0));
                }
            };

            // Swap rows if the pivot is not already on the diagonal.
            if pivot_row != k {
                mat.swap(k, pivot_row);
                sign = normalize::neg(sign);
            }

            // Apply the Bareiss update to all rows below the pivot row.
            let pivot = mat[k][k].clone();
            for i in (k + 1)..n {
                for j in (k + 1)..n {
                    // new_val = (mat[i][j] * pivot - mat[i][k] * mat[k][j]) / prev_pivot
                    let term1 = normalize::mul(mat[i][j].clone(), pivot.clone());
                    let term2 = normalize::mul(mat[i][k].clone(), mat[k][j].clone());
                    let numerator = normalize::sub(term1, term2);
                    mat[i][j] = normalize::div(numerator, prev_pivot.clone());
                }
            }

            prev_pivot = pivot;
        }

        // The determinant is the bottom-right element, adjusted for row-swap sign.
        let raw = mat[n - 1][n - 1].clone();
        Ok(normalize::mul(sign, raw))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::evaluation::evaluate;
    use crate::numeric::SymbolId;
    use std::collections::HashMap;

    fn int(n: i64) -> Arc<Expr> {
        Expr::int(n)
    }

    fn eval(e: &Arc<Expr>) -> Option<f64> {
        let empty: HashMap<SymbolId, f64> = HashMap::new();
        evaluate(e, &empty)
    }

    fn mat(rows: Vec<Vec<i64>>) -> MatrixExpr {
        let elems = rows
            .into_iter()
            .map(|row| row.into_iter().map(int).collect())
            .collect();
        MatrixExpr::from_expr_elements(elems).unwrap()
    }

    #[test]
    fn fast_test_bareiss_matches_laplace_3x3() {
        // [[1, 2, 3], [4, 5, 6], [7, 2, 9]]  det = -36
        let m = mat(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 2, 9]]);
        let laplace = m.determinant_laplace().unwrap();
        let bareiss = m.determinant_bareiss().unwrap();
        assert_eq!(eval(&laplace), eval(&bareiss));
        assert_eq!(eval(&laplace), Some(-36.0));
    }

    #[test]
    fn fast_test_bareiss_matches_laplace_4x4() {
        // Hilbert-style integer 4x4: determinant should agree between methods.
        let m = mat(vec![
            vec![2, 1, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![1, 0, 2, 3],
        ]);
        let laplace = m.determinant_laplace().unwrap();
        let bareiss = m.determinant_bareiss().unwrap();
        assert_eq!(eval(&laplace), eval(&bareiss));
    }

    #[test]
    fn fast_test_bareiss_5x5() {
        // Diagonal matrix with entries 1,2,3,4,5 — det = 120.
        let mut rows = vec![vec![0i64; 5]; 5];
        for i in 0..5 {
            rows[i][i] = (i as i64) + 1;
        }
        let m = mat(rows);
        let det = m.determinant().unwrap();
        assert_eq!(eval(&det), Some(120.0));
    }

    #[test]
    fn fast_test_bareiss_6x6() {
        // Lower-triangular 6x6 with diagonal [1,2,3,4,5,6] — det = 720.
        let mut rows = vec![vec![0i64; 6]; 6];
        for i in 0..6 {
            for j in 0..=i {
                rows[i][j] = if i == j { (i as i64) + 1 } else { 1 };
            }
        }
        let m = mat(rows);
        let det = m.determinant().unwrap();
        assert_eq!(eval(&det), Some(720.0));
    }

    #[test]
    fn fast_test_bareiss_singular() {
        // Row 2 is twice row 1 — determinant must be 0.
        let m = mat(vec![
            vec![1, 2, 3, 4, 5],
            vec![2, 4, 6, 8, 10],
            vec![3, 5, 7, 9, 11],
            vec![0, 1, 2, 3, 4],
            vec![1, 0, 0, 1, 0],
        ]);
        let det = m.determinant().unwrap();
        assert_eq!(eval(&det), Some(0.0));
    }

    #[test]
    fn fast_test_bareiss_with_row_swap() {
        // Leading element in the first column is 0, forcing a row swap.
        // [[0,1,2,3,4],[1,0,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
        let m = mat(vec![
            vec![0, 1, 2, 3, 4],
            vec![1, 0, 0, 0, 0],
            vec![0, 0, 1, 0, 0],
            vec![0, 0, 0, 1, 0],
            vec![0, 0, 0, 0, 1],
        ]);
        // After swapping rows 0 and 1 and expanding, det = -1 * (1*det of remaining 4x4)
        // det of sub = det([[1,2,3,4],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) = 1
        // with sign flip = -1
        let det = m.determinant().unwrap();
        assert_eq!(eval(&det), Some(-1.0));
    }
}
