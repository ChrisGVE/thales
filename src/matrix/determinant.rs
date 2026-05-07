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
    /// - NxN: Uses cofactor expansion along the first row
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
            1 => Ok(self.elements[0][0].clone()),
            2 => {
                // det = a*d - b*c for [[a, b], [c, d]]
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
}
