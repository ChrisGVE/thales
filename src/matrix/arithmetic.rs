//! Transpose and other arithmetic operations on [`MatrixExpr`].

use std::sync::Arc;

use crate::numeric::expr::Expr;

use super::MatrixExpr;

impl MatrixExpr {
    /// Compute the transpose of this matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::numeric::expr::Expr;
    ///
    /// let m = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(1), Expr::int(2), Expr::int(3)],
    ///     vec![Expr::int(4), Expr::int(5), Expr::int(6)],
    /// ]).unwrap();
    ///
    /// let mt = m.transpose();
    /// assert_eq!(mt.rows(), 3);
    /// assert_eq!(mt.cols(), 2);
    /// ```
    pub fn transpose(&self) -> Self {
        let elements: Vec<Vec<Arc<Expr>>> = (0..self.cols)
            .map(|j| {
                (0..self.rows)
                    .map(|i| self.elements[i][j].clone())
                    .collect()
            })
            .collect();
        Self {
            rows: self.cols,
            cols: self.rows,
            elements,
        }
    }
}
