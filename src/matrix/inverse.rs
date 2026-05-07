//! Matrix inverse, adjugate, cofactor matrix, and singularity check.

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

    /// Compute the inverse of the matrix.
    ///
    /// Uses the formula: A^(-1) = adj(A) / det(A)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The matrix is not square
    /// - The matrix is singular (determinant is zero)
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

        let det = self.determinant()?;

        // Check if determinant is zero (symbolically or numerically)
        let is_zero = if det.is_zero() {
            true
        } else {
            // Try numerical evaluation for expressions that simplify to zero
            let empty: std::collections::HashMap<SymbolId, f64> = std::collections::HashMap::new();
            evaluate(&det, &empty).map_or(false, |v| v.abs() < 1e-10)
        };

        if is_zero {
            return Err(MatrixError::InvalidOperation(
                "Matrix is singular (determinant is zero)".to_string(),
            ));
        }

        // For 1x1 matrix
        if self.rows == 1 {
            let inv_element = normalize::div(Expr::int(1), self.elements[0][0].clone());
            return MatrixExpr::from_expr_elements(vec![vec![inv_element]]);
        }

        let adj = self.adjugate()?;

        // Multiply adjugate by 1/det
        let inv_det = normalize::div(Expr::int(1), det);

        Ok(adj.scalar_mul(&inv_det))
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
