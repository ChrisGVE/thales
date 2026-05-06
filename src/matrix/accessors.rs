//! Element accessors for [`MatrixExpr`].

use std::sync::Arc;

use crate::numeric::expr::Expr;

use super::{MatrixError, MatrixExpr, MatrixResult};

impl MatrixExpr {
    /// Get a reference to an element at (row, col).
    ///
    /// # Errors
    ///
    /// Returns an error if indices are out of bounds.
    pub fn get(&self, row: usize, col: usize) -> MatrixResult<&Arc<Expr>> {
        if row >= self.rows || col >= self.cols {
            return Err(MatrixError::IndexOutOfBounds {
                row,
                col,
                rows: self.rows,
                cols: self.cols,
            });
        }
        Ok(&self.elements[row][col])
    }

    /// Set an element at (row, col).
    ///
    /// # Errors
    ///
    /// Returns an error if indices are out of bounds.
    pub fn set(&mut self, row: usize, col: usize, value: Arc<Expr>) -> MatrixResult<()> {
        if row >= self.rows || col >= self.cols {
            return Err(MatrixError::IndexOutOfBounds {
                row,
                col,
                rows: self.rows,
                cols: self.cols,
            });
        }
        self.elements[row][col] = value;
        Ok(())
    }

    /// Get a row as a slice of `Arc<Expr>`.
    pub fn row(&self, index: usize) -> MatrixResult<&Vec<Arc<Expr>>> {
        if index >= self.rows {
            return Err(MatrixError::IndexOutOfBounds {
                row: index,
                col: 0,
                rows: self.rows,
                cols: self.cols,
            });
        }
        Ok(&self.elements[index])
    }

    /// Get a column as a vector of `Arc<Expr>` references.
    pub fn col(&self, index: usize) -> MatrixResult<Vec<&Arc<Expr>>> {
        if index >= self.cols {
            return Err(MatrixError::IndexOutOfBounds {
                row: 0,
                col: index,
                rows: self.rows,
                cols: self.cols,
            });
        }
        Ok(self.elements.iter().map(|row| &row[index]).collect())
    }
}
