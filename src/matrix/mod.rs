//! Matrix expression type with basic linear algebra operations.
//!
//! This module provides a symbolic matrix type where elements are `Arc<Expr>`
//! (the canonical internal representation), supporting operations like
//! addition, multiplication, transpose, and trace with symbolic manipulation
//! capabilities.
//!
//! # Examples
//!
//! ```
//! use thales::matrix::MatrixExpr;
//! use thales::numeric::expr::Expr;
//!
//! // Create a 2x2 identity matrix
//! let identity = MatrixExpr::identity(2);
//!
//! // Create a matrix from Arc<Expr> elements
//! let a = Expr::int(1);
//! let b = Expr::int(2);
//! let c = Expr::int(3);
//! let d = Expr::int(4);
//! let m = MatrixExpr::from_expr_elements(vec![
//!     vec![a, b],
//!     vec![c, d],
//! ]).unwrap();
//!
//! // Transpose
//! let mt = m.transpose();
//! ```

mod accessors;
mod arithmetic;
mod cholesky;
mod constructors;
mod determinant;
mod display;
mod eigen;
mod inverse;
mod lu;
mod operations;
mod types;

pub use types::{BracketStyle, MatrixError, MatrixResult};

use std::sync::Arc;

use crate::numeric::expr::Expr;

/// A matrix of symbolic expressions.
///
/// Each element is an `Arc<Expr>` — the canonical internal CAS representation
/// (Architecture Rule 1). Supports standard matrix operations including
/// addition, multiplication, transpose, and trace.
///
/// # Examples
///
/// ```
/// use thales::matrix::MatrixExpr;
/// use thales::numeric::expr::Expr;
/// use thales::numeric::SymbolId;
/// use std::sync::Arc;
///
/// // Create a 2x2 matrix with symbolic entries
/// let x = Expr::symbol("x");
/// let one = Expr::int(1);
/// let two = Expr::int(2);
/// let three = Expr::int(3);
///
/// let m = MatrixExpr::from_expr_elements(vec![
///     vec![x, one],
///     vec![two, three],
/// ]).unwrap();
///
/// assert_eq!(m.rows(), 2);
/// assert_eq!(m.cols(), 2);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixExpr {
    rows: usize,
    cols: usize,
    elements: Vec<Vec<Arc<Expr>>>,
}

impl MatrixExpr {
    /// Get the number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get the dimensions as (rows, cols).
    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Check if the matrix is square.
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Get a reference to the elements grid.
    pub(crate) fn elements(&self) -> &Vec<Vec<Arc<Expr>>> {
        &self.elements
    }
}

#[cfg(test)]
mod tests;
