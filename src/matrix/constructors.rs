//! Constructors for [`MatrixExpr`].

use std::sync::Arc;

use crate::numeric::expr::Expr;

use super::{MatrixError, MatrixExpr, MatrixResult};

impl MatrixExpr {
    /// Create a matrix from a 2D vector of `Arc<Expr>` elements.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The input is empty
    /// - Any row is empty
    /// - Rows have different lengths (non-rectangular)
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
    /// ```
    pub fn from_expr_elements(elements: Vec<Vec<Arc<Expr>>>) -> MatrixResult<Self> {
        if elements.is_empty() || elements[0].is_empty() {
            return Err(MatrixError::EmptyMatrix);
        }

        let cols = elements[0].len();
        for row in &elements {
            if row.len() != cols {
                return Err(MatrixError::NonRectangular);
            }
        }

        let rows = elements.len();
        Ok(Self {
            rows,
            cols,
            elements,
        })
    }

    /// Create a matrix from pre-validated `Arc<Expr>` elements (internal use).
    pub(crate) fn from_expr_elements_unchecked(
        rows: usize,
        cols: usize,
        elements: Vec<Vec<Arc<Expr>>>,
    ) -> Self {
        Self {
            rows,
            cols,
            elements,
        }
    }

    /// Create an identity matrix of size n x n.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    ///
    /// let i3 = MatrixExpr::identity(3);
    /// assert_eq!(i3.rows(), 3);
    /// assert_eq!(i3.cols(), 3);
    /// ```
    pub fn identity(n: usize) -> Self {
        let elements: Vec<Vec<Arc<Expr>>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { Expr::int(1) } else { Expr::int(0) })
                    .collect()
            })
            .collect();
        Self {
            rows: n,
            cols: n,
            elements,
        }
    }

    /// Create a zero matrix of size rows x cols.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    ///
    /// let z = MatrixExpr::zero(2, 3);
    /// assert_eq!(z.rows(), 2);
    /// assert_eq!(z.cols(), 3);
    /// ```
    pub fn zero(rows: usize, cols: usize) -> Self {
        let elements: Vec<Vec<Arc<Expr>>> = (0..rows)
            .map(|_| (0..cols).map(|_| Expr::int(0)).collect())
            .collect();
        Self {
            rows,
            cols,
            elements,
        }
    }

    /// Create a diagonal matrix from a vector of `Arc<Expr>` values.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::numeric::expr::Expr;
    ///
    /// let diag = MatrixExpr::diagonal(vec![
    ///     Expr::int(1),
    ///     Expr::int(2),
    ///     Expr::int(3),
    /// ]);
    /// assert_eq!(diag.rows(), 3);
    /// assert_eq!(diag.cols(), 3);
    /// ```
    pub fn diagonal(diag: Vec<Arc<Expr>>) -> Self {
        let n = diag.len();
        let elements: Vec<Vec<Arc<Expr>>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if i == j {
                            diag[i].clone()
                        } else {
                            Expr::int(0)
                        }
                    })
                    .collect()
            })
            .collect();
        Self {
            rows: n,
            cols: n,
            elements,
        }
    }
}
