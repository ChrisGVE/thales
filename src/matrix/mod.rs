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

mod display;
mod eigen;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::evaluation::evaluate;
    use crate::numeric::SymbolId;
    use std::collections::HashMap;

    fn int(n: i64) -> Arc<Expr> {
        Expr::int(n)
    }

    fn var(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }

    fn eval(e: &Arc<Expr>) -> Option<f64> {
        let empty: HashMap<SymbolId, f64> = HashMap::new();
        evaluate(e, &empty)
    }

    fn eval_with(e: &Arc<Expr>, vars: &HashMap<SymbolId, f64>) -> Option<f64> {
        evaluate(e, vars)
    }

    #[test]
    fn test_matrix_creation() {
        let m = MatrixExpr::from_expr_elements(vec![vec![int(1), int(2)], vec![int(3), int(4)]])
            .unwrap();

        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
        assert!(m.is_square());
    }

    #[test]
    fn test_identity_matrix() {
        let i3 = MatrixExpr::identity(3);
        assert_eq!(i3.rows(), 3);
        assert_eq!(i3.cols(), 3);

        // Check diagonal elements are 1
        assert_eq!(eval(i3.get(0, 0).unwrap()), Some(1.0));
        assert_eq!(eval(i3.get(1, 1).unwrap()), Some(1.0));
        assert_eq!(eval(i3.get(2, 2).unwrap()), Some(1.0));

        // Check off-diagonal elements are 0
        assert_eq!(eval(i3.get(0, 1).unwrap()), Some(0.0));
        assert_eq!(eval(i3.get(1, 2).unwrap()), Some(0.0));
    }

    #[test]
    fn test_zero_matrix() {
        let z = MatrixExpr::zero(2, 3);
        assert_eq!(z.rows(), 2);
        assert_eq!(z.cols(), 3);

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(eval(z.get(i, j).unwrap()), Some(0.0));
            }
        }
    }

    #[test]
    fn test_diagonal_matrix() {
        let d = MatrixExpr::diagonal(vec![int(1), int(2), int(3)]);
        assert_eq!(d.rows(), 3);
        assert_eq!(d.cols(), 3);

        assert_eq!(eval(d.get(0, 0).unwrap()), Some(1.0));
        assert_eq!(eval(d.get(1, 1).unwrap()), Some(2.0));
        assert_eq!(eval(d.get(2, 2).unwrap()), Some(3.0));
        assert_eq!(eval(d.get(0, 1).unwrap()), Some(0.0));
    }

    #[test]
    fn test_transpose() {
        let m = MatrixExpr::from_expr_elements(vec![
            vec![int(1), int(2), int(3)],
            vec![int(4), int(5), int(6)],
        ])
        .unwrap();

        let mt = m.transpose();
        assert_eq!(mt.rows(), 3);
        assert_eq!(mt.cols(), 2);

        assert_eq!(eval(mt.get(0, 0).unwrap()), Some(1.0));
        assert_eq!(eval(mt.get(0, 1).unwrap()), Some(4.0));
        assert_eq!(eval(mt.get(1, 0).unwrap()), Some(2.0));
        assert_eq!(eval(mt.get(2, 1).unwrap()), Some(6.0));
    }

    #[test]
    fn test_double_transpose() {
        let m = MatrixExpr::from_expr_elements(vec![vec![int(1), int(2)], vec![int(3), int(4)]])
            .unwrap();

        let mtt = m.transpose().transpose();
        assert_eq!(mtt.elements, m.elements);
    }

    #[test]
    fn test_trace() {
        let m = MatrixExpr::from_expr_elements(vec![vec![int(1), int(2)], vec![int(3), int(4)]])
            .unwrap();

        let trace = m.trace().unwrap();
        assert_eq!(eval(&trace), Some(5.0));
    }

    #[test]
    fn test_addition() {
        let a = MatrixExpr::from_expr_elements(vec![vec![int(1), int(2)], vec![int(3), int(4)]])
            .unwrap();

        let b = MatrixExpr::from_expr_elements(vec![vec![int(5), int(6)], vec![int(7), int(8)]])
            .unwrap();

        let sum = a.add(&b).unwrap();

        assert_eq!(eval(sum.get(0, 0).unwrap()), Some(6.0));
        assert_eq!(eval(sum.get(0, 1).unwrap()), Some(8.0));
        assert_eq!(eval(sum.get(1, 0).unwrap()), Some(10.0));
        assert_eq!(eval(sum.get(1, 1).unwrap()), Some(12.0));
    }

    #[test]
    fn test_addition_dimension_check() {
        let a = MatrixExpr::from_expr_elements(vec![vec![int(1), int(2)]]).unwrap();
        let b = MatrixExpr::from_expr_elements(vec![vec![int(1)], vec![int(2)]]).unwrap();
        let result = a.add(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_matrix_multiplication() {
        // 2x3 * 3x2 = 2x2
        let a = MatrixExpr::from_expr_elements(vec![
            vec![int(1), int(2), int(3)],
            vec![int(4), int(5), int(6)],
        ])
        .unwrap();

        let b = MatrixExpr::from_expr_elements(vec![
            vec![int(7), int(8)],
            vec![int(9), int(10)],
            vec![int(11), int(12)],
        ])
        .unwrap();

        let c = a.mul(&b).unwrap();
        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 2);

        // C[0][0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        assert_eq!(eval(c.get(0, 0).unwrap()), Some(58.0));
        // C[0][1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
        assert_eq!(eval(c.get(0, 1).unwrap()), Some(64.0));
        // C[1][0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
        assert_eq!(eval(c.get(1, 0).unwrap()), Some(139.0));
        // C[1][1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
        assert_eq!(eval(c.get(1, 1).unwrap()), Some(154.0));
    }

    #[test]
    fn test_scalar_multiplication() {
        let m = MatrixExpr::identity(2);
        let scaled = m.scalar_mul(&int(3));

        assert_eq!(eval(scaled.get(0, 0).unwrap()), Some(3.0));
        assert_eq!(eval(scaled.get(1, 1).unwrap()), Some(3.0));
        assert_eq!(eval(scaled.get(0, 1).unwrap()), Some(0.0));
    }

    #[test]
    fn test_symbolic_matrix() {
        let m = MatrixExpr::from_expr_elements(vec![
            vec![var("mat_a"), var("mat_b")],
            vec![var("mat_c"), var("mat_d")],
        ])
        .unwrap();

        let mut vars: HashMap<SymbolId, f64> = HashMap::new();
        vars.insert(SymbolId::intern("mat_a"), 1.0);
        vars.insert(SymbolId::intern("mat_b"), 2.0);
        vars.insert(SymbolId::intern("mat_c"), 3.0);
        vars.insert(SymbolId::intern("mat_d"), 4.0);

        let result = m.evaluate(&vars).unwrap();
        assert_eq!(result[0][0], 1.0);
        assert_eq!(result[0][1], 2.0);
        assert_eq!(result[1][0], 3.0);
        assert_eq!(result[1][1], 4.0);
    }

    #[test]
    fn test_latex_output() {
        let m = MatrixExpr::from_expr_elements(vec![vec![int(1), int(2)], vec![int(3), int(4)]])
            .unwrap();

        let latex = m.to_latex(BracketStyle::Parentheses);
        assert!(latex.contains("\\begin{pmatrix}"));
        assert!(latex.contains("\\end{pmatrix}"));
        assert!(latex.contains("1 & 2"));
        assert!(latex.contains("3 & 4"));
    }

    #[test]
    fn test_transpose_multiplication_property() {
        // (AB)^T = B^T A^T
        let a = MatrixExpr::from_expr_elements(vec![vec![int(1), int(2)], vec![int(3), int(4)]])
            .unwrap();

        let b = MatrixExpr::from_expr_elements(vec![vec![int(5), int(6)], vec![int(7), int(8)]])
            .unwrap();

        let ab = a.mul(&b).unwrap();
        let ab_t = ab.transpose();

        let bt_at = b.transpose().mul(&a.transpose()).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(
                    eval(ab_t.get(i, j).unwrap()),
                    eval(bt_at.get(i, j).unwrap())
                );
            }
        }
    }

    #[test]
    fn test_determinant_2x2() {
        // det([[1, 2], [3, 4]]) = 1*4 - 2*3 = -2
        let m = MatrixExpr::from_expr_elements(vec![vec![int(1), int(2)], vec![int(3), int(4)]])
            .unwrap();

        let det = m.determinant().unwrap();
        assert_eq!(eval(&det), Some(-2.0));
    }

    #[test]
    fn test_determinant_3x3() {
        // det([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) = 0 (rows are linearly dependent)
        let m = MatrixExpr::from_expr_elements(vec![
            vec![int(1), int(2), int(3)],
            vec![int(4), int(5), int(6)],
            vec![int(7), int(8), int(9)],
        ])
        .unwrap();

        let det = m.determinant().unwrap();
        assert_eq!(eval(&det), Some(0.0));
    }

    #[test]
    fn test_determinant_3x3_nonzero() {
        // det([[1, 2, 3], [0, 1, 4], [5, 6, 0]]) = 1
        let m = MatrixExpr::from_expr_elements(vec![
            vec![int(1), int(2), int(3)],
            vec![int(0), int(1), int(4)],
            vec![int(5), int(6), int(0)],
        ])
        .unwrap();

        let det = m.determinant().unwrap();
        assert_eq!(eval(&det), Some(1.0));
    }

    #[test]
    fn test_determinant_identity() {
        // det(I) = 1
        let i3 = MatrixExpr::identity(3);
        let det = i3.determinant().unwrap();
        assert_eq!(eval(&det), Some(1.0));
    }

    #[test]
    fn test_determinant_non_square() {
        let m = MatrixExpr::from_expr_elements(vec![
            vec![int(1), int(2), int(3)],
            vec![int(4), int(5), int(6)],
        ])
        .unwrap();

        let result = m.determinant();
        assert!(result.is_err());
    }

    #[test]
    fn test_inverse_2x2() {
        // A = [[4, 7], [2, 6]], det(A) = 24 - 14 = 10
        // A^(-1) = (1/10) * [[6, -7], [-2, 4]] = [[0.6, -0.7], [-0.2, 0.4]]
        let m = MatrixExpr::from_expr_elements(vec![vec![int(4), int(7)], vec![int(2), int(6)]])
            .unwrap();

        let inv = m.inverse().unwrap();

        // Verify A * A^(-1) = I
        let product = m.mul(&inv).unwrap();
        let empty: HashMap<SymbolId, f64> = HashMap::new();
        let result = product.evaluate(&empty).unwrap();

        assert!((result[0][0] - 1.0).abs() < 1e-10);
        assert!((result[0][1] - 0.0).abs() < 1e-10);
        assert!((result[1][0] - 0.0).abs() < 1e-10);
        assert!((result[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_3x3() {
        // A = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
        let m = MatrixExpr::from_expr_elements(vec![
            vec![int(1), int(2), int(3)],
            vec![int(0), int(1), int(4)],
            vec![int(5), int(6), int(0)],
        ])
        .unwrap();

        let inv = m.inverse().unwrap();
        let empty: HashMap<SymbolId, f64> = HashMap::new();

        // Verify A * A^(-1) = I
        let product = m.mul(&inv).unwrap();
        let result = product.evaluate(&empty).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (result[i][j] - expected).abs() < 1e-10,
                    "Expected {} at ({}, {}), got {}",
                    expected,
                    i,
                    j,
                    result[i][j]
                );
            }
        }
    }

    #[test]
    fn test_inverse_singular_matrix() {
        // Singular matrix (det = 0)
        let m = MatrixExpr::from_expr_elements(vec![vec![int(1), int(2)], vec![int(2), int(4)]])
            .unwrap();

        let result = m.inverse();
        assert!(result.is_err());
    }

    #[test]
    fn test_determinant_symbolic() {
        // det([[a, b], [c, d]]) = ad - bc
        let m = MatrixExpr::from_expr_elements(vec![
            vec![var("det_a"), var("det_b")],
            vec![var("det_c"), var("det_d")],
        ])
        .unwrap();

        let det = m.determinant().unwrap();

        let mut vars: HashMap<SymbolId, f64> = HashMap::new();
        vars.insert(SymbolId::intern("det_a"), 2.0);
        vars.insert(SymbolId::intern("det_b"), 3.0);
        vars.insert(SymbolId::intern("det_c"), 4.0);
        vars.insert(SymbolId::intern("det_d"), 5.0);

        // det = 2*5 - 3*4 = 10 - 12 = -2
        let result = eval_with(&det, &vars).unwrap();
        assert!((result - (-2.0)).abs() < 1e-10, "got {}", result);
    }

    #[test]
    fn test_submatrix() {
        let m = MatrixExpr::from_expr_elements(vec![
            vec![int(1), int(2), int(3)],
            vec![int(4), int(5), int(6)],
            vec![int(7), int(8), int(9)],
        ])
        .unwrap();

        // Remove row 1, col 1 -> [[1, 3], [7, 9]]
        let sub = m.submatrix(1, 1).unwrap();

        assert_eq!(sub.rows(), 2);
        assert_eq!(sub.cols(), 2);
        assert_eq!(eval(sub.get(0, 0).unwrap()), Some(1.0));
        assert_eq!(eval(sub.get(0, 1).unwrap()), Some(3.0));
        assert_eq!(eval(sub.get(1, 0).unwrap()), Some(7.0));
        assert_eq!(eval(sub.get(1, 1).unwrap()), Some(9.0));
    }

    #[test]
    fn test_adjugate_2x2() {
        // adj([[a, b], [c, d]]) = [[d, -b], [-c, a]]
        let m = MatrixExpr::from_expr_elements(vec![vec![int(1), int(2)], vec![int(3), int(4)]])
            .unwrap();

        let adj = m.adjugate().unwrap();

        assert_eq!(eval(adj.get(0, 0).unwrap()), Some(4.0));
        assert_eq!(eval(adj.get(0, 1).unwrap()), Some(-2.0));
        assert_eq!(eval(adj.get(1, 0).unwrap()), Some(-3.0));
        assert_eq!(eval(adj.get(1, 1).unwrap()), Some(1.0));
    }

    #[test]
    fn test_is_singular() {
        let singular =
            MatrixExpr::from_expr_elements(vec![vec![int(1), int(2)], vec![int(2), int(4)]])
                .unwrap();

        let non_singular =
            MatrixExpr::from_expr_elements(vec![vec![int(1), int(2)], vec![int(3), int(4)]])
                .unwrap();

        let empty: HashMap<SymbolId, f64> = HashMap::new();
        assert_eq!(singular.is_singular(&empty), Some(true));
        assert_eq!(non_singular.is_singular(&empty), Some(false));
    }

    #[test]
    fn test_inverse_identity() {
        // I^(-1) = I
        let i3 = MatrixExpr::identity(3);
        let inv = i3.inverse().unwrap();
        let empty: HashMap<SymbolId, f64> = HashMap::new();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let got = evaluate(inv.get(i, j).unwrap(), &empty);
                assert_eq!(got, Some(expected));
            }
        }
    }

    // =========================================================================
    // Eigenvalue and Eigenvector Tests
    // =========================================================================

    #[test]
    fn test_characteristic_polynomial_2x2() {
        // A = [[2, 1], [1, 2]], eigenvalues are 1 and 3
        // char poly = (λ - 1)(λ - 3) = λ² - 4λ + 3
        let m = MatrixExpr::from_expr_elements(vec![vec![int(2), int(1)], vec![int(1), int(2)]])
            .unwrap();

        let char_poly = m.characteristic_polynomial("lambda").unwrap();

        // Evaluate at λ = 1 (should be 0)
        let mut vars: HashMap<SymbolId, f64> = HashMap::new();
        vars.insert(SymbolId::intern("lambda"), 1.0);
        let at_1 = evaluate(&char_poly, &vars).unwrap();
        assert!(
            at_1.abs() < 1e-10,
            "char poly at λ=1 should be 0, got {}",
            at_1
        );

        // Evaluate at λ = 3 (should be 0)
        vars.insert(SymbolId::intern("lambda"), 3.0);
        let at_3 = evaluate(&char_poly, &vars).unwrap();
        assert!(
            at_3.abs() < 1e-10,
            "char poly at λ=3 should be 0, got {}",
            at_3
        );
    }

    #[test]
    fn test_eigenvalues_2x2_symmetric() {
        // A = [[2, 1], [1, 2]], eigenvalues are 1 and 3
        let m = MatrixExpr::from_expr_elements(vec![vec![int(2), int(1)], vec![int(1), int(2)]])
            .unwrap();

        let eigenvalues = m.eigenvalues_numeric().unwrap();
        assert_eq!(eigenvalues.len(), 2);

        // Sort eigenvalues for consistent comparison (by real part)
        let mut sorted = eigenvalues.clone();
        sorted.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());

        assert!(
            (sorted[0].re - 1.0).abs() < 1e-10,
            "Expected 1, got {}",
            sorted[0]
        );
        assert!(
            (sorted[1].re - 3.0).abs() < 1e-10,
            "Expected 3, got {}",
            sorted[1]
        );
    }

    #[test]
    fn test_eigenvalues_diagonal() {
        // Diagonal matrix: eigenvalues are the diagonal elements
        let m = MatrixExpr::from_expr_elements(vec![vec![int(5), int(0)], vec![int(0), int(3)]])
            .unwrap();

        let eigenvalues = m.eigenvalues_numeric().unwrap();
        let mut sorted = eigenvalues.clone();
        sorted.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());

        assert!((sorted[0].re - 3.0).abs() < 1e-10);
        assert!((sorted[1].re - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_eigenvalues_identity() {
        // Identity matrix: all eigenvalues are 1
        let m = MatrixExpr::identity(3);

        let eigenvalues = m.eigenvalues_numeric().unwrap();
        assert_eq!(eigenvalues.len(), 3);

        for ev in eigenvalues {
            assert!((ev.re - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_eigenvector_2x2() {
        // A = [[2, 1], [1, 2]], eigenvalue 3 has eigenvector [1, 1]
        let m = MatrixExpr::from_expr_elements(vec![vec![int(2), int(1)], vec![int(1), int(2)]])
            .unwrap();

        let eigenvector = m.eigenvector_numeric(3.0).unwrap();
        assert_eq!(eigenvector.len(), 2);

        // Check Av = λv (up to normalization)
        // v should be proportional to [1, 1]
        let ratio = eigenvector[0] / eigenvector[1];
        assert!(
            (ratio - 1.0).abs() < 1e-5,
            "Expected ratio 1, got {}",
            ratio
        );
    }

    #[test]
    fn test_eigenpairs() {
        let m = MatrixExpr::from_expr_elements(vec![vec![int(2), int(1)], vec![int(1), int(2)]])
            .unwrap();

        let pairs = m.eigenpairs_numeric().unwrap();
        assert_eq!(pairs.len(), 2);

        let empty: HashMap<SymbolId, f64> = HashMap::new();
        for (eigenvalue, eigenvector) in pairs {
            // Verify Av = λv
            let a = m.evaluate(&empty).unwrap();

            // Compute Av
            let av: Vec<f64> = (0..2)
                .map(|i| {
                    a[i].iter()
                        .zip(eigenvector.iter())
                        .map(|(a, v)| a * v)
                        .sum()
                })
                .collect();

            // Compute λv (use real part of eigenvalue — these are real-eigenvalue test cases)
            let lambda_v: Vec<f64> = eigenvector.iter().map(|v| eigenvalue.re * v).collect();

            // Check Av ≈ λv
            for i in 0..2 {
                assert!(
                    (av[i] - lambda_v[i]).abs() < 1e-5,
                    "Av[{}] = {}, λv[{}] = {}, eigenvalue = {}",
                    i,
                    av[i],
                    i,
                    lambda_v[i],
                    eigenvalue
                );
            }
        }
    }

    #[test]
    fn test_eigenvalues_3x3() {
        // A simple 3x3 matrix with known eigenvalues
        // A = [[1, 0, 0], [0, 2, 0], [0, 0, 3]] has eigenvalues 1, 2, 3
        let m = MatrixExpr::from_expr_elements(vec![
            vec![int(1), int(0), int(0)],
            vec![int(0), int(2), int(0)],
            vec![int(0), int(0), int(3)],
        ])
        .unwrap();

        let eigenvalues = m.eigenvalues_numeric().unwrap();
        let mut sorted = eigenvalues.clone();
        sorted.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());

        assert!((sorted[0].re - 1.0).abs() < 1e-10);
        assert!((sorted[1].re - 2.0).abs() < 1e-10);
        assert!((sorted[2].re - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_diagonalizable_symmetric() {
        // Symmetric matrices are always diagonalizable
        let m = MatrixExpr::from_expr_elements(vec![vec![int(2), int(1)], vec![int(1), int(2)]])
            .unwrap();

        assert!(m.is_diagonalizable().unwrap());
    }

    #[test]
    fn test_is_diagonalizable_identity() {
        let m = MatrixExpr::identity(3);
        assert!(m.is_diagonalizable().unwrap());
    }

    #[test]
    fn test_eigenvalues_non_square() {
        let m = MatrixExpr::from_expr_elements(vec![
            vec![int(1), int(2), int(3)],
            vec![int(4), int(5), int(6)],
        ])
        .unwrap();

        let result = m.eigenvalues_numeric();
        assert!(result.is_err());
    }

    #[test]
    fn test_characteristic_polynomial_non_square() {
        let m = MatrixExpr::from_expr_elements(vec![
            vec![int(1), int(2), int(3)],
            vec![int(4), int(5), int(6)],
        ])
        .unwrap();

        let result = m.characteristic_polynomial("lambda");
        assert!(result.is_err());
    }

    // =========================================================================
    // LU decomposition Tests
    // =========================================================================

    /// Build a permutation matrix P from a perm vector so we can verify P·A = L·U.
    fn perm_matrix(perm: &[usize]) -> MatrixExpr {
        let n = perm.len();
        let mut elements = vec![vec![Expr::int(0); n]; n];
        for (i, &src) in perm.iter().enumerate() {
            elements[i][src] = Expr::int(1);
        }
        MatrixExpr::from_expr_elements(elements).unwrap()
    }

    #[test]
    fn test_lu_decompose_2x2() {
        // A = [[4, 3], [6, 3]]
        let a = MatrixExpr::from_expr_elements(vec![vec![int(4), int(3)], vec![int(6), int(3)]])
            .unwrap();

        let (l, u, perm) = a.lu_decompose().unwrap();
        assert_eq!(l.rows(), 2);
        assert_eq!(u.rows(), 2);
        assert_eq!(perm.len(), 2);

        // Verify L·U = P·A
        let pa = perm_matrix(&perm).mul(&a).unwrap();
        let lu_prod = l.mul(&u).unwrap();
        let empty: HashMap<SymbolId, f64> = HashMap::new();
        for i in 0..2 {
            for j in 0..2 {
                let lu_val = evaluate(lu_prod.get(i, j).unwrap(), &empty).unwrap();
                let pa_val = evaluate(pa.get(i, j).unwrap(), &empty).unwrap();
                assert!(
                    (lu_val - pa_val).abs() < 1e-10,
                    "LU[{i}][{j}] = {lu_val}, PA[{i}][{j}] = {pa_val}"
                );
            }
        }
    }

    #[test]
    fn test_lu_decompose_3x3() {
        // A = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
        let a = MatrixExpr::from_expr_elements(vec![
            vec![int(1), int(2), int(3)],
            vec![int(0), int(1), int(4)],
            vec![int(5), int(6), int(0)],
        ])
        .unwrap();

        let (l, u, perm) = a.lu_decompose().unwrap();

        // Verify L·U = P·A
        let pa = perm_matrix(&perm).mul(&a).unwrap();
        let lu_prod = l.mul(&u).unwrap();
        let empty: HashMap<SymbolId, f64> = HashMap::new();
        for i in 0..3 {
            for j in 0..3 {
                let lu_val = evaluate(lu_prod.get(i, j).unwrap(), &empty).unwrap();
                let pa_val = evaluate(pa.get(i, j).unwrap(), &empty).unwrap();
                assert!(
                    (lu_val - pa_val).abs() < 1e-10,
                    "LU[{i}][{j}] = {lu_val}, PA[{i}][{j}] = {pa_val}"
                );
            }
        }
    }

    #[test]
    fn test_lu_decompose_identity() {
        let a = MatrixExpr::identity(3);
        let (l, u, perm) = a.lu_decompose().unwrap();

        // For identity: perm should be identity, L = I, U = I
        let empty: HashMap<SymbolId, f64> = HashMap::new();
        for i in 0..3 {
            assert_eq!(perm[i], i);
            for j in 0..3 {
                let expected_l = if i == j { 1.0 } else { 0.0 };
                let expected_u = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (evaluate(l.get(i, j).unwrap(), &empty).unwrap() - expected_l).abs() < 1e-10
                );
                assert!(
                    (evaluate(u.get(i, j).unwrap(), &empty).unwrap() - expected_u).abs() < 1e-10
                );
            }
        }
    }

    #[test]
    fn test_lu_decompose_non_square_error() {
        let m = MatrixExpr::from_expr_elements(vec![
            vec![int(1), int(2), int(3)],
            vec![int(4), int(5), int(6)],
        ])
        .unwrap();

        assert!(m.lu_decompose().is_err());
    }

    #[test]
    fn test_lu_decompose_singular_error() {
        // Singular matrix: rows are linearly dependent.
        let m = MatrixExpr::from_expr_elements(vec![vec![int(1), int(2)], vec![int(2), int(4)]])
            .unwrap();

        assert!(m.lu_decompose().is_err());
    }

    #[test]
    fn test_solve_system_2x2() {
        // Solve [[2, 1], [1, 3]] x = [[3], [4]]  => x = [1, 1]
        let a = MatrixExpr::from_expr_elements(vec![vec![int(2), int(1)], vec![int(1), int(3)]])
            .unwrap();

        let b = MatrixExpr::from_expr_elements(vec![vec![int(3)], vec![int(4)]]).unwrap();

        let x = a.solve_system(&b).unwrap();
        assert_eq!(x.rows(), 2);
        assert_eq!(x.cols(), 1);

        let empty: HashMap<SymbolId, f64> = HashMap::new();
        let x0 = evaluate(x.get(0, 0).unwrap(), &empty).unwrap();
        let x1 = evaluate(x.get(1, 0).unwrap(), &empty).unwrap();
        assert!((x0 - 1.0).abs() < 1e-10, "x[0] = {x0}, expected 1.0");
        assert!((x1 - 1.0).abs() < 1e-10, "x[1] = {x1}, expected 1.0");
    }

    #[test]
    fn test_solve_system_non_column_vector_error() {
        let a = MatrixExpr::identity(2);
        let b = MatrixExpr::from_expr_elements(vec![vec![int(1), int(2)], vec![int(3), int(4)]])
            .unwrap();

        assert!(a.solve_system(&b).is_err());
    }

    #[test]
    fn test_solve_system_dimension_mismatch_error() {
        let a = MatrixExpr::identity(2);
        let b =
            MatrixExpr::from_expr_elements(vec![vec![int(1)], vec![int(2)], vec![int(3)]]).unwrap();

        assert!(a.solve_system(&b).is_err());
    }
}
