//! Matrix expression type with basic linear algebra operations.
//!
//! This module provides a symbolic matrix type where elements are mathematical expressions,
//! supporting operations like addition, multiplication, transpose, and trace with symbolic
//! manipulation capabilities.
//!
//! # Examples
//!
//! ```
//! use thales::matrix::MatrixExpr;
//! use thales::ast::Expression;
//!
//! // Create a 2x2 identity matrix
//! let identity = MatrixExpr::identity(2);
//!
//! // Create a matrix from expressions
//! let a = Expression::Integer(1);
//! let b = Expression::Integer(2);
//! let c = Expression::Integer(3);
//! let d = Expression::Integer(4);
//! let m = MatrixExpr::from_elements(vec![
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

use crate::ast::Expression;

/// A matrix of symbolic expressions.
///
/// Each element is an [`Expression`] allowing symbolic computation on matrices.
/// Supports standard matrix operations including addition, multiplication,
/// transpose, and trace.
///
/// # Examples
///
/// ```
/// use thales::matrix::MatrixExpr;
/// use thales::ast::{Expression, Variable};
///
/// // Create a 2x2 matrix with symbolic entries
/// let x = Expression::Variable(Variable::new("x"));
/// let one = Expression::Integer(1);
/// let two = Expression::Integer(2);
/// let three = Expression::Integer(3);
///
/// let m = MatrixExpr::from_elements(vec![
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
    elements: Vec<Vec<Expression>>,
}

impl MatrixExpr {
    /// Create a matrix from a 2D vector of expressions.
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
    /// use thales::ast::Expression;
    ///
    /// let m = MatrixExpr::from_elements(vec![
    ///     vec![Expression::Integer(1), Expression::Integer(2)],
    ///     vec![Expression::Integer(3), Expression::Integer(4)],
    /// ]).unwrap();
    /// ```
    pub fn from_elements(elements: Vec<Vec<Expression>>) -> MatrixResult<Self> {
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

    /// Create a matrix from pre-validated elements (internal use).
    pub(crate) fn from_elements_unchecked(
        rows: usize,
        cols: usize,
        elements: Vec<Vec<Expression>>,
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
        let elements: Vec<Vec<Expression>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if i == j {
                            Expression::Integer(1)
                        } else {
                            Expression::Integer(0)
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
        let elements: Vec<Vec<Expression>> = (0..rows)
            .map(|_| (0..cols).map(|_| Expression::Integer(0)).collect())
            .collect();
        Self {
            rows,
            cols,
            elements,
        }
    }

    /// Create a diagonal matrix from a vector of expressions.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::ast::Expression;
    ///
    /// let diag = MatrixExpr::diagonal(vec![
    ///     Expression::Integer(1),
    ///     Expression::Integer(2),
    ///     Expression::Integer(3),
    /// ]);
    /// assert_eq!(diag.rows(), 3);
    /// assert_eq!(diag.cols(), 3);
    /// ```
    pub fn diagonal(diag: Vec<Expression>) -> Self {
        let n = diag.len();
        let elements: Vec<Vec<Expression>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if i == j {
                            diag[i].clone()
                        } else {
                            Expression::Integer(0)
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

    /// Get a reference to the elements.
    pub(crate) fn elements(&self) -> &Vec<Vec<Expression>> {
        &self.elements
    }

    /// Get a reference to an element at (row, col).
    ///
    /// # Errors
    ///
    /// Returns an error if indices are out of bounds.
    pub fn get(&self, row: usize, col: usize) -> MatrixResult<&Expression> {
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
    pub fn set(&mut self, row: usize, col: usize, value: Expression) -> MatrixResult<()> {
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

    /// Get a row as a vector of expressions.
    pub fn row(&self, index: usize) -> MatrixResult<&Vec<Expression>> {
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

    /// Get a column as a vector of expressions.
    pub fn col(&self, index: usize) -> MatrixResult<Vec<&Expression>> {
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
    /// use thales::ast::Expression;
    ///
    /// let m = MatrixExpr::from_elements(vec![
    ///     vec![Expression::Integer(1), Expression::Integer(2), Expression::Integer(3)],
    ///     vec![Expression::Integer(4), Expression::Integer(5), Expression::Integer(6)],
    /// ]).unwrap();
    ///
    /// let mt = m.transpose();
    /// assert_eq!(mt.rows(), 3);
    /// assert_eq!(mt.cols(), 2);
    /// ```
    pub fn transpose(&self) -> Self {
        let elements: Vec<Vec<Expression>> = (0..self.cols)
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
    use crate::ast::{Expression, Variable};
    use std::collections::HashMap;

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    #[test]
    fn test_matrix_creation() {
        let m =
            MatrixExpr::from_elements(vec![vec![int(1), int(2)], vec![int(3), int(4)]]).unwrap();

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
        assert_eq!(i3.get(0, 0).unwrap(), &int(1));
        assert_eq!(i3.get(1, 1).unwrap(), &int(1));
        assert_eq!(i3.get(2, 2).unwrap(), &int(1));

        // Check off-diagonal elements are 0
        assert_eq!(i3.get(0, 1).unwrap(), &int(0));
        assert_eq!(i3.get(1, 2).unwrap(), &int(0));
    }

    #[test]
    fn test_zero_matrix() {
        let z = MatrixExpr::zero(2, 3);
        assert_eq!(z.rows(), 2);
        assert_eq!(z.cols(), 3);

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(z.get(i, j).unwrap(), &int(0));
            }
        }
    }

    #[test]
    fn test_diagonal_matrix() {
        let d = MatrixExpr::diagonal(vec![int(1), int(2), int(3)]);
        assert_eq!(d.rows(), 3);
        assert_eq!(d.cols(), 3);

        assert_eq!(d.get(0, 0).unwrap(), &int(1));
        assert_eq!(d.get(1, 1).unwrap(), &int(2));
        assert_eq!(d.get(2, 2).unwrap(), &int(3));
        assert_eq!(d.get(0, 1).unwrap(), &int(0));
    }

    #[test]
    fn test_transpose() {
        let m = MatrixExpr::from_elements(vec![
            vec![int(1), int(2), int(3)],
            vec![int(4), int(5), int(6)],
        ])
        .unwrap();

        let mt = m.transpose();
        assert_eq!(mt.rows(), 3);
        assert_eq!(mt.cols(), 2);

        assert_eq!(mt.get(0, 0).unwrap(), &int(1));
        assert_eq!(mt.get(0, 1).unwrap(), &int(4));
        assert_eq!(mt.get(1, 0).unwrap(), &int(2));
        assert_eq!(mt.get(2, 1).unwrap(), &int(6));
    }

    #[test]
    fn test_double_transpose() {
        let m =
            MatrixExpr::from_elements(vec![vec![int(1), int(2)], vec![int(3), int(4)]]).unwrap();

        let mtt = m.transpose().transpose();
        assert_eq!(mtt.elements, m.elements);
    }

    #[test]
    fn test_trace() {
        let m =
            MatrixExpr::from_elements(vec![vec![int(1), int(2)], vec![int(3), int(4)]]).unwrap();

        let trace = m.trace().unwrap();
        let vars = HashMap::new();
        assert_eq!(trace.evaluate(&vars), Some(5.0));
    }

    #[test]
    fn test_addition() {
        let a =
            MatrixExpr::from_elements(vec![vec![int(1), int(2)], vec![int(3), int(4)]]).unwrap();

        let b =
            MatrixExpr::from_elements(vec![vec![int(5), int(6)], vec![int(7), int(8)]]).unwrap();

        let sum = a.add(&b).unwrap();
        let vars = HashMap::new();

        assert_eq!(sum.get(0, 0).unwrap().evaluate(&vars), Some(6.0));
        assert_eq!(sum.get(0, 1).unwrap().evaluate(&vars), Some(8.0));
        assert_eq!(sum.get(1, 0).unwrap().evaluate(&vars), Some(10.0));
        assert_eq!(sum.get(1, 1).unwrap().evaluate(&vars), Some(12.0));
    }

    #[test]
    fn test_addition_dimension_check() {
        let a = MatrixExpr::from_elements(vec![vec![int(1), int(2)]]).unwrap();

        let b = MatrixExpr::from_elements(vec![vec![int(1)], vec![int(2)]]).unwrap();

        let result = a.add(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_matrix_multiplication() {
        // 2x3 * 3x2 = 2x2
        let a = MatrixExpr::from_elements(vec![
            vec![int(1), int(2), int(3)],
            vec![int(4), int(5), int(6)],
        ])
        .unwrap();

        let b = MatrixExpr::from_elements(vec![
            vec![int(7), int(8)],
            vec![int(9), int(10)],
            vec![int(11), int(12)],
        ])
        .unwrap();

        let c = a.mul(&b).unwrap();
        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 2);

        let vars = HashMap::new();
        // C[0][0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        assert_eq!(c.get(0, 0).unwrap().evaluate(&vars), Some(58.0));
        // C[0][1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
        assert_eq!(c.get(0, 1).unwrap().evaluate(&vars), Some(64.0));
        // C[1][0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
        assert_eq!(c.get(1, 0).unwrap().evaluate(&vars), Some(139.0));
        // C[1][1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
        assert_eq!(c.get(1, 1).unwrap().evaluate(&vars), Some(154.0));
    }

    #[test]
    fn test_scalar_multiplication() {
        let m = MatrixExpr::identity(2);
        let scaled = m.scalar_mul(&int(3));

        let vars = HashMap::new();
        assert_eq!(scaled.get(0, 0).unwrap().evaluate(&vars), Some(3.0));
        assert_eq!(scaled.get(1, 1).unwrap().evaluate(&vars), Some(3.0));
        assert_eq!(scaled.get(0, 1).unwrap().evaluate(&vars), Some(0.0));
    }

    #[test]
    fn test_symbolic_matrix() {
        let m = MatrixExpr::from_elements(vec![vec![var("a"), var("b")], vec![var("c"), var("d")]])
            .unwrap();

        let mut vars = HashMap::new();
        vars.insert("a".to_string(), 1.0);
        vars.insert("b".to_string(), 2.0);
        vars.insert("c".to_string(), 3.0);
        vars.insert("d".to_string(), 4.0);

        let result = m.evaluate(&vars).unwrap();
        assert_eq!(result[0][0], 1.0);
        assert_eq!(result[0][1], 2.0);
        assert_eq!(result[1][0], 3.0);
        assert_eq!(result[1][1], 4.0);
    }

    #[test]
    fn test_latex_output() {
        let m =
            MatrixExpr::from_elements(vec![vec![int(1), int(2)], vec![int(3), int(4)]]).unwrap();

        let latex = m.to_latex(BracketStyle::Parentheses);
        assert!(latex.contains("\\begin{pmatrix}"));
        assert!(latex.contains("\\end{pmatrix}"));
        assert!(latex.contains("1 & 2"));
        assert!(latex.contains("3 & 4"));
    }

    #[test]
    fn test_transpose_multiplication_property() {
        // (AB)^T = B^T A^T
        let a =
            MatrixExpr::from_elements(vec![vec![int(1), int(2)], vec![int(3), int(4)]]).unwrap();

        let b =
            MatrixExpr::from_elements(vec![vec![int(5), int(6)], vec![int(7), int(8)]]).unwrap();

        let ab = a.mul(&b).unwrap();
        let ab_t = ab.transpose();

        let bt_at = b.transpose().mul(&a.transpose()).unwrap();

        let vars = HashMap::new();
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(
                    ab_t.get(i, j).unwrap().evaluate(&vars),
                    bt_at.get(i, j).unwrap().evaluate(&vars)
                );
            }
        }
    }

    #[test]
    fn test_determinant_2x2() {
        // det([[1, 2], [3, 4]]) = 1*4 - 2*3 = -2
        let m =
            MatrixExpr::from_elements(vec![vec![int(1), int(2)], vec![int(3), int(4)]]).unwrap();

        let det = m.determinant().unwrap();
        let vars = HashMap::new();
        assert_eq!(det.evaluate(&vars), Some(-2.0));
    }

    #[test]
    fn test_determinant_3x3() {
        // det([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) = 0 (rows are linearly dependent)
        let m = MatrixExpr::from_elements(vec![
            vec![int(1), int(2), int(3)],
            vec![int(4), int(5), int(6)],
            vec![int(7), int(8), int(9)],
        ])
        .unwrap();

        let det = m.determinant().unwrap();
        let vars = HashMap::new();
        assert_eq!(det.evaluate(&vars), Some(0.0));
    }

    #[test]
    fn test_determinant_3x3_nonzero() {
        // det([[1, 2, 3], [0, 1, 4], [5, 6, 0]]) = 1
        let m = MatrixExpr::from_elements(vec![
            vec![int(1), int(2), int(3)],
            vec![int(0), int(1), int(4)],
            vec![int(5), int(6), int(0)],
        ])
        .unwrap();

        let det = m.determinant().unwrap();
        let vars = HashMap::new();
        assert_eq!(det.evaluate(&vars), Some(1.0));
    }

    #[test]
    fn test_determinant_identity() {
        // det(I) = 1
        let i3 = MatrixExpr::identity(3);
        let det = i3.determinant().unwrap();
        let vars = HashMap::new();
        assert_eq!(det.evaluate(&vars), Some(1.0));
    }

    #[test]
    fn test_determinant_non_square() {
        let m = MatrixExpr::from_elements(vec![
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
        let m =
            MatrixExpr::from_elements(vec![vec![int(4), int(7)], vec![int(2), int(6)]]).unwrap();

        let inv = m.inverse().unwrap();
        let vars = HashMap::new();

        // Verify A * A^(-1) = I
        let product = m.mul(&inv).unwrap();
        let result = product.evaluate(&vars).unwrap();

        assert!((result[0][0] - 1.0).abs() < 1e-10);
        assert!((result[0][1] - 0.0).abs() < 1e-10);
        assert!((result[1][0] - 0.0).abs() < 1e-10);
        assert!((result[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_3x3() {
        // A = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
        let m = MatrixExpr::from_elements(vec![
            vec![int(1), int(2), int(3)],
            vec![int(0), int(1), int(4)],
            vec![int(5), int(6), int(0)],
        ])
        .unwrap();

        let inv = m.inverse().unwrap();
        let vars = HashMap::new();

        // Verify A * A^(-1) = I
        let product = m.mul(&inv).unwrap();
        let result = product.evaluate(&vars).unwrap();

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
        let m =
            MatrixExpr::from_elements(vec![vec![int(1), int(2)], vec![int(2), int(4)]]).unwrap();

        let result = m.inverse();
        assert!(result.is_err());
    }

    #[test]
    fn test_determinant_symbolic() {
        // det([[a, b], [c, d]]) = ad - bc
        let m = MatrixExpr::from_elements(vec![vec![var("a"), var("b")], vec![var("c"), var("d")]])
            .unwrap();

        let det = m.determinant().unwrap();

        let mut vars = HashMap::new();
        vars.insert("a".to_string(), 2.0);
        vars.insert("b".to_string(), 3.0);
        vars.insert("c".to_string(), 4.0);
        vars.insert("d".to_string(), 5.0);

        // det = 2*5 - 3*4 = 10 - 12 = -2
        assert_eq!(det.evaluate(&vars), Some(-2.0));
    }

    #[test]
    fn test_submatrix() {
        let m = MatrixExpr::from_elements(vec![
            vec![int(1), int(2), int(3)],
            vec![int(4), int(5), int(6)],
            vec![int(7), int(8), int(9)],
        ])
        .unwrap();

        // Remove row 1, col 1 -> [[1, 3], [7, 9]]
        let sub = m.submatrix(1, 1).unwrap();
        let vars = HashMap::new();

        assert_eq!(sub.rows(), 2);
        assert_eq!(sub.cols(), 2);
        assert_eq!(sub.get(0, 0).unwrap().evaluate(&vars), Some(1.0));
        assert_eq!(sub.get(0, 1).unwrap().evaluate(&vars), Some(3.0));
        assert_eq!(sub.get(1, 0).unwrap().evaluate(&vars), Some(7.0));
        assert_eq!(sub.get(1, 1).unwrap().evaluate(&vars), Some(9.0));
    }

    #[test]
    fn test_adjugate_2x2() {
        // adj([[a, b], [c, d]]) = [[d, -b], [-c, a]]
        let m =
            MatrixExpr::from_elements(vec![vec![int(1), int(2)], vec![int(3), int(4)]]).unwrap();

        let adj = m.adjugate().unwrap();
        let vars = HashMap::new();

        assert_eq!(adj.get(0, 0).unwrap().evaluate(&vars), Some(4.0));
        assert_eq!(adj.get(0, 1).unwrap().evaluate(&vars), Some(-2.0));
        assert_eq!(adj.get(1, 0).unwrap().evaluate(&vars), Some(-3.0));
        assert_eq!(adj.get(1, 1).unwrap().evaluate(&vars), Some(1.0));
    }

    #[test]
    fn test_is_singular() {
        let singular =
            MatrixExpr::from_elements(vec![vec![int(1), int(2)], vec![int(2), int(4)]]).unwrap();

        let non_singular =
            MatrixExpr::from_elements(vec![vec![int(1), int(2)], vec![int(3), int(4)]]).unwrap();

        let vars = HashMap::new();
        assert_eq!(singular.is_singular(&vars), Some(true));
        assert_eq!(non_singular.is_singular(&vars), Some(false));
    }

    #[test]
    fn test_inverse_identity() {
        // I^(-1) = I
        let i3 = MatrixExpr::identity(3);
        let inv = i3.inverse().unwrap();
        let vars = HashMap::new();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(inv.get(i, j).unwrap().evaluate(&vars), Some(expected));
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
        let m =
            MatrixExpr::from_elements(vec![vec![int(2), int(1)], vec![int(1), int(2)]]).unwrap();

        let char_poly = m.characteristic_polynomial("lambda").unwrap();

        // Evaluate at λ = 1 (should be 0)
        let mut vars = HashMap::new();
        vars.insert("lambda".to_string(), 1.0);
        let at_1 = char_poly.evaluate(&vars).unwrap();
        assert!(
            at_1.abs() < 1e-10,
            "char poly at λ=1 should be 0, got {}",
            at_1
        );

        // Evaluate at λ = 3 (should be 0)
        vars.insert("lambda".to_string(), 3.0);
        let at_3 = char_poly.evaluate(&vars).unwrap();
        assert!(
            at_3.abs() < 1e-10,
            "char poly at λ=3 should be 0, got {}",
            at_3
        );
    }

    #[test]
    fn test_eigenvalues_2x2_symmetric() {
        // A = [[2, 1], [1, 2]], eigenvalues are 1 and 3
        let m =
            MatrixExpr::from_elements(vec![vec![int(2), int(1)], vec![int(1), int(2)]]).unwrap();

        let eigenvalues = m.eigenvalues_numeric().unwrap();
        assert_eq!(eigenvalues.len(), 2);

        // Sort eigenvalues for consistent comparison
        let mut sorted = eigenvalues.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!(
            (sorted[0] - 1.0).abs() < 1e-10,
            "Expected 1, got {}",
            sorted[0]
        );
        assert!(
            (sorted[1] - 3.0).abs() < 1e-10,
            "Expected 3, got {}",
            sorted[1]
        );
    }

    #[test]
    fn test_eigenvalues_diagonal() {
        // Diagonal matrix: eigenvalues are the diagonal elements
        let m =
            MatrixExpr::from_elements(vec![vec![int(5), int(0)], vec![int(0), int(3)]]).unwrap();

        let eigenvalues = m.eigenvalues_numeric().unwrap();
        let mut sorted = eigenvalues.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!((sorted[0] - 3.0).abs() < 1e-10);
        assert!((sorted[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_eigenvalues_identity() {
        // Identity matrix: all eigenvalues are 1
        let m = MatrixExpr::identity(3);

        let eigenvalues = m.eigenvalues_numeric().unwrap();
        assert_eq!(eigenvalues.len(), 3);

        for ev in eigenvalues {
            assert!((ev - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_eigenvector_2x2() {
        // A = [[2, 1], [1, 2]], eigenvalue 3 has eigenvector [1, 1]
        let m =
            MatrixExpr::from_elements(vec![vec![int(2), int(1)], vec![int(1), int(2)]]).unwrap();

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
        let m =
            MatrixExpr::from_elements(vec![vec![int(2), int(1)], vec![int(1), int(2)]]).unwrap();

        let pairs = m.eigenpairs_numeric().unwrap();
        assert_eq!(pairs.len(), 2);

        for (eigenvalue, eigenvector) in pairs {
            // Verify Av = λv
            let empty = HashMap::new();
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

            // Compute λv
            let lambda_v: Vec<f64> = eigenvector.iter().map(|v| eigenvalue * v).collect();

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
        let m = MatrixExpr::from_elements(vec![
            vec![int(1), int(0), int(0)],
            vec![int(0), int(2), int(0)],
            vec![int(0), int(0), int(3)],
        ])
        .unwrap();

        let eigenvalues = m.eigenvalues_numeric().unwrap();
        let mut sorted = eigenvalues.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!((sorted[0] - 1.0).abs() < 1e-10);
        assert!((sorted[1] - 2.0).abs() < 1e-10);
        assert!((sorted[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_diagonalizable_symmetric() {
        // Symmetric matrices are always diagonalizable
        let m =
            MatrixExpr::from_elements(vec![vec![int(2), int(1)], vec![int(1), int(2)]]).unwrap();

        assert!(m.is_diagonalizable().unwrap());
    }

    #[test]
    fn test_is_diagonalizable_identity() {
        let m = MatrixExpr::identity(3);
        assert!(m.is_diagonalizable().unwrap());
    }

    #[test]
    fn test_eigenvalues_non_square() {
        let m = MatrixExpr::from_elements(vec![
            vec![int(1), int(2), int(3)],
            vec![int(4), int(5), int(6)],
        ])
        .unwrap();

        let result = m.eigenvalues_numeric();
        assert!(result.is_err());
    }

    #[test]
    fn test_characteristic_polynomial_non_square() {
        let m = MatrixExpr::from_elements(vec![
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
        let mut elements = vec![vec![Expression::Integer(0); n]; n];
        for (i, &src) in perm.iter().enumerate() {
            elements[i][src] = Expression::Integer(1);
        }
        MatrixExpr::from_elements(elements).unwrap()
    }

    #[test]
    fn test_lu_decompose_2x2() {
        // A = [[4, 3], [6, 3]]
        let a =
            MatrixExpr::from_elements(vec![vec![int(4), int(3)], vec![int(6), int(3)]]).unwrap();

        let (l, u, perm) = a.lu_decompose().unwrap();
        assert_eq!(l.rows(), 2);
        assert_eq!(u.rows(), 2);
        assert_eq!(perm.len(), 2);

        // Verify L·U = P·A
        let pa = perm_matrix(&perm).mul(&a).unwrap();
        let lu_prod = l.mul(&u).unwrap();
        let vars = HashMap::new();
        for i in 0..2 {
            for j in 0..2 {
                let lu_val = lu_prod.get(i, j).unwrap().evaluate(&vars).unwrap();
                let pa_val = pa.get(i, j).unwrap().evaluate(&vars).unwrap();
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
        let a = MatrixExpr::from_elements(vec![
            vec![int(1), int(2), int(3)],
            vec![int(0), int(1), int(4)],
            vec![int(5), int(6), int(0)],
        ])
        .unwrap();

        let (l, u, perm) = a.lu_decompose().unwrap();

        // Verify L·U = P·A
        let pa = perm_matrix(&perm).mul(&a).unwrap();
        let lu_prod = l.mul(&u).unwrap();
        let vars = HashMap::new();
        for i in 0..3 {
            for j in 0..3 {
                let lu_val = lu_prod.get(i, j).unwrap().evaluate(&vars).unwrap();
                let pa_val = pa.get(i, j).unwrap().evaluate(&vars).unwrap();
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
        let vars = HashMap::new();
        for i in 0..3 {
            assert_eq!(perm[i], i);
            for j in 0..3 {
                let expected_l = if i == j { 1.0 } else { 0.0 };
                let expected_u = if i == j { 1.0 } else { 0.0 };
                assert!((l.get(i, j).unwrap().evaluate(&vars).unwrap() - expected_l).abs() < 1e-10);
                assert!((u.get(i, j).unwrap().evaluate(&vars).unwrap() - expected_u).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_lu_decompose_non_square_error() {
        let m = MatrixExpr::from_elements(vec![
            vec![int(1), int(2), int(3)],
            vec![int(4), int(5), int(6)],
        ])
        .unwrap();

        assert!(m.lu_decompose().is_err());
    }

    #[test]
    fn test_lu_decompose_singular_error() {
        // Singular matrix: rows are linearly dependent.
        let m =
            MatrixExpr::from_elements(vec![vec![int(1), int(2)], vec![int(2), int(4)]]).unwrap();

        assert!(m.lu_decompose().is_err());
    }

    #[test]
    fn test_solve_system_2x2() {
        // Solve [[2, 1], [1, 3]] x = [[3], [4]]  => x = [1, 1]
        let a =
            MatrixExpr::from_elements(vec![vec![int(2), int(1)], vec![int(1), int(3)]]).unwrap();

        let b = MatrixExpr::from_elements(vec![vec![int(3)], vec![int(4)]]).unwrap();

        let x = a.solve_system(&b).unwrap();
        assert_eq!(x.rows(), 2);
        assert_eq!(x.cols(), 1);

        let vars = HashMap::new();
        let x0 = x.get(0, 0).unwrap().evaluate(&vars).unwrap();
        let x1 = x.get(1, 0).unwrap().evaluate(&vars).unwrap();
        assert!((x0 - 1.0).abs() < 1e-10, "x[0] = {x0}, expected 1.0");
        assert!((x1 - 1.0).abs() < 1e-10, "x[1] = {x1}, expected 1.0");
    }

    #[test]
    fn test_solve_system_non_column_vector_error() {
        let a = MatrixExpr::identity(2);
        let b =
            MatrixExpr::from_elements(vec![vec![int(1), int(2)], vec![int(3), int(4)]]).unwrap();

        assert!(a.solve_system(&b).is_err());
    }

    #[test]
    fn test_solve_system_dimension_mismatch_error() {
        let a = MatrixExpr::identity(2);
        let b = MatrixExpr::from_elements(vec![vec![int(1)], vec![int(2)], vec![int(3)]]).unwrap();

        assert!(a.solve_system(&b).is_err());
    }
}
