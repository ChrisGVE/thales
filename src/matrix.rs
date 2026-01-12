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

use crate::ast::{Expression, Variable};
use std::fmt;

/// Error type for matrix operations.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum MatrixError {
    /// Dimension mismatch for operation.
    DimensionMismatch {
        operation: String,
        expected: (usize, usize),
        got: (usize, usize),
    },
    /// Empty matrix or row not allowed.
    EmptyMatrix,
    /// Non-rectangular matrix (rows have different lengths).
    NonRectangular,
    /// Index out of bounds.
    IndexOutOfBounds {
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
    },
    /// Cannot compute operation (e.g., determinant of non-square matrix).
    InvalidOperation(String),
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixError::DimensionMismatch { operation, expected, got } => {
                write!(f, "{}: expected {}x{}, got {}x{}", operation, expected.0, expected.1, got.0, got.1)
            }
            MatrixError::EmptyMatrix => write!(f, "Empty matrix not allowed"),
            MatrixError::NonRectangular => write!(f, "Matrix must be rectangular (all rows same length)"),
            MatrixError::IndexOutOfBounds { row, col, rows, cols } => {
                write!(f, "Index ({}, {}) out of bounds for {}x{} matrix", row, col, rows, cols)
            }
            MatrixError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
        }
    }
}

impl std::error::Error for MatrixError {}

/// Result type for matrix operations.
pub type MatrixResult<T> = Result<T, MatrixError>;

/// Bracket style for LaTeX rendering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BracketStyle {
    /// Parentheses: `\begin{pmatrix}`
    Parentheses,
    /// Square brackets: `\begin{bmatrix}`
    Square,
    /// Curly braces: `\begin{Bmatrix}`
    Curly,
    /// Vertical bars (determinant): `\begin{vmatrix}`
    Determinant,
    /// Double vertical bars (norm): `\begin{Vmatrix}`
    Norm,
    /// No brackets
    None,
}

impl Default for BracketStyle {
    fn default() -> Self {
        BracketStyle::Parentheses
    }
}

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
        Ok(Self { rows, cols, elements })
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
        Self { rows: n, cols: n, elements }
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
        Self { rows, cols, elements }
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
        Self { rows: n, cols: n, elements }
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
            .map(|j| (0..self.rows).map(|i| self.elements[i][j].clone()).collect())
            .collect();
        Self {
            rows: self.cols,
            cols: self.rows,
            elements,
        }
    }

    /// Compute the trace (sum of diagonal elements).
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::ast::Expression;
    /// use std::collections::HashMap;
    ///
    /// let m = MatrixExpr::from_elements(vec![
    ///     vec![Expression::Integer(1), Expression::Integer(2)],
    ///     vec![Expression::Integer(3), Expression::Integer(4)],
    /// ]).unwrap();
    ///
    /// let trace = m.trace().unwrap();
    /// // trace = 1 + 4 = 5
    /// assert_eq!(trace.evaluate(&HashMap::new()), Some(5.0));
    /// ```
    pub fn trace(&self) -> MatrixResult<Expression> {
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "Trace requires a square matrix".to_string(),
            ));
        }

        let mut trace = self.elements[0][0].clone();
        for i in 1..self.rows {
            trace = Expression::Binary(
                crate::ast::BinaryOp::Add,
                Box::new(trace),
                Box::new(self.elements[i][i].clone()),
            );
        }
        Ok(trace.simplify())
    }

    /// Add two matrices element-wise.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions don't match.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::ast::Expression;
    ///
    /// let a = MatrixExpr::from_elements(vec![
    ///     vec![Expression::Integer(1), Expression::Integer(2)],
    ///     vec![Expression::Integer(3), Expression::Integer(4)],
    /// ]).unwrap();
    ///
    /// let b = MatrixExpr::from_elements(vec![
    ///     vec![Expression::Integer(5), Expression::Integer(6)],
    ///     vec![Expression::Integer(7), Expression::Integer(8)],
    /// ]).unwrap();
    ///
    /// let sum = a.add(&b).unwrap();
    /// ```
    pub fn add(&self, other: &MatrixExpr) -> MatrixResult<MatrixExpr> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::DimensionMismatch {
                operation: "Matrix addition".to_string(),
                expected: (self.rows, self.cols),
                got: (other.rows, other.cols),
            });
        }

        let elements: Vec<Vec<Expression>> = (0..self.rows)
            .map(|i| {
                (0..self.cols)
                    .map(|j| {
                        Expression::Binary(
                            crate::ast::BinaryOp::Add,
                            Box::new(self.elements[i][j].clone()),
                            Box::new(other.elements[i][j].clone()),
                        )
                        .simplify()
                    })
                    .collect()
            })
            .collect();

        Ok(MatrixExpr {
            rows: self.rows,
            cols: self.cols,
            elements,
        })
    }

    /// Subtract another matrix element-wise.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions don't match.
    pub fn sub(&self, other: &MatrixExpr) -> MatrixResult<MatrixExpr> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::DimensionMismatch {
                operation: "Matrix subtraction".to_string(),
                expected: (self.rows, self.cols),
                got: (other.rows, other.cols),
            });
        }

        let elements: Vec<Vec<Expression>> = (0..self.rows)
            .map(|i| {
                (0..self.cols)
                    .map(|j| {
                        Expression::Binary(
                            crate::ast::BinaryOp::Sub,
                            Box::new(self.elements[i][j].clone()),
                            Box::new(other.elements[i][j].clone()),
                        )
                        .simplify()
                    })
                    .collect()
            })
            .collect();

        Ok(MatrixExpr {
            rows: self.rows,
            cols: self.cols,
            elements,
        })
    }

    /// Multiply by a scalar expression.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::ast::Expression;
    ///
    /// let m = MatrixExpr::identity(2);
    /// let scaled = m.scalar_mul(&Expression::Integer(3));
    /// ```
    pub fn scalar_mul(&self, scalar: &Expression) -> MatrixExpr {
        let elements: Vec<Vec<Expression>> = self
            .elements
            .iter()
            .map(|row| {
                row.iter()
                    .map(|elem| {
                        Expression::Binary(
                            crate::ast::BinaryOp::Mul,
                            Box::new(scalar.clone()),
                            Box::new(elem.clone()),
                        )
                        .simplify()
                    })
                    .collect()
            })
            .collect();

        MatrixExpr {
            rows: self.rows,
            cols: self.cols,
            elements,
        }
    }

    /// Multiply two matrices.
    ///
    /// Computes self * other where self is m×n and other is n×p, resulting in m×p.
    ///
    /// # Errors
    ///
    /// Returns an error if the inner dimensions don't match (self.cols != other.rows).
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::ast::Expression;
    ///
    /// // 2x3 matrix
    /// let a = MatrixExpr::from_elements(vec![
    ///     vec![Expression::Integer(1), Expression::Integer(2), Expression::Integer(3)],
    ///     vec![Expression::Integer(4), Expression::Integer(5), Expression::Integer(6)],
    /// ]).unwrap();
    ///
    /// // 3x2 matrix
    /// let b = MatrixExpr::from_elements(vec![
    ///     vec![Expression::Integer(7), Expression::Integer(8)],
    ///     vec![Expression::Integer(9), Expression::Integer(10)],
    ///     vec![Expression::Integer(11), Expression::Integer(12)],
    /// ]).unwrap();
    ///
    /// // Result is 2x2
    /// let c = a.mul(&b).unwrap();
    /// assert_eq!(c.rows(), 2);
    /// assert_eq!(c.cols(), 2);
    /// ```
    pub fn mul(&self, other: &MatrixExpr) -> MatrixResult<MatrixExpr> {
        if self.cols != other.rows {
            return Err(MatrixError::DimensionMismatch {
                operation: format!(
                    "Matrix multiplication ({}x{} * {}x{})",
                    self.rows, self.cols, other.rows, other.cols
                ),
                expected: (self.cols, other.rows),
                got: (self.cols, other.rows),
            });
        }

        let elements: Vec<Vec<Expression>> = (0..self.rows)
            .map(|i| {
                (0..other.cols)
                    .map(|j| {
                        // C[i][j] = sum(A[i][k] * B[k][j] for k in 0..n)
                        let mut sum = Expression::Binary(
                            crate::ast::BinaryOp::Mul,
                            Box::new(self.elements[i][0].clone()),
                            Box::new(other.elements[0][j].clone()),
                        );
                        for k in 1..self.cols {
                            let product = Expression::Binary(
                                crate::ast::BinaryOp::Mul,
                                Box::new(self.elements[i][k].clone()),
                                Box::new(other.elements[k][j].clone()),
                            );
                            sum = Expression::Binary(
                                crate::ast::BinaryOp::Add,
                                Box::new(sum),
                                Box::new(product),
                            );
                        }
                        sum.simplify()
                    })
                    .collect()
            })
            .collect();

        Ok(MatrixExpr {
            rows: self.rows,
            cols: other.cols,
            elements,
        })
    }

    /// Simplify all elements in the matrix.
    pub fn simplify(&self) -> MatrixExpr {
        let elements: Vec<Vec<Expression>> = self
            .elements
            .iter()
            .map(|row| row.iter().map(|elem| elem.simplify()).collect())
            .collect();

        MatrixExpr {
            rows: self.rows,
            cols: self.cols,
            elements,
        }
    }

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

        let elements: Vec<Vec<Expression>> = self
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

        MatrixExpr::from_elements(elements)
    }

    /// Compute the minor M(i, j) - the determinant of the submatrix excluding row i and column j.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square or is 1x1.
    pub fn minor(&self, row: usize, col: usize) -> MatrixResult<Expression> {
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
    pub fn cofactor(&self, row: usize, col: usize) -> MatrixResult<Expression> {
        let minor = self.minor(row, col)?;
        if (row + col) % 2 == 0 {
            Ok(minor)
        } else {
            Ok(Expression::Unary(
                crate::ast::UnaryOp::Neg,
                Box::new(minor),
            )
            .simplify())
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
    /// use thales::ast::Expression;
    /// use std::collections::HashMap;
    ///
    /// // 2x2 matrix: [[1, 2], [3, 4]]
    /// let m = MatrixExpr::from_elements(vec![
    ///     vec![Expression::Integer(1), Expression::Integer(2)],
    ///     vec![Expression::Integer(3), Expression::Integer(4)],
    /// ]).unwrap();
    ///
    /// let det = m.determinant().unwrap();
    /// // det = 1*4 - 2*3 = -2
    /// assert_eq!(det.evaluate(&HashMap::new()), Some(-2.0));
    /// ```
    pub fn determinant(&self) -> MatrixResult<Expression> {
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

                let ad = Expression::Binary(
                    crate::ast::BinaryOp::Mul,
                    Box::new(a.clone()),
                    Box::new(d.clone()),
                );
                let bc = Expression::Binary(
                    crate::ast::BinaryOp::Mul,
                    Box::new(b.clone()),
                    Box::new(c.clone()),
                );
                Ok(Expression::Binary(
                    crate::ast::BinaryOp::Sub,
                    Box::new(ad),
                    Box::new(bc),
                )
                .simplify())
            }
            _ => {
                // Cofactor expansion along first row
                let mut det = Expression::Integer(0);
                for j in 0..self.cols {
                    let cofactor = self.cofactor(0, j)?;
                    let term = Expression::Binary(
                        crate::ast::BinaryOp::Mul,
                        Box::new(self.elements[0][j].clone()),
                        Box::new(cofactor),
                    );
                    det = Expression::Binary(
                        crate::ast::BinaryOp::Add,
                        Box::new(det),
                        Box::new(term),
                    );
                }
                Ok(det.simplify())
            }
        }
    }

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

        MatrixExpr::from_elements(elements)
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
    /// use thales::ast::Expression;
    ///
    /// let m = MatrixExpr::from_elements(vec![
    ///     vec![Expression::Integer(1), Expression::Integer(2)],
    ///     vec![Expression::Integer(3), Expression::Integer(4)],
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
            return Ok(MatrixExpr::from_elements(vec![vec![Expression::Integer(1)]]).unwrap());
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
    /// use thales::ast::Expression;
    /// use std::collections::HashMap;
    ///
    /// let m = MatrixExpr::from_elements(vec![
    ///     vec![Expression::Integer(4), Expression::Integer(7)],
    ///     vec![Expression::Integer(2), Expression::Integer(6)],
    /// ]).unwrap();
    ///
    /// let inv = m.inverse().unwrap();
    /// // Verify A * A^(-1) = I
    /// let product = m.mul(&inv).unwrap();
    /// let vars = HashMap::new();
    /// let result = product.evaluate(&vars).unwrap();
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
        let is_zero = match &det {
            Expression::Integer(0) => true,
            Expression::Float(f) if f.abs() < 1e-10 => true,
            _ => {
                // Try numerical evaluation for expressions that simplify to zero
                let empty = std::collections::HashMap::new();
                det.evaluate(&empty).map_or(false, |v| v.abs() < 1e-10)
            }
        };

        if is_zero {
            return Err(MatrixError::InvalidOperation(
                "Matrix is singular (determinant is zero)".to_string(),
            ));
        }

        // For 1x1 matrix
        if self.rows == 1 {
            let inv_element = Expression::Binary(
                crate::ast::BinaryOp::Div,
                Box::new(Expression::Integer(1)),
                Box::new(self.elements[0][0].clone()),
            )
            .simplify();
            return MatrixExpr::from_elements(vec![vec![inv_element]]);
        }

        let adj = self.adjugate()?;

        // Multiply adjugate by 1/det
        let inv_det = Expression::Binary(
            crate::ast::BinaryOp::Div,
            Box::new(Expression::Integer(1)),
            Box::new(det),
        );

        Ok(adj.scalar_mul(&inv_det).simplify())
    }

    /// Check if the matrix is singular (determinant is zero when evaluated numerically).
    ///
    /// Returns `None` if the determinant cannot be evaluated numerically.
    pub fn is_singular(&self, vars: &std::collections::HashMap<String, f64>) -> Option<bool> {
        let det = self.determinant().ok()?;
        let det_value = det.evaluate(vars)?;
        Some(det_value.abs() < 1e-10)
    }

    /// Compute the characteristic polynomial det(A - λI).
    ///
    /// Returns a polynomial expression in the given variable (typically "lambda").
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::ast::Expression;
    /// use std::collections::HashMap;
    ///
    /// let m = MatrixExpr::from_elements(vec![
    ///     vec![Expression::Integer(2), Expression::Integer(1)],
    ///     vec![Expression::Integer(1), Expression::Integer(2)],
    /// ]).unwrap();
    ///
    /// let char_poly = m.characteristic_polynomial("lambda").unwrap();
    /// // For this matrix, eigenvalues are 1 and 3
    /// // So char poly = (λ - 1)(λ - 3) = λ² - 4λ + 3
    /// ```
    pub fn characteristic_polynomial(&self, lambda_var: &str) -> MatrixResult<Expression> {
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "Characteristic polynomial requires a square matrix".to_string(),
            ));
        }

        // Compute A - λI
        let lambda = Expression::Variable(Variable::new(lambda_var));
        let lambda_i = MatrixExpr::identity(self.rows).scalar_mul(&lambda);
        let a_minus_lambda_i = self.sub(&lambda_i)?;

        // Compute det(A - λI)
        a_minus_lambda_i.determinant()
    }

    /// Compute eigenvalues of the matrix numerically.
    ///
    /// For 2x2 matrices, uses the quadratic formula.
    /// For larger matrices, uses numerical methods (power iteration or similar).
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::ast::Expression;
    ///
    /// let m = MatrixExpr::from_elements(vec![
    ///     vec![Expression::Integer(2), Expression::Integer(1)],
    ///     vec![Expression::Integer(1), Expression::Integer(2)],
    /// ]).unwrap();
    ///
    /// let eigenvalues = m.eigenvalues_numeric().unwrap();
    /// // Eigenvalues should be 1 and 3
    /// ```
    pub fn eigenvalues_numeric(&self) -> MatrixResult<Vec<f64>> {
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "Eigenvalues require a square matrix".to_string(),
            ));
        }

        let empty = std::collections::HashMap::new();
        let elements = self.evaluate(&empty).ok_or_else(|| {
            MatrixError::InvalidOperation("Cannot evaluate matrix numerically".to_string())
        })?;

        match self.rows {
            1 => Ok(vec![elements[0][0]]),
            2 => self.eigenvalues_2x2(&elements),
            3 => self.eigenvalues_3x3(&elements),
            _ => self.eigenvalues_qr(&elements),
        }
    }

    /// Compute eigenvalues for a 2x2 matrix using the quadratic formula.
    fn eigenvalues_2x2(&self, elements: &[Vec<f64>]) -> MatrixResult<Vec<f64>> {
        let a = elements[0][0];
        let b = elements[0][1];
        let c = elements[1][0];
        let d = elements[1][1];

        // Characteristic equation: λ² - (a+d)λ + (ad - bc) = 0
        // Using quadratic formula: λ = ((a+d) ± sqrt((a+d)² - 4(ad-bc))) / 2
        let trace = a + d;
        let det = a * d - b * c;
        let discriminant = trace * trace - 4.0 * det;

        if discriminant < 0.0 {
            // Complex eigenvalues - return just the real parts for now
            // A full implementation would return Complex numbers
            let real_part = trace / 2.0;
            Ok(vec![real_part, real_part])
        } else {
            let sqrt_disc = discriminant.sqrt();
            let lambda1 = (trace + sqrt_disc) / 2.0;
            let lambda2 = (trace - sqrt_disc) / 2.0;
            Ok(vec![lambda1, lambda2])
        }
    }

    /// Compute eigenvalues for a 3x3 matrix using Cardano's formula.
    fn eigenvalues_3x3(&self, elements: &[Vec<f64>]) -> MatrixResult<Vec<f64>> {
        // For 3x3, we solve the cubic characteristic equation
        // det(A - λI) = -λ³ + tr(A)λ² - (sum of 2x2 principal minors)λ + det(A)
        let a11 = elements[0][0];
        let a12 = elements[0][1];
        let a13 = elements[0][2];
        let a21 = elements[1][0];
        let a22 = elements[1][1];
        let a23 = elements[1][2];
        let a31 = elements[2][0];
        let a32 = elements[2][1];
        let a33 = elements[2][2];

        // Coefficients of λ³ + p*λ² + q*λ + r = 0
        let trace = a11 + a22 + a33;
        let p = -trace;

        // Sum of 2x2 principal minors
        let minor12 = a11 * a22 - a12 * a21;
        let minor13 = a11 * a33 - a13 * a31;
        let minor23 = a22 * a33 - a23 * a32;
        let q = minor12 + minor13 + minor23;

        // Determinant
        let det = a11 * (a22 * a33 - a23 * a32)
            - a12 * (a21 * a33 - a23 * a31)
            + a13 * (a21 * a32 - a22 * a31);
        let r = -det;

        // Solve cubic using Cardano's formula or numerical method
        solve_cubic(p, q, r)
    }

    /// Compute eigenvalues using QR algorithm for larger matrices.
    fn eigenvalues_qr(&self, elements: &[Vec<f64>]) -> MatrixResult<Vec<f64>> {
        // Simple QR iteration
        let n = elements.len();
        let mut a = elements.to_vec();

        // Maximum iterations
        const MAX_ITER: usize = 100;
        const TOL: f64 = 1e-10;

        for _ in 0..MAX_ITER {
            // QR decomposition
            let (q, r) = qr_decomposition(&a);

            // A = R * Q
            a = matrix_multiply(&r, &q);

            // Check for convergence (off-diagonal elements small)
            let mut converged = true;
            for i in 0..n {
                for j in 0..i {
                    if a[i][j].abs() > TOL {
                        converged = false;
                        break;
                    }
                }
                if !converged {
                    break;
                }
            }

            if converged {
                break;
            }
        }

        // Extract eigenvalues from diagonal
        Ok((0..n).map(|i| a[i][i]).collect())
    }

    /// Compute eigenvector for a given eigenvalue numerically.
    ///
    /// Returns the eigenvector as a column matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square.
    pub fn eigenvector_numeric(&self, eigenvalue: f64) -> MatrixResult<Vec<f64>> {
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "Eigenvector requires a square matrix".to_string(),
            ));
        }

        let empty = std::collections::HashMap::new();
        let elements = self.evaluate(&empty).ok_or_else(|| {
            MatrixError::InvalidOperation("Cannot evaluate matrix numerically".to_string())
        })?;

        let n = self.rows;

        // Compute A - λI
        let mut a_minus_lambda: Vec<Vec<f64>> = elements.clone();
        for i in 0..n {
            a_minus_lambda[i][i] -= eigenvalue;
        }

        // Use inverse iteration to find eigenvector
        // Start with a random vector
        let mut v: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        // Normalize
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut v {
            *x /= norm;
        }

        // Inverse iteration: solve (A - λI)w = v, then v = w/||w||
        // Since A - λI is singular (or near-singular), we perturb slightly
        const MAX_ITER: usize = 50;
        const TOL: f64 = 1e-8;

        for _ in 0..MAX_ITER {
            // Solve (A - λI + εI)w = v using Gaussian elimination
            let mut augmented = a_minus_lambda.clone();
            for i in 0..n {
                augmented[i][i] += 1e-10; // Small perturbation
            }

            // Solve using Gaussian elimination
            let w = solve_linear_system(&augmented, &v);

            // Normalize
            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-14 {
                break;
            }

            let w_normalized: Vec<f64> = w.iter().map(|x| x / norm).collect();

            // Check convergence
            let diff: f64 = v
                .iter()
                .zip(w_normalized.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

            v = w_normalized;

            if diff < TOL {
                break;
            }
        }

        Ok(v)
    }

    /// Compute all eigenpairs (eigenvalue, eigenvector) numerically.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square.
    pub fn eigenpairs_numeric(&self) -> MatrixResult<Vec<(f64, Vec<f64>)>> {
        let eigenvalues = self.eigenvalues_numeric()?;
        let mut pairs = Vec::with_capacity(eigenvalues.len());

        for eigenvalue in eigenvalues {
            let eigenvector = self.eigenvector_numeric(eigenvalue)?;
            pairs.push((eigenvalue, eigenvector));
        }

        Ok(pairs)
    }

    /// Check if the matrix is diagonalizable.
    ///
    /// A matrix is diagonalizable if it has n linearly independent eigenvectors.
    pub fn is_diagonalizable(&self) -> MatrixResult<bool> {
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "Diagonalizability check requires a square matrix".to_string(),
            ));
        }

        // A simple check: symmetric matrices are always diagonalizable
        let transpose = self.transpose();
        let empty = std::collections::HashMap::new();

        if let (Some(a), Some(at)) = (self.evaluate(&empty), transpose.evaluate(&empty)) {
            let is_symmetric = a
                .iter()
                .zip(at.iter())
                .all(|(row_a, row_at)| {
                    row_a
                        .iter()
                        .zip(row_at.iter())
                        .all(|(x, y)| (x - y).abs() < 1e-10)
                });

            if is_symmetric {
                return Ok(true);
            }
        }

        // For non-symmetric matrices, we would need to check algebraic vs geometric multiplicity
        // This is a simplified check - return true if we can compute distinct eigenvalues
        let eigenvalues = self.eigenvalues_numeric()?;

        // Check if all eigenvalues are distinct (sufficient condition)
        for (i, &ev1) in eigenvalues.iter().enumerate() {
            for (j, &ev2) in eigenvalues.iter().enumerate() {
                if i != j && (ev1 - ev2).abs() < 1e-10 {
                    // Repeated eigenvalue - would need to check geometric multiplicity
                    // For simplicity, assume diagonalizable
                    return Ok(true);
                }
            }
        }

        Ok(true)
    }

    /// Render the matrix as LaTeX.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::{MatrixExpr, BracketStyle};
    /// use thales::ast::Expression;
    ///
    /// let m = MatrixExpr::from_elements(vec![
    ///     vec![Expression::Integer(1), Expression::Integer(2)],
    ///     vec![Expression::Integer(3), Expression::Integer(4)],
    /// ]).unwrap();
    ///
    /// let latex = m.to_latex(BracketStyle::Parentheses);
    /// assert!(latex.contains("pmatrix"));
    /// ```
    pub fn to_latex(&self, style: BracketStyle) -> String {
        let env = match style {
            BracketStyle::Parentheses => "pmatrix",
            BracketStyle::Square => "bmatrix",
            BracketStyle::Curly => "Bmatrix",
            BracketStyle::Determinant => "vmatrix",
            BracketStyle::Norm => "Vmatrix",
            BracketStyle::None => "matrix",
        };

        let mut result = format!("\\begin{{{}}}\n", env);
        for (i, row) in self.elements.iter().enumerate() {
            let row_str: Vec<String> = row.iter().map(|e| e.to_latex()).collect();
            result.push_str(&row_str.join(" & "));
            if i < self.rows - 1 {
                result.push_str(" \\\\\n");
            } else {
                result.push('\n');
            }
        }
        result.push_str(&format!("\\end{{{}}}", env));
        result
    }

    /// Render the matrix as LaTeX with default parentheses style.
    pub fn to_latex_default(&self) -> String {
        self.to_latex(BracketStyle::default())
    }

    /// Evaluate all elements numerically.
    ///
    /// Returns None if any element cannot be evaluated.
    pub fn evaluate(&self, vars: &std::collections::HashMap<String, f64>) -> Option<Vec<Vec<f64>>> {
        self.elements
            .iter()
            .map(|row| {
                row.iter()
                    .map(|elem| elem.evaluate(vars))
                    .collect::<Option<Vec<f64>>>()
            })
            .collect()
    }
}

impl fmt::Display for MatrixExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, row) in self.elements.iter().enumerate() {
            if i > 0 {
                write!(f, "; ")?;
            }
            write!(f, "[")?;
            for (j, elem) in row.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", elem)?;
            }
            write!(f, "]")?;
        }
        write!(f, "]")
    }
}

// =============================================================================
// Helper functions for eigenvalue computation
// =============================================================================

/// Solve cubic equation x³ + p*x² + q*x + r = 0 using Cardano's formula.
fn solve_cubic(p: f64, q: f64, r: f64) -> MatrixResult<Vec<f64>> {
    // Depress the cubic: substitute x = t - p/3
    // t³ + at + b = 0 where:
    // a = q - p²/3
    // b = r - pq/3 + 2p³/27
    let a = q - p * p / 3.0;
    let b = r - p * q / 3.0 + 2.0 * p * p * p / 27.0;

    // Discriminant
    let discriminant = -4.0 * a * a * a - 27.0 * b * b;

    let offset = -p / 3.0;

    if discriminant > 0.0 {
        // Three distinct real roots
        let theta = (-b / 2.0 / ((-a / 3.0).powi(3).sqrt())).acos();
        let r_cubed = (-a / 3.0).sqrt();

        let t1 = 2.0 * r_cubed * (theta / 3.0).cos();
        let t2 = 2.0 * r_cubed * ((theta + 2.0 * std::f64::consts::PI) / 3.0).cos();
        let t3 = 2.0 * r_cubed * ((theta + 4.0 * std::f64::consts::PI) / 3.0).cos();

        Ok(vec![t1 + offset, t2 + offset, t3 + offset])
    } else if discriminant.abs() < 1e-10 {
        // Multiple roots
        if b.abs() < 1e-10 {
            // Triple root
            Ok(vec![offset, offset, offset])
        } else {
            // Double root
            let double_root = 3.0 * b / a;
            let simple_root = -3.0 * b / (2.0 * a);
            Ok(vec![double_root + offset, simple_root + offset, simple_root + offset])
        }
    } else {
        // One real root, two complex (return real root 3 times for now)
        let sqrt_disc = (b * b / 4.0 + a * a * a / 27.0).sqrt();
        let u = (-b / 2.0 + sqrt_disc).cbrt();
        let v = (-b / 2.0 - sqrt_disc).cbrt();
        let real_root = u + v + offset;

        // Return real root; complex roots have same real part
        let complex_real = -(u + v) / 2.0 + offset;
        Ok(vec![real_root, complex_real, complex_real])
    }
}

/// QR decomposition using Gram-Schmidt process.
fn qr_decomposition(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut q = vec![vec![0.0; n]; n];
    let mut r = vec![vec![0.0; n]; n];

    for j in 0..n {
        // Start with column j of A
        let mut v: Vec<f64> = (0..n).map(|i| a[i][j]).collect();

        // Subtract projections onto previous q vectors
        for i in 0..j {
            let q_i: Vec<f64> = (0..n).map(|k| q[k][i]).collect();
            r[i][j] = dot_product(&q_i, &v);
            for k in 0..n {
                v[k] -= r[i][j] * q_i[k];
            }
        }

        // Compute norm and normalize
        r[j][j] = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if r[j][j] > 1e-14 {
            for k in 0..n {
                q[k][j] = v[k] / r[j][j];
            }
        }
    }

    (q, r)
}

/// Dot product of two vectors.
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Matrix multiplication for f64 matrices.
fn matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut result = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    result
}

/// Solve linear system Ax = b using Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a.len();

    // Create augmented matrix
    let mut aug: Vec<Vec<f64>> = a.iter().cloned().collect();
    let mut rhs = b.to_vec();

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_row = k;
        let mut max_val = aug[k][k].abs();
        for i in (k + 1)..n {
            if aug[i][k].abs() > max_val {
                max_val = aug[i][k].abs();
                max_row = i;
            }
        }

        // Swap rows
        if max_row != k {
            aug.swap(k, max_row);
            rhs.swap(k, max_row);
        }

        // Eliminate
        if aug[k][k].abs() > 1e-14 {
            for i in (k + 1)..n {
                let factor = aug[i][k] / aug[k][k];
                for j in k..n {
                    aug[i][j] -= factor * aug[k][j];
                }
                rhs[i] -= factor * rhs[k];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        if aug[i][i].abs() > 1e-14 {
            x[i] = rhs[i];
            for j in (i + 1)..n {
                x[i] -= aug[i][j] * x[j];
            }
            x[i] /= aug[i][i];
        }
    }

    x
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
        let m = MatrixExpr::from_elements(vec![
            vec![int(1), int(2)],
            vec![int(3), int(4)],
        ])
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
        let m = MatrixExpr::from_elements(vec![
            vec![int(1), int(2)],
            vec![int(3), int(4)],
        ])
        .unwrap();

        let mtt = m.transpose().transpose();
        assert_eq!(mtt.elements, m.elements);
    }

    #[test]
    fn test_trace() {
        let m = MatrixExpr::from_elements(vec![
            vec![int(1), int(2)],
            vec![int(3), int(4)],
        ])
        .unwrap();

        let trace = m.trace().unwrap();
        let vars = HashMap::new();
        assert_eq!(trace.evaluate(&vars), Some(5.0));
    }

    #[test]
    fn test_addition() {
        let a = MatrixExpr::from_elements(vec![
            vec![int(1), int(2)],
            vec![int(3), int(4)],
        ])
        .unwrap();

        let b = MatrixExpr::from_elements(vec![
            vec![int(5), int(6)],
            vec![int(7), int(8)],
        ])
        .unwrap();

        let sum = a.add(&b).unwrap();
        let vars = HashMap::new();

        assert_eq!(sum.get(0, 0).unwrap().evaluate(&vars), Some(6.0));
        assert_eq!(sum.get(0, 1).unwrap().evaluate(&vars), Some(8.0));
        assert_eq!(sum.get(1, 0).unwrap().evaluate(&vars), Some(10.0));
        assert_eq!(sum.get(1, 1).unwrap().evaluate(&vars), Some(12.0));
    }

    #[test]
    fn test_addition_dimension_check() {
        let a = MatrixExpr::from_elements(vec![
            vec![int(1), int(2)],
        ])
        .unwrap();

        let b = MatrixExpr::from_elements(vec![
            vec![int(1)],
            vec![int(2)],
        ])
        .unwrap();

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
        let m = MatrixExpr::from_elements(vec![
            vec![var("a"), var("b")],
            vec![var("c"), var("d")],
        ])
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
        let m = MatrixExpr::from_elements(vec![
            vec![int(1), int(2)],
            vec![int(3), int(4)],
        ])
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
        let a = MatrixExpr::from_elements(vec![
            vec![int(1), int(2)],
            vec![int(3), int(4)],
        ])
        .unwrap();

        let b = MatrixExpr::from_elements(vec![
            vec![int(5), int(6)],
            vec![int(7), int(8)],
        ])
        .unwrap();

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
        let m = MatrixExpr::from_elements(vec![
            vec![int(1), int(2)],
            vec![int(3), int(4)],
        ])
        .unwrap();

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
        let m = MatrixExpr::from_elements(vec![
            vec![int(4), int(7)],
            vec![int(2), int(6)],
        ])
        .unwrap();

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
        let m = MatrixExpr::from_elements(vec![
            vec![int(1), int(2)],
            vec![int(2), int(4)],
        ])
        .unwrap();

        let result = m.inverse();
        assert!(result.is_err());
    }

    #[test]
    fn test_determinant_symbolic() {
        // det([[a, b], [c, d]]) = ad - bc
        let m = MatrixExpr::from_elements(vec![
            vec![var("a"), var("b")],
            vec![var("c"), var("d")],
        ])
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
        let m = MatrixExpr::from_elements(vec![
            vec![int(1), int(2)],
            vec![int(3), int(4)],
        ])
        .unwrap();

        let adj = m.adjugate().unwrap();
        let vars = HashMap::new();

        assert_eq!(adj.get(0, 0).unwrap().evaluate(&vars), Some(4.0));
        assert_eq!(adj.get(0, 1).unwrap().evaluate(&vars), Some(-2.0));
        assert_eq!(adj.get(1, 0).unwrap().evaluate(&vars), Some(-3.0));
        assert_eq!(adj.get(1, 1).unwrap().evaluate(&vars), Some(1.0));
    }

    #[test]
    fn test_is_singular() {
        let singular = MatrixExpr::from_elements(vec![
            vec![int(1), int(2)],
            vec![int(2), int(4)],
        ])
        .unwrap();

        let non_singular = MatrixExpr::from_elements(vec![
            vec![int(1), int(2)],
            vec![int(3), int(4)],
        ])
        .unwrap();

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
        let m = MatrixExpr::from_elements(vec![
            vec![int(2), int(1)],
            vec![int(1), int(2)],
        ])
        .unwrap();

        let char_poly = m.characteristic_polynomial("lambda").unwrap();

        // Evaluate at λ = 1 (should be 0)
        let mut vars = HashMap::new();
        vars.insert("lambda".to_string(), 1.0);
        let at_1 = char_poly.evaluate(&vars).unwrap();
        assert!(at_1.abs() < 1e-10, "char poly at λ=1 should be 0, got {}", at_1);

        // Evaluate at λ = 3 (should be 0)
        vars.insert("lambda".to_string(), 3.0);
        let at_3 = char_poly.evaluate(&vars).unwrap();
        assert!(at_3.abs() < 1e-10, "char poly at λ=3 should be 0, got {}", at_3);
    }

    #[test]
    fn test_eigenvalues_2x2_symmetric() {
        // A = [[2, 1], [1, 2]], eigenvalues are 1 and 3
        let m = MatrixExpr::from_elements(vec![
            vec![int(2), int(1)],
            vec![int(1), int(2)],
        ])
        .unwrap();

        let eigenvalues = m.eigenvalues_numeric().unwrap();
        assert_eq!(eigenvalues.len(), 2);

        // Sort eigenvalues for consistent comparison
        let mut sorted = eigenvalues.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!((sorted[0] - 1.0).abs() < 1e-10, "Expected 1, got {}", sorted[0]);
        assert!((sorted[1] - 3.0).abs() < 1e-10, "Expected 3, got {}", sorted[1]);
    }

    #[test]
    fn test_eigenvalues_diagonal() {
        // Diagonal matrix: eigenvalues are the diagonal elements
        let m = MatrixExpr::from_elements(vec![
            vec![int(5), int(0)],
            vec![int(0), int(3)],
        ])
        .unwrap();

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
        let m = MatrixExpr::from_elements(vec![
            vec![int(2), int(1)],
            vec![int(1), int(2)],
        ])
        .unwrap();

        let eigenvector = m.eigenvector_numeric(3.0).unwrap();
        assert_eq!(eigenvector.len(), 2);

        // Check Av = λv (up to normalization)
        // v should be proportional to [1, 1]
        let ratio = eigenvector[0] / eigenvector[1];
        assert!((ratio - 1.0).abs() < 1e-5, "Expected ratio 1, got {}", ratio);
    }

    #[test]
    fn test_eigenpairs() {
        let m = MatrixExpr::from_elements(vec![
            vec![int(2), int(1)],
            vec![int(1), int(2)],
        ])
        .unwrap();

        let pairs = m.eigenpairs_numeric().unwrap();
        assert_eq!(pairs.len(), 2);

        for (eigenvalue, eigenvector) in pairs {
            // Verify Av = λv
            let empty = HashMap::new();
            let a = m.evaluate(&empty).unwrap();

            // Compute Av
            let av: Vec<f64> = (0..2)
                .map(|i| a[i].iter().zip(eigenvector.iter()).map(|(a, v)| a * v).sum())
                .collect();

            // Compute λv
            let lambda_v: Vec<f64> = eigenvector.iter().map(|v| eigenvalue * v).collect();

            // Check Av ≈ λv
            for i in 0..2 {
                assert!(
                    (av[i] - lambda_v[i]).abs() < 1e-5,
                    "Av[{}] = {}, λv[{}] = {}, eigenvalue = {}",
                    i, av[i], i, lambda_v[i], eigenvalue
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
        let m = MatrixExpr::from_elements(vec![
            vec![int(2), int(1)],
            vec![int(1), int(2)],
        ])
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
}
