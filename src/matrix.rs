//! Matrix expression type with basic linear algebra operations.
//!
//! This module provides a symbolic matrix type where elements are mathematical expressions,
//! supporting operations like addition, multiplication, transpose, and trace with symbolic
//! manipulation capabilities.
//!
//! # Examples
//!
//! ```
//! use mathsolver_core::matrix::MatrixExpr;
//! use mathsolver_core::ast::Expression;
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

use crate::ast::Expression;
use std::fmt;

/// Error type for matrix operations.
#[derive(Debug, Clone, PartialEq)]
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
/// use mathsolver_core::matrix::MatrixExpr;
/// use mathsolver_core::ast::{Expression, Variable};
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
    /// use mathsolver_core::matrix::MatrixExpr;
    /// use mathsolver_core::ast::Expression;
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
    /// use mathsolver_core::matrix::MatrixExpr;
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
    /// use mathsolver_core::matrix::MatrixExpr;
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
    /// use mathsolver_core::matrix::MatrixExpr;
    /// use mathsolver_core::ast::Expression;
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
    /// use mathsolver_core::matrix::MatrixExpr;
    /// use mathsolver_core::ast::Expression;
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
    /// use mathsolver_core::matrix::MatrixExpr;
    /// use mathsolver_core::ast::Expression;
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
    /// use mathsolver_core::matrix::MatrixExpr;
    /// use mathsolver_core::ast::Expression;
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
    /// use mathsolver_core::matrix::MatrixExpr;
    /// use mathsolver_core::ast::Expression;
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
    /// use mathsolver_core::matrix::MatrixExpr;
    /// use mathsolver_core::ast::Expression;
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

    /// Render the matrix as LaTeX.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::matrix::{MatrixExpr, BracketStyle};
    /// use mathsolver_core::ast::Expression;
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
}
