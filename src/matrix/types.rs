//! Error and configuration types for matrix operations.

use std::fmt;

/// Error type for matrix operations.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum MatrixError {
    /// Dimension mismatch for operation.
    DimensionMismatch {
        /// Name of the operation that encountered the mismatch.
        operation: String,
        /// The (rows, cols) dimensions that were expected.
        expected: (usize, usize),
        /// The (rows, cols) dimensions that were actually provided.
        got: (usize, usize),
    },
    /// Empty matrix or row not allowed.
    EmptyMatrix,
    /// Non-rectangular matrix (rows have different lengths).
    NonRectangular,
    /// Index out of bounds.
    IndexOutOfBounds {
        /// Row index that was accessed.
        row: usize,
        /// Column index that was accessed.
        col: usize,
        /// Total number of rows in the matrix.
        rows: usize,
        /// Total number of columns in the matrix.
        cols: usize,
    },
    /// Cannot compute operation (e.g., determinant of non-square matrix).
    InvalidOperation(String),
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixError::DimensionMismatch {
                operation,
                expected,
                got,
            } => {
                write!(
                    f,
                    "{}: expected {}x{}, got {}x{}",
                    operation, expected.0, expected.1, got.0, got.1
                )
            }
            MatrixError::EmptyMatrix => write!(f, "Empty matrix not allowed"),
            MatrixError::NonRectangular => {
                write!(f, "Matrix must be rectangular (all rows same length)")
            }
            MatrixError::IndexOutOfBounds {
                row,
                col,
                rows,
                cols,
            } => {
                write!(
                    f,
                    "Index ({}, {}) out of bounds for {}x{} matrix",
                    row, col, rows, cols
                )
            }
            MatrixError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
        }
    }
}

impl std::error::Error for MatrixError {}

/// Result type for matrix operations.
pub type MatrixResult<T> = Result<T, MatrixError>;

/// Bracket style for LaTeX output.
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
