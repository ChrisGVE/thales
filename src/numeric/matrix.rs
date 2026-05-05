//! Exact matrix arithmetic over a field.
//!
//! [`Matrix<R>`] provides exact determinant, inverse, and row reduction
//! using field arithmetic (no floating-point rounding).

use super::ring::Field;
use std::fmt;

/// A matrix over a field `R`.
///
/// Stored in row-major order as a flat `Vec<R>`.
#[derive(Clone, Debug)]
pub struct Matrix<R: Field> {
    rows: usize,
    cols: usize,
    data: Vec<R>,
}

/// Error type for matrix operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatrixError {
    /// Matrix dimensions don't match for the operation.
    DimensionMismatch {
        /// Description of what was expected.
        expected: String,
        /// What was actually provided.
        got: String,
    },
    /// Matrix is singular (non-invertible).
    Singular,
    /// Matrix is not square.
    NotSquare,
    /// Empty matrix.
    Empty,
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            MatrixError::Singular => write!(f, "matrix is singular"),
            MatrixError::NotSquare => write!(f, "matrix is not square"),
            MatrixError::Empty => write!(f, "empty matrix"),
        }
    }
}

impl std::error::Error for MatrixError {}

impl<R: Field> Matrix<R> {
    /// Create a matrix from dimensions and row-major data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != rows * cols`.
    pub fn new(rows: usize, cols: usize, data: Vec<R>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "data length {} != rows*cols {}*{}",
            data.len(),
            rows,
            cols
        );
        Matrix { rows, cols, data }
    }

    /// Create from row vectors.
    pub fn from_rows(rows_data: Vec<Vec<R>>) -> Result<Self, MatrixError> {
        if rows_data.is_empty() {
            return Err(MatrixError::Empty);
        }
        let cols = rows_data[0].len();
        if cols == 0 {
            return Err(MatrixError::Empty);
        }
        for row in &rows_data {
            if row.len() != cols {
                return Err(MatrixError::DimensionMismatch {
                    expected: format!("{cols} columns"),
                    got: format!("{} columns", row.len()),
                });
            }
        }
        let rows = rows_data.len();
        let data: Vec<R> = rows_data.into_iter().flatten().collect();
        Ok(Matrix { rows, cols, data })
    }

    /// Identity matrix of size `n × n`.
    pub fn identity(n: usize) -> Self {
        let mut data = vec![R::zero(); n * n];
        for i in 0..n {
            data[i * n + i] = R::one();
        }
        Matrix {
            rows: n,
            cols: n,
            data,
        }
    }

    /// Zero matrix of given dimensions.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![R::zero(); rows * cols],
        }
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Whether this is a square matrix.
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Get element at (row, col). Panics if out of bounds.
    pub fn get(&self, row: usize, col: usize) -> &R {
        &self.data[row * self.cols + col]
    }

    /// Set element at (row, col). Panics if out of bounds.
    pub fn set(&mut self, row: usize, col: usize, val: R) {
        self.data[row * self.cols + col] = val;
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for j in 0..self.cols {
            for i in 0..self.rows {
                data.push(self.get(i, j).clone());
            }
        }
        Matrix {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }

    /// Matrix multiplication.
    pub fn mul(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.cols != other.rows {
            return Err(MatrixError::DimensionMismatch {
                expected: format!("{} columns", self.cols),
                got: format!("{} rows", other.rows),
            });
        }
        let mut data = vec![R::zero(); self.rows * other.cols];
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = R::zero();
                for k in 0..self.cols {
                    sum = sum + self.get(i, k).clone() * other.get(k, j).clone();
                }
                data[i * other.cols + j] = sum;
            }
        }
        Ok(Matrix {
            rows: self.rows,
            cols: other.cols,
            data,
        })
    }

    /// Add two matrices.
    pub fn add(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::DimensionMismatch {
                expected: format!("{}×{}", self.rows, self.cols),
                got: format!("{}×{}", other.rows, other.cols),
            });
        }
        let data: Vec<R> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a.clone() + b.clone())
            .collect();
        Ok(Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        })
    }

    /// Scale by a constant.
    pub fn scale(&self, c: &R) -> Self {
        let data: Vec<R> = self.data.iter().map(|x| x.clone() * c.clone()).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    /// Row echelon form via Gaussian elimination (exact).
    ///
    /// Returns the row echelon form and the number of row swaps
    /// (needed for determinant sign).
    pub fn row_echelon(&self) -> (Self, usize) {
        let mut m = self.clone();
        let mut swaps = 0;
        let mut pivot_row = 0;

        for col in 0..m.cols.min(m.rows) {
            // Find pivot
            let mut found = None;
            for row in pivot_row..m.rows {
                if !m.get(row, col).is_zero() {
                    found = Some(row);
                    break;
                }
            }
            let Some(pivot) = found else { continue };

            // Swap rows
            if pivot != pivot_row {
                for c in 0..m.cols {
                    let idx1 = pivot_row * m.cols + c;
                    let idx2 = pivot * m.cols + c;
                    m.data.swap(idx1, idx2);
                }
                swaps += 1;
            }

            // Eliminate below
            let pivot_val = m.get(pivot_row, col).clone();
            for row in (pivot_row + 1)..m.rows {
                let factor = m.get(row, col).clone() * pivot_val.clone().inv();
                for c in col..m.cols {
                    let val = m.get(row, c).clone() - factor.clone() * m.get(pivot_row, c).clone();
                    m.set(row, c, val);
                }
            }

            pivot_row += 1;
        }

        (m, swaps)
    }

    /// Reduced row echelon form (RREF).
    pub fn rref(&self) -> Self {
        let (mut m, _) = self.row_echelon();
        let rows = m.rows;
        let cols = m.cols;

        // Find pivots
        let mut pivots = Vec::new();
        let mut col = 0;
        for row in 0..rows {
            while col < cols && m.get(row, col).is_zero() {
                col += 1;
            }
            if col < cols {
                pivots.push((row, col));
                col += 1;
            }
        }

        // Back-substitute and normalize
        for &(row, col) in pivots.iter().rev() {
            // Normalize pivot row
            let pivot_val = m.get(row, col).clone();
            if !pivot_val.is_zero() {
                let inv = pivot_val.inv();
                for c in 0..cols {
                    let val = m.get(row, c).clone() * inv.clone();
                    m.set(row, c, val);
                }
            }

            // Eliminate above
            for r in 0..row {
                let factor = m.get(r, col).clone();
                if !factor.is_zero() {
                    for c in 0..cols {
                        let val = m.get(r, c).clone() - factor.clone() * m.get(row, c).clone();
                        m.set(r, c, val);
                    }
                }
            }
        }

        m
    }

    /// Determinant (exact, via row echelon form). Matrix must be square.
    pub fn determinant(&self) -> Result<R, MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::NotSquare);
        }
        let n = self.rows;
        if n == 0 {
            return Ok(R::one());
        }

        let (echelon, swaps) = self.row_echelon();

        // Product of diagonal elements
        let mut det = R::one();
        for i in 0..n {
            det = det * echelon.get(i, i).clone();
        }

        // Adjust sign for row swaps
        if swaps % 2 == 1 {
            det = -det;
        }

        Ok(det)
    }

    /// Inverse (exact). Matrix must be square and non-singular.
    pub fn inverse(&self) -> Result<Self, MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::NotSquare);
        }
        let n = self.rows;

        // Augment with identity: [A | I]
        let mut aug_data = vec![R::zero(); n * 2 * n];
        for i in 0..n {
            for j in 0..n {
                aug_data[i * 2 * n + j] = self.get(i, j).clone();
            }
            aug_data[i * 2 * n + n + i] = R::one();
        }
        let aug = Matrix {
            rows: n,
            cols: 2 * n,
            data: aug_data,
        };

        let rref = aug.rref();

        // Check that left half is identity
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { R::one() } else { R::zero() };
                if rref.get(i, j) != &expected {
                    return Err(MatrixError::Singular);
                }
            }
        }

        // Extract right half
        let mut data = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                data.push(rref.get(i, n + j).clone());
            }
        }

        Ok(Matrix {
            rows: n,
            cols: n,
            data,
        })
    }

    /// Rank of the matrix.
    pub fn rank(&self) -> usize {
        let rref = self.rref();
        let mut rank = 0;
        for i in 0..rref.rows {
            let row_nonzero = (0..rref.cols).any(|j| !rref.get(i, j).is_zero());
            if row_nonzero {
                rank += 1;
            }
        }
        rank
    }
}

impl<R: Field> PartialEq for Matrix<R> {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.data == other.data
    }
}

impl<R: Field> Eq for Matrix<R> {}

impl<R: Field> fmt::Display for Matrix<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            write!(f, "[")?;
            for j in 0..self.cols {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.get(i, j))?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::BigRational;

    type M = Matrix<BigRational>;

    fn int(n: i64) -> BigRational {
        BigRational::from(n)
    }

    fn rat(n: i64, d: i64) -> BigRational {
        BigRational::from_i64(n, d)
    }

    fn mat2x2(a: i64, b: i64, c: i64, d: i64) -> M {
        M::new(2, 2, vec![int(a), int(b), int(c), int(d)])
    }

    #[test]
    fn test_identity() {
        let id = M::identity(3);
        assert_eq!(id.rows(), 3);
        assert_eq!(id.cols(), 3);
        assert_eq!(*id.get(0, 0), int(1));
        assert_eq!(*id.get(0, 1), int(0));
        assert_eq!(*id.get(1, 1), int(1));
    }

    #[test]
    fn test_transpose() {
        let m = M::from_rows(vec![
            vec![int(1), int(2), int(3)],
            vec![int(4), int(5), int(6)],
        ])
        .unwrap();
        let t = m.transpose();
        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 2);
        assert_eq!(*t.get(0, 0), int(1));
        assert_eq!(*t.get(0, 1), int(4));
        assert_eq!(*t.get(2, 0), int(3));
    }

    #[test]
    fn test_mul() {
        let a = mat2x2(1, 2, 3, 4);
        let b = mat2x2(5, 6, 7, 8);
        let c = a.mul(&b).unwrap();
        // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert_eq!(*c.get(0, 0), int(19));
        assert_eq!(*c.get(0, 1), int(22));
        assert_eq!(*c.get(1, 0), int(43));
        assert_eq!(*c.get(1, 1), int(50));
    }

    #[test]
    fn test_determinant_2x2() {
        let m = mat2x2(1, 2, 3, 4);
        // det = 1*4 - 2*3 = -2
        assert_eq!(m.determinant().unwrap(), int(-2));
    }

    #[test]
    fn test_determinant_3x3() {
        let m = M::new(
            3,
            3,
            vec![
                int(1),
                int(2),
                int(3),
                int(4),
                int(5),
                int(6),
                int(7),
                int(8),
                int(0),
            ],
        );
        // det = 1(5*0-6*8) - 2(4*0-6*7) + 3(4*8-5*7) = -48+84-9 = 27
        assert_eq!(m.determinant().unwrap(), int(27));
    }

    #[test]
    fn test_determinant_singular() {
        let m = mat2x2(1, 2, 2, 4);
        assert_eq!(m.determinant().unwrap(), int(0));
    }

    #[test]
    fn test_inverse_2x2() {
        let m = mat2x2(1, 2, 3, 4);
        let inv = m.inverse().unwrap();
        // inv of [[1,2],[3,4]] = 1/(-2) * [[4,-2],[-3,1]] = [[-2, 1],[3/2, -1/2]]
        assert_eq!(*inv.get(0, 0), int(-2));
        assert_eq!(*inv.get(0, 1), int(1));
        assert_eq!(*inv.get(1, 0), rat(3, 2));
        assert_eq!(*inv.get(1, 1), rat(-1, 2));

        // Verify: M * M^(-1) = I
        let product = m.mul(&inv).unwrap();
        assert_eq!(product, M::identity(2));
    }

    #[test]
    fn test_inverse_singular() {
        let m = mat2x2(1, 2, 2, 4);
        assert_eq!(m.inverse(), Err(MatrixError::Singular));
    }

    #[test]
    fn test_inverse_rational() {
        // Matrix with rational entries
        let m = M::new(2, 2, vec![rat(1, 2), rat(1, 3), rat(1, 4), rat(1, 5)]);
        let inv = m.inverse().unwrap();
        let product = m.mul(&inv).unwrap();
        assert_eq!(product, M::identity(2));
    }

    #[test]
    fn test_rref() {
        let m = M::new(2, 3, vec![int(1), int(2), int(3), int(4), int(5), int(6)]);
        let rref = m.rref();
        assert_eq!(*rref.get(0, 0), int(1));
        assert_eq!(*rref.get(0, 1), int(0));
        assert_eq!(*rref.get(1, 0), int(0));
        assert_eq!(*rref.get(1, 1), int(1));
    }

    #[test]
    fn test_rank() {
        let m = mat2x2(1, 2, 3, 4);
        assert_eq!(m.rank(), 2);

        let singular = mat2x2(1, 2, 2, 4);
        assert_eq!(singular.rank(), 1);
    }

    #[test]
    fn test_add() {
        let a = mat2x2(1, 2, 3, 4);
        let b = mat2x2(5, 6, 7, 8);
        let c = a.add(&b).unwrap();
        assert_eq!(*c.get(0, 0), int(6));
        assert_eq!(*c.get(1, 1), int(12));
    }

    #[test]
    fn test_scale() {
        let m = mat2x2(1, 2, 3, 4);
        let scaled = m.scale(&int(3));
        assert_eq!(*scaled.get(0, 0), int(3));
        assert_eq!(*scaled.get(1, 1), int(12));
    }

    #[test]
    fn test_mul_identity() {
        let m = mat2x2(1, 2, 3, 4);
        let id = M::identity(2);
        assert_eq!(m.mul(&id).unwrap(), m);
        assert_eq!(id.mul(&m).unwrap(), m);
    }

    #[test]
    fn test_not_square_determinant() {
        let m = M::new(2, 3, vec![int(1), int(2), int(3), int(4), int(5), int(6)]);
        assert_eq!(m.determinant(), Err(MatrixError::NotSquare));
    }
}
