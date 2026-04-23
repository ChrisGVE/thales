//! Matrix arithmetic operations and linear algebra.

use std::sync::Arc;

use crate::numeric::evaluation::evaluate;
use crate::numeric::expr::Expr;
use crate::numeric::normalize;
use crate::numeric::SymbolId;

use super::{MatrixError, MatrixExpr, MatrixResult};

impl MatrixExpr {
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
    /// use thales::numeric::expr::Expr;
    /// use thales::numeric::evaluation::evaluate;
    /// use thales::numeric::SymbolId;
    /// use std::collections::HashMap;
    ///
    /// let m = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(1), Expr::int(2)],
    ///     vec![Expr::int(3), Expr::int(4)],
    /// ]).unwrap();
    ///
    /// let trace = m.trace().unwrap();
    /// // trace = 1 + 4 = 5
    /// let empty: HashMap<SymbolId, f64> = HashMap::new();
    /// assert_eq!(evaluate(&trace, &empty), Some(5.0));
    /// ```
    pub fn trace(&self) -> MatrixResult<Arc<Expr>> {
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "Trace requires a square matrix".to_string(),
            ));
        }

        let mut trace = self.elements[0][0].clone();
        for i in 1..self.rows {
            trace = normalize::add(trace, self.elements[i][i].clone());
        }
        Ok(trace)
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
    /// use thales::numeric::expr::Expr;
    ///
    /// let a = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(1), Expr::int(2)],
    ///     vec![Expr::int(3), Expr::int(4)],
    /// ]).unwrap();
    ///
    /// let b = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(5), Expr::int(6)],
    ///     vec![Expr::int(7), Expr::int(8)],
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

        let elements: Vec<Vec<Arc<Expr>>> = (0..self.rows)
            .map(|i| {
                (0..self.cols)
                    .map(|j| {
                        normalize::add(self.elements[i][j].clone(), other.elements[i][j].clone())
                    })
                    .collect()
            })
            .collect();

        Ok(MatrixExpr::from_expr_elements_unchecked(
            self.rows, self.cols, elements,
        ))
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

        let elements: Vec<Vec<Arc<Expr>>> = (0..self.rows)
            .map(|i| {
                (0..self.cols)
                    .map(|j| {
                        normalize::sub(self.elements[i][j].clone(), other.elements[i][j].clone())
                    })
                    .collect()
            })
            .collect();

        Ok(MatrixExpr::from_expr_elements_unchecked(
            self.rows, self.cols, elements,
        ))
    }

    /// Multiply by a scalar `Arc<Expr>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::numeric::expr::Expr;
    ///
    /// let m = MatrixExpr::identity(2);
    /// let scaled = m.scalar_mul(&Expr::int(3));
    /// ```
    pub fn scalar_mul(&self, scalar: &Arc<Expr>) -> MatrixExpr {
        let elements: Vec<Vec<Arc<Expr>>> = self
            .elements
            .iter()
            .map(|row| {
                row.iter()
                    .map(|elem| normalize::mul(scalar.clone(), elem.clone()))
                    .collect()
            })
            .collect();

        MatrixExpr::from_expr_elements_unchecked(self.rows, self.cols, elements)
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
    /// use thales::numeric::expr::Expr;
    ///
    /// // 2x3 matrix
    /// let a = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(1), Expr::int(2), Expr::int(3)],
    ///     vec![Expr::int(4), Expr::int(5), Expr::int(6)],
    /// ]).unwrap();
    ///
    /// // 3x2 matrix
    /// let b = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(7), Expr::int(8)],
    ///     vec![Expr::int(9), Expr::int(10)],
    ///     vec![Expr::int(11), Expr::int(12)],
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

        let elements: Vec<Vec<Arc<Expr>>> = (0..self.rows)
            .map(|i| {
                (0..other.cols)
                    .map(|j| {
                        // C[i][j] = sum(A[i][k] * B[k][j] for k in 0..n)
                        let mut sum = normalize::mul(
                            self.elements[i][0].clone(),
                            other.elements[0][j].clone(),
                        );
                        for k in 1..self.cols {
                            let product = normalize::mul(
                                self.elements[i][k].clone(),
                                other.elements[k][j].clone(),
                            );
                            sum = normalize::add(sum, product);
                        }
                        sum
                    })
                    .collect()
            })
            .collect();

        Ok(MatrixExpr::from_expr_elements_unchecked(
            self.rows, other.cols, elements,
        ))
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

        let elements: Vec<Vec<Arc<Expr>>> = self
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

        MatrixExpr::from_expr_elements(elements)
    }

    /// Compute the minor M(i, j) — the determinant of the submatrix excluding row i and column j.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square or is 1x1.
    pub fn minor(&self, row: usize, col: usize) -> MatrixResult<Arc<Expr>> {
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
    pub fn cofactor(&self, row: usize, col: usize) -> MatrixResult<Arc<Expr>> {
        let minor = self.minor(row, col)?;
        if (row + col) % 2 == 0 {
            Ok(minor)
        } else {
            Ok(normalize::neg(minor))
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
    /// use thales::numeric::expr::Expr;
    /// use thales::numeric::evaluation::evaluate;
    /// use thales::numeric::SymbolId;
    /// use std::collections::HashMap;
    ///
    /// // 2x2 matrix: [[1, 2], [3, 4]]
    /// let m = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(1), Expr::int(2)],
    ///     vec![Expr::int(3), Expr::int(4)],
    /// ]).unwrap();
    ///
    /// let det = m.determinant().unwrap();
    /// // det = 1*4 - 2*3 = -2
    /// let empty: HashMap<SymbolId, f64> = HashMap::new();
    /// assert_eq!(evaluate(&det, &empty), Some(-2.0));
    /// ```
    pub fn determinant(&self) -> MatrixResult<Arc<Expr>> {
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

                let ad = normalize::mul(a.clone(), d.clone());
                let bc = normalize::mul(b.clone(), c.clone());
                Ok(normalize::sub(ad, bc))
            }
            _ => {
                // Cofactor expansion along first row
                let mut det = Expr::int(0);
                for j in 0..self.cols {
                    let cofactor = self.cofactor(0, j)?;
                    let term = normalize::mul(self.elements[0][j].clone(), cofactor);
                    det = normalize::add(det, term);
                }
                Ok(det)
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
    /// use thales::numeric::expr::Expr;
    /// use thales::numeric::evaluation::evaluate;
    /// use thales::numeric::SymbolId;
    /// use std::collections::HashMap;
    ///
    /// let m = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(2), Expr::int(1)],
    ///     vec![Expr::int(1), Expr::int(2)],
    /// ]).unwrap();
    ///
    /// let char_poly = m.characteristic_polynomial("lambda").unwrap();
    /// // For this matrix, eigenvalues are 1 and 3
    /// // So char poly = (λ - 1)(λ - 3) = λ² - 4λ + 3
    /// ```
    pub fn characteristic_polynomial(&self, lambda_var: &str) -> MatrixResult<Arc<Expr>> {
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "Characteristic polynomial requires a square matrix".to_string(),
            ));
        }

        // Compute A - λI
        let lambda = Expr::symbol(lambda_var);
        let lambda_i = MatrixExpr::identity(self.rows).scalar_mul(&lambda);
        let a_minus_lambda_i = self.sub(&lambda_i)?;

        // Compute det(A - λI)
        a_minus_lambda_i.determinant()
    }

    /// Evaluate all elements numerically.
    ///
    /// Returns None if any element cannot be evaluated.
    pub fn evaluate(
        &self,
        vars: &std::collections::HashMap<SymbolId, f64>,
    ) -> Option<Vec<Vec<f64>>> {
        self.elements
            .iter()
            .map(|row| {
                row.iter()
                    .map(|elem| evaluate(elem, vars))
                    .collect::<Option<Vec<f64>>>()
            })
            .collect()
    }
}
