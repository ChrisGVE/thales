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
