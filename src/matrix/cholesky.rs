//! Cholesky decomposition (LLᵀ) for [`MatrixExpr`].
//!
//! Implements the Banachiewicz algorithm: A = LLᵀ where L is lower-triangular.
//! Works on both numeric and symbolic matrices. For symbolic matrices the
//! diagonal elements are left as `sqrt(expr)` (i.e. `expr^(1/2)`); for numeric
//! matrices a non-positive diagonal under the square-root produces an error.

use std::sync::Arc;

use crate::numeric::{expr::Expr, normalize};

use super::{MatrixError, MatrixExpr, MatrixResult};

// ── private helpers ──────────────────────────────────────────────────────────

/// Try to extract an `f64` from an `Arc<Expr>`.
///
/// Returns `Some(v)` for `Expr::Integer`, `Expr::Rational`, and `Expr::Float`;
/// `None` for symbolic nodes.
fn as_f64(e: &Arc<Expr>) -> Option<f64> {
    use crate::numeric::evaluation::evaluate;
    let empty = std::collections::HashMap::new();
    evaluate(e, &empty)
}

/// Return `true` when the expression is structurally numeric (no symbols).
fn is_numeric(e: &Arc<Expr>) -> bool {
    as_f64(e).is_some()
}

// ── public impl ──────────────────────────────────────────────────────────────

impl MatrixExpr {
    /// Compute the Cholesky decomposition A = LLᵀ.
    ///
    /// Returns the lower-triangular factor `L` such that `L * Lᵀ == A`.
    ///
    /// For fully numeric matrices every element must be a concrete number; the
    /// diagonal entry under the square-root must be strictly positive, otherwise
    /// the matrix is not positive-definite and an error is returned.
    ///
    /// For symbolic matrices the diagonal pivots are expressed as `expr^(1/2)`
    /// and the computation proceeds in closed form.
    ///
    /// # Errors
    ///
    /// - `InvalidOperation` if the matrix is not square.
    /// - `InvalidOperation` if the matrix is not symmetric (structural check
    ///   after normalization).
    /// - `InvalidOperation` if a numeric diagonal pivot is ≤ 0 (matrix is not
    ///   positive definite).
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::numeric::expr::Expr;
    ///
    /// // 2×2 positive-definite matrix:  [[4, 2], [2, 3]]
    /// let a = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(4), Expr::int(2)],
    ///     vec![Expr::int(2), Expr::int(3)],
    /// ]).unwrap();
    ///
    /// let l = a.cholesky().unwrap();
    /// assert_eq!(l.rows(), 2);
    /// assert_eq!(l.cols(), 2);
    /// // Upper triangle must be zero.
    /// assert!(l.get(0, 1).unwrap().is_zero());
    /// ```
    pub fn cholesky(&self) -> MatrixResult<MatrixExpr> {
        // ── precondition: square ─────────────────────────────────────────────
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "Cholesky decomposition requires a square matrix".to_string(),
            ));
        }

        let n = self.rows;

        // ── precondition: symmetric ──────────────────────────────────────────
        for i in 0..n {
            for j in (i + 1)..n {
                let a_ij = &self.elements[i][j];
                let a_ji = &self.elements[j][i];
                // Structural equality after normalizing the difference.
                let diff = normalize::sub(a_ij.clone(), a_ji.clone());
                if !diff.is_zero() {
                    return Err(MatrixError::InvalidOperation(format!(
                        "Cholesky decomposition requires a symmetric matrix \
                         (elements [{i},{j}] and [{j},{i}] differ)"
                    )));
                }
            }
        }

        // ── Banachiewicz algorithm ────────────────────────────────────────────
        // Build L as a mutable grid; initialise to zero.
        let mut l: Vec<Vec<Arc<Expr>>> = (0..n)
            .map(|_| (0..n).map(|_| Expr::int(0)).collect())
            .collect();

        for j in 0..n {
            // Diagonal element: L[j][j] = sqrt(A[j][j] - sum_{k<j} L[j][k]²)
            let mut diag = self.elements[j][j].clone();
            for k in 0..j {
                let ljk_sq = normalize::mul(l[j][k].clone(), l[j][k].clone());
                diag = normalize::sub(diag, ljk_sq);
            }

            // For numeric matrices: must be strictly positive.
            if is_numeric(&diag) {
                match as_f64(&diag) {
                    Some(v) if v <= 0.0 => {
                        return Err(MatrixError::InvalidOperation(format!(
                            "Cholesky decomposition failed: diagonal pivot at \
                             column {j} is {v:.6} (matrix is not positive definite)"
                        )));
                    }
                    _ => {}
                }
            }

            // L[j][j] = sqrt(diag) = diag^(1/2)
            let half = Expr::rational(1, 2);
            l[j][j] = Expr::pow(diag.clone(), half);

            // Sub-diagonal entries: L[i][j] = (A[i][j] - sum_{k<j} L[i][k]*L[j][k]) / L[j][j]
            for i in (j + 1)..n {
                let mut val = self.elements[i][j].clone();
                for k in 0..j {
                    let prod = normalize::mul(l[i][k].clone(), l[j][k].clone());
                    val = normalize::sub(val, prod);
                }
                l[i][j] = normalize::div(val, l[j][j].clone());
            }
        }

        Ok(MatrixExpr::from_expr_elements_unchecked(n, n, l))
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::numeric::{evaluation::evaluate, SymbolId};

    use super::*;

    /// Evaluate a MatrixExpr to a Vec<Vec<f64>>, panicking on symbolic entries.
    fn to_f64(m: &MatrixExpr) -> Vec<Vec<f64>> {
        let empty: HashMap<SymbolId, f64> = HashMap::new();
        m.evaluate(&empty)
            .expect("matrix must be fully numeric for test evaluation")
    }

    /// Assert two f64 values are close to machine precision.
    fn approx_eq(a: f64, b: f64) {
        assert!(
            (a - b).abs() < 1e-10,
            "expected {b:.15}, got {a:.15}, diff = {:.3e}",
            (a - b).abs()
        );
    }

    /// Multiply two numeric matrices and return the result as Vec<Vec<f64>>.
    fn mat_mul_f64(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = a.len();
        let m = b[0].len();
        let k = b.len();
        let mut c = vec![vec![0.0_f64; m]; n];
        for i in 0..n {
            for j in 0..m {
                for p in 0..k {
                    c[i][j] += a[i][p] * b[p][j];
                }
            }
        }
        c
    }

    /// Transpose a 2D f64 matrix.
    fn transpose_f64(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = a.len();
        let m = a[0].len();
        let mut t = vec![vec![0.0_f64; n]; m];
        for i in 0..n {
            for j in 0..m {
                t[j][i] = a[i][j];
            }
        }
        t
    }

    #[test]
    fn fast_cholesky_non_square_error() {
        let a = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(2), Expr::int(3)],
            vec![Expr::int(4), Expr::int(5), Expr::int(6)],
        ])
        .unwrap();
        let err = a.cholesky().unwrap_err();
        assert!(
            matches!(err, MatrixError::InvalidOperation(_)),
            "expected InvalidOperation, got {err:?}"
        );
    }

    #[test]
    fn fast_cholesky_non_symmetric_error() {
        // [[1, 2], [3, 1]] is not symmetric
        let a = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(2)],
            vec![Expr::int(3), Expr::int(1)],
        ])
        .unwrap();
        let err = a.cholesky().unwrap_err();
        assert!(
            matches!(err, MatrixError::InvalidOperation(_)),
            "expected InvalidOperation, got {err:?}"
        );
    }

    #[test]
    fn fast_cholesky_2x2_lower_triangular() {
        // A = [[4, 2], [2, 3]] → L = [[2, 0], [1, sqrt(2)]]
        let a = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(4), Expr::int(2)],
            vec![Expr::int(2), Expr::int(3)],
        ])
        .unwrap();

        let l = a.cholesky().unwrap();
        assert_eq!(l.rows(), 2);
        assert_eq!(l.cols(), 2);

        // Upper triangle must be zero.
        assert!(
            l.get(0, 1).unwrap().is_zero(),
            "L[0][1] must be zero (upper triangle)"
        );
    }

    #[test]
    fn fast_cholesky_2x2_reconstruct() {
        // A = [[4, 2], [2, 3]]
        // Expected L = [[2, 0], [1, sqrt(2)]]
        // Verify L * Lᵀ == A numerically.
        let a = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(4), Expr::int(2)],
            vec![Expr::int(2), Expr::int(3)],
        ])
        .unwrap();

        let l = a.cholesky().unwrap();
        let l_num = to_f64(&l);
        let lt_num = transpose_f64(&l_num);
        let reconstructed = mat_mul_f64(&l_num, &lt_num);
        let a_num = to_f64(&a);

        for i in 0..2 {
            for j in 0..2 {
                approx_eq(reconstructed[i][j], a_num[i][j]);
            }
        }
    }

    #[test]
    fn fast_cholesky_3x3_reconstruct() {
        // A = [[4, 2, 2], [2, 3, 1], [2, 1, 3]] — positive-definite symmetric
        let a = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(4), Expr::int(2), Expr::int(2)],
            vec![Expr::int(2), Expr::int(3), Expr::int(1)],
            vec![Expr::int(2), Expr::int(1), Expr::int(3)],
        ])
        .unwrap();

        let l = a.cholesky().unwrap();
        assert_eq!(l.rows(), 3);
        assert_eq!(l.cols(), 3);

        // Upper triangle must be zero.
        let empty: HashMap<SymbolId, f64> = HashMap::new();
        for i in 0..3 {
            for j in (i + 1)..3 {
                let val = evaluate(l.get(i, j).unwrap(), &empty).unwrap_or(1.0);
                approx_eq(val, 0.0);
            }
        }

        // L * Lᵀ must recover A.
        let l_num = to_f64(&l);
        let lt_num = transpose_f64(&l_num);
        let reconstructed = mat_mul_f64(&l_num, &lt_num);
        let a_num = to_f64(&a);

        for i in 0..3 {
            for j in 0..3 {
                approx_eq(reconstructed[i][j], a_num[i][j]);
            }
        }
    }

    #[test]
    fn fast_cholesky_not_positive_definite_error() {
        // A = [[1, 0], [0, -1]] — not positive definite
        let a = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(0)],
            vec![Expr::int(0), Expr::int(-1)],
        ])
        .unwrap();
        let err = a.cholesky().unwrap_err();
        assert!(
            matches!(err, MatrixError::InvalidOperation(_)),
            "expected InvalidOperation for non-positive-definite matrix, got {err:?}"
        );
    }
}
