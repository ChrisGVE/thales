//! QR decomposition via modified Gram-Schmidt for [`MatrixExpr`].
//!
//! Implements the modified Gram-Schmidt algorithm: A = QR where Q has
//! orthonormal columns and R is upper-triangular. Works on both numeric and
//! symbolic matrices. For symbolic matrices, norms remain as `sqrt(...)` expressions.

use std::sync::Arc;

use crate::numeric::{expr::Expr, normalize};

use super::{MatrixError, MatrixExpr, MatrixResult};

// ── private helpers ──────────────────────────────────────────────────────────

/// Compute the dot product of two equal-length vectors of `Arc<Expr>`.
fn inner_product(a: &[Arc<Expr>], b: &[Arc<Expr>]) -> Arc<Expr> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).fold(Expr::int(0), |acc, (ai, bi)| {
        normalize::add(acc, normalize::mul(ai.clone(), bi.clone()))
    })
}

/// Compute the Euclidean norm of a vector as `sqrt(sum of squares)`.
///
/// Returns the expression `(v[0]^2 + v[1]^2 + ...)^(1/2)`.
fn vector_norm(v: &[Arc<Expr>]) -> Arc<Expr> {
    let sum_sq = v.iter().fold(Expr::int(0), |acc, vi| {
        normalize::add(acc, normalize::mul(vi.clone(), vi.clone()))
    });
    let half = Expr::rational(1, 2);
    normalize::pow(sum_sq, half)
}

// ── public impl ──────────────────────────────────────────────────────────────

impl MatrixExpr {
    /// Compute the QR decomposition via modified Gram-Schmidt.
    ///
    /// Returns `(Q, R)` where:
    /// - `Q` is an `m × n` matrix with orthonormal columns,
    /// - `R` is an `n × n` upper-triangular matrix,
    /// - `Q * R == A` (up to symbolic simplification).
    ///
    /// The matrix must have at least as many rows as columns (`rows >= cols`).
    /// For symbolic matrices, norms are left as `expr^(1/2)` expressions.
    ///
    /// # Errors
    ///
    /// - `InvalidOperation` if `rows < cols`.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::numeric::expr::Expr;
    ///
    /// // 2×2 invertible matrix
    /// let a = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(1), Expr::int(1)],
    ///     vec![Expr::int(0), Expr::int(1)],
    /// ]).unwrap();
    ///
    /// let (q, r) = a.qr_decompose().unwrap();
    /// assert_eq!(q.rows(), 2);
    /// assert_eq!(q.cols(), 2);
    /// assert_eq!(r.rows(), 2);
    /// assert_eq!(r.cols(), 2);
    /// ```
    pub fn qr_decompose(&self) -> MatrixResult<(MatrixExpr, MatrixExpr)> {
        let m = self.rows;
        let n = self.cols;

        if m < n {
            return Err(MatrixError::InvalidOperation(format!(
                "QR decomposition requires rows >= cols, got {m}x{n}"
            )));
        }

        // Extract columns of A as Vec<Arc<Expr>> slices.
        let cols: Vec<Vec<Arc<Expr>>> = (0..n)
            .map(|j| (0..m).map(|i| self.elements[i][j].clone()).collect())
            .collect();

        // q_vecs[k] holds the k-th orthonormal column of Q.
        let mut q_vecs: Vec<Vec<Arc<Expr>>> = Vec::with_capacity(n);

        // r_vals[i][j] (i <= j) holds R[i][j]; below-diagonal entries are 0.
        let mut r_vals: Vec<Vec<Arc<Expr>>> = (0..n)
            .map(|_| (0..n).map(|_| Expr::int(0)).collect())
            .collect();

        for j in 0..n {
            // Start with the j-th original column.
            let mut v: Vec<Arc<Expr>> = cols[j].clone();

            // Subtract projections onto all previous q vectors.
            for (i, q_i) in q_vecs.iter().enumerate() {
                // R[i][j] = <q_i, a_j>  (inner product of q_i with original column j)
                let r_ij = inner_product(q_i, &cols[j]);
                r_vals[i][j] = r_ij.clone();

                // v -= R[i][j] * q_i
                for k in 0..m {
                    let sub_term = normalize::mul(r_ij.clone(), q_i[k].clone());
                    v[k] = normalize::sub(v[k].clone(), sub_term);
                }
            }

            // R[j][j] = norm of the residual vector v.
            let norm = vector_norm(&v);
            r_vals[j][j] = norm.clone();

            // q_j = v / norm(v)
            let q_j: Vec<Arc<Expr>> = v
                .into_iter()
                .map(|elem| normalize::div(elem, norm.clone()))
                .collect();

            q_vecs.push(q_j);
        }

        // Assemble Q (m × n): Q[i][j] = q_vecs[j][i]
        let q_elements: Vec<Vec<Arc<Expr>>> = (0..m)
            .map(|i| (0..n).map(|j| q_vecs[j][i].clone()).collect())
            .collect();
        let q = MatrixExpr::from_expr_elements_unchecked(m, n, q_elements);

        // Assemble R (n × n).
        let r = MatrixExpr::from_expr_elements_unchecked(n, n, r_vals);

        Ok((q, r))
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::numeric::SymbolId;

    use super::*;

    /// Evaluate every element of a MatrixExpr to f64, panicking on failure.
    fn to_f64(m: &MatrixExpr) -> Vec<Vec<f64>> {
        let empty: HashMap<SymbolId, f64> = HashMap::new();
        m.evaluate(&empty)
            .expect("matrix must be fully numeric for test evaluation")
    }

    /// Assert two f64 values are within 1e-10 of each other.
    fn approx_eq(a: f64, b: f64) {
        assert!(
            (a - b).abs() < 1e-10,
            "expected {b:.15}, got {a:.15}, diff = {:.3e}",
            (a - b).abs()
        );
    }

    /// Multiply two f64 matrices.
    fn mat_mul_f64(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let rows = a.len();
        let cols = b[0].len();
        let inner = b.len();
        let mut c = vec![vec![0.0_f64; cols]; rows];
        for i in 0..rows {
            for j in 0..cols {
                for k in 0..inner {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        c
    }

    /// Transpose a 2-D f64 matrix.
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
    fn fast_qr_2x2_numeric() {
        // A = [[1, 1], [0, 1]]
        let a = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(1)],
            vec![Expr::int(0), Expr::int(1)],
        ])
        .unwrap();

        let (q, r) = a.qr_decompose().unwrap();
        assert_eq!(q.rows(), 2);
        assert_eq!(q.cols(), 2);
        assert_eq!(r.rows(), 2);
        assert_eq!(r.cols(), 2);

        let q_num = to_f64(&q);
        let r_num = to_f64(&r);
        let a_num = to_f64(&a);

        // QR must reconstruct A.
        let qr = mat_mul_f64(&q_num, &r_num);
        for i in 0..2 {
            for j in 0..2 {
                approx_eq(qr[i][j], a_num[i][j]);
            }
        }

        // QᵀQ must be identity (Q has orthonormal columns).
        let qt = transpose_f64(&q_num);
        let qtq = mat_mul_f64(&qt, &q_num);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                approx_eq(qtq[i][j], expected);
            }
        }

        // R must be upper triangular: R[1][0] == 0.
        approx_eq(r_num[1][0], 0.0);
    }

    #[test]
    fn fast_qr_3x3_numeric() {
        // A = [[1, 2, 0], [0, 1, 1], [1, 0, 1]]
        let a = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(2), Expr::int(0)],
            vec![Expr::int(0), Expr::int(1), Expr::int(1)],
            vec![Expr::int(1), Expr::int(0), Expr::int(1)],
        ])
        .unwrap();

        let (q, r) = a.qr_decompose().unwrap();
        let q_num = to_f64(&q);
        let r_num = to_f64(&r);
        let a_num = to_f64(&a);

        // QR must reconstruct A.
        let qr = mat_mul_f64(&q_num, &r_num);
        for i in 0..3 {
            for j in 0..3 {
                approx_eq(qr[i][j], a_num[i][j]);
            }
        }

        // QᵀQ must be identity.
        let qt = transpose_f64(&q_num);
        let qtq = mat_mul_f64(&qt, &q_num);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                approx_eq(qtq[i][j], expected);
            }
        }

        // R must be upper triangular.
        for i in 0..3 {
            for j in 0..i {
                approx_eq(r_num[i][j], 0.0);
            }
        }
    }

    #[test]
    fn fast_qr_identity() {
        // QR of identity should give Q = I, R = I.
        let a = MatrixExpr::identity(3);
        let (q, r) = a.qr_decompose().unwrap();
        let q_num = to_f64(&q);
        let r_num = to_f64(&r);

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                approx_eq(q_num[i][j], expected);
                approx_eq(r_num[i][j], expected);
            }
        }
    }

    #[test]
    fn fast_qr_rectangular_3x2() {
        // A = [[1, 0], [1, 1], [0, 1]] — tall matrix
        let a = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(0)],
            vec![Expr::int(1), Expr::int(1)],
            vec![Expr::int(0), Expr::int(1)],
        ])
        .unwrap();

        let (q, r) = a.qr_decompose().unwrap();
        assert_eq!(q.rows(), 3);
        assert_eq!(q.cols(), 2);
        assert_eq!(r.rows(), 2);
        assert_eq!(r.cols(), 2);

        let q_num = to_f64(&q);
        let r_num = to_f64(&r);
        let a_num = to_f64(&a);

        // QR must reconstruct A.
        let qr = mat_mul_f64(&q_num, &r_num);
        for i in 0..3 {
            for j in 0..2 {
                approx_eq(qr[i][j], a_num[i][j]);
            }
        }

        // QᵀQ must be 2×2 identity.
        let qt = transpose_f64(&q_num);
        let qtq = mat_mul_f64(&qt, &q_num);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                approx_eq(qtq[i][j], expected);
            }
        }

        // R must be upper triangular: R[1][0] == 0.
        approx_eq(r_num[1][0], 0.0);
    }

    #[test]
    fn fast_qr_error_wide_matrix() {
        // cols > rows should return InvalidOperation.
        let a = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(2), Expr::int(3)],
            vec![Expr::int(4), Expr::int(5), Expr::int(6)],
        ])
        .unwrap();

        let err = a.qr_decompose().unwrap_err();
        assert!(
            matches!(err, MatrixError::InvalidOperation(_)),
            "expected InvalidOperation for wide matrix, got {err:?}"
        );
    }

    #[test]
    fn fast_qr_2x2_reconstruction_via_mul() {
        // Verify QR = A using the MatrixExpr::mul method for symbolic compatibility.
        let a = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(3), Expr::int(1)],
            vec![Expr::int(4), Expr::int(2)],
        ])
        .unwrap();

        let (q, r) = a.qr_decompose().unwrap();
        let qr = q.mul(&r).unwrap();

        let qr_num = to_f64(&qr);
        let a_num = to_f64(&a);

        for i in 0..2 {
            for j in 0..2 {
                approx_eq(qr_num[i][j], a_num[i][j]);
            }
        }
    }
}
