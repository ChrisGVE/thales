//! Singular Value Decomposition (SVD) for [`MatrixExpr`].
//!
//! Implements a one-sided Jacobi SVD. The algorithm:
//! 1. Evaluates all matrix elements to `f64` (symbolic matrices are rejected).
//! 2. Computes `B = AᵀA` (n×n symmetric positive-semidefinite).
//! 3. Applies Jacobi eigendecomposition to `B` to get eigenvalues (σ²) and
//!    eigenvectors (`V`).
//! 4. Derives singular values `σᵢ = sqrt(λᵢ)` and computes `U = A V Σ⁻¹`.
//! 5. Sorts the triplet `(U, σ, Vᵀ)` so singular values are descending.

use std::sync::Arc;

use crate::numeric::expr::Expr;

use super::{MatrixError, MatrixExpr, MatrixResult};

// ── algorithm constants ───────────────────────────────────────────────────────

const MAX_ITERATIONS: usize = 1_000;
const TOLERANCE: f64 = 1e-12;

// ── private numeric helpers ───────────────────────────────────────────────────

/// Extract all matrix elements as `f64`.
///
/// Returns `InvalidOperation` when any element is symbolic.
fn evaluate_to_f64(m: &MatrixExpr) -> MatrixResult<Vec<Vec<f64>>> {
    let empty = std::collections::HashMap::new();
    m.evaluate(&empty).ok_or_else(|| {
        MatrixError::InvalidOperation(
            "SVD requires a fully numeric matrix; symbolic elements are not supported".to_string(),
        )
    })
}

/// Convert a `Vec<Vec<f64>>` into a `MatrixExpr` of `Expr::Float` values.
fn f64_to_matrix(data: Vec<Vec<f64>>) -> MatrixResult<MatrixExpr> {
    let rows = data.len();
    if rows == 0 {
        return Err(MatrixError::EmptyMatrix);
    }
    let cols = data[0].len();
    let elements: Vec<Vec<Arc<Expr>>> = data
        .into_iter()
        .map(|row| row.into_iter().map(Expr::float).collect())
        .collect();
    Ok(MatrixExpr::from_expr_elements_unchecked(
        rows, cols, elements,
    ))
}

/// Multiply two dense f64 matrices (m×k) × (k×n) → m×n.
fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let k = b.len();
    let n = if k == 0 { 0 } else { b[0].len() };
    let mut c = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i][p];
            for j in 0..n {
                c[i][j] += aip * b[p][j];
            }
        }
    }
    c
}

/// Transpose a dense f64 matrix.
fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = if m == 0 { 0 } else { a[0].len() };
    let mut t = vec![vec![0.0_f64; m]; n];
    for i in 0..m {
        for j in 0..n {
            t[j][i] = a[i][j];
        }
    }
    t
}

/// Apply a Jacobi rotation to symmetric matrix `a` (in-place) for pivot (p, q).
///
/// Also accumulates the rotation into eigenvector matrix `v`.
fn apply_jacobi_rotation(a: &mut Vec<Vec<f64>>, v: &mut Vec<Vec<f64>>, p: usize, q: usize) {
    let app = a[p][p];
    let aqq = a[q][q];
    let apq = a[p][q];

    let tau = (aqq - app) / (2.0 * apq);
    let t = if tau >= 0.0 {
        1.0 / (tau + (1.0 + tau * tau).sqrt())
    } else {
        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
    };
    let c = 1.0 / (1.0 + t * t).sqrt();
    let s = t * c;

    let n = a.len();

    // Update diagonal entries.
    a[p][p] = app - t * apq;
    a[q][q] = aqq + t * apq;
    a[p][q] = 0.0;
    a[q][p] = 0.0;

    // Update remaining rows/cols (r ≠ p, r ≠ q).
    for r in 0..n {
        if r != p && r != q {
            let arp = a[r][p];
            let arq = a[r][q];
            a[r][p] = c * arp - s * arq;
            a[p][r] = a[r][p];
            a[r][q] = s * arp + c * arq;
            a[q][r] = a[r][q];
        }
    }

    // Accumulate rotation into V.
    for r in 0..n {
        let vrp = v[r][p];
        let vrq = v[r][q];
        v[r][p] = c * vrp - s * vrq;
        v[r][q] = s * vrp + c * vrq;
    }
}

/// Jacobi eigendecomposition of a symmetric matrix `a` (n×n).
///
/// Modifies `a` in-place until it is diagonal (eigenvalues on diagonal).
/// Returns the accumulated eigenvector matrix `V` (columns = eigenvectors).
fn jacobi_eigendecomp(a: &mut Vec<Vec<f64>>, n: usize) -> Vec<Vec<f64>> {
    // V starts as identity.
    let mut v = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    for _ in 0..MAX_ITERATIONS {
        // Compute off-diagonal Frobenius norm (upper triangle only).
        let off_norm: f64 = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum::<f64>()
            .sqrt();

        if off_norm < TOLERANCE {
            break;
        }

        // Sweep all upper-triangle pairs.
        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() > TOLERANCE * 1e-4 {
                    apply_jacobi_rotation(a, &mut v, p, q);
                }
            }
        }
    }

    v
}

// ── public SVD implementation ─────────────────────────────────────────────────

impl MatrixExpr {
    /// Compute the Singular Value Decomposition `A = U Σ Vᵀ`.
    ///
    /// Uses one-sided Jacobi SVD via eigendecomposition of `AᵀA`.
    ///
    /// Returns `(U, singular_values, Vᵀ)` where:
    /// - `U` is m×m orthogonal,
    /// - `singular_values` are non-negative and sorted in descending order,
    /// - `Vᵀ` is n×n orthogonal.
    ///
    /// All matrix elements must evaluate to concrete `f64` values. Symbolic
    /// matrices return an error.
    ///
    /// # Errors
    ///
    /// - `InvalidOperation` if any element contains unresolved symbolic variables.
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
    /// let (u, sigma, vt) = a.svd_numeric().unwrap();
    /// assert_eq!(u.rows(), 2);
    /// assert_eq!(sigma.len(), 2);
    /// assert_eq!(vt.rows(), 2);
    /// ```
    pub fn svd_numeric(&self) -> MatrixResult<(MatrixExpr, Vec<f64>, MatrixExpr)> {
        let m = self.rows;
        let n = self.cols;

        let a = evaluate_to_f64(self)?;

        // B = AᵀA (n×n symmetric positive-semidefinite).
        let at = transpose(&a);
        let mut b = mat_mul(&at, &a);

        // Jacobi eigendecomposition of B; eigenvectors accumulate in v.
        let v = jacobi_eigendecomp(&mut b, n);

        // Eigenvalues of B = σ² — extract from diagonal, clamp negatives to 0.
        let mut sigma: Vec<f64> = (0..n).map(|i| b[i][i].max(0.0).sqrt()).collect();

        // Build U: columns u_i = A·v_i / σ_i; zero column when σ_i ≈ 0.
        let av = mat_mul(&a, &v);
        let mut u_data = vec![vec![0.0_f64; m]; m];

        for j in 0..n.min(m) {
            if sigma[j] > TOLERANCE {
                for i in 0..m {
                    u_data[i][j] = av[i][j] / sigma[j];
                }
            }
            // σ_j ≈ 0 → leave column as zero; orthonormalize below if needed.
        }

        // Fill any remaining U columns (when m > n) with an orthonormal basis
        // via Gram-Schmidt against the already-filled columns.
        if m > n {
            let mut filled = n;
            // Candidate unit vectors e_0, e_1, … until we have m columns.
            let mut candidate = 0;
            while filled < m && candidate < m {
                // Build standard basis vector e_candidate.
                let mut col = vec![0.0_f64; m];
                col[candidate] = 1.0;
                candidate += 1;

                // Gram-Schmidt: subtract projections onto already-filled columns.
                for k in 0..filled {
                    let dot: f64 = (0..m).map(|i| u_data[i][k] * col[i]).sum();
                    for i in 0..m {
                        col[i] -= dot * u_data[i][k];
                    }
                }

                // Normalize.
                let norm: f64 = col.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > TOLERANCE {
                    for i in 0..m {
                        u_data[i][filled] = col[i] / norm;
                    }
                    filled += 1;
                }
            }
        }

        // Sort singular values descending; permute U columns and V columns.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| sigma[b].partial_cmp(&sigma[a]).unwrap());

        let sigma_sorted: Vec<f64> = order.iter().map(|&k| sigma[k]).collect();

        let mut v_sorted = vec![vec![0.0_f64; n]; n];
        for (new_j, &old_j) in order.iter().enumerate() {
            for i in 0..n {
                v_sorted[i][new_j] = v[i][old_j];
            }
        }

        let mut u_sorted = vec![vec![0.0_f64; m]; m];
        // Copy the first n columns according to sort order.
        for (new_j, &old_j) in order.iter().enumerate() {
            for i in 0..m {
                u_sorted[i][new_j] = u_data[i][old_j];
            }
        }
        // The remaining m-n columns (if any) can stay in their current positions.
        for j in n..m {
            for i in 0..m {
                u_sorted[i][j] = u_data[i][j];
            }
        }

        sigma = sigma_sorted;

        let vt = transpose(&v_sorted);
        let u_mat = f64_to_matrix(u_sorted)?;
        let vt_mat = f64_to_matrix(vt)?;

        Ok((u_mat, sigma, vt_mat))
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert two f64 values agree within a loose tolerance.
    fn approx(a: f64, b: f64) {
        assert!(
            (a - b).abs() < 1e-9,
            "expected {b:.15}, got {a:.15}, diff = {:.3e}",
            (a - b).abs()
        );
    }

    /// Reconstruct A from (U, sigma, Vᵀ) and compare element-wise.
    fn check_reconstruction(original: &MatrixExpr, u: &MatrixExpr, sigma: &[f64], vt: &MatrixExpr) {
        let empty = std::collections::HashMap::new();
        let m = original.rows();
        let n = original.cols();
        let k = sigma.len();

        // Compute U Σ Vᵀ manually.
        // First build Σ as an m×n f64 matrix.
        let mut sigma_mat = vec![vec![0.0_f64; n]; m];
        for i in 0..k.min(m).min(n) {
            sigma_mat[i][i] = sigma[i];
        }

        let u_data: Vec<Vec<f64>> = (0..m)
            .map(|i| {
                (0..m)
                    .map(|j| u.evaluate(&empty).expect("U must be numeric")[i][j])
                    .collect()
            })
            .collect();

        let vt_data: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| vt.evaluate(&empty).expect("Vt must be numeric")[i][j])
                    .collect()
            })
            .collect();

        // reconstructed = U * sigma_mat * Vt
        let us = mat_mul(&u_data, &sigma_mat);
        let reconstructed = mat_mul(&us, &vt_data);

        let a_data = original.evaluate(&empty).expect("A must be numeric");
        for i in 0..m {
            for j in 0..n {
                approx(reconstructed[i][j], a_data[i][j]);
            }
        }
    }

    #[test]
    fn fast_test_svd_2x2() {
        let a = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(2)],
            vec![Expr::int(3), Expr::int(4)],
        ])
        .unwrap();

        let (u, sigma, vt) = a.svd_numeric().unwrap();
        assert_eq!(u.rows(), 2);
        assert_eq!(u.cols(), 2);
        assert_eq!(vt.rows(), 2);
        assert_eq!(vt.cols(), 2);
        assert_eq!(sigma.len(), 2);

        check_reconstruction(&a, &u, &sigma, &vt);
    }

    #[test]
    fn fast_test_svd_3x3() {
        let a = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(2), Expr::int(3)],
            vec![Expr::int(4), Expr::int(5), Expr::int(6)],
            vec![Expr::int(7), Expr::int(8), Expr::int(9)],
        ])
        .unwrap();

        let (u, sigma, vt) = a.svd_numeric().unwrap();
        assert_eq!(u.rows(), 3);
        assert_eq!(vt.rows(), 3);
        assert_eq!(sigma.len(), 3);

        check_reconstruction(&a, &u, &sigma, &vt);
    }

    #[test]
    fn fast_test_svd_diagonal() {
        // Diagonal matrix: SVD should return the diagonal values as singular values.
        let a = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(5), Expr::int(0), Expr::int(0)],
            vec![Expr::int(0), Expr::int(3), Expr::int(0)],
            vec![Expr::int(0), Expr::int(0), Expr::int(1)],
        ])
        .unwrap();

        let (_u, sigma, _vt) = a.svd_numeric().unwrap();

        // Singular values should be 5, 3, 1 (descending).
        approx(sigma[0], 5.0);
        approx(sigma[1], 3.0);
        approx(sigma[2], 1.0);
    }

    #[test]
    fn fast_test_svd_singular_values_sorted() {
        let a = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(3), Expr::int(1)],
            vec![Expr::int(1), Expr::int(3)],
        ])
        .unwrap();

        let (_u, sigma, _vt) = a.svd_numeric().unwrap();

        // Singular values must be in descending order.
        for i in 0..sigma.len().saturating_sub(1) {
            assert!(
                sigma[i] >= sigma[i + 1],
                "singular values not sorted: sigma[{i}]={} < sigma[{}]={}",
                sigma[i],
                i + 1,
                sigma[i + 1]
            );
        }
    }

    #[test]
    fn fast_test_svd_singular_values_nonneg() {
        let a = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(-2), Expr::int(1)],
            vec![Expr::int(0), Expr::int(-3)],
        ])
        .unwrap();

        let (_u, sigma, _vt) = a.svd_numeric().unwrap();

        for (i, &s) in sigma.iter().enumerate() {
            assert!(s >= 0.0, "singular value {i} is negative: {s}");
        }
    }

    #[test]
    fn fast_test_svd_error_symbolic() {
        let a = MatrixExpr::from_expr_elements(vec![
            vec![Expr::symbol("x"), Expr::int(1)],
            vec![Expr::int(0), Expr::int(2)],
        ])
        .unwrap();

        let err = a.svd_numeric().unwrap_err();
        assert!(
            matches!(err, MatrixError::InvalidOperation(_)),
            "expected InvalidOperation for symbolic matrix, got {err:?}"
        );
    }
}
