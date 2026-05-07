//! Symbolic eigenvector computation for [`MatrixExpr`].
//!
//! Computes eigenvectors by combining numeric eigenvalue finding with
//! kernel computation of `(A - λI)`. For numeric matrices, eigenvalues are
//! obtained via [`eigenvalues_numeric`](MatrixExpr::eigenvalues_numeric) and
//! eigenvectors via [`kernel`](MatrixExpr::kernel).
//!
//! # Strategy
//!
//! For each distinct real eigenvalue λ:
//! 1. Build `A - λI` by subtracting λ from each diagonal element.
//! 2. Compute the kernel of `A - λI` using RREF.
//! 3. Return `(λ as Arc<Expr>, kernel_basis)`.
//!
//! Eigenvalues with a non-zero imaginary part are skipped (real matrices
//! may have complex conjugate pairs; without a complex eigenvector solver
//! the symbolic output would be meaningless for those).

use std::sync::Arc;

use crate::numeric::expr::Expr;

use super::{MatrixError, MatrixExpr, MatrixResult};

/// Tolerance used for determining whether an eigenvalue is real and for
/// deduplicating nearly-equal eigenvalues.
const EIGEN_TOL: f64 = 1e-10;

impl MatrixExpr {
    /// Compute symbolic eigenvectors for all real eigenvalues.
    ///
    /// Returns a list of `(eigenvalue, eigenvector_basis)` pairs, one per
    /// distinct real eigenvalue.  Each eigenvalue is represented as an
    /// `Arc<Expr>` float literal.  Each eigenvector basis is a `Vec` of
    /// `n×1` column [`MatrixExpr`] values spanning the eigenspace.
    ///
    /// Eigenvalues with a non-zero imaginary part are silently skipped.
    ///
    /// # Errors
    ///
    /// - [`MatrixError::InvalidOperation`] if the matrix is not square.
    /// - [`MatrixError::InvalidOperation`] if the matrix cannot be evaluated
    ///   numerically (symbolic entries are not yet supported).
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::numeric::expr::Expr;
    ///
    /// // Diagonal matrix: eigenvalues 2 and 3, each with a standard basis vector.
    /// let m = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(2), Expr::int(0)],
    ///     vec![Expr::int(0), Expr::int(3)],
    /// ]).unwrap();
    ///
    /// let pairs = m.symbolic_eigenvectors().unwrap();
    /// assert_eq!(pairs.len(), 2);
    /// ```
    pub fn symbolic_eigenvectors(&self) -> MatrixResult<Vec<(Arc<Expr>, Vec<MatrixExpr>)>> {
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "Eigenvectors require a square matrix".to_string(),
            ));
        }

        // Obtain complex eigenvalues numerically; error if matrix is not numeric.
        let complex_eigs = self.eigenvalues_numeric()?;

        // Keep only real eigenvalues and deduplicate within EIGEN_TOL.
        let real_eigs = collect_distinct_real(&complex_eigs);

        let mut result = Vec::with_capacity(real_eigs.len());
        for lambda_val in real_eigs {
            let basis = eigenvectors_for_eigenvalue(self, lambda_val)?;
            let lambda_expr = Expr::float(lambda_val);
            result.push((lambda_expr, basis));
        }

        Ok(result)
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Filter complex eigenvalues to distinct real values (|imag| < EIGEN_TOL).
///
/// Two real values are considered equal when |a − b| < EIGEN_TOL.
fn collect_distinct_real(eigs: &[num_complex::Complex64]) -> Vec<f64> {
    let mut real_vals: Vec<f64> = Vec::new();
    for ev in eigs {
        if ev.im.abs() >= EIGEN_TOL {
            continue;
        }
        let r = ev.re;
        if real_vals.iter().all(|&x| (x - r).abs() >= EIGEN_TOL) {
            real_vals.push(r);
        }
    }
    real_vals
}

/// Build `A - λI` and return its kernel basis.
fn eigenvectors_for_eigenvalue(a: &MatrixExpr, lambda_val: f64) -> MatrixResult<Vec<MatrixExpr>> {
    let shifted = subtract_scalar_identity(a, lambda_val)?;
    shifted.kernel()
}

/// Construct `A - λI` as a matrix of `Expr::float` values.
///
/// Evaluates every entry of `a` numerically, subtracts `lambda_val` from
/// the diagonal, and wraps each result in `Expr::float`.  Using a uniform
/// float representation guarantees that RREF arithmetic sees concrete
/// numbers rather than mixed `Integer`/`Float` symbolic terms that would
/// not simplify to zero at the symbolic level.
///
/// # Errors
///
/// Returns [`MatrixError::InvalidOperation`] if any entry cannot be
/// evaluated numerically.
fn subtract_scalar_identity(a: &MatrixExpr, lambda_val: f64) -> MatrixResult<MatrixExpr> {
    let n = a.rows();
    let empty = std::collections::HashMap::new();

    let numeric = a.evaluate(&empty).ok_or_else(|| {
        MatrixError::InvalidOperation(
            "Cannot evaluate matrix numerically for eigenvector computation".to_string(),
        )
    })?;

    let elements: Vec<Vec<Arc<Expr>>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let val = if i == j {
                        numeric[i][j] - lambda_val
                    } else {
                        numeric[i][j]
                    };
                    Expr::float(val)
                })
                .collect()
        })
        .collect();

    MatrixExpr::from_expr_elements(elements)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::matrix::MatrixExpr;
    use crate::numeric::{evaluation::evaluate, expr::Expr, SymbolId};

    fn eval_f(e: &std::sync::Arc<Expr>) -> f64 {
        let empty: HashMap<SymbolId, f64> = HashMap::new();
        evaluate(e, &empty).expect("expression must be fully numeric")
    }

    fn approx_eq(a: f64, b: f64) {
        assert!(
            (a - b).abs() < 1e-8,
            "expected {b:.12}, got {a:.12}, diff = {:.3e}",
            (a - b).abs()
        );
    }

    // ── fast_test_symbolic_eigen_diagonal_2x2 ────────────────────────────────

    /// diag(2, 3): eigenvalues 2 and 3, each a standard basis vector.
    #[test]
    fn fast_test_symbolic_eigen_diagonal_2x2() {
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(2), Expr::int(0)],
            vec![Expr::int(0), Expr::int(3)],
        ])
        .unwrap();

        let pairs = m.symbolic_eigenvectors().unwrap();
        assert_eq!(pairs.len(), 2, "diagonal 2×2 must have 2 eigenvalues");

        for (_, basis) in &pairs {
            assert_eq!(basis.len(), 1, "each eigenspace must be 1-dimensional");
            assert_eq!(basis[0].rows(), 2);
            assert_eq!(basis[0].cols(), 1);
        }

        // Collect eigenvalues and sort ascending for deterministic comparison.
        let mut evs: Vec<f64> = pairs.iter().map(|(e, _)| eval_f(e)).collect();
        evs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        approx_eq(evs[0], 2.0);
        approx_eq(evs[1], 3.0);
    }

    // ── fast_test_symbolic_eigen_symmetric_2x2 ───────────────────────────────

    /// [[2,1],[1,2]]: eigenvalues 1 and 3.
    #[test]
    fn fast_test_symbolic_eigen_symmetric_2x2() {
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(2), Expr::int(1)],
            vec![Expr::int(1), Expr::int(2)],
        ])
        .unwrap();

        let pairs = m.symbolic_eigenvectors().unwrap();
        assert_eq!(pairs.len(), 2, "symmetric 2×2 must have 2 real eigenvalues");

        let mut evs: Vec<f64> = pairs.iter().map(|(e, _)| eval_f(e)).collect();
        evs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        approx_eq(evs[0], 1.0);
        approx_eq(evs[1], 3.0);
    }

    // ── fast_test_symbolic_eigen_verify_av_lambda_v ──────────────────────────

    /// For [[2,1],[1,2]], verify Av = λv for every returned eigenpair.
    #[test]
    fn fast_test_symbolic_eigen_verify_av_lambda_v() {
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(2), Expr::int(1)],
            vec![Expr::int(1), Expr::int(2)],
        ])
        .unwrap();

        let pairs = m.symbolic_eigenvectors().unwrap();

        for (lambda_expr, basis) in &pairs {
            let lambda_val = eval_f(lambda_expr);
            for v in basis {
                // Av
                let av = m.mul(v).unwrap();
                // λv
                let lambda_arc = Expr::float(lambda_val);
                let lv = v.scalar_mul(&lambda_arc);

                assert_eq!(av.rows(), v.rows());
                assert_eq!(av.cols(), 1);

                for row in 0..av.rows() {
                    let av_i = eval_f(av.get(row, 0).unwrap());
                    let lv_i = eval_f(lv.get(row, 0).unwrap());
                    approx_eq(av_i, lv_i);
                }
            }
        }
    }

    // ── fast_test_symbolic_eigen_3x3 ─────────────────────────────────────────

    /// diag(1, 2, 3): three distinct eigenvalues, each a standard basis vector.
    #[test]
    fn fast_test_symbolic_eigen_3x3() {
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(0), Expr::int(0)],
            vec![Expr::int(0), Expr::int(2), Expr::int(0)],
            vec![Expr::int(0), Expr::int(0), Expr::int(3)],
        ])
        .unwrap();

        let pairs = m.symbolic_eigenvectors().unwrap();
        assert_eq!(pairs.len(), 3, "diagonal 3×3 must have 3 eigenvalues");

        let mut evs: Vec<f64> = pairs.iter().map(|(e, _)| eval_f(e)).collect();
        evs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        approx_eq(evs[0], 1.0);
        approx_eq(evs[1], 2.0);
        approx_eq(evs[2], 3.0);

        // Verify Av = λv for each eigenpair.
        for (lambda_expr, basis) in &pairs {
            let lambda_val = eval_f(lambda_expr);
            for v in basis {
                let av = m.mul(v).unwrap();
                let lv = v.scalar_mul(&Expr::float(lambda_val));
                for row in 0..av.rows() {
                    approx_eq(
                        eval_f(av.get(row, 0).unwrap()),
                        eval_f(lv.get(row, 0).unwrap()),
                    );
                }
            }
        }
    }

    // ── fast_test_symbolic_eigen_non_square_error ────────────────────────────

    /// Non-square matrix must return an error.
    #[test]
    fn fast_test_symbolic_eigen_non_square_error() {
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(2), Expr::int(3)],
            vec![Expr::int(4), Expr::int(5), Expr::int(6)],
        ])
        .unwrap();

        assert!(
            m.symbolic_eigenvectors().is_err(),
            "non-square matrix must return an error"
        );
    }
}
