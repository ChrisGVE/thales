//! Definiteness classification for symmetric matrices via Sylvester's criterion.
//!
//! Sylvester's criterion states that a real symmetric matrix is positive definite
//! if and only if all its leading principal minors are strictly positive.
//! Variants of the criterion give negative definite, semidefinite, and indefinite
//! classifications from the sign pattern of those minors.

use std::sync::Arc;

use crate::numeric::{evaluation::evaluate, expr::Expr};

use super::{MatrixError, MatrixExpr, MatrixResult};

// ── public types ──────────────────────────────────────────────────────────────

/// Definiteness classification of a quadratic form / symmetric matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Definiteness {
    /// All eigenvalues strictly positive; all leading principal minors > 0.
    PositiveDefinite,
    /// All eigenvalues non-negative; at least one is zero.
    PositiveSemidefinite,
    /// All eigenvalues strictly negative; minors alternate in sign starting negative.
    NegativeDefinite,
    /// All eigenvalues non-positive; at least one is zero.
    NegativeSemidefinite,
    /// Eigenvalues of mixed sign.
    Indefinite,
    /// Matrix contains symbolic entries that cannot be evaluated numerically.
    Unknown,
}

// ── private helpers ───────────────────────────────────────────────────────────

/// Try to evaluate an `Arc<Expr>` to `f64` using an empty symbol table.
fn as_f64(e: &Arc<Expr>) -> Option<f64> {
    let empty = std::collections::HashMap::new();
    evaluate(e, &empty)
}

// ── impl ──────────────────────────────────────────────────────────────────────

impl MatrixExpr {
    /// Extract the top-left k×k principal submatrix.
    ///
    /// # Errors
    ///
    /// - `InvalidOperation` if `k == 0` or `k` exceeds the matrix dimensions.
    pub fn submatrix_topleft(&self, k: usize) -> MatrixResult<MatrixExpr> {
        if k == 0 {
            return Err(MatrixError::InvalidOperation(
                "Top-left submatrix size k must be at least 1".to_string(),
            ));
        }
        if k > self.rows || k > self.cols {
            return Err(MatrixError::InvalidOperation(format!(
                "Top-left submatrix size k={k} exceeds matrix dimensions {}×{}",
                self.rows, self.cols
            )));
        }

        let elements: Vec<Vec<Arc<Expr>>> = self.elements[..k]
            .iter()
            .map(|row| row[..k].to_vec())
            .collect();

        Ok(MatrixExpr::from_expr_elements_unchecked(k, k, elements))
    }

    /// Classify the quadratic form of this matrix using Sylvester's criterion.
    ///
    /// Computes the leading principal minors d₁, d₂, …, dₙ (the determinants of
    /// the top-left k×k submatrices for k = 1..n) and inspects their sign pattern.
    ///
    /// | Sign pattern                     | Classification          |
    /// |----------------------------------|-------------------------|
    /// | All dₖ > 0                       | `PositiveDefinite`      |
    /// | All dₖ ≥ 0                       | `PositiveSemidefinite`  |
    /// | (−1)ᵏ dₖ > 0 for all k           | `NegativeDefinite`      |
    /// | (−1)ᵏ dₖ ≥ 0 for all k           | `NegativeSemidefinite`  |
    /// | None of the above                | `Indefinite`            |
    /// | Any minor is symbolic            | `Unknown`               |
    ///
    /// # Errors
    ///
    /// - `InvalidOperation` if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::{MatrixExpr, Definiteness};
    /// use thales::numeric::expr::Expr;
    ///
    /// // 2×2 identity — positive definite
    /// let id = MatrixExpr::identity(2);
    /// assert_eq!(id.classify_definiteness().unwrap(), Definiteness::PositiveDefinite);
    /// ```
    pub fn classify_definiteness(&self) -> MatrixResult<Definiteness> {
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "Definiteness classification requires a square matrix".to_string(),
            ));
        }

        let n = self.rows;
        const TOL: f64 = 1e-10;

        // Collect leading principal minors d₁ … dₙ as f64.
        // If any minor is symbolic, return Unknown immediately.
        let mut minors: Vec<f64> = Vec::with_capacity(n);
        for k in 1..=n {
            let sub = self.submatrix_topleft(k)?;
            let det_expr = sub.determinant()?;
            match as_f64(&det_expr) {
                Some(v) => minors.push(v),
                None => return Ok(Definiteness::Unknown),
            }
        }

        // Check positive definite: all dₖ > 0
        if minors.iter().all(|&d| d > TOL) {
            return Ok(Definiteness::PositiveDefinite);
        }

        // Check negative definite: (−1)ᵏ dₖ > 0  ⟺  d₁ < 0, d₂ > 0, d₃ < 0, …
        let neg_def = minors
            .iter()
            .enumerate()
            .all(|(i, &d)| (i % 2 == 0 && d < -TOL) || (i % 2 == 1 && d > TOL));
        if neg_def {
            return Ok(Definiteness::NegativeDefinite);
        }

        // Check positive semidefinite: all dₖ ≥ 0 (some may be ~0)
        if minors.iter().all(|&d| d >= -TOL) {
            return Ok(Definiteness::PositiveSemidefinite);
        }

        // Check negative semidefinite: (−1)ᵏ dₖ ≥ 0  ⟺  d₁ ≤ 0, d₂ ≥ 0, …
        let neg_semi = minors
            .iter()
            .enumerate()
            .all(|(i, &d)| (i % 2 == 0 && d <= TOL) || (i % 2 == 1 && d >= -TOL));
        if neg_semi {
            return Ok(Definiteness::NegativeSemidefinite);
        }

        Ok(Definiteness::Indefinite)
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(rows: Vec<Vec<i64>>) -> MatrixExpr {
        let elems = rows
            .into_iter()
            .map(|row| row.into_iter().map(Expr::int).collect())
            .collect();
        MatrixExpr::from_expr_elements(elems).unwrap()
    }

    #[test]
    fn fast_test_classify_identity_pd() {
        // n×n identity is always positive definite.
        let id2 = MatrixExpr::identity(2);
        assert_eq!(
            id2.classify_definiteness().unwrap(),
            Definiteness::PositiveDefinite
        );
        let id3 = MatrixExpr::identity(3);
        assert_eq!(
            id3.classify_definiteness().unwrap(),
            Definiteness::PositiveDefinite
        );
    }

    #[test]
    fn fast_test_classify_neg_identity_nd() {
        // −I is negative definite: all eigenvalues −1.
        let neg_i2 = mat(vec![vec![-1, 0], vec![0, -1]]);
        assert_eq!(
            neg_i2.classify_definiteness().unwrap(),
            Definiteness::NegativeDefinite
        );
        let neg_i3 = mat(vec![vec![-1, 0, 0], vec![0, -1, 0], vec![0, 0, -1]]);
        assert_eq!(
            neg_i3.classify_definiteness().unwrap(),
            Definiteness::NegativeDefinite
        );
    }

    #[test]
    fn fast_test_classify_indefinite() {
        // diag(1, −1) has mixed eigenvalues → indefinite.
        let m = mat(vec![vec![1, 0], vec![0, -1]]);
        assert_eq!(m.classify_definiteness().unwrap(), Definiteness::Indefinite);
    }

    #[test]
    fn fast_test_classify_semidefinite() {
        // [[1,1],[1,1]] has det = 0 and one zero eigenvalue → positive semidefinite.
        let m = mat(vec![vec![1, 1], vec![1, 1]]);
        assert_eq!(
            m.classify_definiteness().unwrap(),
            Definiteness::PositiveSemidefinite
        );
    }

    #[test]
    fn fast_test_classify_non_square_error() {
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(2), Expr::int(3)],
            vec![Expr::int(4), Expr::int(5), Expr::int(6)],
        ])
        .unwrap();
        let err = m.classify_definiteness().unwrap_err();
        assert!(
            matches!(err, MatrixError::InvalidOperation(_)),
            "expected InvalidOperation, got {err:?}"
        );
    }

    #[test]
    fn fast_test_classify_3x3_pd() {
        // [[4,2,2],[2,3,1],[2,1,3]] — same PD matrix used in cholesky tests.
        // d₁=4, d₂=det([[4,2],[2,3]])=8, d₃=det(A)=8 — all positive.
        let m = mat(vec![vec![4, 2, 2], vec![2, 3, 1], vec![2, 1, 3]]);
        assert_eq!(
            m.classify_definiteness().unwrap(),
            Definiteness::PositiveDefinite
        );
    }
}
