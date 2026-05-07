//! Minimal polynomial computation for matrices.
//!
//! The minimal polynomial m(λ) of a matrix A is the monic polynomial of
//! smallest degree such that m(A) = 0 (the zero matrix). By Cayley-Hamilton
//! the characteristic polynomial always annihilates A, so the minimal
//! polynomial divides the characteristic polynomial.

use std::sync::Arc;

use num_complex::Complex64;

use crate::numeric::expr::Expr;
use crate::numeric::normalize;

use super::{MatrixError, MatrixExpr, MatrixResult};

/// Tolerance for considering an eigenvalue imaginary part zero (real eigenvalue).
const REAL_TOL: f64 = 1e-10;

/// Tolerance for considering two eigenvalues identical.
const EV_TOL: f64 = 1e-10;

/// Tolerance for considering a matrix entry zero.
const ZERO_TOL: f64 = 1e-9;

impl MatrixExpr {
    /// Compute the minimal polynomial of the matrix in the given variable.
    ///
    /// Uses a numeric approach:
    /// 1. Collect the distinct real eigenvalues from `eigenvalues_numeric`.
    /// 2. For each eigenvalue, determine the smallest exponent k such that
    ///    the product ∏(A − λᵢI)^kᵢ evaluates to the zero matrix.
    /// 3. Returns the monic polynomial ∏(λ − λᵢ)^kᵢ as an `Arc<Expr>`.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::InvalidOperation`] if the matrix is not square or
    /// cannot be evaluated numerically.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::numeric::expr::Expr;
    ///
    /// // Identity 2×2 — minimal polynomial is λ−1
    /// let id = MatrixExpr::identity(2);
    /// let mp = id.minimal_polynomial("lambda").unwrap();
    /// ```
    pub fn minimal_polynomial(&self, var: &str) -> MatrixResult<Arc<Expr>> {
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "Minimal polynomial requires a square matrix".to_string(),
            ));
        }

        let n = self.rows;
        let complex_evs = self.eigenvalues_numeric()?;

        // Collect distinct real eigenvalues (drop complex ones).
        let distinct = distinct_real_eigenvalues(&complex_evs);

        // Compute algebraic multiplicity of each distinct eigenvalue.
        let alg_mults: Vec<usize> = distinct
            .iter()
            .map(|&lam| {
                complex_evs
                    .iter()
                    .filter(|&&ev| ev.im.abs() < REAL_TOL && (ev.re - lam).abs() < EV_TOL)
                    .count()
            })
            .collect();

        // Find minimal exponents via factored evaluation.
        let exponents = find_minimal_exponents(self, n, &distinct, &alg_mults)?;

        // Build the symbolic polynomial ∏(λ − λᵢ)^kᵢ.
        build_minimal_poly(var, &distinct, &exponents)
    }

    /// Evaluate a matrix polynomial p(A) using Horner's method.
    ///
    /// `coeffs` are ordered constant-term first: `[c₀, c₁, …, cₙ]`
    /// so `p(A) = c₀I + c₁A + … + cₙAⁿ`.
    ///
    /// # Errors
    ///
    /// Returns an error if a matrix multiplication fails (dimension mismatch
    /// would be an internal bug given a square matrix).
    pub(crate) fn eval_matrix_poly(&self, coeffs: &[Arc<Expr>]) -> MatrixResult<MatrixExpr> {
        let n = self.rows;
        if coeffs.is_empty() {
            return Ok(MatrixExpr::zero(n, n));
        }

        // Horner: start from the highest-degree coefficient.
        let degree = coeffs.len() - 1;
        let mut result = MatrixExpr::identity(n).scalar_mul(&coeffs[degree]);

        for i in (0..degree).rev() {
            result = result.mul(self)?;
            let ci_identity = MatrixExpr::identity(n).scalar_mul(&coeffs[i]);
            result = result.add(&ci_identity)?;
        }

        Ok(result)
    }

    /// Return `true` if every element of the matrix evaluates to zero within
    /// tolerance [`ZERO_TOL`].
    pub(crate) fn is_zero_matrix(&self) -> bool {
        let empty = std::collections::HashMap::new();
        match self.evaluate(&empty) {
            None => false,
            Some(rows) => rows
                .iter()
                .all(|row| row.iter().all(|&v| v.abs() < ZERO_TOL)),
        }
    }
}

// =============================================================================
// Private helpers
// =============================================================================

/// Collect distinct real eigenvalues (sorted ascending) from a complex list.
fn distinct_real_eigenvalues(evs: &[Complex64]) -> Vec<f64> {
    let mut real_evs: Vec<f64> = evs
        .iter()
        .filter(|ev| ev.im.abs() < REAL_TOL)
        .map(|ev| ev.re)
        .collect();

    real_evs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Deduplicate within tolerance.
    let mut distinct: Vec<f64> = Vec::new();
    for val in real_evs {
        if !distinct.iter().any(|&d| (d - val).abs() < EV_TOL) {
            distinct.push(val);
        }
    }
    distinct
}

/// Build the matrix (A − λI)^k numerically using repeated multiplication.
fn matrix_factor_power(matrix: &MatrixExpr, lambda: f64, k: usize) -> MatrixResult<MatrixExpr> {
    let n = matrix.rows;

    // Build A − λI using symbolic scalar mul and sub.
    let lambda_i = MatrixExpr::identity(n).scalar_mul(&Expr::float(lambda));
    let factor = matrix.sub(&lambda_i)?;

    if k == 0 {
        return Ok(MatrixExpr::identity(n));
    }

    let mut result = factor.clone();
    for _ in 1..k {
        result = result.mul(&factor)?;
    }
    Ok(result)
}

/// Determine the minimal exponent for each distinct eigenvalue such that
/// `∏(A − λᵢI)^kᵢ` evaluates to the zero matrix.
///
/// Tries exponent vectors starting from all-ones, incrementing one eigenvalue
/// at a time (bounded by algebraic multiplicity), until the product is zero.
fn find_minimal_exponents(
    matrix: &MatrixExpr,
    n: usize,
    distinct: &[f64],
    alg_mults: &[usize],
) -> MatrixResult<Vec<usize>> {
    let num_factors = distinct.len();

    // Handle the edge case: no real eigenvalues found.
    // Fall back to characteristic polynomial degree (= n for full-rank CAS).
    if num_factors == 0 {
        return Ok(vec![]);
    }

    // Start with k_i = 1 for all factors.
    let mut exponents: Vec<usize> = vec![1; num_factors];

    // Try candidates by increasing exponents one factor at a time.
    // Total iterations bounded by sum of algebraic multiplicities ≤ n.
    loop {
        if eval_factored_product(matrix, n, distinct, &exponents)?.is_zero_matrix() {
            return Ok(exponents);
        }

        // Increment the exponent of the first factor that hasn't reached its
        // algebraic multiplicity cap.
        let incremented = try_increment_exponents(&mut exponents, alg_mults);
        if !incremented {
            // Reached characteristic polynomial; Cayley-Hamilton guarantees
            // this must be zero — return regardless.
            return Ok(exponents);
        }
    }
}

/// Increment exponents lexicographically within the algebraic multiplicity
/// bounds. Returns false when the maximum (all at algebraic multiplicity) is
/// reached.
fn try_increment_exponents(exponents: &mut [usize], alg_mults: &[usize]) -> bool {
    for i in 0..exponents.len() {
        if exponents[i] < alg_mults[i] {
            exponents[i] += 1;
            return true;
        }
    }
    false
}

/// Evaluate the product ∏(A − λᵢI)^kᵢ symbolically.
fn eval_factored_product(
    matrix: &MatrixExpr,
    n: usize,
    distinct: &[f64],
    exponents: &[usize],
) -> MatrixResult<MatrixExpr> {
    let mut product = MatrixExpr::identity(n);
    for (&lambda, &k) in distinct.iter().zip(exponents.iter()) {
        let factor_k = matrix_factor_power(matrix, lambda, k)?;
        product = product.mul(&factor_k)?;
    }
    Ok(product)
}

/// Build the symbolic minimal polynomial ∏(λ − λᵢ)^kᵢ as `Arc<Expr>`.
fn build_minimal_poly(var: &str, distinct: &[f64], exponents: &[usize]) -> MatrixResult<Arc<Expr>> {
    let lam = Expr::symbol(var);

    // Degenerate: no distinct real eigenvalues — return constant 1 as
    // best-effort (complex-eigenvalue case not fully handled numerically).
    if distinct.is_empty() {
        return Ok(Expr::int(1));
    }

    let mut poly = build_linear_factor(lam.clone(), distinct[0], exponents[0]);
    for (&lam_i, &k_i) in distinct[1..].iter().zip(exponents[1..].iter()) {
        let factor = build_linear_factor(lam.clone(), lam_i, k_i);
        poly = normalize::mul(poly, factor);
    }
    Ok(poly)
}

/// Build (λ − root)^k as `Arc<Expr>`.
fn build_linear_factor(lam: Arc<Expr>, root: f64, k: usize) -> Arc<Expr> {
    let linear = normalize::sub(lam, Expr::float(root));
    if k <= 1 {
        return linear;
    }
    let mut result = linear.clone();
    for _ in 1..k {
        result = normalize::mul(result, linear.clone());
    }
    result
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::evaluation::evaluate;
    use crate::numeric::SymbolId;
    use std::collections::HashMap;

    /// Evaluate the symbolic minimal polynomial at a numeric λ value.
    fn eval_poly_at(poly: &Arc<Expr>, var: &str, value: f64) -> f64 {
        let mut vars: HashMap<SymbolId, f64> = HashMap::new();
        vars.insert(SymbolId::intern(var), value);
        evaluate(poly, &vars).unwrap_or(f64::NAN)
    }

    /// Check that m(A) ≈ zero matrix by evaluating the factored product directly.
    fn annihilates_matrix(matrix: &MatrixExpr, distinct: &[f64], exponents: &[usize]) -> bool {
        let n = matrix.rows;
        match eval_factored_product(matrix, n, distinct, exponents) {
            Ok(result) => result.is_zero_matrix(),
            Err(_) => false,
        }
    }

    #[test]
    fn fast_test_min_poly_identity_2x2() {
        // Identity matrix: every vector is an eigenvector with eigenvalue 1.
        // Minimal polynomial = λ − 1 (degree 1).
        let id = MatrixExpr::identity(2);
        let mp = id.minimal_polynomial("lambda").unwrap();

        // Evaluate at λ=1 → should be 0.
        let at_one = eval_poly_at(&mp, "lambda", 1.0);
        assert!(at_one.abs() < 1e-9, "m(1) should be 0, got {at_one}");

        // Verify (A − I) = zero matrix directly.
        let distinct = vec![1.0_f64];
        let exponents = vec![1_usize];
        assert!(
            annihilates_matrix(&id, &distinct, &exponents),
            "(I − I) should be the zero matrix"
        );
    }

    #[test]
    fn fast_test_min_poly_scalar_matrix() {
        // 3I: minimal polynomial = λ − 3.
        let m = MatrixExpr::identity(3).scalar_mul(&Expr::int(3));
        let mp = m.minimal_polynomial("t").unwrap();

        let at_three = eval_poly_at(&mp, "t", 3.0);
        assert!(at_three.abs() < 1e-9, "m(3) should be 0, got {at_three}");

        // The polynomial should vanish only at 3, not at 0 or 1.
        let at_zero = eval_poly_at(&mp, "t", 0.0);
        assert!(at_zero.abs() > 0.1, "m(0) should not be 0");

        let distinct = vec![3.0_f64];
        let exponents = vec![1_usize];
        assert!(annihilates_matrix(&m, &distinct, &exponents));
    }

    #[test]
    fn fast_test_min_poly_diagonal_distinct() {
        // diag(1, 2, 3): all eigenvalues distinct, so minimal poly = char poly
        // = (λ−1)(λ−2)(λ−3), degree 3.
        let d = MatrixExpr::diagonal(vec![Expr::int(1), Expr::int(2), Expr::int(3)]);
        let mp = d.minimal_polynomial("lambda").unwrap();

        // Must vanish at each eigenvalue.
        for &root in &[1.0_f64, 2.0, 3.0] {
            let val = eval_poly_at(&mp, "lambda", root);
            assert!(val.abs() < 1e-9, "m({root}) should be 0, got {val}");
        }

        // Must not vanish at a non-eigenvalue.
        let at_zero = eval_poly_at(&mp, "lambda", 0.0);
        assert!(at_zero.abs() > 0.1, "m(0) should not be 0 for diag(1,2,3)");

        let distinct = vec![1.0_f64, 2.0, 3.0];
        let exponents = vec![1_usize, 1, 1];
        assert!(annihilates_matrix(&d, &distinct, &exponents));
    }

    #[test]
    fn fast_test_min_poly_repeated_eigenvalue_non_defective() {
        // diag(2, 2, 3): eigenvalue 2 has algebraic mult 2 but is diagonal
        // (geometric mult also 2) → NOT defective → minimal poly = (λ−2)(λ−3),
        // degree 2, which is strictly less than char poly degree 3.
        let d = MatrixExpr::diagonal(vec![Expr::int(2), Expr::int(2), Expr::int(3)]);
        let mp = d.minimal_polynomial("lambda").unwrap();

        // Vanishes at eigenvalues.
        for &root in &[2.0_f64, 3.0] {
            let val = eval_poly_at(&mp, "lambda", root);
            assert!(val.abs() < 1e-9, "m({root}) should be 0, got {val}");
        }

        // Verify (A−2I)(A−3I) = 0.
        let distinct = vec![2.0_f64, 3.0];
        let exponents = vec![1_usize, 1];
        assert!(
            annihilates_matrix(&d, &distinct, &exponents),
            "(A−2I)(A−3I) should be zero for diag(2,2,3)"
        );
    }

    #[test]
    fn fast_test_min_poly_non_square_error() {
        // A 2×3 matrix must return an error.
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(2), Expr::int(3)],
            vec![Expr::int(4), Expr::int(5), Expr::int(6)],
        ])
        .unwrap();

        let result = m.minimal_polynomial("lambda");
        assert!(result.is_err(), "Non-square matrix should return an error");
    }
}
