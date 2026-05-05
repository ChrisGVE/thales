//! Rothstein-Trager algorithm for the logarithmic part of rational integration.
//!
//! Given a proper rational function `c/d` where `d` is squarefree, computes
//! the integral as a finite sum of logarithms over the rationals:
//!
//! ```text
//! ∫ c/d dx = Σ αᵢ · ln(vᵢ(x))
//! ```
//!
//! where the `αᵢ` are rational numbers and `vᵢ(x)` are polynomials.
//!
//! # Algorithm (Bronstein, Chapter 2)
//!
//! 1. Compute `d' = d.derivative()`
//! 2. Build the resultant polynomial `R(t) = Res_x(d, c - t·d')` via
//!    evaluation at `deg(d)+1` points followed by Lagrange interpolation.
//! 3. Find rational roots of `R(t)`.
//! 4. For each rational root `αᵢ`: compute `vᵢ = gcd(d, c - αᵢ·d')`.
//! 5. Return `Σ αᵢ · ln(vᵢ)`.
//!
//! # Limitations
//!
//! Only rational roots of `R(t)` are handled. Algebraic (irrational) roots
//! would require algebraic number arithmetic, which is outside the current
//! scope.

use super::big_rational::BigRational;
use super::dense_poly::DensePolynomial;
use super::hermite::HermiteReduction;
use super::rational_fn::RationalFunction;
use num::traits::{One, Zero};

// ── Public types ──────────────────────────────────────────────────────────────

/// A single logarithmic term `coeff · ln(argument(x))`.
#[derive(Clone, Debug, PartialEq)]
pub struct LogTerm {
    /// The rational coefficient of the logarithm.
    pub coeff: BigRational,
    /// The polynomial argument of the logarithm.
    pub argument: DensePolynomial<BigRational>,
}

/// The logarithmic part of a rational function integral.
///
/// Represents `Σ αᵢ · ln(vᵢ(x))`.
#[derive(Clone, Debug)]
pub struct LogIntegral {
    /// The individual logarithmic terms.
    pub terms: Vec<LogTerm>,
}

impl LogIntegral {
    /// Create an empty (zero) log integral.
    pub fn zero() -> Self {
        LogIntegral { terms: Vec::new() }
    }

    /// Returns `true` if this integral has no terms (is zero).
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }
}

// ── Main entry points ─────────────────────────────────────────────────────────

/// Apply Rothstein-Trager to the log part of a Hermite reduction result.
///
/// # Example
///
/// ```
/// use thales::numeric::{DensePolynomial, BigRational, RationalFunction};
/// use thales::numeric::{hermite_reduce, rothstein_trager_from_hermite};
///
/// // ∫ 1/(x²-1) dx
/// let num = DensePolynomial::from_coeffs(vec![BigRational::from(1_i64)]);
/// let den = DensePolynomial::from_coeffs(vec![
///     BigRational::from(-1_i64),
///     BigRational::from(0_i64),
///     BigRational::from(1_i64),
/// ]);
/// let rf = RationalFunction::new(num, den);
/// let hr = hermite_reduce(&rf);
/// let log_int = rothstein_trager_from_hermite(&hr);
/// assert!(!log_int.is_zero());
/// ```
pub fn rothstein_trager_from_hermite(hr: &HermiteReduction<BigRational>) -> LogIntegral {
    if hr.log_part.is_zero() {
        return LogIntegral::zero();
    }
    let c = hr.log_part.numerator().clone();
    let d = hr.log_part.denominator().clone();
    rothstein_trager(&c, &d)
}

/// Compute `∫ (c/d) dx` logarithmically where `d` is squarefree.
///
/// Returns a [`LogIntegral`] containing only the terms with rational
/// coefficients. Terms whose coefficients are irrational are omitted.
///
/// # Panics
///
/// Panics if `d` is zero.
pub fn rothstein_trager(
    c: &DensePolynomial<BigRational>,
    d: &DensePolynomial<BigRational>,
) -> LogIntegral {
    assert!(
        !d.is_zero(),
        "rothstein_trager: denominator must be non-zero"
    );

    if c.is_zero() {
        return LogIntegral::zero();
    }

    let d_prime = d.derivative();

    // Degree bound for R(t): deg_t R ≤ deg_x d
    let n = d.degree().unwrap_or(0);

    // Evaluate R(tᵢ) = Res_x(d, c - tᵢ·d') at n+1 integer points
    let eval_points = evaluation_points(n + 1);
    let values: Vec<BigRational> = eval_points
        .iter()
        .map(|t| resultant_at(&t.clone(), c, d, &d_prime))
        .collect();

    // Interpolate to get R(t) as polynomial in t
    let r_poly = lagrange_interpolate(&eval_points, &values);

    if r_poly.is_zero() {
        // Degenerate case — no log terms recoverable
        return LogIntegral::zero();
    }

    // Find rational roots of R(t)
    let rational_roots = rational_roots_of(&r_poly);

    let mut terms = Vec::new();
    for alpha in rational_roots {
        // v = gcd(d, c - α·d')
        let alpha_d_prime = d_prime.scale(&alpha);
        let c_minus = c - &alpha_d_prime;
        let v = d.gcd(&c_minus);

        // Omit trivial (constant) GCDs — no logarithm to record
        if v.degree().unwrap_or(0) == 0 {
            continue;
        }
        terms.push(LogTerm {
            coeff: alpha,
            argument: v.make_monic(),
        });
    }

    LogIntegral { terms }
}

/// Convenience wrapper: integrate a squarefree rational function.
///
/// Applies Hermite reduction first, then Rothstein-Trager on the log part.
/// Only returns the logarithmic component; use `hermite_reduce` separately
/// for the rational part.
pub fn integrate_rational_log(rf: &RationalFunction<BigRational>) -> LogIntegral {
    let hr = super::hermite::hermite_reduce(rf);
    rothstein_trager_from_hermite(&hr)
}

// ── Resultant computation ─────────────────────────────────────────────────────

/// Evaluate `Res_x(d(x), c(x) - t₀·d'(x))` at a specific `t₀`.
///
/// This is the scalar resultant of two univariate polynomials in `x`.
fn resultant_at(
    t0: &BigRational,
    c: &DensePolynomial<BigRational>,
    d: &DensePolynomial<BigRational>,
    d_prime: &DensePolynomial<BigRational>,
) -> BigRational {
    // g(x) = c(x) - t₀ · d'(x)
    let t0_dp = d_prime.scale(t0);
    let g = c - &t0_dp;

    scalar_resultant(d, &g)
}

/// Compute the resultant of `f` and `g` as polynomials in `x` over `Q`.
///
/// Uses the subresultant PRS approach: the last non-zero pseudo-remainder
/// relates to the resultant via accumulated scaling factors.
/// Returns 0 if `f` and `g` share a common root (i.e., gcd has degree ≥ 1).
fn scalar_resultant(
    f: &DensePolynomial<BigRational>,
    g: &DensePolynomial<BigRational>,
) -> BigRational {
    if f.is_zero() || g.is_zero() {
        return BigRational::zero();
    }

    // If gcd has positive degree, resultant is 0
    let gcd = f.gcd(g);
    if gcd.degree().unwrap_or(0) >= 1 {
        return BigRational::zero();
    }

    // Use the Sylvester-matrix determinant via PRS for correctness.
    // Build the (m+n) × (m+n) Sylvester matrix and compute its determinant.
    let m = f.degree().unwrap_or(0);
    let n = g.degree().unwrap_or(0);

    if m == 0 {
        // Res(c, g) = c^deg(g) when f is a constant c
        let c = f.leading_coeff().unwrap().clone();
        return rational_pow(c, n);
    }
    if n == 0 {
        // Res(f, c) = c^deg(f) when g is a constant c
        let c = g.leading_coeff().unwrap().clone();
        return rational_pow(c, m);
    }

    sylvester_det(f, g)
}

/// Compute the determinant of the Sylvester matrix of `f` and `g`.
///
/// The Sylvester matrix has size (m+n) × (m+n) where m = deg(f), n = deg(g).
fn sylvester_det(
    f: &DensePolynomial<BigRational>,
    g: &DensePolynomial<BigRational>,
) -> BigRational {
    let m = f.degree().unwrap_or(0);
    let n = g.degree().unwrap_or(0);
    let size = m + n;

    // Build the Sylvester matrix row by row.
    // First n rows: shifts of f coefficients
    // Last m rows: shifts of g coefficients
    let mut mat: Vec<Vec<BigRational>> = vec![vec![BigRational::zero(); size]; size];

    for row in 0..n {
        for col in 0..=m {
            mat[row][row + col] = f.coeff(m - col);
        }
    }
    for row in 0..m {
        for col in 0..=n {
            mat[n + row][row + col] = g.coeff(n - col);
        }
    }

    gaussian_det(mat, size)
}

/// Compute the determinant of a square matrix via Gaussian elimination with
/// rational arithmetic (exact, no pivoting failures).
fn gaussian_det(mut mat: Vec<Vec<BigRational>>, size: usize) -> BigRational {
    let mut sign = BigRational::one();

    for col in 0..size {
        // Find pivot row
        let pivot = (col..size).find(|&r| !mat[r][col].is_zero());
        let pivot = match pivot {
            Some(p) => p,
            None => return BigRational::zero(),
        };

        if pivot != col {
            mat.swap(col, pivot);
            sign = -sign;
        }

        let inv_pivot = mat[col][col].recip();
        for row in (col + 1)..size {
            if mat[row][col].is_zero() {
                continue;
            }
            let factor = &mat[row][col] * &inv_pivot;
            for k in col..size {
                let sub = &factor * &mat[col][k].clone();
                mat[row][k] = &mat[row][k] - &sub;
            }
            mat[row][col] = BigRational::zero();
        }
    }

    let diag_prod = (0..size).fold(BigRational::one(), |acc, i| &acc * &mat[i][i]);
    &sign * &diag_prod
}

// ── Lagrange interpolation ────────────────────────────────────────────────────

/// Build evaluation points `[0, 1, -1, 2, -2, ...]` of length `n`.
fn evaluation_points(n: usize) -> Vec<BigRational> {
    let mut pts = Vec::with_capacity(n);
    pts.push(BigRational::zero());
    let mut k = 1i64;
    while pts.len() < n {
        pts.push(BigRational::from(k));
        if pts.len() < n {
            pts.push(BigRational::from(-k));
        }
        k += 1;
    }
    pts.truncate(n);
    pts
}

/// Lagrange interpolation: given `n+1` points and values, return the unique
/// polynomial of degree ≤ `n` passing through all of them.
fn lagrange_interpolate(xs: &[BigRational], ys: &[BigRational]) -> DensePolynomial<BigRational> {
    assert_eq!(xs.len(), ys.len(), "interpolation: xs and ys must match");
    let n = xs.len();
    let mut result = DensePolynomial::zero();

    for i in 0..n {
        if ys[i].is_zero() {
            continue;
        }
        // Build the Lagrange basis polynomial L_i(t)
        let mut basis = DensePolynomial::constant(BigRational::one());
        let mut denom = BigRational::one();

        for j in 0..n {
            if j == i {
                continue;
            }
            // basis *= (t - xⱼ)
            let factor = DensePolynomial::from_coeffs(vec![-xs[j].clone(), BigRational::one()]);
            basis = &basis * &factor;
            denom = &denom * &(&xs[i] - &xs[j]);
        }

        // Scale by y_i / denom
        let scale = &ys[i] / &denom;
        result = &result + &basis.scale(&scale);
    }

    result
}

// ── Rational root finding ─────────────────────────────────────────────────────

/// Find all rational roots of a polynomial `p` with `BigRational` coefficients.
///
/// Clears denominators to get an integer polynomial, applies `factor_integer_poly`,
/// and extracts roots from linear factors.
fn rational_roots_of(p: &DensePolynomial<BigRational>) -> Vec<BigRational> {
    if p.is_zero() {
        return vec![];
    }

    // Use poly_equation_solver which handles monic conversion properly
    let roots = super::poly_equation_solver::roots_with_multiplicity(p);
    let mut result = Vec::new();

    for rwm in roots {
        // Extract BigRational from Expr
        match rwm.root.as_ref() {
            super::expr::Expr::Integer(n) => {
                result.push(BigRational::from_integer(n.clone()));
            }
            super::expr::Expr::Rational(r) => {
                result.push(r.clone());
            }
            _ => {} // Skip non-rational roots
        }
    }

    result
}

// ── Arithmetic helpers ────────────────────────────────────────────────────────

/// Raise a `BigRational` to a non-negative integer power.
fn rational_pow(base: BigRational, exp: usize) -> BigRational {
    if exp == 0 {
        return BigRational::one();
    }
    let mut result = BigRational::one();
    let mut b = base;
    let mut e = exp;
    while e > 0 {
        if e % 2 == 1 {
            result = &result * &b;
        }
        b = &b * &b;
        e /= 2;
    }
    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{hermite_reduce, BigRational, DensePolynomial, RationalFunction};

    type P = DensePolynomial<BigRational>;

    fn r(n: i64) -> BigRational {
        BigRational::from(n)
    }

    fn rat(n: i64, d: i64) -> BigRational {
        BigRational::from_i64(n, d)
    }

    fn poly(coeffs: &[i64]) -> P {
        P::from_coeffs(coeffs.iter().map(|&c| r(c)).collect())
    }

    // ── evaluation_points ────────────────────────────────────────────────────

    #[test]
    fn test_evaluation_points_count() {
        let pts = evaluation_points(5);
        assert_eq!(pts.len(), 5);
    }

    #[test]
    fn test_evaluation_points_distinct() {
        let pts = evaluation_points(7);
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                assert_ne!(pts[i], pts[j], "evaluation points must be distinct");
            }
        }
    }

    // ── lagrange_interpolate ─────────────────────────────────────────────────

    #[test]
    fn test_interpolate_constant() {
        let xs = vec![r(0), r(1), r(-1)];
        let ys = vec![r(5), r(5), r(5)];
        let p = lagrange_interpolate(&xs, &ys);
        assert_eq!(p.degree(), Some(0));
        assert_eq!(p.coeff(0), r(5));
    }

    #[test]
    fn test_interpolate_linear() {
        // f(t) = 2t + 1
        let xs = vec![r(0), r(1)];
        let ys = vec![r(1), r(3)];
        let p = lagrange_interpolate(&xs, &ys);
        assert_eq!(p.coeff(0), r(1));
        assert_eq!(p.coeff(1), r(2));
    }

    #[test]
    fn test_interpolate_quadratic() {
        // f(t) = t² (values: 0, 1, 4)
        let xs = vec![r(0), r(1), r(2)];
        let ys = vec![r(0), r(1), r(4)];
        let p = lagrange_interpolate(&xs, &ys);
        assert_eq!(p.coeff(0), r(0));
        assert_eq!(p.coeff(1), r(0));
        assert_eq!(p.coeff(2), r(1));
    }

    // ── scalar_resultant ─────────────────────────────────────────────────────

    #[test]
    fn test_resultant_shared_root() {
        // f = x - 1, g = x - 1 → Res = 0 (share root)
        let f = poly(&[-1, 1]);
        let g = poly(&[-1, 1]);
        let res = scalar_resultant(&f, &g);
        assert!(res.is_zero());
    }

    #[test]
    fn test_resultant_coprime_linear() {
        // f = x - 1, g = x - 2 → Res = f(2) = 2-1 = 1
        let f = poly(&[-1, 1]);
        let g = poly(&[-2, 1]);
        let res = scalar_resultant(&f, &g);
        assert!(!res.is_zero());
    }

    #[test]
    fn test_resultant_x2_minus1_and_linear() {
        // f = x²-1 = (x-1)(x+1), g = x-1 → Res = 0
        let f = poly(&[-1, 0, 1]);
        let g = poly(&[-1, 1]);
        let res = scalar_resultant(&f, &g);
        assert!(res.is_zero());
    }

    // ── rothstein_trager core ────────────────────────────────────────────────

    #[test]
    fn test_rt_1_over_x2_minus_1() {
        // ∫ 1/(x²-1) dx = ½ ln(x-1) - ½ ln(x+1)
        // c = 1, d = x²-1 = (x-1)(x+1)
        let c = poly(&[1]);
        let d = poly(&[-1, 0, 1]);
        let log_int = rothstein_trager(&c, &d);

        assert_eq!(log_int.terms.len(), 2, "expected 2 log terms");

        // Sort terms by coefficient for deterministic comparison
        let mut terms = log_int.terms.clone();
        terms.sort_by(|a, b| a.coeff.partial_cmp(&b.coeff).unwrap());

        // First term: -½ · ln(x+1)
        assert_eq!(terms[0].coeff, rat(-1, 2));
        // Second term: ½ · ln(x-1)
        assert_eq!(terms[1].coeff, rat(1, 2));

        // Verify arguments are monic linear factors
        assert_eq!(terms[0].argument.degree(), Some(1));
        assert_eq!(terms[1].argument.degree(), Some(1));
    }

    #[test]
    fn test_rt_1_over_x2_minus_4() {
        // ∫ 1/(x²-4) dx = ¼ ln(x-2) - ¼ ln(x+2)
        let c = poly(&[1]);
        let d = poly(&[-4, 0, 1]);
        let log_int = rothstein_trager(&c, &d);

        assert_eq!(log_int.terms.len(), 2, "expected 2 log terms");

        let mut terms = log_int.terms.clone();
        terms.sort_by(|a, b| a.coeff.partial_cmp(&b.coeff).unwrap());

        assert_eq!(terms[0].coeff, rat(-1, 4));
        assert_eq!(terms[1].coeff, rat(1, 4));
    }

    #[test]
    fn test_rt_zero_numerator() {
        let c = poly(&[]); // zero
        let d = poly(&[-1, 0, 1]);
        let log_int = rothstein_trager(&c, &d);
        assert!(log_int.is_zero());
    }

    #[test]
    fn test_rt_1_over_linear() {
        // ∫ 1/(x-3) dx = ln(x-3)
        let c = poly(&[1]);
        let d = poly(&[-3, 1]);
        let log_int = rothstein_trager(&c, &d);
        assert_eq!(log_int.terms.len(), 1);
        assert_eq!(log_int.terms[0].coeff, r(1));
        // argument should be (x-3), monic
        assert_eq!(log_int.terms[0].argument.coeff(0), r(-3));
        assert_eq!(log_int.terms[0].argument.coeff(1), r(1));
    }

    // ── integrate_rational_log convenience wrapper ───────────────────────────

    #[test]
    fn test_integrate_rational_log_squarefree() {
        // ∫ 1/(x²-1) dx via the convenience wrapper
        let num = poly(&[1]);
        let den = poly(&[-1, 0, 1]);
        let rf = RationalFunction::new(num, den);
        let log_int = integrate_rational_log(&rf);
        assert_eq!(log_int.terms.len(), 2);
    }

    // ── rothstein_trager_from_hermite ────────────────────────────────────────

    #[test]
    fn test_from_hermite_zero_log_part() {
        // ∫ 1/x² dx — Hermite gives purely rational part, zero log part
        let num = poly(&[1]);
        let den = poly(&[0, 0, 1]); // x²
        let rf = RationalFunction::new(num, den);
        let hr = hermite_reduce(&rf);
        let log_int = rothstein_trager_from_hermite(&hr);
        assert!(log_int.is_zero());
    }

    #[test]
    fn test_from_hermite_mixed() {
        // ∫ 1/((x-1)²·(x+2)) dx — Hermite gives rational + log parts
        // log part should have squarefree denominator (x-1)(x+2) or similar
        let xa = poly(&[-1, 1]); // x-1
        let xb = poly(&[2, 1]); // x+2
        let den = &(&xa * &xa) * &xb; // (x-1)²(x+2)
        let num = poly(&[1]);
        let rf = RationalFunction::new(num, den);
        let hr = hermite_reduce(&rf);
        let log_int = rothstein_trager_from_hermite(&hr);
        // Should produce some log terms (exact coefficients depend on Hermite output)
        // We just verify it runs without panic and produces valid structure
        for term in &log_int.terms {
            assert!(term.argument.degree().unwrap_or(0) >= 1);
        }
    }
}
