//! Polynomial equation solver with root multiplicity tracking.
//!
//! Solves univariate polynomial equations over the rationals by combining
//! squarefree factorization with full integer polynomial factorization.
//! Returns roots as [`Arc<Expr>`] values within a [`SolutionSet`].
//!
//! # Algorithm
//!
//! 1. Dispatch on degree: zero poly → all reals, constant → empty.
//! 2. Degree 1: closed-form rational root.
//! 3. Degree 2+: delegate to [`factor_integer_poly`] which runs Yun's
//!    squarefree decomposition followed by Zassenhaus recombination,
//!    yielding `(irreducible_factor, multiplicity)` pairs.
//! 4. Per factor: extract roots from linear and quadratic factors;
//!    higher-degree irreducibles are silently dropped (no real roots
//!    expressible without complex or algebraic extensions).
//!
//! # Limitations
//!
//! - Complex roots are not returned (no `Complex` variant in [`Expr`]).
//! - Cubic/quartic irreducible factors yield no roots (Cardano/Ferrari
//!   formulas not implemented).

use super::big_rational::BigRational;
use super::dense_poly::DensePolynomial;
use super::expr::Expr;
use super::solution_set::SolutionSet;
use super::zassenhaus::factor_integer_poly;
use num::traits::Zero;
use std::sync::Arc;

// ── Public types ─────────────────────────────────────────────────────────────

/// A root of a polynomial together with its algebraic multiplicity.
#[derive(Clone, Debug, PartialEq)]
pub struct RootWithMultiplicity {
    /// The root value as an expression.
    pub root: Arc<Expr>,
    /// The algebraic multiplicity (≥ 1).
    pub multiplicity: usize,
}

// ── Main entry point ─────────────────────────────────────────────────────────

/// Solve `poly(x) = 0` and return the solution set.
///
/// - Zero polynomial → all reals (`SolutionSet::all_reals()`).
/// - Non-zero constant → empty set (`SolutionSet::Empty`).
/// - Degree ≥ 1 → finite set of rational real roots (complex and irrational
///   roots from irreducible cubics/quartics are omitted).
///
/// # Examples
///
/// ```
/// use thales::numeric::{DensePolynomial, BigRational, solve_polynomial, SolutionSet};
/// use num::traits::One;
///
/// // x - 2 = 0  →  {2}
/// let p = DensePolynomial::from_coeffs(vec![
///     BigRational::from(-2_i64),
///     BigRational::one(),
/// ]);
/// let sol = solve_polynomial(&p);
/// assert!(matches!(sol, SolutionSet::Finite(_)));
/// ```
pub fn solve_polynomial(poly: &DensePolynomial<BigRational>) -> SolutionSet {
    if poly.is_zero() {
        return SolutionSet::all_reals();
    }

    let deg = match poly.degree() {
        Some(d) => d,
        None => return SolutionSet::Empty,
    };

    if deg == 0 {
        return SolutionSet::Empty;
    }

    let roots = roots_with_multiplicity(poly);
    SolutionSet::from_values(roots.into_iter().map(|r| r.root))
}

/// Extract roots with multiplicities from a polynomial.
///
/// Returns one [`RootWithMultiplicity`] per distinct rational real root.
pub fn roots_with_multiplicity(poly: &DensePolynomial<BigRational>) -> Vec<RootWithMultiplicity> {
    if poly.is_zero() || poly.degree().map_or(true, |d| d == 0) {
        return vec![];
    }

    let factors = factor_integer_poly(poly);
    let mut roots = Vec::new();

    for (factor, multiplicity) in &factors {
        extract_roots_from_factor(factor, *multiplicity, &mut roots);
    }

    roots
}

// ── Root extraction per factor degree ────────────────────────────────────────

/// Dispatch root extraction by degree of an irreducible factor.
fn extract_roots_from_factor(
    factor: &DensePolynomial<BigRational>,
    multiplicity: usize,
    out: &mut Vec<RootWithMultiplicity>,
) {
    match factor.degree() {
        Some(1) => {
            if let Some(root) = root_of_linear(factor) {
                out.push(RootWithMultiplicity { root, multiplicity });
            }
        }
        Some(2) => {
            let quadratic_roots = roots_of_quadratic(factor);
            for root in quadratic_roots {
                out.push(RootWithMultiplicity { root, multiplicity });
            }
        }
        // Degree 0, None (zero poly), or degree ≥ 3: no extractable real root
        _ => {}
    }
}

/// Extract the root of a monic linear factor `x + b` (or `ax + b`).
///
/// Root = `-b / a`.
fn root_of_linear(factor: &DensePolynomial<BigRational>) -> Option<Arc<Expr>> {
    let a = factor.coeff(1);
    let b = factor.coeff(0);

    if a.is_zero() {
        return None;
    }

    // root = -b / a
    let neg_b = -b;
    let root_rational = neg_b / a;
    Some(bigrational_to_expr(root_rational))
}

/// Extract real rational roots of a quadratic `ax² + bx + c`.
///
/// Computes discriminant D = b² − 4ac:
/// - D < 0: no real roots (complex, not returned).
/// - D = 0: one rational root `−b / (2a)`.
/// - D > 0: two real roots if D is a perfect rational square, otherwise none
///   (irrational surd not expressible in current [`Expr`]).
fn roots_of_quadratic(factor: &DensePolynomial<BigRational>) -> Vec<Arc<Expr>> {
    let a = factor.coeff(2);
    let b = factor.coeff(1);
    let c = factor.coeff(0);

    if a.is_zero() {
        return vec![];
    }

    // D = b² - 4ac
    let b_sq = &b * &b;
    let four = BigRational::from(4_i64);
    let four_ac = &(&four * &a) * &c;
    let discriminant = b_sq - four_ac;

    if discriminant.is_negative() {
        return vec![];
    }

    let two = BigRational::from(2_i64);
    let two_a = &two * &a;

    if discriminant.is_zero() {
        // One repeated root: -b / (2a)
        let root = (-b) / two_a;
        return vec![bigrational_to_expr(root)];
    }

    // D > 0: check whether D is a perfect rational square
    match rational_sqrt(&discriminant) {
        Some(sqrt_d) => {
            // Two roots: (-b ± sqrt_d) / (2a)
            let neg_b = -b;
            let r1 = (neg_b.clone() + sqrt_d.clone()) / two_a.clone();
            let r2 = (neg_b - sqrt_d) / two_a;
            vec![bigrational_to_expr(r1), bigrational_to_expr(r2)]
        }
        None => {
            // Irrational roots: not expressible, skip
            vec![]
        }
    }
}

// ── Arithmetic helpers ────────────────────────────────────────────────────────

/// Convert a [`BigRational`] to the canonical [`Expr`] form.
///
/// Integer-valued rationals become `Expr::Integer`; proper fractions
/// become `Expr::Rational`.
fn bigrational_to_expr(r: BigRational) -> Arc<Expr> {
    if r.is_integer() {
        // Safe: integer-valued rational has denom = 1
        let n = r.numer().to_i64().unwrap_or(0);
        Expr::int(n)
    } else {
        let n = r.numer().to_i64().unwrap_or(0);
        let d = r.denom().to_i64().unwrap_or(1);
        Expr::rational(n, d)
    }
}

/// Attempt to compute the exact rational square root of `r`.
///
/// Returns `Some(s)` if `s² = r` exactly, otherwise `None`.
fn rational_sqrt(r: &BigRational) -> Option<BigRational> {
    if r.is_zero() {
        return Some(BigRational::zero());
    }
    if r.is_negative() {
        return None;
    }

    // r = n/d; sqrt(r) = sqrt(n)/sqrt(d) iff both are perfect squares
    let n_big = r.numer().to_bigint();
    let d_big = r.denom().to_bigint();

    let sqrt_n = n_big.sqrt();
    let sqrt_d = d_big.sqrt();

    if &sqrt_n * &sqrt_n == n_big && &sqrt_d * &sqrt_d == d_big {
        use super::small_int::SmallInt;
        let sn = SmallInt::from(sqrt_n);
        let sd = SmallInt::from(sqrt_d);
        Some(BigRational::new(sn, sd))
    } else {
        None
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    type P = DensePolynomial<BigRational>;

    fn rat(n: i64) -> BigRational {
        BigRational::from(n)
    }

    fn poly(coeffs: &[i64]) -> P {
        P::from_coeffs(coeffs.iter().map(|&c| rat(c)).collect())
    }

    /// Collect integer values from a Finite solution set.
    fn finite_i64s(sol: &SolutionSet) -> Vec<i64> {
        match sol {
            SolutionSet::Finite(vals) => vals
                .iter()
                .filter_map(|e| match e.as_ref() {
                    Expr::Integer(n) => n.to_i64(),
                    _ => None,
                })
                .collect(),
            _ => vec![],
        }
    }

    // ── Zero and constant ────────────────────────────────────────────────────

    #[test]
    fn test_zero_poly_is_all_reals() {
        let p = P::zero();
        let sol = solve_polynomial(&p);
        assert_eq!(sol, SolutionSet::all_reals());
    }

    #[test]
    fn test_constant_poly_is_empty() {
        let p = poly(&[5]);
        let sol = solve_polynomial(&p);
        assert!(sol.is_empty());
    }

    // ── Linear ──────────────────────────────────────────────────────────────

    #[test]
    fn test_linear_integer_root() {
        // x - 3 = 0  →  {3}
        let p = poly(&[-3, 1]);
        let sol = solve_polynomial(&p);
        let vals = finite_i64s(&sol);
        assert_eq!(vals, vec![3]);
    }

    #[test]
    fn test_linear_negative_root() {
        // x + 7 = 0  →  {-7}
        let p = poly(&[7, 1]);
        let sol = solve_polynomial(&p);
        let vals = finite_i64s(&sol);
        assert_eq!(vals, vec![-7]);
    }

    // ── Quadratic ───────────────────────────────────────────────────────────

    #[test]
    fn test_quadratic_two_integer_roots() {
        // (x-1)(x+1) = x² - 1  →  {1, -1}
        let p = poly(&[-1, 0, 1]);
        let sol = solve_polynomial(&p);
        let mut vals = finite_i64s(&sol);
        vals.sort_unstable();
        assert_eq!(vals, vec![-1, 1]);
    }

    #[test]
    fn test_quadratic_perfect_square() {
        // (x-2)² = x² - 4x + 4  →  {2} with mult 2
        let p = poly(&[4, -4, 1]);
        let roots = roots_with_multiplicity(&p);
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].multiplicity, 2);
        match roots[0].root.as_ref() {
            Expr::Integer(n) => assert_eq!(n.to_i64(), Some(2)),
            _ => panic!("expected integer root"),
        }
    }

    #[test]
    fn test_quadratic_no_real_roots() {
        // x² + 1: irreducible, no real roots
        let p = poly(&[1, 0, 1]);
        let sol = solve_polynomial(&p);
        // Empty because the only factor (x²+1) has complex roots
        assert!(sol.is_empty());
    }

    // ── Multiplicity tracking ────────────────────────────────────────────────

    #[test]
    fn test_cubic_with_multiplicity() {
        // (x-1)³ = x³ - 3x² + 3x - 1
        let p = poly(&[-1, 3, -3, 1]);
        let roots = roots_with_multiplicity(&p);
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].multiplicity, 3);
        match roots[0].root.as_ref() {
            Expr::Integer(n) => assert_eq!(n.to_i64(), Some(1)),
            _ => panic!("expected integer root 1"),
        }
    }

    #[test]
    fn test_mixed_multiplicities() {
        // (x-1)²(x+1)³ = (x²-2x+1)(x³+3x²+3x+1)
        //              = x⁵ + x⁴ - 2x³ - 2x² + x + 1... compute manually
        // Coefficients of (x-1)²(x+1)³:
        // (x-1)² = x²-2x+1
        // (x+1)³ = x³+3x²+3x+1
        // product: see below
        let p1 = poly(&[1, -2, 1]); // (x-1)²
        let p3 = poly(&[1, 3, 3, 1]); // (x+1)³
        let product = &p1 * &p3;

        let roots = roots_with_multiplicity(&product);
        let mut roots_sorted = roots.clone();
        roots_sorted.sort_by_key(|r| match r.root.as_ref() {
            Expr::Integer(n) => n.to_i64().unwrap_or(0),
            _ => 0,
        });

        assert_eq!(roots_sorted.len(), 2, "expected 2 distinct roots");

        // Find root -1 with mult 3 and root 1 with mult 2
        let neg1 = roots_sorted
            .iter()
            .find(|r| matches!(r.root.as_ref(), Expr::Integer(n) if n.to_i64() == Some(-1)));
        let pos1 = roots_sorted
            .iter()
            .find(|r| matches!(r.root.as_ref(), Expr::Integer(n) if n.to_i64() == Some(1)));

        assert!(neg1.is_some(), "root -1 not found");
        assert!(pos1.is_some(), "root 1 not found");
        assert_eq!(neg1.unwrap().multiplicity, 3);
        assert_eq!(pos1.unwrap().multiplicity, 2);
    }

    #[test]
    fn test_x4_minus_1() {
        // x⁴ - 1 = (x-1)(x+1)(x²+1)  →  real roots {1, -1}
        let p = poly(&[-1, 0, 0, 0, 1]);
        let sol = solve_polynomial(&p);
        let mut vals = finite_i64s(&sol);
        vals.sort_unstable();
        assert_eq!(vals, vec![-1, 1]);
    }

    // ── Rational roots ───────────────────────────────────────────────────────

    #[test]
    fn test_linear_rational_root() {
        // 2x - 1 = 0  →  {1/2}
        let p = P::from_coeffs(vec![rat(-1), rat(2)]);
        let roots = roots_with_multiplicity(&p);
        assert_eq!(roots.len(), 1);
        match roots[0].root.as_ref() {
            Expr::Rational(r) => {
                assert_eq!(r.numer().to_i64(), Some(1));
                assert_eq!(r.denom().to_i64(), Some(2));
            }
            _ => panic!("expected Rational(1/2), got {:?}", roots[0].root),
        }
    }

    // ── bigrational_to_expr helper ───────────────────────────────────────────

    #[test]
    fn test_bigrational_to_expr_integer() {
        let r = BigRational::from(5_i64);
        let e = bigrational_to_expr(r);
        assert!(matches!(e.as_ref(), Expr::Integer(_)));
    }

    #[test]
    fn test_bigrational_to_expr_rational() {
        let r = BigRational::from_i64(1, 3);
        let e = bigrational_to_expr(r);
        assert!(matches!(e.as_ref(), Expr::Rational(_)));
    }

    // ── rational_sqrt helper ─────────────────────────────────────────────────

    #[test]
    fn test_rational_sqrt_perfect_square() {
        let r = BigRational::from(9_i64);
        let s = rational_sqrt(&r);
        assert!(s.is_some());
        assert_eq!(s.unwrap(), BigRational::from(3_i64));
    }

    #[test]
    fn test_rational_sqrt_not_perfect_square() {
        let r = BigRational::from(2_i64);
        assert!(rational_sqrt(&r).is_none());
    }

    #[test]
    fn test_rational_sqrt_zero() {
        let r = BigRational::zero();
        let s = rational_sqrt(&r);
        assert!(s.is_some());
        assert!(s.unwrap().is_zero());
    }

    #[test]
    fn test_rational_sqrt_negative() {
        let r = BigRational::from(-1_i64);
        assert!(rational_sqrt(&r).is_none());
    }
}
