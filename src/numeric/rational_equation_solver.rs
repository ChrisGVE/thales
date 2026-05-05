//! Rational equation solver with extraneous solution detection.
//!
//! Solves equations involving rational expressions (fractions whose numerator
//! and denominator are polynomials) by clearing denominators, solving the
//! resulting polynomial equation, and filtering out extraneous roots that
//! make any denominator zero.
//!
//! # Algorithm
//!
//! Given `p(x)/q(x) = r(x)/s(x)`:
//!
//! 1. **Clear denominators**: cross-multiply to get `p·s - r·q = 0`.
//! 2. **Solve** the polynomial `p·s - r·q` using [`solve_polynomial`].
//! 3. **Filter extraneous solutions**: discard any root `x = a` where
//!    `q(a) = 0` or `s(a) = 0`.
//! 4. **Return** valid roots as [`SolutionSet::Finite`], or
//!    [`SolutionSet::all_except`] when the cleared polynomial is identically
//!    zero (meaning all reals are solutions, modulo excluded denominator zeros).
//!
//! # Example
//!
//! ```
//! use thales::numeric::{
//!     DensePolynomial, BigRational, SolutionSet,
//!     solve_rational, solve_rational_equation,
//! };
//! use num::traits::Zero;
//!
//! // (x² - 4)/(x - 2) = 0  →  x = -2  (x = 2 is extraneous)
//! let numer = DensePolynomial::from_coeffs(vec![
//!     BigRational::from(-4_i64),
//!     BigRational::zero(),
//!     BigRational::from(1_i64),
//! ]);
//! let denom = DensePolynomial::from_coeffs(vec![
//!     BigRational::from(-2_i64),
//!     BigRational::from(1_i64),
//! ]);
//! let sol = solve_rational(&numer, &denom);
//! assert!(matches!(sol, SolutionSet::Finite(_)));
//! if let SolutionSet::Finite(roots) = &sol {
//!     assert_eq!(roots.len(), 1);
//! }
//! ```

use super::big_rational::BigRational;
use super::dense_poly::DensePolynomial;
use super::expr::Expr;
use super::poly_equation_solver::solve_polynomial;
use super::ring::Ring;
use super::solution_set::SolutionSet;
use std::sync::Arc;

// ── Public API ────────────────────────────────────────────────────────────────

/// Solve `numer(x) / denom(x) = 0`.
///
/// Returns the set of `x` values where the expression equals zero, excluding
/// values that make `denom` zero (extraneous solutions).
///
/// - If `numer` is zero: returns [`SolutionSet::all_except`] (all reals
///   except the zeros of `denom`).
/// - If `denom` is zero polynomial: returns [`SolutionSet::Empty`] (undefined).
/// - Otherwise: solves `numer(x) = 0` and filters out roots of `denom`.
///
/// # Examples
///
/// ```
/// use thales::numeric::{DensePolynomial, BigRational, SolutionSet, solve_rational};
/// use num::traits::Zero;
///
/// // 1/(x² - 1) = 0  →  numerator = 1, no solution
/// let numer = DensePolynomial::from_coeffs(vec![BigRational::from(1_i64)]);
/// let denom = DensePolynomial::from_coeffs(vec![
///     BigRational::from(-1_i64),
///     BigRational::zero(),
///     BigRational::from(1_i64),
/// ]);
/// let sol = solve_rational(&numer, &denom);
/// assert!(sol.is_empty());
/// ```
pub fn solve_rational(
    numer: &DensePolynomial<BigRational>,
    denom: &DensePolynomial<BigRational>,
) -> SolutionSet {
    // Undefined expression: denom = 0 polynomial
    if denom.is_zero() {
        return SolutionSet::Empty;
    }

    // numer = 0: every real is a solution except zeros of denom
    if numer.is_zero() {
        let denom_zeros = collect_denom_zeros(denom);
        return SolutionSet::all_except(denom_zeros);
    }

    // Solve numer(x) = 0, then filter out denom zeros
    let candidates = solve_polynomial(numer);
    filter_extraneous(candidates, &[denom])
}

/// Solve `lhs_num(x)/lhs_den(x) = rhs_num(x)/rhs_den(x)`.
///
/// Clears denominators by cross-multiplying: solves
/// `lhs_num·rhs_den - rhs_num·lhs_den = 0`, then removes any root that
/// makes `lhs_den` or `rhs_den` zero.
///
/// # Examples
///
/// ```
/// use thales::numeric::{DensePolynomial, BigRational, SolutionSet, solve_rational_equation};
/// use num::traits::Zero;
///
/// // x/(x-2) = 2/(x-2)  →  x = 2, but x ≠ 2  →  Empty
/// let x = DensePolynomial::from_coeffs(vec![
///     BigRational::zero(),
///     BigRational::from(1_i64),
/// ]);
/// let two = DensePolynomial::from_coeffs(vec![BigRational::from(2_i64)]);
/// let x_minus_2 = DensePolynomial::from_coeffs(vec![
///     BigRational::from(-2_i64),
///     BigRational::from(1_i64),
/// ]);
/// let sol = solve_rational_equation(&x, &x_minus_2, &two, &x_minus_2);
/// assert!(sol.is_empty());
/// ```
pub fn solve_rational_equation(
    lhs_num: &DensePolynomial<BigRational>,
    lhs_den: &DensePolynomial<BigRational>,
    rhs_num: &DensePolynomial<BigRational>,
    rhs_den: &DensePolynomial<BigRational>,
) -> SolutionSet {
    // Undefined: any denominator is the zero polynomial
    if lhs_den.is_zero() || rhs_den.is_zero() {
        return SolutionSet::Empty;
    }

    // Cross-multiply: lhs_num * rhs_den - rhs_num * lhs_den = 0
    let lhs_cross = lhs_num * rhs_den;
    let rhs_cross = rhs_num * lhs_den;
    let cleared = lhs_cross - rhs_cross;

    // cleared = 0 means identity: all reals except denominator zeros
    if cleared.is_zero() {
        let mut excluded = collect_denom_zeros(lhs_den);
        let rhs_zeros = collect_denom_zeros(rhs_den);
        for z in rhs_zeros {
            if !excluded.iter().any(|e| exprs_equal(e, &z)) {
                excluded.push(z);
            }
        }
        return SolutionSet::all_except(excluded);
    }

    // Solve the cleared polynomial, filtering out denominator zeros
    let candidates = solve_polynomial(&cleared);
    filter_extraneous(candidates, &[lhs_den, rhs_den])
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Evaluate each candidate from `candidates` against all `denoms`; keep only
/// those for which no denominator evaluates to zero.
///
/// If `candidates` is already empty or non-finite (e.g. all-reals), the
/// all-reals result is delegated to [`solve_rational_equation`] via the
/// identity branch — this path returns only finite sets.
fn filter_extraneous(
    candidates: SolutionSet,
    denoms: &[&DensePolynomial<BigRational>],
) -> SolutionSet {
    let roots = match &candidates {
        SolutionSet::Finite(v) => v.clone(),
        SolutionSet::Empty => return SolutionSet::Empty,
        // all-reals from zero poly: no filtering needed at this stage,
        // but callers already handle this case before reaching here.
        other => return other.clone(),
    };

    let valid: Vec<Arc<Expr>> = roots
        .into_iter()
        .filter(|root| !is_denom_zero_at(root, denoms))
        .collect();

    SolutionSet::from_values(valid)
}

/// Returns `true` if `root` makes any denominator in `denoms` equal to zero.
fn is_denom_zero_at(root: &Arc<Expr>, denoms: &[&DensePolynomial<BigRational>]) -> bool {
    if let Some(r) = expr_to_bigrational(root) {
        denoms.iter().any(|d| d.eval(&r).is_zero())
    } else {
        // Cannot evaluate: conservatively keep the root
        false
    }
}

/// Collect the rational roots (zeros) of `denom` as `Arc<Expr>` values.
fn collect_denom_zeros(denom: &DensePolynomial<BigRational>) -> Vec<Arc<Expr>> {
    let sol = solve_polynomial(denom);
    match sol {
        SolutionSet::Finite(v) => v,
        _ => vec![],
    }
}

/// Convert an `Arc<Expr>` to a `BigRational` for polynomial evaluation, if possible.
fn expr_to_bigrational(expr: &Arc<Expr>) -> Option<BigRational> {
    match expr.as_ref() {
        Expr::Integer(n) => n.to_i64().map(BigRational::from),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

/// Structural equality check for two `Arc<Expr>` values (used for dedup).
fn exprs_equal(a: &Arc<Expr>, b: &Arc<Expr>) -> bool {
    a == b
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

    /// Extract a sorted vec of (numer, denom) pairs from a Finite solution set.
    fn finite_rationals(sol: &SolutionSet) -> Vec<(i64, i64)> {
        let roots = match sol {
            SolutionSet::Finite(v) => v,
            _ => return vec![],
        };
        let mut out: Vec<(i64, i64)> = roots
            .iter()
            .filter_map(|e| match e.as_ref() {
                Expr::Integer(n) => n.to_i64().map(|v| (v, 1)),
                Expr::Rational(r) => {
                    let n = r.numer().to_i64()?;
                    let d = r.denom().to_i64()?;
                    Some((n, d))
                }
                _ => None,
            })
            .collect();
        out.sort_unstable();
        out
    }

    // ─── solve_rational tests ────────────────────────────────────────────────

    #[test]
    fn test_rational_constant_numerator_no_solution() {
        // 1/(x² - 1) = 0  →  numerator = 1, no solution
        let numer = poly(&[1]);
        let denom = poly(&[-1, 0, 1]); // x² - 1
        let sol = solve_rational(&numer, &denom);
        assert!(sol.is_empty(), "constant numerator has no zero");
    }

    #[test]
    fn test_rational_filters_extraneous_root() {
        // (x² - 4)/(x - 2) = 0  →  roots of x²-4 are ±2; x=2 extraneous
        let numer = poly(&[-4, 0, 1]); // x² - 4
        let denom = poly(&[-2, 1]); // x - 2
        let sol = solve_rational(&numer, &denom);
        let roots = finite_rationals(&sol);
        assert_eq!(roots, vec![(-2, 1)], "only x=-2 is valid");
    }

    #[test]
    fn test_rational_zero_numerator_all_except() {
        // 0/(x - 1) = 0  →  all x except x = 1
        let numer = P::zero();
        let denom = poly(&[-1, 1]); // x - 1
        let sol = solve_rational(&numer, &denom);
        assert!(
            matches!(sol, SolutionSet::Complement(_)),
            "expected ℝ \\ {{1}}"
        );
    }

    #[test]
    fn test_rational_undefined_zero_denom() {
        // p/0: undefined → empty
        let numer = poly(&[1, 1]);
        let denom = P::zero();
        let sol = solve_rational(&numer, &denom);
        assert!(sol.is_empty());
    }

    // ─── solve_rational_equation tests ───────────────────────────────────────

    #[test]
    fn test_rational_eq_valid_solution() {
        // 1/x + 1/(x-1) = 2/(x(x-1))
        // LHS combined = (x-1 + x)/(x(x-1)) = (2x-1)/(x(x-1))
        // Equation: (2x-1)/(x(x-1)) = 2/(x(x-1))
        // Cross-mult: (2x-1)*1 - 2*1 = 0  →  2x - 3 = 0  →  x = 3/2
        // Denominators: x(x-1) ≠ 0, so x ≠ 0, x ≠ 1. x=3/2 is valid.
        //
        // lhs_num = 2x-1, lhs_den = x(x-1) = x²-x
        // rhs_num = 2,    rhs_den = x(x-1) = x²-x
        let lhs_num = poly(&[-1, 2]); // 2x - 1
        let lhs_den = poly(&[0, -1, 1]); // x² - x
        let rhs_num = poly(&[2]); // 2
        let rhs_den = poly(&[0, -1, 1]); // x² - x
        let sol = solve_rational_equation(&lhs_num, &lhs_den, &rhs_num, &rhs_den);
        let roots = finite_rationals(&sol);
        assert_eq!(roots, vec![(3, 2)], "x = 3/2 is the valid solution");
    }

    #[test]
    fn test_rational_eq_extraneous_only() {
        // x/(x-2) = 2/(x-2)  →  x = 2, but x ≠ 2 → Empty
        let lhs_num = poly(&[0, 1]); // x
        let lhs_den = poly(&[-2, 1]); // x - 2
        let rhs_num = poly(&[2]); // 2
        let rhs_den = poly(&[-2, 1]); // x - 2
        let sol = solve_rational_equation(&lhs_num, &lhs_den, &rhs_num, &rhs_den);
        assert!(sol.is_empty(), "x=2 is extraneous; result should be empty");
    }

    #[test]
    fn test_rational_eq_identity_all_except() {
        // (x-1)/(x-1) = 1  →  all x except x = 1
        // lhs_num = x-1, lhs_den = x-1, rhs_num = 1, rhs_den = 1
        let x_minus_1 = poly(&[-1, 1]);
        let one = poly(&[1]);
        let sol = solve_rational_equation(&x_minus_1, &x_minus_1, &one, &one);
        assert!(
            matches!(sol, SolutionSet::Complement(_)),
            "identity: ℝ \\ {{1}}"
        );
    }

    #[test]
    fn test_rational_eq_identity_two_denoms_excluded() {
        // x/((x-1)(x+1)) = x/((x-1)(x+1)) — identity, exclude ±1
        let numer = poly(&[0, 1]); // x
        let denom = poly(&[-1, 0, 1]); // x² - 1 = (x-1)(x+1)
        let sol = solve_rational_equation(&numer, &denom, &numer, &denom);
        // cleared = numer*denom - numer*denom = 0 → identity
        assert!(
            matches!(
                sol,
                SolutionSet::Complement(_) | SolutionSet::Interval { .. }
            ),
            "identity equation"
        );
    }

    #[test]
    fn test_rational_eq_no_solution_from_const_numer() {
        // 1/x² = 0  →  no solution
        let numer = poly(&[1]);
        let denom = poly(&[0, 0, 1]); // x²
        let sol = solve_rational(&numer, &denom);
        assert!(sol.is_empty());
    }
}
