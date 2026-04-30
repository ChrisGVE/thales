//! Inequality solver using sign analysis between critical points.
//!
//! Solves polynomial and rational inequalities of the form
//! `f(x) < 0`, `f(x) ≤ 0`, `f(x) > 0`, or `f(x) ≥ 0`, where
//! `f(x) = numer(x) / denom(x)` over the reals.
//!
//! # Algorithm
//!
//! 1. Collect all real **roots** of `numer` (sign-change points) and
//!    all real **poles** of `denom` (excluded, undefined points).
//! 2. Sort these critical points on the real line.
//! 3. Test the sign of `f` at a sample point inside each open interval
//!    between consecutive critical points (and on the two unbounded rays).
//! 4. Each interval whose sign satisfies the inequality contributes to
//!    the result; roots are included for non-strict (`≤`, `≥`) unless
//!    they are also poles.
//!
//! # Examples
//!
//! ```rust
//! use thales::numeric::{
//!     DensePolynomial, BigRational,
//!     inequality_solver::{solve_inequality, InequalityType},
//!     SolutionSet,
//! };
//! use num::traits::Zero;
//!
//! // x² - 1 > 0  →  (-∞, -1) ∪ (1, ∞)
//! let numer = DensePolynomial::from_coeffs(vec![
//!     BigRational::from(-1_i64),
//!     BigRational::zero(),
//!     BigRational::from(1_i64),
//! ]);
//! let denom = DensePolynomial::from_coeffs(vec![BigRational::from(1_i64)]);
//! let sol = solve_inequality(&numer, &denom, InequalityType::GreaterThan);
//! match &sol {
//!     SolutionSet::Union(parts) => assert_eq!(parts.len(), 2),
//!     _ => panic!("expected two-interval union"),
//! }
//! ```

use super::big_rational::BigRational;
use super::dense_poly::DensePolynomial;
use super::expr::Expr;
use super::poly_equation_solver::roots_with_multiplicity;
use super::ring::Ring;
use super::solution_set::{IntervalBound, SolutionSet};
use num::traits::Zero;
use std::sync::Arc;

// ── Public types ──────────────────────────────────────────────────────────────

/// The comparison direction of an inequality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InequalityType {
    /// Strict less-than: `f(x) < 0`.
    LessThan,
    /// Non-strict less-than-or-equal: `f(x) ≤ 0`.
    LessEq,
    /// Strict greater-than: `f(x) > 0`.
    GreaterThan,
    /// Non-strict greater-than-or-equal: `f(x) ≥ 0`.
    GreaterEq,
}

impl InequalityType {
    /// Whether the inequality is non-strict (allows equality).
    fn is_non_strict(self) -> bool {
        matches!(self, InequalityType::LessEq | InequalityType::GreaterEq)
    }

    /// Whether the sign `s` of `f` at a test point satisfies this inequality.
    fn sign_satisfies(self, s: Sign) -> bool {
        match self {
            InequalityType::LessThan => s == Sign::Negative,
            InequalityType::LessEq => matches!(s, Sign::Negative | Sign::Zero),
            InequalityType::GreaterThan => s == Sign::Positive,
            InequalityType::GreaterEq => matches!(s, Sign::Positive | Sign::Zero),
        }
    }
}

// ── Internal sign type ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sign {
    Negative,
    Zero,
    Positive,
}

impl Sign {
    fn of(r: &BigRational) -> Self {
        if r.is_negative() {
            Sign::Negative
        } else if Zero::is_zero(r) {
            Sign::Zero
        } else {
            Sign::Positive
        }
    }
}

// ── Critical-point record ─────────────────────────────────────────────────────

/// A critical point on the real line with its origin.
#[derive(Clone, Debug)]
struct CriticalPoint {
    value: BigRational,
    is_pole: bool,
}

// ── Main entry point ──────────────────────────────────────────────────────────

/// Solve `numer(x) / denom(x) <op> 0` and return the solution set.
///
/// - `numer`: polynomial numerator of the expression.
/// - `denom`: polynomial denominator; use a constant `1` polynomial for pure
///   polynomial inequalities.
/// - `ineq`: the comparison type (`<`, `≤`, `>`, `≥`).
///
/// Returns a [`SolutionSet`] which may be [`SolutionSet::Empty`],
/// a single interval, or a [`SolutionSet::Union`] of intervals.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{
///     DensePolynomial, BigRational,
///     inequality_solver::{solve_inequality, InequalityType},
///     SolutionSet,
/// };
/// use num::traits::Zero;
///
/// // 1/x > 0  →  (0, ∞)
/// let numer = DensePolynomial::from_coeffs(vec![BigRational::from(1_i64)]);
/// let denom = DensePolynomial::from_coeffs(vec![
///     BigRational::zero(),
///     BigRational::from(1_i64),
/// ]);
/// let sol = solve_inequality(&numer, &denom, InequalityType::GreaterThan);
/// match &sol {
///     SolutionSet::Interval { .. } => {}
///     _ => panic!("expected a single interval (0, ∞)"),
/// }
/// ```
pub fn solve_inequality(
    numer: &DensePolynomial<BigRational>,
    denom: &DensePolynomial<BigRational>,
    ineq: InequalityType,
) -> SolutionSet {
    // Undefined expression: zero denominator polynomial
    if denom.is_zero() {
        return SolutionSet::Empty;
    }

    // Zero numerator: f(x) = 0 everywhere denom ≠ 0
    if numer.is_zero() {
        return solve_identically_zero(denom, ineq);
    }

    let critical = collect_critical_points(numer, denom);
    build_solution(numer, denom, &critical, ineq)
}

// ── Zero-numerator short-circuit ──────────────────────────────────────────────

/// Handle the case where `numer` is the zero polynomial.
///
/// `f(x) = 0/denom(x) = 0` wherever `denom(x) ≠ 0`.
fn solve_identically_zero(
    denom: &DensePolynomial<BigRational>,
    ineq: InequalityType,
) -> SolutionSet {
    // f = 0 satisfies only ≤ and ≥, not strict inequalities
    if ineq.is_non_strict() {
        // All reals except poles of denom
        let poles = poles_of(denom);
        if poles.is_empty() {
            SolutionSet::all_reals()
        } else {
            SolutionSet::all_except(poles.into_iter().map(|p| rat_to_expr(p.value)).collect())
        }
    } else {
        SolutionSet::Empty
    }
}

// ── Critical-point collection ─────────────────────────────────────────────────

/// Collect and sort all distinct critical points (roots + poles).
fn collect_critical_points(
    numer: &DensePolynomial<BigRational>,
    denom: &DensePolynomial<BigRational>,
) -> Vec<CriticalPoint> {
    let mut pts: Vec<CriticalPoint> = Vec::new();

    for rw in roots_with_multiplicity(numer) {
        if let Some(v) = expr_to_rat(&rw.root) {
            // Mark as not a pole (it's a root of numerator)
            if !pts.iter().any(|p| p.value == v) {
                pts.push(CriticalPoint {
                    value: v,
                    is_pole: false,
                });
            }
        }
    }

    for pole in poles_of(denom) {
        if let Some(pos) = pts.iter().position(|p| p.value == pole.value) {
            // A root that is also a pole becomes a pole (excluded point)
            pts[pos].is_pole = true;
        } else {
            pts.push(CriticalPoint {
                value: pole.value,
                is_pole: true,
            });
        }
    }

    pts.sort_by(|a, b| {
        a.value
            .partial_cmp(&b.value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pts
}

/// Extract the rational poles (zeros of `denom`) as `CriticalPoint` records.
fn poles_of(denom: &DensePolynomial<BigRational>) -> Vec<CriticalPoint> {
    roots_with_multiplicity(denom)
        .into_iter()
        .filter_map(|rw| {
            expr_to_rat(&rw.root).map(|v| CriticalPoint {
                value: v,
                is_pole: true,
            })
        })
        .collect()
}

// ── Interval building ─────────────────────────────────────────────────────────

/// Build the solution `SolutionSet` by testing each sub-interval.
fn build_solution(
    numer: &DensePolynomial<BigRational>,
    denom: &DensePolynomial<BigRational>,
    critical: &[CriticalPoint],
    ineq: InequalityType,
) -> SolutionSet {
    let mut result = SolutionSet::Empty;

    // Iterate over n+1 open intervals defined by n critical points
    let n = critical.len();
    for i in 0..=n {
        let left = if i == 0 { None } else { Some(&critical[i - 1]) };
        let right = if i == n { None } else { Some(&critical[i]) };

        let sample = choose_sample(left, right);
        let sign = eval_sign(numer, denom, &sample);

        if ineq.sign_satisfies(sign) {
            let interval = make_interval(left, right);
            result = result.union(interval);
        }

        // Include the critical point itself when appropriate
        if let Some(cp) = right {
            if include_critical_point(cp, ineq) {
                let pt = SolutionSet::singleton(rat_to_expr(cp.value.clone()));
                result = result.union(pt);
            }
        }
    }

    result
}

/// Decide whether a critical point itself belongs to the solution.
///
/// - Poles are always excluded (f undefined there).
/// - Roots are included only for non-strict inequalities (f = 0 satisfies ≤ and ≥).
fn include_critical_point(cp: &CriticalPoint, ineq: InequalityType) -> bool {
    !cp.is_pole && ineq.is_non_strict()
}

// ── Interval construction ─────────────────────────────────────────────────────

/// Build an open interval `(left, right)` from optional boundary critical points.
fn make_interval(left: Option<&CriticalPoint>, right: Option<&CriticalPoint>) -> SolutionSet {
    let low = match left {
        None => IntervalBound::NegInfinity,
        Some(cp) => IntervalBound::Finite {
            value: rat_to_expr(cp.value.clone()),
            inclusive: false,
        },
    };
    let high = match right {
        None => IntervalBound::PosInfinity,
        Some(cp) => IntervalBound::Finite {
            value: rat_to_expr(cp.value.clone()),
            inclusive: false,
        },
    };
    SolutionSet::Interval { low, high }
}

// ── Sample-point selection ────────────────────────────────────────────────────

/// Choose a rational sample point strictly inside the open interval `(left, right)`.
///
/// - Both bounds finite: midpoint `(a + b) / 2`.
/// - Only right bound: `right - 1`.
/// - Only left bound: `left + 1`.
/// - Neither bound: `0`.
fn choose_sample(left: Option<&CriticalPoint>, right: Option<&CriticalPoint>) -> BigRational {
    match (left, right) {
        (Some(l), Some(r)) => {
            // midpoint = (l + r) / 2
            let sum = l.value.clone() + r.value.clone();
            sum / BigRational::from(2_i64)
        }
        (None, Some(r)) => r.value.clone() - BigRational::from(1_i64),
        (Some(l), None) => l.value.clone() + BigRational::from(1_i64),
        (None, None) => BigRational::from(0_i64),
    }
}

// ── Sign evaluation ───────────────────────────────────────────────────────────

/// Evaluate `sign(numer(x) / denom(x))` at rational `x = pt`.
///
/// The sign of a fraction equals `sign(numer) * sign(denom)`.
/// If `denom(pt) = 0` (unexpected at a non-pole sample), returns `Sign::Zero`.
fn eval_sign(
    numer: &DensePolynomial<BigRational>,
    denom: &DensePolynomial<BigRational>,
    pt: &BigRational,
) -> Sign {
    let n_val = numer.eval(pt);
    let d_val = denom.eval(pt);

    if Ring::is_zero(&d_val) {
        // Sample landed on a pole — should not happen if critical points are correct
        return Sign::Zero;
    }

    // sign(n/d) = sign(n) * sign(d)
    let n_sign = Sign::of(&n_val);
    let d_sign = Sign::of(&d_val);

    match (n_sign, d_sign) {
        (Sign::Zero, _) => Sign::Zero,
        (Sign::Positive, Sign::Positive) | (Sign::Negative, Sign::Negative) => Sign::Positive,
        _ => Sign::Negative,
    }
}

// ── Conversion helpers ────────────────────────────────────────────────────────

/// Try to convert an `Arc<Expr>` to a `BigRational`.
fn expr_to_rat(expr: &Arc<Expr>) -> Option<BigRational> {
    match expr.as_ref() {
        Expr::Integer(n) => n.to_i64().map(BigRational::from),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

/// Convert a `BigRational` to the canonical `Arc<Expr>`.
fn rat_to_expr(r: BigRational) -> Arc<Expr> {
    if r.is_integer() {
        match r.numer().to_i64() {
            Some(n) => Expr::int(n),
            None => Arc::new(Expr::Rational(r)),
        }
    } else {
        match (r.numer().to_i64(), r.denom().to_i64()) {
            (Some(n), Some(d)) => Expr::rational(n, d),
            _ => Arc::new(Expr::Rational(r)),
        }
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

    fn one_poly() -> P {
        poly(&[1])
    }

    /// Extract (lower-bound-f64, upper-bound-f64, low-inclusive, high-inclusive)
    /// from a SolutionSet::Interval.
    fn interval_bounds(sol: &SolutionSet) -> Option<(Option<i64>, Option<i64>, bool, bool)> {
        match sol {
            SolutionSet::Interval { low, high } => {
                let lo = match low {
                    IntervalBound::NegInfinity => (None, false),
                    IntervalBound::Finite { value, inclusive } => {
                        let v = match value.as_ref() {
                            Expr::Integer(n) => n.to_i64(),
                            _ => None,
                        };
                        (v, *inclusive)
                    }
                    IntervalBound::PosInfinity => (None, false),
                };
                let hi = match high {
                    IntervalBound::PosInfinity => (None, false),
                    IntervalBound::Finite { value, inclusive } => {
                        let v = match value.as_ref() {
                            Expr::Integer(n) => n.to_i64(),
                            _ => None,
                        };
                        (v, *inclusive)
                    }
                    IntervalBound::NegInfinity => (None, false),
                };
                Some((lo.0, hi.0, lo.1, hi.1))
            }
            _ => None,
        }
    }

    // ── x² - 1 > 0  →  (-∞,-1) ∪ (1,∞) ─────────────────────────────────────

    #[test]
    fn test_x_squared_minus_1_gt_zero() {
        // numer = x² - 1, denom = 1
        let numer = poly(&[-1, 0, 1]);
        let sol = solve_inequality(&numer, &one_poly(), InequalityType::GreaterThan);
        match &sol {
            SolutionSet::Union(parts) => {
                assert_eq!(parts.len(), 2, "expected two intervals");
                // First part: (-∞, -1)
                let b0 = interval_bounds(&parts[0]).expect("first part is interval");
                assert_eq!(b0, (None, Some(-1), false, false));
                // Second part: (1, ∞)
                let b1 = interval_bounds(&parts[1]).expect("second part is interval");
                assert_eq!(b1, (Some(1), None, false, false));
            }
            _ => panic!("expected Union, got {sol:?}"),
        }
    }

    // ── x² - 1 < 0  →  (-1, 1) ──────────────────────────────────────────────

    #[test]
    fn test_x_squared_minus_1_lt_zero() {
        let numer = poly(&[-1, 0, 1]);
        let sol = solve_inequality(&numer, &one_poly(), InequalityType::LessThan);
        let b = interval_bounds(&sol).expect("expected single interval");
        assert_eq!(b, (Some(-1), Some(1), false, false));
    }

    // ── x² - 1 ≥ 0  →  (-∞,-1] ∪ [1,∞) ────────────────────────────────────

    #[test]
    fn test_x_squared_minus_1_geq_zero() {
        let numer = poly(&[-1, 0, 1]);
        let sol = solve_inequality(&numer, &one_poly(), InequalityType::GreaterEq);
        // Expect: (-∞,-1) ∪ {-1} ∪ (1,∞) ∪ {1}  →  simplified to union parts
        // The solver produces open intervals + singleton points; check membership
        // by counting parts that cover left-of-−1, the point −1, between, the point 1, right-of-1
        let flat = flatten_sol(&sol);
        assert!(
            flat.len() >= 4,
            "expected at least 4 parts (2 intervals + 2 singletons), got {flat:?}"
        );
    }

    // ── x² - 1 ≤ 0  →  [-1, 1] ─────────────────────────────────────────────

    #[test]
    fn test_x_squared_minus_1_leq_zero() {
        let numer = poly(&[-1, 0, 1]);
        let sol = solve_inequality(&numer, &one_poly(), InequalityType::LessEq);
        let flat = flatten_sol(&sol);
        // (-1,1) + {-1} + {1} = at least 3 parts
        assert!(
            flat.len() >= 3,
            "expected interval and two endpoints: {flat:?}"
        );
    }

    // ── 1/x > 0  →  (0, ∞) ──────────────────────────────────────────────────

    #[test]
    fn test_one_over_x_gt_zero() {
        // numer = 1, denom = x
        let numer = poly(&[1]);
        let denom = poly(&[0, 1]); // x
        let sol = solve_inequality(&numer, &denom, InequalityType::GreaterThan);
        let b = interval_bounds(&sol).expect("expected single interval (0, ∞)");
        assert_eq!(b, (Some(0), None, false, false));
    }

    // ── 1/x < 0  →  (-∞, 0) ─────────────────────────────────────────────────

    #[test]
    fn test_one_over_x_lt_zero() {
        let numer = poly(&[1]);
        let denom = poly(&[0, 1]);
        let sol = solve_inequality(&numer, &denom, InequalityType::LessThan);
        let b = interval_bounds(&sol).expect("expected single interval (-∞, 0)");
        assert_eq!(b, (None, Some(0), false, false));
    }

    // ── (x-1)/(x+1) ≥ 0  →  (-∞,-1) ∪ [1,∞) ───────────────────────────────

    #[test]
    fn test_x_minus_1_over_x_plus_1_geq_zero() {
        // numer = x - 1, denom = x + 1
        let numer = poly(&[-1, 1]);
        let denom = poly(&[1, 1]);
        let sol = solve_inequality(&numer, &denom, InequalityType::GreaterEq);
        // Expected: (-∞,-1) open (pole at -1 excluded), [1,∞) (root at 1 included)
        let flat = flatten_sol(&sol);

        // Find the interval that ends at -1 (the (-∞,-1) part)
        let has_neg_inf_to_neg1 = flat
            .iter()
            .any(|s| matches!(interval_bounds(s), Some((None, Some(-1), false, false))));
        // Find the singleton {1} (root included by ≥)
        let has_singleton_1 = flat.iter().any(|s| {
            matches!(s, SolutionSet::Finite(v) if v.len() == 1
                && matches!(v[0].as_ref(), Expr::Integer(n) if n.to_i64() == Some(1)))
        });
        // Find the interval starting at 1 going to +∞ (open, since built without singleton)
        let has_1_to_inf = flat
            .iter()
            .any(|s| matches!(interval_bounds(s), Some((Some(1), None, false, false))));

        assert!(has_neg_inf_to_neg1, "missing (-∞,-1): {flat:?}");
        assert!(has_singleton_1, "missing {{1}}: {flat:?}");
        assert!(has_1_to_inf, "missing (1,∞): {flat:?}");
    }

    // ── zero numerator with strict ineq → empty ──────────────────────────────

    #[test]
    fn test_zero_numerator_strict_gt_is_empty() {
        let numer = P::zero();
        let sol = solve_inequality(&numer, &one_poly(), InequalityType::GreaterThan);
        assert!(sol.is_empty());
    }

    // ── zero numerator with non-strict ineq → all reals ─────────────────────

    #[test]
    fn test_zero_numerator_geq_is_all_reals() {
        let numer = P::zero();
        let sol = solve_inequality(&numer, &one_poly(), InequalityType::GreaterEq);
        assert_eq!(sol, SolutionSet::all_reals());
    }

    // ── zero denominator → empty ─────────────────────────────────────────────

    #[test]
    fn test_zero_denom_is_empty() {
        let numer = poly(&[1]);
        let sol = solve_inequality(&numer, &P::zero(), InequalityType::GreaterThan);
        assert!(sol.is_empty());
    }

    // ── x > 0  →  (0, ∞) ────────────────────────────────────────────────────

    #[test]
    fn test_x_gt_zero() {
        let numer = poly(&[0, 1]); // x
        let sol = solve_inequality(&numer, &one_poly(), InequalityType::GreaterThan);
        let b = interval_bounds(&sol).expect("expected single interval (0, ∞)");
        assert_eq!(b, (Some(0), None, false, false));
    }

    // ── x < 0  →  (-∞, 0) ───────────────────────────────────────────────────

    #[test]
    fn test_x_lt_zero() {
        let numer = poly(&[0, 1]);
        let sol = solve_inequality(&numer, &one_poly(), InequalityType::LessThan);
        let b = interval_bounds(&sol).expect("expected single interval (-∞, 0)");
        assert_eq!(b, (None, Some(0), false, false));
    }

    // ── x² + 1 > 0  →  all reals ────────────────────────────────────────────

    #[test]
    fn test_x_squared_plus_1_gt_zero_all_reals() {
        // x² + 1 has no real roots; always positive
        let numer = poly(&[1, 0, 1]);
        let sol = solve_inequality(&numer, &one_poly(), InequalityType::GreaterThan);
        // Single interval (-∞, ∞) — no critical points
        assert_eq!(sol, SolutionSet::all_reals());
    }

    // ── x² + 1 < 0  →  empty ────────────────────────────────────────────────

    #[test]
    fn test_x_squared_plus_1_lt_zero_empty() {
        let numer = poly(&[1, 0, 1]);
        let sol = solve_inequality(&numer, &one_poly(), InequalityType::LessThan);
        assert!(sol.is_empty());
    }

    // ── helper: flatten union to leaf SolutionSets ───────────────────────────

    fn flatten_sol(s: &SolutionSet) -> Vec<&SolutionSet> {
        match s {
            SolutionSet::Union(parts) => parts.iter().flat_map(flatten_sol).collect(),
            other => vec![other],
        }
    }
}
