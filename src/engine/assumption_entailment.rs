//! Entailment logic for [`AssumptionConstraint`] pairs.
//!
//! This is a private module used only by [`super::assumption_key`].
//! It encodes the domain-subsumption lattice and the cross-form
//! entailment rules described in D012.

use crate::engine::assumption_key::{AssumptionConstraint, Domain};

// ── Domain subsumption lattice ────────────────────────────────────────────────

/// Returns `true` when every value in `smaller` is also in `larger`.
///
/// The lattice is:
/// ```text
/// Complex
///   └─ Reals
///        ├─ NonNegativeReals
///        │    └─ PositiveReals ─── (also NonZeroReals)
///        ├─ NonPositiveReals
///        │    └─ NegativeReals ─── (also NonZeroReals)
///        ├─ NonZeroReals
///        └─ Rationals
///              └─ Integers
///                    ├─ PositiveIntegers
///                    └─ NegativeIntegers
/// ```
pub(super) fn domain_subsumes(larger: Domain, smaller: Domain) -> bool {
    use Domain::*;
    if larger == smaller {
        return true;
    }
    matches!(
        (larger, smaller),
        // Complex contains everything
        (Complex, _)
        // Reals contains all real subsets
        | (Reals, NonNegativeReals)
        | (Reals, NonPositiveReals)
        | (Reals, NonZeroReals)
        | (Reals, PositiveReals)
        | (Reals, NegativeReals)
        | (Reals, Rationals)
        | (Reals, Integers)
        | (Reals, PositiveIntegers)
        | (Reals, NegativeIntegers)
        // NonNegativeReals contains PositiveReals and PositiveIntegers
        | (NonNegativeReals, PositiveReals)
        | (NonNegativeReals, PositiveIntegers)
        // NonPositiveReals contains NegativeReals and NegativeIntegers
        | (NonPositiveReals, NegativeReals)
        | (NonPositiveReals, NegativeIntegers)
        // NonZeroReals contains PositiveReals and NegativeReals
        | (NonZeroReals, PositiveReals)
        | (NonZeroReals, NegativeReals)
        // Rationals contains Integers and both integer subsets
        | (Rationals, Integers)
        | (Rationals, PositiveIntegers)
        | (Rationals, NegativeIntegers)
        // Integers contains PositiveIntegers and NegativeIntegers
        | (Integers, PositiveIntegers)
        | (Integers, NegativeIntegers)
    )
}

// ── Rational bound parsing ────────────────────────────────────────────────────

/// Parse a simple numeric string as an exact rational.
///
/// Accepted forms:
/// - `"3"` → 3/1
/// - `"-2"` → -2/1
/// - `"1/2"` → 1/2
/// - `"3.5"` → 7/2
///
/// Returns `None` for anything that cannot be parsed.
pub(super) fn parse_rational_bound(s: &str) -> Option<num_rational::Ratio<i64>> {
    use num_rational::Ratio;

    let s = s.trim();

    // Fraction form: "p/q"
    if let Some(slash) = s.find('/') {
        let numer: i64 = s[..slash].trim().parse().ok()?;
        let denom: i64 = s[slash + 1..].trim().parse().ok()?;
        if denom == 0 {
            return None;
        }
        return Some(Ratio::new(numer, denom));
    }

    // Decimal form: "3.5" → 7/2
    if let Some(dot) = s.find('.') {
        let int_part: i64 = if dot == 0 || s[..dot].trim() == "-" {
            if s.starts_with('-') {
                -0
            } else {
                0
            }
        } else {
            s[..dot].trim().parse().ok()?
        };
        let frac_str = &s[dot + 1..];
        let frac_digits = frac_str.len() as u32;
        let denom: i64 = 10_i64.checked_pow(frac_digits)?;
        let frac_val: i64 = frac_str.parse().ok()?;
        let negative = s.starts_with('-');
        let numer = if negative {
            int_part * denom - frac_val
        } else {
            int_part * denom + frac_val
        };
        return Some(Ratio::new(numer, denom));
    }

    // Integer form: "3" or "-2"
    let n: i64 = s.parse().ok()?;
    Some(Ratio::from_integer(n))
}

// ── Single-pair entailment ────────────────────────────────────────────────────

/// Returns `true` when `stronger` entails `weaker`.
///
/// Entailment is checked in three stages:
/// 1. Reflexivity — identical constraints entail each other.
/// 2. Domain subsumption — `InDomain(D1)` entails `InDomain(D2)` when D1 ⊆ D2.
/// 3. Cross-form rules — e.g. `Positive` entails `NonNegative`.
pub(super) fn constraint_entails(
    stronger: &AssumptionConstraint,
    weaker: &AssumptionConstraint,
) -> bool {
    use AssumptionConstraint::*;
    use Domain::*;

    // Stage 1: reflexivity
    if stronger == weaker {
        return true;
    }

    match (stronger, weaker) {
        // ── Stage 2: InDomain subsumption ─────────────────────────────────────
        (
            InDomain {
                var: v1,
                domain: d1,
            },
            InDomain {
                var: v2,
                domain: d2,
            },
        ) => v1 == v2 && domain_subsumes(*d2, *d1),

        // ── Stage 3a: InDomain cross-form ─────────────────────────────────────
        // PositiveIntegers ⊆ NonNegativeReals → entails NonNegative [D012-E2]
        (
            InDomain {
                var: v1,
                domain: PositiveIntegers,
            },
            NonNegative { var: v2 },
        ) => v1 == v2,
        // NegativeIntegers ⊆ NonPositiveReals → entails NonPositive [D012-E2]
        (
            InDomain {
                var: v1,
                domain: NegativeIntegers,
            },
            NonPositive { var: v2 },
        ) => v1 == v2,
        // PositiveReals → Positive
        (
            InDomain {
                var: v1,
                domain: PositiveReals,
            },
            Positive { var: v2 },
        ) => v1 == v2,
        // PositiveReals → NonNegative
        (
            InDomain {
                var: v1,
                domain: PositiveReals,
            },
            NonNegative { var: v2 },
        ) => v1 == v2,
        // PositiveReals → NonZero
        (
            InDomain {
                var: v1,
                domain: PositiveReals,
            },
            NonZero { var: v2 },
        ) => v1 == v2,
        // PositiveIntegers → Positive
        (
            InDomain {
                var: v1,
                domain: PositiveIntegers,
            },
            Positive { var: v2 },
        ) => v1 == v2,
        // PositiveIntegers → NonZero
        (
            InDomain {
                var: v1,
                domain: PositiveIntegers,
            },
            NonZero { var: v2 },
        ) => v1 == v2,
        // NegativeReals → Negative
        (
            InDomain {
                var: v1,
                domain: NegativeReals,
            },
            Negative { var: v2 },
        ) => v1 == v2,
        // NegativeReals → NonPositive
        (
            InDomain {
                var: v1,
                domain: NegativeReals,
            },
            NonPositive { var: v2 },
        ) => v1 == v2,
        // NegativeReals → NonZero
        (
            InDomain {
                var: v1,
                domain: NegativeReals,
            },
            NonZero { var: v2 },
        ) => v1 == v2,
        // NegativeIntegers → Negative
        (
            InDomain {
                var: v1,
                domain: NegativeIntegers,
            },
            Negative { var: v2 },
        ) => v1 == v2,
        // NegativeIntegers → NonZero
        (
            InDomain {
                var: v1,
                domain: NegativeIntegers,
            },
            NonZero { var: v2 },
        ) => v1 == v2,
        // NonZeroReals → NonZero
        (
            InDomain {
                var: v1,
                domain: NonZeroReals,
            },
            NonZero { var: v2 },
        ) => v1 == v2,

        // ── Stage 3b: Sign cross-form ──────────────────────────────────────────
        // Positive |- NonNegative
        (Positive { var: v1 }, NonNegative { var: v2 }) => v1 == v2,
        // Positive |- NonZero
        (Positive { var: v1 }, NonZero { var: v2 }) => v1 == v2,
        // Negative |- NonPositive
        (Negative { var: v1 }, NonPositive { var: v2 }) => v1 == v2,
        // Negative |- NonZero
        (Negative { var: v1 }, NonZero { var: v2 }) => v1 == v2,

        // ── Stage 3c: Bound comparisons ────────────────────────────────────────
        // GreaterThan(x, a) |- GreaterThan(x, b) when a >= b
        (GreaterThan { var: v1, bound: b1 }, GreaterThan { var: v2, bound: b2 }) => {
            v1 == v2 && bounds_ge(b1, b2)
        }
        // GreaterThan(x, a) |- AtLeast(x, b) when a > b (strict > non-strict weaker)
        // or a >= b (GreaterThan with same bound entails AtLeast)
        (GreaterThan { var: v1, bound: b1 }, AtLeast { var: v2, bound: b2 }) => {
            v1 == v2 && bounds_ge(b1, b2)
        }
        // AtLeast(x, a) |- AtLeast(x, b) when a >= b
        (AtLeast { var: v1, bound: b1 }, AtLeast { var: v2, bound: b2 }) => {
            v1 == v2 && bounds_ge(b1, b2)
        }
        // LessThan(x, a) |- LessThan(x, b) when a <= b
        (LessThan { var: v1, bound: b1 }, LessThan { var: v2, bound: b2 }) => {
            v1 == v2 && bounds_le(b1, b2)
        }
        // LessThan(x, a) |- AtMost(x, b) when a <= b
        (LessThan { var: v1, bound: b1 }, AtMost { var: v2, bound: b2 }) => {
            v1 == v2 && bounds_le(b1, b2)
        }
        // AtMost(x, a) |- AtMost(x, b) when a <= b
        (AtMost { var: v1, bound: b1 }, AtMost { var: v2, bound: b2 }) => {
            v1 == v2 && bounds_le(b1, b2)
        }
        // GreaterThan(x, 0) |- Positive(x)
        (GreaterThan { var: v1, bound }, Positive { var: v2 }) => v1 == v2 && is_zero_bound(bound),
        // AtLeast(x, 0) |- NonNegative(x)  (x >= 0 means non-negative)
        (AtLeast { var: v1, bound }, NonNegative { var: v2 }) => v1 == v2 && is_zero_bound(bound),
        // LessThan(x, 0) |- Negative(x)
        (LessThan { var: v1, bound }, Negative { var: v2 }) => v1 == v2 && is_zero_bound(bound),
        // AtMost(x, 0) |- NonPositive(x)
        (AtMost { var: v1, bound }, NonPositive { var: v2 }) => v1 == v2 && is_zero_bound(bound),

        _ => false,
    }
}

// ── Bound helpers ─────────────────────────────────────────────────────────────

/// Returns `true` when `a >= b` as rational numbers, or when either cannot
/// be parsed (conservative — returns false on parse failure).
fn bounds_ge(a: &str, b: &str) -> bool {
    match (parse_rational_bound(a), parse_rational_bound(b)) {
        (Some(ra), Some(rb)) => ra >= rb,
        _ => false,
    }
}

/// Returns `true` when `a <= b` as rational numbers.
fn bounds_le(a: &str, b: &str) -> bool {
    match (parse_rational_bound(a), parse_rational_bound(b)) {
        (Some(ra), Some(rb)) => ra <= rb,
        _ => false,
    }
}

/// Returns `true` when the bound string parses to exactly zero.
fn is_zero_bound(bound: &str) -> bool {
    parse_rational_bound(bound).map_or(false, |r| r == num_rational::Ratio::from_integer(0))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use Domain::*;

    // ── domain_subsumes ────────────────────────────────────────────────────────

    #[test]
    fn fast_domain_subsumes_reflexive() {
        for d in [
            PositiveIntegers,
            NegativeIntegers,
            Integers,
            Rationals,
            PositiveReals,
            NegativeReals,
            NonNegativeReals,
            NonPositiveReals,
            NonZeroReals,
            Reals,
            Complex,
        ] {
            assert!(domain_subsumes(d, d), "{d:?} should subsume itself");
        }
    }

    #[test]
    fn fast_domain_subsumes_complex_contains_all() {
        for d in [PositiveIntegers, Integers, Rationals, Reals, NonZeroReals] {
            assert!(domain_subsumes(Complex, d));
        }
    }

    #[test]
    fn fast_domain_subsumes_reals_chain() {
        assert!(domain_subsumes(Reals, PositiveReals));
        assert!(domain_subsumes(Reals, NegativeReals));
        assert!(domain_subsumes(Reals, NonNegativeReals));
        assert!(domain_subsumes(Reals, NonPositiveReals));
        assert!(domain_subsumes(Reals, NonZeroReals));
        assert!(domain_subsumes(Reals, Rationals));
        assert!(domain_subsumes(Reals, Integers));
        assert!(domain_subsumes(Reals, PositiveIntegers));
        assert!(domain_subsumes(Reals, NegativeIntegers));
    }

    #[test]
    fn fast_domain_subsumes_not_reversed() {
        assert!(!domain_subsumes(PositiveReals, Reals));
        assert!(!domain_subsumes(Integers, Rationals));
        assert!(!domain_subsumes(PositiveIntegers, NegativeIntegers));
    }

    // ── parse_rational_bound ───────────────────────────────────────────────────

    #[test]
    fn fast_parse_rational_integer() {
        use num_rational::Ratio;
        assert_eq!(parse_rational_bound("3"), Some(Ratio::from_integer(3)));
        assert_eq!(parse_rational_bound("-2"), Some(Ratio::from_integer(-2)));
        assert_eq!(parse_rational_bound("0"), Some(Ratio::from_integer(0)));
    }

    #[test]
    fn fast_parse_rational_fraction() {
        use num_rational::Ratio;
        assert_eq!(parse_rational_bound("1/2"), Some(Ratio::new(1, 2)));
        assert_eq!(parse_rational_bound("3/4"), Some(Ratio::new(3, 4)));
    }

    #[test]
    fn fast_parse_rational_decimal() {
        use num_rational::Ratio;
        assert_eq!(parse_rational_bound("3.5"), Some(Ratio::new(7, 2)));
        assert_eq!(parse_rational_bound("0.25"), Some(Ratio::new(1, 4)));
    }

    #[test]
    fn fast_parse_rational_non_numeric_returns_none() {
        assert_eq!(parse_rational_bound("x"), None);
        assert_eq!(parse_rational_bound("inf"), None);
        assert_eq!(parse_rational_bound(""), None);
    }

    // ── constraint_entails ─────────────────────────────────────────────────────

    #[test]
    fn fast_entails_reflexive() {
        use AssumptionConstraint::*;
        let c = Positive { var: "x".into() };
        assert!(constraint_entails(&c, &c));
    }

    #[test]
    fn fast_entails_positive_implies_nonnegative() {
        use AssumptionConstraint::*;
        let pos = Positive { var: "x".into() };
        let nn = NonNegative { var: "x".into() };
        assert!(constraint_entails(&pos, &nn));
    }

    #[test]
    fn fast_entails_positive_implies_nonzero() {
        use AssumptionConstraint::*;
        let pos = Positive { var: "x".into() };
        let nz = NonZero { var: "x".into() };
        assert!(constraint_entails(&pos, &nz));
    }

    #[test]
    fn fast_entails_negative_implies_nonpositive() {
        use AssumptionConstraint::*;
        let neg = Negative { var: "x".into() };
        let np = NonPositive { var: "x".into() };
        assert!(constraint_entails(&neg, &np));
    }

    #[test]
    fn fast_entails_negative_implies_nonzero() {
        use AssumptionConstraint::*;
        let neg = Negative { var: "x".into() };
        let nz = NonZero { var: "x".into() };
        assert!(constraint_entails(&neg, &nz));
    }

    #[test]
    fn fast_entails_positive_integers_implies_nonnegative_d012_e2() {
        use AssumptionConstraint::*;
        let pi = InDomain {
            var: "n".into(),
            domain: PositiveIntegers,
        };
        let nn = NonNegative { var: "n".into() };
        assert!(constraint_entails(&pi, &nn));
    }

    #[test]
    fn fast_entails_negative_integers_implies_nonpositive_d012_e2() {
        use AssumptionConstraint::*;
        let ni = InDomain {
            var: "n".into(),
            domain: NegativeIntegers,
        };
        let np = NonPositive { var: "n".into() };
        assert!(constraint_entails(&ni, &np));
    }

    #[test]
    fn fast_entails_domain_subsumption() {
        use AssumptionConstraint::*;
        let pos_reals = InDomain {
            var: "x".into(),
            domain: PositiveReals,
        };
        let reals = InDomain {
            var: "x".into(),
            domain: Reals,
        };
        assert!(constraint_entails(&pos_reals, &reals));
        // Reverse does not hold
        assert!(!constraint_entails(&reals, &pos_reals));
    }

    #[test]
    fn fast_entails_wrong_var_does_not_entail() {
        use AssumptionConstraint::*;
        let pos_x = Positive { var: "x".into() };
        let nn_y = NonNegative { var: "y".into() };
        assert!(!constraint_entails(&pos_x, &nn_y));
    }

    #[test]
    fn fast_entails_greater_than_zero_implies_positive() {
        use AssumptionConstraint::*;
        let gt0 = GreaterThan {
            var: "x".into(),
            bound: "0".into(),
        };
        let pos = Positive { var: "x".into() };
        assert!(constraint_entails(&gt0, &pos));
    }

    #[test]
    fn fast_entails_at_least_zero_implies_nonnegative() {
        use AssumptionConstraint::*;
        let ge0 = AtLeast {
            var: "x".into(),
            bound: "0".into(),
        };
        let nn = NonNegative { var: "x".into() };
        assert!(constraint_entails(&ge0, &nn));
    }
}
