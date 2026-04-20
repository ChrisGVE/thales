//! Domain algebra: `ℕ`, `ℤ`, `ℚ`, `ℝ`, `ℂ`, `ℍ` with qualifiers (positive,
//! negative, nonzero, zero-inclusion), intervals, finite sets, and set
//! algebra (union / intersection / complement).
//!
//! # Scope (v0.8.1 scaffolding)
//!
//! Full type surface is present. Only the `ambient_domain` on
//! [`super::Request`] is consulted at runtime. Mathlex symbol-level domain
//! annotations are consumed in v0.10.0.

use crate::Expression;

use super::Bound;

/// Base number system.
///
/// `Octonion` is intentionally not included in v0.8.1 — track separately if
/// needed.
#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum BaseDomain {
    /// ℕ — non-negative integers (0, 1, 2, …).
    Natural,
    /// ℤ — integers.
    Integer,
    /// ℚ — rationals.
    Rational,
    /// ℝ — real numbers.
    Real,
    /// ℂ — complex numbers.
    Complex,
    /// ℍ — quaternions.
    Quaternion,
}

/// Qualifiers refining a [`BaseDomain`] into a specific subset.
///
/// Examples:
///
/// - `ℝ⁺`  — `Real` with `{ zero: Excluded, positive: true, negative: false }`
/// - `ℝ⁺₀` — `Real` with `{ zero: Allowed,  positive: true, negative: false }`
/// - `ℝ*`  — `Real` with `{ zero: Excluded, positive: true, negative: true  }`
/// - `ℝ`   — `Real` with `{ zero: Allowed,  positive: true, negative: true  }`
#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Qualifier {
    /// Whether zero belongs.
    pub zero: Inclusion,
    /// Whether positive elements belong.
    pub positive: bool,
    /// Whether negative elements belong.
    pub negative: bool,
}

/// Zero-inclusion policy in a [`Qualifier`].
#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum Inclusion {
    /// Zero belongs to the set.
    Allowed,
    /// Zero is explicitly excluded.
    Excluded,
}

impl Qualifier {
    /// Full qualifier covering positive, negative, and zero.
    #[must_use]
    pub const fn full() -> Self {
        Self {
            zero: Inclusion::Allowed,
            positive: true,
            negative: true,
        }
    }

    /// `ℝ⁺` / `ℤ⁺` / `ℚ⁺`: strictly positive.
    #[must_use]
    pub const fn positive_strict() -> Self {
        Self {
            zero: Inclusion::Excluded,
            positive: true,
            negative: false,
        }
    }

    /// `ℝ⁺₀` / `ℤ⁺₀`: non-negative.
    #[must_use]
    pub const fn non_negative() -> Self {
        Self {
            zero: Inclusion::Allowed,
            positive: true,
            negative: false,
        }
    }

    /// `ℝ⁻` / `ℤ⁻`: strictly negative.
    #[must_use]
    pub const fn negative_strict() -> Self {
        Self {
            zero: Inclusion::Excluded,
            positive: false,
            negative: true,
        }
    }

    /// `ℝ*` / `ℤ*`: nonzero (positive or negative, no zero).
    #[must_use]
    pub const fn nonzero() -> Self {
        Self {
            zero: Inclusion::Excluded,
            positive: true,
            negative: true,
        }
    }
}

/// A basic domain: [`BaseDomain`] refined by a [`Qualifier`].
#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Domain {
    /// Underlying number system.
    pub base: BaseDomain,
    /// Refinement.
    pub qualifier: Qualifier,
}

impl Domain {
    /// Natural numbers `ℕ = {0, 1, 2, …}`.
    #[must_use]
    pub const fn natural() -> Self {
        Self {
            base: BaseDomain::Natural,
            qualifier: Qualifier::non_negative(),
        }
    }

    /// Integers `ℤ`.
    #[must_use]
    pub const fn integer() -> Self {
        Self {
            base: BaseDomain::Integer,
            qualifier: Qualifier::full(),
        }
    }

    /// Rationals `ℚ`.
    #[must_use]
    pub const fn rational() -> Self {
        Self {
            base: BaseDomain::Rational,
            qualifier: Qualifier::full(),
        }
    }

    /// Reals `ℝ`.
    #[must_use]
    pub const fn real() -> Self {
        Self {
            base: BaseDomain::Real,
            qualifier: Qualifier::full(),
        }
    }

    /// Positive reals `ℝ⁺`.
    #[must_use]
    pub const fn real_positive() -> Self {
        Self {
            base: BaseDomain::Real,
            qualifier: Qualifier::positive_strict(),
        }
    }

    /// Non-negative reals `ℝ⁺₀`.
    #[must_use]
    pub const fn real_non_negative() -> Self {
        Self {
            base: BaseDomain::Real,
            qualifier: Qualifier::non_negative(),
        }
    }

    /// Complex numbers `ℂ`.
    #[must_use]
    pub const fn complex() -> Self {
        Self {
            base: BaseDomain::Complex,
            qualifier: Qualifier::full(),
        }
    }

    /// Quaternions `ℍ`.
    #[must_use]
    pub const fn quaternion() -> Self {
        Self {
            base: BaseDomain::Quaternion,
            qualifier: Qualifier::full(),
        }
    }
}

/// Compositional domain expression. Allows intervals, finite sets, and set
/// algebra built on top of [`Domain`].
#[derive(Debug, Clone, PartialEq)]
pub enum DomainExpr {
    /// A single base domain with qualifier.
    Base(Domain),
    /// Interval within a domain.
    Interval {
        /// Underlying domain.
        domain: Domain,
        /// Lower bound.
        lower: Bound,
        /// Upper bound.
        upper: Bound,
    },
    /// Finite set of specific elements in a domain.
    FiniteSet {
        /// Underlying domain.
        domain: Domain,
        /// Elements of the set.
        elements: Vec<Expression>,
    },
    /// Union of domain expressions.
    Union(Vec<DomainExpr>),
    /// Intersection of domain expressions.
    Intersection(Vec<DomainExpr>),
    /// Complement of a domain expression (within an implicit universe; usually
    /// the enclosing [`Domain`] of the surrounding computation).
    Complement(Box<DomainExpr>),
}

/// Policy for resolving inconsistent domain declarations on the same symbol
/// or between user-declared domain and engine-derived constraints.
///
/// Example: symbol declared `x ∈ ℝ⁺` and also `x ∈ (−1, ∞)`. Intersection is
/// `[0, ∞)`; explicit intersection narrows. Alternatively, strict mode emits
/// a [`super::DiagnosticCode::InconsistentDomain`] error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DomainPolicy {
    /// Intersect conflicting declarations. Emits
    /// [`super::DiagnosticCode::DomainNarrowed`] informational diagnostic.
    /// Default.
    #[default]
    IntersectOnMismatch,
    /// Emit [`super::DiagnosticCode::InconsistentDomain`] error and halt.
    ErrorOnMismatch,
}

// ── Domain algebra ───────────────────────────────────────────────────────────

impl BaseDomain {
    /// Containment on the standard inclusion chain:
    /// `ℕ ⊂ ℤ ⊂ ℚ ⊂ ℝ ⊂ ℂ ⊂ ℍ`.
    ///
    /// Returns `true` iff every element of `self` also belongs to `other`.
    #[must_use]
    pub const fn is_subset_of(self, other: BaseDomain) -> bool {
        self.rank() <= other.rank()
    }

    /// Numeric rank on the inclusion chain. Used internally by
    /// [`Self::is_subset_of`], [`Self::intersect`], and [`Self::union`].
    const fn rank(self) -> u8 {
        match self {
            BaseDomain::Natural => 0,
            BaseDomain::Integer => 1,
            BaseDomain::Rational => 2,
            BaseDomain::Real => 3,
            BaseDomain::Complex => 4,
            // Quaternion is outside the classical inclusion chain; placed
            // strictly above Complex so that `ℂ ⊂ ℍ` holds structurally.
            BaseDomain::Quaternion => 5,
        }
    }

    /// Intersection of two base domains on the inclusion chain — the
    /// smaller (more restrictive) of the two.
    #[must_use]
    pub const fn intersect(self, other: BaseDomain) -> BaseDomain {
        if self.rank() <= other.rank() {
            self
        } else {
            other
        }
    }

    /// Union of two base domains on the inclusion chain — the larger
    /// (more permissive) of the two.
    #[must_use]
    pub const fn union(self, other: BaseDomain) -> BaseDomain {
        if self.rank() >= other.rank() {
            self
        } else {
            other
        }
    }
}

impl Qualifier {
    /// Intersect two qualifiers: both must admit a value for the result to
    /// admit it.
    #[must_use]
    pub fn intersect(self, other: Qualifier) -> Qualifier {
        Qualifier {
            zero: match (self.zero, other.zero) {
                (Inclusion::Allowed, Inclusion::Allowed) => Inclusion::Allowed,
                _ => Inclusion::Excluded,
            },
            positive: self.positive && other.positive,
            negative: self.negative && other.negative,
        }
    }

    /// Union of two qualifiers: either admitting a value is enough.
    #[must_use]
    pub fn union(self, other: Qualifier) -> Qualifier {
        Qualifier {
            zero: match (self.zero, other.zero) {
                (Inclusion::Excluded, Inclusion::Excluded) => Inclusion::Excluded,
                _ => Inclusion::Allowed,
            },
            positive: self.positive || other.positive,
            negative: self.negative || other.negative,
        }
    }

    /// `true` when the qualifier admits no elements (negative, positive,
    /// and zero all excluded).
    #[must_use]
    pub fn is_empty(self) -> bool {
        !self.positive && !self.negative && matches!(self.zero, Inclusion::Excluded)
    }
}

impl Domain {
    /// Intersect two domains: meet on the base-domain chain, meet on the
    /// qualifier. Returns `None` when the result is empty.
    ///
    /// Example: `ℝ⁺ ∩ (−1, ∞) = [0, ∞)` (the intersection of a positive
    /// real qualifier with an interval that includes part of the negatives
    /// narrows to the non-negative reals — captured at the `Domain` level
    /// only up to qualifier; the interval narrowing must be done at the
    /// [`DomainExpr`] layer).
    ///
    /// Example: `ℤ⁺ ∩ ℝ⁻ = ∅` (qualifier empty → `None`).
    #[must_use]
    pub fn intersect(self, other: Domain) -> Option<Domain> {
        let base = self.base.intersect(other.base);
        let qualifier = self.qualifier.intersect(other.qualifier);
        if qualifier.is_empty() {
            None
        } else {
            Some(Domain { base, qualifier })
        }
    }

    /// Union of two domains. Takes the join on the base-domain chain and
    /// the union on the qualifier. Always returns a concrete `Domain`.
    #[must_use]
    pub fn union(self, other: Domain) -> Domain {
        Domain {
            base: self.base.union(other.base),
            qualifier: self.qualifier.union(other.qualifier),
        }
    }

    /// `true` when every element of `self` is also an element of `other`.
    #[must_use]
    pub fn is_subset_of(self, other: Domain) -> bool {
        // Base must be on or below other's base.
        if !self.base.is_subset_of(other.base) {
            return false;
        }
        // Qualifier: self's admitted parts must all be admitted by other.
        if self.qualifier.positive && !other.qualifier.positive {
            return false;
        }
        if self.qualifier.negative && !other.qualifier.negative {
            return false;
        }
        if matches!(self.qualifier.zero, Inclusion::Allowed)
            && matches!(other.qualifier.zero, Inclusion::Excluded)
        {
            return false;
        }
        true
    }
}

impl DomainExpr {
    /// Convenience: wrap a base [`Domain`].
    #[must_use]
    pub fn of(domain: Domain) -> Self {
        DomainExpr::Base(domain)
    }

    /// Flatten and simplify a [`DomainExpr`]:
    ///
    /// - Collapse nested `Union(Union(...))` / `Intersection(Intersection(...))`.
    /// - Drop duplicates.
    /// - Fold singleton unions / intersections to their single child.
    /// - Double-complement elimination: `!!x = x`.
    ///
    /// Does **not** compute cross-variant intersections (e.g. interval ∩
    /// base) — those arrive in v0.10.0 where the `Expression`-level
    /// ordering needed for interval math is available.
    #[must_use]
    pub fn simplify(self) -> Self {
        match self {
            DomainExpr::Union(children) => {
                let flat = flatten_union(children);
                let deduped = dedup_preserving_order(flat);
                if deduped.len() == 1 {
                    deduped.into_iter().next().unwrap()
                } else {
                    DomainExpr::Union(deduped)
                }
            }
            DomainExpr::Intersection(children) => {
                let flat = flatten_intersection(children);
                let deduped = dedup_preserving_order(flat);
                if deduped.len() == 1 {
                    deduped.into_iter().next().unwrap()
                } else {
                    DomainExpr::Intersection(deduped)
                }
            }
            DomainExpr::Complement(inner) => {
                let inner = inner.simplify();
                match inner {
                    DomainExpr::Complement(grand) => *grand,
                    other => DomainExpr::Complement(Box::new(other)),
                }
            }
            other => other,
        }
    }
}

fn flatten_union(children: Vec<DomainExpr>) -> Vec<DomainExpr> {
    let mut out = Vec::with_capacity(children.len());
    for c in children {
        match c.simplify() {
            DomainExpr::Union(inner) => out.extend(inner),
            other => out.push(other),
        }
    }
    out
}

fn flatten_intersection(children: Vec<DomainExpr>) -> Vec<DomainExpr> {
    let mut out = Vec::with_capacity(children.len());
    for c in children {
        match c.simplify() {
            DomainExpr::Intersection(inner) => out.extend(inner),
            other => out.push(other),
        }
    }
    out
}

fn dedup_preserving_order(children: Vec<DomainExpr>) -> Vec<DomainExpr> {
    let mut out: Vec<DomainExpr> = Vec::with_capacity(children.len());
    for c in children {
        if !out.iter().any(|existing| existing == &c) {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_domain_inclusion_chain() {
        assert!(BaseDomain::Natural.is_subset_of(BaseDomain::Integer));
        assert!(BaseDomain::Integer.is_subset_of(BaseDomain::Rational));
        assert!(BaseDomain::Rational.is_subset_of(BaseDomain::Real));
        assert!(BaseDomain::Real.is_subset_of(BaseDomain::Complex));
        assert!(BaseDomain::Complex.is_subset_of(BaseDomain::Quaternion));
        assert!(!BaseDomain::Real.is_subset_of(BaseDomain::Integer));
    }

    #[test]
    fn base_domain_intersect_takes_smaller() {
        assert_eq!(
            BaseDomain::Real.intersect(BaseDomain::Integer),
            BaseDomain::Integer
        );
        assert_eq!(
            BaseDomain::Complex.intersect(BaseDomain::Natural),
            BaseDomain::Natural
        );
    }

    #[test]
    fn base_domain_union_takes_larger() {
        assert_eq!(
            BaseDomain::Real.union(BaseDomain::Integer),
            BaseDomain::Real
        );
        assert_eq!(
            BaseDomain::Complex.union(BaseDomain::Quaternion),
            BaseDomain::Quaternion
        );
    }

    #[test]
    fn qualifier_intersect() {
        let pos = Qualifier::positive_strict();
        let nonneg = Qualifier::non_negative();
        let meet = pos.intersect(nonneg);
        assert!(meet.positive);
        assert!(!meet.negative);
        assert!(matches!(meet.zero, Inclusion::Excluded));
    }

    #[test]
    fn qualifier_empty_when_all_excluded() {
        let empty = Qualifier {
            zero: Inclusion::Excluded,
            positive: false,
            negative: false,
        };
        assert!(empty.is_empty());
        assert!(!Qualifier::full().is_empty());
    }

    #[test]
    fn domain_intersect_real_positive_and_integer() {
        // ℝ⁺ ∩ ℤ = ℤ⁺
        let rp = Domain::real_positive();
        let z = Domain::integer();
        let meet = rp.intersect(z).unwrap();
        assert_eq!(meet.base, BaseDomain::Integer);
        assert!(meet.qualifier.positive);
        assert!(!meet.qualifier.negative);
        assert!(matches!(meet.qualifier.zero, Inclusion::Excluded));
    }

    #[test]
    fn domain_intersect_empty_qualifier_returns_none() {
        // ℝ⁺ ∩ ℝ⁻ is empty at the qualifier level.
        let rp = Domain::real_positive();
        let rn = Domain {
            base: BaseDomain::Real,
            qualifier: Qualifier::negative_strict(),
        };
        assert!(rp.intersect(rn).is_none());
    }

    #[test]
    fn domain_union_real_positive_with_real_negative() {
        let rp = Domain::real_positive();
        let rn = Domain {
            base: BaseDomain::Real,
            qualifier: Qualifier::negative_strict(),
        };
        let join = rp.union(rn);
        assert_eq!(join.base, BaseDomain::Real);
        assert!(join.qualifier.positive);
        assert!(join.qualifier.negative);
        assert!(matches!(join.qualifier.zero, Inclusion::Excluded));
    }

    #[test]
    fn domain_is_subset_positive_chain() {
        assert!(Domain::natural().is_subset_of(Domain::integer()));
        assert!(Domain::integer().is_subset_of(Domain::rational()));
        assert!(Domain::rational().is_subset_of(Domain::real()));
        assert!(Domain::real_positive().is_subset_of(Domain::real()));
    }

    #[test]
    fn domain_expr_simplify_flattens_nested_unions() {
        // Union(Union(A, B), C) → Union(A, B, C)
        let a = DomainExpr::Base(Domain::integer());
        let b = DomainExpr::Base(Domain::rational());
        let c = DomainExpr::Base(Domain::real());
        let nested = DomainExpr::Union(vec![
            DomainExpr::Union(vec![a.clone(), b.clone()]),
            c.clone(),
        ]);
        let flat = nested.simplify();
        match flat {
            DomainExpr::Union(children) => {
                assert_eq!(children.len(), 3);
                assert_eq!(children[0], a);
                assert_eq!(children[1], b);
                assert_eq!(children[2], c);
            }
            _ => panic!("expected Union"),
        }
    }

    #[test]
    fn domain_expr_simplify_dedupes() {
        let r = DomainExpr::Base(Domain::real());
        let nested = DomainExpr::Union(vec![r.clone(), r.clone(), r.clone()]);
        let flat = nested.simplify();
        assert_eq!(flat, r);
    }

    #[test]
    fn domain_expr_simplify_singleton_union_collapses() {
        let r = DomainExpr::Base(Domain::real());
        let single = DomainExpr::Union(vec![r.clone()]);
        assert_eq!(single.simplify(), r);
    }

    #[test]
    fn domain_expr_simplify_double_complement() {
        let r = DomainExpr::Base(Domain::real());
        let double = DomainExpr::Complement(Box::new(DomainExpr::Complement(Box::new(r.clone()))));
        assert_eq!(double.simplify(), r);
    }

    #[test]
    fn domain_expr_simplify_flattens_intersections() {
        let a = DomainExpr::Base(Domain::integer());
        let b = DomainExpr::Base(Domain::rational());
        let c = DomainExpr::Base(Domain::real());
        let nested = DomainExpr::Intersection(vec![
            DomainExpr::Intersection(vec![a.clone(), b.clone()]),
            c.clone(),
        ]);
        let flat = nested.simplify();
        match flat {
            DomainExpr::Intersection(children) => {
                assert_eq!(children.len(), 3);
                assert_eq!(children[0], a);
                assert_eq!(children[1], b);
                assert_eq!(children[2], c);
            }
            _ => panic!("expected Intersection"),
        }
    }
}
