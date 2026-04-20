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
