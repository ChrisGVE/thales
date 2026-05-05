//! Reason types for engine outcomes.
//!
//! Captures why a strategy failed, was partial, or why a result is impossible,
//! and what additional resources would be needed.

use std::sync::Arc;

// ── FailureReason ─────────────────────────────────────────────────────────────

/// The reason a strategy or sub-problem failed.
#[derive(Debug, Clone)]
pub enum FailureReason {
    /// The technique is not applicable to this expression structure.
    NotApplicable,
    /// No closed-form solution exists within the searched space.
    NoClosedForm,
    /// A guard condition rejected the strategy before any work was done.
    GuardRejected,
    /// An unexpected structural or type error occurred.
    StructuralError(Arc<dyn std::error::Error + Send + Sync>),
    /// A required sub-problem failed for the given reason.
    SubProblemFailed(Box<FailureReason>),
    /// A custom failure description.
    Custom(String),
}

impl PartialEq for FailureReason {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (FailureReason::NotApplicable, FailureReason::NotApplicable) => true,
            (FailureReason::NoClosedForm, FailureReason::NoClosedForm) => true,
            (FailureReason::GuardRejected, FailureReason::GuardRejected) => true,
            // StructuralError: compare by discriminant only (dyn Error is not PartialEq)
            (FailureReason::StructuralError(_), FailureReason::StructuralError(_)) => true,
            (FailureReason::SubProblemFailed(a), FailureReason::SubProblemFailed(b)) => a == b,
            (FailureReason::Custom(a), FailureReason::Custom(b)) => a == b,
            _ => false,
        }
    }
}

// ── ImpossibilityProof ────────────────────────────────────────────────────────

/// Evidence that a result is provably impossible within a given class.
#[derive(Debug, Clone, PartialEq)]
pub enum ImpossibilityProof {
    /// No elementary closure exists (Liouville/Risch theorem).
    ///
    /// # Deprecation
    ///
    /// Prefer [`ImpossibilityProof::NoElementaryAntiderivative`] with
    /// `provenance` set to `"Risch"` or `"Liouville"`.
    #[deprecated(
        since = "0.9.0",
        note = "use NoElementaryAntiderivative { provenance: \"Risch\" } or \
                NoElementaryAntiderivative { provenance: \"Liouville\" } instead"
    )]
    NoElementaryClosure,
    /// No elementary antiderivative via Liouville's theorem.
    ///
    /// # Deprecation
    ///
    /// Prefer [`ImpossibilityProof::NoElementaryAntiderivative`] with
    /// `provenance` set to `"Liouville"`.
    #[deprecated(
        since = "0.9.0",
        note = "use NoElementaryAntiderivative { provenance: \"Liouville\" } instead"
    )]
    NoLiouvillePrimitive,
    /// No Liouvillian solution via Kovacic's algorithm.
    NoKovacicSolution,
    /// No elementary antiderivative exists, as certified by the given decision
    /// procedure.
    ///
    /// `provenance` must be one of:
    /// - `"Risch"` — full Risch decision procedure
    /// - `"Liouville"` — Liouville's structure theorem
    NoElementaryAntiderivative { provenance: &'static str },
    /// No solution in radicals exists for this polynomial.
    ///
    /// Applies to general polynomials with symbolic coefficients. For
    /// specific fixed-coefficient polynomials of degree ≥ 5, solvability
    /// depends on the Galois group of the polynomial — a certificate is
    /// required (deferred to D2).
    NoRadicalSolution { degree: usize },
    /// Custom impossibility description.
    Custom(String),
}

impl ImpossibilityProof {
    /// A short name for the theorem or algorithm that certifies this
    /// impossibility, suitable for use in narrative output.
    #[must_use]
    #[allow(deprecated)]
    pub fn theorem_name(&self) -> &'static str {
        match self {
            ImpossibilityProof::NoElementaryClosure => "Liouville-Risch",
            ImpossibilityProof::NoLiouvillePrimitive => "Liouville",
            ImpossibilityProof::NoKovacicSolution => "Kovacic",
            ImpossibilityProof::NoElementaryAntiderivative { provenance } => provenance,
            ImpossibilityProof::NoRadicalSolution { .. } => "Abel-Ruffini",
            ImpossibilityProof::Custom(_) => "custom",
        }
    }
}

// ── PartialReason ─────────────────────────────────────────────────────────────

/// Why a computation returned a partial result.
#[derive(Debug, Clone, PartialEq)]
pub enum PartialReason {
    /// The step budget was exhausted before completion.
    StepLimitReached,
    /// The memory budget was exhausted before completion.
    MemoryLimitReached,
    /// Numeric precision was insufficient to continue symbolically.
    PrecisionLimit,
    /// A required sub-problem returned partial for the given reason.
    SubProblemPartial(Box<PartialReason>),
    /// Custom partial reason description.
    Custom(String),
}

// ── ResourceRequest ───────────────────────────────────────────────────────────

/// A request for additional resources to complete a computation.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceRequest {
    /// Number of additional steps requested.
    pub steps: Option<u32>,
    /// Additional memory requested, in bytes.
    pub memory_bytes: Option<usize>,
}

impl ResourceRequest {
    /// Construct a new resource request.
    #[must_use]
    pub fn new(steps: Option<u32>, memory_bytes: Option<usize>) -> Self {
        ResourceRequest {
            steps,
            memory_bytes,
        }
    }

    /// Request additional steps only.
    #[must_use]
    pub fn steps(n: u32) -> Self {
        ResourceRequest {
            steps: Some(n),
            memory_bytes: None,
        }
    }

    /// Request additional memory only.
    #[must_use]
    pub fn memory(bytes: usize) -> Self {
        ResourceRequest {
            steps: None,
            memory_bytes: Some(bytes),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_failure_reason_eq_same_variants() {
        assert_eq!(FailureReason::NotApplicable, FailureReason::NotApplicable);
        assert_eq!(FailureReason::NoClosedForm, FailureReason::NoClosedForm);
        assert_eq!(FailureReason::GuardRejected, FailureReason::GuardRejected);
        assert_eq!(
            FailureReason::Custom("x".to_string()),
            FailureReason::Custom("x".to_string())
        );
    }

    #[test]
    fn fast_failure_reason_ne_different_variants() {
        assert_ne!(FailureReason::NotApplicable, FailureReason::NoClosedForm);
        assert_ne!(
            FailureReason::Custom("a".to_string()),
            FailureReason::Custom("b".to_string())
        );
    }

    #[test]
    fn fast_failure_reason_structural_error_eq_by_discriminant() {
        use std::fmt;

        #[derive(Debug)]
        struct DummyErr;
        impl fmt::Display for DummyErr {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "dummy")
            }
        }
        impl std::error::Error for DummyErr {}

        let a = FailureReason::StructuralError(Arc::new(DummyErr));
        let b = FailureReason::StructuralError(Arc::new(DummyErr));
        // Both are StructuralError variant → equal by discriminant rule
        assert_eq!(a, b);
    }

    #[test]
    fn fast_failure_reason_sub_problem_nested() {
        let inner = FailureReason::NotApplicable;
        let outer = FailureReason::SubProblemFailed(Box::new(inner.clone()));
        let outer2 = FailureReason::SubProblemFailed(Box::new(inner));
        assert_eq!(outer, outer2);
    }

    #[test]
    #[allow(deprecated)]
    fn fast_impossibility_proof_eq() {
        assert_eq!(
            ImpossibilityProof::NoElementaryClosure,
            ImpossibilityProof::NoElementaryClosure
        );
        assert_ne!(
            ImpossibilityProof::NoLiouvillePrimitive,
            ImpossibilityProof::NoKovacicSolution
        );
        assert_eq!(
            ImpossibilityProof::Custom("test".to_string()),
            ImpossibilityProof::Custom("test".to_string())
        );
    }

    #[test]
    fn fast_impossibility_proof_new_variants_eq() {
        assert_eq!(
            ImpossibilityProof::NoElementaryAntiderivative {
                provenance: "Risch"
            },
            ImpossibilityProof::NoElementaryAntiderivative {
                provenance: "Risch"
            },
        );
        assert_ne!(
            ImpossibilityProof::NoElementaryAntiderivative {
                provenance: "Risch"
            },
            ImpossibilityProof::NoElementaryAntiderivative {
                provenance: "Liouville"
            },
        );
        assert_eq!(
            ImpossibilityProof::NoRadicalSolution { degree: 5 },
            ImpossibilityProof::NoRadicalSolution { degree: 5 },
        );
        assert_ne!(
            ImpossibilityProof::NoRadicalSolution { degree: 5 },
            ImpossibilityProof::NoRadicalSolution { degree: 6 },
        );
    }

    #[test]
    fn fast_impossibility_proof_theorem_name_new_variants() {
        assert_eq!(
            ImpossibilityProof::NoElementaryAntiderivative {
                provenance: "Risch"
            }
            .theorem_name(),
            "Risch",
        );
        assert_eq!(
            ImpossibilityProof::NoElementaryAntiderivative {
                provenance: "Liouville"
            }
            .theorem_name(),
            "Liouville",
        );
        assert_eq!(
            ImpossibilityProof::NoRadicalSolution { degree: 5 }.theorem_name(),
            "Abel-Ruffini",
        );
        assert_eq!(
            ImpossibilityProof::NoKovacicSolution.theorem_name(),
            "Kovacic",
        );
        assert_eq!(
            ImpossibilityProof::Custom("special".to_string()).theorem_name(),
            "custom",
        );
    }

    #[test]
    #[allow(deprecated)]
    fn fast_impossibility_proof_theorem_name_deprecated_variants() {
        assert_eq!(
            ImpossibilityProof::NoElementaryClosure.theorem_name(),
            "Liouville-Risch",
        );
        assert_eq!(
            ImpossibilityProof::NoLiouvillePrimitive.theorem_name(),
            "Liouville",
        );
    }

    #[test]
    fn fast_partial_reason_eq() {
        assert_eq!(
            PartialReason::StepLimitReached,
            PartialReason::StepLimitReached
        );
        assert_ne!(
            PartialReason::StepLimitReached,
            PartialReason::MemoryLimitReached
        );
        let inner = PartialReason::PrecisionLimit;
        let nested = PartialReason::SubProblemPartial(Box::new(inner.clone()));
        let nested2 = PartialReason::SubProblemPartial(Box::new(inner));
        assert_eq!(nested, nested2);
    }

    #[test]
    fn fast_resource_request_constructors() {
        let r = ResourceRequest::steps(100);
        assert_eq!(r.steps, Some(100));
        assert_eq!(r.memory_bytes, None);

        let r = ResourceRequest::memory(1024);
        assert_eq!(r.steps, None);
        assert_eq!(r.memory_bytes, Some(1024));

        let r = ResourceRequest::new(Some(50), Some(512));
        assert_eq!(r.steps, Some(50));
        assert_eq!(r.memory_bytes, Some(512));
    }
}
