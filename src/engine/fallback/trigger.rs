//! Trigger conditions that activate the numerical fallback path.
//!
//! [`FallbackTrigger`] enumerates every situation that can cause the D0
//! engine to escalate from symbolic search to numerical evaluation.
//! [`ImpossibilityClass`] classifies triggers by whether the impossibility
//! is mathematical, an implementation gap, or a resource constraint.

use crate::engine::reason::{FailureReason, ImpossibilityProof};
use crate::engine::resource::ResourceStatus;

// ── ImpossibilityClass ────────────────────────────────────────────────────────

/// Broad category of why symbolic computation cannot proceed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImpossibilityClass {
    /// A theorem proves no symbolic solution can exist (e.g. Risch, Abel–Ruffini).
    Mathematical,
    /// The engine has not yet implemented the required technique.
    Implementation,
    /// Computation was aborted due to resource exhaustion.
    Resource,
}

// ── FallbackTrigger ───────────────────────────────────────────────────────────

/// The reason the engine decided to fall back to numerical evaluation.
///
/// Constructed by the strategy runner and passed to the fallback executor
/// so that narrative generation can describe exactly why symbolic work
/// stopped and numerical work began.
#[derive(Debug, Clone, PartialEq)]
pub enum FallbackTrigger {
    /// A theorem proves the symbolic problem is inherently unsolvable in the
    /// requested class (e.g. no elementary antiderivative by Risch's theorem).
    MathematicalImpossibility {
        /// The certified proof of impossibility.
        proof: ImpossibilityProof,
        /// Human-readable theorem name (e.g. `"Risch"`, `"Abel-Ruffini"`).
        theorem_name: &'static str,
    },
    /// A previous result stored in the negative cache matches this expression.
    NegativeCacheHit {
        /// The cache key that matched.
        pattern_key: String,
        /// Classification of why this pattern was previously marked impossible.
        classification: ImpossibilityClass,
    },
    /// All registered strategies have been tried and all failed.
    StrategyExhaustion {
        /// Number of strategies attempted before giving up.
        strategies_attempted: usize,
        /// Failure reason reported by the last attempted strategy.
        last_reason: FailureReason,
    },
    /// The shared resource budget (steps, memory, or time) was exceeded before
    /// any symbolic strategy could complete.
    ResourceBudgetExceeded {
        /// The budget status at the point of trigger (always [`ResourceStatus::Exceeded`]).
        status: ResourceStatus,
    },
    /// The expression tree grew beyond the complexity threshold configured in
    /// [`crate::engine::fallback::FallbackConfig::complexity_threshold`].
    ComplexityExplosion {
        /// Actual node count of the expression at trigger time.
        actual_nodes: usize,
        /// The configured threshold that was exceeded.
        threshold_nodes: usize,
    },
}

impl FallbackTrigger {
    /// The broad impossibility class for this trigger.
    ///
    /// Used by callers to decide whether to retry at higher precision
    /// (only meaningful for `Resource` class) or accept the result as final.
    #[must_use]
    pub fn impossibility_class(&self) -> ImpossibilityClass {
        match self {
            FallbackTrigger::MathematicalImpossibility { .. } => ImpossibilityClass::Mathematical,
            FallbackTrigger::NegativeCacheHit { classification, .. } => *classification,
            FallbackTrigger::StrategyExhaustion { .. } => ImpossibilityClass::Implementation,
            FallbackTrigger::ResourceBudgetExceeded { .. } => ImpossibilityClass::Resource,
            FallbackTrigger::ComplexityExplosion { .. } => ImpossibilityClass::Resource,
        }
    }

    /// A human-readable description of why fallback was triggered.
    ///
    /// Suitable for inclusion in a narrative step explaining the transition
    /// from symbolic to numerical evaluation.
    #[must_use]
    pub fn narrative_detail(&self) -> String {
        match self {
            FallbackTrigger::MathematicalImpossibility {
                proof,
                theorem_name,
            } => format!(
                "No symbolic solution exists ({theorem_name} theorem): {}",
                proof.theorem_name()
            ),
            FallbackTrigger::NegativeCacheHit {
                pattern_key,
                classification,
            } => {
                let class_str = match classification {
                    ImpossibilityClass::Mathematical => "mathematically impossible",
                    ImpossibilityClass::Implementation => "not yet implemented",
                    ImpossibilityClass::Resource => "resource-limited",
                };
                format!("Negative cache hit for pattern `{pattern_key}` ({class_str})")
            }
            FallbackTrigger::StrategyExhaustion {
                strategies_attempted,
                ..
            } => format!(
                "All {strategies_attempted} symbolic strateg{} exhausted without result",
                if *strategies_attempted == 1 {
                    "y"
                } else {
                    "ies"
                }
            ),
            FallbackTrigger::ResourceBudgetExceeded { .. } => {
                "Resource budget exceeded before symbolic completion".to_string()
            }
            FallbackTrigger::ComplexityExplosion {
                actual_nodes,
                threshold_nodes,
            } => format!(
                "Expression complexity ({actual_nodes} nodes) exceeded threshold \
                 ({threshold_nodes} nodes)"
            ),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::reason::ImpossibilityProof;

    // ── ImpossibilityClass ────────────────────────────────────────────────────

    #[test]
    fn fast_trigger_impossibility_class_eq() {
        assert_eq!(
            ImpossibilityClass::Mathematical,
            ImpossibilityClass::Mathematical
        );
        assert_ne!(
            ImpossibilityClass::Mathematical,
            ImpossibilityClass::Resource
        );
        assert_ne!(
            ImpossibilityClass::Implementation,
            ImpossibilityClass::Resource
        );
    }

    #[test]
    fn fast_trigger_impossibility_class_copy() {
        let c = ImpossibilityClass::Mathematical;
        let c2 = c; // Copy
        assert_eq!(c, c2);
    }

    // ── FallbackTrigger::impossibility_class ──────────────────────────────────

    #[test]
    fn fast_trigger_mathematical_impossibility_class() {
        let t = FallbackTrigger::MathematicalImpossibility {
            proof: ImpossibilityProof::NoKovacicSolution,
            theorem_name: "Kovacic",
        };
        assert_eq!(t.impossibility_class(), ImpossibilityClass::Mathematical);
    }

    #[test]
    fn fast_trigger_negative_cache_hit_class_propagates() {
        let t = FallbackTrigger::NegativeCacheHit {
            pattern_key: "sin(x)/x".to_string(),
            classification: ImpossibilityClass::Mathematical,
        };
        assert_eq!(t.impossibility_class(), ImpossibilityClass::Mathematical);

        let t2 = FallbackTrigger::NegativeCacheHit {
            pattern_key: "key".to_string(),
            classification: ImpossibilityClass::Implementation,
        };
        assert_eq!(t2.impossibility_class(), ImpossibilityClass::Implementation);
    }

    #[test]
    fn fast_trigger_strategy_exhaustion_class_is_implementation() {
        let t = FallbackTrigger::StrategyExhaustion {
            strategies_attempted: 3,
            last_reason: FailureReason::NotApplicable,
        };
        assert_eq!(t.impossibility_class(), ImpossibilityClass::Implementation);
    }

    #[test]
    fn fast_trigger_resource_budget_class_is_resource() {
        let t = FallbackTrigger::ResourceBudgetExceeded {
            status: ResourceStatus::Exceeded,
        };
        assert_eq!(t.impossibility_class(), ImpossibilityClass::Resource);
    }

    #[test]
    fn fast_trigger_complexity_explosion_class_is_resource() {
        let t = FallbackTrigger::ComplexityExplosion {
            actual_nodes: 1000,
            threshold_nodes: 500,
        };
        assert_eq!(t.impossibility_class(), ImpossibilityClass::Resource);
    }

    // ── FallbackTrigger::narrative_detail ─────────────────────────────────────

    #[test]
    fn fast_trigger_narrative_mathematical_impossibility_nonempty() {
        let t = FallbackTrigger::MathematicalImpossibility {
            proof: ImpossibilityProof::NoElementaryAntiderivative {
                provenance: "Risch",
            },
            theorem_name: "Risch",
        };
        let detail = t.narrative_detail();
        assert!(!detail.is_empty());
        assert!(detail.contains("Risch"));
    }

    #[test]
    fn fast_trigger_narrative_cache_hit_contains_key() {
        let t = FallbackTrigger::NegativeCacheHit {
            pattern_key: "e^(x^2)".to_string(),
            classification: ImpossibilityClass::Mathematical,
        };
        let detail = t.narrative_detail();
        assert!(detail.contains("e^(x^2)"));
        assert!(detail.contains("mathematically impossible"));
    }

    #[test]
    fn fast_trigger_narrative_strategy_exhaustion_singular() {
        let t = FallbackTrigger::StrategyExhaustion {
            strategies_attempted: 1,
            last_reason: FailureReason::NoClosedForm,
        };
        let detail = t.narrative_detail();
        assert!(detail.contains("strategy"));
        assert!(!detail.contains("strategies"));
    }

    #[test]
    fn fast_trigger_narrative_strategy_exhaustion_plural() {
        let t = FallbackTrigger::StrategyExhaustion {
            strategies_attempted: 5,
            last_reason: FailureReason::NoClosedForm,
        };
        let detail = t.narrative_detail();
        assert!(detail.contains("strategies"));
        assert!(detail.contains('5'));
    }

    #[test]
    fn fast_trigger_narrative_resource_budget_nonempty() {
        let t = FallbackTrigger::ResourceBudgetExceeded {
            status: ResourceStatus::Exceeded,
        };
        let detail = t.narrative_detail();
        assert!(!detail.is_empty());
    }

    #[test]
    fn fast_trigger_narrative_complexity_explosion_contains_counts() {
        let t = FallbackTrigger::ComplexityExplosion {
            actual_nodes: 800,
            threshold_nodes: 400,
        };
        let detail = t.narrative_detail();
        assert!(detail.contains("800"));
        assert!(detail.contains("400"));
    }

    // ── FallbackTrigger PartialEq ─────────────────────────────────────────────

    #[test]
    fn fast_trigger_partialeq_same_variant() {
        let t1 = FallbackTrigger::ComplexityExplosion {
            actual_nodes: 100,
            threshold_nodes: 50,
        };
        let t2 = FallbackTrigger::ComplexityExplosion {
            actual_nodes: 100,
            threshold_nodes: 50,
        };
        assert_eq!(t1, t2);
    }

    #[test]
    fn fast_trigger_partialeq_different_values() {
        let t1 = FallbackTrigger::ComplexityExplosion {
            actual_nodes: 100,
            threshold_nodes: 50,
        };
        let t2 = FallbackTrigger::ComplexityExplosion {
            actual_nodes: 200,
            threshold_nodes: 50,
        };
        assert_ne!(t1, t2);
    }
}
