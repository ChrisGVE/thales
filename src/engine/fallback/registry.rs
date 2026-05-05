//! Global registry for [`NumericalEvaluator`] implementations.
//!
//! [`NumericalEvaluatorRegistry`] stores a prioritised list of evaluator
//! instances. The global singleton is accessed via [`global_registry`].
//! Evaluators are keyed by [`StrategyId`]; registering a second evaluator
//! with the same id replaces the first (with a diagnostic warning).

use std::sync::{Arc, OnceLock, RwLock};

use crate::engine::context::SolveContext;
use crate::engine::fallback::evaluator::NumericalEvaluator;
use crate::engine::fallback::trigger::FallbackTrigger;

// ── NumericalEvaluatorRegistry ────────────────────────────────────────────────

/// Thread-safe registry of [`NumericalEvaluator`] instances.
///
/// Evaluators are stored in a `Vec` protected by an `RwLock`. The vec is
/// kept in descending priority order after every `register` call so that
/// `applicable_for` returns candidates pre-sorted for the runner.
pub struct NumericalEvaluatorRegistry {
    evaluators: RwLock<Vec<Arc<dyn NumericalEvaluator>>>,
}

impl NumericalEvaluatorRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            evaluators: RwLock::new(Vec::new()),
        }
    }

    /// Register `evaluator`.
    ///
    /// If an evaluator with the same [`StrategyId`] already exists it is
    /// replaced and a diagnostic message is emitted via `eprintln!`.
    pub fn register(&self, evaluator: Arc<dyn NumericalEvaluator>) {
        let id = evaluator.id();
        let mut guard = self
            .evaluators
            .write()
            .expect("registry write lock poisoned");
        if let Some(pos) = guard.iter().position(|e| e.id() == id) {
            eprintln!(
                "[thales::fallback] replacing duplicate NumericalEvaluator id={:?}",
                id.0
            );
            guard[pos] = evaluator;
        } else {
            guard.push(evaluator);
        }
        // Keep descending priority order.
        guard.sort_by(|a, b| {
            b.priority()
                .partial_cmp(&a.priority())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Return all evaluators whose `applicable` returns `true` for
    /// `(ctx, trigger)`, in descending priority order.
    #[must_use]
    pub fn applicable_for(
        &self,
        ctx: &SolveContext,
        trigger: &FallbackTrigger,
    ) -> Vec<Arc<dyn NumericalEvaluator>> {
        let guard = self.evaluators.read().expect("registry read lock poisoned");
        guard
            .iter()
            .filter(|e| e.applicable(ctx, trigger))
            .cloned()
            .collect()
    }

    /// Total number of registered evaluators (applicable or not).
    #[must_use]
    pub fn len(&self) -> usize {
        self.evaluators
            .read()
            .expect("registry read lock poisoned")
            .len()
    }

    /// Returns `true` when no evaluators are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for NumericalEvaluatorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for NumericalEvaluatorRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let guard = self.evaluators.read().expect("registry read lock poisoned");
        f.debug_struct("NumericalEvaluatorRegistry")
            .field("count", &guard.len())
            .finish()
    }
}

// ── Global registry ───────────────────────────────────────────────────────────

static REGISTRY: OnceLock<NumericalEvaluatorRegistry> = OnceLock::new();

/// Return a reference to the process-global [`NumericalEvaluatorRegistry`].
///
/// The registry is initialised on first call and lives for the duration of
/// the process. Use [`NumericalEvaluatorRegistry::register`] to add
/// evaluators before any fallback run.
#[must_use]
pub fn global_registry() -> &'static NumericalEvaluatorRegistry {
    REGISTRY.get_or_init(NumericalEvaluatorRegistry::new)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::engine::fallback::testutils::{MockNumericalEvaluator, MockOutcome};
    use crate::engine::resource::ResourceBudget;
    use crate::engine::trace_tree::StrategyId;
    use crate::numeric::{Expr, SmallInt};

    fn base_ctx() -> SolveContext {
        SolveContext::new(
            Arc::new(Expr::Integer(SmallInt::from(1))),
            ResourceBudget::unlimited(),
        )
    }

    fn exhaustion_trigger() -> FallbackTrigger {
        use crate::engine::reason::FailureReason;
        FallbackTrigger::StrategyExhaustion {
            strategies_attempted: 1,
            last_reason: FailureReason::NotApplicable,
        }
    }

    fn failed_mock(id: &'static str, priority: f64) -> Arc<dyn NumericalEvaluator> {
        Arc::new(MockNumericalEvaluator::new(
            id,
            priority,
            MockOutcome::Failed("x".into()),
        ))
    }

    #[test]
    fn fast_registry_new_is_empty() {
        let reg = NumericalEvaluatorRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn fast_registry_register_single_evaluator() {
        let reg = NumericalEvaluatorRegistry::new();
        reg.register(failed_mock("ev1", 0.5));
        assert_eq!(reg.len(), 1);
        assert!(!reg.is_empty());
    }

    #[test]
    fn fast_registry_register_multiple_distinct_ids() {
        let reg = NumericalEvaluatorRegistry::new();
        reg.register(failed_mock("ev1", 0.5));
        reg.register(failed_mock("ev2", 0.8));
        reg.register(failed_mock("ev3", 0.3));
        assert_eq!(reg.len(), 3);
    }

    #[test]
    fn fast_registry_register_duplicate_id_replaces() {
        let reg = NumericalEvaluatorRegistry::new();
        reg.register(failed_mock("ev1", 0.5));
        reg.register(failed_mock("ev1", 0.9)); // same id
        assert_eq!(reg.len(), 1);
        // The replacement should have priority 0.9.
        let applicable = reg.applicable_for(&base_ctx(), &exhaustion_trigger());
        assert_eq!(applicable.len(), 1);
        assert!((applicable[0].priority() - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn fast_registry_applicable_for_all_applicable() {
        let reg = NumericalEvaluatorRegistry::new();
        reg.register(failed_mock("ev1", 0.5));
        reg.register(failed_mock("ev2", 0.8));
        let applicable = reg.applicable_for(&base_ctx(), &exhaustion_trigger());
        assert_eq!(applicable.len(), 2);
    }

    #[test]
    fn fast_registry_applicable_for_excludes_non_applicable() {
        let reg = NumericalEvaluatorRegistry::new();
        reg.register(failed_mock("ev1", 0.5));
        // Register a non-applicable evaluator.
        let not_applicable: Arc<dyn NumericalEvaluator> = Arc::new(
            MockNumericalEvaluator::new("ev_no", 0.9, MockOutcome::Failed("x".into()))
                .with_applicable(false),
        );
        reg.register(not_applicable);
        let applicable = reg.applicable_for(&base_ctx(), &exhaustion_trigger());
        assert_eq!(applicable.len(), 1);
        assert_eq!(applicable[0].id(), StrategyId("ev1"));
    }

    #[test]
    fn fast_registry_applicable_for_sorted_descending_priority() {
        let reg = NumericalEvaluatorRegistry::new();
        reg.register(failed_mock("low", 0.1));
        reg.register(failed_mock("high", 0.9));
        reg.register(failed_mock("mid", 0.5));
        let applicable = reg.applicable_for(&base_ctx(), &exhaustion_trigger());
        assert_eq!(applicable.len(), 3);
        assert_eq!(applicable[0].id(), StrategyId("high"));
        assert_eq!(applicable[1].id(), StrategyId("mid"));
        assert_eq!(applicable[2].id(), StrategyId("low"));
    }

    #[test]
    fn fast_registry_applicable_for_empty_registry_returns_empty() {
        let reg = NumericalEvaluatorRegistry::new();
        let applicable = reg.applicable_for(&base_ctx(), &exhaustion_trigger());
        assert!(applicable.is_empty());
    }

    #[test]
    fn fast_registry_debug_shows_count() {
        let reg = NumericalEvaluatorRegistry::new();
        reg.register(failed_mock("ev1", 0.5));
        let s = format!("{:?}", reg);
        assert!(s.contains("NumericalEvaluatorRegistry"));
        assert!(s.contains('1'));
    }

    #[test]
    fn fast_registry_default_is_empty() {
        let reg = NumericalEvaluatorRegistry::default();
        assert!(reg.is_empty());
    }
}
