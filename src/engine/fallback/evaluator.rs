//! [`NumericalEvaluator`] trait — the plug-in interface for numerical fallback.
//!
//! Every concrete numerical evaluation backend (F64 eval, interval arithmetic,
//! multi-precision decimal, …) implements this trait. The [`FallbackRunner`]
//! discovers applicable evaluators via [`NumericalEvaluatorRegistry`] and
//! drives the precision-escalation loop.
//!
//! [`FallbackRunner`]: crate::engine::fallback::runner::FallbackRunner

use crate::engine::context::SolveContext;
use crate::engine::fallback::precision::{PrecisionAttemptOutcome, PrecisionLevel};
use crate::engine::fallback::trigger::FallbackTrigger;
use crate::engine::trace_tree::StrategyId;

// ── NumericalEvaluator ────────────────────────────────────────────────────────

/// Plug-in interface for a single numerical evaluation backend.
///
/// Implementations must be `Send + Sync + 'static` so they can be stored in
/// the global registry and used across threads.
///
/// The `Debug` impl is required for diagnostics and trace reporting.
pub trait NumericalEvaluator: Send + Sync + std::fmt::Debug + 'static {
    /// Stable identifier for this evaluator (used for deduplication and logs).
    fn id(&self) -> StrategyId;

    /// Priority relative to other evaluators. Higher values are tried first.
    ///
    /// Convention: values in `0.0..=1.0` with `1.0` meaning "try me first".
    fn priority(&self) -> f64;

    /// Returns `true` when this evaluator can handle the given context and
    /// trigger combination.
    ///
    /// This is a lightweight structural pre-check; it should not perform
    /// expensive computation.
    fn applicable(&self, ctx: &SolveContext, trigger: &FallbackTrigger) -> bool;

    /// Attempt numerical evaluation at the given precision level.
    ///
    /// The runner calls this after confirming [`applicable`] returns `true`.
    /// Implementations should respect the shared [`ResourceBudget`] in
    /// `ctx.budget` and return [`PrecisionAttemptOutcome::BudgetExhausted`]
    /// if the budget runs out during evaluation.
    ///
    /// [`applicable`]: NumericalEvaluator::applicable
    /// [`ResourceBudget`]: crate::engine::resource::ResourceBudget
    fn evaluate(
        &self,
        ctx: &SolveContext,
        trigger: &FallbackTrigger,
        precision: PrecisionLevel,
    ) -> PrecisionAttemptOutcome;

    /// Lowest precision level this evaluator can use.
    ///
    /// The runner skips all levels below this value in the precision chain.
    /// Default: [`PrecisionLevel::F64`].
    fn min_precision(&self) -> PrecisionLevel {
        PrecisionLevel::F64
    }

    /// Highest precision level this evaluator supports.
    ///
    /// The runner will not request levels above this value.
    /// Default: [`PrecisionLevel::BigDecimal512`].
    fn max_precision(&self) -> PrecisionLevel {
        PrecisionLevel::BigDecimal512
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::engine::fallback::testutils::{MockNumericalEvaluator, MockOutcome};
    use crate::engine::fallback::trigger::FallbackTrigger;
    use crate::engine::reason::FailureReason;

    fn make_trigger(n: usize) -> FallbackTrigger {
        FallbackTrigger::StrategyExhaustion {
            strategies_attempted: n,
            last_reason: FailureReason::NotApplicable,
        }
    }

    #[test]
    fn fast_evaluator_trait_id_returns_strategy_id() {
        let ev = MockNumericalEvaluator::new("test_id", 0.5, MockOutcome::Failed("x".into()));
        assert_eq!(ev.id(), StrategyId("test_id"));
    }

    #[test]
    fn fast_evaluator_trait_priority_returned() {
        let ev = MockNumericalEvaluator::new("ev", 0.8, MockOutcome::Failed("x".into()));
        assert!((ev.priority() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn fast_evaluator_trait_default_min_precision_is_f64() {
        let ev = MockNumericalEvaluator::new("ev", 0.5, MockOutcome::Failed("x".into()));
        assert_eq!(ev.min_precision(), PrecisionLevel::F64);
    }

    #[test]
    fn fast_evaluator_trait_default_max_precision_is_512() {
        let ev = MockNumericalEvaluator::new("ev", 0.5, MockOutcome::Failed("x".into()));
        assert_eq!(ev.max_precision(), PrecisionLevel::BigDecimal512);
    }

    #[test]
    fn fast_evaluator_trait_applicable_always_true_mock() {
        use crate::engine::resource::ResourceBudget;
        let ctx = crate::engine::context::SolveContext::new(
            Arc::new(crate::numeric::Expr::Integer(
                crate::numeric::SmallInt::from(1),
            )),
            ResourceBudget::unlimited(),
        );
        let trigger = make_trigger(1);
        let ev = MockNumericalEvaluator::new("ev", 0.5, MockOutcome::Failed("x".into()));
        assert!(ev.applicable(&ctx, &trigger));
    }
}
