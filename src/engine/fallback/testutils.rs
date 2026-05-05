//! Test utilities for numerical fallback unit tests.
//!
//! This module is compiled only when `cfg(test)` is active. It provides
//! [`MockNumericalEvaluator`] and [`MockOutcome`] for use in fallback tests
//! across the crate without duplicating the scaffolding in each test module.

use std::sync::Arc;

use crate::engine::context::SolveContext;
use crate::engine::fallback::evaluator::NumericalEvaluator;
use crate::engine::fallback::precision::{
    NumericalResult, PrecisionAttemptOutcome, PrecisionLevel,
};
use crate::engine::fallback::trigger::FallbackTrigger;
use crate::engine::trace_tree::StrategyId;
use crate::numeric::{Expr, SmallInt};

// ── MockOutcome ───────────────────────────────────────────────────────────────

/// The result that a [`MockNumericalEvaluator`] is configured to return
/// when its `evaluate` method is called.
#[derive(Debug, Clone)]
pub enum MockOutcome {
    /// Evaluation succeeded at any requested precision level.
    Success {
        /// Digits of precision to report in the result.
        digits: u32,
        /// Whether the result is marked approximate.
        approximate: bool,
    },
    /// Precision was insufficient; carry partial progress to the next level.
    Insufficient {
        /// Digits actually achieved.
        achieved: u32,
        /// Digits the problem required.
        required: u32,
    },
    /// Hard failure (domain error, unsupported expression, etc.).
    Failed(String),
    /// Budget was exhausted before the attempt could complete.
    BudgetExhausted,
}

// ── MockNumericalEvaluator ────────────────────────────────────────────────────

/// A configurable mock that implements [`NumericalEvaluator`] for unit tests.
///
/// `id` and `priority` are fixed at construction time. `outcome` determines
/// what `evaluate` returns regardless of the precision level requested (unless
/// overridden by per-level logic in future extensions).
#[derive(Debug)]
pub struct MockNumericalEvaluator {
    id: &'static str,
    priority: f64,
    outcome: MockOutcome,
    applicable: bool,
    min_precision: PrecisionLevel,
    max_precision: PrecisionLevel,
}

impl MockNumericalEvaluator {
    /// Build a mock with the given id, priority, and outcome.
    ///
    /// `applicable` defaults to `true`; use [`Self::with_applicable`] to
    /// override. Precision bounds default to the full chain.
    #[must_use]
    pub fn new(id: &'static str, priority: f64, outcome: MockOutcome) -> Self {
        Self {
            id,
            priority,
            outcome,
            applicable: true,
            min_precision: PrecisionLevel::F64,
            max_precision: PrecisionLevel::BigDecimal512,
        }
    }

    /// Override the `applicable` return value.
    #[must_use]
    pub fn with_applicable(mut self, applicable: bool) -> Self {
        self.applicable = applicable;
        self
    }

    /// Override the minimum precision level.
    #[must_use]
    pub fn with_min_precision(mut self, level: PrecisionLevel) -> Self {
        self.min_precision = level;
        self
    }

    /// Override the maximum precision level.
    #[must_use]
    pub fn with_max_precision(mut self, level: PrecisionLevel) -> Self {
        self.max_precision = level;
        self
    }
}

impl NumericalEvaluator for MockNumericalEvaluator {
    fn id(&self) -> StrategyId {
        StrategyId(self.id)
    }

    fn priority(&self) -> f64 {
        self.priority
    }

    fn applicable(&self, _ctx: &SolveContext, _trigger: &FallbackTrigger) -> bool {
        self.applicable
    }

    fn evaluate(
        &self,
        _ctx: &SolveContext,
        _trigger: &FallbackTrigger,
        precision: PrecisionLevel,
    ) -> PrecisionAttemptOutcome {
        match &self.outcome {
            MockOutcome::Success {
                digits,
                approximate,
            } => PrecisionAttemptOutcome::Success(NumericalResult {
                value: Arc::new(Expr::Integer(SmallInt::from(0))),
                precision,
                digits_achieved: *digits,
                error_bound: None,
                approximate: *approximate,
                precision_loss: false,
                evaluator_id: StrategyId(self.id),
            }),
            MockOutcome::Insufficient { achieved, required } => {
                PrecisionAttemptOutcome::InsufficientPrecision {
                    partial: Arc::new(Expr::Integer(SmallInt::from(0))),
                    digits_achieved: *achieved,
                    digits_required: *required,
                }
            }
            MockOutcome::Failed(msg) => PrecisionAttemptOutcome::Failed(msg.clone()),
            MockOutcome::BudgetExhausted => PrecisionAttemptOutcome::BudgetExhausted,
        }
    }

    fn min_precision(&self) -> PrecisionLevel {
        self.min_precision
    }

    fn max_precision(&self) -> PrecisionLevel {
        self.max_precision
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::engine::resource::ResourceBudget;

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

    #[test]
    fn fast_mock_evaluator_success_outcome() {
        let ev = MockNumericalEvaluator::new(
            "mock",
            0.5,
            MockOutcome::Success {
                digits: 15,
                approximate: true,
            },
        );
        let outcome = ev.evaluate(&base_ctx(), &exhaustion_trigger(), PrecisionLevel::F64);
        assert!(matches!(outcome, PrecisionAttemptOutcome::Success(_)));
    }

    #[test]
    fn fast_mock_evaluator_insufficient_outcome() {
        let ev = MockNumericalEvaluator::new(
            "mock",
            0.5,
            MockOutcome::Insufficient {
                achieved: 10,
                required: 50,
            },
        );
        let outcome = ev.evaluate(&base_ctx(), &exhaustion_trigger(), PrecisionLevel::F64);
        assert!(matches!(
            outcome,
            PrecisionAttemptOutcome::InsufficientPrecision { .. }
        ));
    }

    #[test]
    fn fast_mock_evaluator_failed_outcome() {
        let ev =
            MockNumericalEvaluator::new("mock", 0.5, MockOutcome::Failed("domain error".into()));
        let outcome = ev.evaluate(&base_ctx(), &exhaustion_trigger(), PrecisionLevel::F64);
        assert!(matches!(outcome, PrecisionAttemptOutcome::Failed(_)));
    }

    #[test]
    fn fast_mock_evaluator_budget_exhausted_outcome() {
        let ev = MockNumericalEvaluator::new("mock", 0.5, MockOutcome::BudgetExhausted);
        let outcome = ev.evaluate(&base_ctx(), &exhaustion_trigger(), PrecisionLevel::F64);
        assert!(matches!(outcome, PrecisionAttemptOutcome::BudgetExhausted));
    }

    #[test]
    fn fast_mock_evaluator_not_applicable() {
        let ev = MockNumericalEvaluator::new("mock", 0.5, MockOutcome::Failed("x".into()))
            .with_applicable(false);
        assert!(!ev.applicable(&base_ctx(), &exhaustion_trigger()));
    }

    #[test]
    fn fast_mock_evaluator_custom_precision_bounds() {
        let ev = MockNumericalEvaluator::new("mock", 0.5, MockOutcome::Failed("x".into()))
            .with_min_precision(PrecisionLevel::BigDecimal128)
            .with_max_precision(PrecisionLevel::BigDecimal256);
        assert_eq!(ev.min_precision(), PrecisionLevel::BigDecimal128);
        assert_eq!(ev.max_precision(), PrecisionLevel::BigDecimal256);
    }
}
