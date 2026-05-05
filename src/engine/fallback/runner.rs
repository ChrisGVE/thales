//! [`FallbackRunner`] — drives the precision-escalation loop.
//!
//! After the symbolic strategy cascade is exhausted, [`FallbackRunner::run`]
//! queries the [`NumericalEvaluatorRegistry`] for applicable evaluators and
//! walks the [`CHAIN`] of precision levels from each evaluator's minimum to
//! its maximum, stopping as soon as one level succeeds.
//!
//! The public entry point [`FallbackRunner::run`] uses the process-global
//! registry. [`FallbackRunner::run_with_registry`] accepts an explicit
//! registry reference for unit-testing without touching global state.

use std::sync::Arc;

use crate::engine::context::SolveContext;
use crate::engine::fallback::precision::{PrecisionAttemptOutcome, CHAIN};
use crate::engine::fallback::registry::{global_registry, NumericalEvaluatorRegistry};
use crate::engine::fallback::trigger::FallbackTrigger;
use crate::engine::strategy::StrategyResult;
use crate::engine::trace_tree::{BranchOutcome, BranchReason, TraceBranch, TraceNode, TraceTree};
use crate::numeric::trace::{Step, TechniqueTag};

// ── FallbackRunner ────────────────────────────────────────────────────────────

/// Stateless driver for the numerical fallback precision-escalation loop.
///
/// There is nothing to construct; all logic is in associated functions.
pub struct FallbackRunner;

impl FallbackRunner {
    /// Run fallback using the process-global [`NumericalEvaluatorRegistry`].
    ///
    /// Returns `Some(StrategyResult::Solved { .. })` on success, `None` when
    /// all evaluators and precision levels are exhausted or the budget runs out.
    #[must_use]
    pub fn run(ctx: &SolveContext, trigger: FallbackTrigger) -> Option<StrategyResult> {
        Self::run_with_registry(ctx, trigger, global_registry())
    }

    /// Run fallback against an explicit registry (useful for tests).
    ///
    /// Algorithm:
    /// 1. Collect applicable evaluators from `registry` (pre-sorted descending
    ///    by priority).
    /// 2. For each evaluator, iterate the precision chain from `min_precision()`
    ///    to `max_precision()` (skipping levels outside that range).
    /// 3. At each level:
    ///    - Return `None` immediately if `ctx.budget` is exhausted.
    ///    - Call `evaluator.evaluate(ctx, &trigger, level)`.
    ///    - `Success` → build `StrategyResult::Solved` with a trace node and
    ///      return `Some`.
    ///    - `InsufficientPrecision` → try the next higher precision level.
    ///    - `Failed` → stop this evaluator and try the next.
    ///    - `BudgetExhausted` → return `None` immediately.
    /// 4. All evaluators and levels exhausted → return `None`.
    #[must_use]
    pub fn run_with_registry(
        ctx: &SolveContext,
        trigger: FallbackTrigger,
        registry: &NumericalEvaluatorRegistry,
    ) -> Option<StrategyResult> {
        let evaluators = registry.applicable_for(ctx, &trigger);
        if evaluators.is_empty() {
            return None;
        }

        for evaluator in &evaluators {
            let min = evaluator.min_precision();
            let max = evaluator.max_precision();

            for &level in CHAIN {
                if level < min || level > max {
                    continue;
                }

                if ctx.budget.is_exhausted() {
                    return None;
                }

                match evaluator.evaluate(ctx, &trigger, level) {
                    PrecisionAttemptOutcome::Success(result) => {
                        let detail = format!(
                            "Numerical fallback succeeded at {} ({} digits) via {}; {}",
                            level.label(),
                            result.digits_achieved,
                            evaluator.id().0,
                            trigger.narrative_detail(),
                        );
                        let step = Step::new(TechniqueTag::NumericalApproximation, detail)
                            .with_output(Arc::clone(&result.value));
                        let mut branch_tree = TraceTree::new();
                        branch_tree.push_step(step);
                        let trace = TraceNode::Branch {
                            reason: BranchReason::NumericalFallback,
                            children: vec![TraceBranch {
                                strategy: evaluator.id(),
                                outcome: BranchOutcome::Succeeded,
                                nodes: branch_tree,
                            }],
                        };
                        return Some(StrategyResult::Solved {
                            expr: result.value,
                            trace,
                        });
                    }
                    PrecisionAttemptOutcome::InsufficientPrecision { .. } => {
                        continue;
                    }
                    PrecisionAttemptOutcome::Failed(_) => {
                        break;
                    }
                    PrecisionAttemptOutcome::BudgetExhausted => {
                        return None;
                    }
                }
            }
        }

        None
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::engine::fallback::precision::PrecisionLevel;
    use crate::engine::fallback::testutils::{MockNumericalEvaluator, MockOutcome};
    use crate::engine::fallback::trigger::FallbackTrigger;
    use crate::engine::reason::FailureReason;
    use crate::engine::resource::ResourceBudget;
    use crate::engine::strategy::StrategyResult;
    use crate::engine::trace_tree::{BranchOutcome, BranchReason, StrategyId};
    use crate::numeric::{Expr, SmallInt};

    fn base_ctx() -> SolveContext {
        SolveContext::new(
            Arc::new(Expr::Integer(SmallInt::from(1))),
            ResourceBudget::unlimited(),
        )
    }

    fn exhaustion_trigger() -> FallbackTrigger {
        FallbackTrigger::StrategyExhaustion {
            strategies_attempted: 2,
            last_reason: FailureReason::NotApplicable,
        }
    }

    fn reg_with(ev: MockNumericalEvaluator) -> NumericalEvaluatorRegistry {
        let reg = NumericalEvaluatorRegistry::new();
        reg.register(Arc::new(ev));
        reg
    }

    #[test]
    fn fast_fallback_runner_empty_registry_returns_none() {
        let reg = NumericalEvaluatorRegistry::new();
        let result = FallbackRunner::run_with_registry(&base_ctx(), exhaustion_trigger(), &reg);
        assert!(result.is_none());
    }

    #[test]
    fn fast_fallback_runner_success_at_f64_returns_solved() {
        let ev = MockNumericalEvaluator::new(
            "ev_f64",
            0.5,
            MockOutcome::Success {
                digits: 15,
                approximate: true,
            },
        );
        let reg = reg_with(ev);
        let result = FallbackRunner::run_with_registry(&base_ctx(), exhaustion_trigger(), &reg);
        assert!(result.is_some());
        assert!(matches!(result.unwrap(), StrategyResult::Solved { .. }));
    }

    #[test]
    fn fast_fallback_runner_escalates_through_insufficient_levels() {
        use std::sync::atomic::{AtomicU8, Ordering};

        let call_count = Arc::new(AtomicU8::new(0));
        let call_count2 = Arc::clone(&call_count);

        #[derive(Debug)]
        struct EscalatingEv {
            calls: Arc<AtomicU8>,
        }

        impl crate::engine::fallback::evaluator::NumericalEvaluator for EscalatingEv {
            fn id(&self) -> StrategyId {
                StrategyId("escalating")
            }
            fn priority(&self) -> f64 {
                0.5
            }
            fn applicable(&self, _: &SolveContext, _: &FallbackTrigger) -> bool {
                true
            }
            fn evaluate(
                &self,
                _: &SolveContext,
                _: &FallbackTrigger,
                level: PrecisionLevel,
            ) -> PrecisionAttemptOutcome {
                let n = self.calls.fetch_add(1, Ordering::SeqCst);
                if level >= PrecisionLevel::BigDecimal128 && n >= 2 {
                    PrecisionAttemptOutcome::Success(
                        crate::engine::fallback::precision::NumericalResult {
                            value: Arc::new(Expr::Integer(SmallInt::from(42))),
                            precision: level,
                            digits_achieved: 38,
                            error_bound: None,
                            approximate: true,
                            precision_loss: false,
                            evaluator_id: StrategyId("escalating"),
                        },
                    )
                } else {
                    PrecisionAttemptOutcome::InsufficientPrecision {
                        partial: Arc::new(Expr::Integer(SmallInt::from(0))),
                        digits_achieved: 10,
                        digits_required: 38,
                    }
                }
            }
        }

        let reg = NumericalEvaluatorRegistry::new();
        reg.register(Arc::new(EscalatingEv { calls: call_count2 }));
        let result = FallbackRunner::run_with_registry(&base_ctx(), exhaustion_trigger(), &reg);
        assert!(result.is_some(), "expected success after escalation");
        assert!(
            call_count.load(std::sync::atomic::Ordering::SeqCst) >= 3,
            "expected >= 3 evaluate calls for precision escalation"
        );
    }

    #[test]
    fn fast_fallback_runner_failed_evaluator_tries_next() {
        let reg = NumericalEvaluatorRegistry::new();
        reg.register(Arc::new(MockNumericalEvaluator::new(
            "ev_fail",
            0.9,
            MockOutcome::Failed("domain error".into()),
        )));
        reg.register(Arc::new(MockNumericalEvaluator::new(
            "ev_ok",
            0.1,
            MockOutcome::Success {
                digits: 15,
                approximate: true,
            },
        )));
        let result = FallbackRunner::run_with_registry(&base_ctx(), exhaustion_trigger(), &reg);
        assert!(result.is_some());
        match result.unwrap() {
            StrategyResult::Solved { trace, .. } => {
                if let TraceNode::Branch { children, .. } = trace {
                    assert_eq!(children[0].strategy, StrategyId("ev_ok"));
                } else {
                    panic!("expected Branch trace node");
                }
            }
            _ => panic!("expected Solved"),
        }
    }

    #[test]
    fn fast_fallback_runner_all_fail_returns_none() {
        let reg = NumericalEvaluatorRegistry::new();
        reg.register(Arc::new(MockNumericalEvaluator::new(
            "ev1",
            0.9,
            MockOutcome::Failed("err1".into()),
        )));
        reg.register(Arc::new(MockNumericalEvaluator::new(
            "ev2",
            0.5,
            MockOutcome::Failed("err2".into()),
        )));
        let result = FallbackRunner::run_with_registry(&base_ctx(), exhaustion_trigger(), &reg);
        assert!(result.is_none());
    }

    #[test]
    fn fast_fallback_runner_budget_exhausted_returns_none() {
        let ev = MockNumericalEvaluator::new("ev", 0.5, MockOutcome::BudgetExhausted);
        let reg = reg_with(ev);
        let result = FallbackRunner::run_with_registry(&base_ctx(), exhaustion_trigger(), &reg);
        assert!(result.is_none());
    }

    #[test]
    fn fast_fallback_runner_non_applicable_skipped() {
        let reg = NumericalEvaluatorRegistry::new();
        reg.register(Arc::new(
            MockNumericalEvaluator::new(
                "ev_no",
                0.9,
                MockOutcome::Success {
                    digits: 15,
                    approximate: false,
                },
            )
            .with_applicable(false),
        ));
        let result = FallbackRunner::run_with_registry(&base_ctx(), exhaustion_trigger(), &reg);
        assert!(result.is_none(), "non-applicable evaluator must be skipped");
    }

    #[test]
    fn fast_fallback_runner_respects_min_max_precision() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let saw_f64 = Arc::new(AtomicBool::new(false));
        let saw_f64_2 = Arc::clone(&saw_f64);

        #[derive(Debug)]
        struct BoundedEv {
            saw_f64: Arc<AtomicBool>,
        }

        impl crate::engine::fallback::evaluator::NumericalEvaluator for BoundedEv {
            fn id(&self) -> StrategyId {
                StrategyId("bounded")
            }
            fn priority(&self) -> f64 {
                0.5
            }
            fn min_precision(&self) -> PrecisionLevel {
                PrecisionLevel::BigDecimal128
            }
            fn max_precision(&self) -> PrecisionLevel {
                PrecisionLevel::BigDecimal256
            }
            fn applicable(&self, _: &SolveContext, _: &FallbackTrigger) -> bool {
                true
            }
            fn evaluate(
                &self,
                _: &SolveContext,
                _: &FallbackTrigger,
                level: PrecisionLevel,
            ) -> PrecisionAttemptOutcome {
                if level < PrecisionLevel::BigDecimal128 {
                    self.saw_f64.store(true, Ordering::SeqCst);
                }
                PrecisionAttemptOutcome::Failed("always".into())
            }
        }

        let reg = NumericalEvaluatorRegistry::new();
        reg.register(Arc::new(BoundedEv { saw_f64: saw_f64_2 }));
        let _ = FallbackRunner::run_with_registry(&base_ctx(), exhaustion_trigger(), &reg);
        assert!(
            !saw_f64.load(std::sync::atomic::Ordering::SeqCst),
            "runner must not call evaluate below min_precision"
        );
    }

    #[test]
    fn fast_fallback_runner_solved_trace_is_numerical_fallback_branch() {
        let ev = MockNumericalEvaluator::new(
            "ev_trace",
            0.5,
            MockOutcome::Success {
                digits: 15,
                approximate: true,
            },
        );
        let reg = reg_with(ev);
        let result =
            FallbackRunner::run_with_registry(&base_ctx(), exhaustion_trigger(), &reg).unwrap();
        match result {
            StrategyResult::Solved { trace, .. } => {
                assert!(
                    matches!(
                        trace,
                        TraceNode::Branch {
                            reason: BranchReason::NumericalFallback,
                            ..
                        }
                    ),
                    "trace root must be NumericalFallback branch"
                );
            }
            _ => panic!("expected Solved"),
        }
    }

    #[test]
    fn fast_fallback_runner_branch_outcome_succeeded() {
        let ev = MockNumericalEvaluator::new(
            "ev_outcome",
            0.5,
            MockOutcome::Success {
                digits: 15,
                approximate: false,
            },
        );
        let reg = reg_with(ev);
        let result =
            FallbackRunner::run_with_registry(&base_ctx(), exhaustion_trigger(), &reg).unwrap();
        match result {
            StrategyResult::Solved { trace, .. } => {
                if let TraceNode::Branch { children, .. } = trace {
                    assert!(!children.is_empty());
                    assert_eq!(children[0].outcome, BranchOutcome::Succeeded);
                } else {
                    panic!("expected Branch node");
                }
            }
            _ => panic!("expected Solved"),
        }
    }
}
