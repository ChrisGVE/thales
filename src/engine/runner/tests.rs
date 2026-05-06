use super::*;
use crate::engine::reason::{ImpossibilityProof, PartialReason};
use crate::engine::resource::ResourceBudget;
use crate::engine::strategy::StrategyResult;
use crate::engine::trace_tree::{BranchReason, TraceNode};
use crate::numeric::SmallInt;

fn int_expr(n: i64) -> Arc<Expr> {
    Arc::new(Expr::Integer(SmallInt::from(n)))
}

fn dummy_trace() -> TraceNode {
    TraceNode::Branch {
        reason: BranchReason::StrategyCascade,
        children: vec![],
    }
}

fn base_ctx() -> SolveContext {
    SolveContext::new(int_expr(1), ResourceBudget::unlimited())
}

// ── Mock strategies ───────────────────────────────────────────────────────

use crate::engine::trace_tree::StrategyId;

#[derive(Debug)]
struct SolveStrategy {
    priority: f64,
    value: i64,
}

impl Strategy for SolveStrategy {
    fn id(&self) -> StrategyId {
        StrategyId("mock::solve")
    }
    fn applicable(&self, _ctx: &SolveContext) -> bool {
        true
    }
    fn priority(&self, _ctx: &SolveContext) -> f64 {
        self.priority
    }
    fn apply(&self, _ctx: SolveContext) -> StrategyResult {
        StrategyResult::Solved {
            expr: int_expr(self.value),
            trace: dummy_trace(),
        }
    }
}

#[derive(Debug)]
struct FailStrategy {
    priority: f64,
}

impl Strategy for FailStrategy {
    fn id(&self) -> StrategyId {
        StrategyId("mock::fail")
    }
    fn applicable(&self, _ctx: &SolveContext) -> bool {
        true
    }
    fn priority(&self, _ctx: &SolveContext) -> f64 {
        self.priority
    }
    fn apply(&self, _ctx: SolveContext) -> StrategyResult {
        StrategyResult::Failed(FailureReason::NotApplicable)
    }
}

#[derive(Debug)]
struct ImpossibleStrategy;

impl Strategy for ImpossibleStrategy {
    fn id(&self) -> StrategyId {
        StrategyId("mock::impossible")
    }
    fn applicable(&self, _ctx: &SolveContext) -> bool {
        true
    }
    fn priority(&self, _ctx: &SolveContext) -> f64 {
        0.0
    }
    fn apply(&self, _ctx: SolveContext) -> StrategyResult {
        StrategyResult::ProvenImpossible {
            certificate: ImpossibilityProof::NoElementaryClosure,
            trace: dummy_trace(),
        }
    }
}

#[derive(Debug)]
struct StructuralErrorStrategy;

impl Strategy for StructuralErrorStrategy {
    fn id(&self) -> StrategyId {
        StrategyId("mock::structural_error")
    }
    fn applicable(&self, _ctx: &SolveContext) -> bool {
        true
    }
    fn priority(&self, _ctx: &SolveContext) -> f64 {
        0.0
    }
    fn apply(&self, _ctx: SolveContext) -> StrategyResult {
        use std::sync::Arc;
        #[derive(Debug)]
        struct TestErr;
        impl std::fmt::Display for TestErr {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "test")
            }
        }
        impl std::error::Error for TestErr {}
        StrategyResult::Failed(FailureReason::StructuralError(Arc::new(TestErr)))
    }
}

#[derive(Debug)]
struct NotApplicableStrategy;

impl Strategy for NotApplicableStrategy {
    fn id(&self) -> StrategyId {
        StrategyId("mock::not_applicable")
    }
    fn applicable(&self, _ctx: &SolveContext) -> bool {
        false
    }
    fn priority(&self, _ctx: &SolveContext) -> f64 {
        0.0
    }
    fn apply(&self, _ctx: SolveContext) -> StrategyResult {
        StrategyResult::Solved {
            expr: int_expr(99),
            trace: dummy_trace(),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[test]
fn fast_runner_sequential_first_success() {
    let runner = SequentialRunner;
    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(FailStrategy { priority: 0.0 }),
        Box::new(SolveStrategy {
            priority: 1.0,
            value: 42,
        }),
    ];
    let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);
    match result {
        StrategyResult::Solved { expr, .. } => {
            assert!(matches!(*expr, Expr::Integer(_)));
        }
        other => panic!("Expected Solved, got {:?}", other),
    }
}

#[test]
fn fast_runner_priority_order_lower_tried_first() {
    let runner = SequentialRunner;
    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(SolveStrategy {
            priority: 2.0,
            value: 10,
        }),
        Box::new(SolveStrategy {
            priority: 0.5,
            value: 7,
        }),
    ];
    let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);
    match result {
        StrategyResult::Solved { expr, .. } => {
            if let Expr::Integer(n) = &*expr {
                assert_eq!(n.to_i64(), Some(7));
            } else {
                panic!("Expected integer result");
            }
        }
        other => panic!("Expected Solved, got {:?}", other),
    }
}

#[test]
fn fast_runner_proven_impossible_terminates_cascade() {
    let runner = SequentialRunner;
    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(ImpossibleStrategy),
        Box::new(SolveStrategy {
            priority: 1.0,
            value: 42,
        }),
    ];
    let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);
    assert!(
        matches!(result, StrategyResult::ProvenImpossible { .. }),
        "Expected ProvenImpossible"
    );
}

#[test]
fn fast_runner_structural_error_terminates_cascade() {
    let runner = SequentialRunner;
    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(StructuralErrorStrategy),
        Box::new(SolveStrategy {
            priority: 1.0,
            value: 42,
        }),
    ];
    let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);
    assert!(
        matches!(
            result,
            StrategyResult::Failed(FailureReason::StructuralError(_))
        ),
        "Expected StructuralError to terminate cascade"
    );
}

#[test]
fn fast_runner_not_applicable_strategy_skipped() {
    let runner = SequentialRunner;
    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(NotApplicableStrategy),
        Box::new(SolveStrategy {
            priority: 1.0,
            value: 5,
        }),
    ];
    let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);
    assert!(matches!(result, StrategyResult::Solved { .. }));
}

#[test]
fn fast_runner_all_exhausted_returns_not_applicable() {
    let runner = SequentialRunner;
    let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(FailStrategy { priority: 0.0 })];
    let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);
    assert!(
        matches!(result, StrategyResult::Failed(FailureReason::NotApplicable)),
        "Expected NotApplicable when all exhausted"
    );
}

#[test]
fn fast_runner_empty_strategies_returns_not_applicable() {
    let runner = SequentialRunner;
    let strategies: Vec<Box<dyn Strategy>> = vec![];
    let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);
    assert!(matches!(
        result,
        StrategyResult::Failed(FailureReason::NotApplicable)
    ));
}

#[test]
fn fast_runner_partial_saved_when_all_exhausted() {
    #[derive(Debug)]
    struct PartialStrategy;
    impl Strategy for PartialStrategy {
        fn id(&self) -> StrategyId {
            StrategyId("mock::partial")
        }
        fn applicable(&self, _: &SolveContext) -> bool {
            true
        }
        fn priority(&self, _: &SolveContext) -> f64 {
            0.0
        }
        fn apply(&self, _: SolveContext) -> StrategyResult {
            StrategyResult::Partial {
                expr: int_expr(3),
                reason: PartialReason::StepLimitReached,
                trace: dummy_trace(),
            }
        }
    }
    let runner = SequentialRunner;
    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(PartialStrategy),
        Box::new(FailStrategy { priority: 1.0 }),
    ];
    let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);
    assert!(
        matches!(result, StrategyResult::Partial { .. }),
        "Expected Partial as best result"
    );
}

// ── RayonRunner tests (rayon feature only) ────────────────────────────────

#[cfg(feature = "rayon")]
#[test]
fn fast_rayon_runner_sequential_mode_finds_solution() {
    let runner = RayonRunner::new(4);
    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(FailStrategy { priority: 0.0 }),
        Box::new(SolveStrategy {
            priority: 1.0,
            value: 77,
        }),
    ];
    let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);
    assert!(
        matches!(result, StrategyResult::Solved { .. }),
        "Expected Solved"
    );
}

#[cfg(feature = "rayon")]
#[test]
fn fast_rayon_runner_tree_search_finds_solution() {
    let runner = RayonRunner::new(4);
    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(SolveStrategy {
            priority: 1.0,
            value: 42,
        }),
        Box::new(FailStrategy { priority: 2.0 }),
    ];
    let result = runner.run(
        base_ctx(),
        &strategies,
        ExecutionMode::TreeSearch {
            max_depth: 2,
            comparison: crate::engine::mode::TreeComparison::FirstSuccess,
        },
    );
    assert!(
        matches!(result, StrategyResult::Solved { .. }),
        "Expected Solved from parallel tree search"
    );
}

#[cfg(feature = "rayon")]
#[test]
fn fast_rayon_runner_zero_depth_tree_search_returns_no_closed_form() {
    let runner = RayonRunner::new(4);
    let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(SolveStrategy {
        priority: 0.0,
        value: 1,
    })];
    let result = runner.run(
        base_ctx(),
        &strategies,
        ExecutionMode::TreeSearch {
            max_depth: 0,
            comparison: crate::engine::mode::TreeComparison::FirstSuccess,
        },
    );
    assert!(
        matches!(result, StrategyResult::Failed(FailureReason::NoClosedForm)),
        "Expected NoClosedForm at depth 0"
    );
}

#[cfg(feature = "rayon")]
#[test]
fn fast_rayon_gate_acquire_and_release() {
    use crate::engine::resource::ResourceGate;
    let gate = ResourceGate::new(2);
    let g1 = gate.acquire();
    let g2 = gate.acquire();
    assert!(g1.is_some());
    assert!(g2.is_some());
    assert_eq!(gate.active_count(), 2);
    assert!(gate.acquire().is_none());
    drop(g1);
    assert_eq!(gate.active_count(), 1);
    assert!(gate.acquire().is_some());
}

// ── Fallback integration tests ────────────────────────────────────────────

use crate::engine::fallback::testutils::{MockNumericalEvaluator, MockOutcome};
use crate::engine::fallback::{FallbackConfig, NumericalEvaluatorRegistry};

fn ctx_with_fallback() -> SolveContext {
    SolveContext::new(int_expr(1), ResourceBudget::unlimited())
        .with_fallback(FallbackConfig::enabled())
}

fn reg_with_success() -> NumericalEvaluatorRegistry {
    let reg = NumericalEvaluatorRegistry::new();
    reg.register(Arc::new(MockNumericalEvaluator::new(
        "runner_test_success",
        0.5,
        MockOutcome::Success {
            digits: 15,
            approximate: true,
        },
    )));
    reg
}

#[test]
fn fast_runner_fallback_disabled_returns_cascade_failure() {
    let runner = SequentialRunner;
    let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(FailStrategy { priority: 0.0 })];
    let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);
    assert!(
        matches!(result, StrategyResult::Failed(FailureReason::NotApplicable)),
        "fallback disabled: cascade failure must be returned unchanged"
    );
}

#[test]
fn fast_runner_fallback_not_invoked_when_cascade_succeeds() {
    let runner = SequentialRunner;
    let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(SolveStrategy {
        priority: 0.0,
        value: 99,
    })];
    let result = runner.run(ctx_with_fallback(), &strategies, ExecutionMode::Sequential);
    match result {
        StrategyResult::Solved { expr, .. } => {
            if let Expr::Integer(n) = &*expr {
                assert_eq!(n.to_i64(), Some(99), "cascade result must be unchanged");
            } else {
                panic!("expected integer");
            }
        }
        other => panic!("expected Solved, got {:?}", other),
    }
}

#[test]
fn fast_runner_fallback_invoked_after_cascade_exhaustion() {
    use crate::engine::fallback::runner::FallbackRunner;
    use crate::engine::fallback::trigger::FallbackTrigger;

    let reg = reg_with_success();
    let ctx = ctx_with_fallback();
    let trigger = FallbackTrigger::StrategyExhaustion {
        strategies_attempted: 1,
        last_reason: FailureReason::NotApplicable,
    };
    let result = FallbackRunner::run_with_registry(&ctx, trigger, &reg);
    assert!(
        result.is_some(),
        "fallback should produce Solved when an evaluator succeeds"
    );
    assert!(matches!(result.unwrap(), StrategyResult::Solved { .. }));
}

#[test]
fn fast_runner_fallback_empty_registry_returns_cascade_result() {
    use crate::engine::fallback::runner::FallbackRunner;
    use crate::engine::fallback::trigger::FallbackTrigger;

    let reg = NumericalEvaluatorRegistry::new();
    let ctx = ctx_with_fallback();
    let trigger = FallbackTrigger::StrategyExhaustion {
        strategies_attempted: 0,
        last_reason: FailureReason::NotApplicable,
    };
    let result = FallbackRunner::run_with_registry(&ctx, trigger, &reg);
    assert!(result.is_none(), "empty registry must return None");
}

#[test]
fn fast_runner_fallback_complexity_gate_triggers_on_threshold_exceeded() {
    use crate::engine::fallback::runner::FallbackRunner;
    use crate::engine::fallback::trigger::FallbackTrigger;

    let reg = reg_with_success();
    let ctx =
        SolveContext::new(int_expr(1), ResourceBudget::unlimited()).with_fallback(FallbackConfig {
            numerical: true,
            numerical_narrative: true,
            complexity_threshold: Some(0),
        });
    let trigger = FallbackTrigger::ComplexityExplosion {
        actual_nodes: 1,
        threshold_nodes: 0,
    };
    let result = FallbackRunner::run_with_registry(&ctx, trigger, &reg);
    assert!(
        result.is_some(),
        "complexity-gate path should invoke FallbackRunner successfully"
    );
}

#[test]
fn fast_runner_fallback_proven_impossible_not_overridden() {
    let runner = SequentialRunner;
    let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(ImpossibleStrategy)];
    let result = runner.run(ctx_with_fallback(), &strategies, ExecutionMode::Sequential);
    assert!(
        matches!(result, StrategyResult::ProvenImpossible { .. }),
        "ProvenImpossible must not be overridden by fallback"
    );
}
