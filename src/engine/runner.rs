//! Sequential (and stub tree/divide-and-conquer) strategy runners.
//!
//! A [`StrategyRunner`] drives the strategy cascade for a single D0 engine
//! invocation. [`SequentialRunner`] tries each applicable strategy in
//! priority order and stops at the first success or certified-impossible
//! result.

use std::sync::Arc;

use crate::engine::context::SolveContext;
use crate::engine::mode::ExecutionMode;
use crate::engine::reason::FailureReason;
use crate::engine::strategy::{Strategy, StrategyResult};
use crate::engine::trace_tree::{BranchHandle, BranchReason, JoinHandle, JoinReason, TraceTree};
use crate::numeric::Expr;

// ── StrategyRunner trait ───────────────────────────────────────────────────────

/// Drives the strategy cascade for a D0 engine run.
///
/// `StrategyRunner` is `Send + Sync` so runners can be stored in registries
/// and shared across threads.
pub trait StrategyRunner: Send + Sync {
    /// Run `strategies` over `ctx` in the given [`ExecutionMode`].
    ///
    /// Returns the final [`StrategyResult`] once the cascade completes.
    fn run(
        &self,
        ctx: SolveContext,
        strategies: &[Box<dyn Strategy>],
        mode: ExecutionMode,
    ) -> StrategyResult;
}

// ── SequentialRunner ──────────────────────────────────────────────────────────

/// Tries each applicable strategy in priority order (lower = first) and
/// returns the first success, certified-impossible, or structural-error result.
///
/// If all strategies are exhausted without a solution, returns the best partial
/// result observed, or [`StrategyResult::Failed`]`(`[`FailureReason::NotApplicable`]`)`
/// when no partial progress was made.
#[derive(Debug, Default)]
pub struct SequentialRunner;

impl StrategyRunner for SequentialRunner {
    fn run(
        &self,
        ctx: SolveContext,
        strategies: &[Box<dyn Strategy>],
        mode: ExecutionMode,
    ) -> StrategyResult {
        match mode {
            ExecutionMode::Sequential => run_sequential(ctx, strategies),
            ExecutionMode::TreeSearch { max_depth, .. } => {
                run_tree_search(ctx, strategies, max_depth)
            }
            ExecutionMode::DivideAndConquer { max_depth } => {
                run_divide_and_conquer(ctx, strategies, max_depth)
            }
        }
    }
}

// ── Sequential cascade ────────────────────────────────────────────────────────

fn run_sequential(ctx: SolveContext, strategies: &[Box<dyn Strategy>]) -> StrategyResult {
    // Filter to applicable strategies.
    let mut candidates: Vec<(usize, f64)> = strategies
        .iter()
        .enumerate()
        .filter(|(_, s)| s.applicable(&ctx))
        .map(|(i, s)| (i, s.priority(&ctx)))
        .collect();

    // Sort ascending by priority (lower value = tried first).
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut best_partial: Option<StrategyResult> = None;

    // Open a branch node in the context trace for this cascade.
    let candidate_ids: Vec<_> = candidates
        .iter()
        .map(|(i, _)| strategies[*i].id())
        .collect();
    let branch_handle: BranchHandle = {
        let mut dummy = TraceTree::new();
        dummy.open_branch(BranchReason::StrategyCascade, candidate_ids)
    };
    let _ = branch_handle; // handle is per-context; trace bookkeeping is advisory here

    for (idx, _priority) in candidates {
        let strategy = &strategies[idx];
        let fork = ctx.fork();
        let result = strategy.apply(fork);

        match result {
            // Immediate success — return.
            StrategyResult::Solved { .. } => return result,

            // Certified impossible — return immediately (do not try fallbacks).
            StrategyResult::ProvenImpossible { .. } => return result,

            // Structural error is a hard stop — further attempts are futile.
            StrategyResult::Failed(FailureReason::StructuralError(_)) => return result,

            // Partial progress — save as best if better than what we have.
            StrategyResult::Partial { .. } => {
                if best_partial.is_none() {
                    best_partial = Some(result);
                }
            }

            // Branch: recurse into the given candidates.
            StrategyResult::Branch(branch_candidates) => {
                // Wrap each candidate as a boxed strategy and recurse.
                let boxed: Vec<Box<dyn Strategy>> =
                    branch_candidates.into_iter().map(|c| c.strategy).collect();
                let sub = run_sequential(ctx.fork(), &boxed);
                match sub {
                    StrategyResult::Solved { .. } | StrategyResult::ProvenImpossible { .. } => {
                        return sub
                    }
                    StrategyResult::Failed(FailureReason::StructuralError(_)) => return sub,
                    StrategyResult::Partial { .. } => {
                        if best_partial.is_none() {
                            best_partial = Some(sub);
                        }
                    }
                    _ => {}
                }
            }

            // Decompose: solve each sub-problem, then merge.
            StrategyResult::Decompose { parts, merger } => {
                let sub_results: Vec<Arc<Expr>> = parts
                    .into_iter()
                    .map(|sp| {
                        let sub = run_sequential(sp.context, strategies);
                        match sub {
                            StrategyResult::Solved { expr, .. } => Some(expr),
                            _ => None,
                        }
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .collect::<Option<Vec<_>>>()
                    .unwrap_or_default();

                if sub_results.is_empty() {
                    // Not all sub-problems solved — try next strategy.
                    continue;
                }
                let merged = merger(sub_results);
                match merged {
                    StrategyResult::Solved { .. } => return merged,
                    StrategyResult::ProvenImpossible { .. } => return merged,
                    StrategyResult::Partial { .. } => {
                        if best_partial.is_none() {
                            best_partial = Some(merged);
                        }
                    }
                    _ => {}
                }
            }

            // Not applicable or resource-limited — try next.
            StrategyResult::Failed(_) | StrategyResult::NeedsResource(_) => {}
        }
    }

    // All strategies exhausted.
    best_partial.unwrap_or(StrategyResult::Failed(FailureReason::NotApplicable))
}

// ── TreeSearch (basic) ────────────────────────────────────────────────────────

fn run_tree_search(
    ctx: SolveContext,
    strategies: &[Box<dyn Strategy>],
    max_depth: u32,
) -> StrategyResult {
    if max_depth == 0 {
        return StrategyResult::Failed(FailureReason::NoClosedForm);
    }
    // For depth > 0, collect all applicable strategies, try all branches,
    // and return the first success found (FirstSuccess comparison policy).
    let mut candidates: Vec<(usize, f64)> = strategies
        .iter()
        .enumerate()
        .filter(|(_, s)| s.applicable(&ctx))
        .map(|(i, s)| (i, s.priority(&ctx)))
        .collect();
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut best_partial: Option<StrategyResult> = None;
    for (idx, _) in candidates {
        let strategy = &strategies[idx];
        let fork = ctx.fork();
        let result = strategy.apply(fork);
        match result {
            StrategyResult::Solved { .. } => return result,
            StrategyResult::ProvenImpossible { .. } => return result,
            StrategyResult::Failed(FailureReason::StructuralError(_)) => return result,
            StrategyResult::Partial { .. } => {
                if best_partial.is_none() {
                    best_partial = Some(result);
                }
            }
            StrategyResult::Branch(branch_candidates) => {
                let boxed: Vec<Box<dyn Strategy>> =
                    branch_candidates.into_iter().map(|c| c.strategy).collect();
                let sub = run_tree_search(ctx.fork(), &boxed, max_depth - 1);
                match sub {
                    StrategyResult::Solved { .. } | StrategyResult::ProvenImpossible { .. } => {
                        return sub;
                    }
                    StrategyResult::Partial { .. } => {
                        if best_partial.is_none() {
                            best_partial = Some(sub);
                        }
                    }
                    _ => {}
                }
            }
            StrategyResult::Decompose { parts, merger } => {
                let sub_results: Vec<Arc<Expr>> = parts
                    .into_iter()
                    .map(|sp| {
                        let sub = run_tree_search(sp.context, strategies, max_depth - 1);
                        match sub {
                            StrategyResult::Solved { expr, .. } => Some(expr),
                            _ => None,
                        }
                    })
                    .collect::<Option<Vec<_>>>()
                    .unwrap_or_default();
                if sub_results.is_empty() {
                    continue;
                }
                let merged = merger(sub_results);
                match merged {
                    StrategyResult::Solved { .. } => return merged,
                    _ => {}
                }
            }
            StrategyResult::Failed(_) | StrategyResult::NeedsResource(_) => {}
        }
    }
    best_partial.unwrap_or(StrategyResult::Failed(FailureReason::NotApplicable))
}

// ── DivideAndConquer (basic) ──────────────────────────────────────────────────

fn run_divide_and_conquer(
    ctx: SolveContext,
    strategies: &[Box<dyn Strategy>],
    max_depth: u32,
) -> StrategyResult {
    // Find the first Decompose strategy and recurse into sub-problems.
    let mut candidates: Vec<(usize, f64)> = strategies
        .iter()
        .enumerate()
        .filter(|(_, s)| s.applicable(&ctx))
        .map(|(i, s)| (i, s.priority(&ctx)))
        .collect();
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let sub_mode = if max_depth > 0 {
        ExecutionMode::DivideAndConquer {
            max_depth: max_depth - 1,
        }
    } else {
        ExecutionMode::Sequential
    };
    let _ = sub_mode; // reserved for recursive depth control

    for (idx, _) in candidates {
        let strategy = &strategies[idx];
        let fork = ctx.fork();
        let result = strategy.apply(fork);
        match result {
            StrategyResult::Solved { .. } => return result,
            StrategyResult::ProvenImpossible { .. } => return result,
            StrategyResult::Failed(FailureReason::StructuralError(_)) => return result,
            StrategyResult::Decompose { parts, merger } => {
                if max_depth == 0 {
                    continue;
                }
                let sub_results: Vec<Arc<Expr>> = parts
                    .into_iter()
                    .map(|sp| {
                        let sub = run_divide_and_conquer(sp.context, strategies, max_depth - 1);
                        match sub {
                            StrategyResult::Solved { expr, .. } => Some(expr),
                            _ => None,
                        }
                    })
                    .collect::<Option<Vec<_>>>()
                    .unwrap_or_default();
                if sub_results.is_empty() {
                    continue;
                }
                let merged = merger(sub_results);
                match merged {
                    StrategyResult::Solved { .. } => return merged,
                    _ => {}
                }
            }
            _ => {}
        }
    }
    StrategyResult::Failed(FailureReason::NotApplicable)
}

// ── Joint helpers (unused but required for trace completeness) ────────────────

/// Open a [`TraceTree`] join node and immediately close it.
/// Used internally to record decomposition joins for diagnostic tracing.
#[allow(dead_code)]
fn open_join(tree: &mut TraceTree) -> JoinHandle {
    tree.open_join(JoinReason::DivideAndConquer, 0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
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
        // Two solve strategies with different priorities and values.
        // Lower priority (0.5) should win.
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
                // The lower-priority (0.5) strategy returns value 7.
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
        // ImpossibleStrategy has lower priority (0.0) than SolveStrategy (1.0).
        // cascade must stop at ProvenImpossible, never reach SolveStrategy.
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
        // NotApplicableStrategy returns applicable=false — must be skipped.
        // SolveStrategy should still be tried and win.
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
}
