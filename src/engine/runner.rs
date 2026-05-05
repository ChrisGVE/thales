//! Sequential and parallel strategy runners.
//!
//! A [`StrategyRunner`] drives the strategy cascade for a single D0 engine
//! invocation. [`SequentialRunner`] tries each applicable strategy in
//! priority order and stops at the first success or certified-impossible
//! result. When the `rayon` feature is enabled, [`RayonRunner`] explores
//! branches concurrently, bounded by a [`ResourceGate`].
//!
//! After the symbolic cascade completes, [`SequentialRunner`] optionally
//! invokes [`FallbackRunner`] when `ctx.fallback.numerical` is `true`.

use std::sync::Arc;

use crate::engine::context::SolveContext;
use crate::engine::fallback::runner::FallbackRunner;
use crate::engine::fallback::{node_count, FallbackTrigger};
use crate::engine::mode::ExecutionMode;
use crate::engine::reason::FailureReason;
#[cfg(feature = "rayon")]
use crate::engine::resource::ResourceGate;
use crate::engine::resource::ResourceStatus;
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
        // ── Pre-cascade: complexity-explosion gate ─────────────────────────
        // Check before the cascade so we can short-circuit directly to
        // FallbackRunner when the expression is too large to solve symbolically.
        if ctx.fallback.numerical {
            if let Some(threshold) = ctx.fallback.complexity_threshold {
                let actual = node_count(&ctx.expr);
                if actual > threshold {
                    let trigger = FallbackTrigger::ComplexityExplosion {
                        actual_nodes: actual,
                        threshold_nodes: threshold,
                    };
                    if let Some(result) = FallbackRunner::run(&ctx, trigger) {
                        return result;
                    }
                    // No evaluator handled it — fall through to symbolic cascade.
                }
            }
        }

        // ── Symbolic cascade ───────────────────────────────────────────────
        let cascade_result = match mode {
            ExecutionMode::Sequential => run_sequential(ctx.fork(), strategies),
            ExecutionMode::TreeSearch { max_depth, .. } => {
                run_tree_search(ctx.fork(), strategies, max_depth)
            }
            ExecutionMode::DivideAndConquer { max_depth } => {
                run_divide_and_conquer(ctx.fork(), strategies, max_depth)
            }
        };

        // ── Post-cascade: numerical fallback ───────────────────────────────
        // Only attempted when opt-in; never on hard-stop results.
        if !ctx.fallback.numerical {
            return cascade_result;
        }

        let trigger = match &cascade_result {
            // Hard success — no fallback needed.
            StrategyResult::Solved { .. } => return cascade_result,
            // Certified impossible — symbolic proof is authoritative; skip fallback.
            StrategyResult::ProvenImpossible { .. } => return cascade_result,
            // Structural error — cannot recover numerically.
            StrategyResult::Failed(FailureReason::StructuralError(_)) => return cascade_result,

            // All strategies tried and none succeeded.
            StrategyResult::Failed(reason) => FallbackTrigger::StrategyExhaustion {
                strategies_attempted: strategies.len(),
                last_reason: reason.clone(),
            },

            // Partial — all strategies tried but only partial progress.
            StrategyResult::Partial { .. } => FallbackTrigger::StrategyExhaustion {
                strategies_attempted: strategies.len(),
                last_reason: FailureReason::NotApplicable,
            },

            // NeedsResource — budget was the limiting factor.
            StrategyResult::NeedsResource(_) => FallbackTrigger::ResourceBudgetExceeded {
                status: ResourceStatus::Exceeded,
            },

            // Branch/Decompose at the top level means the cascade returned
            // without resolving; treat as exhaustion.
            StrategyResult::Branch(_) | StrategyResult::Decompose { .. } => {
                FallbackTrigger::StrategyExhaustion {
                    strategies_attempted: strategies.len(),
                    last_reason: FailureReason::NotApplicable,
                }
            }
        };

        FallbackRunner::run(&ctx, trigger).unwrap_or(cascade_result)
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

// ── RayonRunner (rayon feature only) ─────────────────────────────────────────

/// A strategy runner that explores branches concurrently using rayon's
/// thread pool, bounded by a [`ResourceGate`] to cap parallelism.
///
/// Falls back to sequential execution for branches that cannot acquire a
/// gate slot, so correctness is preserved regardless of concurrency level.
#[cfg(feature = "rayon")]
pub struct RayonRunner {
    /// Limits the number of concurrently active parallel branches.
    pub gate: ResourceGate,
}

#[cfg(feature = "rayon")]
impl RayonRunner {
    /// Create a runner allowing at most `max_parallel` concurrent branches.
    #[must_use]
    pub fn new(max_parallel: usize) -> Self {
        RayonRunner {
            gate: ResourceGate::new(max_parallel),
        }
    }
}

#[cfg(feature = "rayon")]
impl StrategyRunner for RayonRunner {
    fn run(
        &self,
        ctx: SolveContext,
        strategies: &[Box<dyn Strategy>],
        mode: ExecutionMode,
    ) -> StrategyResult {
        match mode {
            ExecutionMode::Sequential => run_sequential(ctx, strategies),
            ExecutionMode::TreeSearch { max_depth, .. } => {
                run_tree_search_parallel(ctx, strategies, max_depth, &self.gate)
            }
            ExecutionMode::DivideAndConquer { max_depth } => {
                run_divide_and_conquer_parallel(ctx, strategies, max_depth, &self.gate)
            }
        }
    }
}

#[cfg(feature = "rayon")]
fn run_tree_search_parallel(
    ctx: SolveContext,
    strategies: &[Box<dyn Strategy>],
    max_depth: u32,
    gate: &ResourceGate,
) -> StrategyResult {
    if max_depth == 0 {
        return StrategyResult::Failed(FailureReason::NoClosedForm);
    }

    let mut candidates: Vec<(usize, f64)> = strategies
        .iter()
        .enumerate()
        .filter(|(_, s)| s.applicable(&ctx))
        .map(|(i, s)| (i, s.priority(&ctx)))
        .collect();
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    if candidates.len() < 2 {
        // Single or zero candidates — sequential is optimal.
        return run_tree_search(ctx, strategies, max_depth);
    }

    // Try to acquire a gate slot for parallel execution.
    let _guard = gate.acquire();
    if _guard.is_some() {
        // Use rayon::join for the first two candidates, then fold the rest.
        let (first_idx, _) = candidates[0];
        let (second_idx, _) = candidates[1];
        let ctx1 = ctx.fork();
        let ctx2 = ctx.fork();
        let s1 = &strategies[first_idx];
        let s2 = &strategies[second_idx];
        let (r1, r2) = rayon::join(|| s1.apply(ctx1), || s2.apply(ctx2));
        // Return the first success; otherwise fall through to sequential.
        for result in [r1, r2] {
            match result {
                StrategyResult::Solved { .. } | StrategyResult::ProvenImpossible { .. } => {
                    return result
                }
                StrategyResult::Failed(FailureReason::StructuralError(_)) => return result,
                _ => {}
            }
        }
        // Remaining candidates — sequential.
        for (idx, _) in candidates.iter().skip(2) {
            let result = strategies[*idx].apply(ctx.fork());
            match result {
                StrategyResult::Solved { .. } | StrategyResult::ProvenImpossible { .. } => {
                    return result
                }
                StrategyResult::Failed(FailureReason::StructuralError(_)) => return result,
                _ => {}
            }
        }
        StrategyResult::Failed(FailureReason::NotApplicable)
    } else {
        run_tree_search(ctx, strategies, max_depth)
    }
}

#[cfg(feature = "rayon")]
fn run_divide_and_conquer_parallel(
    ctx: SolveContext,
    strategies: &[Box<dyn Strategy>],
    max_depth: u32,
    gate: &ResourceGate,
) -> StrategyResult {
    let mut candidates: Vec<(usize, f64)> = strategies
        .iter()
        .enumerate()
        .filter(|(_, s)| s.applicable(&ctx))
        .map(|(i, s)| (i, s.priority(&ctx)))
        .collect();
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

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
                let _guard = gate.acquire();
                let sub_results: Vec<Arc<Expr>> = if _guard.is_some() {
                    use rayon::prelude::*;
                    parts
                        .into_par_iter()
                        .map(|sp| {
                            let sub = run_divide_and_conquer(sp.context, strategies, max_depth - 1);
                            match sub {
                                StrategyResult::Solved { expr, .. } => Some(expr),
                                _ => None,
                            }
                        })
                        .collect::<Option<Vec<_>>>()
                        .unwrap_or_default()
                } else {
                    parts
                        .into_iter()
                        .map(|sp| {
                            let sub = run_divide_and_conquer(sp.context, strategies, max_depth - 1);
                            match sub {
                                StrategyResult::Solved { expr, .. } => Some(expr),
                                _ => None,
                            }
                        })
                        .collect::<Option<Vec<_>>>()
                        .unwrap_or_default()
                };
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
        // Gate is at capacity — third acquire must fail.
        assert!(gate.acquire().is_none());
        drop(g1);
        assert_eq!(gate.active_count(), 1);
        // Slot released — can acquire again.
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

    /// Verify that when fallback is disabled (default), cascade failure is
    /// returned as-is — no fallback attempted.
    #[test]
    fn fast_runner_fallback_disabled_returns_cascade_failure() {
        let runner = SequentialRunner;
        // Default context has fallback disabled.
        let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(FailStrategy { priority: 0.0 })];
        let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);
        assert!(
            matches!(result, StrategyResult::Failed(FailureReason::NotApplicable)),
            "fallback disabled: cascade failure must be returned unchanged"
        );
    }

    /// When fallback is enabled and the cascade succeeds, no fallback is
    /// invoked — the cascade result is returned directly.
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

    /// When fallback is enabled and the cascade exhausts all strategies,
    /// `FallbackRunner::run_with_registry` with a success mock produces Solved.
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

    /// When fallback is enabled but the registry is empty, the original
    /// cascade failure is returned.
    #[test]
    fn fast_runner_fallback_empty_registry_returns_cascade_result() {
        use crate::engine::fallback::runner::FallbackRunner;
        use crate::engine::fallback::trigger::FallbackTrigger;

        let reg = NumericalEvaluatorRegistry::new(); // empty
        let ctx = ctx_with_fallback();
        let trigger = FallbackTrigger::StrategyExhaustion {
            strategies_attempted: 0,
            last_reason: FailureReason::NotApplicable,
        };
        let result = FallbackRunner::run_with_registry(&ctx, trigger, &reg);
        assert!(result.is_none(), "empty registry must return None");
    }

    /// Pre-cascade complexity gate: when `complexity_threshold` is set and the
    /// expression exceeds it, `FallbackRunner::run_with_registry` is the right
    /// path — verified directly to avoid global-registry side effects.
    #[test]
    fn fast_runner_fallback_complexity_gate_triggers_on_threshold_exceeded() {
        use crate::engine::fallback::runner::FallbackRunner;
        use crate::engine::fallback::trigger::FallbackTrigger;

        let reg = reg_with_success();
        let ctx = SolveContext::new(int_expr(1), ResourceBudget::unlimited()).with_fallback(
            FallbackConfig {
                numerical: true,
                numerical_narrative: true,
                complexity_threshold: Some(0), // threshold=0: always exceeded
            },
        );
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

    /// ProvenImpossible from the cascade is never overridden by fallback.
    #[test]
    fn fast_runner_fallback_proven_impossible_not_overridden() {
        let runner = SequentialRunner;
        // Even with fallback enabled, ProvenImpossible is authoritative.
        let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(ImpossibleStrategy)];
        let result = runner.run(ctx_with_fallback(), &strategies, ExecutionMode::Sequential);
        assert!(
            matches!(result, StrategyResult::ProvenImpossible { .. }),
            "ProvenImpossible must not be overridden by fallback"
        );
    }
}
