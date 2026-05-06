//! Sequential and parallel strategy runners.
//!
//! A [`StrategyRunner`] drives the strategy cascade for a single D0 engine
//! invocation. [`SequentialRunner`] tries each applicable strategy in
//! priority order and stops at the first success or certified-impossible
//! result. When the `rayon` feature is enabled, [`RayonRunner`] explores
//! branches concurrently, bounded by a [`ResourceGate`].

#[cfg(feature = "rayon")]
mod rayon;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use crate::engine::context::SolveContext;
use crate::engine::fallback::runner::FallbackRunner;
use crate::engine::fallback::{node_count, FallbackTrigger};
use crate::engine::mode::ExecutionMode;
use crate::engine::reason::FailureReason;
use crate::engine::resource::ResourceStatus;
use crate::engine::strategy::{Strategy, StrategyResult};
use crate::engine::trace_tree::{BranchHandle, BranchReason, JoinHandle, JoinReason, TraceTree};
use crate::numeric::Expr;

#[cfg(feature = "rayon")]
pub use self::rayon::RayonRunner;

// ── StrategyRunner trait ───────────────────────────────────────────────────────

/// Drives the strategy cascade for a D0 engine run.
pub trait StrategyRunner: Send + Sync {
    fn run(
        &self,
        ctx: SolveContext,
        strategies: &[Box<dyn Strategy>],
        mode: ExecutionMode,
    ) -> StrategyResult;
}

// ── SequentialRunner ──────────────────────────────────────────────────────────

/// Tries each applicable strategy in priority order and returns the first
/// success, certified-impossible, or structural-error result.
#[derive(Debug, Default)]
pub struct SequentialRunner;

impl StrategyRunner for SequentialRunner {
    fn run(
        &self,
        ctx: SolveContext,
        strategies: &[Box<dyn Strategy>],
        mode: ExecutionMode,
    ) -> StrategyResult {
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
                }
            }
        }

        let cascade_result = match mode {
            ExecutionMode::Sequential => run_sequential(ctx.fork(), strategies),
            ExecutionMode::TreeSearch { max_depth, .. } => {
                run_tree_search(ctx.fork(), strategies, max_depth)
            }
            ExecutionMode::DivideAndConquer { max_depth } => {
                run_divide_and_conquer(ctx.fork(), strategies, max_depth)
            }
        };

        if !ctx.fallback.numerical {
            return cascade_result;
        }

        let trigger = match &cascade_result {
            StrategyResult::Solved { .. } => return cascade_result,
            StrategyResult::ProvenImpossible { .. } => return cascade_result,
            StrategyResult::Failed(FailureReason::StructuralError(_)) => return cascade_result,

            StrategyResult::Failed(reason) => FallbackTrigger::StrategyExhaustion {
                strategies_attempted: strategies.len(),
                last_reason: reason.clone(),
            },

            StrategyResult::Partial { .. } => FallbackTrigger::StrategyExhaustion {
                strategies_attempted: strategies.len(),
                last_reason: FailureReason::NotApplicable,
            },

            StrategyResult::NeedsResource(_) => FallbackTrigger::ResourceBudgetExceeded {
                status: ResourceStatus::Exceeded,
            },

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

// ── Candidate collection ─────────────────────────────────────────────────────

pub(super) fn sorted_candidates(
    ctx: &SolveContext,
    strategies: &[Box<dyn Strategy>],
) -> Vec<(usize, f64)> {
    let mut candidates: Vec<(usize, f64)> = strategies
        .iter()
        .enumerate()
        .filter(|(_, s)| s.applicable(ctx))
        .map(|(i, s)| (i, s.priority(ctx)))
        .collect();
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates
}

/// Returns `true` for results that terminate the cascade immediately.
pub(super) fn is_hard_stop(result: &StrategyResult) -> bool {
    matches!(
        result,
        StrategyResult::Solved { .. }
            | StrategyResult::ProvenImpossible { .. }
            | StrategyResult::Failed(FailureReason::StructuralError(_))
    )
}

// ── Sequential cascade ────────────────────────────────────────────────────────

pub(crate) fn run_sequential(
    ctx: SolveContext,
    strategies: &[Box<dyn Strategy>],
) -> StrategyResult {
    let candidates = sorted_candidates(&ctx, strategies);

    let mut best_partial: Option<StrategyResult> = None;

    let candidate_ids: Vec<_> = candidates
        .iter()
        .map(|(i, _)| strategies[*i].id())
        .collect();
    let branch_handle: BranchHandle = {
        let mut dummy = TraceTree::new();
        dummy.open_branch(BranchReason::StrategyCascade, candidate_ids)
    };
    let _ = branch_handle;

    for (idx, _priority) in candidates {
        let strategy = &strategies[idx];
        let result = strategy.apply(ctx.fork());

        if is_hard_stop(&result) {
            return result;
        }

        match result {
            StrategyResult::Partial { .. } => {
                if best_partial.is_none() {
                    best_partial = Some(result);
                }
            }
            StrategyResult::Branch(branch_candidates) => {
                let sub = recurse_branch(branch_candidates, &ctx, strategies, |c, s| {
                    run_sequential(c, s)
                });
                if is_hard_stop(&sub) {
                    return sub;
                }
                if best_partial.is_none() {
                    if let StrategyResult::Partial { .. } = &sub {
                        best_partial = Some(sub);
                    }
                }
            }
            StrategyResult::Decompose { parts, merger } => {
                let merged =
                    run_decompose(parts, merger, &ctx, strategies, |c, s| run_sequential(c, s));
                if let Some(m) = merged {
                    if is_hard_stop(&m) {
                        return m;
                    }
                    if best_partial.is_none() {
                        if let StrategyResult::Partial { .. } = &m {
                            best_partial = Some(m);
                        }
                    }
                }
            }
            StrategyResult::Failed(_) | StrategyResult::NeedsResource(_) => {}
            _ => {}
        }
    }

    best_partial.unwrap_or(StrategyResult::Failed(FailureReason::NotApplicable))
}

/// Recurse into Branch candidates.
fn recurse_branch(
    branch_candidates: Vec<crate::engine::strategy::StrategyCandidate>,
    ctx: &SolveContext,
    _strategies: &[Box<dyn Strategy>],
    recurse: impl Fn(SolveContext, &[Box<dyn Strategy>]) -> StrategyResult,
) -> StrategyResult {
    let boxed: Vec<Box<dyn Strategy>> = branch_candidates.into_iter().map(|c| c.strategy).collect();
    recurse(ctx.fork(), &boxed)
}

/// Solve each sub-problem, then merge results.
fn run_decompose(
    parts: Vec<crate::engine::strategy::SubProblem>,
    merger: crate::engine::strategy::MergerFn,
    _ctx: &SolveContext,
    strategies: &[Box<dyn Strategy>],
    recurse: impl Fn(SolveContext, &[Box<dyn Strategy>]) -> StrategyResult,
) -> Option<StrategyResult> {
    let sub_results: Vec<Arc<Expr>> = parts
        .into_iter()
        .map(|sp| {
            let sub = recurse(sp.context, strategies);
            match sub {
                StrategyResult::Solved { expr, .. } => Some(expr),
                _ => None,
            }
        })
        .collect::<Option<Vec<_>>>()?;
    Some(merger(sub_results))
}

// ── TreeSearch ───────────────────────────────────────────────────────────────

pub(crate) fn run_tree_search(
    ctx: SolveContext,
    strategies: &[Box<dyn Strategy>],
    max_depth: u32,
) -> StrategyResult {
    if max_depth == 0 {
        return StrategyResult::Failed(FailureReason::NoClosedForm);
    }
    let candidates = sorted_candidates(&ctx, strategies);
    let mut best_partial: Option<StrategyResult> = None;

    for (idx, _) in candidates {
        let strategy = &strategies[idx];
        let result = strategy.apply(ctx.fork());

        if is_hard_stop(&result) {
            return result;
        }

        match result {
            StrategyResult::Partial { .. } => {
                if best_partial.is_none() {
                    best_partial = Some(result);
                }
            }
            StrategyResult::Branch(branch_candidates) => {
                let sub = recurse_branch(branch_candidates, &ctx, strategies, |c, s| {
                    run_tree_search(c, s, max_depth - 1)
                });
                if is_hard_stop(&sub) {
                    return sub;
                }
                if best_partial.is_none() {
                    if let StrategyResult::Partial { .. } = &sub {
                        best_partial = Some(sub);
                    }
                }
            }
            StrategyResult::Decompose { parts, merger } => {
                let merged = run_decompose(parts, merger, &ctx, strategies, |c, s| {
                    run_tree_search(c, s, max_depth - 1)
                });
                if let Some(m) = merged {
                    if is_hard_stop(&m) {
                        return m;
                    }
                }
            }
            StrategyResult::Failed(_) | StrategyResult::NeedsResource(_) => {}
            _ => {}
        }
    }
    best_partial.unwrap_or(StrategyResult::Failed(FailureReason::NotApplicable))
}

// ── DivideAndConquer ─────────────────────────────────────────────────────────

pub(crate) fn run_divide_and_conquer(
    ctx: SolveContext,
    strategies: &[Box<dyn Strategy>],
    max_depth: u32,
) -> StrategyResult {
    let candidates = sorted_candidates(&ctx, strategies);

    for (idx, _) in candidates {
        let strategy = &strategies[idx];
        let result = strategy.apply(ctx.fork());

        if is_hard_stop(&result) {
            return result;
        }

        if let StrategyResult::Decompose { parts, merger } = result {
            if max_depth == 0 {
                continue;
            }
            let merged = run_decompose(parts, merger, &ctx, strategies, |c, s| {
                run_divide_and_conquer(c, s, max_depth - 1)
            });
            if let Some(m) = merged {
                if is_hard_stop(&m) {
                    return m;
                }
            }
        }
    }
    StrategyResult::Failed(FailureReason::NotApplicable)
}

// ── Joint helpers ────────────────────────────────────────────────────────────

#[allow(dead_code)]
fn open_join(tree: &mut TraceTree) -> JoinHandle {
    tree.open_join(JoinReason::DivideAndConquer, 0)
}
