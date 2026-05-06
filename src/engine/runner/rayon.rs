//! Parallel strategy runner using rayon.

use std::sync::Arc;

use crate::engine::context::SolveContext;
use crate::engine::mode::ExecutionMode;
use crate::engine::reason::FailureReason;
use crate::engine::resource::ResourceGate;
use crate::engine::strategy::{Strategy, StrategyResult};
use crate::numeric::Expr;

use super::{
    is_hard_stop, run_divide_and_conquer, run_sequential, run_tree_search, sorted_candidates,
    StrategyRunner,
};

/// A strategy runner that explores branches concurrently using rayon's
/// thread pool, bounded by a [`ResourceGate`] to cap parallelism.
///
/// Falls back to sequential execution for branches that cannot acquire a
/// gate slot, so correctness is preserved regardless of concurrency level.
pub struct RayonRunner {
    /// Limits the number of concurrently active parallel branches.
    pub gate: ResourceGate,
}

impl RayonRunner {
    #[must_use]
    pub fn new(max_parallel: usize) -> Self {
        RayonRunner {
            gate: ResourceGate::new(max_parallel),
        }
    }
}

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

fn run_tree_search_parallel(
    ctx: SolveContext,
    strategies: &[Box<dyn Strategy>],
    max_depth: u32,
    gate: &ResourceGate,
) -> StrategyResult {
    if max_depth == 0 {
        return StrategyResult::Failed(FailureReason::NoClosedForm);
    }

    let candidates = sorted_candidates(&ctx, strategies);

    if candidates.len() < 2 {
        return run_tree_search(ctx, strategies, max_depth);
    }

    let _guard = gate.acquire();
    if _guard.is_some() {
        let (first_idx, _) = candidates[0];
        let (second_idx, _) = candidates[1];
        let ctx1 = ctx.fork();
        let ctx2 = ctx.fork();
        let s1 = &strategies[first_idx];
        let s2 = &strategies[second_idx];
        let (r1, r2) = rayon::join(|| s1.apply(ctx1), || s2.apply(ctx2));
        for result in [r1, r2] {
            if is_hard_stop(&result) {
                return result;
            }
        }
        for (idx, _) in candidates.iter().skip(2) {
            let result = strategies[*idx].apply(ctx.fork());
            if is_hard_stop(&result) {
                return result;
            }
        }
        StrategyResult::Failed(FailureReason::NotApplicable)
    } else {
        run_tree_search(ctx, strategies, max_depth)
    }
}

fn run_divide_and_conquer_parallel(
    ctx: SolveContext,
    strategies: &[Box<dyn Strategy>],
    max_depth: u32,
    gate: &ResourceGate,
) -> StrategyResult {
    let candidates = sorted_candidates(&ctx, strategies);

    for (idx, _) in candidates {
        let strategy = &strategies[idx];
        let fork = ctx.fork();
        let result = strategy.apply(fork);

        if is_hard_stop(&result) {
            return result;
        }

        if let StrategyResult::Decompose { parts, merger } = result {
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
            if let StrategyResult::Solved { .. } = merged {
                return merged;
            }
        }
    }
    StrategyResult::Failed(FailureReason::NotApplicable)
}
