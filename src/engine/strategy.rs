//! Strategy trait and result types for the D0 search strategy engine.
//!
//! A [`Strategy`] receives a [`SolveContext`] by value and returns a
//! [`StrategyResult`] describing what it produced. Results are branching
//! (multiple candidates to explore), decomposing (divide-and-conquer), or
//! terminal (solved, failed, partial, resource-limited, proven impossible).
//!
//! Strategies are `Send + Sync + 'static` so they can be stored in
//! registries and dispatched across threads.

use std::fmt;
use std::sync::Arc;

use crate::engine::context::SolveContext;
use crate::engine::reason::{FailureReason, ImpossibilityProof, PartialReason, ResourceRequest};
use crate::engine::trace_tree::{StrategyId, TraceNode};
use crate::numeric::Expr;

// ── Strategy ──────────────────────────────────────────────────────────────────

/// A single named strategy in the D0 search space.
///
/// Implementors decide whether they apply to the given context, how
/// urgently they should be tried (priority), and what result they
/// produce when applied.
pub trait Strategy: Send + Sync + fmt::Debug + 'static {
    /// Stable identifier for this strategy.
    fn id(&self) -> StrategyId;

    /// Returns `true` if this strategy can be applied to `ctx`.
    ///
    /// A quick structural pre-check; should not do heavy work.
    fn applicable(&self, ctx: &SolveContext) -> bool;

    /// Priority score for this strategy in the given context.
    ///
    /// Higher values are tried first. The engine uses this to order
    /// candidates before attempting them.
    fn priority(&self, ctx: &SolveContext) -> f64;

    /// Apply this strategy to `ctx` (consumed by value) and return the result.
    fn apply(&self, ctx: SolveContext) -> StrategyResult;
}

// ── MergerFn ──────────────────────────────────────────────────────────────────

/// A shared closure that merges solved sub-problem results.
///
/// Produced by `StrategyResult::Decompose`; called by the engine once all
/// `SubProblem`s are solved. The `Vec<Arc<Expr>>` contains one entry per
/// sub-problem in the same order as `parts`.
pub type MergerFn = Arc<dyn Fn(Vec<Arc<Expr>>) -> StrategyResult + Send + Sync>;

// ── SubProblem ────────────────────────────────────────────────────────────────

/// An independent sub-problem in a divide-and-conquer decomposition.
///
/// `context` carries the sub-expression (as `context.expr`) plus its own
/// fresh trace, budget, assumptions, and properties. `label` is used for
/// narrative purposes.
pub struct SubProblem {
    /// Execution context for this sub-problem, including the sub-expression.
    pub context: SolveContext,
    /// Human-readable label for narrative and diagnostics.
    pub label: String,
}

impl fmt::Debug for SubProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubProblem")
            .field("label", &self.label)
            .field("context", &self.context)
            .finish()
    }
}

// ── StrategyCandidate ─────────────────────────────────────────────────────────

/// A strategy together with its computed priority, ready to be tried.
#[derive(Debug)]
pub struct StrategyCandidate {
    /// The strategy to attempt.
    pub strategy: Box<dyn Strategy>,
    /// Priority score at the moment this candidate was created.
    pub priority: f64,
}

// ── StrategyResult ────────────────────────────────────────────────────────────

/// The outcome of applying a [`Strategy`] to a [`SolveContext`].
///
/// Marked `#[must_use]` because discarding a result silently loses
/// trace information and may incorrectly propagate failures.
#[must_use]
pub enum StrategyResult {
    /// Strategy succeeded and produced a complete result.
    Solved {
        /// The solved expression.
        expr: Arc<Expr>,
        /// The trace node recording how the solution was reached.
        trace: TraceNode,
    },

    /// Strategy cannot decide alone; offers multiple candidates for the
    /// engine to try in priority order.
    Branch(Vec<StrategyCandidate>),

    /// Strategy decomposes the problem into independent sub-problems that
    /// can be solved separately, then merged by `merger`.
    Decompose {
        /// The independent sub-problems.
        parts: Vec<SubProblem>,
        /// Closure that merges solved sub-expressions into a final result.
        merger: MergerFn,
    },

    /// Strategy was not applicable or could not make progress.
    Failed(FailureReason),

    /// Strategy made partial progress but could not finish.
    Partial {
        /// Best partial result reached.
        expr: Arc<Expr>,
        /// Why the result is incomplete.
        reason: PartialReason,
        /// Trace up to the point of partial completion.
        trace: TraceNode,
    },

    /// Strategy needs more resources to complete.
    NeedsResource(ResourceRequest),

    /// Strategy proved the problem has no solution in the searched class.
    ProvenImpossible {
        /// Formal certificate of impossibility.
        certificate: ImpossibilityProof,
        /// Trace of the impossibility proof.
        trace: TraceNode,
    },
}

impl fmt::Debug for StrategyResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StrategyResult::Solved { expr, trace } => f
                .debug_struct("Solved")
                .field("expr", expr)
                .field("trace", trace)
                .finish(),
            StrategyResult::Branch(candidates) => {
                f.debug_tuple("Branch").field(candidates).finish()
            }
            StrategyResult::Decompose { parts, .. } => f
                .debug_struct("Decompose")
                .field("parts", parts)
                // MergerFn is Arc<dyn Fn(...)> — not Debug; print sentinel.
                .field("merger", &"MergerFn(..)")
                .finish(),
            StrategyResult::Failed(r) => f.debug_tuple("Failed").field(r).finish(),
            StrategyResult::Partial {
                expr,
                reason,
                trace,
            } => f
                .debug_struct("Partial")
                .field("expr", expr)
                .field("reason", reason)
                .field("trace", trace)
                .finish(),
            StrategyResult::NeedsResource(r) => f.debug_tuple("NeedsResource").field(r).finish(),
            StrategyResult::ProvenImpossible { certificate, trace } => f
                .debug_struct("ProvenImpossible")
                .field("certificate", certificate)
                .field("trace", trace)
                .finish(),
        }
    }
}

// ── StrategyStatus ────────────────────────────────────────────────────────────

/// High-level outcome of running the full strategy loop for a problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyStatus {
    /// A complete solution was found.
    Success,
    /// All strategies were tried without finding a solution.
    Exhausted,
    /// The resource budget ran out before a solution was found.
    Timeout,
    /// A strategy proved the problem has no solution.
    CertifiedImpossible,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::resource::ResourceBudget;
    use crate::engine::trace_tree::{BranchReason, TraceNode};
    use crate::numeric::SmallInt;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn int_expr(n: i64) -> Arc<Expr> {
        Arc::new(Expr::Integer(SmallInt::from(n)))
    }

    fn dummy_trace_node() -> TraceNode {
        TraceNode::Branch {
            reason: BranchReason::StrategyCascade,
            children: vec![],
        }
    }

    fn base_ctx() -> SolveContext {
        SolveContext::new(int_expr(1), ResourceBudget::unlimited())
    }

    // ── Mock strategy ─────────────────────────────────────────────────────────

    /// Always-successful mock strategy.
    #[derive(Debug)]
    struct AlwaysSolve {
        result_value: i64,
    }

    impl Strategy for AlwaysSolve {
        fn id(&self) -> StrategyId {
            StrategyId("mock::always_solve")
        }

        fn applicable(&self, _ctx: &SolveContext) -> bool {
            true
        }

        fn priority(&self, _ctx: &SolveContext) -> f64 {
            1.0
        }

        fn apply(&self, _ctx: SolveContext) -> StrategyResult {
            StrategyResult::Solved {
                expr: int_expr(self.result_value),
                trace: dummy_trace_node(),
            }
        }
    }

    /// Always-failing mock strategy.
    #[derive(Debug)]
    struct AlwaysFail;

    impl Strategy for AlwaysFail {
        fn id(&self) -> StrategyId {
            StrategyId("mock::always_fail")
        }

        fn applicable(&self, _ctx: &SolveContext) -> bool {
            true
        }

        fn priority(&self, _ctx: &SolveContext) -> f64 {
            0.5
        }

        fn apply(&self, _ctx: SolveContext) -> StrategyResult {
            StrategyResult::Failed(FailureReason::NotApplicable)
        }
    }

    // ── Strategy trait tests ──────────────────────────────────────────────────

    #[test]
    fn fast_strategy_mock_id() {
        let s = AlwaysSolve { result_value: 42 };
        assert_eq!(s.id(), StrategyId("mock::always_solve"));
    }

    #[test]
    fn fast_strategy_mock_applicable_true() {
        let s = AlwaysSolve { result_value: 0 };
        assert!(s.applicable(&base_ctx()));
    }

    #[test]
    fn fast_strategy_mock_priority() {
        let s = AlwaysSolve { result_value: 0 };
        let p = s.priority(&base_ctx());
        assert!((p - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn fast_strategy_apply_solved_contains_expr() {
        let s = AlwaysSolve { result_value: 7 };
        let result = s.apply(base_ctx());
        match result {
            StrategyResult::Solved { expr, .. } => {
                assert!(matches!(*expr, Expr::Integer(_)));
            }
            other => panic!("Expected Solved, got {:?}", other),
        }
    }

    #[test]
    fn fast_strategy_apply_failed() {
        let s = AlwaysFail;
        let result = s.apply(base_ctx());
        assert!(matches!(result, StrategyResult::Failed(_)));
    }

    // ── StrategyResult Debug tests ────────────────────────────────────────────

    #[test]
    fn fast_strategy_result_debug_solved() {
        let r = StrategyResult::Solved {
            expr: int_expr(1),
            trace: dummy_trace_node(),
        };
        let s = format!("{:?}", r);
        assert!(s.contains("Solved"));
    }

    #[test]
    fn fast_strategy_result_debug_failed() {
        let r = StrategyResult::Failed(FailureReason::NotApplicable);
        let s = format!("{:?}", r);
        assert!(s.contains("Failed"));
    }

    #[test]
    fn fast_strategy_result_debug_decompose_shows_sentinel() {
        let merger: MergerFn = Arc::new(|_parts| {
            StrategyResult::Failed(FailureReason::Custom(
                "merger not implemented in test".to_string(),
            ))
        });
        let r = StrategyResult::Decompose {
            parts: vec![],
            merger,
        };
        let s = format!("{:?}", r);
        assert!(s.contains("MergerFn(..)"), "expected sentinel in: {s}");
    }

    #[test]
    fn fast_strategy_result_debug_partial() {
        let r = StrategyResult::Partial {
            expr: int_expr(0),
            reason: PartialReason::StepLimitReached,
            trace: dummy_trace_node(),
        };
        let s = format!("{:?}", r);
        assert!(s.contains("Partial"));
    }

    #[test]
    fn fast_strategy_result_debug_needs_resource() {
        use crate::engine::reason::ResourceRequest;
        let r = StrategyResult::NeedsResource(ResourceRequest::steps(10));
        let s = format!("{:?}", r);
        assert!(s.contains("NeedsResource"));
    }

    #[test]
    fn fast_strategy_result_debug_proven_impossible() {
        let r = StrategyResult::ProvenImpossible {
            certificate: ImpossibilityProof::NoElementaryClosure,
            trace: dummy_trace_node(),
        };
        let s = format!("{:?}", r);
        assert!(s.contains("ProvenImpossible"));
    }

    #[test]
    fn fast_strategy_result_debug_branch() {
        let candidate = StrategyCandidate {
            strategy: Box::new(AlwaysSolve { result_value: 2 }),
            priority: 0.9,
        };
        let r = StrategyResult::Branch(vec![candidate]);
        let s = format!("{:?}", r);
        assert!(s.contains("Branch"));
    }

    // ── StrategyStatus tests ──────────────────────────────────────────────────

    #[test]
    fn fast_strategy_status_eq() {
        assert_eq!(StrategyStatus::Success, StrategyStatus::Success);
        assert_eq!(StrategyStatus::Exhausted, StrategyStatus::Exhausted);
        assert_eq!(StrategyStatus::Timeout, StrategyStatus::Timeout);
        assert_eq!(
            StrategyStatus::CertifiedImpossible,
            StrategyStatus::CertifiedImpossible
        );
    }

    #[test]
    fn fast_strategy_status_ne_different_variants() {
        assert_ne!(StrategyStatus::Success, StrategyStatus::Exhausted);
        assert_ne!(StrategyStatus::Timeout, StrategyStatus::CertifiedImpossible);
    }

    #[test]
    fn fast_strategy_status_copy() {
        let s = StrategyStatus::Success;
        let s2 = s; // Copy
        assert_eq!(s, s2);
    }

    #[test]
    fn fast_strategy_status_debug() {
        assert!(format!("{:?}", StrategyStatus::Success).contains("Success"));
        assert!(
            format!("{:?}", StrategyStatus::CertifiedImpossible).contains("CertifiedImpossible")
        );
    }

    // ── SubProblem tests ──────────────────────────────────────────────────────

    #[test]
    fn fast_strategy_sub_problem_debug() {
        let sp = SubProblem {
            context: base_ctx(),
            label: "left_term".to_string(),
        };
        let s = format!("{:?}", sp);
        assert!(s.contains("SubProblem"));
        assert!(s.contains("left_term"));
    }

    #[test]
    fn fast_strategy_strategy_candidate_debug() {
        let c = StrategyCandidate {
            strategy: Box::new(AlwaysFail),
            priority: 0.1,
        };
        let s = format!("{:?}", c);
        assert!(s.contains("StrategyCandidate"));
    }
}
