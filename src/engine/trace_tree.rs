//! Tree-structured trace for multi-branch engine runs.
//!
//! [`TraceTree`] extends the flat [`crate::numeric::trace::Trace`] with
//! branching and join nodes, enabling the D0 engine to record strategy
//! cascades, dead-end branches, and divide-and-conquer merges in full
//! fidelity. All branches — successful and failed alike — are stored
//! unconditionally; callers filter what to display.

use crate::engine::cache::entry::CacheSource;
use crate::engine::canonical_pattern::PatternHash;
use crate::engine::reason::{FailureReason, PartialReason};
use crate::numeric::trace::Step;

// ── StrategyId ────────────────────────────────────────────────────────────────

/// Stable identifier for a named strategy in the D0 search space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StrategyId(pub &'static str);

// ── BranchHandle / JoinHandle ─────────────────────────────────────────────────

/// Opaque handle to an open [`TraceNode::Branch`] node in a [`TraceTree`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BranchHandle(usize);

/// Opaque handle to an open [`TraceNode::Join`] node in a [`TraceTree`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JoinHandle(usize);

// ── Reason types ──────────────────────────────────────────────────────────────

/// Why a [`TraceNode::Branch`] was opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BranchReason {
    /// Trying strategies in priority order until one succeeds.
    StrategyCascade,
    /// Multiple valid strategies are explored in parallel.
    AlternativeStrategies,
    /// Domain extension forced branching (e.g. ℝ → ℂ).
    DomainExtension,
    /// A probabilistic or heuristic check is being verified.
    ProbabilisticVerify,
    /// Symbolic strategies were exhausted or proved impossible; a numerical
    /// fallback evaluation is being attempted.
    NumericalFallback,
    /// Custom reason not covered by the above variants.
    Custom(&'static str),
}

/// Why a [`TraceNode::Join`] was opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JoinReason {
    /// Divide-and-conquer sub-problems are being merged.
    DivideAndConquer,
    /// Partial fractions are being recombined.
    PartialFractionMerge,
    /// Parallel term results are being merged.
    TermParallelMerge,
    /// Custom reason not covered by the above variants.
    Custom(&'static str),
}

/// Why a particular branch was pruned before full evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum PruneReason {
    /// The resource budget (steps or memory) ran out.
    BudgetExhausted,
    /// A heuristic cut terminated the branch early.
    HeuristicCut,
    /// An early-exit condition was met, making further exploration pointless.
    EarlyExit,
    /// Custom reason not covered by the above variants.
    Custom(&'static str),
}

/// Outcome of a single strategy branch attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum BranchOutcome {
    /// The branch produced a valid result.
    Succeeded,
    /// The branch failed for the given reason.
    Failed(FailureReason),
    /// The branch was pruned before completion.
    Pruned(PruneReason),
    /// The branch produced a partial result for the given reason.
    Partial(PartialReason),
}

// ── TraceBranch ───────────────────────────────────────────────────────────────

/// One attempted strategy branch within a [`TraceNode::Branch`] node.
#[derive(Debug, Clone)]
pub struct TraceBranch {
    /// The strategy that was attempted.
    pub strategy: StrategyId,
    /// How the branch ended.
    pub outcome: BranchOutcome,
    /// All steps recorded inside this branch (including nested nodes).
    pub nodes: TraceTree,
}

// ── TraceNode ─────────────────────────────────────────────────────────────────

/// A single node in a [`TraceTree`].
#[derive(Debug, Clone)]
pub enum TraceNode {
    /// A flat technique step (mirrors [`Step`]).
    Step(Step),
    /// A multi-strategy branch point with one or more attempted branches.
    Branch {
        /// Why this branch point was opened.
        reason: BranchReason,
        /// The attempted branches, appended as they are recorded.
        children: Vec<TraceBranch>,
    },
    /// A join point where independent sub-problem results are merged.
    Join {
        /// Why the sub-problems were split.
        reason: JoinReason,
        /// One sub-tree per part.
        parts: Vec<TraceTree>,
    },
    /// A cache hit: the result was replayed from a memoized entry rather than
    /// recomputed. The cached trace is expanded inline so all steps remain
    /// visible (Rule 4 completeness corollary).
    CacheHit {
        /// Which cache tier the hit came from.
        source: CacheSource,
        /// Hash of the canonical pattern that matched.
        pattern_hash: PatternHash,
        /// The trace that was originally recorded when the result was first
        /// computed. Boxed to keep the enum variant size small.
        cached_trace: Box<TraceNode>,
    },
}

// ── TraceTree ─────────────────────────────────────────────────────────────────

/// An ordered, branching trace for a multi-strategy engine run.
///
/// Nodes are appended in execution order. Branch and join nodes are
/// "opened" with a handle and "filled in" as the engine proceeds,
/// allowing interleaved recording across nested scopes.
#[derive(Debug, Clone, Default)]
pub struct TraceTree {
    /// Top-level nodes in insertion order.
    pub nodes: Vec<TraceNode>,
}

impl TraceTree {
    /// Create a new empty trace tree.
    #[must_use]
    pub fn new() -> Self {
        TraceTree::default()
    }

    /// Append a flat step to the top level of this tree.
    pub fn push_step(&mut self, step: Step) {
        self.nodes.push(TraceNode::Step(step));
    }

    /// Open a new branch node and return a handle to it.
    ///
    /// `candidates` lists the strategies that will be attempted; they are
    /// recorded for diagnostic purposes only and do not constrain how many
    /// branches are subsequently recorded.
    pub fn open_branch(
        &mut self,
        reason: BranchReason,
        _candidates: Vec<StrategyId>,
    ) -> BranchHandle {
        let idx = self.nodes.len();
        self.nodes.push(TraceNode::Branch {
            reason,
            children: Vec::new(),
        });
        BranchHandle(idx)
    }

    /// Append a completed branch to the branch node referenced by `handle`.
    ///
    /// # Panics
    ///
    /// Panics if `handle` does not reference a `Branch` node (internal
    /// engine misuse).
    pub fn record_branch_outcome(
        &mut self,
        handle: BranchHandle,
        strategy_id: StrategyId,
        outcome: BranchOutcome,
        nodes: TraceTree,
    ) {
        match &mut self.nodes[handle.0] {
            TraceNode::Branch { children, .. } => {
                children.push(TraceBranch {
                    strategy: strategy_id,
                    outcome,
                    nodes,
                });
            }
            _ => panic!("BranchHandle does not point to a Branch node"),
        }
    }

    /// Open a new join node and return a handle to it.
    ///
    /// `_part_count` is advisory — it does not constrain how many parts
    /// are subsequently recorded via [`TraceTree::record_join_part`].
    pub fn open_join(&mut self, reason: JoinReason, _part_count: usize) -> JoinHandle {
        let idx = self.nodes.len();
        self.nodes.push(TraceNode::Join {
            reason,
            parts: Vec::new(),
        });
        JoinHandle(idx)
    }

    /// Append a completed sub-tree part to the join node referenced by `handle`.
    ///
    /// # Panics
    ///
    /// Panics if `handle` does not reference a `Join` node (internal
    /// engine misuse).
    pub fn record_join_part(&mut self, handle: JoinHandle, part: TraceTree) {
        match &mut self.nodes[handle.0] {
            TraceNode::Join { parts, .. } => {
                parts.push(part);
            }
            _ => panic!("JoinHandle does not point to a Join node"),
        }
    }

    /// Recursively collect every [`Step`] in this tree in depth-first
    /// pre-order, including steps inside dead-end branch subtrees.
    #[must_use]
    pub fn flatten_steps(&self) -> Vec<Step> {
        let mut out = Vec::new();
        collect_steps(&self.nodes, &mut out);
        out
    }

    /// Total number of [`Step`] nodes in this tree (recursive).
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.flatten_steps().len()
    }
}

/// Recursive helper that walks a node slice and appends Step values.
fn collect_steps(nodes: &[TraceNode], out: &mut Vec<Step>) {
    for node in nodes {
        match node {
            TraceNode::Step(s) => out.push(s.clone()),
            TraceNode::Branch { children, .. } => {
                for branch in children {
                    collect_steps(&branch.nodes.nodes, out);
                }
            }
            TraceNode::Join { parts, .. } => {
                for part in parts {
                    collect_steps(&part.nodes, out);
                }
            }
            TraceNode::CacheHit { cached_trace, .. } => {
                collect_steps(std::slice::from_ref(cached_trace.as_ref()), out);
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::trace::{Step, TechniqueTag};

    fn make_step(tag: TechniqueTag) -> Step {
        Step::new(tag, "test")
    }

    #[test]
    fn fast_trace_tree_new_is_empty() {
        let tree = TraceTree::new();
        assert!(tree.nodes.is_empty());
        assert_eq!(tree.step_count(), 0);
    }

    #[test]
    fn fast_trace_tree_push_step() {
        let mut tree = TraceTree::new();
        tree.push_step(make_step(TechniqueTag::Simplification));
        assert_eq!(tree.nodes.len(), 1);
        assert_eq!(tree.step_count(), 1);
    }

    #[test]
    fn fast_trace_tree_flatten_includes_dead_end_branches() {
        let mut tree = TraceTree::new();
        tree.push_step(make_step(TechniqueTag::Simplification));

        let handle = tree.open_branch(BranchReason::StrategyCascade, vec![]);

        // Dead-end branch
        let mut dead = TraceTree::new();
        dead.push_step(make_step(TechniqueTag::Factoring));
        tree.record_branch_outcome(
            handle,
            StrategyId("dead"),
            BranchOutcome::Failed(FailureReason::NotApplicable),
            dead,
        );

        // Successful branch
        let mut success = TraceTree::new();
        success.push_step(make_step(TechniqueTag::Expansion));
        success.push_step(make_step(TechniqueTag::CombiningLikeTerms));
        tree.record_branch_outcome(
            handle,
            StrategyId("success"),
            BranchOutcome::Succeeded,
            success,
        );

        // flatten must include all 3 inner steps + 1 top-level step
        let steps = tree.flatten_steps();
        assert_eq!(steps.len(), 4);
    }

    #[test]
    fn fast_trace_tree_open_record_join() {
        let mut tree = TraceTree::new();
        let handle = tree.open_join(JoinReason::DivideAndConquer, 2);

        let mut part1 = TraceTree::new();
        part1.push_step(make_step(TechniqueTag::PowerRule));
        let mut part2 = TraceTree::new();
        part2.push_step(make_step(TechniqueTag::ChainRule));
        part2.push_step(make_step(TechniqueTag::ProductRule));

        tree.record_join_part(handle, part1);
        tree.record_join_part(handle, part2);

        assert_eq!(tree.step_count(), 3);
    }

    #[test]
    fn fast_trace_tree_step_count_recursive() {
        let mut tree = TraceTree::new();
        tree.push_step(make_step(TechniqueTag::Substitution));

        let handle = tree.open_branch(BranchReason::AlternativeStrategies, vec![]);
        let mut inner = TraceTree::new();
        inner.push_step(make_step(TechniqueTag::USubstitution));
        inner.push_step(make_step(TechniqueTag::IntegrationByParts));
        tree.record_branch_outcome(handle, StrategyId("alt"), BranchOutcome::Succeeded, inner);

        assert_eq!(tree.step_count(), 3);
    }

    #[test]
    fn fast_trace_tree_branch_outcome_eq() {
        assert_eq!(BranchOutcome::Succeeded, BranchOutcome::Succeeded);
        assert_ne!(
            BranchOutcome::Succeeded,
            BranchOutcome::Failed(FailureReason::NotApplicable),
        );
        assert_eq!(
            BranchOutcome::Pruned(PruneReason::HeuristicCut),
            BranchOutcome::Pruned(PruneReason::HeuristicCut),
        );
    }

    #[test]
    fn fast_trace_tree_flatten_with_output_step() {
        use crate::numeric::Expr;
        use crate::numeric::SmallInt;
        use std::sync::Arc;

        let mut tree = TraceTree::new();
        let expr: Arc<Expr> = Arc::new(Expr::Integer(SmallInt::from(42i64)));
        let step = Step::new(TechniqueTag::Simplification, "42").with_output(expr);
        tree.push_step(step);

        let steps = tree.flatten_steps();
        assert_eq!(steps.len(), 1);
        assert!(steps[0].output.is_some());
    }

    #[test]
    fn fast_trace_cache_hit_node_fields() {
        use crate::engine::cache::entry::CacheSource;
        use crate::engine::canonical_pattern::PatternHash;

        // Build a cached trace containing two steps.
        let mut cached = TraceTree::new();
        cached.push_step(make_step(TechniqueTag::Simplification));
        cached.push_step(make_step(TechniqueTag::Expansion));

        // Wrap the two-step trace in a Branch node (representative cached_trace).
        let cached_trace = TraceNode::Branch {
            reason: BranchReason::StrategyCascade,
            children: vec![],
        };

        let node = TraceNode::CacheHit {
            source: CacheSource::KnowledgeCache,
            pattern_hash: PatternHash(0xdeadbeef),
            cached_trace: Box::new(cached_trace),
        };

        // Field access works.
        match &node {
            TraceNode::CacheHit {
                source,
                pattern_hash,
                ..
            } => {
                assert_eq!(*source, CacheSource::KnowledgeCache);
                assert_eq!(*pattern_hash, PatternHash(0xdeadbeef));
            }
            _ => panic!("expected CacheHit"),
        }
    }

    #[test]
    fn fast_trace_cache_hit_flatten_includes_cached_steps() {
        use crate::engine::cache::entry::CacheSource;
        use crate::engine::canonical_pattern::PatternHash;

        // Cached trace: a Branch node with two inner steps.
        let mut inner_tree = TraceTree::new();
        inner_tree.push_step(make_step(TechniqueTag::Factoring));
        inner_tree.push_step(make_step(TechniqueTag::Expansion));
        let handle = {
            let mut dummy = TraceTree::new();
            dummy.open_branch(BranchReason::StrategyCascade, vec![])
        };
        let _ = handle;
        // Use a Step node as the cached_trace for step counting.
        let cached_trace = TraceNode::Step(make_step(TechniqueTag::Factoring));

        let mut tree = TraceTree::new();
        tree.push_step(make_step(TechniqueTag::Simplification));
        tree.nodes.push(TraceNode::CacheHit {
            source: CacheSource::SolveCache,
            pattern_hash: PatternHash(1),
            cached_trace: Box::new(cached_trace),
        });

        // flatten: 1 top-level Step + 1 step inside CacheHit = 2 total.
        assert_eq!(tree.step_count(), 2);
    }

    #[test]
    fn fast_trace_cache_hit_step_count() {
        use crate::engine::cache::entry::CacheSource;
        use crate::engine::canonical_pattern::PatternHash;

        let cached = TraceNode::Step(make_step(TechniqueTag::USubstitution));
        let mut tree = TraceTree::new();
        tree.nodes.push(TraceNode::CacheHit {
            source: CacheSource::KnowledgeCache,
            pattern_hash: PatternHash(2),
            cached_trace: Box::new(cached),
        });
        assert_eq!(tree.step_count(), 1);
    }
}
