//! Pedagogic trace reordering — expands CacheHit nodes inline.
//!
//! [`PedagogicReorder`] walks a [`TraceNode`] tree and replaces every
//! [`TraceNode::CacheHit`] with the subtree stored in its `cached_trace`
//! field, recursing into the replacement. The output is semantically
//! identical to the original trace but contains no `CacheHit` variants:
//! cached steps appear as if they were computed fresh.

use crate::engine::trace_tree::{TraceBranch, TraceNode, TraceTree};

// ── PedagogicReorder ──────────────────────────────────────────────────────────

/// Reorders a trace for pedagogic presentation by expanding all
/// [`TraceNode::CacheHit`] nodes inline.
///
/// The output contains no `CacheHit` variants — cached steps appear as if
/// they were computed fresh. Branch and Join nodes have their children
/// reordered recursively.
pub struct PedagogicReorder;

impl PedagogicReorder {
    /// Apply pedagogic reordering to a single [`TraceNode`].
    ///
    /// `Step` nodes pass through unchanged. `Branch` and `Join` nodes
    /// recursively reorder their children. `CacheHit` nodes are replaced by
    /// the result of applying reordering to their `cached_trace`, so nested
    /// `CacheHit` nodes are flattened completely.
    #[must_use]
    pub fn apply(node: &TraceNode) -> TraceNode {
        match node {
            TraceNode::Step(_) => node.clone(),

            TraceNode::Branch { reason, children } => TraceNode::Branch {
                reason: reason.clone(),
                children: children
                    .iter()
                    .map(|b| TraceBranch {
                        strategy: b.strategy,
                        outcome: b.outcome.clone(),
                        nodes: Self::apply_tree(&b.nodes),
                    })
                    .collect(),
            },

            TraceNode::Join { reason, parts } => TraceNode::Join {
                reason: reason.clone(),
                parts: parts.iter().map(Self::apply_tree).collect(),
            },

            // Expand the cached trace inline, then recurse into it in case
            // the cached trace itself contains further CacheHit nodes.
            TraceNode::CacheHit { cached_trace, .. } => Self::apply(cached_trace),
        }
    }

    /// Apply pedagogic reordering to an entire [`TraceTree`].
    ///
    /// Every node in `tree.nodes` is reordered via [`Self::apply`].
    #[must_use]
    pub fn apply_tree(tree: &TraceTree) -> TraceTree {
        TraceTree {
            nodes: tree.nodes.iter().map(Self::apply).collect(),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::cache::entry::CacheSource;
    use crate::engine::canonical_pattern::PatternHash;
    use crate::engine::reason::FailureReason;
    use crate::engine::trace_tree::{
        BranchOutcome, BranchReason, JoinReason, StrategyId, TraceBranch, TraceNode, TraceTree,
    };
    use crate::numeric::trace::{Step, TechniqueTag};

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_step(tag: TechniqueTag) -> Step {
        Step::new(tag, "test")
    }

    fn step_node(tag: TechniqueTag) -> TraceNode {
        TraceNode::Step(make_step(tag))
    }

    fn cache_hit(cached: TraceNode) -> TraceNode {
        TraceNode::CacheHit {
            source: CacheSource::SolveCache,
            pattern_hash: PatternHash(0xabcd),
            cached_trace: Box::new(cached),
        }
    }

    /// Recursively checks that no `CacheHit` variant exists anywhere in `node`.
    fn assert_no_cache_hit(node: &TraceNode) {
        match node {
            TraceNode::CacheHit { .. } => panic!("unexpected CacheHit in output"),
            TraceNode::Step(_) => {}
            TraceNode::Branch { children, .. } => {
                for b in children {
                    assert_no_cache_hit_tree(&b.nodes);
                }
            }
            TraceNode::Join { parts, .. } => {
                for part in parts {
                    assert_no_cache_hit_tree(part);
                }
            }
        }
    }

    fn assert_no_cache_hit_tree(tree: &TraceTree) {
        for node in &tree.nodes {
            assert_no_cache_hit(node);
        }
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn fast_reorder_step_unchanged() {
        let node = step_node(TechniqueTag::Simplification);
        let result = PedagogicReorder::apply(&node);
        match result {
            TraceNode::Step(s) => assert_eq!(s.tag, TechniqueTag::Simplification),
            _ => panic!("expected Step"),
        }
    }

    #[test]
    fn fast_reorder_cache_hit_expanded() {
        let inner = step_node(TechniqueTag::Factoring);
        let node = cache_hit(inner);
        let result = PedagogicReorder::apply(&node);
        match result {
            TraceNode::Step(s) => assert_eq!(s.tag, TechniqueTag::Factoring),
            _ => panic!("CacheHit should expand to its inner Step"),
        }
    }

    #[test]
    fn fast_reorder_nested_cache_hit() {
        // CacheHit containing another CacheHit containing a Step
        let innermost = step_node(TechniqueTag::Expansion);
        let middle = cache_hit(innermost);
        let outer = cache_hit(middle);
        let result = PedagogicReorder::apply(&outer);
        match result {
            TraceNode::Step(s) => assert_eq!(s.tag, TechniqueTag::Expansion),
            _ => panic!("nested CacheHits should both be flattened"),
        }
    }

    #[test]
    fn fast_reorder_branch_children_processed() {
        // Branch with one child whose TraceTree contains a CacheHit
        let mut child_tree = TraceTree::new();
        child_tree
            .nodes
            .push(cache_hit(step_node(TechniqueTag::Substitution)));

        let branch_node = TraceNode::Branch {
            reason: BranchReason::StrategyCascade,
            children: vec![TraceBranch {
                strategy: StrategyId("s1"),
                outcome: BranchOutcome::Succeeded,
                nodes: child_tree,
            }],
        };

        let result = PedagogicReorder::apply(&branch_node);
        match &result {
            TraceNode::Branch { children, .. } => {
                assert_eq!(children.len(), 1);
                let reordered_nodes = &children[0].nodes;
                assert_eq!(reordered_nodes.nodes.len(), 1);
                match &reordered_nodes.nodes[0] {
                    TraceNode::Step(s) => assert_eq!(s.tag, TechniqueTag::Substitution),
                    _ => panic!("CacheHit inside branch child should be expanded"),
                }
            }
            _ => panic!("expected Branch"),
        }
    }

    #[test]
    fn fast_reorder_join_parts_processed() {
        // Join with two parts, one of which contains a CacheHit
        let mut part1 = TraceTree::new();
        part1.nodes.push(step_node(TechniqueTag::PowerRule));

        let mut part2 = TraceTree::new();
        part2
            .nodes
            .push(cache_hit(step_node(TechniqueTag::ChainRule)));

        let join_node = TraceNode::Join {
            reason: JoinReason::DivideAndConquer,
            parts: vec![part1, part2],
        };

        let result = PedagogicReorder::apply(&join_node);
        match &result {
            TraceNode::Join { parts, .. } => {
                assert_eq!(parts.len(), 2);
                // part1 unchanged
                match &parts[0].nodes[0] {
                    TraceNode::Step(s) => assert_eq!(s.tag, TechniqueTag::PowerRule),
                    _ => panic!("expected Step in part1"),
                }
                // part2 expanded
                match &parts[1].nodes[0] {
                    TraceNode::Step(s) => assert_eq!(s.tag, TechniqueTag::ChainRule),
                    _ => panic!("CacheHit in part2 should be expanded"),
                }
            }
            _ => panic!("expected Join"),
        }
    }

    #[test]
    fn fast_reorder_idempotent() {
        // Build a tree with a Branch containing a CacheHit in one child,
        // and a top-level CacheHit.
        let mut child_tree = TraceTree::new();
        child_tree
            .nodes
            .push(cache_hit(step_node(TechniqueTag::USubstitution)));

        let mut tree = TraceTree::new();
        tree.nodes.push(TraceNode::Branch {
            reason: BranchReason::AlternativeStrategies,
            children: vec![
                TraceBranch {
                    strategy: StrategyId("a"),
                    outcome: BranchOutcome::Failed(FailureReason::NotApplicable),
                    nodes: child_tree,
                },
                TraceBranch {
                    strategy: StrategyId("b"),
                    outcome: BranchOutcome::Succeeded,
                    nodes: {
                        let mut t = TraceTree::new();
                        t.nodes.push(step_node(TechniqueTag::IntegrationByParts));
                        t
                    },
                },
            ],
        });
        tree.nodes
            .push(cache_hit(step_node(TechniqueTag::Simplification)));

        let once = PedagogicReorder::apply_tree(&tree);
        let twice = PedagogicReorder::apply_tree(&once);

        // Compare step counts — same structure means same step count.
        assert_eq!(once.step_count(), twice.step_count());
        // No CacheHit in either pass.
        assert_no_cache_hit_tree(&once);
        assert_no_cache_hit_tree(&twice);
    }

    #[test]
    fn fast_reorder_no_cache_hit_in_output() {
        // Construct a complex tree with CacheHits at every level.
        let mut part = TraceTree::new();
        part.nodes
            .push(cache_hit(step_node(TechniqueTag::Factoring)));

        let join = TraceNode::Join {
            reason: JoinReason::TermParallelMerge,
            parts: vec![part],
        };
        // Wrap the join inside a CacheHit
        let outer_hit = cache_hit(join);

        let mut child_tree = TraceTree::new();
        child_tree.nodes.push(outer_hit);

        let branch = TraceNode::Branch {
            reason: BranchReason::DomainExtension,
            children: vec![TraceBranch {
                strategy: StrategyId("c"),
                outcome: BranchOutcome::Succeeded,
                nodes: child_tree,
            }],
        };

        let mut tree = TraceTree::new();
        tree.nodes.push(branch);
        tree.nodes
            .push(cache_hit(step_node(TechniqueTag::Cancellation)));

        let result = PedagogicReorder::apply_tree(&tree);
        assert_no_cache_hit_tree(&result);
    }
}
