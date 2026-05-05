//! Convert a [`TraceTree`] to a flat [`Vec<NarratedStep>`].
//!
//! Walks the tree in depth-first pre-order, emitting every step including
//! those inside dead-end branch subtrees (Rule 4 completeness corollary).
//! Branch nodes and join nodes are traversed fully — all children and parts
//! are included, successful paths and failed paths alike.

use crate::api::response::NarratedStep;
use crate::engine::trace_tree::{TraceNode, TraceTree};
use crate::numeric::compile::decompile;
use crate::numeric::trace::Step;

use super::super::api::dispatch::helpers::{build_narrated_step, template_id_for_tag};

// ── Public converter ──────────────────────────────────────────────────────────

/// Convert a [`TraceTree`] into a flat ordered list of [`NarratedStep`]s.
///
/// The walk is depth-first pre-order. For [`TraceNode::Branch`] nodes every
/// child subtree is included — dead-end branches appear in the output before
/// the successful branch, matching the order in which they were attempted.
/// For [`TraceNode::Join`] nodes all parts are included in order.
///
/// Pass `narrate = false` to get an empty vec without walking the tree.
#[must_use]
pub fn trace_tree_to_steps(tree: &TraceTree, narrate: bool) -> Vec<NarratedStep> {
    if !narrate {
        return Vec::new();
    }
    let mut out = Vec::new();
    collect_nodes(&tree.nodes, &mut out);
    out
}

// ── Internal walker ───────────────────────────────────────────────────────────

fn collect_nodes(nodes: &[TraceNode], out: &mut Vec<NarratedStep>) {
    for node in nodes {
        match node {
            TraceNode::Step(step) => {
                out.push(step_to_narrated(step));
            }
            TraceNode::Branch { children, .. } => {
                // Include ALL branches — dead-ends and successful alike.
                for branch in children {
                    collect_nodes(&branch.nodes.nodes, out);
                }
            }
            TraceNode::Join { parts, .. } => {
                for part in parts {
                    collect_nodes(&part.nodes, out);
                }
            }
            // Expand cache hits inline — all cached steps are included
            // so the caller sees the full trace (Rule 4 completeness).
            TraceNode::CacheHit { cached_trace, .. } => {
                collect_nodes(std::slice::from_ref(cached_trace.as_ref()), out);
            }
        }
    }
}

// ── Step conversion ───────────────────────────────────────────────────────────

fn step_to_narrated(step: &Step) -> NarratedStep {
    let template_id = template_id_for_tag(step.tag);
    let narrative = build_narrated_step(template_id, step);
    NarratedStep {
        tag: step.tag,
        difficulty: step.tag.difficulty(),
        narrative,
        path: None,
        input: step.input.as_ref().map(|arc| decompile(arc)),
        output: step.output.as_ref().map(|arc| decompile(arc)),
        unit_trace: None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::reason::FailureReason;
    use crate::engine::trace_tree::{
        BranchOutcome, BranchReason, JoinReason, StrategyId, TraceTree,
    };
    use crate::numeric::trace::{Step, TechniqueTag};
    use crate::numeric::Expr;
    use std::sync::Arc;

    fn make_step(tag: TechniqueTag) -> Step {
        Step::new(tag, "test detail")
    }

    fn make_step_with_output(tag: TechniqueTag, n: i64) -> Step {
        use crate::numeric::SmallInt;
        let expr: Arc<Expr> = Arc::new(Expr::Integer(SmallInt::from(n)));
        Step::new(tag, "with output").with_output(expr)
    }

    // ── narrate=false ─────────────────────────────────────────────────────────

    #[test]
    fn fast_narrate_false_returns_empty() {
        let mut tree = TraceTree::new();
        tree.push_step(make_step(TechniqueTag::Simplification));
        let result = trace_tree_to_steps(&tree, false);
        assert!(result.is_empty(), "narrate=false must produce empty vec");
    }

    // ── Flat tree ─────────────────────────────────────────────────────────────

    #[test]
    fn fast_narrate_flat_tree_single_step() {
        let mut tree = TraceTree::new();
        tree.push_step(make_step(TechniqueTag::Simplification));
        let steps = trace_tree_to_steps(&tree, true);
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].tag, TechniqueTag::Simplification);
    }

    #[test]
    fn fast_narrate_flat_tree_multiple_steps_preserve_order() {
        let mut tree = TraceTree::new();
        tree.push_step(make_step(TechniqueTag::Expansion));
        tree.push_step(make_step(TechniqueTag::Factoring));
        tree.push_step(make_step(TechniqueTag::Simplification));
        let steps = trace_tree_to_steps(&tree, true);
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0].tag, TechniqueTag::Expansion);
        assert_eq!(steps[1].tag, TechniqueTag::Factoring);
        assert_eq!(steps[2].tag, TechniqueTag::Simplification);
    }

    // ── Branch nodes (including dead-ends) ───────────────────────────────────

    #[test]
    fn fast_narrate_branched_tree_includes_dead_ends() {
        let mut tree = TraceTree::new();
        tree.push_step(make_step(TechniqueTag::Simplification));

        let handle = tree.open_branch(BranchReason::StrategyCascade, vec![]);

        // Dead-end branch: 1 step
        let mut dead = TraceTree::new();
        dead.push_step(make_step(TechniqueTag::Factoring));
        tree.record_branch_outcome(
            handle,
            StrategyId("dead"),
            BranchOutcome::Failed(FailureReason::NotApplicable),
            dead,
        );

        // Successful branch: 2 steps
        let mut success = TraceTree::new();
        success.push_step(make_step(TechniqueTag::USubstitution));
        success.push_step(make_step(TechniqueTag::Expansion));
        tree.record_branch_outcome(
            handle,
            StrategyId("success"),
            BranchOutcome::Succeeded,
            success,
        );

        // 1 top-level + 1 dead-end + 2 successful = 4 total
        let steps = trace_tree_to_steps(&tree, true);
        assert_eq!(steps.len(), 4);
        assert_eq!(steps[0].tag, TechniqueTag::Simplification);
        assert_eq!(steps[1].tag, TechniqueTag::Factoring); // dead-end first
        assert_eq!(steps[2].tag, TechniqueTag::USubstitution);
        assert_eq!(steps[3].tag, TechniqueTag::Expansion);
    }

    // ── Join nodes ────────────────────────────────────────────────────────────

    #[test]
    fn fast_narrate_join_includes_all_parts() {
        let mut tree = TraceTree::new();
        let handle = tree.open_join(JoinReason::DivideAndConquer, 2);

        let mut part1 = TraceTree::new();
        part1.push_step(make_step(TechniqueTag::PowerRule));

        let mut part2 = TraceTree::new();
        part2.push_step(make_step(TechniqueTag::ChainRule));
        part2.push_step(make_step(TechniqueTag::ProductRule));

        tree.record_join_part(handle, part1);
        tree.record_join_part(handle, part2);

        let steps = trace_tree_to_steps(&tree, true);
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0].tag, TechniqueTag::PowerRule);
        assert_eq!(steps[1].tag, TechniqueTag::ChainRule);
        assert_eq!(steps[2].tag, TechniqueTag::ProductRule);
    }

    // ── Step metadata propagation ─────────────────────────────────────────────

    #[test]
    fn fast_narrate_step_output_propagated() {
        let mut tree = TraceTree::new();
        tree.push_step(make_step_with_output(TechniqueTag::Simplification, 42));
        let steps = trace_tree_to_steps(&tree, true);
        assert_eq!(steps.len(), 1);
        assert!(
            steps[0].output.is_some(),
            "output expression must be propagated"
        );
    }

    #[test]
    fn fast_narrate_difficulty_mapped_from_tag() {
        use crate::numeric::trace::TechniqueDifficulty;

        let mut tree = TraceTree::new();
        tree.push_step(make_step(TechniqueTag::Simplification));
        tree.push_step(make_step(TechniqueTag::TaylorExpansion));

        let steps = trace_tree_to_steps(&tree, true);
        assert_eq!(steps[0].difficulty, TechniqueDifficulty::Elementary);
        assert_eq!(steps[1].difficulty, TechniqueDifficulty::Advanced);
    }

    // ── Empty tree ────────────────────────────────────────────────────────────

    #[test]
    fn fast_narrate_empty_tree_returns_empty() {
        let tree = TraceTree::new();
        let steps = trace_tree_to_steps(&tree, true);
        assert!(steps.is_empty());
    }
}
