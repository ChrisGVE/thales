//! Integration tests for the D0 engine infrastructure.
//!
//! Tests the strategy cascade, trace tree, execution modes, and fallback
//! through the public API of the `thales::engine` module.

use std::sync::Arc;

use thales::engine::{
    BranchOutcome, BranchReason, ExecutionMode, FailureReason, FallbackConfig, FallbackRunner,
    ImpossibilityProof, JoinReason, NumericalEvaluator, NumericalEvaluatorRegistry,
    NumericalResult, PartialReason, PrecisionAttemptOutcome, PrecisionLevel, ResourceBudget,
    SequentialRunner, SolveContext, StrategyId, StrategyResult, StrategyRunner, TraceNode,
    TraceTree,
};
use thales::engine::{FallbackTrigger, TreeComparison};
use thales::engine::{MergerFn, Strategy, StrategyCandidate, SubProblem, TraceBranch};
use thales::numeric::expr::Expr;
use thales::numeric::SmallInt;

// ── Shared helpers ────────────────────────────────────────────────────────────

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

// ── Mock strategies ───────────────────────────────────────────────────────────

#[derive(Debug)]
struct SolveStrategy {
    priority: f64,
    value: i64,
}

impl Strategy for SolveStrategy {
    fn id(&self) -> StrategyId {
        StrategyId("integration::solve")
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
        StrategyId("integration::fail")
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
        StrategyId("integration::impossible")
    }
    fn applicable(&self, _ctx: &SolveContext) -> bool {
        true
    }
    fn priority(&self, _ctx: &SolveContext) -> f64 {
        0.0
    }
    fn apply(&self, _ctx: SolveContext) -> StrategyResult {
        StrategyResult::ProvenImpossible {
            certificate: ImpossibilityProof::NoKovacicSolution,
            trace: dummy_trace(),
        }
    }
}

#[derive(Debug)]
struct StructuralErrorStrategy;

impl Strategy for StructuralErrorStrategy {
    fn id(&self) -> StrategyId {
        StrategyId("integration::structural_error")
    }
    fn applicable(&self, _ctx: &SolveContext) -> bool {
        true
    }
    fn priority(&self, _ctx: &SolveContext) -> f64 {
        0.0
    }
    fn apply(&self, _ctx: SolveContext) -> StrategyResult {
        #[derive(Debug)]
        struct TestErr;
        impl std::fmt::Display for TestErr {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "structural test error")
            }
        }
        impl std::error::Error for TestErr {}
        StrategyResult::Failed(FailureReason::StructuralError(Arc::new(TestErr)))
    }
}

#[derive(Debug)]
struct PartialStrategy {
    priority: f64,
}

impl Strategy for PartialStrategy {
    fn id(&self) -> StrategyId {
        StrategyId("integration::partial")
    }
    fn applicable(&self, _ctx: &SolveContext) -> bool {
        true
    }
    fn priority(&self, _ctx: &SolveContext) -> f64 {
        self.priority
    }
    fn apply(&self, _ctx: SolveContext) -> StrategyResult {
        StrategyResult::Partial {
            expr: int_expr(3),
            reason: PartialReason::StepLimitReached,
            trace: dummy_trace(),
        }
    }
}

#[derive(Debug)]
struct DecomposeStrategy;

impl Strategy for DecomposeStrategy {
    fn id(&self) -> StrategyId {
        StrategyId("integration::decompose")
    }
    fn applicable(&self, _ctx: &SolveContext) -> bool {
        true
    }
    fn priority(&self, _ctx: &SolveContext) -> f64 {
        0.0
    }
    fn apply(&self, ctx: SolveContext) -> StrategyResult {
        let sub1 = SubProblem {
            context: ctx.fork().with_expr(int_expr(10)),
            label: "left".to_string(),
        };
        let sub2 = SubProblem {
            context: ctx.fork().with_expr(int_expr(20)),
            label: "right".to_string(),
        };
        let merger: MergerFn = Arc::new(|parts| {
            // Merge: return a Solved with an integer 99 to signal merging happened.
            let _ = parts;
            StrategyResult::Solved {
                expr: int_expr(99),
                trace: TraceNode::Branch {
                    reason: BranchReason::StrategyCascade,
                    children: vec![],
                },
            }
        });
        StrategyResult::Decompose {
            parts: vec![sub1, sub2],
            merger,
        }
    }
}

// ── Mock NumericalEvaluator for fallback integration tests ────────────────────

#[derive(Debug)]
struct AlwaysSuccessEvaluator {
    id: &'static str,
}

impl NumericalEvaluator for AlwaysSuccessEvaluator {
    fn id(&self) -> StrategyId {
        StrategyId(self.id)
    }
    fn priority(&self) -> f64 {
        0.5
    }
    fn applicable(&self, _ctx: &SolveContext, _trigger: &FallbackTrigger) -> bool {
        true
    }
    fn evaluate(
        &self,
        _ctx: &SolveContext,
        _trigger: &FallbackTrigger,
        precision: PrecisionLevel,
    ) -> PrecisionAttemptOutcome {
        PrecisionAttemptOutcome::Success(NumericalResult {
            value: int_expr(0),
            precision,
            digits_achieved: 15,
            error_bound: None,
            approximate: true,
            precision_loss: false,
            evaluator_id: StrategyId(self.id),
        })
    }
}

// ── 1. Strategy cascade integration ──────────────────────────────────────────

#[test]
fn fast_cascade_first_success_terminates() {
    let runner = SequentialRunner;
    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(FailStrategy { priority: 0.0 }),
        Box::new(SolveStrategy {
            priority: 1.0,
            value: 42,
        }),
    ];
    let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);
    assert!(
        matches!(result, StrategyResult::Solved { .. }),
        "first success must terminate cascade"
    );
}

#[test]
fn fast_cascade_priority_order_lower_tried_first() {
    // Lower priority value is tried first (ascending sort).
    let runner = SequentialRunner;
    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(SolveStrategy {
            priority: 2.0,
            value: 100,
        }),
        Box::new(SolveStrategy {
            priority: 0.5,
            value: 7,
        }),
    ];
    let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);
    match result {
        StrategyResult::Solved { expr, .. } => {
            if let Expr::Integer(n) = expr.as_ref() {
                assert_eq!(
                    n.to_i64(),
                    Some(7),
                    "lower priority strategy should be tried first"
                );
            } else {
                panic!("expected integer expr");
            }
        }
        other => panic!("expected Solved, got {:?}", other),
    }
}

#[test]
fn fast_cascade_proven_impossible_terminates() {
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
        "ProvenImpossible must terminate cascade immediately"
    );
}

#[test]
fn fast_cascade_structural_error_terminates() {
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
        "StructuralError must terminate cascade immediately"
    );
}

#[test]
fn fast_cascade_all_fail_returns_not_applicable() {
    let runner = SequentialRunner;
    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(FailStrategy { priority: 0.0 }),
        Box::new(FailStrategy { priority: 1.0 }),
    ];
    let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);
    assert!(
        matches!(result, StrategyResult::Failed(FailureReason::NotApplicable)),
        "all-fail cascade must return NotApplicable"
    );
}

#[test]
fn fast_cascade_partial_saved_when_all_fail() {
    let runner = SequentialRunner;
    // Partial at priority 0.0, then Fail at priority 1.0.
    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(PartialStrategy { priority: 0.0 }),
        Box::new(FailStrategy { priority: 1.0 }),
    ];
    let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);
    assert!(
        matches!(result, StrategyResult::Partial { .. }),
        "best partial must be returned when all strategies fail"
    );
}

// ── 2. Trace tree completeness ────────────────────────────────────────────────

#[test]
fn fast_trace_new_is_empty() {
    let tree = TraceTree::new();
    assert!(tree.nodes.is_empty());
    assert_eq!(tree.step_count(), 0);
}

#[test]
fn fast_trace_push_step_records_in_order() {
    use thales::numeric::trace::{Step, TechniqueTag};
    let mut tree = TraceTree::new();
    tree.push_step(Step::new(TechniqueTag::Simplification, "step1"));
    tree.push_step(Step::new(TechniqueTag::Expansion, "step2"));
    assert_eq!(tree.nodes.len(), 2);
    assert_eq!(tree.step_count(), 2);
}

#[test]
fn fast_trace_branch_node_contains_children() {
    use thales::numeric::trace::{Step, TechniqueTag};
    let mut tree = TraceTree::new();
    let handle = tree.open_branch(BranchReason::StrategyCascade, vec![]);
    let mut child_tree = TraceTree::new();
    child_tree.push_step(Step::new(TechniqueTag::Factoring, "factor"));
    tree.record_branch_outcome(
        handle,
        StrategyId("integration::child"),
        BranchOutcome::Succeeded,
        child_tree,
    );
    match &tree.nodes[0] {
        TraceNode::Branch { reason, children } => {
            assert_eq!(*reason, BranchReason::StrategyCascade);
            assert_eq!(children.len(), 1);
            assert_eq!(children[0].strategy, StrategyId("integration::child"));
            assert_eq!(children[0].outcome, BranchOutcome::Succeeded);
        }
        _ => panic!("expected Branch node"),
    }
}

#[test]
fn fast_trace_join_node_contains_parts() {
    use thales::numeric::trace::{Step, TechniqueTag};
    let mut tree = TraceTree::new();
    let handle = tree.open_join(JoinReason::DivideAndConquer, 2);
    let mut part1 = TraceTree::new();
    part1.push_step(Step::new(TechniqueTag::PowerRule, "pow"));
    let mut part2 = TraceTree::new();
    part2.push_step(Step::new(TechniqueTag::ChainRule, "chain"));
    tree.record_join_part(handle, part1);
    tree.record_join_part(handle, part2);
    match &tree.nodes[0] {
        TraceNode::Join { reason, parts } => {
            assert_eq!(*reason, JoinReason::DivideAndConquer);
            assert_eq!(parts.len(), 2);
        }
        _ => panic!("expected Join node"),
    }
    assert_eq!(tree.step_count(), 2);
}

#[test]
fn fast_trace_cache_hit_preserves_cached_trace() {
    use thales::engine::{CacheSource, PatternHash};
    use thales::numeric::trace::{Step, TechniqueTag};
    let cached_inner = TraceNode::Step(Step::new(TechniqueTag::USubstitution, "usub"));
    let mut tree = TraceTree::new();
    tree.nodes.push(TraceNode::CacheHit {
        source: CacheSource::KnowledgeCache,
        pattern_hash: PatternHash(0xabcd),
        cached_trace: Box::new(cached_inner),
    });
    match &tree.nodes[0] {
        TraceNode::CacheHit {
            source,
            pattern_hash,
            cached_trace,
        } => {
            assert_eq!(*source, CacheSource::KnowledgeCache);
            assert_eq!(*pattern_hash, PatternHash(0xabcd));
            assert!(matches!(cached_trace.as_ref(), TraceNode::Step(_)));
        }
        _ => panic!("expected CacheHit node"),
    }
}

#[test]
fn fast_trace_flatten_steps_includes_branch_and_join_steps() {
    use thales::numeric::trace::{Step, TechniqueTag};
    let mut tree = TraceTree::new();
    // Top-level step
    tree.push_step(Step::new(TechniqueTag::Simplification, "top"));
    // Branch with two children, each with a step
    let bhandle = tree.open_branch(BranchReason::AlternativeStrategies, vec![]);
    let mut branch1 = TraceTree::new();
    branch1.push_step(Step::new(TechniqueTag::Factoring, "b1"));
    let mut branch2 = TraceTree::new();
    branch2.push_step(Step::new(TechniqueTag::Expansion, "b2"));
    tree.record_branch_outcome(
        bhandle,
        StrategyId("s1"),
        BranchOutcome::Failed(FailureReason::NotApplicable),
        branch1,
    );
    tree.record_branch_outcome(bhandle, StrategyId("s2"), BranchOutcome::Succeeded, branch2);
    // Join with one part containing a step
    let jhandle = tree.open_join(JoinReason::PartialFractionMerge, 1);
    let mut jpart = TraceTree::new();
    jpart.push_step(Step::new(TechniqueTag::PowerRule, "j1"));
    tree.record_join_part(jhandle, jpart);

    let steps = tree.flatten_steps();
    // top + b1 + b2 + j1 = 4 steps
    assert_eq!(
        steps.len(),
        4,
        "flatten_steps must include dead-end branches and join parts"
    );
}

#[test]
fn fast_trace_flatten_cache_hit_includes_cached_steps() {
    use thales::engine::{CacheSource, PatternHash};
    use thales::numeric::trace::{Step, TechniqueTag};
    let cached = TraceNode::Step(Step::new(TechniqueTag::IntegrationByParts, "cached"));
    let mut tree = TraceTree::new();
    tree.push_step(Step::new(TechniqueTag::Simplification, "live"));
    tree.nodes.push(TraceNode::CacheHit {
        source: CacheSource::SolveCache,
        pattern_hash: PatternHash(1),
        cached_trace: Box::new(cached),
    });
    // live step + cached step = 2
    assert_eq!(tree.step_count(), 2);
}

// ── 3. Execution mode dispatch ────────────────────────────────────────────────

#[test]
fn fast_mode_sequential_finds_first_success() {
    let runner = SequentialRunner;
    let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(SolveStrategy {
        priority: 0.0,
        value: 5,
    })];
    let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);
    assert!(matches!(result, StrategyResult::Solved { .. }));
}

#[test]
fn fast_mode_tree_search_depth_zero_returns_no_closed_form() {
    let runner = SequentialRunner;
    let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(SolveStrategy {
        priority: 0.0,
        value: 1,
    })];
    let result = runner.run(
        base_ctx(),
        &strategies,
        ExecutionMode::TreeSearch {
            max_depth: 0,
            comparison: TreeComparison::FirstSuccess,
        },
    );
    assert!(
        matches!(result, StrategyResult::Failed(FailureReason::NoClosedForm)),
        "depth 0 tree search must return NoClosedForm"
    );
}

#[test]
fn fast_mode_tree_search_positive_depth_finds_solution() {
    let runner = SequentialRunner;
    let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(SolveStrategy {
        priority: 0.0,
        value: 11,
    })];
    let result = runner.run(
        base_ctx(),
        &strategies,
        ExecutionMode::TreeSearch {
            max_depth: 3,
            comparison: TreeComparison::FirstSuccess,
        },
    );
    assert!(
        matches!(result, StrategyResult::Solved { .. }),
        "positive depth tree search with a solve strategy must find solution"
    );
}

#[test]
fn fast_mode_divide_and_conquer_with_decompose_merges() {
    let runner = SequentialRunner;
    // DecomposeStrategy produces two sub-problems solved by SolveStrategy.
    let strategies: Vec<Box<dyn Strategy>> = vec![
        Box::new(DecomposeStrategy),
        Box::new(SolveStrategy {
            priority: 1.0,
            value: 42,
        }),
    ];
    let result = runner.run(
        base_ctx(),
        &strategies,
        ExecutionMode::DivideAndConquer { max_depth: 2 },
    );
    // The decompose merger returns int_expr(99) on success.
    assert!(
        matches!(result, StrategyResult::Solved { .. }),
        "divide-and-conquer with decompose strategy must merge and return Solved"
    );
}

#[test]
fn fast_mode_divide_and_conquer_depth_zero_skips_decompose() {
    let runner = SequentialRunner;
    // At depth 0, Decompose is skipped. No other strategy → NotApplicable.
    let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(DecomposeStrategy)];
    let result = runner.run(
        base_ctx(),
        &strategies,
        ExecutionMode::DivideAndConquer { max_depth: 0 },
    );
    assert!(
        matches!(result, StrategyResult::Failed(FailureReason::NotApplicable)),
        "depth 0 divide-and-conquer must skip Decompose and return NotApplicable"
    );
}

// ── 4. Fallback integration ───────────────────────────────────────────────────

#[test]
fn fast_fallback_disabled_cascade_failure_returned_as_is() {
    let runner = SequentialRunner;
    // FallbackConfig is disabled by default.
    let ctx = base_ctx();
    assert!(
        !ctx.fallback.numerical,
        "default context has fallback disabled"
    );
    let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(FailStrategy { priority: 0.0 })];
    let result = runner.run(ctx, &strategies, ExecutionMode::Sequential);
    assert!(
        matches!(result, StrategyResult::Failed(FailureReason::NotApplicable)),
        "disabled fallback must return cascade failure unchanged"
    );
}

#[test]
fn fast_fallback_enabled_cascade_success_no_fallback() {
    let runner = SequentialRunner;
    let ctx = base_ctx().with_fallback(FallbackConfig::enabled());
    let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(SolveStrategy {
        priority: 0.0,
        value: 77,
    })];
    let result = runner.run(ctx, &strategies, ExecutionMode::Sequential);
    match result {
        StrategyResult::Solved { expr, .. } => {
            if let Expr::Integer(n) = expr.as_ref() {
                assert_eq!(
                    n.to_i64(),
                    Some(77),
                    "cascade success must be returned unchanged"
                );
            } else {
                panic!("expected integer");
            }
        }
        other => panic!("expected Solved from cascade, got {:?}", other),
    }
}

#[test]
fn fast_fallback_empty_registry_returns_none() {
    let reg = NumericalEvaluatorRegistry::new();
    let ctx = base_ctx().with_fallback(FallbackConfig::enabled());
    let trigger = FallbackTrigger::StrategyExhaustion {
        strategies_attempted: 0,
        last_reason: FailureReason::NotApplicable,
    };
    let result = FallbackRunner::run_with_registry(&ctx, trigger, &reg);
    assert!(result.is_none(), "empty registry must return None");
}

#[test]
fn fast_fallback_success_evaluator_produces_solved() {
    let reg = NumericalEvaluatorRegistry::new();
    reg.register(Arc::new(AlwaysSuccessEvaluator {
        id: "integration::success_ev",
    }));
    let ctx = base_ctx().with_fallback(FallbackConfig::enabled());
    let trigger = FallbackTrigger::StrategyExhaustion {
        strategies_attempted: 1,
        last_reason: FailureReason::NotApplicable,
    };
    let result = FallbackRunner::run_with_registry(&ctx, trigger, &reg);
    assert!(result.is_some(), "success evaluator must produce a result");
    assert!(matches!(result.unwrap(), StrategyResult::Solved { .. }));
}

#[test]
fn fast_fallback_proven_impossible_not_overridden_by_fallback() {
    let runner = SequentialRunner;
    let ctx = base_ctx().with_fallback(FallbackConfig::enabled());
    let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(ImpossibleStrategy)];
    let result = runner.run(ctx, &strategies, ExecutionMode::Sequential);
    assert!(
        matches!(result, StrategyResult::ProvenImpossible { .. }),
        "ProvenImpossible must not be overridden even when fallback is enabled"
    );
}
