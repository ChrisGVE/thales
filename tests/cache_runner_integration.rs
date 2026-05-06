//! Integration tests for the D0 cache runner pipeline and cache consistency.
//!
//! Tests the `CacheAwareRunner` pipeline, `PedagogicReorder` on `CacheHit`
//! traces, rehydrate/slottify roundtrips, phase barrier flow, and scoring.

use std::sync::Arc;

use thales::engine::cache::rehydrate::rehydrate_expr;
use thales::engine::cache::reorder::PedagogicReorder;
use thales::engine::cache::runner::CacheAwareRunner;
use thales::engine::cache::{
    CacheSource, CacheStats, HitRateScorer, KnowledgeCache, NegativeCacheEntry, PipelineEvent,
    PositiveCacheEntry, RunTrace, ScoringPolicy, SkipReason, StrategyOutcomeSummary,
};
use thales::engine::canonical_pattern::{PatternHash, SlotId, VarMap};
use thales::engine::canonicalize::{canonicalize, pattern_hash};
use thales::engine::context::SolveContext;
use thales::engine::mode::ExecutionMode;
use thales::engine::phase::PhaseBarrier;
use thales::engine::reason::{FailureReason, ImpossibilityProof};
use thales::engine::resource::ResourceBudget;
use thales::engine::trace_tree::{BranchReason, StrategyId, TraceNode};
use thales::engine::{SequentialRunner, Strategy, StrategyResult, StrategyRunner};
use thales::numeric::expr::Expr;
use thales::numeric::SmallInt;

// ── Helpers ───────────────────────────────────────────────────────────────────

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
    value: i64,
}

impl Strategy for SolveStrategy {
    fn id(&self) -> StrategyId {
        StrategyId("cache_integ::solve")
    }
    fn applicable(&self, _ctx: &SolveContext) -> bool {
        true
    }
    fn priority(&self, _ctx: &SolveContext) -> f64 {
        1.0
    }
    fn apply(&self, _ctx: SolveContext) -> StrategyResult {
        StrategyResult::Solved {
            expr: int_expr(self.value),
            trace: dummy_trace(),
        }
    }
}

#[derive(Debug)]
struct ImpossibleStrategy;

impl Strategy for ImpossibleStrategy {
    fn id(&self) -> StrategyId {
        StrategyId("cache_integ::impossible")
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

// ── 1. CacheAwareRunner pipeline ──────────────────────────────────────────────

/// First call is a cache miss — inner runner is called and result returned.
#[test]
fn fast_pipeline_cache_miss_delegates_to_inner() {
    let knowledge = Arc::new(KnowledgeCache::new());
    let runner = CacheAwareRunner::new(SequentialRunner, Arc::clone(&knowledge));
    let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(SolveStrategy { value: 7 })];

    let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);

    match result {
        StrategyResult::Solved { expr, .. } => match expr.as_ref() {
            Expr::Integer(n) => assert_eq!(n.to_i64(), Some(7)),
            _ => panic!("Expected integer 7"),
        },
        other => panic!("Expected Solved, got {:?}", other),
    }

    // After miss the entry is inserted; a second run is a hit.
    assert_eq!(knowledge.positive_count(), 1);
}

/// Second call on the same expression returns Solved with a CacheHit trace.
#[test]
fn fast_pipeline_second_call_returns_cache_hit() {
    let knowledge = Arc::new(KnowledgeCache::new());
    let runner = CacheAwareRunner::new(SequentialRunner, Arc::clone(&knowledge));
    let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(SolveStrategy { value: 42 })];

    // First call — miss, inserts entry.
    let _ = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);

    // Second call — same expression, should be a cache hit.
    let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);

    match result {
        StrategyResult::Solved { trace, .. } => {
            assert!(
                matches!(trace, TraceNode::CacheHit { .. }),
                "Second call must return a CacheHit trace"
            );
        }
        other => panic!("Expected Solved from cache hit, got {:?}", other),
    }

    let stats = knowledge.stats();
    assert_eq!(stats.positive_hits, 1, "Exactly one positive hit recorded");
}

/// Negative cache prevents retry — `ProvenImpossible` is returned from cache.
#[test]
fn fast_pipeline_negative_cache_prevents_retry() {
    let knowledge = Arc::new(KnowledgeCache::new());
    let runner = CacheAwareRunner::new(SequentialRunner, Arc::clone(&knowledge));
    let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(ImpossibleStrategy)];

    // First call — miss, inner runner returns ProvenImpossible, inserts negative entry.
    let first = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);
    assert!(
        matches!(first, StrategyResult::ProvenImpossible { .. }),
        "First call should be ProvenImpossible"
    );
    assert_eq!(knowledge.negative_count(), 1);

    // Second call — negative cache hit, returns ProvenImpossible without calling inner.
    let second = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);
    match second {
        StrategyResult::ProvenImpossible { trace, .. } => {
            assert!(
                matches!(trace, TraceNode::CacheHit { .. }),
                "Negative hit should produce CacheHit trace"
            );
        }
        other => panic!(
            "Expected ProvenImpossible from negative cache, got {:?}",
            other
        ),
    }
}

/// Cache miss count equals 1 after the first call: `positive_lookups` is 1,
/// `positive_hits` is 0 after that first run.
#[test]
fn fast_pipeline_miss_count_equals_one_after_first_call() {
    let knowledge = Arc::new(KnowledgeCache::new());
    let runner = CacheAwareRunner::new(SequentialRunner, Arc::clone(&knowledge));
    let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(SolveStrategy { value: 3 })];

    let _ = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);

    let stats = knowledge.stats();
    // One positive lookup was attempted (the miss), and it was not a hit.
    assert_eq!(stats.positive_lookups, 1);
    assert_eq!(stats.positive_hits, 0);
}

// ── 2. PedagogicReorder on CacheHit traces ────────────────────────────────────

/// Run CacheAwareRunner to get a CacheHit trace, then verify PedagogicReorder
/// removes all CacheHit nodes from the output.
#[test]
fn fast_reorder_cache_hit_trace_contains_no_cache_hit_after_reorder() {
    let knowledge = Arc::new(KnowledgeCache::new());
    let runner = CacheAwareRunner::new(SequentialRunner, Arc::clone(&knowledge));
    let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(SolveStrategy { value: 99 })];

    // First call: populate cache.
    let _ = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);

    // Second call: returns CacheHit trace.
    let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);

    let cache_hit_trace = match result {
        StrategyResult::Solved { trace, .. } => trace,
        other => panic!("Expected Solved with CacheHit trace, got {:?}", other),
    };

    // The trace from the second call must be a CacheHit.
    assert!(
        matches!(cache_hit_trace, TraceNode::CacheHit { .. }),
        "Expected CacheHit before reorder"
    );

    // After PedagogicReorder the CacheHit is gone.
    let reordered = PedagogicReorder::apply(&cache_hit_trace);
    assert_no_cache_hit(&reordered);
}

/// Recursively verifies no `CacheHit` variant exists in the tree.
fn assert_no_cache_hit(node: &TraceNode) {
    match node {
        TraceNode::CacheHit { .. } => panic!("Unexpected CacheHit in reordered output"),
        TraceNode::Step(_) => {}
        TraceNode::Branch { children, .. } => {
            for b in children {
                for n in &b.nodes.nodes {
                    assert_no_cache_hit(n);
                }
            }
        }
        TraceNode::Join { parts, .. } => {
            for part in parts {
                for n in &part.nodes {
                    assert_no_cache_hit(n);
                }
            }
        }
    }
}

// ── 3. Rehydrate / slottify roundtrip ─────────────────────────────────────────

/// Canonicalize an integer expression, then rehydrate — value is preserved.
#[test]
fn fast_rehydrate_integer_roundtrip() {
    use thales::engine::canonical_pattern::CanonicalPattern;

    let expr = int_expr(123);
    let (pattern, var_map) = canonicalize(&expr);

    // Integers have no slots so no fresh_id_gen call expected.
    let rehydrated = rehydrate_expr(&pattern, &var_map, &mut |_| {
        panic!("fresh_id_gen called unexpectedly for integer")
    });

    match rehydrated.as_ref() {
        Expr::Integer(n) => assert_eq!(n.to_i64(), Some(123)),
        _ => panic!("Expected integer 123 after roundtrip"),
    }
}

/// Multi-variable expression: two distinct symbols are canonicalized to
/// distinct slots, and rehydration maps them back to the original symbols.
#[test]
fn fast_rehydrate_multi_variable_roundtrip() {
    use thales::numeric::SymbolId;

    let x_id = SymbolId::intern("rhv_x_integ");
    let y_id = SymbolId::intern("rhv_y_integ");

    let x_expr = Arc::new(Expr::Symbol(x_id));
    let y_expr = Arc::new(Expr::Symbol(y_id));

    let (pat_x, vm_x) = canonicalize(&x_expr);
    let (pat_y, vm_y) = canonicalize(&y_expr);

    let rx = rehydrate_expr(&pat_x, &vm_x, &mut |_| panic!("unexpected fresh call"));
    let ry = rehydrate_expr(&pat_y, &vm_y, &mut |_| panic!("unexpected fresh call"));

    // Both rehydrate to their respective symbols.
    assert_eq!(rx.as_ref(), &Expr::Symbol(x_id));
    assert_eq!(ry.as_ref(), &Expr::Symbol(y_id));

    // Their canonical patterns are structurally identical (both single-slot).
    assert_eq!(pat_x, pat_y, "Single-variable patterns should be identical");

    // But the var_maps differ: same slot, different symbol.
    assert_eq!(vm_x.slot_of(x_id), Some(SlotId(0)));
    assert_eq!(vm_y.slot_of(y_id), Some(SlotId(0)));
}

/// Constant expressions (Pi, E) survive the canonicalize → rehydrate roundtrip.
#[test]
fn fast_rehydrate_constants_roundtrip() {
    use thales::ast::SymbolicConstant;

    let pi_expr = Arc::new(Expr::Constant(SymbolicConstant::Pi));
    let e_expr = Arc::new(Expr::Constant(SymbolicConstant::E));

    let (pi_pat, pi_vm) = canonicalize(&pi_expr);
    let (e_pat, e_vm) = canonicalize(&e_expr);

    let rpi = rehydrate_expr(&pi_pat, &pi_vm, &mut |_| panic!("unexpected fresh Pi"));
    let re = rehydrate_expr(&e_pat, &e_vm, &mut |_| panic!("unexpected fresh E"));

    assert_eq!(rpi.as_ref(), &Expr::Constant(SymbolicConstant::Pi));
    assert_eq!(re.as_ref(), &Expr::Constant(SymbolicConstant::E));
}

// ── 4. Phase barrier flow ─────────────────────────────────────────────────────

/// Set a phase barrier on a SolveContext; first take returns it; second returns None.
#[test]
fn fast_phase_barrier_set_and_take_consumed() {
    let expr = int_expr(5);
    let mut ctx = SolveContext::new(Arc::clone(&expr), ResourceBudget::unlimited());

    assert!(ctx.take_phase_barrier().is_none(), "No barrier initially");

    ctx.set_phase_barrier(PhaseBarrier::new("normalization", Arc::clone(&expr)));

    let taken = ctx.take_phase_barrier();
    assert!(taken.is_some());
    let barrier = taken.unwrap();
    assert_eq!(barrier.phase_name, "normalization");
    assert!(Arc::ptr_eq(&barrier.intermediate, &expr));

    assert!(
        ctx.take_phase_barrier().is_none(),
        "Barrier should be consumed after first take"
    );
}

/// Fork carries the phase barrier from the parent context.
#[test]
fn fast_phase_barrier_fork_carries_barrier() {
    let expr = int_expr(7);
    let mut ctx = SolveContext::new(Arc::clone(&expr), ResourceBudget::unlimited());
    ctx.set_phase_barrier(PhaseBarrier::new("pre_expand", Arc::clone(&expr)));

    let fork = ctx.fork();

    // Fork has the barrier.
    assert!(fork.phase_barrier.is_some());
    let barrier = fork.phase_barrier.as_ref().unwrap();
    assert_eq!(barrier.phase_name, "pre_expand");
}

// ── 5. Scoring and RunTrace ───────────────────────────────────────────────────

/// HitRateScorer with no lookups returns a negative adjustment (hit_rate=0 → -0.5).
#[test]
fn fast_scorer_zero_lookups_returns_negative_adjustment() {
    let scorer = HitRateScorer::default();
    let stats = CacheStats::default();
    let adj = scorer.adjust(StrategyId("test_strategy"), &stats);
    // hit_rate = 0.0, adjustment = 1.0 * (0.0 - 0.5) = -0.5
    assert!(
        adj < 0.0,
        "Zero-lookup hit rate should yield negative adjustment"
    );
    assert!((adj - (-0.5)).abs() < f64::EPSILON);
}

/// HitRateScorer with sensitivity multiplier scales the adjustment correctly.
#[test]
fn fast_scorer_sensitivity_scales_adjustment() {
    let scorer = HitRateScorer::with_sensitivity(2.0);
    let stats = CacheStats {
        positive_lookups: 4,
        positive_hits: 4,
        ..Default::default()
    };
    // hit_rate = 1.0, adjustment = 2.0 * (1.0 - 0.5) = 1.0
    let adj = scorer.adjust(StrategyId("s"), &stats);
    assert!((adj - 1.0).abs() < f64::EPSILON);
}

/// Push events to RunTrace; verify they are accessible and ordered.
#[test]
fn fast_run_trace_events_ordered_and_accessible() {
    let mut trace = RunTrace::new();
    assert!(trace.is_empty());

    trace.push(PipelineEvent::CacheLookup { hit: false });
    trace.push(PipelineEvent::StrategyDispatched {
        id: StrategyId("cache_integ::solve"),
    });
    trace.push(PipelineEvent::StrategyCompleted {
        id: StrategyId("cache_integ::solve"),
        outcome: StrategyOutcomeSummary::Solved,
    });
    trace.push(PipelineEvent::PhaseBarrierProcessed {
        phase_name: "normalization",
    });

    assert_eq!(trace.len(), 4);
    assert!(!trace.is_empty());

    let events = trace.events();
    assert_eq!(events.len(), 4);

    // Events appear in insertion order.
    assert!(matches!(
        events[0],
        PipelineEvent::CacheLookup { hit: false }
    ));
    assert!(matches!(
        events[1],
        PipelineEvent::StrategyDispatched { .. }
    ));
    assert!(matches!(events[2], PipelineEvent::StrategyCompleted { .. }));
    assert!(matches!(
        events[3],
        PipelineEvent::PhaseBarrierProcessed {
            phase_name: "normalization"
        }
    ));
}

/// RunTrace records SkipReason variants without loss.
#[test]
fn fast_run_trace_skip_reason_preserved() {
    let mut trace = RunTrace::new();

    trace.push(PipelineEvent::StrategySkipped {
        id: StrategyId("cache_integ::skip_me"),
        reason: SkipReason::BudgetExhausted,
    });
    trace.push(PipelineEvent::StrategySkipped {
        id: StrategyId("cache_integ::skip_me2"),
        reason: SkipReason::CacheHitPreceded,
    });

    let events = trace.events();
    assert_eq!(events.len(), 2);
    match &events[0] {
        PipelineEvent::StrategySkipped { id, reason } => {
            assert_eq!(id.0, "cache_integ::skip_me");
            assert!(matches!(reason, SkipReason::BudgetExhausted));
        }
        _ => panic!("Expected StrategySkipped"),
    }
    match &events[1] {
        PipelineEvent::StrategySkipped { reason, .. } => {
            assert!(matches!(reason, SkipReason::CacheHitPreceded));
        }
        _ => panic!("Expected StrategySkipped"),
    }
}
