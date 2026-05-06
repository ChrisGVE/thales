//! Cache-aware strategy runner wrapper.
//!
//! [`CacheAwareRunner`] wraps any [`StrategyRunner`] with a two-tier cache
//! lookup before dispatch and a cache insertion after a successful or
//! proven-impossible result. Positive and negative caches are checked in
//! sequence; on miss the inner runner is called and its result is inserted.

use std::sync::Arc;

use crate::engine::assumption_key::sign_with_varmap;
use crate::engine::cache::entry::CacheSource;
use crate::engine::cache::entry::{NegativeCacheEntry, PositiveCacheEntry};
use crate::engine::cache::knowledge::{CacheLookup, KnowledgeCache};
use crate::engine::canonicalize::{canonicalize, pattern_hash};
use crate::engine::context::SolveContext;
use crate::engine::mode::ExecutionMode;
use crate::engine::runner::StrategyRunner;
use crate::engine::strategy::{Strategy, StrategyResult};
use crate::engine::trace_tree::{BranchReason, TraceNode};

// ── CacheAwareRunner ──────────────────────────────────────────────────────────

/// A runner that wraps any [`StrategyRunner`] with cache lookup before
/// dispatch and cache insertion after a successful result.
///
/// On every call to [`run`][StrategyRunner::run]:
///
/// 1. Canonicalize the input expression and hash the pattern.
/// 2. Check the positive cache; on hit return `Solved` with a `CacheHit`
///    trace wrapping the originally recorded trace.
/// 3. Check the negative cache; on hit return `ProvenImpossible` with a
///    `CacheHit` trace.
/// 4. On miss, delegate to the inner runner.
/// 5. Insert the result into the appropriate cache tier.
pub struct CacheAwareRunner<R: StrategyRunner> {
    inner: R,
    knowledge: Arc<KnowledgeCache>,
}

impl<R: StrategyRunner> CacheAwareRunner<R> {
    /// Wrap `inner` with the given knowledge cache.
    pub fn new(inner: R, knowledge: Arc<KnowledgeCache>) -> Self {
        Self { inner, knowledge }
    }

    /// Access the underlying knowledge cache.
    pub fn knowledge(&self) -> &Arc<KnowledgeCache> {
        &self.knowledge
    }
}

impl<R: StrategyRunner> StrategyRunner for CacheAwareRunner<R> {
    fn run(
        &self,
        ctx: SolveContext,
        strategies: &[Box<dyn Strategy>],
        mode: ExecutionMode,
    ) -> StrategyResult {
        let (pattern, var_map) = canonicalize(&ctx.expr);
        let hash = pattern_hash(&pattern);
        let sig = sign_with_varmap(&ctx.assumptions, &var_map);

        // Positive cache check.
        match self.knowledge.lookup_positive(hash, &pattern, &sig) {
            CacheLookup::PositiveHit(entry) => {
                let cache_trace = TraceNode::CacheHit {
                    source: CacheSource::KnowledgeCache,
                    pattern_hash: hash,
                    cached_trace: Box::new(entry.trace.clone()),
                };
                return StrategyResult::Solved {
                    expr: entry.solution.clone(),
                    trace: cache_trace,
                };
            }
            CacheLookup::NegativeHit(_) | CacheLookup::Miss => {}
        }

        // Negative cache check.
        match self.knowledge.lookup_negative(hash, &pattern, &sig) {
            CacheLookup::NegativeHit(entry) => {
                let cache_trace = TraceNode::CacheHit {
                    source: CacheSource::KnowledgeCache,
                    pattern_hash: hash,
                    cached_trace: Box::new(TraceNode::Branch {
                        reason: BranchReason::StrategyCascade,
                        children: vec![],
                    }),
                };
                return StrategyResult::ProvenImpossible {
                    certificate: entry.proof.clone(),
                    trace: cache_trace,
                };
            }
            CacheLookup::PositiveHit(_) | CacheLookup::Miss => {}
        }

        // Cache miss — delegate to inner runner.
        let result = self.inner.run(ctx.fork(), strategies, mode);

        // Insert result into cache on definite outcomes.
        match &result {
            StrategyResult::Solved { expr, trace } => {
                self.knowledge.insert_positive(
                    hash,
                    PositiveCacheEntry {
                        canonical_pattern: pattern,
                        assumption_sig: sig,
                        var_map,
                        solution: Arc::clone(expr),
                        trace: trace.clone(),
                        strategy_id: None,
                    },
                );
            }
            StrategyResult::ProvenImpossible { certificate, .. } => {
                self.knowledge.insert_negative(
                    hash,
                    NegativeCacheEntry {
                        canonical_pattern: pattern,
                        assumption_sig: sig,
                        proof: certificate.clone(),
                    },
                );
            }
            _ => {}
        }

        result
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::assumption_key::sign_with_varmap;
    use crate::engine::cache::entry::{NegativeCacheEntry, PositiveCacheEntry};
    use crate::engine::cache::knowledge::KnowledgeCache;
    use crate::engine::canonicalize::{canonicalize, pattern_hash};
    use crate::engine::context::SolveContext;
    use crate::engine::mode::ExecutionMode;
    use crate::engine::reason::{FailureReason, ImpossibilityProof};
    use crate::engine::resource::ResourceBudget;
    use crate::engine::runner::SequentialRunner;
    use crate::engine::strategy::{Strategy, StrategyResult};
    use crate::engine::trace_tree::{BranchReason, StrategyId, TraceNode};
    use crate::numeric::{Expr, SmallInt};
    use std::sync::Arc;

    // ── Helpers ───────────────────────────────────────────────────────────────

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

    #[derive(Debug)]
    struct SolveStrategy {
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
    struct FailStrategy;

    impl Strategy for FailStrategy {
        fn id(&self) -> StrategyId {
            StrategyId("mock::fail")
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

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// Empty cache → inner runner is called and its result is returned.
    #[test]
    fn fast_cache_runner_miss_delegates() {
        let knowledge = Arc::new(KnowledgeCache::new());
        let runner = CacheAwareRunner::new(SequentialRunner, Arc::clone(&knowledge));
        let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(SolveStrategy { value: 7 })];

        let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);

        match result {
            StrategyResult::Solved { expr, .. } => {
                if let Expr::Integer(n) = &*expr {
                    assert_eq!(n.to_i64(), Some(7));
                } else {
                    panic!("Expected integer 7");
                }
            }
            other => panic!("Expected Solved, got {:?}", other),
        }
        // Stats: one lookup miss, no hit.
        let stats = knowledge.stats();
        assert_eq!(stats.positive_hits, 0);
    }

    /// Insert a positive entry; lookup hits and returns Solved with CacheHit trace.
    #[test]
    fn fast_cache_runner_positive_hit() {
        let knowledge = Arc::new(KnowledgeCache::new());
        let expr = int_expr(42);
        let (pattern, var_map) = canonicalize(&expr);
        let hash = pattern_hash(&pattern);
        let sig = sign_with_varmap(
            &crate::engine::assumption::AssumptionSet::default(),
            &var_map,
        );

        knowledge.insert_positive(
            hash,
            PositiveCacheEntry {
                canonical_pattern: pattern,
                assumption_sig: sig,
                var_map,
                solution: int_expr(42),
                trace: dummy_trace(),
                strategy_id: None,
            },
        );

        let runner = CacheAwareRunner::new(SequentialRunner, Arc::clone(&knowledge));
        // Use an empty strategy list so inner runner would return NotApplicable if reached.
        let strategies: Vec<Box<dyn Strategy>> = vec![];
        let ctx = SolveContext::new(int_expr(42), ResourceBudget::unlimited());

        let result = runner.run(ctx, &strategies, ExecutionMode::Sequential);

        match result {
            StrategyResult::Solved { trace, .. } => {
                assert!(
                    matches!(trace, TraceNode::CacheHit { .. }),
                    "Expected CacheHit trace on positive hit"
                );
            }
            other => panic!("Expected Solved from cache hit, got {:?}", other),
        }
        assert_eq!(knowledge.stats().positive_hits, 1);
    }

    /// Insert a negative entry; lookup hits and returns ProvenImpossible.
    #[test]
    fn fast_cache_runner_negative_hit() {
        let knowledge = Arc::new(KnowledgeCache::new());
        let expr = int_expr(99);
        let (pattern, var_map) = canonicalize(&expr);
        let hash = pattern_hash(&pattern);
        let sig = sign_with_varmap(
            &crate::engine::assumption::AssumptionSet::default(),
            &var_map,
        );

        knowledge.insert_negative(
            hash,
            NegativeCacheEntry {
                canonical_pattern: pattern,
                assumption_sig: sig,
                proof: ImpossibilityProof::NoElementaryClosure,
            },
        );

        let runner = CacheAwareRunner::new(SequentialRunner, Arc::clone(&knowledge));
        let strategies: Vec<Box<dyn Strategy>> = vec![];
        let ctx = SolveContext::new(int_expr(99), ResourceBudget::unlimited());

        let result = runner.run(ctx, &strategies, ExecutionMode::Sequential);

        match result {
            StrategyResult::ProvenImpossible { certificate, trace } => {
                assert_eq!(certificate, ImpossibilityProof::NoElementaryClosure);
                assert!(
                    matches!(trace, TraceNode::CacheHit { .. }),
                    "Expected CacheHit trace on negative hit"
                );
            }
            other => panic!("Expected ProvenImpossible, got {:?}", other),
        }
    }

    /// Solve via inner runner → entry appears in knowledge cache.
    #[test]
    fn fast_cache_runner_inserts_on_solve() {
        let knowledge = Arc::new(KnowledgeCache::new());
        let runner = CacheAwareRunner::new(SequentialRunner, Arc::clone(&knowledge));
        let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(SolveStrategy { value: 5 })];

        assert_eq!(knowledge.positive_count(), 0);
        let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);
        assert!(matches!(result, StrategyResult::Solved { .. }));
        assert_eq!(
            knowledge.positive_count(),
            1,
            "Expected one positive entry inserted"
        );
    }

    /// ProvenImpossible via inner runner → negative entry inserted.
    #[test]
    fn fast_cache_runner_inserts_negative_on_impossible() {
        let knowledge = Arc::new(KnowledgeCache::new());
        let runner = CacheAwareRunner::new(SequentialRunner, Arc::clone(&knowledge));
        let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(ImpossibleStrategy)];

        assert_eq!(knowledge.negative_count(), 0);
        let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);
        assert!(matches!(result, StrategyResult::ProvenImpossible { .. }));
        assert_eq!(
            knowledge.negative_count(),
            1,
            "Expected one negative entry inserted"
        );
    }
}
