//! Cache entry types for D0 structural memoization.
//!
//! [`PositiveCacheEntry`] records a successful solve result alongside the
//! canonical pattern and assumption signature under which it was found.
//! [`NegativeCacheEntry`] records a proven impossibility. Both are stored in
//! [`CacheEntry`] for uniform handling.

use std::sync::Arc;

use crate::engine::assumption_key::AssumptionSignature;
use crate::engine::canonical_pattern::{CanonicalPattern, VarMap};
use crate::engine::reason::ImpossibilityProof;
use crate::engine::trace_tree::{StrategyId, TraceNode};
use crate::numeric::Expr;

// ── CacheSource ───────────────────────────────────────────────────────────────

/// Which cache tier a hit came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheSource {
    /// Hit from the per-solve transient cache.
    SolveCache,
    /// Hit from the persistent cross-solve knowledge cache.
    KnowledgeCache,
}

// ── PositiveCacheEntry ────────────────────────────────────────────────────────

/// A successful solve result stored in the cache.
#[derive(Debug, Clone)]
pub struct PositiveCacheEntry {
    /// Canonical (variable-renamed) pattern of the input expression.
    pub canonical_pattern: CanonicalPattern,
    /// Assumption signature at the time the result was computed.
    pub assumption_sig: AssumptionSignature,
    /// Mapping from slot IDs back to original symbol IDs.
    pub var_map: VarMap,
    /// The solved expression.
    pub solution: Arc<Expr>,
    /// The trace node recording how the solution was reached.
    pub trace: TraceNode,
    /// Which strategy produced this entry, if known.
    pub strategy_id: Option<StrategyId>,
}

// ── NegativeCacheEntry ────────────────────────────────────────────────────────

/// A proven-impossible result stored in the cache.
#[derive(Debug, Clone)]
pub struct NegativeCacheEntry {
    /// Canonical (variable-renamed) pattern of the input expression.
    pub canonical_pattern: CanonicalPattern,
    /// Assumption signature at the time the proof was computed.
    pub assumption_sig: AssumptionSignature,
    /// Formal certificate of impossibility.
    pub proof: ImpossibilityProof,
}

// ── CacheEntry ────────────────────────────────────────────────────────────────

/// A cache entry — either a solved result or a proven impossibility.
#[derive(Debug, Clone)]
pub enum CacheEntry {
    /// A successful solve result.
    Positive(PositiveCacheEntry),
    /// A proven impossibility.
    Negative(NegativeCacheEntry),
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::canonical_pattern::{CanonicalPattern, VarMap};
    use crate::engine::reason::ImpossibilityProof;
    use crate::engine::trace_tree::{BranchReason, StrategyId, TraceNode};
    use crate::numeric::{Expr, SmallInt};

    fn dummy_trace() -> TraceNode {
        TraceNode::Branch {
            reason: BranchReason::StrategyCascade,
            children: vec![],
        }
    }

    fn int_expr(n: i64) -> Arc<Expr> {
        Arc::new(Expr::Integer(SmallInt::from(n)))
    }

    #[test]
    fn fast_cache_entry_positive_fields_accessible() {
        let entry = PositiveCacheEntry {
            canonical_pattern: CanonicalPattern::Integer(1),
            assumption_sig: AssumptionSignature::default(),
            var_map: VarMap::new(),
            solution: int_expr(42),
            trace: dummy_trace(),
            strategy_id: Some(StrategyId("test::strategy")),
        };
        assert!(matches!(
            entry.canonical_pattern,
            CanonicalPattern::Integer(1)
        ));
        assert_eq!(entry.strategy_id, Some(StrategyId("test::strategy")));
    }

    #[test]
    fn fast_cache_entry_positive_no_strategy() {
        let entry = PositiveCacheEntry {
            canonical_pattern: CanonicalPattern::Integer(0),
            assumption_sig: AssumptionSignature::default(),
            var_map: VarMap::new(),
            solution: int_expr(0),
            trace: dummy_trace(),
            strategy_id: None,
        };
        assert!(entry.strategy_id.is_none());
    }

    #[test]
    fn fast_cache_entry_negative_fields_accessible() {
        let entry = NegativeCacheEntry {
            canonical_pattern: CanonicalPattern::Integer(2),
            assumption_sig: AssumptionSignature::default(),
            proof: ImpossibilityProof::NoElementaryClosure,
        };
        assert_eq!(entry.proof, ImpossibilityProof::NoElementaryClosure);
    }

    #[test]
    fn fast_cache_entry_enum_positive_variant() {
        let pos = PositiveCacheEntry {
            canonical_pattern: CanonicalPattern::Integer(1),
            assumption_sig: AssumptionSignature::default(),
            var_map: VarMap::new(),
            solution: int_expr(1),
            trace: dummy_trace(),
            strategy_id: None,
        };
        let entry = CacheEntry::Positive(pos);
        assert!(matches!(entry, CacheEntry::Positive(_)));
    }

    #[test]
    fn fast_cache_entry_enum_negative_variant() {
        let neg = NegativeCacheEntry {
            canonical_pattern: CanonicalPattern::Integer(3),
            assumption_sig: AssumptionSignature::default(),
            proof: ImpossibilityProof::NoLiouvillePrimitive,
        };
        let entry = CacheEntry::Negative(neg);
        assert!(matches!(entry, CacheEntry::Negative(_)));
    }

    #[test]
    fn fast_cache_source_variants_distinct() {
        assert_ne!(CacheSource::SolveCache, CacheSource::KnowledgeCache);
        assert_eq!(CacheSource::SolveCache, CacheSource::SolveCache);
        assert_eq!(CacheSource::KnowledgeCache, CacheSource::KnowledgeCache);
    }
}
