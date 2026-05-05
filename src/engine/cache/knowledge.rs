//! Persistent cross-solve knowledge cache for D0 memoization.
//!
//! [`KnowledgeCache`] stores positive and negative cache entries keyed by
//! [`PatternHash`]. Lookups perform a two-level check: first a hash match,
//! then structural equality on the [`CanonicalPattern`], then assumption
//! entailment via [`entails`].
//!
//! The cache is `Send + Sync` via an internal [`RwLock`]; reads are
//! concurrent, writes are exclusive.

use std::collections::HashMap;
use std::sync::RwLock;

use crate::engine::assumption_key::{entails, AssumptionSignature};
use crate::engine::cache::entry::{NegativeCacheEntry, PositiveCacheEntry};
use crate::engine::cache::stats::CacheStats;
use crate::engine::canonical_pattern::{CanonicalPattern, PatternHash};

// ── CacheLookup ───────────────────────────────────────────────────────────────

/// Result of a cache lookup.
#[derive(Debug)]
pub enum CacheLookup {
    /// A matching positive (solution) entry was found.
    PositiveHit(PositiveCacheEntry),
    /// A matching negative (impossibility) entry was found.
    NegativeHit(NegativeCacheEntry),
    /// No matching entry was found.
    Miss,
}

// ── KnowledgeCacheInner ───────────────────────────────────────────────────────

struct KnowledgeCacheInner {
    positive: HashMap<PatternHash, Vec<PositiveCacheEntry>>,
    negative: HashMap<PatternHash, Vec<NegativeCacheEntry>>,
}

impl KnowledgeCacheInner {
    fn new() -> Self {
        KnowledgeCacheInner {
            positive: HashMap::new(),
            negative: HashMap::new(),
        }
    }
}

// ── KnowledgeCache ────────────────────────────────────────────────────────────

/// Thread-safe persistent cache for D0 memoized results.
///
/// Entries are keyed first by [`PatternHash`] (cheap), then disambiguated
/// by structural equality on [`CanonicalPattern`] and assumption entailment.
pub struct KnowledgeCache {
    inner: RwLock<KnowledgeCacheInner>,
    stats: RwLock<CacheStats>,
}

impl std::fmt::Debug for KnowledgeCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.read().unwrap();
        f.debug_struct("KnowledgeCache")
            .field("positive_buckets", &inner.positive.len())
            .field("negative_buckets", &inner.negative.len())
            .finish()
    }
}

impl KnowledgeCache {
    /// Create a new empty knowledge cache.
    #[must_use]
    pub fn new() -> Self {
        KnowledgeCache {
            inner: RwLock::new(KnowledgeCacheInner::new()),
            stats: RwLock::new(CacheStats::default()),
        }
    }

    /// Look up a positive entry for `(hash, pattern, current_sig)`.
    ///
    /// Returns `PositiveHit` when a structurally matching entry is found
    /// whose cached assumption signature is entailed by `current_sig`.
    #[must_use]
    pub fn lookup_positive(
        &self,
        hash: PatternHash,
        pattern: &CanonicalPattern,
        current_sig: &AssumptionSignature,
    ) -> CacheLookup {
        {
            let mut stats = self.stats.write().unwrap();
            stats.positive_lookups += 1;
        }
        let inner = self.inner.read().unwrap();
        if let Some(bucket) = inner.positive.get(&hash) {
            for entry in bucket {
                if &entry.canonical_pattern == pattern
                    && entails(current_sig, &entry.assumption_sig)
                {
                    let mut stats = self.stats.write().unwrap();
                    stats.positive_hits += 1;
                    if let Some(sid) = &entry.strategy_id {
                        *stats
                            .per_strategy_hits
                            .entry(sid.0.to_string())
                            .or_insert(0) += 1;
                    }
                    return CacheLookup::PositiveHit(entry.clone());
                }
            }
        }
        CacheLookup::Miss
    }

    /// Look up a negative entry for `(hash, pattern, current_sig)`.
    ///
    /// Returns `NegativeHit` when a structurally matching impossibility proof
    /// is found whose cached assumption signature is entailed by `current_sig`.
    #[must_use]
    pub fn lookup_negative(
        &self,
        hash: PatternHash,
        pattern: &CanonicalPattern,
        current_sig: &AssumptionSignature,
    ) -> CacheLookup {
        {
            let mut stats = self.stats.write().unwrap();
            stats.negative_lookups += 1;
        }
        let inner = self.inner.read().unwrap();
        if let Some(bucket) = inner.negative.get(&hash) {
            for entry in bucket {
                if &entry.canonical_pattern == pattern
                    && entails(current_sig, &entry.assumption_sig)
                {
                    let mut stats = self.stats.write().unwrap();
                    stats.negative_hits += 1;
                    return CacheLookup::NegativeHit(entry.clone());
                }
            }
        }
        CacheLookup::Miss
    }

    /// Insert a positive entry. First-write-wins: if an entry with the same
    /// pattern and assumption signature already exists it is not replaced.
    pub fn insert_positive(&self, hash: PatternHash, entry: PositiveCacheEntry) {
        let mut inner = self.inner.write().unwrap();
        let bucket = inner.positive.entry(hash).or_default();
        let already_present = bucket.iter().any(|e| {
            e.canonical_pattern == entry.canonical_pattern
                && e.assumption_sig == entry.assumption_sig
        });
        if !already_present {
            bucket.push(entry);
        }
    }

    /// Insert a negative entry. First-write-wins: duplicate pattern + sig
    /// combinations are silently ignored.
    pub fn insert_negative(&self, hash: PatternHash, entry: NegativeCacheEntry) {
        let mut inner = self.inner.write().unwrap();
        let bucket = inner.negative.entry(hash).or_default();
        let already_present = bucket.iter().any(|e| {
            e.canonical_pattern == entry.canonical_pattern
                && e.assumption_sig == entry.assumption_sig
        });
        if !already_present {
            bucket.push(entry);
        }
    }

    /// Snapshot of current cache statistics.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.stats.read().unwrap().clone()
    }

    /// Total number of positive entries stored.
    #[must_use]
    pub fn positive_count(&self) -> usize {
        self.inner
            .read()
            .unwrap()
            .positive
            .values()
            .map(Vec::len)
            .sum()
    }

    /// Total number of negative entries stored.
    #[must_use]
    pub fn negative_count(&self) -> usize {
        self.inner
            .read()
            .unwrap()
            .negative
            .values()
            .map(Vec::len)
            .sum()
    }
}

impl Default for KnowledgeCache {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::cache::entry::{NegativeCacheEntry, PositiveCacheEntry};
    use crate::engine::canonical_pattern::{CanonicalPattern, PatternHash, VarMap};
    use crate::engine::reason::ImpossibilityProof;
    use crate::engine::trace_tree::{BranchReason, StrategyId, TraceNode};
    use crate::numeric::{Expr, SmallInt};
    use std::sync::Arc;

    fn dummy_trace() -> TraceNode {
        TraceNode::Branch {
            reason: BranchReason::StrategyCascade,
            children: vec![],
        }
    }

    fn int_expr(n: i64) -> Arc<Expr> {
        Arc::new(Expr::Integer(SmallInt::from(n)))
    }

    fn pos_entry(pat: CanonicalPattern) -> PositiveCacheEntry {
        PositiveCacheEntry {
            canonical_pattern: pat,
            assumption_sig: AssumptionSignature::default(),
            var_map: VarMap::new(),
            solution: int_expr(1),
            trace: dummy_trace(),
            strategy_id: Some(StrategyId("test::strat")),
        }
    }

    fn neg_entry(pat: CanonicalPattern) -> NegativeCacheEntry {
        NegativeCacheEntry {
            canonical_pattern: pat,
            assumption_sig: AssumptionSignature::default(),
            proof: ImpossibilityProof::NoElementaryClosure,
        }
    }

    #[test]
    fn fast_knowledge_new_is_empty() {
        let cache = KnowledgeCache::new();
        assert_eq!(cache.positive_count(), 0);
        assert_eq!(cache.negative_count(), 0);
    }

    #[test]
    fn fast_knowledge_insert_positive_increases_count() {
        let cache = KnowledgeCache::new();
        let hash = PatternHash(42);
        cache.insert_positive(hash, pos_entry(CanonicalPattern::Integer(1)));
        assert_eq!(cache.positive_count(), 1);
    }

    #[test]
    fn fast_knowledge_insert_negative_increases_count() {
        let cache = KnowledgeCache::new();
        let hash = PatternHash(99);
        cache.insert_negative(hash, neg_entry(CanonicalPattern::Integer(2)));
        assert_eq!(cache.negative_count(), 1);
    }

    #[test]
    fn fast_knowledge_lookup_positive_hit() {
        let cache = KnowledgeCache::new();
        let hash = PatternHash(1);
        let pat = CanonicalPattern::Integer(7);
        cache.insert_positive(hash, pos_entry(pat.clone()));
        let sig = AssumptionSignature::default();
        match cache.lookup_positive(hash, &pat, &sig) {
            CacheLookup::PositiveHit(_) => {}
            other => panic!("Expected PositiveHit, got {:?}", other),
        }
    }

    #[test]
    fn fast_knowledge_lookup_negative_hit() {
        let cache = KnowledgeCache::new();
        let hash = PatternHash(2);
        let pat = CanonicalPattern::Integer(9);
        cache.insert_negative(hash, neg_entry(pat.clone()));
        let sig = AssumptionSignature::default();
        match cache.lookup_negative(hash, &pat, &sig) {
            CacheLookup::NegativeHit(_) => {}
            other => panic!("Expected NegativeHit, got {:?}", other),
        }
    }

    #[test]
    fn fast_knowledge_lookup_miss_wrong_hash() {
        let cache = KnowledgeCache::new();
        let insert_hash = PatternHash(10);
        let lookup_hash = PatternHash(11);
        let pat = CanonicalPattern::Integer(5);
        cache.insert_positive(insert_hash, pos_entry(pat.clone()));
        let sig = AssumptionSignature::default();
        match cache.lookup_positive(lookup_hash, &pat, &sig) {
            CacheLookup::Miss => {}
            other => panic!("Expected Miss, got {:?}", other),
        }
    }

    #[test]
    fn fast_knowledge_first_write_wins_duplicate() {
        let cache = KnowledgeCache::new();
        let hash = PatternHash(5);
        let pat = CanonicalPattern::Integer(3);
        cache.insert_positive(hash, pos_entry(pat.clone()));
        cache.insert_positive(hash, pos_entry(pat.clone()));
        // Only one entry should be stored (first-write-wins)
        assert_eq!(cache.positive_count(), 1);
    }

    #[test]
    fn fast_knowledge_stats_updated_on_lookup() {
        let cache = KnowledgeCache::new();
        let hash = PatternHash(7);
        let pat = CanonicalPattern::Integer(4);
        cache.insert_positive(hash, pos_entry(pat.clone()));
        let sig = AssumptionSignature::default();
        let _ = cache.lookup_positive(hash, &pat, &sig);
        let stats = cache.stats();
        assert_eq!(stats.positive_lookups, 1);
        assert_eq!(stats.positive_hits, 1);
    }

    #[test]
    fn fast_knowledge_stats_miss_increments_lookups_not_hits() {
        let cache = KnowledgeCache::new();
        let hash = PatternHash(8);
        let pat = CanonicalPattern::Integer(6);
        let sig = AssumptionSignature::default();
        let _ = cache.lookup_positive(hash, &pat, &sig);
        let stats = cache.stats();
        assert_eq!(stats.positive_lookups, 1);
        assert_eq!(stats.positive_hits, 0);
    }
}
