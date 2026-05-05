//! Per-solve transient cache for D0 memoization.
//!
//! [`SolveCache`] holds entries produced during a single solve invocation.
//! Entries marked [`Promotable::Yes`] can be drained and promoted to the
//! persistent [`KnowledgeCache`] at the end of the solve.
//!
//! The cache uses first-write-wins semantics: once an entry exists for a
//! given `(PatternHash, CanonicalPattern, AssumptionSignature)` triple it
//! is never overwritten.

use std::collections::HashMap;
use std::sync::RwLock;

use crate::engine::assumption_key::AssumptionSignature;
use crate::engine::cache::entry::{NegativeCacheEntry, PositiveCacheEntry};
use crate::engine::cache::knowledge::CacheLookup;
use crate::engine::cache::stats::CacheStats;
use crate::engine::canonical_pattern::{CanonicalPattern, PatternHash};

// ── Promotable ────────────────────────────────────────────────────────────────

/// Whether a [`SolveCache`] entry should be promoted to [`KnowledgeCache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Promotable {
    /// Entry is a candidate for promotion.
    Yes,
    /// Entry should remain local to this solve.
    No,
}

// ── SolveCacheEntry ───────────────────────────────────────────────────────────

struct SolvePosEntry {
    entry: PositiveCacheEntry,
    promotable: Promotable,
}

struct SolveNegEntry {
    entry: NegativeCacheEntry,
    promotable: Promotable,
}

// ── SolveCacheInner ───────────────────────────────────────────────────────────

struct SolveCacheInner {
    positive: HashMap<PatternHash, Vec<SolvePosEntry>>,
    negative: HashMap<PatternHash, Vec<SolveNegEntry>>,
}

impl SolveCacheInner {
    fn new() -> Self {
        SolveCacheInner {
            positive: HashMap::new(),
            negative: HashMap::new(),
        }
    }
}

// ── SolveCache ────────────────────────────────────────────────────────────────

/// Transient per-solve memoization cache.
///
/// Entries survive only for the duration of a single solve call. Entries
/// marked [`Promotable::Yes`] can be extracted via [`drain_promotable`] and
/// stored in the [`KnowledgeCache`] for reuse across solves.
pub struct SolveCache {
    inner: RwLock<SolveCacheInner>,
    stats: RwLock<CacheStats>,
}

impl std::fmt::Debug for SolveCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.read().unwrap();
        f.debug_struct("SolveCache")
            .field("positive_buckets", &inner.positive.len())
            .field("negative_buckets", &inner.negative.len())
            .finish()
    }
}

impl SolveCache {
    /// Create a new empty solve cache.
    #[must_use]
    pub fn new() -> Self {
        SolveCache {
            inner: RwLock::new(SolveCacheInner::new()),
            stats: RwLock::new(CacheStats::default()),
        }
    }

    /// Look up a positive entry. Two-level check: hash → structural equality
    /// on [`CanonicalPattern`] → assumption entailment.
    #[must_use]
    pub fn lookup_positive(
        &self,
        hash: PatternHash,
        pattern: &CanonicalPattern,
        current_sig: &AssumptionSignature,
    ) -> CacheLookup {
        use crate::engine::assumption_key::entails;
        {
            let mut stats = self.stats.write().unwrap();
            stats.positive_lookups += 1;
        }
        let inner = self.inner.read().unwrap();
        if let Some(bucket) = inner.positive.get(&hash) {
            for wrapped in bucket {
                if &wrapped.entry.canonical_pattern == pattern
                    && entails(current_sig, &wrapped.entry.assumption_sig)
                {
                    let mut stats = self.stats.write().unwrap();
                    stats.positive_hits += 1;
                    if let Some(sid) = &wrapped.entry.strategy_id {
                        *stats
                            .per_strategy_hits
                            .entry(sid.0.to_string())
                            .or_insert(0) += 1;
                    }
                    return CacheLookup::PositiveHit(wrapped.entry.clone());
                }
            }
        }
        CacheLookup::Miss
    }

    /// Look up a negative entry. Same two-level check as positive.
    #[must_use]
    pub fn lookup_negative(
        &self,
        hash: PatternHash,
        pattern: &CanonicalPattern,
        current_sig: &AssumptionSignature,
    ) -> CacheLookup {
        use crate::engine::assumption_key::entails;
        {
            let mut stats = self.stats.write().unwrap();
            stats.negative_lookups += 1;
        }
        let inner = self.inner.read().unwrap();
        if let Some(bucket) = inner.negative.get(&hash) {
            for wrapped in bucket {
                if &wrapped.entry.canonical_pattern == pattern
                    && entails(current_sig, &wrapped.entry.assumption_sig)
                {
                    let mut stats = self.stats.write().unwrap();
                    stats.negative_hits += 1;
                    return CacheLookup::NegativeHit(wrapped.entry.clone());
                }
            }
        }
        CacheLookup::Miss
    }

    /// Insert a positive entry with a promotion flag.
    ///
    /// First-write-wins: duplicate `(pattern, assumption_sig)` pairs are
    /// silently ignored.
    pub fn insert_positive(
        &self,
        hash: PatternHash,
        entry: PositiveCacheEntry,
        promotable: Promotable,
    ) {
        let mut inner = self.inner.write().unwrap();
        let bucket = inner.positive.entry(hash).or_default();
        let already_present = bucket.iter().any(|w| {
            w.entry.canonical_pattern == entry.canonical_pattern
                && w.entry.assumption_sig == entry.assumption_sig
        });
        if !already_present {
            bucket.push(SolvePosEntry { entry, promotable });
        }
    }

    /// Insert a negative entry with a promotion flag.
    ///
    /// First-write-wins: duplicate `(pattern, assumption_sig)` pairs are
    /// silently ignored.
    pub fn insert_negative(
        &self,
        hash: PatternHash,
        entry: NegativeCacheEntry,
        promotable: Promotable,
    ) {
        let mut inner = self.inner.write().unwrap();
        let bucket = inner.negative.entry(hash).or_default();
        let already_present = bucket.iter().any(|w| {
            w.entry.canonical_pattern == entry.canonical_pattern
                && w.entry.assumption_sig == entry.assumption_sig
        });
        if !already_present {
            bucket.push(SolveNegEntry { entry, promotable });
        }
    }

    /// Drain all entries marked [`Promotable::Yes`], returning them as
    /// `(PatternHash, PositiveCacheEntry)` and `(PatternHash, NegativeCacheEntry)`
    /// pairs suitable for insertion into a [`KnowledgeCache`].
    pub fn drain_promotable(
        &self,
    ) -> (
        Vec<(PatternHash, PositiveCacheEntry)>,
        Vec<(PatternHash, NegativeCacheEntry)>,
    ) {
        let mut inner = self.inner.write().unwrap();
        let mut pos_out = Vec::new();
        let mut neg_out = Vec::new();

        for (hash, bucket) in &mut inner.positive {
            let (keep, promote): (Vec<_>, Vec<_>) = std::mem::take(bucket)
                .into_iter()
                .partition(|w| w.promotable == Promotable::No);
            *bucket = keep;
            for w in promote {
                pos_out.push((*hash, w.entry));
            }
        }

        for (hash, bucket) in &mut inner.negative {
            let (keep, promote): (Vec<_>, Vec<_>) = std::mem::take(bucket)
                .into_iter()
                .partition(|w| w.promotable == Promotable::No);
            *bucket = keep;
            for w in promote {
                neg_out.push((*hash, w.entry));
            }
        }

        // Also update promotion counter in stats.
        {
            let mut stats = self.stats.write().unwrap();
            stats.promotions += (pos_out.len() + neg_out.len()) as u64;
        }

        (pos_out, neg_out)
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
            .map(|b| b.len())
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
            .map(|b| b.len())
            .sum()
    }
}

impl Default for SolveCache {
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
    use crate::engine::trace_tree::{BranchReason, TraceNode};
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
            strategy_id: None,
        }
    }

    fn neg_entry(pat: CanonicalPattern) -> NegativeCacheEntry {
        NegativeCacheEntry {
            canonical_pattern: pat,
            assumption_sig: AssumptionSignature::default(),
            proof: ImpossibilityProof::NoKovacicSolution,
        }
    }

    #[test]
    fn fast_solve_cache_new_is_empty() {
        let cache = SolveCache::new();
        assert_eq!(cache.positive_count(), 0);
        assert_eq!(cache.negative_count(), 0);
    }

    #[test]
    fn fast_solve_cache_insert_positive_lookup_hit() {
        let cache = SolveCache::new();
        let hash = PatternHash(1);
        let pat = CanonicalPattern::Integer(10);
        cache.insert_positive(hash, pos_entry(pat.clone()), Promotable::Yes);
        let sig = AssumptionSignature::default();
        match cache.lookup_positive(hash, &pat, &sig) {
            CacheLookup::PositiveHit(_) => {}
            other => panic!("Expected PositiveHit, got {:?}", other),
        }
    }

    #[test]
    fn fast_solve_cache_insert_negative_lookup_hit() {
        let cache = SolveCache::new();
        let hash = PatternHash(2);
        let pat = CanonicalPattern::Integer(20);
        cache.insert_negative(hash, neg_entry(pat.clone()), Promotable::No);
        let sig = AssumptionSignature::default();
        match cache.lookup_negative(hash, &pat, &sig) {
            CacheLookup::NegativeHit(_) => {}
            other => panic!("Expected NegativeHit, got {:?}", other),
        }
    }

    #[test]
    fn fast_solve_cache_first_write_wins() {
        let cache = SolveCache::new();
        let hash = PatternHash(3);
        let pat = CanonicalPattern::Integer(30);
        cache.insert_positive(hash, pos_entry(pat.clone()), Promotable::Yes);
        cache.insert_positive(hash, pos_entry(pat.clone()), Promotable::No);
        assert_eq!(cache.positive_count(), 1);
    }

    #[test]
    fn fast_solve_cache_drain_promotable_removes_promotable_entries() {
        let cache = SolveCache::new();
        let hash = PatternHash(4);
        cache.insert_positive(
            hash,
            pos_entry(CanonicalPattern::Integer(1)),
            Promotable::Yes,
        );
        cache.insert_positive(
            hash,
            pos_entry(CanonicalPattern::Integer(2)),
            Promotable::No,
        );

        let (pos, _neg) = cache.drain_promotable();
        assert_eq!(pos.len(), 1, "one promotable positive entry expected");
        // Non-promotable entry stays in the cache.
        assert_eq!(cache.positive_count(), 1);
    }

    #[test]
    fn fast_solve_cache_drain_negative_promotable() {
        let cache = SolveCache::new();
        let hash = PatternHash(5);
        cache.insert_negative(
            hash,
            neg_entry(CanonicalPattern::Integer(9)),
            Promotable::Yes,
        );

        let (_pos, neg) = cache.drain_promotable();
        assert_eq!(neg.len(), 1);
        assert_eq!(cache.negative_count(), 0);
    }

    #[test]
    fn fast_solve_cache_stats_promotions_tracked() {
        let cache = SolveCache::new();
        let hash = PatternHash(6);
        cache.insert_positive(
            hash,
            pos_entry(CanonicalPattern::Integer(7)),
            Promotable::Yes,
        );
        let _ = cache.drain_promotable();
        assert_eq!(cache.stats().promotions, 1);
    }

    #[test]
    fn fast_solve_cache_lookup_miss_updates_stats() {
        let cache = SolveCache::new();
        let hash = PatternHash(99);
        let pat = CanonicalPattern::Integer(0);
        let sig = AssumptionSignature::default();
        let _ = cache.lookup_positive(hash, &pat, &sig);
        let stats = cache.stats();
        assert_eq!(stats.positive_lookups, 1);
        assert_eq!(stats.positive_hits, 0);
    }
}
