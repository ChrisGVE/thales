//! Cache statistics for D0 memoization diagnostics.
//!
//! [`CacheStats`] accumulates lookup and hit counters for both positive and
//! negative entries, as well as per-strategy hit counts. All fields are plain
//! `u64` counters — suitable for diagnostic reporting rather than hot-path use.

use std::collections::HashMap;

// ── CacheStats ────────────────────────────────────────────────────────────────

/// Snapshot statistics for a cache tier.
///
/// Obtained by calling `stats()` on [`KnowledgeCache`] or [`SolveCache`].
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total number of positive (solution) lookups attempted.
    pub positive_lookups: u64,
    /// Number of positive lookups that returned a hit.
    pub positive_hits: u64,
    /// Total number of negative (impossibility) lookups attempted.
    pub negative_lookups: u64,
    /// Number of negative lookups that returned a hit.
    pub negative_hits: u64,
    /// Hit counts broken down by strategy ID string.
    pub per_strategy_hits: HashMap<String, u64>,
    /// Number of entries promoted from SolveCache to KnowledgeCache.
    pub promotions: u64,
}

impl CacheStats {
    /// Fraction of positive lookups that returned a hit, in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when no positive lookups have been made.
    #[must_use]
    pub fn positive_hit_rate(&self) -> f64 {
        if self.positive_lookups == 0 {
            0.0
        } else {
            self.positive_hits as f64 / self.positive_lookups as f64
        }
    }

    /// Fraction of negative lookups that returned a hit, in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when no negative lookups have been made.
    #[must_use]
    pub fn negative_hit_rate(&self) -> f64 {
        if self.negative_lookups == 0 {
            0.0
        } else {
            self.negative_hits as f64 / self.negative_lookups as f64
        }
    }

    /// Number of hits attributed to the strategy identified by `id`.
    ///
    /// Returns `0` when the strategy has no recorded hits.
    #[must_use]
    pub fn strategy_hit_rate(&self, id: &str) -> u64 {
        self.per_strategy_hits.get(id).copied().unwrap_or(0)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_stats_default_all_zero() {
        let s = CacheStats::default();
        assert_eq!(s.positive_lookups, 0);
        assert_eq!(s.positive_hits, 0);
        assert_eq!(s.negative_lookups, 0);
        assert_eq!(s.negative_hits, 0);
        assert_eq!(s.promotions, 0);
        assert!(s.per_strategy_hits.is_empty());
    }

    #[test]
    fn fast_stats_positive_hit_rate_zero_lookups() {
        let s = CacheStats::default();
        assert_eq!(s.positive_hit_rate(), 0.0);
    }

    #[test]
    fn fast_stats_negative_hit_rate_zero_lookups() {
        let s = CacheStats::default();
        assert_eq!(s.negative_hit_rate(), 0.0);
    }

    #[test]
    fn fast_stats_positive_hit_rate_full() {
        let s = CacheStats {
            positive_lookups: 10,
            positive_hits: 10,
            ..Default::default()
        };
        assert!((s.positive_hit_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn fast_stats_positive_hit_rate_half() {
        let s = CacheStats {
            positive_lookups: 4,
            positive_hits: 2,
            ..Default::default()
        };
        assert!((s.positive_hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn fast_stats_negative_hit_rate_partial() {
        let s = CacheStats {
            negative_lookups: 6,
            negative_hits: 3,
            ..Default::default()
        };
        assert!((s.negative_hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn fast_stats_strategy_hit_rate_present() {
        let mut s = CacheStats::default();
        s.per_strategy_hits
            .insert("integration::risch".to_string(), 7);
        assert_eq!(s.strategy_hit_rate("integration::risch"), 7);
    }

    #[test]
    fn fast_stats_strategy_hit_rate_absent() {
        let s = CacheStats::default();
        assert_eq!(s.strategy_hit_rate("nonexistent"), 0);
    }
}
