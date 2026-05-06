//! Dynamic scoring policies for cache-aware strategy prioritization.

use crate::engine::cache::stats::CacheStats;
use crate::engine::trace_tree::StrategyId;

/// Policy for dynamically adjusting strategy priorities based on cache stats.
/// Positive values boost priority; negative values reduce it.
pub trait ScoringPolicy: Send + Sync {
    /// Return a priority adjustment for the given strategy.
    /// Positive values boost priority; negative values reduce it.
    fn adjust(&self, strategy_id: StrategyId, stats: &CacheStats) -> f64;
}

/// Scorer that adjusts priority based on overall cache hit rate.
/// Strategies run in a high-hit-rate environment get a small boost.
#[derive(Debug, Clone)]
pub struct HitRateScorer {
    sensitivity: f64,
}

impl Default for HitRateScorer {
    fn default() -> Self {
        Self { sensitivity: 1.0 }
    }
}

impl HitRateScorer {
    /// Create a scorer with the given sensitivity multiplier.
    #[must_use]
    pub fn with_sensitivity(sensitivity: f64) -> Self {
        Self { sensitivity }
    }

    /// Compute overall hit rate from stats.
    fn overall_hit_rate(stats: &CacheStats) -> f64 {
        let total_lookups = stats.positive_lookups + stats.negative_lookups;
        if total_lookups == 0 {
            return 0.0;
        }
        let total_hits = stats.positive_hits + stats.negative_hits;
        total_hits as f64 / total_lookups as f64
    }
}

impl ScoringPolicy for HitRateScorer {
    fn adjust(&self, _strategy_id: StrategyId, stats: &CacheStats) -> f64 {
        let hit_rate = Self::overall_hit_rate(stats);
        self.sensitivity * (hit_rate - 0.5) // Range: [-0.5, +0.5] * sensitivity
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_scorer_default() {
        let scorer = HitRateScorer::default();
        assert_eq!(scorer.sensitivity, 1.0);
    }

    #[test]
    fn fast_scorer_zero_lookups() {
        let scorer = HitRateScorer::default();
        let stats = CacheStats::default();
        // hit_rate = 0.0, so adjustment = 1.0 * (0.0 - 0.5) = -0.5
        let adj = scorer.adjust(StrategyId("test"), &stats);
        assert!((adj - (-0.5)).abs() < f64::EPSILON);
    }

    #[test]
    fn fast_scorer_high_hit_rate() {
        let scorer = HitRateScorer::default();
        let stats = CacheStats {
            positive_lookups: 10,
            positive_hits: 8,
            negative_lookups: 0,
            negative_hits: 0,
            ..Default::default()
        };
        // hit_rate = 8/10 = 0.8, adjustment = 1.0 * (0.8 - 0.5) = 0.3
        let adj = scorer.adjust(StrategyId("strategy"), &stats);
        assert!(adj > 0.0, "expected positive adjustment for high hit rate");
        assert!((adj - 0.3).abs() < 1e-10);
    }

    #[test]
    fn fast_scorer_sensitivity() {
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
}
