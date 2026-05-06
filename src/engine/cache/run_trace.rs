//! Pipeline event log for cache-aware execution observability.

use crate::engine::trace_tree::StrategyId;

/// Why a strategy was skipped during the pipeline.
#[derive(Debug, Clone)]
pub enum SkipReason {
    NotApplicable,
    CacheHitPreceded,
    BudgetExhausted,
}

/// Summary of a strategy's outcome.
#[derive(Debug, Clone)]
pub enum StrategyOutcomeSummary {
    Solved,
    Failed,
    ProvenImpossible,
    Partial,
}

/// A single event in the pipeline execution log.
#[derive(Debug, Clone)]
pub enum PipelineEvent {
    CacheLookup {
        hit: bool,
    },
    StrategySkipped {
        id: StrategyId,
        reason: SkipReason,
    },
    StrategyDispatched {
        id: StrategyId,
    },
    StrategyCompleted {
        id: StrategyId,
        outcome: StrategyOutcomeSummary,
    },
    PhaseBarrierProcessed {
        phase_name: &'static str,
    },
}

/// Ordered log of pipeline events for a single engine invocation.
#[derive(Debug, Clone, Default)]
pub struct RunTrace {
    events: Vec<PipelineEvent>,
}

impl RunTrace {
    /// Create a new empty run trace.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an event to the trace.
    pub fn push(&mut self, event: PipelineEvent) {
        self.events.push(event);
    }

    /// Return all recorded events in order.
    #[must_use]
    pub fn events(&self) -> &[PipelineEvent] {
        &self.events
    }

    /// Number of events recorded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether no events have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_run_trace_empty() {
        let trace = RunTrace::new();
        assert!(trace.is_empty());
        assert_eq!(trace.len(), 0);
        assert_eq!(trace.events().len(), 0);
    }

    #[test]
    fn fast_run_trace_push_events() {
        let mut trace = RunTrace::new();
        trace.push(PipelineEvent::CacheLookup { hit: true });
        trace.push(PipelineEvent::StrategyDispatched {
            id: StrategyId("integration::risch"),
        });
        trace.push(PipelineEvent::StrategyCompleted {
            id: StrategyId("integration::risch"),
            outcome: StrategyOutcomeSummary::Solved,
        });
        assert_eq!(trace.len(), 3);
        assert!(!trace.is_empty());
    }

    #[test]
    fn fast_run_trace_events_accessible() {
        let mut trace = RunTrace::new();
        trace.push(PipelineEvent::CacheLookup { hit: false });
        trace.push(PipelineEvent::StrategySkipped {
            id: StrategyId("poly::factor"),
            reason: SkipReason::NotApplicable,
        });
        trace.push(PipelineEvent::PhaseBarrierProcessed {
            phase_name: "normalization",
        });

        let events = trace.events();
        assert_eq!(events.len(), 3);
        assert!(matches!(
            events[0],
            PipelineEvent::CacheLookup { hit: false }
        ));
        assert!(matches!(events[1], PipelineEvent::StrategySkipped { .. }));
        assert!(matches!(
            events[2],
            PipelineEvent::PhaseBarrierProcessed {
                phase_name: "normalization"
            }
        ));
    }
}
