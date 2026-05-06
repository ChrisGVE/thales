//! Resource budget and status for engine computations.
//!
//! [`ResourceBudget`] is a cheaply clonable, atomically-tracked budget
//! covering step count, memory, and wall-clock time. Engines call
//! [`ResourceBudget::consume_steps`] at each decision point and abort
//! when [`ResourceStatus::Exceeded`] is returned.
//!
//! When the `rayon` feature is enabled, [`ResourceGate`] limits the
//! number of concurrently active parallel branches.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

// ── ResourceStatus ────────────────────────────────────────────────────────────

/// The current health of a [`ResourceBudget`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceStatus {
    /// Resources are well within limits.
    Ok,
    /// Approaching the limit (≥ 80% consumed); consider wrapping up.
    Approaching,
    /// Limit reached or exceeded; the operation must stop.
    Exceeded,
}

// ── BudgetInner ───────────────────────────────────────────────────────────────

#[derive(Debug)]
struct BudgetInner {
    max_steps: u32,
    steps_used: AtomicUsize,
    max_memory_bytes: usize,
    memory_used: AtomicUsize,
    max_time_ms: u64,
    start: Instant,
}

// ── ResourceBudget ────────────────────────────────────────────────────────────

/// Shared, atomically-tracked resource budget.
///
/// Clone is O(1): all clones share the same counters via `Arc`.
/// This lets sub-engines hold a clone of the parent's budget and
/// consume from the same pool.
#[derive(Debug, Clone)]
pub struct ResourceBudget {
    inner: Arc<BudgetInner>,
}

impl ResourceBudget {
    /// Create a budget with explicit limits.
    ///
    /// - `max_steps`: maximum strategy steps before [`ResourceStatus::Exceeded`].
    /// - `max_memory_bytes`: soft memory ceiling (checked via [`record_memory`]).
    /// - `max_time_ms`: wall-clock time limit in milliseconds.
    ///
    /// [`record_memory`]: ResourceBudget::record_memory
    #[must_use]
    pub fn new(max_steps: u32, max_memory_bytes: usize, max_time_ms: u64) -> Self {
        ResourceBudget {
            inner: Arc::new(BudgetInner {
                max_steps,
                steps_used: AtomicUsize::new(0),
                max_memory_bytes,
                memory_used: AtomicUsize::new(0),
                max_time_ms,
                start: Instant::now(),
            }),
        }
    }

    /// Create an unlimited budget (useful for tests and non-bounded contexts).
    #[must_use]
    pub fn unlimited() -> Self {
        ResourceBudget::new(u32::MAX, usize::MAX, u64::MAX)
    }

    /// Attempt to consume `n` steps from the budget.
    ///
    /// Returns:
    /// - [`ResourceStatus::Ok`] when well within limits (< 80% used).
    /// - [`ResourceStatus::Approaching`] when ≥ 80% of the step budget is used.
    /// - [`ResourceStatus::Exceeded`] when the budget is already exhausted;
    ///   in this case the steps are **not** consumed.
    pub fn consume_steps(&self, n: u32) -> ResourceStatus {
        let n_usize = n as usize;
        let max = self.inner.max_steps as usize;

        // CAS loop: only consume if we are below the limit.
        loop {
            let current = self.inner.steps_used.load(Ordering::Acquire);
            if current >= max {
                return ResourceStatus::Exceeded;
            }
            let next = current.saturating_add(n_usize);
            match self.inner.steps_used.compare_exchange(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // Determine status based on new usage level.
                    if next >= max {
                        return ResourceStatus::Exceeded;
                    }
                    // 80% threshold: next * 10 >= max * 8
                    if next.saturating_mul(10) >= max.saturating_mul(8) {
                        return ResourceStatus::Approaching;
                    }
                    return ResourceStatus::Ok;
                }
                Err(_) => {
                    // Another thread updated; retry.
                    std::hint::spin_loop();
                }
            }
        }
    }

    /// Check whether the wall-clock time budget has been exceeded.
    #[must_use]
    pub fn check_time(&self) -> ResourceStatus {
        let elapsed = self.inner.start.elapsed().as_millis() as u64;
        let max = self.inner.max_time_ms;
        if elapsed >= max {
            ResourceStatus::Exceeded
        } else if elapsed.saturating_mul(10) >= max.saturating_mul(8) {
            ResourceStatus::Approaching
        } else {
            ResourceStatus::Ok
        }
    }

    /// Record additional memory allocation.
    ///
    /// Adds `bytes` to the running total and returns the resulting status.
    pub fn record_memory(&self, bytes: usize) -> ResourceStatus {
        let used = self.inner.memory_used.fetch_add(bytes, Ordering::AcqRel) + bytes;
        let max = self.inner.max_memory_bytes;
        if used >= max {
            ResourceStatus::Exceeded
        } else if used.saturating_mul(10) >= max.saturating_mul(8) {
            ResourceStatus::Approaching
        } else {
            ResourceStatus::Ok
        }
    }

    /// Returns `true` if any resource (steps, memory, or time) is exhausted.
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.inner.steps_used.load(Ordering::Acquire) >= self.inner.max_steps as usize
            || self.inner.memory_used.load(Ordering::Acquire) >= self.inner.max_memory_bytes
            || self.inner.start.elapsed().as_millis() as u64 >= self.inner.max_time_ms
    }
}

// ── ResourceGate (rayon only) ─────────────────────────────────────────────────

/// Limits the number of concurrently active parallel branches when the
/// `rayon` feature is enabled.
///
/// Callers acquire a [`BranchGuard`] via [`ResourceGate::acquire`] before
/// spawning a parallel branch. If the gate is already at capacity, `None`
/// is returned and the caller should fall back to sequential execution.
#[cfg(feature = "rayon")]
pub struct ResourceGate {
    max_concurrent: usize,
    active: Arc<AtomicUsize>,
}

#[cfg(feature = "rayon")]
impl ResourceGate {
    /// Create a gate allowing at most `max` concurrent branches.
    #[must_use]
    pub fn new(max: usize) -> Self {
        ResourceGate {
            max_concurrent: max,
            active: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Try to acquire a branch slot.
    ///
    /// Returns `Some(BranchGuard)` if a slot was available, `None` if
    /// the gate is already at capacity.  The guard releases the slot on drop.
    pub fn acquire(&self) -> Option<BranchGuard<'_>> {
        // CAS loop: increment only if below max.
        loop {
            let current = self.active.load(Ordering::Acquire);
            if current >= self.max_concurrent {
                return None;
            }
            match self.active.compare_exchange(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(BranchGuard { gate: self }),
                Err(_) => std::hint::spin_loop(),
            }
        }
    }

    /// Current number of active parallel branches.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active.load(Ordering::Acquire)
    }
}

/// RAII guard that releases one branch slot from a [`ResourceGate`] on drop.
#[cfg(feature = "rayon")]
pub struct BranchGuard<'a> {
    gate: &'a ResourceGate,
}

#[cfg(feature = "rayon")]
impl Drop for BranchGuard<'_> {
    fn drop(&mut self) {
        self.gate.active.fetch_sub(1, Ordering::Release);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_budget_unlimited_never_exhausted() {
        let b = ResourceBudget::unlimited();
        assert!(!b.is_exhausted());
        assert_eq!(b.consume_steps(1_000_000), ResourceStatus::Ok);
        assert!(!b.is_exhausted());
    }

    #[test]
    fn fast_budget_consume_steps_ok() {
        let b = ResourceBudget::new(100, usize::MAX, u64::MAX);
        assert_eq!(b.consume_steps(10), ResourceStatus::Ok);
    }

    #[test]
    fn fast_budget_consume_steps_approaching() {
        let b = ResourceBudget::new(100, usize::MAX, u64::MAX);
        // Consume 79 steps → still Ok
        assert_eq!(b.consume_steps(79), ResourceStatus::Ok);
        // Consume 1 more → 80% → Approaching
        assert_eq!(b.consume_steps(1), ResourceStatus::Approaching);
    }

    #[test]
    fn fast_budget_consume_steps_exceeded() {
        let b = ResourceBudget::new(10, usize::MAX, u64::MAX);
        // Exhaust budget
        let _ = b.consume_steps(10);
        // Next call should be Exceeded and not consume
        assert_eq!(b.consume_steps(1), ResourceStatus::Exceeded);
        assert!(b.is_exhausted());
    }

    #[test]
    fn fast_budget_consume_steps_exact_limit() {
        let b = ResourceBudget::new(5, usize::MAX, u64::MAX);
        // Consume exactly 5 → at/above limit → Exceeded
        let status = b.consume_steps(5);
        assert_eq!(status, ResourceStatus::Exceeded);
        assert!(b.is_exhausted());
    }

    #[test]
    fn fast_budget_clone_shares_counters() {
        let b1 = ResourceBudget::new(10, usize::MAX, u64::MAX);
        let b2 = b1.clone();
        b1.consume_steps(5);
        // b2 sees the same counter
        assert_eq!(b2.consume_steps(5), ResourceStatus::Exceeded);
    }

    #[test]
    fn fast_budget_memory_tracking() {
        let b = ResourceBudget::new(u32::MAX, 100, u64::MAX);
        assert_eq!(b.record_memory(50), ResourceStatus::Ok);
        assert_eq!(b.record_memory(30), ResourceStatus::Approaching);
        assert_eq!(b.record_memory(30), ResourceStatus::Exceeded);
    }

    #[test]
    fn fast_budget_not_exhausted_below_limit() {
        let b = ResourceBudget::new(100, usize::MAX, u64::MAX);
        b.consume_steps(50);
        assert!(!b.is_exhausted());
    }
}
