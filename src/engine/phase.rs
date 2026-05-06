//! Phase barriers for intermediate state communication between strategy phases.

use crate::numeric::Expr;
use std::sync::Arc;

/// Barrier marking a phase transition with intermediate state.
///
/// Strategies emit phase barriers to communicate intermediate results
/// (e.g., a partial factorization) to the next phase in the pipeline.
#[derive(Debug, Clone)]
pub struct PhaseBarrier {
    /// Name of the phase that produced this barrier.
    pub phase_name: &'static str,
    /// Intermediate expression state at the phase boundary.
    pub intermediate: Arc<Expr>,
}

impl PhaseBarrier {
    /// Create a new phase barrier.
    #[must_use]
    pub fn new(phase_name: &'static str, intermediate: Arc<Expr>) -> Self {
        Self {
            phase_name,
            intermediate,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::context::SolveContext;
    use crate::engine::resource::ResourceBudget;
    use crate::numeric::SmallInt;

    fn make_expr() -> Arc<Expr> {
        Arc::new(Expr::Integer(SmallInt::from(1i64)))
    }

    #[test]
    fn fast_phase_barrier_create() {
        let expr = make_expr();
        let barrier = PhaseBarrier::new("test_phase", Arc::clone(&expr));
        assert_eq!(barrier.phase_name, "test_phase");
        assert!(Arc::ptr_eq(&barrier.intermediate, &expr));
    }

    #[test]
    fn fast_phase_barrier_clone() {
        let expr = make_expr();
        let barrier = PhaseBarrier::new("clone_phase", Arc::clone(&expr));
        let cloned = barrier.clone();
        assert_eq!(cloned.phase_name, barrier.phase_name);
        assert!(Arc::ptr_eq(&cloned.intermediate, &barrier.intermediate));
    }

    #[test]
    fn fast_phase_barrier_set_get() {
        let expr = make_expr();
        let mut ctx = SolveContext::new(Arc::clone(&expr), ResourceBudget::unlimited());

        // Initially no barrier.
        assert!(ctx.take_phase_barrier().is_none());

        // Set a barrier and take it.
        let barrier = PhaseBarrier::new("set_phase", Arc::clone(&expr));
        ctx.set_phase_barrier(barrier);

        let taken = ctx.take_phase_barrier();
        assert!(taken.is_some());
        let taken = taken.unwrap();
        assert_eq!(taken.phase_name, "set_phase");
        assert!(Arc::ptr_eq(&taken.intermediate, &expr));

        // Second take returns None.
        assert!(ctx.take_phase_barrier().is_none());
    }
}
