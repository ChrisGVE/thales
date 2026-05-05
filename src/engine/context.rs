//! Solve context passed to and through D0 strategy execution.
//!
//! [`SolveContext`] bundles the expression under consideration with the
//! accumulated trace, open assumptions, resource budget, and learned
//! properties. Strategies receive context by value and return a
//! [`crate::engine::strategy::StrategyResult`]; forked contexts carry
//! independent traces so dead-end branches do not pollute the parent.

use std::sync::Arc;

use crate::engine::assumption::AssumptionSet;
use crate::engine::fallback::FallbackConfig;
use crate::engine::property::PropertySet;
use crate::engine::resource::ResourceBudget;
use crate::engine::trace_tree::TraceTree;
use crate::numeric::Expr;

// ── SolveContext ──────────────────────────────────────────────────────────────

/// Execution context for a single D0 engine invocation.
///
/// All fields are public to allow strategies direct read access.
/// Mutation is intentional: strategies consume or transform context by value.
///
/// `ResourceBudget` is cloned cheaply (it wraps an `Arc`) so forks draw
/// against the same shared counters as the parent.
#[derive(Debug, Clone)]
pub struct SolveContext {
    /// The expression currently under evaluation.
    pub expr: Arc<Expr>,
    /// Tree-structured trace of all steps taken so far in this context.
    pub trace: TraceTree,
    /// Active assumptions accumulated during this computation.
    pub assumptions: AssumptionSet,
    /// Shared resource budget (clones share the same atomic counters).
    pub budget: ResourceBudget,
    /// Properties learned about sub-expressions during search.
    pub properties: PropertySet,
    /// Numerical fallback configuration for this invocation.
    pub fallback: FallbackConfig,
}

impl SolveContext {
    /// Create a fresh context for `expr` with the given budget.
    ///
    /// Trace, assumptions, properties, and fallback config start at their
    /// defaults (`FallbackConfig::disabled()`).
    #[must_use]
    pub fn new(expr: Arc<Expr>, budget: ResourceBudget) -> Self {
        Self {
            expr,
            trace: TraceTree::new(),
            assumptions: AssumptionSet::default(),
            budget,
            properties: PropertySet::default(),
            fallback: FallbackConfig::default(),
        }
    }

    /// Consume this context and return a new one with `fallback` replaced.
    #[must_use]
    pub fn with_fallback(self, fallback: FallbackConfig) -> Self {
        Self { fallback, ..self }
    }

    /// Fork this context for an independent sub-problem.
    ///
    /// The fork carries a **new empty trace** so the sub-problem's steps do
    /// not appear in the parent trace. All other fields are cloned:
    /// - `expr` is shared via `Arc::clone` (O(1)).
    /// - `assumptions` is deep-cloned (snapshot of current active scopes).
    /// - `budget` is cloned — shares the same underlying atomic counters,
    ///   so resource consumption by the fork reduces the parent's remaining
    ///   budget.
    /// - `properties` is deep-cloned (snapshot).
    #[must_use]
    pub fn fork(&self) -> Self {
        Self {
            expr: Arc::clone(&self.expr),
            trace: TraceTree::new(),
            assumptions: self.assumptions.clone(),
            budget: self.budget.clone(),
            properties: self.properties.clone(),
            fallback: self.fallback.clone(),
        }
    }

    /// Consume this context and return a new one with `expr` replaced.
    ///
    /// All other fields are preserved unchanged.
    #[must_use]
    pub fn with_expr(self, expr: Arc<Expr>) -> Self {
        Self { expr, ..self }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::SmallInt;

    fn make_expr(n: i64) -> Arc<Expr> {
        Arc::new(Expr::Integer(SmallInt::from(n)))
    }

    fn unlimited_ctx(n: i64) -> SolveContext {
        SolveContext::new(make_expr(n), ResourceBudget::unlimited())
    }

    #[test]
    fn fast_context_new_has_empty_trace() {
        let ctx = unlimited_ctx(1);
        assert_eq!(ctx.trace.step_count(), 0);
        assert!(ctx.assumptions.is_empty());
        assert!(ctx.properties.is_empty());
    }

    #[test]
    fn fast_context_new_preserves_expr() {
        let expr = make_expr(42);
        let ctx = SolveContext::new(Arc::clone(&expr), ResourceBudget::unlimited());
        // Pointer identity: same Arc node.
        assert!(Arc::ptr_eq(&ctx.expr, &expr));
    }

    #[test]
    fn fast_context_fork_has_independent_trace() {
        let ctx = unlimited_ctx(7);
        let fork = ctx.fork();
        // Fork starts with an empty trace independent of parent.
        assert_eq!(fork.trace.step_count(), 0);
        // Original is not consumed (we called fork on &self).
        assert_eq!(ctx.trace.step_count(), 0);
    }

    #[test]
    fn fast_context_fork_shares_budget_counters() {
        use crate::engine::resource::ResourceStatus;

        let budget = ResourceBudget::new(10, usize::MAX, u64::MAX);
        let ctx = SolveContext::new(make_expr(1), budget);
        let fork = ctx.fork();

        // Consuming from the fork reduces the budget shared with the parent.
        let status = fork.budget.consume_steps(9);
        assert_eq!(status, ResourceStatus::Approaching);

        // Consuming one more from the parent hits the shared limit.
        let status2 = ctx.budget.consume_steps(2);
        assert_eq!(status2, ResourceStatus::Exceeded);
    }

    #[test]
    fn fast_context_fork_independent_assumptions() {
        use crate::api::diagnostic::Assumption;
        use crate::api::Narrative;

        let ctx = unlimited_ctx(3);
        let fork = ctx.fork();

        // Pushing a scope on the parent does not affect the fork.
        let _guard = ctx.assumptions.push_scope();
        ctx.assumptions.assert(Assumption {
            narrative: Narrative::new("engine.assumption", "x > 0"),
            path: None,
        });
        assert_eq!(ctx.assumptions.len(), 1);
        assert_eq!(fork.assumptions.len(), 0);
    }

    #[test]
    fn fast_context_with_expr_replaces_expr() {
        let ctx = unlimited_ctx(5);
        let new_expr = make_expr(99);
        let ctx2 = ctx.with_expr(Arc::clone(&new_expr));
        assert!(Arc::ptr_eq(&ctx2.expr, &new_expr));
    }

    #[test]
    fn fast_context_with_expr_preserves_budget() {
        let budget = ResourceBudget::new(100, usize::MAX, u64::MAX);
        let ctx = SolveContext::new(make_expr(1), budget.clone());
        let ctx2 = ctx.with_expr(make_expr(2));
        // Budget counters are still shared.
        budget.consume_steps(10);
        assert!(!ctx2.budget.is_exhausted());
    }

    #[test]
    fn fast_context_clone_is_independent_trace() {
        let mut ctx = unlimited_ctx(0);
        let ctx2 = ctx.clone();
        // Pushing a step to the clone's trace does not affect the original.
        use crate::numeric::trace::{Step, TechniqueTag};
        ctx.trace
            .push_step(Step::new(TechniqueTag::Simplification, "test"));
        assert_eq!(ctx.trace.step_count(), 1);
        assert_eq!(ctx2.trace.step_count(), 0);
    }

    #[test]
    fn fast_context_new_has_disabled_fallback() {
        use crate::engine::fallback::FallbackConfig;
        let ctx = unlimited_ctx(1);
        assert_eq!(ctx.fallback, FallbackConfig::disabled());
        assert!(!ctx.fallback.numerical);
    }

    #[test]
    fn fast_context_with_fallback_replaces_config() {
        use crate::engine::fallback::FallbackConfig;
        let ctx = unlimited_ctx(1);
        let ctx2 = ctx.with_fallback(FallbackConfig::enabled());
        assert!(ctx2.fallback.numerical);
    }

    #[test]
    fn fast_context_fork_carries_fallback() {
        use crate::engine::fallback::FallbackConfig;
        let ctx = unlimited_ctx(1).with_fallback(FallbackConfig::enabled_silent());
        let fork = ctx.fork();
        assert_eq!(fork.fallback, FallbackConfig::enabled_silent());
    }
}
