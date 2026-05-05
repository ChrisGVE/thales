//! Execution mode for the D0 search strategy engine.
//!
//! [`ExecutionMode`] controls how the engine explores the strategy
//! search space: sequential single-pass, tree search with depth and
//! comparison criteria, or divide-and-conquer recursion.

// ── ExecutionMode ─────────────────────────────────────────────────────────────

/// Selects the strategy-execution mode for a D0 engine run.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionMode {
    /// Apply strategies in order; stop at the first success.
    Sequential,
    /// Explore a strategy tree up to `max_depth` levels, comparing branches
    /// according to the given [`TreeComparison`] criterion.
    TreeSearch {
        /// Maximum recursion depth in the strategy tree.
        max_depth: u32,
        /// How candidate branches are compared when multiple succeed.
        comparison: TreeComparison,
    },
    /// Split the problem into independent sub-problems, solve each
    /// recursively up to `max_depth`, then merge.
    DivideAndConquer {
        /// Maximum recursion depth for sub-problem splitting.
        max_depth: u32,
    },
}

impl Default for ExecutionMode {
    fn default() -> Self {
        ExecutionMode::Sequential
    }
}

// ── TreeComparison ────────────────────────────────────────────────────────────

/// Criterion for choosing among multiple successful branches in a tree search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeComparison {
    /// Accept the first branch that succeeds; do not explore further.
    FirstSuccess,
    /// Among all successful branches, prefer the one with the simplest result.
    SimplestResult,
    /// Among all successful branches, prefer the one with the fewest steps.
    FewestSteps,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_mode_default_is_sequential() {
        assert_eq!(ExecutionMode::default(), ExecutionMode::Sequential);
    }

    #[test]
    fn fast_mode_clone_eq() {
        let mode = ExecutionMode::TreeSearch {
            max_depth: 5,
            comparison: TreeComparison::FewestSteps,
        };
        assert_eq!(mode.clone(), mode);
    }

    #[test]
    fn fast_mode_sequential_ne_tree_search() {
        let a = ExecutionMode::Sequential;
        let b = ExecutionMode::TreeSearch {
            max_depth: 3,
            comparison: TreeComparison::FirstSuccess,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn fast_mode_tree_search_comparison_variants() {
        assert_ne!(TreeComparison::FirstSuccess, TreeComparison::SimplestResult);
        assert_ne!(TreeComparison::SimplestResult, TreeComparison::FewestSteps);
        assert_eq!(TreeComparison::FirstSuccess, TreeComparison::FirstSuccess);
    }

    #[test]
    fn fast_mode_divide_and_conquer_clone() {
        let m = ExecutionMode::DivideAndConquer { max_depth: 10 };
        assert_eq!(m.clone(), m);
    }

    #[test]
    fn fast_mode_tree_search_different_depth_ne() {
        let a = ExecutionMode::TreeSearch {
            max_depth: 2,
            comparison: TreeComparison::SimplestResult,
        };
        let b = ExecutionMode::TreeSearch {
            max_depth: 4,
            comparison: TreeComparison::SimplestResult,
        };
        assert_ne!(a, b);
    }
}
