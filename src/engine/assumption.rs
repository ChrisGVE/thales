//! RAII-scoped assumption set for engine computations.
//!
//! [`AssumptionSet`] tracks the stack of assumptions an engine has made.
//! Assumptions are pushed in scopes; when the scope guard is dropped, the
//! assumptions from that scope are automatically removed, restoring the
//! previous state.
//!
//! # Interior mutability
//!
//! `AssumptionSet` uses `RefCell` internally so that [`push_scope`],
//! [`assert`], and read accessors can all be called while an
//! [`AssumptionGuard`] is live — the guard holds `&AssumptionSet` (a
//! shared reference) and uses runtime borrowing for its single `pop_scope`
//! call in `Drop`.
//!
//! [`push_scope`]: AssumptionSet::push_scope
//! [`assert`]: AssumptionSet::assert

use std::cell::RefCell;

use crate::api::diagnostic::Assumption;

// ── AssumptionSet ─────────────────────────────────────────────────────────────

/// A stack of scoped assumptions accumulated during a computation.
///
/// Each scope is opened with [`AssumptionSet::push_scope`], which returns an
/// [`AssumptionGuard`]. When the guard is dropped, the scope and all
/// assumptions registered within it are removed.
#[derive(Debug, Default)]
pub struct AssumptionSet {
    /// Stack of assumption lists, one per open scope.
    scopes: RefCell<Vec<Vec<Assumption>>>,
    /// Monotonically increasing ID assigned to each scope.
    scope_ids: RefCell<Vec<usize>>,
    /// Counter for the next scope ID.
    next_id: RefCell<usize>,
}

impl Clone for AssumptionSet {
    fn clone(&self) -> Self {
        AssumptionSet {
            scopes: RefCell::new(self.scopes.borrow().clone()),
            scope_ids: RefCell::new(self.scope_ids.borrow().clone()),
            next_id: RefCell::new(*self.next_id.borrow()),
        }
    }
}

impl AssumptionSet {
    /// Create an empty assumption set with no open scopes.
    #[must_use]
    pub fn new() -> Self {
        AssumptionSet::default()
    }

    /// Open a new assumption scope.
    ///
    /// Returns an [`AssumptionGuard`] that will close the scope when dropped.
    /// Any assumptions added via [`AssumptionSet::assert`] while this guard
    /// is live belong to this scope.
    pub fn push_scope(&self) -> AssumptionGuard<'_> {
        let id = {
            let mut n = self.next_id.borrow_mut();
            let id = *n;
            *n += 1;
            id
        };
        self.scopes.borrow_mut().push(Vec::new());
        self.scope_ids.borrow_mut().push(id);
        AssumptionGuard {
            set: self,
            scope_id: id,
        }
    }

    /// Add an assumption to the innermost open scope.
    ///
    /// Panics in debug builds if no scope is open.
    pub fn assert(&self, assumption: Assumption) {
        let mut scopes = self.scopes.borrow_mut();
        debug_assert!(!scopes.is_empty(), "assert() called with no open scope");
        if let Some(top) = scopes.last_mut() {
            top.push(assumption);
        }
    }

    /// Returns all assumptions currently active across all open scopes,
    /// in order from outermost to innermost scope.
    #[must_use]
    pub fn active(&self) -> Vec<Assumption> {
        self.scopes
            .borrow()
            .iter()
            .flat_map(|s| s.iter().cloned())
            .collect()
    }

    /// Total number of active assumptions across all open scopes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.scopes.borrow().iter().map(|s| s.len()).sum()
    }

    /// Returns `true` if there are no active assumptions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Close the scope identified by `scope_id`.
    ///
    /// Called by [`AssumptionGuard::drop`]. Asserts in debug mode that
    /// `scope_id` matches the top of the stack.
    fn pop_scope(&self, scope_id: usize) {
        let mut ids = self.scope_ids.borrow_mut();
        debug_assert_eq!(
            ids.last().copied(),
            Some(scope_id),
            "AssumptionGuard dropped out of order"
        );
        ids.pop();
        drop(ids);
        self.scopes.borrow_mut().pop();
    }
}

// ── AssumptionGuard ───────────────────────────────────────────────────────────

/// RAII guard that closes an assumption scope when dropped.
///
/// Obtain one from [`AssumptionSet::push_scope`]. While this guard is live,
/// any call to [`AssumptionSet::assert`] adds to the scope this guard owns.
pub struct AssumptionGuard<'a> {
    set: &'a AssumptionSet,
    scope_id: usize,
}

impl Drop for AssumptionGuard<'_> {
    fn drop(&mut self) {
        self.set.pop_scope(self.scope_id);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Narrative;

    fn make_assumption(text: &str) -> Assumption {
        Assumption {
            narrative: Narrative::new("engine.assumption", text),
            path: None,
        }
    }

    #[test]
    fn fast_assumption_set_new_is_empty() {
        let set = AssumptionSet::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert!(set.active().is_empty());
    }

    #[test]
    fn fast_assumption_scope_push_and_pop() {
        let set = AssumptionSet::new();
        {
            let _guard = set.push_scope();
            set.assert(make_assumption("x > 0"));
            assert_eq!(set.len(), 1);
            assert!(!set.is_empty());
        }
        // Guard dropped: scope removed
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn fast_assumption_nested_scopes() {
        let set = AssumptionSet::new();
        {
            let _outer = set.push_scope();
            set.assert(make_assumption("x > 0"));
            {
                let _inner = set.push_scope();
                set.assert(make_assumption("y != 0"));
                assert_eq!(set.len(), 2);
                let active = set.active();
                assert_eq!(active.len(), 2);
            }
            // Inner scope dropped: y != 0 removed
            assert_eq!(set.len(), 1);
            let active = set.active();
            assert_eq!(active.len(), 1);
        }
        // Outer scope dropped: x > 0 removed
        assert!(set.is_empty());
    }

    #[test]
    fn fast_assumption_active_order() {
        let set = AssumptionSet::new();
        {
            let _outer = set.push_scope();
            set.assert(make_assumption("a"));
            {
                let _inner = set.push_scope();
                set.assert(make_assumption("b"));
                let active = set.active();
                // Outermost first
                assert_eq!(active.len(), 2);
            }
        }
    }

    #[test]
    fn fast_assumption_multiple_asserts_in_scope() {
        let set = AssumptionSet::new();
        {
            let _guard = set.push_scope();
            set.assert(make_assumption("p"));
            set.assert(make_assumption("q"));
            set.assert(make_assumption("r"));
            assert_eq!(set.len(), 3);
        }
        assert_eq!(set.len(), 0);
    }
}
