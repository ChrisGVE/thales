//! Property set for expressions encountered during engine search.
//!
//! [`PropertySet`] accumulates structural and domain properties learned about
//! sub-expressions. Properties are keyed by the `u64` structural hash of the
//! expression (via [`structural_hash`]) so that two structurally identical
//! `Arc<Expr>` trees share a bucket regardless of pointer identity.

use std::collections::HashMap;
use std::sync::Arc;

use crate::engine::canonical_pattern::structural_hash;
use crate::numeric::Expr;

// ── PropertyConstraint ────────────────────────────────────────────────────────

/// Domain constraint on a symbolic variable or sub-expression.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyConstraint {
    /// Value is strictly positive.
    Positive,
    /// Value is non-negative.
    NonNegative,
    /// Value is strictly negative.
    Negative,
    /// Value is an integer.
    Integer,
    /// Value is real.
    Real,
    /// Value is complex.
    Complex,
    /// Custom constraint description.
    Custom(String),
}

// ── Property ──────────────────────────────────────────────────────────────────

/// A structural or domain property of a sub-expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Property {
    /// Expression is a polynomial of the given degree.
    Polynomial { degree: u32 },
    /// Expression is a rational function.
    Rational,
    /// Expression is analytic on its domain.
    Analytic,
    /// Expression satisfies a domain constraint.
    Constraint(PropertyConstraint),
    /// Expression evaluates to zero everywhere it is defined.
    Zero,
    /// Expression is a constant (no free variables).
    Constant,
    /// Custom property identified by a string label.
    Custom(Arc<str>),
}

// ── PropertySet ───────────────────────────────────────────────────────────────

/// Accumulated properties keyed by the structural hash of the expression.
///
/// Each bucket stores `(Arc<Expr>, Property)` pairs so that collisions between
/// structurally different expressions that hash to the same value are handled
/// correctly via structural equality on the stored `Arc<Expr>`.
#[derive(Debug, Clone, Default)]
pub struct PropertySet {
    props: HashMap<u64, Vec<(Arc<Expr>, Property)>>,
}

impl PropertySet {
    /// Create an empty property set.
    #[must_use]
    pub fn new() -> Self {
        PropertySet::default()
    }

    /// Record a property for the expression `key`.
    pub fn learn(&mut self, key: &Arc<Expr>, prop: Property) {
        let hash = structural_hash(key);
        self.props
            .entry(hash)
            .or_default()
            .push((Arc::clone(key), prop));
    }

    /// Returns all properties recorded for `key`, or an empty slice.
    #[must_use]
    pub fn get(&self, key: &Arc<Expr>) -> Vec<&Property> {
        let hash = structural_hash(key);
        match self.props.get(&hash) {
            None => Vec::new(),
            Some(bucket) => bucket
                .iter()
                .filter(|(stored, _)| Arc::ptr_eq(stored, key) || stored == key)
                .map(|(_, p)| p)
                .collect(),
        }
    }

    /// Returns `true` if the exact `prop` has been recorded for `key`.
    #[must_use]
    pub fn has(&self, key: &Arc<Expr>, prop: &Property) -> bool {
        self.get(key).contains(&prop)
    }

    /// Total number of property entries across all keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.props.values().map(Vec::len).sum()
    }

    /// Returns `true` if no properties have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{Expr, SmallInt, SymbolId};

    fn int_expr(n: i64) -> Arc<Expr> {
        Arc::new(Expr::Integer(SmallInt::from(n)))
    }

    fn sym_expr(name: &str) -> Arc<Expr> {
        Arc::new(Expr::Symbol(SymbolId::intern(name)))
    }

    #[test]
    fn fast_property_set_new_is_empty() {
        let ps = PropertySet::new();
        assert!(ps.is_empty());
        assert_eq!(ps.len(), 0);
    }

    #[test]
    fn fast_property_learn_and_get_by_expr() {
        let mut ps = PropertySet::new();
        let x = sym_expr("x");
        ps.learn(&x, Property::Rational);
        let props = ps.get(&x);
        assert_eq!(props.len(), 1);
        assert_eq!(props[0], &Property::Rational);
        assert_eq!(ps.len(), 1);
    }

    #[test]
    fn fast_property_get_unknown_key_empty_slice() {
        let ps = PropertySet::new();
        let x = sym_expr("unknown");
        assert!(ps.get(&x).is_empty());
    }

    #[test]
    fn fast_property_has_present() {
        let mut ps = PropertySet::new();
        let e = int_expr(0);
        ps.learn(&e, Property::Zero);
        assert!(ps.has(&e, &Property::Zero));
    }

    #[test]
    fn fast_property_has_absent() {
        let mut ps = PropertySet::new();
        let e = sym_expr("expr");
        ps.learn(&e, Property::Rational);
        assert!(!ps.has(&e, &Property::Analytic));
    }

    #[test]
    fn fast_property_multiple_per_key() {
        let mut ps = PropertySet::new();
        let f = sym_expr("f");
        ps.learn(&f, Property::Analytic);
        ps.learn(&f, Property::Polynomial { degree: 2 });
        assert_eq!(ps.get(&f).len(), 2);
        assert!(ps.has(&f, &Property::Analytic));
        assert!(ps.has(&f, &Property::Polynomial { degree: 2 }));
        assert_eq!(ps.len(), 2);
    }

    #[test]
    fn fast_property_multiple_keys() {
        let mut ps = PropertySet::new();
        let a = sym_expr("a");
        let b = sym_expr("b");
        ps.learn(&a, Property::Constant);
        ps.learn(&b, Property::Zero);
        assert_eq!(ps.len(), 2);
        assert!(ps.has(&a, &Property::Constant));
        assert!(ps.has(&b, &Property::Zero));
        assert!(!ps.has(&a, &Property::Zero));
    }

    #[test]
    fn fast_property_constraint_variant() {
        let mut ps = PropertySet::new();
        let x = sym_expr("x");
        ps.learn(&x, Property::Constraint(PropertyConstraint::Positive));
        assert!(ps.has(&x, &Property::Constraint(PropertyConstraint::Positive)));
        assert!(!ps.has(&x, &Property::Constraint(PropertyConstraint::Negative)));
    }

    #[test]
    fn fast_property_custom() {
        let mut ps = PropertySet::new();
        let m = sym_expr("M");
        let label: Arc<str> = Arc::from("invertible");
        ps.learn(&m, Property::Custom(label.clone()));
        assert!(ps.has(&m, &Property::Custom(label)));
    }

    #[test]
    fn fast_property_learn_and_get_int_expr() {
        let mut ps = PropertySet::new();
        let e = int_expr(42);
        ps.learn(&e, Property::Constant);
        assert!(ps.has(&e, &Property::Constant));
    }

    #[test]
    fn fast_property_distinct_exprs_distinct_buckets() {
        let mut ps = PropertySet::new();
        let x = sym_expr("x");
        let y = sym_expr("y");
        ps.learn(&x, Property::Rational);
        ps.learn(&y, Property::Analytic);
        // x and y have different structural hashes
        assert!(!ps.has(&x, &Property::Analytic));
        assert!(!ps.has(&y, &Property::Rational));
        assert_eq!(ps.len(), 2);
    }
}
