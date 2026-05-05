//! Property set for expressions encountered during engine search.
//!
//! [`PropertySet`] accumulates structural and domain properties learned about
//! sub-expressions. Properties are keyed by a string label (to be upgraded to
//! structural hashes in a later task) and stored as a list of [`Property`] values.

use std::collections::HashMap;
use std::sync::Arc;

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

/// Accumulated properties keyed by a string identifier.
///
/// String keys will be upgraded to `u64` structural hashes in a later task.
/// For now they are plain `String` labels identifying the expression.
#[derive(Debug, Clone, Default)]
pub struct PropertySet {
    props: HashMap<String, Vec<Property>>,
}

impl PropertySet {
    /// Create an empty property set.
    #[must_use]
    pub fn new() -> Self {
        PropertySet::default()
    }

    /// Record a property for the expression identified by `key`.
    pub fn learn(&mut self, key: impl Into<String>, prop: Property) {
        self.props.entry(key.into()).or_default().push(prop);
    }

    /// Returns all properties recorded for `key`, or an empty slice.
    #[must_use]
    pub fn get(&self, key: &str) -> &[Property] {
        self.props.get(key).map_or(&[], Vec::as_slice)
    }

    /// Returns `true` if the exact `prop` has been recorded for `key`.
    #[must_use]
    pub fn has(&self, key: &str, prop: &Property) -> bool {
        self.get(key).contains(prop)
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

    #[test]
    fn fast_property_set_new_is_empty() {
        let ps = PropertySet::new();
        assert!(ps.is_empty());
        assert_eq!(ps.len(), 0);
    }

    #[test]
    fn fast_property_learn_and_get() {
        let mut ps = PropertySet::new();
        ps.learn("x", Property::Rational);
        assert_eq!(ps.get("x"), &[Property::Rational]);
        assert_eq!(ps.len(), 1);
    }

    #[test]
    fn fast_property_get_unknown_key_empty_slice() {
        let ps = PropertySet::new();
        assert_eq!(ps.get("unknown"), &[]);
    }

    #[test]
    fn fast_property_has_present() {
        let mut ps = PropertySet::new();
        ps.learn("expr", Property::Zero);
        assert!(ps.has("expr", &Property::Zero));
    }

    #[test]
    fn fast_property_has_absent() {
        let mut ps = PropertySet::new();
        ps.learn("expr", Property::Rational);
        assert!(!ps.has("expr", &Property::Analytic));
    }

    #[test]
    fn fast_property_multiple_per_key() {
        let mut ps = PropertySet::new();
        ps.learn("f", Property::Analytic);
        ps.learn("f", Property::Polynomial { degree: 2 });
        assert_eq!(ps.get("f").len(), 2);
        assert!(ps.has("f", &Property::Analytic));
        assert!(ps.has("f", &Property::Polynomial { degree: 2 }));
        assert_eq!(ps.len(), 2);
    }

    #[test]
    fn fast_property_multiple_keys() {
        let mut ps = PropertySet::new();
        ps.learn("a", Property::Constant);
        ps.learn("b", Property::Zero);
        assert_eq!(ps.len(), 2);
        assert!(ps.has("a", &Property::Constant));
        assert!(ps.has("b", &Property::Zero));
        assert!(!ps.has("a", &Property::Zero));
    }

    #[test]
    fn fast_property_constraint_variant() {
        let mut ps = PropertySet::new();
        ps.learn("x", Property::Constraint(PropertyConstraint::Positive));
        assert!(ps.has("x", &Property::Constraint(PropertyConstraint::Positive)));
        assert!(!ps.has("x", &Property::Constraint(PropertyConstraint::Negative)));
    }

    #[test]
    fn fast_property_custom() {
        let mut ps = PropertySet::new();
        let label: Arc<str> = Arc::from("invertible");
        ps.learn("M", Property::Custom(label.clone()));
        assert!(ps.has("M", &Property::Custom(label)));
    }
}
