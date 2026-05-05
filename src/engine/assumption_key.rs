//! Assumption normalization, signing, and cache-key types for D0 memoization.
//!
//! This module provides:
//!
//! - [`Domain`] — the domain lattice used in assumption constraints.
//! - [`AssumptionConstraint`] — a normalized single-variable constraint.
//! - [`NormalizedAssumption`] — wrapper around one constraint.
//! - [`AssumptionSignature`] — sorted set of constraints used as a cache key.
//!
//! Free functions:
//! - [`normalize_assumption`] — extract a constraint from an [`Assumption`].
//! - [`sign`] — compute an [`AssumptionSignature`] for an [`AssumptionSet`].
//! - [`sign_with_varmap`] — like [`sign`] but renames variables through a [`VarMap`].
//! - [`entails`] — test whether one signature subsumes another.

use std::collections::BTreeSet;

use crate::api::diagnostic::Assumption;
use crate::engine::assumption::AssumptionSet;
use crate::engine::assumption_entailment::constraint_entails;
use crate::engine::canonical_pattern::VarMap;

// ── Domain ────────────────────────────────────────────────────────────────────

/// The domain of a variable — element of the domain subsumption lattice.
///
/// Ordering is derived so that `BTreeSet` can store domains for use in
/// sorted cache keys. The ordering is arbitrary (lexicographic by
/// discriminant name) and carries no lattice meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Domain {
    /// Strictly positive integers: {1, 2, 3, …}
    PositiveIntegers,
    /// Strictly negative integers: {…, -3, -2, -1}
    NegativeIntegers,
    /// All integers: ℤ
    Integers,
    /// Rational numbers: ℚ
    Rationals,
    /// Strictly positive reals: (0, ∞)
    PositiveReals,
    /// Strictly negative reals: (-∞, 0)
    NegativeReals,
    /// Non-negative reals: [0, ∞)
    NonNegativeReals,
    /// Non-positive reals: (-∞, 0]
    NonPositiveReals,
    /// Non-zero reals: ℝ \ {0}
    NonZeroReals,
    /// All reals: ℝ
    Reals,
    /// Complex numbers: ℂ
    Complex,
}

// ── AssumptionConstraint ──────────────────────────────────────────────────────

/// A normalized single-variable constraint extracted from an [`Assumption`].
///
/// All variants carry the variable name as a `String` so that constraints
/// from different variables remain distinct in a [`BTreeSet`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AssumptionConstraint {
    /// Variable is restricted to a named domain.
    InDomain { var: String, domain: Domain },
    /// Variable is strictly positive.
    Positive { var: String },
    /// Variable is non-negative (≥ 0).
    NonNegative { var: String },
    /// Variable is strictly negative.
    Negative { var: String },
    /// Variable is non-positive (≤ 0).
    NonPositive { var: String },
    /// Variable is non-zero.
    NonZero { var: String },
    /// Variable is strictly greater than `bound`.
    GreaterThan { var: String, bound: String },
    /// Variable is at least `bound` (≥).
    AtLeast { var: String, bound: String },
    /// Variable is strictly less than `bound`.
    LessThan { var: String, bound: String },
    /// Variable is at most `bound` (≤).
    AtMost { var: String, bound: String },
    /// Constraint whose template was not recognized; stored as normalized text.
    Opaque { normalized_text: String },
}

// ── NormalizedAssumption ──────────────────────────────────────────────────────

/// A single normalized constraint, ready for inclusion in a cache key.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NormalizedAssumption {
    pub constraint: AssumptionConstraint,
}

// ── AssumptionSignature ───────────────────────────────────────────────────────

/// A sorted, deduplicated set of normalized constraints used as a cache key.
///
/// Two signatures are equal iff they contain exactly the same constraint set.
/// Build via [`sign`] or [`sign_with_varmap`]; an empty signature is obtained
/// from [`AssumptionSignature::empty`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AssumptionSignature {
    constraints: BTreeSet<NormalizedAssumption>,
}

impl AssumptionSignature {
    /// Return an empty signature (no active assumptions).
    pub fn empty() -> Self {
        Self {
            constraints: BTreeSet::new(),
        }
    }

    /// Build a signature from a non-empty set of constraints.
    ///
    /// # Panics (debug only)
    /// Panics in debug builds if `constraints` is empty — use [`empty()`][Self::empty]
    /// for empty signatures.
    pub(crate) fn from_constraints(constraints: BTreeSet<NormalizedAssumption>) -> Self {
        debug_assert!(!constraints.is_empty(), "use empty() for empty signatures");
        Self { constraints }
    }

    /// Iterate over the normalized constraints in sorted order.
    pub fn iter(&self) -> impl Iterator<Item = &NormalizedAssumption> {
        self.constraints.iter()
    }

    /// Number of constraints in this signature.
    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// `true` when the signature contains no constraints.
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }
}

impl Default for AssumptionSignature {
    fn default() -> Self {
        Self::empty()
    }
}

// ── normalize_assumption ──────────────────────────────────────────────────────

/// Extract a [`NormalizedAssumption`] from an [`Assumption`].
///
/// Normalization uses `template_id` as the primary key. Recognized templates
/// are mapped to typed constraints; anything else becomes
/// [`AssumptionConstraint::Opaque`].
///
/// # Template conventions
///
/// | template_id prefix                     | constraint produced             |
/// |----------------------------------------|---------------------------------|
/// | `"engine.assumption.positive"`         | `Positive { var }`              |
/// | `"engine.assumption.negative"`         | `Negative { var }`              |
/// | `"engine.assumption.nonnegative"`      | `NonNegative { var }`           |
/// | `"engine.assumption.nonpositive"`      | `NonPositive { var }`           |
/// | `"engine.assumption.nonzero"`          | `NonZero { var }`               |
/// | `"engine.assumption.in_domain"`        | `InDomain { var, domain }`      |
/// | `"engine.assumption.greater_than"`     | `GreaterThan { var, bound }`    |
/// | `"engine.assumption.at_least"`         | `AtLeast { var, bound }`        |
/// | `"engine.assumption.less_than"`        | `LessThan { var, bound }`       |
/// | `"engine.assumption.at_most"`          | `AtMost { var, bound }`         |
/// | anything else                          | `Opaque { normalized_text }`    |
///
/// For templates that need `var` / `bound` / `domain` values, the function
/// reads bindings from the narrative by name. If a required binding is absent
/// the result falls back to `Opaque`.
pub fn normalize_assumption(assumption: &Assumption) -> NormalizedAssumption {
    let tid = assumption.narrative.template_id;
    let bindings = &assumption.narrative.bindings;

    let get_text = |name: &str| -> Option<String> {
        bindings.iter().find(|(k, _)| k == name).and_then(|(_, v)| {
            use crate::api::NarrativeValue;
            match v {
                NarrativeValue::Text(s) => Some(s.clone()),
                NarrativeValue::Int(n) => Some(n.to_string()),
                _ => None,
            }
        })
    };

    let constraint = match tid {
        "engine.assumption.positive" => {
            if let Some(var) = get_text("var") {
                AssumptionConstraint::Positive { var }
            } else {
                opaque_from(assumption)
            }
        }
        "engine.assumption.negative" => {
            if let Some(var) = get_text("var") {
                AssumptionConstraint::Negative { var }
            } else {
                opaque_from(assumption)
            }
        }
        "engine.assumption.nonnegative" => {
            if let Some(var) = get_text("var") {
                AssumptionConstraint::NonNegative { var }
            } else {
                opaque_from(assumption)
            }
        }
        "engine.assumption.nonpositive" => {
            if let Some(var) = get_text("var") {
                AssumptionConstraint::NonPositive { var }
            } else {
                opaque_from(assumption)
            }
        }
        "engine.assumption.nonzero" => {
            if let Some(var) = get_text("var") {
                AssumptionConstraint::NonZero { var }
            } else {
                opaque_from(assumption)
            }
        }
        "engine.assumption.in_domain" => {
            let var = get_text("var");
            let domain_str = get_text("domain");
            match (var, domain_str) {
                (Some(var), Some(domain_str)) => {
                    if let Some(domain) = parse_domain(&domain_str) {
                        AssumptionConstraint::InDomain { var, domain }
                    } else {
                        opaque_from(assumption)
                    }
                }
                _ => opaque_from(assumption),
            }
        }
        "engine.assumption.greater_than" => {
            let var = get_text("var");
            let bound = get_text("bound");
            match (var, bound) {
                (Some(var), Some(bound)) => AssumptionConstraint::GreaterThan { var, bound },
                _ => opaque_from(assumption),
            }
        }
        "engine.assumption.at_least" => {
            let var = get_text("var");
            let bound = get_text("bound");
            match (var, bound) {
                (Some(var), Some(bound)) => AssumptionConstraint::AtLeast { var, bound },
                _ => opaque_from(assumption),
            }
        }
        "engine.assumption.less_than" => {
            let var = get_text("var");
            let bound = get_text("bound");
            match (var, bound) {
                (Some(var), Some(bound)) => AssumptionConstraint::LessThan { var, bound },
                _ => opaque_from(assumption),
            }
        }
        "engine.assumption.at_most" => {
            let var = get_text("var");
            let bound = get_text("bound");
            match (var, bound) {
                (Some(var), Some(bound)) => AssumptionConstraint::AtMost { var, bound },
                _ => opaque_from(assumption),
            }
        }
        _ => opaque_from(assumption),
    };

    NormalizedAssumption { constraint }
}

/// Build an `Opaque` constraint from an assumption using `template_id` as the
/// normalized text (avoids using `fallback_md` which can be locale-sensitive).
fn opaque_from(assumption: &Assumption) -> AssumptionConstraint {
    AssumptionConstraint::Opaque {
        normalized_text: assumption.narrative.template_id.to_string(),
    }
}

/// Parse a domain name string into a [`Domain`] variant.
fn parse_domain(s: &str) -> Option<Domain> {
    match s {
        "PositiveIntegers" | "positive_integers" => Some(Domain::PositiveIntegers),
        "NegativeIntegers" | "negative_integers" => Some(Domain::NegativeIntegers),
        "Integers" | "integers" | "Z" => Some(Domain::Integers),
        "Rationals" | "rationals" | "Q" => Some(Domain::Rationals),
        "PositiveReals" | "positive_reals" => Some(Domain::PositiveReals),
        "NegativeReals" | "negative_reals" => Some(Domain::NegativeReals),
        "NonNegativeReals" | "non_negative_reals" => Some(Domain::NonNegativeReals),
        "NonPositiveReals" | "non_positive_reals" => Some(Domain::NonPositiveReals),
        "NonZeroReals" | "non_zero_reals" => Some(Domain::NonZeroReals),
        "Reals" | "reals" | "R" => Some(Domain::Reals),
        "Complex" | "complex" | "C" => Some(Domain::Complex),
        _ => None,
    }
}

// ── sign ──────────────────────────────────────────────────────────────────────

/// Compute an [`AssumptionSignature`] for an [`AssumptionSet`].
///
/// All active assumptions across all open scopes are normalized and collected
/// into a sorted, deduplicated set.
pub(crate) fn sign(set: &AssumptionSet) -> AssumptionSignature {
    let active = set.active();
    if active.is_empty() {
        return AssumptionSignature::empty();
    }
    let constraints: BTreeSet<NormalizedAssumption> =
        active.iter().map(normalize_assumption).collect();
    AssumptionSignature::from_constraints(constraints)
}

// ── sign_with_varmap ──────────────────────────────────────────────────────────

/// Compute an [`AssumptionSignature`] with variables renamed through `var_map`.
///
/// Variable names in each constraint are replaced by their slot-based names
/// (`"$0"`, `"$1"`, …) where a mapping exists. This makes the signature
/// independent of the concrete variable names chosen by the caller, enabling
/// cache hits for structurally identical problems with different variable names.
///
/// Variables not present in `var_map` are kept as-is.
pub fn sign_with_varmap(set: &AssumptionSet, var_map: &VarMap) -> AssumptionSignature {
    let active = set.active();
    if active.is_empty() {
        return AssumptionSignature::empty();
    }
    let constraints: BTreeSet<NormalizedAssumption> = active
        .iter()
        .map(normalize_assumption)
        .map(|na| NormalizedAssumption {
            constraint: rename_constraint(na.constraint, var_map),
        })
        .collect();
    AssumptionSignature::from_constraints(constraints)
}

/// Rename the variable(s) in a constraint using the var-map.
///
/// The var-map maps `SymbolId → SlotId`. We look up the variable name by
/// interning it, and if found, replace the name with the slot's canonical
/// form (`"$0"`, `"$1"`, etc.).
fn rename_constraint(c: AssumptionConstraint, var_map: &VarMap) -> AssumptionConstraint {
    use crate::numeric::SymbolId;
    use AssumptionConstraint::*;

    let rename = |var: String| -> String {
        let sid = SymbolId::intern(&var);
        var_map
            .slot_of(sid)
            .map(|slot| format!("${}", slot.0))
            .unwrap_or(var)
    };

    match c {
        InDomain { var, domain } => InDomain {
            var: rename(var),
            domain,
        },
        Positive { var } => Positive { var: rename(var) },
        NonNegative { var } => NonNegative { var: rename(var) },
        Negative { var } => Negative { var: rename(var) },
        NonPositive { var } => NonPositive { var: rename(var) },
        NonZero { var } => NonZero { var: rename(var) },
        GreaterThan { var, bound } => GreaterThan {
            var: rename(var),
            bound,
        },
        AtLeast { var, bound } => AtLeast {
            var: rename(var),
            bound,
        },
        LessThan { var, bound } => LessThan {
            var: rename(var),
            bound,
        },
        AtMost { var, bound } => AtMost {
            var: rename(var),
            bound,
        },
        Opaque { normalized_text } => Opaque { normalized_text },
    }
}

// ── entails ───────────────────────────────────────────────────────────────────

/// Returns `true` when every constraint in `subset` is entailed by some
/// constraint in `superset`.
///
/// Uses [`constraint_entails`] for pairwise checks. This is O(|subset| ×
/// |superset|) — both sets are small in practice (≤ tens of constraints).
pub fn entails(superset: &AssumptionSignature, subset: &AssumptionSignature) -> bool {
    subset.iter().all(|weak| {
        superset
            .iter()
            .any(|strong| constraint_entails(&strong.constraint, &weak.constraint))
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{Narrative, NarrativeValue};

    // ── helpers ────────────────────────────────────────────────────────────────

    fn make_assumption_raw(template_id: &'static str, fallback: &str) -> Assumption {
        Assumption {
            narrative: Narrative::new(template_id, fallback),
            path: None,
        }
    }

    fn make_assumption_var(template_id: &'static str, fallback: &str, var: &str) -> Assumption {
        Assumption {
            narrative: Narrative::new(template_id, fallback)
                .bind("var", NarrativeValue::Text(var.to_string())),
            path: None,
        }
    }

    fn make_assumption_var_bound(
        template_id: &'static str,
        fallback: &str,
        var: &str,
        bound: &str,
    ) -> Assumption {
        Assumption {
            narrative: Narrative::new(template_id, fallback)
                .bind("var", NarrativeValue::Text(var.to_string()))
                .bind("bound", NarrativeValue::Text(bound.to_string())),
            path: None,
        }
    }

    fn make_assumption_in_domain(fallback: &str, var: &str, domain: &str) -> Assumption {
        Assumption {
            narrative: Narrative::new("engine.assumption.in_domain", fallback)
                .bind("var", NarrativeValue::Text(var.to_string()))
                .bind("domain", NarrativeValue::Text(domain.to_string())),
            path: None,
        }
    }

    // ── normalize_assumption ───────────────────────────────────────────────────

    #[test]
    fn fast_normalize_positive_template() {
        let a = make_assumption_var("engine.assumption.positive", "x > 0", "x");
        let na = normalize_assumption(&a);
        assert_eq!(
            na.constraint,
            AssumptionConstraint::Positive { var: "x".into() }
        );
    }

    #[test]
    fn fast_normalize_negative_template() {
        let a = make_assumption_var("engine.assumption.negative", "x < 0", "x");
        let na = normalize_assumption(&a);
        assert_eq!(
            na.constraint,
            AssumptionConstraint::Negative { var: "x".into() }
        );
    }

    #[test]
    fn fast_normalize_nonnegative_template() {
        let a = make_assumption_var("engine.assumption.nonnegative", "x >= 0", "x");
        let na = normalize_assumption(&a);
        assert_eq!(
            na.constraint,
            AssumptionConstraint::NonNegative { var: "x".into() }
        );
    }

    #[test]
    fn fast_normalize_nonpositive_template() {
        let a = make_assumption_var("engine.assumption.nonpositive", "x <= 0", "x");
        let na = normalize_assumption(&a);
        assert_eq!(
            na.constraint,
            AssumptionConstraint::NonPositive { var: "x".into() }
        );
    }

    #[test]
    fn fast_normalize_nonzero_template() {
        let a = make_assumption_var("engine.assumption.nonzero", "x != 0", "x");
        let na = normalize_assumption(&a);
        assert_eq!(
            na.constraint,
            AssumptionConstraint::NonZero { var: "x".into() }
        );
    }

    #[test]
    fn fast_normalize_in_domain_reals() {
        let a = make_assumption_in_domain("x in R", "x", "Reals");
        let na = normalize_assumption(&a);
        assert_eq!(
            na.constraint,
            AssumptionConstraint::InDomain {
                var: "x".into(),
                domain: Domain::Reals
            }
        );
    }

    #[test]
    fn fast_normalize_in_domain_positive_integers() {
        let a = make_assumption_in_domain("n in Z+", "n", "PositiveIntegers");
        let na = normalize_assumption(&a);
        assert_eq!(
            na.constraint,
            AssumptionConstraint::InDomain {
                var: "n".into(),
                domain: Domain::PositiveIntegers
            }
        );
    }

    #[test]
    fn fast_normalize_greater_than() {
        let a = make_assumption_var_bound("engine.assumption.greater_than", "x > 3", "x", "3");
        let na = normalize_assumption(&a);
        assert_eq!(
            na.constraint,
            AssumptionConstraint::GreaterThan {
                var: "x".into(),
                bound: "3".into()
            }
        );
    }

    #[test]
    fn fast_normalize_at_least() {
        let a = make_assumption_var_bound("engine.assumption.at_least", "x >= 0", "x", "0");
        let na = normalize_assumption(&a);
        assert_eq!(
            na.constraint,
            AssumptionConstraint::AtLeast {
                var: "x".into(),
                bound: "0".into()
            }
        );
    }

    #[test]
    fn fast_normalize_less_than() {
        let a = make_assumption_var_bound("engine.assumption.less_than", "x < 5", "x", "5");
        let na = normalize_assumption(&a);
        assert_eq!(
            na.constraint,
            AssumptionConstraint::LessThan {
                var: "x".into(),
                bound: "5".into()
            }
        );
    }

    #[test]
    fn fast_normalize_at_most() {
        let a = make_assumption_var_bound("engine.assumption.at_most", "x <= 2", "x", "2");
        let na = normalize_assumption(&a);
        assert_eq!(
            na.constraint,
            AssumptionConstraint::AtMost {
                var: "x".into(),
                bound: "2".into()
            }
        );
    }

    #[test]
    fn fast_normalize_unknown_template_is_opaque() {
        let a = make_assumption_raw("engine.assumption.unknown_custom", "some assumption");
        let na = normalize_assumption(&a);
        assert!(matches!(na.constraint, AssumptionConstraint::Opaque { .. }));
    }

    #[test]
    fn fast_normalize_missing_var_binding_is_opaque() {
        // template_id recognized but no "var" binding → falls back to Opaque
        let a = Assumption {
            narrative: Narrative::new("engine.assumption.positive", "x > 0"),
            path: None,
        };
        let na = normalize_assumption(&a);
        assert!(matches!(na.constraint, AssumptionConstraint::Opaque { .. }));
    }

    // ── sign ───────────────────────────────────────────────────────────────────

    #[test]
    fn fast_sign_empty_set_returns_empty_signature() {
        let set = AssumptionSet::new();
        let sig = sign(&set);
        assert!(sig.is_empty());
        assert_eq!(sig.len(), 0);
    }

    #[test]
    fn fast_sign_single_assumption() {
        let set = AssumptionSet::new();
        let _g = set.push_scope();
        set.assert(make_assumption_var(
            "engine.assumption.positive",
            "x > 0",
            "x",
        ));
        let sig = sign(&set);
        assert_eq!(sig.len(), 1);
    }

    #[test]
    fn fast_sign_deduplicates_identical_constraints() {
        let set = AssumptionSet::new();
        let _g = set.push_scope();
        set.assert(make_assumption_var(
            "engine.assumption.positive",
            "x > 0",
            "x",
        ));
        set.assert(make_assumption_var(
            "engine.assumption.positive",
            "x > 0",
            "x",
        ));
        let sig = sign(&set);
        // BTreeSet deduplicates
        assert_eq!(sig.len(), 1);
    }

    #[test]
    fn fast_sign_two_distinct_assumptions() {
        let set = AssumptionSet::new();
        let _g = set.push_scope();
        set.assert(make_assumption_var(
            "engine.assumption.positive",
            "x > 0",
            "x",
        ));
        set.assert(make_assumption_var(
            "engine.assumption.nonzero",
            "y != 0",
            "y",
        ));
        let sig = sign(&set);
        assert_eq!(sig.len(), 2);
    }

    // ── entails ────────────────────────────────────────────────────────────────

    #[test]
    fn fast_entails_empty_subset_always_true() {
        let set = AssumptionSet::new();
        let _g = set.push_scope();
        set.assert(make_assumption_var(
            "engine.assumption.positive",
            "x > 0",
            "x",
        ));
        let superset_sig = sign(&set);
        let empty_sig = AssumptionSignature::empty();
        assert!(entails(&superset_sig, &empty_sig));
    }

    #[test]
    fn fast_entails_identical_signatures() {
        let set = AssumptionSet::new();
        let _g = set.push_scope();
        set.assert(make_assumption_var(
            "engine.assumption.positive",
            "x > 0",
            "x",
        ));
        let sig = sign(&set);
        assert!(entails(&sig, &sig));
    }

    #[test]
    fn fast_entails_positive_implies_nonnegative() {
        // superset has Positive(x), subset asks for NonNegative(x)
        let strong_set = AssumptionSet::new();
        let _g1 = strong_set.push_scope();
        strong_set.assert(make_assumption_var(
            "engine.assumption.positive",
            "x > 0",
            "x",
        ));
        let superset_sig = sign(&strong_set);

        let weak_set = AssumptionSet::new();
        let _g2 = weak_set.push_scope();
        weak_set.assert(make_assumption_var(
            "engine.assumption.nonnegative",
            "x >= 0",
            "x",
        ));
        let subset_sig = sign(&weak_set);

        assert!(entails(&superset_sig, &subset_sig));
    }

    #[test]
    fn fast_no_entails_nonnegative_does_not_imply_positive() {
        let weak_set = AssumptionSet::new();
        let _g1 = weak_set.push_scope();
        weak_set.assert(make_assumption_var(
            "engine.assumption.nonnegative",
            "x >= 0",
            "x",
        ));
        let weak_sig = sign(&weak_set);

        let strong_set = AssumptionSet::new();
        let _g2 = strong_set.push_scope();
        strong_set.assert(make_assumption_var(
            "engine.assumption.positive",
            "x > 0",
            "x",
        ));
        let strong_sig = sign(&strong_set);

        // NonNegative does NOT entail Positive
        assert!(!entails(&weak_sig, &strong_sig));
    }

    #[test]
    fn fast_no_entails_wrong_variable() {
        let set_x = AssumptionSet::new();
        let _g1 = set_x.push_scope();
        set_x.assert(make_assumption_var(
            "engine.assumption.positive",
            "x > 0",
            "x",
        ));
        let sig_x = sign(&set_x);

        let set_y = AssumptionSet::new();
        let _g2 = set_y.push_scope();
        set_y.assert(make_assumption_var(
            "engine.assumption.positive",
            "y > 0",
            "y",
        ));
        let sig_y = sign(&set_y);

        // sig_x has Positive(x), sig_y asks for Positive(y) — different vars
        assert!(!entails(&sig_x, &sig_y));
    }

    #[test]
    fn fast_entails_d012_e2_positive_integers_implies_nonnegative() {
        let pi_set = AssumptionSet::new();
        let _g = pi_set.push_scope();
        pi_set.assert(make_assumption_in_domain(
            "n in Z+",
            "n",
            "PositiveIntegers",
        ));
        let pi_sig = sign(&pi_set);

        let nn_set = AssumptionSet::new();
        let _g2 = nn_set.push_scope();
        nn_set.assert(make_assumption_var(
            "engine.assumption.nonnegative",
            "n >= 0",
            "n",
        ));
        let nn_sig = sign(&nn_set);

        assert!(entails(&pi_sig, &nn_sig));
    }

    #[test]
    fn fast_entails_d012_e2_negative_integers_implies_nonpositive() {
        let ni_set = AssumptionSet::new();
        let _g = ni_set.push_scope();
        ni_set.assert(make_assumption_in_domain(
            "n in Z-",
            "n",
            "NegativeIntegers",
        ));
        let ni_sig = sign(&ni_set);

        let np_set = AssumptionSet::new();
        let _g2 = np_set.push_scope();
        np_set.assert(make_assumption_var(
            "engine.assumption.nonpositive",
            "n <= 0",
            "n",
        ));
        let np_sig = sign(&np_set);

        assert!(entails(&ni_sig, &np_sig));
    }

    // ── proptest ───────────────────────────────────────────────────────────────

    #[cfg(test)]
    mod prop {
        use super::*;
        use proptest::prelude::*;

        fn arb_domain() -> impl Strategy<Value = Domain> {
            prop_oneof![
                Just(Domain::PositiveIntegers),
                Just(Domain::NegativeIntegers),
                Just(Domain::Integers),
                Just(Domain::Rationals),
                Just(Domain::PositiveReals),
                Just(Domain::NegativeReals),
                Just(Domain::NonNegativeReals),
                Just(Domain::NonPositiveReals),
                Just(Domain::NonZeroReals),
                Just(Domain::Reals),
                Just(Domain::Complex),
            ]
        }

        fn arb_constraint() -> impl Strategy<Value = AssumptionConstraint> {
            let var_strat = prop_oneof![Just("x"), Just("y"), Just("z")].prop_map(String::from);
            prop_oneof![
                var_strat
                    .clone()
                    .prop_map(|var| AssumptionConstraint::Positive { var }),
                var_strat
                    .clone()
                    .prop_map(|var| AssumptionConstraint::NonNegative { var }),
                var_strat
                    .clone()
                    .prop_map(|var| AssumptionConstraint::Negative { var }),
                var_strat
                    .clone()
                    .prop_map(|var| AssumptionConstraint::NonPositive { var }),
                var_strat
                    .clone()
                    .prop_map(|var| AssumptionConstraint::NonZero { var }),
                (var_strat.clone(), arb_domain())
                    .prop_map(|(var, domain)| { AssumptionConstraint::InDomain { var, domain } }),
            ]
        }

        fn arb_assumption_signature() -> impl Strategy<Value = AssumptionSignature> {
            // D012-E1: use range 1..=5 — from_constraints rejects empty sets
            proptest::collection::btree_set(
                arb_constraint().prop_map(|c| NormalizedAssumption { constraint: c }),
                1..=5,
            )
            .prop_map(AssumptionSignature::from_constraints)
        }

        proptest! {
            #[test]
            fn prop_entails_reflexive(sig in arb_assumption_signature()) {
                prop_assert!(entails(&sig, &sig));
            }

            #[test]
            fn prop_entails_empty_subset(sig in arb_assumption_signature()) {
                let empty = AssumptionSignature::empty();
                prop_assert!(entails(&sig, &empty));
            }

            #[test]
            fn prop_signature_len_positive(sig in arb_assumption_signature()) {
                prop_assert!(sig.len() >= 1);
                prop_assert!(!sig.is_empty());
            }
        }
    }
}
