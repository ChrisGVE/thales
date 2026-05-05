//! Assumption normalization: [`Assumption`] → [`NormalizedAssumption`].
//!
//! Extracts typed constraints from narrative-based assumptions using
//! `template_id` as the primary key. Unrecognized templates produce
//! [`AssumptionConstraint::Opaque`].

use crate::api::diagnostic::Assumption;
use crate::api::NarrativeValue;
use crate::engine::assumption_key::{AssumptionConstraint, Domain, NormalizedAssumption};

// ── normalize_assumption ──────────────────────────────────────────────────────

/// Extract a [`NormalizedAssumption`] from an [`Assumption`].
///
/// Normalization uses `template_id` as the primary key. Recognized templates
/// are mapped to typed constraints; anything else becomes
/// [`AssumptionConstraint::Opaque`].
pub fn normalize_assumption(assumption: &Assumption) -> NormalizedAssumption {
    let tid = assumption.narrative.template_id;
    let bindings = &assumption.narrative.bindings;

    let constraint = match tid {
        "engine.assumption.positive" => sign_constraint(bindings, assumption, |var| {
            AssumptionConstraint::Positive { var }
        }),
        "engine.assumption.negative" => sign_constraint(bindings, assumption, |var| {
            AssumptionConstraint::Negative { var }
        }),
        "engine.assumption.nonnegative" => sign_constraint(bindings, assumption, |var| {
            AssumptionConstraint::NonNegative { var }
        }),
        "engine.assumption.nonpositive" => sign_constraint(bindings, assumption, |var| {
            AssumptionConstraint::NonPositive { var }
        }),
        "engine.assumption.nonzero" => sign_constraint(bindings, assumption, |var| {
            AssumptionConstraint::NonZero { var }
        }),
        "engine.assumption.in_domain" => normalize_in_domain(bindings, assumption),
        "engine.assumption.greater_than" => bound_constraint(bindings, assumption, |var, bound| {
            AssumptionConstraint::GreaterThan { var, bound }
        }),
        "engine.assumption.at_least" => bound_constraint(bindings, assumption, |var, bound| {
            AssumptionConstraint::AtLeast { var, bound }
        }),
        "engine.assumption.less_than" => bound_constraint(bindings, assumption, |var, bound| {
            AssumptionConstraint::LessThan { var, bound }
        }),
        "engine.assumption.at_most" => bound_constraint(bindings, assumption, |var, bound| {
            AssumptionConstraint::AtMost { var, bound }
        }),
        _ => opaque_from(assumption),
    };

    NormalizedAssumption { constraint }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn get_binding_text(bindings: &[(String, NarrativeValue)], name: &str) -> Option<String> {
    bindings
        .iter()
        .find(|(k, _)| *k == name)
        .and_then(|(_, v)| match v {
            NarrativeValue::Text(s) => Some(s.clone()),
            NarrativeValue::Int(n) => Some(n.to_string()),
            _ => None,
        })
}

fn sign_constraint(
    bindings: &[(String, NarrativeValue)],
    assumption: &Assumption,
    make: impl FnOnce(String) -> AssumptionConstraint,
) -> AssumptionConstraint {
    get_binding_text(bindings, "var").map_or_else(|| opaque_from(assumption), make)
}

fn bound_constraint(
    bindings: &[(String, NarrativeValue)],
    assumption: &Assumption,
    make: impl FnOnce(String, String) -> AssumptionConstraint,
) -> AssumptionConstraint {
    match (
        get_binding_text(bindings, "var"),
        get_binding_text(bindings, "bound"),
    ) {
        (Some(var), Some(bound)) => make(var, bound),
        _ => opaque_from(assumption),
    }
}

fn normalize_in_domain(
    bindings: &[(String, NarrativeValue)],
    assumption: &Assumption,
) -> AssumptionConstraint {
    match (
        get_binding_text(bindings, "var"),
        get_binding_text(bindings, "domain"),
    ) {
        (Some(var), Some(domain_str)) => parse_domain(&domain_str)
            .map(|domain| AssumptionConstraint::InDomain { var, domain })
            .unwrap_or_else(|| opaque_from(assumption)),
        _ => opaque_from(assumption),
    }
}

fn opaque_from(assumption: &Assumption) -> AssumptionConstraint {
    AssumptionConstraint::Opaque {
        normalized_text: assumption.narrative.template_id.to_string(),
    }
}

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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Narrative;

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
        let a = Assumption {
            narrative: Narrative::new("engine.assumption.positive", "x > 0"),
            path: None,
        };
        let na = normalize_assumption(&a);
        assert!(matches!(na.constraint, AssumptionConstraint::Opaque { .. }));
    }
}
