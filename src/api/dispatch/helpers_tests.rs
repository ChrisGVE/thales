use super::*;
use crate::api::narrative::NarrativeValue;
use crate::numeric::trace::{Step, TechniqueTag};

// ── template_id_for_tag spot checks ─────────────────────────────────────

#[test]
fn fast_template_id_add_both_sides() {
    assert_eq!(
        template_id_for_tag(TechniqueTag::AddBothSides),
        "step.add_both_sides"
    );
}

#[test]
fn fast_template_id_power_rule() {
    assert_eq!(
        template_id_for_tag(TechniqueTag::PowerRule),
        "step.power_rule"
    );
}

#[test]
fn fast_template_id_chain_rule() {
    assert_eq!(
        template_id_for_tag(TechniqueTag::ChainRule),
        "step.chain_rule"
    );
}

#[test]
fn fast_template_id_taylor_expansion() {
    assert_eq!(
        template_id_for_tag(TechniqueTag::TaylorExpansion),
        "step.taylor_expansion"
    );
}

#[test]
fn fast_template_id_separation_of_variables() {
    assert_eq!(
        template_id_for_tag(TechniqueTag::SeparationOfVariables),
        "step.separation_of_variables"
    );
}

#[test]
fn fast_template_id_l_hopital_rule() {
    assert_eq!(
        template_id_for_tag(TechniqueTag::LHopitalRule),
        "step.l_hopital_rule"
    );
}

#[test]
fn fast_template_id_numerical_approximation() {
    assert_eq!(
        template_id_for_tag(TechniqueTag::NumericalApproximation),
        "step.numerical_approximation"
    );
}

#[test]
fn fast_template_id_principal_branch() {
    assert_eq!(
        template_id_for_tag(TechniqueTag::PrincipalBranch),
        "step.principal_branch"
    );
}

#[test]
fn fast_template_id_conjugation() {
    assert_eq!(
        template_id_for_tag(TechniqueTag::Conjugation),
        "step.conjugation"
    );
}

#[test]
fn fast_template_id_custom_falls_back_to_generic() {
    assert_eq!(
        template_id_for_tag(TechniqueTag::Custom("foo")),
        "step.generic"
    );
}

// ── geometry template_id spot checks ────────────────────────────────────

#[test]
fn fast_template_id_geom_distance() {
    assert_eq!(
        template_id_for_tag(TechniqueTag::GeomDistance),
        "step.geom_distance"
    );
}

#[test]
fn fast_template_id_geom_intersection() {
    assert_eq!(
        template_id_for_tag(TechniqueTag::GeomIntersection),
        "step.geom_intersection"
    );
}

#[test]
fn fast_template_id_geom_tangent() {
    assert_eq!(
        template_id_for_tag(TechniqueTag::GeomTangent),
        "step.geom_tangent"
    );
}

#[test]
fn fast_template_id_geom_normal() {
    assert_eq!(
        template_id_for_tag(TechniqueTag::GeomNormal),
        "step.geom_normal"
    );
}

#[test]
fn fast_template_id_geom_curvature() {
    assert_eq!(
        template_id_for_tag(TechniqueTag::GeomCurvature),
        "step.geom_curvature"
    );
}

#[test]
fn fast_template_id_geom_transform() {
    assert_eq!(
        template_id_for_tag(TechniqueTag::GeomTransform),
        "step.geom_transform"
    );
}

#[test]
fn fast_template_id_geom_exterior_derivative() {
    assert_eq!(
        template_id_for_tag(TechniqueTag::GeomExteriorDerivative),
        "step.geom_exterior_derivative"
    );
}

// ── build_step_narrative tests ───────────────────────────────────────────

#[test]
fn fast_build_step_narrative_structured_detail() {
    let step = Step::new(TechniqueTag::Substitution, "f=x;g=sin(x)");
    let narr = build_step_narrative("step.substitution", &step);
    let keys: Vec<&str> = narr.bindings.iter().map(|(k, _)| k.as_str()).collect();
    assert!(keys.contains(&"f"), "expected binding 'f'");
    assert!(keys.contains(&"g"), "expected binding 'g'");
}

#[test]
fn fast_build_step_narrative_plain_detail_no_extra_bindings() {
    let step = Step::new(TechniqueTag::Simplification, "simple text");
    let narr = build_step_narrative("step.simplification", &step);
    // No input/output on step, no '=' in detail → no bindings at all.
    assert!(
        narr.bindings.is_empty(),
        "expected no bindings, got {:?}",
        narr.bindings
            .iter()
            .map(|(k, _)| k.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn fast_build_step_narrative_protects_input_output_keys() {
    use crate::numeric::Expr;
    use std::sync::Arc;

    // Build a step with both input/output expressions AND a detail string
    // that tries to overwrite "input"/"output" while also setting "real".
    let expr_arc = Arc::new(Expr::Float(1.0));
    let step = Step::new(TechniqueTag::Substitution, "input=bad;output=bad;real=good")
        .with_input(expr_arc.clone())
        .with_output(expr_arc);

    let narr = build_step_narrative("step.substitution", &step);

    let mut found_input_expr = false;
    let mut found_output_expr = false;
    let mut found_real_text = false;

    for (key, val) in &narr.bindings {
        match key.as_str() {
            "input" => {
                assert!(
                    matches!(val, NarrativeValue::Expr(_)),
                    "'input' binding must be Expr, not text"
                );
                found_input_expr = true;
            }
            "output" => {
                assert!(
                    matches!(val, NarrativeValue::Expr(_)),
                    "'output' binding must be Expr, not text"
                );
                found_output_expr = true;
            }
            "real" => {
                assert!(
                    matches!(val, NarrativeValue::Text(t) if t == "good"),
                    "'real' binding must be Text(\"good\")"
                );
                found_real_text = true;
            }
            other => panic!("unexpected binding key: {other}"),
        }
    }

    assert!(found_input_expr, "missing 'input' Expr binding");
    assert!(found_output_expr, "missing 'output' Expr binding");
    assert!(found_real_text, "missing 'real' Text binding");
}
