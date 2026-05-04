use crate::numeric::trace::{Step, TechniqueTag, Trace};
use crate::numeric::Expr;
use std::sync::Arc;

/// Build a [`Trace`] from an ordered list of textual ODE steps, prepending
/// a classification step when `classify_step` is `Some`.
pub(crate) fn build_ode_trace(
    classify_step: Option<(String, String)>,
    steps: &[String],
    solution_arc: &Arc<Expr>,
) -> Trace {
    let mut trace = Trace::new();

    if let Some((order, ode_type)) = classify_step {
        // Classification of an ODE is an inherently calculus-tier act;
        // pick a concrete first-class tag that maps to Calculus rather
        // than `Custom` (which defaults to Advanced) so difficulty
        // filters keep matching the expected tier.
        let classify_tag = tag_for_ode_type(&ode_type);
        trace.push(
            Step::new(
                classify_tag,
                format!(
                    "order={}, ode_type={}; Classify ODE: {}-order, type = {}",
                    order, ode_type, order, ode_type
                ),
            )
            .with_output(solution_arc.clone()),
        );
    }

    for step_desc in steps {
        let tag = tag_for_ode_method(step_desc);
        trace.push(
            Step::new(tag, format!("method={}; {}", step_desc, step_desc))
                .with_output(solution_arc.clone()),
        );
    }
    trace
}

/// Map the free-form ODE classification string to a `TechniqueTag` at
/// Calculus tier. Falls back to `CharacteristicEquation` for unrecognised
/// constant-coefficient classifications and `SeparationOfVariables`
/// otherwise; both sit at Calculus.
fn tag_for_ode_type(ode_type: &str) -> TechniqueTag {
    let lower = ode_type.to_lowercase();
    if lower.contains("separable") {
        TechniqueTag::SeparationOfVariables
    } else if lower.contains("linear") && !lower.contains("non") {
        TechniqueTag::IntegratingFactor
    } else if lower.contains("constant-coefficient") || lower.contains("characteristic") {
        TechniqueTag::CharacteristicEquation
    } else if lower.contains("non-homogeneous") || lower.contains("undetermined") {
        TechniqueTag::UndeterminedCoefficients
    } else {
        TechniqueTag::SeparationOfVariables
    }
}

/// Pick a concrete `TechniqueTag` for an individual ODE solving step based
/// on the textual step description. All returned tags sit at Calculus tier.
fn tag_for_ode_method(step_desc: &str) -> TechniqueTag {
    let lower = step_desc.to_lowercase();
    if lower.contains("separat") {
        TechniqueTag::SeparationOfVariables
    } else if lower.contains("integrating factor") {
        TechniqueTag::IntegratingFactor
    } else if lower.contains("characteristic") {
        TechniqueTag::CharacteristicEquation
    } else if lower.contains("undetermined") {
        TechniqueTag::UndeterminedCoefficients
    } else if lower.contains("variation of parameters") {
        TechniqueTag::VariationOfParameters
    } else if lower.contains("runge") {
        TechniqueTag::RungeKutta
    } else {
        // Fallback: still at Calculus tier via SeparationOfVariables so
        // profile filters continue to see ODE work as calculus-tier.
        TechniqueTag::SeparationOfVariables
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_ode_difficulty_is_calculus_tier() {
        // The classification tags used by the ODE trace builder are all at
        // Calculus tier so difficulty filters see ODE work consistently.
        for tag in [
            TechniqueTag::SeparationOfVariables,
            TechniqueTag::IntegratingFactor,
            TechniqueTag::CharacteristicEquation,
            TechniqueTag::UndeterminedCoefficients,
            TechniqueTag::VariationOfParameters,
        ] {
            assert_eq!(
                tag.difficulty(),
                crate::numeric::trace::TechniqueDifficulty::Calculus,
                "Expected Calculus tier for {:?}",
                tag
            );
        }
    }
}
