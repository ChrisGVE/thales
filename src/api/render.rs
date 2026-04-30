//! Narrative resolver for [`Response`] objects leaving the crate.
//!
//! Engines populate [`Narrative::fallback_md`] with raw English templates and
//! [`Narrative::template_id`] with the dictionary key. [`render_response`]
//! resolves every narrative against the embedded English dictionary and
//! rewrites `fallback_md` with the substituted Markdown so callers who do not
//! perform their own template lookup still see proper rendered text.

use super::narratives::render_narrative;
use super::response::{Response, ResultValue};

/// Resolve every narrative in a [`Response`] against the English dictionary.
///
/// Mutates each [`Narrative::fallback_md`] in place with the rendered
/// Markdown. Idempotent for unknown template ids: when the dictionary lacks a
/// matching entry the renderer returns the existing `fallback_md` unchanged.
#[must_use]
pub fn render_response(mut response: Response) -> Response {
    for (_, entry) in &mut response.results {
        match &mut entry.value {
            ResultValue::Unsolved { reason } => {
                reason.fallback_md = render_narrative(reason);
            }
            ResultValue::NoSolution { reason, .. } => {
                reason.fallback_md = render_narrative(reason);
            }
            _ => {}
        }
        for step in &mut entry.steps {
            step.narrative.fallback_md = render_narrative(&step.narrative);
        }
    }
    for diagnostic in &mut response.diagnostics {
        diagnostic.narrative.fallback_md = render_narrative(&diagnostic.narrative);
    }
    for assumption in &mut response.assumptions {
        assumption.narrative.fallback_md = render_narrative(&assumption.narrative);
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::diagnostic::{Diagnostic, DiagnosticCode};
    use crate::api::narrative::{Narrative, NarrativeValue};
    use crate::api::response::{
        EngineId, NarratedStep, Response, ResultEntry, ResultKey, ResultShape,
    };
    use crate::numeric::trace::{TechniqueDifficulty, TechniqueTag};

    fn step_with(narrative: Narrative) -> NarratedStep {
        NarratedStep {
            tag: TechniqueTag::Simplification,
            difficulty: TechniqueDifficulty::Elementary,
            narrative,
            path: None,
            input: None,
            output: None,
            unit_trace: None,
        }
    }

    fn entry_with_step(step: NarratedStep) -> ResultEntry {
        ResultEntry {
            value: ResultValue::Symbolic(crate::ast::Expression::Integer(0)),
            structured: None,
            shape: ResultShape::Scalar,
            unit: None,
            steps: vec![step],
            alternatives: Vec::new(),
            engine: EngineId::Simplify,
        }
    }

    #[test]
    fn known_template_is_resolved_in_steps() {
        // step.generic exists in en.json; the resolver should replace the raw
        // fallback string with the dictionary entry.
        let narr = Narrative::new("step.generic", "raw fallback");
        let mut response = Response::default();
        response
            .results
            .push((ResultKey::Single, entry_with_step(step_with(narr))));
        let rendered = render_response(response);
        let body = &rendered.results[0].1.steps[0].narrative.fallback_md;
        assert_ne!(body, "raw fallback");
        assert!(body.contains("Engine step."));
    }

    #[test]
    fn bindings_substitute_into_inline_fallback() {
        // Unknown template id → renderer falls back to fallback_md and
        // substitutes inline placeholders against bindings.
        let narr = Narrative::new("test.inline.binding", "hello {who}")
            .bind("who", NarrativeValue::Text("world".to_string()));
        let mut response = Response::default();
        response
            .results
            .push((ResultKey::Single, entry_with_step(step_with(narr))));
        let rendered = render_response(response);
        assert_eq!(
            rendered.results[0].1.steps[0].narrative.fallback_md,
            "hello world"
        );
    }

    #[test]
    fn unknown_template_keeps_fallback() {
        let narr = Narrative::new("definitely.not.a.template", "raw fallback text");
        let mut response = Response::default();
        response
            .results
            .push((ResultKey::Single, entry_with_step(step_with(narr))));
        let rendered = render_response(response);
        assert_eq!(
            rendered.results[0].1.steps[0].narrative.fallback_md,
            "raw fallback text"
        );
    }

    #[test]
    fn diagnostics_are_rendered() {
        let narr = Narrative::new("command.noop", "raw fallback");
        let response = Response {
            diagnostics: vec![Diagnostic::of(DiagnosticCode::NotImplemented, narr)],
            ..Response::default()
        };
        let rendered = render_response(response);
        let body = &rendered.diagnostics[0].narrative.fallback_md;
        assert_ne!(body, "raw fallback");
    }

    #[test]
    fn unsolved_value_narratives_are_rendered() {
        let reason = Narrative::new("command.noop", "raw fallback");
        let entry = ResultEntry {
            value: ResultValue::Unsolved { reason },
            structured: None,
            shape: ResultShape::Scalar,
            unit: None,
            steps: Vec::new(),
            alternatives: Vec::new(),
            engine: EngineId::Other("test"),
        };
        let mut response = Response::default();
        response.results.push((ResultKey::Single, entry));
        let rendered = render_response(response);
        let value = &rendered.results[0].1.value;
        match value {
            ResultValue::Unsolved { reason } => {
                assert_ne!(reason.fallback_md, "raw fallback");
            }
            _ => panic!("expected Unsolved"),
        }
    }
}
