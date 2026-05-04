//! Narrative + theorem dictionary for the [`super::Response`] surface.
//!
//! Two static dictionaries, parsed once on first access:
//!
//! - `en.json` — English Markdown template bodies keyed by
//!   [`crate::api::Narrative::template_id`] (dot-separated paths like
//!   `command.noop`, `step.generic`).
//! - `theorems.json` — canonical statements keyed by
//!   [`crate::api::TheoremId`] variant strings, inlined when a
//!   [`NarrativeValue::TheoremRef`] appears in a binding list.
//!
//! Template resolution is a best-effort lookup: when the key is absent,
//! callers fall back to [`Narrative::fallback_md`], which the dispatcher
//! always populates.

use std::collections::HashMap;
use std::sync::OnceLock;

use super::narrative::{Narrative, NarrativeValue, TheoremId};

/// English narrative template dictionary.
static EN_DICT: OnceLock<HashMap<String, String>> = OnceLock::new();

/// Theorem dictionary.
static THEOREM_DICT: OnceLock<HashMap<String, String>> = OnceLock::new();

fn en_dict() -> &'static HashMap<String, String> {
    EN_DICT.get_or_init(|| {
        serde_json::from_str(include_str!("en.json"))
            .expect("embedded en.json narrative dictionary must parse")
    })
}

fn theorem_dict() -> &'static HashMap<String, String> {
    THEOREM_DICT.get_or_init(|| {
        serde_json::from_str(include_str!("theorems.json"))
            .expect("embedded theorems.json dictionary must parse")
    })
}

/// Resolve a template id against the English dictionary. Returns the
/// template body when present, or `None` when the caller should fall
/// back to [`Narrative::fallback_md`].
#[must_use]
pub fn resolve_template(template_id: &str) -> Option<&'static str> {
    en_dict().get(template_id).map(String::as_str)
}

/// Resolve a theorem identifier against the theorem dictionary.
#[must_use]
pub fn resolve_theorem(id: &TheoremId) -> Option<&'static str> {
    let key = theorem_key(id);
    theorem_dict().get(&key).map(String::as_str)
}

fn theorem_key(id: &TheoremId) -> String {
    match id {
        TheoremId::Axiom(s) => format!("axiom.{}", s),
        TheoremId::Theorem(s) => format!("theorem.{}", s),
        TheoremId::Lemma(s) => format!("lemma.{}", s),
        TheoremId::Corollary(s) => format!("corollary.{}", s),
        TheoremId::Definition(s) => format!("definition.{}", s),
        TheoremId::Convention(s) => format!("convention.{}", s),
    }
}

/// Render a [`Narrative`] against the English dictionary.
///
/// Substitutes `{name}` placeholders inline and `$${name}$$` placeholders
/// as block math. When the template is absent, returns the raw
/// [`Narrative::fallback_md`] string unchanged.
#[must_use]
pub fn render_narrative(narr: &Narrative) -> String {
    let template = resolve_template(narr.template_id).unwrap_or(narr.fallback_md.as_str());
    let mut out = template.to_string();
    for (name, value) in &narr.bindings {
        let rendered = render_binding(value);
        let inline = format!("{{{}}}", name);
        let block = format!("$${}$$", name);
        out = out.replace(&block, &format!("$$ {} $$", rendered));
        out = out.replace(&inline, &rendered);
    }
    out
}

fn render_binding(value: &NarrativeValue) -> String {
    match value {
        NarrativeValue::Expr(e) => format!("${}$", e),
        NarrativeValue::ExprBlock(e) => format!("{}", e),
        NarrativeValue::Text(t) => t.clone(),
        NarrativeValue::Int(n) => n.to_string(),
        NarrativeValue::Number(n) => n.to_string(),
        NarrativeValue::ExprList(list) => list
            .iter()
            .map(|e| format!("${}$", e))
            .collect::<Vec<_>>()
            .join(", "),
        NarrativeValue::TheoremRef(id) => resolve_theorem(id)
            .map(String::from)
            .unwrap_or_else(|| format!("[theorem:{}]", theorem_key(id))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Expression;

    #[test]
    fn en_dict_parses() {
        let dict = en_dict();
        assert!(dict.contains_key("command.noop"));
        assert!(dict.contains_key("step.generic"));
    }

    #[test]
    fn fast_step_equation_ops_keys_present() {
        let dict = en_dict();
        let keys = [
            "step.add_both_sides",
            "step.apply_function",
            "step.divide_both_sides",
            "step.isolation",
            "step.move_term",
            "step.multiply_both_sides",
            "step.power_both_sides",
            "step.root_both_sides",
            "step.subtract_both_sides",
        ];
        for key in &keys {
            assert!(dict.contains_key(*key), "missing key: {key}");
        }
    }

    #[test]
    fn fast_step_algebra_keys_present() {
        let dict = en_dict();
        let keys = [
            "step.cancellation",
            "step.combine_fractions",
            "step.combining_like_terms",
            "step.conjugation",
            "step.expansion",
            "step.factoring",
            "step.partial_fraction_decomp",
            "step.rationalization",
            "step.simplification",
            "step.substitution",
        ];
        for key in &keys {
            assert!(dict.contains_key(*key), "missing key: {key}");
        }
    }

    #[test]
    fn theorem_dict_parses() {
        let dict = theorem_dict();
        assert!(dict.contains_key("theorem.calc.fundamental"));
        assert!(dict.contains_key("axiom.field.distributivity"));
    }

    #[test]
    fn resolve_known_template_returns_some() {
        assert!(resolve_template("command.noop").is_some());
        assert!(resolve_template("step.generic").is_some());
    }

    #[test]
    fn resolve_unknown_template_returns_none() {
        assert!(resolve_template("not.a.template.id").is_none());
    }

    #[test]
    fn resolve_theorem_known() {
        let id = TheoremId::Theorem("calc.fundamental".to_string());
        assert!(resolve_theorem(&id).is_some());
    }

    #[test]
    fn resolve_theorem_unknown_returns_none() {
        let id = TheoremId::Theorem("not.a.theorem".to_string());
        assert!(resolve_theorem(&id).is_none());
    }

    #[test]
    fn render_falls_back_when_template_missing() {
        let narr = Narrative::new("not.a.template", "raw fallback text");
        assert_eq!(render_narrative(&narr), "raw fallback text");
    }

    #[test]
    fn render_substitutes_inline_bindings() {
        // Use an inline template carried as fallback_md: resolve_template
        // returns None, so render_narrative uses the fallback verbatim and
        // substitutes placeholders against it.
        let narr = Narrative::new("test.inline", "differentiate {expr} with respect to {var}")
            .bind("expr", NarrativeValue::Expr(Expression::Integer(42)))
            .bind("var", NarrativeValue::Text("x".to_string()));
        let out = render_narrative(&narr);
        assert!(out.contains("$42$"));
        assert!(out.contains("x"));
    }

    #[test]
    fn render_substitutes_block_math() {
        let narr = Narrative::new("test.block", "see $$ans$$")
            .bind("ans", NarrativeValue::ExprBlock(Expression::Integer(7)));
        let out = render_narrative(&narr);
        assert!(out.contains("$$ 7 $$"));
    }

    #[test]
    fn render_theorem_ref_inlines_canonical_statement() {
        let narr = Narrative::new("test.theorem", "apply {th}").bind(
            "th",
            NarrativeValue::TheoremRef(TheoremId::Theorem("calc.fundamental".to_string())),
        );
        let out = render_narrative(&narr);
        assert!(out.contains("Fundamental Theorem of Calculus"));
    }
}
