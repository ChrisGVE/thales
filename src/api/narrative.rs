//! [`Narrative`] — Markdown-templated step description.
//!
//! A narrative is a template identifier plus binding values plus a
//! pre-rendered English Markdown fallback. Clients resolve templates via an
//! external dictionary (for i18n); clients that skip template resolution use
//! the fallback string directly.
//!
//! Bindings support inline math (`{name}` → `$expr$`), block math
//! (`$${name}$$`), theorems, axioms, lemmas, and corollaries.

use crate::Expression;

/// Narrated description of one step or diagnostic.
#[derive(Debug, Clone)]
pub struct Narrative {
    /// Identifier of the template in the narrative dictionary.
    ///
    /// Convention: dot-separated category.subcategory path, e.g.
    /// `"factor.difference-of-squares"`, `"chain-rule"`,
    /// `"domain.extended.r-to-c"`.
    pub template_id: &'static str,
    /// Values bound to template placeholders. Placeholders appear in the
    /// template as `{name}` (inline) or `$${name}$$` (block).
    pub bindings: Vec<(String, NarrativeValue)>,
    /// Pre-rendered English Markdown string. Clients that cannot resolve
    /// templates use this directly.
    pub fallback_md: String,
}

impl Narrative {
    /// Build a narrative with no bindings (template + fallback only).
    pub fn new(template_id: &'static str, fallback_md: impl Into<String>) -> Self {
        Self {
            template_id,
            bindings: Vec::new(),
            fallback_md: fallback_md.into(),
        }
    }

    /// Attach a binding.
    #[must_use]
    pub fn bind(mut self, name: impl Into<String>, value: NarrativeValue) -> Self {
        self.bindings.push((name.into(), value));
        self
    }
}

/// A value bound to a narrative template placeholder.
#[derive(Debug, Clone)]
pub enum NarrativeValue {
    /// Inline-rendered math expression (`$expr$`).
    Expr(Expression),
    /// Block-rendered math expression (`$$expr$$`).
    ExprBlock(Expression),
    /// Free-form text, rendered verbatim.
    Text(String),
    /// Integer literal.
    Int(i64),
    /// Floating-point literal.
    Number(f64),
    /// List of expressions. Useful for theorems quantifying over several
    /// variables, or for rendering a bullet list of conditions.
    ExprList(Vec<Expression>),
    /// Reference to a canonical theorem / axiom / lemma / corollary. The
    /// renderer inlines the statement from the dictionary under this
    /// identifier.
    TheoremRef(TheoremId),
}

/// Identifier for a canonical theorem / axiom / lemma / etc., stored in the
/// theorem dictionary under the enclosed string.
///
/// Strings are dot-separated category paths, e.g.
/// `"axiom.field.distributivity"`, `"theorem.calc.fundamental"`,
/// `"lemma.zorn"`, `"corollary.cauchy.residue"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TheoremId {
    /// A foundational axiom.
    Axiom(String),
    /// A proved theorem.
    Theorem(String),
    /// A lemma supporting a theorem.
    Lemma(String),
    /// A corollary derived from a theorem.
    Corollary(String),
    /// A definition fixing terminology.
    Definition(String),
    /// A notational or numerical convention.
    Convention(String),
}
