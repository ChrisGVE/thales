//! Technique trace for numeric engines.
//!
//! A [`Trace`] is an optional sink passed to long-form numeric algorithms
//! (series expansions, residue extraction, reversion, integration, etc.).
//! Each decision point inside an engine pushes a [`Step`] carrying a
//! [`TechniqueTag`] that names the applied technique plus a brief
//! human-readable detail.
//!
//! Callers that care about educational narration own the `Trace` and
//! pass `Some(&mut trace)` into the engine; callers that only need the
//! computed result pass `None` and pay no allocation cost.
//!
//! The tag enum is intentionally engine-oriented, not UI-oriented.
//! Mapping `TechniqueTag → TechniqueDifficulty` lives in the Expression
//! wrapper layer where the difficulty concept already exists
//! (`resolution_path::TechniqueDifficulty`).

use std::sync::Arc;

use super::Expr;

// ── Tag ──────────────────────────────────────────────────────────────────────

/// Named technique applied at a trace step.
///
/// Variants cover the calculus-level engines that emit traces: series
/// expansion, residue / pole classification, integration, limits.
/// Each variant is a single pre-defined label — free-form commentary
/// belongs in [`Step::detail`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TechniqueTag {
    /// Repeated differentiation at the center to build a Taylor series.
    TaylorExpansion,
    /// Laurent series computed via principal + analytic parts at a pole.
    LaurentExpansion,
    /// Asymptotic expansion (e.g. Poincaré-type) at infinity or a boundary.
    AsymptoticExpansion,
    /// Series composition `g(f(x))` via coefficient convolution.
    SeriesComposition,
    /// Inversion of a series via Lagrange reversion (`f^{-1}(y)`).
    LagrangeReversion,
    /// Residue at a singularity via Laurent coefficient or limit formula.
    ResidueTheorem,
    /// Pole order / singularity type classification.
    PoleClassification,
    /// L'Hôpital's rule applied to a `0/0` or `∞/∞` limit.
    LHopitalRule,
    /// Pattern-based antiderivative recognition.
    PatternIntegration,
    /// Risch algorithm verification step on an elementary antiderivative.
    RischVerification,
}

impl TechniqueTag {
    /// Short human-readable label for the tag (stable for UI narration).
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            TechniqueTag::TaylorExpansion => "Taylor expansion",
            TechniqueTag::LaurentExpansion => "Laurent expansion",
            TechniqueTag::AsymptoticExpansion => "Asymptotic expansion",
            TechniqueTag::SeriesComposition => "Series composition",
            TechniqueTag::LagrangeReversion => "Lagrange reversion",
            TechniqueTag::ResidueTheorem => "Residue theorem",
            TechniqueTag::PoleClassification => "Pole classification",
            TechniqueTag::LHopitalRule => "L'Hôpital's rule",
            TechniqueTag::PatternIntegration => "Pattern integration",
            TechniqueTag::RischVerification => "Risch verification",
        }
    }
}

// ── Step ─────────────────────────────────────────────────────────────────────

/// One recorded decision point in an engine run.
///
/// `input` and `output` are optional — leaf steps (e.g. a constant
/// coefficient) need only the tag and a description; substantive steps
/// attach the canonical `Arc<Expr>` operands for downstream narration.
#[derive(Debug, Clone)]
pub struct Step {
    /// Applied technique.
    pub tag: TechniqueTag,
    /// Free-form detail: "coefficient a_3", "pole order 2 at z=0", etc.
    pub detail: String,
    /// Optional input expression (before applying the technique).
    pub input: Option<Arc<Expr>>,
    /// Optional output expression (result of the technique).
    pub output: Option<Arc<Expr>>,
}

impl Step {
    /// Build a step with the tag and detail only. Prefer this when the
    /// operands are not meaningful (leaf coefficients, pure classifications).
    #[must_use]
    pub fn new(tag: TechniqueTag, detail: impl Into<String>) -> Self {
        Step {
            tag,
            detail: detail.into(),
            input: None,
            output: None,
        }
    }

    /// Attach the input expression to this step.
    #[must_use]
    pub fn with_input(mut self, input: Arc<Expr>) -> Self {
        self.input = Some(input);
        self
    }

    /// Attach the output expression to this step.
    #[must_use]
    pub fn with_output(mut self, output: Arc<Expr>) -> Self {
        self.output = Some(output);
        self
    }
}

// ── Trace ────────────────────────────────────────────────────────────────────

/// Ordered list of engine decision points.
///
/// Allocated by the caller that wants narration; engines push [`Step`]
/// values through a `&mut Trace` borrow. A `None` `Option<&mut Trace>`
/// means "no narration requested" — engines must skip allocation entirely
/// in that branch.
#[derive(Debug, Clone, Default)]
pub struct Trace {
    steps: Vec<Step>,
}

impl Trace {
    /// New empty trace.
    #[must_use]
    pub fn new() -> Self {
        Trace { steps: Vec::new() }
    }

    /// Append a step.
    pub fn push(&mut self, step: Step) {
        self.steps.push(step);
    }

    /// Recorded steps in insertion order.
    #[must_use]
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// Number of recorded steps.
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// True when no steps have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// Helper for engines: push a step onto an optional trace, doing nothing
/// when the caller opted out.
///
/// Engine arms use this at every decision point:
///
/// ```ignore
/// use thales::numeric::trace::{record, Step, TechniqueTag};
/// fn classify(trace: Option<&mut Trace>) {
///     record(trace, Step::new(TechniqueTag::PoleClassification, "simple pole at 0"));
/// }
/// ```
pub fn record(trace: Option<&mut Trace>, step: Step) {
    if let Some(t) = trace {
        t.push(step);
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_trace() {
        let t = Trace::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert!(t.steps().is_empty());
    }

    #[test]
    fn push_and_read() {
        let mut t = Trace::new();
        t.push(Step::new(TechniqueTag::TaylorExpansion, "order 3 at 0"));
        t.push(Step::new(TechniqueTag::ResidueTheorem, "simple pole").with_output(Expr::int(5)));
        assert_eq!(t.len(), 2);
        assert_eq!(t.steps()[0].tag, TechniqueTag::TaylorExpansion);
        assert_eq!(t.steps()[0].detail, "order 3 at 0");
        assert!(t.steps()[0].input.is_none());
        assert!(t.steps()[1].output.is_some());
    }

    #[test]
    fn record_some_pushes() {
        let mut t = Trace::new();
        record(
            Some(&mut t),
            Step::new(TechniqueTag::LHopitalRule, "0/0 indeterminate"),
        );
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn record_none_is_noop() {
        let no_trace: Option<&mut Trace> = None;
        // Must compile and do nothing.
        record(no_trace, Step::new(TechniqueTag::LHopitalRule, "ignored"));
    }

    #[test]
    fn tag_labels_are_stable() {
        assert_eq!(TechniqueTag::TaylorExpansion.label(), "Taylor expansion");
        assert_eq!(TechniqueTag::ResidueTheorem.label(), "Residue theorem");
        assert_eq!(
            TechniqueTag::AsymptoticExpansion.label(),
            "Asymptotic expansion"
        );
    }

    #[test]
    fn step_builder_chain() {
        let s = Step::new(TechniqueTag::SeriesComposition, "degree 2")
            .with_input(Expr::int(1))
            .with_output(Expr::int(2));
        assert!(s.input.is_some());
        assert!(s.output.is_some());
    }
}
