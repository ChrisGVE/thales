//! Diagnostics and assumptions attached to a [`super::Response`].

use super::{ExprPath, Narrative};

/// Informational, warning, or error diagnostic.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// Importance level.
    pub severity: Severity,
    /// Stable code for programmatic handling.
    pub code: DiagnosticCode,
    /// Narrative description.
    pub narrative: Narrative,
    /// Optional position into the input expression tree.
    pub path: Option<ExprPath>,
}

/// Stable assumption the engine made.
///
/// Distinct from [`Diagnostic`] because assumptions are load-bearing premises
/// rather than events — callers that want to reproduce the result must
/// accept every assumption.
#[derive(Debug, Clone)]
pub struct Assumption {
    /// Narrative describing the assumption (e.g. "assumed x > 0",
    /// "used principal branch of log").
    pub narrative: Narrative,
    /// Optional position into the input tree where the assumption applies.
    pub path: Option<ExprPath>,
}

/// Diagnostic severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational — no action required.
    Info,
    /// Warning — engine succeeded but the caller should review.
    Warning,
    /// Error — engine failed to produce a valid result for this path.
    Error,
}

/// Stable codes for [`Diagnostic`]. Callers match on these to react
/// programmatically.
///
/// The enum is `#[non_exhaustive]`: new codes may appear in minor releases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DiagnosticCode {
    /// The requested command variant is not yet implemented. Typical in
    /// v0.8.1 skeleton builds.
    NotImplemented,
    /// Domain constraints were narrowed by intersection. Information only;
    /// result is still valid.
    DomainNarrowed,
    /// Domain was extended from the ambient (e.g. ℝ → ℂ) because the result
    /// lies outside the ambient. Information only.
    DomainExtended,
    /// Domain declarations conflict and the policy is `ErrorOnMismatch`.
    InconsistentDomain,
    /// Time budget exhausted. Result is partial.
    TimeoutReached,
    /// Iteration budget exhausted. Result is partial.
    IterationBudgetExhausted,
    /// A numeric method failed to converge within tolerance.
    ConvergenceFailure,
    /// Assumed principal branch of a multi-valued function (log, sqrt,
    /// arcsin, …).
    AssumedPrincipalValue,
    /// Branch cut of a multi-valued function was crossed during the
    /// computation; a different branch may give a different result.
    BranchCutCrossed,
    /// An asymptotic or divergent series was truncated without full
    /// convergence justification.
    DivergentSeriesTruncated,
    /// No solution was found in the requested domain. A solution may exist
    /// in a wider domain; see the accompanying narrative.
    NoSolutionInDomain,
    /// A dimensional inconsistency was detected during a unit-aware
    /// computation (e.g. adding meters to seconds).
    DimensionMismatch,
    /// An undefined or unbound symbol was encountered.
    UndefinedSymbol,
    /// A malformed input was repaired heuristically; see narrative.
    InputRepaired,
    /// Engine-specific code not yet promoted to the enum. Carries a stable
    /// string label.
    Other(&'static str),
}
