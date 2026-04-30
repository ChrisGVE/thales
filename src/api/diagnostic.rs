//! Diagnostics and assumptions attached to a [`super::Response`].
//!
//! Engines emit [`Diagnostic`] and [`Assumption`] values via the
//! convenience constructors on this module, not by hand-building the
//! structs. Constructors pair each [`DiagnosticCode`] with its intrinsic
//! [`Severity`] so severity is never set inconsistently.

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
    /// A request or command field was accepted but not honored; it had no
    /// effect on this result. The accompanying narrative names the field.
    FieldIgnored,
    /// Engine-specific code not yet promoted to the enum. Carries a stable
    /// string label.
    Other(&'static str),
}

impl DiagnosticCode {
    /// Intrinsic severity associated with this code. Used by
    /// [`Diagnostic::of`] to pair a code with its conventional severity;
    /// engines never override severity.
    #[must_use]
    pub const fn severity(self) -> Severity {
        match self {
            // Informational events — caller may log or ignore.
            DiagnosticCode::DomainNarrowed
            | DiagnosticCode::DomainExtended
            | DiagnosticCode::AssumedPrincipalValue
            | DiagnosticCode::InputRepaired => Severity::Info,

            // Warnings — result was produced but may surprise.
            DiagnosticCode::TimeoutReached
            | DiagnosticCode::IterationBudgetExhausted
            | DiagnosticCode::ConvergenceFailure
            | DiagnosticCode::BranchCutCrossed
            | DiagnosticCode::DivergentSeriesTruncated
            | DiagnosticCode::NoSolutionInDomain
            | DiagnosticCode::FieldIgnored => Severity::Warning,

            // Errors — result is invalid or absent.
            DiagnosticCode::NotImplemented
            | DiagnosticCode::InconsistentDomain
            | DiagnosticCode::DimensionMismatch
            | DiagnosticCode::UndefinedSymbol => Severity::Error,

            // Unknown — default to Warning; engine should promote soon.
            DiagnosticCode::Other(_) => Severity::Warning,
        }
    }
}

impl Diagnostic {
    /// Build a diagnostic with the intrinsic severity of `code`.
    #[must_use]
    pub fn of(code: DiagnosticCode, narrative: Narrative) -> Self {
        Self {
            severity: code.severity(),
            code,
            narrative,
            path: None,
        }
    }

    /// Attach a positional path to this diagnostic.
    #[must_use]
    pub fn at(mut self, path: ExprPath) -> Self {
        self.path = Some(path);
        self
    }

    /// Override the intrinsic severity (rare — prefer to let the code carry
    /// its severity).
    #[must_use]
    pub fn with_severity(mut self, severity: Severity) -> Self {
        self.severity = severity;
        self
    }
}

impl Assumption {
    /// Build an assumption from its narrative.
    #[must_use]
    pub fn new(narrative: Narrative) -> Self {
        Self {
            narrative,
            path: None,
        }
    }

    /// Attach a positional path to this assumption.
    #[must_use]
    pub fn at(mut self, path: ExprPath) -> Self {
        self.path = Some(path);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Narrative;

    #[test]
    fn code_severity_partitioning_info() {
        assert_eq!(DiagnosticCode::DomainNarrowed.severity(), Severity::Info);
        assert_eq!(DiagnosticCode::DomainExtended.severity(), Severity::Info);
        assert_eq!(
            DiagnosticCode::AssumedPrincipalValue.severity(),
            Severity::Info
        );
        assert_eq!(DiagnosticCode::InputRepaired.severity(), Severity::Info);
    }

    #[test]
    fn code_severity_partitioning_warning() {
        assert_eq!(DiagnosticCode::TimeoutReached.severity(), Severity::Warning);
        assert_eq!(
            DiagnosticCode::ConvergenceFailure.severity(),
            Severity::Warning
        );
        assert_eq!(
            DiagnosticCode::NoSolutionInDomain.severity(),
            Severity::Warning
        );
    }

    #[test]
    fn code_severity_partitioning_error() {
        assert_eq!(DiagnosticCode::NotImplemented.severity(), Severity::Error);
        assert_eq!(
            DiagnosticCode::InconsistentDomain.severity(),
            Severity::Error
        );
        assert_eq!(
            DiagnosticCode::DimensionMismatch.severity(),
            Severity::Error
        );
    }

    #[test]
    fn diagnostic_of_pairs_code_with_severity() {
        let narr = Narrative::new("test.code", "test");
        let d = Diagnostic::of(DiagnosticCode::NotImplemented, narr);
        assert_eq!(d.severity, Severity::Error);
        assert_eq!(d.code, DiagnosticCode::NotImplemented);
        assert!(d.path.is_none());
    }

    #[test]
    fn diagnostic_at_attaches_path() {
        let narr = Narrative::new("test.code", "test");
        let d = Diagnostic::of(DiagnosticCode::DomainNarrowed, narr).at(ExprPath::root());
        assert!(d.path.is_some());
    }

    #[test]
    fn assumption_constructors() {
        let narr = Narrative::new("assume.positive", "assumed x > 0");
        let a = Assumption::new(narr).at(ExprPath::root());
        assert!(a.path.is_some());
    }

    #[test]
    fn severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
    }
}
