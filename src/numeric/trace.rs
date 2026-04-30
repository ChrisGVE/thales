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
//! Mapping `TechniqueTag → TechniqueDifficulty` is provided by
//! [`TechniqueTag::difficulty`].

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::Expr;

// ── Difficulty ───────────────────────────────────────────────────────────────

/// Intrinsic difficulty of a mathematical technique.
///
/// Classification reflects the kind of mathematics involved, not the number
/// of steps. A single integration step is harder than ten elementary
/// algebra steps. Each tier maps to an approximate educational level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TechniqueDifficulty {
    /// Add, subtract, multiply, divide both sides; move terms; isolate variable.
    /// Educational level: middle school.
    Elementary = 1,
    /// Square/cube both sides, take nth root, apply exponent rules, logarithms.
    /// Educational level: high school algebra.
    PowerAndRoots = 2,
    /// Factor, expand, complete the square, quadratic formula, combine/partial fractions.
    /// Educational level: pre-calculus.
    AlgebraicManip = 3,
    /// Trig inversion (arcsin etc.), trig identities, hyperbolic functions, substitution.
    /// Educational level: trigonometry.
    Transcendental = 4,
    /// Differentiate, integrate, integration by parts, u-substitution, ODE, limits.
    /// Educational level: calculus.
    Calculus = 5,
    /// Matrix operations, series expansion, numerical methods, special functions,
    /// Laplace/Fourier transforms, tensors.
    /// Educational level: university.
    Advanced = 6,
}

impl std::fmt::Display for TechniqueDifficulty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Elementary => write!(f, "elementary"),
            Self::PowerAndRoots => write!(f, "power/roots"),
            Self::AlgebraicManip => write!(f, "algebraic"),
            Self::Transcendental => write!(f, "transcendental"),
            Self::Calculus => write!(f, "calculus"),
            Self::Advanced => write!(f, "advanced"),
        }
    }
}

impl PartialOrd for TechniqueDifficulty {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TechniqueDifficulty {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}

// ── Tag ──────────────────────────────────────────────────────────────────────

/// Named technique applied at a trace step.
///
/// Variants cover every technique the codebase emits — from elementary
/// equation manipulation through calculus, series, and matrix operations.
/// Each variant is a single pre-defined label; free-form commentary belongs
/// in [`Step::detail`].
///
/// The enum covers every technique the codebase emits. The mapping
/// [`TechniqueTag::difficulty`] translates each tag to the educational
/// level at which it is normally introduced via
/// [`TechniqueDifficulty`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TechniqueTag {
    // ── Both-sides equation operations ───────────────────────────────────
    /// Add the same expression to both sides.
    AddBothSides,
    /// Subtract the same expression from both sides.
    SubtractBothSides,
    /// Multiply both sides by the same expression.
    MultiplyBothSides,
    /// Divide both sides by the same expression.
    DivideBothSides,
    /// Raise both sides to the same power.
    PowerBothSides,
    /// Take the same root of both sides.
    RootBothSides,
    /// Apply the same named function to both sides (log, exp, sin, …).
    ApplyFunction,

    // ── Algebraic manipulation ───────────────────────────────────────────
    /// Canonical simplification.
    Simplification,
    /// Distribute products / expand powers.
    Expansion,
    /// Factor expression.
    Factoring,
    /// Combine fractions under a common denominator.
    CombineFractions,
    /// Cancel common factors.
    Cancellation,
    /// Combine like terms.
    CombiningLikeTerms,
    /// Rationalise a denominator.
    Rationalization,
    /// Form the complex conjugate.
    Conjugation,
    /// Partial fraction decomposition.
    PartialFractionDecomp,
    /// Isolate a variable on one side.
    Isolation,
    /// Move a term across the equals sign.
    MoveTerm,

    // ── Substitution ─────────────────────────────────────────────────────
    /// Variable substitution (u-sub / change of variable / bind value).
    Substitution,

    // ── Identities ───────────────────────────────────────────────────────
    /// Apply a generic named identity (unspecified category).
    ApplyIdentity,
    /// Apply a trigonometric identity (Pythagorean, double-angle, sum-to-product, …).
    TrigIdentity,
    /// Apply a logarithmic identity.
    LogIdentity,
    /// Apply an exponential identity.
    ExpIdentity,
    /// Apply a hyperbolic identity.
    HyperbolicIdentity,
    /// Euler's formula `e^(iθ) = cos θ + i sin θ`.
    EulerFormula,
    /// De Moivre's formula `(cos θ + i sin θ)^n = cos nθ + i sin nθ`.
    DeMoivre,

    // ── Quadratic / polynomial techniques ────────────────────────────────
    /// Quadratic formula `x = (−b ± √(b²−4ac)) / 2a`.
    QuadraticFormula,
    /// Complete the square.
    CompleteTheSquare,
    /// Rational-root theorem for polynomial roots.
    RationalRootTheorem,
    /// Synthetic division for polynomial factorisation.
    SyntheticDivision,

    // ── Higher-dimensional calculus ──────────────────────────────────────
    /// Total derivative via chain rule over declared dependencies.
    TotalDifferential,
    /// Divergence of a vector field `∇·F`.
    Divergence,
    /// Curl of a 3-D vector field `∇×F`.
    Curl,
    /// Laplacian `∇²f = Σ ∂²f/∂xᵢ²`.
    Laplacian,
    /// Jacobian matrix `J[i][j] = ∂fᵢ/∂xⱼ`.
    Jacobian,
    /// Hessian matrix `H[i][j] = ∂²f/(∂xᵢ ∂xⱼ)`.
    Hessian,
    /// Directional derivative `∇f · v̂`.
    DirectionalDerivative,

    // ── Special functions ────────────────────────────────────────────────
    /// Special-function evaluation step (Gamma, Beta, Erf, Erfc, …).
    SpecialFunction,

    // ── Differentiation rules ────────────────────────────────────────────
    /// Power rule `d/dx xⁿ = n xⁿ⁻¹`.
    PowerRule,
    /// Product rule `(fg)' = f'g + fg'`.
    ProductRule,
    /// Quotient rule `(f/g)' = (f'g − fg') / g²`.
    QuotientRule,
    /// Chain rule `(f ∘ g)' = f'(g) · g'`.
    ChainRule,
    /// Implicit differentiation of an implicitly defined relation.
    ImplicitDifferentiation,
    /// Logarithmic differentiation.
    LogarithmicDifferentiation,

    // ── Integration techniques ───────────────────────────────────────────
    /// Pattern-based antiderivative recognition.
    PatternIntegration,
    /// Integration by parts `∫ u dv = uv − ∫ v du`.
    IntegrationByParts,
    /// u-substitution in an integral.
    USubstitution,
    /// Trigonometric substitution for √(a²−x²), √(a²+x²), √(x²−a²).
    TrigSubstitution,
    /// Partial-fraction decomposition applied as an integration step.
    PartialFractionIntegration,
    /// Risch algorithm verification step on an elementary antiderivative.
    RischVerification,

    // ── Limits ───────────────────────────────────────────────────────────
    /// L'Hôpital's rule applied to a `0/0` or `∞/∞` limit.
    LHopitalRule,
    /// Squeeze / sandwich theorem.
    SqueezeTheorem,
    /// Limit computed by algebraic manipulation (factor out, conjugate, …).
    LimitAlgebraic,

    // ── Series expansions ────────────────────────────────────────────────
    /// Repeated differentiation at the center to build a Taylor series.
    TaylorExpansion,
    /// Laurent series computed via principal + analytic parts at a pole.
    LaurentExpansion,
    /// Asymptotic expansion (Poincaré-type) at infinity or a boundary.
    AsymptoticExpansion,
    /// Series composition `g(f(x))` via coefficient convolution.
    SeriesComposition,
    /// Inversion of a series via Lagrange reversion (`f^{-1}(y)`).
    LagrangeReversion,
    /// Puiseux series (fractional-power Laurent) around a branch point.
    PuiseuxExpansion,
    /// Frobenius method for series solutions of linear ODEs near a regular
    /// singular point.
    FrobeniusMethod,
    /// Padé approximant construction from a Taylor series.
    PadeApproximant,
    /// WKB (Wentzel-Kramers-Brillouin) semiclassical approximation.
    WkbApproximation,

    // ── Residues and singularities ───────────────────────────────────────
    /// Residue at a singularity via Laurent coefficient or limit formula.
    ResidueTheorem,
    /// Pole order / singularity type classification.
    PoleClassification,

    // ── Transforms ───────────────────────────────────────────────────────
    /// Fourier series coefficient computation.
    FourierSeries,

    // ── ODE techniques ───────────────────────────────────────────────────
    /// Separation of variables for first-order ODEs.
    SeparationOfVariables,
    /// Integrating factor for linear first-order ODEs.
    IntegratingFactor,
    /// Characteristic equation for constant-coefficient linear ODEs.
    CharacteristicEquation,
    /// Method of undetermined coefficients for particular solutions.
    UndeterminedCoefficients,
    /// Variation of parameters for particular solutions.
    VariationOfParameters,
    /// Runge-Kutta numeric ODE integrator.
    RungeKutta,

    // ── Matrix / linear algebra ──────────────────────────────────────────
    /// Gaussian elimination on a linear system.
    GaussianElimination,
    /// Matrix inversion.
    MatrixInverse,
    /// LU decomposition.
    LuDecomposition,
    /// QR decomposition.
    QrDecomposition,
    /// Eigenvalue computation via characteristic polynomial or numeric
    /// iteration.
    Eigendecomposition,
    /// Determinant computation.
    Determinant,

    // ── Numerical fallback ───────────────────────────────────────────────
    /// A numeric approximation replaced a symbolic result.
    NumericalApproximation,
    /// Newton-Raphson iteration.
    NewtonRaphson,
    /// Bisection iteration.
    Bisection,
    /// Brent's method iteration.
    Brent,
    /// Secant-method iteration.
    Secant,

    // ── Domain / diagnostic events ───────────────────────────────────────
    /// Domain narrowing via intersection with user-declared or inferred
    /// constraints.
    DomainNarrowing,
    /// Domain extension (e.g. `ℝ → ℂ` on negative discriminant).
    DomainExtension,
    /// Assumption of a principal branch of a multi-valued function.
    PrincipalBranch,

    // ── Catch-all ─────────────────────────────────────────────────────────
    /// Engine-specific technique carried as a stable string label. Use only
    /// when no existing variant fits; prefer adding a first-class variant.
    Custom(&'static str),
}

impl TechniqueTag {
    /// Short human-readable label for the tag (stable for UI narration).
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            TechniqueTag::AddBothSides => "Add to both sides",
            TechniqueTag::SubtractBothSides => "Subtract from both sides",
            TechniqueTag::MultiplyBothSides => "Multiply both sides",
            TechniqueTag::DivideBothSides => "Divide both sides",
            TechniqueTag::PowerBothSides => "Raise both sides to a power",
            TechniqueTag::RootBothSides => "Take the root of both sides",
            TechniqueTag::ApplyFunction => "Apply function to both sides",
            TechniqueTag::Simplification => "Simplification",
            TechniqueTag::Expansion => "Expansion",
            TechniqueTag::Factoring => "Factoring",
            TechniqueTag::CombineFractions => "Combine fractions",
            TechniqueTag::Cancellation => "Cancellation",
            TechniqueTag::CombiningLikeTerms => "Combine like terms",
            TechniqueTag::Rationalization => "Rationalize denominator",
            TechniqueTag::Conjugation => "Complex conjugation",
            TechniqueTag::PartialFractionDecomp => "Partial fractions",
            TechniqueTag::Isolation => "Isolate variable",
            TechniqueTag::MoveTerm => "Move term",
            TechniqueTag::Substitution => "Substitution",
            TechniqueTag::ApplyIdentity => "Apply identity",
            TechniqueTag::TrigIdentity => "Trigonometric identity",
            TechniqueTag::LogIdentity => "Logarithmic identity",
            TechniqueTag::ExpIdentity => "Exponential identity",
            TechniqueTag::HyperbolicIdentity => "Hyperbolic identity",
            TechniqueTag::EulerFormula => "Euler's formula",
            TechniqueTag::DeMoivre => "De Moivre's formula",
            TechniqueTag::QuadraticFormula => "Quadratic formula",
            TechniqueTag::CompleteTheSquare => "Complete the square",
            TechniqueTag::RationalRootTheorem => "Rational root theorem",
            TechniqueTag::SyntheticDivision => "Synthetic division",
            TechniqueTag::TotalDifferential => "Total differential",
            TechniqueTag::Divergence => "Divergence",
            TechniqueTag::Curl => "Curl",
            TechniqueTag::Laplacian => "Laplacian",
            TechniqueTag::Jacobian => "Jacobian",
            TechniqueTag::Hessian => "Hessian",
            TechniqueTag::DirectionalDerivative => "Directional derivative",
            TechniqueTag::SpecialFunction => "Special function",
            TechniqueTag::PowerRule => "Power rule",
            TechniqueTag::ProductRule => "Product rule",
            TechniqueTag::QuotientRule => "Quotient rule",
            TechniqueTag::ChainRule => "Chain rule",
            TechniqueTag::ImplicitDifferentiation => "Implicit differentiation",
            TechniqueTag::LogarithmicDifferentiation => "Logarithmic differentiation",
            TechniqueTag::PatternIntegration => "Pattern integration",
            TechniqueTag::IntegrationByParts => "Integration by parts",
            TechniqueTag::USubstitution => "u-substitution",
            TechniqueTag::TrigSubstitution => "Trigonometric substitution",
            TechniqueTag::PartialFractionIntegration => "Partial fractions for integration",
            TechniqueTag::RischVerification => "Risch verification",
            TechniqueTag::LHopitalRule => "L'Hôpital's rule",
            TechniqueTag::SqueezeTheorem => "Squeeze theorem",
            TechniqueTag::LimitAlgebraic => "Algebraic limit",
            TechniqueTag::TaylorExpansion => "Taylor expansion",
            TechniqueTag::LaurentExpansion => "Laurent expansion",
            TechniqueTag::AsymptoticExpansion => "Asymptotic expansion",
            TechniqueTag::SeriesComposition => "Series composition",
            TechniqueTag::LagrangeReversion => "Lagrange reversion",
            TechniqueTag::PuiseuxExpansion => "Puiseux expansion",
            TechniqueTag::FrobeniusMethod => "Frobenius method",
            TechniqueTag::PadeApproximant => "Padé approximant",
            TechniqueTag::WkbApproximation => "WKB approximation",
            TechniqueTag::ResidueTheorem => "Residue theorem",
            TechniqueTag::PoleClassification => "Pole classification",
            TechniqueTag::FourierSeries => "Fourier series",
            TechniqueTag::SeparationOfVariables => "Separation of variables",
            TechniqueTag::IntegratingFactor => "Integrating factor",
            TechniqueTag::CharacteristicEquation => "Characteristic equation",
            TechniqueTag::UndeterminedCoefficients => "Undetermined coefficients",
            TechniqueTag::VariationOfParameters => "Variation of parameters",
            TechniqueTag::RungeKutta => "Runge-Kutta",
            TechniqueTag::GaussianElimination => "Gaussian elimination",
            TechniqueTag::MatrixInverse => "Matrix inverse",
            TechniqueTag::LuDecomposition => "LU decomposition",
            TechniqueTag::QrDecomposition => "QR decomposition",
            TechniqueTag::Eigendecomposition => "Eigendecomposition",
            TechniqueTag::Determinant => "Determinant",
            TechniqueTag::NumericalApproximation => "Numerical approximation",
            TechniqueTag::NewtonRaphson => "Newton-Raphson",
            TechniqueTag::Bisection => "Bisection",
            TechniqueTag::Brent => "Brent's method",
            TechniqueTag::Secant => "Secant method",
            TechniqueTag::DomainNarrowing => "Domain narrowing",
            TechniqueTag::DomainExtension => "Domain extension",
            TechniqueTag::PrincipalBranch => "Principal branch",
            TechniqueTag::Custom(label) => label,
        }
    }

    /// Difficulty level normally associated with this technique. The mapping
    /// is intrinsic to the technique — no caller override; callers filter
    /// steps by this level if they want to condense or explode narration.
    #[must_use]
    pub const fn difficulty(self) -> TechniqueDifficulty {
        use TechniqueDifficulty::*;
        match self {
            // Elementary — basic equation manipulation + combining like terms.
            TechniqueTag::AddBothSides
            | TechniqueTag::SubtractBothSides
            | TechniqueTag::MultiplyBothSides
            | TechniqueTag::DivideBothSides
            | TechniqueTag::Isolation
            | TechniqueTag::MoveTerm
            | TechniqueTag::CombiningLikeTerms
            | TechniqueTag::Simplification => Elementary,

            // Powers and roots.
            TechniqueTag::PowerBothSides
            | TechniqueTag::RootBothSides
            | TechniqueTag::LogIdentity
            | TechniqueTag::ExpIdentity => PowerAndRoots,

            // Pre-calculus algebraic manipulation.
            TechniqueTag::ApplyFunction
            | TechniqueTag::Expansion
            | TechniqueTag::Factoring
            | TechniqueTag::CombineFractions
            | TechniqueTag::Cancellation
            | TechniqueTag::Rationalization
            | TechniqueTag::Conjugation
            | TechniqueTag::PartialFractionDecomp
            | TechniqueTag::ApplyIdentity
            | TechniqueTag::QuadraticFormula
            | TechniqueTag::CompleteTheSquare
            | TechniqueTag::RationalRootTheorem
            | TechniqueTag::SyntheticDivision => AlgebraicManip,

            // Transcendental: trig, hyperbolic, substitution.
            TechniqueTag::Substitution
            | TechniqueTag::TrigIdentity
            | TechniqueTag::HyperbolicIdentity
            | TechniqueTag::EulerFormula
            | TechniqueTag::DeMoivre => Transcendental,

            // Higher-dimensional calculus — intermediate.
            TechniqueTag::TotalDifferential
            | TechniqueTag::Divergence
            | TechniqueTag::Laplacian
            | TechniqueTag::DirectionalDerivative => Calculus,

            // Higher-dimensional calculus — advanced.
            TechniqueTag::Curl | TechniqueTag::Jacobian | TechniqueTag::Hessian => Advanced,

            // Special functions — advanced.
            TechniqueTag::SpecialFunction => Advanced,

            // Calculus.
            TechniqueTag::PowerRule
            | TechniqueTag::ProductRule
            | TechniqueTag::QuotientRule
            | TechniqueTag::ChainRule
            | TechniqueTag::ImplicitDifferentiation
            | TechniqueTag::LogarithmicDifferentiation
            | TechniqueTag::PatternIntegration
            | TechniqueTag::IntegrationByParts
            | TechniqueTag::USubstitution
            | TechniqueTag::TrigSubstitution
            | TechniqueTag::PartialFractionIntegration
            | TechniqueTag::LHopitalRule
            | TechniqueTag::SqueezeTheorem
            | TechniqueTag::LimitAlgebraic
            | TechniqueTag::SeparationOfVariables
            | TechniqueTag::IntegratingFactor
            | TechniqueTag::CharacteristicEquation
            | TechniqueTag::UndeterminedCoefficients
            | TechniqueTag::VariationOfParameters => Calculus,

            // Advanced / university.
            TechniqueTag::RischVerification
            | TechniqueTag::TaylorExpansion
            | TechniqueTag::LaurentExpansion
            | TechniqueTag::AsymptoticExpansion
            | TechniqueTag::SeriesComposition
            | TechniqueTag::LagrangeReversion
            | TechniqueTag::PuiseuxExpansion
            | TechniqueTag::FrobeniusMethod
            | TechniqueTag::PadeApproximant
            | TechniqueTag::WkbApproximation
            | TechniqueTag::ResidueTheorem
            | TechniqueTag::PoleClassification
            | TechniqueTag::FourierSeries
            | TechniqueTag::RungeKutta
            | TechniqueTag::GaussianElimination
            | TechniqueTag::MatrixInverse
            | TechniqueTag::LuDecomposition
            | TechniqueTag::QrDecomposition
            | TechniqueTag::Eigendecomposition
            | TechniqueTag::Determinant
            | TechniqueTag::NumericalApproximation
            | TechniqueTag::NewtonRaphson
            | TechniqueTag::Bisection
            | TechniqueTag::Brent
            | TechniqueTag::Secant
            | TechniqueTag::DomainNarrowing
            | TechniqueTag::DomainExtension
            | TechniqueTag::PrincipalBranch => Advanced,

            // Custom: caller's responsibility; default to Advanced.
            TechniqueTag::Custom(_) => Advanced,
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

    /// Alias for [`Trace::len`] retained for callers that previously consumed a
    /// `ResolutionPath` with a `step_count()` method.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Return the highest [`TechniqueDifficulty`] across every recorded step.
    ///
    /// An empty trace returns [`TechniqueDifficulty::Elementary`] as the
    /// baseline.
    #[must_use]
    pub fn max_difficulty(&self) -> TechniqueDifficulty {
        self.steps
            .iter()
            .map(|s| s.tag.difficulty())
            .max()
            .unwrap_or(TechniqueDifficulty::Elementary)
    }

    /// Return a count of steps per difficulty tier as an array indexed by
    /// tier (0 = Elementary, 5 = Advanced).
    #[must_use]
    pub fn difficulty_profile(&self) -> [usize; 6] {
        let mut profile = [0usize; 6];
        for step in &self.steps {
            let tier = step.tag.difficulty() as u8;
            if tier >= 1 && tier <= 6 {
                profile[(tier - 1) as usize] += 1;
            }
        }
        profile
    }

    /// True when every step's difficulty is at or below `level`.
    #[must_use]
    pub fn is_trivial_at(&self, level: TechniqueDifficulty) -> bool {
        self.max_difficulty() <= level
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
