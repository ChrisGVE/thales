//! [`Response`] — output of [`super::execute`].
//!
//! Carries a map of [`ResultKey`] → [`ResultEntry`] alongside assumptions,
//! diagnostics, and execution metadata. All `Expression` values leaving
//! thales are decompiled from the internal `Arc<Expr>` form **at this
//! boundary only** (architecture rule 1).

use crate::Expression;

use super::{Assumption, Condition, Diagnostic, Domain, ExprPath, Narrative, Precision};

/// Typed structured output supplementing [`ResultEntry::value`].
///
/// Each variant carries exactly the metadata its command family needs.
/// `ResultEntry::structured` is `None` when the command produces a single
/// scalar result and the legacy `value` field is sufficient.
///
/// Marked `#[non_exhaustive]` so new variants can be added in minor releases
/// without breaking existing match arms in downstream code.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum StructuredResult {
    /// Single scalar expression (pass-through for commands that have only
    /// one output but want a typed wrapper).
    Scalar(Expression),

    /// Named binding: a label paired with its value.
    /// Used by: `SolveSystem` (variable → value), `Optimize` (var → coordinate).
    Labeled {
        /// Variable or parameter name.
        label: String,
        /// Resolved value.
        value: Expression,
    },

    /// Decomposition with named, typed parts.
    /// Used by: `LU` (L, U, P), `QR` (Q, R), eigenpairs.
    Decomposition {
        /// Ordered list of `(name, part)` pairs. Names are stable labels
        /// such as "L", "U", "P", "eigenvalue", "eigenvector".
        parts: Vec<(String, DecompositionPart)>,
    },

    /// Coefficient array with series metadata.
    /// Used by: `FourierSeries`, `Taylor`, `Laurent`, `Pade`.
    CoefficientArray {
        /// Coefficients in order (a₀, a₁, …).
        coefficients: Vec<Expression>,
        /// Name of the expansion variable (e.g. "x").
        variable: String,
        /// Expansion centre, or `None` for expansions around 0.
        center: Option<Expression>,
        /// Number of terms (order of the truncation).
        order: u32,
    },

    /// Multi-branch result — alternatives valid under different conditions.
    /// Used by: Frobenius (two branches), WKB (two branches), polynomial
    /// roots, piecewise solutions.
    Branches {
        /// One entry per branch. The entries partition the solution space
        /// when conditions are mutually exclusive.
        branches: Vec<BranchEntry>,
    },

    /// Matrix or tensor result with shape metadata.
    /// Used by: matrix operations, tensor operations.
    Shaped {
        /// Elements in row-major order.
        elements: Vec<Expression>,
        /// Shape as `[rows, cols]` for matrices, `[d1, d2, …]` for tensors.
        shape: Vec<usize>,
        /// Optional dimension labels (index names, covariant/contravariant).
        labels: Option<Vec<String>>,
    },

    /// Transform result with convergence domain metadata.
    /// Used by: Laplace, Fourier, Z, Mellin transforms.
    TransformResult {
        /// Transformed expression.
        expression: Expression,
        /// Name of the transform variable (e.g. "s", "ω", "z").
        transform_variable: String,
        /// Region of convergence or validity condition, if known.
        convergence: Option<Expression>,
    },
}

/// One branch in a [`StructuredResult::Branches`] result.
#[derive(Debug, Clone)]
pub struct BranchEntry {
    /// Condition under which this branch applies, if discriminated.
    pub condition: Option<Condition>,
    /// Human-readable label for this branch (e.g. "y_1", "y_2", "n=0").
    pub label: Option<String>,
    /// The branch value.
    pub value: Expression,
}

/// Part of a [`StructuredResult::Decomposition`] result.
#[derive(Debug, Clone)]
pub enum DecompositionPart {
    /// Scalar expression.
    Scalar(Expression),
    /// Dense matrix in row-major order.
    Matrix {
        /// Elements in row-major order.
        elements: Vec<Expression>,
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        cols: usize,
    },
    /// Permutation vector (index `i` → row `perm[i]`).
    Permutation(Vec<usize>),
}

use crate::numeric::trace::{TechniqueDifficulty, TechniqueTag};

/// Output of [`super::execute`].
///
/// `results` is an ordered list rather than a `BTreeMap`: `ResultKey::Branch`
/// wraps a `Condition` that in turn contains `Expression` values, and
/// `Expression` does not implement `Ord`. Uniqueness of keys is a contract
/// guaranteed by the dispatcher; callers may treat `results` as a map by
/// iterating pairs.
#[derive(Debug, Clone, Default)]
pub struct Response {
    /// Results, in insertion order. Keys are unique per dispatcher contract.
    pub results: Vec<(ResultKey, ResultEntry)>,
    /// Assumptions the engine made while producing the results (e.g.
    /// "assumed x > 0", "used principal branch").
    pub assumptions: Vec<Assumption>,
    /// Informational, warning, and error diagnostics.
    pub diagnostics: Vec<Diagnostic>,
    /// Timing / iteration / engine-trace metadata.
    pub meta: ExecutionMeta,
}

/// Key into the [`Response::results`] list.
///
/// [`ResultKey::Single`] denotes a lone result; [`ResultKey::Branch`] carries
/// the [`Condition`] under which the paired [`ResultEntry`] applies.
/// Multiple [`ResultKey::Branch`] entries in the same response partition the
/// solution space (e.g. one per root, one per interval, one per case).
/// [`ResultKey::Component`] names a part in a simultaneous (non-alternative)
/// multi-part result. [`ResultKey::ConvergenceDomain`] is a metadata entry
/// carrying the region of convergence for transform results.
#[derive(Debug, Clone, PartialEq)]
pub enum ResultKey {
    /// Single unconditional result.
    Single,
    /// Result valid under the given [`Condition`].
    Branch(Condition),
    /// Named component in a multi-part simultaneous result (not an
    /// alternative branch). Used for ODE system components, tensor indices,
    /// and transform metadata.
    Component(String),
    /// Convergence domain metadata entry for transform results (Laplace,
    /// Fourier, Z, Mellin). The paired [`ResultEntry`] carries the region of
    /// convergence as its value.
    ConvergenceDomain,
}

/// One result in a [`Response`].
#[derive(Debug, Clone)]
pub struct ResultEntry {
    /// Computed value. May be purely symbolic, purely numeric, or a
    /// symbolic-then-numeric hybrid. See [`ResultValue`].
    pub value: ResultValue,
    /// Optional typed structured data supplementing [`Self::value`].
    ///
    /// When `Some`, clients should prefer this over parsing `value`.
    /// `None` for single-scalar results where the legacy `value` field
    /// is sufficient. New dispatch arms populate this from day one;
    /// existing arms default to `None` for backward compatibility.
    pub structured: Option<StructuredResult>,
    /// Structural shape of [`Self::value`] (scalar, vector, matrix, …).
    /// Allows clients to dispatch without deep introspection.
    pub shape: ResultShape,
    /// Unit composed from input annotations, if any. `None` when the run
    /// is unit-less (no input symbol carried a unit annotation).
    ///
    /// **v0.8.1 scaffolding:** always `None`. Unit propagation lands in
    /// v0.10.0.
    pub unit: Option<UnitPlaceholder>,
    /// Narrated steps produced by the engine (empty when
    /// [`super::Request::narrate`] was `false`).
    pub steps: Vec<NarratedStep>,
    /// Alternative equivalent forms (factored vs. expanded, polar vs.
    /// rectangular, etc.). Primary form is [`Self::value`]; these are
    /// additional renderings of the same result.
    pub alternatives: Vec<Expression>,
    /// Identifier of the engine that produced this entry. Useful for
    /// debugging, reproducibility, and selective re-runs.
    pub engine: EngineId,
}

/// Placeholder for [`mathcore_units::Unit`]. Filled in during v0.10.0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitPlaceholder {
    /// Textual canonical form (e.g. "kg*m/s^2" or "N"). Transitional.
    pub canonical: String,
}

/// The shape of a [`ResultEntry::value`]. Orthogonal to the symbolic /
/// numeric distinction — a `Matrix` can be symbolic, numeric, or hybrid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultShape {
    /// Single expression.
    Scalar,
    /// 1-D list of expressions.
    Vector,
    /// 2-D array of expressions.
    Matrix,
    /// Higher-rank array. Shape metadata carried in the expression itself.
    Tensor,
    /// Set of expressions (roots, eigenvalues, …).
    Set,
    /// Function of one or more variables (returned by inverse-function,
    /// Fourier series as a function of `x`, …).
    Function,
    /// Relation (equation, inequality, membership statement).
    Relation,
}

/// The computed value carried by a [`ResultEntry`].
#[derive(Debug, Clone)]
pub enum ResultValue {
    /// Closed-form symbolic result.
    Symbolic(Expression),
    /// Numeric result reached without a symbolic intermediary.
    Numeric {
        /// Numeric value, wrapped as an `Expression` (typically a `Float` or
        /// a `Complex` leaf).
        value: Expression,
        /// Achieved precision.
        precision: Precision,
        /// Numeric method used.
        method: NumericMethod,
    },
    /// Symbolic progress up to [`Self::Hybrid::last_symbolic`], then numeric
    /// completion.
    Hybrid {
        /// Farthest symbolic form before numeric evaluation.
        last_symbolic: Expression,
        /// Numeric evaluation of `last_symbolic`.
        numeric: Expression,
        /// Achieved precision.
        precision: Precision,
        /// Numeric method used for the final leg.
        method: NumericMethod,
    },
    /// Engine could not produce a result. Narrative describes why.
    Unsolved {
        /// Human-readable reason.
        reason: super::Narrative,
    },
    /// No solution exists in the requested [`Domain`]. The solution may exist
    /// in a wider domain; `domain` names the failing restriction.
    NoSolution {
        /// Domain the engine searched in.
        domain: Domain,
        /// Why the search returned empty.
        reason: super::Narrative,
    },
}

/// Numeric method identifiers used by numeric / hybrid results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NumericMethod {
    /// Newton-Raphson root finder.
    NewtonRaphson,
    /// Brent's method root finder.
    Brent,
    /// Bisection root finder.
    Bisection,
    /// Secant root finder.
    Secant,
    /// Classical fourth-order Runge-Kutta ODE integrator.
    RungeKutta4,
    /// Adaptive Runge-Kutta-Fehlberg.
    RungeKuttaFehlberg,
    /// Composite Simpson integration.
    Simpson,
    /// Gauss-Legendre quadrature.
    GaussLegendre,
    /// Adaptive quadrature.
    AdaptiveQuadrature,
    /// Custom engine-defined method, identified by a stable label.
    Other(&'static str),
}

/// Engine identifier. Stable label naming the internal engine that produced a
/// [`ResultEntry`]. Useful for debugging and reproducibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EngineId {
    /// Symbolic simplifier.
    Simplify,
    /// Differentiation engine.
    Differentiation,
    /// Pattern-based integration.
    PatternIntegration,
    /// Integration by parts.
    IntegrationByParts,
    /// Risch-like verification.
    RischVerify,
    /// L'Hôpital rule.
    LHopital,
    /// Taylor expansion engine.
    TaylorExpansion,
    /// Laurent expansion engine.
    LaurentExpansion,
    /// Asymptotic expansion engine.
    AsymptoticExpansion,
    /// Series composition engine.
    SeriesComposition,
    /// Lagrange reversion engine.
    SeriesReversion,
    /// Puiseux series expansion engine.
    PuiseuxExpansion,
    /// Frobenius method engine.
    FrobeniusMethod,
    /// Padé approximant engine.
    PadeApproximant,
    /// WKB approximation engine.
    WkbExpansion,
    /// Residue computation.
    Residue,
    /// Singularity classifier.
    SingularityClassifier,
    /// Equation solver (linear, polynomial, transcendental, …).
    EquationSolver,
    /// System solver.
    SystemSolver,
    /// Inequality solver.
    InequalitySolver,
    /// ODE solver (first order).
    OdeFirstOrder,
    /// ODE solver (second order).
    OdeSecondOrder,
    /// ODE solver (higher order).
    OdeHigherOrder,
    /// ODE system solver (eigenvalue or numeric).
    OdeSystem,
    /// Matrix operations.
    Matrix,
    /// Fourier series engine.
    FourierSeries,
    /// Special functions.
    SpecialFunctions,
    /// Partial fractions decomposition.
    PartialFractions,
    /// Constrained optimiser.
    Optimizer,
    /// Nabla (del) operator — gradient, divergence, curl, Laplacian, identities.
    Nabla,
    /// Other engine identified by a stable label.
    Other(&'static str),
}

/// One narrated step from an engine run.
///
/// Steps carry a technique tag, a difficulty level mapped from the tag (so
/// the caller can filter by educational level), a Markdown-templated
/// narrative, an optional positional path into the input tree pinpointing
/// the manipulated subexpression, and optional input / output operands.
///
/// The `unit_trace` field is reserved for v0.10.0 dimensional analysis.
#[derive(Debug, Clone)]
pub struct NarratedStep {
    /// Named technique applied.
    pub tag: TechniqueTag,
    /// Educational difficulty level (mapped from `tag` at exit).
    pub difficulty: TechniqueDifficulty,
    /// Markdown-templated narrative with bindings.
    pub narrative: Narrative,
    /// Position into the input expression tree for this step, when the
    /// technique targets a subtree.
    pub path: Option<ExprPath>,
    /// Operand before applying the technique.
    pub input: Option<Expression>,
    /// Operand after applying the technique.
    pub output: Option<Expression>,
    /// Dimensional analysis trace, produced when any input symbol carries a
    /// unit annotation. `None` in v0.8.1 (ambient-only mode).
    pub unit_trace: Option<UnitTracePlaceholder>,
}

/// Placeholder for v0.10.0 `UnitTrace`. Transitional.
#[derive(Debug, Clone)]
pub struct UnitTracePlaceholder {
    /// Textual input-dimension signature.
    pub input_dim: String,
    /// Textual output-dimension signature.
    pub output_dim: String,
    /// `false` when the step would violate dimensional consistency.
    pub consistent: bool,
}

/// Execution metadata attached to every [`Response`].
#[derive(Debug, Clone, Default)]
pub struct ExecutionMeta {
    /// Wall-time in milliseconds.
    pub elapsed_ms: u64,
    /// Iteration count, for iterative engines.
    pub iterations: Option<u64>,
    /// Ordered list of engines the dispatch traversed (first → last).
    pub engine_trace: Vec<EngineId>,
}
