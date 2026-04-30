//! [`Command`] — enumeration of operations thales can perform.
//!
//! Each variant packages the operation name with its `Expression` inputs.
//! All algebraic manipulations (substitute, factor, expand, …) are
//! standalone variants **and** re-usable as narrated techniques inside other
//! engines — the implementation is shared; `Command` only picks the
//! standalone entry point.
//!
//! # v0.8.1 scope
//!
//! Core set covering everything currently exposed via the per-operation FFI
//! surface, plus algebraic manipulations promoted to first-class commands.
//! Expansions beyond Taylor/Laurent/Asymptotic/Compose/Revert, higher-dim
//! integration, integral transforms beyond Fourier series, systems of ODEs,
//! tensor algebra, and special-function expansions live in v0.9.0.

use crate::Expression;

use super::{Condition, Domain, ExprPath};

/// Operation plus inputs. One variant per supported command.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Command {
    // ── Placeholder ─────────────────────────────────────────────────────────
    /// No-op command. Returns an empty [`super::Response`] with a
    /// [`super::DiagnosticCode::NotImplemented`] diagnostic. Used as the
    /// default for [`super::Request::default()`].
    Noop,

    // ── Algebra ─────────────────────────────────────────────────────────────
    /// Canonicalise an expression. When `over` is `Some`, simplification
    /// applies domain-aware rules (e.g. `√(x²) = x` over `ℝ⁺` vs. `|x|`
    /// over `ℝ`).
    Simplify {
        /// Expression to canonicalise.
        expr: Expression,
        /// Which rules to apply.
        rules: SimplifyRules,
        /// Optional domain scoping.
        over: Option<Domain>,
    },
    /// Distribute products, expand powers, etc.
    Expand {
        /// Expression to expand.
        expr: Expression,
        /// When `Some`, only the subexpression at this path is expanded;
        /// the rest of the tree is left untouched.
        target: Option<ExprPath>,
    },
    /// Factor over the requested domain.
    Factor {
        /// Expression to factor.
        expr: Expression,
        /// Domain of factorisation (integer, rational, real, complex, …).
        over: Domain,
        /// Optional sub-tree target.
        target: Option<ExprPath>,
    },
    /// Substitute pairs `(old, new)` at every occurrence of `old`, or only
    /// inside the sub-tree at `target` when provided.
    Substitute {
        /// Expression to operate on.
        expr: Expression,
        /// List of substitutions, applied in order.
        bindings: Vec<(Expression, Expression)>,
        /// Optional sub-tree target.
        target: Option<ExprPath>,
    },
    /// Combine like terms (sum of `x + 2x` → `3x`).
    CombineLikeTerms {
        /// Expression.
        expr: Expression,
        /// Optional sub-tree target.
        target: Option<ExprPath>,
    },
    /// Reduce a rational expression to a common denominator.
    CommonDenominator {
        /// Expression.
        expr: Expression,
        /// Optional sub-tree target.
        target: Option<ExprPath>,
    },
    /// Partial fraction decomposition of a rational function in `var`.
    PartialFractions {
        /// Rational expression.
        expr: Expression,
        /// Variable of decomposition.
        var: String,
    },
    /// Rationalise a denominator (move roots / complexes out of the
    /// denominator).
    Rationalize {
        /// Expression.
        expr: Expression,
        /// Optional sub-tree target.
        target: Option<ExprPath>,
    },
    /// Form the complex conjugate.
    Conjugate {
        /// Expression.
        expr: Expression,
        /// Optional sub-tree target.
        target: Option<ExprPath>,
    },
    /// Generalised inverse-function computation: solve `y = f(x)` for `x`
    /// treating `f` as an expression in `var`, returning the inverse as an
    /// expression in `y`.
    InverseFn {
        /// Expression `f(var)`.
        expr: Expression,
        /// Variable with respect to which `expr` is treated as a function.
        var: String,
    },
    /// Rearrange an equation to isolate the named variable.
    Rearrange {
        /// Equation (relation) to rearrange.
        equation: Expression,
        /// Variable to solve for.
        solve_for: String,
    },
    /// Apply a named identity (trig, log, hyperbolic, exp).
    ApplyIdentity {
        /// Expression.
        expr: Expression,
        /// Identity name (stable label).
        identity: IdentityId,
        /// Optional sub-tree target.
        target: Option<ExprPath>,
    },

    // ── Solve ───────────────────────────────────────────────────────────────
    /// Solve a relation (equation or inequality) for `var`, searching within
    /// `over`.
    SolveFor {
        /// Equation, inequality, or membership relation.
        relation: Expression,
        /// Variable.
        var: String,
        /// Search domain.
        over: Domain,
    },
    /// Simultaneously solve a system of equations.
    SolveSystem {
        /// Equations.
        equations: Vec<Expression>,
        /// Variables to solve for.
        vars: Vec<String>,
        /// Search domain.
        over: Domain,
    },

    // ── Differentiation ─────────────────────────────────────────────────────
    /// Differentiate `expr` with respect to a single variable.
    Diff {
        /// Expression.
        expr: Expression,
        /// Variable.
        var: String,
        /// Differentiation order.
        order: u32,
    },
    /// Mixed / higher-order partial derivative. Each `(var, order)` pair
    /// contributes `∂^order / ∂var^order`.
    PartialDiff {
        /// Expression.
        expr: Expression,
        /// List of (variable, order) pairs, applied in order.
        vars: Vec<(String, u32)>,
    },
    /// Total derivative `d/d var` with explicit dependency declarations.
    TotalDiff {
        /// Expression.
        expr: Expression,
        /// Variable of differentiation.
        var: String,
        /// List of `(name, expression)` pairs declaring that `name`
        /// depends on `var` via the given expression.
        deps: Vec<(String, Expression)>,
    },
    /// Gradient of a scalar field.
    Gradient {
        /// Scalar expression.
        expr: Expression,
        /// Coordinate variables.
        vars: Vec<String>,
    },
    /// Divergence of a vector field.
    Divergence {
        /// Vector components of the field, one per coordinate in `vars`.
        field: Vec<Expression>,
        /// Coordinate variables.
        vars: Vec<String>,
    },
    /// Curl of a vector field (3-D).
    Curl {
        /// Three vector components of the field.
        field: Vec<Expression>,
        /// Three coordinate variables.
        vars: Vec<String>,
    },
    /// Laplacian of a scalar field.
    Laplacian {
        /// Scalar expression.
        expr: Expression,
        /// Coordinate variables.
        vars: Vec<String>,
    },
    /// Jacobian matrix of a vector field.
    Jacobian {
        /// Component expressions.
        fields: Vec<Expression>,
        /// Coordinate variables.
        vars: Vec<String>,
    },
    /// Hessian matrix of a scalar field.
    Hessian {
        /// Scalar expression.
        expr: Expression,
        /// Coordinate variables.
        vars: Vec<String>,
    },
    /// Directional derivative along a direction vector.
    DirectionalDiff {
        /// Scalar expression.
        expr: Expression,
        /// Coordinate variables.
        vars: Vec<String>,
        /// Direction vector components (one per coordinate).
        direction: Vec<Expression>,
    },

    // ── Integration ─────────────────────────────────────────────────────────
    /// Indefinite integral of `expr` in `var`.
    Integrate {
        /// Integrand.
        expr: Expression,
        /// Variable of integration.
        var: String,
    },
    /// Definite integral from `from` to `to`.
    DefIntegrate {
        /// Integrand.
        expr: Expression,
        /// Variable of integration.
        var: String,
        /// Lower bound.
        from: Expression,
        /// Upper bound.
        to: Expression,
    },

    // ── Limits ──────────────────────────────────────────────────────────────
    /// Limit of `expr` as `var → point`, optionally one-sided.
    Limit {
        /// Expression.
        expr: Expression,
        /// Limit variable.
        var: String,
        /// Limit point.
        point: LimitPoint,
        /// Optional one-sided qualifier.
        side: Option<Side>,
    },

    // ── Expansions ──────────────────────────────────────────────────────────
    /// Taylor polynomial of `expr` around `center` up to `order`.
    Taylor {
        /// Expression.
        expr: Expression,
        /// Variable.
        var: String,
        /// Center.
        center: Expression,
        /// Order of the truncated series.
        order: u32,
    },
    /// Laurent expansion (principal + analytic) at `center` up to `order`.
    Laurent {
        /// Expression.
        expr: Expression,
        /// Variable.
        var: String,
        /// Center.
        center: Expression,
        /// Order of the truncated series.
        order: u32,
    },
    /// Asymptotic expansion (Poincaré-type) to `order`.
    Asymptotic {
        /// Expression.
        expr: Expression,
        /// Variable.
        var: String,
        /// Order of the truncated series.
        order: u32,
    },
    /// Series composition of `outer ∘ inner`.
    Compose {
        /// Outer series.
        outer: Expression,
        /// Inner series.
        inner: Expression,
        /// Variable.
        var: String,
        /// Order of the resulting series.
        order: u32,
    },
    /// Lagrange reversion: series inverse of `expr`.
    Revert {
        /// Expression.
        expr: Expression,
        /// Variable.
        var: String,
        /// Order of the reverted series.
        order: u32,
    },
    /// Puiseux series (fractional-power Laurent) of `expr` around `center`
    /// up to `order`.
    Puiseux {
        /// Expression.
        expr: Expression,
        /// Variable.
        var: String,
        /// Center.
        center: Expression,
        /// Order of the truncated series.
        order: u32,
    },
    /// Frobenius method for a linear ODE near a regular singular point.
    Frobenius {
        /// ODE as a relation in the unknown function and its derivatives.
        ode: Expression,
        /// Name of the unknown function.
        fn_name: String,
        /// Independent variable.
        var: String,
        /// Expansion point (regular singular point).
        point: Expression,
        /// Number of terms in each Frobenius series.
        order: u32,
    },
    /// Padé approximant `[m/n]` of `expr` around `center`.
    Pade {
        /// Expression.
        expr: Expression,
        /// Variable.
        var: String,
        /// Expansion center.
        center: Expression,
        /// Numerator degree.
        m: u32,
        /// Denominator degree.
        n: u32,
    },
    /// WKB (Wentzel-Kramers-Brillouin) asymptotic approximation for a
    /// Schrödinger-type ODE.
    Wkb {
        /// ODE as a relation in the unknown function and its derivatives.
        ode: Expression,
        /// Name of the unknown function.
        fn_name: String,
        /// Independent variable.
        var: String,
        /// Small parameter name (ℏ or ε).
        small_param: String,
        /// Approximation order.
        order: u32,
    },

    // ── Transforms ──────────────────────────────────────────────────────────
    /// Fourier series on `[−period/2, period/2]` up to `terms` harmonics.
    FourierSeries {
        /// Expression.
        expr: Expression,
        /// Variable.
        var: String,
        /// Fundamental period `L`. Series computed on `[−L/2, L/2]`.
        period: Expression,
        /// Number of harmonics.
        terms: u32,
    },
    /// Residue of `expr` at `point`.
    Residue {
        /// Expression.
        expr: Expression,
        /// Complex variable.
        var: String,
        /// Singular point.
        point: Expression,
    },

    // ── Special functions ───────────────────────────────────────────────────
    /// Evaluate or simplify a special function.
    SpecialFn {
        /// Function identifier.
        kind: SpecialKind,
        /// Arguments.
        args: Vec<Expression>,
    },

    // ── ODE ─────────────────────────────────────────────────────────────────
    /// Solve an ordinary differential equation.
    Ode {
        /// ODE as a relation in the unknown function and its derivatives.
        equation: Expression,
        /// Name of the unknown function.
        fn_name: String,
        /// Independent variable.
        var: String,
        /// Initial-value data (optional — `None` returns the general
        /// solution).
        ic: Option<IvpData>,
    },

    // ── Matrix / linear algebra ─────────────────────────────────────────────
    /// Matrix / linear algebra operation.
    Matrix {
        /// Operation identifier.
        op: MatrixOp,
        /// Matrix or vector operands.
        operands: Vec<MatrixExpr>,
    },

    // ── Nabla (del operator) ─────────────────────────────────────────────────
    /// Del operator `∇` applied to a scalar or vector field.
    ///
    /// `op` selects which vector-calculus operation to perform; `input`
    /// carries either a scalar expression or the components of a vector
    /// field; `vars` names the coordinate variables in order.
    Nabla {
        /// Operation to perform.
        op: NablaOp,
        /// Scalar or vector-field input.
        input: NablaInput,
        /// Coordinate variable names, in order.
        vars: Vec<String>,
    },

    // ── Optimization ────────────────────────────────────────────────────────
    /// Constrained optimisation.
    Optimize {
        /// Objective expression.
        objective: Expression,
        /// Optimisation variables.
        vars: Vec<String>,
        /// Constraints.
        constraints: Vec<Constraint>,
        /// Min / max.
        sense: OptSense,
    },
    /// Lagrange-multiplier specialisation: equality-constrained extremisation.
    LagrangeMult {
        /// Objective expression.
        objective: Expression,
        /// Optimisation variables.
        vars: Vec<String>,
        /// Equality constraints (each expression interpreted as `= 0`).
        equality_constraints: Vec<Expression>,
    },
}

impl Default for Command {
    fn default() -> Self {
        Command::Noop
    }
}

/// Operation selector for [`Command::Nabla`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NablaOp {
    /// Gradient `∇f`.
    Grad,
    /// Divergence `∇·F`.
    Div,
    /// Curl `∇×F` (3-D only).
    Curl,
    /// Laplacian `∇²f`.
    Laplacian,
    /// Divergence of curl `∇·(∇×F)` — identity that equals 0.
    DivOfCurl,
    /// Curl of gradient `∇×(∇f)` — identity that equals (0,0,0).
    CurlOfGrad,
    /// Divergence of gradient `∇·(∇f) = ∇²f`.
    DivOfGrad,
}

/// Input payload for [`Command::Nabla`].
#[derive(Debug, Clone, PartialEq)]
pub enum NablaInput {
    /// A scalar field expression.
    Scalar(Expression),
    /// Components of a vector field, one per coordinate variable.
    VectorField(Vec<Expression>),
}

/// Bundle of simplification rules to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SimplifyRules {
    /// Apply arithmetic canonicalisation (combining like terms, constant
    /// folding, sign normalisation).
    pub arithmetic: bool,
    /// Apply algebraic simplifications (power rules, log rules).
    pub algebraic: bool,
    /// Apply trigonometric identities.
    pub trigonometric: bool,
    /// Apply logarithmic identities.
    pub logarithmic: bool,
    /// Apply exponential identities.
    pub exponential: bool,
    /// Apply hyperbolic identities.
    pub hyperbolic: bool,
    /// Apply rational-function canonicalisation (common denominator,
    /// reduction).
    pub rational: bool,
}

impl SimplifyRules {
    /// All rule groups enabled.
    #[must_use]
    pub const fn all() -> Self {
        Self {
            arithmetic: true,
            algebraic: true,
            trigonometric: true,
            logarithmic: true,
            exponential: true,
            hyperbolic: true,
            rational: true,
        }
    }
}

/// Named identity for [`Command::ApplyIdentity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum IdentityId {
    /// `sin² + cos² = 1`.
    PythagoreanTrig,
    /// `cosh² − sinh² = 1`.
    PythagoreanHyp,
    /// `sin(2x) = 2 sin x cos x`.
    DoubleAngleSin,
    /// `cos(2x) = cos²x − sin²x`.
    DoubleAngleCos,
    /// `sin(a±b) = sin a cos b ± cos a sin b`.
    SumToProductSin,
    /// `cos(a±b) = cos a cos b ∓ sin a sin b`.
    SumToProductCos,
    /// `log(a·b) = log a + log b`.
    LogProduct,
    /// `log(a^b) = b log a`.
    LogPower,
    /// `e^(a+b) = e^a · e^b`.
    ExpSum,
    /// Euler's formula `e^(iθ) = cos θ + i sin θ`.
    Euler,
    /// De Moivre `(cos θ + i sin θ)^n = cos(nθ) + i sin(nθ)`.
    DeMoivre,
    /// `a² − b² = (a+b)(a−b)`.
    DifferenceOfSquares,
    /// `a³ + b³ = (a+b)(a² − ab + b²)`.
    SumOfCubes,
    /// Other named identity carried as a stable label.
    Other(&'static str),
}

/// Limit target point.
#[derive(Debug, Clone, PartialEq)]
pub enum LimitPoint {
    /// Finite value.
    Finite(Expression),
    /// Positive infinity.
    PosInf,
    /// Negative infinity.
    NegInf,
}

/// One-sided limit qualifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Approach from the left (from below).
    Left,
    /// Approach from the right (from above).
    Right,
}

/// Special-function identifier, for [`Command::SpecialFn`]. v0.8.1 covers the
/// functions currently implemented in `src/special.rs`; v0.9.0 extends this
/// set (see [`crate::api`] scope notes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpecialKind {
    /// Gamma function `Γ(x)`.
    Gamma,
    /// Beta function `B(a, b)`.
    Beta,
    /// Error function `erf(x)`.
    Erf,
    /// Complementary error function `erfc(x)`.
    Erfc,
}

/// Initial-value data for [`Command::Ode`].
#[derive(Debug, Clone, PartialEq)]
pub struct IvpData {
    /// Value of the independent variable.
    pub var_at: Expression,
    /// Value of the unknown function at `var_at`.
    pub fn_at: Expression,
    /// Derivative values at `var_at`, in order: `f'(var_at)`, `f''(var_at)`,
    /// ... up to the order of the ODE minus one. Empty for first-order ODEs.
    pub derivatives_at: Vec<Expression>,
}

/// Matrix / linear algebra operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MatrixOp {
    /// Matrix addition.
    Add,
    /// Matrix subtraction.
    Subtract,
    /// Matrix multiplication.
    Multiply,
    /// Scalar multiplication.
    ScalarMultiply,
    /// Transpose.
    Transpose,
    /// Determinant.
    Determinant,
    /// Inverse.
    Inverse,
    /// Trace.
    Trace,
    /// Rank.
    Rank,
    /// Null space.
    NullSpace,
    /// Eigenvalues.
    Eigenvalues,
    /// Eigenvectors.
    Eigenvectors,
    /// LU decomposition.
    Lu,
    /// QR decomposition.
    Qr,
    /// Solve `Ax = b`.
    SolveLinear,
}

/// Matrix or vector expression operand.
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixExpr {
    /// Scalar expression (for scalar multiplication).
    Scalar(Expression),
    /// Column vector.
    Vector(Vec<Expression>),
    /// 2-D matrix stored row-major.
    Matrix(Vec<Vec<Expression>>),
}

/// Optimisation constraint.
#[derive(Debug, Clone, PartialEq)]
pub enum Constraint {
    /// Equality constraint `expr = 0`.
    Equality(Expression),
    /// Inequality constraint `expr ≤ 0`.
    LessEq(Expression),
    /// Inequality constraint `expr < 0`.
    Less(Expression),
    /// Inequality constraint `expr ≥ 0`.
    GreaterEq(Expression),
    /// Inequality constraint `expr > 0`.
    Greater(Expression),
    /// Conditional constraint: apply the wrapped constraint only when the
    /// [`Condition`] holds.
    Conditional {
        /// When this condition holds.
        when: Condition,
        /// Apply this constraint.
        constraint: Box<Constraint>,
    },
}

/// Optimisation sense.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptSense {
    /// Minimise objective.
    #[default]
    Minimize,
    /// Maximise objective.
    Maximize,
}
