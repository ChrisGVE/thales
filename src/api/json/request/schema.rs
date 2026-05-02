//! Serde-derived JSON mirror types for request parsing.

use mathlex::Expression;
use serde::Deserialize;

// ── Mirror enum ───────────────────────────────────────────────────────────────

/// JSON-layer mirror of [`super::super::super::command::Command`].
///
/// Every [`Command`] variant has exactly one corresponding `JsonCommand`
/// variant. The `_exhaustiveness_check` in `tests.rs` fails compilation if
/// they diverge. Expression inputs are native mathlex [`Expression`] JSON
/// objects; serde deserialises them directly with no string-parsing step.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(in super::super) enum JsonCommand {
    // ── Placeholder ────────────────────────────────────────────────────
    Noop,

    // ── Algebra ────────────────────────────────────────────────────────
    Simplify {
        expr: Expression,
        rules: Option<JsonSimplifyRules>,
        over: Option<String>,
    },
    Expand {
        expr: Expression,
        target: Option<String>,
    },
    Factor {
        expr: Expression,
        over: Option<String>,
        target: Option<String>,
    },
    Substitute {
        expr: Expression,
        bindings: Vec<JsonBinding>,
    },
    CombineLikeTerms {
        expr: Expression,
        target: Option<String>,
    },
    CommonDenominator {
        expr: Expression,
        target: Option<String>,
    },
    PartialFractions {
        expr: Expression,
        var: String,
    },
    Rationalize {
        expr: Expression,
        target: Option<String>,
    },
    Conjugate {
        expr: Expression,
        target: Option<String>,
    },
    InverseFn {
        expr: Expression,
        var: String,
    },
    Rearrange {
        equation: Expression,
        solve_for: String,
    },
    ApplyIdentity {
        expr: Expression,
        identity: String,
        target: Option<String>,
    },

    // ── Solve ──────────────────────────────────────────────────────────
    SolveFor {
        relation: Expression,
        var: String,
        over: Option<String>,
    },
    SolveSystem {
        equations: Vec<Expression>,
        vars: Vec<String>,
        over: Option<String>,
    },

    // ── Differentiation ────────────────────────────────────────────────
    Diff {
        expr: Expression,
        var: String,
        order: Option<u32>,
    },
    PartialDiff {
        expr: Expression,
        vars: Vec<JsonPartialDiffVar>,
    },
    TotalDiff {
        expr: Expression,
        var: String,
        deps: Vec<JsonDep>,
    },
    Gradient {
        expr: Expression,
        vars: Vec<String>,
    },
    Divergence {
        field: Vec<Expression>,
        vars: Vec<String>,
    },
    Curl {
        field: Vec<Expression>,
        vars: Vec<String>,
    },
    Laplacian {
        expr: Expression,
        vars: Vec<String>,
    },
    Jacobian {
        fields: Vec<Expression>,
        vars: Vec<String>,
    },
    Hessian {
        expr: Expression,
        vars: Vec<String>,
    },
    DirectionalDiff {
        expr: Expression,
        vars: Vec<String>,
        direction: Vec<Expression>,
    },

    // ── Integration ────────────────────────────────────────────────────
    Integrate {
        expr: Expression,
        var: String,
    },
    DefIntegrate {
        expr: Expression,
        var: String,
        from: Expression,
        to: Expression,
    },
    MultiIntegrate {
        expr: Expression,
        integrations: Vec<JsonIntegrationStep>,
    },
    ChangeCoords {
        expr: Expression,
        from_vars: Vec<String>,
        to_vars: Vec<String>,
        system: String,
    },
    PathIntegral {
        expr: Expression,
        curve: JsonParamCurve,
    },
    SurfaceIntegral {
        expr: Expression,
        vars: Vec<String>,
    },

    // ── Limits ─────────────────────────────────────────────────────────
    Limit {
        expr: Expression,
        var: String,
        point: Expression,
        side: Option<String>,
    },

    // ── Expansions ─────────────────────────────────────────────────────
    Taylor {
        expr: Expression,
        var: String,
        center: Expression,
        order: Option<u32>,
    },
    Laurent {
        expr: Expression,
        var: String,
        center: Expression,
        order: Option<u32>,
    },
    Asymptotic {
        expr: Expression,
        var: String,
        order: Option<u32>,
    },
    Compose {
        outer: Expression,
        inner: Expression,
        var: String,
        order: Option<u32>,
    },
    Revert {
        expr: Expression,
        var: String,
        order: Option<u32>,
    },
    Puiseux {
        expr: Expression,
        var: String,
        center: Option<Expression>,
        order: Option<u32>,
    },
    Frobenius {
        ode: Expression,
        fn_name: String,
        var: String,
        point: Option<Expression>,
        order: Option<u32>,
    },
    Pade {
        expr: Expression,
        var: String,
        center: Option<Expression>,
        m: u32,
        n: u32,
    },
    Wkb {
        ode: Expression,
        fn_name: String,
        var: String,
        small_param: String,
        order: Option<u32>,
    },

    // ── Transforms ─────────────────────────────────────────────────────
    FourierSeries {
        expr: Expression,
        var: String,
        period: Expression,
        terms: Option<u32>,
    },
    Residue {
        expr: Expression,
        var: String,
        point: Expression,
    },
    LaplaceTransform {
        expr: Expression,
        time_var: String,
        freq_var: Option<String>,
    },
    InverseLaplace {
        expr: Expression,
        freq_var: String,
        time_var: Option<String>,
    },
    FourierTransform {
        expr: Expression,
        time_var: String,
        freq_var: Option<String>,
    },
    InverseFourier {
        expr: Expression,
        freq_var: String,
        time_var: Option<String>,
    },
    ZTransform {
        expr: Expression,
        var: String,
        z_var: Option<String>,
    },
    InverseZTransform {
        expr: Expression,
        z_var: String,
        var: Option<String>,
    },
    MellinTransform {
        expr: Expression,
        var: String,
        s_var: Option<String>,
    },
    InverseMellin {
        expr: Expression,
        s_var: String,
        var: Option<String>,
    },

    // ── Special functions ──────────────────────────────────────────────
    SpecialFn {
        kind: String,
        args: Vec<Expression>,
    },

    // ── ODE ────────────────────────────────────────────────────────────
    Ode {
        equation: Expression,
        fn_name: String,
        var: String,
        ic: Option<JsonIvpData>,
    },
    OdeSystem {
        equations: Vec<Expression>,
        fn_names: Vec<String>,
        var: String,
        ic: Option<JsonSystemIvpData>,
    },
    Pde {
        equation: Expression,
        fn_name: String,
        vars: Vec<String>,
    },

    // ── Matrix ─────────────────────────────────────────────────────────
    Matrix {
        op: String,
        operands: Option<Vec<JsonMatrixOperand>>,
    },

    // ── Nabla ──────────────────────────────────────────────────────────
    /// JSON mirror of [`super::super::super::command::Command::Nabla`].
    ///
    /// `op` is one of: `"Grad"`, `"Div"`, `"Curl"`, `"Laplacian"`,
    /// `"DivOfCurl"`, `"CurlOfGrad"`, `"DivOfGrad"`.
    ///
    /// `input` is either a plain Expression object (scalar ops) or an
    /// array of Expression objects (vector-field ops), encoded as
    /// [`JsonNablaInput`].
    Nabla {
        op: String,
        input: JsonNablaInput,
        vars: Vec<String>,
    },

    // ── Optimization ───────────────────────────────────────────────────
    Optimize {
        objective: Expression,
        vars: Vec<String>,
        constraints: Option<Vec<JsonConstraint>>,
        sense: Option<String>,
    },
    LagrangeMult {
        objective: Expression,
        vars: Vec<String>,
        equality_constraints: Vec<Expression>,
    },
}

// ── Supporting serde types ────────────────────────────────────────────────────

/// One step in a `MultiIntegrate` command.
#[derive(Debug, Deserialize)]
pub(in super::super) struct JsonIntegrationStep {
    pub var: String,
    pub from: Expression,
    pub to: Expression,
}

/// Parametric curve for a `PathIntegral` command.
#[derive(Debug, Deserialize)]
pub(in super::super) struct JsonParamCurve {
    pub components: Vec<Expression>,
    pub param: String,
    pub from: Expression,
    pub to: Expression,
}

/// Subset of simplification rule flags exposed through JSON.
#[derive(Debug, Deserialize)]
pub(in super::super) struct JsonSimplifyRules {
    #[serde(default)]
    pub arithmetic: bool,
    #[serde(default)]
    pub algebraic: bool,
    #[serde(default)]
    pub trigonometric: bool,
    #[serde(default)]
    pub logarithmic: bool,
    #[serde(default)]
    pub exponential: bool,
    #[serde(default)]
    pub hyperbolic: bool,
    #[serde(default)]
    pub rational: bool,
}

/// A `(old, new)` substitution pair.
#[derive(Debug, Deserialize)]
pub(in super::super) struct JsonBinding {
    pub old: Expression,
    pub new: Expression,
}

/// One `(var, order)` entry for `PartialDiff`.
#[derive(Debug, Deserialize)]
pub(in super::super) struct JsonPartialDiffVar {
    pub var: String,
    #[serde(default = "default_order")]
    pub order: u32,
}

fn default_order() -> u32 {
    1
}

/// A `(name, expression)` dependency declaration for `TotalDiff`.
#[derive(Debug, Deserialize)]
pub(in super::super) struct JsonDep {
    pub name: String,
    pub expr: Expression,
}

/// Initial-value data for `Ode`.
#[derive(Debug, Deserialize)]
pub(in super::super) struct JsonIvpData {
    pub var_at: Expression,
    pub fn_at: Expression,
    #[serde(default)]
    pub derivatives_at: Vec<Expression>,
}

/// Initial-value data for `OdeSystem`.
#[derive(Debug, Deserialize)]
pub(in super::super) struct JsonSystemIvpData {
    pub var_at: Expression,
    pub values_at: Vec<Expression>,
}

/// One constraint for `Optimize`.
#[derive(Debug, Deserialize)]
pub(in super::super) struct JsonConstraint {
    pub kind: String,
    pub expr: Expression,
}

/// A single operand for a `Matrix` command.
///
/// `Scalar` wraps a plain Expression; `Matrix` wraps a row-major grid;
/// `Vector` wraps a flat element list.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(in super::super) enum JsonMatrixOperand {
    Matrix { rows: Vec<Vec<Expression>> },
    Vector { elements: Vec<Expression> },
    Scalar(Expression),
}

/// Input payload for a `Nabla` command.
///
/// Scalar ops (Grad, Laplacian, CurlOfGrad, DivOfGrad) send a single
/// Expression; vector ops (Div, Curl, DivOfCurl) send an array.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(in super::super) enum JsonNablaInput {
    VectorField(Vec<Expression>),
    Scalar(Expression),
}

// ── Request wrapper ───────────────────────────────────────────────────────────

fn default_narrate() -> bool {
    true
}

/// JSON mirror of [`super::super::super::request::Request`].
#[derive(Debug, Deserialize)]
pub(in super::super) struct JsonRequest {
    pub command: JsonCommand,
    #[serde(default = "default_narrate")]
    pub narrate: bool,
    #[serde(default)]
    pub mode: Option<String>,
    pub precision: Option<JsonPrecision>,
    pub budget: Option<JsonBudget>,
    pub ambient_domain: Option<String>,
    pub seed: Option<u64>,
}

/// JSON representation of [`super::super::super::request::Precision`].
#[derive(Debug, Deserialize)]
pub(in super::super) struct JsonPrecision {
    pub decimal_digits: u32,
    pub abs_tol: Option<Expression>,
    pub rel_tol: Option<Expression>,
}

/// JSON representation of [`super::super::super::request::Budget`].
#[derive(Debug, Deserialize)]
pub(in super::super) struct JsonBudget {
    pub max_wall_ms: Option<u64>,
    pub max_iterations: Option<u64>,
}
