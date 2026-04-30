//! Serde-derived JSON mirror types for request parsing.

use serde::Deserialize;
use serde_json::Value;

// ── Mirror enum ───────────────────────────────────────────────────────────────

/// JSON-layer mirror of [`super::super::super::command::Command`].
///
/// Every [`Command`] variant has exactly one corresponding `JsonCommand`
/// variant. The `_exhaustiveness_check` in `tests.rs` fails compilation if
/// they diverge. Expression inputs are `String`; they are parsed during
/// conversion in [`super::convert`].
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(in super::super) enum JsonCommand {
    // ── Placeholder ────────────────────────────────────────────────────
    Noop,

    // ── Algebra ────────────────────────────────────────────────────────
    Simplify {
        expr: String,
        rules: Option<JsonSimplifyRules>,
        over: Option<String>,
    },
    Expand {
        expr: String,
        target: Option<String>,
    },
    Factor {
        expr: String,
        over: Option<String>,
        target: Option<String>,
    },
    Substitute {
        expr: String,
        bindings: Vec<JsonBinding>,
    },
    CombineLikeTerms {
        expr: String,
        target: Option<String>,
    },
    CommonDenominator {
        expr: String,
        target: Option<String>,
    },
    PartialFractions {
        expr: String,
        var: String,
    },
    Rationalize {
        expr: String,
        target: Option<String>,
    },
    Conjugate {
        expr: String,
        target: Option<String>,
    },
    InverseFn {
        expr: String,
        var: String,
    },
    Rearrange {
        equation: String,
        solve_for: String,
    },
    ApplyIdentity {
        expr: String,
        identity: String,
        target: Option<String>,
    },

    // ── Solve ──────────────────────────────────────────────────────────
    SolveFor {
        relation: String,
        var: String,
        over: Option<String>,
    },
    SolveSystem {
        equations: Vec<String>,
        vars: Vec<String>,
        over: Option<String>,
    },

    // ── Differentiation ────────────────────────────────────────────────
    Diff {
        expr: String,
        var: String,
        order: Option<u32>,
    },
    PartialDiff {
        expr: String,
        vars: Vec<JsonPartialDiffVar>,
    },
    TotalDiff {
        expr: String,
        var: String,
        deps: Vec<JsonDep>,
    },
    Gradient {
        expr: String,
        vars: Vec<String>,
    },
    Divergence {
        field: Vec<String>,
        vars: Vec<String>,
    },
    Curl {
        field: Vec<String>,
        vars: Vec<String>,
    },
    Laplacian {
        expr: String,
        vars: Vec<String>,
    },
    Jacobian {
        fields: Vec<String>,
        vars: Vec<String>,
    },
    Hessian {
        expr: String,
        vars: Vec<String>,
    },
    DirectionalDiff {
        expr: String,
        vars: Vec<String>,
        direction: Vec<String>,
    },

    // ── Integration ────────────────────────────────────────────────────
    Integrate {
        expr: String,
        var: String,
    },
    DefIntegrate {
        expr: String,
        var: String,
        from: String,
        to: String,
    },

    // ── Limits ─────────────────────────────────────────────────────────
    Limit {
        expr: String,
        var: String,
        point: String,
        side: Option<String>,
    },

    // ── Expansions ─────────────────────────────────────────────────────
    Taylor {
        expr: String,
        var: String,
        center: String,
        order: Option<u32>,
    },
    Laurent {
        expr: String,
        var: String,
        center: String,
        order: Option<u32>,
    },
    Asymptotic {
        expr: String,
        var: String,
        order: Option<u32>,
    },
    Compose {
        outer: String,
        inner: String,
        var: String,
        order: Option<u32>,
    },
    Revert {
        expr: String,
        var: String,
        order: Option<u32>,
    },
    Puiseux {
        expr: String,
        var: String,
        center: Option<String>,
        order: Option<u32>,
    },
    Frobenius {
        ode: String,
        fn_name: String,
        var: String,
        point: Option<String>,
        order: Option<u32>,
    },
    Pade {
        expr: String,
        var: String,
        center: Option<String>,
        m: u32,
        n: u32,
    },
    Wkb {
        ode: String,
        fn_name: String,
        var: String,
        small_param: String,
        order: Option<u32>,
    },

    // ── Transforms ─────────────────────────────────────────────────────
    FourierSeries {
        expr: String,
        var: String,
        period: String,
        terms: Option<u32>,
    },
    Residue {
        expr: String,
        var: String,
        point: String,
    },

    // ── Special functions ──────────────────────────────────────────────
    SpecialFn {
        kind: String,
        args: Vec<String>,
    },

    // ── ODE ────────────────────────────────────────────────────────────
    Ode {
        equation: String,
        fn_name: String,
        var: String,
        ic: Option<JsonIvpData>,
    },

    // ── Matrix ─────────────────────────────────────────────────────────
    Matrix {
        op: String,
        operands: Option<Vec<Value>>,
    },

    // ── Nabla ──────────────────────────────────────────────────────────
    /// JSON mirror of [`super::super::super::command::Command::Nabla`].
    ///
    /// `op` is one of: `"Grad"`, `"Div"`, `"Curl"`, `"Laplacian"`,
    /// `"DivOfCurl"`, `"CurlOfGrad"`, `"DivOfGrad"`.
    ///
    /// `input` is either a plain expression string (scalar ops) or an
    /// array of expression strings (vector-field ops).
    Nabla {
        op: String,
        input: Value,
        vars: Vec<String>,
    },

    // ── Optimization ───────────────────────────────────────────────────
    Optimize {
        objective: String,
        vars: Vec<String>,
        constraints: Option<Vec<JsonConstraint>>,
        sense: Option<String>,
    },
    LagrangeMult {
        objective: String,
        vars: Vec<String>,
        equality_constraints: Vec<String>,
    },
}

// ── Supporting serde types ────────────────────────────────────────────────────

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
    pub old: String,
    pub new: String,
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
    pub expr: String,
}

/// Initial-value data for `Ode`.
#[derive(Debug, Deserialize)]
pub(in super::super) struct JsonIvpData {
    pub var_at: String,
    pub fn_at: String,
    #[serde(default)]
    pub derivatives_at: Vec<String>,
}

/// One constraint for `Optimize`.
#[derive(Debug, Deserialize)]
pub(in super::super) struct JsonConstraint {
    pub kind: String,
    pub expr: String,
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
    pub abs_tol: Option<String>,
    pub rel_tol: Option<String>,
}

/// JSON representation of [`super::super::super::request::Budget`].
#[derive(Debug, Deserialize)]
pub(in super::super) struct JsonBudget {
    pub max_wall_ms: Option<u64>,
    pub max_iterations: Option<u64>,
}
