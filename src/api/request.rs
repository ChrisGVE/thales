//! [`Request`] — input to [`super::execute`].
//!
//! Wraps a [`super::Command`] with execution-wide knobs: whether to record a
//! narrated trace, what solve mode to run under, optional precision / budget /
//! seed / output unit system / ambient domain. All fields except `command`
//! are optional or have defaults, so the minimal call is
//! `Request { command, ..Default::default() }`.

use crate::Expression;

use super::{Command, Domain};

/// Caller-provided input to [`super::execute`].
#[derive(Debug, Clone)]
pub struct Request {
    /// Operation plus its `Expression` inputs.
    pub command: Command,
    /// When `true`, engines record a narrated [`super::NarratedStep`] trace.
    /// When `false`, engines skip trace allocation entirely — caller only
    /// needs the raw result. Default: `true`.
    pub narrate: bool,
    /// Symbolic / numeric / hybrid preference. See [`SolveMode`].
    pub mode: SolveMode,
    /// Required when [`Self::mode`] is not [`SolveMode::Symbolic`].
    pub precision: Option<Precision>,
    /// Unit system for the output. When `None`, result units preserve the
    /// mixed input (feet, kilograms, hours, …) as received.
    pub output_units: Option<UnitSystem>,
    /// Default domain for symbols that carry no domain annotation. When
    /// `None`, unannotated symbols are treated as complex-valued (`ℂ`).
    pub ambient_domain: Option<Domain>,
    /// Time / iteration caps for heavy commands.
    pub budget: Option<Budget>,
    /// Seed for any randomised numeric method. When `None`, a fresh
    /// per-call seed is drawn.
    pub seed: Option<u64>,
}

impl Default for Request {
    fn default() -> Self {
        Self {
            command: Command::default(),
            narrate: true,
            mode: SolveMode::default(),
            precision: None,
            output_units: None,
            ambient_domain: None,
            budget: None,
            seed: None,
        }
    }
}

/// Symbolic / numeric execution preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolveMode {
    /// Refuse any numeric fallback. If the engine cannot produce a closed-form
    /// symbolic result, the [`super::ResultValue`] is [`super::ResultValue::Unsolved`].
    #[default]
    Symbolic,
    /// Force numeric from the start; result is [`super::ResultValue::Numeric`].
    /// A [`Precision`] must be supplied on the [`Request`].
    Numeric,
    /// Attempt symbolic first; on failure, evaluate the farthest symbolic form
    /// numerically. Result is [`super::ResultValue::Hybrid`] or
    /// [`super::ResultValue::Symbolic`] depending on how far the engine got.
    /// A [`Precision`] must be supplied on the [`Request`].
    PreferSymbolic,
}

/// Numeric precision contract.
#[derive(Debug, Clone, PartialEq)]
pub struct Precision {
    /// Target decimal digits in the reported value.
    pub decimal_digits: u32,
    /// Optional absolute tolerance for convergence tests.
    pub abs_tol: Option<Expression>,
    /// Optional relative tolerance for convergence tests.
    pub rel_tol: Option<Expression>,
}

/// Wall-time and iteration caps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Budget {
    /// Maximum wall time in milliseconds. When exceeded, the engine returns
    /// a partial [`super::ResultValue::Unsolved`] plus a
    /// [`super::DiagnosticCode::TimeoutReached`] diagnostic.
    pub max_wall_ms: Option<u64>,
    /// Maximum iteration count for iterative engines. Same partial-result
    /// behaviour on exceed.
    pub max_iterations: Option<u64>,
}

/// Requested output unit system. Result values are converted from the input
/// unit soup to this system's conventional base or derived units.
#[derive(Debug, Clone, PartialEq)]
pub enum UnitSystem {
    /// Système international (kg, m, s, A, K, mol, cd, …).
    Si,
    /// Centimetre-gram-second.
    Cgs,
    /// Foot-pound-second.
    Imperial,
    /// Natural units (ħ = c = 1).
    Natural,
    /// User-supplied mapping from dimension to preferred unit expression.
    /// The payload is reserved for future expansion; v0.8.1 treats a
    /// non-empty `Custom` as `Si` and emits an informational diagnostic.
    Custom(Vec<CustomUnitOverride>),
}

/// Placeholder payload for [`UnitSystem::Custom`]. Fleshed out in v0.10.0 once
/// the `mathcore-units` crate lands.
#[derive(Debug, Clone, PartialEq)]
pub struct CustomUnitOverride {
    /// Textual identifier of the dimension being overridden (e.g. "length").
    pub dimension: String,
    /// Textual identifier of the preferred unit (e.g. "km").
    pub unit: String,
}
