//! LegacyEngine — wraps existing engine functions into the [`Strategy`] trait.
//!
//! Legacy dispatch arms return a `Result<Arc<Expr>, ThalesError>` (or a
//! [`Response`] built from one). [`LegacyEngine`] adapts a function with that
//! signature into a first-class [`Strategy`] so that legacy engines can
//! participate in the [`super::runner::SequentialRunner`] cascade without
//! modification.
//!
//! Error mapping:
//! - [`ThalesError::Parse`], [`ThalesError::Matrix`] (singular/misuse),
//!   [`ThalesError::Integration`] DivisionByZero, and similar hard structural
//!   failures → [`FailureReason::StructuralError`] (terminates cascade).
//! - All other errors → [`FailureReason::NoClosedForm`] (cascade continues).

use std::fmt;
use std::sync::Arc;

use crate::engine::context::SolveContext;
use crate::engine::reason::FailureReason;
use crate::engine::strategy::{Strategy, StrategyResult};
use crate::engine::trace_tree::{BranchReason, StrategyId, TraceNode};
use crate::numeric::Expr;
use crate::ThalesError;

// ── LegacyResult ──────────────────────────────────────────────────────────────

/// Return type for functions wrapped by [`LegacyEngine`].
pub type LegacyResult = Result<Arc<Expr>, ThalesError>;

// ── LegacyEngine ──────────────────────────────────────────────────────────────

/// Wraps a legacy engine function (returning `Result<Arc<Expr>, ThalesError>`)
/// as a [`Strategy`].
///
/// The wrapped function receives only the expression from the context.
/// If the engine needs additional parameters they should be captured in the
/// closure environment.
///
/// Priority and id are set at construction time; `applicable` always returns
/// `true` (legacy engines have no structural pre-check).
pub struct LegacyEngine {
    /// Stable identifier for this engine instance.
    id: StrategyId,
    /// Priority in the cascade (lower = tried first).
    priority: f64,
    /// The wrapped engine function.
    func: Box<dyn Fn(Arc<Expr>) -> LegacyResult + Send + Sync>,
}

impl LegacyEngine {
    /// Create a new [`LegacyEngine`] wrapper.
    ///
    /// - `id`: stable strategy identifier string.
    /// - `priority`: cascade priority (lower = tried first).
    /// - `func`: the engine function to wrap.
    pub fn new(
        id: &'static str,
        priority: f64,
        func: impl Fn(Arc<Expr>) -> LegacyResult + Send + Sync + 'static,
    ) -> Self {
        LegacyEngine {
            id: StrategyId(id),
            priority,
            func: Box::new(func),
        }
    }
}

impl fmt::Debug for LegacyEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LegacyEngine")
            .field("id", &self.id)
            .field("priority", &self.priority)
            .finish()
    }
}

impl Strategy for LegacyEngine {
    fn id(&self) -> StrategyId {
        self.id
    }

    /// Legacy engines always claim applicability; the engine itself decides
    /// whether the input is valid.
    fn applicable(&self, _ctx: &SolveContext) -> bool {
        true
    }

    fn priority(&self, _ctx: &SolveContext) -> f64 {
        self.priority
    }

    fn apply(&self, ctx: SolveContext) -> StrategyResult {
        match (self.func)(Arc::clone(&ctx.expr)) {
            Ok(expr) => StrategyResult::Solved {
                expr,
                trace: dummy_solved_trace(),
            },
            Err(err) => {
                if is_structural_error(&err) {
                    StrategyResult::Failed(FailureReason::StructuralError(Arc::new(err)))
                } else {
                    StrategyResult::Failed(FailureReason::NoClosedForm)
                }
            }
        }
    }
}

// ── Error classification ──────────────────────────────────────────────────────

/// Returns `true` for errors that represent hard structural / type failures
/// from which no further strategy could recover. The cascade terminates on
/// these. All other errors allow the cascade to continue.
///
/// Structural errors are those where the input itself is malformed or where
/// a mathematical impossibility (division by zero, singular matrix on inverse)
/// was detected — retrying with a different strategy cannot help.
fn is_structural_error(err: &ThalesError) -> bool {
    match err {
        // Parse errors are always structural: the input expression is invalid.
        ThalesError::Parse(_) => true,

        // Integration: only DivisionByZero is structural; other integration
        // failures (no closed form, unsupported patterns) allow cascade.
        ThalesError::Integration(e) => {
            matches!(e, crate::integration::IntegrationError::DivisionByZero)
        }

        // Matrix: empty matrix, dimension mismatch, or invalid operation
        // (e.g. determinant of non-square) — these are structural.
        ThalesError::Matrix(e) => {
            use crate::matrix::MatrixError;
            matches!(
                e,
                MatrixError::EmptyMatrix
                    | MatrixError::DimensionMismatch { .. }
                    | MatrixError::InvalidOperation(_)
                    | MatrixError::NonRectangular
            )
        }

        // All other variants: solver, numerical, limits, ODE, special-function,
        // inequality, evaluation, partial-fractions, LaTeX-parse, system
        // errors — treat as "no closed form" (cascade continues).
        _ => false,
    }
}

/// Build a minimal [`TraceNode`] for a successful legacy-engine result.
/// Legacy engines do not produce a rich trace, so we emit a no-child branch.
fn dummy_solved_trace() -> TraceNode {
    TraceNode::Branch {
        reason: BranchReason::StrategyCascade,
        children: vec![],
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mode::ExecutionMode;
    use crate::engine::resource::ResourceBudget;
    use crate::engine::runner::{SequentialRunner, StrategyRunner};
    use crate::numeric::SmallInt;

    fn int_expr(n: i64) -> Arc<Expr> {
        Arc::new(Expr::Integer(SmallInt::from(n)))
    }

    fn base_ctx() -> SolveContext {
        SolveContext::new(int_expr(1), ResourceBudget::unlimited())
    }

    // ── LegacyEngine wrapping tests ───────────────────────────────────────────

    #[test]
    fn fast_legacy_wraps_successful_function() {
        let engine = LegacyEngine::new("test::double", 1.0, |expr| Ok(Arc::clone(&expr)));
        let result = engine.apply(base_ctx());
        assert!(
            matches!(result, StrategyResult::Solved { .. }),
            "Expected Solved from successful function"
        );
    }

    #[test]
    fn fast_legacy_applicable_always_true() {
        let engine = LegacyEngine::new("test::always_applicable", 1.0, |e| Ok(e));
        assert!(engine.applicable(&base_ctx()));
    }

    #[test]
    fn fast_legacy_priority_returned() {
        let engine = LegacyEngine::new("test::priority", 3.14, |e| Ok(e));
        let p = engine.priority(&base_ctx());
        assert!((p - 3.14).abs() < f64::EPSILON);
    }

    #[test]
    fn fast_legacy_id_returned() {
        let engine = LegacyEngine::new("test::my_engine", 0.0, |e| Ok(e));
        assert_eq!(engine.id(), StrategyId("test::my_engine"));
    }

    #[test]
    fn fast_legacy_structural_error_parse() {
        let engine = LegacyEngine::new("test::parse_err", 0.0, |_| {
            Err(ThalesError::Parse(
                crate::parser::ParseError::UnexpectedCharacter { pos: 0, found: '!' },
            ))
        });
        let result = engine.apply(base_ctx());
        assert!(
            matches!(
                result,
                StrategyResult::Failed(FailureReason::StructuralError(_))
            ),
            "Parse error must map to StructuralError"
        );
    }

    #[test]
    fn fast_legacy_non_structural_error_maps_to_no_closed_form() {
        let engine = LegacyEngine::new("test::solver_err", 0.0, |_| {
            Err(ThalesError::Solver(crate::solver::SolverError::NoSolution))
        });
        let result = engine.apply(base_ctx());
        assert!(
            matches!(result, StrategyResult::Failed(FailureReason::NoClosedForm)),
            "Solver error must allow cascade to continue"
        );
    }

    #[test]
    fn fast_legacy_integration_division_by_zero_is_structural() {
        let engine = LegacyEngine::new("test::int_div0", 0.0, |_| {
            Err(ThalesError::Integration(
                crate::integration::IntegrationError::DivisionByZero,
            ))
        });
        let result = engine.apply(base_ctx());
        assert!(
            matches!(
                result,
                StrategyResult::Failed(FailureReason::StructuralError(_))
            ),
            "DivisionByZero must be structural"
        );
    }

    #[test]
    fn fast_legacy_matrix_empty_is_structural() {
        let engine = LegacyEngine::new("test::matrix_empty", 0.0, |_| {
            Err(ThalesError::Matrix(crate::matrix::MatrixError::EmptyMatrix))
        });
        let result = engine.apply(base_ctx());
        assert!(
            matches!(
                result,
                StrategyResult::Failed(FailureReason::StructuralError(_))
            ),
            "EmptyMatrix must be structural"
        );
    }

    #[test]
    fn fast_legacy_in_sequential_runner_succeeds() {
        // LegacyEngine plugs into the SequentialRunner cascade.
        let strategies: Vec<Box<dyn Strategy>> = vec![Box::new(LegacyEngine::new(
            "test::runner_test",
            0.0,
            |_| Ok(int_expr(99)),
        ))];
        let runner = SequentialRunner;
        let result = runner.run(base_ctx(), &strategies, ExecutionMode::Sequential);
        assert!(
            matches!(result, StrategyResult::Solved { .. }),
            "LegacyEngine must succeed in SequentialRunner"
        );
    }

    #[test]
    fn fast_legacy_debug_format() {
        let engine = LegacyEngine::new("test::debug", 0.5, |e| Ok(e));
        let s = format!("{:?}", engine);
        assert!(
            s.contains("LegacyEngine"),
            "Debug output must name the type"
        );
        assert!(s.contains("test::debug"), "Debug output must include id");
    }
}
