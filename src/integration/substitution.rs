//! U-substitution entry points.
//!
//! These functions preserve the legacy public signatures but delegate to the
//! unified pattern-based integrator. The `Arc<Expr>` engine performs
//! u-substitution internally via `pattern_integrate::try_u_substitution`.
//!
//! The "with steps" variant returns a minimal narration rather than the full
//! hand-rolled substitution trace of the legacy implementation.

use crate::ast::Expression;

use super::{IntegrationError, IntegrationResult};

/// Attempt to integrate using u-substitution.
///
/// Delegates to the unified integrator, which runs u-substitution as a
/// pattern-matching pass before falling through to Risch.
// TODO(arc-migration): caller (integration tests) still Expression-native — drop Expression wrapper when callers migrated
pub fn integrate_by_substitution(expr: &Expression, var: &str) -> IntegrationResult {
    super::integrate_impl(expr, var)
}

/// U-substitution with a minimal step narration.
// TODO(arc-migration): caller (integration tests) still Expression-native — drop Expression wrapper when callers migrated
pub fn integrate_with_substitution(
    expr: &Expression,
    var: &str,
) -> Result<(Expression, Vec<String>), IntegrationError> {
    let result = super::integrate_impl(expr, var)?;
    let steps = vec![
        format!("∫{} d{}", expr, var),
        format!("Attempting u-substitution for ∫{} d{}", expr, var),
        format!("Result: {}", result),
    ];
    Ok((result, steps))
}
