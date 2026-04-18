//! Integration-by-parts entry points.
//!
//! These functions preserve the legacy public signatures but delegate to the
//! unified pattern-based integrator. The `Arc<Expr>` engine recognises
//! integration-by-parts patterns internally (tabular trig/polynomial, Risch),
//! so this module is a thin Expression-typed facade.
//!
//! The "with steps" variants return a minimal narration rather than the full
//! hand-rolled LIATE trace of the legacy implementation.

use crate::ast::{BinaryOp, Expression};

use super::{IntegrationError, IntegrationResult};

/// Attempt to integrate using integration by parts: ∫u dv = uv - ∫v du
///
/// Delegates to the unified integrator, which handles parts patterns
/// internally (tabular integration for `xⁿ · trig`, Risch for general cases).
pub fn integrate_by_parts(expr: &Expression, var: &str) -> IntegrationResult {
    super::integrate_impl(expr, var)
}

/// Integration by parts with a minimal step narration.
pub fn integrate_by_parts_with_steps(
    expr: &Expression,
    var: &str,
) -> Result<(Expression, Vec<String>), IntegrationError> {
    let result = super::integrate_impl(expr, var)?;
    let steps = vec![
        format!("∫{} d{}", expr, var),
        "Using integration by parts: ∫u dv = uv - ∫v du".to_string(),
        format!("Result: {}", result),
    ];
    Ok((result, steps))
}

/// Tabular method for integration by parts: ∫P(x)·f(x) dx.
///
/// Delegates to the unified integrator after forming the product `P(x)·f(x)`.
/// The `Arc<Expr>`-native tabular pass in `pattern_integrate` recognises
/// this pattern for small-degree polynomials with `sin`/`cos`/`exp`.
pub fn tabular_integration(
    polynomial: &Expression,
    integrable: &Expression,
    var: &str,
) -> IntegrationResult {
    let product = Expression::Binary(
        BinaryOp::Mul,
        Box::new(polynomial.clone()),
        Box::new(integrable.clone()),
    );
    super::integrate_impl(&product, var)
}
