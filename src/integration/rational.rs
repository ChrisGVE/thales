//! Integration of rational functions via partial fraction decomposition.
//!
//! When a quotient `p(x)/q(x)` is encountered and both are polynomials in the
//! integration variable, this module decomposes the rational function into
//! partial fractions and integrates each simpler term.

use crate::ast::{Expression, Variable};
use crate::partial_fractions::decompose;

use super::helpers::is_polynomial_like;
use super::IntegrationError;
use super::IntegrationResult;

/// Attempt to integrate a rational function `num/denom` using partial
/// fraction decomposition.
///
/// Returns `None` when the expression is not a rational function in `var`
/// (so the caller can try other strategies), or `Some(Err(_))` when
/// decomposition is supported but fails.
///
/// # Example
///
/// ```
/// use thales::integration::integrate;
/// use thales::ast::{BinaryOp, Expression, Variable};
///
/// // ∫ 1/(x²-1) dx
/// let x = Expression::Variable(Variable::new("x"));
/// let num = Expression::Integer(1);
/// let denom = Expression::Binary(
///     BinaryOp::Sub,
///     Box::new(Expression::Power(
///         Box::new(x.clone()),
///         Box::new(Expression::Integer(2)),
///     )),
///     Box::new(Expression::Integer(1)),
/// );
/// let expr = Expression::Binary(BinaryOp::Div, Box::new(num), Box::new(denom));
/// let result = integrate(&expr, "x");
/// assert!(result.is_ok());
/// ```
pub(super) fn try_partial_fraction_integration(
    num: &Expression,
    denom: &Expression,
    var: &str,
) -> Option<IntegrationResult> {
    // Only attempt when both parts are polynomials in the integration variable.
    if !is_polynomial_like(num, var) || !is_polynomial_like(denom, var) {
        return None;
    }

    // The denominator must actually depend on `var`; constant denominators
    // are already handled upstream.
    if !denom.contains_variable(var) {
        return None;
    }

    let variable = Variable::new(var);
    match decompose(num, denom, &variable) {
        Ok(pf_result) => Some(Ok(pf_result.integrate())),
        Err(e) => Some(Err(IntegrationError::CannotIntegrate(format!(
            "Partial fraction decomposition failed: {}",
            e
        )))),
    }
}
