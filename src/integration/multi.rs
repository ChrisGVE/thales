//! Iterated (multiple) definite integration engine.
//!
//! Evaluates an iterated integral of the form
//!
//! ```text
//! ∫_{a_n}^{b_n} ... ∫_{a_1}^{b_1} f dx_1 ... dx_n
//! ```
//!
//! where the [`IntegrationStep`] slice is ordered inner-to-outer (step 0 is
//! the innermost variable). Each step applies [`definite_integral`] on the
//! result of the previous step.

use crate::api::command::IntegrationStep;
use crate::ast::Expression;

use super::{definite_integral, IntegrationError};

/// Evaluate an iterated definite integral.
///
/// `steps` must be non-empty and ordered inner-to-outer:
/// - `steps[0]` is the innermost (first) integration variable.
/// - `steps[last]` is the outermost (last) integration variable.
///
/// Each step applies a definite integral of the current accumulator
/// expression with respect to `step.var` from `step.from` to `step.to`.
/// The result of one step becomes the integrand for the next.
///
/// # Errors
///
/// Returns [`IntegrationError::CannotIntegrate`] if any intermediate step
/// fails, wrapping the step index and inner error message.
pub fn multi_integrate(
    expr: &Expression,
    steps: &[IntegrationStep],
) -> Result<Expression, IntegrationError> {
    if steps.is_empty() {
        return Err(IntegrationError::CannotIntegrate(
            "MultiIntegrate requires at least one integration step".to_string(),
        ));
    }

    let mut current = expr.clone();

    for (i, step) in steps.iter().enumerate() {
        current = definite_integral(&current, &step.var, &step.from, &step.to)
            .map_err(|e| IntegrationError::CannotIntegrate(format!("step {}: {}", i, e)))?;
    }

    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::command::IntegrationStep;
    use crate::ast::{BinaryOp, Expression, Variable};

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn mul(a: Expression, b: Expression) -> Expression {
        Expression::Binary(BinaryOp::Mul, Box::new(a), Box::new(b))
    }

    fn step(v: &str, from: Expression, to: Expression) -> IntegrationStep {
        IntegrationStep {
            var: v.to_string(),
            from,
            to,
        }
    }

    fn eval(e: &Expression) -> f64 {
        e.evaluate(&std::collections::HashMap::new())
            .expect("expression evaluates to a number")
    }

    // ∫₀¹ ∫₀¹ 1 dx dy = 1
    #[test]
    fn fast_double_integral_constant_one() {
        let expr = int(1);
        let steps = vec![step("x", int(0), int(1)), step("y", int(0), int(1))];
        let result = multi_integrate(&expr, &steps).expect("∫₀¹∫₀¹ 1 dx dy should succeed");
        let value = eval(&result);
        assert!(
            (value - 1.0).abs() < 1e-10,
            "∫₀¹∫₀¹ 1 dx dy = 1; got {}",
            value
        );
    }

    // ∫₀¹ ∫₀¹ x*y dx dy = 1/4
    #[test]
    fn fast_double_integral_xy() {
        let expr = mul(var("x"), var("y"));
        let steps = vec![step("x", int(0), int(1)), step("y", int(0), int(1))];
        let result = multi_integrate(&expr, &steps).expect("∫₀¹∫₀¹ x*y dx dy should succeed");
        let value = eval(&result);
        assert!(
            (value - 0.25).abs() < 1e-10,
            "∫₀¹∫₀¹ x*y dx dy = 0.25; got {}",
            value
        );
    }

    // Empty steps must return an error.
    #[test]
    fn fast_empty_steps_returns_error() {
        let result = multi_integrate(&int(1), &[]);
        assert!(result.is_err(), "empty steps should return an error");
    }

    // Single step degenerates to a plain definite integral: ∫₀¹ x dx = 1/2.
    #[test]
    fn fast_single_step_degenerates_to_def_integrate() {
        let expr = var("x");
        let steps = vec![step("x", int(0), int(1))];
        let result = multi_integrate(&expr, &steps).expect("∫₀¹ x dx should succeed");
        let value = eval(&result);
        assert!((value - 0.5).abs() < 1e-10, "∫₀¹ x dx = 0.5; got {}", value);
    }
}
