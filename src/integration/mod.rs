//! Symbolic integration module for computing indefinite integrals.
//!
//! Thin adapter over the `Arc<Expr>`-native pattern-based integrator in
//! [`crate::numeric::pattern_integrate`]. The public API here is still
//! [`Expression`]-typed (the display boundary); the engine is `Arc<Expr>`.
//!
//! Strategy (as applied by [`pattern_integrate`]):
//!
//! 1. Table lookup for known elementary patterns.
//! 2. U-substitution heuristic.
//! 3. Tabular integration by parts for `xⁿ · trig`.
//! 4. Risch algorithm as fallback (rational + logarithmic + exponential towers).
//!
//! # Example
//!
//! ```
//! use thales::integration::integrate;
//! use thales::ast::{Expression, Variable};
//!
//! // Integrate x^2 with respect to x
//! let x = Expression::Variable(Variable::new("x"));
//! let x_squared = Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(2)));
//!
//! let result = integrate(&x_squared, "x").unwrap();
//! // Result is the antiderivative (constant of integration handled separately).
//! ```

mod by_parts;
mod definite;
mod substitution;

use crate::ast::Expression;
use crate::numeric::compile::{compile, decompile};
use crate::numeric::pattern_integrate::pattern_integrate;
use crate::numeric::risch::IntegrationResult as NumericIntegrationResult;
use crate::numeric::SymbolId;
use std::fmt;

// Re-export public API
pub use by_parts::{integrate_by_parts, integrate_by_parts_with_steps, tabular_integration};
pub use definite::{
    definite_integral, definite_integral_with_fallback, definite_integral_with_steps,
    improper_integral_to_infinity, numerical_integrate,
};
pub use substitution::{integrate_by_substitution, integrate_with_substitution};

/// Error types that can occur during integration.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum IntegrationError {
    /// Integration cannot be performed symbolically.
    CannotIntegrate(String),
    /// The integrand contains unsupported constructs.
    UnsupportedExpression(String),
    /// Division by zero would occur.
    DivisionByZero,
}

impl fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntegrationError::CannotIntegrate(msg) => write!(f, "Cannot integrate: {}", msg),
            IntegrationError::UnsupportedExpression(msg) => {
                write!(f, "Unsupported expression: {}", msg)
            }
            IntegrationError::DivisionByZero => write!(f, "Division by zero in integration"),
        }
    }
}

impl std::error::Error for IntegrationError {}

/// Result type for integration operations.
pub type IntegrationResult = Result<Expression, IntegrationError>;

/// Compute the indefinite integral of an expression with respect to a variable.
///
/// This function returns the antiderivative without the constant of integration.
/// The caller should add `+ C` if displaying the full indefinite integral.
///
/// # Arguments
///
/// * `expr` - The expression to integrate
/// * `var` - The variable of integration (e.g., "x")
///
/// # Returns
///
/// The antiderivative expression, or an error if the engine cannot produce a
/// closed-form elementary result.
pub fn integrate(expr: &Expression, var: &str) -> IntegrationResult {
    integrate_impl(expr, var)
}

/// Internal implementation. Compiles `expr` into `Arc<Expr>`, dispatches to
/// [`pattern_integrate`], and decompiles the result back to [`Expression`].
pub(crate) fn integrate_impl(expr: &Expression, var: &str) -> IntegrationResult {
    let var_id = SymbolId::intern(var);
    let compiled = compile(expr);
    match pattern_integrate(&compiled, var_id) {
        NumericIntegrationResult::Elementary(anti) => Ok(decompile(&anti)),
        NumericIntegrationResult::NonElementary(_) => Err(IntegrationError::CannotIntegrate(
            "Integrand has no elementary antiderivative".to_string(),
        )),
        NumericIntegrationResult::Partial { .. } => {
            Err(IntegrationError::CannotIntegrate(format!(
                "No closed-form elementary antiderivative found for {}",
                expr
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Expression, Function, UnaryOp, Variable};

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn pow(base: Expression, exp: Expression) -> Expression {
        Expression::Power(Box::new(base), Box::new(exp))
    }

    fn add(left: Expression, right: Expression) -> Expression {
        Expression::Binary(BinaryOp::Add, Box::new(left), Box::new(right))
    }

    fn mul(left: Expression, right: Expression) -> Expression {
        Expression::Binary(BinaryOp::Mul, Box::new(left), Box::new(right))
    }

    fn div(left: Expression, right: Expression) -> Expression {
        Expression::Binary(BinaryOp::Div, Box::new(left), Box::new(right))
    }

    // Evaluate an expression numerically at `var = value`; returns f64 or panics.
    fn eval_at(e: &Expression, var_name: &str, value: f64) -> f64 {
        let mut env = std::collections::HashMap::new();
        env.insert(var_name.to_string(), value);
        e.evaluate(&env).expect("expression evaluates to a number")
    }

    // Check that the antiderivative's derivative matches the integrand at a
    // handful of sample points.
    fn check_antiderivative(expr: &Expression, antideriv: &Expression, var_name: &str) {
        let derivative = antideriv.differentiate(var_name).simplify();
        for &pt in &[0.3f64, 1.7, 2.4] {
            let got = eval_at(&derivative, var_name, pt);
            let expected = eval_at(expr, var_name, pt);
            assert!(
                (got - expected).abs() < 1e-8,
                "d/dx(F) = {} expected {} at x={}; got F = {}, dF = {}",
                got,
                expected,
                pt,
                antideriv,
                derivative
            );
        }
    }

    #[test]
    fn test_integrate_constant() {
        // ∫5 dx = 5x
        let result = integrate(&int(5), "x").unwrap();
        check_antiderivative(&int(5), &result, "x");
    }

    #[test]
    fn test_integrate_x() {
        // ∫x dx = x^2/2
        let result = integrate(&var("x"), "x").unwrap();
        check_antiderivative(&var("x"), &result, "x");
    }

    #[test]
    fn test_integrate_x_squared() {
        // ∫x^2 dx = x^3/3
        let x_squared = pow(var("x"), int(2));
        let result = integrate(&x_squared, "x").unwrap();
        check_antiderivative(&x_squared, &result, "x");
    }

    #[test]
    fn test_integrate_sum() {
        // ∫(x^2 + x) dx = x^3/3 + x^2/2
        let expr = add(pow(var("x"), int(2)), var("x"));
        let result = integrate(&expr, "x").unwrap();
        check_antiderivative(&expr, &result, "x");
    }

    #[test]
    fn test_integrate_constant_multiple() {
        // ∫3x dx = 3x^2/2
        let expr = mul(int(3), var("x"));
        let result = integrate(&expr, "x").unwrap();
        check_antiderivative(&expr, &result, "x");
    }

    #[test]
    fn test_integrate_reciprocal() {
        // ∫1/x dx = ln(x)  (the engine emits ln(x); |·| wrapping is a display concern)
        let expr = div(int(1), var("x"));
        let result = integrate(&expr, "x").unwrap();
        // Verify d/dx(result) = 1/x at x = 2
        let derivative = result.differentiate("x").simplify();
        let got = eval_at(&derivative, "x", 2.0);
        assert!((got - 0.5).abs() < 1e-8, "d/dx of result = {}", got);
    }

    #[test]
    fn test_integrate_sin() {
        // ∫sin(x) dx = -cos(x)
        let sin_x = Expression::Function(Function::Sin, vec![var("x")]);
        let result = integrate(&sin_x, "x").unwrap();
        check_antiderivative(&sin_x, &result, "x");
    }

    #[test]
    fn test_integrate_cos() {
        // ∫cos(x) dx = sin(x)
        let cos_x = Expression::Function(Function::Cos, vec![var("x")]);
        let result = integrate(&cos_x, "x").unwrap();
        check_antiderivative(&cos_x, &result, "x");
    }

    #[test]
    fn test_integrate_exp() {
        // ∫e^x dx = e^x
        let exp_x = Expression::Function(Function::Exp, vec![var("x")]);
        let result = integrate(&exp_x, "x").unwrap();
        check_antiderivative(&exp_x, &result, "x");
    }

    #[test]
    fn test_integrate_x_power_negative_one() {
        // ∫x^(-1) dx = ln(x)
        let expr = pow(var("x"), int(-1));
        let result = integrate(&expr, "x").unwrap();
        // Verify d/dx(result) = 1/x at x = 3
        let derivative = result.differentiate("x").simplify();
        let got = eval_at(&derivative, "x", 3.0);
        assert!((got - 1.0 / 3.0).abs() < 1e-8, "d/dx of result = {}", got);
    }

    #[test]
    fn test_integrate_polynomial() {
        // ∫(x^2 + 2x + 1) dx
        let poly = add(add(pow(var("x"), int(2)), mul(int(2), var("x"))), int(1));
        let result = integrate(&poly, "x").unwrap();
        check_antiderivative(&poly, &result, "x");
    }

    #[test]
    fn test_integrate_linear_sin() {
        // ∫sin(2x) dx = -cos(2x)/2
        let two_x = mul(int(2), var("x"));
        let sin_2x = Expression::Function(Function::Sin, vec![two_x]);
        let result = integrate(&sin_2x, "x").unwrap();
        check_antiderivative(&sin_2x, &result, "x");
    }

    #[test]
    fn test_differentiate_integral_equals_original() {
        // Fundamental theorem: d/dx(∫f dx) = f, tested over samples
        let x_squared = pow(var("x"), int(2));
        let integral = integrate(&x_squared, "x").unwrap();
        check_antiderivative(&x_squared, &integral, "x");
    }

    // =========================================================================
    // U-Substitution Tests (via public integrate path)
    // =========================================================================

    #[test]
    fn test_integrate_by_substitution_linear() {
        // ∫sin(3x) dx = -cos(3x)/3
        let three_x = mul(int(3), var("x"));
        let sin_3x = Expression::Function(Function::Sin, vec![three_x.clone()]);
        let result = integrate_by_substitution(&sin_3x, "x").unwrap();
        check_antiderivative(&sin_3x, &result, "x");
    }

    #[test]
    fn test_integrate_by_substitution_exp() {
        // ∫e^(2x) dx = e^(2x)/2
        let two_x = mul(int(2), var("x"));
        let exp_2x = Expression::Function(Function::Exp, vec![two_x.clone()]);
        let result = integrate_by_substitution(&exp_2x, "x").unwrap();
        check_antiderivative(&exp_2x, &result, "x");
    }

    #[test]
    fn test_integrate_with_substitution_steps() {
        let x_squared = pow(var("x"), int(2));
        let (result, steps) = integrate_with_substitution(&x_squared, "x").unwrap();
        assert!(!steps.is_empty());
        check_antiderivative(&x_squared, &result, "x");
    }

    // =========================================================================
    // Integration by Parts Tests
    // =========================================================================

    #[test]
    fn test_integrate_by_parts_x_exp() {
        // ∫x * e^x dx = (x - 1) * e^x
        let x = var("x");
        let exp_x = Expression::Function(Function::Exp, vec![x.clone()]);
        let expr = mul(x.clone(), exp_x.clone());
        let result = integrate_by_parts(&expr, "x").unwrap();
        check_antiderivative(&expr, &result, "x");
    }

    #[test]
    fn test_integrate_by_parts_ln_x() {
        // ∫ln(x) dx = x*ln(x) - x
        let ln_x = Expression::Function(Function::Ln, vec![var("x")]);
        let result = integrate(&ln_x, "x").unwrap();
        check_antiderivative(&ln_x, &result, "x");
    }

    #[test]
    fn test_integrate_by_parts_x_sin() {
        // ∫x * sin(x) dx = -x*cos(x) + sin(x)
        let x = var("x");
        let sin_x = Expression::Function(Function::Sin, vec![x.clone()]);
        let expr = mul(x.clone(), sin_x.clone());
        let result = integrate_by_parts(&expr, "x").unwrap();
        check_antiderivative(&expr, &result, "x");
    }

    #[test]
    fn test_integrate_by_parts_x_squared_exp() {
        // ∫x^2 * e^x dx
        let x = var("x");
        let x_squared = pow(x.clone(), int(2));
        let exp_x = Expression::Function(Function::Exp, vec![x.clone()]);
        let expr = mul(x_squared.clone(), exp_x.clone());
        let result = integrate_by_parts(&expr, "x").unwrap();
        check_antiderivative(&expr, &result, "x");
    }

    #[test]
    fn test_tabular_integration_x_exp() {
        // ∫x * e^x dx via tabular
        let x = var("x");
        let exp_x = Expression::Function(Function::Exp, vec![x.clone()]);
        let result = tabular_integration(&x, &exp_x, "x").unwrap();
        let integrand = mul(x.clone(), exp_x.clone());
        check_antiderivative(&integrand, &result, "x");
    }

    #[test]
    fn test_tabular_integration_x_squared_exp() {
        let x = var("x");
        let x_squared = pow(x.clone(), int(2));
        let exp_x = Expression::Function(Function::Exp, vec![x.clone()]);
        let result = tabular_integration(&x_squared, &exp_x, "x").unwrap();
        let integrand = mul(x_squared.clone(), exp_x.clone());
        check_antiderivative(&integrand, &result, "x");
    }

    #[test]
    fn test_tabular_integration_x_sin() {
        let x = var("x");
        let sin_x = Expression::Function(Function::Sin, vec![x.clone()]);
        let result = tabular_integration(&x, &sin_x, "x").unwrap();
        let integrand = mul(x.clone(), sin_x.clone());
        check_antiderivative(&integrand, &result, "x");
    }

    #[test]
    fn test_integrate_by_parts_with_steps() {
        let x = var("x");
        let exp_x = Expression::Function(Function::Exp, vec![x.clone()]);
        let expr = mul(x.clone(), exp_x.clone());
        let (result, steps) = integrate_by_parts_with_steps(&expr, "x").unwrap();
        assert!(!steps.is_empty());
        check_antiderivative(&expr, &result, "x");
    }

    // =========================================================================
    // Definite Integral Tests
    // =========================================================================

    #[test]
    fn test_definite_integral_x_squared() {
        // ∫_0^1 x^2 dx = 1/3
        let x_squared = pow(var("x"), int(2));
        let result = definite_integral(&x_squared, "x", &int(0), &int(1)).unwrap();
        let empty = std::collections::HashMap::new();
        let numeric = result.evaluate(&empty).unwrap();
        assert!((numeric - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_definite_integral_sin() {
        // ∫_0^π sin(x) dx = 2
        let sin_x = Expression::Function(Function::Sin, vec![var("x")]);
        let pi = Expression::Constant(crate::ast::SymbolicConstant::Pi);
        let result = definite_integral(&sin_x, "x", &int(0), &pi).unwrap();
        let empty = std::collections::HashMap::new();
        let numeric = result.evaluate(&empty).unwrap();
        assert!((numeric - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_definite_integral_odd_function() {
        // ∫_{-1}^1 x^3 dx = 0
        let x_cubed = pow(var("x"), int(3));
        let result = definite_integral(&x_cubed, "x", &int(-1), &int(1)).unwrap();
        let empty = std::collections::HashMap::new();
        let numeric = result.evaluate(&empty).unwrap();
        assert!(numeric.abs() < 1e-10);
    }

    #[test]
    fn test_definite_integral_symbolic_upper_bound() {
        // ∫_0^a x dx = a^2/2; at a=2 gives 2
        let x = var("x");
        let a = var("a");
        let result = definite_integral(&x, "x", &int(0), &a).unwrap();
        let mut env = std::collections::HashMap::new();
        env.insert("a".to_string(), 2.0);
        let numeric = result.evaluate(&env).unwrap();
        assert!((numeric - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_definite_integral_polynomial() {
        // ∫_0^2 (3x^2 + 2x + 1) dx = 14
        let x = var("x");
        let poly = add(
            add(mul(int(3), pow(x.clone(), int(2))), mul(int(2), x.clone())),
            int(1),
        );
        let result = definite_integral(&poly, "x", &int(0), &int(2)).unwrap();
        let empty = std::collections::HashMap::new();
        let numeric = result.evaluate(&empty).unwrap();
        assert!((numeric - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_definite_integral_cos() {
        // ∫_0^{π/2} cos(x) dx = 1
        let cos_x = Expression::Function(Function::Cos, vec![var("x")]);
        let pi = Expression::Constant(crate::ast::SymbolicConstant::Pi);
        let upper = div(pi, int(2));
        let result = definite_integral(&cos_x, "x", &int(0), &upper).unwrap();
        let empty = std::collections::HashMap::new();
        let numeric = result.evaluate(&empty).unwrap();
        assert!((numeric - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_definite_integral_exp() {
        // ∫_0^1 e^x dx = e - 1
        let exp_x = Expression::Function(Function::Exp, vec![var("x")]);
        let result = definite_integral(&exp_x, "x", &int(0), &int(1)).unwrap();
        let empty = std::collections::HashMap::new();
        let numeric = result.evaluate(&empty).unwrap();
        let expected = std::f64::consts::E - 1.0;
        assert!((numeric - expected).abs() < 1e-10);
    }

    #[test]
    fn test_numerical_integrate_simple() {
        // ∫_0^1 x^2 dx = 1/3 via Simpson
        let x_squared = pow(var("x"), int(2));
        let value = numerical_integrate(&x_squared, "x", 0.0, 1.0, 1e-8).unwrap();
        assert!((value - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_numerical_integrate_complex() {
        // ∫_0^1 exp(-x^2) dx ≈ 0.74682
        let x = var("x");
        let neg_x_squared = Expression::Unary(UnaryOp::Neg, Box::new(pow(x.clone(), int(2))));
        let exp_neg_x_squared = Expression::Function(Function::Exp, vec![neg_x_squared]);
        let value = numerical_integrate(&exp_neg_x_squared, "x", 0.0, 1.0, 1e-6).unwrap();
        assert!((value - 0.74682).abs() < 0.001);
    }

    #[test]
    fn test_definite_integral_with_fallback() {
        let x_squared = pow(var("x"), int(2));
        let value = definite_integral_with_fallback(&x_squared, "x", 0.0, 1.0, 1e-8).unwrap();
        assert!((value - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_improper_integral_convergent() {
        // ∫_1^∞ x^(-2) dx = 1
        let x = var("x");
        let x_neg_2 = pow(x.clone(), int(-2));
        let result = improper_integral_to_infinity(&x_neg_2, "x", &int(1)).unwrap();
        let empty = std::collections::HashMap::new();
        let numeric = result.evaluate(&empty).unwrap();
        assert!((numeric - 1.0).abs() < 1e-10);
    }

    // =========================================================================
    // Partial Fraction Integration Tests
    // =========================================================================

    #[test]
    fn test_integrate_rational_two_linear_factors() {
        // ∫ 1/(x²-1) dx — engine must produce something
        let denom = add(
            pow(var("x"), int(2)),
            Expression::Unary(UnaryOp::Neg, Box::new(int(1))),
        );
        let expr = div(int(1), denom);
        let result = integrate(&expr, "x").expect("1/(x²-1) integrates");
        check_antiderivative(&expr, &result, "x");
    }

    #[test]
    fn test_integrate_rational_linear_times_x() {
        // ∫ (2x+3)/(x(x+1)) dx
        let x = var("x");
        let denom = add(pow(x.clone(), int(2)), x.clone());
        let num = add(mul(int(2), x.clone()), int(3));
        let expr = div(num, denom);
        let result = integrate(&expr, "x").expect("(2x+3)/(x²+x) integrates");
        check_antiderivative(&expr, &result, "x");
    }

    #[test]
    fn test_integrate_rational_three_linear_factors() {
        // ∫ 1/((x-1)(x-2)(x-3)) dx
        let x = var("x");
        let x3_minus_6x2 = add(
            pow(x.clone(), int(3)),
            Expression::Unary(UnaryOp::Neg, Box::new(mul(int(6), pow(x.clone(), int(2))))),
        );
        let denom = add(
            add(x3_minus_6x2, mul(int(11), x.clone())),
            Expression::Unary(UnaryOp::Neg, Box::new(int(6))),
        );
        let expr = div(int(1), denom);
        let result = integrate(&expr, "x").expect("rational integrates");
        check_antiderivative(&expr, &result, "x");
    }

    #[test]
    fn test_definite_integral_with_steps() {
        let x_squared = pow(var("x"), int(2));
        let (value, steps) =
            definite_integral_with_steps(&x_squared, "x", &int(0), &int(1)).unwrap();
        assert!(!steps.is_empty());
        let empty = std::collections::HashMap::new();
        let numeric = value.evaluate(&empty).unwrap();
        assert!((numeric - 1.0 / 3.0).abs() < 1e-10);
    }
}
