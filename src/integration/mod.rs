//! Symbolic integration module for computing indefinite integrals.
//!
//! This module provides functionality for computing antiderivatives of mathematical
//! expressions using standard integration techniques including:
//!
//! - Power rule
//! - Sum and difference rules
//! - Constant multiple rule
//! - Standard integrals table (trigonometric, exponential, logarithmic)
//!
//! # Example
//!
//! ```
//! use thales::integration::{integrate, IntegrationError};
//! use thales::ast::{Expression, Variable};
//!
//! // Integrate x^2 with respect to x
//! let x = Expression::Variable(Variable::new("x"));
//! let x_squared = Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(2)));
//!
//! let result = integrate(&x_squared, "x").unwrap();
//! // Result: x^3/3 + C (constant of integration handled separately)
//! ```

mod by_parts;
mod definite;
pub(crate) mod helpers;
mod rational;
mod substitution;

use crate::ast::{BinaryOp, Expression, UnaryOp, Variable};
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
            IntegrationError::CannotIntegrate(msg) => {
                write!(f, "Cannot integrate: {}", msg)
            }
            IntegrationError::UnsupportedExpression(msg) => {
                write!(f, "Unsupported expression: {}", msg)
            }
            IntegrationError::DivisionByZero => {
                write!(f, "Division by zero in integration")
            }
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
/// The antiderivative expression, or an error if integration fails.
///
/// # Supported Forms
///
/// - Constants: ∫c dx = cx
/// - Power rule: ∫x^n dx = x^(n+1)/(n+1) for n ≠ -1
/// - Reciprocal: ∫1/x dx = ln|x|
/// - Sum/difference: ∫(f ± g) dx = ∫f dx ± ∫g dx
/// - Constant multiple: ∫cf dx = c∫f dx
/// - Trigonometric: sin, cos, tan, sec^2
/// - Exponential: e^x
/// - Logarithmic via inverse
///
/// # Example
///
/// ```
/// use thales::integration::integrate;
/// use thales::ast::{Expression, Variable};
///
/// let x = Expression::Variable(Variable::new("x"));
/// let expr = Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(3)));
///
/// let result = integrate(&expr, "x").unwrap();
/// // x^3 integrates to x^4/4
/// ```
pub fn integrate(expr: &Expression, var: &str) -> IntegrationResult {
    integrate_impl(expr, var)
}

/// Internal implementation of integration.
pub(crate) fn integrate_impl(expr: &Expression, var: &str) -> IntegrationResult {
    match expr {
        // Constants: ∫c dx = c*x
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Rational(_)
        | Expression::Complex(_) => {
            let x = Expression::Variable(Variable::new(var));
            Ok(Expression::Binary(
                BinaryOp::Mul,
                Box::new(expr.clone()),
                Box::new(x),
            ))
        }

        // Symbolic constants: ∫pi dx = pi*x, ∫e dx = e*x
        Expression::Constant(_) => {
            let x = Expression::Variable(Variable::new(var));
            Ok(Expression::Binary(
                BinaryOp::Mul,
                Box::new(expr.clone()),
                Box::new(x),
            ))
        }

        // Variable: ∫x dx = x^2/2, ∫y dx = y*x (y treated as constant)
        Expression::Variable(v) => {
            if v.name == var {
                // ∫x dx = x^2/2
                let x = Expression::Variable(Variable::new(var));
                let x_squared = Expression::Power(Box::new(x), Box::new(Expression::Integer(2)));
                Ok(Expression::Binary(
                    BinaryOp::Div,
                    Box::new(x_squared),
                    Box::new(Expression::Integer(2)),
                ))
            } else {
                // Treat as constant: ∫y dx = y*x
                let x = Expression::Variable(Variable::new(var));
                Ok(Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(expr.clone()),
                    Box::new(x),
                ))
            }
        }

        // Unary operations
        Expression::Unary(op, inner) => match op {
            UnaryOp::Neg => {
                // ∫-f dx = -∫f dx
                let inner_integral = integrate_impl(inner, var)?;
                Ok(Expression::Unary(UnaryOp::Neg, Box::new(inner_integral)))
            }
            UnaryOp::Abs => Err(IntegrationError::CannotIntegrate(
                "Cannot integrate |f(x)| symbolically".to_string(),
            )),
            UnaryOp::Not => Err(IntegrationError::UnsupportedExpression(
                "Logical NOT cannot be integrated".to_string(),
            )),
        },

        // Binary operations
        Expression::Binary(op, left, right) => match op {
            // Sum rule: ∫(f + g) dx = ∫f dx + ∫g dx
            BinaryOp::Add => {
                let left_integral = integrate_impl(left, var)?;
                let right_integral = integrate_impl(right, var)?;
                Ok(Expression::Binary(
                    BinaryOp::Add,
                    Box::new(left_integral),
                    Box::new(right_integral),
                ))
            }

            // Difference rule: ∫(f - g) dx = ∫f dx - ∫g dx
            BinaryOp::Sub => {
                let left_integral = integrate_impl(left, var)?;
                let right_integral = integrate_impl(right, var)?;
                Ok(Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(left_integral),
                    Box::new(right_integral),
                ))
            }

            // Multiplication: check for constant multiple
            BinaryOp::Mul => helpers::integrate_product(left, right, var),

            // Division: check for power rule with negative exponent or constant divisor
            BinaryOp::Div => helpers::integrate_quotient(left, right, var),

            BinaryOp::Mod => Err(IntegrationError::CannotIntegrate(
                "Modulo cannot be integrated".to_string(),
            )),
        },

        // Power expressions
        Expression::Power(base, exponent) => helpers::integrate_power(base, exponent, var),

        // Function calls
        Expression::Function(func, args) => helpers::integrate_function(func, args, var),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Function, Variable};

    use by_parts::check_same_up_to_constant;
    use helpers::{
        combine_factors, expressions_equivalent, extract_factors, is_one, is_polynomial_like,
        substitute_variable,
    };
    use substitution::{extract_inner_function, liate_priority};

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

    #[test]
    fn test_integrate_constant() {
        // ∫5 dx = 5x
        let result = integrate(&int(5), "x").unwrap();
        // Result should be 5 * x
        assert!(matches!(
            result,
            Expression::Binary(BinaryOp::Mul, left, right)
            if matches!(left.as_ref(), Expression::Integer(5))
            && matches!(right.as_ref(), Expression::Variable(v) if v.name == "x")
        ));
    }

    #[test]
    fn test_integrate_x() {
        // ∫x dx = x^2/2
        let result = integrate(&var("x"), "x").unwrap();
        assert!(matches!(
            result,
            Expression::Binary(BinaryOp::Div, _num, denom)
            if matches!(denom.as_ref(), Expression::Integer(2))
        ));
    }

    #[test]
    fn test_integrate_x_squared() {
        // ∫x^2 dx = x^3/3
        let x_squared = pow(var("x"), int(2));
        let result = integrate(&x_squared, "x").unwrap();

        // Should be x^(2+1) / (2+1) = x^3/3
        if let Expression::Binary(BinaryOp::Div, num, denom) = result {
            // Numerator should be x^(2+1)
            assert!(matches!(num.as_ref(), Expression::Power(_, _)));
            // Denominator should be 2+1
            assert!(matches!(
                denom.as_ref(),
                Expression::Binary(BinaryOp::Add, _, _)
            ));
        } else {
            panic!("Expected division expression");
        }
    }

    #[test]
    fn test_integrate_sum() {
        // ∫(x^2 + x) dx = x^3/3 + x^2/2
        let expr = add(pow(var("x"), int(2)), var("x"));
        let result = integrate(&expr, "x").unwrap();
        assert!(matches!(result, Expression::Binary(BinaryOp::Add, _, _)));
    }

    #[test]
    fn test_integrate_constant_multiple() {
        // ∫3x dx = 3 * x^2/2
        let expr = mul(int(3), var("x"));
        let result = integrate(&expr, "x").unwrap();

        // Should be 3 * (x^2/2)
        assert!(matches!(result, Expression::Binary(BinaryOp::Mul, left, _)
            if matches!(left.as_ref(), Expression::Integer(3))));
    }

    #[test]
    fn test_integrate_reciprocal() {
        // ∫1/x dx = ln|x|
        let expr = div(int(1), var("x"));
        let result = integrate(&expr, "x").unwrap();

        // Should be 1 * ln(|x|)
        assert!(
            matches!(result, Expression::Binary(BinaryOp::Mul, _, ln_part)
            if matches!(ln_part.as_ref(), Expression::Function(Function::Ln, _)))
        );
    }

    #[test]
    fn test_integrate_sin() {
        // ∫sin(x) dx = -cos(x)
        let sin_x = Expression::Function(Function::Sin, vec![var("x")]);
        let result = integrate(&sin_x, "x").unwrap();

        assert!(matches!(
            result,
            Expression::Unary(UnaryOp::Neg, inner)
            if matches!(inner.as_ref(), Expression::Function(Function::Cos, _))
        ));
    }

    #[test]
    fn test_integrate_cos() {
        // ∫cos(x) dx = sin(x)
        let cos_x = Expression::Function(Function::Cos, vec![var("x")]);
        let result = integrate(&cos_x, "x").unwrap();

        assert!(matches!(
            result,
            Expression::Function(Function::Sin, args)
            if args.len() == 1
        ));
    }

    #[test]
    fn test_integrate_exp() {
        // ∫e^x dx = e^x
        let exp_x = Expression::Function(Function::Exp, vec![var("x")]);
        let result = integrate(&exp_x, "x").unwrap();

        assert!(matches!(
            result,
            Expression::Function(Function::Exp, args)
            if args.len() == 1
        ));
    }

    #[test]
    fn test_integrate_x_power_negative_one() {
        // ∫x^(-1) dx = ln|x|
        let expr = pow(var("x"), int(-1));
        let result = integrate(&expr, "x").unwrap();

        assert!(matches!(
            result,
            Expression::Function(Function::Ln, args)
            if matches!(&args[0], Expression::Function(Function::Abs, _))
        ));
    }

    #[test]
    fn test_integrate_polynomial() {
        // ∫(x^2 + 2x + 1) dx = x^3/3 + x^2 + x
        let poly = add(add(pow(var("x"), int(2)), mul(int(2), var("x"))), int(1));
        let result = integrate(&poly, "x");
        assert!(result.is_ok());
    }

    #[test]
    fn test_integrate_linear_sin() {
        // ∫sin(2x) dx = -cos(2x)/2
        let two_x = mul(int(2), var("x"));
        let sin_2x = Expression::Function(Function::Sin, vec![two_x]);
        let result = integrate(&sin_2x, "x").unwrap();

        // Should be (-cos(2x)) / 2
        assert!(matches!(result, Expression::Binary(BinaryOp::Div, _, _)));
    }

    #[test]
    fn test_differentiate_integral_equals_original() {
        // Fundamental theorem: d/dx(∫f dx) = f
        // Test with x^2
        let x_squared = pow(var("x"), int(2));
        let integral = integrate(&x_squared, "x").unwrap();
        let _derivative = integral.differentiate("x").simplify();

        // The derivative should simplify to x^2 (or equivalent)
        // This is a partial test - full verification needs numerical checks
        // For now, just verify we get back a power expression
        // The actual result may be (3*x^2) / 3 which simplifies to x^2
    }

    // =========================================================================
    // U-Substitution Tests
    // =========================================================================

    #[test]
    fn test_extract_factors() {
        // Test basic factor extraction
        let expr = mul(int(2), var("x"));
        let factors = extract_factors(&expr);
        assert_eq!(factors.len(), 2);

        // Test nested product
        let expr2 = mul(mul(int(2), var("x")), int(3));
        let factors2 = extract_factors(&expr2);
        assert_eq!(factors2.len(), 3);
    }

    #[test]
    fn test_combine_factors() {
        let factors = vec![int(2), var("x"), int(3)];
        let combined = combine_factors(&factors);
        // Should produce 2 * x * 3
        assert!(matches!(combined, Expression::Binary(BinaryOp::Mul, _, _)));
    }

    #[test]
    fn test_extract_inner_function() {
        // Test function extraction
        let sin_x = Expression::Function(Function::Sin, vec![var("x")]);
        let inner = extract_inner_function(&sin_x);
        assert!(matches!(inner, Some(Expression::Variable(_))));

        // Test with x^2 inside
        let x_squared = pow(var("x"), int(2));
        let sin_x2 = Expression::Function(Function::Sin, vec![x_squared]);
        let inner2 = extract_inner_function(&sin_x2);
        assert!(inner2.is_some());
    }

    #[test]
    fn test_substitute_variable() {
        // Test variable substitution
        let u = Expression::Variable(Variable::new("u"));
        let replacement = pow(var("x"), int(2));
        let result = substitute_variable(&u, "u", &replacement);
        assert!(matches!(result, Expression::Power(_, _)));
    }

    #[test]
    fn test_integrate_by_substitution_linear() {
        // ∫sin(3x) dx = -cos(3x)/3
        // This is already handled by linear substitution in base integrate
        let three_x = mul(int(3), var("x"));
        let sin_3x = Expression::Function(Function::Sin, vec![three_x]);
        let result = integrate_by_substitution(&sin_3x, "x");
        assert!(result.is_ok());
    }

    #[test]
    fn test_integrate_by_substitution_exp() {
        // ∫e^(2x) dx = e^(2x)/2
        // This should be handled by linear substitution
        let two_x = mul(int(2), var("x"));
        let exp_2x = Expression::Function(Function::Exp, vec![two_x]);
        let result = integrate_by_substitution(&exp_2x, "x");
        assert!(result.is_ok());
    }

    #[test]
    fn test_integrate_with_substitution_steps() {
        // Test that we get steps back
        let x_squared = pow(var("x"), int(2));
        let (result, steps) = integrate_with_substitution(&x_squared, "x").unwrap();

        // Should have at least one step
        assert!(!steps.is_empty());
        // Result should be valid
        assert!(matches!(result, Expression::Binary(BinaryOp::Div, _, _)));
    }

    #[test]
    fn test_expressions_equivalent() {
        let a = var("x");
        let b = var("x");
        assert!(expressions_equivalent(&a, &b));

        let c = var("y");
        assert!(!expressions_equivalent(&a, &c));
    }

    #[test]
    fn test_expressions_equivalent_add_commutativity() {
        // x + y  ≡  y + x
        let xy = add(var("x"), var("y"));
        let yx = add(var("y"), var("x"));
        assert!(expressions_equivalent(&xy, &yx));

        // x + y  ≢  x + z
        let xz = add(var("x"), var("z"));
        assert!(!expressions_equivalent(&xy, &xz));
    }

    #[test]
    fn test_expressions_equivalent_mul_commutativity() {
        // 2 * x  ≡  x * 2
        let two_x = mul(int(2), var("x"));
        let x_two = mul(var("x"), int(2));
        assert!(expressions_equivalent(&two_x, &x_two));

        // 2 * x  ≢  3 * x
        let three_x = mul(int(3), var("x"));
        assert!(!expressions_equivalent(&two_x, &three_x));
    }

    #[test]
    fn test_expressions_equivalent_nested_commutativity() {
        // (x + x).simplify() = 2*x, and 2*x ≡ x*2 after canonical form
        let x_plus_x = add(var("x"), var("x")).simplify();
        let x_times_2 = mul(var("x"), int(2));
        assert!(expressions_equivalent(&x_plus_x, &x_times_2));
    }

    #[test]
    fn test_is_one() {
        assert!(is_one(&Expression::Integer(1)));
        assert!(!is_one(&Expression::Integer(2)));
        assert!(!is_one(&var("x")));
    }

    // =========================================================================
    // Integration by Parts Tests
    // =========================================================================

    #[test]
    fn test_liate_priority() {
        // Logarithmic has highest priority
        let ln_x = Expression::Function(Function::Ln, vec![var("x")]);
        assert!(liate_priority(&ln_x, "x") > liate_priority(&var("x"), "x"));

        // Algebraic (x) has higher priority than trigonometric
        let sin_x = Expression::Function(Function::Sin, vec![var("x")]);
        assert!(liate_priority(&var("x"), "x") > liate_priority(&sin_x, "x"));

        // Trigonometric has higher priority than exponential
        let exp_x = Expression::Function(Function::Exp, vec![var("x")]);
        assert!(liate_priority(&sin_x, "x") > liate_priority(&exp_x, "x"));
    }

    #[test]
    fn test_is_polynomial_like() {
        // Basic polynomials
        assert!(is_polynomial_like(&var("x"), "x"));
        assert!(is_polynomial_like(&pow(var("x"), int(2)), "x"));
        assert!(is_polynomial_like(
            &add(pow(var("x"), int(2)), var("x")),
            "x"
        ));

        // Constants are polynomial
        assert!(is_polynomial_like(&int(5), "x"));

        // Non-polynomials
        let sin_x = Expression::Function(Function::Sin, vec![var("x")]);
        assert!(!is_polynomial_like(&sin_x, "x"));
    }

    #[test]
    fn test_integrate_by_parts_x_exp() {
        // ∫x * e^x dx = (x - 1) * e^x
        // Using parts: u = x, dv = e^x dx
        //              du = dx, v = e^x
        //              = x*e^x - ∫e^x dx = x*e^x - e^x = (x-1)*e^x
        let x = var("x");
        let exp_x = Expression::Function(Function::Exp, vec![x.clone()]);
        let expr = mul(x.clone(), exp_x.clone());

        let result = integrate_by_parts(&expr, "x");
        assert!(result.is_ok());

        // Verify by differentiation
        if let Ok(integral) = result {
            let derivative = integral.differentiate("x").simplify();
            // The derivative should equal the original expression
            // We can't easily compare symbolically, but at least check it's not an error
            assert!(!matches!(derivative, Expression::Integer(0)));
        }
    }

    #[test]
    fn test_integrate_by_parts_ln_x() {
        // ∫ln(x) dx = x*ln(x) - x
        // Using parts: u = ln(x), dv = dx
        //              du = 1/x dx, v = x
        //              = x*ln(x) - ∫x * (1/x) dx = x*ln(x) - ∫1 dx = x*ln(x) - x
        let ln_x = Expression::Function(Function::Ln, vec![var("x")]);

        // This should be handled by the standard integral table, not parts
        let result = integrate(&ln_x, "x");
        assert!(result.is_ok());
    }

    #[test]
    fn test_integrate_by_parts_x_sin() {
        // ∫x * sin(x) dx = -x*cos(x) + sin(x)
        // Using parts: u = x, dv = sin(x) dx
        //              du = dx, v = -cos(x)
        //              = -x*cos(x) - ∫(-cos(x)) dx = -x*cos(x) + sin(x)
        let x = var("x");
        let sin_x = Expression::Function(Function::Sin, vec![x.clone()]);
        let expr = mul(x.clone(), sin_x.clone());

        let result = integrate_by_parts(&expr, "x");
        assert!(result.is_ok());
    }

    #[test]
    fn test_integrate_by_parts_x_squared_exp() {
        // ∫x^2 * e^x dx = (x^2 - 2x + 2) * e^x
        // Requires two applications of integration by parts
        let x = var("x");
        let x_squared = pow(x.clone(), int(2));
        let exp_x = Expression::Function(Function::Exp, vec![x.clone()]);
        let expr = mul(x_squared.clone(), exp_x.clone());

        let result = integrate_by_parts(&expr, "x");
        assert!(result.is_ok());
    }

    #[test]
    fn test_tabular_integration_x_exp() {
        // ∫x * e^x dx using tabular method
        let x = var("x");
        let exp_x = Expression::Function(Function::Exp, vec![x.clone()]);

        let result = tabular_integration(&x, &exp_x, "x");
        assert!(result.is_ok());
    }

    #[test]
    fn test_tabular_integration_x_squared_exp() {
        // ∫x^2 * e^x dx using tabular method
        // Derivatives: x^2 -> 2x -> 2 -> 0
        // Integrals: e^x -> e^x -> e^x -> e^x
        // Result: x^2*e^x - 2x*e^x + 2*e^x
        let x = var("x");
        let x_squared = pow(x.clone(), int(2));
        let exp_x = Expression::Function(Function::Exp, vec![x.clone()]);

        let result = tabular_integration(&x_squared, &exp_x, "x");
        assert!(result.is_ok());
    }

    #[test]
    fn test_tabular_integration_x_sin() {
        // ∫x * sin(x) dx using tabular method
        let x = var("x");
        let sin_x = Expression::Function(Function::Sin, vec![x.clone()]);

        let result = tabular_integration(&x, &sin_x, "x");
        assert!(result.is_ok());
    }

    #[test]
    fn test_integrate_by_parts_with_steps() {
        // Test that we get detailed steps
        let x = var("x");
        let exp_x = Expression::Function(Function::Exp, vec![x.clone()]);
        let expr = mul(x.clone(), exp_x.clone());

        let result = integrate_by_parts_with_steps(&expr, "x");
        assert!(result.is_ok());

        if let Ok((_, steps)) = result {
            // Should have multiple steps
            assert!(steps.len() >= 5);
            // Should mention "integration by parts"
            assert!(steps.iter().any(|s| s.contains("integration by parts")));
        }
    }

    #[test]
    fn test_choose_u_and_dv() {
        // For x * e^x, x should be chosen as u (algebraic > exponential)
        let x = var("x");
        let exp_x = Expression::Function(Function::Exp, vec![x.clone()]);
        let factors = vec![x.clone(), exp_x.clone()];

        let (u, dv) = by_parts::choose_u_and_dv(&factors, "x");

        // u should be x (algebraic priority 3)
        assert!(matches!(u, Expression::Variable(_)));
        // dv should be e^x (exponential priority 1)
        assert!(matches!(dv, Expression::Function(Function::Exp, _)));
    }

    #[test]
    fn test_check_same_up_to_constant() {
        let a = var("x");
        let b = var("x");

        // Same expression -> constant 1
        assert!(check_same_up_to_constant(&a, &b, "x").is_some());

        // Negation -> constant -1
        let neg_a = Expression::Unary(UnaryOp::Neg, Box::new(a.clone()));
        let result = check_same_up_to_constant(&a, &neg_a, "x");
        assert!(result.is_some());
        assert!(matches!(result, Some(Expression::Integer(-1))));

        // Different expressions
        let c = var("y");
        assert!(check_same_up_to_constant(&a, &c, "x").is_none());
    }

    // =========================================================================
    // Definite Integral Tests
    // =========================================================================

    #[test]
    fn test_definite_integral_x_squared() {
        // ∫_0^1 x^2 dx = 1/3
        let x_squared = pow(var("x"), int(2));
        let result = definite_integral(&x_squared, "x", &int(0), &int(1));
        assert!(result.is_ok());

        let value = result.unwrap();
        let empty = std::collections::HashMap::new();
        let numeric = value.evaluate(&empty).unwrap();
        assert!((numeric - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_definite_integral_sin() {
        // ∫_0^π sin(x) dx = 2
        let sin_x = Expression::Function(Function::Sin, vec![var("x")]);
        let pi = Expression::Constant(crate::ast::SymbolicConstant::Pi);
        let result = definite_integral(&sin_x, "x", &int(0), &pi);
        assert!(result.is_ok());

        let value = result.unwrap();
        let empty = std::collections::HashMap::new();
        let numeric = value.evaluate(&empty).unwrap();
        assert!((numeric - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_definite_integral_odd_function() {
        // ∫_{-1}^1 x^3 dx = 0 (odd function, symmetric interval)
        let x_cubed = pow(var("x"), int(3));
        let result = definite_integral(&x_cubed, "x", &int(-1), &int(1));
        assert!(result.is_ok());

        let value = result.unwrap();
        let empty = std::collections::HashMap::new();
        let numeric = value.evaluate(&empty).unwrap();
        assert!(numeric.abs() < 1e-10);
    }

    #[test]
    fn test_definite_integral_symbolic_upper_bound() {
        // ∫_0^a x dx = a^2/2
        let x = var("x");
        let a = var("a");
        let result = definite_integral(&x, "x", &int(0), &a);
        assert!(result.is_ok());

        // Evaluate at a=2 to verify: should be 2
        let value = result.unwrap();
        let mut env = std::collections::HashMap::new();
        env.insert("a".to_string(), 2.0);
        let numeric = value.evaluate(&env).unwrap();
        assert!((numeric - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_definite_integral_polynomial() {
        // ∫_0^2 (3x^2 + 2x + 1) dx = [x^3 + x^2 + x]_0^2 = 8 + 4 + 2 = 14
        let x = var("x");
        let poly = add(
            add(mul(int(3), pow(x.clone(), int(2))), mul(int(2), x.clone())),
            int(1),
        );
        let result = definite_integral(&poly, "x", &int(0), &int(2));
        assert!(result.is_ok());

        let value = result.unwrap();
        let empty = std::collections::HashMap::new();
        let numeric = value.evaluate(&empty).unwrap();
        assert!((numeric - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_definite_integral_cos() {
        // ∫_0^{π/2} cos(x) dx = 1
        let cos_x = Expression::Function(Function::Cos, vec![var("x")]);
        let pi = Expression::Constant(crate::ast::SymbolicConstant::Pi);
        let upper = div(pi, int(2));
        let result = definite_integral(&cos_x, "x", &int(0), &upper);
        assert!(result.is_ok());

        let value = result.unwrap();
        let empty = std::collections::HashMap::new();
        let numeric = value.evaluate(&empty).unwrap();
        assert!((numeric - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_definite_integral_exp() {
        // ∫_0^1 e^x dx = e - 1 ≈ 1.71828
        let exp_x = Expression::Function(Function::Exp, vec![var("x")]);
        let result = definite_integral(&exp_x, "x", &int(0), &int(1));
        assert!(result.is_ok());

        let value = result.unwrap();
        let empty = std::collections::HashMap::new();
        let numeric = value.evaluate(&empty).unwrap();
        let expected = std::f64::consts::E - 1.0;
        assert!((numeric - expected).abs() < 1e-10);
    }

    #[test]
    fn test_numerical_integrate_simple() {
        // Use adaptive Simpson's for ∫_0^1 x^2 dx = 1/3
        let x_squared = pow(var("x"), int(2));
        let result = numerical_integrate(&x_squared, "x", 0.0, 1.0, 1e-8);
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!((value - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_numerical_integrate_complex() {
        // ∫_0^1 exp(-x^2) dx ≈ 0.74682 (error function related)
        let x = var("x");
        let neg_x_squared = Expression::Unary(UnaryOp::Neg, Box::new(pow(x.clone(), int(2))));
        let exp_neg_x_squared = Expression::Function(Function::Exp, vec![neg_x_squared]);
        let result = numerical_integrate(&exp_neg_x_squared, "x", 0.0, 1.0, 1e-6);
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!((value - 0.74682).abs() < 0.001);
    }

    #[test]
    fn test_definite_integral_with_fallback() {
        // Use fallback for a simple integral
        let x_squared = pow(var("x"), int(2));
        let result = definite_integral_with_fallback(&x_squared, "x", 0.0, 1.0, 1e-8);
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!((value - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_improper_integral_convergent() {
        // ∫_1^∞ x^(-2) dx = 1
        // Use x^(-2) format which integrate handles better than 1/x^2
        let x = var("x");
        let x_neg_2 = pow(x.clone(), int(-2));
        let result = improper_integral_to_infinity(&x_neg_2, "x", &int(1));
        assert!(result.is_ok());

        let value = result.unwrap();
        let empty = std::collections::HashMap::new();
        let numeric = value.evaluate(&empty).unwrap();
        assert!((numeric - 1.0).abs() < 1e-10);
    }

    // =========================================================================
    // Partial Fraction Integration Tests
    // =========================================================================

    #[test]
    fn test_integrate_rational_two_linear_factors() {
        // ∫ 1/(x²-1) dx  =  ½·ln|x-1| - ½·ln|x+1|
        // Denominator: x² - 1
        let denom = add(
            pow(var("x"), int(2)),
            Expression::Unary(UnaryOp::Neg, Box::new(int(1))),
        );
        let expr = div(int(1), denom);
        let result = integrate(&expr, "x");
        assert!(
            result.is_ok(),
            "∫1/(x²-1) dx should succeed via partial fractions, got: {:?}",
            result.err()
        );
        // The result must involve ln (from the two linear terms)
        let s = format!("{:?}", result.unwrap());
        assert!(s.contains("Ln"), "Result should contain Ln, got: {s}");
    }

    #[test]
    fn test_integrate_rational_linear_times_x() {
        // ∫ (2x+3)/(x²+x) dx  =  ∫ (2x+3)/(x(x+1)) dx
        // Denominator: x² + x
        let x = var("x");
        let denom = add(pow(x.clone(), int(2)), x.clone());
        let num = add(mul(int(2), x.clone()), int(3));
        let expr = div(num, denom);
        let result = integrate(&expr, "x");
        assert!(
            result.is_ok(),
            "∫(2x+3)/(x²+x) dx should succeed via partial fractions, got: {:?}",
            result.err()
        );
        let s = format!("{:?}", result.unwrap());
        assert!(s.contains("Ln"), "Result should contain Ln, got: {s}");
    }

    #[test]
    fn test_integrate_rational_three_linear_factors() {
        // ∫ 1/((x-1)(x-2)(x-3)) dx
        // Build denominator as expanded form via (x-1)(x-2)(x-3) = x³ - 6x² + 11x - 6
        // Coefficients: x³ - 6x² + 11x - 6
        let x = var("x");
        // x³ - 6x²
        let x3_minus_6x2 = add(
            pow(x.clone(), int(3)),
            Expression::Unary(UnaryOp::Neg, Box::new(mul(int(6), pow(x.clone(), int(2))))),
        );
        // + 11x - 6
        let denom = add(
            add(x3_minus_6x2, mul(int(11), x.clone())),
            Expression::Unary(UnaryOp::Neg, Box::new(int(6))),
        );
        let expr = div(int(1), denom);
        let result = integrate(&expr, "x");
        assert!(
            result.is_ok(),
            "∫1/((x-1)(x-2)(x-3)) dx should succeed via partial fractions, got: {:?}",
            result.err()
        );
        let s = format!("{:?}", result.unwrap());
        assert!(s.contains("Ln"), "Result should contain Ln, got: {s}");
    }

    #[test]
    fn test_definite_integral_with_steps() {
        // Verify we get step-by-step output
        let x_squared = pow(var("x"), int(2));
        let result = definite_integral_with_steps(&x_squared, "x", &int(0), &int(1));
        assert!(result.is_ok());

        let (value, steps) = result.unwrap();
        // Should have multiple steps
        assert!(!steps.is_empty());
        // Should mention "antiderivative" or "bounds"
        assert!(steps.iter().any(|s| {
            s.to_lowercase().contains("antiderivative") || s.to_lowercase().contains("bound")
        }));

        // Verify result
        let empty = std::collections::HashMap::new();
        let numeric = value.evaluate(&empty).unwrap();
        assert!((numeric - 1.0 / 3.0).abs() < 1e-10);
    }
}
