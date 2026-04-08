//! Precision-aware expression evaluation.
//!
//! Provides configurable precision modes, rounding strategies,
//! and tracked arithmetic for numerical computation.

mod context;
mod helpers;
mod types;

pub use context::EvalContext;
pub use types::{EvalError, PrecisionMode, RoundingMode, Value};

#[cfg(test)]
mod tests {
    use super::helpers::*;
    use super::*;
    use crate::ast::{BinaryOp, Expression, Function, SymbolicConstant, Variable};

    fn div(a: Expression, b: Expression) -> Expression {
        Expression::Binary(BinaryOp::Div, Box::new(a), Box::new(b))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn sqrt(x: Expression) -> Expression {
        Expression::Function(Function::Sqrt, vec![x])
    }

    #[test]
    fn test_fixed_decimal_precision() {
        // 1/3 at 6 decimal places = 0.333333
        let ctx = EvalContext::fixed_decimal(6);
        let expr = div(int(1), int(3));
        let result = ctx.evaluate(&expr).unwrap();
        let f = result.as_f64();
        assert!((f - 0.333333).abs() < 1e-10);
    }

    #[test]
    fn test_significant_figures() {
        // 1/3 at 3 significant figures = 0.333
        let ctx = EvalContext::significant_figures(3);
        let expr = div(int(1), int(3));
        let result = ctx.evaluate(&expr).unwrap();
        let f = result.as_f64();
        assert!((f - 0.333).abs() < 1e-10);
    }

    #[test]
    fn test_arbitrary_precision_rational() {
        // 1/3 as exact rational
        let ctx = EvalContext::arbitrary();
        let expr = div(int(1), int(3));
        let result = ctx.evaluate(&expr).unwrap();
        match result {
            Value::Rational(r) => {
                assert_eq!(*r.numer(), 1);
                assert_eq!(*r.denom(), 3);
            }
            _ => panic!("Expected Rational"),
        }
    }

    #[test]
    fn test_sqrt_2_precision() {
        // sqrt(2) at various precisions
        let expr = sqrt(int(2));

        let ctx6 = EvalContext::fixed_decimal(6);
        let r6 = ctx6.evaluate(&expr).unwrap().as_f64();
        assert!((r6 - 1.414214).abs() < 1e-6);

        let ctx3 = EvalContext::significant_figures(3);
        let r3 = ctx3.evaluate(&expr).unwrap().as_f64();
        assert!((r3 - 1.41).abs() < 0.01);
    }

    #[test]
    fn test_complex_from_sqrt_negative() {
        // sqrt(-1) = i
        let ctx = EvalContext::full_precision().with_complex(true);
        let expr = sqrt(int(-1));
        let result = ctx.evaluate(&expr).unwrap();
        match result {
            Value::Complex(re, im) => {
                assert!(re.abs() < 1e-10);
                assert!((im - 1.0).abs() < 1e-10);
            }
            _ => panic!("Expected Complex"),
        }
    }

    #[test]
    fn test_complex_not_allowed() {
        // sqrt(-1) should error when complex not allowed
        let ctx = EvalContext::full_precision().with_complex(false);
        let expr = sqrt(int(-1));
        let result = ctx.evaluate(&expr);
        assert!(result.is_err());
    }

    #[test]
    fn test_overflow_handling() {
        // Very large computation should not panic
        let ctx = EvalContext::full_precision();
        let expr = Expression::Power(Box::new(int(10)), Box::new(int(1000)));
        let result = ctx.evaluate(&expr).unwrap();
        // Should be infinity
        match result {
            Value::PositiveInfinity => {}
            Value::Float(f) if f.is_infinite() => {}
            _ => panic!("Expected infinity"),
        }
    }

    #[test]
    fn test_division_by_zero() {
        let ctx = EvalContext::full_precision();
        let expr = div(int(1), int(0));
        let result = ctx.evaluate(&expr);
        assert!(matches!(result, Err(EvalError::DivisionByZero)));
    }

    #[test]
    fn test_variable_evaluation() {
        let mut ctx = EvalContext::full_precision();
        ctx.set_f64("x", 5.0);
        let expr = var("x");
        let result = ctx.evaluate(&expr).unwrap();
        assert!((result.as_f64() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_undefined_variable() {
        let ctx = EvalContext::full_precision();
        let expr = var("undefined");
        let result = ctx.evaluate(&expr);
        assert!(matches!(result, Err(EvalError::UndefinedVariable(_))));
    }

    #[test]
    fn test_rounding_modes() {
        // Test different rounding modes for 2.5
        let value = 2.5;

        assert_eq!(apply_rounding(value, RoundingMode::HalfUp), 3.0);
        assert_eq!(apply_rounding(value, RoundingMode::HalfEven), 2.0); // Banker's rounding
        assert_eq!(apply_rounding(value, RoundingMode::Truncate), 2.0);
        assert_eq!(apply_rounding(value, RoundingMode::Ceiling), 3.0);
        assert_eq!(apply_rounding(value, RoundingMode::Floor), 2.0);

        // Test 3.5 with banker's rounding (should round to 4)
        assert_eq!(apply_rounding(3.5, RoundingMode::HalfEven), 4.0);
    }

    #[test]
    fn test_complex_arithmetic() {
        let ctx = EvalContext::full_precision();

        // i * i = -1
        let i = Expression::Constant(SymbolicConstant::I);
        let i_squared = Expression::Binary(BinaryOp::Mul, Box::new(i.clone()), Box::new(i));
        let result = ctx.evaluate(&i_squared).unwrap();
        match result {
            Value::Complex(re, im) => {
                assert!((re - (-1.0)).abs() < 1e-10);
                assert!(im.abs() < 1e-10);
            }
            _ => panic!("Expected Complex"),
        }
    }

    #[test]
    fn test_infinity_handling() {
        let ctx = EvalContext::full_precision();

        // 1/0 should be division by zero error (not infinity)
        let expr = div(int(1), int(0));
        assert!(ctx.evaluate(&expr).is_err());

        // But very large numbers should produce infinity
        let expr = Expression::Power(Box::new(int(10)), Box::new(int(500)));
        let result = ctx.evaluate(&expr).unwrap();
        let is_positive_inf = matches!(result, Value::PositiveInfinity)
            || matches!(result, Value::Float(f) if f.is_infinite() && f > 0.0);
        assert!(is_positive_inf);
    }
}
