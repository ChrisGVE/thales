//! Tests for symbolic differentiation and partial derivatives.
//!
//! These tests verify:
//! - Basic differentiation rules (constant, power, product, quotient)
//! - Chain rule application
//! - Trigonometric derivatives
//! - Exponential and logarithmic derivatives
//! - Partial derivative computation
//! - Comparison with numerical derivatives

use mathsolver_core::ast::{BinaryOp, Equation, Expression, Function, UnaryOp, Variable};
use mathsolver_core::solver::{compute_all_partial_derivatives, compute_partial_derivative};
use std::collections::HashMap;

// Helper function to create a variable expression
fn var(name: &str) -> Expression {
    Expression::Variable(Variable::new(name))
}

// Helper function to create an integer expression
fn int(n: i64) -> Expression {
    Expression::Integer(n)
}

// Helper function to create a float expression
fn float(x: f64) -> Expression {
    Expression::Float(x)
}

// Helper function to compute numerical derivative using finite differences
fn numerical_derivative(
    expr: &Expression,
    var_name: &str,
    values: &HashMap<String, f64>,
    h: f64,
) -> f64 {
    let mut values_plus = values.clone();
    let mut values_minus = values.clone();

    let x = values.get(var_name).unwrap();
    values_plus.insert(var_name.to_string(), x + h);
    values_minus.insert(var_name.to_string(), x - h);

    let f_plus = expr.evaluate(&values_plus).unwrap();
    let f_minus = expr.evaluate(&values_minus).unwrap();

    (f_plus - f_minus) / (2.0 * h)
}

#[cfg(test)]
mod basic_derivatives {
    use super::*;

    #[test]
    fn test_constant_derivative() {
        // d/dx[5] = 0
        let expr = int(5);
        let derivative = expr.differentiate("x");

        assert_eq!(derivative, int(0));
    }

    #[test]
    fn test_variable_derivative() {
        // d/dx[x] = 1
        let expr = var("x");
        let derivative = expr.differentiate("x");

        assert_eq!(derivative, int(1));
    }

    #[test]
    fn test_different_variable_derivative() {
        // d/dx[y] = 0
        let expr = var("y");
        let derivative = expr.differentiate("x");

        assert_eq!(derivative, int(0));
    }

    #[test]
    fn test_power_rule_simple() {
        // d/dx[x^2] = 2*x
        let expr = Expression::Power(Box::new(var("x")), Box::new(int(2)));
        let derivative = expr.differentiate("x");

        // Should give: 2 * x^1 * 1 = 2 * x
        let mut values = HashMap::new();
        values.insert("x".to_string(), 3.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, 6.0); // 2 * 3 = 6
    }

    #[test]
    fn test_power_rule_general() {
        // d/dx[x^3] = 3*x^2
        let expr = Expression::Power(Box::new(var("x")), Box::new(int(3)));
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 2.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, 12.0); // 3 * 2^2 = 12
    }

    #[test]
    fn test_power_rule_fractional() {
        // d/dx[x^(1/2)] = (1/2)*x^(-1/2) = 1/(2*sqrt(x))
        let half = Expression::Binary(
            BinaryOp::Div,
            Box::new(int(1)),
            Box::new(int(2))
        );
        let expr = Expression::Power(Box::new(var("x")), Box::new(half));
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 4.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        // (1/2) * 4^(-1/2) = 0.5 * 0.5 = 0.25
        assert!((result - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_sum_rule() {
        // d/dx[x + x^2] = 1 + 2*x
        let expr = Expression::Binary(
            BinaryOp::Add,
            Box::new(var("x")),
            Box::new(Expression::Power(Box::new(var("x")), Box::new(int(2))))
        );
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 3.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, 7.0); // 1 + 2*3 = 7
    }

    #[test]
    fn test_difference_rule() {
        // d/dx[x^3 - x] = 3*x^2 - 1
        let expr = Expression::Binary(
            BinaryOp::Sub,
            Box::new(Expression::Power(Box::new(var("x")), Box::new(int(3)))),
            Box::new(var("x"))
        );
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 2.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, 11.0); // 3*4 - 1 = 11
    }
}

#[cfg(test)]
mod product_quotient_rules {
    use super::*;

    #[test]
    fn test_product_rule() {
        // d/dx[x * x^2] = x * 2*x + x^2 * 1 = 3*x^2
        let expr = Expression::Binary(
            BinaryOp::Mul,
            Box::new(var("x")),
            Box::new(Expression::Power(Box::new(var("x")), Box::new(int(2))))
        );
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 2.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, 12.0); // 3 * 4 = 12
    }

    #[test]
    fn test_product_rule_with_constants() {
        // d/dx[5 * x^2] = 5 * 2*x = 10*x
        let expr = Expression::Binary(
            BinaryOp::Mul,
            Box::new(int(5)),
            Box::new(Expression::Power(Box::new(var("x")), Box::new(int(2))))
        );
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 3.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, 30.0); // 10 * 3 = 30
    }

    #[test]
    fn test_quotient_rule() {
        // d/dx[x^2 / x] = (x * 2*x - x^2 * 1) / x^2 = (2*x^2 - x^2) / x^2 = x^2 / x^2 = 1
        let expr = Expression::Binary(
            BinaryOp::Div,
            Box::new(Expression::Power(Box::new(var("x")), Box::new(int(2)))),
            Box::new(var("x"))
        );
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 5.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_quotient_rule_complex() {
        // d/dx[x / (x + 1)] = ((x+1)*1 - x*1) / (x+1)^2 = 1 / (x+1)^2
        let numerator = var("x");
        let denominator = Expression::Binary(
            BinaryOp::Add,
            Box::new(var("x")),
            Box::new(int(1))
        );
        let expr = Expression::Binary(
            BinaryOp::Div,
            Box::new(numerator),
            Box::new(denominator)
        );
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 3.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, 1.0 / 16.0); // 1 / (3+1)^2 = 1/16
    }
}

#[cfg(test)]
mod chain_rule {
    use super::*;

    #[test]
    fn test_chain_rule_power() {
        // d/dx[(2*x)^2] = 2 * (2*x)^1 * 2 = 8*x
        let inner = Expression::Binary(
            BinaryOp::Mul,
            Box::new(int(2)),
            Box::new(var("x"))
        );
        let expr = Expression::Power(Box::new(inner), Box::new(int(2)));
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 3.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, 24.0); // 8 * 3 = 24
    }

    #[test]
    fn test_chain_rule_nested_power() {
        // d/dx[(x^2)^3] = 3*(x^2)^2 * 2*x = 6*x^5
        let inner = Expression::Power(Box::new(var("x")), Box::new(int(2)));
        let expr = Expression::Power(Box::new(inner), Box::new(int(3)));
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 2.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, 192.0); // 6 * 2^5 = 6 * 32 = 192
    }
}

#[cfg(test)]
mod trig_derivatives {
    use super::*;

    #[test]
    fn test_sin_derivative() {
        // d/dx[sin(x)] = cos(x)
        let expr = Expression::Function(Function::Sin, vec![var("x")]);
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 0.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, 1.0); // cos(0) = 1
    }

    #[test]
    fn test_cos_derivative() {
        // d/dx[cos(x)] = -sin(x)
        let expr = Expression::Function(Function::Cos, vec![var("x")]);
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 0.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, 0.0); // -sin(0) = 0
    }

    #[test]
    fn test_tan_derivative() {
        // d/dx[tan(x)] = sec^2(x) = 1/cos^2(x)
        let expr = Expression::Function(Function::Tan, vec![var("x")]);
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 0.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, 1.0); // 1/cos^2(0) = 1
    }

    #[test]
    fn test_sin_chain_rule() {
        // d/dx[sin(2*x)] = cos(2*x) * 2 = 2*cos(2*x)
        let inner = Expression::Binary(
            BinaryOp::Mul,
            Box::new(int(2)),
            Box::new(var("x"))
        );
        let expr = Expression::Function(Function::Sin, vec![inner]);
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 0.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, 2.0); // 2 * cos(0) = 2
    }

    #[test]
    fn test_asin_derivative() {
        // d/dx[asin(x)] = 1/sqrt(1 - x^2)
        let expr = Expression::Function(Function::Asin, vec![var("x")]);
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 0.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, 1.0); // 1/sqrt(1) = 1
    }

    #[test]
    fn test_atan_derivative() {
        // d/dx[atan(x)] = 1/(1 + x^2)
        let expr = Expression::Function(Function::Atan, vec![var("x")]);
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 0.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, 1.0); // 1/(1 + 0) = 1
    }
}

#[cfg(test)]
mod exp_log_derivatives {
    use super::*;

    #[test]
    fn test_exp_derivative() {
        // d/dx[exp(x)] = exp(x)
        let expr = Expression::Function(Function::Exp, vec![var("x")]);
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 0.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, 1.0); // exp(0) = 1
    }

    #[test]
    fn test_exp_chain_rule() {
        // d/dx[exp(2*x)] = exp(2*x) * 2
        let inner = Expression::Binary(
            BinaryOp::Mul,
            Box::new(int(2)),
            Box::new(var("x"))
        );
        let expr = Expression::Function(Function::Exp, vec![inner]);
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 0.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, 2.0); // exp(0) * 2 = 2
    }

    #[test]
    fn test_ln_derivative() {
        // d/dx[ln(x)] = 1/x
        let expr = Expression::Function(Function::Ln, vec![var("x")]);
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 2.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, 0.5); // 1/2 = 0.5
    }

    #[test]
    fn test_ln_chain_rule() {
        // d/dx[ln(x^2)] = (1/x^2) * 2*x = 2/x
        let inner = Expression::Power(Box::new(var("x")), Box::new(int(2)));
        let expr = Expression::Function(Function::Ln, vec![inner]);
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 3.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert!((result - 2.0/3.0).abs() < 1e-10);
    }

    #[test]
    fn test_log10_derivative() {
        // d/dx[log10(x)] = 1/(x * ln(10))
        let expr = Expression::Function(Function::Log10, vec![var("x")]);
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 10.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        let expected = 1.0 / (10.0 * 10.0_f64.ln());
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_base() {
        // d/dx[2^x] = 2^x * ln(2)
        let expr = Expression::Power(Box::new(int(2)), Box::new(var("x")));
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 3.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        let expected = 8.0 * 2.0_f64.ln(); // 2^3 * ln(2)
        assert!((result - expected).abs() < 1e-10);
    }
}

#[cfg(test)]
mod sqrt_derivatives {
    use super::*;

    #[test]
    fn test_sqrt_derivative() {
        // d/dx[sqrt(x)] = 1/(2*sqrt(x))
        let expr = Expression::Function(Function::Sqrt, vec![var("x")]);
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 4.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, 0.25); // 1/(2*2) = 0.25
    }

    #[test]
    fn test_cbrt_derivative() {
        // d/dx[cbrt(x)] = 1/(3*x^(2/3))
        let expr = Expression::Function(Function::Cbrt, vec![var("x")]);
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 8.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        // 1/(3 * 8^(2/3)) = 1/(3 * 4) = 1/12
        assert!((result - 1.0/12.0).abs() < 1e-10);
    }
}

#[cfg(test)]
mod partial_derivatives {
    use super::*;

    #[test]
    fn test_partial_derivative_simple() {
        // Equation: A = π * r^2
        // ∂A/∂r = 2*π*r
        //
        // Note: We can test the derivative directly without using the solver
        let r = var("r");
        let pi = float(std::f64::consts::PI);
        let r_squared = Expression::Power(Box::new(r.clone()), Box::new(int(2)));
        let area = Expression::Binary(BinaryOp::Mul, Box::new(pi), Box::new(r_squared));

        // Compute derivative directly
        let derivative_expr = area.differentiate("r");

        let mut values = HashMap::new();
        values.insert("r".to_string(), 5.0);

        let derivative = derivative_expr.simplify().evaluate(&values).unwrap();
        let expected = 2.0 * std::f64::consts::PI * 5.0;
        assert!((derivative - expected).abs() < 1e-10);
    }

    #[test]
    fn test_partial_derivative_multivariable() {
        // Equation: V = l * w * h
        // ∂V/∂l = w * h
        let l = var("l");
        let w = var("w");
        let h = var("h");
        let lw = Expression::Binary(BinaryOp::Mul, Box::new(l), Box::new(w));
        let volume = Expression::Binary(BinaryOp::Mul, Box::new(lw), Box::new(h));
        let equation = Equation::new("box_volume", var("V"), volume);

        let mut values = HashMap::new();
        values.insert("l".to_string(), 2.0);
        values.insert("w".to_string(), 3.0);
        values.insert("h".to_string(), 4.0);

        let dv_dl = compute_partial_derivative(&equation, "V", "l", &values).unwrap();
        let dv_dw = compute_partial_derivative(&equation, "V", "w", &values).unwrap();
        let dv_dh = compute_partial_derivative(&equation, "V", "h", &values).unwrap();

        assert_eq!(dv_dl, 12.0); // w * h = 3 * 4
        assert_eq!(dv_dw, 8.0);  // l * h = 2 * 4
        assert_eq!(dv_dh, 6.0);  // l * w = 2 * 3
    }

    #[test]
    fn test_compute_all_partial_derivatives() {
        // Equation: V = l * w * h
        let l = var("l");
        let w = var("w");
        let h = var("h");
        let lw = Expression::Binary(BinaryOp::Mul, Box::new(l), Box::new(w));
        let volume = Expression::Binary(BinaryOp::Mul, Box::new(lw), Box::new(h));
        let equation = Equation::new("box_volume", var("V"), volume);

        let mut values = HashMap::new();
        values.insert("l".to_string(), 2.0);
        values.insert("w".to_string(), 3.0);
        values.insert("h".to_string(), 4.0);

        let input_vars = vec!["l".to_string(), "w".to_string(), "h".to_string()];
        let derivatives = compute_all_partial_derivatives(&equation, "V", &input_vars, &values).unwrap();

        assert_eq!(derivatives.get("l").unwrap(), &12.0);
        assert_eq!(derivatives.get("w").unwrap(), &8.0);
        assert_eq!(derivatives.get("h").unwrap(), &6.0);
    }

    #[test]
    fn test_partial_derivative_pythagorean() {
        // Equation: c^2 = a^2 + b^2
        // Rearranged: c = sqrt(a^2 + b^2)
        // ∂c/∂a = a / sqrt(a^2 + b^2) = a / c
        //
        // Note: We test the derivative of c = sqrt(a^2 + b^2) directly
        let a = var("a");
        let b = var("b");
        let a_squared = Expression::Power(Box::new(a.clone()), Box::new(int(2)));
        let b_squared = Expression::Power(Box::new(b.clone()), Box::new(int(2)));
        let sum = Expression::Binary(BinaryOp::Add, Box::new(a_squared), Box::new(b_squared));
        let c_expr = Expression::Function(Function::Sqrt, vec![sum]);

        let mut values = HashMap::new();
        values.insert("a".to_string(), 3.0);
        values.insert("b".to_string(), 4.0);

        let dc_da_expr = c_expr.differentiate("a");
        let dc_db_expr = c_expr.differentiate("b");

        let dc_da = dc_da_expr.simplify().evaluate(&values).unwrap();
        let dc_db = dc_db_expr.simplify().evaluate(&values).unwrap();

        // ∂c/∂a = a/c = 3/5 = 0.6
        // ∂c/∂b = b/c = 4/5 = 0.8
        assert!((dc_da - 0.6).abs() < 1e-10);
        assert!((dc_db - 0.8).abs() < 1e-10);
    }
}

#[cfg(test)]
mod numerical_comparison {
    use super::*;

    #[test]
    fn test_symbolic_vs_numerical_polynomial() {
        // f(x) = x^3 + 2*x^2 - 5*x + 1
        // f'(x) = 3*x^2 + 4*x - 5
        let x = var("x");
        let x_cubed = Expression::Power(Box::new(x.clone()), Box::new(int(3)));
        let two_x_squared = Expression::Binary(
            BinaryOp::Mul,
            Box::new(int(2)),
            Box::new(Expression::Power(Box::new(x.clone()), Box::new(int(2))))
        );
        let five_x = Expression::Binary(BinaryOp::Mul, Box::new(int(5)), Box::new(x.clone()));

        let term1 = Expression::Binary(BinaryOp::Add, Box::new(x_cubed), Box::new(two_x_squared));
        let term2 = Expression::Binary(BinaryOp::Sub, Box::new(term1), Box::new(five_x));
        let expr = Expression::Binary(BinaryOp::Add, Box::new(term2), Box::new(int(1)));

        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 2.5);

        let symbolic = derivative.simplify().evaluate(&values).unwrap();
        let numerical = numerical_derivative(&expr, "x", &values, 1e-8);

        // Expected: 3*2.5^2 + 4*2.5 - 5 = 18.75 + 10 - 5 = 23.75
        assert!((symbolic - 23.75).abs() < 1e-10);
        assert!((symbolic - numerical).abs() < 1e-6);
    }

    #[test]
    fn test_symbolic_vs_numerical_trig() {
        // f(x) = sin(x) * cos(x)
        // f'(x) = cos(x) * cos(x) + sin(x) * (-sin(x)) = cos^2(x) - sin^2(x)
        let sin_x = Expression::Function(Function::Sin, vec![var("x")]);
        let cos_x = Expression::Function(Function::Cos, vec![var("x")]);
        let expr = Expression::Binary(BinaryOp::Mul, Box::new(sin_x), Box::new(cos_x));

        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 1.0);

        let symbolic = derivative.simplify().evaluate(&values).unwrap();
        let numerical = numerical_derivative(&expr, "x", &values, 1e-8);

        assert!((symbolic - numerical).abs() < 1e-6);
    }

    #[test]
    fn test_symbolic_vs_numerical_exp() {
        // f(x) = exp(x^2)
        // f'(x) = exp(x^2) * 2*x
        let x_squared = Expression::Power(Box::new(var("x")), Box::new(int(2)));
        let expr = Expression::Function(Function::Exp, vec![x_squared]);

        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 1.5);

        let symbolic = derivative.simplify().evaluate(&values).unwrap();
        let numerical = numerical_derivative(&expr, "x", &values, 1e-8);

        assert!((symbolic - numerical).abs() < 1e-6);
    }

    #[test]
    fn test_symbolic_vs_numerical_quotient() {
        // f(x) = (x^2 + 1) / (x - 1)
        // f'(x) = ((x-1)*2*x - (x^2+1)*1) / (x-1)^2
        let numerator = Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Power(Box::new(var("x")), Box::new(int(2)))),
            Box::new(int(1))
        );
        let denominator = Expression::Binary(
            BinaryOp::Sub,
            Box::new(var("x")),
            Box::new(int(1))
        );
        let expr = Expression::Binary(
            BinaryOp::Div,
            Box::new(numerator),
            Box::new(denominator)
        );

        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 3.0);

        let symbolic = derivative.simplify().evaluate(&values).unwrap();
        let numerical = numerical_derivative(&expr, "x", &values, 1e-8);

        assert!((symbolic - numerical).abs() < 1e-6);
    }
}

#[cfg(test)]
mod edge_cases {
    use super::*;

    #[test]
    fn test_derivative_of_constant_times_variable() {
        // d/dx[5*x] = 5
        let expr = Expression::Binary(
            BinaryOp::Mul,
            Box::new(int(5)),
            Box::new(var("x"))
        );
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 10.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_derivative_of_negation() {
        // d/dx[-x^2] = -2*x
        let x_squared = Expression::Power(Box::new(var("x")), Box::new(int(2)));
        let expr = Expression::Unary(UnaryOp::Neg, Box::new(x_squared));
        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 3.0);

        let result = derivative.simplify().evaluate(&values).unwrap();
        assert_eq!(result, -6.0); // -2 * 3 = -6
    }

    #[test]
    fn test_derivative_with_multiple_variables() {
        // f(x, y) = x^2 + y^2
        // ∂f/∂x = 2*x
        // ∂f/∂y = 2*y
        let x_squared = Expression::Power(Box::new(var("x")), Box::new(int(2)));
        let y_squared = Expression::Power(Box::new(var("y")), Box::new(int(2)));
        let expr = Expression::Binary(BinaryOp::Add, Box::new(x_squared), Box::new(y_squared));

        let dx = expr.differentiate("x");
        let dy = expr.differentiate("y");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 3.0);
        values.insert("y".to_string(), 4.0);

        let result_dx = dx.simplify().evaluate(&values).unwrap();
        let result_dy = dy.simplify().evaluate(&values).unwrap();

        assert_eq!(result_dx, 6.0); // 2 * 3 = 6
        assert_eq!(result_dy, 8.0); // 2 * 4 = 8
    }

    #[test]
    fn test_derivative_complex_expression() {
        // f(x) = (x^2 + 1) * exp(x) * sin(x)
        let x_squared_plus_1 = Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Power(Box::new(var("x")), Box::new(int(2)))),
            Box::new(int(1))
        );
        let exp_x = Expression::Function(Function::Exp, vec![var("x")]);
        let sin_x = Expression::Function(Function::Sin, vec![var("x")]);

        let term1 = Expression::Binary(BinaryOp::Mul, Box::new(x_squared_plus_1), Box::new(exp_x));
        let expr = Expression::Binary(BinaryOp::Mul, Box::new(term1), Box::new(sin_x));

        let derivative = expr.differentiate("x");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 0.5);

        let symbolic = derivative.simplify().evaluate(&values).unwrap();
        let numerical = numerical_derivative(&expr, "x", &values, 1e-8);

        assert!((symbolic - numerical).abs() < 1e-5);
    }
}
