//! Property-based tests for thales using proptest.
//!
//! These tests verify mathematical properties that should hold for all inputs,
//! including:
//! - Linearity of derivatives
//! - Product and quotient rules
//! - Round-trip conversions (parse -> format -> parse)
//! - Numerical accuracy of symbolic operations
//! - Known mathematical values
//! - Edge case handling

use proptest::prelude::*;
use std::collections::HashMap;
use thales::ast::{BinaryOp, Expression, UnaryOp, Variable};
use thales::parser::{parse_equation, parse_expression};
use thales::transforms::{Cartesian2D, Cartesian3D};

// ============================================================================
// Mathematical Invariants - Derivatives
// ============================================================================

proptest! {
    /// Test: Derivative is linear - d/dx(a*f + b*g) should behave linearly
    /// For polynomials f = x, g = x, we have d/dx(a*x + b*x) = a + b
    #[test]
    fn derivative_is_linear(a in -100i32..100i32, b in -100i32..100i32) {
        // Create expression: a*x + b*x = (a+b)*x
        let expr = Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Integer(a as i64)),
                Box::new(Expression::Variable(Variable::new("x"))),
            )),
            Box::new(Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Integer(b as i64)),
                Box::new(Expression::Variable(Variable::new("x"))),
            )),
        );

        // Derivative should be a + b
        let derivative = expr.differentiate("x").simplify();

        // Evaluate at any point (derivative is constant)
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 5.0);

        let result = derivative.evaluate(&vars);
        if let Some(value) = result {
            // d/dx(a*x + b*x) = a + b
            prop_assert!((value - (a + b) as f64).abs() < 1e-10);
        }
    }

    /// Test: Product rule - d/dx(f*g) = f'*g + f*g'
    /// For f = x^2 and g = x, we have d/dx(x^3) = 3*x^2
    #[test]
    fn product_rule_holds(coef in 1i32..10i32) {
        // Create expression: x^2 * (coef*x) = coef*x^3
        let f = Expression::Power(
            Box::new(Expression::Variable(Variable::new("x"))),
            Box::new(Expression::Integer(2)),
        );
        let g = Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Integer(coef as i64)),
            Box::new(Expression::Variable(Variable::new("x"))),
        );

        let product = Expression::Binary(BinaryOp::Mul, Box::new(f), Box::new(g));

        // d/dx(coef*x^3) = 3*coef*x^2
        let derivative = product.differentiate("x").simplify();

        // Test at x = 2
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 2.0);

        if let Some(value) = derivative.evaluate(&vars) {
            let expected = 3.0 * coef as f64 * 4.0; // 3*coef*2^2
            prop_assert!((value - expected).abs() < 1e-8);
        }
    }

    /// Test: Chain rule - d/dx(f(g(x))) = f'(g(x)) * g'(x)
    /// For f = u^2 and g = 2*x, we have d/dx((2*x)^2) = 8*x
    #[test]
    fn chain_rule_holds(coef in 1i32..10i32) {
        // Create expression: (coef*x)^2 = coef^2*x^2
        let inner = Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Integer(coef as i64)),
            Box::new(Expression::Variable(Variable::new("x"))),
        );
        let expr = Expression::Power(
            Box::new(inner),
            Box::new(Expression::Integer(2)),
        );

        // d/dx((coef*x)^2) = 2*coef*x * coef = 2*coef^2*x
        let derivative = expr.differentiate("x").simplify();

        // Test at x = 3
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);

        if let Some(value) = derivative.evaluate(&vars) {
            let expected = 2.0 * (coef * coef) as f64 * 3.0;
            prop_assert!((value - expected).abs() < 1e-8);
        }
    }

    /// Test: Constant multiple rule - d/dx(c*f) = c*f'
    #[test]
    fn constant_multiple_rule(c in -100i32..100i32, exp in 1i32..5i32) {
        if c == 0 {
            return Ok(()); // Skip c=0 case
        }

        // Create expression: c*x^exp
        let expr = Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Integer(c as i64)),
            Box::new(Expression::Power(
                Box::new(Expression::Variable(Variable::new("x"))),
                Box::new(Expression::Integer(exp as i64)),
            )),
        );

        // d/dx(c*x^exp) = c*exp*x^(exp-1)
        let derivative = expr.differentiate("x").simplify();

        // Test at x = 2
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 2.0);

        if let Some(value) = derivative.evaluate(&vars) {
            let expected = c as f64 * exp as f64 * 2.0_f64.powi(exp - 1);
            prop_assert!((value - expected).abs() < 1e-8);
        }
    }
}

// ============================================================================
// Round-Trip Tests
// ============================================================================

proptest! {
    /// Test: Parse round-trip for simple polynomials
    /// parse -> to_string -> parse should yield equivalent expressions
    #[test]
    fn parse_roundtrip_linear(a in -100i32..100i32, b in -100i32..100i32) {
        if a == 0 {
            return Ok(()); // Skip degenerate case
        }

        let expr_str = format!("{}*x + {}", a, b);
        let parsed = parse_expression(&expr_str);

        if let Ok(expr) = parsed {
            // Render and re-parse
            let rendered = format!("{}", expr);
            let reparsed = parse_expression(&rendered);

            if let Ok(expr2) = reparsed {
                // Check equivalence by evaluating at test point
                let mut vars = HashMap::new();
                vars.insert("x".to_string(), 5.0);

                let val1 = expr.evaluate(&vars);
                let val2 = expr2.evaluate(&vars);

                if let (Some(v1), Some(v2)) = (val1, val2) {
                    prop_assert!((v1 - v2).abs() < 1e-10);
                }
            }
        }
    }

    /// Test: Parse round-trip for equations
    #[test]
    fn parse_roundtrip_equation(a in 1i32..100i32, b in -100i32..100i32, c in -100i32..100i32) {
        let eq_str = format!("{}*x + {} = {}", a, b, c);
        let parsed = parse_equation(&eq_str);

        if let Ok(eq) = parsed {
            // Render and re-parse (format left and right separately)
            let rendered = format!("{} = {}", eq.left, eq.right);
            let reparsed = parse_equation(&rendered);

            if let Ok(eq2) = reparsed {
                // Check equivalence by evaluating both sides
                let mut vars = HashMap::new();
                vars.insert("x".to_string(), 3.0);

                let lhs1 = eq.left.evaluate(&vars);
                let rhs1 = eq.right.evaluate(&vars);
                let lhs2 = eq2.left.evaluate(&vars);
                let rhs2 = eq2.right.evaluate(&vars);

                if let (Some(l1), Some(r1), Some(l2), Some(r2)) = (lhs1, rhs1, lhs2, rhs2) {
                    prop_assert!((l1 - l2).abs() < 1e-10);
                    prop_assert!((r1 - r2).abs() < 1e-10);
                }
            }
        }
    }
}

// ============================================================================
// Numerical Accuracy Tests
// ============================================================================

proptest! {
    /// Test: Symbolic derivative matches finite difference approximation
    /// For polynomial expressions, compare symbolic vs numerical derivatives
    #[test]
    fn derivative_matches_finite_difference(
        coef in 1i32..20i32,
        x in 1.0f64..10.0f64
    ) {
        // Create expression: coef*x^2
        let expr = Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Integer(coef as i64)),
            Box::new(Expression::Power(
                Box::new(Expression::Variable(Variable::new("x"))),
                Box::new(Expression::Integer(2)),
            )),
        );

        // Symbolic derivative: 2*coef*x
        let derivative = expr.differentiate("x").simplify();

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), x);

        let symbolic_value = derivative.evaluate(&vars);

        // Numerical derivative using central difference
        let h = 1e-6;
        let mut vars_plus = vars.clone();
        let mut vars_minus = vars.clone();
        vars_plus.insert("x".to_string(), x + h);
        vars_minus.insert("x".to_string(), x - h);

        let f_plus = expr.evaluate(&vars_plus);
        let f_minus = expr.evaluate(&vars_minus);

        if let (Some(sym), Some(fp), Some(fm)) = (symbolic_value, f_plus, f_minus) {
            let numerical = (fp - fm) / (2.0 * h);
            prop_assert!((sym - numerical).abs() < 1e-4);
        }
    }

    /// Test: Coordinate transformation preserves magnitude
    #[test]
    fn coordinate_magnitude_preserved(x in -1000.0f64..1000.0f64, y in -1000.0f64..1000.0f64) {
        let cart = Cartesian2D::new(x, y);
        let polar = cart.to_polar();

        let cart_magnitude = cart.magnitude();
        let polar_magnitude = polar.r;

        prop_assert!((cart_magnitude - polar_magnitude).abs() < 1e-10);
    }

    /// Test: 3D coordinate transformation preserves magnitude
    #[test]
    fn coordinate_3d_magnitude_preserved(
        x in -100.0f64..100.0f64,
        y in -100.0f64..100.0f64,
        z in -100.0f64..100.0f64
    ) {
        let cart = Cartesian3D::new(x, y, z);
        let spherical = cart.to_spherical();

        let cart_magnitude = cart.magnitude();
        let spherical_magnitude = spherical.r;

        prop_assert!((cart_magnitude - spherical_magnitude).abs() < 1e-10);
    }
}

// ============================================================================
// Algebraic Properties
// ============================================================================

proptest! {
    /// Test: Associativity of addition - (a + b) + c = a + (b + c)
    #[test]
    fn addition_is_associative(a in -100i32..100i32, b in -100i32..100i32, c in -100i32..100i32) {
        // (a + b) + c
        let left = Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Binary(
                BinaryOp::Add,
                Box::new(Expression::Integer(a as i64)),
                Box::new(Expression::Integer(b as i64)),
            )),
            Box::new(Expression::Integer(c as i64)),
        );

        // a + (b + c)
        let right = Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Integer(a as i64)),
            Box::new(Expression::Binary(
                BinaryOp::Add,
                Box::new(Expression::Integer(b as i64)),
                Box::new(Expression::Integer(c as i64)),
            )),
        );

        let vars = HashMap::new();
        let left_val = left.evaluate(&vars);
        let right_val = right.evaluate(&vars);

        if let (Some(lv), Some(rv)) = (left_val, right_val) {
            prop_assert!((lv - rv).abs() < 1e-10);
        }
    }

    /// Test: Commutativity of multiplication - a * b = b * a
    #[test]
    fn multiplication_is_commutative(a in -100i32..100i32, b in -100i32..100i32) {
        let left = Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Integer(a as i64)),
            Box::new(Expression::Integer(b as i64)),
        );

        let right = Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Integer(b as i64)),
            Box::new(Expression::Integer(a as i64)),
        );

        let vars = HashMap::new();
        let left_val = left.evaluate(&vars);
        let right_val = right.evaluate(&vars);

        if let (Some(lv), Some(rv)) = (left_val, right_val) {
            prop_assert!((lv - rv).abs() < 1e-10);
        }
    }

    /// Test: Distributive property - a * (b + c) = a*b + a*c
    #[test]
    fn distributive_property(a in -50i32..50i32, b in -50i32..50i32, c in -50i32..50i32) {
        // a * (b + c)
        let left = Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Integer(a as i64)),
            Box::new(Expression::Binary(
                BinaryOp::Add,
                Box::new(Expression::Integer(b as i64)),
                Box::new(Expression::Integer(c as i64)),
            )),
        );

        // a*b + a*c
        let right = Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Integer(a as i64)),
                Box::new(Expression::Integer(b as i64)),
            )),
            Box::new(Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Integer(a as i64)),
                Box::new(Expression::Integer(c as i64)),
            )),
        );

        let vars = HashMap::new();
        let left_val = left.simplify().evaluate(&vars);
        let right_val = right.simplify().evaluate(&vars);

        if let (Some(lv), Some(rv)) = (left_val, right_val) {
            prop_assert!((lv - rv).abs() < 1e-10);
        }
    }
}

// ============================================================================
// Known Values Tests
// ============================================================================

#[test]
fn known_trigonometric_values() {
    use std::f64::consts::PI;

    // sin(0) = 0
    let sin_0 = Expression::Function(thales::ast::Function::Sin, vec![Expression::Float(0.0)]);
    assert!((sin_0.evaluate(&HashMap::new()).unwrap() - 0.0).abs() < 1e-10);

    // cos(0) = 1
    let cos_0 = Expression::Function(thales::ast::Function::Cos, vec![Expression::Float(0.0)]);
    assert!((cos_0.evaluate(&HashMap::new()).unwrap() - 1.0).abs() < 1e-10);

    // sin(π/2) = 1
    let sin_pi_2 = Expression::Function(
        thales::ast::Function::Sin,
        vec![Expression::Float(PI / 2.0)],
    );
    assert!((sin_pi_2.evaluate(&HashMap::new()).unwrap() - 1.0).abs() < 1e-10);

    // cos(π/2) ≈ 0
    let cos_pi_2 = Expression::Function(
        thales::ast::Function::Cos,
        vec![Expression::Float(PI / 2.0)],
    );
    assert!(cos_pi_2.evaluate(&HashMap::new()).unwrap().abs() < 1e-10);

    // sin(π/6) = 0.5
    let sin_pi_6 = Expression::Function(
        thales::ast::Function::Sin,
        vec![Expression::Float(PI / 6.0)],
    );
    assert!((sin_pi_6.evaluate(&HashMap::new()).unwrap() - 0.5).abs() < 1e-10);

    // tan(π/4) = 1
    let tan_pi_4 = Expression::Function(
        thales::ast::Function::Tan,
        vec![Expression::Float(PI / 4.0)],
    );
    assert!((tan_pi_4.evaluate(&HashMap::new()).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn known_exponential_values() {
    use std::f64::consts::E;

    // e^0 = 1
    let exp_0 = Expression::Function(thales::ast::Function::Exp, vec![Expression::Float(0.0)]);
    assert!((exp_0.evaluate(&HashMap::new()).unwrap() - 1.0).abs() < 1e-10);

    // e^1 = e
    let exp_1 = Expression::Function(thales::ast::Function::Exp, vec![Expression::Float(1.0)]);
    assert!((exp_1.evaluate(&HashMap::new()).unwrap() - E).abs() < 1e-10);

    // ln(1) = 0
    let ln_1 = Expression::Function(thales::ast::Function::Ln, vec![Expression::Float(1.0)]);
    assert!((ln_1.evaluate(&HashMap::new()).unwrap() - 0.0).abs() < 1e-10);

    // ln(e) = 1
    let ln_e = Expression::Function(thales::ast::Function::Ln, vec![Expression::Float(E)]);
    assert!((ln_e.evaluate(&HashMap::new()).unwrap() - 1.0).abs() < 1e-10);

    // log10(10) = 1
    let log10_10 =
        Expression::Function(thales::ast::Function::Log10, vec![Expression::Float(10.0)]);
    assert!((log10_10.evaluate(&HashMap::new()).unwrap() - 1.0).abs() < 1e-10);

    // log10(100) = 2
    let log10_100 =
        Expression::Function(thales::ast::Function::Log10, vec![Expression::Float(100.0)]);
    assert!((log10_100.evaluate(&HashMap::new()).unwrap() - 2.0).abs() < 1e-10);
}

#[test]
fn known_power_values() {
    // 2^0 = 1
    let expr = Expression::Power(
        Box::new(Expression::Integer(2)),
        Box::new(Expression::Integer(0)),
    );
    assert!((expr.evaluate(&HashMap::new()).unwrap() - 1.0).abs() < 1e-10);

    // 2^3 = 8
    let expr = Expression::Power(
        Box::new(Expression::Integer(2)),
        Box::new(Expression::Integer(3)),
    );
    assert!((expr.evaluate(&HashMap::new()).unwrap() - 8.0).abs() < 1e-10);

    // 5^2 = 25
    let expr = Expression::Power(
        Box::new(Expression::Integer(5)),
        Box::new(Expression::Integer(2)),
    );
    assert!((expr.evaluate(&HashMap::new()).unwrap() - 25.0).abs() < 1e-10);

    // (-1)^2 = 1
    let expr = Expression::Power(
        Box::new(Expression::Integer(-1)),
        Box::new(Expression::Integer(2)),
    );
    assert!((expr.evaluate(&HashMap::new()).unwrap() - 1.0).abs() < 1e-10);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

#[test]
fn edge_case_division_by_zero() {
    // Test that division by zero returns None
    let expr = Expression::Binary(
        BinaryOp::Div,
        Box::new(Expression::Integer(5)),
        Box::new(Expression::Integer(0)),
    );

    let result = expr.evaluate(&HashMap::new());
    assert!(result.is_none() || result.unwrap().is_infinite());
}

#[test]
fn edge_case_empty_input() {
    // Test parsing empty string returns error
    let result = parse_expression("");
    assert!(result.is_err());
}

#[test]
fn edge_case_zero_power_zero() {
    // 0^0 is mathematically undefined, but often defined as 1 in computing
    let expr = Expression::Power(
        Box::new(Expression::Integer(0)),
        Box::new(Expression::Integer(0)),
    );

    let result = expr.evaluate(&HashMap::new());
    // Just verify it handles this edge case without panicking
    assert!(result.is_some());
}

#[test]
fn edge_case_negative_square_root() {
    // sqrt(-1) should return NaN or None
    let expr = Expression::Function(thales::ast::Function::Sqrt, vec![Expression::Integer(-1)]);

    let result = expr.evaluate(&HashMap::new());
    // Should either be None or NaN
    if let Some(val) = result {
        assert!(val.is_nan());
    }
}

#[test]
fn edge_case_overflow_protection() {
    // Test large exponentiation doesn't panic
    let expr = Expression::Power(
        Box::new(Expression::Integer(10)),
        Box::new(Expression::Integer(100)),
    );

    let result = expr.evaluate(&HashMap::new());
    // Should handle large numbers gracefully
    assert!(result.is_some());
    if let Some(val) = result {
        assert!(val.is_finite() || val.is_infinite());
    }
}

#[test]
fn edge_case_zero_cartesian_to_polar() {
    // Converting (0, 0) to polar should handle gracefully
    let cart = Cartesian2D::new(0.0, 0.0);
    let polar = cart.to_polar();

    // Magnitude should be 0, angle is undefined but shouldn't panic
    assert!((polar.r - 0.0).abs() < 1e-10);
    // Angle may be any value when r=0
}

#[test]
fn edge_case_simplification_idempotent() {
    // Simplifying twice should give same result as simplifying once
    let expr = Expression::Binary(
        BinaryOp::Add,
        Box::new(Expression::Integer(0)),
        Box::new(Expression::Variable(Variable::new("x"))),
    );

    let simplified_once = expr.simplify();
    let simplified_twice = simplified_once.simplify();

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 5.0);

    let val1 = simplified_once.evaluate(&vars);
    let val2 = simplified_twice.evaluate(&vars);

    if let (Some(v1), Some(v2)) = (val1, val2) {
        assert!((v1 - v2).abs() < 1e-10);
    }
}

#[test]
fn edge_case_unary_negation() {
    // Test double negation: -(-x) = x
    let x = Expression::Variable(Variable::new("x"));
    let neg_x = Expression::Unary(UnaryOp::Neg, Box::new(x.clone()));
    let double_neg = Expression::Unary(UnaryOp::Neg, Box::new(neg_x));

    let simplified = double_neg.simplify();

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 7.0);

    let result = simplified.evaluate(&vars);
    assert_eq!(result, Some(7.0));
}
