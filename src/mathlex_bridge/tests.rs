//! Tests for the mathlex_bridge conversion and ODE extraction logic.

use super::convert::{convert_equation, convert_expression};
use super::helpers::match_function_name;
use super::ode::{try_extract_ode, ExtractedODE};
use crate::ast::{BinaryOp, Expression, Function, SymbolicConstant, UnaryOp, Variable};

#[test]
fn test_convert_integer() {
    let ml = mathlex::Expression::integer(42);
    let result = convert_expression(&ml).unwrap();
    assert_eq!(result, Expression::Integer(42));
}

#[test]
fn test_convert_variable() {
    let ml = mathlex::Expression::variable("x");
    let result = convert_expression(&ml).unwrap();
    assert_eq!(result, Expression::Variable(Variable::new("x")));
}

#[test]
fn test_convert_pi() {
    let ml = mathlex::Expression::constant(mathlex::MathConstant::Pi);
    let result = convert_expression(&ml).unwrap();
    assert_eq!(result, Expression::Constant(SymbolicConstant::Pi));
}

#[test]
fn test_convert_addition() {
    let ml = mathlex::ExprKind::Binary {
        op: mathlex::BinaryOp::Add,
        left: Box::new(mathlex::Expression::integer(1)),
        right: Box::new(mathlex::Expression::integer(2)),
    }
    .into();
    let result = convert_expression(&ml).unwrap();
    assert_eq!(
        result,
        Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Integer(1)),
            Box::new(Expression::Integer(2))
        )
    );
}

#[test]
fn test_convert_power() {
    let ml = mathlex::ExprKind::Binary {
        op: mathlex::BinaryOp::Pow,
        left: Box::new(mathlex::Expression::variable("x")),
        right: Box::new(mathlex::Expression::integer(2)),
    }
    .into();
    let result = convert_expression(&ml).unwrap();
    assert_eq!(
        result,
        Expression::Power(
            Box::new(Expression::Variable(Variable::new("x"))),
            Box::new(Expression::Integer(2))
        )
    );
}

#[test]
fn test_convert_sin() {
    let ml = mathlex::ExprKind::Function {
        name: "sin".to_string(),
        args: vec![mathlex::Expression::variable("x")],
    }
    .into();
    let result = convert_expression(&ml).unwrap();
    assert_eq!(
        result,
        Expression::Function(
            Function::Sin,
            vec![Expression::Variable(Variable::new("x"))]
        )
    );
}

#[test]
fn test_convert_negation() {
    let ml = mathlex::ExprKind::Unary {
        op: mathlex::UnaryOp::Neg,
        operand: Box::new(mathlex::Expression::integer(5)),
    }
    .into();
    let result = convert_expression(&ml).unwrap();
    assert_eq!(
        result,
        Expression::Unary(UnaryOp::Neg, Box::new(Expression::Integer(5)))
    );
}

#[test]
fn test_function_name_aliases() {
    assert_eq!(match_function_name("arcsin"), Function::Asin);
    assert_eq!(match_function_name("asin"), Function::Asin);
    assert_eq!(match_function_name("sgn"), Function::Sign);
    assert_eq!(match_function_name("sign"), Function::Sign);
    assert_eq!(match_function_name("lg"), Function::Log2);
    assert_eq!(match_function_name("cbrt"), Function::Cbrt);
    assert_eq!(match_function_name("round"), Function::Round);
}

#[test]
fn test_convert_equation() {
    let ml = mathlex::ExprKind::Equation {
        left: Box::new(mathlex::Expression::variable("x")),
        right: Box::new(mathlex::Expression::integer(5)),
    }
    .into();
    let eq = convert_equation(&ml).unwrap();
    assert_eq!(eq.left, Expression::Variable(Variable::new("x")));
    assert_eq!(eq.right, Expression::Integer(5));
}

// ------------------------------------------------------------------
// Derivative conversion
// ------------------------------------------------------------------

#[test]
fn test_convert_derivative_first_order() {
    // d/dx(x^2) should evaluate to 2x
    let ml = mathlex::ExprKind::Derivative {
        expr: Box::new(
            mathlex::ExprKind::Binary {
                op: mathlex::BinaryOp::Pow,
                left: Box::new(mathlex::Expression::variable("x")),
                right: Box::new(mathlex::Expression::integer(2)),
            }
            .into(),
        ),
        var: "x".to_string(),
        order: 1,
    }
    .into();
    let result = convert_expression(&ml).unwrap();
    // d/dx(x^2) = 2x — check it evaluates to 2 at x=1
    let mut env = std::collections::HashMap::new();
    env.insert("x".to_string(), 1.0);
    assert_eq!(result.evaluate(&env), Some(2.0));
}

#[test]
fn test_convert_derivative_second_order() {
    // d²/dx²(x³) should evaluate to 6x
    let ml = mathlex::ExprKind::Derivative {
        expr: Box::new(
            mathlex::ExprKind::Binary {
                op: mathlex::BinaryOp::Pow,
                left: Box::new(mathlex::Expression::variable("x")),
                right: Box::new(mathlex::Expression::integer(3)),
            }
            .into(),
        ),
        var: "x".to_string(),
        order: 2,
    }
    .into();
    let result = convert_expression(&ml).unwrap();
    let mut env = std::collections::HashMap::new();
    env.insert("x".to_string(), 2.0);
    // d²/dx²(x³) = 6x, at x=2: 12.0
    assert_eq!(result.evaluate(&env), Some(12.0));
}

#[test]
fn test_convert_partial_derivative() {
    // ∂/∂x(x²·y) should evaluate to 2xy
    let ml = mathlex::ExprKind::PartialDerivative {
        expr: Box::new(
            mathlex::ExprKind::Binary {
                op: mathlex::BinaryOp::Mul,
                left: Box::new(
                    mathlex::ExprKind::Binary {
                        op: mathlex::BinaryOp::Pow,
                        left: Box::new(mathlex::Expression::variable("x")),
                        right: Box::new(mathlex::Expression::integer(2)),
                    }
                    .into(),
                ),
                right: Box::new(mathlex::Expression::variable("y")),
            }
            .into(),
        ),
        var: "x".to_string(),
        order: 1,
    }
    .into();
    let result = convert_expression(&ml).unwrap();
    let mut env = std::collections::HashMap::new();
    env.insert("x".to_string(), 3.0);
    env.insert("y".to_string(), 2.0);
    // ∂/∂x(x²y) = 2xy, at x=3, y=2: 12.0
    assert_eq!(result.evaluate(&env), Some(12.0));
}

#[test]
fn test_convert_gradient() {
    let ml = mathlex::ExprKind::Gradient {
        expr: Box::new(mathlex::Expression::variable("f")),
    }
    .into();
    let result = convert_expression(&ml).unwrap();
    assert!(matches!(
        result,
        Expression::Function(Function::Custom(ref name), _) if name == "gradient"
    ));
}

// ------------------------------------------------------------------
// Derivative conversion via mathlex parser
// ------------------------------------------------------------------

#[test]
fn test_parse_and_convert_leibniz_derivative() {
    let ml = mathlex::parse("dy/dx").unwrap();
    // Should be Derivative { expr: Variable("y"), var: "x", order: 1 }
    let result = convert_expression(&ml).unwrap();
    // d/dx(y) — y is independent of x, so symbolic diff would give 0.
    // But since that would lose variable "y", the bridge preserves
    // it as an opaque derivative wrapper so the solver can find y.
    assert!(result.contains_variable("y"));
    assert!(matches!(
        result,
        Expression::Function(Function::Custom(_), _)
    ));
}

#[test]
fn test_parse_and_convert_prime_derivative() {
    // y' has var="" (implicit independent variable)
    let ml = mathlex::parse("y'").unwrap();
    assert!(matches!(&ml.kind, mathlex::ExprKind::Derivative { .. }));
}

// ------------------------------------------------------------------
// ODE extraction
// ------------------------------------------------------------------

#[test]
fn test_extract_first_order_ode_leibniz() {
    let ml = mathlex::parse("dy/dx = x*y").unwrap();
    let ode = try_extract_ode(&ml).unwrap();
    match ode {
        ExtractedODE::First(fo) => {
            assert_eq!(fo.dependent, "y");
            assert_eq!(fo.independent, "x");
            // rhs should be x*y
            let mut env = std::collections::HashMap::new();
            env.insert(crate::numeric::SymbolId::intern("x"), 2.0);
            env.insert(crate::numeric::SymbolId::intern("y"), 3.0);
            assert_eq!(
                crate::numeric::evaluation::evaluate(&fo.rhs_arc(), &env),
                Some(6.0)
            );
        }
        _ => panic!("expected FirstOrderODE"),
    }
}

#[test]
fn test_extract_first_order_ode_simple() {
    // dy/dx = y
    let ml = mathlex::parse("dy/dx = y").unwrap();
    let ode = try_extract_ode(&ml).unwrap();
    match ode {
        ExtractedODE::First(fo) => {
            assert_eq!(fo.dependent, "y");
            assert_eq!(fo.independent, "x");
        }
        _ => panic!("expected FirstOrderODE"),
    }
}

#[test]
fn test_extract_second_order_ode_homogeneous() {
    // d2y/dx2 + 3*dy/dx + 2*y = 0
    let ml = mathlex::parse("d2y/dx2 + 3*dy/dx + 2*y = 0").unwrap();
    let ode = try_extract_ode(&ml).unwrap();
    match ode {
        ExtractedODE::Second(so) => {
            assert_eq!(so.dependent, "y");
            assert_eq!(so.independent, "x");
            assert!((so.a - 1.0).abs() < 1e-10);
            assert!((so.b - 3.0).abs() < 1e-10);
            assert!((so.c - 2.0).abs() < 1e-10);
            assert!(so.is_homogeneous());
        }
        _ => panic!("expected SecondOrderODE"),
    }
}

#[test]
fn test_extract_second_order_ode_forced() {
    // d2y/dx2 + 2*dy/dx + y = x
    let ml = mathlex::parse("d2y/dx2 + 2*dy/dx + y = x").unwrap();
    let ode = try_extract_ode(&ml).unwrap();
    match ode {
        ExtractedODE::Second(so) => {
            assert_eq!(so.dependent, "y");
            assert_eq!(so.independent, "x");
            assert!((so.a - 1.0).abs() < 1e-10);
            assert!((so.b - 2.0).abs() < 1e-10);
            assert!((so.c - 1.0).abs() < 1e-10);
            assert!(!so.is_homogeneous());
        }
        _ => panic!("expected SecondOrderODE"),
    }
}

#[test]
fn test_extract_ode_not_an_equation() {
    let ml = mathlex::parse("dy/dx").unwrap();
    assert!(try_extract_ode(&ml).is_none());
}

#[test]
fn test_extract_ode_no_derivatives() {
    let ml = mathlex::parse("x + y = 0").unwrap();
    assert!(try_extract_ode(&ml).is_none());
}

// ==================================================================
// LaTeX variants — each test above has a LaTeX counterpart ensuring
// both parsers produce equivalent results through the bridge.
// ==================================================================

#[test]
fn test_convert_derivative_first_order_latex() {
    // \frac{d}{dx}(x^2) should evaluate to 2x
    let ml = mathlex::parse_latex(r#"\frac{d}{dx}(x^2)"#).unwrap();
    let result = convert_expression(&ml).unwrap();
    let mut env = std::collections::HashMap::new();
    env.insert("x".to_string(), 1.0);
    assert_eq!(result.evaluate(&env), Some(2.0));
}

#[test]
fn test_convert_derivative_second_order_latex() {
    // \frac{d^2}{dx^2}(x^3) should evaluate to 6x
    let ml = mathlex::parse_latex(r#"\frac{d^2}{dx^2}(x^3)"#).unwrap();
    let result = convert_expression(&ml).unwrap();
    let mut env = std::collections::HashMap::new();
    env.insert("x".to_string(), 2.0);
    assert_eq!(result.evaluate(&env), Some(12.0));
}

#[test]
fn test_convert_partial_derivative_latex() {
    // \frac{\partial}{\partial x}(x^2 \cdot y) should evaluate to 2xy
    let ml = mathlex::parse_latex(r#"\frac{\partial}{\partial x}(x^2 \cdot y)"#).unwrap();
    let result = convert_expression(&ml).unwrap();
    let mut env = std::collections::HashMap::new();
    env.insert("x".to_string(), 3.0);
    env.insert("y".to_string(), 2.0);
    assert_eq!(result.evaluate(&env), Some(12.0));
}

#[test]
fn test_convert_gradient_latex() {
    let ml = mathlex::parse_latex(r#"\nabla f"#).unwrap();
    let result = convert_expression(&ml).unwrap();
    assert!(matches!(
        result,
        Expression::Function(Function::Custom(ref name), _) if name == "gradient"
    ));
}

#[test]
fn test_parse_and_convert_derivative_latex() {
    // \frac{d}{dx}(y) — y is independent of x; preserved as opaque derivative
    let ml = mathlex::parse_latex(r#"\frac{d}{dx}(y)"#).unwrap();
    let result = convert_expression(&ml).unwrap();
    assert!(result.contains_variable("y"));
    assert!(matches!(
        result,
        Expression::Function(Function::Custom(_), _)
    ));
}

#[test]
fn test_extract_first_order_ode_latex() {
    // \frac{d}{dx}(y) = x \cdot y
    let ml = mathlex::parse_latex(r#"\frac{d}{dx}(y) = x \cdot y"#).unwrap();
    let ode = try_extract_ode(&ml).unwrap();
    match ode {
        ExtractedODE::First(fo) => {
            assert_eq!(fo.dependent, "y");
            assert_eq!(fo.independent, "x");
            let mut env = std::collections::HashMap::new();
            env.insert(crate::numeric::SymbolId::intern("x"), 2.0);
            env.insert(crate::numeric::SymbolId::intern("y"), 3.0);
            assert_eq!(
                crate::numeric::evaluation::evaluate(&fo.rhs_arc(), &env),
                Some(6.0)
            );
        }
        _ => panic!("expected FirstOrderODE"),
    }
}

#[test]
fn test_extract_first_order_ode_simple_latex() {
    // \frac{d}{dx}(y) = y
    let ml = mathlex::parse_latex(r#"\frac{d}{dx}(y) = y"#).unwrap();
    let ode = try_extract_ode(&ml).unwrap();
    match ode {
        ExtractedODE::First(fo) => {
            assert_eq!(fo.dependent, "y");
            assert_eq!(fo.independent, "x");
        }
        _ => panic!("expected FirstOrderODE"),
    }
}

#[test]
fn test_extract_second_order_ode_homogeneous_latex() {
    // \frac{d^2}{dx^2}(y) + 3\frac{d}{dx}(y) + 2y = 0
    let ml = mathlex::parse_latex(r#"\frac{d^2}{dx^2}(y) + 3\frac{d}{dx}(y) + 2y = 0"#).unwrap();
    let ode = try_extract_ode(&ml).unwrap();
    match ode {
        ExtractedODE::Second(so) => {
            assert_eq!(so.dependent, "y");
            assert_eq!(so.independent, "x");
            assert!((so.a - 1.0).abs() < 1e-10);
            assert!((so.b - 3.0).abs() < 1e-10);
            assert!((so.c - 2.0).abs() < 1e-10);
            assert!(so.is_homogeneous());
        }
        _ => panic!("expected SecondOrderODE"),
    }
}

#[test]
fn test_extract_second_order_ode_forced_latex() {
    // \frac{d^2}{dx^2}(y) + 2\frac{d}{dx}(y) + y = x
    let ml = mathlex::parse_latex(r#"\frac{d^2}{dx^2}(y) + 2\frac{d}{dx}(y) + y = x"#).unwrap();
    let ode = try_extract_ode(&ml).unwrap();
    match ode {
        ExtractedODE::Second(so) => {
            assert_eq!(so.dependent, "y");
            assert_eq!(so.independent, "x");
            assert!((so.a - 1.0).abs() < 1e-10);
            assert!((so.b - 2.0).abs() < 1e-10);
            assert!((so.c - 1.0).abs() < 1e-10);
            assert!(!so.is_homogeneous());
        }
        _ => panic!("expected SecondOrderODE"),
    }
}

#[test]
fn test_extract_ode_not_an_equation_latex() {
    let ml = mathlex::parse_latex(r#"\frac{d}{dx}(y)"#).unwrap();
    assert!(try_extract_ode(&ml).is_none());
}

#[test]
fn test_extract_ode_no_derivatives_latex() {
    let ml = mathlex::parse_latex(r#"x + y = 0"#).unwrap();
    assert!(try_extract_ode(&ml).is_none());
}

// ── match_function_name for Re/Im/Conj ───────────────────────────────

#[test]
fn test_match_function_name_re_lowercase() {
    use crate::ast::Function;
    assert!(matches!(match_function_name("re"), Function::Re));
}

#[test]
fn test_match_function_name_re_capitalized() {
    use crate::ast::Function;
    assert!(matches!(match_function_name("Re"), Function::Re));
}

#[test]
fn test_match_function_name_im_lowercase() {
    use crate::ast::Function;
    assert!(matches!(match_function_name("im"), Function::Im));
}

#[test]
fn test_match_function_name_im_capitalized() {
    use crate::ast::Function;
    assert!(matches!(match_function_name("Im"), Function::Im));
}

#[test]
fn test_match_function_name_conj() {
    use crate::ast::Function;
    assert!(matches!(match_function_name("conj"), Function::Conj));
    assert!(matches!(match_function_name("Conj"), Function::Conj));
}

#[test]
fn test_match_function_name_conj_uppercase() {
    use crate::ast::Function;
    assert!(matches!(match_function_name("CONJ"), Function::Conj));
}
