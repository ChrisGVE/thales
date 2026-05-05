//! Integration tests for the Task 8 parser-integration entry points.
//!
//! These tests verify that:
//! - `parse_to_expr` / `parse_equation_to_expr` / `parse_equation_system_to_expr`
//!   produce canonical internal `Expr` values semantically equivalent to the
//!   legacy `parse_expression` / `parse_equation` / `parse_equation_system`
//!   pipeline.
//! - `parse_latex_to_expr` / `parse_latex_equation_to_expr` do the same for
//!   LaTeX input.
//! - Parse errors surface unchanged (the public error types are preserved).

use std::sync::Arc;

use thales::latex::{parse_latex_equation_to_expr, parse_latex_to_expr};
use thales::numeric::compile::compile;
use thales::numeric::expr::Expr;
use thales::parser::{
    parse_equation, parse_equation_system, parse_equation_system_to_expr, parse_equation_to_expr,
    parse_expression, parse_to_expr,
};

fn compiled(input: &str) -> Arc<Expr> {
    let expression = parse_expression(input).expect("parse_expression must succeed");
    compile(&expression)
}

#[test]
fn parse_to_expr_simple_sum_matches_compiled() {
    let direct = parse_to_expr("x + 2*y").expect("parse_to_expr must succeed");
    let via_compile = compiled("x + 2*y");
    assert_eq!(*direct, *via_compile);
    assert!(matches!(*direct, Expr::Add(_)));
}

#[test]
fn parse_to_expr_single_symbol() {
    let direct = parse_to_expr("x").expect("parse_to_expr must succeed");
    assert!(matches!(*direct, Expr::Symbol(_)));
}

#[test]
fn parse_to_expr_integer_literal() {
    let direct = parse_to_expr("42").expect("parse_to_expr must succeed");
    assert!(matches!(*direct, Expr::Integer(_)));
}

#[test]
fn parse_to_expr_function_call() {
    let direct = parse_to_expr("sin(x)").expect("parse_to_expr must succeed");
    match &*direct {
        Expr::Func(_, args) => assert_eq!(args.len(), 1),
        other => panic!("Expected Func, got {other:?}"),
    }
}

#[test]
fn parse_to_expr_power_preserves_shape() {
    let direct = parse_to_expr("x^3").expect("parse_to_expr must succeed");
    // Canonicalisation may keep this as `Pow` or lift into a `Mul` exponent,
    // but it must equal the compile path exactly.
    assert_eq!(*direct, *compiled("x^3"));
}

#[test]
fn parse_to_expr_nested_expression() {
    let direct = parse_to_expr("sin(x)^2 + cos(x)^2").expect("parse_to_expr must succeed");
    let via_compile = compiled("sin(x)^2 + cos(x)^2");
    assert_eq!(*direct, *via_compile);
}

#[test]
fn parse_to_expr_propagates_errors() {
    let err = parse_to_expr("2 * ").expect_err("must fail");
    assert!(!err.is_empty());
}

#[test]
fn parse_equation_to_expr_matches_compiled_sides() {
    let (left, right) = parse_equation_to_expr("x + 2 = 5").expect("must succeed");
    let legacy = parse_equation("x + 2 = 5").expect("legacy must succeed");
    assert_eq!(*left, *compile(&legacy.left));
    assert_eq!(*right, *compile(&legacy.right));
}

#[test]
fn parse_equation_to_expr_quadratic() {
    let (left, right) = parse_equation_to_expr("x^2 - 4 = 0").expect("must succeed");
    assert_eq!(*left, *compiled("x^2 - 4"));
    assert!(matches!(*right, Expr::Integer(_)));
}

#[test]
fn parse_equation_to_expr_propagates_errors() {
    let err = parse_equation_to_expr("x + 2").expect_err("no '=' should fail");
    assert!(!err.is_empty());
}

#[test]
fn parse_equation_system_to_expr_pairs_match_legacy() {
    let pairs = parse_equation_system_to_expr("x + y = 5; 2*x - y = 1").expect("must succeed");
    let legacy = parse_equation_system("x + y = 5; 2*x - y = 1").expect("legacy must succeed");
    assert_eq!(pairs.len(), legacy.len());
    for (pair, eq) in pairs.iter().zip(legacy.iter()) {
        assert_eq!(*pair.0, *compile(&eq.left));
        assert_eq!(*pair.1, *compile(&eq.right));
    }
}

#[test]
fn parse_equation_system_to_expr_empty_input_returns_empty() {
    let pairs = parse_equation_system_to_expr("").expect("must succeed");
    assert!(pairs.is_empty());
}

#[test]
fn parse_equation_system_to_expr_trailing_semicolon_ignored() {
    let pairs = parse_equation_system_to_expr("x = 1;").expect("must succeed");
    assert_eq!(pairs.len(), 1);
}

#[test]
fn parse_equation_system_to_expr_propagates_errors() {
    let err = parse_equation_system_to_expr("x + y = 5; not_an_equation").expect_err("should fail");
    assert!(!err.is_empty());
}

#[test]
fn parse_latex_to_expr_fraction() {
    let direct = parse_latex_to_expr(r"\frac{1}{2}").expect("must succeed");
    // 1/2 in canonical form should be a Rational literal or a Mul with a
    // power-of-2 factor; exact shape comes from compile().
    let via_compile = {
        let e = thales::latex::parse_latex(r"\frac{1}{2}").expect("legacy must succeed");
        compile(&e)
    };
    assert_eq!(*direct, *via_compile);
}

#[test]
fn parse_latex_to_expr_power_matches_parser() {
    let direct = parse_latex_to_expr(r"x^{2}").expect("must succeed");
    let legacy = parse_to_expr("x^2").expect("parser must succeed");
    assert_eq!(*direct, *legacy);
}

#[test]
fn parse_latex_equation_to_expr_quadratic() {
    let (left, right) = parse_latex_equation_to_expr(r"x^2 = 4").expect("must succeed");
    assert_eq!(*left, *compiled("x^2"));
    assert!(matches!(*right, Expr::Integer(_)));
}

#[test]
fn parse_latex_to_expr_propagates_errors() {
    let err = parse_latex_to_expr(r"\frac{1}").expect_err("missing denom must fail");
    assert!(!err.is_empty());
}
