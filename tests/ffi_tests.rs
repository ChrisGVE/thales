//! Tests for FFI-exposed functionality.
//!
//! These tests verify that the underlying library functions used by the FFI
//! layer work correctly. The FFI functions themselves are tested via Swift
//! integration tests.

use thales::ast::{Expression, Variable};
use thales::parser::parse_expression;
use thales::series::{maclaurin, taylor};
use thales::special::{erf, gamma};

// =============================================================================
// Series Expansion Tests
// =============================================================================

#[test]
fn test_taylor_series_simple_polynomial() {
    // Taylor series of x^2 around x=1 should be 1 + 2(x-1) + (x-1)^2
    let x = Variable::new("x");
    let expr = parse_expression("x^2").unwrap();
    let center = Expression::Float(1.0);

    let result = taylor(&expr, &x, &center, 3);
    assert!(result.is_ok(), "Taylor series computation should succeed");

    let series = result.unwrap();
    let series_expr = series.to_expression();
    let series_str = format!("{}", series_expr);
    assert!(!series_str.is_empty(), "Series should not be empty");
}

#[test]
fn test_taylor_series_exponential() {
    // Taylor series of e^x around x=0
    let x = Variable::new("x");
    let expr = parse_expression("exp(x)").unwrap();
    let center = Expression::Float(0.0);

    let result = taylor(&expr, &x, &center, 4);
    assert!(result.is_ok(), "Taylor series of exp(x) should succeed");

    let series = result.unwrap();
    let series_str = format!("{}", series.to_expression());
    assert!(series_str.contains('x'), "Series should contain variable x");
}

#[test]
fn test_maclaurin_series_sin() {
    // Maclaurin series of sin(x)
    let x = Variable::new("x");
    let expr = parse_expression("sin(x)").unwrap();

    let result = maclaurin(&expr, &x, 5);
    assert!(result.is_ok(), "Maclaurin series of sin(x) should succeed");

    let series = result.unwrap();
    let series_expr = series.to_expression();
    assert!(!format!("{}", series_expr).is_empty(), "Series should not be empty");
}

#[test]
fn test_maclaurin_series_cos() {
    // Maclaurin series of cos(x)
    let x = Variable::new("x");
    let expr = parse_expression("cos(x)").unwrap();

    let result = maclaurin(&expr, &x, 5);
    assert!(result.is_ok(), "Maclaurin series of cos(x) should succeed");

    let series = result.unwrap();
    let series_expr = series.to_expression();
    assert!(!format!("{}", series_expr).is_empty(), "Series should not be empty");
}

#[test]
fn test_maclaurin_series_simple() {
    // Maclaurin series of x^3 + 2*x
    let x = Variable::new("x");
    let expr = parse_expression("x^3 + 2*x").unwrap();

    let result = maclaurin(&expr, &x, 4);
    assert!(result.is_ok(), "Maclaurin series should succeed");

    let series = result.unwrap();
    let series_expr = series.to_expression();
    assert!(!format!("{}", series_expr).is_empty(), "Series should not be empty");
}

// =============================================================================
// Special Functions Tests
// =============================================================================

#[test]
fn test_gamma_positive_integer() {
    // Γ(5) = 4! = 24
    let x = Expression::Integer(5);
    let result = gamma(&x);

    assert!(result.is_ok(), "Gamma of positive integer should succeed");
    let gamma_result = result.unwrap();
    assert_eq!(
        gamma_result.numeric_value,
        Some(24.0),
        "Γ(5) should equal 24"
    );
    assert!(!gamma_result.derivation_steps.is_empty(), "Derivation steps should be present");
}

#[test]
fn test_gamma_one() {
    // Γ(1) = 0! = 1
    let x = Expression::Integer(1);
    let result = gamma(&x);

    assert!(result.is_ok(), "Gamma(1) should succeed");
    assert_eq!(result.unwrap().numeric_value, Some(1.0), "Γ(1) should equal 1");
}

#[test]
fn test_gamma_two() {
    // Γ(2) = 1! = 1
    let x = Expression::Integer(2);
    let result = gamma(&x);

    assert!(result.is_ok(), "Gamma(2) should succeed");
    assert_eq!(result.unwrap().numeric_value, Some(1.0), "Γ(2) should equal 1");
}

#[test]
fn test_gamma_three() {
    // Γ(3) = 2! = 2
    let x = Expression::Integer(3);
    let result = gamma(&x);

    assert!(result.is_ok(), "Gamma(3) should succeed");
    assert_eq!(result.unwrap().numeric_value, Some(2.0), "Γ(3) should equal 2");
}

#[test]
fn test_gamma_four() {
    // Γ(4) = 3! = 6
    let x = Expression::Integer(4);
    let result = gamma(&x);

    assert!(result.is_ok(), "Gamma(4) should succeed");
    assert_eq!(result.unwrap().numeric_value, Some(6.0), "Γ(4) should equal 6");
}

#[test]
fn test_gamma_half() {
    // Γ(1/2) = √π ≈ 1.772453850905516
    let x = Expression::Float(0.5);
    let result = gamma(&x);

    assert!(result.is_ok(), "Gamma(1/2) should succeed");
    let gamma_val = result.unwrap().numeric_value.unwrap();
    assert!(
        (gamma_val - 1.772453850905516).abs() < 0.00001,
        "Γ(1/2) should equal √π, got {}",
        gamma_val
    );
}

#[test]
fn test_gamma_negative() {
    // Γ(-1) is undefined (pole)
    let x = Expression::Integer(-1);
    let result = gamma(&x);

    assert!(result.is_err(), "Gamma of negative integer should fail");
}

#[test]
fn test_erf_zero() {
    // erf(0) = 0
    let x = Expression::Float(0.0);
    let result = erf(&x);

    assert!(result.is_ok(), "erf(0) should succeed");
    let erf_val = result.unwrap().numeric_value.unwrap();
    assert!(
        erf_val.abs() < 0.00001,
        "erf(0) should equal 0, got {}",
        erf_val
    );
}

#[test]
fn test_erf_positive() {
    // erf(1) ≈ 0.8427 (well-known value)
    let x = Expression::Float(1.0);
    let result = erf(&x);

    assert!(result.is_ok(), "erf(1) should succeed");
    let erf_val = result.unwrap().numeric_value.unwrap();
    assert!(
        (erf_val - 0.8427).abs() < 0.01,
        "erf(1) should be approximately 0.8427, got {}",
        erf_val
    );
}

#[test]
fn test_erf_negative() {
    // erf is an odd function: erf(-x) = -erf(x)
    let x_pos = Expression::Float(1.0);
    let x_neg = Expression::Float(-1.0);

    let result_pos = erf(&x_pos).unwrap();
    let result_neg = erf(&x_neg).unwrap();

    let val_pos = result_pos.numeric_value.unwrap();
    let val_neg = result_neg.numeric_value.unwrap();

    assert!(
        (val_pos + val_neg).abs() < 0.00001,
        "erf(-x) should equal -erf(x), got {} and {}",
        val_pos,
        val_neg
    );
}

#[test]
fn test_erf_has_derivation_steps() {
    let x = Expression::Float(1.0);
    let result = erf(&x);

    assert!(result.is_ok(), "erf(1) should succeed");
    let erf_result = result.unwrap();
    assert!(!erf_result.derivation_steps.is_empty(), "Derivation steps should be present");
}

// =============================================================================
// Verify LaTeX output for new functions
// =============================================================================

#[test]
fn test_series_latex_output() {
    let x = Variable::new("x");
    let expr = parse_expression("x^2").unwrap();

    let series = maclaurin(&expr, &x, 3).unwrap();
    let series_expr = series.to_expression();
    let latex = series_expr.to_latex();

    assert!(!latex.is_empty(), "LaTeX output should not be empty");
    assert!(latex.contains('x'), "LaTeX should contain variable");
}

#[test]
fn test_gamma_latex_output() {
    let x = Expression::Integer(5);
    let result = gamma(&x).unwrap();
    let latex = result.value.to_latex();

    assert!(!latex.is_empty(), "LaTeX output should not be empty");
}

#[test]
fn test_erf_latex_output() {
    let x = Expression::Float(1.0);
    let result = erf(&x).unwrap();
    let latex = result.value.to_latex();

    assert!(!latex.is_empty(), "LaTeX output should not be empty");
}
