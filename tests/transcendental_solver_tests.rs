//! Unit tests for transcendental equation solving.
//!
//! Tests trigonometric, logarithmic, and exponential equation solving capabilities.

use std::f64::consts::{E, PI};
use thales::ast::{BinaryOp, Equation, Expression, Function, Variable};
use thales::solver::{Solver, TranscendentalSolver};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a variable expression
fn var(name: &str) -> Expression {
    Expression::Variable(Variable::new(name))
}

/// Create an integer expression
fn int(n: i64) -> Expression {
    Expression::Integer(n)
}

/// Create a float expression
fn float(x: f64) -> Expression {
    Expression::Float(x)
}

/// Create a function call expression
fn func(f: Function, args: Vec<Expression>) -> Expression {
    Expression::Function(f, args)
}

/// Create a binary operation expression
fn binary(op: BinaryOp, left: Expression, right: Expression) -> Expression {
    Expression::Binary(op, Box::new(left), Box::new(right))
}

/// Create a multiplication expression
fn mul(left: Expression, right: Expression) -> Expression {
    binary(BinaryOp::Mul, left, right)
}

/// Create a division expression
fn div(left: Expression, right: Expression) -> Expression {
    binary(BinaryOp::Div, left, right)
}

/// Create a power expression
fn pow(base: Expression, exp: Expression) -> Expression {
    Expression::Power(Box::new(base), Box::new(exp))
}

/// Check if two floats are approximately equal
fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
}

/// Extract float value from expression
fn extract_float(expr: &Expression) -> Option<f64> {
    match expr {
        Expression::Float(x) => Some(*x),
        Expression::Integer(n) => Some(*n as f64),
        _ => None,
    }
}

// ============================================================================
// Trigonometric Equation Tests
// ============================================================================

#[test]
fn test_sin_equation_simple() {
    // sin(x) = 0.5  →  x = asin(0.5) ≈ 0.5236 rad (30°)
    let equation = Equation::new("test", func(Function::Sin, vec![var("x")]), float(0.5));
    let solver = TranscendentalSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok(), "Failed to solve sin(x) = 0.5");

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            // Should be asin(0.5)
            if let Some(val) = extract_float(&expr) {
                // asin(0.5) = π/6 ≈ 0.5236
                assert!(approx_eq(val, PI / 6.0, 1e-4), "Expected π/6, got {}", val);
            } else {
                panic!("Expected numeric result, got: {:?}", expr);
            }
        }
        _ => panic!("Expected unique solution"),
    }
}

#[test]
fn test_cos_equation_simple() {
    // cos(x) = 0.5  →  x = acos(0.5) ≈ 1.0472 rad (60°)
    let equation = Equation::new("test", func(Function::Cos, vec![var("x")]), float(0.5));
    let solver = TranscendentalSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok(), "Failed to solve cos(x) = 0.5");

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            if let Some(val) = extract_float(&expr) {
                // acos(0.5) = π/3 ≈ 1.0472
                assert!(approx_eq(val, PI / 3.0, 1e-4), "Expected π/3, got {}", val);
            } else {
                panic!("Expected numeric result, got: {:?}", expr);
            }
        }
        _ => panic!("Expected unique solution"),
    }
}

#[test]
fn test_tan_equation_simple() {
    // tan(x) = 1  →  x = atan(1) = π/4
    let equation = Equation::new("test", func(Function::Tan, vec![var("x")]), float(1.0));
    let solver = TranscendentalSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok(), "Failed to solve tan(x) = 1");

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            if let Some(val) = extract_float(&expr) {
                // atan(1) = π/4 ≈ 0.7854
                assert!(approx_eq(val, PI / 4.0, 1e-4), "Expected π/4, got {}", val);
            } else {
                panic!("Expected numeric result, got: {:?}", expr);
            }
        }
        _ => panic!("Expected unique solution"),
    }
}

#[test]
fn test_sin_equation_reversed() {
    // 0.5 = sin(x)  →  x = asin(0.5)
    let equation = Equation::new("test", float(0.5), func(Function::Sin, vec![var("x")]));
    let solver = TranscendentalSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok(), "Failed to solve 0.5 = sin(x)");

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            if let Some(val) = extract_float(&expr) {
                assert!(approx_eq(val, PI / 6.0, 1e-4));
            } else {
                panic!("Expected numeric result, got: {:?}", expr);
            }
        }
        _ => panic!("Expected unique solution"),
    }
}

#[test]
fn test_trig_with_coefficient() {
    // 2 * sin(x) = 1  →  sin(x) = 0.5  →  x = asin(0.5)
    let equation = Equation::new(
        "test",
        mul(int(2), func(Function::Sin, vec![var("x")])),
        float(1.0),
    );
    let solver = TranscendentalSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok(), "Failed to solve 2*sin(x) = 1");

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            if let Some(val) = extract_float(&expr) {
                assert!(approx_eq(val, PI / 6.0, 1e-4));
            } else {
                panic!("Expected numeric result, got: {:?}", expr);
            }
        }
        _ => panic!("Expected unique solution"),
    }
}

#[test]
fn test_trig_domain_error() {
    // sin(x) = 2  →  should fail (domain error: |value| must be ≤ 1)
    let equation = Equation::new("test", func(Function::Sin, vec![var("x")]), float(2.0));
    let solver = TranscendentalSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(result.is_err(), "Should fail for sin(x) = 2 (domain error)");
}

// ============================================================================
// Logarithmic Equation Tests
// ============================================================================

#[test]
fn test_ln_equation_simple() {
    // ln(x) = 2  →  x = e^2
    let equation = Equation::new("test", func(Function::Ln, vec![var("x")]), float(2.0));
    let solver = TranscendentalSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok(), "Failed to solve ln(x) = 2");

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            if let Some(val) = extract_float(&expr) {
                // e^2 ≈ 7.389
                assert!(
                    approx_eq(val, E.powf(2.0), 1e-4),
                    "Expected e^2, got {}",
                    val
                );
            } else {
                panic!("Expected numeric result, got: {:?}", expr);
            }
        }
        _ => panic!("Expected unique solution"),
    }
}

#[test]
fn test_log10_equation_simple() {
    // log10(x) = 2  →  x = 10^2 = 100
    let equation = Equation::new("test", func(Function::Log10, vec![var("x")]), float(2.0));
    let solver = TranscendentalSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok(), "Failed to solve log10(x) = 2");

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            if let Some(val) = extract_float(&expr) {
                assert!(approx_eq(val, 100.0, 1e-4), "Expected 100, got {}", val);
            } else {
                panic!("Expected numeric result, got: {:?}", expr);
            }
        }
        _ => panic!("Expected unique solution"),
    }
}

#[test]
fn test_log_base_equation() {
    // log(x, 2) = 3  →  x = 2^3 = 8
    let equation = Equation::new(
        "test",
        func(Function::Log, vec![var("x"), int(2)]),
        float(3.0),
    );
    let solver = TranscendentalSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok(), "Failed to solve log(x, 2) = 3");

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            if let Some(val) = extract_float(&expr) {
                assert!(approx_eq(val, 8.0, 1e-4), "Expected 8, got {}", val);
            } else {
                panic!("Expected numeric result, got: {:?}", expr);
            }
        }
        _ => panic!("Expected unique solution"),
    }
}

#[test]
fn test_ln_equation_reversed() {
    // 2 = ln(x)  →  x = e^2
    let equation = Equation::new("test", float(2.0), func(Function::Ln, vec![var("x")]));
    let solver = TranscendentalSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok(), "Failed to solve 2 = ln(x)");

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            if let Some(val) = extract_float(&expr) {
                assert!(approx_eq(val, E.powf(2.0), 1e-4));
            } else {
                panic!("Expected numeric result, got: {:?}", expr);
            }
        }
        _ => panic!("Expected unique solution"),
    }
}

// ============================================================================
// Exponential Equation Tests
// ============================================================================

#[test]
fn test_exp_equation_simple() {
    // exp(x) = 10  →  x = ln(10)
    let equation = Equation::new("test", func(Function::Exp, vec![var("x")]), float(10.0));
    let solver = TranscendentalSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok(), "Failed to solve exp(x) = 10");

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            if let Some(val) = extract_float(&expr) {
                // ln(10) ≈ 2.3026
                assert!(
                    approx_eq(val, 10_f64.ln(), 1e-4),
                    "Expected ln(10), got {}",
                    val
                );
            } else {
                panic!("Expected numeric result, got: {:?}", expr);
            }
        }
        _ => panic!("Expected unique solution"),
    }
}

#[test]
fn test_power_equation_simple() {
    // 2^x = 8  →  x = ln(8) / ln(2) = 3
    let equation = Equation::new("test", pow(int(2), var("x")), float(8.0));
    let solver = TranscendentalSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok(), "Failed to solve 2^x = 8");

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            if let Some(val) = extract_float(&expr) {
                assert!(approx_eq(val, 3.0, 1e-4), "Expected 3, got {}", val);
            } else {
                panic!("Expected numeric result, got: {:?}", expr);
            }
        }
        _ => panic!("Expected unique solution"),
    }
}

#[test]
fn test_power_equation_reversed() {
    // 8 = 2^x  →  x = 3
    let equation = Equation::new("test", float(8.0), pow(int(2), var("x")));
    let solver = TranscendentalSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok(), "Failed to solve 8 = 2^x");

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            if let Some(val) = extract_float(&expr) {
                assert!(approx_eq(val, 3.0, 1e-4));
            } else {
                panic!("Expected numeric result, got: {:?}", expr);
            }
        }
        _ => panic!("Expected unique solution"),
    }
}

#[test]
fn test_exp_with_coefficient() {
    // 2 * exp(x) = 10  →  exp(x) = 5  →  x = ln(5)
    let equation = Equation::new(
        "test",
        mul(int(2), func(Function::Exp, vec![var("x")])),
        float(10.0),
    );
    let solver = TranscendentalSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok(), "Failed to solve 2*exp(x) = 10");

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            if let Some(val) = extract_float(&expr) {
                // ln(5) ≈ 1.6094
                assert!(
                    approx_eq(val, 5_f64.ln(), 1e-4),
                    "Expected ln(5), got {}",
                    val
                );
            } else {
                panic!("Expected numeric result, got: {:?}", expr);
            }
        }
        _ => panic!("Expected unique solution"),
    }
}

// ============================================================================
// Physics Equation Tests (Projectile Motion)
// ============================================================================

#[test]
fn test_projectile_range_solve_for_theta() {
    // R = (v0^2 * sin(2*theta)) / g, solve for theta given known R, v0, g
    // This is a complex case: sin(2*theta) = R*g/v0^2, then theta = asin(...)/2
    // For now, we test the simpler form: sin(x) = value

    // Simplified test: sin(2*theta) = 0.5  →  2*theta = asin(0.5)  →  theta = asin(0.5) / 2
    let equation = Equation::new(
        "test",
        func(Function::Sin, vec![mul(int(2), var("theta"))]),
        float(0.5),
    );
    let solver = TranscendentalSolver::new();
    let target = Variable::new("theta");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok(), "Failed to solve sin(2*theta) = 0.5");

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            // theta = asin(0.5) / 2 = (π/6) / 2 = π/12
            if let Some(val) = extract_float(&expr) {
                assert!(
                    approx_eq(val, PI / 12.0, 1e-4),
                    "Expected π/12, got {}",
                    val
                );
            } else {
                panic!("Expected numeric result, got: {:?}", expr);
            }
        }
        _ => panic!("Expected unique solution"),
    }
}

// ============================================================================
// can_solve Tests
// ============================================================================

#[test]
fn test_can_solve_trigonometric() {
    let equation = Equation::new("test", func(Function::Sin, vec![var("x")]), float(0.5));
    let solver = TranscendentalSolver::new();
    assert!(solver.can_solve(&equation));
}

#[test]
fn test_can_solve_logarithmic() {
    let equation = Equation::new("test", func(Function::Ln, vec![var("x")]), float(2.0));
    let solver = TranscendentalSolver::new();
    assert!(solver.can_solve(&equation));
}

#[test]
fn test_can_solve_exponential() {
    let equation = Equation::new("test", pow(int(2), var("x")), float(8.0));
    let solver = TranscendentalSolver::new();
    assert!(solver.can_solve(&equation));
}

#[test]
fn test_cannot_solve_linear() {
    // 2*x + 3 = 7 (linear, not transcendental)
    let equation = Equation::new(
        "test",
        binary(BinaryOp::Add, mul(int(2), var("x")), int(3)),
        int(7),
    );
    let solver = TranscendentalSolver::new();
    assert!(!solver.can_solve(&equation));
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[test]
fn test_variable_not_found() {
    // sin(y) = 0.5, but solving for x
    let equation = Equation::new("test", func(Function::Sin, vec![var("y")]), float(0.5));
    let solver = TranscendentalSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(result.is_err(), "Should fail when variable not in equation");
}

#[test]
fn test_complex_pattern_not_supported() {
    // sin(x) + cos(x) = 1 (too complex for pattern matching)
    let equation = Equation::new(
        "test",
        binary(
            BinaryOp::Add,
            func(Function::Sin, vec![var("x")]),
            func(Function::Cos, vec![var("x")]),
        ),
        float(1.0),
    );
    let solver = TranscendentalSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    // This should fail as the pattern is too complex
    assert!(
        result.is_err(),
        "Should fail for complex combined trig functions"
    );
}

#[test]
fn test_asin_domain_validation() {
    // asin requires |x| ≤ 1, test with value = 2
    let equation = Equation::new("test", func(Function::Sin, vec![var("x")]), float(2.0));
    let solver = TranscendentalSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(result.is_err(), "Should fail domain validation for asin(2)");
}

#[test]
fn test_acos_domain_validation() {
    // acos requires |x| ≤ 1, test with value = -1.5
    let equation = Equation::new("test", func(Function::Cos, vec![var("x")]), float(-1.5));
    let solver = TranscendentalSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(
        result.is_err(),
        "Should fail domain validation for acos(-1.5)"
    );
}
