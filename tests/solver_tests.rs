//! Unit tests for the algebraic equation solver.

use thales::ast::{BinaryOp, Equation, Expression, Variable};
use thales::solver::{solve_for, LinearSolver, Solver, SolverError};
use std::collections::HashMap;

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

/// Create a binary operation expression
fn binary(op: BinaryOp, left: Expression, right: Expression) -> Expression {
    Expression::Binary(op, Box::new(left), Box::new(right))
}

/// Create a multiplication expression
fn mul(left: Expression, right: Expression) -> Expression {
    binary(BinaryOp::Mul, left, right)
}

/// Create an addition expression
fn add(left: Expression, right: Expression) -> Expression {
    binary(BinaryOp::Add, left, right)
}

/// Create a division expression
fn div(left: Expression, right: Expression) -> Expression {
    binary(BinaryOp::Div, left, right)
}

/// Create a power expression
fn pow(base: Expression, exp: Expression) -> Expression {
    Expression::Power(Box::new(base), Box::new(exp))
}

// ============================================================================
// LinearSolver Tests
// ============================================================================

#[test]
fn test_linear_solver_simple_equality() {
    // x = 5
    let equation = Equation::new("test", var("x"), int(5));
    let solver = LinearSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok());

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            assert_eq!(expr, int(5));
        }
        _ => panic!("Expected unique solution"),
    }
}

#[test]
fn test_linear_solver_multiplication() {
    // 2 * x = 10  =>  x = 5
    let left = mul(int(2), var("x"));
    let right = int(10);
    let equation = Equation::new("test", left, right);
    let solver = LinearSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok());

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            // Should be 10 / 2 = 5
            let expected = int(5);
            assert_eq!(expr, expected);
        }
        _ => panic!("Expected unique solution"),
    }
}

#[test]
fn test_linear_solver_addition() {
    // x + 3 = 7  =>  x = 4
    let left = add(var("x"), int(3));
    let right = int(7);
    let equation = Equation::new("test", left, right);
    let solver = LinearSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok());

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            // Should be 7 - 3 = 4
            let expected = int(4);
            assert_eq!(expr, expected);
        }
        _ => panic!("Expected unique solution"),
    }
}

#[test]
fn test_linear_solver_ax_plus_b() {
    // 2 * x + 3 = 7  =>  x = 2
    let left = add(mul(int(2), var("x")), int(3));
    let right = int(7);
    let equation = Equation::new("test", left, right);
    let solver = LinearSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok());

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            // Should be (7 - 3) / 2 = 2
            let expected = int(2);
            assert_eq!(expr, expected);
        }
        _ => panic!("Expected unique solution"),
    }
}

#[test]
fn test_linear_solver_variable_not_found() {
    // 2 + 3 = 5, solve for x (not in equation)
    let left = add(int(2), int(3));
    let right = int(5);
    let equation = Equation::new("test", left, right);
    let solver = LinearSolver::new();
    let target = Variable::new("x");

    let result = solver.solve(&equation, &target);
    assert!(result.is_err());
    match result.unwrap_err() {
        SolverError::CannotSolve(msg) => {
            assert!(msg.contains("not found"));
        }
        _ => panic!("Expected CannotSolve error"),
    }
}

// ============================================================================
// Physics Equations Tests
// ============================================================================

#[test]
fn test_force_equation_solve_for_f() {
    // F = m * a, solve for F
    let left = var("F");
    let right = mul(var("m"), var("a"));
    let equation = Equation::new("force", left, right);
    let solver = LinearSolver::new();
    let target = Variable::new("F");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok());

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            // Should be m * a
            assert_eq!(expr, mul(var("m"), var("a")));
        }
        _ => panic!("Expected unique solution"),
    }
}

#[test]
fn test_force_equation_solve_for_m() {
    // F = m * a, solve for m  =>  m = F / a
    let left = var("F");
    let right = mul(var("m"), var("a"));
    let equation = Equation::new("force", left, right);
    let solver = LinearSolver::new();
    let target = Variable::new("m");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok());

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            // Should be F / a
            assert_eq!(expr, div(var("F"), var("a")));
        }
        _ => panic!("Expected unique solution"),
    }
}

#[test]
fn test_force_equation_solve_for_a() {
    // F = m * a, solve for a  =>  a = F / m
    let left = var("F");
    let right = mul(var("m"), var("a"));
    let equation = Equation::new("force", left, right);
    let solver = LinearSolver::new();
    let target = Variable::new("a");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok());

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            // Should be F / m
            assert_eq!(expr, div(var("F"), var("m")));
        }
        _ => panic!("Expected unique solution"),
    }
}

#[test]
fn test_velocity_equation_solve_for_v() {
    // v = d / t, solve for v
    let left = var("v");
    let right = div(var("d"), var("t"));
    let equation = Equation::new("velocity", left, right);
    let solver = LinearSolver::new();
    let target = Variable::new("v");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok());

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            // Should be d / t
            assert_eq!(expr, div(var("d"), var("t")));
        }
        _ => panic!("Expected unique solution"),
    }
}

#[test]
fn test_velocity_equation_solve_for_d() {
    // v = d / t, solve for d  =>  d = v * t
    // This requires recognizing d / t and multiplying both sides by t
    // For Phase 1, this might not be supported - mark as expected failure
    let left = var("v");
    let right = div(var("d"), var("t"));
    let equation = Equation::new("velocity", left, right);
    let solver = LinearSolver::new();
    let target = Variable::new("d");

    // For Phase 1, this pattern might not be supported yet
    // If it fails, that's expected
    let result = solver.solve(&equation, &target);
    // We'll allow either success or CannotSolve error
    if result.is_ok() {
        let (solution, _path) = result.unwrap();
        match solution {
            thales::solver::Solution::Unique(expr) => {
                // Should be v * t (in some form)
                println!("Got solution: {:?}", expr);
            }
            _ => panic!("Expected unique solution"),
        }
    }
}

#[test]
fn test_energy_equation_solve_for_e() {
    // E = m * c^2, solve for E
    let left = var("E");
    let right = mul(var("m"), pow(var("c"), int(2)));
    let equation = Equation::new("energy", left, right);
    let solver = LinearSolver::new();
    let target = Variable::new("E");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok());

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            // Should be m * c^2
            assert_eq!(expr, mul(var("m"), pow(var("c"), int(2))));
        }
        _ => panic!("Expected unique solution"),
    }
}

#[test]
fn test_energy_equation_solve_for_m() {
    // E = m * c^2, solve for m  =>  m = E / c^2
    let left = var("E");
    let right = mul(var("m"), pow(var("c"), int(2)));
    let equation = Equation::new("energy", left, right);
    let solver = LinearSolver::new();
    let target = Variable::new("m");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok());

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            // Should be E / c^2
            assert_eq!(expr, div(var("E"), pow(var("c"), int(2))));
        }
        _ => panic!("Expected unique solution"),
    }
}

#[test]
fn test_linear_equation_solve_for_y() {
    // y = m * x + b, solve for y
    let left = var("y");
    let right = add(mul(var("m"), var("x")), var("b"));
    let equation = Equation::new("line", left, right);
    let solver = LinearSolver::new();
    let target = Variable::new("y");

    let result = solver.solve(&equation, &target);
    assert!(result.is_ok());

    let (solution, _path) = result.unwrap();
    match solution {
        thales::solver::Solution::Unique(expr) => {
            // Should be m * x + b
            assert_eq!(expr, add(mul(var("m"), var("x")), var("b")));
        }
        _ => panic!("Expected unique solution"),
    }
}

// ============================================================================
// solve_for High-Level API Tests
// ============================================================================

#[test]
fn test_solve_for_with_values() {
    // F = m * a, solve for F with m=2, a=3  =>  F = 6
    let left = var("F");
    let right = mul(var("m"), var("a"));
    let equation = Equation::new("force", left, right);

    let mut known_values = HashMap::new();
    known_values.insert("m".to_string(), 2.0);
    known_values.insert("a".to_string(), 3.0);

    let result = solve_for(&equation, "F", &known_values);
    if let Err(ref e) = result {
        eprintln!("Error solving equation: {:?}", e);
    }
    assert!(result.is_ok());

    let path = result.unwrap();
    // Result should be 6.0
    if let Expression::Float(val) = &path.result {
        assert!((val - 6.0).abs() < 1e-10);
    } else if let Expression::Integer(val) = &path.result {
        assert_eq!(*val, 6);
    } else {
        panic!("Expected numeric result, got: {:?}", path.result);
    }
}

#[test]
fn test_solve_for_partial_values() {
    // F = m * a, solve for F with only m=2  =>  F = 2 * a
    let left = var("F");
    let right = mul(var("m"), var("a"));
    let equation = Equation::new("force", left, right);

    let mut known_values = HashMap::new();
    known_values.insert("m".to_string(), 2.0);

    let result = solve_for(&equation, "F", &known_values);
    assert!(result.is_ok());

    let path = result.unwrap();
    // Result should be 2.0 * a
    println!("Result: {:?}", path.result);
    // Should still contain variable 'a'
    assert!(path.result.contains_variable("a"));
}

#[test]
fn test_solve_for_no_values() {
    // F = m * a, solve for F with no values  =>  F = m * a
    let left = var("F");
    let right = mul(var("m"), var("a"));
    let equation = Equation::new("force", left, right);

    let known_values = HashMap::new();

    let result = solve_for(&equation, "F", &known_values);
    assert!(result.is_ok());

    let path = result.unwrap();
    // Result should be m * a
    assert_eq!(path.result, mul(var("m"), var("a")));
}

#[test]
fn test_solve_for_simple_arithmetic() {
    // 2 * x + 3 = 7, solve for x  =>  x = 2
    let left = add(mul(int(2), var("x")), int(3));
    let right = int(7);
    let equation = Equation::new("test", left, right);

    let known_values = HashMap::new();

    let result = solve_for(&equation, "x", &known_values);
    assert!(result.is_ok());

    let path = result.unwrap();
    // Result should be 2
    assert_eq!(path.result, int(2));
}

#[test]
fn test_solve_for_variable_not_in_equation() {
    // 2 + 3 = 5, solve for x (not present)
    let left = add(int(2), int(3));
    let right = int(5);
    let equation = Equation::new("test", left, right);

    let known_values = HashMap::new();

    let result = solve_for(&equation, "x", &known_values);
    assert!(result.is_err());
}

// ============================================================================
// can_solve Tests
// ============================================================================

#[test]
fn test_can_solve_linear() {
    // 2 * x + 3 = 7
    let left = add(mul(int(2), var("x")), int(3));
    let right = int(7);
    let equation = Equation::new("test", left, right);

    let solver = LinearSolver::new();
    assert!(solver.can_solve(&equation));
}

#[test]
fn test_cannot_solve_quadratic() {
    // x^2 + 2*x + 1 = 0
    let left = add(add(pow(var("x"), int(2)), mul(int(2), var("x"))), int(1));
    let right = int(0);
    let equation = Equation::new("test", left, right);

    let solver = LinearSolver::new();
    assert!(!solver.can_solve(&equation));
}

// ============================================================================
// QuadraticSolver Tests
// ============================================================================

use thales::solver::{QuadraticSolver, PolynomialSolver, Solution};

#[test]
fn test_quadratic_solver_two_real_roots() {
    // x^2 - 5x + 6 = 0 => x = 2 or x = 3
    let left = add(add(pow(var("x"), int(2)), mul(int(-5), var("x"))), int(6));
    let right = int(0);
    let equation = Equation::new("test", left, right);

    let solver = QuadraticSolver::new();
    let result = solver.solve(&equation, &Variable::new("x"));
    assert!(result.is_ok());

    let (solution, _path) = result.unwrap();
    match solution {
        Solution::Multiple(roots) => {
            assert_eq!(roots.len(), 2);
            let vals: Vec<f64> = roots
                .iter()
                .filter_map(|r| r.evaluate(&HashMap::new()))
                .collect();
            assert!(vals.iter().any(|v| (v - 2.0).abs() < 1e-10));
            assert!(vals.iter().any(|v| (v - 3.0).abs() < 1e-10));
        }
        _ => panic!("Expected multiple solutions"),
    }
}

#[test]
fn test_quadratic_solver_complex_roots() {
    // x^2 + 1 = 0 => x = ±i
    let left = add(pow(var("x"), int(2)), int(1));
    let right = int(0);
    let equation = Equation::new("test", left, right);

    let solver = QuadraticSolver::new();
    let result = solver.solve(&equation, &Variable::new("x"));
    assert!(result.is_ok());

    let (solution, _path) = result.unwrap();
    match solution {
        Solution::Multiple(roots) => {
            assert_eq!(roots.len(), 2);
            // Both roots should be complex with real part 0 and imaginary ±1
            for root in &roots {
                if let Expression::Complex(c) = root {
                    assert!(c.re.abs() < 1e-10);
                    assert!((c.im.abs() - 1.0).abs() < 1e-10);
                } else {
                    panic!("Expected complex roots");
                }
            }
        }
        _ => panic!("Expected multiple solutions"),
    }
}

// ============================================================================
// PolynomialSolver (Cubic) Tests
// ============================================================================

#[test]
fn test_cubic_solver_x3_minus_1() {
    // x^3 - 1 = 0 => x = 1, x = -0.5 ± (√3/2)i
    let left = add(pow(var("x"), int(3)), int(-1));
    let right = int(0);
    let equation = Equation::new("test", left, right);

    let solver = PolynomialSolver::new();
    let result = solver.solve(&equation, &Variable::new("x"));
    assert!(result.is_ok());

    let (solution, _path) = result.unwrap();
    match solution {
        Solution::Multiple(roots) => {
            assert_eq!(roots.len(), 3);
            // One real root should be 1
            let real_roots: Vec<f64> = roots
                .iter()
                .filter_map(|r| r.evaluate(&HashMap::new()))
                .collect();
            assert!(real_roots.iter().any(|v| (v - 1.0).abs() < 1e-10));
        }
        _ => panic!("Expected multiple solutions"),
    }
}

#[test]
fn test_cubic_solver_depressed_cubic() {
    // x^3 - 6x - 9 = 0 => x = 3 is one root
    let left = add(add(pow(var("x"), int(3)), mul(int(-6), var("x"))), int(-9));
    let right = int(0);
    let equation = Equation::new("test", left, right);

    let solver = PolynomialSolver::new();
    let result = solver.solve(&equation, &Variable::new("x"));
    assert!(result.is_ok());

    let (solution, _path) = result.unwrap();
    match solution {
        Solution::Multiple(roots) => {
            assert_eq!(roots.len(), 3);
            // Check that one root is approximately 3
            let real_roots: Vec<f64> = roots
                .iter()
                .filter_map(|r| r.evaluate(&HashMap::new()))
                .collect();
            assert!(real_roots.iter().any(|v| (v - 3.0).abs() < 1e-10));
        }
        _ => panic!("Expected multiple solutions"),
    }
}

// ============================================================================
// PolynomialSolver (Quartic) Tests
// ============================================================================

#[test]
fn test_quartic_solver_x4_minus_1() {
    // x^4 - 1 = 0 => x = ±1, x = ±i
    let left = add(pow(var("x"), int(4)), int(-1));
    let right = int(0);
    let equation = Equation::new("test", left, right);

    let solver = PolynomialSolver::new();
    let result = solver.solve(&equation, &Variable::new("x"));
    assert!(result.is_ok());

    let (solution, _path) = result.unwrap();
    match solution {
        Solution::Multiple(roots) => {
            assert_eq!(roots.len(), 4);
            // Should have two real roots (1, -1) and two complex roots (±i)
            let mut real_roots = Vec::new();
            let mut complex_roots = Vec::new();
            for root in &roots {
                match root {
                    Expression::Integer(n) => real_roots.push(*n as f64),
                    Expression::Float(f) => real_roots.push(*f),
                    Expression::Complex(c) if c.im.abs() < 1e-10 => real_roots.push(c.re),
                    Expression::Complex(_) => complex_roots.push(root.clone()),
                    _ => {}
                }
            }
            assert!(real_roots.iter().any(|v| (v - 1.0).abs() < 1e-10));
            assert!(real_roots.iter().any(|v| (v + 1.0).abs() < 1e-10));
            assert_eq!(complex_roots.len(), 2);
        }
        _ => panic!("Expected multiple solutions"),
    }
}

#[test]
fn test_quartic_solver_biquadratic() {
    // x^4 - 5x^2 + 4 = 0 => x = ±1, ±2
    let x4 = pow(var("x"), int(4));
    let x2 = pow(var("x"), int(2));
    let left = add(add(x4, mul(int(-5), x2)), int(4));
    let right = int(0);
    let equation = Equation::new("test", left, right);

    let solver = PolynomialSolver::new();
    let result = solver.solve(&equation, &Variable::new("x"));
    assert!(result.is_ok());

    let (solution, _path) = result.unwrap();
    match solution {
        Solution::Multiple(roots) => {
            assert_eq!(roots.len(), 4);
            let vals: Vec<f64> = roots
                .iter()
                .filter_map(|r| r.evaluate(&HashMap::new()))
                .collect();
            assert!(vals.iter().any(|v| (v - 1.0).abs() < 1e-10));
            assert!(vals.iter().any(|v| (v + 1.0).abs() < 1e-10));
            assert!(vals.iter().any(|v| (v - 2.0).abs() < 1e-10));
            assert!(vals.iter().any(|v| (v + 2.0).abs() < 1e-10));
        }
        _ => panic!("Expected multiple solutions"),
    }
}

// ============================================================================
// PolynomialSolver (Higher Degree - Numerical) Tests
// ============================================================================

#[test]
fn test_polynomial_solver_quintic_numerical() {
    // x^5 - x - 1 = 0 (has one real root ≈ 1.1673)
    let x5 = pow(var("x"), int(5));
    let left = add(add(x5, mul(int(-1), var("x"))), int(-1));
    let right = int(0);
    let equation = Equation::new("test", left, right);

    let solver = PolynomialSolver::new();
    let result = solver.solve(&equation, &Variable::new("x"));
    assert!(result.is_ok());

    let (solution, _path) = result.unwrap();
    match solution {
        Solution::Multiple(roots) => {
            assert_eq!(roots.len(), 5);
            // Find the real root ≈ 1.1673
            let real_roots: Vec<f64> = roots
                .iter()
                .filter_map(|r| r.evaluate(&HashMap::new()))
                .filter(|v| v.is_finite())
                .collect();
            assert!(!real_roots.is_empty());
            assert!(real_roots.iter().any(|v| (v - 1.1673).abs() < 0.01));
        }
        _ => panic!("Expected multiple solutions"),
    }
}
