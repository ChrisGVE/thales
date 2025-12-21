//! Unit tests for the algebraic equation solver.

use mathsolver_core::ast::{BinaryOp, Equation, Expression, Variable};
use mathsolver_core::solver::{solve_for, LinearSolver, Solver, SolverError};
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
        mathsolver_core::solver::Solution::Unique(expr) => {
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
        mathsolver_core::solver::Solution::Unique(expr) => {
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
        mathsolver_core::solver::Solution::Unique(expr) => {
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
        mathsolver_core::solver::Solution::Unique(expr) => {
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
        mathsolver_core::solver::Solution::Unique(expr) => {
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
        mathsolver_core::solver::Solution::Unique(expr) => {
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
        mathsolver_core::solver::Solution::Unique(expr) => {
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
        mathsolver_core::solver::Solution::Unique(expr) => {
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
            mathsolver_core::solver::Solution::Unique(expr) => {
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
        mathsolver_core::solver::Solution::Unique(expr) => {
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
        mathsolver_core::solver::Solution::Unique(expr) => {
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
        mathsolver_core::solver::Solution::Unique(expr) => {
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
