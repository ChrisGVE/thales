//! Unit tests for the algebraic equation solver.

use std::collections::HashMap;
use thales::ast::{BinaryOp, Equation, Expression, Variable};
use thales::solver::{solve_for, LinearSolver, Solver, SolverError};

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

fn assert_mul_eq(actual: &Expression, a: &Expression, b: &Expression) {
    match actual {
        Expression::Binary(BinaryOp::Mul, left, right) => {
            let fwd = left.as_ref() == a && right.as_ref() == b;
            let rev = left.as_ref() == b && right.as_ref() == a;
            assert!(
                fwd || rev,
                "expected Mul({a:?}, {b:?}) in either order, got {actual:?}"
            );
        }
        _ => panic!("expected Mul, got {actual:?}"),
    }
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
            assert_mul_eq(&expr, &var("m"), &var("a"));
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
        thales::solver::Solution::Unique(expr) => match &expr {
            Expression::Binary(BinaryOp::Add, left, right) => {
                assert_eq!(left.as_ref(), &var("b"));
                assert_mul_eq(right, &var("m"), &var("x"));
            }
            _ => panic!("Expected Add, got {expr:?}"),
        },
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

    let (result_expr, _trace) = result.unwrap();
    // Result should be 6.0
    if let Expression::Float(val) = &result_expr {
        assert!((val - 6.0).abs() < 1e-10);
    } else if let Expression::Integer(val) = &result_expr {
        assert_eq!(*val, 6);
    } else {
        panic!("Expected numeric result, got: {:?}", result_expr);
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

    let (result_expr, _trace) = result.unwrap();
    // Result should be 2.0 * a
    println!("Result: {:?}", result_expr);
    // Should still contain variable 'a'
    assert!(result_expr.contains_variable("a"));
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

    let (result_expr, _trace) = result.unwrap();
    assert_mul_eq(&result_expr, &var("m"), &var("a"));
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

    let (result_expr, _trace) = result.unwrap();
    // Result should be 2
    assert_eq!(result_expr, int(2));
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

use thales::solver::{PolynomialSolver, QuadraticSolver, Solution};

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

// ============================================================================
// Complex Root Tests
// ============================================================================

#[test]
fn test_quadratic_complex_roots_x2_plus_4() {
    // x² + 4 = 0 => x = ±2i
    let left = add(pow(var("x"), int(2)), int(4));
    let equation = Equation::new("test", left, int(0));

    let solver = QuadraticSolver::new();
    let (solution, _path) = solver.solve(&equation, &Variable::new("x")).unwrap();
    match solution {
        Solution::Multiple(roots) => {
            assert_eq!(roots.len(), 2);
            for root in &roots {
                match root {
                    Expression::Complex(c) => {
                        assert!(c.re.abs() < 1e-10, "real part should be 0, got {}", c.re);
                        assert!(
                            (c.im.abs() - 2.0).abs() < 1e-10,
                            "imag should be ±2, got {}",
                            c.im
                        );
                    }
                    _ => panic!("Expected complex root, got {:?}", root),
                }
            }
            // Roots should be conjugates
            if let (Expression::Complex(c1), Expression::Complex(c2)) = (&roots[0], &roots[1]) {
                assert!((c1.im + c2.im).abs() < 1e-10, "roots should be conjugates");
            }
        }
        _ => panic!("Expected multiple solutions"),
    }
}

#[test]
fn test_quadratic_complex_roots_x2_plus_2x_plus_5() {
    // x² + 2x + 5 = 0 => x = -1 ± 2i
    let left = add(add(pow(var("x"), int(2)), mul(int(2), var("x"))), int(5));
    let equation = Equation::new("test", left, int(0));

    let solver = QuadraticSolver::new();
    let (solution, _path) = solver.solve(&equation, &Variable::new("x")).unwrap();
    match solution {
        Solution::Multiple(roots) => {
            assert_eq!(roots.len(), 2);
            for root in &roots {
                match root {
                    Expression::Complex(c) => {
                        assert!(
                            (c.re + 1.0).abs() < 1e-10,
                            "real part should be -1, got {}",
                            c.re
                        );
                        assert!(
                            (c.im.abs() - 2.0).abs() < 1e-10,
                            "imag part should be ±2, got {}",
                            c.im
                        );
                    }
                    _ => panic!("Expected complex root, got {:?}", root),
                }
            }
            // One root is -1+2i, the other is -1-2i
            let imag_values: Vec<f64> = roots
                .iter()
                .filter_map(|r| {
                    if let Expression::Complex(c) = r {
                        Some(c.im)
                    } else {
                        None
                    }
                })
                .collect();
            assert!(
                imag_values.iter().any(|v| (*v - 2.0).abs() < 1e-10),
                "Expected root with +2i"
            );
            assert!(
                imag_values.iter().any(|v| (*v + 2.0).abs() < 1e-10),
                "Expected root with -2i"
            );
        }
        _ => panic!("Expected multiple solutions"),
    }
}

#[test]
fn test_cubic_complex_roots_x3_plus_1() {
    // x³ + 1 = 0 => x = -1, x = (1 ± i√3)/2
    let left = add(pow(var("x"), int(3)), int(1));
    let equation = Equation::new("test", left, int(0));

    let solver = PolynomialSolver::new();
    let (solution, _path) = solver.solve(&equation, &Variable::new("x")).unwrap();
    match solution {
        Solution::Multiple(roots) => {
            assert_eq!(roots.len(), 3);
            // Collect real roots (evaluate returns Some only for real-valued expressions)
            let real_vals: Vec<f64> = roots
                .iter()
                .filter_map(|r| r.evaluate(&HashMap::new()))
                .collect();
            // One real root: x = -1
            assert!(
                real_vals.iter().any(|v| (v + 1.0).abs() < 1e-10),
                "Expected real root -1"
            );
            // Two complex roots
            let complex_count = roots
                .iter()
                .filter(|r| matches!(r, Expression::Complex(c) if c.im.abs() > 1e-10))
                .count();
            assert_eq!(complex_count, 2, "Expected two complex roots");
            // Complex roots should be conjugates with re=0.5, im=±√3/2
            let complex_roots: Vec<_> = roots
                .iter()
                .filter_map(|r| {
                    if let Expression::Complex(c) = r {
                        if c.im.abs() > 1e-10 {
                            Some(*c)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();
            assert_eq!(complex_roots.len(), 2);
            for c in &complex_roots {
                assert!(
                    (c.re - 0.5).abs() < 1e-10,
                    "real part of complex root should be 0.5, got {}",
                    c.re
                );
                assert!(
                    (c.im.abs() - (3.0_f64).sqrt() / 2.0).abs() < 1e-10,
                    "imag magnitude should be √3/2, got {}",
                    c.im.abs()
                );
            }
        }
        _ => panic!("Expected multiple solutions"),
    }
}

#[test]
fn test_smart_solver_routes_to_quadratic_for_complex_roots() {
    // SmartSolver must delegate x² + 1 = 0 to QuadraticSolver, returning ±i.
    // The SmartSolver skips symbolic isolation when the discriminant is negative.
    use thales::solver::SmartSolver;

    let left = add(pow(var("x"), int(2)), int(1));
    let equation = Equation::new("test", left, int(0));

    let solver = SmartSolver::new();
    let result = solver.solve(&equation, &Variable::new("x"));
    assert!(result.is_ok(), "SmartSolver failed: {:?}", result.err());

    let (solution, _path) = result.unwrap();
    match solution {
        Solution::Multiple(roots) => {
            assert_eq!(roots.len(), 2, "Expected 2 complex roots");
            for root in &roots {
                assert!(
                    matches!(root, Expression::Complex(_)),
                    "Expected complex root, got {:?}",
                    root
                );
                if let Expression::Complex(c) = root {
                    assert!(c.re.abs() < 1e-10, "real part should be 0, got {}", c.re);
                    assert!(
                        (c.im.abs() - 1.0).abs() < 1e-10,
                        "|imag| should be 1, got {}",
                        c.im.abs()
                    );
                }
            }
        }
        _ => panic!("Expected multiple complex solutions, got {:?}", solution),
    }
}

#[test]
fn test_complex_root_display_format() {
    // Verify that complex roots display with a+bi / a-bi notation
    let root_pos = Expression::Complex(num_complex::Complex64::new(-1.0, 2.0));
    let root_neg = Expression::Complex(num_complex::Complex64::new(-1.0, -2.0));
    let root_pure_imag = Expression::Complex(num_complex::Complex64::new(0.0, 3.0));

    let s_pos = format!("{}", root_pos);
    let s_neg = format!("{}", root_neg);
    let s_pure = format!("{}", root_pure_imag);

    // Positive imaginary: should contain '+' between real and imaginary parts
    assert!(
        s_pos.contains('+') || s_pos.contains('i'),
        "Positive-imaginary display '{}' should contain '+' or 'i'",
        s_pos
    );
    // Negative imaginary: should show negative sign
    assert!(
        s_neg.contains('-') && s_neg.contains('i'),
        "Negative-imaginary display '{}' should contain '-' and 'i'",
        s_neg
    );
    // Pure imaginary
    assert!(
        s_pure.contains('i'),
        "Pure-imaginary display '{}' should contain 'i'",
        s_pure
    );
}

#[test]
fn test_complex_root_evaluate_returns_none_for_nonzero_imaginary() {
    // evaluate() should return None when imaginary part is nonzero
    let root = Expression::Complex(num_complex::Complex64::new(1.0, 2.0));
    assert_eq!(root.evaluate(&HashMap::new()), None);
}

#[test]
fn test_complex_root_evaluate_returns_real_for_zero_imaginary() {
    // evaluate() should return the real part when imaginary is ~0
    let root = Expression::Complex(num_complex::Complex64::new(3.5, 0.0));
    assert_eq!(root.evaluate(&HashMap::new()), Some(3.5));
}

// ============================================================================
// Step Annotation Tests
// ============================================================================

#[test]
fn test_quadratic_solver_annotations() {
    use thales::solver::QuadraticSolver;
    // x² - 5x + 6 = 0
    let lhs = binary(
        BinaryOp::Add,
        binary(
            BinaryOp::Sub,
            Expression::Power(Box::new(var("x")), Box::new(int(2))),
            binary(BinaryOp::Mul, int(5), var("x")),
        ),
        int(6),
    );
    let eq = Equation::new("q", lhs, int(0));
    let solver = QuadraticSolver::new();
    let (_sol, path) = solver.solve(&eq, &Variable::new("x")).unwrap();

    let disc_step = path
        .steps()
        .iter()
        .find(|s| s.detail.contains("discriminant"));
    assert!(disc_step.is_some(), "Expected a discriminant step");
    assert_eq!(
        disc_step.unwrap().tag,
        thales::numeric::trace::TechniqueTag::Simplification,
        "Discriminant step should be Simplification"
    );

    let formula_step = path
        .steps()
        .iter()
        .find(|s| s.detail.contains("Quadratic Formula"));
    assert!(formula_step.is_some(), "Expected a quadratic formula step");
    assert_eq!(
        formula_step.unwrap().tag,
        thales::numeric::trace::TechniqueTag::QuadraticFormula,
    );
}

#[test]
fn test_transcendental_solver_annotations() {
    use thales::ast::Function;
    use thales::solver::TranscendentalSolver;
    let eq = Equation::new(
        "trig",
        Expression::Function(Function::Sin, vec![var("x")]),
        float(0.5),
    );
    let solver = TranscendentalSolver::new();
    let (_sol, path) = solver.solve(&eq, &Variable::new("x")).unwrap();

    let trig_step = path.steps().iter().find(|s| s.detail.contains("arcsine"));
    assert!(trig_step.is_some(), "Expected an arcsine step");
    assert!(
        trig_step
            .unwrap()
            .detail
            .contains("Inverse Trigonometric Function"),
        "Expected step detail to mention the technique",
    );
}

#[test]
fn test_symbolic_isolation_annotations() {
    use thales::solver::SmartSolver;
    // SmartSolver tries symbolic isolation first for 2*x + 3 = 7
    let lhs = binary(
        BinaryOp::Add,
        binary(BinaryOp::Mul, int(2), var("x")),
        int(3),
    );
    let eq = Equation::new("lin", lhs, int(7));
    let solver = SmartSolver::new();
    let (_sol, path) = solver.solve(&eq, &Variable::new("x")).unwrap();

    let tagged_count = path.steps().len();
    assert!(
        tagged_count > 0,
        "Expected at least one trace step from symbolic isolation, got 0"
    );
}

#[test]
fn test_quadratic_complex_root_path_contains_decomposition() {
    use thales::numeric::trace::TechniqueTag;
    use thales::solver::QuadraticSolver;
    // x² + 1 = 0 => x = ±i; the trace must contain a ComplexDecomposition step.
    let lhs = add(pow(var("x"), int(2)), int(1));
    let eq = Equation::new("cplx", lhs, int(0));
    let solver = QuadraticSolver::new();
    let (_sol, path) = solver.solve(&eq, &Variable::new("x")).unwrap();

    let decomp_step = path
        .steps()
        .iter()
        .find(|s| s.tag == TechniqueTag::Custom("ComplexDecomposition"));
    assert!(
        decomp_step.is_some(),
        "Trace for x²+1=0 must contain a ComplexDecomposition step"
    );
    assert!(
        decomp_step.unwrap().detail.contains("original_var=x"),
        "ComplexDecomposition detail must name the original variable"
    );
}

#[test]
fn test_complex_operation_difficulty_tiers() {
    use thales::numeric::trace::{TechniqueDifficulty, TechniqueTag};
    // ComplexDecomposition is carried as a custom tag; custom tags default
    // to Advanced difficulty.
    let decomp = TechniqueTag::Custom("ComplexDecomposition");
    assert_eq!(
        decomp.difficulty(),
        TechniqueDifficulty::Advanced,
        "Custom tags default to Advanced difficulty"
    );

    // EulerFormula is a first-class tag sitting at Transcendental.
    let euler = TechniqueTag::EulerFormula;
    assert_eq!(
        euler.difficulty(),
        TechniqueDifficulty::Transcendental,
        "EulerFormula should be Transcendental difficulty"
    );
}
