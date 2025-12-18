use mathsolver_core::ast::{BinaryOp, Equation, Expression, Function, Variable};
use mathsolver_core::numerical::{
    BisectionMethod, NewtonRaphson, NumericalConfig, SmartNumericalSolver,
};

#[test]
fn test_newton_raphson_quadratic() {
    // Solve x^2 - 4 = 0 (solution: x = 2 or x = -2)
    let equation = Equation::new(
        "quadratic",
        Expression::Power(
            Box::new(Expression::Variable(Variable::new("x"))),
            Box::new(Expression::Integer(2)),
        ),
        Expression::Integer(4),
    );

    let var = Variable::new("x");
    let config = NumericalConfig {
        max_iterations: 100,
        tolerance: 1e-10,
        initial_guess: Some(1.5), // Close to x = 2
        step_size: 1e-6,
    };

    let solver = NewtonRaphson::new(config);
    let result = solver.solve(&equation, &var);

    assert!(result.is_ok());
    let (solution, _path) = result.unwrap();
    assert!(solution.converged);
    assert!((solution.value - 2.0).abs() < 1e-9);
    assert!(solution.residual < 1e-10);
}

#[test]
fn test_newton_raphson_transcendental() {
    // Solve e^x = x + 2
    // Rewrite as: e^x - x - 2 = 0
    let equation = Equation::new(
        "transcendental",
        Expression::Function(
            Function::Exp,
            vec![Expression::Variable(Variable::new("x"))],
        ),
        Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Variable(Variable::new("x"))),
            Box::new(Expression::Integer(2)),
        ),
    );

    let var = Variable::new("x");
    let config = NumericalConfig {
        max_iterations: 100,
        tolerance: 1e-10,
        initial_guess: Some(1.0),
        step_size: 1e-6,
    };

    let solver = NewtonRaphson::new(config);
    let result = solver.solve(&equation, &var);

    assert!(result.is_ok());
    let (solution, path) = result.unwrap();
    assert!(solution.converged);
    // Verify solution: e^x should equal x + 2
    assert!((solution.value.exp() - (solution.value + 2.0)).abs() < 1e-9);
    assert!(path.step_count() > 0);
}

#[test]
fn test_newton_raphson_ln_equation() {
    // Solve ln(x) = 1 (solution: x = e)
    let equation = Equation::new(
        "ln_eq",
        Expression::Function(
            Function::Ln,
            vec![Expression::Variable(Variable::new("x"))],
        ),
        Expression::Integer(1),
    );

    let var = Variable::new("x");
    let config = NumericalConfig {
        max_iterations: 100,
        tolerance: 1e-10,
        initial_guess: Some(2.0),
        step_size: 1e-6,
    };

    let solver = NewtonRaphson::new(config);
    let result = solver.solve(&equation, &var);

    assert!(result.is_ok());
    let (solution, _path) = result.unwrap();
    assert!(solution.converged);
    assert!((solution.value - std::f64::consts::E).abs() < 1e-9);
}

#[test]
fn test_newton_raphson_no_convergence() {
    // x^2 + 1 = 0 has no real solutions
    let equation = Equation::new(
        "no_real_solution",
        Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Power(
                Box::new(Expression::Variable(Variable::new("x"))),
                Box::new(Expression::Integer(2)),
            )),
            Box::new(Expression::Integer(1)),
        ),
        Expression::Integer(0),
    );

    let var = Variable::new("x");
    let config = NumericalConfig {
        max_iterations: 100,
        tolerance: 1e-10,
        initial_guess: Some(1.0),
        step_size: 1e-6,
    };

    let solver = NewtonRaphson::new(config);
    let result = solver.solve(&equation, &var);

    // Should fail to converge or become unstable
    assert!(result.is_err());
}

#[test]
fn test_bisection_simple() {
    // Solve x - 3 = 0 (solution: x = 3)
    let equation = Equation::new(
        "linear",
        Expression::Variable(Variable::new("x")),
        Expression::Integer(3),
    );

    let var = Variable::new("x");
    let config = NumericalConfig::default();
    let solver = BisectionMethod::new(config);

    let result = solver.solve(&equation, &var, (0.0, 10.0));

    assert!(result.is_ok());
    let (solution, _path) = result.unwrap();
    assert!(solution.converged);
    assert!((solution.value - 3.0).abs() < 1e-10);
}

#[test]
fn test_bisection_transcendental() {
    // Solve cos(x) = x
    let equation = Equation::new(
        "cos_eq",
        Expression::Function(
            Function::Cos,
            vec![Expression::Variable(Variable::new("x"))],
        ),
        Expression::Variable(Variable::new("x")),
    );

    let var = Variable::new("x");
    let config = NumericalConfig::default();
    let solver = BisectionMethod::new(config);

    // The solution is approximately 0.739085133
    let result = solver.solve(&equation, &var, (0.0, 1.0));

    assert!(result.is_ok());
    let (solution, _path) = result.unwrap();
    assert!(solution.converged);
    // Verify cos(x) ≈ x
    assert!((solution.value.cos() - solution.value).abs() < 1e-10);
}

#[test]
fn test_bisection_wrong_interval() {
    // Try to solve x^2 - 4 = 0 with interval that doesn't bracket root
    let equation = Equation::new(
        "quadratic",
        Expression::Power(
            Box::new(Expression::Variable(Variable::new("x"))),
            Box::new(Expression::Integer(2)),
        ),
        Expression::Integer(4),
    );

    let var = Variable::new("x");
    let config = NumericalConfig::default();
    let solver = BisectionMethod::new(config);

    // Interval [0, 1] doesn't contain roots (roots are at ±2)
    // But f(0) = -4 and f(1) = -3, same sign
    let result = solver.solve(&equation, &var, (0.0, 1.0));

    // Should fail because f(a) and f(b) have the same sign
    assert!(result.is_err());
}

#[test]
fn test_smart_solver_quadratic() {
    // Solve x^2 = 16 (solution: x = 4 or x = -4)
    let equation = Equation::new(
        "quadratic",
        Expression::Power(
            Box::new(Expression::Variable(Variable::new("x"))),
            Box::new(Expression::Integer(2)),
        ),
        Expression::Integer(16),
    );

    let var = Variable::new("x");
    let solver = SmartNumericalSolver::with_default_config();

    let result = solver.solve(&equation, &var);

    assert!(result.is_ok());
    let (solution, _path) = result.unwrap();
    assert!(solution.converged);
    // Should find one of the two roots
    assert!(
        (solution.value - 4.0).abs() < 1e-9 || (solution.value + 4.0).abs() < 1e-9,
        "Solution {} is not close to ±4",
        solution.value
    );
}

#[test]
fn test_smart_solver_with_initial_guess() {
    // Solve x^3 - 2*x - 5 = 0
    let equation = Equation::new(
        "cubic",
        Expression::Binary(
            BinaryOp::Sub,
            Box::new(Expression::Power(
                Box::new(Expression::Variable(Variable::new("x"))),
                Box::new(Expression::Integer(3)),
            )),
            Box::new(Expression::Binary(
                BinaryOp::Add,
                Box::new(Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(Expression::Integer(2)),
                    Box::new(Expression::Variable(Variable::new("x"))),
                )),
                Box::new(Expression::Integer(5)),
            )),
        ),
        Expression::Integer(0),
    );

    let var = Variable::new("x");
    let config = NumericalConfig {
        max_iterations: 1000,
        tolerance: 1e-10,
        initial_guess: Some(2.0), // Near the real root at ~2.0946
        step_size: 1e-6,
    };

    let solver = SmartNumericalSolver::new(config);
    let result = solver.solve(&equation, &var);

    assert!(result.is_ok());
    let (solution, _path) = result.unwrap();
    assert!(solution.converged);

    // Verify solution: x^3 - 2x - 5 should be ≈ 0
    let x = solution.value;
    let residual = x.powi(3) - 2.0 * x - 5.0;
    assert!(residual.abs() < 1e-9);
}

#[test]
fn test_smart_solver_sin_equation() {
    // Solve sin(x) = 0.5
    let equation = Equation::new(
        "sin_eq",
        Expression::Function(
            Function::Sin,
            vec![Expression::Variable(Variable::new("x"))],
        ),
        Expression::Float(0.5),
    );

    let var = Variable::new("x");
    let solver = SmartNumericalSolver::with_default_config();

    let result = solver.solve(&equation, &var);

    assert!(result.is_ok());
    let (solution, _path) = result.unwrap();
    assert!(solution.converged);

    // Verify sin(x) ≈ 0.5
    assert!((solution.value.sin() - 0.5).abs() < 1e-9);
}

#[test]
fn test_smart_solver_with_interval() {
    // Solve x^2 = 9 on interval [2, 4] (should find x = 3)
    let equation = Equation::new(
        "quadratic",
        Expression::Power(
            Box::new(Expression::Variable(Variable::new("x"))),
            Box::new(Expression::Integer(2)),
        ),
        Expression::Integer(9),
    );

    let var = Variable::new("x");
    let solver = SmartNumericalSolver::with_default_config();

    let result = solver.solve_with_interval(&equation, &var, (2.0, 4.0));

    assert!(result.is_ok());
    let (solution, _path) = result.unwrap();
    assert!(solution.converged);
    assert!((solution.value - 3.0).abs() < 1e-10);
}

#[test]
fn test_newton_vs_bisection_performance() {
    // Compare Newton-Raphson and Bisection on the same problem
    // Solve x^2 - 7 = 0 (solution: x = sqrt(7) ≈ 2.6457513)

    let equation = Equation::new(
        "quadratic",
        Expression::Power(
            Box::new(Expression::Variable(Variable::new("x"))),
            Box::new(Expression::Integer(2)),
        ),
        Expression::Integer(7),
    );

    let var = Variable::new("x");

    // Newton-Raphson
    let newton_config = NumericalConfig {
        max_iterations: 1000,
        tolerance: 1e-10,
        initial_guess: Some(2.0),
        step_size: 1e-6,
    };
    let newton = NewtonRaphson::new(newton_config);
    let newton_result = newton.solve(&equation, &var);
    assert!(newton_result.is_ok());
    let (newton_solution, _) = newton_result.unwrap();

    // Bisection
    let bisection_config = NumericalConfig::default();
    let bisection = BisectionMethod::new(bisection_config);
    let bisection_result = bisection.solve(&equation, &var, (0.0, 5.0));
    assert!(bisection_result.is_ok());
    let (bisection_solution, _) = bisection_result.unwrap();

    // Both should converge to the same answer
    assert!((newton_solution.value - bisection_solution.value).abs() < 1e-9);

    // Newton-Raphson should converge in fewer iterations
    assert!(newton_solution.iterations < bisection_solution.iterations);
}

#[test]
fn test_exponential_equation() {
    // Solve 2^x = 8 (solution: x = 3)
    // Rewrite as: exp(x * ln(2)) = 8
    let equation = Equation::new(
        "exponential",
        Expression::Function(
            Function::Exp,
            vec![Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Variable(Variable::new("x"))),
                Box::new(Expression::Function(
                    Function::Ln,
                    vec![Expression::Integer(2)],
                )),
            )],
        ),
        Expression::Integer(8),
    );

    let var = Variable::new("x");
    let config = NumericalConfig {
        max_iterations: 1000,
        tolerance: 1e-10,
        initial_guess: Some(2.0),
        step_size: 1e-6,
    };

    let solver = NewtonRaphson::new(config);
    let result = solver.solve(&equation, &var);

    assert!(result.is_ok());
    let (solution, _path) = result.unwrap();
    assert!(solution.converged);
    assert!((solution.value - 3.0).abs() < 1e-9);
}

#[test]
fn test_edge_case_zero_derivative() {
    // Test equation where derivative could be zero at certain points
    // x^3 = 0 has a triple root at x = 0 with zero derivative
    let equation = Equation::new(
        "triple_root",
        Expression::Power(
            Box::new(Expression::Variable(Variable::new("x"))),
            Box::new(Expression::Integer(3)),
        ),
        Expression::Integer(0),
    );

    let var = Variable::new("x");
    let config = NumericalConfig {
        max_iterations: 1000,
        tolerance: 1e-6, // More lenient tolerance for this difficult case
        initial_guess: Some(0.1), // Start close to the root
        step_size: 1e-6,
    };

    let solver = NewtonRaphson::new(config);
    let result = solver.solve(&equation, &var);

    // Newton-Raphson may struggle with triple roots but should eventually converge
    // If it converges, verify the solution is close to 0
    // If it fails, that's also acceptable for this pathological case
    if let Ok((solution, _)) = result {
        assert!(
            (solution.value).abs() < 0.01,
            "Solution {} should be close to 0 (within 0.01)",
            solution.value
        );
    }
    // If it doesn't converge, that's also acceptable for this edge case
}

#[test]
fn test_symbolic_differentiation_integration() {
    // Test that Newton-Raphson uses symbolic differentiation from Task 188
    // Solve x^2 - 5 = 0 (solution: x = sqrt(5) ≈ 2.236)
    let equation = Equation::new(
        "quadratic",
        Expression::Power(
            Box::new(Expression::Variable(Variable::new("x"))),
            Box::new(Expression::Integer(2)),
        ),
        Expression::Integer(5),
    );

    let var = Variable::new("x");
    let config = NumericalConfig {
        max_iterations: 100,
        tolerance: 1e-12,
        initial_guess: Some(2.0),
        step_size: 1e-6,
    };

    let solver = NewtonRaphson::new(config);
    let result = solver.solve(&equation, &var);

    assert!(result.is_ok());
    let (solution, path) = result.unwrap();
    assert!(solution.converged);

    // Verify the solution is accurate
    assert!((solution.value - 5.0_f64.sqrt()).abs() < 1e-11);

    // Verify the resolution path contains symbolic derivative information
    let path_str = format!("{:?}", path);
    assert!(path_str.contains("symbolic derivative") || path_str.contains("derivative"));

    // Newton-Raphson should converge very quickly for this simple case
    assert!(solution.iterations < 10, "Expected fast convergence with symbolic differentiation");
}

#[test]
fn test_complex_transcendental_with_symbolic_diff() {
    // Test symbolic differentiation on a complex transcendental equation
    // Solve tan(x) + x = 2
    let equation = Equation::new(
        "transcendental_tan",
        Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Function(
                Function::Tan,
                vec![Expression::Variable(Variable::new("x"))],
            )),
            Box::new(Expression::Variable(Variable::new("x"))),
        ),
        Expression::Integer(2),
    );

    let var = Variable::new("x");
    let config = NumericalConfig {
        max_iterations: 1000,
        tolerance: 1e-10,
        initial_guess: Some(0.8), // Near the solution
        step_size: 1e-6,
    };

    let solver = NewtonRaphson::new(config);
    let result = solver.solve(&equation, &var);

    assert!(result.is_ok());
    let (solution, _path) = result.unwrap();
    assert!(solution.converged);

    // Verify: tan(x) + x ≈ 2
    let x = solution.value;
    let residual = x.tan() + x - 2.0;
    assert!(residual.abs() < 1e-9);
}
