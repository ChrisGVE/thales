//! First-Order Ordinary Differential Equation Solver
//!
//! This module provides functionality for solving first-order ODEs including:
//! - Separable equations: dy/dx = g(x) * h(y)
//! - First-order linear equations: dy/dx + P(x)*y = Q(x)
//! - Initial value problems (IVP)
//!
//! # Examples
//!
//! ```rust
//! use thales::ode::{FirstOrderODE, solve_separable, solve_linear};
//! use thales::ast::{Expression, Variable};
//!
//! // Create an ODE: dy/dx = x*y (separable)
//! let x = Expression::Variable(Variable::new("x"));
//! let y = Expression::Variable(Variable::new("y"));
//! let rhs = Expression::Binary(
//!     thales::ast::BinaryOp::Mul,
//!     Box::new(x),
//!     Box::new(y),
//! );
//! let ode = FirstOrderODE::new("y", "x", rhs);
//! ```

pub mod builder;
mod first_order;
pub mod non_homogeneous;
mod second_order;
mod types;
pub mod verify;

pub use builder::{first_order_ode, second_order_homogeneous, ODEBuilder};
pub use first_order::{solve_ivp, solve_linear, solve_separable};
pub use non_homogeneous::{
    identify_forcing_function, particular_solution_undetermined, ForcingType,
};
pub use second_order::{
    solve_characteristic_equation, solve_second_order_homogeneous, solve_second_order_ivp,
    CharacteristicRoots, RootType, SecondOrderODE, SecondOrderSolution,
};
pub use types::{FirstOrderODE, ODEError, ODESolution};

#[cfg(test)]
mod tests {
    use super::first_order::*;
    use super::*;
    use crate::ast::{BinaryOp, Expression, Function, UnaryOp, Variable};

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn mul(left: Expression, right: Expression) -> Expression {
        Expression::Binary(BinaryOp::Mul, Box::new(left), Box::new(right))
    }

    fn add(left: Expression, right: Expression) -> Expression {
        Expression::Binary(BinaryOp::Add, Box::new(left), Box::new(right))
    }

    fn div(left: Expression, right: Expression) -> Expression {
        Expression::Binary(BinaryOp::Div, Box::new(left), Box::new(right))
    }

    fn neg(expr: Expression) -> Expression {
        Expression::Unary(UnaryOp::Neg, Box::new(expr))
    }

    #[test]
    fn test_try_separate_simple_product() {
        // dy/dx = x * y
        let expr = mul(var("x"), var("y"));
        let result = try_separate(&expr, "x", "y");
        assert!(result.is_some());
        let (g_x, h_y) = result.unwrap();
        assert!(matches!(g_x, Expression::Variable(v) if v.name == "x"));
        assert!(matches!(h_y, Expression::Variable(v) if v.name == "y"));
    }

    #[test]
    fn test_try_separate_only_x() {
        // dy/dx = x^2
        let x = var("x");
        let expr = Expression::Power(Box::new(x), Box::new(int(2)));
        let result = try_separate(&expr, "x", "y");
        assert!(result.is_some());
        let (g_x, h_y) = result.unwrap();
        assert!(matches!(g_x, Expression::Power(_, _)));
        assert!(matches!(h_y, Expression::Integer(1)));
    }

    #[test]
    fn test_try_separate_only_y() {
        // dy/dx = y^2
        let y = var("y");
        let expr = Expression::Power(Box::new(y), Box::new(int(2)));
        let result = try_separate(&expr, "x", "y");
        assert!(result.is_some());
        let (g_x, h_y) = result.unwrap();
        assert!(matches!(g_x, Expression::Integer(1)));
        assert!(matches!(h_y, Expression::Power(_, _)));
    }

    #[test]
    fn test_try_separate_constant() {
        // dy/dx = 5
        let expr = int(5);
        let result = try_separate(&expr, "x", "y");
        assert!(result.is_some());
        let (g_x, h_y) = result.unwrap();
        assert!(matches!(g_x, Expression::Integer(5)));
        assert!(matches!(h_y, Expression::Integer(1)));
    }

    #[test]
    fn test_is_separable() {
        // dy/dx = x * y is separable
        let ode = FirstOrderODE::new("y", "x", mul(var("x"), var("y")));
        assert!(ode.is_separable());

        // dy/dx = x + y is NOT separable
        let ode2 = FirstOrderODE::new("y", "x", add(var("x"), var("y")));
        assert!(!ode2.is_separable());
    }

    #[test]
    fn test_is_linear() {
        // dy/dx = -y + x is linear (P(x) = 1, Q(x) = x)
        let ode = FirstOrderODE::new("y", "x", add(neg(var("y")), var("x")));
        assert!(ode.is_linear());

        // dy/dx = y^2 is NOT linear
        let y = var("y");
        let ode2 = FirstOrderODE::new("y", "x", Expression::Power(Box::new(y), Box::new(int(2))));
        assert!(!ode2.is_linear());
    }

    #[test]
    fn test_extract_linear_coefficients() {
        // dy/dx = -2*y + 3*x
        // Standard form: dy/dx + 2*y = 3*x
        // So P(x) = 2, Q(x) = 3*x
        let rhs = add(mul(int(-2), var("y")), mul(int(3), var("x")));
        let result = extract_linear_coefficients(&rhs, "x", "y");
        assert!(result.is_some());
    }

    #[test]
    fn test_solve_separable_simple() {
        // dy/dx = y
        // Solution: y = C * e^x
        let ode = FirstOrderODE::new("y", "x", var("y"));
        let result = solve_separable(&ode);
        assert!(result.is_ok());

        let solution = result.unwrap();
        assert_eq!(solution.method, "Separation of variables");
        assert!(!solution.steps.is_empty());
    }

    #[test]
    fn test_solve_separable_xy() {
        // dy/dx = x*y
        // Separating: (1/y) dy = x dx
        // Integrating: ln|y| = x^2/2 + C
        // Solution: y = A * e^(x^2/2) where A = e^C
        let ode = FirstOrderODE::new("y", "x", mul(var("x"), var("y")));
        let result = solve_separable(&ode);
        assert!(result.is_ok());
    }

    #[test]
    fn test_solve_linear_simple() {
        // dy/dx + y = 0
        // This is dy/dx = -y, so rhs = -y
        // P(x) = 1, Q(x) = 0
        // μ = e^x
        // Solution: y = C * e^(-x)
        let ode = FirstOrderODE::new("y", "x", neg(var("y")));
        let result = solve_linear(&ode);
        assert!(result.is_ok());

        let solution = result.unwrap();
        assert_eq!(solution.method, "Integrating factor");
    }

    #[test]
    fn test_solve_ivp() {
        // dy/dx = y, y(0) = 1
        // General solution: y = C * e^x
        // With y(0) = 1: C = 1
        // Particular solution: y = e^x
        let ode = FirstOrderODE::new("y", "x", var("y"));
        let sol = solve_ivp(&ode, &int(0), &int(1)).expect("IVP must solve");

        // Decision 2b: y-free + IC
        verify::assert_y_free(&sol.general_solution, "y");
        verify::assert_ic_satisfied(&sol.general_solution, "x", 0.0, 1.0, 1e-9);
    }

    #[test]
    fn test_substitute_var() {
        let expr = add(var("x"), var("y"));
        let result = substitute_var(&expr, "x", &int(5));
        // Should get 5 + y
        assert!(matches!(
            result,
            Expression::Binary(BinaryOp::Add, left, _) if matches!(left.as_ref(), Expression::Integer(5))
        ));
    }

    #[test]
    fn test_try_solve_implicit_ln_y() {
        // ln(y) = x + C => y = e^(x + C)
        let left = Expression::Function(Function::Ln, vec![var("y")]);
        let right = add(var("x"), var("C"));
        let result = try_solve_implicit_for_y(&left, &right, "y");
        assert!(result.is_some());
        assert!(matches!(
            result.unwrap(),
            Expression::Function(Function::Exp, _)
        ));
    }

    // =========================================================================
    // Second-Order ODE Tests
    // =========================================================================

    #[test]
    fn test_characteristic_equation_distinct_real() {
        // r² - 1 = 0 => r = ±1
        let roots = solve_characteristic_equation(1.0, 0.0, -1.0).unwrap();
        assert_eq!(roots.root_type, RootType::TwoDistinctReal);
        assert!((roots.r1 - 1.0).abs() < 1e-10);
        assert!((roots.r2 - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_characteristic_equation_complex() {
        // r² + 1 = 0 => r = ±i
        let roots = solve_characteristic_equation(1.0, 0.0, 1.0).unwrap();
        assert_eq!(roots.root_type, RootType::ComplexConjugate);
        assert!(roots.r1.abs() < 1e-10); // alpha = 0
        assert!((roots.r2 - 1.0).abs() < 1e-10); // beta = 1
    }

    #[test]
    fn test_characteristic_equation_repeated() {
        // r² - 2r + 1 = 0 => (r-1)² = 0 => r = 1 (double)
        let roots = solve_characteristic_equation(1.0, -2.0, 1.0).unwrap();
        assert_eq!(roots.root_type, RootType::RepeatedReal);
        assert!((roots.r1 - 1.0).abs() < 1e-10);
        assert!((roots.r2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_second_order_homogeneous_distinct_real() {
        // y'' - y = 0 => y = C1*e^x + C2*e^(-x)
        let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 0.0, -1.0);
        let solution = solve_second_order_homogeneous(&ode).unwrap();

        assert_eq!(
            solution.method,
            "Characteristic equation - distinct real roots"
        );
        assert_eq!(solution.roots.root_type, RootType::TwoDistinctReal);
        assert!(!solution.steps.is_empty());
    }

    #[test]
    fn test_second_order_homogeneous_complex() {
        // y'' + y = 0 => y = C1*cos(x) + C2*sin(x)
        let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 0.0, 1.0);
        let solution = solve_second_order_homogeneous(&ode).unwrap();

        assert_eq!(
            solution.method,
            "Characteristic equation - complex conjugate roots"
        );
        assert_eq!(solution.roots.root_type, RootType::ComplexConjugate);
    }

    #[test]
    fn test_second_order_homogeneous_repeated() {
        // y'' - 2y' + y = 0 => y = (C1 + C2*x)*e^x
        let ode = SecondOrderODE::homogeneous("y", "x", 1.0, -2.0, 1.0);
        let solution = solve_second_order_homogeneous(&ode).unwrap();

        assert_eq!(solution.method, "Characteristic equation - repeated root");
        assert_eq!(solution.roots.root_type, RootType::RepeatedReal);
    }

    #[test]
    fn test_second_order_ivp_complex() {
        // y'' + y = 0, y(0) = 1, y'(0) = 0 => y = cos(x)
        let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 0.0, 1.0);
        let solution = solve_second_order_ivp(&ode, 0.0, 1.0, 0.0).unwrap();

        // Decision 2b: y-free + IC at x₀
        verify::assert_y_free(&solution, "y");
        verify::assert_ic_satisfied(&solution, "x", 0.0, 1.0, 1e-10);

        // Additional spot-check at x = π/2: cos(π/2) = 0
        let mut vars = std::collections::HashMap::new();
        vars.insert("x".to_string(), std::f64::consts::FRAC_PI_2);
        let result = solution.evaluate(&vars).unwrap();
        assert!(result.abs() < 1e-6);
    }

    #[test]
    fn test_second_order_ivp_distinct_real() {
        // y'' - y = 0, y(0) = 1, y'(0) = 0
        // General: y = C1*e^x + C2*e^(-x)
        // y(0) = C1 + C2 = 1
        // y'(0) = C1 - C2 = 0 => C1 = C2 = 0.5
        // y = 0.5*e^x + 0.5*e^(-x) = cosh(x)
        let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 0.0, -1.0);
        let solution = solve_second_order_ivp(&ode, 0.0, 1.0, 0.0).unwrap();

        // Decision 2b: y-free + IC at x₀
        verify::assert_y_free(&solution, "y");
        verify::assert_ic_satisfied(&solution, "x", 0.0, 1.0, 1e-10);

        // Additional spot-check at x = 1: cosh(1)
        let mut vars = std::collections::HashMap::new();
        vars.insert("x".to_string(), 1.0);
        let result = solution.evaluate(&vars).unwrap();
        let expected = 1.0_f64.cosh();
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_second_order_ode_is_homogeneous() {
        let ode1 = SecondOrderODE::homogeneous("y", "x", 1.0, 2.0, 3.0);
        assert!(ode1.is_homogeneous());

        let ode2 = SecondOrderODE::new("y", "x", 1.0, 2.0, 3.0, var("x"));
        assert!(!ode2.is_homogeneous());
    }
}
