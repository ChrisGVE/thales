//! Higher-order ordinary differential equation solver.
//!
//! Solves higher-order linear ODEs with constant coefficients using
//! characteristic equation methods and undetermined coefficients.

mod helpers;
mod particular;
mod solver;
mod types;

pub use particular::{solve_undetermined_coefficients, ForcingKind};
pub use solver::solve_higher_order_homogeneous;
pub use types::{CharRoot, HigherOrderODE, HigherOrderSolution};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Expression;
    use crate::ode::{ODEError, SecondOrderODE};

    // ------------------------------------------------------------------
    // Higher-order homogeneous
    // ------------------------------------------------------------------

    #[test]
    fn test_second_order_y_double_prime_plus_y_prime_minus_2y() {
        // y'' + y' - 2y = 0  =>  char eq r² + r - 2 = 0  =>  r = 1, -2
        let ode = HigherOrderODE::new("y", "x", vec![1.0, 1.0, -2.0]);
        let sol = solve_higher_order_homogeneous(&ode).unwrap();

        assert_eq!(sol.roots.len(), 2);
        let reals: Vec<f64> = {
            let mut v: Vec<f64> = sol.roots.iter().map(|r| r.real).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v
        };
        assert!(
            (reals[0] - (-2.0)).abs() < 1e-4,
            "Expected root -2, got {}",
            reals[0]
        );
        assert!(
            (reals[1] - 1.0).abs() < 1e-4,
            "Expected root 1, got {}",
            reals[1]
        );
    }

    #[test]
    fn test_third_order_char_roots_1_2_3() {
        // y''' - 6y'' + 11y' - 6y = 0
        // char eq: r³ - 6r² + 11r - 6 = 0  =>  r = 1, 2, 3
        let ode = HigherOrderODE::new("y", "x", vec![1.0, -6.0, 11.0, -6.0]);
        let sol = solve_higher_order_homogeneous(&ode).unwrap();

        assert_eq!(sol.roots.len(), 3);
        let mut reals: Vec<f64> = sol.roots.iter().map(|r| r.real).collect();
        reals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((reals[0] - 1.0).abs() < 1e-4, "root 0: {}", reals[0]);
        assert!((reals[1] - 2.0).abs() < 1e-4, "root 1: {}", reals[1]);
        assert!((reals[2] - 3.0).abs() < 1e-4, "root 2: {}", reals[2]);
    }

    #[test]
    fn test_higher_order_solution_is_expression() {
        let ode = HigherOrderODE::new("y", "x", vec![1.0, -6.0, 11.0, -6.0]);
        let sol = solve_higher_order_homogeneous(&ode).unwrap();
        // General solution must be non-trivial.
        let sol_expr = crate::numeric::compile::decompile(&sol.general_solution);
        assert!(!matches!(sol_expr, Expression::Integer(0)));
    }

    #[test]
    fn test_higher_order_invalid_leading_zero() {
        let ode = HigherOrderODE::new("y", "x", vec![0.0, 1.0, -2.0]);
        let result = solve_higher_order_homogeneous(&ode);
        assert!(matches!(
            result,
            Err(ODEError::CharacteristicEquationError(_))
        ));
    }

    #[test]
    fn test_higher_order_too_short_coeffs() {
        let ode = HigherOrderODE::new("y", "x", vec![1.0]);
        let result = solve_higher_order_homogeneous(&ode);
        assert!(matches!(
            result,
            Err(ODEError::CharacteristicEquationError(_))
        ));
    }

    // ------------------------------------------------------------------
    // Undetermined coefficients
    // ------------------------------------------------------------------

    #[test]
    fn test_undetermined_coefficients_exponential() {
        // y'' - 3y' + 2y = e^(4x)
        // Homogeneous: r² - 3r + 2 = 0 => r = 1, 2
        // Particular trial: A·e^(4x); A = 1/(16-12+2) = 1/6
        let ode = SecondOrderODE::new("y", "x", 1.0, -3.0, 2.0, Expression::Integer(0));
        let sol = solve_undetermined_coefficients(&ode, ForcingKind::Exponential(4.0)).unwrap();
        assert!(sol.particular_solution.is_some());
        let part = sol.particular_solution.unwrap();
        // Evaluate at x=0: should be A = 1/6 ≈ 0.1667
        let mut vars = std::collections::HashMap::new();
        vars.insert("x".to_string(), 0.0);
        let val = crate::numeric::compile::decompile(&part)
            .evaluate(&vars)
            .unwrap();
        assert!((val - 1.0 / 6.0).abs() < 1e-6, "A at x=0: {val}");
    }

    #[test]
    fn test_undetermined_coefficients_polynomial_degree1() {
        // y'' + y' + y = x  =>  particular y_p = x - 1
        // A = 1/c = 1, B = -b·A/c = -1
        let ode = SecondOrderODE::new("y", "x", 1.0, 1.0, 1.0, Expression::Integer(0));
        let sol = solve_undetermined_coefficients(&ode, ForcingKind::Polynomial(1)).unwrap();
        assert!(sol.particular_solution.is_some());
    }

    #[test]
    fn test_undetermined_coefficients_sinusoidal() {
        // y'' + 4y = cos(2x) => resonance (ω=2 matches imaginary part β=2)
        let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 0.0, 4.0);
        let result = solve_undetermined_coefficients(&ode, ForcingKind::Sinusoidal(2.0));
        assert!(matches!(result, Err(ODEError::ResonanceDetected(_))));
    }

    #[test]
    fn test_undetermined_coefficients_sinusoidal_no_resonance() {
        // y'' + 9y = cos(2x)  (ω=2 ≠ β=3)
        // p = 9 - 4 = 5, q = 0, A = 5/25 = 1/5, B = 0
        let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 0.0, 9.0);
        let sol = solve_undetermined_coefficients(&ode, ForcingKind::Sinusoidal(2.0)).unwrap();
        assert!(sol.particular_solution.is_some());
        let part = sol.particular_solution.unwrap();
        let mut vars = std::collections::HashMap::new();
        vars.insert("x".to_string(), 0.0);
        let val = crate::numeric::compile::decompile(&part)
            .evaluate(&vars)
            .unwrap();
        // At x=0: A·cos(0) + B·sin(0) = A = 1/5
        assert!((val - 0.2).abs() < 1e-6, "val at x=0: {val}");
    }

    #[test]
    fn test_undetermined_coefficients_resonance_exponential() {
        // y'' - y = e^x  =>  r=1 is a root, resonance
        let ode = SecondOrderODE::new("y", "x", 1.0, 0.0, -1.0, Expression::Integer(0));
        let result = solve_undetermined_coefficients(&ode, ForcingKind::Exponential(1.0));
        assert!(matches!(result, Err(ODEError::ResonanceDetected(_))));
    }
}
