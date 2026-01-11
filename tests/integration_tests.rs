//! Integration tests for thales.
//!
//! Tests the complete workflow from parsing through solving.

use thales::ast::{Expression, Variable};
use thales::transforms::{Cartesian2D, Cartesian3D, Polar};

#[test]
fn test_library_version() {
    let version = thales::version();
    assert!(!version.is_empty());
    assert_eq!(version, "0.2.1");
}

#[test]
fn test_ffi_support() {
    // This will be false unless compiled with --features ffi
    let _has_ffi = thales::has_ffi_support();
}

// Parser integration tests
mod parser_tests {
    #[test]
    fn test_parse_simple_equation() {
        let result = thales::parse_equation("x + 2 = 5");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_expression_with_functions() {
        let result = thales::parse_expression("sin(x) + cos(x)");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_complex_equation() {
        let result = thales::parse_equation("2*x^2 + 3*x - 5 = 0");
        assert!(result.is_ok());
    }
}

// Solver integration tests
mod solver_tests {
    use super::*;
    use thales::solver::{SmartSolver, Solver};

    #[test]
    fn test_solve_linear_equation() {
        // 2x + 3 = 7 => x = 2
        let equation = thales::parse_equation("2*x + 3 = 7").unwrap();
        let solver = SmartSolver::new();
        let result = solver.solve(&equation, &Variable::new("x"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_solve_quadratic_equation() {
        // x^2 - 5x + 6 = 0 => x = 2 or x = 3
        let equation = thales::parse_equation("x^2 - 5*x + 6 = 0").unwrap();
        let solver = SmartSolver::new();
        let result = solver.solve(&equation, &Variable::new("x"));
        assert!(result.is_ok());
    }
}

// Coordinate transformation tests
mod transform_tests {
    use super::*;

    #[test]
    fn test_cartesian_to_polar_conversion() {
        let cart = Cartesian2D::new(3.0, 4.0);
        let polar = cart.to_polar();
        assert!((polar.r - 5.0).abs() < 1e-10);
        assert!((polar.theta - 0.927295218).abs() < 1e-6);
    }

    #[test]
    fn test_polar_to_cartesian_conversion() {
        let polar = Polar::new(5.0, 0.927295218);
        let cart = polar.to_cartesian();
        assert!((cart.x - 3.0).abs() < 1e-6);
        assert!((cart.y - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_cartesian_to_spherical_conversion() {
        let cart = Cartesian3D::new(1.0, 1.0, 1.0);
        let spherical = cart.to_spherical();
        assert!((spherical.r - 1.732050808).abs() < 1e-6);
    }

    #[test]
    fn test_coordinate_round_trip() {
        let original = Cartesian2D::new(5.0, 12.0);
        let polar = original.to_polar();
        let back = polar.to_cartesian();
        assert!((original.x - back.x).abs() < 1e-10);
        assert!((original.y - back.y).abs() < 1e-10);
    }
}

// Numerical solver tests
mod numerical_tests {
    use super::*;
    use thales::numerical::{NumericalConfig, SmartNumericalSolver};

    #[test]
    fn test_numerical_root_finding() {
        // Find root of x^2 - 2 = 0 (should be sqrt(2) ≈ 1.414)
        let equation = thales::parse_equation("x^2 - 2 = 0").unwrap();
        let config = NumericalConfig::default();
        let solver = SmartNumericalSolver::new(config);
        let result = solver.solve(&equation, &Variable::new("x"));
        assert!(result.is_ok());
    }
}

// Dimension and unit tests
mod dimension_tests {
    use thales::dimensions::{BaseDimension, Dimension, UnitRegistry};

    #[test]
    fn test_dimensionless_quantity() {
        let dim = Dimension::dimensionless();
        assert!(dim.is_dimensionless());
    }

    #[test]
    fn test_base_dimension_creation() {
        let length = Dimension::from_base(BaseDimension::Length, 1);
        assert!(!length.is_dimensionless());
    }

    #[test]
    #[ignore] // Unit conversion planned for v0.4.0
    fn test_unit_conversion() {
        let registry = UnitRegistry::with_common_units();
        let result = registry.convert(1000.0, "m", "km");
        assert!(result.is_ok());
        assert!((result.unwrap() - 1.0).abs() < 1e-10);
    }
}

// Resolution path tests
mod resolution_path_tests {
    use super::*;
    use thales::resolution_path::{Operation, ResolutionPathBuilder};

    #[test]
    fn test_empty_resolution_path() {
        let expr = Expression::Integer(5);
        let path = thales::ResolutionPath::new(expr);
        assert!(path.is_empty());
        assert_eq!(path.step_count(), 0);
    }

    #[test]
    fn test_resolution_path_builder() {
        let initial = Expression::Integer(5);
        let path = ResolutionPathBuilder::new(initial.clone())
            .step(
                Operation::Simplify,
                "Simplify expression".to_string(),
                Expression::Integer(5),
            )
            .finish(Expression::Integer(5));

        assert_eq!(path.step_count(), 1);
        assert!(!path.is_empty());
    }
}

// Property-based tests (using proptest)
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_cartesian_polar_round_trip(x in -1000.0..1000.0, y in -1000.0..1000.0) {
            let cart = Cartesian2D::new(x, y);
            let polar = cart.to_polar();
            let back = polar.to_cartesian();
            prop_assert!((cart.x - back.x).abs() < 1e-10);
            prop_assert!((cart.y - back.y).abs() < 1e-10);
        }

        #[test]
        fn test_cartesian_magnitude(x in -1000.0..1000.0, y in -1000.0..1000.0) {
            let cart = Cartesian2D::new(x, y);
            let polar = cart.to_polar();
            prop_assert!((cart.magnitude() - polar.r).abs() < 1e-10);
        }
    }
}

// TODO: Add integration tests for complete solve workflows
// TODO: Add tests for error handling and edge cases
// TODO: Add tests for complex equation systems
// TODO: Add tests for transcendental equations
// TODO: Add tests for numerical stability
// TODO: Add tests for FFI boundary (when feature enabled)
// TODO: Add tests for concurrent solving
// TODO: Add snapshot tests for resolution paths
