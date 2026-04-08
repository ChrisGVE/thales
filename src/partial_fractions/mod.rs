//! Partial fraction decomposition.
//!
//! Decomposes rational functions (ratios of polynomials) into sums of
//! simpler fractions. This is the inverse of combining fractions over
//! a common denominator.

mod decompose;
mod display;
mod polynomial;
mod types;

pub use decompose::decompose;
pub use polynomial::{get_polynomial_degree, is_polynomial, is_rational_function};
pub use types::{DecomposeError, PartialFractionResult, PartialFractionTerm};

#[cfg(test)]
mod tests {
    use super::decompose::*;
    use super::polynomial::*;
    use super::*;
    use crate::ast::{BinaryOp, Expression, Function, UnaryOp, Variable};
    use std::collections::HashMap;

    #[test]
    fn test_is_polynomial() {
        let x = Expression::Variable(Variable::new("x"));
        assert!(is_polynomial(&x, "x"));
        assert!(is_polynomial(&Expression::Integer(5), "x"));

        let x_squared = Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(2)));
        assert!(is_polynomial(&x_squared, "x"));

        // x + 1
        let x_plus_1 = Expression::Binary(
            BinaryOp::Add,
            Box::new(x.clone()),
            Box::new(Expression::Integer(1)),
        );
        assert!(is_polynomial(&x_plus_1, "x"));

        // sin(x) is not a polynomial
        let sin_x = Expression::Function(Function::Sin, vec![x.clone()]);
        assert!(!is_polynomial(&sin_x, "x"));
    }

    #[test]
    fn test_is_rational_function() {
        let x = Expression::Variable(Variable::new("x"));

        // x / (x + 1) is rational
        let rational = Expression::Binary(
            BinaryOp::Div,
            Box::new(x.clone()),
            Box::new(Expression::Binary(
                BinaryOp::Add,
                Box::new(x.clone()),
                Box::new(Expression::Integer(1)),
            )),
        );
        assert!(is_rational_function(&rational, "x"));

        // sin(x) / x is not rational
        let not_rational = Expression::Binary(
            BinaryOp::Div,
            Box::new(Expression::Function(Function::Sin, vec![x.clone()])),
            Box::new(x.clone()),
        );
        assert!(!is_rational_function(&not_rational, "x"));
    }

    #[test]
    fn test_polynomial_degree() {
        let x = Expression::Variable(Variable::new("x"));

        assert_eq!(get_polynomial_degree(&x, "x"), Some(1));
        assert_eq!(get_polynomial_degree(&Expression::Integer(5), "x"), Some(0));

        let x_cubed = Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(3)));
        assert_eq!(get_polynomial_degree(&x_cubed, "x"), Some(3));

        let poly = Expression::Binary(BinaryOp::Add, Box::new(x_cubed), Box::new(x.clone()));
        assert_eq!(get_polynomial_degree(&poly, "x"), Some(3));
    }

    #[test]
    fn test_extract_coefficients() {
        let x = Expression::Variable(Variable::new("x"));

        // x² + 2x + 1
        let x_squared = Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(2)));
        let two_x = Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Integer(2)),
            Box::new(x.clone()),
        );
        let poly = Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Binary(
                BinaryOp::Add,
                Box::new(x_squared),
                Box::new(two_x),
            )),
            Box::new(Expression::Integer(1)),
        );

        let coeffs = extract_coefficients(&poly, "x").unwrap();
        assert!((coeffs.get(&0).unwrap_or(&0.0) - 1.0).abs() < 1e-10);
        assert!((coeffs.get(&1).unwrap_or(&0.0) - 2.0).abs() < 1e-10);
        assert!((coeffs.get(&2).unwrap_or(&0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_find_polynomial_roots_linear() {
        let mut coeffs = HashMap::new();
        coeffs.insert(0, -2.0); // -2
        coeffs.insert(1, 1.0); // +x

        let roots = find_polynomial_roots(&coeffs);
        assert_eq!(roots.len(), 1);
        assert!((roots[0].0 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_find_polynomial_roots_quadratic() {
        // x² - 1 = (x-1)(x+1)
        let mut coeffs = HashMap::new();
        coeffs.insert(0, -1.0);
        coeffs.insert(2, 1.0);

        let roots = find_polynomial_roots(&coeffs);
        assert_eq!(roots.len(), 2);

        let root_values: Vec<f64> = roots.iter().map(|(r, _)| *r).collect();
        assert!(root_values.iter().any(|r| (r - 1.0).abs() < 1e-10));
        assert!(root_values.iter().any(|r| (r + 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_decompose_simple() {
        // 1/(x² - 1) = 1/((x-1)(x+1)) = 1/(2(x-1)) - 1/(2(x+1))
        let x = Expression::Variable(Variable::new("x"));
        let num = Expression::Integer(1);
        let denom = Expression::Binary(
            BinaryOp::Sub,
            Box::new(Expression::Power(
                Box::new(x.clone()),
                Box::new(Expression::Integer(2)),
            )),
            Box::new(Expression::Integer(1)),
        );

        let result = decompose(&num, &denom, &Variable::new("x")).unwrap();
        assert_eq!(result.terms.len(), 2);

        // Check that we have linear terms
        for term in &result.terms {
            match term {
                PartialFractionTerm::Linear {
                    coefficient,
                    root: _,
                    power,
                } => {
                    assert_eq!(*power, 1);
                    // Coefficients should be ±1/2
                    assert!((coefficient.abs() - 0.5).abs() < 1e-10);
                }
                _ => panic!("Expected linear terms"),
            }
        }
    }

    #[test]
    fn test_decompose_x_times_x_minus_1() {
        // 1/(x² - x) = 1/(x(x-1)) = -1/x + 1/(x-1)
        // Use expanded form: x² - x
        let x = Expression::Variable(Variable::new("x"));
        let num = Expression::Integer(1);
        let x_squared = Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(2)));
        let denom = Expression::Binary(BinaryOp::Sub, Box::new(x_squared), Box::new(x.clone()));

        let result = decompose(&num, &denom, &Variable::new("x")).unwrap();
        assert_eq!(result.terms.len(), 2);
    }

    #[test]
    fn test_linear_term_integration() {
        // ∫ 1/(x-2) dx = ln|x-2|
        let term = PartialFractionTerm::Linear {
            coefficient: 1.0,
            root: 2.0,
            power: 1,
        };

        let integral = term.integrate("x");
        // Should contain ln
        assert!(format!("{:?}", integral).contains("Ln"));
    }

    #[test]
    fn test_linear_term_higher_power_integration() {
        // ∫ 1/(x-1)² dx = -1/(x-1)
        let term = PartialFractionTerm::Linear {
            coefficient: 1.0,
            root: 1.0,
            power: 2,
        };

        let integral = term.integrate("x");
        // Should be a fraction with power 1
        assert!(format!("{:?}", integral).contains("Div"));
    }

    #[test]
    fn test_irreducible_quadratic() {
        // x² + 1 has discriminant -4 < 0
        assert!(is_irreducible_quadratic(0.0, 1.0));

        // x² - 1 has discriminant 4 > 0
        assert!(!is_irreducible_quadratic(0.0, -1.0));
    }
}
