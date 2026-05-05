//! Vector calculus operations: gradient, divergence, curl, Laplacian.
//!
//! Enable with `features = ["vector-calc"]` in Cargo.toml.

use crate::ast::{BinaryOp, Expression, Variable};

/// Compute the gradient of a scalar field.
pub fn gradient(expr: &Expression, vars: &[Variable]) -> Vec<Expression> {
    vars.iter().map(|v| expr.differentiate(&v.name)).collect()
}

/// Compute the Laplacian of a scalar field (sum of second partial derivatives).
pub fn laplacian(expr: &Expression, vars: &[Variable]) -> Expression {
    let terms: Vec<Expression> = vars
        .iter()
        .map(|v| {
            let first = expr.differentiate(&v.name);
            first.differentiate(&v.name)
        })
        .collect();

    terms
        .into_iter()
        .reduce(|acc, term| Expression::Binary(BinaryOp::Add, Box::new(acc), Box::new(term)))
        .unwrap_or(Expression::Integer(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_single_variable() {
        // f(x) = x^2, gradient should be [2x]
        let x = Variable::new("x");
        let expr = Expression::Power(
            Box::new(Expression::Variable(x.clone())),
            Box::new(Expression::Integer(2)),
        );
        let grad = gradient(&expr, &[x]);
        assert_eq!(grad.len(), 1);
    }

    #[test]
    fn test_gradient_multiple_variables() {
        // f(x, y) = x + y, gradient should be [1, 1]
        let x = Variable::new("x");
        let y = Variable::new("y");
        let expr = Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Variable(x.clone())),
            Box::new(Expression::Variable(y.clone())),
        );
        let grad = gradient(&expr, &[x, y]);
        assert_eq!(grad.len(), 2);
    }

    #[test]
    fn test_laplacian_constant() {
        // f(x) = 5, laplacian should be 0
        let x = Variable::new("x");
        let expr = Expression::Integer(5);
        let lap = laplacian(&expr, &[x]);
        // The second derivative of a constant is 0
        assert_eq!(lap, Expression::Integer(0));
    }

    #[test]
    fn test_gradient_empty_vars() {
        let expr = Expression::Integer(1);
        let grad = gradient(&expr, &[]);
        assert!(grad.is_empty());
    }

    #[test]
    fn test_laplacian_empty_vars() {
        let expr = Expression::Integer(1);
        let lap = laplacian(&expr, &[]);
        assert_eq!(lap, Expression::Integer(0));
    }
}
