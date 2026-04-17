//! Substitute numeric values for variables in an expression tree.

use std::collections::HashMap;

use crate::ast::Expression;

/// Substitute known variable values into an expression.
///
/// Any variable present in `values` is replaced with its `Expression::Float`
/// value; all other variables are left in place.
pub(crate) fn substitute_values(expr: &Expression, values: &HashMap<String, f64>) -> Expression {
    match expr {
        Expression::Variable(v) => {
            if let Some(&value) = values.get(&v.name) {
                Expression::Float(value)
            } else {
                expr.clone()
            }
        }
        Expression::Unary(op, inner) => {
            Expression::Unary(*op, Box::new(substitute_values(inner, values)))
        }
        Expression::Binary(op, left, right) => Expression::Binary(
            *op,
            Box::new(substitute_values(left, values)),
            Box::new(substitute_values(right, values)),
        ),
        Expression::Function(func, args) => Expression::Function(
            func.clone(),
            args.iter()
                .map(|arg| substitute_values(arg, values))
                .collect(),
        ),
        Expression::Power(base, exp) => Expression::Power(
            Box::new(substitute_values(base, values)),
            Box::new(substitute_values(exp, values)),
        ),
        _ => expr.clone(),
    }
}
