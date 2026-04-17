//! Structural predicates over `Expression` and `Expr`.
//!
//! The Expression-based checks (`contains_variable`, `has_any_variable`, …)
//! are used everywhere in the solver; the Expr-based `contains_symbol`
//! mirrors the same question for the internal compiled form and is used by
//! modules already migrated to `Arc<Expr>`.

use std::collections::HashMap;

use crate::ast::{BinaryOp, Expression};
use crate::numeric::{Expr, SymbolId};

/// Check if expression contains the given variable.
///
/// Convenience wrapper around [`Expression::contains_variable`].
pub(crate) fn contains_variable(expr: &Expression, var: &str) -> bool {
    expr.contains_variable(var)
}

/// Return `true` if `expr` references `var` anywhere in its subtree.
///
/// Expr counterpart of [`contains_variable`]; callers that already work in
/// `Arc<Expr>` can avoid a round-trip through `Expression`.
pub(crate) fn contains_symbol(expr: &Expr, var: SymbolId) -> bool {
    match expr {
        Expr::Symbol(s) => *s == var,
        Expr::Integer(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Complex(_)
        | Expr::Constant(_) => false,
        Expr::Add(node) => node.terms.keys().any(|t| contains_symbol(t, var)),
        Expr::Mul(node) => node
            .factors
            .iter()
            .any(|(b, e)| contains_symbol(b, var) || contains_symbol(e, var)),
        Expr::Pow(base, exp) => contains_symbol(base, var) || contains_symbol(exp, var),
        Expr::Func(_, args) => args.iter().any(|a| contains_symbol(a, var)),
    }
}

/// Check if an expression has obvious non-linear features like `x^2`.
pub(crate) fn has_obvious_nonlinearity(expr: &Expression) -> bool {
    match expr {
        Expression::Power(base, exp) => {
            if has_any_variable(base) {
                if let Some(exp_val) = exp.evaluate(&HashMap::new()) {
                    if exp_val > 1.0 {
                        return true;
                    }
                }
            }
            has_obvious_nonlinearity(base) || has_obvious_nonlinearity(exp)
        }
        Expression::Unary(_, inner) => has_obvious_nonlinearity(inner),
        Expression::Binary(_, left, right) => {
            has_obvious_nonlinearity(left) || has_obvious_nonlinearity(right)
        }
        Expression::Function(_, args) => args.iter().any(|arg| has_obvious_nonlinearity(arg)),
        _ => false,
    }
}

/// Check if expression contains any variables.
pub(crate) fn has_any_variable(expr: &Expression) -> bool {
    match expr {
        Expression::Variable(_) => true,
        Expression::Unary(_, inner) => has_any_variable(inner),
        Expression::Binary(_, left, right) => has_any_variable(left) || has_any_variable(right),
        Expression::Function(_, args) => args.iter().any(has_any_variable),
        Expression::Power(base, exp) => has_any_variable(base) || has_any_variable(exp),
        _ => false,
    }
}

/// Check if an expression is polynomial (contains no transcendental functions).
pub(crate) fn is_polynomial_expression(expr: &Expression) -> bool {
    match expr {
        Expression::Integer(_)
        | Expression::Rational(_)
        | Expression::Float(_)
        | Expression::Complex(_)
        | Expression::Constant(_)
        | Expression::Variable(_) => true,

        Expression::Unary(_, inner) => is_polynomial_expression(inner),

        Expression::Binary(_, left, right) => {
            is_polynomial_expression(left) && is_polynomial_expression(right)
        }

        Expression::Power(base, exp) => {
            if !is_polynomial_expression(base) {
                return false;
            }
            if let Some(exp_val) = exp.evaluate(&HashMap::new()) {
                exp_val >= 0.0 && (exp_val - exp_val.round()).abs() < 1e-10
            } else {
                is_polynomial_expression(exp)
            }
        }

        Expression::Function(_, _) => false,
    }
}

/// Check if an expression is linear with respect to a specific variable.
///
/// An expression is linear in variable `x` if:
/// - `x` appears to at most power 1
/// - `x` does not appear in denominators
/// - `x` does not appear multiplied by itself
/// - `x` does not appear in functions
pub(crate) fn is_linear_in_variable(expr: &Expression, var: &str) -> bool {
    match expr {
        Expression::Integer(_)
        | Expression::Rational(_)
        | Expression::Float(_)
        | Expression::Complex(_)
        | Expression::Constant(_) => true,

        Expression::Variable(_v) => true,

        Expression::Unary(_, inner) => is_linear_in_variable(inner, var),

        Expression::Binary(op, left, right) => {
            let left_has_var = contains_variable(left, var);
            let right_has_var = contains_variable(right, var);

            match op {
                BinaryOp::Add | BinaryOp::Sub => {
                    is_linear_in_variable(left, var) && is_linear_in_variable(right, var)
                }
                BinaryOp::Mul => {
                    if left_has_var && right_has_var {
                        false
                    } else {
                        is_linear_in_variable(left, var) && is_linear_in_variable(right, var)
                    }
                }
                BinaryOp::Div => {
                    if right_has_var {
                        false
                    } else {
                        is_linear_in_variable(left, var)
                    }
                }
                _ => false,
            }
        }

        Expression::Power(base, exp) => {
            !contains_variable(base, var) && is_linear_in_variable(exp, var)
        }

        Expression::Function(_, _) => false,
    }
}
