//! Structural predicates over `Expression` and `Expr`.
//!
//! The Expression-based checks (`contains_variable`, `has_any_variable`, …)
//! are used everywhere in the solver; the Expr-based `contains_symbol`
//! mirrors the same question for the internal compiled form and is used by
//! modules already migrated to `Arc<Expr>`.

use std::collections::HashMap;
use std::sync::Arc;

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

// ── Arc<Expr> ports ───────────────────────────────────────────────────────

/// `true` if `expr` references any symbol. Expr counterpart of
/// [`has_any_variable`].
pub(crate) fn has_any_symbol(expr: &Expr) -> bool {
    match expr {
        Expr::Symbol(_) => true,
        Expr::Integer(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Complex(_)
        | Expr::Constant(_) => false,
        Expr::Add(node) => node.terms.keys().any(|t| has_any_symbol(t)),
        Expr::Mul(node) => node
            .factors
            .iter()
            .any(|(b, e)| has_any_symbol(b) || has_any_symbol(e)),
        Expr::Pow(base, exp) => has_any_symbol(base) || has_any_symbol(exp),
        Expr::Func(_, args) => args.iter().any(|a| has_any_symbol(a)),
    }
}

/// Expr counterpart of [`has_obvious_nonlinearity`]: detects
/// `Pow(base, exp)` where `base` contains any symbol and `exp` is a
/// numeric value greater than 1, or a canonical `Mul` factor `var^n`
/// with numeric `n > 1`.
pub(crate) fn has_obvious_nonlinearity_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Pow(base, exp) => {
            if has_any_symbol(base) && numeric_gt_one(exp) {
                return true;
            }
            has_obvious_nonlinearity_expr(base) || has_obvious_nonlinearity_expr(exp)
        }
        Expr::Mul(node) => node.factors.iter().any(|(b, e)| {
            if has_any_symbol(b) && numeric_gt_one(e) {
                return true;
            }
            has_obvious_nonlinearity_expr(b) || has_obvious_nonlinearity_expr(e)
        }),
        Expr::Add(node) => node.terms.keys().any(|t| has_obvious_nonlinearity_expr(t)),
        Expr::Func(_, args) => args.iter().any(|a| has_obvious_nonlinearity_expr(a)),
        _ => false,
    }
}

/// `true` if `exp` is a numeric literal strictly greater than 1.
fn numeric_gt_one(exp: &Arc<Expr>) -> bool {
    match exp.as_ref() {
        Expr::Integer(n) => n.to_i64().map(|v| v > 1).unwrap_or(false),
        Expr::Rational(r) => r.to_f64() > 1.0,
        Expr::Float(f) => *f > 1.0,
        _ => false,
    }
}

/// Expr counterpart of [`is_polynomial_expression`]: `true` when the
/// expression contains only algebraic structure (no `Func` nodes) and
/// every `Pow` has a non-negative integer exponent (or numeric value
/// equivalent).
pub(crate) fn is_polynomial_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Integer(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Complex(_)
        | Expr::Constant(_)
        | Expr::Symbol(_) => true,
        Expr::Add(node) => node.terms.keys().all(|t| is_polynomial_expr(t)),
        Expr::Mul(node) => node
            .factors
            .iter()
            .all(|(b, e)| is_polynomial_expr(b) && is_polynomial_expr_exp(e)),
        Expr::Pow(base, exp) => is_polynomial_expr(base) && is_polynomial_expr_exp(exp),
        Expr::Func(_, _) => false,
    }
}

/// Exponent predicate: non-negative integer (exact), or numeric value
/// equivalent to a non-negative integer.
fn is_polynomial_expr_exp(exp: &Arc<Expr>) -> bool {
    match exp.as_ref() {
        Expr::Integer(n) => n.to_i64().map(|v| v >= 0).unwrap_or(false),
        Expr::Rational(r) => {
            let v = r.to_f64();
            v >= 0.0 && (v - v.round()).abs() < 1e-10
        }
        Expr::Float(f) => *f >= 0.0 && (*f - f.round()).abs() < 1e-10,
        // Symbolic exponent: treat as polynomial if exponent itself is
        // purely algebraic (matches legacy behaviour where a polynomial
        // exponent is recursively validated).
        _ => is_polynomial_expr(exp),
    }
}

/// Expr counterpart of [`is_linear_in_variable`]: `true` if `var`
/// appears at most to degree 1, not in a denominator, not multiplied by
/// itself, and not inside any function.
pub(crate) fn is_linear_in_variable_expr(expr: &Expr, var: SymbolId) -> bool {
    match expr {
        Expr::Integer(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Complex(_)
        | Expr::Constant(_)
        | Expr::Symbol(_) => true,
        Expr::Add(node) => node
            .terms
            .keys()
            .all(|t| is_linear_in_variable_expr(t, var)),
        Expr::Mul(node) => {
            let mut var_count = 0usize;
            for (base, exp) in &node.factors {
                let base_has = contains_symbol(base, var);
                let exp_has = contains_symbol(exp, var);
                if !base_has && !exp_has {
                    continue;
                }
                if exp_has {
                    return false;
                }
                if let Expr::Symbol(s) = base.as_ref() {
                    if *s == var {
                        if !matches!(
                            exp.as_ref(),
                            Expr::Integer(n) if n.to_i64() == Some(1)
                        ) {
                            return false;
                        }
                        var_count += 1;
                        if var_count > 1 {
                            return false;
                        }
                        continue;
                    }
                }
                return false;
            }
            true
        }
        Expr::Pow(base, exp) => {
            if contains_symbol(base, var) {
                return false;
            }
            is_linear_in_variable_expr(exp, var)
        }
        Expr::Func(_, _) => false,
    }
}
