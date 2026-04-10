//! Calculus wrapper handling for symbolic isolation.
//!
//! Handles factoring target variables out of calculus wrappers such as
//! integrals, sums, products, limits, and derivatives.

use crate::ast::{BinaryOp, Expression, Function, UnaryOp, Variable};
use crate::resolution_path::{Operation, ResolutionPathBuilder, StepAnnotation};

use super::super::helpers::contains_variable;
use super::super::types::SolverError;
use super::unwrap::unwrap_variable;

/// Check whether a function is a calculus wrapper (integral, sum, product, limit).
pub(super) fn is_calculus_wrapper(func: &Function) -> bool {
    matches!(
        func,
        Function::Custom(name) if matches!(name.as_str(), "integral" | "sum" | "product" | "limit" | "derivative")
    )
}

/// Try to factor a target variable out of a product expression.
///
/// Given `a * b * var * c`, returns `Some((var_factor, rest))` where
/// `var_factor` contains the variable and `rest` does not.
/// Works for linear factors (`F * stuff`) and power factors (`T^4 * stuff`).
pub(super) fn try_split_product(expr: &Expression, var: &str) -> Option<(Expression, Expression)> {
    if !contains_variable(expr, var) {
        return None;
    }

    // If the expression IS the variable, factor is var, rest is 1
    if let Expression::Variable(v) = expr {
        if v.name == var {
            return Some((expr.clone(), Expression::Integer(1)));
        }
    }

    // Power of the variable: var^n
    if let Expression::Power(base, exp) = expr {
        if let Expression::Variable(v) = base.as_ref() {
            if v.name == var && !contains_variable(exp, var) {
                return Some((expr.clone(), Expression::Integer(1)));
            }
        }
    }

    // Function containing variable: treat entire function call as the var_factor
    // e.g., sin(omega*t) → (sin(omega*t), 1)
    if let Expression::Function(_, _) = expr {
        if contains_variable(expr, var) {
            return Some((expr.clone(), Expression::Integer(1)));
        }
    }

    // Unary negation: -expr(v) → (-1 * expr(v))
    if let Expression::Unary(UnaryOp::Neg, inner) = expr {
        if let Some((factor, rest)) = try_split_product(inner, var) {
            let neg_rest = Expression::Unary(UnaryOp::Neg, Box::new(rest)).simplify();
            return Some((factor, neg_rest));
        }
    }

    // Multiplication: try to split into var-containing and non-var parts
    if let Expression::Binary(BinaryOp::Mul, left, right) = expr {
        let left_has = contains_variable(left, var);
        let right_has = contains_variable(right, var);

        if left_has && !right_has {
            if let Some((factor, inner_rest)) = try_split_product(left, var) {
                let rest = Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(inner_rest),
                    Box::new(right.as_ref().clone()),
                )
                .simplify();
                return Some((factor, rest));
            }
        } else if right_has && !left_has {
            if let Some((factor, inner_rest)) = try_split_product(right, var) {
                let rest = Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(left.as_ref().clone()),
                    Box::new(inner_rest),
                )
                .simplify();
                return Some((factor, rest));
            }
        }
        // Both have var → can't factor cleanly
    }

    // Division: var_factor / something  OR  something / var_factor
    if let Expression::Binary(BinaryOp::Div, left, right) = expr {
        let left_has = contains_variable(left, var);
        let right_has = contains_variable(right, var);

        if left_has && !right_has {
            if let Some((factor, inner_rest)) = try_split_product(left, var) {
                let rest = Expression::Binary(
                    BinaryOp::Div,
                    Box::new(inner_rest),
                    Box::new(right.as_ref().clone()),
                )
                .simplify();
                return Some((factor, rest));
            }
        } else if right_has && !left_has {
            // something / var_expr → treat as (1/var_expr) * something
            if let Some((factor, inner_rest)) = try_split_product(right, var) {
                let inv_factor = Expression::Binary(
                    BinaryOp::Div,
                    Box::new(Expression::Integer(1)),
                    Box::new(factor),
                )
                .simplify();
                let rest = Expression::Binary(
                    BinaryOp::Div,
                    Box::new(left.as_ref().clone()),
                    Box::new(inner_rest),
                )
                .simplify();
                return Some((inv_factor, rest));
            }
        }
    }

    None
}

/// Try to handle a calculus wrapper by factoring the target variable out of its body.
///
/// For `integral(F*cos(theta), x)` solving for F:
/// → factor F out of integrand → F * integral(cos(theta), x)
/// → then standard algebraic isolation handles F = other / integral(cos(theta), x)
pub(super) fn try_unwrap_calculus_wrapper(
    func: &Function,
    args: &[Expression],
    other: &Expression,
    var: &str,
    path: ResolutionPathBuilder,
) -> Result<(Expression, ResolutionPathBuilder), SolverError> {
    let func_name = match func {
        Function::Custom(name) => name.as_str(),
        _ => return Err(SolverError::CannotSolve("Not a calculus wrapper".into())),
    };

    // Get the body expression (always first argument)
    let body = &args[0];

    if !contains_variable(body, var) {
        return Err(SolverError::CannotSolve(format!(
            "Variable '{}' not found in {} body",
            var, func_name
        )));
    }

    // For integral/sum/product: check that the target variable is NOT the
    // integration/summation variable (second argument).
    if args.len() >= 2 {
        if let Expression::Variable(v) = &args[1] {
            if v.name == var {
                return Err(SolverError::CannotSolve(format!(
                    "Cannot isolate '{}': it is the {} variable",
                    var, func_name
                )));
            }
        }
    }

    // Try to factor the target variable out of the body
    if let Some((var_factor, rest_body)) = try_split_product(body, var) {
        // Reconstruct the calculus wrapper with the remaining body
        let mut new_args = args.to_vec();
        new_args[0] = rest_body;
        let wrapper_expr = Expression::Function(func.clone(), new_args);

        // Now the equation is: var_factor * wrapper = other
        let factored =
            Expression::Binary(BinaryOp::Mul, Box::new(var_factor), Box::new(wrapper_expr));

        let p = path.annotated_step(
            Operation::ApplyFunction(format!("factor_from_{}", func_name)),
            format!("Factor '{}' out of {} body", var, func_name),
            other.clone(),
            StepAnnotation::calculus("Calculus Wrapper Isolation"),
        );

        return unwrap_variable(&factored, other, var, p);
    }

    // If body is JUST the variable (e.g., integral(F, x)) → treat wrapper as identity-like
    if let Expression::Variable(v) = body {
        if v.name == var {
            let mut new_args = args.to_vec();
            new_args[0] = Expression::Integer(1);
            let wrapper_one = Expression::Function(func.clone(), new_args);
            let new_other = Expression::Binary(
                BinaryOp::Div,
                Box::new(other.clone()),
                Box::new(wrapper_one),
            )
            .simplify();

            let p = path.annotated_step(
                Operation::DivideBothSides(Expression::Variable(Variable::new(var))),
                format!("Isolate '{}' from {} body", var, func_name),
                new_other.clone(),
                StepAnnotation::calculus("Calculus Wrapper Isolation"),
            );

            return Ok((new_other, p));
        }
    }

    Err(SolverError::CannotSolve(format!(
        "Cannot factor '{}' out of {} body",
        var, func_name
    )))
}
