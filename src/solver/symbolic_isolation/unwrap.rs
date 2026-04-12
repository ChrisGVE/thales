//! Recursive unwrapping functions for symbolic isolation.
//!
//! These functions peel off operations wrapping the target variable and apply
//! their inverses to the other side of the equation.

use crate::ast::{BinaryOp, Expression, Function, UnaryOp, Variable};
use crate::resolution_path::{Operation, ResolutionPathBuilder, StepAnnotation};

use super::super::helpers::contains_variable;
use super::super::types::SolverError;
use super::calculus::try_unwrap_calculus_wrapper;
use super::linear::collect_linear_terms;

/// Recursively peel off operations wrapping the target variable, applying
/// inverse operations to `other` (the accumulating other-side expression).
pub(super) fn unwrap_variable(
    expr: &Expression,
    other: &Expression,
    var: &str,
    path: ResolutionPathBuilder,
) -> Result<(Expression, ResolutionPathBuilder), SolverError> {
    // Base case: the expression IS the variable
    if let Expression::Variable(v) = expr {
        if v.name == var {
            let result = other.simplify();
            return Ok((result, path));
        }
    }

    match expr {
        // Unary negation: -expr(v) => other = -other
        Expression::Unary(UnaryOp::Neg, inner) if contains_variable(inner, var) => {
            let new_other = Expression::Unary(UnaryOp::Neg, Box::new(other.clone())).simplify();
            let p = path.annotated_step(
                Operation::MultiplyBothSides(Expression::Integer(-1)),
                "Negate both sides".to_string(),
                new_other.clone(),
                StepAnnotation::elementary(),
            );
            unwrap_variable(inner, &new_other, var, p)
        }

        // Binary operations
        Expression::Binary(op, left, right) => unwrap_binary(*op, left, right, other, var, path),

        // Power: base^exp
        Expression::Power(base, exp) => unwrap_power(base, exp, other, var, path),

        // Function application
        Expression::Function(func, args) => unwrap_function(func, args, other, var, path),

        _ => Err(SolverError::CannotSolve(format!(
            "Cannot isolate '{}': unsupported expression structure",
            var
        ))),
    }
}

/// Handle binary operations during unwrapping.
fn unwrap_binary(
    op: BinaryOp,
    left: &Expression,
    right: &Expression,
    other: &Expression,
    var: &str,
    path: ResolutionPathBuilder,
) -> Result<(Expression, ResolutionPathBuilder), SolverError> {
    let left_has = contains_variable(left, var);
    let right_has = contains_variable(right, var);

    // Variable in both children
    if left_has && right_has {
        // For division with var in both numerator and denominator,
        // cross-multiply: f(v)/g(v) = other → f(v) - other*g(v) = 0
        if op == BinaryOp::Div {
            return super::rational::try_cross_multiply(left, right, other, var, path);
        }
        return collect_linear_terms(op, left, right, other, var, path);
    }

    match op {
        // a + expr(v) or expr(v) + a => other - a
        BinaryOp::Add => {
            let (var_child, const_child) = if left_has {
                (left, right)
            } else {
                (right, left)
            };
            let new_other = Expression::Binary(
                BinaryOp::Sub,
                Box::new(other.clone()),
                Box::new(const_child.clone()),
            )
            .simplify();
            let p = path.annotated_step(
                Operation::SubtractBothSides(const_child.clone()),
                format!("Subtract {} from both sides", const_child),
                new_other.clone(),
                StepAnnotation::elementary(),
            );
            unwrap_variable(var_child, &new_other, var, p)
        }

        // expr(v) - a => other + a ; a - expr(v) => other = a - other
        BinaryOp::Sub => {
            if left_has {
                // expr(v) - a = other => expr(v) = other + a
                let new_other = Expression::Binary(
                    BinaryOp::Add,
                    Box::new(other.clone()),
                    Box::new(right.clone()),
                )
                .simplify();
                let p = path.annotated_step(
                    Operation::AddBothSides(right.clone()),
                    format!("Add {} to both sides", right),
                    new_other.clone(),
                    StepAnnotation::elementary(),
                );
                unwrap_variable(left, &new_other, var, p)
            } else {
                // a - expr(v) = other => expr(v) = a - other
                let new_other = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(left.clone()),
                    Box::new(other.clone()),
                )
                .simplify();
                let p = path.annotated_step(
                    Operation::SubtractBothSides(other.clone()),
                    format!(
                        "Rearrange: {} - expr = other becomes expr = {} - other",
                        left, left
                    ),
                    new_other.clone(),
                    StepAnnotation::elementary(),
                );
                unwrap_variable(right, &new_other, var, p)
            }
        }

        // a * expr(v) or expr(v) * a => other / a
        BinaryOp::Mul => {
            let (var_child, const_child) = if left_has {
                (left, right)
            } else {
                (right, left)
            };
            let new_other = Expression::Binary(
                BinaryOp::Div,
                Box::new(other.clone()),
                Box::new(const_child.clone()),
            )
            .simplify();
            let p = path.annotated_step(
                Operation::DivideBothSides(const_child.clone()),
                format!("Divide both sides by {}", const_child),
                new_other.clone(),
                StepAnnotation::elementary(),
            );
            unwrap_variable(var_child, &new_other, var, p)
        }

        // expr(v) / a => other * a ; a / expr(v) => expr(v) = a / other
        BinaryOp::Div => {
            if left_has {
                // expr(v) / a = other => expr(v) = other * a
                let new_other = Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(other.clone()),
                    Box::new(right.clone()),
                )
                .simplify();
                let p = path.annotated_step(
                    Operation::MultiplyBothSides(right.clone()),
                    format!("Multiply both sides by {}", right),
                    new_other.clone(),
                    StepAnnotation::elementary(),
                );
                unwrap_variable(left, &new_other, var, p)
            } else {
                // a / expr(v) = other => expr(v) = a / other
                let new_other = Expression::Binary(
                    BinaryOp::Div,
                    Box::new(left.clone()),
                    Box::new(other.clone()),
                )
                .simplify();
                let p = path.annotated_step(
                    Operation::DivideBothSides(other.clone()),
                    format!(
                        "Rearrange: {} / expr = other becomes expr = {} / other",
                        left, left
                    ),
                    new_other.clone(),
                    StepAnnotation::elementary(),
                );
                unwrap_variable(right, &new_other, var, p)
            }
        }

        _ => Err(SolverError::CannotSolve(format!(
            "Cannot isolate '{}': unsupported binary operator {:?}",
            var, op
        ))),
    }
}

/// Handle power expressions during unwrapping.
fn unwrap_power(
    base: &Expression,
    exp: &Expression,
    other: &Expression,
    var: &str,
    path: ResolutionPathBuilder,
) -> Result<(Expression, ResolutionPathBuilder), SolverError> {
    let base_has = contains_variable(base, var);
    let exp_has = contains_variable(exp, var);

    if base_has && exp_has {
        return Err(SolverError::CannotSolve(format!(
            "Cannot isolate '{}': variable in both base and exponent",
            var
        )));
    }

    if base_has {
        // expr(v)^n = other => expr(v) = other^(1/n)
        let inv_exp = Expression::Binary(
            BinaryOp::Div,
            Box::new(Expression::Integer(1)),
            Box::new(exp.clone()),
        )
        .simplify();
        let new_other =
            Expression::Power(Box::new(other.clone()), Box::new(inv_exp.clone())).simplify();
        let p = path.annotated_step(
            Operation::RootBothSides(exp.clone()),
            format!("Take the {} root of both sides", exp),
            new_other.clone(),
            StepAnnotation::power_and_roots(),
        );
        unwrap_variable(base, &new_other, var, p)
    } else {
        // a^expr(v) = other => expr(v) = ln(other) / ln(a)
        let new_other = Expression::Binary(
            BinaryOp::Div,
            Box::new(Expression::Function(Function::Ln, vec![other.clone()])),
            Box::new(Expression::Function(Function::Ln, vec![base.clone()])),
        )
        .simplify();
        let p = path.annotated_step(
            Operation::ApplyFunction("log".to_string()),
            format!("Take logarithm base {} of both sides", base),
            new_other.clone(),
            StepAnnotation::power_and_roots(),
        );
        unwrap_variable(exp, &new_other, var, p)
    }
}

/// Handle function inversion during unwrapping.
fn unwrap_function(
    func: &Function,
    args: &[Expression],
    other: &Expression,
    var: &str,
    path: ResolutionPathBuilder,
) -> Result<(Expression, ResolutionPathBuilder), SolverError> {
    use super::calculus::is_calculus_wrapper;

    // Calculus wrappers (integral, sum, product, limit): try to factor variable out of body
    if is_calculus_wrapper(func) {
        return try_unwrap_calculus_wrapper(func, args, other, var, path);
    }

    // Multi-argument non-calculus functions: try to find variable in any argument
    if args.len() > 1 {
        // Check if exactly one argument contains the variable
        let var_arg_indices: Vec<usize> = args
            .iter()
            .enumerate()
            .filter(|(_, a)| contains_variable(a, var))
            .map(|(i, _)| i)
            .collect();

        if var_arg_indices.len() == 1 {
            let idx = var_arg_indices[0];
            let inner = &args[idx];

            if let Expression::Variable(v) = inner {
                if v.name == var {
                    return Err(SolverError::CannotSolve(format!(
                        "Cannot isolate '{}': function {:?} is not invertible",
                        var, func
                    )));
                }
            }
        }

        return Err(SolverError::CannotSolve(format!(
            "Cannot isolate '{}': multi-argument function {:?}",
            var, func
        )));
    }

    let inner = &args[0];
    if !contains_variable(inner, var) {
        return Err(SolverError::CannotSolve(format!(
            "Variable '{}' not found in function argument",
            var
        )));
    }

    let (new_other, desc) = match func {
        Function::Sin => (
            Expression::Function(Function::Asin, vec![other.clone()]),
            "Apply arcsin to both sides",
        ),
        Function::Cos => (
            Expression::Function(Function::Acos, vec![other.clone()]),
            "Apply arccos to both sides",
        ),
        Function::Tan => (
            Expression::Function(Function::Atan, vec![other.clone()]),
            "Apply arctan to both sides",
        ),
        Function::Asin => (
            Expression::Function(Function::Sin, vec![other.clone()]),
            "Apply sin to both sides",
        ),
        Function::Acos => (
            Expression::Function(Function::Cos, vec![other.clone()]),
            "Apply cos to both sides",
        ),
        Function::Atan => (
            Expression::Function(Function::Tan, vec![other.clone()]),
            "Apply tan to both sides",
        ),
        Function::Exp => (
            Expression::Function(Function::Ln, vec![other.clone()]),
            "Take natural log of both sides",
        ),
        Function::Ln => (
            Expression::Function(Function::Exp, vec![other.clone()]),
            "Exponentiate both sides",
        ),
        Function::Sqrt => (
            Expression::Power(Box::new(other.clone()), Box::new(Expression::Integer(2))),
            "Square both sides",
        ),
        Function::Cbrt => (
            Expression::Power(Box::new(other.clone()), Box::new(Expression::Integer(3))),
            "Cube both sides",
        ),
        _ => {
            return Err(SolverError::CannotSolve(format!(
                "Cannot isolate '{}': function {:?} is not invertible",
                var, func
            )));
        }
    };

    let simplified = new_other.simplify();
    let func_name = format!("{:?}", func);
    let annotation = match func {
        Function::Sin
        | Function::Cos
        | Function::Tan
        | Function::Asin
        | Function::Acos
        | Function::Atan => StepAnnotation::transcendental("Inverse Trigonometric Function"),
        Function::Exp | Function::Ln => StepAnnotation::power_and_roots(),
        Function::Sqrt | Function::Cbrt => StepAnnotation::power_and_roots(),
        _ => StepAnnotation::elementary(),
    };
    let p = path.annotated_step(
        Operation::ApplyFunction(func_name),
        desc.to_string(),
        simplified.clone(),
        annotation,
    );
    unwrap_variable(inner, &simplified, var, p)
}
