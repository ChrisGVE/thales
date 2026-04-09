//! General symbolic isolation engine for equation rearrangement.
//!
//! Implements a recursive inverse-unwrapping algorithm that can isolate any
//! variable appearing linearly (exactly once) in an equation, handling
//! arithmetic, powers, and invertible functions.

use crate::ast::{BinaryOp, Expression, Function, UnaryOp, Variable};
use crate::resolution_path::{Operation, ResolutionPathBuilder, StepAnnotation};

use super::helpers::contains_variable;
use super::types::SolverError;

/// Attempt to symbolically isolate the target variable in the equation.
///
/// Returns the expression that the variable equals, plus the updated path
/// builder. Works by recursively "peeling off" operations wrapping the
/// variable and applying their inverses to the other side.
pub fn symbolic_isolate(
    lhs: &Expression,
    rhs: &Expression,
    variable: &Variable,
    path: ResolutionPathBuilder,
) -> Result<(Expression, ResolutionPathBuilder), SolverError> {
    let var = &variable.name;
    let left_has = contains_variable(lhs, var);
    let right_has = contains_variable(rhs, var);

    if !left_has && !right_has {
        return Err(SolverError::CannotSolve(format!(
            "Variable '{}' not found in equation",
            var
        )));
    }

    // Determine which side contains the variable
    let (var_side, other_side) = if left_has && !right_has {
        (lhs.clone(), rhs.clone())
    } else if right_has && !left_has {
        (rhs.clone(), lhs.clone())
    } else {
        // Variable on both sides: move everything to the left
        let combined =
            Expression::Binary(BinaryOp::Sub, Box::new(lhs.clone()), Box::new(rhs.clone()))
                .simplify();
        (combined, Expression::Integer(0))
    };

    unwrap_variable(&var_side, &other_side, var, path)
}

/// Count how many times a variable appears in an expression.
fn count_variable(expr: &Expression, var: &str) -> usize {
    match expr {
        Expression::Variable(v) if v.name == var => 1,
        Expression::Variable(_)
        | Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Rational(_)
        | Expression::Complex(_)
        | Expression::Constant(_) => 0,
        Expression::Unary(_, inner) => count_variable(inner, var),
        Expression::Binary(_, left, right) => {
            count_variable(left, var) + count_variable(right, var)
        }
        Expression::Power(base, exp) => count_variable(base, var) + count_variable(exp, var),
        Expression::Function(_, args) => args.iter().map(|a| count_variable(a, var)).sum(),
    }
}

/// Recursively peel off operations wrapping the target variable, applying
/// inverse operations to `other` (the accumulating other-side expression).
fn unwrap_variable(
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

    // Variable in both children — try to collect linear terms
    if left_has && right_has {
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

/// Check whether a function is a calculus wrapper (integral, sum, product, limit).
fn is_calculus_wrapper(func: &Function) -> bool {
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
fn try_split_product(expr: &Expression, var: &str) -> Option<(Expression, Expression)> {
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

    // Multiplication: try to split into var-containing and non-var parts
    if let Expression::Binary(BinaryOp::Mul, left, right) = expr {
        let left_has = contains_variable(left, var);
        let right_has = contains_variable(right, var);

        if left_has && !right_has {
            // var_factor is in left, rest is right
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
            // var_factor is in right, rest is left
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

    // Division: var_factor / something
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
        }
    }

    None
}

/// Try to handle a calculus wrapper by factoring the target variable out of its body.
///
/// For `integral(F*cos(theta), x)` solving for F:
/// → factor F out of integrand → F * integral(cos(theta), x)
/// → then standard algebraic isolation handles F = other / integral(cos(theta), x)
fn try_unwrap_calculus_wrapper(
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
        // Build a multiplication expression and let unwrap_variable handle it
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
            // integral(F, x) = other → F * integral(1, x) = other
            // But integral(1, x) is just x (or bounds difference), which is opaque.
            // Simpler: treat the whole wrapper as coefficient.
            // F = other / integral(1, metadata...)
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

/// Handle function inversion during unwrapping.
fn unwrap_function(
    func: &Function,
    args: &[Expression],
    other: &Expression,
    var: &str,
    path: ResolutionPathBuilder,
) -> Result<(Expression, ResolutionPathBuilder), SolverError> {
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
            // Variable in exactly one argument — treat the function as opaque
            // and attempt to isolate within that argument by treating the whole
            // function as a wrapper around its single variable-bearing argument.
            // This won't always work, but it's better than giving up.
            let idx = var_arg_indices[0];
            let inner = &args[idx];

            // For now, only handle when the variable IS the argument directly
            if let Expression::Variable(v) = inner {
                if v.name == var {
                    // The function wraps the variable directly — it's not invertible
                    // but we can still isolate if the function result is used algebraically.
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

/// Attempt to collect linear terms when the variable appears on both sides
/// of a binary operation.
///
/// Handles patterns like `a*v + b*v = (a+b)*v` and `a*v - b*v = (a-b)*v`.
fn collect_linear_terms(
    op: BinaryOp,
    left: &Expression,
    right: &Expression,
    other: &Expression,
    var: &str,
    path: ResolutionPathBuilder,
) -> Result<(Expression, ResolutionPathBuilder), SolverError> {
    // Only addition and subtraction can have linear collection
    if op != BinaryOp::Add && op != BinaryOp::Sub {
        return Err(SolverError::CannotSolve(format!(
            "Cannot isolate '{}': variable appears in both operands of {:?}",
            var, op
        )));
    }

    // Try to extract coefficients: left = coeff_l * var, right = coeff_r * var
    let coeff_l = try_extract_linear_coeff(left, var);
    let coeff_r = try_extract_linear_coeff(right, var);

    match (coeff_l, coeff_r) {
        (Some(cl), Some(cr)) => {
            // Combine: (cl +/- cr) * var = other
            let combined_coeff = match op {
                BinaryOp::Add => {
                    Expression::Binary(BinaryOp::Add, Box::new(cl), Box::new(cr)).simplify()
                }
                BinaryOp::Sub => {
                    Expression::Binary(BinaryOp::Sub, Box::new(cl), Box::new(cr)).simplify()
                }
                _ => unreachable!(),
            };
            let new_other = Expression::Binary(
                BinaryOp::Div,
                Box::new(other.clone()),
                Box::new(combined_coeff.clone()),
            )
            .simplify();
            let p = path.annotated_step(
                Operation::DivideBothSides(combined_coeff),
                format!("Collect terms and divide to isolate {}", var),
                new_other.clone(),
                StepAnnotation::elementary(),
            );
            Ok((new_other, p))
        }
        _ => Err(SolverError::CannotSolve(format!(
            "Cannot isolate '{}': variable appears non-linearly in both operands",
            var
        ))),
    }
}

/// Try to extract the coefficient of a variable from an expression that is
/// a linear multiple of the variable.
///
/// Returns `Some(coeff)` if `expr = coeff * var`, `None` otherwise.
fn try_extract_linear_coeff(expr: &Expression, var: &str) -> Option<Expression> {
    if count_variable(expr, var) != 1 {
        return None;
    }

    match expr {
        Expression::Variable(v) if v.name == var => Some(Expression::Integer(1)),

        Expression::Unary(UnaryOp::Neg, inner) => try_extract_linear_coeff(inner, var)
            .map(|c| Expression::Unary(UnaryOp::Neg, Box::new(c)).simplify()),

        Expression::Binary(BinaryOp::Mul, left, right) => {
            let left_has = contains_variable(left, var);
            let right_has = contains_variable(right, var);
            if left_has && !right_has {
                try_extract_linear_coeff(left, var).map(|c| {
                    Expression::Binary(BinaryOp::Mul, Box::new(c), Box::new(right.as_ref().clone()))
                        .simplify()
                })
            } else if right_has && !left_has {
                try_extract_linear_coeff(right, var).map(|c| {
                    Expression::Binary(BinaryOp::Mul, Box::new(left.as_ref().clone()), Box::new(c))
                        .simplify()
                })
            } else {
                None
            }
        }

        Expression::Binary(BinaryOp::Div, left, right) => {
            if contains_variable(right, var) {
                None // var in denominator is not linear
            } else {
                try_extract_linear_coeff(left, var).map(|c| {
                    Expression::Binary(BinaryOp::Div, Box::new(c), Box::new(right.as_ref().clone()))
                        .simplify()
                })
            }
        }

        _ => None,
    }
}

#[cfg(test)]
mod tests;
