//! Rational equation solving via denominator clearing.
//!
//! When the target variable appears in a denominator, this module attempts to
//! clear denominators by multiplying through, converting the equation into a
//! polynomial form that can be solved by the standard isolation engine.

use crate::ast::{BinaryOp, Expression, UnaryOp};
use crate::resolution_path::{Operation, ResolutionPathBuilder, StepAnnotation};

use super::super::helpers::contains_variable;
use super::super::types::SolverError;
use super::unwrap::unwrap_variable;

/// Check whether a variable appears in any denominator within an expression.
fn has_var_in_denominator(expr: &Expression, var: &str) -> bool {
    match expr {
        Expression::Binary(BinaryOp::Div, _, right) => {
            if contains_variable(right, var) {
                return true;
            }
            // Also check recursively within both children
            has_var_in_denominator_recursive(expr, var)
        }
        _ => has_var_in_denominator_recursive(expr, var),
    }
}

/// Recursively check all sub-expressions for variable-in-denominator.
fn has_var_in_denominator_recursive(expr: &Expression, var: &str) -> bool {
    match expr {
        Expression::Variable(_)
        | Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Rational(_)
        | Expression::Complex(_)
        | Expression::Constant(_) => false,
        Expression::Unary(_, inner) => has_var_in_denominator(inner, var),
        Expression::Binary(BinaryOp::Div, left, right) => {
            contains_variable(right, var)
                || has_var_in_denominator(left, var)
                || has_var_in_denominator(right, var)
        }
        Expression::Binary(_, left, right) => {
            has_var_in_denominator(left, var) || has_var_in_denominator(right, var)
        }
        Expression::Power(base, exp) => {
            has_var_in_denominator(base, var) || has_var_in_denominator(exp, var)
        }
        Expression::Function(_, args) => args.iter().any(|a| has_var_in_denominator(a, var)),
    }
}

/// Extract the denominator expression containing the variable from a
/// division node. Returns the first denominator found that contains the
/// variable.
fn extract_var_denominator(expr: &Expression, var: &str) -> Option<Expression> {
    match expr {
        Expression::Binary(BinaryOp::Div, left, right) => {
            if contains_variable(right, var) {
                Some(right.as_ref().clone())
            } else {
                // Search deeper in numerator
                extract_var_denominator(left, var)
            }
        }
        Expression::Binary(_, left, right) => {
            extract_var_denominator(left, var).or_else(|| extract_var_denominator(right, var))
        }
        Expression::Unary(_, inner) => extract_var_denominator(inner, var),
        Expression::Power(base, _) => extract_var_denominator(base, var),
        Expression::Function(_, args) => args.iter().find_map(|a| extract_var_denominator(a, var)),
        _ => None,
    }
}

/// Try to clear denominators when the variable appears in a denominator.
///
/// For `a*var - b/(c*var) = other`, multiply everything by `c*var`:
/// `a*var*(c*var) - b = other*(c*var)` → polynomial in var.
///
/// Returns `None` if no denominator clearing is applicable. Returns
/// `Some(result)` with the isolation result if clearing succeeds.
pub(super) fn try_clear_denominators(
    op: BinaryOp,
    left: &Expression,
    right: &Expression,
    other: &Expression,
    var: &str,
    path: &ResolutionPathBuilder,
) -> Option<Result<(Expression, ResolutionPathBuilder), SolverError>> {
    // Check if either child has the variable in a denominator
    let left_denom = has_var_in_denominator(left, var);
    let right_denom = has_var_in_denominator(right, var);

    if !left_denom && !right_denom {
        return None;
    }

    // Extract the denominator to multiply through
    let denom = if left_denom {
        extract_var_denominator(left, var)
    } else {
        extract_var_denominator(right, var)
    };

    let denom = match denom {
        Some(d) => d,
        None => return None,
    };

    // Build the cleared equation: (left op right) * denom = other * denom
    // We multiply each part individually to help simplification
    let new_left = Expression::Binary(
        BinaryOp::Mul,
        Box::new(left.clone()),
        Box::new(denom.clone()),
    )
    .simplify();

    let new_right = Expression::Binary(
        BinaryOp::Mul,
        Box::new(right.clone()),
        Box::new(denom.clone()),
    )
    .simplify();

    let new_other = Expression::Binary(
        BinaryOp::Mul,
        Box::new(other.clone()),
        Box::new(denom.clone()),
    )
    .simplify();

    // Guard against infinite recursion: if the cleared expression still has
    // the variable in a denominator, bail out
    let var_side =
        Expression::Binary(op, Box::new(new_left.clone()), Box::new(new_right.clone())).simplify();

    if has_var_in_denominator(&var_side, var) || has_var_in_denominator(&new_other, var) {
        return None;
    }

    let p = path.clone().annotated_step(
        Operation::MultiplyBothSides(denom.clone()),
        format!("Multiply both sides by {} to clear denominator", denom),
        new_other.clone(),
        StepAnnotation::elementary(),
    );

    Some(unwrap_variable(&var_side, &new_other, var, p))
}

/// Cross-multiply when the variable appears in both numerator and denominator
/// of a division: `f(v)/g(v) = other` → `f(v) - other*g(v) = 0`.
///
/// After cross-multiplication and expansion, the result is a sum of terms.
/// Terms containing the variable are collected on one side, constant terms on
/// the other, and the variable is factored out: `v * coeff_sum = const_sum`.
pub(super) fn try_cross_multiply(
    numerator: &Expression,
    denominator: &Expression,
    other: &Expression,
    var: &str,
    path: ResolutionPathBuilder,
) -> Result<(Expression, ResolutionPathBuilder), SolverError> {
    // Cross-multiply: f(v)/g(v) = other → f(v) = other * g(v)
    // Rearrange to: f(v) - other * g(v) = 0
    let other_times_denom = expand(&Expression::Binary(
        BinaryOp::Mul,
        Box::new(other.clone()),
        Box::new(denominator.clone()),
    ));

    let cleared = Expression::Binary(
        BinaryOp::Sub,
        Box::new(expand(numerator)),
        Box::new(other_times_denom),
    )
    .simplify();

    // Guard: if the cleared expression still has var in a denominator, bail out
    if has_var_in_denominator(&cleared, var) {
        return Err(SolverError::CannotSolve(format!(
            "Cannot isolate '{}': cross-multiplication did not eliminate all denominators",
            var
        )));
    }

    let p = path.annotated_step(
        Operation::MultiplyBothSides(denominator.clone()),
        format!(
            "Cross-multiply: multiply both sides by {} to clear denominator",
            denominator
        ),
        Expression::Integer(0),
        StepAnnotation::elementary(),
    );

    // Flatten the cleared expression into additive terms (with signs),
    // then separate into var-containing and constant terms.
    let mut var_terms: Vec<Expression> = Vec::new();
    let mut const_terms: Vec<Expression> = Vec::new();
    collect_additive_terms(&cleared, true, &mut var_terms, &mut const_terms, var);

    if var_terms.is_empty() {
        return Err(SolverError::CannotSolve(format!(
            "Cannot isolate '{}': variable disappeared after cross-multiplication",
            var
        )));
    }

    // Extract the coefficient of var from each var-containing term.
    // If any term is nonlinear in var, fall back to unwrap_variable.
    let mut coefficients: Vec<Expression> = Vec::new();
    for term in &var_terms {
        match try_extract_var_coeff(term, var) {
            Some(coeff) => coefficients.push(coeff),
            None => {
                // Nonlinear in var — fall back to general unwrap
                return unwrap_variable(&cleared, &Expression::Integer(0), var, p);
            }
        }
    }

    // Sum the coefficients: var * (c1 + c2 + ...) = -(const terms)
    let coeff_sum = coefficients
        .into_iter()
        .reduce(|acc, c| Expression::Binary(BinaryOp::Add, Box::new(acc), Box::new(c)))
        .unwrap()
        .simplify();

    // The constant side: negate and sum (move to RHS)
    let const_sum = const_terms
        .into_iter()
        .reduce(|acc, c| Expression::Binary(BinaryOp::Add, Box::new(acc), Box::new(c)))
        .map(|s| Expression::Unary(UnaryOp::Neg, Box::new(s)).simplify())
        .unwrap_or(Expression::Integer(0));

    // var = const_sum / coeff_sum
    let result = Expression::Binary(
        BinaryOp::Div,
        Box::new(const_sum.clone()),
        Box::new(coeff_sum.clone()),
    )
    .simplify();

    let p = p.annotated_step(
        Operation::DivideBothSides(coeff_sum),
        format!("Collect terms and divide to isolate {}", var),
        result.clone(),
        StepAnnotation::elementary(),
    );

    Ok((result, p))
}

/// Flatten an expression into additive terms, tracking sign.
/// Positive terms go into `var_terms` or `const_terms` depending on whether
/// they contain the variable. Negative terms are negated and added with
/// flipped sign.
fn collect_additive_terms(
    expr: &Expression,
    positive: bool,
    var_terms: &mut Vec<Expression>,
    const_terms: &mut Vec<Expression>,
    var: &str,
) {
    match expr {
        Expression::Binary(BinaryOp::Add, left, right) => {
            collect_additive_terms(left, positive, var_terms, const_terms, var);
            collect_additive_terms(right, positive, var_terms, const_terms, var);
        }
        Expression::Binary(BinaryOp::Sub, left, right) => {
            collect_additive_terms(left, positive, var_terms, const_terms, var);
            collect_additive_terms(right, !positive, var_terms, const_terms, var);
        }
        Expression::Unary(UnaryOp::Neg, inner) => {
            collect_additive_terms(inner, !positive, var_terms, const_terms, var);
        }
        _ => {
            let term = if positive {
                expr.clone()
            } else {
                Expression::Unary(UnaryOp::Neg, Box::new(expr.clone())).simplify()
            };
            if contains_variable(&term, var) {
                var_terms.push(term);
            } else {
                const_terms.push(term);
            }
        }
    }
}

/// Try to extract the coefficient of `var` from an expression that is a
/// single-variable linear term: `coeff * var` → `Some(coeff)`.
/// Returns `None` if the expression is nonlinear in `var`.
fn try_extract_var_coeff(expr: &Expression, var: &str) -> Option<Expression> {
    match expr {
        Expression::Variable(v) if v.name == var => Some(Expression::Integer(1)),
        Expression::Unary(UnaryOp::Neg, inner) => try_extract_var_coeff(inner, var)
            .map(|c| Expression::Unary(UnaryOp::Neg, Box::new(c)).simplify()),
        Expression::Binary(BinaryOp::Mul, left, right) => {
            let left_has = contains_variable(left, var);
            let right_has = contains_variable(right, var);
            if left_has && !right_has {
                try_extract_var_coeff(left, var).map(|c| {
                    Expression::Binary(BinaryOp::Mul, Box::new(c), right.clone()).simplify()
                })
            } else if right_has && !left_has {
                try_extract_var_coeff(right, var).map(|c| {
                    Expression::Binary(BinaryOp::Mul, left.clone(), Box::new(c)).simplify()
                })
            } else {
                None // var in both factors — nonlinear
            }
        }
        Expression::Binary(BinaryOp::Div, left, right) => {
            if contains_variable(right, var) {
                None
            } else {
                try_extract_var_coeff(left, var).map(|c| {
                    Expression::Binary(BinaryOp::Div, Box::new(c), right.clone()).simplify()
                })
            }
        }
        _ => None,
    }
}

/// Distribute multiplication over addition and subtraction.
///
/// Recursively expands products: `a * (b + c)` → `a*b + a*c`,
/// `(a + b) * c` → `a*c + b*c`, and similarly for subtraction.
/// The result is simplified after expansion.
fn expand(expr: &Expression) -> Expression {
    match expr {
        Expression::Binary(BinaryOp::Mul, left, right) => {
            let l = expand(left);
            let r = expand(right);
            expand_product(&l, &r).simplify()
        }
        Expression::Binary(op, left, right) => {
            let l = expand(left);
            let r = expand(right);
            Expression::Binary(*op, Box::new(l), Box::new(r)).simplify()
        }
        Expression::Unary(UnaryOp::Neg, inner) => {
            let expanded = expand(inner);
            Expression::Unary(UnaryOp::Neg, Box::new(expanded)).simplify()
        }
        _ => expr.clone(),
    }
}

/// Expand the product of two expressions, distributing over Add and Sub.
fn expand_product(left: &Expression, right: &Expression) -> Expression {
    match (left, right) {
        // a * (b + c) → a*b + a*c
        (_, Expression::Binary(BinaryOp::Add, rb, rc)) => Expression::Binary(
            BinaryOp::Add,
            Box::new(expand_product(left, rb)),
            Box::new(expand_product(left, rc)),
        ),
        // a * (b - c) → a*b - a*c
        (_, Expression::Binary(BinaryOp::Sub, rb, rc)) => Expression::Binary(
            BinaryOp::Sub,
            Box::new(expand_product(left, rb)),
            Box::new(expand_product(left, rc)),
        ),
        // (a + b) * c → a*c + b*c
        (Expression::Binary(BinaryOp::Add, la, lb), _) => Expression::Binary(
            BinaryOp::Add,
            Box::new(expand_product(la, right)),
            Box::new(expand_product(lb, right)),
        ),
        // (a - b) * c → a*c - b*c
        (Expression::Binary(BinaryOp::Sub, la, lb), _) => Expression::Binary(
            BinaryOp::Sub,
            Box::new(expand_product(la, right)),
            Box::new(expand_product(lb, right)),
        ),
        // a * (-b) → -(a*b) for further expansion
        (_, Expression::Unary(UnaryOp::Neg, inner)) => {
            Expression::Unary(UnaryOp::Neg, Box::new(expand_product(left, inner)))
        }
        // (-a) * b → -(a*b)
        (Expression::Unary(UnaryOp::Neg, inner), _) => {
            Expression::Unary(UnaryOp::Neg, Box::new(expand_product(inner, right)))
        }
        // Base case: no distribution possible
        _ => Expression::Binary(
            BinaryOp::Mul,
            Box::new(left.clone()),
            Box::new(right.clone()),
        ),
    }
}
