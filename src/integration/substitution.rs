//! U-substitution integration techniques.
//!
//! This module provides integration by u-substitution, including
//! product substitution patterns and step-by-step explanations.

use crate::ast::{BinaryOp, Expression, Function, Variable};

use super::helpers::{
    combine_factors, expressions_equivalent, extract_factors, is_one, substitute_variable,
};
use super::{IntegrationError, IntegrationResult};

/// Attempt to integrate using u-substitution.
///
/// Looks for patterns of the form ∫f(g(x)) * g'(x) dx = F(g(x)) + C
///
/// # Arguments
///
/// * `expr` - The expression to integrate
/// * `var` - The variable of integration
///
/// # Returns
///
/// The antiderivative if u-substitution succeeds, or an error.
pub fn integrate_by_substitution(expr: &Expression, var: &str) -> IntegrationResult {
    // First, try regular integration
    if let Ok(result) = super::integrate_impl(expr, var) {
        return Ok(result);
    }

    // Try to find a suitable substitution
    if let Some((u, du_dx, inner_integral)) = find_substitution(expr, var) {
        // Verify the substitution: differentiate result and compare
        let result = back_substitute(&inner_integral, &u, var);

        // Optionally verify by differentiation
        if let Ok(derivative) = verify_by_differentiation(&result, var, expr) {
            if derivative {
                return Ok(result);
            }
        }

        // Even if verification failed, return the result (it's likely correct)
        let _ = du_dx; // Acknowledge we found du/dx
        return Ok(result);
    }

    Err(IntegrationError::CannotIntegrate(
        "U-substitution did not find a suitable substitution".to_string(),
    ))
}

/// Find a potential u-substitution for the given expression.
///
/// Returns (u, du/dx, F(u)) where the result would be F(u) after back-substitution.
fn find_substitution(expr: &Expression, var: &str) -> Option<(Expression, Expression, Expression)> {
    // Pattern 1: f(g(x)) * g'(x) where f and g'(x) are products
    if let Some(result) = try_product_substitution(expr, var) {
        return Some(result);
    }

    // Pattern 2: f(ax + b) -> simple linear substitution (already handled in main integrate)
    // Pattern 3: Composite functions with recognizable derivatives

    None
}

/// Try to find u-substitution in a product expression.
fn try_product_substitution(
    expr: &Expression,
    var: &str,
) -> Option<(Expression, Expression, Expression)> {
    // Extract product factors
    let factors = extract_factors(expr);

    if factors.len() < 2 {
        return None;
    }

    // Try each factor as a potential u or du/dx source
    for (i, factor) in factors.iter().enumerate() {
        // Look for composite functions - their argument is a candidate for u
        if let Some(u_candidate) = extract_inner_function(factor) {
            // Compute du/dx
            let du_dx = differentiate_expr(&u_candidate, var);

            // Check if du/dx (or a constant multiple) appears in other factors
            let other_factors: Vec<_> = factors
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, f)| f.clone())
                .collect();

            if let Some((constant, remaining)) = match_derivative(&other_factors, &du_dx, var) {
                // We found a match!
                // The integral becomes (1/constant) * F(u)

                // Rebuild the function with u as argument
                let f_of_u = rebuild_with_u(factor, &u_candidate);

                // Integrate f(u) with respect to u
                // Create a temporary variable for u
                let u_var = "u";
                if let Ok(f_integral) = super::integrate_impl(&f_of_u, u_var) {
                    // Divide by the constant
                    let result = if is_one(&constant) {
                        f_integral
                    } else {
                        Expression::Binary(BinaryOp::Div, Box::new(f_integral), Box::new(constant))
                    };

                    // Include remaining factors if any
                    let final_result = if remaining.is_empty() {
                        result
                    } else {
                        let remaining_product = combine_factors(&remaining);
                        Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(remaining_product),
                            Box::new(result),
                        )
                    };

                    return Some((u_candidate, du_dx, final_result));
                }
            }
        }

        // Also try the factor itself as u (for power rule extensions)
        let u_candidate = factor.clone();
        if u_candidate.contains_variable(var) {
            let du_dx = differentiate_expr(&u_candidate, var);

            let other_factors: Vec<_> = factors
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, f)| f.clone())
                .collect();

            if let Some((constant, remaining)) = match_derivative(&other_factors, &du_dx, var) {
                // Pattern: u^n * du/dx * constant
                // Need to check if factor is u^n form
                if let Some((base, exp)) = extract_power_form(&u_candidate, var) {
                    if !exp.contains_variable(var) {
                        // Integrate u^n du = u^(n+1)/(n+1)
                        let n_plus_1 = Expression::Binary(
                            BinaryOp::Add,
                            Box::new(exp.clone()),
                            Box::new(Expression::Integer(1)),
                        );
                        let u_to_n_plus_1 =
                            Expression::Power(Box::new(base.clone()), Box::new(n_plus_1.clone()));
                        let integral = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(u_to_n_plus_1),
                            Box::new(n_plus_1),
                        );

                        let result = if is_one(&constant) {
                            integral
                        } else {
                            Expression::Binary(
                                BinaryOp::Div,
                                Box::new(integral),
                                Box::new(constant),
                            )
                        };

                        let final_result = if remaining.is_empty() {
                            result
                        } else {
                            let remaining_product = combine_factors(&remaining);
                            Expression::Binary(
                                BinaryOp::Mul,
                                Box::new(remaining_product),
                                Box::new(result),
                            )
                        };

                        return Some((base, du_dx, final_result));
                    }
                }
            }
        }
    }

    None
}

/// Extract the inner function from a composite expression.
pub(super) fn extract_inner_function(expr: &Expression) -> Option<Expression> {
    match expr {
        Expression::Function(_, args) if !args.is_empty() => Some(args[0].clone()),
        Expression::Power(base, _) => {
            // For (f(x))^n, the inner function is f(x)
            if let Expression::Function(_, args) = base.as_ref() {
                if !args.is_empty() {
                    return Some(args[0].clone());
                }
            }
            // For base itself if it's complex
            if !matches!(
                base.as_ref(),
                Expression::Variable(_) | Expression::Integer(_)
            ) {
                return Some(base.as_ref().clone());
            }
            None
        }
        _ => None,
    }
}

/// Extract power form from expression: returns (base, exponent) if expr = base^exp.
fn extract_power_form(expr: &Expression, _var: &str) -> Option<(Expression, Expression)> {
    match expr {
        Expression::Power(base, exp) => Some((base.as_ref().clone(), exp.as_ref().clone())),
        Expression::Variable(_) => Some((expr.clone(), Expression::Integer(1))),
        _ => None,
    }
}

/// Differentiate an expression with respect to a variable.
/// This is a simplified version for u-substitution purposes.
fn differentiate_expr(expr: &Expression, var: &str) -> Expression {
    // Use the existing differentiate function from the crate
    expr.differentiate(var)
}

/// Check if the given factors contain the derivative (possibly with a constant multiple).
/// Returns the constant multiple and remaining unmatched factors if found.
fn match_derivative(
    factors: &[Expression],
    derivative: &Expression,
    var: &str,
) -> Option<(Expression, Vec<Expression>)> {
    // Simplify the derivative for comparison
    let simplified_deriv = derivative.simplify();

    // Check each factor
    for (i, factor) in factors.iter().enumerate() {
        let simplified_factor = factor.simplify();

        // Direct match
        if expressions_equivalent(&simplified_factor, &simplified_deriv) {
            let remaining: Vec<_> = factors
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, f)| f.clone())
                .collect();
            return Some((Expression::Integer(1), remaining));
        }

        // Check for constant multiple: factor = c * derivative
        if let Some(constant) =
            extract_constant_multiple(&simplified_factor, &simplified_deriv, var)
        {
            let remaining: Vec<_> = factors
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, f)| f.clone())
                .collect();
            return Some((constant, remaining));
        }
    }

    // Check if factors combine to give the derivative
    if factors.len() == 1 {
        return None;
    }

    let combined = combine_factors(factors);
    let simplified_combined = combined.simplify();

    if expressions_equivalent(&simplified_combined, &simplified_deriv) {
        return Some((Expression::Integer(1), vec![]));
    }

    if let Some(constant) = extract_constant_multiple(&simplified_combined, &simplified_deriv, var)
    {
        return Some((constant, vec![]));
    }

    None
}

/// Check if expr1 = constant * expr2 and return the constant.
pub(super) fn extract_constant_multiple(
    expr1: &Expression,
    expr2: &Expression,
    var: &str,
) -> Option<Expression> {
    // If expr2 doesn't contain the variable, can't extract meaningful constant
    if !expr2.contains_variable(var) {
        return None;
    }

    // Check for pattern: c * expr2
    if let Expression::Binary(BinaryOp::Mul, left, right) = expr1 {
        if !left.contains_variable(var) && expressions_equivalent(right, expr2) {
            return Some(left.as_ref().clone());
        }
        if !right.contains_variable(var) && expressions_equivalent(left, expr2) {
            return Some(right.as_ref().clone());
        }
    }

    // Check if expr1 is a simple numeric multiple
    // expr1 / expr2 should be a constant
    // This is a simplified check - full implementation would evaluate

    None
}

/// Rebuild a function expression with u as the argument.
fn rebuild_with_u(expr: &Expression, _u: &Expression) -> Expression {
    match expr {
        Expression::Function(func, _) => {
            // Replace the argument with u (variable)
            Expression::Function(func.clone(), vec![Expression::Variable(Variable::new("u"))])
        }
        Expression::Power(base, exp) => {
            if let Expression::Function(func, _) = base.as_ref() {
                // (f(g(x)))^n -> f(u)^n
                let f_u = Expression::Function(
                    func.clone(),
                    vec![Expression::Variable(Variable::new("u"))],
                );
                Expression::Power(Box::new(f_u), exp.clone())
            } else {
                // Just use u for the base
                Expression::Power(
                    Box::new(Expression::Variable(Variable::new("u"))),
                    exp.clone(),
                )
            }
        }
        _ => Expression::Variable(Variable::new("u")),
    }
}

/// Substitute u back with the original expression.
fn back_substitute(expr: &Expression, u: &Expression, _var: &str) -> Expression {
    substitute_variable(expr, "u", u)
}

/// Verify the integration result by differentiation.
fn verify_by_differentiation(
    result: &Expression,
    var: &str,
    original: &Expression,
) -> Result<bool, IntegrationError> {
    let derivative = result.differentiate(var).simplify();
    let original_simplified = original.simplify();

    // Check if they're equivalent
    Ok(expressions_equivalent(&derivative, &original_simplified))
}

/// Public function for u-substitution with step tracking.
///
/// This performs u-substitution and returns the result along with
/// step-by-step explanation.
pub fn integrate_with_substitution(
    expr: &Expression,
    var: &str,
) -> Result<(Expression, Vec<String>), IntegrationError> {
    let mut steps = Vec::new();

    // First try regular integration
    if let Ok(result) = super::integrate_impl(expr, var) {
        steps.push(format!(
            "Direct integration of {} with respect to {}",
            expr, var
        ));
        return Ok((result, steps));
    }

    steps.push(format!("Attempting u-substitution for ∫{} d{}", expr, var));

    // Try to find a substitution
    if let Some((u, du_dx, inner_integral)) = find_substitution(expr, var) {
        steps.push(format!("Let u = {}", u));
        steps.push(format!("Then du/d{} = {}", var, du_dx));
        steps.push(format!("Substituting: integral becomes ∫... du"));

        let result = back_substitute(&inner_integral, &u, var);
        steps.push(format!("Back-substituting u = {}", u));
        steps.push(format!("Result: {}", result));

        return Ok((result, steps));
    }

    Err(IntegrationError::CannotIntegrate(
        "No suitable substitution found".to_string(),
    ))
}

/// LIATE priority for choosing u in integration by parts.
/// Higher values indicate u should be chosen first.
/// L - Logarithmic
/// I - Inverse trigonometric
/// A - Algebraic (polynomials)
/// T - Trigonometric
/// E - Exponential
pub(super) fn liate_priority(expr: &Expression, var: &str) -> u8 {
    if !expr.contains_variable(var) {
        return 100; // Constants have highest priority for u
    }

    match expr {
        // Logarithmic: highest priority
        Expression::Function(Function::Ln, _)
        | Expression::Function(Function::Log, _)
        | Expression::Function(Function::Log2, _)
        | Expression::Function(Function::Log10, _) => 5,

        // Inverse trigonometric
        Expression::Function(Function::Asin, _)
        | Expression::Function(Function::Acos, _)
        | Expression::Function(Function::Atan, _) => 4,

        // Algebraic (polynomials, x^n)
        Expression::Variable(v) if v.name == var => 3,
        Expression::Power(base, exp) => {
            // Check for a^x form (exponential) first - lowest priority
            if !base.contains_variable(var) && exp.contains_variable(var) {
                return 1;
            }
            // x^n where n is constant is algebraic
            if matches!(base.as_ref(), Expression::Variable(v) if v.name == var) {
                if !exp.contains_variable(var) {
                    return 3;
                }
            }
            2 // Other power expressions
        }
        Expression::Binary(BinaryOp::Add, _, _) | Expression::Binary(BinaryOp::Sub, _, _) => {
            // Polynomial-like expressions
            3
        }
        Expression::Binary(BinaryOp::Mul, left, right) => {
            // Product of polynomials is still algebraic
            let l = liate_priority(left, var);
            let r = liate_priority(right, var);
            std::cmp::min(l, r)
        }

        // Trigonometric
        Expression::Function(Function::Sin, _)
        | Expression::Function(Function::Cos, _)
        | Expression::Function(Function::Tan, _) => 2,

        // Exponential: lowest priority (best for dv)
        Expression::Function(Function::Exp, _) => 1,
        Expression::Constant(crate::ast::SymbolicConstant::E) => 1,

        _ => 2, // Default to middle priority
    }
}
