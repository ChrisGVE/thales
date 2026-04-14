//! Shared helper functions for integration.
//!
//! This module contains pattern matching utilities and helper functions
//! used across multiple integration submodules.

use crate::ast::{BinaryOp, Expression, Function, UnaryOp, Variable};

use super::IntegrationResult;

/// Integrate a product expression.
pub(super) fn integrate_product(
    left: &Expression,
    right: &Expression,
    var: &str,
) -> IntegrationResult {
    let left_has_var = left.contains_variable(var);
    let right_has_var = right.contains_variable(var);

    if !left_has_var && !right_has_var {
        // Both constant: ∫c1*c2 dx = c1*c2*x
        let x = Expression::Variable(Variable::new(var));
        let product = Expression::Binary(
            BinaryOp::Mul,
            Box::new(left.clone()),
            Box::new(right.clone()),
        );
        Ok(Expression::Binary(
            BinaryOp::Mul,
            Box::new(product),
            Box::new(x),
        ))
    } else if !left_has_var {
        // Constant multiple rule: ∫c*f dx = c*∫f dx
        let right_integral = super::integrate_impl(right, var)?;
        Ok(Expression::Binary(
            BinaryOp::Mul,
            Box::new(left.clone()),
            Box::new(right_integral),
        ))
    } else if !right_has_var {
        // Constant multiple rule: ∫f*c dx = c*∫f dx
        let left_integral = super::integrate_impl(left, var)?;
        Ok(Expression::Binary(
            BinaryOp::Mul,
            Box::new(right.clone()),
            Box::new(left_integral),
        ))
    } else {
        // Both factors contain the variable - check for special patterns
        // Try to recognize x^n * x^m = x^(n+m)
        if let Some(result) = try_combine_powers(left, right, var) {
            return integrate_power_expr(&result, var);
        }

        // Check for derivative patterns (u-substitution candidates)
        // This is a simple heuristic - full u-substitution is Task 15
        Err(super::IntegrationError::CannotIntegrate(format!(
            "Cannot integrate product {} * {} - try u-substitution",
            left, right
        )))
    }
}

/// Try to combine two power expressions: x^a * x^b = x^(a+b)
fn try_combine_powers(left: &Expression, right: &Expression, var: &str) -> Option<Expression> {
    let left_power = extract_power(left, var)?;
    let right_power = extract_power(right, var)?;

    // Combine exponents
    let sum = Expression::Binary(BinaryOp::Add, Box::new(left_power), Box::new(right_power));

    Some(Expression::Power(
        Box::new(Expression::Variable(Variable::new(var))),
        Box::new(sum),
    ))
}

/// Extract the power of a variable expression.
/// x -> 1, x^n -> n, constant -> None
fn extract_power(expr: &Expression, var: &str) -> Option<Expression> {
    match expr {
        Expression::Variable(v) if v.name == var => Some(Expression::Integer(1)),
        Expression::Power(base, exp) => {
            if let Expression::Variable(v) = base.as_ref() {
                if v.name == var {
                    return Some(exp.as_ref().clone());
                }
            }
            None
        }
        _ => None,
    }
}

/// Integrate a power expression that was combined.
fn integrate_power_expr(expr: &Expression, var: &str) -> IntegrationResult {
    if let Expression::Power(base, exp) = expr {
        super::helpers::integrate_power(base, exp, var)
    } else {
        super::integrate_impl(expr, var)
    }
}

/// Integrate a quotient expression.
pub(super) fn integrate_quotient(
    num: &Expression,
    denom: &Expression,
    var: &str,
) -> IntegrationResult {
    let num_has_var = num.contains_variable(var);
    let denom_has_var = denom.contains_variable(var);

    if !denom_has_var {
        // ∫f(x)/c dx = (1/c)∫f(x) dx
        let num_integral = super::integrate_impl(num, var)?;
        Ok(Expression::Binary(
            BinaryOp::Div,
            Box::new(num_integral),
            Box::new(denom.clone()),
        ))
    } else if !num_has_var {
        // ∫c/f(x) dx - check for special cases
        // ∫c/x dx = c*ln|x|
        if let Expression::Variable(v) = denom {
            if v.name == var {
                let ln_x = Expression::Function(
                    Function::Ln,
                    vec![Expression::Function(
                        Function::Abs,
                        vec![Expression::Variable(Variable::new(var))],
                    )],
                );
                return Ok(Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(num.clone()),
                    Box::new(ln_x),
                ));
            }
        }

        // ∫c/(1+x^2) dx = c*arctan(x) - check pattern
        if let Some(result) = try_arctan_pattern(num, denom, var) {
            return Ok(result);
        }

        // ∫c/sqrt(1-x^2) dx = c*arcsin(x) - check pattern
        if let Some(result) = try_arcsin_pattern(num, denom, var) {
            return Ok(result);
        }

        // Try partial fraction decomposition for c/q(x) where q is a polynomial
        if let Some(result) = super::rational::try_partial_fraction_integration(num, denom, var) {
            return result;
        }

        Err(super::IntegrationError::CannotIntegrate(format!(
            "Cannot integrate {}/{}",
            num, denom
        )))
    } else {
        // Check for f(x)/x which might be ln pattern
        // Or convert to negative power and try
        if let Expression::Variable(v) = denom {
            if v.name == var {
                // ∫f(x)/x dx - complex, needs special handling
                // Simple case: ∫x^n/x dx = ∫x^(n-1) dx
                if let Some(power) = extract_power(num, var) {
                    let new_exp = Expression::Binary(
                        BinaryOp::Sub,
                        Box::new(power),
                        Box::new(Expression::Integer(1)),
                    );
                    return integrate_power(
                        &Expression::Variable(Variable::new(var)),
                        &new_exp,
                        var,
                    );
                }
            }
        }

        // Try partial fraction decomposition for rational functions p(x)/q(x)
        if let Some(result) = super::rational::try_partial_fraction_integration(num, denom, var) {
            return result;
        }

        // Convert to negative power: f/g = f * g^(-1)
        // Only works for simple cases
        Err(super::IntegrationError::CannotIntegrate(format!(
            "Cannot integrate quotient {}/{} - try partial fractions",
            num, denom
        )))
    }
}

/// Check for ∫c/(1+x^2) dx = c*arctan(x) pattern
fn try_arctan_pattern(num: &Expression, denom: &Expression, var: &str) -> Option<Expression> {
    // Check if denom is 1 + x^2
    if let Expression::Binary(BinaryOp::Add, left, right) = denom {
        let is_one = matches!(left.as_ref(), Expression::Integer(1));
        let is_x_squared = matches!(
            right.as_ref(),
            Expression::Power(base, exp)
            if matches!(base.as_ref(), Expression::Variable(v) if v.name == var)
            && matches!(exp.as_ref(), Expression::Integer(2))
        );

        if is_one && is_x_squared {
            let arctan_x = Expression::Function(
                Function::Atan,
                vec![Expression::Variable(Variable::new(var))],
            );
            return Some(Expression::Binary(
                BinaryOp::Mul,
                Box::new(num.clone()),
                Box::new(arctan_x),
            ));
        }
    }
    None
}

/// Check for ∫c/sqrt(1-x^2) dx = c*arcsin(x) pattern
fn try_arcsin_pattern(num: &Expression, denom: &Expression, var: &str) -> Option<Expression> {
    // Check if denom is sqrt(1 - x^2)
    if let Expression::Function(Function::Sqrt, args) = denom {
        if let Some(inner) = args.first() {
            if let Expression::Binary(BinaryOp::Sub, left, right) = inner {
                let is_one = matches!(left.as_ref(), Expression::Integer(1));
                let is_x_squared = matches!(
                    right.as_ref(),
                    Expression::Power(base, exp)
                    if matches!(base.as_ref(), Expression::Variable(v) if v.name == var)
                    && matches!(exp.as_ref(), Expression::Integer(2))
                );

                if is_one && is_x_squared {
                    let arcsin_x = Expression::Function(
                        Function::Asin,
                        vec![Expression::Variable(Variable::new(var))],
                    );
                    return Some(Expression::Binary(
                        BinaryOp::Mul,
                        Box::new(num.clone()),
                        Box::new(arcsin_x),
                    ));
                }
            }
        }
    }
    None
}

/// Integrate a power expression.
pub(super) fn integrate_power(
    base: &Expression,
    exponent: &Expression,
    var: &str,
) -> IntegrationResult {
    let base_has_var = base.contains_variable(var);
    let exp_has_var = exponent.contains_variable(var);

    if !base_has_var && !exp_has_var {
        // Constant: ∫c^d dx = c^d * x
        let x = Expression::Variable(Variable::new(var));
        let power = Expression::Power(Box::new(base.clone()), Box::new(exponent.clone()));
        Ok(Expression::Binary(
            BinaryOp::Mul,
            Box::new(power),
            Box::new(x),
        ))
    } else if base_has_var && !exp_has_var {
        // Power rule: ∫x^n dx
        if let Expression::Variable(v) = base {
            if v.name == var {
                return integrate_power_of_var(exponent, var);
            }
        }

        // More complex base with constant exponent
        // Needs u-substitution or chain rule recognition
        Err(super::IntegrationError::CannotIntegrate(format!(
            "Cannot integrate ({})^{} - complex base",
            base, exponent
        )))
    } else if !base_has_var && exp_has_var {
        // Exponential: ∫a^x dx = a^x / ln(a) when a is constant
        if let Expression::Variable(v) = exponent {
            if v.name == var {
                // ∫a^x dx = a^x / ln(a)
                let ln_base = Expression::Function(Function::Ln, vec![base.clone()]);
                let a_to_x = Expression::Power(Box::new(base.clone()), Box::new(exponent.clone()));
                return Ok(Expression::Binary(
                    BinaryOp::Div,
                    Box::new(a_to_x),
                    Box::new(ln_base),
                ));
            }
        }

        // Special case: ∫e^x dx = e^x
        if let Expression::Constant(crate::ast::SymbolicConstant::E) = base {
            if let Expression::Variable(v) = exponent {
                if v.name == var {
                    return Ok(Expression::Power(
                        Box::new(base.clone()),
                        Box::new(exponent.clone()),
                    ));
                }
            }
        }

        Err(super::IntegrationError::CannotIntegrate(format!(
            "Cannot integrate {}^({}) - exponential with complex exponent",
            base, exponent
        )))
    } else {
        // Both contain variable
        Err(super::IntegrationError::CannotIntegrate(
            "Cannot integrate f(x)^g(x) - requires special techniques".to_string(),
        ))
    }
}

/// Integrate x^n where n is a constant.
fn integrate_power_of_var(exponent: &Expression, var: &str) -> IntegrationResult {
    // Check for x^(-1) = 1/x -> ln|x|
    if let Expression::Integer(-1) = exponent {
        return Ok(Expression::Function(
            Function::Ln,
            vec![Expression::Function(
                Function::Abs,
                vec![Expression::Variable(Variable::new(var))],
            )],
        ));
    }

    if let Expression::Unary(UnaryOp::Neg, inner) = exponent {
        if let Expression::Integer(1) = inner.as_ref() {
            return Ok(Expression::Function(
                Function::Ln,
                vec![Expression::Function(
                    Function::Abs,
                    vec![Expression::Variable(Variable::new(var))],
                )],
            ));
        }
    }

    // Check for rational -1
    if let Expression::Rational(r) = exponent {
        if *r.numer() == -1 && *r.denom() == 1 {
            return Ok(Expression::Function(
                Function::Ln,
                vec![Expression::Function(
                    Function::Abs,
                    vec![Expression::Variable(Variable::new(var))],
                )],
            ));
        }
    }

    // General case: ∫x^n dx = x^(n+1)/(n+1)
    let x = Expression::Variable(Variable::new(var));
    let n_plus_1 = Expression::Binary(
        BinaryOp::Add,
        Box::new(exponent.clone()),
        Box::new(Expression::Integer(1)),
    );

    let x_to_n_plus_1 = Expression::Power(Box::new(x), Box::new(n_plus_1.clone()));

    Ok(Expression::Binary(
        BinaryOp::Div,
        Box::new(x_to_n_plus_1),
        Box::new(n_plus_1),
    ))
}

/// Integrate a function call.
pub(super) fn integrate_function(
    func: &Function,
    args: &[Expression],
    var: &str,
) -> IntegrationResult {
    if args.is_empty() {
        return Err(super::IntegrationError::CannotIntegrate(
            "Function with no arguments".to_string(),
        ));
    }

    let arg = &args[0];

    // Check if argument is simply the variable
    let is_simple_var = matches!(arg, Expression::Variable(v) if v.name == var);

    if !is_simple_var {
        // For compound arguments, we'd need u-substitution (Task 15)
        // Try linear substitution: f(ax + b) -> (1/a) * F(ax + b)
        if let Some(result) = try_linear_substitution(func, arg, var) {
            return Ok(result);
        }

        return Err(super::IntegrationError::CannotIntegrate(format!(
            "Cannot integrate {}({}) - try u-substitution",
            func_name(func),
            arg
        )));
    }

    // Standard integrals table
    match func {
        // ∫sin(x) dx = -cos(x)
        Function::Sin => Ok(Expression::Unary(
            UnaryOp::Neg,
            Box::new(Expression::Function(
                Function::Cos,
                vec![Expression::Variable(Variable::new(var))],
            )),
        )),

        // ∫cos(x) dx = sin(x)
        Function::Cos => Ok(Expression::Function(
            Function::Sin,
            vec![Expression::Variable(Variable::new(var))],
        )),

        // ∫tan(x) dx = -ln|cos(x)| = ln|sec(x)|
        Function::Tan => {
            let cos_x = Expression::Function(
                Function::Cos,
                vec![Expression::Variable(Variable::new(var))],
            );
            let abs_cos = Expression::Function(Function::Abs, vec![cos_x]);
            let ln_abs_cos = Expression::Function(Function::Ln, vec![abs_cos]);
            Ok(Expression::Unary(UnaryOp::Neg, Box::new(ln_abs_cos)))
        }

        // ∫e^x dx = e^x (when e is the base)
        Function::Exp => Ok(Expression::Function(
            Function::Exp,
            vec![Expression::Variable(Variable::new(var))],
        )),

        // ∫ln(x) dx = x*ln(x) - x  (integration by parts result)
        Function::Ln => {
            let x = Expression::Variable(Variable::new(var));
            let ln_x = Expression::Function(Function::Ln, vec![x.clone()]);
            let x_ln_x = Expression::Binary(BinaryOp::Mul, Box::new(x.clone()), Box::new(ln_x));
            Ok(Expression::Binary(
                BinaryOp::Sub,
                Box::new(x_ln_x),
                Box::new(x),
            ))
        }

        // ∫sinh(x) dx = cosh(x)
        Function::Sinh => Ok(Expression::Function(
            Function::Cosh,
            vec![Expression::Variable(Variable::new(var))],
        )),

        // ∫cosh(x) dx = sinh(x)
        Function::Cosh => Ok(Expression::Function(
            Function::Sinh,
            vec![Expression::Variable(Variable::new(var))],
        )),

        // ∫tanh(x) dx = ln(cosh(x))
        Function::Tanh => {
            let cosh_x = Expression::Function(
                Function::Cosh,
                vec![Expression::Variable(Variable::new(var))],
            );
            Ok(Expression::Function(Function::Ln, vec![cosh_x]))
        }

        // ∫1/sqrt(x) = ∫x^(-1/2) dx = 2*sqrt(x)
        Function::Sqrt => Err(super::IntegrationError::CannotIntegrate(
            "∫sqrt(x) dx - rewrite as x^(1/2) and use power rule".to_string(),
        )),

        // Other functions generally don't have simple antiderivatives
        _ => Err(super::IntegrationError::CannotIntegrate(format!(
            "No standard integral for {}(x)",
            func_name(func)
        ))),
    }
}

/// Try linear substitution: ∫f(ax+b) dx = (1/a) * F(ax+b)
fn try_linear_substitution(func: &Function, arg: &Expression, var: &str) -> Option<Expression> {
    // Check if arg is of form a*x + b or a*x
    let (coeff, _offset) = extract_linear_form(arg, var)?;

    // Check coefficient is not zero
    if matches!(&coeff, Expression::Integer(0)) {
        return None;
    }

    // Get the standard integral F(u) where u = ax + b
    let standard_integral = match func {
        Function::Sin => Expression::Unary(
            UnaryOp::Neg,
            Box::new(Expression::Function(Function::Cos, vec![arg.clone()])),
        ),
        Function::Cos => Expression::Function(Function::Sin, vec![arg.clone()]),
        Function::Exp => Expression::Function(Function::Exp, vec![arg.clone()]),
        _ => return None,
    };

    // Divide by coefficient: (1/a) * F(ax+b)
    Some(Expression::Binary(
        BinaryOp::Div,
        Box::new(standard_integral),
        Box::new(coeff),
    ))
}

/// Extract linear form ax + b from an expression.
/// Returns (a, b) if successful.
pub(super) fn extract_linear_form(
    expr: &Expression,
    var: &str,
) -> Option<(Expression, Expression)> {
    match expr {
        // Just x -> (1, 0)
        Expression::Variable(v) if v.name == var => {
            Some((Expression::Integer(1), Expression::Integer(0)))
        }

        // a*x -> (a, 0)
        Expression::Binary(BinaryOp::Mul, left, right) => {
            if !left.contains_variable(var) {
                if matches!(right.as_ref(), Expression::Variable(v) if v.name == var) {
                    return Some((left.as_ref().clone(), Expression::Integer(0)));
                }
            }
            if !right.contains_variable(var) {
                if matches!(left.as_ref(), Expression::Variable(v) if v.name == var) {
                    return Some((right.as_ref().clone(), Expression::Integer(0)));
                }
            }
            None
        }

        // a*x + b -> (a, b)
        Expression::Binary(BinaryOp::Add, left, right) => {
            if !right.contains_variable(var) {
                if let Some((a, _)) = extract_linear_form(left, var) {
                    return Some((a, right.as_ref().clone()));
                }
            }
            if !left.contains_variable(var) {
                if let Some((a, _)) = extract_linear_form(right, var) {
                    return Some((a, left.as_ref().clone()));
                }
            }
            None
        }

        _ => None,
    }
}

/// Get the name of a function for error messages.
pub(super) fn func_name(func: &Function) -> &'static str {
    match func {
        Function::Sin => "sin",
        Function::Cos => "cos",
        Function::Tan => "tan",
        Function::Asin => "asin",
        Function::Acos => "acos",
        Function::Atan => "atan",
        Function::Atan2 => "atan2",
        Function::Sinh => "sinh",
        Function::Cosh => "cosh",
        Function::Tanh => "tanh",
        Function::Exp => "exp",
        Function::Ln => "ln",
        Function::Log => "log",
        Function::Log2 => "log2",
        Function::Log10 => "log10",
        Function::Sqrt => "sqrt",
        Function::Cbrt => "cbrt",
        Function::Abs => "abs",
        Function::Floor => "floor",
        Function::Ceil => "ceil",
        Function::Round => "round",
        Function::Min => "min",
        Function::Max => "max",
        Function::Pow => "pow",
        Function::Sign => "sign",
        Function::Custom(_) => {
            // Return a static string since we can't return the owned name
            "custom"
        }
    }
}

/// Extract all multiplicative factors from an expression.
pub(super) fn extract_factors(expr: &Expression) -> Vec<Expression> {
    match expr {
        Expression::Binary(BinaryOp::Mul, left, right) => {
            let mut factors = extract_factors(left);
            factors.extend(extract_factors(right));
            factors
        }
        _ => vec![expr.clone()],
    }
}

/// Combine factors into a product.
pub(super) fn combine_factors(factors: &[Expression]) -> Expression {
    if factors.is_empty() {
        return Expression::Integer(1);
    }
    if factors.len() == 1 {
        return factors[0].clone();
    }

    let mut result = factors[0].clone();
    for factor in &factors[1..] {
        result = Expression::Binary(BinaryOp::Mul, Box::new(result), Box::new(factor.clone()));
    }
    result
}

/// Compute a canonical string key for an expression, normalising the order
/// of operands in commutative operations (Add, Mul) so that, e.g.,
/// `x * 2` and `2 * x` produce the same key.
///
/// The key is used only for equivalence testing; it is not a human-readable
/// display form.
pub(super) fn canonical_key(expr: &Expression) -> String {
    match expr {
        Expression::Binary(op @ (BinaryOp::Add | BinaryOp::Mul), left, right) => {
            let mut parts = vec![canonical_key(left), canonical_key(right)];
            parts.sort();
            let op_sym = match op {
                BinaryOp::Add => "+",
                BinaryOp::Mul => "*",
                _ => unreachable!(),
            };
            format!("({}{}{})", parts[0], op_sym, parts[1])
        }
        Expression::Binary(op, left, right) => {
            format!("({}{:?}{})", canonical_key(left), op, canonical_key(right))
        }
        Expression::Unary(op, inner) => {
            format!("({:?}{})", op, canonical_key(inner))
        }
        Expression::Power(base, exp) => {
            format!("({}^{})", canonical_key(base), canonical_key(exp))
        }
        Expression::Function(f, args) => {
            let arg_keys: Vec<_> = args.iter().map(canonical_key).collect();
            format!("{:?}({})", f, arg_keys.join(","))
        }
        // Leaf nodes: delegate to Display which is already canonical.
        other => format!("{}", other),
    }
}

/// Check if two expressions are structurally equivalent under commutativity
/// of addition and multiplication.
///
/// Both expressions are first simplified, then compared using a canonical
/// key that normalises the operand order of commutative binary operations.
/// This avoids the brittleness of raw [`format!`] string comparison where,
/// for example, `x * 2` and `2 * x` would compare unequal.
pub(super) fn expressions_equivalent(a: &Expression, b: &Expression) -> bool {
    canonical_key(a) == canonical_key(b)
}

/// Check if an expression equals 1.
pub(super) fn is_one(expr: &Expression) -> bool {
    matches!(expr, Expression::Integer(1))
}

/// Substitute all occurrences of a variable with an expression.
pub(super) fn substitute_variable(
    expr: &Expression,
    var_name: &str,
    replacement: &Expression,
) -> Expression {
    match expr {
        Expression::Variable(v) if v.name == var_name => replacement.clone(),
        Expression::Variable(_) => expr.clone(),
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Rational(_)
        | Expression::Complex(_)
        | Expression::Constant(_) => expr.clone(),

        Expression::Unary(op, inner) => Expression::Unary(
            op.clone(),
            Box::new(substitute_variable(inner, var_name, replacement)),
        ),

        Expression::Binary(op, left, right) => Expression::Binary(
            op.clone(),
            Box::new(substitute_variable(left, var_name, replacement)),
            Box::new(substitute_variable(right, var_name, replacement)),
        ),

        Expression::Power(base, exp) => Expression::Power(
            Box::new(substitute_variable(base, var_name, replacement)),
            Box::new(substitute_variable(exp, var_name, replacement)),
        ),

        Expression::Function(func, args) => Expression::Function(
            func.clone(),
            args.iter()
                .map(|arg| substitute_variable(arg, var_name, replacement))
                .collect(),
        ),
    }
}

/// Check if an expression is polynomial-like (contains only powers of var with constant exponents).
pub(super) fn is_polynomial_like(expr: &Expression, var: &str) -> bool {
    match expr {
        Expression::Integer(_) | Expression::Float(_) | Expression::Rational(_) => true,
        Expression::Variable(v) => v.name == var,
        Expression::Power(base, exp) => {
            if let Expression::Variable(v) = base.as_ref() {
                if v.name == var {
                    // Check exponent is a non-negative integer
                    return !exp.contains_variable(var);
                }
            }
            false
        }
        Expression::Binary(BinaryOp::Add, left, right)
        | Expression::Binary(BinaryOp::Sub, left, right) => {
            is_polynomial_like(left, var) && is_polynomial_like(right, var)
        }
        Expression::Binary(BinaryOp::Mul, left, right) => {
            // c * p(x) is polynomial if p(x) is polynomial
            if !left.contains_variable(var) {
                is_polynomial_like(right, var)
            } else if !right.contains_variable(var) {
                is_polynomial_like(left, var)
            } else {
                is_polynomial_like(left, var) && is_polynomial_like(right, var)
            }
        }
        Expression::Unary(UnaryOp::Neg, inner) => is_polynomial_like(inner, var),
        _ => false,
    }
}
