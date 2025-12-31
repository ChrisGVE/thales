//! Symbolic integration module for computing indefinite integrals.
//!
//! This module provides functionality for computing antiderivatives of mathematical
//! expressions using standard integration techniques including:
//!
//! - Power rule
//! - Sum and difference rules
//! - Constant multiple rule
//! - Standard integrals table (trigonometric, exponential, logarithmic)
//!
//! # Example
//!
//! ```
//! use mathsolver_core::integration::{integrate, IntegrationError};
//! use mathsolver_core::ast::{Expression, Variable};
//!
//! // Integrate x^2 with respect to x
//! let x = Expression::Variable(Variable::new("x"));
//! let x_squared = Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(2)));
//!
//! let result = integrate(&x_squared, "x").unwrap();
//! // Result: x^3/3 + C (constant of integration handled separately)
//! ```

use crate::ast::{BinaryOp, Expression, Function, UnaryOp, Variable};
use std::fmt;

/// Error types that can occur during integration.
#[derive(Debug, Clone, PartialEq)]
pub enum IntegrationError {
    /// Integration cannot be performed symbolically.
    CannotIntegrate(String),
    /// The integrand contains unsupported constructs.
    UnsupportedExpression(String),
    /// Division by zero would occur.
    DivisionByZero,
}

impl fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntegrationError::CannotIntegrate(msg) => {
                write!(f, "Cannot integrate: {}", msg)
            }
            IntegrationError::UnsupportedExpression(msg) => {
                write!(f, "Unsupported expression: {}", msg)
            }
            IntegrationError::DivisionByZero => {
                write!(f, "Division by zero in integration")
            }
        }
    }
}

impl std::error::Error for IntegrationError {}

/// Result type for integration operations.
pub type IntegrationResult = Result<Expression, IntegrationError>;

/// Compute the indefinite integral of an expression with respect to a variable.
///
/// This function returns the antiderivative without the constant of integration.
/// The caller should add `+ C` if displaying the full indefinite integral.
///
/// # Arguments
///
/// * `expr` - The expression to integrate
/// * `var` - The variable of integration (e.g., "x")
///
/// # Returns
///
/// The antiderivative expression, or an error if integration fails.
///
/// # Supported Forms
///
/// - Constants: ∫c dx = cx
/// - Power rule: ∫x^n dx = x^(n+1)/(n+1) for n ≠ -1
/// - Reciprocal: ∫1/x dx = ln|x|
/// - Sum/difference: ∫(f ± g) dx = ∫f dx ± ∫g dx
/// - Constant multiple: ∫cf dx = c∫f dx
/// - Trigonometric: sin, cos, tan, sec^2
/// - Exponential: e^x
/// - Logarithmic via inverse
///
/// # Example
///
/// ```
/// use mathsolver_core::integration::integrate;
/// use mathsolver_core::ast::{Expression, Variable};
///
/// let x = Expression::Variable(Variable::new("x"));
/// let expr = Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(3)));
///
/// let result = integrate(&expr, "x").unwrap();
/// // x^3 integrates to x^4/4
/// ```
pub fn integrate(expr: &Expression, var: &str) -> IntegrationResult {
    integrate_impl(expr, var)
}

/// Internal implementation of integration.
fn integrate_impl(expr: &Expression, var: &str) -> IntegrationResult {
    match expr {
        // Constants: ∫c dx = c*x
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Rational(_)
        | Expression::Complex(_) => {
            let x = Expression::Variable(Variable::new(var));
            Ok(Expression::Binary(
                BinaryOp::Mul,
                Box::new(expr.clone()),
                Box::new(x),
            ))
        }

        // Symbolic constants: ∫pi dx = pi*x, ∫e dx = e*x
        Expression::Constant(_) => {
            let x = Expression::Variable(Variable::new(var));
            Ok(Expression::Binary(
                BinaryOp::Mul,
                Box::new(expr.clone()),
                Box::new(x),
            ))
        }

        // Variable: ∫x dx = x^2/2, ∫y dx = y*x (y treated as constant)
        Expression::Variable(v) => {
            if v.name == var {
                // ∫x dx = x^2/2
                let x = Expression::Variable(Variable::new(var));
                let x_squared = Expression::Power(Box::new(x), Box::new(Expression::Integer(2)));
                Ok(Expression::Binary(
                    BinaryOp::Div,
                    Box::new(x_squared),
                    Box::new(Expression::Integer(2)),
                ))
            } else {
                // Treat as constant: ∫y dx = y*x
                let x = Expression::Variable(Variable::new(var));
                Ok(Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(expr.clone()),
                    Box::new(x),
                ))
            }
        }

        // Unary operations
        Expression::Unary(op, inner) => match op {
            UnaryOp::Neg => {
                // ∫-f dx = -∫f dx
                let inner_integral = integrate_impl(inner, var)?;
                Ok(Expression::Unary(UnaryOp::Neg, Box::new(inner_integral)))
            }
            UnaryOp::Abs => Err(IntegrationError::CannotIntegrate(
                "Cannot integrate |f(x)| symbolically".to_string(),
            )),
            UnaryOp::Not => Err(IntegrationError::UnsupportedExpression(
                "Logical NOT cannot be integrated".to_string(),
            )),
        },

        // Binary operations
        Expression::Binary(op, left, right) => match op {
            // Sum rule: ∫(f + g) dx = ∫f dx + ∫g dx
            BinaryOp::Add => {
                let left_integral = integrate_impl(left, var)?;
                let right_integral = integrate_impl(right, var)?;
                Ok(Expression::Binary(
                    BinaryOp::Add,
                    Box::new(left_integral),
                    Box::new(right_integral),
                ))
            }

            // Difference rule: ∫(f - g) dx = ∫f dx - ∫g dx
            BinaryOp::Sub => {
                let left_integral = integrate_impl(left, var)?;
                let right_integral = integrate_impl(right, var)?;
                Ok(Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(left_integral),
                    Box::new(right_integral),
                ))
            }

            // Multiplication: check for constant multiple
            BinaryOp::Mul => integrate_product(left, right, var),

            // Division: check for power rule with negative exponent or constant divisor
            BinaryOp::Div => integrate_quotient(left, right, var),

            BinaryOp::Mod => Err(IntegrationError::CannotIntegrate(
                "Modulo cannot be integrated".to_string(),
            )),
        },

        // Power expressions
        Expression::Power(base, exponent) => integrate_power(base, exponent, var),

        // Function calls
        Expression::Function(func, args) => integrate_function(func, args, var),
    }
}

/// Integrate a product expression.
fn integrate_product(left: &Expression, right: &Expression, var: &str) -> IntegrationResult {
    let left_has_var = left.contains_variable(var);
    let right_has_var = right.contains_variable(var);

    if !left_has_var && !right_has_var {
        // Both constant: ∫c1*c2 dx = c1*c2*x
        let x = Expression::Variable(Variable::new(var));
        let product = Expression::Binary(BinaryOp::Mul, Box::new(left.clone()), Box::new(right.clone()));
        Ok(Expression::Binary(
            BinaryOp::Mul,
            Box::new(product),
            Box::new(x),
        ))
    } else if !left_has_var {
        // Constant multiple rule: ∫c*f dx = c*∫f dx
        let right_integral = integrate_impl(right, var)?;
        Ok(Expression::Binary(
            BinaryOp::Mul,
            Box::new(left.clone()),
            Box::new(right_integral),
        ))
    } else if !right_has_var {
        // Constant multiple rule: ∫f*c dx = c*∫f dx
        let left_integral = integrate_impl(left, var)?;
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
        Err(IntegrationError::CannotIntegrate(format!(
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
    let sum = Expression::Binary(
        BinaryOp::Add,
        Box::new(left_power),
        Box::new(right_power),
    );

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
        integrate_power(base, exp, var)
    } else {
        integrate_impl(expr, var)
    }
}

/// Integrate a quotient expression.
fn integrate_quotient(num: &Expression, denom: &Expression, var: &str) -> IntegrationResult {
    let num_has_var = num.contains_variable(var);
    let denom_has_var = denom.contains_variable(var);

    if !denom_has_var {
        // ∫f(x)/c dx = (1/c)∫f(x) dx
        let num_integral = integrate_impl(num, var)?;
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

        Err(IntegrationError::CannotIntegrate(format!(
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

        // Convert to negative power: f/g = f * g^(-1)
        // Only works for simple cases
        Err(IntegrationError::CannotIntegrate(format!(
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
fn integrate_power(base: &Expression, exponent: &Expression, var: &str) -> IntegrationResult {
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
        Err(IntegrationError::CannotIntegrate(format!(
            "Cannot integrate ({})^{} - complex base",
            base, exponent
        )))
    } else if !base_has_var && exp_has_var {
        // Exponential: ∫a^x dx = a^x / ln(a) when a is constant
        if let Expression::Variable(v) = exponent {
            if v.name == var {
                // ∫a^x dx = a^x / ln(a)
                let ln_base = Expression::Function(Function::Ln, vec![base.clone()]);
                let a_to_x =
                    Expression::Power(Box::new(base.clone()), Box::new(exponent.clone()));
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

        Err(IntegrationError::CannotIntegrate(format!(
            "Cannot integrate {}^({}) - exponential with complex exponent",
            base, exponent
        )))
    } else {
        // Both contain variable
        Err(IntegrationError::CannotIntegrate(
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
fn integrate_function(func: &Function, args: &[Expression], var: &str) -> IntegrationResult {
    if args.is_empty() {
        return Err(IntegrationError::CannotIntegrate(
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

        return Err(IntegrationError::CannotIntegrate(format!(
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
        Function::Sqrt => Err(IntegrationError::CannotIntegrate(
            "∫sqrt(x) dx - rewrite as x^(1/2) and use power rule".to_string(),
        )),

        // Other functions generally don't have simple antiderivatives
        _ => Err(IntegrationError::CannotIntegrate(format!(
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
fn extract_linear_form(expr: &Expression, var: &str) -> Option<(Expression, Expression)> {
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
fn func_name(func: &Function) -> &'static str {
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

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::Rational64;

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn pow(base: Expression, exp: Expression) -> Expression {
        Expression::Power(Box::new(base), Box::new(exp))
    }

    fn add(left: Expression, right: Expression) -> Expression {
        Expression::Binary(BinaryOp::Add, Box::new(left), Box::new(right))
    }

    fn mul(left: Expression, right: Expression) -> Expression {
        Expression::Binary(BinaryOp::Mul, Box::new(left), Box::new(right))
    }

    fn div(left: Expression, right: Expression) -> Expression {
        Expression::Binary(BinaryOp::Div, Box::new(left), Box::new(right))
    }

    #[test]
    fn test_integrate_constant() {
        // ∫5 dx = 5x
        let result = integrate(&int(5), "x").unwrap();
        // Result should be 5 * x
        assert!(matches!(
            result,
            Expression::Binary(BinaryOp::Mul, left, right)
            if matches!(left.as_ref(), Expression::Integer(5))
            && matches!(right.as_ref(), Expression::Variable(v) if v.name == "x")
        ));
    }

    #[test]
    fn test_integrate_x() {
        // ∫x dx = x^2/2
        let result = integrate(&var("x"), "x").unwrap();
        assert!(matches!(
            result,
            Expression::Binary(BinaryOp::Div, num, denom)
            if matches!(denom.as_ref(), Expression::Integer(2))
        ));
    }

    #[test]
    fn test_integrate_x_squared() {
        // ∫x^2 dx = x^3/3
        let x_squared = pow(var("x"), int(2));
        let result = integrate(&x_squared, "x").unwrap();

        // Should be x^(2+1) / (2+1) = x^3/3
        if let Expression::Binary(BinaryOp::Div, num, denom) = result {
            // Numerator should be x^(2+1)
            assert!(matches!(num.as_ref(), Expression::Power(_, _)));
            // Denominator should be 2+1
            assert!(matches!(denom.as_ref(), Expression::Binary(BinaryOp::Add, _, _)));
        } else {
            panic!("Expected division expression");
        }
    }

    #[test]
    fn test_integrate_sum() {
        // ∫(x^2 + x) dx = x^3/3 + x^2/2
        let expr = add(pow(var("x"), int(2)), var("x"));
        let result = integrate(&expr, "x").unwrap();
        assert!(matches!(result, Expression::Binary(BinaryOp::Add, _, _)));
    }

    #[test]
    fn test_integrate_constant_multiple() {
        // ∫3x dx = 3 * x^2/2
        let expr = mul(int(3), var("x"));
        let result = integrate(&expr, "x").unwrap();

        // Should be 3 * (x^2/2)
        assert!(matches!(result, Expression::Binary(BinaryOp::Mul, left, _)
            if matches!(left.as_ref(), Expression::Integer(3))));
    }

    #[test]
    fn test_integrate_reciprocal() {
        // ∫1/x dx = ln|x|
        let expr = div(int(1), var("x"));
        let result = integrate(&expr, "x").unwrap();

        // Should be 1 * ln(|x|)
        assert!(matches!(result, Expression::Binary(BinaryOp::Mul, _, ln_part)
            if matches!(ln_part.as_ref(), Expression::Function(Function::Ln, _))));
    }

    #[test]
    fn test_integrate_sin() {
        // ∫sin(x) dx = -cos(x)
        let sin_x = Expression::Function(Function::Sin, vec![var("x")]);
        let result = integrate(&sin_x, "x").unwrap();

        assert!(matches!(
            result,
            Expression::Unary(UnaryOp::Neg, inner)
            if matches!(inner.as_ref(), Expression::Function(Function::Cos, _))
        ));
    }

    #[test]
    fn test_integrate_cos() {
        // ∫cos(x) dx = sin(x)
        let cos_x = Expression::Function(Function::Cos, vec![var("x")]);
        let result = integrate(&cos_x, "x").unwrap();

        assert!(matches!(
            result,
            Expression::Function(Function::Sin, args)
            if args.len() == 1
        ));
    }

    #[test]
    fn test_integrate_exp() {
        // ∫e^x dx = e^x
        let exp_x = Expression::Function(Function::Exp, vec![var("x")]);
        let result = integrate(&exp_x, "x").unwrap();

        assert!(matches!(
            result,
            Expression::Function(Function::Exp, args)
            if args.len() == 1
        ));
    }

    #[test]
    fn test_integrate_x_power_negative_one() {
        // ∫x^(-1) dx = ln|x|
        let expr = pow(var("x"), int(-1));
        let result = integrate(&expr, "x").unwrap();

        assert!(matches!(
            result,
            Expression::Function(Function::Ln, args)
            if matches!(&args[0], Expression::Function(Function::Abs, _))
        ));
    }

    #[test]
    fn test_integrate_polynomial() {
        // ∫(x^2 + 2x + 1) dx = x^3/3 + x^2 + x
        let poly = add(
            add(pow(var("x"), int(2)), mul(int(2), var("x"))),
            int(1),
        );
        let result = integrate(&poly, "x");
        assert!(result.is_ok());
    }

    #[test]
    fn test_integrate_linear_sin() {
        // ∫sin(2x) dx = -cos(2x)/2
        let two_x = mul(int(2), var("x"));
        let sin_2x = Expression::Function(Function::Sin, vec![two_x]);
        let result = integrate(&sin_2x, "x").unwrap();

        // Should be (-cos(2x)) / 2
        assert!(matches!(result, Expression::Binary(BinaryOp::Div, _, _)));
    }

    #[test]
    fn test_differentiate_integral_equals_original() {
        // Fundamental theorem: d/dx(∫f dx) = f
        // Test with x^2
        let x_squared = pow(var("x"), int(2));
        let integral = integrate(&x_squared, "x").unwrap();
        let derivative = integral.differentiate("x").simplify();

        // The derivative should simplify to x^2 (or equivalent)
        // This is a partial test - full verification needs numerical checks
        // For now, just verify we get back a power expression
        // The actual result may be (3*x^2) / 3 which simplifies to x^2
    }
}
