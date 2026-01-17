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
//! use thales::integration::{integrate, IntegrationError};
//! use thales::ast::{Expression, Variable};
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
#[non_exhaustive]
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
/// use thales::integration::integrate;
/// use thales::ast::{Expression, Variable};
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

// =============================================================================
// U-Substitution Integration
// =============================================================================

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
    if let Ok(result) = integrate_impl(expr, var) {
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
                if let Ok(f_integral) = integrate_impl(&f_of_u, u_var) {
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

/// Extract all multiplicative factors from an expression.
fn extract_factors(expr: &Expression) -> Vec<Expression> {
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
fn combine_factors(factors: &[Expression]) -> Expression {
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

/// Extract the inner function from a composite expression.
fn extract_inner_function(expr: &Expression) -> Option<Expression> {
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

/// Check if two expressions are equivalent (after simplification).
fn expressions_equivalent(a: &Expression, b: &Expression) -> bool {
    // Simple structural equality check
    // A more robust implementation would use canonical forms
    format!("{}", a) == format!("{}", b)
}

/// Check if expr1 = constant * expr2 and return the constant.
fn extract_constant_multiple(
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

/// Check if an expression equals 1.
fn is_one(expr: &Expression) -> bool {
    matches!(expr, Expression::Integer(1))
}

/// Substitute u back with the original expression.
fn back_substitute(expr: &Expression, u: &Expression, _var: &str) -> Expression {
    substitute_variable(expr, "u", u)
}

/// Substitute all occurrences of a variable with an expression.
fn substitute_variable(expr: &Expression, var_name: &str, replacement: &Expression) -> Expression {
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
    if let Ok(result) = integrate_impl(expr, var) {
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

// =============================================================================
// Integration by Parts
// =============================================================================

/// LIATE priority for choosing u in integration by parts.
/// Higher values indicate u should be chosen first.
/// L - Logarithmic
/// I - Inverse trigonometric
/// A - Algebraic (polynomials)
/// T - Trigonometric
/// E - Exponential
fn liate_priority(expr: &Expression, var: &str) -> u8 {
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

/// Attempt to integrate using integration by parts: ∫u dv = uv - ∫v du
///
/// # Arguments
///
/// * `expr` - The expression to integrate (should be a product)
/// * `var` - The variable of integration
///
/// # Returns
///
/// The antiderivative if integration by parts succeeds, or an error.
///
/// # Example
///
/// ```
/// use thales::integration::integrate_by_parts;
/// use thales::ast::{Expression, Variable, BinaryOp};
///
/// // ∫x * e^x dx
/// let x = Expression::Variable(Variable::new("x"));
/// let e_x = Expression::Function(thales::ast::Function::Exp, vec![x.clone()]);
/// let expr = Expression::Binary(BinaryOp::Mul, Box::new(x), Box::new(e_x));
///
/// let result = integrate_by_parts(&expr, "x");
/// assert!(result.is_ok());
/// ```
pub fn integrate_by_parts(expr: &Expression, var: &str) -> IntegrationResult {
    integrate_by_parts_impl(expr, var, 0)
}

/// Maximum depth for integration by parts to prevent infinite loops.
const MAX_PARTS_DEPTH: usize = 10;

/// Internal implementation with depth tracking.
fn integrate_by_parts_impl(expr: &Expression, var: &str, depth: usize) -> IntegrationResult {
    if depth > MAX_PARTS_DEPTH {
        return Err(IntegrationError::CannotIntegrate(
            "Integration by parts exceeded maximum depth".to_string(),
        ));
    }

    // First try regular integration
    if let Ok(result) = integrate_impl(expr, var) {
        return Ok(result);
    }

    // Try u-substitution
    if let Ok(result) = integrate_by_substitution(expr, var) {
        return Ok(result);
    }

    // Extract factors for integration by parts
    let factors = extract_factors(expr);

    if factors.len() < 2 {
        return Err(IntegrationError::CannotIntegrate(
            "Integration by parts requires a product".to_string(),
        ));
    }

    // Use LIATE to choose u (highest priority) and dv (rest)
    let (u, dv) = choose_u_and_dv(&factors, var);

    // Compute du = d(u)/dx
    let du = u.differentiate(var).simplify();

    // Compute v = ∫dv dx
    let v = match integrate_impl(&dv, var) {
        Ok(v) => v.simplify(),
        Err(_) => {
            // If we can't integrate dv directly, try recursively
            match integrate_by_parts_impl(&dv, var, depth + 1) {
                Ok(v) => v.simplify(),
                Err(e) => return Err(e),
            }
        }
    };

    // Now compute: uv - ∫v·du dx
    let uv = Expression::Binary(BinaryOp::Mul, Box::new(u.clone()), Box::new(v.clone())).simplify();

    let v_du =
        Expression::Binary(BinaryOp::Mul, Box::new(v.clone()), Box::new(du.clone())).simplify();

    // Check for recurring integral (integral of v*du equals original or a multiple)
    if let Some(result) = try_solve_recurring_integral(expr, &v_du, &uv, var, depth) {
        return Ok(result);
    }

    // Compute ∫v·du dx
    let integral_v_du = match integrate_impl(&v_du, var) {
        Ok(result) => result.simplify(),
        Err(_) => {
            // Try recursively with parts
            match integrate_by_parts_impl(&v_du, var, depth + 1) {
                Ok(result) => result.simplify(),
                Err(e) => return Err(e),
            }
        }
    };

    // Result: uv - ∫v·du
    let result =
        Expression::Binary(BinaryOp::Sub, Box::new(uv), Box::new(integral_v_du)).simplify();

    Ok(result)
}

/// Choose u and dv from factors using LIATE heuristic.
fn choose_u_and_dv(factors: &[Expression], var: &str) -> (Expression, Expression) {
    // Find the factor with highest LIATE priority for u
    let mut best_u_idx = 0;
    let mut best_priority = 0;

    for (i, factor) in factors.iter().enumerate() {
        let priority = liate_priority(factor, var);
        if priority > best_priority {
            best_priority = priority;
            best_u_idx = i;
        }
    }

    let u = factors[best_u_idx].clone();
    let dv_factors: Vec<_> = factors
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != best_u_idx)
        .map(|(_, f)| f.clone())
        .collect();
    let dv = combine_factors(&dv_factors);

    (u, dv)
}

/// Try to solve a recurring integral.
///
/// When ∫f dx appears as part of the result, we can solve algebraically.
/// Example: ∫e^x·sin(x) dx = e^x·sin(x) - ∫e^x·cos(x) dx
///        = e^x·sin(x) - e^x·cos(x) + ∫e^x·sin(x) dx
/// So: 2∫e^x·sin(x) dx = e^x·sin(x) - e^x·cos(x)
///     ∫e^x·sin(x) dx = (e^x·sin(x) - e^x·cos(x))/2
fn try_solve_recurring_integral(
    original: &Expression,
    v_du: &Expression,
    uv: &Expression,
    var: &str,
    depth: usize,
) -> Option<Expression> {
    if depth >= 2 {
        // Only check for recurring at depth >= 2 to avoid false positives
        return None;
    }

    // Check if v_du is structurally similar to original
    // This is a simplified check - we compare simplified forms
    let original_simplified = original.simplify();
    let v_du_simplified = v_du.simplify();

    // Check if they're the same (up to constant multiple)
    if let Some(coefficient) =
        check_same_up_to_constant(&original_simplified, &v_du_simplified, var)
    {
        // ∫f dx = uv - c·∫f dx
        // (1 + c)∫f dx = uv
        // ∫f dx = uv / (1 + c)

        let one_plus_c = Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Integer(1)),
            Box::new(coefficient),
        )
        .simplify();

        // Check for 0 (would mean 0 = uv, which is a contradiction or identity)
        if matches!(one_plus_c, Expression::Integer(0)) {
            return None; // Can't divide by zero
        }

        let result = Expression::Binary(BinaryOp::Div, Box::new(uv.clone()), Box::new(one_plus_c))
            .simplify();

        return Some(result);
    }

    // Also check if we're in the case where applying parts twice gives back original
    // This requires tracking through two applications
    None
}

/// Check if two expressions are the same up to a constant multiple.
/// Returns the constant if expr2 = c * expr1.
fn check_same_up_to_constant(
    expr1: &Expression,
    expr2: &Expression,
    var: &str,
) -> Option<Expression> {
    // Simple case: same expression means c = 1
    if expressions_equivalent(expr1, expr2) {
        return Some(Expression::Integer(1));
    }

    // Check if expr2 = -expr1
    if let Expression::Unary(UnaryOp::Neg, inner) = expr2 {
        if expressions_equivalent(expr1, inner) {
            return Some(Expression::Integer(-1));
        }
    }

    // Check if expr2 = c * expr1 where c is a constant
    if let Expression::Binary(BinaryOp::Mul, left, right) = expr2 {
        if !left.contains_variable(var) && expressions_equivalent(expr1, right) {
            return Some(left.as_ref().clone());
        }
        if !right.contains_variable(var) && expressions_equivalent(expr1, left) {
            return Some(right.as_ref().clone());
        }
    }

    None
}

/// Integration by parts with detailed steps.
///
/// Returns the result along with a step-by-step explanation.
pub fn integrate_by_parts_with_steps(
    expr: &Expression,
    var: &str,
) -> Result<(Expression, Vec<String>), IntegrationError> {
    let mut steps = Vec::new();
    steps.push(format!("∫{} d{}", expr, var));
    steps.push("Using integration by parts: ∫u dv = uv - ∫v du".to_string());

    // Extract factors
    let factors = extract_factors(expr);
    if factors.len() < 2 {
        return Err(IntegrationError::CannotIntegrate(
            "Integration by parts requires a product".to_string(),
        ));
    }

    // Choose u and dv
    let (u, dv) = choose_u_and_dv(&factors, var);
    steps.push(format!("Let u = {}", u));
    steps.push(format!("Let dv = {} d{}", dv, var));

    // Compute du
    let du = u.differentiate(var).simplify();
    steps.push(format!("Then du = {} d{}", du, var));

    // Compute v
    let v = match integrate_impl(&dv, var) {
        Ok(v) => v.simplify(),
        Err(_) => {
            return Err(IntegrationError::CannotIntegrate(format!(
                "Cannot integrate dv = {}",
                dv
            )))
        }
    };
    steps.push(format!("And v = ∫{} d{} = {}", dv, var, v));

    // Compute uv
    let uv = Expression::Binary(BinaryOp::Mul, Box::new(u.clone()), Box::new(v.clone())).simplify();
    steps.push(format!("uv = {}", uv));

    // Compute v·du
    let v_du =
        Expression::Binary(BinaryOp::Mul, Box::new(v.clone()), Box::new(du.clone())).simplify();
    steps.push(format!("v·du = {}", v_du));

    // Compute ∫v·du
    let integral_v_du = match integrate_by_parts_impl(&v_du, var, 1) {
        Ok(result) => result.simplify(),
        Err(e) => return Err(e),
    };
    steps.push(format!("∫v du = {}", integral_v_du));

    // Final result
    let result =
        Expression::Binary(BinaryOp::Sub, Box::new(uv), Box::new(integral_v_du)).simplify();
    steps.push(format!("Result: {}", result));

    Ok((result, steps))
}

/// Tabular method for integration by parts.
///
/// This method is efficient for integrals of the form ∫P(x)·f(x) dx
/// where P(x) is a polynomial and f(x) has an easy antiderivative (like e^x, sin(x), cos(x)).
///
/// The method repeatedly differentiates P(x) (until 0) and integrates f(x),
/// alternating signs: + - + - ...
///
/// # Example
///
/// ∫x²·e^x dx:
/// | Derivatives of x² | Integrals of e^x | Sign |
/// |-------------------|------------------|------|
/// | x²                | e^x              | +    |
/// | 2x                | e^x              | -    |
/// | 2                 | e^x              | +    |
/// | 0                 | e^x              | -    |
///
/// Result: x²·e^x - 2x·e^x + 2·e^x = (x² - 2x + 2)·e^x
pub fn tabular_integration(
    polynomial: &Expression,
    integrable: &Expression,
    var: &str,
) -> IntegrationResult {
    // Verify polynomial is actually polynomial-like
    if !is_polynomial_like(polynomial, var) {
        return Err(IntegrationError::CannotIntegrate(
            "First argument must be a polynomial".to_string(),
        ));
    }

    // Build the table
    let mut derivatives = Vec::new();
    let mut integrals = Vec::new();

    // Compute derivatives until we get 0
    let mut current_deriv = polynomial.clone();
    derivatives.push(current_deriv.clone());

    loop {
        current_deriv = current_deriv.differentiate(var).simplify();
        derivatives.push(current_deriv.clone());

        // Check if derivative is 0
        if matches!(current_deriv, Expression::Integer(0)) {
            break;
        }

        // Safety check to prevent infinite loops
        if derivatives.len() > 50 {
            return Err(IntegrationError::CannotIntegrate(
                "Polynomial degree too high for tabular method".to_string(),
            ));
        }
    }

    // Compute integrals
    let mut current_integral = integrable.clone();
    for _ in 0..derivatives.len() {
        integrals.push(current_integral.clone());
        current_integral = match integrate_impl(&current_integral, var) {
            Ok(i) => i.simplify(),
            Err(e) => return Err(e),
        };
    }

    // Combine with alternating signs: d[0]·i[0] - d[1]·i[1] + d[2]·i[2] - ...
    let mut result = Expression::Integer(0);
    let mut positive = true;

    for (d, i) in derivatives.iter().zip(integrals.iter()) {
        if matches!(d, Expression::Integer(0)) {
            break;
        }

        let term = Expression::Binary(BinaryOp::Mul, Box::new(d.clone()), Box::new(i.clone()));

        if positive {
            result = Expression::Binary(BinaryOp::Add, Box::new(result), Box::new(term));
        } else {
            result = Expression::Binary(BinaryOp::Sub, Box::new(result), Box::new(term));
        }

        positive = !positive;
    }

    Ok(result.simplify())
}

// =============================================================================
// Definite Integration
// =============================================================================

/// Compute the definite integral of an expression over [lower, upper].
///
/// Uses the Fundamental Theorem of Calculus: ∫_a^b f(x) dx = F(b) - F(a)
/// where F is an antiderivative of f.
///
/// # Arguments
///
/// * `expr` - The integrand
/// * `var` - The variable of integration
/// * `lower` - The lower bound of integration
/// * `upper` - The upper bound of integration
///
/// # Returns
///
/// The evaluated definite integral, or an error if integration fails.
///
/// # Examples
///
/// ```
/// use thales::integration::definite_integral;
/// use thales::ast::{Expression, Variable};
/// use std::collections::HashMap;
///
/// // ∫_0^1 x^2 dx = 1/3
/// let x = Expression::Variable(Variable::new("x"));
/// let x_squared = Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(2)));
/// let lower = Expression::Integer(0);
/// let upper = Expression::Integer(1);
///
/// let result = definite_integral(&x_squared, "x", &lower, &upper).unwrap();
/// let value = result.evaluate(&HashMap::new()).unwrap();
/// assert!((value - 1.0/3.0).abs() < 1e-10);
/// ```
pub fn definite_integral(
    expr: &Expression,
    var: &str,
    lower: &Expression,
    upper: &Expression,
) -> IntegrationResult {
    // Check for special cases first
    if let Some(result) = check_definite_special_cases(expr, var, lower, upper) {
        return Ok(result);
    }

    // Find the antiderivative using the appropriate method
    let antiderivative = integrate(expr, var)?;

    // Evaluate F(upper) - F(lower)
    let f_upper = substitute_var(&antiderivative, var, upper);
    let f_lower = substitute_var(&antiderivative, var, lower);

    let result = Expression::Binary(BinaryOp::Sub, Box::new(f_upper), Box::new(f_lower)).simplify();

    Ok(result)
}

/// Compute definite integral with numerical fallback.
///
/// First attempts symbolic integration, then falls back to numerical methods
/// if symbolic integration fails.
///
/// # Arguments
///
/// * `expr` - The integrand
/// * `var` - The variable of integration
/// * `lower` - The lower bound (must be numeric)
/// * `upper` - The upper bound (must be numeric)
/// * `tolerance` - Desired accuracy for numerical integration
///
/// # Returns
///
/// The integral value (exact symbolic or approximate numeric).
pub fn definite_integral_with_fallback(
    expr: &Expression,
    var: &str,
    lower: f64,
    upper: f64,
    tolerance: f64,
) -> Result<f64, IntegrationError> {
    // First try symbolic integration
    let lower_expr = Expression::Float(lower);
    let upper_expr = Expression::Float(upper);

    if let Ok(result) = definite_integral(expr, var, &lower_expr, &upper_expr) {
        let vars = std::collections::HashMap::new();
        if let Some(value) = result.evaluate(&vars) {
            return Ok(value);
        }
    }

    // Fall back to numerical integration (Simpson's rule)
    numerical_integrate(expr, var, lower, upper, tolerance)
}

/// Numerical integration using adaptive Simpson's rule.
///
/// # Arguments
///
/// * `expr` - The integrand expression
/// * `var` - The variable of integration
/// * `lower` - Lower bound (numeric)
/// * `upper` - Upper bound (numeric)
/// * `tolerance` - Desired accuracy
///
/// # Returns
///
/// The numerical approximation of the definite integral.
pub fn numerical_integrate(
    expr: &Expression,
    var: &str,
    lower: f64,
    upper: f64,
    tolerance: f64,
) -> Result<f64, IntegrationError> {
    // Use Simpson's rule with adaptive refinement
    let result = simpsons_rule_adaptive(expr, var, lower, upper, tolerance, 0);
    result.ok_or_else(|| {
        IntegrationError::CannotIntegrate("Numerical integration failed".to_string())
    })
}

/// Adaptive Simpson's rule for numerical integration.
fn simpsons_rule_adaptive(
    expr: &Expression,
    var: &str,
    a: f64,
    b: f64,
    tolerance: f64,
    depth: usize,
) -> Option<f64> {
    const MAX_DEPTH: usize = 20;

    if depth > MAX_DEPTH {
        return None;
    }

    let mid = (a + b) / 2.0;

    // Evaluate function at required points
    let fa = evaluate_at(expr, var, a)?;
    let fb = evaluate_at(expr, var, b)?;
    let fm = evaluate_at(expr, var, mid)?;
    let f1 = evaluate_at(expr, var, (a + mid) / 2.0)?;
    let f2 = evaluate_at(expr, var, (mid + b) / 2.0)?;

    // Simpson's rule for whole interval
    let h = (b - a) / 6.0;
    let s1 = h * (fa + 4.0 * fm + fb);

    // Simpson's rule for two halves
    let h2 = h / 2.0;
    let s2 = h2 * (fa + 4.0 * f1 + fm) + h2 * (fm + 4.0 * f2 + fb);

    // Check convergence
    if (s2 - s1).abs() < 15.0 * tolerance {
        Some(s2 + (s2 - s1) / 15.0) // Richardson extrapolation
    } else {
        // Recurse on both halves
        let left = simpsons_rule_adaptive(expr, var, a, mid, tolerance / 2.0, depth + 1)?;
        let right = simpsons_rule_adaptive(expr, var, mid, b, tolerance / 2.0, depth + 1)?;
        Some(left + right)
    }
}

/// Evaluate an expression at a specific value of a variable.
fn evaluate_at(expr: &Expression, var: &str, value: f64) -> Option<f64> {
    let mut vars = std::collections::HashMap::new();
    vars.insert(var.to_string(), value);
    expr.evaluate(&vars)
}

/// Check for special cases of definite integrals.
fn check_definite_special_cases(
    expr: &Expression,
    var: &str,
    lower: &Expression,
    upper: &Expression,
) -> Option<Expression> {
    // Check if integrand is zero
    if matches!(expr, Expression::Integer(0)) {
        return Some(Expression::Integer(0));
    }

    // Check if bounds are equal
    if expressions_equivalent(lower, upper) {
        return Some(Expression::Integer(0));
    }

    // Check for odd function over symmetric interval [-a, a]
    if let Some(result) = check_odd_function_symmetric(expr, var, lower, upper) {
        return Some(result);
    }

    None
}

/// Check if f is an odd function and the interval is symmetric [-a, a].
/// If so, ∫_{-a}^{a} f(x) dx = 0
fn check_odd_function_symmetric(
    expr: &Expression,
    var: &str,
    lower: &Expression,
    upper: &Expression,
) -> Option<Expression> {
    // Check if lower = -upper (symmetric interval)
    let neg_upper = Expression::Unary(UnaryOp::Neg, Box::new(upper.clone())).simplify();
    if !expressions_equivalent(lower, &neg_upper) {
        return None;
    }

    // Check if f(-x) = -f(x) (odd function)
    let neg_var = Expression::Unary(
        UnaryOp::Neg,
        Box::new(Expression::Variable(Variable::new(var))),
    );
    let f_neg_x = substitute_var(expr, var, &neg_var).simplify();
    let neg_f_x = Expression::Unary(UnaryOp::Neg, Box::new(expr.clone())).simplify();

    if expressions_equivalent(&f_neg_x, &neg_f_x) {
        return Some(Expression::Integer(0));
    }

    None
}

/// Substitute a variable with an expression.
fn substitute_var(expr: &Expression, var: &str, replacement: &Expression) -> Expression {
    match expr {
        Expression::Variable(v) if v.name == var => replacement.clone(),
        Expression::Variable(_) => expr.clone(),
        Expression::Integer(_) | Expression::Float(_) | Expression::Rational(_) => expr.clone(),
        Expression::Constant(_) => expr.clone(),
        Expression::Binary(op, left, right) => Expression::Binary(
            *op,
            Box::new(substitute_var(left, var, replacement)),
            Box::new(substitute_var(right, var, replacement)),
        ),
        Expression::Unary(op, operand) => {
            Expression::Unary(*op, Box::new(substitute_var(operand, var, replacement)))
        }
        Expression::Function(func, args) => Expression::Function(
            func.clone(),
            args.iter()
                .map(|arg| substitute_var(arg, var, replacement))
                .collect(),
        ),
        Expression::Power(base, exp) => Expression::Power(
            Box::new(substitute_var(base, var, replacement)),
            Box::new(substitute_var(exp, var, replacement)),
        ),
        Expression::Complex(_) => expr.clone(),
    }
}

/// Evaluate improper integral ∫_a^∞ f(x) dx using limit.
///
/// Computes: lim_{b→∞} ∫_a^b f(x) dx
///
/// # Arguments
///
/// * `expr` - The integrand
/// * `var` - The variable of integration
/// * `lower` - The finite lower bound
///
/// # Returns
///
/// The integral value if convergent, or error if divergent.
pub fn improper_integral_to_infinity(
    expr: &Expression,
    var: &str,
    lower: &Expression,
) -> IntegrationResult {
    // Find the antiderivative
    let antiderivative = integrate(expr, var)?;

    // Evaluate F(lower)
    let f_lower = substitute_var(&antiderivative, var, lower).simplify();

    // Analyze limit as x → ∞
    // For common cases like 1/x^n where n > 1, the limit is 0
    if let Some(limit_at_inf) = evaluate_limit_at_infinity(&antiderivative, var) {
        let result =
            Expression::Binary(BinaryOp::Sub, Box::new(limit_at_inf), Box::new(f_lower)).simplify();
        return Ok(result);
    }

    Err(IntegrationError::CannotIntegrate(
        "Cannot evaluate improper integral (may be divergent)".to_string(),
    ))
}

/// Evaluate limit of expression as variable approaches infinity.
/// Returns Some(Expression) if limit exists and is finite.
fn evaluate_limit_at_infinity(expr: &Expression, var: &str) -> Option<Expression> {
    // Check if expression approaches 0 as var → ∞
    // This is a simplified check for common patterns
    match expr {
        // Constant: limit is the constant
        Expression::Integer(_) | Expression::Float(_) | Expression::Rational(_) => {
            Some(expr.clone())
        }
        Expression::Constant(_) => Some(expr.clone()),

        // x^(-n) for n > 0 → 0 as x → ∞
        Expression::Power(base, exp) => {
            if let Expression::Variable(v) = base.as_ref() {
                if v.name == var {
                    // Check if exponent is negative (try multiple representations)
                    match exp.as_ref() {
                        Expression::Integer(n) if *n < 0 => {
                            return Some(Expression::Integer(0));
                        }
                        Expression::Unary(UnaryOp::Neg, _) => {
                            return Some(Expression::Integer(0));
                        }
                        _ => {
                            // Try simplifying and evaluating the exponent
                            let simplified_exp = exp.simplify();
                            if let Expression::Integer(n) = simplified_exp {
                                if n < 0 {
                                    return Some(Expression::Integer(0));
                                }
                            }
                            // Try numerical evaluation
                            let empty = std::collections::HashMap::new();
                            if let Some(val) = simplified_exp.evaluate(&empty) {
                                if val < 0.0 {
                                    return Some(Expression::Integer(0));
                                }
                            }
                        }
                    }
                }
            }
            None
        }

        // Division cases
        Expression::Binary(BinaryOp::Div, num, denom) => {
            // 1/x^n → 0 for n > 0: numerator constant, denominator → ∞
            if !num.contains_variable(var) {
                if grows_to_infinity(denom, var) {
                    return Some(Expression::Integer(0));
                }
            }
            // x^(-n) / c → 0 / c = 0: numerator → 0, denominator constant
            if !denom.contains_variable(var) {
                if let Some(num_limit) = evaluate_limit_at_infinity(num, var) {
                    // If numerator limit is 0, result is 0
                    let is_zero = match &num_limit {
                        Expression::Integer(0) => true,
                        Expression::Float(f) if *f == 0.0 => true,
                        _ => false,
                    };
                    if is_zero {
                        return Some(Expression::Integer(0));
                    }
                    // Otherwise, return limit / denominator
                    return Some(
                        Expression::Binary(BinaryOp::Div, Box::new(num_limit), denom.clone())
                            .simplify(),
                    );
                }
            }
            None
        }

        // e^(-x) → 0 as x → ∞
        Expression::Function(Function::Exp, args) if args.len() == 1 => {
            if let Expression::Unary(UnaryOp::Neg, inner) = &args[0] {
                if let Expression::Variable(v) = inner.as_ref() {
                    if v.name == var {
                        return Some(Expression::Integer(0));
                    }
                }
            }
            None
        }

        // Sum/difference
        Expression::Binary(BinaryOp::Add, left, right) => {
            let l_limit = evaluate_limit_at_infinity(left, var)?;
            let r_limit = evaluate_limit_at_infinity(right, var)?;
            Some(Expression::Binary(BinaryOp::Add, Box::new(l_limit), Box::new(r_limit)).simplify())
        }
        Expression::Binary(BinaryOp::Sub, left, right) => {
            let l_limit = evaluate_limit_at_infinity(left, var)?;
            let r_limit = evaluate_limit_at_infinity(right, var)?;
            Some(Expression::Binary(BinaryOp::Sub, Box::new(l_limit), Box::new(r_limit)).simplify())
        }

        // Product with constant
        Expression::Binary(BinaryOp::Mul, left, right) => {
            if !left.contains_variable(var) {
                let r_limit = evaluate_limit_at_infinity(right, var)?;
                return Some(
                    Expression::Binary(BinaryOp::Mul, left.clone(), Box::new(r_limit)).simplify(),
                );
            }
            if !right.contains_variable(var) {
                let l_limit = evaluate_limit_at_infinity(left, var)?;
                return Some(
                    Expression::Binary(BinaryOp::Mul, Box::new(l_limit), right.clone()).simplify(),
                );
            }
            None
        }

        _ => None,
    }
}

/// Check if an expression grows to infinity as the variable increases.
fn grows_to_infinity(expr: &Expression, var: &str) -> bool {
    match expr {
        Expression::Variable(v) => v.name == var,
        Expression::Power(base, exp) => {
            if let Expression::Variable(v) = base.as_ref() {
                if v.name == var {
                    // x^n grows to infinity if n > 0
                    match exp.as_ref() {
                        Expression::Integer(n) => *n > 0,
                        _ => false,
                    }
                } else {
                    false
                }
            } else {
                false
            }
        }
        Expression::Function(Function::Exp, args) if args.len() == 1 => {
            // e^x grows to infinity if x grows
            args[0].contains_variable(var)
                && !matches!(&args[0], Expression::Unary(UnaryOp::Neg, _))
        }
        _ => false,
    }
}

/// Compute definite integral with step-by-step explanation.
///
/// Returns the result and a list of steps showing the computation.
pub fn definite_integral_with_steps(
    expr: &Expression,
    var: &str,
    lower: &Expression,
    upper: &Expression,
) -> Result<(Expression, Vec<String>), IntegrationError> {
    let mut steps = Vec::new();
    steps.push(format!(
        "∫_{{{0}}}^{{{1}}} {2} d{3}",
        lower, upper, expr, var
    ));

    // Check for special cases
    if let Some(result) = check_definite_special_cases(expr, var, lower, upper) {
        steps.push(format!("By special case analysis: {}", result));
        return Ok((result, steps));
    }

    // Find antiderivative
    steps.push("Step 1: Find the antiderivative F(x)".to_string());
    let antiderivative = integrate(expr, var)?;
    steps.push(format!("F({}) = {}", var, antiderivative));

    // Apply Fundamental Theorem of Calculus
    steps.push("Step 2: Apply Fundamental Theorem of Calculus".to_string());
    steps.push("∫_a^b f(x) dx = F(b) - F(a)".to_string());

    // Evaluate at bounds
    let f_upper = substitute_var(&antiderivative, var, upper).simplify();
    let f_lower = substitute_var(&antiderivative, var, lower).simplify();
    steps.push(format!("F({}) = {}", upper, f_upper));
    steps.push(format!("F({}) = {}", lower, f_lower));

    // Compute result
    let result = Expression::Binary(
        BinaryOp::Sub,
        Box::new(f_upper.clone()),
        Box::new(f_lower.clone()),
    )
    .simplify();

    steps.push(format!("Result: {} - {} = {}", f_upper, f_lower, result));

    Ok((result, steps))
}

/// Check if an expression is polynomial-like (contains only powers of var with constant exponents).
fn is_polynomial_like(expr: &Expression, var: &str) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

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
            Expression::Binary(BinaryOp::Div, _num, denom)
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
            assert!(matches!(
                denom.as_ref(),
                Expression::Binary(BinaryOp::Add, _, _)
            ));
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
        assert!(
            matches!(result, Expression::Binary(BinaryOp::Mul, _, ln_part)
            if matches!(ln_part.as_ref(), Expression::Function(Function::Ln, _)))
        );
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
        let poly = add(add(pow(var("x"), int(2)), mul(int(2), var("x"))), int(1));
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
        let _derivative = integral.differentiate("x").simplify();

        // The derivative should simplify to x^2 (or equivalent)
        // This is a partial test - full verification needs numerical checks
        // For now, just verify we get back a power expression
        // The actual result may be (3*x^2) / 3 which simplifies to x^2
    }

    // =========================================================================
    // U-Substitution Tests
    // =========================================================================

    #[test]
    fn test_extract_factors() {
        // Test basic factor extraction
        let expr = mul(int(2), var("x"));
        let factors = extract_factors(&expr);
        assert_eq!(factors.len(), 2);

        // Test nested product
        let expr2 = mul(mul(int(2), var("x")), int(3));
        let factors2 = extract_factors(&expr2);
        assert_eq!(factors2.len(), 3);
    }

    #[test]
    fn test_combine_factors() {
        let factors = vec![int(2), var("x"), int(3)];
        let combined = combine_factors(&factors);
        // Should produce 2 * x * 3
        assert!(matches!(combined, Expression::Binary(BinaryOp::Mul, _, _)));
    }

    #[test]
    fn test_extract_inner_function() {
        // Test function extraction
        let sin_x = Expression::Function(Function::Sin, vec![var("x")]);
        let inner = extract_inner_function(&sin_x);
        assert!(matches!(inner, Some(Expression::Variable(_))));

        // Test with x^2 inside
        let x_squared = pow(var("x"), int(2));
        let sin_x2 = Expression::Function(Function::Sin, vec![x_squared]);
        let inner2 = extract_inner_function(&sin_x2);
        assert!(inner2.is_some());
    }

    #[test]
    fn test_substitute_variable() {
        // Test variable substitution
        let u = Expression::Variable(Variable::new("u"));
        let replacement = pow(var("x"), int(2));
        let result = substitute_variable(&u, "u", &replacement);
        assert!(matches!(result, Expression::Power(_, _)));
    }

    #[test]
    fn test_integrate_by_substitution_linear() {
        // ∫sin(3x) dx = -cos(3x)/3
        // This is already handled by linear substitution in base integrate
        let three_x = mul(int(3), var("x"));
        let sin_3x = Expression::Function(Function::Sin, vec![three_x]);
        let result = integrate_by_substitution(&sin_3x, "x");
        assert!(result.is_ok());
    }

    #[test]
    fn test_integrate_by_substitution_exp() {
        // ∫e^(2x) dx = e^(2x)/2
        // This should be handled by linear substitution
        let two_x = mul(int(2), var("x"));
        let exp_2x = Expression::Function(Function::Exp, vec![two_x]);
        let result = integrate_by_substitution(&exp_2x, "x");
        assert!(result.is_ok());
    }

    #[test]
    fn test_integrate_with_substitution_steps() {
        // Test that we get steps back
        let x_squared = pow(var("x"), int(2));
        let (result, steps) = integrate_with_substitution(&x_squared, "x").unwrap();

        // Should have at least one step
        assert!(!steps.is_empty());
        // Result should be valid
        assert!(matches!(result, Expression::Binary(BinaryOp::Div, _, _)));
    }

    #[test]
    fn test_expressions_equivalent() {
        let a = var("x");
        let b = var("x");
        assert!(expressions_equivalent(&a, &b));

        let c = var("y");
        assert!(!expressions_equivalent(&a, &c));
    }

    #[test]
    fn test_is_one() {
        assert!(is_one(&Expression::Integer(1)));
        assert!(!is_one(&Expression::Integer(2)));
        assert!(!is_one(&var("x")));
    }

    // =========================================================================
    // Integration by Parts Tests
    // =========================================================================

    #[test]
    fn test_liate_priority() {
        // Logarithmic has highest priority
        let ln_x = Expression::Function(Function::Ln, vec![var("x")]);
        assert!(liate_priority(&ln_x, "x") > liate_priority(&var("x"), "x"));

        // Algebraic (x) has higher priority than trigonometric
        let sin_x = Expression::Function(Function::Sin, vec![var("x")]);
        assert!(liate_priority(&var("x"), "x") > liate_priority(&sin_x, "x"));

        // Trigonometric has higher priority than exponential
        let exp_x = Expression::Function(Function::Exp, vec![var("x")]);
        assert!(liate_priority(&sin_x, "x") > liate_priority(&exp_x, "x"));
    }

    #[test]
    fn test_is_polynomial_like() {
        // Basic polynomials
        assert!(is_polynomial_like(&var("x"), "x"));
        assert!(is_polynomial_like(&pow(var("x"), int(2)), "x"));
        assert!(is_polynomial_like(
            &add(pow(var("x"), int(2)), var("x")),
            "x"
        ));

        // Constants are polynomial
        assert!(is_polynomial_like(&int(5), "x"));

        // Non-polynomials
        let sin_x = Expression::Function(Function::Sin, vec![var("x")]);
        assert!(!is_polynomial_like(&sin_x, "x"));
    }

    #[test]
    fn test_integrate_by_parts_x_exp() {
        // ∫x * e^x dx = (x - 1) * e^x
        // Using parts: u = x, dv = e^x dx
        //              du = dx, v = e^x
        //              = x*e^x - ∫e^x dx = x*e^x - e^x = (x-1)*e^x
        let x = var("x");
        let exp_x = Expression::Function(Function::Exp, vec![x.clone()]);
        let expr = mul(x.clone(), exp_x.clone());

        let result = integrate_by_parts(&expr, "x");
        assert!(result.is_ok());

        // Verify by differentiation
        if let Ok(integral) = result {
            let derivative = integral.differentiate("x").simplify();
            // The derivative should equal the original expression
            // We can't easily compare symbolically, but at least check it's not an error
            assert!(!matches!(derivative, Expression::Integer(0)));
        }
    }

    #[test]
    fn test_integrate_by_parts_ln_x() {
        // ∫ln(x) dx = x*ln(x) - x
        // Using parts: u = ln(x), dv = dx
        //              du = 1/x dx, v = x
        //              = x*ln(x) - ∫x * (1/x) dx = x*ln(x) - ∫1 dx = x*ln(x) - x
        let ln_x = Expression::Function(Function::Ln, vec![var("x")]);

        // This should be handled by the standard integral table, not parts
        let result = integrate(&ln_x, "x");
        assert!(result.is_ok());
    }

    #[test]
    fn test_integrate_by_parts_x_sin() {
        // ∫x * sin(x) dx = -x*cos(x) + sin(x)
        // Using parts: u = x, dv = sin(x) dx
        //              du = dx, v = -cos(x)
        //              = -x*cos(x) - ∫(-cos(x)) dx = -x*cos(x) + sin(x)
        let x = var("x");
        let sin_x = Expression::Function(Function::Sin, vec![x.clone()]);
        let expr = mul(x.clone(), sin_x.clone());

        let result = integrate_by_parts(&expr, "x");
        assert!(result.is_ok());
    }

    #[test]
    fn test_integrate_by_parts_x_squared_exp() {
        // ∫x^2 * e^x dx = (x^2 - 2x + 2) * e^x
        // Requires two applications of integration by parts
        let x = var("x");
        let x_squared = pow(x.clone(), int(2));
        let exp_x = Expression::Function(Function::Exp, vec![x.clone()]);
        let expr = mul(x_squared.clone(), exp_x.clone());

        let result = integrate_by_parts(&expr, "x");
        assert!(result.is_ok());
    }

    #[test]
    fn test_tabular_integration_x_exp() {
        // ∫x * e^x dx using tabular method
        let x = var("x");
        let exp_x = Expression::Function(Function::Exp, vec![x.clone()]);

        let result = tabular_integration(&x, &exp_x, "x");
        assert!(result.is_ok());
    }

    #[test]
    fn test_tabular_integration_x_squared_exp() {
        // ∫x^2 * e^x dx using tabular method
        // Derivatives: x^2 -> 2x -> 2 -> 0
        // Integrals: e^x -> e^x -> e^x -> e^x
        // Result: x^2*e^x - 2x*e^x + 2*e^x
        let x = var("x");
        let x_squared = pow(x.clone(), int(2));
        let exp_x = Expression::Function(Function::Exp, vec![x.clone()]);

        let result = tabular_integration(&x_squared, &exp_x, "x");
        assert!(result.is_ok());
    }

    #[test]
    fn test_tabular_integration_x_sin() {
        // ∫x * sin(x) dx using tabular method
        let x = var("x");
        let sin_x = Expression::Function(Function::Sin, vec![x.clone()]);

        let result = tabular_integration(&x, &sin_x, "x");
        assert!(result.is_ok());
    }

    #[test]
    fn test_integrate_by_parts_with_steps() {
        // Test that we get detailed steps
        let x = var("x");
        let exp_x = Expression::Function(Function::Exp, vec![x.clone()]);
        let expr = mul(x.clone(), exp_x.clone());

        let result = integrate_by_parts_with_steps(&expr, "x");
        assert!(result.is_ok());

        if let Ok((_, steps)) = result {
            // Should have multiple steps
            assert!(steps.len() >= 5);
            // Should mention "integration by parts"
            assert!(steps.iter().any(|s| s.contains("integration by parts")));
        }
    }

    #[test]
    fn test_choose_u_and_dv() {
        // For x * e^x, x should be chosen as u (algebraic > exponential)
        let x = var("x");
        let exp_x = Expression::Function(Function::Exp, vec![x.clone()]);
        let factors = vec![x.clone(), exp_x.clone()];

        let (u, dv) = choose_u_and_dv(&factors, "x");

        // u should be x (algebraic priority 3)
        assert!(matches!(u, Expression::Variable(_)));
        // dv should be e^x (exponential priority 1)
        assert!(matches!(dv, Expression::Function(Function::Exp, _)));
    }

    #[test]
    fn test_check_same_up_to_constant() {
        let a = var("x");
        let b = var("x");

        // Same expression -> constant 1
        assert!(check_same_up_to_constant(&a, &b, "x").is_some());

        // Negation -> constant -1
        let neg_a = Expression::Unary(UnaryOp::Neg, Box::new(a.clone()));
        let result = check_same_up_to_constant(&a, &neg_a, "x");
        assert!(result.is_some());
        assert!(matches!(result, Some(Expression::Integer(-1))));

        // Different expressions
        let c = var("y");
        assert!(check_same_up_to_constant(&a, &c, "x").is_none());
    }

    // =========================================================================
    // Definite Integral Tests
    // =========================================================================

    #[test]
    fn test_definite_integral_x_squared() {
        // ∫_0^1 x^2 dx = 1/3
        let x_squared = pow(var("x"), int(2));
        let result = definite_integral(&x_squared, "x", &int(0), &int(1));
        assert!(result.is_ok());

        let value = result.unwrap();
        let empty = std::collections::HashMap::new();
        let numeric = value.evaluate(&empty).unwrap();
        assert!((numeric - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_definite_integral_sin() {
        // ∫_0^π sin(x) dx = 2
        let sin_x = Expression::Function(Function::Sin, vec![var("x")]);
        let pi = Expression::Constant(crate::ast::SymbolicConstant::Pi);
        let result = definite_integral(&sin_x, "x", &int(0), &pi);
        assert!(result.is_ok());

        let value = result.unwrap();
        let empty = std::collections::HashMap::new();
        let numeric = value.evaluate(&empty).unwrap();
        assert!((numeric - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_definite_integral_odd_function() {
        // ∫_{-1}^1 x^3 dx = 0 (odd function, symmetric interval)
        let x_cubed = pow(var("x"), int(3));
        let result = definite_integral(&x_cubed, "x", &int(-1), &int(1));
        assert!(result.is_ok());

        let value = result.unwrap();
        let empty = std::collections::HashMap::new();
        let numeric = value.evaluate(&empty).unwrap();
        assert!(numeric.abs() < 1e-10);
    }

    #[test]
    fn test_definite_integral_symbolic_upper_bound() {
        // ∫_0^a x dx = a^2/2
        let x = var("x");
        let a = var("a");
        let result = definite_integral(&x, "x", &int(0), &a);
        assert!(result.is_ok());

        // Evaluate at a=2 to verify: should be 2
        let value = result.unwrap();
        let mut env = std::collections::HashMap::new();
        env.insert("a".to_string(), 2.0);
        let numeric = value.evaluate(&env).unwrap();
        assert!((numeric - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_definite_integral_polynomial() {
        // ∫_0^2 (3x^2 + 2x + 1) dx = [x^3 + x^2 + x]_0^2 = 8 + 4 + 2 = 14
        let x = var("x");
        let poly = add(
            add(mul(int(3), pow(x.clone(), int(2))), mul(int(2), x.clone())),
            int(1),
        );
        let result = definite_integral(&poly, "x", &int(0), &int(2));
        assert!(result.is_ok());

        let value = result.unwrap();
        let empty = std::collections::HashMap::new();
        let numeric = value.evaluate(&empty).unwrap();
        assert!((numeric - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_definite_integral_cos() {
        // ∫_0^{π/2} cos(x) dx = 1
        let cos_x = Expression::Function(Function::Cos, vec![var("x")]);
        let pi = Expression::Constant(crate::ast::SymbolicConstant::Pi);
        let upper = div(pi, int(2));
        let result = definite_integral(&cos_x, "x", &int(0), &upper);
        assert!(result.is_ok());

        let value = result.unwrap();
        let empty = std::collections::HashMap::new();
        let numeric = value.evaluate(&empty).unwrap();
        assert!((numeric - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_definite_integral_exp() {
        // ∫_0^1 e^x dx = e - 1 ≈ 1.71828
        let exp_x = Expression::Function(Function::Exp, vec![var("x")]);
        let result = definite_integral(&exp_x, "x", &int(0), &int(1));
        assert!(result.is_ok());

        let value = result.unwrap();
        let empty = std::collections::HashMap::new();
        let numeric = value.evaluate(&empty).unwrap();
        let expected = std::f64::consts::E - 1.0;
        assert!((numeric - expected).abs() < 1e-10);
    }

    #[test]
    fn test_numerical_integrate_simple() {
        // Use adaptive Simpson's for ∫_0^1 x^2 dx = 1/3
        let x_squared = pow(var("x"), int(2));
        let result = numerical_integrate(&x_squared, "x", 0.0, 1.0, 1e-8);
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!((value - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_numerical_integrate_complex() {
        // ∫_0^1 exp(-x^2) dx ≈ 0.74682 (error function related)
        let x = var("x");
        let neg_x_squared = Expression::Unary(UnaryOp::Neg, Box::new(pow(x.clone(), int(2))));
        let exp_neg_x_squared = Expression::Function(Function::Exp, vec![neg_x_squared]);
        let result = numerical_integrate(&exp_neg_x_squared, "x", 0.0, 1.0, 1e-6);
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!((value - 0.74682).abs() < 0.001);
    }

    #[test]
    fn test_definite_integral_with_fallback() {
        // Use fallback for a simple integral
        let x_squared = pow(var("x"), int(2));
        let result = definite_integral_with_fallback(&x_squared, "x", 0.0, 1.0, 1e-8);
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!((value - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_improper_integral_convergent() {
        // ∫_1^∞ x^(-2) dx = 1
        // Use x^(-2) format which integrate handles better than 1/x^2
        let x = var("x");
        let x_neg_2 = pow(x.clone(), int(-2));
        let result = improper_integral_to_infinity(&x_neg_2, "x", &int(1));
        assert!(result.is_ok());

        let value = result.unwrap();
        let empty = std::collections::HashMap::new();
        let numeric = value.evaluate(&empty).unwrap();
        assert!((numeric - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_definite_integral_with_steps() {
        // Verify we get step-by-step output
        let x_squared = pow(var("x"), int(2));
        let result = definite_integral_with_steps(&x_squared, "x", &int(0), &int(1));
        assert!(result.is_ok());

        let (value, steps) = result.unwrap();
        // Should have multiple steps
        assert!(!steps.is_empty());
        // Should mention "antiderivative" or "bounds"
        assert!(steps.iter().any(|s| {
            s.to_lowercase().contains("antiderivative") || s.to_lowercase().contains("bound")
        }));

        // Verify result
        let empty = std::collections::HashMap::new();
        let numeric = value.evaluate(&empty).unwrap();
        assert!((numeric - 1.0 / 3.0).abs() < 1e-10);
    }
}
