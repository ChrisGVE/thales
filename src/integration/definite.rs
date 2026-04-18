//! Definite integration and numerical methods.
//!
//! This module provides definite integration using the Fundamental Theorem
//! of Calculus, numerical integration via adaptive Simpson's rule,
//! and improper integral evaluation.

use crate::ast::{BinaryOp, Expression, Function, UnaryOp, Variable};

use super::{IntegrationError, IntegrationResult};

/// Compute a canonical string key for an expression, normalising the order
/// of operands in commutative operations so that, e.g., `x * 2` and `2 * x`
/// produce the same key. Used only for equivalence testing.
fn canonical_key(expr: &Expression) -> String {
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
        other => format!("{}", other),
    }
}

/// Check if two expressions are structurally equivalent under commutativity
/// of `Add` and `Mul`.
fn expressions_equivalent(a: &Expression, b: &Expression) -> bool {
    canonical_key(a) == canonical_key(b)
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
    let antiderivative = super::integrate(expr, var)?;

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
    // Split the interval into initial sub-intervals to avoid aliasing when
    // oscillatory integrands happen to be constant at evenly-spaced points.
    const INITIAL_SUBDIVISIONS: usize = 8;
    let h = (upper - lower) / INITIAL_SUBDIVISIONS as f64;
    let tol_per = tolerance / INITIAL_SUBDIVISIONS as f64;
    let mut total = 0.0;
    for i in 0..INITIAL_SUBDIVISIONS {
        let a = lower + i as f64 * h;
        let b = a + h;
        let part = simpsons_rule_adaptive(expr, var, a, b, tol_per, 0).ok_or_else(|| {
            IntegrationError::CannotIntegrate("Numerical integration failed".to_string())
        })?;
        total += part;
    }
    Ok(total)
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
    let antiderivative = super::integrate(expr, var)?;

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

        // Unary negation: lim(-f) = -lim(f)
        Expression::Unary(UnaryOp::Neg, inner) => {
            let inner_limit = evaluate_limit_at_infinity(inner, var)?;
            Some(Expression::Unary(UnaryOp::Neg, Box::new(inner_limit)).simplify())
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
    let antiderivative = super::integrate(expr, var)?;
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
