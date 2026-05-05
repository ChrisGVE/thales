//! Definite integration and numerical methods.
//!
//! This module provides definite integration using the Fundamental Theorem
//! of Calculus, numerical integration via adaptive Simpson's rule,
//! and improper integral evaluation.
//!
//! All internal logic operates on `Arc<Expr>`. Public functions are
//! Expression-typed at the I/O boundary; callers compile/decompile there.

use std::sync::Arc;

use num::traits::{One, Zero};

use crate::ast::Expression;
use crate::numeric::compile::{compile, decompile};
use crate::numeric::evaluation::evaluate;
use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::normalize;
use crate::numeric::substitute::substitute;
use crate::numeric::SymbolId;

use super::{IntegrationError, IntegrationResult};

// =============================================================================
// Arc<Expr> internal helpers
// =============================================================================

/// Compute a canonical string key for a `Arc<Expr>`, normalising the order of
/// operands in commutative operations (Add, Mul). Used only for equivalence
/// testing inside this module.
fn canonical_key_arc(expr: &Arc<Expr>) -> String {
    match expr.as_ref() {
        Expr::Add(node) => {
            let mut parts: Vec<String> = node
                .terms
                .iter()
                .map(|(t, _c)| canonical_key_arc(t))
                .collect();
            // constant part
            if !node.constant.is_zero() {
                parts.push(format!("{}", node.constant));
            }
            parts.sort();
            format!("(+{})", parts.join("+"))
        }
        Expr::Mul(node) => {
            let mut parts: Vec<String> = node
                .factors
                .iter()
                .map(|(b, e)| format!("({}^{})", canonical_key_arc(b), canonical_key_arc(e)))
                .collect();
            if !node.coeff.is_one() {
                parts.push(format!("{}", node.coeff));
            }
            parts.sort();
            format!("(*{})", parts.join("*"))
        }
        Expr::Pow(base, exp) => {
            format!("({}^{})", canonical_key_arc(base), canonical_key_arc(exp))
        }
        Expr::Func(f, args) => {
            let arg_keys: Vec<_> = args.iter().map(canonical_key_arc).collect();
            format!("{:?}({})", f, arg_keys.join(","))
        }
        other => format!("{}", other),
    }
}

/// Check if two `Arc<Expr>` are structurally equivalent under commutativity
/// of Add and Mul.
fn exprs_equivalent(a: &Arc<Expr>, b: &Arc<Expr>) -> bool {
    canonical_key_arc(a) == canonical_key_arc(b)
}

/// Substitute variable `var` with `replacement` in `expr`.
/// Delegates to `numeric::substitute`.
fn subst_var(expr: &Arc<Expr>, var: SymbolId, replacement: &Arc<Expr>) -> Arc<Expr> {
    substitute(expr, var, replacement)
}

/// Check for special cases of definite integrals.
/// Returns `Some(Arc<Expr>)` when a short-circuit value is known.
fn check_definite_special_cases_arc(
    expr: &Arc<Expr>,
    var: SymbolId,
    lower: &Arc<Expr>,
    upper: &Arc<Expr>,
) -> Option<Arc<Expr>> {
    // Integrand is zero
    if expr.is_zero() {
        return Some(Expr::int(0));
    }

    // Bounds are equal
    if exprs_equivalent(lower, upper) {
        return Some(Expr::int(0));
    }

    // Odd function over symmetric interval [-a, a]
    check_odd_function_symmetric_arc(expr, var, lower, upper)
}

/// Check if f is an odd function and the interval is symmetric [-a, a].
/// If so, ∫_{-a}^{a} f(x) dx = 0.
fn check_odd_function_symmetric_arc(
    expr: &Arc<Expr>,
    var: SymbolId,
    lower: &Arc<Expr>,
    upper: &Arc<Expr>,
) -> Option<Arc<Expr>> {
    // Check if lower = -upper (symmetric interval)
    let neg_upper = normalize::neg(upper.clone());
    let neg_upper_simplified = crate::numeric::simplify::simplify(&neg_upper);
    if !exprs_equivalent(lower, &neg_upper_simplified) {
        return None;
    }

    // Check if f(-x) = -f(x) (odd function)
    let neg_var = normalize::neg(Arc::new(Expr::Symbol(var)));
    let f_neg_x = crate::numeric::simplify::simplify(&subst_var(expr, var, &neg_var));
    let neg_f_x = crate::numeric::simplify::simplify(&normalize::neg(expr.clone()));

    if exprs_equivalent(&f_neg_x, &neg_f_x) {
        return Some(Expr::int(0));
    }

    None
}

/// Evaluate expression at a specific numeric value of a variable.
fn evaluate_at_arc(expr: &Arc<Expr>, var: SymbolId, value: f64) -> Option<f64> {
    let mut bindings = std::collections::HashMap::new();
    bindings.insert(var, value);
    evaluate(expr, &bindings)
}

/// Return true if `expr` syntactically contains the symbol `var`.
fn contains_var_arc(expr: &Arc<Expr>, var: SymbolId) -> bool {
    match expr.as_ref() {
        Expr::Symbol(s) => *s == var,
        Expr::Integer(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Complex(_)
        | Expr::Constant(_) => false,
        Expr::Add(node) => node.terms.keys().any(|t| contains_var_arc(t, var)),
        Expr::Mul(node) => node
            .factors
            .iter()
            .any(|(b, e)| contains_var_arc(b, var) || contains_var_arc(e, var)),
        Expr::Pow(b, e) => contains_var_arc(b, var) || contains_var_arc(e, var),
        Expr::Func(_, args) => args.iter().any(|a| contains_var_arc(a, var)),
    }
}

/// Evaluate limit of expression as variable approaches infinity.
/// Returns `Some(Arc<Expr>)` if the limit exists and is finite.
fn evaluate_limit_at_infinity_arc(expr: &Arc<Expr>, var: SymbolId) -> Option<Arc<Expr>> {
    match expr.as_ref() {
        // Constants: limit is the constant
        Expr::Integer(_) | Expr::Rational(_) | Expr::Float(_) | Expr::Constant(_) => {
            Some(expr.clone())
        }

        // x^n for n < 0 → 0 as x → ∞
        Expr::Pow(base, exp) => {
            if let Expr::Symbol(s) = base.as_ref() {
                if *s == var {
                    match exp.as_ref() {
                        Expr::Integer(n) => {
                            if n.to_i64().map(|v| v < 0).unwrap_or(false) {
                                return Some(Expr::int(0));
                            }
                        }
                        _ => {
                            let simplified_exp = crate::numeric::simplify::simplify(exp);
                            if let Expr::Integer(n) = simplified_exp.as_ref() {
                                if n.to_i64().map(|v| v < 0).unwrap_or(false) {
                                    return Some(Expr::int(0));
                                }
                            }
                            let empty = std::collections::HashMap::new();
                            if let Some(val) = evaluate(&simplified_exp, &empty) {
                                if val < 0.0 {
                                    return Some(Expr::int(0));
                                }
                            }
                        }
                    }
                }
            }
            None
        }

        // Mul node with negative coeff and single factor: e.g. -x → handled by Mul
        Expr::Mul(node) => {
            // Check for negation (coeff=-1 * something)
            if !contains_var_arc(expr, var) {
                return Some(expr.clone());
            }
            // coeff * f(x): if f(x) → 0, whole thing → 0
            // Reconstruct factors without coeff and recurse
            // Build expression from factors
            let mut factors_expr: Arc<Expr> = Expr::int(1);
            for (base, exp) in &node.factors {
                factors_expr =
                    normalize::mul(factors_expr, normalize::pow(base.clone(), exp.clone()));
            }
            let factor_limit = evaluate_limit_at_infinity_arc(&factors_expr, var)?;
            let coeff_expr = Arc::new(Expr::Rational(node.coeff.clone()));
            Some(crate::numeric::simplify::simplify(&normalize::mul(
                coeff_expr,
                factor_limit,
            )))
        }

        // Add node: limit of sum = sum of limits (if all exist)
        Expr::Add(node) => {
            let mut acc: Arc<Expr> = if node.constant.is_zero() {
                Expr::int(0)
            } else {
                Arc::new(Expr::Rational(node.constant.clone()))
            };
            for (term, coeff) in &node.terms {
                // Build scaled term
                let coeff_expr = Arc::new(Expr::Rational(coeff.clone()));
                let scaled = normalize::mul(coeff_expr, term.clone());
                let term_limit = evaluate_limit_at_infinity_arc(&scaled, var)?;
                acc = normalize::add(acc, term_limit);
            }
            Some(crate::numeric::simplify::simplify(&acc))
        }

        // Pow nodes with base as function (already covered above in Pow arm)
        // e^(-x) → 0 as x → ∞
        Expr::Func(FuncId::Exp, args) if args.len() == 1 => {
            // Check if the argument is negated (mul with negative coeff)
            let is_negated = match args[0].as_ref() {
                Expr::Mul(node) => node.coeff.is_negative(),
                _ => false,
            };
            if is_negated {
                Some(Expr::int(0))
            } else {
                None
            }
        }

        _ => None,
    }
}

// =============================================================================
// Public API — Expression-typed boundary
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
// TODO(arc-migration): callers (ffi/calculus_impl.rs, integration tests) still Expression-native — drop compile/decompile wrappers when callers migrated
pub fn definite_integral(
    expr: &Expression,
    var: &str,
    lower: &Expression,
    upper: &Expression,
) -> IntegrationResult {
    let var_id = SymbolId::intern(var);
    let expr_arc = compile(expr);
    let lower_arc = compile(lower);
    let upper_arc = compile(upper);

    let result_arc =
        definite_integral_arc(&expr_arc, var_id, &lower_arc, &upper_arc).map_err(|e| e)?;

    Ok(decompile(&result_arc))
}

/// Arc<Expr>-native definite integral implementation.
fn definite_integral_arc(
    expr: &Arc<Expr>,
    var: SymbolId,
    lower: &Arc<Expr>,
    upper: &Arc<Expr>,
) -> Result<Arc<Expr>, IntegrationError> {
    // Check for special cases first
    if let Some(result) = check_definite_special_cases_arc(expr, var, lower, upper) {
        return Ok(result);
    }

    // Find the antiderivative using the unified integrator
    let antiderivative = integrate_arc(expr, var)?;

    // Evaluate F(upper) - F(lower)
    let f_upper = subst_var(&antiderivative, var, upper);
    let f_lower = subst_var(&antiderivative, var, lower);

    let result = crate::numeric::simplify::simplify(&normalize::sub(f_upper, f_lower));
    Ok(result)
}

/// Integrate `expr` with respect to `var`, returning an `Arc<Expr>` antiderivative.
/// Delegates to `pattern_integrate`.
fn integrate_arc(expr: &Arc<Expr>, var: SymbolId) -> Result<Arc<Expr>, IntegrationError> {
    use crate::numeric::risch::IntegrationResult as NumericIntegrationResult;
    match crate::numeric::pattern_integrate::pattern_integrate(expr, var) {
        NumericIntegrationResult::Elementary(anti) => Ok(anti),
        NumericIntegrationResult::NonElementary(_) => Err(IntegrationError::CannotIntegrate(
            "Integrand has no elementary antiderivative".to_string(),
        )),
        NumericIntegrationResult::Partial { .. } => Err(IntegrationError::CannotIntegrate(
            "No closed-form elementary antiderivative found".to_string(),
        )),
    }
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
// TODO(arc-migration): callers (integration tests) still Expression-native — drop compile wrapper when migrated
pub fn definite_integral_with_fallback(
    expr: &Expression,
    var: &str,
    lower: f64,
    upper: f64,
    tolerance: f64,
) -> Result<f64, IntegrationError> {
    let var_id = SymbolId::intern(var);
    let expr_arc = compile(expr);
    let lower_arc = Expr::float(lower);
    let upper_arc = Expr::float(upper);

    if let Ok(result_arc) = definite_integral_arc(&expr_arc, var_id, &lower_arc, &upper_arc) {
        let empty = std::collections::HashMap::new();
        if let Some(value) = evaluate(&result_arc, &empty) {
            return Ok(value);
        }
    }

    // Fall back to numerical integration (Simpson's rule)
    numerical_integrate_arc(&expr_arc, var_id, lower, upper, tolerance)
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
// TODO(arc-migration): callers (fourier.rs, integration tests) still Expression-native — drop compile wrapper when migrated
pub fn numerical_integrate(
    expr: &Expression,
    var: &str,
    lower: f64,
    upper: f64,
    tolerance: f64,
) -> Result<f64, IntegrationError> {
    let var_id = SymbolId::intern(var);
    let expr_arc = compile(expr);
    numerical_integrate_arc(&expr_arc, var_id, lower, upper, tolerance)
}

/// Arc<Expr>-native adaptive Simpson's rule for numerical integration.
fn numerical_integrate_arc(
    expr: &Arc<Expr>,
    var: SymbolId,
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
        let part = simpsons_rule_adaptive_arc(expr, var, a, b, tol_per, 0).ok_or_else(|| {
            IntegrationError::CannotIntegrate("Numerical integration failed".to_string())
        })?;
        total += part;
    }
    Ok(total)
}

/// Adaptive Simpson's rule — Arc<Expr> variant.
fn simpsons_rule_adaptive_arc(
    expr: &Arc<Expr>,
    var: SymbolId,
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

    let fa = evaluate_at_arc(expr, var, a)?;
    let fb = evaluate_at_arc(expr, var, b)?;
    let fm = evaluate_at_arc(expr, var, mid)?;
    let f1 = evaluate_at_arc(expr, var, (a + mid) / 2.0)?;
    let f2 = evaluate_at_arc(expr, var, (mid + b) / 2.0)?;

    let h = (b - a) / 6.0;
    let s1 = h * (fa + 4.0 * fm + fb);

    let h2 = h / 2.0;
    let s2 = h2 * (fa + 4.0 * f1 + fm) + h2 * (fm + 4.0 * f2 + fb);

    if (s2 - s1).abs() < 15.0 * tolerance {
        Some(s2 + (s2 - s1) / 15.0) // Richardson extrapolation
    } else {
        let left = simpsons_rule_adaptive_arc(expr, var, a, mid, tolerance / 2.0, depth + 1)?;
        let right = simpsons_rule_adaptive_arc(expr, var, mid, b, tolerance / 2.0, depth + 1)?;
        Some(left + right)
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
// TODO(arc-migration): callers (integration tests) still Expression-native — drop compile/decompile wrapper when migrated
pub fn improper_integral_to_infinity(
    expr: &Expression,
    var: &str,
    lower: &Expression,
) -> IntegrationResult {
    let var_id = SymbolId::intern(var);
    let expr_arc = compile(expr);
    let lower_arc = compile(lower);

    let result_arc = improper_integral_to_infinity_arc(&expr_arc, var_id, &lower_arc)?;
    Ok(decompile(&result_arc))
}

/// Arc<Expr>-native improper integral to infinity.
fn improper_integral_to_infinity_arc(
    expr: &Arc<Expr>,
    var: SymbolId,
    lower: &Arc<Expr>,
) -> Result<Arc<Expr>, IntegrationError> {
    let antiderivative = integrate_arc(expr, var)?;

    let f_lower = crate::numeric::simplify::simplify(&subst_var(&antiderivative, var, lower));

    if let Some(limit_at_inf) = evaluate_limit_at_infinity_arc(&antiderivative, var) {
        let result = crate::numeric::simplify::simplify(&normalize::sub(limit_at_inf, f_lower));
        return Ok(result);
    }

    Err(IntegrationError::CannotIntegrate(
        "Cannot evaluate improper integral (may be divergent)".to_string(),
    ))
}

/// Compute definite integral with step-by-step explanation.
///
/// Returns the result and a list of steps showing the computation.
// TODO(arc-migration): callers (integration tests) still Expression-native — drop compile/decompile wrapper when migrated
pub fn definite_integral_with_steps(
    expr: &Expression,
    var: &str,
    lower: &Expression,
    upper: &Expression,
) -> Result<(Expression, Vec<String>), IntegrationError> {
    let var_id = SymbolId::intern(var);
    let expr_arc = compile(expr);
    let lower_arc = compile(lower);
    let upper_arc = compile(upper);

    let mut steps = Vec::new();
    steps.push(format!(
        "∫_{{{0}}}^{{{1}}} {2} d{3}",
        lower, upper, expr, var
    ));

    // Check for special cases
    if let Some(result_arc) =
        check_definite_special_cases_arc(&expr_arc, var_id, &lower_arc, &upper_arc)
    {
        let result = decompile(&result_arc);
        steps.push(format!("By special case analysis: {}", result));
        return Ok((result, steps));
    }

    // Find antiderivative
    steps.push("Step 1: Find the antiderivative F(x)".to_string());
    let antiderivative_arc = integrate_arc(&expr_arc, var_id)?;
    let antiderivative = decompile(&antiderivative_arc);
    steps.push(format!("F({}) = {}", var, antiderivative));

    // Apply Fundamental Theorem of Calculus
    steps.push("Step 2: Apply Fundamental Theorem of Calculus".to_string());
    steps.push("∫_a^b f(x) dx = F(b) - F(a)".to_string());

    // Evaluate at bounds
    let f_upper_arc =
        crate::numeric::simplify::simplify(&subst_var(&antiderivative_arc, var_id, &upper_arc));
    let f_lower_arc =
        crate::numeric::simplify::simplify(&subst_var(&antiderivative_arc, var_id, &lower_arc));
    let f_upper = decompile(&f_upper_arc);
    let f_lower = decompile(&f_lower_arc);
    steps.push(format!("F({}) = {}", upper, f_upper));
    steps.push(format!("F({}) = {}", lower, f_lower));

    // Compute result
    let result_arc = crate::numeric::simplify::simplify(&normalize::sub(f_upper_arc, f_lower_arc));
    let result = decompile(&result_arc);
    steps.push(format!("Result: {} - {} = {}", f_upper, f_lower, result));

    Ok((result, steps))
}
