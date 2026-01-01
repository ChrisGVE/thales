//! Partial fraction decomposition for integrating rational functions.
//!
//! This module provides functionality for decomposing rational functions (ratios of polynomials)
//! into simpler fractions that can be easily integrated.
//!
//! # Algorithm
//!
//! 1. Factor the denominator into linear and irreducible quadratic factors
//! 2. Set up partial fraction form:
//!    - Linear factor (x-a): A/(x-a)
//!    - Repeated linear (x-a)^n: A₁/(x-a) + A₂/(x-a)² + ... + Aₙ/(x-a)^n
//!    - Irreducible quadratic (x²+px+q): (Ax+B)/(x²+px+q)
//! 3. Solve for coefficients using cover-up method or system of equations
//!
//! # Example
//!
//! ```
//! use thales::partial_fractions::{decompose, is_rational_function};
//! use thales::ast::{Expression, Variable, BinaryOp};
//!
//! // Check if 1/(x²-1) is a rational function
//! let x = Expression::Variable(Variable::new("x"));
//! let num = Expression::Integer(1);
//! let denom = Expression::Binary(
//!     BinaryOp::Sub,
//!     Box::new(Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(2)))),
//!     Box::new(Expression::Integer(1))
//! );
//! let expr = Expression::Binary(BinaryOp::Div, Box::new(num), Box::new(denom));
//!
//! assert!(is_rational_function(&expr, "x"));
//! ```

use crate::ast::{BinaryOp, Expression, Function, SymbolicConstant, UnaryOp, Variable};
use std::collections::HashMap;
use std::fmt;

/// Error types for partial fraction decomposition.
#[derive(Debug, Clone, PartialEq)]
pub enum DecomposeError {
    /// The expression is not a rational function.
    NotRational(String),
    /// Cannot factor the denominator.
    CannotFactor(String),
    /// Cannot solve for coefficients.
    CoefficientError(String),
    /// The denominator has degree less than numerator (need polynomial division first).
    ImproperFraction(String),
    /// Division by zero would occur.
    DivisionByZero,
}

impl fmt::Display for DecomposeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecomposeError::NotRational(msg) => write!(f, "Not a rational function: {}", msg),
            DecomposeError::CannotFactor(msg) => write!(f, "Cannot factor denominator: {}", msg),
            DecomposeError::CoefficientError(msg) => {
                write!(f, "Cannot solve for coefficients: {}", msg)
            }
            DecomposeError::ImproperFraction(msg) => write!(f, "Improper fraction: {}", msg),
            DecomposeError::DivisionByZero => write!(f, "Division by zero"),
        }
    }
}

impl std::error::Error for DecomposeError {}

/// A term in a partial fraction decomposition.
#[derive(Debug, Clone, PartialEq)]
pub enum PartialFractionTerm {
    /// A/(x-a)^n - linear factor term
    Linear {
        /// Coefficient A
        coefficient: f64,
        /// Root a (where x-a is the factor)
        root: f64,
        /// Power n
        power: u32,
    },
    /// (Ax+B)/(x²+px+q)^n - irreducible quadratic term
    Quadratic {
        /// Coefficient A (for x term)
        a_coeff: f64,
        /// Coefficient B (constant term)
        b_coeff: f64,
        /// p coefficient in x²+px+q
        p: f64,
        /// q coefficient in x²+px+q
        q: f64,
        /// Power n
        power: u32,
    },
    /// A polynomial term (when numerator degree >= denominator degree)
    Polynomial(Expression),
}

/// Result of partial fraction decomposition.
#[derive(Debug, Clone)]
pub struct PartialFractionResult {
    /// The decomposed terms.
    pub terms: Vec<PartialFractionTerm>,
    /// The variable of decomposition.
    pub variable: String,
    /// Steps taken during decomposition (for resolution path).
    pub steps: Vec<String>,
}

/// Check if an expression is a rational function of the given variable.
///
/// A rational function is a ratio of two polynomials.
///
/// # Arguments
///
/// * `expr` - The expression to check
/// * `var` - The variable name
///
/// # Returns
///
/// `true` if the expression is a rational function of the variable.
///
/// # Example
///
/// ```
/// use thales::partial_fractions::is_rational_function;
/// use thales::ast::{Expression, Variable, BinaryOp};
///
/// let x = Expression::Variable(Variable::new("x"));
/// let rational = Expression::Binary(
///     BinaryOp::Div,
///     Box::new(x.clone()),
///     Box::new(Expression::Binary(
///         BinaryOp::Add,
///         Box::new(x.clone()),
///         Box::new(Expression::Integer(1))
///     ))
/// );
/// assert!(is_rational_function(&rational, "x"));
/// ```
pub fn is_rational_function(expr: &Expression, var: &str) -> bool {
    match expr {
        // Division is the main case we check
        Expression::Binary(BinaryOp::Div, num, denom) => {
            is_polynomial(num, var) && is_polynomial(denom, var)
        }
        // A polynomial by itself is a rational function (denominator = 1)
        _ => is_polynomial(expr, var),
    }
}

/// Check if an expression is a polynomial in the given variable.
pub fn is_polynomial(expr: &Expression, var: &str) -> bool {
    match expr {
        Expression::Integer(_) | Expression::Float(_) | Expression::Rational(_) => true,
        Expression::Variable(_) => true, // Any variable is a polynomial
        Expression::Constant(_) => true,
        Expression::Unary(UnaryOp::Neg, inner) => is_polynomial(inner, var),
        Expression::Binary(op, left, right) => match op {
            BinaryOp::Add | BinaryOp::Sub => is_polynomial(left, var) && is_polynomial(right, var),
            BinaryOp::Mul => is_polynomial(left, var) && is_polynomial(right, var),
            BinaryOp::Div => {
                // Division is only polynomial if denominator doesn't contain the variable
                is_polynomial(left, var) && !contains_variable(right, var)
            }
            _ => false,
        },
        Expression::Power(base, exp) => {
            if let Expression::Integer(n) = exp.as_ref() {
                *n >= 0 && is_polynomial(base, var)
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Check if an expression contains the given variable.
fn contains_variable(expr: &Expression, var: &str) -> bool {
    match expr {
        Expression::Variable(v) => v.name == var,
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Rational(_)
        | Expression::Constant(_) => false,
        Expression::Unary(_, inner) => contains_variable(inner, var),
        Expression::Binary(_, left, right) => {
            contains_variable(left, var) || contains_variable(right, var)
        }
        Expression::Power(base, exp) => {
            contains_variable(base, var) || contains_variable(exp, var)
        }
        Expression::Function(_, args) => args.iter().any(|arg| contains_variable(arg, var)),
        _ => false,
    }
}

/// Get the polynomial degree of an expression.
pub fn get_polynomial_degree(expr: &Expression, var: &str) -> Option<i32> {
    match expr {
        Expression::Integer(_) | Expression::Float(_) | Expression::Rational(_) => Some(0),
        Expression::Constant(_) => Some(0),
        Expression::Variable(v) => {
            if v.name == var {
                Some(1)
            } else {
                Some(0)
            }
        }
        Expression::Unary(UnaryOp::Neg, inner) => get_polynomial_degree(inner, var),
        Expression::Binary(op, left, right) => match op {
            BinaryOp::Add | BinaryOp::Sub => {
                let ld = get_polynomial_degree(left, var)?;
                let rd = get_polynomial_degree(right, var)?;
                Some(ld.max(rd))
            }
            BinaryOp::Mul => {
                let ld = get_polynomial_degree(left, var)?;
                let rd = get_polynomial_degree(right, var)?;
                Some(ld + rd)
            }
            BinaryOp::Div => {
                if !contains_variable(right, var) {
                    get_polynomial_degree(left, var)
                } else {
                    None
                }
            }
            _ => None,
        },
        Expression::Power(base, exp) => {
            if let Expression::Integer(n) = exp.as_ref() {
                if *n >= 0 {
                    let base_deg = get_polynomial_degree(base, var)?;
                    Some(base_deg * (*n as i32))
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Extract polynomial coefficients from an expression.
///
/// Returns a map from power to coefficient, e.g., for 3x² + 2x + 1:
/// {0: 1.0, 1: 2.0, 2: 3.0}
pub fn extract_coefficients(expr: &Expression, var: &str) -> Option<HashMap<i32, f64>> {
    let mut coeffs = HashMap::new();
    if extract_coefficients_impl(expr, var, 1.0, &mut coeffs) {
        Some(coeffs)
    } else {
        None
    }
}

fn extract_coefficients_impl(
    expr: &Expression,
    var: &str,
    multiplier: f64,
    coeffs: &mut HashMap<i32, f64>,
) -> bool {
    match expr {
        Expression::Integer(n) => {
            *coeffs.entry(0).or_insert(0.0) += (*n as f64) * multiplier;
            true
        }
        Expression::Float(f) => {
            *coeffs.entry(0).or_insert(0.0) += f * multiplier;
            true
        }
        Expression::Rational(r) => {
            let val = *r.numer() as f64 / *r.denom() as f64;
            *coeffs.entry(0).or_insert(0.0) += val * multiplier;
            true
        }
        Expression::Constant(c) => {
            let val = match c {
                SymbolicConstant::Pi => std::f64::consts::PI,
                SymbolicConstant::E => std::f64::consts::E,
                SymbolicConstant::I => return false, // Can't handle imaginary
            };
            *coeffs.entry(0).or_insert(0.0) += val * multiplier;
            true
        }
        Expression::Variable(v) => {
            if v.name == var {
                *coeffs.entry(1).or_insert(0.0) += multiplier;
            } else {
                // Treat other variables as constants
                // For now, this is not fully supported
                *coeffs.entry(0).or_insert(0.0) += multiplier;
            }
            true
        }
        Expression::Unary(UnaryOp::Neg, inner) => {
            extract_coefficients_impl(inner, var, -multiplier, coeffs)
        }
        Expression::Binary(op, left, right) => match op {
            BinaryOp::Add => {
                extract_coefficients_impl(left, var, multiplier, coeffs)
                    && extract_coefficients_impl(right, var, multiplier, coeffs)
            }
            BinaryOp::Sub => {
                extract_coefficients_impl(left, var, multiplier, coeffs)
                    && extract_coefficients_impl(right, var, -multiplier, coeffs)
            }
            BinaryOp::Mul => {
                // Try to handle coefficient * variable
                if let Some(val) = evaluate_constant(left) {
                    extract_coefficients_impl(right, var, multiplier * val, coeffs)
                } else if let Some(val) = evaluate_constant(right) {
                    extract_coefficients_impl(left, var, multiplier * val, coeffs)
                } else if !contains_variable(left, var) {
                    if let Some(val) = evaluate_constant(left) {
                        extract_coefficients_impl(right, var, multiplier * val, coeffs)
                    } else {
                        false
                    }
                } else if !contains_variable(right, var) {
                    if let Some(val) = evaluate_constant(right) {
                        extract_coefficients_impl(left, var, multiplier * val, coeffs)
                    } else {
                        false
                    }
                } else {
                    // Both sides contain variable - need to expand
                    // For now, handle simple cases like x * x
                    if let (Expression::Variable(v1), Expression::Variable(v2)) =
                        (left.as_ref(), right.as_ref())
                    {
                        if v1.name == var && v2.name == var {
                            *coeffs.entry(2).or_insert(0.0) += multiplier;
                            return true;
                        }
                    }
                    false
                }
            }
            BinaryOp::Div => {
                if !contains_variable(right, var) {
                    if let Some(val) = evaluate_constant(right) {
                        if val.abs() < 1e-15 {
                            return false;
                        }
                        extract_coefficients_impl(left, var, multiplier / val, coeffs)
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            _ => false,
        },
        Expression::Power(base, exp) => {
            if let Expression::Integer(n) = exp.as_ref() {
                if *n >= 0 {
                    if let Expression::Variable(v) = base.as_ref() {
                        if v.name == var {
                            *coeffs.entry(*n as i32).or_insert(0.0) += multiplier;
                            return true;
                        }
                    }
                }
            }
            false
        }
        _ => false,
    }
}

/// Evaluate a constant expression to a float.
fn evaluate_constant(expr: &Expression) -> Option<f64> {
    match expr {
        Expression::Integer(n) => Some(*n as f64),
        Expression::Float(f) => Some(*f),
        Expression::Rational(r) => Some(*r.numer() as f64 / *r.denom() as f64),
        Expression::Constant(c) => match c {
            SymbolicConstant::Pi => Some(std::f64::consts::PI),
            SymbolicConstant::E => Some(std::f64::consts::E),
            SymbolicConstant::I => None,
        },
        Expression::Unary(UnaryOp::Neg, inner) => evaluate_constant(inner).map(|v| -v),
        Expression::Binary(op, left, right) => {
            let lv = evaluate_constant(left)?;
            let rv = evaluate_constant(right)?;
            match op {
                BinaryOp::Add => Some(lv + rv),
                BinaryOp::Sub => Some(lv - rv),
                BinaryOp::Mul => Some(lv * rv),
                BinaryOp::Div => {
                    if rv.abs() < 1e-15 {
                        None
                    } else {
                        Some(lv / rv)
                    }
                }
                _ => None,
            }
        }
        Expression::Power(base, exp) => {
            let bv = evaluate_constant(base)?;
            let ev = evaluate_constant(exp)?;
            Some(bv.powf(ev))
        }
        _ => None,
    }
}

/// Find the real roots of a polynomial given its coefficients.
///
/// # Arguments
///
/// * `coeffs` - Map from power to coefficient
///
/// # Returns
///
/// A vector of (root, multiplicity) pairs for real roots.
fn find_polynomial_roots(coeffs: &HashMap<i32, f64>) -> Vec<(f64, u32)> {
    let max_degree = coeffs.keys().copied().max().unwrap_or(0);

    if max_degree == 0 {
        return vec![];
    }

    if max_degree == 1 {
        // Linear: ax + b = 0 => x = -b/a
        let a = *coeffs.get(&1).unwrap_or(&0.0);
        let b = *coeffs.get(&0).unwrap_or(&0.0);
        if a.abs() < 1e-15 {
            return vec![];
        }
        return vec![(-b / a, 1)];
    }

    if max_degree == 2 {
        // Quadratic: ax² + bx + c = 0
        let a = *coeffs.get(&2).unwrap_or(&0.0);
        let b = *coeffs.get(&1).unwrap_or(&0.0);
        let c = *coeffs.get(&0).unwrap_or(&0.0);

        if a.abs() < 1e-15 {
            // Actually linear
            if b.abs() < 1e-15 {
                return vec![];
            }
            return vec![(-c / b, 1)];
        }

        let discriminant = b * b - 4.0 * a * c;
        if discriminant < -1e-15 {
            // No real roots (complex roots)
            return vec![];
        } else if discriminant.abs() < 1e-15 {
            // One repeated root
            let root = -b / (2.0 * a);
            return vec![(root, 2)];
        } else {
            // Two distinct roots
            let sqrt_disc = discriminant.sqrt();
            let r1 = (-b + sqrt_disc) / (2.0 * a);
            let r2 = (-b - sqrt_disc) / (2.0 * a);
            return vec![(r1, 1), (r2, 1)];
        }
    }

    // For higher degrees, use numerical methods
    // Try common roots first
    let mut roots = vec![];

    // Check integer roots from -10 to 10
    for i in -10..=10 {
        let x = i as f64;
        if evaluate_polynomial(coeffs, x).abs() < 1e-10 {
            roots.push((x, 1));
        }
    }

    // Try to find more roots using Newton-Raphson
    for start in &[-5.0, -1.0, 0.0, 1.0, 5.0] {
        if let Some(root) = newton_raphson(coeffs, *start) {
            // Check if we already have this root
            let already_found = roots.iter().any(|(r, _)| (r - root).abs() < 1e-10);
            if !already_found {
                roots.push((root, 1));
            }
        }
    }

    // Determine multiplicities by factoring out each root and checking again
    // This is a simplified approach
    roots
}

/// Evaluate a polynomial at a given point.
fn evaluate_polynomial(coeffs: &HashMap<i32, f64>, x: f64) -> f64 {
    coeffs.iter().map(|(pow, coeff)| coeff * x.powi(*pow)).sum()
}

/// Evaluate the derivative of a polynomial at a given point.
fn evaluate_polynomial_derivative(coeffs: &HashMap<i32, f64>, x: f64) -> f64 {
    coeffs
        .iter()
        .filter(|(pow, _)| **pow > 0)
        .map(|(pow, coeff)| (*pow as f64) * coeff * x.powi(*pow - 1))
        .sum()
}

/// Newton-Raphson method to find a root.
fn newton_raphson(coeffs: &HashMap<i32, f64>, start: f64) -> Option<f64> {
    let mut x = start;
    for _ in 0..100 {
        let f = evaluate_polynomial(coeffs, x);
        let df = evaluate_polynomial_derivative(coeffs, x);
        if df.abs() < 1e-15 {
            return None;
        }
        let new_x = x - f / df;
        if (new_x - x).abs() < 1e-12 {
            if evaluate_polynomial(coeffs, new_x).abs() < 1e-10 {
                return Some(new_x);
            } else {
                return None;
            }
        }
        x = new_x;
    }
    None
}

/// Check if a quadratic is irreducible (no real roots).
fn is_irreducible_quadratic(p: f64, q: f64) -> bool {
    // x² + px + q has discriminant p² - 4q
    let discriminant = p * p - 4.0 * q;
    discriminant < 0.0
}

/// Decompose a rational function into partial fractions.
///
/// # Arguments
///
/// * `numerator` - The numerator polynomial
/// * `denominator` - The denominator polynomial
/// * `var` - The variable to decompose with respect to
///
/// # Returns
///
/// A `PartialFractionResult` containing the decomposed terms.
///
/// # Example
///
/// ```
/// use thales::partial_fractions::decompose;
/// use thales::ast::{Expression, Variable, BinaryOp};
///
/// // Decompose 1/(x²-1) = 1/((x-1)(x+1))
/// let x = Expression::Variable(Variable::new("x"));
/// let num = Expression::Integer(1);
/// let denom = Expression::Binary(
///     BinaryOp::Sub,
///     Box::new(Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(2)))),
///     Box::new(Expression::Integer(1))
/// );
///
/// let result = decompose(&num, &denom, &Variable::new("x")).unwrap();
/// // Result: 1/(2(x-1)) - 1/(2(x+1))
/// assert_eq!(result.terms.len(), 2);
/// ```
pub fn decompose(
    numerator: &Expression,
    denominator: &Expression,
    var: &Variable,
) -> Result<PartialFractionResult, DecomposeError> {
    let var_name = &var.name;
    let mut steps = Vec::new();

    // Check that both are polynomials
    if !is_polynomial(numerator, var_name) {
        return Err(DecomposeError::NotRational(
            "Numerator is not a polynomial".to_string(),
        ));
    }
    if !is_polynomial(denominator, var_name) {
        return Err(DecomposeError::NotRational(
            "Denominator is not a polynomial".to_string(),
        ));
    }

    steps.push("Verified expression is a rational function".to_string());

    // Get degrees
    let num_degree = get_polynomial_degree(numerator, var_name).unwrap_or(0);
    let denom_degree = get_polynomial_degree(denominator, var_name).unwrap_or(0);

    // Extract coefficients
    let num_coeffs = extract_coefficients(numerator, var_name)
        .ok_or_else(|| DecomposeError::NotRational("Cannot extract numerator coefficients".to_string()))?;
    let denom_coeffs = extract_coefficients(denominator, var_name)
        .ok_or_else(|| DecomposeError::NotRational("Cannot extract denominator coefficients".to_string()))?;

    steps.push(format!(
        "Numerator degree: {}, Denominator degree: {}",
        num_degree, denom_degree
    ));

    // Handle improper fractions (numerator degree >= denominator degree)
    let mut terms = Vec::new();
    let working_num_coeffs;
    if num_degree >= denom_degree {
        steps.push("Improper fraction: performing polynomial division".to_string());
        let (quotient, remainder) = polynomial_division(&num_coeffs, &denom_coeffs);

        // Add polynomial term
        if !quotient.is_empty() {
            let poly_expr = coefficients_to_expression(&quotient, var_name);
            terms.push(PartialFractionTerm::Polynomial(poly_expr));
            steps.push(format!("Polynomial quotient extracted"));
        }

        working_num_coeffs = remainder;
    } else {
        working_num_coeffs = num_coeffs.clone();
    }

    // Find roots of denominator
    let roots = find_polynomial_roots(&denom_coeffs);
    steps.push(format!("Found {} real roots in denominator", roots.len()));

    if roots.is_empty() && denom_degree > 0 {
        // Denominator might be irreducible quadratic or higher
        if denom_degree == 2 {
            let a = *denom_coeffs.get(&2).unwrap_or(&1.0);
            let b = *denom_coeffs.get(&1).unwrap_or(&0.0);
            let c = *denom_coeffs.get(&0).unwrap_or(&0.0);

            // Normalize: x² + (b/a)x + (c/a)
            let p = b / a;
            let q = c / a;

            if is_irreducible_quadratic(p, q) {
                steps.push("Denominator is an irreducible quadratic".to_string());

                // For Ax+B over irreducible quadratic, solve for A and B
                // From the numerator coefficients
                let num_a = *working_num_coeffs.get(&1).unwrap_or(&0.0) / a;
                let num_b = *working_num_coeffs.get(&0).unwrap_or(&0.0) / a;

                terms.push(PartialFractionTerm::Quadratic {
                    a_coeff: num_a,
                    b_coeff: num_b,
                    p,
                    q,
                    power: 1,
                });

                return Ok(PartialFractionResult {
                    terms,
                    variable: var_name.clone(),
                    steps,
                });
            }
        }

        return Err(DecomposeError::CannotFactor(
            "Cannot factor denominator into linear/quadratic factors".to_string(),
        ));
    }

    // Use cover-up method for simple linear factors
    for (root, multiplicity) in &roots {
        steps.push(format!(
            "Processing root {} with multiplicity {}",
            root, multiplicity
        ));

        for power in 1..=*multiplicity {
            // Cover-up method: substitute x = root into numerator / (remaining factors)
            let coeff = compute_coefficient_cover_up(
                &working_num_coeffs,
                &denom_coeffs,
                *root,
                power,
                &roots,
            );

            terms.push(PartialFractionTerm::Linear {
                coefficient: coeff,
                root: *root,
                power,
            });

            steps.push(format!(
                "Coefficient for 1/(x-{})^{}: {}",
                root, power, coeff
            ));
        }
    }

    Ok(PartialFractionResult {
        terms,
        variable: var_name.clone(),
        steps,
    })
}

/// Perform polynomial long division.
fn polynomial_division(
    num: &HashMap<i32, f64>,
    denom: &HashMap<i32, f64>,
) -> (HashMap<i32, f64>, HashMap<i32, f64>) {
    let mut quotient = HashMap::new();
    let mut remainder = num.clone();

    let denom_degree = denom.keys().copied().max().unwrap_or(0);
    let denom_leading = *denom.get(&denom_degree).unwrap_or(&1.0);

    loop {
        let rem_degree = remainder.keys().copied().max().unwrap_or(-1);
        if rem_degree < denom_degree || rem_degree < 0 {
            break;
        }

        let rem_leading = *remainder.get(&rem_degree).unwrap_or(&0.0);
        let factor = rem_leading / denom_leading;
        let power_diff = rem_degree - denom_degree;

        *quotient.entry(power_diff).or_insert(0.0) += factor;

        // Subtract factor * denom from remainder
        for (pow, coeff) in denom.iter() {
            let new_pow = pow + power_diff;
            *remainder.entry(new_pow).or_insert(0.0) -= factor * coeff;
        }

        // Clean up near-zero coefficients
        remainder.retain(|_, v| v.abs() > 1e-12);
    }

    (quotient, remainder)
}

/// Convert coefficient map back to expression.
fn coefficients_to_expression(coeffs: &HashMap<i32, f64>, var: &str) -> Expression {
    if coeffs.is_empty() {
        return Expression::Integer(0);
    }

    let mut terms: Vec<Expression> = Vec::new();

    let mut powers: Vec<i32> = coeffs.keys().copied().collect();
    powers.sort_by(|a, b| b.cmp(a)); // Descending order

    for pow in powers {
        let coeff = *coeffs.get(&pow).unwrap_or(&0.0);
        if coeff.abs() < 1e-12 {
            continue;
        }

        let term = if pow == 0 {
            float_to_expression(coeff)
        } else if pow == 1 {
            if (coeff - 1.0).abs() < 1e-12 {
                Expression::Variable(Variable::new(var))
            } else if (coeff + 1.0).abs() < 1e-12 {
                Expression::Unary(
                    UnaryOp::Neg,
                    Box::new(Expression::Variable(Variable::new(var))),
                )
            } else {
                Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(float_to_expression(coeff)),
                    Box::new(Expression::Variable(Variable::new(var))),
                )
            }
        } else {
            let var_power = Expression::Power(
                Box::new(Expression::Variable(Variable::new(var))),
                Box::new(Expression::Integer(pow as i64)),
            );
            if (coeff - 1.0).abs() < 1e-12 {
                var_power
            } else if (coeff + 1.0).abs() < 1e-12 {
                Expression::Unary(UnaryOp::Neg, Box::new(var_power))
            } else {
                Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(float_to_expression(coeff)),
                    Box::new(var_power),
                )
            }
        };

        terms.push(term);
    }

    if terms.is_empty() {
        return Expression::Integer(0);
    }

    let mut result = terms.remove(0);
    for term in terms {
        // Check if term is negative
        if let Expression::Unary(UnaryOp::Neg, inner) = &term {
            result = Expression::Binary(BinaryOp::Sub, Box::new(result), inner.clone());
        } else {
            result = Expression::Binary(BinaryOp::Add, Box::new(result), Box::new(term));
        }
    }

    result
}

/// Convert a float to an expression.
fn float_to_expression(f: f64) -> Expression {
    if f < 0.0 {
        Expression::Unary(UnaryOp::Neg, Box::new(float_to_expression(-f)))
    } else if (f.round() - f).abs() < 1e-12 {
        Expression::Integer(f.round() as i64)
    } else {
        Expression::Float(f)
    }
}

/// Compute coefficient using cover-up method.
fn compute_coefficient_cover_up(
    num_coeffs: &HashMap<i32, f64>,
    denom_coeffs: &HashMap<i32, f64>,
    root: f64,
    power: u32,
    all_roots: &[(f64, u32)],
) -> f64 {
    // For simple case (power = 1), use direct cover-up:
    // A = f(root) where f(x) = num(x) / (denom(x) / (x-root))

    let num_val = evaluate_polynomial(num_coeffs, root);

    // Compute denom / (x - root)^power
    // By evaluating the derivative if needed
    if power == 1 {
        // Simple cover-up
        let mut denom_without_root = 1.0;
        for (r, mult) in all_roots {
            if (*r - root).abs() > 1e-12 {
                denom_without_root *= (root - r).powi(*mult as i32);
            }
        }

        // Also account for leading coefficient
        let denom_degree = denom_coeffs.keys().copied().max().unwrap_or(0);
        let leading = *denom_coeffs.get(&denom_degree).unwrap_or(&1.0);
        denom_without_root *= leading;

        if denom_without_root.abs() < 1e-15 {
            0.0
        } else {
            num_val / denom_without_root
        }
    } else {
        // For repeated roots, need to use differentiation approach
        // This is more complex; for now, use numerical approximation
        let h = 1e-6;
        let coeff = (evaluate_polynomial(num_coeffs, root + h)
            / evaluate_polynomial_without_root(denom_coeffs, root + h, root, power))
            .abs();
        coeff
    }
}

/// Evaluate polynomial with (x-root)^power factored out.
fn evaluate_polynomial_without_root(
    coeffs: &HashMap<i32, f64>,
    x: f64,
    root: f64,
    power: u32,
) -> f64 {
    let full = evaluate_polynomial(coeffs, x);
    let factor = (x - root).powi(power as i32);
    if factor.abs() < 1e-15 {
        // Use L'Hôpital-like approach
        1.0
    } else {
        full / factor
    }
}

/// Convert a partial fraction term to an expression.
impl PartialFractionTerm {
    /// Convert this term to an Expression.
    pub fn to_expression(&self, var: &str) -> Expression {
        match self {
            PartialFractionTerm::Linear {
                coefficient,
                root,
                power,
            } => {
                let x_minus_a = if *root >= 0.0 {
                    Expression::Binary(
                        BinaryOp::Sub,
                        Box::new(Expression::Variable(Variable::new(var))),
                        Box::new(float_to_expression(*root)),
                    )
                } else {
                    Expression::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::Variable(Variable::new(var))),
                        Box::new(float_to_expression(-*root)),
                    )
                };

                let denom = if *power == 1 {
                    x_minus_a
                } else {
                    Expression::Power(Box::new(x_minus_a), Box::new(Expression::Integer(*power as i64)))
                };

                Expression::Binary(
                    BinaryOp::Div,
                    Box::new(float_to_expression(*coefficient)),
                    Box::new(denom),
                )
            }
            PartialFractionTerm::Quadratic {
                a_coeff,
                b_coeff,
                p,
                q,
                power,
            } => {
                // (Ax + B) / (x² + px + q)^n
                let x = Expression::Variable(Variable::new(var));

                // Numerator: Ax + B
                let ax = Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(float_to_expression(*a_coeff)),
                    Box::new(x.clone()),
                );
                let numerator = Expression::Binary(
                    BinaryOp::Add,
                    Box::new(ax),
                    Box::new(float_to_expression(*b_coeff)),
                );

                // Denominator: x² + px + q
                let x_squared =
                    Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(2)));
                let px = Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(float_to_expression(*p)),
                    Box::new(x.clone()),
                );
                let quad = Expression::Binary(
                    BinaryOp::Add,
                    Box::new(Expression::Binary(
                        BinaryOp::Add,
                        Box::new(x_squared),
                        Box::new(px),
                    )),
                    Box::new(float_to_expression(*q)),
                );

                let denom = if *power == 1 {
                    quad
                } else {
                    Expression::Power(Box::new(quad), Box::new(Expression::Integer(*power as i64)))
                };

                Expression::Binary(BinaryOp::Div, Box::new(numerator), Box::new(denom))
            }
            PartialFractionTerm::Polynomial(expr) => expr.clone(),
        }
    }

    /// Integrate this partial fraction term.
    pub fn integrate(&self, var: &str) -> Expression {
        match self {
            PartialFractionTerm::Linear {
                coefficient,
                root,
                power,
            } => {
                if *power == 1 {
                    // ∫ A/(x-a) dx = A * ln|x-a|
                    let x_minus_a = if *root >= 0.0 {
                        Expression::Binary(
                            BinaryOp::Sub,
                            Box::new(Expression::Variable(Variable::new(var))),
                            Box::new(float_to_expression(*root)),
                        )
                    } else {
                        Expression::Binary(
                            BinaryOp::Add,
                            Box::new(Expression::Variable(Variable::new(var))),
                            Box::new(float_to_expression(-*root)),
                        )
                    };

                    let ln_term = Expression::Function(
                        Function::Ln,
                        vec![Expression::Function(Function::Abs, vec![x_minus_a])],
                    );

                    if (*coefficient - 1.0).abs() < 1e-12 {
                        ln_term
                    } else {
                        Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(float_to_expression(*coefficient)),
                            Box::new(ln_term),
                        )
                    }
                } else {
                    // ∫ A/(x-a)^n dx = -A/((n-1)(x-a)^(n-1)) for n > 1
                    let x_minus_a = if *root >= 0.0 {
                        Expression::Binary(
                            BinaryOp::Sub,
                            Box::new(Expression::Variable(Variable::new(var))),
                            Box::new(float_to_expression(*root)),
                        )
                    } else {
                        Expression::Binary(
                            BinaryOp::Add,
                            Box::new(Expression::Variable(Variable::new(var))),
                            Box::new(float_to_expression(-*root)),
                        )
                    };

                    let new_coeff = -*coefficient / ((*power - 1) as f64);
                    let new_power = *power - 1;

                    let denom = if new_power == 1 {
                        x_minus_a
                    } else {
                        Expression::Power(
                            Box::new(x_minus_a),
                            Box::new(Expression::Integer(new_power as i64)),
                        )
                    };

                    Expression::Binary(
                        BinaryOp::Div,
                        Box::new(float_to_expression(new_coeff)),
                        Box::new(denom),
                    )
                }
            }
            PartialFractionTerm::Quadratic {
                a_coeff,
                b_coeff,
                p,
                q,
                power,
            } => {
                if *power == 1 {
                    // Split (Ax+B)/(x²+px+q) into A/2 * 2x/(x²+px+q) + (B - Ap/2) * 1/(x²+px+q)
                    // First part: A/2 * ln|x²+px+q|
                    // Second part: (B - Ap/2) * arctan form

                    let x = Expression::Variable(Variable::new(var));

                    // First part: (A/2) * ln|x² + px + q|
                    let x_squared =
                        Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(2)));
                    let px = Expression::Binary(
                        BinaryOp::Mul,
                        Box::new(float_to_expression(*p)),
                        Box::new(x.clone()),
                    );
                    let quad = Expression::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::Binary(
                            BinaryOp::Add,
                            Box::new(x_squared),
                            Box::new(px),
                        )),
                        Box::new(float_to_expression(*q)),
                    );

                    let ln_part = Expression::Binary(
                        BinaryOp::Mul,
                        Box::new(float_to_expression(*a_coeff / 2.0)),
                        Box::new(Expression::Function(
                            Function::Ln,
                            vec![Expression::Function(Function::Abs, vec![quad])],
                        )),
                    );

                    // Second part: arctan
                    // Complete the square: x² + px + q = (x + p/2)² + (q - p²/4)
                    let h = p / 2.0;
                    let k_squared = q - p * p / 4.0;

                    if k_squared > 0.0 {
                        let k = k_squared.sqrt();
                        let c = b_coeff - a_coeff * h;

                        // c/k * arctan((x + h)/k)
                        let x_plus_h = Expression::Binary(
                            BinaryOp::Add,
                            Box::new(x.clone()),
                            Box::new(float_to_expression(h)),
                        );
                        let arg = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(x_plus_h),
                            Box::new(float_to_expression(k)),
                        );
                        let arctan_part = Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(float_to_expression(c / k)),
                            Box::new(Expression::Function(Function::Atan, vec![arg])),
                        );

                        Expression::Binary(BinaryOp::Add, Box::new(ln_part), Box::new(arctan_part))
                    } else {
                        // This shouldn't happen for irreducible quadratic
                        ln_part
                    }
                } else {
                    // Higher powers are more complex; return the term as-is for now
                    self.to_expression(var)
                }
            }
            PartialFractionTerm::Polynomial(expr) => {
                // Integrate the polynomial term by term
                // This is a simplified implementation
                crate::integration::integrate(expr, var).unwrap_or_else(|_| expr.clone())
            }
        }
    }
}

impl PartialFractionResult {
    /// Convert the decomposition back to an expression (sum of terms).
    pub fn to_expression(&self) -> Expression {
        if self.terms.is_empty() {
            return Expression::Integer(0);
        }

        let mut result = self.terms[0].to_expression(&self.variable);
        for term in self.terms.iter().skip(1) {
            result = Expression::Binary(
                BinaryOp::Add,
                Box::new(result),
                Box::new(term.to_expression(&self.variable)),
            );
        }
        result
    }

    /// Integrate all terms and return the combined result.
    pub fn integrate(&self) -> Expression {
        if self.terms.is_empty() {
            return Expression::Integer(0);
        }

        let mut result = self.terms[0].integrate(&self.variable);
        for term in self.terms.iter().skip(1) {
            result = Expression::Binary(
                BinaryOp::Add,
                Box::new(result),
                Box::new(term.integrate(&self.variable)),
            );
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_polynomial() {
        let x = Expression::Variable(Variable::new("x"));
        assert!(is_polynomial(&x, "x"));
        assert!(is_polynomial(&Expression::Integer(5), "x"));

        let x_squared = Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(2)));
        assert!(is_polynomial(&x_squared, "x"));

        // x + 1
        let x_plus_1 = Expression::Binary(
            BinaryOp::Add,
            Box::new(x.clone()),
            Box::new(Expression::Integer(1)),
        );
        assert!(is_polynomial(&x_plus_1, "x"));

        // sin(x) is not a polynomial
        let sin_x = Expression::Function(Function::Sin, vec![x.clone()]);
        assert!(!is_polynomial(&sin_x, "x"));
    }

    #[test]
    fn test_is_rational_function() {
        let x = Expression::Variable(Variable::new("x"));

        // x / (x + 1) is rational
        let rational = Expression::Binary(
            BinaryOp::Div,
            Box::new(x.clone()),
            Box::new(Expression::Binary(
                BinaryOp::Add,
                Box::new(x.clone()),
                Box::new(Expression::Integer(1)),
            )),
        );
        assert!(is_rational_function(&rational, "x"));

        // sin(x) / x is not rational
        let not_rational = Expression::Binary(
            BinaryOp::Div,
            Box::new(Expression::Function(Function::Sin, vec![x.clone()])),
            Box::new(x.clone()),
        );
        assert!(!is_rational_function(&not_rational, "x"));
    }

    #[test]
    fn test_polynomial_degree() {
        let x = Expression::Variable(Variable::new("x"));

        assert_eq!(get_polynomial_degree(&x, "x"), Some(1));
        assert_eq!(get_polynomial_degree(&Expression::Integer(5), "x"), Some(0));

        let x_cubed = Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(3)));
        assert_eq!(get_polynomial_degree(&x_cubed, "x"), Some(3));

        let poly = Expression::Binary(
            BinaryOp::Add,
            Box::new(x_cubed),
            Box::new(x.clone()),
        );
        assert_eq!(get_polynomial_degree(&poly, "x"), Some(3));
    }

    #[test]
    fn test_extract_coefficients() {
        let x = Expression::Variable(Variable::new("x"));

        // x² + 2x + 1
        let x_squared = Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(2)));
        let two_x = Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Integer(2)),
            Box::new(x.clone()),
        );
        let poly = Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Binary(
                BinaryOp::Add,
                Box::new(x_squared),
                Box::new(two_x),
            )),
            Box::new(Expression::Integer(1)),
        );

        let coeffs = extract_coefficients(&poly, "x").unwrap();
        assert!((coeffs.get(&0).unwrap_or(&0.0) - 1.0).abs() < 1e-10);
        assert!((coeffs.get(&1).unwrap_or(&0.0) - 2.0).abs() < 1e-10);
        assert!((coeffs.get(&2).unwrap_or(&0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_find_polynomial_roots_linear() {
        let mut coeffs = HashMap::new();
        coeffs.insert(0, -2.0); // -2
        coeffs.insert(1, 1.0); // +x

        let roots = find_polynomial_roots(&coeffs);
        assert_eq!(roots.len(), 1);
        assert!((roots[0].0 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_find_polynomial_roots_quadratic() {
        // x² - 1 = (x-1)(x+1)
        let mut coeffs = HashMap::new();
        coeffs.insert(0, -1.0);
        coeffs.insert(2, 1.0);

        let roots = find_polynomial_roots(&coeffs);
        assert_eq!(roots.len(), 2);

        let root_values: Vec<f64> = roots.iter().map(|(r, _)| *r).collect();
        assert!(root_values.iter().any(|r| (r - 1.0).abs() < 1e-10));
        assert!(root_values.iter().any(|r| (r + 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_decompose_simple() {
        // 1/(x² - 1) = 1/((x-1)(x+1)) = 1/(2(x-1)) - 1/(2(x+1))
        let x = Expression::Variable(Variable::new("x"));
        let num = Expression::Integer(1);
        let denom = Expression::Binary(
            BinaryOp::Sub,
            Box::new(Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(2)))),
            Box::new(Expression::Integer(1)),
        );

        let result = decompose(&num, &denom, &Variable::new("x")).unwrap();
        assert_eq!(result.terms.len(), 2);

        // Check that we have linear terms
        for term in &result.terms {
            match term {
                PartialFractionTerm::Linear { coefficient, root, power } => {
                    assert_eq!(*power, 1);
                    // Coefficients should be ±1/2
                    assert!((coefficient.abs() - 0.5).abs() < 1e-10);
                }
                _ => panic!("Expected linear terms"),
            }
        }
    }

    #[test]
    fn test_decompose_x_times_x_minus_1() {
        // 1/(x² - x) = 1/(x(x-1)) = -1/x + 1/(x-1)
        // Use expanded form: x² - x
        let x = Expression::Variable(Variable::new("x"));
        let num = Expression::Integer(1);
        let x_squared = Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(2)));
        let denom = Expression::Binary(
            BinaryOp::Sub,
            Box::new(x_squared),
            Box::new(x.clone()),
        );

        let result = decompose(&num, &denom, &Variable::new("x")).unwrap();
        assert_eq!(result.terms.len(), 2);
    }

    #[test]
    fn test_linear_term_integration() {
        // ∫ 1/(x-2) dx = ln|x-2|
        let term = PartialFractionTerm::Linear {
            coefficient: 1.0,
            root: 2.0,
            power: 1,
        };

        let integral = term.integrate("x");
        // Should contain ln
        assert!(format!("{:?}", integral).contains("Ln"));
    }

    #[test]
    fn test_linear_term_higher_power_integration() {
        // ∫ 1/(x-1)² dx = -1/(x-1)
        let term = PartialFractionTerm::Linear {
            coefficient: 1.0,
            root: 1.0,
            power: 2,
        };

        let integral = term.integrate("x");
        // Should be a fraction with power 1
        assert!(format!("{:?}", integral).contains("Div"));
    }

    #[test]
    fn test_irreducible_quadratic() {
        // x² + 1 has discriminant -4 < 0
        assert!(is_irreducible_quadratic(0.0, 1.0));

        // x² - 1 has discriminant 4 > 0
        assert!(!is_irreducible_quadratic(0.0, -1.0));
    }
}
