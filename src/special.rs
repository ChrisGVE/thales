//! Special mathematical functions with step-by-step derivation.
//!
//! This module provides special mathematical functions including Gamma, Beta,
//! and error functions. Each computation includes detailed derivation steps
//! for educational purposes.
//!
//! # Functions
//!
//! - **Gamma function**: `gamma(x)` - Generalization of factorial to real/complex numbers
//! - **Beta function**: `beta(a, b)` - Related to Gamma via B(a,b) = Γ(a)Γ(b)/Γ(a+b)
//! - **Error function**: `erf(x)` - Probability function
//! - **Complementary error function**: `erfc(x) = 1 - erf(x)`
//!
//! # Examples
//!
//! ```rust
//! use thales::special::gamma;
//! use thales::ast::Expression;
//!
//! // Gamma of positive integer: Γ(5) = 4! = 24
//! let result = gamma(&Expression::Integer(5)).unwrap();
//! assert_eq!(result.numeric_value, Some(24.0));
//! assert!(!result.derivation_steps.is_empty());
//! ```

use crate::ast::{BinaryOp, Expression, Function, SymbolicConstant};
use std::f64::consts::{E, PI};
use std::fmt;

/// Error types for special function computations.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SpecialFunctionError {
    /// Invalid argument for the function (e.g., negative integer for Gamma).
    InvalidArgument(String),
    /// The computation method is not yet implemented for this argument type.
    NotImplemented(String),
    /// Computation failed due to numerical issues or other errors.
    ComputationFailed(String),
}

impl fmt::Display for SpecialFunctionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpecialFunctionError::InvalidArgument(msg) => {
                write!(f, "Invalid argument: {}", msg)
            }
            SpecialFunctionError::NotImplemented(msg) => {
                write!(f, "Not implemented: {}", msg)
            }
            SpecialFunctionError::ComputationFailed(msg) => {
                write!(f, "Computation failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for SpecialFunctionError {}

/// Result of a special function computation with derivation steps.
///
/// Contains the symbolic result, optional numeric value, and human-readable
/// derivation steps explaining how the result was obtained.
#[derive(Debug, Clone, PartialEq)]
pub struct SpecialFunctionResult {
    /// The symbolic expression result.
    pub value: Expression,
    /// Numeric approximation if computable.
    pub numeric_value: Option<f64>,
    /// Human-readable steps showing the derivation.
    pub derivation_steps: Vec<String>,
}

impl SpecialFunctionResult {
    /// Create a new result with given value, numeric approximation, and steps.
    pub fn new(
        value: Expression,
        numeric_value: Option<f64>,
        derivation_steps: Vec<String>,
    ) -> Self {
        Self {
            value,
            numeric_value,
            derivation_steps,
        }
    }
}

/// Compute the Gamma function with derivation steps.
///
/// The Gamma function is defined as:
/// - For positive integers: Γ(n) = (n-1)!
/// - General definition: Γ(z) = ∫₀^∞ t^(z-1) e^(-t) dt
///
/// # Special Cases
///
/// - **Positive integers**: Uses factorial formula with recurrence relation
/// - **Half-integers**: Uses reflection formula and known values
/// - **Γ(1) = 1**
/// - **Γ(1/2) = √π**
///
/// # Examples
///
/// ```rust
/// use thales::special::gamma;
/// use thales::ast::Expression;
///
/// // Γ(5) = 4! = 24
/// let result = gamma(&Expression::Integer(5)).unwrap();
/// assert_eq!(result.numeric_value, Some(24.0));
///
/// // Γ(1) = 1
/// let result = gamma(&Expression::Integer(1)).unwrap();
/// assert_eq!(result.numeric_value, Some(1.0));
/// ```
///
/// # Errors
///
/// Returns `InvalidArgument` for:
/// - Zero or negative integers (Γ function has poles)
/// - Complex numbers (not yet implemented)
#[must_use = "computing special functions returns a result that should be used"]
pub fn gamma(x: &Expression) -> Result<SpecialFunctionResult, SpecialFunctionError> {
    let mut steps = Vec::new();
    steps.push(format!("Computing Gamma function: Γ({})", format_expr(x)));

    // Handle positive integers: Γ(n) = (n-1)!
    if let Expression::Integer(n) = x {
        if *n <= 0 {
            return Err(SpecialFunctionError::InvalidArgument(
                format!("Gamma function has a pole at non-positive integer {}", n),
            ));
        }

        steps.push(format!("For positive integer n={}, use Γ(n) = (n-1)!", n));
        steps.push(format!("Γ({}) = {}! = ({})", n, n - 1, n - 1));

        // Compute factorial using recurrence
        let mut factorial = 1i64;
        let mut computation_steps = Vec::new();
        for i in 2..=(*n - 1) {
            factorial *= i;
            computation_steps.push(format!("{}", i));
        }

        if *n > 1 {
            steps.push(format!(
                "Computing {}! = {} = {}",
                n - 1,
                computation_steps.join(" × "),
                factorial
            ));
        } else {
            steps.push(format!("Computing {}! = 1", n - 1));
        }

        let result_expr = Expression::Integer(factorial);
        return Ok(SpecialFunctionResult::new(
            result_expr,
            Some(factorial as f64),
            steps,
        ));
    }

    // Handle Γ(1/2) = √π
    if let Expression::Rational(r) = x {
        if r.numer() == &1 && r.denom() == &2 {
            steps.push("For Γ(1/2), use known value: Γ(1/2) = √π".to_string());
            steps.push(format!("√π ≈ {}", PI.sqrt()));

            let result_expr = Expression::Function(
                Function::Sqrt,
                vec![Expression::Constant(SymbolicConstant::Pi)],
            );
            return Ok(SpecialFunctionResult::new(
                result_expr,
                Some(PI.sqrt()),
                steps,
            ));
        }

        // Handle other half-integers: Γ(n + 1/2) for positive n
        // Using: Γ(z+1) = z·Γ(z), so Γ(n+1/2) = (n-1/2)·(n-3/2)···(1/2)·√π
        if r.denom() == &2 && r.numer() > &0 {
            let n = (r.numer() - 1) / 2;
            steps.push(format!(
                "For half-integer {}/{} = {} + 1/2, use recurrence relation",
                r.numer(),
                r.denom(),
                n
            ));
            steps.push("Γ(z+1) = z·Γ(z), so Γ(n+1/2) = (n-1/2)·Γ(n-1/2)".to_string());

            // Compute product: (n-1/2)·(n-3/2)···(1/2)
            let mut numeric_product = 1.0;
            let mut factors = Vec::new();
            for k in 0..=n {
                let factor = (n - k) as f64 + 0.5;
                if factor > 0.0 {
                    numeric_product *= factor;
                    factors.push(format!("{}", factor));
                }
            }

            if !factors.is_empty() {
                steps.push(format!(
                    "Γ({}) = {} × √π",
                    format_expr(x),
                    factors.join(" × ")
                ));
            }
            steps.push(format!(
                "Γ({}) = {} × √π ≈ {}",
                format_expr(x),
                numeric_product,
                numeric_product * PI.sqrt()
            ));

            // Build symbolic expression
            let sqrt_pi = Expression::Function(
                Function::Sqrt,
                vec![Expression::Constant(SymbolicConstant::Pi)],
            );
            let result_expr = Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Float(numeric_product)),
                Box::new(sqrt_pi),
            );

            return Ok(SpecialFunctionResult::new(
                result_expr,
                Some(numeric_product * PI.sqrt()),
                steps,
            ));
        }
    }

    // Handle Float values using Stirling's approximation or numerical methods
    if let Expression::Float(f) = x {
        if *f <= 0.0 && f.fract() == 0.0 {
            return Err(SpecialFunctionError::InvalidArgument(
                format!("Gamma function has a pole at non-positive integer {}", f),
            ));
        }

        steps.push(format!(
            "For real number x={}, use numerical approximation",
            f
        ));

        // Use Stirling's approximation for large values
        if *f > 10.0 {
            steps.push("Using Stirling's approximation: Γ(x) ≈ √(2π/x) × (x/e)^x".to_string());
            let stirling = ((2.0 * PI / f).sqrt()) * (f / E).powf(*f);
            steps.push(format!("Γ({}) ≈ {}", f, stirling));

            return Ok(SpecialFunctionResult::new(
                Expression::Float(stirling),
                Some(stirling),
                steps,
            ));
        }

        // For smaller values, use recurrence to reduce to known range
        steps.push("Using recurrence relation: Γ(x+1) = x·Γ(x)".to_string());
        let numeric = gamma_numeric(*f);
        steps.push(format!("Γ({}) ≈ {}", f, numeric));

        return Ok(SpecialFunctionResult::new(
            Expression::Float(numeric),
            Some(numeric),
            steps,
        ));
    }

    Err(SpecialFunctionError::NotImplemented(
        format!("Gamma function not implemented for expression type: {}", format_expr(x)),
    ))
}

/// Compute the Beta function with derivation steps.
///
/// The Beta function is defined as:
/// B(a, b) = Γ(a)·Γ(b) / Γ(a+b)
///
/// # Examples
///
/// ```rust
/// use thales::special::beta;
/// use thales::ast::Expression;
///
/// // B(1, 1) = 1
/// let result = beta(&Expression::Integer(1), &Expression::Integer(1)).unwrap();
/// assert_eq!(result.numeric_value, Some(1.0));
/// ```
#[must_use = "computing special functions returns a result that should be used"]
pub fn beta(
    a: &Expression,
    b: &Expression,
) -> Result<SpecialFunctionResult, SpecialFunctionError> {
    let mut steps = Vec::new();
    steps.push(format!(
        "Computing Beta function: B({}, {})",
        format_expr(a),
        format_expr(b)
    ));
    steps.push("Using definition: B(a,b) = Γ(a)·Γ(b) / Γ(a+b)".to_string());

    // Compute Γ(a)
    steps.push(format!("Step 1: Compute Γ({})", format_expr(a)));
    let gamma_a = gamma(a)?;
    for step in &gamma_a.derivation_steps {
        steps.push(format!("  {}", step));
    }

    // Compute Γ(b)
    steps.push(format!("Step 2: Compute Γ({})", format_expr(b)));
    let gamma_b = gamma(b)?;
    for step in &gamma_b.derivation_steps {
        steps.push(format!("  {}", step));
    }

    // Compute a + b
    let a_plus_b = Expression::Binary(BinaryOp::Add, Box::new(a.clone()), Box::new(b.clone()));
    let a_plus_b_simplified = a_plus_b.simplify();

    // Compute Γ(a + b)
    steps.push(format!("Step 3: Compute Γ({} + {}) = Γ({})",
        format_expr(a), format_expr(b), format_expr(&a_plus_b_simplified)));
    let gamma_a_plus_b = gamma(&a_plus_b_simplified)?;
    for step in &gamma_a_plus_b.derivation_steps {
        steps.push(format!("  {}", step));
    }

    // Compute B(a,b) = Γ(a)·Γ(b) / Γ(a+b)
    steps.push(format!(
        "Step 4: Compute B({}, {}) = Γ({})·Γ({}) / Γ({})",
        format_expr(a),
        format_expr(b),
        format_expr(a),
        format_expr(b),
        format_expr(&a_plus_b_simplified)
    ));

    let numerator = Expression::Binary(
        BinaryOp::Mul,
        Box::new(gamma_a.value.clone()),
        Box::new(gamma_b.value.clone()),
    );
    let result_expr = Expression::Binary(
        BinaryOp::Div,
        Box::new(numerator),
        Box::new(gamma_a_plus_b.value.clone()),
    );

    let numeric_value = if let (Some(ga), Some(gb), Some(gab)) = (
        gamma_a.numeric_value,
        gamma_b.numeric_value,
        gamma_a_plus_b.numeric_value,
    ) {
        let result = (ga * gb) / gab;
        steps.push(format!(
            "B({}, {}) = {} × {} / {} = {}",
            format_expr(a),
            format_expr(b),
            ga,
            gb,
            gab,
            result
        ));
        Some(result)
    } else {
        None
    };

    Ok(SpecialFunctionResult::new(
        result_expr.simplify(),
        numeric_value,
        steps,
    ))
}

/// Compute the error function with derivation steps.
///
/// The error function is defined as:
/// erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
///
/// Uses series expansion:
/// erf(x) = (2/√π) Σ_{n=0}^∞ [(-1)ⁿ x^(2n+1)] / [n!(2n+1)]
///
/// # Examples
///
/// ```rust
/// use thales::special::erf;
/// use thales::ast::Expression;
///
/// // erf(0) = 0
/// let result = erf(&Expression::Integer(0)).unwrap();
/// assert_eq!(result.numeric_value, Some(0.0));
/// ```
#[must_use = "computing special functions returns a result that should be used"]
pub fn erf(x: &Expression) -> Result<SpecialFunctionResult, SpecialFunctionError> {
    let mut steps = Vec::new();
    steps.push(format!("Computing error function: erf({})", format_expr(x)));
    steps.push("Definition: erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt".to_string());
    steps.push("Using series expansion: erf(x) = (2/√π) Σ [(-1)ⁿ x^(2n+1)] / [n!(2n+1)]".to_string());

    // Try to extract numeric value
    let x_val = match x {
        Expression::Integer(n) => Some(*n as f64),
        Expression::Float(f) => Some(*f),
        Expression::Rational(r) => Some(*r.numer() as f64 / *r.denom() as f64),
        _ => None,
    };

    if let Some(x_val) = x_val {
        // Special case: erf(0) = 0
        if x_val.abs() < 1e-15 {
            steps.push("erf(0) = 0 (integral from 0 to 0)".to_string());
            return Ok(SpecialFunctionResult::new(
                Expression::Integer(0),
                Some(0.0),
                steps,
            ));
        }

        // Use series expansion for small to moderate values
        steps.push(format!("For x = {}, computing series terms:", x_val));

        let mut sum = 0.0;
        let mut term = x_val;
        let mut n = 0;
        const MAX_TERMS: usize = 50;
        const EPSILON: f64 = 1e-15;

        while n < MAX_TERMS && term.abs() > EPSILON {
            sum += term;
            if n < 5 {
                // Show first few terms
                steps.push(format!(
                    "  n={}: (-1)^{} × {}^{} / ({}! × {}) = {}",
                    n,
                    n,
                    x_val,
                    2 * n + 1,
                    n,
                    2 * n + 1,
                    term
                ));
            } else if n == 5 {
                steps.push("  ... (continuing series expansion)".to_string());
            }

            n += 1;
            term = -term * x_val * x_val / (n as f64) * ((2 * n - 1) as f64 / (2 * n + 1) as f64);
        }

        let coefficient = 2.0 / PI.sqrt();
        let result = coefficient * sum;

        steps.push(format!("Series sum = {}", sum));
        steps.push(format!("erf({}) = (2/√π) × {} = {}", x_val, sum, result));

        return Ok(SpecialFunctionResult::new(
            Expression::Float(result),
            Some(result),
            steps,
        ));
    }

    Err(SpecialFunctionError::NotImplemented(
        format!("Error function not implemented for expression type: {}", format_expr(x)),
    ))
}

/// Compute the complementary error function with derivation steps.
///
/// The complementary error function is defined as:
/// erfc(x) = 1 - erf(x)
///
/// # Examples
///
/// ```rust
/// use thales::special::erfc;
/// use thales::ast::Expression;
///
/// // erfc(0) = 1
/// let result = erfc(&Expression::Integer(0)).unwrap();
/// assert_eq!(result.numeric_value, Some(1.0));
/// ```
#[must_use = "computing special functions returns a result that should be used"]
pub fn erfc(x: &Expression) -> Result<SpecialFunctionResult, SpecialFunctionError> {
    let mut steps = Vec::new();
    steps.push(format!(
        "Computing complementary error function: erfc({})",
        format_expr(x)
    ));
    steps.push("Definition: erfc(x) = 1 - erf(x)".to_string());

    // Compute erf(x)
    steps.push(format!("Step 1: Compute erf({})", format_expr(x)));
    let erf_result = erf(x)?;
    for step in &erf_result.derivation_steps {
        steps.push(format!("  {}", step));
    }

    // Compute 1 - erf(x)
    steps.push(format!(
        "Step 2: Compute erfc({}) = 1 - erf({})",
        format_expr(x),
        format_expr(x)
    ));

    let result_expr = Expression::Binary(
        BinaryOp::Sub,
        Box::new(Expression::Integer(1)),
        Box::new(erf_result.value.clone()),
    );

    let numeric_value = erf_result.numeric_value.map(|erf_val| {
        let result = 1.0 - erf_val;
        steps.push(format!("erfc({}) = 1 - {} = {}", format_expr(x), erf_val, result));
        result
    });

    Ok(SpecialFunctionResult::new(
        result_expr.simplify(),
        numeric_value,
        steps,
    ))
}

// ===== Helper Functions =====

/// Format an expression for display in derivation steps.
fn format_expr(expr: &Expression) -> String {
    match expr {
        Expression::Integer(n) => n.to_string(),
        Expression::Float(f) => f.to_string(),
        Expression::Rational(r) => format!("{}/{}", r.numer(), r.denom()),
        Expression::Variable(v) => v.name.clone(),
        Expression::Constant(c) => format!("{:?}", c),
        _ => format!("{:?}", expr),
    }
}

/// Numerical approximation of Gamma function using Lanczos approximation.
fn gamma_numeric(x: f64) -> f64 {
    // Lanczos coefficients for g=7
    const G: f64 = 7.0;
    const COEF: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        // Use reflection formula: Γ(1-x)Γ(x) = π/sin(πx)
        PI / ((PI * x).sin() * gamma_numeric(1.0 - x))
    } else {
        let x = x - 1.0;
        let mut a = COEF[0];
        for (i, &c) in COEF.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        let t = x + G + 0.5;
        ((2.0 * PI).sqrt()) * t.powf(x + 0.5) * (-t).exp() * a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_positive_integer() {
        // Γ(5) = 4! = 24
        let result = gamma(&Expression::Integer(5)).unwrap();
        assert_eq!(result.numeric_value, Some(24.0));
        assert!(!result.derivation_steps.is_empty());
        assert!(result
            .derivation_steps
            .iter()
            .any(|s| s.contains("4!")));
    }

    #[test]
    fn test_gamma_one() {
        // Γ(1) = 0! = 1
        let result = gamma(&Expression::Integer(1)).unwrap();
        assert_eq!(result.numeric_value, Some(1.0));
        assert!(!result.derivation_steps.is_empty());
    }

    #[test]
    fn test_gamma_half() {
        // Γ(1/2) = √π
        let result = gamma(&Expression::Rational(num_rational::Rational64::new(1, 2))).unwrap();
        assert!(result.numeric_value.is_some());
        let val = result.numeric_value.unwrap();
        assert!((val - PI.sqrt()).abs() < 1e-10);
        assert!(result.derivation_steps.iter().any(|s| s.contains("√π")));
    }

    #[test]
    fn test_gamma_negative_integer() {
        // Γ(0) and Γ(-1) should fail (poles)
        assert!(gamma(&Expression::Integer(0)).is_err());
        assert!(gamma(&Expression::Integer(-1)).is_err());
    }

    #[test]
    fn test_beta_one_one() {
        // B(1, 1) = Γ(1)Γ(1)/Γ(2) = 1×1/1 = 1
        let result = beta(&Expression::Integer(1), &Expression::Integer(1)).unwrap();
        assert_eq!(result.numeric_value, Some(1.0));
        assert!(!result.derivation_steps.is_empty());
    }

    #[test]
    fn test_erf_zero() {
        // erf(0) = 0
        let result = erf(&Expression::Integer(0)).unwrap();
        assert_eq!(result.numeric_value, Some(0.0));
        assert!(!result.derivation_steps.is_empty());
    }

    #[test]
    fn test_erfc_zero() {
        // erfc(0) = 1 - erf(0) = 1
        let result = erfc(&Expression::Integer(0)).unwrap();
        assert_eq!(result.numeric_value, Some(1.0));
        assert!(!result.derivation_steps.is_empty());
    }

    #[test]
    fn test_derivation_steps_non_empty() {
        // All functions should produce derivation steps
        let gamma_result = gamma(&Expression::Integer(5)).unwrap();
        assert!(!gamma_result.derivation_steps.is_empty());

        let beta_result = beta(&Expression::Integer(2), &Expression::Integer(3)).unwrap();
        assert!(!beta_result.derivation_steps.is_empty());

        let erf_result = erf(&Expression::Float(0.5)).unwrap();
        assert!(!erf_result.derivation_steps.is_empty());

        let erfc_result = erfc(&Expression::Float(0.5)).unwrap();
        assert!(!erfc_result.derivation_steps.is_empty());
    }

    #[test]
    fn test_gamma_numeric_approximation() {
        // Test Γ(5.5) ≈ 52.34
        let result = gamma(&Expression::Float(5.5)).unwrap();
        assert!(result.numeric_value.is_some());
        let val = result.numeric_value.unwrap();
        assert!((val - 52.34).abs() < 1.0); // Rough approximation
    }

    #[test]
    fn test_erf_one() {
        // erf(1) ≈ 0.8427
        let result = erf(&Expression::Float(1.0)).unwrap();
        assert!(result.numeric_value.is_some());
        let val = result.numeric_value.unwrap();
        assert!((val - 0.8427).abs() < 0.001);
    }
}
