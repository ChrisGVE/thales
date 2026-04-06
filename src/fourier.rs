//! Fourier series expansion module.
//!
//! This module provides Fourier series computation for mathematical expressions.
//! A real Fourier series represents a periodic function as:
//!
//! ```text
//! f(x) ~ a_0/2 + Σ_{n=1}^{N} [a_n*cos(2πnx/L) + b_n*sin(2πnx/L)]
//! ```
//!
//! where L is the period and the Fourier coefficients are:
//!
//! ```text
//! a_n = (2/L) * ∫_{-L/2}^{L/2} f(x)*cos(2πnx/L) dx
//! b_n = (2/L) * ∫_{-L/2}^{L/2} f(x)*sin(2πnx/L) dx
//! ```
//!
//! Coefficients are computed via numerical integration using adaptive Simpson's rule.
//!
//! # Examples
//!
//! ```rust
//! use thales::fourier::{fourier_series, FourierSeries};
//! use thales::ast::Variable;
//! use thales::parser::parse_expression;
//!
//! // Fourier series of cos(x) with period 2π should recover cos(x)
//! let expr = parse_expression("cos(x)").unwrap();
//! let x = Variable::new("x");
//! let series = fourier_series(&expr, &x, 3, None).unwrap();
//! // a_1 ≈ 1, all other coefficients ≈ 0
//! assert!((series.a_coefficients[1] - 1.0).abs() < 1e-6);
//! ```

use crate::ast::{Expression, Function, Variable};
use crate::integration::numerical_integrate;
use std::f64::consts::PI;
use std::fmt;

/// Errors that can occur during Fourier series computation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum FourierSeriesError {
    /// Integration failed when computing a coefficient.
    IntegrationFailed(String),
    /// The requested number of terms is zero.
    InvalidNumTerms,
    /// The period must be strictly positive.
    InvalidPeriod,
    /// Expression evaluation failed.
    EvaluationFailed(String),
}

impl fmt::Display for FourierSeriesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FourierSeriesError::IntegrationFailed(msg) => {
                write!(
                    f,
                    "Integration failed during coefficient computation: {msg}"
                )
            }
            FourierSeriesError::InvalidNumTerms => {
                write!(f, "Number of Fourier terms must be at least 1")
            }
            FourierSeriesError::InvalidPeriod => {
                write!(f, "Period must be strictly positive")
            }
            FourierSeriesError::EvaluationFailed(msg) => {
                write!(f, "Expression evaluation failed: {msg}")
            }
        }
    }
}

impl std::error::Error for FourierSeriesError {}

/// Result type for Fourier series operations.
pub type FourierSeriesResult<T> = Result<T, FourierSeriesError>;

/// A real Fourier series: a_0/2 + Σ [a_n*cos(2πnx/L) + b_n*sin(2πnx/L)].
///
/// Coefficients are indexed from 0: `a_coefficients[n]` is a_n,
/// `b_coefficients[n]` is b_n (b_0 is always 0).
#[derive(Debug, Clone)]
pub struct FourierSeries {
    /// Cosine coefficients: a_n for n = 0, 1, ..., num_terms.
    pub a_coefficients: Vec<f64>,
    /// Sine coefficients: b_n for n = 1, ..., num_terms (index 0 is unused, set to 0.0).
    pub b_coefficients: Vec<f64>,
    /// Period of the series.
    pub period: f64,
    /// Variable of expansion.
    pub variable: Variable,
    /// Number of harmonic terms (N), not counting the constant a_0/2.
    pub num_terms: usize,
}

impl FourierSeries {
    /// Numerically evaluate the truncated Fourier series at a point `x`.
    pub fn evaluate(&self, x: f64) -> f64 {
        let l = self.period;
        let mut sum = self.a_coefficients[0] / 2.0;
        for n in 1..=self.num_terms {
            let angle = 2.0 * PI * (n as f64) * x / l;
            sum += self.a_coefficients[n] * angle.cos();
            sum += self.b_coefficients[n] * angle.sin();
        }
        sum
    }

    /// Convert the series to a human-readable string representation.
    pub fn to_display_string(&self) -> String {
        let var = &self.variable.name;
        let l = self.period;
        let mut parts: Vec<String> = Vec::new();

        let a0 = self.a_coefficients[0];
        if a0.abs() > 1e-10 {
            parts.push(format!("{:.6}/2", a0));
        }

        for n in 1..=self.num_terms {
            let an = self.a_coefficients[n];
            let bn = self.b_coefficients[n];
            let arg = if (l - 2.0 * PI).abs() < 1e-10 {
                if n == 1 {
                    var.clone()
                } else {
                    format!("{n}{var}")
                }
            } else {
                format!("2π·{n}·{var}/{l:.6}")
            };
            if an.abs() > 1e-10 {
                parts.push(format!("{:.6}·cos({arg})", an));
            }
            if bn.abs() > 1e-10 {
                parts.push(format!("{:.6}·sin({arg})", bn));
            }
        }

        if parts.is_empty() {
            "0".to_string()
        } else {
            parts.join(" + ")
        }
    }

    /// Generate a LaTeX representation of the series.
    pub fn to_latex(&self) -> String {
        let var = &self.variable.name;
        let l = self.period;
        let mut parts: Vec<String> = Vec::new();

        let a0 = self.a_coefficients[0];
        if a0.abs() > 1e-10 {
            parts.push(format!("\\frac{{{:.6}}}{{2}}", a0));
        }

        for n in 1..=self.num_terms {
            let an = self.a_coefficients[n];
            let bn = self.b_coefficients[n];
            let arg = if (l - 2.0 * PI).abs() < 1e-10 {
                if n == 1 {
                    var.clone()
                } else {
                    format!("{n}{var}")
                }
            } else {
                format!("\\frac{{2\\pi \\cdot {n} \\cdot {var}}}{{{l:.6}}}")
            };
            if an.abs() > 1e-10 {
                parts.push(format!("{:.6}\\cos({arg})", an));
            }
            if bn.abs() > 1e-10 {
                parts.push(format!("{:.6}\\sin({arg})", bn));
            }
        }

        if parts.is_empty() {
            "0".to_string()
        } else {
            parts.join(" + ")
        }
    }
}

impl fmt::Display for FourierSeries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_display_string())
    }
}

/// Compute the Fourier series of `expr` with respect to `variable`.
///
/// Computes coefficients numerically using adaptive Simpson's rule integration.
///
/// # Arguments
///
/// * `expr` - The expression f(x) to expand
/// * `variable` - The variable of expansion
/// * `num_terms` - Number of harmonic terms N (produces terms n=1..=N)
/// * `period` - Optional period L (defaults to 2π if `None`)
///
/// # Returns
///
/// A [`FourierSeries`] with numerically computed coefficients, or an error.
///
/// # Examples
///
/// ```rust
/// use thales::fourier::fourier_series;
/// use thales::ast::Variable;
/// use thales::parser::parse_expression;
///
/// let expr = parse_expression("sin(x)").unwrap();
/// let x = Variable::new("x");
/// let series = fourier_series(&expr, &x, 3, None).unwrap();
/// // b_1 ≈ 1, all other coefficients ≈ 0
/// assert!((series.b_coefficients[1] - 1.0).abs() < 1e-6);
/// ```
pub fn fourier_series(
    expr: &Expression,
    variable: &Variable,
    num_terms: usize,
    period: Option<f64>,
) -> FourierSeriesResult<FourierSeries> {
    if num_terms == 0 {
        return Err(FourierSeriesError::InvalidNumTerms);
    }

    let l = period.unwrap_or(2.0 * PI);
    if l <= 0.0 {
        return Err(FourierSeriesError::InvalidPeriod);
    }

    let var_name = variable.name.as_str();
    let half_l = l / 2.0;
    let tolerance = 1e-8;

    // Compute a_0 = (2/L) * ∫_{-L/2}^{L/2} f(x) dx
    let a0 = compute_a0(expr, var_name, half_l, tolerance)?;

    let mut a_coefficients = Vec::with_capacity(num_terms + 1);
    let mut b_coefficients = Vec::with_capacity(num_terms + 1);
    a_coefficients.push(a0);
    b_coefficients.push(0.0); // b_0 is undefined/zero by convention

    for n in 1..=num_terms {
        let an = compute_an(expr, var_name, n, l, half_l, tolerance)?;
        let bn = compute_bn(expr, var_name, n, l, half_l, tolerance)?;
        a_coefficients.push(an);
        b_coefficients.push(bn);
    }

    Ok(FourierSeries {
        a_coefficients,
        b_coefficients,
        period: l,
        variable: variable.clone(),
        num_terms,
    })
}

/// Compute a_0 = (2/L) * ∫_{-L/2}^{L/2} f(x) dx.
fn compute_a0(
    expr: &Expression,
    var_name: &str,
    half_l: f64,
    tolerance: f64,
) -> FourierSeriesResult<f64> {
    let integral = numerical_integrate(expr, var_name, -half_l, half_l, tolerance)
        .map_err(|e| FourierSeriesError::IntegrationFailed(e.to_string()))?;
    Ok(2.0 / (2.0 * half_l) * integral)
}

/// Compute a_n = (2/L) * ∫_{-L/2}^{L/2} f(x)*cos(2πnx/L) dx.
fn compute_an(
    expr: &Expression,
    var_name: &str,
    n: usize,
    l: f64,
    half_l: f64,
    tolerance: f64,
) -> FourierSeriesResult<f64> {
    let cos_factor = build_cos_factor(var_name, n, l);
    let integrand = Expression::Binary(
        crate::ast::BinaryOp::Mul,
        Box::new(expr.clone()),
        Box::new(cos_factor),
    );
    let integral = numerical_integrate(&integrand, var_name, -half_l, half_l, tolerance)
        .map_err(|e| FourierSeriesError::IntegrationFailed(e.to_string()))?;
    Ok(2.0 / l * integral)
}

/// Compute b_n = (2/L) * ∫_{-L/2}^{L/2} f(x)*sin(2πnx/L) dx.
fn compute_bn(
    expr: &Expression,
    var_name: &str,
    n: usize,
    l: f64,
    half_l: f64,
    tolerance: f64,
) -> FourierSeriesResult<f64> {
    let sin_factor = build_sin_factor(var_name, n, l);
    let integrand = Expression::Binary(
        crate::ast::BinaryOp::Mul,
        Box::new(expr.clone()),
        Box::new(sin_factor),
    );
    let integral = numerical_integrate(&integrand, var_name, -half_l, half_l, tolerance)
        .map_err(|e| FourierSeriesError::IntegrationFailed(e.to_string()))?;
    Ok(2.0 / l * integral)
}

/// Build the expression cos(2πnx/L).
fn build_cos_factor(var_name: &str, n: usize, l: f64) -> Expression {
    let arg = build_harmonic_arg(var_name, n, l);
    Expression::Function(Function::Cos, vec![arg])
}

/// Build the expression sin(2πnx/L).
fn build_sin_factor(var_name: &str, n: usize, l: f64) -> Expression {
    let arg = build_harmonic_arg(var_name, n, l);
    Expression::Function(Function::Sin, vec![arg])
}

/// Build the argument 2πnx/L as an Expression.
fn build_harmonic_arg(var_name: &str, n: usize, l: f64) -> Expression {
    let x = Expression::Variable(Variable::new(var_name));
    let coeff = 2.0 * PI * (n as f64) / l;
    Expression::Binary(
        crate::ast::BinaryOp::Mul,
        Box::new(Expression::Float(coeff)),
        Box::new(x),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Variable;
    use crate::parser::parse_expression;

    const COEFF_TOL: f64 = 1e-4;

    #[test]
    fn test_cos_x_fourier_series() {
        // cos(x) on [-π, π] with period 2π should give a_1 = 1, all others ≈ 0
        let expr = parse_expression("cos(x)").unwrap();
        let x = Variable::new("x");
        let series = fourier_series(&expr, &x, 5, None).unwrap();

        assert_eq!(series.num_terms, 5);
        assert!(
            (series.a_coefficients[0]).abs() < COEFF_TOL,
            "a_0 should be 0"
        );
        assert!(
            (series.a_coefficients[1] - 1.0).abs() < COEFF_TOL,
            "a_1 should be 1, got {}",
            series.a_coefficients[1]
        );
        for n in 2..=5 {
            assert!(
                series.a_coefficients[n].abs() < COEFF_TOL,
                "a_{n} should be 0, got {}",
                series.a_coefficients[n]
            );
            assert!(
                series.b_coefficients[n].abs() < COEFF_TOL,
                "b_{n} should be 0, got {}",
                series.b_coefficients[n]
            );
        }
        assert!(
            (series.b_coefficients[1]).abs() < COEFF_TOL,
            "b_1 should be 0"
        );
    }

    #[test]
    fn test_sin_x_fourier_series() {
        // sin(x) on [-π, π] with period 2π should give b_1 = 1, all others ≈ 0
        let expr = parse_expression("sin(x)").unwrap();
        let x = Variable::new("x");
        let series = fourier_series(&expr, &x, 5, None).unwrap();

        assert!(
            (series.a_coefficients[0]).abs() < COEFF_TOL,
            "a_0 should be 0"
        );
        assert!(
            (series.b_coefficients[1] - 1.0).abs() < COEFF_TOL,
            "b_1 should be 1, got {}",
            series.b_coefficients[1]
        );
        assert!(
            (series.a_coefficients[1]).abs() < COEFF_TOL,
            "a_1 should be 0"
        );
        for n in 2..=5 {
            assert!(
                series.a_coefficients[n].abs() < COEFF_TOL,
                "a_{n} should be 0"
            );
            assert!(
                series.b_coefficients[n].abs() < COEFF_TOL,
                "b_{n} should be 0"
            );
        }
    }

    #[test]
    fn test_constant_fourier_series() {
        // f(x) = 1 on [-π, π] should give a_0 = 2, all others ≈ 0
        let expr = parse_expression("1").unwrap();
        let x = Variable::new("x");
        let series = fourier_series(&expr, &x, 3, None).unwrap();

        assert!(
            (series.a_coefficients[0] - 2.0).abs() < COEFF_TOL,
            "a_0 should be 2 (so a_0/2 = 1), got {}",
            series.a_coefficients[0]
        );
        for n in 1..=3 {
            assert!(
                series.a_coefficients[n].abs() < COEFF_TOL,
                "a_{n} should be 0"
            );
            assert!(
                series.b_coefficients[n].abs() < COEFF_TOL,
                "b_{n} should be 0"
            );
        }
    }

    #[test]
    fn test_cos2x_fourier_series() {
        // cos(2x) on [-π, π] should give a_2 = 1, all others ≈ 0
        let expr = parse_expression("cos(2 * x)").unwrap();
        let x = Variable::new("x");

        let series = fourier_series(&expr, &x, 5, None).unwrap();

        assert!(
            (series.a_coefficients[2] - 1.0).abs() < COEFF_TOL,
            "a_2 should be 1, got {}",
            series.a_coefficients[2]
        );
        assert!(
            (series.a_coefficients[1]).abs() < COEFF_TOL,
            "a_1 should be 0"
        );
        assert!(
            (series.a_coefficients[0]).abs() < COEFF_TOL,
            "a_0 should be 0"
        );
    }

    #[test]
    fn test_evaluate_recovers_cos_x() {
        // The series evaluation at several points should match cos(x)
        let expr = parse_expression("cos(x)").unwrap();
        let x = Variable::new("x");
        let series = fourier_series(&expr, &x, 3, None).unwrap();

        for &pt in &[0.0_f64, 0.5, 1.0, 1.5, -1.0] {
            let approx = series.evaluate(pt);
            let exact = pt.cos();
            assert!(
                (approx - exact).abs() < 1e-3,
                "At x={pt}, series={approx}, exact={exact}"
            );
        }
    }

    #[test]
    fn test_custom_period() {
        // f(x) = cos(π x) on period L=2 ([-1, 1]) should give a_1 = 1
        let expr = parse_expression("cos(3.14159265358979 * x)").unwrap();
        let x = Variable::new("x");
        let series = fourier_series(&expr, &x, 3, Some(2.0)).unwrap();

        assert_eq!(series.period, 2.0);
        assert!(
            (series.a_coefficients[1] - 1.0).abs() < 1e-3,
            "a_1 should be ~1, got {}",
            series.a_coefficients[1]
        );
    }

    #[test]
    fn test_zero_num_terms_error() {
        let expr = parse_expression("sin(x)").unwrap();
        let x = Variable::new("x");
        let result = fourier_series(&expr, &x, 0, None);
        assert!(matches!(result, Err(FourierSeriesError::InvalidNumTerms)));
    }

    #[test]
    fn test_invalid_period_error() {
        let expr = parse_expression("sin(x)").unwrap();
        let x = Variable::new("x");
        let result = fourier_series(&expr, &x, 3, Some(-1.0));
        assert!(matches!(result, Err(FourierSeriesError::InvalidPeriod)));
    }

    #[test]
    fn test_to_latex_produces_string() {
        let expr = parse_expression("cos(x)").unwrap();
        let x = Variable::new("x");
        let series = fourier_series(&expr, &x, 2, None).unwrap();
        let latex = series.to_latex();
        assert!(!latex.is_empty());
        assert!(latex.contains("cos"));
    }

    #[test]
    fn test_display_produces_string() {
        let expr = parse_expression("sin(x)").unwrap();
        let x = Variable::new("x");
        let series = fourier_series(&expr, &x, 2, None).unwrap();
        let s = format!("{series}");
        assert!(!s.is_empty());
    }
}
