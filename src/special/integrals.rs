//! Integral special functions: Riemann zeta, Si, Ci, Ei.
//!
//! Provides series-based numeric evaluations with derivation steps.
//! All functions accept `Expression` and return `SpecialFunctionResult`.

use crate::ast::{Expression, Function};
use crate::special::{SpecialFunctionError, SpecialFunctionResult};
use std::f64::consts::PI;

/// Euler-Mascheroni constant γ ≈ 0.5772156649
const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

/// Extract a finite f64 from a numeric Expression, or None for symbolic.
fn numeric_val(x: &Expression) -> Option<f64> {
    match x {
        Expression::Integer(n) => Some(*n as f64),
        Expression::Float(f) => Some(*f),
        Expression::Rational(r) => Some(*r.numer() as f64 / *r.denom() as f64),
        _ => None,
    }
}

/// Format an expression compactly for step text.
fn fmt_expr(x: &Expression) -> String {
    match x {
        Expression::Integer(n) => n.to_string(),
        Expression::Float(f) => format!("{}", f),
        Expression::Rational(r) => format!("{}/{}", r.numer(), r.denom()),
        Expression::Variable(v) => v.name.clone(),
        _ => format!("{:?}", x),
    }
}

/// Compute the Riemann zeta function ζ(s).
///
/// For real s > 1, uses direct summation:
/// ζ(s) = Σ_{n=1}^{N} 1/n^s  (N = 10 000)
///
/// Known exact values returned as float approximations:
/// - ζ(2) = π²/6 ≈ 1.6449
/// - ζ(4) = π⁴/90 ≈ 1.0823
/// - ζ(3) ≈ 1.2021 (Apéry's constant)
///
/// Returns `InvalidArgument` for s ≤ 1 (pole at s = 1).
#[must_use = "computing special functions returns a result that should be used"]
pub fn zeta(s: &Expression) -> Result<SpecialFunctionResult, SpecialFunctionError> {
    let mut steps = Vec::new();
    steps.push(format!("Computing Riemann zeta: ζ({})", fmt_expr(s)));
    steps.push("Definition: ζ(s) = Σ_{n=1}^∞ 1/n^s  (Re(s) > 1)".to_string());

    let s_val = match numeric_val(s) {
        Some(v) => v,
        None => {
            steps.push("Argument is symbolic — returning unevaluated ζ(s)".to_string());
            return Ok(SpecialFunctionResult::new(
                Expression::Function(Function::Zeta, vec![s.clone()]),
                None,
                steps,
            ));
        }
    };

    if s_val <= 1.0 {
        return Err(SpecialFunctionError::InvalidArgument(format!(
            "ζ(s) has a pole at s=1 and is not implemented for s ≤ 1 (got s={})",
            s_val
        )));
    }

    // Known exact values
    if (s_val - 2.0).abs() < 1e-12 {
        let val = PI * PI / 6.0;
        steps.push("Known exact: ζ(2) = π²/6".to_string());
        steps.push(format!("ζ(2) = π²/6 ≈ {}", val));
        return Ok(SpecialFunctionResult::new(
            Expression::Float(val),
            Some(val),
            steps,
        ));
    }
    if (s_val - 4.0).abs() < 1e-12 {
        let val = PI.powi(4) / 90.0;
        steps.push("Known exact: ζ(4) = π⁴/90".to_string());
        steps.push(format!("ζ(4) = π⁴/90 ≈ {}", val));
        return Ok(SpecialFunctionResult::new(
            Expression::Float(val),
            Some(val),
            steps,
        ));
    }
    if (s_val - 3.0).abs() < 1e-12 {
        let val = 1.202_056_903_159_594;
        steps.push("Known approximation: ζ(3) ≈ 1.2021 (Apéry's constant)".to_string());
        steps.push(format!("ζ(3) ≈ {}", val));
        return Ok(SpecialFunctionResult::new(
            Expression::Float(val),
            Some(val),
            steps,
        ));
    }

    steps.push(format!(
        "Using direct summation with N=10000 terms for s={}",
        s_val
    ));
    let sum = zeta_sum(s_val);
    steps.push(format!("ζ({}) ≈ {}", s_val, sum));
    Ok(SpecialFunctionResult::new(
        Expression::Float(sum),
        Some(sum),
        steps,
    ))
}

/// Direct summation ζ(s) = Σ_{n=1}^{N} 1/n^s with N=10000.
fn zeta_sum(s: f64) -> f64 {
    (1..=10_000u32).map(|n| (n as f64).powf(-s)).sum()
}

/// Compute the sine integral Si(x).
///
/// Si(x) = ∫_0^x sin(t)/t dt
///
/// Series: Si(x) = Σ_{k=0}^N (-1)^k · x^(2k+1) / ((2k+1) · (2k+1)!)
#[must_use = "computing special functions returns a result that should be used"]
pub fn si(x: &Expression) -> Result<SpecialFunctionResult, SpecialFunctionError> {
    let mut steps = Vec::new();
    steps.push(format!("Computing sine integral: Si({})", fmt_expr(x)));
    steps.push("Definition: Si(x) = ∫₀ˣ sin(t)/t dt".to_string());
    steps.push("Series: Si(x) = Σ_{k=0}^N (-1)^k · x^(2k+1) / ((2k+1)·(2k+1)!)".to_string());

    let x_val = match numeric_val(x) {
        Some(v) => v,
        None => {
            steps.push("Argument is symbolic — returning unevaluated Si(x)".to_string());
            return Ok(SpecialFunctionResult::new(
                Expression::Function(Function::Si, vec![x.clone()]),
                None,
                steps,
            ));
        }
    };

    let val = si_series(x_val, &mut steps);
    steps.push(format!("Si({}) ≈ {}", x_val, val));
    Ok(SpecialFunctionResult::new(
        Expression::Float(val),
        Some(val),
        steps,
    ))
}

/// Taylor series for Si(x).
fn si_series(x: f64, steps: &mut Vec<String>) -> f64 {
    let mut sum = 0.0;
    let mut factorial: f64 = 1.0;
    let x2 = x * x;
    let mut power = x; // x^(2k+1) starts at x^1
    for k in 0..50usize {
        let term = power / ((2 * k + 1) as f64 * factorial);
        let signed = if k % 2 == 0 { term } else { -term };
        sum += signed;
        if k < 4 {
            steps.push(format!("  k={}: term = {}", k, signed));
        } else if k == 4 {
            steps.push("  ... (continuing)".to_string());
        }
        if signed.abs() < 1e-15 {
            break;
        }
        // Advance: factorial_{k+1} = (2k+2)(2k+3) · factorial_k
        factorial *= ((2 * k + 2) * (2 * k + 3)) as f64;
        power *= x2;
    }
    sum
}

/// Compute the cosine integral Ci(x).
///
/// Ci(x) = γ + ln(x) + ∫_0^x (cos(t)−1)/t dt  for x > 0
///
/// Series: Ci(x) = γ + ln(x) + Σ_{k=1}^N (-1)^k · x^(2k) / (2k·(2k)!)
#[must_use = "computing special functions returns a result that should be used"]
pub fn ci(x: &Expression) -> Result<SpecialFunctionResult, SpecialFunctionError> {
    let mut steps = Vec::new();
    steps.push(format!("Computing cosine integral: Ci({})", fmt_expr(x)));
    steps.push("Definition: Ci(x) = γ + ln(x) + ∫₀ˣ (cos(t)−1)/t dt  (x>0)".to_string());

    let x_val = match numeric_val(x) {
        Some(v) => v,
        None => {
            steps.push("Argument is symbolic — returning unevaluated Ci(x)".to_string());
            return Ok(SpecialFunctionResult::new(
                Expression::Function(Function::Ci, vec![x.clone()]),
                None,
                steps,
            ));
        }
    };

    if x_val <= 0.0 {
        return Err(SpecialFunctionError::InvalidArgument(format!(
            "Ci(x) is only defined for x > 0 (got x={})",
            x_val
        )));
    }

    let series_part = ci_series(x_val, &mut steps);
    let val = EULER_MASCHERONI + x_val.ln() + series_part;
    steps.push(format!("γ = {}", EULER_MASCHERONI));
    steps.push(format!("ln({}) = {}", x_val, x_val.ln()));
    steps.push(format!("Ci({}) = γ + ln(x) + series = {}", x_val, val));
    Ok(SpecialFunctionResult::new(
        Expression::Float(val),
        Some(val),
        steps,
    ))
}

/// Series part for Ci: Σ_{k=1}^N (-1)^k · x^(2k) / (2k·(2k)!)
fn ci_series(x: f64, steps: &mut Vec<String>) -> f64 {
    let mut sum = 0.0;
    let mut factorial: f64 = 2.0; // (2k)! starting at k=1: 2!
    let x2 = x * x;
    let mut power = x2; // x^(2k) starts at x^2
    for k in 1..50usize {
        let term = power / (2 * k) as f64 / factorial;
        let signed = if k % 2 == 1 { -term } else { term };
        sum += signed;
        if k <= 3 {
            steps.push(format!("  k={}: term = {}", k, signed));
        } else if k == 4 {
            steps.push("  ... (continuing)".to_string());
        }
        if signed.abs() < 1e-15 {
            break;
        }
        // factorial_{k+1} = (2k+1)(2k+2) · factorial_k
        factorial *= ((2 * k + 1) * (2 * k + 2)) as f64;
        power *= x2;
    }
    sum
}

/// Compute the exponential integral Ei(x).
///
/// Ei(x) = γ + ln|x| + Σ_{k=1}^N x^k / (k · k!)  for x ≠ 0
#[must_use = "computing special functions returns a result that should be used"]
pub fn ei(x: &Expression) -> Result<SpecialFunctionResult, SpecialFunctionError> {
    let mut steps = Vec::new();
    steps.push(format!(
        "Computing exponential integral: Ei({})",
        fmt_expr(x)
    ));
    steps.push("Series: Ei(x) = γ + ln|x| + Σ_{k=1}^N x^k / (k·k!)".to_string());

    let x_val = match numeric_val(x) {
        Some(v) => v,
        None => {
            steps.push("Argument is symbolic — returning unevaluated Ei(x)".to_string());
            return Ok(SpecialFunctionResult::new(
                Expression::Function(Function::Ei, vec![x.clone()]),
                None,
                steps,
            ));
        }
    };

    if x_val == 0.0 {
        return Err(SpecialFunctionError::InvalidArgument(
            "Ei(x) has a singularity at x=0".to_string(),
        ));
    }

    let series_part = ei_series(x_val, &mut steps);
    let val = EULER_MASCHERONI + x_val.abs().ln() + series_part;
    steps.push(format!("γ = {}", EULER_MASCHERONI));
    steps.push(format!("ln|{}| = {}", x_val, x_val.abs().ln()));
    steps.push(format!("Ei({}) ≈ {}", x_val, val));
    Ok(SpecialFunctionResult::new(
        Expression::Float(val),
        Some(val),
        steps,
    ))
}

/// Series part for Ei: Σ_{k=1}^N x^k / (k · k!)
fn ei_series(x: f64, steps: &mut Vec<String>) -> f64 {
    let mut sum = 0.0;
    let mut factorial: f64 = 1.0;
    let mut power = x;
    for k in 1..100usize {
        let term = power / (k as f64 * factorial);
        sum += term;
        if k <= 3 {
            steps.push(format!("  k={}: term = {}", k, term));
        } else if k == 4 {
            steps.push("  ... (continuing)".to_string());
        }
        if term.abs() < 1e-15 {
            break;
        }
        factorial *= (k + 1) as f64;
        power *= x;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-3;

    #[test]
    fn test_zeta_2() {
        let result = zeta(&Expression::Integer(2)).unwrap();
        let val = result.numeric_value.unwrap();
        assert!((val - 1.6449).abs() < TOL, "ζ(2) = {}", val);
    }

    #[test]
    fn test_zeta_4() {
        let result = zeta(&Expression::Integer(4)).unwrap();
        let val = result.numeric_value.unwrap();
        assert!((val - 1.0823).abs() < TOL, "ζ(4) = {}", val);
    }

    #[test]
    fn test_zeta_pole() {
        assert!(zeta(&Expression::Integer(1)).is_err());
        assert!(zeta(&Expression::Float(0.5)).is_err());
    }

    #[test]
    fn test_zeta_symbolic() {
        use crate::ast::Variable;
        let s = Expression::Variable(Variable::new("s"));
        let result = zeta(&s).unwrap();
        assert!(result.numeric_value.is_none());
    }

    #[test]
    fn test_si_pi() {
        let result = si(&Expression::Float(PI)).unwrap();
        let val = result.numeric_value.unwrap();
        assert!((val - 1.8519).abs() < TOL, "Si(π) = {}", val);
    }

    #[test]
    fn test_si_zero() {
        let result = si(&Expression::Integer(0)).unwrap();
        let val = result.numeric_value.unwrap();
        assert!(val.abs() < 1e-12, "Si(0) = {}", val);
    }

    #[test]
    fn test_ci_one() {
        let result = ci(&Expression::Float(1.0)).unwrap();
        let val = result.numeric_value.unwrap();
        assert!((val - 0.3374).abs() < TOL, "Ci(1) = {}", val);
    }

    #[test]
    fn test_ci_nonpositive() {
        assert!(ci(&Expression::Float(0.0)).is_err());
        assert!(ci(&Expression::Float(-1.0)).is_err());
    }

    #[test]
    fn test_ei_one() {
        let result = ei(&Expression::Float(1.0)).unwrap();
        let val = result.numeric_value.unwrap();
        assert!((val - 1.8951).abs() < TOL, "Ei(1) = {}", val);
    }

    #[test]
    fn test_ei_zero() {
        assert!(ei(&Expression::Integer(0)).is_err());
    }

    #[test]
    fn test_derivation_steps_populated() {
        let zeta_r = zeta(&Expression::Integer(2)).unwrap();
        assert!(!zeta_r.derivation_steps.is_empty());
        let si_r = si(&Expression::Float(1.0)).unwrap();
        assert!(!si_r.derivation_steps.is_empty());
        let ci_r = ci(&Expression::Float(1.0)).unwrap();
        assert!(!ci_r.derivation_steps.is_empty());
        let ei_r = ei(&Expression::Float(1.0)).unwrap();
        assert!(!ei_r.derivation_steps.is_empty());
    }
}
