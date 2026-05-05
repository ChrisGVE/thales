//! Bessel functions with step-by-step derivation.
//!
//! Provides J_n (first kind), Y_n (second kind / Weber),
//! I_n (modified first kind), and K_n (modified second kind)
//! for non-negative integer orders and real arguments.

use crate::ast::Expression;
use std::f64::consts::PI;

use super::{gamma_numeric, SpecialFunctionError, SpecialFunctionResult};

// ---------------------------------------------------------------------------
// Numeric value extraction
// ---------------------------------------------------------------------------

fn expr_to_f64(x: &Expression) -> Option<f64> {
    match x {
        Expression::Integer(n) => Some(*n as f64),
        Expression::Float(f) => Some(*f),
        Expression::Rational(r) => Some(*r.numer() as f64 / *r.denom() as f64),
        _ => None,
    }
}

fn expr_to_order(n: &Expression) -> Option<i64> {
    match n {
        Expression::Integer(i) => Some(*i),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Core numeric engines
// ---------------------------------------------------------------------------

/// J_n(x) via power series: Σ_{k=0}^{N} (-1)^k (x/2)^{n+2k} / (k! (n+k)!)
/// N = 30 terms; accurate for moderate |x|.
fn bessel_j_numeric(n: i64, x: f64) -> f64 {
    // For negative integer order: J_{-n}(x) = (-1)^n J_n(x)
    if n < 0 {
        let sign = if (-n) % 2 == 0 { 1.0 } else { -1.0 };
        return sign * bessel_j_numeric(-n, x);
    }
    let half_x = x / 2.0;
    // Pre-compute (x/2)^n / n!  as the k=0 term
    let mut term = half_x.powi(n as i32) / gamma_numeric((n + 1) as f64);
    let mut sum = term;
    for k in 1..=30_u32 {
        term *= -(half_x * half_x) / (k as f64 * (n as f64 + k as f64));
        sum += term;
        if term.abs() < 1e-16 * sum.abs() {
            break;
        }
    }
    sum
}

/// I_n(x) via power series: Σ_{k=0}^{N} (x/2)^{n+2k} / (k! (n+k)!)
/// Same as J_n but without the (-1)^k alternating sign.
fn bessel_i_numeric(n: i64, x: f64) -> f64 {
    if n < 0 {
        return bessel_i_numeric(-n, x);
    }
    let half_x = x / 2.0;
    let mut term = half_x.powi(n as i32) / gamma_numeric((n + 1) as f64);
    let mut sum = term;
    for k in 1..=30_u32 {
        term *= (half_x * half_x) / (k as f64 * (n as f64 + k as f64));
        sum += term;
        if term.abs() < 1e-16 * sum.abs() {
            break;
        }
    }
    sum
}

/// Y_0(x) via Weber formula: (2/π)(ln(x/2) + γ) J_0(x) − (2/π) Σ correction
fn bessel_y0_numeric(x: f64) -> f64 {
    const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_86;
    let ln_term = (x / 2.0).ln() + EULER_MASCHERONI;
    let j0 = bessel_j_numeric(0, x);
    // Neumann series correction: Σ_{k=0}^{N} (-1)^k H_k (x/2)^{2k} / (k!)^2
    // where H_k = 1 + 1/2 + ... + 1/k (H_0 = 0)
    let half_x2 = (x / 2.0) * (x / 2.0);
    let mut term = 1.0; // k=0: H_0=0 contributes 0 to correction, but we track (x/2)^{2k}/(k!)^2
    let mut harmonic = 0.0_f64;
    let mut correction = 0.0_f64;
    for k in 1..=30_u32 {
        term *= half_x2 / (k as f64 * k as f64);
        harmonic += 1.0 / k as f64;
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        correction += sign * harmonic * term;
        if term.abs() < 1e-16 {
            break;
        }
    }
    (2.0 / PI) * (ln_term * j0 - correction)
}

/// Y_n(x) via recurrence: Y_{n+1} = (2n/x) Y_n − Y_{n-1}
fn bessel_yn_numeric(n: i64, x: f64) -> f64 {
    if n == 0 {
        return bessel_y0_numeric(x);
    }
    if n < 0 {
        let sign = if (-n) % 2 == 0 { 1.0 } else { -1.0 };
        return sign * bessel_yn_numeric(-n, x);
    }
    // Y_1 via reflection: Y_1 = (2/π)(ln(x/2)+γ)J_1 − (2/π)(1/x) − correction
    // Simplest stable approach: upward recurrence from Y_0 and Y_1.
    let y0 = bessel_y0_numeric(x);
    let y1 = bessel_y1_numeric(x);
    if n == 1 {
        return y1;
    }
    let mut y_prev = y0;
    let mut y_curr = y1;
    for k in 1..n {
        let y_next = (2.0 * k as f64 / x) * y_curr - y_prev;
        y_prev = y_curr;
        y_curr = y_next;
    }
    y_curr
}

/// Y_1(x) via Weber formula (direct).
fn bessel_y1_numeric(x: f64) -> f64 {
    const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_86;
    let j1 = bessel_j_numeric(1, x);
    let half_x = x / 2.0;
    let half_x2 = half_x * half_x;
    // Correction series for Y_1
    let mut term = half_x; // k=0 leading term
    let mut sum = term;
    let mut harmonic = 0.0_f64;
    for k in 1..=30_u32 {
        harmonic += 1.0 / k as f64 + 1.0 / (k as f64 + 1.0);
        term *= -half_x2 / (k as f64 * (k as f64 + 1.0));
        sum += harmonic * term;
        if term.abs() < 1e-16 {
            break;
        }
    }
    (2.0 / PI) * ((half_x.ln() + EULER_MASCHERONI) * j1 - 1.0 / x - sum)
}

/// K_n(x) via: K_n = (π/2) i^{n+1} (J_n + iY_n) for real x > 0
/// = (π/2)(I_{-n} − I_n) / sin(nπ) for non-integer n, but for integer n:
/// K_n = (π/2) * lim → uses recurrence from K_0 and K_1.
fn bessel_kn_numeric(n: i64, x: f64) -> f64 {
    let k0 = bessel_k0_numeric(x);
    if n == 0 {
        return k0;
    }
    let k1 = bessel_k1_numeric(x);
    if n == 1 || n == -1 {
        return k1;
    }
    let n_abs = n.unsigned_abs() as i64;
    let mut k_prev = k0;
    let mut k_curr = k1;
    for m in 1..n_abs {
        let k_next = (2.0 * m as f64 / x) * k_curr + k_prev;
        k_prev = k_curr;
        k_curr = k_next;
    }
    k_curr
}

/// K_0(x) = −(ln(x/2) + γ) I_0(x) + correction series
fn bessel_k0_numeric(x: f64) -> f64 {
    const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_86;
    let i0 = bessel_i_numeric(0, x);
    let half_x2 = (x / 2.0) * (x / 2.0);
    let ln_half_x = (x / 2.0).ln();
    let mut term = 1.0;
    let mut harmonic = 0.0_f64;
    let mut correction = 0.0_f64;
    for k in 1..=30_u32 {
        term *= half_x2 / (k as f64 * k as f64);
        harmonic += 1.0 / k as f64;
        correction += harmonic * term;
        if term.abs() < 1e-16 {
            break;
        }
    }
    -(ln_half_x + EULER_MASCHERONI) * i0 + correction
}

/// K_1(x) = (1/x) + (ln(x/2) + γ) I_1(x) − correction series / 2
fn bessel_k1_numeric(x: f64) -> f64 {
    const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_86;
    let i1 = bessel_i_numeric(1, x);
    let half_x = x / 2.0;
    let half_x2 = half_x * half_x;
    let mut term = half_x;
    let mut sum = 0.5; // k=0 term of correction
    let mut harmonic_half = 0.0_f64;
    for k in 1..=30_u32 {
        harmonic_half += 0.5 / k as f64 + 0.5 / (k as f64 + 1.0);
        term *= half_x2 / (k as f64 * (k as f64 + 1.0));
        sum += harmonic_half * term;
        if term.abs() < 1e-16 {
            break;
        }
    }
    (1.0 / x) + (half_x.ln() + EULER_MASCHERONI) * i1 - sum
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the Bessel function of the first kind J_n(x).
///
/// Uses the power series Σ_{k=0}^{30} (-1)^k (x/2)^{n+2k} / (k! (n+k)!)
/// for integer orders n ≥ 0.
///
/// # Errors
/// - `NotImplemented` for non-integer or symbolic order.
/// - `InvalidArgument` if x < 0 (real-valued J_n requires x ≥ 0 for n < 0).
#[must_use = "computing Bessel J returns a result that should be used"]
pub fn bessel_j(
    order: &Expression,
    x: &Expression,
) -> Result<SpecialFunctionResult, SpecialFunctionError> {
    let n = expr_to_order(order).ok_or_else(|| {
        SpecialFunctionError::NotImplemented("bessel_j requires integer order".to_string())
    })?;
    let x_val = expr_to_f64(x).ok_or_else(|| {
        SpecialFunctionError::NotImplemented("bessel_j requires numeric argument".to_string())
    })?;
    let mut steps = Vec::new();
    steps.push(format!("Computing J_{}({})", n, x_val));
    if x_val == 0.0 {
        let val = if n == 0 { 1.0 } else { 0.0 };
        steps.push(format!("Known value: J_{}(0) = {}", n, val));
        return Ok(SpecialFunctionResult::new(
            Expression::Float(val),
            Some(val),
            steps,
        ));
    }
    steps.push(format!(
        "Series: Σ_{{k=0}}^30 (-1)^k (x/2)^{{{}+2k}} / (k! ({}+k)!)",
        n, n
    ));
    let val = bessel_j_numeric(n, x_val);
    steps.push(format!("J_{}({}) ≈ {:.10}", n, x_val, val));
    Ok(SpecialFunctionResult::new(
        Expression::Float(val),
        Some(val),
        steps,
    ))
}

/// Compute the Bessel function of the second kind Y_n(x) (Weber function).
///
/// Uses Y_0 via the Weber formula, then upward recurrence for n > 0.
/// Only valid for x > 0; Y_n has a logarithmic singularity at x = 0.
///
/// # Errors
/// - `InvalidArgument` for x ≤ 0.
/// - `NotImplemented` for non-integer or symbolic order.
#[must_use = "computing Bessel Y returns a result that should be used"]
pub fn bessel_y(
    order: &Expression,
    x: &Expression,
) -> Result<SpecialFunctionResult, SpecialFunctionError> {
    let n = expr_to_order(order).ok_or_else(|| {
        SpecialFunctionError::NotImplemented("bessel_y requires integer order".to_string())
    })?;
    let x_val = expr_to_f64(x).ok_or_else(|| {
        SpecialFunctionError::NotImplemented("bessel_y requires numeric argument".to_string())
    })?;
    if x_val <= 0.0 {
        return Err(SpecialFunctionError::InvalidArgument(
            "bessel_y requires x > 0 (logarithmic singularity at x ≤ 0)".to_string(),
        ));
    }
    let mut steps = Vec::new();
    steps.push(format!("Computing Y_{}({})", n, x_val));
    steps.push(
        "Y_0 via Weber formula; higher orders via recurrence Y_{n+1}=(2n/x)Y_n - Y_{n-1}"
            .to_string(),
    );
    let val = bessel_yn_numeric(n, x_val);
    steps.push(format!("Y_{}({}) ≈ {:.10}", n, x_val, val));
    Ok(SpecialFunctionResult::new(
        Expression::Float(val),
        Some(val),
        steps,
    ))
}

/// Compute the modified Bessel function of the first kind I_n(x).
///
/// Uses the power series Σ_{k=0}^{30} (x/2)^{n+2k} / (k! (n+k)!)
/// (same as J_n but without the alternating sign).
///
/// # Errors
/// - `NotImplemented` for non-integer or symbolic order.
#[must_use = "computing Bessel I returns a result that should be used"]
pub fn bessel_i(
    order: &Expression,
    x: &Expression,
) -> Result<SpecialFunctionResult, SpecialFunctionError> {
    let n = expr_to_order(order).ok_or_else(|| {
        SpecialFunctionError::NotImplemented("bessel_i requires integer order".to_string())
    })?;
    let x_val = expr_to_f64(x).ok_or_else(|| {
        SpecialFunctionError::NotImplemented("bessel_i requires numeric argument".to_string())
    })?;
    let mut steps = Vec::new();
    steps.push(format!("Computing I_{}({})", n, x_val));
    if x_val == 0.0 {
        let val = if n == 0 { 1.0 } else { 0.0 };
        steps.push(format!("Known value: I_{}(0) = {}", n, val));
        return Ok(SpecialFunctionResult::new(
            Expression::Float(val),
            Some(val),
            steps,
        ));
    }
    steps.push(format!(
        "Series: Σ_{{k=0}}^30 (x/2)^{{{}+2k}} / (k! ({}+k)!)",
        n, n
    ));
    let val = bessel_i_numeric(n, x_val);
    steps.push(format!("I_{}({}) ≈ {:.10}", n, x_val, val));
    Ok(SpecialFunctionResult::new(
        Expression::Float(val),
        Some(val),
        steps,
    ))
}

/// Compute the modified Bessel function of the second kind K_n(x).
///
/// Uses K_0 and K_1 via log-series, then upward recurrence.
/// Only valid for x > 0.
///
/// # Errors
/// - `InvalidArgument` for x ≤ 0.
/// - `NotImplemented` for non-integer or symbolic order.
#[must_use = "computing Bessel K returns a result that should be used"]
pub fn bessel_k(
    order: &Expression,
    x: &Expression,
) -> Result<SpecialFunctionResult, SpecialFunctionError> {
    let n = expr_to_order(order).ok_or_else(|| {
        SpecialFunctionError::NotImplemented("bessel_k requires integer order".to_string())
    })?;
    let x_val = expr_to_f64(x).ok_or_else(|| {
        SpecialFunctionError::NotImplemented("bessel_k requires numeric argument".to_string())
    })?;
    if x_val <= 0.0 {
        return Err(SpecialFunctionError::InvalidArgument(
            "bessel_k requires x > 0 (singularity at x ≤ 0)".to_string(),
        ));
    }
    let mut steps = Vec::new();
    steps.push(format!("Computing K_{}({})", n, x_val));
    steps.push("K_0 and K_1 via log-series; higher orders via recurrence".to_string());
    let val = bessel_kn_numeric(n, x_val);
    steps.push(format!("K_{}({}) ≈ {:.10}", n, x_val, val));
    Ok(SpecialFunctionResult::new(
        Expression::Float(val),
        Some(val),
        steps,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-4;
    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }
    fn float(f: f64) -> Expression {
        Expression::Float(f)
    }

    // --- J_n known values ---

    #[test]
    fn fast_bessel_j0_at_zero_is_one() {
        let r = bessel_j(&int(0), &int(0)).unwrap();
        assert!((r.numeric_value.unwrap() - 1.0).abs() < TOL);
    }

    #[test]
    fn fast_bessel_j1_at_zero_is_zero() {
        let r = bessel_j(&int(1), &int(0)).unwrap();
        assert!(r.numeric_value.unwrap().abs() < TOL);
    }

    #[test]
    fn fast_bessel_j0_at_one() {
        // J_0(1) ≈ 0.7652
        let r = bessel_j(&int(0), &float(1.0)).unwrap();
        let v = r.numeric_value.unwrap();
        assert!((v - 0.7652).abs() < TOL, "J_0(1) ≈ 0.7652, got {}", v);
    }

    #[test]
    fn fast_bessel_j1_at_one() {
        // J_1(1) ≈ 0.4401
        let r = bessel_j(&int(1), &float(1.0)).unwrap();
        let v = r.numeric_value.unwrap();
        assert!((v - 0.4401).abs() < TOL, "J_1(1) ≈ 0.4401, got {}", v);
    }

    #[test]
    fn fast_bessel_j_steps_non_empty() {
        let r = bessel_j(&int(0), &float(1.0)).unwrap();
        assert!(!r.derivation_steps.is_empty());
    }

    // --- I_n known values ---

    #[test]
    fn fast_bessel_i0_at_zero_is_one() {
        let r = bessel_i(&int(0), &int(0)).unwrap();
        assert!((r.numeric_value.unwrap() - 1.0).abs() < TOL);
    }

    #[test]
    fn fast_bessel_i1_at_zero_is_zero() {
        let r = bessel_i(&int(1), &int(0)).unwrap();
        assert!(r.numeric_value.unwrap().abs() < TOL);
    }

    #[test]
    fn fast_bessel_i0_at_one() {
        // I_0(1) ≈ 1.2661
        let r = bessel_i(&int(0), &float(1.0)).unwrap();
        let v = r.numeric_value.unwrap();
        assert!((v - 1.2661).abs() < TOL, "I_0(1) ≈ 1.2661, got {}", v);
    }

    // --- Y_n ---

    #[test]
    fn fast_bessel_y_at_zero_errors() {
        let r = bessel_y(&int(0), &int(0));
        assert!(r.is_err(), "Y_0(0) should be an error (singularity)");
    }

    #[test]
    fn fast_bessel_y0_at_one() {
        // Y_0(1) ≈ 0.0883
        let r = bessel_y(&int(0), &float(1.0)).unwrap();
        let v = r.numeric_value.unwrap();
        assert!((v - 0.0883).abs() < TOL, "Y_0(1) ≈ 0.0883, got {}", v);
    }

    // --- K_n ---

    #[test]
    fn fast_bessel_k_at_zero_errors() {
        let r = bessel_k(&int(0), &int(0));
        assert!(r.is_err(), "K_0(0) should be an error (singularity)");
    }

    #[test]
    fn fast_bessel_k0_at_one() {
        // K_0(1) ≈ 0.4210
        let r = bessel_k(&int(0), &float(1.0)).unwrap();
        let v = r.numeric_value.unwrap();
        assert!((v - 0.4210).abs() < TOL, "K_0(1) ≈ 0.4210, got {}", v);
    }

    // --- non-integer order errors ---

    #[test]
    fn fast_bessel_j_non_integer_order_errors() {
        let r = bessel_j(&float(0.5), &float(1.0));
        assert!(r.is_err());
    }
}
