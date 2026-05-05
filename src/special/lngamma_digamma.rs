//! Log-Gamma and Digamma functions with step-by-step derivation.
//!
//! - `lngamma(x)` — ln(Γ(x)), numerically stable via Lanczos log-form
//! - `digamma(x)` — ψ(x) = d/dx ln(Γ(x)), via asymptotic + recurrence

use std::f64::consts::PI;

use crate::ast::Expression;

use super::{SpecialFunctionError, SpecialFunctionResult};

// ---------------------------------------------------------------------------
// Lanczos constants (g = 7, same series as gamma_numeric)
// ---------------------------------------------------------------------------

const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEF: [f64; 9] = [
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

// ln(√(2π))
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_74; // 0.5 * ln(2π)

// ---------------------------------------------------------------------------
// Public helpers (crate-visible) for numeric core
// ---------------------------------------------------------------------------

/// Evaluate the Lanczos series sum A_g(x) for x ≥ 0.5.
/// Caller must have already shifted: pass the *original* x (≥ 0.5).
fn lanczos_sum(x: f64) -> f64 {
    let z = x - 1.0;
    let mut a = LANCZOS_COEF[0];
    for (i, &c) in LANCZOS_COEF.iter().enumerate().skip(1) {
        a += c / (z + i as f64);
    }
    a
}

// ---------------------------------------------------------------------------
// lngamma — numeric core
// ---------------------------------------------------------------------------

/// ln(Γ(x)) for any x where Γ(x) is defined (x not a non-positive integer).
pub(super) fn lngamma_numeric(x: f64) -> f64 {
    if x < 0.5 {
        // Reflection: ln Γ(x) = ln(π/sin(πx)) - ln Γ(1-x)
        let ln_pi_over_sin = (PI / (PI * x).sin()).ln();
        ln_pi_over_sin - lngamma_numeric(1.0 - x)
    } else {
        // Lanczos log form
        let z = x - 1.0;
        let t = z + LANCZOS_G + 0.5;
        let a = lanczos_sum(x);
        LN_SQRT_2PI + (z + 0.5) * t.ln() - t + a.ln()
    }
}

// ---------------------------------------------------------------------------
// digamma — numeric core
// ---------------------------------------------------------------------------

/// ψ(x) = d/dx ln Γ(x).
/// Uses recurrence to shift x > 6 then asymptotic expansion.
pub(super) fn digamma_numeric(x: f64) -> f64 {
    if x <= 0.0 && x.fract() == 0.0 {
        return f64::NAN; // pole
    }
    // Shift up via ψ(x+1) = ψ(x) + 1/x until x > 6
    let mut val = x;
    let mut correction = 0.0;
    while val <= 6.0 {
        correction -= 1.0 / val;
        val += 1.0;
    }
    // Asymptotic expansion for val > 6:
    // ψ(val) ≈ ln(val) - 1/(2val) - 1/(12val²) + 1/(120val⁴) - 1/(252val⁶)
    let inv = 1.0 / val;
    let inv2 = inv * inv;
    let asymptotic =
        val.ln() - 0.5 * inv - inv2 / 12.0 + inv2 * inv2 / 120.0 - inv2 * inv2 * inv2 / 252.0;
    asymptotic + correction
}

// ---------------------------------------------------------------------------
// extract numeric value from Expression
// ---------------------------------------------------------------------------

fn expr_to_f64(x: &Expression) -> Option<f64> {
    match x {
        Expression::Integer(n) => Some(*n as f64),
        Expression::Float(f) => Some(*f),
        Expression::Rational(r) => Some(*r.numer() as f64 / *r.denom() as f64),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// lngamma — public entry
// ---------------------------------------------------------------------------

/// Compute ln(Γ(x)) with derivation steps.
///
/// # Definition
/// ln(Γ(x)) via Lanczos log-form for x ≥ 0.5; reflection formula for x < 0.5.
///
/// # Known values
/// - lnΓ(1) = 0
/// - lnΓ(5) = ln(24) ≈ 3.178
/// - lnΓ(½) = ½·ln(π) ≈ 0.5724
///
/// # Errors
/// - `InvalidArgument` for non-positive integers (poles of Γ).
/// - `NotImplemented` for symbolic expressions.
#[must_use = "computing lngamma returns a result that should be used"]
pub fn lngamma(x: &Expression) -> Result<SpecialFunctionResult, SpecialFunctionError> {
    let mut steps = Vec::new();
    steps.push(format!("Computing ln(Γ({})):", fmt(x)));

    // Guard: non-positive integer → pole
    if let Expression::Integer(n) = x {
        if *n <= 0 {
            return Err(SpecialFunctionError::InvalidArgument(format!(
                "Γ has a pole at non-positive integer {}; ln(Γ) undefined",
                n
            )));
        }
    }

    let x_val = expr_to_f64(x).ok_or_else(|| {
        SpecialFunctionError::NotImplemented(format!(
            "lngamma not implemented for symbolic expression: {}",
            fmt(x)
        ))
    })?;

    if x_val <= 0.0 && x_val.fract() == 0.0 {
        return Err(SpecialFunctionError::InvalidArgument(format!(
            "Γ has a pole at {}; ln(Γ) undefined",
            x_val
        )));
    }

    add_lngamma_steps(&mut steps, x_val);

    let result = lngamma_numeric(x_val);
    steps.push(format!("ln(Γ({})) ≈ {:.10}", x_val, result));

    Ok(SpecialFunctionResult::new(
        Expression::Float(result),
        Some(result),
        steps,
    ))
}

/// Populate derivation steps for lngamma (keeps lngamma() within 80 lines).
fn add_lngamma_steps(steps: &mut Vec<String>, x: f64) {
    if x < 0.5 {
        steps.push("x < 0.5 → use reflection formula:".to_string());
        steps.push("  ln Γ(x) = ln(π / sin(πx)) − ln Γ(1−x)".to_string());
        let ln_pi_sin = (PI / (PI * x).sin()).ln();
        steps.push(format!("  ln(π / sin(π·{})) = {:.10}", x, ln_pi_sin));
        let lngamma_1mx = lngamma_numeric(1.0 - x);
        steps.push(format!("  ln Γ({}) = {:.10}", 1.0 - x, lngamma_1mx));
    } else {
        steps.push("x ≥ 0.5 → use Lanczos log form:".to_string());
        steps.push("  ln Γ(x) = ½·ln(2π) + (x−½)·ln(x+g−½) − (x+g−½) + ln(A_g(x))".to_string());
        let z = x - 1.0;
        let t = z + LANCZOS_G + 0.5;
        let a = lanczos_sum(x);
        steps.push(format!("  g = {}, t = x+g−½ = {:.6}", LANCZOS_G, t));
        steps.push(format!("  A_g({}) = {:.10}", x, a));
        steps.push(format!(
            "  = {:.6} + {:.6}·ln({:.6}) − {:.6} + ln({:.10})",
            LN_SQRT_2PI,
            z + 0.5,
            t,
            t,
            a
        ));
    }
}

// ---------------------------------------------------------------------------
// digamma — public entry
// ---------------------------------------------------------------------------

/// Compute ψ(x) = d/dx ln(Γ(x)) with derivation steps.
///
/// # Algorithm
/// For x > 6: asymptotic expansion.
/// For x ≤ 6: recurrence ψ(x+1) = ψ(x) + 1/x shifts argument up, then asymptotic.
///
/// # Known values
/// - ψ(1) = −γ ≈ −0.5772
/// - ψ(2) = 1 − γ ≈ 0.4228
///
/// # Errors
/// - `InvalidArgument` for non-positive integers (poles).
/// - `NotImplemented` for symbolic expressions.
#[must_use = "computing digamma returns a result that should be used"]
pub fn digamma(x: &Expression) -> Result<SpecialFunctionResult, SpecialFunctionError> {
    let mut steps = Vec::new();
    steps.push(format!("Computing ψ({}) (digamma):", fmt(x)));

    // Guard: non-positive integer → pole
    if let Expression::Integer(n) = x {
        if *n <= 0 {
            return Err(SpecialFunctionError::InvalidArgument(format!(
                "digamma has a pole at non-positive integer {}",
                n
            )));
        }
    }

    let x_val = expr_to_f64(x).ok_or_else(|| {
        SpecialFunctionError::NotImplemented(format!(
            "digamma not implemented for symbolic expression: {}",
            fmt(x)
        ))
    })?;

    if x_val <= 0.0 && x_val.fract() == 0.0 {
        return Err(SpecialFunctionError::InvalidArgument(format!(
            "digamma has a pole at {}",
            x_val
        )));
    }

    add_digamma_steps(&mut steps, x_val);

    let result = digamma_numeric(x_val);
    steps.push(format!("ψ({}) ≈ {:.10}", x_val, result));

    Ok(SpecialFunctionResult::new(
        Expression::Float(result),
        Some(result),
        steps,
    ))
}

/// Populate derivation steps for digamma (keeps digamma() within 80 lines).
fn add_digamma_steps(steps: &mut Vec<String>, x: f64) {
    let mut val = x;
    let mut recurrence_steps: Vec<String> = Vec::new();
    while val <= 6.0 {
        recurrence_steps.push(format!(
            "  ψ({:.4}) = ψ({:.4}) − 1/{:.4}",
            x,
            val + 1.0,
            val
        ));
        val += 1.0;
    }
    if !recurrence_steps.is_empty() {
        steps.push(format!(
            "x = {} ≤ 6 → apply recurrence ψ(x) = ψ(x+1) − 1/x until x > 6:",
            x
        ));
        steps.extend(recurrence_steps);
        steps.push(format!("  shifted argument: {:.4}", val));
    }
    steps.push(format!("Asymptotic expansion for x = {:.4}:", val));
    steps.push("  ψ(x) ≈ ln(x) − 1/(2x) − 1/(12x²) + 1/(120x⁴) − 1/(252x⁶)".to_string());
    let inv = 1.0 / val;
    let inv2 = inv * inv;
    let terms = [
        val.ln(),
        -0.5 * inv,
        -inv2 / 12.0,
        inv2 * inv2 / 120.0,
        -(inv2 * inv2 * inv2) / 252.0,
    ];
    steps.push(format!(
        "  = {:.8} − {:.8} − {:.8} + {:.8} − {:.8}",
        terms[0], -terms[1], -terms[2], terms[3], -terms[4]
    ));
}

// ---------------------------------------------------------------------------
// Formatting helper (mirrors special.rs internal)
// ---------------------------------------------------------------------------

fn fmt(expr: &Expression) -> String {
    match expr {
        Expression::Integer(n) => n.to_string(),
        Expression::Float(f) => f.to_string(),
        Expression::Rational(r) => format!("{}/{}", r.numer(), r.denom()),
        Expression::Variable(v) => v.name.clone(),
        _ => format!("{:?}", expr),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-4;

    // --- lngamma ---

    #[test]
    fn fast_lngamma_one_is_zero() {
        let r = lngamma(&Expression::Integer(1)).unwrap();
        let v = r.numeric_value.unwrap();
        assert!(v.abs() < TOL, "lnΓ(1) should be 0, got {}", v);
        assert!(!r.derivation_steps.is_empty());
    }

    #[test]
    fn fast_lngamma_five_is_ln24() {
        // Γ(5) = 4! = 24, so lnΓ(5) = ln(24)
        let expected = 24.0_f64.ln();
        let r = lngamma(&Expression::Integer(5)).unwrap();
        let v = r.numeric_value.unwrap();
        assert!(
            (v - expected).abs() < TOL,
            "lnΓ(5) ≈ {}, got {}",
            expected,
            v
        );
    }

    #[test]
    fn fast_lngamma_half_is_ln_sqrt_pi() {
        // Γ(½) = √π, so lnΓ(½) = ½·ln(π)
        let expected = 0.5 * PI.ln();
        let r = lngamma(&Expression::Rational(num_rational::Rational64::new(1, 2))).unwrap();
        let v = r.numeric_value.unwrap();
        assert!(
            (v - expected).abs() < TOL,
            "lnΓ(½) ≈ {}, got {}",
            expected,
            v
        );
    }

    #[test]
    fn fast_lngamma_float_input() {
        let r = lngamma(&Expression::Float(3.5)).unwrap();
        assert!(r.numeric_value.is_some());
        // Γ(3.5) = 2.5·1.5·0.5·√π ≈ 3.3234, lnΓ ≈ 1.2009
        let v = r.numeric_value.unwrap();
        assert!((v - 1.2009).abs() < 0.001, "lnΓ(3.5) ≈ 1.2009, got {}", v);
    }

    #[test]
    fn fast_lngamma_negative_integer_errors() {
        assert!(lngamma(&Expression::Integer(0)).is_err());
        assert!(lngamma(&Expression::Integer(-2)).is_err());
    }

    // --- digamma ---

    #[test]
    fn fast_digamma_one_is_neg_euler() {
        // ψ(1) = -γ ≈ -0.5772156649
        let expected = -0.577_215_664_9_f64;
        let r = digamma(&Expression::Integer(1)).unwrap();
        let v = r.numeric_value.unwrap();
        assert!((v - expected).abs() < TOL, "ψ(1) ≈ {}, got {}", expected, v);
        assert!(!r.derivation_steps.is_empty());
    }

    #[test]
    fn fast_digamma_two_is_one_minus_euler() {
        // ψ(2) = 1 - γ ≈ 0.4228
        let expected = 1.0 - 0.577_215_664_9_f64;
        let r = digamma(&Expression::Integer(2)).unwrap();
        let v = r.numeric_value.unwrap();
        assert!((v - expected).abs() < TOL, "ψ(2) ≈ {}, got {}", expected, v);
    }

    #[test]
    fn fast_digamma_large_argument() {
        // For large x, ψ(x) ≈ ln(x). Check ψ(100) is close to ln(100).
        let r = digamma(&Expression::Float(100.0)).unwrap();
        let v = r.numeric_value.unwrap();
        let approx = 100.0_f64.ln();
        assert!((v - approx).abs() < 0.01, "ψ(100) ≈ ln(100), got {}", v);
    }

    #[test]
    fn fast_digamma_negative_integer_errors() {
        assert!(digamma(&Expression::Integer(0)).is_err());
        assert!(digamma(&Expression::Integer(-1)).is_err());
    }

    #[test]
    fn fast_digamma_float_input() {
        let r = digamma(&Expression::Float(1.5)).unwrap();
        // ψ(1.5) = -γ + 2 - 2·ln(2) ≈ +0.03649
        let v = r.numeric_value.unwrap();
        assert!(v.is_finite(), "ψ(1.5) should be finite, got {}", v);
        assert!((v - 0.03649).abs() < 0.001, "ψ(1.5) ≈ +0.036, got {}", v);
    }

    #[test]
    fn fast_lngamma_derivation_steps_mention_lanczos_or_reflection() {
        let r1 = lngamma(&Expression::Integer(5)).unwrap();
        let has_lanczos = r1.derivation_steps.iter().any(|s| s.contains("Lanczos"));
        assert!(has_lanczos, "steps should mention Lanczos for x≥0.5");

        let r2 = lngamma(&Expression::Float(0.3)).unwrap();
        let has_reflection = r2.derivation_steps.iter().any(|s| s.contains("reflection"));
        assert!(has_reflection, "steps should mention reflection for x<0.5");
    }

    #[test]
    fn fast_digamma_steps_mention_recurrence_for_small_x() {
        let r = digamma(&Expression::Integer(2)).unwrap();
        let has_recurrence = r.derivation_steps.iter().any(|s| s.contains("recurrence"));
        assert!(has_recurrence, "steps should mention recurrence for x≤6");
    }
}
