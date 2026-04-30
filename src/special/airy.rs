//! Airy functions with step-by-step derivation.
//!
//! Provides Ai(x) and Bi(x) via Taylor series around x = 0.
//!
//! The series are defined by the two linearly independent solutions of
//! the Airy equation y'' = x·y:
//!
//!   f(x) = 1 + x³/3! + x⁶·2/(6!) + ...   (even-like starting series)
//!   g(x) = x + x⁴/4! + x⁷·2/(7!) + ...   (odd-like starting series)
//!
//! Ai(x) = c1·f(x) − c2·g(x)
//! Bi(x) = √3·(c1·f(x) + c2·g(x))
//!
//! where c1 = 1/(3^{2/3}·Γ(2/3)),  c2 = 1/(3^{1/3}·Γ(1/3)).

use crate::ast::Expression;

use super::{gamma_numeric, SpecialFunctionError, SpecialFunctionResult};

// ---------------------------------------------------------------------------
// Series constants
// ---------------------------------------------------------------------------

/// Number of terms in each subseries.
const N_TERMS: usize = 20;

// ---------------------------------------------------------------------------
// Constants derived from Γ values (computed at call time to avoid lazy-static)
// ---------------------------------------------------------------------------

fn airy_constants() -> (f64, f64) {
    // c1 = 1 / (3^{2/3} · Γ(2/3))
    let c1 = 1.0 / (3.0_f64.powf(2.0 / 3.0) * gamma_numeric(2.0 / 3.0));
    // c2 = 1 / (3^{1/3} · Γ(1/3))
    let c2 = 1.0 / (3.0_f64.powf(1.0 / 3.0) * gamma_numeric(1.0 / 3.0));
    (c1, c2)
}

// ---------------------------------------------------------------------------
// Series evaluation
// ---------------------------------------------------------------------------

/// f(x) = Σ_{k=0}^{N-1}  a_k · x^{3k}
///
/// a_0 = 1;  a_{k+1} = a_k / ((3k+2)(3k+3))
fn airy_f_series(x: f64) -> f64 {
    let x3 = x * x * x;
    let mut term = 1.0_f64;
    let mut sum = term;
    for k in 0..(N_TERMS - 1) {
        let denom = (3 * k + 2) as f64 * (3 * k + 3) as f64;
        term *= x3 / denom;
        sum += term;
        if term.abs() < 1e-16 * sum.abs() {
            break;
        }
    }
    sum
}

/// g(x) = x · Σ_{k=0}^{N-1}  b_k · x^{3k}
///
/// b_0 = 1;  b_{k+1} = b_k / ((3k+3)(3k+4))
fn airy_g_series(x: f64) -> f64 {
    let x3 = x * x * x;
    let mut term = 1.0_f64;
    let mut sum = term;
    for k in 0..(N_TERMS - 1) {
        let denom = (3 * k + 3) as f64 * (3 * k + 4) as f64;
        term *= x3 / denom;
        sum += term;
        if term.abs() < 1e-16 * sum.abs() {
            break;
        }
    }
    x * sum
}

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

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the Airy function Ai(x).
///
/// Uses the Taylor series Ai(x) = c1·f(x) − c2·g(x) around x = 0,
/// where c1 = 1/(3^{2/3}·Γ(2/3)) and c2 = 1/(3^{1/3}·Γ(1/3)).
///
/// # Known values
/// - Ai(0) ≈ 0.3550
///
/// # Errors
/// - `NotImplemented` for symbolic arguments.
#[must_use = "computing Airy Ai returns a result that should be used"]
pub fn airy_ai(x: &Expression) -> Result<SpecialFunctionResult, SpecialFunctionError> {
    let x_val = expr_to_f64(x).ok_or_else(|| {
        SpecialFunctionError::NotImplemented("airy_ai requires a numeric argument".to_string())
    })?;

    let mut steps = Vec::new();
    steps.push(format!("Computing Ai({})", x_val));
    steps.push("Using Taylor series: Ai(x) = c1·f(x) − c2·g(x)".to_string());
    steps.push(format!(
        "  f(x) = Σ_{{k=0}}^{} x^{{3k}} / prod_{{j=0}}^{{k-1}} (3j+2)(3j+3)",
        N_TERMS
    ));
    steps.push(format!(
        "  g(x) = x·Σ_{{k=0}}^{} x^{{3k}} / prod_{{j=0}}^{{k-1}} (3j+3)(3j+4)",
        N_TERMS
    ));

    let (c1, c2) = airy_constants();
    let f_val = airy_f_series(x_val);
    let g_val = airy_g_series(x_val);

    steps.push(format!("  c1 = 1/(3^{{2/3}}·Γ(2/3)) ≈ {:.8}", c1));
    steps.push(format!("  c2 = 1/(3^{{1/3}}·Γ(1/3)) ≈ {:.8}", c2));
    steps.push(format!("  f({}) ≈ {:.10}", x_val, f_val));
    steps.push(format!("  g({}) ≈ {:.10}", x_val, g_val));

    let val = c1 * f_val - c2 * g_val;
    steps.push(format!(
        "Ai({}) = {:.8}·{:.8} − {:.8}·{:.8} ≈ {:.10}",
        x_val, c1, f_val, c2, g_val, val
    ));

    Ok(SpecialFunctionResult::new(
        Expression::Float(val),
        Some(val),
        steps,
    ))
}

/// Compute the Airy function Bi(x).
///
/// Uses the Taylor series Bi(x) = √3·(c1·f(x) + c2·g(x)) around x = 0,
/// where c1 = 1/(3^{2/3}·Γ(2/3)) and c2 = 1/(3^{1/3}·Γ(1/3)).
///
/// # Known values
/// - Bi(0) ≈ 0.6149
///
/// # Errors
/// - `NotImplemented` for symbolic arguments.
#[must_use = "computing Airy Bi returns a result that should be used"]
pub fn airy_bi(x: &Expression) -> Result<SpecialFunctionResult, SpecialFunctionError> {
    let x_val = expr_to_f64(x).ok_or_else(|| {
        SpecialFunctionError::NotImplemented("airy_bi requires a numeric argument".to_string())
    })?;

    let mut steps = Vec::new();
    steps.push(format!("Computing Bi({})", x_val));
    steps.push("Using Taylor series: Bi(x) = √3·(c1·f(x) + c2·g(x))".to_string());

    let (c1, c2) = airy_constants();
    let f_val = airy_f_series(x_val);
    let g_val = airy_g_series(x_val);
    let sqrt3 = 3.0_f64.sqrt();

    steps.push(format!("  c1 ≈ {:.8}, c2 ≈ {:.8}", c1, c2));
    steps.push(format!("  f({}) ≈ {:.10}", x_val, f_val));
    steps.push(format!("  g({}) ≈ {:.10}", x_val, g_val));

    let val = sqrt3 * (c1 * f_val + c2 * g_val);
    steps.push(format!(
        "Bi({}) = √3·({:.8}·{:.8} + {:.8}·{:.8}) ≈ {:.10}",
        x_val, c1, f_val, c2, g_val, val
    ));

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

    fn float(f: f64) -> Expression {
        Expression::Float(f)
    }
    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    #[test]
    fn fast_airy_ai_at_zero() {
        // Ai(0) ≈ 0.3550280539
        let r = airy_ai(&int(0)).unwrap();
        let v = r.numeric_value.unwrap();
        assert!((v - 0.3550).abs() < TOL, "Ai(0) ≈ 0.3550, got {}", v);
    }

    #[test]
    fn fast_airy_bi_at_zero() {
        // Bi(0) ≈ 0.6149266274
        let r = airy_bi(&int(0)).unwrap();
        let v = r.numeric_value.unwrap();
        assert!((v - 0.6149).abs() < TOL, "Bi(0) ≈ 0.6149, got {}", v);
    }

    #[test]
    fn fast_airy_ai_at_one() {
        // Ai(1) ≈ 0.1353
        let r = airy_ai(&float(1.0)).unwrap();
        let v = r.numeric_value.unwrap();
        assert!((v - 0.1353).abs() < TOL, "Ai(1) ≈ 0.1353, got {}", v);
    }

    #[test]
    fn fast_airy_bi_at_one() {
        // Bi(1) ≈ 1.2074
        let r = airy_bi(&float(1.0)).unwrap();
        let v = r.numeric_value.unwrap();
        assert!((v - 1.2074).abs() < TOL, "Bi(1) ≈ 1.2074, got {}", v);
    }

    #[test]
    fn fast_airy_ai_negative_arg() {
        // Ai(-1) ≈ 0.5356
        let r = airy_ai(&float(-1.0)).unwrap();
        let v = r.numeric_value.unwrap();
        assert!((v - 0.5356).abs() < TOL, "Ai(-1) ≈ 0.5356, got {}", v);
    }

    #[test]
    fn fast_airy_steps_non_empty() {
        let r = airy_ai(&float(1.0)).unwrap();
        assert!(!r.derivation_steps.is_empty());
        let r2 = airy_bi(&float(1.0)).unwrap();
        assert!(!r2.derivation_steps.is_empty());
    }

    #[test]
    fn fast_airy_wronskian() {
        // Wronskian: Ai(x)·Bi'(x) − Ai'(x)·Bi(x) = 1/π (not tested here),
        // but confirm Ai and Bi differ meaningfully at x=0.
        let ai0 = airy_ai(&int(0)).unwrap().numeric_value.unwrap();
        let bi0 = airy_bi(&int(0)).unwrap().numeric_value.unwrap();
        assert!(
            (bi0 / ai0 - 3.0_f64.sqrt()).abs() < 0.001,
            "Bi(0)/Ai(0) should equal √3, got {}",
            bi0 / ai0
        );
    }

    #[test]
    fn fast_airy_symbolic_arg_errors() {
        use crate::ast::Variable;
        let var = Expression::Variable(Variable {
            name: "x".to_string(),
            dimension: None,
        });
        assert!(airy_ai(&var).is_err());
        assert!(airy_bi(&var).is_err());
    }

    // Verify the relation Bi(x) = √3 (Ai(x) adjusted) at x=0 using ratio
    #[test]
    fn fast_airy_pi_normalization() {
        // Ai(0) · π^{1/3} · 3^{1/6} should be 1/(Γ(2/3)) — just check finite
        let ai0 = airy_ai(&int(0)).unwrap().numeric_value.unwrap();
        let bi0 = airy_bi(&int(0)).unwrap().numeric_value.unwrap();
        assert!(ai0.is_finite() && bi0.is_finite());
        // Cross-check: Ai(0)^2 + Bi(0)^2 / 3 ≈ constant (not standard formula,
        // just verify both are in the right ballpark).
        let product = ai0 * bi0;
        assert!(product > 0.0, "Ai(0) and Bi(0) should both be positive");
    }
}

// Ai(0) = c1, Bi(0) = √3 · c1 => Bi(0)/Ai(0) = √3  ✓
