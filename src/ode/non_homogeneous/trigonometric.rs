//! Particular solution for trigonometric forcing
//! `A·sin(k·x) + B·cos(k·x)`.
//!
//! Trial: `P·cos(k·x) + Q·sin(k·x)`, multiplied by `x` when `±ki` are
//! characteristic roots (which occurs iff `b = 0` and `c = a·k²`).

use std::collections::HashMap;
use std::sync::Arc;

use crate::numeric::evaluation::evaluate;
use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::{normalize, SymbolId};

use super::polynomial::build_x_power;
use super::{ODEError, SecondOrderODE};

// ---------------------------------------------------------------------------
// Particular solution: trigonometric forcing
// ---------------------------------------------------------------------------

/// Find `y_p` when `f(x) = A·sin(k·x) + B·cos(k·x)`.
///
/// Trial: `y_p = P·cos(k·x) + Q·sin(k·x)`.  If `±ki` are characteristic
/// roots (pure imaginary, which means `b=0` and `c/a = k²`), we multiply
/// by `x`.
pub(super) fn particular_trig(
    ode: &SecondOrderODE,
    k: f64,
    steps: &mut Vec<String>,
) -> Result<Arc<Expr>, ODEError> {
    let x_var = &ode.independent;
    let resonant = is_trig_resonant(ode, k);
    let multiplier = if resonant { 1_u32 } else { 0_u32 };

    steps.push(format!(
        "Trial form: y_p = x^{}·(P·cos({}·{}) + Q·sin({}·{}))",
        multiplier, k, x_var, k, x_var
    ));

    // Extract sin and cos amplitudes from forcing at two points
    let forcing_arc = ode.forcing_arc();
    let x_id = SymbolId::intern(x_var);
    let (f_sin, f_cos) = extract_trig_amplitudes(&forcing_arc, x_id, k)?;

    let (p, q) = solve_trig_system(ode, k, f_sin, f_cos, resonant)?;

    steps.push(format!(
        "Trig coefficients: P (cos) = {:.6}, Q (sin) = {:.6}",
        p, q
    ));

    Ok(build_trig_particular(p, q, k, multiplier, x_var))
}

/// Return `true` if `±ki` are characteristic roots of `a·r² + b·r + c = 0`.
///
/// This holds exactly when `b = 0` and `c = a·k²`.
fn is_trig_resonant(ode: &SecondOrderODE, k: f64) -> bool {
    const EPS: f64 = 1e-10;
    ode.b.abs() < EPS && (ode.c - ode.a * k * k).abs() < EPS
}

/// Extract sin-amplitude `f_s` and cos-amplitude `f_c` from `f(x)` such that
/// `f(x) ≈ f_s·sin(kx) + f_c·cos(kx)`.
///
/// Operates on the canonical `Arc<Expr>` form and samples through
/// [`crate::numeric::evaluation::evaluate`].
fn extract_trig_amplitudes(
    forcing: &Arc<Expr>,
    x: SymbolId,
    k: f64,
) -> Result<(f64, f64), ODEError> {
    let pi = std::f64::consts::PI;
    // Evaluate at x = π/(2k) and x = 0 to separate sin and cos components
    let x1 = if k.abs() > 1e-12 { pi / (2.0 * k) } else { 1.0 };
    let x0 = 0.0_f64;

    let eval_at = |xi: f64| -> f64 {
        let mut env = HashMap::new();
        env.insert(x, xi);
        evaluate(forcing, &env).unwrap_or(0.0)
    };

    let f_at_0 = eval_at(x0); // = f_c·cos(0) + f_s·sin(0) = f_c
    let f_at_x1 = eval_at(x1); // = f_c·cos(π/2) + f_s·sin(π/2) = f_s

    Ok((f_at_x1, f_at_0))
}

/// Solve the 2×2 linear system for trig undetermined coefficients.
fn solve_trig_system(
    ode: &SecondOrderODE,
    k: f64,
    f_sin: f64,
    f_cos: f64,
    resonant: bool,
) -> Result<(f64, f64), ODEError> {
    // Non-resonant: y_p = P·cos(kx) + Q·sin(kx)
    // Substitute and collect:
    //   cos terms: (c - a·k²)·P + b·k·Q = f_cos
    //   sin terms: -b·k·P + (c - a·k²)·Q = f_sin
    if !resonant {
        let alpha = ode.c - ode.a * k * k;
        let beta = ode.b * k;

        let det = alpha * alpha + beta * beta;
        if det.abs() < 1e-12 {
            return Err(ODEError::ResonanceDetected(
                "Resonance in trig system — try resonant trial".to_string(),
            ));
        }
        let p = (alpha * f_cos + beta * f_sin) / det;
        let q = (-beta * f_cos + alpha * f_sin) / det;
        Ok((p, q))
    } else {
        // Resonant: y_p = x·(P·cos(kx) + Q·sin(kx))
        // After substitution, the 2×2 system becomes:
        //   2a·k·Q + b·(P·1 + …)  — for pure b=0 case:
        //   cos terms: 2a·k·Q = f_cos  →  Q = f_cos / (2ak)
        //   sin terms: -2a·k·P = f_sin → P = -f_sin / (2ak)
        let denom = 2.0 * ode.a * k;
        if denom.abs() < 1e-12 {
            return Err(ODEError::ResonanceDetected(
                "Cannot determine trig coefficients — degenerate resonance".to_string(),
            ));
        }
        let q = f_cos / denom;
        let p = -f_sin / denom;
        Ok((p, q))
    }
}

/// Build `x^m·(P·cos(k·x) + Q·sin(k·x))` as an `Arc<Expr>`.
fn build_trig_particular(p: f64, q: f64, k: f64, multiplier: u32, x_var: &str) -> Arc<Expr> {
    // Build k·x
    let x = Expr::symbol(x_var);
    let k_arc = Arc::new(Expr::Float(k));
    let kx = normalize::mul(k_arc, x);

    // cos(k·x) and sin(k·x)
    let cos_term = Expr::func(FuncId::Cos, vec![kx.clone()]);
    let sin_term = Expr::func(FuncId::Sin, vec![kx]);

    let p_cos = if (p - 1.0).abs() < 1e-15 {
        cos_term
    } else {
        normalize::mul(Arc::new(Expr::Float(p)), cos_term)
    };

    let q_sin = if (q - 1.0).abs() < 1e-15 {
        sin_term
    } else {
        normalize::mul(Arc::new(Expr::Float(q)), sin_term)
    };

    let trig_sum = normalize::add(p_cos, q_sin);

    if multiplier == 0 {
        trig_sum
    } else {
        let x_pow = build_x_power(x_var, multiplier as i64);
        normalize::mul(x_pow, trig_sum)
    }
}
