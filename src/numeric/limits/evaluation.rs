//! Finite-point and infinity-point evaluation strategies for `limit`.
//!
//! The finite-point strategy is a four-step cascade:
//!
//! 1. Direct symbolic substitution — when the substituted expression reduces
//!    to a clean numeric literal and no nearby float probe blows up, that's
//!    the limit.
//! 2. Float probes — sample the expression at `point ± 1e-7` to detect
//!    signed infinities, one-sided finite limits, and tight two-sided
//!    agreement.
//! 3. L'Hôpital's rule for ratio forms.
//! 4. Taylor series leading-term analysis.
//!
//! Infinity limits transform `x → ±∞` into `t → 0⁺` via `x = ±1/t` and fall
//! back to the finite-point routine.

use std::sync::Arc;

use super::super::expr::{Expr, FuncId};
use super::super::substitute::substitute;
use super::super::{normalize, SymbolId};
use super::helpers::{expr_to_f64, float_to_exact};
use super::lhopital::try_lhopital;
use super::series::try_series_limit;
use super::{LimitPoint, LimitResult};

// ── Finite-point limits ───────────────────────────────────────────────────────

pub(crate) fn limit_at_finite(
    expr: &Arc<Expr>,
    var: SymbolId,
    point: &Arc<Expr>,
    original_point: &LimitPoint,
) -> LimitResult {
    // Step 1: Direct symbolic substitution for clean finite cases.
    // Note: symbolic substitution collapses 0^(-1) to 0 (the normalizer
    // absorbs singularities), so we must NOT use it as the sole detector.
    let subst = substitute(expr, var, point);
    if let Some(v) = expr_to_f64(&subst) {
        if !v.is_nan() && !v.is_infinite() {
            // Clean finite symbolic result — but verify via float probe to
            // rule out a collapsed singularity disguised as 0.
            if !float_probe_near_singularity(expr, var, point) {
                return LimitResult::Value(subst);
            }
        }
    }

    // Step 2: Float probe to detect signed infinities (e.g. 1/x near 0).
    // Symbolic substitution normalizes 0^(-1) → 0, so we use actual f64
    // arithmetic which preserves infinity.
    if let Some(f_point) = expr_to_f64(point) {
        let probe_r = eval_float(expr, var, f_point + 1e-7);
        let probe_l = eval_float(expr, var, f_point - 1e-7);

        match (probe_l, probe_r) {
            // For a one-sided query we only care about the relevant side —
            // the far side may blow up even when the approach side has a
            // clean finite limit (e.g. `exp(x)` as x → -∞: one side 0, one
            // side ∞).
            (_, Some(r))
                if matches!(original_point, LimitPoint::FromRight(_))
                    && r.is_finite()
                    && r.abs() < 1e6 =>
            {
                return LimitResult::Value(float_to_exact(r));
            }
            (Some(l), _)
                if matches!(original_point, LimitPoint::FromLeft(_))
                    && l.is_finite()
                    && l.abs() < 1e6 =>
            {
                return LimitResult::Value(float_to_exact(l));
            }
            (Some(l), Some(r))
                if l.is_infinite() || r.is_infinite() || l.abs() > 1e6 || r.abs() > 1e6 =>
            {
                return one_sided_limit_float(l, r, original_point);
            }
            (Some(l), Some(r)) if !l.is_nan() && !r.is_nan() => {
                // Both probes finite and equal within tolerance → clean limit.
                if (l - r).abs() < 1e-4 * (l.abs().max(1.0)) {
                    let avg = (l + r) / 2.0;
                    if !avg.is_nan() {
                        // Before trusting the float probe, try L'Hôpital: for
                        // 0/0 forms like `(1 − cos x)/x²` the probe suffers
                        // from catastrophic cancellation near the point, while
                        // L'Hôpital gives an exact result that evaluates
                        // numerically (e.g. `cos(0)/2 = 1/2`).
                        if let Some(lh) = try_lhopital(expr, var, point) {
                            if let LimitResult::Value(ref v) = lh {
                                if let Some(lh_f) = eval_f64(v) {
                                    // Trust L'Hôpital when its result agrees
                                    // with the probe's neighbourhood — it
                                    // provides more precision.
                                    if (lh_f - avg).abs() < 1e-2 * lh_f.abs().max(1.0) {
                                        return lh;
                                    }
                                }
                            }
                        }
                        // Prefer the symbolic substitution when it produced a
                        // clean numeric value; otherwise synthesize from the
                        // float average, promoting near-integer results to
                        // an exact integer so downstream assertions compare
                        // cleanly (`Value(Integer(2))` rather than `Value(Float(2.0))`).
                        if expr_to_f64(&subst).map_or(true, |v| v.is_nan()) {
                            return LimitResult::Value(float_to_exact(avg));
                        }
                        return LimitResult::Value(subst);
                    }
                }
            }
            _ => {}
        }
    }

    // Step 3: L'Hôpital on 0/0 or ∞/∞ rational forms.
    if let Some(result) = try_lhopital(expr, var, point) {
        return result;
    }

    // Step 4: Taylor series leading-term analysis.
    if let Some(result) = try_series_limit(expr, var, point) {
        return result;
    }

    LimitResult::Indeterminate
}

/// Returns true if float evaluation near `point` reveals a large spike
/// (magnitude > 1e6), indicating a collapsed singularity in symbolic form.
fn float_probe_near_singularity(expr: &Arc<Expr>, var: SymbolId, point: &Arc<Expr>) -> bool {
    if let Some(f_point) = expr_to_f64(point) {
        let p1 = eval_float(expr, var, f_point + 1e-6);
        let p2 = eval_float(expr, var, f_point - 1e-6);
        match (p1, p2) {
            (Some(a), Some(b)) => {
                a.abs() > 1e6 || b.abs() > 1e6 || a.is_infinite() || b.is_infinite()
            }
            _ => false,
        }
    } else {
        false
    }
}

/// Evaluate `expr` at a float value for `var` using f64 arithmetic via substitute.
pub(crate) fn eval_float(expr: &Arc<Expr>, var: SymbolId, val: f64) -> Option<f64> {
    let probe = Expr::float(val);
    let result = substitute(expr, var, &probe);
    eval_f64(&result)
}

/// Recursively evaluate an expression tree to f64, handling infinities properly.
/// Unlike `expr_to_f64`, this handles Pow/Func/Mul/Add nodes.
fn eval_f64(expr: &Arc<Expr>) -> Option<f64> {
    match expr.as_ref() {
        Expr::Integer(n) => n.to_i64().map(|v| v as f64),
        Expr::Rational(r) => Some(r.to_f64()),
        Expr::Float(f) => Some(*f),
        Expr::Pow(base, exp) => {
            let b = eval_f64(base)?;
            let e = eval_f64(exp)?;
            Some(b.powf(e))
        }
        Expr::Mul(node) => {
            let mut acc = node.coeff.to_f64();
            for (base, exp) in &node.factors {
                let b = eval_f64(base)?;
                let e = eval_f64(exp)?;
                acc *= b.powf(e);
            }
            Some(acc)
        }
        Expr::Add(node) => {
            let mut acc = node.constant.to_f64();
            for (term, coeff) in &node.terms {
                let t = eval_f64(term)?;
                acc += coeff.to_f64() * t;
            }
            Some(acc)
        }
        Expr::Func(id, args) => {
            if args.len() != 1 {
                return None;
            }
            let a = eval_f64(&args[0])?;
            Some(match id {
                FuncId::Sin => a.sin(),
                FuncId::Cos => a.cos(),
                FuncId::Tan => a.tan(),
                FuncId::Asin => a.asin(),
                FuncId::Acos => a.acos(),
                FuncId::Atan => a.atan(),
                FuncId::Sinh => a.sinh(),
                FuncId::Cosh => a.cosh(),
                FuncId::Tanh => a.tanh(),
                FuncId::Ln => a.ln(),
                FuncId::Exp => a.exp(),
                FuncId::Log2 => a.log2(),
                FuncId::Log10 => a.log10(),
                FuncId::Sqrt => a.sqrt(),
                FuncId::Cbrt => a.cbrt(),
                FuncId::Floor => a.floor(),
                FuncId::Ceil => a.ceil(),
                FuncId::Round => a.round(),
                FuncId::Abs => a.abs(),
                FuncId::Sign => a.signum(),
                // multi-arg functions not handled by the single-arg path above
                FuncId::Atan2 | FuncId::Log | FuncId::Min | FuncId::Max => return None,
                FuncId::Re | FuncId::Im | FuncId::Conj => return None,
                // Special functions: not evaluable in this f64 path
                FuncId::Gamma
                | FuncId::LnGamma
                | FuncId::Digamma
                | FuncId::BetaFn
                | FuncId::Erf
                | FuncId::Erfc
                | FuncId::BesselJ
                | FuncId::BesselY
                | FuncId::BesselI
                | FuncId::BesselK
                | FuncId::AiryAi
                | FuncId::AiryBi
                | FuncId::Zeta
                | FuncId::Si
                | FuncId::Ci
                | FuncId::Ei
                | FuncId::Heaviside
                | FuncId::DiracDelta => return None,
                FuncId::Other(_) => return None,
            })
        }
        _ => None,
    }
}

fn one_sided_limit_float(left: f64, right: f64, original_point: &LimitPoint) -> LimitResult {
    let left_sign = if left > 0.0 {
        1i32
    } else if left < 0.0 {
        -1
    } else {
        0
    };
    let right_sign = if right > 0.0 {
        1i32
    } else if right < 0.0 {
        -1
    } else {
        0
    };

    match original_point {
        LimitPoint::FromRight(_) => sign_to_result(right_sign),
        LimitPoint::FromLeft(_) => sign_to_result(left_sign),
        LimitPoint::Value(_) => {
            if left_sign == right_sign {
                sign_to_result(left_sign)
            } else {
                LimitResult::DoesNotExist
            }
        }
        _ => LimitResult::Indeterminate,
    }
}

fn sign_to_result(sign: i32) -> LimitResult {
    match sign {
        1 => LimitResult::PosInfinity,
        -1 => LimitResult::NegInfinity,
        _ => LimitResult::Indeterminate,
    }
}

// ── Infinity limits ───────────────────────────────────────────────────────────

pub(crate) fn limit_at_infinity(expr: &Arc<Expr>, var: SymbolId, negative: bool) -> LimitResult {
    // Substitute x = 1/t and take the limit as t → 0⁺.
    // For x → -∞: also negate, i.e. x = -(1/t).
    let t_id = SymbolId::intern("__lim_t__");
    let t = Expr::symbol("__lim_t__");

    let one_over_t = normalize::div(Expr::int(1), t.clone());
    let subst_val = if negative {
        normalize::neg(one_over_t)
    } else {
        one_over_t
    };

    let transformed = substitute(expr, var, &subst_val);
    limit_at_finite(
        &transformed,
        t_id,
        &Expr::int(0),
        &LimitPoint::FromRight(Expr::int(0)),
    )
}
