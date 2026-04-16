//! Limit computation for symbolic expressions.
//!
//! Implements limit evaluation using three strategies in order:
//!
//! 1. **Direct substitution** — substitute the point value and check for a
//!    well-defined numeric result.
//! 2. **L'Hôpital's rule** — for 0/0 or ∞/∞ indeterminate forms on `Div`
//!    expressions, differentiate numerator and denominator repeatedly (up to
//!    [`LHOPITAL_MAX_ITER`] times).
//! 3. **Taylor series** — expand the expression around the limit point, find
//!    the leading non-zero coefficient, and return its value.
//!
//! Limits at ±∞ are handled via the substitution `x = 1/t`, `t → 0⁺`.

use std::sync::Arc;

use num::traits::{One, Zero};

use super::{
    differentiation::diff_arc,
    expr::{Expr, FuncId},
    normalize,
    series::taylor::substitute,
    SymbolId,
};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum number of L'Hôpital iterations before giving up.
const LHOPITAL_MAX_ITER: usize = 5;

/// Taylor series order used when falling back to series analysis.
const SERIES_ORDER: usize = 6;

// ── Public types ──────────────────────────────────────────────────────────────

/// The point at which a limit is evaluated.
#[derive(Clone, Debug)]
pub enum LimitPoint {
    /// Finite point: `x → a`.
    Value(Arc<Expr>),
    /// `x → +∞`.
    PosInfinity,
    /// `x → -∞`.
    NegInfinity,
    /// Left-hand limit: `x → a⁻`.
    FromLeft(Arc<Expr>),
    /// Right-hand limit: `x → a⁺`.
    FromRight(Arc<Expr>),
}

/// The outcome of a limit computation.
#[derive(Clone, Debug, PartialEq)]
pub enum LimitResult {
    /// The limit exists and equals a finite expression.
    Value(Arc<Expr>),
    /// The limit is +∞.
    PosInfinity,
    /// The limit is -∞.
    NegInfinity,
    /// The limit does not exist (e.g. left ≠ right, oscillates).
    DoesNotExist,
    /// The algorithm could not determine the limit.
    Indeterminate,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute `lim_{x → point} expr`.
///
/// # Arguments
///
/// * `expr`  — expression to take the limit of.
/// * `var`   — the variable that approaches the limit point.
/// * `point` — the value the variable approaches.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{Expr, FuncId, SymbolId};
/// use thales::numeric::limits::{limit, LimitPoint, LimitResult};
///
/// // lim_{x→0} sin(x)/x = 1
/// let x_id = SymbolId::intern("lim_sinc_x");
/// let x = Expr::symbol("lim_sinc_x");
/// let sin_x = Expr::func(FuncId::Sin, vec![x.clone()]);
/// let expr = thales::numeric::normalize::div(sin_x, x);
/// let result = limit(&expr, x_id, &LimitPoint::Value(Expr::int(0)));
/// assert_eq!(result, LimitResult::Value(Expr::int(1)));
/// ```
pub fn limit(expr: &Arc<Expr>, var: SymbolId, point: &LimitPoint) -> LimitResult {
    match point {
        LimitPoint::Value(a) | LimitPoint::FromLeft(a) | LimitPoint::FromRight(a) => {
            limit_at_finite(expr, var, a, point)
        }
        LimitPoint::PosInfinity => limit_at_infinity(expr, var, false),
        LimitPoint::NegInfinity => limit_at_infinity(expr, var, true),
    }
}

// ── Finite-point limits ───────────────────────────────────────────────────────

fn limit_at_finite(
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
            (Some(l), Some(r)) if l.is_infinite() || r.is_infinite() => {
                return one_sided_limit_float(l, r, original_point);
            }
            (Some(l), Some(r)) if !l.is_nan() && !r.is_nan() => {
                // Both probes finite and equal within tolerance → clean limit.
                if (l - r).abs() < 1e-4 * (l.abs().max(1.0)) {
                    let avg = (l + r) / 2.0;
                    if !avg.is_nan() {
                        // Return as float only if symbolic failed.
                        if expr_to_f64(&subst).map_or(true, |v| v.is_nan()) {
                            return LimitResult::Value(Expr::float(avg));
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
fn eval_float(expr: &Arc<Expr>, var: SymbolId, val: f64) -> Option<f64> {
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

// ── Infinity limits ───────────────────────────────────────────────────────────

fn limit_at_infinity(expr: &Arc<Expr>, var: SymbolId, negative: bool) -> LimitResult {
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

// ── One-sided limits for division by zero ────────────────────────────────────

fn one_sided_limit(
    expr: &Arc<Expr>,
    var: SymbolId,
    point: &Arc<Expr>,
    original_point: &LimitPoint,
) -> LimitResult {
    // Probe the sign of the expression slightly to the left and right.
    let sign_right = probe_sign(expr, var, point, true);
    let sign_left = probe_sign(expr, var, point, false);

    match original_point {
        LimitPoint::FromRight(_) => sign_to_result(sign_right),
        LimitPoint::FromLeft(_) => sign_to_result(sign_left),
        LimitPoint::Value(_) => {
            // Two-sided: left and right must agree.
            if sign_right == sign_left {
                sign_to_result(sign_right)
            } else {
                LimitResult::DoesNotExist
            }
        }
        _ => LimitResult::Indeterminate,
    }
}

/// Probe the sign of `expr` at `point ± epsilon`.
fn probe_sign(expr: &Arc<Expr>, var: SymbolId, point: &Arc<Expr>, from_right: bool) -> i32 {
    // Use a small offset in float domain.
    let base_f = expr_to_f64(point);
    if base_f.is_none() {
        return 0;
    }
    let base = base_f.unwrap();
    let probe = if from_right { base + 1e-7 } else { base - 1e-7 };
    let probe_expr = Expr::float(probe);
    let val = substitute(expr, var, &probe_expr);
    match expr_to_f64(&val) {
        Some(v) if v > 0.0 => 1,
        Some(v) if v < 0.0 => -1,
        _ => 0,
    }
}

fn sign_to_result(sign: i32) -> LimitResult {
    match sign {
        1 => LimitResult::PosInfinity,
        -1 => LimitResult::NegInfinity,
        _ => LimitResult::Indeterminate,
    }
}

// ── L'Hôpital's rule ─────────────────────────────────────────────────────────

fn try_lhopital(expr: &Arc<Expr>, var: SymbolId, point: &Arc<Expr>) -> Option<LimitResult> {
    let f_point = expr_to_f64(point)?;
    let (mut num, mut den) = extract_ratio(expr)?;

    for _ in 0..LHOPITAL_MAX_ITER {
        // Use float evaluation to detect indeterminate forms reliably.
        // Symbolic substitution normalizes 0^(-1) → 0, hiding singularities.
        let num_f = eval_float(&num, var, f_point);
        let den_f = eval_float(&den, var, f_point);

        let is_zero_over_zero = matches!(
            (num_f, den_f),
            (Some(n), Some(d)) if n.abs() < 1e-10 && d.abs() < 1e-10
        );
        let is_inf_over_inf = matches!(
            (num_f, den_f),
            (Some(n), Some(d)) if (n.is_infinite() || n.abs() > 1e10)
                                && (d.is_infinite() || d.abs() > 1e10)
        );

        if !is_zero_over_zero && !is_inf_over_inf {
            // Not an indeterminate form — evaluate directly if possible.
            if let (Some(n), Some(d)) = (num_f, den_f) {
                if d.abs() > 1e-10 && !d.is_nan() {
                    // Confirmed finite ratio: return symbolic or float value.
                    let sym_num = substitute(&num, var, point);
                    let sym_den = substitute(&den, var, point);
                    let result = normalize::div(sym_num, sym_den);
                    return Some(LimitResult::Value(result));
                }
            }
            break;
        }

        // Apply L'Hôpital: differentiate numerator and denominator.
        num = diff_arc(&num, var);
        den = diff_arc(&den, var);

        // Check the new ratio after differentiation.
        let new_num_f = eval_float(&num, var, f_point);
        let new_den_f = eval_float(&den, var, f_point);

        match (new_num_f, new_den_f) {
            (Some(n), Some(d)) if !d.is_nan() && d.abs() > 1e-10 => {
                // Non-zero denominator: compute symbolic result.
                let sym_num = substitute(&num, var, point);
                let sym_den = substitute(&den, var, point);
                let result = normalize::div(sym_num, sym_den);
                return Some(LimitResult::Value(result));
            }
            (Some(n), Some(d)) if n.abs() < 1e-10 && d.abs() < 1e-10 => {
                // Still 0/0 — iterate again.
                continue;
            }
            (Some(n), Some(d)) if (n.is_infinite() || n.abs() > 1e10) && d.abs() < 1e-6 => {
                return Some(if n > 0.0 {
                    LimitResult::PosInfinity
                } else {
                    LimitResult::NegInfinity
                });
            }
            (Some(n), Some(d)) if n.abs() < 1e-6 && (d.is_infinite() || d.abs() > 1e10) => {
                return Some(LimitResult::Value(Expr::int(0)));
            }
            _ => break,
        }
    }
    None
}

// ── Taylor series analysis ────────────────────────────────────────────────────

fn try_series_limit(expr: &Arc<Expr>, var: SymbolId, point: &Arc<Expr>) -> Option<LimitResult> {
    use super::series::taylor::taylor;

    let ts = taylor(expr, var, point, SERIES_ORDER);

    // Find the leading non-zero coefficient.
    for n in 0..=ts.order {
        let coeff = ts.coeff(n);
        if !coeff.is_zero() {
            if n == 0 {
                // Constant term is the limit.
                return Some(LimitResult::Value(coeff));
            } else {
                // Leading term is (x - center)^n * coeff.
                // As x → center, this → 0 for any finite non-zero coeff.
                // (The limit of x^n * coeff as x→0 is 0 for n≥1.)
                return Some(LimitResult::Value(Expr::int(0)));
            }
        }
    }

    // All coefficients are zero: limit is 0.
    Some(LimitResult::Value(Expr::int(0)))
}

// ── Helper: extract numerator and denominator ─────────────────────────────────

/// If `expr` looks like a ratio (Mul with a factor raised to -1), extract
/// the numerator and denominator for L'Hôpital.
fn extract_ratio(expr: &Arc<Expr>) -> Option<(Arc<Expr>, Arc<Expr>)> {
    match expr.as_ref() {
        Expr::Mul(node) => {
            // Look for a factor with exponent -1 (i.e. a denominator).
            let mut num_factors: Vec<(Arc<Expr>, Arc<Expr>)> = Vec::new();
            let mut den_factors: Vec<(Arc<Expr>, Arc<Expr>)> = Vec::new();

            for (base, exp) in &node.factors {
                if is_negative_one(exp) {
                    den_factors.push((base.clone(), Expr::int(1)));
                } else if let Expr::Integer(n) = exp.as_ref() {
                    if let Some(v) = n.to_i64() {
                        if v < 0 {
                            den_factors.push((base.clone(), Expr::int(-v)));
                        } else {
                            num_factors.push((base.clone(), exp.clone()));
                        }
                    } else {
                        num_factors.push((base.clone(), exp.clone()));
                    }
                } else {
                    num_factors.push((base.clone(), exp.clone()));
                }
            }

            if den_factors.is_empty() {
                return None;
            }

            let num = build_product(&node.coeff, &num_factors);
            let den = build_product(&super::BigRational::from_i64(1, 1), &den_factors);
            Some((num, den))
        }
        Expr::Pow(base, exp) if is_negative_one(exp) => Some((Expr::int(1), base.clone())),
        _ => None,
    }
}

fn is_negative_one(expr: &Arc<Expr>) -> bool {
    match expr.as_ref() {
        Expr::Integer(n) => n.to_i64().map(|v| v == -1).unwrap_or(false),
        Expr::Rational(r) => {
            use num::traits::One;
            r.numer().to_i64().map(|v| v == -1).unwrap_or(false) && r.denom().is_one()
        }
        _ => false,
    }
}

fn build_product(coeff: &super::BigRational, factors: &[(Arc<Expr>, Arc<Expr>)]) -> Arc<Expr> {
    use num::traits::One;
    let coeff_expr: Arc<Expr> = {
        if coeff.is_one() {
            Expr::int(1)
        } else if coeff.denom().is_one() {
            if let Some(n) = coeff.numer().to_i64() {
                Expr::int(n)
            } else {
                Arc::new(Expr::Rational(coeff.clone()))
            }
        } else {
            Arc::new(Expr::Rational(coeff.clone()))
        }
    };

    let mut acc = coeff_expr;
    for (base, exp) in factors {
        let term = normalize::pow(base.clone(), exp.clone());
        acc = normalize::mul(acc, term);
    }
    acc
}

// ── Expression classification ─────────────────────────────────────────────────

#[derive(Debug, PartialEq)]
enum ExprClass {
    /// A well-defined finite value.
    Finite,
    /// Contains division by zero (expression is NaN or ±Inf float).
    DivByZero,
    /// Positive infinity.
    PosInf,
    /// Negative infinity.
    NegInf,
    /// 0/0 indeterminate form (both numerator and denominator vanish).
    Indeterminate,
}

fn classify_expr(expr: &Arc<Expr>) -> ExprClass {
    match expr.as_ref() {
        Expr::Integer(_) | Expr::Rational(_) => ExprClass::Finite,
        Expr::Float(f) => {
            if f.is_nan() {
                ExprClass::Indeterminate
            } else if f.is_infinite() {
                if *f > 0.0 {
                    ExprClass::PosInf
                } else {
                    ExprClass::NegInf
                }
            } else {
                ExprClass::Finite
            }
        }
        Expr::Mul(node) => {
            // Check if the coefficient is infinite (from propagated f64 ops).
            if let Some(v) = rational_to_f64_opt(&node.coeff) {
                if v.is_nan() {
                    return ExprClass::Indeterminate;
                }
                if v.is_infinite() {
                    return if v > 0.0 {
                        ExprClass::PosInf
                    } else {
                        ExprClass::NegInf
                    };
                }
            }
            // If any factor evaluates to a numeric infinity, propagate.
            for (base, exp) in &node.factors {
                let b_cls = classify_expr(base);
                let e_cls = classify_expr(exp);
                if matches!(b_cls, ExprClass::DivByZero | ExprClass::Indeterminate)
                    || matches!(e_cls, ExprClass::DivByZero | ExprClass::Indeterminate)
                {
                    return ExprClass::Indeterminate;
                }
            }
            ExprClass::Finite
        }
        // A Pow with a zero base and negative exponent = division by zero.
        Expr::Pow(base, exp) => {
            if base.is_zero() {
                if let Some(e) = expr_to_f64(exp) {
                    if e < 0.0 {
                        return ExprClass::DivByZero;
                    }
                }
            }
            let bc = classify_expr(base);
            let ec = classify_expr(exp);
            if matches!(bc, ExprClass::Indeterminate) || matches!(ec, ExprClass::Indeterminate) {
                return ExprClass::Indeterminate;
            }
            ExprClass::Finite
        }
        Expr::Func(id, args) => classify_func(*id, args),
        _ => ExprClass::Finite,
    }
}

fn classify_func(id: FuncId, args: &[Arc<Expr>]) -> ExprClass {
    if args.len() == 1 {
        let arg_cls = classify_expr(&args[0]);
        if matches!(arg_cls, ExprClass::Indeterminate | ExprClass::DivByZero) {
            return ExprClass::Indeterminate;
        }
        // ln(0) = -∞
        if matches!(id, FuncId::Ln) && args[0].is_zero() {
            return ExprClass::NegInf;
        }
        // exp at a finite numeric arg.
        if matches!(id, FuncId::Exp) {
            if let Some(v) = expr_to_f64(&args[0]) {
                let r = v.exp();
                if r.is_infinite() {
                    return if r > 0.0 {
                        ExprClass::PosInf
                    } else {
                        ExprClass::NegInf
                    };
                }
                if r.is_nan() {
                    return ExprClass::Indeterminate;
                }
                return ExprClass::Finite;
            }
        }
    }
    ExprClass::Finite
}

// ── Numeric helpers ───────────────────────────────────────────────────────────

fn expr_to_f64(expr: &Arc<Expr>) -> Option<f64> {
    match expr.as_ref() {
        Expr::Integer(n) => n.to_i64().map(|v| v as f64),
        Expr::Rational(r) => Some(r.to_f64()),
        Expr::Float(f) => Some(*f),
        _ => None,
    }
}

fn rational_to_f64_opt(r: &super::BigRational) -> Option<f64> {
    Some(r.to_f64())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{normalize, Expr, FuncId, SymbolId};

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }

    fn lim(expr: &Arc<Expr>, name: &str, point: LimitPoint) -> LimitResult {
        let id = SymbolId::intern(name);
        limit(expr, id, &point)
    }

    // ── Direct substitution ───────────────────────────────────────────────

    #[test]
    fn test_limit_polynomial_finite() {
        // lim_{x→3} x^2 = 9
        let x = sym("lim_poly_x");
        let expr = normalize::pow(x, Expr::int(2));
        let result = lim(&expr, "lim_poly_x", LimitPoint::Value(Expr::int(3)));
        assert_eq!(result, LimitResult::Value(Expr::int(9)));
    }

    #[test]
    fn test_limit_constant() {
        // lim_{x→5} 7 = 7
        let expr = Expr::int(7);
        let result = lim(&expr, "lim_const_x", LimitPoint::Value(Expr::int(5)));
        assert_eq!(result, LimitResult::Value(Expr::int(7)));
    }

    // ── L'Hôpital 0/0 ────────────────────────────────────────────────────

    #[test]
    fn test_limit_lhopital_polynomial_ratio() {
        // lim_{x→1} (x^2 - 1) / (x - 1) = 2
        let x = sym("lim_lh_x");
        let x2_minus_1 = normalize::sub(normalize::pow(x.clone(), Expr::int(2)), Expr::int(1));
        let x_minus_1 = normalize::sub(x, Expr::int(1));
        let expr = normalize::div(x2_minus_1, x_minus_1);
        let result = lim(&expr, "lim_lh_x", LimitPoint::Value(Expr::int(1)));
        assert_eq!(result, LimitResult::Value(Expr::int(2)));
    }

    // ── Sinc limit via L'Hôpital or series ───────────────────────────────

    #[test]
    fn test_limit_sinc() {
        // lim_{x→0} sin(x)/x = 1
        let x = sym("lim_sinc_x");
        let sin_x = Expr::func(FuncId::Sin, vec![x.clone()]);
        let expr = normalize::div(sin_x, x);
        let result = lim(&expr, "lim_sinc_x", LimitPoint::Value(Expr::int(0)));
        assert_eq!(result, LimitResult::Value(Expr::int(1)));
    }

    // ── One-sided limits ──────────────────────────────────────────────────

    #[test]
    fn test_limit_one_over_x_right() {
        // lim_{x→0⁺} 1/x = +∞
        let x = sym("lim_1x_r");
        let expr = normalize::div(Expr::int(1), x);
        let result = lim(&expr, "lim_1x_r", LimitPoint::FromRight(Expr::int(0)));
        assert_eq!(result, LimitResult::PosInfinity);
    }

    #[test]
    fn test_limit_one_over_x_left() {
        // lim_{x→0⁻} 1/x = -∞
        let x = sym("lim_1x_l");
        let expr = normalize::div(Expr::int(1), x);
        let result = lim(&expr, "lim_1x_l", LimitPoint::FromLeft(Expr::int(0)));
        assert_eq!(result, LimitResult::NegInfinity);
    }

    #[test]
    fn test_limit_one_over_x_two_sided() {
        // lim_{x→0} 1/x does not exist (left ≠ right)
        let x = sym("lim_1x_ts");
        let expr = normalize::div(Expr::int(1), x);
        let result = lim(&expr, "lim_1x_ts", LimitPoint::Value(Expr::int(0)));
        assert_eq!(result, LimitResult::DoesNotExist);
    }

    // ── Infinity limits ───────────────────────────────────────────────────

    #[test]
    fn test_limit_exp_neg_inf() {
        // lim_{x→-∞} exp(x) = 0
        let x = sym("lim_exp_ni");
        let expr = Expr::func(FuncId::Exp, vec![x]);
        let result = lim(&expr, "lim_exp_ni", LimitPoint::NegInfinity);
        assert_eq!(result, LimitResult::Value(Expr::int(0)));
    }

    #[test]
    fn test_limit_one_over_x_at_pos_inf() {
        // lim_{x→+∞} 1/x = 0
        let x = sym("lim_1x_inf");
        let expr = normalize::div(Expr::int(1), x);
        let result = lim(&expr, "lim_1x_inf", LimitPoint::PosInfinity);
        assert_eq!(result, LimitResult::Value(Expr::int(0)));
    }
}
