//! Laurent series expansion for `Arc<Expr>`.
//!
//! Factors any `(x − center)^s` prefix from `expr` using structural inspection
//! of `Mul`/`Pow`, then Taylor-expands the analytic remainder `r(x)` around
//! `center` and re-indexes the coefficients by the signed shift `s`:
//!
//! `expr(x) = (x − center)^s · r(x)   ⇒   a_n = b_{n − s}`,
//!
//! where `a_n` is the coefficient of `(x − center)^n` in the Laurent series
//! and `b_m` is the Taylor coefficient of `r` at order `m`.
//!
//! # Trace
//!
//! Records one [`TechniqueTag::LaurentExpansion`] step per call when a
//! `&mut Trace` is supplied.
//!
//! # Scope
//!
//! - Handles `(x − c)^s · r(x)` where `s` is a structurally detectable integer
//!   and `r(x)` is regular at `c`. Covers `1/x^n`, `1/(x − c)^n`,
//!   `(x − c)^k · f(x)`, and bare analytic expressions (fallback to Taylor).
//! - Rejects shifts whose absolute value exceeds [`MAX_LAURENT_SHIFT`] to keep
//!   Taylor orders bounded.
//! - Does not invert arbitrary denominators (e.g. `1/sin(x)` at `0`); those
//!   require full Laurent division and are out of scope.

use std::sync::Arc;

use num::traits::{One, Zero};

use super::super::{
    expr::Expr,
    normalize,
    trace::{record, Step, TechniqueTag, Trace},
    BigRational, SymbolId,
};
use super::{taylor::taylor, LaurentSeries};

/// Largest absolute signed shift the structural extractor accepts before
/// the engine bails out. Caps the Taylor order that the remainder expansion
/// may trigger.
pub const MAX_LAURENT_SHIFT: i32 = 20;

// ── Public API ───────────────────────────────────────────────────────────────

/// Compute a truncated Laurent expansion of `expr` around `center` covering
/// powers `[-neg_order, pos_order]` of `(x − center)`.
///
/// Returns `None` when the structural analysis detects a shift whose absolute
/// value exceeds [`MAX_LAURENT_SHIFT`], or when the residual cannot be
/// expanded structurally. Pure analytic inputs succeed via the Taylor
/// fallback.
pub fn laurent_expand(
    expr: &Arc<Expr>,
    var: SymbolId,
    center: &Arc<Expr>,
    neg_order: u32,
    pos_order: u32,
    mut trace: Option<&mut Trace>,
) -> Option<LaurentSeries> {
    record(
        trace.as_deref_mut(),
        Step::new(
            TechniqueTag::LaurentExpansion,
            format!("at {center}, range [-{neg_order}, {pos_order}]"),
        )
        .with_input(expr.clone()),
    );

    let (shift, remainder) = extract_shift(expr, var, center);

    if shift.abs() > MAX_LAURENT_SHIFT {
        return None;
    }

    let neg_span = neg_order as i32;
    let pos_span = pos_order as i32;
    let len = (neg_span + pos_span + 1) as usize;

    // Maximum Taylor index needed: m = n − shift for the largest n = pos_span.
    let max_m = pos_span.saturating_sub(shift);
    if max_m < 0 {
        // All requested powers lie strictly below the shift — series is zero
        // over the requested window.
        let coefficients = vec![Expr::int(0); len];
        return Some(LaurentSeries::from_coefficients(
            center.clone(),
            var,
            coefficients,
            -neg_span,
        ));
    }

    let taylor_order = max_m as usize;
    let ts = taylor(&remainder, var, center, taylor_order);

    let mut coefficients: Vec<Arc<Expr>> = Vec::with_capacity(len);
    for i in 0..len {
        let n = -neg_span + i as i32;
        let m = n - shift;
        if m < 0 {
            coefficients.push(Expr::int(0));
        } else {
            coefficients.push(ts.coeff(m as usize));
        }
    }

    Some(LaurentSeries::from_coefficients(
        center.clone(),
        var,
        coefficients,
        -neg_span,
    ))
}

// ── Shift extraction ─────────────────────────────────────────────────────────

/// Factor `(x − center)^s` out of `expr` structurally, returning `(s,
/// remainder)` with `expr == (x − center)^s · remainder`. When no matching
/// factor is present, returns `(0, expr)`.
fn extract_shift(expr: &Arc<Expr>, var: SymbolId, center: &Arc<Expr>) -> (i32, Arc<Expr>) {
    if let Some(s) = match_shift_power(expr, var, center) {
        return (s, Expr::int(1));
    }

    if let Expr::Mul(node) = expr.as_ref() {
        let mut shift = 0i32;
        let mut keep: Vec<(Arc<Expr>, Arc<Expr>)> = Vec::new();
        for (base, exp) in &node.factors {
            if let Some(delta) = factor_shift_delta(base, exp, var, center) {
                shift = shift.saturating_add(delta);
            } else {
                keep.push((base.clone(), exp.clone()));
            }
        }
        if shift != 0 {
            let remainder = rebuild_mul(node.coeff.clone(), keep);
            return (shift, remainder);
        }
    }

    (0, expr.clone())
}

/// Whole-expression match against `(x − center)^n` (or the bare base shape).
fn match_shift_power(expr: &Arc<Expr>, var: SymbolId, center: &Arc<Expr>) -> Option<i32> {
    if matches_shift_base(expr, var, center) {
        return Some(1);
    }
    if let Expr::Pow(base, exp) = expr.as_ref() {
        if matches_shift_base(base, var, center) {
            return exp_as_int(exp);
        }
    }
    None
}

/// Signed exponent contributed by a MulNode factor `base^exp` when it
/// represents a power of `(x − center)`. Handles both canonical shapes:
///
/// - `(base = Symbol(x) or (x−c), exp = Integer n)`
/// - `(base = Pow(inner, k), exp = Integer n)` where `inner` matches.
fn factor_shift_delta(
    base: &Arc<Expr>,
    exp: &Arc<Expr>,
    var: SymbolId,
    center: &Arc<Expr>,
) -> Option<i32> {
    if matches_shift_base(base, var, center) {
        return exp_as_int(exp);
    }
    if let Expr::Pow(inner, inner_exp) = base.as_ref() {
        if matches_shift_base(inner, var, center) {
            let ie = exp_as_int(inner_exp)?;
            let oe = exp_as_int(exp)?;
            return Some(ie.saturating_mul(oe));
        }
    }
    None
}

/// True when `base` is structurally equal to `(x − center)` (or plain `x`
/// when `center` is zero).
fn matches_shift_base(base: &Arc<Expr>, var: SymbolId, center: &Arc<Expr>) -> bool {
    if center.is_zero() {
        return matches!(base.as_ref(), Expr::Symbol(s) if *s == var);
    }
    let Expr::Add(node) = base.as_ref() else {
        return false;
    };
    if node.terms.len() != 1 {
        return false;
    }
    let Some((term, coeff)) = node.terms.iter().next() else {
        return false;
    };
    if !coeff.is_one() {
        return false;
    }
    let Expr::Symbol(s) = term.as_ref() else {
        return false;
    };
    if *s != var {
        return false;
    }
    let Some(center_val) = rational_of(center) else {
        return false;
    };
    let neg_const = -&node.constant;
    neg_const == center_val
}

fn exp_as_int(exp: &Arc<Expr>) -> Option<i32> {
    match exp.as_ref() {
        Expr::Integer(n) => n.to_i64().and_then(|v| i32::try_from(v).ok()),
        _ => None,
    }
}

fn rational_of(expr: &Arc<Expr>) -> Option<BigRational> {
    match expr.as_ref() {
        Expr::Integer(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

fn rebuild_mul(coeff: BigRational, factors: Vec<(Arc<Expr>, Arc<Expr>)>) -> Arc<Expr> {
    let mut acc = rational_to_expr(coeff);
    for (base, exp) in factors {
        let factor = normalize::pow(base, exp);
        acc = normalize::mul(acc, factor);
    }
    acc
}

fn rational_to_expr(r: BigRational) -> Arc<Expr> {
    if r.is_zero() {
        return Expr::int(0);
    }
    if r.denom().is_one() {
        if let Some(n) = r.numer().to_i64() {
            return Expr::int(n);
        }
    }
    Arc::new(Expr::Rational(r))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::FuncId;

    fn sym(name: &str) -> (SymbolId, Arc<Expr>) {
        (SymbolId::intern(name), Expr::symbol(name))
    }

    #[test]
    fn bare_analytic_is_taylor() {
        // exp(x) Laurent at 0 with neg=0 is Taylor.
        let (x_id, x) = sym("laur_exp");
        let expr = Expr::func(FuncId::Exp, vec![x]);
        let ls = laurent_expand(&expr, x_id, &Expr::int(0), 0, 3, None).expect("laurent");
        assert_eq!(ls.leading_power, 0);
        assert!(ls.coeff(0).is_one());
        assert!(ls.coeff(1).is_one());
    }

    #[test]
    fn one_over_x_at_zero() {
        // 1/x = x^(-1). a_{-1}=1, rest zero.
        let (x_id, x) = sym("laur_1x");
        let expr = normalize::pow(x, Expr::int(-1));
        let ls = laurent_expand(&expr, x_id, &Expr::int(0), 2, 2, None).expect("laurent");
        assert_eq!(ls.leading_power, -2);
        assert!(ls.coeff(-2).is_zero());
        assert!(ls.coeff(-1).is_one());
        assert!(ls.coeff(0).is_zero());
        assert!(ls.coeff(1).is_zero());
        assert!(ls.coeff(2).is_zero());
    }

    #[test]
    fn one_over_x_squared() {
        let (x_id, x) = sym("laur_1x2");
        let expr = normalize::pow(x, Expr::int(-2));
        let ls = laurent_expand(&expr, x_id, &Expr::int(0), 3, 1, None).expect("laurent");
        assert!(ls.coeff(-2).is_one());
        assert!(ls.coeff(-1).is_zero());
        assert!(ls.coeff(-3).is_zero());
    }

    #[test]
    fn shifted_simple_pole() {
        // 1/(x - 2) at x=2: a_{-1}=1.
        let (x_id, x) = sym("laur_shift");
        let shifted = normalize::sub(x, Expr::int(2));
        let expr = normalize::pow(shifted, Expr::int(-1));
        let ls = laurent_expand(&expr, x_id, &Expr::int(2), 1, 2, None).expect("laurent");
        assert!(ls.coeff(-1).is_one());
        assert!(ls.coeff(0).is_zero());
    }

    #[test]
    fn exp_over_x_at_zero() {
        // exp(x)/x = x^{-1} + 1 + x/2 + x^2/6 + ...
        let (x_id, x) = sym("laur_exp_x");
        let ex = Expr::func(FuncId::Exp, vec![x.clone()]);
        let inv_x = normalize::pow(x, Expr::int(-1));
        let expr = normalize::mul(ex, inv_x);
        let ls = laurent_expand(&expr, x_id, &Expr::int(0), 1, 3, None).expect("laurent");
        assert!(ls.coeff(-1).is_one());
        assert!(ls.coeff(0).is_one());
        // a_1 = 1/2
        let a1 = ls.coeff(1);
        match a1.as_ref() {
            Expr::Rational(r) => assert_eq!(r.to_f64(), 0.5),
            other => panic!("expected 1/2, got {other}"),
        }
    }

    #[test]
    fn shift_above_pos_order_yields_zero_window() {
        // f = x^5 with request window [-1, 2]: a_n = 0 for n ≤ 2 < 5.
        let (x_id, x) = sym("laur_high");
        let expr = normalize::pow(x, Expr::int(5));
        let ls = laurent_expand(&expr, x_id, &Expr::int(0), 1, 2, None).expect("laurent");
        for n in -1..=2 {
            assert!(ls.coeff(n).is_zero(), "a_{n} should be zero");
        }
    }

    #[test]
    fn records_trace_step() {
        let (x_id, x) = sym("laur_trace");
        let expr = normalize::pow(x, Expr::int(-1));
        let mut trace = Trace::new();
        let _ = laurent_expand(&expr, x_id, &Expr::int(0), 1, 1, Some(&mut trace));
        assert_eq!(trace.steps()[0].tag, TechniqueTag::LaurentExpansion);
    }

    #[test]
    fn excessive_shift_returns_none() {
        let (x_id, x) = sym("laur_cap");
        let expr = normalize::pow(x, Expr::int((MAX_LAURENT_SHIFT + 5) as i64));
        assert!(laurent_expand(&expr, x_id, &Expr::int(0), 1, 1, None).is_none());
    }
}
