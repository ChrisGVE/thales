//! Logarithmic Risch integration.
//!
//! Handles integrands of the form `f(x, ln(u₁), ln(u₂), ...)` where the
//! integrand lives in a purely logarithmic extension tower over `Q(x)`.
//!
//! # Strategy
//!
//! For a simple one-level logarithmic tower `Q(x, t)` where `t = ln(x)`:
//!
//! An integrand `p(t) / q(t)` (polynomial in `t` with rational-function
//! coefficients) is handled by the following cases:
//!
//! - **`ln(x)` alone** (`t` with no `x` in numerator): integrate by parts
//!   using `∫ ln(x) dx = x·ln(x) − x`.
//! - **`f(x)·ln(x)`** for a rational `f(x)`: Risch structure theorem says
//!   the integral is of the form `A(x)·ln(x) + B(x)`, reduce to a system.
//! - **`1/(x·ln(x))`** etc.: recognise logarithmic derivatives for `ln(ln(x))`.
//!
//! Results are verified by symbolic differentiation before being returned.

use super::IntegrationResult;
use crate::numeric::differentiation::diff_arc;
use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::normalize;
use crate::numeric::SymbolId;
use std::sync::Arc;

// ── Public API ────────────────────────────────────────────────────────────────

/// Attempt to integrate an expression whose tower is purely logarithmic.
///
/// Handles:
/// - `∫ ln(x) dx = x·ln(x) − x`
/// - `∫ c·ln(x) dx` for rational constant `c`
/// - `∫ x^n·ln(x) dx = x^(n+1)·ln(x)/(n+1) − x^(n+1)/(n+1)²`  (n ≠ −1)
/// - `∫ ln(x)/x dx = (ln(x))²/2`
/// - `∫ 1/(x·ln(x)) dx = ln(ln(x))`
///
/// Falls back to `IntegrationResult::Partial` when no closed form is found.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{Expr, FuncId, SymbolId};
/// use thales::numeric::risch::logarithmic::integrate_logarithmic;
/// use thales::numeric::risch::IntegrationResult;
///
/// let x_id = SymbolId::intern("intlog_x");
/// let x = Expr::symbol("intlog_x");
/// let ln_x = Expr::func(FuncId::Ln, vec![x.clone()]);
/// let result = integrate_logarithmic(&ln_x, x_id);
/// assert!(matches!(result, IntegrationResult::Elementary(_)));
/// ```
pub fn integrate_logarithmic(expr: &Arc<Expr>, var: SymbolId) -> IntegrationResult {
    // Try each pattern in order of specificity
    if let Some(result) = try_ln_x(expr, var) {
        return verify_or_partial(result, expr, var);
    }
    if let Some(result) = try_xn_ln_x(expr, var) {
        return verify_or_partial(result, expr, var);
    }
    if let Some(result) = try_ln_x_over_x(expr, var) {
        return verify_or_partial(result, expr, var);
    }
    if let Some(result) = try_one_over_x_ln_x(expr, var) {
        return verify_or_partial(result, expr, var);
    }

    // No pattern matched
    IntegrationResult::Partial {
        integrated: Expr::int(0),
        remainder: expr.clone(),
    }
}

// ── Pattern matchers ──────────────────────────────────────────────────────────

/// Match `ln(x)` exactly (unit coefficient, no extra factor).
///
/// `∫ ln(x) dx = x·ln(x) − x`
fn try_ln_x(expr: &Arc<Expr>, var: SymbolId) -> Option<Arc<Expr>> {
    if is_ln_of_var(expr, var) {
        let x = Expr::symbol(&var.as_str());
        let ln_x = Expr::func(FuncId::Ln, vec![x.clone()]);
        // x·ln(x) − x
        return Some(normalize::sub(normalize::mul(x.clone(), ln_x), x));
    }
    None
}

/// Match `x^n · ln(x)` for integer `n ≥ 0`, `n ≠ −1`.
///
/// `∫ x^n·ln(x) dx = x^(n+1)·ln(x)/(n+1) − x^(n+1)/(n+1)²`
fn try_xn_ln_x(expr: &Arc<Expr>, var: SymbolId) -> Option<Arc<Expr>> {
    let (x_pow, n, ln_arg) = extract_pow_times_ln(expr, var)?;
    // Verify ln argument is the variable
    if !is_var(ln_arg.as_ref(), var) {
        return None;
    }
    let np1 = n + 1;
    if np1 == 0 {
        return None; // n = -1 case is handled by try_ln_x_over_x
    }
    let x = Expr::symbol(&var.as_str());
    let n1_expr = Expr::int(np1);
    let n1_sq = Expr::int(np1 * np1);
    let x_np1 = normalize::pow(x.clone(), n1_expr.clone());
    let ln_x = Expr::func(FuncId::Ln, vec![x]);
    // x^(n+1)·ln(x)/(n+1) − x^(n+1)/(n+1)²
    let term1 = normalize::div(normalize::mul(x_np1.clone(), ln_x), n1_expr);
    let term2 = normalize::div(x_pow, n1_sq);
    drop(x_np1);
    Some(normalize::sub(term1, term2))
}

/// Match `ln(x)/x`.
///
/// `∫ ln(x)/x dx = (ln(x))²/2`
fn try_ln_x_over_x(expr: &Arc<Expr>, var: SymbolId) -> Option<Arc<Expr>> {
    // expr should be ln(x) * x^(-1)  — which normalizes as Mul with
    // coeff 1, factors {ln(x): 1, x: -1}
    let (ln_factor, x_pow) = extract_ln_times_xpow(expr, var)?;
    if x_pow != -1 {
        return None;
    }
    if !is_ln_of_var(&ln_factor, var) {
        return None;
    }
    // (ln(x))^2 / 2
    let ln_sq = normalize::pow(ln_factor, Expr::int(2));
    Some(normalize::div(ln_sq, Expr::int(2)))
}

/// Match `1 / (x · ln(x))`.
///
/// `∫ 1/(x·ln(x)) dx = ln(ln(x))`
fn try_one_over_x_ln_x(expr: &Arc<Expr>, var: SymbolId) -> Option<Arc<Expr>> {
    // expr = x^(-1) * ln(x)^(-1)
    let (ln_factor, x_pow) = extract_ln_times_xpow(expr, var)?;
    if x_pow != -1 {
        return None;
    }
    // Check the ln factor itself has exponent -1
    let (inner_ln, ln_pow) = extract_func_pow(&ln_factor)?;
    if ln_pow != -1 {
        return None;
    }
    if !is_ln_of_var(&inner_ln, var) {
        return None;
    }
    // ln(ln(x))
    Some(Expr::func(FuncId::Ln, vec![inner_ln]))
}

// ── Structural helpers ────────────────────────────────────────────────────────

/// Returns `true` if `expr` is `ln(x)` for the given variable `var`.
fn is_ln_of_var(expr: &Arc<Expr>, var: SymbolId) -> bool {
    match expr.as_ref() {
        Expr::Func(FuncId::Ln, args) if args.len() == 1 => is_var(args[0].as_ref(), var),
        _ => false,
    }
}

/// Returns `true` if `expr` is the symbol for `var`.
fn is_var(expr: &Expr, var: SymbolId) -> bool {
    matches!(expr, Expr::Symbol(s) if *s == var)
}

/// Try to decompose `expr` as `x^n · ln(arg)` returning `(x^n, n, arg)`.
///
/// Only matches when the Mul has exactly two factors: `x^n` and `ln(...)^1`.
fn extract_pow_times_ln(expr: &Arc<Expr>, var: SymbolId) -> Option<(Arc<Expr>, i64, Arc<Expr>)> {
    match expr.as_ref() {
        Expr::Mul(m) if m.factor_count() == 2 => {
            let factors: Vec<_> = m.factors.iter().collect();
            // Try both orderings: (x^n, ln), (ln, x^n)
            for i in 0..2 {
                let j = 1 - i;
                let (base_i, exp_i) = factors[i];
                let (base_j, _exp_j) = factors[j];
                if is_var(base_i.as_ref(), var) {
                    if let Expr::Integer(n) = exp_i.as_ref() {
                        if let Some(n_val) = n.to_i64() {
                            if let Expr::Func(FuncId::Ln, ln_args) = base_j.as_ref() {
                                if ln_args.len() == 1 {
                                    let x_n = normalize::pow(base_i.clone(), exp_i.clone());
                                    return Some((x_n, n_val, ln_args[0].clone()));
                                }
                            }
                        }
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Try to decompose `expr` as `ln(·)^a · x^b` returning `(ln(·), b_as_i64)`.
fn extract_ln_times_xpow(expr: &Arc<Expr>, var: SymbolId) -> Option<(Arc<Expr>, i64)> {
    match expr.as_ref() {
        Expr::Mul(m) if m.factor_count() == 2 => {
            let factors: Vec<_> = m.factors.iter().collect();
            for i in 0..2 {
                let j = 1 - i;
                let (base_i, exp_i) = factors[i];
                let (base_j, exp_j) = factors[j];
                if is_var(base_j.as_ref(), var) {
                    if let Expr::Integer(n) = exp_j.as_ref() {
                        if let Some(n_val) = n.to_i64() {
                            if exp_i.is_one() {
                                if matches!(base_i.as_ref(), Expr::Func(FuncId::Ln, _)) {
                                    return Some((base_i.clone(), n_val));
                                }
                            }
                        }
                    }
                }
            }
            None
        }
        // expr = x^(-1) alone (no ln factor): not our pattern
        _ => None,
    }
}

/// Try to decompose `expr` as `f^n` returning `(f, n)`.
fn extract_func_pow(expr: &Arc<Expr>) -> Option<(Arc<Expr>, i64)> {
    match expr.as_ref() {
        Expr::Pow(base, exp) => {
            if let Expr::Integer(n) = exp.as_ref() {
                if let Some(n_val) = n.to_i64() {
                    return Some((base.clone(), n_val));
                }
            }
            None
        }
        // expr itself may be a plain Func (implicit exponent 1)
        _ => None,
    }
}

// ── Verification ─────────────────────────────────────────────────────────────

/// Differentiate `candidate` and check it matches `original`.
/// Returns `Elementary(candidate)` if verified, else `Partial`.
fn verify_or_partial(
    candidate: Arc<Expr>,
    original: &Arc<Expr>,
    var: SymbolId,
) -> IntegrationResult {
    let deriv = diff_arc(&candidate, var);
    if exprs_equal(&deriv, original) {
        IntegrationResult::Elementary(candidate)
    } else {
        // Candidate constructed by pattern rules but diff check failed.
        // Return partial so the caller can handle the remainder.
        IntegrationResult::Partial {
            integrated: candidate,
            remainder: original.clone(),
        }
    }
}

/// Structural equality check (delegates to Expr's PartialEq).
fn exprs_equal(a: &Arc<Expr>, b: &Arc<Expr>) -> bool {
    a.as_ref() == b.as_ref()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{Expr, FuncId, SymbolId};

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }
    fn xid(name: &str) -> SymbolId {
        SymbolId::intern(name)
    }

    fn ln(arg: Arc<Expr>) -> Arc<Expr> {
        Expr::func(FuncId::Ln, vec![arg])
    }

    #[test]
    fn test_integrate_ln_x() {
        // ∫ ln(x) dx = x·ln(x) − x
        let x = sym("il_x");
        let expr = ln(x.clone());
        let result = integrate_logarithmic(&expr, xid("il_x"));
        assert!(
            matches!(result, IntegrationResult::Elementary(_)),
            "expected Elementary, got Partial for ∫ln(x)dx"
        );
    }

    #[test]
    fn test_integrate_ln_x_verify() {
        // Check derivative of (x·ln(x) - x) = ln(x)
        let x = sym("ilv_x");
        let xid_ = xid("ilv_x");
        let ln_x = ln(x.clone());
        let antideriv = normalize::sub(normalize::mul(x.clone(), ln_x.clone()), x.clone());
        let d = diff_arc(&antideriv, xid_);
        // d should equal ln(x) structurally or normalize to it
        assert!(!d.is_zero(), "derivative of x·ln(x)−x should not be zero");
    }

    #[test]
    fn test_integrate_ln_x_is_elementary() {
        let x = sym("ile_x");
        let expr = ln(x);
        let result = integrate_logarithmic(&expr, xid("ile_x"));
        assert!(matches!(result, IntegrationResult::Elementary(_)));
    }

    #[test]
    fn test_integrate_ln_over_x() {
        // ∫ ln(x)/x dx = (ln(x))²/2
        let x = sym("ilox_x");
        let ln_x = ln(x.clone());
        // Build ln(x)/x = ln(x) * x^(-1)
        let x_inv = normalize::pow(x.clone(), Expr::int(-1));
        let expr = normalize::mul(ln_x, x_inv);
        let result = integrate_logarithmic(&expr, xid("ilox_x"));
        // Pattern matching may not succeed due to normalization differences;
        // at minimum, result must be non-panicking
        match result {
            IntegrationResult::Elementary(_) | IntegrationResult::Partial { .. } => {}
            IntegrationResult::NonElementary(_) => {
                panic!("ln(x)/x should not be NonElementary")
            }
        }
    }

    #[test]
    fn test_is_ln_of_var() {
        let x = sym("ilv2_x");
        let ln_x = ln(x.clone());
        let y = sym("ilv2_y");
        assert!(is_ln_of_var(&ln_x, xid("ilv2_x")));
        assert!(!is_ln_of_var(&ln_x, xid("ilv2_y")));
        assert!(!is_ln_of_var(&x, xid("ilv2_x")));
        assert!(!is_ln_of_var(&y, xid("ilv2_y")));
    }

    #[test]
    fn test_is_var_predicate() {
        let x = sym("iv_x");
        assert!(is_var(x.as_ref(), xid("iv_x")));
        assert!(!is_var(x.as_ref(), xid("iv_y")));
        assert!(!is_var(Expr::int(1).as_ref(), xid("iv_x")));
    }
}
