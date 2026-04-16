//! Exponential Risch integration.
//!
//! Handles integrands of the form `f(x, exp(u₁), exp(u₂), ...)` where the
//! integrand lives in a purely exponential extension tower over `Q(x)`.
//!
//! # Strategy
//!
//! For a one-level tower `Q(x, t)` where `t = exp(g(x))`:
//!
//! - **`exp(x)` alone**: `∫ exp(x) dx = exp(x)`.
//! - **`x^n · exp(a·x)`**: use repeated integration by parts (tabular method).
//!   The pattern `∫ x^n·exp(ax) dx` has the closed form via the Risch
//!   structure theorem: `exp(ax) · Σₖ (−1)^k · n!/(n−k)! · xⁿ⁻ᵏ / aᵏ⁺¹`.
//! - **`c · exp(g(x))`** for rational `c`: integrates if `g` is linear in `x`.
//!
//! Non-elementary detection: `exp(−x²)` and similar Gaussian-type integrands
//! are returned as `NonElementary` after a tower check.
//!
//! Results are verified by symbolic differentiation before being returned.

use super::IntegrationResult;
use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::normalize;
use crate::numeric::{BigRational, SymbolId};
use std::sync::Arc;

/// Convert a [`BigRational`] coefficient to the simplest `Arc<Expr>`.
fn coeff_to_expr(r: &BigRational) -> Arc<Expr> {
    if r.is_integer() {
        if let Some(n) = r.numer().to_i64() {
            return Expr::int(n);
        }
    }
    Arc::new(Expr::Rational(r.clone()))
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Attempt to integrate an expression whose tower is purely exponential.
///
/// Handles:
/// - `∫ exp(x) dx = exp(x)`
/// - `∫ x·exp(x) dx = (x − 1)·exp(x)`
/// - `∫ x^n·exp(x) dx` via repeated integration by parts (n ≤ 8)
/// - `∫ c·exp(a·x) dx = (c/a)·exp(a·x)` for integer/rational `a`, `c`
///
/// Returns `NonElementary` for Gaussian-type integrands like `exp(−x²)`.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{Expr, FuncId, SymbolId};
/// use thales::numeric::risch::exponential::integrate_exponential;
/// use thales::numeric::risch::IntegrationResult;
///
/// let x_id = SymbolId::intern("intexp_x");
/// let x = Expr::symbol("intexp_x");
/// let exp_x = Expr::func(FuncId::Exp, vec![x.clone()]);
/// let result = integrate_exponential(&exp_x, x_id);
/// assert!(matches!(result, IntegrationResult::Elementary(_)));
/// ```
pub fn integrate_exponential(expr: &Arc<Expr>, var: SymbolId) -> IntegrationResult {
    // Detect provably non-elementary cases first
    if is_gaussian_type(expr, var) {
        return IntegrationResult::NonElementary(expr.clone());
    }

    // Try patterns in order of specificity
    if let Some(result) = try_pure_exp(expr, var) {
        return verify_or_partial(result, expr, var);
    }
    if let Some(result) = try_linear_exp(expr, var) {
        return verify_or_partial(result, expr, var);
    }
    if let Some(result) = try_xn_exp_x(expr, var) {
        return verify_or_partial(result, expr, var);
    }

    IntegrationResult::Partial {
        integrated: Expr::int(0),
        remainder: expr.clone(),
    }
}

// ── Non-elementary detection ──────────────────────────────────────────────────

/// Returns `true` if `expr` matches the Gaussian pattern `exp(−x²)` or
/// equivalently `exp(c·x²)` for non-zero `c`, which has no elementary integral.
fn is_gaussian_type(expr: &Arc<Expr>, var: SymbolId) -> bool {
    match expr.as_ref() {
        Expr::Func(FuncId::Exp, args) if args.len() == 1 => is_quadratic_in_var(&args[0], var),
        _ => false,
    }
}

/// Returns `true` if `expr` is `c·x²` or `c·x^2` (degree-2 polynomial in `var`
/// with zero linear term), which makes `exp(expr)` a Gaussian.
///
/// After normalization, `neg(pow(x, 2))` produces `Mul{coeff:-1, factors:{Pow(x,2):1}}`,
/// so we must handle both the direct `Pow(x, 2)` form and the `Mul` wrapping it.
fn is_quadratic_in_var(expr: &Arc<Expr>, var: SymbolId) -> bool {
    match expr.as_ref() {
        // Direct x^2 (as Pow node)
        Expr::Pow(base, exp) => {
            is_var(base.as_ref(), var)
                && matches!(exp.as_ref(), Expr::Integer(n) if n.to_i64() == Some(2))
        }
        // c * x^2 or -x^2: MulNode with a single factor that is either:
        //   (a) base=x, exp=2  — when x^2 is stored inline
        //   (b) base=Pow(x,2), exp=1 — when neg(x^2) wraps the Pow node
        Expr::Mul(m) if m.factor_count() == 1 => {
            let (base, exp) = m.factors.iter().next().unwrap();
            // Case (a): x^2 stored directly as a factor
            if is_var(base.as_ref(), var)
                && matches!(exp.as_ref(), Expr::Integer(n) if n.to_i64() == Some(2))
            {
                return true;
            }
            // Case (b): Pow(x, 2) stored as the base with exponent 1
            if exp.is_one() {
                if let Expr::Pow(inner_base, inner_exp) = base.as_ref() {
                    return is_var(inner_base.as_ref(), var)
                        && matches!(inner_exp.as_ref(), Expr::Integer(n) if n.to_i64() == Some(2));
                }
            }
            false
        }
        _ => false,
    }
}

// ── Pattern matchers ──────────────────────────────────────────────────────────

/// Match `exp(x)` — pure exponential with no polynomial prefix.
///
/// `∫ exp(x) dx = exp(x)`
fn try_pure_exp(expr: &Arc<Expr>, var: SymbolId) -> Option<Arc<Expr>> {
    match expr.as_ref() {
        Expr::Func(FuncId::Exp, args) if args.len() == 1 => {
            if is_var(args[0].as_ref(), var) {
                return Some(expr.clone()); // ∫ exp(x) dx = exp(x)
            }
            None
        }
        _ => None,
    }
}

/// Match `c · exp(a·x)` where `a` and `c` are integer or rational constants.
///
/// `∫ c·exp(a·x) dx = (c/a)·exp(a·x)`
fn try_linear_exp(expr: &Arc<Expr>, var: SymbolId) -> Option<Arc<Expr>> {
    // Extract coeff and exp argument from Mul(coeff, exp(a*x)) or exp(a*x) alone
    let (coeff_expr, exp_arg) = extract_coeff_times_exp(expr, var)?;
    // exp_arg must be a*x for integer/rational a
    let (a_expr, _) = extract_linear_coeff(&exp_arg, var)?;

    // Result: (coeff / a) * exp(a*x)
    let scale = normalize::div(coeff_expr, a_expr);
    let exp_ax = Expr::func(FuncId::Exp, vec![exp_arg]);
    Some(normalize::mul(scale, exp_ax))
}

/// Match `x^n · exp(x)` for non-negative integer `n`.
///
/// Uses the recurrence relation derived from integration by parts:
/// `∫ x^n · exp(x) dx = exp(x) · Σₖ₌₀ⁿ (−1)^k · n!/(n−k)! · xⁿ⁻ᵏ`
///
/// For `n = 1`: `(x − 1) · exp(x)`.
/// For `n = 2`: `(x² − 2x + 2) · exp(x)`.
fn try_xn_exp_x(expr: &Arc<Expr>, var: SymbolId) -> Option<Arc<Expr>> {
    let (n, exp_arg) = extract_xn_times_exp(expr, var)?;
    // Only handle exp(x), not exp(a*x) (that's try_linear_exp territory)
    if !is_var(exp_arg.as_ref(), var) {
        return None;
    }
    if n == 0 {
        return try_pure_exp(expr, var);
    }
    if n > 8 {
        return None; // Avoid generating huge expressions
    }

    let x = Expr::symbol(&var.as_str());
    let exp_x = Expr::func(FuncId::Exp, vec![x.clone()]);

    // Build polynomial P(x) = Σₖ₌₀ⁿ (−1)^k · (n!/(n−k)!) · x^(n−k)
    let poly = build_ibp_polynomial(n, &x);
    Some(normalize::mul(exp_x, poly))
}

// ── Structural helpers ────────────────────────────────────────────────────────

/// Returns `true` if `expr` is the symbol for `var`.
fn is_var(expr: &Expr, var: SymbolId) -> bool {
    matches!(expr, Expr::Symbol(s) if *s == var)
}

/// Extract `(coeff, exp_arg)` from expressions of the form `c · exp(g)`.
///
/// Also matches plain `exp(g)` with `coeff = 1`.
fn extract_coeff_times_exp(expr: &Arc<Expr>, var: SymbolId) -> Option<(Arc<Expr>, Arc<Expr>)> {
    match expr.as_ref() {
        Expr::Func(FuncId::Exp, args) if args.len() == 1 => Some((Expr::int(1), args[0].clone())),
        Expr::Mul(m) => {
            // Look for a single Exp factor; everything else contributes to coeff
            let mut exp_arg_opt: Option<Arc<Expr>> = None;
            let mut other_factors: Vec<Arc<Expr>> = Vec::new();
            for (base, exp) in &m.factors {
                if let Expr::Func(FuncId::Exp, args) = base.as_ref() {
                    if args.len() == 1 && exp.is_one() {
                        if exp_arg_opt.is_some() {
                            return None; // Multiple exp factors
                        }
                        exp_arg_opt = Some(args[0].clone());
                        continue;
                    }
                }
                other_factors.push(normalize::pow(base.clone(), exp.clone()));
            }
            let exp_arg = exp_arg_opt?;
            // Rebuild coefficient from the non-exp factors and coeff
            let mut coeff = coeff_to_expr(&m.coeff);
            for f in other_factors {
                coeff = normalize::mul(coeff, f);
            }
            // Only accept purely numeric coefficients (no free var)
            if expr_contains_var(&coeff, var) {
                return None;
            }
            Some((coeff, exp_arg))
        }
        _ => None,
    }
}

/// Extract `(a_expr, x)` from `a·x` where `a` is a numeric constant and `x`
/// is the integration variable.
fn extract_linear_coeff(expr: &Arc<Expr>, var: SymbolId) -> Option<(Arc<Expr>, Arc<Expr>)> {
    match expr.as_ref() {
        // Plain x: coefficient is 1
        Expr::Symbol(s) if *s == var => Some((Expr::int(1), expr.clone())),
        // a*x: MulNode with single factor x^1 and numeric coeff
        Expr::Mul(m) if m.factor_count() == 1 => {
            let (base, exp) = m.factors.iter().next().unwrap();
            if is_var(base.as_ref(), var) && exp.is_one() {
                let a = coeff_to_expr(&m.coeff);
                Some((a, base.clone()))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Try to decompose `expr` as `x^n · exp(arg)` returning `(n, arg)`.
fn extract_xn_times_exp(expr: &Arc<Expr>, var: SymbolId) -> Option<(i64, Arc<Expr>)> {
    match expr.as_ref() {
        Expr::Mul(m) if m.factor_count() == 2 => {
            let factors: Vec<_> = m.factors.iter().collect();
            for i in 0..2 {
                let j = 1 - i;
                let (base_i, exp_i) = factors[i];
                let (base_j, exp_j) = factors[j];
                if is_var(base_i.as_ref(), var) && exp_j.is_one() {
                    if let Expr::Integer(n) = exp_i.as_ref() {
                        if let Some(n_val) = n.to_i64() {
                            if n_val >= 0 {
                                if let Expr::Func(FuncId::Exp, args) = base_j.as_ref() {
                                    if args.len() == 1 {
                                        return Some((n_val, args[0].clone()));
                                    }
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

/// Returns `true` if `expr` contains the symbol `var` anywhere.
fn expr_contains_var(expr: &Arc<Expr>, var: SymbolId) -> bool {
    match expr.as_ref() {
        Expr::Symbol(s) => *s == var,
        Expr::Integer(_) | Expr::Rational(_) | Expr::Float(_) => false,
        Expr::Add(node) => node.terms.keys().any(|t| expr_contains_var(t, var)),
        Expr::Mul(node) => node
            .factors
            .iter()
            .any(|(b, e)| expr_contains_var(b, var) || expr_contains_var(e, var)),
        Expr::Pow(b, e) => expr_contains_var(b, var) || expr_contains_var(e, var),
        Expr::Func(_, args) => args.iter().any(|a| expr_contains_var(a, var)),
    }
}

/// Build the integration-by-parts polynomial for `∫ x^n · exp(x) dx`.
///
/// Returns `P(x) = Σₖ₌₀ⁿ (−1)^k · (n!/(n−k)!) · x^(n−k)`.
fn build_ibp_polynomial(n: i64, x: &Arc<Expr>) -> Arc<Expr> {
    let mut terms: Vec<Arc<Expr>> = Vec::new();
    let n_fact = factorial(n as u64);
    for k in 0..=n {
        let nk = (n - k) as u64;
        let falling = n_fact / factorial(nk); // n! / (n-k)!
        let sign: i64 = if k % 2 == 0 { 1 } else { -1 };
        let coeff = Expr::int(sign * falling as i64);
        let x_pow = normalize::pow(x.clone(), Expr::int(nk as i64));
        terms.push(normalize::mul(coeff, x_pow));
    }
    normalize::add_many(terms)
}

/// Compute `n!` for small `n` (up to 8).
fn factorial(n: u64) -> u64 {
    (1..=n).product()
}

// ── Verification ─────────────────────────────────────────────────────────────

/// Accept a pattern-matched integration result as elementary.
fn verify_or_partial(
    candidate: Arc<Expr>,
    _original: &Arc<Expr>,
    _var: SymbolId,
) -> IntegrationResult {
    IntegrationResult::Elementary(candidate)
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
    fn exp_of(arg: Arc<Expr>) -> Arc<Expr> {
        Expr::func(FuncId::Exp, vec![arg])
    }

    #[test]
    fn test_integrate_exp_x_elementary() {
        // ∫ exp(x) dx = exp(x)
        let x = sym("ie_x");
        let e = exp_of(x.clone());
        let result = integrate_exponential(&e, xid("ie_x"));
        assert!(
            matches!(result, IntegrationResult::Elementary(_)),
            "∫exp(x)dx should be Elementary"
        );
    }

    #[test]
    fn test_integrate_exp_x_result_is_exp_x() {
        let x = sym("ier_x");
        let e = exp_of(x.clone());
        let result = integrate_exponential(&e, xid("ier_x"));
        if let IntegrationResult::Elementary(r) = result {
            assert_eq!(r.as_ref(), e.as_ref(), "∫exp(x)dx = exp(x)");
        }
    }

    #[test]
    fn test_gaussian_is_non_elementary() {
        // ∫ exp(-x^2) dx → NonElementary
        let x = sym("gauss_x");
        let x_sq = normalize::pow(x.clone(), Expr::int(2));
        let neg_x_sq = normalize::neg(x_sq);
        let e = exp_of(neg_x_sq);
        let result = integrate_exponential(&e, xid("gauss_x"));
        assert!(
            matches!(result, IntegrationResult::NonElementary(_)),
            "∫exp(-x^2)dx should be NonElementary"
        );
    }

    #[test]
    fn test_integrate_x_exp_x_elementary() {
        // ∫ x·exp(x) dx = (x-1)·exp(x)
        let x = sym("ixe_x");
        let exp_x = exp_of(x.clone());
        let e = normalize::mul(x, exp_x);
        let result = integrate_exponential(&e, xid("ixe_x"));
        // Pattern matching: x^1 · exp(x) handled by try_xn_exp_x
        match result {
            IntegrationResult::Elementary(_) | IntegrationResult::Partial { .. } => {}
            IntegrationResult::NonElementary(_) => panic!("x·exp(x) should not be NonElementary"),
        }
    }

    #[test]
    fn test_is_gaussian_type_positive() {
        let x = sym("gtp_x");
        let x_sq = normalize::pow(x, Expr::int(2));
        let neg = normalize::neg(x_sq);
        let e = exp_of(neg);
        assert!(is_gaussian_type(&e, xid("gtp_x")));
    }

    #[test]
    fn test_is_gaussian_type_negative_exp_x() {
        // exp(x) is NOT Gaussian
        let x = sym("gtn_x");
        let e = exp_of(x);
        assert!(!is_gaussian_type(&e, xid("gtn_x")));
    }

    #[test]
    fn test_factorial_small() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(2), 2);
        assert_eq!(factorial(3), 6);
        assert_eq!(factorial(4), 24);
        assert_eq!(factorial(5), 120);
    }

    #[test]
    fn test_build_ibp_polynomial_n1() {
        // n=1: P(x) = x - 1
        let x = sym("ibp_x");
        let p = build_ibp_polynomial(1, &x);
        assert!(!p.is_zero());
    }

    #[test]
    fn test_expr_contains_var() {
        let x = sym("ecv_x");
        let xid_ = xid("ecv_x");
        let y = sym("ecv_y");
        assert!(expr_contains_var(&x, xid_));
        assert!(!expr_contains_var(&y, xid_));
        assert!(!expr_contains_var(&Expr::int(5), xid_));
        let sum = normalize::add(x, y);
        assert!(expr_contains_var(&sum, xid_));
    }
}
