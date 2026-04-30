//! Pattern-based integration with heuristics.
//!
//! Provides a layered integration strategy that runs before (or alongside) the
//! full Risch algorithm:
//!
//! 1. **Table lookup** — direct integral patterns for common elementary functions.
//! 2. **U-substitution** — detects `f(g(x)) · g'(x)` and reduces to `∫ f(u) du`.
//! 3. **Tabular integration by parts** — handles `xⁿ · sin(x)` and `xⁿ · cos(x)`
//!    for small non-negative integer `n`.
//! 4. **Risch fallback** — delegates to [`risch_integrate`] when the above fail.
//!
//! # Entry point
//!
//! ```rust
//! use thales::numeric::{Expr, FuncId, SymbolId};
//! use thales::numeric::pattern_integrate::pattern_integrate;
//! use thales::numeric::risch::IntegrationResult;
//!
//! let x_id = SymbolId::intern("pi_exp_x");
//! let x    = Expr::symbol("pi_exp_x");
//! let e    = Expr::func(FuncId::Exp, vec![x]);
//! let res  = pattern_integrate(&e, x_id);
//! assert!(matches!(res, IntegrationResult::Elementary(_)));
//! ```

use crate::numeric::differentiation::diff_arc;
use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::risch::{risch_integrate, IntegrationResult};
use crate::numeric::substitute::substitute;
use crate::numeric::{normalize, SymbolId};
use std::sync::Arc;

// ── Public entry point ────────────────────────────────────────────────────────

/// Integrate `expr` with respect to `var` using pattern matching and heuristics.
///
/// Strategy (in order):
/// 1. Table lookup for known elementary patterns.
/// 2. U-substitution heuristic.
/// 3. Tabular integration by parts for `xⁿ · trig`.
/// 4. Risch algorithm as fallback.
pub fn pattern_integrate(expr: &Arc<Expr>, var: SymbolId) -> IntegrationResult {
    if let Some(result) = table_lookup(expr, var) {
        return IntegrationResult::Elementary(result);
    }

    if let Some(result) = try_u_substitution(expr, var) {
        return IntegrationResult::Elementary(result);
    }

    if let Some(result) = try_tabular_parts(expr, var) {
        return IntegrationResult::Elementary(result);
    }

    risch_integrate(expr, var)
}

// ── Table lookup ──────────────────────────────────────────────────────────────

/// Return the antiderivative for a handful of canonical patterns, or `None`.
///
/// Covers:
/// - `xⁿ` (n ≠ −1) → `xⁿ⁺¹ / (n+1)`
/// - `1/x`         → `ln(x)`
/// - `sin(x)`      → `−cos(x)`
/// - `cos(x)`      → `sin(x)`
/// - `exp(x)`      → `exp(x)`
fn table_lookup(expr: &Arc<Expr>, var: SymbolId) -> Option<Arc<Expr>> {
    let x = Expr::symbol(&var.as_str());

    // ── sin(x) → -cos(x) ─────────────────────────────────────────────────────
    if let Some(inner) = match_func1(expr, FuncId::Sin) {
        if is_var(inner, var) {
            return Some(normalize::neg(Expr::func(FuncId::Cos, vec![x])));
        }
    }

    // ── cos(x) → sin(x) ─────────────────────────────────────────────────────
    if let Some(inner) = match_func1(expr, FuncId::Cos) {
        if is_var(inner, var) {
            return Some(Expr::func(FuncId::Sin, vec![x]));
        }
    }

    // ── exp(x) → exp(x) ─────────────────────────────────────────────────────
    if let Some(inner) = match_func1(expr, FuncId::Exp) {
        if is_var(inner, var) {
            return Some(expr.clone());
        }
    }

    // ── power and reciprocal patterns ────────────────────────────────────────
    table_lookup_power(expr, var, &x)
}

/// Handle `xⁿ` (n ≠ −1) and `1/x` cases.
fn table_lookup_power(expr: &Arc<Expr>, var: SymbolId, x: &Arc<Expr>) -> Option<Arc<Expr>> {
    // 1/x → ln(x)  encoded as x^(-1) after normalize::div
    if let Expr::Pow(base, exp) = expr.as_ref() {
        if is_var(base, var) {
            if let Some(n) = as_integer(exp) {
                if n == -1 {
                    return Some(Expr::func(FuncId::Ln, vec![x.clone()]));
                }
                // x^n → x^(n+1) / (n+1)
                let np1 = n + 1;
                let new_pow = normalize::pow(x.clone(), Expr::int(np1));
                let denom = Expr::int(np1);
                return Some(normalize::div(new_pow, denom));
            }
        }
    }

    // Bare symbol x → x^1 → x^2 / 2
    if is_var(expr, var) {
        let x_sq = normalize::pow(x.clone(), Expr::int(2));
        return Some(normalize::div(x_sq, Expr::int(2)));
    }

    // Constant (no var) → c * x
    if !contains_var(expr, var) {
        return Some(normalize::mul(expr.clone(), x.clone()));
    }

    // c * x^n  (Mul node where one factor contains var, the rest are constant)
    //
    // When the expression is a product and we can peel off a constant coefficient
    // (w.r.t. `var`) from a single var-dependent factor, integrate that factor and
    // multiply by the constant.  This covers the common multivariate case e.g.
    // ∫ y·x dx = y · x²/2  where `y` is free w.r.t. `x`.
    if let Some((coeff, factors)) = extract_mul_factors(expr) {
        let var_factors: Vec<&Arc<Expr>> =
            factors.iter().filter(|f| contains_var(f, var)).collect();
        let const_factors: Vec<&Arc<Expr>> =
            factors.iter().filter(|f| !contains_var(f, var)).collect();

        // Only handle the case of exactly one var-containing factor; anything
        // more complex (e.g. x·sin(x)) is handled by tabular/by-parts.
        if var_factors.len() == 1 {
            let var_part = var_factors[0];
            if let Some(anti_var) = table_lookup_power(var_part, var, x) {
                // constant part = coeff * product of const_factors
                let const_part = const_factors
                    .iter()
                    .fold(coeff, |acc, f| normalize::mul(acc, (*f).clone()));
                return Some(normalize::mul(const_part, anti_var));
            }
        }
    }

    None
}

// ── U-substitution ────────────────────────────────────────────────────────────

/// Detect `coeff · F(inner) · inner'` and return `coeff · ∫F(u)du` with u=inner.
///
/// The heuristic tries every single-argument function sub-expression as a
/// candidate `u = g(x)`, computes `g'(x)`, and checks whether the remaining
/// factor simplifies to a constant multiple of `g'(x)`.
fn try_u_substitution(expr: &Arc<Expr>, var: SymbolId) -> Option<Arc<Expr>> {
    // Accept both products and bare single-function expressions like `sin(2x)`.
    let (coeff, factors) =
        extract_mul_factors(expr).unwrap_or_else(|| (Expr::int(1), vec![expr.clone()]));

    // Find the first factor that is a known single-arg function
    let (func_idx, func_id, inner) =
        factors
            .iter()
            .enumerate()
            .find_map(|(i, f)| match f.as_ref() {
                Expr::Func(id, args) if args.len() == 1 => Some((i, *id, args[0].clone())),
                _ => None,
            })?;

    // g'(x)
    let g_prime = diff_arc(&inner, var);
    if g_prime.is_zero() {
        return None;
    }

    // Product of all factors except the function itself
    let remainder: Arc<Expr> = factors
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != func_idx)
        .fold(Expr::int(1), |acc, (_, f)| normalize::mul(acc, f.clone()));

    // Check if remainder / g'(x) is a constant w.r.t. var
    let ratio = normalize::div(remainder, g_prime);
    if contains_var(&ratio, var) {
        return None;
    }

    // ∫ F(u) du with u = inner
    let u = Expr::symbol("__u_sub__");
    let u_id = SymbolId::intern("__u_sub__");
    let f_of_u = Expr::func(func_id, vec![u]);
    let integral_f = table_lookup(&f_of_u, u_id)?;

    // Substitute back: replace __u_sub__ with inner
    let result_back = substitute(&integral_f, u_id, &inner);
    let scaled = normalize::mul(normalize::mul(coeff, ratio), result_back);
    Some(scaled)
}

// ── Tabular integration by parts ─────────────────────────────────────────────

/// Handle `xⁿ · sin(x)`, `xⁿ · cos(x)`, and `xⁿ · exp(x)` for `n ∈ {0..=6}`
/// via the tabular (repeated differentiation / integration) method.
fn try_tabular_parts(expr: &Arc<Expr>, var: SymbolId) -> Option<Arc<Expr>> {
    let (n, func_id) = extract_xn_elementary(expr, var)?;
    if !(0..=6).contains(&n) {
        return None;
    }
    let x = Expr::symbol(&var.as_str());
    match func_id {
        FuncId::Sin | FuncId::Cos => Some(tabular_xn_trig(n, func_id, &x, var)),
        FuncId::Exp => Some(tabular_xn_exp(n, &x, var)),
        _ => None,
    }
}

/// Extract `(n, FuncId)` from an expression matching `xⁿ · f(x)` where `f` is
/// one of `sin`, `cos`, `exp`. Also matches the bare function for `n=0` and
/// the single-factor `x · f(x)` product for `n=1`.
fn extract_xn_elementary(expr: &Arc<Expr>, var: SymbolId) -> Option<(i64, FuncId)> {
    // Direct: f(x) with f ∈ {sin, cos, exp}
    for fid in [FuncId::Sin, FuncId::Cos, FuncId::Exp] {
        if let Some(inner) = match_func1(expr, fid) {
            if is_var(inner, var) {
                return Some((0, fid));
            }
        }
    }

    // Product: must have exactly xⁿ and a matching func factor
    let (_, factors) = extract_mul_factors(expr)?;
    let mut pow_n: Option<i64> = None;
    let mut func_id: Option<FuncId> = None;

    for f in &factors {
        match f.as_ref() {
            Expr::Func(id @ (FuncId::Sin | FuncId::Cos | FuncId::Exp), args)
                if args.len() == 1 && is_var(&args[0], var) =>
            {
                func_id = Some(*id);
            }
            Expr::Pow(base, exp) if is_var(base, var) => {
                pow_n = as_integer(exp);
            }
            _ if is_var(f, var) => {
                pow_n = Some(1);
            }
            _ => {}
        }
    }

    match (pow_n, func_id) {
        (Some(n), Some(id)) if n >= 0 => Some((n, id)),
        _ => None,
    }
}

/// Compute `∫ xⁿ · sin/cos(x) dx` via the tabular method.
///
/// Tabular method: differentiate `xⁿ` repeatedly until 0, integrate
/// `sin/cos` repeatedly, and alternate signs.
fn tabular_xn_trig(n: i64, trig: FuncId, x: &Arc<Expr>, var: SymbolId) -> Arc<Expr> {
    // Build columns: derivatives of x^n (stops at 0)
    let x_pow = normalize::pow(x.clone(), Expr::int(n));
    let mut derivs: Vec<Arc<Expr>> = vec![x_pow];
    for _ in 0..=n {
        let last = derivs.last().unwrap().clone();
        let d = diff_arc(&last, var);
        derivs.push(d);
    }

    // Build column of integrals of trig, alternating between sin/cos
    // starting with ∫trig: sin→-cos, cos→sin, -cos→-sin, -sin→cos
    let mut integrals: Vec<(Arc<Expr>, bool)> = Vec::new();
    let mut cur_func = trig;
    let mut cur_neg = false;
    for _ in 0..=n {
        let (next_func, next_neg) = integrate_pure_trig(cur_func, cur_neg);
        integrals.push((
            build_signed_trig(next_func, next_neg, x),
            false, // sign already folded in
        ));
        cur_func = next_func;
        cur_neg = next_neg;
    }

    // Sum: (+u₀·v₁) + (−u₁·v₂) + (+u₂·v₃) + …
    let mut sign_pos = true;
    let mut acc = Expr::int(0);
    for i in 0..derivs.len().min(integrals.len()) {
        let d = &derivs[i];
        let v = &integrals[i].0;
        let term = normalize::mul(d.clone(), v.clone());
        if sign_pos {
            acc = normalize::add(acc, term);
        } else {
            acc = normalize::sub(acc, term);
        }
        sign_pos = !sign_pos;
    }
    acc
}

/// Compute `∫ xⁿ · exp(x) dx` via the tabular method.
///
/// `∫exp(x) = exp(x)` (self-repeating), so the table is:
/// derivatives `xⁿ, n·xⁿ⁻¹, …, n!, 0` against integrals `exp(x), exp(x), …`
/// with alternating signs: `+u₀·v₁ − u₁·v₂ + u₂·v₃ − …`.
fn tabular_xn_exp(n: i64, x: &Arc<Expr>, var: SymbolId) -> Arc<Expr> {
    let x_pow = normalize::pow(x.clone(), Expr::int(n));
    let mut derivs: Vec<Arc<Expr>> = vec![x_pow];
    for _ in 0..=n {
        let last = derivs.last().unwrap().clone();
        let d = diff_arc(&last, var);
        derivs.push(d);
    }
    let exp_x = Expr::func(FuncId::Exp, vec![x.clone()]);

    let mut sign_pos = true;
    let mut acc = Expr::int(0);
    for d in &derivs {
        if d.is_zero() {
            break;
        }
        let term = normalize::mul(d.clone(), exp_x.clone());
        if sign_pos {
            acc = normalize::add(acc, term);
        } else {
            acc = normalize::sub(acc, term);
        }
        sign_pos = !sign_pos;
    }
    acc
}

/// Given a trig function in a repeating integration cycle, return the next.
///
/// Cycle: sin → -cos → -sin → cos → sin …
/// Returns `(next_FuncId, negated)`.
fn integrate_pure_trig(func: FuncId, negated: bool) -> (FuncId, bool) {
    match (func, negated) {
        (FuncId::Sin, false) => (FuncId::Cos, true), // ∫sin = -cos
        (FuncId::Cos, true) => (FuncId::Sin, true),  // ∫(-cos) = -sin
        (FuncId::Sin, true) => (FuncId::Cos, false), // ∫(-sin) = cos
        (FuncId::Cos, false) => (FuncId::Sin, false), // ∫cos = sin
        _ => (FuncId::Sin, false),
    }
}

/// Build `±func(x)` as an expression.
fn build_signed_trig(func: FuncId, negated: bool, x: &Arc<Expr>) -> Arc<Expr> {
    let f = Expr::func(func, vec![x.clone()]);
    if negated {
        normalize::neg(f)
    } else {
        f
    }
}

// ── Small utilities ───────────────────────────────────────────────────────────

/// If `expr` is `Func(id, [arg])`, return `&arg`, else `None`.
fn match_func1(expr: &Arc<Expr>, id: FuncId) -> Option<&Arc<Expr>> {
    match expr.as_ref() {
        Expr::Func(f, args) if *f == id && args.len() == 1 => Some(&args[0]),
        _ => None,
    }
}

/// Return `true` iff `expr` is the symbol `var`.
fn is_var(expr: &Arc<Expr>, var: SymbolId) -> bool {
    matches!(expr.as_ref(), Expr::Symbol(s) if *s == var)
}

/// Return the integer value if `expr` is `Expr::Integer(n)`, else `None`.
fn as_integer(expr: &Arc<Expr>) -> Option<i64> {
    match expr.as_ref() {
        Expr::Integer(n) => n.to_i64(),
        _ => None,
    }
}

/// Return `true` if `expr` syntactically contains the symbol `var`.
fn contains_var(expr: &Arc<Expr>, var: SymbolId) -> bool {
    match expr.as_ref() {
        Expr::Symbol(s) => *s == var,
        Expr::Integer(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Complex(_)
        | Expr::Constant(_) => false,
        Expr::Add(node) => node.terms.keys().any(|t| contains_var(t, var)),
        Expr::Mul(node) => node
            .factors
            .iter()
            .any(|(b, e)| contains_var(b, var) || contains_var(e, var)),
        Expr::Pow(b, e) => contains_var(b, var) || contains_var(e, var),
        Expr::Func(_, args) => args.iter().any(|a| contains_var(a, var)),
    }
}

/// Decompose a product into `(scalar_coeff, Vec<factor>)`.
///
/// Returns `None` if `expr` is not a `Mul` node (caller handles single-factor
/// cases separately).
fn extract_mul_factors(expr: &Arc<Expr>) -> Option<(Arc<Expr>, Vec<Arc<Expr>>)> {
    match expr.as_ref() {
        Expr::Mul(node) => {
            let coeff = Arc::new(Expr::Rational(node.coeff.clone()));
            let factors: Vec<Arc<Expr>> = node
                .factors
                .iter()
                .map(|(b, e)| {
                    if e.is_one() {
                        b.clone()
                    } else {
                        normalize::pow(b.clone(), e.clone())
                    }
                })
                .collect();
            Some((coeff, factors))
        }
        _ => None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::SymbolId;

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }
    fn xid(name: &str) -> SymbolId {
        SymbolId::intern(name)
    }

    // ── table: sin(x) → -cos(x) ─────────────────────────────────────────────

    #[test]
    fn test_table_sin_x() {
        let x = sym("pt_sin_x");
        let e = Expr::func(FuncId::Sin, vec![x]);
        let r = pattern_integrate(&e, xid("pt_sin_x"));
        assert!(matches!(r, IntegrationResult::Elementary(_)));
        if let IntegrationResult::Elementary(antideriv) = r {
            // d/dx(-cos(x)) = sin(x)
            let d = diff_arc(&antideriv, xid("pt_sin_x"));
            assert_eq!(d.as_ref(), e.as_ref());
        }
    }

    // ── table: cos(x) → sin(x) ───────────────────────────────────────────────

    #[test]
    fn test_table_cos_x() {
        let x = sym("pt_cos_x");
        let e = Expr::func(FuncId::Cos, vec![x]);
        let r = pattern_integrate(&e, xid("pt_cos_x"));
        assert!(matches!(r, IntegrationResult::Elementary(_)));
        if let IntegrationResult::Elementary(antideriv) = r {
            let d = diff_arc(&antideriv, xid("pt_cos_x"));
            assert_eq!(d.as_ref(), e.as_ref());
        }
    }

    // ── table: exp(x) → exp(x) ───────────────────────────────────────────────

    #[test]
    fn test_table_exp_x() {
        let x = sym("pt_exp_x");
        let e = Expr::func(FuncId::Exp, vec![x]);
        let r = pattern_integrate(&e, xid("pt_exp_x"));
        assert!(matches!(r, IntegrationResult::Elementary(_)));
        if let IntegrationResult::Elementary(antideriv) = r {
            assert_eq!(antideriv.as_ref(), e.as_ref());
        }
    }

    // ── table: x → x²/2 ─────────────────────────────────────────────────────

    #[test]
    fn test_table_x_power_1() {
        let x = sym("pt_x1_x");
        let r = pattern_integrate(&x, xid("pt_x1_x"));
        assert!(matches!(r, IntegrationResult::Elementary(_)));
        if let IntegrationResult::Elementary(antideriv) = r {
            let d = diff_arc(&antideriv, xid("pt_x1_x"));
            assert_eq!(d.as_ref(), x.as_ref());
        }
    }

    // ── table: x² → x³/3 ────────────────────────────────────────────────────

    #[test]
    fn test_table_x_power_2() {
        let x = sym("pt_x2_x");
        let e = normalize::pow(x.clone(), Expr::int(2));
        let r = pattern_integrate(&e, xid("pt_x2_x"));
        assert!(matches!(r, IntegrationResult::Elementary(_)));
        if let IntegrationResult::Elementary(antideriv) = r {
            let d = diff_arc(&antideriv, xid("pt_x2_x"));
            assert_eq!(d.as_ref(), e.as_ref());
        }
    }

    // ── table: 1/x → ln(x) ──────────────────────────────────────────────────

    #[test]
    fn test_table_one_over_x() {
        let x = sym("pt_1x_x");
        let e = normalize::div(Expr::int(1), x.clone());
        let r = pattern_integrate(&e, xid("pt_1x_x"));
        assert!(matches!(r, IntegrationResult::Elementary(_)));
        if let IntegrationResult::Elementary(antideriv) = r {
            // antiderivative should be ln(x) — verify by differentiation
            let d = diff_arc(&antideriv, xid("pt_1x_x"));
            assert_eq!(d.as_ref(), e.as_ref());
        }
    }

    // ── table: constant → c*x ───────────────────────────────────────────────

    #[test]
    fn test_table_constant() {
        let x = sym("pt_const_x");
        let c = Expr::int(5);
        let r = pattern_integrate(&c, xid("pt_const_x"));
        assert!(matches!(r, IntegrationResult::Elementary(_)));
        if let IntegrationResult::Elementary(antideriv) = r {
            let d = diff_arc(&antideriv, xid("pt_const_x"));
            assert_eq!(d.as_ref(), c.as_ref());
        }
    }

    // ── u-sub: sin(x²) · 2x → -cos(x²) ─────────────────────────────────────

    #[test]
    fn test_u_sub_sin_x2() {
        // ∫ 2x · sin(x²) dx = -cos(x²)
        let x = sym("pt_us_x");
        let x2 = normalize::pow(x.clone(), Expr::int(2));
        let sin_x2 = Expr::func(FuncId::Sin, vec![x2]);
        let two_x = normalize::mul(Expr::int(2), x);
        let integrand = normalize::mul(two_x, sin_x2);
        let r = pattern_integrate(&integrand, xid("pt_us_x"));
        assert!(
            matches!(r, IntegrationResult::Elementary(_)),
            "u-sub for 2x*sin(x²) should succeed"
        );
        if let IntegrationResult::Elementary(antideriv) = r {
            let d = diff_arc(&antideriv, xid("pt_us_x"));
            assert_eq!(d.as_ref(), integrand.as_ref());
        }
    }

    // ── u-sub: cos(x²) · 2x → sin(x²) ──────────────────────────────────────

    #[test]
    fn test_u_sub_cos_x2() {
        let x = sym("pt_uc_x");
        let x2 = normalize::pow(x.clone(), Expr::int(2));
        let cos_x2 = Expr::func(FuncId::Cos, vec![x2]);
        let two_x = normalize::mul(Expr::int(2), x);
        let integrand = normalize::mul(two_x, cos_x2);
        let r = pattern_integrate(&integrand, xid("pt_uc_x"));
        assert!(
            matches!(r, IntegrationResult::Elementary(_)),
            "u-sub for 2x*cos(x²) should succeed"
        );
        if let IntegrationResult::Elementary(antideriv) = r {
            let d = diff_arc(&antideriv, xid("pt_uc_x"));
            assert_eq!(d.as_ref(), integrand.as_ref());
        }
    }

    // ── tabular: sin(x) = x⁰·sin(x) ────────────────────────────────────────

    #[test]
    fn test_tabular_n0_sin() {
        let x = sym("pt_t0s_x");
        let e = Expr::func(FuncId::Sin, vec![x]);
        let r = pattern_integrate(&e, xid("pt_t0s_x"));
        assert!(matches!(r, IntegrationResult::Elementary(_)));
    }

    // ── tabular: x · sin(x) ─────────────────────────────────────────────────

    #[test]
    fn test_tabular_x_sin_x() {
        // ∫ x sin(x) dx = sin(x) - x·cos(x)
        let x = sym("pt_xsin_x");
        let sin_x = Expr::func(FuncId::Sin, vec![x.clone()]);
        let cos_x = Expr::func(FuncId::Cos, vec![x.clone()]);
        let integrand = normalize::mul(x.clone(), sin_x.clone());
        let r = pattern_integrate(&integrand, xid("pt_xsin_x"));
        assert!(
            matches!(r, IntegrationResult::Elementary(_)),
            "tabular x·sin(x) should give Elementary"
        );
        if let IntegrationResult::Elementary(antideriv) = r {
            // Expected: sin(x) - x·cos(x)
            let expected = normalize::sub(sin_x, normalize::mul(x.clone(), cos_x));
            assert_eq!(
                antideriv.as_ref(),
                expected.as_ref(),
                "∫x·sin(x)dx = sin(x) - x·cos(x)"
            );
        }
    }

    // ── tabular: x · cos(x) ─────────────────────────────────────────────────

    #[test]
    fn test_tabular_x_cos_x() {
        // ∫ x cos(x) dx = cos(x) + x·sin(x)
        let x = sym("pt_xcos_x");
        let sin_x = Expr::func(FuncId::Sin, vec![x.clone()]);
        let cos_x = Expr::func(FuncId::Cos, vec![x.clone()]);
        let integrand = normalize::mul(x.clone(), cos_x.clone());
        let r = pattern_integrate(&integrand, xid("pt_xcos_x"));
        assert!(
            matches!(r, IntegrationResult::Elementary(_)),
            "tabular x·cos(x) should give Elementary"
        );
        if let IntegrationResult::Elementary(antideriv) = r {
            // Expected: cos(x) + x·sin(x)
            let expected = normalize::add(cos_x, normalize::mul(x.clone(), sin_x));
            assert_eq!(
                antideriv.as_ref(),
                expected.as_ref(),
                "∫x·cos(x)dx = cos(x) + x·sin(x)"
            );
        }
    }

    // ── tabular: x² · sin(x) ────────────────────────────────────────────────

    #[test]
    fn test_tabular_x2_sin_x() {
        // ∫ x²·sin(x) dx = (2-x²)·cos(x) + 2x·sin(x)
        let x = sym("pt_x2sin_x");
        let x2 = normalize::pow(x.clone(), Expr::int(2));
        let sin_x = Expr::func(FuncId::Sin, vec![x.clone()]);
        let cos_x = Expr::func(FuncId::Cos, vec![x.clone()]);
        let integrand = normalize::mul(x2, sin_x.clone());
        let r = pattern_integrate(&integrand, xid("pt_x2sin_x"));
        assert!(
            matches!(r, IntegrationResult::Elementary(_)),
            "tabular x²·sin(x) should give Elementary"
        );
        if let IntegrationResult::Elementary(antideriv) = r {
            // Algorithm normalizes to: 2·cos(x) - x²·cos(x) + 2x·sin(x)
            // (canonical expanded form rather than factored (2-x²)·cos(x) + 2x·sin(x))
            let x2e = normalize::pow(x.clone(), Expr::int(2));
            let two_x = normalize::mul(Expr::int(2), x.clone());
            let expected = normalize::add_many(vec![
                normalize::mul(Expr::int(2), cos_x.clone()),
                normalize::neg(normalize::mul(x2e, cos_x)),
                normalize::mul(two_x, sin_x),
            ]);
            assert_eq!(
                antideriv.as_ref(),
                expected.as_ref(),
                "∫x²·sin(x)dx = 2·cos(x) - x²·cos(x) + 2x·sin(x)"
            );
        }
    }

    // ── risch fallback: ln(x) ───────────────────────────────────────────────

    #[test]
    fn test_fallback_to_risch_ln_x() {
        let x = sym("pt_fall_x");
        let ln_x = Expr::func(FuncId::Ln, vec![x]);
        let r = pattern_integrate(&ln_x, xid("pt_fall_x"));
        // Risch handles ln(x) → Elementary
        assert!(matches!(r, IntegrationResult::Elementary(_)));
    }
}
