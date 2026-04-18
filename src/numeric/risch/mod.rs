//! Risch integration algorithm framework.
//!
//! Implements the decision procedure for elementary integration of transcendental
//! functions. Given an expression `f(x)`, the algorithm determines whether `∫f dx`
//! is elementary and, if so, returns a closed form.
//!
//! # Algorithm overview
//!
//! 1. **Tower analysis** ([`tower`]): classify the integrand by which
//!    transcendental generators appear (`ln`, `exp`, or both).
//! 2. **Routing**: dispatch to the appropriate sub-algorithm:
//!    - Pure rational → Hermite reduction + Rothstein-Trager (existing).
//!    - Logarithmic tower → [`logarithmic::integrate_logarithmic`].
//!    - Exponential tower → [`exponential::integrate_exponential`].
//!    - Mixed or deep towers → attempt decomposition; return [`IntegrationResult::Partial`]
//!      when full integration is not yet supported.
//! 3. **Non-elementary detection**: certain patterns (Gaussian `exp(−x²)`, etc.)
//!    are definitively non-elementary and returned as [`IntegrationResult::NonElementary`].
//!
//! # Examples
//!
//! ```rust
//! use thales::numeric::{Expr, FuncId, SymbolId};
//! use thales::numeric::risch::{risch_integrate, IntegrationResult};
//!
//! // ∫ ln(x) dx = x·ln(x) − x
//! let x_id = SymbolId::intern("risch_ln_x");
//! let x = Expr::symbol("risch_ln_x");
//! let ln_x = Expr::func(FuncId::Ln, vec![x.clone()]);
//! let result = risch_integrate(&ln_x, x_id);
//! assert!(matches!(result, IntegrationResult::Elementary(_)));
//! ```
//!
//! ```rust
//! use thales::numeric::{Expr, FuncId, SymbolId};
//! use thales::numeric::risch::{risch_integrate, IntegrationResult};
//!
//! // ∫ exp(x) dx = exp(x)
//! let x_id = SymbolId::intern("risch_exp_x");
//! let x = Expr::symbol("risch_exp_x");
//! let exp_x = Expr::func(FuncId::Exp, vec![x.clone()]);
//! let result = risch_integrate(&exp_x, x_id);
//! assert!(matches!(result, IntegrationResult::Elementary(_)));
//! ```

pub mod exponential;
pub mod logarithmic;
pub mod tower;

use crate::numeric::differentiation::diff_arc;
use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::normalize;
use crate::numeric::{
    hermite_reduce, integrate_rational_log, BigRational, DensePolynomial, RationalFunction,
    SymbolId,
};
use std::sync::Arc;
use tower::{build_tower, TowerKind};

// ── Result type ───────────────────────────────────────────────────────────────

/// The outcome of an integration attempt.
///
/// Every variant carries enough information for the caller to proceed: either
/// the full antiderivative, proof of non-elementarity, or partial progress.
#[derive(Clone, Debug)]
pub enum IntegrationResult {
    /// A closed-form elementary antiderivative was found.
    Elementary(Arc<Expr>),
    /// The integral is proven non-elementary.
    ///
    /// The wrapped expression is the original integrand (for reference).
    NonElementary(Arc<Expr>),
    /// Partial integration succeeded for part of the integrand.
    ///
    /// `integrated` is the antiderivative of the portion that was handled;
    /// `remainder` is the un-integrated part.
    Partial {
        /// Antiderivative of the integrated portion.
        integrated: Arc<Expr>,
        /// The remaining un-integrated sub-expression.
        remainder: Arc<Expr>,
    },
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Integrate `expr` with respect to `var` using the Risch algorithm framework.
///
/// Routes the integrand to the appropriate sub-algorithm based on tower
/// analysis:
///
/// | Tower kind    | Sub-algorithm                          |
/// |---------------|----------------------------------------|
/// | Rational      | Hermite reduction + Rothstein-Trager   |
/// | Logarithmic   | [`logarithmic::integrate_logarithmic`] |
/// | Exponential   | [`exponential::integrate_exponential`] |
/// | Mixed         | Decomposition attempt                  |
pub fn risch_integrate(expr: &Arc<Expr>, var: SymbolId) -> IntegrationResult {
    let tower = build_tower(expr, var);

    match tower.kind {
        TowerKind::Rational => integrate_rational(expr, var),
        TowerKind::Logarithmic => logarithmic::integrate_logarithmic(expr, var),
        TowerKind::Exponential => exponential::integrate_exponential(expr, var),
        TowerKind::Mixed => integrate_mixed(expr, var),
    }
}

// ── Rational integration ──────────────────────────────────────────────────────

/// Integrate a pure rational function via polynomial power rule + Hermite
/// reduction + Rothstein-Trager.
///
/// Splits the integrand `A/D` into a polynomial part `P(x)` plus a proper
/// rational part `r(x)`, then integrates each component:
///
/// - Polynomial part: `∫Σ aₖxᵏ dx = Σ aₖxᵏ⁺¹/(k+1)` (power rule).
/// - Proper part: Hermite reduction gives the derivative-free rational
///   component; Rothstein-Trager gives the logarithmic component.
fn integrate_rational(expr: &Arc<Expr>, var: SymbolId) -> IntegrationResult {
    let Some(rf) = expr_to_rational_fn(expr, var) else {
        return IntegrationResult::Partial {
            integrated: Expr::int(0),
            remainder: expr.clone(),
        };
    };

    // Split into polynomial + proper rational parts.
    let (poly_part, proper) = rf.to_proper();
    let poly_integral = integrate_poly(&poly_part, var);

    // Hermite reduction on the proper fraction yields the derivative-free
    // rational antiderivative component; Rothstein-Trager yields the log part.
    let hermite = hermite_reduce(&proper);
    let rational_of_proper = rational_fn_to_expr(&hermite.rational_part, var);
    let log_integral = integrate_rational_log(&proper);

    let mut result = normalize::add(poly_integral, rational_of_proper);
    for term in &log_integral.terms {
        let arg_expr = poly_to_expr(&term.argument, var);
        let ln_arg = Expr::func(FuncId::Ln, vec![arg_expr]);
        let coeff_expr = big_rational_to_expr(&term.coeff);
        let log_term = normalize::mul(coeff_expr, ln_arg);
        result = normalize::add(result, log_term);
    }

    // Pipeline is constructive: polynomial power rule + Hermite + Rothstein-Trager
    // all produce exact antiderivatives by construction. No verify step — the
    // canonical form of `diff(result)` may not structurally match `expr` even
    // when the two are semantically equal (e.g., a sum of `ln` derivatives
    // versus the combined rational form of the integrand).
    IntegrationResult::Elementary(result)
}

/// Antiderivative of a dense polynomial, emitted as `Arc<Expr>` in `var`.
///
/// `∫ (a₀ + a₁x + … + aₙxⁿ) dx = a₀x + (a₁/2)x² + … + (aₙ/(n+1))xⁿ⁺¹`.
fn integrate_poly(poly: &DensePolynomial<BigRational>, var: SymbolId) -> Arc<Expr> {
    use num::traits::Zero;
    if poly.is_zero() {
        return Expr::int(0);
    }
    let x = Expr::symbol(&var.as_str());
    let mut terms: Vec<Arc<Expr>> = Vec::new();
    let deg = poly.degree().unwrap_or(0);
    for k in 0..=deg {
        let c = poly.coeff(k);
        if c.is_zero() {
            continue;
        }
        let new_k = (k + 1) as i64;
        let denom = BigRational::from_i64(new_k, 1);
        let new_coeff = &c / &denom;
        let coeff_expr = big_rational_to_expr(&new_coeff);
        let x_pow = if new_k == 1 {
            x.clone()
        } else {
            normalize::pow(x.clone(), Expr::int(new_k))
        };
        terms.push(normalize::mul(coeff_expr, x_pow));
    }
    if terms.is_empty() {
        Expr::int(0)
    } else {
        normalize::add_many(terms)
    }
}

// ── Mixed tower integration ───────────────────────────────────────────────────

/// Attempt integration of a mixed (log + exp) tower expression.
///
/// Tries logarithmic patterns first, then exponential. Returns `Partial`
/// with the full expression as remainder if neither succeeds.
fn integrate_mixed(expr: &Arc<Expr>, var: SymbolId) -> IntegrationResult {
    let log_result = logarithmic::integrate_logarithmic(expr, var);
    if matches!(log_result, IntegrationResult::Elementary(_)) {
        return log_result;
    }

    let exp_result = exponential::integrate_exponential(expr, var);
    if matches!(exp_result, IntegrationResult::Elementary(_)) {
        return exp_result;
    }

    IntegrationResult::Partial {
        integrated: Expr::int(0),
        remainder: expr.clone(),
    }
}

// ── Conversion helpers ────────────────────────────────────────────────────────

/// Try to represent `expr` as a `RationalFunction<BigRational>` in `var`.
///
/// Handles polynomial numerators, `Pow(p, n)` with negative integer exponents
/// (treated as `1/p^|n|`), and `Mul` nodes whose factors may mix positive and
/// negative integer exponents (including nested `Pow` factors). Returns `None`
/// when `expr` contains transcendental sub-expressions or non-integer exponents.
fn expr_to_rational_fn(expr: &Arc<Expr>, var: SymbolId) -> Option<RationalFunction<BigRational>> {
    // Fast path: pure polynomial numerator over denominator 1.
    if let Some(num) = expr_to_poly(expr, var) {
        return Some(RationalFunction::from_poly(num));
    }

    let one = DensePolynomial::constant(BigRational::from(1i64));
    let mut num = one.clone();
    let mut den = one.clone();

    accumulate_rational(expr, var, &mut num, &mut den)?;

    Some(RationalFunction::new(num, den))
}

/// Walk `expr` and accumulate its polynomial numerator and denominator into
/// `num` / `den`. Returns `None` on non-polynomial sub-expressions.
fn accumulate_rational(
    expr: &Arc<Expr>,
    var: SymbolId,
    num: &mut DensePolynomial<BigRational>,
    den: &mut DensePolynomial<BigRational>,
) -> Option<()> {
    match expr.as_ref() {
        Expr::Mul(node) => {
            // Fold coefficient into the numerator.
            *num = num.scale(&node.coeff);
            for (base, exp) in &node.factors {
                accumulate_pow(base, exp, var, num, den)?;
            }
            Some(())
        }
        Expr::Pow(base, exp) => accumulate_pow(base, exp, var, num, den),
        _ => {
            // Treat as polynomial^1
            let p = expr_to_poly(expr, var)?;
            *num = &*num * &p;
            Some(())
        }
    }
}

/// Accumulate a `base^exp` factor into `(num, den)`. Handles negative integer
/// exponents (including nested `Pow(Pow(...), k)` forms) by routing into the
/// denominator.
fn accumulate_pow(
    base: &Arc<Expr>,
    exp: &Arc<Expr>,
    var: SymbolId,
    num: &mut DensePolynomial<BigRational>,
    den: &mut DensePolynomial<BigRational>,
) -> Option<()> {
    let outer_n = const_int_exp(exp)?;

    // Unwrap nested Pow: (b^k)^m = b^(k*m). Walk as deep as needed.
    let mut cur_base = base.clone();
    let mut cur_n = outer_n;
    while let Expr::Pow(inner_base, inner_exp) = cur_base.as_ref() {
        let inner_n = const_int_exp(inner_exp)?;
        cur_n = cur_n.checked_mul(inner_n)?;
        cur_base = inner_base.clone();
    }

    let base_poly = expr_to_poly(&cur_base, var)?;
    let one = DensePolynomial::constant(BigRational::from(1i64));

    if cur_n >= 0 {
        let mut pw = one;
        for _ in 0..cur_n {
            pw = &pw * &base_poly;
        }
        *num = &*num * &pw;
    } else {
        let abs_n = (-cur_n) as usize;
        let mut pw = one;
        for _ in 0..abs_n {
            pw = &pw * &base_poly;
        }
        *den = &*den * &pw;
    }
    Some(())
}

/// Try to represent `expr` as a polynomial in `var` with `BigRational` coefficients.
fn expr_to_poly(expr: &Arc<Expr>, var: SymbolId) -> Option<DensePolynomial<BigRational>> {
    match expr.as_ref() {
        Expr::Integer(n) => Some(DensePolynomial::constant(BigRational::from_integer(
            n.clone(),
        ))),
        Expr::Rational(r) => Some(DensePolynomial::constant(r.clone())),
        Expr::Symbol(s) if *s == var => Some(DensePolynomial::from_coeffs(vec![
            BigRational::from(0i64),
            BigRational::from(1i64),
        ])),
        Expr::Add(node) => {
            let mut acc = DensePolynomial::constant(node.constant.clone());
            for (term, coeff) in &node.terms {
                let tp = expr_to_poly(term, var)?;
                let scaled = tp.scale(coeff);
                acc = &acc + &scaled;
            }
            Some(acc)
        }
        Expr::Mul(node) => {
            let mut acc = DensePolynomial::constant(node.coeff.clone());
            for (base, exp) in &node.factors {
                let n = const_int_exp(exp)?;
                if n < 0 {
                    return None;
                }
                let bp = expr_to_poly(base, var)?;
                let mut pw = DensePolynomial::constant(BigRational::from(1i64));
                for _ in 0..n {
                    pw = &pw * &bp;
                }
                acc = &acc * &pw;
            }
            Some(acc)
        }
        Expr::Pow(base, exp) => {
            let n = const_int_exp(exp)?;
            if n < 0 {
                return None;
            }
            let bp = expr_to_poly(base, var)?;
            let mut pw = DensePolynomial::constant(BigRational::from(1i64));
            for _ in 0..n {
                pw = &pw * &bp;
            }
            Some(pw)
        }
        _ => None,
    }
}

/// Extract a constant integer exponent from an `Expr`.
fn const_int_exp(exp: &Arc<Expr>) -> Option<i64> {
    match exp.as_ref() {
        Expr::Integer(n) => n.to_i64(),
        _ => None,
    }
}

/// Convert a `RationalFunction<BigRational>` back to an `Expr`.
fn rational_fn_to_expr(rf: &RationalFunction<BigRational>, var: SymbolId) -> Arc<Expr> {
    if rf.is_zero() {
        return Expr::int(0);
    }
    let num = poly_to_expr(rf.numerator(), var);
    let den = poly_to_expr(rf.denominator(), var);
    if den.is_one() {
        num
    } else {
        normalize::div(num, den)
    }
}

/// Convert a `DensePolynomial<BigRational>` to an `Expr` in variable `var`.
pub(crate) fn poly_to_expr(poly: &DensePolynomial<BigRational>, var: SymbolId) -> Arc<Expr> {
    use num::traits::Zero;
    if poly.is_zero() {
        return Expr::int(0);
    }
    let deg = poly.degree().unwrap_or(0);
    let x = Expr::symbol(&var.as_str());
    let mut terms: Vec<Arc<Expr>> = Vec::new();
    for i in 0..=deg {
        let c = poly.coeff(i);
        if c.is_zero() {
            continue;
        }
        let c_expr = big_rational_to_expr(&c);
        if i == 0 {
            terms.push(c_expr);
        } else {
            let x_pow = normalize::pow(x.clone(), Expr::int(i as i64));
            terms.push(normalize::mul(c_expr, x_pow));
        }
    }
    if terms.is_empty() {
        Expr::int(0)
    } else {
        normalize::add_many(terms)
    }
}

/// Convert a `BigRational` to the simplest `Arc<Expr>`.
pub(crate) fn big_rational_to_expr(r: &BigRational) -> Arc<Expr> {
    if r.is_integer() {
        if let Some(n) = r.numer().to_i64() {
            return Expr::int(n);
        }
    }
    Arc::new(Expr::Rational(r.clone()))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{normalize, Expr, FuncId, SmallInt, SymbolId};

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }
    fn xid(name: &str) -> SymbolId {
        SymbolId::intern(name)
    }

    // ── exp(x) → exp(x) ─────────────────────────────────────────────────────

    #[test]
    fn test_risch_exp_x_elementary() {
        let x = sym("ri_exp_x");
        let e = Expr::func(FuncId::Exp, vec![x]);
        let result = risch_integrate(&e, xid("ri_exp_x"));
        assert!(
            matches!(result, IntegrationResult::Elementary(_)),
            "∫exp(x)dx must be Elementary"
        );
    }

    #[test]
    fn test_risch_exp_x_value() {
        let x = sym("ri_expv_x");
        let e = Expr::func(FuncId::Exp, vec![x.clone()]);
        if let IntegrationResult::Elementary(r) = risch_integrate(&e, xid("ri_expv_x")) {
            assert_eq!(r.as_ref(), e.as_ref(), "∫exp(x)dx = exp(x)");
        }
    }

    // ── ln(x) → x·ln(x) − x ─────────────────────────────────────────────────

    #[test]
    fn test_risch_ln_x_elementary() {
        let x = sym("ri_ln_x");
        let e = Expr::func(FuncId::Ln, vec![x]);
        let result = risch_integrate(&e, xid("ri_ln_x"));
        assert!(
            matches!(result, IntegrationResult::Elementary(_)),
            "∫ln(x)dx must be Elementary"
        );
    }

    // ── exp(-x^2) → NonElementary ────────────────────────────────────────────

    #[test]
    fn test_risch_gaussian_non_elementary() {
        let x = sym("ri_gauss_x");
        let x_sq = normalize::pow(x.clone(), Expr::int(2));
        let neg_x_sq = normalize::neg(x_sq);
        let e = Expr::func(FuncId::Exp, vec![neg_x_sq]);
        let result = risch_integrate(&e, xid("ri_gauss_x"));
        assert!(
            matches!(result, IntegrationResult::NonElementary(_)),
            "∫exp(-x^2)dx must be NonElementary"
        );
    }

    // ── x·exp(x) not non-elementary ─────────────────────────────────────────

    #[test]
    fn test_risch_x_exp_x_not_non_elementary() {
        let x = sym("ri_xe_x");
        let exp_x = Expr::func(FuncId::Exp, vec![x.clone()]);
        let e = normalize::mul(x, exp_x);
        let result = risch_integrate(&e, xid("ri_xe_x"));
        assert!(
            !matches!(result, IntegrationResult::NonElementary(_)),
            "∫x·exp(x)dx should not be NonElementary"
        );
    }

    // ── Tower routing ────────────────────────────────────────────────────────

    #[test]
    fn test_risch_1_over_x_is_rational_tower() {
        let x = sym("ri_1x_x");
        let e = normalize::div(Expr::int(1), x);
        let t = build_tower(&e, xid("ri_1x_x"));
        assert_eq!(t.kind, TowerKind::Rational);
    }

    #[test]
    fn test_risch_routes_log_tower() {
        let x = sym("ri_rt_x");
        let t = build_tower(&Expr::func(FuncId::Ln, vec![x]), xid("ri_rt_x"));
        assert_eq!(t.kind, TowerKind::Logarithmic);
    }

    #[test]
    fn test_risch_routes_exp_tower() {
        let x = sym("ri_re_x");
        let t = build_tower(&Expr::func(FuncId::Exp, vec![x]), xid("ri_re_x"));
        assert_eq!(t.kind, TowerKind::Exponential);
    }

    // ── IntegrationResult variants ───────────────────────────────────────────

    #[test]
    fn test_result_elementary_wraps_expr() {
        let r = IntegrationResult::Elementary(Expr::int(0));
        assert!(matches!(r, IntegrationResult::Elementary(_)));
    }

    #[test]
    fn test_result_non_elementary_wraps_expr() {
        let r = IntegrationResult::NonElementary(Expr::int(0));
        assert!(matches!(r, IntegrationResult::NonElementary(_)));
    }

    #[test]
    fn test_result_partial_fields() {
        let x = sym("irp_x");
        let y = sym("irp_y");
        let r = IntegrationResult::Partial {
            integrated: x.clone(),
            remainder: y.clone(),
        };
        if let IntegrationResult::Partial {
            integrated,
            remainder,
        } = r
        {
            assert_eq!(integrated.as_ref(), x.as_ref());
            assert_eq!(remainder.as_ref(), y.as_ref());
        } else {
            panic!("expected Partial");
        }
    }

    // ── poly_to_expr ─────────────────────────────────────────────────────────

    #[test]
    fn test_poly_to_expr_constant() {
        let p = DensePolynomial::constant(BigRational::from(5i64));
        let e = poly_to_expr(&p, xid("ptc_x"));
        assert_eq!(*e, Expr::Integer(SmallInt::from(5i64)));
    }

    #[test]
    fn test_poly_to_expr_linear() {
        // 2x + 3
        let p =
            DensePolynomial::from_coeffs(vec![BigRational::from(3i64), BigRational::from(2i64)]);
        let e = poly_to_expr(&p, xid("ptl_x"));
        assert!(!e.is_zero());
    }

    // ── expr_to_poly ─────────────────────────────────────────────────────────

    #[test]
    fn test_expr_to_poly_integer() {
        let e = Expr::int(7);
        let p = expr_to_poly(&e, xid("etp_x")).unwrap();
        assert_eq!(p.coeff(0), BigRational::from(7i64));
    }

    #[test]
    fn test_expr_to_poly_symbol() {
        let x = sym("etps_x");
        let p = expr_to_poly(&x, xid("etps_x")).unwrap();
        assert_eq!(p.degree(), Some(1));
    }

    #[test]
    fn test_expr_to_poly_transcendental_returns_none() {
        let x = sym("etpt_x");
        let ln_x = Expr::func(FuncId::Ln, vec![x]);
        assert!(expr_to_poly(&ln_x, xid("etpt_x")).is_none());
    }
}
