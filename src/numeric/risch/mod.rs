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

/// Integrate a pure rational function via Hermite reduction + Rothstein-Trager.
///
/// Converts the `Expr` to a `RationalFunction<BigRational>` if possible,
/// runs the full rational integration pipeline, and converts the result back
/// to `Expr` form.
fn integrate_rational(expr: &Arc<Expr>, var: SymbolId) -> IntegrationResult {
    match expr_to_rational_fn(expr, var) {
        Some(rf) => {
            let hermite = hermite_reduce(&rf);
            let log_integral = integrate_rational_log(&rf);

            // Build the antiderivative expression
            let mut result = rational_fn_to_expr(&hermite.rational_part, var);
            for term in &log_integral.terms {
                let arg_expr = poly_to_expr(&term.argument, var);
                let ln_arg = Expr::func(FuncId::Ln, vec![arg_expr]);
                let coeff_expr = big_rational_to_expr(&term.coeff);
                let log_term = normalize::mul(coeff_expr, ln_arg);
                result = normalize::add(result, log_term);
            }

            // Verify via differentiation
            let deriv = diff_arc(&result, var);
            if deriv.as_ref() == expr.as_ref() {
                IntegrationResult::Elementary(result)
            } else {
                IntegrationResult::Partial {
                    integrated: result,
                    remainder: expr.clone(),
                }
            }
        }
        None => IntegrationResult::Partial {
            integrated: Expr::int(0),
            remainder: expr.clone(),
        },
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
/// Returns `None` when the expression involves non-polynomial sub-expressions.
fn expr_to_rational_fn(expr: &Arc<Expr>, var: SymbolId) -> Option<RationalFunction<BigRational>> {
    let num = expr_to_poly(expr, var)?;
    Some(RationalFunction::from_poly(num))
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
