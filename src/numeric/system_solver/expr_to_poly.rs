//! Convert symbolic [`Expr`] trees to [`MultivariatePolynomial`]s.
//!
//! Only handles polynomial expressions: integer/rational constants, symbols
//! (variables), sums via [`Expr::Add`], products via [`Expr::Mul`], and
//! non-negative integer powers via [`Expr::Pow`].
//!
//! Returns `None` for transcendental functions, floating-point values, or
//! fractional/negative exponents — anything that cannot be represented as a
//! polynomial with rational coefficients over a fixed variable set.

use crate::numeric::big_rational::BigRational;
use crate::numeric::expr::Expr;
use crate::numeric::multivariate_poly::MultivariatePolynomial;
use crate::numeric::ring::Ring;
use crate::numeric::SymbolId;
use std::sync::Arc;

type MP = MultivariatePolynomial<BigRational>;

/// Convert an [`Expr`] to a multivariate polynomial over the given variables.
///
/// Returns `None` if any non-polynomial sub-expression is encountered.
///
/// # Supported forms
///
/// - `Integer(n)`, `Rational(p/q)` → constant polynomial
/// - `Symbol(v)` where `v ∈ vars` → single-variable polynomial
/// - `Add(node)` → constant + sum of `coeff * term` terms
/// - `Mul(node)` → `coeff * Π(base ^ exp)` for integer exponents ≥ 0
/// - `Pow(base, Integer(n))` with `n ≥ 0` → repeated multiplication
///
/// # Example
///
/// ```
/// use thales::numeric::{expr_to_multipoly, Expr, SymbolId};
///
/// let x = SymbolId::intern("ep_x");
/// let y = SymbolId::intern("ep_y");
///
/// // Constant 5
/// let c5 = Expr::int(5);
/// let p = expr_to_multipoly(&c5, &[x, y]).unwrap();
/// assert!(p.is_constant());
///
/// // Symbol x
/// let xp = expr_to_multipoly(&Expr::symbol("ep_x"), &[x, y]).unwrap();
/// assert!(!xp.is_constant());
/// ```
pub fn expr_to_multipoly(e: &Arc<Expr>, vars: &[SymbolId]) -> Option<MP> {
    match e.as_ref() {
        Expr::Integer(n) => {
            let i = n.to_i64()?;
            Some(MP::constant(BigRational::from(i)))
        }
        Expr::Rational(r) => Some(MP::constant(r.clone())),
        Expr::Float(_) => None,
        Expr::Symbol(v) => {
            if vars.contains(v) {
                Some(MP::var(*v))
            } else {
                None
            }
        }
        Expr::Add(node) => {
            // AddNode: constant + Σ(coeff · term)
            let mut result = MP::constant(node.constant.clone());
            for (term, coeff) in &node.terms {
                let term_poly = expr_to_multipoly(term, vars)?;
                let scaled = term_poly.scale(coeff);
                result = &result + &scaled;
            }
            Some(result)
        }
        Expr::Mul(node) => {
            // MulNode: coeff · Π(base ^ exp)
            let mut result = MP::constant(node.coeff.clone());
            for (base, exp) in &node.factors {
                let n = extract_non_neg_int(exp)?;
                let base_poly = expr_to_multipoly(base, vars)?;
                let powered = poly_pow(&base_poly, n)?;
                result = &result * &powered;
            }
            Some(result)
        }
        Expr::Pow(base, exp) => {
            let n = extract_non_neg_int(exp)?;
            let base_poly = expr_to_multipoly(base, vars)?;
            poly_pow(&base_poly, n)
        }
        Expr::Func(_, _) => None,
    }
}

// ── Arithmetic helpers ────────────────────────────────────────────────────────

/// Raise a multivariate polynomial to a non-negative integer power.
fn poly_pow(p: &MP, n: u32) -> Option<MP> {
    let mut result = MP::constant(BigRational::one());
    for _ in 0..n {
        result = &result * p;
    }
    Some(result)
}

/// Extract a non-negative integer from an `Expr` node.
fn extract_non_neg_int(e: &Arc<Expr>) -> Option<u32> {
    match e.as_ref() {
        Expr::Integer(n) => {
            let i = n.to_i64()?;
            if i >= 0 {
                Some(i as u32)
            } else {
                None
            }
        }
        _ => None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{BigRational, Monomial, MultivariatePolynomial, SymbolId};

    fn x() -> SymbolId {
        SymbolId::intern("ep_x2")
    }

    fn y() -> SymbolId {
        SymbolId::intern("ep_y2")
    }

    fn vars() -> Vec<SymbolId> {
        vec![x(), y()]
    }

    #[test]
    fn test_integer_constant() {
        let e = Expr::int(5);
        let p = expr_to_multipoly(&e, &vars()).unwrap();
        assert!(p.is_constant());
        assert_eq!(p.constant_term(), BigRational::from(5_i64));
    }

    #[test]
    fn test_rational_constant() {
        let e = Expr::rational(1, 3);
        let p = expr_to_multipoly(&e, &vars()).unwrap();
        assert!(p.is_constant());
    }

    #[test]
    fn test_symbol() {
        let e = Expr::symbol("ep_x2");
        let p = expr_to_multipoly(&e, &vars()).unwrap();
        assert_eq!(p.coeff(&Monomial::var(x())), BigRational::from(1_i64));
    }

    #[test]
    fn test_unknown_symbol_returns_none() {
        let e = Expr::symbol("unknown_var_99");
        let result = expr_to_multipoly(&e, &vars());
        assert!(result.is_none());
    }

    #[test]
    fn test_pow_integer_exp() {
        // x^2 → polynomial with x^2 term coefficient = 1
        let e = Expr::pow(Expr::symbol("ep_x2"), Expr::int(2));
        let p = expr_to_multipoly(&e, &vars()).unwrap();
        assert_eq!(
            p.coeff(&Monomial::var_pow(x(), 2)),
            BigRational::from(1_i64)
        );
    }

    #[test]
    fn test_pow_zero() {
        // x^0 = 1
        let e = Expr::pow(Expr::symbol("ep_x2"), Expr::int(0));
        let p = expr_to_multipoly(&e, &vars()).unwrap();
        assert!(p.is_constant());
        assert_eq!(p.constant_term(), BigRational::from(1_i64));
    }

    #[test]
    fn test_float_returns_none() {
        let e = Expr::float(1.5);
        assert!(expr_to_multipoly(&e, &vars()).is_none());
    }

    #[test]
    fn test_negative_pow_returns_none() {
        // x^(-1) is not a polynomial
        let e = Expr::pow(Expr::symbol("ep_x2"), Expr::int(-1));
        assert!(expr_to_multipoly(&e, &vars()).is_none());
    }

    #[test]
    fn test_func_returns_none() {
        use crate::numeric::expr::FuncId;
        let e = Expr::func(FuncId::Sin, vec![Expr::symbol("ep_x2")]);
        assert!(expr_to_multipoly(&e, &vars()).is_none());
    }
}
