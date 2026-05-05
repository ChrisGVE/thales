//! Coefficient extraction: convert a symbolic polynomial in lambda to
//! a `DensePolynomial<BigRational>`.

use super::super::big_rational::BigRational;
use super::super::dense_poly::DensePolynomial;
use super::super::expr::Expr;
use super::super::symbol::SymbolId;
use num::traits::{One, Zero};
use std::sync::Arc;

/// Attempt to extract a `DensePolynomial<BigRational>` from a symbolic
/// expression that should be a polynomial in `lambda`.
///
/// Returns `None` if the expression contains symbols other than `lambda`.
pub(super) fn extract_rational_poly(
    expr: &Arc<Expr>,
    lambda: SymbolId,
    degree: usize,
) -> Option<DensePolynomial<BigRational>> {
    let mut coeffs = vec![BigRational::zero(); degree + 1];
    let one = BigRational::one();
    collect_poly_coeffs(expr, lambda, 0, &one, &mut coeffs)?;
    Some(DensePolynomial::from_coeffs(coeffs))
}

/// Recursively traverse `expr` and accumulate polynomial coefficients.
///
/// `lambda_power` tracks the current power of lambda contributed by the call
/// site; `scale` is the rational scalar multiplier from enclosing `Mul` nodes.
pub(crate) fn collect_poly_coeffs(
    expr: &Arc<Expr>,
    lambda: SymbolId,
    lambda_power: usize,
    scale: &BigRational,
    coeffs: &mut Vec<BigRational>,
) -> Option<()> {
    match expr.as_ref() {
        // Pure rational constant: add to coeff[lambda_power]
        Expr::Integer(n) => {
            if lambda_power >= coeffs.len() {
                return None; // degree too high
            }
            let val = BigRational::from_integer(n.clone()) * scale.clone();
            coeffs[lambda_power] = coeffs[lambda_power].clone() + val;
            Some(())
        }
        Expr::Rational(r) => {
            if lambda_power >= coeffs.len() {
                return None;
            }
            let val = r * scale;
            coeffs[lambda_power] = coeffs[lambda_power].clone() + val;
            Some(())
        }
        // Lambda symbol: contributes to power lambda_power + 1
        Expr::Symbol(s) if *s == lambda => {
            let new_power = lambda_power + 1;
            if new_power >= coeffs.len() {
                return None;
            }
            // coefficient is `scale * 1`
            coeffs[new_power] = coeffs[new_power].clone() + scale.clone();
            Some(())
        }
        // Any other symbol: this is a symbolic entry; cannot extract
        Expr::Symbol(_) => None,
        // Sum: process each term
        Expr::Add(node) => {
            let const_val = node.constant.clone() * scale.clone();
            if lambda_power >= coeffs.len() {
                return None;
            }
            coeffs[lambda_power] = coeffs[lambda_power].clone() + const_val;
            for (term, coeff) in &node.terms {
                let new_scale = coeff * scale;
                collect_poly_coeffs(term, lambda, lambda_power, &new_scale, coeffs)?;
            }
            Some(())
        }
        // Product: handle lambda^k * constant forms
        Expr::Mul(node) => {
            let new_scale = &node.coeff * scale;
            let mut lam_exp: usize = 0;
            for (base, exp) in &node.factors {
                match (base.as_ref(), exp.as_ref()) {
                    // lambda^integer_exponent
                    (Expr::Symbol(s), Expr::Integer(k)) if *s == lambda => {
                        let k_u = k.to_i64().and_then(|v| usize::try_from(v).ok())?;
                        lam_exp = lam_exp.checked_add(k_u)?;
                    }
                    // Non-lambda base: must be a numeric constant (exponent=1)
                    (_, Expr::Integer(k)) if k.to_i64() == Some(1) => {
                        match base.as_ref() {
                            Expr::Integer(_) | Expr::Rational(_) => {
                                // numeric base^1 is already folded into coeff
                            }
                            // Symbolic non-lambda: cannot extract
                            _ => return None,
                        }
                    }
                    _ => return None,
                }
            }
            let total_power = lambda_power + lam_exp;
            if total_power >= coeffs.len() {
                return None;
            }
            coeffs[total_power] = coeffs[total_power].clone() + new_scale;
            Some(())
        }
        // Pow(lambda, k) as a standalone node
        Expr::Pow(base, exp) => {
            if let (Expr::Symbol(s), Expr::Integer(k)) = (base.as_ref(), exp.as_ref()) {
                if *s == lambda {
                    let k_u = k.to_i64().and_then(|v| usize::try_from(v).ok())?;
                    let total_power = lambda_power + k_u;
                    if total_power >= coeffs.len() {
                        return None;
                    }
                    coeffs[total_power] = coeffs[total_power].clone() + scale.clone();
                    return Some(());
                }
            }
            None
        }
        _ => None,
    }
}
