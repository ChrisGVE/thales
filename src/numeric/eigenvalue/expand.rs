//! Polynomial expansion helpers: distribute products over sums.

use super::super::big_rational::BigRational;
use super::super::expr::Expr;
use super::super::normalize;
use num::traits::{One, Zero};
use std::sync::Arc;

/// Fully expand an expression by distributing products over sums.
///
/// Ensures the result is a sum of products (monomials) suitable for
/// polynomial coefficient extraction.
pub(super) fn expand_expr(expr: &Arc<Expr>) -> Arc<Expr> {
    match expr.as_ref() {
        Expr::Integer(_) | Expr::Rational(_) | Expr::Float(_) | Expr::Symbol(_) => expr.clone(),
        Expr::Pow(base, exp) => {
            let base_exp = expand_expr(base);
            let exp_exp = expand_expr(exp);
            // If base is an Add and exponent is small integer, expand
            if let Expr::Integer(n) = exp_exp.as_ref() {
                if let Some(k) = n.to_i64() {
                    if k >= 2 && k <= 4 {
                        let mut result = base_exp.clone();
                        for _ in 1..k {
                            result = expand_mul(result, base_exp.clone());
                        }
                        return result;
                    }
                }
            }
            normalize::pow(base_exp, exp_exp)
        }
        Expr::Add(node) => {
            let mut terms = Vec::new();
            if !node.constant.is_zero() {
                terms.push(rational_to_expr(&node.constant));
            }
            for (term, coeff) in &node.terms {
                let expanded = expand_expr(term);
                terms.push(normalize::mul(rational_to_expr(coeff), expanded));
            }
            normalize::add_many(terms)
        }
        Expr::Mul(node) => {
            let mut result = rational_to_expr(&node.coeff);
            for (base, exp) in &node.factors {
                let factor = expand_expr(&normalize::pow(base.clone(), exp.clone()));
                result = expand_mul(result, factor);
            }
            result
        }
        Expr::Func(_, _) => expr.clone(),
    }
}

/// Multiply two expressions distributing over sums.
pub(super) fn expand_mul(a: Arc<Expr>, b: Arc<Expr>) -> Arc<Expr> {
    let a_terms = to_sum_terms(&a);
    let b_terms = to_sum_terms(&b);
    let mut result_terms = Vec::new();
    for at in &a_terms {
        for bt in &b_terms {
            result_terms.push(normalize::mul(at.clone(), bt.clone()));
        }
    }
    normalize::add_many(result_terms)
}

/// Decompose an expression into additive terms.
pub(super) fn to_sum_terms(expr: &Arc<Expr>) -> Vec<Arc<Expr>> {
    match expr.as_ref() {
        Expr::Add(node) => {
            let mut terms = Vec::new();
            if !node.constant.is_zero() {
                terms.push(rational_to_expr(&node.constant));
            }
            for (term, coeff) in &node.terms {
                if coeff.is_one() {
                    terms.push(term.clone());
                } else {
                    terms.push(normalize::mul(rational_to_expr(coeff), term.clone()));
                }
            }
            terms
        }
        _ => vec![expr.clone()],
    }
}

/// Convert a `BigRational` to an `Expr`.
pub(super) fn rational_to_expr(r: &BigRational) -> Arc<Expr> {
    if r.is_integer() {
        if let Some(n) = r.numer().to_i64() {
            return Expr::int(n);
        }
    }
    Arc::new(Expr::Rational(r.clone()))
}
