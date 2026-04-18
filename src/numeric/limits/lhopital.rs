//! L'Hôpital's rule for 0/0 and ∞/∞ indeterminate forms.
//!
//! Extracts the numerator and denominator from a ratio-shaped expression
//! (either `Mul` with negative-exponent factors or a bare `Pow` with a -1
//! exponent), then differentiates both sides repeatedly until a definite
//! value emerges or [`LHOPITAL_MAX_ITER`] iterations have passed.

use std::sync::Arc;

use num::traits::One;

use super::super::differentiation::diff_arc;
use super::super::expr::Expr;
use super::super::series::taylor::substitute;
use super::super::{normalize, BigRational, SymbolId};
use super::evaluation::eval_float;
use super::helpers::{expr_to_f64, LHOPITAL_MAX_ITER};
use super::LimitResult;

pub(crate) fn try_lhopital(
    expr: &Arc<Expr>,
    var: SymbolId,
    point: &Arc<Expr>,
) -> Option<LimitResult> {
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
            if let (Some(_n), Some(d)) = (num_f, den_f) {
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
            (Some(_n), Some(d)) if !d.is_nan() && d.abs() > 1e-10 => {
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

/// If `expr` looks like a ratio (Mul with a factor raised to -1, or a bare
/// `Pow(base, -1)`), extract the numerator and denominator for L'Hôpital.
fn extract_ratio(expr: &Arc<Expr>) -> Option<(Arc<Expr>, Arc<Expr>)> {
    match expr.as_ref() {
        Expr::Mul(node) => {
            let mut num_factors: Vec<(Arc<Expr>, Arc<Expr>)> = Vec::new();
            let mut den_factors: Vec<(Arc<Expr>, Arc<Expr>)> = Vec::new();

            for (base, exp) in &node.factors {
                // Compute the effective exponent by unwrapping `Pow(p, k)` as
                // base with exp=1 into `(p, k)`. This handles canonical forms
                // like `Pow(x, -2)` stored as a factor with exp=1.
                let (effective_base, effective_exp) = match (base.as_ref(), exp.as_ref()) {
                    (Expr::Pow(inner_base, inner_exp), _) if exp.is_one() => {
                        (inner_base.clone(), inner_exp.clone())
                    }
                    _ => (base.clone(), exp.clone()),
                };

                if is_negative_one(&effective_exp) {
                    den_factors.push((effective_base, Expr::int(1)));
                } else if let Expr::Integer(n) = effective_exp.as_ref() {
                    if let Some(v) = n.to_i64() {
                        if v < 0 {
                            den_factors.push((effective_base, Expr::int(-v)));
                        } else {
                            num_factors.push((effective_base, effective_exp));
                        }
                    } else {
                        num_factors.push((effective_base, effective_exp));
                    }
                } else {
                    num_factors.push((effective_base, effective_exp));
                }
            }

            if den_factors.is_empty() {
                return None;
            }

            let num = build_product(&node.coeff, &num_factors);
            let den = build_product(&BigRational::from_i64(1, 1), &den_factors);
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
            r.numer().to_i64().map(|v| v == -1).unwrap_or(false) && r.denom().is_one()
        }
        _ => false,
    }
}

fn build_product(coeff: &BigRational, factors: &[(Arc<Expr>, Arc<Expr>)]) -> Arc<Expr> {
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
