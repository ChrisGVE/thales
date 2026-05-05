//! Coefficient extraction, polynomial-degree inspection, and small numeric
//! utilities.
//!
//! All internal logic operates on `Arc<Expr>`. The two boundary helpers
//! (`evaluate_constants`, `simplify_numeric_expression`) produce `Expression`
//! values for the public `Solution` output types.

use std::collections::HashMap;
use std::sync::Arc;

use num::traits::Zero;

use crate::ast::Expression;
use crate::numeric::{BigRational, Expr, SymbolId};

use super::detection::{contains_symbol, has_any_variable};

// ── Boundary helpers (produce Expression for Solution output) ────────────────

/// Evaluate constant expressions to their numeric values.
///
/// Simplifies first; if nothing depends on a variable, evaluates to a
/// literal. Near-integer results collapse to `Expression::Integer`.
pub(crate) fn evaluate_constants(expr: &Expression) -> Expression {
    let simplified = expr.simplify();

    if !has_any_variable(&simplified) {
        if let Some(value) = simplified.evaluate(&HashMap::new()) {
            if value.fract().abs() < 1e-10 {
                return Expression::Integer(value.round() as i64);
            } else {
                return Expression::Float(value);
            }
        }
    }

    simplified
}

/// Simplify a numeric value to the best Expression representation.
pub(crate) fn simplify_numeric_expression(val: f64) -> Expression {
    let rounded = val.round();
    if (val - rounded).abs() < 1e-10 && rounded.abs() < i64::MAX as f64 {
        Expression::Integer(rounded as i64)
    } else {
        Expression::Float(val)
    }
}

// ── Arc<Expr> helpers ────────────────────────────────────────────────────────

/// Get the polynomial degree of `expr` with respect to `var`.
///
/// Walks the normalized tree and reports the polynomial degree of
/// `expr` with respect to `var`. Returns `0` when `expr` is constant in
/// `var`, and is conservative (returns `0`) when a shape is outside the
/// polynomial form — callers should gate on
/// [`super::detection::is_polynomial_expr`] first.
pub(crate) fn get_polynomial_degree_expr(expr: &Arc<Expr>, var: SymbolId) -> usize {
    match expr.as_ref() {
        Expr::Integer(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Complex(_)
        | Expr::Constant(_) => 0,
        Expr::Symbol(s) => {
            if *s == var {
                1
            } else {
                0
            }
        }
        Expr::Add(node) => {
            let mut best = 0usize;
            for (term, _coeff) in &node.terms {
                let d = get_polynomial_degree_expr(term, var);
                if d > best {
                    best = d;
                }
            }
            best
        }
        Expr::Mul(node) => {
            let mut total = 0usize;
            for (base, exp) in &node.factors {
                let base_deg = get_polynomial_degree_expr(base, var);
                if base_deg == 0 {
                    continue;
                }
                let exp_val = match exp.as_ref() {
                    Expr::Integer(n) => n.to_i64().filter(|v| *v >= 0).map(|v| v as usize),
                    _ => None,
                };
                total += base_deg * exp_val.unwrap_or(0);
            }
            total
        }
        Expr::Pow(base, exp) => {
            let base_deg = get_polynomial_degree_expr(base, var);
            if base_deg == 0 {
                return 0;
            }
            match exp.as_ref() {
                Expr::Integer(n) => n
                    .to_i64()
                    .filter(|v| *v >= 0)
                    .map(|v| base_deg * (v as usize))
                    .unwrap_or(0),
                _ => 0,
            }
        }
        Expr::Func(_, _) => 0,
    }
}

/// Extract coefficients `(a, b, c)` such that `expr == a·var² + b·var + c`,
/// using exact `BigRational` coefficients.
///
/// Returns `(a, b, c)` for polynomial `expr` of degree ≤ 2. Non-polynomial
/// terms and shapes outside the expected form contribute nothing — callers
/// should pre-filter with [`get_polynomial_degree_expr`].
pub(crate) fn extract_quadratic_coefficients_expr(
    expr: &Arc<Expr>,
    var: SymbolId,
) -> (BigRational, BigRational, BigRational) {
    let zero = BigRational::zero();
    let mut a = zero.clone();
    let mut b = zero.clone();
    let mut c = zero.clone();
    accumulate_quad(expr, var, &BigRational::from(1), &mut a, &mut b, &mut c);
    (a, b, c)
}

fn accumulate_quad(
    expr: &Arc<Expr>,
    var: SymbolId,
    mult: &BigRational,
    a: &mut BigRational,
    b: &mut BigRational,
    c: &mut BigRational,
) {
    match expr.as_ref() {
        Expr::Integer(n) => {
            if let Some(v) = n.to_i64() {
                *c = &*c + &(mult * &BigRational::from(v));
            }
        }
        Expr::Rational(r) => {
            *c = &*c + &(mult * r);
        }
        Expr::Symbol(s) => {
            if *s == var {
                *b = &*b + mult;
            }
        }
        Expr::Add(node) => {
            if !node.constant.is_zero() {
                *c = &*c + &(mult * &node.constant);
            }
            for (term, coeff) in &node.terms {
                let new_mult = mult * coeff;
                accumulate_quad(term, var, &new_mult, a, b, c);
            }
        }
        Expr::Mul(node) => {
            // Fold the canonical MulNode into (scalar * var^deg) where
            // `scalar` carries every non-var factor. Anything more exotic
            // (a factor containing `var` in a non-power position) is
            // outside the quadratic envelope.
            let mut scalar = node.coeff.clone();
            let mut var_deg: usize = 0;
            let mut bad = false;
            for (base, exp) in &node.factors {
                if bad {
                    break;
                }
                let base_has = contains_symbol(base, var);
                let exp_has = contains_symbol(exp, var);
                if !base_has && !exp_has {
                    // Non-var factor — collapse to BigRational if possible.
                    if let Some(rational) = mul_rational_factor(base, exp) {
                        scalar = &scalar * &rational;
                    } else {
                        bad = true;
                    }
                    continue;
                }
                if exp_has {
                    bad = true;
                    continue;
                }
                let factor_deg = match base.as_ref() {
                    Expr::Symbol(s) if *s == var => 1usize,
                    _ => {
                        // Non-symbol base containing var; not a simple quad term.
                        bad = true;
                        continue;
                    }
                };
                let exp_val = match exp.as_ref() {
                    Expr::Integer(n) => n.to_i64().filter(|v| *v >= 0).map(|v| v as usize),
                    _ => None,
                };
                match exp_val {
                    Some(k) => var_deg += factor_deg * k,
                    None => bad = true,
                }
            }
            if bad {
                return;
            }
            let contribution = mult * &scalar;
            match var_deg {
                0 => *c = &*c + &contribution,
                1 => *b = &*b + &contribution,
                2 => *a = &*a + &contribution,
                _ => {}
            }
        }
        Expr::Pow(base, exp) => {
            let base_is_var = matches!(base.as_ref(), Expr::Symbol(s) if *s == var);
            if !base_is_var {
                return;
            }
            if contains_symbol(exp, var) {
                return;
            }
            let k = match exp.as_ref() {
                Expr::Integer(n) => n.to_i64().filter(|v| *v >= 0).map(|v| v as usize),
                _ => None,
            };
            match k {
                Some(1) => *b = &*b + mult,
                Some(2) => *a = &*a + mult,
                Some(0) => *c = &*c + mult,
                _ => {}
            }
        }
        _ => {}
    }
}

/// Multiply `base ^ exp` into a `BigRational` when both are exactly
/// representable — used during quadratic factor folding.
fn mul_rational_factor(base: &Arc<Expr>, exp: &Arc<Expr>) -> Option<BigRational> {
    let base_rat = match base.as_ref() {
        Expr::Integer(n) => BigRational::from(n.to_i64()?),
        Expr::Rational(r) => r.clone(),
        _ => return None,
    };
    let k = match exp.as_ref() {
        Expr::Integer(n) => n.to_i64()?,
        _ => return None,
    };
    if k == 0 {
        return Some(BigRational::from(1));
    }
    let positive_k = k.unsigned_abs() as usize;
    let mut acc = BigRational::from(1);
    for _ in 0..positive_k {
        acc = &acc * &base_rat;
    }
    if k < 0 {
        if acc.is_zero() {
            return None;
        }
        // 1/acc
        let one = BigRational::from(1);
        acc = &one / &acc;
    }
    Some(acc)
}
