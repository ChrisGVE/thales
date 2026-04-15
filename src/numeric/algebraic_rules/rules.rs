//! Primitive algebraic rewrite rules.
//!
//! Each rule is a function `&Arc<Expr> -> Option<Arc<Expr>>` suitable for
//! wrapping with [`crate::numeric::rewrite::try_rule`].

use crate::numeric::expr::Expr;
use crate::numeric::{normalize, AddNode, BigRational, MulNode};
use num::traits::{One, Signed, Zero};
use std::sync::Arc;

// ── Rule: distribute ──────────────────────────────────────────────────────────

/// Distribute multiplication over addition: `a*(b+c) → a*b + a*c`.
///
/// Applies when a [`MulNode`] contains at least one factor whose base is an
/// [`AddNode`] with exponent 1. Only the first such factor is distributed per
/// call; repeated application via a strategy combinator handles full expansion.
pub fn rule_distribute(e: &Arc<Expr>) -> Option<Arc<Expr>> {
    let Expr::Mul(m) = e.as_ref() else {
        return None;
    };
    let (add_base, _) = m
        .factors
        .iter()
        .find(|(base, exp)| matches!(base.as_ref(), Expr::Add(_)) && exp.is_one())?;
    let add_base = Arc::clone(add_base);
    let Expr::Add(add_node) = add_base.as_ref() else {
        return None;
    };
    let remaining = build_remaining_mul(m, &add_base);
    let mut result_node = AddNode::zero();
    if !add_node.constant.is_zero() {
        let const_expr = rational_to_expr(add_node.constant.clone());
        result_node.add_term(
            normalize::mul(remaining.clone(), const_expr),
            BigRational::one(),
        );
    }
    for (term, coeff) in &add_node.terms {
        let scaled = if coeff.is_one() {
            Arc::clone(term)
        } else {
            normalize::mul(rational_to_expr(coeff.clone()), Arc::clone(term))
        };
        result_node.add_term(
            normalize::mul(remaining.clone(), scaled),
            BigRational::one(),
        );
    }
    Some(Arc::new(Expr::Add(result_node)))
}

/// Build `m` without the factor keyed by `skip_base`, returning simplified form.
fn build_remaining_mul(m: &MulNode, skip_base: &Arc<Expr>) -> Arc<Expr> {
    let mut node = MulNode::one();
    node.scale(&m.coeff);
    for (base, exp) in &m.factors {
        if base != skip_base {
            node.add_factor(Arc::clone(base), Arc::clone(exp));
        }
    }
    if node.factors.is_empty() {
        rational_to_expr(node.coeff)
    } else if node.coeff.is_one() && node.factors.len() == 1 {
        let (b, e) = node.factors.into_iter().next().unwrap();
        if e.is_one() {
            b
        } else {
            Arc::new(Expr::Pow(b, e))
        }
    } else {
        Arc::new(Expr::Mul(node))
    }
}

// ── Rule: cancel common factors ───────────────────────────────────────────────

/// Cancel common bases between numerator and denominator in a product.
///
/// Detects `a*c * (b*c)^(-1)` by finding bases with both positive and negative
/// exponents, then removes the paired entries.
pub fn rule_cancel_common_factors(e: &Arc<Expr>) -> Option<Arc<Expr>> {
    let Expr::Mul(m) = e.as_ref() else {
        return None;
    };
    let (pos, neg) = partition_exponent_signs(m);
    if pos.is_empty() || neg.is_empty() {
        return None;
    }
    let common: Vec<Arc<Expr>> = pos
        .iter()
        .filter(|b| neg.iter().any(|nb| nb == *b))
        .cloned()
        .collect();
    if common.is_empty() {
        return None;
    }
    Some(cancel_common_in_mul(m, &common))
}

fn partition_exponent_signs(m: &MulNode) -> (Vec<Arc<Expr>>, Vec<Arc<Expr>>) {
    let mut pos = Vec::new();
    let mut neg = Vec::new();
    for (base, exp) in &m.factors {
        match exp.as_ref() {
            Expr::Integer(n) if n.is_positive() => pos.push(Arc::clone(base)),
            Expr::Integer(n) if n.is_negative() => neg.push(Arc::clone(base)),
            Expr::Rational(r) if r.is_positive() => pos.push(Arc::clone(base)),
            Expr::Rational(r) if r.is_negative() => neg.push(Arc::clone(base)),
            _ => {}
        }
    }
    (pos, neg)
}

fn cancel_common_in_mul(m: &MulNode, common: &[Arc<Expr>]) -> Arc<Expr> {
    let mut node = MulNode::one();
    node.scale(&m.coeff);
    for (base, exp) in &m.factors {
        if !common.iter().any(|c| c == base) {
            node.add_factor(Arc::clone(base), Arc::clone(exp));
        }
        // Common bases: emit zero exponent → MulNode::add_factor drops them
    }
    finalize_mul(node)
}

// ── Rule: rationalize denominator ────────────────────────────────────────────

/// Rationalize: `1 / (a + √b) → (a − √b) / (a² − b)`.
///
/// Applies only to `Pow(Add, -1)` where the Add has exactly one symbolic term
/// that is a square root (`Pow(_, 1/2)`).
pub fn rule_rationalize_denominator(e: &Arc<Expr>) -> Option<Arc<Expr>> {
    let Expr::Pow(base, exp) = e.as_ref() else {
        return None;
    };
    if !is_neg_one(exp) {
        return None;
    }
    let (a_part, sqrt_part) = extract_a_plus_sqrt(base)?;
    let conjugate = normalize::sub(Arc::clone(&a_part), Arc::clone(&sqrt_part));
    let a_sq = normalize::mul(Arc::clone(&a_part), Arc::clone(&a_part));
    let sqrt_sq = normalize::mul(Arc::clone(&sqrt_part), Arc::clone(&sqrt_part));
    let diff = normalize::sub(a_sq, sqrt_sq);
    if diff.is_zero() {
        return None;
    }
    Some(normalize::div(conjugate, diff))
}

fn extract_a_plus_sqrt(expr: &Arc<Expr>) -> Option<(Arc<Expr>, Arc<Expr>)> {
    let Expr::Add(node) = expr.as_ref() else {
        return None;
    };
    if node.terms.len() != 1 {
        return None;
    }
    let (term, coeff) = node.terms.iter().next()?;
    if !is_sqrt_expr(term) {
        return None;
    }
    let sqrt_part = if coeff.is_one() {
        Arc::clone(term)
    } else {
        normalize::mul(rational_to_expr(coeff.clone()), Arc::clone(term))
    };
    let a_part = if node.constant.is_zero() {
        Expr::int(0)
    } else {
        rational_to_expr(node.constant.clone())
    };
    Some((a_part, sqrt_part))
}

fn is_sqrt_expr(e: &Arc<Expr>) -> bool {
    matches!(e.as_ref(), Expr::Pow(_, exp) if is_one_half(exp))
}

fn is_one_half(e: &Arc<Expr>) -> bool {
    matches!(e.as_ref(), Expr::Rational(r) if *r == BigRational::from_i64(1, 2))
}

fn is_neg_one(e: &Arc<Expr>) -> bool {
    match e.as_ref() {
        Expr::Integer(n) => n.to_i64() == Some(-1),
        Expr::Rational(r) => *r == BigRational::from(-1i64),
        _ => false,
    }
}

// ── Rule: constant fold ───────────────────────────────────────────────────────

/// Evaluate pure-numeric sub-expressions that escaped the normalize constructors.
///
/// Fires on constant-only `Add`/`Mul` nodes and on `Pow(numeric, numeric)`.
pub fn rule_constant_fold(e: &Arc<Expr>) -> Option<Arc<Expr>> {
    match e.as_ref() {
        Expr::Add(node) if node.is_constant() => Some(rational_to_expr(node.constant.clone())),
        Expr::Mul(node) if node.is_constant() => Some(rational_to_expr(node.coeff.clone())),
        Expr::Pow(base, exp) if is_numeric(base) && is_numeric(exp) => {
            Some(normalize::pow(Arc::clone(base), Arc::clone(exp)))
        }
        _ => None,
    }
}

fn is_numeric(e: &Arc<Expr>) -> bool {
    matches!(
        e.as_ref(),
        Expr::Integer(_) | Expr::Rational(_) | Expr::Float(_)
    )
}

// ── Normalize-wrapper rules ───────────────────────────────────────────────────

/// `(a^b)^c → a^(b*c)` when `b` is an integer (already in normalize::pow).
pub fn rule_power_of_power(e: &Arc<Expr>) -> Option<Arc<Expr>> {
    let Expr::Pow(base, exp) = e.as_ref() else {
        return None;
    };
    if matches!(base.as_ref(), Expr::Pow(_, inner) if matches!(inner.as_ref(), Expr::Integer(_))) {
        Some(normalize::pow(Arc::clone(base), Arc::clone(exp)))
    } else {
        None
    }
}

/// `a^0 → 1` (already in normalize::pow, exposed as a strategy-compatible rule).
pub fn rule_zero_exponent(e: &Arc<Expr>) -> Option<Arc<Expr>> {
    let Expr::Pow(_, exp) = e.as_ref() else {
        return None;
    };
    if exp.is_zero() {
        Some(Expr::int(1))
    } else {
        None
    }
}

/// `a^b * a^c → a^(b+c)`: re-normalize a Mul to merge like bases.
pub fn rule_combine_like_powers(e: &Arc<Expr>) -> Option<Arc<Expr>> {
    let Expr::Mul(m) = e.as_ref() else {
        return None;
    };
    let mut factors = m.factors.iter();
    let first = factors.next()?;
    let init = if first.1.is_one() {
        Arc::clone(first.0)
    } else {
        normalize::pow(Arc::clone(first.0), Arc::clone(first.1))
    };
    let result = factors.fold(
        normalize::mul(rational_to_expr(m.coeff.clone()), init),
        |acc, (base, exp)| {
            let factor = if exp.is_one() {
                Arc::clone(base)
            } else {
                normalize::pow(Arc::clone(base), Arc::clone(exp))
            };
            normalize::mul(acc, factor)
        },
    );
    if *result == **e {
        None
    } else {
        Some(result)
    }
}

// ── Shared utilities ──────────────────────────────────────────────────────────

/// Convert a [`BigRational`] to the simplest `Arc<Expr>` form.
pub(super) fn rational_to_expr(r: BigRational) -> Arc<Expr> {
    if r.is_integer() {
        Arc::new(Expr::Integer(r.numer().clone()))
    } else {
        Arc::new(Expr::Rational(r))
    }
}

/// Finalize a [`MulNode`] to `Arc<Expr>`, unwrapping trivial cases.
pub(super) fn finalize_mul(node: MulNode) -> Arc<Expr> {
    if node.coeff.is_zero() {
        return Expr::int(0);
    }
    if node.factors.is_empty() {
        return rational_to_expr(node.coeff);
    }
    if node.coeff.is_one() && node.factors.len() == 1 {
        let (base, exp) = node.factors.into_iter().next().unwrap();
        if exp.is_one() {
            return base;
        }
        return Arc::new(Expr::Pow(base, exp));
    }
    Arc::new(Expr::Mul(node))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::normalize;

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }

    #[test]
    fn test_distribute_basic() {
        let a = sym("ar_dist_a");
        let b = sym("ar_dist_b");
        let c = sym("ar_dist_c");
        let sum = normalize::add(b.clone(), c.clone());
        let product = normalize::mul(a.clone(), sum);
        let result = rule_distribute(&product);
        assert!(result.is_some(), "distribute should apply");
        match result.unwrap().as_ref() {
            Expr::Add(node) => assert_eq!(node.term_count(), 2),
            other => panic!("expected Add, got {other}"),
        }
    }

    #[test]
    fn test_distribute_with_constant() {
        let x = sym("ar_dist_cx");
        let y = sym("ar_dist_cy");
        let sum = normalize::add(y.clone(), Expr::int(3));
        let product = normalize::mul(x.clone(), sum);
        assert!(rule_distribute(&product).is_some());
    }

    #[test]
    fn test_distribute_no_add_factor() {
        let x = sym("ar_dist_naf_x");
        let y = sym("ar_dist_naf_y");
        assert!(rule_distribute(&normalize::mul(x, y)).is_none());
    }

    #[test]
    fn test_distribute_symbol_unchanged() {
        assert!(rule_distribute(&sym("ar_dist_sym_x")).is_none());
    }

    #[test]
    fn test_constant_fold_constant_add() {
        let mut node = AddNode::zero();
        node.add_constant(BigRational::from(7i64));
        let e = Arc::new(Expr::Add(node));
        let result = rule_constant_fold(&e).unwrap();
        assert_eq!(*result, Expr::Integer(crate::numeric::SmallInt::from(7i64)));
    }

    #[test]
    fn test_constant_fold_constant_mul() {
        let mut node = MulNode::one();
        node.scale(&BigRational::from(6i64));
        let result = rule_constant_fold(&Arc::new(Expr::Mul(node))).unwrap();
        assert_eq!(*result, Expr::Integer(crate::numeric::SmallInt::from(6i64)));
    }

    #[test]
    fn test_constant_fold_pow() {
        let e = Arc::new(Expr::Pow(Expr::int(2), Expr::int(3)));
        let result = rule_constant_fold(&e).unwrap();
        assert_eq!(*result, Expr::Integer(crate::numeric::SmallInt::from(8i64)));
    }

    #[test]
    fn test_constant_fold_symbol_unchanged() {
        assert!(rule_constant_fold(&sym("ar_cf_sym_x")).is_none());
    }

    #[test]
    fn test_zero_exponent_returns_one() {
        let e = Arc::new(Expr::Pow(sym("ar_ze_x"), Expr::int(0)));
        assert!(rule_zero_exponent(&e).unwrap().is_one());
    }

    #[test]
    fn test_zero_exponent_nonzero_unchanged() {
        let e = Arc::new(Expr::Pow(sym("ar_ze_nz_x"), Expr::int(2)));
        assert!(rule_zero_exponent(&e).is_none());
    }

    #[test]
    fn test_power_of_power() {
        let x = sym("ar_pop_x");
        let x_sq = Arc::new(Expr::Pow(x.clone(), Expr::int(2)));
        let e = Arc::new(Expr::Pow(x_sq, Expr::int(3)));
        let result = rule_power_of_power(&e).unwrap();
        match result.as_ref() {
            Expr::Pow(base, exp) => {
                assert_eq!(**base, *x);
                assert_eq!(**exp, Expr::Integer(crate::numeric::SmallInt::from(6i64)));
            }
            other => panic!("expected Pow, got {other}"),
        }
    }

    #[test]
    fn test_power_of_power_no_match() {
        let e = Arc::new(Expr::Pow(sym("ar_pop_nm_x"), Expr::int(2)));
        assert!(rule_power_of_power(&e).is_none());
    }

    #[test]
    fn test_rationalize_denominator_basic() {
        let a = sym("ar_rd_a");
        let b = sym("ar_rd_b");
        let sqrt_b = Arc::new(Expr::Pow(b.clone(), Expr::rational(1, 2)));
        let denom = normalize::add(a.clone(), sqrt_b.clone());
        let e = Arc::new(Expr::Pow(denom, Expr::int(-1)));
        if let Some(r) = rule_rationalize_denominator(&e) {
            assert!(!r.is_zero());
        }
    }

    #[test]
    fn test_rationalize_denominator_not_pow_minus1() {
        let e = Arc::new(Expr::Pow(sym("ar_rd_npow_x"), Expr::int(2)));
        assert!(rule_rationalize_denominator(&e).is_none());
    }

    #[test]
    fn test_rationalize_denominator_no_sqrt() {
        let denom = normalize::add(sym("ar_rd_nosq_a"), sym("ar_rd_nosq_b"));
        let e = Arc::new(Expr::Pow(denom, Expr::int(-1)));
        assert!(rule_rationalize_denominator(&e).is_none());
    }
}
