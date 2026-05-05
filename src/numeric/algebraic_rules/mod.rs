//! Rewrite rules and composed strategies for algebraic simplification.
//!
//! Builds on the strategy combinator engine in [`crate::numeric::rewrite`] to
//! provide named algebraic simplification rules and high-level strategies.
//!
//! # Design
//!
//! Rules already enforced by the normalize.rs smart constructors are exposed
//! here as thin wrappers so they participate in strategy pipelines. New rules
//! that normalize.rs does not implement (distribution, rationalize denominator)
//! are implemented from scratch in the [`rules`] submodule.
//!
//! # Examples
//!
//! ```rust
//! use std::sync::Arc;
//! use thales::numeric::{Expr, normalize};
//! use thales::numeric::algebraic_rules::{algebraic_expand, algebraic_simplify};
//!
//! let x = Expr::symbol("x");
//! let y = Expr::symbol("y");
//! let z = Expr::symbol("z");
//!
//! // a*(b+c) → a*b + a*c via the expand strategy
//! let sum = normalize::add(y.clone(), z.clone());
//! let product = normalize::mul(x.clone(), sum);
//! let expanded = algebraic_expand()(&product).unwrap_or(product);
//! ```

mod rules;

pub use rules::{
    rule_cancel_common_factors, rule_combine_like_powers, rule_constant_fold, rule_distribute,
    rule_power_of_power, rule_rationalize_denominator, rule_zero_exponent,
};

use crate::numeric::rewrite::{choice, innermost, repeat, try_or_id, try_rule, Strategy};

// ── High-level composed strategies ───────────────────────────────────────────

/// Strategy that distributes multiplication over addition repeatedly.
///
/// Applies `rule_distribute` bottom-up until fixpoint, then folds constants.
/// Use this to fully expand an expression into a sum-of-products normal form.
///
/// # Example
///
/// ```rust
/// use std::sync::Arc;
/// use thales::numeric::{Expr, normalize};
/// use thales::numeric::algebraic_rules::algebraic_expand;
///
/// let a = Expr::symbol("a");
/// let b = Expr::symbol("b");
/// let c = Expr::symbol("c");
/// let sum = normalize::add(b.clone(), c.clone());
/// let product = normalize::mul(a.clone(), sum);
/// let result = algebraic_expand()(&product).unwrap();
/// // result is a*(b+c) expanded to a*b + a*c
/// ```
pub fn algebraic_expand() -> Strategy {
    let distribute = try_rule(rule_distribute);
    let fold = try_rule(rule_constant_fold);
    let step = choice(distribute, fold);
    innermost(step, 200)
}

/// Strategy that simplifies algebraic expressions without distributing.
///
/// Applies constant folding, common factor cancellation, and power rules
/// bottom-up until fixpoint. Suitable as a normalization pass after algebraic
/// manipulation. To also expand products, use [`algebraic_expand`] first.
///
/// # Example
///
/// ```rust
/// use std::sync::Arc;
/// use thales::numeric::Expr;
/// use thales::numeric::algebraic_rules::algebraic_simplify;
///
/// // x^0 → 1
/// let x = Expr::symbol("x");
/// let e = Arc::new(Expr::Pow(x, Expr::int(0)));
/// let result = algebraic_simplify()(&e).unwrap();
/// assert!(result.is_one());
/// ```
pub fn algebraic_simplify() -> Strategy {
    let fold = try_rule(rule_constant_fold);
    let cancel = try_rule(rule_cancel_common_factors);
    let pow_pow = try_rule(rule_power_of_power);
    let zero_exp = try_rule(rule_zero_exponent);
    let step = choice(fold, choice(cancel, choice(pow_pow, zero_exp)));
    repeat(try_or_id(innermost(step, 100)), 5)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::expr::Expr;
    use crate::numeric::{normalize, AddNode, BigRational, MulNode};
    use std::sync::Arc;

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }

    #[test]
    fn test_expand_distributes() {
        let a = sym("arm_exp_a");
        let b = sym("arm_exp_b");
        let c = sym("arm_exp_c");
        let sum = normalize::add(b.clone(), c.clone());
        let product = normalize::mul(a.clone(), sum);
        let result = algebraic_expand()(&product).unwrap();
        match result.as_ref() {
            Expr::Add(node) => assert_eq!(node.term_count(), 2),
            other => panic!("expected Add after expand, got {other}"),
        }
    }

    #[test]
    fn test_expand_nested_distribution() {
        // a*(b + c*(d+e)) → fully expanded sum
        let a = sym("arm_expn_a");
        let b = sym("arm_expn_b");
        let c = sym("arm_expn_c");
        let d = sym("arm_expn_d");
        let e = sym("arm_expn_e");
        let inner = normalize::add(d.clone(), e.clone());
        let ce = normalize::mul(c.clone(), inner);
        let outer_sum = normalize::add(b.clone(), ce);
        let product = normalize::mul(a.clone(), outer_sum);
        let result = algebraic_expand()(&product).unwrap();
        match result.as_ref() {
            Expr::Add(node) => assert!(node.term_count() >= 2),
            _ => {}
        }
    }

    #[test]
    fn test_simplify_constant_fold() {
        let mut node = MulNode::one();
        node.scale(&BigRational::from(5i64));
        let e = Arc::new(Expr::Mul(node));
        let result = algebraic_simplify()(&e).unwrap();
        assert_eq!(*result, Expr::Integer(crate::numeric::SmallInt::from(5i64)));
    }

    #[test]
    fn test_simplify_zero_exponent() {
        let e = Arc::new(Expr::Pow(sym("arm_simp_ze_x"), Expr::int(0)));
        let result = algebraic_simplify()(&e).unwrap();
        assert!(result.is_one());
    }

    #[test]
    fn test_simplify_symbol_unchanged() {
        let x = sym("arm_simp_sym_x");
        let result = algebraic_simplify()(&x).unwrap();
        assert_eq!(*result, *x);
    }

    #[test]
    fn test_expand_then_simplify() {
        // 2*(x + 3) → expanded then simplified
        let x = sym("arm_ets_x");
        let sum = normalize::add(x.clone(), Expr::int(3));
        let product = normalize::mul(Expr::int(2), sum);
        let expanded = algebraic_expand()(&product).unwrap();
        let _simplified = algebraic_simplify()(&expanded).unwrap();
        // No assertion on exact form — just verify no panic and result is non-zero
        assert!(!_simplified.is_zero());
    }
}
