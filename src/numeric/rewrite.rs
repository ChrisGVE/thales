//! Strategy combinator rewriting engine for expression simplification.
//!
//! Provides a composable system of rewrite strategies that can be combined
//! to form complex simplification pipelines. Rules are leaf transformations;
//! strategies are composed transformations.
//!
//! # Design
//!
//! Both [`Rule`] and [`Strategy`] are type aliases for boxed closures with the
//! same signature. The distinction is conceptual: rules are primitive, strategies
//! compose rules via combinators.
//!
//! # Examples
//!
//! ```rust
//! use std::sync::Arc;
//! use thales::numeric::{Expr, normalize};
//! use thales::numeric::rewrite::{try_rule, innermost};
//! use num::traits::{Zero, One};
//!
//! // Rule: x + 0 → x
//! let rule = try_rule(|e| {
//!     if let Expr::Add(node) = e.as_ref() {
//!         if node.term_count() == 1 && node.constant.is_zero() {
//!             let (term, coeff) = node.terms.iter().next().unwrap();
//!             if coeff.is_one() { return Some(term.clone()); }
//!         }
//!     }
//!     None
//! });
//! let strategy = innermost(rule, 100);
//! let x = Expr::symbol("x");
//! let expr = normalize::add(x.clone(), Expr::int(0));
//! ```

use crate::numeric::expr::Expr;
use crate::numeric::{AddNode, MulNode};
use std::sync::Arc;

// ── Type aliases ─────────────────────────────────────────────────────────────

/// A rewrite rule: transforms an expression or returns `None` if not applicable.
pub type Rule = Box<dyn Fn(&Arc<Expr>) -> Option<Arc<Expr>> + Send + Sync>;

/// A rewrite strategy: a composed transformation on expressions.
pub type Strategy = Box<dyn Fn(&Arc<Expr>) -> Option<Arc<Expr>> + Send + Sync>;

// ── Primitive strategies ──────────────────────────────────────────────────────

/// Wrap a rule function as a strategy.
///
/// The resulting strategy applies `rule` to the expression and returns
/// `Some(result)` on success or `None` if the rule does not apply.
pub fn try_rule(
    rule: impl Fn(&Arc<Expr>) -> Option<Arc<Expr>> + Send + Sync + 'static,
) -> Strategy {
    Box::new(move |e| rule(e))
}

/// Identity strategy: always succeeds, returning the input unchanged.
pub fn id() -> Strategy {
    Box::new(|e| Some(Arc::clone(e)))
}

/// Failing strategy: always returns `None`.
pub fn fail() -> Strategy {
    Box::new(|_e| None)
}

/// Apply `s` only when `pred` holds; otherwise return `None`.
pub fn guard(pred: impl Fn(&Arc<Expr>) -> bool + Send + Sync + 'static, s: Strategy) -> Strategy {
    Box::new(move |e| if pred(e) { s(e) } else { None })
}

// ── Sequential and choice combinators ────────────────────────────────────────

/// Apply `s1`, then apply `s2` to the result. Returns `None` if either fails.
pub fn sequence(s1: Strategy, s2: Strategy) -> Strategy {
    Box::new(move |e| {
        let mid = s1(e)?;
        s2(&mid)
    })
}

/// Try `s1`; if it returns `None`, try `s2`.
pub fn choice(s1: Strategy, s2: Strategy) -> Strategy {
    Box::new(move |e| s1(e).or_else(|| s2(e)))
}

// ── Iteration ─────────────────────────────────────────────────────────────────

/// Apply `s` repeatedly until it returns `None` or `max_iter` is reached.
///
/// Always returns `Some` — the accumulated result after successful applications,
/// or the original input if the strategy never applies.
pub fn repeat(s: Strategy, max_iter: usize) -> Strategy {
    Box::new(move |e| {
        let mut current = Arc::clone(e);
        for _ in 0..max_iter {
            match s(&current) {
                Some(next) => current = next,
                None => break,
            }
        }
        Some(current)
    })
}

/// Like `choice(s, id())`: apply `s`, or return input unchanged on failure.
pub fn try_or_id(s: Strategy) -> Strategy {
    Box::new(move |e| Some(s(e).unwrap_or_else(|| Arc::clone(e))))
}

// ── Child decomposition helpers ───────────────────────────────────────────────

/// Apply a strategy to all children of an `AddNode`, rebuilding if any changed.
///
/// Returns `None` if any child application fails; `Some(rebuilt)` on success.
fn apply_to_add_children(node: &AddNode, s: &Strategy) -> Option<Arc<Expr>> {
    let mut new_node = AddNode::zero();
    new_node.add_constant(node.constant.clone());

    for (term, coeff) in &node.terms {
        let new_term = s(term)?;
        new_node.add_term(new_term, coeff.clone());
    }
    Some(Arc::new(Expr::Add(new_node)))
}

/// Apply a strategy to all children of a `MulNode`, rebuilding if any changed.
///
/// Returns `None` if any child application fails; `Some(rebuilt)` on success.
fn apply_to_mul_children(node: &MulNode, s: &Strategy) -> Option<Arc<Expr>> {
    let mut new_node = MulNode::one();
    new_node.scale(&node.coeff);

    for (base, exp) in &node.factors {
        let new_base = s(base)?;
        let new_exp = s(exp)?;
        new_node.add_factor(new_base, new_exp);
    }
    Some(Arc::new(Expr::Mul(new_node)))
}

/// Apply a strategy to all children of a `Pow` node.
///
/// Returns `None` if either child application fails.
fn apply_to_pow_children(base: &Arc<Expr>, exp: &Arc<Expr>, s: &Strategy) -> Option<Arc<Expr>> {
    let new_base = s(base)?;
    let new_exp = s(exp)?;
    Some(Arc::new(Expr::Pow(new_base, new_exp)))
}

// ── Traversal combinators ─────────────────────────────────────────────────────

/// Apply `s` to all children of a compound expression.
///
/// Returns `None` if `s` fails on any child.
/// Leaf nodes (Integer, Rational, Float, Symbol) succeed vacuously.
pub fn all(s: Strategy) -> Strategy {
    Box::new(move |e| match e.as_ref() {
        Expr::Add(node) => apply_to_add_children(node, &s),
        Expr::Mul(node) => apply_to_mul_children(node, &s),
        Expr::Pow(base, exp) => apply_to_pow_children(base, exp, &s),
        // Leaves: no children — vacuously succeed, return input unchanged
        _ => Some(Arc::clone(e)),
    })
}

/// Apply `s` to the first child for which it succeeds.
///
/// Returns `None` if no child produces a successful application.
/// Leaf nodes (Integer, Rational, Float, Symbol) always return `None`.
pub fn one(s: Strategy) -> Strategy {
    Box::new(move |e| match e.as_ref() {
        Expr::Add(node) => one_add(node, &s),
        Expr::Mul(node) => one_mul(node, &s),
        Expr::Pow(base, exp) => one_pow(base, exp, &s),
        // Leaves: no children to apply to
        _ => None,
    })
}

/// Try `s` on each term of an `AddNode`, replacing the first success.
fn one_add(node: &AddNode, s: &Strategy) -> Option<Arc<Expr>> {
    let terms: Vec<_> = node.terms.iter().collect();
    for (idx, (term, coeff)) in terms.iter().enumerate() {
        if let Some(new_term) = s(term) {
            let mut new_node = AddNode::zero();
            new_node.add_constant(node.constant.clone());
            for (i, (t, c)) in terms.iter().enumerate() {
                if i == idx {
                    new_node.add_term(new_term.clone(), (*c).clone());
                } else {
                    new_node.add_term((*t).clone(), (*c).clone());
                }
            }
            // Suppress unused warning — coeff is part of the iteration pattern
            let _ = coeff;
            return Some(Arc::new(Expr::Add(new_node)));
        }
    }
    None
}

/// Try `s` on each base/exp of a `MulNode`, replacing the first success.
fn one_mul(node: &MulNode, s: &Strategy) -> Option<Arc<Expr>> {
    let factors: Vec<_> = node.factors.iter().collect();
    for (idx, (base, exp)) in factors.iter().enumerate() {
        // Try base first
        if let Some(new_base) = s(base) {
            let mut new_node = MulNode::one();
            new_node.scale(&node.coeff);
            for (i, (b, e)) in factors.iter().enumerate() {
                if i == idx {
                    new_node.add_factor(new_base.clone(), (*e).clone());
                } else {
                    new_node.add_factor((*b).clone(), (*e).clone());
                }
            }
            return Some(Arc::new(Expr::Mul(new_node)));
        }
        // Try exponent
        if let Some(new_exp) = s(exp) {
            let mut new_node = MulNode::one();
            new_node.scale(&node.coeff);
            for (i, (b, e)) in factors.iter().enumerate() {
                if i == idx {
                    new_node.add_factor((*b).clone(), new_exp.clone());
                } else {
                    new_node.add_factor((*b).clone(), (*e).clone());
                }
            }
            return Some(Arc::new(Expr::Mul(new_node)));
        }
    }
    None
}

/// Try `s` on base or exponent of a `Pow` node, returning first success.
fn one_pow(base: &Arc<Expr>, exp: &Arc<Expr>, s: &Strategy) -> Option<Arc<Expr>> {
    if let Some(new_base) = s(base) {
        return Some(Arc::new(Expr::Pow(new_base, Arc::clone(exp))));
    }
    if let Some(new_exp) = s(exp) {
        return Some(Arc::new(Expr::Pow(Arc::clone(base), new_exp)));
    }
    None
}

// ── Tree traversal strategies ─────────────────────────────────────────────────

/// Apply `s` at the root, then recursively to all children (top-down).
///
/// Returns `None` if `s` fails at the root and the node has no children,
/// or `Some` with the result of applying `s` recursively to children.
pub fn topdown(s: Strategy) -> Strategy {
    Box::new(move |e| {
        // Apply at root; if fails, keep original for child traversal
        let after_root = s(e).unwrap_or_else(|| Arc::clone(e));
        // Recurse into children using the same strategy (via all)
        apply_all_recursive(&after_root, &s)
    })
}

/// Recursively apply `s` to children first, then to the root (bottom-up).
pub fn bottomup(s: Strategy) -> Strategy {
    Box::new(move |e| {
        // Recurse into children first
        let after_children = apply_all_recursive(e, &s);
        let target = after_children.unwrap_or_else(|| Arc::clone(e));
        // Apply at root; if fails, return target unchanged
        Some(s(&target).unwrap_or(target))
    })
}

/// Apply `s` to all children recursively (helper for topdown/bottomup).
///
/// For leaf nodes, returns `Some(e.clone())`. For compound nodes, applies
/// `s` recursively via an inner `all`-like traversal.
fn apply_all_recursive(e: &Arc<Expr>, s: &Strategy) -> Option<Arc<Expr>> {
    match e.as_ref() {
        Expr::Add(node) => {
            let mut new_node = AddNode::zero();
            new_node.add_constant(node.constant.clone());
            for (term, coeff) in &node.terms {
                let child_result = apply_all_recursive(term, s);
                let child = child_result.unwrap_or_else(|| Arc::clone(term));
                let transformed = s(&child).unwrap_or(child);
                new_node.add_term(transformed, coeff.clone());
            }
            Some(Arc::new(Expr::Add(new_node)))
        }
        Expr::Mul(node) => {
            let mut new_node = MulNode::one();
            new_node.scale(&node.coeff);
            for (base, exp) in &node.factors {
                let new_base = {
                    let r = apply_all_recursive(base, s);
                    let c = r.unwrap_or_else(|| Arc::clone(base));
                    s(&c).unwrap_or(c)
                };
                let new_exp = {
                    let r = apply_all_recursive(exp, s);
                    let c = r.unwrap_or_else(|| Arc::clone(exp));
                    s(&c).unwrap_or(c)
                };
                new_node.add_factor(new_base, new_exp);
            }
            Some(Arc::new(Expr::Mul(new_node)))
        }
        Expr::Pow(base, exp) => {
            let new_base = {
                let r = apply_all_recursive(base, s);
                let c = r.unwrap_or_else(|| Arc::clone(base));
                s(&c).unwrap_or(c)
            };
            let new_exp = {
                let r = apply_all_recursive(exp, s);
                let c = r.unwrap_or_else(|| Arc::clone(exp));
                s(&c).unwrap_or(c)
            };
            Some(Arc::new(Expr::Pow(new_base, new_exp)))
        }
        // Leaves return unchanged
        _ => Some(Arc::clone(e)),
    }
}

// ── Fixpoint strategies ───────────────────────────────────────────────────────

/// Apply `s` bottom-up repeatedly until fixpoint or `max_iter` is reached.
///
/// This is the most common strategy for algebraic simplification: it ensures
/// all inner redexes are reduced before outer ones, converging to a normal form.
pub fn innermost(s: Strategy, max_iter: usize) -> Strategy {
    repeat(bottomup(try_or_id(s)), max_iter)
}

/// Apply `s` top-down repeatedly until fixpoint or `max_iter` is reached.
pub fn outermost(s: Strategy, max_iter: usize) -> Strategy {
    repeat(topdown(try_or_id(s)), max_iter)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::BigRational;
    use num::traits::{One, Zero};

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }

    /// Rule: `x + 0 → x` — strip zero addend from AddNode with single term.
    fn rule_x_plus_0() -> Strategy {
        try_rule(|e| {
            if let Expr::Add(node) = e.as_ref() {
                if node.term_count() == 1 && node.constant.is_zero() {
                    let (term, coeff) = node.terms.iter().next().unwrap();
                    if coeff.is_one() {
                        return Some(Arc::clone(term));
                    }
                }
            }
            None
        })
    }

    /// Rule: `x * 1 → x` — strip unit coefficient from MulNode single factor.
    fn rule_x_times_1() -> Strategy {
        try_rule(|e| {
            if let Expr::Mul(node) = e.as_ref() {
                if node.factor_count() == 1 && node.coeff.is_one() {
                    let (base, exp) = node.factors.iter().next().unwrap();
                    if exp.is_one() {
                        return Some(Arc::clone(base));
                    }
                }
            }
            None
        })
    }

    #[test]
    fn test_id_always_succeeds() {
        let x = sym("rw_id_x");
        let result = id()(&x);
        assert!(result.is_some());
        assert!(Arc::ptr_eq(&result.unwrap(), &x));
    }

    #[test]
    fn test_fail_always_returns_none() {
        let x = sym("rw_fail_x");
        assert!(fail()(&x).is_none());
    }

    #[test]
    fn test_try_rule_applies_when_matches() {
        // Construct Add(x, 0) manually — normalize would simplify it immediately
        let x = sym("rw_try_x");
        let mut node = AddNode::zero();
        node.add_term(Arc::clone(&x), BigRational::one());
        let expr = Arc::new(Expr::Add(node));
        let result = rule_x_plus_0()(&expr);
        assert!(result.is_some());
        assert_eq!(*result.unwrap(), *x);
    }

    #[test]
    fn test_try_rule_returns_none_on_mismatch() {
        let x = sym("rw_try_nm_x");
        // Symbol has no Add — rule should not apply
        assert!(rule_x_plus_0()(&x).is_none());
    }

    #[test]
    fn test_sequence_both_succeed() {
        // Build Add node with coeff=1 term and zero constant, wrapped in Mul with coeff=1
        let x = sym("rw_seq_x");
        let mut add_node = AddNode::zero();
        add_node.add_term(Arc::clone(&x), BigRational::one());
        let add_expr = Arc::new(Expr::Add(add_node));
        // Wrap in Mul to test sequence
        let mut mul_node = MulNode::one();
        mul_node.add_factor(Arc::clone(&add_expr), Expr::int(1));
        let mul_expr = Arc::new(Expr::Mul(mul_node));

        // sequence: first simplify Mul(Add(x,0)) → Add(x,0) via rule_x_times_1,
        // but rule_x_times_1 needs single factor exp=1 and coeff=1.
        // The inner factor IS the add_expr, so rule_x_times_1 should strip the Mul wrapper.
        let s = sequence(rule_x_times_1(), rule_x_plus_0());
        let result = s(&mul_expr);
        assert!(result.is_some());
        assert_eq!(*result.unwrap(), *x);
    }

    #[test]
    fn test_sequence_first_fails() {
        let x = sym("rw_seq_ff_x");
        let s = sequence(fail(), id());
        assert!(s(&x).is_none());
    }

    #[test]
    fn test_sequence_second_fails() {
        let x = sym("rw_seq_sf_x");
        let s = sequence(id(), fail());
        assert!(s(&x).is_none());
    }

    #[test]
    fn test_choice_first_succeeds() {
        let x = sym("rw_ch_fs_x");
        let s = choice(id(), fail());
        let result = s(&x);
        assert!(result.is_some());
        assert!(Arc::ptr_eq(&result.unwrap(), &x));
    }

    #[test]
    fn test_choice_fallback_to_second() {
        let x = sym("rw_ch_fb_x");
        let s = choice(fail(), id());
        let result = s(&x);
        assert!(result.is_some());
        assert!(Arc::ptr_eq(&result.unwrap(), &x));
    }

    #[test]
    fn test_choice_both_fail() {
        let x = sym("rw_ch_bf_x");
        let s = choice(fail(), fail());
        assert!(s(&x).is_none());
    }

    #[test]
    fn test_guard_passes_when_predicate_holds() {
        let x = Expr::int(42);
        let s = guard(|e| matches!(e.as_ref(), Expr::Integer(_)), id());
        assert!(s(&x).is_some());
    }

    #[test]
    fn test_guard_blocks_when_predicate_fails() {
        let x = sym("rw_guard_sym");
        let s = guard(|e| matches!(e.as_ref(), Expr::Integer(_)), id());
        assert!(s(&x).is_none());
    }

    #[test]
    fn test_repeat_converges() {
        // Build ((Add(Add(x,0),0),0) manually so rule applies three times
        let x = sym("rw_rep_x");
        let layer = |inner: Arc<Expr>| {
            let mut node = AddNode::zero();
            node.add_term(inner, BigRational::one());
            Arc::new(Expr::Add(node))
        };
        let expr = layer(layer(layer(Arc::clone(&x))));
        let s = repeat(rule_x_plus_0(), 100);
        let result = s(&expr).unwrap();
        assert_eq!(*result, *x);
    }

    #[test]
    fn test_repeat_max_iter_limits() {
        // Counter via atomic to verify iteration stops
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc as StdArc;
        let count = StdArc::new(AtomicUsize::new(0));
        let count2 = StdArc::clone(&count);
        let x = sym("rw_rep_max_x");
        // Strategy that always "succeeds" (identity) — would loop forever without limit
        let s = repeat(
            Box::new(move |e| {
                count2.fetch_add(1, Ordering::Relaxed);
                Some(Arc::clone(e))
            }),
            5,
        );
        let _ = s(&x);
        assert_eq!(count.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_all_succeeds_on_leaf() {
        let x = sym("rw_all_leaf_x");
        // all with fail() on a leaf succeeds vacuously
        let s = all(fail());
        assert!(s(&x).is_some());
    }

    #[test]
    fn test_all_applies_to_add_children() {
        let x = sym("rw_all_add_x");
        let mut node = AddNode::zero();
        node.add_term(Arc::clone(&x), BigRational::one());
        let expr = Arc::new(Expr::Add(node));
        // all with id() should return Some with the same structure
        let s = all(id());
        assert!(s(&expr).is_some());
    }

    #[test]
    fn test_all_fails_if_child_fails() {
        let x = sym("rw_all_fail_x");
        let mut node = AddNode::zero();
        node.add_term(Arc::clone(&x), BigRational::one());
        let expr = Arc::new(Expr::Add(node));
        // all with fail() on compound node should return None
        let s = all(fail());
        assert!(s(&expr).is_none());
    }

    #[test]
    fn test_one_fails_on_leaf() {
        let x = sym("rw_one_leaf_x");
        assert!(one(id())(&x).is_none());
    }

    #[test]
    fn test_one_applies_to_first_matching_child() {
        let x = sym("rw_one_add_x");
        let mut node = AddNode::zero();
        node.add_term(Arc::clone(&x), BigRational::one());
        let expr = Arc::new(Expr::Add(node));
        let s = one(id());
        // Should succeed — finds the single term
        assert!(s(&expr).is_some());
    }

    #[test]
    fn test_bottomup_processes_leaves_first() {
        // A bottomup traversal should apply to leaves before compound nodes.
        // Track visit order via side-effect.
        use std::sync::{Arc as StdArc, Mutex};
        let visited = StdArc::new(Mutex::new(Vec::new()));
        let visited2 = StdArc::clone(&visited);

        let x = sym("rw_bu_x");
        let mut node = AddNode::zero();
        node.add_term(Arc::clone(&x), BigRational::one());
        let expr = Arc::new(Expr::Add(node));

        let s = bottomup(Box::new(move |e| {
            let variant = match e.as_ref() {
                Expr::Symbol(_) => "symbol",
                Expr::Add(_) => "add",
                _ => "other",
            };
            visited2.lock().unwrap().push(variant.to_string());
            None
        }));
        let _ = s(&expr);
        let v = visited.lock().unwrap();
        // Symbol (leaf) should appear before Add (parent)
        let sym_pos = v.iter().position(|s| s == "symbol").unwrap();
        let add_pos = v.iter().position(|s| s == "add").unwrap();
        assert!(sym_pos < add_pos, "symbol should be visited before add");
    }

    #[test]
    fn test_innermost_reaches_fixpoint() {
        // Build Add(Add(Add(x,0),0),0) manually; innermost should reduce to x
        let x = sym("rw_inner_x");
        let layer = |inner: Arc<Expr>| {
            let mut node = AddNode::zero();
            node.add_term(inner, BigRational::one());
            Arc::new(Expr::Add(node))
        };
        let expr = layer(layer(layer(Arc::clone(&x))));
        let s = innermost(rule_x_plus_0(), 100);
        let result = s(&expr).unwrap();
        assert_eq!(*result, *x);
    }

    #[test]
    fn test_innermost_choice_two_rules() {
        // Build Mul(Add(x,0)) — innermost with choice should first reduce the Add,
        // then the Mul wrapper.
        let x = sym("rw_inn_ch_x");
        let mut add_node = AddNode::zero();
        add_node.add_term(Arc::clone(&x), BigRational::one());
        let add_expr = Arc::new(Expr::Add(add_node));
        let mut mul_node = MulNode::one();
        mul_node.add_factor(Arc::clone(&add_expr), Expr::int(1));
        let expr = Arc::new(Expr::Mul(mul_node));

        let s = innermost(choice(rule_x_plus_0(), rule_x_times_1()), 100);
        let result = s(&expr).unwrap();
        assert_eq!(*result, *x);
    }

    #[test]
    fn test_outermost_reduces() {
        let x = sym("rw_outer_x");
        let layer = |inner: Arc<Expr>| {
            let mut node = AddNode::zero();
            node.add_term(inner, BigRational::one());
            Arc::new(Expr::Add(node))
        };
        let expr = layer(layer(Arc::clone(&x)));
        let s = outermost(rule_x_plus_0(), 100);
        let result = s(&expr).unwrap();
        assert_eq!(*result, *x);
    }

    #[test]
    fn test_try_or_id_succeeds_on_match() {
        let x = sym("rw_toi_m_x");
        let mut node = AddNode::zero();
        node.add_term(Arc::clone(&x), BigRational::one());
        let expr = Arc::new(Expr::Add(node));
        let result = try_or_id(rule_x_plus_0())(&expr).unwrap();
        assert_eq!(*result, *x);
    }

    #[test]
    fn test_try_or_id_returns_input_on_mismatch() {
        let x = sym("rw_toi_nm_x");
        let result = try_or_id(rule_x_plus_0())(&x).unwrap();
        assert!(Arc::ptr_eq(&result, &x));
    }

    #[test]
    fn test_pow_children_traversed_by_all() {
        let x = sym("rw_pow_all_x");
        let expr = Arc::new(Expr::Pow(Arc::clone(&x), Expr::int(2)));
        let s = all(id());
        assert!(s(&expr).is_some());
    }

    #[test]
    fn test_pow_child_traversal_by_one() {
        let x = sym("rw_pow_one_x");
        let expr = Arc::new(Expr::Pow(Arc::clone(&x), Expr::int(2)));
        let s = one(id());
        // Should succeed on base (x)
        assert!(s(&expr).is_some());
    }
}
