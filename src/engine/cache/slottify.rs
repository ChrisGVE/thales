//! Expression and trace slottification for cache-internal use.
//!
//! Converts live `Arc<Expr>` trees and [`TraceNode`] trees into their
//! canonical (slot-based) form, building the [`VarMap`] incrementally.
//! This is the same logic as [`crate::engine::canonicalize`] but accepts
//! an existing `VarMap` for continued slot numbering.

use std::sync::Arc;

use crate::engine::canonical_pattern::{CanonicalPattern, SlotId, VarMap};
use crate::engine::trace_tree::{TraceBranch, TraceNode, TraceTree};
use crate::numeric::ring::Ring;
use crate::numeric::{BigRational, Expr};

use super::rehydrate::rehydrate_expr;

// ── slottify_expr ────────────────────────────────────────────────────────────

/// Convert an `Arc<Expr>` to a [`CanonicalPattern`], building `var_map`
/// incrementally.
///
/// Callers can supply an existing `VarMap` and continue slot numbering
/// across multiple expressions.
pub(crate) fn slottify_expr(expr: &Arc<Expr>, var_map: &mut VarMap) -> CanonicalPattern {
    match expr.as_ref() {
        Expr::Integer(n) => {
            let v = n.to_i64().unwrap_or(i64::MAX);
            CanonicalPattern::Integer(v)
        }
        Expr::Rational(r) => {
            let (n, d) = rational_parts(r);
            CanonicalPattern::Rational(n, d)
        }
        Expr::Float(f) => CanonicalPattern::Float(f.to_bits()),
        Expr::Complex(c) => CanonicalPattern::Complex(c.re.to_bits(), c.im.to_bits()),
        Expr::Constant(c) => CanonicalPattern::Constant(*c),
        Expr::Symbol(id) => {
            let slot = if let Some(s) = var_map.slot_of(*id) {
                s
            } else {
                let s = SlotId(var_map.len() as u32);
                var_map.insert(*id, s);
                s
            };
            CanonicalPattern::Slot(slot)
        }
        Expr::Add(add) => slottify_add(add, var_map),
        Expr::Mul(mul) => slottify_mul(mul, var_map),
        Expr::Pow(base, exp) => {
            let b = slottify_expr(base, var_map);
            let e = slottify_expr(exp, var_map);
            CanonicalPattern::Pow(Arc::new(b), Arc::new(e))
        }
        Expr::Func(id, args) => {
            let canon_args: Vec<Arc<CanonicalPattern>> = args
                .iter()
                .map(|a| Arc::new(slottify_expr(a, var_map)))
                .collect();
            CanonicalPattern::Func(*id, canon_args)
        }
    }
}

fn slottify_add(add: &crate::numeric::AddNode, map: &mut VarMap) -> CanonicalPattern {
    let mut children: Vec<Arc<CanonicalPattern>> = Vec::new();
    if !add.constant.is_zero() {
        let (n, d) = rational_parts(&add.constant);
        let c = if d == 1 {
            CanonicalPattern::Integer(n)
        } else {
            CanonicalPattern::Rational(n, d)
        };
        children.push(Arc::new(c));
    }
    for (term, coeff) in &add.terms {
        let t = slottify_expr(term, map);
        if coeff.is_one() {
            children.push(Arc::new(t));
        } else {
            let (cn, cd) = rational_parts(coeff);
            let coeff_pat = if cd == 1 {
                CanonicalPattern::Integer(cn)
            } else {
                CanonicalPattern::Rational(cn, cd)
            };
            let mul = CanonicalPattern::Mul(vec![Arc::new(coeff_pat), Arc::new(t)]);
            children.push(Arc::new(mul));
        }
    }
    children.sort_by(|a, b| a.as_ref().cmp(b.as_ref()));
    CanonicalPattern::Add(children)
}

fn slottify_mul(mul: &crate::numeric::MulNode, map: &mut VarMap) -> CanonicalPattern {
    let mut children: Vec<Arc<CanonicalPattern>> = Vec::new();
    if !mul.coeff.is_one() {
        let (n, d) = rational_parts(&mul.coeff);
        let c = if d == 1 {
            CanonicalPattern::Integer(n)
        } else {
            CanonicalPattern::Rational(n, d)
        };
        children.push(Arc::new(c));
    }
    for (base, exp) in &mul.factors {
        let b = slottify_expr(base, map);
        let exp_is_one = matches!(exp.as_ref(), Expr::Integer(n) if n.to_i64() == Some(1));
        if exp_is_one {
            children.push(Arc::new(b));
        } else {
            let e = slottify_expr(exp, map);
            let pow = CanonicalPattern::Pow(Arc::new(b), Arc::new(e));
            children.push(Arc::new(pow));
        }
    }
    children.sort_by(|a, b| a.as_ref().cmp(b.as_ref()));
    CanonicalPattern::Mul(children)
}

fn rational_parts(r: &BigRational) -> (i64, i64) {
    let n = r.numer().to_i64().unwrap_or(i64::MAX);
    let d = r.denom().to_i64().unwrap_or(i64::MAX);
    (n, d)
}

// ── slottify_trace ───────────────────────────────────────────────────────────

/// Convert a [`TraceNode`] tree to canonical (slotted) form by slottifying
/// all embedded `Arc<Expr>` fields.
pub(crate) fn slottify_trace(trace: &TraceNode, var_map: &mut VarMap) -> TraceNode {
    match trace {
        TraceNode::Step(step) => {
            let new_input = step.input.as_ref().map(|e| {
                let pat = slottify_expr(e, var_map);
                rehydrate_expr(&pat, var_map, &mut |_slot| {
                    unreachable!(
                        "slottify_trace: fresh slot encountered during immediate rehydrate"
                    )
                })
            });
            let new_output = step.output.as_ref().map(|e| {
                let pat = slottify_expr(e, var_map);
                rehydrate_expr(&pat, var_map, &mut |_slot| {
                    unreachable!(
                        "slottify_trace: fresh slot encountered during immediate rehydrate"
                    )
                })
            });
            let mut new_step = step.clone();
            new_step.input = new_input;
            new_step.output = new_output;
            TraceNode::Step(new_step)
        }
        TraceNode::Branch { reason, children } => {
            let new_children: Vec<TraceBranch> = children
                .iter()
                .map(|branch| TraceBranch {
                    strategy: branch.strategy,
                    outcome: branch.outcome.clone(),
                    nodes: slottify_tree(&branch.nodes, var_map),
                })
                .collect();
            TraceNode::Branch {
                reason: reason.clone(),
                children: new_children,
            }
        }
        TraceNode::Join { reason, parts } => {
            let new_parts: Vec<TraceTree> = parts
                .iter()
                .map(|part| slottify_tree(part, var_map))
                .collect();
            TraceNode::Join {
                reason: reason.clone(),
                parts: new_parts,
            }
        }
        TraceNode::CacheHit {
            source,
            pattern_hash,
            cached_trace,
        } => {
            let new_cached = slottify_trace(cached_trace, var_map);
            TraceNode::CacheHit {
                source: *source,
                pattern_hash: *pattern_hash,
                cached_trace: Box::new(new_cached),
            }
        }
    }
}

fn slottify_tree(tree: &TraceTree, var_map: &mut VarMap) -> TraceTree {
    let new_nodes = tree
        .nodes
        .iter()
        .map(|n| slottify_trace(n, var_map))
        .collect();
    TraceTree { nodes: new_nodes }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::canonical_pattern::{CanonicalPattern, SlotId, VarMap};
    use crate::numeric::{AddNode, BigRational, Expr, SmallInt, SymbolId};

    #[test]
    fn fast_slottify_symbol_to_slot() {
        let x_id = SymbolId::intern("slottify2_x");
        let expr = Arc::new(Expr::Symbol(x_id));
        let mut vm = VarMap::new();
        let pat = slottify_expr(&expr, &mut vm);
        assert_eq!(pat, CanonicalPattern::Slot(SlotId(0)));
        assert_eq!(vm.len(), 1);
        assert_eq!(vm.slot_of(x_id), Some(SlotId(0)));
    }

    #[test]
    fn fast_slottify_rehydrate_roundtrip() {
        let x_id = SymbolId::intern("roundtrip2_x");
        let x_expr = Arc::new(Expr::Symbol(x_id));
        let mut add = AddNode::zero();
        add.add_term(x_expr.clone(), BigRational::from_i64(1, 1));
        add.add_constant(BigRational::from_i64(1, 1));
        let expr = Arc::new(Expr::Add(add));

        let mut vm = VarMap::new();
        let pat = slottify_expr(&expr, &mut vm);
        assert!(matches!(pat, CanonicalPattern::Add(_)));

        fn no_fresh(_slot: SlotId) -> SymbolId {
            panic!("fresh_id_gen called unexpectedly")
        }
        let back = rehydrate_expr(&pat, &vm, &mut no_fresh);
        assert!(matches!(back.as_ref(), Expr::Add(_)));
        if let Expr::Add(add_back) = back.as_ref() {
            assert_eq!(add_back.constant.numer().to_i64().unwrap_or(-999), 1);
            assert_eq!(add_back.terms.len(), 1);
        }
    }

    #[test]
    fn fast_slottify_integer_passthrough() {
        let expr = Arc::new(Expr::Integer(SmallInt::from(42i64)));
        let mut vm = VarMap::new();
        let pat = slottify_expr(&expr, &mut vm);
        assert_eq!(pat, CanonicalPattern::Integer(42));
        assert!(vm.is_empty());
    }

    #[test]
    fn fast_slottify_two_symbols_distinct_slots() {
        let x_id = SymbolId::intern("slottify2_a");
        let y_id = SymbolId::intern("slottify2_b");
        let mut vm = VarMap::new();
        let px = slottify_expr(&Arc::new(Expr::Symbol(x_id)), &mut vm);
        let py = slottify_expr(&Arc::new(Expr::Symbol(y_id)), &mut vm);
        assert_eq!(px, CanonicalPattern::Slot(SlotId(0)));
        assert_eq!(py, CanonicalPattern::Slot(SlotId(1)));
        assert_eq!(vm.len(), 2);
    }
}
