//! Cache rehydration and slottification for D0 structural memoization.
//!
//! This module provides the inverse of [`crate::engine::canonicalize`]:
//!
//! - [`rehydrate_expr`] — convert a [`CanonicalPattern`] back to a live
//!   `Arc<Expr>`, substituting slot IDs for real symbols.
//! - [`rehydrate_trace`] — walk a [`TraceNode`] tree, rehydrating any
//!   embedded `Arc<Expr>` fields.
//! - [`slottify_expr`] — convert an `Arc<Expr>` to a [`CanonicalPattern`],
//!   building the [`VarMap`] incrementally (cache-internal use).
//! - [`slottify_trace`] — walk a [`TraceNode`] tree, slottifying any
//!   embedded `Arc<Expr>` fields.

use std::sync::Arc;

use crate::engine::canonical_pattern::{CanonicalPattern, SlotId, VarMap};
use crate::engine::trace_tree::{TraceBranch, TraceNode, TraceTree};
use crate::numeric::ring::Ring;
use crate::numeric::{AddNode, BigRational, Expr, MulNode, SmallInt, SymbolId};

// ── Sentinel base ─────────────────────────────────────────────────────────────

/// Sentinel base for strategy-introduced variables.
///
/// `SymbolId` indices at or above this value are sentinel IDs that were
/// created by `fresh_id_gen` during rehydration and have not been interned in
/// the global symbol table. Their [`Display`] representation is `$slot_N`
/// where `N = u32::MAX - index`.
pub const SENTINEL_BASE: u32 = u32::MAX - 65535;

// ── rehydrate_expr ────────────────────────────────────────────────────────────

/// Rehydrate a [`CanonicalPattern`] back to a live `Arc<Expr>`.
///
/// Each variant maps directly to the corresponding [`Expr`] variant:
///
/// - `Integer(n)` → `Expr::Integer`
/// - `Rational(n, d)` → `Expr::Rational`
/// - `Float(bits)` → `Expr::Float`
/// - `Complex(re, im)` → `Expr::Complex`
/// - `Constant(c)` → `Expr::Constant`
/// - `Slot(s)` → look up `var_map.reverse[s]`; if absent, call
///   `fresh_id_gen(s)` to allocate a new sentinel [`SymbolId`].
/// - `Add/Mul/Pow/Func` → recurse into children then build the node.
pub fn rehydrate_expr(
    pattern: &CanonicalPattern,
    var_map: &VarMap,
    fresh_id_gen: &mut dyn FnMut(SlotId) -> SymbolId,
) -> Arc<Expr> {
    match pattern {
        CanonicalPattern::Integer(n) => Arc::new(Expr::Integer(SmallInt::from(*n))),
        CanonicalPattern::Rational(n, d) => Arc::new(Expr::Rational(BigRational::from_i64(*n, *d))),
        CanonicalPattern::Float(bits) => Arc::new(Expr::Float(f64::from_bits(*bits))),
        CanonicalPattern::Complex(re_bits, im_bits) => {
            use num_complex::Complex64;
            Arc::new(Expr::Complex(Complex64::new(
                f64::from_bits(*re_bits),
                f64::from_bits(*im_bits),
            )))
        }
        CanonicalPattern::Constant(c) => Arc::new(Expr::Constant(*c)),
        CanonicalPattern::Slot(slot) => {
            let sym_id = var_map
                .symbol_of(*slot)
                .unwrap_or_else(|| fresh_id_gen(*slot));
            Arc::new(Expr::Symbol(sym_id))
        }
        CanonicalPattern::Add(terms) => rehydrate_add(terms, var_map, fresh_id_gen),
        CanonicalPattern::Mul(factors) => rehydrate_mul(factors, var_map, fresh_id_gen),
        CanonicalPattern::Pow(base_pat, exp_pat) => {
            let base = rehydrate_expr(base_pat, var_map, fresh_id_gen);
            let exp = rehydrate_expr(exp_pat, var_map, fresh_id_gen);
            Arc::new(Expr::Pow(base, exp))
        }
        CanonicalPattern::Func(id, args) => {
            let rehydrated: Vec<Arc<Expr>> = args
                .iter()
                .map(|a| rehydrate_expr(a, var_map, fresh_id_gen))
                .collect();
            Arc::new(Expr::Func(*id, rehydrated))
        }
    }
}

/// Reconstruct an `Expr::Add` from a canonical Add pattern's children.
///
/// Each child is either a standalone term (coefficient 1) or a Mul whose
/// first child is the coefficient. We build an `AddNode` incrementally.
fn rehydrate_add(
    terms: &[Arc<CanonicalPattern>],
    var_map: &VarMap,
    fresh_id_gen: &mut dyn FnMut(SlotId) -> SymbolId,
) -> Arc<Expr> {
    let mut add = AddNode::zero();
    for term_pat in terms {
        match term_pat.as_ref() {
            CanonicalPattern::Integer(n) => {
                add.add_constant(BigRational::from_i64(*n, 1));
            }
            CanonicalPattern::Rational(n, d) => {
                add.add_constant(BigRational::from_i64(*n, *d));
            }
            CanonicalPattern::Mul(factors) => {
                // Check if this Mul encodes a `coeff * term` pair (exactly 2
                // children where the first is numeric).
                if let Some((coeff, term)) = extract_coeff_term(factors, var_map, fresh_id_gen) {
                    add.add_term(term, coeff);
                } else {
                    // Not a coeff*term pair — rehydrate the whole Mul as a term.
                    let expr = rehydrate_mul(factors, var_map, fresh_id_gen);
                    add.add_term(expr, BigRational::from_i64(1, 1));
                }
            }
            other => {
                let expr = rehydrate_expr(other, var_map, fresh_id_gen);
                add.add_term(expr, BigRational::from_i64(1, 1));
            }
        }
    }
    Arc::new(Expr::Add(add))
}

/// Try to extract a `(coeff, term)` pair from a Mul pattern with exactly 2
/// children where the first child is a numeric constant.
fn extract_coeff_term(
    factors: &[Arc<CanonicalPattern>],
    var_map: &VarMap,
    fresh_id_gen: &mut dyn FnMut(SlotId) -> SymbolId,
) -> Option<(BigRational, Arc<Expr>)> {
    if factors.len() != 2 {
        return None;
    }
    let coeff = match factors[0].as_ref() {
        CanonicalPattern::Integer(n) => BigRational::from_i64(*n, 1),
        CanonicalPattern::Rational(n, d) => BigRational::from_i64(*n, *d),
        _ => return None,
    };
    let term = rehydrate_expr(&factors[1], var_map, fresh_id_gen);
    Some((coeff, term))
}

/// Reconstruct an `Expr::Mul` from a canonical Mul pattern's children.
fn rehydrate_mul(
    factors: &[Arc<CanonicalPattern>],
    var_map: &VarMap,
    fresh_id_gen: &mut dyn FnMut(SlotId) -> SymbolId,
) -> Arc<Expr> {
    let mut mul = MulNode::one();
    for factor_pat in factors {
        match factor_pat.as_ref() {
            CanonicalPattern::Integer(n) => {
                mul.scale(&BigRational::from_i64(*n, 1));
            }
            CanonicalPattern::Rational(n, d) => {
                mul.scale(&BigRational::from_i64(*n, *d));
            }
            CanonicalPattern::Pow(base_pat, exp_pat) => {
                let base = rehydrate_expr(base_pat, var_map, fresh_id_gen);
                let exp = rehydrate_expr(exp_pat, var_map, fresh_id_gen);
                mul.add_factor(base, exp);
            }
            other => {
                let expr = rehydrate_expr(other, var_map, fresh_id_gen);
                mul.add_factor(expr, Expr::int(1));
            }
        }
    }
    Arc::new(Expr::Mul(mul))
}

// ── rehydrate_trace ───────────────────────────────────────────────────────────

/// Rehydrate a [`TraceNode`] tree, replacing slot-based expressions with live
/// ones. Embedded `Arc<Expr>` fields in [`Step`] nodes are rehydrated via
/// [`rehydrate_expr`]. Branch, Join, and CacheHit nodes recurse into their
/// children using the same `fresh_id_gen` for consistency.
pub fn rehydrate_trace(
    trace: &TraceNode,
    var_map: &VarMap,
    fresh_id_gen: &mut dyn FnMut(SlotId) -> SymbolId,
) -> TraceNode {
    match trace {
        TraceNode::Step(step) => {
            let new_input = step.input.as_ref().map(|e| {
                rehydrate_expr(&slottify_expr(e, &mut VarMap::new()), var_map, fresh_id_gen)
            });
            let new_output = step.output.as_ref().map(|e| {
                rehydrate_expr(&slottify_expr(e, &mut VarMap::new()), var_map, fresh_id_gen)
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
                    nodes: rehydrate_tree(&branch.nodes, var_map, fresh_id_gen),
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
                .map(|part| rehydrate_tree(part, var_map, fresh_id_gen))
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
            let new_cached = rehydrate_trace(cached_trace, var_map, fresh_id_gen);
            TraceNode::CacheHit {
                source: *source,
                pattern_hash: *pattern_hash,
                cached_trace: Box::new(new_cached),
            }
        }
    }
}

/// Walk a [`TraceTree`], rehydrating every node.
fn rehydrate_tree(
    tree: &TraceTree,
    var_map: &VarMap,
    fresh_id_gen: &mut dyn FnMut(SlotId) -> SymbolId,
) -> TraceTree {
    let new_nodes = tree
        .nodes
        .iter()
        .map(|n| rehydrate_trace(n, var_map, fresh_id_gen))
        .collect();
    TraceTree { nodes: new_nodes }
}

// ── slottify_expr ─────────────────────────────────────────────────────────────

/// Convert an `Arc<Expr>` to a [`CanonicalPattern`], building `var_map`
/// incrementally.
///
/// This is the same logic as `canonicalize_node` in
/// [`crate::engine::canonicalize`], exposed here for cache-internal use so
/// callers can supply an existing `VarMap` and continue slot numbering across
/// multiple expressions.
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

/// Extract (numerator, denominator) as `i64` from a [`BigRational`].
fn rational_parts(r: &BigRational) -> (i64, i64) {
    let n = r.numer().to_i64().unwrap_or(i64::MAX);
    let d = r.denom().to_i64().unwrap_or(i64::MAX);
    (n, d)
}

// ── slottify_trace ────────────────────────────────────────────────────────────

/// Convert a [`TraceNode`] tree to canonical (slotted) form by slottifying
/// all embedded `Arc<Expr>` fields.
pub(crate) fn slottify_trace(trace: &TraceNode, var_map: &mut VarMap) -> TraceNode {
    match trace {
        TraceNode::Step(step) => {
            let new_input = step.input.as_ref().map(|e| {
                let pat = slottify_expr(e, var_map);
                rehydrate_expr(&pat, var_map, &mut |_slot| {
                    // Should not be called: all slots were just inserted into var_map.
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

/// Walk a [`TraceTree`], slottifying every node.
fn slottify_tree(tree: &TraceTree, var_map: &mut VarMap) -> TraceTree {
    let new_nodes = tree
        .nodes
        .iter()
        .map(|n| slottify_trace(n, var_map))
        .collect();
    TraceTree { nodes: new_nodes }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::canonical_pattern::{CanonicalPattern, SlotId, VarMap};
    use crate::numeric::{Expr, SmallInt, SymbolId};

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn int_pat(n: i64) -> CanonicalPattern {
        CanonicalPattern::Integer(n)
    }

    fn slot_pat(n: u32) -> CanonicalPattern {
        CanonicalPattern::Slot(SlotId(n))
    }

    /// A no-op fresh_id_gen that panics if called — used when all slots are
    /// expected to be in the VarMap.
    fn no_fresh(_slot: SlotId) -> SymbolId {
        panic!("fresh_id_gen called unexpectedly")
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn fast_rehydrate_integer() {
        let pat = int_pat(42);
        let vm = VarMap::new();
        let expr = rehydrate_expr(&pat, &vm, &mut no_fresh);
        assert_eq!(expr.as_ref(), &Expr::Integer(SmallInt::from(42i64)));
    }

    #[test]
    fn fast_rehydrate_slot_to_var() {
        // Slot 0 is mapped to symbol "x".
        let x_id = SymbolId::intern("rh_slot_x");
        let mut vm = VarMap::new();
        vm.insert(x_id, SlotId(0));

        let pat = slot_pat(0);
        let expr = rehydrate_expr(&pat, &vm, &mut no_fresh);
        assert_eq!(expr.as_ref(), &Expr::Symbol(x_id));
    }

    #[test]
    fn fast_rehydrate_slot_fresh() {
        // Slot 7 is NOT in the VarMap — fresh_id_gen must be called.
        let vm = VarMap::new();
        let mut fresh_called = false;
        let pat = slot_pat(7);
        let _expr = rehydrate_expr(&pat, &vm, &mut |slot| {
            assert_eq!(slot, SlotId(7));
            fresh_called = true;
            SymbolId::intern("fresh_sym_slot7")
        });
        assert!(fresh_called, "fresh_id_gen was not called for unknown slot");
    }

    #[test]
    fn fast_slottify_symbol_to_slot() {
        let x_id = SymbolId::intern("slottify_x");
        let expr = Arc::new(Expr::Symbol(x_id));
        let mut vm = VarMap::new();
        let pat = slottify_expr(&expr, &mut vm);
        assert_eq!(pat, CanonicalPattern::Slot(SlotId(0)));
        assert_eq!(vm.len(), 1);
        assert_eq!(vm.slot_of(x_id), Some(SlotId(0)));
    }

    #[test]
    fn fast_slottify_rehydrate_roundtrip() {
        // Build `x + 1` using an AddNode.
        let x_id = SymbolId::intern("roundtrip_x");
        let x_expr = Arc::new(Expr::Symbol(x_id));
        let mut add = AddNode::zero();
        add.add_term(x_expr.clone(), BigRational::from_i64(1, 1));
        add.add_constant(BigRational::from_i64(1, 1));
        let expr = Arc::new(Expr::Add(add));

        // Slottify.
        let mut vm = VarMap::new();
        let pat = slottify_expr(&expr, &mut vm);
        assert!(matches!(pat, CanonicalPattern::Add(_)));

        // Rehydrate using the same VarMap.
        let back = rehydrate_expr(&pat, &vm, &mut no_fresh);
        // Must be an Add node.
        assert!(matches!(back.as_ref(), Expr::Add(_)));
        // The AddNode must contain the symbol and the constant.
        if let Expr::Add(add_back) = back.as_ref() {
            // constant == 1
            assert_eq!(
                add_back.constant.numer().to_i64().unwrap_or(-999),
                1,
                "constant part should be 1"
            );
            // exactly one non-constant term
            assert_eq!(add_back.terms.len(), 1);
        }
    }

    #[test]
    fn fast_sentinel_display() {
        // SymbolId whose index >= SENTINEL_BASE should display as "$slot_N".
        // We can't construct a sentinel SymbolId through intern(), but we can
        // verify SENTINEL_BASE constant and its relationship to display by
        // testing the formula directly.
        let slot_num: u32 = 5;
        let sentinel_index = u32::MAX - slot_num;
        assert!(sentinel_index >= SENTINEL_BASE);
        // Verify the reverse formula: u32::MAX - sentinel_index == slot_num.
        let recovered = u32::MAX - sentinel_index;
        assert_eq!(recovered, slot_num);
    }

    #[test]
    fn fast_rehydrate_add() {
        // Add(Integer(3), Slot(0)) with Slot 0 = "add_var"
        let var_id = SymbolId::intern("add_var");
        let mut vm = VarMap::new();
        vm.insert(var_id, SlotId(0));

        let pat = CanonicalPattern::Add(vec![
            Arc::new(CanonicalPattern::Integer(3)),
            Arc::new(CanonicalPattern::Slot(SlotId(0))),
        ]);
        let expr = rehydrate_expr(&pat, &vm, &mut no_fresh);
        assert!(
            matches!(expr.as_ref(), Expr::Add(_)),
            "expected Expr::Add, got something else"
        );
    }

    #[test]
    fn fast_rehydrate_pow() {
        // Pow(Slot(0), Integer(2)) with Slot 0 = "pow_var"
        let var_id = SymbolId::intern("pow_var");
        let mut vm = VarMap::new();
        vm.insert(var_id, SlotId(0));

        let pat = CanonicalPattern::Pow(
            Arc::new(CanonicalPattern::Slot(SlotId(0))),
            Arc::new(CanonicalPattern::Integer(2)),
        );
        let expr = rehydrate_expr(&pat, &vm, &mut no_fresh);
        match expr.as_ref() {
            Expr::Pow(base, exp) => {
                assert_eq!(base.as_ref(), &Expr::Symbol(var_id));
                assert_eq!(exp.as_ref(), &Expr::Integer(SmallInt::from(2i64)));
            }
            other => panic!("expected Expr::Pow, got {:?}", other),
        }
    }

    #[test]
    fn fast_rehydrate_func() {
        use crate::numeric::FuncId;
        // Sin(Slot(0)) with Slot 0 = "func_var"
        let var_id = SymbolId::intern("func_var");
        let mut vm = VarMap::new();
        vm.insert(var_id, SlotId(0));

        let pat = CanonicalPattern::Func(
            FuncId::Sin,
            vec![Arc::new(CanonicalPattern::Slot(SlotId(0)))],
        );
        let expr = rehydrate_expr(&pat, &vm, &mut no_fresh);
        match expr.as_ref() {
            Expr::Func(id, args) => {
                assert_eq!(*id, FuncId::Sin);
                assert_eq!(args.len(), 1);
                assert_eq!(args[0].as_ref(), &Expr::Symbol(var_id));
            }
            other => panic!("expected Expr::Func, got {:?}", other),
        }
    }
}
