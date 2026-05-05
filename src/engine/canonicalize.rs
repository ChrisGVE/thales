//! Canonicalization and hashing for [`CanonicalPattern`].
//!
//! - [`canonicalize`] — walk an `Arc<Expr>` and produce `(CanonicalPattern, VarMap)`.
//! - [`pattern_hash`] — fast hash of a `CanonicalPattern`.
//! - [`structural_hash`] — hash the raw `Arc<Expr>` tree directly.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::engine::canonical_pattern::{CanonicalPattern, PatternHash, SlotId, VarMap};
use crate::numeric::ring::Ring;
use crate::numeric::{BigRational, Expr, FuncId, SymbolId};

// ── Canonicalize ─────────────────────────────────────────────────────────────

/// Walk an `Arc<Expr>` depth-first and produce a canonical pattern
/// together with the variable map.
///
/// Symbols are assigned [`SlotId`]s in first-encounter order: the first
/// distinct symbol encountered during the DFS gets `SlotId(0)`, the
/// second gets `SlotId(1)`, and so on.
pub fn canonicalize(expr: &Arc<Expr>) -> (CanonicalPattern, VarMap) {
    let mut map = VarMap::new();
    let pattern = canonicalize_node(expr, &mut map);
    (pattern, map)
}

fn canonicalize_node(expr: &Arc<Expr>, map: &mut VarMap) -> CanonicalPattern {
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
            let slot = if let Some(s) = map.slot_of(*id) {
                s
            } else {
                let s = SlotId(map.len() as u32);
                map.insert(*id, s);
                s
            };
            CanonicalPattern::Slot(slot)
        }
        Expr::Add(add) => canonicalize_add(add, map),
        Expr::Mul(mul) => canonicalize_mul(mul, map),
        Expr::Pow(base, exp) => {
            let b = canonicalize_node(base, map);
            let e = canonicalize_node(exp, map);
            CanonicalPattern::Pow(Arc::new(b), Arc::new(e))
        }
        Expr::Func(id, args) => {
            let canon_args: Vec<Arc<CanonicalPattern>> = args
                .iter()
                .map(|a| Arc::new(canonicalize_node(a, map)))
                .collect();
            CanonicalPattern::Func(*id, canon_args)
        }
    }
}

fn canonicalize_add(add: &crate::numeric::AddNode, map: &mut VarMap) -> CanonicalPattern {
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
        let t = canonicalize_node(term, map);
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

fn canonicalize_mul(mul: &crate::numeric::MulNode, map: &mut VarMap) -> CanonicalPattern {
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
        let b = canonicalize_node(base, map);
        let exp_is_one = matches!(exp.as_ref(), Expr::Integer(n) if n.to_i64() == Some(1));
        if exp_is_one {
            children.push(Arc::new(b));
        } else {
            let e = canonicalize_node(exp, map);
            let pow = CanonicalPattern::Pow(Arc::new(b), Arc::new(e));
            children.push(Arc::new(pow));
        }
    }
    children.sort_by(|a, b| a.as_ref().cmp(b.as_ref()));
    CanonicalPattern::Mul(children)
}

/// Extract (numerator, denominator) as i64 from a BigRational.
fn rational_parts(r: &BigRational) -> (i64, i64) {
    let n = r.numer().to_i64().unwrap_or(i64::MAX);
    let d = r.denom().to_i64().unwrap_or(i64::MAX);
    (n, d)
}

// ── pattern_hash ──────────────────────────────────────────────────────────────

/// Compute a 64-bit hash of a [`CanonicalPattern`].
pub fn pattern_hash(pattern: &CanonicalPattern) -> PatternHash {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    pattern.hash(&mut h);
    PatternHash(h.finish())
}

// ── structural_hash ───────────────────────────────────────────────────────────

/// Hash an [`Expr`] tree directly without canonicalization.
///
/// Two structurally identical expressions (same shape and same symbol
/// names) will produce the same hash.
pub fn structural_hash(expr: &Arc<Expr>) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    expr.as_ref().hash(&mut h);
    h.finish()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::SymbolicConstant;
    use crate::numeric::SmallInt;

    fn sym(name: &'static str) -> Arc<Expr> {
        Arc::new(Expr::Symbol(SymbolId::intern(name)))
    }

    fn int(n: i64) -> Arc<Expr> {
        Arc::new(Expr::Integer(SmallInt::from(n)))
    }

    #[test]
    fn fast_canonical_integer() {
        let (p, map) = canonicalize(&int(7));
        assert_eq!(p, CanonicalPattern::Integer(7));
        assert!(map.is_empty());
    }

    #[test]
    fn fast_canonical_symbol_gets_slot_zero() {
        let (p, map) = canonicalize(&sym("x"));
        assert_eq!(p, CanonicalPattern::Slot(SlotId(0)));
        assert_eq!(map.len(), 1);
        let id = SymbolId::intern("x");
        assert_eq!(map.slot_of(id), Some(SlotId(0)));
    }

    #[test]
    fn fast_canonical_two_symbols_get_distinct_slots() {
        use crate::numeric::AddNode;
        use crate::numeric::BigRational;
        let x_id = SymbolId::intern("x");
        let y_id = SymbolId::intern("y");
        let x_expr = Arc::new(Expr::Symbol(x_id));
        let y_expr = Arc::new(Expr::Symbol(y_id));
        let mut add = AddNode::zero();
        add.terms.insert(x_expr, BigRational::from_i64(1, 1));
        add.terms.insert(y_expr, BigRational::from_i64(1, 1));
        let expr = Arc::new(Expr::Add(add));
        let (_p, map) = canonicalize(&expr);
        assert_eq!(map.len(), 2);
        let sx = map.slot_of(x_id);
        let sy = map.slot_of(y_id);
        assert!(sx.is_some() && sy.is_some());
        assert_ne!(sx, sy);
    }

    #[test]
    fn fast_canonical_same_symbol_same_slot() {
        use crate::numeric::AddNode;
        use crate::numeric::BigRational;
        let x_id = SymbolId::intern("x2");
        let x1 = Arc::new(Expr::Symbol(x_id));
        let x2 = Arc::new(Expr::Symbol(x_id));
        let mut add = AddNode::zero();
        add.terms.insert(x1, BigRational::from_i64(1, 1));
        add.terms.insert(x2.clone(), BigRational::from_i64(2, 1));
        let (p, map) = canonicalize(&x2);
        assert_eq!(p, CanonicalPattern::Slot(SlotId(0)));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn fast_canonical_deterministic() {
        let expr = sym("alpha");
        let (p1, _) = canonicalize(&expr);
        let (p2, _) = canonicalize(&expr);
        assert_eq!(p1, p2);
    }

    #[test]
    fn fast_pattern_hash_same_pattern_same_hash() {
        let (p1, _) = canonicalize(&sym("a"));
        let (p2, _) = canonicalize(&sym("b"));
        assert_eq!(pattern_hash(&p1), pattern_hash(&p2));
    }

    #[test]
    fn fast_pattern_hash_different_patterns() {
        let (p_sym, _) = canonicalize(&sym("a"));
        let (p_int, _) = canonicalize(&int(42));
        assert_ne!(pattern_hash(&p_sym), pattern_hash(&p_int));
    }

    #[test]
    fn fast_structural_hash_same_expr() {
        let e = sym("q");
        assert_eq!(structural_hash(&e), structural_hash(&e));
    }

    #[test]
    fn fast_structural_hash_different_symbols() {
        let e1 = sym("p");
        let e2 = sym("q");
        assert_ne!(structural_hash(&e1), structural_hash(&e2));
    }

    #[test]
    fn fast_canonical_float() {
        let f = Arc::new(Expr::Float(3.14));
        let (p, _) = canonicalize(&f);
        assert_eq!(p, CanonicalPattern::Float(3.14f64.to_bits()));
    }

    #[test]
    fn fast_canonical_constant() {
        let c = Arc::new(Expr::Constant(SymbolicConstant::Pi));
        let (p, _) = canonicalize(&c);
        assert_eq!(p, CanonicalPattern::Constant(SymbolicConstant::Pi));
    }

    #[test]
    fn fast_canonical_pow() {
        use crate::numeric::SmallInt;
        let base = sym("x");
        let exp = Arc::new(Expr::Integer(SmallInt::from(2i64)));
        let expr = Arc::new(Expr::Pow(base, exp));
        let (p, map) = canonicalize(&expr);
        assert!(matches!(p, CanonicalPattern::Pow(_, _)));
        assert_eq!(map.len(), 1);
    }
}
