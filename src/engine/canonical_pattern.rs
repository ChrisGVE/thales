//! Canonical pattern representation for structural memoization.
//!
//! A [`CanonicalPattern`] is a shape-only view of an [`Expr`] tree where
//! every [`Expr::Symbol`] leaf is replaced by a numbered [`SlotId`].
//! Slot IDs are assigned in depth-first, first-encounter order, so two
//! expressions that differ only in variable names canonicalize to the
//! same pattern.
//!
//! The primary use-case is D0 memoization: before invoking a strategy,
//! hash the input pattern; if the hash matches a cached result for this
//! strategy, replay the cached steps instead of re-running the strategy.
//!
//! # Functions
//!
//! - [`canonicalize`] — walk an `Arc<Expr>` and produce `(CanonicalPattern, VarMap)`.
//! - [`pattern_hash`] — fast hash of a `CanonicalPattern`.
//! - [`structural_hash`] — hash the raw `Arc<Expr>` tree directly.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::ast::SymbolicConstant;
use crate::numeric::ring::Ring;
use crate::numeric::{BigRational, Expr, FuncId, SymbolId};

// ── SlotId ────────────────────────────────────────────────────────────────────

/// Identifies a variable slot in a [`CanonicalPattern`].
///
/// Slots are numbered from 0 in depth-first, first-encounter order.
/// Slot 0 is the first distinct symbol encountered during a DFS walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SlotId(pub u32);

// ── PatternHash ───────────────────────────────────────────────────────────────

/// A 64-bit hash of a [`CanonicalPattern`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PatternHash(pub u64);

// ── VarMap ────────────────────────────────────────────────────────────────────

/// Bidirectional mapping between [`SymbolId`]s and [`SlotId`]s.
///
/// Built alongside canonicalization so the caller can recover which
/// original symbol each slot corresponds to.
#[derive(Debug, Clone, Default)]
pub struct VarMap {
    /// Forward map: original symbol → canonical slot.
    pub forward: HashMap<SymbolId, SlotId>,
    /// Reverse map: canonical slot → original symbol.
    pub reverse: HashMap<SlotId, SymbolId>,
}

impl VarMap {
    /// Create an empty variable map.
    #[must_use]
    pub fn new() -> Self {
        VarMap::default()
    }

    /// Insert a `(symbol, slot)` pair in both directions.
    pub fn insert(&mut self, id: SymbolId, slot: SlotId) {
        self.forward.insert(id, slot);
        self.reverse.insert(slot, id);
    }

    /// Look up the slot assigned to `id`, if any.
    #[must_use]
    pub fn slot_of(&self, id: SymbolId) -> Option<SlotId> {
        self.forward.get(&id).copied()
    }

    /// Look up the symbol assigned to `slot`, if any.
    #[must_use]
    pub fn symbol_of(&self, slot: SlotId) -> Option<SymbolId> {
        self.reverse.get(&slot).copied()
    }

    /// Number of (symbol, slot) pairs stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.forward.len()
    }

    /// True when no pairs have been inserted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.forward.is_empty()
    }
}

// ── CanonicalPattern ──────────────────────────────────────────────────────────

/// Shape-only representation of an [`Expr`] tree.
///
/// All symbol leaves are replaced by [`SlotId`]s; numeric values are
/// preserved so that patterns like `2*x` and `3*x` are distinct.
///
/// # Ordering
///
/// `CanonicalPattern` implements a total order used for sorting children
/// in canonical Add/Mul nodes. Variant discriminant is compared first;
/// within a variant, fields are compared in declaration order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalPattern {
    /// Exact integer value.
    Integer(i64),
    /// Rational number as (numerator, denominator) — always fully reduced,
    /// denominator positive.
    Rational(i64, i64),
    /// Floating-point value, stored as raw bits for deterministic equality.
    Float(u64),
    /// Complex number (re bits, im bits).
    Complex(u64, u64),
    /// A named symbolic constant (Pi, E, I).
    Constant(SymbolicConstant),
    /// A variable, replaced by its canonical slot ID.
    Slot(SlotId),
    /// N-ary sum.
    Add(Vec<Arc<CanonicalPattern>>),
    /// N-ary product.
    Mul(Vec<Arc<CanonicalPattern>>),
    /// Power: base ^ exponent.
    Pow(Arc<CanonicalPattern>, Arc<CanonicalPattern>),
    /// Function application.
    Func(FuncId, Vec<Arc<CanonicalPattern>>),
}

impl Hash for CanonicalPattern {
    fn hash<H: Hasher>(&self, state: &mut H) {
        pattern_discriminant(self).hash(state);
        match self {
            CanonicalPattern::Integer(n) => n.hash(state),
            CanonicalPattern::Rational(n, d) => {
                n.hash(state);
                d.hash(state);
            }
            CanonicalPattern::Float(bits) => bits.hash(state),
            CanonicalPattern::Complex(re, im) => {
                re.hash(state);
                im.hash(state);
            }
            CanonicalPattern::Constant(c) => c.hash(state),
            CanonicalPattern::Slot(s) => s.hash(state),
            CanonicalPattern::Add(terms) => {
                for t in terms {
                    t.hash(state);
                }
            }
            CanonicalPattern::Mul(factors) => {
                for f in factors {
                    f.hash(state);
                }
            }
            CanonicalPattern::Pow(b, e) => {
                b.hash(state);
                e.hash(state);
            }
            CanonicalPattern::Func(id, args) => {
                id.hash(state);
                for a in args {
                    a.hash(state);
                }
            }
        }
    }
}

fn pattern_discriminant(p: &CanonicalPattern) -> u8 {
    match p {
        CanonicalPattern::Integer(_) => 0,
        CanonicalPattern::Rational(_, _) => 1,
        CanonicalPattern::Float(_) => 2,
        CanonicalPattern::Complex(_, _) => 3,
        CanonicalPattern::Constant(_) => 4,
        CanonicalPattern::Slot(_) => 5,
        CanonicalPattern::Add(_) => 6,
        CanonicalPattern::Mul(_) => 7,
        CanonicalPattern::Pow(_, _) => 8,
        CanonicalPattern::Func(_, _) => 9,
    }
}

impl PartialOrd for CanonicalPattern {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CanonicalPattern {
    fn cmp(&self, other: &Self) -> Ordering {
        pattern_discriminant(self)
            .cmp(&pattern_discriminant(other))
            .then_with(|| match (self, other) {
                (CanonicalPattern::Integer(a), CanonicalPattern::Integer(b)) => a.cmp(b),
                (CanonicalPattern::Rational(an, ad), CanonicalPattern::Rational(bn, bd)) => {
                    an.cmp(bn).then_with(|| ad.cmp(bd))
                }
                (CanonicalPattern::Float(a), CanonicalPattern::Float(b)) => a.cmp(b),
                (CanonicalPattern::Complex(ar, ai), CanonicalPattern::Complex(br, bi)) => {
                    ar.cmp(br).then_with(|| ai.cmp(bi))
                }
                (CanonicalPattern::Constant(a), CanonicalPattern::Constant(b)) => {
                    const_rank(a).cmp(&const_rank(b))
                }
                (CanonicalPattern::Slot(a), CanonicalPattern::Slot(b)) => a.cmp(b),
                (CanonicalPattern::Add(a), CanonicalPattern::Add(b)) => {
                    a.len().cmp(&b.len()).then_with(|| {
                        a.iter()
                            .zip(b.iter())
                            .map(|(x, y)| x.as_ref().cmp(y.as_ref()))
                            .find(|o| *o != Ordering::Equal)
                            .unwrap_or(Ordering::Equal)
                    })
                }
                (CanonicalPattern::Mul(a), CanonicalPattern::Mul(b)) => {
                    a.len().cmp(&b.len()).then_with(|| {
                        a.iter()
                            .zip(b.iter())
                            .map(|(x, y)| x.as_ref().cmp(y.as_ref()))
                            .find(|o| *o != Ordering::Equal)
                            .unwrap_or(Ordering::Equal)
                    })
                }
                (CanonicalPattern::Pow(ab, ae), CanonicalPattern::Pow(bb, be)) => ab
                    .as_ref()
                    .cmp(bb.as_ref())
                    .then_with(|| ae.as_ref().cmp(be.as_ref())),
                (CanonicalPattern::Func(fa, aa), CanonicalPattern::Func(fb, ab)) => {
                    fa.cmp(fb).then_with(|| {
                        aa.len().cmp(&ab.len()).then_with(|| {
                            aa.iter()
                                .zip(ab.iter())
                                .map(|(x, y)| x.as_ref().cmp(y.as_ref()))
                                .find(|o| *o != Ordering::Equal)
                                .unwrap_or(Ordering::Equal)
                        })
                    })
                }
                // Unreachable: discriminant equality implies same variant.
                _ => Ordering::Equal,
            })
    }
}

fn const_rank(c: &SymbolicConstant) -> u8 {
    match c {
        SymbolicConstant::Pi => 0,
        SymbolicConstant::E => 1,
        SymbolicConstant::I => 2,
    }
}

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
        Expr::Add(add) => {
            // Emit constant as first child (if non-zero), then sorted terms.
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
        Expr::Mul(mul) => {
            // Emit coefficient as first child (if != 1), then sorted factors.
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
                // Check if exponent is the integer 1 — if so, emit base only.
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

/// Extract (numerator, denominator) as i64 from a BigRational.
/// Falls back to i64::MAX / 1 for values exceeding i64 range.
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
        // Build x + y via AddNode manually
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
        // Slots 0 and 1 assigned (order depends on BTreeMap ordering)
        let sx = map.slot_of(x_id);
        let sy = map.slot_of(y_id);
        assert!(sx.is_some() && sy.is_some());
        assert_ne!(sx, sy);
    }

    #[test]
    fn fast_canonical_same_symbol_same_slot() {
        // x + x → both occurrences map to slot 0
        use crate::numeric::AddNode;
        use crate::numeric::BigRational;
        let x_id = SymbolId::intern("x2");
        let x1 = Arc::new(Expr::Symbol(x_id));
        let x2 = Arc::new(Expr::Symbol(x_id));
        let mut add = AddNode::zero();
        add.terms.insert(x1, BigRational::from_i64(1, 1));
        add.terms.insert(x2.clone(), BigRational::from_i64(2, 1));
        // BTreeMap deduplicates equal keys — we just canonicalize x2 alone
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
    fn fast_varmap_insert_lookup() {
        let mut m = VarMap::new();
        let id = SymbolId::intern("v");
        let slot = SlotId(3);
        m.insert(id, slot);
        assert_eq!(m.slot_of(id), Some(slot));
        assert_eq!(m.symbol_of(slot), Some(id));
        assert_eq!(m.len(), 1);
        assert!(!m.is_empty());
    }

    #[test]
    fn fast_varmap_empty() {
        let m = VarMap::new();
        assert_eq!(m.len(), 0);
        assert!(m.is_empty());
        assert_eq!(m.slot_of(SymbolId::intern("z")), None);
    }

    #[test]
    fn fast_pattern_hash_same_pattern_same_hash() {
        let (p1, _) = canonicalize(&sym("a"));
        let (p2, _) = canonicalize(&sym("b"));
        // Different names → same structural slot pattern → same hash
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
        // Different SymbolIds → different structural hashes
        assert_ne!(structural_hash(&e1), structural_hash(&e2));
    }

    #[test]
    fn fast_ord_integer_before_slot() {
        let i = CanonicalPattern::Integer(1);
        let s = CanonicalPattern::Slot(SlotId(0));
        assert!(i < s);
    }

    #[test]
    fn fast_ord_same_variant_by_value() {
        let a = CanonicalPattern::Integer(1);
        let b = CanonicalPattern::Integer(2);
        assert!(a < b);
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
