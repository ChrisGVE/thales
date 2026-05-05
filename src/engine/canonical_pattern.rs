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
//! Canonicalization and hashing functions live in [`super::canonicalize`].

use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::ast::SymbolicConstant;
use crate::numeric::{FuncId, SymbolId};

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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
}
