//! N-ary additive canonical form: `constant + Σ(coeff_i · term_i)`.
//!
//! [`AddNode`] stores a sum as a rational constant plus a map from
//! distinct terms to their coefficients. Construction normalizes
//! automatically: combines like terms, removes zero coefficients,
//! and extracts numeric constants.

use super::BigRational;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use super::expr::Expr;

/// A canonical additive node: `constant + Σ(coeff · term)`.
///
/// Terms are keyed by `Arc<Expr>`, enabling structural sharing
/// and O(1) clone of sub-expressions.
///
/// # Invariants
///
/// - No term has a zero coefficient
/// - Terms are sorted by the `BTreeMap` key ordering (structural `Ord` on `Expr`)
/// - Numeric values are folded into `constant`
#[derive(Clone, Debug)]
pub struct AddNode {
    /// The constant (numeric) part of the sum.
    pub constant: BigRational,
    /// Map from term to its coefficient.
    pub terms: BTreeMap<Arc<Expr>, BigRational>,
}

impl AddNode {
    /// Create a new empty sum (equal to zero).
    pub fn zero() -> Self {
        AddNode {
            constant: BigRational::zero(),
            terms: BTreeMap::new(),
        }
    }

    /// Create an AddNode representing a single constant.
    pub fn from_constant(c: BigRational) -> Self {
        AddNode {
            constant: c,
            terms: BTreeMap::new(),
        }
    }

    /// Create an AddNode representing `coeff * term`.
    pub fn from_term(term: Arc<Expr>, coeff: BigRational) -> Self {
        let mut node = Self::zero();
        if !coeff.is_zero() {
            node.terms.insert(term, coeff);
        }
        node
    }

    /// Add a term with coefficient. Combines with existing if present.
    pub fn add_term(&mut self, term: Arc<Expr>, coeff: BigRational) {
        if coeff.is_zero() {
            return;
        }
        if let Some(existing) = self.terms.get(&term) {
            let new_coeff = existing + &coeff;
            if new_coeff.is_zero() {
                self.terms.remove(&term);
            } else {
                self.terms.insert(term, new_coeff);
            }
        } else {
            self.terms.insert(term, coeff);
        }
    }

    /// Add a constant to the sum.
    pub fn add_constant(&mut self, c: BigRational) {
        self.constant = &self.constant + &c;
    }

    /// Merge another AddNode into this one.
    pub fn merge(&mut self, other: &AddNode) {
        self.add_constant(other.constant.clone());
        for (term, coeff) in &other.terms {
            self.add_term(term.clone(), coeff.clone());
        }
    }

    /// Returns the number of distinct non-zero terms (excluding constant).
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    /// Returns `true` if this is just a constant (no variable terms).
    pub fn is_constant(&self) -> bool {
        self.terms.is_empty()
    }

    /// Returns `true` if this represents zero.
    pub fn is_zero(&self) -> bool {
        self.constant.is_zero() && self.terms.is_empty()
    }

    /// Negate the entire sum.
    pub fn negate(&self) -> Self {
        let mut result = AddNode {
            constant: -&self.constant,
            terms: BTreeMap::new(),
        };
        for (term, coeff) in &self.terms {
            result.terms.insert(term.clone(), -coeff);
        }
        result
    }

    /// Scale the entire sum by a constant factor.
    pub fn scale(&self, factor: &BigRational) -> Self {
        if factor.is_zero() {
            return Self::zero();
        }
        let mut result = AddNode {
            constant: &self.constant * factor,
            terms: BTreeMap::new(),
        };
        for (term, coeff) in &self.terms {
            let new_coeff = coeff * factor;
            if !new_coeff.is_zero() {
                result.terms.insert(term.clone(), new_coeff);
            }
        }
        result
    }
}

use num::traits::Zero;

// ── Equality ─────────────────────────────────────────────────────────────────

impl PartialEq for AddNode {
    fn eq(&self, other: &Self) -> bool {
        self.constant == other.constant && self.terms == other.terms
    }
}

impl Eq for AddNode {}

// ── Hashing ──────────────────────────────────────────────────────────────────

impl Hash for AddNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.constant.hash(state);
        // BTreeMap iteration is ordered, so hashing is deterministic.
        for (term, coeff) in &self.terms {
            term.hash(state);
            coeff.hash(state);
        }
    }
}

// ── Ordering ─────────────────────────────────────────────────────────────────

impl PartialOrd for AddNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AddNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.constant.cmp(&other.constant).then_with(|| {
            let mut a_iter = self.terms.iter();
            let mut b_iter = other.terms.iter();
            loop {
                match (a_iter.next(), b_iter.next()) {
                    (None, None) => return Ordering::Equal,
                    (None, Some(_)) => return Ordering::Less,
                    (Some(_), None) => return Ordering::Greater,
                    (Some((ak, av)), Some((bk, bv))) => {
                        let c = ak.cmp(bk).then_with(|| av.cmp(bv));
                        if c != Ordering::Equal {
                            return c;
                        }
                    }
                }
            }
        })
    }
}

// ── Display ──────────────────────────────────────────────────────────────────

impl fmt::Display for AddNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;

        if !self.constant.is_zero() || self.terms.is_empty() {
            write!(f, "{}", self.constant)?;
            first = false;
        }

        for (term, coeff) in &self.terms {
            if first {
                if coeff.is_one() {
                    write!(f, "{term}")?;
                } else if *coeff == BigRational::from(-1i64) {
                    write!(f, "-{term}")?;
                } else {
                    write!(f, "{coeff}*{term}")?;
                }
                first = false;
            } else if coeff.is_negative() {
                if *coeff == BigRational::from(-1i64) {
                    write!(f, " - {term}")?;
                } else {
                    write!(f, " - {}*{term}", coeff.abs())?;
                }
            } else if coeff.is_one() {
                write!(f, " + {term}")?;
            } else {
                write!(f, " + {coeff}*{term}")?;
            }
        }

        Ok(())
    }
}

use num::traits::One;

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }

    #[test]
    fn test_zero() {
        let node = AddNode::zero();
        assert!(node.is_zero());
        assert!(node.is_constant());
        assert_eq!(node.term_count(), 0);
    }

    #[test]
    fn test_from_constant() {
        let node = AddNode::from_constant(BigRational::from(5i64));
        assert!(node.is_constant());
        assert!(!node.is_zero());
        assert_eq!(node.constant, BigRational::from(5i64));
    }

    #[test]
    fn test_combine_like_terms() {
        let x = sym("add_clt_x");
        let mut node = AddNode::from_term(x.clone(), BigRational::from(1i64));
        node.add_term(x.clone(), BigRational::from(2i64));
        assert_eq!(node.term_count(), 1);
        assert_eq!(node.terms[&x], BigRational::from(3i64));
    }

    #[test]
    fn test_cancellation() {
        let x = sym("add_cancel_x");
        let mut node = AddNode::from_term(x.clone(), BigRational::from(1i64));
        node.add_term(x, BigRational::from(-1i64));
        assert!(node.is_zero());
        assert_eq!(node.term_count(), 0);
    }

    #[test]
    fn test_merge() {
        let x = sym("add_merge_x");
        let y = sym("add_merge_y");

        let mut a = AddNode::from_term(x.clone(), BigRational::from(1i64));
        a.add_constant(BigRational::from(3i64));

        let mut b = AddNode::from_term(x.clone(), BigRational::from(2i64));
        b.add_term(y.clone(), BigRational::from(1i64));
        b.add_constant(BigRational::from(1i64));

        a.merge(&b);
        assert_eq!(a.constant, BigRational::from(4i64));
        assert_eq!(a.terms[&x], BigRational::from(3i64));
        assert_eq!(a.terms[&y], BigRational::from(1i64));
        assert_eq!(a.term_count(), 2);
    }

    #[test]
    fn test_negate() {
        let x = sym("add_neg_x");
        let mut node = AddNode::from_term(x.clone(), BigRational::from(2i64));
        node.add_constant(BigRational::from(3i64));
        let neg = node.negate();
        assert_eq!(neg.constant, BigRational::from(-3i64));
        assert_eq!(neg.terms[&x], BigRational::from(-2i64));
    }

    #[test]
    fn test_scale() {
        let x = sym("add_scale_x");
        let mut node = AddNode::from_term(x.clone(), BigRational::from(3i64));
        node.add_constant(BigRational::from(1i64));
        let scaled = node.scale(&BigRational::from(2i64));
        assert_eq!(scaled.constant, BigRational::from(2i64));
        assert_eq!(scaled.terms[&x], BigRational::from(6i64));
    }

    #[test]
    fn test_scale_by_zero() {
        let x = sym("add_scale0_x");
        let node = AddNode::from_term(x, BigRational::from(3i64));
        let scaled = node.scale(&BigRational::zero());
        assert!(scaled.is_zero());
    }

    #[test]
    fn test_display() {
        let x = sym("add_disp_x");
        let y = sym("add_disp_y");
        let mut node = AddNode::from_term(x, BigRational::from(2i64));
        node.add_term(y, BigRational::from(1i64));
        node.add_constant(BigRational::from(3i64));
        let s = node.to_string();
        assert!(s.contains("3"));
    }

    #[test]
    fn test_display_zero() {
        assert_eq!(AddNode::zero().to_string(), "0");
    }

    #[test]
    fn test_zero_coefficient_not_stored() {
        let x = sym("add_zc_x");
        let node = AddNode::from_term(x, BigRational::zero());
        assert!(node.is_zero());
        assert_eq!(node.term_count(), 0);
    }

    #[test]
    fn test_equality() {
        let x = sym("add_eq_x");
        let a = AddNode::from_term(x.clone(), BigRational::from(2i64));
        let b = AddNode::from_term(x, BigRational::from(2i64));
        assert_eq!(a, b);
    }

    #[test]
    fn test_ordering() {
        let a = AddNode::from_constant(BigRational::from(1i64));
        let b = AddNode::from_constant(BigRational::from(2i64));
        assert!(a < b);
    }
}
