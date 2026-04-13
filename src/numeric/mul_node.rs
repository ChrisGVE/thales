//! N-ary multiplicative canonical form: `coeff · Π(base_i ^ exp_i)`.
//!
//! [`MulNode`] stores a product as a rational coefficient times a map from
//! distinct bases to their exponents. Construction normalizes automatically:
//! combines like bases, removes exponent-0 factors, extracts numeric coefficient.

use super::BigRational;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use super::expr::{add_exponents, negate_exponent, Expr};

/// A canonical multiplicative node: `coeff · Π(base ^ exponent)`.
///
/// Bases and exponents are `Arc<Expr>`, enabling structural sharing
/// and O(1) clone of sub-expressions.
///
/// # Invariants
///
/// - No factor has a zero exponent
/// - Factors are sorted by the `BTreeMap` key ordering (structural `Ord` on `Expr`)
/// - Numeric coefficient is always extracted
#[derive(Clone, Debug)]
pub struct MulNode {
    /// The numeric coefficient.
    pub coeff: BigRational,
    /// Map from base expression to its exponent expression.
    pub factors: BTreeMap<Arc<Expr>, Arc<Expr>>,
}

impl MulNode {
    /// Create a new MulNode representing 1 (multiplicative identity).
    pub fn one() -> Self {
        MulNode {
            coeff: BigRational::one(),
            factors: BTreeMap::new(),
        }
    }

    /// Create a MulNode representing zero.
    pub fn zero() -> Self {
        MulNode {
            coeff: BigRational::zero(),
            factors: BTreeMap::new(),
        }
    }

    /// Create a MulNode from a coefficient only.
    pub fn from_coeff(c: BigRational) -> Self {
        MulNode {
            coeff: c,
            factors: BTreeMap::new(),
        }
    }

    /// Create a MulNode representing `base^exp`.
    pub fn from_factor(base: Arc<Expr>, exp: Arc<Expr>) -> Self {
        let mut node = Self::one();
        if !exp.is_zero() {
            node.factors.insert(base, exp);
        }
        node
    }

    /// Create a MulNode representing `coeff * base^1`.
    pub fn from_coeff_and_base(coeff: BigRational, base: Arc<Expr>) -> Self {
        let mut node = MulNode {
            coeff,
            factors: BTreeMap::new(),
        };
        node.factors.insert(base, Expr::int(1));
        node
    }

    /// Add a factor `base^exp`. If base already exists, adds exponents.
    pub fn add_factor(&mut self, base: Arc<Expr>, exp: Arc<Expr>) {
        if exp.is_zero() {
            return;
        }
        if let Some(existing) = self.factors.get(&base) {
            let new_exp = add_exponents(existing, &exp);
            if new_exp.is_zero() {
                self.factors.remove(&base);
            } else {
                self.factors.insert(base, new_exp);
            }
        } else {
            self.factors.insert(base, exp);
        }
    }

    /// Multiply by a scalar coefficient.
    pub fn scale(&mut self, factor: &BigRational) {
        self.coeff = &self.coeff * factor;
    }

    /// Merge another MulNode into this one (multiply).
    pub fn merge(&mut self, other: &MulNode) {
        self.scale(&other.coeff);
        for (base, exp) in &other.factors {
            self.add_factor(base.clone(), exp.clone());
        }
    }

    /// Returns the number of distinct base factors.
    pub fn factor_count(&self) -> usize {
        self.factors.len()
    }

    /// Returns `true` if this is just a coefficient (no variable factors).
    pub fn is_constant(&self) -> bool {
        self.factors.is_empty()
    }

    /// Returns `true` if this represents zero.
    pub fn is_zero(&self) -> bool {
        self.coeff.is_zero()
    }

    /// Returns `true` if this represents one.
    pub fn is_one(&self) -> bool {
        self.coeff.is_one() && self.factors.is_empty()
    }

    /// Reciprocal: 1/self. Negates all exponents and inverts coefficient.
    pub fn reciprocal(&self) -> Self {
        assert!(!self.is_zero(), "reciprocal of zero");
        let mut result = MulNode {
            coeff: self.coeff.recip(),
            factors: BTreeMap::new(),
        };
        for (base, exp) in &self.factors {
            result.factors.insert(base.clone(), negate_exponent(exp));
        }
        result
    }
}

use num::traits::{One, Zero};

// ── Equality ─────────────────────────────────────────────────────────────────

impl PartialEq for MulNode {
    fn eq(&self, other: &Self) -> bool {
        self.coeff == other.coeff && self.factors == other.factors
    }
}

impl Eq for MulNode {}

// ── Hashing ──────────────────────────────────────────────────────────────────

impl Hash for MulNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.coeff.hash(state);
        for (base, exp) in &self.factors {
            base.hash(state);
            exp.hash(state);
        }
    }
}

// ── Ordering ─────────────────────────────────────────────────────────────────

impl PartialOrd for MulNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MulNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.coeff.cmp(&other.coeff).then_with(|| {
            let mut a_iter = self.factors.iter();
            let mut b_iter = other.factors.iter();
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

impl fmt::Display for MulNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;

        if !self.coeff.is_one() || self.factors.is_empty() {
            write!(f, "{}", self.coeff)?;
            first = false;
        }

        for (base, exp) in &self.factors {
            if !first {
                write!(f, "*")?;
            }
            if exp.is_one() {
                write!(f, "{base}")?;
            } else {
                write!(f, "{base}^{exp}")?;
            }
            first = false;
        }

        Ok(())
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::SmallInt;

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }

    fn int_exp(n: i64) -> Arc<Expr> {
        Expr::int(n)
    }

    #[test]
    fn test_one() {
        let node = MulNode::one();
        assert!(node.is_one());
        assert!(node.is_constant());
        assert_eq!(node.factor_count(), 0);
    }

    #[test]
    fn test_zero() {
        let node = MulNode::zero();
        assert!(node.is_zero());
    }

    #[test]
    fn test_combine_like_bases() {
        let x = sym("mul_clb_x");
        let mut node = MulNode::from_factor(x.clone(), int_exp(1));
        node.add_factor(x.clone(), int_exp(1));
        assert_eq!(node.factor_count(), 1);
        assert_eq!(*node.factors[&x], Expr::Integer(SmallInt::from(2i64)));
    }

    #[test]
    fn test_cancellation() {
        let x = sym("mul_cancel_x");
        let mut node = MulNode::from_factor(x.clone(), int_exp(1));
        node.add_factor(x, int_exp(-1));
        assert!(node.is_one());
        assert_eq!(node.factor_count(), 0);
    }

    #[test]
    fn test_merge() {
        let x = sym("mul_merge_x");
        let y = sym("mul_merge_y");

        let mut a = MulNode::from_coeff_and_base(BigRational::from(2i64), x.clone());
        let mut b = MulNode::from_coeff_and_base(BigRational::from(3i64), x.clone());
        b.add_factor(y.clone(), int_exp(1));

        a.merge(&b);
        assert_eq!(a.coeff, BigRational::from(6i64));
        assert_eq!(*a.factors[&x], Expr::Integer(SmallInt::from(2i64)));
        assert_eq!(*a.factors[&y], Expr::Integer(SmallInt::from(1i64)));
    }

    #[test]
    fn test_reciprocal() {
        let x = sym("mul_recip_x");
        let mut node = MulNode::from_coeff(BigRational::from(2i64));
        node.add_factor(x.clone(), int_exp(3));
        let recip = node.reciprocal();
        assert_eq!(recip.coeff, BigRational::from_i64(1, 2));
        assert_eq!(*recip.factors[&x], Expr::Integer(SmallInt::from(-3i64)));
    }

    #[test]
    fn test_display() {
        let x = sym("mul_disp_x");
        let y = sym("mul_disp_y");
        let mut node = MulNode::from_coeff_and_base(BigRational::from(2i64), x);
        node.add_factor(y, int_exp(3));
        let s = node.to_string();
        assert!(s.contains("2"));
    }

    #[test]
    fn test_display_one() {
        assert_eq!(MulNode::one().to_string(), "1");
    }

    #[test]
    fn test_display_zero() {
        assert_eq!(MulNode::zero().to_string(), "0");
    }

    #[test]
    fn test_zero_exponent_not_stored() {
        let x = sym("mul_ze_x");
        let node = MulNode::from_factor(x, int_exp(0));
        assert!(node.is_one());
        assert_eq!(node.factor_count(), 0);
    }

    #[test]
    fn test_equality() {
        let x = sym("mul_eq_x");
        let a = MulNode::from_factor(x.clone(), int_exp(2));
        let b = MulNode::from_factor(x, int_exp(2));
        assert_eq!(a, b);
    }

    #[test]
    fn test_ordering() {
        let a = MulNode::from_coeff(BigRational::from(1i64));
        let b = MulNode::from_coeff(BigRational::from(2i64));
        assert!(a < b);
    }
}
