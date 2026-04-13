//! N-ary multiplicative canonical form: `coeff · Π(base_i ^ exp_i)`.
//!
//! [`MulNode`] stores a product as a rational coefficient times a map from
//! distinct bases to their exponents. Construction normalizes automatically:
//! combines like bases, removes exponent-0 factors, extracts numeric coefficient.

use super::BigRational;
use std::collections::BTreeMap;
use std::fmt;

/// A canonical multiplicative node: `coeff · Π(base ^ exponent)`.
///
/// Bases are keyed by a string representation for now (will be replaced
/// by `Arc<Expr>` once the new expression type is available in task 12).
/// Exponents are stored as strings too (representing arbitrary expressions).
///
/// # Invariants
///
/// - No factor has a zero exponent
/// - Factors are sorted by the `BTreeMap` key ordering
/// - Numeric coefficient is always extracted
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MulNode {
    /// The numeric coefficient.
    pub coeff: BigRational,
    /// Map from base representation to its exponent representation.
    pub factors: BTreeMap<String, String>,
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
    pub fn from_factor(base: String, exp: String) -> Self {
        let mut node = Self::one();
        if exp != "0" {
            node.factors.insert(base, exp);
        }
        node
    }

    /// Create a MulNode representing `coeff * base^1`.
    pub fn from_coeff_and_base(coeff: BigRational, base: String) -> Self {
        let mut node = MulNode {
            coeff,
            factors: BTreeMap::new(),
        };
        node.factors.insert(base, "1".to_string());
        node
    }

    /// Add a factor `base^exp`. If base already exists, adds exponents.
    ///
    /// Note: exponent addition is string-based placeholder. Will use
    /// proper expression addition once integrated with Expr type.
    pub fn add_factor(&mut self, base: String, exp: String) {
        if exp == "0" {
            return;
        }
        if let Some(existing) = self.factors.get(&base) {
            let new_exp = add_exponent_strings(existing, &exp);
            if new_exp == "0" {
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
            result
                .factors
                .insert(base.clone(), negate_exponent_string(exp));
        }
        result
    }
}

/// Placeholder exponent addition for string-based exponents.
/// Handles simple integer cases; complex expressions left as-is.
fn add_exponent_strings(a: &str, b: &str) -> String {
    if let (Ok(ai), Ok(bi)) = (a.parse::<i64>(), b.parse::<i64>()) {
        (ai + bi).to_string()
    } else {
        format!("({a}) + ({b})")
    }
}

/// Placeholder exponent negation for string-based exponents.
fn negate_exponent_string(s: &str) -> String {
    if let Ok(v) = s.parse::<i64>() {
        (-v).to_string()
    } else {
        format!("-({s})")
    }
}

use num::traits::{One, Zero};

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
            if exp == "1" {
                write!(f, "{base}")?;
            } else {
                write!(f, "{base}^{exp}")?;
            }
            first = false;
        }

        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
        // x * x = x^2
        let mut node = MulNode::from_factor("x".to_string(), "1".to_string());
        node.add_factor("x".to_string(), "1".to_string());
        assert_eq!(node.factor_count(), 1);
        assert_eq!(node.factors["x"], "2");
    }

    #[test]
    fn test_cancellation() {
        // x * x^(-1) = 1
        let mut node = MulNode::from_factor("x".to_string(), "1".to_string());
        node.add_factor("x".to_string(), "-1".to_string());
        assert!(node.is_one());
        assert_eq!(node.factor_count(), 0);
    }

    #[test]
    fn test_merge() {
        // (2*x) * (3*x*y) = 6*x^2*y
        let mut a = MulNode::from_coeff_and_base(BigRational::from(2i64), "x".to_string());
        let mut b = MulNode::from_coeff_and_base(BigRational::from(3i64), "x".to_string());
        b.add_factor("y".to_string(), "1".to_string());

        a.merge(&b);
        assert_eq!(a.coeff, BigRational::from(6i64));
        assert_eq!(a.factors["x"], "2");
        assert_eq!(a.factors["y"], "1");
    }

    #[test]
    fn test_reciprocal() {
        // 1/(2*x^3) = (1/2)*x^(-3)
        let mut node = MulNode::from_coeff(BigRational::from(2i64));
        node.add_factor("x".to_string(), "3".to_string());
        let recip = node.reciprocal();
        assert_eq!(recip.coeff, BigRational::from_i64(1, 2));
        assert_eq!(recip.factors["x"], "-3");
    }

    #[test]
    fn test_display() {
        let mut node = MulNode::from_coeff_and_base(BigRational::from(2i64), "x".to_string());
        node.add_factor("y".to_string(), "3".to_string());
        let s = node.to_string();
        assert!(s.contains("2"));
        assert!(s.contains("x"));
        assert!(s.contains("y"));
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
        let node = MulNode::from_factor("x".to_string(), "0".to_string());
        assert!(node.is_one());
        assert_eq!(node.factor_count(), 0);
    }
}
