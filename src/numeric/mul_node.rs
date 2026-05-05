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
        use super::expr::{fmt_maybe_paren, needs_parens_as_mul_factor, needs_parens_as_pow_base};

        if self.is_zero() {
            return write!(f, "0");
        }

        // Split factors into numerator (exponent != -1) and denominator
        // (integer exponent == -1, rendered after "/").
        let mut numer_parts: Vec<(&Arc<Expr>, &Arc<Expr>)> = Vec::new();
        let mut denom_parts: Vec<&Arc<Expr>> = Vec::new();

        for (base, exp) in &self.factors {
            if matches!(exp.as_ref(), Expr::Integer(n) if *n == super::SmallInt::from(-1i64)) {
                denom_parts.push(base);
            } else {
                numer_parts.push((base, exp));
            }
        }

        let neg_one = BigRational::from(-1i64);
        let coeff_is_one = self.coeff.is_one();
        let coeff_is_neg_one = self.coeff == neg_one;
        let has_numer = !numer_parts.is_empty();
        let _has_any_factors = has_numer || !denom_parts.is_empty();

        // `printed_coeff` is true when a numeric coefficient token was emitted
        // and the *next* numerator factor needs no `*` separator (e.g. `2x`).
        // `printed_something` is the general "a `*` is needed before the next
        // factor" flag (used between successive numerator factors).
        let mut printed_coeff = false;
        let mut printed_something = false;

        if coeff_is_neg_one && has_numer {
            write!(f, "-")?;
        } else if !coeff_is_one || !has_numer {
            // Always print the coefficient when there are no numerator parts
            // so that "1/x" renders as "1/x" rather than "/x".
            write!(f, "{}", self.coeff)?;
            printed_coeff = true;
            printed_something = true;
        }

        for (i, (base, exp)) in numer_parts.iter().enumerate() {
            if printed_something && !(i == 0 && printed_coeff) {
                write!(f, "*")?;
            }
            if exp.is_one() {
                let wrap = needs_parens_as_mul_factor(base);
                fmt_maybe_paren(base, wrap, f)?;
            } else {
                let wrap_base = needs_parens_as_pow_base(base);
                let wrap_exp =
                    matches!(exp.as_ref(), Expr::Add(_) | Expr::Mul(_) | Expr::Pow(_, _));
                fmt_maybe_paren(base, wrap_base, f)?;
                write!(f, "^")?;
                fmt_maybe_paren(exp, wrap_exp, f)?;
            }
            printed_something = true;
        }

        if !denom_parts.is_empty() {
            write!(f, "/")?;
            if denom_parts.len() == 1 {
                let base = denom_parts[0];
                fmt_maybe_paren(base, needs_parens_as_mul_factor(base), f)?;
            } else {
                write!(f, "(")?;
                for (i, base) in denom_parts.iter().enumerate() {
                    if i > 0 {
                        write!(f, "*")?;
                    }
                    fmt_maybe_paren(base, needs_parens_as_mul_factor(base), f)?;
                }
                write!(f, ")")?;
            }
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

    // ── Display improvement tests ────────────────────────────────────────────

    /// `2*x` should display as `2x` (no explicit `*` between coeff and symbol).
    #[test]
    fn test_display_coeff_times_symbol() {
        let x = sym("x");
        let node = MulNode::from_coeff_and_base(BigRational::from(2i64), x);
        assert_eq!(node.to_string(), "2x");
    }

    /// `(-1)*x` should display as `-x`.
    #[test]
    fn test_display_neg_one_coeff() {
        let x = sym("x");
        let node = MulNode::from_coeff_and_base(BigRational::from(-1i64), x);
        assert_eq!(node.to_string(), "-x");
    }

    /// `(-2)*x` should display as `-2x`.
    #[test]
    fn test_display_neg_coeff() {
        let x = sym("x");
        let node = MulNode::from_coeff_and_base(BigRational::from(-2i64), x);
        assert_eq!(node.to_string(), "-2x");
    }

    /// `x * y^(-1)` should display as `x/y`.
    #[test]
    fn test_display_division() {
        let x = sym("x");
        let y = sym("y");
        let mut node = MulNode::from_factor(x, int_exp(1));
        node.add_factor(y, int_exp(-1));
        assert_eq!(node.to_string(), "x/y");
    }

    /// `x^2` (coeff=1, single factor with exp=2) should display as `x^2`.
    #[test]
    fn test_display_power() {
        let x = sym("x");
        let node = MulNode::from_factor(x, int_exp(2));
        assert_eq!(node.to_string(), "x^2");
    }

    /// `1/x` (coeff=1, x^-1) should display as `1/x`.
    #[test]
    fn test_display_reciprocal_only() {
        let x = sym("x");
        let node = MulNode::from_factor(x, int_exp(-1));
        assert_eq!(node.to_string(), "1/x");
    }

    /// `x / (y*z)` should display as `x/(y*z)`.
    #[test]
    fn test_display_multi_denom() {
        let x = sym("mul_md_x");
        let y = sym("mul_md_y");
        let z = sym("mul_md_z");
        let mut node = MulNode::from_factor(x, int_exp(1));
        node.add_factor(y, int_exp(-1));
        node.add_factor(z, int_exp(-1));
        let s = node.to_string();
        assert!(s.contains("/("), "expected grouped denominator, got: {s}");
    }
}
