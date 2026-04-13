//! Exact rational number type using [`SmallInt`] for numerator and denominator.
//!
//! Automatically normalizes on construction (GCD reduction, positive denominator).
//! Arithmetic preserves normalized form.

use super::SmallInt;
use num::traits::{One, Signed, Zero};
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// An exact rational number `num / denom` where both components are [`SmallInt`].
///
/// Invariants maintained by construction:
/// - `denom > 0` (sign carried by numerator)
/// - `gcd(|num|, denom) == 1` (fully reduced)
/// - Zero is represented as `0/1`
#[derive(Clone, Debug)]
pub struct BigRational {
    num: SmallInt,
    denom: SmallInt,
}

// ── Construction ──────────────────────────────────────────────────────────────

impl BigRational {
    /// Create a new rational and normalize it.
    ///
    /// # Panics
    ///
    /// Panics if `denom` is zero.
    pub fn new(num: SmallInt, denom: SmallInt) -> Self {
        assert!(!denom.is_zero(), "BigRational: denominator is zero");
        Self::normalize(num, denom)
    }

    /// Create from `i64` numerator and denominator.
    pub fn from_i64(num: i64, denom: i64) -> Self {
        Self::new(SmallInt::from(num), SmallInt::from(denom))
    }

    /// Create an integer-valued rational (denom = 1).
    pub fn from_integer(n: SmallInt) -> Self {
        BigRational {
            num: n,
            denom: SmallInt::from(1i64),
        }
    }

    /// Normalize: reduce by GCD and ensure positive denominator.
    fn normalize(num: SmallInt, denom: SmallInt) -> Self {
        if num.is_zero() {
            return BigRational {
                num: SmallInt::from(0i64),
                denom: SmallInt::from(1i64),
            };
        }

        let g = num.gcd(&denom);
        let mut n = &num / &g;
        let mut d = &denom / &g;

        // Ensure positive denominator
        if d.is_negative() {
            n = -n;
            d = -d;
        }

        BigRational { num: n, denom: d }
    }

    /// Returns the numerator.
    pub fn numer(&self) -> &SmallInt {
        &self.num
    }

    /// Returns the denominator (always positive).
    pub fn denom(&self) -> &SmallInt {
        &self.denom
    }

    /// Convert to `f64`.
    pub fn to_f64(&self) -> f64 {
        match (self.num.to_i64(), self.denom.to_i64()) {
            (Some(n), Some(d)) => n as f64 / d as f64,
            _ => {
                // Fallback for big values
                let n: f64 = self.num.to_bigint().to_string().parse().unwrap_or(f64::NAN);
                let d: f64 = self
                    .denom
                    .to_bigint()
                    .to_string()
                    .parse()
                    .unwrap_or(f64::NAN);
                n / d
            }
        }
    }

    /// Returns `true` if this is an integer (denominator is 1).
    pub fn is_integer(&self) -> bool {
        self.denom.is_one()
    }

    /// Reciprocal (1/self). Panics if self is zero.
    pub fn recip(&self) -> Self {
        assert!(!self.is_zero(), "reciprocal of zero");
        BigRational::new(self.denom.clone(), self.num.clone())
    }
}

// ── Zero / One ────────────────────────────────────────────────────────────────

impl Zero for BigRational {
    fn zero() -> Self {
        BigRational {
            num: SmallInt::from(0i64),
            denom: SmallInt::from(1i64),
        }
    }

    fn is_zero(&self) -> bool {
        self.num.is_zero()
    }
}

impl One for BigRational {
    fn one() -> Self {
        BigRational {
            num: SmallInt::from(1i64),
            denom: SmallInt::from(1i64),
        }
    }

    fn is_one(&self) -> bool
    where
        Self: PartialEq,
    {
        self.num.is_one() && self.denom.is_one()
    }
}

// ── Display ───────────────────────────────────────────────────────────────────

impl fmt::Display for BigRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denom.is_one() {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.denom)
        }
    }
}

// ── Equality and Ordering ─────────────────────────────────────────────────────

impl PartialEq for BigRational {
    fn eq(&self, other: &Self) -> bool {
        // Both are normalized, so direct comparison works
        self.num == other.num && self.denom == other.denom
    }
}

impl Eq for BigRational {}

impl PartialOrd for BigRational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigRational {
    fn cmp(&self, other: &Self) -> Ordering {
        // a/b vs c/d → a*d vs c*b (denominators always positive)
        let lhs = &self.num * &other.denom;
        let rhs = &other.num * &self.denom;
        lhs.cmp(&rhs)
    }
}

impl Hash for BigRational {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Both are normalized, so hash components directly
        self.num.hash(state);
        self.denom.hash(state);
    }
}

// ── Arithmetic: Add ───────────────────────────────────────────────────────────

impl Add for BigRational {
    type Output = BigRational;

    fn add(self, rhs: Self) -> Self::Output {
        // a/b + c/d = (a*d + c*b) / (b*d)
        let num = &self.num * &rhs.denom + &rhs.num * &self.denom;
        let denom = &self.denom * &rhs.denom;
        BigRational::normalize(num, denom)
    }
}

impl Add for &BigRational {
    type Output = BigRational;

    fn add(self, rhs: Self) -> Self::Output {
        let num = &self.num * &rhs.denom + &rhs.num * &self.denom;
        let denom = &self.denom * &rhs.denom;
        BigRational::normalize(num, denom)
    }
}

// ── Arithmetic: Sub ───────────────────────────────────────────────────────────

impl Sub for BigRational {
    type Output = BigRational;

    fn sub(self, rhs: Self) -> Self::Output {
        let num = &self.num * &rhs.denom - &rhs.num * &self.denom;
        let denom = &self.denom * &rhs.denom;
        BigRational::normalize(num, denom)
    }
}

impl Sub for &BigRational {
    type Output = BigRational;

    fn sub(self, rhs: Self) -> Self::Output {
        let num = &self.num * &rhs.denom - &rhs.num * &self.denom;
        let denom = &self.denom * &rhs.denom;
        BigRational::normalize(num, denom)
    }
}

// ── Arithmetic: Mul ───────────────────────────────────────────────────────────

impl Mul for BigRational {
    type Output = BigRational;

    fn mul(self, rhs: Self) -> Self::Output {
        let num = &self.num * &rhs.num;
        let denom = &self.denom * &rhs.denom;
        BigRational::normalize(num, denom)
    }
}

impl Mul for &BigRational {
    type Output = BigRational;

    fn mul(self, rhs: Self) -> Self::Output {
        let num = &self.num * &rhs.num;
        let denom = &self.denom * &rhs.denom;
        BigRational::normalize(num, denom)
    }
}

// ── Arithmetic: Div ───────────────────────────────────────────────────────────

impl Div for BigRational {
    type Output = BigRational;

    /// Panics if rhs is zero.
    fn div(self, rhs: Self) -> Self::Output {
        assert!(!rhs.is_zero(), "division by zero");
        let num = &self.num * &rhs.denom;
        let denom = &self.denom * &rhs.num;
        BigRational::normalize(num, denom)
    }
}

impl Div for &BigRational {
    type Output = BigRational;

    fn div(self, rhs: Self) -> Self::Output {
        assert!(!rhs.is_zero(), "division by zero");
        let num = &self.num * &rhs.denom;
        let denom = &self.denom * &rhs.num;
        BigRational::normalize(num, denom)
    }
}

// ── Arithmetic: Neg ───────────────────────────────────────────────────────────

impl Neg for BigRational {
    type Output = BigRational;

    fn neg(self) -> Self::Output {
        BigRational {
            num: -self.num,
            denom: self.denom,
        }
    }
}

impl Neg for &BigRational {
    type Output = BigRational;

    fn neg(self) -> Self::Output {
        BigRational {
            num: -&self.num,
            denom: self.denom.clone(),
        }
    }
}

// ── Signed ────────────────────────────────────────────────────────────────────

impl BigRational {
    /// Returns `true` if positive.
    pub fn is_positive(&self) -> bool {
        self.num.is_positive()
    }

    /// Returns `true` if negative.
    pub fn is_negative(&self) -> bool {
        self.num.is_negative()
    }

    /// Absolute value.
    pub fn abs(&self) -> Self {
        BigRational {
            num: self.num.abs(),
            denom: self.denom.clone(),
        }
    }
}

// ── Conversions ───────────────────────────────────────────────────────────────

impl From<i64> for BigRational {
    fn from(v: i64) -> Self {
        BigRational::from_integer(SmallInt::from(v))
    }
}

impl From<i32> for BigRational {
    fn from(v: i32) -> Self {
        BigRational::from_integer(SmallInt::from(v))
    }
}

impl From<SmallInt> for BigRational {
    fn from(v: SmallInt) -> Self {
        BigRational::from_integer(v)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalization() {
        let r = BigRational::from_i64(6, 4);
        assert_eq!(r.numer().to_i64(), Some(3));
        assert_eq!(r.denom().to_i64(), Some(2));
    }

    #[test]
    fn test_negative_denominator_normalized() {
        let r = BigRational::from_i64(-1, -2);
        assert_eq!(r.numer().to_i64(), Some(1));
        assert_eq!(r.denom().to_i64(), Some(2));
    }

    #[test]
    fn test_zero_normalization() {
        let r = BigRational::from_i64(0, 5);
        assert_eq!(r.numer().to_i64(), Some(0));
        assert_eq!(r.denom().to_i64(), Some(1));
    }

    #[test]
    #[should_panic(expected = "denominator is zero")]
    fn test_zero_denominator_panics() {
        BigRational::from_i64(1, 0);
    }

    #[test]
    fn test_addition() {
        let a = BigRational::from_i64(1, 3);
        let b = BigRational::from_i64(1, 6);
        let sum = &a + &b;
        assert_eq!(sum.numer().to_i64(), Some(1));
        assert_eq!(sum.denom().to_i64(), Some(2));
    }

    #[test]
    fn test_subtraction() {
        let a = BigRational::from_i64(1, 2);
        let b = BigRational::from_i64(1, 3);
        let diff = &a - &b;
        assert_eq!(diff.numer().to_i64(), Some(1));
        assert_eq!(diff.denom().to_i64(), Some(6));
    }

    #[test]
    fn test_multiplication() {
        let a = BigRational::from_i64(2, 3);
        let b = BigRational::from_i64(3, 4);
        let prod = &a * &b;
        assert_eq!(prod.numer().to_i64(), Some(1));
        assert_eq!(prod.denom().to_i64(), Some(2));
    }

    #[test]
    fn test_division() {
        let a = BigRational::from_i64(1, 2);
        let b = BigRational::from_i64(3, 4);
        let quot = &a / &b;
        assert_eq!(quot.numer().to_i64(), Some(2));
        assert_eq!(quot.denom().to_i64(), Some(3));
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn test_division_by_zero_panics() {
        let a = BigRational::from_i64(1, 2);
        let b = BigRational::from_i64(0, 1);
        let _ = &a / &b;
    }

    #[test]
    fn test_negation() {
        let r = BigRational::from_i64(3, 4);
        let neg = -&r;
        assert_eq!(neg.numer().to_i64(), Some(-3));
        assert_eq!(neg.denom().to_i64(), Some(4));
    }

    #[test]
    fn test_equality() {
        assert_eq!(BigRational::from_i64(2, 4), BigRational::from_i64(1, 2));
        assert_ne!(BigRational::from_i64(1, 2), BigRational::from_i64(1, 3));
    }

    #[test]
    fn test_ordering() {
        assert!(BigRational::from_i64(1, 3) < BigRational::from_i64(1, 2));
        assert!(BigRational::from_i64(-1, 2) < BigRational::from_i64(1, 2));
    }

    #[test]
    fn test_display() {
        assert_eq!(BigRational::from_i64(3, 4).to_string(), "3/4");
        assert_eq!(BigRational::from_i64(6, 1).to_string(), "6");
        assert_eq!(BigRational::from_i64(0, 5).to_string(), "0");
    }

    #[test]
    fn test_to_f64() {
        let r = BigRational::from_i64(1, 4);
        assert!((r.to_f64() - 0.25).abs() < 1e-15);
    }

    #[test]
    fn test_is_integer() {
        assert!(BigRational::from_i64(6, 3).is_integer()); // normalizes to 2/1
        assert!(!BigRational::from_i64(1, 3).is_integer());
    }

    #[test]
    fn test_recip() {
        let r = BigRational::from_i64(3, 4);
        let inv = r.recip();
        assert_eq!(inv.numer().to_i64(), Some(4));
        assert_eq!(inv.denom().to_i64(), Some(3));
    }

    #[test]
    fn test_abs() {
        let r = BigRational::from_i64(-3, 4);
        let a = r.abs();
        assert_eq!(a.numer().to_i64(), Some(3));
        assert_eq!(a.denom().to_i64(), Some(4));
    }

    #[test]
    fn test_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;

        let a = BigRational::from_i64(2, 4); // normalizes to 1/2
        let b = BigRational::from_i64(3, 6); // normalizes to 1/2

        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn test_zero_one_traits() {
        assert!(BigRational::zero().is_zero());
        assert!(!BigRational::one().is_zero());
        assert!(BigRational::one().is_one());
    }

    #[test]
    fn test_from_conversions() {
        let from_i64: BigRational = 42i64.into();
        assert!(from_i64.is_integer());
        assert_eq!(from_i64.numer().to_i64(), Some(42));

        let from_smallint: BigRational = SmallInt::from(7i64).into();
        assert_eq!(from_smallint.numer().to_i64(), Some(7));
    }
}
