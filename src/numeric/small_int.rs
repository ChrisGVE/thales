//! A tagged integer type that stores small values inline as `i64` and
//! promotes to heap-allocated `BigInt` on overflow.

use num::bigint::BigInt;
use num::traits::{One, Signed, Zero};
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

/// An integer that is either an inline `i64` or a heap-allocated [`BigInt`].
///
/// Arithmetic that stays within `i64` range uses fast inline operations.
/// Overflow automatically promotes to `BigInt`, so callers never need to
/// worry about silent wrapping or panics.
#[derive(Clone, Debug)]
pub enum SmallInt {
    /// Value fits in 64 bits.
    Inline(i64),
    /// Value exceeds i64 range.
    Heap(BigInt),
}

// ── Construction ──────────────────────────────────────────────────────────────

impl SmallInt {
    /// Create from an `i64` value (always inline).
    #[inline]
    pub fn from_i64(v: i64) -> Self {
        SmallInt::Inline(v)
    }

    /// Try to demote a `BigInt` back to inline if it fits in `i64`.
    fn normalize(big: BigInt) -> Self {
        match i64::try_from(&big) {
            Ok(v) => SmallInt::Inline(v),
            Err(_) => SmallInt::Heap(big),
        }
    }

    /// Convert to `BigInt` (cloning if inline).
    pub fn to_bigint(&self) -> BigInt {
        match self {
            SmallInt::Inline(v) => BigInt::from(*v),
            SmallInt::Heap(b) => b.clone(),
        }
    }

    /// Try to extract as `i64`. Returns `None` if value exceeds `i64` range.
    pub fn to_i64(&self) -> Option<i64> {
        match self {
            SmallInt::Inline(v) => Some(*v),
            SmallInt::Heap(b) => i64::try_from(b).ok(),
        }
    }

    /// Returns `true` if stored inline.
    #[inline]
    pub fn is_inline(&self) -> bool {
        matches!(self, SmallInt::Inline(_))
    }
}

// ── Zero / One ────────────────────────────────────────────────────────────────

impl Zero for SmallInt {
    fn zero() -> Self {
        SmallInt::Inline(0)
    }

    fn is_zero(&self) -> bool {
        match self {
            SmallInt::Inline(v) => *v == 0,
            SmallInt::Heap(b) => b.is_zero(),
        }
    }
}

impl One for SmallInt {
    fn one() -> Self {
        SmallInt::Inline(1)
    }

    fn is_one(&self) -> bool
    where
        Self: PartialEq,
    {
        match self {
            SmallInt::Inline(v) => *v == 1,
            SmallInt::Heap(b) => b.is_one(),
        }
    }
}

// ── Display ───────────────────────────────────────────────────────────────────

impl fmt::Display for SmallInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmallInt::Inline(v) => write!(f, "{v}"),
            SmallInt::Heap(b) => write!(f, "{b}"),
        }
    }
}

// ── Equality and Ordering ─────────────────────────────────────────────────────

impl PartialEq for SmallInt {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SmallInt::Inline(a), SmallInt::Inline(b)) => a == b,
            (SmallInt::Heap(a), SmallInt::Heap(b)) => a == b,
            // If one is inline and the other heap, they can't be equal
            // (heap means value doesn't fit in i64).
            _ => false,
        }
    }
}

impl Eq for SmallInt {}

impl PartialOrd for SmallInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SmallInt {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (SmallInt::Inline(a), SmallInt::Inline(b)) => a.cmp(b),
            _ => self.to_bigint().cmp(&other.to_bigint()),
        }
    }
}

impl Hash for SmallInt {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Ensure consistent hashing: always hash the i64 value when possible.
        match self {
            SmallInt::Inline(v) => v.hash(state),
            SmallInt::Heap(b) => {
                // If it fits in i64, hash as i64 for consistency
                if let Ok(v) = i64::try_from(b) {
                    v.hash(state);
                } else {
                    b.hash(state);
                }
            }
        }
    }
}

// ── Arithmetic: Add ───────────────────────────────────────────────────────────

impl Add for SmallInt {
    type Output = SmallInt;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (SmallInt::Inline(a), SmallInt::Inline(b)) => match a.checked_add(b) {
                Some(v) => SmallInt::Inline(v),
                None => SmallInt::normalize(BigInt::from(a) + BigInt::from(b)),
            },
            (a, b) => SmallInt::normalize(a.to_bigint() + b.to_bigint()),
        }
    }
}

impl Add for &SmallInt {
    type Output = SmallInt;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (SmallInt::Inline(a), SmallInt::Inline(b)) => match a.checked_add(*b) {
                Some(v) => SmallInt::Inline(v),
                None => SmallInt::normalize(BigInt::from(*a) + BigInt::from(*b)),
            },
            (a, b) => SmallInt::normalize(a.to_bigint() + b.to_bigint()),
        }
    }
}

// ── Arithmetic: Sub ───────────────────────────────────────────────────────────

impl Sub for SmallInt {
    type Output = SmallInt;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (SmallInt::Inline(a), SmallInt::Inline(b)) => match a.checked_sub(b) {
                Some(v) => SmallInt::Inline(v),
                None => SmallInt::normalize(BigInt::from(a) - BigInt::from(b)),
            },
            (a, b) => SmallInt::normalize(a.to_bigint() - b.to_bigint()),
        }
    }
}

impl Sub for &SmallInt {
    type Output = SmallInt;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (SmallInt::Inline(a), SmallInt::Inline(b)) => match a.checked_sub(*b) {
                Some(v) => SmallInt::Inline(v),
                None => SmallInt::normalize(BigInt::from(*a) - BigInt::from(*b)),
            },
            (a, b) => SmallInt::normalize(a.to_bigint() - b.to_bigint()),
        }
    }
}

// ── Arithmetic: Mul ───────────────────────────────────────────────────────────

impl Mul for SmallInt {
    type Output = SmallInt;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (SmallInt::Inline(a), SmallInt::Inline(b)) => match a.checked_mul(b) {
                Some(v) => SmallInt::Inline(v),
                None => SmallInt::normalize(BigInt::from(a) * BigInt::from(b)),
            },
            (a, b) => SmallInt::normalize(a.to_bigint() * b.to_bigint()),
        }
    }
}

impl Mul for &SmallInt {
    type Output = SmallInt;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (SmallInt::Inline(a), SmallInt::Inline(b)) => match a.checked_mul(*b) {
                Some(v) => SmallInt::Inline(v),
                None => SmallInt::normalize(BigInt::from(*a) * BigInt::from(*b)),
            },
            (a, b) => SmallInt::normalize(a.to_bigint() * b.to_bigint()),
        }
    }
}

// ── Arithmetic: Div (truncating) ──────────────────────────────────────────────

impl Div for SmallInt {
    type Output = SmallInt;

    /// Truncating integer division. Panics on division by zero.
    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (SmallInt::Inline(a), SmallInt::Inline(b)) => {
                assert!(b != 0, "division by zero");
                // i64::MIN / -1 overflows
                match a.checked_div(b) {
                    Some(v) => SmallInt::Inline(v),
                    None => SmallInt::normalize(BigInt::from(a) / BigInt::from(b)),
                }
            }
            (a, b) => {
                let b_big = b.to_bigint();
                assert!(!b_big.is_zero(), "division by zero");
                SmallInt::normalize(a.to_bigint() / b_big)
            }
        }
    }
}

impl Div for &SmallInt {
    type Output = SmallInt;

    fn div(self, rhs: Self) -> Self::Output {
        self.clone() / rhs.clone()
    }
}

// ── Arithmetic: Rem ───────────────────────────────────────────────────────────

impl Rem for SmallInt {
    type Output = SmallInt;

    fn rem(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (SmallInt::Inline(a), SmallInt::Inline(b)) => {
                assert!(b != 0, "division by zero in remainder");
                match a.checked_rem(b) {
                    Some(v) => SmallInt::Inline(v),
                    None => SmallInt::normalize(BigInt::from(a) % BigInt::from(b)),
                }
            }
            (a, b) => {
                let b_big = b.to_bigint();
                assert!(!b_big.is_zero(), "division by zero in remainder");
                SmallInt::normalize(a.to_bigint() % b_big)
            }
        }
    }
}

impl Rem for &SmallInt {
    type Output = SmallInt;

    fn rem(self, rhs: Self) -> Self::Output {
        self.clone() % rhs.clone()
    }
}

// ── Arithmetic: Neg ───────────────────────────────────────────────────────────

impl Neg for SmallInt {
    type Output = SmallInt;

    fn neg(self) -> Self::Output {
        match self {
            SmallInt::Inline(v) => match v.checked_neg() {
                Some(n) => SmallInt::Inline(n),
                None => SmallInt::Heap(-BigInt::from(v)),
            },
            SmallInt::Heap(b) => SmallInt::normalize(-b),
        }
    }
}

impl Neg for &SmallInt {
    type Output = SmallInt;

    fn neg(self) -> Self::Output {
        match self {
            SmallInt::Inline(v) => match v.checked_neg() {
                Some(n) => SmallInt::Inline(n),
                None => SmallInt::Heap(-BigInt::from(*v)),
            },
            SmallInt::Heap(b) => SmallInt::normalize(-b.clone()),
        }
    }
}

// ── Num ───────────────────────────────────────────────────────────────────────

impl num::Num for SmallInt {
    type FromStrRadixErr = num::bigint::ParseBigIntError;

    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        // Try i64 first
        if let Ok(v) = i64::from_str_radix(s, radix) {
            return Ok(SmallInt::Inline(v));
        }
        BigInt::from_str_radix(s, radix).map(SmallInt::normalize)
    }
}

// ── Signed ────────────────────────────────────────────────────────────────────

impl Signed for SmallInt {
    fn abs(&self) -> Self {
        match self {
            SmallInt::Inline(v) => match v.checked_abs() {
                Some(a) => SmallInt::Inline(a),
                None => SmallInt::Heap(BigInt::from(*v).abs()),
            },
            SmallInt::Heap(b) => SmallInt::normalize(b.abs()),
        }
    }

    fn abs_sub(&self, other: &Self) -> Self {
        let diff = self - other;
        if diff >= SmallInt::zero() {
            diff
        } else {
            SmallInt::zero()
        }
    }

    fn signum(&self) -> Self {
        match self {
            SmallInt::Inline(v) => SmallInt::Inline(v.signum()),
            SmallInt::Heap(b) => SmallInt::Inline(b.sign() as i64),
        }
    }

    fn is_positive(&self) -> bool {
        match self {
            SmallInt::Inline(v) => *v > 0,
            SmallInt::Heap(b) => b.is_positive(),
        }
    }

    fn is_negative(&self) -> bool {
        match self {
            SmallInt::Inline(v) => *v < 0,
            SmallInt::Heap(b) => b.is_negative(),
        }
    }
}

// ── Conversions ───────────────────────────────────────────────────────────────

impl From<i64> for SmallInt {
    fn from(v: i64) -> Self {
        SmallInt::Inline(v)
    }
}

impl From<i32> for SmallInt {
    fn from(v: i32) -> Self {
        SmallInt::Inline(v as i64)
    }
}

impl From<BigInt> for SmallInt {
    fn from(v: BigInt) -> Self {
        SmallInt::normalize(v)
    }
}

// ── GCD ───────────────────────────────────────────────────────────────────────

impl SmallInt {
    /// Compute the greatest common divisor. Result is always non-negative.
    pub fn gcd(&self, other: &Self) -> Self {
        match (self, other) {
            (SmallInt::Inline(a), SmallInt::Inline(b)) => SmallInt::Inline(gcd_i64(*a, *b)),
            _ => {
                let a = self.to_bigint();
                let b = other.to_bigint();
                SmallInt::normalize(num::integer::Integer::gcd(&a, &b))
            }
        }
    }
}

impl SmallInt {
    /// Raise to a non-negative integer power via repeated squaring.
    pub fn pow(&self, mut exp: u32) -> Self {
        if exp == 0 {
            return SmallInt::Inline(1);
        }
        let mut base = self.clone();
        let mut result = SmallInt::Inline(1);
        while exp > 1 {
            if exp % 2 == 1 {
                result = &result * &base;
            }
            base = &base * &base;
            exp /= 2;
        }
        &result * &base
    }
}

/// GCD for i64 using the Euclidean algorithm. Returns non-negative result.
fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    a = a.wrapping_abs();
    b = b.wrapping_abs();
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inline_arithmetic() {
        let a = SmallInt::from(10i64);
        let b = SmallInt::from(3i64);

        assert_eq!((&a + &b).to_i64(), Some(13));
        assert_eq!((&a - &b).to_i64(), Some(7));
        assert_eq!((&a * &b).to_i64(), Some(30));
        assert_eq!((&a / &b).to_i64(), Some(3));
        assert_eq!((a % b).to_i64(), Some(1));
    }

    #[test]
    fn test_overflow_promotes_to_bigint() {
        let a = SmallInt::from(i64::MAX);
        let b = SmallInt::from(1i64);
        let result = a + b;
        assert!(!result.is_inline());
        assert_eq!(result.to_bigint(), BigInt::from(i64::MAX) + BigInt::from(1));
    }

    #[test]
    fn test_mul_overflow_promotes() {
        let a = SmallInt::from(i64::MAX);
        let b = SmallInt::from(2i64);
        let result = a * b;
        assert!(!result.is_inline());
        assert_eq!(result.to_bigint(), BigInt::from(i64::MAX) * BigInt::from(2));
    }

    #[test]
    fn test_neg_i64_min_promotes() {
        let a = SmallInt::from(i64::MIN);
        let result = -a;
        assert!(!result.is_inline());
        assert_eq!(result.to_bigint(), -BigInt::from(i64::MIN));
    }

    #[test]
    fn test_bigint_demotes_when_fits() {
        let big = SmallInt::from(BigInt::from(42));
        assert!(big.is_inline());
        assert_eq!(big.to_i64(), Some(42));
    }

    #[test]
    fn test_equality() {
        assert_eq!(SmallInt::from(5i64), SmallInt::from(5i64));
        assert_ne!(SmallInt::from(5i64), SmallInt::from(6i64));
    }

    #[test]
    fn test_ordering() {
        assert!(SmallInt::from(3i64) < SmallInt::from(5i64));
        assert!(SmallInt::from(-1i64) < SmallInt::from(0i64));
    }

    #[test]
    fn test_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;

        let a = SmallInt::from(42i64);
        let b = SmallInt::from(BigInt::from(42));

        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn test_zero_one() {
        assert!(SmallInt::zero().is_zero());
        assert!(!SmallInt::one().is_zero());
        assert!(SmallInt::one().is_one());
        assert!(!SmallInt::zero().is_one());
    }

    #[test]
    fn test_display() {
        assert_eq!(SmallInt::from(42i64).to_string(), "42");
        assert_eq!(SmallInt::from(-7i64).to_string(), "-7");
    }

    #[test]
    fn test_gcd() {
        assert_eq!(
            SmallInt::from(12i64).gcd(&SmallInt::from(8i64)).to_i64(),
            Some(4)
        );
        assert_eq!(
            SmallInt::from(0i64).gcd(&SmallInt::from(5i64)).to_i64(),
            Some(5)
        );
        assert_eq!(
            SmallInt::from(7i64).gcd(&SmallInt::from(0i64)).to_i64(),
            Some(7)
        );
        assert_eq!(
            SmallInt::from(-12i64).gcd(&SmallInt::from(8i64)).to_i64(),
            Some(4)
        );
    }

    #[test]
    fn test_signed_traits() {
        let neg = SmallInt::from(-5i64);
        assert!(neg.is_negative());
        assert!(!neg.is_positive());
        assert_eq!(neg.abs().to_i64(), Some(5));
        assert_eq!(neg.signum().to_i64(), Some(-1));
    }

    #[test]
    fn test_mixed_inline_heap_arithmetic() {
        let small = SmallInt::from(10i64);
        let big = SmallInt::from(BigInt::from(i64::MAX) + BigInt::from(1));
        let result = small + big.clone();
        assert!(!result.is_inline());
        assert_eq!(
            result.to_bigint(),
            BigInt::from(10) + BigInt::from(i64::MAX) + BigInt::from(1)
        );
    }

    #[test]
    fn test_sub_underflow_promotes() {
        let a = SmallInt::from(i64::MIN);
        let b = SmallInt::from(1i64);
        let result = a - b;
        assert!(!result.is_inline());
        assert_eq!(result.to_bigint(), BigInt::from(i64::MIN) - BigInt::from(1));
    }
}
