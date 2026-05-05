//! Algebraic trait hierarchy for type-safe polynomial and expression operations.
//!
//! Follows the standard algebraic hierarchy:
//!
//! ```text
//! Ring ⊃ GcdDomain ⊃ EuclideanDomain ⊃ Field
//! ```
//!
//! - [`Ring`] — addition, multiplication, zero, one, negation
//! - [`GcdDomain`] — GCD and unit detection
//! - [`EuclideanDomain`] — division with remainder and Euclidean function
//! - [`Field`] — multiplicative inverse (every non-zero element)

use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// A ring with identity: closed under addition, subtraction, multiplication,
/// with additive identity (zero) and multiplicative identity (one).
pub trait Ring:
    Clone
    + fmt::Debug
    + fmt::Display
    + PartialEq
    + Eq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + Sized
{
    /// Additive identity.
    fn zero() -> Self;

    /// Multiplicative identity.
    fn one() -> Self;

    /// Returns `true` if this element is the additive identity.
    fn is_zero(&self) -> bool;

    /// Returns `true` if this element is the multiplicative identity.
    fn is_one(&self) -> bool;
}

/// An integral domain where every pair of elements has a greatest common divisor.
pub trait GcdDomain: Ring {
    /// Greatest common divisor. Returns a non-negative associate.
    fn gcd(&self, other: &Self) -> Self;

    /// Returns `true` if this element is a unit (has a multiplicative inverse in the ring).
    fn is_unit(&self) -> bool;
}

/// A GCD domain with a Euclidean function supporting division with remainder.
///
/// For any `a` and non-zero `b`, there exist `q` and `r` such that
/// `a = b*q + r` and `euclidean_degree(r) < euclidean_degree(b)` (or `r = 0`).
pub trait EuclideanDomain: GcdDomain {
    /// Division with remainder: returns `(quotient, remainder)`.
    ///
    /// # Panics
    ///
    /// Panics if `other` is zero.
    fn div_rem(&self, other: &Self) -> (Self, Self);

    /// The Euclidean function (degree/norm). Zero maps to `None`.
    fn euclidean_degree(&self) -> Option<u64>;
}

/// A field: a commutative ring where every non-zero element has a multiplicative inverse.
pub trait Field: EuclideanDomain {
    /// Multiplicative inverse. Panics if `self` is zero.
    fn inv(&self) -> Self;
}

// ── SmallInt implements Ring, GcdDomain, EuclideanDomain ──────────────────────

use super::SmallInt;
use num::traits::Signed as NumSigned;

impl Ring for SmallInt {
    fn zero() -> Self {
        num::traits::Zero::zero()
    }

    fn one() -> Self {
        num::traits::One::one()
    }

    fn is_zero(&self) -> bool {
        num::traits::Zero::is_zero(self)
    }

    fn is_one(&self) -> bool {
        num::traits::One::is_one(self)
    }
}

impl GcdDomain for SmallInt {
    fn gcd(&self, other: &Self) -> Self {
        SmallInt::gcd(self, other)
    }

    fn is_unit(&self) -> bool {
        // In the integers, only ±1 are units
        match self {
            SmallInt::Inline(v) => *v == 1 || *v == -1,
            SmallInt::Heap(b) => {
                use num::traits::One;
                NumSigned::abs(b) == num::BigInt::one()
            }
        }
    }
}

impl EuclideanDomain for SmallInt {
    fn div_rem(&self, other: &Self) -> (Self, Self) {
        assert!(!other.is_zero(), "division by zero in EuclideanDomain");
        let q = self / other;
        let r = self.clone() - other.clone() * q.clone();
        (q, r)
    }

    fn euclidean_degree(&self) -> Option<u64> {
        if self.is_zero() {
            None
        } else {
            // For integers, the Euclidean function is |n|
            match self {
                SmallInt::Inline(v) => Some(v.unsigned_abs()),
                SmallInt::Heap(b) => {
                    use num::ToPrimitive;
                    NumSigned::abs(b).to_u64()
                }
            }
        }
    }
}

// ── BigRational implements Ring, GcdDomain, EuclideanDomain, Field ────────────

use super::BigRational;

impl Ring for BigRational {
    fn zero() -> Self {
        num::traits::Zero::zero()
    }

    fn one() -> Self {
        num::traits::One::one()
    }

    fn is_zero(&self) -> bool {
        num::traits::Zero::is_zero(self)
    }

    fn is_one(&self) -> bool {
        num::traits::One::is_one(self)
    }
}

impl GcdDomain for BigRational {
    fn gcd(&self, other: &Self) -> Self {
        // In a field, gcd is trivial: any non-zero element divides any other
        if self.is_zero() && other.is_zero() {
            BigRational::zero()
        } else {
            BigRational::one()
        }
    }

    fn is_unit(&self) -> bool {
        // Every non-zero rational is a unit in the field of rationals
        !self.is_zero()
    }
}

impl EuclideanDomain for BigRational {
    fn div_rem(&self, other: &Self) -> (Self, Self) {
        assert!(!other.is_zero(), "division by zero in EuclideanDomain");
        // In a field, division is exact: remainder is always zero
        (self / other, BigRational::zero())
    }

    fn euclidean_degree(&self) -> Option<u64> {
        if self.is_zero() {
            None
        } else {
            // In a field, every non-zero element has degree 0
            Some(0)
        }
    }
}

impl Field for BigRational {
    fn inv(&self) -> Self {
        self.recip()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smallint_ring() {
        let a = SmallInt::from(5i64);
        let b = SmallInt::from(3i64);
        assert_eq!(a.clone() + b.clone(), SmallInt::from(8i64));
        assert_eq!(a.clone() - b.clone(), SmallInt::from(2i64));
        assert_eq!(a.clone() * b.clone(), SmallInt::from(15i64));
        assert_eq!(-a, SmallInt::from(-5i64));
        assert!(SmallInt::zero().is_zero());
        assert!(SmallInt::one().is_one());
    }

    #[test]
    fn test_smallint_gcd_domain() {
        let a = SmallInt::from(12i64);
        let b = SmallInt::from(8i64);
        assert_eq!(GcdDomain::gcd(&a, &b), SmallInt::from(4i64));
        assert!(SmallInt::from(1i64).is_unit());
        assert!(SmallInt::from(-1i64).is_unit());
        assert!(!SmallInt::from(2i64).is_unit());
    }

    #[test]
    fn test_smallint_euclidean_domain() {
        let a = SmallInt::from(17i64);
        let b = SmallInt::from(5i64);
        let (q, r) = a.div_rem(&b);
        assert_eq!(q, SmallInt::from(3i64));
        assert_eq!(r, SmallInt::from(2i64));
        assert_eq!(SmallInt::from(0i64).euclidean_degree(), None);
        assert_eq!(SmallInt::from(7i64).euclidean_degree(), Some(7));
    }

    #[test]
    fn test_bigrational_ring() {
        let a = BigRational::from_i64(1, 3);
        let b = BigRational::from_i64(1, 6);
        let sum = a.clone() + b.clone();
        assert_eq!(sum, BigRational::from_i64(1, 2));
        assert!(BigRational::zero().is_zero());
        assert!(BigRational::one().is_one());
    }

    #[test]
    fn test_bigrational_field() {
        let r = BigRational::from_i64(3, 4);
        let inv = Field::inv(&r);
        assert_eq!(inv, BigRational::from_i64(4, 3));
        // r * inv = 1
        assert!((r * inv).is_one());
    }

    #[test]
    fn test_bigrational_euclidean() {
        let a = BigRational::from_i64(7, 3);
        let b = BigRational::from_i64(2, 5);
        let (q, r) = a.div_rem(&b);
        // In a field, division is exact
        assert!(r.is_zero());
        // q = (7/3) / (2/5) = 35/6
        assert_eq!(q, BigRational::from_i64(35, 6));
    }

    #[test]
    fn test_bigrational_gcd_trivial() {
        let a = BigRational::from_i64(3, 4);
        let b = BigRational::from_i64(5, 7);
        // In a field, gcd of non-zero elements is 1
        assert!(GcdDomain::gcd(&a, &b).is_one());
        assert!(a.is_unit());
        assert!(!BigRational::zero().is_unit());
    }
}
