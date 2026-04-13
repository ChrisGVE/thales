//! Dense univariate polynomial with generic coefficient ring.
//!
//! [`DensePolynomial<R>`] stores coefficients in a `Vec<R>` where
//! index = degree. Trailing zeros are trimmed automatically.

use super::ring::Ring;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// A dense univariate polynomial over a ring `R`.
///
/// Coefficients are stored as `Vec<R>` where `coeffs[i]` is the
/// coefficient of `x^i`. The zero polynomial has an empty `coeffs` vec.
///
/// # Invariant
///
/// Trailing zero coefficients are always trimmed, so the last element
/// (if any) is non-zero.
#[derive(Clone, Debug)]
pub struct DensePolynomial<R: Ring> {
    coeffs: Vec<R>,
}

impl<R: Ring> DensePolynomial<R> {
    /// The zero polynomial.
    pub fn zero() -> Self {
        DensePolynomial { coeffs: Vec::new() }
    }

    /// A constant polynomial.
    pub fn constant(c: R) -> Self {
        if c.is_zero() {
            Self::zero()
        } else {
            DensePolynomial { coeffs: vec![c] }
        }
    }

    /// The polynomial `x` (identity monomial).
    pub fn x() -> Self {
        DensePolynomial {
            coeffs: vec![R::zero(), R::one()],
        }
    }

    /// Create from a coefficient vector (index = degree). Trims trailing zeros.
    pub fn from_coeffs(coeffs: Vec<R>) -> Self {
        let mut p = DensePolynomial { coeffs };
        p.trim();
        p
    }

    /// Create a monomial `c * x^deg`.
    pub fn monomial(c: R, deg: usize) -> Self {
        if c.is_zero() {
            return Self::zero();
        }
        let mut coeffs = Vec::with_capacity(deg + 1);
        for _ in 0..deg {
            coeffs.push(R::zero());
        }
        coeffs.push(c);
        DensePolynomial { coeffs }
    }

    /// Returns `true` if this is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Degree of the polynomial. Returns `None` for the zero polynomial.
    pub fn degree(&self) -> Option<usize> {
        if self.coeffs.is_empty() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// Leading coefficient. Returns `None` for the zero polynomial.
    pub fn leading_coeff(&self) -> Option<&R> {
        self.coeffs.last()
    }

    /// Coefficient of `x^i`. Returns zero if `i` exceeds degree.
    pub fn coeff(&self, i: usize) -> R {
        if i < self.coeffs.len() {
            self.coeffs[i].clone()
        } else {
            R::zero()
        }
    }

    /// Slice of all coefficients (index = degree).
    pub fn coefficients(&self) -> &[R] {
        &self.coeffs
    }

    /// Number of stored coefficients.
    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    /// Whether storage is empty (zero polynomial).
    pub fn is_empty(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Evaluate the polynomial at a point using Horner's method.
    pub fn eval(&self, x: &R) -> R {
        if self.coeffs.is_empty() {
            return R::zero();
        }
        let mut result = self.coeffs.last().unwrap().clone();
        for c in self.coeffs.iter().rev().skip(1) {
            result = result * x.clone() + c.clone();
        }
        result
    }

    /// Scale all coefficients by a constant.
    pub fn scale(&self, c: &R) -> Self {
        if c.is_zero() {
            return Self::zero();
        }
        let coeffs: Vec<R> = self.coeffs.iter().map(|a| a.clone() * c.clone()).collect();
        DensePolynomial::from_coeffs(coeffs)
    }

    /// Remove trailing zero coefficients.
    fn trim(&mut self) {
        while self.coeffs.last().is_some_and(|c| c.is_zero()) {
            self.coeffs.pop();
        }
    }
}

// ── Equality ─────────────────────────────────────────────────────────────────

impl<R: Ring> PartialEq for DensePolynomial<R> {
    fn eq(&self, other: &Self) -> bool {
        self.coeffs == other.coeffs
    }
}

impl<R: Ring> Eq for DensePolynomial<R> {}

// ── Add ──────────────────────────────────────────────────────────────────────

impl<R: Ring> Add for DensePolynomial<R> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let max_len = self.coeffs.len().max(rhs.coeffs.len());
        let mut coeffs = Vec::with_capacity(max_len);
        for i in 0..max_len {
            let a = self.coeff(i);
            let b = rhs.coeff(i);
            coeffs.push(a + b);
        }
        DensePolynomial::from_coeffs(coeffs)
    }
}

impl<R: Ring> Add for &DensePolynomial<R> {
    type Output = DensePolynomial<R>;

    fn add(self, rhs: Self) -> DensePolynomial<R> {
        let max_len = self.coeffs.len().max(rhs.coeffs.len());
        let mut coeffs = Vec::with_capacity(max_len);
        for i in 0..max_len {
            coeffs.push(self.coeff(i) + rhs.coeff(i));
        }
        DensePolynomial::from_coeffs(coeffs)
    }
}

// ── Sub ──────────────────────────────────────────────────────────────────────

impl<R: Ring> Sub for DensePolynomial<R> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let max_len = self.coeffs.len().max(rhs.coeffs.len());
        let mut coeffs = Vec::with_capacity(max_len);
        for i in 0..max_len {
            coeffs.push(self.coeff(i) - rhs.coeff(i));
        }
        DensePolynomial::from_coeffs(coeffs)
    }
}

impl<R: Ring> Sub for &DensePolynomial<R> {
    type Output = DensePolynomial<R>;

    fn sub(self, rhs: Self) -> DensePolynomial<R> {
        let max_len = self.coeffs.len().max(rhs.coeffs.len());
        let mut coeffs = Vec::with_capacity(max_len);
        for i in 0..max_len {
            coeffs.push(self.coeff(i) - rhs.coeff(i));
        }
        DensePolynomial::from_coeffs(coeffs)
    }
}

// ── Neg ──────────────────────────────────────────────────────────────────────

impl<R: Ring> Neg for DensePolynomial<R> {
    type Output = Self;

    fn neg(self) -> Self {
        let coeffs: Vec<R> = self.coeffs.into_iter().map(|c| -c).collect();
        DensePolynomial { coeffs }
    }
}

impl<R: Ring> Neg for &DensePolynomial<R> {
    type Output = DensePolynomial<R>;

    fn neg(self) -> DensePolynomial<R> {
        let coeffs: Vec<R> = self.coeffs.iter().map(|c| -c.clone()).collect();
        DensePolynomial { coeffs }
    }
}

// ── Mul (schoolbook) ─────────────────────────────────────────────────────────

impl<R: Ring> Mul for DensePolynomial<R> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        if self.is_zero() || rhs.is_zero() {
            return DensePolynomial::zero();
        }
        let n = self.coeffs.len() + rhs.coeffs.len() - 1;
        let mut coeffs = vec![R::zero(); n];
        for (i, a) in self.coeffs.iter().enumerate() {
            for (j, b) in rhs.coeffs.iter().enumerate() {
                coeffs[i + j] = coeffs[i + j].clone() + a.clone() * b.clone();
            }
        }
        DensePolynomial::from_coeffs(coeffs)
    }
}

impl<R: Ring> Mul for &DensePolynomial<R> {
    type Output = DensePolynomial<R>;

    fn mul(self, rhs: Self) -> DensePolynomial<R> {
        if self.is_zero() || rhs.is_zero() {
            return DensePolynomial::zero();
        }
        let n = self.coeffs.len() + rhs.coeffs.len() - 1;
        let mut coeffs = vec![R::zero(); n];
        for (i, a) in self.coeffs.iter().enumerate() {
            for (j, b) in rhs.coeffs.iter().enumerate() {
                coeffs[i + j] = coeffs[i + j].clone() + a.clone() * b.clone();
            }
        }
        DensePolynomial::from_coeffs(coeffs)
    }
}

// ── Display ──────────────────────────────────────────────────────────────────

impl<R: Ring> fmt::Display for DensePolynomial<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        let mut first = true;
        for (i, c) in self.coeffs.iter().enumerate().rev() {
            if c.is_zero() {
                continue;
            }
            if !first {
                write!(f, " + ")?;
            }
            match i {
                0 => write!(f, "{c}")?,
                1 if c.is_one() => write!(f, "x")?,
                1 => write!(f, "{c}*x")?,
                _ if c.is_one() => write!(f, "x^{i}")?,
                _ => write!(f, "{c}*x^{i}")?,
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

    type Poly = DensePolynomial<SmallInt>;

    fn p(coeffs: &[i64]) -> Poly {
        Poly::from_coeffs(coeffs.iter().map(|&c| SmallInt::from(c)).collect())
    }

    #[test]
    fn test_zero() {
        let z = Poly::zero();
        assert!(z.is_zero());
        assert_eq!(z.degree(), None);
        assert_eq!(z.leading_coeff(), None);
    }

    #[test]
    fn test_constant() {
        let c = Poly::constant(SmallInt::from(5i64));
        assert_eq!(c.degree(), Some(0));
        assert_eq!(c.coeff(0), SmallInt::from(5i64));
    }

    #[test]
    fn test_constant_zero_is_zero() {
        let c = Poly::constant(SmallInt::from(0i64));
        assert!(c.is_zero());
    }

    #[test]
    fn test_x() {
        let x = Poly::x();
        assert_eq!(x.degree(), Some(1));
        assert_eq!(x.coeff(0), SmallInt::from(0i64));
        assert_eq!(x.coeff(1), SmallInt::from(1i64));
    }

    #[test]
    fn test_monomial() {
        let m = Poly::monomial(SmallInt::from(3i64), 4);
        assert_eq!(m.degree(), Some(4));
        assert_eq!(m.coeff(4), SmallInt::from(3i64));
        assert_eq!(m.coeff(0), SmallInt::from(0i64));
    }

    #[test]
    fn test_from_coeffs_trims() {
        let poly = p(&[1, 2, 0, 0]);
        assert_eq!(poly.degree(), Some(1));
    }

    #[test]
    fn test_add() {
        // (1 + 2x) + (3 + x) = 4 + 3x
        let a = p(&[1, 2]);
        let b = p(&[3, 1]);
        let sum = a + b;
        assert_eq!(sum.coeff(0), SmallInt::from(4i64));
        assert_eq!(sum.coeff(1), SmallInt::from(3i64));
    }

    #[test]
    fn test_add_different_degrees() {
        // (1 + x^2) + (2x) = 1 + 2x + x^2
        let a = p(&[1, 0, 1]);
        let b = p(&[0, 2]);
        let sum = a + b;
        assert_eq!(sum.degree(), Some(2));
        assert_eq!(sum.coeff(0), SmallInt::from(1i64));
        assert_eq!(sum.coeff(1), SmallInt::from(2i64));
        assert_eq!(sum.coeff(2), SmallInt::from(1i64));
    }

    #[test]
    fn test_add_cancellation() {
        // (1 + x) + (-1 - x) = 0
        let a = p(&[1, 1]);
        let b = p(&[-1, -1]);
        let sum = a + b;
        assert!(sum.is_zero());
    }

    #[test]
    fn test_sub() {
        // (3 + 2x) - (1 + x) = 2 + x
        let a = p(&[3, 2]);
        let b = p(&[1, 1]);
        let diff = a - b;
        assert_eq!(diff.coeff(0), SmallInt::from(2i64));
        assert_eq!(diff.coeff(1), SmallInt::from(1i64));
    }

    #[test]
    fn test_neg() {
        let a = p(&[1, -2, 3]);
        let neg = -a;
        assert_eq!(neg.coeff(0), SmallInt::from(-1i64));
        assert_eq!(neg.coeff(1), SmallInt::from(2i64));
        assert_eq!(neg.coeff(2), SmallInt::from(-3i64));
    }

    #[test]
    fn test_mul() {
        // (1 + x) * (1 - x) = 1 - x^2
        let a = p(&[1, 1]);
        let b = p(&[1, -1]);
        let prod = a * b;
        assert_eq!(prod.coeff(0), SmallInt::from(1i64));
        assert_eq!(prod.coeff(1), SmallInt::from(0i64));
        assert_eq!(prod.coeff(2), SmallInt::from(-1i64));
    }

    #[test]
    fn test_mul_by_zero() {
        let a = p(&[1, 2, 3]);
        let z = Poly::zero();
        assert!((a * z).is_zero());
    }

    #[test]
    fn test_mul_degree() {
        // deg(f*g) = deg(f) + deg(g)
        let a = p(&[1, 1, 1]); // deg 2
        let b = p(&[1, 1]); // deg 1
        let prod = a * b;
        assert_eq!(prod.degree(), Some(3));
    }

    #[test]
    fn test_eval() {
        // 1 + 2x + 3x^2, evaluate at x=2: 1 + 4 + 12 = 17
        let poly = p(&[1, 2, 3]);
        let result = poly.eval(&SmallInt::from(2i64));
        assert_eq!(result, SmallInt::from(17i64));
    }

    #[test]
    fn test_eval_zero_poly() {
        let z = Poly::zero();
        assert_eq!(z.eval(&SmallInt::from(5i64)), SmallInt::from(0i64));
    }

    #[test]
    fn test_eval_constant() {
        let c = Poly::constant(SmallInt::from(7i64));
        assert_eq!(c.eval(&SmallInt::from(100i64)), SmallInt::from(7i64));
    }

    #[test]
    fn test_scale() {
        let poly = p(&[1, 2, 3]);
        let scaled = poly.scale(&SmallInt::from(2i64));
        assert_eq!(scaled.coeff(0), SmallInt::from(2i64));
        assert_eq!(scaled.coeff(1), SmallInt::from(4i64));
        assert_eq!(scaled.coeff(2), SmallInt::from(6i64));
    }

    #[test]
    fn test_scale_by_zero() {
        let poly = p(&[1, 2, 3]);
        assert!(poly.scale(&SmallInt::from(0i64)).is_zero());
    }

    #[test]
    fn test_equality() {
        let a = p(&[1, 2, 3]);
        let b = p(&[1, 2, 3]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_inequality() {
        let a = p(&[1, 2]);
        let b = p(&[1, 3]);
        assert_ne!(a, b);
    }

    #[test]
    fn test_display_zero() {
        assert_eq!(Poly::zero().to_string(), "0");
    }

    #[test]
    fn test_display_constant() {
        assert_eq!(Poly::constant(SmallInt::from(5i64)).to_string(), "5");
    }

    #[test]
    fn test_display_poly() {
        // 1 + 2x + x^2 displayed as "x^2 + 2*x + 1"
        let poly = p(&[1, 2, 1]);
        let s = poly.to_string();
        assert!(s.contains("x^2"));
        assert!(s.contains("2*x"));
        assert!(s.contains("1"));
    }

    #[test]
    fn test_ref_arithmetic() {
        let a = p(&[1, 1]);
        let b = p(&[2, 3]);
        let sum = &a + &b;
        assert_eq!(sum.coeff(0), SmallInt::from(3i64));
        let diff = &a - &b;
        assert_eq!(diff.coeff(0), SmallInt::from(-1i64));
        let prod = &a * &b;
        assert_eq!(prod.degree(), Some(2));
    }

    // ── BigRational polynomial tests ─────────────────────────────────────────

    #[test]
    fn test_rational_polynomial() {
        use crate::numeric::BigRational;
        type RPoly = DensePolynomial<BigRational>;

        let half = BigRational::from_i64(1, 2);
        let third = BigRational::from_i64(1, 3);

        let a = RPoly::from_coeffs(vec![half.clone(), BigRational::one()]);
        let b = RPoly::from_coeffs(vec![third.clone(), BigRational::one()]);
        let sum = &a + &b;
        // 1/2 + 1/3 = 5/6
        assert_eq!(sum.coeff(0), BigRational::from_i64(5, 6));
        assert_eq!(sum.coeff(1), BigRational::from(2i64));
    }
}
