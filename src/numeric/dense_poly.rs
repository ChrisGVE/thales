//! Dense univariate polynomial with generic coefficient ring.
//!
//! [`DensePolynomial<R>`] stores coefficients in a `Vec<R>` where
//! index = degree. Trailing zeros are trimmed automatically.

use super::ring::{Field, Ring};
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

// ── Euclidean division (requires Field) ──────────────────────────────────────

impl<R: Field> DensePolynomial<R> {
    /// Euclidean division: returns `(quotient, remainder)` such that
    /// `self = quotient * divisor + remainder` and
    /// `deg(remainder) < deg(divisor)`.
    ///
    /// # Panics
    ///
    /// Panics if `divisor` is zero.
    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        assert!(!divisor.is_zero(), "polynomial division by zero");

        if self.is_zero() {
            return (Self::zero(), Self::zero());
        }

        let d_deg = divisor.degree().unwrap();
        let mut remainder = self.clone();

        match remainder.degree() {
            None => return (Self::zero(), Self::zero()),
            Some(r_deg) if r_deg < d_deg => {
                return (Self::zero(), remainder);
            }
            _ => {}
        }

        let r_deg = remainder.degree().unwrap();
        let lc_inv = divisor.leading_coeff().unwrap().inv();

        let mut q_coeffs = vec![R::zero(); r_deg - d_deg + 1];

        while let Some(rem_deg) = remainder.degree() {
            if rem_deg < d_deg {
                break;
            }
            let shift = rem_deg - d_deg;
            let factor = remainder.leading_coeff().unwrap().clone() * lc_inv.clone();

            q_coeffs[shift] = factor.clone();

            // remainder -= factor * x^shift * divisor
            for (i, dc) in divisor.coeffs.iter().enumerate() {
                let idx = i + shift;
                remainder.coeffs[idx] = remainder.coeffs[idx].clone() - factor.clone() * dc.clone();
            }
            remainder.trim();
        }

        (DensePolynomial::from_coeffs(q_coeffs), remainder)
    }

    /// Make the polynomial monic (leading coefficient = 1).
    /// Returns zero polynomial unchanged.
    pub fn make_monic(&self) -> Self {
        match self.leading_coeff() {
            None => Self::zero(),
            Some(lc) => {
                let inv = lc.inv();
                self.scale(&inv)
            }
        }
    }

    /// Euclidean GCD of two polynomials, normalized to monic.
    ///
    /// Uses the Euclidean algorithm (repeated `div_rem`).
    /// Returns the zero polynomial if both inputs are zero.
    pub fn gcd(&self, other: &Self) -> Self {
        let mut a = self.clone();
        let mut b = other.clone();
        while !b.is_zero() {
            let (_, r) = a.div_rem(&b);
            a = b;
            b = r;
        }
        a.make_monic()
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

    // ── Euclidean division tests ─────────────────────────────────────────────

    mod div_rem_tests {
        use super::*;
        use crate::numeric::BigRational;
        type RPoly = DensePolynomial<BigRational>;

        fn rp(coeffs: &[(i64, i64)]) -> RPoly {
            RPoly::from_coeffs(
                coeffs
                    .iter()
                    .map(|&(n, d)| BigRational::from_i64(n, d))
                    .collect(),
            )
        }

        fn ri(coeffs: &[i64]) -> RPoly {
            RPoly::from_coeffs(coeffs.iter().map(|&c| BigRational::from(c)).collect())
        }

        #[test]
        fn test_div_rem_exact() {
            // (x^3 + 1) / (x + 1) = x^2 - x + 1, remainder 0
            let f = ri(&[1, 0, 0, 1]);
            let g = ri(&[1, 1]);
            let (q, r) = f.div_rem(&g);
            assert_eq!(q, ri(&[1, -1, 1]));
            assert!(r.is_zero());
        }

        #[test]
        fn test_div_rem_with_remainder() {
            // (x^2 + 1) / (x + 1) = x - 1, remainder 2
            let f = ri(&[1, 0, 1]);
            let g = ri(&[1, 1]);
            let (q, r) = f.div_rem(&g);
            assert_eq!(q, ri(&[-1, 1]));
            assert_eq!(r, ri(&[2]));
        }

        #[test]
        fn test_div_rem_identity() {
            // f = q * g + r
            let f = ri(&[3, 2, 5, 1]);
            let g = ri(&[1, 1]);
            let (q, r) = f.div_rem(&g);
            let reconstructed = &(&q * &g) + &r;
            assert_eq!(reconstructed, f);
        }

        #[test]
        fn test_div_rem_degree_smaller() {
            // deg(f) < deg(g) → q = 0, r = f
            let f = ri(&[1, 1]);
            let g = ri(&[1, 0, 1]);
            let (q, r) = f.div_rem(&g);
            assert!(q.is_zero());
            assert_eq!(r, f);
        }

        #[test]
        fn test_div_rem_zero_dividend() {
            let f = RPoly::zero();
            let g = ri(&[1, 1]);
            let (q, r) = f.div_rem(&g);
            assert!(q.is_zero());
            assert!(r.is_zero());
        }

        #[test]
        #[should_panic(expected = "polynomial division by zero")]
        fn test_div_rem_zero_divisor() {
            let f = ri(&[1, 1]);
            let g = RPoly::zero();
            let _ = f.div_rem(&g);
        }

        #[test]
        fn test_div_rem_rational_coefficients() {
            // (x^2 - 1/4) / (x - 1/2) = x + 1/2, remainder 0
            let f = rp(&[(-1, 4), (0, 1), (1, 1)]);
            let g = rp(&[(-1, 2), (1, 1)]);
            let (q, r) = f.div_rem(&g);
            assert_eq!(q, rp(&[(1, 2), (1, 1)]));
            assert!(r.is_zero());
        }

        #[test]
        fn test_make_monic() {
            let f = ri(&[2, 4, 2]);
            let m = f.make_monic();
            assert_eq!(m.leading_coeff(), Some(&BigRational::one()));
            assert_eq!(m, ri(&[1, 2, 1]));
        }

        #[test]
        fn test_div_rem_random_identity() {
            // For random-ish polynomials, verify f = q*g + r
            let f = ri(&[1, -3, 0, 2, 5]);
            let g = ri(&[2, 0, 1]);
            let (q, r) = f.div_rem(&g);
            let reconstructed = &(&q * &g) + &r;
            assert_eq!(reconstructed, f);
            assert!(r.degree().unwrap_or(0) < g.degree().unwrap());
        }
    }

    // ── GCD tests ────────────────────────────────────────────────────────────

    mod gcd_tests {
        use super::*;
        use crate::numeric::BigRational;
        type RPoly = DensePolynomial<BigRational>;

        fn ri(coeffs: &[i64]) -> RPoly {
            RPoly::from_coeffs(coeffs.iter().map(|&c| BigRational::from(c)).collect())
        }

        #[test]
        fn test_gcd_basic() {
            // gcd(x^2-1, x-1) = x-1 (monic)
            let f = ri(&[-1, 0, 1]); // x^2 - 1
            let g = ri(&[-1, 1]); // x - 1
            let d = f.gcd(&g);
            assert_eq!(d, ri(&[-1, 1]));
        }

        #[test]
        fn test_gcd_coprime() {
            // gcd(x^2+1, x+1) = 1
            let f = ri(&[1, 0, 1]);
            let g = ri(&[1, 1]);
            let d = f.gcd(&g);
            assert_eq!(d.degree(), Some(0));
            assert!(d.coeff(0).is_one());
        }

        #[test]
        fn test_gcd_with_zero() {
            let f = ri(&[-1, 0, 1]);
            let z = RPoly::zero();
            // gcd(f, 0) = monic(f)
            let d = f.gcd(&z);
            assert_eq!(d, f.make_monic());
            // gcd(0, f) = monic(f)
            let d2 = z.gcd(&f);
            assert_eq!(d2, f.make_monic());
        }

        #[test]
        fn test_gcd_both_zero() {
            let z = RPoly::zero();
            let d = z.gcd(&z);
            assert!(d.is_zero());
        }

        #[test]
        fn test_gcd_x3_minus_x_and_x2_minus_1() {
            // gcd(x^3-x, x^2-1) = x^2-1 (both divisible by (x-1)(x+1))
            let f = ri(&[0, -1, 0, 1]); // x^3 - x = x(x^2-1)
            let g = ri(&[-1, 0, 1]); // x^2 - 1
            let d = f.gcd(&g);
            assert_eq!(d, ri(&[-1, 0, 1]));
        }

        #[test]
        fn test_gcd_result_is_monic() {
            // gcd of 2x+2 and 3x+3 should be x+1 (monic)
            let f = ri(&[2, 2]);
            let g = ri(&[3, 3]);
            let d = f.gcd(&g);
            assert_eq!(d, ri(&[1, 1]));
        }

        #[test]
        fn test_gcd_divides_both() {
            let f = ri(&[-1, 0, 0, 1]); // x^3 - 1
            let g = ri(&[-1, 0, 1]); // x^2 - 1
            let d = f.gcd(&g);
            let (_, r1) = f.div_rem(&d);
            let (_, r2) = g.div_rem(&d);
            assert!(r1.is_zero());
            assert!(r2.is_zero());
        }
    }
}
