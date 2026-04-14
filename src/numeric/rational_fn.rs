//! Rational function type: numerator/denominator polynomial fraction.
//!
//! [`RationalFunction<R>`] represents `p(x)/q(x)` where `p` and `q` are
//! dense univariate polynomials over a field `R`. Automatically reduces
//! to lowest terms via GCD on construction.

use super::dense_poly::DensePolynomial;
use super::ring::Field;
use std::fmt;

/// A rational function `numerator / denominator` over a field `R`.
///
/// # Invariants
///
/// - Denominator is never zero
/// - Fraction is always in lowest terms (GCD-reduced)
/// - Denominator is monic (leading coefficient = 1)
#[derive(Clone, Debug)]
pub struct RationalFunction<R: Field> {
    num: DensePolynomial<R>,
    den: DensePolynomial<R>,
}

impl<R: Field> RationalFunction<R> {
    /// Create a new rational function, reducing to lowest terms.
    ///
    /// # Panics
    ///
    /// Panics if `den` is zero.
    pub fn new(num: DensePolynomial<R>, den: DensePolynomial<R>) -> Self {
        assert!(!den.is_zero(), "rational function with zero denominator");
        let mut rf = RationalFunction { num, den };
        rf.reduce();
        rf
    }

    /// Create from a polynomial (denominator = 1).
    pub fn from_poly(p: DensePolynomial<R>) -> Self {
        RationalFunction {
            num: p,
            den: DensePolynomial::constant(R::one()),
        }
    }

    /// The zero rational function.
    pub fn zero() -> Self {
        RationalFunction {
            num: DensePolynomial::zero(),
            den: DensePolynomial::constant(R::one()),
        }
    }

    /// The constant rational function `c/1`.
    pub fn constant(c: R) -> Self {
        RationalFunction {
            num: DensePolynomial::constant(c),
            den: DensePolynomial::constant(R::one()),
        }
    }

    /// The identity rational function `x/1`.
    pub fn x() -> Self {
        Self::from_poly(DensePolynomial::x())
    }

    /// Reference to numerator polynomial.
    pub fn numerator(&self) -> &DensePolynomial<R> {
        &self.num
    }

    /// Reference to denominator polynomial.
    pub fn denominator(&self) -> &DensePolynomial<R> {
        &self.den
    }

    /// Consume and return (numerator, denominator).
    pub fn into_parts(self) -> (DensePolynomial<R>, DensePolynomial<R>) {
        (self.num, self.den)
    }

    /// Returns `true` if the rational function is zero.
    pub fn is_zero(&self) -> bool {
        self.num.is_zero()
    }

    /// Returns `true` if this is a polynomial (denominator is constant).
    pub fn is_polynomial(&self) -> bool {
        self.den.degree().unwrap_or(0) == 0
    }

    /// Evaluate at a point. Returns `None` if denominator is zero at that point.
    pub fn eval(&self, x: &R) -> Option<R> {
        let d = self.den.eval(x);
        if d.is_zero() {
            None
        } else {
            Some(self.num.eval(x) * d.inv())
        }
    }

    /// Reciprocal: `den/num`. Panics if numerator is zero.
    pub fn recip(&self) -> Self {
        assert!(!self.num.is_zero(), "reciprocal of zero rational function");
        Self::new(self.den.clone(), self.num.clone())
    }

    /// Negate the rational function.
    pub fn neg(&self) -> Self {
        RationalFunction {
            num: -&self.num,
            den: self.den.clone(),
        }
    }

    /// Add two rational functions: `a/b + c/d = (ad + bc) / bd`, reduced.
    pub fn add(&self, other: &Self) -> Self {
        let num = &(&self.num * &other.den) + &(&other.num * &self.den);
        let den = &self.den * &other.den;
        Self::new(num, den)
    }

    /// Subtract: `self - other`.
    pub fn sub(&self, other: &Self) -> Self {
        let neg_other_num = -&other.num;
        let num = &(&self.num * &other.den) + &(&neg_other_num * &self.den);
        let den = &self.den * &other.den;
        Self::new(num, den)
    }

    /// Multiply two rational functions, reducing cross-terms.
    pub fn mul(&self, other: &Self) -> Self {
        // Cross-reduce before multiplying to keep coefficients small:
        // (a/b) * (c/d) — reduce gcd(a,d) and gcd(c,b) first
        let g1 = self.num.gcd(&other.den);
        let g2 = other.num.gcd(&self.den);

        let a = self.num.div_rem(&g1).0;
        let d = other.den.div_rem(&g1).0;
        let c = other.num.div_rem(&g2).0;
        let b = self.den.div_rem(&g2).0;

        let num = &a * &c;
        let den = &b * &d;

        Self::new(num, den)
    }

    /// Divide: `self / other`. Panics if `other` is zero.
    pub fn div(&self, other: &Self) -> Self {
        assert!(!other.is_zero(), "division by zero rational function");
        self.mul(&other.recip())
    }

    /// Formal derivative using quotient rule: `(n'd - nd') / d^2`.
    pub fn derivative(&self) -> Self {
        let np = self.num.derivative();
        let dp = self.den.derivative();
        let num = &(&np * &self.den) - &(&self.num * &dp);
        let den = &self.den * &self.den;
        Self::new(num, den)
    }

    /// Reduce fraction to lowest terms and make denominator monic.
    fn reduce(&mut self) {
        if self.num.is_zero() {
            self.den = DensePolynomial::constant(R::one());
            return;
        }

        let g = self.num.gcd(&self.den);
        if g.degree().unwrap_or(0) > 0 || !g.leading_coeff().map_or(true, |c| c.is_one()) {
            self.num = self.num.div_rem(&g).0;
            self.den = self.den.div_rem(&g).0;
        }

        // Make denominator monic
        if let Some(lc) = self.den.leading_coeff() {
            if !lc.is_one() {
                let inv = lc.inv();
                self.num = self.num.scale(&inv);
                self.den = self.den.scale(&inv);
            }
        }
    }
}

// ── Equality ────────────────────────────────────────────────────────────────

impl<R: Field> PartialEq for RationalFunction<R> {
    fn eq(&self, other: &Self) -> bool {
        // Both are reduced with monic denominators, so direct comparison works
        self.num == other.num && self.den == other.den
    }
}

impl<R: Field> Eq for RationalFunction<R> {}

// ── Display ─────────────────────────────────────────────────────────────────

impl<R: Field> fmt::Display for RationalFunction<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_polynomial() {
            write!(f, "{}", self.num)
        } else {
            write!(f, "({}) / ({})", self.num, self.den)
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::BigRational;

    type RPoly = DensePolynomial<BigRational>;
    type RFn = RationalFunction<BigRational>;

    fn ri(coeffs: &[i64]) -> RPoly {
        RPoly::from_coeffs(coeffs.iter().map(|&c| BigRational::from(c)).collect())
    }

    fn rfn(num: &[i64], den: &[i64]) -> RFn {
        RFn::new(ri(num), ri(den))
    }

    // ── Construction and reduction ──────────────────────────────────────

    #[test]
    fn test_auto_reduces() {
        // (x^2 - 1) / (x - 1) = x + 1
        let f = rfn(&[-1, 0, 1], &[-1, 1]);
        assert!(f.is_polynomial());
        assert_eq!(f.numerator().clone(), ri(&[1, 1]));
    }

    #[test]
    fn test_monic_denominator() {
        // (2x) / (2x + 2) should reduce to x / (x + 1)
        let f = rfn(&[0, 2], &[2, 2]);
        assert_eq!(
            *f.denominator().leading_coeff().unwrap(),
            BigRational::from(1i64)
        );
    }

    #[test]
    fn test_from_poly() {
        let p = ri(&[1, 2, 3]);
        let f = RFn::from_poly(p.clone());
        assert!(f.is_polynomial());
        assert_eq!(*f.numerator(), p);
    }

    #[test]
    fn test_zero() {
        let f = RFn::zero();
        assert!(f.is_zero());
        assert!(f.is_polynomial());
    }

    #[test]
    fn test_constant() {
        let f = RFn::constant(BigRational::from(5i64));
        assert!(f.is_polynomial());
        assert_eq!(f.numerator().coeff(0), BigRational::from(5i64));
    }

    #[test]
    #[should_panic(expected = "zero denominator")]
    fn test_zero_denominator_panics() {
        let _ = rfn(&[1, 1], &[]);
    }

    // ── Arithmetic ──────────────────────────────────────────────────────

    #[test]
    fn test_add() {
        // 1/(x-1) + 1/(x+1) = 2x/(x^2-1)
        let a = rfn(&[1], &[-1, 1]);
        let b = rfn(&[1], &[1, 1]);
        let sum = a.add(&b);
        // numerator should be 2x, denominator x^2-1
        assert_eq!(*sum.numerator(), ri(&[0, 2]));
        assert_eq!(*sum.denominator(), ri(&[-1, 0, 1]));
    }

    #[test]
    fn test_add_reduces() {
        // x/(x+1) + 1/(x+1) = (x+1)/(x+1) = 1
        let a = rfn(&[0, 1], &[1, 1]);
        let b = rfn(&[1], &[1, 1]);
        let sum = a.add(&b);
        assert!(sum.is_polynomial());
        assert_eq!(sum.numerator().coeff(0), BigRational::from(1i64));
        assert_eq!(sum.numerator().degree(), Some(0));
    }

    #[test]
    fn test_sub() {
        // 1/(x-1) - 1/(x+1) = 2/(x^2-1)
        let a = rfn(&[1], &[-1, 1]);
        let b = rfn(&[1], &[1, 1]);
        let diff = a.sub(&b);
        assert_eq!(*diff.numerator(), ri(&[2]));
        assert_eq!(*diff.denominator(), ri(&[-1, 0, 1]));
    }

    #[test]
    fn test_mul() {
        // (x+1)/1 * 1/(x+1) = 1
        let a = RFn::from_poly(ri(&[1, 1]));
        let b = rfn(&[1], &[1, 1]);
        let prod = a.mul(&b);
        assert!(prod.is_polynomial());
        assert_eq!(prod.numerator().degree(), Some(0));
    }

    #[test]
    fn test_mul_cross_reduces() {
        // (x^2-1)/(x+2) * (x+2)/(x-1) = x+1
        let a = rfn(&[-1, 0, 1], &[2, 1]);
        let b = rfn(&[2, 1], &[-1, 1]);
        let prod = a.mul(&b);
        assert!(prod.is_polynomial());
        assert_eq!(*prod.numerator(), ri(&[1, 1]));
    }

    #[test]
    fn test_div() {
        // (x/1) / (x/1) = 1
        let x = RFn::from_poly(ri(&[0, 1]));
        let q = x.div(&x);
        assert!(q.is_polynomial());
        assert_eq!(q.numerator().degree(), Some(0));
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn test_div_by_zero_panics() {
        let f = RFn::from_poly(ri(&[1, 1]));
        let _ = f.div(&RFn::zero());
    }

    // ── Evaluation ──────────────────────────────────────────────────────

    #[test]
    fn test_eval() {
        // (x+1)/(x-1) at x=3 → 4/2 = 2
        let f = rfn(&[1, 1], &[-1, 1]);
        let result = f.eval(&BigRational::from(3i64));
        assert_eq!(result, Some(BigRational::from(2i64)));
    }

    #[test]
    fn test_eval_at_pole() {
        // 1/(x-1) at x=1 → None
        let f = rfn(&[1], &[-1, 1]);
        assert_eq!(f.eval(&BigRational::from(1i64)), None);
    }

    // ── Derivative ──────────────────────────────────────────────────────

    #[test]
    fn test_derivative_polynomial() {
        // d/dx(x^2) = 2x as rational function
        let f = RFn::from_poly(ri(&[0, 0, 1]));
        let fp = f.derivative();
        assert!(fp.is_polynomial());
        assert_eq!(*fp.numerator(), ri(&[0, 2]));
    }

    #[test]
    fn test_derivative_fraction() {
        // d/dx(1/x) = -1/x^2
        // 1/x → num=1, den=x → (0*x - 1*1)/x^2 = -1/x^2
        let f = rfn(&[1], &[0, 1]);
        let fp = f.derivative();
        assert_eq!(*fp.numerator(), ri(&[-1]));
        assert_eq!(*fp.denominator(), ri(&[0, 0, 1]));
    }

    // ── Equality ────────────────────────────────────────────────────────

    #[test]
    fn test_equality() {
        let a = rfn(&[-1, 0, 1], &[-1, 1]); // (x^2-1)/(x-1) = x+1
        let b = RFn::from_poly(ri(&[1, 1])); // x+1
        assert_eq!(a, b);
    }

    #[test]
    fn test_inequality() {
        let a = rfn(&[1], &[-1, 1]);
        let b = rfn(&[1], &[1, 1]);
        assert_ne!(a, b);
    }

    // ── Display ─────────────────────────────────────────────────────────

    #[test]
    fn test_display_polynomial() {
        let f = RFn::from_poly(ri(&[1, 2]));
        let s = f.to_string();
        assert!(!s.contains("/"));
    }

    #[test]
    fn test_display_fraction() {
        let f = rfn(&[1], &[-1, 1]);
        let s = f.to_string();
        assert!(s.contains("/"));
    }

    // ── Recip ───────────────────────────────────────────────────────────

    #[test]
    fn test_recip() {
        let f = rfn(&[1, 1], &[-1, 1]);
        let r = f.recip();
        assert_eq!(*r.numerator(), ri(&[-1, 1]));
        assert_eq!(*r.denominator(), ri(&[1, 1]));
    }

    #[test]
    #[should_panic(expected = "reciprocal of zero")]
    fn test_recip_zero_panics() {
        let _ = RFn::zero().recip();
    }

    // ── Neg ─────────────────────────────────────────────────────────────

    #[test]
    fn test_neg() {
        let f = rfn(&[1, 1], &[-1, 1]);
        let n = f.neg();
        assert_eq!(*n.numerator(), ri(&[-1, -1]));
        assert_eq!(*n.denominator(), ri(&[-1, 1]));
    }
}
