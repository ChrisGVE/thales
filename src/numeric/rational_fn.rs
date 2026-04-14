//! Rational function type: numerator/denominator polynomial fraction.
//!
//! [`RationalFunction<R>`] represents `p(x)/q(x)` where `p` and `q` are
//! dense univariate polynomials over a field `R`. Automatically reduces
//! to lowest terms via GCD on construction.

use super::dense_poly::DensePolynomial;
use super::poly_factoring::SqfFactor;
use super::ring::Field;
use std::fmt;
use std::ops;

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

/// A single partial fraction term: `numerator / (factor^power)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartialFractionTerm<R: Field> {
    /// Numerator polynomial (degree < degree of `factor`).
    pub numerator: DensePolynomial<R>,
    /// Irreducible (squarefree) denominator factor.
    pub factor: DensePolynomial<R>,
    /// Power of the factor in the decomposition.
    pub power: usize,
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

    /// Returns `true` if improper (deg numerator >= deg denominator).
    pub fn is_improper(&self) -> bool {
        match (self.num.degree(), self.den.degree()) {
            (Some(n), Some(d)) => n >= d,
            (Some(_), None) => true, // den is zero — shouldn't happen
            (None, _) => false,      // zero numerator is proper
        }
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

    /// Add two rational functions using GCD-optimized denominator.
    ///
    /// Instead of naive `(ad+bc)/bd`, computes `g = gcd(b,d)` and uses
    /// `lcm(b,d) = b*(d/g)` as denominator, keeping intermediate
    /// coefficients smaller.
    pub fn add(&self, other: &Self) -> Self {
        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }

        let g = self.den.gcd(&other.den);

        if g.degree().unwrap_or(0) == 0 && g.leading_coeff().map_or(true, |c| c.is_one()) {
            // Denominators are coprime — fall back to direct computation
            let num = &(&self.num * &other.den) + &(&other.num * &self.den);
            let den = &self.den * &other.den;
            return Self::new(num, den);
        }

        // den1 = g * d1_cofactor, den2 = g * d2_cofactor
        let d1_cofactor = self.den.div_rem(&g).0;
        let d2_cofactor = other.den.div_rem(&g).0;

        // numerator = num1 * d2_cofactor + num2 * d1_cofactor
        let num = &(&self.num * &d2_cofactor) + &(&other.num * &d1_cofactor);
        // denominator = d1_cofactor * den2 = lcm(den1, den2)
        let den = &d1_cofactor * &other.den;

        Self::new(num, den)
    }

    /// Subtract: `self - other`.
    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.neg())
    }

    /// Multiply two rational functions, reducing cross-terms.
    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }

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

    /// Polynomial long division: split into polynomial part + proper remainder.
    ///
    /// Returns `(q, r)` where `self = q + r/den` and `deg(r) < deg(den)`.
    /// If already proper, `q` is zero.
    pub fn polynomial_division(&self) -> (DensePolynomial<R>, DensePolynomial<R>) {
        if !self.is_improper() {
            return (DensePolynomial::zero(), self.num.clone());
        }
        self.num.div_rem(&self.den)
    }

    /// Extract the polynomial part of an improper rational function.
    ///
    /// For `p(x)/q(x)` where `deg(p) >= deg(q)`, returns the quotient
    /// of `p / q`.
    pub fn polynomial_part(&self) -> DensePolynomial<R> {
        self.polynomial_division().0
    }

    /// Convert to proper form: returns `(polynomial_part, proper_rational)`.
    ///
    /// `self = polynomial_part + proper_rational` where
    /// `deg(proper_rational.numerator) < deg(proper_rational.denominator)`.
    pub fn to_proper(&self) -> (DensePolynomial<R>, Self) {
        let (q, r) = self.polynomial_division();
        let proper = RationalFunction {
            num: r,
            den: self.den.clone(),
        };
        // proper is already reduced since self was reduced and q*den + r = num
        // with gcd(num, den) = 1, we have gcd(r, den) = 1
        (q, proper)
    }

    /// Formal derivative using quotient rule: `(n'd - nd') / d^2`.
    pub fn derivative(&self) -> Self {
        let np = self.num.derivative();
        let dp = self.den.derivative();
        let num = &(&np * &self.den) - &(&self.num * &dp);
        let den = &self.den * &self.den;
        Self::new(num, den)
    }

    /// Partial fraction decomposition.
    ///
    /// Decomposes a proper rational function into a sum of simpler fractions.
    /// If the rational function is improper, extracts the polynomial part first.
    ///
    /// Returns `(polynomial_part, terms)` where each term has
    /// `deg(term.numerator) < deg(term.factor)`.
    ///
    /// Uses squarefree factorization of the denominator via Yun's algorithm,
    /// then the extended partial fraction algorithm for each power.
    pub fn partial_fraction_decomposition(
        &self,
    ) -> (DensePolynomial<R>, Vec<PartialFractionTerm<R>>) {
        // Extract polynomial part if improper
        let (poly_part, proper) = self.to_proper();

        if proper.is_zero() {
            return (poly_part, vec![]);
        }

        // Squarefree factorization of denominator
        let sqf_factors = proper.den.yun_sqf();

        if sqf_factors.is_empty() {
            // Denominator is constant → proper part is zero (already handled)
            return (poly_part, vec![]);
        }

        // Single irreducible factor raised to some power — no decomposition needed
        if sqf_factors.len() == 1 && sqf_factors[0].multiplicity == 1 {
            return (
                poly_part,
                vec![PartialFractionTerm {
                    numerator: proper.num,
                    factor: sqf_factors[0].factor.clone(),
                    power: 1,
                }],
            );
        }

        let mut terms = Vec::new();
        let mut remainder = proper.num.clone();

        for SqfFactor {
            factor,
            multiplicity,
        } in &sqf_factors
        {
            // For factor f^k, extract terms A_1/f + A_2/f^2 + ... + A_k/f^k
            // where deg(A_i) < deg(f)
            let mut f_power = factor.clone();
            for j in 1..=*multiplicity {
                // Compute the "other" part of the denominator:
                // other = den / f_power (evaluated at current state)
                let den_over_fpow = proper.den.div_rem(&f_power).0;

                // Extended Euclidean-style: find the numerator for this term
                // remainder = A_j * den_over_fpow + new_remainder * factor
                // where deg(A_j) < deg(factor)
                let (q, a_j) = remainder.div_rem(factor);

                if a_j.degree().unwrap_or(0) > 0 || !a_j.is_zero() {
                    // We need to verify: multiply back and adjust
                    // Actually, use the Hermite-like approach:
                    // remainder / (f^k * other) = A_j / f^j + rest
                    // where A_j = remainder mod factor, adjusted by inverse of other mod factor

                    let other_mod_f = den_over_fpow.div_rem(factor).1;

                    if !other_mod_f.is_zero() {
                        // Compute inverse of other_mod_f modulo factor
                        // Using extended GCD: gcd(other_mod_f, factor) should be 1
                        // (since squarefree factors are coprime)
                        // inv = other_mod_f^(-1) mod factor
                        let inv = poly_mod_inverse(&other_mod_f, factor);

                        if let Some(inv) = inv {
                            let a_j_adjusted = (&remainder * &inv).div_rem(factor).1;
                            if !a_j_adjusted.is_zero() {
                                terms.push(PartialFractionTerm {
                                    numerator: a_j_adjusted.clone(),
                                    factor: factor.clone(),
                                    power: *multiplicity - j + 1,
                                });
                            }

                            // Update remainder: remainder -= a_j_adjusted * den_over_fpow
                            let consumed = &a_j_adjusted * &den_over_fpow;
                            remainder = (&remainder - &consumed).div_rem(&f_power).0;
                            // After division by f, we continue decomposing
                            // the next power down
                        } else {
                            // Inverse doesn't exist — factors not coprime.
                            // Fallback: emit remainder over full remaining denominator.
                            break;
                        }
                    }
                }

                f_power = &f_power * factor;
            }
        }

        (poly_part, terms)
    }

    /// Compose with another rational function: `self(other(x))`.
    pub fn compose(&self, other: &Self) -> Self {
        // Evaluate numerator and denominator polynomials at other
        let num = poly_eval_rational(&self.num, other);
        let den = poly_eval_rational(&self.den, other);
        num.div(&den)
    }

    /// Raise to a non-negative integer power.
    pub fn pow(&self, n: u32) -> Self {
        if n == 0 {
            return Self::constant(R::one());
        }
        let mut result = self.clone();
        for _ in 1..n {
            result = result.mul(self);
        }
        result
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

/// Evaluate a polynomial at a rational function using Horner's method.
fn poly_eval_rational<R: Field>(
    poly: &DensePolynomial<R>,
    at: &RationalFunction<R>,
) -> RationalFunction<R> {
    if poly.is_zero() {
        return RationalFunction::zero();
    }
    let coeffs = poly.coefficients();
    let mut result = RationalFunction::constant(coeffs.last().unwrap().clone());
    for c in coeffs.iter().rev().skip(1) {
        result = result.mul(at).add(&RationalFunction::constant(c.clone()));
    }
    result
}

/// Compute modular inverse of `a` modulo `m` using extended Euclidean algorithm.
///
/// Returns `Some(inv)` where `a * inv ≡ 1 (mod m)`, or `None` if
/// `gcd(a, m) != 1`.
fn poly_mod_inverse<R: Field>(
    a: &DensePolynomial<R>,
    m: &DensePolynomial<R>,
) -> Option<DensePolynomial<R>> {
    // Extended GCD: find s, t such that a*s + m*t = gcd(a, m)
    let mut old_r = a.clone();
    let mut r = m.clone();
    let mut old_s = DensePolynomial::constant(R::one());
    let mut s = DensePolynomial::zero();

    while !r.is_zero() {
        let (q, rem) = old_r.div_rem(&r);
        old_r = r;
        r = rem;

        let new_s = &old_s - &(&q * &s);
        old_s = s;
        s = new_s;
    }

    // old_r = gcd, old_s = inverse coefficient
    // Check gcd is constant (degree 0)
    if old_r.degree().unwrap_or(0) > 0 {
        return None;
    }

    // Normalize: old_s / leading_coeff(old_r)
    if let Some(lc) = old_r.leading_coeff() {
        let inv_lc = lc.inv();
        let result = old_s.scale(&inv_lc);
        // Reduce modulo m
        let (_, reduced) = result.div_rem(m);
        Some(reduced)
    } else {
        None
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

// ── Operator overloads ──────────────────────────────────────────────────────

impl<R: Field> ops::Add for &RationalFunction<R> {
    type Output = RationalFunction<R>;

    fn add(self, rhs: Self) -> RationalFunction<R> {
        RationalFunction::add(self, rhs)
    }
}

impl<R: Field> ops::Add for RationalFunction<R> {
    type Output = RationalFunction<R>;

    fn add(self, rhs: Self) -> RationalFunction<R> {
        RationalFunction::add(&self, &rhs)
    }
}

impl<R: Field> ops::Sub for &RationalFunction<R> {
    type Output = RationalFunction<R>;

    fn sub(self, rhs: Self) -> RationalFunction<R> {
        RationalFunction::sub(self, rhs)
    }
}

impl<R: Field> ops::Sub for RationalFunction<R> {
    type Output = RationalFunction<R>;

    fn sub(self, rhs: Self) -> RationalFunction<R> {
        RationalFunction::sub(&self, &rhs)
    }
}

impl<R: Field> ops::Mul for &RationalFunction<R> {
    type Output = RationalFunction<R>;

    fn mul(self, rhs: Self) -> RationalFunction<R> {
        RationalFunction::mul(self, rhs)
    }
}

impl<R: Field> ops::Mul for RationalFunction<R> {
    type Output = RationalFunction<R>;

    fn mul(self, rhs: Self) -> RationalFunction<R> {
        RationalFunction::mul(&self, &rhs)
    }
}

impl<R: Field> ops::Div for &RationalFunction<R> {
    type Output = RationalFunction<R>;

    fn div(self, rhs: Self) -> RationalFunction<R> {
        RationalFunction::div(self, rhs)
    }
}

impl<R: Field> ops::Div for RationalFunction<R> {
    type Output = RationalFunction<R>;

    fn div(self, rhs: Self) -> RationalFunction<R> {
        RationalFunction::div(&self, &rhs)
    }
}

impl<R: Field> ops::Neg for &RationalFunction<R> {
    type Output = RationalFunction<R>;

    fn neg(self) -> RationalFunction<R> {
        RationalFunction::neg(self)
    }
}

impl<R: Field> ops::Neg for RationalFunction<R> {
    type Output = RationalFunction<R>;

    fn neg(self) -> RationalFunction<R> {
        RationalFunction::neg(&self)
    }
}

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
    use crate::numeric::ring::Ring;
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
    fn test_add_same_denominator() {
        // 1/(x+1) + x/(x+1) should use GCD optimization (dens are identical)
        let a = rfn(&[1], &[1, 1]);
        let b = rfn(&[0, 1], &[1, 1]);
        let sum = a.add(&b);
        assert!(sum.is_polynomial());
        assert_eq!(sum.numerator().coeff(0), BigRational::from(1i64));
    }

    #[test]
    fn test_add_with_zero() {
        let a = rfn(&[1], &[-1, 1]);
        let z = RFn::zero();
        assert_eq!(a.add(&z), a);
        assert_eq!(z.add(&a), a);
    }

    #[test]
    fn test_add_gcd_optimization() {
        // 1/(x*(x+1)) + 1/(x*(x-1))
        // Denominators share factor x, so GCD = x
        // Result = ((x-1) + (x+1)) / (x*(x+1)*(x-1)) = 2x / (x*(x^2-1)) = 2/(x^2-1)
        let a = rfn(&[1], &[0, 1, 1]); // 1/(x^2+x) = 1/(x(x+1))
        let b = rfn(&[1], &[0, -1, 1]); // 1/(x^2-x) = 1/(x(x-1))
        let sum = a.add(&b);
        assert_eq!(*sum.numerator(), ri(&[2]));
        assert_eq!(*sum.denominator(), ri(&[-1, 0, 1])); // x^2-1
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
    fn test_mul_by_zero() {
        let a = rfn(&[1, 1], &[-1, 1]);
        let z = RFn::zero();
        assert!(a.mul(&z).is_zero());
        assert!(z.mul(&a).is_zero());
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

    // ── Operator overloads ──────────────────────────────────────────────

    #[test]
    fn test_ops_add() {
        let a = rfn(&[1], &[-1, 1]);
        let b = rfn(&[1], &[1, 1]);
        let sum = &a + &b;
        assert_eq!(sum, a.add(&b));
    }

    #[test]
    fn test_ops_sub() {
        let a = rfn(&[1], &[-1, 1]);
        let b = rfn(&[1], &[1, 1]);
        let diff = &a - &b;
        assert_eq!(diff, a.sub(&b));
    }

    #[test]
    fn test_ops_mul() {
        let a = RFn::from_poly(ri(&[1, 1]));
        let b = rfn(&[1], &[1, 1]);
        let prod = &a * &b;
        assert_eq!(prod, a.mul(&b));
    }

    #[test]
    fn test_ops_div() {
        let a = rfn(&[1, 1], &[-1, 1]);
        let b = rfn(&[1], &[1, 1]);
        let q = &a / &b;
        assert_eq!(q, a.div(&b));
    }

    #[test]
    fn test_ops_neg() {
        let a = rfn(&[1, 1], &[-1, 1]);
        let neg = -&a;
        assert_eq!(neg, a.neg());
    }

    #[test]
    fn test_ops_owned() {
        let a = rfn(&[1], &[-1, 1]);
        let b = rfn(&[1], &[1, 1]);
        let expected = a.add(&b);
        let sum = a + b;
        assert_eq!(sum, expected);
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
        let f = rfn(&[1], &[0, 1]);
        let fp = f.derivative();
        assert_eq!(*fp.numerator(), ri(&[-1]));
        assert_eq!(*fp.denominator(), ri(&[0, 0, 1]));
    }

    // ── Polynomial division / proper form ───────────────────────────────

    #[test]
    fn test_is_improper() {
        // (x^2 + 1) / (x + 1) is improper
        let f = rfn(&[1, 0, 1], &[1, 1]);
        assert!(f.is_improper());
    }

    #[test]
    fn test_is_proper() {
        // 1 / (x + 1) is proper
        let f = rfn(&[1], &[1, 1]);
        assert!(!f.is_improper());
    }

    #[test]
    fn test_polynomial_division() {
        // (x^2 + 1) / (x + 1) = (x - 1) + 2/(x+1)
        let f = rfn(&[1, 0, 1], &[1, 1]);
        let (q, r) = f.polynomial_division();
        assert_eq!(q, ri(&[-1, 1])); // x - 1
        assert_eq!(r, ri(&[2])); // 2
    }

    #[test]
    fn test_polynomial_division_already_proper() {
        let f = rfn(&[1], &[1, 1]);
        let (q, r) = f.polynomial_division();
        assert!(q.is_zero());
        assert_eq!(r, ri(&[1]));
    }

    #[test]
    fn test_polynomial_part() {
        // (x^3) / (x + 1) = x^2 - x + 1 remainder -1
        let f = rfn(&[0, 0, 0, 1], &[1, 1]);
        let pp = f.polynomial_part();
        // x^3 / (x+1) = x^2 - x + 1 - 1/(x+1)
        assert_eq!(pp, ri(&[1, -1, 1]));
    }

    #[test]
    fn test_to_proper() {
        // (x^2 + x + 1) / (x + 1) = x + 1/(x+1)
        // Wait: x^2+x+1 = (x+1)*x + 1, so quotient = x, remainder = 1
        let f = rfn(&[1, 1, 1], &[1, 1]);
        let (poly, proper) = f.to_proper();
        assert_eq!(poly, ri(&[0, 1])); // x
        assert_eq!(*proper.numerator(), ri(&[1])); // 1
        assert_eq!(*proper.denominator(), ri(&[1, 1])); // x+1
    }

    #[test]
    fn test_to_proper_reconstructs() {
        // Verify: poly + proper = original
        let f = rfn(&[1, 0, 0, 1], &[1, 1]); // (x^3+1)/(x+1) = x^2-x+1 (exact)
        let (poly, proper) = f.to_proper();
        let reconstructed = RFn::from_poly(poly).add(&proper);
        assert_eq!(reconstructed, f);
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

    // ── Compose ─────────────────────────────────────────────────────────

    #[test]
    fn test_compose_polynomial_at_x() {
        // (x+1) composed with x = x+1
        let f = RFn::from_poly(ri(&[1, 1]));
        let x = RFn::x();
        assert_eq!(f.compose(&x), f);
    }

    #[test]
    fn test_compose_polynomial_at_value() {
        // (x^2) composed with (x+1) = (x+1)^2 = x^2+2x+1
        let f = RFn::from_poly(ri(&[0, 0, 1]));
        let g = RFn::from_poly(ri(&[1, 1]));
        let result = f.compose(&g);
        assert!(result.is_polynomial());
        assert_eq!(*result.numerator(), ri(&[1, 2, 1]));
    }

    // ── Pow ─────────────────────────────────────────────────────────────

    #[test]
    fn test_pow_zero() {
        let f = rfn(&[1, 1], &[-1, 1]);
        let p = f.pow(0);
        assert!(p.is_polynomial());
        assert_eq!(p.numerator().coeff(0), BigRational::from(1i64));
    }

    #[test]
    fn test_pow_one() {
        let f = rfn(&[1, 1], &[-1, 1]);
        assert_eq!(f.pow(1), f);
    }

    #[test]
    fn test_pow_two() {
        // ((x+1)/(x-1))^2 = (x^2+2x+1)/(x^2-2x+1)
        let f = rfn(&[1, 1], &[-1, 1]);
        let p = f.pow(2);
        assert_eq!(*p.numerator(), ri(&[1, 2, 1]));
        assert_eq!(*p.denominator(), ri(&[1, -2, 1]));
    }

    // ── Partial fraction decomposition ──────────────────────────────────

    #[test]
    fn test_pfd_simple_linear() {
        // 1/(x+1) — already partial
        let f = rfn(&[1], &[1, 1]);
        let (poly, terms) = f.partial_fraction_decomposition();
        assert!(poly.is_zero());
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].power, 1);
    }

    #[test]
    fn test_pfd_improper() {
        // (x^2+1)/(x+1) = x - 1 + 2/(x+1)
        let f = rfn(&[1, 0, 1], &[1, 1]);
        let (poly, terms) = f.partial_fraction_decomposition();
        assert_eq!(poly, ri(&[-1, 1])); // x - 1
        assert!(!terms.is_empty());
    }

    #[test]
    fn test_pfd_zero() {
        let f = RFn::zero();
        let (poly, terms) = f.partial_fraction_decomposition();
        assert!(poly.is_zero());
        assert!(terms.is_empty());
    }

    #[test]
    fn test_pfd_polynomial_input() {
        // x^2 + 1 (no denominator) — should return polynomial, no terms
        let f = RFn::from_poly(ri(&[1, 0, 1]));
        let (poly, terms) = f.partial_fraction_decomposition();
        assert_eq!(poly, ri(&[1, 0, 1]));
        assert!(terms.is_empty());
    }

    // ── Coefficient size doesn't explode ────────────────────────────────

    #[test]
    fn test_large_fraction_arithmetic_stays_small() {
        // Build up (1/(x-1) + 1/(x-2) + 1/(x-3) + 1/(x-4))
        // and verify it doesn't produce huge intermediate coefficients
        let mut sum = RFn::zero();
        for k in 1..=4 {
            let term = rfn(&[1], &[-(k as i64), 1]);
            sum = sum.add(&term);
        }
        // Result should be reduced. Denominator degree = 4.
        assert_eq!(sum.denominator().degree(), Some(4));
        // Numerator degree = 3 (proper fraction)
        assert_eq!(sum.numerator().degree(), Some(3));
    }

    // ── Mod inverse utility ─────────────────────────────────────────────

    #[test]
    fn test_poly_mod_inverse() {
        // Inverse of (x+1) mod (x^2-1) doesn't exist (gcd = x+1)
        let a = ri(&[1, 1]);
        let m = ri(&[-1, 0, 1]);
        assert!(poly_mod_inverse(&a, &m).is_none());
    }

    #[test]
    fn test_poly_mod_inverse_exists() {
        // Inverse of 2 mod (x+1) = 1/2
        let a = ri(&[2]);
        let m = ri(&[1, 1]);
        let inv = poly_mod_inverse(&a, &m).unwrap();
        // 2 * inv ≡ 1 (mod x+1)
        let product = (&a * &inv).div_rem(&m).1;
        assert_eq!(product.degree(), Some(0));
        assert!(product.coeff(0).is_one());
    }
}
