//! Core `RationalFunction<R>` type: struct, constructors, accessors,
//! basic predicates, arithmetic (add/sub/mul/div/neg/recip/pow),
//! operator overloads, `PartialEq`/`Eq`, and `Display`.

use super::super::dense_poly::DensePolynomial;
use super::super::ring::Field;
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
    pub(super) num: DensePolynomial<R>,
    pub(super) den: DensePolynomial<R>,
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
    pub(super) fn reduce(&mut self) {
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
