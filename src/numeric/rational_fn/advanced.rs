//! Advanced `RationalFunction<R>` operations: polynomial long division,
//! proper-form conversion, formal derivative, partial fraction decomposition,
//! composition, and polynomial helper utilities.

use super::super::dense_poly::DensePolynomial;
use super::super::poly_factoring::SqfFactor;
use super::super::ring::Field;
use super::core::{PartialFractionTerm, RationalFunction};

impl<R: Field> RationalFunction<R> {
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
        let (poly_part, proper) = self.to_proper();

        if proper.is_zero() {
            return (poly_part, vec![]);
        }

        let sqf_factors = proper.den.yun_sqf();

        if sqf_factors.is_empty() {
            return (poly_part, vec![]);
        }

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
            let mut f_power = factor.clone();
            for j in 1..=*multiplicity {
                let den_over_fpow = proper.den.div_rem(&f_power).0;
                let (_q, a_j) = remainder.div_rem(factor);

                if a_j.degree().unwrap_or(0) > 0 || !a_j.is_zero() {
                    let other_mod_f = den_over_fpow.div_rem(factor).1;

                    if !other_mod_f.is_zero() {
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

                            let consumed = &a_j_adjusted * &den_over_fpow;
                            remainder = (&remainder - &consumed).div_rem(&f_power).0;
                        } else {
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
        let num = poly_eval_rational(&self.num, other);
        let den = poly_eval_rational(&self.den, other);
        num.div(&den)
    }
}

/// Evaluate a polynomial at a rational function using Horner's method.
pub(super) fn poly_eval_rational<R: Field>(
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
pub(super) fn poly_mod_inverse<R: Field>(
    a: &DensePolynomial<R>,
    m: &DensePolynomial<R>,
) -> Option<DensePolynomial<R>> {
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

    // old_r = gcd; must be degree-0 (constant) for inverse to exist
    if old_r.degree().unwrap_or(0) > 0 {
        return None;
    }

    if let Some(lc) = old_r.leading_coeff() {
        let inv_lc = lc.inv();
        let result = old_s.scale(&inv_lc);
        let (_, reduced) = result.div_rem(m);
        Some(reduced)
    } else {
        None
    }
}
