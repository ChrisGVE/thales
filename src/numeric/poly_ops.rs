//! Polynomial division and GCD operations.
//!
//! Euclidean division and GCD for [`DensePolynomial`] over fields,
//! and subresultant PRS GCD over Euclidean domains.

use super::dense_poly::DensePolynomial;
use super::ring::{EuclideanDomain, Field, GcdDomain, Ring};

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
            for (i, dc) in divisor.coefficients().iter().enumerate() {
                let idx = i + shift;
                remainder.coeffs_mut()[idx] =
                    remainder.coefficients()[idx].clone() - factor.clone() * dc.clone();
            }
            remainder.trim_zeros();
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

// ── Pseudo-division and Subresultant PRS (EuclideanDomain) ───────────────────

impl<R: EuclideanDomain> DensePolynomial<R> {
    /// Pseudo-remainder: `lc(g)^(deg(f)-deg(g)+1) * f mod g`.
    ///
    /// Does not require field inversion. Returns `None` if `g` is zero.
    pub fn pseudo_rem(&self, g: &Self) -> Option<Self> {
        if g.is_zero() {
            return None;
        }
        if self.is_zero() {
            return Some(Self::zero());
        }
        let g_deg = g.degree().unwrap();
        match self.degree() {
            Some(d) if d >= g_deg => {}
            _ => return Some(self.clone()),
        };

        let lc_g = g.leading_coeff().unwrap().clone();
        let mut r = self.clone();

        while let Some(r_deg) = r.degree() {
            if r_deg < g_deg {
                break;
            }
            let shift = r_deg - g_deg;
            let lc_r = r.leading_coeff().unwrap().clone();

            // r = lc_g * r - lc_r * x^shift * g
            let g_coeffs = g.coefficients();
            let r_coeffs = r.coefficients().to_vec();
            let mut new_coeffs = Vec::with_capacity(r_coeffs.len());
            for (i, rc) in r_coeffs.iter().enumerate() {
                let mut val = rc.clone() * lc_g.clone();
                if i >= shift && (i - shift) < g_coeffs.len() {
                    val = val - lc_r.clone() * g_coeffs[i - shift].clone();
                }
                new_coeffs.push(val);
            }
            r = DensePolynomial::from_coeffs(new_coeffs);
        }
        Some(r)
    }

    /// Subresultant PRS GCD for polynomials over a GCD domain.
    ///
    /// Avoids coefficient explosion of naive Euclidean algorithm
    /// by dividing out known content at each step.
    /// Result is primitive (content = 1 up to units).
    pub fn subresultant_gcd(&self, other: &Self) -> Self {
        if self.is_zero() {
            return other.primitive_part();
        }
        if other.is_zero() {
            return self.primitive_part();
        }

        // Ensure deg(a) >= deg(b)
        let (a, b) = if self.degree() >= other.degree() {
            (self.primitive_part(), other.primitive_part())
        } else {
            (other.primitive_part(), self.primitive_part())
        };

        if b.is_zero() {
            return a;
        }

        let mut prev = a;
        let mut curr = b;
        let mut g = R::one();
        let mut h = R::one();

        loop {
            let delta = prev.degree().unwrap() - curr.degree().unwrap();

            let r = match prev.pseudo_rem(&curr) {
                Some(r) => r,
                None => break,
            };

            if r.is_zero() {
                break;
            }

            // r = prem(prev, curr) / (g * h^delta)
            let divisor = g.clone() * pow_ring(h.clone(), delta);
            let r = exact_content_div(r, &divisor);

            prev = curr;
            curr = r;

            // Update g, h
            g = prev.leading_coeff().unwrap().clone();
            h = if delta == 0 {
                h
            } else {
                let g_pow = pow_ring(g.clone(), delta);
                let h_pow = pow_ring(h.clone(), delta - 1);
                exact_content_div_scalar(g_pow, &h_pow)
            };
        }

        curr.primitive_part()
    }

    /// Content: GCD of all coefficients.
    pub fn content(&self) -> R {
        if self.is_zero() {
            return R::zero();
        }
        let coeffs = self.coefficients();
        let mut g = coeffs[0].clone();
        for c in &coeffs[1..] {
            g = GcdDomain::gcd(&g, c);
            if g.is_one() {
                return g;
            }
        }
        g
    }

    /// Primitive part: self / content(self).
    pub fn primitive_part(&self) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let c = self.content();
        if c.is_one() {
            return self.clone();
        }
        let coeffs = self
            .coefficients()
            .iter()
            .map(|a| {
                let (q, _) = a.div_rem(&c);
                q
            })
            .collect();
        DensePolynomial::from_coeffs(coeffs)
    }
}

/// Raise a ring element to a non-negative integer power.
fn pow_ring<R: Ring>(mut base: R, mut exp: usize) -> R {
    if exp == 0 {
        return R::one();
    }
    let mut result = R::one();
    while exp > 1 {
        if exp % 2 == 1 {
            result = result * base.clone();
        }
        base = base.clone() * base;
        exp /= 2;
    }
    result * base
}

/// Divide every coefficient of a polynomial by a scalar (exact division).
fn exact_content_div<R: EuclideanDomain>(
    poly: DensePolynomial<R>,
    divisor: &R,
) -> DensePolynomial<R> {
    if divisor.is_one() {
        return poly;
    }
    let coeffs = poly
        .coefficients()
        .iter()
        .map(|c| {
            let (q, _) = c.div_rem(divisor);
            q
        })
        .collect();
    DensePolynomial::from_coeffs(coeffs)
}

/// Exact scalar division for ring elements.
fn exact_content_div_scalar<R: EuclideanDomain>(a: R, b: &R) -> R {
    let (q, _) = a.div_rem(b);
    q
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{BigRational, SmallInt};

    // ── Euclidean division tests ─────────────────────────────────────────────

    mod div_rem_tests {
        use super::*;
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
            let f = ri(&[1, 0, 0, 1]);
            let g = ri(&[1, 1]);
            let (q, r) = f.div_rem(&g);
            assert_eq!(q, ri(&[1, -1, 1]));
            assert!(r.is_zero());
        }

        #[test]
        fn test_div_rem_with_remainder() {
            let f = ri(&[1, 0, 1]);
            let g = ri(&[1, 1]);
            let (q, r) = f.div_rem(&g);
            assert_eq!(q, ri(&[-1, 1]));
            assert_eq!(r, ri(&[2]));
        }

        #[test]
        fn test_div_rem_identity() {
            let f = ri(&[3, 2, 5, 1]);
            let g = ri(&[1, 1]);
            let (q, r) = f.div_rem(&g);
            let reconstructed = &(&q * &g) + &r;
            assert_eq!(reconstructed, f);
        }

        #[test]
        fn test_div_rem_degree_smaller() {
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
        type RPoly = DensePolynomial<BigRational>;

        fn ri(coeffs: &[i64]) -> RPoly {
            RPoly::from_coeffs(coeffs.iter().map(|&c| BigRational::from(c)).collect())
        }

        #[test]
        fn test_gcd_basic() {
            let f = ri(&[-1, 0, 1]);
            let g = ri(&[-1, 1]);
            let d = f.gcd(&g);
            assert_eq!(d, ri(&[-1, 1]));
        }

        #[test]
        fn test_gcd_coprime() {
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
            let d = f.gcd(&z);
            assert_eq!(d, f.make_monic());
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
            let f = ri(&[0, -1, 0, 1]);
            let g = ri(&[-1, 0, 1]);
            let d = f.gcd(&g);
            assert_eq!(d, ri(&[-1, 0, 1]));
        }

        #[test]
        fn test_gcd_result_is_monic() {
            let f = ri(&[2, 2]);
            let g = ri(&[3, 3]);
            let d = f.gcd(&g);
            assert_eq!(d, ri(&[1, 1]));
        }

        #[test]
        fn test_gcd_divides_both() {
            let f = ri(&[-1, 0, 0, 1]);
            let g = ri(&[-1, 0, 1]);
            let d = f.gcd(&g);
            let (_, r1) = f.div_rem(&d);
            let (_, r2) = g.div_rem(&d);
            assert!(r1.is_zero());
            assert!(r2.is_zero());
        }
    }

    // ── Subresultant PRS GCD tests ───────────────────────────────────────────

    mod subresultant_tests {
        use super::*;
        type IPoly = DensePolynomial<SmallInt>;

        fn ip(coeffs: &[i64]) -> IPoly {
            IPoly::from_coeffs(coeffs.iter().map(|&c| SmallInt::from(c)).collect())
        }

        #[test]
        fn test_content() {
            let f = ip(&[6, 12, 18]);
            assert_eq!(f.content(), SmallInt::from(6i64));
        }

        #[test]
        fn test_content_coprime() {
            let f = ip(&[3, 5, 7]);
            assert_eq!(f.content(), SmallInt::from(1i64));
        }

        #[test]
        fn test_primitive_part() {
            let f = ip(&[6, 12, 18]);
            let pp = f.primitive_part();
            assert_eq!(pp, ip(&[1, 2, 3]));
        }

        #[test]
        fn test_pseudo_rem() {
            let f = ip(&[1, 0, 1]);
            let g = ip(&[1, 1]);
            let r = f.pseudo_rem(&g).unwrap();
            assert_eq!(r, ip(&[2]));
        }

        #[test]
        fn test_subresultant_gcd_basic() {
            let f = ip(&[-1, 0, 1]);
            let g = ip(&[-1, 1]);
            let d = f.subresultant_gcd(&g);
            let r1 = f.pseudo_rem(&d).unwrap();
            let r2 = g.pseudo_rem(&d).unwrap();
            assert!(r1.is_zero(), "gcd should divide f");
            assert!(r2.is_zero(), "gcd should divide g");
            assert_eq!(d.degree(), Some(1));
        }

        #[test]
        fn test_subresultant_gcd_coprime() {
            let f = ip(&[1, 0, 1]);
            let g = ip(&[1, 1]);
            let d = f.subresultant_gcd(&g);
            assert_eq!(d.degree(), Some(0));
        }

        #[test]
        fn test_subresultant_gcd_with_zero() {
            let f = ip(&[-1, 0, 1]);
            let z = IPoly::zero();
            let d = f.subresultant_gcd(&z);
            assert_eq!(d, f.primitive_part());
        }

        #[test]
        fn test_subresultant_gcd_both_zero() {
            let z = IPoly::zero();
            assert!(z.subresultant_gcd(&z).is_zero());
        }

        #[test]
        fn test_subresultant_no_coefficient_explosion() {
            let factor = ip(&[3, 2]);
            let cofactor = ip(&[1, 1]);
            let f = &(&factor * &factor) * &cofactor;
            let cofactor2 = ip(&[-1, 1]);
            let g = &factor * &cofactor2;
            let d = f.subresultant_gcd(&g);
            assert_eq!(d.degree(), Some(1));
            let r1 = f.pseudo_rem(&d).unwrap();
            let r2 = g.pseudo_rem(&d).unwrap();
            assert!(r1.is_zero());
            assert!(r2.is_zero());
        }
    }
}
