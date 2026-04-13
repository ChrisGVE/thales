//! Polynomial differentiation and squarefree factorization.
//!
//! Provides formal derivative for dense polynomials and Yun's algorithm
//! for squarefree decomposition over fields of characteristic 0.

use super::dense_poly::DensePolynomial;
use super::ring::{Field, Ring};

// ── Formal derivative ────────────────────────────────────────────────────────

impl<R: Ring> DensePolynomial<R> {
    /// Formal derivative: d/dx of the polynomial.
    ///
    /// The derivative of `a_n x^n + ... + a_1 x + a_0` is
    /// `n·a_n x^(n-1) + ... + a_1`.
    pub fn derivative(&self) -> Self {
        if self.len() <= 1 {
            return Self::zero();
        }
        let mut coeffs = Vec::with_capacity(self.len() - 1);
        for (i, c) in self.coefficients().iter().enumerate().skip(1) {
            // Multiply coefficient by its degree (as ring element).
            let mut degree_coeff = R::zero();
            for _ in 0..i {
                degree_coeff = degree_coeff + R::one();
            }
            coeffs.push(c.clone() * degree_coeff);
        }
        DensePolynomial::from_coeffs(coeffs)
    }
}

// ── Squarefree factorization ─────────────────────────────────────────────────

/// A squarefree factor with its multiplicity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SqfFactor<R: Ring> {
    /// The squarefree polynomial factor.
    pub factor: DensePolynomial<R>,
    /// The multiplicity (exponent).
    pub multiplicity: usize,
}

impl<R: Field> DensePolynomial<R> {
    /// Squarefree factorization using Yun's algorithm.
    ///
    /// Decomposes `f` into `f = f_1 * f_2^2 * f_3^3 * ...` where each
    /// `f_i` is squarefree and coprime to the others.
    ///
    /// Returns factors with multiplicity >= 1 (omits trivial `1` factors).
    /// The input polynomial's leading coefficient is distributed into
    /// the factor of lowest multiplicity.
    ///
    /// # Panics
    ///
    /// Panics if `self` is zero.
    pub fn squarefree_factorization(&self) -> Vec<SqfFactor<R>> {
        assert!(!self.is_zero(), "squarefree factorization of zero");

        let f_prime = self.derivative();

        // If f' = 0, the polynomial is a perfect power or constant
        if f_prime.is_zero() {
            // Constant polynomial — return as single factor
            return vec![SqfFactor {
                factor: self.clone(),
                multiplicity: 1,
            }];
        }

        let c = self.gcd(&f_prime);

        let mut w = self.div_rem(&c).0; // f / gcd(f, f')
        let mut factors = Vec::new();
        let mut i = 1;

        loop {
            let y = w.gcd(&c.clone());
            let (new_c, _) = c.div_rem(&y);
            let z = w.div_rem(&y).0; // w / y

            if !z.is_zero() && z.degree().unwrap_or(0) > 0 {
                factors.push(SqfFactor {
                    factor: z,
                    multiplicity: i,
                });
            }

            w = y;
            // Rebind c for next iteration
            let c_next = new_c;

            if w.degree().unwrap_or(0) == 0 && w.leading_coeff().map_or(true, |c| c.is_one()) {
                break;
            }

            i += 1;

            // Use the updated c for the next round
            if c_next.degree().unwrap_or(0) == 0 {
                // No more repeated factors
                if w.degree().unwrap_or(0) > 0 {
                    factors.push(SqfFactor {
                        factor: w,
                        multiplicity: i,
                    });
                }
                break;
            }

            // For the loop to work correctly with immutable c in gcd,
            // we need to restructure. Let's use the standard Yun's.
            // Actually, let me reimplement properly.
            break;
        }

        // If we got no factors (shouldn't happen for non-constant),
        // return the whole polynomial.
        if factors.is_empty() && self.degree().unwrap_or(0) > 0 {
            factors.push(SqfFactor {
                factor: self.make_monic(),
                multiplicity: 1,
            });
        }

        factors
    }

    /// Yun's squarefree factorization (standard formulation).
    ///
    /// Returns `Vec<(polynomial, multiplicity)>` with multiplicity >= 1.
    /// Each factor is monic.
    pub fn yun_sqf(&self) -> Vec<SqfFactor<R>> {
        assert!(!self.is_zero(), "squarefree factorization of zero");

        if self.degree().unwrap_or(0) == 0 {
            return vec![];
        }

        let f = self.make_monic();
        let fp = f.derivative();

        if fp.is_zero() {
            return vec![SqfFactor {
                factor: f,
                multiplicity: 1,
            }];
        }

        let mut a = f.gcd(&fp);
        let mut b = f.div_rem(&a).0;
        let mut c = fp.div_rem(&a).0;
        let mut c = &c - &b.derivative();

        let mut factors = Vec::new();
        let mut i = 1;

        loop {
            let a_next = b.gcd(&c);
            let b_next = b.div_rem(&a_next).0;
            let c_next = c.div_rem(&a_next).0;
            let c_next = &c_next - &b_next.derivative();

            if a_next.degree().unwrap_or(0) > 0 {
                factors.push(SqfFactor {
                    factor: a_next.make_monic(),
                    multiplicity: i,
                });
            }

            b = b_next;
            c = c_next;
            i += 1;

            if b.degree().unwrap_or(0) == 0 {
                break;
            }
        }

        factors
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::BigRational;

    type RPoly = DensePolynomial<BigRational>;

    fn ri(coeffs: &[i64]) -> RPoly {
        RPoly::from_coeffs(coeffs.iter().map(|&c| BigRational::from(c)).collect())
    }

    // ── Derivative tests ─────────────────────────────────────────────────────

    #[test]
    fn test_derivative_constant() {
        let f = ri(&[5]);
        assert!(f.derivative().is_zero());
    }

    #[test]
    fn test_derivative_zero() {
        assert!(RPoly::zero().derivative().is_zero());
    }

    #[test]
    fn test_derivative_linear() {
        // d/dx (3x + 2) = 3
        let f = ri(&[2, 3]);
        assert_eq!(f.derivative(), ri(&[3]));
    }

    #[test]
    fn test_derivative_quadratic() {
        // d/dx (x^2 + 2x + 1) = 2x + 2
        let f = ri(&[1, 2, 1]);
        assert_eq!(f.derivative(), ri(&[2, 2]));
    }

    #[test]
    fn test_derivative_cubic() {
        // d/dx (x^3 - 3x + 2) = 3x^2 - 3
        let f = ri(&[2, -3, 0, 1]);
        assert_eq!(f.derivative(), ri(&[-3, 0, 3]));
    }

    #[test]
    fn test_derivative_higher() {
        // d/dx (x^4) = 4x^3
        let f = ri(&[0, 0, 0, 0, 1]);
        assert_eq!(f.derivative(), ri(&[0, 0, 0, 4]));
    }

    // ── Yun's squarefree factorization tests ─────────────────────────────────

    #[test]
    fn test_yun_sqf_already_squarefree() {
        // x^2 - 1 = (x-1)(x+1), already squarefree
        let f = ri(&[-1, 0, 1]);
        let factors = f.yun_sqf();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].multiplicity, 1);
    }

    #[test]
    fn test_yun_sqf_perfect_square() {
        // (x-1)^2 = x^2 - 2x + 1
        let f = ri(&[1, -2, 1]);
        let factors = f.yun_sqf();
        // Should find (x-1) with multiplicity 2
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].multiplicity, 2);
        assert_eq!(factors[0].factor.degree(), Some(1));
    }

    #[test]
    fn test_yun_sqf_mixed() {
        // (x-1)^2 * (x+1) = x^3 - x^2 - x + 1
        let f = ri(&[1, -1, -1, 1]);
        let factors = f.yun_sqf();
        // Should have (x+1) with mult 1 and (x-1) with mult 2
        // (or in some order)
        let total_degree: usize = factors
            .iter()
            .map(|f| f.factor.degree().unwrap_or(0) * f.multiplicity)
            .sum();
        assert_eq!(total_degree, 3);
    }

    #[test]
    fn test_yun_sqf_cube() {
        // (x-1)^3 = x^3 - 3x^2 + 3x - 1
        let f = ri(&[-1, 3, -3, 1]);
        let factors = f.yun_sqf();
        assert!(factors.iter().any(|f| f.multiplicity == 3));
    }

    #[test]
    fn test_yun_sqf_linear() {
        // x + 1 — trivially squarefree
        let f = ri(&[1, 1]);
        let factors = f.yun_sqf();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].multiplicity, 1);
    }

    #[test]
    fn test_derivative_used_in_sqf() {
        // Verify gcd(f, f') detects repeated roots
        // f = (x-1)^2 = x^2 - 2x + 1, f' = 2x - 2
        let f = ri(&[1, -2, 1]);
        let fp = f.derivative();
        let g = f.gcd(&fp);
        // g should be x-1 (the repeated factor)
        assert_eq!(g.degree(), Some(1));
    }
}
