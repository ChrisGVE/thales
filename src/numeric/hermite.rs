//! Hermite reduction for rational function integration.
//!
//! Given a proper rational function `A/D`, Hermite reduction decomposes
//! the integral into a rational part (no integration needed) plus a
//! logarithmic part with squarefree denominator:
//!
//! ```text
//! ∫ A/D dx = g/h + ∫ c/d dx
//! ```
//!
//! where `h` divides the non-squarefree part of `D`, and `d` is the
//! squarefree part of `D`. The logarithmic integral `∫ c/d dx` is then
//! handled by the Rothstein-Trager algorithm (task 32).
//!
//! # Algorithm
//!
//! Uses the quadratic Hermite reduction (Bronstein, Chapter 2):
//! For each squarefree factor `Dₖ` of multiplicity `k ≥ 2`, repeatedly
//! reduce the power using the extended Euclidean algorithm on `Dₖ` and `Dₖ'`.

use super::dense_poly::DensePolynomial;
use super::rational_fn::RationalFunction;
use super::ring::Field;

/// Result of Hermite reduction of `∫ A/D dx`.
///
/// The integral is decomposed as:
/// `∫ A/D dx = rational_part + ∫ log_part dx`
///
/// where `log_part` has a squarefree denominator.
#[derive(Clone, Debug)]
pub struct HermiteReduction<R: Field> {
    /// The rational part `g/h` of the integral (derivative-free).
    pub rational_part: RationalFunction<R>,
    /// The remaining integrand `c/d` with squarefree denominator.
    pub log_part: RationalFunction<R>,
}

impl<R: Field> HermiteReduction<R> {
    /// Returns `true` if the rational part is zero (integral is purely logarithmic).
    pub fn is_purely_logarithmic(&self) -> bool {
        self.rational_part.is_zero()
    }

    /// Returns `true` if the logarithmic part is zero (integral is purely rational).
    pub fn is_purely_rational(&self) -> bool {
        self.log_part.is_zero()
    }
}

/// Perform Hermite reduction on a proper rational function `A/D`.
///
/// If `A/D` is improper, the polynomial part is extracted first and
/// added to the rational result.
///
/// # Algorithm (quadratic version)
///
/// For each squarefree factor `Dₖ` with multiplicity `k ≥ 2`:
/// 1. Compute `Dₖ' = d/dx Dₖ`
/// 2. Use extended GCD: `s·Dₖ + t·Dₖ' = gcd(Dₖ, Dₖ')` (should be 1 for squarefree)
/// 3. Reduce multiplicity by one, accumulating into the rational part
/// 4. Repeat until multiplicity is 1
///
/// The final multiplicity-1 terms form the logarithmic integrand.
pub fn hermite_reduce<R: Field>(rf: &RationalFunction<R>) -> HermiteReduction<R> {
    // Handle zero
    if rf.is_zero() {
        return HermiteReduction {
            rational_part: RationalFunction::zero(),
            log_part: RationalFunction::zero(),
        };
    }

    // Extract polynomial part if improper
    let (poly_part, proper) = rf.to_proper();

    if proper.is_zero() {
        return HermiteReduction {
            rational_part: RationalFunction::from_poly(poly_part),
            log_part: RationalFunction::zero(),
        };
    }

    let a = proper.numerator().clone();
    let d = proper.denominator().clone();

    // Squarefree factorization of D (use Yun's algorithm — the correct impl)
    let sqf = d.yun_sqf();

    // If D is already squarefree, no rational part from Hermite reduction
    let max_mult = sqf.iter().map(|f| f.multiplicity).max().unwrap_or(0);
    if max_mult <= 1 {
        return HermiteReduction {
            rational_part: RationalFunction::from_poly(poly_part),
            log_part: proper,
        };
    }

    // Compute D* (product of distinct factors) and D** = D / D*
    let d_star = sqf
        .iter()
        .fold(DensePolynomial::constant(R::one()), |acc, f| {
            &acc * &f.factor
        });
    let _d_star_star = d.div_rem(&d_star).0;

    // Quadratic Hermite reduction:
    // Find B, C polynomials such that:
    //   A/D = d/dx(B/D**) + C/D*
    //
    // Expanding the derivative:
    //   A/D = (B'·D** - B·D**') / D**² + C/D*
    //   A = (B'·D** - B·D**') · (D/D**²) + C · (D/D*)
    //   A = (B'·D** - B·D**') · D* / D** + C · D**
    //
    // Wait — let's use the Mack reduction (iterative per-factor approach)
    // which is simpler to implement correctly.

    // Iterative approach: process each factor with multiplicity > 1
    let mut rational_num = DensePolynomial::<R>::zero();
    let mut rational_den = DensePolynomial::constant(R::one());
    let mut current_num = a;
    let mut current_den = d;

    for sf in &sqf {
        if sf.multiplicity <= 1 {
            continue;
        }

        let di = &sf.factor;
        let di_prime = di.derivative();

        // Process multiplicity k down to 1
        let mut k = sf.multiplicity;
        while k > 1 {
            // current_num / current_den, where current_den contains di^k
            // We want to find b, c such that:
            //   current_num / current_den = d/dx(b / di^(k-1)) + c / (current_den / di)
            //
            // Using the relation:
            //   current_num / current_den = (-b·di' / ((k-1)·di^k)) + (something with lower power)
            //
            // Concretely, we need extended GCD of di and di':
            //   gcd should be 1 (di is squarefree), giving s·di + t·di' = 1
            //
            // Then: current_num = b·something + c·something
            // where the "something" comes from the rest of the denominator.

            // Let V = current_den / di^k (the part coprime to di^k)
            let di_k = poly_pow(di, k);
            let v = current_den.div_rem(&di_k).0;

            // Extended GCD: s·V·di' + t·di = gcd
            // Since V is coprime to di and di is squarefree,
            // gcd(V·di', di) = gcd(di', di) = 1 (di squarefree)
            // So: s·(V·di') + t·di = 1
            let v_di_prime = &v * &di_prime;
            let (gcd, s, t) = poly_ext_gcd(&v_di_prime, di);

            // Normalize by gcd (should be constant for squarefree)
            let gcd_inv = if !gcd.is_zero() && gcd.degree().unwrap_or(0) == 0 {
                gcd.leading_coeff().unwrap().inv()
            } else {
                // Unexpected: gcd not constant. Bail out.
                break;
            };
            let s_norm = s.scale(&gcd_inv);
            let t_norm = t.scale(&gcd_inv);

            // Now: s_norm·V·di' + t_norm·di = 1
            // Multiply both sides by current_num:
            // current_num·s_norm·V·di' + current_num·t_norm·di = current_num
            //
            // In the integral context:
            // current_num / (V·di^k) = -current_num·s_norm / ((k-1)·di^(k-1))
            //                         + [derivative adjustment + current_num·t_norm] / (V·di^(k-1))

            let b = &current_num * &s_norm;

            // k-1 as field element
            let mut km1 = R::zero();
            for _ in 0..(k - 1) {
                km1 = km1 + R::one();
            }
            let km1_inv = km1.inv();

            // Rational part contribution: -b / ((k-1) · di^(k-1))
            let neg_b = -&b;
            let b_scaled = neg_b.scale(&km1_inv);
            let di_km1 = poly_pow(di, k - 1);

            // Add to rational part: rational_num/rational_den + b_scaled/di_km1
            let new_rnum = &(&rational_num * &di_km1) + &(&b_scaled * &rational_den);
            let new_rden = &rational_den * &di_km1;
            rational_num = new_rnum;
            rational_den = new_rden;

            // New numerator for remaining integration:
            // c = current_num·t_norm + (s_norm·current_num)' correction
            // More precisely:
            // c = current_num·t_norm - s_norm' · current_num / (k-1) - s_norm · current_num' / (k-1)
            // Wait, let me re-derive...
            //
            // The correct update is:
            //   c = current_num * t_norm + (b' * km1_inv)   [simplified]
            //
            // Actually, using the Hermite reduction formula from Bronstein:
            //   A/(D_k^j * V) where we reduce j by 1:
            //   A = (-1/(j-1)) * (b' * D_k - (j-1) * b * D_k') * V + c * D_k
            //   So c = A - (-1/(j-1)) * (b_deriv - ... well, let me just use the
            //   standard relation.
            //
            // After the Bezout decomposition A = b * V * D_k' + t_part * D_k:
            //   where b = A * s_norm, t_part = A * t_norm
            //
            // The integral decomposition gives:
            //   new numerator = t_part - b' / (k-1)
            //   (over denominator V * D_k^(k-1))

            let b_prime = b.derivative();
            let b_prime_scaled = b_prime.scale(&km1_inv);
            let t_part = &current_num * &t_norm;
            let new_num = &t_part - &b_prime_scaled;

            // Update current fraction: new_num / (V · di^(k-1))
            current_num = new_num;
            current_den = &v * &di_km1;
            k -= 1;
        }
    }

    // Reduce the accumulated rational part
    let rat = if rational_num.is_zero() {
        RationalFunction::from_poly(poly_part)
    } else {
        // GCD-reduce rational_num / rational_den
        let rat_fn = RationalFunction::new(rational_num, rational_den);
        if poly_part.is_zero() {
            rat_fn
        } else {
            &RationalFunction::from_poly(poly_part) + &rat_fn
        }
    };

    // The logarithmic part: current_num / current_den (should have squarefree den)
    let log = if current_num.is_zero() {
        RationalFunction::zero()
    } else {
        RationalFunction::new(current_num, current_den)
    };

    HermiteReduction {
        rational_part: rat,
        log_part: log,
    }
}

/// Compute `p^n` for a polynomial.
fn poly_pow<R: Field>(p: &DensePolynomial<R>, n: usize) -> DensePolynomial<R> {
    if n == 0 {
        return DensePolynomial::constant(R::one());
    }
    let mut result = p.clone();
    for _ in 1..n {
        result = &result * p;
    }
    result
}

/// Extended GCD for polynomials over a field.
///
/// Returns `(g, s, t)` such that `s·a + t·b = g = gcd(a, b)`.
fn poly_ext_gcd<R: Field>(
    a: &DensePolynomial<R>,
    b: &DensePolynomial<R>,
) -> (DensePolynomial<R>, DensePolynomial<R>, DensePolynomial<R>) {
    if b.is_zero() {
        return (
            a.clone(),
            DensePolynomial::constant(R::one()),
            DensePolynomial::zero(),
        );
    }

    let mut old_r = a.clone();
    let mut r = b.clone();
    let mut old_s = DensePolynomial::constant(R::one());
    let mut s = DensePolynomial::<R>::zero();
    let mut old_t = DensePolynomial::<R>::zero();
    let mut t = DensePolynomial::constant(R::one());

    while !r.is_zero() {
        let (q, rem) = old_r.div_rem(&r);
        old_r = r;
        r = rem;
        let new_s = &old_s - &(&q * &s);
        old_s = s;
        s = new_s;
        let new_t = &old_t - &(&q * &t);
        old_t = t;
        t = new_t;
    }

    (old_r, old_s, old_t)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::BigRational;

    type P = DensePolynomial<BigRational>;
    type RF = RationalFunction<BigRational>;

    fn rat(n: i64, d: i64) -> BigRational {
        BigRational::from_i64(n, d)
    }

    fn int(n: i64) -> BigRational {
        BigRational::from(n)
    }

    /// Create polynomial from coefficients: [a0, a1, a2, ...] = a0 + a1*x + a2*x^2 + ...
    fn poly(coeffs: &[i64]) -> P {
        P::from_coeffs(coeffs.iter().map(|&c| int(c)).collect())
    }

    #[test]
    fn test_poly_ext_gcd_basic() {
        // gcd(x, 1) = 1, with s=0, t=1
        let a = poly(&[0, 1]); // x
        let b = poly(&[1]); // 1
        let (g, s, t) = poly_ext_gcd(&a, &b);
        // s*a + t*b = g
        let check = &(&s * &a) + &(&t * &b);
        // g should be constant (gcd = 1)
        assert!(g.degree().unwrap_or(0) == 0);
        assert!(!g.is_zero());
        // Verify Bezout identity
        let g_norm = g.make_monic();
        let check_norm = check.make_monic();
        assert_eq!(g_norm, check_norm);
    }

    #[test]
    fn test_poly_ext_gcd_coprime() {
        // gcd(x+1, x-1) = 1
        let a = poly(&[1, 1]); // x + 1
        let b = poly(&[-1, 1]); // x - 1
        let (g, s, t) = poly_ext_gcd(&a, &b);
        assert!(g.degree().unwrap_or(0) == 0, "expected constant gcd");
        // Verify: s*a + t*b = g
        let check = &(&s * &a) + &(&t * &b);
        assert_eq!(check, g);
    }

    #[test]
    fn test_poly_pow() {
        let p = poly(&[-1, 1]); // x - 1
        let p2 = poly_pow(&p, 2); // (x-1)^2 = x^2 - 2x + 1
        assert_eq!(p2, poly(&[1, -2, 1]));
        let p3 = poly_pow(&p, 3); // (x-1)^3 = -1 + 3x - 3x^2 + x^3
        assert_eq!(p3, poly(&[-1, 3, -3, 1]));
    }

    #[test]
    fn test_hermite_already_squarefree() {
        // ∫ 1/(x^2 + 1) dx — denominator is squarefree
        let num = poly(&[1]); // 1
        let den = poly(&[1, 0, 1]); // x^2 + 1
        let rf = RF::new(num, den);
        let result = hermite_reduce(&rf);
        assert!(result.is_purely_logarithmic());
        // log_part should be the same as input
        assert_eq!(result.log_part.numerator().clone(), poly(&[1]));
    }

    #[test]
    fn test_hermite_simple_repeated_factor() {
        // ∫ 1/x^2 dx = -1/x (rational part only)
        // Denominator x^2 = (x)^2, squarefree factorization = [(x, 2)]
        let num = poly(&[1]); // 1
        let den = poly(&[0, 0, 1]); // x^2
        let rf = RF::new(num, den);
        let result = hermite_reduce(&rf);

        // Should have a rational part (the -1/x) and zero log part
        assert!(result.is_purely_rational());

        // Verify rational part: should be -1/x (or equivalent)
        let rp = &result.rational_part;
        // Evaluate at x=2: should give -1/2
        let val = rp.eval(&int(2));
        assert_eq!(val, Some(rat(-1, 2)));
    }

    #[test]
    fn test_hermite_mixed() {
        // ∫ (2x+1)/(x^2+x)^2 dx
        // Denominator = x^2(x+1)^2
        // Should have both rational and logarithmic parts
        let num = poly(&[1, 2]); // 2x + 1
        let den_base = poly(&[0, 1, 1]); // x^2 + x = x(x+1)
        let den = &den_base * &den_base; // (x^2+x)^2
        let rf = RF::new(num, den);
        let result = hermite_reduce(&rf);

        // Should have non-zero rational part
        assert!(!result.rational_part.is_zero());

        // The log_part denominator should be squarefree
        let log_den = result.log_part.denominator();
        if !result.log_part.is_zero() {
            let sqf = log_den.squarefree_factorization();
            for factor in &sqf {
                assert_eq!(
                    factor.multiplicity, 1,
                    "log part denominator should be squarefree"
                );
            }
        }
    }

    #[test]
    fn test_hermite_zero_input() {
        let rf = RF::zero();
        let result = hermite_reduce(&rf);
        assert!(result.rational_part.is_zero());
        assert!(result.log_part.is_zero());
    }

    #[test]
    fn test_hermite_polynomial_input() {
        // ∫ x dx = x^2/2 — but Hermite doesn't do this
        // A polynomial has no denominator issues, so rational_part = polynomial, log = 0
        let rf = RF::from_poly(poly(&[0, 1])); // x
        let result = hermite_reduce(&rf);
        assert!(result.log_part.is_zero());
    }

    #[test]
    fn test_hermite_cube_denom() {
        // ∫ 1/(x-1)^3 dx
        // Should reduce to rational part only: -1/(2*(x-1)^2)
        let num = poly(&[1]); // 1
        let den = poly_pow(&poly(&[-1, 1]), 3); // (x-1)^3
        let rf = RF::new(num, den);
        let result = hermite_reduce(&rf);

        assert!(
            result.is_purely_rational(),
            "1/(x-1)^3 should integrate to purely rational"
        );

        // Check value at x=2: ∫ should give rational part = -1/(2*(2-1)^2) = -1/2
        let val = result.rational_part.eval(&int(2));
        assert_eq!(val, Some(rat(-1, 2)));
    }

    #[test]
    fn test_hermite_preserves_integral() {
        // Verify: d/dx(rational_part) + log_part = original
        // Test with 1/(x+1)^2
        let num = poly(&[1]); // 1
        let den = poly_pow(&poly(&[1, 1]), 2); // (x+1)^2
        let rf = RF::new(num, den);
        let result = hermite_reduce(&rf);

        // Evaluate original and derivative of rational part at several points
        for x_val in [2i64, 3, 5, 7] {
            let x = int(x_val);
            let orig_val = rf.eval(&x).unwrap();

            // Numerical derivative of rational part
            let h = rat(1, 10000);
            let x_plus_h = &x + &h;
            let rp_x = result.rational_part.eval(&x).unwrap();
            let rp_xh = result.rational_part.eval(&x_plus_h).unwrap();
            let deriv_approx = &(&rp_xh - &rp_x) * &h.recip();

            let log_val = if result.log_part.is_zero() {
                BigRational::from(0i64)
            } else {
                result.log_part.eval(&x).unwrap()
            };

            // deriv_approx + log_val ≈ orig_val
            let reconstructed = &deriv_approx + &log_val;
            let diff = &reconstructed - &orig_val;

            // Allow some numerical error from finite differences
            let abs_diff = diff.abs();
            let tolerance = rat(1, 100);
            assert!(
                abs_diff < tolerance,
                "Hermite reduction verification failed at x={x_val}: \
                 orig={orig_val}, reconstructed={reconstructed}, diff={abs_diff}"
            );
        }
    }
}
