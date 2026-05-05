//! Modular polynomial GCD using Chinese Remainder Theorem.
//!
//! For polynomials with large integer coefficients, computes GCD modulo
//! several primes and reconstructs the result via CRT. Much faster than
//! subresultant PRS when coefficients are large.
//!
//! # Algorithm (simplified)
//!
//! 1. Pick prime `p` not dividing `lc(f)` or `lc(g)`
//! 2. Reduce `f, g` mod `p` → polynomials over Z_p
//! 3. Compute GCD in Z_p[x] (Euclidean, since Z_p is a field)
//! 4. Repeat for more primes, reconstruct coefficients via CRT
//! 5. Verify result divides both inputs

use super::dense_poly::DensePolynomial;
use super::number_theory::{crt, mod_inverse};
use super::BigRational;
use super::SmallInt;
use num::traits::{One, Signed, Zero};

/// Good primes for modular GCD (large enough to reduce collisions).
const PRIMES: [i64; 20] = [
    65537, 65539, 65543, 65551, 65557, 65563, 65579, 65581, 65587, 65599, 65609, 65617, 65629,
    65633, 65647, 65651, 65657, 65677, 65687, 65699,
];

/// Compute GCD of two integer polynomials using modular approach.
///
/// Both polynomials must have integer coefficients (BigRational with denom=1).
/// Falls back to subresultant PRS if modular approach fails.
pub fn modular_gcd(
    f: &DensePolynomial<BigRational>,
    g: &DensePolynomial<BigRational>,
) -> DensePolynomial<BigRational> {
    // Handle trivial cases
    if f.is_zero() {
        return g.clone();
    }
    if g.is_zero() {
        return f.clone();
    }

    // Ensure integer coefficients
    if !is_integer_poly(f) || !is_integer_poly(g) {
        // Fall back to Euclidean GCD for rational polynomials
        return f.gcd(g);
    }

    let lc_f = f.leading_coeff().unwrap().numer().clone();
    let lc_g = g.leading_coeff().unwrap().numer().clone();

    // Content and primitive parts
    let content_f = integer_content(f);
    let content_g = integer_content(g);
    let content_gcd = content_f.gcd(&content_g);

    let prim_f = scale_integer_poly(f, &content_f);
    let prim_g = scale_integer_poly(g, &content_g);

    // Collect modular GCDs
    let mut mod_gcds: Vec<(SmallInt, Vec<SmallInt>)> = Vec::new(); // (prime, gcd_coeffs)
    let mut target_deg: Option<usize> = None;

    for &p in &PRIMES {
        let sp = SmallInt::from(p);

        // Skip primes dividing leading coefficients
        if (&lc_f % &sp).is_zero() || (&lc_g % &sp).is_zero() {
            continue;
        }

        // Reduce mod p
        let fp = reduce_mod_p(&prim_f, p);
        let gp = reduce_mod_p(&prim_g, p);

        if fp.is_empty() || gp.is_empty() {
            continue;
        }

        // GCD in Z_p[x] using Euclidean algorithm
        let gcd_p_i64 = euclidean_gcd_mod_p(&fp, &gp, p);
        let gcd_p: Vec<SmallInt> = gcd_p_i64.iter().map(|&c| SmallInt::from(c)).collect();

        let deg = gcd_p.len().saturating_sub(1);

        match target_deg {
            None => {
                target_deg = Some(deg);
                mod_gcds.push((sp, gcd_p));
            }
            Some(td) => {
                if deg < td {
                    // Unlucky prime previously — restart with this degree
                    target_deg = Some(deg);
                    mod_gcds.clear();
                    mod_gcds.push((sp, gcd_p));
                } else if deg == td {
                    mod_gcds.push((sp, gcd_p));
                }
                // deg > td → this prime is unlucky, skip
            }
        }

        // Try to reconstruct after collecting enough primes
        if mod_gcds.len() >= 3 {
            if let Some(result) = try_reconstruct(&mod_gcds, &content_gcd, &prim_f, &prim_g) {
                return result;
            }
        }
    }

    // Try reconstruct with what we have
    if !mod_gcds.is_empty() {
        if let Some(result) = try_reconstruct(&mod_gcds, &content_gcd, &prim_f, &prim_g) {
            return result;
        }
    }

    // Fallback to subresultant PRS
    f.gcd(g)
}

/// Check if all coefficients are integers (denom = 1).
fn is_integer_poly(p: &DensePolynomial<BigRational>) -> bool {
    p.coefficients().iter().all(|c| c.is_integer())
}

/// Compute integer content (GCD of all integer coefficients).
fn integer_content(p: &DensePolynomial<BigRational>) -> SmallInt {
    let mut g = SmallInt::from(0i64);
    for c in p.coefficients() {
        g = g.gcd(&c.numer().abs());
    }
    if g.is_zero() {
        SmallInt::from(1i64)
    } else {
        g
    }
}

/// Divide all coefficients by the content.
fn scale_integer_poly(
    p: &DensePolynomial<BigRational>,
    content: &SmallInt,
) -> DensePolynomial<BigRational> {
    if content.is_one() {
        return p.clone();
    }
    let divisor = BigRational::from_integer(content.clone());
    let coeffs: Vec<BigRational> = p.coefficients().iter().map(|c| c / &divisor).collect();
    DensePolynomial::from_coeffs(coeffs)
}

/// Reduce polynomial coefficients mod p, returning Vec<i64>.
fn reduce_mod_p(p: &DensePolynomial<BigRational>, prime: i64) -> Vec<i64> {
    let sp = SmallInt::from(prime);
    let mut coeffs: Vec<i64> = p
        .coefficients()
        .iter()
        .map(|c| {
            let n = c.numer();
            let mut r = (n % &sp).to_i64().unwrap_or(0);
            if r < 0 {
                r += prime;
            }
            r
        })
        .collect();
    // Trim trailing zeros
    while coeffs.last() == Some(&0) && coeffs.len() > 1 {
        coeffs.pop();
    }
    if coeffs == [0] {
        coeffs.clear();
    }
    coeffs
}

/// Euclidean GCD of two polynomials over Z_p (all arithmetic mod p).
fn euclidean_gcd_mod_p(a: &[i64], b: &[i64], p: i64) -> Vec<i64> {
    if b.is_empty() {
        return make_monic_mod_p(a, p);
    }

    let mut r0 = a.to_vec();
    let mut r1 = b.to_vec();

    while !r1.is_empty() {
        let rem = poly_rem_mod_p(&r0, &r1, p);
        r0 = r1;
        r1 = rem;
    }

    make_monic_mod_p(&r0, p)
}

/// Polynomial remainder mod p.
fn poly_rem_mod_p(a: &[i64], b: &[i64], p: i64) -> Vec<i64> {
    if a.len() < b.len() {
        return a.to_vec();
    }
    if b.is_empty() {
        return vec![];
    }

    let mut r = a.to_vec();
    let b_lc_inv = mod_inv_i64(*b.last().unwrap(), p);

    while r.len() >= b.len() && !is_zero_poly(&r) {
        let coeff = (r[r.len() - 1] * b_lc_inv) % p;
        let shift = r.len() - b.len();
        for (i, &bc) in b.iter().enumerate() {
            r[shift + i] = ((r[shift + i] - coeff * bc) % p + p) % p;
        }
        // Trim leading zeros
        while r.last() == Some(&0) && r.len() > 1 {
            r.pop();
        }
        if r == [0] {
            r.clear();
            break;
        }
    }

    r
}

/// Make polynomial monic mod p.
fn make_monic_mod_p(a: &[i64], p: i64) -> Vec<i64> {
    if a.is_empty() {
        return vec![];
    }
    let lc = *a.last().unwrap();
    if lc == 0 {
        return vec![];
    }
    let lc_inv = mod_inv_i64(lc, p);
    a.iter().map(|&c| (c * lc_inv % p + p) % p).collect()
}

/// Modular inverse for i64.
fn mod_inv_i64(a: i64, p: i64) -> i64 {
    let si_a = SmallInt::from(a);
    let si_p = SmallInt::from(p);
    mod_inverse(&si_a, &si_p)
        .and_then(|inv| inv.to_i64())
        .unwrap_or(1)
}

fn is_zero_poly(a: &[i64]) -> bool {
    a.is_empty() || a.iter().all(|&c| c == 0)
}

/// Try to reconstruct the GCD from modular images using CRT.
fn try_reconstruct(
    mod_gcds: &[(SmallInt, Vec<SmallInt>)],
    content_gcd: &SmallInt,
    prim_f: &DensePolynomial<BigRational>,
    prim_g: &DensePolynomial<BigRational>,
) -> Option<DensePolynomial<BigRational>> {
    if mod_gcds.is_empty() {
        return None;
    }

    let deg = mod_gcds[0].1.len();
    if deg == 0 {
        return None;
    }

    // GCD degree 0 means coprime
    if deg == 1 && mod_gcds[0].1 == vec![SmallInt::from(1i64)] {
        return Some(DensePolynomial::constant(BigRational::from_integer(
            content_gcd.clone(),
        )));
    }

    let primes: Vec<SmallInt> = mod_gcds.iter().map(|(p, _)| p.clone()).collect();

    // CRT each coefficient
    let mut coeffs = Vec::with_capacity(deg);
    for i in 0..deg {
        let residues: Vec<SmallInt> = mod_gcds
            .iter()
            .map(|(_, gc)| gc.get(i).cloned().unwrap_or_else(|| SmallInt::from(0i64)))
            .collect();
        let coeff = crt(&residues, &primes)?;

        // Convert to symmetric representation: if coeff > M/2, subtract M
        let big_m: SmallInt = primes.iter().fold(SmallInt::from(1i64), |a, b| &a * b);
        let half_m = &big_m / &SmallInt::from(2i64);
        let sym_coeff = if coeff > half_m {
            &coeff - &big_m
        } else {
            coeff
        };

        coeffs.push(BigRational::from_integer(sym_coeff));
    }

    let candidate = DensePolynomial::from_coeffs(coeffs);
    if candidate.is_zero() {
        return None;
    }

    // Scale by content GCD
    let scaled = if content_gcd.is_one() {
        candidate
    } else {
        candidate.scale(&BigRational::from_integer(content_gcd.clone()))
    };

    // Make monic for verification
    let monic = scaled.make_monic();

    // Verify: monic must divide both prim_f and prim_g
    let (_, rem_f) = prim_f.div_rem(&monic);
    if !rem_f.is_zero() {
        return None;
    }
    let (_, rem_g) = prim_g.div_rem(&monic);
    if !rem_g.is_zero() {
        return None;
    }

    // Restore content
    if content_gcd.is_one() {
        Some(monic)
    } else {
        Some(monic.scale(&BigRational::from_integer(content_gcd.clone())))
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    type P = DensePolynomial<BigRational>;

    fn int(n: i64) -> BigRational {
        BigRational::from(n)
    }

    fn poly(coeffs: &[i64]) -> P {
        P::from_coeffs(coeffs.iter().map(|&c| int(c)).collect())
    }

    #[test]
    fn test_modular_gcd_coprime() {
        // gcd(x+1, x-1) = 1
        let f = poly(&[1, 1]); // x + 1
        let g = poly(&[-1, 1]); // x - 1
        let gcd = modular_gcd(&f, &g);
        assert_eq!(gcd.degree(), Some(0));
    }

    #[test]
    fn test_modular_gcd_common_factor() {
        // gcd(x^2-1, x^2-2x+1) = gcd((x-1)(x+1), (x-1)^2) = x-1
        let f = poly(&[-1, 0, 1]); // x^2 - 1
        let g = poly(&[1, -2, 1]); // x^2 - 2x + 1
        let gcd = modular_gcd(&f, &g);
        // Should be x-1 (monic)
        assert_eq!(gcd.degree(), Some(1));
        // Verify it divides both
        assert!(f.div_rem(&gcd).1.is_zero());
        assert!(g.div_rem(&gcd).1.is_zero());
    }

    #[test]
    fn test_modular_gcd_same_poly() {
        let f = poly(&[1, -2, 1]); // (x-1)^2
        let gcd = modular_gcd(&f, &f);
        // GCD of f with itself is f (monic)
        assert_eq!(gcd.degree(), f.degree());
    }

    #[test]
    fn test_modular_gcd_zero() {
        let f = poly(&[1, 1]);
        let z = P::zero();
        assert_eq!(modular_gcd(&f, &z).degree(), f.degree());
        assert_eq!(modular_gcd(&z, &f).degree(), f.degree());
    }

    #[test]
    fn test_modular_gcd_large_coefficients() {
        // Polynomials with large coefficients where subresultant would suffer
        // f = 1000000*x^2 + 999999*x + 999998
        // g = 1000000*x + 999999
        // gcd should divide both
        let f = poly(&[999998, 999999, 1000000]);
        let g = poly(&[999999, 1000000]);
        let gcd = modular_gcd(&f, &g);
        assert!(f.div_rem(&gcd).1.is_zero());
        assert!(g.div_rem(&gcd).1.is_zero());
    }

    #[test]
    fn test_modular_gcd_matches_euclidean() {
        // Compare modular GCD with Euclidean GCD for several cases
        let cases: Vec<(P, P)> = vec![
            (poly(&[6, 7, 1]), poly(&[6, 5, 1])), // (x+1)(x+6), (x+2)(x+3)
            (poly(&[0, 0, 1]), poly(&[0, 1])),    // x^2, x
            (poly(&[-1, 0, 0, 1]), poly(&[-1, 0, 1])), // x^3-1, x^2-1
        ];

        for (f, g) in cases {
            let mod_gcd = modular_gcd(&f, &g);
            let euc_gcd = f.gcd(&g);
            // Both should be monic and have same degree
            assert_eq!(
                mod_gcd.degree(),
                euc_gcd.degree(),
                "degree mismatch for gcd({f}, {g})"
            );
        }
    }

    #[test]
    fn test_euclidean_gcd_mod_p() {
        // gcd(x^2-1, x-1) mod 65537 = x-1
        let a = vec![65536, 0, 1]; // x^2 - 1 mod 65537 (= x^2 + 65536)
        let b = vec![65536, 1]; // x - 1 mod 65537
        let g = euclidean_gcd_mod_p(&a, &b, 65537);
        // Should be monic degree 1
        assert_eq!(g.len(), 2);
        assert_eq!(g[1], 1); // monic
    }

    #[test]
    fn test_reduce_mod_p() {
        let p = poly(&[10, -3, 7]);
        let reduced = reduce_mod_p(&p, 7);
        // 10%7=3, -3%7=4, 7%7=0 → trailing zero trimmed → [3, 4]
        assert_eq!(reduced, vec![3, 4]);
    }
}
