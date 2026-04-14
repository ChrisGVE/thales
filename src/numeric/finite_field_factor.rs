//! Polynomial factorization over finite fields GF(p).
//!
//! Implements Cantor-Zassenhaus algorithm:
//! 1. Distinct-degree factorization (DDF)
//! 2. Equal-degree splitting (EDS)
//!
//! All polynomial arithmetic is done with `i64` coefficients modulo a prime `p`.

use super::number_theory::mod_inverse;
use super::SmallInt;

/// A polynomial over GF(p), stored as coefficients `[a0, a1, ..., an]`.
/// Coefficients are in `[0, p)`. Empty vec represents zero.
type PolyMod = Vec<i64>;

/// Result of factoring a polynomial over GF(p).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteFieldFactors {
    /// The prime field characteristic.
    pub prime: i64,
    /// Irreducible factors (each monic) with multiplicities.
    pub factors: Vec<(PolyMod, usize)>,
}

// ── Polynomial arithmetic mod p ─────────────────────────────────────────────

fn is_zero(a: &PolyMod) -> bool {
    a.is_empty() || a.iter().all(|&c| c == 0)
}

fn degree(a: &PolyMod) -> Option<usize> {
    if is_zero(a) {
        None
    } else {
        Some(a.len() - 1)
    }
}

fn trim(a: &mut PolyMod) {
    while a.last() == Some(&0) {
        a.pop();
    }
}

fn add_mod(a: &PolyMod, b: &PolyMod, p: i64) -> PolyMod {
    let len = a.len().max(b.len());
    let mut result = vec![0i64; len];
    for (i, r) in result.iter_mut().enumerate() {
        let ai = a.get(i).copied().unwrap_or(0);
        let bi = b.get(i).copied().unwrap_or(0);
        *r = (ai + bi) % p;
    }
    trim(&mut result);
    result
}

fn sub_mod(a: &PolyMod, b: &PolyMod, p: i64) -> PolyMod {
    let len = a.len().max(b.len());
    let mut result = vec![0i64; len];
    for (i, r) in result.iter_mut().enumerate() {
        let ai = a.get(i).copied().unwrap_or(0);
        let bi = b.get(i).copied().unwrap_or(0);
        *r = ((ai - bi) % p + p) % p;
    }
    trim(&mut result);
    result
}

fn mul_mod(a: &PolyMod, b: &PolyMod, p: i64) -> PolyMod {
    if is_zero(a) || is_zero(b) {
        return vec![];
    }
    let mut result = vec![0i64; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] = (result[i + j] + ai * bj) % p;
        }
    }
    trim(&mut result);
    result
}

fn scale_mod(a: &PolyMod, c: i64, p: i64) -> PolyMod {
    let mut result: Vec<i64> = a.iter().map(|&x| (x * c) % p).collect();
    trim(&mut result);
    result
}

/// Polynomial remainder: a mod b (in GF(p)[x]).
fn rem_mod(a: &PolyMod, b: &PolyMod, p: i64) -> PolyMod {
    if is_zero(b) {
        panic!("division by zero polynomial");
    }
    if is_zero(a) || a.len() < b.len() {
        return a.clone();
    }

    let b_lc_inv = mod_inv_i64(*b.last().unwrap(), p);
    let mut r = a.clone();

    while !is_zero(&r) && r.len() >= b.len() {
        let coeff = (r[r.len() - 1] * b_lc_inv) % p;
        let shift = r.len() - b.len();
        for (i, &bc) in b.iter().enumerate() {
            r[shift + i] = ((r[shift + i] - coeff * bc) % p + p) % p;
        }
        trim(&mut r);
    }

    r
}

/// GCD in GF(p)[x].
fn gcd_mod(a: &PolyMod, b: &PolyMod, p: i64) -> PolyMod {
    let mut r0 = a.clone();
    let mut r1 = b.clone();
    while !is_zero(&r1) {
        let rem = rem_mod(&r0, &r1, p);
        r0 = r1;
        r1 = rem;
    }
    make_monic(&r0, p)
}

fn make_monic(a: &PolyMod, p: i64) -> PolyMod {
    if is_zero(a) {
        return vec![];
    }
    let lc_inv = mod_inv_i64(*a.last().unwrap(), p);
    scale_mod(a, lc_inv, p)
}

/// Modular inverse for i64.
fn mod_inv_i64(a: i64, p: i64) -> i64 {
    mod_inverse(&SmallInt::from(a), &SmallInt::from(p))
        .and_then(|inv| inv.to_i64())
        .unwrap_or(1)
}

/// Compute `base^exp mod modulus` in GF(p)[x].
/// (Polynomial exponentiation modulo another polynomial.)
fn pow_poly_mod(base: &PolyMod, exp: i64, modulus: &PolyMod, p: i64) -> PolyMod {
    if exp == 0 {
        return vec![1];
    }

    let mut result = vec![1i64]; // 1
    let mut b = rem_mod(base, modulus, p);
    let mut e = exp;

    while e > 0 {
        if e % 2 == 1 {
            result = mul_mod(&result, &b, p);
            result = rem_mod(&result, modulus, p);
        }
        b = mul_mod(&b, &b, p);
        b = rem_mod(&b, modulus, p);
        e /= 2;
    }

    result
}

// ── Distinct-degree factorization ───────────────────────────────────────────

/// Distinct-degree factorization of a squarefree polynomial over GF(p).
///
/// Returns pairs `(factor_product, degree)` where `factor_product` is the
/// product of all irreducible factors of the given degree.
fn distinct_degree_factorization(f: &PolyMod, p: i64) -> Vec<(PolyMod, usize)> {
    let mut result = Vec::new();
    let mut f_star = f.clone();
    let n = degree(&f_star).unwrap_or(0);

    // h = x
    let mut h = vec![0, 1]; // x

    for d in 1..=n / 2 {
        // h = h^p mod f_star (Frobenius)
        h = pow_poly_mod(&h, p, &f_star, p);

        // g = gcd(h - x, f_star)
        let h_minus_x = sub_mod(&h, &vec![0, 1], p);
        let g = gcd_mod(&h_minus_x, &f_star, p);

        if !is_zero(&g) && degree(&g).unwrap_or(0) > 0 {
            result.push((g.clone(), d));
            // f_star = f_star / g
            let (q, _) = div_mod(&f_star, &g, p);
            f_star = q;
            // h = h mod f_star
            h = rem_mod(&h, &f_star, p);
        }

        if degree(&f_star).unwrap_or(0) == 0 {
            break;
        }
    }

    // Remaining f_star is irreducible (if degree > 0)
    if degree(&f_star).unwrap_or(0) > 0 {
        let d = degree(&f_star).unwrap();
        result.push((f_star, d));
    }

    result
}

/// Polynomial division in GF(p)[x]. Returns (quotient, remainder).
fn div_mod(a: &PolyMod, b: &PolyMod, p: i64) -> (PolyMod, PolyMod) {
    if is_zero(b) {
        panic!("division by zero");
    }
    if is_zero(a) || a.len() < b.len() {
        return (vec![], a.clone());
    }

    let b_lc_inv = mod_inv_i64(*b.last().unwrap(), p);
    let mut r = a.clone();
    let mut q = vec![0i64; a.len() - b.len() + 1];

    while !is_zero(&r) && r.len() >= b.len() {
        let coeff = (r[r.len() - 1] * b_lc_inv) % p;
        let shift = r.len() - b.len();
        q[shift] = coeff;
        for (i, &bc) in b.iter().enumerate() {
            r[shift + i] = ((r[shift + i] - coeff * bc) % p + p) % p;
        }
        trim(&mut r);
    }

    trim(&mut q);
    (q, r)
}

// ── Equal-degree splitting ──────────────────────────────────────────────────

/// Split a product of equal-degree irreducible factors using Cantor-Zassenhaus.
///
/// `f` is a product of irreducible factors all of degree `d` over GF(p).
/// Returns the individual irreducible factors.
fn equal_degree_split(f: &PolyMod, d: usize, p: i64) -> Vec<PolyMod> {
    let n = degree(f).unwrap_or(0);
    if n == 0 {
        return vec![];
    }
    if n == d {
        return vec![f.clone()];
    }

    // Number of factors
    let num_factors = n / d;
    if num_factors <= 1 {
        return vec![f.clone()];
    }

    // Use deterministic attempts with different "random" polynomials
    for seed in 1..100i64 {
        let r = random_poly(seed, n, p);
        let g = if p == 2 {
            // For GF(2): compute r + r^2 + r^4 + ... + r^(2^(d-1))
            trace_map_char2(&r, d, f, p)
        } else {
            // For odd p: compute r^((p^d - 1)/2) - 1 mod f
            let exp = (pow_i64(p, d as u32) - 1) / 2;
            let rp = pow_poly_mod(&r, exp, f, p);
            sub_mod(&rp, &vec![1], p)
        };

        let factor = gcd_mod(&g, f, p);
        let fd = degree(&factor).unwrap_or(0);

        if fd > 0 && fd < n {
            // Found a non-trivial factor — recurse
            let other = div_mod(f, &factor, p).0;
            let mut result = equal_degree_split(&factor, d, p);
            result.extend(equal_degree_split(&other, d, p));
            return result;
        }
    }

    // Failed to split — return as-is (shouldn't happen for reasonable inputs)
    vec![f.clone()]
}

/// Generate a "random" polynomial of degree < n for deterministic splitting.
fn random_poly(seed: i64, n: usize, p: i64) -> PolyMod {
    let mut coeffs = Vec::with_capacity(n);
    let mut val = seed;
    for _ in 0..n {
        coeffs.push(((val % p) + p) % p);
        val = (val * 31 + 17) % (p * 1000); // Simple PRNG
    }
    trim(&mut coeffs);
    if is_zero(&coeffs) {
        coeffs = vec![1]; // Avoid zero polynomial
    }
    coeffs
}

/// Trace map for characteristic 2: T(r) = r + r^2 + r^4 + ... + r^(2^(d-1)).
fn trace_map_char2(r: &PolyMod, d: usize, modulus: &PolyMod, p: i64) -> PolyMod {
    let mut result = r.clone();
    let mut current = r.clone();
    for _ in 1..d {
        current = pow_poly_mod(&current, 2, modulus, p);
        result = add_mod(&result, &current, p);
        result = rem_mod(&result, modulus, p);
    }
    result
}

/// Compute p^d as i64 (may overflow for large values).
fn pow_i64(base: i64, exp: u32) -> i64 {
    let mut result = 1i64;
    for _ in 0..exp {
        result = result.saturating_mul(base);
    }
    result
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Factor a squarefree polynomial over GF(p) into irreducible factors.
///
/// The input polynomial should be squarefree and have coefficients in `[0, p)`.
/// Returns monic irreducible factors.
pub fn factor_over_gfp(f: &[i64], p: i64) -> Vec<PolyMod> {
    let f = make_monic(&f.to_vec(), p);
    if is_zero(&f) || degree(&f).unwrap_or(0) == 0 {
        return vec![];
    }

    // Step 1: Distinct-degree factorization
    let ddf = distinct_degree_factorization(&f, p);

    // Step 2: Equal-degree splitting for each group
    let mut factors = Vec::new();
    for (product, d) in &ddf {
        let split = equal_degree_split(product, *d, p);
        factors.extend(split);
    }

    // Sort for deterministic output
    factors.sort();
    factors
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poly_arithmetic_mod() {
        // (x + 1) + (x + 2) = 2x + 3 mod 5
        let a = vec![1, 1]; // x + 1
        let b = vec![2, 1]; // x + 2
        let sum = add_mod(&a, &b, 5);
        assert_eq!(sum, vec![3, 2]); // 2x + 3
    }

    #[test]
    fn test_poly_mul_mod() {
        // (x + 1)(x + 2) = x^2 + 3x + 2 mod 5
        let a = vec![1, 1];
        let b = vec![2, 1];
        let prod = mul_mod(&a, &b, 5);
        assert_eq!(prod, vec![2, 3, 1]);
    }

    #[test]
    fn test_gcd_mod() {
        // gcd(x^2 - 1, x - 1) mod 7 = x - 1 (monic: x + 6)
        let a = vec![6, 0, 1]; // x^2 - 1 ≡ x^2 + 6 mod 7
        let b = vec![6, 1]; // x - 1 ≡ x + 6 mod 7
        let g = gcd_mod(&a, &b, 7);
        assert_eq!(g, vec![6, 1]); // x + 6 = x - 1 mod 7
    }

    #[test]
    fn test_pow_poly_mod() {
        // x^3 mod (x^2 + 1) over GF(5) = -x = 4x
        let base = vec![0, 1]; // x
        let modulus = vec![1, 0, 1]; // x^2 + 1
        let result = pow_poly_mod(&base, 3, &modulus, 5);
        assert_eq!(result, vec![0, 4]); // 4x ≡ -x mod 5
    }

    #[test]
    fn test_factor_irreducible() {
        // x^2 + 1 is irreducible over GF(3) (no roots: 0^2+1=1, 1^2+1=2, 2^2+1=2)
        let f = vec![1, 0, 1]; // x^2 + 1
        let factors = factor_over_gfp(&f, 3);
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0], vec![1, 0, 1]);
    }

    #[test]
    fn test_factor_splits() {
        // x^2 - 1 = (x-1)(x+1) over GF(5)
        // x^2 - 1 ≡ x^2 + 4 mod 5
        // factors: x+4 (= x-1) and x+1
        let f = vec![4, 0, 1]; // x^2 + 4 = x^2 - 1 mod 5
        let factors = factor_over_gfp(&f, 5);
        assert_eq!(factors.len(), 2);
        // Should be two linear factors
        for factor in &factors {
            assert_eq!(degree(factor), Some(1));
        }
    }

    #[test]
    fn test_factor_linear() {
        // x + 3 over GF(7) is already irreducible
        let f = vec![3, 1];
        let factors = factor_over_gfp(&f, 7);
        assert_eq!(factors.len(), 1);
    }

    #[test]
    fn test_ddf_basic() {
        // x^4 + x + 1 over GF(2) factors into (x^2+x+1) and (x^2+x+1)? No...
        // Actually x^4 + x + 1 over GF(2): check roots: f(0)=1, f(1)=1+1+1=1.
        // No linear factors. Check if irreducible or product of two quadratics.
        // (x^2+x+1)^2 = x^4+x^2+1 ≠ x^4+x+1. So it's either irreducible
        // or factors into two distinct quadratics.
        // x^4+x+1 = (x^2+x+1)(x^2+x+1) mod 2? No.
        // Actually GF(2): x^4+x+1 is irreducible.
        let f = vec![1, 1, 0, 0, 1]; // x^4 + x + 1
        let factors = factor_over_gfp(&f, 2);
        assert_eq!(factors.len(), 1, "x^4+x+1 should be irreducible over GF(2)");
    }

    #[test]
    fn test_factor_product_reconstruction() {
        // Factor x^3 - x = x(x-1)(x+1) over GF(5)
        // = x(x+4)(x+1) mod 5
        let f = vec![0, 4, 0, 1]; // x^3 - x = x^3 + 4x mod 5
        let factors = factor_over_gfp(&f, 5);

        // Should get 3 linear factors
        assert_eq!(factors.len(), 3);

        // Verify product equals original (monic)
        let mut product = vec![1i64];
        for factor in &factors {
            product = mul_mod(&product, factor, 5);
        }
        let monic_f = make_monic(&f, 5);
        assert_eq!(product, monic_f);
    }

    #[test]
    fn test_div_mod() {
        // (x^2 + 2x + 1) / (x + 1) = x + 1, rem 0 over GF(5)
        let a = vec![1, 2, 1];
        let b = vec![1, 1];
        let (q, r) = div_mod(&a, &b, 5);
        assert_eq!(q, vec![1, 1]);
        assert!(is_zero(&r));
    }
}
