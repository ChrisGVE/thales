//! Zassenhaus recombination for integer polynomial factorization.
//!
//! Given Hensel-lifted factors of `f` modulo `p^k`, recombine subsets
//! to find the true integer factors of `f`. Uses exhaustive subset
//! enumeration with coefficient-bound pruning.
//!
//! # Pipeline
//!
//! 1. Squarefree decomposition (poly_factoring)
//! 2. Factor mod p (finite_field_factor)
//! 3. Hensel lift to mod p^k (hensel)
//! 4. **Recombine** lifted factors (this module)

use super::dense_poly::DensePolynomial;
use super::finite_field_factor::factor_over_gfp;
use super::hensel::hensel_lift;
use super::BigRational;
use super::SmallInt;
use num::traits::{Signed, Zero};

/// Factor an integer polynomial into irreducible factors over Z.
///
/// Returns monic irreducible factors. The leading coefficient and
/// content are factored out separately.
///
/// # Algorithm
///
/// 1. Extract content (GCD of coefficients)
/// 2. Make primitive and monic
/// 3. Squarefree factorization
/// 4. For each squarefree factor:
///    a. Choose a prime p not dividing lc
///    b. Factor mod p using Cantor-Zassenhaus
///    c. Hensel lift to mod p^k (k chosen from coefficient bound)
///    d. Recombine lifted factors
pub fn factor_integer_poly(
    f: &DensePolynomial<BigRational>,
) -> Vec<(DensePolynomial<BigRational>, usize)> {
    if f.is_zero() {
        return vec![];
    }

    let deg = match f.degree() {
        Some(d) => d,
        None => return vec![],
    };

    if deg == 0 {
        return vec![];
    }

    // Linear polynomial is irreducible
    if deg == 1 {
        let monic = f.make_monic();
        return vec![(monic, 1)];
    }

    // Squarefree decomposition
    let sqf = f.yun_sqf();
    let mut result = Vec::new();

    for sf in &sqf {
        if sf.factor.degree().unwrap_or(0) == 0 {
            continue;
        }
        if sf.factor.degree() == Some(1) {
            result.push((sf.factor.make_monic(), sf.multiplicity));
            continue;
        }

        let factors = factor_squarefree_integer(&sf.factor);
        for factor in factors {
            result.push((factor, sf.multiplicity));
        }
    }

    result
}

/// Factor a squarefree integer polynomial into irreducible factors.
fn factor_squarefree_integer(
    f: &DensePolynomial<BigRational>,
) -> Vec<DensePolynomial<BigRational>> {
    let deg = f.degree().unwrap_or(0);
    if deg <= 1 {
        return vec![f.make_monic()];
    }

    // Make monic
    let monic = f.make_monic();

    // Choose a good prime
    let p = choose_prime(&monic);

    // Reduce mod p and factor
    let coeffs_mod_p = reduce_to_i64(&monic, p);
    let mod_factors = factor_over_gfp(&coeffs_mod_p, p);

    if mod_factors.len() <= 1 {
        // Irreducible mod p → irreducible over Z
        return vec![monic];
    }

    // Compute coefficient bound for Hensel lifting
    let bound = mignotte_bound(&monic);
    let k = lifting_steps(p, &bound);

    // Hensel lift: lift pairs of factors
    let lifted = lift_all_factors(&coeffs_mod_p, &mod_factors, p, k);

    // Recombine
    recombine(&monic, &lifted, p, k)
}

/// Choose a prime that doesn't divide the leading coefficient
/// and gives a squarefree factorization mod p.
fn choose_prime(f: &DensePolynomial<BigRational>) -> i64 {
    let lc = f.leading_coeff().unwrap().numer().clone();
    let primes = [
        65537i64, 65539, 65543, 65551, 65557, 65563, 65579, 65581, 97, 101, 103, 107, 109, 113,
    ];

    for &p in &primes {
        let sp = SmallInt::from(p);
        if (&lc % &sp).is_zero() {
            continue;
        }

        // Check that f mod p is squarefree
        let fp = reduce_to_i64(f, p);
        let fp_deriv = poly_derivative_mod(&fp, p);
        let gcd = poly_gcd_mod(&fp, &fp_deriv, p);
        if gcd.len() <= 1 {
            return p;
        }
    }

    // Fallback
    65537
}

/// Mignotte bound: upper bound on the absolute value of any coefficient
/// of any factor of `f`.
fn mignotte_bound(f: &DensePolynomial<BigRational>) -> SmallInt {
    let n = f.degree().unwrap_or(0);
    // Bound = (n choose n/2) * ||f||_2
    // Simplified: 2^n * max_coeff
    let max_coeff = f
        .coefficients()
        .iter()
        .map(|c| c.numer().abs())
        .max()
        .unwrap_or_else(|| SmallInt::from(1i64));

    // 2^n * max_coeff
    let two_n = SmallInt::from(2i64).pow(n as u32);
    &two_n * &max_coeff
}

/// Number of Hensel lifting steps needed: smallest k with p^k > 2*bound.
fn lifting_steps(p: i64, bound: &SmallInt) -> u32 {
    let two_bound = &(bound * &SmallInt::from(2i64));
    let mut pk = SmallInt::from(p);
    let mut k = 1u32;
    while &pk <= two_bound && k < 50 {
        pk = &pk * &SmallInt::from(p);
        k += 1;
    }
    k
}

/// Lift all modular factors to mod p^k using sequential Hensel lifting.
fn lift_all_factors(f: &[i64], factors: &[Vec<i64>], p: i64, k: u32) -> Vec<Vec<i64>> {
    if factors.len() <= 1 {
        return factors.to_vec();
    }

    // Build a binary tree of products and lift pairwise
    // Simple approach: lift pairs sequentially
    // f = f1 * f2 * ... * fn mod p
    // Lift f = (f1*f2*...*f_{n/2}) * (f_{n/2+1}*...*fn) etc.

    // For simplicity, lift pairs: f = g * h where g = factors[0], h = product of rest
    let mut lifted = Vec::new();
    let mut remaining = f.to_vec();

    for i in 0..factors.len() - 1 {
        let g0 = &factors[i];
        // h0 = remaining / g0 mod p
        let h0 = poly_div_exact_mod(&remaining, g0, p);

        let lift = hensel_lift(&remaining, g0, &h0, p, k);
        lifted.push(lift.g);
        remaining = lift.h;
    }
    lifted.push(remaining);

    lifted
}

/// Exact polynomial division mod p.
fn poly_div_exact_mod(a: &[i64], b: &[i64], p: i64) -> Vec<i64> {
    if b.is_empty() {
        return vec![];
    }
    if a.len() < b.len() {
        return vec![];
    }

    let b_lc_inv = mod_inv_i64(*b.last().unwrap(), p);
    let mut r = a.to_vec();
    let mut q = vec![0i64; a.len() - b.len() + 1];

    while r.len() >= b.len() {
        let lc = ((*r.last().unwrap() % p) + p) % p;
        if lc == 0 {
            r.pop();
            continue;
        }
        let coeff = (lc * b_lc_inv) % p;
        let shift = r.len() - b.len();
        q[shift] = coeff;
        for (i, &bc) in b.iter().enumerate() {
            r[shift + i] = ((r[shift + i] - coeff * bc) % p + p) % p;
        }
        while r.last() == Some(&0) {
            r.pop();
        }
    }

    while q.last() == Some(&0) && q.len() > 1 {
        q.pop();
    }
    q
}

/// Recombine Hensel-lifted factors into true integer factors.
///
/// Tries all subsets of `lifted` factors of increasing size.
/// A product that divides `f` over Z is a true factor.
fn recombine(
    f: &DensePolynomial<BigRational>,
    lifted: &[Vec<i64>],
    p: i64,
    k: u32,
) -> Vec<DensePolynomial<BigRational>> {
    let n = lifted.len();
    if n == 0 {
        return vec![f.clone()];
    }
    if n == 1 {
        return vec![f.make_monic()];
    }

    let modulus = SmallInt::from(p).pow(k);
    let half_mod = &modulus / &SmallInt::from(2i64);

    let mut factors = Vec::new();
    let mut remaining = f.clone();
    let mut used = vec![false; n];

    // Try subsets of increasing size
    for subset_size in 1..=(n / 2) {
        let mut changed = true;
        while changed {
            changed = false;

            // Generate all subsets of given size from unused factors
            let unused: Vec<usize> = (0..n).filter(|&i| !used[i]).collect();
            if unused.len() < subset_size {
                break;
            }

            for combo in combinations(&unused, subset_size) {
                // Multiply the lifted factors in this combo
                let candidate = multiply_lifted(&combo, lifted, &modulus, &half_mod);

                // Convert to BigRational polynomial
                let candidate_poly = i64_to_bigrational_poly(&candidate);

                // Check if it divides remaining
                if candidate_poly.degree().unwrap_or(0) == 0 {
                    continue;
                }
                let (q, r) = remaining.div_rem(&candidate_poly);
                if r.is_zero() && !q.is_zero() {
                    factors.push(candidate_poly.make_monic());
                    remaining = q;
                    for &idx in &combo {
                        used[idx] = true;
                    }
                    changed = true;
                    break;
                }
            }
        }
    }

    // Any remaining is an irreducible factor
    if remaining.degree().unwrap_or(0) > 0 {
        factors.push(remaining.make_monic());
    }

    factors
}

/// Multiply selected lifted factors mod p^k with centered representation.
fn multiply_lifted(
    indices: &[usize],
    lifted: &[Vec<i64>],
    modulus: &SmallInt,
    half_mod: &SmallInt,
) -> Vec<i64> {
    let mut product = vec![1i64];
    for &idx in indices {
        product = poly_mul_i64(&product, &lifted[idx]);
        // Reduce mod p^k with centered representation
        product = product
            .iter()
            .map(|&c| {
                let si = SmallInt::from(c);
                let r = &si % modulus;
                let r = if &r > half_mod { &r - modulus } else { r };
                r.to_i64().unwrap_or(c)
            })
            .collect();
        trim_i64(&mut product);
    }
    product
}

fn poly_mul_i64(a: &[i64], b: &[i64]) -> Vec<i64> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut result = vec![0i64; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }
    result
}

fn trim_i64(a: &mut Vec<i64>) {
    while a.last() == Some(&0) && a.len() > 1 {
        a.pop();
    }
}

fn i64_to_bigrational_poly(coeffs: &[i64]) -> DensePolynomial<BigRational> {
    DensePolynomial::from_coeffs(coeffs.iter().map(|&c| BigRational::from(c)).collect())
}

/// Generate all combinations of `k` elements from `items`.
fn combinations(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if items.len() < k {
        return vec![];
    }
    if k == 1 {
        return items.iter().map(|&x| vec![x]).collect();
    }

    let mut result = Vec::new();
    for (i, &item) in items.iter().enumerate() {
        let rest = &items[i + 1..];
        for mut combo in combinations(rest, k - 1) {
            combo.insert(0, item);
            result.push(combo);
        }
    }
    result
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn reduce_to_i64(f: &DensePolynomial<BigRational>, p: i64) -> Vec<i64> {
    let sp = SmallInt::from(p);
    let mut coeffs: Vec<i64> = f
        .coefficients()
        .iter()
        .map(|c| {
            let n = c.numer();
            let mut r = (n % &sp).to_i64().unwrap_or(0);
            if r < 0 {
                r += p;
            }
            r
        })
        .collect();
    while coeffs.last() == Some(&0) && coeffs.len() > 1 {
        coeffs.pop();
    }
    coeffs
}

fn poly_derivative_mod(f: &[i64], p: i64) -> Vec<i64> {
    if f.len() <= 1 {
        return vec![];
    }
    let mut result = Vec::with_capacity(f.len() - 1);
    for (i, &c) in f.iter().enumerate().skip(1) {
        result.push((c * (i as i64)) % p);
    }
    while result.last() == Some(&0) && !result.is_empty() {
        result.pop();
    }
    result
}

fn poly_gcd_mod(a: &[i64], b: &[i64], p: i64) -> Vec<i64> {
    let mut r0 = a.to_vec();
    let mut r1 = b.to_vec();
    while !r1.is_empty() {
        let rem = poly_rem_mod(&r0, &r1, p);
        r0 = r1;
        r1 = rem;
    }
    r0
}

fn poly_rem_mod(a: &[i64], b: &[i64], p: i64) -> Vec<i64> {
    if b.is_empty() || a.len() < b.len() {
        return a.to_vec();
    }
    let b_lc_inv = mod_inv_i64(*b.last().unwrap(), p);
    let mut r = a.to_vec();
    while r.len() >= b.len() && !r.iter().all(|&c| c == 0) {
        let coeff = (r[r.len() - 1] * b_lc_inv) % p;
        let shift = r.len() - b.len();
        for (i, &bc) in b.iter().enumerate() {
            r[shift + i] = ((r[shift + i] - coeff * bc) % p + p) % p;
        }
        while r.last() == Some(&0) && !r.is_empty() {
            r.pop();
        }
    }
    r
}

fn mod_inv_i64(a: i64, p: i64) -> i64 {
    use super::number_theory::mod_inverse;
    mod_inverse(&SmallInt::from(a), &SmallInt::from(p))
        .and_then(|inv| inv.to_i64())
        .unwrap_or(1)
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
    fn test_factor_irreducible() {
        // x^2 + 1 is irreducible over Z
        let f = poly(&[1, 0, 1]);
        let factors = factor_integer_poly(&f);
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, 1); // multiplicity 1
    }

    #[test]
    fn test_factor_x2_minus_1() {
        // x^2 - 1 = (x-1)(x+1)
        let f = poly(&[-1, 0, 1]);
        let factors = factor_integer_poly(&f);
        assert_eq!(factors.len(), 2);
        // Each should be linear
        for (fac, mult) in &factors {
            assert_eq!(fac.degree(), Some(1));
            assert_eq!(*mult, 1);
        }
    }

    #[test]
    fn test_factor_with_multiplicity() {
        // (x-1)^2 = x^2 - 2x + 1
        let f = poly(&[1, -2, 1]);
        let factors = factor_integer_poly(&f);
        // Should be [(x-1, 2)]
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].0.degree(), Some(1));
        assert_eq!(factors[0].1, 2);
    }

    #[test]
    fn test_factor_linear() {
        let f = poly(&[3, 1]); // x + 3
        let factors = factor_integer_poly(&f);
        assert_eq!(factors.len(), 1);
    }

    #[test]
    fn test_factor_constant() {
        let f = poly(&[5]); // constant
        let factors = factor_integer_poly(&f);
        assert!(factors.is_empty());
    }

    #[test]
    fn test_factor_reconstruct() {
        // (x-1)(x+1)(x-2) = x^3 - 2x^2 - x + 2
        let f = poly(&[2, -1, -2, 1]);
        let factors = factor_integer_poly(&f);

        // Should get 3 linear factors
        assert_eq!(factors.len(), 3, "expected 3 factors, got {:?}", factors);

        // Verify product = original (up to leading coefficient)
        let mut product = poly(&[1]);
        for (fac, mult) in &factors {
            for _ in 0..*mult {
                product = (&product * fac).make_monic();
            }
        }
        // Original monic
        let monic_f = f.make_monic();
        assert_eq!(product, monic_f);
    }

    #[test]
    fn test_combinations() {
        let items = vec![0, 1, 2, 3];
        let c2 = combinations(&items, 2);
        assert_eq!(c2.len(), 6); // 4 choose 2
        let c1 = combinations(&items, 1);
        assert_eq!(c1.len(), 4);
    }

    #[test]
    fn test_mignotte_bound() {
        let f = poly(&[6, 5, 1]); // x^2 + 5x + 6
        let bound = mignotte_bound(&f);
        // 2^2 * 6 = 24
        assert!(bound >= SmallInt::from(24i64));
    }
}
