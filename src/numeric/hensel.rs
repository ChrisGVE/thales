//! Hensel lifting for polynomial factorization.
//!
//! Given a factorization `f ≡ g·h (mod p)` where `gcd(g,h) = 1 mod p`,
//! lift to `f ≡ G·H (mod p^k)` for increasing `k`. Used to reconstruct
//! integer polynomial factors from modular factorizations.
//!
//! Implements linear Hensel lifting (one step at a time, doubling precision).

use super::number_theory::mod_inverse;
use super::SmallInt;

/// Result of Hensel lifting.
#[derive(Clone, Debug)]
pub struct HenselLift {
    /// Lifted factor g with `f ≡ g*h (mod modulus)`.
    pub g: Vec<i64>,
    /// Lifted factor h with `f ≡ g*h (mod modulus)`.
    pub h: Vec<i64>,
    /// Current modulus (p^k after k steps).
    pub modulus: i64,
}

/// Perform linear Hensel lifting.
///
/// Given:
/// - `f`: the polynomial to factor (integer coefficients)
/// - `g0, h0`: factorization mod `p` (i.e., `f ≡ g0*h0 (mod p)`)
/// - `p`: the prime
/// - `k`: number of lifting steps (result is mod `p^k`)
///
/// Requires: `gcd(g0, h0) = 1 mod p` (coprime modular factors).
///
/// Returns lifted `g, h` such that `f ≡ g*h (mod p^k)`.
pub fn hensel_lift(f: &[i64], g0: &[i64], h0: &[i64], p: i64, k: u32) -> HenselLift {
    assert!(k >= 1, "k must be at least 1");

    // Extended GCD to find s, t with s*g0 + t*h0 ≡ 1 (mod p)
    let (_, s0, t0) = poly_ext_gcd_mod(g0, h0, p);

    let mut g = g0.to_vec();
    let mut h = h0.to_vec();
    let mut s = s0;
    let mut t = t0;
    let mut modulus = p;

    for _ in 1..k {
        let new_modulus = modulus * p;

        // Error: e = f - g*h (mod new_modulus)
        let gh = poly_mul(&g, &h);
        let e = poly_sub(f, &gh);
        let e_mod: Vec<i64> = e.iter().map(|&c| centered_mod(c, new_modulus)).collect();

        // If error is zero mod new_modulus, we're done early
        if e_mod.iter().all(|&c| c % new_modulus == 0) {
            modulus = new_modulus;
            continue;
        }

        // Divide error by current modulus to get the "correction" term
        // e = modulus * e1 (approximately)
        // Actually, work with e directly mod new_modulus

        // Compute corrections: sigma = s*e mod h, tau from the rest
        let se = poly_mul_mod(&s, &e_mod, new_modulus);
        let (q_se, sigma) = poly_div_mod(&se, &h, new_modulus);
        let te = poly_mul_mod(&t, &e_mod, new_modulus);
        let tau = poly_add_mod(&te, &poly_mul_mod(&q_se, &g, new_modulus), new_modulus);

        // Update: g = g + tau, h = h + sigma
        g = poly_add_mod(&g, &tau, new_modulus);
        h = poly_add_mod(&h, &sigma, new_modulus);

        // Update s, t for the new modulus (optional, needed for multi-step)
        // s*g + t*h ≡ 1 (mod new_modulus)
        let sg = poly_mul_mod(&s, &g, new_modulus);
        let th = poly_mul_mod(&t, &h, new_modulus);
        let sth_sum = poly_add_mod(&sg, &th, new_modulus);
        let one_vec = vec![1i64];
        let err_st = poly_sub_mod(&one_vec, &sth_sum, new_modulus);

        let s_corr = poly_mul_mod(&s, &err_st, new_modulus);
        let (_, s_corr_rem) = poly_div_mod(&s_corr, &h, new_modulus);
        s = poly_add_mod(&s, &s_corr_rem, new_modulus);

        let t_corr = poly_mul_mod(&t, &err_st, new_modulus);
        let (_, t_corr_rem) = poly_div_mod(&t_corr, &g, new_modulus);
        t = poly_add_mod(&t, &t_corr_rem, new_modulus);

        modulus = new_modulus;
    }

    // Normalize to symmetric representation
    g = g.iter().map(|&c| centered_mod(c, modulus)).collect();
    h = h.iter().map(|&c| centered_mod(c, modulus)).collect();
    trim(&mut g);
    trim(&mut h);

    HenselLift { g, h, modulus }
}

/// Symmetric (centered) modular reduction to `(-m/2, m/2]`.
fn centered_mod(a: i64, m: i64) -> i64 {
    let r = ((a % m) + m) % m;
    if r > m / 2 {
        r - m
    } else {
        r
    }
}

fn trim(a: &mut Vec<i64>) {
    while a.last() == Some(&0) && a.len() > 1 {
        a.pop();
    }
}

// ── Polynomial arithmetic (integer) ─────────────────────────────────────────

fn poly_mul(a: &[i64], b: &[i64]) -> Vec<i64> {
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

fn poly_sub(a: &[i64], b: &[i64]) -> Vec<i64> {
    let len = a.len().max(b.len());
    let mut result = vec![0i64; len];
    for i in 0..len {
        let ai = a.get(i).copied().unwrap_or(0);
        let bi = b.get(i).copied().unwrap_or(0);
        result[i] = ai - bi;
    }
    result
}

// ── Polynomial arithmetic (mod m) ───────────────────────────────────────────

fn poly_mul_mod(a: &[i64], b: &[i64], m: i64) -> Vec<i64> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut result = vec![0i64; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] = (result[i + j] + ai * bj) % m;
        }
    }
    trim(&mut result);
    result
}

fn poly_add_mod(a: &[i64], b: &[i64], m: i64) -> Vec<i64> {
    let len = a.len().max(b.len());
    let mut result = vec![0i64; len];
    for i in 0..len {
        let ai = a.get(i).copied().unwrap_or(0);
        let bi = b.get(i).copied().unwrap_or(0);
        result[i] = ((ai + bi) % m + m) % m;
    }
    trim(&mut result);
    result
}

fn poly_sub_mod(a: &[i64], b: &[i64], m: i64) -> Vec<i64> {
    let len = a.len().max(b.len());
    let mut result = vec![0i64; len];
    for i in 0..len {
        let ai = a.get(i).copied().unwrap_or(0);
        let bi = b.get(i).copied().unwrap_or(0);
        result[i] = ((ai - bi) % m + m) % m;
    }
    trim(&mut result);
    result
}

/// Polynomial division mod m. Returns (quotient, remainder).
fn poly_div_mod(a: &[i64], b: &[i64], m: i64) -> (Vec<i64>, Vec<i64>) {
    if b.is_empty() || b.iter().all(|&c| c % m == 0) {
        return (vec![], a.to_vec());
    }
    if a.len() < b.len() {
        return (vec![], a.to_vec());
    }

    let b_lc = *b.last().unwrap();
    let b_lc_inv = mod_inv_i64(((b_lc % m) + m) % m, m);
    let mut r = a.to_vec();
    let mut q = vec![0i64; a.len() - b.len() + 1];

    while r.len() >= b.len() {
        let lc_r = ((*r.last().unwrap() % m) + m) % m;
        if lc_r == 0 {
            r.pop();
            continue;
        }
        let coeff = (lc_r * b_lc_inv) % m;
        let shift = r.len() - b.len();
        q[shift] = coeff;
        for (i, &bc) in b.iter().enumerate() {
            r[shift + i] = ((r[shift + i] - coeff * bc) % m + m) % m;
        }
        trim(&mut r);
    }

    trim(&mut q);
    (q, r)
}

/// Extended GCD for polynomials mod p.
fn poly_ext_gcd_mod(a: &[i64], b: &[i64], p: i64) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    if b.is_empty() || b.iter().all(|&c| c % p == 0) {
        return (a.to_vec(), vec![1], vec![]);
    }

    let mut old_r = a.to_vec();
    let mut r = b.to_vec();
    let mut old_s: Vec<i64> = vec![1];
    let mut s: Vec<i64> = vec![];
    let mut old_t: Vec<i64> = vec![];
    let mut t: Vec<i64> = vec![1];

    while !r.is_empty() && !r.iter().all(|&c| c % p == 0) {
        let (q, rem) = poly_div_mod(&old_r, &r, p);
        old_r = r;
        r = rem;

        let qs = poly_mul_mod(&q, &s, p);
        let new_s = poly_sub_mod(&old_s, &qs, p);
        old_s = s;
        s = new_s;

        let qt = poly_mul_mod(&q, &t, p);
        let new_t = poly_sub_mod(&old_t, &qt, p);
        old_t = t;
        t = new_t;
    }

    // Make gcd monic
    if !old_r.is_empty() {
        let lc = *old_r.last().unwrap();
        let lc_mod = ((lc % p) + p) % p;
        if lc_mod != 0 && lc_mod != 1 {
            let inv = mod_inv_i64(lc_mod, p);
            old_r = old_r.iter().map(|&c| (c * inv % p + p) % p).collect();
            old_s = old_s.iter().map(|&c| (c * inv % p + p) % p).collect();
            old_t = old_t.iter().map(|&c| (c * inv % p + p) % p).collect();
        }
    }

    (old_r, old_s, old_t)
}

fn mod_inv_i64(a: i64, p: i64) -> i64 {
    mod_inverse(&SmallInt::from(a), &SmallInt::from(p))
        .and_then(|inv| inv.to_i64())
        .unwrap_or(1)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_centered_mod() {
        assert_eq!(centered_mod(7, 5), 2);
        assert_eq!(centered_mod(3, 5), -2);
        assert_eq!(centered_mod(0, 5), 0);
        assert_eq!(centered_mod(-1, 5), -1);
    }

    #[test]
    fn test_poly_ext_gcd_mod() {
        // gcd(x+1, x-1) mod 5 = 1
        let a = vec![1, 1]; // x + 1
        let b = vec![4, 1]; // x - 1 = x + 4 mod 5
        let (g, s, t) = poly_ext_gcd_mod(&a, &b, 5);
        // g should be constant (gcd = 1)
        assert_eq!(g.len(), 1, "gcd should be constant, got {:?}", g);
        // Verify: s*a + t*b ≡ g (mod 5)
        let sa = poly_mul_mod(&s, &a, 5);
        let tb = poly_mul_mod(&t, &b, 5);
        let sum = poly_add_mod(&sa, &tb, 5);
        assert_eq!(sum, g);
    }

    #[test]
    fn test_hensel_lift_x2_minus_1() {
        // f = x^2 - 1, factors mod 5: (x-1)(x+1) = (x+4)(x+1)
        let f = vec![-1, 0, 1]; // x^2 - 1
        let g0 = vec![4, 1]; // x + 4 = x - 1 mod 5
        let h0 = vec![1, 1]; // x + 1

        let result = hensel_lift(&f, &g0, &h0, 5, 3);

        // Verify: g*h ≡ f (mod 5^3 = 125)
        let gh = poly_mul(&result.g, &result.h);
        for (i, &fi) in f.iter().enumerate() {
            let ghi = gh.get(i).copied().unwrap_or(0);
            assert_eq!(
                ((fi - ghi) % result.modulus + result.modulus) % result.modulus,
                0,
                "coefficient {i} mismatch: f={fi}, g*h={ghi}, mod={}",
                result.modulus
            );
        }
    }

    #[test]
    fn test_hensel_lift_simple() {
        // f = x^2 + 5x + 6 = (x+2)(x+3)
        // mod 7: g0 = x+2, h0 = x+3
        let f = vec![6, 5, 1];
        let g0 = vec![2, 1]; // x + 2
        let h0 = vec![3, 1]; // x + 3

        let result = hensel_lift(&f, &g0, &h0, 7, 2);

        // g*h should equal f mod 49
        let gh = poly_mul(&result.g, &result.h);
        for (i, &fi) in f.iter().enumerate() {
            let ghi = gh.get(i).copied().unwrap_or(0);
            assert_eq!(
                ((fi - ghi) % result.modulus + result.modulus) % result.modulus,
                0,
                "lift failed at coeff {i}"
            );
        }
    }

    #[test]
    fn test_hensel_lift_k1_no_change() {
        // k=1 means no lifting, just return original factors
        let f = vec![-1, 0, 1];
        let g0 = vec![4, 1];
        let h0 = vec![1, 1];

        let result = hensel_lift(&f, &g0, &h0, 5, 1);
        assert_eq!(result.modulus, 5);
    }

    #[test]
    fn test_hensel_recovers_exact_factors() {
        // f = (x+2)(x+3) = x^2 + 5x + 6
        // Factor mod 11: g0 = x+2, h0 = x+3
        // After lifting, should recover exact factors since coefficients are small
        let f = vec![6, 5, 1];
        let g0 = vec![2, 1];
        let h0 = vec![3, 1];

        let result = hensel_lift(&f, &g0, &h0, 11, 2);

        // With modulus 121, the centered representation should give exact factors
        assert_eq!(result.g, vec![2, 1], "expected x+2");
        assert_eq!(result.h, vec![3, 1], "expected x+3");
    }
}
