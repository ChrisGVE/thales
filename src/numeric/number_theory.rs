//! Number theory utilities: extended GCD, modular arithmetic, CRT.
//!
//! All operations work on [`SmallInt`] for overflow-safe computation.

use super::SmallInt;

/// Extended GCD result: `gcd = s * a + t * b`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtGcdResult {
    /// The greatest common divisor (non-negative).
    pub gcd: SmallInt,
    /// Bezout coefficient for `a`.
    pub s: SmallInt,
    /// Bezout coefficient for `b`.
    pub t: SmallInt,
}

/// Extended Euclidean algorithm.
///
/// Returns `(gcd, s, t)` such that `gcd = s * a + t * b`.
/// The GCD is always non-negative.
pub fn ext_gcd(a: &SmallInt, b: &SmallInt) -> ExtGcdResult {
    use num::traits::{Signed, Zero};

    if b.is_zero() {
        let sign = if a.is_negative() {
            SmallInt::from(-1i64)
        } else {
            SmallInt::from(1i64)
        };
        return ExtGcdResult {
            gcd: a.abs(),
            s: sign,
            t: SmallInt::from(0i64),
        };
    }

    let mut old_r = a.clone();
    let mut r = b.clone();
    let mut old_s = SmallInt::from(1i64);
    let mut s = SmallInt::from(0i64);
    let mut old_t = SmallInt::from(0i64);
    let mut t = SmallInt::from(1i64);

    while !r.is_zero() {
        let q = &old_r / &r;

        let new_r = &old_r - &(&q * &r);
        old_r = r;
        r = new_r;

        let new_s = &old_s - &(&q * &s);
        old_s = s;
        s = new_s;

        let new_t = &old_t - &(&q * &t);
        old_t = t;
        t = new_t;
    }

    // Ensure gcd is non-negative
    if old_r.is_negative() {
        old_r = -old_r;
        old_s = -old_s;
        old_t = -old_t;
    }

    ExtGcdResult {
        gcd: old_r,
        s: old_s,
        t: old_t,
    }
}

/// Modular inverse of `a` modulo `m`.
///
/// Returns `a^(-1) mod m` if gcd(a, m) = 1, otherwise `None`.
pub fn mod_inverse(a: &SmallInt, m: &SmallInt) -> Option<SmallInt> {
    use num::traits::Zero;

    let result = ext_gcd(a, m);
    if result.gcd != SmallInt::from(1i64) {
        return None;
    }

    // Normalize s to be in [0, m)
    let m_abs = if m < &SmallInt::zero() {
        -m.clone()
    } else {
        m.clone()
    };
    let mut inv = &result.s % &m_abs;
    if inv < SmallInt::from(0i64) {
        inv = &inv + &m_abs;
    }
    Some(inv)
}

/// Modular exponentiation: `base^exp mod m`.
///
/// Uses repeated squaring. Requires `m > 0`.
pub fn mod_pow(base: &SmallInt, exp: &SmallInt, m: &SmallInt) -> SmallInt {
    use num::traits::{One, Zero};

    assert!(!m.is_zero(), "modulus must be non-zero");

    if exp.is_zero() {
        return SmallInt::from(1i64);
    }

    // Handle negative exponent: base^(-exp) = (base^(-1))^exp mod m
    if exp < &SmallInt::zero() {
        let inv = mod_inverse(base, m).expect("base must be invertible for negative exponent");
        let pos_exp = -exp.clone();
        return mod_pow(&inv, &pos_exp, m);
    }

    let mut result = SmallInt::from(1i64);
    let base_mod = &(base % m) + m;
    let mut base = &base_mod % m; // Ensure positive
    let mut exp = exp.clone();
    let two = SmallInt::from(2i64);

    while exp > SmallInt::zero() {
        if &(&exp % &two) == &SmallInt::one() {
            result = &(&result * &base) % m;
        }
        exp = &exp / &two;
        base = &(&base * &base) % m;
    }

    result
}

/// Chinese Remainder Theorem.
///
/// Given residues `r[i]` and moduli `m[i]` (pairwise coprime),
/// finds the unique `x` such that `x ≡ r[i] (mod m[i])` for all i,
/// with `0 ≤ x < M` where `M = product of all m[i]`.
///
/// Returns `None` if moduli are not pairwise coprime.
pub fn crt(residues: &[SmallInt], moduli: &[SmallInt]) -> Option<SmallInt> {
    use num::traits::Zero;

    assert_eq!(
        residues.len(),
        moduli.len(),
        "residues and moduli must have equal length"
    );

    if residues.is_empty() {
        return Some(SmallInt::from(0i64));
    }

    // Compute M = product of all moduli
    let big_m: SmallInt = moduli.iter().fold(SmallInt::from(1i64), |acc, m| &acc * m);

    let mut x = SmallInt::from(0i64);

    for (r, m) in residues.iter().zip(moduli.iter()) {
        let mi = &big_m / m; // M / m_i
        let inv = mod_inverse(&mi, m)?; // mi^(-1) mod m_i
        let term = &(&(r * &mi) * &inv) % &big_m;
        x = &(&x + &term) % &big_m;
    }

    // Normalize to [0, M)
    if x < SmallInt::zero() {
        x = &x + &big_m;
    }

    Some(x)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn si(n: i64) -> SmallInt {
        SmallInt::from(n)
    }

    #[test]
    fn test_ext_gcd_basic() {
        // gcd(12, 8) = 4, and 1*12 + (-1)*8 = 4
        let r = ext_gcd(&si(12), &si(8));
        assert_eq!(r.gcd, si(4));
        // Verify Bezout identity
        let check = &(&r.s * &si(12)) + &(&r.t * &si(8));
        assert_eq!(check, si(4));
    }

    #[test]
    fn test_ext_gcd_coprime() {
        let r = ext_gcd(&si(7), &si(5));
        assert_eq!(r.gcd, si(1));
        let check = &(&r.s * &si(7)) + &(&r.t * &si(5));
        assert_eq!(check, si(1));
    }

    #[test]
    fn test_ext_gcd_zero() {
        let r = ext_gcd(&si(5), &si(0));
        assert_eq!(r.gcd, si(5));
        assert_eq!(r.s, si(1));
        assert_eq!(r.t, si(0));
    }

    #[test]
    fn test_ext_gcd_negative() {
        let r = ext_gcd(&si(-12), &si(8));
        assert_eq!(r.gcd, si(4));
        let check = &(&r.s * &si(-12)) + &(&r.t * &si(8));
        assert_eq!(check, si(4));
    }

    #[test]
    fn test_mod_inverse_basic() {
        // 3^(-1) mod 7 = 5 (since 3*5 = 15 ≡ 1 mod 7)
        let inv = mod_inverse(&si(3), &si(7)).unwrap();
        assert_eq!(inv, si(5));
    }

    #[test]
    fn test_mod_inverse_no_inverse() {
        // gcd(6, 4) = 2, so no inverse
        assert!(mod_inverse(&si(6), &si(4)).is_none());
    }

    #[test]
    fn test_mod_inverse_one() {
        let inv = mod_inverse(&si(1), &si(7)).unwrap();
        assert_eq!(inv, si(1));
    }

    #[test]
    fn test_mod_pow_basic() {
        // 2^10 mod 1000 = 1024 mod 1000 = 24
        let result = mod_pow(&si(2), &si(10), &si(1000));
        assert_eq!(result, si(24));
    }

    #[test]
    fn test_mod_pow_zero_exp() {
        let result = mod_pow(&si(5), &si(0), &si(7));
        assert_eq!(result, si(1));
    }

    #[test]
    fn test_mod_pow_large() {
        // 3^100 mod 7
        // 3^1=3, 3^2=2, 3^3=6, 3^4=4, 3^5=5, 3^6=1 mod 7 (order 6)
        // 100 mod 6 = 4, so 3^100 ≡ 3^4 = 81 ≡ 4 mod 7
        let result = mod_pow(&si(3), &si(100), &si(7));
        assert_eq!(result, si(4));
    }

    #[test]
    fn test_crt_basic() {
        // x ≡ 2 (mod 3), x ≡ 3 (mod 5) → x = 8
        let x = crt(&[si(2), si(3)], &[si(3), si(5)]).unwrap();
        assert_eq!(x, si(8));
    }

    #[test]
    fn test_crt_three_moduli() {
        // x ≡ 1 (mod 2), x ≡ 2 (mod 3), x ≡ 3 (mod 5)
        // x = 23 (mod 30): check 23%2=1, 23%3=2, 23%5=3
        let x = crt(&[si(1), si(2), si(3)], &[si(2), si(3), si(5)]).unwrap();
        assert_eq!(&x % &si(2), si(1));
        assert_eq!(&x % &si(3), si(2));
        assert_eq!(&x % &si(5), si(3));
    }

    #[test]
    fn test_crt_not_coprime() {
        // 4 and 6 share factor 2, CRT doesn't apply
        assert!(crt(&[si(1), si(2)], &[si(4), si(6)]).is_none());
    }

    #[test]
    fn test_crt_empty() {
        let x = crt(&[], &[]).unwrap();
        assert_eq!(x, si(0));
    }

    #[test]
    fn test_crt_single() {
        let x = crt(&[si(3)], &[si(7)]).unwrap();
        assert_eq!(x, si(3));
    }
}
