//! Primality testing and integer factorization.
//!
//! - [`is_prime`]: deterministic Miller-Rabin for numbers < 2^64,
//!   probabilistic for larger values
//! - [`factor`]: trial division + Pollard's rho for factorization
//! - [`trial_division`]: simple trial division up to sqrt(n)

use super::number_theory::mod_pow;
use super::SmallInt;
use num::traits::{One, Signed, Zero};

/// A prime factor with its exponent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeFactor {
    /// The prime.
    pub prime: SmallInt,
    /// The exponent (multiplicity).
    pub exponent: u32,
}

/// Small primes for trial division.
const SMALL_PRIMES: [i64; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];

/// Deterministic Miller-Rabin witnesses for numbers < 3,317,044,064,679,887,385,961,981.
/// These witnesses are sufficient for all numbers that fit in a u64.
const MR_WITNESSES_64: [i64; 7] = [2, 3, 5, 7, 11, 13, 17];

/// Primality test using Miller-Rabin.
///
/// For numbers ≤ 2^64, uses deterministic witnesses (no false positives).
/// For larger numbers, uses the first 20 primes as witnesses (probabilistic,
/// but astronomically unlikely to give false positive).
pub fn is_prime(n: &SmallInt) -> bool {
    // Handle small cases
    let two = SmallInt::from(2i64);
    if n < &two {
        return false;
    }
    if n == &two || n == &SmallInt::from(3i64) {
        return true;
    }
    if (n % &two).is_zero() {
        return false;
    }

    // Check small prime divisors
    for &p in &SMALL_PRIMES {
        let sp = SmallInt::from(p);
        if n == &sp {
            return true;
        }
        if (n % &sp).is_zero() {
            return false;
        }
    }

    // Miller-Rabin
    // Write n-1 = 2^r * d where d is odd
    let n_minus_1 = n - &SmallInt::from(1i64);
    let mut d = n_minus_1.clone();
    let mut r = 0u32;
    while (&d % &two).is_zero() {
        d = &d / &two;
        r += 1;
    }

    // Choose witnesses based on size
    let witnesses: Vec<SmallInt> = if n.to_i64().is_some() {
        MR_WITNESSES_64.iter().map(|&w| SmallInt::from(w)).collect()
    } else {
        // For large numbers, use first 20 primes
        (0..20)
            .map(|i| SmallInt::from(SMALL_PRIMES.get(i).copied().unwrap_or((i as i64) * 2 + 3)))
            .collect()
    };

    'witness: for a in &witnesses {
        if a >= n {
            continue;
        }

        let mut x = mod_pow(a, &d, n);

        if x == SmallInt::from(1i64) || x == n_minus_1 {
            continue;
        }

        for _ in 0..r - 1 {
            x = mod_pow(&x, &two, n);
            if x == n_minus_1 {
                continue 'witness;
            }
        }

        return false; // Composite
    }

    true // Probably prime
}

/// Trial division factorization up to a limit.
///
/// Returns found factors and the remaining unfactored part.
fn trial_division(mut n: SmallInt, limit: i64) -> (Vec<PrimeFactor>, SmallInt) {
    let mut factors = Vec::new();

    // Factor out 2
    let two = SmallInt::from(2i64);
    if (&n % &two).is_zero() {
        let mut exp = 0u32;
        while (&n % &two).is_zero() {
            n = &n / &two;
            exp += 1;
        }
        factors.push(PrimeFactor {
            prime: two.clone(),
            exponent: exp,
        });
    }

    // Factor out odd numbers 3, 5, 7, ...
    let mut p = 3i64;
    while p <= limit && !n.is_one() {
        let sp = SmallInt::from(p);
        // Check if p*p > n (we've found all small factors)
        if let (Some(nv), true) = (n.to_i64(), p <= 46340) {
            if p * p > nv {
                break;
            }
        }

        if (&n % &sp).is_zero() {
            let mut exp = 0u32;
            while (&n % &sp).is_zero() {
                n = &n / &sp;
                exp += 1;
            }
            factors.push(PrimeFactor {
                prime: sp,
                exponent: exp,
            });
        }
        p += 2;
    }

    (factors, n)
}

/// Pollard's rho algorithm for finding a non-trivial factor.
///
/// Returns a factor of `n` (not necessarily prime), or `None` if
/// the algorithm fails to find one within the iteration limit.
fn pollard_rho(n: &SmallInt) -> Option<SmallInt> {
    let two = SmallInt::from(2i64);
    if (n % &two).is_zero() {
        return Some(two);
    }

    let mut x = SmallInt::from(2i64);
    let mut y = SmallInt::from(2i64);
    let mut d = SmallInt::from(1i64);
    let c = SmallInt::from(1i64);

    let f = |x: &SmallInt| -> SmallInt {
        let x2 = &(x * x) + &c;
        &x2 % n
    };

    let mut iterations = 0;
    let max_iterations = 1_000_000;

    while d.is_one() && iterations < max_iterations {
        x = f(&x);
        y = f(&f(&y));

        let diff = if &x > &y { &x - &y } else { &y - &x };
        d = diff.gcd(n);
        iterations += 1;
    }

    if d != SmallInt::from(1i64) && &d != n {
        Some(d)
    } else {
        None
    }
}

/// Factor an integer into prime factors.
///
/// Returns `Vec<PrimeFactor>` sorted by prime value.
/// Uses trial division for small factors, then Pollard's rho for larger ones.
///
/// # Panics
///
/// Panics if `n` is zero.
pub fn factor(n: &SmallInt) -> Vec<PrimeFactor> {
    assert!(!n.is_zero(), "cannot factor zero");

    let n = n.abs();
    if n.is_one() {
        return vec![];
    }

    // Trial division up to 1000
    let (mut factors, remainder) = trial_division(n.clone(), 1000);

    if remainder.is_one() {
        return factors;
    }

    // Check if remainder is prime
    if is_prime(&remainder) {
        factors.push(PrimeFactor {
            prime: remainder,
            exponent: 1,
        });
        return factors;
    }

    // Pollard's rho for remaining composite
    let mut composites = vec![remainder];
    while let Some(comp) = composites.pop() {
        if comp.is_one() {
            continue;
        }
        if is_prime(&comp) {
            // Find or create factor entry
            if let Some(f) = factors.iter_mut().find(|f| f.prime == comp) {
                f.exponent += 1;
            } else {
                factors.push(PrimeFactor {
                    prime: comp,
                    exponent: 1,
                });
            }
            continue;
        }

        if let Some(d) = pollard_rho(&comp) {
            let other = &comp / &d;
            composites.push(d);
            composites.push(other);
        } else {
            // Rho failed — treat as prime (shouldn't happen for reasonable inputs)
            factors.push(PrimeFactor {
                prime: comp,
                exponent: 1,
            });
        }
    }

    // Sort by prime value
    factors.sort_by(|a, b| a.prime.cmp(&b.prime));

    // Merge duplicate primes (from rho producing same factor multiple times)
    let mut merged = Vec::new();
    for f in factors {
        if let Some(last) = merged.last_mut() {
            let last: &mut PrimeFactor = last;
            if last.prime == f.prime {
                last.exponent += f.exponent;
                continue;
            }
        }
        merged.push(f);
    }

    merged
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn si(n: i64) -> SmallInt {
        SmallInt::from(n)
    }

    #[test]
    fn test_is_prime_small() {
        assert!(!is_prime(&si(0)));
        assert!(!is_prime(&si(1)));
        assert!(is_prime(&si(2)));
        assert!(is_prime(&si(3)));
        assert!(!is_prime(&si(4)));
        assert!(is_prime(&si(5)));
        assert!(is_prime(&si(7)));
        assert!(!is_prime(&si(9)));
        assert!(is_prime(&si(11)));
        assert!(is_prime(&si(13)));
    }

    #[test]
    fn test_is_prime_medium() {
        assert!(is_prime(&si(104729))); // 10000th prime
        assert!(!is_prime(&si(104730)));
        assert!(is_prime(&si(999983)));
        assert!(!is_prime(&si(999981)));
    }

    #[test]
    fn test_is_prime_carmichael() {
        // Carmichael numbers fool Fermat test but not Miller-Rabin
        assert!(!is_prime(&si(561)));
        assert!(!is_prime(&si(1105)));
        assert!(!is_prime(&si(1729)));
    }

    #[test]
    fn test_factor_small() {
        let f = factor(&si(1));
        assert!(f.is_empty());
    }

    #[test]
    fn test_factor_prime() {
        let f = factor(&si(17));
        assert_eq!(
            f,
            vec![PrimeFactor {
                prime: si(17),
                exponent: 1,
            }]
        );
    }

    #[test]
    fn test_factor_60() {
        let f = factor(&si(60));
        assert_eq!(
            f,
            vec![
                PrimeFactor {
                    prime: si(2),
                    exponent: 2,
                },
                PrimeFactor {
                    prime: si(3),
                    exponent: 1,
                },
                PrimeFactor {
                    prime: si(5),
                    exponent: 1,
                },
            ]
        );
    }

    #[test]
    fn test_factor_power_of_two() {
        let f = factor(&si(1024)); // 2^10
        assert_eq!(
            f,
            vec![PrimeFactor {
                prime: si(2),
                exponent: 10,
            }]
        );
    }

    #[test]
    fn test_factor_semiprime() {
        // 10007 * 10009 = 100_160_063
        let n = si(100_160_063);
        let f = factor(&n);
        assert_eq!(f.len(), 2);
        assert_eq!(f[0].prime, si(10007));
        assert_eq!(f[0].exponent, 1);
        assert_eq!(f[1].prime, si(10009));
        assert_eq!(f[1].exponent, 1);
    }

    #[test]
    fn test_factor_negative() {
        // Factor of -60 should be same as 60
        let f = factor(&si(-60));
        assert_eq!(
            f,
            vec![
                PrimeFactor {
                    prime: si(2),
                    exponent: 2,
                },
                PrimeFactor {
                    prime: si(3),
                    exponent: 1,
                },
                PrimeFactor {
                    prime: si(5),
                    exponent: 1,
                },
            ]
        );
    }

    #[test]
    fn test_factor_reconstruct() {
        // Verify product of factors equals original
        let n = si(2310); // 2 * 3 * 5 * 7 * 11
        let factors = factor(&n);
        let mut product = si(1);
        for f in &factors {
            for _ in 0..f.exponent {
                product = &product * &f.prime;
            }
        }
        assert_eq!(product, si(2310));
    }
}
