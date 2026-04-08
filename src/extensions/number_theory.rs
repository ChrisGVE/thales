//! Number theory operations: GCD, LCM, modular arithmetic.
//!
//! Enable with `features = ["number-theory"]` in Cargo.toml.

/// Compute GCD using Euclidean algorithm.
pub fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Compute LCM.
pub fn lcm(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 {
        0
    } else {
        (a / gcd(a, b)) * b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd_basic() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(17, 13), 1);
        assert_eq!(gcd(100, 75), 25);
    }

    #[test]
    fn test_gcd_with_zero() {
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(5, 0), 5);
        assert_eq!(gcd(0, 0), 0);
    }

    #[test]
    fn test_gcd_negative() {
        assert_eq!(gcd(-12, 8), 4);
        assert_eq!(gcd(12, -8), 4);
        assert_eq!(gcd(-12, -8), 4);
    }

    #[test]
    fn test_lcm_basic() {
        assert_eq!(lcm(4, 6), 12);
        assert_eq!(lcm(3, 5), 15);
        assert_eq!(lcm(12, 8), 24);
    }

    #[test]
    fn test_lcm_with_zero() {
        assert_eq!(lcm(0, 5), 0);
        assert_eq!(lcm(5, 0), 0);
    }

    #[test]
    fn test_lcm_same_number() {
        assert_eq!(lcm(7, 7), 7);
    }
}
