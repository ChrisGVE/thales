//! Tests for advanced `RationalFunction` operations: polynomial division,
//! proper form, derivative, compose, pow, partial fractions, mod inverse.

use crate::numeric::dense_poly::DensePolynomial;
use crate::numeric::rational_fn::advanced::poly_mod_inverse;
use crate::numeric::ring::Ring;
use crate::numeric::BigRational;
use crate::numeric::RationalFunction;

type RPoly = DensePolynomial<BigRational>;
type RFn = RationalFunction<BigRational>;

fn ri(coeffs: &[i64]) -> RPoly {
    RPoly::from_coeffs(coeffs.iter().map(|&c| BigRational::from(c)).collect())
}

fn rfn(num: &[i64], den: &[i64]) -> RFn {
    RFn::new(ri(num), ri(den))
}

// ── Derivative ───────────────────────────────────────────────────────────────

#[test]
fn test_derivative_polynomial() {
    // d/dx(x^2) = 2x as rational function
    let f = RFn::from_poly(ri(&[0, 0, 1]));
    let fp = f.derivative();
    assert!(fp.is_polynomial());
    assert_eq!(*fp.numerator(), ri(&[0, 2]));
}

#[test]
fn test_derivative_fraction() {
    // d/dx(1/x) = -1/x^2
    let f = rfn(&[1], &[0, 1]);
    let fp = f.derivative();
    assert_eq!(*fp.numerator(), ri(&[-1]));
    assert_eq!(*fp.denominator(), ri(&[0, 0, 1]));
}

// ── Polynomial division / proper form ────────────────────────────────────────

#[test]
fn test_is_improper() {
    // (x^2 + 1) / (x + 1) is improper
    let f = rfn(&[1, 0, 1], &[1, 1]);
    assert!(f.is_improper());
}

#[test]
fn test_is_proper() {
    // 1 / (x + 1) is proper
    let f = rfn(&[1], &[1, 1]);
    assert!(!f.is_improper());
}

#[test]
fn test_polynomial_division() {
    // (x^2 + 1) / (x + 1) = (x - 1) + 2/(x+1)
    let f = rfn(&[1, 0, 1], &[1, 1]);
    let (q, r) = f.polynomial_division();
    assert_eq!(q, ri(&[-1, 1])); // x - 1
    assert_eq!(r, ri(&[2])); // 2
}

#[test]
fn test_polynomial_division_already_proper() {
    let f = rfn(&[1], &[1, 1]);
    let (q, r) = f.polynomial_division();
    assert!(q.is_zero());
    assert_eq!(r, ri(&[1]));
}

#[test]
fn test_polynomial_part() {
    // x^3 / (x + 1) = x^2 - x + 1 - 1/(x+1)
    let f = rfn(&[0, 0, 0, 1], &[1, 1]);
    let pp = f.polynomial_part();
    assert_eq!(pp, ri(&[1, -1, 1]));
}

#[test]
fn test_to_proper() {
    // (x^2 + x + 1) / (x + 1): quotient = x, remainder = 1
    let f = rfn(&[1, 1, 1], &[1, 1]);
    let (poly, proper) = f.to_proper();
    assert_eq!(poly, ri(&[0, 1])); // x
    assert_eq!(*proper.numerator(), ri(&[1])); // 1
    assert_eq!(*proper.denominator(), ri(&[1, 1])); // x+1
}

#[test]
fn test_to_proper_reconstructs() {
    // Verify: poly + proper = original
    let f = rfn(&[1, 0, 0, 1], &[1, 1]); // (x^3+1)/(x+1) = x^2-x+1 (exact)
    let (poly, proper) = f.to_proper();
    let reconstructed = RFn::from_poly(poly).add(&proper);
    assert_eq!(reconstructed, f);
}

// ── Compose ──────────────────────────────────────────────────────────────────

#[test]
fn test_compose_polynomial_at_x() {
    // (x+1) composed with x = x+1
    let f = RFn::from_poly(ri(&[1, 1]));
    let x = RFn::x();
    assert_eq!(f.compose(&x), f);
}

#[test]
fn test_compose_polynomial_at_value() {
    // (x^2) composed with (x+1) = (x+1)^2 = x^2+2x+1
    let f = RFn::from_poly(ri(&[0, 0, 1]));
    let g = RFn::from_poly(ri(&[1, 1]));
    let result = f.compose(&g);
    assert!(result.is_polynomial());
    assert_eq!(*result.numerator(), ri(&[1, 2, 1]));
}

// ── Pow ──────────────────────────────────────────────────────────────────────

#[test]
fn test_pow_zero() {
    let f = rfn(&[1, 1], &[-1, 1]);
    let p = f.pow(0);
    assert!(p.is_polynomial());
    assert_eq!(p.numerator().coeff(0), BigRational::from(1i64));
}

#[test]
fn test_pow_one() {
    let f = rfn(&[1, 1], &[-1, 1]);
    assert_eq!(f.pow(1), f);
}

#[test]
fn test_pow_two() {
    // ((x+1)/(x-1))^2 = (x^2+2x+1)/(x^2-2x+1)
    let f = rfn(&[1, 1], &[-1, 1]);
    let p = f.pow(2);
    assert_eq!(*p.numerator(), ri(&[1, 2, 1]));
    assert_eq!(*p.denominator(), ri(&[1, -2, 1]));
}

// ── Partial fraction decomposition ───────────────────────────────────────────

#[test]
fn test_pfd_simple_linear() {
    // 1/(x+1) — already partial
    let f = rfn(&[1], &[1, 1]);
    let (poly, terms) = f.partial_fraction_decomposition();
    assert!(poly.is_zero());
    assert_eq!(terms.len(), 1);
    assert_eq!(terms[0].power, 1);
}

#[test]
fn test_pfd_improper() {
    // (x^2+1)/(x+1) = x - 1 + 2/(x+1)
    let f = rfn(&[1, 0, 1], &[1, 1]);
    let (poly, terms) = f.partial_fraction_decomposition();
    assert_eq!(poly, ri(&[-1, 1])); // x - 1
    assert!(!terms.is_empty());
}

#[test]
fn test_pfd_zero() {
    let f = RFn::zero();
    let (poly, terms) = f.partial_fraction_decomposition();
    assert!(poly.is_zero());
    assert!(terms.is_empty());
}

#[test]
fn test_pfd_polynomial_input() {
    // x^2 + 1 (no denominator) — should return polynomial, no terms
    let f = RFn::from_poly(ri(&[1, 0, 1]));
    let (poly, terms) = f.partial_fraction_decomposition();
    assert_eq!(poly, ri(&[1, 0, 1]));
    assert!(terms.is_empty());
}

// ── Mod inverse utility ───────────────────────────────────────────────────────

#[test]
fn test_poly_mod_inverse() {
    // Inverse of (x+1) mod (x^2-1) doesn't exist (gcd = x+1)
    let a = ri(&[1, 1]);
    let m = ri(&[-1, 0, 1]);
    assert!(poly_mod_inverse(&a, &m).is_none());
}

#[test]
fn test_poly_mod_inverse_exists() {
    // Inverse of 2 mod (x+1) = 1/2
    let a = ri(&[2]);
    let m = ri(&[1, 1]);
    let inv = poly_mod_inverse(&a, &m).unwrap();
    let product = (&a * &inv).div_rem(&m).1;
    assert_eq!(product.degree(), Some(0));
    assert!(product.coeff(0).is_one());
}
