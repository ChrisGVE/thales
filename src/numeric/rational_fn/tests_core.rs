//! Tests for core `RationalFunction` operations: construction, arithmetic,
//! operator overloads, evaluation, display, recip, and neg.

use crate::numeric::dense_poly::DensePolynomial;
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

// ── Construction and reduction ──────────────────────────────────────────────

#[test]
fn test_auto_reduces() {
    // (x^2 - 1) / (x - 1) = x + 1
    let f = rfn(&[-1, 0, 1], &[-1, 1]);
    assert!(f.is_polynomial());
    assert_eq!(f.numerator().clone(), ri(&[1, 1]));
}

#[test]
fn test_monic_denominator() {
    // (2x) / (2x + 2) should reduce to x / (x + 1)
    let f = rfn(&[0, 2], &[2, 2]);
    assert_eq!(
        *f.denominator().leading_coeff().unwrap(),
        BigRational::from(1i64)
    );
}

#[test]
fn test_from_poly() {
    let p = ri(&[1, 2, 3]);
    let f = RFn::from_poly(p.clone());
    assert!(f.is_polynomial());
    assert_eq!(*f.numerator(), p);
}

#[test]
fn test_zero() {
    let f = RFn::zero();
    assert!(f.is_zero());
    assert!(f.is_polynomial());
}

#[test]
fn test_constant() {
    let f = RFn::constant(BigRational::from(5i64));
    assert!(f.is_polynomial());
    assert_eq!(f.numerator().coeff(0), BigRational::from(5i64));
}

#[test]
#[should_panic(expected = "zero denominator")]
fn test_zero_denominator_panics() {
    let _ = rfn(&[1, 1], &[]);
}

// ── Arithmetic ──────────────────────────────────────────────────────────────

#[test]
fn test_add() {
    // 1/(x-1) + 1/(x+1) = 2x/(x^2-1)
    let a = rfn(&[1], &[-1, 1]);
    let b = rfn(&[1], &[1, 1]);
    let sum = a.add(&b);
    assert_eq!(*sum.numerator(), ri(&[0, 2]));
    assert_eq!(*sum.denominator(), ri(&[-1, 0, 1]));
}

#[test]
fn test_add_reduces() {
    // x/(x+1) + 1/(x+1) = (x+1)/(x+1) = 1
    let a = rfn(&[0, 1], &[1, 1]);
    let b = rfn(&[1], &[1, 1]);
    let sum = a.add(&b);
    assert!(sum.is_polynomial());
    assert_eq!(sum.numerator().coeff(0), BigRational::from(1i64));
    assert_eq!(sum.numerator().degree(), Some(0));
}

#[test]
fn test_add_same_denominator() {
    // 1/(x+1) + x/(x+1) should use GCD optimization (dens are identical)
    let a = rfn(&[1], &[1, 1]);
    let b = rfn(&[0, 1], &[1, 1]);
    let sum = a.add(&b);
    assert!(sum.is_polynomial());
    assert_eq!(sum.numerator().coeff(0), BigRational::from(1i64));
}

#[test]
fn test_add_with_zero() {
    let a = rfn(&[1], &[-1, 1]);
    let z = RFn::zero();
    assert_eq!(a.add(&z), a);
    assert_eq!(z.add(&a), a);
}

#[test]
fn test_add_gcd_optimization() {
    // 1/(x*(x+1)) + 1/(x*(x-1))
    // GCD of denominators = x → result = 2/(x^2-1)
    let a = rfn(&[1], &[0, 1, 1]);
    let b = rfn(&[1], &[0, -1, 1]);
    let sum = a.add(&b);
    assert_eq!(*sum.numerator(), ri(&[2]));
    assert_eq!(*sum.denominator(), ri(&[-1, 0, 1]));
}

#[test]
fn test_sub() {
    // 1/(x-1) - 1/(x+1) = 2/(x^2-1)
    let a = rfn(&[1], &[-1, 1]);
    let b = rfn(&[1], &[1, 1]);
    let diff = a.sub(&b);
    assert_eq!(*diff.numerator(), ri(&[2]));
    assert_eq!(*diff.denominator(), ri(&[-1, 0, 1]));
}

#[test]
fn test_mul() {
    // (x+1)/1 * 1/(x+1) = 1
    let a = RFn::from_poly(ri(&[1, 1]));
    let b = rfn(&[1], &[1, 1]);
    let prod = a.mul(&b);
    assert!(prod.is_polynomial());
    assert_eq!(prod.numerator().degree(), Some(0));
}

#[test]
fn test_mul_cross_reduces() {
    // (x^2-1)/(x+2) * (x+2)/(x-1) = x+1
    let a = rfn(&[-1, 0, 1], &[2, 1]);
    let b = rfn(&[2, 1], &[-1, 1]);
    let prod = a.mul(&b);
    assert!(prod.is_polynomial());
    assert_eq!(*prod.numerator(), ri(&[1, 1]));
}

#[test]
fn test_mul_by_zero() {
    let a = rfn(&[1, 1], &[-1, 1]);
    let z = RFn::zero();
    assert!(a.mul(&z).is_zero());
    assert!(z.mul(&a).is_zero());
}

#[test]
fn test_div() {
    // (x/1) / (x/1) = 1
    let x = RFn::from_poly(ri(&[0, 1]));
    let q = x.div(&x);
    assert!(q.is_polynomial());
    assert_eq!(q.numerator().degree(), Some(0));
}

#[test]
#[should_panic(expected = "division by zero")]
fn test_div_by_zero_panics() {
    let f = RFn::from_poly(ri(&[1, 1]));
    let _ = f.div(&RFn::zero());
}

// ── Operator overloads ──────────────────────────────────────────────────────

#[test]
fn test_ops_add() {
    let a = rfn(&[1], &[-1, 1]);
    let b = rfn(&[1], &[1, 1]);
    let sum = &a + &b;
    assert_eq!(sum, a.add(&b));
}

#[test]
fn test_ops_sub() {
    let a = rfn(&[1], &[-1, 1]);
    let b = rfn(&[1], &[1, 1]);
    let diff = &a - &b;
    assert_eq!(diff, a.sub(&b));
}

#[test]
fn test_ops_mul() {
    let a = RFn::from_poly(ri(&[1, 1]));
    let b = rfn(&[1], &[1, 1]);
    let prod = &a * &b;
    assert_eq!(prod, a.mul(&b));
}

#[test]
fn test_ops_div() {
    let a = rfn(&[1, 1], &[-1, 1]);
    let b = rfn(&[1], &[1, 1]);
    let q = &a / &b;
    assert_eq!(q, a.div(&b));
}

#[test]
fn test_ops_neg() {
    let a = rfn(&[1, 1], &[-1, 1]);
    let neg = -&a;
    assert_eq!(neg, a.neg());
}

#[test]
fn test_ops_owned() {
    let a = rfn(&[1], &[-1, 1]);
    let b = rfn(&[1], &[1, 1]);
    let expected = a.add(&b);
    let sum = a + b;
    assert_eq!(sum, expected);
}

// ── Evaluation ──────────────────────────────────────────────────────────────

#[test]
fn test_eval() {
    // (x+1)/(x-1) at x=3 → 4/2 = 2
    let f = rfn(&[1, 1], &[-1, 1]);
    let result = f.eval(&BigRational::from(3i64));
    assert_eq!(result, Some(BigRational::from(2i64)));
}

#[test]
fn test_eval_at_pole() {
    // 1/(x-1) at x=1 → None
    let f = rfn(&[1], &[-1, 1]);
    assert_eq!(f.eval(&BigRational::from(1i64)), None);
}

// ── Equality ────────────────────────────────────────────────────────────────

#[test]
fn test_equality() {
    let a = rfn(&[-1, 0, 1], &[-1, 1]); // (x^2-1)/(x-1) = x+1
    let b = RFn::from_poly(ri(&[1, 1])); // x+1
    assert_eq!(a, b);
}

#[test]
fn test_inequality() {
    let a = rfn(&[1], &[-1, 1]);
    let b = rfn(&[1], &[1, 1]);
    assert_ne!(a, b);
}

// ── Display ─────────────────────────────────────────────────────────────────

#[test]
fn test_display_polynomial() {
    let f = RFn::from_poly(ri(&[1, 2]));
    let s = f.to_string();
    assert!(!s.contains('/'));
}

#[test]
fn test_display_fraction() {
    let f = rfn(&[1], &[-1, 1]);
    let s = f.to_string();
    assert!(s.contains('/'));
}

// ── Recip ────────────────────────────────────────────────────────────────────

#[test]
fn test_recip() {
    let f = rfn(&[1, 1], &[-1, 1]);
    let r = f.recip();
    assert_eq!(*r.numerator(), ri(&[-1, 1]));
    assert_eq!(*r.denominator(), ri(&[1, 1]));
}

#[test]
#[should_panic(expected = "reciprocal of zero")]
fn test_recip_zero_panics() {
    let _ = RFn::zero().recip();
}

// ── Neg ──────────────────────────────────────────────────────────────────────

#[test]
fn test_neg() {
    let f = rfn(&[1, 1], &[-1, 1]);
    let n = f.neg();
    assert_eq!(*n.numerator(), ri(&[-1, -1]));
    assert_eq!(*n.denominator(), ri(&[-1, 1]));
}

// ── Coefficient size doesn't explode ────────────────────────────────────────

#[test]
fn test_large_fraction_arithmetic_stays_small() {
    // Build (1/(x-1) + 1/(x-2) + 1/(x-3) + 1/(x-4)) and verify reduction
    let mut sum = RFn::zero();
    for k in 1..=4 {
        let term = rfn(&[1], &[-(k as i64), 1]);
        sum = sum.add(&term);
    }
    assert_eq!(sum.denominator().degree(), Some(4));
    assert_eq!(sum.numerator().degree(), Some(3));
}
