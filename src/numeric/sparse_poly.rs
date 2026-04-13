//! Sparse univariate polynomial with generic coefficient ring.
//!
//! [`SparsePolynomial<R>`] stores coefficients in a `BTreeMap<usize, R>`
//! where keys are exponent degrees. Efficient for polynomials with many
//! zero coefficients (e.g., `x^1000 + 1`).

use super::dense_poly::DensePolynomial;
use super::ring::{Field, Ring};
use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// A sparse univariate polynomial over a ring `R`.
///
/// Coefficients stored as `BTreeMap<usize, R>` where key = degree.
/// Zero coefficients are never stored.
///
/// # Invariant
///
/// Every stored coefficient is non-zero.
#[derive(Clone, Debug)]
pub struct SparsePolynomial<R: Ring> {
    terms: BTreeMap<usize, R>,
}

impl<R: Ring> SparsePolynomial<R> {
    /// The zero polynomial.
    pub fn zero() -> Self {
        SparsePolynomial {
            terms: BTreeMap::new(),
        }
    }

    /// A constant polynomial.
    pub fn constant(c: R) -> Self {
        if c.is_zero() {
            Self::zero()
        } else {
            let mut terms = BTreeMap::new();
            terms.insert(0, c);
            SparsePolynomial { terms }
        }
    }

    /// The polynomial `x`.
    pub fn x() -> Self {
        let mut terms = BTreeMap::new();
        terms.insert(1, R::one());
        SparsePolynomial { terms }
    }

    /// Create a monomial `c * x^deg`.
    pub fn monomial(c: R, deg: usize) -> Self {
        if c.is_zero() {
            Self::zero()
        } else {
            let mut terms = BTreeMap::new();
            terms.insert(deg, c);
            SparsePolynomial { terms }
        }
    }

    /// Create from a map of (degree, coefficient) pairs.
    pub fn from_terms(terms: BTreeMap<usize, R>) -> Self {
        let terms = terms.into_iter().filter(|(_, c)| !c.is_zero()).collect();
        SparsePolynomial { terms }
    }

    /// Returns `true` if this is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Degree of the polynomial. Returns `None` for the zero polynomial.
    pub fn degree(&self) -> Option<usize> {
        self.terms.keys().next_back().copied()
    }

    /// Leading coefficient. Returns `None` for the zero polynomial.
    pub fn leading_coeff(&self) -> Option<&R> {
        self.terms.values().next_back()
    }

    /// Coefficient of `x^i`. Returns zero if not stored.
    pub fn coeff(&self, i: usize) -> R {
        self.terms.get(&i).cloned().unwrap_or_else(R::zero)
    }

    /// Number of non-zero terms.
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    /// Reference to the internal term map.
    pub fn terms(&self) -> &BTreeMap<usize, R> {
        &self.terms
    }

    /// Set a coefficient. Removes if zero.
    pub fn set_coeff(&mut self, deg: usize, c: R) {
        if c.is_zero() {
            self.terms.remove(&deg);
        } else {
            self.terms.insert(deg, c);
        }
    }

    /// Evaluate using Horner-like scheme over sparse terms.
    pub fn eval(&self, x: &R) -> R {
        if self.terms.is_empty() {
            return R::zero();
        }
        let mut result = R::zero();
        let mut prev_deg = None;
        // Iterate from highest to lowest degree
        for (&deg, coeff) in self.terms.iter().rev() {
            match prev_deg {
                None => {
                    result = coeff.clone();
                }
                Some(pd) => {
                    // Multiply by x^(pd - deg) to bridge the gap
                    for _ in 0..(pd - deg) {
                        result = result * x.clone();
                    }
                    result = result + coeff.clone();
                }
            }
            prev_deg = Some(deg);
        }
        // Multiply by x^lowest_degree
        if let Some(lowest) = prev_deg {
            for _ in 0..lowest {
                result = result * x.clone();
            }
        }
        result
    }

    /// Scale all coefficients by a constant.
    pub fn scale(&self, c: &R) -> Self {
        if c.is_zero() {
            return Self::zero();
        }
        let terms = self
            .terms
            .iter()
            .map(|(&deg, coeff)| (deg, coeff.clone() * c.clone()))
            .filter(|(_, c)| !c.is_zero())
            .collect();
        SparsePolynomial { terms }
    }

    /// Convert to dense representation.
    pub fn to_dense(&self) -> DensePolynomial<R> {
        match self.degree() {
            None => DensePolynomial::zero(),
            Some(deg) => {
                let mut coeffs = Vec::with_capacity(deg + 1);
                for i in 0..=deg {
                    coeffs.push(self.coeff(i));
                }
                DensePolynomial::from_coeffs(coeffs)
            }
        }
    }
}

/// Sparsity ratio: fraction of zero coefficients. Returns 0.0 for zero polynomial.
impl<R: Ring> SparsePolynomial<R> {
    pub fn sparsity(&self) -> f64 {
        match self.degree() {
            None => 0.0,
            Some(deg) => {
                let total = deg + 1;
                let nonzero = self.terms.len();
                1.0 - (nonzero as f64 / total as f64)
            }
        }
    }
}

// ── Euclidean division (requires Field) ──────────────────────────────────────

impl<R: Field> SparsePolynomial<R> {
    /// Euclidean division via dense delegation.
    ///
    /// Converts to dense, performs division, converts back.
    ///
    /// # Panics
    ///
    /// Panics if `divisor` is zero.
    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        let (q, r) = self.to_dense().div_rem(&divisor.to_dense());
        (SparsePolynomial::from(q), SparsePolynomial::from(r))
    }

    /// Make the polynomial monic (leading coefficient = 1).
    pub fn make_monic(&self) -> Self {
        SparsePolynomial::from(self.to_dense().make_monic())
    }
}

/// Convert a DensePolynomial to sparse.
impl<R: Ring> From<DensePolynomial<R>> for SparsePolynomial<R> {
    fn from(dense: DensePolynomial<R>) -> Self {
        let terms = dense
            .coefficients()
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.is_zero())
            .map(|(i, c)| (i, c.clone()))
            .collect();
        SparsePolynomial { terms }
    }
}

// ── Equality ─────────────────────────────────────────────────────────────────

impl<R: Ring> PartialEq for SparsePolynomial<R> {
    fn eq(&self, other: &Self) -> bool {
        self.terms == other.terms
    }
}

impl<R: Ring> Eq for SparsePolynomial<R> {}

// ── Add ──────────────────────────────────────────────────────────────────────

impl<R: Ring> Add for SparsePolynomial<R> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut result = self.terms;
        for (deg, coeff) in rhs.terms {
            let entry = result.entry(deg).or_insert_with(R::zero);
            *entry = entry.clone() + coeff;
        }
        result.retain(|_, c| !c.is_zero());
        SparsePolynomial { terms: result }
    }
}

// ── Sub ──────────────────────────────────────────────────────────────────────

impl<R: Ring> Sub for SparsePolynomial<R> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let mut result = self.terms;
        for (deg, coeff) in rhs.terms {
            let entry = result.entry(deg).or_insert_with(R::zero);
            *entry = entry.clone() - coeff;
        }
        result.retain(|_, c| !c.is_zero());
        SparsePolynomial { terms: result }
    }
}

// ── Neg ──────────────────────────────────────────────────────────────────────

impl<R: Ring> Neg for SparsePolynomial<R> {
    type Output = Self;

    fn neg(self) -> Self {
        let terms = self.terms.into_iter().map(|(d, c)| (d, -c)).collect();
        SparsePolynomial { terms }
    }
}

// ── Mul ──────────────────────────────────────────────────────────────────────

impl<R: Ring> Mul for SparsePolynomial<R> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        if self.is_zero() || rhs.is_zero() {
            return SparsePolynomial::zero();
        }
        let mut result = BTreeMap::new();
        for (&da, ca) in &self.terms {
            for (&db, cb) in &rhs.terms {
                let deg = da + db;
                let prod = ca.clone() * cb.clone();
                let entry = result.entry(deg).or_insert_with(R::zero);
                *entry = entry.clone() + prod;
            }
        }
        result.retain(|_, c| !c.is_zero());
        SparsePolynomial { terms: result }
    }
}

// ── Display ──────────────────────────────────────────────────────────────────

impl<R: Ring> fmt::Display for SparsePolynomial<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        let mut first = true;
        for (&deg, coeff) in self.terms.iter().rev() {
            if !first {
                write!(f, " + ")?;
            }
            match deg {
                0 => write!(f, "{coeff}")?,
                1 if coeff.is_one() => write!(f, "x")?,
                1 => write!(f, "{coeff}*x")?,
                _ if coeff.is_one() => write!(f, "x^{deg}")?,
                _ => write!(f, "{coeff}*x^{deg}")?,
            }
            first = false;
        }
        Ok(())
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::SmallInt;

    type Poly = SparsePolynomial<SmallInt>;

    fn sp(pairs: &[(usize, i64)]) -> Poly {
        let terms = pairs.iter().map(|&(d, c)| (d, SmallInt::from(c))).collect();
        Poly::from_terms(terms)
    }

    #[test]
    fn test_zero() {
        let z = Poly::zero();
        assert!(z.is_zero());
        assert_eq!(z.degree(), None);
    }

    #[test]
    fn test_constant() {
        let c = Poly::constant(SmallInt::from(5i64));
        assert_eq!(c.degree(), Some(0));
        assert_eq!(c.coeff(0), SmallInt::from(5i64));
    }

    #[test]
    fn test_monomial_high_degree() {
        // x^1000 + 1
        let p = sp(&[(0, 1), (1000, 1)]);
        assert_eq!(p.degree(), Some(1000));
        assert_eq!(p.term_count(), 2);
        assert!(p.sparsity() > 0.99);
    }

    #[test]
    fn test_add() {
        let a = sp(&[(0, 1), (2, 1)]); // 1 + x^2
        let b = sp(&[(1, 2)]); // 2x
        let sum = a + b;
        assert_eq!(sum.coeff(0), SmallInt::from(1i64));
        assert_eq!(sum.coeff(1), SmallInt::from(2i64));
        assert_eq!(sum.coeff(2), SmallInt::from(1i64));
    }

    #[test]
    fn test_add_cancellation() {
        let a = sp(&[(0, 1), (1, 1)]);
        let b = sp(&[(0, -1), (1, -1)]);
        assert!((a + b).is_zero());
    }

    #[test]
    fn test_sub() {
        let a = sp(&[(0, 3), (1, 2)]);
        let b = sp(&[(0, 1), (1, 1)]);
        let diff = a - b;
        assert_eq!(diff.coeff(0), SmallInt::from(2i64));
        assert_eq!(diff.coeff(1), SmallInt::from(1i64));
    }

    #[test]
    fn test_neg() {
        let p = sp(&[(0, 1), (3, -2)]);
        let neg = -p;
        assert_eq!(neg.coeff(0), SmallInt::from(-1i64));
        assert_eq!(neg.coeff(3), SmallInt::from(2i64));
    }

    #[test]
    fn test_mul() {
        // (1 + x) * (1 - x) = 1 - x^2
        let a = sp(&[(0, 1), (1, 1)]);
        let b = sp(&[(0, 1), (1, -1)]);
        let prod = a * b;
        assert_eq!(prod.coeff(0), SmallInt::from(1i64));
        assert_eq!(prod.coeff(2), SmallInt::from(-1i64));
        assert_eq!(prod.term_count(), 2); // x^1 cancelled
    }

    #[test]
    fn test_mul_high_degree() {
        // x^500 * x^500 = x^1000
        let a = Poly::monomial(SmallInt::from(1i64), 500);
        let b = Poly::monomial(SmallInt::from(1i64), 500);
        let prod = a * b;
        assert_eq!(prod.degree(), Some(1000));
        assert_eq!(prod.term_count(), 1);
    }

    #[test]
    fn test_eval() {
        // x^2 + 1, at x=3: 10
        let p = sp(&[(0, 1), (2, 1)]);
        assert_eq!(p.eval(&SmallInt::from(3i64)), SmallInt::from(10i64));
    }

    #[test]
    fn test_eval_zero() {
        assert_eq!(
            Poly::zero().eval(&SmallInt::from(5i64)),
            SmallInt::from(0i64)
        );
    }

    #[test]
    fn test_scale() {
        let p = sp(&[(0, 1), (3, 2)]);
        let scaled = p.scale(&SmallInt::from(3i64));
        assert_eq!(scaled.coeff(0), SmallInt::from(3i64));
        assert_eq!(scaled.coeff(3), SmallInt::from(6i64));
    }

    #[test]
    fn test_to_dense() {
        let s = sp(&[(0, 1), (2, 3)]);
        let d = s.to_dense();
        assert_eq!(d.degree(), Some(2));
        assert_eq!(d.coeff(0), SmallInt::from(1i64));
        assert_eq!(d.coeff(1), SmallInt::from(0i64));
        assert_eq!(d.coeff(2), SmallInt::from(3i64));
    }

    #[test]
    fn test_from_dense() {
        let d = DensePolynomial::from_coeffs(vec![
            SmallInt::from(1i64),
            SmallInt::from(0i64),
            SmallInt::from(3i64),
        ]);
        let s = SparsePolynomial::from(d);
        assert_eq!(s.term_count(), 2);
        assert_eq!(s.coeff(0), SmallInt::from(1i64));
        assert_eq!(s.coeff(2), SmallInt::from(3i64));
    }

    #[test]
    fn test_display() {
        let p = sp(&[(0, 1), (1000, 1)]);
        let s = p.to_string();
        assert!(s.contains("x^1000"));
        assert!(s.contains("1"));
    }

    #[test]
    fn test_equality() {
        let a = sp(&[(0, 1), (2, 3)]);
        let b = sp(&[(0, 1), (2, 3)]);
        assert_eq!(a, b);
    }

    // ── Euclidean division tests ─────────────────────────────────────────────

    mod div_rem_tests {
        use super::*;
        use crate::numeric::BigRational;
        type RPoly = SparsePolynomial<BigRational>;

        fn rsp(pairs: &[(usize, i64)]) -> RPoly {
            let terms = pairs
                .iter()
                .map(|&(d, c)| (d, BigRational::from(c)))
                .collect();
            RPoly::from_terms(terms)
        }

        #[test]
        fn test_div_rem_exact() {
            // (x^3 + 1) / (x + 1) = x^2 - x + 1
            let f = rsp(&[(0, 1), (3, 1)]);
            let g = rsp(&[(0, 1), (1, 1)]);
            let (q, r) = f.div_rem(&g);
            assert_eq!(q, rsp(&[(0, 1), (1, -1), (2, 1)]));
            assert!(r.is_zero());
        }

        #[test]
        fn test_div_rem_identity() {
            let f = rsp(&[(0, 3), (1, 2), (3, 1)]);
            let g = rsp(&[(0, 1), (1, 1)]);
            let (q, r) = f.div_rem(&g);
            let reconstructed = q.to_dense() * g.to_dense() + r.to_dense();
            assert_eq!(reconstructed, f.to_dense());
        }

        #[test]
        #[should_panic(expected = "polynomial division by zero")]
        fn test_div_rem_zero_divisor() {
            let f = rsp(&[(0, 1)]);
            let g = RPoly::zero();
            let _ = f.div_rem(&g);
        }
    }
}
