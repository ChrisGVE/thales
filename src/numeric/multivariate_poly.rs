//! Multivariate polynomial with `BTreeMap<Monomial, R>` representation.
//!
//! [`MultivariatePolynomial<R>`] represents a polynomial in arbitrarily
//! many variables with coefficients in a ring `R`. Monomials are ordered
//! by graded reverse lexicographic (grevlex) order by default.

use super::ring::Ring;
use super::SymbolId;
use std::collections::BTreeMap;
use std::fmt;
use std::ops;

/// A monomial: a product of variables raised to non-negative integer powers.
///
/// Represented as a sorted map from variable to exponent.
/// Only stores non-zero exponents. The empty monomial is the constant `1`.
///
/// Ordering is graded reverse lexicographic (grevlex):
/// first by total degree (descending), then by variable order (reversed).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Monomial {
    /// Map from variable to exponent (only non-zero exponents stored).
    vars: BTreeMap<SymbolId, u32>,
    /// Cached total degree for fast comparison.
    total_deg: u32,
}

impl Monomial {
    /// The constant monomial (degree 0).
    pub fn one() -> Self {
        Monomial {
            vars: BTreeMap::new(),
            total_deg: 0,
        }
    }

    /// A single variable monomial: `x^1`.
    pub fn var(id: SymbolId) -> Self {
        let mut vars = BTreeMap::new();
        vars.insert(id, 1);
        Monomial { vars, total_deg: 1 }
    }

    /// A single variable raised to a power: `x^exp`.
    pub fn var_pow(id: SymbolId, exp: u32) -> Self {
        if exp == 0 {
            return Self::one();
        }
        let mut vars = BTreeMap::new();
        vars.insert(id, exp);
        Monomial {
            vars,
            total_deg: exp,
        }
    }

    /// Create from a map of variable exponents.
    pub fn from_vars(vars: BTreeMap<SymbolId, u32>) -> Self {
        let total_deg = vars.values().sum();
        let vars: BTreeMap<_, _> = vars.into_iter().filter(|(_, e)| *e > 0).collect();
        Monomial { vars, total_deg }
    }

    /// Total degree of the monomial.
    pub fn total_degree(&self) -> u32 {
        self.total_deg
    }

    /// Exponent of a specific variable (0 if absent).
    pub fn exponent(&self, var: &SymbolId) -> u32 {
        self.vars.get(var).copied().unwrap_or(0)
    }

    /// Iterator over (variable, exponent) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&SymbolId, &u32)> {
        self.vars.iter()
    }

    /// Number of distinct variables in this monomial.
    pub fn num_vars(&self) -> usize {
        self.vars.len()
    }

    /// Whether this is the constant monomial (degree 0).
    pub fn is_constant(&self) -> bool {
        self.total_deg == 0
    }

    /// Multiply two monomials.
    pub fn mul(&self, other: &Self) -> Self {
        let mut vars = self.vars.clone();
        for (&id, &exp) in &other.vars {
            *vars.entry(id).or_insert(0) += exp;
        }
        Monomial {
            total_deg: self.total_deg + other.total_deg,
            vars,
        }
    }

    /// Check if `self` is divisible by `other` (all exponents ≥).
    pub fn is_divisible_by(&self, other: &Self) -> bool {
        for (&id, &exp) in &other.vars {
            if self.exponent(&id) < exp {
                return false;
            }
        }
        true
    }

    /// Divide `self` by `other`. Returns `None` if not divisible.
    pub fn checked_div(&self, other: &Self) -> Option<Self> {
        let mut vars = self.vars.clone();
        for (&id, &exp) in &other.vars {
            let e = vars.get(&id).copied().unwrap_or(0);
            if e < exp {
                return None;
            }
            let new_e = e - exp;
            if new_e == 0 {
                vars.remove(&id);
            } else {
                vars.insert(id, new_e);
            }
        }
        Some(Monomial {
            total_deg: self.total_deg - other.total_deg,
            vars,
        })
    }

    /// Set of variables appearing in this monomial.
    pub fn variables(&self) -> Vec<SymbolId> {
        self.vars.keys().copied().collect()
    }
}

// ── Grevlex ordering ────────────────────────────────────────────────────────

impl PartialOrd for Monomial {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Monomial {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        // Grevlex: higher total degree first
        match self.total_deg.cmp(&other.total_deg) {
            Equal => {
                // For equal total degree, reverse lexicographic:
                // compare variables in reverse order; the monomial with
                // a SMALLER exponent on the LAST differing variable is greater.
                let all_vars = self.vars.keys().chain(other.vars.keys()).copied();
                let mut vars: Vec<_> = all_vars.collect();
                vars.sort();
                vars.dedup();
                // Reverse iterate for grevlex
                for &v in vars.iter().rev() {
                    let a = self.exponent(&v);
                    let b = other.exponent(&v);
                    match a.cmp(&b) {
                        Equal => continue,
                        // In grevlex, smaller exponent on last variable = greater
                        Less => return Greater,
                        Greater => return Less,
                    }
                }
                Equal
            }
            other_ord => other_ord,
        }
    }
}

impl fmt::Display for Monomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_constant() {
            return write!(f, "1");
        }
        let mut first = true;
        for (&id, &exp) in &self.vars {
            if !first {
                write!(f, "*")?;
            }
            if exp == 1 {
                write!(f, "{id}")?;
            } else {
                write!(f, "{id}^{exp}")?;
            }
            first = false;
        }
        Ok(())
    }
}

// ── MultivariatePolynomial ──────────────────────────────────────────────────

/// A multivariate polynomial over a ring `R`.
///
/// Stored as a map from monomials to coefficients. Zero coefficients
/// are never stored. The zero polynomial has an empty map.
#[derive(Clone, Debug)]
pub struct MultivariatePolynomial<R: Ring> {
    terms: BTreeMap<Monomial, R>,
}

impl<R: Ring> MultivariatePolynomial<R> {
    /// The zero polynomial.
    pub fn zero() -> Self {
        MultivariatePolynomial {
            terms: BTreeMap::new(),
        }
    }

    /// A constant polynomial.
    pub fn constant(c: R) -> Self {
        if c.is_zero() {
            return Self::zero();
        }
        let mut terms = BTreeMap::new();
        terms.insert(Monomial::one(), c);
        MultivariatePolynomial { terms }
    }

    /// A single variable polynomial: `c * x^exp`.
    pub fn monomial(coeff: R, mono: Monomial) -> Self {
        if coeff.is_zero() {
            return Self::zero();
        }
        let mut terms = BTreeMap::new();
        terms.insert(mono, coeff);
        MultivariatePolynomial { terms }
    }

    /// A single variable: `x` (coefficient 1).
    pub fn var(id: SymbolId) -> Self {
        Self::monomial(R::one(), Monomial::var(id))
    }

    /// Number of non-zero terms.
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    /// Whether this is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Whether this is a constant (degree 0 or zero).
    pub fn is_constant(&self) -> bool {
        self.terms.is_empty()
            || (self.terms.len() == 1 && self.terms.keys().next().unwrap().is_constant())
    }

    /// Total degree of the polynomial (max monomial degree), or None if zero.
    pub fn total_degree(&self) -> Option<u32> {
        self.terms.keys().map(|m| m.total_degree()).max()
    }

    /// Leading monomial (highest in grevlex order).
    pub fn leading_monomial(&self) -> Option<&Monomial> {
        self.terms.keys().next_back()
    }

    /// Leading coefficient.
    pub fn leading_coeff(&self) -> Option<&R> {
        self.terms.values().next_back()
    }

    /// Leading term (monomial, coefficient).
    pub fn leading_term(&self) -> Option<(&Monomial, &R)> {
        self.terms.iter().next_back()
    }

    /// Get coefficient of a specific monomial.
    pub fn coeff(&self, mono: &Monomial) -> R {
        self.terms.get(mono).cloned().unwrap_or_else(R::zero)
    }

    /// Get the constant term.
    pub fn constant_term(&self) -> R {
        self.coeff(&Monomial::one())
    }

    /// Iterator over (monomial, coefficient) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&Monomial, &R)> {
        self.terms.iter()
    }

    /// Set of all variables appearing in the polynomial.
    pub fn variables(&self) -> Vec<SymbolId> {
        let mut vars: Vec<_> = self.terms.keys().flat_map(|m| m.variables()).collect();
        vars.sort();
        vars.dedup();
        vars
    }

    /// Add a term, combining with existing if monomial matches.
    pub fn add_term(&mut self, mono: Monomial, coeff: R) {
        if coeff.is_zero() {
            return;
        }
        if let Some(existing) = self.terms.get(&mono) {
            let sum = existing.clone() + coeff;
            if sum.is_zero() {
                self.terms.remove(&mono);
            } else {
                self.terms.insert(mono, sum);
            }
        } else {
            self.terms.insert(mono, coeff);
        }
    }

    /// Negate the polynomial.
    pub fn neg(&self) -> Self {
        let terms = self
            .terms
            .iter()
            .map(|(m, c)| (m.clone(), -c.clone()))
            .collect();
        MultivariatePolynomial { terms }
    }

    /// Scale by a constant.
    pub fn scale(&self, c: &R) -> Self {
        if c.is_zero() {
            return Self::zero();
        }
        let terms = self
            .terms
            .iter()
            .filter_map(|(m, coeff)| {
                let scaled = coeff.clone() * c.clone();
                if scaled.is_zero() {
                    None
                } else {
                    Some((m.clone(), scaled))
                }
            })
            .collect();
        MultivariatePolynomial { terms }
    }

    /// View as univariate in one variable, with multivariate coefficients.
    ///
    /// Returns coefficients indexed by degree: `[a_0, a_1, ..., a_n]`
    /// where `self = a_0 + a_1*x + ... + a_n*x^n` and each `a_i` is a
    /// multivariate polynomial in the remaining variables.
    pub fn as_univariate(&self, var: SymbolId) -> Vec<MultivariatePolynomial<R>> {
        let max_deg = self
            .terms
            .keys()
            .map(|m| m.exponent(&var))
            .max()
            .unwrap_or(0);

        let mut coeffs = vec![MultivariatePolynomial::zero(); (max_deg + 1) as usize];

        for (mono, coeff) in &self.terms {
            let deg = mono.exponent(&var);
            // Build monomial without var
            let mut reduced_vars = mono.vars.clone();
            reduced_vars.remove(&var);
            let reduced_mono = Monomial::from_vars(reduced_vars);
            coeffs[deg as usize].add_term(reduced_mono, coeff.clone());
        }

        coeffs
    }

    /// Evaluate at a specific value for one variable, returning a new
    /// multivariate polynomial with that variable eliminated.
    pub fn eval_var(&self, var: SymbolId, val: &R) -> Self {
        let mut result = Self::zero();
        for (mono, coeff) in &self.terms {
            let exp = mono.exponent(&var);
            // Compute val^exp
            let mut power = R::one();
            for _ in 0..exp {
                power = power * val.clone();
            }
            let new_coeff = coeff.clone() * power;
            // Build monomial without var
            let mut reduced_vars = mono.vars.clone();
            reduced_vars.remove(&var);
            let reduced_mono = Monomial::from_vars(reduced_vars);
            result.add_term(reduced_mono, new_coeff);
        }
        result
    }
}

// ── Equality ────────────────────────────────────────────────────────────────

impl<R: Ring> PartialEq for MultivariatePolynomial<R> {
    fn eq(&self, other: &Self) -> bool {
        self.terms == other.terms
    }
}

impl<R: Ring> Eq for MultivariatePolynomial<R> {}

// ── Arithmetic ──────────────────────────────────────────────────────────────

impl<R: Ring> ops::Add for &MultivariatePolynomial<R> {
    type Output = MultivariatePolynomial<R>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = self.clone();
        for (mono, coeff) in &rhs.terms {
            result.add_term(mono.clone(), coeff.clone());
        }
        result
    }
}

impl<R: Ring> ops::Sub for &MultivariatePolynomial<R> {
    type Output = MultivariatePolynomial<R>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = self.clone();
        for (mono, coeff) in &rhs.terms {
            result.add_term(mono.clone(), -coeff.clone());
        }
        result
    }
}

impl<R: Ring> ops::Mul for &MultivariatePolynomial<R> {
    type Output = MultivariatePolynomial<R>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut result = MultivariatePolynomial::zero();
        for (m1, c1) in &self.terms {
            for (m2, c2) in &rhs.terms {
                let mono = m1.mul(m2);
                let coeff = c1.clone() * c2.clone();
                result.add_term(mono, coeff);
            }
        }
        result
    }
}

impl<R: Ring> ops::Neg for &MultivariatePolynomial<R> {
    type Output = MultivariatePolynomial<R>;

    fn neg(self) -> Self::Output {
        self.neg()
    }
}

// ── Display ─────────────────────────────────────────────────────────────────

impl<R: Ring> fmt::Display for MultivariatePolynomial<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        // Display in descending monomial order
        let terms: Vec<_> = self.terms.iter().rev().collect();
        for (i, (mono, coeff)) in terms.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            if mono.is_constant() {
                write!(f, "{coeff}")?;
            } else if coeff.is_one() {
                write!(f, "{mono}")?;
            } else {
                write!(f, "{coeff}*{mono}")?;
            }
        }
        Ok(())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::BigRational;

    type MP = MultivariatePolynomial<BigRational>;

    fn int(n: i64) -> BigRational {
        BigRational::from(n)
    }

    fn x() -> SymbolId {
        SymbolId::intern("mv_x")
    }

    fn y() -> SymbolId {
        SymbolId::intern("mv_y")
    }

    fn z() -> SymbolId {
        SymbolId::intern("mv_z")
    }

    #[test]
    fn test_zero() {
        let p = MP::zero();
        assert!(p.is_zero());
        assert_eq!(p.term_count(), 0);
        assert_eq!(p.total_degree(), None);
    }

    #[test]
    fn test_constant() {
        let p = MP::constant(int(5));
        assert!(p.is_constant());
        assert_eq!(p.total_degree(), Some(0));
        assert_eq!(p.constant_term(), int(5));
    }

    #[test]
    fn test_single_var() {
        let p = MP::var(x());
        assert!(!p.is_zero());
        assert_eq!(p.total_degree(), Some(1));
        assert_eq!(p.term_count(), 1);
    }

    #[test]
    fn test_monomial_mul() {
        let m1 = Monomial::var(x());
        let m2 = Monomial::var(y());
        let prod = m1.mul(&m2);
        assert_eq!(prod.total_degree(), 2);
        assert_eq!(prod.exponent(&x()), 1);
        assert_eq!(prod.exponent(&y()), 1);
    }

    #[test]
    fn test_monomial_grevlex_order() {
        // x^2 > xy > y^2 > x > y > 1 in grevlex
        let x2 = Monomial::var_pow(x(), 2);
        let xy = Monomial::var(x()).mul(&Monomial::var(y()));
        let y2 = Monomial::var_pow(y(), 2);
        let xm = Monomial::var(x());
        let ym = Monomial::var(y());
        let one = Monomial::one();

        assert!(x2 > xy || x2 > y2); // Same total degree, grevlex ordering
        assert!(x2 > xm);
        assert!(xm > one);
        assert!(ym > one);
    }

    #[test]
    fn test_polynomial_add() {
        // (x + y) + (x + 1) = 2x + y + 1
        let p1 = &MP::var(x()) + &MP::var(y());
        let p2 = &MP::var(x()) + &MP::constant(int(1));
        let sum = &p1 + &p2;
        assert_eq!(sum.term_count(), 3); // 2x, y, 1
        assert_eq!(sum.coeff(&Monomial::var(x())), int(2));
        assert_eq!(sum.coeff(&Monomial::var(y())), int(1));
        assert_eq!(sum.constant_term(), int(1));
    }

    #[test]
    fn test_polynomial_sub() {
        let p = MP::var(x());
        let result = &p - &p;
        assert!(result.is_zero());
    }

    #[test]
    fn test_polynomial_mul() {
        // (x + 1)(y + 1) = xy + x + y + 1
        let p1 = &MP::var(x()) + &MP::constant(int(1));
        let p2 = &MP::var(y()) + &MP::constant(int(1));
        let prod = &p1 * &p2;
        assert_eq!(prod.term_count(), 4);
        let xy = Monomial::var(x()).mul(&Monomial::var(y()));
        assert_eq!(prod.coeff(&xy), int(1));
        assert_eq!(prod.coeff(&Monomial::var(x())), int(1));
        assert_eq!(prod.coeff(&Monomial::var(y())), int(1));
        assert_eq!(prod.constant_term(), int(1));
    }

    #[test]
    fn test_as_univariate() {
        // xy + x + y + 1 viewed in x → (y+1)x + (y+1) = [y+1, y+1]
        // Actually: coefficients of x^0 = y+1, x^1 = y+1
        let xy = MP::monomial(int(1), Monomial::var(x()).mul(&Monomial::var(y())));
        let p = &(&(&xy + &MP::var(x())) + &MP::var(y())) + &MP::constant(int(1));

        let coeffs = p.as_univariate(x());
        assert_eq!(coeffs.len(), 2); // degree 0 and degree 1 in x

        // coeff of x^0: y + 1
        assert_eq!(coeffs[0].term_count(), 2);
        assert_eq!(coeffs[0].coeff(&Monomial::var(y())), int(1));
        assert_eq!(coeffs[0].constant_term(), int(1));

        // coeff of x^1: y + 1
        assert_eq!(coeffs[1].term_count(), 2);
        assert_eq!(coeffs[1].coeff(&Monomial::var(y())), int(1));
        assert_eq!(coeffs[1].constant_term(), int(1));
    }

    #[test]
    fn test_eval_var() {
        // p = 2xy + 3x + y + 5, evaluate at x=2 → 4y + 6 + y + 5 = 5y + 11
        let txy = MP::monomial(int(2), Monomial::var(x()).mul(&Monomial::var(y())));
        let tx = MP::monomial(int(3), Monomial::var(x()));
        let ty = MP::var(y());
        let tc = MP::constant(int(5));
        let p = &(&(&txy + &tx) + &ty) + &tc;

        let result = p.eval_var(x(), &int(2));
        // 2*2*y + 3*2 + y + 5 = 4y + 6 + y + 5 = 5y + 11
        assert_eq!(result.coeff(&Monomial::var(y())), int(5));
        assert_eq!(result.constant_term(), int(11));
        assert_eq!(result.term_count(), 2);
    }

    #[test]
    fn test_variables() {
        let xy = MP::monomial(int(1), Monomial::var(x()).mul(&Monomial::var(y())));
        let p = &xy + &MP::var(z());
        let vars = p.variables();
        assert_eq!(vars.len(), 3);
    }

    #[test]
    fn test_monomial_divisibility() {
        let xy = Monomial::var(x()).mul(&Monomial::var(y()));
        let xm = Monomial::var(x());
        assert!(xy.is_divisible_by(&xm));
        assert!(!xm.is_divisible_by(&xy));

        let div = xy.checked_div(&xm).unwrap();
        assert_eq!(div.exponent(&y()), 1);
        assert_eq!(div.exponent(&x()), 0);
    }

    #[test]
    fn test_display() {
        let p = &MP::var(x()) + &MP::constant(int(1));
        let s = p.to_string();
        assert!(s.contains("mv_x"));
        assert!(s.contains("1"));
    }

    #[test]
    fn test_leading_term() {
        // 3x^2 + 2xy + y^2 — leading in grevlex should be one of the degree-2 terms
        let x2 = MP::monomial(int(3), Monomial::var_pow(x(), 2));
        let xy = MP::monomial(int(2), Monomial::var(x()).mul(&Monomial::var(y())));
        let y2 = MP::monomial(int(1), Monomial::var_pow(y(), 2));
        let p = &(&x2 + &xy) + &y2;
        assert_eq!(p.total_degree(), Some(2));
        assert!(p.leading_coeff().is_some());
    }

    #[test]
    fn test_scale() {
        let p = &MP::var(x()) + &MP::constant(int(1));
        let scaled = p.scale(&int(3));
        assert_eq!(scaled.coeff(&Monomial::var(x())), int(3));
        assert_eq!(scaled.constant_term(), int(3));
    }

    #[test]
    fn test_scale_by_zero() {
        let p = &MP::var(x()) + &MP::constant(int(1));
        let scaled = p.scale(&int(0));
        assert!(scaled.is_zero());
    }
}
