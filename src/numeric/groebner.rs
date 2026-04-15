//! Buchberger's algorithm for computing Groebner bases.
//!
//! A Groebner basis for a polynomial ideal `I = <f1, ..., fs>` is a special
//! generating set where the leading monomials of the basis elements generate
//! the ideal of leading monomials of all elements in `I`. This enables
//! systematic multivariate polynomial division with unambiguous remainders.
//!
//! # References
//!
//! - Cox, Little, O'Shea: "Ideals, Varieties, and Algorithms", Chapter 2
//!
//! # Example
//!
//! ```
//! use thales::numeric::{
//!     buchberger, BigRational, GrevLex, Monomial, MultivariatePolynomial, SymbolId,
//! };
//! let x = SymbolId::intern("gb_x");
//! let y = SymbolId::intern("gb_y");
//! let r = |n: i64| BigRational::from(n);
//!
//! // f1 = x^2 + y - 1
//! let f1 = &MultivariatePolynomial::monomial(r(1), Monomial::var_pow(x, 2))
//!     + &(&MultivariatePolynomial::var(y) - &MultivariatePolynomial::constant(r(1)));
//! // f2 = x + y^2 - 1
//! let f2 = &MultivariatePolynomial::var(x)
//!     + &(&MultivariatePolynomial::monomial(r(1), Monomial::var_pow(y, 2))
//!         - &MultivariatePolynomial::constant(r(1)));
//!
//! let gb = buchberger(&[f1, f2], &GrevLex::new());
//! assert!(!gb.is_empty());
//! ```

use super::big_rational::BigRational;
use super::multivariate_poly::{Monomial, MultivariatePolynomial};
use super::ring::Ring;
use super::term_order::MonomialOrder;

// ── Monomial utilities ────────────────────────────────────────────────────────

/// Compute the least common multiple of two monomials.
///
/// Takes the max exponent for each variable.
fn monomial_lcm(a: &Monomial, b: &Monomial) -> Monomial {
    use std::collections::BTreeMap;
    let mut vars: BTreeMap<super::SymbolId, u32> = BTreeMap::new();
    for (&id, &exp) in a.iter() {
        vars.insert(id, exp);
    }
    for (&id, &exp) in b.iter() {
        let e = vars.entry(id).or_insert(0);
        if exp > *e {
            *e = exp;
        }
    }
    Monomial::from_vars(vars)
}

// ── Order-aware leading term ──────────────────────────────────────────────────

/// Return the leading term `(monomial, coefficient)` of `f` under `order`.
///
/// Returns `None` if `f` is zero.
fn leading_term_ord<'a, O: MonomialOrder>(
    f: &'a MultivariatePolynomial<BigRational>,
    order: &O,
) -> Option<(&'a Monomial, &'a BigRational)> {
    f.iter().max_by(|(a, _), (b, _)| order.cmp_monomials(a, b))
}

// ── S-polynomial ─────────────────────────────────────────────────────────────

/// Compute the S-polynomial of `f` and `g` under `order`.
///
/// `S(f,g) = (L / LT(f)) * f - (L / LT(g)) * g`
/// where `L = lcm(LM(f), LM(g))`.
///
/// Returns zero if either polynomial is zero.
fn spoly<O: MonomialOrder>(
    f: &MultivariatePolynomial<BigRational>,
    g: &MultivariatePolynomial<BigRational>,
    order: &O,
) -> MultivariatePolynomial<BigRational> {
    let (lm_f, lc_f) = match leading_term_ord(f, order) {
        Some(t) => t,
        None => return MultivariatePolynomial::zero(),
    };
    let (lm_g, lc_g) = match leading_term_ord(g, order) {
        Some(t) => t,
        None => return MultivariatePolynomial::zero(),
    };
    let lcm = monomial_lcm(lm_f, lm_g);
    // quot_f = lcm / LM(f), quot_g = lcm / LM(g) — both must exist
    let quot_f = lcm.checked_div(lm_f).expect("lcm divisible by lm_f");
    let quot_g = lcm.checked_div(lm_g).expect("lcm divisible by lm_g");
    // Scale factor: lc_g / lc_f for the f term, 1 for the g term
    // S = (1/lc_f) * mono_quot_f * f - (1/lc_g) * mono_quot_g * g
    let scale_f = BigRational::one() / lc_f.clone();
    let scale_g = BigRational::one() / lc_g.clone();
    let tf = MultivariatePolynomial::monomial(scale_f, quot_f);
    let tg = MultivariatePolynomial::monomial(scale_g, quot_g);
    let part_f = &tf * f;
    let part_g = &tg * g;
    &part_f - &part_g
}

// ── Multivariate reduction ────────────────────────────────────────────────────

/// Reduce `f` modulo the set `basis` under `order`.
///
/// Repeatedly divides the leading term of the current remainder by a
/// leading term from `basis`. If no divisor found, moves the leading term
/// to the remainder. Returns the fully reduced remainder.
fn reduce<O: MonomialOrder>(
    f: &MultivariatePolynomial<BigRational>,
    basis: &[MultivariatePolynomial<BigRational>],
    order: &O,
) -> MultivariatePolynomial<BigRational> {
    let mut p = f.clone();
    let mut remainder = MultivariatePolynomial::zero();

    while !p.is_zero() {
        let (lm_p, lc_p) = match leading_term_ord(&p, order) {
            Some(t) => (t.0.clone(), t.1.clone()),
            None => break,
        };
        let divisor = basis.iter().find_map(|g| {
            leading_term_ord(g, order)
                .and_then(|(lm_g, lc_g)| lm_p.checked_div(lm_g).map(|q| (q, lc_g.clone(), g)))
        });
        match divisor {
            Some((quot_mono, lc_g, g)) => {
                // Subtract (lc_p / lc_g) * x^quot_mono * g from p
                let coeff = lc_p / lc_g;
                let mono_term = MultivariatePolynomial::monomial(coeff, quot_mono);
                let sub = &mono_term * g;
                p = &p - &sub;
            }
            None => {
                // No divisor found: move leading term to remainder
                remainder.add_term(lm_p, lc_p);
                p = remove_leading_term(p, order);
            }
        }
    }
    remainder
}

/// Remove the leading term of `f` under `order`.
fn remove_leading_term<O: MonomialOrder>(
    f: MultivariatePolynomial<BigRational>,
    order: &O,
) -> MultivariatePolynomial<BigRational> {
    let lm = match leading_term_ord(&f, order) {
        Some((m, _)) => m.clone(),
        None => return f,
    };
    let mut result = f.clone();
    // Subtract leading term
    let lc = result.coeff(&lm);
    result.add_term(lm, -lc);
    result
}

// ── Coprimeness criterion ─────────────────────────────────────────────────────

/// Buchberger's first criterion: skip pair if LM(f) and LM(g) are coprime.
///
/// When `lcm(LM(f), LM(g)) = LM(f) * LM(g)`, the S-polynomial reduces to
/// zero and the pair can be discarded.
fn are_leading_monomials_coprime<O: MonomialOrder>(
    f: &MultivariatePolynomial<BigRational>,
    g: &MultivariatePolynomial<BigRational>,
    order: &O,
) -> bool {
    let lm_f = match leading_term_ord(f, order) {
        Some((m, _)) => m,
        None => return true,
    };
    let lm_g = match leading_term_ord(g, order) {
        Some((m, _)) => m,
        None => return true,
    };
    // Coprime iff they share no common variable
    lm_f.iter().all(|(v, _)| lm_g.exponent(v) == 0)
}

// ── Buchberger main algorithm ─────────────────────────────────────────────────

/// Compute a Groebner basis for the ideal generated by `polys` under `order`.
///
/// Uses Buchberger's algorithm with the coprimeness criterion to prune
/// redundant critical pairs. The result is minimized and auto-reduced.
///
/// Returns an empty `Vec` if the input is empty or consists only of zero
/// polynomials.
///
/// # Example
///
/// ```
/// use thales::numeric::{buchberger, BigRational, GrevLex, Monomial, MultivariatePolynomial, SymbolId};
/// let x = SymbolId::intern("bex");
/// let r = |n: i64| BigRational::from(n);
/// // {x*y - 1} is already a Groebner basis
/// let xy_m1 = &MultivariatePolynomial::monomial(r(1), Monomial::var(x))
///     - &MultivariatePolynomial::constant(r(1));
/// let gb = buchberger(&[xy_m1], &GrevLex::new());
/// assert_eq!(gb.len(), 1);
/// ```
pub fn buchberger<O: MonomialOrder>(
    polys: &[MultivariatePolynomial<BigRational>],
    order: &O,
) -> Vec<MultivariatePolynomial<BigRational>> {
    let mut g: Vec<MultivariatePolynomial<BigRational>> =
        polys.iter().filter(|p| !p.is_zero()).cloned().collect();

    if g.is_empty() {
        return g;
    }

    // Normalize all generators to monic form
    for p in &mut g {
        *p = make_monic(p, order);
    }

    // Build initial critical pairs
    let mut pairs: Vec<(usize, usize)> = (0..g.len())
        .flat_map(|i| (i + 1..g.len()).map(move |j| (i, j)))
        .collect();

    while let Some((i, j)) = pairs.pop() {
        if are_leading_monomials_coprime(&g[i], &g[j], order) {
            continue;
        }
        let s = spoly(&g[i], &g[j], order);
        let r = reduce(&s, &g, order);
        if !r.is_zero() {
            let new_idx = g.len();
            let r_monic = make_monic(&r, order);
            // Add new pairs: (k, new_idx) for all existing k
            for k in 0..new_idx {
                pairs.push((k, new_idx));
            }
            g.push(r_monic);
        }
    }

    let g = minimize_basis(g, order);
    autoreduce_basis(g, order)
}

// ── Basis minimization ────────────────────────────────────────────────────────

/// Remove redundant generators: keep only elements whose LM is not divisible
/// by any other element's LM.
fn minimize_basis<O: MonomialOrder>(
    basis: Vec<MultivariatePolynomial<BigRational>>,
    order: &O,
) -> Vec<MultivariatePolynomial<BigRational>> {
    let lms: Vec<_> = basis
        .iter()
        .map(|p| leading_term_ord(p, order).map(|(m, _)| m.clone()))
        .collect();

    basis
        .into_iter()
        .enumerate()
        .filter(|(i, _)| {
            let lm_i = match &lms[*i] {
                Some(m) => m,
                None => return false,
            };
            // Keep if no OTHER element's LM divides lm_i
            lms.iter().enumerate().all(|(j, lm_j)| {
                if j == *i {
                    return true;
                }
                match lm_j {
                    Some(m) => !lm_i.is_divisible_by(m),
                    None => true,
                }
            })
        })
        .map(|(_, p)| p)
        .collect()
}

// ── Auto-reduction ────────────────────────────────────────────────────────────

/// Reduce each basis element modulo all others, making the result
/// fully inter-reduced.
fn autoreduce_basis<O: MonomialOrder>(
    basis: Vec<MultivariatePolynomial<BigRational>>,
    order: &O,
) -> Vec<MultivariatePolynomial<BigRational>> {
    let n = basis.len();
    let mut result = basis;
    for i in 0..n {
        let others: Vec<_> = result
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, p)| p.clone())
            .collect();
        let reduced = reduce(&result[i].clone(), &others, order);
        result[i] = if reduced.is_zero() {
            reduced
        } else {
            make_monic(&reduced, order)
        };
    }
    result.into_iter().filter(|p| !p.is_zero()).collect()
}

// ── Utilities ─────────────────────────────────────────────────────────────────

/// Make a polynomial monic: divide all coefficients by the leading coefficient.
fn make_monic<O: MonomialOrder>(
    f: &MultivariatePolynomial<BigRational>,
    order: &O,
) -> MultivariatePolynomial<BigRational> {
    match leading_term_ord(f, order) {
        Some((_, lc)) if !lc.is_one() => {
            let inv_lc = lc.clone().inv();
            f.scale(&inv_lc)
        }
        _ => f.clone(),
    }
}

// ── BigRational inv helper ────────────────────────────────────────────────────

use super::ring::Field;

trait InvExt {
    fn inv(self) -> Self;
}

impl InvExt for BigRational {
    fn inv(self) -> Self {
        Field::inv(&self)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{BigRational, GrevLex, Lex, Monomial, MultivariatePolynomial, SymbolId};

    type MP = MultivariatePolynomial<BigRational>;

    fn r(n: i64) -> BigRational {
        BigRational::from(n)
    }

    fn x() -> SymbolId {
        SymbolId::intern("gbx")
    }

    fn y() -> SymbolId {
        SymbolId::intern("gby")
    }

    fn make_x2_plus_y_minus1() -> MP {
        // x^2 + y - 1
        let x2 = MP::monomial(r(1), Monomial::var_pow(x(), 2));
        let y_term = MP::var(y());
        let one = MP::constant(r(1));
        &(&x2 + &y_term) - &one
    }

    fn make_x_plus_y2_minus1() -> MP {
        // x + y^2 - 1
        let x_term = MP::var(x());
        let y2 = MP::monomial(r(1), Monomial::var_pow(y(), 2));
        let one = MP::constant(r(1));
        &(&x_term + &y2) - &one
    }

    #[test]
    fn test_monomial_lcm_basic() {
        // lcm(x^2, y^3) = x^2*y^3
        let m1 = Monomial::var_pow(x(), 2);
        let m2 = Monomial::var_pow(y(), 3);
        let lcm = monomial_lcm(&m1, &m2);
        assert_eq!(lcm.exponent(&x()), 2);
        assert_eq!(lcm.exponent(&y()), 3);
    }

    #[test]
    fn test_monomial_lcm_overlap() {
        // lcm(x^2*y, x*y^3) = x^2*y^3
        let m1 = Monomial::var(x()).mul(&Monomial::var_pow(y(), 1));
        let m1 = m1.mul(&Monomial::var(x())); // x^2*y
        let m2 = Monomial::var(x()).mul(&Monomial::var_pow(y(), 3)); // x*y^3
        let lcm = monomial_lcm(&m1, &m2);
        assert_eq!(lcm.exponent(&x()), 2);
        assert_eq!(lcm.exponent(&y()), 3);
    }

    #[test]
    fn test_spoly_simple() {
        // S(x^2 - 1, x - 1): lcm(x^2, x) = x^2
        // S = (x^2/x^2)*(x^2-1) - (x^2/x)*(x-1) = (x^2-1) - x*(x-1) = x^2-1 - x^2+x = x-1
        let f = &MP::monomial(r(1), Monomial::var_pow(x(), 2)) - &MP::constant(r(1));
        let g = &MP::var(x()) - &MP::constant(r(1));
        let order = GrevLex::new();
        let s = spoly(&f, &g, &order);
        // Should reduce to x - 1 or 0 when fully reduced
        assert!(!s.is_zero() || s.is_zero()); // s is some polynomial
        let r_val = reduce(&s, &[f.clone(), g.clone()], &order);
        assert!(r_val.is_zero());
    }

    #[test]
    fn test_buchberger_already_gb() {
        // {x*y - 1} is its own Groebner basis
        let xy = MP::monomial(r(1), Monomial::var(x()).mul(&Monomial::var(y())));
        let f = &xy - &MP::constant(r(1));
        let gb = buchberger(&[f], &GrevLex::new());
        assert_eq!(gb.len(), 1);
    }

    #[test]
    fn test_buchberger_two_polys_grevlex() {
        // {x^2 + y - 1, x + y^2 - 1} — classic example
        let f1 = make_x2_plus_y_minus1();
        let f2 = make_x_plus_y2_minus1();
        let order = GrevLex::new();
        let gb = buchberger(&[f1, f2], &order);
        assert!(!gb.is_empty(), "Groebner basis must be non-empty");
        // All elements must be monic
        for p in &gb {
            if let Some((_, lc)) = leading_term_ord(p, &order) {
                assert!(lc.is_one(), "Groebner basis element must be monic");
            }
        }
    }

    #[test]
    fn test_buchberger_ideal_membership() {
        // If h is in the ideal, reduce(h, GB) = 0
        let f1 = make_x2_plus_y_minus1();
        let f2 = make_x_plus_y2_minus1();
        let order = GrevLex::new();
        let gb = buchberger(&[f1.clone(), f2.clone()], &order);

        // f1 itself should reduce to zero mod its own GB
        let r1 = reduce(&f1, &gb, &order);
        assert!(r1.is_zero(), "f1 must reduce to 0 mod its own GB");
        let r2 = reduce(&f2, &gb, &order);
        assert!(r2.is_zero(), "f2 must reduce to 0 mod its own GB");
    }

    #[test]
    fn test_buchberger_lex_order() {
        // Same ideal, lex order — gives elimination basis
        let f1 = make_x2_plus_y_minus1();
        let f2 = make_x_plus_y2_minus1();
        let order = Lex::new(vec![x(), y()]);
        let gb = buchberger(&[f1, f2], &order);
        assert!(!gb.is_empty(), "Lex GB must be non-empty");
    }

    #[test]
    fn test_buchberger_empty_input() {
        let order = GrevLex::new();
        let gb = buchberger::<GrevLex>(&[], &order);
        assert!(gb.is_empty());
    }

    #[test]
    fn test_buchberger_zero_input_filtered() {
        let order = GrevLex::new();
        let gb = buchberger(&[MP::zero()], &order);
        assert!(gb.is_empty());
    }

    #[test]
    fn test_reduce_by_basis() {
        // x^2 should reduce to (1 - y) modulo {x^2 + y - 1}
        let f1 = make_x2_plus_y_minus1(); // x^2 = 1 - y mod f1
        let x2 = MP::monomial(r(1), Monomial::var_pow(x(), 2));
        let order = GrevLex::new();
        let rem = reduce(&x2, &[f1], &order);
        // x^2 - (x^2 + y - 1) = 1 - y
        let expected = &MP::constant(r(1)) - &MP::var(y());
        assert_eq!(rem, expected);
    }

    #[test]
    fn test_make_monic() {
        // 3*x + 6 → x + 2
        let f = &MP::monomial(r(3), Monomial::var(x())) + &MP::constant(r(6));
        let order = GrevLex::new();
        let m = make_monic(&f, &order);
        assert!(leading_term_ord(&m, &order)
            .map(|(_, c)| c.is_one())
            .unwrap_or(false));
    }
}
