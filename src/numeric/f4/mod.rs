//! Faugère's F4 algorithm for computing Groebner bases.
//!
//! F4 improves on Buchberger's algorithm by processing multiple critical pairs
//! simultaneously using linear algebra (sparse matrix row reduction) rather than
//! reducing one S-polynomial at a time.
//!
//! # Algorithm overview
//!
//! 1. Maintain a set of critical pairs `(i, j)`.
//! 2. Each iteration, select all pending pairs and build a set of "symbolic
//!    preprocessed" polynomials — the S-polynomial numerators and all multiples
//!    of basis elements needed to reduce them.
//! 3. Assemble these polynomials into a matrix (rows = polynomials,
//!    columns = monomials in order).
//! 4. Row-reduce the matrix (RREF over ℚ).
//! 5. Rows whose pivot monomial was **not** already a leading monomial of the
//!    current basis become new basis elements.
//! 6. Add new critical pairs and repeat until no new elements arise.
//!
//! # References
//!
//! - Faugère, J.-C.: "A new efficient algorithm for computing Groebner bases
//!   (F4)", Journal of Pure and Applied Algebra 139 (1999), pp. 61-88.
//!
//! # Example
//!
//! ```
//! use thales::numeric::{f4, BigRational, GrevLex, Monomial, MultivariatePolynomial, SymbolId};
//!
//! let x = SymbolId::intern("f4_x");
//! let y = SymbolId::intern("f4_y");
//! let r = |n: i64| BigRational::from(n);
//!
//! // f1 = x^2 + y - 1,  f2 = x + y^2 - 1
//! let f1 = &MultivariatePolynomial::monomial(r(1), Monomial::var_pow(x, 2))
//!     + &(&MultivariatePolynomial::var(y) - &MultivariatePolynomial::constant(r(1)));
//! let f2 = &MultivariatePolynomial::var(x)
//!     + &(&MultivariatePolynomial::monomial(r(1), Monomial::var_pow(y, 2))
//!         - &MultivariatePolynomial::constant(r(1)));
//!
//! let gb = f4(&[f1, f2], &GrevLex::new());
//! assert!(!gb.is_empty());
//! ```

mod matrix_ops;

use super::big_rational::BigRational;
use super::multivariate_poly::{Monomial, MultivariatePolynomial};
use super::ring::{Field, Ring};
use super::term_order::MonomialOrder;
use matrix_ops::{build_columns, pivot_col, poly_to_row, row_reduce, row_to_poly};
use std::collections::{BTreeSet, HashMap};

// ── Monomial utilities ────────────────────────────────────────────────────────

/// Compute the LCM of two monomials (max exponent per variable).
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

/// Return the leading `(monomial, coefficient)` of `f` under `order`, or `None`.
fn leading_term<'a, O: MonomialOrder>(
    f: &'a MultivariatePolynomial<BigRational>,
    order: &O,
) -> Option<(&'a Monomial, &'a BigRational)> {
    f.iter().max_by(|(a, _), (b, _)| order.cmp_monomials(a, b))
}

/// Make a polynomial monic under `order`.
fn make_monic<O: MonomialOrder>(
    f: &MultivariatePolynomial<BigRational>,
    order: &O,
) -> MultivariatePolynomial<BigRational> {
    match leading_term(f, order) {
        Some((_, lc)) if !lc.is_one() => {
            let inv = Field::inv(lc);
            f.scale(&inv)
        }
        _ => f.clone(),
    }
}

/// Returns `true` if the leading monomials of `f` and `g` are coprime.
fn leading_coprime<O: MonomialOrder>(
    f: &MultivariatePolynomial<BigRational>,
    g: &MultivariatePolynomial<BigRational>,
    order: &O,
) -> bool {
    let lm_f = match leading_term(f, order) {
        Some((m, _)) => m,
        None => return true,
    };
    let lm_g = match leading_term(g, order) {
        Some((m, _)) => m,
        None => return true,
    };
    lm_f.iter().all(|(v, _)| lm_g.exponent(v) == 0)
}

// ── Symbolic preprocessing ────────────────────────────────────────────────────

/// Collect all multiples of basis elements needed to reduce the S-polynomial
/// numerators.  Returns the set of polynomials to include in the matrix.
///
/// For each critical pair `(i, j)`:
///   - compute `L = lcm(lm(g_i), lm(g_j))`
///   - add `(L / lm(g_i)) * g_i`  and  `(L / lm(g_j)) * g_j`
///
/// Then repeatedly: for every monomial `m` in the collected set, if `m` is a
/// leading monomial of some `g_k`, add the appropriate multiple of `g_k` to
/// reduce `m` (symbolic preprocessing step).
fn symbolic_preprocess<O: MonomialOrder>(
    pairs: &[(usize, usize)],
    basis: &[MultivariatePolynomial<BigRational>],
    order: &O,
) -> Vec<MultivariatePolynomial<BigRational>> {
    let mut polys: Vec<MultivariatePolynomial<BigRational>> = Vec::new();
    // Monomials already "covered" by a polynomial in `polys`
    let mut covered: BTreeSet<Monomial> = BTreeSet::new();

    // Seed from critical pairs
    for &(i, j) in pairs {
        let lm_i = match leading_term(&basis[i], order) {
            Some((m, _)) => m.clone(),
            None => continue,
        };
        let lm_j = match leading_term(&basis[j], order) {
            Some((m, _)) => m.clone(),
            None => continue,
        };
        let lcm = monomial_lcm(&lm_i, &lm_j);
        // multiplier for basis[i]: lcm / lm_i
        if let Some(q) = lcm.checked_div(&lm_i) {
            let mult = MultivariatePolynomial::monomial(BigRational::one(), q);
            polys.push(&mult * &basis[i]);
        }
        // multiplier for basis[j]: lcm / lm_j
        if let Some(q) = lcm.checked_div(&lm_j) {
            let mult = MultivariatePolynomial::monomial(BigRational::one(), q);
            polys.push(&mult * &basis[j]);
        }
    }

    // Collect all monomials that appear as non-leading terms
    let mut todo: Vec<Monomial> = polys
        .iter()
        .flat_map(|p| {
            let lm = leading_term(p, order).map(|(m, _)| m.clone());
            p.iter()
                .map(|(m, _)| m.clone())
                .filter(move |m| Some(m) != lm.as_ref())
        })
        .collect();

    while let Some(mono) = todo.pop() {
        if covered.contains(&mono) {
            continue;
        }
        // Find a basis element whose LM divides `mono`
        let reducer = basis.iter().find_map(|g| {
            leading_term(g, order).and_then(|(lm, _)| {
                mono.checked_div(lm).map(|q| {
                    let mult = MultivariatePolynomial::monomial(BigRational::one(), q);
                    &mult * g
                })
            })
        });
        if let Some(red) = reducer {
            // Add its non-leading monomials to the todo list
            if let Some((lm, _)) = leading_term(&red, order) {
                covered.insert(lm.clone());
                for (m, _) in red.iter() {
                    if m != lm && !covered.contains(m) {
                        todo.push(m.clone());
                    }
                }
            }
            polys.push(red);
        } else {
            covered.insert(mono);
        }
    }

    polys
}

// ── Basis minimization / auto-reduction (shared with Buchberger) ──────────────

fn minimize_basis<O: MonomialOrder>(
    basis: Vec<MultivariatePolynomial<BigRational>>,
    order: &O,
) -> Vec<MultivariatePolynomial<BigRational>> {
    let lms: Vec<_> = basis
        .iter()
        .map(|p| leading_term(p, order).map(|(m, _)| m.clone()))
        .collect();
    basis
        .into_iter()
        .enumerate()
        .filter(|(i, _)| {
            let lm_i = match &lms[*i] {
                Some(m) => m,
                None => return false,
            };
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

fn reduce_poly<O: MonomialOrder>(
    f: &MultivariatePolynomial<BigRational>,
    basis: &[MultivariatePolynomial<BigRational>],
    order: &O,
) -> MultivariatePolynomial<BigRational> {
    let mut p = f.clone();
    let mut remainder = MultivariatePolynomial::zero();
    while !p.is_zero() {
        let (lm_p, lc_p) = match leading_term(&p, order) {
            Some((m, c)) => (m.clone(), c.clone()),
            None => break,
        };
        let divisor = basis.iter().find_map(|g| {
            leading_term(g, order)
                .and_then(|(lm_g, lc_g)| lm_p.checked_div(lm_g).map(|q| (q, lc_g.clone(), g)))
        });
        match divisor {
            Some((q, lc_g, g)) => {
                let coeff = lc_p / lc_g;
                let sub = &MultivariatePolynomial::monomial(coeff, q) * g;
                p = &p - &sub;
            }
            None => {
                let lm_p2 = lm_p.clone();
                remainder.add_term(lm_p, lc_p);
                // remove leading term
                let lc2 = p.coeff(&lm_p2);
                p.add_term(lm_p2, -lc2);
            }
        }
    }
    remainder
}

fn autoreduce<O: MonomialOrder>(
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
        let red = reduce_poly(&result[i].clone(), &others, order);
        result[i] = if red.is_zero() {
            red
        } else {
            make_monic(&red, order)
        };
    }
    result.into_iter().filter(|p| !p.is_zero()).collect()
}

// ── F4 main algorithm ─────────────────────────────────────────────────────────

/// Compute a Groebner basis using Faugère's F4 algorithm.
///
/// F4 processes batches of critical pairs simultaneously using matrix
/// row reduction, which is more efficient than Buchberger on dense systems.
/// The result is a minimal, auto-reduced Groebner basis.
///
/// Returns an empty `Vec` if the input is empty or all-zero.
///
/// # Example
///
/// ```
/// use thales::numeric::{f4, BigRational, GrevLex, Monomial, MultivariatePolynomial, SymbolId};
///
/// let x = SymbolId::intern("f4ex");
/// let r = |n: i64| BigRational::from(n);
/// // {x - 1} is already a Groebner basis
/// let p = &MultivariatePolynomial::var(x) - &MultivariatePolynomial::constant(r(1));
/// let gb = f4(&[p], &GrevLex::new());
/// assert_eq!(gb.len(), 1);
/// ```
pub fn f4<O: MonomialOrder>(
    polys: &[MultivariatePolynomial<BigRational>],
    order: &O,
) -> Vec<MultivariatePolynomial<BigRational>> {
    let mut g: Vec<MultivariatePolynomial<BigRational>> =
        polys.iter().filter(|p| !p.is_zero()).cloned().collect();

    if g.is_empty() {
        return g;
    }

    for p in &mut g {
        *p = make_monic(p, order);
    }

    // Initial critical pairs
    let mut pairs: Vec<(usize, usize)> = (0..g.len())
        .flat_map(|i| (i + 1..g.len()).map(move |j| (i, j)))
        .collect();

    while !pairs.is_empty() {
        // Filter coprime pairs (Buchberger criterion 1)
        let batch: Vec<(usize, usize)> = pairs
            .drain(..)
            .filter(|&(i, j)| !leading_coprime(&g[i], &g[j], order))
            .collect();

        if batch.is_empty() {
            break;
        }

        // Symbolic preprocessing
        let matrix_polys = symbolic_preprocess(&batch, &g, order);

        if matrix_polys.is_empty() {
            continue;
        }

        // Record existing leading monomials before reduction
        let existing_lms: BTreeSet<Monomial> = g
            .iter()
            .filter_map(|p| leading_term(p, order).map(|(m, _)| m.clone()))
            .collect();

        // Build and row-reduce the matrix
        let cols = build_columns(&matrix_polys, order);
        let rows: Vec<_> = matrix_polys.iter().map(|p| poly_to_row(p, &cols)).collect();
        let reduced = row_reduce(rows);

        // Extract new basis elements: rows whose pivot is a *new* leading monomial
        let mut new_elements = extract_new_elements(reduced, &cols, &existing_lms, order);

        if new_elements.is_empty() {
            break;
        }

        // Add new elements and record the index range they occupy
        let added_start = g.len();
        for new_p in new_elements {
            g.push(new_p);
        }
        let added_end = g.len();

        // New pairs: (k, new_idx) for all existing k < new_idx
        for new_idx in added_start..added_end {
            for k in 0..new_idx {
                pairs.push((k, new_idx));
            }
        }
    }

    let g = minimize_basis(g, order);
    autoreduce(g, order)
}

/// Extract polynomials from reduced rows that introduce new leading monomials.
fn extract_new_elements<O: MonomialOrder>(
    reduced_rows: Vec<matrix_ops::Row>,
    cols: &[Monomial],
    existing_lms: &BTreeSet<Monomial>,
    order: &O,
) -> Vec<MultivariatePolynomial<BigRational>> {
    let mut result = Vec::new();
    // Build a map from pivot column index → leading monomial
    let pivot_lms: HashMap<usize, Monomial> = reduced_rows
        .iter()
        .filter_map(|row| pivot_col(row).map(|c| (c, cols[c].clone())))
        .collect();

    for row in &reduced_rows {
        if let Some(pc) = pivot_col(row) {
            let lm = &pivot_lms[&pc];
            if !existing_lms.contains(lm) {
                let p = row_to_poly(row, cols);
                if !p.is_zero() {
                    // Verify the leading monomial under `order` matches expectation
                    if let Some((actual_lm, _)) = leading_term(&p, order) {
                        if actual_lm == lm {
                            result.push(p);
                        }
                    }
                }
            }
        }
    }
    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{
        buchberger, BigRational, GrevLex, Lex, Monomial, MultivariatePolynomial, SymbolId,
    };

    type MP = MultivariatePolynomial<BigRational>;

    fn r(n: i64) -> BigRational {
        BigRational::from(n)
    }
    fn x() -> SymbolId {
        SymbolId::intern("f4tx")
    }
    fn y() -> SymbolId {
        SymbolId::intern("f4ty")
    }
    fn z() -> SymbolId {
        SymbolId::intern("f4tz")
    }
    fn w() -> SymbolId {
        SymbolId::intern("f4tw")
    }

    fn make_f1() -> MP {
        // x^2 + y - 1
        let x2 = MP::monomial(r(1), Monomial::var_pow(x(), 2));
        &(&x2 + &MP::var(y())) - &MP::constant(r(1))
    }
    fn make_f2() -> MP {
        // x + y^2 - 1
        let y2 = MP::monomial(r(1), Monomial::var_pow(y(), 2));
        &(&MP::var(x()) + &y2) - &MP::constant(r(1))
    }

    /// Compare two Groebner bases for the same ideal: both reduce each other's
    /// elements to zero.
    fn same_ideal<O: MonomialOrder>(gb1: &[MP], gb2: &[MP], order: &O) -> bool {
        gb1.iter().all(|p| reduce_poly(p, gb2, order).is_zero())
            && gb2.iter().all(|p| reduce_poly(p, gb1, order).is_zero())
    }

    #[test]
    fn test_f4_empty_input() {
        let order = GrevLex::new();
        let gb = f4::<GrevLex>(&[], &order);
        assert!(gb.is_empty());
    }

    #[test]
    fn test_f4_zero_input_filtered() {
        let order = GrevLex::new();
        let gb = f4(&[MP::zero()], &order);
        assert!(gb.is_empty());
    }

    #[test]
    fn test_f4_single_poly() {
        let order = GrevLex::new();
        let p = &MP::var(x()) - &MP::constant(r(1));
        let gb = f4(&[p], &order);
        assert_eq!(gb.len(), 1);
    }

    #[test]
    fn test_f4_two_polys_grevlex() {
        let order = GrevLex::new();
        let gb = f4(&[make_f1(), make_f2()], &order);
        assert!(!gb.is_empty(), "F4 GB must be non-empty");
        // All elements must be monic
        for p in &gb {
            if let Some((_, lc)) = leading_term(p, &order) {
                assert!(lc.is_one(), "GB element must be monic");
            }
        }
    }

    #[test]
    fn test_f4_same_ideal_as_buchberger_grevlex() {
        let order = GrevLex::new();
        let f4_gb = f4(&[make_f1(), make_f2()], &order);
        let buch_gb = buchberger(&[make_f1(), make_f2()], &order);
        assert!(
            same_ideal(&f4_gb, &buch_gb, &order),
            "F4 and Buchberger must produce bases for the same ideal"
        );
    }

    #[test]
    fn test_f4_ideal_membership() {
        let order = GrevLex::new();
        let gb = f4(&[make_f1(), make_f2()], &order);
        // Original generators must reduce to zero mod GB
        assert!(reduce_poly(&make_f1(), &gb, &order).is_zero());
        assert!(reduce_poly(&make_f2(), &gb, &order).is_zero());
    }

    #[test]
    fn test_f4_lex_order() {
        let order = Lex::new(vec![x(), y()]);
        let gb = f4(&[make_f1(), make_f2()], &order);
        assert!(!gb.is_empty(), "Lex F4 GB must be non-empty");
        let buch_gb = buchberger(&[make_f1(), make_f2()], &order);
        assert!(same_ideal(&gb, &buch_gb, &order));
    }

    #[test]
    fn test_f4_cyclic4_same_as_buchberger() {
        // Cyclic-4 system: a classic benchmark for GB algorithms
        // e1 = x+y+z+w, e2 = xy+yz+zw+wx, e3 = xyz+yzw+zwx+wxy, e4 = xyzw-1
        let order = GrevLex::new();

        let e1 = &(&(&MP::var(x()) + &MP::var(y())) + &MP::var(z())) + &MP::var(w());
        let xy = MP::monomial(r(1), Monomial::var(x()).mul(&Monomial::var(y())));
        let yz = MP::monomial(r(1), Monomial::var(y()).mul(&Monomial::var(z())));
        let zw = MP::monomial(r(1), Monomial::var(z()).mul(&Monomial::var(w())));
        let wx = MP::monomial(r(1), Monomial::var(w()).mul(&Monomial::var(x())));
        let e2 = &(&(&xy + &yz) + &zw) + &wx;

        let xyz = MP::monomial(
            r(1),
            Monomial::var(x())
                .mul(&Monomial::var(y()))
                .mul(&Monomial::var(z())),
        );
        let yzw = MP::monomial(
            r(1),
            Monomial::var(y())
                .mul(&Monomial::var(z()))
                .mul(&Monomial::var(w())),
        );
        let zwx = MP::monomial(
            r(1),
            Monomial::var(z())
                .mul(&Monomial::var(w()))
                .mul(&Monomial::var(x())),
        );
        let wxy = MP::monomial(
            r(1),
            Monomial::var(w())
                .mul(&Monomial::var(x()))
                .mul(&Monomial::var(y())),
        );
        let e3 = &(&(&xyz + &yzw) + &zwx) + &wxy;

        let xyzw = MP::monomial(
            r(1),
            Monomial::var(x())
                .mul(&Monomial::var(y()))
                .mul(&Monomial::var(z()))
                .mul(&Monomial::var(w())),
        );
        let e4 = &xyzw - &MP::constant(r(1));

        let system = [e1.clone(), e2.clone(), e3.clone(), e4.clone()];
        let f4_gb = f4(&system, &order);
        let buch_gb = buchberger(&system, &order);

        assert!(!f4_gb.is_empty(), "cyclic-4 F4 GB must be non-empty");
        assert!(
            same_ideal(&f4_gb, &buch_gb, &order),
            "F4 and Buchberger must produce bases for the same ideal (cyclic-4)"
        );
    }
}
