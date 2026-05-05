//! Differential forms: types, wedge product, and exterior derivative.
//!
//! # Representation
//!
//! A degree-k [`DifferentialForm`] is a formal sum of [`FormTerm`]s. Each
//! term carries:
//! - a symbolic `coefficient: Arc<Expr>`, and
//! - a `basis: Vec<SymbolId>` of length `k` whose entries identify the
//!   coordinate differentials appearing in the basis element
//!   (e.g. `[x, y]` represents `dx ∧ dy`).
//!
//! # Basis canonicalisation
//!
//! Every `FormTerm` is maintained in *canonical form*: the `basis` vector is
//! sorted in ascending [`SymbolId`] order. When a sorting permutation is odd,
//! the coefficient is negated to reflect the antisymmetry of the wedge product
//! (`dx ∧ dy = -dy ∧ dx`). Terms whose basis contains a repeated element
//! vanish (they are dropped entirely), which encodes `dx ∧ dx = 0`.
//!
//! Callers must not assume anything about the sign embedded in `coefficient`
//! beyond what follows from the canonical basis order.

use crate::numeric::{differentiation::diff_arc, normalize, Expr, SymbolId};
use std::sync::Arc;

// ── Public types ──────────────────────────────────────────────────────────────

/// A single term `coefficient · dx_i1 ∧ dx_i2 ∧ … ∧ dx_ik`.
///
/// Invariant: `basis.len() == form.degree` and `basis` is sorted in ascending
/// [`SymbolId`] order. The coefficient incorporates any sign change required to
/// achieve that order (odd permutation ⇒ coefficient is negated).
#[derive(Debug, Clone)]
pub struct FormTerm {
    /// Symbolic coefficient, e.g. `3*x` in `3x dx ∧ dy`.
    pub coefficient: Arc<Expr>,
    /// Ordered basis indices.  `[sym_x, sym_y]` represents `dx ∧ dy`.
    pub basis: Vec<SymbolId>,
}

/// A degree-k differential form: a formal sum of [`FormTerm`]s.
///
/// The degree is stored separately so that a zero form still carries
/// degree information.
#[derive(Debug, Clone)]
pub struct DifferentialForm {
    /// The degree k: number of basis differentials in each term.
    pub degree: usize,
    /// The terms of the form.  May be empty (zero form).
    pub terms: Vec<FormTerm>,
}

// ── Public constructors ───────────────────────────────────────────────────────

/// Construct the zero k-form.
pub fn zero(degree: usize) -> DifferentialForm {
    DifferentialForm {
        degree,
        terms: Vec::new(),
    }
}

/// Construct a 1-form `Σ aᵢ dxᵢ` from coefficient–variable pairs.
///
/// Each pair `(aᵢ, xᵢ)` contributes the term `aᵢ · dx_xᵢ`.
pub fn one_form(var_coefs: &[(Arc<Expr>, SymbolId)]) -> DifferentialForm {
    let terms = var_coefs
        .iter()
        .filter_map(|(coef, var)| {
            if coef.is_zero() {
                None
            } else {
                Some(FormTerm {
                    coefficient: coef.clone(),
                    basis: vec![*var],
                })
            }
        })
        .collect();
    DifferentialForm { degree: 1, terms }
}

// ── Wedge product ─────────────────────────────────────────────────────────────

/// Compute `a ∧ b`.
///
/// For each pair of terms `(tₐ, t_b)`, the result term has:
/// - basis = `tₐ.basis ++ t_b.basis`, then canonicalised (sorted with sign
///   tracking),
/// - coefficient = `tₐ.coefficient * t_b.coefficient * sign`.
///
/// Terms with a repeated basis element vanish and are dropped.
pub fn wedge(a: &DifferentialForm, b: &DifferentialForm) -> DifferentialForm {
    let degree = a.degree + b.degree;
    let mut terms: Vec<FormTerm> = Vec::new();

    for ta in &a.terms {
        for tb in &b.terms {
            let mut basis: Vec<SymbolId> =
                ta.basis.iter().chain(tb.basis.iter()).copied().collect();

            // Canonicalise: sort basis, track sign parity.
            let sign = canonicalise_basis(&mut basis);

            // Vanishing term: repeated basis element.
            if sign == 0 {
                continue;
            }

            // Multiply coefficients and apply sign.
            let mut coef = normalize::mul(ta.coefficient.clone(), tb.coefficient.clone());
            if sign < 0 {
                coef = normalize::neg(coef);
            }

            if !coef.is_zero() {
                terms.push(FormTerm {
                    coefficient: coef,
                    basis,
                });
            }
        }
    }

    DifferentialForm {
        degree,
        terms: collect_terms(terms),
    }
}

// ── Exterior derivative ───────────────────────────────────────────────────────

/// Compute `dω` for a k-form `ω`, yielding a (k+1)-form.
///
/// For each term `f · dx_i1 ∧ … ∧ dx_ik` and each variable `x_j` in
/// `all_vars`, the contribution is:
///
/// ```text
/// (∂f/∂x_j) · dx_j ∧ dx_i1 ∧ … ∧ dx_ik
/// ```
///
/// After prepending `dx_j`, the basis is canonicalised (sorted with sign
/// adjustment).  Terms with a repeated element vanish.
///
/// `all_vars` should contain every coordinate variable; the result is
/// independent of the ordering of `all_vars`.
pub fn exterior_derivative(form: &DifferentialForm, all_vars: &[SymbolId]) -> DifferentialForm {
    let out_degree = form.degree + 1;
    let mut terms: Vec<FormTerm> = Vec::new();

    for term in &form.terms {
        for &var in all_vars {
            let partial = diff_arc(&term.coefficient, var);
            if partial.is_zero() {
                continue;
            }

            // Build basis: prepend dx_var before existing basis, then canonicalise.
            let mut basis: Vec<SymbolId> = std::iter::once(var)
                .chain(term.basis.iter().copied())
                .collect();

            let sign = canonicalise_basis(&mut basis);

            // Vanishing: repeated element (var already appears in term.basis).
            if sign == 0 {
                continue;
            }

            let coef = if sign < 0 {
                normalize::neg(partial)
            } else {
                partial
            };

            if !coef.is_zero() {
                terms.push(FormTerm {
                    coefficient: coef,
                    basis,
                });
            }
        }
    }

    DifferentialForm {
        degree: out_degree,
        terms: collect_terms(terms),
    }
}

// ── Like-term collection ──────────────────────────────────────────────────────

/// Collect terms sharing the same canonical basis by adding their coefficients.
///
/// Terms whose summed coefficient is zero are dropped, giving a fully reduced
/// form.  The relative order of surviving basis elements is preserved.
fn collect_terms(mut terms: Vec<FormTerm>) -> Vec<FormTerm> {
    // Group indices by basis.
    // We iterate in order, accumulating coefficients for each seen basis.
    let mut result: Vec<FormTerm> = Vec::with_capacity(terms.len());

    'outer: for term in terms.drain(..) {
        for existing in &mut result {
            if existing.basis == term.basis {
                existing.coefficient =
                    normalize::add(existing.coefficient.clone(), term.coefficient);
                continue 'outer;
            }
        }
        result.push(term);
    }

    // Drop zero-coefficient terms.
    result.retain(|t| !t.coefficient.is_zero());
    result
}

// ── Basis canonicalisation ────────────────────────────────────────────────────

/// Sort `basis` in-place using insertion sort, tracking the sign of the
/// permutation (±1) or returning 0 if a duplicate element is found.
///
/// Uses insertion sort so that the swap count equals the number of
/// inversions, giving the correct sign without computing a full permutation
/// parity separately.
fn canonicalise_basis(basis: &mut Vec<SymbolId>) -> i32 {
    let n = basis.len();
    let mut swaps: u32 = 0;

    for i in 1..n {
        let key = basis[i];
        let mut j = i;
        while j > 0 {
            if basis[j - 1] == key {
                // Repeated element → wedge product vanishes.
                return 0;
            }
            if basis[j - 1] > key {
                basis[j] = basis[j - 1];
                j -= 1;
                swaps += 1;
            } else {
                break;
            }
        }
        basis[j] = key;
    }

    if swaps % 2 == 0 {
        1
    } else {
        -1
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{normalize, Expr, SymbolId};

    // ── helpers ──────────────────────────────────────────────────────────────

    fn sid(name: &str) -> SymbolId {
        SymbolId::intern(name)
    }

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }

    fn int(n: i64) -> Arc<Expr> {
        Expr::int(n)
    }

    /// Build a pure basis 1-form: `dx` with coefficient 1.
    fn dx(var: &str) -> DifferentialForm {
        one_form(&[(int(1), sid(var))])
    }

    // ── canonicalise_basis ────────────────────────────────────────────────────

    #[test]
    fn test_canon_already_sorted_sign_plus() {
        let x = sid("cf_x");
        let y = sid("cf_y");
        // Ensure x < y by interning order. We rely on intern order being stable.
        // x was interned before y, so x.index() < y.index().
        let mut basis = vec![x, y];
        let sign = canonicalise_basis(&mut basis);
        // Already sorted: 0 swaps → sign +1.
        assert_eq!(sign, 1);
        assert_eq!(basis, vec![x, y]);
    }

    #[test]
    fn test_canon_reverse_sign_minus() {
        // Intern in order so a < b.
        let a = sid("cf_a");
        let b = sid("cf_b");
        // Reverse: [b, a] needs 1 swap → sign -1.
        let mut basis = vec![b, a];
        let sign = canonicalise_basis(&mut basis);
        assert_eq!(sign, -1);
        assert_eq!(basis, vec![a, b]);
    }

    #[test]
    fn test_canon_repeated_returns_zero() {
        let x = sid("cf_rep");
        let mut basis = vec![x, x];
        let sign = canonicalise_basis(&mut basis);
        assert_eq!(sign, 0);
    }

    // ── zero / one_form ───────────────────────────────────────────────────────

    #[test]
    fn test_zero_form_empty() {
        let f = zero(2);
        assert_eq!(f.degree, 2);
        assert!(f.terms.is_empty());
    }

    #[test]
    fn test_one_form_basic() {
        let xid = sid("of_x");
        let f = one_form(&[(int(3), xid)]);
        assert_eq!(f.degree, 1);
        assert_eq!(f.terms.len(), 1);
        assert_eq!(f.terms[0].basis, vec![xid]);
    }

    #[test]
    fn test_one_form_drops_zero_coef() {
        let xid = sid("of_zero_x");
        let f = one_form(&[(int(0), xid)]);
        assert!(f.terms.is_empty());
    }

    // ── wedge ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_wedge_dx_dy_nonzero() {
        // dx ∧ dy must be non-zero.
        let result = wedge(&dx("wdg_x"), &dx("wdg_y"));
        assert_eq!(result.degree, 2);
        assert_eq!(result.terms.len(), 1);
    }

    #[test]
    fn test_wedge_dx_dx_vanishes() {
        // dx ∧ dx = 0 (antisymmetry / nilpotency).
        let result = wedge(&dx("wdg_xx"), &dx("wdg_xx"));
        assert!(result.terms.is_empty(), "dx ∧ dx must be zero");
    }

    #[test]
    fn test_wedge_antisymmetry() {
        // dx ∧ dy = -(dy ∧ dx).
        // We verify by checking that the coefficient of (dx ∧ dy) + (dy ∧ dx) = 0.
        let x = "wdg_anti_x";
        let y = "wdg_anti_y";

        let dxdy = wedge(&dx(x), &dx(y));
        let dydx = wedge(&dx(y), &dx(x));

        // Both have the same canonical basis [sid(x), sid(y)] (x < y by intern
        // order since x was interned first in this test function).
        assert_eq!(dxdy.terms.len(), 1);
        assert_eq!(dydx.terms.len(), 1);

        let coef_dxdy = &dxdy.terms[0].coefficient;
        let coef_dydx = &dydx.terms[0].coefficient;

        // coef_dxdy + coef_dydx should be zero (they are +1 and -1).
        let sum = normalize::add(coef_dxdy.clone(), coef_dydx.clone());
        assert!(
            sum.is_zero(),
            "dx∧dy + dy∧dx coefficient must be 0, got {sum}"
        );
    }

    #[test]
    fn test_wedge_with_scalar_coefficient() {
        // (2 dx) ∧ (3 dy) = 6 dx∧dy
        let xid = sid("wdg_sc_x");
        let yid = sid("wdg_sc_y");
        let two_dx = one_form(&[(int(2), xid)]);
        let three_dy = one_form(&[(int(3), yid)]);
        let result = wedge(&two_dx, &three_dy);
        assert_eq!(result.terms.len(), 1);
        // Coefficient should be 6.
        let coef = &result.terms[0].coefficient;
        assert_eq!(*coef.as_ref(), *Expr::int(6).as_ref());
    }

    // ── exterior derivative ───────────────────────────────────────────────────

    #[test]
    fn test_exterior_derivative_of_constant_is_zero() {
        // d(5) = 0  (0-form with no dependence on any var)
        let xid = sid("ed_cx");
        let yid = sid("ed_cy");
        let coef = int(5);
        // A 0-form is just a function (coefficient with empty basis).
        let f = DifferentialForm {
            degree: 0,
            terms: vec![FormTerm {
                coefficient: coef,
                basis: vec![],
            }],
        };
        let df = exterior_derivative(&f, &[xid, yid]);
        assert!(df.terms.is_empty(), "d(5) must be zero");
    }

    #[test]
    fn test_exterior_derivative_of_x_dy() {
        // d(x · dy) = dx ∧ dy  in (x, y).
        // The 1-form ω = x dy has one term: coef=x, basis=[dy].
        let xid = sid("ed_xdy_x");
        let yid = sid("ed_xdy_y");

        let omega = DifferentialForm {
            degree: 1,
            terms: vec![FormTerm {
                coefficient: sym("ed_xdy_x"),
                basis: vec![yid],
            }],
        };

        let d_omega = exterior_derivative(&omega, &[xid, yid]);

        // Result must be a 2-form with exactly one term.
        assert_eq!(d_omega.degree, 2);
        assert_eq!(
            d_omega.terms.len(),
            1,
            "d(x dy) should have exactly one term, got {:?}",
            d_omega.terms
        );

        // The single term must have coefficient 1 and basis [xid, yid].
        let term = &d_omega.terms[0];
        assert!(
            term.coefficient.is_one(),
            "coefficient of d(x dy) must be 1, got {}",
            term.coefficient
        );
        // Canonical basis: xid < yid (x was interned before y above).
        assert_eq!(term.basis, vec![xid, yid]);
    }

    #[test]
    fn test_exterior_derivative_of_y_dx() {
        // d(y · dx) = dy ∧ dx = -dx ∧ dy.
        // So the coefficient of the canonical term (dx∧dy) should be -1.
        let xid = sid("ed_ydx_x");
        let yid = sid("ed_ydx_y");

        let omega = DifferentialForm {
            degree: 1,
            terms: vec![FormTerm {
                coefficient: sym("ed_ydx_y"),
                basis: vec![xid],
            }],
        };

        let d_omega = exterior_derivative(&omega, &[xid, yid]);
        assert_eq!(d_omega.degree, 2);
        assert_eq!(d_omega.terms.len(), 1);

        let term = &d_omega.terms[0];
        // Coefficient should be -1.
        let neg_one = normalize::neg(Expr::int(1));
        assert_eq!(
            *term.coefficient.as_ref(),
            *neg_one.as_ref(),
            "coefficient of d(y dx) must be -1, got {}",
            term.coefficient
        );
        assert_eq!(term.basis, vec![xid, yid]);
    }

    #[test]
    fn test_poincare_dd_zero_on_function() {
        // For a smooth 0-form f = x*y, d(df) = 0.
        // df = y dx + x dy
        // d(df) = d(y dx) + d(x dy)
        //       = dy∧dx + dx∧dy
        //       = -dx∧dy + dx∧dy = 0
        let xid = sid("poinc_x");
        let yid = sid("poinc_y");

        let x = sym("poinc_x");
        let y = sym("poinc_y");

        // f = x*y as a 0-form.
        let xy = normalize::mul(x, y);
        let f = DifferentialForm {
            degree: 0,
            terms: vec![FormTerm {
                coefficient: xy,
                basis: vec![],
            }],
        };

        // First exterior derivative: df.
        let df = exterior_derivative(&f, &[xid, yid]);
        assert_eq!(df.degree, 1);

        // Second exterior derivative: d(df).
        let ddf = exterior_derivative(&df, &[xid, yid]);
        assert_eq!(ddf.degree, 2);

        // All terms must have zero coefficient (i.e., the term list is empty,
        // because we drop zero terms in exterior_derivative).
        assert!(
            ddf.terms.is_empty(),
            "d(d(f)) must be zero for any smooth function f; terms: {:?}",
            ddf.terms
        );
    }

    #[test]
    fn test_poincare_dd_zero_on_one_form() {
        // ω = x² dy.  dω = 2x dx∧dy.  d(dω) = 0.
        let xid = sid("p1f_x");
        let yid = sid("p1f_y");

        let x = sym("p1f_x");
        let x_sq = normalize::mul(x.clone(), x);

        let omega = DifferentialForm {
            degree: 1,
            terms: vec![FormTerm {
                coefficient: x_sq,
                basis: vec![yid],
            }],
        };

        let d_omega = exterior_derivative(&omega, &[xid, yid]);
        let ddomega = exterior_derivative(&d_omega, &[xid, yid]);

        assert!(
            ddomega.terms.is_empty(),
            "d² = 0 must hold on any 1-form; d(d(ω)) terms: {:?}",
            ddomega.terms
        );
    }
}
