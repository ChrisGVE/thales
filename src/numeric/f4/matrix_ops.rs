//! Sparse row-reduction over `BigRational` for the F4 algorithm.
//!
//! Represents a polynomial as a dense row indexed by a global monomial list,
//! then performs Gaussian elimination to find new reducers.

use crate::numeric::big_rational::BigRational;
use crate::numeric::multivariate_poly::{Monomial, MultivariatePolynomial};
use crate::numeric::ring::{Field, Ring};
use crate::numeric::term_order::MonomialOrder;

// ── Row / column layout ───────────────────────────────────────────────────────

/// A macrorow: a dense vector indexed by a sorted monomial column list.
pub(super) type Row = Vec<BigRational>;

/// Build a sorted, deduplicated list of all monomials in the given polynomials.
///
/// Sorted in ascending order under `order` so the pivot (leading term) ends up
/// at the largest index — matching standard row-echelon convention.
pub(super) fn build_columns<O: MonomialOrder>(
    polys: &[MultivariatePolynomial<BigRational>],
    order: &O,
) -> Vec<Monomial> {
    let mut cols: Vec<Monomial> = polys
        .iter()
        .flat_map(|p| p.iter().map(|(m, _)| m.clone()))
        .collect();
    cols.sort_by(|a, b| order.cmp_monomials(a, b));
    cols.dedup();
    cols
}

/// Encode a polynomial as a dense row over the column index.
pub(super) fn poly_to_row(p: &MultivariatePolynomial<BigRational>, cols: &[Monomial]) -> Row {
    let mut row = vec![BigRational::zero(); cols.len()];
    for (mono, coeff) in p.iter() {
        // Binary search since cols is sorted
        if let Ok(idx) = cols.binary_search_by(|c| {
            // cols are sorted ascending; compare c vs mono
            c.partial_cmp(mono).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            row[idx] = coeff.clone();
        }
    }
    row
}

/// Decode a dense row back to a polynomial.
pub(super) fn row_to_poly(row: &Row, cols: &[Monomial]) -> MultivariatePolynomial<BigRational> {
    let mut p = MultivariatePolynomial::zero();
    for (i, coeff) in row.iter().enumerate() {
        if !coeff.is_zero() {
            p.add_term(cols[i].clone(), coeff.clone());
        }
    }
    p
}

// ── Pivot helpers ─────────────────────────────────────────────────────────────

/// Return the index of the rightmost non-zero entry (the pivot column).
///
/// Because columns are sorted in ascending monomial order, the rightmost
/// non-zero entry corresponds to the leading term under the ordering.
pub(super) fn pivot_col(row: &Row) -> Option<usize> {
    row.iter().rposition(|c| !c.is_zero())
}

// ── Gaussian elimination ──────────────────────────────────────────────────────

/// Row-reduce a matrix in place over `BigRational`.
///
/// Returns a basis of the row space (one row per pivot column, monic).
/// Rows are fully reduced (reduced row echelon form, RREF).
pub(super) fn row_reduce(mut rows: Vec<Row>) -> Vec<Row> {
    if rows.is_empty() {
        return rows;
    }
    let ncols = rows[0].len();
    let mut pivot_row: Vec<Option<usize>> = vec![None; ncols]; // pivot_row[col] = row idx
    let mut row_pivot_col: Vec<Option<usize>> = vec![None; rows.len()];

    let mut cur_row = 0usize;

    // Forward elimination (right-to-left because leading term is rightmost)
    // Work from the rightmost column leftward
    let mut col = ncols;
    while col > 0 && cur_row < rows.len() {
        col -= 1;

        // Find a row at or below cur_row with a non-zero entry in this column
        let found = (cur_row..rows.len()).find(|&r| !rows[r][col].is_zero());
        let pivot = match found {
            Some(r) => r,
            None => {
                col = col.wrapping_add(1); // no pivot in this col; try next
                                           // Actually just continue searching leftward
                continue;
            }
        };

        rows.swap(cur_row, pivot);
        let pivot_r = cur_row;

        // Scale pivot row to make it monic (leading coeff = 1)
        let inv_lead = Field::inv(&rows[pivot_r][col]);
        for v in rows[pivot_r].iter_mut() {
            let new_v = v.clone() * inv_lead.clone();
            *v = new_v;
        }

        // Eliminate this column from all other rows (full RREF)
        for r in 0..rows.len() {
            if r == pivot_r {
                continue;
            }
            let factor = rows[r][col].clone();
            if factor.is_zero() {
                continue;
            }
            for c in 0..ncols {
                let sub = rows[pivot_r][c].clone() * factor.clone();
                let old = rows[r][c].clone();
                rows[r][c] = old - sub;
            }
        }

        pivot_row[col] = Some(pivot_r);
        row_pivot_col[pivot_r] = Some(col);
        cur_row += 1;
    }

    // Collect non-zero rows that have a pivot
    rows.into_iter()
        .filter(|r| pivot_col(r).is_some())
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{BigRational, GrevLex, Monomial, SymbolId};

    fn r(n: i64) -> BigRational {
        BigRational::from(n)
    }

    fn x() -> SymbolId {
        SymbolId::intern("f4mx")
    }
    fn y() -> SymbolId {
        SymbolId::intern("f4my")
    }

    #[test]
    fn test_build_columns_deduplicates() {
        let order = GrevLex::new();
        let p1 = MultivariatePolynomial::monomial(r(1), Monomial::var(x()));
        let p2 = MultivariatePolynomial::monomial(r(2), Monomial::var(x()));
        let cols = build_columns(&[p1, p2], &order);
        assert_eq!(cols.len(), 1);
    }

    #[test]
    fn test_poly_roundtrip() {
        use crate::numeric::MultivariatePolynomial;
        let order = GrevLex::new();
        // p = x + y + 1
        let p = &(&MultivariatePolynomial::var(x()) + &MultivariatePolynomial::var(y()))
            + &MultivariatePolynomial::constant(r(1));
        let cols = build_columns(&[p.clone()], &order);
        let row = poly_to_row(&p, &cols);
        let p2 = row_to_poly(&row, &cols);
        assert_eq!(p, p2);
    }

    #[test]
    fn test_row_reduce_identity() {
        // Two linearly dependent rows → one row after reduction
        // Row 1: [1, 2, 3]  Row 2: [2, 4, 6] (= 2 * row 1)
        let r1 = vec![r(1), r(2), r(3)];
        let r2 = vec![r(2), r(4), r(6)];
        let reduced = row_reduce(vec![r1, r2]);
        assert_eq!(reduced.len(), 1);
    }

    #[test]
    fn test_row_reduce_two_independent() {
        // [1, 0, 1] and [0, 1, 1] are independent
        let r1 = vec![r(1), r(0), r(1)];
        let r2 = vec![r(0), r(1), r(1)];
        let reduced = row_reduce(vec![r1, r2]);
        assert_eq!(reduced.len(), 2);
    }

    #[test]
    fn test_pivot_col() {
        assert_eq!(pivot_col(&vec![r(0), r(0), r(3)]), Some(2));
        assert_eq!(pivot_col(&vec![r(0), r(0), r(0)]), None);
        assert_eq!(pivot_col(&vec![r(1), r(0), r(0)]), Some(0));
    }
}
