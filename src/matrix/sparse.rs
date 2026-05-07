//! Compressed Sparse Row (CSR) matrix type for symbolic matrices.
//!
//! Stores only structurally non-zero elements using the standard CSR
//! representation: `values` and `col_indices` are parallel arrays of
//! non-zero entries; `row_ptrs[i]..row_ptrs[i+1]` indexes into them
//! for row `i`.  All values are `Arc<Expr>` — the canonical internal
//! representation (Architecture Rule 1).

use std::sync::Arc;

use crate::numeric::{expr::Expr, normalize};

use super::{MatrixError, MatrixExpr, MatrixResult};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Return `true` when the expression is structurally the integer/float zero.
fn is_structural_zero(e: &Arc<Expr>) -> bool {
    e.is_zero()
}

// ── type ──────────────────────────────────────────────────────────────────────

/// A symbolic matrix stored in Compressed Sparse Row (CSR) format.
///
/// Only structurally non-zero elements are stored.  Zero elements
/// (detected via `Expr::is_zero()`) are never inserted; elements that
/// later evaluate to zero are still stored if they were symbolically
/// non-zero at construction time (conservative: no algebraic simplifier).
///
/// # Examples
///
/// ```
/// use thales::matrix::CsrMatrix;
/// use thales::numeric::expr::Expr;
///
/// // 3×3 sparse matrix with three non-zero entries
/// let triplets = vec![
///     (0, 0, Expr::int(1)),
///     (1, 2, Expr::int(5)),
///     (2, 1, Expr::int(3)),
/// ];
/// let csr = CsrMatrix::from_triplets(3, 3, triplets);
/// assert_eq!(csr.nnz(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    rows: usize,
    cols: usize,
    values: Vec<Arc<Expr>>,
    col_indices: Vec<usize>,
    /// `row_ptrs[i]` is the index into `values`/`col_indices` where row `i`
    /// begins.  `row_ptrs[rows]` == `values.len()`.
    row_ptrs: Vec<usize>,
}

impl CsrMatrix {
    // ── constructors ─────────────────────────────────────────────────────────

    /// Build a CSR matrix from an unordered list of `(row, col, value)` triplets.
    ///
    /// Triplets with structurally-zero values are silently dropped.
    /// Duplicate `(row, col)` positions are allowed; they are stored
    /// as separate entries (the last write wins for `get`, but both
    /// appear in the stored arrays — if you need deduplication, ensure
    /// unique positions before calling).
    ///
    /// # Panics
    ///
    /// Panics in debug builds if any `row >= rows` or `col >= cols`.
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        mut triplets: Vec<(usize, usize, Arc<Expr>)>,
    ) -> Self {
        // Drop structural zeros.
        triplets.retain(|(_, _, v)| !is_structural_zero(v));

        // Sort by (row, col) for canonical CSR order.
        triplets.sort_by_key(|(r, c, _)| (*r, *c));

        let nnz = triplets.len();
        let mut values = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut row_ptrs = vec![0usize; rows + 1];

        for (r, c, v) in &triplets {
            debug_assert!(*r < rows, "row index {r} out of bounds (rows={rows})");
            debug_assert!(*c < cols, "col index {c} out of bounds (cols={cols})");
            row_ptrs[r + 1] += 1;
            col_indices.push(*c);
            values.push(v.clone());
        }

        // Convert per-row counts to prefix sums.
        for i in 1..=rows {
            row_ptrs[i] += row_ptrs[i - 1];
        }

        Self {
            rows,
            cols,
            values,
            col_indices,
            row_ptrs,
        }
    }

    /// Convert a dense [`MatrixExpr`] to CSR format, dropping zero entries.
    pub fn from_dense(m: &MatrixExpr) -> Self {
        let rows = m.rows();
        let cols = m.cols();
        let mut triplets = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let v = m.get(r, c).expect("indices in bounds");
                if !is_structural_zero(v) {
                    triplets.push((r, c, v.clone()));
                }
            }
        }
        Self::from_triplets(rows, cols, triplets)
    }

    // ── conversion ───────────────────────────────────────────────────────────

    /// Convert CSR back to a dense [`MatrixExpr`].
    ///
    /// Missing positions are filled with `Expr::int(0)`.
    ///
    /// # Errors
    ///
    /// Returns `MatrixError::InvalidOperation` if the matrix is empty
    /// (0 rows or 0 cols).
    pub fn to_dense(&self) -> MatrixResult<MatrixExpr> {
        if self.rows == 0 || self.cols == 0 {
            return Err(MatrixError::InvalidOperation(
                "cannot convert empty CSR matrix to dense form".to_string(),
            ));
        }

        let mut elements: Vec<Vec<Arc<Expr>>> = (0..self.rows)
            .map(|_| (0..self.cols).map(|_| Expr::int(0)).collect())
            .collect();

        for r in 0..self.rows {
            let start = self.row_ptrs[r];
            let end = self.row_ptrs[r + 1];
            for idx in start..end {
                elements[r][self.col_indices[idx]] = self.values[idx].clone();
            }
        }

        MatrixExpr::from_expr_elements(elements)
    }

    // ── element access ────────────────────────────────────────────────────────

    /// Look up the value at `(row, col)`.
    ///
    /// Returns `None` when the position is a structural zero (not stored).
    /// Uses a linear scan over the (usually small) non-zero range of the row.
    pub fn get(&self, row: usize, col: usize) -> Option<&Arc<Expr>> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        let start = self.row_ptrs[row];
        let end = self.row_ptrs[row + 1];
        let slice = &self.col_indices[start..end];
        // Binary search since triplets are sorted by (row, col).
        match slice.binary_search(&col) {
            Ok(offset) => Some(&self.values[start + offset]),
            Err(_) => None,
        }
    }

    // ── bulk operations ───────────────────────────────────────────────────────

    /// Sparse matrix-vector multiply: computes `A * v`.
    ///
    /// Returns a vector of length `self.rows`.  Entries that have no
    /// contribution from any stored non-zero are set to `Expr::int(0)`.
    ///
    /// # Panics
    ///
    /// Panics if `v.len() != self.cols`.
    pub fn mul_vec(&self, v: &[Arc<Expr>]) -> Vec<Arc<Expr>> {
        assert_eq!(
            v.len(),
            self.cols,
            "vector length {} does not match matrix cols {}",
            v.len(),
            self.cols
        );

        (0..self.rows)
            .map(|r| {
                let start = self.row_ptrs[r];
                let end = self.row_ptrs[r + 1];
                let mut acc: Arc<Expr> = Expr::int(0);
                for idx in start..end {
                    let prod =
                        normalize::mul(self.values[idx].clone(), v[self.col_indices[idx]].clone());
                    acc = normalize::add(acc, prod);
                }
                acc
            })
            .collect()
    }

    /// Compute the transpose as a new `CsrMatrix`.
    ///
    /// Implemented via COO round-trip: collect `(col, row, val)` triplets
    /// from the current matrix and build a `(cols × rows)` CSR matrix.
    pub fn transpose(&self) -> Self {
        let mut triplets: Vec<(usize, usize, Arc<Expr>)> = Vec::with_capacity(self.values.len());
        for r in 0..self.rows {
            let start = self.row_ptrs[r];
            let end = self.row_ptrs[r + 1];
            for idx in start..end {
                // Swap (row, col) → (col, row) for the transposed matrix.
                triplets.push((self.col_indices[idx], r, self.values[idx].clone()));
            }
        }
        Self::from_triplets(self.cols, self.rows, triplets)
    }

    // ── accessors ─────────────────────────────────────────────────────────────

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of stored (structurally non-zero) entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::numeric::{evaluation::evaluate, SymbolId};

    use super::*;

    /// Evaluate a single `Arc<Expr>` to `f64`, panicking on failure.
    fn eval(e: &Arc<Expr>) -> f64 {
        let empty: HashMap<SymbolId, f64> = HashMap::new();
        evaluate(e, &empty).expect("expression must be fully numeric")
    }

    #[test]
    fn fast_test_csr_from_triplets() {
        // 3×3 sparse with three explicit entries.
        let triplets = vec![
            (0, 0, Expr::int(1)),
            (1, 2, Expr::int(5)),
            (2, 1, Expr::int(3)),
        ];
        let csr = CsrMatrix::from_triplets(3, 3, triplets);
        assert_eq!(csr.rows(), 3);
        assert_eq!(csr.cols(), 3);
        assert_eq!(csr.nnz(), 3);
    }

    #[test]
    fn fast_test_csr_get() {
        let triplets = vec![
            (0, 0, Expr::int(7)),
            (1, 1, Expr::int(9)),
            (2, 2, Expr::int(4)),
        ];
        let csr = CsrMatrix::from_triplets(3, 3, triplets);

        // Existing entries.
        assert_eq!(eval(csr.get(0, 0).unwrap()), 7.0);
        assert_eq!(eval(csr.get(1, 1).unwrap()), 9.0);
        assert_eq!(eval(csr.get(2, 2).unwrap()), 4.0);

        // Structural zeros — not stored.
        assert!(csr.get(0, 1).is_none());
        assert!(csr.get(1, 0).is_none());
        assert!(csr.get(2, 0).is_none());
    }

    #[test]
    fn fast_test_csr_roundtrip() {
        // Build a dense 3×3, convert to sparse, convert back.
        let m = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(0), Expr::int(2)],
            vec![Expr::int(0), Expr::int(3), Expr::int(0)],
            vec![Expr::int(4), Expr::int(0), Expr::int(5)],
        ])
        .unwrap();

        let csr = CsrMatrix::from_dense(&m);
        assert_eq!(csr.nnz(), 5); // 0 entries excluded.

        let m2 = csr.to_dense().unwrap();
        assert_eq!(m2.rows(), 3);
        assert_eq!(m2.cols(), 3);

        // Every element must match the original.
        for r in 0..3 {
            for c in 0..3 {
                let orig = eval(m.get(r, c).unwrap());
                let back = eval(m2.get(r, c).unwrap());
                assert_eq!(
                    orig, back,
                    "mismatch at ({r},{c}): original={orig}, roundtrip={back}"
                );
            }
        }
    }

    #[test]
    fn fast_test_csr_spmv() {
        // A = [[2, 0, 0], [0, 3, 1], [1, 0, 4]], v = [1, 2, 3]
        // Expected: [2, 9, 13]
        let triplets = vec![
            (0, 0, Expr::int(2)),
            (1, 1, Expr::int(3)),
            (1, 2, Expr::int(1)),
            (2, 0, Expr::int(1)),
            (2, 2, Expr::int(4)),
        ];
        let csr = CsrMatrix::from_triplets(3, 3, triplets);

        let v = vec![Expr::int(1), Expr::int(2), Expr::int(3)];
        let result = csr.mul_vec(&v);

        assert_eq!(result.len(), 3);
        assert_eq!(eval(&result[0]), 2.0);
        assert_eq!(eval(&result[1]), 9.0);
        assert_eq!(eval(&result[2]), 13.0);
    }

    #[test]
    fn fast_test_csr_transpose() {
        // A = [[1, 2], [3, 4], [5, 6]]  (3×2)
        // Aᵀ should be [[1, 3, 5], [2, 4, 6]]  (2×3)
        let triplets = vec![
            (0, 0, Expr::int(1)),
            (0, 1, Expr::int(2)),
            (1, 0, Expr::int(3)),
            (1, 1, Expr::int(4)),
            (2, 0, Expr::int(5)),
            (2, 1, Expr::int(6)),
        ];
        let csr = CsrMatrix::from_triplets(3, 2, triplets);
        let t = csr.transpose();

        assert_eq!(t.rows(), 2);
        assert_eq!(t.cols(), 3);
        assert_eq!(t.nnz(), 6);

        // Spot-check transposed entries.
        assert_eq!(eval(t.get(0, 0).unwrap()), 1.0);
        assert_eq!(eval(t.get(0, 1).unwrap()), 3.0);
        assert_eq!(eval(t.get(0, 2).unwrap()), 5.0);
        assert_eq!(eval(t.get(1, 0).unwrap()), 2.0);
        assert_eq!(eval(t.get(1, 1).unwrap()), 4.0);
        assert_eq!(eval(t.get(1, 2).unwrap()), 6.0);
    }

    #[test]
    fn fast_test_csr_zero_matrix() {
        // All-zero 4×4 — no entries should be stored.
        let m = MatrixExpr::zero(4, 4);
        let csr = CsrMatrix::from_dense(&m);
        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.rows(), 4);
        assert_eq!(csr.cols(), 4);

        // get always returns None for zero matrix.
        for r in 0..4 {
            for c in 0..4 {
                assert!(
                    csr.get(r, c).is_none(),
                    "({r},{c}) should be structural zero"
                );
            }
        }

        // to_dense should reconstruct an all-zero dense matrix.
        let dense = csr.to_dense().unwrap();
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(eval(dense.get(r, c).unwrap()), 0.0);
            }
        }
    }

    #[test]
    fn fast_test_csr_1x1() {
        // 1×1 non-zero.
        let csr = CsrMatrix::from_triplets(1, 1, vec![(0, 0, Expr::int(42))]);
        assert_eq!(csr.nnz(), 1);
        assert_eq!(eval(csr.get(0, 0).unwrap()), 42.0);

        let dense = csr.to_dense().unwrap();
        assert_eq!(eval(dense.get(0, 0).unwrap()), 42.0);

        let t = csr.transpose();
        assert_eq!(t.rows(), 1);
        assert_eq!(t.cols(), 1);
        assert_eq!(eval(t.get(0, 0).unwrap()), 42.0);

        // 1×1 zero.
        let zero = CsrMatrix::from_triplets(1, 1, vec![(0, 0, Expr::int(0))]);
        assert_eq!(zero.nnz(), 0);
        assert!(zero.get(0, 0).is_none());
    }
}
