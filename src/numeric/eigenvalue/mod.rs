//! Symbolic eigenvalue computation via characteristic polynomial.
//!
//! Computes eigenvalues of a square matrix by:
//! 1. Building the characteristic polynomial `det(A - λI)` symbolically.
//! 2. Factoring or solving the polynomial for the eigenvalues.
//!
//! # Numeric matrices
//!
//! When all matrix entries evaluate to rational constants, the characteristic
//! polynomial is converted to a [`DensePolynomial<BigRational>`] and solved
//! with [`roots_with_multiplicity`], which tracks algebraic multiplicity.
//!
//! # Symbolic matrices
//!
//! When entries contain symbolic sub-expressions, the characteristic polynomial
//! is returned as an [`Expr`] for the caller to analyse. Eigenvalues cannot be
//! determined automatically in this case.
//!
//! # Examples
//!
//! ```rust
//! use thales::numeric::{
//!     eigenvalue::{characteristic_polynomial, eigenvalues, ExprMatrix, EigenvalueResult},
//!     BigRational, Expr, SymbolId,
//! };
//!
//! // Matrix [[1, 2], [3, 4]]
//! let m: ExprMatrix = vec![
//!     vec![Expr::int(1), Expr::int(2)],
//!     vec![Expr::int(3), Expr::int(4)],
//! ];
//! let lambda = SymbolId::intern("lambda");
//! let char_poly = characteristic_polynomial(&m, lambda).unwrap();
//!
//! // Eigenvalues of [[1,2],[3,4]] come from λ²−5λ−2 = 0
//! let result = eigenvalues(&m).unwrap();
//! match result {
//!     EigenvalueResult::Numeric(evs) => assert_eq!(evs.len(), 2),
//!     EigenvalueResult::Symbolic(_) => panic!("expected numeric eigenvalues"),
//! }
//! ```

use super::expr::Expr;
use super::normalize;
use super::poly_equation_solver::roots_with_multiplicity;
use super::symbol::SymbolId;
use std::sync::Arc;

#[cfg(test)]
pub(crate) use super::big_rational::BigRational;

mod expand;
mod extract;

#[cfg(test)]
mod tests;

// ── Public types ──────────────────────────────────────────────────────────────

/// A matrix whose entries are symbolic expressions.
///
/// Stored as a row-major `Vec<Vec<Arc<Expr>>>`. Each inner `Vec` is one row,
/// all rows must have equal length.
pub type ExprMatrix = Vec<Vec<Arc<Expr>>>;

/// Error type for eigenvalue operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EigenError {
    /// Matrix is not square.
    NotSquare,
    /// Matrix is empty.
    Empty,
    /// Row lengths are inconsistent.
    RaggedMatrix,
}

impl std::fmt::Display for EigenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EigenError::NotSquare => write!(f, "matrix must be square for eigenvalue computation"),
            EigenError::Empty => write!(f, "matrix is empty"),
            EigenError::RaggedMatrix => write!(f, "all rows must have the same length"),
        }
    }
}

impl std::error::Error for EigenError {}

/// Result of an eigenvalue computation.
#[derive(Clone, Debug)]
pub enum EigenvalueResult {
    /// Exact eigenvalues with algebraic multiplicity, obtained when all
    /// matrix entries are rational constants.
    Numeric(Vec<(Arc<Expr>, usize)>),
    /// The characteristic polynomial as a symbolic expression; eigenvalues
    /// could not be resolved because the matrix has symbolic entries or
    /// the roots are not expressible as rationals.
    Symbolic(Arc<Expr>),
}

// ── Validation ────────────────────────────────────────────────────────────────

/// Return the size of a square `ExprMatrix`, or an error.
fn matrix_size(mat: &ExprMatrix) -> Result<usize, EigenError> {
    if mat.is_empty() {
        return Err(EigenError::Empty);
    }
    let n = mat[0].len();
    if n == 0 {
        return Err(EigenError::Empty);
    }
    for row in mat {
        if row.len() != n {
            return Err(EigenError::RaggedMatrix);
        }
    }
    if mat.len() != n {
        return Err(EigenError::NotSquare);
    }
    Ok(n)
}

// ── Characteristic polynomial ─────────────────────────────────────────────────

/// Compute `det(A - λI)` as a symbolic expression.
///
/// The result is a polynomial in `lambda` with the same type as the entries
/// of `mat`. For a 2×2 matrix `[[a,b],[c,d]]` this yields
/// `λ² − (a+d)λ + (ad−bc)`.
///
/// # Errors
///
/// Returns [`EigenError`] if the matrix is not square or is malformed.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{
///     eigenvalue::{characteristic_polynomial, ExprMatrix},
///     Expr, SymbolId,
/// };
///
/// let m: ExprMatrix = vec![
///     vec![Expr::int(1), Expr::int(2)],
///     vec![Expr::int(3), Expr::int(4)],
/// ];
/// let lambda = SymbolId::intern("lam");
/// let cp = characteristic_polynomial(&m, lambda).unwrap();
/// // cp represents λ² − 5λ − 2
/// assert!(matches!(cp.as_ref(), thales::numeric::Expr::Add(_) | thales::numeric::Expr::Mul(_)));
/// ```
pub fn characteristic_polynomial(
    mat: &ExprMatrix,
    lambda: SymbolId,
) -> Result<Arc<Expr>, EigenError> {
    let n = matrix_size(mat)?;
    // Build (A - λI): entry (i,j) = mat[i][j] - λ * δ(i,j)
    let lam_expr = Arc::new(Expr::Symbol(lambda));
    let shifted = build_shifted(mat, n, &lam_expr);
    let det = symbolic_det(&shifted, n);
    Ok(expand::expand_expr(&det))
}

/// Build the matrix `A - λI` as a flat row-major `Vec<Arc<Expr>>`.
fn build_shifted(mat: &ExprMatrix, n: usize, lam: &Arc<Expr>) -> Vec<Arc<Expr>> {
    let mut entries = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let a_ij = mat[i][j].clone();
            if i == j {
                entries.push(normalize::sub(a_ij, lam.clone()));
            } else {
                entries.push(a_ij);
            }
        }
    }
    entries
}

/// Compute the determinant of an n×n matrix stored as a flat `Vec<Arc<Expr>>`
/// using cofactor (Laplace) expansion along the first row.
///
/// Complexity is O(n!) but eigenvalue computations are normally used on
/// small matrices (2×2 and 3×3) where this is fast.
fn symbolic_det(entries: &[Arc<Expr>], n: usize) -> Arc<Expr> {
    if n == 1 {
        return entries[0].clone();
    }
    if n == 2 {
        // ad - bc
        let ad = normalize::mul(entries[0].clone(), entries[3].clone());
        let bc = normalize::mul(entries[1].clone(), entries[2].clone());
        return normalize::sub(ad, bc);
    }

    // Expand along first row
    let mut terms: Vec<Arc<Expr>> = Vec::with_capacity(n);
    for col in 0..n {
        let minor = cofactor_minor(entries, n, 0, col);
        let minor_det = symbolic_det(&minor, n - 1);
        let entry = entries[col].clone();
        let term = normalize::mul(entry, minor_det);
        // sign: (+1)^(col) — even col positive, odd col negative
        let signed = if col % 2 == 0 {
            term
        } else {
            normalize::neg(term)
        };
        terms.push(signed);
    }
    normalize::add_many(terms)
}

/// Return the minor matrix (entries with row `row` and column `col` removed).
fn cofactor_minor(entries: &[Arc<Expr>], n: usize, row: usize, col: usize) -> Vec<Arc<Expr>> {
    let m = n - 1;
    let mut minor = Vec::with_capacity(m * m);
    for i in 0..n {
        if i == row {
            continue;
        }
        for j in 0..n {
            if j == col {
                continue;
            }
            minor.push(entries[i * n + j].clone());
        }
    }
    minor
}

// ── Eigenvalue computation ────────────────────────────────────────────────────

/// Compute the eigenvalues of a square matrix.
///
/// When all entries are rational constants, returns
/// [`EigenvalueResult::Numeric`] with exact eigenvalues and algebraic
/// multiplicities. Otherwise returns [`EigenvalueResult::Symbolic`] with
/// the characteristic polynomial.
///
/// # Errors
///
/// Returns [`EigenError`] if the matrix is not square or malformed.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{
///     eigenvalue::{eigenvalues, ExprMatrix, EigenvalueResult},
///     Expr,
/// };
///
/// // Identity matrix 2×2 — eigenvalue 1 with multiplicity 2
/// let m: ExprMatrix = vec![
///     vec![Expr::int(1), Expr::int(0)],
///     vec![Expr::int(0), Expr::int(1)],
/// ];
/// let result = eigenvalues(&m).unwrap();
/// match result {
///     EigenvalueResult::Numeric(evs) => {
///         assert_eq!(evs.len(), 1);
///         assert_eq!(evs[0].1, 2);
///     }
///     EigenvalueResult::Symbolic(_) => panic!("expected numeric result"),
/// }
/// ```
pub fn eigenvalues(mat: &ExprMatrix) -> Result<EigenvalueResult, EigenError> {
    let n = matrix_size(mat)?;
    // Use a fresh lambda symbol unlikely to collide with matrix entries
    let lambda = SymbolId::intern("__eigenvalue_lambda__");
    let lam_expr = Arc::new(Expr::Symbol(lambda));
    let shifted = build_shifted(mat, n, &lam_expr);
    let char_poly_expr = expand::expand_expr(&symbolic_det(&shifted, n));

    // Try to extract rational coefficients for the numeric solver
    match extract::extract_rational_poly(&char_poly_expr, lambda, n) {
        Some(dense) => {
            let roots = roots_with_multiplicity(&dense);
            let pairs = roots
                .into_iter()
                .map(|r| (r.root, r.multiplicity))
                .collect();
            Ok(EigenvalueResult::Numeric(pairs))
        }
        None => Ok(EigenvalueResult::Symbolic(char_poly_expr)),
    }
}
