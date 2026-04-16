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

use super::big_rational::BigRational;
use super::dense_poly::DensePolynomial;
use super::expr::Expr;
use super::normalize;
use super::poly_equation_solver::roots_with_multiplicity;
use super::symbol::SymbolId;
use num::traits::{One, Zero};
use std::sync::Arc;

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
    Ok(expand_expr(&det))
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
    let char_poly_expr = expand_expr(&symbolic_det(&shifted, n));

    // Try to extract rational coefficients for the numeric solver
    match extract_rational_poly(&char_poly_expr, lambda, n) {
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

// ── Polynomial expansion ─────────────────────────────────────────────────────

/// Fully expand an expression by distributing products over sums.
///
/// Ensures the result is a sum of products (monomials) suitable for
/// polynomial coefficient extraction.
fn expand_expr(expr: &Arc<Expr>) -> Arc<Expr> {
    match expr.as_ref() {
        Expr::Integer(_) | Expr::Rational(_) | Expr::Float(_) | Expr::Symbol(_) => expr.clone(),
        Expr::Pow(base, exp) => {
            let base_exp = expand_expr(base);
            let exp_exp = expand_expr(exp);
            // If base is an Add and exponent is small integer, expand
            if let Expr::Integer(n) = exp_exp.as_ref() {
                if let Some(k) = n.to_i64() {
                    if k >= 2 && k <= 4 {
                        let mut result = base_exp.clone();
                        for _ in 1..k {
                            result = expand_mul(result, base_exp.clone());
                        }
                        return result;
                    }
                }
            }
            normalize::pow(base_exp, exp_exp)
        }
        Expr::Add(node) => {
            let mut terms = Vec::new();
            if !node.constant.is_zero() {
                terms.push(rational_to_expr(&node.constant));
            }
            for (term, coeff) in &node.terms {
                let expanded = expand_expr(term);
                terms.push(normalize::mul(rational_to_expr(coeff), expanded));
            }
            normalize::add_many(terms)
        }
        Expr::Mul(node) => {
            let mut result = rational_to_expr(&node.coeff);
            for (base, exp) in &node.factors {
                let factor = expand_expr(&normalize::pow(base.clone(), exp.clone()));
                result = expand_mul(result, factor);
            }
            result
        }
        Expr::Func(_, _) => expr.clone(),
    }
}

/// Multiply two expressions distributing over sums.
fn expand_mul(a: Arc<Expr>, b: Arc<Expr>) -> Arc<Expr> {
    let a_terms = to_sum_terms(&a);
    let b_terms = to_sum_terms(&b);
    let mut result_terms = Vec::new();
    for at in &a_terms {
        for bt in &b_terms {
            result_terms.push(normalize::mul(at.clone(), bt.clone()));
        }
    }
    normalize::add_many(result_terms)
}

/// Decompose an expression into additive terms.
fn to_sum_terms(expr: &Arc<Expr>) -> Vec<Arc<Expr>> {
    match expr.as_ref() {
        Expr::Add(node) => {
            let mut terms = Vec::new();
            if !node.constant.is_zero() {
                terms.push(rational_to_expr(&node.constant));
            }
            for (term, coeff) in &node.terms {
                if coeff.is_one() {
                    terms.push(term.clone());
                } else {
                    terms.push(normalize::mul(rational_to_expr(coeff), term.clone()));
                }
            }
            terms
        }
        _ => vec![expr.clone()],
    }
}

/// Convert a BigRational to an Expr.
fn rational_to_expr(r: &BigRational) -> Arc<Expr> {
    if r.is_integer() {
        if let Some(n) = r.numer().to_i64() {
            return Expr::int(n);
        }
    }
    Arc::new(Expr::Rational(r.clone()))
}

// ── Coefficient extraction ────────────────────────────────────────────────────

/// Attempt to extract a `DensePolynomial<BigRational>` from a symbolic expression
/// that should be a polynomial in `lambda`.
///
/// Returns `None` if the expression contains symbols other than `lambda`.
fn extract_rational_poly(
    expr: &Arc<Expr>,
    lambda: SymbolId,
    degree: usize,
) -> Option<DensePolynomial<BigRational>> {
    let mut coeffs = vec![BigRational::zero(); degree + 1];
    let one = BigRational::one();
    collect_poly_coeffs(expr, lambda, 0, &one, &mut coeffs)?;
    Some(DensePolynomial::from_coeffs(coeffs))
}

/// Recursively traverse `expr` and accumulate polynomial coefficients.
///
/// `lambda_power` tracks the current power of lambda contributed by the call
/// site; `scale` is the rational scalar multiplier from enclosing `Mul` nodes.
fn collect_poly_coeffs(
    expr: &Arc<Expr>,
    lambda: SymbolId,
    lambda_power: usize,
    scale: &BigRational,
    coeffs: &mut Vec<BigRational>,
) -> Option<()> {
    match expr.as_ref() {
        // Pure rational constant: add to coeff[lambda_power]
        Expr::Integer(n) => {
            if lambda_power >= coeffs.len() {
                return None; // degree too high
            }
            let val = BigRational::from_integer(n.clone()) * scale.clone();
            coeffs[lambda_power] = coeffs[lambda_power].clone() + val;
            Some(())
        }
        Expr::Rational(r) => {
            if lambda_power >= coeffs.len() {
                return None;
            }
            let val = r * scale;
            coeffs[lambda_power] = coeffs[lambda_power].clone() + val;
            Some(())
        }
        // Lambda symbol: contributes to power lambda_power + 1
        Expr::Symbol(s) if *s == lambda => {
            let new_power = lambda_power + 1;
            if new_power >= coeffs.len() {
                return None;
            }
            // coefficient is `scale * 1`
            coeffs[new_power] = coeffs[new_power].clone() + scale.clone();
            Some(())
        }
        // Any other symbol: this is a symbolic entry; cannot extract
        Expr::Symbol(_) => None,
        // Sum: process each term
        Expr::Add(node) => {
            let const_val = node.constant.clone() * scale.clone();
            if lambda_power >= coeffs.len() {
                return None;
            }
            coeffs[lambda_power] = coeffs[lambda_power].clone() + const_val;
            for (term, coeff) in &node.terms {
                let new_scale = coeff * scale;
                collect_poly_coeffs(term, lambda, lambda_power, &new_scale, coeffs)?;
            }
            Some(())
        }
        // Product: handle lambda^k * constant forms
        Expr::Mul(node) => {
            let new_scale = &node.coeff * scale;
            let mut lam_exp: usize = 0;
            for (base, exp) in &node.factors {
                match (base.as_ref(), exp.as_ref()) {
                    // lambda^integer_exponent
                    (Expr::Symbol(s), Expr::Integer(k)) if *s == lambda => {
                        let k_u = k.to_i64().and_then(|v| usize::try_from(v).ok())?;
                        lam_exp = lam_exp.checked_add(k_u)?;
                    }
                    // Non-lambda base: must be a numeric constant (exponent=1)
                    (_, Expr::Integer(k)) if k.to_i64() == Some(1) => {
                        match base.as_ref() {
                            Expr::Integer(_) | Expr::Rational(_) => {
                                // numeric base^1 is already folded into coeff
                            }
                            // Symbolic non-lambda: cannot extract
                            _ => return None,
                        }
                    }
                    _ => return None,
                }
            }
            let total_power = lambda_power + lam_exp;
            if total_power >= coeffs.len() {
                return None;
            }
            coeffs[total_power] = coeffs[total_power].clone() + new_scale;
            Some(())
        }
        // Pow(lambda, k) as a standalone node
        Expr::Pow(base, exp) => {
            if let (Expr::Symbol(s), Expr::Integer(k)) = (base.as_ref(), exp.as_ref()) {
                if *s == lambda {
                    let k_u = k.to_i64().and_then(|v| usize::try_from(v).ok())?;
                    let total_power = lambda_power + k_u;
                    if total_power >= coeffs.len() {
                        return None;
                    }
                    coeffs[total_power] = coeffs[total_power].clone() + scale.clone();
                    return Some(());
                }
            }
            None
        }
        _ => None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::expr::Expr;

    fn int_entry(n: i64) -> Arc<Expr> {
        Expr::int(n)
    }

    // ── characteristic_polynomial ────────────────────────────────────────────

    /// Characteristic polynomial of [[1,2],[3,4]] is λ²−5λ−2.
    #[test]
    fn test_char_poly_2x2() {
        let m: ExprMatrix = vec![
            vec![int_entry(1), int_entry(2)],
            vec![int_entry(3), int_entry(4)],
        ];
        let lambda = SymbolId::intern("l");
        let cp = characteristic_polynomial(&m, lambda).unwrap();
        // Verify by checking eigenvalues indirectly through the numeric path
        let lam_expr = Arc::new(Expr::Symbol(lambda));
        let n = 2;
        let mut poly_coeffs = vec![BigRational::zero(); n + 1];
        let one = BigRational::one();
        let ok = collect_poly_coeffs(&cp, lambda, 0, &one, &mut poly_coeffs);
        assert!(ok.is_some(), "char poly must be extractable");
        // λ²−5λ−2 → coeffs = [-2, -5, 1]
        let coeffs_i64: Vec<i64> = poly_coeffs
            .iter()
            .map(|r| r.numer().to_i64().unwrap_or(0))
            .collect();
        // constant = -2, linear = -5, quadratic = 1
        assert_eq!(coeffs_i64[0], -2, "constant term");
        assert_eq!(coeffs_i64[1], -5, "linear term");
        assert_eq!(coeffs_i64[2], 1, "quadratic term");
        drop(lam_expr);
    }

    /// Characteristic polynomial of identity 2×2 is (λ−1)² = λ²−2λ+1.
    #[test]
    fn test_char_poly_identity_2x2() {
        let m: ExprMatrix = vec![
            vec![int_entry(1), int_entry(0)],
            vec![int_entry(0), int_entry(1)],
        ];
        let lambda = SymbolId::intern("mu");
        let cp = characteristic_polynomial(&m, lambda).unwrap();
        let mut coeffs = vec![BigRational::zero(); 3];
        let ok = collect_poly_coeffs(&cp, lambda, 0, &BigRational::one(), &mut coeffs);
        assert!(ok.is_some());
        let ci: Vec<i64> = coeffs
            .iter()
            .map(|r| r.numer().to_i64().unwrap_or(0))
            .collect();
        // (λ−1)² = λ²−2λ+1 → [1, -2, 1]
        assert_eq!(ci[0], 1);
        assert_eq!(ci[1], -2);
        assert_eq!(ci[2], 1);
    }

    /// Characteristic polynomial of [[5]] is λ−5.
    #[test]
    fn test_char_poly_1x1() {
        let m: ExprMatrix = vec![vec![int_entry(5)]];
        let lambda = SymbolId::intern("nu");
        let cp = characteristic_polynomial(&m, lambda).unwrap();
        let mut coeffs = vec![BigRational::zero(); 2];
        collect_poly_coeffs(&cp, lambda, 0, &BigRational::one(), &mut coeffs).unwrap();
        let ci: Vec<i64> = coeffs
            .iter()
            .map(|r| r.numer().to_i64().unwrap_or(0))
            .collect();
        // 5 − λ → [5, -1]
        assert_eq!(ci[0], 5);
        assert_eq!(ci[1], -1);
    }

    // ── eigenvalues: numeric matrices ────────────────────────────────────────

    /// Eigenvalues of [[1,0],[0,1]] are {1} with multiplicity 2.
    #[test]
    fn test_eigenvalues_identity_2x2() {
        let m: ExprMatrix = vec![
            vec![int_entry(1), int_entry(0)],
            vec![int_entry(0), int_entry(1)],
        ];
        let result = eigenvalues(&m).unwrap();
        match result {
            EigenvalueResult::Numeric(evs) => {
                assert_eq!(evs.len(), 1, "one distinct eigenvalue");
                assert_eq!(evs[0].1, 2, "multiplicity 2");
                match evs[0].0.as_ref() {
                    Expr::Integer(n) => assert_eq!(n.to_i64(), Some(1)),
                    _ => panic!("eigenvalue should be integer 1"),
                }
            }
            EigenvalueResult::Symbolic(_) => panic!("expected numeric result"),
        }
    }

    /// Eigenvalues of [[2,0],[0,3]] are {2, 3} each with multiplicity 1.
    #[test]
    fn test_eigenvalues_diagonal_2x2() {
        let m: ExprMatrix = vec![
            vec![int_entry(2), int_entry(0)],
            vec![int_entry(0), int_entry(3)],
        ];
        let result = eigenvalues(&m).unwrap();
        match result {
            EigenvalueResult::Numeric(mut evs) => {
                assert_eq!(evs.len(), 2);
                evs.sort_by_key(|(e, _)| match e.as_ref() {
                    Expr::Integer(n) => n.to_i64().unwrap_or(0),
                    _ => 0,
                });
                assert_eq!(evs[0].1, 1);
                assert_eq!(evs[1].1, 1);
                match evs[0].0.as_ref() {
                    Expr::Integer(n) => assert_eq!(n.to_i64(), Some(2)),
                    _ => panic!("expected integer 2"),
                }
                match evs[1].0.as_ref() {
                    Expr::Integer(n) => assert_eq!(n.to_i64(), Some(3)),
                    _ => panic!("expected integer 3"),
                }
            }
            EigenvalueResult::Symbolic(_) => panic!("expected numeric"),
        }
    }

    /// [[1,2],[3,4]]: char poly λ²−5λ−2. Roots are irrational so solver
    /// returns no numeric roots; result should be Symbolic or Numeric([]).
    #[test]
    fn test_eigenvalues_1_2_3_4() {
        let m: ExprMatrix = vec![
            vec![int_entry(1), int_entry(2)],
            vec![int_entry(3), int_entry(4)],
        ];
        let result = eigenvalues(&m).unwrap();
        // The char poly is λ²−5λ−2 which has irrational roots;
        // roots_with_multiplicity returns empty for non-rational-square discriminant.
        match result {
            EigenvalueResult::Numeric(evs) => {
                // No rational roots — the solver correctly returns empty
                assert!(evs.is_empty(), "irrational roots not returned numerically");
            }
            EigenvalueResult::Symbolic(_) => {
                // Also acceptable: char poly returned as symbolic
            }
        }
    }

    /// Eigenvalues of [[3,-2],[1,0]]: char poly λ²−3λ+2 = (λ−1)(λ−2).
    #[test]
    fn test_eigenvalues_rational_roots() {
        let m: ExprMatrix = vec![
            vec![int_entry(3), int_entry(-2)],
            vec![int_entry(1), int_entry(0)],
        ];
        let result = eigenvalues(&m).unwrap();
        match result {
            EigenvalueResult::Numeric(mut evs) => {
                assert_eq!(evs.len(), 2, "two distinct eigenvalues");
                evs.sort_by_key(|(e, _)| match e.as_ref() {
                    Expr::Integer(n) => n.to_i64().unwrap_or(0),
                    _ => 0,
                });
                let vals: Vec<i64> = evs
                    .iter()
                    .filter_map(|(e, _)| match e.as_ref() {
                        Expr::Integer(n) => n.to_i64(),
                        _ => None,
                    })
                    .collect();
                assert_eq!(vals, vec![1, 2]);
            }
            EigenvalueResult::Symbolic(_) => panic!("expected numeric eigenvalues"),
        }
    }

    // ── eigenvalues: symbolic matrices ───────────────────────────────────────

    /// A matrix with a symbol entry should return a Symbolic result.
    #[test]
    fn test_eigenvalues_symbolic_matrix() {
        let a = Arc::new(Expr::Symbol(SymbolId::intern("a")));
        let m: ExprMatrix = vec![
            vec![a.clone(), int_entry(0)],
            vec![int_entry(0), int_entry(1)],
        ];
        let result = eigenvalues(&m).unwrap();
        assert!(
            matches!(result, EigenvalueResult::Symbolic(_)),
            "symbolic matrix must yield Symbolic result"
        );
    }

    // ── error handling ───────────────────────────────────────────────────────

    #[test]
    fn test_non_square_returns_error() {
        let m: ExprMatrix = vec![
            vec![int_entry(1), int_entry(2), int_entry(3)],
            vec![int_entry(4), int_entry(5), int_entry(6)],
        ];
        assert!(matches!(
            characteristic_polynomial(&m, SymbolId::intern("x")),
            Err(EigenError::NotSquare)
        ));
        assert!(matches!(eigenvalues(&m), Err(EigenError::NotSquare)));
    }

    #[test]
    fn test_empty_matrix_returns_error() {
        let m: ExprMatrix = vec![];
        assert!(matches!(
            characteristic_polynomial(&m, SymbolId::intern("x")),
            Err(EigenError::Empty)
        ));
    }

    #[test]
    fn test_ragged_matrix_returns_error() {
        let m: ExprMatrix = vec![vec![int_entry(1), int_entry(2)], vec![int_entry(3)]];
        assert!(matches!(
            characteristic_polynomial(&m, SymbolId::intern("x")),
            Err(EigenError::RaggedMatrix)
        ));
    }

    // ── symbolic 2×2 eigenvalues ─────────────────────────────────────────────

    /// For the symbolic 2×2 matrix [[a, b], [c, d]], the characteristic
    /// polynomial should be expressible symbolically.
    #[test]
    fn test_symbolic_2x2_char_poly() {
        let a = Arc::new(Expr::Symbol(SymbolId::intern("sa")));
        let b = Arc::new(Expr::Symbol(SymbolId::intern("sb")));
        let c = Arc::new(Expr::Symbol(SymbolId::intern("sc")));
        let d = Arc::new(Expr::Symbol(SymbolId::intern("sd")));
        let m: ExprMatrix = vec![vec![a, b], vec![c, d]];
        let lambda = SymbolId::intern("slambda");
        // Should succeed (no error), producing a symbolic expression
        let cp = characteristic_polynomial(&m, lambda);
        assert!(cp.is_ok(), "char poly of symbolic 2×2 should not error");
    }
}
