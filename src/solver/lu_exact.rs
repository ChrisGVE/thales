//! Exact LU decomposition with row pivoting for square linear systems.
//!
//! Operates entirely on canonical [`Arc<Expr>`] entries. Pivoting selects
//! the first canonical non-zero row (no magnitude threshold needed under
//! exact arithmetic). Singular systems yield a `SolverError::Other` with
//! a `"Singular matrix"` prefix to match the pre-migration behaviour.

use std::collections::HashMap;
use std::sync::Arc;

use crate::ast::Variable;
use crate::numeric::compile::decompile;
use crate::numeric::normalize;
use crate::numeric::Expr;

use super::system::SystemSolution;
use super::types::{SolverError, SolverResult};

/// Solve `A x = b` using exact LU decomposition with row pivoting.
///
/// Requires a square system (`coeffs.len() == variables.len()`). The
/// algorithm decomposes `P A = L U` in-place on a working copy of `A`,
/// applies `P` to `b`, then runs forward and back substitution.
pub(super) fn solve_via_lu_exact(
    coeffs: &[Vec<Arc<Expr>>],
    constants: &[Arc<Expr>],
    variables: &[Variable],
) -> SolverResult<SystemSolution> {
    let n = variables.len();
    let n_eqs = coeffs.len();

    if n_eqs != n {
        return Err(SolverError::Other(format!(
            "solve_via_lu requires a square system ({} eqs, {} vars)",
            n_eqs, n
        )));
    }

    let mut a: Vec<Vec<Arc<Expr>>> = coeffs.iter().map(|r| r.clone()).collect();
    let mut perm: Vec<usize> = (0..n).collect();

    for k in 0..n {
        let Some(p) = (k..n).find(|&i| !a[i][k].is_zero()) else {
            return Err(SolverError::Other(
                "Singular matrix: LU decomposition failed".to_string(),
            ));
        };
        if p != k {
            a.swap(k, p);
            perm.swap(k, p);
        }

        let pivot_val = a[k][k].clone();
        for i in (k + 1)..n {
            if a[i][k].is_zero() {
                continue;
            }
            let m = normalize::div(a[i][k].clone(), pivot_val.clone());
            a[i][k] = m.clone();
            for j in (k + 1)..n {
                let prod = normalize::mul(m.clone(), a[k][j].clone());
                a[i][j] = normalize::sub(a[i][j].clone(), prod);
            }
        }
    }

    let pb: Vec<Arc<Expr>> = perm.iter().map(|&i| constants[i].clone()).collect();

    // Forward: L y = P b (L is unit lower; multipliers in a[i][j] for j < i).
    let mut y: Vec<Arc<Expr>> = vec![Expr::int(0); n];
    for i in 0..n {
        let mut sum = pb[i].clone();
        for j in 0..i {
            if a[i][j].is_zero() || y[j].is_zero() {
                continue;
            }
            let prod = normalize::mul(a[i][j].clone(), y[j].clone());
            sum = normalize::sub(sum, prod);
        }
        y[i] = sum;
    }

    // Back: U x = y.
    let mut x: Vec<Arc<Expr>> = vec![Expr::int(0); n];
    for i in (0..n).rev() {
        let mut sum = y[i].clone();
        for j in (i + 1)..n {
            if a[i][j].is_zero() || x[j].is_zero() {
                continue;
            }
            let prod = normalize::mul(a[i][j].clone(), x[j].clone());
            sum = normalize::sub(sum, prod);
        }
        x[i] = normalize::div(sum, a[i][i].clone());
    }

    let mut result = HashMap::new();
    for (i, var) in variables.iter().enumerate() {
        result.insert(var.clone(), decompile(&x[i]));
    }
    Ok(SystemSolution::Unique(result))
}
