//! Exact Gaussian elimination over canonical [`Arc<Expr>`] matrices.
//!
//! Operates on coefficient rows and a constant vector already compiled to the
//! canonical `Arc<Expr>` form. Arithmetic is exact: pivoting scans for the
//! first canonical non-zero entry (no floating-point tolerance). Purely
//! numeric entries reduce to `BigRational` via `normalize`; symbolic entries
//! are preserved and combined algebraically.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::ast::Variable;
use crate::numeric::compile::decompile;
use crate::numeric::normalize;
use crate::numeric::{Expr, SymbolId};

use super::system::SystemSolution;
use super::types::SolverResult;

/// Gaussian elimination with first-nonzero pivoting on canonical
/// `Arc<Expr>` entries.
pub(super) fn solve_gaussian(
    coefficients: &[Vec<Arc<Expr>>],
    constants: &[Arc<Expr>],
    variables: &[Variable],
) -> SolverResult<SystemSolution> {
    let n_eqs = coefficients.len();
    let n_vars = variables.len();

    let mut aug: Vec<Vec<Arc<Expr>>> = coefficients
        .iter()
        .zip(constants.iter())
        .map(|(row, c)| {
            let mut new_row: Vec<Arc<Expr>> = row.clone();
            new_row.push(c.clone());
            new_row
        })
        .collect();

    let mut pivot_row = 0usize;
    let mut pivot_cols: Vec<usize> = Vec::new();

    for col in 0..n_vars {
        if pivot_row >= n_eqs {
            break;
        }
        let Some(pr) = (pivot_row..n_eqs).find(|&r| !aug[r][col].is_zero()) else {
            continue;
        };
        if pr != pivot_row {
            aug.swap(pivot_row, pr);
        }
        pivot_cols.push(col);

        let pivot_val = aug[pivot_row][col].clone();
        for row in (pivot_row + 1)..n_eqs {
            if aug[row][col].is_zero() {
                continue;
            }
            let factor = normalize::div(aug[row][col].clone(), pivot_val.clone());
            aug[row][col] = Expr::int(0);
            for c in (col + 1)..=n_vars {
                let prod = normalize::mul(factor.clone(), aug[pivot_row][c].clone());
                aug[row][c] = normalize::sub(aug[row][c].clone(), prod);
            }
        }
        pivot_row += 1;
    }

    let rank = pivot_cols.len();

    for row in rank..n_eqs {
        let rhs = &aug[row][n_vars];
        let coeffs_zero = aug[row][0..n_vars].iter().all(|e| e.is_zero());
        if coeffs_zero && !rhs.is_zero() {
            return Ok(SystemSolution::NoSolution);
        }
    }

    if rank == n_vars {
        Ok(back_substitute_unique(&aug, &pivot_cols, variables))
    } else {
        Ok(build_infinite_solution(
            &aug,
            &pivot_cols,
            n_vars,
            variables,
        ))
    }
}

fn back_substitute_unique(
    aug: &[Vec<Arc<Expr>>],
    pivot_cols: &[usize],
    variables: &[Variable],
) -> SystemSolution {
    let rank = pivot_cols.len();
    let n_vars = variables.len();
    let mut sol: Vec<Arc<Expr>> = (0..n_vars).map(|_| Expr::int(0)).collect();

    for i in (0..rank).rev() {
        let col = pivot_cols[i];
        let mut sum = aug[i][n_vars].clone();
        for j in (col + 1)..n_vars {
            if sol[j].is_zero() || aug[i][j].is_zero() {
                continue;
            }
            let prod = normalize::mul(aug[i][j].clone(), sol[j].clone());
            sum = normalize::sub(sum, prod);
        }
        sol[col] = normalize::div(sum, aug[i][col].clone());
    }

    let mut result = HashMap::new();
    for (i, var) in variables.iter().enumerate() {
        result.insert(var.clone(), decompile(&sol[i]));
    }
    SystemSolution::Unique(result)
}

fn build_infinite_solution(
    aug: &[Vec<Arc<Expr>>],
    pivot_cols: &[usize],
    n_vars: usize,
    variables: &[Variable],
) -> SystemSolution {
    let rank = pivot_cols.len();
    let pivot_set: HashSet<usize> = pivot_cols.iter().cloned().collect();
    let free_cols: Vec<usize> = (0..n_vars).filter(|c| !pivot_set.contains(c)).collect();
    let free_vars: Vec<Variable> = free_cols.iter().map(|&c| variables[c].clone()).collect();

    let mut bound = HashMap::new();

    for i in (0..rank).rev() {
        let col = pivot_cols[i];
        let rhs = aug[i][n_vars].clone();
        let pivot_coeff = aug[i][col].clone();

        let mut expr = rhs;
        for &fc in &free_cols {
            let a_ij = aug[i][fc].clone();
            if a_ij.is_zero() {
                continue;
            }
            let sym = Arc::new(Expr::Symbol(SymbolId::intern(&variables[fc].name)));
            let term = normalize::mul(normalize::neg(a_ij), sym);
            expr = normalize::add(expr, term);
        }

        let final_expr = if pivot_coeff.is_one() {
            expr
        } else {
            normalize::div(expr, pivot_coeff)
        };
        bound.insert(variables[col].clone(), decompile(&final_expr));
    }

    SystemSolution::Infinite {
        bound,
        free: free_vars,
    }
}
