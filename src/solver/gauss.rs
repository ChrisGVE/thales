//! Gaussian elimination helpers for linear system solving.

use std::collections::HashMap;

use crate::ast::{BinaryOp, Expression, UnaryOp, Variable};

use super::system::SystemSolution;
use super::types::SolverResult;

// ── determinant helpers ───────────────────────────────────────────────────────

pub(super) fn det_2x2(m: &[Vec<f64>]) -> f64 {
    m[0][0] * m[1][1] - m[0][1] * m[1][0]
}

pub(super) fn det_3x3(m: &[Vec<f64>]) -> f64 {
    let minor1 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let minor2 = m[1][0] * m[2][2] - m[1][2] * m[2][0];
    let minor3 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    m[0][0] * minor1 - m[0][1] * minor2 + m[0][2] * minor3
}

// ── expression helpers ────────────────────────────────────────────────────────

pub(super) fn f64_to_expr(val: f64) -> Expression {
    if (val - val.round()).abs() < 1e-10 {
        Expression::Integer(val.round() as i64)
    } else {
        Expression::Float(val)
    }
}

pub(super) fn build_coeff_term(coeff: f64, free_var: Expression) -> Expression {
    if (coeff - coeff.round()).abs() < 1e-10 {
        let int_coeff = coeff.round() as i64;
        match int_coeff {
            1 => free_var,
            -1 => Expression::Unary(UnaryOp::Neg, Box::new(free_var)),
            _ => Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Integer(int_coeff)),
                Box::new(free_var),
            ),
        }
    } else {
        Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Float(coeff)),
            Box::new(free_var),
        )
    }
}

pub(super) fn combine_terms(mut terms: Vec<Expression>) -> Expression {
    if terms.is_empty() {
        return Expression::Integer(0);
    }
    if terms.len() == 1 {
        return terms.remove(0);
    }
    let mut result = terms.remove(0);
    for term in terms {
        result = Expression::Binary(BinaryOp::Add, Box::new(result), Box::new(term));
    }
    result
}

// ── Gaussian elimination ──────────────────────────────────────────────────────

/// Gaussian elimination with partial pivoting.
///
/// Returns a `SystemSolution` from coefficient rows, constant vector and
/// variable list.
pub(super) fn solve_gaussian(
    coefficients: &[Vec<f64>],
    constants: &[f64],
    variables: &[Variable],
) -> SolverResult<SystemSolution> {
    let n_eqs = coefficients.len();
    let n_vars = variables.len();

    let mut augmented: Vec<Vec<f64>> = coefficients
        .iter()
        .zip(constants.iter())
        .map(|(row, &c)| {
            let mut new_row = row.clone();
            new_row.push(c);
            new_row
        })
        .collect();

    let mut pivot_row = 0;
    let mut pivot_cols: Vec<usize> = Vec::new();

    for col in 0..n_vars {
        if pivot_row >= n_eqs {
            break;
        }
        let (max_row, max_val) = find_pivot(&augmented, pivot_row, n_eqs, col);
        if max_val < 1e-15 {
            continue;
        }
        if max_row != pivot_row {
            augmented.swap(pivot_row, max_row);
        }
        pivot_cols.push(col);
        eliminate_below(&mut augmented, pivot_row, col, n_eqs, n_vars);
        pivot_row += 1;
    }

    let rank = pivot_cols.len();

    for row in rank..n_eqs {
        let rhs = augmented[row][n_vars];
        if augmented[row][0..n_vars].iter().all(|&x| x.abs() < 1e-15) && rhs.abs() > 1e-15 {
            return Ok(SystemSolution::NoSolution);
        }
    }

    if rank == n_vars {
        Ok(back_substitute_unique(&augmented, &pivot_cols, variables))
    } else {
        Ok(build_infinite_solution(
            &augmented,
            &pivot_cols,
            n_vars,
            variables,
        ))
    }
}

fn find_pivot(augmented: &[Vec<f64>], start_row: usize, n_eqs: usize, col: usize) -> (usize, f64) {
    let mut max_row = start_row;
    let mut max_val = augmented[start_row][col].abs();
    for row in (start_row + 1)..n_eqs {
        if augmented[row][col].abs() > max_val {
            max_val = augmented[row][col].abs();
            max_row = row;
        }
    }
    (max_row, max_val)
}

fn eliminate_below(
    augmented: &mut Vec<Vec<f64>>,
    pivot_row: usize,
    col: usize,
    n_eqs: usize,
    n_vars: usize,
) {
    let pivot_val = augmented[pivot_row][col];
    for row in (pivot_row + 1)..n_eqs {
        let factor = augmented[row][col] / pivot_val;
        augmented[row][col] = 0.0;
        for c in (col + 1)..=n_vars {
            augmented[row][c] -= factor * augmented[pivot_row][c];
        }
    }
}

fn back_substitute_unique(
    augmented: &[Vec<f64>],
    pivot_cols: &[usize],
    variables: &[Variable],
) -> SystemSolution {
    let rank = pivot_cols.len();
    let n_vars = variables.len();
    let mut solution_values = vec![0.0_f64; n_vars];

    for i in (0..rank).rev() {
        let col = pivot_cols[i];
        let mut sum = augmented[i][n_vars];
        for j in (col + 1)..n_vars {
            sum -= augmented[i][j] * solution_values[j];
        }
        solution_values[col] = sum / augmented[i][col];
    }

    let mut result = HashMap::new();
    for (i, var) in variables.iter().enumerate() {
        result.insert(var.clone(), f64_to_expr(solution_values[i]));
    }
    SystemSolution::Unique(result)
}

fn build_infinite_solution(
    augmented: &[Vec<f64>],
    pivot_cols: &[usize],
    n_vars: usize,
    variables: &[Variable],
) -> SystemSolution {
    let rank = pivot_cols.len();
    let pivot_set: std::collections::HashSet<_> = pivot_cols.iter().cloned().collect();
    let free_cols: Vec<_> = (0..n_vars).filter(|c| !pivot_set.contains(c)).collect();
    let free_vars: Vec<_> = free_cols.iter().map(|&c| variables[c].clone()).collect();

    let mut bound = HashMap::new();

    for i in (0..rank).rev() {
        let col = pivot_cols[i];
        let rhs = augmented[i][n_vars];
        let pivot_coeff = augmented[i][col];

        let mut terms: Vec<Expression> = Vec::new();
        if rhs.abs() > 1e-15 {
            terms.push(f64_to_expr(rhs));
        }

        for &free_col in &free_cols {
            let coeff = -augmented[i][free_col] / pivot_coeff;
            if coeff.abs() > 1e-15 {
                let free_var = Expression::Variable(variables[free_col].clone());
                terms.push(build_coeff_term(coeff, free_var));
            }
        }

        let expr = combine_terms(terms);
        let final_expr = if (pivot_coeff - 1.0).abs() < 1e-15 {
            expr
        } else {
            Expression::Binary(
                BinaryOp::Div,
                Box::new(expr),
                Box::new(f64_to_expr(pivot_coeff)),
            )
        };
        bound.insert(variables[col].clone(), final_expr);
    }

    SystemSolution::Infinite {
        bound,
        free: free_vars,
    }
}
