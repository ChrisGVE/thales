//! Bordered Hessian classification for constrained optimization.
//!
//! Implements second-order sufficient conditions via the bordered Hessian
//! matrix for problems with a single equality constraint.

use crate::ast::{BinaryOp, Expression, Variable};

use super::lagrangian::OptimizationType;

/// Step size for bordered Hessian finite differences.
const BH_H: f64 = 1e-5;

/// Evaluate a single expression given parallel `names`/`values` slices.
pub(super) fn eval_at(expr: &Expression, names: &[String], values: &[f64]) -> Option<f64> {
    let vars: std::collections::HashMap<String, f64> =
        names.iter().cloned().zip(values.iter().copied()).collect();
    expr.evaluate(&vars)
}

/// Build the Lagrangian expression L = f + Σ λⱼ·gⱼ (uses symbolic lambda vars).
pub(super) fn build_lagrangian(objective: &Expression, constraints: &[Expression]) -> Expression {
    let mut lagrangian = objective.clone();
    for (j, g) in constraints.iter().enumerate() {
        let lambda = Expression::Variable(Variable::new(&format!("__lambda_{j}")));
        let term = Expression::Binary(BinaryOp::Mul, Box::new(lambda), Box::new(g.clone()));
        lagrangian = Expression::Binary(BinaryOp::Add, Box::new(lagrangian), Box::new(term));
    }
    lagrangian
}

/// Compute the n×n Hessian of `expr` w.r.t. `variables` at `point` via central differences.
pub(super) fn lagrangian_hessian(
    expr: &Expression,
    variables: &[Variable],
    names: &[String],
    point: &[f64],
) -> Option<Vec<Vec<f64>>> {
    let n = variables.len();
    let mut h = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            h[i][j] = mixed_partial(expr, names, point, i, j)?;
        }
    }
    Some(h)
}

/// Compute ∂²f/∂xᵢ∂xⱼ via four-point central difference.
fn mixed_partial(
    expr: &Expression,
    names: &[String],
    point: &[f64],
    i: usize,
    j: usize,
) -> Option<f64> {
    let mut pp = point.to_vec();
    let mut pm = point.to_vec();
    let mut mp = point.to_vec();
    let mut mm = point.to_vec();
    pp[i] += BH_H;
    pp[j] += BH_H;
    pm[i] += BH_H;
    pm[j] -= BH_H;
    mp[i] -= BH_H;
    mp[j] += BH_H;
    mm[i] -= BH_H;
    mm[j] -= BH_H;
    let fpp = eval_at(expr, names, &pp)?;
    let fpm = eval_at(expr, names, &pm)?;
    let fmp = eval_at(expr, names, &mp)?;
    let fmm = eval_at(expr, names, &mm)?;
    Some((fpp - fpm - fmp + fmm) / (4.0 * BH_H * BH_H))
}

/// Compute the gradient of `constraint` w.r.t. `variables` at `point`.
pub(super) fn constraint_gradient(
    constraint: &Expression,
    variables: &[Variable],
    names: &[String],
    point: &[f64],
) -> Option<Vec<f64>> {
    let n = variables.len();
    let mut grad = vec![0.0_f64; n];
    for i in 0..n {
        let mut p_plus = point.to_vec();
        let mut p_minus = point.to_vec();
        p_plus[i] += BH_H;
        p_minus[i] -= BH_H;
        let fp = eval_at(constraint, names, &p_plus)?;
        let fm = eval_at(constraint, names, &p_minus)?;
        grad[i] = (fp - fm) / (2.0 * BH_H);
    }
    Some(grad)
}

/// Compute the determinant of an n×n matrix via Gaussian elimination with partial pivoting.
pub(super) fn determinant(mut mat: Vec<Vec<f64>>) -> f64 {
    let n = mat.len();
    let mut sign = 1.0_f64;
    for col in 0..n {
        let pivot =
            (col..n).max_by(|&a, &b| mat[a][col].abs().partial_cmp(&mat[b][col].abs()).unwrap());
        let Some(pivot) = pivot else { return 0.0 };
        if pivot != col {
            mat.swap(col, pivot);
            sign = -sign;
        }
        let diag = mat[col][col];
        if diag.abs() < 1e-15 {
            return 0.0;
        }
        for row in (col + 1)..n {
            let factor = mat[row][col] / diag;
            for k in col..n {
                let sub = factor * mat[col][k];
                mat[row][k] -= sub;
            }
        }
    }
    let diag_product: f64 = (0..n).map(|i| mat[i][i]).product();
    sign * diag_product
}

/// Classify using the bordered Hessian for exactly one constraint.
///
/// The bordered Hessian is the (n+1)×(n+1) matrix:
///   B = [ 0    | ∇gᵀ ]
///       [ ∇g   |  H_L ]
///
/// Sign convention (Chiang, "Fundamental Methods of Mathematical Economics"):
/// for m=1 constraint and n=2 variables (3×3 bordered Hessian):
///   det(B) < 0  →  local minimum
///   det(B) > 0  →  local maximum
pub(super) fn classify_1c(h: &[Vec<f64>], grad_g: &[f64], n: usize) -> OptimizationType {
    let size = n + 1;
    let mut b = vec![vec![0.0_f64; size]; size];
    for i in 0..n {
        b[0][i + 1] = grad_g[i];
        b[i + 1][0] = grad_g[i];
    }
    for i in 0..n {
        for j in 0..n {
            b[i + 1][j + 1] = h[i][j];
        }
    }
    let det = determinant(b);
    if det.abs() < 1e-8 {
        OptimizationType::Inconclusive
    } else if det < 0.0 {
        OptimizationType::LocalMinimum
    } else {
        OptimizationType::LocalMaximum
    }
}
