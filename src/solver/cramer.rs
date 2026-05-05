//! Cramer's rule for 2×2 and 3×3 linear systems over canonical
//! [`Arc<Expr>`] matrices.
//!
//! Exact symbolic determinants. When the coefficient matrix is singular
//! (`det(A)` is canonically zero), the solver falls back to
//! [`super::gauss::solve_gaussian`].

use std::collections::HashMap;
use std::sync::Arc;

use crate::ast::Variable;
use crate::numeric::compile::decompile;
use crate::numeric::normalize;
use crate::numeric::Expr;

use super::gauss::solve_gaussian;
use super::system::SystemSolution;
use super::types::{SolverError, SolverResult};

/// 2×2 determinant.
pub(super) fn det_2x2(m: &[Vec<Arc<Expr>>]) -> Arc<Expr> {
    let a = normalize::mul(m[0][0].clone(), m[1][1].clone());
    let b = normalize::mul(m[0][1].clone(), m[1][0].clone());
    normalize::sub(a, b)
}

/// 3×3 determinant via cofactor expansion along the first row.
pub(super) fn det_3x3(m: &[Vec<Arc<Expr>>]) -> Arc<Expr> {
    let minor1 = normalize::sub(
        normalize::mul(m[1][1].clone(), m[2][2].clone()),
        normalize::mul(m[1][2].clone(), m[2][1].clone()),
    );
    let minor2 = normalize::sub(
        normalize::mul(m[1][0].clone(), m[2][2].clone()),
        normalize::mul(m[1][2].clone(), m[2][0].clone()),
    );
    let minor3 = normalize::sub(
        normalize::mul(m[1][0].clone(), m[2][1].clone()),
        normalize::mul(m[1][1].clone(), m[2][0].clone()),
    );

    let t1 = normalize::mul(m[0][0].clone(), minor1);
    let t2 = normalize::mul(m[0][1].clone(), minor2);
    let t3 = normalize::mul(m[0][2].clone(), minor3);

    normalize::add(normalize::sub(t1, t2), t3)
}

/// Solve a 2×2 or 3×3 linear system via Cramer's rule.
///
/// Falls back to Gaussian elimination when `det(A)` is canonically zero.
pub(super) fn solve_cramer(
    coeffs: &[Vec<Arc<Expr>>],
    constants: &[Arc<Expr>],
    variables: &[Variable],
) -> SolverResult<SystemSolution> {
    let n = variables.len();
    if coeffs.len() != n {
        return Err(SolverError::Other(
            "Cramer's rule requires square system".to_string(),
        ));
    }
    if n != 2 && n != 3 {
        return Err(SolverError::Other(
            "Cramer's rule only implemented for 2x2 and 3x3 systems".to_string(),
        ));
    }

    let det_a = if n == 2 {
        det_2x2(coeffs)
    } else {
        det_3x3(coeffs)
    };

    if det_a.is_zero() {
        return solve_gaussian(coeffs, constants, variables);
    }

    let mut result = HashMap::new();
    for i in 0..n {
        let mut modified: Vec<Vec<Arc<Expr>>> = coeffs.iter().map(|r| r.clone()).collect();
        for (row, c) in constants.iter().enumerate() {
            modified[row][i] = c.clone();
        }
        let det_i = if n == 2 {
            det_2x2(&modified)
        } else {
            det_3x3(&modified)
        };
        let val = normalize::div(det_i, det_a.clone());
        result.insert(variables[i].clone(), decompile(&val));
    }

    Ok(SystemSolution::Unique(result))
}
