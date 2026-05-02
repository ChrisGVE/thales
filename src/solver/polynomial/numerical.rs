//! Numerical polynomial root finder using the Durand-Kerner method.

use std::sync::Arc;

use crate::numeric::compile::compile;
use crate::numeric::trace::{Step, TechniqueTag, Trace};
use crate::numeric::Expr;

use crate::solver::helpers::simplify_numeric_expression;
use crate::solver::types::{Solution, SolverError, SolverResult};

use super::symbolic_complex;

/// Run Durand-Kerner iterations on `roots` (modified in place) for the
/// polynomial given by `coeffs` (constant term first). Returns `true` if
/// converged within `max_iter` steps at `tolerance`.
pub(super) fn durand_kerner_iterate(
    coeffs: &[f64],
    roots: &mut Vec<num_complex::Complex64>,
    max_iter: usize,
    tolerance: f64,
) -> bool {
    let degree = roots.len();
    for _ in 0..max_iter {
        let mut max_change: f64 = 0.0;
        for i in 0..degree {
            let mut p_val = num_complex::Complex64::new(0.0, 0.0);
            let mut power = num_complex::Complex64::new(1.0, 0.0);
            for &coeff in coeffs.iter() {
                p_val += num_complex::Complex64::new(coeff, 0.0) * power;
                power *= roots[i];
            }
            let mut denom = num_complex::Complex64::new(1.0, 0.0);
            for j in 0..degree {
                if i != j {
                    denom *= roots[i] - roots[j];
                }
            }
            if denom.norm() > 1e-15 {
                let delta = p_val / denom;
                roots[i] -= delta;
                max_change = max_change.max(delta.norm());
            }
        }
        if max_change < tolerance {
            return true;
        }
    }
    false
}

/// Solve polynomial of degree 5+ using numerical methods (Durand-Kerner).
pub(super) fn solve_polynomial_numerically(
    coeffs: &[f64],
    _var: &str,
    trace: &mut Trace,
) -> SolverResult<Solution> {
    let degree = coeffs.len() - 1;
    if degree < 1 {
        return Err(SolverError::CannotSolve("Invalid polynomial".to_string()));
    }

    let leading = coeffs[degree];
    if leading.abs() < 1e-15 {
        return Err(SolverError::CannotSolve(
            "Leading coefficient is zero".to_string(),
        ));
    }

    trace.push(Step::new(
        TechniqueTag::NumericalApproximation,
        format!(
            "Solving degree {} polynomial numerically (Durand-Kerner method)",
            degree
        ),
    ));

    let radius = 1.0
        + coeffs
            .iter()
            .take(degree)
            .map(|c| (c / leading).abs())
            .fold(0.0, f64::max);
    let mut roots: Vec<num_complex::Complex64> = (0..degree)
        .map(|k| {
            let angle = 2.0 * std::f64::consts::PI * (k as f64) / (degree as f64) + 0.4;
            num_complex::Complex64::new(radius * angle.cos(), radius * angle.sin())
        })
        .collect();

    let max_iter = 100;
    let tolerance = 1e-12;
    let converged = durand_kerner_iterate(coeffs, &mut roots, max_iter, tolerance);

    if !converged {
        trace.push(Step::new(
            TechniqueTag::NumericalApproximation,
            format!(
                "Warning: Durand-Kerner did not converge after {} iterations (tolerance {})",
                max_iter, tolerance
            ),
        ));
    }

    let root_exprs: Vec<Arc<Expr>> = roots
        .iter()
        .map(|r| {
            if r.im.abs() < 1e-10 {
                compile(&simplify_numeric_expression(r.re))
            } else {
                symbolic_complex(r.re, r.im)
            }
        })
        .collect();

    trace.push(
        Step::new(
            TechniqueTag::NumericalApproximation,
            format!("Found {} roots numerically", degree),
        )
        .with_output(root_exprs[0].clone()),
    );

    Ok(Solution::multiple_from_expr(&root_exprs))
}
