//! Cubic equation solver using Cardano's formula and the trigonometric method.

use std::sync::Arc;

use crate::numeric::compile::compile;
use crate::numeric::trace::{Step, TechniqueTag, Trace};
use crate::numeric::Expr;

use crate::solver::helpers::simplify_numeric_expression;
use crate::solver::types::{Solution, SolverError, SolverResult};

use super::symbolic_complex;

/// Depress the monic cubic `x³ + px² + qx + r = 0` via `x = t − p/3`.
/// Returns `(dep_p, dep_q, shift)` for the depressed form `t³ + dep_p·t + dep_q = 0`.
pub(super) fn cubic_depress(p: f64, q: f64, r: f64) -> (f64, f64, f64) {
    let dep_p = q - p * p / 3.0;
    let dep_q = r - p * q / 3.0 + 2.0 * p * p * p / 27.0;
    let shift = -p / 3.0;
    (dep_p, dep_q, shift)
}

/// Compute the three roots of the depressed cubic `t³ + dep_p·t + dep_q = 0`
/// and apply `shift` to recover `x = t + shift`.
pub(super) fn cubic_roots_from_depressed(dep_p: f64, dep_q: f64, shift: f64) -> Vec<Arc<Expr>> {
    let discriminant = -4.0 * dep_p * dep_p * dep_p - 27.0 * dep_q * dep_q;

    if discriminant.abs() < 1e-10 {
        // All real — at least two equal.
        if dep_p.abs() < 1e-10 && dep_q.abs() < 1e-10 {
            let root = compile(&simplify_numeric_expression(shift));
            vec![root.clone(), root.clone(), root]
        } else {
            let t1 = 3.0 * dep_q / dep_p;
            let t2 = -3.0 * dep_q / (2.0 * dep_p);
            vec![
                compile(&simplify_numeric_expression(t1 + shift)),
                compile(&simplify_numeric_expression(t2 + shift)),
                compile(&simplify_numeric_expression(t2 + shift)),
            ]
        }
    } else if discriminant > 0.0 {
        // Three distinct real roots — trigonometric method.
        let m = 2.0 * (-dep_p / 3.0).sqrt();
        let acos_arg = (3.0 * dep_q / (dep_p * m)).clamp(-1.0, 1.0);
        let theta = acos_arg.acos() / 3.0;
        vec![
            compile(&simplify_numeric_expression(m * theta.cos() + shift)),
            compile(&simplify_numeric_expression(
                m * (theta - 2.0 * std::f64::consts::PI / 3.0).cos() + shift,
            )),
            compile(&simplify_numeric_expression(
                m * (theta - 4.0 * std::f64::consts::PI / 3.0).cos() + shift,
            )),
        ]
    } else {
        // One real + two complex conjugates — Cardano's formula.
        let sqrt_term = (dep_q * dep_q / 4.0 + dep_p * dep_p * dep_p / 27.0).sqrt();
        let u = (-dep_q / 2.0 + sqrt_term).cbrt();
        let v = (-dep_q / 2.0 - sqrt_term).cbrt();
        let t_real = u + v;
        let real_part = -0.5 * (u + v) + shift;
        let imag_part = (3.0_f64).sqrt() / 2.0 * (u - v);
        vec![
            compile(&simplify_numeric_expression(t_real + shift)),
            symbolic_complex(real_part, imag_part),
            symbolic_complex(real_part, -imag_part),
        ]
    }
}

/// Solve cubic equation ax³ + bx² + cx + d = 0 using Cardano's formula.
/// `coeffs = [d, c, b, a]` (constant term first).
pub(super) fn solve_cubic(coeffs: &[f64], _var: &str, trace: &mut Trace) -> SolverResult<Solution> {
    if coeffs.len() < 4 {
        return Err(SolverError::CannotSolve(
            "Not a cubic polynomial".to_string(),
        ));
    }

    let (d, c, b, a) = (coeffs[0], coeffs[1], coeffs[2], coeffs[3]);
    if a.abs() < 1e-15 {
        return Err(SolverError::CannotSolve(
            "Leading coefficient is zero".to_string(),
        ));
    }

    let (p, q, r) = (b / a, c / a, d / a);
    trace.push(Step::new(
        TechniqueTag::Simplification,
        format!("Normalized cubic: x³ + {}x² + {}x + {} = 0", p, q, r),
    ));

    let (dep_p, dep_q, shift) = cubic_depress(p, q, r);
    trace.push(Step::new(
        TechniqueTag::Simplification,
        format!(
            "Tschirnhaus Transformation: Depressed cubic: t³ + {}t + {} = 0",
            dep_p, dep_q
        ),
    ));

    let discriminant = -4.0 * dep_p * dep_p * dep_p - 27.0 * dep_q * dep_q;
    trace.push(Step::new(
        TechniqueTag::Simplification,
        format!("Discriminant: Δ = {}", discriminant),
    ));

    let roots = cubic_roots_from_depressed(dep_p, dep_q, shift);
    trace.push(
        Step::new(
            TechniqueTag::Simplification,
            "Cardano's Formula: Applied Cardano's formula".to_string(),
        )
        .with_output(roots[0].clone()),
    );

    Ok(Solution::multiple_from_expr(&roots))
}
