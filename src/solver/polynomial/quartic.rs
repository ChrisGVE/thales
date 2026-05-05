//! Quartic equation solver using Ferrari's method.

use std::sync::Arc;

use crate::numeric::compile::compile;
use crate::numeric::trace::{Step, TechniqueTag, Trace};
use crate::numeric::Expr;

use crate::solver::helpers::simplify_numeric_expression;
use crate::solver::types::{Solution, SolverError, SolverResult};

use super::numerical::solve_polynomial_numerically;
use super::symbolic_complex;

/// Solve the biquadratic case of a depressed quartic: `y⁴ + α·y² + γ = 0`.
///
/// Substitutes `u = y²` and extracts four roots, applying `shift` to recover `x`.
pub(super) fn quartic_biquadratic_roots(alpha: f64, gamma: f64, shift: f64) -> Vec<Arc<Expr>> {
    let disc = alpha * alpha - 4.0 * gamma;

    if disc < -1e-15 {
        // Complex discriminant — all four roots are complex.
        let u1_real = -alpha / 2.0;
        let u1_imag = (-disc).sqrt() / 2.0;
        let mut roots: Vec<Arc<Expr>> = Vec::new();
        for sign1 in [-1.0_f64, 1.0] {
            let u_real = u1_real;
            let u_imag = sign1 * u1_imag;
            let r = (u_real * u_real + u_imag * u_imag).sqrt();
            let sqrt_real = ((r + u_real) / 2.0).sqrt();
            let sqrt_imag = u_imag.signum() * ((r - u_real) / 2.0).sqrt();
            roots.push(symbolic_complex(sqrt_real + shift, sqrt_imag));
            roots.push(symbolic_complex(-sqrt_real + shift, -sqrt_imag));
        }
        roots
    } else {
        let u1 = (-alpha + disc.sqrt()) / 2.0;
        let u2 = (-alpha - disc.sqrt()) / 2.0;
        let mut roots: Vec<Arc<Expr>> = Vec::new();
        for u in [u1, u2] {
            if u >= 0.0 {
                roots.push(compile(&simplify_numeric_expression(u.sqrt() + shift)));
                roots.push(compile(&simplify_numeric_expression(-u.sqrt() + shift)));
            } else {
                let imag = (-u).sqrt();
                roots.push(symbolic_complex(shift, imag));
                roots.push(symbolic_complex(shift, -imag));
            }
        }
        roots
    }
}

/// Find one real root of the resolvent cubic for Ferrari's method.
///
/// The resolvent cubic is `m³ + (α/2)m² + ((α²−4γ)/16)m − β²/64 = 0`,
/// given as `resolvent_coeffs = [−β²/64, (α²−4γ)/16, α/2, 1]`.
pub(super) fn quartic_resolvent_cubic_root(resolvent_coeffs: &[f64]) -> f64 {
    let dep_p = resolvent_coeffs[1] - resolvent_coeffs[2] * resolvent_coeffs[2] / 3.0;
    let dep_q = resolvent_coeffs[0] - resolvent_coeffs[2] * resolvent_coeffs[1] / 3.0
        + 2.0 * resolvent_coeffs[2] * resolvent_coeffs[2] * resolvent_coeffs[2] / 27.0;
    let disc_cubic = -4.0 * dep_p * dep_p * dep_p - 27.0 * dep_q * dep_q;

    if disc_cubic > 1e-10 {
        let sqrt_term = 2.0 * (-dep_p / 3.0).sqrt();
        let acos_arg = (3.0 * dep_q / (dep_p * sqrt_term)).clamp(-1.0, 1.0);
        let theta = acos_arg.acos() / 3.0;
        sqrt_term * theta.cos() - resolvent_coeffs[2] / 3.0
    } else {
        let sqrt_term = (dep_q * dep_q / 4.0 + dep_p * dep_p * dep_p / 27.0)
            .abs()
            .sqrt();
        let sign = if dep_q < 0.0 { 1.0 } else { -1.0 };
        let u = (sign * sqrt_term - dep_q / 2.0).abs().cbrt()
            * (sign * sqrt_term - dep_q / 2.0).signum();
        let v = if u.abs() > 1e-10 {
            -dep_p / (3.0 * u)
        } else {
            0.0
        };
        u + v - resolvent_coeffs[2] / 3.0
    }
}

/// Factor the depressed quartic via Ferrari's method using resolvent root `m`.
///
/// Splits into two quadratics and returns up to four roots with `shift` applied.
/// Returns `None` if `2m + α < 0` (caller falls back to numerical).
pub(super) fn quartic_ferrari_roots(
    alpha: f64,
    beta: f64,
    m: f64,
    shift: f64,
) -> Option<Vec<Arc<Expr>>> {
    let val_2m_alpha = 2.0 * m + alpha;
    if val_2m_alpha < -1e-12 {
        return None;
    }
    let sqrt_2m_alpha = val_2m_alpha.max(0.0).sqrt();
    let term = if sqrt_2m_alpha.abs() > 1e-10 {
        beta / (2.0 * sqrt_2m_alpha)
    } else {
        0.0
    };

    let mut roots: Vec<Arc<Expr>> = Vec::new();

    // First quadratic: y² + sqrt(2m+α)·y + (m + term) = 0
    let (b1, c1) = (sqrt_2m_alpha, m + term);
    let disc1 = b1 * b1 - 4.0 * c1;
    if disc1 >= 0.0 {
        roots.push(compile(&simplify_numeric_expression(
            (-b1 + disc1.sqrt()) / 2.0 + shift,
        )));
        roots.push(compile(&simplify_numeric_expression(
            (-b1 - disc1.sqrt()) / 2.0 + shift,
        )));
    } else {
        let (real, imag) = (-b1 / 2.0 + shift, (-disc1).sqrt() / 2.0);
        roots.push(symbolic_complex(real, imag));
        roots.push(symbolic_complex(real, -imag));
    }

    // Second quadratic: y² − sqrt(2m+α)·y + (m − term) = 0
    let (b2, c2) = (-sqrt_2m_alpha, m - term);
    let disc2 = b2 * b2 - 4.0 * c2;
    if disc2 >= 0.0 {
        roots.push(compile(&simplify_numeric_expression(
            (-b2 + disc2.sqrt()) / 2.0 + shift,
        )));
        roots.push(compile(&simplify_numeric_expression(
            (-b2 - disc2.sqrt()) / 2.0 + shift,
        )));
    } else {
        let (real, imag) = (-b2 / 2.0 + shift, (-disc2).sqrt() / 2.0);
        roots.push(symbolic_complex(real, imag));
        roots.push(symbolic_complex(real, -imag));
    }

    Some(roots)
}

/// Solve quartic equation ax⁴ + bx³ + cx² + dx + e = 0 using Ferrari's method.
/// `coeffs = [e, d, c, b, a]` (constant term first).
pub(super) fn solve_quartic(
    coeffs: &[f64],
    var: &str,
    trace: &mut Trace,
) -> SolverResult<Solution> {
    if coeffs.len() < 5 {
        return Err(SolverError::CannotSolve(
            "Not a quartic polynomial".to_string(),
        ));
    }

    let (e, d, c, b, a) = (coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]);
    if a.abs() < 1e-15 {
        return Err(SolverError::CannotSolve(
            "Leading coefficient is zero".to_string(),
        ));
    }

    let (p, q, r, s) = (b / a, c / a, d / a, e / a);
    trace.push(Step::new(
        TechniqueTag::Simplification,
        format!(
            "Normalized quartic: x⁴ + {}x³ + {}x² + {}x + {} = 0",
            p, q, r, s
        ),
    ));

    let alpha = q - 3.0 * p * p / 8.0;
    let beta = r - p * q / 2.0 + p * p * p / 8.0;
    let gamma = s - p * r / 4.0 + p * p * q / 16.0 - 3.0 * p * p * p * p / 256.0;
    let shift = -p / 4.0;
    trace.push(Step::new(
        TechniqueTag::Simplification,
        format!(
            "Tschirnhaus Transformation: Depressed quartic: y⁴ + {}y² + {}y + {} = 0",
            alpha, beta, gamma
        ),
    ));

    // Special case: β = 0 (biquadratic).
    if beta.abs() < 1e-15 {
        let roots = quartic_biquadratic_roots(alpha, gamma, shift);
        let label = if (alpha * alpha - 4.0 * gamma) < -1e-15 {
            "Solved biquadratic via complex square roots".to_string()
        } else {
            let u1 = (-alpha + (alpha * alpha - 4.0 * gamma).sqrt()) / 2.0;
            let u2 = (-alpha - (alpha * alpha - 4.0 * gamma).sqrt()) / 2.0;
            format!("Solved biquadratic: u = y², u₁ = {}, u₂ = {}", u1, u2)
        };
        trace.push(Step::new(TechniqueTag::Simplification, label).with_output(roots[0].clone()));
        return Ok(Solution::multiple_from_expr(&roots));
    }

    // General case: solve resolvent cubic for one real root m.
    let resolvent_coeffs = vec![
        -beta * beta / 64.0,
        (alpha * alpha - 4.0 * gamma) / 16.0,
        alpha / 2.0,
        1.0,
    ];
    let m = quartic_resolvent_cubic_root(&resolvent_coeffs);
    trace.push(Step::new(
        TechniqueTag::Simplification,
        format!("Resolvent cubic root: m = {}", m),
    ));

    // Ferrari factorization.
    let roots = match quartic_ferrari_roots(alpha, beta, m, shift) {
        Some(r) => r,
        None => return solve_polynomial_numerically(coeffs, var, trace),
    };

    trace.push(
        Step::new(
            TechniqueTag::Simplification,
            "Ferrari's Method: Applied Ferrari's method".to_string(),
        )
        .with_output(roots[0].clone()),
    );

    Ok(Solution::multiple_from_expr(&roots))
}
