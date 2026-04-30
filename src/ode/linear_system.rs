//! Eigenvalue-based solver for 2×2 linear constant-coefficient ODE systems.
//!
//! This module is private to the `ode` subsystem. The public entry point
//! [`solve_linear_system`] is re-exported from `ode::system`.

use std::sync::Arc;

use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::normalize;

use super::system::{OdeSystem, OdeSystemSolution};
use super::types::ODEError;

// ── Public entry ─────────────────────────────────────────────────────────────

/// Solve a linear constant-coefficient ODE system symbolically (2×2 only).
///
/// Uses eigenvalue decomposition of the coefficient matrix A extracted from
/// y' = Ay. Returns [`ODEError::NotLinearConstantCoefficient`] for n > 2.
pub fn solve_linear_system(system: &OdeSystem) -> Result<OdeSystemSolution, ODEError> {
    if system.n != 2 {
        return Err(ODEError::NotLinearConstantCoefficient(
            "systems larger than 2×2 not yet supported".into(),
        ));
    }
    let mut steps: Vec<String> = Vec::new();
    let a = super::system::extract_linear_system_matrix(system)?;
    steps.push(format!(
        "Extracted coefficient matrix A = [[{}, {}], [{}, {}]]",
        a[0][0], a[0][1], a[1][0], a[1][1]
    ));
    let (eigenvalues, kind) = compute_eigenvalues_2x2(&a, &mut steps)?;
    let t = &system.var;
    let components = build_system_solution_2x2(&a, &eigenvalues, kind, t, &mut steps)?;
    Ok(OdeSystemSolution {
        components,
        method: "eigenvalue decomposition (2×2)".into(),
        steps,
    })
}

// ── Eigenvalue kind ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum EigenKind {
    TwoDistinctReal,
    Repeated,
    Complex,
}

// ── Eigenvalue computation ────────────────────────────────────────────────────

/// Compute eigenvalues of a 2×2 matrix, return (λ₁, λ₂) and kind.
///
/// For complex roots, λ₁ = α (real part), λ₂ = β (imaginary part).
fn compute_eigenvalues_2x2(
    a: &[Vec<f64>],
    steps: &mut Vec<String>,
) -> Result<((f64, f64), EigenKind), ODEError> {
    let tr = a[0][0] + a[1][1];
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let disc = tr * tr - 4.0 * det;
    const EPS: f64 = 1e-10;
    if disc > EPS {
        let sq = disc.sqrt();
        let l1 = (tr + sq) / 2.0;
        let l2 = (tr - sq) / 2.0;
        steps.push(format!("Computed eigenvalues λ₁ = {l1:.6}, λ₂ = {l2:.6}"));
        Ok(((l1, l2), EigenKind::TwoDistinctReal))
    } else if disc < -EPS {
        let alpha = tr / 2.0;
        let beta = (-disc).sqrt() / 2.0;
        steps.push(format!(
            "Computed complex eigenvalues λ = {alpha:.6} ± {beta:.6}i"
        ));
        Ok(((alpha, beta), EigenKind::Complex))
    } else {
        let l = tr / 2.0;
        steps.push(format!("Computed repeated eigenvalue λ = {l:.6}"));
        Ok(((l, l), EigenKind::Repeated))
    }
}

// ── Solution dispatch ─────────────────────────────────────────────────────────

/// Dispatch to the correct 2×2 solution builder based on eigenvalue kind.
fn build_system_solution_2x2(
    a: &[Vec<f64>],
    eigenvalues: &(f64, f64),
    kind: EigenKind,
    t: &str,
    steps: &mut Vec<String>,
) -> Result<Vec<Arc<Expr>>, ODEError> {
    match kind {
        EigenKind::TwoDistinctReal => {
            build_distinct_real_solution(a, eigenvalues.0, eigenvalues.1, t, steps)
        }
        EigenKind::Complex => build_complex_solution(a, eigenvalues.0, eigenvalues.1, t, steps),
        EigenKind::Repeated => build_repeated_solution(a, eigenvalues.0, t, steps),
    }
}

// ── Eigenvector helper ────────────────────────────────────────────────────────

/// Compute eigenvector for eigenvalue λ of 2×2 matrix A.
/// Returns (v0, v1); uses null-space of (A − λI).
fn eigenvector_2x2(a: &[Vec<f64>], lambda: f64) -> (f64, f64) {
    let r0 = a[0][0] - lambda;
    let r1 = a[0][1];
    if r0.abs() > 1e-10 || r1.abs() > 1e-10 {
        return (-r1, r0);
    }
    let s0 = a[1][0];
    let s1 = a[1][1] - lambda;
    (-s1, s0)
}

// ── Distinct real eigenvalues ─────────────────────────────────────────────────

fn build_distinct_real_solution(
    a: &[Vec<f64>],
    l1: f64,
    l2: f64,
    t: &str,
    steps: &mut Vec<String>,
) -> Result<Vec<Arc<Expr>>, ODEError> {
    let (v1_0, v1_1) = eigenvector_2x2(a, l1);
    let (v2_0, v2_1) = eigenvector_2x2(a, l2);
    steps.push(format!(
        "Constructed general solution: \
         y₁ = C1·{v1_0:.4}·e^({l1:.4}t) + C2·{v2_0:.4}·e^({l2:.4}t)"
    ));
    Ok(vec![
        build_two_term_exp(v1_0, l1, v2_0, l2, t),
        build_two_term_exp(v1_1, l1, v2_1, l2, t),
    ])
}

/// Build `C1·a·exp(λ1·t) + C2·b·exp(λ2·t)`.
fn build_two_term_exp(a: f64, l1: f64, b: f64, l2: f64, t: &str) -> Arc<Expr> {
    let c1 = Expr::symbol("C1");
    let c2 = Expr::symbol("C2");
    let t_sym = Expr::symbol(t);
    let e1 = Expr::func(
        FuncId::Exp,
        vec![normalize::mul(Expr::float(l1), t_sym.clone())],
    );
    let e2 = Expr::func(FuncId::Exp, vec![normalize::mul(Expr::float(l2), t_sym)]);
    let term1 = normalize::mul(c1, normalize::mul(Expr::float(a), e1));
    let term2 = normalize::mul(c2, normalize::mul(Expr::float(b), e2));
    normalize::add(term1, term2)
}

// ── Complex eigenvalues ───────────────────────────────────────────────────────

fn build_complex_solution(
    a: &[Vec<f64>],
    alpha: f64,
    beta: f64,
    t: &str,
    steps: &mut Vec<String>,
) -> Result<Vec<Arc<Expr>>, ODEError> {
    let (vr0, vr1) = eigenvector_2x2(a, alpha);
    let vi0 = if beta.abs() > 1e-14 {
        ((a[0][0] - alpha) * vr0 + a[0][1] * vr1) / beta
    } else {
        0.0
    };
    let vi1 = if beta.abs() > 1e-14 {
        (a[1][0] * vr0 + (a[1][1] - alpha) * vr1) / beta
    } else {
        0.0
    };
    steps.push(format!(
        "Constructed general solution using \
         e^({alpha:.4}t)(C1·cos({beta:.4}t) + C2·sin({beta:.4}t))"
    ));
    Ok(vec![
        build_complex_component(vr0, vi0, alpha, beta, t),
        build_complex_component(vr1, vi1, alpha, beta, t),
    ])
}

/// Build one component for complex eigenvalue case.
/// y_j = e^(αt)·[(vr·C1 − vi·C2)·cos(βt) + (vi·C1 + vr·C2)·sin(βt)]
fn build_complex_component(vr: f64, vi: f64, alpha: f64, beta: f64, t: &str) -> Arc<Expr> {
    let c1 = Expr::symbol("C1");
    let c2 = Expr::symbol("C2");
    let t_sym = Expr::symbol(t);
    let beta_t = normalize::mul(Expr::float(beta), t_sym.clone());
    let cos_t = Expr::func(FuncId::Cos, vec![beta_t.clone()]);
    let sin_t = Expr::func(FuncId::Sin, vec![beta_t]);
    let cos_coeff = normalize::add(
        normalize::mul(Expr::float(vr), c1.clone()),
        normalize::mul(Expr::float(-vi), c2.clone()),
    );
    let sin_coeff = normalize::add(
        normalize::mul(Expr::float(vi), c1),
        normalize::mul(Expr::float(vr), c2),
    );
    let oscillatory = normalize::add(
        normalize::mul(cos_coeff, cos_t),
        normalize::mul(sin_coeff, sin_t),
    );
    if alpha.abs() < 1e-14 {
        oscillatory
    } else {
        let exp_term = Expr::func(FuncId::Exp, vec![normalize::mul(Expr::float(alpha), t_sym)]);
        normalize::mul(exp_term, oscillatory)
    }
}

// ── Repeated eigenvalue ───────────────────────────────────────────────────────

fn build_repeated_solution(
    a: &[Vec<f64>],
    lambda: f64,
    t: &str,
    steps: &mut Vec<String>,
) -> Result<Vec<Arc<Expr>>, ODEError> {
    let (v0, v1) = eigenvector_2x2(a, lambda);
    let (w0, w1) = generalised_eigenvector_2x2(a, lambda, v0, v1);
    steps.push(format!(
        "Constructed general solution: y = (C1·v + C2·(v·t + w))·e^({lambda:.4}t)"
    ));
    Ok(vec![
        build_repeated_component(v0, w0, lambda, t),
        build_repeated_component(v1, w1, lambda, t),
    ])
}

/// Solve (A − λI)w = v for the generalised eigenvector w.
fn generalised_eigenvector_2x2(a: &[Vec<f64>], lambda: f64, v0: f64, v1: f64) -> (f64, f64) {
    let b00 = a[0][0] - lambda;
    let b01 = a[0][1];
    let b10 = a[1][0];
    let b11 = a[1][1] - lambda;
    if b00.abs() >= b01.abs() && b00.abs() > 1e-14 {
        let w1 = if b01.abs() > 1e-14 { v0 / b01 } else { 0.0 };
        let w0 = (v0 - b01 * w1) / b00;
        (w0, w1)
    } else if b10.abs() > 1e-14 || b11.abs() > 1e-14 {
        let w0 = if b10.abs() > 1e-14 { v1 / b10 } else { 0.0 };
        let w1 = if b11.abs() > 1e-14 {
            (v1 - b10 * w0) / b11
        } else {
            0.0
        };
        (w0, w1)
    } else {
        (0.0, 0.0)
    }
}

/// Build one component for the repeated-eigenvalue case.
/// y_j = (C1·v + C2·(w + v·t))·exp(λt)
fn build_repeated_component(v: f64, w: f64, lambda: f64, t: &str) -> Arc<Expr> {
    let c1 = Expr::symbol("C1");
    let c2 = Expr::symbol("C2");
    let t_sym = Expr::symbol(t);
    let term1 = normalize::mul(c1, Expr::float(v));
    let v_t = normalize::mul(Expr::float(v), t_sym.clone());
    let w_plus_vt = normalize::add(Expr::float(w), v_t);
    let term2 = normalize::mul(c2, w_plus_vt);
    let bracket = normalize::add(term1, term2);
    let exp_arg = normalize::mul(Expr::float(lambda), t_sym);
    let exp_term = Expr::func(FuncId::Exp, vec![exp_arg]);
    normalize::mul(bracket, exp_term)
}
