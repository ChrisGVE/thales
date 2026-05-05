//! Helper functions for higher-order ODE solving.

use crate::ast::{BinaryOp, Expression, Function, Variable};
use crate::ode::{ODEError, SecondOrderODE, SecondOrderSolution};
use std::fmt;

use super::particular::ForcingKind;
use super::types::CharRoot;

pub(super) fn find_characteristic_roots(coeffs: &[f64]) -> Result<Vec<CharRoot>, ODEError> {
    let n = coeffs.len() - 1; // polynomial degree
    let lead = coeffs[0];

    // Normalise: monic polynomial.
    let p: Vec<f64> = coeffs.iter().map(|c| c / lead).collect();

    // Durand-Kerner initial guesses: equally-spaced on a circle.
    use std::f64::consts::TAU;
    let mut roots: Vec<(f64, f64)> = (0..n)
        .map(|k| {
            let angle = TAU * k as f64 / n as f64;
            (0.4 * angle.cos(), 0.4 * angle.sin())
        })
        .collect();

    // Iterate until convergence.
    for _ in 0..200 {
        let prev = roots.clone();
        for i in 0..n {
            let (re, im) = roots[i];
            let pval = poly_eval_complex(&p, re, im);
            let mut denom = (1.0, 0.0);
            for j in 0..n {
                if j != i {
                    let diff_re = re - roots[j].0;
                    let diff_im = im - roots[j].1;
                    denom = complex_mul(denom, (diff_re, diff_im));
                }
            }
            let update = complex_div(pval, denom)?;
            roots[i] = (re - update.0, im - update.1);
        }
        // Check convergence.
        let max_delta = roots
            .iter()
            .zip(&prev)
            .map(|((r, i), (pr, pi))| ((r - pr).powi(2) + (i - pi).powi(2)).sqrt())
            .fold(0.0_f64, f64::max);
        if max_delta < 1e-12 {
            break;
        }
    }

    // Round near-zero imaginary parts.
    for r in &mut roots {
        if r.1.abs() < 1e-8 {
            r.1 = 0.0;
        }
    }

    // Cluster roots with equal real + imag parts (multiplicity).
    let char_roots = cluster_roots(roots);
    Ok(char_roots)
}

/// Evaluate a monic polynomial with real coefficients at complex point (re, im).
pub(super) fn poly_eval_complex(p: &[f64], re: f64, im: f64) -> (f64, f64) {
    let (mut re_acc, mut im_acc) = (0.0_f64, 0.0_f64);
    for &c in p {
        let new_re = re_acc * re - im_acc * im + c;
        let new_im = re_acc * im + im_acc * re;
        re_acc = new_re;
        im_acc = new_im;
    }
    (re_acc, im_acc)
}

pub(super) fn complex_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

pub(super) fn complex_div(num: (f64, f64), den: (f64, f64)) -> Result<(f64, f64), ODEError> {
    let denom = den.0 * den.0 + den.1 * den.1;
    if denom < 1e-30 {
        return Err(ODEError::CharacteristicEquationError(
            "Division by zero in Durand-Kerner iteration".to_string(),
        ));
    }
    Ok((
        (num.0 * den.0 + num.1 * den.1) / denom,
        (num.1 * den.0 - num.0 * den.1) / denom,
    ))
}

/// Group near-identical roots and assign multiplicities.
pub(super) fn cluster_roots(roots: Vec<(f64, f64)>) -> Vec<CharRoot> {
    const TOL: f64 = 1e-6;
    let mut result: Vec<CharRoot> = Vec::new();
    let mut used = vec![false; roots.len()];

    for i in 0..roots.len() {
        if used[i] {
            continue;
        }
        let (ri, ii) = roots[i];
        let mut mult = 1;
        for j in (i + 1)..roots.len() {
            if !used[j] {
                let (rj, ij) = roots[j];
                if ((ri - rj).powi(2) + (ii - ij).powi(2)).sqrt() < TOL {
                    mult += 1;
                    used[j] = true;
                }
            }
        }
        result.push(CharRoot {
            real: ri,
            imag: ii,
            multiplicity: mult,
        });
    }
    result
}

// ---------------------------------------------------------------------------
// Build general solution from roots
// ---------------------------------------------------------------------------

pub(super) fn build_general_solution(
    roots: &[CharRoot],
    x_var: &str,
    steps: &mut Vec<String>,
) -> Expression {
    let x = Expression::Variable(Variable::new(x_var));
    let mut terms: Vec<Expression> = Vec::new();
    let mut c_index = 1usize;

    for root in roots {
        if root.is_real() {
            // Real root r with multiplicity m:
            // (C1 + C2*x + … + Cm*x^(m-1)) * e^(r*x)
            let exp_term = make_exp(root.real, &x, x_var);
            let poly = make_poly_constants(root.multiplicity, &x, &mut c_index);
            let term = if matches!(&poly, Expression::Integer(1)) {
                exp_term
            } else {
                Expression::Binary(BinaryOp::Mul, Box::new(poly), Box::new(exp_term))
            };
            steps.push(format!(
                "Root r={:.4} (mult {}): contributes polynomial × e^({:.4}·{})",
                root.real, root.multiplicity, root.real, x_var
            ));
            terms.push(term);
        } else if root.imag > 0.0 {
            // Complex pair α ± βi with multiplicity m:
            // e^(αx) * (C1*cos(βx) + C2*sin(βx)) × polynomial in x
            let alpha = root.real;
            let beta = root.imag;
            let osc = make_oscillatory(alpha, beta, root.multiplicity, &x, x_var, &mut c_index);
            steps.push(format!(
                "Complex roots {:.4}±{:.4}i (mult {}): contributes oscillatory term",
                alpha, beta, root.multiplicity
            ));
            terms.push(osc);
            // Skip conjugate root (handled here with paired constants).
        }
        // Negative-imaginary roots are conjugates — already handled above.
    }

    // Sum all terms.
    terms
        .into_iter()
        .reduce(|acc, t| Expression::Binary(BinaryOp::Add, Box::new(acc), Box::new(t)))
        .unwrap_or(Expression::Integer(0))
}

/// Build `e^(r*x)` (or `1` when r == 0).
pub(super) fn make_exp(r: f64, x: &Expression, _x_var: &str) -> Expression {
    if r.abs() < 1e-10 {
        return Expression::Integer(1);
    }
    let arg = Expression::Binary(
        BinaryOp::Mul,
        Box::new(Expression::Float(r)),
        Box::new(x.clone()),
    );
    Expression::Function(Function::Exp, vec![arg])
}

/// Build `C_k + C_{k+1}*x + … + C_{k+m-1}*x^(m-1)`, incrementing c_index.
pub(super) fn make_poly_constants(
    multiplicity: usize,
    x: &Expression,
    c_index: &mut usize,
) -> Expression {
    let mut terms: Vec<Expression> = Vec::new();
    for k in 0..multiplicity {
        let c = Expression::Variable(Variable::new(&format!("C{}", *c_index)));
        *c_index += 1;
        if k == 0 {
            terms.push(c);
        } else {
            let xk =
                Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(k as i64)));
            terms.push(Expression::Binary(BinaryOp::Mul, Box::new(c), Box::new(xk)));
        }
    }
    terms
        .into_iter()
        .reduce(|acc, t| Expression::Binary(BinaryOp::Add, Box::new(acc), Box::new(t)))
        .unwrap_or(Expression::Integer(1))
}

/// Build `e^(αx) * [poly(x)·cos(βx) + poly(x)·sin(βx)]` for complex roots.
pub(super) fn make_oscillatory(
    alpha: f64,
    beta: f64,
    multiplicity: usize,
    x: &Expression,
    x_var: &str,
    c_index: &mut usize,
) -> Expression {
    let beta_x = Expression::Binary(
        BinaryOp::Mul,
        Box::new(Expression::Float(beta)),
        Box::new(x.clone()),
    );
    let cos_part = Expression::Function(Function::Cos, vec![beta_x.clone()]);
    let sin_part = Expression::Function(Function::Sin, vec![beta_x]);

    // cos group: poly_cos(x) * cos(βx)
    let poly_cos = make_poly_constants(multiplicity, x, c_index);
    let cos_term = Expression::Binary(BinaryOp::Mul, Box::new(poly_cos), Box::new(cos_part));

    // sin group: poly_sin(x) * sin(βx)
    let poly_sin = make_poly_constants(multiplicity, x, c_index);
    let sin_term = Expression::Binary(BinaryOp::Mul, Box::new(poly_sin), Box::new(sin_part));

    let oscillatory = Expression::Binary(BinaryOp::Add, Box::new(cos_term), Box::new(sin_term));

    let exp_env = make_exp(alpha, x, x_var);
    if matches!(&exp_env, Expression::Integer(1)) {
        oscillatory
    } else {
        Expression::Binary(BinaryOp::Mul, Box::new(exp_env), Box::new(oscillatory))
    }
}

// ---------------------------------------------------------------------------
// Undetermined coefficients – particular solution
// ---------------------------------------------------------------------------

pub(super) fn find_particular_solution(
    ode: &SecondOrderODE,
    kind: &ForcingKind,
    hom: &SecondOrderSolution,
    steps: &mut Vec<String>,
) -> Result<Expression, ODEError> {
    match kind {
        ForcingKind::Exponential(alpha) => particular_exponential(ode, *alpha, hom, steps),
        ForcingKind::Polynomial(degree) => particular_polynomial(ode, *degree, steps),
        ForcingKind::Sinusoidal(omega) => particular_sinusoidal(ode, *omega, hom, steps),
        ForcingKind::PolynomialTimesExp { degree, alpha } => {
            particular_poly_exp(ode, *degree, *alpha, hom, steps)
        }
    }
}

/// Particular solution for g(x) = e^(α·x).
///
/// Trial: A·e^(αx), or A·x·e^(αx) under resonance.
pub(super) fn particular_exponential(
    ode: &SecondOrderODE,
    alpha: f64,
    hom: &SecondOrderSolution,
    steps: &mut Vec<String>,
) -> Result<Expression, ODEError> {
    // Check resonance: α is a root of the characteristic equation.
    let is_resonance = hom_has_root(hom, alpha, 0.0);

    if is_resonance {
        return Err(ODEError::ResonanceDetected(format!(
            "α = {alpha} is a characteristic root; \
             multiply trial by x (not yet implemented)"
        )));
    }

    // Substitute A·e^(αx) into the ODE:
    // A·[a·α² + b·α + c]·e^(αx) = e^(αx)
    // => A = 1 / (a·α² + b·α + c)
    let denom = ode.a * alpha * alpha + ode.b * alpha + ode.c;
    if denom.abs() < 1e-12 {
        return Err(ODEError::ResonanceDetected(format!(
            "Characteristic polynomial vanishes at α = {alpha}"
        )));
    }
    let a_coeff = 1.0 / denom;
    steps.push(format!(
        "Trial y_p = A·e^({alpha}·x); A = 1/({}) = {a_coeff:.6}",
        denom
    ));

    let x = Expression::Variable(Variable::new(&ode.independent));
    let exp_arg = Expression::Binary(
        BinaryOp::Mul,
        Box::new(Expression::Float(alpha)),
        Box::new(x),
    );
    let exp_term = Expression::Function(Function::Exp, vec![exp_arg]);
    Ok(Expression::Binary(
        BinaryOp::Mul,
        Box::new(Expression::Float(a_coeff)),
        Box::new(exp_term),
    ))
}

/// Particular solution for g(x) = x^n (polynomial of degree n).
///
/// Trial: A_n·x^n + … + A_0 (all coefficients determined by substitution).
pub(super) fn particular_polynomial(
    ode: &SecondOrderODE,
    degree: usize,
    steps: &mut Vec<String>,
) -> Result<Expression, ODEError> {
    if ode.c.abs() < 1e-12 {
        return Err(ODEError::CannotSolve(
            "Undetermined coefficients for polynomial forcing requires c ≠ 0".to_string(),
        ));
    }
    // For g(x) = 1 (constant), trial A; for g(x) = x, trial Ax + B; etc.
    // Coefficient matching gives: A_k = d_k / c   (simplified for pure polynomial,
    // where we only handle the leading term here for degree ≤ 2).
    // This is a simplified implementation handling degree 0, 1, 2.
    if degree > 2 {
        return Err(ODEError::CannotSolve(
            "Polynomial undetermined coefficients implemented for degree ≤ 2".to_string(),
        ));
    }

    steps.push(format!(
        "Trial y_p = polynomial of degree {degree} with undetermined coefficients"
    ));

    let x = Expression::Variable(Variable::new(&ode.independent));

    // Build symbolic trial y_p = A*x^2 + B*x + D (or subsets)
    // and match by substituting into the ODE. We solve the resulting
    // linear system analytically.
    match degree {
        0 => {
            // g = const k, trial y_p = A; 0 + 0 + c·A = k => A = k/c
            let a_val = 1.0 / ode.c; // normalised to g(x)=1
            steps.push(format!("y_p = {a_val:.6}"));
            Ok(Expression::Float(a_val))
        }
        1 => {
            // g = x, trial y_p = Ax + B
            // y_p'' = 0, y_p' = A
            // a·0 + b·A + c·(Ax+B) = x
            // => c·A = 1 => A = 1/c
            // => b·A + c·B = 0 => B = -b·A/c = -b/c²
            let a_val = 1.0 / ode.c;
            let b_val = -ode.b * a_val / ode.c;
            steps.push(format!("y_p = {a_val:.6}·x + {b_val:.6}"));
            let ax = Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Float(a_val)),
                Box::new(x.clone()),
            );
            Ok(Expression::Binary(
                BinaryOp::Add,
                Box::new(ax),
                Box::new(Expression::Float(b_val)),
            ))
        }
        2 => {
            // g = x², trial y_p = Ax² + Bx + D
            // y_p' = 2Ax + B, y_p'' = 2A
            // a·2A + b·(2Ax+B) + c·(Ax²+Bx+D) = x²
            // x² coeff: c·A = 1 => A = 1/c
            // x¹ coeff: 2b·A + c·B = 0 => B = -2bA/c
            // x⁰ coeff: 2a·A + b·B + c·D = 0 => D = -(2aA + bB)/c
            let a_val = 1.0 / ode.c;
            let b_val = -2.0 * ode.b * a_val / ode.c;
            let d_val = -(2.0 * ode.a * a_val + ode.b * b_val) / ode.c;
            steps.push(format!("y_p = {a_val:.6}·x² + {b_val:.6}·x + {d_val:.6}"));
            let ax2 = Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Float(a_val)),
                Box::new(Expression::Power(
                    Box::new(x.clone()),
                    Box::new(Expression::Integer(2)),
                )),
            );
            let bx = Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Float(b_val)),
                Box::new(x),
            );
            Ok(Expression::Binary(
                BinaryOp::Add,
                Box::new(Expression::Binary(
                    BinaryOp::Add,
                    Box::new(ax2),
                    Box::new(bx),
                )),
                Box::new(Expression::Float(d_val)),
            ))
        }
        _ => unreachable!(),
    }
}

/// Particular solution for g(x) = cos(ωx) or sin(ωx).
///
/// Trial: A·cos(ωx) + B·sin(ωx); resonance when ω matches imaginary part.
pub(super) fn particular_sinusoidal(
    ode: &SecondOrderODE,
    omega: f64,
    hom: &SecondOrderSolution,
    steps: &mut Vec<String>,
) -> Result<Expression, ODEError> {
    use crate::ode::RootType;
    if matches!(hom.roots.root_type, RootType::ComplexConjugate)
        && (hom.roots.r2 - omega).abs() < 1e-10
    {
        return Err(ODEError::ResonanceDetected(format!(
            "ω = {omega} matches imaginary part of characteristic roots"
        )));
    }

    // Substitute A·cos + B·sin into a·y'' + b·y' + c·y = cos(ω·x).
    // y_p = A·cos(ωx) + B·sin(ωx)
    // y_p' = -Aω·sin + Bω·cos
    // y_p'' = -Aω²·cos - Bω²·sin
    //
    // Collecting cos terms: (c - aω²)A + bωB = 1
    // Collecting sin terms: -bωA + (c - aω²)B = 0
    let p = ode.c - ode.a * omega * omega;
    let q = ode.b * omega;
    let det = p * p + q * q;
    if det.abs() < 1e-12 {
        return Err(ODEError::ResonanceDetected(format!(
            "Resonance: determinant vanishes for ω = {omega}"
        )));
    }
    let a_coeff = p / det;
    let b_coeff = q / det;
    steps.push(format!(
        "Trial y_p = A·cos({omega}x) + B·sin({omega}x); A={a_coeff:.6}, B={b_coeff:.6}"
    ));

    let x = Expression::Variable(Variable::new(&ode.independent));
    let omega_x = Expression::Binary(
        BinaryOp::Mul,
        Box::new(Expression::Float(omega)),
        Box::new(x),
    );
    let cos_t = Expression::Function(Function::Cos, vec![omega_x.clone()]);
    let sin_t = Expression::Function(Function::Sin, vec![omega_x]);
    let acos = Expression::Binary(
        BinaryOp::Mul,
        Box::new(Expression::Float(a_coeff)),
        Box::new(cos_t),
    );
    let bsin = Expression::Binary(
        BinaryOp::Mul,
        Box::new(Expression::Float(b_coeff)),
        Box::new(sin_t),
    );
    Ok(Expression::Binary(
        BinaryOp::Add,
        Box::new(acos),
        Box::new(bsin),
    ))
}

/// Particular solution for g(x) = x^n · e^(αx).
pub(super) fn particular_poly_exp(
    ode: &SecondOrderODE,
    degree: usize,
    alpha: f64,
    hom: &SecondOrderSolution,
    steps: &mut Vec<String>,
) -> Result<Expression, ODEError> {
    // Check resonance.
    if hom_has_root(hom, alpha, 0.0) {
        return Err(ODEError::ResonanceDetected(format!(
            "α = {alpha} is a characteristic root; modify trial solution"
        )));
    }
    if degree > 2 {
        return Err(ODEError::CannotSolve(
            "PolynomialTimesExp implemented for degree ≤ 2".to_string(),
        ));
    }
    steps.push(format!(
        "Trial y_p = polynomial(degree {degree}) · e^({alpha}·x)"
    ));

    // Substitute into the ODE and extract coefficients analytically.
    // For y_p = (A·x^n + … + D)·e^(α·x), use operator method:
    // L[e^(αx)·p(x)] = e^(αx)·L_shifted[p(x)]  where L_shifted uses
    // coefficients evaluated at (D + α).
    // For degree 0 (just e^(αx)):
    let poly_part = particular_polynomial(
        &SecondOrderODE::new(
            &ode.dependent,
            &ode.independent,
            ode.a,
            2.0 * ode.a * alpha + ode.b,
            ode.a * alpha * alpha + ode.b * alpha + ode.c,
            crate::ast::Expression::Integer(0),
        ),
        degree,
        steps,
    )?;

    let x = Expression::Variable(Variable::new(&ode.independent));
    let exp_arg = Expression::Binary(
        BinaryOp::Mul,
        Box::new(Expression::Float(alpha)),
        Box::new(x),
    );
    let exp_term = Expression::Function(Function::Exp, vec![exp_arg]);
    Ok(Expression::Binary(
        BinaryOp::Mul,
        Box::new(poly_part),
        Box::new(exp_term),
    ))
}

/// Returns true if the homogeneous solution has a root with given real/imag parts.
pub(super) fn hom_has_root(hom: &SecondOrderSolution, re: f64, im: f64) -> bool {
    use crate::ode::RootType;
    match hom.roots.root_type {
        RootType::TwoDistinctReal => {
            im.abs() < 1e-10
                && ((hom.roots.r1 - re).abs() < 1e-10 || (hom.roots.r2 - re).abs() < 1e-10)
        }
        RootType::RepeatedReal => im.abs() < 1e-10 && (hom.roots.r1 - re).abs() < 1e-10,
        RootType::ComplexConjugate => {
            (hom.roots.r1 - re).abs() < 1e-10
                && ((hom.roots.r2 - im).abs() < 1e-10 || (hom.roots.r2 + im).abs() < 1e-10)
        }
    }
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

pub(super) fn format_roots(roots: &[CharRoot]) -> String {
    roots
        .iter()
        .map(|r| {
            if r.is_real() {
                format!("{:.4} (mult {})", r.real, r.multiplicity)
            } else {
                format!("{:.4}±{:.4}i (mult {})", r.real, r.imag, r.multiplicity)
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

impl fmt::Display for CharRoot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_real() {
            write!(f, "{:.4} (mult {})", self.real, self.multiplicity)
        } else {
            write!(
                f,
                "{:.4}±{:.4}i (mult {})",
                self.real, self.imag, self.multiplicity
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
