//! Polynomial equation solver for general degree n polynomials.
//!
//! The [`Solver`] impl extracts real-valued polynomial coefficients directly
//! from the canonical [`Arc<Expr>`] residual (`lhs − rhs`), dispatches by
//! degree, and hands off to the appropriate closed-form or numerical
//! routine. Closed-form routines still operate on plain `f64` slices.

mod cubic;
mod numerical;
mod quartic;

use std::sync::Arc;

use crate::ast::{Equation, Variable};
use crate::numeric::compile::compile;
use crate::numeric::trace::{Step, TechniqueTag, Trace};
use crate::numeric::{normalize, Expr, SymbolId};

use super::helpers::{contains_symbol, is_polynomial_expr};
use super::linear::LinearSolver;
use super::quadratic::QuadraticSolver;
use super::types::{Solution, SolverError, SolverResult};
use super::Solver;

use cubic::solve_cubic;
use numerical::solve_polynomial_numerically;
use quartic::solve_quartic;

// ── Shared helper (used by cubic, quartic, and numerical submodules) ──────────

/// Build a complex number `re + im*i` as an `Arc<Expr>`.
///
/// For `im == 0` returns `Expr::float(re)`.
/// Otherwise returns `Expr::complex(re, im)` which decompiles to
/// `Expression::Complex(Complex64{re, im})`.
pub(self) fn symbolic_complex(re: f64, im: f64) -> Arc<Expr> {
    if im == 0.0 {
        return Expr::float(re);
    }
    Expr::complex(re, im)
}

// ── Public solver struct ──────────────────────────────────────────────────────

/// Polynomial equation solver for general degree n polynomials.
///
/// Solves polynomial equations in one variable using closed-form algebraic formulas
/// for degrees 1-4, and numerical methods for higher degrees.
///
/// # Mathematical Foundation
///
/// A polynomial equation has the general form:
/// ```text
/// aₙxⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₂x² + a₁x + a₀ = 0
/// ```
/// where n is the degree and aₙ ≠ 0.
///
/// # Solution Methods by Degree
///
/// - **Degree 1 (Linear)**: Direct division: x = -a₀/a₁
/// - **Degree 2 (Quadratic)**: Quadratic formula: x = (-b ± √(b²-4ac))/(2a)
/// - **Degree 3 (Cubic)**: Cardano's formula or trigonometric method
/// - **Degree 4 (Quartic)**: Ferrari's method or resolvent cubic
/// - **Degree 5+ (Quintic and higher)**: Numerical root-finding methods
///
/// # See Also
///
/// - [`super::LinearSolver`]: Specialized solver for degree 1
/// - [`super::QuadraticSolver`]: Specialized solver for degree 2
/// - [`super::SmartSolver`]: Automatically selects PolynomialSolver for polynomial equations
///
/// # References
///
/// - [Cubic function](https://en.wikipedia.org/wiki/Cubic_function)
/// - [Quartic function](https://en.wikipedia.org/wiki/Quartic_function)
/// - [Abel-Ruffini theorem](https://en.wikipedia.org/wiki/Abel%E2%80%93Ruffini_theorem)
/// - [Durand-Kerner method](https://en.wikipedia.org/wiki/Durand%E2%80%93Kerner_method)
#[derive(Debug, Default)]
pub struct PolynomialSolver;

impl PolynomialSolver {
    /// Create a new polynomial equation solver.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::solver::PolynomialSolver;
    ///
    /// let solver = PolynomialSolver::new();
    /// ```
    pub fn new() -> Self {
        Self
    }
}

impl Solver for PolynomialSolver {
    fn solve(&self, equation: &Equation, variable: &Variable) -> SolverResult<(Solution, Trace)> {
        let var_name = &variable.name;
        let var_id = SymbolId::intern(var_name);

        let lhs = compile(&equation.left);
        let rhs = compile(&equation.right);
        let residual = normalize::sub(lhs, rhs);

        let mut trace = Trace::new();

        if !contains_symbol(&residual, var_id) {
            if residual.is_zero() {
                return Ok((Solution::Infinite, trace));
            }
            return Err(SolverError::NoSolution);
        }

        let coeffs = extract_poly_coeffs_expr(&residual, var_id).ok_or_else(|| {
            SolverError::CannotSolve(format!(
                "Could not extract polynomial coefficients for '{}'",
                var_name
            ))
        })?;
        let degree = coeffs.len().saturating_sub(1);

        trace.push(Step::new(
            TechniqueTag::Simplification,
            format!("Identified polynomial of degree {}", degree),
        ));

        match degree {
            0 => Err(SolverError::CannotSolve(
                "Cannot evaluate constant expression".to_string(),
            )),
            1 => LinearSolver::new().solve(equation, variable),
            2 => QuadraticSolver::new().solve(equation, variable),
            3 => {
                let sol = solve_cubic(&coeffs, var_name, &mut trace)?;
                Ok((sol, trace))
            }
            4 => {
                let sol = solve_quartic(&coeffs, var_name, &mut trace)?;
                Ok((sol, trace))
            }
            _ => {
                let sol = solve_polynomial_numerically(&coeffs, var_name, &mut trace)?;
                Ok((sol, trace))
            }
        }
    }

    fn can_solve(&self, equation: &Equation) -> bool {
        let lhs = compile(&equation.left);
        let rhs = compile(&equation.right);
        is_polynomial_expr(&lhs) && is_polynomial_expr(&rhs)
    }
}

// ── Expr-based polynomial coefficient extraction ──────────────────────────────

/// Extract real-valued polynomial coefficients `[a0, a1, ..., aN]` from an
/// `Arc<Expr>` residual, where the coefficient of `var^i` is at index `i`.
/// Returns `None` when the expression contains a shape incompatible with
/// a pure real polynomial in `var` (symbolic coefficient, negative power,
/// product of variable factors, etc.).
fn extract_poly_coeffs_expr(residual: &Arc<Expr>, var: SymbolId) -> Option<Vec<f64>> {
    let mut coeffs: Vec<f64> = vec![0.0];
    add_into_coeffs(residual, var, 1.0, &mut coeffs)?;
    while coeffs.len() > 1 && coeffs.last().copied().unwrap_or(0.0).abs() < 1e-15 {
        coeffs.pop();
    }
    Some(coeffs)
}

fn add_into_coeffs(
    expr: &Arc<Expr>,
    var: SymbolId,
    scale: f64,
    coeffs: &mut Vec<f64>,
) -> Option<()> {
    match expr.as_ref() {
        Expr::Integer(n) => {
            bump(coeffs, 0, n.to_i64()? as f64 * scale);
            Some(())
        }
        Expr::Rational(r) => {
            bump(coeffs, 0, r.to_f64() * scale);
            Some(())
        }
        Expr::Float(f) => {
            bump(coeffs, 0, f * scale);
            Some(())
        }
        Expr::Symbol(s) if *s == var => {
            bump(coeffs, 1, scale);
            Some(())
        }
        Expr::Pow(base, exp) => {
            let base_is_var = matches!(base.as_ref(), Expr::Symbol(s) if *s == var);
            if !base_is_var {
                return None;
            }
            let deg = match exp.as_ref() {
                Expr::Integer(n) => n.to_i64()?,
                _ => return None,
            };
            if deg < 0 {
                return None;
            }
            bump(coeffs, deg as usize, scale);
            Some(())
        }
        Expr::Add(node) => {
            bump(coeffs, 0, node.constant.to_f64() * scale);
            for (term, coeff) in &node.terms {
                add_into_coeffs(term, var, scale * coeff.to_f64(), coeffs)?;
            }
            Some(())
        }
        Expr::Mul(node) => {
            let mut combined = scale * node.coeff.to_f64();
            let mut degree: usize = 0;
            let mut saw_var = false;
            for (base, exp) in &node.factors {
                if contains_symbol(base, var) || contains_symbol(exp, var) {
                    if saw_var {
                        return None;
                    }
                    match (base.as_ref(), exp.as_ref()) {
                        (Expr::Symbol(s), Expr::Integer(n)) if *s == var => {
                            let d = n.to_i64()?;
                            if d < 0 {
                                return None;
                            }
                            degree = d as usize;
                            saw_var = true;
                        }
                        _ => return None,
                    }
                } else {
                    match (base.as_ref(), exp.as_ref()) {
                        (Expr::Integer(n), Expr::Integer(e)) => {
                            let b = n.to_i64()? as f64;
                            let ev = e.to_i64()?;
                            combined *= b.powi(ev as i32);
                        }
                        (Expr::Rational(r), Expr::Integer(e)) => {
                            let ev = e.to_i64()?;
                            combined *= r.to_f64().powi(ev as i32);
                        }
                        (Expr::Float(f), Expr::Integer(e)) => {
                            let ev = e.to_i64()?;
                            combined *= f.powi(ev as i32);
                        }
                        _ => return None,
                    }
                }
            }
            bump(coeffs, degree, combined);
            Some(())
        }
        _ => None,
    }
}

fn bump(coeffs: &mut Vec<f64>, degree: usize, value: f64) {
    while coeffs.len() <= degree {
        coeffs.push(0.0);
    }
    coeffs[degree] += value;
}
