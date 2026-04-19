//! Particular solution for polynomial forcing functions.
//!
//! Trial form: `A_d·x^d + … + A_0`, multiplied by `x^m` when the
//! characteristic equation has `0` as a root with multiplicity `m`.

use std::collections::HashMap;
use std::sync::Arc;

use crate::ast::{BinaryOp, Expression, Function, Variable};
use crate::numeric::evaluation::evaluate;
use crate::numeric::expr::Expr;
use crate::numeric::SymbolId;

use super::{ForcingType, ODEError, SecondOrderODE};

// ---------------------------------------------------------------------------
// Particular solution: polynomial forcing
// ---------------------------------------------------------------------------

/// Find `y_p` when `f(x)` is a polynomial of degree `d`.
///
/// The trial form is `A_d·x^d + … + A_0`.  If `c = 0` (zero is a
/// characteristic root), we multiply by `x` (or `x²` for double root).
pub(super) fn particular_polynomial(
    ode: &SecondOrderODE,
    degree: u32,
    steps: &mut Vec<String>,
) -> Result<Expression, ODEError> {
    let x_var = &ode.independent;

    // Determine resonance multiplier
    let multiplier = resonance_multiplier_polynomial(ode)?;
    let trial_degree = degree + multiplier;

    steps.push(format!(
        "Trial form: y_p = polynomial of degree {} (multiplier x^{})",
        trial_degree, multiplier
    ));

    // Collect coefficients from the forcing polynomial
    let forcing_arc = ode.forcing_arc();
    let x_id = SymbolId::intern(x_var);
    let forcing_coeffs = extract_polynomial_coeffs(&forcing_arc, x_id, degree)?;

    // Solve for undetermined coefficients by matching powers of x
    let yp_coeffs = solve_polynomial_system(ode, &forcing_coeffs, multiplier)?;

    steps.push(format!("Particular solution coefficients: {:?}", yp_coeffs));

    Ok(build_polynomial_expr(&yp_coeffs, multiplier, x_var))
}

/// Return the power of `x` by which the trial must be multiplied.
///
/// Returns `0`, `1`, or `2` based on whether `0` is a simple/double or
/// no characteristic root.
fn resonance_multiplier_polynomial(ode: &SecondOrderODE) -> Result<u32, ODEError> {
    const EPS: f64 = 1e-12;
    // Characteristic roots are roots of ar² + br + c = 0.
    // Root = 0 iff c = 0 (and a ≠ 0).
    if ode.c.abs() > EPS {
        return Ok(0); // 0 is not a root
    }
    // c = 0: r = 0 is a root.  If b = 0 as well, double root at 0.
    if ode.b.abs() > EPS {
        Ok(1)
    } else {
        // b = 0 and c = 0: characteristic eq is a·r² = 0, double root at 0
        Ok(2)
    }
}

/// Extract polynomial coefficients `[c0, c1, …, cn]` (constant term first)
/// by numerical evaluation of `f(x)` at `n+1` distinct points.
///
/// This is a Vandermonde interpolation — works for polynomials of degree
/// ≤ 20. Operates on the canonical `Arc<Expr>` form and samples through
/// [`crate::numeric::evaluation::evaluate`].
fn extract_polynomial_coeffs(
    expr: &Arc<Expr>,
    x: SymbolId,
    degree: u32,
) -> Result<Vec<f64>, ODEError> {
    let n = (degree + 1) as usize;
    let points: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let values: Vec<f64> = points
        .iter()
        .map(|&xi| {
            let mut env = HashMap::new();
            env.insert(x, xi);
            evaluate(expr, &env).unwrap_or(0.0)
        })
        .collect();

    // Vandermonde solve (Gaussian elimination for small n)
    vandermonde_solve(&points, &values).ok_or_else(|| {
        ODEError::CannotSolve("failed to extract polynomial coefficients".to_string())
    })
}

/// Solve `y_p` coefficients from the ODE constraint for polynomial forcing.
///
/// Given `y_p = Σ a_j · x^(j+m)` where `m` = multiplier, substitute into
/// the ODE and match coefficients of each power of `x`.
fn solve_polynomial_system(
    ode: &SecondOrderODE,
    forcing: &[f64],
    multiplier: u32,
) -> Result<Vec<f64>, ODEError> {
    let n = forcing.len();
    let m = multiplier as usize;
    let total = n + m; // highest power in trial is (n-1) + m

    // Build coefficients for each power in the trial y_p = Σ A_j x^(j+m)
    // y_p'' contribution and y_p' contribution are computed analytically.
    let mut coeffs = vec![0.0f64; n];

    // Work from highest to lowest power to determine A_{n-1}, …, A_0
    // For power x^(j+m), the equation is:
    //   a·(j+m)(j+m-1)·A_j + b·(j+m)·A_{j} + c·A_j
    //   + contributions from higher A_k = f_j
    // We do back-substitution (highest degree first).
    let mut determined = vec![0.0f64; n];

    for j in (0..n).rev() {
        let power = (j + m) as f64;
        // Direct contribution of A_j to this power's equation:
        let coeff_a_j = ode.a * power * (power - 1.0) + ode.b * power + ode.c;

        // Contribution from A_{j+1} (one degree higher) to this power
        // via first derivative: b*(j+1+m)*A_{j+1}*x^(j+m) — already determined
        let mut rhs = forcing[j];
        if j + 1 < n {
            // a·(j+1+m)·(j+m)·A_{j+1}  (second deriv drops power by 2 → not relevant here)
            // b·(j+1+m)·A_{j+1}  (first deriv of x^(j+1+m) = (j+1+m)·x^(j+m))
            let p1 = (j + 1 + m) as f64;
            rhs -= ode.b * p1 * determined[j + 1];
        }
        if j + 2 < n {
            // a·(j+2+m)·(j+1+m)·A_{j+2}  (second deriv of x^(j+2+m) = (j+2+m)(j+1+m)·x^(j+m))
            let p2 = (j + 2 + m) as f64;
            let p2m1 = (j + 1 + m) as f64;
            rhs -= ode.a * p2 * p2m1 * determined[j + 2];
        }

        if coeff_a_j.abs() < 1e-12 {
            return Err(ODEError::ResonanceDetected(format!(
                "Undetermined coefficient for x^{} vanishes — higher-order resonance not handled",
                j + m
            )));
        }
        determined[j] = rhs / coeff_a_j;
        coeffs[j] = determined[j];
    }

    Ok(coeffs)
}

/// Build `Σ a_j · x^(j+m)` as an `Expression`.
fn build_polynomial_expr(coeffs: &[f64], multiplier: u32, x_var: &str) -> Expression {
    let m = multiplier as i64;
    let mut terms: Vec<Expression> = Vec::new();

    for (j, &a_j) in coeffs.iter().enumerate() {
        if a_j.abs() < 1e-15 {
            continue;
        }
        let power = j as i64 + m;
        let x_pow = build_x_power(x_var, power);
        let term = if (a_j - 1.0).abs() < 1e-15 {
            x_pow
        } else {
            Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Float(a_j)),
                Box::new(x_pow),
            )
        };
        terms.push(term);
    }

    terms_to_sum(terms)
}

/// Build `x^n` as an `Expression`.
pub(super) fn build_x_power(x_var: &str, n: i64) -> Expression {
    let x = Expression::Variable(Variable::new(x_var));
    match n {
        0 => Expression::Integer(1),
        1 => x,
        _ => Expression::Power(Box::new(x), Box::new(Expression::Integer(n))),
    }
}

// ---------------------------------------------------------------------------
// Vandermonde solve (Lagrange interpolation for polynomial coefficients)
// ---------------------------------------------------------------------------

/// Solve the Vandermonde system `V·c = y` where `V[i][j] = x_i^j`.
///
/// Returns the coefficient vector `[c_0, c_1, …, c_{n-1}]` or `None` if
/// the system is singular.
fn vandermonde_solve(xs: &[f64], ys: &[f64]) -> Option<Vec<f64>> {
    let n = xs.len();
    if n == 0 {
        return None;
    }
    // Build augmented matrix [V | y]
    let mut mat: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row: Vec<f64> = (0..n).map(|j| xs[i].powi(j as i32)).collect();
            row.push(ys[i]);
            row
        })
        .collect();

    gaussian_eliminate(&mut mat, n)
}

/// Gaussian elimination with partial pivoting; returns solution vector.
fn gaussian_eliminate(mat: &mut Vec<Vec<f64>>, n: usize) -> Option<Vec<f64>> {
    const EPS: f64 = 1e-12;

    for col in 0..n {
        // Partial pivot
        let max_row =
            (col..n).max_by(|&a, &b| mat[a][col].abs().partial_cmp(&mat[b][col].abs()).unwrap())?;
        mat.swap(col, max_row);

        let pivot = mat[col][col];
        if pivot.abs() < EPS {
            return None;
        }

        for row in (col + 1)..n {
            let factor = mat[row][col] / pivot;
            for k in col..=n {
                let v = mat[col][k];
                mat[row][k] -= factor * v;
            }
        }
    }

    // Back substitution
    let mut result = vec![0.0; n];
    for row in (0..n).rev() {
        let mut val = mat[row][n];
        for k in (row + 1)..n {
            val -= mat[row][k] * result[k];
        }
        result[row] = val / mat[row][row];
    }
    Some(result)
}

// ---------------------------------------------------------------------------
// Utility: build sum from terms
// ---------------------------------------------------------------------------

/// Fold a `Vec<Expression>` into a left-associated sum.
/// Returns `Expression::Integer(0)` if empty.
fn terms_to_sum(terms: Vec<Expression>) -> Expression {
    if terms.is_empty() {
        return Expression::Integer(0);
    }
    terms
        .into_iter()
        .reduce(|acc, t| Expression::Binary(BinaryOp::Add, Box::new(acc), Box::new(t)))
        .expect("non-empty iterator always reduces")
}
