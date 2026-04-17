//! Polynomial equation solver for general degree n polynomials.
//!
//! The [`Solver`] impl extracts real-valued polynomial coefficients directly
//! from the canonical [`Arc<Expr>`] residual (`lhs − rhs`), dispatches by
//! degree, and hands off to the appropriate closed-form or numerical
//! routine. Closed-form routines still operate on plain `f64` slices.

use std::sync::Arc;

use crate::ast::{BinaryOp, Equation, Expression, Variable};
use crate::numeric::compile::compile;
use crate::numeric::{normalize, Expr, SymbolId};
use crate::resolution_path::{Operation, ResolutionPath, ResolutionPathBuilder, StepAnnotation};

use super::helpers::{contains_symbol, is_polynomial_expr, simplify_numeric_expression};
use super::linear::LinearSolver;
use super::quadratic::QuadraticSolver;
use super::types::{Solution, SolverError, SolverResult};
use super::Solver;

/// Solve cubic equation ax³ + bx² + cx + d = 0 using Cardano's formula.
/// coeffs = [d, c, b, a] (constant term first)
fn solve_cubic(
    coeffs: &[f64],
    _var: &str,
    mut path: ResolutionPathBuilder,
) -> SolverResult<(Solution, ResolutionPath)> {
    if coeffs.len() < 4 {
        return Err(SolverError::CannotSolve(
            "Not a cubic polynomial".to_string(),
        ));
    }

    let d = coeffs[0];
    let c = coeffs[1];
    let b = coeffs[2];
    let a = coeffs[3];

    if a.abs() < 1e-15 {
        // Not actually cubic, delegate to quadratic
        return Err(SolverError::CannotSolve(
            "Leading coefficient is zero".to_string(),
        ));
    }

    // Normalize to monic form: x³ + px² + qx + r = 0
    let p = b / a;
    let q = c / a;
    let r = d / a;

    // Build an expression representing the normalized monic cubic
    let monic_cubic = Expression::Binary(
        BinaryOp::Add,
        Box::new(Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Binary(
                BinaryOp::Add,
                Box::new(Expression::Power(
                    Box::new(Expression::Variable(Variable::new(_var))),
                    Box::new(Expression::Integer(3)),
                )),
                Box::new(Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(Expression::Float(p)),
                    Box::new(Expression::Power(
                        Box::new(Expression::Variable(Variable::new(_var))),
                        Box::new(Expression::Integer(2)),
                    )),
                )),
            )),
            Box::new(Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Float(q)),
                Box::new(Expression::Variable(Variable::new(_var))),
            )),
        )),
        Box::new(Expression::Float(r)),
    );

    path = path.annotated_step(
        Operation::Simplify,
        format!("Normalized cubic: x³ + {}x² + {}x + {} = 0", p, q, r),
        monic_cubic,
        StepAnnotation::elementary(),
    );

    // Depress the cubic: substitute x = t - p/3
    // t³ + pt + q = 0 where:
    // p = q - p²/3
    // q = r - pq/3 + 2p³/27
    let dep_p = q - p * p / 3.0;
    let dep_q = r - p * q / 3.0 + 2.0 * p * p * p / 27.0;

    // Build an expression representing the depressed cubic coefficients
    let depressed_cubic = Expression::Binary(
        BinaryOp::Add,
        Box::new(Expression::Float(dep_p)),
        Box::new(Expression::Float(dep_q)),
    );

    path = path.annotated_step(
        Operation::Simplify,
        format!("Depressed cubic: t³ + {}t + {} = 0", dep_p, dep_q),
        depressed_cubic,
        StepAnnotation::algebraic("Tschirnhaus Transformation"),
    );

    // Discriminant: Δ = -4p³ - 27q²
    let discriminant = -4.0 * dep_p * dep_p * dep_p - 27.0 * dep_q * dep_q;

    path = path.step(
        Operation::Simplify,
        format!("Discriminant: Δ = {}", discriminant),
        Expression::Float(discriminant),
    );

    let shift = -p / 3.0;
    let roots: Vec<Expression>;

    if discriminant.abs() < 1e-10 {
        // All roots are real, at least two are equal
        if dep_p.abs() < 1e-10 && dep_q.abs() < 1e-10 {
            // Triple root at t = 0
            let root = simplify_numeric_expression(shift);
            roots = vec![root.clone(), root.clone(), root];
        } else {
            // One single root and one double root
            let t1 = 3.0 * dep_q / dep_p;
            let t2 = -3.0 * dep_q / (2.0 * dep_p);
            roots = vec![
                simplify_numeric_expression(t1 + shift),
                simplify_numeric_expression(t2 + shift),
                simplify_numeric_expression(t2 + shift),
            ];
        }
    } else if discriminant > 0.0 {
        // Three distinct real roots (casus irreducibilis)
        // Use trigonometric method
        let m = 2.0 * (-dep_p / 3.0).sqrt();
        let theta = (3.0 * dep_q / (dep_p * m)).acos() / 3.0;

        let t1 = m * theta.cos();
        let t2 = m * (theta - 2.0 * std::f64::consts::PI / 3.0).cos();
        let t3 = m * (theta - 4.0 * std::f64::consts::PI / 3.0).cos();

        roots = vec![
            simplify_numeric_expression(t1 + shift),
            simplify_numeric_expression(t2 + shift),
            simplify_numeric_expression(t3 + shift),
        ];
    } else {
        // One real root and two complex conjugate roots
        // Use Cardano's formula
        let sqrt_term = (dep_q * dep_q / 4.0 + dep_p * dep_p * dep_p / 27.0).sqrt();
        let u = (-dep_q / 2.0 + sqrt_term).cbrt();
        let v = (-dep_q / 2.0 - sqrt_term).cbrt();

        let t_real = u + v;
        let real_part = -0.5 * (u + v) + shift;
        let imag_part = (3.0_f64).sqrt() / 2.0 * (u - v);

        roots = vec![
            simplify_numeric_expression(t_real + shift),
            Expression::Complex(num_complex::Complex64::new(real_part, imag_part)),
            Expression::Complex(num_complex::Complex64::new(real_part, -imag_part)),
        ];
    }

    path = path.annotated_step(
        Operation::Simplify,
        "Applied Cardano's formula".to_string(),
        roots[0].clone(),
        StepAnnotation::algebraic("Cardano's Formula"),
    );

    let resolution_path = path.finish(roots[0].clone());
    Ok((Solution::Multiple(roots), resolution_path))
}

/// Solve quartic equation ax⁴ + bx³ + cx² + dx + e = 0 using Ferrari's method.
/// coeffs = [e, d, c, b, a] (constant term first)
fn solve_quartic(
    coeffs: &[f64],
    _var: &str,
    mut path: ResolutionPathBuilder,
) -> SolverResult<(Solution, ResolutionPath)> {
    if coeffs.len() < 5 {
        return Err(SolverError::CannotSolve(
            "Not a quartic polynomial".to_string(),
        ));
    }

    let e = coeffs[0];
    let d = coeffs[1];
    let c = coeffs[2];
    let b = coeffs[3];
    let a = coeffs[4];

    if a.abs() < 1e-15 {
        return Err(SolverError::CannotSolve(
            "Leading coefficient is zero".to_string(),
        ));
    }

    // Normalize to monic form: x⁴ + px³ + qx² + rx + s = 0
    let p = b / a;
    let q = c / a;
    let r = d / a;
    let s = e / a;

    // Build an expression representing the normalized monic quartic
    let monic_quartic = Expression::Binary(
        BinaryOp::Add,
        Box::new(Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Binary(
                BinaryOp::Add,
                Box::new(Expression::Binary(
                    BinaryOp::Add,
                    Box::new(Expression::Power(
                        Box::new(Expression::Variable(Variable::new(_var))),
                        Box::new(Expression::Integer(4)),
                    )),
                    Box::new(Expression::Binary(
                        BinaryOp::Mul,
                        Box::new(Expression::Float(p)),
                        Box::new(Expression::Power(
                            Box::new(Expression::Variable(Variable::new(_var))),
                            Box::new(Expression::Integer(3)),
                        )),
                    )),
                )),
                Box::new(Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(Expression::Float(q)),
                    Box::new(Expression::Power(
                        Box::new(Expression::Variable(Variable::new(_var))),
                        Box::new(Expression::Integer(2)),
                    )),
                )),
            )),
            Box::new(Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Float(r)),
                Box::new(Expression::Variable(Variable::new(_var))),
            )),
        )),
        Box::new(Expression::Float(s)),
    );

    path = path.annotated_step(
        Operation::Simplify,
        format!(
            "Normalized quartic: x⁴ + {}x³ + {}x² + {}x + {} = 0",
            p, q, r, s
        ),
        monic_quartic,
        StepAnnotation::elementary(),
    );

    // Depress the quartic: substitute x = y - p/4
    // y⁴ + αy² + βy + γ = 0
    let alpha = q - 3.0 * p * p / 8.0;
    let beta = r - p * q / 2.0 + p * p * p / 8.0;
    let gamma = s - p * r / 4.0 + p * p * q / 16.0 - 3.0 * p * p * p * p / 256.0;

    // Build an expression representing the depressed quartic coefficients
    let depressed_quartic = Expression::Binary(
        BinaryOp::Add,
        Box::new(Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Float(alpha)),
            Box::new(Expression::Float(beta)),
        )),
        Box::new(Expression::Float(gamma)),
    );

    path = path.annotated_step(
        Operation::Simplify,
        format!(
            "Depressed quartic: y⁴ + {}y² + {}y + {} = 0",
            alpha, beta, gamma
        ),
        depressed_quartic,
        StepAnnotation::algebraic("Tschirnhaus Transformation"),
    );

    let shift = -p / 4.0;

    // Handle special case: β = 0 (biquadratic)
    if beta.abs() < 1e-15 {
        // y⁴ + αy² + γ = 0, substitute u = y²
        let disc = alpha * alpha - 4.0 * gamma;
        if disc < -1e-15 {
            // Complex roots
            let u1_real = -alpha / 2.0;
            let u1_imag = (-disc).sqrt() / 2.0;

            // y² = u gives y = ±√u (complex square roots)
            let mut roots = Vec::new();
            for sign1 in [-1.0, 1.0] {
                let u_real = u1_real;
                let u_imag = sign1 * u1_imag;
                // √(a + bi) = ±(√((r+a)/2) + i*sign(b)*√((r-a)/2))
                let r = (u_real * u_real + u_imag * u_imag).sqrt();
                let sqrt_real = ((r + u_real) / 2.0).sqrt();
                let sqrt_imag = u_imag.signum() * ((r - u_real) / 2.0).sqrt();
                roots.push(Expression::Complex(num_complex::Complex64::new(
                    sqrt_real + shift,
                    sqrt_imag,
                )));
                roots.push(Expression::Complex(num_complex::Complex64::new(
                    -sqrt_real + shift,
                    -sqrt_imag,
                )));
            }
            path = path.step(
                Operation::Simplify,
                "Solved biquadratic via complex square roots".to_string(),
                roots[0].clone(),
            );
            let resolution_path = path.finish(roots[0].clone());
            return Ok((Solution::Multiple(roots), resolution_path));
        } else {
            let u1 = (-alpha + disc.sqrt()) / 2.0;
            let u2 = (-alpha - disc.sqrt()) / 2.0;

            let mut roots = Vec::new();
            for u in [u1, u2] {
                if u >= 0.0 {
                    roots.push(simplify_numeric_expression(u.sqrt() + shift));
                    roots.push(simplify_numeric_expression(-u.sqrt() + shift));
                } else {
                    let imag = (-u).sqrt();
                    roots.push(Expression::Complex(num_complex::Complex64::new(
                        shift, imag,
                    )));
                    roots.push(Expression::Complex(num_complex::Complex64::new(
                        shift, -imag,
                    )));
                }
            }
            path = path.step(
                Operation::Simplify,
                format!("Solved biquadratic: u = y², u₁ = {}, u₂ = {}", u1, u2),
                roots[0].clone(),
            );
            let resolution_path = path.finish(roots[0].clone());
            return Ok((Solution::Multiple(roots), resolution_path));
        }
    }

    // Solve resolvent cubic: m³ + (α/2)m² + ((α² - 4γ)/16)m - β²/64 = 0
    let resolvent_coeffs = vec![
        -beta * beta / 64.0,
        (alpha * alpha - 4.0 * gamma) / 16.0,
        alpha / 2.0,
        1.0,
    ];

    // Get one real root of the resolvent cubic
    let dep_p = resolvent_coeffs[1] - resolvent_coeffs[2] * resolvent_coeffs[2] / 3.0;
    let dep_q = resolvent_coeffs[0] - resolvent_coeffs[2] * resolvent_coeffs[1] / 3.0
        + 2.0 * resolvent_coeffs[2] * resolvent_coeffs[2] * resolvent_coeffs[2] / 27.0;

    let disc_cubic = -4.0 * dep_p * dep_p * dep_p - 27.0 * dep_q * dep_q;

    let m: f64;
    if disc_cubic > 1e-10 {
        // Use trigonometric method for real root
        let sqrt_term = 2.0 * (-dep_p / 3.0).sqrt();
        let theta = (3.0 * dep_q / (dep_p * sqrt_term)).acos() / 3.0;
        m = sqrt_term * theta.cos() - resolvent_coeffs[2] / 3.0;
    } else {
        // Use Cardano's formula
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
        m = u + v - resolvent_coeffs[2] / 3.0;
    }

    path = path.step(
        Operation::Simplify,
        format!("Resolvent cubic root: m = {}", m),
        Expression::Float(m),
    );

    // Factor quartic: (y² + m)² = (α + 2m)y² - βy + (m² + αm + γ - γ)
    // Using Ferrari's factorization into two quadratics
    let sqrt_2m_alpha = (2.0 * m + alpha).max(0.0).sqrt();

    // y² + sqrt(2m+α)y + (m + β/(2*sqrt(2m+α))) = 0
    // y² - sqrt(2m+α)y + (m - β/(2*sqrt(2m+α))) = 0
    let term = if sqrt_2m_alpha.abs() > 1e-10 {
        beta / (2.0 * sqrt_2m_alpha)
    } else {
        0.0
    };

    let mut roots = Vec::new();

    // First quadratic: y² + sqrt(2m+α)y + (m + term) = 0
    let a1 = 1.0;
    let b1 = sqrt_2m_alpha;
    let c1 = m + term;
    let disc1 = b1 * b1 - 4.0 * a1 * c1;

    if disc1 >= 0.0 {
        roots.push(simplify_numeric_expression(
            (-b1 + disc1.sqrt()) / 2.0 + shift,
        ));
        roots.push(simplify_numeric_expression(
            (-b1 - disc1.sqrt()) / 2.0 + shift,
        ));
    } else {
        let real = -b1 / 2.0 + shift;
        let imag = (-disc1).sqrt() / 2.0;
        roots.push(Expression::Complex(num_complex::Complex64::new(real, imag)));
        roots.push(Expression::Complex(num_complex::Complex64::new(
            real, -imag,
        )));
    }

    // Second quadratic: y² - sqrt(2m+α)y + (m - term) = 0
    let b2 = -sqrt_2m_alpha;
    let c2 = m - term;
    let disc2 = b2 * b2 - 4.0 * a1 * c2;

    if disc2 >= 0.0 {
        roots.push(simplify_numeric_expression(
            (-b2 + disc2.sqrt()) / 2.0 + shift,
        ));
        roots.push(simplify_numeric_expression(
            (-b2 - disc2.sqrt()) / 2.0 + shift,
        ));
    } else {
        let real = -b2 / 2.0 + shift;
        let imag = (-disc2).sqrt() / 2.0;
        roots.push(Expression::Complex(num_complex::Complex64::new(real, imag)));
        roots.push(Expression::Complex(num_complex::Complex64::new(
            real, -imag,
        )));
    }

    path = path.annotated_step(
        Operation::Simplify,
        "Applied Ferrari's method".to_string(),
        roots[0].clone(),
        StepAnnotation::algebraic("Ferrari's Method"),
    );

    let resolution_path = path.finish(roots[0].clone());
    Ok((Solution::Multiple(roots), resolution_path))
}

/// Solve polynomial of degree 5+ using numerical methods (Durand-Kerner).
fn solve_polynomial_numerically(
    coeffs: &[f64],
    _var: &str,
    mut path: ResolutionPathBuilder,
) -> SolverResult<(Solution, ResolutionPath)> {
    let degree = coeffs.len() - 1;
    if degree < 1 {
        return Err(SolverError::CannotSolve("Invalid polynomial".to_string()));
    }

    // Find leading coefficient
    let leading = coeffs[degree];
    if leading.abs() < 1e-15 {
        return Err(SolverError::CannotSolve(
            "Leading coefficient is zero".to_string(),
        ));
    }

    path = path.step(
        Operation::Simplify,
        format!(
            "Solving degree {} polynomial numerically (Durand-Kerner method)",
            degree
        ),
        Expression::Integer(degree as i64),
    );

    // Initial guess: roots evenly spaced on a circle
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

    // Durand-Kerner iteration
    let max_iter = 100;
    let tolerance = 1e-12;

    for _ in 0..max_iter {
        let mut max_change: f64 = 0.0;

        for i in 0..degree {
            // Evaluate polynomial at roots[i]
            let mut p_val = num_complex::Complex64::new(0.0, 0.0);
            let mut power = num_complex::Complex64::new(1.0, 0.0);
            for &coeff in coeffs.iter() {
                p_val += num_complex::Complex64::new(coeff, 0.0) * power;
                power *= roots[i];
            }

            // Compute denominator product
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
            break;
        }
    }

    // Convert to Expression
    let root_exprs: Vec<Expression> = roots
        .iter()
        .map(|r| {
            if r.im.abs() < 1e-10 {
                simplify_numeric_expression(r.re)
            } else {
                Expression::Complex(*r)
            }
        })
        .collect();

    path = path.step(
        Operation::Simplify,
        format!("Found {} roots numerically", degree),
        root_exprs[0].clone(),
    );

    let resolution_path = path.finish(root_exprs[0].clone());
    Ok((Solution::Multiple(root_exprs), resolution_path))
}

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
    fn solve(
        &self,
        equation: &Equation,
        variable: &Variable,
    ) -> SolverResult<(Solution, ResolutionPath)> {
        let var_name = &variable.name;
        let var_id = SymbolId::intern(var_name);

        let lhs = compile(&equation.left);
        let rhs = compile(&equation.right);
        let residual = normalize::sub(lhs, rhs);

        let initial_expr = Expression::Binary(
            BinaryOp::Sub,
            Box::new(equation.left.clone()),
            Box::new(equation.right.clone()),
        );
        let mut path = ResolutionPathBuilder::new(initial_expr);

        if !contains_symbol(&residual, var_id) {
            // Residual is a constant: 0 = const.
            if residual.is_zero() {
                let resolution_path = path.finish(Expression::Integer(0));
                return Ok((Solution::Infinite, resolution_path));
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

        path = path.step(
            Operation::Simplify,
            format!("Identified polynomial of degree {}", degree),
            Expression::Integer(degree as i64),
        );

        match degree {
            0 => {
                // Already handled by the contains_symbol branch above; any
                // residual that reaches here with degree 0 and still
                // contains the variable is an extractor error.
                Err(SolverError::CannotSolve(
                    "Cannot evaluate constant expression".to_string(),
                ))
            }
            1 => LinearSolver::new().solve(equation, variable),
            2 => QuadraticSolver::new().solve(equation, variable),
            3 => solve_cubic(&coeffs, var_name, path),
            4 => solve_quartic(&coeffs, var_name, path),
            _ => solve_polynomial_numerically(&coeffs, var_name, path),
        }
    }

    fn can_solve(&self, equation: &Equation) -> bool {
        let lhs = compile(&equation.left);
        let rhs = compile(&equation.right);
        is_polynomial_expr(&lhs) && is_polynomial_expr(&rhs)
    }
}

// ── Expr-based polynomial coefficient extraction ─────────────────────────

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
