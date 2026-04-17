//! Quadratic equation solver for equations of the form `ax² + bx + c = 0`.
//!
//! Compiles both sides of the equation to canonical [`Arc<Expr>`] form,
//! extracts the `(a, b, c)` coefficients directly from the resulting
//! [`AddNode`], and applies the quadratic formula over `f64` arithmetic.

use std::sync::Arc;

use num_complex::Complex64;

use crate::ast::{BinaryOp, Equation, Expression, Variable};
use crate::numeric::compile::compile;
use crate::numeric::{normalize, Expr, SymbolId};
use crate::resolution_path::{Operation, ResolutionPath, ResolutionPathBuilder, StepAnnotation};

use super::helpers::{contains_symbol, has_obvious_nonlinearity_expr, simplify_numeric_expression};
use super::types::{Solution, SolverError, SolverResult};
use super::Solver;

/// Quadratic equation solver for equations of the form `ax² + bx + c = 0`.
///
/// Applies the quadratic formula `x = (-b ± √(b² − 4ac)) / (2a)` and
/// returns zero, one, or two real roots, or a pair of complex conjugate
/// roots depending on the sign of the discriminant.
///
/// # See Also
///
/// - [`super::LinearSolver`]: for the degenerate case `a = 0`.
/// - [`super::PolynomialSolver`]: general polynomial solver; delegates
///   to this one at degree 2.
/// - [`super::SmartSolver`]: dispatcher that selects this solver when
///   the equation is quadratic.
#[derive(Debug, Default)]
pub struct QuadraticSolver;

impl QuadraticSolver {
    /// Create a new quadratic equation solver.
    pub fn new() -> Self {
        Self
    }
}

impl Solver for QuadraticSolver {
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
            return Err(SolverError::CannotSolve(format!(
                "Variable '{}' not found in equation",
                var_name
            )));
        }

        let (a, b, c) = extract_quadratic(&residual, var_id).ok_or_else(|| {
            SolverError::CannotSolve(format!(
                "Could not extract quadratic coefficients for '{}'",
                var_name
            ))
        })?;

        path = path.step(
            Operation::Simplify,
            format!("Identified coefficients: a={}, b={}, c={}", a, b, c),
            Expression::Integer(0),
        );

        // Degenerate: a = 0 → linear / constant.
        if a.abs() < 1e-15 {
            if b.abs() < 1e-15 {
                if c.abs() < 1e-15 {
                    let resolution_path = path.finish(Expression::Integer(0));
                    return Ok((Solution::Infinite, resolution_path));
                }
                return Err(SolverError::NoSolution);
            }
            let solution = simplify_numeric_expression(-c / b);
            let resolution_path = path.finish(solution.clone());
            return Ok((Solution::Unique(solution), resolution_path));
        }

        let discriminant = b * b - 4.0 * a * c;
        path = path.annotated_step(
            Operation::Simplify,
            format!("Computed discriminant: Δ = b² - 4ac = {}", discriminant),
            Expression::Float(discriminant),
            StepAnnotation::elementary(),
        );

        let epsilon = 1e-15;
        if discriminant > epsilon {
            let sqrt_disc = discriminant.sqrt();
            let x1 = (-b + sqrt_disc) / (2.0 * a);
            let x2 = (-b - sqrt_disc) / (2.0 * a);
            let root1 = simplify_numeric_expression(x1);
            let root2 = simplify_numeric_expression(x2);
            path = path.annotated_step(
                Operation::Simplify,
                format!("Quadratic formula: x = (-b ± √Δ)/(2a) = {} or {}", x1, x2),
                root1.clone(),
                StepAnnotation::algebraic("Quadratic Formula"),
            );
            let resolution_path = path.finish(root1.clone());
            Ok((Solution::Multiple(vec![root1, root2]), resolution_path))
        } else if discriminant.abs() <= epsilon {
            let x = -b / (2.0 * a);
            let root = simplify_numeric_expression(x);
            path = path.annotated_step(
                Operation::Simplify,
                format!("Quadratic formula (Δ = 0): x = -b/(2a) = {}", x),
                root.clone(),
                StepAnnotation::algebraic("Quadratic Formula"),
            );
            let resolution_path = path.finish(root.clone());
            Ok((Solution::Unique(root), resolution_path))
        } else {
            let real_part = -b / (2.0 * a);
            let imag_part = (-discriminant).sqrt() / (2.0 * a);
            let root1 = Expression::Complex(Complex64::new(real_part, imag_part));
            let root2 = Expression::Complex(Complex64::new(real_part, -imag_part));
            path = path.annotated_step(
                Operation::Simplify,
                format!("Complex roots: x = {} ± {}i", real_part, imag_part),
                root1.clone(),
                StepAnnotation::algebraic("Quadratic Formula"),
            );
            path = path.annotated_step(
                Operation::ComplexDecomposition {
                    original_var: var_name.clone(),
                    real_var: format!("{}_re", var_name),
                    imag_var: format!("{}_im", var_name),
                },
                format!(
                    "Complex Roots: real part = {}, imaginary part = ±{}",
                    real_part, imag_part
                ),
                root1.clone(),
                StepAnnotation::algebraic("Complex Roots"),
            );
            let resolution_path = path.finish(root1.clone());
            Ok((Solution::Multiple(vec![root1, root2]), resolution_path))
        }
    }

    fn can_solve(&self, equation: &Equation) -> bool {
        let lhs = compile(&equation.left);
        let rhs = compile(&equation.right);
        has_obvious_nonlinearity_expr(&lhs) || has_obvious_nonlinearity_expr(&rhs)
    }
}

/// Extract `(a, b, c)` such that `residual = a·var² + b·var + c`.
/// Returns `None` when the expression is not a quadratic in `var` with
/// numeric coefficients (symbolic coefficients, higher-degree terms, or
/// other variables all trigger rejection).
fn extract_quadratic(residual: &Arc<Expr>, var: SymbolId) -> Option<(f64, f64, f64)> {
    match residual.as_ref() {
        Expr::Add(node) => {
            let mut a = 0.0;
            let mut b = 0.0;
            let mut c = node.constant.to_f64();
            for (term, coeff) in &node.terms {
                let cr = coeff.to_f64();
                match classify_term(term, var)? {
                    TermKind::Constant => c += cr,
                    TermKind::Linear => b += cr,
                    TermKind::Quadratic => a += cr,
                }
            }
            Some((a, b, c))
        }
        Expr::Pow(base, exp) if is_var_pow(base, exp, var, 2) => Some((1.0, 0.0, 0.0)),
        Expr::Symbol(s) if *s == var => Some((0.0, 1.0, 0.0)),
        Expr::Integer(n) => n.to_i64().map(|v| (0.0, 0.0, v as f64)),
        Expr::Rational(r) => Some((0.0, 0.0, r.to_f64())),
        Expr::Float(f) => Some((0.0, 0.0, *f)),
        _ => None,
    }
}

enum TermKind {
    Constant,
    Linear,
    Quadratic,
}

fn classify_term(term: &Arc<Expr>, var: SymbolId) -> Option<TermKind> {
    if !contains_symbol(term, var) {
        return Some(TermKind::Constant);
    }
    match term.as_ref() {
        Expr::Symbol(s) if *s == var => Some(TermKind::Linear),
        Expr::Pow(base, exp) if is_var_pow(base, exp, var, 2) => Some(TermKind::Quadratic),
        Expr::Pow(base, exp) if is_var_pow(base, exp, var, 1) => Some(TermKind::Linear),
        _ => None,
    }
}

fn is_var_pow(base: &Arc<Expr>, exp: &Arc<Expr>, var: SymbolId, degree: i64) -> bool {
    let base_is_var = matches!(base.as_ref(), Expr::Symbol(s) if *s == var);
    let exp_matches = matches!(exp.as_ref(), Expr::Integer(n) if n.to_i64() == Some(degree));
    base_is_var && exp_matches
}
