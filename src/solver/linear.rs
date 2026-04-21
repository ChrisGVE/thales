//! Linear equation solver for equations of the form `ax + b = c`.
//!
//! Operates on [`Arc<Expr>`] internals: compiles both sides of the equation
//! to canonical form, attempts a fast purely-rational coefficient extraction
//! through [`super::coeff::extract_linear_coefficients`], and falls back to
//! [`super::symbolic_isolation::symbolic_isolate`] for linear equations that
//! carry symbolic (non-rational) coefficients.

use num::traits::Zero;

use crate::ast::{Equation, Expression, Variable};
use crate::numeric::compile::{compile, decompile};
use crate::numeric::trace::{Step, TechniqueTag, Trace};
use crate::numeric::{normalize, BigRational, Expr, SymbolId};

use super::coeff::extract_linear_coefficients;
use super::helpers::{contains_symbol, has_obvious_nonlinearity_expr, is_linear_in_variable_expr};
use super::symbolic_isolation::symbolic_isolate;
use super::types::{Solution, SolverError, SolverResult};
use super::Solver;

/// Linear equation solver for equations of the form `ax + b = c`.
///
/// Solves first-degree polynomial equations in one variable. Handles
/// linear patterns including:
/// - Simple variable: `x = 5`
/// - Multiplication: `3x = 12`
/// - Addition: `x + 7 = 10`
/// - Combined: `2x + 3 = 11`
/// - Parametric: `a*x + b = c`
///
/// # Mathematical Foundation
///
/// A linear equation in one variable has the general form `ax + b = c`.
/// The solution is obtained by subtracting `b` from both sides and
/// dividing by `a`, giving `x = (c - b) / a`.
///
/// # Limitations
///
/// - Only handles linear equations (degree 1)
/// - Cannot solve equations with the variable in denominators (`1/x = 2`)
/// - Cannot solve equations with the variable in exponents (`2^x = 8`)
/// - Cannot handle products of variables (`x*y = 5`)
///
/// For more complex equations, use [`super::TranscendentalSolver`] or
/// [`super::SmartSolver`].
///
/// # Examples
///
/// ```
/// use thales::solver::{LinearSolver, Solver};
/// use thales::ast::{Equation, Expression, Variable, BinaryOp};
///
/// // Solve: 2x + 3 = 11
/// let x = Expression::Variable(Variable::new("x"));
/// let two_x = Expression::Binary(
///     BinaryOp::Mul,
///     Box::new(Expression::Integer(2)),
///     Box::new(x),
/// );
/// let left = Expression::Binary(
///     BinaryOp::Add,
///     Box::new(two_x),
///     Box::new(Expression::Integer(3)),
/// );
/// let equation = Equation::new("linear", left, Expression::Integer(11));
///
/// let solver = LinearSolver::new();
/// let (solution, _path) = solver.solve(&equation, &Variable::new("x")).unwrap();
/// ```
#[derive(Debug, Default)]
pub struct LinearSolver;

impl LinearSolver {
    /// Create a new linear equation solver.
    pub fn new() -> Self {
        Self
    }
}

impl Solver for LinearSolver {
    fn solve(&self, equation: &Equation, variable: &Variable) -> SolverResult<(Solution, Trace)> {
        let var_name = &variable.name;
        let var_id = SymbolId::intern(var_name);

        // Compile both sides to Arc<Expr> canonical form.
        let lhs_arc = compile(&equation.left);
        let rhs_arc = compile(&equation.right);
        let residual = normalize::sub(lhs_arc.clone(), rhs_arc.clone());

        if !contains_symbol(&residual, var_id) {
            return Err(SolverError::CannotSolve(format!(
                "Variable '{}' not found in equation",
                var_name
            )));
        }

        if !is_linear_in_variable_expr(&lhs_arc, var_id)
            || !is_linear_in_variable_expr(&rhs_arc, var_id)
        {
            return Err(SolverError::UnsupportedEquationType);
        }

        let mut trace = Trace::new();

        // Fast path: the residual is `coeff·var + constant` with rational
        // coefficient and rational constant — solve directly.
        if let Ok((coeffs, constant)) =
            extract_linear_coefficients(&residual, std::slice::from_ref(variable))
        {
            let coeff = &coeffs[0];
            let result_expr = solve_rational_linear(coeff, &constant)?;
            trace.push(
                Step::new(
                    TechniqueTag::Isolation,
                    format!("Isolate {} on one side", variable),
                )
                .with_input(residual.clone())
                .with_output(compile(&result_expr)),
            );
            return Ok((Solution::Unique(result_expr), trace));
        }

        // Symbolic path: linear shape but coefficient or constant is not a
        // pure rational — hand off to the full isolation engine.
        let result_expr = symbolic_isolate(&lhs_arc, &rhs_arc, variable, &mut trace)?;
        trace.push(
            Step::new(
                TechniqueTag::Isolation,
                format!("Isolate {} on one side", variable),
            )
            .with_output(compile(&result_expr)),
        );
        Ok((Solution::Unique(result_expr), trace))
    }

    fn can_solve(&self, equation: &Equation) -> bool {
        let lhs = compile(&equation.left);
        let rhs = compile(&equation.right);
        !has_obvious_nonlinearity_expr(&lhs) && !has_obvious_nonlinearity_expr(&rhs)
    }
}

/// Solve `coeff · var + constant = 0` when both are exact rationals.
fn solve_rational_linear(coeff: &BigRational, constant: &BigRational) -> SolverResult<Expression> {
    if coeff.is_zero() {
        if constant.is_zero() {
            return Err(SolverError::InfiniteSolutions);
        }
        return Err(SolverError::NoSolution);
    }
    let numer = -constant;
    let solution = &numer / coeff;
    Ok(decompile(&rational_to_expr(solution)))
}

fn rational_to_expr(r: BigRational) -> std::sync::Arc<Expr> {
    if r.is_integer() {
        std::sync::Arc::new(Expr::Integer(r.numer().clone()))
    } else {
        std::sync::Arc::new(Expr::Rational(r))
    }
}
