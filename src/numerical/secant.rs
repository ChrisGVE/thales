//! Secant method root finder (derivative-free).

use crate::ast::{Equation, Expression, Variable};
use crate::resolution_path::{Operation, ResolutionPath, ResolutionPathBuilder};
use std::collections::HashMap;

use super::{NumericalConfig, NumericalError, NumericalResult, NumericalSolution};

/// Secant method root finder (derivative-free).
///
/// Implements the secant method for finding roots of equations.  Instead of
/// evaluating the analytical (or symbolic) derivative, the secant method
/// approximates it from two previous function evaluations:
///
/// **x_{n+1} = x_n - f(x_n) · (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))**
///
/// # Convergence
///
/// * **Super-linear convergence**: order ≈ 1.618 (golden ratio) when the
///   starting points are close to the root.
/// * **No bracketing required**: unlike bisection, the root need not lie
///   between the two initial points.
/// * **No derivative needed**: useful when symbolic or numerical
///   differentiation is expensive or unavailable.
///
/// # Failure modes
///
/// * The denominator `f(x_n) - f(x_{n-1})` can become very small (near a
///   local extremum), causing the next iterate to fly far from the root.
///   The solver returns [`NumericalError::Unstable`] in that case.
/// * Starting points that are too far from the root may diverge; the solver
///   returns [`NumericalError::NoConvergence`] after `max_iterations`.
///
/// # Example
///
/// ```
/// use thales::numerical::{SecantMethod, NumericalConfig};
/// use thales::ast::{Equation, Expression, Variable};
///
/// // Solve x² = 2  (find √2)
/// let equation = Equation::new(
///     "sqrt2",
///     Expression::Power(
///         Box::new(Expression::Variable(Variable::new("x"))),
///         Box::new(Expression::Integer(2)),
///     ),
///     Expression::Integer(2),
/// );
///
/// let solver = SecantMethod::with_default_config();
/// let (solution, _path) = solver
///     .solve(&equation, &Variable::new("x"), (1.0, 2.0))
///     .unwrap();
///
/// assert!((solution.value - std::f64::consts::SQRT_2).abs() < 1e-10);
/// assert!(solution.converged);
/// ```
#[derive(Debug)]
pub struct SecantMethod {
    config: NumericalConfig,
}

impl SecantMethod {
    /// Creates a new secant method solver with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Numerical configuration (tolerance, iterations, etc.)
    pub fn new(config: NumericalConfig) -> Self {
        Self { config }
    }

    /// Creates a new secant method solver with default configuration.
    pub fn with_default_config() -> Self {
        Self {
            config: NumericalConfig::default(),
        }
    }

    /// Find a root using the secant method.
    ///
    /// # Arguments
    ///
    /// * `equation` - The equation to solve (interpreted as `lhs - rhs = 0`)
    /// * `variable` - The variable to solve for
    /// * `initial_points` - Two distinct starting points `(x0, x1)`; they do
    ///   not need to bracket the root
    ///
    /// # Errors
    ///
    /// * [`NumericalError::Unstable`] – denominator too close to zero
    /// * [`NumericalError::NoConvergence`] – max iterations reached
    /// * [`NumericalError::EvaluationFailed`] – function evaluation failed
    pub fn solve(
        &self,
        equation: &Equation,
        variable: &Variable,
        initial_points: (f64, f64),
    ) -> NumericalResult<(NumericalSolution, ResolutionPath)> {
        let (f, eval) = secant_make_eval(equation, variable);
        let (mut x_prev, mut x_curr) = initial_points;
        let mut f_prev = eval(x_prev)?;
        let mut f_curr = eval(x_curr)?;

        let mut path = ResolutionPathBuilder::new(f);
        path = path.step(
            Operation::NumericalApproximation,
            format!(
                "Starting secant: x0={x_prev}, x1={x_curr}: \
                 f(x0)={f_prev:.6e}, f(x1)={f_curr:.6e}"
            ),
            Expression::Float(x_curr),
        );

        let (solution, x_final) = secant_iterate(
            &eval,
            &mut path,
            &mut x_prev,
            &mut x_curr,
            &mut f_prev,
            &mut f_curr,
            self.config.max_iterations,
            self.config.tolerance,
        )?;

        path = path.step(
            Operation::NumericalApproximation,
            format!("Converged: x={x_final:.15}, |f(x)|={:.6e}", f_curr.abs()),
            Expression::Float(x_final),
        );

        Ok((solution, path.finish(Expression::Float(x_final))))
    }
}

// ============================================================================
// Secant method helpers
// ============================================================================

/// Build the residual expression `f(x) = lhs - rhs` and a point-evaluator
/// closure for the secant method.
fn secant_make_eval(
    equation: &Equation,
    variable: &Variable,
) -> (Expression, impl Fn(f64) -> NumericalResult<f64>) {
    let f = Expression::Binary(
        crate::ast::BinaryOp::Sub,
        Box::new(equation.left.clone()),
        Box::new(equation.right.clone()),
    );
    let f_owned = f.clone();
    let var_name = variable.name.clone();
    let eval = move |xv: f64| -> NumericalResult<f64> {
        let mut vars = HashMap::new();
        vars.insert(var_name.clone(), xv);
        f_owned.evaluate(&vars).ok_or_else(|| {
            NumericalError::EvaluationFailed(format!("Failed to evaluate at x = {xv}"))
        })
    };
    (f, eval)
}

/// Compute the next secant iterate.
///
/// Returns `Ok(x_next)` or `Err(NumericalError::Unstable)` when the
/// denominator is too small (near-zero slope between the two points).
fn secant_step(x_prev: f64, x_curr: f64, f_prev: f64, f_curr: f64) -> NumericalResult<f64> {
    let denom = f_curr - f_prev;
    if denom.abs() < f64::EPSILON * 10.0 {
        return Err(NumericalError::Unstable);
    }
    Ok(x_curr - f_curr * (x_curr - x_prev) / denom)
}

/// Run the secant iteration loop, updating path and state in place.
///
/// Separated from `SecantMethod::solve` so that `solve` stays under 80 lines.
#[allow(clippy::too_many_arguments)]
fn secant_iterate(
    eval: &impl Fn(f64) -> NumericalResult<f64>,
    path: &mut ResolutionPathBuilder,
    x_prev: &mut f64,
    x_curr: &mut f64,
    f_prev: &mut f64,
    f_curr: &mut f64,
    max_iterations: usize,
    tolerance: f64,
) -> NumericalResult<(NumericalSolution, f64)> {
    let mut iterations = 0_usize;
    let mut converged = false;

    for i in 0..max_iterations {
        iterations = i + 1;

        if f_curr.abs() < tolerance {
            converged = true;
            break;
        }

        let x_next = secant_step(*x_prev, *x_curr, *f_prev, *f_curr)?;

        if i % 10 == 0 {
            *path = (*path).clone().step(
                Operation::NumericalApproximation,
                format!(
                    "Iter {iterations}: x={x_next:.10}, f(x_curr)={:.6e}",
                    f_curr
                ),
                Expression::Float(x_next),
            );
        }

        let f_next = eval(x_next)?;
        *x_prev = *x_curr;
        *f_prev = *f_curr;
        *x_curr = x_next;
        *f_curr = f_next;

        if (*x_curr - *x_prev).abs() < tolerance {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(NumericalError::NoConvergence);
    }

    let solution = NumericalSolution {
        value: *x_curr,
        iterations,
        residual: f_curr.abs(),
        converged,
    };
    Ok((solution, *x_curr))
}
