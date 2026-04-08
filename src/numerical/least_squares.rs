//! Levenberg-Marquardt algorithm for nonlinear least squares and gradient descent.

use crate::ast::{BinaryOp, Equation, Expression, Variable};
use std::collections::HashMap;

use super::{NumericalConfig, NumericalError, NumericalResult};

/// Gradient descent optimizer for minimization problems.
#[derive(Debug)]
pub struct GradientDescent {
    config: NumericalConfig,
    learning_rate: f64,
}

impl GradientDescent {
    /// Creates a new gradient descent optimizer with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Numerical configuration (tolerance, iterations, etc.)
    /// * `learning_rate` - Step size for gradient descent (typically 0.001 to 0.1)
    pub fn new(config: NumericalConfig, learning_rate: f64) -> Self {
        Self {
            config,
            learning_rate,
        }
    }

    /// Minimize an expression with respect to the given variables.
    ///
    /// Uses symbolic differentiation to compute gradients and iteratively
    /// updates variable values in the direction of steepest descent.
    ///
    /// # Arguments
    ///
    /// * `expression` - The objective function to minimize
    /// * `variables` - Variables to optimize over
    ///
    /// # Returns
    ///
    /// A mapping from each variable to its optimized value, or an error
    /// if the method fails to converge.
    pub fn minimize(
        &self,
        expression: &Expression,
        variables: &[Variable],
    ) -> NumericalResult<HashMap<Variable, f64>> {
        if variables.is_empty() {
            return Err(NumericalError::Other(
                "No variables to optimize".to_string(),
            ));
        }

        // Compute symbolic derivatives for each variable
        let derivatives: Vec<Expression> = variables
            .iter()
            .map(|v| expression.differentiate(&v.name))
            .collect();

        // Initialize values from initial_guess or default to 1.0
        let mut values: HashMap<String, f64> = variables
            .iter()
            .map(|v| (v.name.clone(), self.config.initial_guess.unwrap_or(1.0)))
            .collect();

        let mut prev_value = f64::INFINITY;

        for _iteration in 0..self.config.max_iterations {
            // Evaluate objective function
            let current_value = expression.evaluate(&values).ok_or_else(|| {
                NumericalError::EvaluationFailed("Failed to evaluate objective".to_string())
            })?;

            // Check convergence
            if (prev_value - current_value).abs() < self.config.tolerance {
                return Ok(variables
                    .iter()
                    .map(|v| (v.clone(), values[&v.name]))
                    .collect());
            }
            prev_value = current_value;

            // Compute and apply gradient updates
            for (i, var) in variables.iter().enumerate() {
                let grad = derivatives[i].evaluate(&values).ok_or_else(|| {
                    NumericalError::EvaluationFailed(format!(
                        "Failed to evaluate gradient for {}",
                        var.name
                    ))
                })?;

                if !grad.is_finite() {
                    return Err(NumericalError::Unstable);
                }

                let current = values[&var.name];
                values.insert(var.name.clone(), current - self.learning_rate * grad);
            }
        }

        Err(NumericalError::NoConvergence)
    }
}

/// Levenberg-Marquardt algorithm for nonlinear least squares.
///
/// Minimizes the sum of squared residuals `Σ (lhs_i - rhs_i)²` by
/// iteratively solving a damped Gauss-Newton system. The damping
/// parameter λ interpolates between gradient descent (large λ) and
/// Gauss-Newton (small λ), providing robust convergence.
#[derive(Debug)]
pub struct LevenbergMarquardt {
    config: NumericalConfig,
}

impl LevenbergMarquardt {
    /// Creates a new Levenberg-Marquardt solver with custom configuration.
    pub fn new(config: NumericalConfig) -> Self {
        Self { config }
    }

    /// Creates a new Levenberg-Marquardt solver with default configuration.
    pub fn with_default_config() -> Self {
        Self {
            config: NumericalConfig::default(),
        }
    }

    /// Solve nonlinear least squares problem.
    ///
    /// Finds variable values that minimize `Σ (lhs_i - rhs_i)²` over
    /// the given equations, using a damped Gauss-Newton iteration.
    pub fn solve_least_squares(
        &self,
        equations: &[Equation],
        variables: &[Variable],
    ) -> NumericalResult<HashMap<Variable, f64>> {
        if equations.is_empty() || variables.is_empty() {
            return Err(NumericalError::Other(
                "Need at least one equation and one variable".to_string(),
            ));
        }

        let n = variables.len();

        // Build residual expressions: lhs - rhs for each equation
        let residuals: Vec<Expression> = equations
            .iter()
            .map(|eq| {
                Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(eq.left.clone()),
                    Box::new(eq.right.clone()),
                )
            })
            .collect();

        // Build symbolic Jacobian: J[i][j] = d(residual_i) / d(variable_j)
        let jacobian: Vec<Vec<Expression>> = residuals
            .iter()
            .map(|r| variables.iter().map(|v| r.differentiate(&v.name)).collect())
            .collect();

        // Initialize variable values
        let mut vals: HashMap<String, f64> = variables
            .iter()
            .map(|v| (v.name.clone(), self.config.initial_guess.unwrap_or(0.0)))
            .collect();

        let mut lambda = 1e-3_f64;

        for _iter in 0..self.config.max_iterations {
            // Evaluate residuals and cost
            let r_vals: Vec<f64> = residuals
                .iter()
                .map(|r| r.evaluate(&vals).unwrap_or(f64::NAN))
                .collect();

            if r_vals.iter().any(|v| !v.is_finite()) {
                return Err(NumericalError::Unstable);
            }

            let cost: f64 = r_vals.iter().map(|v| v * v).sum();
            if cost < self.config.tolerance * self.config.tolerance {
                return Ok(variables
                    .iter()
                    .map(|v| (v.clone(), vals[&v.name]))
                    .collect());
            }

            // Evaluate Jacobian matrix
            let j_vals: Vec<Vec<f64>> = jacobian
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|e| e.evaluate(&vals).unwrap_or(0.0))
                        .collect()
                })
                .collect();

            // Compute J^T * J and J^T * r
            let mut jtj = vec![vec![0.0; n]; n];
            let mut jtr = vec![0.0; n];

            for (i, j_row) in j_vals.iter().enumerate() {
                for j in 0..n {
                    jtr[j] += j_row[j] * r_vals[i];
                    for k in 0..n {
                        jtj[j][k] += j_row[j] * j_row[k];
                    }
                }
            }

            // Add damping: (J^T J + λI) δ = -J^T r
            for j in 0..n {
                jtj[j][j] += lambda;
            }

            // Solve the linear system using simple Gaussian elimination
            let delta = solve_linear_nxn(&jtj, &jtr.iter().map(|v| -v).collect::<Vec<_>>())
                .ok_or_else(|| NumericalError::Other("Singular matrix".to_string()))?;

            // Trial step
            let mut trial = vals.clone();
            for (j, var) in variables.iter().enumerate() {
                *trial.get_mut(&var.name).unwrap() += delta[j];
            }

            let trial_cost: f64 = residuals
                .iter()
                .map(|r| r.evaluate(&trial).unwrap_or(f64::NAN).powi(2))
                .sum();

            if trial_cost < cost {
                vals = trial;
                lambda *= 0.5;
            } else {
                lambda *= 2.0;
            }

            if delta.iter().map(|d| d.abs()).fold(0.0_f64, f64::max) < self.config.tolerance {
                return Ok(variables
                    .iter()
                    .map(|v| (v.clone(), vals[&v.name]))
                    .collect());
            }
        }

        Err(NumericalError::NoConvergence)
    }
}

/// Solve an NxN linear system Ax = b via Gaussian elimination with partial pivoting.
fn solve_linear_nxn(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(b[i]);
            r
        })
        .collect();

    for col in 0..n {
        // Partial pivoting
        let max_row =
            (col..n).max_by(|&a, &b| aug[a][col].abs().partial_cmp(&aug[b][col].abs()).unwrap())?;
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-15 {
            return None;
        }

        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                let val = aug[col][j];
                aug[row][j] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }
    Some(x)
}
