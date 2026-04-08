//! Nonlinear system solver implementations.
//!
//! Provides Newton-Raphson, Broyden, and fixed-point iteration methods
//! for solving systems of nonlinear equations, along with supporting
//! types for configuration, results, convergence diagnostics, and
//! Jacobian validation.

use std::collections::HashMap;
use std::fmt;

use crate::ast::{BinaryOp, Equation, Expression, Variable};

// ============================================================================
// Nonlinear System Solver
// ============================================================================

/// Errors that can occur during nonlinear system solving.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum NonlinearSystemSolverError {
    /// System did not converge within max iterations.
    NoConvergence {
        /// Number of iterations attempted.
        iterations: usize,
        /// Final residual norm.
        final_residual: f64,
    },
    /// Jacobian matrix is singular or near-singular.
    SingularJacobian {
        /// Condition number estimate (if available).
        condition_estimate: Option<f64>,
    },
    /// Dimension mismatch between equations and variables.
    DimensionMismatch {
        /// Number of equations.
        num_equations: usize,
        /// Number of variables.
        num_variables: usize,
    },
    /// Failed to evaluate function at a point.
    EvaluationFailed {
        /// Point at which evaluation failed.
        point: Vec<f64>,
        /// Reason for failure.
        reason: String,
    },
    /// Invalid configuration.
    InvalidConfig(String),
    /// Differentiation failed when computing Jacobian.
    DifferentiationFailed(String),
}

impl fmt::Display for NonlinearSystemSolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoConvergence {
                iterations,
                final_residual,
            } => {
                write!(
                    f,
                    "No convergence after {} iterations (residual: {:.2e})",
                    iterations, final_residual
                )
            }
            Self::SingularJacobian { condition_estimate } => {
                if let Some(cond) = condition_estimate {
                    write!(f, "Singular Jacobian (condition ~{:.2e})", cond)
                } else {
                    write!(f, "Singular Jacobian")
                }
            }
            Self::DimensionMismatch {
                num_equations,
                num_variables,
            } => {
                write!(
                    f,
                    "Dimension mismatch: {} equations, {} variables",
                    num_equations, num_variables
                )
            }
            Self::EvaluationFailed { point, reason } => {
                write!(f, "Evaluation failed at {:?}: {}", point, reason)
            }
            Self::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            Self::DifferentiationFailed(msg) => write!(f, "Differentiation failed: {}", msg),
        }
    }
}

impl std::error::Error for NonlinearSystemSolverError {}

/// Configuration for nonlinear system solvers.
#[derive(Debug, Clone)]
pub struct NonlinearSystemConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Tolerance for residual norm.
    pub tolerance: f64,
    /// Tolerance for step size.
    pub step_tolerance: f64,
    /// Damping factor for damped Newton (1.0 = no damping).
    pub damping_factor: f64,
    /// Whether to use line search for step size.
    pub use_line_search: bool,
    /// Finite difference epsilon for Jacobian validation.
    pub finite_diff_epsilon: f64,
    /// Minimum step size for line search.
    pub min_step_size: f64,
    /// Regularization parameter for near-singular Jacobians.
    pub regularization: f64,
}

impl Default for NonlinearSystemConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-10,
            step_tolerance: 1e-12,
            damping_factor: 1.0,
            use_line_search: false,
            finite_diff_epsilon: 1e-8,
            min_step_size: 1e-10,
            regularization: 1e-12,
        }
    }
}

impl NonlinearSystemConfig {
    /// Create a config with damped Newton settings.
    pub fn damped() -> Self {
        Self {
            damping_factor: 0.5,
            use_line_search: true,
            ..Default::default()
        }
    }

    /// Create a config for Broyden's method.
    pub fn for_broyden() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 1e-8,
            ..Default::default()
        }
    }
}

/// Result of nonlinear system solving.
#[derive(Debug, Clone)]
pub struct NonlinearSystemSolverResult {
    /// Solution vector.
    pub solution: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final residual norm.
    pub final_residual: f64,
    /// Convergence history (residual norm at each iteration).
    pub convergence_history: Vec<f64>,
    /// Whether the solver converged.
    pub converged: bool,
    /// Method used for solving.
    pub method: String,
    /// Variable names in order.
    pub variable_names: Vec<String>,
}

impl NonlinearSystemSolverResult {
    /// Get solution as a map from variable name to value.
    pub fn as_map(&self) -> HashMap<String, f64> {
        self.variable_names
            .iter()
            .zip(self.solution.iter())
            .map(|(name, &val)| (name.clone(), val))
            .collect()
    }

    /// Estimate convergence rate (linear rate estimate from last few iterations).
    pub fn convergence_rate(&self) -> Option<f64> {
        if self.convergence_history.len() < 3 {
            return None;
        }

        let n = self.convergence_history.len();
        let r1 = self.convergence_history[n - 2];
        let r2 = self.convergence_history[n - 1];
        let r0 = self.convergence_history[n - 3];

        if r0 > 1e-15 && r1 > 1e-15 {
            // Linear convergence rate estimate: r_{n+1}/r_n
            Some(r2 / r1)
        } else {
            None
        }
    }
}

/// Convergence diagnostics for analyzing solver behavior.
#[derive(Debug, Clone)]
pub struct ConvergenceDiagnostics {
    /// Residual history.
    pub residual_history: Vec<f64>,
    /// Step size history.
    pub step_history: Vec<f64>,
    /// Estimated convergence rate.
    pub estimated_rate: Option<f64>,
    /// Detected behavior.
    pub behavior: ConvergenceBehavior,
}

/// Type of convergence behavior observed.
#[derive(Debug, Clone, PartialEq)]
pub enum ConvergenceBehavior {
    /// Fast quadratic convergence (Newton).
    Quadratic,
    /// Linear convergence.
    Linear,
    /// Sublinear convergence.
    Sublinear,
    /// Oscillating.
    Oscillating,
    /// Diverging.
    Diverging,
    /// Stalled.
    Stalled,
}

impl ConvergenceDiagnostics {
    /// Analyze convergence from history.
    pub fn analyze(residuals: &[f64], steps: &[f64]) -> Self {
        let behavior = if residuals.len() < 3 {
            ConvergenceBehavior::Linear
        } else {
            let n = residuals.len();
            let r1 = residuals[n - 2];
            let r2 = residuals[n - 1];
            let r0 = residuals[n - 3];

            if r2 > r1 * 1.1 {
                ConvergenceBehavior::Diverging
            } else if r2 > r1 * 0.99 {
                ConvergenceBehavior::Stalled
            } else if r0 > 1e-15 && r1 > 1e-15 {
                let rate1 = r1 / r0;
                let rate2 = r2 / r1;
                if rate2 < rate1 * rate1 * 2.0 {
                    ConvergenceBehavior::Quadratic
                } else if rate2 < 0.9 {
                    ConvergenceBehavior::Linear
                } else {
                    ConvergenceBehavior::Sublinear
                }
            } else {
                ConvergenceBehavior::Linear
            }
        };

        let estimated_rate = if residuals.len() >= 2 {
            let n = residuals.len();
            if residuals[n - 2] > 1e-15 {
                Some(residuals[n - 1] / residuals[n - 2])
            } else {
                None
            }
        } else {
            None
        };

        Self {
            residual_history: residuals.to_vec(),
            step_history: steps.to_vec(),
            estimated_rate,
            behavior,
        }
    }
}

/// A system of nonlinear equations F(x) = 0.
#[derive(Debug, Clone)]
pub struct NonlinearSystem {
    /// The equations F_i(x) (each should equal 0).
    pub equations: Vec<Expression>,
    /// The variables in order.
    pub variables: Vec<Variable>,
}

impl NonlinearSystem {
    /// Create a new nonlinear system.
    pub fn new(equations: Vec<Expression>, variables: Vec<Variable>) -> Self {
        Self {
            equations,
            variables,
        }
    }

    /// Create from Equation types (converts to F(x) = 0 form).
    pub fn from_equations(equations: Vec<Equation>, variables: Vec<Variable>) -> Self {
        // Convert each equation to the form: left - right = 0
        let exprs: Vec<Expression> = equations
            .into_iter()
            .map(|eq| Expression::Binary(BinaryOp::Sub, Box::new(eq.left), Box::new(eq.right)))
            .collect();
        Self::new(exprs, variables)
    }

    /// Number of equations.
    pub fn num_equations(&self) -> usize {
        self.equations.len()
    }

    /// Number of variables.
    pub fn num_variables(&self) -> usize {
        self.variables.len()
    }

    /// Check if the system is square (n equations, n variables).
    pub fn is_square(&self) -> bool {
        self.num_equations() == self.num_variables()
    }

    /// Evaluate all equations at a point.
    pub fn evaluate(&self, point: &[f64]) -> Result<Vec<f64>, NonlinearSystemSolverError> {
        if point.len() != self.variables.len() {
            return Err(NonlinearSystemSolverError::DimensionMismatch {
                num_equations: self.equations.len(),
                num_variables: point.len(),
            });
        }

        // Build variable map
        let var_map: HashMap<String, f64> = self
            .variables
            .iter()
            .zip(point.iter())
            .map(|(v, &val)| (v.name.clone(), val))
            .collect();

        // Evaluate each equation
        let mut result = Vec::with_capacity(self.equations.len());
        for (i, eq) in self.equations.iter().enumerate() {
            match eq.evaluate(&var_map) {
                Some(val) => result.push(val),
                None => {
                    return Err(NonlinearSystemSolverError::EvaluationFailed {
                        point: point.to_vec(),
                        reason: format!("Could not evaluate equation {}", i),
                    })
                }
            }
        }

        Ok(result)
    }

    /// Compute the Jacobian matrix symbolically.
    /// Returns J[i][j] = ∂F_i/∂x_j
    pub fn jacobian(&self) -> Result<Vec<Vec<Expression>>, NonlinearSystemSolverError> {
        let mut jacobian = Vec::with_capacity(self.equations.len());

        for eq in &self.equations {
            let mut row = Vec::with_capacity(self.variables.len());
            for var in &self.variables {
                // Use Expression::differentiate method
                let deriv = eq.differentiate(&var.name);
                row.push(deriv);
            }
            jacobian.push(row);
        }

        Ok(jacobian)
    }

    /// Evaluate the Jacobian matrix at a point.
    pub fn evaluate_jacobian(
        &self,
        point: &[f64],
    ) -> Result<Vec<Vec<f64>>, NonlinearSystemSolverError> {
        let symbolic_jacobian = self.jacobian()?;

        // Build variable map
        let var_map: HashMap<String, f64> = self
            .variables
            .iter()
            .zip(point.iter())
            .map(|(v, &val)| (v.name.clone(), val))
            .collect();

        let mut result = Vec::with_capacity(symbolic_jacobian.len());
        for (i, row) in symbolic_jacobian.iter().enumerate() {
            let mut eval_row = Vec::with_capacity(row.len());
            for (j, expr) in row.iter().enumerate() {
                match expr.evaluate(&var_map) {
                    Some(val) => eval_row.push(val),
                    None => {
                        return Err(NonlinearSystemSolverError::EvaluationFailed {
                            point: point.to_vec(),
                            reason: format!("Could not evaluate J[{}][{}]", i, j),
                        })
                    }
                }
            }
            result.push(eval_row);
        }

        Ok(result)
    }

    /// Variable names as strings.
    pub fn variable_names(&self) -> Vec<String> {
        self.variables.iter().map(|v| v.name.clone()).collect()
    }
}

/// Calculate L2 norm of residuals.
pub fn residual_norm(residuals: &[f64]) -> f64 {
    residuals.iter().map(|r| r * r).sum::<f64>().sqrt()
}

/// Solve a linear system Ax = b using LU decomposition with partial pivoting.
pub fn solve_linear_system_lu(
    matrix: &[Vec<f64>],
    rhs: &[f64],
) -> Result<Vec<f64>, NonlinearSystemSolverError> {
    let n = matrix.len();
    if n == 0 || rhs.len() != n {
        return Err(NonlinearSystemSolverError::DimensionMismatch {
            num_equations: n,
            num_variables: rhs.len(),
        });
    }

    // Copy matrix for LU decomposition
    let mut lu: Vec<Vec<f64>> = matrix.to_vec();
    let mut p: Vec<usize> = (0..n).collect(); // Permutation vector
    let b = rhs.to_vec();

    // LU decomposition with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_val = lu[k][k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            if lu[i][k].abs() > max_val {
                max_val = lu[i][k].abs();
                max_row = i;
            }
        }

        if max_val < 1e-15 {
            return Err(NonlinearSystemSolverError::SingularJacobian {
                condition_estimate: None,
            });
        }

        // Swap rows
        if max_row != k {
            lu.swap(k, max_row);
            p.swap(k, max_row);
        }

        // Elimination
        for i in (k + 1)..n {
            lu[i][k] /= lu[k][k];
            for j in (k + 1)..n {
                lu[i][j] -= lu[i][k] * lu[k][j];
            }
        }
    }

    // Apply permutation to b
    let mut pb = vec![0.0; n];
    for i in 0..n {
        pb[i] = b[p[i]];
    }

    // Forward substitution (Ly = Pb)
    let mut y = pb;
    for i in 1..n {
        for j in 0..i {
            y[i] -= lu[i][j] * y[j];
        }
    }

    // Back substitution (Ux = y)
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = y[i];
        for j in (i + 1)..n {
            x[i] -= lu[i][j] * x[j];
        }
        x[i] /= lu[i][i];
    }

    Ok(x)
}

/// Newton-Raphson solver for nonlinear systems.
pub fn newton_raphson_system(
    system: &NonlinearSystem,
    initial_guess: &[f64],
    config: &NonlinearSystemConfig,
) -> Result<NonlinearSystemSolverResult, NonlinearSystemSolverError> {
    if !system.is_square() {
        return Err(NonlinearSystemSolverError::DimensionMismatch {
            num_equations: system.num_equations(),
            num_variables: system.num_variables(),
        });
    }

    let n = system.num_variables();
    let mut x = initial_guess.to_vec();
    let mut convergence_history = Vec::new();

    for iter in 0..config.max_iterations {
        // Evaluate F(x)
        let f = system.evaluate(&x)?;
        let residual = residual_norm(&f);
        convergence_history.push(residual);

        // Check convergence
        if residual < config.tolerance {
            return Ok(NonlinearSystemSolverResult {
                solution: x,
                iterations: iter,
                final_residual: residual,
                convergence_history,
                converged: true,
                method: "Newton-Raphson".to_string(),
                variable_names: system.variable_names(),
            });
        }

        // Compute Jacobian
        let jacobian = system.evaluate_jacobian(&x)?;

        // Apply regularization if needed
        let mut reg_jacobian = jacobian.clone();
        for i in 0..n {
            reg_jacobian[i][i] += config.regularization;
        }

        // Solve J * delta = -F
        let neg_f: Vec<f64> = f.iter().map(|v| -v).collect();
        let delta = solve_linear_system_lu(&reg_jacobian, &neg_f)?;

        // Check step size
        let step_norm: f64 = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
        if step_norm < config.step_tolerance {
            return Ok(NonlinearSystemSolverResult {
                solution: x,
                iterations: iter,
                final_residual: residual,
                convergence_history,
                converged: true,
                method: "Newton-Raphson".to_string(),
                variable_names: system.variable_names(),
            });
        }

        // Apply damping or line search
        let alpha = if config.use_line_search {
            // Simple backtracking line search
            let mut alpha = 1.0;
            let c = 0.5;
            let rho = 0.5;

            for _ in 0..20 {
                let x_new: Vec<f64> = x
                    .iter()
                    .zip(delta.iter())
                    .map(|(xi, di)| xi + alpha * di)
                    .collect();

                if let Ok(f_new) = system.evaluate(&x_new) {
                    let new_residual = residual_norm(&f_new);
                    if new_residual < residual * (1.0 - c * alpha) {
                        break;
                    }
                }

                alpha *= rho;
                if alpha < config.min_step_size {
                    break;
                }
            }
            alpha
        } else {
            config.damping_factor
        };

        // Update x
        for i in 0..n {
            x[i] += alpha * delta[i];
        }
    }

    // Did not converge
    let f = system.evaluate(&x)?;
    let final_residual = residual_norm(&f);

    Err(NonlinearSystemSolverError::NoConvergence {
        iterations: config.max_iterations,
        final_residual,
    })
}

/// Fixed-point iteration solver for nonlinear systems.
/// Requires the system to be expressed as x = G(x).
pub fn fixed_point_system(
    system: &NonlinearSystem,
    initial_guess: &[f64],
    config: &NonlinearSystemConfig,
) -> Result<NonlinearSystemSolverResult, NonlinearSystemSolverError> {
    if !system.is_square() {
        return Err(NonlinearSystemSolverError::DimensionMismatch {
            num_equations: system.num_equations(),
            num_variables: system.num_variables(),
        });
    }

    let n = system.num_variables();
    let mut x = initial_guess.to_vec();
    let mut convergence_history = Vec::new();

    for iter in 0..config.max_iterations {
        // Evaluate F(x) - the equations should be in fixed-point form x = G(x)
        // so F(x) = G(x) - x, and we want x_new = G(x) = x + F(x)
        let f = system.evaluate(&x)?;
        let residual = residual_norm(&f);
        convergence_history.push(residual);

        // Check convergence
        if residual < config.tolerance {
            return Ok(NonlinearSystemSolverResult {
                solution: x,
                iterations: iter,
                final_residual: residual,
                convergence_history,
                converged: true,
                method: "Fixed-Point".to_string(),
                variable_names: system.variable_names(),
            });
        }

        // Apply fixed-point update with damping
        let mut x_new = vec![0.0; n];
        for i in 0..n {
            // x_new = x - damping * F(x) (for systems in standard form)
            x_new[i] = x[i] - config.damping_factor * f[i];
        }

        // Check for divergence
        if let Ok(f_new) = system.evaluate(&x_new) {
            let new_residual = residual_norm(&f_new);
            if new_residual > residual * 10.0 {
                return Err(NonlinearSystemSolverError::NoConvergence {
                    iterations: iter,
                    final_residual: new_residual,
                });
            }
        }

        x = x_new;
    }

    let f = system.evaluate(&x)?;
    let final_residual = residual_norm(&f);

    Err(NonlinearSystemSolverError::NoConvergence {
        iterations: config.max_iterations,
        final_residual,
    })
}

/// Broyden's method - a quasi-Newton method that avoids Jacobian recomputation.
pub fn broyden_system(
    system: &NonlinearSystem,
    initial_guess: &[f64],
    config: &NonlinearSystemConfig,
) -> Result<NonlinearSystemSolverResult, NonlinearSystemSolverError> {
    if !system.is_square() {
        return Err(NonlinearSystemSolverError::DimensionMismatch {
            num_equations: system.num_equations(),
            num_variables: system.num_variables(),
        });
    }

    let n = system.num_variables();
    let mut x = initial_guess.to_vec();
    let mut convergence_history = Vec::new();

    // Initial Jacobian
    let mut b = system.evaluate_jacobian(&x)?;

    // Initial function evaluation
    let mut f = system.evaluate(&x)?;
    let mut residual = residual_norm(&f);
    convergence_history.push(residual);

    for iter in 0..config.max_iterations {
        if residual < config.tolerance {
            return Ok(NonlinearSystemSolverResult {
                solution: x,
                iterations: iter,
                final_residual: residual,
                convergence_history,
                converged: true,
                method: "Broyden".to_string(),
                variable_names: system.variable_names(),
            });
        }

        // Apply regularization
        let mut reg_b = b.clone();
        for i in 0..n {
            reg_b[i][i] += config.regularization;
        }

        // Solve B * s = -F
        let neg_f: Vec<f64> = f.iter().map(|v| -v).collect();
        let s = solve_linear_system_lu(&reg_b, &neg_f)?;

        // Update x
        let mut x_new = x.clone();
        for i in 0..n {
            x_new[i] += config.damping_factor * s[i];
        }

        // Evaluate at new point
        let f_new = system.evaluate(&x_new)?;
        let new_residual = residual_norm(&f_new);

        // Compute y = F(x_new) - F(x)
        let y: Vec<f64> = f_new.iter().zip(f.iter()).map(|(a, b)| a - b).collect();

        // Broyden update: B_new = B + (y - B*s) * s^T / (s^T * s)
        // Compute B*s
        let mut bs = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                bs[i] += b[i][j] * s[j];
            }
        }

        // Compute y - B*s
        let diff: Vec<f64> = y.iter().zip(bs.iter()).map(|(a, b)| a - b).collect();

        // Compute s^T * s
        let s_dot_s: f64 = s.iter().map(|si| si * si).sum();

        if s_dot_s > 1e-15 {
            // Update B: B_new[i][j] = B[i][j] + diff[i] * s[j] / s_dot_s
            for i in 0..n {
                for j in 0..n {
                    b[i][j] += diff[i] * s[j] / s_dot_s;
                }
            }
        }

        x = x_new;
        f = f_new;
        residual = new_residual;
        convergence_history.push(residual);
    }

    Err(NonlinearSystemSolverError::NoConvergence {
        iterations: config.max_iterations,
        final_residual: residual,
    })
}

/// Trait for nonlinear system solvers.
pub trait NonlinearSystemSolver {
    /// Solve the nonlinear system.
    fn solve(
        &self,
        system: &NonlinearSystem,
        initial_guess: &[f64],
        config: &NonlinearSystemConfig,
    ) -> Result<NonlinearSystemSolverResult, NonlinearSystemSolverError>;

    /// Method name.
    fn method_name(&self) -> &str;
}

/// Newton-Raphson solver implementation.
pub struct NewtonRaphsonSolver;

impl NonlinearSystemSolver for NewtonRaphsonSolver {
    fn solve(
        &self,
        system: &NonlinearSystem,
        initial_guess: &[f64],
        config: &NonlinearSystemConfig,
    ) -> Result<NonlinearSystemSolverResult, NonlinearSystemSolverError> {
        newton_raphson_system(system, initial_guess, config)
    }

    fn method_name(&self) -> &str {
        "Newton-Raphson"
    }
}

/// Broyden solver implementation.
pub struct BroydenSolver;

impl NonlinearSystemSolver for BroydenSolver {
    fn solve(
        &self,
        system: &NonlinearSystem,
        initial_guess: &[f64],
        config: &NonlinearSystemConfig,
    ) -> Result<NonlinearSystemSolverResult, NonlinearSystemSolverError> {
        broyden_system(system, initial_guess, config)
    }

    fn method_name(&self) -> &str {
        "Broyden"
    }
}

/// Fixed-point solver implementation.
pub struct FixedPointSolver;

impl NonlinearSystemSolver for FixedPointSolver {
    fn solve(
        &self,
        system: &NonlinearSystem,
        initial_guess: &[f64],
        config: &NonlinearSystemConfig,
    ) -> Result<NonlinearSystemSolverResult, NonlinearSystemSolverError> {
        fixed_point_system(system, initial_guess, config)
    }

    fn method_name(&self) -> &str {
        "Fixed-Point"
    }
}

/// Smart nonlinear system solver that selects method based on system properties.
pub struct SmartNonlinearSystemSolver {
    /// Configuration override.
    config: Option<NonlinearSystemConfig>,
}

impl SmartNonlinearSystemSolver {
    /// Create a new smart solver.
    pub fn new() -> Self {
        Self { config: None }
    }

    /// Create with specific configuration.
    pub fn with_config(config: NonlinearSystemConfig) -> Self {
        Self {
            config: Some(config),
        }
    }

    /// Solve using automatic method selection.
    pub fn solve(
        &self,
        system: &NonlinearSystem,
        initial_guess: &[f64],
    ) -> Result<NonlinearSystemSolverResult, NonlinearSystemSolverError> {
        let config = self.config.clone().unwrap_or_default();

        // Try Newton-Raphson first (fastest when it works)
        match newton_raphson_system(system, initial_guess, &config) {
            Ok(result) if result.converged => return Ok(result),
            _ => {}
        }

        // Fall back to Broyden (more robust, doesn't recompute Jacobian)
        let broyden_config = NonlinearSystemConfig {
            max_iterations: config.max_iterations * 2,
            damping_factor: 0.8,
            ..config.clone()
        };

        match broyden_system(system, initial_guess, &broyden_config) {
            Ok(result) if result.converged => return Ok(result),
            _ => {}
        }

        // Try damped Newton with line search
        let damped_config = NonlinearSystemConfig {
            damping_factor: 0.5,
            use_line_search: true,
            max_iterations: config.max_iterations * 2,
            ..config
        };

        newton_raphson_system(system, initial_guess, &damped_config)
    }

    /// Find all solutions by trying multiple initial guesses.
    pub fn find_all_solutions(
        &self,
        system: &NonlinearSystem,
        initial_guesses: &[Vec<f64>],
    ) -> Vec<NonlinearSystemSolverResult> {
        let config = self.config.clone().unwrap_or_default();
        let mut solutions = Vec::new();
        let tolerance = config.tolerance * 100.0; // Tolerance for considering solutions distinct

        for guess in initial_guesses {
            if let Ok(result) = self.solve(system, guess) {
                if result.converged {
                    // Check if this is a new solution
                    let is_new = solutions
                        .iter()
                        .all(|existing: &NonlinearSystemSolverResult| {
                            let diff: f64 = existing
                                .solution
                                .iter()
                                .zip(result.solution.iter())
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f64>()
                                .sqrt();
                            diff > tolerance
                        });

                    if is_new {
                        solutions.push(result);
                    }
                }
            }
        }

        solutions
    }
}

impl Default for SmartNonlinearSystemSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate the Jacobian using finite differences.
pub fn validate_jacobian(
    system: &NonlinearSystem,
    point: &[f64],
    epsilon: f64,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>, f64), NonlinearSystemSolverError> {
    let analytic = system.evaluate_jacobian(point)?;
    let n = system.num_variables();
    let m = system.num_equations();

    // Compute finite difference Jacobian
    let f0 = system.evaluate(point)?;
    let mut numeric = vec![vec![0.0; n]; m];

    for j in 0..n {
        let mut point_plus = point.to_vec();
        point_plus[j] += epsilon;

        let f_plus = system.evaluate(&point_plus)?;

        for i in 0..m {
            numeric[i][j] = (f_plus[i] - f0[i]) / epsilon;
        }
    }

    // Compute max absolute difference
    let mut max_diff = 0.0;
    for i in 0..m {
        for j in 0..n {
            let diff = (analytic[i][j] - numeric[i][j]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    Ok((analytic, numeric, max_diff))
}

#[cfg(test)]
mod nonlinear_tests {
    use super::*;

    fn make_circle_line_system() -> NonlinearSystem {
        // x^2 + y^2 - 1 = 0 (unit circle)
        // x - y = 0 (diagonal line)
        let x = Variable::new("x");
        let y = Variable::new("y");

        // x^2 + y^2 - 1
        let eq1 = Expression::Binary(
            BinaryOp::Sub,
            Box::new(Expression::Binary(
                BinaryOp::Add,
                Box::new(Expression::Power(
                    Box::new(Expression::Variable(x.clone())),
                    Box::new(Expression::Integer(2)),
                )),
                Box::new(Expression::Power(
                    Box::new(Expression::Variable(y.clone())),
                    Box::new(Expression::Integer(2)),
                )),
            )),
            Box::new(Expression::Integer(1)),
        );

        // x - y
        let eq2 = Expression::Binary(
            BinaryOp::Sub,
            Box::new(Expression::Variable(x.clone())),
            Box::new(Expression::Variable(y.clone())),
        );

        NonlinearSystem::new(vec![eq1, eq2], vec![x, y])
    }

    fn make_hyperbola_line_system() -> NonlinearSystem {
        // x*y - 1 = 0 (hyperbola xy = 1)
        // x + y - 3 = 0 (line x + y = 3)
        let x = Variable::new("x");
        let y = Variable::new("y");

        // x*y - 1
        let eq1 = Expression::Binary(
            BinaryOp::Sub,
            Box::new(Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Variable(x.clone())),
                Box::new(Expression::Variable(y.clone())),
            )),
            Box::new(Expression::Integer(1)),
        );

        // x + y - 3
        let eq2 = Expression::Binary(
            BinaryOp::Sub,
            Box::new(Expression::Binary(
                BinaryOp::Add,
                Box::new(Expression::Variable(x.clone())),
                Box::new(Expression::Variable(y.clone())),
            )),
            Box::new(Expression::Integer(3)),
        );

        NonlinearSystem::new(vec![eq1, eq2], vec![x, y])
    }

    #[test]
    fn test_nonlinear_system_creation() {
        let system = make_circle_line_system();
        assert_eq!(system.num_equations(), 2);
        assert_eq!(system.num_variables(), 2);
        assert!(system.is_square());
    }

    #[test]
    fn test_evaluate() {
        let system = make_circle_line_system();

        // At (1, 0): x^2 + y^2 - 1 = 0, x - y = 1
        let result = system.evaluate(&[1.0, 0.0]).unwrap();
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);

        // At (0.5, 0.5): x^2 + y^2 - 1 = -0.5, x - y = 0
        let result = system.evaluate(&[0.5, 0.5]).unwrap();
        assert!((result[0] - (-0.5)).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_residual_norm() {
        let residuals = vec![3.0, 4.0];
        assert!((residual_norm(&residuals) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_linear_system_lu() {
        // Simple 2x2 system: x + y = 3, x - y = 1 => x = 2, y = 1
        let matrix = vec![vec![1.0, 1.0], vec![1.0, -1.0]];
        let rhs = vec![3.0, 1.0];

        let solution = solve_linear_system_lu(&matrix, &rhs).unwrap();
        assert!((solution[0] - 2.0).abs() < 1e-10);
        assert!((solution[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_newton_raphson_circle_line() {
        let system = make_circle_line_system();
        let config = NonlinearSystemConfig::default();

        // Starting from (0.5, 0.5), should converge to (√2/2, √2/2)
        let result = newton_raphson_system(&system, &[0.5, 0.5], &config).unwrap();

        assert!(result.converged);
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!((result.solution[0] - expected).abs() < 1e-8);
        assert!((result.solution[1] - expected).abs() < 1e-8);
    }

    #[test]
    fn test_newton_raphson_negative_solution() {
        let system = make_circle_line_system();
        let config = NonlinearSystemConfig::default();

        // Starting from (-0.5, -0.5), should converge to (-√2/2, -√2/2)
        let result = newton_raphson_system(&system, &[-0.5, -0.5], &config).unwrap();

        assert!(result.converged);
        let expected = -std::f64::consts::FRAC_1_SQRT_2;
        assert!((result.solution[0] - expected).abs() < 1e-8);
        assert!((result.solution[1] - expected).abs() < 1e-8);
    }

    #[test]
    fn test_hyperbola_line_solution() {
        let system = make_hyperbola_line_system();
        let config = NonlinearSystemConfig::default();

        // Starting from (1.5, 1.5), should converge to (1, 2) or (2, 1)
        let result = newton_raphson_system(&system, &[1.5, 1.5], &config).unwrap();

        assert!(result.converged);
        // Either (1, 2) or (2, 1)
        let x = result.solution[0];
        let y = result.solution[1];
        assert!((x * y - 1.0).abs() < 1e-8);
        assert!((x + y - 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_broyden_circle_line() {
        let system = make_circle_line_system();
        let config = NonlinearSystemConfig::for_broyden();

        let result = broyden_system(&system, &[0.5, 0.5], &config).unwrap();

        assert!(result.converged);
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!((result.solution[0] - expected).abs() < 1e-6);
        assert!((result.solution[1] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_jacobian_validation() {
        let system = make_circle_line_system();
        let point = vec![0.5, 0.5];
        let epsilon = 1e-6;

        let (analytic, _numeric, max_diff) = validate_jacobian(&system, &point, epsilon).unwrap();

        // Analytic Jacobian at (0.5, 0.5):
        // [2x, 2y] = [1.0, 1.0]
        // [1, -1] = [1.0, -1.0]
        assert!((analytic[0][0] - 1.0).abs() < 1e-8);
        assert!((analytic[0][1] - 1.0).abs() < 1e-8);
        assert!((analytic[1][0] - 1.0).abs() < 1e-8);
        assert!((analytic[1][1] - (-1.0)).abs() < 1e-8);

        // Numeric should match analytic closely
        assert!(max_diff < 1e-4);
    }

    #[test]
    fn test_smart_solver() {
        let system = make_circle_line_system();
        let solver = SmartNonlinearSystemSolver::new();

        let result = solver.solve(&system, &[0.5, 0.5]).unwrap();
        assert!(result.converged);
    }

    #[test]
    fn test_find_all_solutions() {
        let system = make_circle_line_system();
        let solver = SmartNonlinearSystemSolver::new();

        // Try multiple initial guesses to find both solutions
        let guesses = vec![
            vec![0.5, 0.5],
            vec![-0.5, -0.5],
            vec![1.0, 0.0],
            vec![-1.0, 0.0],
        ];

        let solutions = solver.find_all_solutions(&system, &guesses);

        // Should find 2 distinct solutions
        assert_eq!(solutions.len(), 2);

        // Verify both solutions
        let sqrt2_over_2 = std::f64::consts::FRAC_1_SQRT_2;
        let has_positive = solutions
            .iter()
            .any(|s| (s.solution[0] - sqrt2_over_2).abs() < 1e-6);
        let has_negative = solutions
            .iter()
            .any(|s| (s.solution[0] + sqrt2_over_2).abs() < 1e-6);
        assert!(has_positive);
        assert!(has_negative);
    }

    #[test]
    fn test_convergence_diagnostics() {
        let residuals = vec![1.0, 0.1, 0.01, 0.001];
        let steps = vec![1.0, 0.5, 0.25, 0.125];

        let diagnostics = ConvergenceDiagnostics::analyze(&residuals, &steps);

        assert_eq!(diagnostics.behavior, ConvergenceBehavior::Linear);
        assert!(diagnostics.estimated_rate.is_some());
        assert!((diagnostics.estimated_rate.unwrap() - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let system = make_circle_line_system();
        let config = NonlinearSystemConfig::default();

        // Wrong number of initial values
        let result = newton_raphson_system(&system, &[0.5], &config);
        assert!(matches!(
            result,
            Err(NonlinearSystemSolverError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_compare_newton_vs_broyden_iterations() {
        let system = make_hyperbola_line_system();

        // Use a better initial guess closer to the solution (2, 1)
        let initial_guess = [2.1, 0.9];

        let newton_config = NonlinearSystemConfig::default();
        let broyden_config = NonlinearSystemConfig {
            max_iterations: 200,
            tolerance: 1e-8,
            ..Default::default()
        };

        let newton_result = newton_raphson_system(&system, &initial_guess, &newton_config).unwrap();
        let broyden_result = broyden_system(&system, &initial_guess, &broyden_config).unwrap();

        // Both should converge
        assert!(newton_result.converged);
        assert!(broyden_result.converged);

        // Newton typically converges faster (fewer iterations)
        // But Broyden avoids Jacobian recomputation
        println!(
            "Newton iterations: {}, Broyden iterations: {}",
            newton_result.iterations, broyden_result.iterations
        );
    }

    #[test]
    fn test_solution_as_map() {
        let system = make_circle_line_system();
        let config = NonlinearSystemConfig::default();

        let result = newton_raphson_system(&system, &[0.5, 0.5], &config).unwrap();
        let map = result.as_map();

        assert!(map.contains_key("x"));
        assert!(map.contains_key("y"));
    }

    #[test]
    fn test_nonlinear_system_solver_trait() {
        let system = make_circle_line_system();
        let config = NonlinearSystemConfig::default();

        // Test with trait object
        let solvers: Vec<Box<dyn NonlinearSystemSolver>> =
            vec![Box::new(NewtonRaphsonSolver), Box::new(BroydenSolver)];

        for solver in solvers {
            let result = solver.solve(&system, &[0.5, 0.5], &config).unwrap();
            assert!(result.converged);
            println!("{}: {} iterations", solver.method_name(), result.iterations);
        }
    }
}
