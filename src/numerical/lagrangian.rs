//! Lagrangian (constrained optimization) solver.
//!
//! Solves constrained optimization problems using the Lagrange multiplier method.
//!
//! # Method
//!
//! Given: minimize f(x₁,...,xₙ) subject to g₁ = 0, ..., gₘ = 0
//!
//! 1. Form the Lagrangian: L = f + λ₁g₁ + ... + λₘgₘ
//! 2. Differentiate symbolically: ∂L/∂xᵢ = 0 and ∂L/∂λⱼ = gⱼ = 0
//! 3. Solve the resulting (n + m) × (n + m) nonlinear system numerically
//!    via Newton-Raphson with a finite-difference Jacobian.
//!
//! # Example
//!
//! ```
//! use thales::ast::{Expression, Variable, BinaryOp};
//! use thales::numerical::lagrangian::LagrangianSolver;
//!
//! // Minimize x² + y² subject to x + y - 1 = 0
//! let x = Expression::Variable(Variable::new("x"));
//! let y = Expression::Variable(Variable::new("y"));
//!
//! // objective: x^2 + y^2
//! let objective = Expression::Binary(
//!     BinaryOp::Add,
//!     Box::new(Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(2)))),
//!     Box::new(Expression::Power(Box::new(y.clone()), Box::new(Expression::Integer(2)))),
//! );
//!
//! // constraint: x + y - 1 = 0
//! let constraint = Expression::Binary(
//!     BinaryOp::Sub,
//!     Box::new(Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(y))),
//!     Box::new(Expression::Integer(1)),
//! );
//!
//! let solver = LagrangianSolver::new();
//! let result = solver.solve(&objective, &[constraint], &[Variable::new("x"), Variable::new("y")]).unwrap();
//! assert!((result.objective_value - 0.5).abs() < 1e-6);
//! ```

use crate::ast::{BinaryOp, Expression, Variable};
use crate::numerical::bordered_hessian;
use crate::numerical::penalty;
use crate::solver::SolverError;
use std::collections::HashMap;

/// Maximum Newton-Raphson iterations for system solving.
const MAX_ITER: usize = 1000;

/// Step size for finite-difference Jacobian approximation.
const FD_H: f64 = 1e-7;

/// Classification of a critical point found by the Lagrangian solver.
///
/// Determined by the bordered Hessian second-order sufficiency conditions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationType {
    /// The critical point satisfies the second-order sufficient conditions for a local minimum.
    LocalMinimum,
    /// The critical point satisfies the second-order sufficient conditions for a local maximum.
    LocalMaximum,
    /// The bordered Hessian test indicates a saddle point.
    SaddlePoint,
    /// The bordered Hessian test is inconclusive (degenerate or not implemented for m > 1).
    Inconclusive,
}

/// Result of a Lagrangian constrained optimization.
#[derive(Debug, Clone)]
pub struct LagrangianResult {
    /// Solution point as (variable_name, value) pairs, in the same order as
    /// the `variables` slice passed to [`LagrangianSolver::solve`].
    pub point: Vec<(String, f64)>,
    /// Lagrange multiplier values (one per constraint).
    pub multipliers: Vec<f64>,
    /// Objective function value at the solution.
    pub objective_value: f64,
    /// Classification of the critical point via the bordered Hessian test.
    pub classification: OptimizationType,
}

/// Solver for constrained optimization via the Lagrange multiplier method.
///
/// Uses symbolic differentiation to form the Karush-Kuhn-Tucker (KKT) stationarity
/// conditions and then solves the resulting nonlinear system numerically with
/// Newton-Raphson iteration and a finite-difference Jacobian.
#[derive(Debug, Clone)]
pub struct LagrangianSolver {
    /// Convergence tolerance (both residual norm and step norm).
    pub tolerance: f64,
}

impl Default for LagrangianSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl LagrangianSolver {
    /// Create a solver with the default tolerance of 1e-8.
    #[must_use]
    pub fn new() -> Self {
        Self { tolerance: 1e-8 }
    }

    /// Create a solver with a custom tolerance.
    #[must_use]
    pub fn with_tolerance(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// Solve the constrained optimization problem.
    ///
    /// # Arguments
    ///
    /// * `objective`    – The function f to minimize (an `Expression` in the variables).
    /// * `constraints`  – Slice of constraint expressions gⱼ; the solver enforces gⱼ = 0.
    /// * `variables`    – The decision variables x₁, …, xₙ.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CannotSolve`] when:
    /// - The Lagrangian gradient equations cannot be evaluated at the initial point.
    /// - Newton-Raphson fails to converge within [`MAX_ITER`] iterations.
    pub fn solve(
        &self,
        objective: &Expression,
        constraints: &[Expression],
        variables: &[Variable],
    ) -> Result<LagrangianResult, SolverError> {
        let n = variables.len();
        let m = constraints.len();
        let total = n + m;

        // Build the Lagrangian gradient equations symbolically.
        let equations = self.build_kkt_equations(objective, constraints, variables);

        // All unknowns: [x₁, …, xₙ, λ₁, …, λₘ]
        let mut names: Vec<String> = variables.iter().map(|v| v.name.clone()).collect();
        for j in 0..m {
            names.push(format!("__lambda_{}", j));
        }

        // Generate multiple initial guesses to handle different constraint geometries.
        let initial_guesses = generate_initial_guesses(n, m);

        // Try each initial guess; collect all converged solutions and pick the
        // one with the smallest objective value (minimization).
        let mut best: Option<(Vec<f64>, f64)> = None;
        let mut last_err = None;

        for guess in &initial_guesses {
            let mut point = guess.clone();
            match newton_raphson_system(&equations, &names, &mut point, total, self.tolerance) {
                Ok(()) => {
                    if let Some(obj) = eval_at(objective, &names, &point) {
                        let dominated = best.as_ref().is_some_and(|(_, best_obj)| obj >= *best_obj);
                        if !dominated {
                            best = Some((point, obj));
                        }
                    }
                }
                Err(e) => {
                    last_err = Some(e);
                }
            }
        }

        let (point, obj_val) = best.ok_or_else(|| {
            last_err.unwrap_or_else(|| {
                SolverError::CannotSolve("All initial guesses failed to converge".to_string())
            })
        })?;

        // Extract results.
        let var_point: Vec<(String, f64)> = variables
            .iter()
            .zip(point.iter())
            .map(|(v, &val)| (v.name.clone(), val))
            .collect();

        let multipliers: Vec<f64> = point[n..].to_vec();

        let classification =
            self.classify_critical_point(objective, constraints, variables, &point, &names);

        Ok(LagrangianResult {
            point: var_point,
            multipliers,
            objective_value: obj_val,
            classification,
        })
    }

    /// Classify the critical point via the bordered Hessian second-order conditions.
    ///
    /// Supported for exactly one constraint (m = 1) with n ≥ 2 variables.
    /// Returns [`OptimizationType::Inconclusive`] for other cases.
    pub fn classify_critical_point(
        &self,
        objective: &Expression,
        constraints: &[Expression],
        variables: &[Variable],
        point: &[f64],
        names: &[String],
    ) -> OptimizationType {
        let n = variables.len();
        let m = constraints.len();
        if m != 1 || n < 2 {
            return OptimizationType::Inconclusive;
        }
        let lagrangian = bordered_hessian::build_lagrangian(objective, constraints);
        let Some(h) = bordered_hessian::lagrangian_hessian(&lagrangian, variables, names, point)
        else {
            return OptimizationType::Inconclusive;
        };
        let Some(grad_g) =
            bordered_hessian::constraint_gradient(&constraints[0], variables, names, point)
        else {
            return OptimizationType::Inconclusive;
        };
        bordered_hessian::classify_1c(&h, &grad_g, n)
    }

    /// Build the n + m KKT stationarity equations (as `Expression`s set equal to zero).
    ///
    /// The first n equations are ∂L/∂xᵢ = 0.
    /// The last m equations are the constraints gⱼ = 0.
    fn build_kkt_equations(
        &self,
        objective: &Expression,
        constraints: &[Expression],
        variables: &[Variable],
    ) -> Vec<Expression> {
        let m = constraints.len();
        let mut lagrangian = objective.clone();

        // L = f + Σ λⱼ · gⱼ
        for (j, g) in constraints.iter().enumerate() {
            let lambda = Expression::Variable(Variable::new(&format!("__lambda_{}", j)));
            let term = Expression::Binary(BinaryOp::Mul, Box::new(lambda), Box::new(g.clone()));
            lagrangian = Expression::Binary(BinaryOp::Add, Box::new(lagrangian), Box::new(term));
        }

        let mut equations = Vec::with_capacity(variables.len() + m);

        // ∂L/∂xᵢ = 0 for each decision variable.
        for v in variables {
            equations.push(lagrangian.differentiate(&v.name).simplify());
        }

        // ∂L/∂λⱼ = gⱼ = 0 for each constraint.
        for g in constraints {
            equations.push(g.clone());
        }

        equations
    }
}

// ============================================================================
// Unified public API
// ============================================================================

/// Solve a constrained optimization problem with automatic method selection.
///
/// Tries the Lagrange multiplier method first.  If that fails (e.g. singular
/// Jacobian or non-convergence), the quadratic penalty method is used as a
/// fallback and the result is wrapped in a [`LagrangianResult`] with
/// [`OptimizationType::Inconclusive`] classification.
///
/// # Arguments
///
/// * `objective`   – The function f to optimize.
/// * `constraints` – Constraint expressions gⱼ; the solver enforces gⱼ = 0.
/// * `variables`   – The decision variables.
///
/// # Errors
///
/// Returns [`SolverError::CannotSolve`] when both methods fail.
///
/// # Example
///
/// ```
/// use thales::ast::{Expression, Variable, BinaryOp};
/// use thales::numerical::lagrangian::optimize_constrained;
///
/// // Minimize x² + y² subject to x + y - 1 = 0
/// let x = Expression::Variable(Variable::new("x"));
/// let y = Expression::Variable(Variable::new("y"));
/// let objective = Expression::Binary(
///     BinaryOp::Add,
///     Box::new(Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(2)))),
///     Box::new(Expression::Power(Box::new(y.clone()), Box::new(Expression::Integer(2)))),
/// );
/// let constraint = Expression::Binary(
///     BinaryOp::Sub,
///     Box::new(Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(y))),
///     Box::new(Expression::Integer(1)),
/// );
/// let vars = vec![Variable::new("x"), Variable::new("y")];
/// let result = optimize_constrained(&objective, &[constraint], &vars).unwrap();
/// assert!((result.objective_value - 0.5).abs() < 1e-5);
/// ```
pub fn optimize_constrained(
    objective: &Expression,
    constraints: &[Expression],
    variables: &[Variable],
) -> Result<LagrangianResult, SolverError> {
    let solver = LagrangianSolver::new();
    match solver.solve(objective, constraints, variables) {
        Ok(result) => Ok(result),
        Err(_) => {
            let penalty_result = penalty::solve_penalty(objective, constraints, variables, 1e-6)?;
            let point: Vec<(String, f64)> = variables
                .iter()
                .zip(penalty_result.point.iter())
                .map(|(v, &val)| (v.name.clone(), val))
                .collect();
            let names: Vec<String> = variables.iter().map(|v| v.name.clone()).collect();
            let obj_val = eval_at(objective, &names, &penalty_result.point).ok_or_else(|| {
                SolverError::CannotSolve(
                    "Failed to evaluate objective at penalty solution".to_string(),
                )
            })?;
            Ok(LagrangianResult {
                point,
                multipliers: vec![0.0; constraints.len()],
                objective_value: obj_val,
                classification: OptimizationType::Inconclusive,
            })
        }
    }
}

// ============================================================================
// Newton-Raphson solver for nonlinear systems
// ============================================================================

/// Generate a set of diverse initial guesses for the Newton-Raphson solver.
///
/// Returns several starting points with different variable signs and magnitudes
/// to increase the chance of converging to the global minimum rather than a
/// local maximum or saddle point.
fn generate_initial_guesses(n: usize, m: usize) -> Vec<Vec<f64>> {
    let total = n + m;
    // Seed values for each variable coordinate (multipliers always start at 0).
    let seeds: &[f64] = &[1.0, -1.0, 0.5, -0.5, 0.1];
    let mut guesses: Vec<Vec<f64>> = Vec::new();

    // All-same-value guesses.
    for &s in seeds {
        let mut g = vec![s; n];
        g.resize(total, 0.0);
        guesses.push(g);
    }

    // Axis-aligned guesses: one variable at ±1, rest at 0.
    for i in 0..n {
        for &sign in &[1.0_f64, -1.0] {
            let mut g = vec![0.0; total];
            g[i] = sign;
            guesses.push(g);
        }
    }

    guesses
}

/// Evaluate a single expression at a point given by parallel `names`/`values` slices.
fn eval_at(expr: &Expression, names: &[String], values: &[f64]) -> Option<f64> {
    let vars: HashMap<String, f64> = names.iter().cloned().zip(values.iter().copied()).collect();
    expr.evaluate(&vars)
}

/// Evaluate all equations in the system and return the residual vector F(x).
fn eval_system(equations: &[Expression], names: &[String], point: &[f64]) -> Option<Vec<f64>> {
    equations
        .iter()
        .map(|eq| eval_at(eq, names, point))
        .collect()
}

/// Compute the Jacobian matrix J[i][j] = ∂Fᵢ/∂xⱼ via central finite differences.
fn finite_diff_jacobian(
    equations: &[Expression],
    names: &[String],
    point: &[f64],
    n: usize,
) -> Option<Vec<Vec<f64>>> {
    let mut jac = vec![vec![0.0_f64; n]; n];

    for j in 0..n {
        let mut p_plus = point.to_vec();
        let mut p_minus = point.to_vec();
        p_plus[j] += FD_H;
        p_minus[j] -= FD_H;

        let f_plus = eval_system(equations, names, &p_plus)?;
        let f_minus = eval_system(equations, names, &p_minus)?;

        for i in 0..n {
            jac[i][j] = (f_plus[i] - f_minus[i]) / (2.0 * FD_H);
        }
    }

    Some(jac)
}

/// Solve J·delta = -F using Gaussian elimination with partial pivoting.
///
/// Returns the solution vector `delta` or `None` if the system is singular.
fn gaussian_elimination(jac: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = rhs.len();
    // Augmented matrix [J | rhs]
    let mut aug: Vec<Vec<f64>> = jac
        .iter()
        .zip(rhs.iter())
        .map(|(row, &b)| {
            let mut r = row.clone();
            r.push(-b);
            r
        })
        .collect();

    for col in 0..n {
        // Partial pivoting.
        let pivot =
            (col..n).max_by(|&a, &b| aug[a][col].abs().partial_cmp(&aug[b][col].abs()).unwrap())?;
        aug.swap(col, pivot);

        let diag = aug[col][col];
        if diag.abs() < 1e-15 {
            return None; // Singular
        }

        for row in (col + 1)..n {
            let factor = aug[row][col] / diag;
            for k in col..=n {
                let sub = factor * aug[col][k];
                aug[row][k] -= sub;
            }
        }
    }

    // Back substitution.
    let mut sol = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = aug[i][n];
        for j in (i + 1)..n {
            s -= aug[i][j] * sol[j];
        }
        sol[i] = s / aug[i][i];
    }

    Some(sol)
}

/// Regularization epsilon added to the Jacobian diagonal when it is near-singular.
const REGULARIZATION_EPS: f64 = 1e-8;

/// Add a small epsilon to the diagonal of the Jacobian to handle near-singular cases.
fn regularize_jacobian(jac: &[Vec<f64>], eps: f64) -> Vec<Vec<f64>> {
    let mut reg = jac.to_vec();
    for i in 0..reg.len() {
        reg[i][i] += eps;
    }
    reg
}

/// Perform Newton-Raphson iteration on a system of equations in-place.
fn newton_raphson_system(
    equations: &[Expression],
    names: &[String],
    point: &mut Vec<f64>,
    n: usize,
    tol: f64,
) -> Result<(), SolverError> {
    for _iter in 0..MAX_ITER {
        let f = eval_system(equations, names, point).ok_or_else(|| {
            SolverError::CannotSolve("System evaluation failed during Newton-Raphson".to_string())
        })?;

        let residual: f64 = f.iter().map(|v| v * v).sum::<f64>().sqrt();
        if residual < tol {
            return Ok(());
        }

        let jac = finite_diff_jacobian(equations, names, point, n)
            .ok_or_else(|| SolverError::CannotSolve("Jacobian evaluation failed".to_string()))?;

        // Try direct solve first; if singular, apply diagonal regularization.
        let delta = match gaussian_elimination(&jac, &f) {
            Some(d) => d,
            None => {
                let reg_jac = regularize_jacobian(&jac, REGULARIZATION_EPS);
                gaussian_elimination(&reg_jac, &f).ok_or_else(|| {
                    SolverError::CannotSolve(
                        "Singular Jacobian — cannot proceed even with regularization".to_string(),
                    )
                })?
            }
        };

        let step_norm: f64 = delta.iter().map(|v| v * v).sum::<f64>().sqrt();
        for i in 0..n {
            point[i] += delta[i];
        }

        if step_norm < tol {
            let f_post = eval_system(equations, names, point).ok_or_else(|| {
                SolverError::CannotSolve(
                    "System evaluation failed during convergence recheck".to_string(),
                )
            })?;
            let residual_post: f64 = f_post.iter().map(|v| v * v).sum::<f64>().sqrt();
            if residual_post < tol {
                return Ok(());
            }
        }
    }

    Err(SolverError::CannotSolve(format!(
        "Lagrangian solver did not converge within {} iterations",
        MAX_ITER
    )))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Expression, UnaryOp, Variable};

    /// Build `x^2 + y^2`.
    fn norm_squared(x_name: &str, y_name: &str) -> Expression {
        let x2 = Expression::Power(
            Box::new(Expression::Variable(Variable::new(x_name))),
            Box::new(Expression::Integer(2)),
        );
        let y2 = Expression::Power(
            Box::new(Expression::Variable(Variable::new(y_name))),
            Box::new(Expression::Integer(2)),
        );
        Expression::Binary(BinaryOp::Add, Box::new(x2), Box::new(y2))
    }

    /// Build `x + y - c`.
    fn linear_sum_constraint(x_name: &str, y_name: &str, c: i64) -> Expression {
        let sum = Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Variable(Variable::new(x_name))),
            Box::new(Expression::Variable(Variable::new(y_name))),
        );
        Expression::Binary(
            BinaryOp::Sub,
            Box::new(sum),
            Box::new(Expression::Integer(c)),
        )
    }

    #[test]
    fn test_minimize_norm_squared_with_sum_constraint() {
        // min x² + y²  s.t. x + y = 1  → x = y = 0.5, f = 0.5
        let objective = norm_squared("x", "y");
        let constraint = linear_sum_constraint("x", "y", 1);
        let vars = vec![Variable::new("x"), Variable::new("y")];

        let solver = LagrangianSolver::new();
        let result = solver.solve(&objective, &[constraint], &vars).unwrap();

        let x_val = result.point.iter().find(|(n, _)| n == "x").unwrap().1;
        let y_val = result.point.iter().find(|(n, _)| n == "y").unwrap().1;

        assert!((x_val - 0.5).abs() < 1e-6, "x should be 0.5, got {}", x_val);
        assert!((y_val - 0.5).abs() < 1e-6, "y should be 0.5, got {}", y_val);
        assert!(
            (result.objective_value - 0.5).abs() < 1e-6,
            "f should be 0.5, got {}",
            result.objective_value
        );
    }

    #[test]
    fn test_minimize_sum_on_unit_circle() {
        // min x + y  s.t. x² + y² - 1 = 0  → minimum at (-1/√2, -1/√2)
        let x = Expression::Variable(Variable::new("x"));
        let y = Expression::Variable(Variable::new("y"));
        let objective = Expression::Binary(BinaryOp::Add, Box::new(x.clone()), Box::new(y.clone()));

        // constraint: x² + y² - 1 = 0
        let x2 = Expression::Power(Box::new(x), Box::new(Expression::Integer(2)));
        let y2 = Expression::Power(Box::new(y), Box::new(Expression::Integer(2)));
        let sum_sq = Expression::Binary(BinaryOp::Add, Box::new(x2), Box::new(y2));
        let constraint = Expression::Binary(
            BinaryOp::Sub,
            Box::new(sum_sq),
            Box::new(Expression::Integer(1)),
        );

        let vars = vec![Variable::new("x"), Variable::new("y")];
        let solver = LagrangianSolver::new();
        let result = solver.solve(&objective, &[constraint], &vars).unwrap();

        let x_val = result.point.iter().find(|(n, _)| n == "x").unwrap().1;
        let y_val = result.point.iter().find(|(n, _)| n == "y").unwrap().1;
        let expected = -1.0_f64 / 2.0_f64.sqrt();

        assert!(
            (x_val - expected).abs() < 1e-5,
            "x should be -1/√2 ≈ {:.6}, got {:.6}",
            expected,
            x_val
        );
        assert!(
            (y_val - expected).abs() < 1e-5,
            "y should be -1/√2 ≈ {:.6}, got {:.6}",
            expected,
            y_val
        );
    }

    #[test]
    fn test_no_constraints_gradient_zero() {
        // Unconstrained: min x² + y²  → solution at (0, 0) via ∇f = 0
        let objective = norm_squared("x", "y");
        let vars = vec![Variable::new("x"), Variable::new("y")];

        let solver = LagrangianSolver::new();
        let result = solver.solve(&objective, &[], &vars).unwrap();

        let x_val = result.point.iter().find(|(n, _)| n == "x").unwrap().1;
        let y_val = result.point.iter().find(|(n, _)| n == "y").unwrap().1;

        assert!(x_val.abs() < 1e-5, "x should be ~0, got {}", x_val);
        assert!(y_val.abs() < 1e-5, "y should be ~0, got {}", y_val);
        assert!(
            result.objective_value.abs() < 1e-10,
            "f(0,0) should be 0, got {}",
            result.objective_value
        );
    }

    #[test]
    fn test_classification_minimum() {
        // min x² + y²  s.t. x + y = 1  → local minimum at (0.5, 0.5)
        let objective = norm_squared("x", "y");
        let constraint = linear_sum_constraint("x", "y", 1);
        let vars = vec![Variable::new("x"), Variable::new("y")];

        let solver = LagrangianSolver::new();
        let result = solver.solve(&objective, &[constraint], &vars).unwrap();

        assert_eq!(
            result.classification,
            OptimizationType::LocalMinimum,
            "expected LocalMinimum, got {:?}",
            result.classification
        );
    }

    #[test]
    fn test_classification_maximum() {
        // max x² + y²  s.t. x + y = 1
        // Equivalently: min -(x² + y²)  s.t. x + y = 1
        // The critical point (0.5, 0.5) is a constrained maximum of x²+y².
        // Objective: -(x² + y²)
        let neg_norm_sq = Expression::Unary(UnaryOp::Neg, Box::new(norm_squared("x", "y")));
        let constraint = linear_sum_constraint("x", "y", 1);
        let vars = vec![Variable::new("x"), Variable::new("y")];

        // Classify via classify_critical_point directly at the known point.
        let solver = LagrangianSolver::new();
        // point: x=0.5, y=0.5, lambda=-1 (KKT for -(x²+y²) + λ(x+y-1))
        let point = vec![0.5_f64, 0.5, -1.0];
        let names = vec!["x".to_string(), "y".to_string(), "__lambda_0".to_string()];
        let classification =
            solver.classify_critical_point(&neg_norm_sq, &[constraint], &vars, &point, &names);
        // -(x²+y²) is concave so the bordered Hessian test yields LocalMaximum.
        assert_eq!(
            classification,
            OptimizationType::LocalMaximum,
            "expected LocalMaximum, got {:?}",
            classification
        );
    }

    #[test]
    fn test_optimize_constrained_uses_lagrangian() {
        // Verify the unified API succeeds via the Lagrange path when possible.
        let objective = norm_squared("x", "y");
        let constraint = linear_sum_constraint("x", "y", 1);
        let vars = vec![Variable::new("x"), Variable::new("y")];

        let result =
            optimize_constrained(&objective, &[constraint], &vars).expect("should succeed");

        let x_val = result.point.iter().find(|(n, _)| n == "x").unwrap().1;
        let y_val = result.point.iter().find(|(n, _)| n == "y").unwrap().1;
        assert!((x_val - 0.5).abs() < 1e-6, "x should be 0.5, got {}", x_val);
        assert!((y_val - 0.5).abs() < 1e-6, "y should be 0.5, got {}", y_val);
        assert!(
            (result.objective_value - 0.5).abs() < 1e-6,
            "f should be 0.5, got {}",
            result.objective_value
        );
        // Lagrange path gives a proper classification.
        assert_eq!(result.classification, OptimizationType::LocalMinimum);
    }

    #[test]
    fn test_optimize_constrained_no_constraints() {
        // With no constraints the Lagrangian reduces to ∇f = 0.
        let objective = norm_squared("x", "y");
        let vars = vec![Variable::new("x"), Variable::new("y")];

        let result = optimize_constrained(&objective, &[], &vars).expect("should succeed");

        let x_val = result.point.iter().find(|(n, _)| n == "x").unwrap().1;
        let y_val = result.point.iter().find(|(n, _)| n == "y").unwrap().1;
        assert!(x_val.abs() < 1e-5, "x should be ~0, got {}", x_val);
        assert!(y_val.abs() < 1e-5, "y should be ~0, got {}", y_val);
    }

    #[test]
    fn test_optimize_constrained_result_fields() {
        // Verify all fields of LagrangianResult are populated correctly.
        let objective = norm_squared("x", "y");
        let constraint = linear_sum_constraint("x", "y", 1);
        let vars = vec![Variable::new("x"), Variable::new("y")];

        let result =
            optimize_constrained(&objective, &[constraint], &vars).expect("should succeed");

        // Point has correct variable names and count.
        assert_eq!(result.point.len(), 2);
        assert!(result.point.iter().any(|(n, _)| n == "x"));
        assert!(result.point.iter().any(|(n, _)| n == "y"));
        // One multiplier per constraint.
        assert_eq!(result.multipliers.len(), 1);
    }

    #[test]
    fn test_hessian_numerics_quadratic() {
        // For f = x² + y², the Hessian should be diag(2, 2).
        let objective = norm_squared("x", "y");
        let vars = vec![Variable::new("x"), Variable::new("y")];
        // Names include a dummy lambda so names slice has right length.
        let names = vec!["x".to_string(), "y".to_string(), "__lambda_0".to_string()];
        let point = vec![0.5_f64, 0.5, 0.0];
        // Lagrangian with zero-weighted constraint: L ≈ f here.
        let dummy_constraint = linear_sum_constraint("x", "y", 1);
        let lagrangian = bordered_hessian::build_lagrangian(&objective, &[dummy_constraint]);
        let h = bordered_hessian::lagrangian_hessian(&lagrangian, &vars, &names, &point)
            .expect("Hessian should be computable");
        assert!((h[0][0] - 2.0).abs() < 1e-4, "H[0][0] ≈ 2, got {}", h[0][0]);
        assert!((h[1][1] - 2.0).abs() < 1e-4, "H[1][1] ≈ 2, got {}", h[1][1]);
        assert!(h[0][1].abs() < 1e-4, "H[0][1] ≈ 0, got {}", h[0][1]);
        assert!(h[1][0].abs() < 1e-4, "H[1][0] ≈ 0, got {}", h[1][0]);
    }
}
