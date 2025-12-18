//! Numerical approximation methods for equations.
//!
//! Provides numerical solvers for equations that cannot be solved symbolically,
//! using root-finding and optimization algorithms.
//!
//! # Methods
//!
//! ## Newton-Raphson Method
//! Fast convergence for smooth functions with good initial guesses.
//! Uses **symbolic differentiation** from the AST module for exact derivatives.
//!
//! ## Bisection Method
//! Guaranteed convergence for continuous functions when root is bracketed.
//! More robust but slower than Newton-Raphson.
//!
//! ## Smart Solver
//! Automatically selects the best method based on the problem characteristics.
//!
//! # Integration with Symbolic Differentiation
//!
//! The Newton-Raphson solver integrates with the symbolic differentiation
//! capability from `ast::Expression::differentiate()` (Task 188). This provides:
//!
//! - **Exact derivatives** instead of finite difference approximations
//! - **Faster convergence** due to precise derivative calculations
//! - **Better numerical stability** by avoiding finite difference errors
//! - **Clear resolution paths** showing the derivative expressions used
//!
//! # Example
//!
//! ```ignore
//! use mathsolver_core::numerical::{NewtonRaphson, NumericalConfig};
//! use mathsolver_core::ast::{Equation, Expression, Variable};
//!
//! // Solve x^2 = 5
//! let equation = Equation::new("quad",
//!     Expression::power(Expression::var("x"), 2),
//!     Expression::Integer(5));
//!
//! let solver = NewtonRaphson::with_default_config();
//! let (solution, path) = solver.solve(&equation, &Variable::new("x")).unwrap();
//! println!("x = {}", solution.value); // x ≈ 2.236
//! ```

use crate::ast::{Equation, Expression, Variable};
use crate::resolution_path::{Operation, ResolutionPath};
use std::collections::HashMap;

/// Error types for numerical solving.
#[derive(Debug, Clone, PartialEq)]
pub enum NumericalError {
    /// Failed to converge within iteration limit
    NoConvergence,
    /// Numerical instability detected
    Unstable,
    /// Invalid initial guess
    InvalidInitialGuess,
    /// Function evaluation failed
    EvaluationFailed(String),
    /// Other error
    Other(String),
}

/// Result type for numerical operations.
pub type NumericalResult<T> = Result<T, NumericalError>;

/// Configuration for numerical solvers.
#[derive(Debug, Clone)]
pub struct NumericalConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Tolerance for convergence
    pub tolerance: f64,
    /// Initial guess (if not provided, will be estimated)
    pub initial_guess: Option<f64>,
    /// Step size for derivative approximation
    pub step_size: f64,
}

impl Default for NumericalConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-10,
            initial_guess: None,
            step_size: 1e-6,
        }
    }
}

/// Numerical solution with convergence information.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericalSolution {
    /// The approximate solution value
    pub value: f64,
    /// Number of iterations taken
    pub iterations: usize,
    /// Final residual (how close to zero)
    pub residual: f64,
    /// Whether convergence was achieved
    pub converged: bool,
}

/// Newton-Raphson root finder.
#[derive(Debug)]
pub struct NewtonRaphson {
    config: NumericalConfig,
}

impl NewtonRaphson {
    pub fn new(config: NumericalConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self {
            config: NumericalConfig::default(),
        }
    }

    /// Find a root of the equation using Newton-Raphson method.
    pub fn solve(
        &self,
        equation: &Equation,
        variable: &Variable,
    ) -> NumericalResult<(NumericalSolution, ResolutionPath)> {
        use crate::resolution_path::ResolutionPathBuilder;

        // Convert equation to form f(x) = 0 by subtracting right side from left
        let f = Expression::Binary(
            crate::ast::BinaryOp::Sub,
            Box::new(equation.left.clone()),
            Box::new(equation.right.clone()),
        );

        // Compute symbolic derivative f'(x) once at the beginning
        let f_prime = f.differentiate(&variable.name);

        // Initial guess: use provided or estimate from domain
        let mut x = self.config.initial_guess.unwrap_or(1.0);

        // Build resolution path
        let mut path = ResolutionPathBuilder::new(f.clone());
        path = path.step(
            Operation::NumericalApproximation,
            format!("Starting Newton-Raphson method with initial guess x₀ = {}", x),
            Expression::Float(x),
        );
        path = path.step(
            Operation::NumericalApproximation,
            format!("Using symbolic derivative: f'(x) = {}", f_prime),
            f_prime.clone(),
        );

        let mut converged = false;
        let mut iterations = 0;
        let mut residual = 0.0;

        for i in 0..self.config.max_iterations {
            iterations = i + 1;

            // Evaluate f(x) at current point
            let mut vars = HashMap::new();
            vars.insert(variable.name.clone(), x);

            let fx = f.evaluate(&vars).ok_or_else(|| {
                NumericalError::EvaluationFailed(format!(
                    "Failed to evaluate function at x = {}",
                    x
                ))
            })?;

            residual = fx.abs();

            // Check convergence
            if residual < self.config.tolerance {
                converged = true;
                path = path.step(
                    Operation::NumericalApproximation,
                    format!("Converged: |f(x)| = {} < {}", residual, self.config.tolerance),
                    Expression::Float(x),
                );
                break;
            }

            // Evaluate derivative f'(x) at current point using symbolic differentiation
            let derivative = f_prime.evaluate(&vars).ok_or_else(|| {
                NumericalError::EvaluationFailed(format!(
                    "Failed to evaluate derivative at x = {}",
                    x
                ))
            })?;

            // Check for zero derivative (would cause division by zero)
            if derivative.abs() < 1e-14 {
                return Err(NumericalError::Unstable);
            }

            // Newton-Raphson update: x_{n+1} = x_n - f(x_n) / f'(x_n)
            let x_next = x - fx / derivative;

            // Check for NaN or infinity
            if !x_next.is_finite() {
                return Err(NumericalError::Unstable);
            }

            // Add step to path every 10 iterations or at end
            if i % 10 == 0 || i == self.config.max_iterations - 1 {
                path = path.step(
                    Operation::NumericalApproximation,
                    format!("Iteration {}: x = {}, f(x) = {}, f'(x) = {}",
                            iterations, x_next, fx, derivative),
                    Expression::Float(x_next),
                );
            }

            // Check step size for convergence
            if (x_next - x).abs() < self.config.tolerance {
                x = x_next;
                converged = true;
                path = path.step(
                    Operation::NumericalApproximation,
                    format!("Converged: |Δx| = {} < {}", (x_next - x).abs(), self.config.tolerance),
                    Expression::Float(x),
                );
                break;
            }

            x = x_next;
        }

        if !converged {
            return Err(NumericalError::NoConvergence);
        }

        let solution = NumericalSolution {
            value: x,
            iterations,
            residual,
            converged,
        };

        let final_path = path.finish(Expression::Float(x));

        Ok((solution, final_path))
    }
}

/// Secant method root finder (derivative-free).
#[derive(Debug)]
pub struct SecantMethod {
    config: NumericalConfig,
}

impl SecantMethod {
    pub fn new(config: NumericalConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self {
            config: NumericalConfig::default(),
        }
    }

    /// Find a root using the secant method.
    pub fn solve(
        &self,
        _equation: &Equation,
        _variable: &Variable,
    ) -> NumericalResult<(NumericalSolution, ResolutionPath)> {
        // TODO: Implement secant method
        // Uses two initial points instead of derivative
        // x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
        Err(NumericalError::Other("Not yet implemented".to_string()))
    }
}

/// Bisection method root finder (guaranteed convergence for continuous functions).
#[derive(Debug)]
pub struct BisectionMethod {
    config: NumericalConfig,
}

impl BisectionMethod {
    pub fn new(config: NumericalConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self {
            config: NumericalConfig::default(),
        }
    }

    /// Find a root using bisection method.
    ///
    /// Requires interval [a, b] where f(a) and f(b) have opposite signs.
    pub fn solve(
        &self,
        equation: &Equation,
        variable: &Variable,
        interval: (f64, f64),
    ) -> NumericalResult<(NumericalSolution, ResolutionPath)> {
        use crate::resolution_path::ResolutionPathBuilder;

        // Convert equation to form f(x) = 0
        let f = Expression::Binary(
            crate::ast::BinaryOp::Sub,
            Box::new(equation.left.clone()),
            Box::new(equation.right.clone()),
        );

        let mut a = interval.0;
        let mut b = interval.1;

        // Ensure a < b
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }

        // Build resolution path
        let mut path = ResolutionPathBuilder::new(f.clone());

        // Evaluate at endpoints
        let mut vars = HashMap::new();
        vars.insert(variable.name.clone(), a);
        let fa = f.evaluate(&vars).ok_or_else(|| {
            NumericalError::EvaluationFailed(format!("Failed to evaluate function at x = {}", a))
        })?;

        vars.insert(variable.name.clone(), b);
        let fb = f.evaluate(&vars).ok_or_else(|| {
            NumericalError::EvaluationFailed(format!("Failed to evaluate function at x = {}", b))
        })?;

        // Check that f(a) and f(b) have opposite signs
        if fa * fb > 0.0 {
            return Err(NumericalError::Other(format!(
                "Bisection requires f(a) and f(b) to have opposite signs. f({}) = {}, f({}) = {}",
                a, fa, b, fb
            )));
        }

        path = path.step(
            Operation::NumericalApproximation,
            format!("Starting bisection method on interval [{}, {}]. f({}) = {}, f({}) = {}",
                    a, b, a, fa, b, fb),
            Expression::Float((a + b) / 2.0),
        );

        let mut iterations = 0;
        let mut c = (a + b) / 2.0;
        let mut fc = 0.0;

        for i in 0..self.config.max_iterations {
            iterations = i + 1;

            // Midpoint
            c = (a + b) / 2.0;

            // Evaluate at midpoint
            vars.insert(variable.name.clone(), c);
            fc = f.evaluate(&vars).ok_or_else(|| {
                NumericalError::EvaluationFailed(format!("Failed to evaluate function at x = {}", c))
            })?;

            // Check convergence by residual
            if fc.abs() < self.config.tolerance {
                path = path.step(
                    Operation::NumericalApproximation,
                    format!("Converged: |f({})| = {} < {}", c, fc.abs(), self.config.tolerance),
                    Expression::Float(c),
                );
                break;
            }

            // Check convergence by interval width
            if (b - a) / 2.0 < self.config.tolerance {
                path = path.step(
                    Operation::NumericalApproximation,
                    format!("Converged: interval width {} < {}", (b - a) / 2.0, self.config.tolerance),
                    Expression::Float(c),
                );
                break;
            }

            // Determine which half contains the root
            vars.insert(variable.name.clone(), a);
            let fa_curr = f.evaluate(&vars).ok_or_else(|| {
                NumericalError::EvaluationFailed(format!("Failed to evaluate function at x = {}", a))
            })?;

            if fa_curr * fc < 0.0 {
                // Root is in left half [a, c]
                b = c;
            } else {
                // Root is in right half [c, b]
                a = c;
            }

            // Log progress every 10 iterations
            if i % 10 == 0 || i == self.config.max_iterations - 1 {
                path = path.step(
                    Operation::NumericalApproximation,
                    format!("Iteration {}: interval = [{}, {}], midpoint = {}, f(midpoint) = {}",
                            iterations, a, b, c, fc),
                    Expression::Float(c),
                );
            }
        }

        let solution = NumericalSolution {
            value: c,
            iterations,
            residual: fc.abs(),
            converged: fc.abs() < self.config.tolerance || (b - a) / 2.0 < self.config.tolerance,
        };

        if !solution.converged {
            return Err(NumericalError::NoConvergence);
        }

        let final_path = path.finish(Expression::Float(c));

        Ok((solution, final_path))
    }
}

/// Brent's method (hybrid root finder, very robust).
#[derive(Debug)]
pub struct BrentsMethod {
    config: NumericalConfig,
}

impl BrentsMethod {
    pub fn new(config: NumericalConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self {
            config: NumericalConfig::default(),
        }
    }

    /// Find a root using Brent's method.
    pub fn solve(
        &self,
        _equation: &Equation,
        _variable: &Variable,
        _interval: (f64, f64),
    ) -> NumericalResult<(NumericalSolution, ResolutionPath)> {
        // TODO: Implement Brent's method
        // Combines bisection, secant, and inverse quadratic interpolation
        // Very robust and efficient
        Err(NumericalError::Other("Not yet implemented".to_string()))
    }
}

/// Gradient descent optimizer for minimization problems.
#[derive(Debug)]
pub struct GradientDescent {
    config: NumericalConfig,
    learning_rate: f64,
}

impl GradientDescent {
    pub fn new(config: NumericalConfig, learning_rate: f64) -> Self {
        Self {
            config,
            learning_rate,
        }
    }

    /// Minimize an expression with respect to variables.
    pub fn minimize(
        &self,
        _expression: &Expression,
        _variables: &[Variable],
    ) -> NumericalResult<HashMap<Variable, f64>> {
        // TODO: Implement gradient descent
        // TODO: Compute gradient using symbolic differentiation or finite differences
        // TODO: Support momentum and adaptive learning rates
        Err(NumericalError::Other("Not yet implemented".to_string()))
    }
}

/// Levenberg-Marquardt algorithm for nonlinear least squares.
#[derive(Debug)]
pub struct LevenbergMarquardt {
    config: NumericalConfig,
}

impl LevenbergMarquardt {
    pub fn new(config: NumericalConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self {
            config: NumericalConfig::default(),
        }
    }

    /// Solve nonlinear least squares problem.
    pub fn solve_least_squares(
        &self,
        _equations: &[Equation],
        _variables: &[Variable],
    ) -> NumericalResult<HashMap<Variable, f64>> {
        // TODO: Implement Levenberg-Marquardt using argmin
        // TODO: Useful for overdetermined systems
        // TODO: Minimizes sum of squared residuals
        Err(NumericalError::Other("Not yet implemented".to_string()))
    }
}

/// Smart numerical solver that selects appropriate method.
#[derive(Debug)]
pub struct SmartNumericalSolver {
    config: NumericalConfig,
}

impl SmartNumericalSolver {
    pub fn new(config: NumericalConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self {
            config: NumericalConfig::default(),
        }
    }

    /// Solve equation numerically using most appropriate method.
    pub fn solve(
        &self,
        equation: &Equation,
        variable: &Variable,
    ) -> NumericalResult<(NumericalSolution, ResolutionPath)> {
        // Convert equation to f(x) = 0 form
        let f = Expression::Binary(
            crate::ast::BinaryOp::Sub,
            Box::new(equation.left.clone()),
            Box::new(equation.right.clone()),
        );

        // Strategy 1: Try Newton-Raphson if initial guess is provided
        if self.config.initial_guess.is_some() {
            let newton = NewtonRaphson::new(self.config.clone());
            if let Ok(result) = newton.solve(equation, variable) {
                return Ok(result);
            }
        }

        // Strategy 2: Try to bracket the root and use bisection
        let initial_guess = self.config.initial_guess.unwrap_or(1.0);
        if let Some((a, b)) = bracket_root(&f, variable, initial_guess, 10000.0) {
            let bisection = BisectionMethod::new(self.config.clone());
            if let Ok(result) = bisection.solve(equation, variable, (a, b)) {
                return Ok(result);
            }
        }

        // Strategy 3: Try Newton-Raphson with multiple initial guesses
        let initial_guesses = vec![0.0, 1.0, -1.0, 10.0, -10.0, 100.0, -100.0];

        for guess in initial_guesses {
            let mut config = self.config.clone();
            config.initial_guess = Some(guess);

            let newton = NewtonRaphson::new(config);
            if let Ok(result) = newton.solve(equation, variable) {
                return Ok(result);
            }
        }

        // Strategy 4: Try bracketing around different centers
        let centers = vec![0.0, 1.0, -1.0, 10.0, -10.0, 100.0];
        for center in centers {
            if let Some((a, b)) = bracket_root(&f, variable, center, 10000.0) {
                let bisection = BisectionMethod::new(self.config.clone());
                if let Ok(result) = bisection.solve(equation, variable, (a, b)) {
                    return Ok(result);
                }
            }
        }

        // No method succeeded
        Err(NumericalError::NoConvergence)
    }

    /// Solve equation numerically with a specified interval for bracketing methods.
    pub fn solve_with_interval(
        &self,
        equation: &Equation,
        variable: &Variable,
        interval: (f64, f64),
    ) -> NumericalResult<(NumericalSolution, ResolutionPath)> {
        // Prefer bisection when an interval is provided
        let bisection = BisectionMethod::new(self.config.clone());
        bisection.solve(equation, variable, interval)
    }
}

/// Expression evaluator with variable substitution.
pub struct Evaluator {
    variables: HashMap<Variable, f64>,
}

impl Evaluator {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    pub fn with_variables(variables: HashMap<Variable, f64>) -> Self {
        Self { variables }
    }

    pub fn set_variable(&mut self, var: Variable, value: f64) {
        self.variables.insert(var, value);
    }

    /// Evaluate an expression to a floating point value.
    pub fn evaluate(&self, _expression: &Expression) -> Result<f64, String> {
        // TODO: Implement expression evaluation
        // TODO: Handle all expression types (binary ops, functions, etc.)
        // TODO: Substitute variables from the variable map
        // TODO: Use fasteval for efficient evaluation
        Err("Not yet implemented".to_string())
    }
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute derivative using finite differences (central difference method).
///
/// f'(x) ≈ [f(x + h) - f(x - h)] / (2h)
fn compute_derivative_fd(
    expr: &Expression,
    variable: &Variable,
    x: f64,
    h: f64,
) -> NumericalResult<f64> {
    let mut vars = HashMap::new();

    // Evaluate f(x + h)
    vars.insert(variable.name.clone(), x + h);
    let f_plus = expr.evaluate(&vars).ok_or_else(|| {
        NumericalError::EvaluationFailed(format!("Failed to evaluate function at x = {}", x + h))
    })?;

    // Evaluate f(x - h)
    vars.insert(variable.name.clone(), x - h);
    let f_minus = expr.evaluate(&vars).ok_or_else(|| {
        NumericalError::EvaluationFailed(format!("Failed to evaluate function at x = {}", x - h))
    })?;

    // Central difference approximation
    let derivative = (f_plus - f_minus) / (2.0 * h);

    if !derivative.is_finite() {
        return Err(NumericalError::Unstable);
    }

    Ok(derivative)
}

/// Find a suitable initial interval for root finding by bracketing.
///
/// Returns (a, b) such that f(a) and f(b) have opposite signs.
fn bracket_root(
    expr: &Expression,
    variable: &Variable,
    center: f64,
    max_range: f64,
) -> Option<(f64, f64)> {
    let mut vars = HashMap::new();

    // Try expanding intervals around the center
    for scale in [1.0_f64, 10.0, 100.0, 1000.0] {
        let range = scale.min(max_range);

        for offset in [0.0, range / 4.0, range / 2.0, 3.0 * range / 4.0] {
            let a = center - range + offset;
            let b = center + range - offset;

            vars.insert(variable.name.clone(), a);
            let fa = expr.evaluate(&vars)?;

            vars.insert(variable.name.clone(), b);
            let fb = expr.evaluate(&vars)?;

            if fa * fb < 0.0 {
                return Some((a, b));
            }
        }
    }

    None
}

// TODO: Add support for interval arithmetic
// TODO: Add automatic differentiation
// TODO: Add support for complex-valued functions
// TODO: Add parallel evaluation for gradient computation
// TODO: Add sensitivity analysis
// TODO: Add uncertainty propagation
