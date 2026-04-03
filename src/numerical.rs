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
//! use thales::numerical::{NewtonRaphson, NumericalConfig};
//! use thales::ast::{Equation, Expression, Variable};
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
///
/// Represents the various failure modes that can occur during numerical
/// root-finding and optimization.
///
/// # Variants
///
/// * `NoConvergence` - The algorithm did not converge within the maximum number
///   of iterations. This can happen if the initial guess is too far from the root,
///   the function is poorly behaved, or the tolerance is too tight.
///
/// * `Unstable` - Numerical instability was detected, such as division by zero
///   (when derivative is zero in Newton-Raphson), NaN values, or infinite values.
///
/// * `InvalidInitialGuess` - The provided initial guess is invalid (e.g., outside
///   the valid domain of the function).
///
/// * `EvaluationFailed` - Function or derivative evaluation failed at a specific
///   point. The string contains details about the failure.
///
/// * `Other` - Any other error with a descriptive message.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum NumericalError {
    /// Failed to converge within iteration limit
    NoConvergence,
    /// Numerical instability detected (zero derivative, NaN, infinity)
    Unstable,
    /// Invalid initial guess
    InvalidInitialGuess,
    /// Function evaluation failed
    EvaluationFailed(String),
    /// Other error
    Other(String),
}

impl std::fmt::Display for NumericalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NumericalError::NoConvergence => write!(f, "Failed to converge within iteration limit"),
            NumericalError::Unstable => write!(f, "Numerical instability detected"),
            NumericalError::InvalidInitialGuess => write!(f, "Invalid initial guess"),
            NumericalError::EvaluationFailed(msg) => {
                write!(f, "Function evaluation failed: {}", msg)
            }
            NumericalError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for NumericalError {}

/// Result type for numerical operations.
pub type NumericalResult<T> = Result<T, NumericalError>;

/// Configuration for numerical solvers.
///
/// Controls the behavior and termination criteria for numerical root-finding
/// algorithms.
///
/// # Fields
///
/// * `max_iterations` - Maximum number of iterations before giving up.
///   Default: 1000
///
/// * `tolerance` - Convergence tolerance. The algorithm stops when either:
///   - The residual |f(x)| < tolerance, or
///   - The step size |Δx| < tolerance
///   Default: 1e-10
///
/// * `initial_guess` - Starting point for the algorithm. If `None`, the solver
///   will attempt to estimate a reasonable starting point. For Newton-Raphson,
///   a good initial guess close to the actual root leads to faster convergence.
///   Default: None (will use 1.0)
///
/// * `step_size` - Step size for finite difference derivative approximation.
///   This is only used as a fallback; the Newton-Raphson solver primarily uses
///   symbolic differentiation for exact derivatives.
///   Default: 1e-6
///
/// # Example
///
/// ```
/// use thales::numerical::NumericalConfig;
///
/// // Use default configuration
/// let config = NumericalConfig::default();
///
/// // Custom configuration for high precision
/// let precise_config = NumericalConfig {
///     max_iterations: 10000,
///     tolerance: 1e-15,
///     initial_guess: Some(2.0),
///     step_size: 1e-8,
/// };
/// ```
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
///
/// Contains the result of a numerical root-finding operation along with
/// diagnostic information about the convergence process.
///
/// # Fields
///
/// * `value` - The approximate solution (root) found by the algorithm
///
/// * `iterations` - Number of iterations performed before convergence or
///   termination
///
/// * `residual` - Final residual value |f(x)|. For a perfect solution, this
///   would be 0.0. In practice, it should be smaller than the configured
///   tolerance.
///
/// * `converged` - Whether the algorithm successfully converged to a solution
///   within the tolerance and iteration limits
///
/// # Example
///
/// ```
/// use thales::numerical::{NewtonRaphson, NumericalConfig};
/// use thales::ast::{Equation, Expression, Variable};
///
/// // Solve x^2 = 5
/// let equation = Equation::new(
///     "find_sqrt5",
///     Expression::Power(
///         Box::new(Expression::Variable(Variable::new("x"))),
///         Box::new(Expression::Integer(2))
///     ),
///     Expression::Integer(5)
/// );
///
/// let solver = NewtonRaphson::with_default_config();
/// let (solution, _path) = solver.solve(&equation, &Variable::new("x")).unwrap();
///
/// assert!(solution.converged);
/// assert!((solution.value - 2.236067977).abs() < 1e-6); // √5 ≈ 2.236
/// assert!(solution.residual < 1e-10);
/// println!("Solution found in {} iterations", solution.iterations);
/// ```
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

/// Newton-Raphson root finder with symbolic differentiation.
///
/// Implements the Newton-Raphson method for finding roots of equations:
///
/// **x_{n+1} = x_n - f(x_n) / f'(x_n)**
///
/// This is an iterative method that starts from an initial guess and refines
/// it by linearizing the function at each step. It converges quadratically
/// for smooth functions when the initial guess is sufficiently close to the root.
///
/// # Key Features
///
/// * **Symbolic differentiation**: Uses `Expression::differentiate()` for exact
///   derivatives instead of finite difference approximations
/// * **Fast convergence**: Quadratic convergence rate near the root
/// * **Resolution path tracking**: Records all steps for educational purposes
/// * **Robust error handling**: Detects zero derivatives, NaN, and divergence
///
/// # Algorithm
///
/// 1. Convert equation to form f(x) = 0
/// 2. Compute symbolic derivative f'(x) once
/// 3. Start from initial guess x₀
/// 4. Iterate: x_{n+1} = x_n - f(x_n) / f'(x_n)
/// 5. Check convergence: |f(x)| < tolerance or |Δx| < tolerance
/// 6. Return solution when converged
///
/// # Convergence Criteria
///
/// The algorithm stops when either:
/// - Residual criterion: |f(x)| < tolerance
/// - Step size criterion: |x_{n+1} - x_n| < tolerance
/// - Maximum iterations reached (returns error)
///
/// # Limitations
///
/// * Requires good initial guess (close to actual root)
/// * Fails when f'(x) = 0 at iteration point
/// * May diverge for poorly behaved functions
/// * Only finds one root at a time
///
/// # Example: Square Root
///
/// ```
/// use thales::numerical::{NewtonRaphson, NumericalConfig};
/// use thales::ast::{Equation, Expression, Variable};
///
/// // Solve x^2 = 5 to find √5
/// let equation = Equation::new(
///     "sqrt5",
///     Expression::Power(
///         Box::new(Expression::Variable(Variable::new("x"))),
///         Box::new(Expression::Integer(2))
///     ),
///     Expression::Integer(5)
/// );
///
/// let solver = NewtonRaphson::with_default_config();
/// let (solution, _path) = solver.solve(&equation, &Variable::new("x")).unwrap();
///
/// assert!((solution.value - 2.236067977).abs() < 1e-6);
/// assert!(solution.converged);
/// println!("√5 ≈ {} (found in {} iterations)", solution.value, solution.iterations);
/// ```
///
/// # Example: Custom Configuration
///
/// ```
/// use thales::numerical::{NewtonRaphson, NumericalConfig};
/// use thales::ast::{Equation, Expression, Variable};
///
/// // High precision configuration
/// let config = NumericalConfig {
///     max_iterations: 1000,
///     tolerance: 1e-15,
///     initial_guess: Some(1.5), // good guess for √5
///     step_size: 1e-8,
/// };
///
/// let equation = Equation::new(
///     "cubic",
///     Expression::Power(
///         Box::new(Expression::Variable(Variable::new("x"))),
///         Box::new(Expression::Integer(3))
///     ),
///     Expression::Integer(27)
/// );
///
/// let solver = NewtonRaphson::new(config);
/// let (solution, _) = solver.solve(&equation, &Variable::new("x")).unwrap();
/// assert!((solution.value - 3.0).abs() < 1e-10); // ∛27 = 3
/// ```
#[derive(Debug)]
pub struct NewtonRaphson {
    config: NumericalConfig,
}

impl NewtonRaphson {
    /// Create a new Newton-Raphson solver with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration controlling iteration limits, tolerance, and
    ///   initial guess
    ///
    /// # Example
    ///
    /// ```
    /// use thales::numerical::{NewtonRaphson, NumericalConfig};
    ///
    /// let config = NumericalConfig {
    ///     max_iterations: 500,
    ///     tolerance: 1e-12,
    ///     initial_guess: Some(2.0),
    ///     step_size: 1e-7,
    /// };
    ///
    /// let solver = NewtonRaphson::new(config);
    /// ```
    pub fn new(config: NumericalConfig) -> Self {
        Self { config }
    }

    /// Create a new Newton-Raphson solver with default configuration.
    ///
    /// Uses:
    /// - max_iterations: 1000
    /// - tolerance: 1e-10
    /// - initial_guess: None (will use 1.0)
    /// - step_size: 1e-6
    ///
    /// # Example
    ///
    /// ```
    /// use thales::numerical::NewtonRaphson;
    ///
    /// let solver = NewtonRaphson::with_default_config();
    /// ```
    pub fn with_default_config() -> Self {
        Self {
            config: NumericalConfig::default(),
        }
    }

    /// Find a root of the equation using Newton-Raphson method.
    ///
    /// # Algorithm Steps
    ///
    /// 1. Convert equation to f(x) = 0 form: f(x) = left - right
    /// 2. Compute symbolic derivative f'(x) using `Expression::differentiate()`
    /// 3. Initialize x from config or default to 1.0
    /// 4. Iterate Newton-Raphson formula: x_{n+1} = x_n - f(x_n)/f'(x_n)
    /// 5. Check convergence at each step
    /// 6. Return solution and resolution path
    ///
    /// # Arguments
    ///
    /// * `equation` - The equation to solve
    /// * `variable` - The variable to solve for
    ///
    /// # Returns
    ///
    /// * `Ok((solution, path))` - The numerical solution and resolution path
    /// * `Err(NumericalError)` - If solving fails
    ///
    /// # Errors
    ///
    /// * `NumericalError::NoConvergence` - Did not converge within max_iterations
    /// * `NumericalError::Unstable` - Zero derivative or NaN encountered
    /// * `NumericalError::EvaluationFailed` - Function evaluation failed
    ///
    /// # Example
    ///
    /// ```
    /// use thales::numerical::NewtonRaphson;
    /// use thales::ast::{Equation, Expression, Variable};
    ///
    /// // Solve x^2 = 5
    /// let equation = Equation::new(
    ///     "quadratic",
    ///     Expression::Power(
    ///         Box::new(Expression::Variable(Variable::new("x"))),
    ///         Box::new(Expression::Integer(2))
    ///     ),
    ///     Expression::Integer(5)
    /// );
    ///
    /// let solver = NewtonRaphson::with_default_config();
    /// match solver.solve(&equation, &Variable::new("x")) {
    ///     Ok((solution, _path)) => {
    ///         println!("Found root: x = {}", solution.value);
    ///         println!("Iterations: {}", solution.iterations);
    ///         println!("Residual: {}", solution.residual);
    ///     }
    ///     Err(e) => println!("Failed to solve: {:?}", e),
    /// }
    /// ```
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
            format!(
                "Starting Newton-Raphson method with initial guess x₀ = {}",
                x
            ),
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
                    format!(
                        "Converged: |f(x)| = {} < {}",
                        residual, self.config.tolerance
                    ),
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
                    format!(
                        "Iteration {}: x = {}, f(x) = {}, f'(x) = {}",
                        iterations, x_next, fx, derivative
                    ),
                    Expression::Float(x_next),
                );
            }

            // Check step size for convergence
            if (x_next - x).abs() < self.config.tolerance {
                x = x_next;
                converged = true;
                path = path.step(
                    Operation::NumericalApproximation,
                    format!(
                        "Converged: |Δx| = {} < {}",
                        (x_next - x).abs(),
                        self.config.tolerance
                    ),
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
///
/// Implements the bisection method for finding roots of equations. This is the most
/// robust root-finding algorithm - it is guaranteed to converge to a root if:
/// 1. The function is continuous on the interval [a, b]
/// 2. f(a) and f(b) have opposite signs (the root is bracketed)
///
/// # Algorithm
///
/// 1. Start with interval [a, b] where f(a) and f(b) have opposite signs
/// 2. Compute midpoint c = (a + b) / 2
/// 3. Evaluate f(c)
/// 4. If f(c) ≈ 0, return c as the root
/// 5. If f(a) and f(c) have opposite signs, set b = c
/// 6. Otherwise, set a = c
/// 7. Repeat until convergence
///
/// # Convergence
///
/// * **Linear convergence**: Error approximately halves each iteration
/// * **Guaranteed convergence**: Always converges if root is bracketed
/// * **Predictable iterations**: Number of iterations is log₂((b-a)/tolerance)
/// * **Robust**: Works even for poorly behaved functions
///
/// # Trade-offs
///
/// **Advantages:**
/// - Guaranteed convergence when root is bracketed
/// - Very robust, works for non-smooth functions
/// - No derivative needed
/// - No risk of divergence
///
/// **Disadvantages:**
/// - Slower than Newton-Raphson (linear vs quadratic convergence)
/// - Requires initial interval with sign change
/// - Only finds one root per interval
/// - Cannot find roots at extrema (where f'(x) = 0)
///
/// # Example: Square Root
///
/// ```
/// use thales::numerical::BisectionMethod;
/// use thales::ast::{Equation, Expression, Variable};
///
/// // Solve x^2 = 5 to find √5
/// // We know root is between 2 and 3 because 2² = 4 < 5 and 3² = 9 > 5
/// let equation = Equation::new(
///     "sqrt5",
///     Expression::Power(
///         Box::new(Expression::Variable(Variable::new("x"))),
///         Box::new(Expression::Integer(2))
///     ),
///     Expression::Integer(5)
/// );
///
/// let solver = BisectionMethod::with_default_config();
/// let (solution, _path) = solver.solve(&equation, &Variable::new("x"), (2.0, 3.0)).unwrap();
///
/// assert!((solution.value - 2.236067977).abs() < 1e-6);
/// assert!(solution.converged);
/// println!("√5 ≈ {} (found in {} iterations)", solution.value, solution.iterations);
/// ```
///
/// # Example: Finding Multiple Roots
///
/// ```
/// use thales::numerical::BisectionMethod;
/// use thales::ast::{Equation, Expression, Variable};
///
/// // Solve x^2 - 1 = 0 which has roots at x = -1 and x = 1
/// let equation = Equation::new(
///     "quadratic",
///     Expression::Binary(
///         thales::ast::BinaryOp::Sub,
///         Box::new(Expression::Power(
///             Box::new(Expression::Variable(Variable::new("x"))),
///             Box::new(Expression::Integer(2))
///         )),
///         Box::new(Expression::Integer(1))
///     ),
///     Expression::Integer(0)
/// );
///
/// let solver = BisectionMethod::with_default_config();
///
/// // Find negative root in [-2, 0]
/// let (solution1, _) = solver.solve(&equation, &Variable::new("x"), (-2.0, 0.0)).unwrap();
/// assert!((solution1.value - (-1.0)).abs() < 1e-6);
///
/// // Find positive root in [0, 2]
/// let (solution2, _) = solver.solve(&equation, &Variable::new("x"), (0.0, 2.0)).unwrap();
/// assert!((solution2.value - 1.0).abs() < 1e-6);
/// ```
#[derive(Debug)]
pub struct BisectionMethod {
    config: NumericalConfig,
}

impl BisectionMethod {
    /// Creates a new bisection method solver with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Numerical configuration (tolerance, iterations, etc.)
    pub fn new(config: NumericalConfig) -> Self {
        Self { config }
    }

    /// Creates a new bisection method solver with default configuration.
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
            format!(
                "Starting bisection method on interval [{}, {}]. f({}) = {}, f({}) = {}",
                a, b, a, fa, b, fb
            ),
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
                NumericalError::EvaluationFailed(format!(
                    "Failed to evaluate function at x = {}",
                    c
                ))
            })?;

            // Check convergence by residual
            if fc.abs() < self.config.tolerance {
                path = path.step(
                    Operation::NumericalApproximation,
                    format!(
                        "Converged: |f({})| = {} < {}",
                        c,
                        fc.abs(),
                        self.config.tolerance
                    ),
                    Expression::Float(c),
                );
                break;
            }

            // Check convergence by interval width
            if (b - a) / 2.0 < self.config.tolerance {
                path = path.step(
                    Operation::NumericalApproximation,
                    format!(
                        "Converged: interval width {} < {}",
                        (b - a) / 2.0,
                        self.config.tolerance
                    ),
                    Expression::Float(c),
                );
                break;
            }

            // Determine which half contains the root
            vars.insert(variable.name.clone(), a);
            let fa_curr = f.evaluate(&vars).ok_or_else(|| {
                NumericalError::EvaluationFailed(format!(
                    "Failed to evaluate function at x = {}",
                    a
                ))
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
                    format!(
                        "Iteration {}: interval = [{}, {}], midpoint = {}, f(midpoint) = {}",
                        iterations, a, b, c, fc
                    ),
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
///
/// Implements Brent's method, a root-finding algorithm that combines the
/// reliability of bisection with the speed of the secant method and inverse
/// quadratic interpolation.
///
/// At each step the algorithm picks the fastest method whose step falls within
/// the current bracket and is smaller than half the previous step; otherwise it
/// falls back to bisection.  This guarantees convergence whenever the root is
/// bracketed while achieving super-linear convergence in practice.
///
/// # Example
///
/// ```
/// use thales::numerical::{BrentsMethod, NumericalConfig};
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
/// let solver = BrentsMethod::with_default_config();
/// let (solution, _path) = solver.solve(&equation, &Variable::new("x"), (1.0, 2.0)).unwrap();
///
/// assert!((solution.value - std::f64::consts::SQRT_2).abs() < 1e-10);
/// assert!(solution.converged);
/// ```
#[derive(Debug)]
pub struct BrentsMethod {
    config: NumericalConfig,
}

impl BrentsMethod {
    /// Creates a new Brent's method solver with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Numerical configuration (tolerance, iterations, etc.)
    pub fn new(config: NumericalConfig) -> Self {
        Self { config }
    }

    /// Creates a new Brent's method solver with default configuration.
    pub fn with_default_config() -> Self {
        Self {
            config: NumericalConfig::default(),
        }
    }

    /// Find a root using Brent's method.
    ///
    /// Requires a bracketing interval `(a, b)` such that `f(a)` and `f(b)` have
    /// opposite signs.  Returns an error if the interval does not bracket a root.
    ///
    /// # Errors
    ///
    /// * `NumericalError::Other` – interval does not bracket a root
    /// * `NumericalError::NoConvergence` – max iterations reached
    /// * `NumericalError::EvaluationFailed` – function evaluation failed
    pub fn solve(
        &self,
        equation: &Equation,
        variable: &Variable,
        interval: (f64, f64),
    ) -> NumericalResult<(NumericalSolution, ResolutionPath)> {
        use crate::resolution_path::ResolutionPathBuilder;

        let (f, eval) = brent_make_eval(equation, variable);
        let (mut a, mut b, mut fa, mut fb) = brent_init_bracket(interval, &eval)?;

        let mut path = ResolutionPathBuilder::new(f);
        path = path.step(
            Operation::NumericalApproximation,
            format!("Starting Brent on [{a}, {b}]: f(a)={fa:.6e}, f(b)={fb:.6e}"),
            Expression::Float(b),
        );

        let mut st = BrentState {
            c: a,
            fc: fa,
            d: b - a,
            e: b - a,
            mflag: true,
        };
        let mut iterations = 0;
        let mut converged = false;

        for i in 0..self.config.max_iterations {
            iterations = i + 1;
            if fb.abs() < self.config.tolerance {
                converged = true;
                break;
            }

            let (s, bisected) = brent_next_point(a, b, fa, fb, &st, self.config.tolerance);
            st.mflag = bisected;
            let fs = eval(s)?;

            if i % 10 == 0 {
                let method = if bisected { "bisect" } else { "interpolate" };
                path = path.step(
                    Operation::NumericalApproximation,
                    format!("Iter {iterations}: x={s:.10}, f(x)={fs:.6e} [{method}]"),
                    Expression::Float(s),
                );
            }

            brent_update_bracket(&mut a, &mut b, &mut st, &mut fa, &mut fb, s, fs);
            if (b - a).abs() < self.config.tolerance {
                converged = true;
                break;
            }
        }

        if !converged {
            return Err(NumericalError::NoConvergence);
        }

        path = path.step(
            Operation::NumericalApproximation,
            format!("Converged: x={b:.15}, |f(x)|={:.6e}", fb.abs()),
            Expression::Float(b),
        );
        let sol = NumericalSolution {
            value: b,
            iterations,
            residual: fb.abs(),
            converged,
        };
        Ok((sol, path.finish(Expression::Float(b))))
    }
}

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
    /// Creates a new Levenberg-Marquardt solver with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Numerical configuration (tolerance, iterations, etc.)
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

/// Smart numerical solver that automatically selects the best method.
///
/// This solver attempts multiple strategies to find a root, choosing the most
/// appropriate method based on the problem characteristics and available information.
/// It provides a "best effort" solution without requiring the user to understand
/// the trade-offs between different numerical methods.
///
/// # Strategy Selection
///
/// The solver tries methods in order of preference:
///
/// 1. **Newton-Raphson** (if initial guess provided)
///    - Fast quadratic convergence with symbolic differentiation
///    - Used when user provides a good initial guess
///
/// 2. **Bisection** (if root can be bracketed)
///    - Attempts to find interval [a, b] where f(a) and f(b) have opposite signs
///    - Guaranteed convergence, slower but robust
///
/// 3. **Newton-Raphson with multiple guesses**
///    - Tries several initial guesses: 0, ±1, ±10, ±100
///    - Increases chances of finding a root for unknown functions
///
/// 4. **Bisection with multiple centers**
///    - Attempts bracketing around different center points
///    - Last resort for difficult functions
///
/// # When to Use
///
/// Use `SmartNumericalSolver` when:
/// - You don't know which method is best for your equation
/// - You want a "just solve it" approach without tuning
/// - The function behavior is unknown or complex
/// - You need robust solving without manual intervention
///
/// Use specialized solvers (Newton-Raphson, Bisection) when:
/// - You know the appropriate method for your problem
/// - You need fine control over convergence parameters
/// - Performance is critical and you can optimize the method choice
///
/// # Examples
///
/// ## Basic Usage
///
/// ```
/// use thales::numerical::SmartNumericalSolver;
/// use thales::ast::{Equation, Expression, Variable};
///
/// // Solve x^3 = 27 (we don't know a good initial guess)
/// let equation = Equation::new(
///     "cubic",
///     Expression::Power(
///         Box::new(Expression::Variable(Variable::new("x"))),
///         Box::new(Expression::Integer(3))
///     ),
///     Expression::Integer(27)
/// );
///
/// let solver = SmartNumericalSolver::with_default_config();
/// let (solution, _path) = solver.solve(&equation, &Variable::new("x")).unwrap();
///
/// assert!((solution.value - 3.0).abs() < 1e-6);
/// println!("∛27 = {} (method chosen automatically)", solution.value);
/// ```
///
/// ## With Known Interval
///
/// ```
/// use thales::numerical::SmartNumericalSolver;
/// use thales::ast::{Equation, Expression, Variable};
///
/// // Solve x^2 = 5, we know root is between 2 and 3
/// let equation = Equation::new(
///     "sqrt5",
///     Expression::Power(
///         Box::new(Expression::Variable(Variable::new("x"))),
///         Box::new(Expression::Integer(2))
///     ),
///     Expression::Integer(5)
/// );
///
/// let solver = SmartNumericalSolver::with_default_config();
/// // Provide interval for guaranteed convergence
/// let (solution, _) = solver.solve_with_interval(
///     &equation,
///     &Variable::new("x"),
///     (2.0, 3.0)
/// ).unwrap();
///
/// assert!((solution.value - 2.236067977).abs() < 1e-6);
/// ```
///
/// ## With Initial Guess
///
/// ```
/// use thales::numerical::{SmartNumericalSolver, NumericalConfig};
/// use thales::ast::{Equation, Expression, Variable};
///
/// // Provide initial guess to use fast Newton-Raphson first
/// let config = NumericalConfig {
///     initial_guess: Some(2.0),
///     ..Default::default()
/// };
///
/// let equation = Equation::new(
///     "sqrt5",
///     Expression::Power(
///         Box::new(Expression::Variable(Variable::new("x"))),
///         Box::new(Expression::Integer(2))
///     ),
///     Expression::Integer(5)
/// );
///
/// let solver = SmartNumericalSolver::new(config);
/// let (solution, _) = solver.solve(&equation, &Variable::new("x")).unwrap();
///
/// assert!((solution.value - 2.236067977).abs() < 1e-6);
/// // Will use Newton-Raphson since initial_guess is provided
/// ```
#[derive(Debug)]
pub struct SmartNumericalSolver {
    config: NumericalConfig,
}

impl SmartNumericalSolver {
    /// Creates a new smart numerical solver with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Numerical configuration (tolerance, iterations, initial guess, etc.)
    pub fn new(config: NumericalConfig) -> Self {
        Self { config }
    }

    /// Creates a new smart numerical solver with default configuration.
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
    /// Creates a new evaluator with no variables defined.
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    /// Creates a new evaluator with predefined variable values.
    ///
    /// # Arguments
    ///
    /// * `variables` - Map of variables to their numeric values
    pub fn with_variables(variables: HashMap<Variable, f64>) -> Self {
        Self { variables }
    }

    /// Sets or updates the value of a variable.
    ///
    /// # Arguments
    ///
    /// * `var` - Variable to set
    /// * `value` - Numeric value to assign
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
/// Searches for an interval [a, b] where the function changes sign, which
/// guarantees the existence of at least one root (by the Intermediate Value
/// Theorem for continuous functions).
///
/// # Algorithm
///
/// The function uses an expanding search strategy:
/// 1. Tries intervals of increasing size: 1, 10, 100, 1000 (up to max_range)
/// 2. For each size, tries different offsets from the center point
/// 3. Returns the first interval where f(a) and f(b) have opposite signs
///
/// This approach balances thoroughness with efficiency, checking common
/// ranges first before expanding to larger intervals.
///
/// # Arguments
///
/// * `expr` - The function expression to evaluate (should be in f(x) = 0 form)
/// * `variable` - The variable to solve for
/// * `center` - The center point around which to search for a bracket
/// * `max_range` - Maximum distance from center to search (prevents unbounded searches)
///
/// # Returns
///
/// * `Some((a, b))` - An interval where f(a) and f(b) have opposite signs
/// * `None` - No bracketing interval found within the search range
///
/// # Usage
///
/// This helper function is used internally by `SmartNumericalSolver` to
/// automatically find suitable intervals for bisection when the user hasn't
/// provided an initial guess or interval. It's particularly useful for:
/// - Unknown function behavior
/// - Automated solving without manual interval specification
/// - Fallback when Newton-Raphson fails
///
/// # Example Context
///
/// ```ignore
/// // Used internally by SmartNumericalSolver
/// let f = Expression::Binary(BinaryOp::Sub, lhs, rhs); // f(x) = 0 form
/// if let Some((a, b)) = bracket_root(&f, &var, 1.0, 10000.0) {
///     // Found bracketing interval, can use bisection method
///     let bisection = BisectionMethod::new(config);
///     bisection.solve(equation, variable, (a, b))
/// }
/// ```
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

/// Auxiliary state carried between iterations of Brent's method.
struct BrentState {
    /// Previous best estimate (before last swap)
    c: f64,
    /// f(c)
    fc: f64,
    /// Step from the iteration before last
    d: f64,
    /// Step from last iteration
    e: f64,
    /// Was the last accepted step a bisection?
    mflag: bool,
}

/// Build the residual expression `f(x) = lhs - rhs` and a point-evaluator closure.
///
/// The returned closure owns all data it needs (cloned from the inputs).
fn brent_make_eval(
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

/// Validate and orient the initial bracket for Brent's method.
///
/// Returns `(a, b, fa, fb)` with the guarantee that `|f(b)| <= |f(a)|`.
/// Errors if `f(a)` and `f(b)` have the same sign (no root bracketed).
fn brent_init_bracket(
    interval: (f64, f64),
    eval: &impl Fn(f64) -> NumericalResult<f64>,
) -> NumericalResult<(f64, f64, f64, f64)> {
    let (mut a, mut b) = interval;
    let mut fa = eval(a)?;
    let mut fb = eval(b)?;
    if fa * fb > 0.0 {
        return Err(NumericalError::Other(format!(
            "Brent's method requires f(a) and f(b) to have opposite signs. \
             f({}) = {}, f({}) = {}",
            a, fa, b, fb
        )));
    }
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }
    Ok((a, b, fa, fb))
}

/// Compute and validate the next Brent step, falling back to bisection when needed.
///
/// Returns `(step_value, used_bisection)`.
fn brent_next_point(a: f64, b: f64, fa: f64, fb: f64, st: &BrentState, tol: f64) -> (f64, bool) {
    let s = brent_interpolation_step(a, b, st.c, fa, fb, st.fc, tol);
    let bisect_mid = (a + b) / 2.0;
    let lo = (3.0 * a + b) / 4.0;
    let hi = b;
    let in_bracket = if lo <= hi {
        lo <= s && s <= hi
    } else {
        hi <= s && s <= lo
    };
    let reject = !in_bracket
        || (st.mflag && (s - b).abs() >= (b - st.c).abs() / 2.0)
        || (!st.mflag && (s - b).abs() >= st.e.abs() / 2.0)
        || (st.mflag && (b - st.c).abs() < tol)
        || (!st.mflag && st.e.abs() < tol);
    if reject {
        (bisect_mid, true)
    } else {
        (s, false)
    }
}

/// Update the bracket and `BrentState` after evaluating the accepted step.
///
/// Maintains `f(a) * f(b) <= 0` and `|f(b)| <= |f(a)|`.
fn brent_update_bracket(
    a: &mut f64,
    b: &mut f64,
    st: &mut BrentState,
    fa: &mut f64,
    fb: &mut f64,
    s_final: f64,
    fs: f64,
) {
    st.d = st.e;
    st.e = *b - *a;
    st.c = *b;
    st.fc = *fb;
    if *fa * fs < 0.0 {
        *b = s_final;
        *fb = fs;
    } else {
        *a = s_final;
        *fa = fs;
    }
    if fa.abs() < fb.abs() {
        std::mem::swap(a, b);
        std::mem::swap(fa, fb);
    }
}

/// Compute the raw interpolation candidate (IQI or secant) for Brent's method.
///
/// Returns a value that may lie outside the bracket; the caller (`brent_next_point`)
/// decides whether to use it or fall back to bisection.
fn brent_interpolation_step(a: f64, b: f64, c: f64, fa: f64, fb: f64, fc: f64, tol: f64) -> f64 {
    if (fa - fc).abs() > tol && (fb - fc).abs() > tol {
        let r = fb / fc;
        let s = fb / fa;
        let t = fa / fc;
        let p = s * (t * (r - t) * (c - b) - (1.0 - r) * (b - a));
        let q = (t - 1.0) * (r - 1.0) * (s - 1.0);
        if q.abs() > tol {
            return b + p / q;
        }
    }
    let denom = fb - fa;
    if denom.abs() > tol {
        b - fb * (b - a) / denom
    } else {
        b + 2.0 * (b - a) // signal: outside bracket, force bisection
    }
}

// TODO: Add support for interval arithmetic
// TODO: Add automatic differentiation
// TODO: Add support for complex-valued functions
// TODO: Add parallel evaluation for gradient computation
// TODO: Add sensitivity analysis
// TODO: Add uncertainty propagation
