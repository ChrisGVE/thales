//! Newton-Raphson root finder with symbolic differentiation.

use crate::ast::{Equation, Expression, Variable};
use crate::numeric::expr::Expr;
use crate::numeric::trace::{Step, TechniqueTag, Trace};
use std::collections::HashMap;

use super::{NumericalConfig, NumericalError, NumericalResult, NumericalSolution};

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
    ) -> NumericalResult<(NumericalSolution, Trace)> {
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

        let mut trace = Trace::new();
        trace.push(
            Step::new(
                TechniqueTag::NewtonRaphson,
                format!(
                    "Starting Newton-Raphson method with initial guess x₀ = {}",
                    x
                ),
            )
            .with_output(Expr::float(x)),
        );
        trace.push(Step::new(
            TechniqueTag::NewtonRaphson,
            format!("Using symbolic derivative: f'(x) = {}", f_prime),
        ));

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
                trace.push(
                    Step::new(
                        TechniqueTag::NewtonRaphson,
                        format!(
                            "Converged: |f(x)| = {} < {}",
                            residual, self.config.tolerance
                        ),
                    )
                    .with_output(Expr::float(x)),
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

            // Add step to trace every 10 iterations or at end
            if i % 10 == 0 || i == self.config.max_iterations - 1 {
                trace.push(
                    Step::new(
                        TechniqueTag::NewtonRaphson,
                        format!(
                            "Iteration {}: x = {}, f(x) = {}, f'(x) = {}",
                            iterations, x_next, fx, derivative
                        ),
                    )
                    .with_output(Expr::float(x_next)),
                );
            }

            // Check step size for convergence
            if (x_next - x).abs() < self.config.tolerance {
                x = x_next;
                converged = true;
                trace.push(
                    Step::new(
                        TechniqueTag::NewtonRaphson,
                        format!(
                            "Converged: |Δx| = {} < {}",
                            (x_next - x).abs(),
                            self.config.tolerance
                        ),
                    )
                    .with_output(Expr::float(x)),
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

        Ok((solution, trace))
    }
}
