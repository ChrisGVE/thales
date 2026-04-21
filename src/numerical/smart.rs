//! Smart numerical solver that automatically selects the best method.

use crate::ast::{Equation, Expression, Variable};
use crate::numeric::trace::Trace;

use super::{
    bracket_root, BisectionMethod, NewtonRaphson, NumericalConfig, NumericalError, NumericalResult,
    NumericalSolution,
};

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
    ) -> NumericalResult<(NumericalSolution, Trace)> {
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
    ) -> NumericalResult<(NumericalSolution, Trace)> {
        // Prefer bisection when an interval is provided
        let bisection = BisectionMethod::new(self.config.clone());
        bisection.solve(equation, variable, interval)
    }
}
