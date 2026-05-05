//! Bisection method root finder (guaranteed convergence for continuous functions).

use crate::ast::{Equation, Expression, Variable};
use crate::numeric::expr::Expr;
use crate::numeric::trace::{Step, TechniqueTag, Trace};
use std::collections::HashMap;

use super::{NumericalConfig, NumericalError, NumericalResult, NumericalSolution};

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
    ) -> NumericalResult<(NumericalSolution, Trace)> {
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

        let mut trace = Trace::new();

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

        trace.push(
            Step::new(
                TechniqueTag::Bisection,
                format!(
                    "Starting bisection method on interval [{}, {}]. f({}) = {}, f({}) = {}",
                    a, b, a, fa, b, fb
                ),
            )
            .with_output(Expr::float((a + b) / 2.0)),
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
                trace.push(
                    Step::new(
                        TechniqueTag::Bisection,
                        format!(
                            "Converged: |f({})| = {} < {}",
                            c,
                            fc.abs(),
                            self.config.tolerance
                        ),
                    )
                    .with_output(Expr::float(c)),
                );
                break;
            }

            // Check convergence by interval width
            if (b - a) / 2.0 < self.config.tolerance {
                trace.push(
                    Step::new(
                        TechniqueTag::Bisection,
                        format!(
                            "Converged: interval width {} < {}",
                            (b - a) / 2.0,
                            self.config.tolerance
                        ),
                    )
                    .with_output(Expr::float(c)),
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
                trace.push(
                    Step::new(
                        TechniqueTag::Bisection,
                        format!(
                            "Iteration {}: interval = [{}, {}], midpoint = {}, f(midpoint) = {}",
                            iterations, a, b, c, fc
                        ),
                    )
                    .with_output(Expr::float(c)),
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

        Ok((solution, trace))
    }
}
