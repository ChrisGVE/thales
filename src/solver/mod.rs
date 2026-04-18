//! Algebraic equation solver with symbolic manipulation.
//!
//! This module provides a comprehensive framework for solving algebraic equations
//! symbolically. It supports linear, quadratic, polynomial, and transcendental
//! equations, with automatic method selection via the [`SmartSolver`].
//!
//! # Overview
//!
//! The solver works by:
//! 1. Analyzing the equation structure to determine appropriate solving method
//! 2. Applying symbolic transformations to isolate the target variable
//! 3. Simplifying and evaluating the result
//! 4. Recording all steps in a [`ResolutionPath`] for display
//!
//! # Solver Types
//!
//! - [`LinearSolver`]: Solves equations of the form `ax + b = c`
//! - [`QuadraticSolver`]: Solves equations with x² terms (not yet implemented)
//! - [`PolynomialSolver`]: General polynomial equations (not yet implemented)
//! - [`TranscendentalSolver`]: Equations with sin, cos, tan, exp, ln, log functions
//! - [`SmartSolver`]: Automatically selects the appropriate solver
//!
//! # Solution Types
//!
//! Solutions can be:
//! - [`Solution::Unique`]: Single solution (e.g., x = 5)
//! - [`Solution::Multiple`]: Discrete solutions (e.g., x = 2 or x = -2)
//! - [`Solution::Parametric`]: Solution depends on other variables
//! - [`Solution::None`]: No solution exists (inconsistent equation)
//! - [`Solution::Infinite`]: All values satisfy the equation (identity)
//!
//! # Examples
//!
//! ## Basic Linear Equation
//!
//! ```
//! use thales::solver::{LinearSolver, Solver};
//! use thales::ast::{Equation, Expression, Variable, BinaryOp};
//!
//! // Solve: 2x + 3 = 11
//! let x = Expression::Variable(Variable::new("x"));
//! let left = Expression::Binary(
//!     BinaryOp::Add,
//!     Box::new(Expression::Binary(
//!         BinaryOp::Mul,
//!         Box::new(Expression::Integer(2)),
//!         Box::new(x),
//!     )),
//!     Box::new(Expression::Integer(3)),
//! );
//! let right = Expression::Integer(11);
//! let equation = Equation::new("linear_eq", left, right);
//!
//! let solver = LinearSolver::new();
//! let (solution, path) = solver.solve(&equation, &Variable::new("x")).unwrap();
//!
//! // Solution is x = 4
//! # use thales::solver::Solution;
//! # match solution {
//! #     Solution::Unique(expr) => {
//! #         assert_eq!(expr.evaluate(&std::collections::HashMap::new()), Some(4.0));
//! #     }
//! #     _ => panic!("Expected unique solution"),
//! # }
//! ```
//!
//! ## Using SmartSolver
//!
//! ```
//! use thales::solver::{SmartSolver, Solver};
//! use thales::ast::{Equation, Expression, Variable, BinaryOp};
//!
//! // SmartSolver automatically picks the right method
//! let solver = SmartSolver::new();
//!
//! // Solve: 3x = 12
//! let x = Expression::Variable(Variable::new("x"));
//! let left = Expression::Binary(
//!     BinaryOp::Mul,
//!     Box::new(Expression::Integer(3)),
//!     Box::new(x),
//! );
//! let equation = Equation::new("simple", left, Expression::Integer(12));
//!
//! let (solution, _path) = solver.solve(&equation, &Variable::new("x")).unwrap();
//! // Solution is x = 4
//! ```
//!
//! ## High-Level API with Known Values
//!
//! ```
//! use thales::solver::solve_for;
//! use thales::ast::{Equation, Expression, Variable, BinaryOp};
//! use std::collections::HashMap;
//!
//! // Solve: ax + b = c for x, given a=2, b=3, c=11
//! let a = Expression::Variable(Variable::new("a"));
//! let x = Expression::Variable(Variable::new("x"));
//! let b = Expression::Variable(Variable::new("b"));
//! let c = Expression::Variable(Variable::new("c"));
//!
//! let ax = Expression::Binary(BinaryOp::Mul, Box::new(a), Box::new(x));
//! let left = Expression::Binary(BinaryOp::Add, Box::new(ax), Box::new(b));
//! let equation = Equation::new("parametric", left, c);
//!
//! let mut known = HashMap::new();
//! known.insert("a".to_string(), 2.0);
//! known.insert("b".to_string(), 3.0);
//! known.insert("c".to_string(), 11.0);
//!
//! let path = solve_for(&equation, "x", &known).unwrap();
//! // Result is x = 4.0
//! # assert_eq!(path.result.evaluate(&HashMap::new()), Some(4.0));
//! ```

mod coeff;
mod cramer;
mod gauss;
mod helpers;
pub mod linear;
pub mod linear_system;
mod lu_exact;
pub mod ode_classifier;
pub mod ode_solver;
pub mod polynomial;
pub mod quadratic;
pub mod symbolic_isolation;
pub mod system;
pub mod transcendental;
pub mod types;

// Re-export all public types for backward compatibility
pub use linear::LinearSolver;
pub use linear_system::LinearSystem;
pub use ode_classifier::{
    classify_first_order, classify_second_order, ODEClassification, ODELinearity, ODEOrder, ODEType,
};
pub use ode_solver::{
    solve_ode_first_order, solve_ode_from_latex, solve_ode_from_text, solve_ode_second_order,
    OdeSolver,
};
pub use polynomial::PolynomialSolver;
pub use quadratic::QuadraticSolver;
pub use system::{SystemSolution, SystemSolver};
pub use transcendental::TranscendentalSolver;
pub use types::{Constraint, Solution, SolverError, SolverResult, SymbolicFailureReason};

use crate::ast::{BinaryOp, Equation, Expression, Variable};
use crate::numerical::SmartNumericalSolver;
use crate::resolution_path::{Operation, ResolutionPath, ResolutionPathBuilder, ResolutionStep};
use helpers::{
    evaluate_constants, get_polynomial_degree, is_polynomial_expression, substitute_values,
};
use std::collections::HashMap;

/// Trait for equation solvers.
///
/// Implementors of this trait provide methods to solve equations symbolically.
/// Each solver specializes in a particular type of equation (linear, quadratic, etc.).
///
/// # Design
///
/// The trait has two methods:
/// - [`can_solve`](Solver::can_solve): Quick check if equation is suitable for this solver
/// - [`solve`](Solver::solve): Perform the actual solving and return solution with steps
///
/// # Examples
///
/// ```
/// use thales::solver::{Solver, LinearSolver};
/// use thales::ast::{Equation, Expression, Variable, BinaryOp};
///
/// let solver = LinearSolver::new();
///
/// // Build equation: 5x = 20
/// let x = Expression::Variable(Variable::new("x"));
/// let left = Expression::Binary(
///     BinaryOp::Mul,
///     Box::new(Expression::Integer(5)),
///     Box::new(x),
/// );
/// let eq = Equation::new("test", left, Expression::Integer(20));
///
/// // Check if solver can handle it
/// assert!(solver.can_solve(&eq));
///
/// // Solve it
/// let (solution, path) = solver.solve(&eq, &Variable::new("x")).unwrap();
/// // Solution is x = 4
/// ```
pub trait Solver {
    /// Solve an equation for the specified variable.
    ///
    /// Returns the solution(s) and a [`ResolutionPath`] showing the steps taken.
    fn solve(
        &self,
        equation: &Equation,
        variable: &Variable,
    ) -> SolverResult<(Solution, ResolutionPath)>;

    /// Check if this solver can handle the given equation.
    ///
    /// This is a fast pre-check that examines the equation structure without
    /// actually solving it. It's used by [`SmartSolver`] to select the
    /// appropriate solver.
    fn can_solve(&self, equation: &Equation) -> bool;
}

#[derive(Debug)]
/// Automatic solver dispatcher that selects the appropriate solving method.
///
/// `SmartSolver` examines the equation structure and dispatches to the most
/// suitable specialized solver. This eliminates the need to manually choose
/// between linear, quadratic, polynomial, or transcendental solvers.
///
/// # Priority Order
///
/// The solver tries methods in this priority order:
/// 1. **Linear** ([`LinearSolver`]): Fastest, handles equations like `ax + b = c`
/// 2. **Quadratic** ([`QuadraticSolver`]): Equations with x² terms
/// 3. **Polynomial** ([`PolynomialSolver`]): General polynomial equations
/// 4. **Transcendental** ([`TranscendentalSolver`]): Equations with sin, cos, exp, ln, log
/// 5. **ODE** ([`OdeSolver`]): Ordinary differential equations (future: when parser supports derivatives)
///
/// # Examples
///
/// ## Linear Equation
///
/// ```
/// use thales::solver::{SmartSolver, Solver, Solution};
/// use thales::ast::{Equation, Expression, Variable, BinaryOp};
///
/// let solver = SmartSolver::new();
///
/// // Solve: 2x + 3 = 11
/// let x = Expression::Variable(Variable::new("x"));
/// let two_x = Expression::Binary(
///     BinaryOp::Mul,
///     Box::new(Expression::Integer(2)),
///     Box::new(x.clone()),
/// );
/// let left = Expression::Binary(
///     BinaryOp::Add,
///     Box::new(two_x),
///     Box::new(Expression::Integer(3)),
/// );
/// let equation = Equation::new("example", left, Expression::Integer(11));
///
/// let (solution, path) = solver.solve(&equation, &Variable::new("x")).unwrap();
///
/// // SmartSolver automatically selected LinearSolver
/// match solution {
///     Solution::Unique(expr) => {
///         assert_eq!(expr.evaluate(&std::collections::HashMap::new()), Some(4.0));
///     }
///     _ => panic!("Expected unique solution"),
/// }
/// ```
///
/// ## Transcendental Equation
///
/// ```no_run
/// use thales::solver::{SmartSolver, Solver, Solution};
/// use thales::ast::{Equation, Expression, Variable, Function};
///
/// let solver = SmartSolver::new();
///
/// // Solve: sin(x) = 0.5
/// let x = Expression::Variable(Variable::new("x"));
/// let sin_x = Expression::Function(Function::Sin, vec![x]);
/// let equation = Equation::new("trig", sin_x, Expression::Float(0.5));
///
/// let (solution, path) = solver.solve(&equation, &Variable::new("x")).unwrap();
///
/// // SmartSolver automatically selected TranscendentalSolver
/// match solution {
///     Solution::Unique(expr) => {
///         // expr contains asin(0.5)
///         let result = expr.evaluate(&std::collections::HashMap::new()).unwrap();
///         assert!((result - 0.5236).abs() < 0.001); // π/6 radians
///     }
///     _ => panic!("Expected unique solution"),
/// }
/// ```
///
/// ## Error Handling
///
/// ```
/// use thales::solver::{SmartSolver, Solver, SolverError};
/// use thales::ast::{Equation, Expression, Variable};
///
/// let solver = SmartSolver::new();
///
/// // Variable not in equation
/// let equation = Equation::new("bad", Expression::Integer(0), Expression::Integer(5));
/// let result = solver.solve(&equation, &Variable::new("x"));
///
/// // Since x doesn't appear in the equation, solver cannot handle it
/// assert!(result.is_err());
/// match result {
///     Err(SolverError::CannotSolve(_)) | Err(SolverError::UnsupportedEquationType) => {
///         // Expected - x not in equation or equation not supported
///     }
///     _ => panic!("Expected CannotSolve or UnsupportedEquationType error"),
/// }
/// ```
///
/// # See Also
///
/// - [`solve_for`]: High-level API that uses `SmartSolver` and handles value substitution
/// - [`Solver`]: Base trait implemented by all solvers
/// - [`LinearSolver`], [`QuadraticSolver`], [`PolynomialSolver`], [`TranscendentalSolver`]: Specialized solvers
pub struct SmartSolver {
    linear: LinearSolver,
    quadratic: QuadraticSolver,
    polynomial: PolynomialSolver,
    transcendental: TranscendentalSolver,
    ode: OdeSolver,
}

impl SmartSolver {
    /// Creates a new smart solver with all specialized solvers initialized.
    pub fn new() -> Self {
        Self {
            linear: LinearSolver::new(),
            quadratic: QuadraticSolver::new(),
            polynomial: PolynomialSolver::new(),
            transcendental: TranscendentalSolver::new(),
            ode: OdeSolver::new(),
        }
    }
}

impl Default for SmartSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver for SmartSolver {
    fn solve(
        &self,
        equation: &Equation,
        variable: &Variable,
    ) -> SolverResult<(Solution, ResolutionPath)> {
        // Only skip symbolic isolation for equations that are polynomial of
        // degree ≥ 2 in the target variable and have a negative discriminant
        // (complex roots).  For real-root quadratics the symbolic isolation
        // path gives the expected Unique result; for complex-root cases it
        // would produce an unevaluable expression like (-1)^(1/2).
        let combined_deg = crate::ast::Expression::Binary(
            crate::ast::BinaryOp::Sub,
            Box::new(equation.left.clone()),
            Box::new(equation.right.clone()),
        )
        .simplify();
        let var_degree = if is_polynomial_expression(&combined_deg) {
            get_polynomial_degree(&combined_deg, &variable.name)
        } else {
            0
        };

        // Compute discriminant only for true quadratics in the target variable.
        let has_complex_roots = if var_degree == 2 {
            use helpers::extract_quadratic_coefficients;
            let (a, b, c) = extract_quadratic_coefficients(&combined_deg, &variable.name);
            a.abs() > 1e-15 && (b * b - 4.0 * a * c) < -1e-15
        } else {
            false
        };

        // Try general symbolic isolation first — it handles arbitrary
        // rearrangements that the specialized solvers miss.
        // Skip it when the equation has complex roots: symbolic isolation
        // returns a single Unique result and cannot represent complex pairs.
        if !has_complex_roots {
            let path_builder = ResolutionPathBuilder::new(equation.left.clone());
            let lhs_arc = crate::numeric::compile::compile(&equation.left);
            let rhs_arc = crate::numeric::compile::compile(&equation.right);
            if let Ok((result_expr, builder)) =
                symbolic_isolation::symbolic_isolate(&lhs_arc, &rhs_arc, variable, path_builder)
            {
                let path = builder.finish(result_expr.clone());
                return Ok((Solution::Unique(result_expr), path));
            }
        }

        // Fall back: linear -> quadratic -> polynomial -> transcendental
        let symbolic_result = if self.linear.can_solve(equation) {
            Some(self.linear.solve(equation, variable))
        } else if self.quadratic.can_solve(equation) {
            Some(self.quadratic.solve(equation, variable))
        } else if self.polynomial.can_solve(equation) {
            Some(self.polynomial.solve(equation, variable))
        } else if self.transcendental.can_solve(equation) {
            Some(self.transcendental.solve(equation, variable))
        } else if self.ode.can_solve(equation) {
            Some(self.ode.solve(equation, variable))
        } else {
            None
        };

        // If a specialized solver succeeded, return its result
        if let Some(Ok(result)) = symbolic_result {
            return Ok(result);
        }

        // All symbolic methods exhausted — attempt numerical handoff
        let symbolic_error = symbolic_result
            .and_then(|r| r.err())
            .unwrap_or(SolverError::UnsupportedEquationType);

        // Guard: if the variable doesn't appear in the equation at all,
        // numerical solving is meaningless — return the symbolic error.
        if !equation.left.contains_variable(&variable.name)
            && !equation.right.contains_variable(&variable.name)
        {
            return Err(symbolic_error);
        }

        let failure_reason = analyze_symbolic_failure(equation, variable);
        let recommended = recommend_numerical_method(equation);

        // Build a path recording the handoff
        let mut path = ResolutionPath::new(equation.left.clone());
        path.add_step(ResolutionStep::new(
            Operation::SymbolicToNumericalHandoff {
                reason: format!("{}", failure_reason),
                recommended_method: recommended.clone(),
            },
            format!(
                "Symbolic methods exhausted: {}. Switching to {}.",
                failure_reason, recommended
            ),
            equation.left.clone(),
        ));

        // Try numerical fallback
        match try_numerical_solve(equation, variable) {
            Ok((num_solution, num_path)) => {
                // Append numerical steps to path
                for step in &num_path.steps {
                    path.add_step(step.clone());
                }
                let result_expr = Expression::Float(num_solution);
                path.set_result(result_expr.clone());
                Ok((Solution::Unique(result_expr), path))
            }
            Err(_) => {
                // Numerical also failed — return the original symbolic error
                Err(symbolic_error)
            }
        }
    }

    fn can_solve(&self, equation: &Equation) -> bool {
        self.linear.can_solve(equation)
            || self.quadratic.can_solve(equation)
            || self.polynomial.can_solve(equation)
            || self.transcendental.can_solve(equation)
            || self.ode.can_solve(equation)
    }
}

// ============================================================================
// Symbolic-to-Numerical Handoff Helpers
// ============================================================================

/// Count the number of times a variable appears in an expression.
fn count_variable_occurrences(expr: &Expression, var_name: &str) -> usize {
    match expr {
        Expression::Variable(v) if v.name == var_name => 1,
        Expression::Variable(_)
        | Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Rational(_)
        | Expression::Complex(_)
        | Expression::Constant(_) => 0,
        Expression::Unary(_, inner) => count_variable_occurrences(inner, var_name),
        Expression::Binary(_, left, right) => {
            count_variable_occurrences(left, var_name) + count_variable_occurrences(right, var_name)
        }
        Expression::Function(_, args) => args
            .iter()
            .map(|a| count_variable_occurrences(a, var_name))
            .sum(),
        Expression::Power(base, exp) => {
            count_variable_occurrences(base, var_name) + count_variable_occurrences(exp, var_name)
        }
    }
}

/// Check whether the variable appears inside a transcendental function.
fn variable_in_transcendental(expr: &Expression, var_name: &str) -> bool {
    match expr {
        Expression::Function(_, args) => args.iter().any(|a| a.contains_variable(var_name)),
        Expression::Unary(_, inner) => variable_in_transcendental(inner, var_name),
        Expression::Binary(_, left, right) => {
            variable_in_transcendental(left, var_name)
                || variable_in_transcendental(right, var_name)
        }
        Expression::Power(base, exp) => {
            variable_in_transcendental(base, var_name) || variable_in_transcendental(exp, var_name)
        }
        _ => false,
    }
}

/// Check whether the variable appears in an algebraic (non-transcendental) position.
fn variable_in_algebraic(expr: &Expression, var_name: &str) -> bool {
    match expr {
        Expression::Variable(v) if v.name == var_name => true,
        Expression::Variable(_)
        | Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Rational(_)
        | Expression::Complex(_)
        | Expression::Constant(_) => false,
        Expression::Function(_, _) => false, // Skip into function args
        Expression::Unary(_, inner) => variable_in_algebraic(inner, var_name),
        Expression::Binary(_, left, right) => {
            variable_in_algebraic(left, var_name) || variable_in_algebraic(right, var_name)
        }
        Expression::Power(base, exp) => {
            // x^n where x contains var is algebraic; n^x is transcendental
            if base.contains_variable(var_name) && !exp.contains_variable(var_name) {
                true
            } else {
                variable_in_algebraic(base, var_name)
            }
        }
    }
}

/// Check whether the expression mixes algebraic and transcendental uses of the variable.
fn has_transcendental_mixing(expr: &Expression, var_name: &str) -> bool {
    variable_in_transcendental(expr, var_name) && variable_in_algebraic(expr, var_name)
}

/// Analyze why symbolic solving failed and return a structured reason.
fn analyze_symbolic_failure(equation: &Equation, variable: &Variable) -> SymbolicFailureReason {
    let var_name = &variable.name;
    let expr = Expression::Binary(
        BinaryOp::Sub,
        Box::new(equation.left.clone()),
        Box::new(equation.right.clone()),
    );

    let occurrences = count_variable_occurrences(&expr, var_name);

    if occurrences == 0 {
        return SymbolicFailureReason::NonIsolable {
            reason: "Variable not found in equation".to_string(),
            occurrences: 0,
        };
    }

    // Check for mixed algebraic-transcendental usage (e.g. x * exp(x) = 5)
    if has_transcendental_mixing(&expr, var_name) {
        return SymbolicFailureReason::Transcendental {
            equation_type: "mixed algebraic-transcendental".to_string(),
        };
    }

    // Pure transcendental usage
    if variable_in_transcendental(&expr, var_name) {
        return SymbolicFailureReason::Transcendental {
            equation_type: "transcendental".to_string(),
        };
    }

    if occurrences > 1 {
        return SymbolicFailureReason::NonIsolable {
            reason: format!(
                "Variable appears {} times in non-combinable positions",
                occurrences
            ),
            occurrences,
        };
    }

    // Single occurrence, algebraic, but still unsolvable — generic fallback
    SymbolicFailureReason::Transcendental {
        equation_type: "unknown".to_string(),
    }
}

/// Recommend a numerical method based on equation characteristics.
fn recommend_numerical_method(_equation: &Equation) -> String {
    // Simple heuristic: Newton-Raphson for smooth functions
    "Newton-Raphson".to_string()
}

/// Attempt to solve the equation numerically using the smart numerical solver.
///
/// Returns the solution value and the resolution path from the numerical solver.
fn try_numerical_solve(
    equation: &Equation,
    variable: &Variable,
) -> Result<(f64, ResolutionPath), SolverError> {
    let solver = SmartNumericalSolver::with_default_config();
    match solver.solve(equation, variable) {
        Ok((solution, mut path)) => {
            if solution.converged {
                // Record convergence information in the resolution path
                let method_name = infer_numerical_method(&path);
                path.add_step(ResolutionStep::new(
                    Operation::NumericalConverged {
                        method: method_name,
                        iterations: solution.iterations,
                        final_error: solution.residual,
                    },
                    format!(
                        "Converged to x = {:.8} in {} iterations (error: {:.2e})",
                        solution.value, solution.iterations, solution.residual
                    ),
                    Expression::Float(solution.value),
                ));
                Ok((solution.value, path))
            } else {
                Err(SolverError::Other(
                    "Numerical solver did not converge".to_string(),
                ))
            }
        }
        Err(e) => Err(SolverError::Other(format!(
            "Numerical solving failed: {}",
            e
        ))),
    }
}

/// Infer which numerical method was used by inspecting the resolution path steps.
fn infer_numerical_method(path: &ResolutionPath) -> String {
    for step in &path.steps {
        let explanation = step.explanation.to_lowercase();
        if explanation.contains("newton") {
            return "Newton-Raphson".to_string();
        }
        if explanation.contains("bisection") {
            return "Bisection".to_string();
        }
        if explanation.contains("brent") {
            return "Brent".to_string();
        }
        if explanation.contains("secant") {
            return "Secant".to_string();
        }
    }
    "Numerical".to_string()
}

// ============================================================================
// High-Level API
// ============================================================================

/// Solve an equation for a specific variable with known values substituted.
///
/// This is the primary high-level API for equation solving. It combines symbolic
/// solving with numeric evaluation in three steps:
///
/// 1. **Symbolic solving**: Uses [`SmartSolver`] to solve for the target variable
/// 2. **Value substitution**: Replaces known variables with their numeric values
/// 3. **Simplification**: Evaluates constants and simplifies the result
///
/// # Arguments
///
/// * `equation` - The equation to solve (e.g., `ax + b = c`)
/// * `target` - Name of the variable to solve for (e.g., `"x"`)
/// * `known_values` - HashMap mapping variable names to their numeric values
///
/// # Returns
///
/// A [`ResolutionPath`] containing all solving steps and the final result.
///
/// # Errors
///
/// Returns [`SolverError`] if solving fails.
///
/// # Examples
///
/// ```
/// use thales::solver::solve_for;
/// use thales::ast::{Equation, Expression, Variable, BinaryOp};
/// use std::collections::HashMap;
///
/// // Solve: ax + b = c for x, given a=2, b=3, c=11
/// let a = Expression::Variable(Variable::new("a"));
/// let x = Expression::Variable(Variable::new("x"));
/// let b = Expression::Variable(Variable::new("b"));
/// let c = Expression::Variable(Variable::new("c"));
///
/// let ax = Expression::Binary(BinaryOp::Mul, Box::new(a), Box::new(x));
/// let left = Expression::Binary(BinaryOp::Add, Box::new(ax), Box::new(b));
/// let equation = Equation::new("linear", left, c);
///
/// let mut known = HashMap::new();
/// known.insert("a".to_string(), 2.0);
/// known.insert("b".to_string(), 3.0);
/// known.insert("c".to_string(), 11.0);
///
/// let path = solve_for(&equation, "x", &known).unwrap();
/// assert_eq!(path.result.evaluate(&HashMap::new()), Some(4.0));
/// ```
pub fn solve_for(
    equation: &Equation,
    target: &str,
    known_values: &HashMap<String, f64>,
) -> Result<ResolutionPath, SolverError> {
    // Create Variable from target string
    let target_var = Variable::new(target);

    // Try solving with SmartSolver
    let solver = SmartSolver::new();
    let (solution, mut path) = solver.solve(equation, &target_var)?;

    // Extract the solution expression
    let solution_expr = match solution {
        Solution::Unique(expr) => expr,
        Solution::Multiple(_) => {
            return Err(SolverError::Other(
                "Multiple solutions not yet supported in solve_for".to_string(),
            ))
        }
        Solution::None => return Err(SolverError::NoSolution),
        Solution::Infinite => return Err(SolverError::InfiniteSolutions),
        Solution::Parametric { .. } => {
            return Err(SolverError::Other(
                "Parametric solutions not yet supported in solve_for".to_string(),
            ))
        }
    };

    // Substitute known values
    if !known_values.is_empty() {
        let substituted = substitute_values(&solution_expr, known_values);
        let simplified = substituted.simplify();
        let evaluated = evaluate_constants(&simplified);

        path.add_step(ResolutionStep::new(
            Operation::Substitute {
                variable: Variable::new("known_values"),
                value: Expression::Integer(0), // Placeholder
            },
            "Substitute known values and evaluate".to_string(),
            evaluated.clone(),
        ));

        path.set_result(evaluated);
    } else {
        path.set_result(solution_expr);
    }

    Ok(path)
}

/// Compute a partial derivative for uncertainty propagation and sensitivity analysis.
///
/// Given an equation defining an output variable in terms of input variables,
/// this function computes the partial derivative ∂output/∂input and evaluates
/// it at the given point.
///
/// # Examples
///
/// ```
/// use thales::ast::{Equation, Expression, Variable, BinaryOp};
/// use thales::solver::compute_partial_derivative;
/// use std::collections::HashMap;
///
/// // Equation: V = l * w * h
/// let l = Expression::Variable(Variable::new("l"));
/// let w = Expression::Variable(Variable::new("w"));
/// let h = Expression::Variable(Variable::new("h"));
/// let v = Expression::Variable(Variable::new("V"));
///
/// let lw = Expression::Binary(BinaryOp::Mul, Box::new(l), Box::new(w));
/// let lwh = Expression::Binary(BinaryOp::Mul, Box::new(lw), Box::new(h));
/// let equation = Equation::new("box_volume", v, lwh);
///
/// let mut values = HashMap::new();
/// values.insert("l".to_string(), 2.0);
/// values.insert("w".to_string(), 3.0);
/// values.insert("h".to_string(), 4.0);
///
/// // ∂V/∂l = w * h = 3 * 4 = 12.0
/// let dv_dl = compute_partial_derivative(&equation, "V", "l", &values).unwrap();
/// assert_eq!(dv_dl, 12.0);
/// ```
pub fn compute_partial_derivative(
    equation: &Equation,
    output_var: &str,
    input_var: &str,
    values: &HashMap<String, f64>,
) -> Result<f64, SolverError> {
    // Get the expression for the output variable
    let output_expr = if let Expression::Variable(v) = &equation.left {
        if v.name == output_var {
            &equation.right
        } else if let Expression::Variable(v2) = &equation.right {
            if v2.name == output_var {
                &equation.left
            } else {
                return Err(SolverError::CannotSolve(format!(
                    "Output variable '{}' not found in equation",
                    output_var
                )));
            }
        } else {
            return Err(SolverError::CannotSolve(format!(
                "Output variable '{}' not found in equation",
                output_var
            )));
        }
    } else if let Expression::Variable(v) = &equation.right {
        if v.name == output_var {
            &equation.left
        } else {
            return Err(SolverError::CannotSolve(format!(
                "Output variable '{}' not found in equation",
                output_var
            )));
        }
    } else {
        return Err(SolverError::CannotSolve(
            "Equation does not have output variable isolated".to_string(),
        ));
    };

    // Compute the derivative symbolically
    let derivative_expr = output_expr.differentiate(input_var);

    // Simplify the derivative
    let simplified = derivative_expr.simplify();

    // Evaluate the derivative at the given values
    simplified.evaluate(values).ok_or_else(|| {
        SolverError::Other("Failed to evaluate derivative - missing or invalid values".to_string())
    })
}

/// Compute all partial derivatives for complete uncertainty propagation.
///
/// # Examples
///
/// ```
/// use thales::ast::{Equation, Expression, Variable, BinaryOp};
/// use thales::solver::compute_all_partial_derivatives;
/// use std::collections::HashMap;
///
/// // Equation: V = l * w * h
/// let l = Expression::Variable(Variable::new("l"));
/// let w = Expression::Variable(Variable::new("w"));
/// let h = Expression::Variable(Variable::new("h"));
/// let v = Expression::Variable(Variable::new("V"));
///
/// let lw = Expression::Binary(BinaryOp::Mul, Box::new(l), Box::new(w));
/// let lwh = Expression::Binary(BinaryOp::Mul, Box::new(lw), Box::new(h));
/// let equation = Equation::new("box_volume", v, lwh);
///
/// let mut values = HashMap::new();
/// values.insert("l".to_string(), 2.0);
/// values.insert("w".to_string(), 3.0);
/// values.insert("h".to_string(), 4.0);
///
/// let input_vars = vec!["l".to_string(), "w".to_string(), "h".to_string()];
/// let derivatives = compute_all_partial_derivatives(
///     &equation, "V", &input_vars, &values
/// ).unwrap();
///
/// assert_eq!(derivatives.get("l").unwrap(), &12.0);
/// assert_eq!(derivatives.get("w").unwrap(), &8.0);
/// assert_eq!(derivatives.get("h").unwrap(), &6.0);
/// ```
pub fn compute_all_partial_derivatives(
    equation: &Equation,
    output_var: &str,
    input_vars: &[String],
    values: &HashMap<String, f64>,
) -> Result<HashMap<String, f64>, SolverError> {
    let mut derivatives = HashMap::new();

    for input_var in input_vars {
        let derivative = compute_partial_derivative(equation, output_var, input_var, values)?;
        derivatives.insert(input_var.clone(), derivative);
    }

    Ok(derivatives)
}

// TODO: Add equation simplification before solving
// TODO: Add symbolic manipulation utilities
// TODO: Add support for inequalities
// TODO: Add support for absolute value equations
// TODO: Add support for piecewise functions
// TODO: Add step-by-step explanation generation

#[cfg(test)]
mod system_solver_tests {
    use super::*;
    use crate::ast::{BinaryOp, Equation, Expression, Variable};

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn add(left: Expression, right: Expression) -> Expression {
        Expression::Binary(BinaryOp::Add, Box::new(left), Box::new(right))
    }

    fn sub(left: Expression, right: Expression) -> Expression {
        Expression::Binary(BinaryOp::Sub, Box::new(left), Box::new(right))
    }

    fn mul(left: Expression, right: Expression) -> Expression {
        Expression::Binary(BinaryOp::Mul, Box::new(left), Box::new(right))
    }

    #[test]
    fn test_2x2_unique_solution() {
        // Solve: x + y = 5, x - y = 1
        // Solution: x = 3, y = 2
        let x = Variable::new("x");
        let y = Variable::new("y");

        let eq1 = Equation::new("eq1", add(var("x"), var("y")), int(5));
        let eq2 = Equation::new("eq2", sub(var("x"), var("y")), int(1));

        let solver = SystemSolver::new();
        let result = solver
            .solve_linear_system(&[eq1, eq2], &[x.clone(), y.clone()])
            .unwrap();

        match result {
            SystemSolution::Unique(sol) => {
                let x_val = sol.get(&x).unwrap();
                let y_val = sol.get(&y).unwrap();

                let empty: HashMap<String, f64> = HashMap::new();
                assert_eq!(x_val.evaluate(&empty), Some(3.0));
                assert_eq!(y_val.evaluate(&empty), Some(2.0));
            }
            _ => panic!("Expected unique solution"),
        }
    }

    #[test]
    fn test_2x2_with_coefficients() {
        // Solve: 2x + 3y = 8, 4x - y = 2
        // Solution: x = 1, y = 2
        let x = Variable::new("x");
        let y = Variable::new("y");

        let eq1 = Equation::new(
            "eq1",
            add(mul(int(2), var("x")), mul(int(3), var("y"))),
            int(8),
        );
        let eq2 = Equation::new("eq2", sub(mul(int(4), var("x")), var("y")), int(2));

        let solver = SystemSolver::new();
        let result = solver
            .solve_linear_system(&[eq1, eq2], &[x.clone(), y.clone()])
            .unwrap();

        match result {
            SystemSolution::Unique(sol) => {
                let x_val = sol.get(&x).unwrap();
                let y_val = sol.get(&y).unwrap();

                let empty: HashMap<String, f64> = HashMap::new();
                assert_eq!(x_val.evaluate(&empty), Some(1.0));
                assert_eq!(y_val.evaluate(&empty), Some(2.0));
            }
            _ => panic!("Expected unique solution"),
        }
    }

    #[test]
    fn test_3x3_unique_solution() {
        // Solve: x + y + z = 6, 2x + y - z = 1, x - y + 2z = 5
        // Solution: x = 1, y = 2, z = 3
        let x = Variable::new("x");
        let y = Variable::new("y");
        let z = Variable::new("z");

        let eq1 = Equation::new("eq1", add(add(var("x"), var("y")), var("z")), int(6));
        let eq2 = Equation::new(
            "eq2",
            sub(add(mul(int(2), var("x")), var("y")), var("z")),
            int(1),
        );
        let eq3 = Equation::new(
            "eq3",
            add(sub(var("x"), var("y")), mul(int(2), var("z"))),
            int(5),
        );

        let solver = SystemSolver::new();
        let result = solver
            .solve_linear_system(&[eq1, eq2, eq3], &[x.clone(), y.clone(), z.clone()])
            .unwrap();

        match result {
            SystemSolution::Unique(sol) => {
                let empty: HashMap<String, f64> = HashMap::new();
                assert_eq!(sol.get(&x).unwrap().evaluate(&empty), Some(1.0));
                assert_eq!(sol.get(&y).unwrap().evaluate(&empty), Some(2.0));
                assert_eq!(sol.get(&z).unwrap().evaluate(&empty), Some(3.0));
            }
            _ => panic!("Expected unique solution"),
        }
    }

    #[test]
    fn test_underdetermined_system() {
        // Solve: x + y = 5 (one equation, two unknowns)
        // Should have infinite solutions
        let x = Variable::new("x");
        let y = Variable::new("y");

        let eq1 = Equation::new("eq1", add(var("x"), var("y")), int(5));

        let solver = SystemSolver::new();
        let result = solver
            .solve_linear_system(&[eq1], &[x.clone(), y.clone()])
            .unwrap();

        match result {
            SystemSolution::Infinite { bound, free } => {
                // One variable should be free
                assert!(!free.is_empty());
                // The other should be bound to an expression
                assert!(!bound.is_empty());
            }
            _ => panic!("Expected infinite solutions"),
        }
    }

    #[test]
    fn test_inconsistent_system() {
        // Solve: x + y = 5, x + y = 6 (no solution)
        let x = Variable::new("x");
        let y = Variable::new("y");

        let eq1 = Equation::new("eq1", add(var("x"), var("y")), int(5));
        let eq2 = Equation::new("eq2", add(var("x"), var("y")), int(6));

        let solver = SystemSolver::new();
        let result = solver
            .solve_linear_system(&[eq1, eq2], &[x.clone(), y.clone()])
            .unwrap();

        assert!(matches!(result, SystemSolution::NoSolution));
    }

    #[test]
    fn test_cramers_rule_2x2() {
        // Same as test_2x2_unique_solution but using Cramer's rule
        let x = Variable::new("x");
        let y = Variable::new("y");

        let eq1 = Equation::new("eq1", add(var("x"), var("y")), int(5));
        let eq2 = Equation::new("eq2", sub(var("x"), var("y")), int(1));

        let solver = SystemSolver::new();
        let result = solver
            .solve_cramers(&[eq1, eq2], &[x.clone(), y.clone()])
            .unwrap();

        match result {
            SystemSolution::Unique(sol) => {
                let empty: HashMap<String, f64> = HashMap::new();
                assert_eq!(sol.get(&x).unwrap().evaluate(&empty), Some(3.0));
                assert_eq!(sol.get(&y).unwrap().evaluate(&empty), Some(2.0));
            }
            _ => panic!("Expected unique solution"),
        }
    }

    #[test]
    fn test_linear_system_struct() {
        // Test LinearSystem::from_equations
        let x = Variable::new("x");
        let y = Variable::new("y");

        let eq1 = Equation::new("eq1", add(var("x"), var("y")), int(5));
        let eq2 = Equation::new("eq2", sub(var("x"), var("y")), int(1));

        let system = LinearSystem::from_equations(&[eq1, eq2], &[x.clone(), y.clone()]).unwrap();

        // Verify matrix dimensions: 2 equations × 2 variables
        assert_eq!(system.num_equations(), 2);
        assert_eq!(system.num_variables(), 2);
    }

    #[test]
    fn test_overdetermined_consistent() {
        // Solve: x + y = 5, x - y = 1, 2x = 6
        // All three equations are consistent with x = 3, y = 2
        let x = Variable::new("x");
        let y = Variable::new("y");

        let eq1 = Equation::new("eq1", add(var("x"), var("y")), int(5));
        let eq2 = Equation::new("eq2", sub(var("x"), var("y")), int(1));
        let eq3 = Equation::new("eq3", mul(int(2), var("x")), int(6));

        let solver = SystemSolver::new();
        let result = solver
            .solve_linear_system(&[eq1, eq2, eq3], &[x.clone(), y.clone()])
            .unwrap();

        match result {
            SystemSolution::Unique(sol) => {
                let empty: HashMap<String, f64> = HashMap::new();
                assert_eq!(sol.get(&x).unwrap().evaluate(&empty), Some(3.0));
                assert_eq!(sol.get(&y).unwrap().evaluate(&empty), Some(2.0));
            }
            _ => panic!("Expected unique solution"),
        }
    }
}

#[cfg(test)]
mod handoff_tests {
    use super::*;
    use crate::ast::{BinaryOp, Equation, Expression, Function, Variable};
    use crate::resolution_path::Operation;

    /// Test that a transcendental equation (x * exp(x) = 5) triggers the
    /// symbolic-to-numerical handoff and produces an approximate solution.
    #[test]
    fn test_transcendental_handoff_x_exp_x() {
        // Equation: x * exp(x) = 5
        let x = Expression::Variable(Variable::new("x"));
        let exp_x = Expression::Function(Function::Exp, vec![x.clone()]);
        let left = Expression::Binary(BinaryOp::Mul, Box::new(x), Box::new(exp_x));
        let right = Expression::Integer(5);
        let equation = Equation::new("transcendental", left, right);

        let solver = SmartSolver::new();
        let result = solver.solve(&equation, &Variable::new("x"));

        // Should succeed via numerical handoff
        assert!(result.is_ok(), "Expected Ok, got {:?}", result);

        let (solution, path) = result.unwrap();

        // Check that the handoff step is recorded in the path
        let has_handoff = path.steps.iter().any(|step| {
            matches!(
                &step.operation,
                Operation::SymbolicToNumericalHandoff { .. }
            )
        });
        assert!(has_handoff, "Expected a SymbolicToNumericalHandoff step");

        // The solution should be numerical (x ≈ 1.3267)
        match solution {
            Solution::Unique(expr) => {
                let val = expr.evaluate(&HashMap::new()).expect("Should evaluate");
                // x * exp(x) = 5 => x ≈ 1.3267 (Lambert W function)
                // Verify by computing x * exp(x) ≈ 5
                let check = val * val.exp();
                assert!(
                    (check - 5.0).abs() < 1e-4,
                    "x*exp(x) should be ≈ 5, got {} (x={})",
                    check,
                    val
                );
            }
            _ => panic!("Expected Unique solution, got {:?}", solution),
        }
    }

    /// Test that a normal linear equation does NOT trigger the handoff —
    /// it should be solved purely symbolically.
    #[test]
    fn test_linear_no_handoff() {
        // Equation: 3x + 6 = 15
        let x = Expression::Variable(Variable::new("x"));
        let three_x =
            Expression::Binary(BinaryOp::Mul, Box::new(Expression::Integer(3)), Box::new(x));
        let left = Expression::Binary(
            BinaryOp::Add,
            Box::new(three_x),
            Box::new(Expression::Integer(6)),
        );
        let right = Expression::Integer(15);
        let equation = Equation::new("linear", left, right);

        let solver = SmartSolver::new();
        let result = solver.solve(&equation, &Variable::new("x"));
        assert!(result.is_ok());

        let (solution, path) = result.unwrap();

        // Should NOT have any handoff step
        let has_handoff = path.steps.iter().any(|step| {
            matches!(
                &step.operation,
                Operation::SymbolicToNumericalHandoff { .. }
            )
        });
        assert!(
            !has_handoff,
            "Linear equation should not trigger numerical handoff"
        );

        // Solution should be x = 3
        match solution {
            Solution::Unique(expr) => {
                let val = expr.evaluate(&HashMap::new()).expect("Should evaluate");
                assert!((val - 3.0).abs() < 1e-10, "Expected x = 3, got {}", val);
            }
            _ => panic!("Expected Unique solution"),
        }
    }
}
