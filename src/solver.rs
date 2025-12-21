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
//! use mathsolver_core::solver::{LinearSolver, Solver};
//! use mathsolver_core::ast::{Equation, Expression, Variable, BinaryOp};
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
//! # use mathsolver_core::solver::Solution;
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
//! use mathsolver_core::solver::{SmartSolver, Solver};
//! use mathsolver_core::ast::{Equation, Expression, Variable, BinaryOp};
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
//! use mathsolver_core::solver::solve_for;
//! use mathsolver_core::ast::{Equation, Expression, Variable, BinaryOp};
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

use crate::ast::{BinaryOp, Equation, Expression, Variable};
use crate::resolution_path::{Operation, ResolutionPath, ResolutionPathBuilder, ResolutionStep};
use std::collections::HashMap;

/// Error types for equation solving.
///
/// These errors represent different failure modes when attempting to solve
/// an equation symbolically.
///
/// # Examples
///
/// ```
/// use mathsolver_core::solver::{SolverError, LinearSolver, Solver};
/// use mathsolver_core::ast::{Equation, Expression, Variable, BinaryOp};
///
/// // NoSolution: 0 = 5 (inconsistent equation)
/// let eq = Equation::new("bad", Expression::Integer(0), Expression::Integer(5));
/// // This would fail with an error during solving
///
/// // CannotSolve: x² = 4 (not linear, LinearSolver can't handle it)
/// let x = Expression::Variable(Variable::new("x"));
/// let x_squared = Expression::Power(
///     Box::new(x),
///     Box::new(Expression::Integer(2)),
/// );
/// let eq = Equation::new("quadratic", x_squared, Expression::Integer(4));
/// let solver = LinearSolver::new();
/// let result = solver.solve(&eq, &Variable::new("x"));
/// assert!(result.is_err());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum SolverError {
    /// Equation has no solution (inconsistent).
    ///
    /// Example: `0 = 5` or `x + 1 = x + 2`
    NoSolution,

    /// Equation has infinite solutions (identity).
    ///
    /// Example: `x = x` or `2(x + 1) = 2x + 2`
    InfiniteSolutions,

    /// Cannot solve for the given variable with this solver.
    ///
    /// This typically means the equation is too complex for the solver,
    /// or the variable doesn't appear in a solvable form. The message
    /// provides specific details about why solving failed.
    ///
    /// Example: Variable not in equation, or pattern not recognized
    CannotSolve(String),

    /// Equation type is not supported by this solver.
    ///
    /// Example: Trying to solve a quadratic equation with LinearSolver
    UnsupportedEquationType,

    /// Division by zero encountered during solving.
    ///
    /// Example: Attempting to divide by a coefficient that evaluates to zero
    DivisionByZero,

    /// Other error with description.
    ///
    /// Used for errors that don't fit other categories, such as
    /// domain errors (e.g., asin(2)) or not-yet-implemented features.
    Other(String),
}

/// Result type for solver operations.
pub type SolverResult<T> = Result<T, SolverError>;

/// Solution to an equation.
///
/// Represents the different types of solutions an equation can have.
/// Each variant captures a different solution structure.
///
/// # Examples
///
/// ```
/// use mathsolver_core::solver::{Solution, LinearSolver, Solver};
/// use mathsolver_core::ast::{Equation, Expression, Variable, BinaryOp};
///
/// // Unique solution: 2x = 8 → x = 4
/// let x = Expression::Variable(Variable::new("x"));
/// let left = Expression::Binary(
///     BinaryOp::Mul,
///     Box::new(Expression::Integer(2)),
///     Box::new(x),
/// );
/// let eq = Equation::new("simple", left, Expression::Integer(8));
///
/// let solver = LinearSolver::new();
/// let (solution, _) = solver.solve(&eq, &Variable::new("x")).unwrap();
///
/// match solution {
///     Solution::Unique(expr) => {
///         // expr evaluates to 4
///         assert_eq!(expr.evaluate(&std::collections::HashMap::new()), Some(4.0));
///     }
///     _ => panic!("Expected unique solution"),
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Solution {
    /// Single unique solution.
    ///
    /// The equation has exactly one solution, represented as an expression.
    ///
    /// # Examples
    ///
    /// - Linear: `2x + 3 = 11` → `x = 4`
    /// - Transcendental: `sin(x) = 0.5` → `x = asin(0.5)`
    Unique(Expression),

    /// Multiple discrete solutions.
    ///
    /// The equation has a finite number of distinct solutions.
    ///
    /// # Examples
    ///
    /// - Quadratic: `x² - 4 = 0` → `x = 2` or `x = -2`
    /// - Trigonometric: `sin(x) = 0` on [0, 2π] → `x = 0, π, 2π`
    Multiple(Vec<Expression>),

    /// Parametric solution with constraints.
    ///
    /// The solution depends on other variables, with optional constraints.
    /// Useful for underdetermined systems or equations with parameters.
    ///
    /// # Examples
    ///
    /// - `x + y = 5` solving for x → `x = 5 - y` (y is a parameter)
    /// - `sqrt(x) = 2` → `x = 4` with constraint `x ≥ 0`
    Parametric {
        /// The solution expression, potentially containing other variables
        expression: Expression,
        /// Constraints that must be satisfied
        constraints: Vec<Constraint>,
    },

    /// No solution exists.
    ///
    /// The equation is inconsistent and has no values that satisfy it.
    ///
    /// # Examples
    ///
    /// - `0 = 5` (contradiction)
    /// - `x + 1 = x + 2` (no solution)
    None,

    /// Infinite solutions (identity).
    ///
    /// The equation is satisfied by all values (tautology).
    ///
    /// # Examples
    ///
    /// - `x = x` (trivial identity)
    /// - `2(x + 1) = 2x + 2` (identity after simplification)
    Infinite,
}

/// Constraint on a solution.
///
/// Represents a condition that must be satisfied for a solution to be valid.
/// Typically used with parametric solutions to specify domain restrictions.
///
/// # Examples
///
/// ```
/// use mathsolver_core::ast::{Variable, Expression};
/// use mathsolver_core::solver::Constraint;
///
/// // Constraint: x != 0 (for denominators)
/// // Note: The condition expression format depends on application needs
/// let constraint = Constraint {
///     variable: Variable::new("x"),
///     condition: Expression::Variable(Variable::new("x")),  // Placeholder for non-zero condition
/// };
/// ```
///
/// # Note
///
/// The exact representation of constraints is application-specific. Common uses include:
/// - Domain restrictions (e.g., x > 0 for sqrt, log)
/// - Non-zero denominators
/// - Parameter ranges
#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    /// The variable being constrained
    pub variable: Variable,
    /// The condition that must hold (e.g., x >= 0)
    pub condition: Expression,
}

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
/// This design allows for solver selection (see [`SmartSolver`]) and error handling.
///
/// # Examples
///
/// ```
/// use mathsolver_core::solver::{Solver, LinearSolver};
/// use mathsolver_core::ast::{Equation, Expression, Variable, BinaryOp};
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
    ///
    /// # Arguments
    ///
    /// * `equation` - The equation to solve
    /// * `variable` - The variable to solve for
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - [`Solution`]: The solution (unique, multiple, none, etc.)
    /// - [`ResolutionPath`]: Step-by-step record of solving process
    ///
    /// # Errors
    ///
    /// Returns [`SolverError`] if:
    /// - Variable not found in equation
    /// - Equation type not supported by this solver
    /// - Equation has no solution or infinite solutions
    /// - Other solving failures (see [`SolverError`] variants)
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
    ///
    /// # Note
    ///
    /// Returning `true` doesn't guarantee successful solving - the equation
    /// might still have no solution or be too complex. This method only
    /// checks if the equation type matches this solver's capabilities.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::solver::{Solver, LinearSolver};
    /// use mathsolver_core::ast::{Equation, Expression, Variable};
    ///
    /// let solver = LinearSolver::new();
    ///
    /// // Linear equation: x + 5 = 10
    /// let eq1 = Equation::new(
    ///     "linear",
    ///     Expression::Variable(Variable::new("x")),
    ///     Expression::Integer(5),
    /// );
    /// assert!(solver.can_solve(&eq1));
    ///
    /// // Quadratic equation: x² = 4
    /// let x = Expression::Variable(Variable::new("x"));
    /// let x_squared = Expression::Power(
    ///     Box::new(x),
    ///     Box::new(Expression::Integer(2)),
    /// );
    /// let eq2 = Equation::new("quadratic", x_squared, Expression::Integer(4));
    /// assert!(!solver.can_solve(&eq2)); // LinearSolver rejects quadratics
    /// ```
    fn can_solve(&self, equation: &Equation) -> bool;
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if expression contains the given variable.
///
/// This is a convenience wrapper around [`Expression::contains_variable`].
///
/// # Examples
///
/// ```ignore
/// let x = Expression::Variable(Variable::new("x"));
/// let y = Expression::Variable(Variable::new("y"));
/// let expr = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(Expression::Integer(5)));
///
/// assert!(contains_variable(&expr, "x"));
/// assert!(!contains_variable(&expr, "y"));
/// ```
fn contains_variable(expr: &Expression, var: &str) -> bool {
    expr.contains_variable(var)
}

/// Extract the coefficient of a variable from an expression.
///
/// Recognizes patterns like:
/// - `x` → coefficient is 1
/// - `3 * x` → coefficient is 3
/// - `x * 3` → coefficient is 3
/// - `a * x` → coefficient is a (where a doesn't contain x)
///
/// Returns `None` if the variable is not found or appears in a non-linear way.
///
/// # Examples
///
/// ```ignore
/// // 3 * x → Some(3)
/// let expr = Expression::Binary(
///     BinaryOp::Mul,
///     Box::new(Expression::Integer(3)),
///     Box::new(Expression::Variable(Variable::new("x"))),
/// );
/// let coeff = extract_coefficient(&expr, "x");
/// assert_eq!(coeff, Some(Expression::Integer(3)));
///
/// // x → Some(1)
/// let expr = Expression::Variable(Variable::new("x"));
/// let coeff = extract_coefficient(&expr, "x");
/// assert_eq!(coeff, Some(Expression::Integer(1)));
///
/// // x + 5 → None (not a pure coefficient pattern)
/// let expr = Expression::Binary(
///     BinaryOp::Add,
///     Box::new(Expression::Variable(Variable::new("x"))),
///     Box::new(Expression::Integer(5)),
/// );
/// let coeff = extract_coefficient(&expr, "x");
/// assert_eq!(coeff, None);
/// ```
fn extract_coefficient(expr: &Expression, var: &str) -> Option<Expression> {
    match expr {
        // x -> coefficient is 1
        Expression::Variable(v) if v.name == var => Some(Expression::Integer(1)),

        // a * x or x * a
        Expression::Binary(BinaryOp::Mul, left, right) => {
            if let Expression::Variable(v) = left.as_ref() {
                if v.name == var && !contains_variable(right, var) {
                    return Some(right.as_ref().clone());
                }
            }
            if let Expression::Variable(v) = right.as_ref() {
                if v.name == var && !contains_variable(left, var) {
                    return Some(left.as_ref().clone());
                }
            }
            None
        }

        _ => None,
    }
}

/// Collect terms with and without the variable from an expression.
///
/// Recursively traverses an expression and separates additive terms into two groups:
/// - Terms containing the target variable
/// - Terms not containing the target variable
///
/// This is useful for rearranging equations into the form `terms_with_var = -terms_without_var`.
///
/// # Arguments
///
/// * `expr` - The expression to analyze
/// * `var` - The variable name to search for
///
/// # Returns
///
/// A tuple `(terms_with_var, terms_without_var)` where each is a vector of expressions.
///
/// # Examples
///
/// ```ignore
/// // 2x + 3 + 5x - 7
/// // Returns: ([2x, 5x], [3, -7])
/// let x = Expression::Variable(Variable::new("x"));
/// let expr = /* ... */;
/// let (with_x, without_x) = collect_terms(&expr, "x");
/// ```
///
/// # Note
///
/// Subtraction is handled by negating the right operand before collecting.
fn collect_terms(expr: &Expression, var: &str) -> (Vec<Expression>, Vec<Expression>) {
    let mut with_var = Vec::new();
    let mut without_var = Vec::new();

    collect_terms_recursive(expr, var, &mut with_var, &mut without_var);

    (with_var, without_var)
}

fn collect_terms_recursive(
    expr: &Expression,
    var: &str,
    with_var: &mut Vec<Expression>,
    without_var: &mut Vec<Expression>,
) {
    match expr {
        Expression::Binary(BinaryOp::Add, left, right) => {
            collect_terms_recursive(left, var, with_var, without_var);
            collect_terms_recursive(right, var, with_var, without_var);
        }
        Expression::Binary(BinaryOp::Sub, left, right) => {
            collect_terms_recursive(left, var, with_var, without_var);
            // Negate the right side when collecting
            let negated =
                Expression::Unary(crate::ast::UnaryOp::Neg, Box::new(right.as_ref().clone()));
            collect_terms_recursive(&negated, var, with_var, without_var);
        }
        _ => {
            if contains_variable(expr, var) {
                with_var.push(expr.clone());
            } else {
                without_var.push(expr.clone());
            }
        }
    }
}

/// Combine a list of expressions into a single expression with addition.
fn combine_with_add(terms: Vec<Expression>) -> Expression {
    if terms.is_empty() {
        return Expression::Integer(0);
    }

    terms
        .into_iter()
        .reduce(|acc, term| Expression::Binary(BinaryOp::Add, Box::new(acc), Box::new(term)))
        .unwrap()
}

/// Evaluate constant expressions to their numeric values.
/// If the expression contains only constants, evaluate it completely.
fn evaluate_constants(expr: &Expression) -> Expression {
    // First simplify
    let simplified = expr.simplify();

    // Try to evaluate if it's all constants
    if !has_any_variable(&simplified) {
        if let Some(value) = simplified.evaluate(&HashMap::new()) {
            // Check if it's an integer value
            if value.fract().abs() < 1e-10 {
                return Expression::Integer(value.round() as i64);
            } else {
                return Expression::Float(value);
            }
        }
    }

    simplified
}

/// Isolate a variable in an equation.
///
/// Rearranges the equation to solve for the target variable, returning the
/// expression that equals the variable. This is the core solving logic for
/// linear equations.
///
/// # Algorithm
///
/// Recognizes and solves several linear patterns:
/// 1. Variable already isolated: `x = expr` or `expr = x`
/// 2. Coefficient pattern: `a*x = c` → `x = c/a`
/// 3. Addition pattern: `x + b = c` → `x = c - b`
/// 4. Combined pattern: `a*x + b = c` → `x = (c - b)/a`
///
/// All patterns are checked in both left-to-right and right-to-left orientations.
///
/// # Arguments
///
/// * `equation` - The equation to solve
/// * `var` - The variable name to isolate
/// * `path` - Resolution path builder to record solving steps
///
/// # Returns
///
/// An [`Expression`] representing the isolated variable's value.
///
/// # Errors
///
/// Returns [`SolverError`] if:
/// - Variable not found in equation
/// - Equation pattern not recognized (too complex for Phase 1)
///
/// # Examples
///
/// ```ignore
/// // 2x + 3 = 11
/// // Step 1: Recognize pattern a*x + b = c
/// // Step 2: Compute (c - b) / a = (11 - 3) / 2 = 4
/// // Returns: Expression::Integer(4)
///
/// let equation = /* ... */;
/// let mut path = ResolutionPathBuilder::new(/* ... */);
/// let result = isolate_variable(&equation, "x", &mut path)?;
/// ```
fn isolate_variable(
    equation: &Equation,
    var: &str,
    path: &mut ResolutionPathBuilder,
) -> Result<Expression, SolverError> {
    let left = &equation.left;
    let right = &equation.right;

    // Check if variable exists in equation
    if !contains_variable(left, var) && !contains_variable(right, var) {
        return Err(SolverError::CannotSolve(format!(
            "Variable '{}' not found in equation",
            var
        )));
    }

    // Special case: variable already isolated (x = expr or expr = x)
    if let Expression::Variable(v) = left {
        if v.name == var && !contains_variable(right, var) {
            return Ok(right.clone());
        }
    }
    if let Expression::Variable(v) = right {
        if v.name == var && !contains_variable(left, var) {
            return Ok(left.clone());
        }
    }

    // Try to solve simple patterns

    // Pattern: a * x = c  =>  x = c / a
    if let Some(coeff) = extract_coefficient(left, var) {
        if !contains_variable(right, var) {
            let result = Expression::Binary(
                BinaryOp::Div,
                Box::new(right.clone()),
                Box::new(coeff.clone()),
            )
            .simplify();
            let evaluated = evaluate_constants(&result);
            return Ok(evaluated);
        }
    }

    // Pattern: c = a * x  =>  x = c / a
    if let Some(coeff) = extract_coefficient(right, var) {
        if !contains_variable(left, var) {
            let result = Expression::Binary(
                BinaryOp::Div,
                Box::new(left.clone()),
                Box::new(coeff.clone()),
            )
            .simplify();
            let evaluated = evaluate_constants(&result);
            return Ok(evaluated);
        }
    }

    // Pattern: x + b = c  =>  x = c - b
    if let Expression::Binary(BinaryOp::Add, l, r) = left {
        if let Expression::Variable(v) = l.as_ref() {
            if v.name == var && !contains_variable(r, var) && !contains_variable(right, var) {
                let result = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(right.clone()),
                    Box::new(r.as_ref().clone()),
                )
                .simplify();
                let evaluated = evaluate_constants(&result);
                return Ok(evaluated);
            }
        }
        if let Expression::Variable(v) = r.as_ref() {
            if v.name == var && !contains_variable(l, var) && !contains_variable(right, var) {
                let result = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(right.clone()),
                    Box::new(l.as_ref().clone()),
                )
                .simplify();
                let evaluated = evaluate_constants(&result);
                return Ok(evaluated);
            }
        }
    }

    // Pattern: c = x + b  =>  x = c - b
    if let Expression::Binary(BinaryOp::Add, l, r) = right {
        if let Expression::Variable(v) = l.as_ref() {
            if v.name == var && !contains_variable(r, var) && !contains_variable(left, var) {
                let result = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(left.clone()),
                    Box::new(r.as_ref().clone()),
                )
                .simplify();
                let evaluated = evaluate_constants(&result);
                return Ok(evaluated);
            }
        }
        if let Expression::Variable(v) = r.as_ref() {
            if v.name == var && !contains_variable(l, var) && !contains_variable(left, var) {
                let result = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(left.clone()),
                    Box::new(l.as_ref().clone()),
                )
                .simplify();
                let evaluated = evaluate_constants(&result);
                return Ok(evaluated);
            }
        }
    }

    // Pattern: a * x + b = c  =>  x = (c - b) / a
    if let Expression::Binary(BinaryOp::Add, l, r) = left {
        if let Some(coeff) = extract_coefficient(l, var) {
            if !contains_variable(r, var) && !contains_variable(right, var) {
                let numerator = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(right.clone()),
                    Box::new(r.as_ref().clone()),
                );
                let result =
                    Expression::Binary(BinaryOp::Div, Box::new(numerator), Box::new(coeff))
                        .simplify();
                let evaluated = evaluate_constants(&result);
                return Ok(evaluated);
            }
        }
        if let Some(coeff) = extract_coefficient(r, var) {
            if !contains_variable(l, var) && !contains_variable(right, var) {
                let numerator = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(right.clone()),
                    Box::new(l.as_ref().clone()),
                );
                let result =
                    Expression::Binary(BinaryOp::Div, Box::new(numerator), Box::new(coeff))
                        .simplify();
                let evaluated = evaluate_constants(&result);
                return Ok(evaluated);
            }
        }
    }

    // More complex cases not yet supported
    Err(SolverError::CannotSolve(
        "Equation pattern not yet supported for Phase 1".to_string(),
    ))
}

/// Linear equation solver for equations of the form `ax + b = c`.
///
/// Solves first-degree polynomial equations in one variable by pattern matching
/// and algebraic manipulation. Handles various linear patterns including:
/// - Simple variable: `x = 5`
/// - Multiplication: `3x = 12`
/// - Addition: `x + 7 = 10`
/// - Combined: `2x + 3 = 11`
///
/// # Mathematical Foundation
///
/// A linear equation in one variable has the general form:
/// ```text
/// ax + b = c
/// ```
///
/// The solution is obtained by:
/// 1. Subtracting `b` from both sides: `ax = c - b`
/// 2. Dividing both sides by `a`: `x = (c - b) / a`
///
/// The solver recognizes these patterns automatically and applies the
/// appropriate transformations.
///
/// # Limitations
///
/// - Only handles linear equations (degree 1)
/// - Cannot solve equations with the variable in denominators (e.g., `1/x = 2`)
/// - Cannot solve equations with the variable in exponents (e.g., `2^x = 8`)
/// - Cannot handle products of variables (e.g., `x*y = 5`)
///
/// For more complex equations, use [`TranscendentalSolver`] or [`SmartSolver`].
///
/// # Examples
///
/// ## Simple Linear Equation
///
/// ```
/// use mathsolver_core::solver::{LinearSolver, Solver};
/// use mathsolver_core::ast::{Equation, Expression, Variable, BinaryOp};
///
/// // Solve: 2x + 3 = 11
/// let x = Expression::Variable(Variable::new("x"));
/// let two_x = Expression::Binary(
///     BinaryOp::Mul,
///     Box::new(Expression::Integer(2)),
///     Box::new(x),
/// );
/// let left = Expression::Binary(
///     BinaryOp::Add,
///     Box::new(two_x),
///     Box::new(Expression::Integer(3)),
/// );
/// let equation = Equation::new("linear", left, Expression::Integer(11));
///
/// let solver = LinearSolver::new();
/// let (solution, path) = solver.solve(&equation, &Variable::new("x")).unwrap();
///
/// // Verify solution: x = 4
/// # use mathsolver_core::solver::Solution;
/// # use std::collections::HashMap;
/// # match solution {
/// #     Solution::Unique(expr) => {
/// #         assert_eq!(expr.evaluate(&HashMap::new()), Some(4.0));
/// #     }
/// #     _ => panic!("Expected unique solution"),
/// # }
/// ```
///
/// ## Equation with Parametric Coefficients
///
/// ```
/// use mathsolver_core::solver::{LinearSolver, Solver};
/// use mathsolver_core::ast::{Equation, Expression, Variable, BinaryOp};
/// use std::collections::HashMap;
///
/// // Solve: ax = b for x (symbolic)
/// let a = Expression::Variable(Variable::new("a"));
/// let x = Expression::Variable(Variable::new("x"));
/// let b = Expression::Variable(Variable::new("b"));
///
/// let left = Expression::Binary(BinaryOp::Mul, Box::new(a), Box::new(x));
/// let equation = Equation::new("parametric", left, b);
///
/// let solver = LinearSolver::new();
/// let (solution, _path) = solver.solve(&equation, &Variable::new("x")).unwrap();
///
/// // Solution is symbolic: x = b/a
/// # use mathsolver_core::solver::Solution;
/// # match solution {
/// #     Solution::Unique(expr) => {
/// #         // Can substitute values later
/// #         let mut values = HashMap::new();
/// #         values.insert("a".to_string(), 3.0);
/// #         values.insert("b".to_string(), 12.0);
/// #         // Result would be 4.0
/// #     }
/// #     _ => panic!("Expected unique solution"),
/// # }
/// ```
///
/// ## Checking Solver Applicability
///
/// ```
/// use mathsolver_core::solver::{LinearSolver, Solver};
/// use mathsolver_core::ast::{Equation, Expression, Variable};
///
/// let solver = LinearSolver::new();
///
/// // Can solve linear equation
/// let linear = Equation::new(
///     "linear",
///     Expression::Variable(Variable::new("x")),
///     Expression::Integer(5),
/// );
/// assert!(solver.can_solve(&linear));
///
/// // Cannot solve quadratic equation
/// let x = Expression::Variable(Variable::new("x"));
/// let x_squared = Expression::Power(
///     Box::new(x),
///     Box::new(Expression::Integer(2)),
/// );
/// let quadratic = Equation::new("quadratic", x_squared, Expression::Integer(4));
/// assert!(!solver.can_solve(&quadratic));
/// ```
///
/// # See Also
///
/// - [`SmartSolver`]: Automatically selects LinearSolver for linear equations
/// - [`TranscendentalSolver`]: For equations with sin, cos, exp, ln, etc.
/// - [`solve_for`]: High-level API that uses SmartSolver and substitutes known values
#[derive(Debug, Default)]
pub struct LinearSolver;

impl LinearSolver {
    /// Create a new linear equation solver.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::solver::LinearSolver;
    ///
    /// let solver = LinearSolver::new();
    /// ```
    pub fn new() -> Self {
        Self
    }
}

impl Solver for LinearSolver {
    fn solve(
        &self,
        equation: &Equation,
        variable: &Variable,
    ) -> SolverResult<(Solution, ResolutionPath)> {
        let var_name = &variable.name;

        // Initialize resolution path
        let initial_expr = Expression::Binary(
            BinaryOp::Sub,
            Box::new(equation.left.clone()),
            Box::new(equation.right.clone()),
        );
        let mut path = ResolutionPathBuilder::new(initial_expr.clone());

        // Check if variable appears in equation
        let left_has_var = contains_variable(&equation.left, var_name);
        let right_has_var = contains_variable(&equation.right, var_name);

        if !left_has_var && !right_has_var {
            return Err(SolverError::CannotSolve(format!(
                "Variable '{}' not found in equation",
                var_name
            )));
        }

        // Check if equation is linear in the target variable
        if !is_linear_in_variable(&equation.left, var_name)
            || !is_linear_in_variable(&equation.right, var_name)
        {
            return Err(SolverError::UnsupportedEquationType);
        }

        // Isolate the variable
        let result_expr = isolate_variable(equation, var_name, &mut path)?;

        // Add isolation step
        path = path.step(
            Operation::Isolate(variable.clone()),
            format!("Isolate {} on one side", variable),
            result_expr.clone(),
        );

        // Build final resolution path
        let resolution_path = path.finish(result_expr.clone());

        Ok((Solution::Unique(result_expr), resolution_path))
    }

    fn can_solve(&self, equation: &Equation) -> bool {
        // Check if equation has obvious non-linear features (powers > 1 with variables)
        // We're more permissive here since we don't know the target variable yet,
        // but we can still reject clearly quadratic/polynomial equations.
        !has_obvious_nonlinearity(&equation.left) && !has_obvious_nonlinearity(&equation.right)
    }
}

/// Check if an expression has obvious non-linear features like x^2.
fn has_obvious_nonlinearity(expr: &Expression) -> bool {
    match expr {
        Expression::Power(base, exp) => {
            // x^2 or any variable raised to power > 1
            if has_any_variable(base) {
                // Check if exponent is > 1
                if let Some(exp_val) = exp.evaluate(&HashMap::new()) {
                    if exp_val > 1.0 {
                        return true;
                    }
                }
            }
            has_obvious_nonlinearity(base) || has_obvious_nonlinearity(exp)
        }
        Expression::Unary(_, inner) => has_obvious_nonlinearity(inner),
        Expression::Binary(_, left, right) => {
            has_obvious_nonlinearity(left) || has_obvious_nonlinearity(right)
        }
        Expression::Function(_, args) => args.iter().any(|arg| has_obvious_nonlinearity(arg)),
        _ => false,
    }
}

/// Check if an expression is linear (no variable powers, products, or functions).
fn is_linear_equation(expr: &Expression) -> bool {
    match expr {
        Expression::Integer(_)
        | Expression::Rational(_)
        | Expression::Float(_)
        | Expression::Complex(_)
        | Expression::Variable(_) => true,

        Expression::Unary(_, inner) => is_linear_equation(inner),

        Expression::Binary(op, left, right) => {
            let left_linear = is_linear_equation(left);
            let right_linear = is_linear_equation(right);

            match op {
                BinaryOp::Add | BinaryOp::Sub => left_linear && right_linear,
                BinaryOp::Mul | BinaryOp::Div => {
                    // For multiplication/division to be linear, at most one side can have variables
                    let left_has_var = has_any_variable(left);
                    let right_has_var = has_any_variable(right);
                    left_linear && right_linear && !(left_has_var && right_has_var)
                }
                _ => false,
            }
        }

        Expression::Power(base, exp) => {
            // Only allow constant powers, and base must not have variables
            !has_any_variable(base) && is_linear_equation(exp)
        }

        Expression::Function(_, _) => {
            // For Phase 1, we don't support functions in linear equations
            false
        }
    }
}

/// Check if expression contains any variables.
fn has_any_variable(expr: &Expression) -> bool {
    match expr {
        Expression::Variable(_) => true,
        Expression::Unary(_, inner) => has_any_variable(inner),
        Expression::Binary(_, left, right) => has_any_variable(left) || has_any_variable(right),
        Expression::Function(_, args) => args.iter().any(has_any_variable),
        Expression::Power(base, exp) => has_any_variable(base) || has_any_variable(exp),
        _ => false,
    }
}

/// Check if an expression is linear with respect to a specific variable.
/// An expression is linear in variable x if:
/// - x appears to at most power 1
/// - x does not appear in denominators
/// - x does not appear multiplied by itself
/// - x does not appear in functions
fn is_linear_in_variable(expr: &Expression, var: &str) -> bool {
    match expr {
        Expression::Integer(_)
        | Expression::Rational(_)
        | Expression::Float(_)
        | Expression::Complex(_) => true,

        Expression::Variable(v) => {
            // The target variable itself is linear
            true
        }

        Expression::Unary(_, inner) => is_linear_in_variable(inner, var),

        Expression::Binary(op, left, right) => {
            let left_has_var = contains_variable(left, var);
            let right_has_var = contains_variable(right, var);

            match op {
                BinaryOp::Add | BinaryOp::Sub => {
                    // x + y and x - y are linear if both sides are linear
                    is_linear_in_variable(left, var) && is_linear_in_variable(right, var)
                }
                BinaryOp::Mul => {
                    // For multiplication to be linear in x, at most one side can contain x
                    if left_has_var && right_has_var {
                        // x * x or x * f(x) is not linear
                        false
                    } else {
                        // a * x is linear
                        is_linear_in_variable(left, var) && is_linear_in_variable(right, var)
                    }
                }
                BinaryOp::Div => {
                    // x / a is linear, but a / x is not
                    if right_has_var {
                        false // Variable in denominator makes it non-linear
                    } else {
                        is_linear_in_variable(left, var)
                    }
                }
                _ => false,
            }
        }

        Expression::Power(base, exp) => {
            // x^2 is not linear, but a^x could be (though we don't handle that in Phase 1)
            // For Phase 1, we only allow constant powers where base doesn't have the variable
            !contains_variable(base, var) && is_linear_in_variable(exp, var)
        }

        Expression::Function(_, _) => {
            // For Phase 1, functions are not supported
            false
        }
    }
}

/// Quadratic equation solver (ax² + bx + c = 0).
#[derive(Debug, Default)]
pub struct QuadraticSolver;

impl QuadraticSolver {
    pub fn new() -> Self {
        Self
    }
}

impl Solver for QuadraticSolver {
    fn solve(
        &self,
        _equation: &Equation,
        _variable: &Variable,
    ) -> SolverResult<(Solution, ResolutionPath)> {
        // TODO: Implement quadratic formula
        // TODO: Handle complex roots
        // TODO: Handle degenerate cases (a=0, becomes linear)
        // TODO: Handle discriminant = 0 (repeated root)
        Err(SolverError::Other("Not yet implemented".to_string()))
    }

    fn can_solve(&self, _equation: &Equation) -> bool {
        // TODO: Check if equation is quadratic in the target variable
        false
    }
}

/// Polynomial equation solver (general degree n).
#[derive(Debug, Default)]
pub struct PolynomialSolver;

impl PolynomialSolver {
    pub fn new() -> Self {
        Self
    }
}

impl Solver for PolynomialSolver {
    fn solve(
        &self,
        _equation: &Equation,
        _variable: &Variable,
    ) -> SolverResult<(Solution, ResolutionPath)> {
        // TODO: Implement polynomial solving
        // TODO: Use appropriate method based on degree:
        //   - degree 1: linear
        //   - degree 2: quadratic formula
        //   - degree 3: cubic formula
        //   - degree 4: quartic formula
        //   - degree 5+: numerical methods
        Err(SolverError::Other("Not yet implemented".to_string()))
    }

    fn can_solve(&self, _equation: &Equation) -> bool {
        // TODO: Check if equation is polynomial
        false
    }
}

/// Transcendental equation solver (equations with trig, exp, log functions).
#[derive(Debug, Default)]
pub struct TranscendentalSolver;

impl TranscendentalSolver {
    pub fn new() -> Self {
        Self
    }

    /// Try to solve a trigonometric equation for the target variable.
    fn solve_trig_equation(
        &self,
        equation: &Equation,
        variable: &Variable,
        path: &mut ResolutionPath,
    ) -> Option<Expression> {
        let var_name = &variable.name;

        // Pattern: sin(x) = a  →  x = asin(a)
        if let Some((result, func, value)) = self.match_trig_pattern_with_validation(
            &equation.left,
            &equation.right,
            var_name,
            crate::ast::Function::Sin,
            crate::ast::Function::Asin,
        ) {
            // Validate domain before creating result
            if let Err(e) = Self::validate_trig_domain(value, &func) {
                return None; // Return None to allow error propagation at higher level
            }
            path.add_step(ResolutionStep::new(
                Operation::ApplyFunction("asin".to_string()),
                format!("Apply arcsine to solve sin({}) = value", variable),
                result.clone(),
            ));
            return Some(result);
        }

        // Pattern: a = sin(x)  →  x = asin(a)
        if let Some((result, func, value)) = self.match_trig_pattern_with_validation(
            &equation.right,
            &equation.left,
            var_name,
            crate::ast::Function::Sin,
            crate::ast::Function::Asin,
        ) {
            if let Err(e) = Self::validate_trig_domain(value, &func) {
                return None;
            }
            path.add_step(ResolutionStep::new(
                Operation::ApplyFunction("asin".to_string()),
                format!("Apply arcsine to solve sin({}) = value", variable),
                result.clone(),
            ));
            return Some(result);
        }

        // Pattern: cos(x) = a  →  x = acos(a)
        if let Some((result, func, value)) = self.match_trig_pattern_with_validation(
            &equation.left,
            &equation.right,
            var_name,
            crate::ast::Function::Cos,
            crate::ast::Function::Acos,
        ) {
            if let Err(e) = Self::validate_trig_domain(value, &func) {
                return None;
            }
            path.add_step(ResolutionStep::new(
                Operation::ApplyFunction("acos".to_string()),
                format!("Apply arccosine to solve cos({}) = value", variable),
                result.clone(),
            ));
            return Some(result);
        }

        // Pattern: a = cos(x)  →  x = acos(a)
        if let Some((result, func, value)) = self.match_trig_pattern_with_validation(
            &equation.right,
            &equation.left,
            var_name,
            crate::ast::Function::Cos,
            crate::ast::Function::Acos,
        ) {
            if let Err(e) = Self::validate_trig_domain(value, &func) {
                return None;
            }
            path.add_step(ResolutionStep::new(
                Operation::ApplyFunction("acos".to_string()),
                format!("Apply arccosine to solve cos({}) = value", variable),
                result.clone(),
            ));
            return Some(result);
        }

        // Pattern: tan(x) = a  →  x = atan(a)
        if let Some(result) = self.match_trig_pattern(
            &equation.left,
            &equation.right,
            var_name,
            crate::ast::Function::Tan,
            crate::ast::Function::Atan,
        ) {
            path.add_step(ResolutionStep::new(
                Operation::ApplyFunction("atan".to_string()),
                format!("Apply arctangent to solve tan({}) = value", variable),
                result.clone(),
            ));
            return Some(result);
        }

        // Pattern: a = tan(x)  →  x = atan(a)
        if let Some(result) = self.match_trig_pattern(
            &equation.right,
            &equation.left,
            var_name,
            crate::ast::Function::Tan,
            crate::ast::Function::Atan,
        ) {
            path.add_step(ResolutionStep::new(
                Operation::ApplyFunction("atan".to_string()),
                format!("Apply arctangent to solve tan({}) = value", variable),
                result.clone(),
            ));
            return Some(result);
        }

        None
    }

    /// Match pattern with validation: returns (result, inverse_func, input_value)
    fn match_trig_pattern_with_validation(
        &self,
        left: &Expression,
        right: &Expression,
        var: &str,
        func: crate::ast::Function,
        inverse_func: crate::ast::Function,
    ) -> Option<(Expression, crate::ast::Function, f64)> {
        // Check if right side contains the variable
        if contains_variable(right, var) {
            return None;
        }

        // Try to evaluate the right side as a constant
        let value = match right.evaluate(&HashMap::new()) {
            Some(v) => v,
            None => return None, // Can't validate if not a constant
        };

        // Pattern 1: func(x) = a  →  x = inverse_func(a)
        if let Expression::Function(f, args) = left {
            if *f == func && args.len() == 1 {
                // Check if arg is exactly the variable
                if let Expression::Variable(v) = &args[0] {
                    if v.name == var {
                        let result =
                            Expression::Function(inverse_func.clone(), vec![right.clone()]);
                        return Some((result.simplify(), inverse_func, value));
                    }
                }

                // Check if arg is a linear expression like a*x
                if let Some(coeff) = extract_coefficient(&args[0], var) {
                    // func(a*x) = b  →  a*x = inverse_func(b)  →  x = inverse_func(b) / a
                    let inverse_applied =
                        Expression::Function(inverse_func.clone(), vec![right.clone()]);
                    let result = Expression::Binary(
                        BinaryOp::Div,
                        Box::new(inverse_applied),
                        Box::new(coeff),
                    );
                    return Some((result.simplify(), inverse_func, value));
                }
            }
        }

        // Pattern 2: a * func(x) = b  →  func(x) = b/a  →  x = inverse_func(b/a)
        if let Expression::Binary(BinaryOp::Mul, mul_left, mul_right) = left {
            // Check left side of multiplication
            if let Expression::Function(f, args) = mul_left.as_ref() {
                if *f == func && args.len() == 1 && !contains_variable(mul_right, var) {
                    if let Expression::Variable(v) = &args[0] {
                        if v.name == var {
                            // a * func(x) = b  →  func(x) = b/a  →  x = inverse_func(b/a)
                            let divided = Expression::Binary(
                                BinaryOp::Div,
                                Box::new(right.clone()),
                                Box::new(mul_right.as_ref().clone()),
                            );
                            // Need to evaluate the divided value
                            let divided_value = divided.evaluate(&HashMap::new()).unwrap_or(value);
                            let result = Expression::Function(inverse_func.clone(), vec![divided]);
                            return Some((result.simplify(), inverse_func, divided_value));
                        }
                    }
                }
            }

            // Check right side of multiplication
            if let Expression::Function(f, args) = mul_right.as_ref() {
                if *f == func && args.len() == 1 && !contains_variable(mul_left, var) {
                    if let Expression::Variable(v) = &args[0] {
                        if v.name == var {
                            // func(x) * a = b  →  func(x) = b/a  →  x = inverse_func(b/a)
                            let divided = Expression::Binary(
                                BinaryOp::Div,
                                Box::new(right.clone()),
                                Box::new(mul_left.as_ref().clone()),
                            );
                            let divided_value = divided.evaluate(&HashMap::new()).unwrap_or(value);
                            let result = Expression::Function(inverse_func.clone(), vec![divided]);
                            return Some((result.simplify(), inverse_func, divided_value));
                        }
                    }
                }
            }
        }

        None
    }

    /// Match pattern: func(var) = value or coeff * func(var) = value
    fn match_trig_pattern(
        &self,
        left: &Expression,
        right: &Expression,
        var: &str,
        func: crate::ast::Function,
        inverse_func: crate::ast::Function,
    ) -> Option<Expression> {
        // Check if right side contains the variable
        if contains_variable(right, var) {
            return None;
        }

        // Pattern 1: func(x) = a  →  x = inverse_func(a)
        if let Expression::Function(f, args) = left {
            if *f == func && args.len() == 1 {
                // Check if arg is exactly the variable
                if let Expression::Variable(v) = &args[0] {
                    if v.name == var {
                        let result = Expression::Function(inverse_func, vec![right.clone()]);
                        return Some(result.simplify());
                    }
                }

                // Check if arg is a linear expression like a*x
                if let Some(coeff) = extract_coefficient(&args[0], var) {
                    // func(a*x) = b  →  a*x = inverse_func(b)  →  x = inverse_func(b) / a
                    let inverse_applied = Expression::Function(inverse_func, vec![right.clone()]);
                    let result = Expression::Binary(
                        BinaryOp::Div,
                        Box::new(inverse_applied),
                        Box::new(coeff),
                    );
                    return Some(result.simplify());
                }
            }
        }

        // Pattern 2: a * func(x) = b  →  func(x) = b/a  →  x = inverse_func(b/a)
        if let Expression::Binary(BinaryOp::Mul, mul_left, mul_right) = left {
            // Check left side of multiplication
            if let Expression::Function(f, args) = mul_left.as_ref() {
                if *f == func && args.len() == 1 && !contains_variable(mul_right, var) {
                    if let Expression::Variable(v) = &args[0] {
                        if v.name == var {
                            // a * func(x) = b  →  func(x) = b/a  →  x = inverse_func(b/a)
                            let divided = Expression::Binary(
                                BinaryOp::Div,
                                Box::new(right.clone()),
                                Box::new(mul_right.as_ref().clone()),
                            );
                            let result = Expression::Function(inverse_func, vec![divided]);
                            return Some(result.simplify());
                        }
                    }
                }
            }

            // Check right side of multiplication
            if let Expression::Function(f, args) = mul_right.as_ref() {
                if *f == func && args.len() == 1 && !contains_variable(mul_left, var) {
                    if let Expression::Variable(v) = &args[0] {
                        if v.name == var {
                            // func(x) * a = b  →  func(x) = b/a  →  x = inverse_func(b/a)
                            let divided = Expression::Binary(
                                BinaryOp::Div,
                                Box::new(right.clone()),
                                Box::new(mul_left.as_ref().clone()),
                            );
                            let result = Expression::Function(inverse_func, vec![divided]);
                            return Some(result.simplify());
                        }
                    }
                }
            }
        }

        None
    }

    /// Try to solve a logarithmic equation for the target variable.
    fn solve_log_equation(
        &self,
        equation: &Equation,
        variable: &Variable,
        path: &mut ResolutionPath,
    ) -> Option<Expression> {
        let var_name = &variable.name;

        // Pattern: ln(x) = a  →  x = exp(a)
        if let Some(result) = self.match_log_pattern(&equation.left, &equation.right, var_name) {
            path.add_step(ResolutionStep::new(
                Operation::ApplyFunction("exp".to_string()),
                format!("Apply exponential to solve ln({}) = value", variable),
                result.clone(),
            ));
            return Some(result);
        }

        // Pattern: a = ln(x)  →  x = exp(a)
        if let Some(result) = self.match_log_pattern(&equation.right, &equation.left, var_name) {
            path.add_step(ResolutionStep::new(
                Operation::ApplyFunction("exp".to_string()),
                format!("Apply exponential to solve ln({}) = value", variable),
                result.clone(),
            ));
            return Some(result);
        }

        // Pattern: log10(x) = a  →  x = 10^a
        if let Some(result) = self.match_log10_pattern(&equation.left, &equation.right, var_name) {
            path.add_step(ResolutionStep::new(
                Operation::PowerBothSides(Expression::Integer(10)),
                format!("Apply 10^x to solve log10({}) = value", variable),
                result.clone(),
            ));
            return Some(result);
        }

        // Pattern: a = log10(x)  →  x = 10^a
        if let Some(result) = self.match_log10_pattern(&equation.right, &equation.left, var_name) {
            path.add_step(ResolutionStep::new(
                Operation::PowerBothSides(Expression::Integer(10)),
                format!("Apply 10^x to solve log10({}) = value", variable),
                result.clone(),
            ));
            return Some(result);
        }

        // Pattern: log(x, b) = a  →  x = b^a
        if let Some(result) = self.match_log_base_pattern(&equation.left, &equation.right, var_name)
        {
            path.add_step(ResolutionStep::new(
                Operation::ApplyLogProperty("exponential form".to_string()),
                format!(
                    "Convert logarithm to exponential form to solve for {}",
                    variable
                ),
                result.clone(),
            ));
            return Some(result);
        }

        // Pattern: a = log(x, b)  →  x = b^a
        if let Some(result) = self.match_log_base_pattern(&equation.right, &equation.left, var_name)
        {
            path.add_step(ResolutionStep::new(
                Operation::ApplyLogProperty("exponential form".to_string()),
                format!(
                    "Convert logarithm to exponential form to solve for {}",
                    variable
                ),
                result.clone(),
            ));
            return Some(result);
        }

        None
    }

    /// Match pattern: ln(var) = value or coeff * ln(var) = value
    fn match_log_pattern(
        &self,
        left: &Expression,
        right: &Expression,
        var: &str,
    ) -> Option<Expression> {
        // Check if right side contains the variable
        if contains_variable(right, var) {
            return None;
        }

        // Pattern 1: ln(x) = a  →  x = exp(a)
        if let Expression::Function(crate::ast::Function::Ln, args) = left {
            if args.len() == 1 {
                if let Expression::Variable(v) = &args[0] {
                    if v.name == var {
                        let result =
                            Expression::Function(crate::ast::Function::Exp, vec![right.clone()]);
                        return Some(result.simplify());
                    }
                }
            }
        }

        // Pattern 2: a * ln(x) = b  →  ln(x) = b/a  →  x = exp(b/a)
        if let Expression::Binary(BinaryOp::Mul, mul_left, mul_right) = left {
            if let Expression::Function(crate::ast::Function::Ln, args) = mul_left.as_ref() {
                if args.len() == 1 && !contains_variable(mul_right, var) {
                    if let Expression::Variable(v) = &args[0] {
                        if v.name == var {
                            let divided = Expression::Binary(
                                BinaryOp::Div,
                                Box::new(right.clone()),
                                Box::new(mul_right.as_ref().clone()),
                            );
                            let result =
                                Expression::Function(crate::ast::Function::Exp, vec![divided]);
                            return Some(result.simplify());
                        }
                    }
                }
            }

            if let Expression::Function(crate::ast::Function::Ln, args) = mul_right.as_ref() {
                if args.len() == 1 && !contains_variable(mul_left, var) {
                    if let Expression::Variable(v) = &args[0] {
                        if v.name == var {
                            let divided = Expression::Binary(
                                BinaryOp::Div,
                                Box::new(right.clone()),
                                Box::new(mul_left.as_ref().clone()),
                            );
                            let result =
                                Expression::Function(crate::ast::Function::Exp, vec![divided]);
                            return Some(result.simplify());
                        }
                    }
                }
            }
        }

        None
    }

    /// Match pattern: log10(var) = value
    fn match_log10_pattern(
        &self,
        left: &Expression,
        right: &Expression,
        var: &str,
    ) -> Option<Expression> {
        if contains_variable(right, var) {
            return None;
        }

        // Pattern: log10(x) = a  →  x = 10^a
        if let Expression::Function(crate::ast::Function::Log10, args) = left {
            if args.len() == 1 {
                if let Expression::Variable(v) = &args[0] {
                    if v.name == var {
                        let result = Expression::Power(
                            Box::new(Expression::Integer(10)),
                            Box::new(right.clone()),
                        );
                        return Some(result.simplify());
                    }
                }
            }
        }

        None
    }

    /// Match pattern: log(var, base) = value
    fn match_log_base_pattern(
        &self,
        left: &Expression,
        right: &Expression,
        var: &str,
    ) -> Option<Expression> {
        if contains_variable(right, var) {
            return None;
        }

        // Pattern: log(x, b) = a  →  x = b^a
        if let Expression::Function(crate::ast::Function::Log, args) = left {
            if args.len() == 2 {
                if let Expression::Variable(v) = &args[0] {
                    if v.name == var && !contains_variable(&args[1], var) {
                        let result =
                            Expression::Power(Box::new(args[1].clone()), Box::new(right.clone()));
                        return Some(result.simplify());
                    }
                }
            }
        }

        None
    }

    /// Try to solve an exponential equation for the target variable.
    fn solve_exp_equation(
        &self,
        equation: &Equation,
        variable: &Variable,
        path: &mut ResolutionPath,
    ) -> Option<Expression> {
        let var_name = &variable.name;

        // Pattern: exp(x) = a  →  x = ln(a)
        if let Some(result) = self.match_exp_pattern(&equation.left, &equation.right, var_name) {
            path.add_step(ResolutionStep::new(
                Operation::ApplyFunction("ln".to_string()),
                format!("Apply natural logarithm to solve exp({}) = value", variable),
                result.clone(),
            ));
            return Some(result);
        }

        // Pattern: a = exp(x)  →  x = ln(a)
        if let Some(result) = self.match_exp_pattern(&equation.right, &equation.left, var_name) {
            path.add_step(ResolutionStep::new(
                Operation::ApplyFunction("ln".to_string()),
                format!("Apply natural logarithm to solve exp({}) = value", variable),
                result.clone(),
            ));
            return Some(result);
        }

        // Pattern: a^x = b  →  x = ln(b) / ln(a)
        if let Some(result) = self.match_power_pattern(&equation.left, &equation.right, var_name) {
            path.add_step(ResolutionStep::new(
                Operation::ApplyLogProperty("change of base".to_string()),
                format!("Apply logarithm to solve for {} in exponent", variable),
                result.clone(),
            ));
            return Some(result);
        }

        // Pattern: b = a^x  →  x = ln(b) / ln(a)
        if let Some(result) = self.match_power_pattern(&equation.right, &equation.left, var_name) {
            path.add_step(ResolutionStep::new(
                Operation::ApplyLogProperty("change of base".to_string()),
                format!("Apply logarithm to solve for {} in exponent", variable),
                result.clone(),
            ));
            return Some(result);
        }

        None
    }

    /// Match pattern: exp(var) = value or coeff * exp(var) = value
    fn match_exp_pattern(
        &self,
        left: &Expression,
        right: &Expression,
        var: &str,
    ) -> Option<Expression> {
        if contains_variable(right, var) {
            return None;
        }

        // Pattern 1: exp(x) = a  →  x = ln(a)
        if let Expression::Function(crate::ast::Function::Exp, args) = left {
            if args.len() == 1 {
                if let Expression::Variable(v) = &args[0] {
                    if v.name == var {
                        let result =
                            Expression::Function(crate::ast::Function::Ln, vec![right.clone()]);
                        return Some(result.simplify());
                    }
                }

                // Pattern: exp(a*x) = b  →  a*x = ln(b)  →  x = ln(b)/a
                if let Some(coeff) = extract_coefficient(&args[0], var) {
                    let ln_applied =
                        Expression::Function(crate::ast::Function::Ln, vec![right.clone()]);
                    let result =
                        Expression::Binary(BinaryOp::Div, Box::new(ln_applied), Box::new(coeff));
                    return Some(result.simplify());
                }
            }
        }

        // Pattern 2: a * exp(x) = b  →  exp(x) = b/a  →  x = ln(b/a)
        if let Expression::Binary(BinaryOp::Mul, mul_left, mul_right) = left {
            if let Expression::Function(crate::ast::Function::Exp, args) = mul_left.as_ref() {
                if args.len() == 1 && !contains_variable(mul_right, var) {
                    if let Expression::Variable(v) = &args[0] {
                        if v.name == var {
                            let divided = Expression::Binary(
                                BinaryOp::Div,
                                Box::new(right.clone()),
                                Box::new(mul_right.as_ref().clone()),
                            );
                            let result =
                                Expression::Function(crate::ast::Function::Ln, vec![divided]);
                            return Some(result.simplify());
                        }
                    }
                }
            }

            if let Expression::Function(crate::ast::Function::Exp, args) = mul_right.as_ref() {
                if args.len() == 1 && !contains_variable(mul_left, var) {
                    if let Expression::Variable(v) = &args[0] {
                        if v.name == var {
                            let divided = Expression::Binary(
                                BinaryOp::Div,
                                Box::new(right.clone()),
                                Box::new(mul_left.as_ref().clone()),
                            );
                            let result =
                                Expression::Function(crate::ast::Function::Ln, vec![divided]);
                            return Some(result.simplify());
                        }
                    }
                }
            }
        }

        None
    }

    /// Match pattern: base^var = value
    fn match_power_pattern(
        &self,
        left: &Expression,
        right: &Expression,
        var: &str,
    ) -> Option<Expression> {
        if contains_variable(right, var) {
            return None;
        }

        // Pattern: a^x = b  →  x = ln(b) / ln(a)
        if let Expression::Power(base, exp) = left {
            if !contains_variable(base, var) && contains_variable(exp, var) {
                // Simple case: a^x = b
                if let Expression::Variable(v) = exp.as_ref() {
                    if v.name == var {
                        let ln_right =
                            Expression::Function(crate::ast::Function::Ln, vec![right.clone()]);
                        let ln_base = Expression::Function(
                            crate::ast::Function::Ln,
                            vec![base.as_ref().clone()],
                        );
                        let result = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(ln_right),
                            Box::new(ln_base),
                        );
                        return Some(result.simplify());
                    }
                }

                // Pattern: a^(b*x) = c  →  b*x = ln(c)/ln(a)  →  x = ln(c)/(b*ln(a))
                if let Some(coeff) = extract_coefficient(exp, var) {
                    let ln_right =
                        Expression::Function(crate::ast::Function::Ln, vec![right.clone()]);
                    let ln_base =
                        Expression::Function(crate::ast::Function::Ln, vec![base.as_ref().clone()]);
                    let divided =
                        Expression::Binary(BinaryOp::Div, Box::new(ln_right), Box::new(ln_base));
                    let result =
                        Expression::Binary(BinaryOp::Div, Box::new(divided), Box::new(coeff));
                    return Some(result.simplify());
                }
            }
        }

        None
    }

    /// Check if an equation contains transcendental functions.
    fn has_transcendental_function(expr: &Expression) -> bool {
        match expr {
            Expression::Function(func, _) => {
                matches!(
                    func,
                    crate::ast::Function::Sin
                        | crate::ast::Function::Cos
                        | crate::ast::Function::Tan
                        | crate::ast::Function::Asin
                        | crate::ast::Function::Acos
                        | crate::ast::Function::Atan
                        | crate::ast::Function::Sinh
                        | crate::ast::Function::Cosh
                        | crate::ast::Function::Tanh
                        | crate::ast::Function::Exp
                        | crate::ast::Function::Ln
                        | crate::ast::Function::Log
                        | crate::ast::Function::Log2
                        | crate::ast::Function::Log10
                )
            }
            Expression::Unary(_, inner) => Self::has_transcendental_function(inner),
            Expression::Binary(_, left, right) => {
                Self::has_transcendental_function(left) || Self::has_transcendental_function(right)
            }
            Expression::Power(base, exp) => {
                // Check if variable appears in exponent (exponential form)
                has_any_variable(exp)
                    || Self::has_transcendental_function(base)
                    || Self::has_transcendental_function(exp)
            }
            _ => false,
        }
    }

    /// Validate domain restrictions for inverse trigonometric functions.
    fn validate_trig_domain(value: f64, func: &crate::ast::Function) -> Result<(), SolverError> {
        match func {
            crate::ast::Function::Asin | crate::ast::Function::Acos => {
                if value.abs() > 1.0 {
                    return Err(SolverError::Other(format!(
                        "Domain error: {:?} requires |value| ≤ 1, got {}",
                        func, value
                    )));
                }
            }
            _ => {}
        }
        Ok(())
    }
}

impl Solver for TranscendentalSolver {
    fn solve(
        &self,
        equation: &Equation,
        variable: &Variable,
    ) -> SolverResult<(Solution, ResolutionPath)> {
        let var_name = &variable.name;

        // Check if variable appears in equation
        let left_has_var = contains_variable(&equation.left, var_name);
        let right_has_var = contains_variable(&equation.right, var_name);

        if !left_has_var && !right_has_var {
            return Err(SolverError::CannotSolve(format!(
                "Variable '{}' not found in equation",
                var_name
            )));
        }

        // Initialize resolution path
        let initial_expr = Expression::Binary(
            BinaryOp::Sub,
            Box::new(equation.left.clone()),
            Box::new(equation.right.clone()),
        );
        let mut path = ResolutionPath::new(initial_expr);

        // Try trigonometric equation patterns
        if let Some(result) = self.solve_trig_equation(equation, variable, &mut path) {
            // Validate domain if result is a constant
            if let Expression::Function(func, args) = &result {
                if args.len() == 1 {
                    if let Some(val) = args[0].evaluate(&HashMap::new()) {
                        Self::validate_trig_domain(val, func)?;
                    }
                }
            }

            let evaluated = evaluate_constants(&result);
            path.set_result(evaluated.clone());
            return Ok((Solution::Unique(evaluated), path));
        }

        // Try logarithmic equation patterns
        if let Some(result) = self.solve_log_equation(equation, variable, &mut path) {
            let evaluated = evaluate_constants(&result);
            path.set_result(evaluated.clone());
            return Ok((Solution::Unique(evaluated), path));
        }

        // Try exponential equation patterns
        if let Some(result) = self.solve_exp_equation(equation, variable, &mut path) {
            let evaluated = evaluate_constants(&result);
            path.set_result(evaluated.clone());
            return Ok((Solution::Unique(evaluated), path));
        }

        // If no pattern matched, cannot solve
        Err(SolverError::CannotSolve(
            "Transcendental equation pattern not recognized or too complex".to_string(),
        ))
    }

    fn can_solve(&self, equation: &Equation) -> bool {
        // Check if equation contains transcendental functions
        Self::has_transcendental_function(&equation.left)
            || Self::has_transcendental_function(&equation.right)
    }
}

/// System of equations solver.
#[derive(Debug, Default)]
pub struct SystemSolver;

impl SystemSolver {
    pub fn new() -> Self {
        Self
    }

    /// Solve a system of equations for multiple variables.
    pub fn solve_system(
        &self,
        _equations: &[Equation],
        _variables: &[Variable],
    ) -> SolverResult<HashMap<Variable, Solution>> {
        // TODO: Implement system solving
        // TODO: Support linear systems (Gaussian elimination)
        // TODO: Support nonlinear systems (Newton-Raphson)
        // TODO: Detect under/over-determined systems
        Err(SolverError::Other("Not yet implemented".to_string()))
    }
}

/// Smart solver that dispatches to appropriate specialized solver.
#[derive(Debug)]
pub struct SmartSolver {
    linear: LinearSolver,
    quadratic: QuadraticSolver,
    polynomial: PolynomialSolver,
    transcendental: TranscendentalSolver,
}

impl SmartSolver {
    pub fn new() -> Self {
        Self {
            linear: LinearSolver::new(),
            quadratic: QuadraticSolver::new(),
            polynomial: PolynomialSolver::new(),
            transcendental: TranscendentalSolver::new(),
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
        // TODO: Analyze equation and dispatch to appropriate solver
        // Priority order: linear -> quadratic -> polynomial -> transcendental
        if self.linear.can_solve(equation) {
            self.linear.solve(equation, variable)
        } else if self.quadratic.can_solve(equation) {
            self.quadratic.solve(equation, variable)
        } else if self.polynomial.can_solve(equation) {
            self.polynomial.solve(equation, variable)
        } else if self.transcendental.can_solve(equation) {
            self.transcendental.solve(equation, variable)
        } else {
            Err(SolverError::UnsupportedEquationType)
        }
    }

    fn can_solve(&self, equation: &Equation) -> bool {
        self.linear.can_solve(equation)
            || self.quadratic.can_solve(equation)
            || self.polynomial.can_solve(equation)
            || self.transcendental.can_solve(equation)
    }
}

// ============================================================================
// High-Level API
// ============================================================================

/// Solve an equation for a specific variable.
///
/// This is the main entry point for solving equations. It attempts to solve
/// the equation symbolically, then substitutes known values and simplifies.
///
/// # Arguments
/// * `equation` - The equation to solve
/// * `target` - The name of the variable to solve for
/// * `known_values` - HashMap of known variable values
///
/// # Returns
/// A ResolutionPath showing the solution steps and final result
///
/// # Errors
/// Returns SolverError if the equation cannot be solved
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

/// Substitute known variable values into an expression.
fn substitute_values(expr: &Expression, values: &HashMap<String, f64>) -> Expression {
    match expr {
        Expression::Variable(v) => {
            if let Some(&value) = values.get(&v.name) {
                Expression::Float(value)
            } else {
                expr.clone()
            }
        }
        Expression::Unary(op, inner) => {
            Expression::Unary(*op, Box::new(substitute_values(inner, values)))
        }
        Expression::Binary(op, left, right) => Expression::Binary(
            *op,
            Box::new(substitute_values(left, values)),
            Box::new(substitute_values(right, values)),
        ),
        Expression::Function(func, args) => Expression::Function(
            func.clone(),
            args.iter()
                .map(|arg| substitute_values(arg, values))
                .collect(),
        ),
        Expression::Power(base, exp) => Expression::Power(
            Box::new(substitute_values(base, values)),
            Box::new(substitute_values(exp, values)),
        ),
        _ => expr.clone(),
    }
}

/// Compute a partial derivative of an output variable with respect to an input variable.
///
/// Given an equation defining the output variable in terms of input variables,
/// this function computes the partial derivative ∂output/∂input and evaluates
/// it at the given values.
///
/// # Arguments
/// * `equation` - Equation defining the output variable (e.g., "V = l * w * h")
/// * `output_var` - Name of the output variable (e.g., "V")
/// * `input_var` - Name of the input variable to differentiate with respect to (e.g., "l")
/// * `values` - HashMap of variable values at which to evaluate the derivative
///
/// # Returns
/// The numerical value of the partial derivative at the given point
///
/// # Errors
/// Returns `SolverError` if the equation cannot be solved for the output variable
/// or if evaluation fails due to missing variables
///
/// # Example
/// ```ignore
/// use mathsolver_core::ast::{Equation, Expression, Variable, BinaryOp};
/// use mathsolver_core::solver::compute_partial_derivative;
/// use std::collections::HashMap;
///
/// // Equation: V = l * w * h
/// let l = Expression::Variable(Variable::new("l"));
/// let w = Expression::Variable(Variable::new("w"));
/// let h = Expression::Variable(Variable::new("h"));
/// let lw = Expression::Binary(BinaryOp::Mul, Box::new(l), Box::new(w));
/// let volume = Expression::Binary(BinaryOp::Mul, Box::new(lw), Box::new(h));
/// let equation = Equation::new("box_volume", Expression::Variable(Variable::new("V")), volume);
///
/// let mut values = HashMap::new();
/// values.insert("l".to_string(), 2.0);
/// values.insert("w".to_string(), 3.0);
/// values.insert("h".to_string(), 4.0);
///
/// // ∂V/∂l = w * h = 3 * 4 = 12
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
    // If equation is "output = expr", use expr
    // If equation is "expr = output", use expr
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
        SolverError::Other(format!(
            "Failed to evaluate derivative - missing or invalid values"
        ))
    })
}

/// Compute all partial derivatives of an output variable with respect to all input variables.
///
/// Given an equation defining the output variable in terms of input variables,
/// this function computes all partial derivatives ∂output/∂input_i and evaluates
/// them at the given values.
///
/// # Arguments
/// * `equation` - Equation defining the output variable
/// * `output_var` - Name of the output variable
/// * `input_vars` - List of input variable names to compute derivatives for
/// * `values` - HashMap of variable values at which to evaluate the derivatives
///
/// # Returns
/// A HashMap mapping each input variable name to its partial derivative value
///
/// # Errors
/// Returns `SolverError` if any partial derivative computation fails
///
/// # Example
/// ```ignore
/// use mathsolver_core::ast::{Equation, Expression, Variable, BinaryOp};
/// use mathsolver_core::solver::compute_all_partial_derivatives;
/// use std::collections::HashMap;
///
/// // Equation: V = l * w * h
/// let l = Expression::Variable(Variable::new("l"));
/// let w = Expression::Variable(Variable::new("w"));
/// let h = Expression::Variable(Variable::new("h"));
/// let lw = Expression::Binary(BinaryOp::Mul, Box::new(l), Box::new(w));
/// let volume = Expression::Binary(BinaryOp::Mul, Box::new(lw), Box::new(h));
/// let equation = Equation::new("box_volume", Expression::Variable(Variable::new("V")), volume);
///
/// let mut values = HashMap::new();
/// values.insert("l".to_string(), 2.0);
/// values.insert("w".to_string(), 3.0);
/// values.insert("h".to_string(), 4.0);
///
/// let input_vars = vec!["l".to_string(), "w".to_string(), "h".to_string()];
/// let derivatives = compute_all_partial_derivatives(&equation, "V", &input_vars, &values).unwrap();
///
/// assert_eq!(derivatives.get("l").unwrap(), &12.0); // w * h
/// assert_eq!(derivatives.get("w").unwrap(), &8.0);  // l * h
/// assert_eq!(derivatives.get("h").unwrap(), &6.0);  // l * w
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
