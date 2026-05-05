//! Error types, result types, and solution enums for equation solving.
//!
//! Public types stay `Expression`-based. Internal solvers work in `Arc<Expr>`
//! and build `Solution` values via the `*_from_expr` constructors, which
//! decompile to `Expression` at the boundary.

use std::sync::Arc;

use crate::ast::{Expression, Variable};
use crate::numeric::compile::decompile;
use crate::numeric::Expr;

/// Error types for equation solving.
///
/// These errors represent different failure modes when attempting to solve
/// an equation symbolically.
///
/// # Examples
///
/// ```
/// use thales::solver::{SolverError, LinearSolver, Solver};
/// use thales::ast::{Equation, Expression, Variable, BinaryOp};
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
#[non_exhaustive]
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

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverError::NoSolution => write!(f, "Equation has no solution"),
            SolverError::InfiniteSolutions => write!(f, "Equation has infinite solutions"),
            SolverError::CannotSolve(msg) => write!(f, "Cannot solve: {}", msg),
            SolverError::UnsupportedEquationType => write!(f, "Equation type is not supported"),
            SolverError::DivisionByZero => write!(f, "Division by zero encountered"),
            SolverError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for SolverError {}

/// Explains why symbolic solving could not produce a closed-form solution.
///
/// This enum provides structured information about the mathematical reason
/// a symbolic solver failed, enabling informed handoff to numerical methods.
///
/// # Examples
///
/// ```
/// use thales::solver::SymbolicFailureReason;
///
/// let reason = SymbolicFailureReason::NoElementaryInverse {
///     description: "x*e^x cannot be inverted with elementary functions".to_string(),
///     special_function: Some("Lambert W".to_string()),
/// };
/// println!("{}", reason);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolicFailureReason {
    /// No elementary inverse exists (e.g., x*e^x requires Lambert W).
    NoElementaryInverse {
        /// Human-readable description of why no inverse exists
        description: String,
        /// Name of a special function that could solve it, if known
        special_function: Option<String>,
    },
    /// Variable appears in multiple non-combinable positions.
    NonIsolable {
        /// Explanation of why the variable cannot be isolated
        reason: String,
        /// Number of distinct occurrences of the variable
        occurrences: usize,
    },
    /// Equation is transcendental with no known closed form.
    Transcendental {
        /// Classification of the transcendental equation type
        equation_type: String,
    },
    /// System is underdetermined.
    Underdetermined {
        /// Number of equations in the system
        equations: usize,
        /// Number of unknowns in the system
        unknowns: usize,
    },
    /// Polynomial degree too high for radical solution (degree > 4).
    HighDegreePolynomial {
        /// Degree of the polynomial
        degree: usize,
    },
}

impl std::fmt::Display for SymbolicFailureReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymbolicFailureReason::NoElementaryInverse {
                description,
                special_function,
            } => {
                write!(f, "No elementary inverse: {}", description)?;
                if let Some(func) = special_function {
                    write!(f, " (requires {})", func)?;
                }
                Ok(())
            }
            SymbolicFailureReason::NonIsolable {
                reason,
                occurrences,
            } => {
                write!(
                    f,
                    "Variable not isolable: {} ({} occurrences)",
                    reason, occurrences
                )
            }
            SymbolicFailureReason::Transcendental { equation_type } => {
                write!(
                    f,
                    "Transcendental equation with no closed form: {}",
                    equation_type
                )
            }
            SymbolicFailureReason::Underdetermined {
                equations,
                unknowns,
            } => {
                write!(
                    f,
                    "Underdetermined system: {} equations, {} unknowns",
                    equations, unknowns
                )
            }
            SymbolicFailureReason::HighDegreePolynomial { degree } => {
                write!(
                    f,
                    "Polynomial of degree {} has no general radical solution",
                    degree
                )
            }
        }
    }
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
/// use thales::solver::{Solution, LinearSolver, Solver};
/// use thales::ast::{Equation, Expression, Variable, BinaryOp};
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

impl Solution {
    /// Build `Solution::Unique` from an internal `Arc<Expr>` result.
    ///
    /// Decompiles the `Expr` to the public `Expression` form at the boundary.
    pub fn unique_from_expr(expr: &Arc<Expr>) -> Self {
        Solution::Unique(decompile(expr))
    }

    /// Build `Solution::Multiple` from internal `Arc<Expr>` results.
    pub fn multiple_from_expr(exprs: &[Arc<Expr>]) -> Self {
        Solution::Multiple(exprs.iter().map(|e| decompile(e)).collect())
    }

    /// Build `Solution::Parametric` from an internal `Arc<Expr>` expression
    /// and already-constructed constraints.
    pub fn parametric_from_expr(expr: &Arc<Expr>, constraints: Vec<Constraint>) -> Self {
        Solution::Parametric {
            expression: decompile(expr),
            constraints,
        }
    }
}

/// Constraint on a solution.
///
/// Represents a condition that must be satisfied for a solution to be valid.
/// Typically used with parametric solutions to specify domain restrictions.
///
/// # Examples
///
/// ```
/// use thales::ast::{Variable, Expression};
/// use thales::solver::Constraint;
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

impl Constraint {
    /// Build a `Constraint` from an internal `Arc<Expr>` condition,
    /// decompiling at the boundary.
    pub fn from_expr(variable: Variable, condition: &Arc<Expr>) -> Self {
        Self {
            variable,
            condition: decompile(condition),
        }
    }
}
