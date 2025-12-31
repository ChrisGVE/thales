//! Resolution path tracking for equation solving.
//!
//! This module provides a comprehensive system for tracking and explaining the step-by-step
//! solution process when solving algebraic equations. It enables educational applications
//! to show students not just the answer, but the complete reasoning path.
//!
//! # Core Concepts
//!
//! The resolution path system consists of three main components:
//!
//! - **[`ResolutionPath`]**: The complete solution journey from initial equation to final answer,
//!   including all intermediate steps and transformations.
//! - **[`ResolutionStep`]**: A single operation in the solution process, consisting of the
//!   operation performed, why it was chosen, and the resulting expression state.
//! - **[`Operation`]**: The type of algebraic manipulation applied (e.g., "add to both sides",
//!   "simplify", "factor").
//!
//! # Use Cases
//!
//! This module is designed for:
//!
//! - **Educational Apps**: Show students how to solve equations step-by-step
//! - **Hint Generation**: Provide progressive hints based on solution steps
//! - **Solution Validation**: Verify that each step is mathematically sound
//! - **Multiple Solution Methods**: Compare different approaches to the same problem
//! - **Slide Rule Training**: Generate appropriate hints for slide rule calculations
//!
//! # Example: Building a Solution Path
//!
//! ```rust
//! use mathsolver_core::resolution_path::{ResolutionPathBuilder, Operation};
//! use mathsolver_core::ast::{Expression, Variable};
//!
//! // Solve: 2x + 3 = 7
//! let initial = Expression::Integer(7); // Starting from right side
//! let x = Variable::new("x");
//!
//! let path = ResolutionPathBuilder::new(initial)
//!     .step(
//!         Operation::SubtractBothSides(Expression::Integer(3)),
//!         "Subtract 3 from both sides to isolate the term with x".to_string(),
//!         Expression::Integer(4), // 7 - 3 = 4
//!     )
//!     .step(
//!         Operation::DivideBothSides(Expression::Integer(2)),
//!         "Divide both sides by 2 to solve for x".to_string(),
//!         Expression::Integer(2), // 4 / 2 = 2
//!     )
//!     .finish(Expression::Integer(2));
//!
//! assert_eq!(path.step_count(), 2);
//! assert!(!path.is_empty());
//! ```
//!
//! # Example: Using the Fluent Builder API
//!
//! ```rust
//! use mathsolver_core::resolution_path::ResolutionPathBuilder;
//! use mathsolver_core::ast::{Expression, Variable};
//!
//! let x = Variable::new("x");
//! let initial = Expression::Integer(10);
//!
//! // Specialized builder methods for common operations
//! let path = ResolutionPathBuilder::new(initial)
//!     .simplify("Combine like terms".to_string(), Expression::Integer(10))
//!     .isolate(x, "Isolate x on left side".to_string(), Expression::Integer(5))
//!     .finish(Expression::Integer(5));
//!
//! assert_eq!(path.step_count(), 2);
//! ```
//!
//! # Integration with Solver
//!
//! Solvers implementing the [`crate::solver::Solver`] trait return both the solution
//! and the resolution path, enabling applications to show the work:
//!
//! ```ignore
//! use mathsolver_core::solver::Solver;
//! use mathsolver_core::ast::{Equation, Variable};
//!
//! let equation = /* ... */;
//! let variable = Variable::new("x");
//!
//! let (solution, path) = solver.solve(&equation, &variable)?;
//!
//! // Generate hints for students
//! for (i, step) in path.steps.iter().enumerate() {
//!     println!("Step {}: {}", i + 1, step.explanation);
//!     println!("  Operation: {}", step.operation.describe());
//!     println!("  Result: {:?}", step.result);
//! }
//! ```
//!
//! # Slide Rule Application Integration
//!
//! For the SlideRuleCoach app, resolution paths can generate appropriate calculation hints:
//!
//! ```ignore
//! // When solving problems like "Find the area: A = πr²"
//! // The path can suggest which slide rule scales to use
//! let path = ResolutionPathBuilder::new(equation)
//!     .step(
//!         Operation::Substitute { variable: r, value: Expression::Float(2.5) },
//!         "Substitute r = 2.5 into the formula".to_string(),
//!         substituted_expr,
//!     )
//!     .step(
//!         Operation::PowerBothSides(Expression::Integer(2)),
//!         "Square the radius using A and B scales on slide rule".to_string(),
//!         squared_expr,
//!     )
//!     .step(
//!         Operation::MultiplyBothSides(Expression::Float(3.14159)),
//!         "Multiply by π using C and D scales".to_string(),
//!         final_result,
//!     )
//!     .finish(final_result);
//! ```

use crate::ast::{Expression, Variable};
use serde_json::{json, Value as JsonValue};

/// Level of detail for resolution path output.
///
/// Controls how much detail is included when rendering a resolution path
/// to text, LaTeX, or other formats.
///
/// # Example
///
/// ```rust
/// use mathsolver_core::resolution_path::{ResolutionPath, Verbosity};
/// use mathsolver_core::ast::Expression;
///
/// let path = ResolutionPath::new(Expression::Integer(10));
/// let minimal = path.to_text(Verbosity::Minimal);
/// let detailed = path.to_text(Verbosity::Detailed);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Verbosity {
    /// Key transformations only - shows just the essential operations.
    ///
    /// Best for quick overviews or when the user is already familiar
    /// with the solution technique.
    Minimal,

    /// Major steps - standard level of detail for educational purposes.
    ///
    /// Shows all significant steps without overwhelming detail.
    #[default]
    Standard,

    /// Every algebraic manipulation - maximum detail for learning.
    ///
    /// Includes all intermediate steps and detailed explanations,
    /// ideal for students learning the technique for the first time.
    Detailed,
}

/// Statistics about a resolution path.
///
/// Provides aggregate information about the solution process, useful for
/// analyzing problem difficulty, comparing solution methods, or tracking
/// student progress.
///
/// # Example
///
/// ```rust
/// use mathsolver_core::resolution_path::{ResolutionPath, ResolutionStep, Operation};
/// use mathsolver_core::ast::Expression;
///
/// let mut path = ResolutionPath::new(Expression::Integer(10));
/// path.add_step(ResolutionStep::new(
///     Operation::Simplify,
///     "Simplify".to_string(),
///     Expression::Integer(10),
/// ));
/// path.add_step(ResolutionStep::new(
///     Operation::DivideBothSides(Expression::Integer(2)),
///     "Divide by 2".to_string(),
///     Expression::Integer(5),
/// ));
///
/// let stats = path.statistics();
/// assert_eq!(stats.total_steps, 2);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PathStatistics {
    /// Total number of steps in the path.
    pub total_steps: usize,

    /// Count of operations by category.
    pub operation_counts: OperationCounts,

    /// Unique operation types used.
    pub unique_operations: usize,

    /// Whether the path uses advanced methods (quadratic formula, etc.).
    pub uses_advanced_methods: bool,

    /// Whether the path involves calculus operations.
    pub uses_calculus: bool,

    /// Whether the path involves matrix operations.
    pub uses_matrix_operations: bool,
}

/// Count of operations by category.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OperationCounts {
    /// Both-sides operations (add, subtract, multiply, divide, etc.)
    pub both_sides: usize,
    /// Expression transformations (simplify, expand, factor, etc.)
    pub transformations: usize,
    /// Variable manipulations (substitute, isolate, move term)
    pub variable_ops: usize,
    /// Identity applications (algebraic, trig, log)
    pub identities: usize,
    /// Advanced methods (quadratic formula, complete square, numerical)
    pub advanced: usize,
    /// Calculus operations (differentiate, integrate, limit, etc.)
    pub calculus: usize,
    /// Matrix operations
    pub matrix: usize,
    /// Custom operations
    pub custom: usize,
}

/// Represents a complete solution path from equation to answer.
///
/// A `ResolutionPath` captures the entire solution journey, from the initial equation
/// state through all intermediate transformations to the final result. This enables
/// educational applications to provide step-by-step explanations and progressive hints.
///
/// # Structure
///
/// - `initial`: The starting equation or expression state before any operations
/// - `steps`: Ordered sequence of operations and their results
/// - `result`: The final simplified expression after all operations
///
/// # Example
///
/// ```rust
/// use mathsolver_core::resolution_path::{ResolutionPath, ResolutionStep, Operation};
/// use mathsolver_core::ast::Expression;
///
/// // Create a path for solving 5x = 15
/// let mut path = ResolutionPath::new(Expression::Integer(15));
///
/// // Add step: divide both sides by 5
/// path.add_step(ResolutionStep::new(
///     Operation::DivideBothSides(Expression::Integer(5)),
///     "Divide both sides by 5 to isolate x".to_string(),
///     Expression::Integer(3),
/// ));
///
/// path.set_result(Expression::Integer(3));
///
/// assert_eq!(path.step_count(), 1);
/// assert!(!path.is_empty());
/// ```
///
/// # Educational Use
///
/// Resolution paths support progressive hint systems:
///
/// ```rust
/// use mathsolver_core::resolution_path::ResolutionPath;
/// use mathsolver_core::ast::Expression;
///
/// let path = ResolutionPath::new(Expression::Integer(10));
/// // ... add steps ...
///
/// // Hint level 1: Show number of steps needed
/// println!("This problem requires {} steps", path.step_count());
///
/// // Hint level 2: Show first operation type
/// if let Some(first_step) = path.steps.first() {
///     println!("First, try: {}", first_step.operation.describe());
/// }
///
/// // Hint level 3: Show full explanation
/// if let Some(first_step) = path.steps.first() {
///     println!("{}", first_step.explanation);
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ResolutionPath {
    /// Initial equation state before any transformations.
    ///
    /// This represents the starting point of the solution process,
    /// typically the original equation or one side of it.
    pub initial: Expression,

    /// Sequence of operations applied to reach the solution.
    ///
    /// Each step includes the operation performed, an explanation of why,
    /// and the resulting expression state. Steps are ordered chronologically.
    pub steps: Vec<ResolutionStep>,

    /// Final simplified result after all operations.
    ///
    /// This is the solution to the equation, fully simplified.
    /// For example, solving "2x + 3 = 7" would result in "x = 2".
    pub result: Expression,
}

impl ResolutionPath {
    /// Create a new resolution path starting from an initial expression.
    ///
    /// The path begins with no steps, and the result is initially set to
    /// the same value as the initial expression.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mathsolver_core::resolution_path::ResolutionPath;
    /// use mathsolver_core::ast::Expression;
    ///
    /// let initial = Expression::Integer(42);
    /// let path = ResolutionPath::new(initial.clone());
    ///
    /// assert_eq!(path.initial, initial);
    /// assert_eq!(path.result, initial);
    /// assert!(path.is_empty());
    /// ```
    pub fn new(initial: Expression) -> Self {
        Self {
            initial: initial.clone(),
            steps: Vec::new(),
            result: initial,
        }
    }

    /// Add a step to the resolution path.
    ///
    /// Steps are appended in chronological order. Each step records the
    /// operation performed, the reasoning behind it, and the resulting
    /// expression state.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mathsolver_core::resolution_path::{ResolutionPath, ResolutionStep, Operation};
    /// use mathsolver_core::ast::Expression;
    ///
    /// let mut path = ResolutionPath::new(Expression::Integer(10));
    ///
    /// path.add_step(ResolutionStep::new(
    ///     Operation::Simplify,
    ///     "Simplify the expression".to_string(),
    ///     Expression::Integer(10),
    /// ));
    ///
    /// assert_eq!(path.step_count(), 1);
    /// ```
    pub fn add_step(&mut self, step: ResolutionStep) {
        self.steps.push(step);
    }

    /// Set the final result of the solution path.
    ///
    /// This should be called after all steps have been added to record
    /// the final simplified solution.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mathsolver_core::resolution_path::ResolutionPath;
    /// use mathsolver_core::ast::Expression;
    ///
    /// let mut path = ResolutionPath::new(Expression::Integer(10));
    /// // ... add steps ...
    /// path.set_result(Expression::Integer(5));
    ///
    /// assert_eq!(path.result, Expression::Integer(5));
    /// ```
    pub fn set_result(&mut self, result: Expression) {
        self.result = result;
    }

    /// Get the number of steps in the path.
    ///
    /// Returns the count of operations that have been applied. This is useful
    /// for educational applications to indicate problem complexity or to
    /// track student progress.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mathsolver_core::resolution_path::{ResolutionPath, ResolutionStep, Operation};
    /// use mathsolver_core::ast::Expression;
    ///
    /// let mut path = ResolutionPath::new(Expression::Integer(20));
    ///
    /// assert_eq!(path.step_count(), 0);
    ///
    /// path.add_step(ResolutionStep::new(
    ///     Operation::Simplify,
    ///     "Simplify".to_string(),
    ///     Expression::Integer(20),
    /// ));
    ///
    /// assert_eq!(path.step_count(), 1);
    /// ```
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Check if the path is empty (no operations applied).
    ///
    /// Returns `true` if no steps have been added to the path yet.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mathsolver_core::resolution_path::{ResolutionPath, ResolutionStep, Operation};
    /// use mathsolver_core::ast::Expression;
    ///
    /// let mut path = ResolutionPath::new(Expression::Integer(100));
    /// assert!(path.is_empty());
    ///
    /// path.add_step(ResolutionStep::new(
    ///     Operation::Simplify,
    ///     "Simplify".to_string(),
    ///     Expression::Integer(100),
    /// ));
    ///
    /// assert!(!path.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Generate human-readable explanation of the solution path.
    ///
    /// This method produces a formatted text description of all steps taken
    /// to solve the equation. Currently returns a summary of step count;
    /// full implementation will include detailed step-by-step explanations.
    ///
    /// # Future Implementation
    ///
    /// - Format each step with LaTeX or plain text
    /// - Include reasoning for each operation
    /// - Support different verbosity levels
    /// - Provide slide rule-specific hints
    ///
    /// # Example
    ///
    /// ```rust
    /// use mathsolver_core::resolution_path::ResolutionPath;
    /// use mathsolver_core::ast::Expression;
    ///
    /// let path = ResolutionPath::new(Expression::Integer(7));
    /// let explanation = path.explain();
    /// assert!(explanation.contains("0 steps"));
    /// ```
    pub fn explain(&self) -> String {
        // TODO: Implement explanation generation
        // TODO: Format each step with LaTeX or plain text
        // TODO: Include reasoning for each operation
        format!("Solution path with {} steps", self.steps.len())
    }

    /// Convert the resolution path to a human-readable text format.
    ///
    /// Generates a formatted text representation of the solution path,
    /// with detail level controlled by the verbosity parameter.
    ///
    /// # Arguments
    ///
    /// * `verbosity` - Level of detail to include in the output
    ///
    /// # Example
    ///
    /// ```rust
    /// use mathsolver_core::resolution_path::{ResolutionPath, ResolutionStep, Operation, Verbosity};
    /// use mathsolver_core::ast::Expression;
    ///
    /// let mut path = ResolutionPath::new(Expression::Integer(10));
    /// path.add_step(ResolutionStep::new(
    ///     Operation::SubtractBothSides(Expression::Integer(3)),
    ///     "Subtract 3 from both sides".to_string(),
    ///     Expression::Integer(7),
    /// ));
    /// path.set_result(Expression::Integer(7));
    ///
    /// let text = path.to_text(Verbosity::Standard);
    /// assert!(text.contains("Step 1"));
    /// assert!(text.contains("Subtract"));
    /// ```
    pub fn to_text(&self, verbosity: Verbosity) -> String {
        let mut output = String::new();

        match verbosity {
            Verbosity::Minimal => {
                // Just initial, key operations, and result
                output.push_str(&format!("Start: {:?}\n", self.initial));
                for (i, step) in self.steps.iter().enumerate() {
                    if step.operation.is_key_operation() {
                        output.push_str(&format!(
                            "Step {}: {}\n",
                            i + 1,
                            step.operation.describe()
                        ));
                    }
                }
                output.push_str(&format!("Result: {:?}\n", self.result));
            }
            Verbosity::Standard => {
                // All steps with operation and result
                output.push_str(&format!("Initial: {:?}\n\n", self.initial));
                for (i, step) in self.steps.iter().enumerate() {
                    output.push_str(&format!("Step {}: {}\n", i + 1, step.operation.describe()));
                    output.push_str(&format!("  → {:?}\n\n", step.result));
                }
                output.push_str(&format!("Final result: {:?}\n", self.result));
            }
            Verbosity::Detailed => {
                // Full explanation including reasoning
                output.push_str(&format!("=== Solution Path ===\n\n"));
                output.push_str(&format!("Starting expression: {:?}\n\n", self.initial));
                for (i, step) in self.steps.iter().enumerate() {
                    output.push_str(&format!("--- Step {} ---\n", i + 1));
                    output.push_str(&format!("Operation: {}\n", step.operation.describe()));
                    output.push_str(&format!("Explanation: {}\n", step.explanation));
                    output.push_str(&format!("Result: {:?}\n\n", step.result));
                }
                output.push_str(&format!("=== Final Result ===\n"));
                output.push_str(&format!("{:?}\n", self.result));
            }
        }

        output
    }

    /// Convert the resolution path to LaTeX format.
    ///
    /// Generates a LaTeX representation of the solution path, suitable
    /// for rendering in documents, educational materials, or web applications.
    ///
    /// # Arguments
    ///
    /// * `verbosity` - Level of detail to include in the output
    ///
    /// # Example
    ///
    /// ```rust
    /// use mathsolver_core::resolution_path::{ResolutionPath, ResolutionStep, Operation, Verbosity};
    /// use mathsolver_core::ast::Expression;
    ///
    /// let mut path = ResolutionPath::new(Expression::Integer(10));
    /// path.add_step(ResolutionStep::new(
    ///     Operation::DivideBothSides(Expression::Integer(2)),
    ///     "Divide by 2".to_string(),
    ///     Expression::Integer(5),
    /// ));
    /// path.set_result(Expression::Integer(5));
    ///
    /// let latex = path.to_latex(Verbosity::Standard);
    /// assert!(latex.contains("\\begin{align*}"));
    /// ```
    pub fn to_latex(&self, verbosity: Verbosity) -> String {
        let mut output = String::new();

        output.push_str("\\begin{align*}\n");

        match verbosity {
            Verbosity::Minimal => {
                output.push_str(&format!("  & {} \\\\\n", self.initial.to_latex()));
                // Show only key steps
                for step in &self.steps {
                    if step.operation.is_key_operation() {
                        output.push_str(&format!(
                            "  &\\quad \\text{{{}}} \\\\\n",
                            step.operation.describe_latex()
                        ));
                        output.push_str(&format!("  &= {} \\\\\n", step.result.to_latex()));
                    }
                }
            }
            Verbosity::Standard => {
                output.push_str(&format!("  & {} \\\\\n", self.initial.to_latex()));
                for step in &self.steps {
                    output.push_str(&format!(
                        "  &\\quad \\text{{{}}} \\\\\n",
                        step.operation.describe_latex()
                    ));
                    output.push_str(&format!("  &= {} \\\\\n", step.result.to_latex()));
                }
            }
            Verbosity::Detailed => {
                output.push_str(&format!(
                    "  & \\text{{Initial: }} {} \\\\\n",
                    self.initial.to_latex()
                ));
                for (i, step) in self.steps.iter().enumerate() {
                    output.push_str(&format!(
                        "  &\\quad \\text{{Step {}: {}}} \\\\\n",
                        i + 1,
                        step.operation.describe_latex()
                    ));
                    output.push_str(&format!(
                        "  &\\quad \\text{{({}}})\\\\\n",
                        escape_latex_text(&step.explanation)
                    ));
                    output.push_str(&format!("  &= {} \\\\\n", step.result.to_latex()));
                }
            }
        }

        output.push_str("\\end{align*}\n");
        output
    }

    /// Convert the resolution path to JSON format.
    ///
    /// Generates a JSON representation of the complete solution path,
    /// including all steps, explanations, and metadata. Useful for
    /// API responses, storage, or programmatic manipulation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mathsolver_core::resolution_path::{ResolutionPath, ResolutionStep, Operation};
    /// use mathsolver_core::ast::Expression;
    ///
    /// let mut path = ResolutionPath::new(Expression::Integer(10));
    /// path.add_step(ResolutionStep::new(
    ///     Operation::Simplify,
    ///     "Combine terms".to_string(),
    ///     Expression::Integer(10),
    /// ));
    /// path.set_result(Expression::Integer(10));
    ///
    /// let json = path.to_json();
    /// assert!(json["steps"].is_array());
    /// assert_eq!(json["step_count"], 1);
    /// ```
    pub fn to_json(&self) -> JsonValue {
        let steps: Vec<JsonValue> = self
            .steps
            .iter()
            .enumerate()
            .map(|(i, step)| {
                json!({
                    "step_number": i + 1,
                    "operation": step.operation.describe(),
                    "operation_category": step.operation.category(),
                    "explanation": step.explanation,
                    "result": format!("{:?}", step.result),
                    "result_latex": step.result.to_latex(),
                })
            })
            .collect();

        let stats = self.statistics();

        json!({
            "initial": format!("{:?}", self.initial),
            "initial_latex": self.initial.to_latex(),
            "steps": steps,
            "result": format!("{:?}", self.result),
            "result_latex": self.result.to_latex(),
            "step_count": self.steps.len(),
            "statistics": {
                "total_steps": stats.total_steps,
                "unique_operations": stats.unique_operations,
                "uses_advanced_methods": stats.uses_advanced_methods,
                "uses_calculus": stats.uses_calculus,
                "uses_matrix_operations": stats.uses_matrix_operations,
                "operation_counts": {
                    "both_sides": stats.operation_counts.both_sides,
                    "transformations": stats.operation_counts.transformations,
                    "variable_ops": stats.operation_counts.variable_ops,
                    "identities": stats.operation_counts.identities,
                    "advanced": stats.operation_counts.advanced,
                    "calculus": stats.operation_counts.calculus,
                    "matrix": stats.operation_counts.matrix,
                    "custom": stats.operation_counts.custom,
                }
            }
        })
    }

    /// Calculate statistics about the resolution path.
    ///
    /// Returns aggregate information about the solution process, including
    /// operation counts by category and flags for advanced methods.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mathsolver_core::resolution_path::{ResolutionPath, ResolutionStep, Operation};
    /// use mathsolver_core::ast::Expression;
    ///
    /// let mut path = ResolutionPath::new(Expression::Integer(20));
    /// path.add_step(ResolutionStep::new(
    ///     Operation::Simplify,
    ///     "Simplify".to_string(),
    ///     Expression::Integer(20),
    /// ));
    /// path.add_step(ResolutionStep::new(
    ///     Operation::QuadraticFormula,
    ///     "Apply quadratic formula".to_string(),
    ///     Expression::Integer(5),
    /// ));
    ///
    /// let stats = path.statistics();
    /// assert_eq!(stats.total_steps, 2);
    /// assert!(stats.uses_advanced_methods);
    /// assert_eq!(stats.operation_counts.transformations, 1);
    /// assert_eq!(stats.operation_counts.advanced, 1);
    /// ```
    pub fn statistics(&self) -> PathStatistics {
        let mut counts = OperationCounts::default();
        let mut unique_ops: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut uses_advanced = false;
        let mut uses_calculus = false;
        let mut uses_matrix = false;

        for step in &self.steps {
            let category = step.operation.category();
            unique_ops.insert(step.operation.describe());

            match category.as_str() {
                "both_sides" => counts.both_sides += 1,
                "transformation" => counts.transformations += 1,
                "variable" => counts.variable_ops += 1,
                "identity" => counts.identities += 1,
                "advanced" => {
                    counts.advanced += 1;
                    uses_advanced = true;
                }
                "calculus" => {
                    counts.calculus += 1;
                    uses_calculus = true;
                }
                "matrix" => {
                    counts.matrix += 1;
                    uses_matrix = true;
                }
                "custom" => counts.custom += 1,
                _ => {}
            }
        }

        PathStatistics {
            total_steps: self.steps.len(),
            operation_counts: counts,
            unique_operations: unique_ops.len(),
            uses_advanced_methods: uses_advanced,
            uses_calculus,
            uses_matrix_operations: uses_matrix,
        }
    }
}

/// Escape special LaTeX characters in text.
fn escape_latex_text(text: &str) -> String {
    text.replace('\\', "\\textbackslash{}")
        .replace('{', "\\{")
        .replace('}', "\\}")
        .replace('$', "\\$")
        .replace('%', "\\%")
        .replace('&', "\\&")
        .replace('#', "\\#")
        .replace('_', "\\_")
        .replace('^', "\\^{}")
        .replace('~', "\\~{}")
}

/// A single step in the solution process.
///
/// Each `ResolutionStep` represents one algebraic operation applied during equation
/// solving. It captures what was done, why it was done, and what the result was.
///
/// # Structure
///
/// - `operation`: The type of algebraic manipulation (e.g., add to both sides, simplify)
/// - `explanation`: Human-readable reasoning for why this step was chosen
/// - `result`: The expression state after applying the operation
///
/// # Example
///
/// ```rust
/// use mathsolver_core::resolution_path::{ResolutionStep, Operation};
/// use mathsolver_core::ast::Expression;
///
/// // Step: Divide both sides by 2
/// let step = ResolutionStep::new(
///     Operation::DivideBothSides(Expression::Integer(2)),
///     "Divide both sides by 2 to isolate x".to_string(),
///     Expression::Integer(5),
/// );
///
/// assert_eq!(step.operation.describe(), "Divide both sides by Integer(2)");
/// assert_eq!(step.result, Expression::Integer(5));
/// ```
///
/// # Educational Context
///
/// Steps can be revealed progressively to provide hints:
///
/// ```rust
/// use mathsolver_core::resolution_path::{ResolutionStep, Operation};
/// use mathsolver_core::ast::Expression;
///
/// let step = ResolutionStep::new(
///     Operation::Simplify,
///     "Combine like terms: 3x + 2x = 5x".to_string(),
///     Expression::Integer(5),
/// );
///
/// // Hint level 1: Operation type only
/// println!("Try: {}", step.operation.describe());
///
/// // Hint level 2: Full explanation
/// println!("Explanation: {}", step.explanation);
///
/// // Hint level 3: Show result
/// println!("Result: {:?}", step.result);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ResolutionStep {
    /// The operation performed in this step.
    ///
    /// Describes the algebraic manipulation applied (e.g., adding to both sides,
    /// factoring, simplifying). See [`Operation`] for all available operation types.
    pub operation: Operation,

    /// Description of why this step was taken.
    ///
    /// Provides the educational reasoning behind the operation. For example:
    /// "Subtract 3 from both sides to isolate the term with x" or
    /// "Factor out the common term to simplify".
    pub explanation: String,

    /// Expression state after this step.
    ///
    /// The resulting expression after applying the operation. This becomes
    /// the input for the next step in the solution path.
    pub result: Expression,
}

impl ResolutionStep {
    /// Create a new resolution step.
    ///
    /// Constructs a step with the specified operation, explanation, and result.
    ///
    /// # Arguments
    ///
    /// * `operation` - The algebraic operation to perform
    /// * `explanation` - Human-readable reason for this step
    /// * `result` - Expression state after the operation
    ///
    /// # Example
    ///
    /// ```rust
    /// use mathsolver_core::resolution_path::{ResolutionStep, Operation};
    /// use mathsolver_core::ast::Expression;
    ///
    /// let step = ResolutionStep::new(
    ///     Operation::AddBothSides(Expression::Integer(5)),
    ///     "Add 5 to both sides to eliminate the negative term".to_string(),
    ///     Expression::Integer(10),
    /// );
    ///
    /// assert_eq!(step.result, Expression::Integer(10));
    /// ```
    pub fn new(operation: Operation, explanation: String, result: Expression) -> Self {
        Self {
            operation,
            explanation,
            result,
        }
    }
}

/// Operations that can be performed during equation solving.
///
/// This enum represents all types of algebraic manipulations that can be applied
/// when solving equations. Each variant captures a specific mathematical operation
/// along with any necessary parameters.
///
/// # Operation Categories
///
/// ## Both-Sides Operations
/// Operations that maintain equation equality by applying the same transformation
/// to both sides: [`AddBothSides`](Operation::AddBothSides),
/// [`SubtractBothSides`](Operation::SubtractBothSides),
/// [`MultiplyBothSides`](Operation::MultiplyBothSides),
/// [`DivideBothSides`](Operation::DivideBothSides),
/// [`PowerBothSides`](Operation::PowerBothSides),
/// [`RootBothSides`](Operation::RootBothSides),
/// [`ApplyFunction`](Operation::ApplyFunction).
///
/// ## Expression Transformations
/// Operations that restructure expressions:
/// [`Simplify`](Operation::Simplify),
/// [`Expand`](Operation::Expand),
/// [`Factor`](Operation::Factor),
/// [`CombineFractions`](Operation::CombineFractions),
/// [`Cancel`](Operation::Cancel).
///
/// ## Variable Manipulation
/// Operations for working with variables:
/// [`Substitute`](Operation::Substitute),
/// [`Isolate`](Operation::Isolate),
/// [`MoveTerm`](Operation::MoveTerm).
///
/// ## Identity Applications
/// Operations that apply mathematical identities:
/// [`ApplyIdentity`](Operation::ApplyIdentity),
/// [`ApplyTrigIdentity`](Operation::ApplyTrigIdentity),
/// [`ApplyLogProperty`](Operation::ApplyLogProperty).
///
/// ## Advanced Methods
/// Specialized solving techniques:
/// [`QuadraticFormula`](Operation::QuadraticFormula),
/// [`CompleteSquare`](Operation::CompleteSquare),
/// [`NumericalApproximation`](Operation::NumericalApproximation).
///
/// # Example
///
/// ```rust
/// use mathsolver_core::resolution_path::Operation;
/// use mathsolver_core::ast::{Expression, Variable};
///
/// // Both-sides operation
/// let op1 = Operation::AddBothSides(Expression::Integer(5));
/// assert_eq!(op1.describe(), "Add Integer(5) to both sides");
///
/// // Simplification
/// let op2 = Operation::Simplify;
/// assert_eq!(op2.describe(), "Simplify expression");
///
/// // Variable substitution
/// let x = Variable::new("x");
/// let op3 = Operation::Substitute {
///     variable: x,
///     value: Expression::Integer(10),
/// };
/// assert!(op3.describe().contains("Substitute"));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    // ===== Both-Sides Operations =====
    /// Add the same expression to both sides of the equation.
    ///
    /// Used to eliminate negative terms or move constants across the equation.
    ///
    /// # Example
    ///
    /// Solving `x - 3 = 7`:
    /// - Apply `AddBothSides(3)` to get `x = 10`
    AddBothSides(Expression),

    /// Subtract the same expression from both sides of the equation.
    ///
    /// Used to eliminate positive terms or move constants across the equation.
    ///
    /// # Example
    ///
    /// Solving `x + 5 = 12`:
    /// - Apply `SubtractBothSides(5)` to get `x = 7`
    SubtractBothSides(Expression),

    /// Multiply both sides of the equation by the same expression.
    ///
    /// Used to eliminate fractions or isolate variables with fractional coefficients.
    ///
    /// # Example
    ///
    /// Solving `x/2 = 6`:
    /// - Apply `MultiplyBothSides(2)` to get `x = 12`
    MultiplyBothSides(Expression),

    /// Divide both sides of the equation by the same expression.
    ///
    /// Used to isolate variables with integer coefficients.
    ///
    /// # Example
    ///
    /// Solving `3x = 15`:
    /// - Apply `DivideBothSides(3)` to get `x = 5`
    DivideBothSides(Expression),

    /// Raise both sides of the equation to the specified power.
    ///
    /// Used to eliminate roots or solve radical equations.
    ///
    /// # Example
    ///
    /// Solving `√x = 4`:
    /// - Apply `PowerBothSides(2)` to get `x = 16`
    PowerBothSides(Expression),

    /// Take the specified root of both sides.
    ///
    /// Used to solve equations with powers.
    ///
    /// # Example
    ///
    /// Solving `x² = 25`:
    /// - Apply `RootBothSides(2)` to get `x = ±5`
    RootBothSides(Expression),

    /// Apply a named function to both sides.
    ///
    /// Used for inverse operations like taking logarithms, applying trig functions, etc.
    ///
    /// # Example
    ///
    /// Solving `e^x = 10`:
    /// - Apply `ApplyFunction("ln")` to get `x = ln(10)`
    ApplyFunction(String),

    // ===== Expression Transformations =====
    /// Simplify the expression by combining like terms, reducing fractions, etc.
    ///
    /// # Example
    ///
    /// Transform `3x + 2x` into `5x`
    Simplify,

    /// Expand the expression by distributing multiplication over addition.
    ///
    /// # Example
    ///
    /// Transform `(x + 2)(x - 3)` into `x² - x - 6`
    Expand,

    /// Factor the expression into a product of simpler expressions.
    ///
    /// # Example
    ///
    /// Transform `x² - 5x + 6` into `(x - 2)(x - 3)`
    Factor,

    /// Combine multiple fractions into a single fraction.
    ///
    /// # Example
    ///
    /// Transform `1/x + 2/y` into `(y + 2x)/(xy)`
    CombineFractions,

    /// Cancel common factors from numerator and denominator.
    ///
    /// # Example
    ///
    /// Transform `(2x)/(4x)` into `1/2`
    Cancel,

    // ===== Variable Manipulation =====
    /// Substitute a variable with a specific value or expression.
    ///
    /// # Example
    ///
    /// In equation `2x + y = 10`, substitute `y = 3` to get `2x + 3 = 10`
    Substitute {
        /// The variable being replaced
        variable: Variable,
        /// The value or expression to substitute
        value: Expression,
    },

    /// Isolate the specified variable on one side of the equation.
    ///
    /// # Example
    ///
    /// Transform `2x + 3y = 12` to `x = (12 - 3y)/2` (isolating x)
    Isolate(Variable),

    /// Move a term from one side of the equation to the other.
    ///
    /// Equivalent to adding or subtracting the term from both sides.
    ///
    /// # Example
    ///
    /// Transform `x + 5 = 12` to `x = 12 - 5`
    MoveTerm(Expression),

    // ===== Identity Applications =====
    /// Apply a named algebraic identity.
    ///
    /// # Example
    ///
    /// Apply "difference of squares": `a² - b² = (a + b)(a - b)`
    ApplyIdentity(String),

    /// Apply a named trigonometric identity.
    ///
    /// # Example
    ///
    /// Apply "Pythagorean identity": `sin²θ + cos²θ = 1`
    ApplyTrigIdentity(String),

    /// Apply a named logarithm property.
    ///
    /// # Example
    ///
    /// Apply "product rule": `log(ab) = log(a) + log(b)`
    ApplyLogProperty(String),

    // ===== Advanced Methods =====
    /// Solve using the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a
    ///
    /// Used for equations in the form `ax² + bx + c = 0`
    QuadraticFormula,

    /// Solve by completing the square.
    ///
    /// Transform `x² + bx + c` into `(x + p)² + q` form
    CompleteSquare,

    /// Use numerical methods to approximate the solution.
    ///
    /// Applied when symbolic methods fail or exact solutions are impractical
    NumericalApproximation,

    // ===== Calculus Operations =====
    /// Differentiate an expression with respect to a variable.
    ///
    /// # Example
    ///
    /// Differentiate `x^3 + 2x` with respect to x: `3x^2 + 2`
    Differentiate {
        /// The variable to differentiate with respect to
        variable: Variable,
        /// The differentiation rule applied (e.g., "power rule", "chain rule")
        rule: String,
    },

    /// Integrate an expression with respect to a variable.
    ///
    /// # Example
    ///
    /// Integrate `2x` with respect to x: `x^2 + C`
    Integrate {
        /// The variable of integration
        variable: Variable,
        /// The integration technique used
        technique: String,
    },

    /// Evaluate a limit.
    ///
    /// # Example
    ///
    /// Evaluate `lim_{x->0} sin(x)/x = 1`
    EvaluateLimit {
        /// The variable approaching a value
        variable: Variable,
        /// The value being approached
        approaches: Expression,
        /// The method used (e.g., "direct substitution", "L'Hôpital's rule")
        method: String,
    },

    /// Apply integration by parts: ∫u dv = uv - ∫v du
    IntegrationByParts {
        /// The function chosen as u
        u: Expression,
        /// The function chosen as dv
        dv: Expression,
    },

    /// Apply u-substitution for integration.
    USubstitution {
        /// The substitution u = g(x)
        substitution: Expression,
    },

    /// Solve an ODE.
    SolveODE {
        /// The method used (e.g., "separation of variables", "integrating factor")
        method: String,
    },

    // ===== Matrix Operations =====
    /// Perform a matrix operation.
    MatrixOperation {
        /// The type of operation (e.g., "row reduction", "transpose", "inverse")
        operation: String,
    },

    /// Apply Gaussian elimination.
    GaussianElimination,

    /// Compute determinant using a specific method.
    ComputeDeterminant {
        /// The method used (e.g., "cofactor expansion", "LU decomposition")
        method: String,
    },

    /// Custom operation with a free-form description.
    ///
    /// Use this for operations not covered by the other variants
    Custom(String),
}

impl Operation {
    /// Get a human-readable description of this operation.
    ///
    /// Returns a string describing what this operation does, suitable for display
    /// in educational contexts, hints, or step-by-step explanations.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mathsolver_core::resolution_path::Operation;
    /// use mathsolver_core::ast::{Expression, Variable};
    ///
    /// let op = Operation::DivideBothSides(Expression::Integer(3));
    /// assert_eq!(op.describe(), "Divide both sides by Integer(3)");
    ///
    /// let op2 = Operation::Simplify;
    /// assert_eq!(op2.describe(), "Simplify expression");
    ///
    /// let x = Variable::new("x");
    /// let op3 = Operation::Isolate(x);
    /// assert_eq!(op3.describe(), "Isolate x");
    /// ```
    ///
    /// # Usage in Educational Applications
    ///
    /// ```rust
    /// use mathsolver_core::resolution_path::{ResolutionStep, Operation};
    /// use mathsolver_core::ast::Expression;
    ///
    /// let step = ResolutionStep::new(
    ///     Operation::AddBothSides(Expression::Integer(7)),
    ///     "Add 7 to eliminate the negative term".to_string(),
    ///     Expression::Integer(14),
    /// );
    ///
    /// // Show just the operation type to student
    /// println!("Hint: {}", step.operation.describe());
    /// // Output: "Add Integer(7) to both sides"
    /// ```
    pub fn describe(&self) -> String {
        match self {
            Operation::AddBothSides(expr) => format!("Add {:?} to both sides", expr),
            Operation::SubtractBothSides(expr) => format!("Subtract {:?} from both sides", expr),
            Operation::MultiplyBothSides(expr) => format!("Multiply both sides by {:?}", expr),
            Operation::DivideBothSides(expr) => format!("Divide both sides by {:?}", expr),
            Operation::PowerBothSides(expr) => format!("Raise both sides to power {:?}", expr),
            Operation::RootBothSides(expr) => format!("Take {:?} root of both sides", expr),
            Operation::ApplyFunction(func) => format!("Apply {} to both sides", func),
            Operation::Simplify => "Simplify expression".to_string(),
            Operation::Expand => "Expand expression".to_string(),
            Operation::Factor => "Factor expression".to_string(),
            Operation::CombineFractions => "Combine fractions".to_string(),
            Operation::Cancel => "Cancel common factors".to_string(),
            Operation::Substitute { variable, value } => {
                format!("Substitute {} = {:?}", variable, value)
            }
            Operation::Isolate(var) => format!("Isolate {}", var),
            Operation::MoveTerm(expr) => format!("Move {:?} to other side", expr),
            Operation::ApplyIdentity(name) => format!("Apply identity: {}", name),
            Operation::ApplyTrigIdentity(name) => format!("Apply trig identity: {}", name),
            Operation::ApplyLogProperty(name) => format!("Apply log property: {}", name),
            Operation::QuadraticFormula => "Apply quadratic formula".to_string(),
            Operation::CompleteSquare => "Complete the square".to_string(),
            Operation::NumericalApproximation => "Use numerical approximation".to_string(),
            // Calculus operations
            Operation::Differentiate { variable, rule } => {
                format!("Differentiate with respect to {} ({})", variable, rule)
            }
            Operation::Integrate { variable, technique } => {
                format!("Integrate with respect to {} ({})", variable, technique)
            }
            Operation::EvaluateLimit {
                variable,
                approaches,
                method,
            } => {
                format!(
                    "Evaluate limit as {} → {:?} ({})",
                    variable, approaches, method
                )
            }
            Operation::IntegrationByParts { u, dv } => {
                format!("Integration by parts: u = {:?}, dv = {:?}", u, dv)
            }
            Operation::USubstitution { substitution } => {
                format!("U-substitution: u = {:?}", substitution)
            }
            Operation::SolveODE { method } => format!("Solve ODE ({})", method),
            // Matrix operations
            Operation::MatrixOperation { operation } => {
                format!("Matrix operation: {}", operation)
            }
            Operation::GaussianElimination => "Apply Gaussian elimination".to_string(),
            Operation::ComputeDeterminant { method } => {
                format!("Compute determinant ({})", method)
            }
            Operation::Custom(desc) => desc.clone(),
        }
    }

    /// Get a LaTeX-friendly description of this operation.
    ///
    /// Similar to `describe()` but with proper escaping for LaTeX text mode.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mathsolver_core::resolution_path::Operation;
    /// use mathsolver_core::ast::Expression;
    ///
    /// let op = Operation::QuadraticFormula;
    /// assert_eq!(op.describe_latex(), "Apply quadratic formula");
    /// ```
    pub fn describe_latex(&self) -> String {
        // For now, reuse describe() - most descriptions are already LaTeX-safe
        // Special characters in expressions would need escaping
        match self {
            Operation::AddBothSides(expr) => {
                format!("Add {} to both sides", expr.to_latex())
            }
            Operation::SubtractBothSides(expr) => {
                format!("Subtract {} from both sides", expr.to_latex())
            }
            Operation::MultiplyBothSides(expr) => {
                format!("Multiply both sides by {}", expr.to_latex())
            }
            Operation::DivideBothSides(expr) => {
                format!("Divide both sides by {}", expr.to_latex())
            }
            Operation::PowerBothSides(expr) => {
                format!("Raise both sides to power {}", expr.to_latex())
            }
            Operation::RootBothSides(expr) => {
                format!("Take {} root of both sides", expr.to_latex())
            }
            Operation::Substitute { variable, value } => {
                format!("Substitute {} = {}", variable, value.to_latex())
            }
            Operation::MoveTerm(expr) => {
                format!("Move {} to other side", expr.to_latex())
            }
            Operation::IntegrationByParts { u, dv } => {
                format!(
                    "Integration by parts: u = {}, dv = {}",
                    u.to_latex(),
                    dv.to_latex()
                )
            }
            Operation::USubstitution { substitution } => {
                format!("U-substitution: u = {}", substitution.to_latex())
            }
            Operation::EvaluateLimit {
                variable,
                approaches,
                method,
            } => {
                format!(
                    "Evaluate limit as {} \\to {} ({})",
                    variable,
                    approaches.to_latex(),
                    method
                )
            }
            _ => self.describe(),
        }
    }

    /// Get the category of this operation for statistics.
    ///
    /// Returns a string identifier for the operation category.
    ///
    /// # Categories
    ///
    /// - `"both_sides"`: Operations applied to both sides of equation
    /// - `"transformation"`: Expression transformations (simplify, expand, etc.)
    /// - `"variable"`: Variable manipulations (substitute, isolate)
    /// - `"identity"`: Identity applications (algebraic, trig, log)
    /// - `"advanced"`: Advanced methods (quadratic formula, etc.)
    /// - `"calculus"`: Calculus operations (differentiate, integrate, etc.)
    /// - `"matrix"`: Matrix operations
    /// - `"custom"`: Custom operations
    ///
    /// # Example
    ///
    /// ```rust
    /// use mathsolver_core::resolution_path::Operation;
    /// use mathsolver_core::ast::Expression;
    ///
    /// let op = Operation::Simplify;
    /// assert_eq!(op.category(), "transformation");
    ///
    /// let op2 = Operation::QuadraticFormula;
    /// assert_eq!(op2.category(), "advanced");
    /// ```
    pub fn category(&self) -> String {
        match self {
            // Both-sides operations
            Operation::AddBothSides(_)
            | Operation::SubtractBothSides(_)
            | Operation::MultiplyBothSides(_)
            | Operation::DivideBothSides(_)
            | Operation::PowerBothSides(_)
            | Operation::RootBothSides(_)
            | Operation::ApplyFunction(_) => "both_sides".to_string(),

            // Transformations
            Operation::Simplify
            | Operation::Expand
            | Operation::Factor
            | Operation::CombineFractions
            | Operation::Cancel => "transformation".to_string(),

            // Variable operations
            Operation::Substitute { .. } | Operation::Isolate(_) | Operation::MoveTerm(_) => {
                "variable".to_string()
            }

            // Identity applications
            Operation::ApplyIdentity(_)
            | Operation::ApplyTrigIdentity(_)
            | Operation::ApplyLogProperty(_) => "identity".to_string(),

            // Advanced methods
            Operation::QuadraticFormula
            | Operation::CompleteSquare
            | Operation::NumericalApproximation => "advanced".to_string(),

            // Calculus operations
            Operation::Differentiate { .. }
            | Operation::Integrate { .. }
            | Operation::EvaluateLimit { .. }
            | Operation::IntegrationByParts { .. }
            | Operation::USubstitution { .. }
            | Operation::SolveODE { .. } => "calculus".to_string(),

            // Matrix operations
            Operation::MatrixOperation { .. }
            | Operation::GaussianElimination
            | Operation::ComputeDeterminant { .. } => "matrix".to_string(),

            // Custom
            Operation::Custom(_) => "custom".to_string(),
        }
    }

    /// Check if this operation is a "key" operation for minimal verbosity.
    ///
    /// Key operations are major transformations that significantly change
    /// the equation state. Simplification and other minor adjustments
    /// are typically not considered key operations.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mathsolver_core::resolution_path::Operation;
    /// use mathsolver_core::ast::Expression;
    ///
    /// assert!(!Operation::Simplify.is_key_operation());
    /// assert!(Operation::QuadraticFormula.is_key_operation());
    /// assert!(Operation::DivideBothSides(Expression::Integer(2)).is_key_operation());
    /// ```
    pub fn is_key_operation(&self) -> bool {
        match self {
            // Minor operations - not key
            Operation::Simplify | Operation::Cancel => false,

            // All other operations are considered key
            _ => true,
        }
    }
}

/// Builder for constructing resolution paths with a fluent API.
///
/// `ResolutionPathBuilder` provides a convenient way to construct resolution paths
/// using method chaining. It automatically tracks intermediate results and provides
/// specialized methods for common operations.
///
/// # Fluent API Pattern
///
/// The builder uses the fluent pattern where each method returns `self`, allowing
/// you to chain multiple operations together. Call [`finish()`](ResolutionPathBuilder::finish)
/// at the end to get the final [`ResolutionPath`].
///
/// # Example: Basic Usage
///
/// ```rust
/// use mathsolver_core::resolution_path::{ResolutionPathBuilder, Operation};
/// use mathsolver_core::ast::{Expression, Variable};
///
/// // Solve: 2x + 3 = 11
/// let path = ResolutionPathBuilder::new(Expression::Integer(11))
///     .step(
///         Operation::SubtractBothSides(Expression::Integer(3)),
///         "Subtract 3 from both sides".to_string(),
///         Expression::Integer(8),  // 11 - 3 = 8
///     )
///     .step(
///         Operation::DivideBothSides(Expression::Integer(2)),
///         "Divide both sides by 2".to_string(),
///         Expression::Integer(4),  // 8 / 2 = 4
///     )
///     .finish(Expression::Integer(4));
///
/// assert_eq!(path.step_count(), 2);
/// ```
///
/// # Example: Using Specialized Methods
///
/// ```rust
/// use mathsolver_core::resolution_path::ResolutionPathBuilder;
/// use mathsolver_core::ast::{Expression, Variable};
///
/// let x = Variable::new("x");
///
/// let path = ResolutionPathBuilder::new(Expression::Integer(20))
///     .simplify("Combine like terms".to_string(), Expression::Integer(20))
///     .isolate(x, "Move x to left side".to_string(), Expression::Integer(10))
///     .finish(Expression::Integer(10));
///
/// assert_eq!(path.step_count(), 2);
/// ```
///
/// # Example: Complex Multi-Step Solution
///
/// ```rust
/// use mathsolver_core::resolution_path::{ResolutionPathBuilder, Operation};
/// use mathsolver_core::ast::{Expression, Variable};
///
/// // Solve: x² = 16
/// let x = Variable::new("x");
///
/// let path = ResolutionPathBuilder::new(Expression::Integer(16))
///     .step(
///         Operation::RootBothSides(Expression::Integer(2)),
///         "Take square root of both sides".to_string(),
///         Expression::Integer(4),
///     )
///     .step(
///         Operation::Custom("Consider both positive and negative roots".to_string()),
///         "Solutions are x = 4 and x = -4".to_string(),
///         Expression::Integer(4),
///     )
///     .finish(Expression::Integer(4));
///
/// assert_eq!(path.step_count(), 2);
/// ```
pub struct ResolutionPathBuilder {
    /// The resolution path being constructed
    path: ResolutionPath,
}

impl ResolutionPathBuilder {
    /// Create a new builder starting from an initial expression.
    ///
    /// The builder begins with an empty step list. Add steps using
    /// [`step()`](ResolutionPathBuilder::step) or specialized methods like
    /// [`simplify()`](ResolutionPathBuilder::simplify) and
    /// [`isolate()`](ResolutionPathBuilder::isolate).
    ///
    /// # Arguments
    ///
    /// * `initial` - The starting expression (typically one side of an equation)
    ///
    /// # Example
    ///
    /// ```rust
    /// use mathsolver_core::resolution_path::ResolutionPathBuilder;
    /// use mathsolver_core::ast::Expression;
    ///
    /// let builder = ResolutionPathBuilder::new(Expression::Integer(42));
    /// // Add steps and finish...
    /// ```
    pub fn new(initial: Expression) -> Self {
        Self {
            path: ResolutionPath::new(initial),
        }
    }

    /// Add a step with automatic result tracking.
    ///
    /// This is the general-purpose method for adding any operation to the path.
    /// For common operations, consider using the specialized methods like
    /// [`simplify()`](ResolutionPathBuilder::simplify) or
    /// [`isolate()`](ResolutionPathBuilder::isolate).
    ///
    /// # Arguments
    ///
    /// * `operation` - The operation being performed
    /// * `explanation` - Human-readable explanation of why this step is necessary
    /// * `result` - The expression state after applying the operation
    ///
    /// # Returns
    ///
    /// Returns `self` for method chaining.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mathsolver_core::resolution_path::{ResolutionPathBuilder, Operation};
    /// use mathsolver_core::ast::Expression;
    ///
    /// let path = ResolutionPathBuilder::new(Expression::Integer(15))
    ///     .step(
    ///         Operation::DivideBothSides(Expression::Integer(3)),
    ///         "Divide by 3 to solve for x".to_string(),
    ///         Expression::Integer(5),
    ///     )
    ///     .finish(Expression::Integer(5));
    ///
    /// assert_eq!(path.step_count(), 1);
    /// ```
    pub fn step(mut self, operation: Operation, explanation: String, result: Expression) -> Self {
        self.path
            .add_step(ResolutionStep::new(operation, explanation, result));
        self
    }

    /// Add a simplification step.
    ///
    /// Convenience method for adding a [`Simplify`](Operation::Simplify) operation.
    /// Use this when combining like terms, reducing fractions, or performing
    /// other simplification operations.
    ///
    /// # Arguments
    ///
    /// * `explanation` - Description of what is being simplified
    /// * `result` - The simplified expression
    ///
    /// # Returns
    ///
    /// Returns `self` for method chaining.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mathsolver_core::resolution_path::ResolutionPathBuilder;
    /// use mathsolver_core::ast::Expression;
    ///
    /// let path = ResolutionPathBuilder::new(Expression::Integer(12))
    ///     .simplify("Combine like terms".to_string(), Expression::Integer(12))
    ///     .finish(Expression::Integer(12));
    ///
    /// assert_eq!(path.steps[0].operation, mathsolver_core::resolution_path::Operation::Simplify);
    /// ```
    pub fn simplify(self, explanation: String, result: Expression) -> Self {
        self.step(Operation::Simplify, explanation, result)
    }

    /// Add an isolation step.
    ///
    /// Convenience method for adding an [`Isolate`](Operation::Isolate) operation.
    /// Use this when isolating a variable on one side of the equation.
    ///
    /// # Arguments
    ///
    /// * `variable` - The variable being isolated
    /// * `explanation` - Description of the isolation process
    /// * `result` - The expression with the variable isolated
    ///
    /// # Returns
    ///
    /// Returns `self` for method chaining.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mathsolver_core::resolution_path::ResolutionPathBuilder;
    /// use mathsolver_core::ast::{Expression, Variable};
    ///
    /// let x = Variable::new("x");
    ///
    /// let path = ResolutionPathBuilder::new(Expression::Integer(7))
    ///     .isolate(x, "Isolate x on the left side".to_string(), Expression::Integer(7))
    ///     .finish(Expression::Integer(7));
    ///
    /// assert_eq!(path.step_count(), 1);
    /// ```
    pub fn isolate(self, variable: Variable, explanation: String, result: Expression) -> Self {
        self.step(Operation::Isolate(variable), explanation, result)
    }

    /// Finalize the path with the final result.
    ///
    /// This method consumes the builder and returns the completed [`ResolutionPath`].
    /// Call this after adding all steps to get the final path object.
    ///
    /// # Arguments
    ///
    /// * `result` - The final solution expression
    ///
    /// # Returns
    ///
    /// The completed [`ResolutionPath`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use mathsolver_core::resolution_path::{ResolutionPathBuilder, Operation};
    /// use mathsolver_core::ast::Expression;
    ///
    /// let path = ResolutionPathBuilder::new(Expression::Integer(10))
    ///     .step(
    ///         Operation::Simplify,
    ///         "Simplify".to_string(),
    ///         Expression::Integer(10),
    ///     )
    ///     .finish(Expression::Integer(10));
    ///
    /// assert_eq!(path.result, Expression::Integer(10));
    /// assert_eq!(path.step_count(), 1);
    /// ```
    pub fn finish(mut self, result: Expression) -> ResolutionPath {
        self.path.set_result(result);
        self.path
    }
}

// TODO: Add validation that steps are mathematically sound
// TODO: Add ability to replay/verify steps
// TODO: Add support for branching paths (multiple solution methods)
// TODO: Add difficulty rating for each step
// TODO: Add hints generation from resolution path
// TODO: Add support for partial paths (incomplete solutions)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Expression;

    #[test]
    fn test_verbosity_default() {
        let verbosity = Verbosity::default();
        assert_eq!(verbosity, Verbosity::Standard);
    }

    #[test]
    fn test_to_text_minimal() {
        let mut path = ResolutionPath::new(Expression::Integer(10));
        path.add_step(ResolutionStep::new(
            Operation::Simplify,
            "Simplify".to_string(),
            Expression::Integer(10),
        ));
        path.add_step(ResolutionStep::new(
            Operation::DivideBothSides(Expression::Integer(2)),
            "Divide by 2".to_string(),
            Expression::Integer(5),
        ));
        path.set_result(Expression::Integer(5));

        let text = path.to_text(Verbosity::Minimal);
        // Minimal should skip Simplify (not a key operation)
        assert!(text.contains("Start:"));
        assert!(text.contains("Result:"));
        assert!(text.contains("Divide"));
        // Should NOT contain the simplify step in minimal mode
        assert!(!text.contains("Simplify"));
    }

    #[test]
    fn test_to_text_standard() {
        let mut path = ResolutionPath::new(Expression::Integer(10));
        path.add_step(ResolutionStep::new(
            Operation::SubtractBothSides(Expression::Integer(3)),
            "Subtract 3".to_string(),
            Expression::Integer(7),
        ));
        path.set_result(Expression::Integer(7));

        let text = path.to_text(Verbosity::Standard);
        assert!(text.contains("Initial:"));
        assert!(text.contains("Step 1:"));
        assert!(text.contains("Subtract"));
        assert!(text.contains("Final result:"));
    }

    #[test]
    fn test_to_text_detailed() {
        let mut path = ResolutionPath::new(Expression::Integer(10));
        path.add_step(ResolutionStep::new(
            Operation::AddBothSides(Expression::Integer(5)),
            "Add 5 to isolate the term".to_string(),
            Expression::Integer(15),
        ));
        path.set_result(Expression::Integer(15));

        let text = path.to_text(Verbosity::Detailed);
        assert!(text.contains("=== Solution Path ==="));
        assert!(text.contains("Starting expression:"));
        assert!(text.contains("--- Step 1 ---"));
        assert!(text.contains("Operation:"));
        assert!(text.contains("Explanation:"));
        assert!(text.contains("Add 5 to isolate the term"));
        assert!(text.contains("=== Final Result ==="));
    }

    #[test]
    fn test_to_latex_standard() {
        let mut path = ResolutionPath::new(Expression::Integer(10));
        path.add_step(ResolutionStep::new(
            Operation::DivideBothSides(Expression::Integer(2)),
            "Divide by 2".to_string(),
            Expression::Integer(5),
        ));
        path.set_result(Expression::Integer(5));

        let latex = path.to_latex(Verbosity::Standard);
        assert!(latex.contains("\\begin{align*}"));
        assert!(latex.contains("\\end{align*}"));
        assert!(latex.contains("\\text{"));
    }

    #[test]
    fn test_to_json() {
        let mut path = ResolutionPath::new(Expression::Integer(20));
        path.add_step(ResolutionStep::new(
            Operation::Simplify,
            "Combine like terms".to_string(),
            Expression::Integer(20),
        ));
        path.add_step(ResolutionStep::new(
            Operation::QuadraticFormula,
            "Apply quadratic formula".to_string(),
            Expression::Integer(5),
        ));
        path.set_result(Expression::Integer(5));

        let json = path.to_json();
        assert!(json["steps"].is_array());
        assert_eq!(json["step_count"], 2);
        assert_eq!(json["statistics"]["total_steps"], 2);
        assert_eq!(json["statistics"]["uses_advanced_methods"], true);
        assert_eq!(json["statistics"]["operation_counts"]["transformations"], 1);
        assert_eq!(json["statistics"]["operation_counts"]["advanced"], 1);
    }

    #[test]
    fn test_statistics() {
        let mut path = ResolutionPath::new(Expression::Integer(100));

        // Add various operations
        path.add_step(ResolutionStep::new(
            Operation::AddBothSides(Expression::Integer(5)),
            "Add".to_string(),
            Expression::Integer(105),
        ));
        path.add_step(ResolutionStep::new(
            Operation::Simplify,
            "Simplify".to_string(),
            Expression::Integer(105),
        ));
        path.add_step(ResolutionStep::new(
            Operation::Factor,
            "Factor".to_string(),
            Expression::Integer(105),
        ));
        path.add_step(ResolutionStep::new(
            Operation::Isolate(Variable::new("x")),
            "Isolate x".to_string(),
            Expression::Integer(21),
        ));
        path.add_step(ResolutionStep::new(
            Operation::QuadraticFormula,
            "Apply quadratic formula".to_string(),
            Expression::Integer(7),
        ));

        let stats = path.statistics();

        assert_eq!(stats.total_steps, 5);
        assert_eq!(stats.operation_counts.both_sides, 1);
        assert_eq!(stats.operation_counts.transformations, 2); // Simplify + Factor
        assert_eq!(stats.operation_counts.variable_ops, 1);
        assert_eq!(stats.operation_counts.advanced, 1);
        assert!(stats.uses_advanced_methods);
        assert!(!stats.uses_calculus);
        assert!(!stats.uses_matrix_operations);
    }

    #[test]
    fn test_statistics_calculus() {
        let mut path = ResolutionPath::new(Expression::Integer(0));
        path.add_step(ResolutionStep::new(
            Operation::Differentiate {
                variable: Variable::new("x"),
                rule: "power rule".to_string(),
            },
            "Differentiate".to_string(),
            Expression::Integer(0),
        ));
        path.add_step(ResolutionStep::new(
            Operation::Integrate {
                variable: Variable::new("x"),
                technique: "substitution".to_string(),
            },
            "Integrate".to_string(),
            Expression::Integer(0),
        ));

        let stats = path.statistics();
        assert!(stats.uses_calculus);
        assert_eq!(stats.operation_counts.calculus, 2);
    }

    #[test]
    fn test_statistics_matrix() {
        let mut path = ResolutionPath::new(Expression::Integer(0));
        path.add_step(ResolutionStep::new(
            Operation::GaussianElimination,
            "Apply Gaussian elimination".to_string(),
            Expression::Integer(0),
        ));
        path.add_step(ResolutionStep::new(
            Operation::ComputeDeterminant {
                method: "cofactor expansion".to_string(),
            },
            "Compute determinant".to_string(),
            Expression::Integer(0),
        ));

        let stats = path.statistics();
        assert!(stats.uses_matrix_operations);
        assert_eq!(stats.operation_counts.matrix, 2);
    }

    #[test]
    fn test_operation_category() {
        assert_eq!(
            Operation::AddBothSides(Expression::Integer(1)).category(),
            "both_sides"
        );
        assert_eq!(Operation::Simplify.category(), "transformation");
        assert_eq!(
            Operation::Isolate(Variable::new("x")).category(),
            "variable"
        );
        assert_eq!(
            Operation::ApplyIdentity("difference of squares".to_string()).category(),
            "identity"
        );
        assert_eq!(Operation::QuadraticFormula.category(), "advanced");
        assert_eq!(
            Operation::Differentiate {
                variable: Variable::new("x"),
                rule: "power".to_string()
            }
            .category(),
            "calculus"
        );
        assert_eq!(Operation::GaussianElimination.category(), "matrix");
        assert_eq!(
            Operation::Custom("custom op".to_string()).category(),
            "custom"
        );
    }

    #[test]
    fn test_is_key_operation() {
        assert!(!Operation::Simplify.is_key_operation());
        assert!(!Operation::Cancel.is_key_operation());
        assert!(Operation::QuadraticFormula.is_key_operation());
        assert!(Operation::DivideBothSides(Expression::Integer(2)).is_key_operation());
        assert!(Operation::Factor.is_key_operation());
        assert!(Operation::GaussianElimination.is_key_operation());
    }

    #[test]
    fn test_describe_latex() {
        let op = Operation::AddBothSides(Expression::Integer(5));
        let latex = op.describe_latex();
        assert!(latex.contains("Add"));
        assert!(latex.contains("5"));
        assert!(latex.contains("both sides"));

        let op2 = Operation::QuadraticFormula;
        assert_eq!(op2.describe_latex(), "Apply quadratic formula");
    }

    #[test]
    fn test_operation_counts_default() {
        let counts = OperationCounts::default();
        assert_eq!(counts.both_sides, 0);
        assert_eq!(counts.transformations, 0);
        assert_eq!(counts.variable_ops, 0);
        assert_eq!(counts.identities, 0);
        assert_eq!(counts.advanced, 0);
        assert_eq!(counts.calculus, 0);
        assert_eq!(counts.matrix, 0);
        assert_eq!(counts.custom, 0);
    }

    #[test]
    fn test_empty_path_statistics() {
        let path = ResolutionPath::new(Expression::Integer(42));
        let stats = path.statistics();

        assert_eq!(stats.total_steps, 0);
        assert_eq!(stats.unique_operations, 0);
        assert!(!stats.uses_advanced_methods);
        assert!(!stats.uses_calculus);
        assert!(!stats.uses_matrix_operations);
    }

    #[test]
    fn test_unique_operations_count() {
        let mut path = ResolutionPath::new(Expression::Integer(10));

        // Add same operation twice
        path.add_step(ResolutionStep::new(
            Operation::Simplify,
            "First simplify".to_string(),
            Expression::Integer(10),
        ));
        path.add_step(ResolutionStep::new(
            Operation::Simplify,
            "Second simplify".to_string(),
            Expression::Integer(10),
        ));
        // Add different operation
        path.add_step(ResolutionStep::new(
            Operation::Factor,
            "Factor".to_string(),
            Expression::Integer(10),
        ));

        let stats = path.statistics();
        assert_eq!(stats.total_steps, 3);
        // Unique operations should be 2 (Simplify and Factor)
        assert_eq!(stats.unique_operations, 2);
    }

    #[test]
    fn test_escape_latex_text() {
        let text = "Use $x_1$ and {braces} with 50% & 100#";
        let escaped = escape_latex_text(text);
        assert!(escaped.contains("\\$"));
        assert!(escaped.contains("\\_"));
        assert!(escaped.contains("\\{"));
        assert!(escaped.contains("\\}"));
        assert!(escaped.contains("\\%"));
        assert!(escaped.contains("\\&"));
        assert!(escaped.contains("\\#"));
    }
}
