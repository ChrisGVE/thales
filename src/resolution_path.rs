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
            Operation::Custom(desc) => desc.clone(),
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
        self.path.add_step(ResolutionStep::new(operation, explanation, result));
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
// TODO: Add LaTeX rendering of steps
// TODO: Add support for branching paths (multiple solution methods)
// TODO: Add difficulty rating for each step
// TODO: Add hints generation from resolution path
// TODO: Add support for partial paths (incomplete solutions)
