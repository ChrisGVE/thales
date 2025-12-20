//! Abstract Syntax Tree definitions for mathematical expressions.
//!
//! This module provides the core data structures for representing mathematical equations,
//! expressions, variables, operators, and functions in a tree structure suitable for
//! parsing, manipulation, symbolic differentiation, simplification, and numerical evaluation.
//!
//! # Overview
//!
//! The AST is built around the [`Expression`] enum, which can represent:
//! - Numeric literals (integers, rationals, floats, complex numbers)
//! - Variables with optional dimension information
//! - Unary operations (negation, absolute value, logical NOT)
//! - Binary operations (addition, subtraction, multiplication, division, modulo)
//! - Mathematical functions (trigonometric, exponential, logarithmic, etc.)
//! - Power operations (exponentiation)
//!
//! Expressions can be manipulated through methods like [`Expression::simplify`],
//! [`Expression::differentiate`], and [`Expression::evaluate`].
//!
//! # Examples
//!
//! ```
//! use mathsolver_core::ast::{Expression, Variable, BinaryOp};
//! use std::collections::HashMap;
//!
//! // Create expression: x + 2
//! let x = Expression::Variable(Variable::new("x"));
//! let two = Expression::Integer(2);
//! let expr = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(two));
//!
//! // Evaluate with x = 5
//! let mut vars = HashMap::new();
//! vars.insert("x".to_string(), 5.0);
//! assert_eq!(expr.evaluate(&vars), Some(7.0));
//! ```
//!
//! # See Also
//!
//! - [`Equation`] - Represents complete equations with left and right sides
//! - [`Variable`] - Variable identifiers with optional dimension metadata
//! - [`UnaryOp`] - Unary operators
//! - [`BinaryOp`] - Binary operators
//! - [`Function`] - Mathematical functions

use num_complex::Complex64;
use num_rational::Rational64;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;

/// Represents a complete equation with left and right expressions.
///
/// An equation has the form `left = right`, where both sides are
/// arbitrary expressions. Equations are identified by a unique ID
/// for tracking and reference purposes.
///
/// # Structure
///
/// An `Equation` consists of three components:
/// - **id**: Unique string identifier for the equation (e.g., "pythagorean", "ohms_law")
/// - **left**: Left-hand side [`Expression`]
/// - **right**: Right-hand side [`Expression`]
///
/// Both sides can be arbitrary mathematical expressions including variables,
/// constants, operators, and functions.
///
/// # Examples
///
/// ## Linear equations
///
/// ```
/// use mathsolver_core::ast::{Equation, Expression, Variable, BinaryOp};
///
/// // Create equation: x + 2 = 5
/// let left = Expression::Binary(
///     BinaryOp::Add,
///     Box::new(Expression::Variable(Variable::new("x"))),
///     Box::new(Expression::Integer(2))
/// );
/// let right = Expression::Integer(5);
/// let eq = Equation::new("linear", left, right);
///
/// assert_eq!(eq.id, "linear");
/// ```
///
/// ## Quadratic equations
///
/// ```
/// use mathsolver_core::ast::{Equation, Expression, Variable, BinaryOp};
///
/// // Create equation: x² - 5x + 6 = 0
/// let x = Expression::Variable(Variable::new("x"));
///
/// // x²
/// let x_squared = Expression::Power(
///     Box::new(x.clone()),
///     Box::new(Expression::Integer(2))
/// );
///
/// // 5x
/// let five_x = Expression::Binary(
///     BinaryOp::Mul,
///     Box::new(Expression::Integer(5)),
///     Box::new(x.clone())
/// );
///
/// // x² - 5x
/// let term1 = Expression::Binary(
///     BinaryOp::Sub,
///     Box::new(x_squared),
///     Box::new(five_x)
/// );
///
/// // x² - 5x + 6
/// let left = Expression::Binary(
///     BinaryOp::Add,
///     Box::new(term1),
///     Box::new(Expression::Integer(6))
/// );
///
/// let eq = Equation::new("quadratic", left, Expression::Integer(0));
/// assert_eq!(eq.id, "quadratic");
/// ```
///
/// ## Transcendental equations
///
/// ```
/// use mathsolver_core::ast::{Equation, Expression, Variable, Function, BinaryOp};
///
/// // Create equation: sin(x) = 0.5
/// let x = Expression::Variable(Variable::new("x"));
/// let sin_x = Expression::Function(Function::Sin, vec![x]);
/// let half = Expression::Float(0.5);
///
/// let eq = Equation::new("transcendental", sin_x, half);
/// assert_eq!(eq.id, "transcendental");
/// ```
///
/// ## Physics equations
///
/// ```
/// use mathsolver_core::ast::{Equation, Expression, Variable, BinaryOp};
///
/// // Ohm's Law: V = I × R
/// let v = Expression::Variable(Variable::with_dimension("V", "volts"));
/// let i = Expression::Variable(Variable::with_dimension("I", "amperes"));
/// let r = Expression::Variable(Variable::with_dimension("R", "ohms"));
///
/// let i_times_r = Expression::Binary(
///     BinaryOp::Mul,
///     Box::new(i),
///     Box::new(r)
/// );
///
/// let ohms_law = Equation::new("ohms_law", v, i_times_r);
/// assert_eq!(ohms_law.id, "ohms_law");
/// ```
///
/// # Variable Extraction
///
/// Extract all variables from both sides of an equation:
///
/// ```
/// use mathsolver_core::ast::{Equation, Expression, Variable, BinaryOp};
/// use std::collections::HashSet;
///
/// // Create equation: a + b = c
/// let a = Expression::Variable(Variable::new("a"));
/// let b = Expression::Variable(Variable::new("b"));
/// let c = Expression::Variable(Variable::new("c"));
///
/// let left = Expression::Binary(BinaryOp::Add, Box::new(a), Box::new(b));
/// let eq = Equation::new("sum", left, c);
///
/// // Extract variables from left side
/// let left_vars = eq.left.variables();
/// assert!(left_vars.contains("a"));
/// assert!(left_vars.contains("b"));
///
/// // Extract variables from right side
/// let right_vars = eq.right.variables();
/// assert!(right_vars.contains("c"));
///
/// // Extract all variables from both sides
/// let mut all_vars = HashSet::new();
/// all_vars.extend(eq.left.variables());
/// all_vars.extend(eq.right.variables());
/// assert_eq!(all_vars.len(), 3);
/// assert!(all_vars.contains("a"));
/// assert!(all_vars.contains("b"));
/// assert!(all_vars.contains("c"));
/// ```
///
/// # Mathematical Notation
///
/// Equations can represent various mathematical forms:
///
/// - Linear: `ax + b = c`
/// - Quadratic: `ax² + bx + c = 0`
/// - Polynomial: `aₙxⁿ + ... + a₁x + a₀ = 0`
/// - Exponential: `a·eᵇˣ = c`
/// - Logarithmic: `a·ln(x) + b = c`
/// - Trigonometric: `a·sin(bx + c) = d`
/// - Rational: `p(x)/q(x) = r(x)`
/// - Implicit: `f(x,y) = g(x,y)`
///
/// # Integration with Solver
///
/// Equations are typically passed to solver modules for finding solutions:
///
/// ```ignore
/// // This example shows the typical workflow (solver module not yet implemented)
/// use mathsolver_core::ast::{Equation, Expression, Variable, BinaryOp};
/// // use mathsolver_core::solver::Solver; // Future solver module
///
/// // Create equation: 2x + 3 = 7
/// let x = Expression::Variable(Variable::new("x"));
/// let two_x = Expression::Binary(
///     BinaryOp::Mul,
///     Box::new(Expression::Integer(2)),
///     Box::new(x)
/// );
/// let left = Expression::Binary(
///     BinaryOp::Add,
///     Box::new(two_x),
///     Box::new(Expression::Integer(3))
/// );
/// let eq = Equation::new("example", left, Expression::Integer(7));
///
/// // Solve for x (future API)
/// // let solutions = Solver::solve(&eq, "x")?;
/// // assert_eq!(solutions[0], 2.0); // x = 2
/// ```
///
/// # See Also
///
/// - [`Expression`] - The expression type used for left and right sides
/// - [`Expression::variables`] - Extract variables from expressions
/// - [`Expression::evaluate`] - Evaluate expressions with variable values
/// - [`Expression::simplify`] - Simplify expressions algebraically
/// - Future: `solver` module for solving equations
#[derive(Debug, Clone, PartialEq)]
pub struct Equation {
    /// Unique identifier for this equation.
    ///
    /// Used for tracking equations in systems, referencing in error messages,
    /// and organizing equation collections. Examples: "eq1", "pythagorean",
    /// "ohms_law", "conservation_energy".
    pub id: String,

    /// Left-hand side expression.
    ///
    /// Can be any valid mathematical expression including variables, constants,
    /// operators, and functions. Represents the expression on the left side
    /// of the equals sign.
    pub left: Expression,

    /// Right-hand side expression.
    ///
    /// Can be any valid mathematical expression including variables, constants,
    /// operators, and functions. Represents the expression on the right side
    /// of the equals sign.
    pub right: Expression,
}

impl Equation {
    /// Create a new equation from two expressions.
    ///
    /// Constructs an equation of the form `left = right` with a unique identifier.
    /// The `id` parameter accepts any type that implements `Into<String>`, allowing
    /// both string literals and owned strings.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the equation (accepts `&str` or `String`)
    /// * `left` - Left-hand side expression
    /// * `right` - Right-hand side expression
    ///
    /// # Examples
    ///
    /// ## Simple constant equation
    ///
    /// ```
    /// use mathsolver_core::ast::{Equation, Expression};
    ///
    /// // Create equation: 3 = 3
    /// let eq = Equation::new(
    ///     "identity",
    ///     Expression::Integer(3),
    ///     Expression::Integer(3)
    /// );
    /// assert_eq!(eq.id, "identity");
    /// ```
    ///
    /// ## Linear equation with variable
    ///
    /// ```
    /// use mathsolver_core::ast::{Equation, Expression, Variable, BinaryOp};
    ///
    /// // Create equation: 2x = 10
    /// let x = Expression::Variable(Variable::new("x"));
    /// let two_x = Expression::Binary(
    ///     BinaryOp::Mul,
    ///     Box::new(Expression::Integer(2)),
    ///     Box::new(x)
    /// );
    ///
    /// let eq = Equation::new("linear", two_x, Expression::Integer(10));
    /// assert_eq!(eq.id, "linear");
    /// assert!(eq.left.contains_variable("x"));
    /// ```
    ///
    /// ## Pythagorean theorem
    ///
    /// ```
    /// use mathsolver_core::ast::{Equation, Expression, Variable, BinaryOp};
    ///
    /// // Create equation: a² + b² = c²
    /// let a = Expression::Variable(Variable::new("a"));
    /// let b = Expression::Variable(Variable::new("b"));
    /// let c = Expression::Variable(Variable::new("c"));
    ///
    /// let a_squared = Expression::Power(
    ///     Box::new(a),
    ///     Box::new(Expression::Integer(2))
    /// );
    /// let b_squared = Expression::Power(
    ///     Box::new(b),
    ///     Box::new(Expression::Integer(2))
    /// );
    /// let c_squared = Expression::Power(
    ///     Box::new(c),
    ///     Box::new(Expression::Integer(2))
    /// );
    ///
    /// let left = Expression::Binary(
    ///     BinaryOp::Add,
    ///     Box::new(a_squared),
    ///     Box::new(b_squared)
    /// );
    ///
    /// let pythagorean = Equation::new("pythagorean", left, c_squared);
    /// assert_eq!(pythagorean.id, "pythagorean");
    /// ```
    ///
    /// ## Using owned String for ID
    ///
    /// ```
    /// use mathsolver_core::ast::{Equation, Expression};
    ///
    /// let equation_name = format!("eq_{}", 42);
    /// let eq = Equation::new(
    ///     equation_name,
    ///     Expression::Integer(1),
    ///     Expression::Integer(1)
    /// );
    /// assert_eq!(eq.id, "eq_42");
    /// ```
    ///
    /// ## Creating equation from parser output
    ///
    /// ```
    /// use mathsolver_core::ast::{Equation, Expression, Variable, BinaryOp, Function};
    ///
    /// // Typical use case: wrapping parsed expressions into an equation
    /// // Example: exp(x) = 2.718
    /// let x = Expression::Variable(Variable::new("x"));
    /// let exp_x = Expression::Function(Function::Exp, vec![x]);
    /// let e = Expression::Float(2.718281828);
    ///
    /// let eq = Equation::new("exponential", exp_x, e);
    /// assert_eq!(eq.id, "exponential");
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Expression`] - For building equation expressions
    /// - [`Variable::new`] - Creating variables for equations
    /// - [`BinaryOp`] - Binary operators for building expressions
    pub fn new(id: impl Into<String>, left: Expression, right: Expression) -> Self {
        Self {
            id: id.into(),
            left,
            right,
        }
    }
}

/// Represents a mathematical expression in tree form.
///
/// An `Expression` is a recursive data structure that can represent any mathematical
/// expression from simple constants to complex formulas involving variables, operators,
/// and functions. Expressions support simplification, differentiation, and numerical evaluation.
///
/// # Variants
///
/// - [`Integer`](Expression::Integer) - Whole number constants
/// - [`Rational`](Expression::Rational) - Exact fractions (p/q where p, q are integers)
/// - [`Float`](Expression::Float) - Floating-point numbers
/// - [`Complex`](Expression::Complex) - Complex numbers (a + bi)
/// - [`Variable`](Expression::Variable) - Named variables (e.g., x, y, velocity)
/// - [`Unary`](Expression::Unary) - Single-argument operations (e.g., -x, |x|)
/// - [`Binary`](Expression::Binary) - Two-argument operations (e.g., x + y, x * y)
/// - [`Function`](Expression::Function) - Mathematical functions (e.g., sin(x), log(x))
/// - [`Power`](Expression::Power) - Exponentiation (base^exponent)
///
/// # Examples
///
/// ## Creating expressions programmatically
///
/// ```
/// use mathsolver_core::ast::{Expression, Variable, BinaryOp, UnaryOp, Function};
///
/// // Simple constant: 42
/// let constant = Expression::Integer(42);
///
/// // Variable: x
/// let x = Expression::Variable(Variable::new("x"));
///
/// // Negation: -x
/// let neg_x = Expression::Unary(UnaryOp::Neg, Box::new(x.clone()));
///
/// // Binary operation: x + 5
/// let x_plus_5 = Expression::Binary(
///     BinaryOp::Add,
///     Box::new(x.clone()),
///     Box::new(Expression::Integer(5))
/// );
///
/// // Function call: sin(x)
/// let sin_x = Expression::Function(Function::Sin, vec![x.clone()]);
///
/// // Power: x^2
/// let x_squared = Expression::Power(
///     Box::new(x.clone()),
///     Box::new(Expression::Integer(2))
/// );
/// ```
///
/// ## Simplification
///
/// ```
/// use mathsolver_core::ast::{Expression, BinaryOp};
///
/// // Create: 0 + 5
/// let expr = Expression::Binary(
///     BinaryOp::Add,
///     Box::new(Expression::Integer(0)),
///     Box::new(Expression::Integer(5))
/// );
///
/// // Simplify to: 5
/// let simplified = expr.simplify();
/// assert_eq!(simplified, Expression::Integer(5));
/// ```
///
/// ## Evaluation
///
/// ```
/// use mathsolver_core::ast::{Expression, Variable, BinaryOp};
/// use std::collections::HashMap;
///
/// // Create: x * 2 + 3
/// let x = Expression::Variable(Variable::new("x"));
/// let x_times_2 = Expression::Binary(
///     BinaryOp::Mul,
///     Box::new(x),
///     Box::new(Expression::Integer(2))
/// );
/// let expr = Expression::Binary(
///     BinaryOp::Add,
///     Box::new(x_times_2),
///     Box::new(Expression::Integer(3))
/// );
///
/// // Evaluate with x = 10
/// let mut vars = HashMap::new();
/// vars.insert("x".to_string(), 10.0);
/// assert_eq!(expr.evaluate(&vars), Some(23.0));
/// ```
///
/// ## Symbolic differentiation
///
/// ```
/// use mathsolver_core::ast::{Expression, Variable, Function};
///
/// // Create: sin(x)
/// let x = Expression::Variable(Variable::new("x"));
/// let sin_x = Expression::Function(Function::Sin, vec![x]);
///
/// // Differentiate: d/dx[sin(x)] = cos(x)
/// let derivative = sin_x.differentiate("x");
/// // Result is cos(x) * 1 (chain rule applied)
/// ```
///
/// # See Also
///
/// - [`Variable`] - Variable identifiers with optional dimension metadata
/// - [`UnaryOp`] - Available unary operators
/// - [`BinaryOp`] - Available binary operators
/// - [`Function`] - Available mathematical functions
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// Integer literal.
    ///
    /// Represents whole number constants (e.g., 0, 42, -17).
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::ast::Expression;
    ///
    /// let zero = Expression::Integer(0);
    /// let answer = Expression::Integer(42);
    /// let negative = Expression::Integer(-17);
    /// ```
    Integer(i64),

    /// Rational number (exact fraction).
    ///
    /// Represents fractions as numerator/denominator pairs (e.g., 1/2, 22/7).
    /// Useful for exact arithmetic without floating-point errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::ast::Expression;
    /// use num_rational::Rational64;
    ///
    /// // One half: 1/2
    /// let half = Expression::Rational(Rational64::new(1, 2));
    ///
    /// // Pi approximation: 22/7
    /// let pi_approx = Expression::Rational(Rational64::new(22, 7));
    /// ```
    Rational(Rational64),

    /// Floating-point number.
    ///
    /// Represents real numbers with decimal precision (e.g., 3.14159, 2.718).
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::ast::Expression;
    ///
    /// let pi = Expression::Float(3.14159);
    /// let e = Expression::Float(2.71828);
    /// ```
    Float(f64),

    /// Complex number.
    ///
    /// Represents complex numbers of the form a + bi where a is the real part
    /// and b is the imaginary part (e.g., 3+4i, -2i).
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::ast::Expression;
    /// use num_complex::Complex64;
    ///
    /// // 3 + 4i
    /// let z1 = Expression::Complex(Complex64::new(3.0, 4.0));
    ///
    /// // -2i (purely imaginary)
    /// let z2 = Expression::Complex(Complex64::new(0.0, -2.0));
    /// ```
    Complex(Complex64),

    /// Variable reference.
    ///
    /// Represents a named variable (e.g., x, velocity, temperature).
    /// Variables can optionally include dimension/unit information.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::ast::{Expression, Variable};
    ///
    /// // Simple variable: x
    /// let x = Expression::Variable(Variable::new("x"));
    ///
    /// // Variable with dimension: velocity [m/s]
    /// let v = Expression::Variable(Variable::with_dimension("velocity", "m/s"));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Variable`] - Variable struct with dimension support
    Variable(Variable),

    /// Unary operation (single operand).
    ///
    /// Represents operations that take one argument (e.g., -x, |x|, !x).
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::ast::{Expression, Variable, UnaryOp};
    ///
    /// let x = Expression::Variable(Variable::new("x"));
    ///
    /// // Negation: -x
    /// let neg_x = Expression::Unary(UnaryOp::Neg, Box::new(x.clone()));
    ///
    /// // Absolute value: |x|
    /// let abs_x = Expression::Unary(UnaryOp::Abs, Box::new(x));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`UnaryOp`] - Available unary operators
    Unary(UnaryOp, Box<Expression>),

    /// Binary operation (two operands).
    ///
    /// Represents operations that take two arguments (e.g., x + y, a * b).
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::ast::{Expression, Variable, BinaryOp};
    ///
    /// let x = Expression::Variable(Variable::new("x"));
    /// let y = Expression::Variable(Variable::new("y"));
    ///
    /// // Addition: x + y
    /// let sum = Expression::Binary(BinaryOp::Add, Box::new(x.clone()), Box::new(y.clone()));
    ///
    /// // Multiplication: x * y
    /// let product = Expression::Binary(BinaryOp::Mul, Box::new(x), Box::new(y));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`BinaryOp`] - Available binary operators
    Binary(BinaryOp, Box<Expression>, Box<Expression>),

    /// Function call with arguments.
    ///
    /// Represents mathematical function application (e.g., sin(x), log(x, base)).
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::ast::{Expression, Variable, Function};
    ///
    /// let x = Expression::Variable(Variable::new("x"));
    ///
    /// // Sine: sin(x)
    /// let sin_x = Expression::Function(Function::Sin, vec![x.clone()]);
    ///
    /// // Square root: sqrt(x)
    /// let sqrt_x = Expression::Function(Function::Sqrt, vec![x.clone()]);
    ///
    /// // Logarithm: log(x, 10)
    /// let log_x = Expression::Function(
    ///     Function::Log,
    ///     vec![x, Expression::Integer(10)]
    /// );
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Function`] - Available mathematical functions
    Function(Function, Vec<Expression>),

    /// Power operation (exponentiation).
    ///
    /// Represents base raised to an exponent (e.g., x^2, 2^n, e^x).
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::ast::{Expression, Variable};
    ///
    /// let x = Expression::Variable(Variable::new("x"));
    ///
    /// // Square: x^2
    /// let x_squared = Expression::Power(
    ///     Box::new(x.clone()),
    ///     Box::new(Expression::Integer(2))
    /// );
    ///
    /// // Exponential: 2^x
    /// let two_to_x = Expression::Power(
    ///     Box::new(Expression::Integer(2)),
    ///     Box::new(x)
    /// );
    /// ```
    Power(Box<Expression>, Box<Expression>),
}

/// Variable identifier with optional metadata.
///
/// Represents a named variable in mathematical expressions. Variables can optionally
/// carry dimension/unit information for dimensional analysis.
///
/// # Examples
///
/// ```
/// use mathsolver_core::ast::Variable;
///
/// // Simple variable without dimension
/// let x = Variable::new("x");
/// assert_eq!(x.name, "x");
/// assert_eq!(x.dimension, None);
///
/// // Variable with dimension (e.g., physical quantity)
/// let velocity = Variable::with_dimension("v", "m/s");
/// assert_eq!(velocity.name, "v");
/// assert_eq!(velocity.dimension, Some("m/s".to_string()));
/// ```
///
/// # See Also
///
/// - [`Expression::Variable`] - Expression variant that wraps this type
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Variable {
    /// Variable name (e.g., "x", "velocity", "temperature")
    pub name: String,
    /// Optional dimension/unit information (e.g., "m/s", "kg", "meters")
    pub dimension: Option<String>,
}

impl Variable {
    /// Create a new variable with the given name.
    ///
    /// Creates a variable without dimension information.
    ///
    /// # Arguments
    ///
    /// * `name` - Variable name
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::ast::Variable;
    ///
    /// let x = Variable::new("x");
    /// let theta = Variable::new("theta");
    /// ```
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dimension: None,
        }
    }

    /// Create a variable with dimension information.
    ///
    /// Creates a variable annotated with physical dimension or unit information.
    /// Useful for dimensional analysis and unit checking.
    ///
    /// # Arguments
    ///
    /// * `name` - Variable name
    /// * `dimension` - Dimension or unit string (e.g., "m/s", "kg", "meters")
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::ast::Variable;
    ///
    /// let velocity = Variable::with_dimension("v", "m/s");
    /// let mass = Variable::with_dimension("m", "kg");
    /// let distance = Variable::with_dimension("d", "meters");
    /// ```
    pub fn with_dimension(name: impl Into<String>, dimension: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dimension: Some(dimension.into()),
        }
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Unary operators (single operand).
///
/// Represents operations that take a single argument.
///
/// # Variants
///
/// - [`Neg`](UnaryOp::Neg) - Arithmetic negation (-)
/// - [`Not`](UnaryOp::Not) - Logical NOT (!)
/// - [`Abs`](UnaryOp::Abs) - Absolute value (|x|)
///
/// # Examples
///
/// ```
/// use mathsolver_core::ast::{Expression, Variable, UnaryOp};
///
/// let x = Expression::Variable(Variable::new("x"));
///
/// // Negation: -x
/// let neg = Expression::Unary(UnaryOp::Neg, Box::new(x.clone()));
///
/// // Absolute value: |x|
/// let abs = Expression::Unary(UnaryOp::Abs, Box::new(x));
/// ```
///
/// # See Also
///
/// - [`Expression::Unary`] - Expression variant using these operators
/// - [`BinaryOp`] - Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Arithmetic negation: -x
    ///
    /// Returns the additive inverse of the operand.
    ///
    /// # Mathematical notation
    ///
    /// -x or -(expr)
    Neg,

    /// Logical NOT: !x
    ///
    /// Logical negation. Treats 0 as false and non-zero as true.
    ///
    /// # Mathematical notation
    ///
    /// !x or ¬x
    Not,

    /// Absolute value: |x|
    ///
    /// Returns the magnitude of the operand (always non-negative).
    ///
    /// # Mathematical notation
    ///
    /// |x| or abs(x)
    Abs,
}

/// Binary operators (two operands).
///
/// Represents operations that take two arguments. All binary operators
/// are left-associative and have defined precedence levels.
///
/// # Precedence
///
/// Higher precedence operators bind tighter:
/// - Precedence 2: `*`, `/`, `%` (multiplication, division, modulo)
/// - Precedence 1: `+`, `-` (addition, subtraction)
///
/// # Examples
///
/// ```
/// use mathsolver_core::ast::{Expression, BinaryOp};
///
/// let two = Expression::Integer(2);
/// let three = Expression::Integer(3);
///
/// // Addition: 2 + 3
/// let sum = Expression::Binary(BinaryOp::Add, Box::new(two.clone()), Box::new(three.clone()));
///
/// // Multiplication: 2 * 3
/// let product = Expression::Binary(BinaryOp::Mul, Box::new(two), Box::new(three));
/// ```
///
/// # See Also
///
/// - [`Expression::Binary`] - Expression variant using these operators
/// - [`UnaryOp`] - Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// Addition: x + y
    ///
    /// Returns the sum of two operands.
    ///
    /// Precedence: 1
    Add,

    /// Subtraction: x - y
    ///
    /// Returns the difference of two operands.
    ///
    /// Precedence: 1
    Sub,

    /// Multiplication: x * y
    ///
    /// Returns the product of two operands.
    ///
    /// Precedence: 2
    Mul,

    /// Division: x / y
    ///
    /// Returns the quotient of two operands.
    ///
    /// Precedence: 2
    Div,

    /// Modulo: x % y
    ///
    /// Returns the remainder after division.
    ///
    /// Precedence: 2
    Mod,
}

impl BinaryOp {
    /// Returns the precedence level of this operator.
    ///
    /// Higher numbers bind tighter. Used for proper parenthesization
    /// when formatting expressions.
    ///
    /// # Precedence levels
    ///
    /// - 2: `*`, `/`, `%`
    /// - 1: `+`, `-`
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::ast::BinaryOp;
    ///
    /// assert_eq!(BinaryOp::Add.precedence(), 1);
    /// assert_eq!(BinaryOp::Mul.precedence(), 2);
    /// assert!(BinaryOp::Mul.precedence() > BinaryOp::Add.precedence());
    /// ```
    pub fn precedence(self) -> u8 {
        match self {
            BinaryOp::Add | BinaryOp::Sub => 1,
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => 2,
        }
    }
}

/// Mathematical functions.
///
/// Represents standard mathematical functions that can be applied to expressions.
/// Functions are organized into categories: trigonometric, hyperbolic, exponential/logarithmic,
/// power/root, rounding, and utility functions.
///
/// # Categories
///
/// ## Trigonometric
/// - [`Sin`](Function::Sin), [`Cos`](Function::Cos), [`Tan`](Function::Tan)
/// - [`Asin`](Function::Asin), [`Acos`](Function::Acos), [`Atan`](Function::Atan), [`Atan2`](Function::Atan2)
///
/// ## Hyperbolic
/// - [`Sinh`](Function::Sinh), [`Cosh`](Function::Cosh), [`Tanh`](Function::Tanh)
///
/// ## Exponential and Logarithmic
/// - [`Exp`](Function::Exp), [`Ln`](Function::Ln)
/// - [`Log`](Function::Log), [`Log2`](Function::Log2), [`Log10`](Function::Log10)
///
/// ## Power and Root
/// - [`Sqrt`](Function::Sqrt), [`Cbrt`](Function::Cbrt), [`Pow`](Function::Pow)
///
/// ## Rounding
/// - [`Floor`](Function::Floor), [`Ceil`](Function::Ceil), [`Round`](Function::Round)
///
/// ## Utility
/// - [`Abs`](Function::Abs), [`Sign`](Function::Sign)
/// - [`Min`](Function::Min), [`Max`](Function::Max)
///
/// # Examples
///
/// ```
/// use mathsolver_core::ast::{Expression, Variable, Function};
/// use std::collections::HashMap;
///
/// let x = Expression::Variable(Variable::new("x"));
///
/// // Trigonometric: sin(x)
/// let sin_x = Expression::Function(Function::Sin, vec![x.clone()]);
///
/// // Exponential: exp(x)
/// let exp_x = Expression::Function(Function::Exp, vec![x.clone()]);
///
/// // Square root: sqrt(4)
/// let sqrt_4 = Expression::Function(Function::Sqrt, vec![Expression::Integer(4)]);
/// let mut vars = HashMap::new();
/// assert_eq!(sqrt_4.evaluate(&vars), Some(2.0));
/// ```
///
/// # See Also
///
/// - [`Expression::Function`] - Expression variant using functions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Function {
    // Trigonometric functions

    /// Sine function: sin(x)
    ///
    /// Returns the sine of the argument (in radians).
    ///
    /// # Mathematical notation
    ///
    /// sin(x)
    Sin,

    /// Cosine function: cos(x)
    ///
    /// Returns the cosine of the argument (in radians).
    ///
    /// # Mathematical notation
    ///
    /// cos(x)
    Cos,

    /// Tangent function: tan(x)
    ///
    /// Returns the tangent of the argument (in radians).
    ///
    /// # Mathematical notation
    ///
    /// tan(x) = sin(x) / cos(x)
    Tan,

    /// Arcsine function: asin(x)
    ///
    /// Returns the inverse sine (in radians). Domain: [-1, 1].
    ///
    /// # Mathematical notation
    ///
    /// arcsin(x) or sin⁻¹(x)
    Asin,

    /// Arccosine function: acos(x)
    ///
    /// Returns the inverse cosine (in radians). Domain: [-1, 1].
    ///
    /// # Mathematical notation
    ///
    /// arccos(x) or cos⁻¹(x)
    Acos,

    /// Arctangent function: atan(x)
    ///
    /// Returns the inverse tangent (in radians).
    ///
    /// # Mathematical notation
    ///
    /// arctan(x) or tan⁻¹(x)
    Atan,

    /// Two-argument arctangent: atan2(y, x)
    ///
    /// Returns the angle in radians between the positive x-axis and the point (x, y).
    ///
    /// # Mathematical notation
    ///
    /// atan2(y, x)
    Atan2,

    // Hyperbolic functions

    /// Hyperbolic sine: sinh(x)
    ///
    /// # Mathematical notation
    ///
    /// sinh(x) = (eˣ - e⁻ˣ) / 2
    Sinh,

    /// Hyperbolic cosine: cosh(x)
    ///
    /// # Mathematical notation
    ///
    /// cosh(x) = (eˣ + e⁻ˣ) / 2
    Cosh,

    /// Hyperbolic tangent: tanh(x)
    ///
    /// # Mathematical notation
    ///
    /// tanh(x) = sinh(x) / cosh(x)
    Tanh,

    // Exponential and logarithmic functions

    /// Exponential function: exp(x)
    ///
    /// Returns e raised to the power x.
    ///
    /// # Mathematical notation
    ///
    /// exp(x) = eˣ
    Exp,

    /// Natural logarithm: ln(x)
    ///
    /// Returns the natural logarithm (base e).
    ///
    /// # Mathematical notation
    ///
    /// ln(x) or logₑ(x)
    Ln,

    /// Logarithm with arbitrary base: log(x, base)
    ///
    /// Returns the logarithm of x to the given base.
    ///
    /// # Mathematical notation
    ///
    /// log_base(x)
    Log,

    /// Binary logarithm: log2(x)
    ///
    /// Returns the base-2 logarithm.
    ///
    /// # Mathematical notation
    ///
    /// log₂(x)
    Log2,

    /// Common logarithm: log10(x)
    ///
    /// Returns the base-10 logarithm.
    ///
    /// # Mathematical notation
    ///
    /// log₁₀(x) or log(x)
    Log10,

    // Power and root functions

    /// Square root: sqrt(x)
    ///
    /// Returns the principal square root.
    ///
    /// # Mathematical notation
    ///
    /// √x or x^(1/2)
    Sqrt,

    /// Cube root: cbrt(x)
    ///
    /// Returns the cube root.
    ///
    /// # Mathematical notation
    ///
    /// ∛x or x^(1/3)
    Cbrt,

    /// Power function: pow(x, y)
    ///
    /// Returns x raised to the power y.
    ///
    /// # Mathematical notation
    ///
    /// x^y
    Pow,

    // Rounding functions

    /// Floor function: floor(x)
    ///
    /// Returns the largest integer less than or equal to x.
    ///
    /// # Mathematical notation
    ///
    /// ⌊x⌋
    Floor,

    /// Ceiling function: ceil(x)
    ///
    /// Returns the smallest integer greater than or equal to x.
    ///
    /// # Mathematical notation
    ///
    /// ⌈x⌉
    Ceil,

    /// Round function: round(x)
    ///
    /// Returns the nearest integer, rounding half-way cases away from zero.
    ///
    /// # Mathematical notation
    ///
    /// round(x)
    Round,

    // Utility functions

    /// Absolute value: abs(x)
    ///
    /// Returns the magnitude (always non-negative).
    ///
    /// # Mathematical notation
    ///
    /// |x|
    Abs,

    /// Sign function: sign(x)
    ///
    /// Returns -1 for negative, 0 for zero, +1 for positive.
    ///
    /// # Mathematical notation
    ///
    /// sgn(x)
    Sign,

    /// Minimum: min(x₁, x₂, ..., xₙ)
    ///
    /// Returns the smallest value among arguments.
    ///
    /// # Mathematical notation
    ///
    /// min(x₁, x₂, ..., xₙ)
    Min,

    /// Maximum: max(x₁, x₂, ..., xₙ)
    ///
    /// Returns the largest value among arguments.
    ///
    /// # Mathematical notation
    ///
    /// max(x₁, x₂, ..., xₙ)
    Max,

    /// User-defined custom function.
    ///
    /// Represents a function not built into the standard set.
    /// Evaluation of custom functions returns `None` unless
    /// a custom evaluator is provided.
    Custom(String),
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Function::Sin => write!(f, "sin"),
            Function::Cos => write!(f, "cos"),
            Function::Tan => write!(f, "tan"),
            Function::Asin => write!(f, "asin"),
            Function::Acos => write!(f, "acos"),
            Function::Atan => write!(f, "atan"),
            Function::Atan2 => write!(f, "atan2"),
            Function::Sinh => write!(f, "sinh"),
            Function::Cosh => write!(f, "cosh"),
            Function::Tanh => write!(f, "tanh"),
            Function::Exp => write!(f, "exp"),
            Function::Ln => write!(f, "ln"),
            Function::Log => write!(f, "log"),
            Function::Log2 => write!(f, "log2"),
            Function::Log10 => write!(f, "log10"),
            Function::Sqrt => write!(f, "sqrt"),
            Function::Cbrt => write!(f, "cbrt"),
            Function::Pow => write!(f, "pow"),
            Function::Floor => write!(f, "floor"),
            Function::Ceil => write!(f, "ceil"),
            Function::Round => write!(f, "round"),
            Function::Abs => write!(f, "abs"),
            Function::Sign => write!(f, "sign"),
            Function::Min => write!(f, "min"),
            Function::Max => write!(f, "max"),
            Function::Custom(name) => write!(f, "{}", name),
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with_precedence(f, 0)
    }
}

impl Expression {
    /// Format the expression with precedence awareness.
    fn fmt_with_precedence(&self, f: &mut fmt::Formatter<'_>, parent_prec: u8) -> fmt::Result {
        match self {
            Expression::Integer(n) => write!(f, "{}", n),
            Expression::Rational(r) => write!(f, "{}/{}", r.numer(), r.denom()),
            Expression::Float(x) => write!(f, "{}", x),
            Expression::Complex(c) => {
                if c.im >= 0.0 {
                    write!(f, "{}+{}i", c.re, c.im)
                } else {
                    write!(f, "{}{}i", c.re, c.im)
                }
            }
            Expression::Variable(v) => write!(f, "{}", v),
            Expression::Unary(UnaryOp::Neg, expr) => {
                write!(f, "-")?;
                expr.fmt_with_precedence(f, 3)
            }
            Expression::Unary(UnaryOp::Not, expr) => {
                write!(f, "!")?;
                expr.fmt_with_precedence(f, 3)
            }
            Expression::Unary(UnaryOp::Abs, expr) => {
                write!(f, "|")?;
                expr.fmt_with_precedence(f, 0)?;
                write!(f, "|")
            }
            Expression::Binary(op, left, right) => {
                let prec = op.precedence();
                let needs_parens = prec < parent_prec;

                if needs_parens {
                    write!(f, "(")?;
                }

                left.fmt_with_precedence(f, prec)?;

                match op {
                    BinaryOp::Add => write!(f, " + ")?,
                    BinaryOp::Sub => write!(f, " - ")?,
                    BinaryOp::Mul => write!(f, " * ")?,
                    BinaryOp::Div => write!(f, " / ")?,
                    BinaryOp::Mod => write!(f, " % ")?,
                }

                // Right side needs higher precedence for left-associativity
                right.fmt_with_precedence(f, prec + 1)?;

                if needs_parens {
                    write!(f, ")")?;
                }

                Ok(())
            }
            Expression::Function(func, args) => {
                write!(f, "{}", func)?;
                write!(f, "(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    arg.fmt_with_precedence(f, 0)?;
                }
                write!(f, ")")
            }
            Expression::Power(base, exp) => {
                let needs_parens = parent_prec > 3;

                if needs_parens {
                    write!(f, "(")?;
                }

                base.fmt_with_precedence(f, 4)?;
                write!(f, "^")?;
                exp.fmt_with_precedence(f, 4)?;

                if needs_parens {
                    write!(f, ")")?;
                }

                Ok(())
            }
        }
    }

    /// Returns a HashSet of all variable names in the expression.
    ///
    /// Recursively traverses the expression tree to collect all unique variable names.
    /// Variables appearing multiple times are only included once.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::ast::{Expression, Variable, BinaryOp};
    ///
    /// // x + y * x
    /// let x = Expression::Variable(Variable::new("x"));
    /// let y = Expression::Variable(Variable::new("y"));
    /// let y_times_x = Expression::Binary(BinaryOp::Mul, Box::new(y), Box::new(x.clone()));
    /// let expr = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(y_times_x));
    ///
    /// let vars = expr.variables();
    /// assert_eq!(vars.len(), 2);
    /// assert!(vars.contains("x"));
    /// assert!(vars.contains("y"));
    /// ```
    ///
    /// # Returns
    ///
    /// A `HashSet<String>` containing all unique variable names.
    ///
    /// # See Also
    ///
    /// - [`contains_variable`](Expression::contains_variable) - Check if a specific variable is present
    pub fn variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_variables(&mut vars);
        vars
    }

    /// Helper function to recursively collect variables.
    fn collect_variables(&self, vars: &mut HashSet<String>) {
        match self {
            Expression::Variable(v) => {
                vars.insert(v.name.clone());
            }
            Expression::Unary(_, expr) => {
                expr.collect_variables(vars);
            }
            Expression::Binary(_, left, right) => {
                left.collect_variables(vars);
                right.collect_variables(vars);
            }
            Expression::Function(_, args) => {
                for arg in args {
                    arg.collect_variables(vars);
                }
            }
            Expression::Power(base, exp) => {
                base.collect_variables(vars);
                exp.collect_variables(vars);
            }
            _ => {}
        }
    }

    /// Returns true if expression contains the named variable.
    ///
    /// Recursively searches the expression tree for a variable with the given name.
    ///
    /// # Arguments
    ///
    /// * `name` - The variable name to search for
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::ast::{Expression, Variable, BinaryOp};
    ///
    /// // x + 5
    /// let x = Expression::Variable(Variable::new("x"));
    /// let expr = Expression::Binary(
    ///     BinaryOp::Add,
    ///     Box::new(x),
    ///     Box::new(Expression::Integer(5))
    /// );
    ///
    /// assert!(expr.contains_variable("x"));
    /// assert!(!expr.contains_variable("y"));
    /// ```
    ///
    /// # Returns
    ///
    /// `true` if the variable is found, `false` otherwise.
    ///
    /// # See Also
    ///
    /// - [`variables`](Expression::variables) - Get all variables in the expression
    pub fn contains_variable(&self, name: &str) -> bool {
        match self {
            Expression::Variable(v) => v.name == name,
            Expression::Unary(_, expr) => expr.contains_variable(name),
            Expression::Binary(_, left, right) => {
                left.contains_variable(name) || right.contains_variable(name)
            }
            Expression::Function(_, args) => args.iter().any(|arg| arg.contains_variable(name)),
            Expression::Power(base, exp) => {
                base.contains_variable(name) || exp.contains_variable(name)
            }
            _ => false,
        }
    }

    /// Recursively transform the expression using a mapping function.
    ///
    /// Applies the given function to every node in the expression tree, bottom-up.
    /// The transformation is applied to child nodes first, then to the current node.
    ///
    /// # Type Parameters
    ///
    /// * `F` - Function type that maps `&Expression` to `Expression`
    ///
    /// # Arguments
    ///
    /// * `f` - Transformation function to apply to each node
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::ast::{Expression, Variable, BinaryOp};
    ///
    /// // Replace all variables named "x" with constant 10
    /// let x = Expression::Variable(Variable::new("x"));
    /// let y = Expression::Variable(Variable::new("y"));
    /// let expr = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(y));
    ///
    /// let transformed = expr.map(&|e| {
    ///     match e {
    ///         Expression::Variable(v) if v.name == "x" => Expression::Integer(10),
    ///         _ => e.clone()
    ///     }
    /// });
    ///
    /// // Result: 10 + y
    /// assert!(transformed.contains_variable("y"));
    /// assert!(!transformed.contains_variable("x"));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`fold`](Expression::fold) - Accumulate values from the expression tree
    pub fn map<F>(&self, f: &F) -> Expression
    where
        F: Fn(&Expression) -> Expression,
    {
        let mapped = match self {
            Expression::Unary(op, expr) => {
                Expression::Unary(*op, Box::new(expr.map(f)))
            }
            Expression::Binary(op, left, right) => {
                Expression::Binary(*op, Box::new(left.map(f)), Box::new(right.map(f)))
            }
            Expression::Function(func, args) => {
                Expression::Function(func.clone(), args.iter().map(|arg| arg.map(f)).collect())
            }
            Expression::Power(base, exp) => {
                Expression::Power(Box::new(base.map(f)), Box::new(exp.map(f)))
            }
            _ => self.clone(),
        };
        f(&mapped)
    }

    /// Fold/reduce the expression tree.
    ///
    /// Accumulates a value by traversing the expression tree and applying a function
    /// to each node along with the accumulated value. Traversal is depth-first.
    ///
    /// # Type Parameters
    ///
    /// * `T` - Type of the accumulated value
    /// * `F` - Function type that combines accumulated value with each node
    ///
    /// # Arguments
    ///
    /// * `init` - Initial accumulated value
    /// * `f` - Reduction function `fn(accumulator, node) -> accumulator`
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::ast::{Expression, Variable, BinaryOp};
    ///
    /// // Count all nodes in the expression tree
    /// let x = Expression::Variable(Variable::new("x"));
    /// let five = Expression::Integer(5);
    /// let expr = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(five));
    ///
    /// let node_count = expr.fold(0, &|count, _node| count + 1);
    /// assert_eq!(node_count, 3); // Binary node + 2 leaf nodes
    /// ```
    ///
    /// # See Also
    ///
    /// - [`map`](Expression::map) - Transform each node in the expression tree
    pub fn fold<T, F>(&self, init: T, f: &F) -> T
    where
        F: Fn(T, &Expression) -> T,
    {
        let acc = f(init, self);
        match self {
            Expression::Unary(_, expr) => expr.fold(acc, f),
            Expression::Binary(_, left, right) => {
                let acc = left.fold(acc, f);
                right.fold(acc, f)
            }
            Expression::Function(_, args) => {
                args.iter().fold(acc, |acc, arg| arg.fold(acc, f))
            }
            Expression::Power(base, exp) => {
                let acc = base.fold(acc, f);
                exp.fold(acc, f)
            }
            _ => acc,
        }
    }

    /// Apply basic algebraic simplifications.
    ///
    /// Recursively simplifies the expression using standard algebraic identities
    /// and constant folding. This method applies:
    ///
    /// # Simplification Rules
    ///
    /// ## Identity simplifications
    /// - `0 + x` → `x`, `x + 0` → `x`
    /// - `x - 0` → `x`
    /// - `0 * x` → `0`, `x * 0` → `0`
    /// - `1 * x` → `x`, `x * 1` → `x`
    /// - `x / 1` → `x`
    /// - `x^0` → `1` (where x ≠ 0)
    /// - `x^1` → `x`
    /// - `-(-x)` → `x`
    ///
    /// ## Constant folding
    /// - Evaluates operations on numeric constants (e.g., `2 + 3` → `5`)
    /// - Evaluates function calls with constant arguments (e.g., `sin(0)` → `0`)
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::ast::{Expression, Variable, BinaryOp};
    ///
    /// // 0 + x simplifies to x
    /// let x = Expression::Variable(Variable::new("x"));
    /// let expr = Expression::Binary(
    ///     BinaryOp::Add,
    ///     Box::new(Expression::Integer(0)),
    ///     Box::new(x.clone())
    /// );
    /// assert_eq!(expr.simplify(), x);
    ///
    /// // 2 * 3 simplifies to 6
    /// let expr2 = Expression::Binary(
    ///     BinaryOp::Mul,
    ///     Box::new(Expression::Integer(2)),
    ///     Box::new(Expression::Integer(3))
    /// );
    /// assert_eq!(expr2.simplify(), Expression::Integer(6));
    /// ```
    ///
    /// # Returns
    ///
    /// A new simplified expression. The original expression is unchanged.
    ///
    /// # See Also
    ///
    /// - [`evaluate`](Expression::evaluate) - Numerical evaluation with variable values
    pub fn simplify(&self) -> Expression {
        // First, recursively simplify sub-expressions
        let simplified = match self {
            Expression::Unary(op, expr) => {
                let simplified_expr = expr.simplify();
                match op {
                    UnaryOp::Neg => {
                        // -(-x) → x
                        if let Expression::Unary(UnaryOp::Neg, inner) = &simplified_expr {
                            inner.as_ref().clone()
                        } else {
                            Expression::Unary(*op, Box::new(simplified_expr))
                        }
                    }
                    _ => Expression::Unary(*op, Box::new(simplified_expr)),
                }
            }
            Expression::Binary(op, left, right) => {
                let left_simplified = left.simplify();
                let right_simplified = right.simplify();

                // First apply identity simplifications
                let after_identity = match op {
                    BinaryOp::Add => {
                        // 0 + x → x
                        if Self::is_zero(&left_simplified) {
                            return right_simplified;
                        }
                        // x + 0 → x
                        if Self::is_zero(&right_simplified) {
                            return left_simplified;
                        }
                        None
                    }
                    BinaryOp::Sub => {
                        // x - 0 → x
                        if Self::is_zero(&right_simplified) {
                            return left_simplified;
                        }
                        None
                    }
                    BinaryOp::Mul => {
                        // 0 * x → 0
                        if Self::is_zero(&left_simplified) {
                            return Expression::Integer(0);
                        }
                        // x * 0 → 0
                        if Self::is_zero(&right_simplified) {
                            return Expression::Integer(0);
                        }
                        // 1 * x → x
                        if Self::is_one(&left_simplified) {
                            return right_simplified;
                        }
                        // x * 1 → x
                        if Self::is_one(&right_simplified) {
                            return left_simplified;
                        }
                        None
                    }
                    BinaryOp::Div => {
                        // x / 1 → x
                        if Self::is_one(&right_simplified) {
                            return left_simplified;
                        }
                        None
                    }
                    _ => None,
                };

                if after_identity.is_some() {
                    return after_identity.unwrap();
                }

                // Constant folding: if both operands are numeric constants, evaluate
                if Self::is_numeric_constant(&left_simplified)
                    && Self::is_numeric_constant(&right_simplified)
                {
                    if let (Some(left_val), Some(right_val)) = (
                        Self::extract_numeric_value(&left_simplified),
                        Self::extract_numeric_value(&right_simplified),
                    ) {
                        let result = match op {
                            BinaryOp::Add => Some(left_val + right_val),
                            BinaryOp::Sub => Some(left_val - right_val),
                            BinaryOp::Mul => Some(left_val * right_val),
                            BinaryOp::Div => {
                                if right_val.abs() > 1e-10 {
                                    Some(left_val / right_val)
                                } else {
                                    None // Avoid division by zero
                                }
                            }
                            BinaryOp::Mod => Some(left_val % right_val),
                        };

                        if let Some(value) = result {
                            return Self::from_numeric_value(value);
                        }
                    }
                }

                Expression::Binary(*op, Box::new(left_simplified), Box::new(right_simplified))
            }
            Expression::Function(func, args) => {
                let simplified_args: Vec<Expression> = args.iter().map(|arg| arg.simplify()).collect();

                // Constant folding: if all arguments are numeric constants, evaluate the function
                if simplified_args.iter().all(Self::is_numeric_constant) {
                    // Try to evaluate the function with constant arguments
                    let temp_expr = Expression::Function(func.clone(), simplified_args.clone());
                    if let Some(value) = temp_expr.evaluate(&HashMap::new()) {
                        return Self::from_numeric_value(value);
                    }
                }

                Expression::Function(func.clone(), simplified_args)
            }
            Expression::Power(base, exp) => {
                let base_simplified = base.simplify();
                let exp_simplified = exp.simplify();

                // x^0 → 1 (where x != 0)
                if Self::is_zero(&exp_simplified) && !Self::is_zero(&base_simplified) {
                    return Expression::Integer(1);
                }
                // x^1 → x
                if Self::is_one(&exp_simplified) {
                    return base_simplified;
                }

                // Constant folding: if both base and exponent are numeric constants, evaluate
                if Self::is_numeric_constant(&base_simplified)
                    && Self::is_numeric_constant(&exp_simplified)
                {
                    if let (Some(base_val), Some(exp_val)) = (
                        Self::extract_numeric_value(&base_simplified),
                        Self::extract_numeric_value(&exp_simplified),
                    ) {
                        let result = base_val.powf(exp_val);
                        if result.is_finite() {
                            return Self::from_numeric_value(result);
                        }
                    }
                }

                Expression::Power(Box::new(base_simplified), Box::new(exp_simplified))
            }
            _ => self.clone(),
        };

        simplified
    }

    /// Check if expression is zero.
    fn is_zero(expr: &Expression) -> bool {
        match expr {
            Expression::Integer(0) => true,
            Expression::Float(x) if *x == 0.0 => true,
            _ => false,
        }
    }

    /// Check if expression is one.
    fn is_one(expr: &Expression) -> bool {
        match expr {
            Expression::Integer(1) => true,
            Expression::Float(x) if *x == 1.0 => true,
            _ => false,
        }
    }

    /// Check if expression is a numeric constant.
    fn is_numeric_constant(expr: &Expression) -> bool {
        matches!(
            expr,
            Expression::Integer(_) | Expression::Float(_) | Expression::Rational(_)
        )
    }

    /// Extract numeric value from constant expression.
    fn extract_numeric_value(expr: &Expression) -> Option<f64> {
        match expr {
            Expression::Integer(n) => Some(*n as f64),
            Expression::Float(x) => Some(*x),
            Expression::Rational(r) => Some(*r.numer() as f64 / *r.denom() as f64),
            _ => None,
        }
    }

    /// Create numeric expression from value (Integer if whole, Float otherwise).
    fn from_numeric_value(value: f64) -> Expression {
        if value.is_finite() && value.fract().abs() < 1e-10 {
            Expression::Integer(value.round() as i64)
        } else {
            Expression::Float(value)
        }
    }

    /// Compute the symbolic derivative of this expression with respect to a variable.
    ///
    /// Performs symbolic differentiation using standard calculus rules. The result
    /// is an exact symbolic expression (not a numerical approximation) that may
    /// benefit from simplification.
    ///
    /// # Differentiation Rules
    ///
    /// ## Basic rules
    /// - Constant rule: d/dx[c] = 0
    /// - Variable rule: d/dx[x] = 1, d/dx[y] = 0
    /// - Power rule: d/dx[x^n] = n·x^(n-1)
    /// - Sum rule: d/dx[u+v] = du/dx + dv/dx
    /// - Difference rule: d/dx[u-v] = du/dx - dv/dx
    ///
    /// ## Product and quotient
    /// - Product rule: d/dx[u·v] = u·(dv/dx) + v·(du/dx)
    /// - Quotient rule: d/dx[u/v] = (v·du/dx - u·dv/dx) / v²
    ///
    /// ## Chain rule
    /// - d/dx[f(g(x))] = f'(g(x))·g'(x)
    ///
    /// ## Exponential and logarithmic
    /// - d/dx[exp(u)] = exp(u)·du/dx
    /// - d/dx[ln(u)] = (1/u)·du/dx
    /// - d/dx[log₁₀(u)] = (1/(u·ln(10)))·du/dx
    ///
    /// ## Trigonometric
    /// - d/dx[sin(u)] = cos(u)·du/dx
    /// - d/dx[cos(u)] = -sin(u)·du/dx
    /// - d/dx[tan(u)] = sec²(u)·du/dx
    ///
    /// # Arguments
    ///
    /// * `with_respect_to` - Name of the variable to differentiate with respect to
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::ast::{Expression, Variable, BinaryOp, Function};
    ///
    /// // d/dx[x^2] = 2*x
    /// let x = Expression::Variable(Variable::new("x"));
    /// let x_squared = Expression::Power(
    ///     Box::new(x.clone()),
    ///     Box::new(Expression::Integer(2))
    /// );
    /// let derivative = x_squared.differentiate("x");
    /// // Result is: 2 * x^(2-1) = 2 * x
    ///
    /// // d/dx[sin(x)] = cos(x)
    /// let sin_x = Expression::Function(Function::Sin, vec![x.clone()]);
    /// let d_sin = sin_x.differentiate("x");
    /// // Result is: cos(x) * 1 = cos(x)
    /// ```
    ///
    /// # Returns
    ///
    /// A new expression representing the derivative. The result may contain
    /// redundant terms (like multiplication by 1) that can be removed with
    /// [`simplify`](Expression::simplify).
    ///
    /// # See Also
    ///
    /// - [`simplify`](Expression::simplify) - Simplify the derivative result
    pub fn differentiate(&self, with_respect_to: &str) -> Expression {
        match self {
            // Constant rule: d/dx[c] = 0
            Expression::Integer(_) | Expression::Rational(_) | Expression::Float(_) | Expression::Complex(_) => {
                Expression::Integer(0)
            }

            // Variable rule: d/dx[x] = 1, d/dx[y] = 0
            Expression::Variable(v) => {
                if v.name == with_respect_to {
                    Expression::Integer(1)
                } else {
                    Expression::Integer(0)
                }
            }

            // Unary operations
            Expression::Unary(op, expr) => {
                let inner_derivative = expr.differentiate(with_respect_to);
                match op {
                    // d/dx[-f] = -f'
                    UnaryOp::Neg => {
                        Expression::Unary(UnaryOp::Neg, Box::new(inner_derivative))
                    }
                    // d/dx[|f|] = sign(f) * f' (simplified, assumes f != 0)
                    UnaryOp::Abs => {
                        let sign = Expression::Function(Function::Sign, vec![expr.as_ref().clone()]);
                        Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(sign),
                            Box::new(inner_derivative),
                        )
                    }
                    // d/dx[!f] = 0 (logical NOT is discrete)
                    UnaryOp::Not => Expression::Integer(0),
                }
            }

            // Binary operations
            Expression::Binary(op, left, right) => {
                let left_deriv = left.differentiate(with_respect_to);
                let right_deriv = right.differentiate(with_respect_to);

                match op {
                    // Sum rule: d/dx[u + v] = du/dx + dv/dx
                    BinaryOp::Add => {
                        Expression::Binary(BinaryOp::Add, Box::new(left_deriv), Box::new(right_deriv))
                    }

                    // Difference rule: d/dx[u - v] = du/dx - dv/dx
                    BinaryOp::Sub => {
                        Expression::Binary(BinaryOp::Sub, Box::new(left_deriv), Box::new(right_deriv))
                    }

                    // Product rule: d/dx[u * v] = u * dv/dx + v * du/dx
                    BinaryOp::Mul => {
                        let term1 = Expression::Binary(
                            BinaryOp::Mul,
                            left.clone(),
                            Box::new(right_deriv),
                        );
                        let term2 = Expression::Binary(
                            BinaryOp::Mul,
                            right.clone(),
                            Box::new(left_deriv),
                        );
                        Expression::Binary(BinaryOp::Add, Box::new(term1), Box::new(term2))
                    }

                    // Quotient rule: d/dx[u / v] = (v * du/dx - u * dv/dx) / v^2
                    BinaryOp::Div => {
                        let numerator_term1 = Expression::Binary(
                            BinaryOp::Mul,
                            right.clone(),
                            Box::new(left_deriv),
                        );
                        let numerator_term2 = Expression::Binary(
                            BinaryOp::Mul,
                            left.clone(),
                            Box::new(right_deriv),
                        );
                        let numerator = Expression::Binary(
                            BinaryOp::Sub,
                            Box::new(numerator_term1),
                            Box::new(numerator_term2),
                        );
                        let denominator = Expression::Power(
                            right.clone(),
                            Box::new(Expression::Integer(2)),
                        );
                        Expression::Binary(BinaryOp::Div, Box::new(numerator), Box::new(denominator))
                    }

                    // Modulo: derivative is complex, not commonly needed
                    BinaryOp::Mod => Expression::Integer(0),
                }
            }

            // Power rule with chain rule
            Expression::Power(base, exponent) => {
                let base_has_var = base.contains_variable(with_respect_to);
                let exp_has_var = exponent.contains_variable(with_respect_to);

                if !base_has_var && !exp_has_var {
                    // d/dx[c^d] = 0 (constant)
                    Expression::Integer(0)
                } else if base_has_var && !exp_has_var {
                    // Power rule: d/dx[u^n] = n * u^(n-1) * du/dx
                    let base_deriv = base.differentiate(with_respect_to);
                    let n_minus_1 = Expression::Binary(
                        BinaryOp::Sub,
                        exponent.clone(),
                        Box::new(Expression::Integer(1)),
                    );
                    let power_term = Expression::Power(base.clone(), Box::new(n_minus_1));
                    let scaled = Expression::Binary(
                        BinaryOp::Mul,
                        exponent.clone(),
                        Box::new(power_term),
                    );
                    Expression::Binary(
                        BinaryOp::Mul,
                        Box::new(scaled),
                        Box::new(base_deriv),
                    )
                } else if !base_has_var && exp_has_var {
                    // Exponential rule: d/dx[a^v] = a^v * ln(a) * dv/dx
                    let exp_deriv = exponent.differentiate(with_respect_to);
                    let ln_base = Expression::Function(Function::Ln, vec![base.as_ref().clone()]);
                    let power_term = Expression::Power(base.clone(), exponent.clone());
                    let scaled = Expression::Binary(
                        BinaryOp::Mul,
                        Box::new(power_term),
                        Box::new(ln_base),
                    );
                    Expression::Binary(
                        BinaryOp::Mul,
                        Box::new(scaled),
                        Box::new(exp_deriv),
                    )
                } else {
                    // General case: d/dx[u^v] = u^v * (v' * ln(u) + v * u'/u)
                    // This is the full logarithmic differentiation formula
                    let base_deriv = base.differentiate(with_respect_to);
                    let exp_deriv = exponent.differentiate(with_respect_to);

                    let ln_base = Expression::Function(Function::Ln, vec![base.as_ref().clone()]);
                    let term1 = Expression::Binary(
                        BinaryOp::Mul,
                        Box::new(exp_deriv),
                        Box::new(ln_base),
                    );

                    let u_prime_over_u = Expression::Binary(
                        BinaryOp::Div,
                        Box::new(base_deriv),
                        base.clone(),
                    );
                    let term2 = Expression::Binary(
                        BinaryOp::Mul,
                        exponent.clone(),
                        Box::new(u_prime_over_u),
                    );

                    let sum = Expression::Binary(BinaryOp::Add, Box::new(term1), Box::new(term2));
                    let power = Expression::Power(base.clone(), exponent.clone());

                    Expression::Binary(
                        BinaryOp::Mul,
                        Box::new(power),
                        Box::new(sum),
                    )
                }
            }

            // Function derivatives with chain rule
            Expression::Function(func, args) => {
                if args.is_empty() {
                    return Expression::Integer(0);
                }

                match func {
                    // Trigonometric functions
                    Function::Sin => {
                        // d/dx[sin(u)] = cos(u) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let cos_u = Expression::Function(Function::Cos, vec![arg.clone()]);
                        Expression::Binary(BinaryOp::Mul, Box::new(cos_u), Box::new(arg_deriv))
                    }

                    Function::Cos => {
                        // d/dx[cos(u)] = -sin(u) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let sin_u = Expression::Function(Function::Sin, vec![arg.clone()]);
                        let neg_sin = Expression::Unary(UnaryOp::Neg, Box::new(sin_u));
                        Expression::Binary(BinaryOp::Mul, Box::new(neg_sin), Box::new(arg_deriv))
                    }

                    Function::Tan => {
                        // d/dx[tan(u)] = sec^2(u) * du/dx = (1/cos^2(u)) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let cos_u = Expression::Function(Function::Cos, vec![arg.clone()]);
                        let cos_squared = Expression::Power(
                            Box::new(cos_u),
                            Box::new(Expression::Integer(2)),
                        );
                        let sec_squared = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(1)),
                            Box::new(cos_squared),
                        );
                        Expression::Binary(BinaryOp::Mul, Box::new(sec_squared), Box::new(arg_deriv))
                    }

                    // Inverse trigonometric functions
                    Function::Asin => {
                        // d/dx[asin(u)] = 1/sqrt(1 - u^2) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let u_squared = Expression::Power(
                            Box::new(arg.clone()),
                            Box::new(Expression::Integer(2)),
                        );
                        let one_minus_u_sq = Expression::Binary(
                            BinaryOp::Sub,
                            Box::new(Expression::Integer(1)),
                            Box::new(u_squared),
                        );
                        let sqrt_term = Expression::Function(Function::Sqrt, vec![one_minus_u_sq]);
                        let deriv_factor = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(1)),
                            Box::new(sqrt_term),
                        );
                        Expression::Binary(BinaryOp::Mul, Box::new(deriv_factor), Box::new(arg_deriv))
                    }

                    Function::Acos => {
                        // d/dx[acos(u)] = -1/sqrt(1 - u^2) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let u_squared = Expression::Power(
                            Box::new(arg.clone()),
                            Box::new(Expression::Integer(2)),
                        );
                        let one_minus_u_sq = Expression::Binary(
                            BinaryOp::Sub,
                            Box::new(Expression::Integer(1)),
                            Box::new(u_squared),
                        );
                        let sqrt_term = Expression::Function(Function::Sqrt, vec![one_minus_u_sq]);
                        let deriv_factor = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(1)),
                            Box::new(sqrt_term),
                        );
                        let neg_deriv = Expression::Unary(UnaryOp::Neg, Box::new(deriv_factor));
                        Expression::Binary(BinaryOp::Mul, Box::new(neg_deriv), Box::new(arg_deriv))
                    }

                    Function::Atan => {
                        // d/dx[atan(u)] = 1/(1 + u^2) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let u_squared = Expression::Power(
                            Box::new(arg.clone()),
                            Box::new(Expression::Integer(2)),
                        );
                        let one_plus_u_sq = Expression::Binary(
                            BinaryOp::Add,
                            Box::new(Expression::Integer(1)),
                            Box::new(u_squared),
                        );
                        let deriv_factor = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(1)),
                            Box::new(one_plus_u_sq),
                        );
                        Expression::Binary(BinaryOp::Mul, Box::new(deriv_factor), Box::new(arg_deriv))
                    }

                    Function::Atan2 => {
                        // d/dx[atan2(y, x)] is more complex, not commonly needed for uncertainty propagation
                        Expression::Integer(0)
                    }

                    // Hyperbolic functions
                    Function::Sinh => {
                        // d/dx[sinh(u)] = cosh(u) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let cosh_u = Expression::Function(Function::Cosh, vec![arg.clone()]);
                        Expression::Binary(BinaryOp::Mul, Box::new(cosh_u), Box::new(arg_deriv))
                    }

                    Function::Cosh => {
                        // d/dx[cosh(u)] = sinh(u) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let sinh_u = Expression::Function(Function::Sinh, vec![arg.clone()]);
                        Expression::Binary(BinaryOp::Mul, Box::new(sinh_u), Box::new(arg_deriv))
                    }

                    Function::Tanh => {
                        // d/dx[tanh(u)] = sech^2(u) * du/dx = (1/cosh^2(u)) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let cosh_u = Expression::Function(Function::Cosh, vec![arg.clone()]);
                        let cosh_squared = Expression::Power(
                            Box::new(cosh_u),
                            Box::new(Expression::Integer(2)),
                        );
                        let sech_squared = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(1)),
                            Box::new(cosh_squared),
                        );
                        Expression::Binary(BinaryOp::Mul, Box::new(sech_squared), Box::new(arg_deriv))
                    }

                    // Exponential and logarithmic functions
                    Function::Exp => {
                        // d/dx[exp(u)] = exp(u) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let exp_u = Expression::Function(Function::Exp, vec![arg.clone()]);
                        Expression::Binary(BinaryOp::Mul, Box::new(exp_u), Box::new(arg_deriv))
                    }

                    Function::Ln => {
                        // d/dx[ln(u)] = (1/u) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let one_over_u = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(1)),
                            Box::new(arg.clone()),
                        );
                        Expression::Binary(BinaryOp::Mul, Box::new(one_over_u), Box::new(arg_deriv))
                    }

                    Function::Log10 => {
                        // d/dx[log10(u)] = 1/(u * ln(10)) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let ln_10 = Expression::Function(
                            Function::Ln,
                            vec![Expression::Integer(10)],
                        );
                        let u_times_ln10 = Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(arg.clone()),
                            Box::new(ln_10),
                        );
                        let deriv_factor = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(1)),
                            Box::new(u_times_ln10),
                        );
                        Expression::Binary(BinaryOp::Mul, Box::new(deriv_factor), Box::new(arg_deriv))
                    }

                    Function::Log2 => {
                        // d/dx[log2(u)] = 1/(u * ln(2)) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let ln_2 = Expression::Function(
                            Function::Ln,
                            vec![Expression::Integer(2)],
                        );
                        let u_times_ln2 = Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(arg.clone()),
                            Box::new(ln_2),
                        );
                        let deriv_factor = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(1)),
                            Box::new(u_times_ln2),
                        );
                        Expression::Binary(BinaryOp::Mul, Box::new(deriv_factor), Box::new(arg_deriv))
                    }

                    Function::Log => {
                        // d/dx[log(u, b)] = 1/(u * ln(b)) * du/dx
                        if args.len() >= 2 {
                            let arg = &args[0];
                            let base = &args[1];
                            let arg_deriv = arg.differentiate(with_respect_to);
                            let ln_base = Expression::Function(Function::Ln, vec![base.clone()]);
                            let u_times_lnb = Expression::Binary(
                                BinaryOp::Mul,
                                Box::new(arg.clone()),
                                Box::new(ln_base),
                            );
                            let deriv_factor = Expression::Binary(
                                BinaryOp::Div,
                                Box::new(Expression::Integer(1)),
                                Box::new(u_times_lnb),
                            );
                            Expression::Binary(BinaryOp::Mul, Box::new(deriv_factor), Box::new(arg_deriv))
                        } else {
                            Expression::Integer(0)
                        }
                    }

                    // Root functions
                    Function::Sqrt => {
                        // d/dx[sqrt(u)] = 1/(2*sqrt(u)) * du/dx = (1/2) * u^(-1/2) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let sqrt_u = Expression::Function(Function::Sqrt, vec![arg.clone()]);
                        let two_sqrt_u = Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(Expression::Integer(2)),
                            Box::new(sqrt_u),
                        );
                        let deriv_factor = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(1)),
                            Box::new(two_sqrt_u),
                        );
                        Expression::Binary(BinaryOp::Mul, Box::new(deriv_factor), Box::new(arg_deriv))
                    }

                    Function::Cbrt => {
                        // d/dx[cbrt(u)] = 1/(3*u^(2/3)) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let two_thirds = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(2)),
                            Box::new(Expression::Integer(3)),
                        );
                        let u_to_2_3 = Expression::Power(
                            Box::new(arg.clone()),
                            Box::new(two_thirds),
                        );
                        let three_u_2_3 = Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(Expression::Integer(3)),
                            Box::new(u_to_2_3),
                        );
                        let deriv_factor = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(1)),
                            Box::new(three_u_2_3),
                        );
                        Expression::Binary(BinaryOp::Mul, Box::new(deriv_factor), Box::new(arg_deriv))
                    }

                    Function::Pow => {
                        // pow(u, v) is equivalent to u^v, handle like Power
                        if args.len() >= 2 {
                            let power_expr = Expression::Power(
                                Box::new(args[0].clone()),
                                Box::new(args[1].clone()),
                            );
                            power_expr.differentiate(with_respect_to)
                        } else {
                            Expression::Integer(0)
                        }
                    }

                    // Rounding functions (derivatives are 0 almost everywhere)
                    Function::Floor | Function::Ceil | Function::Round => Expression::Integer(0),

                    // Absolute value and sign
                    Function::Abs => {
                        // d/dx[abs(u)] = sign(u) * du/dx (simplified)
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let sign_u = Expression::Function(Function::Sign, vec![arg.clone()]);
                        Expression::Binary(BinaryOp::Mul, Box::new(sign_u), Box::new(arg_deriv))
                    }

                    Function::Sign => {
                        // Derivative of sign function is 0 almost everywhere
                        Expression::Integer(0)
                    }

                    // Min/Max (derivatives are piecewise, simplified to 0)
                    Function::Min | Function::Max => Expression::Integer(0),

                    // Custom functions - cannot differentiate
                    Function::Custom(_) => Expression::Integer(0),
                }
            }
        }
    }

    /// Evaluate the expression with the given variable values.
    ///
    /// Recursively evaluates the expression tree to produce a single floating-point result.
    /// All variables must have values provided in the `vars` map, otherwise evaluation fails.
    ///
    /// # Arguments
    ///
    /// * `vars` - HashMap mapping variable names to their numeric values
    ///
    /// # Returns
    ///
    /// - `Some(f64)` - The computed result if evaluation succeeds
    /// - `None` - If evaluation fails due to:
    ///   - Missing variable value
    ///   - Division by zero
    ///   - Complex result when real number expected
    ///   - Invalid function argument (e.g., sqrt of negative, ln of negative)
    ///   - Custom function encountered (not evaluable)
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::ast::{Expression, Variable, BinaryOp, Function};
    /// use std::collections::HashMap;
    ///
    /// // Evaluate: x^2 + 2*x + 1 with x = 3
    /// let x = Expression::Variable(Variable::new("x"));
    /// let x_squared = Expression::Power(
    ///     Box::new(x.clone()),
    ///     Box::new(Expression::Integer(2))
    /// );
    /// let two_x = Expression::Binary(
    ///     BinaryOp::Mul,
    ///     Box::new(Expression::Integer(2)),
    ///     Box::new(x.clone())
    /// );
    /// let sum1 = Expression::Binary(BinaryOp::Add, Box::new(x_squared), Box::new(two_x));
    /// let expr = Expression::Binary(BinaryOp::Add, Box::new(sum1), Box::new(Expression::Integer(1)));
    ///
    /// let mut vars = HashMap::new();
    /// vars.insert("x".to_string(), 3.0);
    /// assert_eq!(expr.evaluate(&vars), Some(16.0)); // 3^2 + 2*3 + 1 = 16
    ///
    /// // Evaluate function: sqrt(16)
    /// let sqrt_16 = Expression::Function(
    ///     Function::Sqrt,
    ///     vec![Expression::Integer(16)]
    /// );
    /// assert_eq!(sqrt_16.evaluate(&HashMap::new()), Some(4.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`simplify`](Expression::simplify) - Symbolic simplification before evaluation
    /// - [`variables`](Expression::variables) - Get all variables that need values
    pub fn evaluate(&self, vars: &HashMap<String, f64>) -> Option<f64> {
        match self {
            Expression::Integer(n) => Some(*n as f64),
            Expression::Rational(r) => Some(*r.numer() as f64 / *r.denom() as f64),
            Expression::Float(x) => Some(*x),
            Expression::Complex(c) => {
                // Only return real part if imaginary is zero
                if c.im.abs() < 1e-10 {
                    Some(c.re)
                } else {
                    None
                }
            }
            Expression::Variable(v) => vars.get(&v.name).copied(),
            Expression::Unary(op, expr) => {
                let val = expr.evaluate(vars)?;
                match op {
                    UnaryOp::Neg => Some(-val),
                    UnaryOp::Not => Some(if val == 0.0 { 1.0 } else { 0.0 }),
                    UnaryOp::Abs => Some(val.abs()),
                }
            }
            Expression::Binary(op, left, right) => {
                let left_val = left.evaluate(vars)?;
                let right_val = right.evaluate(vars)?;
                match op {
                    BinaryOp::Add => Some(left_val + right_val),
                    BinaryOp::Sub => Some(left_val - right_val),
                    BinaryOp::Mul => Some(left_val * right_val),
                    BinaryOp::Div => {
                        if right_val.abs() < 1e-10 {
                            None
                        } else {
                            Some(left_val / right_val)
                        }
                    }
                    BinaryOp::Mod => Some(left_val % right_val),
                }
            }
            Expression::Function(func, args) => {
                let arg_vals: Option<Vec<f64>> =
                    args.iter().map(|arg| arg.evaluate(vars)).collect();
                let arg_vals = arg_vals?;

                match func {
                    Function::Sin => Some(arg_vals.get(0)?.sin()),
                    Function::Cos => Some(arg_vals.get(0)?.cos()),
                    Function::Tan => Some(arg_vals.get(0)?.tan()),
                    Function::Asin => Some(arg_vals.get(0)?.asin()),
                    Function::Acos => Some(arg_vals.get(0)?.acos()),
                    Function::Atan => Some(arg_vals.get(0)?.atan()),
                    Function::Atan2 => Some(arg_vals.get(0)?.atan2(*arg_vals.get(1)?)),
                    Function::Sinh => Some(arg_vals.get(0)?.sinh()),
                    Function::Cosh => Some(arg_vals.get(0)?.cosh()),
                    Function::Tanh => Some(arg_vals.get(0)?.tanh()),
                    Function::Exp => Some(arg_vals.get(0)?.exp()),
                    Function::Ln => Some(arg_vals.get(0)?.ln()),
                    Function::Log => Some(arg_vals.get(0)?.log(*arg_vals.get(1)?)),
                    Function::Log2 => Some(arg_vals.get(0)?.log2()),
                    Function::Log10 => Some(arg_vals.get(0)?.log10()),
                    Function::Sqrt => Some(arg_vals.get(0)?.sqrt()),
                    Function::Cbrt => Some(arg_vals.get(0)?.cbrt()),
                    Function::Pow => Some(arg_vals.get(0)?.powf(*arg_vals.get(1)?)),
                    Function::Floor => Some(arg_vals.get(0)?.floor()),
                    Function::Ceil => Some(arg_vals.get(0)?.ceil()),
                    Function::Round => Some(arg_vals.get(0)?.round()),
                    Function::Abs => Some(arg_vals.get(0)?.abs()),
                    Function::Sign => Some(arg_vals.get(0)?.signum()),
                    Function::Min => {
                        arg_vals.iter().copied().reduce(f64::min)
                    }
                    Function::Max => {
                        arg_vals.iter().copied().reduce(f64::max)
                    }
                    Function::Custom(_) => None,
                }
            }
            Expression::Power(base, exp) => {
                let base_val = base.evaluate(vars)?;
                let exp_val = exp.evaluate(vars)?;
                Some(base_val.powf(exp_val))
            }
        }
    }
}

// TODO: Add support for matrices and vectors
// TODO: Add support for units and dimensional analysis
