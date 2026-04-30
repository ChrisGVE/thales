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
//! use thales::ast::{Expression, Variable, BinaryOp};
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

mod construction;
mod differentiation;
mod display;
mod evaluation;
mod simplification;

use num_complex::Complex64;
use num_rational::Rational64;
use serde::{Deserialize, Serialize};
use std::fmt;

// Sub-modules add `impl Expression` blocks for their respective functionality.
// No re-exports needed since the sub-modules don't define new public types.

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
/// use thales::ast::{Equation, Expression, Variable, BinaryOp};
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
/// use thales::ast::{Equation, Expression, Variable, BinaryOp};
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
/// use thales::ast::{Equation, Expression, Variable, Function, BinaryOp};
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
/// use thales::ast::{Equation, Expression, Variable, BinaryOp};
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
/// use thales::ast::{Equation, Expression, Variable, BinaryOp};
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
/// use thales::ast::{Equation, Expression, Variable, BinaryOp};
/// // use thales::solver::Solver; // Future solver module
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
    /// use thales::ast::{Equation, Expression};
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
    /// use thales::ast::{Equation, Expression, Variable, BinaryOp};
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
    /// use thales::ast::{Equation, Expression, Variable, BinaryOp};
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
    /// use thales::ast::{Equation, Expression};
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
    /// use thales::ast::{Equation, Expression, Variable, BinaryOp, Function};
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

impl fmt::Display for Equation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} = {}", self.left, self.right)
    }
}

/// Symbolic mathematical constants.
///
/// Represents well-known mathematical constants that should be preserved symbolically
/// during manipulation and only evaluated numerically when explicitly requested.
///
/// # Examples
///
/// ```
/// use thales::ast::{Expression, SymbolicConstant};
///
/// // Create symbolic pi
/// let pi = Expression::Constant(SymbolicConstant::Pi);
/// assert_eq!(format!("{}", pi), "π");
///
/// // Create Euler's number
/// let e = Expression::Constant(SymbolicConstant::E);
/// assert_eq!(format!("{}", e), "e");
///
/// // Create imaginary unit
/// let i = Expression::Constant(SymbolicConstant::I);
/// assert_eq!(format!("{}", i), "i");
/// ```
///
/// # Numerical Values
///
/// When evaluated numerically:
/// - `Pi` → 3.141592653589793 (std::f64::consts::PI)
/// - `E` → 2.718281828459045 (std::f64::consts::E)
/// - `I` → Complex(0, 1) (imaginary unit)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymbolicConstant {
    /// Pi (π ≈ 3.14159...) - ratio of circle's circumference to diameter
    Pi,
    /// Euler's number (e ≈ 2.71828...) - base of natural logarithm
    E,
    /// Imaginary unit (i) - satisfies i² = -1
    I,
}

impl fmt::Display for SymbolicConstant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymbolicConstant::Pi => write!(f, "π"),
            SymbolicConstant::E => write!(f, "e"),
            SymbolicConstant::I => write!(f, "i"),
        }
    }
}

/// A mathematical expression in the AST.
///
/// Represents any mathematical value or operation, from simple numeric literals
/// to complex nested expressions involving variables, operators, and functions.
/// This is the core type for building and manipulating symbolic mathematics.
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
/// use thales::ast::{Expression, Variable, BinaryOp, UnaryOp, Function};
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
/// use thales::ast::{Expression, BinaryOp};
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
/// use thales::ast::{Expression, Variable, BinaryOp};
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
/// use thales::ast::{Expression, Variable, Function};
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
/// - [`SymbolicConstant`] - Mathematical constants (Pi, E, I)
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// Integer literal.
    ///
    /// Represents whole number constants (e.g., 0, 42, -17).
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::ast::Expression;
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
    /// use thales::ast::Expression;
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
    /// use thales::ast::Expression;
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
    /// use thales::ast::Expression;
    /// use num_complex::Complex64;
    ///
    /// // 3 + 4i
    /// let z1 = Expression::Complex(Complex64::new(3.0, 4.0));
    ///
    /// // -2i (purely imaginary)
    /// let z2 = Expression::Complex(Complex64::new(0.0, -2.0));
    /// ```
    Complex(Complex64),

    /// Symbolic mathematical constant.
    ///
    /// Represents well-known constants (π, e, i) that are preserved symbolically
    /// during manipulation and only evaluated numerically when requested.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::ast::{Expression, SymbolicConstant};
    ///
    /// // Pi: π
    /// let pi = Expression::Constant(SymbolicConstant::Pi);
    ///
    /// // Euler's number: e
    /// let euler = Expression::Constant(SymbolicConstant::E);
    ///
    /// // Imaginary unit: i
    /// let imag = Expression::Constant(SymbolicConstant::I);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`SymbolicConstant`] - Available symbolic constants
    Constant(SymbolicConstant),

    /// Variable reference.
    ///
    /// Represents a named variable (e.g., x, velocity, temperature).
    /// Variables can optionally include dimension/unit information.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::ast::{Expression, Variable};
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
    /// use thales::ast::{Expression, Variable, UnaryOp};
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
    /// use thales::ast::{Expression, Variable, BinaryOp};
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
    /// use thales::ast::{Expression, Variable, Function};
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
    /// use thales::ast::{Expression, Variable};
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
/// use thales::ast::Variable;
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
    /// use thales::ast::Variable;
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
    /// use thales::ast::Variable;
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
/// use thales::ast::{Expression, Variable, UnaryOp};
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
#[non_exhaustive]
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
/// use thales::ast::{Expression, BinaryOp};
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
#[non_exhaustive]
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
    /// use thales::ast::BinaryOp;
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
/// use thales::ast::{Expression, Variable, Function};
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
#[non_exhaustive]
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

    /// Logarithm with arbitrary base: log(value, base)
    ///
    /// Returns the logarithm of `value` in the given `base`.
    /// With a single argument, log(value) is equivalent to log10(value).
    ///
    /// # Mathematical notation
    ///
    /// log_base(value)
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

    /// Real part of a complex expression: Re(z).
    ///
    /// For real inputs, Re(x) = x.
    Re,

    /// Imaginary part of a complex expression: Im(z).
    ///
    /// For real inputs, Im(x) = 0.
    Im,

    /// Complex conjugate: Conj(z) = Re(z) - i·Im(z).
    ///
    /// For real inputs, Conj(x) = x.
    Conj,

    // ── Special functions ────────────────────────────────────────────────────
    /// Gamma function Γ(x)
    Gamma,
    /// Log-gamma ln(Γ(x))
    LnGamma,
    /// Digamma ψ(x)
    Digamma,
    /// Beta function B(a,b) — note: 2 arguments
    BetaFn,
    /// Error function erf(x)
    Erf,
    /// Complementary error function erfc(x)
    Erfc,
    /// Bessel J_ν(x) — 2 arguments (order, argument)
    BesselJ,
    /// Bessel Y_ν(x)
    BesselY,
    /// Modified Bessel I_ν(x)
    BesselI,
    /// Modified Bessel K_ν(x)
    BesselK,
    /// Airy Ai(x)
    AiryAi,
    /// Airy Bi(x)
    AiryBi,
    /// Riemann zeta ζ(s)
    Zeta,
    /// Sine integral Si(x)
    Si,
    /// Cosine integral Ci(x)
    Ci,
    /// Exponential integral Ei(x)
    Ei,
    /// Heaviside step H(x)
    Heaviside,
    /// Dirac delta δ(x)
    DiracDelta,

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
            Function::Re => write!(f, "Re"),
            Function::Im => write!(f, "Im"),
            Function::Conj => write!(f, "Conj"),
            Function::Gamma => write!(f, "gamma"),
            Function::LnGamma => write!(f, "lngamma"),
            Function::Digamma => write!(f, "digamma"),
            Function::BetaFn => write!(f, "beta"),
            Function::Erf => write!(f, "erf"),
            Function::Erfc => write!(f, "erfc"),
            Function::BesselJ => write!(f, "besselJ"),
            Function::BesselY => write!(f, "besselY"),
            Function::BesselI => write!(f, "besselI"),
            Function::BesselK => write!(f, "besselK"),
            Function::AiryAi => write!(f, "airyAi"),
            Function::AiryBi => write!(f, "airyBi"),
            Function::Zeta => write!(f, "zeta"),
            Function::Si => write!(f, "Si"),
            Function::Ci => write!(f, "Ci"),
            Function::Ei => write!(f, "Ei"),
            Function::Heaviside => write!(f, "heaviside"),
            Function::DiracDelta => write!(f, "dirac"),
            Function::Custom(name) => write!(f, "{}", name),
        }
    }
}

// TODO: Add support for matrices and vectors
// TODO: Add support for units and dimensional analysis
