//! Abstract Syntax Tree definitions for mathematical expressions.
//!
//! Provides the core data structures for representing mathematical equations,
//! expressions, variables, operators, and functions in a tree structure
//! suitable for parsing, manipulation, and solving.

use num_complex::Complex64;
use num_rational::Rational64;
use std::fmt;

/// Represents a complete equation with left and right expressions.
#[derive(Debug, Clone, PartialEq)]
pub struct Equation {
    pub left: Expression,
    pub right: Expression,
}

impl Equation {
    /// Create a new equation from two expressions.
    pub fn new(left: Expression, right: Expression) -> Self {
        Self { left, right }
    }
}

/// Represents a mathematical expression in tree form.
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// Integer literal
    Integer(i64),
    /// Rational number (fraction)
    Rational(Rational64),
    /// Floating point number
    Float(f64),
    /// Complex number
    Complex(Complex64),
    /// Variable reference
    Variable(Variable),
    /// Unary operation (e.g., -x, sin(x))
    Unary(UnaryOp, Box<Expression>),
    /// Binary operation (e.g., x + y, x * y)
    Binary(BinaryOp, Box<Expression>, Box<Expression>),
    /// Function call with arguments
    Function(Function, Vec<Expression>),
    /// Power operation (base ^ exponent)
    Power(Box<Expression>, Box<Expression>),
}

/// Variable identifier with optional metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Variable {
    pub name: String,
    /// Optional dimension/unit information
    pub dimension: Option<String>,
}

impl Variable {
    /// Create a new variable with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dimension: None,
        }
    }

    /// Create a variable with dimension information.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Negation (-)
    Neg,
    /// Logical NOT
    Not,
    /// Absolute value
    Abs,
}

/// Binary operators (two operands).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// Addition (+)
    Add,
    /// Subtraction (-)
    Sub,
    /// Multiplication (*)
    Mul,
    /// Division (/)
    Div,
    /// Modulo (%)
    Mod,
}

impl BinaryOp {
    /// Returns the precedence level of this operator.
    /// Higher numbers bind tighter.
    pub fn precedence(self) -> u8 {
        match self {
            BinaryOp::Add | BinaryOp::Sub => 1,
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => 2,
        }
    }
}

/// Mathematical functions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Function {
    // Trigonometric
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,

    // Hyperbolic
    Sinh,
    Cosh,
    Tanh,

    // Exponential and logarithmic
    Exp,
    Ln,
    Log,
    Log2,
    Log10,

    // Power and root
    Sqrt,
    Cbrt,
    Pow,

    // Rounding
    Floor,
    Ceil,
    Round,

    // Other
    Abs,
    Sign,
    Min,
    Max,

    // User-defined function
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

// TODO: Implement expression simplification
// TODO: Implement expression evaluation with variable substitution
// TODO: Implement expression differentiation
// TODO: Add support for matrices and vectors
// TODO: Add support for units and dimensional analysis
