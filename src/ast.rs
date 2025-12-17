//! Abstract Syntax Tree definitions for mathematical expressions.
//!
//! Provides the core data structures for representing mathematical equations,
//! expressions, variables, operators, and functions in a tree structure
//! suitable for parsing, manipulation, and solving.

use num_complex::Complex64;
use num_rational::Rational64;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;

/// Represents a complete equation with left and right expressions.
#[derive(Debug, Clone, PartialEq)]
pub struct Equation {
    pub id: String,
    pub left: Expression,
    pub right: Expression,
}

impl Equation {
    /// Create a new equation from two expressions.
    pub fn new(id: impl Into<String>, left: Expression, right: Expression) -> Self {
        Self {
            id: id.into(),
            left,
            right,
        }
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

                match op {
                    BinaryOp::Add => {
                        // 0 + x → x
                        if Self::is_zero(&left_simplified) {
                            return right_simplified;
                        }
                        // x + 0 → x
                        if Self::is_zero(&right_simplified) {
                            return left_simplified;
                        }
                        Expression::Binary(*op, Box::new(left_simplified), Box::new(right_simplified))
                    }
                    BinaryOp::Sub => {
                        // x - 0 → x
                        if Self::is_zero(&right_simplified) {
                            return left_simplified;
                        }
                        Expression::Binary(*op, Box::new(left_simplified), Box::new(right_simplified))
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
                        Expression::Binary(*op, Box::new(left_simplified), Box::new(right_simplified))
                    }
                    BinaryOp::Div => {
                        // x / 1 → x
                        if Self::is_one(&right_simplified) {
                            return left_simplified;
                        }
                        Expression::Binary(*op, Box::new(left_simplified), Box::new(right_simplified))
                    }
                    _ => Expression::Binary(*op, Box::new(left_simplified), Box::new(right_simplified)),
                }
            }
            Expression::Function(func, args) => {
                let simplified_args: Vec<Expression> = args.iter().map(|arg| arg.simplify()).collect();
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

    /// Evaluate the expression with the given variable values.
    /// Returns None if variables are missing or evaluation fails.
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

// TODO: Implement expression differentiation
// TODO: Add support for matrices and vectors
// TODO: Add support for units and dimensional analysis
