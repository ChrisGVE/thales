//! Abstract Syntax Tree definitions for mathematical expressions.
//!
//! Provides the core data structures for representing mathematical equations,
//! expressions, variables, operators, and functions in a tree structure
//! suitable for parsing, manipulation, and solving.

use num_complex::Complex64;
use num_rational::Rational64;
use serde::{Serialize, Deserialize};
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
    /// Uses standard differentiation rules:
    /// - Constant rule: d/dx[c] = 0
    /// - Power rule: d/dx[x^n] = n*x^(n-1)
    /// - Sum rule: d/dx[u+v] = du/dx + dv/dx
    /// - Product rule: d/dx[u*v] = u*dv/dx + v*du/dx
    /// - Quotient rule: d/dx[u/v] = (v*du/dx - u*dv/dx) / v^2
    /// - Chain rule: d/dx[f(g(x))] = f'(g(x)) * g'(x)
    ///
    /// # Arguments
    /// * `with_respect_to` - Name of the variable to differentiate with respect to
    ///
    /// # Returns
    /// A new expression representing the derivative
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

// TODO: Add support for matrices and vectors
// TODO: Add support for units and dimensional analysis
