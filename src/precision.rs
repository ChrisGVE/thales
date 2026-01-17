//! Numerical precision controls for expression evaluation.
//!
//! This module provides configurable precision settings for evaluating mathematical
//! expressions, supporting fixed decimal places, significant figures, and arbitrary
//! precision arithmetic.
//!
//! # Precision Modes
//!
//! - **Fixed Decimal**: Results rounded to a specific number of decimal places
//! - **Significant Figures**: Results maintain a specific number of significant digits
//! - **Arbitrary Precision**: Exact rational arithmetic for fractions
//!
//! # Examples
//!
//! ```
//! use thales::precision::{PrecisionMode, EvalContext, RoundingMode};
//! use thales::ast::Expression;
//! use std::collections::HashMap;
//!
//! // Evaluate 1/3 with 6 decimal places
//! let ctx = EvalContext::new(PrecisionMode::FixedDecimal(6));
//! let expr = Expression::Binary(
//!     thales::ast::BinaryOp::Div,
//!     Box::new(Expression::Integer(1)),
//!     Box::new(Expression::Integer(3)),
//! );
//! let result = ctx.evaluate(&expr).unwrap();
//! assert!((result.as_f64() - 0.333333).abs() < 1e-10);
//! ```

use crate::ast::{BinaryOp, Expression, Function, SymbolicConstant, UnaryOp};
use num_rational::Rational64;
use std::collections::HashMap;
use std::fmt;

/// Error types for precision evaluation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum EvalError {
    /// Division by zero.
    DivisionByZero,
    /// Undefined variable.
    UndefinedVariable(String),
    /// Domain error (e.g., sqrt of negative).
    DomainError(String),
    /// Overflow in computation.
    Overflow,
    /// Cannot evaluate expression (e.g., contains unevaluable parts).
    CannotEvaluate(String),
    /// Invalid operation.
    InvalidOperation(String),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::DivisionByZero => write!(f, "Division by zero"),
            EvalError::UndefinedVariable(name) => write!(f, "Undefined variable: {}", name),
            EvalError::DomainError(msg) => write!(f, "Domain error: {}", msg),
            EvalError::Overflow => write!(f, "Overflow in computation"),
            EvalError::CannotEvaluate(msg) => write!(f, "Cannot evaluate: {}", msg),
            EvalError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
        }
    }
}

impl std::error::Error for EvalError {}

/// Result type for precision evaluation.
pub type EvalResult<T> = Result<T, EvalError>;

/// Precision mode for numerical evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrecisionMode {
    /// Fixed number of decimal places (e.g., 6 decimal places).
    FixedDecimal(u32),
    /// Fixed number of significant figures (e.g., 10 significant figures).
    SignificantFigures(u32),
    /// Arbitrary precision using exact rationals where possible.
    Arbitrary,
    /// Full floating-point precision (default f64).
    Full,
}

impl Default for PrecisionMode {
    fn default() -> Self {
        PrecisionMode::Full
    }
}

/// Rounding mode for precision operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoundingMode {
    /// Round half up (0.5 → 1, -0.5 → 0).
    HalfUp,
    /// Round half even (banker's rounding: 0.5 → 0, 1.5 → 2).
    HalfEven,
    /// Truncate toward zero.
    Truncate,
    /// Round toward positive infinity (ceiling).
    Ceiling,
    /// Round toward negative infinity (floor).
    Floor,
}

impl Default for RoundingMode {
    fn default() -> Self {
        RoundingMode::HalfEven
    }
}

/// Value representation with precision information.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Exact integer value.
    Integer(i64),
    /// Exact rational value (fraction).
    Rational(Rational64),
    /// Floating-point value.
    Float(f64),
    /// Complex value (real, imaginary).
    Complex(f64, f64),
    /// Positive infinity.
    PositiveInfinity,
    /// Negative infinity.
    NegativeInfinity,
    /// Not a number.
    NaN,
}

impl Value {
    /// Convert value to f64, if possible.
    pub fn as_f64(&self) -> f64 {
        match self {
            Value::Integer(n) => *n as f64,
            Value::Rational(r) => *r.numer() as f64 / *r.denom() as f64,
            Value::Float(f) => *f,
            Value::Complex(re, _) => *re, // Real part only
            Value::PositiveInfinity => f64::INFINITY,
            Value::NegativeInfinity => f64::NEG_INFINITY,
            Value::NaN => f64::NAN,
        }
    }

    /// Check if value is finite.
    pub fn is_finite(&self) -> bool {
        match self {
            Value::Integer(_) | Value::Rational(_) => true,
            Value::Float(f) => f.is_finite(),
            Value::Complex(re, im) => re.is_finite() && im.is_finite(),
            _ => false,
        }
    }

    /// Check if value is real (not complex with imaginary part).
    pub fn is_real(&self) -> bool {
        match self {
            Value::Complex(_, im) => im.abs() < 1e-15,
            _ => true,
        }
    }

    /// Check if value is NaN.
    pub fn is_nan(&self) -> bool {
        matches!(self, Value::NaN) || matches!(self, Value::Float(f) if f.is_nan())
    }

    /// Check if value is zero.
    pub fn is_zero(&self) -> bool {
        match self {
            Value::Integer(0) => true,
            Value::Rational(r) => *r.numer() == 0,
            Value::Float(f) => f.abs() < 1e-15,
            Value::Complex(re, im) => re.abs() < 1e-15 && im.abs() < 1e-15,
            _ => false,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Integer(n) => write!(f, "{}", n),
            Value::Rational(r) => write!(f, "{}/{}", r.numer(), r.denom()),
            Value::Float(x) => write!(f, "{}", x),
            Value::Complex(re, im) => {
                if im.abs() < 1e-15 {
                    write!(f, "{}", re)
                } else if re.abs() < 1e-15 {
                    write!(f, "{}i", im)
                } else if *im >= 0.0 {
                    write!(f, "{}+{}i", re, im)
                } else {
                    write!(f, "{}{}i", re, im)
                }
            }
            Value::PositiveInfinity => write!(f, "∞"),
            Value::NegativeInfinity => write!(f, "-∞"),
            Value::NaN => write!(f, "NaN"),
        }
    }
}

/// Evaluation context with precision settings.
#[derive(Debug, Clone)]
pub struct EvalContext {
    /// Precision mode for evaluation.
    pub precision: PrecisionMode,
    /// Rounding mode for precision operations.
    pub rounding: RoundingMode,
    /// Variable bindings.
    pub variables: HashMap<String, Value>,
    /// Whether to allow complex results from real inputs.
    pub allow_complex: bool,
}

impl EvalContext {
    /// Create a new evaluation context with the given precision mode.
    pub fn new(precision: PrecisionMode) -> Self {
        Self {
            precision,
            rounding: RoundingMode::default(),
            variables: HashMap::new(),
            allow_complex: true,
        }
    }

    /// Create a new context with full precision.
    pub fn full_precision() -> Self {
        Self::new(PrecisionMode::Full)
    }

    /// Create a context with fixed decimal places.
    pub fn fixed_decimal(places: u32) -> Self {
        Self::new(PrecisionMode::FixedDecimal(places))
    }

    /// Create a context with significant figures.
    pub fn significant_figures(figures: u32) -> Self {
        Self::new(PrecisionMode::SignificantFigures(figures))
    }

    /// Create a context with arbitrary precision.
    pub fn arbitrary() -> Self {
        Self::new(PrecisionMode::Arbitrary)
    }

    /// Set the rounding mode.
    pub fn with_rounding(mut self, mode: RoundingMode) -> Self {
        self.rounding = mode;
        self
    }

    /// Set whether to allow complex results.
    pub fn with_complex(mut self, allow: bool) -> Self {
        self.allow_complex = allow;
        self
    }

    /// Set a variable value.
    pub fn set_variable(&mut self, name: &str, value: Value) {
        self.variables.insert(name.to_string(), value);
    }

    /// Set a variable from f64.
    pub fn set_f64(&mut self, name: &str, value: f64) {
        self.variables.insert(name.to_string(), Value::Float(value));
    }

    /// Set multiple variables from a HashMap.
    pub fn with_variables(mut self, vars: HashMap<String, f64>) -> Self {
        for (name, value) in vars {
            self.variables.insert(name, Value::Float(value));
        }
        self
    }

    /// Evaluate an expression with precision controls.
    pub fn evaluate(&self, expr: &Expression) -> EvalResult<Value> {
        let value = self.eval_recursive(expr)?;
        self.apply_precision(value)
    }

    /// Recursive evaluation of expressions.
    fn eval_recursive(&self, expr: &Expression) -> EvalResult<Value> {
        match expr {
            Expression::Integer(n) => Ok(Value::Integer(*n)),

            Expression::Rational(r) => {
                if matches!(self.precision, PrecisionMode::Arbitrary) {
                    Ok(Value::Rational(*r))
                } else {
                    Ok(Value::Float(*r.numer() as f64 / *r.denom() as f64))
                }
            }

            Expression::Float(f) => {
                if f.is_nan() {
                    Ok(Value::NaN)
                } else if f.is_infinite() {
                    if *f > 0.0 {
                        Ok(Value::PositiveInfinity)
                    } else {
                        Ok(Value::NegativeInfinity)
                    }
                } else {
                    Ok(Value::Float(*f))
                }
            }

            Expression::Complex(c) => Ok(Value::Complex(c.re, c.im)),

            Expression::Constant(c) => match c {
                SymbolicConstant::Pi => Ok(Value::Float(std::f64::consts::PI)),
                SymbolicConstant::E => Ok(Value::Float(std::f64::consts::E)),
                SymbolicConstant::I => Ok(Value::Complex(0.0, 1.0)),
            },

            Expression::Variable(v) => self
                .variables
                .get(&v.name)
                .cloned()
                .ok_or_else(|| EvalError::UndefinedVariable(v.name.clone())),

            Expression::Unary(op, inner) => {
                let val = self.eval_recursive(inner)?;
                self.eval_unary(*op, val)
            }

            Expression::Binary(op, left, right) => {
                let l = self.eval_recursive(left)?;
                let r = self.eval_recursive(right)?;
                self.eval_binary(*op, l, r)
            }

            Expression::Power(base, exp) => {
                let b = self.eval_recursive(base)?;
                let e = self.eval_recursive(exp)?;
                self.eval_power(b, e)
            }

            Expression::Function(func, args) => {
                let vals: Result<Vec<_>, _> = args.iter().map(|a| self.eval_recursive(a)).collect();
                self.eval_function(func, vals?)
            }
        }
    }

    /// Evaluate a unary operation.
    fn eval_unary(&self, op: UnaryOp, val: Value) -> EvalResult<Value> {
        match op {
            UnaryOp::Neg => match val {
                Value::Integer(n) => Ok(Value::Integer(-n)),
                Value::Rational(r) => Ok(Value::Rational(-r)),
                Value::Float(f) => Ok(Value::Float(-f)),
                Value::Complex(re, im) => Ok(Value::Complex(-re, -im)),
                Value::PositiveInfinity => Ok(Value::NegativeInfinity),
                Value::NegativeInfinity => Ok(Value::PositiveInfinity),
                Value::NaN => Ok(Value::NaN),
            },
            UnaryOp::Abs => {
                let f = val.as_f64();
                Ok(Value::Float(f.abs()))
            }
            UnaryOp::Not => {
                // Logical not: 0 → 1, non-zero → 0
                if val.is_zero() {
                    Ok(Value::Integer(1))
                } else {
                    Ok(Value::Integer(0))
                }
            }
        }
    }

    /// Evaluate a binary operation.
    fn eval_binary(&self, op: BinaryOp, left: Value, right: Value) -> EvalResult<Value> {
        // Handle special cases first
        if left.is_nan() || right.is_nan() {
            return Ok(Value::NaN);
        }

        match op {
            BinaryOp::Add => self.eval_add(left, right),
            BinaryOp::Sub => self.eval_sub(left, right),
            BinaryOp::Mul => self.eval_mul(left, right),
            BinaryOp::Div => self.eval_div(left, right),
            BinaryOp::Mod => self.eval_mod(left, right),
        }
    }

    fn eval_add(&self, left: Value, right: Value) -> EvalResult<Value> {
        match (left, right) {
            (Value::Integer(a), Value::Integer(b)) => a
                .checked_add(b)
                .map(Value::Integer)
                .ok_or(EvalError::Overflow),
            (Value::Rational(a), Value::Rational(b)) => Ok(Value::Rational(a + b)),
            (Value::Complex(re1, im1), Value::Complex(re2, im2)) => {
                Ok(Value::Complex(re1 + re2, im1 + im2))
            }
            (Value::Complex(re, im), other) | (other, Value::Complex(re, im)) => {
                let f = other.as_f64();
                Ok(Value::Complex(re + f, im))
            }
            (a, b) => Ok(Value::Float(a.as_f64() + b.as_f64())),
        }
    }

    fn eval_sub(&self, left: Value, right: Value) -> EvalResult<Value> {
        match (left, right) {
            (Value::Integer(a), Value::Integer(b)) => a
                .checked_sub(b)
                .map(Value::Integer)
                .ok_or(EvalError::Overflow),
            (Value::Rational(a), Value::Rational(b)) => Ok(Value::Rational(a - b)),
            (Value::Complex(re1, im1), Value::Complex(re2, im2)) => {
                Ok(Value::Complex(re1 - re2, im1 - im2))
            }
            (Value::Complex(re, im), other) => {
                let f = other.as_f64();
                Ok(Value::Complex(re - f, im))
            }
            (other, Value::Complex(re, im)) => {
                let f = other.as_f64();
                Ok(Value::Complex(f - re, -im))
            }
            (a, b) => Ok(Value::Float(a.as_f64() - b.as_f64())),
        }
    }

    fn eval_mul(&self, left: Value, right: Value) -> EvalResult<Value> {
        match (left, right) {
            (Value::Integer(a), Value::Integer(b)) => a
                .checked_mul(b)
                .map(Value::Integer)
                .ok_or(EvalError::Overflow),
            (Value::Rational(a), Value::Rational(b)) => Ok(Value::Rational(a * b)),
            (Value::Complex(re1, im1), Value::Complex(re2, im2)) => {
                // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                let re = re1 * re2 - im1 * im2;
                let im = re1 * im2 + im1 * re2;
                Ok(Value::Complex(re, im))
            }
            (Value::Complex(re, im), other) | (other, Value::Complex(re, im)) => {
                let f = other.as_f64();
                Ok(Value::Complex(re * f, im * f))
            }
            (a, b) => Ok(Value::Float(a.as_f64() * b.as_f64())),
        }
    }

    fn eval_div(&self, left: Value, right: Value) -> EvalResult<Value> {
        if right.is_zero() {
            return Err(EvalError::DivisionByZero);
        }

        match (left, right) {
            (Value::Integer(a), Value::Integer(b)) if a % b == 0 => Ok(Value::Integer(a / b)),
            (Value::Integer(a), Value::Integer(b)) => {
                if matches!(self.precision, PrecisionMode::Arbitrary) {
                    Ok(Value::Rational(Rational64::new(a, b)))
                } else {
                    Ok(Value::Float(a as f64 / b as f64))
                }
            }
            (Value::Rational(a), Value::Rational(b)) => Ok(Value::Rational(a / b)),
            (Value::Complex(re1, im1), Value::Complex(re2, im2)) => {
                // (a+bi)/(c+di) = (ac+bd)/(c²+d²) + (bc-ad)/(c²+d²)i
                let denom = re2 * re2 + im2 * im2;
                let re = (re1 * re2 + im1 * im2) / denom;
                let im = (im1 * re2 - re1 * im2) / denom;
                Ok(Value::Complex(re, im))
            }
            (Value::Complex(re, im), other) => {
                let f = other.as_f64();
                Ok(Value::Complex(re / f, im / f))
            }
            (other, Value::Complex(re2, im2)) => {
                let re1 = other.as_f64();
                let denom = re2 * re2 + im2 * im2;
                let re = (re1 * re2) / denom;
                let im = (-re1 * im2) / denom;
                Ok(Value::Complex(re, im))
            }
            (a, b) => Ok(Value::Float(a.as_f64() / b.as_f64())),
        }
    }

    fn eval_mod(&self, left: Value, right: Value) -> EvalResult<Value> {
        if right.is_zero() {
            return Err(EvalError::DivisionByZero);
        }
        let l = left.as_f64();
        let r = right.as_f64();
        Ok(Value::Float(l % r))
    }

    /// Evaluate a power operation.
    fn eval_power(&self, base: Value, exp: Value) -> EvalResult<Value> {
        let b = base.as_f64();
        let e = exp.as_f64();

        // Handle special cases
        if e == 0.0 {
            return Ok(Value::Integer(1));
        }
        if b == 0.0 && e > 0.0 {
            return Ok(Value::Integer(0));
        }
        if b == 1.0 {
            return Ok(Value::Integer(1));
        }

        // Check for complex result from negative base with non-integer exponent
        if b < 0.0 && e.fract() != 0.0 {
            if self.allow_complex {
                // Use complex logarithm: b^e = exp(e * ln(b))
                let ln_abs_b = b.abs().ln();
                let angle = std::f64::consts::PI; // arg of negative real is π
                let re = (e * ln_abs_b).exp() * (e * angle).cos();
                let im = (e * ln_abs_b).exp() * (e * angle).sin();
                return Ok(Value::Complex(re, im));
            } else {
                return Err(EvalError::DomainError(
                    "Negative base with non-integer exponent".to_string(),
                ));
            }
        }

        // Integer exponent with integer base
        if let (Value::Integer(base_int), Value::Integer(exp_int)) = (&base, &exp) {
            if *exp_int >= 0 && *exp_int <= 62 {
                if let Some(result) = base_int.checked_pow(*exp_int as u32) {
                    return Ok(Value::Integer(result));
                }
            }
        }

        let result = b.powf(e);
        if result.is_nan() {
            Ok(Value::NaN)
        } else if result.is_infinite() {
            if result > 0.0 {
                Ok(Value::PositiveInfinity)
            } else {
                Ok(Value::NegativeInfinity)
            }
        } else {
            Ok(Value::Float(result))
        }
    }

    /// Evaluate a function call.
    fn eval_function(&self, func: &Function, args: Vec<Value>) -> EvalResult<Value> {
        if args.is_empty() {
            return Err(EvalError::InvalidOperation(
                "Function requires arguments".to_string(),
            ));
        }

        let x = args[0].as_f64();

        let result = match func {
            Function::Sin => x.sin(),
            Function::Cos => x.cos(),
            Function::Tan => x.tan(),
            Function::Asin => {
                if x.abs() > 1.0 {
                    if self.allow_complex {
                        // asin(x) for |x|>1 has complex result
                        let im = ((x * x - 1.0).sqrt() + x.abs()).ln();
                        return Ok(Value::Complex(
                            if x >= 0.0 {
                                std::f64::consts::FRAC_PI_2
                            } else {
                                -std::f64::consts::FRAC_PI_2
                            },
                            if x >= 0.0 { im } else { -im },
                        ));
                    }
                    return Err(EvalError::DomainError(
                        "asin requires -1 <= x <= 1".to_string(),
                    ));
                }
                x.asin()
            }
            Function::Acos => {
                if x.abs() > 1.0 {
                    return Err(EvalError::DomainError(
                        "acos requires -1 <= x <= 1".to_string(),
                    ));
                }
                x.acos()
            }
            Function::Atan => x.atan(),
            Function::Sqrt => {
                if x < 0.0 {
                    if self.allow_complex {
                        return Ok(Value::Complex(0.0, (-x).sqrt()));
                    }
                    return Err(EvalError::DomainError("sqrt requires x >= 0".to_string()));
                }
                x.sqrt()
            }
            Function::Cbrt => x.cbrt(),
            Function::Exp => x.exp(),
            Function::Ln => {
                if x <= 0.0 {
                    if x == 0.0 {
                        return Ok(Value::NegativeInfinity);
                    }
                    if self.allow_complex {
                        return Ok(Value::Complex((-x).ln(), std::f64::consts::PI));
                    }
                    return Err(EvalError::DomainError("ln requires x > 0".to_string()));
                }
                x.ln()
            }
            Function::Log10 => {
                if x <= 0.0 {
                    return Err(EvalError::DomainError("log10 requires x > 0".to_string()));
                }
                x.log10()
            }
            Function::Log => {
                // log(x, base)
                if args.len() < 2 {
                    return Err(EvalError::InvalidOperation(
                        "log requires two arguments".to_string(),
                    ));
                }
                let base = args[1].as_f64();
                if x <= 0.0 || base <= 0.0 || base == 1.0 {
                    return Err(EvalError::DomainError(
                        "log requires x > 0 and base > 0, base != 1".to_string(),
                    ));
                }
                x.log(base)
            }
            Function::Abs => x.abs(),
            Function::Floor => x.floor(),
            Function::Ceil => x.ceil(),
            Function::Round => x.round(),
            Function::Min => {
                let mut min = x;
                for arg in &args[1..] {
                    let v = arg.as_f64();
                    if v < min {
                        min = v;
                    }
                }
                min
            }
            Function::Max => {
                let mut max = x;
                for arg in &args[1..] {
                    let v = arg.as_f64();
                    if v > max {
                        max = v;
                    }
                }
                max
            }
            Function::Atan2 => {
                if args.len() < 2 {
                    return Err(EvalError::InvalidOperation(
                        "atan2 requires two arguments".to_string(),
                    ));
                }
                let y = x;
                let x_arg = args[1].as_f64();
                y.atan2(x_arg)
            }
            Function::Sinh => x.sinh(),
            Function::Cosh => x.cosh(),
            Function::Tanh => x.tanh(),
            Function::Log2 => {
                if x <= 0.0 {
                    return Err(EvalError::DomainError("log2 requires x > 0".to_string()));
                }
                x.log2()
            }
            Function::Pow => {
                // pow(base, exp) - handled via Power expression typically
                if args.len() < 2 {
                    return Err(EvalError::InvalidOperation(
                        "pow requires two arguments".to_string(),
                    ));
                }
                let exp = args[1].as_f64();
                x.powf(exp)
            }
            Function::Sign => {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Function::Custom(name) => {
                return Err(EvalError::CannotEvaluate(format!(
                    "Unknown function: {}",
                    name
                )));
            }
        };

        if result.is_nan() {
            Ok(Value::NaN)
        } else if result.is_infinite() {
            if result > 0.0 {
                Ok(Value::PositiveInfinity)
            } else {
                Ok(Value::NegativeInfinity)
            }
        } else {
            Ok(Value::Float(result))
        }
    }

    /// Apply precision settings to a value.
    fn apply_precision(&self, value: Value) -> EvalResult<Value> {
        match &self.precision {
            PrecisionMode::Full => Ok(value),
            PrecisionMode::Arbitrary => Ok(value), // Already handled during evaluation
            PrecisionMode::FixedDecimal(places) => {
                let f = value.as_f64();
                if f.is_nan() || f.is_infinite() {
                    return Ok(value);
                }
                let rounded = round_to_decimal(f, *places, self.rounding);
                Ok(Value::Float(rounded))
            }
            PrecisionMode::SignificantFigures(figures) => {
                let f = value.as_f64();
                if f.is_nan() || f.is_infinite() || f == 0.0 {
                    return Ok(value);
                }
                let rounded = round_to_sig_figs(f, *figures, self.rounding);
                Ok(Value::Float(rounded))
            }
        }
    }
}

impl Default for EvalContext {
    fn default() -> Self {
        Self::full_precision()
    }
}

/// Round a number to a specific number of decimal places.
fn round_to_decimal(value: f64, places: u32, mode: RoundingMode) -> f64 {
    let factor = 10_f64.powi(places as i32);
    let scaled = value * factor;
    let rounded = apply_rounding(scaled, mode);
    rounded / factor
}

/// Round a number to a specific number of significant figures.
fn round_to_sig_figs(value: f64, figures: u32, mode: RoundingMode) -> f64 {
    if value == 0.0 {
        return 0.0;
    }
    let magnitude = value.abs().log10().floor() as i32;
    let scale = 10_f64.powi(figures as i32 - 1 - magnitude);
    let scaled = value * scale;
    let rounded = apply_rounding(scaled, mode);
    rounded / scale
}

/// Apply rounding mode to a value.
fn apply_rounding(value: f64, mode: RoundingMode) -> f64 {
    match mode {
        RoundingMode::HalfUp => {
            let floor = value.floor();
            if value - floor >= 0.5 {
                floor + 1.0
            } else {
                floor
            }
        }
        RoundingMode::HalfEven => {
            let floor = value.floor();
            let frac = value - floor;
            if frac > 0.5 || (frac == 0.5 && floor as i64 % 2 != 0) {
                floor + 1.0
            } else {
                floor
            }
        }
        RoundingMode::Truncate => value.trunc(),
        RoundingMode::Ceiling => value.ceil(),
        RoundingMode::Floor => value.floor(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Expression, Variable};

    fn div(a: Expression, b: Expression) -> Expression {
        Expression::Binary(BinaryOp::Div, Box::new(a), Box::new(b))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn sqrt(x: Expression) -> Expression {
        Expression::Function(Function::Sqrt, vec![x])
    }

    #[test]
    fn test_fixed_decimal_precision() {
        // 1/3 at 6 decimal places = 0.333333
        let ctx = EvalContext::fixed_decimal(6);
        let expr = div(int(1), int(3));
        let result = ctx.evaluate(&expr).unwrap();
        let f = result.as_f64();
        assert!((f - 0.333333).abs() < 1e-10);
    }

    #[test]
    fn test_significant_figures() {
        // 1/3 at 3 significant figures = 0.333
        let ctx = EvalContext::significant_figures(3);
        let expr = div(int(1), int(3));
        let result = ctx.evaluate(&expr).unwrap();
        let f = result.as_f64();
        assert!((f - 0.333).abs() < 1e-10);
    }

    #[test]
    fn test_arbitrary_precision_rational() {
        // 1/3 as exact rational
        let ctx = EvalContext::arbitrary();
        let expr = div(int(1), int(3));
        let result = ctx.evaluate(&expr).unwrap();
        match result {
            Value::Rational(r) => {
                assert_eq!(*r.numer(), 1);
                assert_eq!(*r.denom(), 3);
            }
            _ => panic!("Expected Rational"),
        }
    }

    #[test]
    fn test_sqrt_2_precision() {
        // sqrt(2) at various precisions
        let expr = sqrt(int(2));

        let ctx6 = EvalContext::fixed_decimal(6);
        let r6 = ctx6.evaluate(&expr).unwrap().as_f64();
        assert!((r6 - 1.414214).abs() < 1e-6);

        let ctx3 = EvalContext::significant_figures(3);
        let r3 = ctx3.evaluate(&expr).unwrap().as_f64();
        assert!((r3 - 1.41).abs() < 0.01);
    }

    #[test]
    fn test_complex_from_sqrt_negative() {
        // sqrt(-1) = i
        let ctx = EvalContext::full_precision().with_complex(true);
        let expr = sqrt(int(-1));
        let result = ctx.evaluate(&expr).unwrap();
        match result {
            Value::Complex(re, im) => {
                assert!(re.abs() < 1e-10);
                assert!((im - 1.0).abs() < 1e-10);
            }
            _ => panic!("Expected Complex"),
        }
    }

    #[test]
    fn test_complex_not_allowed() {
        // sqrt(-1) should error when complex not allowed
        let ctx = EvalContext::full_precision().with_complex(false);
        let expr = sqrt(int(-1));
        let result = ctx.evaluate(&expr);
        assert!(result.is_err());
    }

    #[test]
    fn test_overflow_handling() {
        // Very large computation should not panic
        let ctx = EvalContext::full_precision();
        let expr = Expression::Power(Box::new(int(10)), Box::new(int(1000)));
        let result = ctx.evaluate(&expr).unwrap();
        // Should be infinity
        match result {
            Value::PositiveInfinity => {}
            Value::Float(f) if f.is_infinite() => {}
            _ => panic!("Expected infinity"),
        }
    }

    #[test]
    fn test_division_by_zero() {
        let ctx = EvalContext::full_precision();
        let expr = div(int(1), int(0));
        let result = ctx.evaluate(&expr);
        assert!(matches!(result, Err(EvalError::DivisionByZero)));
    }

    #[test]
    fn test_variable_evaluation() {
        let mut ctx = EvalContext::full_precision();
        ctx.set_f64("x", 5.0);
        let expr = var("x");
        let result = ctx.evaluate(&expr).unwrap();
        assert!((result.as_f64() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_undefined_variable() {
        let ctx = EvalContext::full_precision();
        let expr = var("undefined");
        let result = ctx.evaluate(&expr);
        assert!(matches!(result, Err(EvalError::UndefinedVariable(_))));
    }

    #[test]
    fn test_rounding_modes() {
        // Test different rounding modes for 2.5
        let value = 2.5;

        assert_eq!(apply_rounding(value, RoundingMode::HalfUp), 3.0);
        assert_eq!(apply_rounding(value, RoundingMode::HalfEven), 2.0); // Banker's rounding
        assert_eq!(apply_rounding(value, RoundingMode::Truncate), 2.0);
        assert_eq!(apply_rounding(value, RoundingMode::Ceiling), 3.0);
        assert_eq!(apply_rounding(value, RoundingMode::Floor), 2.0);

        // Test 3.5 with banker's rounding (should round to 4)
        assert_eq!(apply_rounding(3.5, RoundingMode::HalfEven), 4.0);
    }

    #[test]
    fn test_complex_arithmetic() {
        let ctx = EvalContext::full_precision();

        // i * i = -1
        let i = Expression::Constant(SymbolicConstant::I);
        let i_squared = Expression::Binary(BinaryOp::Mul, Box::new(i.clone()), Box::new(i));
        let result = ctx.evaluate(&i_squared).unwrap();
        match result {
            Value::Complex(re, im) => {
                assert!((re - (-1.0)).abs() < 1e-10);
                assert!(im.abs() < 1e-10);
            }
            _ => panic!("Expected Complex"),
        }
    }

    #[test]
    fn test_infinity_handling() {
        let ctx = EvalContext::full_precision();

        // 1/0 should be division by zero error (not infinity)
        let expr = div(int(1), int(0));
        assert!(ctx.evaluate(&expr).is_err());

        // But very large numbers should produce infinity
        let expr = Expression::Power(Box::new(int(10)), Box::new(int(500)));
        let result = ctx.evaluate(&expr).unwrap();
        let is_positive_inf = matches!(result, Value::PositiveInfinity)
            || matches!(result, Value::Float(f) if f.is_infinite() && f > 0.0);
        assert!(is_positive_inf);
    }
}
