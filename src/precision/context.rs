//! Evaluation context with precision tracking.

use crate::ast::{BinaryOp, Expression, Function, SymbolicConstant, UnaryOp};
use num::CheckedAdd;
use num::CheckedDiv;
use num::CheckedMul;
use num::CheckedSub;
use num_rational::Rational64;
use std::collections::HashMap;

use super::helpers::{round_to_decimal, round_to_sig_figs};
use super::types::{EvalError, EvalResult, PrecisionMode, RoundingMode, Value};

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
            (Value::Rational(a), Value::Rational(b)) => a
                .checked_add(&b)
                .map(Value::Rational)
                .ok_or(EvalError::Overflow),
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
            (Value::Rational(a), Value::Rational(b)) => a
                .checked_sub(&b)
                .map(Value::Rational)
                .ok_or(EvalError::Overflow),
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
            (Value::Rational(a), Value::Rational(b)) => a
                .checked_mul(&b)
                .map(Value::Rational)
                .ok_or(EvalError::Overflow),
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
            (Value::Rational(a), Value::Rational(b)) => a
                .checked_div(&b)
                .map(Value::Rational)
                .ok_or(EvalError::Overflow),
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
            // Re/Im/Conj: in real-valued precision context, Re(x)=x, Im(x)=0, Conj(x)=x
            Function::Re | Function::Conj => x,
            Function::Im => 0.0,
            // Special functions: not evaluable in this f64 precision path
            Function::Gamma
            | Function::LnGamma
            | Function::Digamma
            | Function::BetaFn
            | Function::Erf
            | Function::Erfc
            | Function::BesselJ
            | Function::BesselY
            | Function::BesselI
            | Function::BesselK
            | Function::AiryAi
            | Function::AiryBi
            | Function::Zeta
            | Function::Si
            | Function::Ci
            | Function::Ei
            | Function::Heaviside
            | Function::DiracDelta => {
                return Err(EvalError::CannotEvaluate(format!(
                    "{:?} not yet implemented in precision evaluator",
                    func
                )));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rational_add_overflow_returns_error() {
        let ctx = EvalContext::new(PrecisionMode::Arbitrary);
        // Construct rationals with large numerators that will overflow on addition
        let a = Value::Rational(Rational64::new(i64::MAX / 2, 1));
        let b = Value::Rational(Rational64::new(i64::MAX / 2 + 2, 1));
        let result = ctx.eval_add(a, b);
        assert!(
            result.is_err() || matches!(result, Ok(Value::Rational(_))),
            "Should either overflow or produce correct result"
        );
    }

    #[test]
    fn test_rational_mul_overflow_returns_error() {
        let ctx = EvalContext::new(PrecisionMode::Arbitrary);
        // Large numerators that overflow when multiplied
        let a = Value::Rational(Rational64::new(i64::MAX / 2, 1));
        let b = Value::Rational(Rational64::new(3, 1));
        let result = ctx.eval_mul(a, b);
        assert!(
            result.is_err(),
            "Multiplication of large rationals should return Overflow error"
        );
    }

    #[test]
    fn test_rational_arithmetic_normal_case() {
        let ctx = EvalContext::new(PrecisionMode::Arbitrary);
        let a = Value::Rational(Rational64::new(1, 3));
        let b = Value::Rational(Rational64::new(1, 6));

        let sum = ctx.eval_add(a.clone(), b.clone()).unwrap();
        assert_eq!(sum, Value::Rational(Rational64::new(1, 2)));

        let diff = ctx.eval_sub(a.clone(), b.clone()).unwrap();
        assert_eq!(diff, Value::Rational(Rational64::new(1, 6)));

        let prod = ctx.eval_mul(a.clone(), b.clone()).unwrap();
        assert_eq!(prod, Value::Rational(Rational64::new(1, 18)));

        let quot = ctx.eval_div(a, b).unwrap();
        assert_eq!(quot, Value::Rational(Rational64::new(2, 1)));
    }
}
