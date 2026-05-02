//! Expression simplification and algebraic identity rules.

use super::{BinaryOp, Expression, UnaryOp};
use crate::pattern::apply_rules_to_fixpoint;
use crate::simplification_rules::all_simplification_rules;
use std::collections::HashMap;

impl Expression {
    /// Apply basic algebraic simplifications.
    ///
    /// Recursively simplifies the expression tree using standard algebraic identities
    /// and constant folding. The simplification process works bottom-up: child nodes
    /// are simplified first, then algebraic rules are applied to the current node.
    ///
    /// # Simplification Strategy
    ///
    /// The method performs three types of simplification:
    /// 1. **Recursive simplification**: All subexpressions are simplified first
    /// 2. **Algebraic identities**: Common patterns are replaced with simpler forms
    /// 3. **Constant folding**: Numeric subexpressions are evaluated to single values
    ///
    /// # Algebraic Identity Rules
    ///
    /// ## Addition identities
    /// - `0 + x` → `x` (additive identity, left)
    /// - `x + 0` → `x` (additive identity, right)
    ///
    /// ## Subtraction identities
    /// - `x - 0` → `x` (subtracting zero)
    ///
    /// ## Multiplication identities
    /// - `0 * x` → `0` (annihilator, left)
    /// - `x * 0` → `0` (annihilator, right)
    /// - `1 * x` → `x` (multiplicative identity, left)
    /// - `x * 1` → `x` (multiplicative identity, right)
    ///
    /// ## Division identities
    /// - `x / 1` → `x` (dividing by one)
    ///
    /// ## Power identities
    /// - `x^0` → `1` (anything to power zero equals one, where x ≠ 0)
    /// - `x^1` → `x` (anything to power one equals itself)
    ///
    /// ## Negation identities
    /// - `-(-x)` → `x` (double negation elimination)
    ///
    /// # Constant Folding
    ///
    /// When both operands of a binary operation are numeric constants (Integer, Float,
    /// or Rational), the operation is evaluated immediately:
    ///
    /// - Arithmetic: `2 + 3` → `5`, `10 / 2` → `5`, `3 * 4` → `12`
    /// - Powers: `2^3` → `8`, `4^0.5` → `2.0`
    /// - Functions: `sin(0)` → `0`, `sqrt(16)` → `4.0`, `ln(1)` → `0`
    ///
    /// Division by zero is avoided (returns unsimplified expression).
    ///
    /// # Helper Methods
    ///
    /// The simplification process uses several private helper methods:
    ///
    /// - `is_zero`: Checks if expression equals zero (Integer 0 or Float 0.0)
    /// - `is_one`: Checks if expression equals one (Integer 1 or Float 1.0)
    /// - `is_numeric_constant`: True for Integer, Float, Rational
    /// - `extract_numeric_value`: Converts constant to f64
    /// - `from_numeric_value`: Creates Integer if whole, Float otherwise
    ///
    /// # Limitations
    ///
    /// This method performs only basic algebraic simplification. It does NOT:
    ///
    /// - Factor expressions (e.g., `x^2 - 1` is not factored to `(x-1)(x+1)`)
    /// - Combine like terms (e.g., `x + 2*x` remains `x + 2*x`, not `3*x`)
    /// - Expand products (e.g., `(x+1)(x-1)` remains unexpanded)
    /// - Apply trigonometric identities (e.g., `sin^2(x) + cos^2(x)` remains as is)
    /// - Rationalize denominators
    /// - Simplify complex fractions
    ///
    /// For symbolic equation solving, see the [`solver`](crate::solver) module which
    /// uses simplification as part of algebraic manipulation.
    ///
    /// # Examples
    ///
    /// ## Identity simplification
    ///
    /// ```
    /// use thales::ast::{Expression, Variable, BinaryOp};
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
    /// // x * 1 simplifies to x
    /// let expr2 = Expression::Binary(
    ///     BinaryOp::Mul,
    ///     Box::new(x.clone()),
    ///     Box::new(Expression::Integer(1))
    /// );
    /// assert_eq!(expr2.simplify(), x);
    ///
    /// // x^1 simplifies to x
    /// let expr3 = Expression::Power(
    ///     Box::new(x.clone()),
    ///     Box::new(Expression::Integer(1))
    /// );
    /// assert_eq!(expr3.simplify(), x);
    /// ```
    ///
    /// ## Constant folding
    ///
    /// ```
    /// use thales::ast::{Expression, BinaryOp, Function};
    ///
    /// // 2 + 3 simplifies to 5
    /// let expr = Expression::Binary(
    ///     BinaryOp::Add,
    ///     Box::new(Expression::Integer(2)),
    ///     Box::new(Expression::Integer(3))
    /// );
    /// assert_eq!(expr.simplify(), Expression::Integer(5));
    ///
    /// // 2^3 simplifies to 8
    /// let expr2 = Expression::Power(
    ///     Box::new(Expression::Integer(2)),
    ///     Box::new(Expression::Integer(3))
    /// );
    /// assert_eq!(expr2.simplify(), Expression::Integer(8));
    ///
    /// // sqrt(16) simplifies to 4
    /// let expr3 = Expression::Function(
    ///     Function::Sqrt,
    ///     vec![Expression::Integer(16)]
    /// );
    /// assert_eq!(expr3.simplify(), Expression::Integer(4));
    /// ```
    ///
    /// ## Double negation elimination
    ///
    /// ```
    /// use thales::ast::{Expression, Variable, UnaryOp};
    ///
    /// // -(-x) simplifies to x
    /// let x = Expression::Variable(Variable::new("x"));
    /// let neg_x = Expression::Unary(UnaryOp::Neg, Box::new(x.clone()));
    /// let neg_neg_x = Expression::Unary(UnaryOp::Neg, Box::new(neg_x));
    /// assert_eq!(neg_neg_x.simplify(), x);
    /// ```
    ///
    /// ## Recursive simplification
    ///
    /// ```
    /// use thales::ast::{Expression, Variable, BinaryOp};
    ///
    /// // (x + 0) * 1 simplifies to x
    /// let x = Expression::Variable(Variable::new("x"));
    /// let x_plus_0 = Expression::Binary(
    ///     BinaryOp::Add,
    ///     Box::new(x.clone()),
    ///     Box::new(Expression::Integer(0))
    /// );
    /// let expr = Expression::Binary(
    ///     BinaryOp::Mul,
    ///     Box::new(x_plus_0),
    ///     Box::new(Expression::Integer(1))
    /// );
    /// assert_eq!(expr.simplify(), x);
    /// ```
    ///
    /// ## Zero annihilation
    ///
    /// ```
    /// use thales::ast::{Expression, Variable, BinaryOp};
    ///
    /// // x * 0 simplifies to 0
    /// let x = Expression::Variable(Variable::new("x"));
    /// let expr = Expression::Binary(
    ///     BinaryOp::Mul,
    ///     Box::new(x),
    ///     Box::new(Expression::Integer(0))
    /// );
    /// assert_eq!(expr.simplify(), Expression::Integer(0));
    /// ```
    ///
    /// # Returns
    ///
    /// A new simplified expression. The original expression is unchanged. The result
    /// may be structurally identical if no simplifications apply.
    ///
    /// # See Also
    ///
    /// - [`evaluate`](Expression::evaluate) - Numerical evaluation with variable values
    /// - [`differentiate`](Expression::differentiate) - Symbolic differentiation (results benefit from simplification)
    /// - [`solver`](crate::solver) - Equation solving using simplification
    pub fn simplify(&self) -> Expression {
        let simplified = match self {
            Expression::Unary(op, expr) => Self::simplify_unary(*op, expr),
            Expression::Binary(op, left, right) => Self::simplify_binary(*op, left, right),
            Expression::Function(func, args) => Self::simplify_function(func, args),
            Expression::Power(base, exp) => Self::simplify_power(base, exp),
            _ => self.clone(),
        };

        // Apply pattern-matching simplification rules as a final pass
        let rules = all_simplification_rules();
        apply_rules_to_fixpoint(&simplified, &rules, 20)
    }

    /// Simplify a unary expression (double-negation elimination, recursive descent).
    fn simplify_unary(op: UnaryOp, expr: &Expression) -> Expression {
        let simplified_expr = expr.simplify();
        match op {
            UnaryOp::Neg => {
                // -(-x) → x
                if let Expression::Unary(UnaryOp::Neg, inner) = &simplified_expr {
                    inner.as_ref().clone()
                } else {
                    Expression::Unary(op, Box::new(simplified_expr))
                }
            }
            _ => Expression::Unary(op, Box::new(simplified_expr)),
        }
    }

    /// Simplify a binary expression: identity rules then constant folding.
    fn simplify_binary(op: BinaryOp, left: &Expression, right: &Expression) -> Expression {
        let left_s = left.simplify();
        let right_s = right.simplify();

        // Identity rules per operator
        match op {
            BinaryOp::Add => {
                if let Some(r) = Self::simplify_add(&left_s, &right_s) {
                    return r;
                }
            }
            BinaryOp::Sub => {
                if let Some(r) = Self::simplify_sub(&left_s, &right_s) {
                    return r;
                }
            }
            BinaryOp::Mul => {
                if let Some(r) = Self::simplify_mul(&left_s, &right_s) {
                    return r;
                }
            }
            BinaryOp::Div => {
                // x / 1 → x
                if Self::is_one(&right_s) {
                    return left_s;
                }
            }
            _ => {}
        }

        // Constant folding
        if Self::is_numeric_constant(&left_s) && Self::is_numeric_constant(&right_s) {
            if let (Some(lv), Some(rv)) = (
                Self::extract_numeric_value(&left_s),
                Self::extract_numeric_value(&right_s),
            ) {
                let result = match op {
                    BinaryOp::Add => Some(lv + rv),
                    BinaryOp::Sub => Some(lv - rv),
                    BinaryOp::Mul => Some(lv * rv),
                    BinaryOp::Div => {
                        if rv.abs() > 1e-10 {
                            Some(lv / rv)
                        } else {
                            None
                        }
                    }
                    BinaryOp::Mod => Some(lv % rv),
                };
                if let Some(value) = result {
                    return Self::from_numeric_value(value);
                }
            }
        }

        Expression::Binary(op, Box::new(left_s), Box::new(right_s))
    }

    /// Addition identity rules and like-terms collection.
    fn simplify_add(left_s: &Expression, right_s: &Expression) -> Option<Expression> {
        // 0 + x → x
        if Self::is_zero(left_s) {
            return Some(right_s.clone());
        }
        // x + 0 → x
        if Self::is_zero(right_s) {
            return Some(left_s.clone());
        }
        // Like terms: 2x + 3x → 5x
        let (coef1, base1) = Self::extract_coefficient_and_base(left_s);
        let (coef2, base2) = Self::extract_coefficient_and_base(right_s);
        if Self::bases_equal(&base1, &base2) && !Self::is_one(&base1) {
            let new_coef = coef1 + coef2;
            if new_coef.abs() < 1e-10 {
                return Some(Expression::Integer(0));
            }
            let coef_expr = Self::from_numeric_value(new_coef);
            if Self::is_one(&coef_expr) {
                return Some(base1);
            }
            return Some(Expression::Binary(
                BinaryOp::Mul,
                Box::new(coef_expr),
                Box::new(base1),
            ));
        }
        None
    }

    /// Subtraction identity rules and like-terms collection.
    fn simplify_sub(left_s: &Expression, right_s: &Expression) -> Option<Expression> {
        // x - 0 → x
        if Self::is_zero(right_s) {
            return Some(left_s.clone());
        }
        // x - x → 0
        if left_s == right_s {
            return Some(Expression::Integer(0));
        }
        // Like terms: 5x - 3x → 2x
        let (coef1, base1) = Self::extract_coefficient_and_base(left_s);
        let (coef2, base2) = Self::extract_coefficient_and_base(right_s);
        if Self::bases_equal(&base1, &base2) && !Self::is_one(&base1) {
            let new_coef = coef1 - coef2;
            if new_coef.abs() < 1e-10 {
                return Some(Expression::Integer(0));
            }
            let coef_expr = Self::from_numeric_value(new_coef);
            if Self::is_one(&coef_expr) {
                return Some(base1);
            }
            if new_coef < 0.0 {
                // Negative coefficient: return as -|coef| * base
                return Some(Expression::Unary(
                    UnaryOp::Neg,
                    Box::new(Expression::Binary(
                        BinaryOp::Mul,
                        Box::new(Self::from_numeric_value(-new_coef)),
                        Box::new(base1),
                    )),
                ));
            }
            return Some(Expression::Binary(
                BinaryOp::Mul,
                Box::new(coef_expr),
                Box::new(base1),
            ));
        }
        None
    }

    /// Multiplication identity rules and power-law combinations.
    fn simplify_mul(left_s: &Expression, right_s: &Expression) -> Option<Expression> {
        // 0 * x → 0
        if Self::is_zero(left_s) {
            return Some(Expression::Integer(0));
        }
        // x * 0 → 0
        if Self::is_zero(right_s) {
            return Some(Expression::Integer(0));
        }
        // 1 * x → x
        if Self::is_one(left_s) {
            return Some(right_s.clone());
        }
        // x * 1 → x
        if Self::is_one(right_s) {
            return Some(left_s.clone());
        }
        // x^a * x^b → x^(a+b)
        if let (Expression::Power(base1, exp1), Expression::Power(base2, exp2)) = (left_s, right_s)
        {
            if base1 == base2 {
                let new_exp =
                    Expression::Binary(BinaryOp::Add, exp1.clone(), exp2.clone()).simplify();
                return Some(Expression::Power(base1.clone(), Box::new(new_exp)));
            }
        }
        // x * x → x^2
        if left_s == right_s {
            return Some(Expression::Power(
                Box::new(left_s.clone()),
                Box::new(Expression::Integer(2)),
            ));
        }
        // x * x^n → x^(n+1)
        if let Expression::Power(base, exp) = right_s {
            if **base == *left_s {
                let new_exp = Expression::Binary(
                    BinaryOp::Add,
                    exp.clone(),
                    Box::new(Expression::Integer(1)),
                )
                .simplify();
                return Some(Expression::Power(base.clone(), Box::new(new_exp)));
            }
        }
        // x^n * x → x^(n+1)
        if let Expression::Power(base, exp) = left_s {
            if **base == *right_s {
                let new_exp = Expression::Binary(
                    BinaryOp::Add,
                    exp.clone(),
                    Box::new(Expression::Integer(1)),
                )
                .simplify();
                return Some(Expression::Power(base.clone(), Box::new(new_exp)));
            }
        }
        None
    }

    /// Simplify a function application: constant folding when all args are numeric.
    fn simplify_function(func: &super::Function, args: &[Expression]) -> Expression {
        let simplified_args: Vec<Expression> = args.iter().map(|a| a.simplify()).collect();

        if simplified_args.iter().all(Self::is_numeric_constant) {
            let temp_expr = Expression::Function(func.clone(), simplified_args.clone());
            if let Some(value) = temp_expr.evaluate(&HashMap::new()) {
                return Self::from_numeric_value(value);
            }
        }

        Expression::Function(func.clone(), simplified_args)
    }

    /// Simplify a power expression: identity rules, power-of-power, constant folding.
    fn simplify_power(base: &Expression, exp: &Expression) -> Expression {
        let base_s = base.simplify();
        let exp_s = exp.simplify();

        // x^0 → 1 (where x != 0)
        if Self::is_zero(&exp_s) && !Self::is_zero(&base_s) {
            return Expression::Integer(1);
        }
        // x^1 → x
        if Self::is_one(&exp_s) {
            return base_s;
        }
        // (x^a)^b → x^(a*b) — only safe when inner exponent is an integer;
        // for non-integer `a` the identity can give wrong results.
        if let Expression::Power(inner_base, inner_exp) = &base_s {
            if Self::is_integer_expr(inner_exp) {
                let new_exp =
                    Expression::Binary(BinaryOp::Mul, inner_exp.clone(), Box::new(exp_s.clone()))
                        .simplify();
                return Expression::Power(inner_base.clone(), Box::new(new_exp));
            }
        }

        // Constant folding
        if Self::is_numeric_constant(&base_s) && Self::is_numeric_constant(&exp_s) {
            if let (Some(bv), Some(ev)) = (
                Self::extract_numeric_value(&base_s),
                Self::extract_numeric_value(&exp_s),
            ) {
                let result = bv.powf(ev);
                if result.is_finite() {
                    return Self::from_numeric_value(result);
                }
            }
        }

        Expression::Power(Box::new(base_s), Box::new(exp_s))
    }

    /// Check if expression is zero.
    ///
    /// Returns `true` if the expression is exactly zero, either as an integer (0)
    /// or floating-point (0.0). Used by [`simplify`](Expression::simplify) to
    /// apply additive identity and annihilator rules.
    ///
    /// # Arguments
    ///
    /// * `expr` - Expression to test
    ///
    /// # Returns
    ///
    /// `true` if expression is `Integer(0)` or `Float(0.0)`, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Integer zero
    /// assert!(Expression::is_zero(&Expression::Integer(0)));
    ///
    /// // Float zero
    /// assert!(Expression::is_zero(&Expression::Float(0.0)));
    ///
    /// // Non-zero values
    /// assert!(!Expression::is_zero(&Expression::Integer(1)));
    /// assert!(!Expression::is_zero(&Expression::Float(0.001)));
    /// ```
    pub(crate) fn is_zero(expr: &Expression) -> bool {
        match expr {
            Expression::Integer(0) => true,
            Expression::Float(x) if *x == 0.0 => true,
            _ => false,
        }
    }

    /// Check if expression is one.
    ///
    /// Returns `true` if the expression is exactly one, either as an integer (1)
    /// or floating-point (1.0). Used by [`simplify`](Expression::simplify) to
    /// apply multiplicative identity and power identity rules.
    ///
    /// # Arguments
    ///
    /// * `expr` - Expression to test
    ///
    /// # Returns
    ///
    /// `true` if expression is `Integer(1)` or `Float(1.0)`, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Integer one
    /// assert!(Expression::is_one(&Expression::Integer(1)));
    ///
    /// // Float one
    /// assert!(Expression::is_one(&Expression::Float(1.0)));
    ///
    /// // Non-one values
    /// assert!(!Expression::is_one(&Expression::Integer(0)));
    /// assert!(!Expression::is_one(&Expression::Float(1.001)));
    /// ```
    pub(crate) fn is_one(expr: &Expression) -> bool {
        match expr {
            Expression::Integer(1) => true,
            Expression::Float(x) if *x == 1.0 => true,
            _ => false,
        }
    }

    /// Check if expression is a numeric constant.
    ///
    /// Returns `true` if the expression is any kind of numeric literal that can be
    /// evaluated without variable values: Integer, Float, or Rational. Used by
    /// [`simplify`](Expression::simplify) to identify constant folding opportunities.
    ///
    /// # Arguments
    ///
    /// * `expr` - Expression to test
    ///
    /// # Returns
    ///
    /// `true` for Integer, Float, or Rational variants, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Numeric constants
    /// assert!(Expression::is_numeric_constant(&Expression::Integer(42)));
    /// assert!(Expression::is_numeric_constant(&Expression::Float(3.14)));
    /// assert!(Expression::is_numeric_constant(&Expression::Rational(Rational64::new(1, 2))));
    ///
    /// // Non-constants
    /// assert!(!Expression::is_numeric_constant(&Expression::Variable(Variable::new("x"))));
    /// assert!(!Expression::is_numeric_constant(&Expression::Complex(Complex64::new(1.0, 0.0))));
    /// ```
    /// Check whether an expression is an integer value.
    ///
    /// Returns `true` for `Integer(_)`, negated integers, and `Float` values
    /// that are exact integers. Used to guard simplification rules that are only
    /// valid for integer exponents (e.g., the power-of-power rule).
    pub(crate) fn is_integer_expr(expr: &Expression) -> bool {
        match expr {
            Expression::Integer(_) => true,
            Expression::Unary(UnaryOp::Neg, inner) => Self::is_integer_expr(inner),
            Expression::Float(f) => f.is_finite() && *f == f.trunc() && *f != 0.0,
            _ => false,
        }
    }

    pub(crate) fn is_numeric_constant(expr: &Expression) -> bool {
        matches!(
            expr,
            Expression::Integer(_) | Expression::Float(_) | Expression::Rational(_)
        )
    }

    /// Extract numeric value from constant expression.
    ///
    /// Converts a numeric constant expression to f64 for evaluation. Works with
    /// Integer, Float, and Rational variants. Returns `None` for non-numeric
    /// expressions. Used by [`simplify`](Expression::simplify) for constant folding.
    ///
    /// # Arguments
    ///
    /// * `expr` - Expression to extract value from
    ///
    /// # Returns
    ///
    /// - `Some(f64)` for Integer, Float, or Rational expressions
    /// - `None` for all other expression types
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Extract integer
    /// assert_eq!(Expression::extract_numeric_value(&Expression::Integer(42)), Some(42.0));
    ///
    /// // Extract float
    /// assert_eq!(Expression::extract_numeric_value(&Expression::Float(3.14)), Some(3.14));
    ///
    /// // Extract rational (1/2 = 0.5)
    /// let half = Expression::Rational(Rational64::new(1, 2));
    /// assert_eq!(Expression::extract_numeric_value(&half), Some(0.5));
    ///
    /// // Non-numeric returns None
    /// let x = Expression::Variable(Variable::new("x"));
    /// assert_eq!(Expression::extract_numeric_value(&x), None);
    /// ```
    pub(crate) fn extract_numeric_value(expr: &Expression) -> Option<f64> {
        match expr {
            Expression::Integer(n) => Some(*n as f64),
            Expression::Float(x) => Some(*x),
            Expression::Rational(r) => Some(*r.numer() as f64 / *r.denom() as f64),
            _ => None,
        }
    }

    /// Create numeric expression from value (Integer if whole, Float otherwise).
    ///
    /// Converts a floating-point value to the most appropriate Expression variant.
    /// If the value is finite and very close to an integer (within 1e-10), creates
    /// an Integer expression. Otherwise creates a Float expression. Used by
    /// [`simplify`](Expression::simplify) after constant folding operations.
    ///
    /// # Arguments
    ///
    /// * `value` - Floating-point value to convert
    ///
    /// # Returns
    ///
    /// - `Integer(i64)` if value is finite and within 1e-10 of a whole number
    /// - `Float(f64)` otherwise
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Whole numbers become Integer
    /// assert_eq!(Expression::from_numeric_value(5.0), Expression::Integer(5));
    /// assert_eq!(Expression::from_numeric_value(5.0000000001), Expression::Integer(5));
    ///
    /// // Non-whole numbers become Float
    /// assert_eq!(Expression::from_numeric_value(3.14), Expression::Float(3.14));
    /// assert_eq!(Expression::from_numeric_value(5.1), Expression::Float(5.1));
    ///
    /// // Special values become Float
    /// assert_eq!(Expression::from_numeric_value(f64::INFINITY), Expression::Float(f64::INFINITY));
    /// ```
    pub(crate) fn from_numeric_value(value: f64) -> Expression {
        if value.is_finite() && value.fract().abs() < 1e-10 {
            Expression::Integer(value.round() as i64)
        } else {
            Expression::Float(value)
        }
    }

    /// Extract coefficient and base from a term for like-terms collection.
    ///
    /// Decomposes expressions into (coefficient, base) pairs:
    /// - `c * x` → `(c, x)` where c is numeric
    /// - `x * c` → `(c, x)` where c is numeric
    /// - `x` → `(1, x)` for any non-numeric expression
    /// - `-x` → `(-1, x)` for negated expressions
    /// - `c` → `(c, 1)` for pure numeric constants
    ///
    /// Used by `simplify()` to combine like terms (e.g., 2x + 3x → 5x).
    fn extract_coefficient_and_base(expr: &Expression) -> (f64, Expression) {
        match expr {
            // Pure numeric constant: coefficient with base 1
            Expression::Integer(n) => (*n as f64, Expression::Integer(1)),
            Expression::Float(x) => (*x, Expression::Integer(1)),
            Expression::Rational(r) => (
                *r.numer() as f64 / *r.denom() as f64,
                Expression::Integer(1),
            ),
            // Negation: extract inner and negate coefficient
            Expression::Unary(UnaryOp::Neg, inner) => {
                let (coef, base) = Self::extract_coefficient_and_base(inner);
                (-coef, base)
            }
            // Multiplication: check if one side is numeric
            Expression::Binary(BinaryOp::Mul, left, right) => {
                if let Some(coef) = Self::extract_numeric_value(left) {
                    // c * expr
                    let (inner_coef, base) = Self::extract_coefficient_and_base(right);
                    (coef * inner_coef, base)
                } else if let Some(coef) = Self::extract_numeric_value(right) {
                    // expr * c
                    let (inner_coef, base) = Self::extract_coefficient_and_base(left);
                    (coef * inner_coef, base)
                } else {
                    // Neither side is numeric, treat whole expr as base
                    (1.0, expr.clone())
                }
            }
            // Division by constant: treat as multiplication by 1/c
            Expression::Binary(BinaryOp::Div, left, right) => {
                if let Some(divisor) = Self::extract_numeric_value(right) {
                    if divisor.abs() > 1e-10 {
                        let (coef, base) = Self::extract_coefficient_and_base(left);
                        (coef / divisor, base)
                    } else {
                        (1.0, expr.clone())
                    }
                } else {
                    (1.0, expr.clone())
                }
            }
            // Any other expression: coefficient is 1
            _ => (1.0, expr.clone()),
        }
    }

    /// Check if two expressions are structurally equal as bases for like-terms.
    ///
    /// Used by `simplify()` to determine if two terms can be combined.
    fn bases_equal(base1: &Expression, base2: &Expression) -> bool {
        base1 == base2
    }
}
