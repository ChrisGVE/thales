//! Compiler from the legacy `ast::Expression` tree to the CAS `numeric::Expr`.
//!
//! [`compile`] performs a structural translation, mapping each `Expression`
//! variant to its `Expr` equivalent and using the smart constructors in
//! [`normalize`] so the output is already in canonical form.
//!
//! [`decompile`] performs the inverse translation, mapping each `Expr` variant
//! back to its `Expression` equivalent. Because `compile` normalizes expressions,
//! the round-trip is semantically equivalent but may differ structurally.

use crate::ast::{BinaryOp, Expression, Function, UnaryOp, Variable};
use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::normalize;
use crate::numeric::{AddNode, BigRational, MulNode, SmallInt, SymbolId};
use num::traits::{One, Zero};
use num_rational::Rational64;
use std::sync::Arc;

// ── Public API ────────────────────────────────────────────────────────────────

/// Compile a legacy [`Expression`] into a normalized [`Expr`].
///
/// The output is always wrapped in `Arc<Expr>` and is in canonical form
/// thanks to the normalizing smart constructors used internally.
///
/// # Examples
///
/// ```rust
/// use thales::ast::{Expression, Variable, BinaryOp};
/// use thales::numeric::compile::compile;
/// use thales::numeric::expr::Expr;
///
/// let x = Expression::Variable(Variable::new("x"));
/// let two = Expression::Integer(2);
/// let sum = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(two));
///
/// let result = compile(&sum);
/// // result is a normalized Expr representing x + 2
/// ```
pub fn compile(expr: &Expression) -> Arc<Expr> {
    match expr {
        // ── Numeric literals ─────────────────────────────────────────────────
        Expression::Integer(n) => Arc::new(Expr::Integer(SmallInt::from(*n))),

        Expression::Rational(r) => {
            let numer = *r.numer();
            let denom = *r.denom();
            Arc::new(Expr::Rational(BigRational::from_i64(numer, denom)))
        }

        Expression::Float(f) => Arc::new(Expr::Float(*f)),

        Expression::Complex(c) => Arc::new(Expr::Complex(*c)),

        // ── Symbolic constant ────────────────────────────────────────────────
        Expression::Constant(c) => Arc::new(Expr::Constant(*c)),

        // ── Variable ─────────────────────────────────────────────────────────
        Expression::Variable(v) => Arc::new(Expr::Symbol(SymbolId::intern(&v.name))),

        // ── Unary operations ─────────────────────────────────────────────────
        Expression::Unary(op, inner) => {
            let compiled = compile(inner);
            match op {
                UnaryOp::Neg => normalize::neg(compiled),
                UnaryOp::Abs => Expr::func(FuncId::Abs, vec![compiled]),
                UnaryOp::Not => Expr::func(FuncId::Other(SymbolId::intern("not")), vec![compiled]),
            }
        }

        // ── Binary operations ────────────────────────────────────────────────
        Expression::Binary(op, lhs, rhs) => {
            let l = compile(lhs);
            let r = compile(rhs);
            match op {
                BinaryOp::Add => normalize::add(l, r),
                BinaryOp::Sub => normalize::sub(l, r),
                BinaryOp::Mul => normalize::mul(l, r),
                BinaryOp::Div => normalize::div(l, r),
                BinaryOp::Mod => Expr::func(FuncId::Other(SymbolId::intern("mod")), vec![l, r]),
            }
        }

        // ── Power ────────────────────────────────────────────────────────────
        Expression::Power(base, exp) => normalize::pow(compile(base), compile(exp)),

        // ── Function application ─────────────────────────────────────────────
        Expression::Function(f, args) => {
            let compiled_args: Vec<Arc<Expr>> = args.iter().map(compile).collect();
            let fid = map_func_id(f);
            // Function::Pow(x, y) is expressed as normalize::pow to keep
            // the canonical Pow node form rather than a Func application.
            if let FuncId::Other(sym) = fid {
                if sym == SymbolId::intern("__pow__") {
                    if compiled_args.len() == 2 {
                        return normalize::pow(compiled_args[0].clone(), compiled_args[1].clone());
                    }
                }
                Expr::func(FuncId::Other(sym), compiled_args)
            } else {
                Expr::func(fid, compiled_args)
            }
        }
    }
}

// ── Function ID mapping ───────────────────────────────────────────────────────

/// Map an `ast::Function` variant to the corresponding [`FuncId`].
///
/// [`Function::Pow`] is mapped to `FuncId::Other("__pow__")` so the caller
/// can detect it and use [`normalize::pow`] instead of a `Func` node.
pub fn map_func_id(f: &Function) -> FuncId {
    match f {
        Function::Sin => FuncId::Sin,
        Function::Cos => FuncId::Cos,
        Function::Tan => FuncId::Tan,
        Function::Asin => FuncId::Asin,
        Function::Acos => FuncId::Acos,
        Function::Atan => FuncId::Atan,
        Function::Atan2 => FuncId::Atan2,
        Function::Sinh => FuncId::Sinh,
        Function::Cosh => FuncId::Cosh,
        Function::Tanh => FuncId::Tanh,
        Function::Exp => FuncId::Exp,
        Function::Ln => FuncId::Ln,
        Function::Log => FuncId::Log,
        Function::Log2 => FuncId::Log2,
        Function::Log10 => FuncId::Log10,
        Function::Sqrt => FuncId::Sqrt,
        Function::Cbrt => FuncId::Cbrt,
        Function::Floor => FuncId::Floor,
        Function::Ceil => FuncId::Ceil,
        Function::Round => FuncId::Round,
        Function::Abs => FuncId::Abs,
        Function::Sign => FuncId::Sign,
        Function::Min => FuncId::Min,
        Function::Max => FuncId::Max,
        Function::Re => FuncId::Re,
        Function::Im => FuncId::Im,
        Function::Conj => FuncId::Conj,
        // Pow → sentinel so compile() can route to normalize::pow
        Function::Pow => FuncId::Other(SymbolId::intern("__pow__")),
        Function::Custom(name) => FuncId::Other(SymbolId::intern(name)),
        // Non-exhaustive guard: map any future variants to Other.
        #[allow(unreachable_patterns)]
        _ => FuncId::Other(SymbolId::intern("__unknown__")),
    }
}

// ── Decompiler ────────────────────────────────────────────────────────────────

/// Decompile a [`Expr`] back into a legacy [`Expression`].
///
/// This is the inverse of [`compile`]. It maps each `Expr` variant back to
/// its `Expression` equivalent. Because `compile` normalizes expressions (e.g.
/// folding constants, combining like terms), the round-trip is not guaranteed to
/// be structurally identical, but will be semantically equivalent.
///
/// # Overflow handling
///
/// - `Integer`: if the [`SmallInt`] value exceeds `i64` range, falls back to 0.
/// - `Rational`: if numerator or denominator exceeds `i64` range, falls back to
///   `Expression::Float` using the rational's `to_f64()` approximation.
/// - `Rational` with denominator 1 is returned as `Expression::Integer`.
///
/// # Examples
///
/// ```rust
/// use thales::ast::{Expression, Variable, BinaryOp};
/// use thales::numeric::compile::{compile, decompile};
///
/// let x = Expression::Variable(Variable::new("x"));
/// let two = Expression::Integer(2);
/// let sum = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(two));
///
/// let compiled = compile(&sum);
/// let decompiled = decompile(&compiled);
/// // decompiled is semantically equivalent to sum
/// ```
pub fn decompile(expr: &Expr) -> Expression {
    match expr {
        // ── Numeric literals ─────────────────────────────────────────────────
        Expr::Integer(n) => match n.to_i64() {
            Some(i) => Expression::Integer(i),
            None => Expression::Float(n.to_string().parse::<f64>().unwrap_or(f64::INFINITY)),
        },

        Expr::Rational(r) => big_rational_to_expr(r),

        Expr::Float(f) => Expression::Float(*f),

        Expr::Complex(c) => Expression::Complex(*c),

        // ── Symbolic constant ────────────────────────────────────────────────
        Expr::Constant(c) => Expression::Constant(*c),

        // ── Variable ─────────────────────────────────────────────────────────
        Expr::Symbol(s) => Expression::Variable(Variable::new(s.as_str())),

        // ── AddNode ──────────────────────────────────────────────────────────
        Expr::Add(node) => decompile_add_node(node),

        // ── MulNode ──────────────────────────────────────────────────────────
        Expr::Mul(node) => decompile_mul_node(node),

        // ── Power ────────────────────────────────────────────────────────────
        Expr::Pow(base, exp) => {
            Expression::Power(Box::new(decompile(base)), Box::new(decompile(exp)))
        }

        // ── Function ─────────────────────────────────────────────────────────
        Expr::Func(id, args) => {
            let f = reverse_map_func_id(id);
            let decompiled_args: Vec<Expression> = args.iter().map(|a| decompile(a)).collect();
            Expression::Function(f, decompiled_args)
        }
    }
}

/// Reverse-map a [`FuncId`] back to the corresponding [`Function`] variant.
///
/// [`FuncId::Other`] with name `"__pow__"` maps back to [`Function::Pow`].
/// Any other [`FuncId::Other`] maps to [`Function::Custom`].
pub fn reverse_map_func_id(id: &FuncId) -> Function {
    match id {
        FuncId::Sin => Function::Sin,
        FuncId::Cos => Function::Cos,
        FuncId::Tan => Function::Tan,
        FuncId::Asin => Function::Asin,
        FuncId::Acos => Function::Acos,
        FuncId::Atan => Function::Atan,
        FuncId::Atan2 => Function::Atan2,
        FuncId::Sinh => Function::Sinh,
        FuncId::Cosh => Function::Cosh,
        FuncId::Tanh => Function::Tanh,
        FuncId::Exp => Function::Exp,
        FuncId::Ln => Function::Ln,
        FuncId::Log => Function::Log,
        FuncId::Log2 => Function::Log2,
        FuncId::Log10 => Function::Log10,
        FuncId::Sqrt => Function::Sqrt,
        FuncId::Cbrt => Function::Cbrt,
        FuncId::Floor => Function::Floor,
        FuncId::Ceil => Function::Ceil,
        FuncId::Round => Function::Round,
        FuncId::Abs => Function::Abs,
        FuncId::Sign => Function::Sign,
        FuncId::Min => Function::Min,
        FuncId::Max => Function::Max,
        FuncId::Re => Function::Re,
        FuncId::Im => Function::Im,
        FuncId::Conj => Function::Conj,
        FuncId::Other(sym) => {
            let name = sym.as_str();
            if name == "__pow__" {
                Function::Pow
            } else {
                Function::Custom(name)
            }
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Convert a [`BigRational`] to an [`Expression`] literal.
///
/// Produces `Integer` for whole-number rationals (denom == 1), `Rational`
/// when numerator and denominator both fit in `i64`, and `Float` as a last
/// resort for values that overflow `i64`.
fn big_rational_to_expr(r: &BigRational) -> Expression {
    if r.is_integer() {
        match r.numer().to_i64() {
            Some(i) => Expression::Integer(i),
            None => Expression::Float(r.to_f64()),
        }
    } else {
        match (r.numer().to_i64(), r.denom().to_i64()) {
            (Some(n), Some(d)) => Expression::Rational(Rational64::new(n, d)),
            _ => Expression::Float(r.to_f64()),
        }
    }
}

/// Reconstruct an [`Expression`] from an [`AddNode`].
///
/// Layout: `constant + Σ(coeff_i · term_i)`.
/// Zero-coefficient terms are excluded by invariant. The accumulator starts
/// with the constant (if non-zero) or the first term, and additional terms
/// are folded in as `Binary(Add, …)` / `Binary(Sub, …)` depending on sign.
fn decompile_add_node(node: &AddNode) -> Expression {
    let mut acc: Option<Expression> = None;

    // Start with the constant part if it is non-zero.
    if !node.constant.is_zero() {
        acc = Some(big_rational_to_expr(&node.constant));
    }

    for (term, coeff) in &node.terms {
        let term_expr = decompile(term);

        // Build `coeff * term`, or just `term` / `-term` for ±1.
        let scaled = if coeff.is_one() {
            term_expr
        } else if *coeff == BigRational::from(-1i64) {
            Expression::Unary(UnaryOp::Neg, Box::new(term_expr))
        } else {
            let coeff_expr = big_rational_to_expr(coeff);
            Expression::Binary(BinaryOp::Mul, Box::new(coeff_expr), Box::new(term_expr))
        };

        acc = Some(match acc {
            None => scaled,
            Some(prev) => {
                // When `scaled` is a negation we can emit `prev - inner`
                // instead of `prev + (-inner)` for readability.
                if let Expression::Unary(UnaryOp::Neg, inner) = scaled {
                    Expression::Binary(BinaryOp::Sub, Box::new(prev), inner)
                } else {
                    Expression::Binary(BinaryOp::Add, Box::new(prev), Box::new(scaled))
                }
            }
        });
    }

    // Empty AddNode (all terms cancelled, constant also zero) → 0.
    acc.unwrap_or(Expression::Integer(0))
}

/// Reconstruct an [`Expression`] from a [`MulNode`].
///
/// Layout: `coeff · Π(base_i ^ exp_i)`.
/// The coefficient is prepended when it is not 1. Special-case: coeff == -1
/// wraps the result in `Unary(Neg, …)`. Factors with exp == 1 contribute the
/// base directly; exp == -1 produces `Binary(Div, acc, base)`; otherwise a
/// `Power(base, exp)` node is inserted.
fn decompile_mul_node(node: &MulNode) -> Expression {
    let neg_one = BigRational::from(-1i64);

    let mut acc: Option<Expression> = None;

    // Fold in the coefficient, but handle -1 specially (applied at the end).
    let coeff_is_neg_one = node.coeff == neg_one;
    if !node.coeff.is_one() && !coeff_is_neg_one {
        acc = Some(big_rational_to_expr(&node.coeff));
    }

    // Partition factors by sign of integer exponent so that positive-exponent
    // factors enter the accumulator first (building the numerator) and
    // negative-exponent factors emit as `Div` against the already-built
    // accumulator (building `num / denom`). A factor whose exponent is not
    // a negative integer is treated as positive.
    let mut positive_factors: Vec<(&Arc<Expr>, &Arc<Expr>)> = Vec::new();
    let mut negative_factors: Vec<(&Arc<Expr>, i64)> = Vec::new();
    for (base, exp) in &node.factors {
        if let Expr::Integer(n) = exp.as_ref() {
            if let Some(ni) = n.to_i64() {
                if ni < 0 {
                    negative_factors.push((base, ni));
                    continue;
                }
            }
        }
        positive_factors.push((base, exp));
    }

    for (base, exp) in positive_factors {
        let base_expr = decompile(base);
        let factor = if exp.is_one() {
            base_expr
        } else {
            Expression::Power(Box::new(base_expr), Box::new(decompile(exp)))
        };
        acc = Some(match acc {
            None => factor,
            Some(prev) => Expression::Binary(BinaryOp::Mul, Box::new(prev), Box::new(factor)),
        });
    }

    for (base, ni) in negative_factors {
        let base_expr = decompile(base);
        let denom = if ni == -1 {
            base_expr
        } else {
            Expression::Power(Box::new(base_expr), Box::new(Expression::Integer(-ni)))
        };
        let current = acc.take().unwrap_or(Expression::Integer(1));
        acc = Some(Expression::Binary(
            BinaryOp::Div,
            Box::new(current),
            Box::new(denom),
        ));
    }

    let result = acc.unwrap_or(Expression::Integer(1));

    // Apply the -1 coefficient as a wrapping negation.
    if coeff_is_neg_one {
        Expression::Unary(UnaryOp::Neg, Box::new(result))
    } else {
        result
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Expression, Function, SymbolicConstant, UnaryOp, Variable};
    use crate::numeric::expr::{Expr, FuncId};
    use num_complex::Complex64;
    use num_rational::Rational64;

    // ── Literals ─────────────────────────────────────────────────────────────

    #[test]
    fn compile_integer() {
        let e = Expression::Integer(42);
        let result = compile(&e);
        assert_eq!(*result, Expr::Integer(SmallInt::from(42i64)));
    }

    #[test]
    fn compile_integer_negative() {
        let e = Expression::Integer(-7);
        let result = compile(&e);
        assert_eq!(*result, Expr::Integer(SmallInt::from(-7i64)));
    }

    #[test]
    fn compile_rational() {
        let e = Expression::Rational(Rational64::new(1, 2));
        let result = compile(&e);
        assert_eq!(*result, Expr::Rational(BigRational::from_i64(1, 2)));
    }

    #[test]
    fn compile_float() {
        let e = Expression::Float(3.14);
        let result = compile(&e);
        match result.as_ref() {
            Expr::Float(f) => assert_eq!(*f, 3.14),
            _ => panic!("expected Float"),
        }
    }

    #[test]
    fn compile_complex() {
        let c = Complex64::new(1.0, -2.0);
        let e = Expression::Complex(c);
        let result = compile(&e);
        match result.as_ref() {
            Expr::Complex(v) => {
                assert_eq!(v.re, 1.0);
                assert_eq!(v.im, -2.0);
            }
            _ => panic!("expected Complex"),
        }
    }

    // ── Constants ────────────────────────────────────────────────────────────

    #[test]
    fn compile_constant_pi() {
        let e = Expression::Constant(SymbolicConstant::Pi);
        let result = compile(&e);
        assert_eq!(*result, Expr::Constant(SymbolicConstant::Pi));
    }

    #[test]
    fn compile_constant_e() {
        let e = Expression::Constant(SymbolicConstant::E);
        let result = compile(&e);
        assert_eq!(*result, Expr::Constant(SymbolicConstant::E));
    }

    #[test]
    fn compile_constant_i() {
        let e = Expression::Constant(SymbolicConstant::I);
        let result = compile(&e);
        assert_eq!(*result, Expr::Constant(SymbolicConstant::I));
    }

    // ── Variable ─────────────────────────────────────────────────────────────

    #[test]
    fn compile_variable() {
        let e = Expression::Variable(Variable::new("x"));
        let result = compile(&e);
        assert_eq!(*result, Expr::Symbol(SymbolId::intern("x")));
    }

    #[test]
    fn compile_variable_with_dimension() {
        let e = Expression::Variable(Variable::with_dimension("v", "m/s"));
        let result = compile(&e);
        // Dimension info is not carried into Expr; only the name matters
        assert_eq!(*result, Expr::Symbol(SymbolId::intern("v")));
    }

    // ── Unary ops ────────────────────────────────────────────────────────────

    #[test]
    fn compile_unary_neg_integer() {
        let e = Expression::Unary(UnaryOp::Neg, Box::new(Expression::Integer(5)));
        let result = compile(&e);
        // normalize::neg(5) → Integer(-5)
        assert_eq!(*result, Expr::Integer(SmallInt::from(-5i64)));
    }

    #[test]
    fn compile_unary_neg_variable() {
        let e = Expression::Unary(
            UnaryOp::Neg,
            Box::new(Expression::Variable(Variable::new("y"))),
        );
        let result = compile(&e);
        // normalize::neg(y) → Mul{ coeff=-1, y^1 }
        match result.as_ref() {
            Expr::Mul(_) => {} // canonical -1*y
            _ => panic!("expected Mul for negated variable, got: {result}"),
        }
    }

    #[test]
    fn compile_unary_abs() {
        let e = Expression::Unary(UnaryOp::Abs, Box::new(Expression::Integer(3)));
        let result = compile(&e);
        match result.as_ref() {
            Expr::Func(FuncId::Abs, args) => assert_eq!(args.len(), 1),
            _ => panic!("expected Func(Abs, ...)"),
        }
    }

    #[test]
    fn compile_unary_not() {
        let e = Expression::Unary(UnaryOp::Not, Box::new(Expression::Integer(1)));
        let result = compile(&e);
        match result.as_ref() {
            Expr::Func(FuncId::Other(sym), args) => {
                assert_eq!(sym, &SymbolId::intern("not"));
                assert_eq!(args.len(), 1);
            }
            _ => panic!("expected Func(Other(not), ...)"),
        }
    }

    // ── Binary ops ───────────────────────────────────────────────────────────

    #[test]
    fn compile_binary_add_constants() {
        // 3 + 4 → Integer(7)
        let e = Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Integer(3)),
            Box::new(Expression::Integer(4)),
        );
        let result = compile(&e);
        assert_eq!(*result, Expr::Integer(SmallInt::from(7i64)));
    }

    #[test]
    fn compile_binary_sub_constants() {
        // 10 - 3 → Integer(7)
        let e = Expression::Binary(
            BinaryOp::Sub,
            Box::new(Expression::Integer(10)),
            Box::new(Expression::Integer(3)),
        );
        let result = compile(&e);
        assert_eq!(*result, Expr::Integer(SmallInt::from(7i64)));
    }

    #[test]
    fn compile_binary_mul_constants() {
        // 3 * 4 → Integer(12)
        let e = Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Integer(3)),
            Box::new(Expression::Integer(4)),
        );
        let result = compile(&e);
        assert_eq!(*result, Expr::Integer(SmallInt::from(12i64)));
    }

    #[test]
    fn compile_binary_div_constants() {
        // 6 / 2 → Integer(3) or Rational(3/1)
        let e = Expression::Binary(
            BinaryOp::Div,
            Box::new(Expression::Integer(6)),
            Box::new(Expression::Integer(2)),
        );
        let result = compile(&e);
        // normalize::div(6, 2) → 3
        match result.as_ref() {
            Expr::Integer(n) => assert_eq!(n, &SmallInt::from(3i64)),
            Expr::Rational(r) => assert_eq!(*r, BigRational::from_i64(3, 1)),
            _ => panic!("expected numeric result for 6/2, got: {result}"),
        }
    }

    #[test]
    fn compile_binary_div_symbolic() {
        // x / y — must not panic; shape is Pow or Mul
        let e = Expression::Binary(
            BinaryOp::Div,
            Box::new(Expression::Variable(Variable::new("xa"))),
            Box::new(Expression::Variable(Variable::new("ya"))),
        );
        let result = compile(&e);
        // Just check it compiles without panic
        let _ = result.to_string();
    }

    #[test]
    fn compile_binary_mod() {
        let e = Expression::Binary(
            BinaryOp::Mod,
            Box::new(Expression::Variable(Variable::new("xm"))),
            Box::new(Expression::Variable(Variable::new("ym"))),
        );
        let result = compile(&e);
        match result.as_ref() {
            Expr::Func(FuncId::Other(sym), args) => {
                assert_eq!(sym, &SymbolId::intern("mod"));
                assert_eq!(args.len(), 2);
            }
            _ => panic!("expected Func(Other(mod), ...)"),
        }
    }

    #[test]
    fn compile_binary_add_symbolic() {
        // x + x → 2*x  (normalization combines like terms)
        let x1 = Expression::Variable(Variable::new("cx"));
        let x2 = Expression::Variable(Variable::new("cx"));
        let e = Expression::Binary(BinaryOp::Add, Box::new(x1), Box::new(x2));
        let result = compile(&e);
        // The result should represent 2*x
        let s = result.to_string();
        assert!(s.contains("cx"), "expected x in: {s}");
    }

    // ── Power ─────────────────────────────────────────────────────────────────

    #[test]
    fn compile_power_integer() {
        // 2^3 → Integer(8) after evaluation
        let e = Expression::Power(
            Box::new(Expression::Integer(2)),
            Box::new(Expression::Integer(3)),
        );
        let result = compile(&e);
        assert_eq!(*result, Expr::Integer(SmallInt::from(8i64)));
    }

    #[test]
    fn compile_power_symbolic() {
        // x^2 → Pow(Symbol(x), Integer(2))
        let e = Expression::Power(
            Box::new(Expression::Variable(Variable::new("xp"))),
            Box::new(Expression::Integer(2)),
        );
        let result = compile(&e);
        match result.as_ref() {
            Expr::Pow(base, exp) => {
                assert_eq!(**base, Expr::Symbol(SymbolId::intern("xp")));
                assert_eq!(**exp, Expr::Integer(SmallInt::from(2i64)));
            }
            Expr::Mul(_) => {
                // normalize may return Mul for single-factor with exp
                let s = result.to_string();
                assert!(s.contains("xp"), "expected xp in: {s}");
            }
            _ => panic!("expected Pow or Mul for x^2, got: {result}"),
        }
    }

    // ── Functions ─────────────────────────────────────────────────────────────

    #[test]
    fn compile_function_sin() {
        let e = Expression::Function(
            Function::Sin,
            vec![Expression::Variable(Variable::new("xs"))],
        );
        let result = compile(&e);
        match result.as_ref() {
            Expr::Func(FuncId::Sin, args) => assert_eq!(args.len(), 1),
            _ => panic!("expected Func(Sin, ...), got: {result}"),
        }
    }

    #[test]
    fn compile_function_cos() {
        let e = Expression::Function(
            Function::Cos,
            vec![Expression::Variable(Variable::new("xc"))],
        );
        let result = compile(&e);
        match result.as_ref() {
            Expr::Func(FuncId::Cos, _) => {}
            _ => panic!("expected Func(Cos, ...)"),
        }
    }

    #[test]
    fn compile_function_exp() {
        let e = Expression::Function(Function::Exp, vec![Expression::Integer(1)]);
        let result = compile(&e);
        match result.as_ref() {
            Expr::Func(FuncId::Exp, _) => {}
            _ => panic!("expected Func(Exp, ...)"),
        }
    }

    #[test]
    fn compile_function_ln() {
        let e = Expression::Function(
            Function::Ln,
            vec![Expression::Variable(Variable::new("xl"))],
        );
        let result = compile(&e);
        match result.as_ref() {
            Expr::Func(FuncId::Ln, _) => {}
            _ => panic!("expected Func(Ln, ...)"),
        }
    }

    #[test]
    fn compile_function_sqrt() {
        let e = Expression::Function(Function::Sqrt, vec![Expression::Integer(4)]);
        let result = compile(&e);
        match result.as_ref() {
            Expr::Func(FuncId::Sqrt, _) => {}
            _ => panic!("expected Func(Sqrt, ...)"),
        }
    }

    #[test]
    fn compile_function_abs() {
        let e = Expression::Function(
            Function::Abs,
            vec![Expression::Variable(Variable::new("xabs"))],
        );
        let result = compile(&e);
        match result.as_ref() {
            Expr::Func(FuncId::Abs, _) => {}
            _ => panic!("expected Func(Abs, ...)"),
        }
    }

    #[test]
    fn compile_function_atan2() {
        let e = Expression::Function(
            Function::Atan2,
            vec![
                Expression::Variable(Variable::new("yat")),
                Expression::Variable(Variable::new("xat")),
            ],
        );
        let result = compile(&e);
        match result.as_ref() {
            Expr::Func(FuncId::Atan2, args) => assert_eq!(args.len(), 2),
            _ => panic!("expected Func(Atan2, ...)"),
        }
    }

    #[test]
    fn compile_function_min_max() {
        let a = Expression::Integer(2);
        let b = Expression::Integer(5);
        let min_e = Expression::Function(Function::Min, vec![a.clone(), b.clone()]);
        let max_e = Expression::Function(Function::Max, vec![a, b]);
        match compile(&min_e).as_ref() {
            Expr::Func(FuncId::Min, _) => {}
            _ => panic!("expected Func(Min, ...)"),
        }
        match compile(&max_e).as_ref() {
            Expr::Func(FuncId::Max, _) => {}
            _ => panic!("expected Func(Max, ...)"),
        }
    }

    #[test]
    fn compile_function_pow_routes_to_pow_node() {
        // ast::Function::Pow(x, 2) should produce Expr::Pow, not Expr::Func
        let e = Expression::Function(
            Function::Pow,
            vec![
                Expression::Variable(Variable::new("xfp")),
                Expression::Integer(2),
            ],
        );
        let result = compile(&e);
        // normalize::pow(x, 2) → Pow(x, 2) or Mul depending on normalizer
        match result.as_ref() {
            Expr::Pow(_, _) | Expr::Mul(_) => {}
            _ => panic!("expected Pow or Mul for Function::Pow, got: {result}"),
        }
    }

    #[test]
    fn compile_function_custom() {
        let e = Expression::Function(
            Function::Custom("my_fn".to_string()),
            vec![Expression::Integer(1)],
        );
        let result = compile(&e);
        match result.as_ref() {
            Expr::Func(FuncId::Other(sym), _) => {
                assert_eq!(sym, &SymbolId::intern("my_fn"));
            }
            _ => panic!("expected Func(Other(my_fn), ...)"),
        }
    }

    // ── map_func_id exhaustive check ──────────────────────────────────────────

    #[test]
    fn map_func_id_all_direct_variants() {
        assert_eq!(map_func_id(&Function::Sin), FuncId::Sin);
        assert_eq!(map_func_id(&Function::Cos), FuncId::Cos);
        assert_eq!(map_func_id(&Function::Tan), FuncId::Tan);
        assert_eq!(map_func_id(&Function::Asin), FuncId::Asin);
        assert_eq!(map_func_id(&Function::Acos), FuncId::Acos);
        assert_eq!(map_func_id(&Function::Atan), FuncId::Atan);
        assert_eq!(map_func_id(&Function::Atan2), FuncId::Atan2);
        assert_eq!(map_func_id(&Function::Sinh), FuncId::Sinh);
        assert_eq!(map_func_id(&Function::Cosh), FuncId::Cosh);
        assert_eq!(map_func_id(&Function::Tanh), FuncId::Tanh);
        assert_eq!(map_func_id(&Function::Exp), FuncId::Exp);
        assert_eq!(map_func_id(&Function::Ln), FuncId::Ln);
        assert_eq!(map_func_id(&Function::Log), FuncId::Log);
        assert_eq!(map_func_id(&Function::Log2), FuncId::Log2);
        assert_eq!(map_func_id(&Function::Log10), FuncId::Log10);
        assert_eq!(map_func_id(&Function::Sqrt), FuncId::Sqrt);
        assert_eq!(map_func_id(&Function::Cbrt), FuncId::Cbrt);
        assert_eq!(map_func_id(&Function::Floor), FuncId::Floor);
        assert_eq!(map_func_id(&Function::Ceil), FuncId::Ceil);
        assert_eq!(map_func_id(&Function::Round), FuncId::Round);
        assert_eq!(map_func_id(&Function::Abs), FuncId::Abs);
        assert_eq!(map_func_id(&Function::Sign), FuncId::Sign);
        assert_eq!(map_func_id(&Function::Min), FuncId::Min);
        assert_eq!(map_func_id(&Function::Max), FuncId::Max);
    }

    #[test]
    fn map_func_id_custom() {
        assert_eq!(
            map_func_id(&Function::Custom("foo".to_string())),
            FuncId::Other(SymbolId::intern("foo")),
        );
    }

    #[test]
    fn map_func_id_pow_sentinel() {
        assert_eq!(
            map_func_id(&Function::Pow),
            FuncId::Other(SymbolId::intern("__pow__")),
        );
    }

    // ── Nested expression ─────────────────────────────────────────────────────

    #[test]
    fn compile_nested_sin_of_sum() {
        // sin(x + 1)
        let x = Expression::Variable(Variable::new("xn"));
        let one = Expression::Integer(1);
        let sum = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(one));
        let e = Expression::Function(Function::Sin, vec![sum]);
        let result = compile(&e);
        match result.as_ref() {
            Expr::Func(FuncId::Sin, args) => {
                assert_eq!(args.len(), 1);
                let s = args[0].to_string();
                assert!(s.contains("xn"), "expected xn in: {s}");
            }
            _ => panic!("expected Func(Sin, ...), got: {result}"),
        }
    }

    #[test]
    fn compile_zero_annihilation() {
        // 0 * x → Integer(0)
        let e = Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Integer(0)),
            Box::new(Expression::Variable(Variable::new("xz"))),
        );
        let result = compile(&e);
        assert_eq!(*result, Expr::Integer(SmallInt::from(0i64)));
    }

    #[test]
    fn compile_identity_removal() {
        // 1 * x → Symbol(x)
        let e = Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Integer(1)),
            Box::new(Expression::Variable(Variable::new("xi"))),
        );
        let result = compile(&e);
        assert_eq!(*result, Expr::Symbol(SymbolId::intern("xi")));
    }

    // ── decompile: basic literals ─────────────────────────────────────────────

    #[test]
    fn decompile_integer() {
        let expr = Expr::Integer(SmallInt::from(7i64));
        assert_eq!(decompile(&expr), Expression::Integer(7));
    }

    #[test]
    fn decompile_integer_negative() {
        let expr = Expr::Integer(SmallInt::from(-42i64));
        assert_eq!(decompile(&expr), Expression::Integer(-42));
    }

    #[test]
    fn decompile_rational() {
        let expr = Expr::Rational(BigRational::from_i64(3, 4));
        assert_eq!(
            decompile(&expr),
            Expression::Rational(Rational64::new(3, 4))
        );
    }

    #[test]
    fn decompile_rational_integer_valued() {
        // 6/3 normalizes to 2/1 inside BigRational; decompile should return Integer(2)
        let expr = Expr::Rational(BigRational::from_i64(6, 3));
        assert_eq!(decompile(&expr), Expression::Integer(2));
    }

    #[test]
    fn decompile_float() {
        let expr = Expr::Float(2.718);
        assert_eq!(decompile(&expr), Expression::Float(2.718));
    }

    #[test]
    fn decompile_complex() {
        let c = Complex64::new(1.0, -2.0);
        let expr = Expr::Complex(c);
        assert_eq!(decompile(&expr), Expression::Complex(c));
    }

    #[test]
    fn decompile_constant_pi() {
        let expr = Expr::Constant(SymbolicConstant::Pi);
        assert_eq!(decompile(&expr), Expression::Constant(SymbolicConstant::Pi));
    }

    #[test]
    fn decompile_constant_e() {
        let expr = Expr::Constant(SymbolicConstant::E);
        assert_eq!(decompile(&expr), Expression::Constant(SymbolicConstant::E));
    }

    #[test]
    fn decompile_symbol() {
        let expr = Expr::Symbol(SymbolId::intern("alpha"));
        assert_eq!(
            decompile(&expr),
            Expression::Variable(Variable::new("alpha"))
        );
    }

    // ── decompile: Pow ────────────────────────────────────────────────────────

    #[test]
    fn decompile_pow() {
        let base = Arc::new(Expr::Symbol(SymbolId::intern("xpow")));
        let exp = Arc::new(Expr::Integer(SmallInt::from(3i64)));
        let expr = Expr::Pow(base, exp);
        match decompile(&expr) {
            Expression::Power(b, e) => {
                assert_eq!(*b, Expression::Variable(Variable::new("xpow")));
                assert_eq!(*e, Expression::Integer(3));
            }
            other => panic!("expected Power, got {other:?}"),
        }
    }

    // ── decompile: Func ───────────────────────────────────────────────────────

    #[test]
    fn decompile_func_sin() {
        let arg = Arc::new(Expr::Symbol(SymbolId::intern("xsin")));
        let expr = Expr::Func(FuncId::Sin, vec![arg]);
        match decompile(&expr) {
            Expression::Function(Function::Sin, args) => assert_eq!(args.len(), 1),
            other => panic!("expected Function(Sin, ...), got {other:?}"),
        }
    }

    #[test]
    fn decompile_func_ln() {
        let arg = Arc::new(Expr::Symbol(SymbolId::intern("xln")));
        let expr = Expr::Func(FuncId::Ln, vec![arg]);
        match decompile(&expr) {
            Expression::Function(Function::Ln, _) => {}
            other => panic!("expected Function(Ln, ...), got {other:?}"),
        }
    }

    #[test]
    fn decompile_func_abs() {
        let arg = Arc::new(Expr::Integer(SmallInt::from(-5i64)));
        let expr = Expr::Func(FuncId::Abs, vec![arg]);
        match decompile(&expr) {
            Expression::Function(Function::Abs, args) => assert_eq!(args.len(), 1),
            other => panic!("expected Function(Abs, ...), got {other:?}"),
        }
    }

    #[test]
    fn decompile_func_custom() {
        let arg = Arc::new(Expr::Integer(SmallInt::from(1i64)));
        let expr = Expr::Func(FuncId::Other(SymbolId::intern("my_fn")), vec![arg]);
        match decompile(&expr) {
            Expression::Function(Function::Custom(name), _) => {
                assert_eq!(name, "my_fn");
            }
            other => panic!("expected Function(Custom(my_fn), ...), got {other:?}"),
        }
    }

    // ── decompile: AddNode ────────────────────────────────────────────────────

    #[test]
    fn decompile_add_constant_only() {
        let node = AddNode::from_constant(BigRational::from(5i64));
        let expr = Expr::Add(node);
        assert_eq!(decompile(&expr), Expression::Integer(5));
    }

    #[test]
    fn decompile_add_single_term_coeff_one() {
        // AddNode: 0 + 1*x → x
        let x = Arc::new(Expr::Symbol(SymbolId::intern("xadd1")));
        let node = AddNode::from_term(x, BigRational::from(1i64));
        let expr = Expr::Add(node);
        match decompile(&expr) {
            Expression::Variable(v) => assert_eq!(v.name, "xadd1"),
            other => panic!("expected Variable, got {other:?}"),
        }
    }

    #[test]
    fn decompile_add_single_term_coeff_neg_one() {
        // AddNode: 0 + (-1)*x → -x (Unary Neg)
        let x = Arc::new(Expr::Symbol(SymbolId::intern("xaddneg")));
        let node = AddNode::from_term(x, BigRational::from(-1i64));
        let expr = Expr::Add(node);
        match decompile(&expr) {
            Expression::Unary(UnaryOp::Neg, _) => {}
            other => panic!("expected Unary(Neg, ...), got {other:?}"),
        }
    }

    #[test]
    fn decompile_add_two_terms() {
        // AddNode: x + y — result contains both variable names
        let x = Arc::new(Expr::Symbol(SymbolId::intern("xadd2a")));
        let y = Arc::new(Expr::Symbol(SymbolId::intern("xadd2b")));
        let mut node = AddNode::from_term(x, BigRational::from(1i64));
        node.add_term(y, BigRational::from(1i64));
        let expr = Expr::Add(node);
        let result = decompile(&expr);
        let s = format!("{result:?}");
        assert!(s.contains("xadd2a"), "missing xadd2a in {s}");
        assert!(s.contains("xadd2b"), "missing xadd2b in {s}");
    }

    #[test]
    fn decompile_add_with_negative_coeff_gives_sub() {
        // AddNode: 3 + (-1)*x → Sub(3, x)
        let x = Arc::new(Expr::Symbol(SymbolId::intern("xaddsub")));
        let mut node = AddNode::from_constant(BigRational::from(3i64));
        node.add_term(x, BigRational::from(-1i64));
        let expr = Expr::Add(node);
        match decompile(&expr) {
            Expression::Binary(BinaryOp::Sub, lhs, _) => {
                assert_eq!(*lhs, Expression::Integer(3));
            }
            other => panic!("expected Binary(Sub, ...), got {other:?}"),
        }
    }

    #[test]
    fn decompile_add_with_coeff_two() {
        // AddNode: 2*x → Binary(Mul, 2, x)
        let x = Arc::new(Expr::Symbol(SymbolId::intern("xaddmul")));
        let node = AddNode::from_term(x, BigRational::from(2i64));
        let expr = Expr::Add(node);
        match decompile(&expr) {
            Expression::Binary(BinaryOp::Mul, coeff, var) => {
                assert_eq!(*coeff, Expression::Integer(2));
                assert_eq!(*var, Expression::Variable(Variable::new("xaddmul")));
            }
            other => panic!("expected Binary(Mul, 2, x), got {other:?}"),
        }
    }

    #[test]
    fn decompile_add_empty_node_is_zero() {
        let expr = Expr::Add(AddNode::zero());
        assert_eq!(decompile(&expr), Expression::Integer(0));
    }

    // ── decompile: MulNode ────────────────────────────────────────────────────

    #[test]
    fn decompile_mul_coeff_only() {
        // MulNode with only coeff 7 (no factors) → Integer(7)
        let node = MulNode::from_coeff(BigRational::from(7i64));
        let expr = Expr::Mul(node);
        assert_eq!(decompile(&expr), Expression::Integer(7));
    }

    #[test]
    fn decompile_mul_single_factor_exp_one() {
        // MulNode: 1 * x^1 → x
        let x = Arc::new(Expr::Symbol(SymbolId::intern("xmul1")));
        let node = MulNode::from_factor(x, Arc::new(Expr::Integer(SmallInt::from(1i64))));
        let expr = Expr::Mul(node);
        match decompile(&expr) {
            Expression::Variable(v) => assert_eq!(v.name, "xmul1"),
            other => panic!("expected Variable, got {other:?}"),
        }
    }

    #[test]
    fn decompile_mul_single_factor_exp_neg_one_gives_div() {
        // MulNode: 1 * x^(-1) → Div(1, x)
        let x = Arc::new(Expr::Symbol(SymbolId::intern("xdiv")));
        let node = MulNode::from_factor(x, Arc::new(Expr::Integer(SmallInt::from(-1i64))));
        let expr = Expr::Mul(node);
        match decompile(&expr) {
            Expression::Binary(BinaryOp::Div, lhs, rhs) => {
                assert_eq!(*lhs, Expression::Integer(1));
                assert_eq!(*rhs, Expression::Variable(Variable::new("xdiv")));
            }
            other => panic!("expected Binary(Div, 1, x), got {other:?}"),
        }
    }

    #[test]
    fn decompile_mul_neg_one_coeff_gives_neg() {
        // MulNode: -1 * x → Unary(Neg, x)
        let x = Arc::new(Expr::Symbol(SymbolId::intern("xmulneg")));
        let node = MulNode::from_coeff_and_base(BigRational::from(-1i64), x);
        let expr = Expr::Mul(node);
        match decompile(&expr) {
            Expression::Unary(UnaryOp::Neg, _) => {}
            other => panic!("expected Unary(Neg, ...), got {other:?}"),
        }
    }

    #[test]
    fn decompile_mul_two_factors() {
        // MulNode: x * y
        let x = Arc::new(Expr::Symbol(SymbolId::intern("xmul2a")));
        let y = Arc::new(Expr::Symbol(SymbolId::intern("xmul2b")));
        let mut node = MulNode::from_factor(x, Arc::new(Expr::Integer(SmallInt::from(1i64))));
        node.add_factor(y, Arc::new(Expr::Integer(SmallInt::from(1i64))));
        let expr = Expr::Mul(node);
        let result = decompile(&expr);
        let s = format!("{result:?}");
        assert!(s.contains("xmul2a"), "missing xmul2a in {s}");
        assert!(s.contains("xmul2b"), "missing xmul2b in {s}");
    }

    #[test]
    fn decompile_mul_factor_with_exp_two_gives_power() {
        // MulNode: x^2 → Power(x, 2)
        let x = Arc::new(Expr::Symbol(SymbolId::intern("xpow2")));
        let node = MulNode::from_factor(x, Arc::new(Expr::Integer(SmallInt::from(2i64))));
        let expr = Expr::Mul(node);
        match decompile(&expr) {
            Expression::Power(base, exp) => {
                assert_eq!(*base, Expression::Variable(Variable::new("xpow2")));
                assert_eq!(*exp, Expression::Integer(2));
            }
            other => panic!("expected Power, got {other:?}"),
        }
    }

    #[test]
    fn decompile_mul_empty_node_is_one() {
        let expr = Expr::Mul(MulNode::one());
        assert_eq!(decompile(&expr), Expression::Integer(1));
    }

    // ── decompile: reverse_map_func_id ───────────────────────────────────────

    #[test]
    fn reverse_map_func_id_all_direct() {
        assert_eq!(reverse_map_func_id(&FuncId::Sin), Function::Sin);
        assert_eq!(reverse_map_func_id(&FuncId::Cos), Function::Cos);
        assert_eq!(reverse_map_func_id(&FuncId::Tan), Function::Tan);
        assert_eq!(reverse_map_func_id(&FuncId::Asin), Function::Asin);
        assert_eq!(reverse_map_func_id(&FuncId::Acos), Function::Acos);
        assert_eq!(reverse_map_func_id(&FuncId::Atan), Function::Atan);
        assert_eq!(reverse_map_func_id(&FuncId::Atan2), Function::Atan2);
        assert_eq!(reverse_map_func_id(&FuncId::Sinh), Function::Sinh);
        assert_eq!(reverse_map_func_id(&FuncId::Cosh), Function::Cosh);
        assert_eq!(reverse_map_func_id(&FuncId::Tanh), Function::Tanh);
        assert_eq!(reverse_map_func_id(&FuncId::Exp), Function::Exp);
        assert_eq!(reverse_map_func_id(&FuncId::Ln), Function::Ln);
        assert_eq!(reverse_map_func_id(&FuncId::Log), Function::Log);
        assert_eq!(reverse_map_func_id(&FuncId::Log2), Function::Log2);
        assert_eq!(reverse_map_func_id(&FuncId::Log10), Function::Log10);
        assert_eq!(reverse_map_func_id(&FuncId::Sqrt), Function::Sqrt);
        assert_eq!(reverse_map_func_id(&FuncId::Cbrt), Function::Cbrt);
        assert_eq!(reverse_map_func_id(&FuncId::Floor), Function::Floor);
        assert_eq!(reverse_map_func_id(&FuncId::Ceil), Function::Ceil);
        assert_eq!(reverse_map_func_id(&FuncId::Round), Function::Round);
        assert_eq!(reverse_map_func_id(&FuncId::Abs), Function::Abs);
        assert_eq!(reverse_map_func_id(&FuncId::Sign), Function::Sign);
        assert_eq!(reverse_map_func_id(&FuncId::Min), Function::Min);
        assert_eq!(reverse_map_func_id(&FuncId::Max), Function::Max);
    }

    #[test]
    fn reverse_map_func_id_pow_sentinel() {
        assert_eq!(
            reverse_map_func_id(&FuncId::Other(SymbolId::intern("__pow__"))),
            Function::Pow
        );
    }

    #[test]
    fn reverse_map_func_id_custom() {
        assert_eq!(
            reverse_map_func_id(&FuncId::Other(SymbolId::intern("foo_bar"))),
            Function::Custom("foo_bar".to_string())
        );
    }

    // ── round-trip tests (compile → decompile) ────────────────────────────────

    fn eval(e: &Expression) -> Option<f64> {
        use std::collections::HashMap;
        e.evaluate(&HashMap::new())
    }

    #[test]
    fn roundtrip_integer() {
        let original = Expression::Integer(13);
        let compiled = compile(&original);
        let back = decompile(&compiled);
        assert_eq!(eval(&back), eval(&original));
    }

    #[test]
    fn roundtrip_rational() {
        let original = Expression::Rational(Rational64::new(1, 4));
        let compiled = compile(&original);
        let back = decompile(&compiled);
        let orig_val = eval(&original).unwrap();
        let back_val = eval(&back).unwrap();
        assert!(
            (orig_val - back_val).abs() < 1e-14,
            "{orig_val} vs {back_val}"
        );
    }

    #[test]
    fn roundtrip_float() {
        let original = Expression::Float(1.5);
        let compiled = compile(&original);
        let back = decompile(&compiled);
        assert_eq!(eval(&back), eval(&original));
    }

    #[test]
    fn roundtrip_constant_pi() {
        let original = Expression::Constant(SymbolicConstant::Pi);
        let compiled = compile(&original);
        let back = decompile(&compiled);
        assert_eq!(eval(&back), eval(&original));
    }

    #[test]
    fn roundtrip_power_two_cubed() {
        // 2^3 normalizes to Integer(8) in compile; decompile gives Integer(8)
        let original = Expression::Power(
            Box::new(Expression::Integer(2)),
            Box::new(Expression::Integer(3)),
        );
        let compiled = compile(&original);
        let back = decompile(&compiled);
        assert_eq!(eval(&back), Some(8.0));
    }

    #[test]
    fn roundtrip_add_two_constants() {
        // 3 + 4 = 7
        let original = Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Integer(3)),
            Box::new(Expression::Integer(4)),
        );
        let compiled = compile(&original);
        let back = decompile(&compiled);
        assert_eq!(eval(&back), Some(7.0));
    }

    #[test]
    fn roundtrip_mul_two_constants() {
        // 3 * 4 = 12
        let original = Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Integer(3)),
            Box::new(Expression::Integer(4)),
        );
        let compiled = compile(&original);
        let back = decompile(&compiled);
        assert_eq!(eval(&back), Some(12.0));
    }

    #[test]
    fn roundtrip_sin_of_constant() {
        // sin(0) = 0
        let original = Expression::Function(Function::Sin, vec![Expression::Integer(0)]);
        let compiled = compile(&original);
        let back = decompile(&compiled);
        let v = eval(&back).unwrap();
        assert!(v.abs() < 1e-14, "sin(0) should be 0, got {v}");
    }

    #[test]
    fn roundtrip_nested_expr() {
        // (2 + 3) * 4 = 20
        let sum = Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Integer(2)),
            Box::new(Expression::Integer(3)),
        );
        let original = Expression::Binary(
            BinaryOp::Mul,
            Box::new(sum),
            Box::new(Expression::Integer(4)),
        );
        let compiled = compile(&original);
        let back = decompile(&compiled);
        assert_eq!(eval(&back), Some(20.0));
    }

    // ── compile_then_eval_matches ─────────────────────────────────────────────

    /// Evaluate `expr` with both evaluators using identical variable bindings and
    /// assert the results agree within `epsilon`.
    fn assert_eval_match(expr: &Expression, str_bindings: &[(&str, f64)], epsilon: f64) {
        use crate::numeric::evaluation::evaluate as eval_new;
        use std::collections::HashMap;

        // Build string-keyed map for old evaluator.
        let mut old_map: HashMap<String, f64> = HashMap::new();
        // Build SymbolId-keyed map for new evaluator.
        let mut new_map: HashMap<SymbolId, f64> = HashMap::new();
        for (name, val) in str_bindings {
            old_map.insert(name.to_string(), *val);
            new_map.insert(SymbolId::intern(name), *val);
        }

        let old_val = expr
            .evaluate(&old_map)
            .expect("old evaluator returned None");
        let compiled = compile(expr);
        let new_val = eval_new(&compiled, &new_map).expect("new evaluator returned None");

        assert!(
            (old_val - new_val).abs() < epsilon,
            "compile_then_eval mismatch: old={old_val} new={new_val} (expr={expr:?})"
        );
    }

    #[test]
    fn compile_then_eval_simple_sum() {
        // x + 2 with x=3 → 5.0
        let x = Expression::Variable(Variable::new("rt_x"));
        let expr = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(Expression::Integer(2)));
        assert_eval_match(&expr, &[("rt_x", 3.0)], 1e-10);
    }

    #[test]
    fn compile_then_eval_polynomial() {
        // x^2 - 3*x + 2 with x=5 → 12.0
        let x = Expression::Variable(Variable::new("rt_p"));
        let x_sq = Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(2)));
        let three_x = Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Integer(3)),
            Box::new(x.clone()),
        );
        let minus_three_x = Expression::Binary(BinaryOp::Sub, Box::new(x_sq), Box::new(three_x));
        let expr = Expression::Binary(
            BinaryOp::Add,
            Box::new(minus_three_x),
            Box::new(Expression::Integer(2)),
        );
        assert_eval_match(&expr, &[("rt_p", 5.0)], 1e-10);
    }

    #[test]
    fn compile_then_eval_transcendental() {
        // sin(x) + cos(x) with x = π/4
        let x = Expression::Variable(Variable::new("rt_t"));
        let sin_x = Expression::Function(Function::Sin, vec![x.clone()]);
        let cos_x = Expression::Function(Function::Cos, vec![x]);
        let expr = Expression::Binary(BinaryOp::Add, Box::new(sin_x), Box::new(cos_x));
        assert_eval_match(&expr, &[("rt_t", std::f64::consts::FRAC_PI_4)], 1e-10);
    }

    #[test]
    fn compile_then_eval_ln_exp_nested() {
        // ln(exp(x)) with x=2 → 2.0
        let x = Expression::Variable(Variable::new("rt_le"));
        let exp_x = Expression::Function(Function::Exp, vec![x]);
        let expr = Expression::Function(Function::Ln, vec![exp_x]);
        assert_eval_match(&expr, &[("rt_le", 2.0)], 1e-10);
    }

    #[test]
    fn compile_then_eval_division() {
        // (x + 1) / (x - 1) with x=3 → 2.0
        let x = Expression::Variable(Variable::new("rt_div"));
        let numer = Expression::Binary(
            BinaryOp::Add,
            Box::new(x.clone()),
            Box::new(Expression::Integer(1)),
        );
        let denom =
            Expression::Binary(BinaryOp::Sub, Box::new(x), Box::new(Expression::Integer(1)));
        let expr = Expression::Binary(BinaryOp::Div, Box::new(numer), Box::new(denom));
        assert_eval_match(&expr, &[("rt_div", 3.0)], 1e-10);
    }

    #[test]
    fn compile_then_eval_multi_variable() {
        // 2*x + 3*y - z with x=1, y=2, z=3 → 4.0
        let x = Expression::Variable(Variable::new("rt_mx"));
        let y = Expression::Variable(Variable::new("rt_my"));
        let z = Expression::Variable(Variable::new("rt_mz"));
        let two_x =
            Expression::Binary(BinaryOp::Mul, Box::new(Expression::Integer(2)), Box::new(x));
        let three_y =
            Expression::Binary(BinaryOp::Mul, Box::new(Expression::Integer(3)), Box::new(y));
        let sum = Expression::Binary(BinaryOp::Add, Box::new(two_x), Box::new(three_y));
        let expr = Expression::Binary(BinaryOp::Sub, Box::new(sum), Box::new(z));
        assert_eval_match(
            &expr,
            &[("rt_mx", 1.0), ("rt_my", 2.0), ("rt_mz", 3.0)],
            1e-10,
        );
    }

    // ── decompile_then_eval_matches ───────────────────────────────────────────

    /// Build an Expr, decompile it to Expression, evaluate both with the same
    /// bindings and assert agreement within epsilon.
    fn assert_decompile_eval_match(
        new_expr: &crate::numeric::expr::Expr,
        str_bindings: &[(&str, f64)],
        epsilon: f64,
    ) {
        use crate::numeric::evaluation::evaluate as eval_new;
        use std::collections::HashMap;

        let mut old_map: HashMap<String, f64> = HashMap::new();
        let mut new_map: HashMap<SymbolId, f64> = HashMap::new();
        for (name, val) in str_bindings {
            old_map.insert(name.to_string(), *val);
            new_map.insert(SymbolId::intern(name), *val);
        }

        let new_val = eval_new(new_expr, &new_map).expect("new evaluator returned None");
        let old_expr = decompile(new_expr);
        let old_val = old_expr
            .evaluate(&old_map)
            .expect("old evaluator returned None");

        assert!(
            (old_val - new_val).abs() < epsilon,
            "decompile_then_eval mismatch: new={new_val} old={old_val}"
        );
    }

    #[test]
    fn decompile_then_eval_add_node_multiple_terms() {
        // AddNode: 1 + 2*x + 3*y, with x=4, y=5 → 1 + 8 + 15 = 24
        let x = Expr::symbol("rt_ax");
        let y = Expr::symbol("rt_ay");
        let mut node = AddNode::from_constant(BigRational::from(1i64));
        node.add_term(x, BigRational::from(2i64));
        node.add_term(y, BigRational::from(3i64));
        let expr = Expr::Add(node);
        assert_decompile_eval_match(&expr, &[("rt_ax", 4.0), ("rt_ay", 5.0)], 1e-10);
    }

    #[test]
    fn decompile_then_eval_mul_node_multiple_factors() {
        // MulNode: 2 * x^2 * y^1, with x=3, y=4 → 2 * 9 * 4 = 72
        let x = Expr::symbol("rt_mx2");
        let y = Expr::symbol("rt_my2");
        let mut node = MulNode::from_coeff(BigRational::from(2i64));
        node.add_factor(x, Expr::int(2));
        node.add_factor(y, Expr::int(1));
        let expr = Expr::Mul(node);
        assert_decompile_eval_match(&expr, &[("rt_mx2", 3.0), ("rt_my2", 4.0)], 1e-10);
    }

    #[test]
    fn decompile_then_eval_pow_and_func() {
        // sqrt(x^2) with x=5 → 5.0
        let x = Expr::symbol("rt_pf");
        let pow_expr = Arc::new(Expr::Pow(x, Expr::int(2)));
        let expr = Expr::Func(crate::numeric::expr::FuncId::Sqrt, vec![pow_expr]);
        assert_decompile_eval_match(&expr, &[("rt_pf", 5.0)], 1e-10);
    }

    // ── round_trip_compile_decompile ─────────────────────────────────────────

    /// compile(expr) → decompile → evaluate old; compare with evaluate old on
    /// original. Tests that the full forward trip preserves semantics.
    fn assert_compile_decompile_match(
        expr: &Expression,
        str_bindings: &[(&str, f64)],
        epsilon: f64,
    ) {
        use std::collections::HashMap;
        let mut old_map: HashMap<String, f64> = HashMap::new();
        for (name, val) in str_bindings {
            old_map.insert(name.to_string(), *val);
        }
        let orig_val = expr
            .evaluate(&old_map)
            .expect("original evaluate returned None");
        let back = decompile(&compile(expr));
        let back_val = back
            .evaluate(&old_map)
            .expect("roundtrip evaluate returned None");
        assert!(
            (orig_val - back_val).abs() < epsilon,
            "compile→decompile mismatch: orig={orig_val} back={back_val} (expr={expr:?})"
        );
    }

    #[test]
    fn round_trip_integer() {
        assert_compile_decompile_match(&Expression::Integer(42), &[], 1e-10);
    }

    #[test]
    fn round_trip_rational_literal() {
        assert_compile_decompile_match(&Expression::Rational(Rational64::new(3, 7)), &[], 1e-10);
    }

    #[test]
    fn round_trip_float_literal() {
        assert_compile_decompile_match(&Expression::Float(2.71828), &[], 1e-10);
    }

    #[test]
    fn round_trip_constant_pi_e() {
        assert_compile_decompile_match(
            &Expression::Constant(crate::ast::SymbolicConstant::Pi),
            &[],
            1e-10,
        );
        assert_compile_decompile_match(
            &Expression::Constant(crate::ast::SymbolicConstant::E),
            &[],
            1e-10,
        );
    }

    #[test]
    fn round_trip_variable() {
        let expr = Expression::Variable(Variable::new("rt_var"));
        assert_compile_decompile_match(&expr, &[("rt_var", 7.0)], 1e-10);
    }

    #[test]
    fn round_trip_binop_add() {
        let x = Expression::Variable(Variable::new("rt_ba"));
        let expr = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(Expression::Integer(5)));
        assert_compile_decompile_match(&expr, &[("rt_ba", 3.0)], 1e-10);
    }

    #[test]
    fn round_trip_binop_sub() {
        let x = Expression::Variable(Variable::new("rt_bs"));
        let expr = Expression::Binary(BinaryOp::Sub, Box::new(x), Box::new(Expression::Integer(2)));
        assert_compile_decompile_match(&expr, &[("rt_bs", 10.0)], 1e-10);
    }

    #[test]
    fn round_trip_binop_mul() {
        let x = Expression::Variable(Variable::new("rt_bm"));
        let expr = Expression::Binary(BinaryOp::Mul, Box::new(x), Box::new(Expression::Integer(4)));
        assert_compile_decompile_match(&expr, &[("rt_bm", 3.0)], 1e-10);
    }

    #[test]
    fn round_trip_binop_div() {
        let x = Expression::Variable(Variable::new("rt_bd"));
        let expr = Expression::Binary(BinaryOp::Div, Box::new(x), Box::new(Expression::Integer(4)));
        assert_compile_decompile_match(&expr, &[("rt_bd", 8.0)], 1e-10);
    }

    #[test]
    fn round_trip_binop_mod_structural() {
        // Mod compiles to Func(Other("mod"), ...) which neither evaluator handles
        // numerically, so we verify the structural round-trip preserves the operator.
        let x = Expression::Variable(Variable::new("rt_bmod"));
        let expr = Expression::Binary(BinaryOp::Mod, Box::new(x), Box::new(Expression::Integer(3)));
        let back = decompile(&compile(&expr));
        let debug = format!("{back:?}");
        assert!(
            debug.contains("mod") || debug.contains("Mod"),
            "expected mod/Mod in round-tripped expression, got: {debug}"
        );
    }

    #[test]
    fn round_trip_unary_neg() {
        let x = Expression::Variable(Variable::new("rt_uneg"));
        let expr = Expression::Unary(UnaryOp::Neg, Box::new(x));
        assert_compile_decompile_match(&expr, &[("rt_uneg", 5.0)], 1e-10);
    }

    #[test]
    fn round_trip_unary_abs() {
        let x = Expression::Variable(Variable::new("rt_uabs"));
        let expr = Expression::Unary(UnaryOp::Abs, Box::new(x));
        assert_compile_decompile_match(&expr, &[("rt_uabs", -3.0)], 1e-10);
    }

    #[test]
    fn round_trip_func_sin() {
        let x = Expression::Variable(Variable::new("rt_fsin"));
        let expr = Expression::Function(Function::Sin, vec![x]);
        assert_compile_decompile_match(&expr, &[("rt_fsin", std::f64::consts::FRAC_PI_2)], 1e-10);
    }

    #[test]
    fn round_trip_func_cos() {
        let x = Expression::Variable(Variable::new("rt_fcos"));
        let expr = Expression::Function(Function::Cos, vec![x]);
        assert_compile_decompile_match(&expr, &[("rt_fcos", 0.0)], 1e-10);
    }

    #[test]
    fn round_trip_func_exp() {
        let x = Expression::Variable(Variable::new("rt_fexp"));
        let expr = Expression::Function(Function::Exp, vec![x]);
        assert_compile_decompile_match(&expr, &[("rt_fexp", 1.0)], 1e-10);
    }

    #[test]
    fn round_trip_func_ln() {
        let x = Expression::Variable(Variable::new("rt_fln"));
        let expr = Expression::Function(Function::Ln, vec![x]);
        assert_compile_decompile_match(&expr, &[("rt_fln", std::f64::consts::E)], 1e-10);
    }

    #[test]
    fn round_trip_func_sqrt() {
        let x = Expression::Variable(Variable::new("rt_fsqrt"));
        let expr = Expression::Function(Function::Sqrt, vec![x]);
        assert_compile_decompile_match(&expr, &[("rt_fsqrt", 9.0)], 1e-10);
    }

    // ── round_trip_decompile_compile ─────────────────────────────────────────

    /// decompile(expr) → compile → evaluate new; compare with evaluate new on
    /// original. Tests that the reverse trip preserves semantics.
    fn assert_decompile_compile_match(
        new_expr: &crate::numeric::expr::Expr,
        str_bindings: &[(&str, f64)],
        epsilon: f64,
    ) {
        use crate::numeric::evaluation::evaluate as eval_new;
        use std::collections::HashMap;

        let mut new_map: HashMap<SymbolId, f64> = HashMap::new();
        for (name, val) in str_bindings {
            new_map.insert(SymbolId::intern(name), *val);
        }

        let orig_val = eval_new(new_expr, &new_map).expect("original new evaluator returned None");
        let recompiled = compile(&decompile(new_expr));
        let back_val =
            eval_new(&recompiled, &new_map).expect("recompiled new evaluator returned None");

        assert!(
            (orig_val - back_val).abs() < epsilon,
            "decompile→compile mismatch: orig={orig_val} back={back_val}"
        );
    }

    #[test]
    fn round_trip_dc_integer() {
        let expr = Expr::Integer(SmallInt::from(99i64));
        assert_decompile_compile_match(&expr, &[], 1e-10);
    }

    #[test]
    fn round_trip_dc_rational() {
        let expr = Expr::Rational(BigRational::from_i64(5, 8));
        assert_decompile_compile_match(&expr, &[], 1e-10);
    }

    #[test]
    fn round_trip_dc_float() {
        let expr = Expr::Float(1.23456);
        assert_decompile_compile_match(&expr, &[], 1e-10);
    }

    #[test]
    fn round_trip_dc_symbol() {
        let expr = Expr::Symbol(SymbolId::intern("rt_dc_sym"));
        assert_decompile_compile_match(&expr, &[("rt_dc_sym", 4.0)], 1e-10);
    }

    #[test]
    fn round_trip_dc_pow_node() {
        // Expr::Pow(x, 3) with x=2 → 8
        let x = Arc::new(Expr::Symbol(SymbolId::intern("rt_dc_pow")));
        let expr = Expr::Pow(x, Expr::int(3));
        assert_decompile_compile_match(&expr, &[("rt_dc_pow", 2.0)], 1e-10);
    }

    #[test]
    fn round_trip_dc_add_node() {
        // AddNode: 5 + 2*x with x=3 → 11
        let x = Expr::symbol("rt_dc_add");
        let mut node = AddNode::from_constant(BigRational::from(5i64));
        node.add_term(x, BigRational::from(2i64));
        let expr = Expr::Add(node);
        assert_decompile_compile_match(&expr, &[("rt_dc_add", 3.0)], 1e-10);
    }

    #[test]
    fn round_trip_dc_mul_node() {
        // MulNode: 3 * x^2 with x=4 → 48
        let x = Expr::symbol("rt_dc_mul");
        let mut node = MulNode::from_coeff(BigRational::from(3i64));
        node.add_factor(x, Expr::int(2));
        let expr = Expr::Mul(node);
        assert_decompile_compile_match(&expr, &[("rt_dc_mul", 4.0)], 1e-10);
    }

    #[test]
    fn round_trip_dc_func_exp() {
        let expr = Expr::Func(crate::numeric::expr::FuncId::Exp, vec![Expr::float(1.0)]);
        assert_decompile_compile_match(&expr, &[], 1e-10);
    }

    #[test]
    fn round_trip_dc_func_sqrt() {
        let expr = Expr::Func(crate::numeric::expr::FuncId::Sqrt, vec![Expr::float(16.0)]);
        assert_decompile_compile_match(&expr, &[], 1e-10);
    }

    #[test]
    fn test_re_compile_decompile_roundtrip() {
        use crate::ast::Function;
        use crate::numeric::compile::{compile, decompile};
        use crate::numeric::expr::FuncId;
        let x = crate::ast::Expression::Variable(crate::ast::Variable::new("z"));
        let re_x = crate::ast::Expression::Function(Function::Re, vec![x]);
        let compiled = compile(&re_x);
        match compiled.as_ref() {
            Expr::Func(FuncId::Re, _) => {}
            _ => panic!("expected Func(Re, ...) after compile"),
        }
        let back = decompile(&compiled);
        assert_eq!(back.to_string(), "Re(z)");
    }

    #[test]
    fn test_im_compile_decompile_roundtrip() {
        use crate::ast::Function;
        use crate::numeric::compile::{compile, decompile};
        use crate::numeric::expr::FuncId;
        let x = crate::ast::Expression::Variable(crate::ast::Variable::new("z"));
        let im_x = crate::ast::Expression::Function(Function::Im, vec![x]);
        let compiled = compile(&im_x);
        match compiled.as_ref() {
            Expr::Func(FuncId::Im, _) => {}
            _ => panic!("expected Func(Im, ...) after compile"),
        }
        let back = decompile(&compiled);
        assert_eq!(back.to_string(), "Im(z)");
    }

    #[test]
    fn test_conj_compile_decompile_roundtrip() {
        use crate::ast::Function;
        use crate::numeric::compile::{compile, decompile};
        use crate::numeric::expr::FuncId;
        let x = crate::ast::Expression::Variable(crate::ast::Variable::new("z"));
        let conj_x = crate::ast::Expression::Function(Function::Conj, vec![x]);
        let compiled = compile(&conj_x);
        match compiled.as_ref() {
            Expr::Func(FuncId::Conj, _) => {}
            _ => panic!("expected Func(Conj, ...) after compile"),
        }
        let back = decompile(&compiled);
        assert_eq!(back.to_string(), "Conj(z)");
    }
}
