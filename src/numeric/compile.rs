//! Compiler from the legacy `ast::Expression` tree to the CAS `numeric::Expr`.
//!
//! [`compile`] performs a structural translation, mapping each `Expression`
//! variant to its `Expr` equivalent and using the smart constructors in
//! [`normalize`] so the output is already in canonical form.

use crate::ast::{BinaryOp, Expression, Function, UnaryOp};
use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::normalize;
use crate::numeric::{BigRational, SmallInt, SymbolId};
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
        // Pow → sentinel so compile() can route to normalize::pow
        Function::Pow => FuncId::Other(SymbolId::intern("__pow__")),
        Function::Custom(name) => FuncId::Other(SymbolId::intern(name)),
        // Non-exhaustive guard: map any future variants to Other
        _ => FuncId::Other(SymbolId::intern("__unknown__")),
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
}
