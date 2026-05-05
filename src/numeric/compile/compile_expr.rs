//! Compile stage: [`Expression`] → [`Arc<Expr>`].

use crate::ast::{BinaryOp, Expression, Function, UnaryOp, Variable};
use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::normalize;
use crate::numeric::{BigRational, SmallInt, SymbolId};
use std::sync::Arc;

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
        Function::Gamma => FuncId::Gamma,
        Function::LnGamma => FuncId::LnGamma,
        Function::Digamma => FuncId::Digamma,
        Function::BetaFn => FuncId::BetaFn,
        Function::Erf => FuncId::Erf,
        Function::Erfc => FuncId::Erfc,
        Function::BesselJ => FuncId::BesselJ,
        Function::BesselY => FuncId::BesselY,
        Function::BesselI => FuncId::BesselI,
        Function::BesselK => FuncId::BesselK,
        Function::AiryAi => FuncId::AiryAi,
        Function::AiryBi => FuncId::AiryBi,
        Function::Zeta => FuncId::Zeta,
        Function::Si => FuncId::Si,
        Function::Ci => FuncId::Ci,
        Function::Ei => FuncId::Ei,
        Function::Heaviside => FuncId::Heaviside,
        Function::DiracDelta => FuncId::DiracDelta,
        // Pow → sentinel so compile() can route to normalize::pow
        Function::Pow => FuncId::Other(SymbolId::intern("__pow__")),
        Function::Custom(name) => FuncId::Other(SymbolId::intern(name)),
        // Non-exhaustive guard: map any future variants to Other.
        #[allow(unreachable_patterns)]
        _ => FuncId::Other(SymbolId::intern("__unknown__")),
    }
}
