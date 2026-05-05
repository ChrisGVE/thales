//! Decompile stage: [`Expr`] → [`Expression`].

use crate::ast::{BinaryOp, Expression, Function, UnaryOp, Variable};
use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::{AddNode, BigRational, MulNode, SymbolId};
use num::traits::{One, Zero};
use num_rational::Rational64;
use std::sync::Arc;

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
        FuncId::Gamma => Function::Gamma,
        FuncId::LnGamma => Function::LnGamma,
        FuncId::Digamma => Function::Digamma,
        FuncId::BetaFn => Function::BetaFn,
        FuncId::Erf => Function::Erf,
        FuncId::Erfc => Function::Erfc,
        FuncId::BesselJ => Function::BesselJ,
        FuncId::BesselY => Function::BesselY,
        FuncId::BesselI => Function::BesselI,
        FuncId::BesselK => Function::BesselK,
        FuncId::AiryAi => Function::AiryAi,
        FuncId::AiryBi => Function::AiryBi,
        FuncId::Zeta => Function::Zeta,
        FuncId::Si => Function::Si,
        FuncId::Ci => Function::Ci,
        FuncId::Ei => Function::Ei,
        FuncId::Heaviside => Function::Heaviside,
        FuncId::DiracDelta => Function::DiracDelta,
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
pub(super) fn big_rational_to_expr(r: &BigRational) -> Expression {
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
