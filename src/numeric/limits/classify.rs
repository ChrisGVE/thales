//! Expression classification for limit analysis — detect infinities,
//! division by zero, and indeterminate forms that arise after symbolic
//! substitution.

use std::sync::Arc;

use super::super::expr::{Expr, FuncId};
use super::helpers::{expr_to_f64, rational_to_f64_opt};

#[allow(dead_code)]
#[derive(Debug, PartialEq)]
pub(crate) enum ExprClass {
    /// A well-defined finite value.
    Finite,
    /// Contains division by zero (expression is NaN or ±Inf float).
    DivByZero,
    /// Positive infinity.
    PosInf,
    /// Negative infinity.
    NegInf,
    /// 0/0 indeterminate form (both numerator and denominator vanish).
    Indeterminate,
}

#[allow(dead_code)]
pub(crate) fn classify_expr(expr: &Arc<Expr>) -> ExprClass {
    match expr.as_ref() {
        Expr::Integer(_) | Expr::Rational(_) => ExprClass::Finite,
        Expr::Float(f) => {
            if f.is_nan() {
                ExprClass::Indeterminate
            } else if f.is_infinite() {
                if *f > 0.0 {
                    ExprClass::PosInf
                } else {
                    ExprClass::NegInf
                }
            } else {
                ExprClass::Finite
            }
        }
        Expr::Mul(node) => {
            // Check if the coefficient is infinite (from propagated f64 ops).
            if let Some(v) = rational_to_f64_opt(&node.coeff) {
                if v.is_nan() {
                    return ExprClass::Indeterminate;
                }
                if v.is_infinite() {
                    return if v > 0.0 {
                        ExprClass::PosInf
                    } else {
                        ExprClass::NegInf
                    };
                }
            }
            // If any factor evaluates to a numeric infinity, propagate.
            for (base, exp) in &node.factors {
                let b_cls = classify_expr(base);
                let e_cls = classify_expr(exp);
                if matches!(b_cls, ExprClass::DivByZero | ExprClass::Indeterminate)
                    || matches!(e_cls, ExprClass::DivByZero | ExprClass::Indeterminate)
                {
                    return ExprClass::Indeterminate;
                }
            }
            ExprClass::Finite
        }
        // A Pow with a zero base and negative exponent = division by zero.
        Expr::Pow(base, exp) => {
            if base.is_zero() {
                if let Some(e) = expr_to_f64(exp) {
                    if e < 0.0 {
                        return ExprClass::DivByZero;
                    }
                }
            }
            let bc = classify_expr(base);
            let ec = classify_expr(exp);
            if matches!(bc, ExprClass::Indeterminate) || matches!(ec, ExprClass::Indeterminate) {
                return ExprClass::Indeterminate;
            }
            ExprClass::Finite
        }
        Expr::Func(id, args) => classify_func(*id, args),
        _ => ExprClass::Finite,
    }
}

#[allow(dead_code)]
fn classify_func(id: FuncId, args: &[Arc<Expr>]) -> ExprClass {
    if args.len() == 1 {
        let arg_cls = classify_expr(&args[0]);
        if matches!(arg_cls, ExprClass::Indeterminate | ExprClass::DivByZero) {
            return ExprClass::Indeterminate;
        }
        // ln(0) = -∞
        if matches!(id, FuncId::Ln) && args[0].is_zero() {
            return ExprClass::NegInf;
        }
        // exp at a finite numeric arg.
        if matches!(id, FuncId::Exp) {
            if let Some(v) = expr_to_f64(&args[0]) {
                let r = v.exp();
                if r.is_infinite() {
                    return if r > 0.0 {
                        ExprClass::PosInf
                    } else {
                        ExprClass::NegInf
                    };
                }
                if r.is_nan() {
                    return ExprClass::Indeterminate;
                }
                return ExprClass::Finite;
            }
        }
    }
    ExprClass::Finite
}
