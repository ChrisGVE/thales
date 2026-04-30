//! Symbol substitution on `Arc<Expr>` trees.
//!
//! Replaces every occurrence of a [`SymbolId`] with a replacement expression,
//! rebuilding the tree through the normalizing smart constructors in
//! [`crate::numeric::normalize`] so that constant folding, identity removal,
//! and like-term combination apply automatically.
//!
//! Built-in unary functions (`sin`, `cos`, `exp`, `ln`, `sqrt`, …) are
//! additionally folded to a numeric literal when their argument is numeric.
//! This matches the behaviour expected by Taylor series coefficient
//! construction (`f^(n)(center) / n!` with a numeric `center`) without
//! forcing callers that do not need folding to pay for it separately.

use std::sync::Arc;

use num::traits::{One, Zero};

use super::expr::{Expr, FuncId};
use super::{normalize, BigRational, SymbolId};

/// Substitute every occurrence of `var` in `expr` with `replacement`.
///
/// Rebuilds through normalize constructors and numerically folds built-in
/// unary functions applied to numeric arguments.
pub(crate) fn substitute(expr: &Arc<Expr>, var: SymbolId, replacement: &Arc<Expr>) -> Arc<Expr> {
    match expr.as_ref() {
        Expr::Integer(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Complex(_)
        | Expr::Constant(_) => expr.clone(),

        Expr::Symbol(s) => {
            if *s == var {
                replacement.clone()
            } else {
                expr.clone()
            }
        }

        Expr::Add(node) => {
            let mut acc: Arc<Expr> = if node.constant.is_zero() {
                Expr::int(0)
            } else {
                rational_to_arc(&node.constant)
            };
            for (term, coeff) in &node.terms {
                let new_term = substitute(term, var, replacement);
                let scaled = normalize::mul(rational_to_arc(coeff), new_term);
                acc = normalize::add(acc, scaled);
            }
            acc
        }

        Expr::Mul(node) => {
            let mut acc: Arc<Expr> = rational_to_arc(&node.coeff);
            for (base, exp) in &node.factors {
                let new_base = substitute(base, var, replacement);
                let new_exp = substitute(exp, var, replacement);
                acc = normalize::mul(acc, normalize::pow(new_base, new_exp));
            }
            acc
        }

        Expr::Pow(base, exp) => {
            let new_base = substitute(base, var, replacement);
            let new_exp = substitute(exp, var, replacement);
            normalize::pow(new_base, new_exp)
        }

        Expr::Func(id, args) => {
            let new_args: Vec<Arc<Expr>> = args
                .iter()
                .map(|a| substitute(a, var, replacement))
                .collect();
            rebuild_func(*id, new_args)
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Convert a `&BigRational` coefficient to an `Arc<Expr>`, preferring
/// `Expr::Integer` when the rational is integer-valued.
fn rational_to_arc(r: &BigRational) -> Arc<Expr> {
    if r.denom().is_one() {
        if let Some(n) = r.numer().to_i64() {
            return Expr::int(n);
        }
    }
    Arc::new(Expr::Rational(r.clone()))
}

/// Rebuild a function application, folding to a numeric literal when a
/// single-argument built-in is applied to a numeric constant.
fn rebuild_func(id: FuncId, args: Vec<Arc<Expr>>) -> Arc<Expr> {
    if args.len() == 1 {
        if let Some(result) = try_eval_func(id, &args[0]) {
            return result;
        }
    }
    Arc::new(Expr::Func(id, args))
}

/// Attempt to evaluate a built-in unary function at a numeric constant.
///
/// Returns `None` for symbolic arguments, out-of-domain inputs, or functions
/// that require more than one argument.
fn try_eval_func(id: FuncId, arg: &Arc<Expr>) -> Option<Arc<Expr>> {
    let v = numeric_f64(arg)?;
    let result = match id {
        FuncId::Sin => v.sin(),
        FuncId::Cos => v.cos(),
        FuncId::Tan => v.tan(),
        FuncId::Asin => v.asin(),
        FuncId::Acos => v.acos(),
        FuncId::Atan => v.atan(),
        FuncId::Sinh => v.sinh(),
        FuncId::Cosh => v.cosh(),
        FuncId::Tanh => v.tanh(),
        FuncId::Ln => {
            if v <= 0.0 {
                return None;
            }
            v.ln()
        }
        FuncId::Exp => v.exp(),
        FuncId::Log2 => {
            if v <= 0.0 {
                return None;
            }
            v.log2()
        }
        FuncId::Log10 => {
            if v <= 0.0 {
                return None;
            }
            v.log10()
        }
        FuncId::Sqrt => {
            if v < 0.0 {
                return None;
            }
            v.sqrt()
        }
        FuncId::Cbrt => v.cbrt(),
        FuncId::Floor => v.floor(),
        FuncId::Ceil => v.ceil(),
        FuncId::Round => v.round(),
        FuncId::Abs => v.abs(),
        FuncId::Sign => v.signum(),
        FuncId::Atan2 | FuncId::Log | FuncId::Min | FuncId::Max => return None,
        FuncId::Re | FuncId::Im | FuncId::Conj => return None,
        FuncId::Gamma
        | FuncId::LnGamma
        | FuncId::Digamma
        | FuncId::BetaFn
        | FuncId::Erf
        | FuncId::Erfc
        | FuncId::BesselJ
        | FuncId::BesselY
        | FuncId::BesselI
        | FuncId::BesselK
        | FuncId::AiryAi
        | FuncId::AiryBi
        | FuncId::Zeta
        | FuncId::Si
        | FuncId::Ci
        | FuncId::Ei
        | FuncId::Heaviside
        | FuncId::DiracDelta => return None,
        FuncId::Other(_) => return None,
    };
    if result == 0.0 {
        Some(Expr::int(0))
    } else if result == 1.0 {
        Some(Expr::int(1))
    } else {
        Some(Expr::float(result))
    }
}

/// Extract an `f64` from a numeric `Expr`, returning `None` for non-numeric.
fn numeric_f64(expr: &Arc<Expr>) -> Option<f64> {
    match expr.as_ref() {
        Expr::Integer(n) => n.to_i64().map(|v| v as f64),
        Expr::Rational(r) => Some(r.to_f64()),
        Expr::Float(f) => Some(*f),
        _ => None,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::expr::FuncId;

    fn sym(name: &str) -> Arc<Expr> {
        Arc::new(Expr::Symbol(SymbolId::intern(name)))
    }

    #[test]
    fn symbol_hit_returns_replacement() {
        let expr = sym("x");
        let value = Expr::int(5);
        let result = substitute(&expr, SymbolId::intern("x"), &value);
        assert_eq!(*result, *value);
    }

    #[test]
    fn symbol_miss_returns_same_handle() {
        let expr = sym("y");
        let value = Expr::int(5);
        let result = substitute(&expr, SymbolId::intern("x"), &value);
        assert!(Arc::ptr_eq(&expr, &result));
    }

    #[test]
    fn numeric_leaves_untouched() {
        let expr = Expr::int(7);
        let value = Expr::int(5);
        let result = substitute(&expr, SymbolId::intern("x"), &value);
        assert!(Arc::ptr_eq(&expr, &result));
    }

    #[test]
    fn add_folds_after_substitution() {
        let expr = normalize::add(sym("x"), sym("y"));
        let value = Expr::int(3);
        let result = substitute(&expr, SymbolId::intern("x"), &value);
        match result.as_ref() {
            Expr::Add(node) => {
                assert_eq!(
                    node.constant,
                    BigRational::from_integer(crate::numeric::SmallInt::from(3i64))
                );
                assert_eq!(node.terms.len(), 1);
            }
            _ => panic!("expected AddNode, got {:?}", result),
        }
    }

    #[test]
    fn func_argument_substituted() {
        // sin(x) with x → y — symbolic arg, no fold.
        let expr = Arc::new(Expr::Func(FuncId::Sin, vec![sym("x")]));
        let result = substitute(&expr, SymbolId::intern("x"), &sym("y"));
        match result.as_ref() {
            Expr::Func(FuncId::Sin, args) => {
                assert_eq!(args.len(), 1);
                assert!(matches!(*args[0], Expr::Symbol(s) if s == SymbolId::intern("y")));
            }
            _ => panic!("expected sin(y), got {:?}", result),
        }
    }

    #[test]
    fn pow_base_and_exp_substituted() {
        let expr = normalize::pow(sym("x"), sym("x"));
        let value = Expr::int(2);
        let result = substitute(&expr, SymbolId::intern("x"), &value);
        assert_eq!(*result, Expr::Integer(crate::numeric::SmallInt::from(4i64)));
    }

    #[test]
    fn sin_at_zero_folds_to_zero() {
        // sin(x) with x → 0 must fold to the integer 0.
        let expr = Arc::new(Expr::Func(FuncId::Sin, vec![sym("x")]));
        let result = substitute(&expr, SymbolId::intern("x"), &Expr::int(0));
        assert!(
            result.is_zero(),
            "sin(0) should fold to 0, got {:?}",
            result
        );
    }

    #[test]
    fn cos_at_zero_folds_to_one() {
        let expr = Arc::new(Expr::Func(FuncId::Cos, vec![sym("x")]));
        let result = substitute(&expr, SymbolId::intern("x"), &Expr::int(0));
        assert!(result.is_one(), "cos(0) should fold to 1, got {:?}", result);
    }

    #[test]
    fn exp_at_zero_folds_to_one() {
        let expr = Arc::new(Expr::Func(FuncId::Exp, vec![sym("x")]));
        let result = substitute(&expr, SymbolId::intern("x"), &Expr::int(0));
        assert!(result.is_one(), "exp(0) should fold to 1, got {:?}", result);
    }

    #[test]
    fn ln_at_non_positive_stays_symbolic() {
        // ln is undefined at 0; the fold returns None and the Func stays.
        let expr = Arc::new(Expr::Func(FuncId::Ln, vec![sym("x")]));
        let result = substitute(&expr, SymbolId::intern("x"), &Expr::int(0));
        assert!(
            matches!(result.as_ref(), Expr::Func(FuncId::Ln, _)),
            "ln(0) should remain symbolic, got {:?}",
            result
        );
    }
}
