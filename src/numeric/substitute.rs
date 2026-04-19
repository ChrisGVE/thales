//! Symbol substitution on `Arc<Expr>` trees.
//!
//! Replaces every occurrence of a [`SymbolId`] with a replacement expression,
//! rebuilding the tree through the normalizing smart constructors in
//! [`crate::numeric::normalize`] so that constant folding, identity removal,
//! and like-term combination apply automatically.

use std::sync::Arc;

use super::expr::Expr;
use super::{normalize, SymbolId};
use num::traits::Zero;

/// Substitute every occurrence of `var` in `expr` with `replacement`.
///
/// Rebuilds through normalize constructors; numeric literals and the
/// target replacement are returned cloned when no rewrite applies.
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
                Arc::new(Expr::Rational(node.constant.clone()))
            };
            for (term, coeff) in &node.terms {
                let new_term = substitute(term, var, replacement);
                let scaled = normalize::mul(Arc::new(Expr::Rational(coeff.clone())), new_term);
                acc = normalize::add(acc, scaled);
            }
            acc
        }

        Expr::Mul(node) => {
            let mut acc: Arc<Expr> = Arc::new(Expr::Rational(node.coeff.clone()));
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
            Expr::func(*id, new_args)
        }
    }
}

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
        // (x + y) with x → 3, resulting 3 + y via normalize::add
        let expr = normalize::add(sym("x"), sym("y"));
        let value = Expr::int(3);
        let result = substitute(&expr, SymbolId::intern("x"), &value);
        // Expect AddNode with constant=3, single term y.
        match result.as_ref() {
            Expr::Add(node) => {
                assert_eq!(
                    node.constant,
                    crate::numeric::BigRational::from_integer(crate::numeric::SmallInt::from(3i64))
                );
                assert_eq!(node.terms.len(), 1);
            }
            _ => panic!("expected AddNode, got {:?}", result),
        }
    }

    #[test]
    fn func_argument_substituted() {
        // sin(x) with x → y
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
        // x^x with x → 2  →  2^2 → 4 via normalize::pow
        let expr = normalize::pow(sym("x"), sym("x"));
        let value = Expr::int(2);
        let result = substitute(&expr, SymbolId::intern("x"), &value);
        assert_eq!(*result, Expr::Integer(crate::numeric::SmallInt::from(4i64)));
    }
}
