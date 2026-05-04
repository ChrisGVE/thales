//! Symbolic differentiation for the [`Expr`] type.
//!
//! Provides `diff` for computing the derivative of an expression with respect
//! to a variable, and `implicit_diff` for implicit differentiation of an
//! equation `F(x, y) = 0`.
//!
//! # Design
//!
//! Derivatives are computed by structural recursion over the [`Expr`] tree.
//! Every intermediate result is built through the smart constructors in
//! [`crate::numeric::normalize`], so constant folding, identity removal,
//! and canonicalization happen automatically — no separate simplification
//! pass is required.
//!
//! # Supported rules
//!
//! | Expression         | Derivative                         |
//! |--------------------|------------------------------------|
//! | constant           | 0                                  |
//! | x (var)            | 1                                  |
//! | other symbol       | 0                                  |
//! | -u                 | -u'                                |
//! | u + v              | u' + v'                            |
//! | u * v              | u'*v + u*v'  (product rule)        |
//! | u ^ n (n const)    | n * u^(n-1) * u'  (power rule)     |
//! | u ^ v (general)    | u^v * (v'*ln(u) + v*u'/u)          |
//! | sin(u)             | cos(u) * u'                        |
//! | cos(u)             | -sin(u) * u'                       |
//! | tan(u)             | (1 + tan²(u)) * u'                 |
//! | ln(u)              | u' / u                             |
//! | exp(u)             | exp(u) * u'                        |
//! | sqrt(u)            | u' / (2 * sqrt(u))                 |
//! | abs(u)             | u * u' / abs(u)                    |
//! | other func(u)      | func'(u) * u'  (opaque chain rule) |

use std::sync::Arc;

use super::{expr::Expr, normalize, SymbolId};

use rules::{diff_add, diff_mul_node, diff_pow};
use rules_elementary::diff_func;

pub(crate) mod rules;
pub(crate) mod rules_elementary;

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute the symbolic derivative of `expr` with respect to `var`.
///
/// Results are normalized via smart constructors: constant expressions
/// evaluate to integers/rationals, and identities (0+x, 1*x) are removed
/// automatically.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{Expr, SymbolId};
/// use thales::numeric::differentiation::diff;
///
/// // d/dx(x^2) = 2*x
/// let x_id = SymbolId::intern("diff_x2");
/// let x = Expr::symbol("diff_x2");
/// let x_sq = Expr::pow(x.clone(), Expr::int(2));
/// let result = diff(&x_sq, x_id);
/// // Normalized: 2*x
/// assert!(!result.is_zero());
/// ```
///
/// ```rust
/// use thales::numeric::{Expr, SymbolId};
/// use thales::numeric::differentiation::diff;
///
/// // d/dx(5) = 0
/// let x_id = SymbolId::intern("diff_const");
/// let result = diff(&Expr::int(5), x_id);
/// assert!(result.is_zero());
/// ```
pub fn diff(expr: &Expr, var: SymbolId) -> Arc<Expr> {
    diff_arc(&Arc::new(expr.clone()), var)
}

/// Compute implicit derivative dy/dx from an equation `F(x, y) = 0`.
///
/// Uses the formula:  dy/dx = -(∂F/∂x) / (∂F/∂y)
///
/// Both partial derivatives are computed symbolically and the result is
/// normalized through the smart constructors.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{Expr, SymbolId, normalize};
/// use thales::numeric::differentiation::implicit_diff;
///
/// // F = x^2 + y^2 - 1 (unit circle)
/// // dy/dx = -x/y  (implicit differentiation)
/// let x_id = SymbolId::intern("impl_x");
/// let y_id = SymbolId::intern("impl_y");
/// let x = Expr::symbol("impl_x");
/// let y = Expr::symbol("impl_y");
/// let x_sq = Expr::pow(x.clone(), Expr::int(2));
/// let y_sq = Expr::pow(y.clone(), Expr::int(2));
/// let f = normalize::sub(normalize::add(x_sq, y_sq), Expr::int(1));
/// let result = implicit_diff(&f, x_id, y_id);
/// // result = -x/y (up to canonical form)
/// assert!(!result.is_zero());
/// ```
pub fn implicit_diff(equation: &Expr, x: SymbolId, y: SymbolId) -> Arc<Expr> {
    let eq = Arc::new(equation.clone());
    let df_dx = diff_arc(&eq, x);
    let df_dy = diff_arc(&eq, y);
    // dy/dx = -(∂F/∂x) / (∂F/∂y)
    let neg_df_dx = normalize::neg(df_dx);
    normalize::div(neg_df_dx, df_dy)
}

// ── Core recursive differentiator ────────────────────────────────────────────

/// Internal: differentiate an `Arc<Expr>`, returning an `Arc<Expr>`.
pub(crate) fn diff_arc(expr: &Arc<Expr>, var: SymbolId) -> Arc<Expr> {
    match expr.as_ref() {
        // Constants → 0
        Expr::Integer(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Complex(_)
        | Expr::Constant(_) => Expr::int(0),

        // Variable match → 1, otherwise 0
        Expr::Symbol(s) => {
            if *s == var {
                Expr::int(1)
            } else {
                Expr::int(0)
            }
        }

        // Neg is encoded as MulNode with coeff -1; handle Add which subsumes Neg
        Expr::Add(node) => diff_add(node, var),

        Expr::Mul(node) => diff_mul_node(node, expr, var),

        Expr::Pow(base, exp) => diff_pow(base, exp, var),

        Expr::Func(id, args) => diff_func(*id, args, expr, var),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{expr::FuncId, normalize, Expr, SmallInt, SymbolId};

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }
    fn x_id(name: &str) -> SymbolId {
        SymbolId::intern(name)
    }

    // ── Constants ────────────────────────────────────────────────────────

    #[test]
    fn test_diff_integer_is_zero() {
        assert!(diff(&Expr::int(7), x_id("dc_x")).is_zero());
    }

    #[test]
    fn test_diff_rational_is_zero() {
        let r = Expr::rational(3, 4);
        assert!(diff(&r, x_id("dc_rx")).is_zero());
    }

    #[test]
    fn test_diff_float_is_zero() {
        assert!(diff(&Expr::Float(3.14), x_id("dc_fx")).is_zero());
    }

    // ── Variables ────────────────────────────────────────────────────────

    #[test]
    fn test_diff_var_wrt_itself() {
        let xid = x_id("dv_x");
        let x = sym("dv_x");
        assert!(diff(&x, xid).is_one());
    }

    #[test]
    fn test_diff_var_wrt_other() {
        let x = sym("dv_other_x");
        let yid = x_id("dv_other_y");
        assert!(diff(&x, yid).is_zero());
    }

    // ── Addition ─────────────────────────────────────────────────────────

    #[test]
    fn test_diff_sum() {
        // d/dx (x + y) = 1
        let xid = x_id("ds_x");
        let x = sym("ds_x");
        let y = sym("ds_y");
        let e = normalize::add(x, y);
        let result = diff(&e, xid);
        assert!(result.is_one(), "expected 1, got {result}");
    }

    #[test]
    fn test_diff_sum_both_vars() {
        // d/dx (x + x) = 2
        let xid = x_id("dsb_x");
        let x = sym("dsb_x");
        let e = normalize::add(x.clone(), x.clone());
        let result = diff(&e, xid);
        assert_eq!(*result, Expr::Integer(SmallInt::from(2i64)));
    }

    // ── Implicit differentiation ─────────────────────────────────────────

    #[test]
    fn test_implicit_diff_circle() {
        // F = x^2 + y^2 - 1; dy/dx = -x/y
        let xid = x_id("imp_x");
        let yid = x_id("imp_y");
        let x = sym("imp_x");
        let y = sym("imp_y");
        let x_sq = Expr::pow(x.clone(), Expr::int(2));
        let y_sq = Expr::pow(y.clone(), Expr::int(2));
        let f = normalize::sub(normalize::add(x_sq, y_sq), Expr::int(1));
        let result = implicit_diff(&f, xid, yid);
        assert!(
            !result.is_zero(),
            "implicit diff should not be zero for circle"
        );
    }

    #[test]
    fn test_implicit_diff_linear() {
        // F = x + y; dy/dx = -1
        let xid = x_id("iml_x");
        let yid = x_id("iml_y");
        let x = sym("iml_x");
        let y = sym("iml_y");
        let f = normalize::add(x, y);
        let result = implicit_diff(&f, xid, yid);
        // dy/dx = -(1) / (1) = -1
        assert_eq!(
            *result,
            Expr::Integer(SmallInt::from(-1i64)),
            "expected -1, got {result}"
        );
    }

    // ── Neg (encoded as -1 * x) ──────────────────────────────────────────

    #[test]
    fn test_diff_neg_x() {
        // d/dx(-x) = -1
        let xid = x_id("dneg_x");
        let x = sym("dneg_x");
        let e = normalize::neg(x);
        let result = diff(&e, xid);
        assert_eq!(
            *result,
            Expr::Integer(SmallInt::from(-1i64)),
            "expected -1, got {result}"
        );
    }

    // ── Re/Im/Conj differentiation ────────────────────────────────────────

    #[test]
    fn test_diff_re_of_sin() {
        // d/dx Re(sin(x)) = Re(cos(x))  — Re distributes linearly
        let xid = x_id("dre_sin");
        let x = sym("dre_sin");
        let sin_x = Expr::func(FuncId::Sin, vec![x.clone()]);
        let re_sin_x = Expr::func(FuncId::Re, vec![sin_x]);
        let result = diff(&re_sin_x, xid);
        // Result should be Re(cos(x))
        match result.as_ref() {
            Expr::Func(FuncId::Re, inner) if inner.len() == 1 => {
                assert!(
                    matches!(inner[0].as_ref(), Expr::Func(FuncId::Cos, _)),
                    "expected Re(cos(x)), got Re({:?})",
                    inner[0]
                );
            }
            _ => panic!("expected Re(cos(x)), got {:?}", result),
        }
    }

    #[test]
    fn test_diff_im_of_exp() {
        // d/dx Im(exp(x)) = Im(exp(x))  — Im distributes linearly
        let xid = x_id("dim_exp");
        let x = sym("dim_exp");
        let exp_x = Expr::func(FuncId::Exp, vec![x.clone()]);
        let im_exp_x = Expr::func(FuncId::Im, vec![exp_x]);
        let result = diff(&im_exp_x, xid);
        match result.as_ref() {
            Expr::Func(FuncId::Im, inner) if inner.len() == 1 => {
                assert!(
                    matches!(inner[0].as_ref(), Expr::Func(FuncId::Exp, _)),
                    "expected Im(exp(x)), got Im({:?})",
                    inner[0]
                );
            }
            _ => panic!("expected Im(exp(x)), got {:?}", result),
        }
    }

    #[test]
    fn test_diff_conj_of_poly() {
        // d/dx Conj(x^2 + 3) = Conj(2x)
        let xid = x_id("dconj_poly");
        let x = sym("dconj_poly");
        let x_sq = normalize::pow(x.clone(), Expr::int(2));
        let poly = normalize::add(x_sq, Expr::int(3));
        let conj_poly = Expr::func(FuncId::Conj, vec![poly]);
        let result = diff(&conj_poly, xid);
        // Result should be Conj(2*x) or Conj(derivative)
        assert!(
            matches!(result.as_ref(), Expr::Func(FuncId::Conj, _)),
            "expected Conj(...), got {:?}",
            result
        );
    }

    #[test]
    fn test_diff_re_constant() {
        // d/dx Re(5) = Re(0) = 0
        let xid = x_id("dre_const");
        let re_5 = Expr::func(FuncId::Re, vec![Expr::int(5)]);
        let result = diff(&re_5, xid);
        // Re(0) should simplify to 0
        assert!(
            result.is_zero() || matches!(result.as_ref(), Expr::Func(FuncId::Re, _)),
            "expected 0 or Re(0), got {:?}",
            result
        );
    }
}
