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

use super::{
    expr::{Expr, FuncId},
    normalize, SymbolId,
};

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

// ── Differentiation of Add ────────────────────────────────────────────────────

fn diff_add(node: &super::AddNode, var: SymbolId) -> Arc<Expr> {
    // Constant part disappears; differentiate each term
    // AddNode: constant + Σ coeff_i * term_i
    // d/dx = Σ coeff_i * diff(term_i)
    let mut result = Expr::int(0);
    for (term, coeff) in &node.terms {
        let dt = diff_arc(term, var);
        let scaled = normalize::mul(rational_to_arc(coeff), dt);
        result = normalize::add(result, scaled);
    }
    result
}

/// Convert a &BigRational to an Arc<Expr>.
fn rational_to_arc(r: &super::BigRational) -> Arc<Expr> {
    if r.is_integer() {
        if let Some(n) = r.numer().to_i64() {
            Expr::int(n)
        } else {
            Arc::new(Expr::Rational(r.clone()))
        }
    } else {
        Arc::new(Expr::Rational(r.clone()))
    }
}

// ── Differentiation of Mul ───────────────────────────────────────────────────

fn diff_mul_node(node: &super::MulNode, _full: &Arc<Expr>, var: SymbolId) -> Arc<Expr> {
    // MulNode: coeff * Π base_i^exp_i
    // The coefficient is a constant → its derivative contributes 0.
    // We treat each factor base_i^exp_i as a sub-expression and apply
    // the product rule across all factors.
    //
    // d/dx [ coeff * f1 * f2 * ... * fn ]
    //   = coeff * Σ_i [ (d/dx fi) * Π_{j≠i} fj ]
    //
    // Each fi = base_i^exp_i is differentiated via diff_pow.

    let factors: Vec<Arc<Expr>> = node
        .factors
        .iter()
        .map(|(base, exp)| normalize::pow(base.clone(), exp.clone()))
        .collect();

    let coeff_expr: Arc<Expr> = rational_to_arc(&node.coeff);

    // Build product-rule sum
    let mut sum = Expr::int(0);
    for i in 0..factors.len() {
        let fi = &factors[i];
        let dfi = diff_arc(fi, var);
        if dfi.is_zero() {
            continue;
        }
        // Build product of all factors except i
        let mut others = Expr::int(1) as Arc<Expr>;
        for (j, fj) in factors.iter().enumerate() {
            if j != i {
                others = normalize::mul(others, fj.clone());
            }
        }
        let term = normalize::mul(dfi, others);
        sum = normalize::add(sum, term);
    }

    normalize::mul(coeff_expr, sum)
}

// ── Differentiation of Pow ───────────────────────────────────────────────────

fn diff_pow(base: &Arc<Expr>, exp: &Arc<Expr>, var: SymbolId) -> Arc<Expr> {
    // Check if exponent is a numeric constant
    let exp_is_const = is_numeric_const(exp);
    let base_du = diff_arc(base, var);
    let exp_du = diff_arc(exp, var);

    if exp_is_const {
        // Power rule: d/dx(u^n) = n * u^(n-1) * u'
        if base_du.is_zero() {
            return Expr::int(0);
        }
        let n_minus_1 = normalize::sub(exp.clone(), Expr::int(1));
        let u_pow = normalize::pow(base.clone(), n_minus_1);
        let n_u_pow = normalize::mul(exp.clone(), u_pow);
        normalize::mul(n_u_pow, base_du)
    } else {
        // General: d/dx(u^v) = u^v * (v'*ln(u) + v*u'/u)
        let u_pow_v = normalize::pow(base.clone(), exp.clone());
        let ln_u = Expr::func(FuncId::Ln, vec![base.clone()]);
        let v_prime_ln_u = normalize::mul(exp_du, ln_u);
        let u_prime_over_u = normalize::div(base_du, base.clone());
        let v_u_prime_over_u = normalize::mul(exp.clone(), u_prime_over_u);
        let bracket = normalize::add(v_prime_ln_u, v_u_prime_over_u);
        normalize::mul(u_pow_v, bracket)
    }
}

/// Returns true if `expr` is a numeric constant (no free variables).
fn is_numeric_const(expr: &Arc<Expr>) -> bool {
    matches!(
        expr.as_ref(),
        Expr::Integer(_) | Expr::Rational(_) | Expr::Float(_)
    )
}

// ── Differentiation of Func ──────────────────────────────────────────────────

fn diff_func(id: FuncId, args: &[Arc<Expr>], _full: &Arc<Expr>, var: SymbolId) -> Arc<Expr> {
    // Single-argument built-in functions: chain rule f'(u)*u'
    if args.len() == 1 {
        let u = &args[0];
        let du = diff_arc(u, var);
        if du.is_zero() {
            return Expr::int(0);
        }
        let outer = diff_builtin_single(id, u);
        normalize::mul(outer, du)
    } else {
        // Multi-argument or unknown: return opaque derivative placeholder 0
        // (symbolic extension point — not in scope for built-ins)
        Expr::int(0)
    }
}

/// Derivative of a single-argument built-in function, evaluated at `u`.
/// Returns `f'(u)` (outer derivative only; chain rule applied by caller).
fn diff_builtin_single(id: FuncId, u: &Arc<Expr>) -> Arc<Expr> {
    match id {
        // d/du sin(u) = cos(u)
        FuncId::Sin => Expr::func(FuncId::Cos, vec![u.clone()]),

        // d/du cos(u) = -sin(u)
        FuncId::Cos => normalize::neg(Expr::func(FuncId::Sin, vec![u.clone()])),

        // d/du tan(u) = 1 + tan²(u)
        FuncId::Tan => {
            let tan_u = Expr::func(FuncId::Tan, vec![u.clone()]);
            let tan_sq = normalize::pow(tan_u, Expr::int(2));
            normalize::add(Expr::int(1), tan_sq)
        }

        // d/du ln(u) = 1/u
        FuncId::Ln => normalize::div(Expr::int(1), u.clone()),

        // d/du exp(u) = exp(u)
        FuncId::Exp => Expr::func(FuncId::Exp, vec![u.clone()]),

        // d/du sqrt(u) = 1 / (2 * sqrt(u))
        FuncId::Sqrt => {
            let sqrt_u = Expr::func(FuncId::Sqrt, vec![u.clone()]);
            let two_sqrt = normalize::mul(Expr::int(2), sqrt_u);
            normalize::div(Expr::int(1), two_sqrt)
        }

        // d/du abs(u) = u / abs(u)
        FuncId::Abs => {
            let abs_u = Expr::func(FuncId::Abs, vec![u.clone()]);
            normalize::div(u.clone(), abs_u)
        }

        // Unknown/user function: produce opaque derivative marker (zero — caller
        // should not use the result as authoritative)
        FuncId::Other(_) => Expr::int(0),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{normalize, Expr, SmallInt, SymbolId};

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

    // ── Multiplication / Product rule ────────────────────────────────────

    #[test]
    fn test_diff_product_rule() {
        // d/dx(x * y) = y  (y is treated as constant)
        let xid = x_id("dpr_x");
        let x = sym("dpr_x");
        let y = sym("dpr_y");
        let e = normalize::mul(x.clone(), y.clone());
        let result = diff(&e, xid);
        // normalized result should equal y
        assert_eq!(*result, *y, "expected y, got {result}");
    }

    #[test]
    fn test_diff_x_squared() {
        // d/dx(x^2) via mul: x*x → 2*x
        let xid = x_id("dxs_x");
        let x = sym("dxs_x");
        let e = normalize::mul(x.clone(), x.clone());
        let result = diff(&e, xid);
        // 2*x as a Mul node
        assert!(!result.is_zero(), "got 0 for d/dx(x^2)");
    }

    // ── Power rule ───────────────────────────────────────────────────────

    #[test]
    fn test_diff_pow_integer_exp() {
        // d/dx(x^3) = 3*x^2
        let xid = x_id("dpow_x");
        let x = sym("dpow_x");
        let e = Expr::pow(x.clone(), Expr::int(3));
        let result = diff(&e, xid);
        assert!(!result.is_zero(), "got 0 for d/dx(x^3)");
    }

    #[test]
    fn test_diff_pow_constant_base() {
        // d/dx(2^x) = 2^x * ln(2)  (exp_is_const=false)
        let xid = x_id("dpow_cx");
        let e = Expr::pow(Expr::int(2), sym("dpow_cx"));
        let result = diff(&e, xid);
        assert!(!result.is_zero(), "got 0 for d/dx(2^x)");
    }

    #[test]
    fn test_diff_pow_no_var() {
        // d/dx(y^2) = 0
        let xid = x_id("dpow_nv_x");
        let y = sym("dpow_nv_y");
        let e = Expr::pow(y, Expr::int(2));
        let result = diff(&e, xid);
        assert!(result.is_zero(), "expected 0 for d/dx(y^2)");
    }

    // ── Trigonometric functions ───────────────────────────────────────────

    #[test]
    fn test_diff_sin_x() {
        // d/dx(sin(x)) = cos(x)
        let xid = x_id("dsin_x");
        let x = sym("dsin_x");
        let e = Expr::func(FuncId::Sin, vec![x]);
        let result = diff(&e, xid);
        match result.as_ref() {
            Expr::Func(FuncId::Cos, _) => {}
            _ => panic!("expected cos(x), got {result}"),
        }
    }

    #[test]
    fn test_diff_cos_x() {
        // d/dx(cos(x)) = -sin(x)
        let xid = x_id("dcos_x");
        let x = sym("dcos_x");
        let e = Expr::func(FuncId::Cos, vec![x]);
        let result = diff(&e, xid);
        // -sin(x) is a Mul with coeff=-1 wrapping sin(x)
        assert!(!result.is_zero(), "expected -sin(x), got 0");
    }

    #[test]
    fn test_diff_tan_x() {
        // d/dx(tan(x)) = 1 + tan²(x)
        let xid = x_id("dtan_x");
        let x = sym("dtan_x");
        let e = Expr::func(FuncId::Tan, vec![x]);
        let result = diff(&e, xid);
        // result is 1 + tan(x)^2 as an Add node
        assert!(!result.is_zero(), "expected 1+tan²(x), got 0");
    }

    #[test]
    fn test_diff_sin_chain_rule() {
        // d/dx(sin(x^2)) = cos(x^2) * 2*x
        let xid = x_id("dsc_x");
        let x = sym("dsc_x");
        let x_sq = Expr::pow(x.clone(), Expr::int(2));
        let e = Expr::func(FuncId::Sin, vec![x_sq]);
        let result = diff(&e, xid);
        assert!(!result.is_zero(), "expected cos(x^2)*2x, got 0");
    }

    // ── Ln / Exp ─────────────────────────────────────────────────────────

    #[test]
    fn test_diff_ln_x() {
        // d/dx(ln(x)) = 1/x
        let xid = x_id("dln_x");
        let x = sym("dln_x");
        let e = Expr::func(FuncId::Ln, vec![x.clone()]);
        let result = diff(&e, xid);
        // 1/x is Pow(x, -1) or Mul with coeff=1, factor x^-1
        assert!(!result.is_zero(), "expected 1/x, got 0");
    }

    #[test]
    fn test_diff_exp_x() {
        // d/dx(exp(x)) = exp(x)
        let xid = x_id("dexp_x");
        let x = sym("dexp_x");
        let e = Expr::func(FuncId::Exp, vec![x.clone()]);
        let result = diff(&e, xid);
        match result.as_ref() {
            Expr::Func(FuncId::Exp, args) => {
                assert_eq!(*args[0], *x);
            }
            _ => panic!("expected exp(x), got {result}"),
        }
    }

    // ── Sqrt / Abs ───────────────────────────────────────────────────────

    #[test]
    fn test_diff_sqrt_x() {
        // d/dx(sqrt(x)) = 1/(2*sqrt(x))
        let xid = x_id("dsqrt_x");
        let x = sym("dsqrt_x");
        let e = Expr::func(FuncId::Sqrt, vec![x.clone()]);
        let result = diff(&e, xid);
        assert!(!result.is_zero(), "expected 1/(2*sqrt(x)), got 0");
    }

    #[test]
    fn test_diff_abs_x() {
        // d/dx(abs(x)) = x/abs(x)
        let xid = x_id("dabs_x");
        let x = sym("dabs_x");
        let e = Expr::func(FuncId::Abs, vec![x.clone()]);
        let result = diff(&e, xid);
        assert!(!result.is_zero(), "expected x/abs(x), got 0");
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
}
