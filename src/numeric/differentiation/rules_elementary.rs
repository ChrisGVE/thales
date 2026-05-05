//! Differentiation rules for elementary and built-in functions.

use std::sync::Arc;

use super::super::{
    expr::{Expr, FuncId},
    normalize,
};
use super::diff_arc;

// ── Differentiation of Func ──────────────────────────────────────────────────

pub(crate) fn diff_func(
    id: FuncId,
    args: &[Arc<Expr>],
    _full: &Arc<Expr>,
    var: super::super::SymbolId,
) -> Arc<Expr> {
    // Re/Im/Conj are linear functionals: d/dx Re(f) = Re(df/dx), etc.
    if args.len() == 1 {
        match id {
            FuncId::Re | FuncId::Im | FuncId::Conj => {
                let du = diff_arc(&args[0], var);
                return Expr::func(id, vec![du]);
            }
            _ => {}
        }
    }

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
pub(crate) fn diff_builtin_single(id: FuncId, u: &Arc<Expr>) -> Arc<Expr> {
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

        // d/du asin(u) = 1 / sqrt(1 - u²)
        FuncId::Asin => {
            let u_sq = normalize::pow(u.clone(), Expr::int(2));
            let one_minus_u_sq = normalize::sub(Expr::int(1), u_sq);
            let sqrt_inner = Expr::func(FuncId::Sqrt, vec![one_minus_u_sq]);
            normalize::div(Expr::int(1), sqrt_inner)
        }

        // d/du acos(u) = -1 / sqrt(1 - u²)
        FuncId::Acos => {
            let u_sq = normalize::pow(u.clone(), Expr::int(2));
            let one_minus_u_sq = normalize::sub(Expr::int(1), u_sq);
            let sqrt_inner = Expr::func(FuncId::Sqrt, vec![one_minus_u_sq]);
            normalize::neg(normalize::div(Expr::int(1), sqrt_inner))
        }

        // d/du atan(u) = 1 / (1 + u²)
        FuncId::Atan => {
            let u_sq = normalize::pow(u.clone(), Expr::int(2));
            let denom = normalize::add(Expr::int(1), u_sq);
            normalize::div(Expr::int(1), denom)
        }

        // d/du sinh(u) = cosh(u)
        FuncId::Sinh => Expr::func(FuncId::Cosh, vec![u.clone()]),

        // d/du cosh(u) = sinh(u)
        FuncId::Cosh => Expr::func(FuncId::Sinh, vec![u.clone()]),

        // d/du tanh(u) = 1 - tanh²(u)
        FuncId::Tanh => {
            let tanh_u = Expr::func(FuncId::Tanh, vec![u.clone()]);
            let tanh_sq = normalize::pow(tanh_u, Expr::int(2));
            normalize::sub(Expr::int(1), tanh_sq)
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

        // d/du cbrt(u) = 1 / (3 * u^(2/3))
        FuncId::Cbrt => {
            let exp = Expr::rational(2, 3);
            let u_pow = normalize::pow(u.clone(), exp);
            let three_u_pow = normalize::mul(Expr::int(3), u_pow);
            normalize::div(Expr::int(1), three_u_pow)
        }

        // d/du abs(u) = u / abs(u)
        FuncId::Abs => {
            let abs_u = Expr::func(FuncId::Abs, vec![u.clone()]);
            normalize::div(u.clone(), abs_u)
        }

        // Multi-argument functions handled in diff_func; not reachable here with 1 arg,
        // but treat as non-differentiable for safety.
        FuncId::Atan2 | FuncId::Log | FuncId::Min | FuncId::Max => Expr::int(0),

        // Logarithm variants (single-arg: natural log is Ln; these are non-standard)
        FuncId::Log2 => normalize::div(
            Expr::int(1),
            normalize::mul(u.clone(), Expr::float(std::f64::consts::LN_2)),
        ),
        FuncId::Log10 => normalize::div(
            Expr::int(1),
            normalize::mul(u.clone(), Expr::float(std::f64::consts::LN_10)),
        ),

        // Rounding and sign: not classically differentiable; return 0
        FuncId::Floor | FuncId::Ceil | FuncId::Round | FuncId::Sign => Expr::int(0),

        // Re/Im/Conj intercepted by diff_func — unreachable at runtime via this path.
        FuncId::Re | FuncId::Im | FuncId::Conj => Expr::int(0),

        // Special functions: derivatives not yet implemented; return opaque zero.
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
        | FuncId::DiracDelta => Expr::int(0),

        // Unknown/user function: produce opaque derivative marker (zero — caller
        // should not use the result as authoritative)
        FuncId::Other(_) => Expr::int(0),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::diff;
    use crate::numeric::{
        expr::{Expr, FuncId},
        SymbolId,
    };
    use std::sync::Arc;

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }
    fn x_id(name: &str) -> SymbolId {
        SymbolId::intern(name)
    }

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

    #[test]
    fn test_diff_asin_x() {
        // d/dx(asin(x)) = 1 / sqrt(1 - x^2)
        let xid = x_id("dasin_x");
        let x = sym("dasin_x");
        let e = Expr::func(FuncId::Asin, vec![x.clone()]);
        let result = diff(&e, xid);
        assert!(!result.is_zero(), "expected 1/sqrt(1-x^2), got 0");
    }

    #[test]
    fn test_diff_acos_x() {
        // d/dx(acos(x)) = -1 / sqrt(1 - x^2)
        let xid = x_id("dacos_x");
        let x = sym("dacos_x");
        let e = Expr::func(FuncId::Acos, vec![x.clone()]);
        let result = diff(&e, xid);
        assert!(!result.is_zero(), "expected -1/sqrt(1-x^2), got 0");
    }

    #[test]
    fn test_diff_atan_x() {
        // d/dx(atan(x)) = 1 / (1 + x^2)
        let xid = x_id("datan_x");
        let x = sym("datan_x");
        let e = Expr::func(FuncId::Atan, vec![x.clone()]);
        let result = diff(&e, xid);
        assert!(!result.is_zero(), "expected 1/(1+x^2), got 0");
    }

    #[test]
    fn test_diff_sinh_x() {
        // d/dx(sinh(x)) = cosh(x)
        let xid = x_id("dsinh_x");
        let x = sym("dsinh_x");
        let e = Expr::func(FuncId::Sinh, vec![x.clone()]);
        let result = diff(&e, xid);
        match result.as_ref() {
            Expr::Func(FuncId::Cosh, _) => {}
            _ => panic!("expected cosh(x), got {result}"),
        }
    }

    #[test]
    fn test_diff_cosh_x() {
        // d/dx(cosh(x)) = sinh(x)
        let xid = x_id("dcosh_x");
        let x = sym("dcosh_x");
        let e = Expr::func(FuncId::Cosh, vec![x.clone()]);
        let result = diff(&e, xid);
        match result.as_ref() {
            Expr::Func(FuncId::Sinh, _) => {}
            _ => panic!("expected sinh(x), got {result}"),
        }
    }

    #[test]
    fn test_diff_tanh_x() {
        // d/dx(tanh(x)) = 1 - tanh^2(x)
        let xid = x_id("dtanh_x");
        let x = sym("dtanh_x");
        let e = Expr::func(FuncId::Tanh, vec![x.clone()]);
        let result = diff(&e, xid);
        assert!(!result.is_zero(), "expected 1 - tanh^2(x), got 0");
    }

    #[test]
    fn test_diff_cbrt_x() {
        // d/dx(cbrt(x)) = 1 / (3 * x^(2/3))
        let xid = x_id("dcbrt_x");
        let x = sym("dcbrt_x");
        let e = Expr::func(FuncId::Cbrt, vec![x.clone()]);
        let result = diff(&e, xid);
        assert!(!result.is_zero(), "expected 1/(3*x^(2/3)), got 0");
    }

    #[test]
    fn test_diff_asin_const_inner() {
        // d/dx(asin(5)) = 0  (chain rule: outer * 0)
        let xid = x_id("dasin_c");
        let e = Expr::func(FuncId::Asin, vec![Expr::int(5)]);
        let result = diff(&e, xid);
        assert!(
            result.is_zero(),
            "expected 0 for d/dx(asin(5)), got {result}"
        );
    }
}
