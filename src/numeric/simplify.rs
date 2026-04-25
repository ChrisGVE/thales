//! Post-construction algebraic simplification for [`Expr`].
//!
//! [`simplify`] applies algebraic identities that are not handled at
//! construction time by the smart constructors in [`normalize`]:
//!
//! - Power identities: `x^0 → 1`, `x^1 → x`, `0^n → 0`, `1^n → 1`,
//!   `a^b → eval` when both are small integers.
//! - Known-value function evaluation: `sin(0) → 0`, `cos(0) → 1`,
//!   `ln(1) → 0`, `exp(0) → 1`, `sqrt(0) → 0`, `sqrt(1) → 1`,
//!   `abs(x) → x` when x is a non-negative numeric.
//! - Recursive child simplification before each node rule.
//! - Add/Mul rebuild: simplify children then reconstruct via [`normalize`]
//!   so that like-term merging and identity removal apply to the result.

use std::sync::Arc;

use super::expr::{Expr, FuncId};
use super::normalize;
use super::{AddNode, BigRational, MulNode};
use num::traits::{One, Signed, Zero};

// ── Public API ───────────────────────────────────────────────────────────────

/// Simplify an expression by recursively applying algebraic identities.
///
/// The function first recurses into children, then applies node-level rules.
/// It is idempotent: calling it twice on an already-simplified expression
/// returns a structurally equal result.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{simplify::simplify, expr::{Expr, FuncId}};
///
/// // sin(0) → 0
/// let sin_zero = Expr::func(FuncId::Sin, vec![Expr::int(0)]);
/// let result = simplify(&sin_zero);
/// assert!(result.is_zero());
///
/// // x^1 → x
/// let x = Expr::symbol("x");
/// let x_pow_1 = Expr::pow(x.clone(), Expr::int(1));
/// let result = simplify(&x_pow_1);
/// assert_eq!(*result, *x);
/// ```
pub fn simplify(expr: &Arc<Expr>) -> Arc<Expr> {
    match expr.as_ref() {
        Expr::Integer(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Complex(_)
        | Expr::Constant(_)
        | Expr::Symbol(_) => expr.clone(),

        Expr::Pow(base, exp) => simplify_pow(base, exp),

        Expr::Func(id, args) => simplify_func(*id, args),

        Expr::Add(node) => simplify_add(node),

        Expr::Mul(node) => simplify_mul(node),
    }
}

// ── Pow simplification ───────────────────────────────────────────────────────

fn simplify_pow(base: &Arc<Expr>, exp: &Arc<Expr>) -> Arc<Expr> {
    let base = simplify(base);
    let exp = simplify(exp);

    // Delegate to the normalize smart constructor which encodes all power rules.
    // normalize::pow handles: x^0→1, x^1→x, 0^n→0, 1^n→1,
    // integer^integer evaluation, (a^b)^c→a^(b*c), and Mul distribution.
    normalize::pow(base, exp)
}

// ── Func simplification ──────────────────────────────────────────────────────

fn simplify_func(id: FuncId, args: &[Arc<Expr>]) -> Arc<Expr> {
    // Recursively simplify each argument first.
    let simplified: Vec<Arc<Expr>> = args.iter().map(simplify).collect();

    // Dispatch Re/Im/Conj to their dedicated simplifier.
    if matches!(id, FuncId::Re | FuncId::Im | FuncId::Conj) {
        return super::complex_simplify::simplify_complex_func(id, &simplified);
    }

    match id {
        // ── Single-argument known-value rules ────────────────────────────────
        FuncId::Sin if simplified.len() == 1 && simplified[0].is_zero() => Expr::int(0),
        FuncId::Cos if simplified.len() == 1 && simplified[0].is_zero() => Expr::int(1),
        FuncId::Tan if simplified.len() == 1 && simplified[0].is_zero() => Expr::int(0),
        FuncId::Sinh if simplified.len() == 1 && simplified[0].is_zero() => Expr::int(0),
        FuncId::Cosh if simplified.len() == 1 && simplified[0].is_zero() => Expr::int(1),
        FuncId::Tanh if simplified.len() == 1 && simplified[0].is_zero() => Expr::int(0),

        FuncId::Ln if simplified.len() == 1 && simplified[0].is_one() => Expr::int(0),
        FuncId::Log2 if simplified.len() == 1 && simplified[0].is_one() => Expr::int(0),
        FuncId::Log10 if simplified.len() == 1 && simplified[0].is_one() => Expr::int(0),

        FuncId::Exp if simplified.len() == 1 && simplified[0].is_zero() => Expr::int(1),

        FuncId::Sqrt if simplified.len() == 1 && simplified[0].is_zero() => Expr::int(0),
        FuncId::Sqrt if simplified.len() == 1 && simplified[0].is_one() => Expr::int(1),

        FuncId::Cbrt if simplified.len() == 1 && simplified[0].is_zero() => Expr::int(0),
        FuncId::Cbrt if simplified.len() == 1 && simplified[0].is_one() => Expr::int(1),

        FuncId::Abs if simplified.len() == 1 => simplify_abs(&simplified[0]),

        FuncId::Sign if simplified.len() == 1 => simplify_sign(&simplified[0]),

        // Floor/Ceil/Round on integer → identity
        FuncId::Floor | FuncId::Ceil | FuncId::Round
            if simplified.len() == 1 && is_integer_value(&simplified[0]) =>
        {
            simplified[0].clone()
        }

        // No rule matched: rebuild with simplified arguments.
        _ => Arc::new(Expr::Func(id, simplified)),
    }
}

/// Simplify `abs(x)` when x is a known non-negative numeric.
fn simplify_abs(arg: &Arc<Expr>) -> Arc<Expr> {
    match arg.as_ref() {
        Expr::Integer(n) => Arc::new(Expr::Integer(n.abs())),
        Expr::Rational(r) => {
            if r.is_negative() {
                Arc::new(Expr::Rational(-r))
            } else {
                arg.clone()
            }
        }
        Expr::Float(f) => Arc::new(Expr::Float(f.abs())),
        _ => Arc::new(Expr::Func(FuncId::Abs, vec![arg.clone()])),
    }
}

/// Simplify `sign(x)` when x is a known numeric.
fn simplify_sign(arg: &Arc<Expr>) -> Arc<Expr> {
    match arg.as_ref() {
        Expr::Integer(n) => {
            if n.is_zero() {
                Expr::int(0)
            } else if n.is_positive() {
                Expr::int(1)
            } else {
                Expr::int(-1)
            }
        }
        Expr::Rational(r) => {
            if r.is_zero() {
                Expr::int(0)
            } else if r.is_positive() {
                Expr::int(1)
            } else {
                Expr::int(-1)
            }
        }
        Expr::Float(f) => {
            if *f == 0.0 {
                Expr::int(0)
            } else if *f > 0.0 {
                Expr::int(1)
            } else {
                Expr::int(-1)
            }
        }
        _ => Arc::new(Expr::Func(FuncId::Sign, vec![arg.clone()])),
    }
}

/// Returns true when expr is an exact integer (Integer or Rational with denom 1).
fn is_integer_value(e: &Expr) -> bool {
    match e {
        Expr::Integer(_) => true,
        Expr::Rational(r) => r.is_integer(),
        _ => false,
    }
}

// ── Add simplification ───────────────────────────────────────────────────────

fn simplify_add(node: &AddNode) -> Arc<Expr> {
    // Simplify each term, then rebuild via normalize::add_many so all
    // construction-time rules (like-term merging, identity removal) apply.
    let mut terms: Vec<Arc<Expr>> = Vec::with_capacity(node.terms.len() + 1);

    // Constant part
    if !node.constant.is_zero() {
        terms.push(rational_to_expr(node.constant.clone()));
    }

    // Scaled terms: coeff * term
    for (term, coeff) in &node.terms {
        let simplified_term = simplify(term);
        let scaled = if coeff.is_one() {
            simplified_term
        } else {
            normalize::mul(rational_to_expr(coeff.clone()), simplified_term)
        };
        terms.push(scaled);
    }

    normalize::add_many(terms)
}

// ── Mul simplification ───────────────────────────────────────────────────────

fn simplify_mul(node: &MulNode) -> Arc<Expr> {
    // Simplify each factor (base^exp), then rebuild via normalize::mul so
    // all construction-time rules (zero annihilation, coefficient absorption,
    // like-base merging) apply.
    let mut result = rational_to_expr(node.coeff.clone());

    for (base, exp) in &node.factors {
        let simplified_base = simplify(base);
        let simplified_exp = simplify(exp);
        let factor = normalize::pow(simplified_base, simplified_exp);
        result = normalize::mul(result, factor);
    }

    result
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn rational_to_expr(r: BigRational) -> Arc<Expr> {
    if r.is_integer() {
        Arc::new(Expr::Integer(r.numer().clone()))
    } else {
        Arc::new(Expr::Rational(r))
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::expr::{Expr, FuncId};
    use crate::numeric::SmallInt;

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }

    // ── Power rules ──────────────────────────────────────────────────────────

    #[test]
    fn test_pow_zero_exp() {
        // x^0 → 1
        let x = sym("simp_pow_x0");
        let e = Expr::pow(x, Expr::int(0));
        let result = simplify(&e);
        assert_eq!(*result, Expr::Integer(SmallInt::from(1i64)));
    }

    #[test]
    fn test_pow_one_exp() {
        // x^1 → x
        let x = sym("simp_pow_x1");
        let e = Expr::pow(x.clone(), Expr::int(1));
        let result = simplify(&e);
        assert_eq!(*result, *x);
    }

    #[test]
    fn test_pow_zero_base_positive_exp() {
        // 0^n → 0  (n > 0)
        let e = Expr::pow(Expr::int(0), Expr::int(3));
        let result = simplify(&e);
        assert!(result.is_zero());
    }

    #[test]
    fn test_pow_one_base() {
        // 1^n → 1
        let e = Expr::pow(Expr::int(1), Expr::int(100));
        let result = simplify(&e);
        assert!(result.is_one());
    }

    #[test]
    fn test_pow_integer_constant_fold() {
        // 2^3 → 8
        let e = Expr::pow(Expr::int(2), Expr::int(3));
        let result = simplify(&e);
        assert_eq!(*result, Expr::Integer(SmallInt::from(8i64)));
    }

    #[test]
    fn test_pow_negative_integer_exp() {
        // 2^(-1) → 1/2
        let e = Expr::pow(Expr::int(2), Expr::int(-1));
        let result = simplify(&e);
        match result.as_ref() {
            Expr::Rational(r) => {
                assert_eq!(*r, BigRational::from_i64(1, 2));
            }
            _ => panic!("expected Rational, got {result}"),
        }
    }

    #[test]
    fn test_pow_recursive_simplification() {
        // (sin(0))^2 → 0^2 → 0
        let sin_zero = Expr::func(FuncId::Sin, vec![Expr::int(0)]);
        let e = Expr::pow(sin_zero, Expr::int(2));
        let result = simplify(&e);
        assert!(result.is_zero());
    }

    // ── Function evaluation rules ─────────────────────────────────────────────

    #[test]
    fn test_sin_zero() {
        let e = Expr::func(FuncId::Sin, vec![Expr::int(0)]);
        assert!(simplify(&e).is_zero());
    }

    #[test]
    fn test_cos_zero() {
        let e = Expr::func(FuncId::Cos, vec![Expr::int(0)]);
        assert!(simplify(&e).is_one());
    }

    #[test]
    fn test_tan_zero() {
        let e = Expr::func(FuncId::Tan, vec![Expr::int(0)]);
        assert!(simplify(&e).is_zero());
    }

    #[test]
    fn test_sinh_zero() {
        let e = Expr::func(FuncId::Sinh, vec![Expr::int(0)]);
        assert!(simplify(&e).is_zero());
    }

    #[test]
    fn test_cosh_zero() {
        let e = Expr::func(FuncId::Cosh, vec![Expr::int(0)]);
        assert!(simplify(&e).is_one());
    }

    #[test]
    fn test_tanh_zero() {
        let e = Expr::func(FuncId::Tanh, vec![Expr::int(0)]);
        assert!(simplify(&e).is_zero());
    }

    #[test]
    fn test_ln_one() {
        let e = Expr::func(FuncId::Ln, vec![Expr::int(1)]);
        assert!(simplify(&e).is_zero());
    }

    #[test]
    fn test_log2_one() {
        let e = Expr::func(FuncId::Log2, vec![Expr::int(1)]);
        assert!(simplify(&e).is_zero());
    }

    #[test]
    fn test_log10_one() {
        let e = Expr::func(FuncId::Log10, vec![Expr::int(1)]);
        assert!(simplify(&e).is_zero());
    }

    #[test]
    fn test_exp_zero() {
        let e = Expr::func(FuncId::Exp, vec![Expr::int(0)]);
        assert!(simplify(&e).is_one());
    }

    #[test]
    fn test_sqrt_zero() {
        let e = Expr::func(FuncId::Sqrt, vec![Expr::int(0)]);
        assert!(simplify(&e).is_zero());
    }

    #[test]
    fn test_sqrt_one() {
        let e = Expr::func(FuncId::Sqrt, vec![Expr::int(1)]);
        assert!(simplify(&e).is_one());
    }

    #[test]
    fn test_cbrt_zero() {
        let e = Expr::func(FuncId::Cbrt, vec![Expr::int(0)]);
        assert!(simplify(&e).is_zero());
    }

    #[test]
    fn test_cbrt_one() {
        let e = Expr::func(FuncId::Cbrt, vec![Expr::int(1)]);
        assert!(simplify(&e).is_one());
    }

    // ── Abs rules ────────────────────────────────────────────────────────────

    #[test]
    fn test_abs_positive_integer() {
        let e = Expr::func(FuncId::Abs, vec![Expr::int(5)]);
        let result = simplify(&e);
        assert_eq!(*result, Expr::Integer(SmallInt::from(5i64)));
    }

    #[test]
    fn test_abs_negative_integer() {
        let e = Expr::func(FuncId::Abs, vec![Expr::int(-7)]);
        let result = simplify(&e);
        assert_eq!(*result, Expr::Integer(SmallInt::from(7i64)));
    }

    #[test]
    fn test_abs_zero() {
        let e = Expr::func(FuncId::Abs, vec![Expr::int(0)]);
        let result = simplify(&e);
        assert!(result.is_zero());
    }

    #[test]
    fn test_abs_negative_rational() {
        let e = Expr::func(FuncId::Abs, vec![Expr::rational(-3, 4)]);
        let result = simplify(&e);
        assert_eq!(*result, Expr::Rational(BigRational::from_i64(3, 4)));
    }

    #[test]
    fn test_abs_positive_float() {
        let e = Expr::func(FuncId::Abs, vec![Expr::float(2.5)]);
        let result = simplify(&e);
        assert_eq!(*result, Expr::Float(2.5));
    }

    #[test]
    fn test_abs_negative_float() {
        let e = Expr::func(FuncId::Abs, vec![Expr::float(-2.5)]);
        let result = simplify(&e);
        assert_eq!(*result, Expr::Float(2.5));
    }

    #[test]
    fn test_abs_symbolic_stays() {
        let x = sym("simp_abs_sym");
        let e = Expr::func(FuncId::Abs, vec![x.clone()]);
        let result = simplify(&e);
        match result.as_ref() {
            Expr::Func(FuncId::Abs, args) => assert_eq!(*args[0], *x),
            _ => panic!("expected Func(Abs, x), got {result}"),
        }
    }

    // ── Sign rules ───────────────────────────────────────────────────────────

    #[test]
    fn test_sign_positive() {
        let e = Expr::func(FuncId::Sign, vec![Expr::int(5)]);
        assert!(simplify(&e).is_one());
    }

    #[test]
    fn test_sign_negative() {
        let e = Expr::func(FuncId::Sign, vec![Expr::int(-3)]);
        let result = simplify(&e);
        assert_eq!(*result, Expr::Integer(SmallInt::from(-1i64)));
    }

    #[test]
    fn test_sign_zero() {
        let e = Expr::func(FuncId::Sign, vec![Expr::int(0)]);
        assert!(simplify(&e).is_zero());
    }

    // ── Floor/Ceil/Round on integer ──────────────────────────────────────────

    #[test]
    fn test_floor_integer_identity() {
        let e = Expr::func(FuncId::Floor, vec![Expr::int(7)]);
        let result = simplify(&e);
        assert_eq!(*result, Expr::Integer(SmallInt::from(7i64)));
    }

    #[test]
    fn test_ceil_integer_identity() {
        let e = Expr::func(FuncId::Ceil, vec![Expr::int(-3)]);
        let result = simplify(&e);
        assert_eq!(*result, Expr::Integer(SmallInt::from(-3i64)));
    }

    #[test]
    fn test_round_integer_identity() {
        let e = Expr::func(FuncId::Round, vec![Expr::int(0)]);
        let result = simplify(&e);
        assert!(result.is_zero());
    }

    // ── Add/Mul recursive simplification ─────────────────────────────────────

    #[test]
    fn test_add_simplifies_children() {
        // (sin(0) + cos(0)) → 0 + 1 → 1
        let sin_zero = Expr::func(FuncId::Sin, vec![Expr::int(0)]);
        let cos_zero = Expr::func(FuncId::Cos, vec![Expr::int(0)]);
        let e = normalize::add(sin_zero, cos_zero);
        let result = simplify(&e);
        assert!(result.is_one());
    }

    #[test]
    fn test_mul_simplifies_children() {
        // cos(0) * 5 → 1 * 5 → 5
        let cos_zero = Expr::func(FuncId::Cos, vec![Expr::int(0)]);
        let e = normalize::mul(cos_zero, Expr::int(5));
        let result = simplify(&e);
        assert_eq!(*result, Expr::Integer(SmallInt::from(5i64)));
    }

    #[test]
    fn test_mul_with_zero_child() {
        // sin(0) * x → 0 * x → 0
        let sin_zero = Expr::func(FuncId::Sin, vec![Expr::int(0)]);
        let x = sym("simp_mzc_x");
        let e = normalize::mul(sin_zero, x);
        let result = simplify(&e);
        assert!(result.is_zero());
    }

    // ── Leaf nodes pass through unchanged ────────────────────────────────────

    #[test]
    fn test_integer_passthrough() {
        let e = Expr::int(42);
        let result = simplify(&e);
        assert_eq!(*result, Expr::Integer(SmallInt::from(42i64)));
    }

    #[test]
    fn test_symbol_passthrough() {
        let x = sym("simp_sym_pass");
        let result = simplify(&x);
        assert_eq!(*result, *x);
    }

    #[test]
    fn test_float_passthrough() {
        let e = Expr::float(3.14);
        let result = simplify(&e);
        assert_eq!(*result, Expr::Float(3.14));
    }

    // ── Idempotency ──────────────────────────────────────────────────────────

    #[test]
    fn test_simplify_idempotent_pow() {
        let x = sym("simp_idem_x");
        let e = Expr::pow(x, Expr::int(2));
        let once = simplify(&e);
        let twice = simplify(&once);
        assert_eq!(*once, *twice);
    }

    #[test]
    fn test_simplify_idempotent_func() {
        let x = sym("simp_idem_fx");
        let e = Expr::func(FuncId::Sin, vec![x]);
        let once = simplify(&e);
        let twice = simplify(&once);
        assert_eq!(*once, *twice);
    }
}
