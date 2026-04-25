//! Numerical evaluation of [`Expr`] given variable bindings.
//!
//! Traverses an expression tree and computes an `f64` result. Returns `None`
//! when evaluation is impossible — e.g., a variable is not in `bindings`, the
//! expression contains the imaginary unit, or a function receives an out-of-
//! domain argument (e.g., `sqrt(-1)`).

use std::collections::HashMap;

use crate::ast::SymbolicConstant;

use super::expr::{Expr, FuncId};
use super::SymbolId;

/// Evaluate an expression numerically, returning `None` if evaluation fails.
///
/// Failure conditions include:
/// - A [`Expr::Symbol`] whose [`SymbolId`] is absent from `bindings`.
/// - [`Expr::Constant`] with [`SymbolicConstant::I`] (imaginary unit).
/// - [`Expr::Complex`] with a non-zero imaginary part.
/// - A [`FuncId::Other`] function (unknown, no f64 implementation).
/// - Any recursive sub-expression that fails to evaluate.
///
/// # Examples
///
/// ```rust
/// use std::collections::HashMap;
/// use thales::numeric::{SymbolId, expr::Expr};
/// use thales::numeric::evaluation::evaluate;
///
/// let x_id = SymbolId::intern("x");
/// let expr = Expr::Pow(Expr::symbol("x"), Expr::int(2));
/// let mut bindings = HashMap::new();
/// bindings.insert(x_id, 3.0_f64);
/// assert!((evaluate(&expr, &bindings).unwrap() - 9.0).abs() < 1e-12);
/// ```
pub fn evaluate(expr: &Expr, bindings: &HashMap<SymbolId, f64>) -> Option<f64> {
    match expr {
        Expr::Integer(n) => {
            // Convert to i64 first; fall back to bigint string for huge values.
            if let Some(v) = n.to_i64() {
                Some(v as f64)
            } else {
                n.to_bigint().to_string().parse::<f64>().ok()
            }
        }

        Expr::Rational(r) => Some(r.to_f64()),

        Expr::Float(f) => Some(*f),

        Expr::Complex(c) => {
            if c.im == 0.0 {
                Some(c.re)
            } else {
                None
            }
        }

        Expr::Constant(c) => match c {
            SymbolicConstant::Pi => Some(std::f64::consts::PI),
            SymbolicConstant::E => Some(std::f64::consts::E),
            SymbolicConstant::I => None,
        },

        Expr::Symbol(s) => bindings.get(s).copied(),

        Expr::Add(node) => {
            let mut acc = node.constant.to_f64();
            for (term, coeff) in &node.terms {
                let term_val = evaluate(term, bindings)?;
                acc += coeff.to_f64() * term_val;
            }
            Some(acc)
        }

        Expr::Mul(node) => {
            let mut acc = node.coeff.to_f64();
            for (base, exp) in &node.factors {
                let base_val = evaluate(base, bindings)?;
                let exp_val = evaluate(exp, bindings)?;
                acc *= base_val.powf(exp_val);
            }
            Some(acc)
        }

        Expr::Pow(base, exp) => {
            let b = evaluate(base, bindings)?;
            let e = evaluate(exp, bindings)?;
            Some(b.powf(e))
        }

        Expr::Func(id, args) => eval_func(*id, args, bindings),
    }
}

/// Evaluate a built-in function given already-dispatched arguments.
fn eval_func(
    id: FuncId,
    args: &[std::sync::Arc<Expr>],
    bindings: &HashMap<SymbolId, f64>,
) -> Option<f64> {
    match id {
        // ── Single-argument functions ────────────────────────────────────────
        FuncId::Sin
        | FuncId::Cos
        | FuncId::Tan
        | FuncId::Asin
        | FuncId::Acos
        | FuncId::Atan
        | FuncId::Sinh
        | FuncId::Cosh
        | FuncId::Tanh
        | FuncId::Ln
        | FuncId::Exp
        | FuncId::Log2
        | FuncId::Log10
        | FuncId::Sqrt
        | FuncId::Cbrt
        | FuncId::Floor
        | FuncId::Ceil
        | FuncId::Round
        | FuncId::Abs
        | FuncId::Sign => {
            if args.len() != 1 {
                return None;
            }
            let x = evaluate(&args[0], bindings)?;
            Some(apply_unary(id, x))
        }

        // ── Two-argument functions ───────────────────────────────────────────
        FuncId::Log | FuncId::Atan2 | FuncId::Min | FuncId::Max => {
            if args.len() != 2 {
                return None;
            }
            let a = evaluate(&args[0], bindings)?;
            let b = evaluate(&args[1], bindings)?;
            Some(apply_binary(id, a, b))
        }

        // ── Complex-projection functions ─────────────────────────────────────
        // Re(z): for real args (or Expr::Complex), extract real part.
        FuncId::Re => {
            if args.len() != 1 {
                return None;
            }
            evaluate_as_real_part(&args[0], bindings)
        }

        // Im(z): for real args returns 0; for Expr::Complex extracts im.
        FuncId::Im => {
            if args.len() != 1 {
                return None;
            }
            evaluate_as_imag_part(&args[0], bindings)
        }

        // Conj(z): for real args returns the value unchanged (conjugate of real = real).
        FuncId::Conj => {
            if args.len() != 1 {
                return None;
            }
            // Conj of a real is itself; we can only return f64, so only handle real args.
            evaluate(&args[0], bindings)
        }

        // ── Unknown / user-defined ───────────────────────────────────────────
        FuncId::Other(_) => None,
    }
}

/// Extract the real part of an expression for f64 evaluation.
///
/// Returns `Some(re)` when the argument evaluates to a known real or complex literal.
/// Returns `None` for symbolic unknowns.
fn evaluate_as_real_part(
    arg: &std::sync::Arc<Expr>,
    bindings: &HashMap<SymbolId, f64>,
) -> Option<f64> {
    match arg.as_ref() {
        Expr::Complex(c) => Some(c.re),
        _ => evaluate(arg, bindings),
    }
}

/// Extract the imaginary part of an expression for f64 evaluation.
///
/// Returns `Some(im)` when the argument is a complex literal (extracts `im`)
/// or a purely real expression (returns `0.0`). Returns `None` for unknowns.
fn evaluate_as_imag_part(
    arg: &std::sync::Arc<Expr>,
    bindings: &HashMap<SymbolId, f64>,
) -> Option<f64> {
    match arg.as_ref() {
        Expr::Complex(c) => Some(c.im),
        _ => {
            // If the arg evaluates as a real f64, its imaginary part is 0.
            evaluate(arg, bindings).map(|_| 0.0)
        }
    }
}

/// Apply a unary built-in function. Only called for the single-argument variants.
#[inline]
fn apply_unary(id: FuncId, x: f64) -> f64 {
    match id {
        FuncId::Sin => x.sin(),
        FuncId::Cos => x.cos(),
        FuncId::Tan => x.tan(),
        FuncId::Asin => x.asin(),
        FuncId::Acos => x.acos(),
        FuncId::Atan => x.atan(),
        FuncId::Sinh => x.sinh(),
        FuncId::Cosh => x.cosh(),
        FuncId::Tanh => x.tanh(),
        FuncId::Ln => x.ln(),
        FuncId::Exp => x.exp(),
        FuncId::Log2 => x.log2(),
        FuncId::Log10 => x.log10(),
        FuncId::Sqrt => x.sqrt(),
        FuncId::Cbrt => x.cbrt(),
        FuncId::Floor => x.floor(),
        FuncId::Ceil => x.ceil(),
        FuncId::Round => x.round(),
        FuncId::Abs => x.abs(),
        FuncId::Sign => x.signum(),
        // Unreachable: caller only dispatches single-arg variants here.
        _ => unreachable!("apply_unary called with binary FuncId"),
    }
}

/// Apply a binary built-in function. Only called for the two-argument variants.
///
/// For [`FuncId::Log`], `a` is the value and `b` is the base: `log_b(a)`.
/// For [`FuncId::Atan2`], follows `f64::atan2(y, x)` convention: `a = y, b = x`.
#[inline]
fn apply_binary(id: FuncId, a: f64, b: f64) -> f64 {
    match id {
        FuncId::Log => a.log(b),
        FuncId::Atan2 => a.atan2(b),
        FuncId::Min => a.min(b),
        FuncId::Max => a.max(b),
        // Unreachable: caller only dispatches two-arg variants here.
        _ => unreachable!("apply_binary called with non-binary FuncId"),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{AddNode, BigRational, MulNode, SmallInt};
    use std::collections::HashMap;
    use std::sync::Arc;

    fn no_bindings() -> HashMap<SymbolId, f64> {
        HashMap::new()
    }

    fn bindings_x(v: f64) -> HashMap<SymbolId, f64> {
        let mut m = HashMap::new();
        m.insert(SymbolId::intern("eval_x"), v);
        m
    }

    // ── Leaf variants ────────────────────────────────────────────────────────

    #[test]
    fn test_integer_positive() {
        let e = Expr::Integer(SmallInt::from(7i64));
        assert_eq!(evaluate(&e, &no_bindings()), Some(7.0));
    }

    #[test]
    fn test_integer_negative() {
        let e = Expr::Integer(SmallInt::from(-3i64));
        assert_eq!(evaluate(&e, &no_bindings()), Some(-3.0));
    }

    #[test]
    fn test_integer_zero() {
        let e = Expr::Integer(SmallInt::from(0i64));
        assert_eq!(evaluate(&e, &no_bindings()), Some(0.0));
    }

    #[test]
    fn test_rational_half() {
        let e = Expr::Rational(BigRational::from_i64(1, 2));
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_rational_negative() {
        let e = Expr::Rational(BigRational::from_i64(-3, 4));
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - (-0.75)).abs() < 1e-15);
    }

    #[test]
    fn test_float_passthrough() {
        let e = Expr::Float(2.718);
        assert_eq!(evaluate(&e, &no_bindings()), Some(2.718));
    }

    #[test]
    fn test_float_nan() {
        let e = Expr::Float(f64::NAN);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!(v.is_nan());
    }

    #[test]
    fn test_complex_pure_real() {
        use num_complex::Complex64;
        let e = Expr::Complex(Complex64::new(5.0, 0.0));
        assert_eq!(evaluate(&e, &no_bindings()), Some(5.0));
    }

    #[test]
    fn test_complex_with_imag_returns_none() {
        use num_complex::Complex64;
        let e = Expr::Complex(Complex64::new(1.0, 2.0));
        assert_eq!(evaluate(&e, &no_bindings()), None);
    }

    #[test]
    fn test_constant_pi() {
        let e = Expr::Constant(SymbolicConstant::Pi);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn test_constant_e() {
        let e = Expr::Constant(SymbolicConstant::E);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - std::f64::consts::E).abs() < 1e-15);
    }

    #[test]
    fn test_constant_i_returns_none() {
        let e = Expr::Constant(SymbolicConstant::I);
        assert_eq!(evaluate(&e, &no_bindings()), None);
    }

    #[test]
    fn test_symbol_found() {
        let e = Expr::Symbol(SymbolId::intern("eval_x"));
        assert_eq!(evaluate(&e, &bindings_x(42.0)), Some(42.0));
    }

    #[test]
    fn test_symbol_missing_returns_none() {
        let e = Expr::Symbol(SymbolId::intern("eval_x"));
        assert_eq!(evaluate(&e, &no_bindings()), None);
    }

    // ── Compound expressions ────────────────────────────────────────────────

    #[test]
    fn test_pow_integer_exponent() {
        // 3^4 = 81
        let e = Expr::Pow(Expr::int(3), Expr::int(4));
        assert_eq!(evaluate(&e, &no_bindings()), Some(81.0));
    }

    #[test]
    fn test_pow_fractional_exponent() {
        // 4^0.5 = 2
        let base = Arc::new(Expr::Float(4.0));
        let exp = Arc::new(Expr::Float(0.5));
        let e = Expr::Pow(base, exp);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_pow_missing_variable_returns_none() {
        let e = Expr::Pow(Expr::symbol("eval_x"), Expr::int(2));
        assert_eq!(evaluate(&e, &no_bindings()), None);
    }

    #[test]
    fn test_add_node_constant_only() {
        // AddNode with just constant = 5
        let node = AddNode::from_constant(BigRational::from_i64(5, 1));
        let e = Expr::Add(node);
        assert_eq!(evaluate(&e, &no_bindings()), Some(5.0));
    }

    #[test]
    fn test_add_node_with_terms() {
        // 3 + 2*x  where x = 4  → 11
        let x = Expr::symbol("eval_x");
        let mut node = AddNode::from_constant(BigRational::from_i64(3, 1));
        node.add_term(x, BigRational::from_i64(2, 1));
        let e = Expr::Add(node);
        let v = evaluate(&e, &bindings_x(4.0)).unwrap();
        assert!((v - 11.0).abs() < 1e-12);
    }

    #[test]
    fn test_add_node_missing_variable_returns_none() {
        let x = Expr::symbol("eval_x");
        let mut node = AddNode::from_constant(BigRational::from_i64(1, 1));
        node.add_term(x, BigRational::from_i64(1, 1));
        let e = Expr::Add(node);
        assert_eq!(evaluate(&e, &no_bindings()), None);
    }

    #[test]
    fn test_mul_node_coeff_only() {
        // MulNode with just coeff = 7
        let node = MulNode::from_coeff(BigRational::from_i64(7, 1));
        let e = Expr::Mul(node);
        assert_eq!(evaluate(&e, &no_bindings()), Some(7.0));
    }

    #[test]
    fn test_mul_node_with_factors() {
        // 2 * x^3  where x = 2  → 2 * 8 = 16
        let x = Expr::symbol("eval_x");
        let mut node = MulNode::from_coeff(BigRational::from_i64(2, 1));
        node.add_factor(x, Expr::int(3));
        let e = Expr::Mul(node);
        let v = evaluate(&e, &bindings_x(2.0)).unwrap();
        assert!((v - 16.0).abs() < 1e-12);
    }

    #[test]
    fn test_mul_node_missing_variable_returns_none() {
        let x = Expr::symbol("eval_x");
        let node = MulNode::from_coeff_and_base(BigRational::from_i64(1, 1), x);
        let e = Expr::Mul(node);
        assert_eq!(evaluate(&e, &no_bindings()), None);
    }

    // ── Function evaluation ─────────────────────────────────────────────────

    #[test]
    fn test_func_sin() {
        let e = Expr::Func(FuncId::Sin, vec![Expr::float(0.0)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!(v.abs() < 1e-15);
    }

    #[test]
    fn test_func_cos() {
        let e = Expr::Func(FuncId::Cos, vec![Expr::float(0.0)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_func_tan() {
        let e = Expr::Func(FuncId::Tan, vec![Expr::float(0.0)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!(v.abs() < 1e-15);
    }

    #[test]
    fn test_func_asin() {
        let e = Expr::Func(FuncId::Asin, vec![Expr::float(1.0)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn test_func_acos() {
        let e = Expr::Func(FuncId::Acos, vec![Expr::float(1.0)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!(v.abs() < 1e-15);
    }

    #[test]
    fn test_func_atan() {
        let e = Expr::Func(FuncId::Atan, vec![Expr::float(1.0)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
    }

    #[test]
    fn test_func_sinh() {
        let e = Expr::Func(FuncId::Sinh, vec![Expr::float(0.0)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!(v.abs() < 1e-15);
    }

    #[test]
    fn test_func_cosh() {
        let e = Expr::Func(FuncId::Cosh, vec![Expr::float(0.0)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_func_tanh() {
        let e = Expr::Func(FuncId::Tanh, vec![Expr::float(0.0)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!(v.abs() < 1e-15);
    }

    #[test]
    fn test_func_ln() {
        let e = Expr::Func(FuncId::Ln, vec![Expr::e()]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_func_exp() {
        let e = Expr::Func(FuncId::Exp, vec![Expr::float(1.0)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - std::f64::consts::E).abs() < 1e-12);
    }

    #[test]
    fn test_func_log2() {
        let e = Expr::Func(FuncId::Log2, vec![Expr::float(8.0)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_func_log10() {
        let e = Expr::Func(FuncId::Log10, vec![Expr::float(1000.0)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_func_log_base() {
        // log base 2 of 8 = 3
        let e = Expr::Func(FuncId::Log, vec![Expr::float(8.0), Expr::float(2.0)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_func_sqrt() {
        let e = Expr::Func(FuncId::Sqrt, vec![Expr::float(9.0)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_func_cbrt() {
        let e = Expr::Func(FuncId::Cbrt, vec![Expr::float(27.0)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_func_abs() {
        let e = Expr::Func(FuncId::Abs, vec![Expr::float(-5.0)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_func_sign_positive() {
        let e = Expr::Func(FuncId::Sign, vec![Expr::float(3.0)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_func_sign_negative() {
        let e = Expr::Func(FuncId::Sign, vec![Expr::float(-7.0)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn test_func_floor() {
        let e = Expr::Func(FuncId::Floor, vec![Expr::float(2.9)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - 2.0).abs() < 1e-15);
    }

    #[test]
    fn test_func_ceil() {
        let e = Expr::Func(FuncId::Ceil, vec![Expr::float(2.1)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_func_round() {
        let e = Expr::Func(FuncId::Round, vec![Expr::float(2.5)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        // Rust rounds half away from zero → 3
        assert!((v - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_func_atan2() {
        // atan2(1, 1) = π/4
        let e = Expr::Func(FuncId::Atan2, vec![Expr::float(1.0), Expr::float(1.0)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
    }

    #[test]
    fn test_func_min() {
        let e = Expr::Func(FuncId::Min, vec![Expr::float(3.0), Expr::float(7.0)]);
        assert_eq!(evaluate(&e, &no_bindings()), Some(3.0));
    }

    #[test]
    fn test_func_max() {
        let e = Expr::Func(FuncId::Max, vec![Expr::float(3.0), Expr::float(7.0)]);
        assert_eq!(evaluate(&e, &no_bindings()), Some(7.0));
    }

    #[test]
    fn test_func_other_returns_none() {
        let e = Expr::Func(
            FuncId::Other(SymbolId::intern("my_fn")),
            vec![Expr::float(1.0)],
        );
        assert_eq!(evaluate(&e, &no_bindings()), None);
    }

    #[test]
    fn test_func_wrong_arity_returns_none() {
        // Sin expects 1 argument; give it 2.
        let e = Expr::Func(FuncId::Sin, vec![Expr::float(0.0), Expr::float(0.0)]);
        assert_eq!(evaluate(&e, &no_bindings()), None);
    }

    #[test]
    fn test_func_missing_variable_returns_none() {
        let e = Expr::Func(FuncId::Sin, vec![Expr::symbol("eval_x")]);
        assert_eq!(evaluate(&e, &no_bindings()), None);
    }

    // ── Nested / compound ────────────────────────────────────────────────────

    #[test]
    fn test_compound_sin_of_pi() {
        // sin(π) ≈ 0
        let e = Expr::Func(FuncId::Sin, vec![Expr::pi()]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!(v.abs() < 1e-14);
    }

    #[test]
    fn test_compound_exp_of_variable() {
        // exp(x) where x = 0 → 1
        let e = Expr::Func(FuncId::Exp, vec![Expr::symbol("eval_x")]);
        let v = evaluate(&e, &bindings_x(0.0)).unwrap();
        assert!((v - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_compound_pow_of_add() {
        // (1 + x)^2 where x = 2 → 9
        let x = Expr::symbol("eval_x");
        let mut add_node = AddNode::from_constant(BigRational::from_i64(1, 1));
        add_node.add_term(x, BigRational::from_i64(1, 1));
        let base = Arc::new(Expr::Add(add_node));
        let e = Expr::Pow(base, Expr::int(2));
        let v = evaluate(&e, &bindings_x(2.0)).unwrap();
        assert!((v - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_compound_nested_functions() {
        // sqrt(abs(-4)) = 2
        let inner = Expr::Func(FuncId::Abs, vec![Expr::float(-4.0)]);
        let e = Expr::Func(FuncId::Sqrt, vec![Arc::new(inner)]);
        let v = evaluate(&e, &no_bindings()).unwrap();
        assert!((v - 2.0).abs() < 1e-12);
    }
}
