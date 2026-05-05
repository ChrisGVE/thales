//! Inverse Laplace transform: symbolic table lookup and algebraic properties.
//!
//! Implements `L^{-1}{F(s)} = f(t)` for elementary rational and transcendental
//! forms via a direct pattern table plus recursive linearity decomposition.
//!
//! # Table entries
//!
//! | `F(s)`              | `f(t)`                      |
//! |---------------------|-----------------------------|
//! | 1/s                 | 1                           |
//! | 1/s^n               | t^(n-1)/(n-1)!              |
//! | 1/(s-a)             | e^(a·t)                     |
//! | 1/(s-a)^n           | t^(n-1)·e^(a·t)/(n-1)!     |
//! | ω/(s²+ω²)           | sin(ω·t)                    |
//! | s/(s²+ω²)           | cos(ω·t)                    |
//! | a/(s²-a²)           | sinh(a·t)                   |
//! | s/(s²-a²)           | cosh(a·t)                   |

use std::sync::Arc;

use crate::numeric::{
    expr::{Expr, FuncId},
    normalize, SymbolId,
};

use super::{as_constant, contains_var, split_linear_terms, TransformError, TransformResult};

// ── Public entry point ────────────────────────────────────────────────────────

/// Compute the inverse Laplace transform of `expr` with respect to `s_var`,
/// returning a result in the `t_var` (time) domain.
///
/// Uses table lookup for elementary forms and the linearity property
/// `L^{-1}{a·F + b·G} = a·L^{-1}{F} + b·L^{-1}{G}`.
pub fn inverse_laplace(
    expr: &Arc<Expr>,
    s_var: SymbolId,
    t_var: SymbolId,
) -> Result<TransformResult, TransformError> {
    let mut steps: Vec<String> = Vec::new();
    let result_expr = invert_expr(expr, s_var, t_var, &mut steps)?;

    Ok(TransformResult {
        expr: result_expr,
        domain_var: t_var.as_str().to_owned(),
        convergence: None,
        steps,
    })
}

// ── Core dispatch ─────────────────────────────────────────────────────────────

fn invert_expr(
    expr: &Arc<Expr>,
    s_var: SymbolId,
    t_var: SymbolId,
    steps: &mut Vec<String>,
) -> Result<Arc<Expr>, TransformError> {
    // 1. Direct table match.
    if let Some(result) = table_lookup(expr, s_var, t_var, steps) {
        return result;
    }

    // 2. Linearity over AddNode.
    if matches!(expr.as_ref(), Expr::Add(_)) {
        return apply_linearity(expr, s_var, t_var, steps);
    }

    // 3. Scaled single term: c·F(s).
    if let Expr::Mul(_) = expr.as_ref() {
        if let Some(result) = try_scaled_term(expr, s_var, t_var, steps) {
            return result;
        }
    }

    Err(TransformError::NoTableEntry(format!("{expr}")))
}

// ── Table lookup ──────────────────────────────────────────────────────────────

/// Try to match `expr` against a known inverse-Laplace table entry.
fn table_lookup(
    expr: &Arc<Expr>,
    s_var: SymbolId,
    t_var: SymbolId,
    steps: &mut Vec<String>,
) -> Option<Result<Arc<Expr>, TransformError>> {
    // All table entries are of the form (something)^(-1) or a product involving
    // (something)^(-1).  Dispatch based on structure.
    match expr.as_ref() {
        // ── Pure Pow: base^exp ────────────────────────────────────────────────
        Expr::Pow(base, exp) => match_pow_entry(base, exp, s_var, t_var, steps),

        // ── Mul: c · base^exp  or  s · denom^(-1) ────────────────────────────
        Expr::Mul(node) => match_mul_entry(expr, node, s_var, t_var, steps),

        _ => None,
    }
}

// ── Pow entry ─────────────────────────────────────────────────────────────────

/// Match `base^exp` where `exp` is a negative integer → various 1/f(s) forms.
fn match_pow_entry(
    base: &Arc<Expr>,
    exp: &Arc<Expr>,
    s_var: SymbolId,
    t_var: SymbolId,
    steps: &mut Vec<String>,
) -> Option<Result<Arc<Expr>, TransformError>> {
    // Only negative integer exponents represent denominators.
    let neg_n = match as_constant(exp) {
        Some(v) if v < 0.0 && v.fract() == 0.0 && v >= -20.0 => -(v as i64) as u64,
        _ => return None,
    };
    // base must involve s_var
    if !contains_var(base, s_var) {
        return None;
    }
    // n = order of the pole
    let n = neg_n;
    match_denom_pattern(base, n, s_var, t_var, steps)
}

/// Given `denom^(-n)`, classify the denominator and build the t-domain result.
///
/// Handles:
/// - `s^n`       → `1/s^n`     → `t^(n-1)/(n-1)!`
/// - `(s-a)^n`   → `1/(s-a)^n` → `t^(n-1) e^(at) / (n-1)!`
/// - `s²+ω²`     → sin/cos (only for n=1; numerator distinguishes them)
/// - `s²-a²`     → sinh/cosh (only for n=1)
fn match_denom_pattern(
    denom: &Arc<Expr>,
    n: u64,
    s_var: SymbolId,
    t_var: SymbolId,
    steps: &mut Vec<String>,
) -> Option<Result<Arc<Expr>, TransformError>> {
    // ── Case 1: denom = s (pure symbol) ──────────────────────────────────────
    if is_s(denom, s_var) {
        return Some(match_one_over_s_n(n, t_var, steps));
    }

    // ── Case 2: denom = (s - a)  i.e. AddNode{s + const(-a)} ────────────────
    if let Some(a) = extract_shift(denom, s_var) {
        return Some(match_shifted_pole(a, n, t_var, steps));
    }

    // ── Cases 3 & 4: denom = s² ± ω² (only for n = 1) ───────────────────────
    // These require a numerator so we cannot handle them from a bare Pow.
    // We fall through to the Mul matcher which supplies the numerator.
    let _ = (n, t_var, steps);
    None
}

// ── Mul entry ─────────────────────────────────────────────────────────────────

/// Match `MulNode` terms that encode `ω/denom` or `s/denom` patterns.
fn match_mul_entry(
    _expr: &Arc<Expr>,
    node: &crate::numeric::MulNode,
    s_var: SymbolId,
    t_var: SymbolId,
    steps: &mut Vec<String>,
) -> Option<Result<Arc<Expr>, TransformError>> {
    // Strategy: collect all (base, exp) pairs where exp is a negative integer
    // and the base involves s_var.  Separately track numeric coefficient and
    // any "numerator" factor (base with positive exponent).

    let coeff = node.coeff.to_f64();

    // Partition factors into denominator bases (exp < 0) and numerator parts.
    let mut denom_base: Option<(Arc<Expr>, u64)> = None; // (base, |exp|)
    let mut num_is_s = false; // numerator is the bare `s` symbol

    for (base, exp) in &node.factors {
        match as_constant(exp) {
            Some(v) if v < 0.0 && v.fract() == 0.0 => {
                // Denominator factor
                if denom_base.is_some() {
                    return None; // multiple denominator factors — not a table form
                }
                denom_base = Some((Arc::clone(base), -(v as i64) as u64));
            }
            Some(v) if (v - 1.0).abs() < f64::EPSILON => {
                // Exponent = 1 → numerator factor
                if is_s(base, s_var) {
                    num_is_s = true;
                } else {
                    return None; // unexpected symbolic numerator
                }
            }
            _ => return None,
        }
    }

    let (denom, n) = denom_base?;

    // ── Numerator = s, n = 1 ─────────────────────────────────────────────────
    if num_is_s && n == 1 {
        // s / (s² + ω²) → cos or  s / (s² - a²) → cosh
        if let Some((omega, kind)) = extract_quadratic(denom.as_ref(), s_var) {
            let t = Arc::new(Expr::Symbol(t_var));
            let omega_t = normalize::mul(Expr::float(coeff * omega), Arc::clone(&t));
            let result = match kind {
                QuadKind::Plus => {
                    steps.push(format!(
                        "Applied L^{{-1}}{{s/(s²+ω²)}} = cos(ω·t) with ω = {omega}, scaled by {coeff}"
                    ));
                    Arc::new(Expr::Func(FuncId::Cos, vec![omega_t]))
                }
                QuadKind::Minus => {
                    steps.push(format!(
                        "Applied L^{{-1}}{{s/(s²-a²)}} = cosh(a·t) with a = {omega}, scaled by {coeff}"
                    ));
                    Arc::new(Expr::Func(FuncId::Cosh, vec![omega_t]))
                }
            };
            return Some(Ok(result));
        }
        return None;
    }

    // ── No explicit numerator (coeff only) ───────────────────────────────────
    if !num_is_s {
        // coeff / denom^n  or  coeff / s^n  etc.

        // coeff / (s² ± a²) — sin or sinh form (n must be 1)
        if n == 1 {
            if let Some((omega, kind)) = extract_quadratic(denom.as_ref(), s_var) {
                let t = Arc::new(Expr::Symbol(t_var));
                let omega_t = normalize::mul(Expr::float(omega), Arc::clone(&t));
                // coeff / (s² + ω²) = (coeff/ω) · ω/(s²+ω²)  → (coeff/ω) sin(ωt)
                // We absorb the coeff directly; caller already scaled via split_linear_terms.
                let result = match kind {
                    QuadKind::Plus => {
                        steps.push(format!(
                            "Applied L^{{-1}}{{ω/(s²+ω²)}} = sin(ω·t) with ω = {omega}, coeff = {coeff}"
                        ));
                        Arc::new(Expr::Func(FuncId::Sin, vec![omega_t]))
                    }
                    QuadKind::Minus => {
                        steps.push(format!(
                            "Applied L^{{-1}}{{a/(s²-a²)}} = sinh(a·t) with a = {omega}, coeff = {coeff}"
                        ));
                        Arc::new(Expr::Func(FuncId::Sinh, vec![omega_t]))
                    }
                };
                // Scale by coeff / omega (the table entry has omega in numerator)
                let scale = coeff / omega;
                let scaled = if (scale - 1.0).abs() < f64::EPSILON {
                    result
                } else {
                    normalize::mul(Expr::float(scale), result)
                };
                return Some(Ok(scaled));
            }
        }

        // coeff / s^n  or  coeff / (s-a)^n
        let inner = match_denom_pattern(&denom, n, s_var, t_var, steps)?;
        return Some(inner.map(|ft| {
            if (coeff - 1.0).abs() < f64::EPSILON {
                ft
            } else {
                normalize::mul(Expr::float(coeff), ft)
            }
        }));
    }

    None
}

// ── Individual table formulas ─────────────────────────────────────────────────

/// `1/s^n` → `t^(n-1) / (n-1)!`
fn match_one_over_s_n(
    n: u64,
    t_var: SymbolId,
    steps: &mut Vec<String>,
) -> Result<Arc<Expr>, TransformError> {
    if n == 0 {
        return Err(TransformError::InvalidInput(
            "L^{-1}{s^0} = L^{-1}{1} not in table".to_owned(),
        ));
    }
    let t = Arc::new(Expr::Symbol(t_var));
    if n == 1 {
        // L^{-1}{1/s} = 1
        steps.push("Applied L^{-1}{1/s} = 1".to_owned());
        return Ok(Expr::int(1));
    }
    // L^{-1}{1/s^n} = t^(n-1) / (n-1)!
    let m = n - 1;
    let t_pow = normalize::pow(t, Expr::int(m as i64));
    let fact = factorial(m) as f64;
    let result = normalize::mul(Expr::float(1.0 / fact), t_pow);
    steps.push(format!("Applied L^{{-1}}{{1/s^{n}}} = t^{m}/{m}! "));
    Ok(result)
}

/// `1/(s-a)^n` → `t^(n-1) e^(at) / (n-1)!`
fn match_shifted_pole(
    a: f64,
    n: u64,
    t_var: SymbolId,
    steps: &mut Vec<String>,
) -> Result<Arc<Expr>, TransformError> {
    let t = Arc::new(Expr::Symbol(t_var));
    let at = normalize::mul(Expr::float(a), Arc::clone(&t));
    let exp_at = Arc::new(Expr::Func(FuncId::Exp, vec![at]));

    if n == 1 {
        // L^{-1}{1/(s-a)} = e^(at)
        steps.push(format!("Applied L^{{-1}}{{1/(s-a)}} = e^(at) with a = {a}"));
        return Ok(exp_at);
    }
    // L^{-1}{1/(s-a)^n} = t^(n-1) e^(at) / (n-1)!
    let m = n - 1;
    let t_pow = normalize::pow(t, Expr::int(m as i64));
    let fact = factorial(m) as f64;
    let result = normalize::mul(normalize::mul(Expr::float(1.0 / fact), t_pow), exp_at);
    steps.push(format!(
        "Applied L^{{-1}}{{1/(s-a)^{n}}} = t^{m}·e^(at)/{m}! with a = {a}"
    ));
    Ok(result)
}

// ── Pattern recognisers ───────────────────────────────────────────────────────

/// Return `true` if `expr` is the bare `s` symbol.
fn is_s(expr: &Arc<Expr>, s_var: SymbolId) -> bool {
    matches!(expr.as_ref(), Expr::Symbol(id) if *id == s_var)
}

/// If `expr` represents `(s + c)` where `c` is a non-zero constant, return `Some(-c)`.
/// That is, it identifies the form `s - a` and returns `a`.
fn extract_shift(expr: &Arc<Expr>, s_var: SymbolId) -> Option<f64> {
    let Expr::Add(node) = expr.as_ref() else {
        return None;
    };
    // Must have exactly one symbolic term which is bare `s`.
    if node.terms.len() != 1 {
        return None;
    }
    let (term, coeff) = node.terms.iter().next()?;
    if !matches!(term.as_ref(), Expr::Symbol(id) if *id == s_var) {
        return None;
    }
    // The coefficient on s must be 1 (the addnode coeff is rational 1).
    use num::traits::One;
    if !coeff.is_one() {
        return None;
    }
    // constant part = -a  →  a = -constant
    let c = node.constant.to_f64();
    if c == 0.0 {
        return None; // that's just `s`; no shift
    }
    Some(-c) // shift a = -c
}

/// Quadratic denominator kind.
#[derive(Debug, Clone, Copy)]
enum QuadKind {
    Plus,  // s² + ω²
    Minus, // s² - ω²  (i.e. s² + (-ω²))
}

/// If `expr` is `s² + k` where `k ≠ 0` and `k` is a non-zero constant,
/// return `(|k|^(1/2), kind)`.
fn extract_quadratic(expr: &Expr, s_var: SymbolId) -> Option<(f64, QuadKind)> {
    let Expr::Add(node) = expr else {
        return None;
    };
    // Must have exactly one symbolic term: s^2.
    if node.terms.len() != 1 {
        return None;
    }
    let (term, coeff) = node.terms.iter().next()?;
    // term must be s^2 (encoded as a Mul or Pow)
    if !is_s_squared(term, s_var) {
        return None;
    }
    // Coefficient on s^2 must be 1.
    use num::traits::One;
    if !coeff.is_one() {
        return None;
    }
    // Constant part = ±ω²
    let k = node.constant.to_f64();
    if k == 0.0 {
        return None;
    }
    if k > 0.0 {
        Some((k.sqrt(), QuadKind::Plus))
    } else {
        Some(((-k).sqrt(), QuadKind::Minus))
    }
}

/// Return true if `expr` represents `s^2`.
fn is_s_squared(expr: &Arc<Expr>, s_var: SymbolId) -> bool {
    match expr.as_ref() {
        // Pow(s, 2)
        Expr::Pow(base, exp) => {
            is_s(base, s_var) && matches!(exp.as_ref(), Expr::Integer(n) if n.to_i64() == Some(2))
        }
        // MulNode with single factor s^2
        Expr::Mul(node) => {
            if node.factors.len() != 1 {
                return false;
            }
            use num::traits::One;
            if !node.coeff.is_one() {
                return false;
            }
            let (base, exp) = node.factors.iter().next().unwrap();
            is_s(base, s_var) && matches!(exp.as_ref(), Expr::Integer(n) if n.to_i64() == Some(2))
        }
        _ => false,
    }
}

// ── Algebraic helpers ─────────────────────────────────────────────────────────

/// Apply `L^{-1}{a·F + b·G + ...} = a·L^{-1}{F} + b·L^{-1}{G} + ...`.
fn apply_linearity(
    expr: &Arc<Expr>,
    s_var: SymbolId,
    t_var: SymbolId,
    steps: &mut Vec<String>,
) -> Result<Arc<Expr>, TransformError> {
    let terms = split_linear_terms(expr, s_var);
    steps.push(format!("Applying linearity to {} terms", terms.len()));

    let mut parts: Vec<Arc<Expr>> = Vec::with_capacity(terms.len());
    for (coeff, term) in terms {
        let inverted = invert_expr(&term, s_var, t_var, steps)?;
        let scaled = if (coeff - 1.0).abs() < f64::EPSILON {
            inverted
        } else {
            normalize::mul(Expr::float(coeff), inverted)
        };
        parts.push(scaled);
    }
    Ok(normalize::add_many(parts))
}

/// Handle `c · F(s)` as a scaled single term.
fn try_scaled_term(
    expr: &Arc<Expr>,
    s_var: SymbolId,
    t_var: SymbolId,
    steps: &mut Vec<String>,
) -> Option<Result<Arc<Expr>, TransformError>> {
    let Expr::Mul(node) = expr.as_ref() else {
        return None;
    };
    // Must have exactly one symbolic factor with exponent 1.
    if node.factors.len() != 1 {
        return None;
    }
    let (base, exp) = node.factors.iter().next()?;
    let is_exp1 = matches!(exp.as_ref(), Expr::Integer(n) if n.to_i64() == Some(1));
    if !is_exp1 {
        return None;
    }
    let coeff = node.coeff.to_f64();
    let inner = invert_expr(base, s_var, t_var, steps);
    Some(inner.map(|ft| {
        if (coeff - 1.0).abs() < f64::EPSILON {
            ft
        } else {
            normalize::mul(Expr::float(coeff), ft)
        }
    }))
}

// ── Numeric helpers ───────────────────────────────────────────────────────────

fn factorial(n: u64) -> u64 {
    (1..=n).product()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Expression, Function, Variable};
    use crate::numeric::compile::compile;
    use crate::numeric::substitute::substitute;

    fn s_sym() -> SymbolId {
        SymbolId::intern("s")
    }

    fn t_sym() -> SymbolId {
        SymbolId::intern("t")
    }

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn float(f: f64) -> Expression {
        Expression::Float(f)
    }

    fn mul_expr(a: Expression, b: Expression) -> Expression {
        Expression::Binary(BinaryOp::Mul, Box::new(a), Box::new(b))
    }

    fn add_expr(a: Expression, b: Expression) -> Expression {
        Expression::Binary(BinaryOp::Add, Box::new(a), Box::new(b))
    }

    fn sub_expr(a: Expression, b: Expression) -> Expression {
        Expression::Binary(BinaryOp::Sub, Box::new(a), Box::new(b))
    }

    fn div_expr(a: Expression, b: Expression) -> Expression {
        Expression::Binary(BinaryOp::Div, Box::new(a), Box::new(b))
    }

    fn pow_expr(base: Expression, exp: Expression) -> Expression {
        Expression::Power(Box::new(base), Box::new(exp))
    }

    /// Evaluate `f(t)` at a given numeric `t` value.
    fn eval_at(ft: &Arc<Expr>, t_id: SymbolId, t_val: f64) -> f64 {
        let t_expr = Expr::float(t_val);
        let subst = substitute(ft, t_id, &t_expr);
        match subst.as_ref() {
            Expr::Float(v) => *v,
            Expr::Integer(n) => n.to_i64().unwrap_or(0) as f64,
            Expr::Rational(r) => r.to_f64(),
            _ => f64::NAN,
        }
    }

    // L^{-1}{1/s} = 1  →  at t=1: 1.0
    #[test]
    fn fast_inv_laplace_one_over_s() {
        // Build 1/s = s^(-1)
        let s = var("s");
        let expr = compile(&pow_expr(s, int(-1)));
        let result = inverse_laplace(&expr, s_sym(), t_sym()).expect("L^{-1}{1/s}");
        let val = eval_at(&result.expr, t_sym(), 1.0);
        assert!((val - 1.0).abs() < 1e-10, "expected 1.0, got {val}");
        assert!(!result.steps.is_empty());
    }

    // L^{-1}{1/s^2} = t  →  at t=3: 3.0
    #[test]
    fn fast_inv_laplace_one_over_s2() {
        let s = var("s");
        let expr = compile(&pow_expr(s, int(-2)));
        let result = inverse_laplace(&expr, s_sym(), t_sym()).expect("L^{-1}{1/s²}");
        let val = eval_at(&result.expr, t_sym(), 3.0);
        assert!((val - 3.0).abs() < 1e-10, "expected 3.0, got {val}");
    }

    // L^{-1}{1/s^3} = t²/2  →  at t=2: 4/2 = 2.0
    #[test]
    fn fast_inv_laplace_one_over_s3() {
        let s = var("s");
        let expr = compile(&pow_expr(s, int(-3)));
        let result = inverse_laplace(&expr, s_sym(), t_sym()).expect("L^{-1}{1/s³}");
        let val = eval_at(&result.expr, t_sym(), 2.0);
        assert!((val - 2.0).abs() < 1e-10, "expected 2.0, got {val}");
    }

    // L^{-1}{1/(s-2)} = e^(2t)  →  at t=1: e^2 ≈ 7.389
    #[test]
    fn fast_inv_laplace_shifted_pole() {
        // Build 1/(s-2) = (s + (-2))^{-1}
        let s = var("s");
        let expr = compile(&div_expr(int(1), sub_expr(s.clone(), int(2))));
        let result = inverse_laplace(&expr, s_sym(), t_sym()).expect("L^{-1}{1/(s-2)}");
        let val = eval_at(&result.expr, t_sym(), 1.0);
        let expected = std::f64::consts::E * std::f64::consts::E;
        assert!(
            (val - expected).abs() < 1e-8,
            "expected e^2 ≈ {expected}, got {val}"
        );
    }

    // L^{-1}{3/(s²+9)} = sin(3t)  →  at t=π/6: sin(π/2) = 1.0
    #[test]
    fn fast_inv_laplace_sin() {
        // Build 3/(s^2 + 9)
        let s = var("s");
        let denom = add_expr(pow_expr(s.clone(), int(2)), int(9));
        let expr = compile(&div_expr(int(3), denom));
        let result = inverse_laplace(&expr, s_sym(), t_sym()).expect("L^{-1}{3/(s²+9)}");
        let t_val = std::f64::consts::PI / 6.0;
        let val = eval_at(&result.expr, t_sym(), t_val);
        assert!((val - 1.0).abs() < 1e-8, "expected 1.0, got {val}");
    }

    // L^{-1}{s/(s²+4)} = cos(2t)  →  at t=0: cos(0) = 1.0
    #[test]
    fn fast_inv_laplace_cos() {
        let s = var("s");
        let denom = add_expr(pow_expr(s.clone(), int(2)), int(4));
        let expr = compile(&div_expr(s.clone(), denom));
        let result = inverse_laplace(&expr, s_sym(), t_sym()).expect("L^{-1}{s/(s²+4)}");
        let val = eval_at(&result.expr, t_sym(), 0.0);
        assert!((val - 1.0).abs() < 1e-8, "expected 1.0, got {val}");
    }

    // L^{-1}{2/(s²-4)} = sinh(2t)  →  at t=1: sinh(2) ≈ 3.6269
    #[test]
    fn fast_inv_laplace_sinh() {
        let s = var("s");
        let denom = sub_expr(pow_expr(s.clone(), int(2)), int(4));
        let expr = compile(&div_expr(int(2), denom));
        let result = inverse_laplace(&expr, s_sym(), t_sym()).expect("L^{-1}{2/(s²-4)}");
        let val = eval_at(&result.expr, t_sym(), 1.0);
        let expected = (2.0_f64).sinh();
        assert!(
            (val - expected).abs() < 1e-8,
            "expected {expected}, got {val}"
        );
    }

    // L^{-1}{s/(s²-9)} = cosh(3t)  →  at t=0: cosh(0) = 1.0
    #[test]
    fn fast_inv_laplace_cosh() {
        let s = var("s");
        let denom = sub_expr(pow_expr(s.clone(), int(2)), int(9));
        let expr = compile(&div_expr(s.clone(), denom));
        let result = inverse_laplace(&expr, s_sym(), t_sym()).expect("L^{-1}{s/(s²-9)}");
        let val = eval_at(&result.expr, t_sym(), 0.0);
        assert!((val - 1.0).abs() < 1e-8, "expected 1.0, got {val}");
    }

    // L^{-1}{1/(s-2)^2} = t·e^(2t)  →  at t=1: 1·e^2 ≈ 7.389
    #[test]
    fn fast_inv_laplace_repeated_shifted_pole() {
        let s = var("s");
        let denom = pow_expr(sub_expr(s.clone(), int(2)), int(2));
        let expr = compile(&div_expr(int(1), denom));
        let result = inverse_laplace(&expr, s_sym(), t_sym()).expect("L^{-1}{1/(s-2)²}");
        let val = eval_at(&result.expr, t_sym(), 1.0);
        let expected = std::f64::consts::E * std::f64::consts::E; // 1·e^2
        assert!(
            (val - expected).abs() < 1e-8,
            "expected e^2 ≈ {expected}, got {val}"
        );
    }

    // Linearity: L^{-1}{1/s + 1/(s-1)} = 1 + e^t  →  at t=1: 1 + e ≈ 3.718
    #[test]
    fn fast_inv_laplace_linearity() {
        let s = var("s");
        let f1 = pow_expr(s.clone(), int(-1));
        let f2 = div_expr(int(1), sub_expr(s.clone(), int(1)));
        let expr = compile(&add_expr(f1, f2));
        let result = inverse_laplace(&expr, s_sym(), t_sym()).expect("L^{-1}{1/s + 1/(s-1)}");
        let val = eval_at(&result.expr, t_sym(), 1.0);
        let expected = 1.0 + std::f64::consts::E;
        assert!(
            (val - expected).abs() < 1e-8,
            "expected 1+e ≈ {expected}, got {val}"
        );
    }

    // Unsupported: tan(s) → NoTableEntry
    #[test]
    fn fast_inv_laplace_no_entry() {
        use crate::ast::Function;
        let s_expr = Expression::Function(Function::Tan, vec![var("s")]);
        let expr = compile(&s_expr);
        let err = inverse_laplace(&expr, s_sym(), t_sym());
        assert!(matches!(err, Err(TransformError::NoTableEntry(_))));
    }

    // Unused suppression — float/sub_expr used in test helpers
    #[allow(dead_code)]
    fn _use_helpers(a: Expression, b: Expression) -> (Expression, Expression) {
        (float(1.0), sub_expr(a, b))
    }
}
