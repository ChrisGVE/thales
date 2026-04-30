//! Laplace transform: symbolic table lookup with linearity.
//!
//! Implements `L{f(t)} = F(s)` for elementary functions via a direct pattern
//! table plus recursive linearity decomposition.
//!
//! # Table entries
//!
//! | `f(t)`       | `F(s)`              |
//! |--------------|---------------------|
//! | 1            | 1/s                 |
//! | t            | 1/s²                |
//! | t^n          | n!/s^(n+1)          |
//! | e^(a·t)      | 1/(s − a)           |
//! | sin(ω·t)     | ω/(s² + ω²)         |
//! | cos(ω·t)     | s/(s² + ω²)         |
//! | sinh(a·t)    | a/(s² − a²)         |
//! | cosh(a·t)    | s/(s² − a²)         |

use std::sync::Arc;

use crate::numeric::{
    expr::{Expr, FuncId},
    normalize, SymbolId,
};

use super::{as_constant, contains_var, split_linear_terms, TransformError, TransformResult};

// ── Public entry point ────────────────────────────────────────────────────────

/// Compute the Laplace transform of `expr` with respect to `t_var`,
/// returning a result in the `s_var` domain.
///
/// Uses table lookup for elementary forms and the linearity property
/// `L{a·f + b·g} = a·L{f} + b·L{g}` to extend coverage.
pub fn laplace_transform(
    expr: &Arc<Expr>,
    t_var: SymbolId,
    s_var: SymbolId,
) -> Result<TransformResult, TransformError> {
    let mut steps: Vec<String> = Vec::new();
    let result_expr = transform_expr(expr, t_var, s_var, &mut steps)?;

    Ok(TransformResult {
        expr: result_expr,
        domain_var: s_var.as_str().to_owned(),
        convergence: None,
        steps,
    })
}

// ── Core dispatch ─────────────────────────────────────────────────────────────

/// Recursively transform `expr`, appending narrated steps.
fn transform_expr(
    expr: &Arc<Expr>,
    t_var: SymbolId,
    s_var: SymbolId,
    steps: &mut Vec<String>,
) -> Result<Arc<Expr>, TransformError> {
    // 1. Try direct table match first.
    if let Some(result) = table_lookup(expr, t_var, s_var, steps) {
        return result;
    }

    // 2. Linearity: split into (coeff, term) pairs and transform each.
    if matches!(expr.as_ref(), Expr::Add(_)) {
        return apply_linearity(expr, t_var, s_var, steps);
    }

    // 3. Scaled term: c · f(t) — extract coefficient and recurse.
    if let Expr::Mul(_) = expr.as_ref() {
        if let Some(result) = try_scaled_term(expr, t_var, s_var, steps) {
            return result;
        }
    }

    Err(TransformError::NoTableEntry(format!("{expr}")))
}

// ── Table lookup ──────────────────────────────────────────────────────────────

/// Try to match `expr` against a known Laplace table entry.
///
/// Returns `Some(Ok(F(s)))` on a hit, `Some(Err(_))` on a structural match
/// that fails a precondition, and `None` when no table entry applies.
fn table_lookup(
    expr: &Arc<Expr>,
    t_var: SymbolId,
    s_var: SymbolId,
    steps: &mut Vec<String>,
) -> Option<Result<Arc<Expr>, TransformError>> {
    let s = Arc::new(Expr::Symbol(s_var));

    match expr.as_ref() {
        // ── Constant (including Integer/Rational/Float) ────────────────────
        Expr::Integer(_) | Expr::Rational(_) | Expr::Float(_) => {
            if contains_var(expr, t_var) {
                return None;
            }
            // L{c} = c/s
            let c = Arc::clone(expr);
            let result = normalize::mul(c.clone(), normalize::pow(s, Expr::int(-1)));
            steps.push(format!("Applied L{{c}} = c/s with c = {c}"));
            Some(Ok(result))
        }

        // ── Variable t ────────────────────────────────────────────────────
        Expr::Symbol(id) if *id == t_var => {
            // L{t} = 1/s²
            let result = normalize::pow(s, Expr::int(-2));
            steps.push("Applied L{t} = 1/s²".to_owned());
            Some(Ok(result))
        }

        // ── Symbol that is NOT t: treat as constant ────────────────────────
        Expr::Symbol(_) => {
            if contains_var(expr, t_var) {
                return None;
            }
            let c = Arc::clone(expr);
            let result = normalize::mul(c.clone(), normalize::pow(s, Expr::int(-1)));
            steps.push(format!("Applied L{{c}} = c/s with c = {c}"));
            Some(Ok(result))
        }

        // ── Power: t^n ────────────────────────────────────────────────────
        Expr::Pow(base, exp) => match_pow(base, exp, t_var, s_var, steps),

        // ── Function: Exp, Sin, Cos, Sinh, Cosh ───────────────────────────
        Expr::Func(fid, args) if args.len() == 1 => match_func(*fid, &args[0], t_var, &s, steps),

        // ── Mul: may encode e^(a·t) as Func(Exp, [Mul(...,t,...)]) ─────────
        // Already handled via try_scaled_term / linearity above.
        _ => None,
    }
}

/// Match `base^exp` where base == t_var.
fn match_pow(
    base: &Arc<Expr>,
    exp: &Arc<Expr>,
    t_var: SymbolId,
    s_var: SymbolId,
    steps: &mut Vec<String>,
) -> Option<Result<Arc<Expr>, TransformError>> {
    // Must be t^n
    let is_t = matches!(base.as_ref(), Expr::Symbol(id) if *id == t_var);
    if !is_t {
        return None;
    }

    let n = match as_constant(exp) {
        Some(v) if v >= 1.0 && v.fract() == 0.0 && v <= 20.0 => v as u64,
        Some(_) => {
            return Some(Err(TransformError::InvalidInput(
                "L{t^n} requires a positive integer n ≤ 20".to_owned(),
            )))
        }
        None => return None,
    };

    // L{t^n} = n! / s^(n+1)
    let factorial_n = factorial(n);
    let s = Arc::new(Expr::Symbol(s_var));
    let s_pow = normalize::pow(s, Expr::int((n + 1) as i64));
    let result = normalize::mul(
        Expr::int(factorial_n as i64),
        normalize::pow(s_pow, Expr::int(-1)),
    );
    steps.push(format!(
        "Applied L{{t^{n}}} = {n}!/s^{} with n = {n}",
        n + 1
    ));
    Some(Ok(result))
}

/// Match `Func(fid, [arg])` against known single-argument table entries.
fn match_func(
    fid: FuncId,
    arg: &Arc<Expr>,
    t_var: SymbolId,
    s: &Arc<Expr>,
    steps: &mut Vec<String>,
) -> Option<Result<Arc<Expr>, TransformError>> {
    match fid {
        FuncId::Exp => match_exp(arg, t_var, s, steps),
        FuncId::Sin => match_sin(arg, t_var, s, steps),
        FuncId::Cos => match_cos(arg, t_var, s, steps),
        FuncId::Sinh => match_sinh(arg, t_var, s, steps),
        FuncId::Cosh => match_cosh(arg, t_var, s, steps),
        _ => None,
    }
}

// ── Individual function matchers ─────────────────────────────────────────────

/// `e^(a·t)` → `1/(s − a)`.  Arg must be `a·t` or just `t`.
fn match_exp(
    arg: &Arc<Expr>,
    t_var: SymbolId,
    s: &Arc<Expr>,
    steps: &mut Vec<String>,
) -> Option<Result<Arc<Expr>, TransformError>> {
    let a = extract_linear_coeff(arg, t_var)?;
    // L{e^(a·t)} = 1/(s − a)
    let a_expr = Expr::float(a);
    let denom = normalize::sub(Arc::clone(s), a_expr);
    let result = normalize::pow(denom, Expr::int(-1));
    steps.push(format!("Applied L{{e^(a·t)}} = 1/(s-a) with a = {a}"));
    Some(Ok(result))
}

/// `sin(ω·t)` → `ω/(s² + ω²)`.
fn match_sin(
    arg: &Arc<Expr>,
    t_var: SymbolId,
    s: &Arc<Expr>,
    steps: &mut Vec<String>,
) -> Option<Result<Arc<Expr>, TransformError>> {
    let omega = extract_linear_coeff(arg, t_var)?;
    let (numer, denom) = sin_cos_parts(omega, s);
    let result = normalize::mul(numer, normalize::pow(denom, Expr::int(-1)));
    steps.push(format!(
        "Applied L{{sin(ω·t)}} = ω/(s²+ω²) with ω = {omega}"
    ));
    Some(Ok(result))
}

/// `cos(ω·t)` → `s/(s² + ω²)`.
fn match_cos(
    arg: &Arc<Expr>,
    t_var: SymbolId,
    s: &Arc<Expr>,
    steps: &mut Vec<String>,
) -> Option<Result<Arc<Expr>, TransformError>> {
    let omega = extract_linear_coeff(arg, t_var)?;
    let (_numer, denom) = sin_cos_parts(omega, s);
    let result = normalize::mul(Arc::clone(s), normalize::pow(denom, Expr::int(-1)));
    steps.push(format!(
        "Applied L{{cos(ω·t)}} = s/(s²+ω²) with ω = {omega}"
    ));
    Some(Ok(result))
}

/// `sinh(a·t)` → `a/(s² − a²)`.
fn match_sinh(
    arg: &Arc<Expr>,
    t_var: SymbolId,
    s: &Arc<Expr>,
    steps: &mut Vec<String>,
) -> Option<Result<Arc<Expr>, TransformError>> {
    let a = extract_linear_coeff(arg, t_var)?;
    let (numer, denom) = sinh_cosh_parts(a, s);
    let result = normalize::mul(numer, normalize::pow(denom, Expr::int(-1)));
    steps.push(format!("Applied L{{sinh(a·t)}} = a/(s²-a²) with a = {a}"));
    Some(Ok(result))
}

/// `cosh(a·t)` → `s/(s² − a²)`.
fn match_cosh(
    arg: &Arc<Expr>,
    t_var: SymbolId,
    s: &Arc<Expr>,
    steps: &mut Vec<String>,
) -> Option<Result<Arc<Expr>, TransformError>> {
    let a = extract_linear_coeff(arg, t_var)?;
    let (_numer, denom) = sinh_cosh_parts(a, s);
    let result = normalize::mul(Arc::clone(s), normalize::pow(denom, Expr::int(-1)));
    steps.push(format!("Applied L{{cosh(a·t)}} = s/(s²-a²) with a = {a}"));
    Some(Ok(result))
}

// ── Algebraic helpers ─────────────────────────────────────────────────────────

/// Build `(ω_expr, s² + ω²)` for sin/cos table entries.
fn sin_cos_parts(omega: f64, s: &Arc<Expr>) -> (Arc<Expr>, Arc<Expr>) {
    let omega_expr = Expr::float(omega);
    let s2 = normalize::pow(Arc::clone(s), Expr::int(2));
    let omega2 = Expr::float(omega * omega);
    let denom = normalize::add(s2, omega2);
    (omega_expr, denom)
}

/// Build `(a_expr, s² − a²)` for sinh/cosh table entries.
fn sinh_cosh_parts(a: f64, s: &Arc<Expr>) -> (Arc<Expr>, Arc<Expr>) {
    let a_expr = Expr::float(a);
    let s2 = normalize::pow(Arc::clone(s), Expr::int(2));
    let a2 = Expr::float(a * a);
    let denom = normalize::sub(s2, a2);
    (a_expr, denom)
}

/// Extract the scalar coefficient `a` from an expression of the form `a·t`
/// (or just `t`, in which case `a = 1.0`).
///
/// Returns `None` if the expression does not have that linear shape.
fn extract_linear_coeff(expr: &Arc<Expr>, t_var: SymbolId) -> Option<f64> {
    match expr.as_ref() {
        // Plain symbol t
        Expr::Symbol(id) if *id == t_var => Some(1.0),
        // a · t  encoded as MulNode
        Expr::Mul(node) => {
            // Exactly one factor which is t^1
            if node.factors.len() != 1 {
                return None;
            }
            let (base, exp) = node.factors.iter().next()?;
            let is_t = matches!(base.as_ref(), Expr::Symbol(id) if *id == t_var);
            let is_exp1 = matches!(exp.as_ref(), Expr::Integer(n) if n.to_i64() == Some(1));
            if is_t && is_exp1 {
                Some(node.coeff.to_f64())
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Apply `L{a·f + b·g + ...} = a·L{f} + b·L{g} + ...`.
fn apply_linearity(
    expr: &Arc<Expr>,
    t_var: SymbolId,
    s_var: SymbolId,
    steps: &mut Vec<String>,
) -> Result<Arc<Expr>, TransformError> {
    let terms = split_linear_terms(expr, t_var);
    steps.push(format!("Applying linearity to {} terms", terms.len()));

    let mut parts: Vec<Arc<Expr>> = Vec::with_capacity(terms.len());

    for (coeff, term) in terms {
        let transformed = transform_expr(&term, t_var, s_var, steps)?;
        let scaled = if (coeff - 1.0).abs() < f64::EPSILON {
            transformed
        } else {
            normalize::mul(Expr::float(coeff), transformed)
        };
        parts.push(scaled);
    }

    Ok(normalize::add_many(parts))
}

/// Try to handle `c · f(t)` as a scaled single-function term.
fn try_scaled_term(
    expr: &Arc<Expr>,
    t_var: SymbolId,
    s_var: SymbolId,
    steps: &mut Vec<String>,
) -> Option<Result<Arc<Expr>, TransformError>> {
    let Expr::Mul(node) = expr.as_ref() else {
        return None;
    };

    // Require exactly one factor.
    if node.factors.len() != 1 {
        return None;
    }

    let (base, exp) = node.factors.iter().next()?;

    // Factor must appear with exponent 1.
    let is_exp1 = matches!(exp.as_ref(), Expr::Integer(n) if n.to_i64() == Some(1));
    if !is_exp1 {
        return None;
    }

    let coeff = node.coeff.to_f64();
    let inner_transform = transform_expr(base, t_var, s_var, steps);

    Some(inner_transform.map(|f_s| {
        if (coeff - 1.0).abs() < f64::EPSILON {
            f_s
        } else {
            normalize::mul(Expr::float(coeff), f_s)
        }
    }))
}

// ── Numeric helpers ───────────────────────────────────────────────────────────

/// Compute `n!` for `n ≤ 20` (fits in u64).
fn factorial(n: u64) -> u64 {
    (1..=n).product()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Expression, Function, Variable};
    use crate::numeric::compile::compile;

    fn t_sym() -> SymbolId {
        SymbolId::intern("t")
    }

    fn s_sym() -> SymbolId {
        SymbolId::intern("s")
    }

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn mul_expr(a: Expression, b: Expression) -> Expression {
        Expression::Binary(BinaryOp::Mul, Box::new(a), Box::new(b))
    }

    fn add_expr(a: Expression, b: Expression) -> Expression {
        Expression::Binary(BinaryOp::Add, Box::new(a), Box::new(b))
    }

    fn pow_expr(base: Expression, exp: Expression) -> Expression {
        Expression::Power(Box::new(base), Box::new(exp))
    }

    fn func_expr(name: &str, arg: Expression) -> Expression {
        let func = match name {
            "exp" => Function::Exp,
            "sin" => Function::Sin,
            "cos" => Function::Cos,
            "sinh" => Function::Sinh,
            "cosh" => Function::Cosh,
            "tan" => Function::Tan,
            "abs" => Function::Abs,
            other => panic!("func_expr: unknown function '{other}'"),
        };
        Expression::Function(func, vec![arg])
    }

    /// Evaluate `F(s)` at a given numeric `s` value.
    fn eval_at(f_s: &Arc<Expr>, s_id: SymbolId, s_val: f64) -> f64 {
        use crate::numeric::substitute::substitute;
        let s_expr = Expr::float(s_val);
        let subst = substitute(f_s, s_id, &s_expr);
        match subst.as_ref() {
            Expr::Float(v) => *v,
            Expr::Integer(n) => n.to_i64().unwrap_or(0) as f64,
            Expr::Rational(r) => r.to_f64(),
            _ => f64::NAN,
        }
    }

    // L{1} = 1/s  →  at s=2: 1/2
    #[test]
    fn fast_laplace_constant_one() {
        let expr = compile(&int(1));
        let result = laplace_transform(&expr, t_sym(), s_sym()).expect("L{1}");
        let val = eval_at(&result.expr, s_sym(), 2.0);
        assert!((val - 0.5).abs() < 1e-10, "expected 0.5, got {val}");
        assert!(!result.steps.is_empty());
    }

    // L{t} = 1/s²  →  at s=3: 1/9
    #[test]
    fn fast_laplace_t() {
        let expr = compile(&var("t"));
        let result = laplace_transform(&expr, t_sym(), s_sym()).expect("L{t}");
        let val = eval_at(&result.expr, s_sym(), 3.0);
        assert!((val - 1.0 / 9.0).abs() < 1e-10, "expected 1/9, got {val}");
    }

    // L{t²} = 2/s³  →  at s=2: 2/8 = 0.25
    #[test]
    fn fast_laplace_t_squared() {
        let expr = compile(&pow_expr(var("t"), int(2)));
        let result = laplace_transform(&expr, t_sym(), s_sym()).expect("L{t²}");
        let val = eval_at(&result.expr, s_sym(), 2.0);
        assert!((val - 0.25).abs() < 1e-10, "expected 0.25, got {val}");
    }

    // L{e^(2t)} = 1/(s-2)  →  at s=5: 1/3
    #[test]
    fn fast_laplace_exp_2t() {
        let expr = compile(&func_expr("exp", mul_expr(int(2), var("t"))));
        let result = laplace_transform(&expr, t_sym(), s_sym()).expect("L{e^(2t)}");
        let val = eval_at(&result.expr, s_sym(), 5.0);
        assert!((val - 1.0 / 3.0).abs() < 1e-10, "expected 1/3, got {val}");
    }

    // L{sin(3t)} = 3/(s²+9)  →  at s=4: 3/25 = 0.12
    #[test]
    fn fast_laplace_sin_3t() {
        let expr = compile(&func_expr("sin", mul_expr(int(3), var("t"))));
        let result = laplace_transform(&expr, t_sym(), s_sym()).expect("L{sin(3t)}");
        let val = eval_at(&result.expr, s_sym(), 4.0);
        assert!((val - 3.0 / 25.0).abs() < 1e-10, "expected 0.12, got {val}");
    }

    // L{cos(3t)} = s/(s²+9)  →  at s=4: 4/25 = 0.16
    #[test]
    fn fast_laplace_cos_3t() {
        let expr = compile(&func_expr("cos", mul_expr(int(3), var("t"))));
        let result = laplace_transform(&expr, t_sym(), s_sym()).expect("L{cos(3t)}");
        let val = eval_at(&result.expr, s_sym(), 4.0);
        assert!((val - 4.0 / 25.0).abs() < 1e-10, "expected 0.16, got {val}");
    }

    // L{sinh(2t)} = 2/(s²-4)  →  at s=3: 2/5 = 0.4
    #[test]
    fn fast_laplace_sinh_2t() {
        let expr = compile(&func_expr("sinh", mul_expr(int(2), var("t"))));
        let result = laplace_transform(&expr, t_sym(), s_sym()).expect("L{sinh(2t)}");
        let val = eval_at(&result.expr, s_sym(), 3.0);
        assert!((val - 0.4).abs() < 1e-10, "expected 0.4, got {val}");
    }

    // L{cosh(2t)} = s/(s²-4)  →  at s=3: 3/5 = 0.6
    #[test]
    fn fast_laplace_cosh_2t() {
        let expr = compile(&func_expr("cosh", mul_expr(int(2), var("t"))));
        let result = laplace_transform(&expr, t_sym(), s_sym()).expect("L{cosh(2t)}");
        let val = eval_at(&result.expr, s_sym(), 3.0);
        assert!((val - 0.6).abs() < 1e-10, "expected 0.6, got {val}");
    }

    // L{2t + 3} = 2/s² + 3/s  →  at s=2: 2/4 + 3/2 = 0.5 + 1.5 = 2.0
    #[test]
    fn fast_laplace_linearity() {
        let expr = compile(&add_expr(mul_expr(int(2), var("t")), int(3)));
        let result = laplace_transform(&expr, t_sym(), s_sym()).expect("L{2t+3}");
        let val = eval_at(&result.expr, s_sym(), 2.0);
        assert!((val - 2.0).abs() < 1e-10, "expected 2.0, got {val}");
    }

    // Unsupported input must return NoTableEntry.
    #[test]
    fn fast_laplace_no_entry_for_unsupported() {
        // tan(t) — not in the table
        let expr = compile(&func_expr("tan", var("t")));
        let err = laplace_transform(&expr, t_sym(), s_sym());
        assert!(err.is_err());
        assert!(matches!(err, Err(TransformError::NoTableEntry(_))));
    }
}
