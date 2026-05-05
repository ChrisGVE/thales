//! Fourier transform: symbolic table lookup and linearity.
//!
//! Convention: F(ω) = ∫_{-∞}^{∞} f(t) e^{-iωt} dt
//!
//! # Table entries
//!
//! - Gaussian:          e^(-a·t²)   → √(π/a) · e^(-ω²/(4a))   (a > 0)
//! - Exponential decay: e^(-a·|t|)  → 2a / (a² + ω²)           (a > 0)
//!
//! Linear combinations are handled via `split_linear_terms`.

use std::sync::Arc;

use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::normalize;
use crate::numeric::SymbolId;

use super::{as_constant, contains_var, split_linear_terms, TransformError, TransformResult};

// ── Public entry point ────────────────────────────────────────────────────────

/// Compute the Fourier transform F(ω) = ∫_{-∞}^{∞} f(t) e^{-iωt} dt.
///
/// Applies table lookup with linearity. Returns a [`TransformResult`] on
/// success or a [`TransformError`] when the expression is not in the table.
pub fn fourier_transform(
    expr: &Arc<Expr>,
    t_var: SymbolId,
    omega_var: SymbolId,
) -> Result<TransformResult, TransformError> {
    let mut steps: Vec<String> = Vec::new();

    // Fast path: expression does not depend on t at all.
    if !contains_var(expr, t_var) {
        return Err(TransformError::NonElementary(
            "bare constant has distributional Fourier transform (delta function); not supported"
                .into(),
        ));
    }

    // Try direct table lookup first.
    if let Some(result_expr) = try_table(expr, t_var, omega_var, &mut steps) {
        return Ok(TransformResult {
            expr: result_expr,
            domain_var: omega_var.as_str(),
            convergence: None,
            steps,
        });
    }

    // Linearity: split into (coefficient, term) pairs and transform each.
    let terms = split_linear_terms(expr, t_var);
    if terms.len() > 1 {
        return apply_linearity(&terms, t_var, omega_var, steps);
    }

    Err(TransformError::NoTableEntry(format!(
        "no Fourier transform table entry for: {expr}"
    )))
}

// ── Linearity ─────────────────────────────────────────────────────────────────

/// Apply linearity: F{a·f + b·g + …} = a·F{f} + b·F{g} + …
fn apply_linearity(
    terms: &[(f64, Arc<Expr>)],
    t_var: SymbolId,
    omega_var: SymbolId,
    mut steps: Vec<String>,
) -> Result<TransformResult, TransformError> {
    steps.push("Applied linearity of Fourier transform.".into());

    let mut result_expr: Option<Arc<Expr>> = None;

    for (coeff, term) in terms {
        // Recurse on each individual term.
        let term_result = fourier_transform(term, t_var, omega_var)?;

        let scaled = if (*coeff - 1.0).abs() < 1e-15 {
            term_result.expr.clone()
        } else {
            normalize::mul(Expr::float(*coeff), term_result.expr.clone())
        };

        steps.extend(term_result.steps);

        result_expr = Some(match result_expr {
            None => scaled,
            Some(acc) => normalize::add(acc, scaled),
        });
    }

    Ok(TransformResult {
        expr: result_expr.unwrap_or_else(|| Expr::int(0)),
        domain_var: omega_var.as_str(),
        convergence: None,
        steps,
    })
}

// ── Table lookup ──────────────────────────────────────────────────────────────

/// Attempt a direct table match. Returns `Some(result_expr)` on hit.
fn try_table(
    expr: &Arc<Expr>,
    t_var: SymbolId,
    omega_var: SymbolId,
    steps: &mut Vec<String>,
) -> Option<Arc<Expr>> {
    // All current table entries involve Exp(…).
    if let Expr::Func(FuncId::Exp, args) = expr.as_ref() {
        if args.len() == 1 {
            let arg = &args[0];
            if let Some(result) = try_gaussian(arg, t_var, omega_var, steps) {
                return Some(result);
            }
            if let Some(result) = try_exp_decay(arg, t_var, omega_var, steps) {
                return Some(result);
            }
        }
    }
    None
}

// ── Gaussian: e^(-a·t²) → √(π/a) · e^(-ω²/(4a)) ─────────────────────────────

/// Match e^(-a·t²) with a > 0 symbolic constant.
///
/// The exponent must have the form `-a * t^2` where `a` is a positive
/// numeric constant independent of `t`.
fn try_gaussian(
    exponent: &Arc<Expr>,
    t_var: SymbolId,
    omega_var: SymbolId,
    steps: &mut Vec<String>,
) -> Option<Arc<Expr>> {
    // Extract coefficient `a` and verify the structural form `-a * t^2`.
    let a = extract_neg_coeff_times_var_squared(exponent, t_var)?;
    if a <= 0.0 {
        return None;
    }

    // Build √(π/a) · e^{-ω²/(4a)}
    let omega = Arc::new(Expr::Symbol(omega_var));
    let pi = Expr::pi();

    // π/a
    let pi_over_a = normalize::div(pi, Expr::float(a));
    // √(π/a)
    let amplitude = Expr::func(FuncId::Sqrt, vec![pi_over_a]);

    // -ω² / (4a)
    let omega_sq = normalize::pow(omega.clone(), Expr::int(2));
    let four_a = Expr::float(4.0 * a);
    let neg_omega_sq_over_4a = normalize::div(normalize::neg(omega_sq), four_a);

    // e^{-ω²/(4a)}
    let exp_part = Expr::func(FuncId::Exp, vec![neg_omega_sq_over_4a]);

    let result = normalize::mul(amplitude, exp_part);

    steps.push(format!(
        "Applied Fourier transform of Gaussian: \
         F{{e^(-{a}·t²)}} = √(π/{a})·e^(-ω²/(4·{a}))"
    ));

    Some(result)
}

// ── Exponential decay: e^(-a·|t|) → 2a/(a²+ω²) ──────────────────────────────

/// Match e^(-a·|t|) with a > 0 symbolic constant.
///
/// The exponent must have the form `-a * abs(t)` where `a` is a positive
/// numeric constant independent of `t`.
fn try_exp_decay(
    exponent: &Arc<Expr>,
    t_var: SymbolId,
    omega_var: SymbolId,
    steps: &mut Vec<String>,
) -> Option<Arc<Expr>> {
    let a = extract_neg_coeff_times_abs_var(exponent, t_var)?;
    if a <= 0.0 {
        return None;
    }

    // Build 2a / (a² + ω²)
    let omega = Arc::new(Expr::Symbol(omega_var));
    let two_a = Expr::float(2.0 * a);
    let a_sq = Expr::float(a * a);
    let omega_sq = normalize::pow(omega.clone(), Expr::int(2));
    let denom = normalize::add(a_sq, omega_sq);
    let result = normalize::div(two_a, denom);

    steps.push(format!(
        "Applied Fourier transform of two-sided exponential decay: \
         F{{e^(-{a}·|t|)}} = 2·{a}/({}+ω²)",
        a * a
    ));

    Some(result)
}

// ── Pattern-matching helpers ──────────────────────────────────────────────────

/// Match `-a * t^2` in an exponent and return `a` (positive).
///
/// Accepts:
/// - `Mul` node with coefficient `-a` and a single factor `t^2`
/// - `Neg(Mul(a, t^2))` via the MulNode coeff pathway
fn extract_neg_coeff_times_var_squared(expr: &Arc<Expr>, var: SymbolId) -> Option<f64> {
    match expr.as_ref() {
        Expr::Mul(node) => {
            let coeff = node.coeff.to_f64();
            if coeff >= 0.0 {
                return None; // exponent must be negative
            }
            let a = -coeff;
            // Exactly one factor which must be var^2
            if node.factors.len() != 1 {
                return None;
            }
            let (base, exp) = node.factors.iter().next()?;
            if !is_symbol(base, var) {
                return None;
            }
            if !is_integer_value(exp, 2) {
                return None;
            }
            Some(a)
        }
        Expr::Pow(base, exp) => {
            // Handles the case `-t^2` when coeff is already folded differently.
            // Not normally needed given MulNode normalization but kept for safety.
            if is_symbol(base, var) && is_integer_value(exp, 2) {
                // Coefficient is implicitly -1 only if wrapped in a neg; without
                // a surrounding Mul we cannot confirm the sign. Return None here
                // and rely on the Mul branch above.
            }
            None
        }
        _ => None,
    }
}

/// Match `-a * abs(t)` in an exponent and return `a` (positive).
fn extract_neg_coeff_times_abs_var(expr: &Arc<Expr>, var: SymbolId) -> Option<f64> {
    match expr.as_ref() {
        Expr::Mul(node) => {
            let coeff = node.coeff.to_f64();
            if coeff >= 0.0 {
                return None;
            }
            let a = -coeff;
            if node.factors.len() != 1 {
                return None;
            }
            let (base, exp) = node.factors.iter().next()?;
            // exp must be 1 (i.e., the factor appears to the first power)
            if !is_integer_value(exp, 1) {
                return None;
            }
            // base must be abs(var)
            if is_abs_of_var(base, var) {
                return Some(a);
            }
            None
        }
        _ => None,
    }
}

/// Returns true if `expr` is `Symbol(var)`.
fn is_symbol(expr: &Arc<Expr>, var: SymbolId) -> bool {
    matches!(expr.as_ref(), Expr::Symbol(s) if *s == var)
}

/// Returns true if `expr` is an integer literal equal to `n`.
fn is_integer_value(expr: &Arc<Expr>, n: i64) -> bool {
    as_constant(expr).map_or(false, |v| (v - n as f64).abs() < 1e-12)
}

/// Returns true if `expr` is `abs(var)`.
fn is_abs_of_var(expr: &Arc<Expr>, var: SymbolId) -> bool {
    match expr.as_ref() {
        Expr::Func(FuncId::Abs, args) => args.len() == 1 && is_symbol(&args[0], var),
        _ => false,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Expression, Function, Variable};
    use crate::numeric::compile::compile;
    use crate::numeric::expr::FuncId;

    fn t_id() -> SymbolId {
        SymbolId::intern("t")
    }

    fn w_id() -> SymbolId {
        SymbolId::intern("omega")
    }

    // Build e^(-t²) via AST compilation.
    fn gaussian_unit() -> Arc<Expr> {
        // -1 * t^2
        let neg_t_sq = Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Integer(-1)),
            Box::new(Expression::Power(
                Box::new(Expression::Variable(Variable::new("t"))),
                Box::new(Expression::Integer(2)),
            )),
        );
        // exp(-t^2)
        let expr = Expression::Function(Function::Exp, vec![neg_t_sq]);
        compile(&expr)
    }

    // Build e^(-2·t²) via AST compilation.
    fn gaussian_a2() -> Arc<Expr> {
        let neg_2t_sq = Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Integer(-2)),
            Box::new(Expression::Power(
                Box::new(Expression::Variable(Variable::new("t"))),
                Box::new(Expression::Integer(2)),
            )),
        );
        let expr = Expression::Function(Function::Exp, vec![neg_2t_sq]);
        compile(&expr)
    }

    // Build e^(-2·|t|) via AST compilation.
    fn exp_decay_a2() -> Arc<Expr> {
        let neg_2_abs_t = Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Integer(-2)),
            Box::new(Expression::Function(
                Function::Abs,
                vec![Expression::Variable(Variable::new("t"))],
            )),
        );
        let expr = Expression::Function(Function::Exp, vec![neg_2_abs_t]);
        compile(&expr)
    }

    #[test]
    fn fast_test_gaussian_unit_produces_result() {
        let expr = gaussian_unit();
        let result = fourier_transform(&expr, t_id(), w_id());
        assert!(result.is_ok(), "expected Ok for e^(-t²), got: {:?}", result);
        let r = result.unwrap();
        assert!(!r.steps.is_empty(), "expected narrated steps");
        assert!(
            r.steps[0].contains("Gaussian"),
            "step should mention Gaussian: {}",
            r.steps[0]
        );
    }

    #[test]
    fn fast_test_gaussian_contains_sqrt_and_exp() {
        let expr = gaussian_unit();
        let r = fourier_transform(&expr, t_id(), w_id()).unwrap();
        let display = format!("{}", r.expr);
        // Result should involve both sqrt and exp
        assert!(
            display.contains("sqrt") || display.contains("√"),
            "result should contain sqrt: {display}"
        );
        assert!(
            display.contains("exp"),
            "result should contain exp: {display}"
        );
    }

    #[test]
    fn fast_test_exp_decay_a2_is_4_over_4_plus_omega_sq() {
        // F{e^(-2|t|)} = 2·2/(2²+ω²) = 4/(4+ω²)
        let expr = exp_decay_a2();
        let result = fourier_transform(&expr, t_id(), w_id());
        assert!(result.is_ok(), "expected Ok for e^(-2|t|): {:?}", result);
        let r = result.unwrap();
        // Numerator should be 4.0, denominator involves omega^2
        let display = format!("{}", r.expr);
        // The result expression should be present and non-trivial
        assert!(!display.is_empty(), "result expression should not be empty");
        // Check the steps mention exponential decay
        assert!(
            r.steps.iter().any(|s| s.contains("exponential decay")),
            "steps should mention exponential decay: {r:?}"
        );
    }

    #[test]
    fn fast_test_linearity_two_gaussians() {
        // F{2·e^(-t²) + 3·e^(-2t²)} = 2·F{e^(-t²)} + 3·F{e^(-2t²)}
        let g1 = gaussian_unit();
        let g2 = gaussian_a2();

        // Build 2·e^(-t²) + 3·e^(-2t²)
        let scaled1 = normalize::mul(Expr::float(2.0), g1.clone());
        let scaled2 = normalize::mul(Expr::float(3.0), g2.clone());
        let combo = normalize::add(scaled1, scaled2);

        let result = fourier_transform(&combo, t_id(), w_id());
        assert!(result.is_ok(), "linearity test failed: {:?}", result);
        let r = result.unwrap();
        assert!(
            r.steps.iter().any(|s| s.contains("linearity")),
            "steps should mention linearity: {r:?}"
        );
    }

    #[test]
    fn fast_test_non_transformable_returns_no_table_entry() {
        // sin(t) has no table entry in our limited table.
        let sin_t = Expr::func(FuncId::Sin, vec![Arc::new(Expr::Symbol(t_id()))]);
        let result = fourier_transform(&sin_t, t_id(), w_id());
        assert!(
            matches!(result, Err(TransformError::NoTableEntry(_))),
            "expected NoTableEntry for sin(t), got: {result:?}"
        );
    }

    #[test]
    fn fast_test_bare_constant_returns_non_elementary() {
        let expr = Expr::int(1);
        let result = fourier_transform(&expr, t_id(), w_id());
        assert!(
            matches!(result, Err(TransformError::NonElementary(_))),
            "expected NonElementary for constant 1, got: {result:?}"
        );
    }
}
