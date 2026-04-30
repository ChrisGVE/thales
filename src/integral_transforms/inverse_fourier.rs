//! Inverse Fourier transform: symbolic table lookup and algebraic properties.
//!
//! Convention: f(t) = (1/2π) ∫_{-∞}^{∞} F(ω) e^{iωt} dω
//!
//! # Table entries
//!
//! - Lorentzian:  2a/(a²+ω²)            → e^{-a|t|}      (a > 0)
//! - Gaussian:    √(π/a)·e^{-ω²/(4a)}  → e^{-at²}       (a > 0)
//!
//! Linear combinations are handled via `split_linear_terms`.

use std::sync::Arc;

use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::normalize;
use crate::numeric::SymbolId;
use num::traits::One;

use super::{as_constant, contains_var, split_linear_terms, TransformError, TransformResult};

// ── Public entry point ────────────────────────────────────────────────────────

/// Compute the inverse Fourier transform f(t) = (1/2π) ∫ F(ω) e^{iωt} dω.
///
/// Applies table lookup with linearity. Returns a [`TransformResult`] on
/// success or a [`TransformError`] when the expression is not in the table.
pub fn inverse_fourier(
    expr: &Arc<Expr>,
    omega_var: SymbolId,
    t_var: SymbolId,
) -> Result<TransformResult, TransformError> {
    let mut steps: Vec<String> = Vec::new();

    // Fast path: expression does not depend on ω at all.
    if !contains_var(expr, omega_var) {
        return Err(TransformError::NonElementary(
            "bare constant has distributional inverse Fourier transform (delta function); \
             not supported"
                .into(),
        ));
    }

    // Try direct table lookup first.
    if let Some(result_expr) = try_table(expr, omega_var, t_var, &mut steps) {
        return Ok(TransformResult {
            expr: result_expr,
            domain_var: t_var.as_str(),
            convergence: None,
            steps,
        });
    }

    // Linearity: split into (coefficient, term) pairs and transform each.
    let terms = split_linear_terms(expr, omega_var);
    if terms.len() > 1 {
        return apply_linearity(&terms, omega_var, t_var, steps);
    }

    Err(TransformError::NoTableEntry(format!(
        "no inverse Fourier transform table entry for: {expr}"
    )))
}

// ── Linearity ─────────────────────────────────────────────────────────────────

/// Apply linearity: F^{-1}{a·F + b·G + …} = a·f + b·g + …
fn apply_linearity(
    terms: &[(f64, Arc<Expr>)],
    omega_var: SymbolId,
    t_var: SymbolId,
    mut steps: Vec<String>,
) -> Result<TransformResult, TransformError> {
    steps.push("Applied linearity of inverse Fourier transform.".into());

    let mut result_expr: Option<Arc<Expr>> = None;

    for (coeff, term) in terms {
        let term_result = inverse_fourier(term, omega_var, t_var)?;

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
        domain_var: t_var.as_str(),
        convergence: None,
        steps,
    })
}

// ── Table lookup ──────────────────────────────────────────────────────────────

/// Attempt a direct table match. Returns `Some(result_expr)` on hit.
fn try_table(
    expr: &Arc<Expr>,
    omega_var: SymbolId,
    t_var: SymbolId,
    steps: &mut Vec<String>,
) -> Option<Arc<Expr>> {
    // Try Lorentzian: 2a / (a² + ω²)
    if let Some(result) = try_lorentzian(expr, omega_var, t_var, steps) {
        return Some(result);
    }
    // Try Gaussian: √(π/a) · e^{-ω²/(4a)}
    if let Some(result) = try_gaussian_spectrum(expr, omega_var, t_var, steps) {
        return Some(result);
    }
    None
}

// ── Lorentzian: 2a/(a²+ω²) → e^{-a|t|} ──────────────────────────────────────

/// Match 2a/(a²+ω²) with a > 0 and return e^{-a|t|}.
///
/// Canonical form after `normalize::div(float(2a), add(float(a²), pow(ω,2)))`:
///   `Mul { coeff=2a, factors={ Pow(Add(a²+ω²), -1) → 1 } }`
/// but the normalizer may absorb the float coefficient into `MulNode.coeff`
/// and represent the denominator as `Pow(denom_expr, -1)`.
fn try_lorentzian(
    expr: &Arc<Expr>,
    omega_var: SymbolId,
    t_var: SymbolId,
    steps: &mut Vec<String>,
) -> Option<Arc<Expr>> {
    // The canonical representation is: coefficient * denom^(-1)
    // where coefficient = 2a and denom = a² + ω²
    let (numerator_coeff, denom_expr) = extract_ratio(expr, omega_var)?;

    if numerator_coeff <= 0.0 {
        return None;
    }

    // denom_expr must be a² + ω², i.e. Add with constant a² and one term ω²
    let a = extract_lorentzian_param(denom_expr, omega_var)?;

    // F^{-1}{c / (a²+ω²)} = (c / (2a)) · e^{-a|t|}
    // The canonical table entry is c = 2a giving scale = 1, but any positive c
    // is valid — the result is simply scaled by c/(2a).
    let scale = numerator_coeff / (2.0 * a);

    // Build e^{-a·|t|}
    let t = Arc::new(Expr::Symbol(t_var));
    let abs_t = Expr::func(FuncId::Abs, vec![t]);
    let neg_a_abs_t = normalize::mul(Expr::float(-a), abs_t);
    let exp_decay = Expr::func(FuncId::Exp, vec![neg_a_abs_t]);

    let result = if (scale - 1.0).abs() < 1e-12 {
        exp_decay
    } else {
        normalize::mul(Expr::float(scale), exp_decay)
    };

    steps.push(format!(
        "Applied inverse Fourier transform of Lorentzian: \
         F^{{-1}}{{{numerator_coeff}/({}+ω²)}} = {scale}·e^(-{a}·|t|)",
        a * a
    ));

    Some(result)
}

/// Extract `(numerator_coeff, denominator_expr)` from an expression of the
/// form `c * denom^(-1)`.
///
/// Handles two normalized layouts produced by `normalize::div`:
///
/// 1. `Mul { coeff=1, factors={ Float(c)^1, Add(denom)^(-1) } }` — the
///    numerator float and the denominator are both factors in the MulNode.
/// 2. `Mul { coeff=c, factors={ Add(denom)^(-1) } }` — numerator absorbed
///    into `coeff` (less common after normalisation of integer numerators).
/// 3. `Pow(base, -1)` — plain reciprocal, numerator = 1.
///
/// Returns `None` if the expression does not match this pattern or depends
/// on more than one ω-containing factor in the denominator.
fn extract_ratio(expr: &Arc<Expr>, omega_var: SymbolId) -> Option<(f64, &Arc<Expr>)> {
    match expr.as_ref() {
        // Mul node: coeff * prod(base^exp)
        Expr::Mul(node) => {
            let node_coeff = node.coeff.to_f64();
            if node_coeff <= 0.0 {
                return None;
            }

            // Layout 1: two factors — one is a positive numeric scalar^n
            // (the numerator), the other is Add(denom)^(-1).
            // The numeric scalar may carry any positive-integer exponent when
            // produced by chained multiplications (e.g. Float(2)^2 = 4).
            if node.factors.len() == 2 {
                let mut num_val: Option<f64> = None;
                let mut denom_base: Option<&Arc<Expr>> = None;

                for (base, exp) in &node.factors {
                    if is_integer_value(exp, -1) && contains_var(base, omega_var) {
                        denom_base = Some(base);
                    } else if let Some(v) = eval_numeric_power(base, exp) {
                        if v > 0.0 {
                            num_val = Some(v);
                        }
                    }
                }

                if let (Some(n), Some(d)) = (num_val, denom_base) {
                    return Some((node_coeff * n, d));
                }
            }

            // Layout 2: exactly one factor denom^(-1), numerator in coeff.
            if node.factors.len() == 1 {
                let (base, exp) = node.factors.iter().next()?;
                if !is_integer_value(exp, -1) {
                    return None;
                }
                if !contains_var(base, omega_var) {
                    return None;
                }
                return Some((node_coeff, base));
            }

            None
        }
        // Pow node: expr^(-1) (i.e. 1/expr) — coeff = 1
        Expr::Pow(base, exp) => {
            if !is_integer_value(exp, -1) {
                return None;
            }
            if !contains_var(base, omega_var) {
                return None;
            }
            Some((1.0, base))
        }
        _ => None,
    }
}

/// Return the numeric value of a pure-numeric Expr (Float or Integer), or
/// `None` if the expression contains any symbolic part.
fn as_pure_numeric(expr: &Arc<Expr>) -> Option<f64> {
    match expr.as_ref() {
        Expr::Float(v) => Some(*v),
        Expr::Integer(n) => n.to_i64().map(|i| i as f64),
        Expr::Rational(r) => Some(r.to_f64()),
        _ => None,
    }
}

/// Evaluate `base ^ exp` when both are pure-numeric, returning `None` if
/// either contains a symbolic component or the result would not be finite.
///
/// Only non-negative integer or float exponents are accepted as "numerator"
/// candidates (the denominator factor is identified by `exp == -1`).
fn eval_numeric_power(base: &Arc<Expr>, exp: &Arc<Expr>) -> Option<f64> {
    let b = as_pure_numeric(base)?;
    let e = as_pure_numeric(exp)?;
    if e < 0.0 {
        return None; // denominator factor — handled separately
    }
    let result = b.powf(e);
    if result.is_finite() {
        Some(result)
    } else {
        None
    }
}

/// Extract parameter `a` from a denominator of the form `a² + ω²`.
///
/// Two normalized layouts are handled:
///
/// **Layout A** — constant in `AddNode.constant`, one symbolic term `ω²`:
/// ```text
/// Add { constant: a², terms: { Pow(ω,2): 1 } }
/// ```
/// Produced when the constant addend is an integer (e.g. `1 + ω²`).
///
/// **Layout B** — constant stored as a Float *key* in `terms`, alongside `ω²`:
/// ```text
/// Add { constant: 0, terms: { Float(a²): 1, Pow(ω,2): 1 } }
/// ```
/// Produced when the constant addend is a float (e.g. `4.0 + ω²`).
fn extract_lorentzian_param(denom: &Arc<Expr>, omega_var: SymbolId) -> Option<f64> {
    match denom.as_ref() {
        Expr::Add(node) => {
            let raw_const = node.constant.to_f64();

            // ── Layout A: constant in `node.constant`, one term ω² ────────────
            if raw_const > 0.0 && node.terms.len() == 1 {
                let (term_key, term_coeff) = node.terms.iter().next()?;
                if (term_coeff.to_f64() - 1.0).abs() <= 1e-12 && is_var_squared(term_key, omega_var)
                {
                    return Some(raw_const.sqrt());
                }
            }

            // ── Layout B: constant stored as Float key in terms ────────────────
            // AddNode has constant=0 and two terms: Float(a²)^1 and Pow(ω,2)^1.
            if (raw_const).abs() < 1e-15 && node.terms.len() == 2 {
                let mut float_const: Option<f64> = None;
                let mut has_omega_sq = false;

                for (term_key, term_coeff) in &node.terms {
                    if (term_coeff.to_f64() - 1.0).abs() > 1e-12 {
                        return None; // unexpected coefficient
                    }
                    if is_var_squared(term_key, omega_var) {
                        has_omega_sq = true;
                    } else if let Some(v) = as_pure_numeric(term_key) {
                        if v > 0.0 {
                            float_const = Some(v);
                        }
                    }
                }

                if has_omega_sq {
                    if let Some(a_sq) = float_const {
                        return Some(a_sq.sqrt());
                    }
                }
            }

            None
        }
        _ => None,
    }
}

// ── Gaussian spectrum: √(π/a)·e^{-ω²/(4a)} → e^{-at²} ───────────────────────

/// Match √(π/a)·e^{-ω²/(4a)} and return e^{-a·t²}.
///
/// Canonical form: Mul { coeff=√(π/a), factors={ Exp(-ω²/(4a)) → 1 } }
/// or as a product of two factors at the Mul level.
fn try_gaussian_spectrum(
    expr: &Arc<Expr>,
    omega_var: SymbolId,
    t_var: SymbolId,
    steps: &mut Vec<String>,
) -> Option<Arc<Expr>> {
    // Extract (amplitude_coeff, exp_arg) where amplitude_coeff = √(π/a)
    // and exp_arg = -ω²/(4a).
    match expr.as_ref() {
        Expr::Mul(node) => {
            // Two factors: Sqrt(π/a) and Exp(-ω²/(4a))
            // After normalization Sqrt is likely a factor with exponent 1.
            let a = extract_gaussian_spectrum_param(node, omega_var)?;

            // Build e^{-a·t²}
            let t = Arc::new(Expr::Symbol(t_var));
            let t_sq = normalize::pow(t, Expr::int(2));
            let neg_a_t_sq = normalize::mul(Expr::float(-a), t_sq);
            let result = Expr::func(FuncId::Exp, vec![neg_a_t_sq]);

            steps.push(format!(
                "Applied inverse Fourier transform of Gaussian spectrum: \
                 F^{{-1}}{{√(π/{a})·e^(-ω²/(4·{a}))}} = e^(-{a}·t²)"
            ));

            Some(result)
        }
        // Handle case where expression is just Exp(-ω²/(4a)) — amplitude = 1, so a = π
        Expr::Func(FuncId::Exp, args) if args.len() == 1 => {
            let a = extract_neg_omega_sq_over_4a(&args[0], omega_var)?;
            // amplitude = 1 means √(π/a) = 1 → π/a = 1 → a = π
            let expected_a = std::f64::consts::PI;
            if (a - expected_a).abs() > 1e-9 * expected_a {
                return None;
            }
            let t = Arc::new(Expr::Symbol(t_var));
            let t_sq = normalize::pow(t, Expr::int(2));
            let neg_a_t_sq = normalize::mul(Expr::float(-a), t_sq);
            let result = Expr::func(FuncId::Exp, vec![neg_a_t_sq]);

            steps.push(format!(
                "Applied inverse Fourier transform of Gaussian spectrum: \
                 F^{{-1}}{{e^(-ω²/(4·{a}))}} = e^(-{a}·t²)"
            ));

            Some(result)
        }
        _ => None,
    }
}

/// Extract `a` from a MulNode of the form `√(π/a) * Exp(-ω²/(4a))`.
///
/// The amplitude `√(π/a)` may appear as:
/// - A `Func(Sqrt, ...)` factor with exponent 1.
/// - A `Pow(something, 1/2)` factor.
/// - A plain `Float` factor with exponent 1 (the common layout after
///   `normalize::mul(Expr::float(amplitude), exp_part)`).
/// - Absorbed into `MulNode.coeff` (less common).
fn extract_gaussian_spectrum_param(
    node: &crate::numeric::MulNode,
    omega_var: SymbolId,
) -> Option<f64> {
    // Find the Exp factor and extract a from its exponent.
    let mut a_from_exp: Option<f64> = None;
    let mut has_amplitude_factor = false;
    let mut float_amplitude: Option<f64> = None;

    for (base, exp) in &node.factors {
        if !is_integer_value(exp, 1) {
            // Check for Pow(something, 1/2) — sqrt encoded as ^(1/2)
            if is_half_exponent(exp) {
                has_amplitude_factor = true;
                continue;
            }
            continue;
        }
        match base.as_ref() {
            Expr::Func(FuncId::Exp, args) if args.len() == 1 => {
                a_from_exp = extract_neg_omega_sq_over_4a(&args[0], omega_var);
            }
            Expr::Func(FuncId::Sqrt, _) => {
                has_amplitude_factor = true;
            }
            Expr::Float(v) if *v > 0.0 => {
                // Plain float factor — may be the numeric amplitude √(π/a).
                float_amplitude = Some(*v);
                has_amplitude_factor = true;
            }
            Expr::Integer(n) => {
                if let Some(i) = n.to_i64() {
                    if i > 0 {
                        float_amplitude = Some(i as f64);
                        has_amplitude_factor = true;
                    }
                }
            }
            _ => {}
        }
    }

    let a = a_from_exp?;

    if has_amplitude_factor {
        // If we have a float amplitude, verify it equals √(π/a).
        if let Some(amp) = float_amplitude {
            let expected_amplitude = (std::f64::consts::PI / a).sqrt();
            if (amp - expected_amplitude).abs() > 1e-6 * expected_amplitude.max(1.0) {
                return None;
            }
        }
        // For Sqrt / Pow(x,1/2) factors we trust the Exp-derived `a`.
        return Some(a);
    }

    // No explicit amplitude factor — check MulNode.coeff encodes √(π/a).
    let coeff = node.coeff.to_f64();
    let expected_amplitude = (std::f64::consts::PI / a).sqrt();
    if (coeff - expected_amplitude).abs() > 1e-6 * expected_amplitude.max(1.0) {
        return None;
    }
    Some(a)
}

/// Extract `a` from an exponent of the form `-ω²/(4a)`.
///
/// Two normalized layouts are handled:
///
/// **Layout A** — scalar absorbed into `coeff`:
/// ```text
/// Mul { coeff: -1/(4a), factors: { Pow(ω,2)^1 } }
/// ```
///
/// **Layout B** — scalar stored as a Float *factor*:
/// ```text
/// Mul { coeff: -1, factors: { Float(1/(4a))^1, Pow(ω,2)^1 } }
/// ```
/// This is produced by `normalize::div(normalize::neg(ω²), float(4a))`.
fn extract_neg_omega_sq_over_4a(exponent: &Arc<Expr>, omega_var: SymbolId) -> Option<f64> {
    match exponent.as_ref() {
        Expr::Mul(node) => {
            let coeff = node.coeff.to_f64();
            if coeff >= 0.0 {
                return None; // must be negative
            }

            // ── Layout A: one factor, ω² ──────────────────────────────────────
            if node.factors.len() == 1 {
                let (base, exp) = node.factors.iter().next()?;
                if !is_integer_value(exp, 1) {
                    return None;
                }
                if !is_var_squared(base, omega_var) {
                    return None;
                }
                // coeff = -1/(4a) → a = -1/(4*coeff)
                let a = -1.0 / (4.0 * coeff);
                if a <= 0.0 {
                    return None;
                }
                return Some(a);
            }

            // ── Layout B: two factors — Float scalar and Pow(ω,2) ────────────
            if node.factors.len() == 2 {
                let mut scalar: Option<f64> = None;
                let mut has_omega_sq = false;

                for (base, exp) in &node.factors {
                    if !is_integer_value(exp, 1) {
                        return None;
                    }
                    if is_var_squared(base, omega_var) {
                        has_omega_sq = true;
                    } else if let Some(v) = as_pure_numeric(base) {
                        if v > 0.0 {
                            scalar = Some(v);
                        }
                    }
                }

                if has_omega_sq {
                    if let Some(s) = scalar {
                        // Overall factor on ω² is coeff * s (both negative * positive).
                        // total_factor = coeff * s = -1/(4a) → a = -1/(4 * coeff * s)
                        let total = coeff * s;
                        if total >= 0.0 {
                            return None;
                        }
                        let a = -1.0 / (4.0 * total);
                        if a > 0.0 {
                            return Some(a);
                        }
                    }
                }
            }

            None
        }
        _ => None,
    }
}

// ── Structural helpers ────────────────────────────────────────────────────────

/// Returns true if `expr` is an integer literal equal to `n`.
fn is_integer_value(expr: &Arc<Expr>, n: i64) -> bool {
    as_constant(expr).map_or(false, |v| (v - n as f64).abs() < 1e-12)
}

/// Returns true if `expr` is a `Pow(var, 2)` or equivalent.
fn is_var_squared(expr: &Arc<Expr>, var: SymbolId) -> bool {
    match expr.as_ref() {
        Expr::Pow(base, exp) => {
            matches!(base.as_ref(), Expr::Symbol(s) if *s == var) && is_integer_value(exp, 2)
        }
        // In MulNode a squared var may appear as a key with exponent 2 in a
        // nested Mul. Check for Mul { coeff=1, factors={var → 2} }.
        Expr::Mul(node) => {
            if node.factors.len() != 1 || !node.coeff.is_one() {
                return false;
            }
            let (base, exp) = node.factors.iter().next().unwrap();
            matches!(base.as_ref(), Expr::Symbol(s) if *s == var) && is_integer_value(exp, 2)
        }
        _ => false,
    }
}

/// Returns true if `expr` represents 1/2 (half exponent, i.e. sqrt).
fn is_half_exponent(expr: &Arc<Expr>) -> bool {
    match expr.as_ref() {
        Expr::Rational(r) => {
            let (n, d) = (r.numer(), r.denom());
            n.to_i64() == Some(1) && d.to_i64() == Some(2)
        }
        Expr::Float(f) => (f - 0.5).abs() < 1e-12,
        _ => false,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Expression, Function, Variable};
    use crate::numeric::compile::compile;

    fn omega_id() -> SymbolId {
        SymbolId::intern("omega")
    }

    fn t_id() -> SymbolId {
        SymbolId::intern("t")
    }

    /// Build 2/(1+ω²) by compiling (2·1) / (1 + ω²) via normalize.
    fn lorentzian_unit() -> Arc<Expr> {
        // numerator = 2, denominator = 1 + ω²
        let numerator = Expr::float(2.0);
        let omega_sq = normalize::pow(Arc::new(Expr::Symbol(omega_id())), Expr::int(2));
        let denom = normalize::add(Expr::int(1), omega_sq);
        normalize::div(numerator, denom)
    }

    /// Build 4/(4+ω²) — corresponds to a=2.
    fn lorentzian_a2() -> Arc<Expr> {
        let numerator = Expr::float(4.0);
        let omega_sq = normalize::pow(Arc::new(Expr::Symbol(omega_id())), Expr::int(2));
        let denom = normalize::add(Expr::float(4.0), omega_sq);
        normalize::div(numerator, denom)
    }

    #[test]
    fn debug_lorentzian_structure() {
        let expr = lorentzian_unit();
        eprintln!("LORENTZIAN DEBUG: {:?}", expr);
        let expr2 = lorentzian_a2();
        eprintln!("LORENTZIAN_A2 DEBUG: {:?}", expr2);
        // Gaussian
        use crate::numeric::expr::FuncId;
        let a = 1.0f64;
        let omega_sq = normalize::pow(Arc::new(Expr::Symbol(omega_id())), Expr::int(2));
        let four_a = Expr::float(4.0 * a);
        let neg_exp_arg = normalize::div(normalize::neg(omega_sq), four_a);
        let exp_part = Expr::func(FuncId::Exp, vec![neg_exp_arg]);
        let amplitude = (std::f64::consts::PI / a).sqrt();
        let full_expr = normalize::mul(Expr::float(amplitude), exp_part);
        eprintln!("GAUSSIAN SPEC DEBUG: {:?}", full_expr);
    }

    #[test]
    fn fast_test_lorentzian_unit_returns_exp_decay() {
        let expr = lorentzian_unit();
        let result = inverse_fourier(&expr, omega_id(), t_id());
        assert!(
            result.is_ok(),
            "expected Ok for 2/(1+ω²), got: {:?}",
            result
        );
        let r = result.unwrap();
        assert!(!r.steps.is_empty(), "expected narrated steps");
        assert!(
            r.steps[0].contains("Lorentzian"),
            "step should mention Lorentzian: {}",
            r.steps[0]
        );
    }

    #[test]
    fn fast_test_lorentzian_unit_at_t0_is_one() {
        // F^{-1}{2/(1+ω²)} = e^{-|t|}, so at t=0 → e^0 = 1.0
        use crate::numeric::evaluation::evaluate;
        use std::collections::HashMap;
        let expr = lorentzian_unit();
        let r = inverse_fourier(&expr, omega_id(), t_id()).unwrap();
        let mut bindings = HashMap::new();
        bindings.insert(t_id(), 0.0);
        let val = evaluate(&r.expr, &bindings).expect("evaluation should succeed at t=0");
        assert!(
            (val - 1.0).abs() < 1e-10,
            "expected e^0 = 1.0 at t=0, got {val}"
        );
    }

    #[test]
    fn fast_test_lorentzian_unit_at_t1_is_exp_neg1() {
        // F^{-1}{2/(1+ω²)} = e^{-|t|}, so at t=1 → e^{-1} ≈ 0.3679
        use crate::numeric::evaluation::evaluate;
        use std::collections::HashMap;
        let expr = lorentzian_unit();
        let r = inverse_fourier(&expr, omega_id(), t_id()).unwrap();
        let mut bindings = HashMap::new();
        bindings.insert(t_id(), 1.0);
        let val = evaluate(&r.expr, &bindings).expect("evaluation should succeed at t=1");
        let expected = (-1.0f64).exp();
        assert!(
            (val - expected).abs() < 1e-10,
            "expected e^(-1) ≈ {expected} at t=1, got {val}"
        );
    }

    #[test]
    fn fast_test_lorentzian_a2_returns_exp_decay_a2() {
        // F^{-1}{4/(4+ω²)} = e^{-2|t|}
        let expr = lorentzian_a2();
        let result = inverse_fourier(&expr, omega_id(), t_id());
        assert!(
            result.is_ok(),
            "expected Ok for 4/(4+ω²), got: {:?}",
            result
        );
        let r = result.unwrap();
        let display = format!("{}", r.expr);
        assert!(
            display.contains("exp"),
            "result should contain exp: {display}"
        );
    }

    #[test]
    fn fast_test_no_table_entry_for_sin() {
        let sin_omega = Expr::func(FuncId::Sin, vec![Arc::new(Expr::Symbol(omega_id()))]);
        let result = inverse_fourier(&sin_omega, omega_id(), t_id());
        assert!(
            matches!(result, Err(TransformError::NoTableEntry(_))),
            "expected NoTableEntry for sin(ω), got: {result:?}"
        );
    }

    #[test]
    fn fast_test_bare_constant_returns_non_elementary() {
        let expr = Expr::int(1);
        let result = inverse_fourier(&expr, omega_id(), t_id());
        assert!(
            matches!(result, Err(TransformError::NonElementary(_))),
            "expected NonElementary for constant 1, got: {result:?}"
        );
    }

    #[test]
    fn fast_test_linearity_two_lorentzians() {
        // F^{-1}{2·(2/(1+ω²)) + 3·(4/(4+ω²))} = 2·e^{-|t|} + 3·e^{-2|t|}
        let l1 = lorentzian_unit();
        let l2 = lorentzian_a2();
        let scaled1 = normalize::mul(Expr::float(2.0), l1);
        let scaled2 = normalize::mul(Expr::float(3.0), l2);
        let combo = normalize::add(scaled1, scaled2);

        let result = inverse_fourier(&combo, omega_id(), t_id());
        assert!(result.is_ok(), "linearity test failed: {:?}", result);
        let r = result.unwrap();
        assert!(
            r.steps.iter().any(|s| s.contains("linearity")),
            "steps should mention linearity: {r:?}"
        );
    }

    // ── Compile-path test (Expression AST → compile → inverse_fourier) ────────

    #[test]
    fn fast_test_lorentzian_via_compile() {
        // Build 2/(1+ω²) via the Expression AST path to verify compile round-trip.
        let omega_var = Expression::Variable(Variable::new("omega"));
        let omega_sq = Expression::Power(
            Box::new(omega_var.clone()),
            Box::new(Expression::Integer(2)),
        );
        let denom = Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Integer(1)),
            Box::new(omega_sq),
        );
        let expr_ast = Expression::Binary(
            BinaryOp::Div,
            Box::new(Expression::Integer(2)),
            Box::new(denom),
        );
        let compiled = compile(&expr_ast);
        let result = inverse_fourier(&compiled, omega_id(), t_id());
        assert!(
            result.is_ok(),
            "compile-path lorentzian should succeed, got: {:?}",
            result
        );
    }

    // ── Gaussian spectrum (inverse) ───────────────────────────────────────────

    #[test]
    fn fast_test_gaussian_spectrum_produces_result() {
        // Build √(π/1) · e^{-ω²/4} via normalize (a=1)
        use crate::numeric::expr::FuncId;
        let a = 1.0f64;
        let omega_sq = normalize::pow(Arc::new(Expr::Symbol(omega_id())), Expr::int(2));
        let four_a = Expr::float(4.0 * a);
        let neg_exp_arg = normalize::div(normalize::neg(omega_sq), four_a);
        let exp_part = Expr::func(FuncId::Exp, vec![neg_exp_arg]);
        let amplitude = (std::f64::consts::PI / a).sqrt();
        let full_expr = normalize::mul(Expr::float(amplitude), exp_part);

        let result = inverse_fourier(&full_expr, omega_id(), t_id());
        assert!(
            result.is_ok(),
            "expected Ok for Gaussian spectrum, got: {:?}",
            result
        );
        let r = result.unwrap();
        assert!(
            r.steps.iter().any(|s| s.contains("Gaussian")),
            "steps should mention Gaussian: {r:?}"
        );
    }
}
