//! Coefficient extraction from linear expressions.
//!
//! Operates on `Arc<Expr>` internals. Input expressions must be compiled
//! from `Expression` via `crate::numeric::compile::compile` before being
//! passed here; callers at the Expression boundary are expected to perform
//! that compile step themselves (see `linear_system.rs`).
//!
//! The heavy lifting is already done by `Expr` normalization: a linear
//! combination like `2*x + 3*y - 5` normalizes to an `AddNode` whose
//! `terms` map directly carries each variable's rational coefficient, and
//! whose `constant` field carries the standalone term. Extraction is then a
//! straight map over that structure.

use std::sync::Arc;

use crate::ast::Variable;
use crate::numeric::{BigRational, Expr, SymbolId};

use super::helpers::contains_symbol;
use super::types::{SolverError, SolverResult};

/// Extract the scalar coefficient of `var` from a purely multiplicative
/// `term` and return it as an exact `BigRational`.
///
/// Accepts bare `Symbol`, and `Mul`/`Neg`-style products whose sole
/// `var`-bearing factor is `var` to the first power. Anything else (a
/// power ≥ 2, a product with multiple `var` factors, a function of `var`,
/// …) is reported as a non-linear term.
pub(super) fn extract_coefficient(term: &Arc<Expr>, var: SymbolId) -> SolverResult<BigRational> {
    match term.as_ref() {
        Expr::Symbol(s) if *s == var => Ok(BigRational::from(1_i64)),

        Expr::Mul(node) => {
            let mut var_power: u32 = 0;
            let mut scalar = node.coeff.clone();

            for (base, exp) in &node.factors {
                if contains_symbol(base, var) || contains_symbol(exp, var) {
                    // The only accepted shape here is `Symbol(var)` raised
                    // to the integer literal 1; Expr normalization keeps a
                    // first-power factor in canonical form but we defend
                    // against more complex shapes to catch non-linearity.
                    let is_target_symbol = matches!(
                        base.as_ref(),
                        Expr::Symbol(s) if *s == var
                    );
                    let exp_is_one = matches!(
                        exp.as_ref(),
                        Expr::Integer(n) if n.to_i64() == Some(1)
                    );
                    if !(is_target_symbol && exp_is_one) || var_power > 0 {
                        return Err(SolverError::Other(format!("Non-linear term: {}", term)));
                    }
                    var_power = 1;
                } else {
                    // Constant factor relative to `var`. Must be a numeric
                    // literal for the linear coefficient to be scalar.
                    let Some(factor_val) = pure_rational_factor(base, exp) else {
                        return Err(SolverError::Other(format!(
                            "Cannot evaluate coefficient factor: {} ^ {}",
                            base, exp
                        )));
                    };
                    scalar = &scalar * &factor_val;
                }
            }

            if var_power == 0 {
                // `term` is a constant Mul; should be treated as a constant
                // contribution rather than a coefficient of `var`.
                return Err(SolverError::Other(format!(
                    "Term does not contain {}: {}",
                    var.as_str(),
                    term
                )));
            }
            Ok(scalar)
        }

        _ => {
            if contains_symbol(term, var) {
                Err(SolverError::Other(format!(
                    "Cannot extract coefficient from: {}",
                    term
                )))
            } else {
                Ok(BigRational::from(0_i64))
            }
        }
    }
}

/// Attempt to reduce `base ^ exp` to an exact rational, when both are
/// numeric literals and the exponent is a non-negative integer.
fn pure_rational_factor(base: &Arc<Expr>, exp: &Arc<Expr>) -> Option<BigRational> {
    let base_val = expr_to_rational(base)?;
    match exp.as_ref() {
        Expr::Integer(n) => {
            let n = n.to_i64()?;
            if n >= 0 {
                let p: u32 = n.try_into().ok()?;
                Some(base_val.pow_u32(p))
            } else {
                let p: u32 = (-n).try_into().ok()?;
                Some(base_val.pow_u32(p).recip())
            }
        }
        _ => None,
    }
}

/// Return `Some(value)` when `expr` is an exact rational literal.
fn expr_to_rational(expr: &Arc<Expr>) -> Option<BigRational> {
    match expr.as_ref() {
        Expr::Integer(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

/// Extract `(variable_coefficients, constant_term)` from a linear
/// expression.
///
/// Coefficients are returned in the order of `variables`. Any expression
/// containing a tracked variable outside a linear position, or any
/// non-numeric constant term, yields a `SolverError`.
pub(super) fn extract_linear_coefficients(
    expr: &Arc<Expr>,
    variables: &[Variable],
) -> SolverResult<(Vec<BigRational>, BigRational)> {
    let var_ids: Vec<SymbolId> = variables
        .iter()
        .map(|v| SymbolId::intern(&v.name))
        .collect();

    let mut coeffs = vec![BigRational::from(0_i64); variables.len()];
    let mut constant = BigRational::from(0_i64);

    match expr.as_ref() {
        Expr::Add(node) => {
            constant = &constant + &node.constant;
            for (base, coeff) in &node.terms {
                apply_linear_term(base, coeff, &var_ids, &mut coeffs)?;
            }
        }
        Expr::Integer(_) | Expr::Rational(_) => {
            if let Some(val) = expr_to_rational(expr) {
                constant = &constant + &val;
            }
        }
        Expr::Symbol(s) => {
            if let Some(i) = var_ids.iter().position(|v| v == s) {
                coeffs[i] = &coeffs[i] + &BigRational::from(1_i64);
            } else {
                return Err(SolverError::Other(format!(
                    "Non-numeric constant term: {}",
                    s.as_str()
                )));
            }
        }
        Expr::Mul(_) => {
            // Single multiplicative term: identify its variable, if any.
            let maybe_var = var_ids.iter().find(|&&v| contains_symbol(expr, v));
            if let Some(&v) = maybe_var {
                let i = var_ids.iter().position(|&id| id == v).unwrap();
                let c = extract_coefficient(expr, v)?;
                coeffs[i] = &coeffs[i] + &c;
            } else if let Some(val) = expr_to_rational(expr) {
                constant = &constant + &val;
            } else {
                return Err(SolverError::Other(format!(
                    "Cannot evaluate constant term: {}",
                    expr
                )));
            }
        }
        _ => {
            if var_ids.iter().any(|&v| contains_symbol(expr, v)) {
                return Err(SolverError::Other(format!("Non-linear term: {}", expr)));
            }
            return Err(SolverError::Other(format!(
                "Cannot evaluate constant term: {}",
                expr
            )));
        }
    }

    Ok((coeffs, constant))
}

fn apply_linear_term(
    base: &Arc<Expr>,
    coeff: &BigRational,
    var_ids: &[SymbolId],
    coeffs: &mut [BigRational],
) -> SolverResult<()> {
    if let Expr::Symbol(s) = base.as_ref() {
        if let Some(i) = var_ids.iter().position(|v| v == s) {
            coeffs[i] = &coeffs[i] + coeff;
            return Ok(());
        }
        return Err(SolverError::Other(format!(
            "Non-numeric coefficient for symbol {}",
            s.as_str()
        )));
    }

    Err(SolverError::Other(format!(
        "Non-linear or unsupported term: {}",
        base
    )))
}
