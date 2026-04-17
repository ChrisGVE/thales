//! Multi-term linear factoring over `Expr`.
//!
//! When an `Expr::Add` has multiple terms containing the target variable,
//! try to factor it out: `Σ coeff_i · (rest_i · var) = other`
//! → `var = other / Σ coeff_i · rest_i`. Each term must be linear in `var`
//! (i.e. carry exactly one `var^1` factor and no other `var`-dependence).

use std::sync::Arc;

use crate::numeric::compile::decompile;
use crate::numeric::{normalize, BigRational, Expr, MulNode, SymbolId};
use crate::resolution_path::{Operation, ResolutionPathBuilder, StepAnnotation};

use super::super::helpers::contains_symbol;
use super::super::types::SolverError;
use super::unwrap::{finish_mul_like, rational_to_arc};

type Unwrapped = (Arc<Expr>, ResolutionPathBuilder);

/// Isolate `var` from a sum with multiple var-containing terms.
pub(super) fn collect_linear_var_terms(
    var_terms: &[(Arc<Expr>, BigRational)],
    other: &Arc<Expr>,
    var: SymbolId,
    path: ResolutionPathBuilder,
) -> Result<Unwrapped, SolverError> {
    let mut factors: Vec<Arc<Expr>> = Vec::new();
    for (term, coeff) in var_terms {
        match divide_out_var(term, var, coeff) {
            Some(f) => factors.push(f),
            None => {
                return Err(SolverError::CannotSolve(format!(
                    "Cannot isolate '{}': nonlinear term {}",
                    var.as_str(),
                    term
                )));
            }
        }
    }

    let combined_coeff = normalize::add_many(factors);
    if combined_coeff.is_zero() {
        return Err(SolverError::CannotSolve(format!(
            "Variable '{}' cancels — no unique solution",
            var.as_str()
        )));
    }

    let new_other = normalize::div(other.clone(), combined_coeff.clone());
    let coeff_expr = decompile(&combined_coeff);
    let new_other_expr = decompile(&new_other);
    let path = path.annotated_step(
        Operation::DivideBothSides(coeff_expr),
        format!("Collect terms and divide to isolate {}", var.as_str()),
        new_other_expr,
        StepAnnotation::elementary(),
    );
    Ok((new_other, path))
}

/// Return `coeff · (term / var)` as an `Arc<Expr>` when `term` is linear in
/// `var` (holds `var` as a simple `var^1` factor and nothing else involving
/// `var`). Returns `None` for any nonlinear form.
pub(super) fn divide_out_var(
    term: &Arc<Expr>,
    var: SymbolId,
    coeff: &BigRational,
) -> Option<Arc<Expr>> {
    match term.as_ref() {
        Expr::Symbol(s) if *s == var => Some(rational_to_arc(coeff.clone())),

        Expr::Mul(node) => {
            let mut has_var = false;
            let combined = &node.coeff * coeff;
            let mut result = MulNode::from_coeff(combined);
            for (base, exp) in &node.factors {
                if let Expr::Symbol(s) = base.as_ref() {
                    if *s == var {
                        match exp.as_ref() {
                            Expr::Integer(n) if n.to_i64() == Some(1) => {
                                has_var = true;
                                continue;
                            }
                            _ => return None,
                        }
                    }
                }
                if contains_symbol(base, var) || contains_symbol(exp, var) {
                    return None;
                }
                result.add_factor(base.clone(), exp.clone());
            }
            if !has_var {
                return None;
            }
            Some(finish_mul_like(result))
        }

        _ => None,
    }
}
