//! Cross-multiplication and rational-equation handling in `Expr`.
//!
//! Activated from [`super::unwrap::unwrap_mul`] when a `Mul` carries more
//! than one var-containing factor (e.g. `R1 · R2 · (R1+R2)^(-1)`).
//! Rearranges `Π base_i^exp_i = other` as `numer − other·denom = 0`,
//! distributes products over sums via [`expand`], then hands the resulting
//! polynomial equation back to the general unwrap engine. Finally checks
//! the candidate solution against any denominator that contains `var` and
//! rejects it if it produces an undefined expression (extraneous root).

use std::sync::Arc;

use num::traits::Zero;

use crate::numeric::compile::decompile;
use crate::numeric::trace::{Step, TechniqueTag, Trace};
use crate::numeric::{normalize, Expr, MulNode, SymbolId};

use super::super::types::SolverError;
use super::unwrap::{finish_mul_like, rational_to_arc, unwrap_variable};

/// Cross-multiply a `Mul` factorization when multiple factors contain `var`.
///
/// `var_factors` is the list of `(base, exp)` pairs from the `MulNode`
/// where either `base` or `exp` contains `var`. Numeric coefficient and
/// non-var factors have already been divided out of `other`.
pub(super) fn try_cross_multiply_mul(
    var_factors: &[(Arc<Expr>, Arc<Expr>)],
    other: &Arc<Expr>,
    var: SymbolId,
    trace: &mut Trace,
) -> Result<Arc<Expr>, SolverError> {
    // Split factors by sign of the integer exponent: negative → denom.
    // Flatten `Pow(b, e)` factors so the inner exponent participates in the
    // sign decision (e.g. `Pow(x+1, -1)` with outer exp `1` is a denominator).
    let mut numer_node = MulNode::one();
    let mut denom_node = MulNode::one();
    for (base, exp) in var_factors {
        let (b, e) = if let Expr::Pow(inner_b, inner_e) = base.as_ref() {
            (
                inner_b.clone(),
                normalize::mul(inner_e.clone(), exp.clone()),
            )
        } else {
            (base.clone(), exp.clone())
        };
        match e.as_ref() {
            Expr::Integer(n) if n.to_i64().map_or(false, |v| v < 0) => {
                let pos_exp = Arc::new(Expr::Integer(-n));
                denom_node.add_factor(b, pos_exp);
            }
            _ => {
                numer_node.add_factor(b, e);
            }
        }
    }
    let numer = finish_mul_like(numer_node);
    let denom = finish_mul_like(denom_node);

    // No denominator to clear means the factorization is purely
    // multiplicative in `var` (e.g. `x · exp(x)`); cross-multiplication
    // would not make progress and would recurse forever.
    if denom.is_one() {
        return Err(SolverError::CannotSolve(format!(
            "Cannot isolate '{}': variable appears in multiple numerator factors",
            var.as_str()
        )));
    }

    // numer = other · denom  →  numer − other·denom = 0
    let rhs = expand(&normalize::mul(other.clone(), denom.clone()));
    let cleared = normalize::sub(expand(&numer), rhs);

    let denom_expr = decompile(&denom);
    trace.push(
        Step::new(
            TechniqueTag::MultiplyBothSides,
            format!("Cross-multiply by {} to clear denominator", denom_expr),
        )
        .with_output(cleared.clone()),
    );

    // Hand the cleared polynomial equation back to the general engine.
    let solution = unwrap_variable(&cleared, &Expr::int(0), var, trace)?;

    // Extraneous check: the cross-multiplication denominator, evaluated at
    // the candidate solution, must not be zero.
    let denom_at = substitute_symbol(&denom, var, &solution);
    if denom_at.is_zero() {
        return Err(SolverError::CannotSolve(format!(
            "Extraneous solution: {} = {} makes denominator {} equal to zero",
            var.as_str(),
            solution,
            denom_expr
        )));
    }

    Ok(solution)
}

// ── Distribution ───────────────────────────────────────────────────────────

/// Distribute products over sums so that `(A + B) · C` becomes `A·C + B·C`.
/// Leaves everything else structurally intact.
pub(super) fn expand(expr: &Arc<Expr>) -> Arc<Expr> {
    match expr.as_ref() {
        Expr::Mul(node) => {
            // Seed the working list with the numeric coefficient.
            let mut summands: Vec<Arc<Expr>> = vec![rational_to_arc(node.coeff.clone())];
            for (base, exp) in &node.factors {
                let base_e = expand(base);
                let factor = if exp.is_one() {
                    base_e
                } else {
                    // We do not distribute across non-unit exponents; treat
                    // `base^exp` as opaque when exp != 1.
                    normalize::pow(base_e, exp.clone())
                };
                summands = distribute_mul(&summands, &factor);
            }
            if summands.len() == 1 {
                summands.into_iter().next().unwrap()
            } else {
                normalize::add_many(summands)
            }
        }

        Expr::Add(node) => {
            let mut terms: Vec<Arc<Expr>> = Vec::new();
            if !node.constant.is_zero() {
                terms.push(rational_to_arc(node.constant.clone()));
            }
            for (term, coeff) in &node.terms {
                let et = expand(term);
                let scaled = if coeff.is_integer() && coeff.numer().to_i64() == Some(1) {
                    et
                } else {
                    normalize::mul(rational_to_arc(coeff.clone()), et)
                };
                terms.push(scaled);
            }
            if terms.is_empty() {
                Expr::int(0)
            } else {
                normalize::add_many(terms)
            }
        }

        Expr::Pow(base, exp) => normalize::pow(expand(base), exp.clone()),

        Expr::Func(fid, args) => {
            let eargs: Vec<Arc<Expr>> = args.iter().map(expand).collect();
            Expr::func(*fid, eargs)
        }

        _ => expr.clone(),
    }
}

fn distribute_mul(summands: &[Arc<Expr>], factor: &Arc<Expr>) -> Vec<Arc<Expr>> {
    match factor.as_ref() {
        Expr::Add(node) => {
            let mut result: Vec<Arc<Expr>> = Vec::new();
            for s in summands {
                if !node.constant.is_zero() {
                    result.push(normalize::mul(
                        s.clone(),
                        rational_to_arc(node.constant.clone()),
                    ));
                }
                for (term, coeff) in &node.terms {
                    let st = normalize::mul(s.clone(), term.clone());
                    let scaled = if coeff.is_integer() && coeff.numer().to_i64() == Some(1) {
                        st
                    } else {
                        normalize::mul(rational_to_arc(coeff.clone()), st)
                    };
                    result.push(scaled);
                }
            }
            result
        }
        _ => summands
            .iter()
            .map(|s| normalize::mul(s.clone(), factor.clone()))
            .collect(),
    }
}

// ── Substitution ───────────────────────────────────────────────────────────

/// Substitute every `Symbol(var)` in `expr` with `replacement`.
fn substitute_symbol(expr: &Arc<Expr>, var: SymbolId, replacement: &Arc<Expr>) -> Arc<Expr> {
    match expr.as_ref() {
        Expr::Symbol(s) if *s == var => replacement.clone(),
        Expr::Symbol(_)
        | Expr::Integer(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Complex(_)
        | Expr::Constant(_) => expr.clone(),

        Expr::Add(node) => {
            let mut terms: Vec<Arc<Expr>> = Vec::new();
            if !node.constant.is_zero() {
                terms.push(rational_to_arc(node.constant.clone()));
            }
            for (term, coeff) in &node.terms {
                let st = substitute_symbol(term, var, replacement);
                let scaled = if coeff.is_integer() && coeff.numer().to_i64() == Some(1) {
                    st
                } else {
                    normalize::mul(rational_to_arc(coeff.clone()), st)
                };
                terms.push(scaled);
            }
            if terms.is_empty() {
                Expr::int(0)
            } else {
                normalize::add_many(terms)
            }
        }

        Expr::Mul(node) => {
            let mut out = rational_to_arc(node.coeff.clone());
            for (base, exp) in &node.factors {
                let sb = substitute_symbol(base, var, replacement);
                let se = substitute_symbol(exp, var, replacement);
                out = normalize::mul(out, normalize::pow(sb, se));
            }
            out
        }

        Expr::Pow(base, exp) => {
            let sb = substitute_symbol(base, var, replacement);
            let se = substitute_symbol(exp, var, replacement);
            normalize::pow(sb, se)
        }

        Expr::Func(fid, args) => {
            let sargs: Vec<Arc<Expr>> = args
                .iter()
                .map(|a| substitute_symbol(a, var, replacement))
                .collect();
            Expr::func(*fid, sargs)
        }
    }
}
