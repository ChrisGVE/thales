//! Expr-native recursive unwrapping engine for symbolic isolation.
//!
//! Given `expr(var) = other` (both `Arc<Expr>`), peel off operations
//! wrapping `var` and apply inverses to `other` until `var` stands alone.
//! The Expr canonical form (flat `AddNode`, `MulNode`) makes most of the
//! traversal declarative instead of pattern-matching on binary operators.

use std::sync::Arc;

use num::traits::{One, Signed, Zero};

use crate::numeric::compile::decompile;
use crate::numeric::expr::FuncId;
use crate::numeric::trace::{Step, TechniqueTag, Trace};
use crate::numeric::{normalize, BigRational, Expr, MulNode, SymbolId};

use super::super::helpers::contains_symbol;
use super::super::types::SolverError;

use super::calculus::{is_calculus_wrapper, try_unwrap_calculus_wrapper};
use super::linear::collect_linear_var_terms;
use super::rational::try_cross_multiply_mul;

/// Convert a `BigRational` constant into an `Arc<Expr>` in canonical form.
pub(super) fn rational_to_arc(r: BigRational) -> Arc<Expr> {
    if r.is_integer() {
        Arc::new(Expr::Integer(r.numer().clone()))
    } else {
        Arc::new(Expr::Rational(r))
    }
}

/// Build an `Arc<Expr>` for a `MulNode` emulating `finish_mul_like`:
/// collapse empty/single-factor forms into simpler variants.
pub(super) fn finish_mul_like(node: MulNode) -> Arc<Expr> {
    if node.coeff.is_zero() {
        return Expr::int(0);
    }
    if node.factors.is_empty() {
        return rational_to_arc(node.coeff);
    }
    if node.coeff.is_one() && node.factors.len() == 1 {
        let (base, exp) = node.factors.into_iter().next().unwrap();
        if exp.is_one() {
            return base;
        }
        return Arc::new(Expr::Pow(base, exp));
    }
    Arc::new(Expr::Mul(node))
}

/// Main entry point for recursive isolation.
pub(super) fn unwrap_variable(
    expr: &Arc<Expr>,
    other: &Arc<Expr>,
    var: SymbolId,
    trace: &mut Trace,
) -> Result<Arc<Expr>, SolverError> {
    // Base case: the expression IS the variable.
    if let Expr::Symbol(s) = expr.as_ref() {
        if *s == var {
            return Ok(other.clone());
        }
    }

    match expr.as_ref() {
        Expr::Add(_) => unwrap_add(expr, other, var, trace),
        Expr::Mul(_) => unwrap_mul(expr, other, var, trace),
        Expr::Pow(base, exp) => unwrap_power(base, exp, other, var, trace),
        Expr::Func(fid, args) => unwrap_function(*fid, args, other, var, trace),
        _ => Err(SolverError::CannotSolve(format!(
            "Cannot isolate '{}': unsupported expression structure",
            var.as_str()
        ))),
    }
}

/// Unwrap `constant + Σ coeff_i·term_i = other`.
fn unwrap_add(
    expr: &Arc<Expr>,
    other: &Arc<Expr>,
    var: SymbolId,
    trace: &mut Trace,
) -> Result<Arc<Expr>, SolverError> {
    let node = match expr.as_ref() {
        Expr::Add(n) => n,
        _ => unreachable!(),
    };

    // Partition terms into those containing var and those without.
    let mut non_var: Vec<Arc<Expr>> = Vec::new();
    if !node.constant.is_zero() {
        non_var.push(rational_to_arc(node.constant.clone()));
    }
    let mut var_terms: Vec<(Arc<Expr>, BigRational)> = Vec::new();
    for (term, coeff) in &node.terms {
        if contains_symbol(term, var) {
            var_terms.push((term.clone(), coeff.clone()));
        } else {
            non_var.push(scale_term(term.clone(), coeff.clone()));
        }
    }

    if var_terms.is_empty() {
        return Err(SolverError::CannotSolve(format!(
            "Variable '{}' not found in sum",
            var.as_str()
        )));
    }

    let mut new_other = other.clone();

    // Move non-var part to the other side in one step.
    // Tag selection:
    // - AddBothSides   : const_part is a negative literal (subtracting a
    //                    negative = adding the absolute value)
    // - MoveTerm       : const_part came from a single non-var term
    // - SubtractBothSides : compound non-var sum, positive
    if !non_var.is_empty() {
        let non_var_count = non_var.len();
        let const_part = normalize::add_many(non_var);
        new_other = normalize::sub(new_other, const_part.clone());
        let is_negative_literal = match const_part.as_ref() {
            Expr::Integer(n) => n.is_negative(),
            Expr::Rational(r) => r.is_negative(),
            _ => false,
        };
        let (tag, detail) = if is_negative_literal {
            let abs_part = normalize::neg(const_part.clone());
            let abs_expr = decompile(&abs_part);
            (
                TechniqueTag::AddBothSides,
                format!("Add {} to both sides", abs_expr),
            )
        } else if non_var_count == 1 {
            let const_expr = decompile(&const_part);
            (
                TechniqueTag::MoveTerm,
                format!("Move {} to the other side", const_expr),
            )
        } else {
            let const_expr = decompile(&const_part);
            (
                TechniqueTag::SubtractBothSides,
                format!("Subtract {} from both sides", const_expr),
            )
        };
        trace.push(Step::new(tag, detail).with_output(new_other.clone()));
    }

    // Single var-containing term: divide by its rational coefficient (if any)
    // and recurse into the term.
    if var_terms.len() == 1 {
        let (term, coeff) = var_terms.into_iter().next().unwrap();
        if !is_one_rational(&coeff) {
            let coeff_arc = rational_to_arc(coeff);
            new_other = normalize::div(new_other, coeff_arc.clone());
            let coeff_expr = decompile(&coeff_arc);
            trace.push(
                Step::new(
                    TechniqueTag::DivideBothSides,
                    format!("Divide both sides by {}", coeff_expr),
                )
                .with_output(new_other.clone()),
            );
        }
        return unwrap_variable(&term, &new_other, var, trace);
    }

    // Multiple var-containing terms: try to factor var out linearly.
    collect_linear_var_terms(&var_terms, &new_other, var, trace)
}

/// Unwrap `coeff · Π base_i^exp_i = other`.
fn unwrap_mul(
    expr: &Arc<Expr>,
    other: &Arc<Expr>,
    var: SymbolId,
    trace: &mut Trace,
) -> Result<Arc<Expr>, SolverError> {
    let node = match expr.as_ref() {
        Expr::Mul(n) => n,
        _ => unreachable!(),
    };

    let mut var_factors: Vec<(Arc<Expr>, Arc<Expr>)> = Vec::new();
    let mut non_var_factors: Vec<(Arc<Expr>, Arc<Expr>)> = Vec::new();
    for (base, exp) in &node.factors {
        if contains_symbol(base, var) || contains_symbol(exp, var) {
            var_factors.push((base.clone(), exp.clone()));
        } else {
            non_var_factors.push((base.clone(), exp.clone()));
        }
    }

    let mut new_other = other.clone();

    // Divide numeric coefficient and non-var factors out of `other`.
    let denom = make_mul_from_parts(&node.coeff, &non_var_factors);
    if !denom.is_one() {
        new_other = normalize::div(new_other, denom.clone());
        let denom_expr = decompile(&denom);
        trace.push(
            Step::new(
                TechniqueTag::DivideBothSides,
                format!("Divide both sides by {}", denom_expr),
            )
            .with_output(new_other.clone()),
        );
    }

    if var_factors.is_empty() {
        return Err(SolverError::CannotSolve(format!(
            "Variable '{}' not found in product",
            var.as_str()
        )));
    }

    // Multiple var factors → cross-multiply to clear denominators.
    if var_factors.len() > 1 {
        return try_cross_multiply_mul(&var_factors, &new_other, var, trace);
    }

    // Exactly one var factor.
    let (base, exp) = var_factors.into_iter().next().unwrap();
    if exp.is_one() {
        return unwrap_variable(&base, &new_other, var, trace);
    }
    unwrap_power(&base, &exp, &new_other, var, trace)
}

/// Unwrap `base^exp = other`.
pub(super) fn unwrap_power(
    base: &Arc<Expr>,
    exp: &Arc<Expr>,
    other: &Arc<Expr>,
    var: SymbolId,
    trace: &mut Trace,
) -> Result<Arc<Expr>, SolverError> {
    let base_has = contains_symbol(base, var);
    let exp_has = contains_symbol(exp, var);
    if base_has && exp_has {
        return Err(SolverError::CannotSolve(format!(
            "Cannot isolate '{}': variable in both base and exponent",
            var.as_str()
        )));
    }
    if base_has {
        // Special case `base^(-1) = other` → `base = 1/other`. This is
        // the reciprocal of both sides, which is an elementary
        // manipulation, not a root extraction.
        if matches!(exp.as_ref(), Expr::Integer(n) if n.to_i64() == Some(-1)) {
            let new_other = normalize::div(Expr::int(1), other.clone());
            trace.push(
                Step::new(
                    TechniqueTag::DivideBothSides,
                    "reciprocal; Take the reciprocal of both sides".to_string(),
                )
                .with_output(new_other.clone()),
            );
            return unwrap_variable(base, &new_other, var, trace);
        }

        // base^exp = other → base = other^(1/exp)
        let inv_exp = normalize::div(Expr::int(1), exp.clone());
        let new_other = normalize::pow(other.clone(), inv_exp);
        let exp_expr = decompile(exp);
        trace.push(
            Step::new(
                TechniqueTag::RootBothSides,
                format!("Take the {} root of both sides", exp_expr),
            )
            .with_output(new_other.clone()),
        );
        return unwrap_variable(base, &new_other, var, trace);
    }
    // a^exp(v) = other → exp = log_base(other) = ln(other)/ln(base)
    let numer = Expr::func(FuncId::Ln, vec![other.clone()]);
    let denom = Expr::func(FuncId::Ln, vec![base.clone()]);
    let new_other = normalize::div(numer, denom);
    let base_expr = decompile(base);
    trace.push(
        Step::new(
            TechniqueTag::ApplyFunction,
            format!("log; Take logarithm base {} of both sides", base_expr),
        )
        .with_output(new_other.clone()),
    );
    unwrap_variable(exp, &new_other, var, trace)
}

/// Unwrap a function application `f(arg) = other` by applying its inverse.
fn unwrap_function(
    fid: FuncId,
    args: &[Arc<Expr>],
    other: &Arc<Expr>,
    var: SymbolId,
    trace: &mut Trace,
) -> Result<Arc<Expr>, SolverError> {
    if is_calculus_wrapper(fid) {
        return try_unwrap_calculus_wrapper(fid, args, other, var, trace);
    }

    if args.len() > 1 {
        let var_indices: Vec<usize> = args
            .iter()
            .enumerate()
            .filter(|(_, a)| contains_symbol(a, var))
            .map(|(i, _)| i)
            .collect();
        if var_indices.len() == 1 {
            let inner = &args[var_indices[0]];
            if matches!(inner.as_ref(), Expr::Symbol(s) if *s == var) {
                return Err(SolverError::CannotSolve(format!(
                    "Cannot isolate '{}': function {} is not invertible",
                    var.as_str(),
                    fid
                )));
            }
        }
        return Err(SolverError::CannotSolve(format!(
            "Cannot isolate '{}': multi-argument function {}",
            var.as_str(),
            fid
        )));
    }

    let inner = &args[0];
    if !contains_symbol(inner, var) {
        return Err(SolverError::CannotSolve(format!(
            "Variable '{}' not found in function argument",
            var.as_str()
        )));
    }

    let (new_other, desc) = match fid {
        FuncId::Sin => (
            Expr::func(FuncId::Asin, vec![other.clone()]),
            "Inverse Trigonometric Function: arcsin; Apply arcsine to both sides",
        ),
        FuncId::Cos => (
            Expr::func(FuncId::Acos, vec![other.clone()]),
            "Inverse Trigonometric Function: arccos; Apply arccos to both sides",
        ),
        FuncId::Tan => (
            Expr::func(FuncId::Atan, vec![other.clone()]),
            "Inverse Trigonometric Function: arctan; Apply arctan to both sides",
        ),
        FuncId::Asin => (
            Expr::func(FuncId::Sin, vec![other.clone()]),
            "Inverse Trigonometric Function: sin; Apply sin to both sides",
        ),
        FuncId::Acos => (
            Expr::func(FuncId::Cos, vec![other.clone()]),
            "Inverse Trigonometric Function: cos; Apply cos to both sides",
        ),
        FuncId::Atan => (
            Expr::func(FuncId::Tan, vec![other.clone()]),
            "Inverse Trigonometric Function: tan; Apply tan to both sides",
        ),
        FuncId::Exp => (
            Expr::func(FuncId::Ln, vec![other.clone()]),
            "ln; Take natural log of both sides",
        ),
        FuncId::Ln => (
            Expr::func(FuncId::Exp, vec![other.clone()]),
            "exp; Exponentiate both sides",
        ),
        FuncId::Sqrt => (
            normalize::pow(other.clone(), Expr::int(2)),
            "Square both sides",
        ),
        FuncId::Cbrt => (
            normalize::pow(other.clone(), Expr::int(3)),
            "Cube both sides",
        ),
        _ => {
            return Err(SolverError::CannotSolve(format!(
                "Cannot isolate '{}': function {} is not invertible",
                var.as_str(),
                fid
            )));
        }
    };

    let tag = match fid {
        FuncId::Sqrt | FuncId::Cbrt => TechniqueTag::RootBothSides,
        // Applying ln to undo exp is a logarithm-identity step (PowerAndRoots
        // tier); applying exp to undo ln is an exponential-identity step
        // (also PowerAndRoots). The generic `ApplyFunction` tag sits at the
        // AlgebraicManip tier, which would over-classify these elementary
        // unwraps.
        FuncId::Exp => TechniqueTag::LogIdentity,
        FuncId::Ln => TechniqueTag::ExpIdentity,
        _ => TechniqueTag::ApplyFunction,
    };
    trace.push(Step::new(tag, desc.to_string()).with_output(new_other.clone()));
    unwrap_variable(inner, &new_other, var, trace)
}

// ── helpers ────────────────────────────────────────────────────────────────

fn is_one_rational(r: &BigRational) -> bool {
    r.is_integer() && r.numer().to_i64() == Some(1)
}

fn scale_term(term: Arc<Expr>, coeff: BigRational) -> Arc<Expr> {
    if is_one_rational(&coeff) {
        term
    } else {
        normalize::mul(rational_to_arc(coeff), term)
    }
}

/// Build an `Arc<Expr>` for `coeff · Π base_i^exp_i` without going through
/// `normalize::mul` (which does not merge pre-existing `Pow` factors into a
/// shared `MulNode`).
fn make_mul_from_parts(coeff: &BigRational, parts: &[(Arc<Expr>, Arc<Expr>)]) -> Arc<Expr> {
    let mut node = MulNode::from_coeff(coeff.clone());
    for (base, exp) in parts {
        node.add_factor(base.clone(), exp.clone());
    }
    finish_mul_like(node)
}
