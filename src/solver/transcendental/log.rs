//! Logarithmic equation patterns.
//!
//! Operates on pre-compiled `Arc<Expr>` sides.
//!
//! - `ln(x) = a` → `x = exp(a)`
//! - `c · ln(x) = a` → `x = exp(a/c)`
//! - `log10(x) = a` → `x = 10^a`
//! - `log(x, b) = a` → `x = b^a`

use std::sync::Arc;

use crate::ast::Variable;
use crate::numeric::trace::{Step, TechniqueTag, Trace};
use crate::numeric::{normalize, BigRational, Expr, FuncId, MulNode, SymbolId};

use super::super::helpers::contains_symbol;

pub(super) fn solve_log_equation(
    lhs: &Arc<Expr>,
    rhs: &Arc<Expr>,
    var: SymbolId,
    variable: &Variable,
    trace: &mut Trace,
) -> Result<Arc<Expr>, ()> {
    // ln / natural log.
    for (left, right) in [(lhs, rhs), (rhs, lhs)] {
        if let Some(result) = match_ln(left, right, var) {
            append_step_ln(trace, &result, variable);
            return Ok(result);
        }
    }
    // log10.
    for (left, right) in [(lhs, rhs), (rhs, lhs)] {
        if let Some(result) = match_log10(left, right, var) {
            append_step_log10(trace, &result, variable);
            return Ok(result);
        }
    }
    // log(var, base) = value.
    for (left, right) in [(lhs, rhs), (rhs, lhs)] {
        if let Some(result) = match_log_base(left, right, var) {
            append_step_log_base(trace, &result, variable);
            return Ok(result);
        }
    }
    Err(())
}

fn match_ln(left: &Arc<Expr>, right: &Arc<Expr>, var: SymbolId) -> Option<Arc<Expr>> {
    if contains_symbol(right, var) {
        return None;
    }
    // ln(var) = right → var = exp(right).
    if let Expr::Func(FuncId::Ln, args) = left.as_ref() {
        if args.len() == 1 && is_target_var(&args[0], var) {
            return Some(Expr::func(FuncId::Exp, vec![Arc::clone(right)]));
        }
    }
    // c·ln(var) = right → var = exp(right / c).
    if let Expr::Mul(node) = left.as_ref() {
        if let Some(coeff) = split_mul_of_func_of_var(node, FuncId::Ln, var) {
            let divided = normalize::div(Arc::clone(right), Arc::new(Expr::Rational(coeff)));
            return Some(Expr::func(FuncId::Exp, vec![divided]));
        }
    }
    None
}

fn match_log10(left: &Arc<Expr>, right: &Arc<Expr>, var: SymbolId) -> Option<Arc<Expr>> {
    if contains_symbol(right, var) {
        return None;
    }
    if let Expr::Func(FuncId::Log10, args) = left.as_ref() {
        if args.len() == 1 && is_target_var(&args[0], var) {
            return Some(normalize::pow(Expr::int(10), Arc::clone(right)));
        }
    }
    None
}

fn match_log_base(left: &Arc<Expr>, right: &Arc<Expr>, var: SymbolId) -> Option<Arc<Expr>> {
    if contains_symbol(right, var) {
        return None;
    }
    // log(var, base) with base variable-free → var = base^right.
    if let Expr::Func(FuncId::Log, args) = left.as_ref() {
        if args.len() == 2 && is_target_var(&args[0], var) && !contains_symbol(&args[1], var) {
            return Some(normalize::pow(Arc::clone(&args[1]), Arc::clone(right)));
        }
    }
    None
}

fn is_target_var(expr: &Arc<Expr>, var: SymbolId) -> bool {
    matches!(expr.as_ref(), Expr::Symbol(s) if *s == var)
}

/// Decompose a `MulNode` that represents `c · func(var)`. Returns `c`
/// when exactly one factor is `Func(func, [Symbol(var)])^1` and all
/// other factors are variable-free.
fn split_mul_of_func_of_var(node: &MulNode, func: FuncId, var: SymbolId) -> Option<BigRational> {
    let mut matched = false;
    for (base, exp) in &node.factors {
        let base_has = contains_symbol(base, var);
        let exp_has = contains_symbol(exp, var);
        if !base_has && !exp_has {
            continue;
        }
        if exp_has || !matches!(exp.as_ref(), Expr::Integer(n) if n.to_i64() == Some(1)) {
            return None;
        }
        if let Expr::Func(f, inner_args) = base.as_ref() {
            if *f == func && inner_args.len() == 1 && is_target_var(&inner_args[0], var) {
                if matched {
                    return None;
                }
                matched = true;
                continue;
            }
        }
        return None;
    }
    if matched {
        Some(node.coeff.clone())
    } else {
        None
    }
}

fn append_step_ln(trace: &mut Trace, solution: &Arc<Expr>, variable: &Variable) {
    trace.push(
        Step::new(
            TechniqueTag::ApplyFunction,
            format!("exp; Apply exponential to solve ln({}) = value", variable),
        )
        .with_output(Arc::clone(solution)),
    );
}

fn append_step_log10(trace: &mut Trace, solution: &Arc<Expr>, variable: &Variable) {
    trace.push(
        Step::new(
            TechniqueTag::PowerBothSides,
            format!("10; Apply 10^x to solve log10({}) = value", variable),
        )
        .with_output(Arc::clone(solution)),
    );
}

fn append_step_log_base(trace: &mut Trace, solution: &Arc<Expr>, variable: &Variable) {
    trace.push(
        Step::new(
            TechniqueTag::LogIdentity,
            format!(
                "exponential form; Convert logarithm to exponential form to solve for {}",
                variable
            ),
        )
        .with_output(Arc::clone(solution)),
    );
}
