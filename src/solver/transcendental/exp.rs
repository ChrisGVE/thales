//! Exponential equation patterns.
//!
//! Operates on pre-compiled `Arc<Expr>` sides.
//!
//! - `exp(x) = a` → `x = ln(a)`
//! - `exp(c·x) = a` → `x = ln(a) / c`
//! - `c · exp(x) = a` → `x = ln(a / c)`
//! - `a^x = b` → `x = ln(b) / ln(a)` (change of base)
//! - `a^(c·x) = b` → `x = ln(b) / (c · ln(a))`

use std::sync::Arc;

use crate::ast::Variable;
use crate::numeric::trace::{Step, TechniqueTag, Trace};
use crate::numeric::{normalize, BigRational, Expr, FuncId, MulNode, SymbolId};

use super::super::coeff::extract_coefficient;
use super::super::helpers::contains_symbol;

pub(super) fn solve_exp_equation(
    lhs: &Arc<Expr>,
    rhs: &Arc<Expr>,
    var: SymbolId,
    variable: &Variable,
    trace: &mut Trace,
) -> Result<Arc<Expr>, ()> {
    // exp(...) = ...
    for (left, right) in [(lhs, rhs), (rhs, lhs)] {
        if let Some(result) = match_exp(left, right, var) {
            append_step_exp(trace, &result, variable);
            return Ok(result);
        }
    }
    // a^x = ...
    for (left, right) in [(lhs, rhs), (rhs, lhs)] {
        if let Some(result) = match_power(left, right, var) {
            append_step_power(trace, &result, variable);
            return Ok(result);
        }
    }
    Err(())
}

fn match_exp(left: &Arc<Expr>, right: &Arc<Expr>, var: SymbolId) -> Option<Arc<Expr>> {
    if contains_symbol(right, var) {
        return None;
    }
    // exp(arg) = right.
    if let Expr::Func(FuncId::Exp, args) = left.as_ref() {
        if args.len() == 1 {
            // exp(var) = right → var = ln(right).
            if matches!(args[0].as_ref(), Expr::Symbol(s) if *s == var) {
                return Some(Expr::func(FuncId::Ln, vec![Arc::clone(right)]));
            }
            // exp(c·var) = right → var = ln(right) / c.
            if let Ok(coeff) = extract_coefficient(&args[0], var) {
                let ln_applied = Expr::func(FuncId::Ln, vec![Arc::clone(right)]);
                return Some(normalize::div(ln_applied, Arc::new(Expr::Rational(coeff))));
            }
        }
    }
    // c·exp(var) = right → var = ln(right/c).
    if let Expr::Mul(node) = left.as_ref() {
        if let Some(coeff) = split_mul_of_func_of_var(node, FuncId::Exp, var) {
            let divided = normalize::div(Arc::clone(right), Arc::new(Expr::Rational(coeff)));
            return Some(Expr::func(FuncId::Ln, vec![divided]));
        }
    }
    None
}

fn match_power(left: &Arc<Expr>, right: &Arc<Expr>, var: SymbolId) -> Option<Arc<Expr>> {
    if contains_symbol(right, var) {
        return None;
    }
    // a^x = b with base variable-free and var in exponent.
    if let Expr::Pow(base, exp) = left.as_ref() {
        if contains_symbol(base, var) || !contains_symbol(exp, var) {
            return None;
        }
        let ln_right = Expr::func(FuncId::Ln, vec![Arc::clone(right)]);
        let ln_base = Expr::func(FuncId::Ln, vec![Arc::clone(base)]);
        // exp is var → var = ln(right)/ln(base).
        if matches!(exp.as_ref(), Expr::Symbol(s) if *s == var) {
            return Some(normalize::div(ln_right, ln_base));
        }
        // exp is c·var → var = ln(right) / (c · ln(base)).
        let coeff = extract_coefficient(exp, var).ok()?;
        let divided = normalize::div(ln_right, ln_base);
        Some(normalize::div(divided, Arc::new(Expr::Rational(coeff))))
    } else {
        None
    }
}

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
            if *f == func
                && inner_args.len() == 1
                && matches!(inner_args[0].as_ref(), Expr::Symbol(s) if *s == var)
            {
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

fn append_step_exp(trace: &mut Trace, solution: &Arc<Expr>, variable: &Variable) {
    trace.push(
        Step::new(
            TechniqueTag::ApplyFunction,
            format!(
                "ln; Apply natural logarithm to solve exp({}) = value",
                variable
            ),
        )
        .with_output(Arc::clone(solution)),
    );
}

fn append_step_power(trace: &mut Trace, solution: &Arc<Expr>, variable: &Variable) {
    trace.push(
        Step::new(
            TechniqueTag::LogIdentity,
            format!(
                "change of base; Apply logarithm to solve for {} in exponent",
                variable
            ),
        )
        .with_output(Arc::clone(solution)),
    );
}
