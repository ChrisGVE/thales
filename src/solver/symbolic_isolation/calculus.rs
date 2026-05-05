//! Calculus-wrapper handling in `Expr`.
//!
//! Handles factoring the target variable out of a calculus wrapper's body:
//! `integral(F · cos(theta), x)` solving for `F` becomes
//! `F · integral(cos(theta), x)`, which the standard engine then isolates.

use std::sync::Arc;

use crate::numeric::expr::FuncId;
use crate::numeric::trace::{Step, TechniqueTag, Trace};
use crate::numeric::{normalize, Expr, SymbolId};

use super::super::helpers::contains_symbol;
use super::super::types::SolverError;
use super::unwrap::unwrap_variable;

/// Recognise `integral`, `sum`, `product`, `limit`, and `derivative` wrappers.
pub(super) fn is_calculus_wrapper(fid: FuncId) -> bool {
    if let FuncId::Other(sym) = fid {
        matches!(
            sym.as_str().as_str(),
            "integral" | "sum" | "product" | "limit" | "derivative"
        )
    } else {
        false
    }
}

/// Try to factor the target variable out of a calculus wrapper's body.
pub(super) fn try_unwrap_calculus_wrapper(
    fid: FuncId,
    args: &[Arc<Expr>],
    other: &Arc<Expr>,
    var: SymbolId,
    trace: &mut Trace,
) -> Result<Arc<Expr>, SolverError> {
    let func_name = match fid {
        FuncId::Other(sym) => sym.as_str(),
        _ => return Err(SolverError::CannotSolve("Not a calculus wrapper".into())),
    };

    if args.is_empty() {
        return Err(SolverError::CannotSolve(format!(
            "{} wrapper has no body",
            func_name
        )));
    }

    let body = &args[0];
    if !contains_symbol(body, var) {
        return Err(SolverError::CannotSolve(format!(
            "Variable '{}' not found in {} body",
            var.as_str(),
            func_name
        )));
    }

    // For integral/sum/product: the second argument is the bound variable.
    // The target must not be that.
    if args.len() >= 2 {
        if let Expr::Symbol(s) = args[1].as_ref() {
            if *s == var {
                return Err(SolverError::CannotSolve(format!(
                    "Cannot isolate '{}': it is the {} variable",
                    var.as_str(),
                    func_name
                )));
            }
        }
    }

    // Try to split the body into (var_factor, rest).
    if let Some((var_factor, rest_body)) = try_split_product(body, var) {
        let mut new_args = args.to_vec();
        new_args[0] = rest_body;
        let wrapper_expr = Expr::func(fid, new_args);
        let factored = normalize::mul(var_factor, wrapper_expr);

        trace.push(
            Step::new(
                TechniqueTag::ApplyFunction,
                format!(
                    "Calculus Wrapper Isolation: factor_from_{}; Factor '{}' out of {} body",
                    func_name,
                    var.as_str(),
                    func_name
                ),
            )
            .with_output(other.clone()),
        );
        return unwrap_variable(&factored, other, var, trace);
    }

    // Body is just the variable: wrapper becomes wrapper(1) · var.
    if let Expr::Symbol(s) = body.as_ref() {
        if *s == var {
            let mut new_args = args.to_vec();
            new_args[0] = Expr::int(1);
            let wrapper_one = Expr::func(fid, new_args);
            let new_other = normalize::div(other.clone(), wrapper_one);
            let name = var.as_str();
            trace.push(
                Step::new(
                    TechniqueTag::DivideBothSides,
                    format!(
                        "Calculus Wrapper Isolation: {}; Isolate '{}' from {} body",
                        name, name, func_name
                    ),
                )
                .with_output(new_other.clone()),
            );
            return Ok(new_other);
        }
    }

    Err(SolverError::CannotSolve(format!(
        "Cannot factor '{}' out of {} body",
        var.as_str(),
        func_name
    )))
}

/// Split `expr` into `(var_factor, rest)` such that `expr = var_factor · rest`
/// and the `var_factor` carries all `var`-dependence. Returns `None` when
/// `var` appears in more than one factor or inside an uninvertible shape.
pub(super) fn try_split_product(expr: &Arc<Expr>, var: SymbolId) -> Option<(Arc<Expr>, Arc<Expr>)> {
    if !contains_symbol(expr, var) {
        return None;
    }

    // expr IS var
    if let Expr::Symbol(s) = expr.as_ref() {
        if *s == var {
            return Some((expr.clone(), Expr::int(1)));
        }
    }

    // Power whose base is a bare var symbol: treat entire power as the factor.
    if let Expr::Pow(base, exp) = expr.as_ref() {
        if let Expr::Symbol(s) = base.as_ref() {
            if *s == var && !contains_symbol(exp, var) {
                return Some((expr.clone(), Expr::int(1)));
            }
        }
    }

    // Function application containing var: treat whole call as the factor.
    if matches!(expr.as_ref(), Expr::Func(_, _)) {
        return Some((expr.clone(), Expr::int(1)));
    }

    // Mul: collect all var-containing factors vs non-var factors.
    if let Expr::Mul(node) = expr.as_ref() {
        let mut var_factor = Expr::int(1);
        let mut rest = super::unwrap::rational_to_arc(node.coeff.clone());
        for (base, exp) in &node.factors {
            if contains_symbol(base, var) || contains_symbol(exp, var) {
                var_factor = normalize::mul(var_factor, normalize::pow(base.clone(), exp.clone()));
            } else {
                rest = normalize::mul(rest, normalize::pow(base.clone(), exp.clone()));
            }
        }
        if var_factor.is_one() {
            return None;
        }
        return Some((var_factor, rest));
    }

    None
}
