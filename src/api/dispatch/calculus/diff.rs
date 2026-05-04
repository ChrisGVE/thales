//! Differentiation command dispatchers: diff and partial_diff.

use std::sync::Arc;

use crate::api::response::{EngineId, Response, ResultKey};
use crate::ast::Expression;
use crate::numeric::compile::{compile, decompile};
use crate::numeric::differentiation::diff_arc;
use crate::numeric::expr::Expr;
use crate::numeric::trace::{Step, TechniqueTag, Trace};
use crate::numeric::SymbolId;
use crate::solver::helpers::detection::contains_symbol;

use super::super::helpers::{steps_from_trace, symbolic_entry};

// ── Technique classification helpers ─────────────────────────────────────────

fn classify_diff_tag(expr: &Arc<Expr>, var: SymbolId) -> TechniqueTag {
    match expr.as_ref() {
        // Product with 2+ var-dependent factors → ProductRule
        Expr::Mul(node)
            if node
                .factors
                .keys()
                .filter(|b| contains_symbol(b, var))
                .count()
                > 1 =>
        {
            TechniqueTag::ProductRule
        }

        // Power with constant exponent, var-dependent base → PowerRule
        Expr::Pow(base, exp) if !contains_symbol(exp, var) && contains_symbol(base, var) => {
            TechniqueTag::PowerRule
        }

        // Named function with var-dependent args → ChainRule
        Expr::Func(_, args) if args.iter().any(|a| contains_symbol(a, var)) => {
            TechniqueTag::ChainRule
        }

        // Power where exponent depends on var → ChainRule (general exponential)
        Expr::Pow(_, exp) if contains_symbol(exp, var) => TechniqueTag::ChainRule,

        // Symbol equal to var → PowerRule (degenerate d/dx(x) = 1)
        Expr::Symbol(s) if *s == var => TechniqueTag::PowerRule,

        // Default: constants, sums, etc. → Simplification
        _ => TechniqueTag::Simplification,
    }
}

fn build_diff_detail(expr: &Arc<Expr>, tag: TechniqueTag, var: SymbolId) -> String {
    match tag {
        TechniqueTag::ProductRule => {
            if let Expr::Mul(node) = expr.as_ref() {
                let deps: Vec<&Arc<Expr>> = node
                    .factors
                    .keys()
                    .filter(|b| contains_symbol(b, var))
                    .take(2)
                    .collect();
                if deps.len() == 2 {
                    return format!("f={};g={}", decompile(deps[0]), decompile(deps[1]));
                }
            }
            format!("var={}", var.as_str())
        }
        TechniqueTag::ChainRule => {
            if let Expr::Func(id, args) = expr.as_ref() {
                if !args.is_empty() {
                    return format!("outer={};inner={}", id, decompile(&args[0]));
                }
            }
            if let Expr::Pow(base, _) = expr.as_ref() {
                return format!("outer=pow;inner={}", decompile(base));
            }
            format!("var={}", var.as_str())
        }
        TechniqueTag::PowerRule => format!("var={}", var.as_str()),
        _ => String::new(),
    }
}

fn diff_traced(expr: &Arc<Expr>, var: SymbolId, trace: &mut Trace) -> Arc<Expr> {
    let result = diff_arc(expr, var);
    let tag = classify_diff_tag(expr, var);
    let detail = build_diff_detail(expr, tag, var);
    trace.push(
        Step::new(tag, detail)
            .with_input(expr.clone())
            .with_output(result.clone()),
    );
    result
}

// ── Command dispatchers ───────────────────────────────────────────────────────

pub(in crate::api::dispatch) fn diff_cmd(
    expr: &Expression,
    var: &str,
    order: u32,
    narrate: bool,
) -> Response {
    let var_id = SymbolId::intern(var);
    let mut current = compile(expr);
    let mut trace = Trace::new();
    for _ in 0..order {
        current = if narrate {
            diff_traced(&current, var_id, &mut trace)
        } else {
            diff_arc(&current, var_id)
        };
    }
    let result_expr = decompile(&current);
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(
            result_expr,
            EngineId::Differentiation,
            steps_from_trace(&trace),
        ),
    ));
    r.meta.engine_trace.push(EngineId::Differentiation);
    r
}

pub(in crate::api::dispatch) fn partial_diff_cmd(
    expr: &Expression,
    vars: &[(String, u32)],
    narrate: bool,
) -> Response {
    let mut current = compile(expr);
    let mut trace = Trace::new();
    for (var, order) in vars {
        let var_id = SymbolId::intern(var);
        for _ in 0..*order {
            current = if narrate {
                diff_traced(&current, var_id, &mut trace)
            } else {
                diff_arc(&current, var_id)
            };
        }
    }
    let result_expr = decompile(&current);
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(
            result_expr,
            EngineId::Differentiation,
            steps_from_trace(&trace),
        ),
    ));
    r.meta.engine_trace.push(EngineId::Differentiation);
    r
}
