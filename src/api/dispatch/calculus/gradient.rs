//! Gradient and total-diff command dispatchers.

use crate::api::response::{EngineId, Response, ResultEntry, ResultKey, ResultShape, ResultValue};
use crate::ast::{BinaryOp, Expression};
use crate::numeric::compile::compile;
use crate::numeric::trace::{Step, TechniqueTag, Trace};

use super::super::helpers::{steps_from_trace, symbolic_entry};

pub(in crate::api::dispatch) fn gradient_cmd(
    expr: &Expression,
    vars: &[String],
    narrate: bool,
) -> Response {
    let mut components = Vec::with_capacity(vars.len());
    let mut trace = Trace::new();
    for var in vars {
        let d = expr.differentiate(var).simplify();
        if narrate {
            trace.push(
                Step::new(
                    TechniqueTag::PowerRule,
                    format!("Gradient component ∂/∂{}", var),
                )
                .with_output(compile(&d)),
            );
        }
        components.push(d);
    }
    let (primary, alternatives) = match components.split_first() {
        Some((first, rest)) => (first.clone(), rest.to_vec()),
        None => (Expression::Integer(0), Vec::new()),
    };
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        ResultEntry {
            value: ResultValue::Symbolic(primary),
            structured: None,
            shape: ResultShape::Vector,
            unit: None,
            steps: if narrate {
                steps_from_trace(&trace)
            } else {
                Vec::new()
            },
            alternatives,
            engine: EngineId::Differentiation,
        },
    ));
    r.meta.engine_trace.push(EngineId::Differentiation);
    r
}

/// Total derivative `df/dvar = ∂f/∂var + Σᵢ (∂f/∂nameᵢ)·(d nameᵢ/d var)`
/// where `d nameᵢ/d var = differentiate(depᵢ, var)`.
pub(in crate::api::dispatch) fn total_diff_cmd(
    expr: &Expression,
    var: &str,
    deps: &[(String, Expression)],
    narrate: bool,
) -> Response {
    let mut trace = Trace::new();

    // ∂f/∂var — direct partial.
    let direct = expr.differentiate(var).simplify();
    if narrate {
        trace.push(
            Step::new(
                TechniqueTag::TotalDifferential,
                format!("Direct partial ∂/∂{}", var),
            )
            .with_output(compile(&direct)),
        );
    }

    // Chain terms: Σᵢ (∂f/∂nameᵢ) · (d nameᵢ/d var)
    let mut total = direct;
    for (name, dep_expr) in deps {
        let df_dname = expr.differentiate(name).simplify();
        let dname_dvar = dep_expr.differentiate(var).simplify();
        let chain_term = Expression::Binary(
            BinaryOp::Mul,
            Box::new(df_dname.clone()),
            Box::new(dname_dvar.clone()),
        )
        .simplify();
        if narrate {
            trace.push(
                Step::new(
                    TechniqueTag::TotalDifferential,
                    format!("Chain term (∂f/∂{}) · (d {}/d {})", name, name, var),
                )
                .with_output(compile(&chain_term)),
            );
        }
        total = Expression::Binary(BinaryOp::Add, Box::new(total), Box::new(chain_term)).simplify();
    }

    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(total, EngineId::Differentiation, steps_from_trace(&trace)),
    ));
    r.meta.engine_trace.push(EngineId::Differentiation);
    r
}
