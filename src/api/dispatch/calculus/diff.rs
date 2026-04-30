//! Differentiation command dispatchers: diff and partial_diff.

use crate::api::response::{EngineId, Response, ResultKey};
use crate::ast::Expression;
use crate::numeric::compile::compile;
use crate::numeric::trace::{Step, TechniqueTag, Trace};

use super::super::helpers::{steps_from_trace, symbolic_entry};

pub(in crate::api::dispatch) fn diff_cmd(
    expr: &Expression,
    var: &str,
    order: u32,
    narrate: bool,
) -> Response {
    let mut current = expr.clone();
    let mut trace = Trace::new();
    for _ in 0..order {
        let next = current.differentiate(var);
        if narrate {
            trace.push(
                Step::new(
                    TechniqueTag::PowerRule,
                    format!("Differentiate with respect to {}", var),
                )
                .with_output(compile(&next)),
            );
        }
        current = next;
    }
    let simplified = current.simplify();
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(
            simplified,
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
    let mut current = expr.clone();
    let mut trace = Trace::new();
    for (var, order) in vars {
        for _ in 0..*order {
            current = current.differentiate(var);
            if narrate {
                trace.push(
                    Step::new(
                        TechniqueTag::PowerRule,
                        format!("Partial derivative ∂/∂{}", var),
                    )
                    .with_output(compile(&current)),
                );
            }
        }
    }
    let simplified = current.simplify();
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(
            simplified,
            EngineId::Differentiation,
            steps_from_trace(&trace),
        ),
    ));
    r.meta.engine_trace.push(EngineId::Differentiation);
    r
}
