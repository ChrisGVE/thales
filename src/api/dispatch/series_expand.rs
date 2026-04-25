//! F1c series-expansion dispatchers (Taylor, Laurent, Asymptotic, Compose,
//! Revert) wired to `crate::numeric::series` engines.

use crate::api::response::{EngineId, Response, ResultKey};
use crate::ast::Expression;
use crate::numeric::compile::{compile, decompile};
use crate::numeric::series::{
    asymptotic, compose, laurent_expand, revert, taylor, AsymptoticDirection,
};
use crate::numeric::trace::{Step, TechniqueTag, Trace};
use crate::numeric::SymbolId;

use super::helpers::{engine_error, steps_from_trace, symbolic_entry};

pub(super) fn taylor_cmd(
    expr: &Expression,
    var: &str,
    center: &Expression,
    order: u32,
    narrate: bool,
) -> Response {
    let expr_arc = compile(expr);
    let center_arc = compile(center);
    let var_id = SymbolId::intern(var);
    let mut trace = Trace::new();
    if narrate {
        trace.push(
            Step::new(
                TechniqueTag::TaylorExpansion,
                format!(
                    "Taylor series in {} around {} to order {}",
                    var, center, order
                ),
            )
            .with_input(expr_arc.clone()),
        );
    }
    let series = taylor(&expr_arc, var_id, &center_arc, order as usize);
    let result_arc = series.to_expr();
    let result = decompile(&result_arc);
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(result, EngineId::TaylorExpansion, steps_from_trace(&trace)),
    ));
    r.meta.engine_trace.push(EngineId::TaylorExpansion);
    r
}

pub(super) fn laurent_cmd(
    expr: &Expression,
    var: &str,
    center: &Expression,
    order: u32,
    narrate: bool,
) -> Response {
    let expr_arc = compile(expr);
    let center_arc = compile(center);
    let var_id = SymbolId::intern(var);
    let mut trace = Trace::new();
    let span = order;
    let trace_handle = if narrate { Some(&mut trace) } else { None };
    let series = laurent_expand(&expr_arc, var_id, &center_arc, span, span, trace_handle);
    match series {
        Some(s) => {
            let result_arc = s.to_expr();
            let result = decompile(&result_arc);
            let mut r = Response::default();
            r.results.push((
                ResultKey::Single,
                symbolic_entry(result, EngineId::LaurentExpansion, steps_from_trace(&trace)),
            ));
            r.meta.engine_trace.push(EngineId::LaurentExpansion);
            r
        }
        None => engine_error(
            "command.laurent",
            "Laurent expansion failed: shift exceeds maximum or residual not analytic".to_string(),
        ),
    }
}

pub(super) fn asymptotic_cmd(expr: &Expression, var: &str, order: u32, narrate: bool) -> Response {
    let expr_arc = compile(expr);
    let var_id = SymbolId::intern(var);
    let mut trace = Trace::new();
    let direction = AsymptoticDirection::PosInfinity;
    let trace_handle = if narrate { Some(&mut trace) } else { None };
    let series = asymptotic(&expr_arc, var_id, direction, order as usize, trace_handle);
    match series {
        Some(s) => {
            let result_arc = s.to_expr();
            let result = decompile(&result_arc);
            let mut r = Response::default();
            r.results.push((
                ResultKey::Single,
                symbolic_entry(
                    result,
                    EngineId::AsymptoticExpansion,
                    steps_from_trace(&trace),
                ),
            ));
            r.meta.engine_trace.push(EngineId::AsymptoticExpansion);
            r
        }
        None => engine_error(
            "command.asymptotic",
            format!(
                "asymptotic expansion failed: {} not reducible to a Laurent polynomial in {}",
                expr, var
            ),
        ),
    }
}

pub(super) fn compose_cmd(
    outer: &Expression,
    inner: &Expression,
    var: &str,
    order: u32,
    narrate: bool,
) -> Response {
    let outer_arc = compile(outer);
    let inner_arc = compile(inner);
    let var_id = SymbolId::intern(var);
    let center = crate::numeric::expr::Expr::int(0);
    let outer_ts = taylor(&outer_arc, var_id, &center, order as usize);
    let inner_ts = taylor(&inner_arc, var_id, &center, order as usize);
    let mut trace = Trace::new();
    let trace_handle = if narrate { Some(&mut trace) } else { None };
    match compose(&outer_ts, &inner_ts, trace_handle) {
        Some(composed) => {
            let result_arc = composed.to_expr();
            let result = decompile(&result_arc);
            let mut r = Response::default();
            r.results.push((
                ResultKey::Single,
                symbolic_entry(
                    result,
                    EngineId::SeriesComposition,
                    steps_from_trace(&trace),
                ),
            ));
            r.meta.engine_trace.push(EngineId::SeriesComposition);
            r
        }
        None => engine_error(
            "command.compose",
            "series composition requires inner.coeff(0)==0 and shared center/var".to_string(),
        ),
    }
}

pub(super) fn revert_cmd(expr: &Expression, var: &str, order: u32, narrate: bool) -> Response {
    let expr_arc = compile(expr);
    let var_id = SymbolId::intern(var);
    let center = crate::numeric::expr::Expr::int(0);
    let series = taylor(&expr_arc, var_id, &center, order as usize);
    let mut trace = Trace::new();
    let trace_handle = if narrate { Some(&mut trace) } else { None };
    match revert(&series, trace_handle) {
        Some(reverted) => {
            let result_arc = reverted.to_expr();
            let result = decompile(&result_arc);
            let mut r = Response::default();
            r.results.push((
                ResultKey::Single,
                symbolic_entry(result, EngineId::SeriesReversion, steps_from_trace(&trace)),
            ));
            r.meta.engine_trace.push(EngineId::SeriesReversion);
            r
        }
        None => engine_error(
            "command.revert",
            "Lagrange reversion requires a_0=0 and a_1!=0 in the Taylor expansion".to_string(),
        ),
    }
}
