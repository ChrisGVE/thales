//! Limit command dispatcher.

use crate::api::command::{LimitPoint as ApiLimitPoint, Side};
use crate::api::response::{EngineId, Response, ResultKey};
use crate::ast::Expression;
use crate::numeric::compile::compile;
use crate::numeric::trace::{Step, TechniqueTag, Trace};

use super::helpers::{engine_error, expression_to_f64, steps_from_trace, symbolic_entry};

pub(super) fn limit_cmd(
    expr: &Expression,
    var: &str,
    point: ApiLimitPoint,
    _side: Option<Side>,
    narrate: bool,
) -> Response {
    let api_point = match point {
        ApiLimitPoint::Finite(e) => {
            let v = expression_to_f64(&e).unwrap_or(0.0);
            crate::limits::LimitPoint::Value(v)
        }
        ApiLimitPoint::PosInf => crate::limits::LimitPoint::PositiveInfinity,
        ApiLimitPoint::NegInf => crate::limits::LimitPoint::NegativeInfinity,
    };
    match crate::limits::limit(expr, var, api_point) {
        Ok(res) => {
            let value = match res {
                crate::limits::LimitResult::Value(v) => Expression::Float(v),
                crate::limits::LimitResult::PositiveInfinity => Expression::Float(f64::INFINITY),
                crate::limits::LimitResult::NegativeInfinity => {
                    Expression::Float(f64::NEG_INFINITY)
                }
                crate::limits::LimitResult::Expression(e) => e,
            };
            let mut trace = Trace::new();
            if narrate {
                trace.push(
                    Step::new(TechniqueTag::LimitAlgebraic, "Limit evaluation")
                        .with_output(compile(&value)),
                );
            }
            let mut r = Response::default();
            r.results.push((
                ResultKey::Single,
                symbolic_entry(value, EngineId::LHopital, steps_from_trace(&trace)),
            ));
            r.meta.engine_trace.push(EngineId::LHopital);
            r
        }
        Err(e) => engine_error("command.limit", format!("{}", e)),
    }
}
