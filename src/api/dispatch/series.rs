//! Series command dispatchers (FourierSeries; Taylor/Laurent/Asymptotic/
//! Compose/Revert routed via not_implemented in mod.rs until engines wire up).

use crate::api::response::{EngineId, Response, ResultKey};
use crate::ast::{Expression, Variable};
use crate::numeric::compile::compile;
use crate::numeric::trace::{Step, TechniqueTag, Trace};

use super::helpers::{engine_error, expression_to_f64, steps_from_trace, symbolic_entry};

pub(super) fn fourier_series_cmd(
    expr: &Expression,
    var: &str,
    period: &Expression,
    terms: u32,
    narrate: bool,
) -> Response {
    let period_val = match expression_to_f64(period) {
        Some(v) => v,
        None => {
            return engine_error(
                "command.fourier_series",
                "period must evaluate to a finite number".to_string(),
            );
        }
    };
    let variable = Variable::new(var);
    match crate::fourier::fourier_series(expr, &variable, terms as usize, Some(period_val)) {
        Ok(series) => {
            // `FourierSeries` carries coefficients, not a reconstructed
            // `Expression`. Return the display string via Narrative steps and
            // keep the raw primary value as the leading a_0/2 constant so
            // downstream pattern matches see a scalar.
            let primary = Expression::Float(series.a_coefficients[0] / 2.0);
            let mut trace = Trace::new();
            if narrate {
                trace.push(
                    Step::new(
                        TechniqueTag::FourierSeries,
                        format!(
                            "Fourier series on period {}: {}",
                            period_val,
                            series.to_display_string()
                        ),
                    )
                    .with_output(compile(&primary)),
                );
            }
            let mut r = Response::default();
            r.results.push((
                ResultKey::Single,
                symbolic_entry(primary, EngineId::FourierSeries, steps_from_trace(&trace)),
            ));
            r.meta.engine_trace.push(EngineId::FourierSeries);
            r
        }
        Err(e) => engine_error("command.fourier_series", format!("{:?}", e)),
    }
}
