//! ODE command dispatcher.

use crate::api::command::IvpData;
use crate::api::response::{EngineId, Response, ResultKey};
use crate::ast::Expression;
use crate::numeric::compile::decompile;
use crate::numeric::trace::{Step, TechniqueTag, Trace};

use super::helpers::{engine_error, steps_from_trace, symbolic_entry};

pub(super) fn ode_cmd(
    equation: &Expression,
    fn_name: &str,
    var: &str,
    ic: Option<IvpData>,
    narrate: bool,
) -> Response {
    // The v0.8.1 Expression AST cannot carry derivative nodes. The dispatch
    // accepts an `equation` describing the right-hand side of dy/dx = rhs and
    // routes to the first-order engine. IVP data maps to solve_ivp when
    // provided.
    use crate::ode::FirstOrderODE;
    let ode = FirstOrderODE::new(fn_name, var, equation.clone());
    let result = if let Some(ic) = ic {
        crate::ode::solve_ivp(&ode, &ic.var_at, &ic.fn_at)
    } else if ode.is_separable() {
        crate::ode::solve_separable(&ode)
    } else if ode.is_linear() {
        crate::ode::solve_linear(&ode)
    } else {
        return engine_error(
            "command.ode",
            "ODE is neither separable nor linear".to_string(),
        );
    };

    match result {
        Ok(sol) => {
            let value = decompile(&sol.general_solution);
            let mut trace = Trace::new();
            if narrate {
                trace.push(
                    Step::new(TechniqueTag::SeparationOfVariables, sol.method.clone())
                        .with_output(sol.general_solution.clone()),
                );
            }
            let mut r = Response::default();
            r.results.push((
                ResultKey::Single,
                symbolic_entry(value, EngineId::OdeFirstOrder, steps_from_trace(&trace)),
            ));
            r.meta.engine_trace.push(EngineId::OdeFirstOrder);
            r
        }
        Err(e) => engine_error("command.ode", format!("{}", e)),
    }
}
