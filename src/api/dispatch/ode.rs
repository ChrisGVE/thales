//! ODE command dispatcher.

use std::sync::Arc;

use crate::api::command::{IvpData, SystemIvpData};
use crate::api::diagnostic::{Diagnostic, DiagnosticCode};
use crate::api::narrative::Narrative;
use crate::api::response::{EngineId, Response, ResultKey};
use crate::ast::Expression;
use crate::numeric::compile::decompile;
use crate::numeric::trace::{Step, TechniqueTag, Trace};

use super::helpers::{engine_error, expression_to_f64, steps_from_trace, symbolic_entry};

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
                let (tag, detail) = if ode.is_separable() {
                    (
                        TechniqueTag::SeparationOfVariables,
                        format!(
                            "dep={dep};indep={indep}",
                            dep = ode.dependent,
                            indep = ode.independent
                        ),
                    )
                } else {
                    (
                        TechniqueTag::IntegratingFactor,
                        format!(
                            "method={method};var={var}",
                            method = sol.method,
                            var = ode.independent
                        ),
                    )
                };
                trace.push(Step::new(tag, detail).with_output(sol.general_solution.clone()));
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

pub(super) fn ode_system_cmd(
    equations: &[Expression],
    fn_names: &[String],
    var: &str,
    ic: Option<SystemIvpData>,
    narrate: bool,
) -> Response {
    use crate::ode::system::{solve_linear_system, solve_system_numeric, OdeSystem};

    let system = match OdeSystem::new(equations.to_vec(), fn_names.to_vec(), var.to_string()) {
        Ok(s) => s,
        Err(e) => return engine_error("command.ode_system", format!("{}", e)),
    };

    // Try symbolic path first (2×2 linear constant-coefficient only).
    if let Ok(sol) = solve_linear_system(&system) {
        let mut r = Response::default();
        for (i, component) in sol.components.iter().enumerate() {
            let value = decompile(component);
            let name = fn_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("y{}", i + 1));
            let steps = if narrate {
                let mut t = Trace::new();
                for s in &sol.steps {
                    t.push(
                        Step::new(TechniqueTag::CharacteristicEquation, s.clone())
                            .with_output(Arc::clone(component)),
                    );
                }
                steps_from_trace(&t)
            } else {
                Vec::new()
            };
            r.results.push((
                ResultKey::Component(name),
                symbolic_entry(value, EngineId::OdeSystem, steps),
            ));
        }
        r.meta.engine_trace.push(EngineId::OdeSystem);
        return r;
    }

    // Fallback: numeric RK4 when IC provided.
    if let Some(ref ic_data) = ic {
        let t0 = match expression_to_f64(&ic_data.var_at) {
            Some(v) => v,
            None => {
                return engine_error(
                    "command.ode_system",
                    "initial condition var_at is not a numeric value".into(),
                )
            }
        };
        let y0: Option<Vec<f64>> = ic_data.values_at.iter().map(expression_to_f64).collect();
        let y0 = match y0 {
            Some(v) => v,
            None => {
                return engine_error(
                    "command.ode_system",
                    "initial condition values_at contains non-numeric expression".into(),
                )
            }
        };
        let t_end = t0 + 10.0;
        let steps = 1000;
        match solve_system_numeric(&system, t0, y0, t_end, steps) {
            Ok(num_sol) => {
                let mut r = Response::default();
                for (i, &yf) in num_sol.y_final.iter().enumerate() {
                    let name = fn_names
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("y{}", i + 1));
                    r.results.push((
                        ResultKey::Component(name),
                        symbolic_entry(Expression::Float(yf), EngineId::OdeSystem, Vec::new()),
                    ));
                }
                r.meta.engine_trace.push(EngineId::OdeSystem);
                r
            }
            Err(e) => engine_error("command.ode_system", format!("{}", e)),
        }
    } else {
        engine_error(
            "command.ode_system",
            "system is not linear constant-coefficient; provide initial conditions for numeric fallback".into(),
        )
    }
}

pub(super) fn pde_cmd(
    _equation: &Expression,
    _fn_name: &str,
    _vars: &[String],
    _narrate: bool,
) -> Response {
    let mut r = Response::default();
    r.diagnostics.push(Diagnostic::of(
        DiagnosticCode::NotImplemented,
        Narrative::new("command.pde", "PDE solving not yet supported (v0.12.0)."),
    ));
    r
}
