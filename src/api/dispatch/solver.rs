//! Solver command dispatchers (SolveFor, SolveSystem).

use crate::api::response::{EngineId, Response, ResultEntry, ResultKey, ResultShape, ResultValue};
use crate::ast::{Expression, Variable};

use super::algebra::rearrange_cmd;
use super::helpers::{engine_error, expression_to_equation, steps_from_trace};

pub(super) fn solve_for_cmd(relation: &Expression, var: &str, narrate: bool) -> Response {
    rearrange_cmd(relation, var, narrate)
}

pub(super) fn solve_system_cmd(
    equations: &[Expression],
    vars: &[String],
    narrate: bool,
) -> Response {
    let eqs: Vec<_> = equations.iter().map(expression_to_equation).collect();
    let variables: Vec<_> = vars.iter().map(|v| Variable::new(v)).collect();
    let solver = crate::equation_system::SmartSystemSolver::new();
    match solver.solve_with_path(&eqs, &variables) {
        Ok((pairs, trace)) => {
            let mut r = Response::default();
            for (name, value) in pairs {
                r.results.push((
                    ResultKey::Single,
                    ResultEntry {
                        value: ResultValue::Symbolic(value),
                        shape: ResultShape::Scalar,
                        unit: None,
                        steps: if narrate {
                            steps_from_trace(&trace)
                        } else {
                            Vec::new()
                        },
                        alternatives: Vec::new(),
                        engine: EngineId::SystemSolver,
                    },
                ));
                let _ = name;
            }
            r.meta.engine_trace.push(EngineId::SystemSolver);
            r
        }
        Err(e) => engine_error("command.solve_system", format!("{:?}", e)),
    }
}
