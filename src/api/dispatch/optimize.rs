//! F1f optimization dispatchers (Optimize, LagrangeMult).
//!
//! Both commands route to `crate::numerical::optimize_constrained`. The
//! Optimize command lifts api-level [`Constraint`] values into equality
//! expressions when possible (a `Constraint::Equality(g)` becomes the
//! constraint `g = 0`); inequality constraints surface an engine error
//! since the underlying solver handles equality KKT only.
//!
//! For Optimize with [`OptSense::Maximize`], the objective is negated
//! before solving and the resulting `objective_value` is negated back.

use crate::api::command::{Constraint, OptSense};
use crate::api::response::{
    EngineId, NarratedStep, Response, ResultEntry, ResultKey, ResultShape, ResultValue,
};
use crate::ast::{Expression, UnaryOp, Variable};
use crate::numeric::trace::{Step, TechniqueTag, Trace};
use crate::numerical::{optimize_constrained, OptimizationType};

use super::helpers::{engine_error, steps_from_trace};

fn classification_label(t: &OptimizationType) -> &'static str {
    use OptimizationType::*;
    match t {
        LocalMinimum => "local minimum",
        LocalMaximum => "local maximum",
        SaddlePoint => "saddle point",
        Inconclusive => "inconclusive",
    }
}

fn negate(e: &Expression) -> Expression {
    Expression::Unary(UnaryOp::Neg, Box::new(e.clone()))
}

fn build_steps_from_result(
    point: &[(String, f64)],
    multipliers: &[f64],
    objective_value: f64,
    classification: &OptimizationType,
    narrate: bool,
) -> Vec<NarratedStep> {
    if !narrate {
        return Vec::new();
    }
    let mut trace = Trace::new();
    let coord = point
        .iter()
        .map(|(n, v)| format!("{}={:.6}", n, v))
        .collect::<Vec<_>>()
        .join(", ");
    trace.push(Step::new(
        TechniqueTag::CharacteristicEquation,
        format!("Critical point: ({})", coord),
    ));
    if !multipliers.is_empty() {
        let mults = multipliers
            .iter()
            .enumerate()
            .map(|(i, m)| format!("λ{}={:.6}", i, m))
            .collect::<Vec<_>>()
            .join(", ");
        trace.push(Step::new(
            TechniqueTag::CharacteristicEquation,
            format!("Lagrange multipliers: {}", mults),
        ));
    }
    trace.push(Step::new(
        TechniqueTag::CharacteristicEquation,
        format!(
            "Objective value: {:.6} ({})",
            objective_value,
            classification_label(classification)
        ),
    ));
    steps_from_trace(&trace)
}

pub(super) fn optimize_cmd(
    objective: &Expression,
    vars: &[String],
    constraints: &[Constraint],
    sense: OptSense,
    narrate: bool,
) -> Response {
    // Lift Constraint::Equality(g) into the bare equality expression g = 0.
    // Reject other variants — the engine handles equality KKT only.
    let mut equality_constraints: Vec<Expression> = Vec::with_capacity(constraints.len());
    for c in constraints {
        match c {
            Constraint::Equality(e) => equality_constraints.push(e.clone()),
            _ => {
                return engine_error(
                    "command.optimize",
                    "inequality and conditional constraints are not yet supported by the v0.9.0 optimiser".to_string(),
                );
            }
        }
    }

    let variables: Vec<Variable> = vars.iter().map(|s| Variable::new(s)).collect();
    let solve_obj = match sense {
        OptSense::Minimize => objective.clone(),
        OptSense::Maximize => negate(objective),
    };

    match optimize_constrained(&solve_obj, &equality_constraints, &variables) {
        Ok(mut r) => {
            // Undo the negation for Maximize so the returned objective_value
            // matches the user-supplied objective.
            if matches!(sense, OptSense::Maximize) {
                r.objective_value = -r.objective_value;
            }
            let coord_pairs: Vec<Expression> = r
                .point
                .iter()
                .flat_map(|(name, value)| {
                    vec![
                        Expression::Variable(Variable::new(name)),
                        Expression::Float(*value),
                    ]
                })
                .collect();
            // Primary value carries objective at optimum; alternatives encode
            // the (variable, value) pairs in row-major order.
            let primary = Expression::Float(r.objective_value);
            let entry = ResultEntry {
                value: ResultValue::Symbolic(primary),
                shape: ResultShape::Set,
                unit: None,
                steps: build_steps_from_result(
                    &r.point,
                    &r.multipliers,
                    r.objective_value,
                    &r.classification,
                    narrate,
                ),
                alternatives: coord_pairs,
                engine: EngineId::Optimizer,
            };
            let mut response = Response::default();
            response.results.push((ResultKey::Single, entry));
            response.meta.engine_trace.push(EngineId::Optimizer);
            response
        }
        Err(e) => engine_error("command.optimize", format!("{}", e)),
    }
}

pub(super) fn lagrange_mult_cmd(
    objective: &Expression,
    vars: &[String],
    equality_constraints: &[Expression],
    narrate: bool,
) -> Response {
    let variables: Vec<Variable> = vars.iter().map(|s| Variable::new(s)).collect();
    match optimize_constrained(objective, equality_constraints, &variables) {
        Ok(r) => {
            let coord_pairs: Vec<Expression> = r
                .point
                .iter()
                .flat_map(|(name, value)| {
                    vec![
                        Expression::Variable(Variable::new(name)),
                        Expression::Float(*value),
                    ]
                })
                .collect();
            let multiplier_values: Vec<Expression> = r
                .multipliers
                .iter()
                .map(|m| Expression::Float(*m))
                .collect();
            let mut alternatives = coord_pairs;
            alternatives.extend(multiplier_values);

            let primary = Expression::Float(r.objective_value);
            let entry = ResultEntry {
                value: ResultValue::Symbolic(primary),
                shape: ResultShape::Set,
                unit: None,
                steps: build_steps_from_result(
                    &r.point,
                    &r.multipliers,
                    r.objective_value,
                    &r.classification,
                    narrate,
                ),
                alternatives,
                engine: EngineId::Optimizer,
            };
            let mut response = Response::default();
            response.results.push((ResultKey::Single, entry));
            response.meta.engine_trace.push(EngineId::Optimizer);
            response
        }
        Err(e) => engine_error("command.lagrange_mult", format!("{}", e)),
    }
}
