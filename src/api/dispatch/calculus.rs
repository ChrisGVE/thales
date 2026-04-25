//! Calculus command dispatchers (Diff, PartialDiff, Gradient, Integrate,
//! DefIntegrate).

use crate::api::request::{Precision, SolveMode};
use crate::api::response::{
    EngineId, NumericMethod, Response, ResultEntry, ResultKey, ResultShape, ResultValue,
};
use crate::ast::Expression;
use crate::numeric::compile::compile;
use crate::numeric::trace::{Step, TechniqueTag, Trace};

use super::helpers::{engine_error, expression_to_f64, steps_from_trace, symbolic_entry};

pub(super) fn diff_cmd(expr: &Expression, var: &str, order: u32, narrate: bool) -> Response {
    let mut current = expr.clone();
    let mut trace = Trace::new();
    for _ in 0..order.max(1) {
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

pub(super) fn partial_diff_cmd(
    expr: &Expression,
    vars: &[(String, u32)],
    narrate: bool,
) -> Response {
    let mut current = expr.clone();
    let mut trace = Trace::new();
    for (var, order) in vars {
        for _ in 0..(*order).max(1) {
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

pub(super) fn gradient_cmd(expr: &Expression, vars: &[String], narrate: bool) -> Response {
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

pub(super) fn integrate_cmd(expr: &Expression, var: &str, narrate: bool) -> Response {
    match crate::integrate(expr, var) {
        Ok(result) => {
            let mut trace = Trace::new();
            if narrate {
                trace.push(
                    Step::new(
                        TechniqueTag::PatternIntegration,
                        format!("Indefinite integral in {}", var),
                    )
                    .with_output(compile(&result)),
                );
            }
            let mut r = Response::default();
            r.results.push((
                ResultKey::Single,
                symbolic_entry(
                    result,
                    EngineId::PatternIntegration,
                    steps_from_trace(&trace),
                ),
            ));
            r.meta.engine_trace.push(EngineId::PatternIntegration);
            r
        }
        Err(e) => engine_error("command.integrate", format!("{}", e)),
    }
}

pub(super) fn def_integrate_cmd(
    expr: &Expression,
    var: &str,
    from: &Expression,
    to: &Expression,
    narrate: bool,
    mode: SolveMode,
) -> Response {
    let symbolic = crate::integration::definite_integral(expr, var, from, to).ok();
    if let Some(value) = symbolic {
        let mut trace = Trace::new();
        if narrate {
            trace.push(
                Step::new(
                    TechniqueTag::PatternIntegration,
                    format!("Definite integral ∫_{{{}}}^{{{}}}", from, to),
                )
                .with_output(compile(&value)),
            );
        }
        let mut r = Response::default();
        r.results.push((
            ResultKey::Single,
            symbolic_entry(
                value,
                EngineId::PatternIntegration,
                steps_from_trace(&trace),
            ),
        ));
        r.meta.engine_trace.push(EngineId::PatternIntegration);
        return r;
    }

    if matches!(mode, SolveMode::Numeric | SolveMode::PreferSymbolic) {
        if let (Some(a), Some(b)) = (expression_to_f64(from), expression_to_f64(to)) {
            match crate::integration::numerical_integrate(expr, var, a, b, 1e-9) {
                Ok(value) => {
                    let mut r = Response::default();
                    r.results.push((
                        ResultKey::Single,
                        ResultEntry {
                            value: ResultValue::Numeric {
                                value: Expression::Float(value),
                                precision: Precision {
                                    decimal_digits: 12,
                                    abs_tol: None,
                                    rel_tol: None,
                                },
                                method: NumericMethod::AdaptiveQuadrature,
                            },
                            shape: ResultShape::Scalar,
                            unit: None,
                            steps: Vec::new(),
                            alternatives: Vec::new(),
                            engine: EngineId::PatternIntegration,
                        },
                    ));
                    r.meta.engine_trace.push(EngineId::PatternIntegration);
                    return r;
                }
                Err(e) => return engine_error("command.def_integrate", format!("{}", e)),
            }
        }
    }

    engine_error(
        "command.def_integrate",
        "definite integral failed symbolically and no numeric fallback available".to_string(),
    )
}
