//! Vector calculus command dispatchers: divergence, curl, laplacian,
//! jacobian, hessian, directional_diff.

use crate::api::response::{EngineId, Response, ResultEntry, ResultKey, ResultShape, ResultValue};
use crate::ast::{BinaryOp, Expression};
use crate::calculus::multivar;
use crate::numeric::compile::{compile, decompile};
use crate::numeric::trace::{Step, TechniqueTag, Trace};
use crate::numeric::SymbolId;

use super::super::helpers::{engine_error, steps_from_trace, symbolic_entry};

/// Divergence `∇·F = Σᵢ ∂fieldᵢ/∂varsᵢ`. Returns an engine error when
/// `field.len() != vars.len()`.
pub(in crate::api::dispatch) fn divergence_cmd(
    field: &[Expression],
    vars: &[String],
    narrate: bool,
) -> Response {
    if field.len() != vars.len() {
        return engine_error(
            "command.divergence",
            format!(
                "field has {} components but {} variables were supplied",
                field.len(),
                vars.len()
            ),
        );
    }
    let arc_field: Vec<_> = field.iter().map(compile).collect();
    let sym_ids: Vec<SymbolId> = vars.iter().map(|v| SymbolId::intern(v)).collect();

    let mut trace = Trace::new();
    if narrate {
        for (arc_fi, xi) in arc_field.iter().zip(vars.iter()) {
            let term = multivar::divergence(std::slice::from_ref(arc_fi), &[SymbolId::intern(xi)]);
            trace.push(
                Step::new(TechniqueTag::Divergence, format!("Component ∂/∂{}", xi))
                    .with_output(term),
            );
        }
    }
    let result = multivar::divergence(&arc_field, &sym_ids);
    let sum = decompile(&result);

    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(sum, EngineId::Differentiation, steps_from_trace(&trace)),
    ));
    r.meta.engine_trace.push(EngineId::Differentiation);
    r
}

/// Curl `∇×F` for a 3-D vector field. Returns an engine error when
/// `field.len() != 3 || vars.len() != 3`.
///
/// Result: `(∂F₃/∂y − ∂F₂/∂z, ∂F₁/∂z − ∂F₃/∂x, ∂F₂/∂x − ∂F₁/∂y)`.
/// Packaged as `ResultShape::Vector` with primary = first component.
pub(in crate::api::dispatch) fn curl_cmd(
    field: &[Expression],
    vars: &[String],
    narrate: bool,
) -> Response {
    if field.len() != 3 || vars.len() != 3 {
        return engine_error(
            "command.curl",
            format!(
                "curl requires exactly 3 field components and 3 variables, got {} and {}",
                field.len(),
                vars.len()
            ),
        );
    }
    let arc_field: [_; 3] = [compile(&field[0]), compile(&field[1]), compile(&field[2])];
    let sym_ids: [SymbolId; 3] = [
        SymbolId::intern(&vars[0]),
        SymbolId::intern(&vars[1]),
        SymbolId::intern(&vars[2]),
    ];
    let (x, y, z) = (&vars[0], &vars[1], &vars[2]);

    let [cx_arc, cy_arc, cz_arc] = multivar::curl(&arc_field, &sym_ids);
    let cx = decompile(&cx_arc);
    let cy = decompile(&cy_arc);
    let cz = decompile(&cz_arc);

    let mut trace = Trace::new();
    if narrate {
        trace.push(
            Step::new(
                TechniqueTag::Curl,
                format!("Curl x-component ∂F₃/∂{} − ∂F₂/∂{}", y, z),
            )
            .with_output(cx_arc),
        );
        trace.push(
            Step::new(
                TechniqueTag::Curl,
                format!("Curl y-component ∂F₁/∂{} − ∂F₃/∂{}", z, x),
            )
            .with_output(cy_arc),
        );
        trace.push(
            Step::new(
                TechniqueTag::Curl,
                format!("Curl z-component ∂F₂/∂{} − ∂F₁/∂{}", x, y),
            )
            .with_output(cz_arc),
        );
    }

    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        ResultEntry {
            value: ResultValue::Symbolic(cx),
            structured: None,
            shape: ResultShape::Vector,
            unit: None,
            steps: if narrate {
                steps_from_trace(&trace)
            } else {
                Vec::new()
            },
            alternatives: vec![cy, cz],
            engine: EngineId::Differentiation,
        },
    ));
    r.meta.engine_trace.push(EngineId::Differentiation);
    r
}

/// Laplacian `∇²f = Σᵢ ∂²f/∂varsᵢ²`.
pub(in crate::api::dispatch) fn laplacian_cmd(
    expr: &Expression,
    vars: &[String],
    narrate: bool,
) -> Response {
    let arc_expr = compile(expr);
    let sym_ids: Vec<SymbolId> = vars.iter().map(|v| SymbolId::intern(v)).collect();

    let mut trace = Trace::new();
    if narrate {
        for xi in vars {
            let xi_id = SymbolId::intern(xi);
            let d2 = multivar::laplacian(&arc_expr, std::slice::from_ref(&xi_id));
            trace.push(
                Step::new(
                    TechniqueTag::Laplacian,
                    format!("Second partial ∂²/∂{}²", xi),
                )
                .with_output(d2),
            );
        }
    }
    let result = multivar::laplacian(&arc_expr, &sym_ids);
    let sum = decompile(&result);

    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(sum, EngineId::Differentiation, steps_from_trace(&trace)),
    ));
    r.meta.engine_trace.push(EngineId::Differentiation);
    r
}

/// Jacobian matrix `J[i][j] = ∂fieldsᵢ/∂varsⱼ`.
/// Packaged as `ResultShape::Matrix`; primary = J[0][0], alternatives = remaining
/// row-major entries.
pub(in crate::api::dispatch) fn jacobian_cmd(
    fields: &[Expression],
    vars: &[String],
    narrate: bool,
) -> Response {
    let mut trace = Trace::new();
    let mut entries: Vec<Expression> = Vec::with_capacity(fields.len() * vars.len());

    for fi in fields {
        for xj in vars {
            let entry = fi.differentiate(xj).simplify();
            if narrate {
                trace.push(
                    Step::new(TechniqueTag::Jacobian, format!("Jacobian entry ∂f/∂{}", xj))
                        .with_output(compile(&entry)),
                );
            }
            entries.push(entry);
        }
    }

    let (primary, alternatives) = match entries.split_first() {
        Some((first, rest)) => (first.clone(), rest.to_vec()),
        None => (Expression::Integer(0), Vec::new()),
    };
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        ResultEntry {
            value: ResultValue::Symbolic(primary),
            structured: None,
            shape: ResultShape::Matrix,
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

/// Hessian matrix `H[i][j] = ∂²f/(∂varsᵢ ∂varsⱼ)`.
/// Same matrix packaging as Jacobian.
pub(in crate::api::dispatch) fn hessian_cmd(
    expr: &Expression,
    vars: &[String],
    narrate: bool,
) -> Response {
    let mut trace = Trace::new();
    let mut entries: Vec<Expression> = Vec::with_capacity(vars.len() * vars.len());

    for xi in vars {
        for xj in vars {
            let entry = expr.differentiate(xi).differentiate(xj).simplify();
            if narrate {
                trace.push(
                    Step::new(
                        TechniqueTag::Hessian,
                        format!("Hessian entry ∂²f/(∂{} ∂{})", xi, xj),
                    )
                    .with_output(compile(&entry)),
                );
            }
            entries.push(entry);
        }
    }

    let (primary, alternatives) = match entries.split_first() {
        Some((first, rest)) => (first.clone(), rest.to_vec()),
        None => (Expression::Integer(0), Vec::new()),
    };
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        ResultEntry {
            value: ResultValue::Symbolic(primary),
            structured: None,
            shape: ResultShape::Matrix,
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

/// Directional derivative `Σᵢ directionᵢ · ∂expr/∂varsᵢ`.
/// Returns an engine error when `direction.len() != vars.len()`.
pub(in crate::api::dispatch) fn directional_diff_cmd(
    expr: &Expression,
    vars: &[String],
    direction: &[Expression],
    narrate: bool,
) -> Response {
    if direction.len() != vars.len() {
        return engine_error(
            "command.directional_diff",
            format!(
                "direction has {} components but {} variables were supplied",
                direction.len(),
                vars.len()
            ),
        );
    }
    let mut trace = Trace::new();
    let mut sum = Expression::Integer(0);
    for ((xi, vi), fi) in vars.iter().zip(direction.iter()).zip(
        vars.iter()
            .map(|xi| expr.differentiate(xi).simplify())
            .collect::<Vec<_>>()
            .iter(),
    ) {
        let term = Expression::Binary(BinaryOp::Mul, Box::new(vi.clone()), Box::new(fi.clone()))
            .simplify();
        if narrate {
            trace.push(
                Step::new(
                    TechniqueTag::DirectionalDerivative,
                    format!("Direction component v_{} · ∂/∂{}", xi, xi),
                )
                .with_output(compile(&term)),
            );
        }
        sum = Expression::Binary(BinaryOp::Add, Box::new(sum), Box::new(term)).simplify();
    }
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(sum, EngineId::Differentiation, steps_from_trace(&trace)),
    ));
    r.meta.engine_trace.push(EngineId::Differentiation);
    r
}
