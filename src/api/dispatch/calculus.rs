//! Calculus command dispatchers (Diff, PartialDiff, Gradient, TotalDiff,
//! Divergence, Curl, Laplacian, Jacobian, Hessian, DirectionalDiff,
//! Integrate, DefIntegrate).

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

/// Total derivative `df/dvar = ∂f/∂var + Σᵢ (∂f/∂nameᵢ)·(d nameᵢ/d var)`
/// where `d nameᵢ/d var = differentiate(depᵢ, var)`.
pub(super) fn total_diff_cmd(
    expr: &Expression,
    var: &str,
    deps: &[(String, Expression)],
    narrate: bool,
) -> Response {
    let mut trace = Trace::new();

    // ∂f/∂var — direct partial.
    let direct = expr.differentiate(var).simplify();
    if narrate {
        trace.push(
            Step::new(
                TechniqueTag::TotalDifferential,
                format!("Direct partial ∂/∂{}", var),
            )
            .with_output(compile(&direct)),
        );
    }

    // Chain terms: Σᵢ (∂f/∂nameᵢ) · (d nameᵢ/d var)
    let mut total = direct;
    for (name, dep_expr) in deps {
        let df_dname = expr.differentiate(name).simplify();
        let dname_dvar = dep_expr.differentiate(var).simplify();
        let chain_term = Expression::Binary(
            crate::ast::BinaryOp::Mul,
            Box::new(df_dname.clone()),
            Box::new(dname_dvar.clone()),
        )
        .simplify();
        if narrate {
            trace.push(
                Step::new(
                    TechniqueTag::TotalDifferential,
                    format!("Chain term (∂f/∂{}) · (d {}/d {})", name, name, var),
                )
                .with_output(compile(&chain_term)),
            );
        }
        total = Expression::Binary(
            crate::ast::BinaryOp::Add,
            Box::new(total),
            Box::new(chain_term),
        )
        .simplify();
    }

    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(total, EngineId::Differentiation, steps_from_trace(&trace)),
    ));
    r.meta.engine_trace.push(EngineId::Differentiation);
    r
}

/// Divergence `∇·F = Σᵢ ∂fieldᵢ/∂varsᵢ`. Returns an engine error when
/// `field.len() != vars.len()`.
pub(super) fn divergence_cmd(field: &[Expression], vars: &[String], narrate: bool) -> Response {
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
    let mut trace = Trace::new();
    let mut sum = Expression::Integer(0);
    for (fi, xi) in field.iter().zip(vars.iter()) {
        let term = fi.differentiate(xi).simplify();
        if narrate {
            trace.push(
                Step::new(TechniqueTag::Divergence, format!("Component ∂/∂{}", xi))
                    .with_output(compile(&term)),
            );
        }
        sum =
            Expression::Binary(crate::ast::BinaryOp::Add, Box::new(sum), Box::new(term)).simplify();
    }
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
pub(super) fn curl_cmd(field: &[Expression], vars: &[String], narrate: bool) -> Response {
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
    let (f1, f2, f3) = (&field[0], &field[1], &field[2]);
    let (x, y, z) = (&vars[0], &vars[1], &vars[2]);

    let mut trace = Trace::new();

    let make_diff = |expr: &Expression, v: &str| expr.differentiate(v).simplify();

    let cx = Expression::Binary(
        crate::ast::BinaryOp::Sub,
        Box::new(make_diff(f3, y)),
        Box::new(make_diff(f2, z)),
    )
    .simplify();
    let cy = Expression::Binary(
        crate::ast::BinaryOp::Sub,
        Box::new(make_diff(f1, z)),
        Box::new(make_diff(f3, x)),
    )
    .simplify();
    let cz = Expression::Binary(
        crate::ast::BinaryOp::Sub,
        Box::new(make_diff(f2, x)),
        Box::new(make_diff(f1, y)),
    )
    .simplify();

    if narrate {
        trace.push(
            Step::new(
                TechniqueTag::Curl,
                format!("Curl x-component ∂F₃/∂{} − ∂F₂/∂{}", y, z),
            )
            .with_output(compile(&cx)),
        );
        trace.push(
            Step::new(
                TechniqueTag::Curl,
                format!("Curl y-component ∂F₁/∂{} − ∂F₃/∂{}", z, x),
            )
            .with_output(compile(&cy)),
        );
        trace.push(
            Step::new(
                TechniqueTag::Curl,
                format!("Curl z-component ∂F₂/∂{} − ∂F₁/∂{}", x, y),
            )
            .with_output(compile(&cz)),
        );
    }

    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        ResultEntry {
            value: ResultValue::Symbolic(cx),
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
pub(super) fn laplacian_cmd(expr: &Expression, vars: &[String], narrate: bool) -> Response {
    let mut trace = Trace::new();
    let mut sum = Expression::Integer(0);
    for xi in vars {
        let d2 = expr.differentiate(xi).differentiate(xi).simplify();
        if narrate {
            trace.push(
                Step::new(
                    TechniqueTag::Laplacian,
                    format!("Second partial ∂²/∂{}²", xi),
                )
                .with_output(compile(&d2)),
            );
        }
        sum = Expression::Binary(crate::ast::BinaryOp::Add, Box::new(sum), Box::new(d2)).simplify();
    }
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
pub(super) fn jacobian_cmd(fields: &[Expression], vars: &[String], narrate: bool) -> Response {
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
pub(super) fn hessian_cmd(expr: &Expression, vars: &[String], narrate: bool) -> Response {
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
pub(super) fn directional_diff_cmd(
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
        let term = Expression::Binary(
            crate::ast::BinaryOp::Mul,
            Box::new(vi.clone()),
            Box::new(fi.clone()),
        )
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
        sum =
            Expression::Binary(crate::ast::BinaryOp::Add, Box::new(sum), Box::new(term)).simplify();
    }
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(sum, EngineId::Differentiation, steps_from_trace(&trace)),
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
