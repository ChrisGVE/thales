//! F1c series-expansion dispatchers (Taylor, Laurent, Asymptotic, Compose,
//! Revert, Puiseux, Frobenius, Pade, Wkb). All nine commands are wired to
//! their respective engines in `crate::numeric::series`.

use crate::api::response::{
    BranchEntry, EngineId, Response, ResultEntry, ResultKey, ResultShape, ResultValue,
    StructuredResult,
};
use crate::ast::Expression;
use crate::numeric::compile::{compile, decompile};
use crate::numeric::series::{
    asymptotic, compose, frobenius, laurent_expand, pade, puiseux, revert, taylor, wkb,
    AsymptoticDirection,
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

/// Puiseux series expansion command.
pub(super) fn puiseux_cmd(
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
                TechniqueTag::PuiseuxExpansion,
                format!(
                    "Puiseux series in {} around {} to order {}",
                    var, center, order
                ),
            )
            .with_input(expr_arc.clone()),
        );
    }
    let mut engine_trace = Trace::new();
    match puiseux(&expr_arc, var_id, &center_arc, order, &mut engine_trace) {
        Some(series) => {
            let result_arc = series.to_expr();
            let result = decompile(&result_arc);
            let coeffs: Vec<Expression> = series
                .terms
                .iter()
                .map(|t| decompile(&t.coefficient))
                .collect();
            let structured = StructuredResult::CoefficientArray {
                coefficients: coeffs,
                variable: var.to_string(),
                center: if center_arc.is_zero() {
                    None
                } else {
                    Some(center.clone())
                },
                order,
            };
            let steps = if narrate {
                steps_from_trace(&engine_trace)
            } else {
                Vec::new()
            };
            let mut entry = symbolic_entry(result, EngineId::PuiseuxExpansion, steps);
            entry.structured = Some(structured);
            let mut r = Response::default();
            r.results.push((ResultKey::Single, entry));
            r.meta.engine_trace.push(EngineId::PuiseuxExpansion);
            r
        }
        None => engine_error(
            "command.puiseux",
            format!(
                "Puiseux expansion failed for {} in {} at {}",
                expr, var, center
            ),
        ),
    }
}

/// Frobenius method command.
///
/// `ode` is the left-hand side of `a(x)y'' + b(x)y' + c(x)y = 0`,
/// encoded as the triple `[a, b, c]` of coefficient expressions in a
/// single `Expression::List` (or a `Mul` placeholder — the dispatcher
/// extracts coefficients directly from the `ode_coefficients` parameter).
///
/// For now the dispatcher accepts `ode` as `[a(x), b(x), c(x)]` encoded
/// as a three-element tuple expression, or falls back to treating the
/// whole expression as `a(x)` with `b=0`, `c=0` if parsing fails.
pub(super) fn frobenius_cmd(
    ode: &Expression,
    _fn_name: &str,
    var: &str,
    point: &Expression,
    order: u32,
    narrate: bool,
) -> Response {
    let var_id = SymbolId::intern(var);
    let point_arc = compile(point);
    let mut engine_trace = Trace::new();

    // Extract ODE coefficients [a, b, c] from the expression.
    // Convention: ode is a list-like expression; we extract via compile.
    let coeffs = extract_ode_coefficients(ode);
    let coeffs_arc: Vec<_> = coeffs.iter().map(|e| compile(e)).collect();

    if narrate {
        engine_trace.push(Step::new(
            TechniqueTag::FrobeniusMethod,
            format!("Frobenius method at {} to order {}", point, order),
        ));
    }

    match frobenius(&coeffs_arc, var_id, &point_arc, order, &mut engine_trace) {
        Some(sol) => {
            let branches: Vec<BranchEntry> = sol
                .solutions
                .iter()
                .enumerate()
                .map(|(i, branch)| {
                    let expr_arc = branch.to_expr(var_id, &point_arc);
                    BranchEntry {
                        condition: None,
                        label: Some(format!("y_{}", i + 1)),
                        value: decompile(&expr_arc),
                    }
                })
                .collect();

            let structured = StructuredResult::Branches {
                branches: branches.clone(),
            };

            let steps = if narrate {
                steps_from_trace(&engine_trace)
            } else {
                Vec::new()
            };

            let mut r = Response::default();
            for (i, b) in branches.iter().enumerate() {
                let mut entry = ResultEntry {
                    value: ResultValue::Symbolic(b.value.clone()),
                    structured: if i == 0 {
                        Some(structured.clone())
                    } else {
                        None
                    },
                    shape: ResultShape::Scalar,
                    unit: None,
                    steps: if i == 0 { steps.clone() } else { Vec::new() },
                    alternatives: Vec::new(),
                    engine: EngineId::FrobeniusMethod,
                };
                // Attach indicial root info to first branch structured.
                if i == 0 {
                    entry.alternatives = sol.indicial_roots.iter().map(|r| decompile(r)).collect();
                }
                r.results.push((ResultKey::Single, entry));
            }
            r.meta.engine_trace.push(EngineId::FrobeniusMethod);
            r
        }
        None => engine_error(
            "command.frobenius",
            format!(
                "Frobenius method failed: check that {} is a regular singular point",
                point
            ),
        ),
    }
}

/// Padé approximant command.
pub(super) fn pade_cmd(
    expr: &Expression,
    var: &str,
    center: &Expression,
    m: u32,
    n: u32,
    narrate: bool,
) -> Response {
    let expr_arc = compile(expr);
    let center_arc = compile(center);
    let var_id = SymbolId::intern(var);
    let mut engine_trace = Trace::new();

    if narrate {
        engine_trace.push(
            Step::new(
                TechniqueTag::PadeApproximant,
                format!("[{m}/{n}] Padé approximant in {} at {}", var, center),
            )
            .with_input(expr_arc.clone()),
        );
    }

    match pade(&expr_arc, var_id, &center_arc, m, n, &mut engine_trace) {
        Some(pa) => {
            let result_arc = pa.to_expr();
            let result = decompile(&result_arc);
            let num_coeffs: Vec<Expression> = pa.numerator.iter().map(|c| decompile(c)).collect();
            let den_coeffs: Vec<Expression> = pa.denominator.iter().map(|c| decompile(c)).collect();
            let structured = StructuredResult::CoefficientArray {
                coefficients: num_coeffs.into_iter().chain(den_coeffs).collect(),
                variable: var.to_string(),
                center: if center_arc.is_zero() {
                    None
                } else {
                    Some(center.clone())
                },
                order: m + n,
            };
            let steps = if narrate {
                steps_from_trace(&engine_trace)
            } else {
                Vec::new()
            };
            let mut entry = symbolic_entry(result, EngineId::PadeApproximant, steps);
            entry.structured = Some(structured);
            let mut r = Response::default();
            r.results.push((ResultKey::Single, entry));
            r.meta.engine_trace.push(EngineId::PadeApproximant);
            r
        }
        None => engine_error(
            "command.pade",
            format!(
                "Padé [{m}/{n}] approximant failed for {} in {} at {}",
                expr, var, center
            ),
        ),
    }
}

/// WKB approximation command.
///
/// `ode` encodes `Q(x)` in `ε²y'' + Q(x)y = 0`.
pub(super) fn wkb_cmd(
    ode: &Expression,
    _fn_name: &str,
    var: &str,
    small_param: &str,
    order: u32,
    narrate: bool,
) -> Response {
    let q_arc = compile(ode);
    let var_id = SymbolId::intern(var);
    let eps_id = SymbolId::intern(small_param);
    let mut engine_trace = Trace::new();

    if narrate {
        engine_trace.push(
            Step::new(
                TechniqueTag::WkbApproximation,
                format!(
                    "WKB approximation for Q={}, ε={}, to order {}",
                    ode, small_param, order
                ),
            )
            .with_input(q_arc.clone()),
        );
    }

    match wkb(&q_arc, var_id, eps_id, order, &mut engine_trace) {
        Some(sol) => {
            let (y_plus, y_minus) = sol.to_expr();
            let branches = vec![
                BranchEntry {
                    condition: None,
                    label: Some("y_+".to_string()),
                    value: decompile(&y_plus),
                },
                BranchEntry {
                    condition: None,
                    label: Some("y_-".to_string()),
                    value: decompile(&y_minus),
                },
            ];
            let structured = StructuredResult::Branches {
                branches: branches.clone(),
            };
            let steps = if narrate {
                steps_from_trace(&engine_trace)
            } else {
                Vec::new()
            };
            let mut r = Response::default();
            for (i, b) in branches.iter().enumerate() {
                let mut entry = ResultEntry {
                    value: ResultValue::Symbolic(b.value.clone()),
                    structured: if i == 0 {
                        Some(structured.clone())
                    } else {
                        None
                    },
                    shape: ResultShape::Scalar,
                    unit: None,
                    steps: if i == 0 { steps.clone() } else { Vec::new() },
                    alternatives: Vec::new(),
                    engine: EngineId::WkbExpansion,
                };
                r.results.push((ResultKey::Single, entry));
            }
            r.meta.engine_trace.push(EngineId::WkbExpansion);
            r
        }
        None => engine_error(
            "command.wkb",
            format!("WKB approximation failed for Q={} in {}", ode, var),
        ),
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Extract ODE coefficients `[a(x), b(x), c(x)]` from an `Expression`.
///
/// Convention: the expression is either a `Function::Custom("coefficients")`
/// or `Function::Custom("list")` with three arguments, or the expression
/// itself is treated as `a(x)` with `b = 0`, `c = 0`.
fn extract_ode_coefficients(ode: &Expression) -> Vec<Expression> {
    use crate::ast::Function as AstF;
    if let Expression::Function(AstF::Custom(name), args) = ode {
        if (name == "list" || name == "coefficients") && args.len() == 3 {
            return args.clone();
        }
    }
    // Fallback: treat the whole expression as a(x), with b=0, c=0.
    vec![ode.clone(), Expression::Integer(0), Expression::Integer(0)]
}
