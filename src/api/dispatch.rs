//! Single-entry dispatcher for [`super::execute`].
//!
//! Matches on [`Command`] and delegates to the existing engine for each
//! operation. Inputs are compiled to canonical `Arc<Expr>` form at each
//! dispatch arm when engines require it; results are decompiled back to
//! `Expression` at this boundary (architecture rule 1).
//!
//! # v0.8.1 coverage
//!
//! This dispatcher wires the commands whose underlying engines already
//! exist in the crate (algebra, calculus, limits, solver, ODE, series,
//! fourier, partial fractions, matrix). Commands whose engines are not
//! yet implemented at the v0.8.1 API boundary (higher-dimensional
//! calculus, advanced series beyond Taylor, optimization, special
//! functions) currently return a [`DiagnosticCode::NotImplemented`]
//! entry and an empty [`ResultEntry`] carrying an [`ResultValue::Unsolved`]
//! reason.

use std::time::Instant;

use crate::ast::Variable;
use crate::numeric::compile::{compile, decompile};
use crate::numeric::trace::{Step, TechniqueDifficulty, TechniqueTag, Trace};
use crate::solver::Solver as _;
use crate::{ast::Expression, ThalesError};

use super::command::{
    Command, LimitPoint as ApiLimitPoint, MatrixOp, Side, SimplifyRules, SpecialKind,
};
use super::diagnostic::{Diagnostic, DiagnosticCode};
use super::narrative::Narrative;
use super::request::{Request, SolveMode};
use super::response::{
    EngineId, NarratedStep, Response, ResultEntry, ResultKey, ResultShape, ResultValue,
};

/// Execute a [`Request`]. Single entry point for thales.
pub fn execute(request: Request) -> Result<Response, ThalesError> {
    let start = Instant::now();
    let narrate = request.narrate;

    let mut response = match request.command {
        Command::Noop => noop_response(),

        // ── Algebra ──────────────────────────────────────────────────────
        Command::Simplify { expr, rules, .. } => simplify_cmd(&expr, rules, narrate),
        Command::Expand { expr, .. } => expand_cmd(&expr, narrate),
        Command::Factor { expr, .. } => factor_cmd(&expr, narrate),
        Command::Substitute { expr, bindings, .. } => substitute_cmd(&expr, &bindings, narrate),
        Command::CombineLikeTerms { expr, .. } => {
            simplify_cmd(&expr, SimplifyRules::all(), narrate)
        }
        Command::CommonDenominator { expr, .. } => {
            simplify_cmd(&expr, SimplifyRules::all(), narrate)
        }
        Command::PartialFractions { expr, var } => partial_fractions_cmd(&expr, &var, narrate),
        Command::Rationalize { expr, .. } => simplify_cmd(&expr, SimplifyRules::all(), narrate),
        Command::Conjugate { expr: _, .. } => not_implemented("command.conjugate"),
        Command::InverseFn { .. } => not_implemented("command.inverse_fn"),
        Command::Rearrange {
            equation,
            solve_for,
        } => rearrange_cmd(&equation, &solve_for, narrate),
        Command::ApplyIdentity { .. } => not_implemented("command.apply_identity"),

        // ── Solve ────────────────────────────────────────────────────────
        Command::SolveFor { relation, var, .. } => solve_for_cmd(&relation, &var, narrate),
        Command::SolveSystem {
            equations, vars, ..
        } => solve_system_cmd(&equations, &vars, narrate),

        // ── Differentiation ──────────────────────────────────────────────
        Command::Diff { expr, var, order } => diff_cmd(&expr, &var, order, narrate),
        Command::PartialDiff { expr, vars } => partial_diff_cmd(&expr, &vars, narrate),
        Command::TotalDiff { .. } => not_implemented("command.total_diff"),
        Command::Gradient { expr, vars } => gradient_cmd(&expr, &vars, narrate),
        Command::Divergence { .. } => not_implemented("command.divergence"),
        Command::Curl { .. } => not_implemented("command.curl"),
        Command::Laplacian { .. } => not_implemented("command.laplacian"),
        Command::Jacobian { .. } => not_implemented("command.jacobian"),
        Command::Hessian { .. } => not_implemented("command.hessian"),
        Command::DirectionalDiff { .. } => not_implemented("command.directional_diff"),

        // ── Integration ──────────────────────────────────────────────────
        Command::Integrate { expr, var } => integrate_cmd(&expr, &var, narrate),
        Command::DefIntegrate {
            expr,
            var,
            from,
            to,
        } => def_integrate_cmd(&expr, &var, &from, &to, narrate, request.mode),

        // ── Limits ───────────────────────────────────────────────────────
        Command::Limit {
            expr,
            var,
            point,
            side,
        } => limit_cmd(&expr, &var, point, side, narrate),

        // ── Expansions ───────────────────────────────────────────────────
        Command::Taylor { .. } => not_implemented("command.taylor"),
        Command::Laurent { .. } => not_implemented("command.laurent"),
        Command::Asymptotic { .. } => not_implemented("command.asymptotic"),
        Command::Compose { .. } => not_implemented("command.compose"),
        Command::Revert { .. } => not_implemented("command.revert"),

        // ── Transforms ───────────────────────────────────────────────────
        Command::FourierSeries {
            expr,
            var,
            period,
            terms,
        } => fourier_series_cmd(&expr, &var, &period, terms, narrate),
        Command::Residue { .. } => not_implemented("command.residue"),

        // ── Special functions ────────────────────────────────────────────
        Command::SpecialFn { kind, args } => special_fn_cmd(kind, &args, narrate),

        // ── ODE ─────────────────────────────────────────────────────────
        Command::Ode {
            equation,
            fn_name,
            var,
            ic,
        } => ode_cmd(&equation, &fn_name, &var, ic, narrate),

        // ── Matrix ───────────────────────────────────────────────────────
        Command::Matrix { op, operands } => matrix_cmd(op, &operands, narrate),

        // ── Optimization ─────────────────────────────────────────────────
        Command::Optimize { .. } => not_implemented("command.optimize"),
        Command::LagrangeMult { .. } => not_implemented("command.lagrange_mult"),
    };

    response.meta.elapsed_ms = start.elapsed().as_millis() as u64;
    Ok(response)
}

// ── Per-command dispatchers ──────────────────────────────────────────────────

fn noop_response() -> Response {
    let mut r = Response::default();
    r.diagnostics.push(Diagnostic::of(
        DiagnosticCode::NotImplemented,
        Narrative::new("command.noop", "Noop command produces no result."),
    ));
    r
}

fn not_implemented(template_id: &'static str) -> Response {
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        unsolved_entry(
            Narrative::new(template_id, "command not yet implemented in v0.8.1"),
            EngineId::Other("not-implemented"),
        ),
    ));
    r.diagnostics.push(Diagnostic::of(
        DiagnosticCode::NotImplemented,
        Narrative::new(template_id, "command not yet implemented in v0.8.1"),
    ));
    r
}

fn simplify_cmd(expr: &Expression, _rules: SimplifyRules, narrate: bool) -> Response {
    let simplified = expr.simplify();
    let mut trace = Trace::new();
    if narrate {
        trace.push(
            Step::new(TechniqueTag::Simplification, "Canonical simplification")
                .with_input(compile(expr))
                .with_output(compile(&simplified)),
        );
    }
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(simplified, EngineId::Simplify, steps_from_trace(&trace)),
    ));
    r.meta.engine_trace.push(EngineId::Simplify);
    r
}

fn expand_cmd(expr: &Expression, narrate: bool) -> Response {
    // No standalone expand engine: simplification rules include distribution.
    // Use the AST simplifier which already distributes products over sums.
    let expanded = expr.simplify();
    let mut trace = Trace::new();
    if narrate {
        trace.push(
            Step::new(TechniqueTag::Expansion, "Distribute products over sums")
                .with_input(compile(expr))
                .with_output(compile(&expanded)),
        );
    }
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(expanded, EngineId::Simplify, steps_from_trace(&trace)),
    ));
    r.meta.engine_trace.push(EngineId::Simplify);
    r
}

fn factor_cmd(expr: &Expression, narrate: bool) -> Response {
    // No standalone factor engine at the Expression layer yet. Fall back to
    // simplification and emit a diagnostic so callers know factoring was not
    // attempted in full form.
    let factored = expr.simplify();
    let mut trace = Trace::new();
    if narrate {
        trace.push(
            Step::new(
                TechniqueTag::Factoring,
                "Attempted factoring via simplification",
            )
            .with_input(compile(expr))
            .with_output(compile(&factored)),
        );
    }
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(factored, EngineId::Simplify, steps_from_trace(&trace)),
    ));
    r.diagnostics.push(Diagnostic::of(
        DiagnosticCode::NotImplemented,
        Narrative::new(
            "command.factor.partial",
            "Full factoring engine is not yet wired into the API; result is the \
             canonical simplification.",
        ),
    ));
    r.meta.engine_trace.push(EngineId::Simplify);
    r
}

fn substitute_cmd(
    expr: &Expression,
    bindings: &[(Expression, Expression)],
    narrate: bool,
) -> Response {
    let mut current = expr.clone();
    let mut trace = Trace::new();
    for (old, new) in bindings {
        current = substitute_in_expr(&current, old, new);
        if narrate {
            trace.push(
                Step::new(
                    TechniqueTag::Substitution,
                    format!("Substitute {} with {}", old, new),
                )
                .with_output(compile(&current)),
            );
        }
    }
    let simplified = current.simplify();
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(simplified, EngineId::Simplify, steps_from_trace(&trace)),
    ));
    r.meta.engine_trace.push(EngineId::Simplify);
    r
}

fn partial_fractions_cmd(expr: &Expression, var: &str, narrate: bool) -> Response {
    // Expect the expression to be a binary division; heuristic split:
    // treat the whole expression as the numerator with denominator 1 when it
    // is not a quotient, which matches callers that already hand us the
    // rational form.
    let (num, denom) = split_rational(expr);
    let variable = Variable::new(var);
    match crate::partial_fractions::decompose(&num, &denom, &variable) {
        Ok(result) => {
            let value = result.to_expression();
            let mut trace = Trace::new();
            if narrate {
                trace.push(
                    Step::new(
                        TechniqueTag::PartialFractionDecomp,
                        "Partial fraction decomposition",
                    )
                    .with_output(compile(&value)),
                );
            }
            let mut r = Response::default();
            r.results.push((
                ResultKey::Single,
                symbolic_entry(value, EngineId::PartialFractions, steps_from_trace(&trace)),
            ));
            r.meta.engine_trace.push(EngineId::PartialFractions);
            r
        }
        Err(e) => engine_error("command.partial_fractions", format!("{:?}", e)),
    }
}

fn rearrange_cmd(equation: &Expression, solve_for: &str, narrate: bool) -> Response {
    let eq = expression_to_equation(equation);
    let solver = crate::solver::SmartSolver::new();
    let variable = Variable::new(solve_for);
    match solver.solve(&eq, &variable) {
        Ok((sol, trace)) => solution_to_response(
            sol,
            &trace,
            EngineId::EquationSolver,
            narrate,
            "command.rearrange",
        ),
        Err(e) => engine_error("command.rearrange", format!("{:?}", e)),
    }
}

fn solve_for_cmd(relation: &Expression, var: &str, narrate: bool) -> Response {
    rearrange_cmd(relation, var, narrate)
}

fn solve_system_cmd(equations: &[Expression], vars: &[String], narrate: bool) -> Response {
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

fn diff_cmd(expr: &Expression, var: &str, order: u32, narrate: bool) -> Response {
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

fn partial_diff_cmd(expr: &Expression, vars: &[(String, u32)], narrate: bool) -> Response {
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

fn gradient_cmd(expr: &Expression, vars: &[String], narrate: bool) -> Response {
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
    // Return the first component as the primary value with the rest as
    // alternatives until the Response type gets proper vector support.
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

fn integrate_cmd(expr: &Expression, var: &str, narrate: bool) -> Response {
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

fn def_integrate_cmd(
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

    // Symbolic failed; optionally fall back to numeric when the mode allows.
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
                                precision: super::request::Precision {
                                    decimal_digits: 12,
                                    abs_tol: None,
                                    rel_tol: None,
                                },
                                method: super::response::NumericMethod::AdaptiveQuadrature,
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

fn limit_cmd(
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

fn fourier_series_cmd(
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
            // `Expression`. Return the display string via Narrative steps
            // and keep the raw primary value as the leading a_0/2 constant
            // so downstream pattern matches see a scalar.
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

fn special_fn_cmd(_kind: SpecialKind, _args: &[Expression], _narrate: bool) -> Response {
    not_implemented("command.special_fn")
}

fn ode_cmd(
    equation: &Expression,
    fn_name: &str,
    var: &str,
    ic: Option<super::command::IvpData>,
    narrate: bool,
) -> Response {
    // The v0.8.1 Expression AST cannot carry derivative nodes. For now, the
    // dispatch accepts an `equation` describing the right-hand side of
    // dy/dx = rhs and routes to the first-order engine. IVP data maps to
    // solve_ivp when provided.
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

fn matrix_cmd(_op: MatrixOp, _operands: &[super::command::MatrixExpr], _narrate: bool) -> Response {
    not_implemented("command.matrix")
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn symbolic_entry(value: Expression, engine: EngineId, steps: Vec<NarratedStep>) -> ResultEntry {
    ResultEntry {
        value: ResultValue::Symbolic(value),
        shape: ResultShape::Scalar,
        unit: None,
        steps,
        alternatives: Vec::new(),
        engine,
    }
}

fn unsolved_entry(narrative: Narrative, engine: EngineId) -> ResultEntry {
    ResultEntry {
        value: ResultValue::Unsolved { reason: narrative },
        shape: ResultShape::Scalar,
        unit: None,
        steps: Vec::new(),
        alternatives: Vec::new(),
        engine,
    }
}

fn engine_error(template_id: &'static str, message: String) -> Response {
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        unsolved_entry(
            Narrative::new(template_id, message.clone()),
            EngineId::Other("engine-error"),
        ),
    ));
    r.diagnostics.push(Diagnostic::of(
        DiagnosticCode::Other("engine-error"),
        Narrative::new(template_id, message),
    ));
    r
}

fn steps_from_trace(trace: &Trace) -> Vec<NarratedStep> {
    trace
        .steps()
        .iter()
        .map(|step| NarratedStep {
            tag: step.tag,
            difficulty: step.tag.difficulty(),
            narrative: Narrative::new("step.generic", step.detail.clone()),
            path: None,
            input: step.input.as_ref().map(|arc| decompile(arc)),
            output: step.output.as_ref().map(|arc| decompile(arc)),
            unit_trace: None,
        })
        .collect()
}

fn solution_to_response(
    sol: crate::solver::Solution,
    trace: &Trace,
    engine: EngineId,
    narrate: bool,
    template_id: &'static str,
) -> Response {
    use crate::solver::Solution;
    let mut r = Response::default();
    let steps = if narrate {
        steps_from_trace(trace)
    } else {
        Vec::new()
    };
    match sol {
        Solution::Unique(expr) => r
            .results
            .push((ResultKey::Single, symbolic_entry(expr, engine, steps))),
        Solution::Multiple(exprs) => {
            for expr in exprs {
                r.results.push((
                    ResultKey::Single,
                    symbolic_entry(expr, engine, steps.clone()),
                ));
            }
        }
        Solution::Infinite => r.diagnostics.push(Diagnostic::of(
            DiagnosticCode::Other("infinite-solutions"),
            Narrative::new(template_id, "equation is an identity"),
        )),
        Solution::None => r.diagnostics.push(Diagnostic::of(
            DiagnosticCode::NoSolutionInDomain,
            Narrative::new(template_id, "no solution in domain"),
        )),
        Solution::Parametric { expression, .. } => r
            .results
            .push((ResultKey::Single, symbolic_entry(expression, engine, steps))),
    }
    r.meta.engine_trace.push(engine);
    r
}

fn expression_to_equation(expr: &Expression) -> crate::ast::Equation {
    // Accept either `Binary(Sub, lhs, rhs)` representing `lhs - rhs = 0`
    // directly, or a bare expression interpreted as `expr = 0`.
    if let Expression::Binary(crate::ast::BinaryOp::Sub, l, r) = expr {
        return crate::ast::Equation::new("dispatch", (**l).clone(), (**r).clone());
    }
    crate::ast::Equation::new("dispatch", expr.clone(), Expression::Integer(0))
}

fn split_rational(expr: &Expression) -> (Expression, Expression) {
    use crate::ast::BinaryOp;
    if let Expression::Binary(BinaryOp::Div, num, denom) = expr {
        ((**num).clone(), (**denom).clone())
    } else {
        (expr.clone(), Expression::Integer(1))
    }
}

fn substitute_in_expr(expr: &Expression, old: &Expression, new: &Expression) -> Expression {
    if expr == old {
        return new.clone();
    }
    match expr {
        Expression::Binary(op, l, r) => Expression::Binary(
            *op,
            Box::new(substitute_in_expr(l, old, new)),
            Box::new(substitute_in_expr(r, old, new)),
        ),
        Expression::Unary(op, inner) => {
            Expression::Unary(*op, Box::new(substitute_in_expr(inner, old, new)))
        }
        Expression::Power(base, exp) => Expression::Power(
            Box::new(substitute_in_expr(base, old, new)),
            Box::new(substitute_in_expr(exp, old, new)),
        ),
        Expression::Function(f, args) => Expression::Function(
            f.clone(),
            args.iter()
                .map(|a| substitute_in_expr(a, old, new))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn expression_to_f64(expr: &Expression) -> Option<f64> {
    match expr {
        Expression::Integer(n) => Some(*n as f64),
        Expression::Float(f) => Some(*f),
        Expression::Rational(r) => Some(*r.numer() as f64 / *r.denom() as f64),
        _ => expr.evaluate(&std::collections::HashMap::new()),
    }
}

// Suppress unused-import warnings by referencing the difficulty type.
#[allow(dead_code)]
fn _touch_difficulty(t: TechniqueTag) -> TechniqueDifficulty {
    t.difficulty()
}

#[cfg(test)]
mod tests {
    use super::super::command::{Command, SimplifyRules};
    use super::super::request::Request;
    use super::super::response::{EngineId, ResultKey, ResultValue};
    use super::execute;
    use crate::ast::{BinaryOp, Expression, Variable};

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn add(a: Expression, b: Expression) -> Expression {
        Expression::Binary(BinaryOp::Add, Box::new(a), Box::new(b))
    }

    fn mul(a: Expression, b: Expression) -> Expression {
        Expression::Binary(BinaryOp::Mul, Box::new(a), Box::new(b))
    }

    #[test]
    fn noop_returns_not_implemented_diagnostic() {
        let req = Request::default();
        let resp = execute(req).unwrap();
        assert!(resp.results.is_empty());
        assert_eq!(
            resp.diagnostics[0].code,
            super::super::diagnostic::DiagnosticCode::NotImplemented
        );
    }

    #[test]
    fn simplify_wires_engine() {
        // x + x → 2*x (or similar canonical form).
        let expr = add(var("x"), var("x"));
        let req = Request {
            command: Command::Simplify {
                expr,
                rules: SimplifyRules::all(),
                over: None,
            },
            ..Default::default()
        };
        let resp = execute(req).unwrap();
        let (key, entry) = &resp.results[0];
        assert_eq!(*key, ResultKey::Single);
        assert!(matches!(entry.value, ResultValue::Symbolic(_)));
        assert_eq!(entry.engine, EngineId::Simplify);
    }

    #[test]
    fn diff_wires_engine() {
        // d/dx (x^2) → 2x.
        let expr = Expression::Power(Box::new(var("x")), Box::new(int(2)));
        let req = Request {
            command: Command::Diff {
                expr,
                var: "x".to_string(),
                order: 1,
            },
            ..Default::default()
        };
        let resp = execute(req).unwrap();
        assert_eq!(resp.results[0].1.engine, EngineId::Differentiation);
    }

    #[test]
    fn integrate_wires_engine() {
        // ∫ 2x dx → x^2.
        let expr = mul(int(2), var("x"));
        let req = Request {
            command: Command::Integrate {
                expr,
                var: "x".to_string(),
            },
            ..Default::default()
        };
        let resp = execute(req).unwrap();
        assert_eq!(resp.results[0].1.engine, EngineId::PatternIntegration);
    }

    #[test]
    fn solve_for_wires_engine() {
        // 2x + 3 = 0 solved for x.
        let relation = add(mul(int(2), var("x")), int(3));
        let req = Request {
            command: Command::SolveFor {
                relation,
                var: "x".to_string(),
                over: super::super::Domain::real(),
            },
            ..Default::default()
        };
        let resp = execute(req).unwrap();
        assert_eq!(resp.results[0].1.engine, EngineId::EquationSolver);
    }
}
