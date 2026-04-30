//! Shared dispatch helpers.
//!
//! Helpers used by per-command-family submodules. All conversions between
//! `Expression` and `Arc<Expr>` happen at this seam (architecture rule 2).

use crate::api::diagnostic::{Diagnostic, DiagnosticCode};
use crate::api::domain::Domain;
use crate::api::narrative::Narrative;
use crate::api::request::{Budget, Precision, Request, SolveMode};
use crate::api::response::{
    BranchEntry, EngineId, NarratedStep, Response, ResultEntry, ResultKey, ResultShape,
    ResultValue, StructuredResult,
};
use crate::ast::Expression;
use crate::numeric::compile::decompile;
use crate::numeric::trace::Trace;

/// Emit a [`DiagnosticCode::FieldIgnored`] warning on `response` for a
/// field that was provided in the request but had no effect on this command.
pub(super) fn warn_ignored_field(
    response: &mut Response,
    field_name: &'static str,
    template_id: &'static str,
) {
    response.diagnostics.push(Diagnostic::of(
        DiagnosticCode::FieldIgnored,
        Narrative::new(
            template_id,
            format!(
                "field `{}` was provided but is not yet implemented; it had no effect",
                field_name
            ),
        ),
    ));
}

/// Execution-wide context extracted from [`Request`] fields that the
/// current dispatcher arm may or may not honour.
///
/// Created once at the top of [`super::execute`] and passed down so that
/// every arm can emit [`DiagnosticCode::FieldIgnored`] diagnostics for
/// fields it does not consume, rather than silently dropping them.
pub(super) struct DispatchContext {
    pub narrate: bool,
    pub mode: SolveMode,
    pub precision: Option<Precision>,
    pub budget: Option<Budget>,
    pub ambient_domain: Option<Domain>,
}

impl DispatchContext {
    /// Extract the context fields from a [`Request`].
    pub fn from_request(request: &Request) -> Self {
        Self {
            narrate: request.narrate,
            mode: request.mode,
            precision: request.precision.clone(),
            budget: request.budget,
            ambient_domain: request.ambient_domain,
        }
    }

    /// Emit [`DiagnosticCode::FieldIgnored`] warnings for every non-default
    /// context field that this command does not honour.
    ///
    /// `honors_mode` — pass `true` only for commands that genuinely
    /// consume [`Request::mode`] (currently only `DefIntegrate`).
    pub fn warn_unhandled(&self, response: &mut Response, honors_mode: bool) {
        if !honors_mode && self.mode != SolveMode::Symbolic {
            warn_ignored_field(response, "mode", "request.field_ignored");
        }
        if self.precision.is_some() {
            warn_ignored_field(response, "precision", "request.field_ignored");
        }
        if self.budget.is_some() {
            warn_ignored_field(response, "budget", "request.field_ignored");
        }
        if self.ambient_domain.is_some() {
            warn_ignored_field(response, "ambient_domain", "request.field_ignored");
        }
    }
}

pub(super) fn symbolic_entry(
    value: Expression,
    engine: EngineId,
    steps: Vec<NarratedStep>,
) -> ResultEntry {
    ResultEntry {
        value: ResultValue::Symbolic(value),
        structured: None,
        shape: ResultShape::Scalar,
        unit: None,
        steps,
        alternatives: Vec::new(),
        engine,
    }
}

pub(super) fn unsolved_entry(narrative: Narrative, engine: EngineId) -> ResultEntry {
    ResultEntry {
        value: ResultValue::Unsolved { reason: narrative },
        structured: None,
        shape: ResultShape::Scalar,
        unit: None,
        steps: Vec::new(),
        alternatives: Vec::new(),
        engine,
    }
}

pub(super) fn engine_error(template_id: &'static str, message: String) -> Response {
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

pub(super) fn steps_from_trace(trace: &Trace) -> Vec<NarratedStep> {
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

pub(super) fn solution_to_response(
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
            let branches: Vec<BranchEntry> = exprs
                .iter()
                .enumerate()
                .map(|(i, e)| BranchEntry {
                    condition: None,
                    label: Some(format!("root_{}", i + 1)),
                    value: e.clone(),
                })
                .collect();
            let structured = StructuredResult::Branches { branches };
            for (i, expr) in exprs.into_iter().enumerate() {
                let mut entry = symbolic_entry(expr.clone(), engine, steps.clone());
                // Only the first result entry carries the full Branches
                // structured data; subsequent entries carry None to avoid
                // redundant cloning for large solution sets.
                entry.structured = if i == 0 {
                    Some(structured.clone())
                } else {
                    None
                };
                r.results.push((ResultKey::Single, entry));
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

pub(super) fn expression_to_equation(expr: &Expression) -> crate::ast::Equation {
    if let Expression::Binary(crate::ast::BinaryOp::Sub, l, r) = expr {
        return crate::ast::Equation::new("dispatch", (**l).clone(), (**r).clone());
    }
    crate::ast::Equation::new("dispatch", expr.clone(), Expression::Integer(0))
}

pub(super) fn split_rational(expr: &Expression) -> (Expression, Expression) {
    use crate::ast::BinaryOp;
    if let Expression::Binary(BinaryOp::Div, num, denom) = expr {
        ((**num).clone(), (**denom).clone())
    } else {
        (expr.clone(), Expression::Integer(1))
    }
}

pub(super) fn substitute_in_expr(
    expr: &Expression,
    old: &Expression,
    new: &Expression,
) -> Expression {
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

pub(super) fn expression_to_f64(expr: &Expression) -> Option<f64> {
    match expr {
        Expression::Integer(n) => Some(*n as f64),
        Expression::Float(f) => Some(*f),
        Expression::Rational(r) => Some(*r.numer() as f64 / *r.denom() as f64),
        _ => expr.evaluate(&std::collections::HashMap::new()),
    }
}
