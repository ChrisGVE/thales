//! Single-entry dispatcher for [`super::execute`].
//!
//! Matches on [`Command`] and delegates to per-family submodules
//! (algebra, calculus, limits, solver, ode, series). Inputs are compiled to
//! canonical `Arc<Expr>` form at each dispatch arm when engines require it;
//! results are decompiled back to `Expression` at this boundary
//! (architecture rule 2).
//!
//! # v0.8.1 coverage
//!
//! Commands whose engines are wired: algebra (Simplify/Expand/Factor/
//! Substitute/PartialFractions/Rearrange and the like-term/common-denom
//! aliases), calculus (Diff/PartialDiff/Gradient/Integrate/DefIntegrate),
//! Limit, SolveFor/SolveSystem, Ode, FourierSeries.
//!
//! Commands without engines yet (higher-dim calculus, advanced series,
//! transforms, optimization, special functions, matrix) return a
//! [`DiagnosticCode::NotImplemented`] entry and an empty [`ResultEntry`]
//! carrying an [`ResultValue::Unsolved`] reason.

use std::time::Instant;

use crate::api::command::{Command, SimplifyRules};
use crate::api::diagnostic::{Diagnostic, DiagnosticCode};
use crate::api::narrative::Narrative;
use crate::api::request::Request;
use crate::api::response::{EngineId, Response, ResultKey};
use crate::ThalesError;

mod algebra;
mod calculus;
mod helpers;
mod limits;
mod ode;
mod series;
mod solver;

use helpers::unsolved_entry;

/// Execute a [`Request`]. Single entry point for thales.
pub fn execute(request: Request) -> Result<Response, ThalesError> {
    let start = Instant::now();
    let narrate = request.narrate;

    let mut response = match request.command {
        Command::Noop => noop_response(),

        // ── Algebra ──────────────────────────────────────────────────────
        Command::Simplify { expr, rules, .. } => algebra::simplify_cmd(&expr, rules, narrate),
        Command::Expand { expr, .. } => algebra::expand_cmd(&expr, narrate),
        Command::Factor { expr, .. } => algebra::factor_cmd(&expr, narrate),
        Command::Substitute { expr, bindings, .. } => {
            algebra::substitute_cmd(&expr, &bindings, narrate)
        }
        Command::CombineLikeTerms { expr, .. } => {
            algebra::simplify_cmd(&expr, SimplifyRules::all(), narrate)
        }
        Command::CommonDenominator { expr, .. } => {
            algebra::simplify_cmd(&expr, SimplifyRules::all(), narrate)
        }
        Command::PartialFractions { expr, var } => {
            algebra::partial_fractions_cmd(&expr, &var, narrate)
        }
        Command::Rationalize { expr, .. } => {
            algebra::simplify_cmd(&expr, SimplifyRules::all(), narrate)
        }
        Command::Conjugate { expr, .. } => algebra::conjugate_cmd(&expr, narrate),
        Command::InverseFn { expr, var } => algebra::inverse_fn_cmd(&expr, &var, narrate),
        Command::Rearrange {
            equation,
            solve_for,
        } => algebra::rearrange_cmd(&equation, &solve_for, narrate),
        Command::ApplyIdentity { expr, identity, .. } => {
            algebra::apply_identity_cmd(&expr, &identity, narrate)
        }

        // ── Solve ────────────────────────────────────────────────────────
        Command::SolveFor { relation, var, .. } => solver::solve_for_cmd(&relation, &var, narrate),
        Command::SolveSystem {
            equations, vars, ..
        } => solver::solve_system_cmd(&equations, &vars, narrate),

        // ── Differentiation ──────────────────────────────────────────────
        Command::Diff { expr, var, order } => calculus::diff_cmd(&expr, &var, order, narrate),
        Command::PartialDiff { expr, vars } => calculus::partial_diff_cmd(&expr, &vars, narrate),
        Command::TotalDiff { expr, var, deps } => {
            calculus::total_diff_cmd(&expr, &var, &deps, narrate)
        }
        Command::Gradient { expr, vars } => calculus::gradient_cmd(&expr, &vars, narrate),
        Command::Divergence { field, vars } => calculus::divergence_cmd(&field, &vars, narrate),
        Command::Curl { field, vars } => calculus::curl_cmd(&field, &vars, narrate),
        Command::Laplacian { expr, vars } => calculus::laplacian_cmd(&expr, &vars, narrate),
        Command::Jacobian { fields, vars } => calculus::jacobian_cmd(&fields, &vars, narrate),
        Command::Hessian { expr, vars } => calculus::hessian_cmd(&expr, &vars, narrate),
        Command::DirectionalDiff {
            expr,
            vars,
            direction,
        } => calculus::directional_diff_cmd(&expr, &vars, &direction, narrate),

        // ── Integration ──────────────────────────────────────────────────
        Command::Integrate { expr, var } => calculus::integrate_cmd(&expr, &var, narrate),
        Command::DefIntegrate {
            expr,
            var,
            from,
            to,
        } => calculus::def_integrate_cmd(&expr, &var, &from, &to, narrate, request.mode),

        // ── Limits ───────────────────────────────────────────────────────
        Command::Limit {
            expr,
            var,
            point,
            side,
        } => limits::limit_cmd(&expr, &var, point, side, narrate),

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
        } => series::fourier_series_cmd(&expr, &var, &period, terms, narrate),
        Command::Residue { .. } => not_implemented("command.residue"),

        // ── Special functions ────────────────────────────────────────────
        Command::SpecialFn { .. } => not_implemented("command.special_fn"),

        // ── ODE ─────────────────────────────────────────────────────────
        Command::Ode {
            equation,
            fn_name,
            var,
            ic,
        } => ode::ode_cmd(&equation, &fn_name, &var, ic, narrate),

        // ── Matrix ───────────────────────────────────────────────────────
        Command::Matrix { .. } => not_implemented("command.matrix"),

        // ── Optimization ─────────────────────────────────────────────────
        Command::Optimize { .. } => not_implemented("command.optimize"),
        Command::LagrangeMult { .. } => not_implemented("command.lagrange_mult"),
    };

    response.meta.elapsed_ms = start.elapsed().as_millis() as u64;
    Ok(crate::api::render::render_response(response))
}

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

#[cfg(test)]
mod tests {
    use super::execute;
    use crate::api::command::{Command, SimplifyRules};
    use crate::api::request::Request;
    use crate::api::response::{EngineId, ResultKey, ResultValue};
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
            crate::api::diagnostic::DiagnosticCode::NotImplemented
        );
    }

    #[test]
    fn simplify_wires_engine() {
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
        let relation = add(mul(int(2), var("x")), int(3));
        let req = Request {
            command: Command::SolveFor {
                relation,
                var: "x".to_string(),
                over: crate::api::Domain::real(),
            },
            ..Default::default()
        };
        let resp = execute(req).unwrap();
        assert_eq!(resp.results[0].1.engine, EngineId::EquationSolver);
    }
}
