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
use crate::api::response::Response;
use crate::ThalesError;

use helpers::DispatchContext;

mod algebra;
mod calculus;
mod helpers;
mod limits;
mod matrix;
mod nabla;
mod ode;
mod optimize;
mod series;
mod series_expand;
mod solver;
mod special;

/// Execute a [`Request`]. Single entry point for thales.
pub fn execute(request: Request) -> Result<Response, ThalesError> {
    let start = Instant::now();
    let ctx = DispatchContext::from_request(&request);
    let narrate = ctx.narrate;

    // `honors_mode` is set to true only for commands that genuinely consume
    // request.mode (currently only DefIntegrate). All other arms will emit a
    // FieldIgnored diagnostic if mode is non-Symbolic.
    let mut honors_mode = false;

    let mut response = match request.command {
        Command::Noop => noop_response(),

        // ── Algebra ──────────────────────────────────────────────────────
        Command::Simplify { expr, rules, over } => {
            let mut r = algebra::simplify_cmd(&expr, rules, narrate);
            if over.is_some() {
                helpers::warn_ignored_field(&mut r, "Simplify.over", "request.field_ignored");
            }
            r
        }
        Command::Expand { expr, target } => {
            let mut r = algebra::expand_cmd(&expr, narrate);
            if target.is_some() {
                helpers::warn_ignored_field(&mut r, "Expand.target", "request.field_ignored");
            }
            r
        }
        Command::Factor { expr, over, target } => {
            let mut r = algebra::factor_cmd(&expr, over, narrate);
            if target.is_some() {
                helpers::warn_ignored_field(&mut r, "Factor.target", "request.field_ignored");
            }
            r
        }
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
        Command::SolveFor {
            relation,
            var,
            over,
        } => solver::solve_for_cmd(&relation, &var, over, narrate),
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
        } => {
            honors_mode = true;
            calculus::def_integrate_cmd(&expr, &var, &from, &to, narrate, ctx.mode)
        }

        // ── Limits ───────────────────────────────────────────────────────
        Command::Limit {
            expr,
            var,
            point,
            side,
        } => limits::limit_cmd(&expr, &var, point, side, narrate),

        // ── Expansions ───────────────────────────────────────────────────
        Command::Taylor {
            expr,
            var,
            center,
            order,
        } => series_expand::taylor_cmd(&expr, &var, &center, order, narrate),
        Command::Laurent {
            expr,
            var,
            center,
            order,
        } => series_expand::laurent_cmd(&expr, &var, &center, order, narrate),
        Command::Asymptotic { expr, var, order } => {
            series_expand::asymptotic_cmd(&expr, &var, order, narrate)
        }
        Command::Compose {
            outer,
            inner,
            var,
            order,
        } => series_expand::compose_cmd(&outer, &inner, &var, order, narrate),
        Command::Revert { expr, var, order } => {
            series_expand::revert_cmd(&expr, &var, order, narrate)
        }
        Command::Puiseux {
            expr,
            var,
            center,
            order,
        } => series_expand::puiseux_cmd(&expr, &var, &center, order, narrate),
        Command::Frobenius {
            ode,
            fn_name,
            var,
            point,
            order,
        } => series_expand::frobenius_cmd(&ode, &fn_name, &var, &point, order, narrate),
        Command::Pade {
            expr,
            var,
            center,
            m,
            n,
        } => series_expand::pade_cmd(&expr, &var, &center, m, n, narrate),
        Command::Wkb {
            ode,
            fn_name,
            var,
            small_param,
            order,
        } => series_expand::wkb_cmd(&ode, &fn_name, &var, &small_param, order, narrate),

        // ── Transforms ───────────────────────────────────────────────────
        Command::FourierSeries {
            expr,
            var,
            period,
            terms,
        } => series::fourier_series_cmd(&expr, &var, &period, terms, narrate),
        Command::Residue { expr, var, point } => special::residue_cmd(&expr, &var, &point, narrate),

        // ── Special functions ────────────────────────────────────────────
        Command::SpecialFn { kind, args } => special::special_fn_cmd(kind, &args, narrate),

        // ── ODE ─────────────────────────────────────────────────────────
        Command::Ode {
            equation,
            fn_name,
            var,
            ic,
        } => ode::ode_cmd(&equation, &fn_name, &var, ic, narrate),

        Command::OdeSystem { .. } => {
            let mut r = noop_response();
            r.diagnostics.push(Diagnostic::of(
                DiagnosticCode::NotImplemented,
                Narrative::new(
                    "command.ode_system",
                    "ODE system solver not yet implemented (v0.9.0).",
                ),
            ));
            r
        }

        Command::Pde { .. } => {
            let mut r = noop_response();
            r.diagnostics.push(Diagnostic::of(
                DiagnosticCode::NotImplemented,
                Narrative::new("command.pde", "PDE solving not yet supported (v0.12.0)."),
            ));
            r
        }

        // ── Matrix ───────────────────────────────────────────────────────
        Command::Matrix { op, operands } => matrix::matrix_cmd(op, &operands, narrate),

        // ── Nabla ────────────────────────────────────────────────────────
        Command::Nabla { op, input, vars } => nabla::nabla_cmd(op, input, &vars, narrate),

        // ── Optimization ─────────────────────────────────────────────────
        Command::Optimize {
            objective,
            vars,
            constraints,
            sense,
        } => optimize::optimize_cmd(&objective, &vars, &constraints, sense, narrate),
        Command::LagrangeMult {
            objective,
            vars,
            equality_constraints,
        } => optimize::lagrange_mult_cmd(&objective, &vars, &equality_constraints, narrate),
    };

    // Emit FieldIgnored warnings for request-level context fields that the
    // dispatched command does not honour.
    ctx.warn_unhandled(&mut response, honors_mode);

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
