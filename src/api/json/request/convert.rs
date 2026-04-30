//! Conversion from JSON mirror types to internal [`Command`] / [`Request`].

use serde_json::Value;

use super::super::super::command::{Command, Constraint, IvpData, LimitPoint, SimplifyRules};
use super::super::super::domain::Domain;
use super::super::super::request::{Budget, Precision, Request};
use super::parsers::{
    parse_domain_str, parse_expr_str, parse_identity_id, parse_matrix_expr, parse_matrix_op,
    parse_opt_sense, parse_side, parse_solve_mode, parse_special_kind,
};
use super::schema::{
    JsonBudget, JsonCommand, JsonIvpData, JsonPrecision, JsonRequest, JsonSimplifyRules,
};

// ── Public entry point ────────────────────────────────────────────────────────

pub(in super::super) fn request_from_json(val: &Value) -> Result<Request, String> {
    let json_req: JsonRequest =
        serde_json::from_value(val.clone()).map_err(|e| format!("invalid request: {}", e))?;

    let mode = json_req
        .mode
        .as_deref()
        .map(parse_solve_mode)
        .transpose()?
        .unwrap_or_default();

    let precision = json_req
        .precision
        .map(json_precision_to_precision)
        .transpose()?;
    let budget = json_req.budget.map(json_budget_to_budget);
    let ambient_domain = json_req
        .ambient_domain
        .as_deref()
        .map(parse_domain_str)
        .transpose()?;

    let command = json_command_to_command(json_req.command)?;

    Ok(Request {
        command,
        narrate: json_req.narrate,
        mode,
        precision,
        output_units: None,
        ambient_domain,
        budget,
        seed: json_req.seed,
    })
}

// ── JsonCommand → Command ─────────────────────────────────────────────────────

fn json_command_to_command(cmd: JsonCommand) -> Result<Command, String> {
    match cmd {
        JsonCommand::Noop => Ok(Command::Noop),

        JsonCommand::Simplify { expr, rules, over } => Ok(Command::Simplify {
            expr: parse_expr_str(&expr)?,
            rules: rules
                .map(json_rules_to_rules)
                .unwrap_or_else(SimplifyRules::all),
            over: over.as_deref().map(parse_domain_str).transpose()?,
        }),

        JsonCommand::Expand { expr, .. } => Ok(Command::Expand {
            expr: parse_expr_str(&expr)?,
            target: None,
        }),

        JsonCommand::Factor { expr, over, .. } => Ok(Command::Factor {
            expr: parse_expr_str(&expr)?,
            over: over
                .as_deref()
                .map(parse_domain_str)
                .transpose()?
                .unwrap_or_else(Domain::real),
            target: None,
        }),

        JsonCommand::Substitute { expr, bindings } => {
            let parsed_bindings = bindings
                .into_iter()
                .map(|b| Ok((parse_expr_str(&b.old)?, parse_expr_str(&b.new)?)))
                .collect::<Result<Vec<_>, String>>()?;
            Ok(Command::Substitute {
                expr: parse_expr_str(&expr)?,
                bindings: parsed_bindings,
                target: None,
            })
        }

        JsonCommand::CombineLikeTerms { expr, .. } => Ok(Command::CombineLikeTerms {
            expr: parse_expr_str(&expr)?,
            target: None,
        }),

        JsonCommand::CommonDenominator { expr, .. } => Ok(Command::CommonDenominator {
            expr: parse_expr_str(&expr)?,
            target: None,
        }),

        JsonCommand::PartialFractions { expr, var } => Ok(Command::PartialFractions {
            expr: parse_expr_str(&expr)?,
            var,
        }),

        JsonCommand::Rationalize { expr, .. } => Ok(Command::Rationalize {
            expr: parse_expr_str(&expr)?,
            target: None,
        }),

        JsonCommand::Conjugate { expr, .. } => Ok(Command::Conjugate {
            expr: parse_expr_str(&expr)?,
            target: None,
        }),

        JsonCommand::InverseFn { expr, var } => Ok(Command::InverseFn {
            expr: parse_expr_str(&expr)?,
            var,
        }),

        JsonCommand::Rearrange {
            equation,
            solve_for,
        } => Ok(Command::Rearrange {
            equation: parse_expr_str(&equation)?,
            solve_for,
        }),

        JsonCommand::ApplyIdentity { expr, identity, .. } => Ok(Command::ApplyIdentity {
            expr: parse_expr_str(&expr)?,
            identity: parse_identity_id(&identity)?,
            target: None,
        }),

        JsonCommand::SolveFor {
            relation,
            var,
            over,
        } => Ok(Command::SolveFor {
            relation: parse_expr_str(&relation)?,
            var,
            over: over
                .as_deref()
                .map(parse_domain_str)
                .transpose()?
                .unwrap_or_else(Domain::real),
        }),

        JsonCommand::SolveSystem {
            equations,
            vars,
            over,
        } => {
            let parsed = equations
                .iter()
                .map(|s| parse_expr_str(s))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Command::SolveSystem {
                equations: parsed,
                vars,
                over: over
                    .as_deref()
                    .map(parse_domain_str)
                    .transpose()?
                    .unwrap_or_else(Domain::real),
            })
        }

        JsonCommand::Diff { expr, var, order } => Ok(Command::Diff {
            expr: parse_expr_str(&expr)?,
            var,
            order: order.unwrap_or(1),
        }),

        JsonCommand::PartialDiff { expr, vars } => {
            let parsed_vars = vars
                .into_iter()
                .map(|v| Ok((v.var, v.order)))
                .collect::<Result<Vec<_>, String>>()?;
            Ok(Command::PartialDiff {
                expr: parse_expr_str(&expr)?,
                vars: parsed_vars,
            })
        }

        JsonCommand::TotalDiff { expr, var, deps } => {
            let parsed_deps = deps
                .into_iter()
                .map(|d| Ok((d.name, parse_expr_str(&d.expr)?)))
                .collect::<Result<Vec<_>, String>>()?;
            Ok(Command::TotalDiff {
                expr: parse_expr_str(&expr)?,
                var,
                deps: parsed_deps,
            })
        }

        JsonCommand::Gradient { expr, vars } => Ok(Command::Gradient {
            expr: parse_expr_str(&expr)?,
            vars,
        }),

        JsonCommand::Divergence { field, vars } => {
            let parsed = field
                .iter()
                .map(|s| parse_expr_str(s))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Command::Divergence {
                field: parsed,
                vars,
            })
        }

        JsonCommand::Curl { field, vars } => {
            let parsed = field
                .iter()
                .map(|s| parse_expr_str(s))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Command::Curl {
                field: parsed,
                vars,
            })
        }

        JsonCommand::Laplacian { expr, vars } => Ok(Command::Laplacian {
            expr: parse_expr_str(&expr)?,
            vars,
        }),

        JsonCommand::Jacobian { fields, vars } => {
            let parsed = fields
                .iter()
                .map(|s| parse_expr_str(s))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Command::Jacobian {
                fields: parsed,
                vars,
            })
        }

        JsonCommand::Hessian { expr, vars } => Ok(Command::Hessian {
            expr: parse_expr_str(&expr)?,
            vars,
        }),

        JsonCommand::DirectionalDiff {
            expr,
            vars,
            direction,
        } => {
            let parsed_dir = direction
                .iter()
                .map(|s| parse_expr_str(s))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Command::DirectionalDiff {
                expr: parse_expr_str(&expr)?,
                vars,
                direction: parsed_dir,
            })
        }

        JsonCommand::Integrate { expr, var } => Ok(Command::Integrate {
            expr: parse_expr_str(&expr)?,
            var,
        }),

        JsonCommand::DefIntegrate {
            expr,
            var,
            from,
            to,
        } => Ok(Command::DefIntegrate {
            expr: parse_expr_str(&expr)?,
            var,
            from: parse_expr_str(&from)?,
            to: parse_expr_str(&to)?,
        }),

        JsonCommand::Limit {
            expr,
            var,
            point,
            side,
        } => {
            let point = match point.as_str() {
                "+inf" => LimitPoint::PosInf,
                "-inf" => LimitPoint::NegInf,
                s => LimitPoint::Finite(parse_expr_str(s)?),
            };
            let side = side.as_deref().map(parse_side).transpose()?;
            Ok(Command::Limit {
                expr: parse_expr_str(&expr)?,
                var,
                point,
                side,
            })
        }

        JsonCommand::Taylor {
            expr,
            var,
            center,
            order,
        } => Ok(Command::Taylor {
            expr: parse_expr_str(&expr)?,
            var,
            center: parse_expr_str(&center)?,
            order: order.unwrap_or(3),
        }),

        JsonCommand::Laurent {
            expr,
            var,
            center,
            order,
        } => Ok(Command::Laurent {
            expr: parse_expr_str(&expr)?,
            var,
            center: parse_expr_str(&center)?,
            order: order.unwrap_or(3),
        }),

        JsonCommand::Asymptotic { expr, var, order } => Ok(Command::Asymptotic {
            expr: parse_expr_str(&expr)?,
            var,
            order: order.unwrap_or(3),
        }),

        JsonCommand::Compose {
            outer,
            inner,
            var,
            order,
        } => Ok(Command::Compose {
            outer: parse_expr_str(&outer)?,
            inner: parse_expr_str(&inner)?,
            var,
            order: order.unwrap_or(3),
        }),

        JsonCommand::Revert { expr, var, order } => Ok(Command::Revert {
            expr: parse_expr_str(&expr)?,
            var,
            order: order.unwrap_or(3),
        }),

        JsonCommand::Puiseux {
            expr,
            var,
            center,
            order,
        } => Ok(Command::Puiseux {
            expr: parse_expr_str(&expr)?,
            var,
            center: center
                .as_deref()
                .map(parse_expr_str)
                .transpose()?
                .unwrap_or(crate::ast::Expression::Integer(0)),
            order: order.unwrap_or(3),
        }),

        JsonCommand::Frobenius {
            ode,
            fn_name,
            var,
            point,
            order,
        } => Ok(Command::Frobenius {
            ode: parse_expr_str(&ode)?,
            fn_name,
            var,
            point: point
                .as_deref()
                .map(parse_expr_str)
                .transpose()?
                .unwrap_or(crate::ast::Expression::Integer(0)),
            order: order.unwrap_or(3),
        }),

        JsonCommand::Pade {
            expr,
            var,
            center,
            m,
            n,
        } => Ok(Command::Pade {
            expr: parse_expr_str(&expr)?,
            var,
            center: center
                .as_deref()
                .map(parse_expr_str)
                .transpose()?
                .unwrap_or(crate::ast::Expression::Integer(0)),
            m,
            n,
        }),

        JsonCommand::Wkb {
            ode,
            fn_name,
            var,
            small_param,
            order,
        } => Ok(Command::Wkb {
            ode: parse_expr_str(&ode)?,
            fn_name,
            var,
            small_param,
            order: order.unwrap_or(2),
        }),

        JsonCommand::FourierSeries {
            expr,
            var,
            period,
            terms,
        } => Ok(Command::FourierSeries {
            expr: parse_expr_str(&expr)?,
            var,
            period: parse_expr_str(&period)?,
            terms: terms.unwrap_or(3),
        }),

        JsonCommand::Residue { expr, var, point } => Ok(Command::Residue {
            expr: parse_expr_str(&expr)?,
            var,
            point: parse_expr_str(&point)?,
        }),

        JsonCommand::SpecialFn { kind, args } => {
            let parsed_args = args
                .iter()
                .map(|s| parse_expr_str(s))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Command::SpecialFn {
                kind: parse_special_kind(&kind)?,
                args: parsed_args,
            })
        }

        JsonCommand::Ode {
            equation,
            fn_name,
            var,
            ic,
        } => {
            let ic = ic.map(json_ivp_to_ivp).transpose()?;
            Ok(Command::Ode {
                equation: parse_expr_str(&equation)?,
                fn_name,
                var,
                ic,
            })
        }

        JsonCommand::Matrix { op, operands } => {
            let matrix_op = parse_matrix_op(&op)?;
            let parsed_operands = operands
                .unwrap_or_default()
                .iter()
                .enumerate()
                .map(|(i, v)| parse_matrix_expr(v, i))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Command::Matrix {
                op: matrix_op,
                operands: parsed_operands,
            })
        }

        JsonCommand::Optimize {
            objective,
            vars,
            constraints,
            sense,
        } => {
            let parsed_constraints = constraints
                .unwrap_or_default()
                .into_iter()
                .map(|c| {
                    let expr = parse_expr_str(&c.expr)?;
                    match c.kind.as_str() {
                        "Equality" => Ok(Constraint::Equality(expr)),
                        "LessEq" => Ok(Constraint::LessEq(expr)),
                        "Less" => Ok(Constraint::Less(expr)),
                        "GreaterEq" => Ok(Constraint::GreaterEq(expr)),
                        "Greater" => Ok(Constraint::Greater(expr)),
                        other => Err(format!("unknown constraint kind `{}`", other)),
                    }
                })
                .collect::<Result<Vec<_>, String>>()?;
            let parsed_sense = sense
                .as_deref()
                .map(parse_opt_sense)
                .transpose()?
                .unwrap_or_default();
            Ok(Command::Optimize {
                objective: parse_expr_str(&objective)?,
                vars,
                constraints: parsed_constraints,
                sense: parsed_sense,
            })
        }

        JsonCommand::LagrangeMult {
            objective,
            vars,
            equality_constraints,
        } => {
            let parsed_constraints = equality_constraints
                .iter()
                .map(|s| parse_expr_str(s))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Command::LagrangeMult {
                objective: parse_expr_str(&objective)?,
                vars,
                equality_constraints: parsed_constraints,
            })
        }
    }
}

// ── Small conversions ─────────────────────────────────────────────────────────

fn json_ivp_to_ivp(j: JsonIvpData) -> Result<IvpData, String> {
    let derivatives_at = j
        .derivatives_at
        .iter()
        .enumerate()
        .map(|(i, s)| parse_expr_str(s).map_err(|e| format!("derivatives_at[{}]: {}", i, e)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(IvpData {
        var_at: parse_expr_str(&j.var_at)?,
        fn_at: parse_expr_str(&j.fn_at)?,
        derivatives_at,
    })
}

fn json_precision_to_precision(j: JsonPrecision) -> Result<Precision, String> {
    let abs_tol = j.abs_tol.as_deref().map(parse_expr_str).transpose()?;
    let rel_tol = j.rel_tol.as_deref().map(parse_expr_str).transpose()?;
    Ok(Precision {
        decimal_digits: j.decimal_digits,
        abs_tol,
        rel_tol,
    })
}

fn json_budget_to_budget(j: JsonBudget) -> Budget {
    Budget {
        max_wall_ms: j.max_wall_ms,
        max_iterations: j.max_iterations,
    }
}

fn json_rules_to_rules(j: JsonSimplifyRules) -> SimplifyRules {
    SimplifyRules {
        arithmetic: j.arithmetic,
        algebraic: j.algebraic,
        trigonometric: j.trigonometric,
        logarithmic: j.logarithmic,
        exponential: j.exponential,
        hyperbolic: j.hyperbolic,
        rational: j.rational,
    }
}
