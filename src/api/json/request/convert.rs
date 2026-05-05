//! Conversion from JSON mirror types to internal [`Command`] / [`Request`].

use crate::ast::Expression;
use crate::mathlex_bridge::convert_expression;
use crate::transforms::CoordSystem;

use super::super::super::command::{
    Command, Constraint, IntegrationStep, IvpData, LimitPoint, MatrixExpr, NablaInput, NablaOp,
    ParamCurve, SimplifyRules, SystemIvpData,
};
use super::super::super::domain::Domain;
use super::super::super::request::{Budget, Precision, Request};
use super::parsers::{
    parse_domain_str, parse_identity_id, parse_matrix_op, parse_opt_sense, parse_side,
    parse_solve_mode, parse_special_kind,
};
use super::schema::{
    JsonBudget, JsonCommand, JsonIntegrationStep, JsonIvpData, JsonMatrixOperand, JsonNablaInput,
    JsonParamCurve, JsonPrecision, JsonRequest, JsonSimplifyRules, JsonSystemIvpData,
};

// ── Helper ────────────────────────────────────────────────────────────────────

/// Convert a mathlex `Expression` into a thales internal `Expression`.
fn cvt(e: mathlex::Expression) -> Result<Expression, String> {
    convert_expression(&e)
}

// ── Public entry point ────────────────────────────────────────────────────────

pub(in super::super) fn request_from_json(val: &serde_json::Value) -> Result<Request, String> {
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
            expr: cvt(expr)?,
            rules: rules
                .map(json_rules_to_rules)
                .unwrap_or_else(SimplifyRules::all),
            over: over.as_deref().map(parse_domain_str).transpose()?,
        }),

        JsonCommand::Expand { expr, .. } => Ok(Command::Expand {
            expr: cvt(expr)?,
            target: None,
        }),

        JsonCommand::Factor { expr, over, .. } => Ok(Command::Factor {
            expr: cvt(expr)?,
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
                .map(|b| Ok((cvt(b.old)?, cvt(b.new)?)))
                .collect::<Result<Vec<_>, String>>()?;
            Ok(Command::Substitute {
                expr: cvt(expr)?,
                bindings: parsed_bindings,
                target: None,
            })
        }

        JsonCommand::CombineLikeTerms { expr, .. } => Ok(Command::CombineLikeTerms {
            expr: cvt(expr)?,
            target: None,
        }),

        JsonCommand::CommonDenominator { expr, .. } => Ok(Command::CommonDenominator {
            expr: cvt(expr)?,
            target: None,
        }),

        JsonCommand::PartialFractions { expr, var } => Ok(Command::PartialFractions {
            expr: cvt(expr)?,
            var,
        }),

        JsonCommand::Rationalize { expr, .. } => Ok(Command::Rationalize {
            expr: cvt(expr)?,
            target: None,
        }),

        JsonCommand::Conjugate { expr, .. } => Ok(Command::Conjugate {
            expr: cvt(expr)?,
            target: None,
        }),

        JsonCommand::InverseFn { expr, var } => Ok(Command::InverseFn {
            expr: cvt(expr)?,
            var,
        }),

        JsonCommand::Rearrange {
            equation,
            solve_for,
        } => Ok(Command::Rearrange {
            equation: cvt(equation)?,
            solve_for,
        }),

        JsonCommand::ApplyIdentity { expr, identity, .. } => Ok(Command::ApplyIdentity {
            expr: cvt(expr)?,
            identity: parse_identity_id(&identity)?,
            target: None,
        }),

        JsonCommand::SolveFor {
            relation,
            var,
            over,
        } => Ok(Command::SolveFor {
            relation: cvt(relation)?,
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
                .into_iter()
                .map(cvt)
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
            expr: cvt(expr)?,
            var,
            order: order.unwrap_or(1),
        }),

        JsonCommand::PartialDiff { expr, vars } => {
            let parsed_vars = vars
                .into_iter()
                .map(|v| Ok((v.var, v.order)))
                .collect::<Result<Vec<_>, String>>()?;
            Ok(Command::PartialDiff {
                expr: cvt(expr)?,
                vars: parsed_vars,
            })
        }

        JsonCommand::TotalDiff { expr, var, deps } => {
            let parsed_deps = deps
                .into_iter()
                .map(|d| Ok((d.name, cvt(d.expr)?)))
                .collect::<Result<Vec<_>, String>>()?;
            Ok(Command::TotalDiff {
                expr: cvt(expr)?,
                var,
                deps: parsed_deps,
            })
        }

        JsonCommand::Gradient { expr, vars } => Ok(Command::Gradient {
            expr: cvt(expr)?,
            vars,
        }),

        JsonCommand::Divergence { field, vars } => {
            let parsed = field.into_iter().map(cvt).collect::<Result<Vec<_>, _>>()?;
            Ok(Command::Divergence {
                field: parsed,
                vars,
            })
        }

        JsonCommand::Curl { field, vars } => {
            let parsed = field.into_iter().map(cvt).collect::<Result<Vec<_>, _>>()?;
            Ok(Command::Curl {
                field: parsed,
                vars,
            })
        }

        JsonCommand::Laplacian { expr, vars } => Ok(Command::Laplacian {
            expr: cvt(expr)?,
            vars,
        }),

        JsonCommand::Jacobian { fields, vars } => {
            let parsed = fields.into_iter().map(cvt).collect::<Result<Vec<_>, _>>()?;
            Ok(Command::Jacobian {
                fields: parsed,
                vars,
            })
        }

        JsonCommand::Hessian { expr, vars } => Ok(Command::Hessian {
            expr: cvt(expr)?,
            vars,
        }),

        JsonCommand::DirectionalDiff {
            expr,
            vars,
            direction,
        } => {
            let parsed_dir = direction
                .into_iter()
                .map(cvt)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Command::DirectionalDiff {
                expr: cvt(expr)?,
                vars,
                direction: parsed_dir,
            })
        }

        JsonCommand::Integrate { expr, var } => Ok(Command::Integrate {
            expr: cvt(expr)?,
            var,
        }),

        JsonCommand::DefIntegrate {
            expr,
            var,
            from,
            to,
        } => Ok(Command::DefIntegrate {
            expr: cvt(expr)?,
            var,
            from: cvt(from)?,
            to: cvt(to)?,
        }),

        JsonCommand::MultiIntegrate { expr, integrations } => {
            let parsed_integrations = integrations
                .into_iter()
                .map(json_integration_step_to_step)
                .collect::<Result<Vec<_>, String>>()?;
            Ok(Command::MultiIntegrate {
                expr: cvt(expr)?,
                integrations: parsed_integrations,
            })
        }

        JsonCommand::ChangeCoords {
            expr,
            from_vars,
            to_vars,
            system,
        } => Ok(Command::ChangeCoords {
            expr: cvt(expr)?,
            from_vars,
            to_vars,
            system: parse_coord_system(&system)?,
        }),

        JsonCommand::PathIntegral { expr, curve } => Ok(Command::PathIntegral {
            expr: cvt(expr)?,
            curve: json_param_curve_to_param_curve(curve)?,
        }),

        JsonCommand::SurfaceIntegral { expr, vars } => Ok(Command::SurfaceIntegral {
            expr: cvt(expr)?,
            vars,
        }),

        JsonCommand::Limit {
            expr,
            var,
            point,
            side,
        } => {
            let point = match &point.kind {
                mathlex::ExprKind::Constant(mathlex::MathConstant::Infinity) => LimitPoint::PosInf,
                mathlex::ExprKind::Constant(mathlex::MathConstant::NegInfinity) => {
                    LimitPoint::NegInf
                }
                _ => LimitPoint::Finite(cvt(point)?),
            };
            let side = side.as_deref().map(parse_side).transpose()?;
            Ok(Command::Limit {
                expr: cvt(expr)?,
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
            expr: cvt(expr)?,
            var,
            center: cvt(center)?,
            order: order.unwrap_or(3),
        }),

        JsonCommand::Laurent {
            expr,
            var,
            center,
            order,
        } => Ok(Command::Laurent {
            expr: cvt(expr)?,
            var,
            center: cvt(center)?,
            order: order.unwrap_or(3),
        }),

        JsonCommand::Asymptotic { expr, var, order } => Ok(Command::Asymptotic {
            expr: cvt(expr)?,
            var,
            order: order.unwrap_or(3),
        }),

        JsonCommand::Compose {
            outer,
            inner,
            var,
            order,
        } => Ok(Command::Compose {
            outer: cvt(outer)?,
            inner: cvt(inner)?,
            var,
            order: order.unwrap_or(3),
        }),

        JsonCommand::Revert { expr, var, order } => Ok(Command::Revert {
            expr: cvt(expr)?,
            var,
            order: order.unwrap_or(3),
        }),

        JsonCommand::Puiseux {
            expr,
            var,
            center,
            order,
        } => Ok(Command::Puiseux {
            expr: cvt(expr)?,
            var,
            center: center
                .map(cvt)
                .transpose()?
                .unwrap_or(Expression::Integer(0)),
            order: order.unwrap_or(3),
        }),

        JsonCommand::Frobenius {
            ode,
            fn_name,
            var,
            point,
            order,
        } => Ok(Command::Frobenius {
            ode: cvt(ode)?,
            fn_name,
            var,
            point: point
                .map(cvt)
                .transpose()?
                .unwrap_or(Expression::Integer(0)),
            order: order.unwrap_or(3),
        }),

        JsonCommand::Pade {
            expr,
            var,
            center,
            m,
            n,
        } => Ok(Command::Pade {
            expr: cvt(expr)?,
            var,
            center: center
                .map(cvt)
                .transpose()?
                .unwrap_or(Expression::Integer(0)),
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
            ode: cvt(ode)?,
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
            expr: cvt(expr)?,
            var,
            period: cvt(period)?,
            terms: terms.unwrap_or(3),
        }),

        JsonCommand::Residue { expr, var, point } => Ok(Command::Residue {
            expr: cvt(expr)?,
            var,
            point: cvt(point)?,
        }),

        JsonCommand::LaplaceTransform {
            expr,
            time_var,
            freq_var,
        } => Ok(Command::LaplaceTransform {
            expr: cvt(expr)?,
            time_var,
            freq_var: freq_var.unwrap_or_else(|| "s".into()),
        }),

        JsonCommand::InverseLaplace {
            expr,
            freq_var,
            time_var,
        } => Ok(Command::InverseLaplace {
            expr: cvt(expr)?,
            freq_var,
            time_var: time_var.unwrap_or_else(|| "t".into()),
        }),

        JsonCommand::FourierTransform {
            expr,
            time_var,
            freq_var,
        } => Ok(Command::FourierTransform {
            expr: cvt(expr)?,
            time_var,
            freq_var: freq_var.unwrap_or_else(|| "omega".into()),
        }),

        JsonCommand::InverseFourier {
            expr,
            freq_var,
            time_var,
        } => Ok(Command::InverseFourier {
            expr: cvt(expr)?,
            freq_var,
            time_var: time_var.unwrap_or_else(|| "t".into()),
        }),

        JsonCommand::ZTransform { expr, var, z_var } => Ok(Command::ZTransform {
            expr: cvt(expr)?,
            var,
            z_var: z_var.unwrap_or_else(|| "z".into()),
        }),

        JsonCommand::InverseZTransform { expr, z_var, var } => Ok(Command::InverseZTransform {
            expr: cvt(expr)?,
            z_var,
            var: var.unwrap_or_else(|| "n".into()),
        }),

        JsonCommand::MellinTransform { expr, var, s_var } => Ok(Command::MellinTransform {
            expr: cvt(expr)?,
            var,
            s_var: s_var.unwrap_or_else(|| "s".into()),
        }),

        JsonCommand::InverseMellin { expr, s_var, var } => Ok(Command::InverseMellin {
            expr: cvt(expr)?,
            s_var,
            var: var.unwrap_or_else(|| "x".into()),
        }),

        JsonCommand::SpecialFn { kind, args } => {
            let parsed_args = args.into_iter().map(cvt).collect::<Result<Vec<_>, _>>()?;
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
                equation: cvt(equation)?,
                fn_name,
                var,
                ic,
            })
        }

        JsonCommand::Matrix { op, operands } => {
            let matrix_op = parse_matrix_op(&op)?;
            let parsed_operands = operands
                .unwrap_or_default()
                .into_iter()
                .map(json_matrix_operand_to_matrix_expr)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Command::Matrix {
                op: matrix_op,
                operands: parsed_operands,
            })
        }

        JsonCommand::Nabla { op, input, vars } => {
            let nabla_op = parse_nabla_op(&op)?;
            let nabla_input = json_nabla_input_to_nabla_input(input, &nabla_op)?;
            Ok(Command::Nabla {
                op: nabla_op,
                input: nabla_input,
                vars,
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
                    let expr = cvt(c.expr)?;
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
                objective: cvt(objective)?,
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
                .into_iter()
                .map(cvt)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Command::LagrangeMult {
                objective: cvt(objective)?,
                vars,
                equality_constraints: parsed_constraints,
            })
        }

        JsonCommand::OdeSystem {
            equations,
            fn_names,
            var,
            ic,
        } => {
            let parsed_eqs: Vec<_> = equations.into_iter().map(cvt).collect::<Result<_, _>>()?;
            let parsed_ic = ic.map(json_system_ivp_to_system_ivp).transpose()?;
            Ok(Command::OdeSystem {
                equations: parsed_eqs,
                fn_names,
                var,
                ic: parsed_ic,
            })
        }

        JsonCommand::Pde {
            equation,
            fn_name,
            vars,
        } => Ok(Command::Pde {
            equation: cvt(equation)?,
            fn_name,
            vars,
        }),
    }
}

// ── Small conversions ─────────────────────────────────────────────────────────

fn json_ivp_to_ivp(j: JsonIvpData) -> Result<IvpData, String> {
    let derivatives_at = j
        .derivatives_at
        .into_iter()
        .enumerate()
        .map(|(i, e)| cvt(e).map_err(|err| format!("derivatives_at[{}]: {}", i, err)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(IvpData {
        var_at: cvt(j.var_at)?,
        fn_at: cvt(j.fn_at)?,
        derivatives_at,
    })
}

fn json_system_ivp_to_system_ivp(j: JsonSystemIvpData) -> Result<SystemIvpData, String> {
    let values_at = j
        .values_at
        .into_iter()
        .enumerate()
        .map(|(i, e)| cvt(e).map_err(|err| format!("values_at[{}]: {}", i, err)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SystemIvpData {
        var_at: cvt(j.var_at)?,
        values_at,
    })
}

fn json_precision_to_precision(j: JsonPrecision) -> Result<Precision, String> {
    let abs_tol = j.abs_tol.map(cvt).transpose()?;
    let rel_tol = j.rel_tol.map(cvt).transpose()?;
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

fn json_integration_step_to_step(j: JsonIntegrationStep) -> Result<IntegrationStep, String> {
    Ok(IntegrationStep {
        var: j.var,
        from: cvt(j.from)?,
        to: cvt(j.to)?,
    })
}

fn json_param_curve_to_param_curve(j: JsonParamCurve) -> Result<ParamCurve, String> {
    let components = j
        .components
        .into_iter()
        .enumerate()
        .map(|(i, e)| cvt(e).map_err(|err| format!("curve.components[{}]: {}", i, err)))
        .collect::<Result<Vec<_>, String>>()?;
    Ok(ParamCurve {
        components,
        param: j.param,
        from: cvt(j.from)?,
        to: cvt(j.to)?,
    })
}

fn json_matrix_operand_to_matrix_expr(op: JsonMatrixOperand) -> Result<MatrixExpr, String> {
    match op {
        JsonMatrixOperand::Scalar(e) => Ok(MatrixExpr::Scalar(cvt(e)?)),
        JsonMatrixOperand::Matrix { rows } => {
            let parsed_rows = rows
                .into_iter()
                .enumerate()
                .map(|(r, row)| {
                    row.into_iter()
                        .enumerate()
                        .map(|(c, cell)| {
                            cvt(cell).map_err(|err| format!("matrix.rows[{}][{}]: {}", r, c, err))
                        })
                        .collect::<Result<Vec<_>, String>>()
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(MatrixExpr::Matrix(parsed_rows))
        }
        JsonMatrixOperand::Vector { elements } => {
            let parsed = elements
                .into_iter()
                .enumerate()
                .map(|(i, e)| cvt(e).map_err(|err| format!("vector.elements[{}]: {}", i, err)))
                .collect::<Result<Vec<_>, String>>()?;
            Ok(MatrixExpr::Vector(parsed))
        }
    }
}

fn parse_coord_system(s: &str) -> Result<CoordSystem, String> {
    match s {
        "Cartesian2D" => Ok(CoordSystem::Cartesian2D),
        "Polar2D" => Ok(CoordSystem::Polar2D),
        "Cartesian3D" => Ok(CoordSystem::Cartesian3D),
        "Cylindrical" => Ok(CoordSystem::Cylindrical),
        "Spherical" => Ok(CoordSystem::Spherical),
        "Parabolic2D" => Ok(CoordSystem::Parabolic2D),
        "Elliptic2D" => Ok(CoordSystem::Elliptic2D),
        "Custom" => Ok(CoordSystem::Custom),
        other => Err(format!("unknown CoordSystem `{}`", other)),
    }
}

pub(super) fn parse_nabla_op(s: &str) -> Result<NablaOp, String> {
    match s {
        "Grad" => Ok(NablaOp::Grad),
        "Div" => Ok(NablaOp::Div),
        "Curl" => Ok(NablaOp::Curl),
        "Laplacian" => Ok(NablaOp::Laplacian),
        "DivOfCurl" => Ok(NablaOp::DivOfCurl),
        "CurlOfGrad" => Ok(NablaOp::CurlOfGrad),
        "DivOfGrad" => Ok(NablaOp::DivOfGrad),
        other => Err(format!("unknown NablaOp `{}`", other)),
    }
}

/// Convert [`JsonNablaInput`] to [`NablaInput`], validating op compatibility.
///
/// Scalar ops (Grad, Laplacian, CurlOfGrad, DivOfGrad) expect a scalar
/// Expression; vector ops (Div, Curl, DivOfCurl) expect a vector field.
fn json_nabla_input_to_nabla_input(
    input: JsonNablaInput,
    op: &NablaOp,
) -> Result<NablaInput, String> {
    match op {
        NablaOp::Grad | NablaOp::Laplacian | NablaOp::CurlOfGrad | NablaOp::DivOfGrad => {
            match input {
                JsonNablaInput::Scalar(e) => Ok(NablaInput::Scalar(cvt(e)?)),
                JsonNablaInput::VectorField(_) => Err(format!(
                    "Nabla {:?}: expected a scalar expression, got a vector field",
                    op
                )),
            }
        }
        NablaOp::Div | NablaOp::Curl | NablaOp::DivOfCurl => match input {
            JsonNablaInput::VectorField(components) => {
                let parsed = components
                    .into_iter()
                    .enumerate()
                    .map(|(i, e)| {
                        cvt(e).map_err(|err| format!("Nabla {:?}: input[{}]: {}", op, i, err))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok(NablaInput::VectorField(parsed))
            }
            JsonNablaInput::Scalar(_) => Err(format!(
                "Nabla {:?}: expected a vector field (array), got a scalar",
                op
            )),
        },
    }
}
