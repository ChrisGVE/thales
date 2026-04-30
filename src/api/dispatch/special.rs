//! F1d special-function and residue dispatchers.
//!
//! `SpecialFn` (Gamma / Beta / Erf / Erfc) routes to `crate::special`. The
//! special module already returns a structured `SpecialFunctionResult` with
//! a symbolic value, an optional numeric value, and a list of
//! human-readable derivation steps; the dispatcher relays them as a
//! `Response` carrying the symbolic value and one trace step per
//! derivation entry.
//!
//! `Residue` routes to `crate::numeric::series::residue`, which is the
//! Arc<Expr>-typed engine. Inputs are compiled at the I/O seam, the
//! result is decompiled back to `Expression`.

use crate::api::command::SpecialKind;
use crate::api::response::{EngineId, Response, ResultKey};
use crate::ast::Expression;
use crate::numeric::compile::{compile, decompile};
use crate::numeric::series::residue;
use crate::numeric::trace::{Step, TechniqueTag, Trace};
use crate::numeric::SymbolId;
use crate::special;

use super::helpers::{engine_error, steps_from_trace, symbolic_entry};

pub(super) fn special_fn_cmd(kind: SpecialKind, args: &[Expression], narrate: bool) -> Response {
    let result = match kind {
        SpecialKind::Gamma => {
            if args.len() != 1 {
                return engine_error(
                    "command.special_fn",
                    format!("Gamma takes 1 argument, got {}", args.len()),
                );
            }
            special::gamma(&args[0])
        }
        SpecialKind::Beta => {
            if args.len() != 2 {
                return engine_error(
                    "command.special_fn",
                    format!("Beta takes 2 arguments, got {}", args.len()),
                );
            }
            special::beta(&args[0], &args[1])
        }
        SpecialKind::Erf => {
            if args.len() != 1 {
                return engine_error(
                    "command.special_fn",
                    format!("Erf takes 1 argument, got {}", args.len()),
                );
            }
            special::erf(&args[0])
        }
        SpecialKind::Erfc => {
            if args.len() != 1 {
                return engine_error(
                    "command.special_fn",
                    format!("Erfc takes 1 argument, got {}", args.len()),
                );
            }
            special::erfc(&args[0])
        }
        SpecialKind::LnGamma => {
            if args.len() != 1 {
                return engine_error(
                    "command.special_fn",
                    format!("LnGamma takes 1 argument, got {}", args.len()),
                );
            }
            special::lngamma(&args[0])
        }
        SpecialKind::Digamma => {
            if args.len() != 1 {
                return engine_error(
                    "command.special_fn",
                    format!("Digamma takes 1 argument, got {}", args.len()),
                );
            }
            special::digamma(&args[0])
        }
        SpecialKind::BesselJ => {
            if args.len() != 2 {
                return engine_error(
                    "command.special_fn",
                    format!("BesselJ takes 2 arguments, got {}", args.len()),
                );
            }
            special::bessel_j(&args[0], &args[1])
        }
        SpecialKind::BesselY => {
            if args.len() != 2 {
                return engine_error(
                    "command.special_fn",
                    format!("BesselY takes 2 arguments, got {}", args.len()),
                );
            }
            special::bessel_y(&args[0], &args[1])
        }
        SpecialKind::BesselI => {
            if args.len() != 2 {
                return engine_error(
                    "command.special_fn",
                    format!("BesselI takes 2 arguments, got {}", args.len()),
                );
            }
            special::bessel_i(&args[0], &args[1])
        }
        SpecialKind::BesselK => {
            if args.len() != 2 {
                return engine_error(
                    "command.special_fn",
                    format!("BesselK takes 2 arguments, got {}", args.len()),
                );
            }
            special::bessel_k(&args[0], &args[1])
        }
        SpecialKind::AiryAi => {
            if args.len() != 1 {
                return engine_error(
                    "command.special_fn",
                    format!("AiryAi takes 1 argument, got {}", args.len()),
                );
            }
            special::airy_ai(&args[0])
        }
        SpecialKind::AiryBi => {
            if args.len() != 1 {
                return engine_error(
                    "command.special_fn",
                    format!("AiryBi takes 1 argument, got {}", args.len()),
                );
            }
            special::airy_bi(&args[0])
        }
        SpecialKind::Zeta => {
            if args.len() != 1 {
                return engine_error(
                    "command.special_fn",
                    format!("Zeta takes 1 argument, got {}", args.len()),
                );
            }
            special::zeta(&args[0])
        }
        SpecialKind::Si => {
            if args.len() != 1 {
                return engine_error(
                    "command.special_fn",
                    format!("Si takes 1 argument, got {}", args.len()),
                );
            }
            special::si(&args[0])
        }
        SpecialKind::Ci => {
            if args.len() != 1 {
                return engine_error(
                    "command.special_fn",
                    format!("Ci takes 1 argument, got {}", args.len()),
                );
            }
            special::ci(&args[0])
        }
        SpecialKind::Ei => {
            if args.len() != 1 {
                return engine_error(
                    "command.special_fn",
                    format!("Ei takes 1 argument, got {}", args.len()),
                );
            }
            special::ei(&args[0])
        }
        SpecialKind::Heaviside => {
            if args.len() != 1 {
                return engine_error(
                    "command.special_fn",
                    format!("Heaviside takes 1 argument, got {}", args.len()),
                );
            }
            special::heaviside(&args[0])
        }
        SpecialKind::DiracDelta => {
            if args.len() != 1 {
                return engine_error(
                    "command.special_fn",
                    format!("DiracDelta takes 1 argument, got {}", args.len()),
                );
            }
            special::dirac_delta(&args[0])
        }
    };

    match result {
        Ok(special_result) => {
            let mut trace = Trace::new();
            if narrate {
                for step in &special_result.derivation_steps {
                    trace.push(Step::new(TechniqueTag::SpecialFunction, step.clone()));
                }
            }
            let mut r = Response::default();
            r.results.push((
                ResultKey::Single,
                symbolic_entry(
                    special_result.value,
                    EngineId::SpecialFunctions,
                    steps_from_trace(&trace),
                ),
            ));
            r.meta.engine_trace.push(EngineId::SpecialFunctions);
            r
        }
        Err(e) => engine_error("command.special_fn", format!("{}", e)),
    }
}

pub(super) fn residue_cmd(
    expr: &Expression,
    var: &str,
    point: &Expression,
    narrate: bool,
) -> Response {
    let expr_arc = compile(expr);
    let pole_arc = compile(point);
    let var_id = SymbolId::intern(var);
    let mut trace = Trace::new();
    let trace_handle = if narrate { Some(&mut trace) } else { None };
    match residue(&expr_arc, var_id, &pole_arc, trace_handle) {
        Some(value_arc) => {
            let value = decompile(&value_arc);
            let mut r = Response::default();
            r.results.push((
                ResultKey::Single,
                symbolic_entry(value, EngineId::Residue, steps_from_trace(&trace)),
            ));
            r.meta.engine_trace.push(EngineId::Residue);
            r
        }
        None => engine_error(
            "command.residue",
            format!(
                "residue computation failed at {}: essential singularity or non-rational form",
                point
            ),
        ),
    }
}
