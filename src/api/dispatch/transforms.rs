//! Integral transform command dispatchers.
//!
//! Wires Laplace, Inverse-Laplace, Fourier, and Inverse-Fourier to their
//! respective engines in `crate::integral_transforms`. Z-transform and
//! Mellin-transform stubs return [`DiagnosticCode::NotImplemented`] until
//! their engines land in v0.10.0.

use crate::api::diagnostic::{Diagnostic, DiagnosticCode};
use crate::api::narrative::Narrative;
use crate::api::response::{EngineId, Response, ResultKey};
use crate::ast::Expression;
use crate::numeric::compile::{compile, decompile};
use crate::numeric::trace::{Step, TechniqueTag, Trace};
use crate::numeric::SymbolId;

use super::helpers::{engine_error, steps_from_trace, symbolic_entry};

// ── Laplace ───────────────────────────────────────────────────────────────────

pub(super) fn laplace_transform_cmd(
    expr: &Expression,
    time_var: &str,
    freq_var: &str,
    narrate: bool,
) -> Response {
    let compiled = compile(expr);
    let t = SymbolId::intern(time_var);
    let s = SymbolId::intern(freq_var);
    match crate::integral_transforms::laplace::laplace_transform(&compiled, t, s) {
        Ok(result) => {
            let value = decompile(&result.expr);
            let mut trace = Trace::new();
            if narrate {
                for step_text in &result.steps {
                    trace.push(
                        Step::new(TechniqueTag::Custom("laplace-transform"), step_text.clone())
                            .with_output(result.expr.clone()),
                    );
                }
            }
            let mut r = Response::default();
            r.results.push((
                ResultKey::Single,
                symbolic_entry(value, EngineId::LaplaceTransform, steps_from_trace(&trace)),
            ));
            r.meta.engine_trace.push(EngineId::LaplaceTransform);
            r
        }
        Err(e) => engine_error("command.laplace_transform", format!("{e}")),
    }
}

// ── Inverse Laplace ───────────────────────────────────────────────────────────

pub(super) fn inverse_laplace_cmd(
    expr: &Expression,
    freq_var: &str,
    time_var: &str,
    narrate: bool,
) -> Response {
    let compiled = compile(expr);
    let s = SymbolId::intern(freq_var);
    let t = SymbolId::intern(time_var);
    match crate::integral_transforms::inverse_laplace::inverse_laplace(&compiled, s, t) {
        Ok(result) => {
            let value = decompile(&result.expr);
            let mut trace = Trace::new();
            if narrate {
                for step_text in &result.steps {
                    trace.push(
                        Step::new(
                            TechniqueTag::Custom("inverse-laplace-transform"),
                            step_text.clone(),
                        )
                        .with_output(result.expr.clone()),
                    );
                }
            }
            let mut r = Response::default();
            r.results.push((
                ResultKey::Single,
                symbolic_entry(value, EngineId::InverseLaplace, steps_from_trace(&trace)),
            ));
            r.meta.engine_trace.push(EngineId::InverseLaplace);
            r
        }
        Err(e) => engine_error("command.inverse_laplace", format!("{e}")),
    }
}

// ── Fourier ───────────────────────────────────────────────────────────────────

pub(super) fn fourier_transform_cmd(
    expr: &Expression,
    time_var: &str,
    freq_var: &str,
    narrate: bool,
) -> Response {
    let compiled = compile(expr);
    let t = SymbolId::intern(time_var);
    let omega = SymbolId::intern(freq_var);
    match crate::integral_transforms::fourier_transform::fourier_transform(&compiled, t, omega) {
        Ok(result) => {
            let value = decompile(&result.expr);
            let mut trace = Trace::new();
            if narrate {
                for step_text in &result.steps {
                    trace.push(
                        Step::new(TechniqueTag::Custom("fourier-transform"), step_text.clone())
                            .with_output(result.expr.clone()),
                    );
                }
            }
            let mut r = Response::default();
            r.results.push((
                ResultKey::Single,
                symbolic_entry(value, EngineId::FourierTransform, steps_from_trace(&trace)),
            ));
            r.meta.engine_trace.push(EngineId::FourierTransform);
            r
        }
        Err(e) => engine_error("command.fourier_transform", format!("{e}")),
    }
}

// ── Inverse Fourier ───────────────────────────────────────────────────────────

pub(super) fn inverse_fourier_cmd(
    expr: &Expression,
    freq_var: &str,
    time_var: &str,
    narrate: bool,
) -> Response {
    let compiled = compile(expr);
    let omega = SymbolId::intern(freq_var);
    let t = SymbolId::intern(time_var);
    match crate::integral_transforms::inverse_fourier::inverse_fourier(&compiled, omega, t) {
        Ok(result) => {
            let value = decompile(&result.expr);
            let mut trace = Trace::new();
            if narrate {
                for step_text in &result.steps {
                    trace.push(
                        Step::new(
                            TechniqueTag::Custom("inverse-fourier-transform"),
                            step_text.clone(),
                        )
                        .with_output(result.expr.clone()),
                    );
                }
            }
            let mut r = Response::default();
            r.results.push((
                ResultKey::Single,
                symbolic_entry(value, EngineId::InverseFourier, steps_from_trace(&trace)),
            ));
            r.meta.engine_trace.push(EngineId::InverseFourier);
            r
        }
        Err(e) => engine_error("command.inverse_fourier", format!("{e}")),
    }
}

// ── Stubs (v0.10.0) ───────────────────────────────────────────────────────────

pub(super) fn z_transform_stub(_narrate: bool) -> Response {
    not_implemented_response(
        "command.z_transform",
        "Z-transform dispatch pending (v0.10.0).",
    )
}

pub(super) fn inverse_z_transform_stub(_narrate: bool) -> Response {
    not_implemented_response(
        "command.inverse_z_transform",
        "Inverse Z-transform dispatch pending (v0.10.0).",
    )
}

pub(super) fn mellin_transform_stub(_narrate: bool) -> Response {
    not_implemented_response(
        "command.mellin_transform",
        "Mellin transform dispatch pending (v0.10.0).",
    )
}

pub(super) fn inverse_mellin_stub(_narrate: bool) -> Response {
    not_implemented_response(
        "command.inverse_mellin",
        "Inverse Mellin transform dispatch pending (v0.10.0).",
    )
}

fn not_implemented_response(template_id: &'static str, message: &'static str) -> Response {
    let mut r = Response::default();
    r.diagnostics.push(Diagnostic::of(
        DiagnosticCode::NotImplemented,
        Narrative::new(template_id, message),
    ));
    r
}
