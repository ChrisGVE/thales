//! Nabla (del operator) command dispatcher.
//!
//! Routes [`Command::Nabla`] to the appropriate [`crate::calculus::nabla::Nabla`]
//! method based on [`NablaOp`].  Inputs are compiled from `Expression` to
//! `Arc<Expr>` here; outputs are decompiled back to `Expression` before
//! returning.

use crate::api::command::{NablaInput, NablaOp};
use crate::api::response::{EngineId, Response, ResultEntry, ResultKey, ResultShape, ResultValue};
use crate::ast::Expression;
use crate::calculus::nabla::Nabla;
use crate::numeric::compile::{compile, decompile};
use crate::numeric::SymbolId;

use super::helpers::{engine_error, steps_from_trace, symbolic_entry};
use crate::numeric::trace::{Step, TechniqueTag, Trace};

/// Dispatch a Nabla command.
///
/// Returns a [`Response`] containing:
/// - scalar ops: `ResultShape::Scalar`, primary = computed scalar.
/// - vector ops: `ResultShape::Vector`, primary = first component,
///   `alternatives` = remaining components.
/// - identity ops: same shape as their output (scalar or vector).
pub(in crate::api::dispatch) fn nabla_cmd(
    op: NablaOp,
    input: NablaInput,
    vars: &[String],
    narrate: bool,
) -> Response {
    let sym_ids: Vec<SymbolId> = vars.iter().map(|v| SymbolId::intern(v)).collect();
    let nabla = Nabla::new(sym_ids);
    let mut trace = Trace::new();

    match op {
        NablaOp::Grad => dispatch_grad(input, &nabla, vars, narrate, &mut trace),
        NablaOp::Div => dispatch_div(input, &nabla, vars, narrate, &mut trace),
        NablaOp::Curl => dispatch_curl(input, &nabla, vars, narrate, &mut trace),
        NablaOp::Laplacian => dispatch_laplacian(input, &nabla, vars, narrate, &mut trace),
        NablaOp::DivOfCurl => dispatch_div_of_curl(input, &nabla, vars, narrate, &mut trace),
        NablaOp::CurlOfGrad => dispatch_curl_of_grad(input, &nabla, narrate, &mut trace),
        NablaOp::DivOfGrad => dispatch_div_of_grad(input, &nabla, narrate, &mut trace),
    }
}

// ── Scalar: grad ──────────────────────────────────────────────────────────────

fn dispatch_grad(
    input: NablaInput,
    nabla: &Nabla,
    vars: &[String],
    narrate: bool,
    trace: &mut Trace,
) -> Response {
    let f = match require_scalar(input, NablaOp::Grad) {
        Ok(f) => f,
        Err(r) => return r,
    };
    let arc_f = compile(&f);
    let components = nabla.grad(&arc_f);
    if narrate {
        for (xi, ci) in vars.iter().zip(components.iter()) {
            trace.push(
                Step::new(TechniqueTag::PowerRule, format!("Gradient ∂/∂{}", xi))
                    .with_output(ci.clone()),
            );
        }
    }
    let decomp: Vec<Expression> = components.iter().map(|c| decompile(c)).collect();
    vector_response(decomp, steps_from_trace(trace))
}

// ── Vector: div ───────────────────────────────────────────────────────────────

fn dispatch_div(
    input: NablaInput,
    nabla: &Nabla,
    vars: &[String],
    narrate: bool,
    trace: &mut Trace,
) -> Response {
    let field = match require_vector(input, NablaOp::Div) {
        Ok(f) => f,
        Err(r) => return r,
    };
    let arc_field: Vec<_> = field.iter().map(compile).collect();
    if narrate {
        for (arc_fi, xi) in arc_field.iter().zip(vars.iter()) {
            trace.push(
                Step::new(TechniqueTag::Divergence, format!("Divergence ∂/∂{}", xi))
                    .with_output(arc_fi.clone()),
            );
        }
    }
    let result = nabla.div(&arc_field);
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(decompile(&result), EngineId::Nabla, steps_from_trace(trace)),
    ));
    r.meta.engine_trace.push(EngineId::Nabla);
    r
}

// ── Vector: curl ──────────────────────────────────────────────────────────────

fn dispatch_curl(
    input: NablaInput,
    nabla: &Nabla,
    vars: &[String],
    narrate: bool,
    trace: &mut Trace,
) -> Response {
    let field = match require_vector(input, NablaOp::Curl) {
        Ok(f) => f,
        Err(r) => return r,
    };
    if field.len() != 3 || vars.len() < 3 {
        return engine_error(
            "command.nabla.curl",
            format!(
                "Curl requires exactly 3 field components and 3 variables, got {} and {}",
                field.len(),
                vars.len()
            ),
        );
    }
    let arc_field: [_; 3] = [compile(&field[0]), compile(&field[1]), compile(&field[2])];
    let [cx, cy, cz] = nabla.curl(&arc_field);
    if narrate {
        let (x, y, z) = (&vars[0], &vars[1], &vars[2]);
        trace.push(
            Step::new(
                TechniqueTag::Curl,
                format!("Curl x: ∂F₃/∂{} − ∂F₂/∂{}", y, z),
            )
            .with_output(cx.clone()),
        );
        trace.push(
            Step::new(
                TechniqueTag::Curl,
                format!("Curl y: ∂F₁/∂{} − ∂F₃/∂{}", z, x),
            )
            .with_output(cy.clone()),
        );
        trace.push(
            Step::new(
                TechniqueTag::Curl,
                format!("Curl z: ∂F₂/∂{} − ∂F₁/∂{}", x, y),
            )
            .with_output(cz.clone()),
        );
    }
    let components = vec![decompile(&cx), decompile(&cy), decompile(&cz)];
    vector_response_with_steps(components, steps_from_trace(trace))
}

// ── Scalar: laplacian ─────────────────────────────────────────────────────────

fn dispatch_laplacian(
    input: NablaInput,
    nabla: &Nabla,
    vars: &[String],
    narrate: bool,
    trace: &mut Trace,
) -> Response {
    let f = match require_scalar(input, NablaOp::Laplacian) {
        Ok(f) => f,
        Err(r) => return r,
    };
    let arc_f = compile(&f);
    if narrate {
        for xi in vars {
            trace.push(Step::new(
                TechniqueTag::Laplacian,
                format!("Laplacian ∂²/∂{}²", xi),
            ));
        }
    }
    let result = nabla.laplacian(&arc_f);
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(decompile(&result), EngineId::Nabla, steps_from_trace(trace)),
    ));
    r.meta.engine_trace.push(EngineId::Nabla);
    r
}

// ── Identity: div_of_curl ─────────────────────────────────────────────────────

fn dispatch_div_of_curl(
    input: NablaInput,
    nabla: &Nabla,
    vars: &[String],
    narrate: bool,
    trace: &mut Trace,
) -> Response {
    let field = match require_vector(input, NablaOp::DivOfCurl) {
        Ok(f) => f,
        Err(r) => return r,
    };
    if field.len() != 3 || vars.len() < 3 {
        return engine_error(
            "command.nabla.div_of_curl",
            format!(
                "DivOfCurl requires exactly 3 field components and 3 variables, got {} and {}",
                field.len(),
                vars.len()
            ),
        );
    }
    let arc_field: [_; 3] = [compile(&field[0]), compile(&field[1]), compile(&field[2])];
    if narrate {
        trace.push(Step::new(
            TechniqueTag::Curl,
            "Step 1: compute curl(F)".to_string(),
        ));
        trace.push(Step::new(
            TechniqueTag::Divergence,
            "Step 2: compute div(curl(F)) — identity = 0".to_string(),
        ));
    }
    let result = nabla.div_of_curl(&arc_field);
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(decompile(&result), EngineId::Nabla, steps_from_trace(trace)),
    ));
    r.meta.engine_trace.push(EngineId::Nabla);
    r
}

// ── Identity: curl_of_grad ────────────────────────────────────────────────────

fn dispatch_curl_of_grad(
    input: NablaInput,
    nabla: &Nabla,
    narrate: bool,
    trace: &mut Trace,
) -> Response {
    let f = match require_scalar(input, NablaOp::CurlOfGrad) {
        Ok(f) => f,
        Err(r) => return r,
    };
    let arc_f = compile(&f);
    if narrate {
        trace.push(Step::new(
            TechniqueTag::PowerRule,
            "Step 1: compute grad(f)".to_string(),
        ));
        trace.push(Step::new(
            TechniqueTag::Curl,
            "Step 2: compute curl(grad(f)) — identity = (0,0,0)".to_string(),
        ));
    }
    let [cx, cy, cz] = nabla.curl_of_grad(&arc_f);
    let components = vec![decompile(&cx), decompile(&cy), decompile(&cz)];
    vector_response_with_steps(components, steps_from_trace(trace))
}

// ── Identity: div_of_grad ─────────────────────────────────────────────────────

fn dispatch_div_of_grad(
    input: NablaInput,
    nabla: &Nabla,
    narrate: bool,
    trace: &mut Trace,
) -> Response {
    let f = match require_scalar(input, NablaOp::DivOfGrad) {
        Ok(f) => f,
        Err(r) => return r,
    };
    let arc_f = compile(&f);
    if narrate {
        trace.push(Step::new(
            TechniqueTag::PowerRule,
            "Step 1: compute grad(f)".to_string(),
        ));
        trace.push(Step::new(
            TechniqueTag::Divergence,
            "Step 2: compute div(grad(f)) = laplacian(f)".to_string(),
        ));
    }
    let result = nabla.div_of_grad(&arc_f);
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(decompile(&result), EngineId::Nabla, steps_from_trace(trace)),
    ));
    r.meta.engine_trace.push(EngineId::Nabla);
    r
}

// ── Input validation helpers ──────────────────────────────────────────────────

fn require_scalar(input: NablaInput, op: NablaOp) -> Result<Expression, Response> {
    match input {
        NablaInput::Scalar(e) => Ok(e),
        NablaInput::VectorField(_) => Err(engine_error(
            "command.nabla",
            format!("{:?} requires a scalar input, not a vector field", op),
        )),
    }
}

fn require_vector(input: NablaInput, op: NablaOp) -> Result<Vec<Expression>, Response> {
    match input {
        NablaInput::VectorField(v) => Ok(v),
        NablaInput::Scalar(_) => Err(engine_error(
            "command.nabla",
            format!("{:?} requires a vector-field input, not a scalar", op),
        )),
    }
}

// ── Response builders ─────────────────────────────────────────────────────────

/// Build a vector response: primary = first component, alternatives = rest.
fn vector_response(
    components: Vec<Expression>,
    steps: Vec<crate::api::response::NarratedStep>,
) -> Response {
    vector_response_with_steps(components, steps)
}

fn vector_response_with_steps(
    components: Vec<Expression>,
    steps: Vec<crate::api::response::NarratedStep>,
) -> Response {
    let (primary, alternatives) = match components.split_first() {
        Some((first, rest)) => (first.clone(), rest.to_vec()),
        None => (Expression::Integer(0), Vec::new()),
    };
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        ResultEntry {
            value: ResultValue::Symbolic(primary),
            structured: None,
            shape: ResultShape::Vector,
            unit: None,
            steps,
            alternatives,
            engine: EngineId::Nabla,
        },
    ));
    r.meta.engine_trace.push(EngineId::Nabla);
    r
}
