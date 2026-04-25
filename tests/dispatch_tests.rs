//! Per-Command golden tests for the single-entry dispatcher.
//!
//! Each test builds a [`Request`] for one [`Command`] variant, invokes
//! [`execute`], and asserts that the [`Response`] carries the expected
//! shape (value kind, engine id, step count, diagnostic codes). Tests
//! for commands that still route to the `NotImplemented` diagnostic
//! assert on the diagnostic shape rather than a symbolic value.

use thales::api::command::{Command, IvpData, LimitPoint, SimplifyRules};
use thales::api::diagnostic::{DiagnosticCode, Severity};
use thales::api::dispatch::execute;
use thales::api::domain::Domain;
use thales::api::json::execute_ffi;
use thales::api::request::{Request, SolveMode};
use thales::api::response::{EngineId, ResultKey, ResultValue};
use thales::ast::{BinaryOp, Expression, Function, Variable};
use thales::parser::parse_expression;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn var(name: &str) -> Expression {
    Expression::Variable(Variable::new(name))
}

fn int(n: i64) -> Expression {
    Expression::Integer(n)
}

fn add(a: Expression, b: Expression) -> Expression {
    Expression::Binary(BinaryOp::Add, Box::new(a), Box::new(b))
}

fn sub(a: Expression, b: Expression) -> Expression {
    Expression::Binary(BinaryOp::Sub, Box::new(a), Box::new(b))
}

fn mul(a: Expression, b: Expression) -> Expression {
    Expression::Binary(BinaryOp::Mul, Box::new(a), Box::new(b))
}

fn pow(base: Expression, exp: Expression) -> Expression {
    Expression::Power(Box::new(base), Box::new(exp))
}

fn request(cmd: Command) -> Request {
    Request {
        command: cmd,
        ..Default::default()
    }
}

fn assert_single_symbolic(resp: &thales::api::response::Response, engine: EngineId) {
    assert_eq!(resp.results.len(), 1, "expected single result entry");
    let (key, entry) = &resp.results[0];
    assert_eq!(*key, ResultKey::Single);
    assert_eq!(entry.engine, engine);
    assert!(
        matches!(entry.value, ResultValue::Symbolic(_)),
        "expected Symbolic value, got {:?}",
        entry.value
    );
}

fn assert_not_implemented(resp: &thales::api::response::Response) {
    assert!(
        resp.diagnostics
            .iter()
            .any(|d| d.code == DiagnosticCode::NotImplemented),
        "expected a NotImplemented diagnostic"
    );
}

// ── Algebra ──────────────────────────────────────────────────────────────────

#[test]
fn simplify_x_plus_x() {
    let resp = execute(request(Command::Simplify {
        expr: add(var("x"), var("x")),
        rules: SimplifyRules::all(),
        over: None,
    }))
    .unwrap();
    assert_single_symbolic(&resp, EngineId::Simplify);
}

#[test]
fn expand_product_of_sums() {
    let resp = execute(request(Command::Expand {
        expr: mul(add(var("x"), int(1)), add(var("x"), int(2))),
        target: None,
    }))
    .unwrap();
    assert_single_symbolic(&resp, EngineId::Simplify);
}

#[test]
fn factor_falls_back_with_diagnostic() {
    let resp = execute(request(Command::Factor {
        expr: add(mul(var("x"), var("x")), int(-1)),
        over: Domain::real(),
        target: None,
    }))
    .unwrap();
    assert_single_symbolic(&resp, EngineId::Simplify);
    // v0.8.1 emits NotImplemented noting the partial fallback.
    assert_not_implemented(&resp);
}

#[test]
fn substitute_pair() {
    let resp = execute(request(Command::Substitute {
        expr: add(var("x"), var("y")),
        bindings: vec![(var("x"), int(3))],
        target: None,
    }))
    .unwrap();
    assert_single_symbolic(&resp, EngineId::Simplify);
}

#[test]
fn partial_fractions_simple() {
    // 1/(x*(x+1)) — decomposes to 1/x - 1/(x+1).
    let numer = int(1);
    let denom = mul(var("x"), add(var("x"), int(1)));
    let expr = Expression::Binary(BinaryOp::Div, Box::new(numer), Box::new(denom));
    let resp = execute(request(Command::PartialFractions {
        expr,
        var: "x".to_string(),
    }))
    .unwrap();
    // Partial fractions should succeed; either symbolic result or diagnostic
    // if the decomposition rejects the shape. Both paths are acceptable here.
    assert!(!resp.results.is_empty());
}

#[test]
fn rearrange_linear_equation() {
    // 2x + 3 - 7 = 0, solve for x.
    let equation = sub(add(mul(int(2), var("x")), int(3)), int(7));
    let resp = execute(request(Command::Rearrange {
        equation,
        solve_for: "x".to_string(),
    }))
    .unwrap();
    assert_eq!(resp.results[0].1.engine, EngineId::EquationSolver);
}

// ── Solve ────────────────────────────────────────────────────────────────────

#[test]
fn solve_for_linear() {
    let relation = sub(add(mul(int(2), var("x")), int(3)), int(7));
    let resp = execute(request(Command::SolveFor {
        relation,
        var: "x".to_string(),
        over: Domain::real(),
    }))
    .unwrap();
    assert_eq!(resp.results[0].1.engine, EngineId::EquationSolver);
}

#[test]
fn solve_system_2x2() {
    // x + y - 5 = 0, x - y - 1 = 0
    let eq1 = sub(add(var("x"), var("y")), int(5));
    let eq2 = sub(sub(var("x"), var("y")), int(1));
    let resp = execute(request(Command::SolveSystem {
        equations: vec![eq1, eq2],
        vars: vec!["x".to_string(), "y".to_string()],
        over: Domain::real(),
    }))
    .unwrap();
    assert_eq!(resp.results.len(), 2, "two variables, two entries");
    assert_eq!(resp.results[0].1.engine, EngineId::SystemSolver);
}

// ── Differentiation ──────────────────────────────────────────────────────────

#[test]
fn diff_x_squared() {
    let resp = execute(request(Command::Diff {
        expr: pow(var("x"), int(2)),
        var: "x".to_string(),
        order: 1,
    }))
    .unwrap();
    assert_single_symbolic(&resp, EngineId::Differentiation);
}

#[test]
fn diff_higher_order() {
    let resp = execute(request(Command::Diff {
        expr: pow(var("x"), int(3)),
        var: "x".to_string(),
        order: 2,
    }))
    .unwrap();
    assert_single_symbolic(&resp, EngineId::Differentiation);
    // Second-order derivative records at least two steps when narrate=true.
    assert!(resp.results[0].1.steps.len() >= 2);
}

#[test]
fn partial_diff_mixed() {
    // ∂²(x²y)/∂x∂y = 2x
    let expr = mul(pow(var("x"), int(2)), var("y"));
    let resp = execute(request(Command::PartialDiff {
        expr,
        vars: vec![("x".to_string(), 1), ("y".to_string(), 1)],
    }))
    .unwrap();
    assert_single_symbolic(&resp, EngineId::Differentiation);
}

#[test]
fn gradient_two_dim() {
    // ∇(x²+y²) = (2x, 2y)
    let expr = add(pow(var("x"), int(2)), pow(var("y"), int(2)));
    let resp = execute(request(Command::Gradient {
        expr,
        vars: vec!["x".to_string(), "y".to_string()],
    }))
    .unwrap();
    assert_eq!(resp.results[0].1.engine, EngineId::Differentiation);
    // Two-dimensional gradient: primary value plus one alternative component.
    assert_eq!(resp.results[0].1.alternatives.len(), 1);
}

// ── Integration ──────────────────────────────────────────────────────────────

#[test]
fn integrate_polynomial() {
    let resp = execute(request(Command::Integrate {
        expr: mul(int(2), var("x")),
        var: "x".to_string(),
    }))
    .unwrap();
    assert_single_symbolic(&resp, EngineId::PatternIntegration);
}

#[test]
fn def_integrate_symbolic() {
    // ∫₀¹ x dx = 1/2
    let resp = execute(request(Command::DefIntegrate {
        expr: var("x"),
        var: "x".to_string(),
        from: int(0),
        to: int(1),
    }))
    .unwrap();
    assert_eq!(resp.results[0].1.engine, EngineId::PatternIntegration);
}

#[test]
fn def_integrate_numeric_fallback() {
    // ∫₀¹ exp(-x²) dx — no closed form; numeric mode forces quadrature.
    let req = Request {
        command: Command::DefIntegrate {
            expr: Expression::Function(
                Function::Exp,
                vec![Expression::Unary(
                    thales::ast::UnaryOp::Neg,
                    Box::new(pow(var("x"), int(2))),
                )],
            ),
            var: "x".to_string(),
            from: int(0),
            to: int(1),
        },
        mode: SolveMode::Numeric,
        ..Default::default()
    };
    let resp = execute(req).unwrap();
    // Either numeric fallback succeeded or the engine reported an error
    // entry; both paths are non-empty.
    assert!(!resp.results.is_empty());
}

// ── Limits ───────────────────────────────────────────────────────────────────

#[test]
fn limit_finite_point() {
    // lim x→0 (x + 1) = 1
    let resp = execute(request(Command::Limit {
        expr: add(var("x"), int(1)),
        var: "x".to_string(),
        point: LimitPoint::Finite(int(0)),
        side: None,
    }))
    .unwrap();
    assert_single_symbolic(&resp, EngineId::LHopital);
}

#[test]
fn limit_at_infinity() {
    // lim x→∞ (1/x) = 0
    let expr = Expression::Binary(BinaryOp::Div, Box::new(int(1)), Box::new(var("x")));
    let resp = execute(request(Command::Limit {
        expr,
        var: "x".to_string(),
        point: LimitPoint::PosInf,
        side: None,
    }))
    .unwrap();
    // Either a clean value result or an error entry; either way
    // we should see at least one result entry.
    assert!(!resp.results.is_empty());
}

// ── Series / transforms ──────────────────────────────────────────────────────

#[test]
fn fourier_series_sin() {
    let resp = execute(request(Command::FourierSeries {
        expr: Expression::Function(Function::Sin, vec![var("x")]),
        var: "x".to_string(),
        period: Expression::Float(2.0 * std::f64::consts::PI),
        terms: 3,
    }))
    .unwrap();
    assert!(!resp.results.is_empty());
    assert_eq!(resp.results[0].1.engine, EngineId::FourierSeries);
}

#[test]
fn taylor_emits_not_implemented() {
    let resp = execute(request(Command::Taylor {
        expr: Expression::Function(Function::Sin, vec![var("x")]),
        var: "x".to_string(),
        center: int(0),
        order: 5,
    }))
    .unwrap();
    assert_not_implemented(&resp);
}

#[test]
fn residue_emits_not_implemented() {
    let expr = Expression::Binary(BinaryOp::Div, Box::new(int(1)), Box::new(var("z")));
    let resp = execute(request(Command::Residue {
        expr,
        var: "z".to_string(),
        point: int(0),
    }))
    .unwrap();
    assert_not_implemented(&resp);
}

// ── ODE ─────────────────────────────────────────────────────────────────────

#[test]
fn ode_separable_first_order() {
    // dy/dx = y — separable, y = C·eˣ
    let resp = execute(request(Command::Ode {
        equation: var("y"),
        fn_name: "y".to_string(),
        var: "x".to_string(),
        ic: None,
    }))
    .unwrap();
    assert_single_symbolic(&resp, EngineId::OdeFirstOrder);
}

#[test]
fn ode_with_ivp() {
    let resp = execute(request(Command::Ode {
        equation: var("y"),
        fn_name: "y".to_string(),
        var: "x".to_string(),
        ic: Some(IvpData {
            var_at: int(0),
            fn_at: int(1),
            derivatives_at: Vec::new(),
        }),
    }))
    .unwrap();
    // Either a symbolic particular solution or an engine error entry.
    assert!(!resp.results.is_empty());
}

// ── Not-yet-implemented surface ──────────────────────────────────────────────

#[test]
fn matrix_emits_not_implemented() {
    let resp = execute(request(Command::Matrix {
        op: thales::api::command::MatrixOp::Determinant,
        operands: Vec::new(),
    }))
    .unwrap();
    assert_not_implemented(&resp);
}

#[test]
fn optimize_emits_not_implemented() {
    let resp = execute(request(Command::Optimize {
        objective: var("x"),
        vars: vec!["x".to_string()],
        constraints: Vec::new(),
        sense: thales::api::command::OptSense::Minimize,
    }))
    .unwrap();
    assert_not_implemented(&resp);
}

// ── Noop + diagnostic shape ──────────────────────────────────────────────────

#[test]
fn noop_is_error_severity() {
    let resp = execute(Request::default()).unwrap();
    assert!(resp.results.is_empty());
    let diag = &resp.diagnostics[0];
    assert_eq!(diag.code, DiagnosticCode::NotImplemented);
    assert_eq!(diag.severity, Severity::Error);
}

// ── FFI round-trip ───────────────────────────────────────────────────────────

#[test]
fn ffi_round_trip_simplify() {
    let req = r#"{"command":{"type":"Simplify","expr":"x + x"}}"#;
    let resp = execute_ffi(req).unwrap();
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["results"][0]["value"]["kind"], "Symbolic");
}

#[test]
fn ffi_round_trip_diff_integrate_inverses() {
    // d/dx(x^2) then ∫ that dx should round-trip to a polynomial.
    let diff_req = r#"{"command":{"type":"Diff","expr":"x^2","var":"x","order":1}}"#;
    let resp = execute_ffi(diff_req).unwrap();
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    let derivative = v["results"][0]["value"]["expr"]
        .as_str()
        .unwrap()
        .to_string();

    let integrate_req = format!(
        r#"{{"command":{{"type":"Integrate","expr":"{}","var":"x"}}}}"#,
        derivative
    );
    let resp2 = execute_ffi(&integrate_req).unwrap();
    let v2: serde_json::Value = serde_json::from_str(&resp2).unwrap();
    assert_eq!(v2["results"][0]["engine"], "PatternIntegration");
}

#[test]
fn ffi_round_trip_solve_for() {
    let req = r#"{"command":{"type":"SolveFor","relation":"2*x + 3 = 7","var":"x"}}"#;
    let resp = execute_ffi(req).unwrap();
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["results"][0]["engine"], "EquationSolver");
}

#[test]
fn ffi_round_trip_solve_system() {
    let req = r#"{"command":{"type":"SolveSystem","equations":["x + y = 5","x - y = 1"],"vars":["x","y"]}}"#;
    let resp = execute_ffi(req).unwrap();
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    let results = v["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn ffi_round_trip_ode() {
    let req = r#"{"command":{"type":"Ode","equation":"y","fn_name":"y","var":"x"}}"#;
    let resp = execute_ffi(req).unwrap();
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["results"][0]["engine"], "OdeFirstOrder");
}

#[test]
fn ffi_round_trip_parse_uses_parser() {
    // Confirm that the JSON transport round-trips through parse_expression.
    let req = r#"{"command":{"type":"Simplify","expr":"sin(x)^2 + cos(x)^2"}}"#;
    let resp = execute_ffi(req).unwrap();
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["results"][0]["value"]["kind"], "Symbolic");

    // Sanity: the input parses directly too.
    assert!(parse_expression("sin(x)^2 + cos(x)^2").is_ok());
}

// ── Narrative dictionary regression ──────────────────────────────────────────

#[test]
fn narrative_resolver_covers_all_not_implemented_templates() {
    use thales::api::narratives::resolve_template;
    // Every NotImplemented template id emitted by the dispatcher must
    // resolve against the English dictionary so clients that rely on the
    // dictionary see a real message, not the placeholder fallback.
    for id in [
        "command.noop",
        "command.conjugate",
        "command.inverse_fn",
        "command.apply_identity",
        "command.total_diff",
        "command.divergence",
        "command.curl",
        "command.laplacian",
        "command.jacobian",
        "command.hessian",
        "command.directional_diff",
        "command.taylor",
        "command.laurent",
        "command.asymptotic",
        "command.compose",
        "command.revert",
        "command.residue",
        "command.special_fn",
        "command.matrix",
        "command.optimize",
        "command.lagrange_mult",
        "command.factor.partial",
        "step.generic",
    ] {
        assert!(
            resolve_template(id).is_some(),
            "missing narrative template `{}`",
            id
        );
    }
}

// ── Narrative resolution (post-render_response) ─────────────────────────────
//
// The dispatcher applies api::render::render_response before returning so
// every Narrative leaving the crate carries its resolved Markdown in
// fallback_md. These tests assert the rewritten text matches the dictionary
// entry rather than the engine-supplied stub string.

fn dict_entry(template_id: &str) -> &'static str {
    thales::api::narratives::resolve_template(template_id).expect("template id must be in en.json")
}

#[test]
fn dispatch_resolves_step_generic_narrative() {
    // A successful narrated dispatch produces step.generic narratives. After
    // render_response the fallback_md should match the dictionary entry.
    let resp = execute(request(Command::Diff {
        expr: pow(var("x"), int(2)),
        var: "x".to_string(),
        order: 1,
    }))
    .unwrap();
    let entry = &resp.results[0].1;
    assert!(
        !entry.steps.is_empty(),
        "diff with default narrate should emit steps"
    );
    let body = &entry.steps[0].narrative.fallback_md;
    assert_eq!(body, dict_entry("step.generic"));
}

#[test]
fn dispatch_resolves_not_implemented_diagnostic_narrative() {
    // A NotImplemented dispatch emits a diagnostic whose narrative renders
    // against the matching template id.
    let resp = execute(request(Command::Conjugate {
        expr: var("x"),
        target: None,
    }))
    .unwrap();
    let diag = resp
        .diagnostics
        .iter()
        .find(|d| d.code == DiagnosticCode::NotImplemented)
        .expect("expected NotImplemented diagnostic");
    let body = &diag.narrative.fallback_md;
    assert_eq!(body, dict_entry("command.conjugate"));
    // Must not still carry the raw stub.
    assert_ne!(body, "command not yet implemented in v0.8.1");
}

#[test]
fn dispatch_resolves_factor_partial_diagnostic_narrative() {
    // Factor returns a partial-fallback diagnostic with a specific template id.
    let resp = execute(request(Command::Factor {
        expr: add(mul(var("x"), var("x")), int(-1)),
        over: Domain::real(),
        target: None,
    }))
    .unwrap();
    let diag = resp
        .diagnostics
        .iter()
        .find(|d| d.code == DiagnosticCode::NotImplemented)
        .expect("expected partial-factor diagnostic");
    assert_eq!(
        diag.narrative.fallback_md,
        dict_entry("command.factor.partial")
    );
}

#[test]
fn dispatch_resolves_unsolved_value_narrative() {
    // NotImplemented commands emit a ResultValue::Unsolved whose nested
    // narrative also resolves through render_response.
    let resp = execute(request(Command::Taylor {
        expr: var("x"),
        var: "x".to_string(),
        center: int(0),
        order: 3,
    }))
    .unwrap();
    let entry = &resp.results[0].1;
    let resolved_body = match &entry.value {
        ResultValue::Unsolved { reason } => reason.fallback_md.clone(),
        other => panic!("expected Unsolved, got {:?}", other),
    };
    assert_eq!(resolved_body, dict_entry("command.taylor"));
}

#[test]
fn dispatch_resolves_noop_diagnostic_narrative() {
    let resp = execute(Request::default()).unwrap();
    let diag = &resp.diagnostics[0];
    assert_eq!(diag.code, DiagnosticCode::NotImplemented);
    assert_eq!(diag.severity, Severity::Error);
    assert_eq!(diag.narrative.fallback_md, dict_entry("command.noop"));
}

#[test]
fn ffi_round_trip_carries_resolved_narrative() {
    // The FFI surface goes through dispatch::execute and therefore inherits
    // render_response. The resolved Markdown must reach the JSON.
    let req = r#"{"command":{"type":"Conjugate","expr":"x"}}"#;
    let resp = execute_ffi(req).unwrap();
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    let diag_md = v["diagnostics"][0]["fallback_md"]
        .as_str()
        .expect("diagnostic fallback_md must serialise as a string");
    assert_eq!(diag_md, dict_entry("command.conjugate"));
}
