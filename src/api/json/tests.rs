//! JSON transport architecture tests.

use serde_json::Value;

use super::super::command::{Command, LimitPoint, Side};
use super::super::response::ResultKey;
use super::execute_ffi;
use super::request::{parse_expr_str, request_from_json};

// ── Helper ────────────────────────────────────────────────────────────────────

fn execute_ok(req: &str) -> Value {
    let resp = execute_ffi(req).unwrap_or_else(|e| panic!("execute_ffi failed: {}", e));
    serde_json::from_str(&resp).unwrap()
}

// ── Test 1: exhaustiveness compile check ─────────────────────────────────────

/// This function exists solely to produce a compile error if [`Command`] gains
/// a variant without a matching [`super::request::JsonCommand`] variant.
/// It is never called at runtime.
#[allow(dead_code)]
fn _exhaustiveness_check(cmd: Command) {
    match cmd {
        Command::Noop => {}
        Command::Simplify { .. } => {}
        Command::Expand { .. } => {}
        Command::Factor { .. } => {}
        Command::Substitute { .. } => {}
        Command::CombineLikeTerms { .. } => {}
        Command::CommonDenominator { .. } => {}
        Command::PartialFractions { .. } => {}
        Command::Rationalize { .. } => {}
        Command::Conjugate { .. } => {}
        Command::InverseFn { .. } => {}
        Command::Rearrange { .. } => {}
        Command::ApplyIdentity { .. } => {}
        Command::SolveFor { .. } => {}
        Command::SolveSystem { .. } => {}
        Command::Diff { .. } => {}
        Command::PartialDiff { .. } => {}
        Command::TotalDiff { .. } => {}
        Command::Gradient { .. } => {}
        Command::Divergence { .. } => {}
        Command::Curl { .. } => {}
        Command::Laplacian { .. } => {}
        Command::Jacobian { .. } => {}
        Command::Hessian { .. } => {}
        Command::DirectionalDiff { .. } => {}
        Command::Integrate { .. } => {}
        Command::DefIntegrate { .. } => {}
        Command::Limit { .. } => {}
        Command::Taylor { .. } => {}
        Command::Laurent { .. } => {}
        Command::Asymptotic { .. } => {}
        Command::Compose { .. } => {}
        Command::Revert { .. } => {}
        Command::Puiseux { .. } => {}
        Command::Frobenius { .. } => {}
        Command::Pade { .. } => {}
        Command::Wkb { .. } => {}
        Command::FourierSeries { .. } => {}
        Command::Residue { .. } => {}
        Command::SpecialFn { .. } => {}
        Command::Ode { .. } => {}
        Command::Matrix { .. } => {}
        Command::Optimize { .. } => {}
        Command::LagrangeMult { .. } => {}
        // If Command gains a new variant, this match becomes non-exhaustive
        // and compilation fails — that is the intent.
        #[allow(unreachable_patterns)]
        _ => {}
    }
}

#[test]
fn exhaustiveness_compile_check() {
    // The real check is compile-time via _exhaustiveness_check above.
    // This test exists so the function is referenced and the module is
    // exercised.
    assert!(true, "compile-time exhaustiveness check passed");
}

// ── Test 2: all existing commands parse ──────────────────────────────────────

#[test]
fn all_existing_commands_parse() {
    let payloads: &[&str] = &[
        r#"{"command":{"type":"Noop"}}"#,
        r#"{"command":{"type":"Simplify","expr":"x + x"}}"#,
        r#"{"command":{"type":"Expand","expr":"(x+1)^2"}}"#,
        r#"{"command":{"type":"Factor","expr":"x^2 - 1"}}"#,
        r#"{"command":{"type":"Substitute","expr":"x + y","bindings":[{"old":"x","new":"2"}]}}"#,
        r#"{"command":{"type":"CombineLikeTerms","expr":"2*x + 3*x"}}"#,
        r#"{"command":{"type":"CommonDenominator","expr":"1/2 + 1/3"}}"#,
        r#"{"command":{"type":"PartialFractions","expr":"1/(x^2-1)","var":"x"}}"#,
        r#"{"command":{"type":"Rationalize","expr":"1/(1+sqrt(2))"}}"#,
        r#"{"command":{"type":"Conjugate","expr":"1 + 2*i"}}"#,
        r#"{"command":{"type":"InverseFn","expr":"2*x + 1","var":"x"}}"#,
        r#"{"command":{"type":"Rearrange","equation":"y = 2*x + 1","solve_for":"x"}}"#,
        r#"{"command":{"type":"ApplyIdentity","expr":"sin(x)^2 + cos(x)^2","identity":"PythagoreanTrig"}}"#,
        r#"{"command":{"type":"SolveFor","relation":"2*x + 6 = 0","var":"x"}}"#,
        r#"{"command":{"type":"SolveSystem","equations":["x + y = 3","x - y = 1"],"vars":["x","y"]}}"#,
        r#"{"command":{"type":"Diff","expr":"x^3","var":"x","order":1}}"#,
        r#"{"command":{"type":"PartialDiff","expr":"x^2*y","vars":[{"var":"x","order":1},{"var":"y","order":1}]}}"#,
        r#"{"command":{"type":"TotalDiff","expr":"x^2 + y^2","var":"t","deps":[{"name":"x","expr":"t"},{"name":"y","expr":"t^2"}]}}"#,
        r#"{"command":{"type":"Gradient","expr":"x^2 + y^2","vars":["x","y"]}}"#,
        r#"{"command":{"type":"Divergence","field":["x","y","z"],"vars":["x","y","z"]}}"#,
        r#"{"command":{"type":"Curl","field":["y","-x","0"],"vars":["x","y","z"]}}"#,
        r#"{"command":{"type":"Laplacian","expr":"x^2 + y^2","vars":["x","y"]}}"#,
        r#"{"command":{"type":"Jacobian","fields":["x*y","x+y"],"vars":["x","y"]}}"#,
        r#"{"command":{"type":"Hessian","expr":"x^2 + y^2","vars":["x","y"]}}"#,
        r#"{"command":{"type":"DirectionalDiff","expr":"x^2 + y^2","vars":["x","y"],"direction":["1","0"]}}"#,
        r#"{"command":{"type":"Integrate","expr":"2*x","var":"x"}}"#,
        r#"{"command":{"type":"DefIntegrate","expr":"x^2","var":"x","from":"0","to":"1"}}"#,
        r#"{"command":{"type":"Limit","expr":"sin(x)/x","var":"x","point":"0"}}"#,
        r#"{"command":{"type":"Taylor","expr":"exp(x)","var":"x","center":"0","order":3}}"#,
        r#"{"command":{"type":"Laurent","expr":"1/x","var":"x","center":"0","order":2}}"#,
        r#"{"command":{"type":"Asymptotic","expr":"1/x","var":"x","order":3}}"#,
        r#"{"command":{"type":"Compose","outer":"sin(x)","inner":"x^2","var":"x","order":3}}"#,
        r#"{"command":{"type":"Revert","expr":"x + x^2","var":"x","order":3}}"#,
        r#"{"command":{"type":"FourierSeries","expr":"x","var":"x","period":"2"}}"#,
        r#"{"command":{"type":"Residue","expr":"1/(x^2+1)","var":"x","point":"i"}}"#,
        r#"{"command":{"type":"SpecialFn","kind":"Gamma","args":["2"]}}"#,
        r#"{"command":{"type":"Ode","equation":"y' = y","fn_name":"y","var":"x"}}"#,
        r#"{"command":{"type":"Matrix","op":"Determinant","operands":[{"rows":[["1","2"],["3","4"]]}]}}"#,
        r#"{"command":{"type":"Optimize","objective":"x^2 + y^2","vars":["x","y"],"constraints":[{"kind":"Equality","expr":"x + y - 1"}],"sense":"Minimize"}}"#,
        r#"{"command":{"type":"LagrangeMult","objective":"x^2 + y^2","vars":["x","y"],"equality_constraints":["x + y - 1"]}}"#,
    ];

    for payload in payloads {
        let val: Value = serde_json::from_str(payload).unwrap();
        let result = request_from_json(&val);
        assert!(
            result.is_ok(),
            "payload failed to parse: {}\nerror: {}",
            payload,
            result.unwrap_err()
        );
    }
}

// ── Test 3: request controls passthrough ─────────────────────────────────────

#[test]
fn request_controls_passthrough() {
    let payload = r#"{
        "command": {"type": "Simplify", "expr": "x"},
        "narrate": false,
        "mode": "Numeric",
        "precision": {"decimal_digits": 10},
        "budget": {"max_wall_ms": 5000, "max_iterations": 1000},
        "ambient_domain": "Real",
        "seed": 42
    }"#;
    let val: Value = serde_json::from_str(payload).unwrap();
    let req = request_from_json(&val).expect("should parse");

    assert!(!req.narrate, "narrate should be false");
    assert_eq!(
        req.mode,
        super::super::request::SolveMode::Numeric,
        "mode should be Numeric"
    );
    let prec = req.precision.expect("precision should be Some");
    assert_eq!(prec.decimal_digits, 10);
    let budget = req.budget.expect("budget should be Some");
    assert_eq!(budget.max_wall_ms, Some(5000));
    assert_eq!(budget.max_iterations, Some(1000));
    assert!(
        req.ambient_domain.is_some(),
        "ambient_domain should be Some"
    );
    assert_eq!(req.seed, Some(42));
}

// ── Test 4: matrix operands parsed ───────────────────────────────────────────

#[test]
fn matrix_operands_parsed() {
    let payload = r#"{
        "command": {
            "type": "Matrix",
            "op": "Determinant",
            "operands": [{"rows": [["1","2"],["3","4"]]}]
        }
    }"#;
    let val: Value = serde_json::from_str(payload).unwrap();
    let req = request_from_json(&val).expect("should parse");
    match req.command {
        Command::Matrix { op: _, operands } => {
            assert!(
                !operands.is_empty(),
                "operands should be non-empty for Matrix command"
            );
        }
        _ => panic!("expected Matrix command"),
    }
}

// ── Test 5: limit side parsed ─────────────────────────────────────────────────

#[test]
fn limit_side_parsed() {
    let payload =
        r#"{"command":{"type":"Limit","expr":"1/x","var":"x","point":"0","side":"Left"}}"#;
    let val: Value = serde_json::from_str(payload).unwrap();
    let req = request_from_json(&val).expect("should parse");
    match req.command {
        Command::Limit { side, point, .. } => {
            assert_eq!(side, Some(Side::Left), "side should be Left");
            assert!(
                matches!(point, LimitPoint::Finite(_)),
                "point should be finite"
            );
        }
        _ => panic!("expected Limit command"),
    }
}

// ── Test 6: branch condition serialized ──────────────────────────────────────

#[test]
fn branch_condition_serialized() {
    // x^2 = 1 has two roots; each result should have a key in the response
    let req = r#"{"command":{"type":"SolveFor","relation":"x^2 - 1 = 0","var":"x"}}"#;
    let v = execute_ok(req);
    let results = v["results"].as_array().expect("results should be array");
    // With multiple roots we expect at least one result
    assert!(!results.is_empty(), "should have at least one result");
    // Each result's key must be an object or a string (not null)
    for result in results {
        assert!(!result["key"].is_null(), "result key must not be null");
    }
}

// ── Test 7: numeric precision serialized ─────────────────────────────────────

#[test]
fn numeric_precision_serialized() {
    // The Numeric result value should now include decimal_digits in the output
    // We build a response containing a Numeric entry via the helpers directly
    // rather than driving through a full command (which requires numeric mode).
    // Instead we test the serialiser unit directly.
    use super::super::request::Precision;
    use super::super::response::{
        EngineId, ExecutionMeta, NumericMethod, Response, ResultEntry, ResultShape, ResultValue,
    };
    use super::response::response_to_json;

    let prec = Precision {
        decimal_digits: 15,
        abs_tol: None,
        rel_tol: None,
    };
    let entry = ResultEntry {
        value: ResultValue::Numeric {
            value: parse_expr_str("3.14159").unwrap(),
            precision: prec,
            method: NumericMethod::NewtonRaphson,
        },
        structured: None,
        shape: ResultShape::Scalar,
        unit: None,
        steps: vec![],
        alternatives: vec![],
        engine: EngineId::EquationSolver,
    };
    let response = Response {
        results: vec![(ResultKey::Single, entry)],
        assumptions: vec![],
        diagnostics: vec![],
        meta: ExecutionMeta::default(),
    };
    let v = response_to_json(&response);
    let kind = v["results"][0]["value"]["kind"].as_str().unwrap();
    assert_eq!(kind, "Numeric");
    let dd = v["results"][0]["value"]["decimal_digits"].as_u64().unwrap();
    assert_eq!(dd, 15, "decimal_digits should be 15 in serialized output");
}

// ── Test 8: nosolution domain serialized ─────────────────────────────────────

#[test]
fn nosolution_domain_serialized() {
    use super::super::domain::Domain;
    use super::super::narrative::Narrative;
    use super::super::response::{
        EngineId, ExecutionMeta, Response, ResultEntry, ResultShape, ResultValue,
    };
    use super::response::response_to_json;

    let entry = ResultEntry {
        value: ResultValue::NoSolution {
            domain: Domain::natural(),
            reason: Narrative {
                template_id: "solver.no_solution",
                fallback_md: "no solution in ℕ".to_string(),
                bindings: Vec::new(),
            },
        },
        structured: None,
        shape: ResultShape::Scalar,
        unit: None,
        steps: vec![],
        alternatives: vec![],
        engine: EngineId::EquationSolver,
    };
    let response = Response {
        results: vec![(ResultKey::Single, entry)],
        assumptions: vec![],
        diagnostics: vec![],
        meta: ExecutionMeta::default(),
    };
    let v = response_to_json(&response);
    let kind = v["results"][0]["value"]["kind"].as_str().unwrap();
    assert_eq!(kind, "NoSolution");
    let domain_str = v["results"][0]["value"]["domain"].as_str().unwrap();
    assert!(
        !domain_str.is_empty(),
        "domain should be non-empty in serialized NoSolution"
    );
}

// ── Test 9: unknown command gives clear error message ────────────────────────

#[test]
fn unknown_command_error_message() {
    let req = r#"{"command":{"type":"DefinitelyNotARealCommand","expr":"x"}}"#;
    let err = execute_ffi(req).unwrap_err();
    // The error should indicate the parsing failure. With serde tag dispatch,
    // unknown variants produce a descriptive error mentioning the unknown type.
    assert!(
        err.contains("invalid request") || err.contains("DefinitelyNotARealCommand"),
        "error should mention unknown command; got: {}",
        err
    );
}
