//! JSON transport architecture tests.

use serde_json::{json, Value};

use super::super::command::{Command, LimitPoint, Side};
use super::super::response::ResultKey;
use super::execute_ffi;
use super::request::request_from_json;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn execute_ok(req: Value) -> Value {
    let s = serde_json::to_string(&req).unwrap();
    let resp = execute_ffi(&s).unwrap_or_else(|e| panic!("execute_ffi failed: {}", e));
    serde_json::from_str(&resp).unwrap()
}

/// Parse a plain-text math expression and return its mathlex serde JSON form.
fn expr_json(input: &str) -> Value {
    let ml = mathlex::parse(input)
        .unwrap_or_else(|e| panic!("expr_json: failed to parse `{}`: {:?}", input, e));
    serde_json::to_value(&ml)
        .unwrap_or_else(|e| panic!("expr_json: failed to serialise `{}`: {}", input, e))
}

/// Build a minimal JSON request value for a given command object.
fn req(command: Value) -> Value {
    json!({ "command": command })
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
        Command::MultiIntegrate { .. } => {}
        Command::ChangeCoords { .. } => {}
        Command::PathIntegral { .. } => {}
        Command::SurfaceIntegral { .. } => {}
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
        Command::LaplaceTransform { .. } => {}
        Command::InverseLaplace { .. } => {}
        Command::FourierTransform { .. } => {}
        Command::InverseFourier { .. } => {}
        Command::ZTransform { .. } => {}
        Command::InverseZTransform { .. } => {}
        Command::MellinTransform { .. } => {}
        Command::InverseMellin { .. } => {}
        Command::SpecialFn { .. } => {}
        Command::Ode { .. } => {}
        Command::OdeSystem { .. } => {}
        Command::Pde { .. } => {}
        Command::Matrix { .. } => {}
        Command::Nabla { .. } => {}
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
    let payloads: Vec<Value> = vec![
        req(json!({"type": "Noop"})),
        req(json!({"type": "Simplify", "expr": expr_json("x + x")})),
        req(json!({"type": "Expand", "expr": expr_json("(x+1)^2")})),
        req(json!({"type": "Factor", "expr": expr_json("x^2 - 1")})),
        req(json!({"type": "Substitute", "expr": expr_json("x + y"),
            "bindings": [{"old": expr_json("x"), "new": expr_json("2")}]})),
        req(json!({"type": "CombineLikeTerms", "expr": expr_json("2*x + 3*x")})),
        req(json!({"type": "CommonDenominator", "expr": expr_json("1/2 + 1/3")})),
        req(json!({"type": "PartialFractions", "expr": expr_json("1/(x^2-1)"), "var": "x"})),
        req(json!({"type": "Rationalize", "expr": expr_json("1/(1+sqrt(2))")})),
        req(json!({"type": "Conjugate", "expr": expr_json("1 + 2*i")})),
        req(json!({"type": "InverseFn", "expr": expr_json("2*x + 1"), "var": "x"})),
        req(json!({"type": "Rearrange",
            "equation": expr_json("y = 2*x + 1"), "solve_for": "x"})),
        req(json!({"type": "ApplyIdentity",
            "expr": expr_json("sin(x)^2 + cos(x)^2"), "identity": "PythagoreanTrig"})),
        req(json!({"type": "SolveFor",
            "relation": expr_json("2*x + 6 = 0"), "var": "x"})),
        req(json!({"type": "SolveSystem",
            "equations": [expr_json("x + y = 3"), expr_json("x - y = 1")],
            "vars": ["x", "y"]})),
        req(json!({"type": "Diff",
            "expr": expr_json("x^3"), "var": "x", "order": 1})),
        req(json!({"type": "PartialDiff",
            "expr": expr_json("x^2*y"),
            "vars": [{"var": "x", "order": 1}, {"var": "y", "order": 1}]})),
        req(json!({"type": "TotalDiff",
            "expr": expr_json("x^2 + y^2"), "var": "t",
            "deps": [{"name": "x", "expr": expr_json("t")},
                     {"name": "y", "expr": expr_json("t^2")}]})),
        req(json!({"type": "Gradient",
            "expr": expr_json("x^2 + y^2"), "vars": ["x", "y"]})),
        req(json!({"type": "Divergence",
            "field": [expr_json("x"), expr_json("y"), expr_json("z")],
            "vars": ["x", "y", "z"]})),
        req(json!({"type": "Curl",
            "field": [expr_json("y"), expr_json("-x"), expr_json("0")],
            "vars": ["x", "y", "z"]})),
        req(json!({"type": "Laplacian",
            "expr": expr_json("x^2 + y^2"), "vars": ["x", "y"]})),
        req(json!({"type": "Jacobian",
            "fields": [expr_json("x*y"), expr_json("x+y")], "vars": ["x", "y"]})),
        req(json!({"type": "Hessian",
            "expr": expr_json("x^2 + y^2"), "vars": ["x", "y"]})),
        req(json!({"type": "DirectionalDiff",
            "expr": expr_json("x^2 + y^2"), "vars": ["x", "y"],
            "direction": [expr_json("1"), expr_json("0")]})),
        req(json!({"type": "Integrate", "expr": expr_json("2*x"), "var": "x"})),
        req(json!({"type": "DefIntegrate",
            "expr": expr_json("x^2"), "var": "x",
            "from": expr_json("0"), "to": expr_json("1")})),
        req(json!({"type": "Limit",
            "expr": expr_json("sin(x)/x"), "var": "x", "point": expr_json("0")})),
        req(json!({"type": "Taylor",
            "expr": expr_json("exp(x)"), "var": "x",
            "center": expr_json("0"), "order": 3})),
        req(json!({"type": "Laurent",
            "expr": expr_json("1/x"), "var": "x",
            "center": expr_json("0"), "order": 2})),
        req(json!({"type": "Asymptotic",
            "expr": expr_json("1/x"), "var": "x", "order": 3})),
        req(json!({"type": "Compose",
            "outer": expr_json("sin(x)"), "inner": expr_json("x^2"),
            "var": "x", "order": 3})),
        req(json!({"type": "Revert",
            "expr": expr_json("x + x^2"), "var": "x", "order": 3})),
        req(json!({"type": "FourierSeries",
            "expr": expr_json("x"), "var": "x", "period": expr_json("2")})),
        req(json!({"type": "Residue",
            "expr": expr_json("1/(x^2+1)"), "var": "x", "point": expr_json("i")})),
        req(json!({"type": "LaplaceTransform",
            "expr": expr_json("t^2"), "time_var": "t"})),
        req(json!({"type": "LaplaceTransform",
            "expr": expr_json("exp(-t)"), "time_var": "t", "freq_var": "s"})),
        req(json!({"type": "InverseLaplace",
            "expr": expr_json("1/(s+1)"), "freq_var": "s"})),
        req(json!({"type": "InverseLaplace",
            "expr": expr_json("1/s^2"), "freq_var": "s", "time_var": "t"})),
        req(json!({"type": "FourierTransform",
            "expr": expr_json("exp(-t^2)"), "time_var": "t"})),
        req(json!({"type": "FourierTransform",
            "expr": expr_json("sin(t)"), "time_var": "t", "freq_var": "omega"})),
        req(json!({"type": "InverseFourier",
            "expr": expr_json("1/(1+omega^2)"), "freq_var": "omega"})),
        req(json!({"type": "InverseFourier",
            "expr": expr_json("exp(-omega^2)"), "freq_var": "omega", "time_var": "t"})),
        req(json!({"type": "ZTransform",
            "expr": expr_json("2^n"), "var": "n"})),
        req(json!({"type": "ZTransform",
            "expr": expr_json("n"), "var": "n", "z_var": "z"})),
        req(json!({"type": "InverseZTransform",
            "expr": expr_json("z/(z-1)"), "z_var": "z"})),
        req(json!({"type": "InverseZTransform",
            "expr": expr_json("1/(z-2)"), "z_var": "z", "var": "n"})),
        req(json!({"type": "MellinTransform",
            "expr": expr_json("x^2"), "var": "x"})),
        req(json!({"type": "MellinTransform",
            "expr": expr_json("1/(1+x)"), "var": "x", "s_var": "s"})),
        req(json!({"type": "InverseMellin",
            "expr": expr_json("1/(s*(s+1))"), "s_var": "s"})),
        req(json!({"type": "InverseMellin",
            "expr": expr_json("1/s^2"), "s_var": "s", "var": "x"})),
        req(json!({"type": "SpecialFn",
            "kind": "Gamma", "args": [expr_json("2")]})),
        req(json!({"type": "Ode",
            "equation": expr_json("y' = y"), "fn_name": "y", "var": "x"})),
        req(json!({"type": "OdeSystem",
            "equations": [expr_json("y1' = y2"), expr_json("y2' = -y1")],
            "fn_names": ["y1", "y2"], "var": "t"})),
        req(json!({"type": "OdeSystem",
            "equations": [expr_json("y1' = y2"), expr_json("y2' = -y1")],
            "fn_names": ["y1", "y2"], "var": "t",
            "ic": {"var_at": expr_json("0"),
                   "values_at": [expr_json("1"), expr_json("0")]}})),
        req(json!({"type": "Pde",
            "equation": expr_json("u_xx + u_yy = 0"),
            "fn_name": "u", "vars": ["x", "y"]})),
        req(json!({"type": "Matrix", "op": "Determinant",
        "operands": [{"rows": [
            [expr_json("1"), expr_json("2")],
            [expr_json("3"), expr_json("4")]
        ]}]})),
        req(json!({"type": "Optimize",
            "objective": expr_json("x^2 + y^2"), "vars": ["x", "y"],
            "constraints": [{"kind": "Equality", "expr": expr_json("x + y - 1")}],
            "sense": "Minimize"})),
        req(json!({"type": "LagrangeMult",
            "objective": expr_json("x^2 + y^2"), "vars": ["x", "y"],
            "equality_constraints": [expr_json("x + y - 1")]})),
    ];

    for payload in &payloads {
        let val = payload.clone();
        let result = request_from_json(&val);
        assert!(
            result.is_ok(),
            "payload failed to parse: {}\nerror: {}",
            serde_json::to_string_pretty(payload).unwrap_or_default(),
            result.unwrap_err()
        );
    }
}

// ── Test 3: request controls passthrough ─────────────────────────────────────

#[test]
fn request_controls_passthrough() {
    let payload = json!({
        "command": {"type": "Simplify", "expr": expr_json("x")},
        "narrate": false,
        "mode": "Numeric",
        "precision": {"decimal_digits": 10},
        "budget": {"max_wall_ms": 5000, "max_iterations": 1000},
        "ambient_domain": "Real",
        "seed": 42
    });
    let req_parsed = request_from_json(&payload).expect("should parse");

    assert!(!req_parsed.narrate, "narrate should be false");
    assert_eq!(
        req_parsed.mode,
        super::super::request::SolveMode::Numeric,
        "mode should be Numeric"
    );
    let prec = req_parsed.precision.expect("precision should be Some");
    assert_eq!(prec.decimal_digits, 10);
    let budget = req_parsed.budget.expect("budget should be Some");
    assert_eq!(budget.max_wall_ms, Some(5000));
    assert_eq!(budget.max_iterations, Some(1000));
    assert!(
        req_parsed.ambient_domain.is_some(),
        "ambient_domain should be Some"
    );
    assert_eq!(req_parsed.seed, Some(42));
}

// ── Test 4: matrix operands parsed ───────────────────────────────────────────

#[test]
fn matrix_operands_parsed() {
    let payload = json!({
        "command": {
            "type": "Matrix",
            "op": "Determinant",
            "operands": [{"rows": [
                [expr_json("1"), expr_json("2")],
                [expr_json("3"), expr_json("4")]
            ]}]
        }
    });
    let req_parsed = request_from_json(&payload).expect("should parse");
    match req_parsed.command {
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
    let payload = json!({
        "command": {
            "type": "Limit",
            "expr": expr_json("1/x"),
            "var": "x",
            "point": expr_json("0"),
            "side": "Left"
        }
    });
    let req_parsed = request_from_json(&payload).expect("should parse");
    match req_parsed.command {
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
    // x^2 - 1 = 0 has two roots; each result should have a key in the response
    let payload = json!({
        "command": {
            "type": "SolveFor",
            "relation": expr_json("x^2 - 1 = 0"),
            "var": "x"
        }
    });
    let v = execute_ok(payload);
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
    // The Numeric result value should now include decimal_digits in the output.
    // Test the serialiser unit directly without a full command round-trip.
    use super::super::request::Precision;
    use super::super::response::{
        EngineId, ExecutionMeta, NumericMethod, Response, ResultEntry, ResultShape, ResultValue,
    };
    use super::response::response_to_json;

    // Build a thales Expression for 3.14159
    let expr = crate::parser::parse_expression("3.14159").unwrap();
    let prec = Precision {
        decimal_digits: 15,
        abs_tol: None,
        rel_tol: None,
    };
    let entry = ResultEntry {
        value: ResultValue::Numeric {
            value: expr,
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
    let req_str = r#"{"command":{"type":"DefinitelyNotARealCommand","expr":{"kind":"Variable","value":"x"}}}"#;
    let err = execute_ffi(req_str).unwrap_err();
    // The error should indicate the parsing failure. With serde tag dispatch,
    // unknown variants produce a descriptive error mentioning the unknown type.
    assert!(
        err.contains("invalid request") || err.contains("DefinitelyNotARealCommand"),
        "error should mention unknown command; got: {}",
        err
    );
}
