//! JSON transport round-trip tests.
//!
//! Each test drives a full request through [`thales::api::json::execute_ffi`]
//! and asserts on the parsed JSON response — verifying that the serde-based
//! JSON parsing and serialisation pipeline carries the expected fields and
//! values end-to-end.

use serde_json::Value;
use thales::api::json::execute_ffi;

// ── Helper ────────────────────────────────────────────────────────────────────

fn run(req: &str) -> Value {
    let resp =
        execute_ffi(req).unwrap_or_else(|e| panic!("execute_ffi failed for `{}`: {}", req, e));
    serde_json::from_str(&resp).unwrap_or_else(|e| panic!("response JSON invalid: {}", e))
}

// ── Category D — 5 JSON transport round-trip tests ───────────────────────────

#[test]
fn golden_json_simplify_roundtrip() {
    // A Simplify request round-trips through JSON: the response carries a
    // Symbolic result whose expr string contains the simplified form.
    let v = run(r#"{"command":{"type":"Simplify","expr":"x + x"}}"#);
    let kind = v["results"][0]["value"]["kind"]
        .as_str()
        .expect("value.kind must be a string");
    assert_eq!(
        kind, "Symbolic",
        "simplify roundtrip: expected Symbolic kind"
    );
    let expr_str = v["results"][0]["value"]["expr"]
        .as_str()
        .expect("value.expr must be a string");
    assert!(
        expr_str.contains('2') && expr_str.contains('x'),
        "simplify(x+x) roundtrip: expected 2*x in expr, got: {}",
        expr_str
    );
}

#[test]
fn golden_json_diff_with_order() {
    // Diff with order:2 — the JSON field is parsed, and the result is a
    // second-order derivative (6*x for x^3).
    let v = run(r#"{"command":{"type":"Diff","expr":"x^3","var":"x","order":2}}"#);
    let kind = v["results"][0]["value"]["kind"]
        .as_str()
        .expect("value.kind must be a string");
    assert_eq!(
        kind, "Symbolic",
        "diff order:2 roundtrip: expected Symbolic"
    );
    let expr_str = v["results"][0]["value"]["expr"]
        .as_str()
        .expect("value.expr must be a string");
    // d²/dx²(x³) = 6x. The simplifier may leave this as "6 * x" or "3 * (2 * x)".
    let has_six = expr_str.contains('6') && expr_str.contains('x');
    let has_three_two = expr_str.contains('3') && expr_str.contains('2') && expr_str.contains('x');
    assert!(
        has_six || has_three_two,
        "diff(x^3, x, order=2): expected 6*x (or equivalent) in expr, got: {}",
        expr_str
    );
    // Verify engine is reported in response.
    let engine = v["results"][0]["engine"]
        .as_str()
        .expect("engine must be a string");
    assert_eq!(engine, "Differentiation", "expected Differentiation engine");
}

#[test]
fn golden_json_solve_system() {
    // SolveSystem for x+y=5, x-y=1 → x=3, y=2.  Two result entries (one per
    // variable) with Labeled structured result.
    let v = run(r#"{
        "command": {
            "type": "SolveSystem",
            "equations": ["x + y = 5", "x - y = 1"],
            "vars": ["x", "y"]
        }
    }"#);
    let results = v["results"].as_array().expect("results must be an array");
    assert_eq!(results.len(), 2, "SolveSystem: expected 2 result entries");
    // Each entry must carry an engine label.
    for entry in results {
        let engine = entry["engine"].as_str().expect("engine must be a string");
        assert_eq!(engine, "SystemSolver", "expected SystemSolver engine");
    }
    // Collect symbolic values and confirm 3 and 2 appear.
    let values: Vec<String> = results
        .iter()
        .filter_map(|e| e["value"]["expr"].as_str().map(str::to_owned))
        .collect();
    let joined = values.join(" ");
    assert!(
        joined.contains('3'),
        "SolveSystem x+y=5, x-y=1: expected x=3 somewhere in {:?}",
        values
    );
    assert!(
        joined.contains('2'),
        "SolveSystem x+y=5, x-y=1: expected y=2 somewhere in {:?}",
        values
    );
}

#[test]
fn golden_json_matrix_lu() {
    // LU decomposition of [[4,3],[6,3]]: response must carry a Matrix engine
    // label and exactly 8 flat cell entries (primary + 7 alternatives for L+U).
    let v = run(r#"{
        "command": {
            "type": "Matrix",
            "op": "Lu",
            "operands": [{"rows": [["4","3"],["6","3"]]}]
        }
    }"#);
    let engine = v["results"][0]["engine"]
        .as_str()
        .expect("engine must be a string");
    assert_eq!(engine, "Matrix", "LU decomposition: expected Matrix engine");
    let alternatives = v["results"][0]["alternatives"]
        .as_array()
        .expect("alternatives must be an array");
    // 2×2 L (4 cells) + 2×2 U (4 cells) = 8 cells: primary + 7 alternatives.
    assert_eq!(
        alternatives.len(),
        7,
        "LU of 2×2 matrix: expected 7 alternatives (8 cells total)"
    );
}

#[test]
fn golden_json_all_request_fields() {
    // All optional request-level fields — precision, budget, ambient_domain,
    // seed, narrate, mode — parse and pass through without error.
    let payload = r#"{
        "command": {"type": "Simplify", "expr": "x + x"},
        "narrate": false,
        "mode": "Symbolic",
        "precision": {"decimal_digits": 12},
        "budget": {"max_wall_ms": 10000, "max_iterations": 500},
        "ambient_domain": "Real",
        "seed": 99
    }"#;
    let v = run(payload);
    // A clean response with at least one result confirms all fields parsed.
    let results = v["results"].as_array().expect("results must be an array");
    assert!(
        !results.is_empty(),
        "all-fields request: expected at least one result"
    );
    let kind = results[0]["value"]["kind"]
        .as_str()
        .expect("value.kind must be a string");
    assert_eq!(
        kind, "Symbolic",
        "all-fields simplify: expected Symbolic result"
    );
}
