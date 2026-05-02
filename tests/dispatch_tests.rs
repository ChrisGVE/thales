//! Per-Command golden tests for the single-entry dispatcher.
//!
//! Each test builds a [`Request`] for one [`Command`] variant, invokes
//! [`execute`], and asserts that the [`Response`] carries the expected
//! shape (value kind, engine id, step count, diagnostic codes). Tests
//! for commands that still route to the `NotImplemented` diagnostic
//! assert on the diagnostic shape rather than a symbolic value.

use thales::api::command::{
    Command, IvpData, LimitPoint, MatrixExpr as ApiMatrixExpr, MatrixOp, SimplifyRules, SpecialKind,
};
use thales::api::diagnostic::{DiagnosticCode, Severity};
use thales::api::domain::Domain;
use thales::api::execute;
use thales::api::json::execute_ffi;
use thales::api::request::{Budget, Request, SolveMode};
use thales::api::response::{EngineId, ResultKey, ResultValue, StructuredResult};
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

/// Parse a plain-text math expression and return its mathlex serde JSON form.
fn expr_json(input: &str) -> serde_json::Value {
    let ml = mathlex::parse(input)
        .unwrap_or_else(|e| panic!("expr_json: failed to parse `{}`: {:?}", input, e));
    serde_json::to_value(&ml)
        .unwrap_or_else(|e| panic!("expr_json: failed to serialise `{}`: {}", input, e))
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

// ── Higher-dimensional calculus (F1b) ────────────────────────────────────────

#[test]
fn total_diff_chain_rule() {
    // f = x² + y²; var = x; deps = [(y, x)] → df/dx = 2x + 2y.
    let expr = add(pow(var("x"), int(2)), pow(var("y"), int(2)));
    let resp = execute(request(Command::TotalDiff {
        expr,
        var: "x".to_string(),
        deps: vec![("y".to_string(), var("x"))],
    }))
    .unwrap();
    assert_single_symbolic(&resp, EngineId::Differentiation);
    // Expect 2x + 2y; equality of simplified compound expressions is fragile,
    // so check structural shape: an Add of two terms each linear in 2.
    if let ResultValue::Symbolic(e) = &resp.results[0].1.value {
        let s = format!("{}", e);
        assert!(
            s.contains("2") && s.contains("x") && s.contains("y"),
            "expected 2x + 2y form, got {}",
            s
        );
    }
}

#[test]
fn divergence_three_dim_identity_field() {
    // ∇·(x, y, z) = 3.
    let resp = execute(request(Command::Divergence {
        field: vec![var("x"), var("y"), var("z")],
        vars: vec!["x".to_string(), "y".to_string(), "z".to_string()],
    }))
    .unwrap();
    assert_single_symbolic(&resp, EngineId::Differentiation);
    if let ResultValue::Symbolic(e) = &resp.results[0].1.value {
        assert_eq!(*e, int(3), "expected divergence == 3, got {}", e);
    }
}

#[test]
fn divergence_arity_mismatch_returns_engine_error() {
    let resp = execute(request(Command::Divergence {
        field: vec![var("x"), var("y")],
        vars: vec!["x".to_string(), "y".to_string(), "z".to_string()],
    }))
    .unwrap();
    assert!(
        resp.diagnostics
            .iter()
            .any(|d| matches!(d.code, DiagnosticCode::Other(s) if s == "engine-error")),
        "expected engine-error diagnostic for arity mismatch"
    );
}

#[test]
fn curl_three_dim_planar_field() {
    // F = (y, -x, 0). ∇×F = (0, 0, -2).
    let neg_x = sub(int(0), var("x"));
    let resp = execute(request(Command::Curl {
        field: vec![var("y"), neg_x, int(0)],
        vars: vec!["x".to_string(), "y".to_string(), "z".to_string()],
    }))
    .unwrap();
    let entry = &resp.results[0].1;
    assert_eq!(entry.engine, EngineId::Differentiation);
    assert_eq!(entry.alternatives.len(), 2);
    if let ResultValue::Symbolic(primary) = &entry.value {
        assert_eq!(
            *primary,
            int(0),
            "curl x-component expected 0, got {}",
            primary
        );
    }
    assert_eq!(
        entry.alternatives[0],
        int(0),
        "curl y-component expected 0, got {}",
        entry.alternatives[0]
    );
    // z-component: -2 (could be Integer(-2) or Unary(Neg, Integer(2)) — accept either).
    let cz = format!("{}", entry.alternatives[1]);
    assert!(
        cz.contains("2") && cz.contains('-'),
        "curl z-component expected -2, got {}",
        cz
    );
}

#[test]
fn curl_arity_mismatch_returns_engine_error() {
    let resp = execute(request(Command::Curl {
        field: vec![var("x"), var("y")],
        vars: vec!["x".to_string(), "y".to_string()],
    }))
    .unwrap();
    assert!(
        resp.diagnostics
            .iter()
            .any(|d| matches!(d.code, DiagnosticCode::Other(s) if s == "engine-error")),
        "expected engine-error diagnostic for non-3D curl"
    );
}

#[test]
fn laplacian_two_dim_quadratic() {
    // ∇²(x² + y²) = 4.
    let expr = add(pow(var("x"), int(2)), pow(var("y"), int(2)));
    let resp = execute(request(Command::Laplacian {
        expr,
        vars: vec!["x".to_string(), "y".to_string()],
    }))
    .unwrap();
    assert_single_symbolic(&resp, EngineId::Differentiation);
    if let ResultValue::Symbolic(e) = &resp.results[0].1.value {
        assert_eq!(*e, int(4), "expected laplacian == 4, got {}", e);
    }
}

#[test]
fn jacobian_two_two_polynomial() {
    // F = (x², x*y); J = [[2x, 0], [y, x]].
    let resp = execute(request(Command::Jacobian {
        fields: vec![pow(var("x"), int(2)), mul(var("x"), var("y"))],
        vars: vec!["x".to_string(), "y".to_string()],
    }))
    .unwrap();
    let entry = &resp.results[0].1;
    assert_eq!(entry.engine, EngineId::Differentiation);
    // 2×2 Jacobian: primary J[0][0] + 3 alternatives.
    assert_eq!(entry.alternatives.len(), 3);
    assert_eq!(entry.alternatives[0], int(0), "J[0][1] expected 0");
    assert_eq!(entry.alternatives[1], var("y"), "J[1][0] expected y");
    assert_eq!(entry.alternatives[2], var("x"), "J[1][1] expected x");
}

#[test]
fn hessian_two_two_quadratic() {
    // f = x² + y²; H = [[2, 0], [0, 2]].
    let expr = add(pow(var("x"), int(2)), pow(var("y"), int(2)));
    let resp = execute(request(Command::Hessian {
        expr,
        vars: vec!["x".to_string(), "y".to_string()],
    }))
    .unwrap();
    let entry = &resp.results[0].1;
    assert_eq!(entry.engine, EngineId::Differentiation);
    // 2×2 Hessian: primary H[0][0] + 3 alternatives.
    assert_eq!(entry.alternatives.len(), 3);
    if let ResultValue::Symbolic(primary) = &entry.value {
        assert_eq!(*primary, int(2), "H[0][0] expected 2");
    }
    assert_eq!(entry.alternatives[0], int(0), "H[0][1] expected 0");
    assert_eq!(entry.alternatives[1], int(0), "H[1][0] expected 0");
    assert_eq!(entry.alternatives[2], int(2), "H[1][1] expected 2");
}

#[test]
fn directional_diff_unit_diagonal() {
    // ∇(x²+y²) · (1, 1) = 2x + 2y.
    let expr = add(pow(var("x"), int(2)), pow(var("y"), int(2)));
    let resp = execute(request(Command::DirectionalDiff {
        expr,
        vars: vec!["x".to_string(), "y".to_string()],
        direction: vec![int(1), int(1)],
    }))
    .unwrap();
    assert_single_symbolic(&resp, EngineId::Differentiation);
    if let ResultValue::Symbolic(e) = &resp.results[0].1.value {
        let s = format!("{}", e);
        assert!(
            s.contains("2") && s.contains("x") && s.contains("y"),
            "expected 2x + 2y form, got {}",
            s
        );
    }
}

// ── Series expansions (F1c) ──────────────────────────────────────────────────

#[test]
fn taylor_exp_around_zero_order_4() {
    // exp(x) = 1 + x + x²/2 + x³/6 + x⁴/24 truncated at order 4.
    let expr = Expression::Function(Function::Exp, vec![var("x")]);
    let resp = execute(request(Command::Taylor {
        expr,
        var: "x".to_string(),
        center: int(0),
        order: 4,
    }))
    .unwrap();
    assert_single_symbolic(&resp, EngineId::TaylorExpansion);
    if let ResultValue::Symbolic(e) = &resp.results[0].1.value {
        // The reassembled series carries powers of x up to 4 plus a constant 1.
        let s = format!("{}", e);
        assert!(s.contains("x"), "expected polynomial in x, got {}", s);
    }
}

#[test]
fn taylor_sin_around_zero_order_3() {
    // sin(x) = x - x³/6 + O(x⁵). Truncated at order 3 reproduces x - x³/6.
    let expr = Expression::Function(Function::Sin, vec![var("x")]);
    let resp = execute(request(Command::Taylor {
        expr,
        var: "x".to_string(),
        center: int(0),
        order: 3,
    }))
    .unwrap();
    assert_single_symbolic(&resp, EngineId::TaylorExpansion);
}

#[test]
fn laurent_around_zero() {
    // 1/x at center 0 with neg/pos order 2 should produce a series whose
    // reassembled expression contains a 1/x term.
    let expr = Expression::Binary(
        thales::ast::BinaryOp::Div,
        Box::new(int(1)),
        Box::new(var("x")),
    );
    let resp = execute(request(Command::Laurent {
        expr,
        var: "x".to_string(),
        center: int(0),
        order: 2,
    }))
    .unwrap();
    assert_eq!(resp.results[0].1.engine, EngineId::LaurentExpansion);
}

#[test]
fn asymptotic_polynomial_to_infinity() {
    // x² + x as x → ∞ — leading term is x²; series reassembly should expose
    // a power of x.
    let expr = add(pow(var("x"), int(2)), var("x"));
    let resp = execute(request(Command::Asymptotic {
        expr,
        var: "x".to_string(),
        order: 3,
    }))
    .unwrap();
    assert_single_symbolic(&resp, EngineId::AsymptoticExpansion);
}

#[test]
fn compose_taylor_series() {
    // outer = exp(x), inner = x (with a_0=0). exp(x) ∘ x = exp(x), still a
    // valid Taylor series of order 4.
    let outer = Expression::Function(Function::Exp, vec![var("x")]);
    let inner = var("x");
    let resp = execute(request(Command::Compose {
        outer,
        inner,
        var: "x".to_string(),
        order: 4,
    }))
    .unwrap();
    assert_single_symbolic(&resp, EngineId::SeriesComposition);
}

#[test]
fn compose_rejects_nonzero_inner_constant() {
    // inner = 1 has a_0 = 1 ≠ 0, so composition must fail with an engine
    // error.
    let outer = Expression::Function(Function::Exp, vec![var("x")]);
    let inner = int(1);
    let resp = execute(request(Command::Compose {
        outer,
        inner,
        var: "x".to_string(),
        order: 3,
    }))
    .unwrap();
    assert!(
        resp.diagnostics
            .iter()
            .any(|d| matches!(d.code, DiagnosticCode::Other(s) if s == "engine-error")),
        "expected engine-error for non-zero inner constant"
    );
}

#[test]
fn revert_sin_series() {
    // sin(x) Taylor series has a_0=0, a_1=1, so reversion succeeds. The
    // inverse is arcsin(x) ≈ x + x³/6 + 3x⁵/40 + ...
    let expr = Expression::Function(Function::Sin, vec![var("x")]);
    let resp = execute(request(Command::Revert {
        expr,
        var: "x".to_string(),
        order: 3,
    }))
    .unwrap();
    assert_single_symbolic(&resp, EngineId::SeriesReversion);
}

#[test]
fn revert_rejects_nonzero_constant() {
    // Reversion requires a_0 = 0; `x + 1` has a_0 = 1 → engine error.
    let expr = add(var("x"), int(1));
    let resp = execute(request(Command::Revert {
        expr,
        var: "x".to_string(),
        order: 3,
    }))
    .unwrap();
    assert!(
        resp.diagnostics
            .iter()
            .any(|d| matches!(d.code, DiagnosticCode::Other(s) if s == "engine-error")),
        "expected engine-error for non-revertible series"
    );
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
fn taylor_returns_taylor_engine() {
    // Taylor is now wired; routing must surface the Taylor engine.
    let resp = execute(request(Command::Taylor {
        expr: Expression::Function(Function::Sin, vec![var("x")]),
        var: "x".to_string(),
        center: int(0),
        order: 5,
    }))
    .unwrap();
    assert_eq!(resp.results[0].1.engine, EngineId::TaylorExpansion);
}

#[test]
fn special_fn_gamma_integer() {
    // Γ(5) = 24.
    let resp = execute(request(Command::SpecialFn {
        kind: SpecialKind::Gamma,
        args: vec![int(5)],
    }))
    .unwrap();
    assert_eq!(resp.results[0].1.engine, EngineId::SpecialFunctions);
    if let ResultValue::Symbolic(e) = &resp.results[0].1.value {
        assert_eq!(*e, int(24), "Γ(5) expected 24, got {}", e);
    }
}

#[test]
fn special_fn_beta_integers() {
    // B(2, 3) = 1/12.
    let resp = execute(request(Command::SpecialFn {
        kind: SpecialKind::Beta,
        args: vec![int(2), int(3)],
    }))
    .unwrap();
    assert_eq!(resp.results[0].1.engine, EngineId::SpecialFunctions);
}

#[test]
fn special_fn_erf_zero() {
    // erf(0) = 0.
    let resp = execute(request(Command::SpecialFn {
        kind: SpecialKind::Erf,
        args: vec![int(0)],
    }))
    .unwrap();
    assert_eq!(resp.results[0].1.engine, EngineId::SpecialFunctions);
}

#[test]
fn special_fn_erfc_zero() {
    // erfc(0) = 1.
    let resp = execute(request(Command::SpecialFn {
        kind: SpecialKind::Erfc,
        args: vec![int(0)],
    }))
    .unwrap();
    assert_eq!(resp.results[0].1.engine, EngineId::SpecialFunctions);
}

#[test]
fn special_fn_arity_mismatch_returns_engine_error() {
    // Gamma takes 1 arg; supplying 2 must trip the arity check.
    let resp = execute(request(Command::SpecialFn {
        kind: SpecialKind::Gamma,
        args: vec![int(1), int(2)],
    }))
    .unwrap();
    assert!(
        resp.diagnostics
            .iter()
            .any(|d| matches!(d.code, DiagnosticCode::Other(s) if s == "engine-error")),
        "expected engine-error diagnostic for SpecialFn arity mismatch"
    );
}

#[test]
fn residue_simple_pole() {
    // Residue of 1/z at z=0 is 1.
    let expr = Expression::Binary(BinaryOp::Div, Box::new(int(1)), Box::new(var("z")));
    let resp = execute(request(Command::Residue {
        expr,
        var: "z".to_string(),
        point: int(0),
    }))
    .unwrap();
    assert_eq!(resp.results[0].1.engine, EngineId::Residue);
    if let ResultValue::Symbolic(e) = &resp.results[0].1.value {
        assert_eq!(*e, int(1), "residue of 1/z at 0 expected 1, got {}", e);
    }
}

#[test]
fn residue_at_regular_point() {
    // Residue at a regular point is 0.  1/z at z=2 is regular.
    let expr = Expression::Binary(BinaryOp::Div, Box::new(int(1)), Box::new(var("z")));
    let resp = execute(request(Command::Residue {
        expr,
        var: "z".to_string(),
        point: int(2),
    }))
    .unwrap();
    assert_eq!(resp.results[0].1.engine, EngineId::Residue);
    if let ResultValue::Symbolic(e) = &resp.results[0].1.value {
        assert_eq!(*e, int(0), "residue at regular point expected 0, got {}", e);
    }
}

#[test]
fn residue_returns_residue_engine() {
    // Residue is now wired; routing must surface the Residue engine.
    let expr = Expression::Binary(BinaryOp::Div, Box::new(int(1)), Box::new(var("z")));
    let resp = execute(request(Command::Residue {
        expr,
        var: "z".to_string(),
        point: int(0),
    }))
    .unwrap();
    assert_eq!(resp.results[0].1.engine, EngineId::Residue);
}

// ── Matrix operations (F1e) ──────────────────────────────────────────────────

fn m22(a: i64, b: i64, c: i64, d: i64) -> ApiMatrixExpr {
    ApiMatrixExpr::Matrix(vec![vec![int(a), int(b)], vec![int(c), int(d)]])
}

#[test]
fn matrix_determinant_identity() {
    // det([[1,0],[0,1]]) = 1.
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::Determinant,
        operands: vec![m22(1, 0, 0, 1)],
    }))
    .unwrap();
    assert_single_symbolic(&resp, EngineId::Matrix);
    if let ResultValue::Symbolic(e) = &resp.results[0].1.value {
        assert_eq!(*e, int(1), "expected determinant == 1, got {}", e);
    }
}

#[test]
fn matrix_transpose_two_two() {
    // [[1,2],[3,4]]ᵀ = [[1,3],[2,4]]; primary=1, alternatives=[3,2,4].
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::Transpose,
        operands: vec![m22(1, 2, 3, 4)],
    }))
    .unwrap();
    let entry = &resp.results[0].1;
    assert_eq!(entry.engine, EngineId::Matrix);
    assert_eq!(entry.alternatives.len(), 3);
}

#[test]
fn matrix_addition() {
    // [[1,2],[3,4]] + [[5,6],[7,8]] = [[6,8],[10,12]]
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::Add,
        operands: vec![m22(1, 2, 3, 4), m22(5, 6, 7, 8)],
    }))
    .unwrap();
    let entry = &resp.results[0].1;
    assert_eq!(entry.engine, EngineId::Matrix);
    assert_eq!(entry.alternatives.len(), 3);
}

#[test]
fn matrix_multiplication() {
    // [[1,2],[3,4]] * [[1,0],[0,1]] = [[1,2],[3,4]]
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::Multiply,
        operands: vec![m22(1, 2, 3, 4), m22(1, 0, 0, 1)],
    }))
    .unwrap();
    let entry = &resp.results[0].1;
    assert_eq!(entry.engine, EngineId::Matrix);
    if let ResultValue::Symbolic(primary) = &entry.value {
        assert_eq!(*primary, int(1), "(0,0) entry expected 1, got {}", primary);
    }
}

#[test]
fn matrix_trace() {
    // trace([[1,2],[3,4]]) = 5.
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::Trace,
        operands: vec![m22(1, 2, 3, 4)],
    }))
    .unwrap();
    assert_single_symbolic(&resp, EngineId::Matrix);
    if let ResultValue::Symbolic(e) = &resp.results[0].1.value {
        assert_eq!(*e, int(5), "expected trace == 5, got {}", e);
    }
}

#[test]
fn matrix_inverse_two_two() {
    // [[1,2],[3,4]]⁻¹ exists (det = -2). The result is a 2x2 matrix; we
    // verify routing rather than exact symbolic form (which depends on
    // simplifier output).
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::Inverse,
        operands: vec![m22(1, 2, 3, 4)],
    }))
    .unwrap();
    let entry = &resp.results[0].1;
    assert_eq!(entry.engine, EngineId::Matrix);
    assert_eq!(entry.alternatives.len(), 3);
}

#[test]
fn matrix_eigenvalues_diagonal() {
    // Eigenvalues of diag(2, 3) are {2, 3}.
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::Eigenvalues,
        operands: vec![m22(2, 0, 0, 3)],
    }))
    .unwrap();
    let entry = &resp.results[0].1;
    assert_eq!(entry.engine, EngineId::Matrix);
    // Two eigenvalues: primary + 1 alternative.
    assert_eq!(entry.alternatives.len(), 1);
}

#[test]
fn matrix_lu_decomposition() {
    // LU of [[4,3],[6,3]]: returned as (L, U) flattened.
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::Lu,
        operands: vec![m22(4, 3, 6, 3)],
    }))
    .unwrap();
    let entry = &resp.results[0].1;
    assert_eq!(entry.engine, EngineId::Matrix);
    // 2x2 L (4 cells) + 2x2 U (4 cells) = 8 cells; primary + 7 alternatives.
    assert_eq!(entry.alternatives.len(), 7);
}

#[test]
fn matrix_rank_returns_engine_error() {
    // Rank engine not yet implemented; must surface an engine error.
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::Rank,
        operands: vec![m22(1, 2, 3, 4)],
    }))
    .unwrap();
    assert!(
        resp.diagnostics
            .iter()
            .any(|d| matches!(d.code, DiagnosticCode::Other(s) if s == "engine-error")),
        "expected engine-error diagnostic for unimplemented Rank"
    );
}

// ── Optimization (F1f) ───────────────────────────────────────────────────────

#[test]
fn optimize_unconstrained_paraboloid_minimum() {
    // f(x,y) = x² + y² has its global minimum 0 at (0,0).
    let objective = add(pow(var("x"), int(2)), pow(var("y"), int(2)));
    let resp = execute(request(Command::Optimize {
        objective,
        vars: vec!["x".to_string(), "y".to_string()],
        constraints: Vec::new(),
        sense: thales::api::command::OptSense::Minimize,
    }))
    .unwrap();
    assert_eq!(resp.results[0].1.engine, EngineId::Optimizer);
    if let ResultValue::Symbolic(Expression::Float(obj)) = &resp.results[0].1.value {
        assert!(
            obj.abs() < 1e-3,
            "expected minimum near 0, got objective {}",
            obj
        );
    }
}

#[test]
fn optimize_inequality_constraint_returns_engine_error() {
    // The v0.9.0 optimiser handles equality KKT only; an inequality
    // constraint must surface as an engine error.
    let resp = execute(request(Command::Optimize {
        objective: pow(var("x"), int(2)),
        vars: vec!["x".to_string()],
        constraints: vec![thales::api::command::Constraint::LessEq(var("x"))],
        sense: thales::api::command::OptSense::Minimize,
    }))
    .unwrap();
    assert!(
        resp.diagnostics
            .iter()
            .any(|d| matches!(d.code, DiagnosticCode::Other(s) if s == "engine-error")),
        "expected engine-error diagnostic for inequality constraint"
    );
}

#[test]
fn lagrange_mult_paraboloid_on_line() {
    // Minimise x²+y² subject to x+y-1 = 0 → optimum at (1/2, 1/2),
    // objective value 1/2.
    let objective = add(pow(var("x"), int(2)), pow(var("y"), int(2)));
    let constraint = add(add(var("x"), var("y")), int(-1));
    let resp = execute(request(Command::LagrangeMult {
        objective,
        vars: vec!["x".to_string(), "y".to_string()],
        equality_constraints: vec![constraint],
    }))
    .unwrap();
    assert_eq!(resp.results[0].1.engine, EngineId::Optimizer);
    if let ResultValue::Symbolic(Expression::Float(obj)) = &resp.results[0].1.value {
        assert!(
            (*obj - 0.5).abs() < 1e-3,
            "expected objective 1/2, got {}",
            obj
        );
    }
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
fn matrix_returns_matrix_engine() {
    // Matrix is now wired; routing must surface the Matrix engine.
    // Determinant of [[1,0],[0,1]] = 1.
    let i2 =
        thales::api::command::MatrixExpr::Matrix(vec![vec![int(1), int(0)], vec![int(0), int(1)]]);
    let resp = execute(request(Command::Matrix {
        op: thales::api::command::MatrixOp::Determinant,
        operands: vec![i2],
    }))
    .unwrap();
    assert_eq!(resp.results[0].1.engine, EngineId::Matrix);
}

#[test]
fn optimize_returns_optimizer_engine() {
    // Optimize is now wired; routing must surface the Optimizer engine.
    // f(x,y) = x² + y² has its global minimum at the origin.
    let objective = add(pow(var("x"), int(2)), pow(var("y"), int(2)));
    let resp = execute(request(Command::Optimize {
        objective,
        vars: vec!["x".to_string(), "y".to_string()],
        constraints: Vec::new(),
        sense: thales::api::command::OptSense::Minimize,
    }))
    .unwrap();
    assert_eq!(resp.results[0].1.engine, EngineId::Optimizer);
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
    let payload = serde_json::json!({
        "command": {"type": "Simplify", "expr": expr_json("x + x")}
    });
    let resp = execute_ffi(&serde_json::to_string(&payload).unwrap()).unwrap();
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["results"][0]["value"]["kind"], "Symbolic");
}

#[test]
fn ffi_round_trip_diff_integrate_inverses() {
    // d/dx(x^2) then ∫ that dx should round-trip to a polynomial.
    let diff_payload = serde_json::json!({
        "command": {"type": "Diff", "expr": expr_json("x^2"), "var": "x", "order": 1}
    });
    let resp = execute_ffi(&serde_json::to_string(&diff_payload).unwrap()).unwrap();
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    // Response now emits structured mathlex Expression JSON; extract it for use
    // as input to a second request.
    let derivative_expr = v["results"][0]["value"]["expr"].clone();
    assert!(
        !derivative_expr.is_null(),
        "derivative expr must not be null"
    );

    let integrate_payload = serde_json::json!({
        "command": {"type": "Integrate", "expr": derivative_expr, "var": "x"}
    });
    let resp2 = execute_ffi(&serde_json::to_string(&integrate_payload).unwrap()).unwrap();
    let v2: serde_json::Value = serde_json::from_str(&resp2).unwrap();
    assert_eq!(v2["results"][0]["engine"], "PatternIntegration");
}

#[test]
fn ffi_round_trip_solve_for() {
    let payload = serde_json::json!({
        "command": {"type": "SolveFor", "relation": expr_json("2*x + 3 = 7"), "var": "x"}
    });
    let resp = execute_ffi(&serde_json::to_string(&payload).unwrap()).unwrap();
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["results"][0]["engine"], "EquationSolver");
}

#[test]
fn ffi_round_trip_solve_system() {
    let payload = serde_json::json!({
        "command": {
            "type": "SolveSystem",
            "equations": [expr_json("x + y = 5"), expr_json("x - y = 1")],
            "vars": ["x", "y"]
        }
    });
    let resp = execute_ffi(&serde_json::to_string(&payload).unwrap()).unwrap();
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    let results = v["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn ffi_round_trip_ode() {
    let payload = serde_json::json!({
        "command": {"type": "Ode", "equation": expr_json("y"), "fn_name": "y", "var": "x"}
    });
    let resp = execute_ffi(&serde_json::to_string(&payload).unwrap()).unwrap();
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["results"][0]["engine"], "OdeFirstOrder");
}

#[test]
fn ffi_round_trip_parse_uses_parser() {
    // Confirm that the JSON transport round-trips through parse_expression.
    let payload = serde_json::json!({
        "command": {"type": "Simplify", "expr": expr_json("sin(x)^2 + cos(x)^2")}
    });
    let resp = execute_ffi(&serde_json::to_string(&payload).unwrap()).unwrap();
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
fn dispatch_resolves_noop_diagnostic_narrative() {
    let resp = execute(Request::default()).unwrap();
    let diag = &resp.diagnostics[0];
    assert_eq!(diag.code, DiagnosticCode::NotImplemented);
    assert_eq!(diag.severity, Severity::Error);
    assert_eq!(diag.narrative.fallback_md, dict_entry("command.noop"));
}

#[test]
fn ffi_round_trip_carries_resolved_narrative() {
    // The FFI surface goes through the same dispatcher as api::execute and inherits
    // render_response. The resolved Markdown must reach the JSON.
    // Matrix (Determinant) is not yet wired; use it to test the narrative path.
    let req = r#"{"command":{"type":"Matrix","op":"Determinant"}}"#;
    let resp = execute_ffi(req).unwrap();
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    let diag_md = v["diagnostics"][0]["fallback_md"]
        .as_str()
        .expect("diagnostic fallback_md must serialise as a string");
    assert_eq!(diag_md, dict_entry("command.matrix"));
}

// ── Phase 0A: C-11 diff order 0 ────────────────────────────────────────────

#[test]
fn diff_order_zero_returns_original() {
    let req = request(Command::Diff {
        expr: add(pow(var("x"), int(2)), int(1)),
        var: "x".to_string(),
        order: 0,
    });
    let resp = execute(req).unwrap();
    assert_eq!(resp.results.len(), 1);
    let (_, entry) = &resp.results[0];
    if let ResultValue::Symbolic(e) = &entry.value {
        assert_ne!(
            format!("{:?}", e),
            format!("{:?}", mul(int(2), var("x"))),
            "order 0 should not differentiate"
        );
    }
}

#[test]
fn partial_diff_order_zero_returns_original() {
    let req = request(Command::PartialDiff {
        expr: add(pow(var("x"), int(2)), pow(var("y"), int(2))),
        vars: vec![("x".to_string(), 0)],
    });
    let resp = execute(req).unwrap();
    assert_eq!(resp.results.len(), 1);
}

// ── Phase 0A: C-10 limit point error ───────────────────────────────────────

#[test]
fn limit_at_numeric_point_works() {
    let req = request(Command::Limit {
        expr: pow(var("x"), int(2)),
        var: "x".to_string(),
        point: LimitPoint::Finite(int(2)),
        side: None,
    });
    let resp = execute(req).unwrap();
    assert!(!resp.results.is_empty());
}

#[test]
fn limit_at_symbolic_point_returns_error() {
    let req = request(Command::Limit {
        expr: pow(var("x"), int(2)),
        var: "x".to_string(),
        point: LimitPoint::Finite(var("z")),
        side: None,
    });
    let resp = execute(req).unwrap();
    assert!(
        !resp.diagnostics.is_empty(),
        "symbolic limit point should produce a diagnostic"
    );
}

// ── Phase 0A: C-12 u64 overflow, C-13 ic error, C-14 derivatives_at ───────

#[test]
fn json_rejects_oversized_diff_order() {
    // 4294967297 = 2^32 + 1 exceeds u32::MAX; must be rejected.
    // The serde-based parser emits "invalid value: integer `N`, expected u32".
    let payload = serde_json::json!({
        "command": {"type": "Diff", "expr": expr_json("x^2"), "var": "x", "order": 4294967297u64}
    });
    let result = execute_ffi(&serde_json::to_string(&payload).unwrap());
    assert!(
        result.is_err(),
        "oversized order must be rejected; got Ok: {}",
        result.unwrap()
    );
}

#[test]
fn json_accepts_valid_diff_order() {
    let payload = serde_json::json!({
        "command": {"type": "Diff", "expr": expr_json("x^2"), "var": "x", "order": 3}
    });
    let result = execute_ffi(&serde_json::to_string(&payload).unwrap());
    assert!(result.is_ok());
}

#[test]
fn json_ode_missing_ic_is_ok() {
    let payload = serde_json::json!({
        "command": {"type": "Ode", "equation": expr_json("y"), "fn_name": "y", "var": "x"}
    });
    let result = execute_ffi(&serde_json::to_string(&payload).unwrap());
    assert!(result.is_ok());
}

// ── Phase 0A: C-15 JSON Matrix operands ────────────────────────────────────

#[test]
fn json_matrix_determinant_with_operand() {
    let payload = serde_json::json!({
        "command": {
            "type": "Matrix",
            "op": "Determinant",
            "operands": [{"rows": [
                [expr_json("1"), expr_json("2")],
                [expr_json("3"), expr_json("4")]
            ]}]
        }
    });
    let result = execute_ffi(&serde_json::to_string(&payload).unwrap());
    assert!(result.is_ok(), "should parse matrix operand: {:?}", result);
}

#[test]
fn json_matrix_no_operands_still_works() {
    let json = r#"{"command":{"type":"Matrix","op":"Determinant"}}"#;
    let result = execute_ffi(json);
    assert!(result.is_ok());
}

// ── Phase 0B Stream 1: StructuredResult golden tests ─────────────────────────

#[test]
fn structured_solve_system_labels() {
    // SolveSystem should produce Labeled structured entries for each variable.
    let resp = execute(request(Command::SolveSystem {
        equations: vec![
            parse_expression("x+y-3").unwrap(),
            parse_expression("x-y-1").unwrap(),
        ],
        vars: vec!["x".to_string(), "y".to_string()],
        over: Domain::real(),
    }))
    .unwrap();

    assert!(!resp.results.is_empty(), "expected at least one result");
    let labeled_count = resp
        .results
        .iter()
        .filter(|(_, e)| matches!(&e.structured, Some(StructuredResult::Labeled { .. })))
        .count();
    assert_eq!(labeled_count, 2, "expected two Labeled entries (x and y)");

    // Each Labeled entry should have the variable name as label.
    let labels: Vec<&str> = resp
        .results
        .iter()
        .filter_map(|(_, e)| match &e.structured {
            Some(StructuredResult::Labeled { label, .. }) => Some(label.as_str()),
            _ => None,
        })
        .collect();
    assert!(labels.contains(&"x"), "expected label 'x'");
    assert!(labels.contains(&"y"), "expected label 'y'");
}

#[test]
fn structured_multiple_roots() {
    // SolveFor x^2 + x - 6 = 0 (roots: x=2, x=-3).
    // The symbolic isolation path cannot handle mixed degree-1/degree-2;
    // the quadratic solver fires and returns Solution::Multiple, which the
    // dispatcher wraps as Branches on the first result entry.
    let relation = add(sub(pow(var("x"), int(2)), int(6)), var("x"));
    let resp = execute(request(Command::SolveFor {
        relation,
        var: "x".to_string(),
        over: Domain::real(),
    }))
    .unwrap();

    // Expect two results (both roots).
    assert_eq!(resp.results.len(), 2, "expected two roots for x^2+x-6=0");

    // The first result entry carries the Branches structured data.
    let first_structured = &resp.results[0].1.structured;
    assert!(
        matches!(first_structured, Some(StructuredResult::Branches { .. })),
        "expected Branches on first result, got {:?}",
        first_structured
    );
    if let Some(StructuredResult::Branches { branches }) = first_structured {
        assert_eq!(branches.len(), 2, "expected two branch entries");
        assert_eq!(
            branches[0].label.as_deref(),
            Some("root_1"),
            "first branch label should be root_1"
        );
        assert_eq!(
            branches[1].label.as_deref(),
            Some("root_2"),
            "second branch label should be root_2"
        );
    }
}

#[test]
fn structured_lu_decomposition() {
    // LU of a 2×2 matrix should produce a Decomposition with L, U, P parts.
    let matrix = ApiMatrixExpr::Matrix(vec![vec![int(2), int(1)], vec![int(4), int(3)]]);
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::Lu,
        operands: vec![matrix],
    }))
    .unwrap();

    assert_eq!(resp.results.len(), 1, "expected one result");
    let structured = &resp.results[0].1.structured;
    assert!(
        matches!(structured, Some(StructuredResult::Decomposition { .. })),
        "expected Decomposition, got {:?}",
        structured
    );
    if let Some(StructuredResult::Decomposition { parts }) = structured {
        let part_names: Vec<&str> = parts.iter().map(|(n, _)| n.as_str()).collect();
        assert!(part_names.contains(&"L"), "expected L part");
        assert!(part_names.contains(&"U"), "expected U part");
        assert!(part_names.contains(&"P"), "expected P part");
    }
}

#[test]
fn structured_eigenvalues_pairing() {
    // Eigenvectors of a 2×2 diagonal matrix should produce a Decomposition
    // with eigenvalue-eigenvector pair parts.
    let matrix = ApiMatrixExpr::Matrix(vec![vec![int(2), int(0)], vec![int(0), int(3)]]);
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::Eigenvectors,
        operands: vec![matrix],
    }))
    .unwrap();

    assert_eq!(resp.results.len(), 1, "expected one result");
    let structured = &resp.results[0].1.structured;
    assert!(
        matches!(structured, Some(StructuredResult::Decomposition { .. })),
        "expected Decomposition for eigenpairs, got {:?}",
        structured
    );
    if let Some(StructuredResult::Decomposition { parts }) = structured {
        // Expect pair_1_eigenvalue, pair_1_eigenvector, pair_2_eigenvalue, …
        assert!(
            parts.len() >= 4,
            "expected at least 4 parts for 2 eigenpairs, got {}",
            parts.len()
        );
    }
}

#[test]
fn structured_optimize_labels() {
    // Optimize a simple unconstrained quadratic: min (x-1)^2 + (y-2)^2
    // The optimum is x=1, y=2; expect Labeled entries.
    let objective = add(
        pow(sub(var("x"), int(1)), int(2)),
        pow(sub(var("y"), int(2)), int(2)),
    );
    let resp = execute(request(Command::Optimize {
        objective,
        vars: vec!["x".to_string(), "y".to_string()],
        constraints: vec![],
        sense: thales::api::command::OptSense::Minimize,
    }))
    .unwrap();

    assert!(
        !resp.results.is_empty(),
        "expected at least one result from Optimize"
    );
    let labeled_count = resp
        .results
        .iter()
        .filter(|(_, e)| matches!(&e.structured, Some(StructuredResult::Labeled { .. })))
        .count();
    assert!(
        labeled_count >= 2,
        "expected Labeled entries for x, y (and objective)"
    );

    let labels: Vec<&str> = resp
        .results
        .iter()
        .filter_map(|(_, e)| match &e.structured {
            Some(StructuredResult::Labeled { label, .. }) => Some(label.as_str()),
            _ => None,
        })
        .collect();
    assert!(labels.contains(&"x"), "expected label 'x'");
    assert!(labels.contains(&"y"), "expected label 'y'");
    assert!(labels.contains(&"objective"), "expected label 'objective'");
}

#[test]
fn structured_none_for_scalar() {
    // Simplify of a single expression should have structured = None.
    let resp = execute(request(Command::Simplify {
        expr: add(var("x"), var("x")),
        rules: SimplifyRules::all(),
        over: None,
    }))
    .unwrap();

    assert_eq!(resp.results.len(), 1, "expected single result");
    let (_, entry) = &resp.results[0];
    assert!(
        entry.structured.is_none(),
        "expected structured = None for scalar Simplify result"
    );
    assert!(
        matches!(entry.value, ResultValue::Symbolic(_)),
        "expected Symbolic value"
    );
}

#[test]
fn structured_json_roundtrip_labeled() {
    // JSON SolveSystem response should contain structured.kind = "Labeled"
    // with label and value fields.
    let payload = serde_json::json!({
        "command": {
            "type": "SolveSystem",
            "equations": [expr_json("x+y-3"), expr_json("x-y-1")],
            "vars": ["x", "y"]
        }
    });
    let resp_str =
        execute_ffi(&serde_json::to_string(&payload).unwrap()).expect("execute_ffi should succeed");
    let v: serde_json::Value = serde_json::from_str(&resp_str).unwrap();
    let results = v["results"].as_array().unwrap();

    let labeled: Vec<_> = results
        .iter()
        .filter(|r| r["structured"]["kind"] == "Labeled")
        .collect();
    assert_eq!(labeled.len(), 2, "expected two Labeled results in JSON");

    let labels: Vec<&str> = labeled
        .iter()
        .filter_map(|r| r["structured"]["label"].as_str())
        .collect();
    assert!(labels.contains(&"x"), "JSON should contain label 'x'");
    assert!(labels.contains(&"y"), "JSON should contain label 'y'");
}

#[test]
fn structured_json_roundtrip_decomposition() {
    // JSON LU response should contain structured.kind = "Decomposition"
    // with parts array containing L, U, P.
    let payload = serde_json::json!({
        "command": {
            "type": "Matrix",
            "op": "Lu",
            "operands": [{"rows": [
                [expr_json("2"), expr_json("1")],
                [expr_json("4"), expr_json("3")]
            ]}]
        }
    });
    let resp_str =
        execute_ffi(&serde_json::to_string(&payload).unwrap()).expect("execute_ffi should succeed");
    let v: serde_json::Value = serde_json::from_str(&resp_str).unwrap();
    let results = v["results"].as_array().unwrap();
    assert_eq!(results.len(), 1, "expected single LU result");

    let structured = &results[0]["structured"];
    assert_eq!(
        structured["kind"], "Decomposition",
        "expected Decomposition kind"
    );

    let parts = structured["parts"].as_array().unwrap();
    let part_names: Vec<&str> = parts.iter().filter_map(|p| p["name"].as_str()).collect();
    assert!(part_names.contains(&"L"), "JSON LU should contain L part");
    assert!(part_names.contains(&"U"), "JSON LU should contain U part");
    assert!(part_names.contains(&"P"), "JSON LU should contain P part");
}

// ── Field audit / DispatchContext tests ──────────────────────────────────────

/// A non-Symbolic mode on a command that does not honour mode (Simplify)
/// must produce a FieldIgnored diagnostic naming "mode".
#[test]
fn field_ignored_mode_numeric() {
    let resp = execute(Request {
        command: Command::Simplify {
            expr: add(var("x"), var("x")),
            rules: SimplifyRules::all(),
            over: None,
        },
        mode: SolveMode::Numeric,
        ..Default::default()
    })
    .unwrap();

    let has_field_ignored = resp.diagnostics.iter().any(|d| {
        d.code == DiagnosticCode::FieldIgnored && d.narrative.fallback_md.contains("mode")
    });
    assert!(
        has_field_ignored,
        "expected FieldIgnored diagnostic mentioning 'mode', got diagnostics: {:?}",
        resp.diagnostics
    );
}

/// A non-None budget on any command must produce a FieldIgnored diagnostic
/// naming "budget".
#[test]
fn field_ignored_budget() {
    let resp = execute(Request {
        command: Command::Simplify {
            expr: add(var("x"), var("x")),
            rules: SimplifyRules::all(),
            over: None,
        },
        budget: Some(Budget {
            max_wall_ms: Some(100),
            max_iterations: None,
        }),
        ..Default::default()
    })
    .unwrap();

    let has_field_ignored = resp.diagnostics.iter().any(|d| {
        d.code == DiagnosticCode::FieldIgnored && d.narrative.fallback_md.contains("budget")
    });
    assert!(
        has_field_ignored,
        "expected FieldIgnored diagnostic mentioning 'budget', got diagnostics: {:?}",
        resp.diagnostics
    );
}

/// DefIntegrate with mode: PreferSymbolic must NOT produce a FieldIgnored
/// for "mode" — it is the one command that honours mode.
#[test]
fn field_not_ignored_def_integrate_mode() {
    let resp = execute(Request {
        command: Command::DefIntegrate {
            expr: mul(int(2), var("x")),
            var: "x".to_string(),
            from: int(0),
            to: int(1),
        },
        mode: SolveMode::PreferSymbolic,
        ..Default::default()
    })
    .unwrap();

    let mode_ignored = resp.diagnostics.iter().any(|d| {
        d.code == DiagnosticCode::FieldIgnored && d.narrative.fallback_md.contains("mode")
    });
    assert!(
        !mode_ignored,
        "DefIntegrate should honour mode; must not emit FieldIgnored for 'mode'"
    );
}

/// Factor with over: Domain::integer() — the domain is now passed through
/// to factor_cmd rather than silently dropped.  The result is the same
/// (simplification fallback) but no FieldIgnored for "Factor.over" should
/// appear — the field was consumed, not ignored.
#[test]
fn factor_over_wired() {
    let resp = execute(request(Command::Factor {
        expr: sub(mul(var("x"), var("x")), int(9)),
        over: Domain::integer(),
        target: None,
    }))
    .unwrap();

    // Result must be Symbolic (engine ran, even if partial).
    assert_single_symbolic(&resp, EngineId::Simplify);

    // Wiring means no FieldIgnored diagnostic for Factor.over.
    let over_ignored = resp.diagnostics.iter().any(|d| {
        d.code == DiagnosticCode::FieldIgnored && d.narrative.fallback_md.contains("Factor.over")
    });
    assert!(
        !over_ignored,
        "Factor.over must not produce FieldIgnored when the domain is passed to the engine"
    );
}

/// SolveFor with over: Domain::natural() — domain is passed to solve_for_cmd
/// rather than silently dropped.  No FieldIgnored for "SolveFor.over".
#[test]
fn solve_for_over_wired() {
    let resp = execute(Request {
        command: Command::SolveFor {
            relation: sub(mul(int(2), var("x")), int(6)),
            var: "x".to_string(),
            over: Domain::natural(),
        },
        ..Default::default()
    })
    .unwrap();

    // Solver ran successfully.
    assert!(
        !resp.results.is_empty() || !resp.diagnostics.is_empty(),
        "expected a result or diagnostic from SolveFor"
    );

    // Wiring means no FieldIgnored for SolveFor.over.
    let over_ignored = resp.diagnostics.iter().any(|d| {
        d.code == DiagnosticCode::FieldIgnored && d.narrative.fallback_md.contains("SolveFor.over")
    });
    assert!(
        !over_ignored,
        "SolveFor.over must not produce FieldIgnored when the domain is passed to the engine"
    );
}
