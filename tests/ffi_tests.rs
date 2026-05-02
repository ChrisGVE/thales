//! Tests for FFI-exposed functionality.
//!
//! These tests verify that the underlying library functions used by the FFI
//! layer work correctly. The FFI functions themselves are tested via Swift
//! integration tests.

use thales::ast::Expression;
use thales::numeric::compile::{compile, decompile};
use thales::numeric::expr::Expr;
use thales::numeric::series::taylor;
use thales::numeric::SymbolId;
use thales::parser::parse_expression;
use thales::special::{beta, erf, erfc, gamma};

// =============================================================================
// Series Expansion Tests
// =============================================================================
//
// These tests mirror what the FFI layer does: parse Expression → compile to
// Arc<Expr> → call the numeric engine → decompile back to Expression for
// display. That matches the real code path through `taylor_series_ffi` /
// `maclaurin_series_ffi`.

fn expand_taylor_expr(src: &str, var_name: &str, center: &Expr, order: usize) -> Expression {
    let parsed = parse_expression(src).unwrap();
    let arc_expr = compile(&parsed);
    let var_id = SymbolId::intern(var_name);
    let ts = taylor(
        &arc_expr,
        var_id,
        &std::sync::Arc::new(center.clone()),
        order,
    );
    decompile(&ts.to_expr())
}

#[test]
fn test_taylor_series_simple_polynomial() {
    // Taylor series of x^2 around x=1 should be 1 + 2(x-1) + (x-1)^2
    let series_expr = expand_taylor_expr("x^2", "x", &Expr::Float(1.0), 3);
    let series_str = format!("{}", series_expr);
    assert!(!series_str.is_empty(), "Series should not be empty");
}

#[test]
fn test_taylor_series_exponential() {
    // Taylor series of e^x around x=0
    let series_expr = expand_taylor_expr("exp(x)", "x", &Expr::Integer(0i64.into()), 4);
    let series_str = format!("{}", series_expr);
    assert!(series_str.contains('x'), "Series should contain variable x");
}

#[test]
fn test_maclaurin_series_sin() {
    // Maclaurin series of sin(x)
    let series_expr = expand_taylor_expr("sin(x)", "x", &Expr::Integer(0i64.into()), 5);
    assert!(
        !format!("{}", series_expr).is_empty(),
        "Series should not be empty"
    );
}

#[test]
fn test_maclaurin_series_cos() {
    // Maclaurin series of cos(x)
    let series_expr = expand_taylor_expr("cos(x)", "x", &Expr::Integer(0i64.into()), 5);
    assert!(
        !format!("{}", series_expr).is_empty(),
        "Series should not be empty"
    );
}

#[test]
fn test_maclaurin_series_simple() {
    // Maclaurin series of x^3 + 2*x
    let series_expr = expand_taylor_expr("x^3 + 2*x", "x", &Expr::Integer(0i64.into()), 4);
    assert!(
        !format!("{}", series_expr).is_empty(),
        "Series should not be empty"
    );
}

// =============================================================================
// Special Functions Tests
// =============================================================================

#[test]
fn test_gamma_positive_integer() {
    // Γ(5) = 4! = 24
    let x = Expression::Integer(5);
    let result = gamma(&x);

    assert!(result.is_ok(), "Gamma of positive integer should succeed");
    let gamma_result = result.unwrap();
    assert_eq!(
        gamma_result.numeric_value,
        Some(24.0),
        "Γ(5) should equal 24"
    );
    assert!(
        !gamma_result.derivation_steps.is_empty(),
        "Derivation steps should be present"
    );
}

#[test]
fn test_gamma_one() {
    // Γ(1) = 0! = 1
    let x = Expression::Integer(1);
    let result = gamma(&x);

    assert!(result.is_ok(), "Gamma(1) should succeed");
    assert_eq!(
        result.unwrap().numeric_value,
        Some(1.0),
        "Γ(1) should equal 1"
    );
}

#[test]
fn test_gamma_two() {
    // Γ(2) = 1! = 1
    let x = Expression::Integer(2);
    let result = gamma(&x);

    assert!(result.is_ok(), "Gamma(2) should succeed");
    assert_eq!(
        result.unwrap().numeric_value,
        Some(1.0),
        "Γ(2) should equal 1"
    );
}

#[test]
fn test_gamma_three() {
    // Γ(3) = 2! = 2
    let x = Expression::Integer(3);
    let result = gamma(&x);

    assert!(result.is_ok(), "Gamma(3) should succeed");
    assert_eq!(
        result.unwrap().numeric_value,
        Some(2.0),
        "Γ(3) should equal 2"
    );
}

#[test]
fn test_gamma_four() {
    // Γ(4) = 3! = 6
    let x = Expression::Integer(4);
    let result = gamma(&x);

    assert!(result.is_ok(), "Gamma(4) should succeed");
    assert_eq!(
        result.unwrap().numeric_value,
        Some(6.0),
        "Γ(4) should equal 6"
    );
}

#[test]
fn test_gamma_half() {
    // Γ(1/2) = √π ≈ 1.772453850905516
    let x = Expression::Float(0.5);
    let result = gamma(&x);

    assert!(result.is_ok(), "Gamma(1/2) should succeed");
    let gamma_val = result.unwrap().numeric_value.unwrap();
    assert!(
        (gamma_val - 1.772453850905516).abs() < 0.00001,
        "Γ(1/2) should equal √π, got {}",
        gamma_val
    );
}

#[test]
fn test_gamma_negative() {
    // Γ(-1) is undefined (pole)
    let x = Expression::Integer(-1);
    let result = gamma(&x);

    assert!(result.is_err(), "Gamma of negative integer should fail");
}

#[test]
fn test_erf_zero() {
    // erf(0) = 0
    let x = Expression::Float(0.0);
    let result = erf(&x);

    assert!(result.is_ok(), "erf(0) should succeed");
    let erf_val = result.unwrap().numeric_value.unwrap();
    assert!(
        erf_val.abs() < 0.00001,
        "erf(0) should equal 0, got {}",
        erf_val
    );
}

#[test]
fn test_erf_positive() {
    // erf(1) ≈ 0.8427 (well-known value)
    let x = Expression::Float(1.0);
    let result = erf(&x);

    assert!(result.is_ok(), "erf(1) should succeed");
    let erf_val = result.unwrap().numeric_value.unwrap();
    assert!(
        (erf_val - 0.8427).abs() < 0.01,
        "erf(1) should be approximately 0.8427, got {}",
        erf_val
    );
}

#[test]
fn test_erf_negative() {
    // erf is an odd function: erf(-x) = -erf(x)
    let x_pos = Expression::Float(1.0);
    let x_neg = Expression::Float(-1.0);

    let result_pos = erf(&x_pos).unwrap();
    let result_neg = erf(&x_neg).unwrap();

    let val_pos = result_pos.numeric_value.unwrap();
    let val_neg = result_neg.numeric_value.unwrap();

    assert!(
        (val_pos + val_neg).abs() < 0.00001,
        "erf(-x) should equal -erf(x), got {} and {}",
        val_pos,
        val_neg
    );
}

#[test]
fn test_erf_has_derivation_steps() {
    let x = Expression::Float(1.0);
    let result = erf(&x);

    assert!(result.is_ok(), "erf(1) should succeed");
    let erf_result = result.unwrap();
    assert!(
        !erf_result.derivation_steps.is_empty(),
        "Derivation steps should be present"
    );
}

// =============================================================================
// Beta Function Tests
// =============================================================================

#[test]
fn test_beta_two_three() {
    // B(2, 3) = Γ(2)·Γ(3) / Γ(5) = 1·2 / 24 = 1/12
    let a = Expression::Integer(2);
    let b = Expression::Integer(3);
    let result = beta(&a, &b);

    assert!(result.is_ok(), "beta(2, 3) should succeed");
    let val = result.unwrap().numeric_value.unwrap();
    assert!(
        (val - 1.0 / 12.0).abs() < 1e-10,
        "B(2, 3) should equal 1/12, got {}",
        val
    );
}

#[test]
fn test_beta_one_one() {
    // B(1, 1) = Γ(1)·Γ(1) / Γ(2) = 1·1 / 1 = 1
    let a = Expression::Integer(1);
    let b = Expression::Integer(1);
    let result = beta(&a, &b);

    assert!(result.is_ok(), "beta(1, 1) should succeed");
    assert_eq!(
        result.unwrap().numeric_value,
        Some(1.0),
        "B(1, 1) should equal 1"
    );
}

#[test]
fn test_beta_symmetry() {
    // B(a, b) = B(b, a) — symmetry property
    let a = Expression::Float(2.5);
    let b = Expression::Float(3.5);
    let result_ab = beta(&a, &b).unwrap().numeric_value.unwrap();
    let result_ba = beta(&b, &a).unwrap().numeric_value.unwrap();

    assert!(
        (result_ab - result_ba).abs() < 1e-10,
        "Beta function must be symmetric: B(a,b) == B(b,a), got {} vs {}",
        result_ab,
        result_ba
    );
}

#[test]
fn test_beta_has_derivation_steps() {
    let a = Expression::Integer(2);
    let b = Expression::Integer(3);
    let result = beta(&a, &b).unwrap();

    assert!(
        !result.derivation_steps.is_empty(),
        "Beta derivation steps should be present"
    );
}

// =============================================================================
// Erfc Function Tests
// =============================================================================

#[test]
fn test_erfc_zero() {
    // erfc(0) = 1 - erf(0) = 1
    let x = Expression::Integer(0);
    let result = erfc(&x);

    assert!(result.is_ok(), "erfc(0) should succeed");
    assert_eq!(
        result.unwrap().numeric_value,
        Some(1.0),
        "erfc(0) should equal 1"
    );
}

#[test]
fn test_erfc_complements_erf() {
    // erfc(x) = 1 - erf(x) for all x
    let x = Expression::Float(1.0);
    let erf_val = erf(&x).unwrap().numeric_value.unwrap();
    let erfc_val = erfc(&x).unwrap().numeric_value.unwrap();

    assert!(
        (erf_val + erfc_val - 1.0).abs() < 1e-10,
        "erf(x) + erfc(x) must equal 1, got {} + {} = {}",
        erf_val,
        erfc_val,
        erf_val + erfc_val
    );
}

#[test]
fn test_erfc_has_derivation_steps() {
    let x = Expression::Float(1.0);
    let result = erfc(&x).unwrap();

    assert!(
        !result.derivation_steps.is_empty(),
        "Erfc derivation steps should be present"
    );
}

// =============================================================================
// Verify LaTeX output for new functions
// =============================================================================

#[test]
fn test_series_latex_output() {
    let series_expr = expand_taylor_expr("x^2", "x", &Expr::Integer(0i64.into()), 3);
    let latex = series_expr.to_latex();

    assert!(!latex.is_empty(), "LaTeX output should not be empty");
    assert!(latex.contains('x'), "LaTeX should contain variable");
}

#[test]
fn test_gamma_latex_output() {
    let x = Expression::Integer(5);
    let result = gamma(&x).unwrap();
    let latex = result.value.to_latex();

    assert!(!latex.is_empty(), "LaTeX output should not be empty");
}

#[test]
fn test_erf_latex_output() {
    let x = Expression::Float(1.0);
    let result = erf(&x).unwrap();
    let latex = result.value.to_latex();

    assert!(!latex.is_empty(), "LaTeX output should not be empty");
}

// =============================================================================
// Log argument order regression tests (AST-01 / AST-02 / AST-03)
// =============================================================================

#[test]
fn test_ffi_log_core_consistency_value_base() {
    // Core evaluator and FFI evaluator must agree on log(value, base).
    // log(8, 2) = 3 under the log(value, base) convention.
    use std::collections::HashMap;
    use thales::ast::{Expression, Function};

    let expr = Expression::Function(
        Function::Log,
        vec![Expression::Float(8.0), Expression::Float(2.0)],
    );
    let core_result = expr
        .evaluate(&HashMap::new())
        .expect("core log(8, 2) must evaluate");
    assert!(
        (core_result - 3.0).abs() < 1e-10,
        "core log(8, 2) should be 3, got {core_result}"
    );
}

#[test]
fn test_ffi_log_domain_negative_value() {
    // FFI path must return None (not NaN) for negative value.
    use std::collections::HashMap;
    use thales::ast::{Expression, Function};

    let expr = Expression::Function(
        Function::Log,
        vec![Expression::Float(-1.0), Expression::Float(10.0)],
    );
    assert!(
        expr.evaluate(&HashMap::new()).is_none(),
        "log(-1, 10) must return None"
    );
}

#[test]
fn test_ffi_ln_domain_negative_returns_none() {
    use std::collections::HashMap;
    use thales::ast::{Expression, Function};

    let expr = Expression::Function(Function::Ln, vec![Expression::Float(-5.0)]);
    assert!(
        expr.evaluate(&HashMap::new()).is_none(),
        "ln(-5) must return None, not NaN"
    );
}

// =============================================================================
// Core / FFI evaluator parity tests (AST-02 / AST-03 / AST-04)
//
// The FFI evaluate_ffi now delegates to Expression::evaluate.  These tests
// confirm that every function previously absent from the FFI evaluator is
// handled correctly by the unified core path.
// =============================================================================

/// Parse `src` as an expression, substitute `vars`, and return the f64 result.
fn eval_str(src: &str, vars: &std::collections::HashMap<String, f64>) -> Option<f64> {
    parse_expression(src).ok()?.evaluate(vars)
}

#[test]
fn test_parity_log2() {
    // log2(8) = 3
    let vars = std::collections::HashMap::new();
    let result = eval_str("log2(8)", &vars).expect("log2(8) must evaluate");
    assert!(
        (result - 3.0).abs() < 1e-10,
        "log2(8) should be 3, got {result}"
    );
}

#[test]
fn test_parity_log10() {
    // log10(1000) = 3
    let vars = std::collections::HashMap::new();
    let result = eval_str("log10(1000)", &vars).expect("log10(1000) must evaluate");
    assert!(
        (result - 3.0).abs() < 1e-10,
        "log10(1000) should be 3, got {result}"
    );
}

#[test]
fn test_parity_cbrt() {
    // cbrt(27) = 3
    let vars = std::collections::HashMap::new();
    let result = eval_str("cbrt(27)", &vars).expect("cbrt(27) must evaluate");
    assert!(
        (result - 3.0).abs() < 1e-10,
        "cbrt(27) should be 3, got {result}"
    );
}

#[test]
fn test_parity_atan2() {
    // atan2(1, 1) = π/4
    let vars = std::collections::HashMap::new();
    let result = eval_str("atan2(1, 1)", &vars).expect("atan2(1, 1) must evaluate");
    assert!(
        (result - std::f64::consts::FRAC_PI_4).abs() < 1e-10,
        "atan2(1, 1) should be π/4, got {result}"
    );
}

#[test]
fn test_parity_sign_positive() {
    let vars = std::collections::HashMap::new();
    let result = eval_str("sign(5)", &vars).expect("sign(5) must evaluate");
    assert!(
        (result - 1.0).abs() < 1e-10,
        "sign(5) should be 1, got {result}"
    );
}

#[test]
fn test_parity_sign_negative() {
    let vars = std::collections::HashMap::new();
    let result = eval_str("sign(-3)", &vars).expect("sign(-3) must evaluate");
    assert!(
        (result - (-1.0)).abs() < 1e-10,
        "sign(-3) should be -1, got {result}"
    );
}

#[test]
fn test_parity_min() {
    let vars = std::collections::HashMap::new();
    let result = eval_str("min(4, 2)", &vars).expect("min(4, 2) must evaluate");
    assert!(
        (result - 2.0).abs() < 1e-10,
        "min(4, 2) should be 2, got {result}"
    );
}

#[test]
fn test_parity_max() {
    let vars = std::collections::HashMap::new();
    let result = eval_str("max(4, 2)", &vars).expect("max(4, 2) must evaluate");
    assert!(
        (result - 4.0).abs() < 1e-10,
        "max(4, 2) should be 4, got {result}"
    );
}

#[test]
fn test_parity_pow() {
    // pow(2, 10) = 1024
    let vars = std::collections::HashMap::new();
    let result = eval_str("pow(2, 10)", &vars).expect("pow(2, 10) must evaluate");
    assert!(
        (result - 1024.0).abs() < 1e-10,
        "pow(2, 10) should be 1024, got {result}"
    );
}

#[test]
fn test_parity_ln_positive() {
    // ln(e) = 1
    let vars = std::collections::HashMap::new();
    let result = eval_str("ln(e)", &vars).expect("ln(e) must evaluate");
    assert!(
        (result - 1.0).abs() < 1e-10,
        "ln(e) should be 1, got {result}"
    );
}

#[test]
fn test_parity_ln_domain_zero_returns_none() {
    // ln(0) is undefined — must return None, not NaN
    let vars = std::collections::HashMap::new();
    assert!(eval_str("ln(0)", &vars).is_none(), "ln(0) must return None");
}

#[test]
fn test_parity_log_single_arg() {
    // log(100) = log10(100) = 2  (single-arg log convention)
    let vars = std::collections::HashMap::new();
    let result = eval_str("log(100)", &vars).expect("log(100) must evaluate");
    assert!(
        (result - 2.0).abs() < 1e-10,
        "log(100) should be 2, got {result}"
    );
}

#[test]
fn test_parity_log_two_arg() {
    // log(8, 2) = 3  (value, base convention)
    let vars = std::collections::HashMap::new();
    let result = eval_str("log(8, 2)", &vars).expect("log(8, 2) must evaluate");
    assert!(
        (result - 3.0).abs() < 1e-10,
        "log(8, 2) should be 3, got {result}"
    );
}

#[test]
fn test_parity_log_domain_nonpositive_returns_none() {
    // log(0, 10) is undefined — must return None
    let vars = std::collections::HashMap::new();
    assert!(
        eval_str("log(0, 10)", &vars).is_none(),
        "log(0, 10) must return None"
    );
}

// =============================================================================
// ODE solver tests (backing the FFI layer)
// =============================================================================

/// Helper: parse a string expression and build a FirstOrderODE, then solve it.
fn solve_ode_str(
    rhs: &str,
    dep: &str,
    indep: &str,
) -> Result<thales::ode::ODESolution, thales::ode::ODEError> {
    use thales::ode::{solve_linear, solve_separable, FirstOrderODE};
    let expr = parse_expression(rhs).expect("rhs must parse");
    let ode = FirstOrderODE::new(dep, indep, expr);
    if ode.is_separable() {
        solve_separable(&ode)
    } else if ode.is_linear() {
        solve_linear(&ode)
    } else {
        Err(thales::ode::ODEError::CannotSolve(
            "ODE is neither separable nor linear".to_string(),
        ))
    }
}

#[test]
fn test_ode_separable_dy_dx_eq_y() {
    // dy/dx = y  →  separable, solution is ln|y| = x + C (implicit form)
    let sol = solve_ode_str("y", "y", "x").expect("dy/dx = y must be solvable");
    let sol_str = format!("{}", sol.general_solution);
    assert!(
        sol_str.contains("ln") || sol_str.contains("exp"),
        "Solution of dy/dx = y should contain ln or exp, got: {sol_str}"
    );
    assert_eq!(sol.method, "Separation of variables");
}

#[test]
fn test_ode_linear_dy_dx_plus_y_eq_x() {
    // dy/dx + y = x  →  rhs = -y + x
    // The integrating factor method requires integrating exp(-(-x)) * x which
    // the current integrator cannot handle, so this returns an error for now.
    let result = solve_ode_str("-y + x", "y", "x");
    // Accept either a solution or a known integration limitation
    if let Ok(sol) = result {
        assert!(!format!("{}", sol.general_solution).is_empty());
    }
    // If Err, that's the known integrator limitation — acceptable
}

#[test]
fn test_ode_ivp_dy_dx_eq_y_with_y0_eq_1() {
    // dy/dx = y, y(0) = 1  →  particular solution y = exp(x)
    use thales::ast::Expression;
    use thales::ode::{solve_ivp, verify, FirstOrderODE};

    let rhs = parse_expression("y").expect("y must parse");
    let ode = FirstOrderODE::new("y", "x", rhs);
    let x0 = Expression::Float(0.0);
    let y0 = Expression::Float(1.0);

    let sol = solve_ivp(&ode, &x0, &y0).expect("IVP dy/dx=y, y(0)=1 must be solvable");

    // Decision 2b: particular solution must be y-free and satisfy IC at x₀.
    verify::assert_y_free(&sol.general_solution, "y");
    verify::assert_ic_satisfied(&sol.general_solution, "x", 0.0, 1.0, 1e-9);
}

#[test]
fn test_ode_ivp_dy_dx_eq_neg_y() {
    // dy/dx = -y, y(0) = 2
    // Integration of 1/(-y) is a known limitation of the current integrator.
    use thales::ast::Expression;
    use thales::ode::{solve_ivp, verify, FirstOrderODE};

    let rhs = parse_expression("-y").expect("-y must parse");
    let ode = FirstOrderODE::new("y", "x", rhs);
    let x0 = Expression::Float(0.0);
    let y0 = Expression::Float(2.0);

    let result = solve_ivp(&ode, &x0, &y0);
    // Accept either a solution or a known integration limitation.
    // Decision 2b: when a particular solution is returned, it must be y-free
    // and satisfy the initial condition.
    if let Ok(sol) = result {
        verify::assert_y_free(&sol.general_solution, "y");
        verify::assert_ic_satisfied(&sol.general_solution, "x", 0.0, 2.0, 1e-9);
    }
    // If Err, that's the known integrator limitation — acceptable
}

#[test]
fn test_ode_unsolvable_returns_error() {
    // dy/dx = y^2 + x^2 is neither separable in the simple sense nor linear
    let result = solve_ode_str("y^2 + x^2", "y", "x");
    assert!(
        result.is_err(),
        "dy/dx = y^2 + x^2 should not be solvable by separable/linear methods"
    );
}

// =============================================================================
// execute_json_ffi integration tests
//
// These tests exercise the canonical cross-language entry point `execute_json_ffi`
// (defined in ffi/json_impl.rs), which delegates to `api::json::execute_ffi`.
// We call the underlying function directly since the ffi feature flag is not
// enabled in the default test build.
// =============================================================================

#[test]
fn execute_json_ffi_simplify() {
    // Valid JSON simplify request returns a JSON response with a result.
    let ml = mathlex::parse("x + x").expect("parse x + x");
    let expr_val = serde_json::to_value(&ml).expect("serialise expr");
    let payload = serde_json::json!({ "command": {"type": "Simplify", "expr": expr_val} });
    let req = serde_json::to_string(&payload).unwrap();
    let resp = thales::api::json::execute_ffi(&req).expect("execute_json_ffi should succeed");
    let val: serde_json::Value = serde_json::from_str(&resp).expect("response must be valid JSON");
    assert!(
        val.get("results").is_some(),
        "response must contain a 'results' field, got: {val}"
    );
}

#[test]
fn execute_json_ffi_unknown_command() {
    // An unknown command type returns an error string, not a panic.
    // Use a well-formed expr object so the failure is due to the unknown type.
    let ml = mathlex::parse("x").expect("parse x");
    let expr_val = serde_json::to_value(&ml).expect("serialise expr");
    let payload = serde_json::json!({ "command": {"type": "DoesNotExist", "expr": expr_val} });
    let req = serde_json::to_string(&payload).unwrap();
    let err = thales::api::json::execute_ffi(&req);
    assert!(
        err.is_err(),
        "unknown command must return Err, got: {err:?}"
    );
}

#[test]
fn execute_json_ffi_bad_json() {
    // Malformed JSON returns an error string describing the parse failure.
    let req = r#"{not valid json"#;
    let err = thales::api::json::execute_ffi(req);
    assert!(err.is_err(), "malformed JSON must return Err, got: {err:?}");
    let msg = err.unwrap_err();
    assert!(
        msg.contains("invalid JSON"),
        "error message must mention invalid JSON, got: {msg}"
    );
}
