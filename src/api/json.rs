//! JSON wire protocol for [`super::execute`].
//!
//! v0.8.1 ships a pragmatic schema: each JSON request carries a
//! `command.type` tag plus string-encoded `Expression` fields. The
//! dispatcher parses strings via [`crate::parser::parse_expression`]
//! before handing to the symbolic engines. Responses serialise
//! `Expression` values back to canonical display strings.
//!
//! A full serde-derived round-trip (required for typed Swift wrappers)
//! lands alongside T6 Phase B when the AST grows serde derives.

use serde_json::{json, Value};

use crate::ast::{Expression, Variable};
use crate::parser::parse_equation;
use crate::ThalesError;

use super::command::{Command, IvpData, LimitPoint, MatrixOp, SimplifyRules, SpecialKind};
use super::domain::Domain;
use super::request::{Request, SolveMode};
use super::response::{NarratedStep, Response, ResultEntry, ResultKey, ResultValue};

/// FFI-shaped entry point: JSON request → JSON response, with errors
/// stringified for cross-language transport.
pub fn execute_ffi(request_json: &str) -> Result<String, String> {
    let request_val: Value =
        serde_json::from_str(request_json).map_err(|e| format!("invalid JSON request: {}", e))?;
    let request = request_from_json(&request_val)?;
    let response = super::dispatch::execute(request)
        .map_err(|e: ThalesError| format!("dispatch error: {}", e))?;
    let response_val = response_to_json(&response);
    serde_json::to_string(&response_val).map_err(|e| format!("failed to serialise response: {}", e))
}

// ── Request parsing ──────────────────────────────────────────────────────────

fn request_from_json(val: &Value) -> Result<Request, String> {
    let command = val
        .get("command")
        .ok_or_else(|| "missing `command` field".to_string())?;
    let command = command_from_json(command)?;

    let narrate = val.get("narrate").and_then(|v| v.as_bool()).unwrap_or(true);

    let mode = val
        .get("mode")
        .and_then(|v| v.as_str())
        .map(parse_solve_mode)
        .transpose()?
        .unwrap_or_default();

    Ok(Request {
        command,
        narrate,
        mode,
        precision: None,
        output_units: None,
        ambient_domain: None,
        budget: None,
        seed: None,
    })
}

fn command_from_json(val: &Value) -> Result<Command, String> {
    let ty = val
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "command: missing `type`".to_string())?;

    match ty {
        "Noop" => Ok(Command::Noop),

        "Simplify" => Ok(Command::Simplify {
            expr: get_expr(val, "expr")?,
            rules: SimplifyRules::all(),
            over: None,
        }),
        "Expand" => Ok(Command::Expand {
            expr: get_expr(val, "expr")?,
            target: None,
        }),
        "Factor" => Ok(Command::Factor {
            expr: get_expr(val, "expr")?,
            over: Domain::real(),
            target: None,
        }),
        "Substitute" => {
            let expr = get_expr(val, "expr")?;
            let bindings = get_bindings(val)?;
            Ok(Command::Substitute {
                expr,
                bindings,
                target: None,
            })
        }
        "CombineLikeTerms" => Ok(Command::CombineLikeTerms {
            expr: get_expr(val, "expr")?,
            target: None,
        }),
        "CommonDenominator" => Ok(Command::CommonDenominator {
            expr: get_expr(val, "expr")?,
            target: None,
        }),
        "PartialFractions" => Ok(Command::PartialFractions {
            expr: get_expr(val, "expr")?,
            var: get_string(val, "var")?,
        }),
        "Rationalize" => Ok(Command::Rationalize {
            expr: get_expr(val, "expr")?,
            target: None,
        }),
        "Conjugate" => Ok(Command::Conjugate {
            expr: get_expr(val, "expr")?,
            target: None,
        }),
        "InverseFn" => Ok(Command::InverseFn {
            expr: get_expr(val, "expr")?,
            var: get_string(val, "var")?,
        }),
        "Rearrange" => Ok(Command::Rearrange {
            equation: get_expr(val, "equation")?,
            solve_for: get_string(val, "solve_for")?,
        }),

        "SolveFor" => Ok(Command::SolveFor {
            relation: get_expr(val, "relation")?,
            var: get_string(val, "var")?,
            over: Domain::real(),
        }),
        "SolveSystem" => Ok(Command::SolveSystem {
            equations: get_expr_list(val, "equations")?,
            vars: get_string_list(val, "vars")?,
            over: Domain::real(),
        }),

        "Diff" => Ok(Command::Diff {
            expr: get_expr(val, "expr")?,
            var: get_string(val, "var")?,
            order: val.get("order").and_then(|v| v.as_u64()).unwrap_or(1) as u32,
        }),
        "PartialDiff" => Ok(Command::PartialDiff {
            expr: get_expr(val, "expr")?,
            vars: get_partial_diff_vars(val)?,
        }),
        "Gradient" => Ok(Command::Gradient {
            expr: get_expr(val, "expr")?,
            vars: get_string_list(val, "vars")?,
        }),

        "Integrate" => Ok(Command::Integrate {
            expr: get_expr(val, "expr")?,
            var: get_string(val, "var")?,
        }),
        "DefIntegrate" => Ok(Command::DefIntegrate {
            expr: get_expr(val, "expr")?,
            var: get_string(val, "var")?,
            from: get_expr(val, "from")?,
            to: get_expr(val, "to")?,
        }),

        "Limit" => {
            let point_val = val
                .get("point")
                .ok_or_else(|| "Limit: missing `point`".to_string())?;
            let point = match point_val.as_str() {
                Some("+inf") => LimitPoint::PosInf,
                Some("-inf") => LimitPoint::NegInf,
                Some(s) => LimitPoint::Finite(parse_expr_str(s)?),
                None => return Err("Limit: `point` must be a string".into()),
            };
            Ok(Command::Limit {
                expr: get_expr(val, "expr")?,
                var: get_string(val, "var")?,
                point,
                side: None,
            })
        }

        "FourierSeries" => Ok(Command::FourierSeries {
            expr: get_expr(val, "expr")?,
            var: get_string(val, "var")?,
            period: get_expr(val, "period")?,
            terms: val.get("terms").and_then(|v| v.as_u64()).unwrap_or(3) as u32,
        }),

        "Ode" => {
            let ic = val.get("ic").and_then(|v| ivp_from_json(v).ok());
            Ok(Command::Ode {
                equation: get_expr(val, "equation")?,
                fn_name: get_string(val, "fn_name")?,
                var: get_string(val, "var")?,
                ic,
            })
        }

        "SpecialFn" => Ok(Command::SpecialFn {
            kind: parse_special_kind(
                val.get("kind")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "SpecialFn: missing `kind`".to_string())?,
            )?,
            args: get_expr_list(val, "args")?,
        }),

        "Matrix" => Ok(Command::Matrix {
            op: parse_matrix_op(
                val.get("op")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "Matrix: missing `op`".to_string())?,
            )?,
            operands: Vec::new(),
        }),

        other => Err(format!(
            "unsupported command type `{}` in v0.8.1 JSON transport",
            other
        )),
    }
}

// ── Response serialisation ───────────────────────────────────────────────────

fn response_to_json(response: &Response) -> Value {
    json!({
        "results": response
            .results
            .iter()
            .map(|(k, e)| result_entry_to_json(k, e))
            .collect::<Vec<_>>(),
        "diagnostics": response
            .diagnostics
            .iter()
            .map(|d| json!({
                "severity": format!("{:?}", d.severity),
                "code": format!("{:?}", d.code),
                "template_id": d.narrative.template_id,
                "fallback_md": d.narrative.fallback_md,
            }))
            .collect::<Vec<_>>(),
        "assumptions": response
            .assumptions
            .iter()
            .map(|a| json!({
                "template_id": a.narrative.template_id,
                "fallback_md": a.narrative.fallback_md,
            }))
            .collect::<Vec<_>>(),
        "meta": json!({
            "elapsed_ms": response.meta.elapsed_ms,
            "iterations": response.meta.iterations,
            "engine_trace": response
                .meta
                .engine_trace
                .iter()
                .map(|e| format!("{:?}", e))
                .collect::<Vec<_>>(),
        }),
    })
}

fn result_entry_to_json(key: &ResultKey, entry: &ResultEntry) -> Value {
    let key_label = match key {
        ResultKey::Single => "Single".to_string(),
        ResultKey::Branch(_) => "Branch".to_string(),
    };
    json!({
        "key": key_label,
        "value": result_value_to_json(&entry.value),
        "shape": format!("{:?}", entry.shape),
        "engine": format!("{:?}", entry.engine),
        "steps": entry
            .steps
            .iter()
            .map(step_to_json)
            .collect::<Vec<_>>(),
        "alternatives": entry
            .alternatives
            .iter()
            .map(|e| format!("{}", e))
            .collect::<Vec<_>>(),
    })
}

fn result_value_to_json(value: &ResultValue) -> Value {
    match value {
        ResultValue::Symbolic(e) => json!({
            "kind": "Symbolic",
            "expr": format!("{}", e),
        }),
        ResultValue::Numeric { value, method, .. } => json!({
            "kind": "Numeric",
            "expr": format!("{}", value),
            "method": format!("{:?}", method),
        }),
        ResultValue::Hybrid {
            last_symbolic,
            numeric,
            method,
            ..
        } => json!({
            "kind": "Hybrid",
            "last_symbolic": format!("{}", last_symbolic),
            "numeric": format!("{}", numeric),
            "method": format!("{:?}", method),
        }),
        ResultValue::Unsolved { reason } => json!({
            "kind": "Unsolved",
            "reason_md": reason.fallback_md,
            "template_id": reason.template_id,
        }),
        ResultValue::NoSolution { reason, .. } => json!({
            "kind": "NoSolution",
            "reason_md": reason.fallback_md,
            "template_id": reason.template_id,
        }),
    }
}

fn step_to_json(step: &NarratedStep) -> Value {
    json!({
        "tag": format!("{:?}", step.tag),
        "difficulty": format!("{:?}", step.difficulty),
        "detail_md": step.narrative.fallback_md,
        "template_id": step.narrative.template_id,
        "input": step.input.as_ref().map(|e| format!("{}", e)),
        "output": step.output.as_ref().map(|e| format!("{}", e)),
    })
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn get_string(val: &Value, key: &str) -> Result<String, String> {
    val.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| format!("missing string field `{}`", key))
}

fn get_expr(val: &Value, key: &str) -> Result<Expression, String> {
    let s = val
        .get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("missing expression field `{}`", key))?;
    parse_expr_str(s)
}

fn parse_expr_str(s: &str) -> Result<Expression, String> {
    // Use parse_equation so callers can pass `a = b` for relations; if the
    // input parses as an equation, collapse into `lhs - rhs` so the engines
    // that expect a bare expression handle it uniformly. Fall back to
    // parse_expression for plain inputs.
    if s.contains('=') {
        let eq = parse_equation(s).map_err(|e| format!("failed to parse `{}`: {:?}", s, e))?;
        Ok(Expression::Binary(
            crate::ast::BinaryOp::Sub,
            Box::new(eq.left),
            Box::new(eq.right),
        ))
    } else {
        crate::parser::parse_expression(s).map_err(|e| format!("failed to parse `{}`: {:?}", s, e))
    }
}

fn get_expr_list(val: &Value, key: &str) -> Result<Vec<Expression>, String> {
    let arr = val
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("missing array field `{}`", key))?;
    arr.iter()
        .map(|v| {
            v.as_str()
                .ok_or_else(|| format!("`{}` entries must be strings", key))
                .and_then(parse_expr_str)
        })
        .collect()
}

fn get_string_list(val: &Value, key: &str) -> Result<Vec<String>, String> {
    let arr = val
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("missing array field `{}`", key))?;
    arr.iter()
        .map(|v| {
            v.as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| format!("`{}` entries must be strings", key))
        })
        .collect()
}

fn get_bindings(val: &Value) -> Result<Vec<(Expression, Expression)>, String> {
    let arr = val
        .get("bindings")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "Substitute: missing `bindings` array".to_string())?;
    arr.iter()
        .map(|entry| {
            let obj = entry
                .as_object()
                .ok_or_else(|| "binding entries must be objects".to_string())?;
            let old = obj
                .get("old")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "binding: missing `old`".to_string())?;
            let new = obj
                .get("new")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "binding: missing `new`".to_string())?;
            Ok((parse_expr_str(old)?, parse_expr_str(new)?))
        })
        .collect()
}

fn get_partial_diff_vars(val: &Value) -> Result<Vec<(String, u32)>, String> {
    let arr = val
        .get("vars")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "PartialDiff: missing `vars` array".to_string())?;
    arr.iter()
        .map(|entry| {
            let obj = entry
                .as_object()
                .ok_or_else(|| "PartialDiff entry must be an object".to_string())?;
            let var = obj
                .get("var")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "PartialDiff entry: missing `var`".to_string())?
                .to_string();
            let order = obj.get("order").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
            Ok((var, order))
        })
        .collect()
}

fn ivp_from_json(val: &Value) -> Result<IvpData, String> {
    let var_at = val
        .get("var_at")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "ic: missing `var_at`".to_string())
        .and_then(parse_expr_str)?;
    let fn_at = val
        .get("fn_at")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "ic: missing `fn_at`".to_string())
        .and_then(parse_expr_str)?;
    let derivatives_at = val
        .get("derivatives_at")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .filter_map(|s| parse_expr_str(s).ok())
                .collect()
        })
        .unwrap_or_default();
    let _ = Variable::new("placeholder_for_compiler_hint");
    Ok(IvpData {
        var_at,
        fn_at,
        derivatives_at,
    })
}

fn parse_solve_mode(s: &str) -> Result<SolveMode, String> {
    match s {
        "Symbolic" => Ok(SolveMode::Symbolic),
        "Numeric" => Ok(SolveMode::Numeric),
        "PreferSymbolic" => Ok(SolveMode::PreferSymbolic),
        other => Err(format!("unknown SolveMode `{}`", other)),
    }
}

fn parse_special_kind(s: &str) -> Result<SpecialKind, String> {
    match s {
        "Gamma" => Ok(SpecialKind::Gamma),
        "Beta" => Ok(SpecialKind::Beta),
        "Erf" => Ok(SpecialKind::Erf),
        "Erfc" => Ok(SpecialKind::Erfc),
        other => Err(format!("unknown SpecialKind `{}`", other)),
    }
}

fn parse_matrix_op(s: &str) -> Result<MatrixOp, String> {
    match s {
        "Add" => Ok(MatrixOp::Add),
        "Subtract" => Ok(MatrixOp::Subtract),
        "Multiply" => Ok(MatrixOp::Multiply),
        "ScalarMultiply" => Ok(MatrixOp::ScalarMultiply),
        "Transpose" => Ok(MatrixOp::Transpose),
        "Determinant" => Ok(MatrixOp::Determinant),
        "Inverse" => Ok(MatrixOp::Inverse),
        "Trace" => Ok(MatrixOp::Trace),
        "Rank" => Ok(MatrixOp::Rank),
        "NullSpace" => Ok(MatrixOp::NullSpace),
        "Eigenvalues" => Ok(MatrixOp::Eigenvalues),
        "Eigenvectors" => Ok(MatrixOp::Eigenvectors),
        "Lu" => Ok(MatrixOp::Lu),
        "Qr" => Ok(MatrixOp::Qr),
        "SolveLinear" => Ok(MatrixOp::SolveLinear),
        other => Err(format!("unknown MatrixOp `{}`", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplify_round_trip() {
        let req = r#"{"command":{"type":"Simplify","expr":"x + x"}}"#;
        let resp = execute_ffi(req).expect("execute_ffi should succeed");
        let v: Value = serde_json::from_str(&resp).unwrap();
        let results = v.get("results").and_then(|r| r.as_array()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["value"]["kind"], "Symbolic");
    }

    #[test]
    fn diff_round_trip() {
        let req = r#"{"command":{"type":"Diff","expr":"x^2","var":"x","order":1}}"#;
        let resp = execute_ffi(req).unwrap();
        let v: Value = serde_json::from_str(&resp).unwrap();
        let engine = v["results"][0]["engine"].as_str().unwrap();
        assert_eq!(engine, "Differentiation");
    }

    #[test]
    fn integrate_round_trip() {
        let req = r#"{"command":{"type":"Integrate","expr":"2*x","var":"x"}}"#;
        let resp = execute_ffi(req).unwrap();
        let v: Value = serde_json::from_str(&resp).unwrap();
        let engine = v["results"][0]["engine"].as_str().unwrap();
        assert_eq!(engine, "PatternIntegration");
    }

    #[test]
    fn solve_for_round_trip() {
        let req = r#"{"command":{"type":"SolveFor","relation":"2*x + 3 = 7","var":"x"}}"#;
        let resp = execute_ffi(req).unwrap();
        let v: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(v["results"][0]["engine"], "EquationSolver");
    }

    #[test]
    fn unknown_command_errors() {
        let req = r#"{"command":{"type":"DoesNotExist","expr":"x"}}"#;
        let err = execute_ffi(req).unwrap_err();
        assert!(err.contains("unsupported command"));
    }

    #[test]
    fn bad_json_errors() {
        let err = execute_ffi("{not json").unwrap_err();
        assert!(err.contains("invalid JSON"));
    }

    #[test]
    fn noop_reports_diagnostic() {
        let req = r#"{"command":{"type":"Noop"}}"#;
        let resp = execute_ffi(req).unwrap();
        let v: Value = serde_json::from_str(&resp).unwrap();
        let diags = v["diagnostics"].as_array().unwrap();
        assert!(!diags.is_empty());
        assert_eq!(diags[0]["code"], "NotImplemented");
    }
}
