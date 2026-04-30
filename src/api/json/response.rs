//! Response serialisation — [`Response`] → `serde_json::Value`.

use serde_json::{json, Value};

use super::super::response::{
    DecompositionPart, NarratedStep, Response, ResultEntry, ResultKey, ResultValue,
    StructuredResult,
};

pub(super) fn response_to_json(response: &Response) -> Value {
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

pub(super) fn result_entry_to_json(key: &ResultKey, entry: &ResultEntry) -> Value {
    let key_json = result_key_to_json(key);
    json!({
        "key": key_json,
        "value": result_value_to_json(&entry.value),
        "structured": entry.structured.as_ref().map(structured_result_to_json),
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

fn result_key_to_json(key: &ResultKey) -> Value {
    match key {
        ResultKey::Single => json!("Single"),
        ResultKey::Branch(condition) => json!({
            "type": "Branch",
            "condition": format!("{:?}", condition),
        }),
        ResultKey::Component(name) => json!({
            "type": "Component",
            "name": name,
        }),
        ResultKey::ConvergenceDomain => json!("ConvergenceDomain"),
    }
}

pub(super) fn structured_result_to_json(s: &StructuredResult) -> Value {
    match s {
        StructuredResult::Scalar(e) => json!({
            "kind": "Scalar",
            "expr": format!("{}", e),
        }),
        StructuredResult::Labeled { label, value } => json!({
            "kind": "Labeled",
            "label": label,
            "value": format!("{}", value),
        }),
        StructuredResult::Decomposition { parts } => json!({
            "kind": "Decomposition",
            "parts": parts
                .iter()
                .map(|(name, part)| json!({
                    "name": name,
                    "part": decomposition_part_to_json(part),
                }))
                .collect::<Vec<_>>(),
        }),
        StructuredResult::CoefficientArray {
            coefficients,
            variable,
            center,
            order,
        } => json!({
            "kind": "CoefficientArray",
            "coefficients": coefficients
                .iter()
                .map(|e| format!("{}", e))
                .collect::<Vec<_>>(),
            "variable": variable,
            "center": center.as_ref().map(|e| format!("{}", e)),
            "order": order,
        }),
        StructuredResult::Branches { branches } => json!({
            "kind": "Branches",
            "branches": branches
                .iter()
                .map(|b| json!({
                    "label": b.label,
                    "condition": b.condition.as_ref().map(|c| format!("{:?}", c)),
                    "value": format!("{}", b.value),
                }))
                .collect::<Vec<_>>(),
        }),
        StructuredResult::Shaped {
            elements,
            shape,
            labels,
        } => json!({
            "kind": "Shaped",
            "elements": elements
                .iter()
                .map(|e| format!("{}", e))
                .collect::<Vec<_>>(),
            "shape": shape,
            "labels": labels,
        }),
        StructuredResult::TransformResult {
            expression,
            transform_variable,
            convergence,
        } => json!({
            "kind": "TransformResult",
            "expression": format!("{}", expression),
            "transform_variable": transform_variable,
            "convergence": convergence.as_ref().map(|e| format!("{}", e)),
        }),
        _ => json!({ "kind": "Unknown" }),
    }
}

fn decomposition_part_to_json(part: &DecompositionPart) -> Value {
    match part {
        DecompositionPart::Scalar(e) => json!({
            "kind": "Scalar",
            "expr": format!("{}", e),
        }),
        DecompositionPart::Matrix {
            elements,
            rows,
            cols,
        } => json!({
            "kind": "Matrix",
            "elements": elements
                .iter()
                .map(|e| format!("{}", e))
                .collect::<Vec<_>>(),
            "rows": rows,
            "cols": cols,
        }),
        DecompositionPart::Permutation(p) => json!({
            "kind": "Permutation",
            "indices": p,
        }),
    }
}

fn result_value_to_json(value: &ResultValue) -> Value {
    match value {
        ResultValue::Symbolic(e) => json!({
            "kind": "Symbolic",
            "expr": format!("{}", e),
        }),
        ResultValue::Numeric {
            value,
            precision,
            method,
        } => json!({
            "kind": "Numeric",
            "expr": format!("{}", value),
            "decimal_digits": precision.decimal_digits,
            "method": format!("{:?}", method),
        }),
        ResultValue::Hybrid {
            last_symbolic,
            numeric,
            precision,
            method,
        } => json!({
            "kind": "Hybrid",
            "last_symbolic": format!("{}", last_symbolic),
            "numeric": format!("{}", numeric),
            "decimal_digits": precision.decimal_digits,
            "method": format!("{:?}", method),
        }),
        ResultValue::Unsolved { reason } => json!({
            "kind": "Unsolved",
            "reason_md": reason.fallback_md,
            "template_id": reason.template_id,
        }),
        ResultValue::NoSolution { reason, domain } => json!({
            "kind": "NoSolution",
            "domain": format!("{:?}", domain),
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
