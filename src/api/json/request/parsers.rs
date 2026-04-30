//! String-to-type parsers used during JSON → Command conversion.

use serde_json::Value;

use crate::ast::Expression;
use crate::parser::parse_equation;

use super::super::super::command::{IdentityId, MatrixExpr, MatrixOp, OptSense, Side, SpecialKind};
use super::super::super::domain::Domain;
use super::super::super::request::SolveMode;

// ── Expression parsing ────────────────────────────────────────────────────────

/// Parse an expression string. When the input contains `=`, parse as an
/// equation and return `lhs - rhs` so callers that expect a bare expression
/// handle it uniformly. Falls back to [`crate::parser::parse_expression`] for
/// plain expressions.
pub(in super::super) fn parse_expr_str(s: &str) -> Result<Expression, String> {
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

// ── Enum parsers ──────────────────────────────────────────────────────────────

pub(super) fn parse_solve_mode(s: &str) -> Result<SolveMode, String> {
    match s {
        "Symbolic" => Ok(SolveMode::Symbolic),
        "Numeric" => Ok(SolveMode::Numeric),
        "PreferSymbolic" => Ok(SolveMode::PreferSymbolic),
        other => Err(format!("unknown SolveMode `{}`", other)),
    }
}

pub(super) fn parse_domain_str(s: &str) -> Result<Domain, String> {
    match s {
        "Natural" | "natural" => Ok(Domain::natural()),
        "Integer" | "integer" => Ok(Domain::integer()),
        "Rational" | "rational" => Ok(Domain::rational()),
        "Real" | "real" => Ok(Domain::real()),
        "RealPositive" | "real_positive" => Ok(Domain::real_positive()),
        "RealNonNegative" | "real_non_negative" => Ok(Domain::real_non_negative()),
        "Complex" | "complex" => Ok(Domain::complex()),
        other => Err(format!("unknown domain `{}`", other)),
    }
}

pub(super) fn parse_side(s: &str) -> Result<Side, String> {
    match s {
        "Left" => Ok(Side::Left),
        "Right" => Ok(Side::Right),
        other => Err(format!("unknown Side `{}`", other)),
    }
}

pub(super) fn parse_special_kind(s: &str) -> Result<SpecialKind, String> {
    match s {
        "Gamma" => Ok(SpecialKind::Gamma),
        "Beta" => Ok(SpecialKind::Beta),
        "Erf" => Ok(SpecialKind::Erf),
        "Erfc" => Ok(SpecialKind::Erfc),
        other => Err(format!("unknown SpecialKind `{}`", other)),
    }
}

pub(super) fn parse_matrix_op(s: &str) -> Result<MatrixOp, String> {
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

pub(super) fn parse_identity_id(s: &str) -> Result<IdentityId, String> {
    match s {
        "PythagoreanTrig" => Ok(IdentityId::PythagoreanTrig),
        "PythagoreanHyp" => Ok(IdentityId::PythagoreanHyp),
        "DoubleAngleSin" => Ok(IdentityId::DoubleAngleSin),
        "DoubleAngleCos" => Ok(IdentityId::DoubleAngleCos),
        "SumToProductSin" => Ok(IdentityId::SumToProductSin),
        "SumToProductCos" => Ok(IdentityId::SumToProductCos),
        "LogProduct" => Ok(IdentityId::LogProduct),
        "LogPower" => Ok(IdentityId::LogPower),
        "ExpSum" => Ok(IdentityId::ExpSum),
        "Euler" => Ok(IdentityId::Euler),
        "DeMoivre" => Ok(IdentityId::DeMoivre),
        "DifferenceOfSquares" => Ok(IdentityId::DifferenceOfSquares),
        "SumOfCubes" => Ok(IdentityId::SumOfCubes),
        other => Err(format!("unknown IdentityId `{}`", other)),
    }
}

pub(super) fn parse_opt_sense(s: &str) -> Result<OptSense, String> {
    match s {
        "Minimize" => Ok(OptSense::Minimize),
        "Maximize" => Ok(OptSense::Maximize),
        other => Err(format!("unknown OptSense `{}`", other)),
    }
}

// ── Composite parsers ─────────────────────────────────────────────────────────

pub(super) fn parse_matrix_expr(val: &Value, index: usize) -> Result<MatrixExpr, String> {
    if let Some(s) = val.as_str() {
        return Ok(MatrixExpr::Scalar(parse_expr_str(s)?));
    }
    if let Some(obj) = val.as_object() {
        if let Some(rows) = obj.get("rows").and_then(|v| v.as_array()) {
            let parsed_rows: Result<Vec<Vec<Expression>>, String> = rows
                .iter()
                .enumerate()
                .map(|(r, row)| {
                    let cols = row.as_array().ok_or_else(|| {
                        format!("operands[{}].rows[{}]: expected array", index, r)
                    })?;
                    cols.iter()
                        .enumerate()
                        .map(|(c, cell)| {
                            let s = cell.as_str().ok_or_else(|| {
                                format!("operands[{}].rows[{}][{}]: expected string", index, r, c)
                            })?;
                            parse_expr_str(s)
                        })
                        .collect()
                })
                .collect();
            return Ok(MatrixExpr::Matrix(parsed_rows?));
        }
        if let Some(elems) = obj.get("elements").and_then(|v| v.as_array()) {
            let parsed: Result<Vec<Expression>, String> = elems
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    let s = e.as_str().ok_or_else(|| {
                        format!("operands[{}].elements[{}]: expected string", index, i)
                    })?;
                    parse_expr_str(s)
                })
                .collect();
            return Ok(MatrixExpr::Vector(parsed?));
        }
    }
    Err(format!(
        "operands[{}]: expected string (scalar), or object with 'rows' or 'elements'",
        index
    ))
}
