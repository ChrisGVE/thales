//! F1e matrix-operations dispatcher.
//!
//! Compiles api-level [`MatrixExpr`] operands at the I/O seam, defers to the
//! Arc<Expr>-typed engines in `crate::matrix`, and decompiles results back to
//! `Expression`. All operations are fully wired; no stubs remain.

use std::sync::Arc;

use crate::api::command::{MatrixExpr as ApiMatrixExpr, MatrixOp};
use crate::api::response::{
    DecompositionPart, EngineId, Response, ResultEntry, ResultKey, ResultShape, ResultValue,
};
use crate::ast::Expression;
use crate::matrix::MatrixExpr as CoreMatrix;
use crate::numeric::compile::{compile, decompile};
use crate::numeric::expr::Expr;
use crate::numeric::trace::{Step, TechniqueTag, Trace};

use super::helpers::{engine_error, steps_from_trace, symbolic_entry};

mod analysis;
mod basic;
mod decomp;

pub(crate) fn compile_matrix(operand: &ApiMatrixExpr) -> Result<CoreMatrix, String> {
    match operand {
        ApiMatrixExpr::Scalar(_) => Err("expected matrix operand, got scalar".to_string()),
        ApiMatrixExpr::Vector(cells) => {
            let elements: Vec<Vec<Arc<Expr>>> = cells.iter().map(|c| vec![compile(c)]).collect();
            CoreMatrix::from_expr_elements(elements).map_err(|e| format!("{:?}", e))
        }
        ApiMatrixExpr::Matrix(rows) => {
            let elements: Vec<Vec<Arc<Expr>>> = rows
                .iter()
                .map(|row| row.iter().map(compile).collect())
                .collect();
            CoreMatrix::from_expr_elements(elements).map_err(|e| format!("{:?}", e))
        }
    }
}

pub(crate) fn compile_scalar(operand: &ApiMatrixExpr) -> Result<Arc<Expr>, String> {
    match operand {
        ApiMatrixExpr::Scalar(e) => Ok(compile(e)),
        _ => Err("expected scalar operand".to_string()),
    }
}

pub(crate) fn matrix_to_value(m: &CoreMatrix) -> ResultEntry {
    let elements: Vec<Expression> = (0..m.rows())
        .flat_map(|r| (0..m.cols()).map(move |c| decompile(m.get(r, c).expect("in-range cell"))))
        .collect();
    let (primary, alternatives) = match elements.split_first() {
        Some((first, rest)) => (first.clone(), rest.to_vec()),
        None => (Expression::Integer(0), Vec::new()),
    };
    ResultEntry {
        value: ResultValue::Symbolic(primary),
        structured: None,
        shape: ResultShape::Matrix,
        unit: None,
        steps: Vec::new(),
        alternatives,
        engine: EngineId::Matrix,
    }
}

pub(crate) fn record(narrate: bool, trace: &mut Trace, tag: TechniqueTag, detail: impl Into<String>) {
    if narrate {
        trace.push(Step::new(tag, detail.into()));
    }
}

/// Convert a `CoreMatrix` into a `DecompositionPart::Matrix`.
pub(crate) fn mat_to_part(mat: &CoreMatrix) -> DecompositionPart {
    let rows = mat.rows();
    let cols = mat.cols();
    let elements: Vec<Expression> = (0..rows)
        .flat_map(|r| {
            (0..cols).map(move |c| decompile(mat.get(r, c).expect("in-range cell")))
        })
        .collect();
    DecompositionPart::Matrix { elements, rows, cols }
}

/// Build a flat cell list from a slice of matrices (row-major, matrix-by-matrix).
pub(crate) fn flatten_matrices(matrices: &[&CoreMatrix]) -> Vec<Expression> {
    matrices
        .iter()
        .flat_map(|m| {
            (0..m.rows()).flat_map(move |r| {
                (0..m.cols()).map(move |c| decompile(m.get(r, c).expect("in-range cell")))
            })
        })
        .collect()
}

pub(super) fn matrix_cmd(op: MatrixOp, operands: &[ApiMatrixExpr], narrate: bool) -> Response {
    let mut trace = Trace::new();

    let result: Result<ResultEntry, String> = match op {
        MatrixOp::Add => basic::add_cmd(operands, narrate, &mut trace),
        MatrixOp::Subtract => basic::subtract_cmd(operands, narrate, &mut trace),
        MatrixOp::Multiply => basic::multiply_cmd(operands, narrate, &mut trace),
        MatrixOp::ScalarMultiply => basic::scalar_multiply_cmd(operands, narrate, &mut trace),
        MatrixOp::Transpose => basic::transpose_cmd(operands, narrate, &mut trace),
        MatrixOp::Trace => basic::trace_cmd(operands, narrate, &mut trace),
        MatrixOp::SolveLinear => basic::solve_linear_cmd(operands, narrate, &mut trace),
        MatrixOp::KroneckerProduct => basic::kronecker_cmd(operands, narrate, &mut trace),
        MatrixOp::Lu => decomp::lu_cmd(operands, narrate, &mut trace),
        MatrixOp::Qr => decomp::qr_cmd(operands, narrate, &mut trace),
        MatrixOp::Cholesky => decomp::cholesky_cmd(operands, narrate, &mut trace),
        MatrixOp::Svd => decomp::svd_cmd(operands, narrate, &mut trace),
        MatrixOp::Eigenvalues => decomp::eigenvalues_cmd(operands, narrate, &mut trace),
        MatrixOp::Eigenvectors => decomp::eigenvectors_cmd(operands, narrate, &mut trace),
        MatrixOp::SymbolicEigenvectors => {
            decomp::symbolic_eigenvectors_cmd(operands, narrate, &mut trace)
        }
        MatrixOp::Determinant => analysis::determinant_cmd(operands, narrate, &mut trace),
        MatrixOp::Inverse => analysis::inverse_cmd(operands, narrate, &mut trace),
        MatrixOp::Rank => analysis::rank_cmd(operands, narrate, &mut trace),
        MatrixOp::NullSpace => analysis::null_space_cmd(operands, narrate, &mut trace),
        MatrixOp::ColumnSpace => analysis::column_space_cmd(operands, narrate, &mut trace),
        MatrixOp::RowEchelon => analysis::row_echelon_cmd(operands, narrate, &mut trace),
        MatrixOp::CharacteristicPolynomial => {
            analysis::characteristic_polynomial_cmd(operands, narrate, &mut trace)
        }
        MatrixOp::MinimalPolynomial => {
            analysis::minimal_polynomial_cmd(operands, narrate, &mut trace)
        }
        MatrixOp::QuadraticFormClassify => {
            analysis::quadratic_form_classify_cmd(operands, narrate, &mut trace)
        }
    };

    match result {
        Ok(entry) => {
            let mut r = Response::default();
            r.results.push((ResultKey::Single, entry));
            r.meta.engine_trace.push(EngineId::Matrix);
            r
        }
        Err(msg) => engine_error("command.matrix", msg),
    }
}
