//! Matrix analysis operations: Determinant, Inverse, Rank, NullSpace,
//! ColumnSpace, RowEchelon, CharacteristicPolynomial, MinimalPolynomial,
//! QuadraticFormClassify.

use crate::api::command::MatrixExpr as ApiMatrixExpr;
use crate::api::response::{EngineId, ResultEntry, ResultShape, ResultValue, StructuredResult};
use crate::ast::Expression;
use crate::matrix::Definiteness;
use crate::numeric::compile::decompile;
use crate::numeric::trace::{TechniqueTag, Trace};

use super::{
    compile_matrix, mat_to_part, matrix_to_value, record, steps_from_trace, symbolic_entry,
};

pub(super) fn determinant_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 1 {
        return Err(format!(
            "Determinant requires 1 operand, got {}",
            operands.len()
        ));
    }
    let m = compile_matrix(&operands[0])?;
    record(
        narrate,
        trace,
        TechniqueTag::PowerRule,
        "Determinant via Laplace expansion",
    );
    m.determinant()
        .map(|d| {
            let value = decompile(&d);
            symbolic_entry(value, EngineId::Matrix, steps_from_trace(trace))
        })
        .map_err(|e| format!("{:?}", e))
}

pub(super) fn inverse_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 1 {
        return Err(format!(
            "Inverse requires 1 operand, got {}",
            operands.len()
        ));
    }
    let m = compile_matrix(&operands[0])?;
    record(
        narrate,
        trace,
        TechniqueTag::MatrixInverse,
        "Matrix inverse via adjugate",
    );
    m.inverse()
        .map(|inv| matrix_to_value(&inv))
        .map_err(|e| format!("{:?}", e))
}

pub(super) fn rank_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 1 {
        return Err(format!("Rank requires 1 operand, got {}", operands.len()));
    }
    let m = compile_matrix(&operands[0])?;
    record(
        narrate,
        trace,
        TechniqueTag::RankComputation,
        "Rank via pivot count in row echelon form",
    );
    m.rank()
        .map(|rank| {
            let value = Expression::Integer(rank as i64);
            symbolic_entry(value, EngineId::Matrix, steps_from_trace(trace))
        })
        .map_err(|e| format!("{:?}", e))
}

pub(super) fn null_space_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 1 {
        return Err(format!(
            "NullSpace requires 1 operand, got {}",
            operands.len()
        ));
    }
    let m = compile_matrix(&operands[0])?;
    record(
        narrate,
        trace,
        TechniqueTag::NullSpaceComputation,
        "Kernel basis via RREF back-substitution",
    );
    m.kernel()
        .map(|basis| basis_to_entry(basis, trace))
        .map_err(|e| format!("{:?}", e))
}

pub(super) fn column_space_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 1 {
        return Err(format!(
            "ColumnSpace requires 1 operand, got {}",
            operands.len()
        ));
    }
    let m = compile_matrix(&operands[0])?;
    record(
        narrate,
        trace,
        TechniqueTag::ColumnSpaceComputation,
        "Column-space basis via pivot columns in RREF",
    );
    m.column_space()
        .map(|basis| basis_to_entry(basis, trace))
        .map_err(|e| format!("{:?}", e))
}

pub(super) fn row_echelon_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 1 {
        return Err(format!(
            "RowEchelon requires 1 operand, got {}",
            operands.len()
        ));
    }
    let m = compile_matrix(&operands[0])?;
    record(
        narrate,
        trace,
        TechniqueTag::GaussJordanElimination,
        "Reduced row echelon form via Gauss-Jordan elimination",
    );
    m.rref()
        .map(|(mat, _pivots)| {
            let mut entry = matrix_to_value(&mat);
            entry.steps = steps_from_trace(trace);
            entry
        })
        .map_err(|e| format!("{:?}", e))
}

pub(super) fn characteristic_polynomial_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 1 {
        return Err(format!(
            "CharacteristicPolynomial requires 1 operand, got {}",
            operands.len()
        ));
    }
    let m = compile_matrix(&operands[0])?;
    record(
        narrate,
        trace,
        TechniqueTag::CharacteristicEquation,
        "Characteristic polynomial det(λI − A)",
    );
    m.characteristic_polynomial("lambda")
        .map(|poly| {
            let value = decompile(&poly);
            symbolic_entry(value, EngineId::Matrix, steps_from_trace(trace))
        })
        .map_err(|e| format!("{:?}", e))
}

pub(super) fn minimal_polynomial_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 1 {
        return Err(format!(
            "MinimalPolynomial requires 1 operand, got {}",
            operands.len()
        ));
    }
    let m = compile_matrix(&operands[0])?;
    record(
        narrate,
        trace,
        TechniqueTag::MinimalPolynomial,
        "Minimal polynomial via Cayley-Hamilton annihilator",
    );
    m.minimal_polynomial("lambda")
        .map(|poly| {
            let value = decompile(&poly);
            symbolic_entry(value, EngineId::Matrix, steps_from_trace(trace))
        })
        .map_err(|e| format!("{:?}", e))
}

pub(super) fn quadratic_form_classify_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 1 {
        return Err(format!(
            "QuadraticFormClassify requires 1 operand, got {}",
            operands.len()
        ));
    }
    let m = compile_matrix(&operands[0])?;
    record(
        narrate,
        trace,
        TechniqueTag::QuadraticFormAnalysis,
        "Definiteness classification via Sylvester's criterion",
    );
    m.classify_definiteness()
        .map(|d| {
            let label = definiteness_label(d);
            let value = Expression::Variable(crate::ast::Variable::new(label));
            symbolic_entry(value, EngineId::Matrix, steps_from_trace(trace))
        })
        .map_err(|e| format!("{:?}", e))
}

// ── private helpers ───────────────────────────────────────────────────────────

/// Convert a basis (Vec of column vectors) into a `ResultEntry` with a
/// typed Decomposition structured result.
fn basis_to_entry(basis: Vec<crate::matrix::MatrixExpr>, trace: &mut Trace) -> ResultEntry {
    let parts: Vec<(String, crate::api::response::DecompositionPart)> = basis
        .iter()
        .enumerate()
        .map(|(i, v)| (format!("basis_{}", i + 1), mat_to_part(v)))
        .collect();
    let cells: Vec<Expression> = basis
        .iter()
        .flat_map(|v| {
            (0..v.rows()).flat_map(move |r| {
                (0..v.cols()).map(move |c| decompile(v.get(r, c).expect("in-range")))
            })
        })
        .collect();
    let (primary, alternatives) = match cells.split_first() {
        Some((first, rest)) => (first.clone(), rest.to_vec()),
        None => (Expression::Integer(0), Vec::new()),
    };
    ResultEntry {
        value: ResultValue::Symbolic(primary),
        structured: Some(StructuredResult::Decomposition { parts }),
        shape: ResultShape::Matrix,
        unit: None,
        steps: steps_from_trace(trace),
        alternatives,
        engine: EngineId::Matrix,
    }
}

fn definiteness_label(d: Definiteness) -> &'static str {
    match d {
        Definiteness::PositiveDefinite => "PositiveDefinite",
        Definiteness::PositiveSemidefinite => "PositiveSemidefinite",
        Definiteness::NegativeDefinite => "NegativeDefinite",
        Definiteness::NegativeSemidefinite => "NegativeSemidefinite",
        Definiteness::Indefinite => "Indefinite",
        Definiteness::Unknown => "Unknown",
    }
}
