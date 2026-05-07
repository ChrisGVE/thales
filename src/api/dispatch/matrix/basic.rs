//! Basic matrix operations: Add, Subtract, Multiply, ScalarMultiply,
//! Transpose, Trace, SolveLinear, KroneckerProduct.

use crate::api::command::MatrixExpr as ApiMatrixExpr;
use crate::api::response::{EngineId, ResultEntry};
use crate::numeric::compile::decompile;
use crate::numeric::trace::{TechniqueTag, Trace};

use super::{
    compile_matrix, compile_scalar, matrix_to_value, record, steps_from_trace, symbolic_entry,
};

pub(super) fn add_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 2 {
        return Err(format!("Add requires 2 operands, got {}", operands.len()));
    }
    let a = compile_matrix(&operands[0])?;
    let b = compile_matrix(&operands[1])?;
    record(narrate, trace, TechniqueTag::PowerRule, "Matrix addition");
    a.add(&b)
        .map(|m| matrix_to_value(&m))
        .map_err(|e| format!("{:?}", e))
}

pub(super) fn subtract_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 2 {
        return Err(format!(
            "Subtract requires 2 operands, got {}",
            operands.len()
        ));
    }
    let a = compile_matrix(&operands[0])?;
    let b = compile_matrix(&operands[1])?;
    record(
        narrate,
        trace,
        TechniqueTag::PowerRule,
        "Matrix subtraction",
    );
    a.sub(&b)
        .map(|m| matrix_to_value(&m))
        .map_err(|e| format!("{:?}", e))
}

pub(super) fn multiply_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 2 {
        return Err(format!(
            "Multiply requires 2 operands, got {}",
            operands.len()
        ));
    }
    let a = compile_matrix(&operands[0])?;
    let b = compile_matrix(&operands[1])?;
    record(
        narrate,
        trace,
        TechniqueTag::PowerRule,
        "Matrix multiplication",
    );
    a.mul(&b)
        .map(|m| matrix_to_value(&m))
        .map_err(|e| format!("{:?}", e))
}

pub(super) fn scalar_multiply_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 2 {
        return Err(format!(
            "ScalarMultiply requires 2 operands (scalar, matrix), got {}",
            operands.len()
        ));
    }
    let scalar = compile_scalar(&operands[0])?;
    let m = compile_matrix(&operands[1])?;
    record(
        narrate,
        trace,
        TechniqueTag::PowerRule,
        "Scalar-matrix multiplication",
    );
    Ok(matrix_to_value(&m.scalar_mul(&scalar)))
}

pub(super) fn transpose_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 1 {
        return Err(format!(
            "Transpose requires 1 operand, got {}",
            operands.len()
        ));
    }
    let m = compile_matrix(&operands[0])?;
    record(narrate, trace, TechniqueTag::PowerRule, "Matrix transpose");
    Ok(matrix_to_value(&m.transpose()))
}

pub(super) fn trace_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 1 {
        return Err(format!("Trace requires 1 operand, got {}", operands.len()));
    }
    let m = compile_matrix(&operands[0])?;
    record(
        narrate,
        trace,
        TechniqueTag::PowerRule,
        "Sum of diagonal entries",
    );
    m.trace()
        .map(|t| {
            let value = decompile(&t);
            symbolic_entry(value, EngineId::Matrix, steps_from_trace(trace))
        })
        .map_err(|e| format!("{:?}", e))
}

pub(super) fn solve_linear_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 2 {
        return Err(format!(
            "SolveLinear requires 2 operands (A, b), got {}",
            operands.len()
        ));
    }
    let a = compile_matrix(&operands[0])?;
    let b = compile_matrix(&operands[1])?;
    record(
        narrate,
        trace,
        TechniqueTag::LuDecomposition,
        "Solve Ax = b via LU",
    );
    a.solve_system(&b)
        .map(|x| matrix_to_value(&x))
        .map_err(|e| format!("{:?}", e))
}

pub(super) fn kronecker_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 2 {
        return Err(format!(
            "KroneckerProduct requires 2 operands, got {}",
            operands.len()
        ));
    }
    let a = compile_matrix(&operands[0])?;
    let b = compile_matrix(&operands[1])?;
    record(narrate, trace, TechniqueTag::PowerRule, "Kronecker product");
    a.kronecker_product(&b)
        .map(|result| {
            let mut entry = matrix_to_value(&result);
            entry.steps = steps_from_trace(trace);
            entry
        })
        .map_err(|e| format!("{:?}", e))
}
