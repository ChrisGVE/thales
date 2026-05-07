//! Matrix decomposition operations: Lu, Qr, Cholesky, Svd,
//! Eigenvalues, Eigenvectors, SymbolicEigenvectors.

use crate::api::command::MatrixExpr as ApiMatrixExpr;
use crate::api::response::{
    DecompositionPart, EngineId, ResultEntry, ResultShape, ResultValue, StructuredResult,
};
use crate::ast::Expression;
use crate::numeric::compile::decompile;
use crate::numeric::trace::{TechniqueTag, Trace};

use super::{compile_matrix, flatten_matrices, mat_to_part, record, steps_from_trace};

pub(super) fn lu_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 1 {
        return Err(format!("Lu requires 1 operand, got {}", operands.len()));
    }
    let m = compile_matrix(&operands[0])?;
    record(
        narrate,
        trace,
        TechniqueTag::LuDecomposition,
        "LU decomposition",
    );
    m.lu_decompose()
        .map(|(l, u, perm)| {
            let structured = StructuredResult::Decomposition {
                parts: vec![
                    ("L".to_string(), mat_to_part(&l)),
                    ("U".to_string(), mat_to_part(&u)),
                    ("P".to_string(), DecompositionPart::Permutation(perm)),
                ],
            };
            let cells = flatten_matrices(&[&l, &u]);
            let (primary, alternatives) = split_cells(cells);
            ResultEntry {
                value: ResultValue::Symbolic(primary),
                structured: Some(structured),
                shape: ResultShape::Matrix,
                unit: None,
                steps: steps_from_trace(trace),
                alternatives,
                engine: EngineId::Matrix,
            }
        })
        .map_err(|e| format!("{:?}", e))
}

pub(super) fn qr_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 1 {
        return Err(format!("Qr requires 1 operand, got {}", operands.len()));
    }
    let m = compile_matrix(&operands[0])?;
    record(
        narrate,
        trace,
        TechniqueTag::QrDecomposition,
        "QR decomposition",
    );
    m.qr_decompose()
        .map(|(q, r)| {
            let structured = StructuredResult::Decomposition {
                parts: vec![
                    ("Q".to_string(), mat_to_part(&q)),
                    ("R".to_string(), mat_to_part(&r)),
                ],
            };
            let cells = flatten_matrices(&[&q, &r]);
            let (primary, alternatives) = split_cells(cells);
            ResultEntry {
                value: ResultValue::Symbolic(primary),
                structured: Some(structured),
                shape: ResultShape::Matrix,
                unit: None,
                steps: steps_from_trace(trace),
                alternatives,
                engine: EngineId::Matrix,
            }
        })
        .map_err(|e| format!("{:?}", e))
}

pub(super) fn cholesky_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 1 {
        return Err(format!(
            "Cholesky requires 1 operand, got {}",
            operands.len()
        ));
    }
    let m = compile_matrix(&operands[0])?;
    record(
        narrate,
        trace,
        TechniqueTag::CholeskyDecomposition,
        "Cholesky LLᵀ decomposition",
    );
    m.cholesky()
        .map(|l| {
            let structured = StructuredResult::Decomposition {
                parts: vec![("L".to_string(), mat_to_part(&l))],
            };
            let cells = flatten_matrices(&[&l]);
            let (primary, alternatives) = split_cells(cells);
            ResultEntry {
                value: ResultValue::Symbolic(primary),
                structured: Some(structured),
                shape: ResultShape::Matrix,
                unit: None,
                steps: steps_from_trace(trace),
                alternatives,
                engine: EngineId::Matrix,
            }
        })
        .map_err(|e| format!("{:?}", e))
}

pub(super) fn svd_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 1 {
        return Err(format!("Svd requires 1 operand, got {}", operands.len()));
    }
    let m = compile_matrix(&operands[0])?;
    record(
        narrate,
        trace,
        TechniqueTag::SvdDecomposition,
        "Singular value decomposition",
    );
    m.svd_numeric()
        .map(|(u, sigma_vec, vt)| {
            let sigma_exprs: Vec<Expression> =
                sigma_vec.iter().map(|&s| Expression::Float(s)).collect();
            let structured = StructuredResult::Decomposition {
                parts: vec![
                    ("U".to_string(), mat_to_part(&u)),
                    (
                        "Sigma".to_string(),
                        DecompositionPart::Matrix {
                            cols: sigma_exprs.len(),
                            rows: 1,
                            elements: sigma_exprs.clone(),
                        },
                    ),
                    ("V_transpose".to_string(), mat_to_part(&vt)),
                ],
            };
            let mut cells: Vec<Expression> = flatten_matrices(&[&u]);
            cells.extend(sigma_exprs);
            cells.extend(flatten_matrices(&[&vt]));
            let (primary, alternatives) = split_cells(cells);
            ResultEntry {
                value: ResultValue::Symbolic(primary),
                structured: Some(structured),
                shape: ResultShape::Matrix,
                unit: None,
                steps: steps_from_trace(trace),
                alternatives,
                engine: EngineId::Matrix,
            }
        })
        .map_err(|e| format!("{:?}", e))
}

pub(super) fn eigenvalues_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 1 {
        return Err(format!(
            "Eigenvalues requires 1 operand, got {}",
            operands.len()
        ));
    }
    let m = compile_matrix(&operands[0])?;
    record(
        narrate,
        trace,
        TechniqueTag::CharacteristicEquation,
        "Numeric eigenvalues",
    );
    m.eigenvalues_numeric()
        .map(|values| {
            let exprs: Vec<Expression> = values
                .into_iter()
                .map(|c| {
                    if c.im.abs() < 1e-10 {
                        Expression::Float(c.re)
                    } else {
                        Expression::Complex(c)
                    }
                })
                .collect();
            let (primary, alternatives) = split_cells(exprs);
            ResultEntry {
                value: ResultValue::Symbolic(primary),
                structured: None,
                shape: ResultShape::Set,
                unit: None,
                steps: steps_from_trace(trace),
                alternatives,
                engine: EngineId::Matrix,
            }
        })
        .map_err(|e| format!("{:?}", e))
}

pub(super) fn eigenvectors_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 1 {
        return Err(format!(
            "Eigenvectors requires 1 operand, got {}",
            operands.len()
        ));
    }
    let m = compile_matrix(&operands[0])?;
    record(
        narrate,
        trace,
        TechniqueTag::CharacteristicEquation,
        "Numeric eigenpairs",
    );
    m.eigenpairs_numeric()
        .map(|pairs| {
            let decomp_parts: Vec<(String, DecompositionPart)> = pairs
                .iter()
                .enumerate()
                .flat_map(|(i, (lambda, vec_components))| {
                    let label = format!("pair_{}", i + 1);
                    let ev_expr = if lambda.im == 0.0 {
                        Expression::Float(lambda.re)
                    } else {
                        Expression::Complex(*lambda)
                    };
                    let ev_part = (
                        format!("{}_eigenvalue", label),
                        DecompositionPart::Scalar(ev_expr),
                    );
                    let n = vec_components.len();
                    let evec_elements: Vec<Expression> = vec_components
                        .iter()
                        .map(|c| Expression::Float(*c))
                        .collect();
                    let evec_part = (
                        format!("{}_eigenvector", label),
                        DecompositionPart::Matrix {
                            elements: evec_elements,
                            rows: n,
                            cols: 1,
                        },
                    );
                    [ev_part, evec_part]
                })
                .collect();
            let structured = StructuredResult::Decomposition {
                parts: decomp_parts,
            };
            let cells: Vec<Expression> = pairs
                .iter()
                .flat_map(|(_lambda, vec_components)| {
                    vec_components.iter().map(|c| Expression::Float(*c))
                })
                .collect();
            let (primary, alternatives) = split_cells(cells);
            ResultEntry {
                value: ResultValue::Symbolic(primary),
                structured: Some(structured),
                shape: ResultShape::Matrix,
                unit: None,
                steps: steps_from_trace(trace),
                alternatives,
                engine: EngineId::Matrix,
            }
        })
        .map_err(|e| format!("{:?}", e))
}

pub(super) fn symbolic_eigenvectors_cmd(
    operands: &[ApiMatrixExpr],
    narrate: bool,
    trace: &mut Trace,
) -> Result<ResultEntry, String> {
    if operands.len() != 1 {
        return Err(format!(
            "SymbolicEigenvectors requires 1 operand, got {}",
            operands.len()
        ));
    }
    let m = compile_matrix(&operands[0])?;
    record(
        narrate,
        trace,
        TechniqueTag::SymbolicEigenvector,
        "Symbolic eigenvectors via characteristic polynomial",
    );
    m.symbolic_eigenvectors()
        .map(|pairs| {
            let decomp_parts: Vec<(String, DecompositionPart)> = pairs
                .iter()
                .enumerate()
                .flat_map(|(i, (eigenval, eigenvecs))| {
                    let label = format!("pair_{}", i + 1);
                    let ev_part = (
                        format!("{}_eigenvalue", label),
                        DecompositionPart::Scalar(decompile(eigenval)),
                    );
                    let vecs: Vec<(String, DecompositionPart)> = eigenvecs
                        .iter()
                        .enumerate()
                        .map(|(j, vec_mat)| {
                            (
                                format!("{}_eigenvector_{}", label, j + 1),
                                mat_to_part(vec_mat),
                            )
                        })
                        .collect();
                    let mut parts = vec![ev_part];
                    parts.extend(vecs);
                    parts
                })
                .collect();
            let structured = StructuredResult::Decomposition {
                parts: decomp_parts,
            };
            // Legacy primary: first eigenvector component from first pair.
            let primary_expr = pairs
                .first()
                .and_then(|(_, vecs)| vecs.first())
                .and_then(|v| {
                    if v.rows() > 0 && v.cols() > 0 {
                        Some(decompile(v.get(0, 0).expect("in-range cell")))
                    } else {
                        None
                    }
                })
                .unwrap_or(Expression::Integer(0));
            ResultEntry {
                value: ResultValue::Symbolic(primary_expr),
                structured: Some(structured),
                shape: ResultShape::Matrix,
                unit: None,
                steps: steps_from_trace(trace),
                alternatives: Vec::new(),
                engine: EngineId::Matrix,
            }
        })
        .map_err(|e| format!("{:?}", e))
}

/// Split a cell list into (primary, alternatives), returning `(Integer(0), [])` for
/// an empty list so callers never see an uninitialized primary value.
fn split_cells(cells: Vec<Expression>) -> (Expression, Vec<Expression>) {
    match cells.split_first() {
        Some((first, rest)) => (first.clone(), rest.to_vec()),
        None => (Expression::Integer(0), Vec::new()),
    }
}
