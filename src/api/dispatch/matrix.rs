//! F1e matrix-operations dispatcher.
//!
//! Compiles api-level [`MatrixExpr`] operands at the I/O seam, defers to the
//! Arc<Expr>-typed engines in `crate::matrix`, and decompiles results back to
//! `Expression`. Operations whose engines are not yet present in this crate
//! (Rank, NullSpace, Qr) surface as engine errors rather than dispatcher
//! stubs.

use std::sync::Arc;

use crate::api::command::{MatrixExpr as ApiMatrixExpr, MatrixOp};
use crate::api::response::{
    DecompositionPart, EngineId, Response, ResultEntry, ResultKey, ResultShape, ResultValue,
    StructuredResult,
};
use crate::ast::Expression;
use crate::matrix::MatrixExpr as CoreMatrix;
use crate::numeric::compile::{compile, decompile};
use crate::numeric::expr::Expr;
use crate::numeric::trace::{Step, TechniqueTag, Trace};

use super::helpers::{engine_error, steps_from_trace, symbolic_entry};

fn compile_matrix(operand: &ApiMatrixExpr) -> Result<CoreMatrix, String> {
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

fn compile_scalar(operand: &ApiMatrixExpr) -> Result<Arc<Expr>, String> {
    match operand {
        ApiMatrixExpr::Scalar(e) => Ok(compile(e)),
        _ => Err("expected scalar operand".to_string()),
    }
}

fn matrix_to_value(m: &CoreMatrix) -> ResultEntry {
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

fn record(narrate: bool, trace: &mut Trace, tag: TechniqueTag, detail: impl Into<String>) {
    if narrate {
        trace.push(Step::new(tag, detail.into()));
    }
}

pub(super) fn matrix_cmd(op: MatrixOp, operands: &[ApiMatrixExpr], narrate: bool) -> Response {
    let mut trace = Trace::new();

    let result: Result<ResultEntry, String> = (|| -> Result<ResultEntry, String> {
        match op {
            MatrixOp::Add => {
                if operands.len() != 2 {
                    Err(format!("Add requires 2 operands, got {}", operands.len()))
                } else {
                    let a = compile_matrix(&operands[0])?;
                    let b = compile_matrix(&operands[1])?;
                    record(
                        narrate,
                        &mut trace,
                        TechniqueTag::PowerRule,
                        "Matrix addition",
                    );
                    a.add(&b)
                        .map(|m| matrix_to_value(&m))
                        .map_err(|e| format!("{:?}", e))
                }
            }
            MatrixOp::Subtract => {
                if operands.len() != 2 {
                    Err(format!(
                        "Subtract requires 2 operands, got {}",
                        operands.len()
                    ))
                } else {
                    let a = compile_matrix(&operands[0])?;
                    let b = compile_matrix(&operands[1])?;
                    record(
                        narrate,
                        &mut trace,
                        TechniqueTag::PowerRule,
                        "Matrix subtraction",
                    );
                    a.sub(&b)
                        .map(|m| matrix_to_value(&m))
                        .map_err(|e| format!("{:?}", e))
                }
            }
            MatrixOp::Multiply => {
                if operands.len() != 2 {
                    Err(format!(
                        "Multiply requires 2 operands, got {}",
                        operands.len()
                    ))
                } else {
                    let a = compile_matrix(&operands[0])?;
                    let b = compile_matrix(&operands[1])?;
                    record(
                        narrate,
                        &mut trace,
                        TechniqueTag::PowerRule,
                        "Matrix multiplication",
                    );
                    a.mul(&b)
                        .map(|m| matrix_to_value(&m))
                        .map_err(|e| format!("{:?}", e))
                }
            }
            MatrixOp::ScalarMultiply => {
                if operands.len() != 2 {
                    Err(format!(
                        "ScalarMultiply requires 2 operands (scalar, matrix), got {}",
                        operands.len()
                    ))
                } else {
                    let scalar = compile_scalar(&operands[0])?;
                    let m = compile_matrix(&operands[1])?;
                    record(
                        narrate,
                        &mut trace,
                        TechniqueTag::PowerRule,
                        "Scalar-matrix multiplication",
                    );
                    Ok(matrix_to_value(&m.scalar_mul(&scalar)))
                }
            }
            MatrixOp::Transpose => {
                if operands.len() != 1 {
                    Err(format!(
                        "Transpose requires 1 operand, got {}",
                        operands.len()
                    ))
                } else {
                    let m = compile_matrix(&operands[0])?;
                    record(
                        narrate,
                        &mut trace,
                        TechniqueTag::PowerRule,
                        "Matrix transpose",
                    );
                    Ok(matrix_to_value(&m.transpose()))
                }
            }
            MatrixOp::Determinant => {
                if operands.len() != 1 {
                    Err(format!(
                        "Determinant requires 1 operand, got {}",
                        operands.len()
                    ))
                } else {
                    let m = compile_matrix(&operands[0])?;
                    record(
                        narrate,
                        &mut trace,
                        TechniqueTag::PowerRule,
                        "Determinant via Laplace expansion",
                    );
                    m.determinant()
                        .map(|d| {
                            let value = decompile(&d);
                            symbolic_entry(value, EngineId::Matrix, steps_from_trace(&trace))
                        })
                        .map_err(|e| format!("{:?}", e))
                }
            }
            MatrixOp::Inverse => {
                if operands.len() != 1 {
                    Err(format!(
                        "Inverse requires 1 operand, got {}",
                        operands.len()
                    ))
                } else {
                    let m = compile_matrix(&operands[0])?;
                    record(
                        narrate,
                        &mut trace,
                        TechniqueTag::MatrixInverse,
                        "Matrix inverse via adjugate",
                    );
                    m.inverse()
                        .map(|inv| matrix_to_value(&inv))
                        .map_err(|e| format!("{:?}", e))
                }
            }
            MatrixOp::Trace => {
                if operands.len() != 1 {
                    Err(format!("Trace requires 1 operand, got {}", operands.len()))
                } else {
                    let m = compile_matrix(&operands[0])?;
                    record(
                        narrate,
                        &mut trace,
                        TechniqueTag::PowerRule,
                        "Sum of diagonal entries",
                    );
                    m.trace()
                        .map(|t| {
                            let value = decompile(&t);
                            symbolic_entry(value, EngineId::Matrix, steps_from_trace(&trace))
                        })
                        .map_err(|e| format!("{:?}", e))
                }
            }
            MatrixOp::Eigenvalues => {
                if operands.len() != 1 {
                    Err(format!(
                        "Eigenvalues requires 1 operand, got {}",
                        operands.len()
                    ))
                } else {
                    let m = compile_matrix(&operands[0])?;
                    record(
                        narrate,
                        &mut trace,
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
                            let (primary, alternatives) = match exprs.split_first() {
                                Some((first, rest)) => (first.clone(), rest.to_vec()),
                                None => (Expression::Integer(0), Vec::new()),
                            };
                            ResultEntry {
                                value: ResultValue::Symbolic(primary),
                                structured: None,
                                shape: ResultShape::Set,
                                unit: None,
                                steps: steps_from_trace(&trace),
                                alternatives,
                                engine: EngineId::Matrix,
                            }
                        })
                        .map_err(|e| format!("{:?}", e))
                }
            }
            MatrixOp::Eigenvectors => {
                if operands.len() != 1 {
                    Err(format!(
                        "Eigenvectors requires 1 operand, got {}",
                        operands.len()
                    ))
                } else {
                    let m = compile_matrix(&operands[0])?;
                    record(
                        narrate,
                        &mut trace,
                        TechniqueTag::CharacteristicEquation,
                        "Numeric eigenpairs",
                    );
                    m.eigenpairs_numeric()
                        .map(|pairs| {
                            // Build typed Decomposition: one (eigenvalue, eigenvector) pair
                            // per column, labeled "pair_1", "pair_2", …
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
                            // Legacy flat cells: eigenvectors in column-major order.
                            let cells: Vec<Expression> = pairs
                                .iter()
                                .flat_map(|(_lambda, vec_components)| {
                                    vec_components.iter().map(|c| Expression::Float(*c))
                                })
                                .collect();
                            let (primary, alternatives) = match cells.split_first() {
                                Some((first, rest)) => (first.clone(), rest.to_vec()),
                                None => (Expression::Integer(0), Vec::new()),
                            };
                            ResultEntry {
                                value: ResultValue::Symbolic(primary),
                                structured: Some(structured),
                                shape: ResultShape::Matrix,
                                unit: None,
                                steps: steps_from_trace(&trace),
                                alternatives,
                                engine: EngineId::Matrix,
                            }
                        })
                        .map_err(|e| format!("{:?}", e))
                }
            }
            MatrixOp::Lu => {
                if operands.len() != 1 {
                    Err(format!("Lu requires 1 operand, got {}", operands.len()))
                } else {
                    let m = compile_matrix(&operands[0])?;
                    record(
                        narrate,
                        &mut trace,
                        TechniqueTag::LuDecomposition,
                        "LU decomposition",
                    );
                    m.lu_decompose()
                        .map(|(l, u, perm)| {
                            // Build typed Decomposition parts: L, U, P.
                            let mat_to_part = |mat: &CoreMatrix| -> DecompositionPart {
                                let rows = mat.rows();
                                let cols = mat.cols();
                                let elements: Vec<Expression> = (0..rows)
                                    .flat_map(|r| {
                                        (0..cols).map(move |c| {
                                            decompile(mat.get(r, c).expect("in-range cell"))
                                        })
                                    })
                                    .collect();
                                DecompositionPart::Matrix {
                                    elements,
                                    rows,
                                    cols,
                                }
                            };
                            let structured = StructuredResult::Decomposition {
                                parts: vec![
                                    ("L".to_string(), mat_to_part(&l)),
                                    ("U".to_string(), mat_to_part(&u)),
                                    ("P".to_string(), DecompositionPart::Permutation(perm)),
                                ],
                            };
                            // Flatten L then U into row-major cells for legacy fields.
                            let mut cells: Vec<Expression> = Vec::new();
                            for matrix_part in [&l, &u] {
                                for r in 0..matrix_part.rows() {
                                    for c in 0..matrix_part.cols() {
                                        cells.push(decompile(
                                            matrix_part.get(r, c).expect("in-range cell"),
                                        ));
                                    }
                                }
                            }
                            let (primary, alternatives) = match cells.split_first() {
                                Some((first, rest)) => (first.clone(), rest.to_vec()),
                                None => (Expression::Integer(0), Vec::new()),
                            };
                            ResultEntry {
                                value: ResultValue::Symbolic(primary),
                                structured: Some(structured),
                                shape: ResultShape::Matrix,
                                unit: None,
                                steps: steps_from_trace(&trace),
                                alternatives,
                                engine: EngineId::Matrix,
                            }
                        })
                        .map_err(|e| format!("{:?}", e))
                }
            }
            MatrixOp::SolveLinear => {
                if operands.len() != 2 {
                    Err(format!(
                        "SolveLinear requires 2 operands (A, b), got {}",
                        operands.len()
                    ))
                } else {
                    let a = compile_matrix(&operands[0])?;
                    let b = compile_matrix(&operands[1])?;
                    record(
                        narrate,
                        &mut trace,
                        TechniqueTag::LuDecomposition,
                        "Solve Ax = b via LU",
                    );
                    a.solve_system(&b)
                        .map(|x| matrix_to_value(&x))
                        .map_err(|e| format!("{:?}", e))
                }
            }
            MatrixOp::Rank => {
                if operands.len() != 1 {
                    Err(format!("Rank requires 1 operand, got {}", operands.len()))
                } else {
                    let m = compile_matrix(&operands[0])?;
                    record(
                        narrate,
                        &mut trace,
                        TechniqueTag::RankComputation,
                        "Rank via pivot count in row echelon form",
                    );
                    m.rank()
                        .map(|rank| {
                            let value = Expression::Integer(rank as i64);
                            symbolic_entry(value, EngineId::Matrix, steps_from_trace(&trace))
                        })
                        .map_err(|e| format!("{:?}", e))
                }
            }
            MatrixOp::NullSpace => {
                if operands.len() != 1 {
                    Err(format!(
                        "NullSpace requires 1 operand, got {}",
                        operands.len()
                    ))
                } else {
                    let m = compile_matrix(&operands[0])?;
                    record(
                        narrate,
                        &mut trace,
                        TechniqueTag::NullSpaceComputation,
                        "Kernel basis via RREF back-substitution",
                    );
                    m.kernel()
                        .map(|basis| {
                            let mat_to_part = |mat: &CoreMatrix| -> DecompositionPart {
                                let rows = mat.rows();
                                let cols = mat.cols();
                                let elements: Vec<Expression> = (0..rows)
                                    .flat_map(|r| {
                                        (0..cols).map(move |c| {
                                            decompile(mat.get(r, c).expect("in-range cell"))
                                        })
                                    })
                                    .collect();
                                DecompositionPart::Matrix {
                                    elements,
                                    rows,
                                    cols,
                                }
                            };
                            let parts: Vec<(String, DecompositionPart)> = basis
                                .iter()
                                .enumerate()
                                .map(|(i, v)| (format!("basis_{}", i + 1), mat_to_part(v)))
                                .collect();
                            let cells: Vec<Expression> = basis
                                .iter()
                                .flat_map(|v| {
                                    (0..v.rows()).flat_map(move |r| {
                                        (0..v.cols())
                                            .map(move |c| decompile(v.get(r, c).expect("in-range")))
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
                                steps: steps_from_trace(&trace),
                                alternatives,
                                engine: EngineId::Matrix,
                            }
                        })
                        .map_err(|e| format!("{:?}", e))
                }
            }
            MatrixOp::Qr => {
                Err("QR decomposition engine not yet implemented in v0.9.0".to_string())
            }
            MatrixOp::ColumnSpace => {
                if operands.len() != 1 {
                    Err(format!(
                        "ColumnSpace requires 1 operand, got {}",
                        operands.len()
                    ))
                } else {
                    let m = compile_matrix(&operands[0])?;
                    record(
                        narrate,
                        &mut trace,
                        TechniqueTag::ColumnSpaceComputation,
                        "Column-space basis via pivot columns in RREF",
                    );
                    m.column_space()
                        .map(|basis| {
                            let mat_to_part = |mat: &CoreMatrix| -> DecompositionPart {
                                let rows = mat.rows();
                                let cols = mat.cols();
                                let elements: Vec<Expression> = (0..rows)
                                    .flat_map(|r| {
                                        (0..cols).map(move |c| {
                                            decompile(mat.get(r, c).expect("in-range cell"))
                                        })
                                    })
                                    .collect();
                                DecompositionPart::Matrix {
                                    elements,
                                    rows,
                                    cols,
                                }
                            };
                            let parts: Vec<(String, DecompositionPart)> = basis
                                .iter()
                                .enumerate()
                                .map(|(i, v)| (format!("basis_{}", i + 1), mat_to_part(v)))
                                .collect();
                            let cells: Vec<Expression> = basis
                                .iter()
                                .flat_map(|v| {
                                    (0..v.rows()).flat_map(move |r| {
                                        (0..v.cols())
                                            .map(move |c| decompile(v.get(r, c).expect("in-range")))
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
                                steps: steps_from_trace(&trace),
                                alternatives,
                                engine: EngineId::Matrix,
                            }
                        })
                        .map_err(|e| format!("{:?}", e))
                }
            }
            MatrixOp::RowEchelon => {
                if operands.len() != 1 {
                    Err(format!(
                        "RowEchelon requires 1 operand, got {}",
                        operands.len()
                    ))
                } else {
                    let m = compile_matrix(&operands[0])?;
                    record(
                        narrate,
                        &mut trace,
                        TechniqueTag::GaussJordanElimination,
                        "Reduced row echelon form via Gauss-Jordan elimination",
                    );
                    m.rref()
                        .map(|(mat, _pivots)| {
                            let mut entry = matrix_to_value(&mat);
                            entry.steps = steps_from_trace(&trace);
                            entry
                        })
                        .map_err(|e| format!("{:?}", e))
                }
            }
            MatrixOp::MinimalPolynomial => {
                Err("MinimalPolynomial engine not yet implemented".to_string())
            }
            MatrixOp::SymbolicEigenvectors => {
                Err("SymbolicEigenvectors engine not yet implemented".to_string())
            }
            MatrixOp::Cholesky => Err("Cholesky engine not yet implemented".to_string()),
            MatrixOp::Svd => Err("SVD engine not yet implemented".to_string()),
            MatrixOp::CharacteristicPolynomial => {
                Err("CharacteristicPolynomial engine not yet implemented".to_string())
            }
            MatrixOp::QuadraticFormClassify => {
                Err("QuadraticFormClassify engine not yet implemented".to_string())
            }
            MatrixOp::KroneckerProduct => {
                Err("KroneckerProduct engine not yet implemented".to_string())
            }
        }
    })();

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
