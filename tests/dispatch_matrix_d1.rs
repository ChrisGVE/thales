//! D1 linear algebra integration tests.
//!
//! Each test sends a `Command::Matrix { op, operands }` through `execute()` and
//! verifies the response shape. All tests use simple integer matrices for
//! reliable, numerically stable results.
//!
//! Naming: `fast_d1_*` — eligible for the nextest `fast` profile.

use thales::api::command::{Command, MatrixExpr as ApiMatrixExpr, MatrixOp};
use thales::api::execute;
use thales::api::request::Request;
use thales::api::response::{EngineId, ResultValue, StructuredResult};
use thales::ast::Expression;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn int(n: i64) -> Expression {
    Expression::Integer(n)
}

fn m22(a: i64, b: i64, c: i64, d: i64) -> ApiMatrixExpr {
    ApiMatrixExpr::Matrix(vec![vec![int(a), int(b)], vec![int(c), int(d)]])
}

fn request(cmd: Command) -> Request {
    Request {
        command: cmd,
        ..Default::default()
    }
}

fn assert_matrix_engine(resp: &thales::api::response::Response) {
    assert!(
        !resp.results.is_empty(),
        "expected at least one result entry"
    );
    assert_eq!(
        resp.results[0].1.engine,
        EngineId::Matrix,
        "expected Matrix engine"
    );
}

fn assert_no_engine_error(resp: &thales::api::response::Response) {
    assert!(
        !resp
            .diagnostics
            .iter()
            .any(|d| matches!(&d.code, thales::api::diagnostic::DiagnosticCode::Other(s) if *s == "engine-error")),
        "unexpected engine-error diagnostic: {:?}",
        resp.diagnostics
    );
}

fn has_decomposition_part(resp: &thales::api::response::Response, name: &str) -> bool {
    match &resp.results[0].1.structured {
        Some(StructuredResult::Decomposition { parts }) => parts.iter().any(|(n, _)| n == name),
        _ => false,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn fast_d1_qr_decomposition() {
    // QR of [[1,2],[3,4]]: expect Decomposition with Q and R parts.
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::Qr,
        operands: vec![m22(1, 2, 3, 4)],
    }))
    .unwrap();

    assert_matrix_engine(&resp);
    assert_no_engine_error(&resp);
    assert!(
        matches!(
            &resp.results[0].1.structured,
            Some(StructuredResult::Decomposition { .. })
        ),
        "expected Decomposition structured result, got {:?}",
        resp.results[0].1.structured
    );
    assert!(has_decomposition_part(&resp, "Q"), "expected Q part in QR");
    assert!(has_decomposition_part(&resp, "R"), "expected R part in QR");
}

#[test]
fn fast_d1_cholesky() {
    // Cholesky of SPD matrix [[4,2],[2,3]]: expect Decomposition with L part.
    let spd = ApiMatrixExpr::Matrix(vec![vec![int(4), int(2)], vec![int(2), int(3)]]);
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::Cholesky,
        operands: vec![spd],
    }))
    .unwrap();

    assert_matrix_engine(&resp);
    assert_no_engine_error(&resp);
    assert!(
        matches!(
            &resp.results[0].1.structured,
            Some(StructuredResult::Decomposition { .. })
        ),
        "expected Decomposition for Cholesky, got {:?}",
        resp.results[0].1.structured
    );
    assert!(
        has_decomposition_part(&resp, "L"),
        "expected L part in Cholesky decomposition"
    );
}

#[test]
fn fast_d1_svd() {
    // SVD of [[1,0],[0,2]]: expect Decomposition with U, Sigma, V_transpose parts.
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::Svd,
        operands: vec![m22(1, 0, 0, 2)],
    }))
    .unwrap();

    assert_matrix_engine(&resp);
    assert_no_engine_error(&resp);
    assert!(
        matches!(
            &resp.results[0].1.structured,
            Some(StructuredResult::Decomposition { .. })
        ),
        "expected Decomposition for SVD, got {:?}",
        resp.results[0].1.structured
    );
    assert!(has_decomposition_part(&resp, "U"), "expected U part in SVD");
    assert!(
        has_decomposition_part(&resp, "Sigma"),
        "expected Sigma part in SVD"
    );
    assert!(
        has_decomposition_part(&resp, "V_transpose"),
        "expected V_transpose part in SVD"
    );
}

#[test]
fn fast_d1_characteristic_polynomial() {
    // char poly of [[2,0],[0,3]] = (λ-2)(λ-3) = λ²-5λ+6.
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::CharacteristicPolynomial,
        operands: vec![m22(2, 0, 0, 3)],
    }))
    .unwrap();

    assert_matrix_engine(&resp);
    assert_no_engine_error(&resp);
    // Result must be a symbolic polynomial expression.
    assert!(
        matches!(resp.results[0].1.value, ResultValue::Symbolic(_)),
        "expected Symbolic result for CharacteristicPolynomial"
    );
}

#[test]
fn fast_d1_minimal_polynomial() {
    // Minimal polynomial of the identity matrix [[1,0],[0,1]] is (λ-1).
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::MinimalPolynomial,
        operands: vec![m22(1, 0, 0, 1)],
    }))
    .unwrap();

    assert_matrix_engine(&resp);
    assert_no_engine_error(&resp);
    assert!(
        matches!(resp.results[0].1.value, ResultValue::Symbolic(_)),
        "expected Symbolic result for MinimalPolynomial"
    );
    // The minimal polynomial of I₂ is degree 1 (λ-1); verify the result
    // contains 'lambda' in its string representation.
    if let ResultValue::Symbolic(ref e) = resp.results[0].1.value {
        let s = format!("{:?}", e);
        // lambda appears as a Variable in the polynomial expression.
        assert!(
            s.contains("lambda") || s.contains("Lambda"),
            "expected lambda variable in minimal polynomial, got {:?}",
            e
        );
    }
}

#[test]
fn fast_d1_symbolic_eigenvectors() {
    // [[2,1],[1,2]] has eigenvalues 1 and 3 with clear symbolic eigenvectors.
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::SymbolicEigenvectors,
        operands: vec![m22(2, 1, 1, 2)],
    }))
    .unwrap();

    assert_matrix_engine(&resp);
    assert_no_engine_error(&resp);
    assert!(
        matches!(
            &resp.results[0].1.structured,
            Some(StructuredResult::Decomposition { .. })
        ),
        "expected Decomposition for SymbolicEigenvectors, got {:?}",
        resp.results[0].1.structured
    );
    // Must have at least one eigenpair in the decomposition.
    if let Some(StructuredResult::Decomposition { parts }) = &resp.results[0].1.structured {
        assert!(
            !parts.is_empty(),
            "expected at least one part in symbolic eigenvector decomposition"
        );
    }
}

#[test]
fn fast_d1_quadratic_form_classify() {
    // [[1,0],[0,1]] (identity) represents x²+y²: positive definite.
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::QuadraticFormClassify,
        operands: vec![m22(1, 0, 0, 1)],
    }))
    .unwrap();

    assert_matrix_engine(&resp);
    assert_no_engine_error(&resp);
    // Result is a Variable containing the definiteness label.
    if let ResultValue::Symbolic(Expression::Variable(ref v)) = resp.results[0].1.value {
        assert_eq!(
            v.name, "PositiveDefinite",
            "identity matrix should be PositiveDefinite, got {}",
            v.name
        );
    } else {
        panic!(
            "expected Variable result for QuadraticFormClassify, got {:?}",
            resp.results[0].1.value
        );
    }
}

#[test]
fn fast_d1_kronecker_product() {
    // [[1,0],[0,1]] ⊗ [[2,3],[4,5]] = 4×4 block diagonal matrix.
    // Resulting matrix is 4×4 = 16 cells; primary + 15 alternatives.
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::KroneckerProduct,
        operands: vec![m22(1, 0, 0, 1), m22(2, 3, 4, 5)],
    }))
    .unwrap();

    assert_matrix_engine(&resp);
    assert_no_engine_error(&resp);
    // 4×4 result: primary + 15 alternatives = 16 cells total.
    let total = 1 + resp.results[0].1.alternatives.len();
    assert_eq!(
        total, 16,
        "2×2 ⊗ 2×2 Kronecker product must produce a 4×4 matrix (16 cells), got {}",
        total
    );
}

#[test]
fn fast_d1_null_space_full_rank() {
    // The 3×3 identity has full rank 3; its null space is trivial (empty basis).
    let identity3 = ApiMatrixExpr::Matrix(vec![
        vec![int(1), int(0), int(0)],
        vec![int(0), int(1), int(0)],
        vec![int(0), int(0), int(1)],
    ]);
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::NullSpace,
        operands: vec![identity3],
    }))
    .unwrap();

    assert_matrix_engine(&resp);
    assert_no_engine_error(&resp);
    // Full-rank matrix: null space is empty, so Decomposition has no parts.
    match &resp.results[0].1.structured {
        Some(StructuredResult::Decomposition { parts }) => {
            assert!(
                parts.is_empty(),
                "full-rank matrix null space should have no basis vectors, got {} parts",
                parts.len()
            );
        }
        other => panic!("expected Decomposition for NullSpace, got {:?}", other),
    }
}

#[test]
fn fast_d1_column_space() {
    // [[1,2],[3,4]] has rank 2; column space has 2 basis vectors.
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::ColumnSpace,
        operands: vec![m22(1, 2, 3, 4)],
    }))
    .unwrap();

    assert_matrix_engine(&resp);
    assert_no_engine_error(&resp);
    match &resp.results[0].1.structured {
        Some(StructuredResult::Decomposition { parts }) => {
            assert_eq!(
                parts.len(),
                2,
                "rank-2 matrix must have 2 column-space basis vectors, got {}",
                parts.len()
            );
        }
        other => panic!("expected Decomposition for ColumnSpace, got {:?}", other),
    }
}

#[test]
fn fast_d1_row_echelon() {
    // [[2,4],[1,3]] → RREF = [[1,0],[0,1]] (identity after row ops).
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::RowEchelon,
        operands: vec![m22(2, 4, 1, 3)],
    }))
    .unwrap();

    assert_matrix_engine(&resp);
    assert_no_engine_error(&resp);
    assert_eq!(
        resp.results[0].1.shape,
        thales::api::response::ResultShape::Matrix,
        "RowEchelon must return a Matrix shape"
    );
    // 2×2 result: primary + 3 alternatives = 4 cells.
    let total = 1 + resp.results[0].1.alternatives.len();
    assert_eq!(
        total, 4,
        "2×2 RowEchelon result must have 4 cells, got {}",
        total
    );
}

#[test]
fn fast_d1_rank_rectangular() {
    // 3×2 matrix [[1,0],[0,1],[0,0]] has rank 2 (two pivot columns).
    let rect = ApiMatrixExpr::Matrix(vec![
        vec![int(1), int(0)],
        vec![int(0), int(1)],
        vec![int(0), int(0)],
    ]);
    let resp = execute(request(Command::Matrix {
        op: MatrixOp::Rank,
        operands: vec![rect],
    }))
    .unwrap();

    assert_matrix_engine(&resp);
    assert_no_engine_error(&resp);
    if let ResultValue::Symbolic(ref e) = resp.results[0].1.value {
        assert_eq!(
            *e,
            int(2),
            "rank of [[1,0],[0,1],[0,0]] expected 2, got {:?}",
            e
        );
    } else {
        panic!(
            "expected Symbolic result for Rank, got {:?}",
            resp.results[0].1.value
        );
    }
}
