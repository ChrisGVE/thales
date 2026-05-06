//! String-to-type parsers used during JSON → Command conversion.

use super::super::super::command::{IdentityId, MatrixOp, OptSense, Side, SpecialKind};
use super::super::super::domain::Domain;
use super::super::super::request::SolveMode;

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
        "LnGamma" => Ok(SpecialKind::LnGamma),
        "Digamma" => Ok(SpecialKind::Digamma),
        "BesselJ" => Ok(SpecialKind::BesselJ),
        "BesselY" => Ok(SpecialKind::BesselY),
        "BesselI" => Ok(SpecialKind::BesselI),
        "BesselK" => Ok(SpecialKind::BesselK),
        "AiryAi" => Ok(SpecialKind::AiryAi),
        "AiryBi" => Ok(SpecialKind::AiryBi),
        "Zeta" => Ok(SpecialKind::Zeta),
        "Si" => Ok(SpecialKind::Si),
        "Ci" => Ok(SpecialKind::Ci),
        "Ei" => Ok(SpecialKind::Ei),
        "Heaviside" => Ok(SpecialKind::Heaviside),
        "DiracDelta" => Ok(SpecialKind::DiracDelta),
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
        "ColumnSpace" => Ok(MatrixOp::ColumnSpace),
        "RowEchelon" => Ok(MatrixOp::RowEchelon),
        "MinimalPolynomial" => Ok(MatrixOp::MinimalPolynomial),
        "SymbolicEigenvectors" => Ok(MatrixOp::SymbolicEigenvectors),
        "Cholesky" => Ok(MatrixOp::Cholesky),
        "Svd" => Ok(MatrixOp::Svd),
        "CharacteristicPolynomial" => Ok(MatrixOp::CharacteristicPolynomial),
        "QuadraticFormClassify" => Ok(MatrixOp::QuadraticFormClassify),
        "KroneckerProduct" => Ok(MatrixOp::KroneckerProduct),
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
