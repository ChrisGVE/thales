//! Types for higher-order ODE solving.

use std::sync::Arc;

use crate::ast::Expression;
use crate::numeric::Expr;

/// A single root of the characteristic polynomial, with its multiplicity.
#[derive(Debug, Clone)]
pub struct CharRoot {
    /// Real part of the root.
    pub real: f64,
    /// Imaginary part (0.0 for purely real roots).
    pub imag: f64,
    /// Algebraic multiplicity (≥ 1).
    pub multiplicity: usize,
}

impl CharRoot {
    pub(crate) fn is_real(&self) -> bool {
        self.imag.abs() < 1e-10
    }
}

// ---------------------------------------------------------------------------
// HigherOrderODE
// ---------------------------------------------------------------------------

/// An n-th order constant-coefficient linear homogeneous ODE.
///
/// Represents: `coeffs[0]*y^(n) + coeffs[1]*y^(n-1) + … + coeffs[n]*y = 0`
///
/// `coeffs[0]` must be non-zero (leading coefficient).
#[derive(Debug, Clone)]
pub struct HigherOrderODE {
    /// Dependent variable name (e.g., `"y"`).
    pub dependent: String,
    /// Independent variable name (e.g., `"x"`).
    pub independent: String,
    /// Coefficients from highest-order term to zero-th order term.
    pub coeffs: Vec<f64>,
}

impl HigherOrderODE {
    /// Create a new higher-order ODE.
    ///
    /// `coeffs` must have length ≥ 2; `coeffs[0]` is the leading coefficient.
    pub fn new(dependent: &str, independent: &str, coeffs: Vec<f64>) -> Self {
        Self {
            dependent: dependent.to_string(),
            independent: independent.to_string(),
            coeffs,
        }
    }

    /// Order of the ODE (`coeffs.len() - 1`).
    pub fn order(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }
}

// ---------------------------------------------------------------------------
// Solution type
// ---------------------------------------------------------------------------

/// Solution of a higher-order homogeneous ODE.
#[derive(Debug, Clone)]
pub struct HigherOrderSolution {
    /// The general solution expression (contains C1, C2, … constants), in
    /// canonical [`Arc<Expr>`] form.
    pub general_solution: Arc<Expr>,
    /// All characteristic roots (with multiplicity).
    pub roots: Vec<CharRoot>,
    /// Human-readable solution steps.
    pub steps: Vec<String>,
    /// Description of the method used.
    pub method: String,
}
