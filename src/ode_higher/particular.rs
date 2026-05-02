//! Particular solution methods (undetermined coefficients).

use super::helpers::*;
use crate::ast::{BinaryOp, Expression};
use crate::numeric::compile::{compile, decompile};
use crate::ode::{solve_second_order_homogeneous, ODEError, SecondOrderODE, SecondOrderSolution};

/// Forcing function shape supported by the undetermined-coefficients method.
#[derive(Debug, Clone)]
pub enum ForcingKind {
    /// Polynomial of given degree: `p_n(x)`.
    Polynomial(usize),
    /// Pure exponential: `e^(ax)`.
    Exponential(f64),
    /// Sinusoidal: `cos(ωx)` or `sin(ωx)`.
    Sinusoidal(f64),
    /// Product of polynomial (degree n) and exponential: `p_n(x)·e^(ax)`.
    PolynomialTimesExp {
        /// Degree of the polynomial factor.
        degree: usize,
        /// Exponent coefficient in the exponential factor.
        alpha: f64,
    },
}

/// Solve a 2nd-order non-homogeneous ODE using undetermined coefficients.
///
/// The ODE has the form `a·y'' + b·y' + c·y = g(x)` where `g(x)` is
/// characterised by `forcing_kind`.
///
/// Returns [`SecondOrderSolution`] with both homogeneous and particular parts.
///
/// # Errors
///
/// Returns [`ODEError::ResonanceDetected`] when the trial particular solution
/// overlaps with the homogeneous solution (resonance case), and
/// [`ODEError::CannotSolve`] for unsupported forcing shapes.
pub fn solve_undetermined_coefficients(
    ode: &SecondOrderODE,
    forcing_kind: ForcingKind,
) -> Result<SecondOrderSolution, ODEError> {
    // Step 1: solve homogeneous part.
    let hom = solve_second_order_homogeneous(ode)?;
    let mut steps = hom.steps.clone();
    steps.push(format!("Non-homogeneous forcing kind: {:?}", forcing_kind));

    // Step 2: build and solve for the particular solution coefficients.
    let particular = find_particular_solution(ode, &forcing_kind, &hom, &mut steps)?;

    // Step 3: general = homogeneous + particular.
    let hom_expr = decompile(&hom.general_solution);
    let general = Expression::Binary(
        BinaryOp::Add,
        Box::new(hom_expr),
        Box::new(particular.clone()),
    );

    steps.push(format!(
        "General solution: y_h + y_p = (homogeneous) + {}",
        particular
    ));

    Ok(SecondOrderSolution {
        homogeneous_solution: hom.general_solution,
        particular_solution: Some(compile(&particular)),
        general_solution: compile(&general),
        method: "Undetermined coefficients".to_string(),
        roots: hom.roots,
        steps,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
