//! Higher-order ODE solving algorithms.

use crate::ast::{BinaryOp, Expression, Function, Variable};
use crate::ode::{solve_second_order_homogeneous, ODEError, SecondOrderODE, SecondOrderSolution};
use crate::resolution_path::{Operation, ResolutionPath, ResolutionPathBuilder};
use std::fmt;

use super::helpers::*;
use super::types::{CharRoot, HigherOrderODE, HigherOrderSolution};

// ---------------------------------------------------------------------------
// Public solver: higher-order homogeneous
// ---------------------------------------------------------------------------

/// Solve an n-th order constant-coefficient homogeneous ODE.
///
/// Uses companion-matrix eigenvalue finding (via companion matrix QR for
/// degrees ≤ 4, or direct analytic formulas for degrees 2 and 3).
///
/// # Errors
///
/// Returns [`ODEError::CharacteristicEquationError`] if the leading
/// coefficient is zero or if root-finding fails.
pub fn solve_higher_order_homogeneous(
    ode: &HigherOrderODE,
) -> Result<HigherOrderSolution, ODEError> {
    validate_ode(ode)?;

    let mut steps = Vec::new();
    steps.push(format_ode_string(ode));

    // Delegate order-2 to the existing second-order solver for consistency.
    if ode.order() == 2 {
        return solve_via_second_order(ode, steps);
    }

    let roots = find_characteristic_roots(&ode.coeffs)?;
    steps.push(format!("Characteristic roots: {}", format_roots(&roots)));

    let solution = build_general_solution(&roots, &ode.independent, &mut steps);

    Ok(HigherOrderSolution {
        general_solution: solution,
        roots,
        steps,
        method: "Characteristic equation".to_string(),
    })
}
pub(super) fn validate_ode(ode: &HigherOrderODE) -> Result<(), ODEError> {
    if ode.coeffs.len() < 2 {
        return Err(ODEError::CharacteristicEquationError(
            "ODE must have order ≥ 1 (coeffs.len() ≥ 2)".to_string(),
        ));
    }
    if ode.coeffs[0].abs() < 1e-15 {
        return Err(ODEError::CharacteristicEquationError(
            "Leading coefficient must be non-zero".to_string(),
        ));
    }
    Ok(())
}

pub(super) fn format_ode_string(ode: &HigherOrderODE) -> String {
    let n = ode.order();
    let terms: Vec<String> = ode
        .coeffs
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let order = n - i;
            match order {
                0 => format!("{c}·{}", ode.dependent),
                1 => format!("{c}·{}'", ode.dependent),
                _ => format!("{c}·{}^({})", ode.dependent, order),
            }
        })
        .collect();
    format!("Given ODE: {} = 0", terms.join(" + "))
}

/// Solve order-2 by delegating to the existing `SecondOrderODE` solver.
pub(super) fn solve_via_second_order(
    ode: &HigherOrderODE,
    mut steps: Vec<String>,
) -> Result<HigherOrderSolution, ODEError> {
    let snd = SecondOrderODE::homogeneous(
        &ode.dependent,
        &ode.independent,
        ode.coeffs[0],
        ode.coeffs[1],
        ode.coeffs[2],
    );
    let sol = solve_second_order_homogeneous(&snd)?;
    steps.extend(sol.steps.clone());

    // Re-express roots as CharRoot for the unified interface.
    use crate::ode::RootType;
    let roots = match sol.roots.root_type {
        RootType::TwoDistinctReal => vec![
            CharRoot {
                real: sol.roots.r1,
                imag: 0.0,
                multiplicity: 1,
            },
            CharRoot {
                real: sol.roots.r2,
                imag: 0.0,
                multiplicity: 1,
            },
        ],
        RootType::RepeatedReal => vec![CharRoot {
            real: sol.roots.r1,
            imag: 0.0,
            multiplicity: 2,
        }],
        RootType::ComplexConjugate => vec![
            CharRoot {
                real: sol.roots.r1,
                imag: sol.roots.r2,
                multiplicity: 1,
            },
            CharRoot {
                real: sol.roots.r1,
                imag: -sol.roots.r2,
                multiplicity: 1,
            },
        ],
    };

    Ok(HigherOrderSolution {
        general_solution: sol.general_solution,
        roots,
        steps,
        method: "Characteristic equation (2nd order)".to_string(),
    })
}
