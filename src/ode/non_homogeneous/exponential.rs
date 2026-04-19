//! Particular solution for exponential forcing `A·e^(k·x)`.
//!
//! Trial: `B·e^(k·x)`, multiplied by `x^m` when `k` is a characteristic
//! root with multiplicity `m` (simple or repeated).

use std::collections::HashMap;

use crate::ast::{BinaryOp, Expression, Function, Variable};
use crate::numeric::evaluation::evaluate;
use crate::numeric::SymbolId;

use super::polynomial::build_x_power;
use super::{ODEError, SecondOrderODE};
use crate::ode::second_order::solve_characteristic_equation;

// ---------------------------------------------------------------------------
// Particular solution: exponential forcing
// ---------------------------------------------------------------------------

/// Find `y_p` when `f(x) = A·e^(k·x)`.
///
/// Trial: `y_p = B·e^(k·x)`.  If `k` is a simple characteristic root,
/// multiply by `x`; if double root, multiply by `x²`.
pub(super) fn particular_exponential(
    ode: &SecondOrderODE,
    k: f64,
    steps: &mut Vec<String>,
) -> Result<Expression, ODEError> {
    let x_var = &ode.independent;
    let multiplier = resonance_multiplier_exp(ode, k)?;

    steps.push(format!(
        "Trial form: y_p = B·x^{}·e^({}·{})",
        multiplier, k, x_var
    ));

    // Evaluate forcing at one point to get amplitude A
    let amp = {
        let forcing_arc = ode.forcing_arc();
        let x_id = SymbolId::intern(x_var);
        let mut env = HashMap::new();
        env.insert(x_id, 1.0);
        evaluate(&forcing_arc, &env).unwrap_or(1.0) / (k * 1.0_f64).exp()
    };

    // Compute denominator: substitute y_p = B·x^m·e^(kx) into ODE, divide by e^(kx)
    // For m=0: a·k² + b·k + c
    // For m=1: 2a·k + b
    // For m=2: 2a
    let denom = match multiplier {
        0 => ode.a * k * k + ode.b * k + ode.c,
        1 => 2.0 * ode.a * k + ode.b,
        _ => 2.0 * ode.a,
    };

    if denom.abs() < 1e-12 {
        return Err(ODEError::ResonanceDetected(
            "Triple resonance in exponential case — not supported".to_string(),
        ));
    }

    let b_coeff = amp / denom;
    steps.push(format!("Coefficient B = {}", b_coeff));

    Ok(build_exp_particular(b_coeff, k, multiplier, x_var))
}

/// Determine resonance multiplier for exponential forcing `e^(k·x)`.
fn resonance_multiplier_exp(ode: &SecondOrderODE, k: f64) -> Result<u32, ODEError> {
    const EPS: f64 = 1e-10;

    let roots = solve_characteristic_equation(ode.a, ode.b, ode.c)
        .map_err(|e| ODEError::CannotSolve(e.to_string()))?;

    let matches_r1 = (roots.r1 - k).abs() < EPS;
    let matches_r2 = (roots.r2 - k).abs() < EPS;

    match (matches_r1, matches_r2) {
        (false, false) => Ok(0),
        (true, false) | (false, true) => Ok(1),
        (true, true) => Ok(2),
    }
}

/// Build `B · x^m · e^(k·x)` as an `Expression`.
fn build_exp_particular(b: f64, k: f64, multiplier: u32, x_var: &str) -> Expression {
    let exp_arg = Expression::Binary(
        BinaryOp::Mul,
        Box::new(Expression::Float(k)),
        Box::new(Expression::Variable(Variable::new(x_var))),
    );
    let exp_term = Expression::Function(Function::Exp, vec![exp_arg]);

    let base = if (b - 1.0).abs() < 1e-15 {
        exp_term
    } else {
        Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Float(b)),
            Box::new(exp_term),
        )
    };

    if multiplier == 0 {
        return base;
    }

    let x_pow = build_x_power(x_var, multiplier as i64);
    Expression::Binary(BinaryOp::Mul, Box::new(x_pow), Box::new(base))
}
