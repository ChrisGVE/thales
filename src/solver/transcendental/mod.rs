//! Transcendental equation solver for equations with trig, exp, and log functions.
//!
//! Operates on [`Arc<Expr>`] internals: compiles both sides of the
//! equation to canonical form, dispatches through the trig, log, and
//! exp submodules, and decompiles the isolated variable expression to
//! [`Expression`] only at the boundary (for resolution-path display).
//!
//! # Supported Equation Types
//!
//! ## Trigonometric Equations
//!
//! - `sin(x) = a` → `x = asin(a)` (requires |a| ≤ 1)
//! - `cos(x) = a` → `x = acos(a)` (requires |a| ≤ 1)
//! - `tan(x) = a` → `x = atan(a)`
//! - `c · sin(x) = b` → `x = asin(b/c)`
//! - `sin(c·x) = a` → `x = asin(a) / c`
//!
//! ## Logarithmic Equations
//!
//! - `ln(x) = a` → `x = exp(a)`
//! - `log10(x) = a` → `x = 10^a`
//! - `log(x, b) = a` → `x = b^a`
//! - `c · ln(x) = a` → `x = exp(a/c)`
//!
//! ## Exponential Equations
//!
//! - `exp(x) = a` → `x = ln(a)`
//! - `a^x = b` → `x = ln(b)/ln(a)`
//! - `exp(c·x) = a` → `x = ln(a)/c`
//! - `c · exp(x) = a` → `x = ln(a/c)`

mod detection;
mod exp;
mod log;
mod trig;

use crate::ast::{Equation, Variable};
use crate::numeric::compile::{compile, decompile};
use crate::numeric::trace::Trace;
use crate::numeric::SymbolId;

use super::helpers::{contains_symbol, evaluate_constants};
use super::types::{Solution, SolverError, SolverResult};
use super::Solver;

use self::detection::has_transcendental_function;

#[derive(Debug, Default)]
pub struct TranscendentalSolver;

impl TranscendentalSolver {
    pub fn new() -> Self {
        Self
    }
}

impl Solver for TranscendentalSolver {
    fn solve(&self, equation: &Equation, variable: &Variable) -> SolverResult<(Solution, Trace)> {
        let var_name = &variable.name;
        let var_id = SymbolId::intern(var_name);

        let lhs = compile(&equation.left);
        let rhs = compile(&equation.right);

        if !contains_symbol(&lhs, var_id) && !contains_symbol(&rhs, var_id) {
            return Err(SolverError::CannotSolve(format!(
                "Variable '{}' not found in equation",
                var_name
            )));
        }

        // Try families in order: trig → log → exp. Each receives a fresh
        // trace via a scratch buffer; a successful family returns its trace
        // as-is, misses discard theirs.
        let mut trace = Trace::new();
        if let Ok(sol) = trig::solve_trig_equation(&lhs, &rhs, var_id, variable, &mut trace) {
            return finish(sol, trace);
        }

        let mut trace = Trace::new();
        if let Ok(sol) = log::solve_log_equation(&lhs, &rhs, var_id, variable, &mut trace) {
            return finish(sol, trace);
        }

        let mut trace = Trace::new();
        if let Ok(sol) = exp::solve_exp_equation(&lhs, &rhs, var_id, variable, &mut trace) {
            return finish(sol, trace);
        }

        Err(SolverError::CannotSolve(
            "Transcendental equation pattern not recognized or too complex".to_string(),
        ))
    }

    fn can_solve(&self, equation: &Equation) -> bool {
        let lhs = compile(&equation.left);
        let rhs = compile(&equation.right);
        has_transcendental_function(&lhs) || has_transcendental_function(&rhs)
    }
}

fn finish(
    solution: std::sync::Arc<crate::numeric::Expr>,
    trace: Trace,
) -> SolverResult<(Solution, Trace)> {
    let expr = decompile(&solution);
    let evaluated = evaluate_constants(&expr);
    Ok((Solution::Unique(evaluated), trace))
}
