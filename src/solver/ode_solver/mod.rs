//! ODE solver adapter implementing the [`Solver`] trait for differential equations.
//!
//! This module bridges the ODE-specific solving functions in [`crate::ode`] with
//! the generic [`Solver`] trait used throughout this crate.  It also exposes
//! convenience functions for callers that already hold a typed ODE struct and do
//! not need to go through the [`SmartSolver`](crate::solver::SmartSolver) dispatch
//! path.
//!
//! # Parsing-based ODE solving
//!
//! The [`solve_ode_from_text`] function accepts a string containing an ODE in
//! derivative notation (Leibniz, prime, or functional) and automatically extracts
//! and solves it:
//!
//! ```rust
//! use thales::solver::ode_solver::solve_ode_from_text;
//! use thales::solver::Solution;
//!
//! let (solution, _path) = solve_ode_from_text("dy/dx = y").unwrap();
//! assert!(matches!(solution, Solution::Unique(_)));
//! ```
//!
//! # Direct ODE solving
//!
//! Use [`solve_ode_first_order`] or [`solve_ode_second_order`] when you already
//! have a [`FirstOrderODE`] or [`SecondOrderODE`] struct.
//!
//! # Trait-based solving
//!
//! The [`OdeSolver`] struct implements the [`Solver`] trait but always returns
//! `false` from `can_solve` because the thales AST does not carry derivative
//! nodes. For text-based ODE detection, use [`solve_ode_from_text`] instead.
//!
//! # Examples
//!
//! ```rust
//! use thales::ast::{BinaryOp, Expression, Variable};
//! use thales::ode::FirstOrderODE;
//! use thales::solver::ode_solver::solve_ode_first_order;
//! use thales::solver::Solution;
//!
//! // dy/dx = y  →  y = C·eˣ
//! let ode = FirstOrderODE::new(
//!     "y", "x",
//!     Expression::Variable(Variable::new("y")),
//! );
//! let (solution, path) = solve_ode_first_order(&ode).unwrap();
//! assert!(matches!(solution, Solution::Unique(_)));
//! assert!(!path.steps().is_empty());
//! ```

pub(crate) mod trace;

use crate::ast::{BinaryOp, Equation, Variable};
use crate::numeric::trace::Trace;
use crate::ode::{
    particular_solution_undetermined, solve_linear, solve_second_order_homogeneous,
    solve_separable, FirstOrderODE, SecondOrderODE,
};
use crate::solver::ode_classifier::{classify_first_order, classify_second_order, ODEType};
use crate::solver::types::{Solution, SolverError, SolverResult};
use crate::solver::Solver;
use trace::build_ode_trace;

// ---------------------------------------------------------------------------
// OdeSolver struct
// ---------------------------------------------------------------------------

/// Solver adapter for ordinary differential equations.
///
/// This struct implements the [`Solver`] trait but cannot accept plain
/// [`Equation`] values because the AST has no derivative node.  It exists
/// primarily as a marker for the solver hierarchy; actual ODE solving is done
/// through [`solve_ode_first_order`] and [`solve_ode_second_order`].
#[derive(Debug, Default)]
pub struct OdeSolver;

impl OdeSolver {
    /// Create a new [`OdeSolver`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Solver for OdeSolver {
    /// Always returns `false` — ODEs cannot be detected from a plain
    /// [`Equation`] without a derivative AST node.
    fn can_solve(&self, _equation: &Equation) -> bool {
        false
    }

    /// Always returns [`SolverError::UnsupportedEquationType`].
    ///
    /// Use [`solve_ode_first_order`] or [`solve_ode_second_order`] instead.
    fn solve(&self, _equation: &Equation, _variable: &Variable) -> SolverResult<(Solution, Trace)> {
        Err(SolverError::UnsupportedEquationType)
    }
}

// ---------------------------------------------------------------------------
// Public convenience functions
// ---------------------------------------------------------------------------

/// Solve a first-order ODE directly, returning a [`Solution`] and a
/// [`ResolutionPath`] with the step-by-step explanation.
///
/// The ODE type is classified first (separable vs. linear) and the matching
/// solver is applied.  If the ODE does not fit either supported type the
/// function returns [`SolverError::CannotSolve`].
///
/// # Errors
///
/// Returns [`SolverError::CannotSolve`] when the ODE type is not yet
/// supported or when the underlying solver fails.
///
/// # Examples
///
/// ```rust
/// use thales::ast::{Expression, Variable};
/// use thales::ode::FirstOrderODE;
/// use thales::solver::ode_solver::solve_ode_first_order;
/// use thales::solver::Solution;
///
/// // dy/dx = y  (separable)
/// let ode = FirstOrderODE::new("y", "x", Expression::Variable(Variable::new("y")));
/// let (solution, _path) = solve_ode_first_order(&ode).unwrap();
/// assert!(matches!(solution, Solution::Unique(_)));
/// ```
pub fn solve_ode_first_order(ode: &FirstOrderODE) -> SolverResult<(Solution, Trace)> {
    let classification = classify_first_order(ode);

    // For ODEs classified as Separable but also linear, try separable first and
    // fall back to the integrating-factor method when integration fails.
    let ode_result = match classification.ode_type {
        ODEType::Separable => solve_separable(ode)
            .or_else(|_| solve_linear(ode))
            .map_err(|e| SolverError::CannotSolve(e.to_string())),
        ODEType::Linear => solve_linear(ode).map_err(|e| SolverError::CannotSolve(e.to_string())),
        _ => Err(SolverError::CannotSolve(
            "ODE type not yet supported".to_string(),
        )),
    }?;

    let ode_type_str = match classification.ode_type {
        ODEType::Separable => "separable",
        ODEType::Linear => "linear",
        ODEType::Bernoulli => "Bernoulli",
        ODEType::Unknown => "unknown",
        _ => "other",
    };
    let classify = Some(("first".to_string(), ode_type_str.to_string()));
    let path = build_ode_trace(classify, &ode_result.steps, &ode_result.general_solution);
    Ok((
        Solution::unique_from_expr(&ode_result.general_solution),
        path,
    ))
}

/// Solve a second-order linear ODE with constant coefficients directly.
///
/// Uses the characteristic equation method for the homogeneous part.  For
/// non-homogeneous ODEs with polynomial, exponential, or trigonometric forcing
/// the method of undetermined coefficients is applied to find a particular
/// solution; the general solution is then `y_h + y_p`.
///
/// # Errors
///
/// Returns [`SolverError::CannotSolve`] when:
/// - The characteristic equation solver fails.
/// - The forcing function is not a supported type for undetermined coefficients.
///
/// # Examples
///
/// ```rust
/// use thales::ode::SecondOrderODE;
/// use thales::solver::ode_solver::solve_ode_second_order;
/// use thales::solver::Solution;
///
/// // y'' - y = 0  →  y = C₁·eˣ + C₂·e^(−x)
/// let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 0.0, -1.0);
/// let (solution, _path) = solve_ode_second_order(&ode).unwrap();
/// assert!(matches!(solution, Solution::Unique(_)));
///
/// // y'' + y = 1  →  y = C₁·cos(x) + C₂·sin(x) + 1
/// use thales::ast::Expression;
/// let ode2 = SecondOrderODE::new("y", "x", 1.0, 0.0, 1.0, Expression::Integer(1));
/// let (solution2, _path2) = solve_ode_second_order(&ode2).unwrap();
/// assert!(matches!(solution2, Solution::Unique(_)));
/// ```
pub fn solve_ode_second_order(ode: &SecondOrderODE) -> SolverResult<(Solution, Trace)> {
    let hom_result =
        solve_second_order_homogeneous(ode).map_err(|e| SolverError::CannotSolve(e.to_string()))?;

    let cls = classify_second_order(ode);
    let ode_type_str = match cls.ode_type {
        ODEType::ConstantCoefficient => "homogeneous constant-coefficient",
        ODEType::NonHomogeneousConstantCoefficient => "non-homogeneous constant-coefficient",
        _ => "other",
    };
    let classify = Some(("second".to_string(), ode_type_str.to_string()));

    if ode.is_homogeneous() {
        let path = build_ode_trace(classify, &hom_result.steps, &hom_result.general_solution);
        return Ok((
            Solution::unique_from_expr(&hom_result.general_solution),
            path,
        ));
    }

    // Non-homogeneous: find particular solution then combine.
    let (yp, yp_steps) = particular_solution_undetermined(ode)
        .map_err(|e| SolverError::CannotSolve(e.to_string()))?;

    // TODO(arc-migration): particular_solution_undetermined returns Expression;
    // combine at Arc<Expr> level once src/ode/ migrates. Until then, compile
    // the Expression::Binary sum to Arc<Expr> for the trace and keep the
    // Expression form for the Solution output boundary.
    let hom_expr = crate::numeric::compile::decompile(&hom_result.general_solution);
    let general = crate::ast::Expression::Binary(BinaryOp::Add, Box::new(hom_expr), Box::new(yp));
    let general_arc = crate::numeric::compile::compile(&general);

    let mut all_steps = hom_result.steps;
    all_steps.extend(yp_steps);
    all_steps.push(format!("General solution: y = y_h + y_p"));

    let path = build_ode_trace(classify, &all_steps, &general_arc);
    Ok((Solution::Unique(general), path))
}

/// Parse an ODE from text and solve it, returning a [`Solution`] and
/// [`ResolutionPath`].
///
/// Accepts any derivative notation supported by mathlex: Leibniz (`dy/dx`),
/// prime (`y'`), or functional (`diff(y, x)`). Both first and second-order
/// constant-coefficient ODEs are supported.
///
/// # Errors
///
/// Returns [`SolverError::CannotSolve`] when:
/// - The input cannot be parsed as an equation.
/// - The equation does not contain recognizable ODE derivative terms.
/// - The extracted ODE type is not yet supported by the solver.
///
/// # Examples
///
/// ```rust
/// use thales::solver::ode_solver::solve_ode_from_text;
/// use thales::solver::Solution;
///
/// // First-order separable: dy/dx = y
/// let (solution, path) = solve_ode_from_text("dy/dx = y").unwrap();
/// assert!(matches!(solution, Solution::Unique(_)));
/// assert!(!path.steps().is_empty());
///
/// // Second-order homogeneous: y'' - y = 0
/// let (solution, _) = solve_ode_from_text("d2y/dx2 - y = 0").unwrap();
/// assert!(matches!(solution, Solution::Unique(_)));
/// ```
pub fn solve_ode_from_text(input: &str) -> SolverResult<(Solution, Trace)> {
    use crate::mathlex_bridge::{try_extract_ode, ExtractedODE};

    let ml_expr = mathlex::parse(input)
        .map_err(|e| SolverError::CannotSolve(format!("failed to parse ODE: {}", e)))?;

    let extracted = try_extract_ode(&ml_expr).ok_or_else(|| {
        SolverError::CannotSolve(
            "equation does not contain recognizable ODE derivative terms".to_string(),
        )
    })?;

    match extracted {
        ExtractedODE::First(ode) => solve_ode_first_order(&ode),
        ExtractedODE::Second(ode) => solve_ode_second_order(&ode),
    }
}

/// Parse a LaTeX ODE and solve it, returning a [`Solution`] and
/// [`ResolutionPath`].
///
/// Accepts LaTeX derivative notation such as `\frac{d}{dx}(y)` and
/// `\frac{d^2}{dx^2}(y)`. Both first and second-order constant-coefficient
/// ODEs are supported.
///
/// # Errors
///
/// Returns [`SolverError::CannotSolve`] when:
/// - The input cannot be parsed as a LaTeX equation.
/// - The equation does not contain recognizable ODE derivative terms.
/// - The extracted ODE type is not yet supported by the solver.
///
/// # Examples
///
/// ```rust
/// use thales::solver::ode_solver::solve_ode_from_latex;
/// use thales::solver::Solution;
///
/// let (solution, path) = solve_ode_from_latex(r#"\frac{d}{dx}(y) = y"#).unwrap();
/// assert!(matches!(solution, Solution::Unique(_)));
/// ```
pub fn solve_ode_from_latex(input: &str) -> SolverResult<(Solution, Trace)> {
    use crate::mathlex_bridge::{try_extract_ode, ExtractedODE};

    let ml_expr = mathlex::parse_latex(input)
        .map_err(|e| SolverError::CannotSolve(format!("failed to parse LaTeX ODE: {}", e)))?;

    let extracted = try_extract_ode(&ml_expr).ok_or_else(|| {
        SolverError::CannotSolve(
            "equation does not contain recognizable ODE derivative terms".to_string(),
        )
    })?;

    match extracted {
        ExtractedODE::First(ode) => solve_ode_first_order(&ode),
        ExtractedODE::Second(ode) => solve_ode_second_order(&ode),
    }
}

#[cfg(test)]
mod tests;
