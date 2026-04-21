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

use crate::ast::{BinaryOp, Equation, Variable};
use crate::numeric::compile::{compile, decompile};
use crate::numeric::trace::{Step, TechniqueTag, Trace};
use crate::ode::{
    particular_solution_undetermined, solve_linear, solve_second_order_homogeneous,
    solve_separable, FirstOrderODE, SecondOrderODE,
};
use crate::solver::ode_classifier::{classify_first_order, classify_second_order, ODEType};
use crate::solver::types::{Solution, SolverError, SolverResult};
use crate::solver::Solver;

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
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a [`Trace`] from an ordered list of textual ODE steps, prepending
/// a classification step when `classify_step` is `Some`.
fn build_ode_trace(
    classify_step: Option<(String, String)>,
    steps: &[String],
    solution_expr: &crate::ast::Expression,
) -> Trace {
    let mut trace = Trace::new();
    let solution_arc = compile(solution_expr);

    if let Some((order, ode_type)) = classify_step {
        // Classification of an ODE is an inherently calculus-tier act;
        // pick a concrete first-class tag that maps to Calculus rather
        // than `Custom` (which defaults to Advanced) so difficulty
        // filters keep matching the expected tier.
        let classify_tag = tag_for_ode_type(&ode_type);
        trace.push(
            Step::new(
                classify_tag,
                format!(
                    "order={}, ode_type={}; Classify ODE: {}-order, type = {}",
                    order, ode_type, order, ode_type
                ),
            )
            .with_output(solution_arc.clone()),
        );
    }

    for step_desc in steps {
        let tag = tag_for_ode_method(step_desc);
        trace.push(
            Step::new(tag, format!("method={}; {}", step_desc, step_desc))
                .with_output(solution_arc.clone()),
        );
    }
    trace
}

/// Map the free-form ODE classification string to a `TechniqueTag` at
/// Calculus tier. Falls back to `CharacteristicEquation` for unrecognised
/// constant-coefficient classifications and `SeparationOfVariables`
/// otherwise; both sit at Calculus.
fn tag_for_ode_type(ode_type: &str) -> TechniqueTag {
    let lower = ode_type.to_lowercase();
    if lower.contains("separable") {
        TechniqueTag::SeparationOfVariables
    } else if lower.contains("linear") && !lower.contains("non") {
        TechniqueTag::IntegratingFactor
    } else if lower.contains("constant-coefficient") || lower.contains("characteristic") {
        TechniqueTag::CharacteristicEquation
    } else if lower.contains("non-homogeneous") || lower.contains("undetermined") {
        TechniqueTag::UndeterminedCoefficients
    } else {
        TechniqueTag::SeparationOfVariables
    }
}

/// Pick a concrete `TechniqueTag` for an individual ODE solving step based
/// on the textual step description. All returned tags sit at Calculus tier.
fn tag_for_ode_method(step_desc: &str) -> TechniqueTag {
    let lower = step_desc.to_lowercase();
    if lower.contains("separat") {
        TechniqueTag::SeparationOfVariables
    } else if lower.contains("integrating factor") {
        TechniqueTag::IntegratingFactor
    } else if lower.contains("characteristic") {
        TechniqueTag::CharacteristicEquation
    } else if lower.contains("undetermined") {
        TechniqueTag::UndeterminedCoefficients
    } else if lower.contains("variation of parameters") {
        TechniqueTag::VariationOfParameters
    } else if lower.contains("runge") {
        TechniqueTag::RungeKutta
    } else {
        // Fallback: still at Calculus tier via SeparationOfVariables so
        // profile filters continue to see ODE work as calculus-tier.
        TechniqueTag::SeparationOfVariables
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
    let general_expr = decompile(&ode_result.general_solution);
    let path = build_ode_trace(classify, &ode_result.steps, &general_expr);
    Ok((Solution::Unique(general_expr), path))
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

    let hom_expr = decompile(&hom_result.general_solution);
    if ode.is_homogeneous() {
        let path = build_ode_trace(classify, &hom_result.steps, &hom_expr);
        return Ok((Solution::Unique(hom_expr), path));
    }

    // Non-homogeneous: find particular solution then combine.
    let (yp, yp_steps) = particular_solution_undetermined(ode)
        .map_err(|e| SolverError::CannotSolve(e.to_string()))?;

    let general = crate::ast::Expression::Binary(BinaryOp::Add, Box::new(hom_expr), Box::new(yp));

    let mut all_steps = hom_result.steps;
    all_steps.extend(yp_steps);
    all_steps.push(format!("General solution: y = y_h + y_p"));

    let path = build_ode_trace(classify, &all_steps, &general);
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Expression, Variable};
    use crate::ode::{FirstOrderODE, SecondOrderODE};

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn neg(expr: Expression) -> Expression {
        use crate::ast::UnaryOp;
        Expression::Unary(UnaryOp::Neg, Box::new(expr))
    }

    // ------------------------------------------------------------------
    // OdeSolver trait tests
    // ------------------------------------------------------------------

    #[test]
    fn can_solve_always_false() {
        use crate::ast::Equation;
        let solver = OdeSolver::new();
        let eq = Equation::new("test", Expression::Integer(0), Expression::Integer(0));
        assert!(!solver.can_solve(&eq));
    }

    #[test]
    fn solve_trait_returns_unsupported() {
        use crate::ast::Equation;
        let solver = OdeSolver::new();
        let eq = Equation::new("test", Expression::Integer(0), Expression::Integer(0));
        let result = solver.solve(&eq, &Variable::new("x"));
        assert!(matches!(result, Err(SolverError::UnsupportedEquationType)));
    }

    // ------------------------------------------------------------------
    // First-order ODE convenience function tests
    // ------------------------------------------------------------------

    #[test]
    fn solve_first_order_separable_dy_equals_y() {
        // dy/dx = y  →  separable, general solution y = C·eˣ
        let ode = FirstOrderODE::new("y", "x", var("y"));
        let result = solve_ode_first_order(&ode);
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
        let (solution, path) = result.unwrap();
        assert!(matches!(solution, Solution::Unique(_)));
        assert!(!path.steps().is_empty());
    }

    #[test]
    fn solve_first_order_linear_dy_equals_minus_y() {
        // dy/dx = -y  →  linear (P = 1, Q = 0), solution y = C·e^(−x)
        let ode = FirstOrderODE::new("y", "x", neg(var("y")));
        let result = solve_ode_first_order(&ode);
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
        let (solution, path) = result.unwrap();
        assert!(matches!(solution, Solution::Unique(_)));
        assert!(!path.steps().is_empty());
    }

    // ------------------------------------------------------------------
    // Second-order ODE convenience function tests
    // ------------------------------------------------------------------

    #[test]
    fn solve_second_order_homogeneous_y_pp_minus_y() {
        // y'' - y = 0  →  y = C₁·eˣ + C₂·e^(−x)
        let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 0.0, -1.0);
        let result = solve_ode_second_order(&ode);
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
        let (solution, path) = result.unwrap();
        assert!(matches!(solution, Solution::Unique(_)));
        assert!(!path.steps().is_empty());
    }

    #[test]
    fn solve_second_order_non_homogeneous_polynomial_forcing() {
        // y'' + y = x  →  particular solution y_p = x
        let ode = SecondOrderODE::new("y", "x", 1.0, 0.0, 1.0, var("x"));
        let result = solve_ode_second_order(&ode);
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
        let (solution, path) = result.unwrap();
        assert!(matches!(solution, Solution::Unique(_)));
        assert!(!path.steps().is_empty());
    }

    #[test]
    fn solve_second_order_non_homogeneous_exp_forcing() {
        // y'' - y = e^(2x)  →  particular solution y_p = (1/3)·e^(2x)
        use crate::ast::{BinaryOp, Expression, Function};
        let kx = Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Float(2.0)),
            Box::new(var("x")),
        );
        let forcing = Expression::Function(Function::Exp, vec![kx]);
        let ode = SecondOrderODE::new("y", "x", 1.0, 0.0, -1.0, forcing);
        let result = solve_ode_second_order(&ode);
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
        let (solution, _path) = result.unwrap();
        assert!(matches!(solution, Solution::Unique(_)));
    }

    #[test]
    fn solve_second_order_non_homogeneous_trig_resonance() {
        // y'' + y = sin(x)  →  resonant (k=1, char roots ±i)
        use crate::ast::{Expression, Function};
        let forcing = Expression::Function(Function::Sin, vec![var("x")]);
        let ode = SecondOrderODE::new("y", "x", 1.0, 0.0, 1.0, forcing);
        let result = solve_ode_second_order(&ode);
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
        let (solution, _path) = result.unwrap();
        assert!(matches!(solution, Solution::Unique(_)));
    }

    #[test]
    fn solve_second_order_non_homogeneous_unsupported_returns_error() {
        // y'' + y = tan(x)  →  unsupported forcing type
        use crate::ast::{Expression, Function};
        let forcing = Expression::Function(Function::Tan, vec![var("x")]);
        let ode = SecondOrderODE::new("y", "x", 1.0, 0.0, 1.0, forcing);
        let result = solve_ode_second_order(&ode);
        assert!(
            matches!(result, Err(SolverError::CannotSolve(_))),
            "Expected CannotSolve, got {result:?}"
        );
    }

    // ------------------------------------------------------------------
    // Resolution path content tests
    // ------------------------------------------------------------------

    #[test]
    fn first_order_path_starts_with_classify_step() {
        // dy/dx = y → separable; first step must classify as separable ODE
        let ode = FirstOrderODE::new("y", "x", var("y"));
        let (_solution, trace) = solve_ode_first_order(&ode).unwrap();
        let first = trace.steps().first().expect("trace must have steps");
        assert_eq!(first.tag, TechniqueTag::SeparationOfVariables);
        assert!(first.detail.contains("order=first"));
        assert!(first.detail.contains("ode_type=separable"));
    }

    #[test]
    fn second_order_path_starts_with_classify_and_contains_solve_steps() {
        // y'' - y = 0 → second-order homogeneous constant-coefficient; first
        // step tags classification as a characteristic-equation technique.
        let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 0.0, -1.0);
        let (_solution, trace) = solve_ode_second_order(&ode).unwrap();
        let first = trace.steps().first().expect("trace must have steps");
        assert_eq!(first.tag, TechniqueTag::CharacteristicEquation);
        assert!(first.detail.contains("order=second"));
        // Subsequent solving steps stay at calculus tier.
        let all_calculus = trace
            .steps()
            .iter()
            .all(|s| s.tag.difficulty() == crate::numeric::trace::TechniqueDifficulty::Calculus);
        assert!(all_calculus, "Expected every trace step at Calculus tier");
    }

    #[test]
    fn classify_ode_difficulty_is_calculus_tier() {
        // The classification tags used by the ODE trace builder are all at
        // Calculus tier so difficulty filters see ODE work consistently.
        for tag in [
            TechniqueTag::SeparationOfVariables,
            TechniqueTag::IntegratingFactor,
            TechniqueTag::CharacteristicEquation,
            TechniqueTag::UndeterminedCoefficients,
            TechniqueTag::VariationOfParameters,
        ] {
            assert_eq!(
                tag.difficulty(),
                crate::numeric::trace::TechniqueDifficulty::Calculus,
                "Expected Calculus tier for {:?}",
                tag
            );
        }
    }

    // ------------------------------------------------------------------
    // Text-based ODE solving (solve_ode_from_text)
    // ------------------------------------------------------------------

    #[test]
    fn solve_from_text_first_order_separable() {
        // dy/dx = y → separable, y = C·eˣ
        let result = solve_ode_from_text("dy/dx = y");
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
        let (solution, path) = result.unwrap();
        assert!(matches!(solution, Solution::Unique(_)));
        assert!(!path.steps().is_empty());
    }

    #[test]
    fn solve_from_text_first_order_linear() {
        // dy/dx = -y → linear, y = C·e^(-x)
        let result = solve_ode_from_text("dy/dx = -y");
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
        let (solution, _) = result.unwrap();
        assert!(matches!(solution, Solution::Unique(_)));
    }

    #[test]
    fn solve_from_text_second_order_homogeneous() {
        // d2y/dx2 - y = 0 → y = C₁·eˣ + C₂·e^(-x)
        let result = solve_ode_from_text("d2y/dx2 - y = 0");
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
        let (solution, path) = result.unwrap();
        assert!(matches!(solution, Solution::Unique(_)));
        assert!(!path.steps().is_empty());
    }

    #[test]
    fn solve_from_text_second_order_with_first_deriv() {
        // d2y/dx2 + 2*dy/dx + y = 0 → repeated root, y = (C₁ + C₂·x)·e^(-x)
        let result = solve_ode_from_text("d2y/dx2 + 2*dy/dx + y = 0");
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
        let (solution, _) = result.unwrap();
        assert!(matches!(solution, Solution::Unique(_)));
    }

    #[test]
    fn solve_from_text_not_an_ode() {
        let result = solve_ode_from_text("x + y = 0");
        assert!(matches!(result, Err(SolverError::CannotSolve(_))));
    }

    #[test]
    fn solve_from_text_invalid_input() {
        let result = solve_ode_from_text("not valid math @@@");
        assert!(matches!(result, Err(SolverError::CannotSolve(_))));
    }

    #[test]
    fn solve_from_text_diff_notation() {
        // diff(y, x) = y → same as dy/dx = y
        let result = solve_ode_from_text("diff(y, x) = y");
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
        let (solution, _) = result.unwrap();
        assert!(matches!(solution, Solution::Unique(_)));
    }

    // ------------------------------------------------------------------
    // LaTeX-based ODE solving (solve_ode_from_latex)
    // ------------------------------------------------------------------

    #[test]
    fn solve_from_latex_first_order_separable() {
        // \frac{d}{dx}(y) = y → separable, y = C·eˣ
        let result = solve_ode_from_latex(r#"\frac{d}{dx}(y) = y"#);
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
        let (solution, path) = result.unwrap();
        assert!(matches!(solution, Solution::Unique(_)));
        assert!(!path.steps().is_empty());
    }

    #[test]
    fn solve_from_latex_first_order_linear() {
        // \frac{d}{dx}(y) = -y → linear, y = C·e^(-x)
        let result = solve_ode_from_latex(r#"\frac{d}{dx}(y) = -y"#);
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
        let (solution, _) = result.unwrap();
        assert!(matches!(solution, Solution::Unique(_)));
    }

    #[test]
    fn solve_from_latex_second_order_homogeneous() {
        // \frac{d^2}{dx^2}(y) - y = 0
        let result = solve_ode_from_latex(r#"\frac{d^2}{dx^2}(y) - y = 0"#);
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
        let (solution, path) = result.unwrap();
        assert!(matches!(solution, Solution::Unique(_)));
        assert!(!path.steps().is_empty());
    }

    #[test]
    fn solve_from_latex_second_order_with_first_deriv() {
        // \frac{d^2}{dx^2}(y) + 2\frac{d}{dx}(y) + y = 0
        let result = solve_ode_from_latex(r#"\frac{d^2}{dx^2}(y) + 2\frac{d}{dx}(y) + y = 0"#);
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
        let (solution, _) = result.unwrap();
        assert!(matches!(solution, Solution::Unique(_)));
    }

    #[test]
    fn solve_from_latex_not_an_ode() {
        let result = solve_ode_from_latex(r#"x + y = 0"#);
        assert!(matches!(result, Err(SolverError::CannotSolve(_))));
    }

    #[test]
    fn solve_from_latex_invalid_input() {
        let result = solve_ode_from_latex(r#"\invalid{bad"#);
        assert!(matches!(result, Err(SolverError::CannotSolve(_))));
    }

    // ------------------------------------------------------------------
    // Text vs LaTeX equivalence
    // ------------------------------------------------------------------

    #[test]
    fn text_and_latex_produce_same_first_order_solution() {
        let (text_sol, _) = solve_ode_from_text("dy/dx = y").unwrap();
        let (latex_sol, _) = solve_ode_from_latex(r#"\frac{d}{dx}(y) = y"#).unwrap();

        // Both should produce Unique solutions
        let text_expr = match text_sol {
            Solution::Unique(e) => e,
            _ => panic!("text: expected Unique"),
        };
        let latex_expr = match latex_sol {
            Solution::Unique(e) => e,
            _ => panic!("latex: expected Unique"),
        };

        // Evaluate both at x=1 — they should give the same result
        let mut env = std::collections::HashMap::new();
        env.insert("x".to_string(), 1.0);
        env.insert("C".to_string(), 1.0);
        let text_val = text_expr.evaluate(&env);
        let latex_val = latex_expr.evaluate(&env);
        assert_eq!(text_val, latex_val, "text and latex solutions diverge");
    }

    #[test]
    fn text_and_latex_produce_same_second_order_solution() {
        let (text_sol, _) = solve_ode_from_text("d2y/dx2 - y = 0").unwrap();
        let (latex_sol, _) = solve_ode_from_latex(r#"\frac{d^2}{dx^2}(y) - y = 0"#).unwrap();

        let text_expr = match text_sol {
            Solution::Unique(e) => e,
            _ => panic!("text: expected Unique"),
        };
        let latex_expr = match latex_sol {
            Solution::Unique(e) => e,
            _ => panic!("latex: expected Unique"),
        };

        let mut env = std::collections::HashMap::new();
        env.insert("x".to_string(), 1.0);
        env.insert("C_1".to_string(), 1.0);
        env.insert("C_2".to_string(), 1.0);
        let text_val = text_expr.evaluate(&env);
        let latex_val = latex_expr.evaluate(&env);
        assert_eq!(text_val, latex_val, "text and latex solutions diverge");
    }
}
