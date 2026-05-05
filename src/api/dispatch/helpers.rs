//! Shared dispatch helpers.
//!
//! Helpers used by per-command-family submodules. All conversions between
//! `Expression` and `Arc<Expr>` happen at this seam (architecture rule 2).

use crate::api::diagnostic::{Diagnostic, DiagnosticCode};
use crate::api::domain::Domain;
use crate::api::narrative::{Narrative, NarrativeValue};
use crate::api::request::{Budget, Precision, Request, SolveMode};
use crate::api::response::{
    BranchEntry, EngineId, NarratedStep, Response, ResultEntry, ResultKey, ResultShape,
    ResultValue, StructuredResult,
};
use crate::ast::Expression;
use crate::numeric::compile::decompile;
use crate::numeric::trace::{Step, TechniqueTag, Trace};

/// Emit a [`DiagnosticCode::FieldIgnored`] warning on `response` for a
/// field that was provided in the request but had no effect on this command.
pub(super) fn warn_ignored_field(
    response: &mut Response,
    field_name: &'static str,
    template_id: &'static str,
) {
    response.diagnostics.push(Diagnostic::of(
        DiagnosticCode::FieldIgnored,
        Narrative::new(
            template_id,
            format!(
                "field `{}` was provided but is not yet implemented; it had no effect",
                field_name
            ),
        ),
    ));
}

/// Execution-wide context extracted from [`Request`] fields that the
/// current dispatcher arm may or may not honour.
///
/// Created once at the top of [`super::execute`] and passed down so that
/// every arm can emit [`DiagnosticCode::FieldIgnored`] diagnostics for
/// fields it does not consume, rather than silently dropping them.
pub(super) struct DispatchContext {
    pub narrate: bool,
    pub mode: SolveMode,
    pub precision: Option<Precision>,
    pub budget: Option<Budget>,
    pub ambient_domain: Option<Domain>,
}

impl DispatchContext {
    /// Extract the context fields from a [`Request`].
    pub fn from_request(request: &Request) -> Self {
        Self {
            narrate: request.narrate,
            mode: request.mode,
            precision: request.precision.clone(),
            budget: request.budget,
            ambient_domain: request.ambient_domain,
        }
    }

    /// Emit [`DiagnosticCode::FieldIgnored`] warnings for every non-default
    /// context field that this command does not honour.
    ///
    /// `honors_mode` — pass `true` only for commands that genuinely
    /// consume [`Request::mode`] (currently only `DefIntegrate`).
    pub fn warn_unhandled(&self, response: &mut Response, honors_mode: bool) {
        if !honors_mode && self.mode != SolveMode::Symbolic {
            warn_ignored_field(response, "mode", "request.field_ignored");
        }
        if self.precision.is_some() {
            warn_ignored_field(response, "precision", "request.field_ignored");
        }
        if self.budget.is_some() {
            warn_ignored_field(response, "budget", "request.field_ignored");
        }
        if self.ambient_domain.is_some() {
            warn_ignored_field(response, "ambient_domain", "request.field_ignored");
        }
    }
}

pub(super) fn symbolic_entry(
    value: Expression,
    engine: EngineId,
    steps: Vec<NarratedStep>,
) -> ResultEntry {
    ResultEntry {
        value: ResultValue::Symbolic(value),
        structured: None,
        shape: ResultShape::Scalar,
        unit: None,
        steps,
        alternatives: Vec::new(),
        engine,
    }
}

pub(super) fn unsolved_entry(narrative: Narrative, engine: EngineId) -> ResultEntry {
    ResultEntry {
        value: ResultValue::Unsolved { reason: narrative },
        structured: None,
        shape: ResultShape::Scalar,
        unit: None,
        steps: Vec::new(),
        alternatives: Vec::new(),
        engine,
    }
}

pub(super) fn engine_error(template_id: &'static str, message: String) -> Response {
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        unsolved_entry(
            Narrative::new(template_id, message.clone()),
            EngineId::Other("engine-error"),
        ),
    ));
    r.diagnostics.push(Diagnostic::of(
        DiagnosticCode::Other("engine-error"),
        Narrative::new(template_id, message),
    ));
    r
}

pub(super) fn steps_from_trace(trace: &Trace) -> Vec<NarratedStep> {
    trace
        .steps()
        .iter()
        .map(|step| NarratedStep {
            tag: step.tag,
            difficulty: step.tag.difficulty(),
            narrative: build_step_narrative(template_id_for_tag(step.tag), step),
            path: None,
            input: step.input.as_ref().map(|arc| decompile(arc)),
            output: step.output.as_ref().map(|arc| decompile(arc)),
            unit_trace: None,
        })
        .collect()
}

/// Map a [`TechniqueTag`] to its template identifier string.
///
/// Every variant is listed explicitly so that adding a new variant to
/// `TechniqueTag` causes a compile error here until the mapping is extended.
pub(super) fn template_id_for_tag(tag: TechniqueTag) -> &'static str {
    match tag {
        TechniqueTag::AddBothSides => "step.add_both_sides",
        TechniqueTag::SubtractBothSides => "step.subtract_both_sides",
        TechniqueTag::MultiplyBothSides => "step.multiply_both_sides",
        TechniqueTag::DivideBothSides => "step.divide_both_sides",
        TechniqueTag::PowerBothSides => "step.power_both_sides",
        TechniqueTag::RootBothSides => "step.root_both_sides",
        TechniqueTag::ApplyFunction => "step.apply_function",
        TechniqueTag::Simplification => "step.simplification",
        TechniqueTag::Expansion => "step.expansion",
        TechniqueTag::Factoring => "step.factoring",
        TechniqueTag::CombineFractions => "step.combine_fractions",
        TechniqueTag::Cancellation => "step.cancellation",
        TechniqueTag::CombiningLikeTerms => "step.combining_like_terms",
        TechniqueTag::Rationalization => "step.rationalization",
        TechniqueTag::Conjugation => "step.conjugation",
        TechniqueTag::PartialFractionDecomp => "step.partial_fraction_decomp",
        TechniqueTag::Isolation => "step.isolation",
        TechniqueTag::MoveTerm => "step.move_term",
        TechniqueTag::Substitution => "step.substitution",
        TechniqueTag::ApplyIdentity => "step.apply_identity",
        TechniqueTag::TrigIdentity => "step.trig_identity",
        TechniqueTag::LogIdentity => "step.log_identity",
        TechniqueTag::ExpIdentity => "step.exp_identity",
        TechniqueTag::HyperbolicIdentity => "step.hyperbolic_identity",
        TechniqueTag::EulerFormula => "step.euler_formula",
        TechniqueTag::DeMoivre => "step.de_moivre",
        TechniqueTag::QuadraticFormula => "step.quadratic_formula",
        TechniqueTag::CompleteTheSquare => "step.complete_the_square",
        TechniqueTag::RationalRootTheorem => "step.rational_root_theorem",
        TechniqueTag::SyntheticDivision => "step.synthetic_division",
        TechniqueTag::TotalDifferential => "step.total_differential",
        TechniqueTag::Divergence => "step.divergence",
        TechniqueTag::Curl => "step.curl",
        TechniqueTag::Laplacian => "step.laplacian",
        TechniqueTag::Jacobian => "step.jacobian",
        TechniqueTag::Hessian => "step.hessian",
        TechniqueTag::DirectionalDerivative => "step.directional_derivative",
        TechniqueTag::SpecialFunction => "step.special_function",
        TechniqueTag::PowerRule => "step.power_rule",
        TechniqueTag::ProductRule => "step.product_rule",
        TechniqueTag::QuotientRule => "step.quotient_rule",
        TechniqueTag::ChainRule => "step.chain_rule",
        TechniqueTag::ImplicitDifferentiation => "step.implicit_differentiation",
        TechniqueTag::LogarithmicDifferentiation => "step.logarithmic_differentiation",
        TechniqueTag::PatternIntegration => "step.pattern_integration",
        TechniqueTag::IntegrationByParts => "step.integration_by_parts",
        TechniqueTag::USubstitution => "step.u_substitution",
        TechniqueTag::TrigSubstitution => "step.trig_substitution",
        TechniqueTag::PartialFractionIntegration => "step.partial_fraction_integration",
        TechniqueTag::RischVerification => "step.risch_verification",
        TechniqueTag::LHopitalRule => "step.l_hopital_rule",
        TechniqueTag::SqueezeTheorem => "step.squeeze_theorem",
        TechniqueTag::LimitAlgebraic => "step.limit_algebraic",
        TechniqueTag::TaylorExpansion => "step.taylor_expansion",
        TechniqueTag::LaurentExpansion => "step.laurent_expansion",
        TechniqueTag::AsymptoticExpansion => "step.asymptotic_expansion",
        TechniqueTag::SeriesComposition => "step.series_composition",
        TechniqueTag::LagrangeReversion => "step.lagrange_reversion",
        TechniqueTag::PuiseuxExpansion => "step.puiseux_expansion",
        TechniqueTag::FrobeniusMethod => "step.frobenius_method",
        TechniqueTag::PadeApproximant => "step.pade_approximant",
        TechniqueTag::WkbApproximation => "step.wkb_approximation",
        TechniqueTag::ResidueTheorem => "step.residue_theorem",
        TechniqueTag::PoleClassification => "step.pole_classification",
        TechniqueTag::FourierSeries => "step.fourier_series",
        TechniqueTag::SeparationOfVariables => "step.separation_of_variables",
        TechniqueTag::IntegratingFactor => "step.integrating_factor",
        TechniqueTag::CharacteristicEquation => "step.characteristic_equation",
        TechniqueTag::UndeterminedCoefficients => "step.undetermined_coefficients",
        TechniqueTag::VariationOfParameters => "step.variation_of_parameters",
        TechniqueTag::RungeKutta => "step.runge_kutta",
        TechniqueTag::GaussianElimination => "step.gaussian_elimination",
        TechniqueTag::MatrixInverse => "step.matrix_inverse",
        TechniqueTag::LuDecomposition => "step.lu_decomposition",
        TechniqueTag::QrDecomposition => "step.qr_decomposition",
        TechniqueTag::Eigendecomposition => "step.eigendecomposition",
        TechniqueTag::Determinant => "step.determinant",
        TechniqueTag::NumericalApproximation => "step.numerical_approximation",
        TechniqueTag::NewtonRaphson => "step.newton_raphson",
        TechniqueTag::Bisection => "step.bisection",
        TechniqueTag::Brent => "step.brent",
        TechniqueTag::Secant => "step.secant",
        TechniqueTag::DomainNarrowing => "step.domain_narrowing",
        TechniqueTag::DomainExtension => "step.domain_extension",
        TechniqueTag::PrincipalBranch => "step.principal_branch",
        TechniqueTag::GeomDistance => "step.geom_distance",
        TechniqueTag::GeomIntersection => "step.geom_intersection",
        TechniqueTag::GeomTangent => "step.geom_tangent",
        TechniqueTag::GeomNormal => "step.geom_normal",
        TechniqueTag::GeomCurvature => "step.geom_curvature",
        TechniqueTag::GeomTransform => "step.geom_transform",
        TechniqueTag::GeomExteriorDerivative => "step.geom_exterior_derivative",
        TechniqueTag::Custom(_) => "step.generic",
    }
}

/// Build a [`Narrative`] for a [`Step`] using the given template identifier.
///
/// Binds `input` and `output` expression values when present, then parses
/// structured `"key=val;key=val"` pairs in [`Step::detail`] into additional
/// text bindings. The `input` and `output` keys from the detail string are
/// silently ignored so they cannot overwrite the expression bindings.
fn build_step_narrative(template_id: &'static str, step: &Step) -> Narrative {
    let mut narr = Narrative::new(template_id, step.detail.clone());

    if let Some(ref arc) = step.input {
        narr = narr.bind("input", NarrativeValue::Expr(decompile(arc)));
    }
    if let Some(ref arc) = step.output {
        narr = narr.bind("output", NarrativeValue::Expr(decompile(arc)));
    }

    if step.detail.contains('=') {
        for pair in step.detail.split(';') {
            if let Some((key, val)) = pair.split_once('=') {
                let key = key.trim();
                let val = val.trim();
                if key != "input" && key != "output" && !key.is_empty() {
                    narr = narr.bind(key, NarrativeValue::Text(val.to_string()));
                }
            }
        }
    }

    narr
}

pub(super) fn solution_to_response(
    sol: crate::solver::Solution,
    trace: &Trace,
    engine: EngineId,
    narrate: bool,
    template_id: &'static str,
) -> Response {
    use crate::solver::Solution;
    let mut r = Response::default();
    let steps = if narrate {
        steps_from_trace(trace)
    } else {
        Vec::new()
    };
    match sol {
        Solution::Unique(expr) => r
            .results
            .push((ResultKey::Single, symbolic_entry(expr, engine, steps))),
        Solution::Multiple(exprs) => {
            let branches: Vec<BranchEntry> = exprs
                .iter()
                .enumerate()
                .map(|(i, e)| BranchEntry {
                    condition: None,
                    label: Some(format!("root_{}", i + 1)),
                    value: e.clone(),
                })
                .collect();
            let structured = StructuredResult::Branches { branches };
            for (i, expr) in exprs.into_iter().enumerate() {
                let mut entry = symbolic_entry(expr.clone(), engine, steps.clone());
                // Only the first result entry carries the full Branches
                // structured data; subsequent entries carry None to avoid
                // redundant cloning for large solution sets.
                entry.structured = if i == 0 {
                    Some(structured.clone())
                } else {
                    None
                };
                r.results.push((ResultKey::Single, entry));
            }
        }
        Solution::Infinite => r.diagnostics.push(Diagnostic::of(
            DiagnosticCode::Other("infinite-solutions"),
            Narrative::new(template_id, "equation is an identity"),
        )),
        Solution::None => r.diagnostics.push(Diagnostic::of(
            DiagnosticCode::NoSolutionInDomain,
            Narrative::new(template_id, "no solution in domain"),
        )),
        Solution::Parametric { expression, .. } => r
            .results
            .push((ResultKey::Single, symbolic_entry(expression, engine, steps))),
    }
    r.meta.engine_trace.push(engine);
    r
}

pub(super) fn expression_to_equation(expr: &Expression) -> crate::ast::Equation {
    if let Expression::Binary(crate::ast::BinaryOp::Sub, l, r) = expr {
        return crate::ast::Equation::new("dispatch", (**l).clone(), (**r).clone());
    }
    crate::ast::Equation::new("dispatch", expr.clone(), Expression::Integer(0))
}

pub(super) fn split_rational(expr: &Expression) -> (Expression, Expression) {
    use crate::ast::BinaryOp;
    if let Expression::Binary(BinaryOp::Div, num, denom) = expr {
        ((**num).clone(), (**denom).clone())
    } else {
        (expr.clone(), Expression::Integer(1))
    }
}

pub(super) fn substitute_in_expr(
    expr: &Expression,
    old: &Expression,
    new: &Expression,
) -> Expression {
    if expr == old {
        return new.clone();
    }
    match expr {
        Expression::Binary(op, l, r) => Expression::Binary(
            *op,
            Box::new(substitute_in_expr(l, old, new)),
            Box::new(substitute_in_expr(r, old, new)),
        ),
        Expression::Unary(op, inner) => {
            Expression::Unary(*op, Box::new(substitute_in_expr(inner, old, new)))
        }
        Expression::Power(base, exp) => Expression::Power(
            Box::new(substitute_in_expr(base, old, new)),
            Box::new(substitute_in_expr(exp, old, new)),
        ),
        Expression::Function(f, args) => Expression::Function(
            f.clone(),
            args.iter()
                .map(|a| substitute_in_expr(a, old, new))
                .collect(),
        ),
        other => other.clone(),
    }
}

pub(super) fn expression_to_f64(expr: &Expression) -> Option<f64> {
    match expr {
        Expression::Integer(n) => Some(*n as f64),
        Expression::Float(f) => Some(*f),
        Expression::Rational(r) => Some(*r.numer() as f64 / *r.denom() as f64),
        _ => expr.evaluate(&std::collections::HashMap::new()),
    }
}

#[cfg(test)]
#[path = "helpers_tests.rs"]
mod tests;
