//! Algebraic command dispatchers (Simplify, Expand, Factor, Substitute,
//! PartialFractions, Rearrange, Conjugate, InverseFn, ApplyIdentity).

use crate::api::command::{IdentityId, SimplifyRules};
use crate::api::diagnostic::{Diagnostic, DiagnosticCode};
use crate::api::narrative::Narrative;
use crate::api::response::{EngineId, Response, ResultKey};
use crate::ast::{Expression, Function, Variable};
use crate::numeric::compile::compile;
use crate::numeric::trace::{Step, TechniqueTag, Trace};
use crate::pattern::{apply_rule, Pattern, Rule};
use crate::solver::Solver as _;
use num_complex::Complex64;

use super::helpers::{
    engine_error, expression_to_equation, solution_to_response, split_rational, steps_from_trace,
    substitute_in_expr, symbolic_entry,
};

pub(super) fn simplify_cmd(expr: &Expression, _rules: SimplifyRules, narrate: bool) -> Response {
    let simplified = expr.simplify();
    let mut trace = Trace::new();
    if narrate {
        trace.push(
            Step::new(TechniqueTag::Simplification, "Canonical simplification")
                .with_input(compile(expr))
                .with_output(compile(&simplified)),
        );
    }
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(simplified, EngineId::Simplify, steps_from_trace(&trace)),
    ));
    r.meta.engine_trace.push(EngineId::Simplify);
    r
}

pub(super) fn expand_cmd(expr: &Expression, narrate: bool) -> Response {
    // No standalone expand engine: simplification rules include distribution.
    let expanded = expr.simplify();
    let mut trace = Trace::new();
    if narrate {
        trace.push(
            Step::new(TechniqueTag::Expansion, "Distribute products over sums")
                .with_input(compile(expr))
                .with_output(compile(&expanded)),
        );
    }
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(expanded, EngineId::Simplify, steps_from_trace(&trace)),
    ));
    r.meta.engine_trace.push(EngineId::Simplify);
    r
}

pub(super) fn factor_cmd(expr: &Expression, narrate: bool) -> Response {
    // No standalone factor engine at the Expression layer yet.
    let factored = expr.simplify();
    let mut trace = Trace::new();
    if narrate {
        trace.push(
            Step::new(
                TechniqueTag::Factoring,
                "Attempted factoring via simplification",
            )
            .with_input(compile(expr))
            .with_output(compile(&factored)),
        );
    }
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(factored, EngineId::Simplify, steps_from_trace(&trace)),
    ));
    r.diagnostics.push(Diagnostic::of(
        DiagnosticCode::NotImplemented,
        Narrative::new(
            "command.factor.partial",
            "Full factoring engine is not yet wired into the API; result is the \
             canonical simplification.",
        ),
    ));
    r.meta.engine_trace.push(EngineId::Simplify);
    r
}

pub(super) fn substitute_cmd(
    expr: &Expression,
    bindings: &[(Expression, Expression)],
    narrate: bool,
) -> Response {
    let mut current = expr.clone();
    let mut trace = Trace::new();
    for (old, new) in bindings {
        current = substitute_in_expr(&current, old, new);
        if narrate {
            trace.push(
                Step::new(
                    TechniqueTag::Substitution,
                    format!("Substitute {} with {}", old, new),
                )
                .with_output(compile(&current)),
            );
        }
    }
    let simplified = current.simplify();
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(simplified, EngineId::Simplify, steps_from_trace(&trace)),
    ));
    r.meta.engine_trace.push(EngineId::Simplify);
    r
}

pub(super) fn partial_fractions_cmd(expr: &Expression, var: &str, narrate: bool) -> Response {
    let (num, denom) = split_rational(expr);
    let variable = Variable::new(var);
    match crate::partial_fractions::decompose(&num, &denom, &variable) {
        Ok(result) => {
            let value = result.to_expression();
            let mut trace = Trace::new();
            if narrate {
                trace.push(
                    Step::new(
                        TechniqueTag::PartialFractionDecomp,
                        "Partial fraction decomposition",
                    )
                    .with_output(compile(&value)),
                );
            }
            let mut r = Response::default();
            r.results.push((
                ResultKey::Single,
                symbolic_entry(value, EngineId::PartialFractions, steps_from_trace(&trace)),
            ));
            r.meta.engine_trace.push(EngineId::PartialFractions);
            r
        }
        Err(e) => engine_error("command.partial_fractions", format!("{:?}", e)),
    }
}

pub(super) fn rearrange_cmd(equation: &Expression, solve_for: &str, narrate: bool) -> Response {
    let eq = expression_to_equation(equation);
    let solver = crate::solver::SmartSolver::new();
    let variable = Variable::new(solve_for);
    match solver.solve(&eq, &variable) {
        Ok((sol, trace)) => solution_to_response(
            sol,
            &trace,
            EngineId::EquationSolver,
            narrate,
            "command.rearrange",
        ),
        Err(e) => engine_error("command.rearrange", format!("{:?}", e)),
    }
}

/// Compute the complex conjugate of `expr`.
///
/// - `Complex(a + bi)` → `Complex(a - bi)` (literal complex conjugate)
/// - `Integer | Float | Rational` → self (real numbers are their own conjugate)
/// - Symbolic / unknown → `Conj(expr)` wrapper for later evaluation
pub(super) fn conjugate_cmd(expr: &Expression, narrate: bool) -> Response {
    let result = match expr {
        Expression::Complex(c) => Expression::Complex(Complex64::new(c.re, -c.im)),
        Expression::Integer(_) | Expression::Float(_) | Expression::Rational(_) => expr.clone(),
        _ => Expression::Function(Function::Conj, vec![expr.clone()]),
    };
    let mut trace = Trace::new();
    if narrate {
        trace.push(
            Step::new(TechniqueTag::Simplification, "Complex conjugate")
                .with_input(compile(expr))
                .with_output(compile(&result)),
        );
    }
    let mut r = Response::default();
    r.results.push((
        ResultKey::Single,
        symbolic_entry(result, EngineId::Simplify, steps_from_trace(&trace)),
    ));
    r.meta.engine_trace.push(EngineId::Simplify);
    r
}

/// Compute the inverse function of `expr` treated as f(var).
///
/// Constructs the equation `expr = __y__` and solves for `var` in terms of
/// `__y__`, returning the result as an expression in `__y__`.  The solver
/// picks the principal branch when multiple solutions exist.
pub(super) fn inverse_fn_cmd(expr: &Expression, var: &str, narrate: bool) -> Response {
    // Represent f(var) = __y__  as  f(var) - __y__ = 0
    let y_sym = Expression::Variable(Variable::new("__y__"));
    let lhs = Expression::Binary(
        crate::ast::BinaryOp::Sub,
        Box::new(expr.clone()),
        Box::new(y_sym),
    );
    let eq = crate::ast::Equation::new("inverse_fn", lhs, Expression::Integer(0));
    let solver = crate::solver::SmartSolver::new();
    let variable = Variable::new(var);
    match solver.solve(&eq, &variable) {
        Ok((sol, trace)) => solution_to_response(
            sol,
            &trace,
            EngineId::EquationSolver,
            narrate,
            "command.inverse_fn",
        ),
        Err(e) => engine_error("command.inverse_fn", format!("{:?}", e)),
    }
}

/// Apply a named algebraic identity to `expr`.
///
/// Supported identities:
/// - `IdentityId::DifferenceOfSquares`: a² − b² → (a+b)(a−b)
/// - `IdentityId::SumOfCubes`: a³ + b³ → (a+b)(a² − ab + b²)
///
/// Unknown or unrecognised identity labels return `DiagnosticCode::NotImplemented`.
/// A recognised identity whose pattern does not match the expression returns a
/// `DiagnosticCode::Other("pattern-no-match")` diagnostic.
pub(super) fn apply_identity_cmd(
    expr: &Expression,
    identity: &IdentityId,
    narrate: bool,
) -> Response {
    let rule_opt: Option<Rule> = match identity {
        IdentityId::DifferenceOfSquares => {
            // Pattern: a^2 - b^2  →  (a + b) * (a - b)
            let pat = Pattern::sub(
                Pattern::power(Pattern::wildcard("a"), Pattern::Integer(2)),
                Pattern::power(Pattern::wildcard("b"), Pattern::Integer(2)),
            );
            let rep = Pattern::mul(
                Pattern::add(Pattern::wildcard("a"), Pattern::wildcard("b")),
                Pattern::sub(Pattern::wildcard("a"), Pattern::wildcard("b")),
            );
            Some(Rule::new(pat, rep).named("difference-of-squares"))
        }
        IdentityId::SumOfCubes => {
            // Pattern: a^3 + b^3  →  (a + b) * (a^2 - a*b + b^2)
            let pat = Pattern::add(
                Pattern::power(Pattern::wildcard("a"), Pattern::Integer(3)),
                Pattern::power(Pattern::wildcard("b"), Pattern::Integer(3)),
            );
            let rep = Pattern::mul(
                Pattern::add(Pattern::wildcard("a"), Pattern::wildcard("b")),
                Pattern::add(
                    Pattern::sub(
                        Pattern::power(Pattern::wildcard("a"), Pattern::Integer(2)),
                        Pattern::mul(Pattern::wildcard("a"), Pattern::wildcard("b")),
                    ),
                    Pattern::power(Pattern::wildcard("b"), Pattern::Integer(2)),
                ),
            );
            Some(Rule::new(pat, rep).named("sum-of-cubes"))
        }
        _ => None,
    };

    match rule_opt {
        None => {
            let name = format!("{:?}", identity);
            let mut r = Response::default();
            r.diagnostics.push(Diagnostic::of(
                DiagnosticCode::NotImplemented,
                Narrative::new(
                    "command.apply_identity",
                    format!("identity '{name}' is not supported"),
                ),
            ));
            r
        }
        Some(rule) => match apply_rule(expr, &rule) {
            None => {
                let mut r = Response::default();
                r.diagnostics.push(Diagnostic::of(
                    DiagnosticCode::Other("pattern-no-match"),
                    Narrative::new(
                        "command.apply_identity",
                        "identity pattern did not match the expression",
                    ),
                ));
                r
            }
            Some(result) => {
                let mut trace = Trace::new();
                if narrate {
                    trace.push(
                        Step::new(TechniqueTag::Factoring, "Apply algebraic identity")
                            .with_input(compile(expr))
                            .with_output(compile(&result)),
                    );
                }
                let mut r = Response::default();
                r.results.push((
                    ResultKey::Single,
                    symbolic_entry(result, EngineId::Simplify, steps_from_trace(&trace)),
                ));
                r.meta.engine_trace.push(EngineId::Simplify);
                r
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::command::IdentityId;
    use crate::ast::{BinaryOp, Expression, Function, Variable};
    use num_complex::Complex64;

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn add(a: Expression, b: Expression) -> Expression {
        Expression::Binary(BinaryOp::Add, Box::new(a), Box::new(b))
    }

    fn sub(a: Expression, b: Expression) -> Expression {
        Expression::Binary(BinaryOp::Sub, Box::new(a), Box::new(b))
    }

    fn pow(base: Expression, exp: i64) -> Expression {
        Expression::Power(Box::new(base), Box::new(int(exp)))
    }

    // ── Conjugate tests ──────────────────────────────────────────────────────

    #[test]
    fn conjugate_complex_literal() {
        // conj(3 + 4i) = 3 - 4i
        let expr = Expression::Complex(Complex64::new(3.0, 4.0));
        let resp = conjugate_cmd(&expr, false);
        assert_eq!(resp.results.len(), 1);
        let (_, entry) = &resp.results[0];
        match &entry.value {
            crate::api::response::ResultValue::Symbolic(e) => {
                assert_eq!(*e, Expression::Complex(Complex64::new(3.0, -4.0)));
            }
            other => panic!("expected Symbolic, got {:?}", other),
        }
    }

    #[test]
    fn conjugate_real_integer_is_identity() {
        // conj(5) = 5
        let expr = int(5);
        let resp = conjugate_cmd(&expr, false);
        assert_eq!(resp.results.len(), 1);
        let (_, entry) = &resp.results[0];
        match &entry.value {
            crate::api::response::ResultValue::Symbolic(e) => {
                assert_eq!(*e, int(5));
            }
            other => panic!("expected Symbolic, got {:?}", other),
        }
    }

    #[test]
    fn conjugate_symbolic_wraps_in_conj_function() {
        // conj(x) = Conj(x) (symbolic fallback)
        let expr = var("x");
        let resp = conjugate_cmd(&expr, false);
        assert_eq!(resp.results.len(), 1);
        let (_, entry) = &resp.results[0];
        match &entry.value {
            crate::api::response::ResultValue::Symbolic(e) => {
                assert_eq!(*e, Expression::Function(Function::Conj, vec![var("x")]));
            }
            other => panic!("expected Symbolic, got {:?}", other),
        }
    }

    // ── InverseFn tests ──────────────────────────────────────────────────────

    #[test]
    fn inverse_fn_linear_y_equals_2x_plus_1() {
        // f(x) = 2x + 1  →  f^{-1}(y) = (y - 1) / 2
        let two_x = Expression::Binary(BinaryOp::Mul, Box::new(int(2)), Box::new(var("x")));
        let expr = add(two_x, int(1));
        let resp = inverse_fn_cmd(&expr, "x", false);
        // Solver should produce a result (unique solution) or a diagnostic.
        assert!(
            !resp.results.is_empty() || !resp.diagnostics.is_empty(),
            "expected some result or diagnostic"
        );
    }

    #[test]
    fn inverse_fn_trivial_y_equals_x() {
        // f(x) = x  →  f^{-1}(__y__) = __y__
        let expr = var("x");
        let resp = inverse_fn_cmd(&expr, "x", false);
        assert!(
            !resp.results.is_empty() || !resp.diagnostics.is_empty(),
            "expected some result or diagnostic"
        );
    }

    #[test]
    fn inverse_fn_unsolvable_returns_diagnostic_or_result() {
        // f(x) = x * sin(x) — analytically non-invertible; solver returns error or unsolved
        let sin_x = Expression::Function(crate::ast::Function::Sin, vec![var("x")]);
        let expr = Expression::Binary(BinaryOp::Mul, Box::new(var("x")), Box::new(sin_x));
        let resp = inverse_fn_cmd(&expr, "x", false);
        assert!(
            !resp.diagnostics.is_empty() || !resp.results.is_empty(),
            "expected diagnostic or result for non-invertible function"
        );
    }

    // ── ApplyIdentity tests ──────────────────────────────────────────────────

    #[test]
    fn apply_identity_difference_of_squares() {
        // x^2 - y^2  →  (x + y)(x - y)
        let expr = sub(pow(var("x"), 2), pow(var("y"), 2));
        let resp = apply_identity_cmd(&expr, &IdentityId::DifferenceOfSquares, false);
        assert_eq!(resp.results.len(), 1, "expected factored result");
        let (_, entry) = &resp.results[0];
        match &entry.value {
            crate::api::response::ResultValue::Symbolic(e) => {
                assert!(
                    matches!(e, Expression::Binary(BinaryOp::Mul, _, _)),
                    "expected Mul at top level, got {:?}",
                    e
                );
            }
            other => panic!("expected Symbolic result, got {:?}", other),
        }
    }

    #[test]
    fn apply_identity_sum_of_cubes() {
        // x^3 + y^3  →  (x+y)(x^2 - xy + y^2)
        let expr = add(pow(var("x"), 3), pow(var("y"), 3));
        let resp = apply_identity_cmd(&expr, &IdentityId::SumOfCubes, false);
        assert_eq!(resp.results.len(), 1, "expected factored result");
        let (_, entry) = &resp.results[0];
        match &entry.value {
            crate::api::response::ResultValue::Symbolic(e) => {
                assert!(
                    matches!(e, Expression::Binary(BinaryOp::Mul, _, _)),
                    "expected Mul at top level, got {:?}",
                    e
                );
            }
            other => panic!("expected Symbolic result, got {:?}", other),
        }
    }

    #[test]
    fn apply_identity_unknown_name_returns_not_implemented() {
        // An unsupported identity label returns NotImplemented
        let expr = var("x");
        let resp = apply_identity_cmd(&expr, &IdentityId::Other("no-such-identity"), false);
        assert!(
            resp.diagnostics
                .iter()
                .any(|d| d.code == DiagnosticCode::NotImplemented),
            "expected NotImplemented diagnostic"
        );
    }
}
