//! Conformance tests for `Trace` quality.
//!
//! Validates that traces produced by the solver contain meaningful technique
//! tags, non-empty details, and proper convergence information.

use thales::ast::Variable;
use thales::numeric::trace::{TechniqueTag, Trace};
use thales::solver::{SmartSolver, Solver};

// ============================================================================
// Helper Functions
// ============================================================================

/// Validate that a trace meets minimum quality standards.
///
/// Returns `Ok(())` when every step has a non-empty detail, or `Err` with a
/// list of violations.
fn validate_trace_quality(trace: &Trace) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    for (i, step) in trace.steps().iter().enumerate() {
        if step.detail.trim().is_empty() {
            errors.push(format!("Step {} has empty detail", i + 1));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Check whether any step's detail contains the given substring
/// (case-insensitive). Technique names are encoded as a prefix in the
/// `detail` string.
fn trace_has_technique(trace: &Trace, needle: &str) -> bool {
    let needle_lower = needle.to_lowercase();
    trace
        .steps()
        .iter()
        .any(|s| s.detail.to_lowercase().contains(&needle_lower))
}

/// Check whether any step flags a numerical convergence or handoff event.
fn trace_has_numerical_convergence(trace: &Trace) -> bool {
    trace.steps().iter().any(|s| {
        s.tag == TechniqueTag::NumericalApproximation
            || s.tag == TechniqueTag::Custom("SymbolicToNumericalHandoff")
    })
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn linear_equation_path_quality() {
    let equation = thales::parse_equation("2*x + 3 = 11").unwrap();
    let solver = SmartSolver::new();
    let (_solution, trace) = solver.solve(&equation, &Variable::new("x")).unwrap();

    assert!(
        !trace.steps().is_empty(),
        "Linear equation trace should have at least one step"
    );
    if let Err(errors) = validate_trace_quality(&trace) {
        panic!("Trace quality check failed:\n{}", errors.join("\n"));
    }
}

#[test]
fn quadratic_formula_annotations() {
    let equation = thales::parse_equation("x^2 + 5*x + 6 = 0").unwrap();
    let solver = SmartSolver::new();
    let (_solution, trace) = solver.solve(&equation, &Variable::new("x")).unwrap();

    assert!(
        !trace.steps().is_empty(),
        "Quadratic equation trace should have steps"
    );

    assert!(
        trace_has_technique(&trace, "Quadratic Formula"),
        "Quadratic equation trace should mention 'Quadratic Formula'. Steps: {:?}",
        trace.steps().iter().map(|s| &s.detail).collect::<Vec<_>>()
    );
}

#[test]
fn physics_formula_path_quality() {
    // E = m * c^2, solve for m
    let equation = thales::parse_equation("E = m * c^2").unwrap();
    let solver = SmartSolver::new();
    let (_solution, trace) = solver.solve(&equation, &Variable::new("m")).unwrap();

    assert!(
        !trace.steps().is_empty(),
        "Physics formula trace should have steps"
    );
    if let Err(errors) = validate_trace_quality(&trace) {
        panic!("Trace quality check failed:\n{}", errors.join("\n"));
    }
}

#[test]
#[ignore = "Cubic solver annotations not yet fully populated in all traces"]
fn cubic_solver_annotations() {
    let equation = thales::parse_equation("x^3 - 6*x^2 + 11*x - 6 = 0").unwrap();
    let solver = SmartSolver::new();
    let (_solution, trace) = solver.solve(&equation, &Variable::new("x")).unwrap();

    assert!(
        trace_has_technique(&trace, "Cardano"),
        "Cubic solver trace should mention Cardano's Formula. Steps: {:?}",
        trace.steps().iter().map(|s| &s.detail).collect::<Vec<_>>()
    );
}

#[test]
fn symbolic_isolation_has_steps() {
    // v = v0 + a*t, solve for t
    let equation = thales::parse_equation("v = v0 + a*t").unwrap();
    let solver = SmartSolver::new();
    let (_solution, trace) = solver.solve(&equation, &Variable::new("t")).unwrap();

    assert!(
        !trace.steps().is_empty(),
        "Symbolic isolation trace for kinematic formula should have steps"
    );
}

#[test]
fn display_produces_nonempty_output() {
    let equation = thales::parse_equation("2*x + 3 = 11").unwrap();
    let solver = SmartSolver::new();
    let (_solution, trace) = solver.solve(&equation, &Variable::new("x")).unwrap();

    let display = format!("{:?}", trace);
    assert!(
        !display.trim().is_empty(),
        "Debug output for Trace should be non-empty"
    );
}

#[test]
fn numerical_path_has_convergence_info() {
    // x * exp(x) = 5 — transcendental, requires numerical methods
    let equation = thales::parse_equation("x * exp(x) = 5").unwrap();
    let solver = SmartSolver::new();
    let result = solver.solve(&equation, &Variable::new("x"));

    match result {
        Ok((_solution, trace)) => {
            assert!(
                !trace.steps().is_empty(),
                "Numerical solution trace should have steps"
            );
            assert!(
                trace_has_numerical_convergence(&trace),
                "Transcendental equation trace should contain NumericalApproximation or \
                 SymbolicToNumericalHandoff step. Tags found: {:?}",
                trace.steps().iter().map(|s| s.tag).collect::<Vec<_>>()
            );
        }
        Err(e) => {
            panic!(
                "Solver failed on transcendental equation x * exp(x) = 5: {:?}",
                e
            );
        }
    }
}
