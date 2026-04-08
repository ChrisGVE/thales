//! Conformance tests for resolution path quality.
//!
//! Validates that resolution paths produced by the solver contain meaningful
//! annotations, non-empty explanations, and proper convergence information.

use thales::ast::Variable;
use thales::resolution_path::{Operation, ResolutionPath};
use thales::solver::{SmartSolver, Solver};

// ============================================================================
// Helper Functions
// ============================================================================

/// Validate that a resolution path meets minimum quality standards.
///
/// Returns `Ok(())` when every step has a non-empty explanation, or `Err`
/// with a list of violations.
fn validate_path_quality(path: &ResolutionPath) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    for (i, step) in path.steps.iter().enumerate() {
        if step.explanation.trim().is_empty() {
            errors.push(format!("Step {} has empty explanation", i + 1));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Check whether any step in the path has a technique annotation matching the
/// given substring (case-insensitive).
fn path_has_technique(path: &ResolutionPath, needle: &str) -> bool {
    let needle_lower = needle.to_lowercase();
    path.steps.iter().any(|step| {
        step.annotation
            .as_ref()
            .and_then(|a| a.technique.as_ref())
            .is_some_and(|t| t.to_lowercase().contains(&needle_lower))
    })
}

/// Check whether any step uses a `NumericalConverged` or
/// `SymbolicToNumericalHandoff` operation.
fn path_has_numerical_convergence(path: &ResolutionPath) -> bool {
    path.steps.iter().any(|step| {
        matches!(
            step.operation,
            Operation::NumericalConverged { .. } | Operation::SymbolicToNumericalHandoff { .. }
        )
    })
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn linear_equation_path_quality() {
    let equation = thales::parse_equation("2*x + 3 = 11").unwrap();
    let solver = SmartSolver::new();
    let (_solution, path) = solver.solve(&equation, &Variable::new("x")).unwrap();

    assert!(
        !path.steps.is_empty(),
        "Linear equation path should have at least one step"
    );
    if let Err(errors) = validate_path_quality(&path) {
        panic!("Path quality check failed:\n{}", errors.join("\n"));
    }
}

#[test]
fn quadratic_formula_annotations() {
    let equation = thales::parse_equation("x^2 + 5*x + 6 = 0").unwrap();
    let solver = SmartSolver::new();
    let (_solution, path) = solver.solve(&equation, &Variable::new("x")).unwrap();

    assert!(
        !path.steps.is_empty(),
        "Quadratic equation path should have steps"
    );

    // The path should reference the Quadratic Formula technique
    assert!(
        path_has_technique(&path, "Quadratic Formula"),
        "Quadratic equation path should contain 'Quadratic Formula' technique annotation. \
         Techniques found: {:?}",
        path.steps
            .iter()
            .filter_map(|s| s.annotation.as_ref().and_then(|a| a.technique.as_ref()))
            .collect::<Vec<_>>()
    );
}

#[test]
fn physics_formula_path_quality() {
    // E = m * c^2, solve for m
    let equation = thales::parse_equation("E = m * c^2").unwrap();
    let solver = SmartSolver::new();
    let (_solution, path) = solver.solve(&equation, &Variable::new("m")).unwrap();

    assert!(
        !path.steps.is_empty(),
        "Physics formula path should have steps"
    );
    if let Err(errors) = validate_path_quality(&path) {
        panic!("Path quality check failed:\n{}", errors.join("\n"));
    }
}

#[test]
#[ignore = "Cubic solver annotations not yet fully populated in all paths"]
fn cubic_solver_annotations() {
    // x^3 - 6*x^2 + 11*x - 6 = 0  (roots: 1, 2, 3)
    let equation = thales::parse_equation("x^3 - 6*x^2 + 11*x - 6 = 0").unwrap();
    let solver = SmartSolver::new();
    let (_solution, path) = solver.solve(&equation, &Variable::new("x")).unwrap();

    assert!(
        path_has_technique(&path, "Cardano"),
        "Cubic solver path should contain Cardano's Formula technique annotation. \
         Techniques found: {:?}",
        path.steps
            .iter()
            .filter_map(|s| s.annotation.as_ref().and_then(|a| a.technique.as_ref()))
            .collect::<Vec<_>>()
    );
}

#[test]
fn symbolic_isolation_has_steps() {
    // v = v0 + a*t, solve for t
    let equation = thales::parse_equation("v = v0 + a*t").unwrap();
    let solver = SmartSolver::new();
    let (_solution, path) = solver.solve(&equation, &Variable::new("t")).unwrap();

    assert!(
        !path.steps.is_empty(),
        "Symbolic isolation path for kinematic formula should have steps"
    );
}

#[test]
fn display_produces_nonempty_output() {
    let equation = thales::parse_equation("2*x + 3 = 11").unwrap();
    let solver = SmartSolver::new();
    let (_solution, path) = solver.solve(&equation, &Variable::new("x")).unwrap();

    let display = format!("{}", path);
    assert!(
        !display.trim().is_empty(),
        "Display output for resolution path should be non-empty"
    );
}

#[test]
fn numerical_path_has_convergence_info() {
    // x * exp(x) = 5 — transcendental, requires numerical methods
    let equation = thales::parse_equation("x * exp(x) = 5").unwrap();
    let solver = SmartSolver::new();
    let result = solver.solve(&equation, &Variable::new("x"));

    match result {
        Ok((_solution, path)) => {
            assert!(
                !path.steps.is_empty(),
                "Numerical solution path should have steps"
            );
            // Check for numerical convergence or handoff operations
            assert!(
                path_has_numerical_convergence(&path),
                "Transcendental equation path should contain NumericalConverged or \
                 SymbolicToNumericalHandoff operation. Operations found: {:?}",
                path.steps
                    .iter()
                    .map(|s| format!("{:?}", s.operation))
                    .collect::<Vec<_>>()
            );
        }
        Err(e) => {
            // If the solver can't handle this equation yet, skip gracefully
            panic!(
                "Solver failed on transcendental equation x * exp(x) = 5: {:?}",
                e
            );
        }
    }
}
