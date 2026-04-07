//! FFI implementation functions for parsing, solving, evaluation, and simplification.

use crate::parser::{parse_equation, parse_expression};

/// Parse equation and return string representation.
pub(super) fn parse_equation_ffi(input: &str) -> Result<String, String> {
    parse_equation(input)
        .map(|eq| format!("{}", eq))
        .map_err(|e| format!("Parse error: {:?}", e))
}

/// Parse expression and return string representation.
pub(super) fn parse_expression_ffi(input: &str) -> Result<String, String> {
    parse_expression(input)
        .map(|expr| format!("{}", expr))
        .map_err(|e| format!("Parse error: {:?}", e))
}

/// Solve equation symbolically.
pub(super) fn solve_equation_ffi(equation: &str, variable: &str) -> Result<String, String> {
    use crate::solver::solve_for;
    use std::collections::HashMap;

    let parsed_equation =
        parse_equation(equation).map_err(|e| format!("Failed to parse equation: {:?}", e))?;

    let known_values = HashMap::new();
    let resolution_path = solve_for(&parsed_equation, variable, &known_values)
        .map_err(|e| format!("Failed to solve equation: {:?}", e))?;

    Ok(format!("{:?}", resolution_path.result))
}

/// Solve equation numerically.
pub(super) fn solve_numerically_ffi(
    equation: &str,
    variable: &str,
    initial_guess: f64,
) -> Result<f64, String> {
    use crate::ast::Variable;
    use crate::numerical::{NumericalConfig, SmartNumericalSolver};

    let parsed_equation =
        parse_equation(equation).map_err(|e| format!("Failed to parse equation: {:?}", e))?;

    let mut config = NumericalConfig::default();
    config.initial_guess = Some(initial_guess);
    let solver = SmartNumericalSolver::new(config);

    let target_var = Variable::new(variable);
    let (solution, _path) = solver
        .solve(&parsed_equation, &target_var)
        .map_err(|e| format!("Numerical solving failed: {:?}", e))?;

    if !solution.converged {
        return Err(format!(
            "Did not converge after {} iterations (residual: {})",
            solution.iterations, solution.residual
        ));
    }

    Ok(solution.value)
}

/// Solve equation with known values and return full resolution path.
pub(super) fn solve_with_values_ffi(
    equation: &str,
    variable: &str,
    known_values_json: &str,
) -> Result<super::ffi::ResolutionPathFFI, String> {
    use crate::solver::solve_for;
    use std::collections::HashMap;

    let parsed_equation =
        parse_equation(equation).map_err(|e| format!("Failed to parse equation: {:?}", e))?;

    let known_values: HashMap<String, f64> = if known_values_json.is_empty() {
        HashMap::new()
    } else {
        serde_json::from_str(known_values_json)
            .map_err(|e| format!("Failed to parse known values JSON: {}", e))?
    };

    let resolution_path = solve_for(&parsed_equation, variable, &known_values)
        .map_err(|e| format!("Failed to solve equation: {:?}", e))?;

    let steps: Vec<serde_json::Value> = resolution_path
        .steps
        .iter()
        .map(|step| {
            serde_json::json!({
                "operation": step.operation.describe(),
                "explanation": step.explanation,
                "result": format!("{:?}", step.result)
            })
        })
        .collect();

    let steps_json =
        serde_json::to_string(&steps).map_err(|e| format!("Failed to serialize steps: {}", e))?;

    Ok(super::ffi::ResolutionPathFFI {
        initial_expr: format!("{:?}", resolution_path.initial),
        steps_json,
        result_expr: format!("{:?}", resolution_path.result),
        success: true,
    })
}

// =============================================================================
// LaTeX functions
// =============================================================================

/// Parse LaTeX expression and return human-readable string representation.
pub(super) fn parse_latex_ffi(input: &str) -> Result<String, String> {
    use crate::latex::parse_latex;
    parse_latex(input)
        .map(|expr| format!("{}", expr))
        .map_err(|e| format!("LaTeX parse error: {:?}", e))
}

/// Parse LaTeX expression and return as LaTeX output.
pub(super) fn parse_latex_to_latex_ffi(input: &str) -> Result<String, String> {
    use crate::latex::parse_latex;
    parse_latex(input)
        .map(|expr| expr.to_latex())
        .map_err(|e| format!("LaTeX parse error: {:?}", e))
}

/// Get the LaTeX representation of an expression.
pub(super) fn to_latex_ffi(expression: &str) -> Result<String, String> {
    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;
    Ok(expr.to_latex())
}

/// Parse LaTeX calculus notations like \int_{a}^{b}, \lim_{x \to a}, \sum_{i=a}^{b}.
///
/// Returns the parsed expression as a string, or an error if parsing fails.
pub(super) fn parse_latex_calculus_ffi(input: &str) -> Result<String, String> {
    // Delegate to the existing LaTeX parser which handles these notations
    parse_latex_ffi(input)
}

// =============================================================================
// Expression evaluation and simplification
// =============================================================================

/// Evaluate an expression with given variable values.
///
/// Delegates directly to [`Expression::evaluate`] from the core AST, ensuring
/// full parity with the core evaluator (Log2, Log10, Cbrt, Atan2, Sign, Min,
/// Max, Pow, and correct domain handling for Ln/Log).
pub(super) fn evaluate_ffi(
    expression: &str,
    values_json: &str,
) -> Result<super::ffi::EvaluationResultFFI, String> {
    use std::collections::HashMap;

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let values: HashMap<String, f64> = serde_json::from_str(values_json)
        .map_err(|e| format!("Failed to parse values JSON: {}", e))?;

    match expr.evaluate(&values) {
        Some(value) => Ok(super::ffi::EvaluationResultFFI {
            original: expression.to_string(),
            value,
            success: true,
            error_message: String::new(),
        }),
        None => Ok(super::ffi::EvaluationResultFFI {
            original: expression.to_string(),
            value: f64::NAN,
            success: false,
            error_message:
                "Cannot evaluate expression (may contain undefined variables or operations)"
                    .to_string(),
        }),
    }
}

/// Simplify an expression.
pub(super) fn simplify_ffi(
    expression: &str,
) -> Result<super::ffi::SimplificationResultFFI, String> {
    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let simplified = expr.simplify();

    Ok(super::ffi::SimplificationResultFFI {
        original: expression.to_string(),
        simplified: format!("{}", simplified),
        simplified_latex: simplified.to_latex(),
    })
}

/// Simplify trigonometric expression.
pub(super) fn simplify_trig_ffi(
    expression: &str,
) -> Result<super::ffi::SimplificationResultFFI, String> {
    use crate::trigonometric::simplify_trig;

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let simplified = simplify_trig(&expr);

    Ok(super::ffi::SimplificationResultFFI {
        original: expression.to_string(),
        simplified: format!("{}", simplified),
        simplified_latex: simplified.to_latex(),
    })
}

/// Simplify trigonometric expression with steps.
pub(super) fn simplify_trig_with_steps_ffi(expression: &str) -> Result<String, String> {
    use crate::trigonometric::simplify_trig_with_steps;

    let expr = parse_expression(expression).map_err(|e| format!("Parse error: {:?}", e))?;

    let (simplified, steps) = simplify_trig_with_steps(&expr);

    let result = serde_json::json!({
        "original": expression,
        "simplified": format!("{}", simplified),
        "simplified_latex": simplified.to_latex(),
        "steps": steps
    });

    serde_json::to_string(&result).map_err(|e| format!("Failed to serialize result: {}", e))
}

// =============================================================================
// Advanced solving operations
// =============================================================================

/// Solve a system of linear equations.
pub(super) fn solve_system_ffi(equations_json: &str) -> Result<String, String> {
    use crate::ast::Variable;
    use crate::solver::SystemSolver;

    let equations: Vec<String> = serde_json::from_str(equations_json)
        .map_err(|e| format!("Failed to parse equations JSON: {}", e))?;

    let mut parsed_equations = Vec::new();
    for eq_str in &equations {
        let eq = parse_equation(eq_str)
            .map_err(|e| format!("Failed to parse equation '{}': {:?}", eq_str, e))?;
        parsed_equations.push(eq);
    }

    // Extract variables from equations
    let mut vars = std::collections::HashSet::new();
    for eq in &parsed_equations {
        collect_variables(&eq.left, &mut vars);
        collect_variables(&eq.right, &mut vars);
    }
    let variables: Vec<Variable> = vars.into_iter().map(|s| Variable::new(&s)).collect();

    let solver = SystemSolver::new();

    match solver.solve_system(&parsed_equations, &variables) {
        Ok(solutions) => {
            let result: std::collections::HashMap<String, String> = solutions
                .iter()
                .map(|(var, sol)| (var.name.clone(), format!("{:?}", sol)))
                .collect();
            serde_json::to_string(&result).map_err(|e| format!("Failed to serialize result: {}", e))
        }
        Err(e) => Err(format!("Failed to solve system: {:?}", e)),
    }
}

/// Helper to collect variable names from expression.
pub(super) fn collect_variables(
    expr: &crate::ast::Expression,
    vars: &mut std::collections::HashSet<String>,
) {
    use crate::ast::Expression;
    match expr {
        Expression::Variable(v) => {
            vars.insert(v.name.clone());
        }
        Expression::Unary(_, inner) => collect_variables(inner, vars),
        Expression::Binary(_, left, right) => {
            collect_variables(left, vars);
            collect_variables(right, vars);
        }
        Expression::Power(base, exp) => {
            collect_variables(base, vars);
            collect_variables(exp, vars);
        }
        Expression::Function(_, args) => {
            for arg in args {
                collect_variables(arg, vars);
            }
        }
        _ => {}
    }
}

/// Solve an inequality.
pub(super) fn solve_inequality_ffi(inequality: &str, variable: &str) -> Result<String, String> {
    use crate::inequality::{solve_inequality, Inequality};

    // Parse inequality (expects format like "expr < value" or "expr > value")
    let parts: Vec<&str> = if inequality.contains("<=") {
        vec![
            &inequality[..inequality.find("<=").unwrap()],
            &inequality[inequality.find("<=").unwrap() + 2..],
            "<=",
        ]
    } else if inequality.contains(">=") {
        vec![
            &inequality[..inequality.find(">=").unwrap()],
            &inequality[inequality.find(">=").unwrap() + 2..],
            ">=",
        ]
    } else if inequality.contains('<') {
        vec![
            &inequality[..inequality.find('<').unwrap()],
            &inequality[inequality.find('<').unwrap() + 1..],
            "<",
        ]
    } else if inequality.contains('>') {
        vec![
            &inequality[..inequality.find('>').unwrap()],
            &inequality[inequality.find('>').unwrap() + 1..],
            ">",
        ]
    } else {
        return Err("Invalid inequality format. Use <, >, <=, or >=".to_string());
    };

    let left = parse_expression(parts[0].trim())
        .map_err(|e| format!("Parse error in left side: {:?}", e))?;
    let right = parse_expression(parts[1].trim())
        .map_err(|e| format!("Parse error in right side: {:?}", e))?;

    let ineq = match parts[2] {
        "<" => Inequality::LessThan(left, right),
        ">" => Inequality::GreaterThan(left, right),
        "<=" => Inequality::LessEqual(left, right),
        ">=" => Inequality::GreaterEqual(left, right),
        _ => return Err("Invalid operator".to_string()),
    };

    match solve_inequality(&ineq, variable) {
        Ok(solution) => Ok(format!("{:?}", solution)),
        Err(e) => Err(format!("Failed to solve inequality: {:?}", e)),
    }
}

/// Partial fraction decomposition.
pub(super) fn partial_fractions_ffi(
    numerator: &str,
    denominator: &str,
    variable: &str,
) -> Result<String, String> {
    use crate::ast::Variable;
    use crate::partial_fractions::decompose;

    let num =
        parse_expression(numerator).map_err(|e| format!("Parse error in numerator: {:?}", e))?;
    let denom = parse_expression(denominator)
        .map_err(|e| format!("Parse error in denominator: {:?}", e))?;

    match decompose(&num, &denom, &Variable::new(variable)) {
        Ok(result) => {
            let expr = result.to_expression();
            let output = serde_json::json!({
                "original_numerator": numerator,
                "original_denominator": denominator,
                "decomposition": format!("{}", expr),
                "decomposition_latex": expr.to_latex(),
                "terms_count": result.terms.len(),
                "steps": result.steps
            });
            serde_json::to_string(&output).map_err(|e| format!("Failed to serialize result: {}", e))
        }
        Err(e) => Err(format!("Decomposition failed: {:?}", e)),
    }
}

/// Solve a multi-equation system.
pub(super) fn solve_equation_system_ffi(
    equations_json: &str,
    known_values_json: &str,
    targets_json: &str,
) -> Result<String, String> {
    use crate::equation_system::{EquationSystem, MultiEquationSolver, SystemContext};
    use std::collections::HashMap;

    // Parse equations JSON: {"id": "equation_str", ...}
    let equations_map: HashMap<String, String> = serde_json::from_str(equations_json)
        .map_err(|e| format!("Failed to parse equations JSON: {}", e))?;

    // Parse known values JSON: {"var": value, ...}
    let known_values: HashMap<String, f64> = serde_json::from_str(known_values_json)
        .map_err(|e| format!("Failed to parse known values JSON: {}", e))?;

    // Parse targets JSON: ["var1", "var2", ...]
    let targets: Vec<String> = serde_json::from_str(targets_json)
        .map_err(|e| format!("Failed to parse targets JSON: {}", e))?;

    // Build the equation system
    let mut system = EquationSystem::new();
    for (id, eq_str) in equations_map {
        let equation = parse_equation(&eq_str)
            .map_err(|e| format!("Failed to parse equation '{}': {:?}", id, e))?;
        system.add_equation(id, equation);
    }

    // Build the context
    let mut context = SystemContext::new();
    for (var, val) in known_values {
        context = context.with_known_value(var, val);
    }
    for target in targets {
        context = context.with_target(target);
    }

    // Solve the system
    let solver = MultiEquationSolver::new();
    let solution = solver
        .solve(&system, &context)
        .map_err(|e| format!("Failed to solve system: {}", e))?;

    // Build the result JSON
    let mut solutions_map: HashMap<String, serde_json::Value> = HashMap::new();
    for (var, val) in &solution.solutions {
        if let Some(num) = val.as_numeric() {
            solutions_map.insert(var.clone(), serde_json::json!(num));
        } else {
            solutions_map.insert(
                var.clone(),
                serde_json::json!(format!("{}", val.to_expression())),
            );
        }
    }

    // Build step descriptions
    let steps: Vec<serde_json::Value> = solution
        .resolution_path
        .steps
        .iter()
        .map(|step| {
            serde_json::json!({
                "step_number": step.step_number,
                "equation_id": step.equation_id,
                "operation": format!("{}", step.operation),
                "explanation": step.explanation
            })
        })
        .collect();

    let output = serde_json::json!({
        "solutions": solutions_map,
        "steps": steps,
        "unsolved": solution.unsolved,
        "warnings": solution.warnings,
        "is_complete": solution.is_complete()
    });

    serde_json::to_string(&output).map_err(|e| format!("Failed to serialize result: {}", e))
}
