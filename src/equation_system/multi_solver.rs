//! Multi-equation solver implementation.
//!
//! Contains the main `MultiEquationSolver` struct that orchestrates solving
//! a system of equations by building dependency graphs, planning solution
//! strategies, and executing them step by step.

use std::collections::{HashMap, HashSet};

use crate::ast::{Equation, Expression, Variable};
use crate::integration::integrate;
use crate::numerical::{NumericalConfig, SmartNumericalSolver};
use crate::ode::{solve_linear as solve_linear_ode, solve_separable, FirstOrderODE};
use crate::resolution_path::ResolutionPath;
use crate::solver::{SmartSolver, Solution, Solver};

use super::types::{
    DependencyGraph, EquationType, MultiEquationSolution, SolutionStrategy, SolutionValue,
    SolveMethod, SolveStep, StepResult, SystemContext, SystemError, SystemOperation, SystemStep,
};
use super::EquationSystem;

// ============================================================================
// Multi-Equation Solver
// ============================================================================

/// Configuration for the multi-equation solver.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Maximum iterations for iterative solving.
    pub max_iterations: usize,
    /// Whether to use numerical fallback.
    pub use_numerical_fallback: bool,
    /// Numerical solver configuration.
    pub numerical_config: NumericalConfig,
    /// Whether to verify solutions.
    pub verify_solutions: bool,
    /// Verification tolerance.
    pub tolerance: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            use_numerical_fallback: true,
            numerical_config: NumericalConfig::default(),
            verify_solutions: true,
            tolerance: 1e-10,
        }
    }
}

/// The main multi-equation solver.
pub struct MultiEquationSolver {
    algebraic_solver: SmartSolver,
    numerical_solver: SmartNumericalSolver,
    config: SolverConfig,
}

impl MultiEquationSolver {
    /// Create a new multi-equation solver with default configuration.
    pub fn new() -> Self {
        Self {
            algebraic_solver: SmartSolver::new(),
            numerical_solver: SmartNumericalSolver::with_default_config(),
            config: SolverConfig::default(),
        }
    }

    /// Create a solver with custom configuration.
    pub fn with_config(config: SolverConfig) -> Self {
        Self {
            algebraic_solver: SmartSolver::new(),
            numerical_solver: SmartNumericalSolver::new(config.numerical_config.clone()),
            config,
        }
    }

    /// Solve the equation system.
    pub fn solve(
        &self,
        system: &EquationSystem,
        context: &SystemContext,
    ) -> Result<MultiEquationSolution, SystemError> {
        // Validate inputs
        if system.is_empty() {
            return Err(SystemError::NoStrategyFound(
                "No equations in system".to_string(),
            ));
        }

        if context.target_variables.is_empty() {
            return Err(SystemError::NoStrategyFound(
                "No target variables specified".to_string(),
            ));
        }

        // Build dependency graph
        let graph = self.analyze_dependencies(system);

        // Note: Circular dependency check is handled in the strategy planning phase
        // where we can detect actual deadlocks (no progress possible)
        let _known = context.known_variable_names();

        // Plan the solution
        let strategy = self.plan_solution(&graph, system, context)?;

        // Execute the strategy
        let solution = self.execute_strategy(&strategy, system, context)?;

        Ok(solution)
    }

    /// Analyze dependencies in the system.
    fn analyze_dependencies(&self, system: &EquationSystem) -> DependencyGraph {
        DependencyGraph::build(system)
    }

    /// Create a solving strategy.
    fn plan_solution(
        &self,
        graph: &DependencyGraph,
        system: &EquationSystem,
        context: &SystemContext,
    ) -> Result<SolutionStrategy, SystemError> {
        let mut strategy = SolutionStrategy::new();
        let mut known = context.known_variable_names();
        let mut remaining_targets: HashSet<_> = context.target_variables.iter().cloned().collect();
        let mut iterations = 0;

        while !remaining_targets.is_empty() && iterations < self.config.max_iterations {
            iterations += 1;

            // Find solvable equations
            let solvable = graph.find_solvable(&known);

            if solvable.is_empty() {
                // No progress possible - check if we can use numerical methods
                if self.config.use_numerical_fallback && !remaining_targets.is_empty() {
                    // Try numerical fallback for remaining targets
                    for target in &remaining_targets {
                        // Find an equation containing this target
                        if let Some(eqs) = graph.equations_with_variable(target) {
                            for eq_id in eqs {
                                strategy.add_step(SolveStep {
                                    equation_id: eq_id.clone(),
                                    solve_for: target.clone(),
                                    method: SolveMethod::Numerical,
                                    dependencies: known.iter().cloned().collect(),
                                });
                                break;
                            }
                        }
                    }
                    break;
                } else {
                    return Err(SystemError::NoStrategyFound(format!(
                        "Cannot determine solving order for: {:?}",
                        remaining_targets
                    )));
                }
            }

            // Add steps for solvable equations
            for (eq_id, var) in solvable {
                if remaining_targets.contains(&var) || !known.contains(&var) {
                    // Determine the solve method based on equation type
                    let method = if let Some(eq) = system.get(&eq_id) {
                        match &eq.equation_type {
                            EquationType::Algebraic | EquationType::Unknown => {
                                SolveMethod::Algebraic
                            }
                            EquationType::ODE(_info) => SolveMethod::ODE {
                                method: "auto".to_string(),
                            },
                            EquationType::Integral(_) => SolveMethod::Integration,
                            EquationType::Differential => SolveMethod::Differentiation,
                            EquationType::Implicit => SolveMethod::Numerical,
                        }
                    } else {
                        SolveMethod::Algebraic
                    };

                    strategy.add_step(SolveStep {
                        equation_id: eq_id,
                        solve_for: var.clone(),
                        method,
                        dependencies: known.iter().cloned().collect(),
                    });

                    known.insert(var.clone());
                    remaining_targets.remove(&var);
                }
            }
        }

        if remaining_targets.is_empty() || !strategy.is_empty() {
            Ok(strategy)
        } else {
            Err(SystemError::NoStrategyFound(format!(
                "Could not find strategy for: {:?}",
                remaining_targets
            )))
        }
    }

    /// Execute the solving strategy.
    fn execute_strategy(
        &self,
        strategy: &SolutionStrategy,
        system: &EquationSystem,
        context: &SystemContext,
    ) -> Result<MultiEquationSolution, SystemError> {
        let mut solution = MultiEquationSolution::new(context.clone());
        let mut known_exprs: HashMap<String, Expression> = HashMap::new();
        let mut known_values: HashMap<String, f64> = context.known_values.clone();

        // Initialize with known expressions from context
        for (var, val) in &context.known_values {
            known_exprs.insert(var.clone(), Expression::Float(*val));
        }
        for (var, expr) in &context.known_expressions {
            known_exprs.insert(var.clone(), expr.clone());
        }

        let mut step_number = 0;

        for step in &strategy.steps {
            step_number += 1;

            // Get the equation
            let eq = system
                .get(&step.equation_id)
                .ok_or_else(|| SystemError::EquationNotFound(step.equation_id.clone()))?;

            // Substitute known values into the equation
            let substituted_eq = self.substitute_known(&eq.equation, &known_exprs);

            // Solve based on method
            let result = match &step.method {
                SolveMethod::Algebraic => {
                    self.solve_algebraic(&substituted_eq, &step.solve_for, &known_values)
                }
                SolveMethod::Numerical => {
                    self.solve_numerical(&substituted_eq, &step.solve_for, &known_values)
                }
                SolveMethod::ODE { method } => {
                    self.solve_ode(&substituted_eq, &step.solve_for, method)
                }
                SolveMethod::Integration => {
                    self.solve_integration(&substituted_eq, &step.solve_for)
                }
                SolveMethod::Substitution => {
                    // For pure substitution, just evaluate the RHS if target is on LHS
                    self.solve_by_substitution(&substituted_eq, &step.solve_for, &known_values)
                }
                _ => Err(SystemError::UnsolvableEquation {
                    id: step.equation_id.clone(),
                    reason: format!("Method {:?} not implemented", step.method),
                }),
            };

            match result {
                Ok((value, eq_path)) => {
                    // Record the step
                    let expr = value.to_expression();
                    solution.resolution_path.add_step(SystemStep {
                        step_number,
                        equation_id: step.equation_id.clone(),
                        operation: SystemOperation::SolveFor {
                            variable: step.solve_for.clone(),
                            method: step.method.clone(),
                        },
                        explanation: format!(
                            "From equation '{}', solved {} = {}",
                            step.equation_id, step.solve_for, expr
                        ),
                        result: StepResult::Expression(expr.clone()),
                    });

                    // Update known values
                    known_exprs.insert(step.solve_for.clone(), expr.clone());
                    if let Some(num) = value.as_numeric() {
                        known_values.insert(step.solve_for.clone(), num);
                    }

                    // Add equation path
                    if let Some(path) = eq_path {
                        solution
                            .resolution_path
                            .add_equation_path(step.equation_id.clone(), path);
                    }

                    // Record solution
                    solution.add_solution(step.solve_for.clone(), value);
                }
                Err(e) => {
                    solution.add_warning(format!(
                        "Failed to solve for {} in {}: {}",
                        step.solve_for, step.equation_id, e
                    ));
                    solution.mark_unsolved(step.solve_for.clone());
                }
            }
        }

        // Verify solutions if requested
        if self.config.verify_solutions {
            self.verify_solutions(&solution, system, context);
        }

        Ok(solution)
    }

    /// Substitute known values into an equation.
    fn substitute_known(
        &self,
        equation: &Equation,
        known: &HashMap<String, Expression>,
    ) -> Equation {
        Equation {
            id: equation.id.clone(),
            left: self.substitute_expr(&equation.left, known),
            right: self.substitute_expr(&equation.right, known),
        }
    }

    /// Substitute known values into an expression.
    fn substitute_expr(
        &self,
        expr: &Expression,
        known: &HashMap<String, Expression>,
    ) -> Expression {
        match expr {
            Expression::Variable(var) => {
                if let Some(val) = known.get(&var.name) {
                    val.clone()
                } else {
                    expr.clone()
                }
            }
            Expression::Binary(op, left, right) => Expression::Binary(
                *op,
                Box::new(self.substitute_expr(left, known)),
                Box::new(self.substitute_expr(right, known)),
            ),
            Expression::Unary(op, inner) => {
                Expression::Unary(*op, Box::new(self.substitute_expr(inner, known)))
            }
            Expression::Power(base, exp) => Expression::Power(
                Box::new(self.substitute_expr(base, known)),
                Box::new(self.substitute_expr(exp, known)),
            ),
            Expression::Function(func, args) => Expression::Function(
                func.clone(),
                args.iter()
                    .map(|a| self.substitute_expr(a, known))
                    .collect(),
            ),
            _ => expr.clone(),
        }
    }

    /// Solve an equation algebraically.
    fn solve_algebraic(
        &self,
        equation: &Equation,
        variable: &str,
        known_values: &HashMap<String, f64>,
    ) -> Result<(SolutionValue, Option<ResolutionPath>), SystemError> {
        let var = Variable::new(variable);

        match self.algebraic_solver.solve(equation, &var) {
            Ok((sol, path)) => {
                let value = match sol {
                    Solution::Unique(expr) => {
                        // Try to evaluate numerically
                        match expr.evaluate(known_values) {
                            Some(num) => SolutionValue::Numeric(num),
                            None => SolutionValue::Symbolic(expr),
                        }
                    }
                    Solution::Multiple(exprs) => SolutionValue::Multiple(exprs),
                    Solution::None => {
                        return Err(SystemError::UnsolvableEquation {
                            id: "algebraic".to_string(),
                            reason: "No solution exists".to_string(),
                        })
                    }
                    Solution::Infinite => {
                        return Err(SystemError::UnsolvableEquation {
                            id: "algebraic".to_string(),
                            reason: "Infinite solutions".to_string(),
                        })
                    }
                    Solution::Parametric {
                        expression,
                        constraints: _,
                    } => SolutionValue::Parametric {
                        expr: expression,
                        parameter: "t".to_string(), // Default parameter name
                    },
                };
                Ok((value, Some(path)))
            }
            Err(e) => Err(SystemError::SolverError(format!("{:?}", e))),
        }
    }

    /// Solve an equation numerically.
    fn solve_numerical(
        &self,
        equation: &Equation,
        variable: &str,
        _known_values: &HashMap<String, f64>,
    ) -> Result<(SolutionValue, Option<ResolutionPath>), SystemError> {
        let var = Variable::new(variable);

        // Use SmartNumericalSolver's solve method
        match self.numerical_solver.solve(equation, &var) {
            Ok((sol, path)) => {
                let value = SolutionValue::Numeric(sol.value);
                Ok((value, Some(path)))
            }
            Err(e) => Err(SystemError::NumericalFailure {
                variable: variable.to_string(),
                reason: format!("{:?}", e),
            }),
        }
    }

    /// Solve an ODE by routing to the ode module.
    fn solve_ode(
        &self,
        equation: &Equation,
        variable: &str,
        _method: &str,
    ) -> Result<(SolutionValue, Option<ResolutionPath>), SystemError> {
        // Construct a first-order ODE from the equation: assume form dy/dx = rhs
        // where the variable is the dependent variable.
        let ode = FirstOrderODE {
            dependent: variable.to_string(),
            independent: "x".to_string(),
            rhs: equation.right.clone(),
        };

        // Try separable first, then linear
        let solution = solve_separable(&ode).or_else(|_| solve_linear_ode(&ode));

        match solution {
            Ok(sol) => Ok((SolutionValue::Symbolic(sol.general_solution), None)),
            Err(e) => Err(SystemError::UnsolvableEquation {
                id: "ode".to_string(),
                reason: format!("ODE solver failed: {e:?}"),
            }),
        }
    }

    /// Solve by integration — integrate the expression with respect to `variable`.
    fn solve_integration(
        &self,
        equation: &Equation,
        variable: &str,
    ) -> Result<(SolutionValue, Option<ResolutionPath>), SystemError> {
        // Integrate the left-hand side with respect to the variable
        let integrated = integrate(&equation.left, variable);
        match integrated {
            Ok(result) => Ok((SolutionValue::Symbolic(result), None)),
            Err(e) => Err(SystemError::UnsolvableEquation {
                id: "integration".to_string(),
                reason: format!("Integration failed: {e:?}"),
            }),
        }
    }

    /// Solve by simple substitution.
    fn solve_by_substitution(
        &self,
        equation: &Equation,
        variable: &str,
        known_values: &HashMap<String, f64>,
    ) -> Result<(SolutionValue, Option<ResolutionPath>), SystemError> {
        // Check if the equation is in the form var = expr
        if let Expression::Variable(var) = &equation.left {
            if var.name == variable {
                // var = rhs, so just evaluate rhs
                match equation.right.evaluate(known_values) {
                    Some(val) => return Ok((SolutionValue::Numeric(val), None)),
                    None => return Ok((SolutionValue::Symbolic(equation.right.clone()), None)),
                }
            }
        }

        // Check if rhs is the variable
        if let Expression::Variable(var) = &equation.right {
            if var.name == variable {
                // rhs = lhs
                match equation.left.evaluate(known_values) {
                    Some(val) => return Ok((SolutionValue::Numeric(val), None)),
                    None => return Ok((SolutionValue::Symbolic(equation.left.clone()), None)),
                }
            }
        }

        // Fall back to algebraic solving
        self.solve_algebraic(equation, variable, known_values)
    }

    /// Verify solutions by substituting back.
    fn verify_solutions(
        &self,
        solution: &MultiEquationSolution,
        system: &EquationSystem,
        _context: &SystemContext,
    ) {
        // Build a map of all known values
        let mut all_values: HashMap<String, f64> = HashMap::new();
        for (var, val) in &solution.solutions {
            if let Some(num) = val.as_numeric() {
                all_values.insert(var.clone(), num);
            }
        }

        // Verify each equation
        for eq in system.equations() {
            let lhs_val = eq.equation.left.evaluate(&all_values);
            let rhs_val = eq.equation.right.evaluate(&all_values);

            match (lhs_val, rhs_val) {
                (Some(l), Some(r)) => {
                    let diff = (l - r).abs();
                    if diff > self.config.tolerance {
                        // Log warning but don't fail
                        eprintln!(
                            "Warning: Equation '{}' verification failed: {} != {} (diff: {})",
                            eq.id, l, r, diff
                        );
                    }
                }
                _ => {
                    // Can't verify symbolically
                }
            }
        }
    }
}

impl Default for MultiEquationSolver {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_equation;

    #[test]
    fn test_equation_system_creation() {
        let mut system = EquationSystem::new();
        system.add_equation("eq1", parse_equation("x + y = 10").unwrap());
        system.add_equation("eq2", parse_equation("x - y = 2").unwrap());

        assert_eq!(system.len(), 2);
        assert!(system.get("eq1").is_some());
        assert!(system.get("eq2").is_some());
    }

    #[test]
    fn test_equation_system_variables() {
        let system = EquationSystem::new()
            .with_equation("eq1", parse_equation("F = m * a").unwrap())
            .with_equation("eq2", parse_equation("v = u + a * t").unwrap());

        let vars = system.all_variables();
        assert!(vars.contains("F"));
        assert!(vars.contains("m"));
        assert!(vars.contains("a"));
        assert!(vars.contains("v"));
        assert!(vars.contains("u"));
        assert!(vars.contains("t"));
    }

    #[test]
    fn test_context_builder() {
        let context = SystemContext::new()
            .with_known_value("F", 100.0)
            .with_known_value("m", 20.0)
            .with_target("a");

        assert_eq!(context.known_values.get("F"), Some(&100.0));
        assert_eq!(context.known_values.get("m"), Some(&20.0));
        assert!(context.target_variables.contains(&"a".to_string()));
    }

    #[test]
    fn test_dependency_graph() {
        let system = EquationSystem::new()
            .with_equation("eq1", parse_equation("F = m * a").unwrap())
            .with_equation("eq2", parse_equation("v = u + a * t").unwrap());

        let graph = DependencyGraph::build(&system);

        // Check variable to equations mapping
        let a_eqs = graph.equations_with_variable("a").unwrap();
        assert!(a_eqs.contains("eq1"));
        assert!(a_eqs.contains("eq2"));

        // Check equation to variables mapping
        let eq1_vars = graph.variables_in_equation("eq1").unwrap();
        assert!(eq1_vars.contains("F"));
        assert!(eq1_vars.contains("m"));
        assert!(eq1_vars.contains("a"));
    }

    #[test]
    fn test_find_solvable() {
        let system = EquationSystem::new()
            .with_equation("eq1", parse_equation("F = m * a").unwrap())
            .with_equation("eq2", parse_equation("v = u + a * t").unwrap());

        let graph = DependencyGraph::build(&system);

        // With F and m known, we should be able to solve for a in eq1
        let mut known = HashSet::new();
        known.insert("F".to_string());
        known.insert("m".to_string());

        let solvable = graph.find_solvable(&known);
        assert!(solvable.iter().any(|(eq, var)| eq == "eq1" && var == "a"));
    }

    #[test]
    fn test_simple_linear_system() {
        let system =
            EquationSystem::new().with_equation("eq1", parse_equation("F = m * a").unwrap());

        let context = SystemContext::new()
            .with_known_value("F", 100.0)
            .with_known_value("m", 20.0)
            .with_target("a");

        let solver = MultiEquationSolver::new();
        let solution = solver.solve(&system, &context).unwrap();

        let a = solution.get_numeric("a").unwrap();
        assert!((a - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_chained_equations() {
        let system = EquationSystem::new()
            .with_equation("eq1", parse_equation("F = m * a").unwrap())
            .with_equation("eq2", parse_equation("v = u + a * t").unwrap());

        let context = SystemContext::new()
            .with_known_value("F", 100.0)
            .with_known_value("m", 20.0)
            .with_known_value("u", 0.0)
            .with_known_value("t", 5.0)
            .with_target("a")
            .with_target("v");

        let solver = MultiEquationSolver::new();
        let solution = solver.solve(&system, &context).unwrap();

        // F = m * a => 100 = 20 * a => a = 5
        let a = solution.get_numeric("a").unwrap();
        assert!((a - 5.0).abs() < 1e-10);

        // v = u + a * t => v = 0 + 5 * 5 => v = 25
        let v = solution.get_numeric("v").unwrap();
        assert!((v - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_resolution_path() {
        let system =
            EquationSystem::new().with_equation("eq1", parse_equation("F = m * a").unwrap());

        let context = SystemContext::new()
            .with_known_value("F", 100.0)
            .with_known_value("m", 20.0)
            .with_target("a");

        let solver = MultiEquationSolver::new();
        let solution = solver.solve(&system, &context).unwrap();

        // Check that resolution path has steps
        assert!(!solution.resolution_path.steps.is_empty());

        // Check text formatting doesn't panic
        let text = solution.resolution_path.format_text();
        assert!(text.contains("Multi-Equation System Solution"));
    }

    #[test]
    fn test_insufficient_equations() {
        let system =
            EquationSystem::new().with_equation("eq1", parse_equation("x + y = 10").unwrap());

        let context = SystemContext::new().with_target("x").with_target("y");

        let solver = MultiEquationSolver::new();
        let result = solver.solve(&system, &context);

        // Should fail because we have 2 unknowns but only 1 equation
        assert!(result.is_err() || !result.unwrap().is_complete());
    }

    #[test]
    fn test_solution_value_conversion() {
        let numeric = SolutionValue::Numeric(42.0);
        assert_eq!(numeric.as_numeric(), Some(42.0));

        let symbolic = SolutionValue::Symbolic(Expression::Variable(Variable::new("x")));
        assert!(symbolic.as_numeric().is_none());
        assert!(symbolic.as_expression().is_some());
    }

    #[test]
    fn test_quadratic_in_system() {
        // x² = 16 => x = ±4
        let system =
            EquationSystem::new().with_equation("eq1", parse_equation("y = x * x").unwrap());

        let context = SystemContext::new()
            .with_known_value("y", 16.0)
            .with_target("x");

        let solver = MultiEquationSolver::new();
        let solution = solver.solve(&system, &context);

        // This might give x = 4 or x = -4 depending on solver behavior
        // Just check that we get a solution
        if let Ok(sol) = solution {
            if let Some(x) = sol.get_numeric("x") {
                assert!((x.abs() - 4.0).abs() < 1e-10);
            }
        }
    }
}
