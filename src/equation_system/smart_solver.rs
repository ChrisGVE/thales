//! Smart system solver that dispatches to the best available method.
//!
//! Classifies a system of equations as linear, small nonlinear (2×2), or
//! large nonlinear, then routes it to the appropriate solver.
//!
//! # Example
//!
//! ```
//! use thales::equation_system::SmartSystemSolver;
//! use thales::ast::{Equation, Expression, BinaryOp, Variable};
//!
//! // Solve linear 2×2: x + y = 5, x - y = 1  →  x=3, y=2
//! let x = Variable::new("x");
//! let y = Variable::new("y");
//!
//! let eq1 = Equation::new(
//!     "eq1",
//!     Expression::Binary(BinaryOp::Add,
//!         Box::new(Expression::Variable(x.clone())),
//!         Box::new(Expression::Variable(y.clone()))),
//!     Expression::Integer(5),
//! );
//! let eq2 = Equation::new(
//!     "eq2",
//!     Expression::Binary(BinaryOp::Sub,
//!         Box::new(Expression::Variable(x.clone())),
//!         Box::new(Expression::Variable(y.clone()))),
//!     Expression::Integer(1),
//! );
//!
//! let solver = SmartSystemSolver::new();
//! let result = solver.solve(&[eq1, eq2], &[x, y]).unwrap();
//! assert_eq!(result.len(), 2);
//! ```

use crate::ast::{BinaryOp, Equation, Expression, Variable};
use crate::equation_system::substitution::SubstitutionSolver;
use crate::resolution_path::{Operation, ResolutionPath, ResolutionStep};
use crate::solver::system::SystemSolution;
use crate::solver::{SolverError, SystemSolver};

/// Classification of a system of equations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemType {
    /// All equations are linear in the given variables.
    Linear,
    /// 2 equations, 2 variables, and at least one equation is nonlinear.
    NonlinearSmall,
    /// 3 or more equations, or substitution is unlikely to succeed.
    NonlinearLarge,
}

/// Dispatch solver that automatically selects the best method for a system.
#[derive(Debug, Default)]
pub struct SmartSystemSolver {
    substitution_solver: SubstitutionSolver,
}

impl SmartSystemSolver {
    /// Create a new `SmartSystemSolver`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Classify a system of equations based on linearity and size.
    pub fn classify(equations: &[Equation], variables: &[Variable]) -> SystemType {
        let all_linear = equations.iter().all(|eq| Self::is_linear_in(eq, variables));

        if all_linear {
            return SystemType::Linear;
        }

        if equations.len() == 2 && variables.len() == 2 {
            SystemType::NonlinearSmall
        } else {
            SystemType::NonlinearLarge
        }
    }

    /// Return `true` when `equation` is linear in all `variables`.
    ///
    /// An equation is linear when each variable appears only with degree 1,
    /// with no products between variables and no non-integer powers.
    pub fn is_linear_in(equation: &Equation, variables: &[Variable]) -> bool {
        let combined = Expression::Binary(
            BinaryOp::Sub,
            Box::new(equation.left.clone()),
            Box::new(equation.right.clone()),
        );
        variables
            .iter()
            .all(|v| expr_is_linear_in_var(&combined, &v.name, variables))
    }

    /// Solve a system, automatically selecting the best method.
    ///
    /// Returns a list of `(variable_name, value_expression)` pairs for each
    /// variable in the system.  For nonlinear 2×2 systems the substitution
    /// solver may return multiple solution sets; only the first set is
    /// returned here.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CannotSolve`] when the system type is not yet
    /// supported, and propagates errors from the underlying solvers otherwise.
    pub fn solve(
        &self,
        equations: &[Equation],
        variables: &[Variable],
    ) -> Result<Vec<(String, Expression)>, SolverError> {
        match Self::classify(equations, variables) {
            SystemType::Linear => self.solve_linear(equations, variables),
            SystemType::NonlinearSmall => self.solve_nonlinear_small(equations, variables),
            SystemType::NonlinearLarge => Err(SolverError::CannotSolve(
                "systems with 3+ nonlinear equations are not yet supported".to_string(),
            )),
        }
    }

    /// Solve a system and record the resolution path describing each step.
    ///
    /// Returns the same variable-value pairs as [`solve`](SmartSystemSolver::solve)
    /// together with a [`ResolutionPath`] documenting which method was chosen
    /// and which algebraic operations were performed.
    ///
    /// # Errors
    ///
    /// Same error conditions as [`solve`](SmartSystemSolver::solve).
    pub fn solve_with_path(
        &self,
        equations: &[Equation],
        variables: &[Variable],
    ) -> Result<(Vec<(String, Expression)>, ResolutionPath), SolverError> {
        let system_type = Self::classify(equations, variables);
        let initial = Expression::Integer(equations.len() as i64);
        let mut path = ResolutionPath::new(initial);

        let pairs = match system_type {
            SystemType::Linear => {
                path.add_step(ResolutionStep::new(
                    Operation::GaussianElimination,
                    "Classify as linear system; apply Gaussian elimination".to_string(),
                    Expression::Integer(0),
                ));
                let result = self.solve_linear(equations, variables)?;
                for (name, _) in &result {
                    path.add_step(ResolutionStep::new(
                        Operation::BackSubstitute {
                            variable: name.clone(),
                        },
                        format!("Back-substitute to resolve {}", name),
                        Expression::Integer(0),
                    ));
                }
                result
            }
            SystemType::NonlinearSmall => {
                path.add_step(ResolutionStep::new(
                    Operation::SystemSubstitution {
                        variable: variables
                            .first()
                            .map(|v| v.name.clone())
                            .unwrap_or_default(),
                    },
                    "Classify as nonlinear 2×2 system; apply substitution".to_string(),
                    Expression::Integer(0),
                ));
                self.solve_nonlinear_small(equations, variables)?
            }
            SystemType::NonlinearLarge => {
                return Err(SolverError::CannotSolve(
                    "systems with 3+ nonlinear equations are not yet supported".to_string(),
                ));
            }
        };

        let result_expr = Expression::Integer(pairs.len() as i64);
        path.set_result(result_expr);
        Ok((pairs, path))
    }

    // ── private dispatch helpers ──────────────────────────────────────────────

    fn solve_linear(
        &self,
        equations: &[Equation],
        variables: &[Variable],
    ) -> Result<Vec<(String, Expression)>, SolverError> {
        let solver = SystemSolver::new();
        let sol = solver.solve_best_effort(equations, variables)?;
        system_solution_to_pairs(sol, variables)
    }

    fn solve_nonlinear_small(
        &self,
        equations: &[Equation],
        variables: &[Variable],
    ) -> Result<Vec<(String, Expression)>, SolverError> {
        if equations.len() == 2 && variables.len() == 2 {
            let result = self.substitution_solver.solve(
                &equations[0],
                &equations[1],
                &variables[0],
                &variables[1],
            );

            match result {
                Ok(pairs) if !pairs.is_empty() => {
                    // Return first solution set.
                    let (v0, v1) = pairs.into_iter().next().unwrap();
                    Ok(vec![
                        (variables[0].name.clone(), v0),
                        (variables[1].name.clone(), v1),
                    ])
                }
                Ok(_) => Err(SolverError::NoSolution),
                Err(_) => Err(SolverError::CannotSolve(
                    "substitution failed for nonlinear 2×2 system".to_string(),
                )),
            }
        } else {
            Err(SolverError::CannotSolve(
                "NonlinearSmall dispatch requires exactly 2 equations and 2 variables".to_string(),
            ))
        }
    }
}

// ── free helpers ──────────────────────────────────────────────────────────────

/// Convert a `SystemSolution::Unique` into a name–expression pair list.
fn system_solution_to_pairs(
    sol: SystemSolution,
    variables: &[Variable],
) -> Result<Vec<(String, Expression)>, SolverError> {
    match sol {
        SystemSolution::Unique(map) => {
            let pairs = variables
                .iter()
                .filter_map(|v| map.get(v).map(|e| (v.name.clone(), e.clone())))
                .collect();
            Ok(pairs)
        }
        SystemSolution::Multiple(points) => {
            // Caller expects a single assignment — return the first solution
            // point. A richer API for multi-point consumers belongs at a
            // higher layer than this helper.
            let map = points.into_iter().next().ok_or(SolverError::NoSolution)?;
            let pairs = variables
                .iter()
                .filter_map(|v| map.get(v).map(|e| (v.name.clone(), e.clone())))
                .collect();
            Ok(pairs)
        }
        SystemSolution::NoSolution => Err(SolverError::NoSolution),
        SystemSolution::Infinite { .. } => Err(SolverError::InfiniteSolutions),
    }
}

/// Return `true` when `expr` is linear in `var_name`, given `all_vars`.
///
/// The expression must not contain:
/// - `var_name` raised to a power other than 1
/// - products of `var_name` with any other variable in `all_vars`
/// - `var_name` inside a function call
fn expr_is_linear_in_var(expr: &Expression, var_name: &str, all_vars: &[Variable]) -> bool {
    match expr {
        Expression::Variable(v) => true || v.name == var_name, // always linear at leaf
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Rational(_)
        | Expression::Constant(_)
        | Expression::Complex(_) => true,

        Expression::Unary(_, inner) => expr_is_linear_in_var(inner, var_name, all_vars),

        Expression::Binary(BinaryOp::Add | BinaryOp::Sub, l, r) => {
            expr_is_linear_in_var(l, var_name, all_vars)
                && expr_is_linear_in_var(r, var_name, all_vars)
        }

        Expression::Binary(BinaryOp::Mul, l, r) => {
            let l_has = l.contains_variable(var_name);
            let r_has = r.contains_variable(var_name);

            match (l_has, r_has) {
                // Neither side involves var_name — always linear.
                (false, false) => true,
                // Only one side involves var_name — linear only if the other
                // side contains no variable from `all_vars` (i.e. it is a
                // plain coefficient) and the var_name side is itself linear.
                (true, false) => {
                    let r_is_coeff = all_vars.iter().all(|v| !r.contains_variable(&v.name));
                    r_is_coeff && expr_is_linear_in_var(l, var_name, all_vars)
                }
                (false, true) => {
                    let l_is_coeff = all_vars.iter().all(|v| !l.contains_variable(&v.name));
                    l_is_coeff && expr_is_linear_in_var(r, var_name, all_vars)
                }
                // Both sides contain var_name — always nonlinear.
                (true, true) => false,
            }
        }

        Expression::Binary(BinaryOp::Div, l, r) => {
            // Division is linear only if the denominator is free of var_name.
            !r.contains_variable(var_name) && expr_is_linear_in_var(l, var_name, all_vars)
        }

        // Power: x^n with n != 1 is nonlinear; constant^x is also nonlinear.
        Expression::Power(base, exp) => {
            let base_has = base.contains_variable(var_name);
            let exp_has = exp.contains_variable(var_name);
            if exp_has {
                return false; // exponential in var_name
            }
            if base_has {
                // Linear only when exponent is the integer 1.
                matches!(exp.as_ref(), Expression::Integer(1))
            } else {
                true
            }
        }

        // Function calls containing var_name are nonlinear.
        Expression::Function(_, args) => !args.iter().any(|a| a.contains_variable(var_name)),

        _ => false,
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Equation, Expression, Variable};

    fn var(name: &str) -> Variable {
        Variable::new(name)
    }
    fn vexpr(name: &str) -> Expression {
        Expression::Variable(var(name))
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
    fn mul(a: Expression, b: Expression) -> Expression {
        Expression::Binary(BinaryOp::Mul, Box::new(a), Box::new(b))
    }

    fn linear_2x2() -> ([Equation; 2], [Variable; 2]) {
        // x + y = 5, x - y = 1  →  x=3, y=2
        let x = var("x");
        let y = var("y");
        let eq1 = Equation::new("eq1", add(vexpr("x"), vexpr("y")), int(5));
        let eq2 = Equation::new("eq2", sub(vexpr("x"), vexpr("y")), int(1));
        ([eq1, eq2], [x, y])
    }

    fn nonlinear_2x2() -> ([Equation; 2], [Variable; 2]) {
        // x * y = 6, x + y = 5  →  (2,3) or (3,2)
        let x = var("x");
        let y = var("y");
        let eq1 = Equation::new("eq1", mul(vexpr("x"), vexpr("y")), int(6));
        let eq2 = Equation::new("eq2", add(vexpr("x"), vexpr("y")), int(5));
        ([eq1, eq2], [x, y])
    }

    // ── classify ─────────────────────────────────────────────────────────────

    #[test]
    fn test_classify_linear() {
        let ([eq1, eq2], [x, y]) = linear_2x2();
        assert_eq!(
            SmartSystemSolver::classify(&[eq1, eq2], &[x, y]),
            SystemType::Linear
        );
    }

    #[test]
    fn test_classify_nonlinear_small() {
        let ([eq1, eq2], [x, y]) = nonlinear_2x2();
        assert_eq!(
            SmartSystemSolver::classify(&[eq1, eq2], &[x, y]),
            SystemType::NonlinearSmall
        );
    }

    #[test]
    fn test_classify_nonlinear_large_when_three_equations() {
        let ([eq1, eq2], [x, y]) = nonlinear_2x2();
        let z = var("z");
        let eq3 = Equation::new("eq3", vexpr("z"), int(0));
        assert_eq!(
            SmartSystemSolver::classify(&[eq1, eq2, eq3], &[x, y, z]),
            SystemType::NonlinearLarge
        );
    }

    // ── is_linear_in ─────────────────────────────────────────────────────────

    #[test]
    fn test_is_linear_true_for_linear_equation() {
        let ([eq1, _], [x, y]) = linear_2x2();
        assert!(SmartSystemSolver::is_linear_in(&eq1, &[x, y]));
    }

    #[test]
    fn test_is_linear_false_for_product_of_vars() {
        let ([eq1, _], [x, y]) = nonlinear_2x2();
        // eq1 is x*y = 6
        assert!(!SmartSystemSolver::is_linear_in(&eq1, &[x, y]));
    }

    // ── solve: linear dispatch ────────────────────────────────────────────────

    #[test]
    fn test_solve_linear_2x2_correct() {
        let ([eq1, eq2], [x, y]) = linear_2x2();
        let solver = SmartSystemSolver::new();
        let result = solver.solve(&[eq1, eq2], &[x, y]).unwrap();
        assert_eq!(result.len(), 2);
        let empty = std::collections::HashMap::<String, f64>::new();
        for (name, expr) in &result {
            let val = expr.evaluate(&empty).unwrap();
            if name == "x" {
                assert!((val - 3.0).abs() < 1e-9, "x={val}");
            } else {
                assert!((val - 2.0).abs() < 1e-9, "y={val}");
            }
        }
    }

    // ── solve: nonlinear substitution dispatch ────────────────────────────────

    #[test]
    fn test_solve_nonlinear_2x2_substitution() {
        let ([eq1, eq2], [x, y]) = nonlinear_2x2();
        let solver = SmartSystemSolver::new();
        let result = solver.solve(&[eq1, eq2], &[x, y]).unwrap();
        assert_eq!(result.len(), 2);
        let empty = std::collections::HashMap::<String, f64>::new();
        // Valid pairs: (2,3) or (3,2)
        let xv = result
            .iter()
            .find(|(n, _)| n == "x")
            .unwrap()
            .1
            .evaluate(&empty)
            .unwrap();
        let yv = result
            .iter()
            .find(|(n, _)| n == "y")
            .unwrap()
            .1
            .evaluate(&empty)
            .unwrap();
        let ok = ((xv - 2.0).abs() < 1e-9 && (yv - 3.0).abs() < 1e-9)
            || ((xv - 3.0).abs() < 1e-9 && (yv - 2.0).abs() < 1e-9);
        assert!(ok, "expected (2,3) or (3,2), got ({xv},{yv})");
    }

    // ── solve_with_path ───────────────────────────────────────────────────────

    #[test]
    fn test_solve_with_path_linear_contains_gaussian_elimination() {
        use crate::resolution_path::Operation;

        let ([eq1, eq2], [x, y]) = linear_2x2();
        let solver = SmartSystemSolver::new();
        let (_pairs, path) = solver.solve_with_path(&[eq1, eq2], &[x, y]).unwrap();

        let has_gauss = path
            .steps
            .iter()
            .any(|s| matches!(s.operation, Operation::GaussianElimination));
        assert!(
            has_gauss,
            "linear solve path must contain GaussianElimination"
        );
    }

    #[test]
    fn test_solve_with_path_linear_has_back_substitute_steps() {
        use crate::resolution_path::Operation;

        let ([eq1, eq2], [x, y]) = linear_2x2();
        let solver = SmartSystemSolver::new();
        let (_pairs, path) = solver
            .solve_with_path(&[eq1, eq2], &[x.clone(), y.clone()])
            .unwrap();

        let back_subs = path
            .steps
            .iter()
            .filter(|s| matches!(s.operation, Operation::BackSubstitute { .. }))
            .count();
        assert_eq!(back_subs, 2, "expected one BackSubstitute per variable");
    }

    #[test]
    fn test_solve_with_path_nonlinear_contains_system_substitution() {
        use crate::resolution_path::Operation;

        let ([eq1, eq2], [x, y]) = nonlinear_2x2();
        let solver = SmartSystemSolver::new();
        let (_pairs, path) = solver.solve_with_path(&[eq1, eq2], &[x, y]).unwrap();

        let has_sub = path
            .steps
            .iter()
            .any(|s| matches!(s.operation, Operation::SystemSubstitution { .. }));
        assert!(has_sub, "nonlinear path must contain SystemSubstitution");
    }

    #[test]
    fn test_solve_with_path_difficulty_is_advanced() {
        use crate::resolution_path::TechniqueDifficulty;

        let ([eq1, eq2], [x, y]) = linear_2x2();
        let solver = SmartSystemSolver::new();
        let (_pairs, path) = solver.solve_with_path(&[eq1, eq2], &[x, y]).unwrap();

        assert_eq!(path.max_difficulty(), TechniqueDifficulty::Advanced);
    }

    // ── solve: unsupported large nonlinear ────────────────────────────────────

    #[test]
    fn test_solve_nonlinear_large_returns_error() {
        let x = var("x");
        let y = var("y");
        let z = var("z");
        // x*y = 1 (nonlinear), x + z = 2, y - z = 0
        let eq1 = Equation::new("eq1", mul(vexpr("x"), vexpr("y")), int(1));
        let eq2 = Equation::new("eq2", add(vexpr("x"), vexpr("z")), int(2));
        let eq3 = Equation::new("eq3", sub(vexpr("y"), vexpr("z")), int(0));
        let solver = SmartSystemSolver::new();
        let result = solver.solve(&[eq1, eq2, eq3], &[x, y, z]);
        assert!(result.is_err());
    }
}
