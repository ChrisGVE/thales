//! Algebraic equation solver.
//!
//! Provides symbolic manipulation and solving capabilities for equations
//! with one or more unknowns.

use crate::ast::{BinaryOp, Equation, Expression, Variable};
use crate::resolution_path::{Operation, ResolutionPath, ResolutionPathBuilder, ResolutionStep};
use std::collections::HashMap;

/// Error types for equation solving.
#[derive(Debug, Clone, PartialEq)]
pub enum SolverError {
    /// Equation has no solution
    NoSolution,
    /// Equation has infinite solutions
    InfiniteSolutions,
    /// Cannot solve for the given variable
    CannotSolve(String),
    /// Equation is not linear/polynomial as expected
    UnsupportedEquationType,
    /// Division by zero encountered
    DivisionByZero,
    /// Other error with description
    Other(String),
}

/// Result type for solver operations.
pub type SolverResult<T> = Result<T, SolverError>;

/// Solution to an equation.
#[derive(Debug, Clone, PartialEq)]
pub enum Solution {
    /// Single unique solution
    Unique(Expression),
    /// Multiple discrete solutions
    Multiple(Vec<Expression>),
    /// Parametric solution with constraints
    Parametric {
        expression: Expression,
        constraints: Vec<Constraint>,
    },
    /// No solution exists
    None,
    /// Infinite solutions (identity)
    Infinite,
}

/// Constraint on a solution.
#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    pub variable: Variable,
    pub condition: Expression,
}

/// Trait for equation solvers.
pub trait Solver {
    /// Solve an equation for the specified variable.
    ///
    /// Returns the solution(s) and the resolution path showing the steps taken.
    fn solve(
        &self,
        equation: &Equation,
        variable: &Variable,
    ) -> SolverResult<(Solution, ResolutionPath)>;

    /// Check if this solver can handle the given equation.
    fn can_solve(&self, equation: &Equation) -> bool;
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if expression contains the given variable.
fn contains_variable(expr: &Expression, var: &str) -> bool {
    expr.contains_variable(var)
}

/// Extract the coefficient of a variable from an expression.
/// For example: in "3 * x", returns Some(3)
/// Returns None if variable not found or pattern doesn't match.
fn extract_coefficient(expr: &Expression, var: &str) -> Option<Expression> {
    match expr {
        // x -> coefficient is 1
        Expression::Variable(v) if v.name == var => Some(Expression::Integer(1)),

        // a * x or x * a
        Expression::Binary(BinaryOp::Mul, left, right) => {
            if let Expression::Variable(v) = left.as_ref() {
                if v.name == var && !contains_variable(right, var) {
                    return Some(right.as_ref().clone());
                }
            }
            if let Expression::Variable(v) = right.as_ref() {
                if v.name == var && !contains_variable(left, var) {
                    return Some(left.as_ref().clone());
                }
            }
            None
        }

        _ => None,
    }
}

/// Collect terms with and without the variable from an expression.
/// Returns (terms_with_var, terms_without_var)
fn collect_terms(expr: &Expression, var: &str) -> (Vec<Expression>, Vec<Expression>) {
    let mut with_var = Vec::new();
    let mut without_var = Vec::new();

    collect_terms_recursive(expr, var, &mut with_var, &mut without_var);

    (with_var, without_var)
}

fn collect_terms_recursive(
    expr: &Expression,
    var: &str,
    with_var: &mut Vec<Expression>,
    without_var: &mut Vec<Expression>,
) {
    match expr {
        Expression::Binary(BinaryOp::Add, left, right) => {
            collect_terms_recursive(left, var, with_var, without_var);
            collect_terms_recursive(right, var, with_var, without_var);
        }
        Expression::Binary(BinaryOp::Sub, left, right) => {
            collect_terms_recursive(left, var, with_var, without_var);
            // Negate the right side when collecting
            let negated = Expression::Unary(
                crate::ast::UnaryOp::Neg,
                Box::new(right.as_ref().clone()),
            );
            collect_terms_recursive(&negated, var, with_var, without_var);
        }
        _ => {
            if contains_variable(expr, var) {
                with_var.push(expr.clone());
            } else {
                without_var.push(expr.clone());
            }
        }
    }
}

/// Combine a list of expressions into a single expression with addition.
fn combine_with_add(terms: Vec<Expression>) -> Expression {
    if terms.is_empty() {
        return Expression::Integer(0);
    }

    terms.into_iter().reduce(|acc, term| {
        Expression::Binary(BinaryOp::Add, Box::new(acc), Box::new(term))
    }).unwrap()
}

/// Evaluate constant expressions to their numeric values.
/// If the expression contains only constants, evaluate it completely.
fn evaluate_constants(expr: &Expression) -> Expression {
    // First simplify
    let simplified = expr.simplify();

    // Try to evaluate if it's all constants
    if !has_any_variable(&simplified) {
        if let Some(value) = simplified.evaluate(&HashMap::new()) {
            // Check if it's an integer value
            if value.fract().abs() < 1e-10 {
                return Expression::Integer(value.round() as i64);
            } else {
                return Expression::Float(value);
            }
        }
    }

    simplified
}

/// Isolate a variable in an equation.
/// Returns the expression that the variable equals.
fn isolate_variable(
    equation: &Equation,
    var: &str,
    path: &mut ResolutionPathBuilder,
) -> Result<Expression, SolverError> {
    let left = &equation.left;
    let right = &equation.right;

    // Check if variable exists in equation
    if !contains_variable(left, var) && !contains_variable(right, var) {
        return Err(SolverError::CannotSolve(format!(
            "Variable '{}' not found in equation",
            var
        )));
    }

    // Special case: variable already isolated (x = expr or expr = x)
    if let Expression::Variable(v) = left {
        if v.name == var && !contains_variable(right, var) {
            return Ok(right.clone());
        }
    }
    if let Expression::Variable(v) = right {
        if v.name == var && !contains_variable(left, var) {
            return Ok(left.clone());
        }
    }

    // Try to solve simple patterns

    // Pattern: a * x = c  =>  x = c / a
    if let Some(coeff) = extract_coefficient(left, var) {
        if !contains_variable(right, var) {
            let result = Expression::Binary(
                BinaryOp::Div,
                Box::new(right.clone()),
                Box::new(coeff.clone()),
            ).simplify();
            let evaluated = evaluate_constants(&result);
            return Ok(evaluated);
        }
    }

    // Pattern: c = a * x  =>  x = c / a
    if let Some(coeff) = extract_coefficient(right, var) {
        if !contains_variable(left, var) {
            let result = Expression::Binary(
                BinaryOp::Div,
                Box::new(left.clone()),
                Box::new(coeff.clone()),
            ).simplify();
            let evaluated = evaluate_constants(&result);
            return Ok(evaluated);
        }
    }

    // Pattern: x + b = c  =>  x = c - b
    if let Expression::Binary(BinaryOp::Add, l, r) = left {
        if let Expression::Variable(v) = l.as_ref() {
            if v.name == var && !contains_variable(r, var) && !contains_variable(right, var) {
                let result = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(right.clone()),
                    Box::new(r.as_ref().clone()),
                ).simplify();
                let evaluated = evaluate_constants(&result);
                return Ok(evaluated);
            }
        }
        if let Expression::Variable(v) = r.as_ref() {
            if v.name == var && !contains_variable(l, var) && !contains_variable(right, var) {
                let result = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(right.clone()),
                    Box::new(l.as_ref().clone()),
                ).simplify();
                let evaluated = evaluate_constants(&result);
                return Ok(evaluated);
            }
        }
    }

    // Pattern: c = x + b  =>  x = c - b
    if let Expression::Binary(BinaryOp::Add, l, r) = right {
        if let Expression::Variable(v) = l.as_ref() {
            if v.name == var && !contains_variable(r, var) && !contains_variable(left, var) {
                let result = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(left.clone()),
                    Box::new(r.as_ref().clone()),
                ).simplify();
                let evaluated = evaluate_constants(&result);
                return Ok(evaluated);
            }
        }
        if let Expression::Variable(v) = r.as_ref() {
            if v.name == var && !contains_variable(l, var) && !contains_variable(left, var) {
                let result = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(left.clone()),
                    Box::new(l.as_ref().clone()),
                ).simplify();
                let evaluated = evaluate_constants(&result);
                return Ok(evaluated);
            }
        }
    }

    // Pattern: a * x + b = c  =>  x = (c - b) / a
    if let Expression::Binary(BinaryOp::Add, l, r) = left {
        if let Some(coeff) = extract_coefficient(l, var) {
            if !contains_variable(r, var) && !contains_variable(right, var) {
                let numerator = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(right.clone()),
                    Box::new(r.as_ref().clone()),
                );
                let result = Expression::Binary(
                    BinaryOp::Div,
                    Box::new(numerator),
                    Box::new(coeff),
                ).simplify();
                let evaluated = evaluate_constants(&result);
                return Ok(evaluated);
            }
        }
        if let Some(coeff) = extract_coefficient(r, var) {
            if !contains_variable(l, var) && !contains_variable(right, var) {
                let numerator = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(right.clone()),
                    Box::new(l.as_ref().clone()),
                );
                let result = Expression::Binary(
                    BinaryOp::Div,
                    Box::new(numerator),
                    Box::new(coeff),
                ).simplify();
                let evaluated = evaluate_constants(&result);
                return Ok(evaluated);
            }
        }
    }

    // More complex cases not yet supported
    Err(SolverError::CannotSolve(
        "Equation pattern not yet supported for Phase 1".to_string(),
    ))
}

/// Linear equation solver (ax + b = c).
#[derive(Debug, Default)]
pub struct LinearSolver;

impl LinearSolver {
    pub fn new() -> Self {
        Self
    }
}

impl Solver for LinearSolver {
    fn solve(
        &self,
        equation: &Equation,
        variable: &Variable,
    ) -> SolverResult<(Solution, ResolutionPath)> {
        let var_name = &variable.name;

        // Initialize resolution path
        let initial_expr = Expression::Binary(
            BinaryOp::Sub,
            Box::new(equation.left.clone()),
            Box::new(equation.right.clone()),
        );
        let mut path = ResolutionPathBuilder::new(initial_expr.clone());

        // Check if variable appears in equation
        let left_has_var = contains_variable(&equation.left, var_name);
        let right_has_var = contains_variable(&equation.right, var_name);

        if !left_has_var && !right_has_var {
            return Err(SolverError::CannotSolve(format!(
                "Variable '{}' not found in equation",
                var_name
            )));
        }

        // Check if equation is linear in the target variable
        if !is_linear_in_variable(&equation.left, var_name)
            || !is_linear_in_variable(&equation.right, var_name)
        {
            return Err(SolverError::UnsupportedEquationType);
        }

        // Isolate the variable
        let result_expr = isolate_variable(equation, var_name, &mut path)?;

        // Add isolation step
        path = path.step(
            Operation::Isolate(variable.clone()),
            format!("Isolate {} on one side", variable),
            result_expr.clone(),
        );

        // Build final resolution path
        let resolution_path = path.finish(result_expr.clone());

        Ok((Solution::Unique(result_expr), resolution_path))
    }

    fn can_solve(&self, equation: &Equation) -> bool {
        // Check if equation has obvious non-linear features (powers > 1 with variables)
        // We're more permissive here since we don't know the target variable yet,
        // but we can still reject clearly quadratic/polynomial equations.
        !has_obvious_nonlinearity(&equation.left) && !has_obvious_nonlinearity(&equation.right)
    }
}

/// Check if an expression has obvious non-linear features like x^2.
fn has_obvious_nonlinearity(expr: &Expression) -> bool {
    match expr {
        Expression::Power(base, exp) => {
            // x^2 or any variable raised to power > 1
            if has_any_variable(base) {
                // Check if exponent is > 1
                if let Some(exp_val) = exp.evaluate(&HashMap::new()) {
                    if exp_val > 1.0 {
                        return true;
                    }
                }
            }
            has_obvious_nonlinearity(base) || has_obvious_nonlinearity(exp)
        }
        Expression::Unary(_, inner) => has_obvious_nonlinearity(inner),
        Expression::Binary(_, left, right) => {
            has_obvious_nonlinearity(left) || has_obvious_nonlinearity(right)
        }
        Expression::Function(_, args) => args.iter().any(|arg| has_obvious_nonlinearity(arg)),
        _ => false,
    }
}

/// Check if an expression is linear (no variable powers, products, or functions).
fn is_linear_equation(expr: &Expression) -> bool {
    match expr {
        Expression::Integer(_)
        | Expression::Rational(_)
        | Expression::Float(_)
        | Expression::Complex(_)
        | Expression::Variable(_) => true,

        Expression::Unary(_, inner) => is_linear_equation(inner),

        Expression::Binary(op, left, right) => {
            let left_linear = is_linear_equation(left);
            let right_linear = is_linear_equation(right);

            match op {
                BinaryOp::Add | BinaryOp::Sub => left_linear && right_linear,
                BinaryOp::Mul | BinaryOp::Div => {
                    // For multiplication/division to be linear, at most one side can have variables
                    let left_has_var = has_any_variable(left);
                    let right_has_var = has_any_variable(right);
                    left_linear && right_linear && !(left_has_var && right_has_var)
                }
                _ => false,
            }
        }

        Expression::Power(base, exp) => {
            // Only allow constant powers, and base must not have variables
            !has_any_variable(base) && is_linear_equation(exp)
        }

        Expression::Function(_, _) => {
            // For Phase 1, we don't support functions in linear equations
            false
        }
    }
}

/// Check if expression contains any variables.
fn has_any_variable(expr: &Expression) -> bool {
    match expr {
        Expression::Variable(_) => true,
        Expression::Unary(_, inner) => has_any_variable(inner),
        Expression::Binary(_, left, right) => has_any_variable(left) || has_any_variable(right),
        Expression::Function(_, args) => args.iter().any(has_any_variable),
        Expression::Power(base, exp) => has_any_variable(base) || has_any_variable(exp),
        _ => false,
    }
}

/// Check if an expression is linear with respect to a specific variable.
/// An expression is linear in variable x if:
/// - x appears to at most power 1
/// - x does not appear in denominators
/// - x does not appear multiplied by itself
/// - x does not appear in functions
fn is_linear_in_variable(expr: &Expression, var: &str) -> bool {
    match expr {
        Expression::Integer(_)
        | Expression::Rational(_)
        | Expression::Float(_)
        | Expression::Complex(_) => true,

        Expression::Variable(v) => {
            // The target variable itself is linear
            true
        }

        Expression::Unary(_, inner) => is_linear_in_variable(inner, var),

        Expression::Binary(op, left, right) => {
            let left_has_var = contains_variable(left, var);
            let right_has_var = contains_variable(right, var);

            match op {
                BinaryOp::Add | BinaryOp::Sub => {
                    // x + y and x - y are linear if both sides are linear
                    is_linear_in_variable(left, var) && is_linear_in_variable(right, var)
                }
                BinaryOp::Mul => {
                    // For multiplication to be linear in x, at most one side can contain x
                    if left_has_var && right_has_var {
                        // x * x or x * f(x) is not linear
                        false
                    } else {
                        // a * x is linear
                        is_linear_in_variable(left, var) && is_linear_in_variable(right, var)
                    }
                }
                BinaryOp::Div => {
                    // x / a is linear, but a / x is not
                    if right_has_var {
                        false // Variable in denominator makes it non-linear
                    } else {
                        is_linear_in_variable(left, var)
                    }
                }
                _ => false,
            }
        }

        Expression::Power(base, exp) => {
            // x^2 is not linear, but a^x could be (though we don't handle that in Phase 1)
            // For Phase 1, we only allow constant powers where base doesn't have the variable
            !contains_variable(base, var) && is_linear_in_variable(exp, var)
        }

        Expression::Function(_, _) => {
            // For Phase 1, functions are not supported
            false
        }
    }
}

/// Quadratic equation solver (ax² + bx + c = 0).
#[derive(Debug, Default)]
pub struct QuadraticSolver;

impl QuadraticSolver {
    pub fn new() -> Self {
        Self
    }
}

impl Solver for QuadraticSolver {
    fn solve(
        &self,
        _equation: &Equation,
        _variable: &Variable,
    ) -> SolverResult<(Solution, ResolutionPath)> {
        // TODO: Implement quadratic formula
        // TODO: Handle complex roots
        // TODO: Handle degenerate cases (a=0, becomes linear)
        // TODO: Handle discriminant = 0 (repeated root)
        Err(SolverError::Other("Not yet implemented".to_string()))
    }

    fn can_solve(&self, _equation: &Equation) -> bool {
        // TODO: Check if equation is quadratic in the target variable
        false
    }
}

/// Polynomial equation solver (general degree n).
#[derive(Debug, Default)]
pub struct PolynomialSolver;

impl PolynomialSolver {
    pub fn new() -> Self {
        Self
    }
}

impl Solver for PolynomialSolver {
    fn solve(
        &self,
        _equation: &Equation,
        _variable: &Variable,
    ) -> SolverResult<(Solution, ResolutionPath)> {
        // TODO: Implement polynomial solving
        // TODO: Use appropriate method based on degree:
        //   - degree 1: linear
        //   - degree 2: quadratic formula
        //   - degree 3: cubic formula
        //   - degree 4: quartic formula
        //   - degree 5+: numerical methods
        Err(SolverError::Other("Not yet implemented".to_string()))
    }

    fn can_solve(&self, _equation: &Equation) -> bool {
        // TODO: Check if equation is polynomial
        false
    }
}

/// Transcendental equation solver (equations with trig, exp, log functions).
#[derive(Debug, Default)]
pub struct TranscendentalSolver;

impl TranscendentalSolver {
    pub fn new() -> Self {
        Self
    }
}

impl Solver for TranscendentalSolver {
    fn solve(
        &self,
        _equation: &Equation,
        _variable: &Variable,
    ) -> SolverResult<(Solution, ResolutionPath)> {
        // TODO: Implement solving for transcendental equations
        // TODO: Use symbolic manipulation where possible
        // TODO: Fall back to numerical methods when needed
        Err(SolverError::Other("Not yet implemented".to_string()))
    }

    fn can_solve(&self, _equation: &Equation) -> bool {
        // TODO: Check for transcendental functions
        false
    }
}

/// System of equations solver.
#[derive(Debug, Default)]
pub struct SystemSolver;

impl SystemSolver {
    pub fn new() -> Self {
        Self
    }

    /// Solve a system of equations for multiple variables.
    pub fn solve_system(
        &self,
        _equations: &[Equation],
        _variables: &[Variable],
    ) -> SolverResult<HashMap<Variable, Solution>> {
        // TODO: Implement system solving
        // TODO: Support linear systems (Gaussian elimination)
        // TODO: Support nonlinear systems (Newton-Raphson)
        // TODO: Detect under/over-determined systems
        Err(SolverError::Other("Not yet implemented".to_string()))
    }
}

/// Smart solver that dispatches to appropriate specialized solver.
#[derive(Debug)]
pub struct SmartSolver {
    linear: LinearSolver,
    quadratic: QuadraticSolver,
    polynomial: PolynomialSolver,
    transcendental: TranscendentalSolver,
}

impl SmartSolver {
    pub fn new() -> Self {
        Self {
            linear: LinearSolver::new(),
            quadratic: QuadraticSolver::new(),
            polynomial: PolynomialSolver::new(),
            transcendental: TranscendentalSolver::new(),
        }
    }
}

impl Default for SmartSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver for SmartSolver {
    fn solve(
        &self,
        equation: &Equation,
        variable: &Variable,
    ) -> SolverResult<(Solution, ResolutionPath)> {
        // TODO: Analyze equation and dispatch to appropriate solver
        // Priority order: linear -> quadratic -> polynomial -> transcendental
        if self.linear.can_solve(equation) {
            self.linear.solve(equation, variable)
        } else if self.quadratic.can_solve(equation) {
            self.quadratic.solve(equation, variable)
        } else if self.polynomial.can_solve(equation) {
            self.polynomial.solve(equation, variable)
        } else if self.transcendental.can_solve(equation) {
            self.transcendental.solve(equation, variable)
        } else {
            Err(SolverError::UnsupportedEquationType)
        }
    }

    fn can_solve(&self, equation: &Equation) -> bool {
        self.linear.can_solve(equation)
            || self.quadratic.can_solve(equation)
            || self.polynomial.can_solve(equation)
            || self.transcendental.can_solve(equation)
    }
}

// ============================================================================
// High-Level API
// ============================================================================

/// Solve an equation for a specific variable.
///
/// This is the main entry point for solving equations. It attempts to solve
/// the equation symbolically, then substitutes known values and simplifies.
///
/// # Arguments
/// * `equation` - The equation to solve
/// * `target` - The name of the variable to solve for
/// * `known_values` - HashMap of known variable values
///
/// # Returns
/// A ResolutionPath showing the solution steps and final result
///
/// # Errors
/// Returns SolverError if the equation cannot be solved
pub fn solve_for(
    equation: &Equation,
    target: &str,
    known_values: &HashMap<String, f64>,
) -> Result<ResolutionPath, SolverError> {
    // Create Variable from target string
    let target_var = Variable::new(target);

    // Try solving with SmartSolver
    let solver = SmartSolver::new();
    let (solution, mut path) = solver.solve(equation, &target_var)?;

    // Extract the solution expression
    let solution_expr = match solution {
        Solution::Unique(expr) => expr,
        Solution::Multiple(_) => {
            return Err(SolverError::Other(
                "Multiple solutions not yet supported in solve_for".to_string(),
            ))
        }
        Solution::None => return Err(SolverError::NoSolution),
        Solution::Infinite => return Err(SolverError::InfiniteSolutions),
        Solution::Parametric { .. } => {
            return Err(SolverError::Other(
                "Parametric solutions not yet supported in solve_for".to_string(),
            ))
        }
    };

    // Substitute known values
    if !known_values.is_empty() {
        let substituted = substitute_values(&solution_expr, known_values);
        let simplified = substituted.simplify();
        let evaluated = evaluate_constants(&simplified);

        path.add_step(ResolutionStep::new(
            Operation::Substitute {
                variable: Variable::new("known_values"),
                value: Expression::Integer(0), // Placeholder
            },
            "Substitute known values and evaluate".to_string(),
            evaluated.clone(),
        ));

        path.set_result(evaluated);
    } else {
        path.set_result(solution_expr);
    }

    Ok(path)
}

/// Substitute known variable values into an expression.
fn substitute_values(expr: &Expression, values: &HashMap<String, f64>) -> Expression {
    match expr {
        Expression::Variable(v) => {
            if let Some(&value) = values.get(&v.name) {
                Expression::Float(value)
            } else {
                expr.clone()
            }
        }
        Expression::Unary(op, inner) => {
            Expression::Unary(*op, Box::new(substitute_values(inner, values)))
        }
        Expression::Binary(op, left, right) => Expression::Binary(
            *op,
            Box::new(substitute_values(left, values)),
            Box::new(substitute_values(right, values)),
        ),
        Expression::Function(func, args) => Expression::Function(
            func.clone(),
            args.iter().map(|arg| substitute_values(arg, values)).collect(),
        ),
        Expression::Power(base, exp) => Expression::Power(
            Box::new(substitute_values(base, values)),
            Box::new(substitute_values(exp, values)),
        ),
        _ => expr.clone(),
    }
}

// TODO: Add equation simplification before solving
// TODO: Add symbolic manipulation utilities
// TODO: Add support for inequalities
// TODO: Add support for absolute value equations
// TODO: Add support for piecewise functions
// TODO: Add step-by-step explanation generation
