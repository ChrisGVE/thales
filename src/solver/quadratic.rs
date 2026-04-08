//! Quadratic equation solver for equations of the form ax² + bx + c = 0.

use crate::ast::{BinaryOp, Equation, Expression, Variable};
use crate::resolution_path::{Operation, ResolutionPath, ResolutionPathBuilder, StepAnnotation};

use super::helpers::{
    contains_variable, extract_quadratic_coefficients, has_obvious_nonlinearity,
    simplify_numeric_expression,
};
use super::types::{Solution, SolverError, SolverResult};
use super::Solver;

/// Quadratic equation solver for equations of the form ax² + bx + c = 0.
///
/// Solves second-degree polynomial equations in one variable using the quadratic
/// formula and returns either zero, one, or two real solutions, or two complex solutions.
///
/// # Mathematical Foundation
///
/// A quadratic equation has the general form:
/// ```text
/// ax² + bx + c = 0    where a ≠ 0
/// ```
///
/// The solution is obtained using the quadratic formula:
/// ```text
/// x = (-b ± √(b² - 4ac)) / (2a)
/// ```
///
/// The discriminant Δ = b² - 4ac determines the nature of the roots:
/// - Δ > 0: Two distinct real roots
/// - Δ = 0: One repeated real root (multiplicity 2)
/// - Δ < 0: Two complex conjugate roots
///
/// # See Also
///
/// - [`super::LinearSolver`]: For degenerate case when a = 0
/// - [`super::PolynomialSolver`]: General polynomial solver (uses QuadraticSolver for degree 2)
/// - [`super::SmartSolver`]: Automatically selects QuadraticSolver for quadratic equations
#[derive(Debug, Default)]
pub struct QuadraticSolver;

impl QuadraticSolver {
    /// Create a new quadratic equation solver.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::solver::QuadraticSolver;
    ///
    /// let solver = QuadraticSolver::new();
    /// ```
    pub fn new() -> Self {
        Self
    }
}

impl Solver for QuadraticSolver {
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
        if !contains_variable(&equation.left, var_name)
            && !contains_variable(&equation.right, var_name)
        {
            return Err(SolverError::CannotSolve(format!(
                "Variable '{}' not found in equation",
                var_name
            )));
        }

        // Extract coefficients a, b, c from ax² + bx + c = 0
        // Move everything to left side: left - right = 0
        let combined = Expression::Binary(
            BinaryOp::Sub,
            Box::new(equation.left.clone()),
            Box::new(equation.right.clone()),
        )
        .simplify();

        // Extract polynomial coefficients
        let (a, b, c) = extract_quadratic_coefficients(&combined, var_name);

        // Add step showing coefficients
        path = path.step(
            Operation::Simplify,
            format!("Identified coefficients: a={}, b={}, c={}", a, b, c),
            combined.clone(),
        );

        // Check for degenerate case (a = 0)
        if a.abs() < 1e-15 {
            if b.abs() < 1e-15 {
                if c.abs() < 1e-15 {
                    let resolution_path = path.finish(Expression::Integer(0));
                    return Ok((Solution::Infinite, resolution_path));
                } else {
                    return Err(SolverError::NoSolution);
                }
            }
            // Linear equation: bx + c = 0 -> x = -c/b
            let solution = Expression::Float(-c / b);
            let resolution_path = path.finish(solution.clone());
            return Ok((Solution::Unique(solution), resolution_path));
        }

        // Compute discriminant Δ = b² - 4ac
        let discriminant = b * b - 4.0 * a * c;

        path = path.annotated_step(
            Operation::Simplify,
            format!("Computed discriminant: Δ = b² - 4ac = {}", discriminant),
            Expression::Float(discriminant),
            StepAnnotation::elementary(),
        );

        let epsilon = 1e-15;
        if discriminant > epsilon {
            // Two distinct real roots
            let sqrt_disc = discriminant.sqrt();
            let x1 = (-b + sqrt_disc) / (2.0 * a);
            let x2 = (-b - sqrt_disc) / (2.0 * a);

            let root1 = simplify_numeric_expression(x1);
            let root2 = simplify_numeric_expression(x2);

            path = path.annotated_step(
                Operation::Simplify,
                format!("Quadratic formula: x = (-b ± √Δ)/(2a) = {} or {}", x1, x2),
                root1.clone(),
                StepAnnotation::algebraic("Quadratic Formula"),
            );

            let resolution_path = path.finish(root1.clone());
            Ok((Solution::Multiple(vec![root1, root2]), resolution_path))
        } else if discriminant.abs() <= epsilon {
            // One repeated real root
            let x = -b / (2.0 * a);
            let root = simplify_numeric_expression(x);

            path = path.annotated_step(
                Operation::Simplify,
                format!("Quadratic formula (Δ = 0): x = -b/(2a) = {}", x),
                root.clone(),
                StepAnnotation::algebraic("Quadratic Formula"),
            );

            let resolution_path = path.finish(root.clone());
            Ok((Solution::Unique(root), resolution_path))
        } else {
            // Complex roots: x = -b/(2a) ± i√(-Δ)/(2a)
            let real_part = -b / (2.0 * a);
            let imag_part = (-discriminant).sqrt() / (2.0 * a);

            let root1 = Expression::Complex(num_complex::Complex64::new(real_part, imag_part));
            let root2 = Expression::Complex(num_complex::Complex64::new(real_part, -imag_part));

            path = path.annotated_step(
                Operation::Simplify,
                format!("Complex roots: x = {} ± {}i", real_part, imag_part),
                root1.clone(),
                StepAnnotation::algebraic("Quadratic Formula"),
            );

            let resolution_path = path.finish(root1.clone());
            Ok((Solution::Multiple(vec![root1, root2]), resolution_path))
        }
    }

    fn can_solve(&self, equation: &Equation) -> bool {
        // Check if equation has quadratic terms
        has_obvious_nonlinearity(&equation.left) || has_obvious_nonlinearity(&equation.right)
    }
}
