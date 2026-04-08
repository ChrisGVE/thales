//! Transcendental equation solver for equations with trig, exp, and log functions.

use crate::ast::{BinaryOp, Equation, Expression, Variable};
use crate::resolution_path::{Operation, ResolutionPath, ResolutionStep, StepAnnotation};
use std::collections::HashMap;

use super::helpers::{
    contains_variable, evaluate_constants, extract_coefficient, has_any_variable,
};
use super::types::{Solution, SolverError, SolverResult};
use super::Solver;

/// Transcendental equation solver for equations with trig, exp, and log functions.
///
/// Solves equations involving transcendental functions - functions that cannot be
/// expressed in terms of algebraic operations alone. This includes trigonometric,
/// exponential, and logarithmic functions.
///
/// # Supported Equation Types
///
/// ## Trigonometric Equations
///
/// - `sin(x) = a` → `x = asin(a)` (requires |a| ≤ 1)
/// - `cos(x) = a` → `x = acos(a)` (requires |a| ≤ 1)
/// - `tan(x) = a` → `x = atan(a)` (no domain restriction)
/// - `c * sin(x) = b` → `x = asin(b/c)`
/// - `sin(c*x) = a` → `x = asin(a)/c`
///
/// ## Logarithmic Equations
///
/// - `ln(x) = a` → `x = exp(a)`
/// - `log10(x) = a` → `x = 10^a`
/// - `log(x, b) = a` → `x = b^a`
/// - `c * ln(x) = a` → `x = exp(a/c)`
///
/// ## Exponential Equations
///
/// - `exp(x) = a` → `x = ln(a)`
/// - `a^x = b` → `x = ln(b)/ln(a)` (change of base formula)
/// - `exp(c*x) = a` → `x = ln(a)/c`
/// - `c * exp(x) = a` → `x = ln(a/c)`
///
/// # Examples
///
/// ## Solving sin(x) = 0.5
///
/// ```
/// use thales::solver::{TranscendentalSolver, Solver};
/// use thales::ast::{Equation, Expression, Variable, Function};
///
/// let x = Expression::Variable(Variable::new("x"));
/// let sin_x = Expression::Function(Function::Sin, vec![x]);
/// let half = Expression::Float(0.5);
/// let equation = Equation::new("trig", sin_x, half);
///
/// let solver = TranscendentalSolver::new();
/// let (solution, path) = solver.solve(&equation, &Variable::new("x")).unwrap();
///
/// # use thales::solver::Solution;
/// # use std::collections::HashMap;
/// # match solution {
/// #     Solution::Unique(expr) => {
/// #         let result = expr.evaluate(&HashMap::new()).unwrap();
/// #         assert!((result - 0.5235987755982989).abs() < 1e-10);
/// #     }
/// #     _ => panic!("Expected unique solution"),
/// # }
/// ```
///
/// ## Solving ln(x) = 2
///
/// ```
/// use thales::solver::{TranscendentalSolver, Solver};
/// use thales::ast::{Equation, Expression, Variable, Function};
///
/// let x = Expression::Variable(Variable::new("x"));
/// let ln_x = Expression::Function(Function::Ln, vec![x]);
/// let two = Expression::Integer(2);
/// let equation = Equation::new("log", ln_x, two);
///
/// let solver = TranscendentalSolver::new();
/// let (solution, path) = solver.solve(&equation, &Variable::new("x")).unwrap();
///
/// # use thales::solver::Solution;
/// # use std::collections::HashMap;
/// # match solution {
/// #     Solution::Unique(expr) => {
/// #         let result = expr.evaluate(&HashMap::new()).unwrap();
/// #         assert!((result - std::f64::consts::E.powi(2)).abs() < 1e-10);
/// #     }
/// #     _ => panic!("Expected unique solution"),
/// # }
/// ```
///
/// ## Solving 2^x = 8
///
/// ```
/// use thales::solver::{TranscendentalSolver, Solver};
/// use thales::ast::{Equation, Expression, Variable};
///
/// let x = Expression::Variable(Variable::new("x"));
/// let two_pow_x = Expression::Power(
///     Box::new(Expression::Integer(2)),
///     Box::new(x),
/// );
/// let eight = Expression::Integer(8);
/// let equation = Equation::new("exp", two_pow_x, eight);
///
/// let solver = TranscendentalSolver::new();
/// let (solution, path) = solver.solve(&equation, &Variable::new("x")).unwrap();
///
/// # use thales::solver::Solution;
/// # use std::collections::HashMap;
/// # match solution {
/// #     Solution::Unique(expr) => {
/// #         let result = expr.evaluate(&HashMap::new()).unwrap();
/// #         assert!((result - 3.0).abs() < 1e-10);
/// #     }
/// #     _ => panic!("Expected unique solution"),
/// # }
/// ```
///
/// # See Also
///
/// - [`super::LinearSolver`]: For linear equations (ax + b = c)
/// - [`super::SmartSolver`]: Automatically selects TranscendentalSolver for transcendental equations
/// - [`super::solve_for`]: High-level API that handles solver selection and value substitution
#[derive(Debug, Default)]
pub struct TranscendentalSolver;

impl TranscendentalSolver {
    /// Creates a new transcendental equation solver.
    pub fn new() -> Self {
        Self
    }

    /// Try to solve a trigonometric equation for the target variable.
    fn solve_trig_equation(
        &self,
        equation: &Equation,
        variable: &Variable,
        path: &mut ResolutionPath,
    ) -> Option<Expression> {
        let var_name = &variable.name;

        // Pattern: sin(x) = a  →  x = asin(a)
        if let Some((result, func, value)) = self.match_trig_pattern_with_validation(
            &equation.left,
            &equation.right,
            var_name,
            crate::ast::Function::Sin,
            crate::ast::Function::Asin,
        ) {
            if let Err(_e) = Self::validate_trig_domain(value, &func) {
                return None;
            }
            path.add_step(ResolutionStep::with_annotation(
                Operation::ApplyFunction("asin".to_string()),
                format!("Apply arcsine to solve sin({}) = value", variable),
                result.clone(),
                StepAnnotation::technique("Inverse Trigonometric Function"),
            ));
            return Some(result);
        }

        // Pattern: a = sin(x)  →  x = asin(a)
        if let Some((result, func, value)) = self.match_trig_pattern_with_validation(
            &equation.right,
            &equation.left,
            var_name,
            crate::ast::Function::Sin,
            crate::ast::Function::Asin,
        ) {
            if let Err(_e) = Self::validate_trig_domain(value, &func) {
                return None;
            }
            path.add_step(ResolutionStep::with_annotation(
                Operation::ApplyFunction("asin".to_string()),
                format!("Apply arcsine to solve sin({}) = value", variable),
                result.clone(),
                StepAnnotation::technique("Inverse Trigonometric Function"),
            ));
            return Some(result);
        }

        // Pattern: cos(x) = a  →  x = acos(a)
        if let Some((result, func, value)) = self.match_trig_pattern_with_validation(
            &equation.left,
            &equation.right,
            var_name,
            crate::ast::Function::Cos,
            crate::ast::Function::Acos,
        ) {
            if let Err(_e) = Self::validate_trig_domain(value, &func) {
                return None;
            }
            path.add_step(ResolutionStep::with_annotation(
                Operation::ApplyFunction("acos".to_string()),
                format!("Apply arccosine to solve cos({}) = value", variable),
                result.clone(),
                StepAnnotation::technique("Inverse Trigonometric Function"),
            ));
            return Some(result);
        }

        // Pattern: a = cos(x)  →  x = acos(a)
        if let Some((result, func, value)) = self.match_trig_pattern_with_validation(
            &equation.right,
            &equation.left,
            var_name,
            crate::ast::Function::Cos,
            crate::ast::Function::Acos,
        ) {
            if let Err(_e) = Self::validate_trig_domain(value, &func) {
                return None;
            }
            path.add_step(ResolutionStep::with_annotation(
                Operation::ApplyFunction("acos".to_string()),
                format!("Apply arccosine to solve cos({}) = value", variable),
                result.clone(),
                StepAnnotation::technique("Inverse Trigonometric Function"),
            ));
            return Some(result);
        }

        // Pattern: tan(x) = a  →  x = atan(a)
        if let Some(result) = self.match_trig_pattern(
            &equation.left,
            &equation.right,
            var_name,
            crate::ast::Function::Tan,
            crate::ast::Function::Atan,
        ) {
            path.add_step(ResolutionStep::with_annotation(
                Operation::ApplyFunction("atan".to_string()),
                format!("Apply arctangent to solve tan({}) = value", variable),
                result.clone(),
                StepAnnotation::technique("Inverse Trigonometric Function"),
            ));
            return Some(result);
        }

        // Pattern: a = tan(x)  →  x = atan(a)
        if let Some(result) = self.match_trig_pattern(
            &equation.right,
            &equation.left,
            var_name,
            crate::ast::Function::Tan,
            crate::ast::Function::Atan,
        ) {
            path.add_step(ResolutionStep::with_annotation(
                Operation::ApplyFunction("atan".to_string()),
                format!("Apply arctangent to solve tan({}) = value", variable),
                result.clone(),
                StepAnnotation::technique("Inverse Trigonometric Function"),
            ));
            return Some(result);
        }

        None
    }

    /// Match pattern with validation: returns (result, inverse_func, input_value)
    fn match_trig_pattern_with_validation(
        &self,
        left: &Expression,
        right: &Expression,
        var: &str,
        func: crate::ast::Function,
        inverse_func: crate::ast::Function,
    ) -> Option<(Expression, crate::ast::Function, f64)> {
        // Check if right side contains the variable
        if contains_variable(right, var) {
            return None;
        }

        // Try to evaluate the right side as a constant
        let value = match right.evaluate(&HashMap::new()) {
            Some(v) => v,
            None => return None, // Can't validate if not a constant
        };

        // Pattern 1: func(x) = a  →  x = inverse_func(a)
        if let Expression::Function(f, args) = left {
            if *f == func && args.len() == 1 {
                // Check if arg is exactly the variable
                if let Expression::Variable(v) = &args[0] {
                    if v.name == var {
                        let result =
                            Expression::Function(inverse_func.clone(), vec![right.clone()]);
                        return Some((result.simplify(), inverse_func, value));
                    }
                }

                // Check if arg is a linear expression like a*x
                if let Some(coeff) = extract_coefficient(&args[0], var) {
                    // func(a*x) = b  →  a*x = inverse_func(b)  →  x = inverse_func(b) / a
                    let inverse_applied =
                        Expression::Function(inverse_func.clone(), vec![right.clone()]);
                    let result = Expression::Binary(
                        BinaryOp::Div,
                        Box::new(inverse_applied),
                        Box::new(coeff),
                    );
                    return Some((result.simplify(), inverse_func, value));
                }
            }
        }

        // Pattern 2: a * func(x) = b  →  func(x) = b/a  →  x = inverse_func(b/a)
        if let Expression::Binary(BinaryOp::Mul, mul_left, mul_right) = left {
            // Check left side of multiplication
            if let Expression::Function(f, args) = mul_left.as_ref() {
                if *f == func && args.len() == 1 && !contains_variable(mul_right, var) {
                    if let Expression::Variable(v) = &args[0] {
                        if v.name == var {
                            // a * func(x) = b  →  func(x) = b/a  →  x = inverse_func(b/a)
                            let divided = Expression::Binary(
                                BinaryOp::Div,
                                Box::new(right.clone()),
                                Box::new(mul_right.as_ref().clone()),
                            );
                            // Need to evaluate the divided value
                            let divided_value = divided.evaluate(&HashMap::new()).unwrap_or(value);
                            let result = Expression::Function(inverse_func.clone(), vec![divided]);
                            return Some((result.simplify(), inverse_func, divided_value));
                        }
                    }
                }
            }

            // Check right side of multiplication
            if let Expression::Function(f, args) = mul_right.as_ref() {
                if *f == func && args.len() == 1 && !contains_variable(mul_left, var) {
                    if let Expression::Variable(v) = &args[0] {
                        if v.name == var {
                            // func(x) * a = b  →  func(x) = b/a  →  x = inverse_func(b/a)
                            let divided = Expression::Binary(
                                BinaryOp::Div,
                                Box::new(right.clone()),
                                Box::new(mul_left.as_ref().clone()),
                            );
                            let divided_value = divided.evaluate(&HashMap::new()).unwrap_or(value);
                            let result = Expression::Function(inverse_func.clone(), vec![divided]);
                            return Some((result.simplify(), inverse_func, divided_value));
                        }
                    }
                }
            }
        }

        None
    }

    /// Match pattern: func(var) = value or coeff * func(var) = value
    fn match_trig_pattern(
        &self,
        left: &Expression,
        right: &Expression,
        var: &str,
        func: crate::ast::Function,
        inverse_func: crate::ast::Function,
    ) -> Option<Expression> {
        // Check if right side contains the variable
        if contains_variable(right, var) {
            return None;
        }

        // Pattern 1: func(x) = a  →  x = inverse_func(a)
        if let Expression::Function(f, args) = left {
            if *f == func && args.len() == 1 {
                // Check if arg is exactly the variable
                if let Expression::Variable(v) = &args[0] {
                    if v.name == var {
                        let result = Expression::Function(inverse_func, vec![right.clone()]);
                        return Some(result.simplify());
                    }
                }

                // Check if arg is a linear expression like a*x
                if let Some(coeff) = extract_coefficient(&args[0], var) {
                    // func(a*x) = b  →  a*x = inverse_func(b)  →  x = inverse_func(b) / a
                    let inverse_applied = Expression::Function(inverse_func, vec![right.clone()]);
                    let result = Expression::Binary(
                        BinaryOp::Div,
                        Box::new(inverse_applied),
                        Box::new(coeff),
                    );
                    return Some(result.simplify());
                }
            }
        }

        // Pattern 2: a * func(x) = b  →  func(x) = b/a  →  x = inverse_func(b/a)
        if let Expression::Binary(BinaryOp::Mul, mul_left, mul_right) = left {
            // Check left side of multiplication
            if let Expression::Function(f, args) = mul_left.as_ref() {
                if *f == func && args.len() == 1 && !contains_variable(mul_right, var) {
                    if let Expression::Variable(v) = &args[0] {
                        if v.name == var {
                            let divided = Expression::Binary(
                                BinaryOp::Div,
                                Box::new(right.clone()),
                                Box::new(mul_right.as_ref().clone()),
                            );
                            let result = Expression::Function(inverse_func, vec![divided]);
                            return Some(result.simplify());
                        }
                    }
                }
            }

            // Check right side of multiplication
            if let Expression::Function(f, args) = mul_right.as_ref() {
                if *f == func && args.len() == 1 && !contains_variable(mul_left, var) {
                    if let Expression::Variable(v) = &args[0] {
                        if v.name == var {
                            let divided = Expression::Binary(
                                BinaryOp::Div,
                                Box::new(right.clone()),
                                Box::new(mul_left.as_ref().clone()),
                            );
                            let result = Expression::Function(inverse_func, vec![divided]);
                            return Some(result.simplify());
                        }
                    }
                }
            }
        }

        None
    }

    /// Try to solve a logarithmic equation for the target variable.
    fn solve_log_equation(
        &self,
        equation: &Equation,
        variable: &Variable,
        path: &mut ResolutionPath,
    ) -> Option<Expression> {
        let var_name = &variable.name;

        // Pattern: ln(x) = a  →  x = exp(a)
        if let Some(result) = self.match_log_pattern(&equation.left, &equation.right, var_name) {
            path.add_step(ResolutionStep::with_annotation(
                Operation::ApplyFunction("exp".to_string()),
                format!("Apply exponential to solve ln({}) = value", variable),
                result.clone(),
                StepAnnotation::technique("Natural Logarithm"),
            ));
            return Some(result);
        }

        // Pattern: a = ln(x)  →  x = exp(a)
        if let Some(result) = self.match_log_pattern(&equation.right, &equation.left, var_name) {
            path.add_step(ResolutionStep::with_annotation(
                Operation::ApplyFunction("exp".to_string()),
                format!("Apply exponential to solve ln({}) = value", variable),
                result.clone(),
                StepAnnotation::technique("Natural Logarithm"),
            ));
            return Some(result);
        }

        // Pattern: log10(x) = a  →  x = 10^a
        if let Some(result) = self.match_log10_pattern(&equation.left, &equation.right, var_name) {
            path.add_step(ResolutionStep::with_annotation(
                Operation::PowerBothSides(Expression::Integer(10)),
                format!("Apply 10^x to solve log10({}) = value", variable),
                result.clone(),
                StepAnnotation::technique("Logarithm"),
            ));
            return Some(result);
        }

        // Pattern: a = log10(x)  →  x = 10^a
        if let Some(result) = self.match_log10_pattern(&equation.right, &equation.left, var_name) {
            path.add_step(ResolutionStep::with_annotation(
                Operation::PowerBothSides(Expression::Integer(10)),
                format!("Apply 10^x to solve log10({}) = value", variable),
                result.clone(),
                StepAnnotation::technique("Logarithm"),
            ));
            return Some(result);
        }

        // Pattern: log(x, b) = a  →  x = b^a
        if let Some(result) = self.match_log_base_pattern(&equation.left, &equation.right, var_name)
        {
            path.add_step(ResolutionStep::with_annotation(
                Operation::ApplyLogProperty("exponential form".to_string()),
                format!(
                    "Convert logarithm to exponential form to solve for {}",
                    variable
                ),
                result.clone(),
                StepAnnotation::technique("Logarithm"),
            ));
            return Some(result);
        }

        // Pattern: a = log(x, b)  →  x = b^a
        if let Some(result) = self.match_log_base_pattern(&equation.right, &equation.left, var_name)
        {
            path.add_step(ResolutionStep::with_annotation(
                Operation::ApplyLogProperty("exponential form".to_string()),
                format!(
                    "Convert logarithm to exponential form to solve for {}",
                    variable
                ),
                result.clone(),
                StepAnnotation::technique("Logarithm"),
            ));
            return Some(result);
        }

        None
    }

    /// Match pattern: ln(var) = value or coeff * ln(var) = value
    fn match_log_pattern(
        &self,
        left: &Expression,
        right: &Expression,
        var: &str,
    ) -> Option<Expression> {
        // Check if right side contains the variable
        if contains_variable(right, var) {
            return None;
        }

        // Pattern 1: ln(x) = a  →  x = exp(a)
        if let Expression::Function(crate::ast::Function::Ln, args) = left {
            if args.len() == 1 {
                if let Expression::Variable(v) = &args[0] {
                    if v.name == var {
                        let result =
                            Expression::Function(crate::ast::Function::Exp, vec![right.clone()]);
                        return Some(result.simplify());
                    }
                }
            }
        }

        // Pattern 2: a * ln(x) = b  →  ln(x) = b/a  →  x = exp(b/a)
        if let Expression::Binary(BinaryOp::Mul, mul_left, mul_right) = left {
            if let Expression::Function(crate::ast::Function::Ln, args) = mul_left.as_ref() {
                if args.len() == 1 && !contains_variable(mul_right, var) {
                    if let Expression::Variable(v) = &args[0] {
                        if v.name == var {
                            let divided = Expression::Binary(
                                BinaryOp::Div,
                                Box::new(right.clone()),
                                Box::new(mul_right.as_ref().clone()),
                            );
                            let result =
                                Expression::Function(crate::ast::Function::Exp, vec![divided]);
                            return Some(result.simplify());
                        }
                    }
                }
            }

            if let Expression::Function(crate::ast::Function::Ln, args) = mul_right.as_ref() {
                if args.len() == 1 && !contains_variable(mul_left, var) {
                    if let Expression::Variable(v) = &args[0] {
                        if v.name == var {
                            let divided = Expression::Binary(
                                BinaryOp::Div,
                                Box::new(right.clone()),
                                Box::new(mul_left.as_ref().clone()),
                            );
                            let result =
                                Expression::Function(crate::ast::Function::Exp, vec![divided]);
                            return Some(result.simplify());
                        }
                    }
                }
            }
        }

        None
    }

    /// Match pattern: log10(var) = value
    fn match_log10_pattern(
        &self,
        left: &Expression,
        right: &Expression,
        var: &str,
    ) -> Option<Expression> {
        if contains_variable(right, var) {
            return None;
        }

        // Pattern: log10(x) = a  →  x = 10^a
        if let Expression::Function(crate::ast::Function::Log10, args) = left {
            if args.len() == 1 {
                if let Expression::Variable(v) = &args[0] {
                    if v.name == var {
                        let result = Expression::Power(
                            Box::new(Expression::Integer(10)),
                            Box::new(right.clone()),
                        );
                        return Some(result.simplify());
                    }
                }
            }
        }

        None
    }

    /// Match pattern: log(var, base) = value
    fn match_log_base_pattern(
        &self,
        left: &Expression,
        right: &Expression,
        var: &str,
    ) -> Option<Expression> {
        if contains_variable(right, var) {
            return None;
        }

        // Pattern: log(x, b) = a  →  x = b^a
        if let Expression::Function(crate::ast::Function::Log, args) = left {
            if args.len() == 2 {
                if let Expression::Variable(v) = &args[0] {
                    if v.name == var && !contains_variable(&args[1], var) {
                        let result =
                            Expression::Power(Box::new(args[1].clone()), Box::new(right.clone()));
                        return Some(result.simplify());
                    }
                }
            }
        }

        None
    }

    /// Try to solve an exponential equation for the target variable.
    fn solve_exp_equation(
        &self,
        equation: &Equation,
        variable: &Variable,
        path: &mut ResolutionPath,
    ) -> Option<Expression> {
        let var_name = &variable.name;

        // Pattern: exp(x) = a  →  x = ln(a)
        if let Some(result) = self.match_exp_pattern(&equation.left, &equation.right, var_name) {
            path.add_step(ResolutionStep::with_annotation(
                Operation::ApplyFunction("ln".to_string()),
                format!("Apply natural logarithm to solve exp({}) = value", variable),
                result.clone(),
                StepAnnotation::technique("Natural Logarithm"),
            ));
            return Some(result);
        }

        // Pattern: a = exp(x)  →  x = ln(a)
        if let Some(result) = self.match_exp_pattern(&equation.right, &equation.left, var_name) {
            path.add_step(ResolutionStep::with_annotation(
                Operation::ApplyFunction("ln".to_string()),
                format!("Apply natural logarithm to solve exp({}) = value", variable),
                result.clone(),
                StepAnnotation::technique("Natural Logarithm"),
            ));
            return Some(result);
        }

        // Pattern: a^x = b  →  x = ln(b) / ln(a)
        if let Some(result) = self.match_power_pattern(&equation.left, &equation.right, var_name) {
            path.add_step(ResolutionStep::with_annotation(
                Operation::ApplyLogProperty("change of base".to_string()),
                format!("Apply logarithm to solve for {} in exponent", variable),
                result.clone(),
                StepAnnotation::technique("Natural Logarithm"),
            ));
            return Some(result);
        }

        // Pattern: b = a^x  →  x = ln(b) / ln(a)
        if let Some(result) = self.match_power_pattern(&equation.right, &equation.left, var_name) {
            path.add_step(ResolutionStep::with_annotation(
                Operation::ApplyLogProperty("change of base".to_string()),
                format!("Apply logarithm to solve for {} in exponent", variable),
                result.clone(),
                StepAnnotation::technique("Natural Logarithm"),
            ));
            return Some(result);
        }

        None
    }

    /// Match pattern: exp(var) = value or coeff * exp(var) = value
    fn match_exp_pattern(
        &self,
        left: &Expression,
        right: &Expression,
        var: &str,
    ) -> Option<Expression> {
        if contains_variable(right, var) {
            return None;
        }

        // Pattern 1: exp(x) = a  →  x = ln(a)
        if let Expression::Function(crate::ast::Function::Exp, args) = left {
            if args.len() == 1 {
                if let Expression::Variable(v) = &args[0] {
                    if v.name == var {
                        let result =
                            Expression::Function(crate::ast::Function::Ln, vec![right.clone()]);
                        return Some(result.simplify());
                    }
                }

                // Pattern: exp(a*x) = b  →  a*x = ln(b)  →  x = ln(b)/a
                if let Some(coeff) = extract_coefficient(&args[0], var) {
                    let ln_applied =
                        Expression::Function(crate::ast::Function::Ln, vec![right.clone()]);
                    let result =
                        Expression::Binary(BinaryOp::Div, Box::new(ln_applied), Box::new(coeff));
                    return Some(result.simplify());
                }
            }
        }

        // Pattern 2: a * exp(x) = b  →  exp(x) = b/a  →  x = ln(b/a)
        if let Expression::Binary(BinaryOp::Mul, mul_left, mul_right) = left {
            if let Expression::Function(crate::ast::Function::Exp, args) = mul_left.as_ref() {
                if args.len() == 1 && !contains_variable(mul_right, var) {
                    if let Expression::Variable(v) = &args[0] {
                        if v.name == var {
                            let divided = Expression::Binary(
                                BinaryOp::Div,
                                Box::new(right.clone()),
                                Box::new(mul_right.as_ref().clone()),
                            );
                            let result =
                                Expression::Function(crate::ast::Function::Ln, vec![divided]);
                            return Some(result.simplify());
                        }
                    }
                }
            }

            if let Expression::Function(crate::ast::Function::Exp, args) = mul_right.as_ref() {
                if args.len() == 1 && !contains_variable(mul_left, var) {
                    if let Expression::Variable(v) = &args[0] {
                        if v.name == var {
                            let divided = Expression::Binary(
                                BinaryOp::Div,
                                Box::new(right.clone()),
                                Box::new(mul_left.as_ref().clone()),
                            );
                            let result =
                                Expression::Function(crate::ast::Function::Ln, vec![divided]);
                            return Some(result.simplify());
                        }
                    }
                }
            }
        }

        None
    }

    /// Match pattern: base^var = value
    fn match_power_pattern(
        &self,
        left: &Expression,
        right: &Expression,
        var: &str,
    ) -> Option<Expression> {
        if contains_variable(right, var) {
            return None;
        }

        // Pattern: a^x = b  →  x = ln(b) / ln(a)
        if let Expression::Power(base, exp) = left {
            if !contains_variable(base, var) && contains_variable(exp, var) {
                // Simple case: a^x = b
                if let Expression::Variable(v) = exp.as_ref() {
                    if v.name == var {
                        let ln_right =
                            Expression::Function(crate::ast::Function::Ln, vec![right.clone()]);
                        let ln_base = Expression::Function(
                            crate::ast::Function::Ln,
                            vec![base.as_ref().clone()],
                        );
                        let result = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(ln_right),
                            Box::new(ln_base),
                        );
                        return Some(result.simplify());
                    }
                }

                // Pattern: a^(b*x) = c  →  b*x = ln(c)/ln(a)  →  x = ln(c)/(b*ln(a))
                if let Some(coeff) = extract_coefficient(exp, var) {
                    let ln_right =
                        Expression::Function(crate::ast::Function::Ln, vec![right.clone()]);
                    let ln_base =
                        Expression::Function(crate::ast::Function::Ln, vec![base.as_ref().clone()]);
                    let divided =
                        Expression::Binary(BinaryOp::Div, Box::new(ln_right), Box::new(ln_base));
                    let result =
                        Expression::Binary(BinaryOp::Div, Box::new(divided), Box::new(coeff));
                    return Some(result.simplify());
                }
            }
        }

        None
    }

    /// Check if an expression contains transcendental functions.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::solver::TranscendentalSolver;
    /// use thales::ast::{Expression, Variable, Function};
    ///
    /// // sin(x) contains transcendental function
    /// let x = Expression::Variable(Variable::new("x"));
    /// let sin_x = Expression::Function(Function::Sin, vec![x.clone()]);
    /// // Note: has_transcendental_function is private, tested via can_solve
    ///
    /// // x^2 does not contain transcendental function (algebraic)
    /// let x_squared = Expression::Power(
    ///     Box::new(x.clone()),
    ///     Box::new(Expression::Integer(2)),
    /// );
    ///
    /// // 2^x contains transcendental function (variable in exponent)
    /// let two_pow_x = Expression::Power(
    ///     Box::new(Expression::Integer(2)),
    ///     Box::new(x.clone()),
    /// );
    /// ```
    fn has_transcendental_function(expr: &Expression) -> bool {
        match expr {
            Expression::Function(func, _) => {
                matches!(
                    func,
                    crate::ast::Function::Sin
                        | crate::ast::Function::Cos
                        | crate::ast::Function::Tan
                        | crate::ast::Function::Asin
                        | crate::ast::Function::Acos
                        | crate::ast::Function::Atan
                        | crate::ast::Function::Sinh
                        | crate::ast::Function::Cosh
                        | crate::ast::Function::Tanh
                        | crate::ast::Function::Exp
                        | crate::ast::Function::Ln
                        | crate::ast::Function::Log
                        | crate::ast::Function::Log2
                        | crate::ast::Function::Log10
                )
            }
            Expression::Unary(_, inner) => Self::has_transcendental_function(inner),
            Expression::Binary(_, left, right) => {
                Self::has_transcendental_function(left) || Self::has_transcendental_function(right)
            }
            Expression::Power(base, exp) => {
                // Check if variable appears in exponent (exponential form)
                has_any_variable(exp)
                    || Self::has_transcendental_function(base)
                    || Self::has_transcendental_function(exp)
            }
            _ => false,
        }
    }

    /// Validate domain restrictions for inverse trigonometric functions.
    fn validate_trig_domain(value: f64, func: &crate::ast::Function) -> Result<(), SolverError> {
        match func {
            crate::ast::Function::Asin | crate::ast::Function::Acos => {
                if value.abs() > 1.0 {
                    return Err(SolverError::Other(format!(
                        "Domain error: {:?} requires |value| ≤ 1, got {}",
                        func, value
                    )));
                }
            }
            _ => {}
        }
        Ok(())
    }
}

impl Solver for TranscendentalSolver {
    fn solve(
        &self,
        equation: &Equation,
        variable: &Variable,
    ) -> SolverResult<(Solution, ResolutionPath)> {
        let var_name = &variable.name;

        // Check if variable appears in equation
        let left_has_var = contains_variable(&equation.left, var_name);
        let right_has_var = contains_variable(&equation.right, var_name);

        if !left_has_var && !right_has_var {
            return Err(SolverError::CannotSolve(format!(
                "Variable '{}' not found in equation",
                var_name
            )));
        }

        // Initialize resolution path
        let initial_expr = Expression::Binary(
            BinaryOp::Sub,
            Box::new(equation.left.clone()),
            Box::new(equation.right.clone()),
        );
        let mut path = ResolutionPath::new(initial_expr);

        // Try trigonometric equation patterns
        if let Some(result) = self.solve_trig_equation(equation, variable, &mut path) {
            // Validate domain if result is a constant
            if let Expression::Function(func, args) = &result {
                if args.len() == 1 {
                    if let Some(val) = args[0].evaluate(&HashMap::new()) {
                        Self::validate_trig_domain(val, func)?;
                    }
                }
            }

            let evaluated = evaluate_constants(&result);
            path.set_result(evaluated.clone());
            return Ok((Solution::Unique(evaluated), path));
        }

        // Try logarithmic equation patterns
        if let Some(result) = self.solve_log_equation(equation, variable, &mut path) {
            let evaluated = evaluate_constants(&result);
            path.set_result(evaluated.clone());
            return Ok((Solution::Unique(evaluated), path));
        }

        // Try exponential equation patterns
        if let Some(result) = self.solve_exp_equation(equation, variable, &mut path) {
            let evaluated = evaluate_constants(&result);
            path.set_result(evaluated.clone());
            return Ok((Solution::Unique(evaluated), path));
        }

        // If no pattern matched, cannot solve
        Err(SolverError::CannotSolve(
            "Transcendental equation pattern not recognized or too complex".to_string(),
        ))
    }

    fn can_solve(&self, equation: &Equation) -> bool {
        // Check if equation contains transcendental functions
        Self::has_transcendental_function(&equation.left)
            || Self::has_transcendental_function(&equation.right)
    }
}
