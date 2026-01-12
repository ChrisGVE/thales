//! First-Order Ordinary Differential Equation Solver
//!
//! This module provides functionality for solving first-order ODEs including:
//! - Separable equations: dy/dx = g(x) * h(y)
//! - First-order linear equations: dy/dx + P(x)*y = Q(x)
//! - Initial value problems (IVP)
//!
//! # Examples
//!
//! ```rust
//! use thales::ode::{FirstOrderODE, solve_separable, solve_linear};
//! use thales::ast::{Expression, Variable};
//!
//! // Create an ODE: dy/dx = x*y (separable)
//! let x = Expression::Variable(Variable::new("x"));
//! let y = Expression::Variable(Variable::new("y"));
//! let rhs = Expression::Binary(
//!     thales::ast::BinaryOp::Mul,
//!     Box::new(x),
//!     Box::new(y),
//! );
//! let ode = FirstOrderODE::new("y", "x", rhs);
//! ```

use crate::ast::{BinaryOp, Expression, Function, UnaryOp, Variable};
use crate::integration::{integrate, IntegrationError};

/// Error types for ODE solving
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ODEError {
    /// The equation is not in the expected form
    NotInExpectedForm(String),
    /// Cannot solve this type of ODE
    CannotSolve(String),
    /// Integration failed during solving
    IntegrationFailed(IntegrationError),
    /// Initial condition cannot be applied
    InitialConditionError(String),
    /// The ODE is not separable
    NotSeparable,
    /// The ODE is not linear
    NotLinear,
    /// Characteristic equation solving failed
    CharacteristicEquationError(String),
    /// Coefficients are not constant (depend on independent variable)
    NonConstantCoefficients(String),
    /// Boundary value problem error
    BoundaryValueError(String),
    /// Resonance detected in particular solution
    ResonanceDetected(String),
}

impl From<IntegrationError> for ODEError {
    fn from(e: IntegrationError) -> Self {
        ODEError::IntegrationFailed(e)
    }
}

impl std::fmt::Display for ODEError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ODEError::NotInExpectedForm(msg) => write!(f, "ODE not in expected form: {}", msg),
            ODEError::CannotSolve(msg) => write!(f, "Cannot solve ODE: {}", msg),
            ODEError::IntegrationFailed(e) => write!(f, "Integration failed: {}", e),
            ODEError::InitialConditionError(msg) => {
                write!(f, "Initial condition error: {}", msg)
            }
            ODEError::NotSeparable => write!(f, "ODE is not separable"),
            ODEError::NotLinear => write!(f, "ODE is not first-order linear"),
            ODEError::CharacteristicEquationError(msg) => {
                write!(f, "Characteristic equation error: {}", msg)
            }
            ODEError::NonConstantCoefficients(msg) => {
                write!(f, "Non-constant coefficients: {}", msg)
            }
            ODEError::BoundaryValueError(msg) => write!(f, "Boundary value error: {}", msg),
            ODEError::ResonanceDetected(msg) => write!(f, "Resonance detected: {}", msg),
        }
    }
}

impl std::error::Error for ODEError {}

/// Represents a first-order ordinary differential equation: dy/dx = f(x, y)
#[derive(Debug, Clone)]
pub struct FirstOrderODE {
    /// The dependent variable (e.g., "y")
    pub dependent: String,
    /// The independent variable (e.g., "x")
    pub independent: String,
    /// The right-hand side expression f(x, y) where dy/dx = f(x, y)
    pub rhs: Expression,
}

impl FirstOrderODE {
    /// Create a new first-order ODE.
    ///
    /// # Arguments
    ///
    /// * `dependent` - The dependent variable name (e.g., "y")
    /// * `independent` - The independent variable name (e.g., "x")
    /// * `rhs` - The expression f(x, y) such that dy/dx = f(x, y)
    pub fn new(dependent: &str, independent: &str, rhs: Expression) -> Self {
        Self {
            dependent: dependent.to_string(),
            independent: independent.to_string(),
            rhs,
        }
    }

    /// Check if this ODE is separable (can be written as g(x) * h(y)).
    pub fn is_separable(&self) -> bool {
        try_separate(&self.rhs, &self.independent, &self.dependent).is_some()
    }

    /// Check if this ODE is first-order linear (dy/dx + P(x)*y = Q(x)).
    pub fn is_linear(&self) -> bool {
        extract_linear_coefficients(&self.rhs, &self.independent, &self.dependent).is_some()
    }
}

/// Result of solving an ODE
#[derive(Debug, Clone)]
pub struct ODESolution {
    /// The general solution expression (may contain constant C)
    pub general_solution: Expression,
    /// Description of the solution method used
    pub method: String,
    /// Solution steps for educational output
    pub steps: Vec<String>,
}

/// Solve a separable ODE: dy/dx = g(x) * h(y)
///
/// The solution method:
/// 1. Separate: (1/h(y)) dy = g(x) dx
/// 2. Integrate both sides: ∫(1/h(y)) dy = ∫g(x) dx + C
///
/// # Arguments
///
/// * `ode` - The first-order ODE to solve
///
/// # Returns
///
/// The general solution as an expression in terms of y, x, and C.
#[must_use = "solving returns a result that should be used"]
pub fn solve_separable(ode: &FirstOrderODE) -> Result<ODESolution, ODEError> {
    let mut steps = Vec::new();
    steps.push(format!(
        "Given ODE: d{}/d{} = {}",
        ode.dependent, ode.independent, ode.rhs
    ));

    // Try to separate the equation
    let (g_x, h_y) = try_separate(&ode.rhs, &ode.independent, &ode.dependent)
        .ok_or(ODEError::NotSeparable)?;

    steps.push(format!(
        "Separating: d{}/d{} = ({}) * ({})",
        ode.dependent, ode.independent, g_x, h_y
    ));
    steps.push(format!(
        "Rearranging: (1/({})) d{} = ({}) d{}",
        h_y, ode.dependent, g_x, ode.independent
    ));

    // Compute 1/h(y)
    let one_over_h_y = Expression::Binary(
        BinaryOp::Div,
        Box::new(Expression::Integer(1)),
        Box::new(h_y.clone()),
    );

    // Integrate both sides
    let left_integral = integrate(&one_over_h_y, &ode.dependent)?;
    let right_integral = integrate(&g_x, &ode.independent)?;

    steps.push(format!(
        "Integrating left side: ∫(1/({})) d{} = {}",
        h_y, ode.dependent, left_integral
    ));
    steps.push(format!(
        "Integrating right side: ∫({}) d{} = {} + C",
        g_x, ode.independent, right_integral
    ));

    // Create the implicit solution: left_integral = right_integral + C
    let c = Expression::Variable(Variable::new("C"));
    let rhs_with_c = Expression::Binary(BinaryOp::Add, Box::new(right_integral), Box::new(c));

    // The solution is an implicit relation
    // For some common cases, we can solve explicitly for y
    let solution = try_solve_implicit_for_y(&left_integral, &rhs_with_c, &ode.dependent)
        .unwrap_or_else(|| {
            // Return implicit form: left = right + C
            Expression::Binary(
                BinaryOp::Sub,
                Box::new(left_integral.clone()),
                Box::new(rhs_with_c.clone()),
            )
        });

    steps.push(format!("General solution: {} = {}", ode.dependent, solution));

    Ok(ODESolution {
        general_solution: solution,
        method: "Separation of variables".to_string(),
        steps,
    })
}

/// Solve a first-order linear ODE: dy/dx + P(x)*y = Q(x)
///
/// The solution method uses an integrating factor:
/// 1. Compute integrating factor: μ(x) = e^(∫P(x)dx)
/// 2. Multiply through: d/dx(μ*y) = μ*Q
/// 3. Integrate: μ*y = ∫μ*Q dx + C
/// 4. Solve for y: y = (1/μ)(∫μ*Q dx + C)
///
/// # Arguments
///
/// * `ode` - The first-order ODE to solve (must be in form dy/dx + P(x)*y = Q(x))
///
/// # Returns
///
/// The general solution expression.
#[must_use = "solving returns a result that should be used"]
pub fn solve_linear(ode: &FirstOrderODE) -> Result<ODESolution, ODEError> {
    let mut steps = Vec::new();
    steps.push(format!(
        "Given ODE: d{}/d{} = {}",
        ode.dependent, ode.independent, ode.rhs
    ));

    // Extract P(x) and Q(x) from dy/dx = -P(x)*y + Q(x)
    // which is equivalent to dy/dx + P(x)*y = Q(x)
    let (p_x, q_x) =
        extract_linear_coefficients(&ode.rhs, &ode.independent, &ode.dependent)
            .ok_or(ODEError::NotLinear)?;

    // The ODE is dy/dx = rhs, and we extracted it as dy/dx = -P*y + Q
    // So the standard form is dy/dx + P*y = Q
    steps.push(format!(
        "Standard form: d{}/d{} + ({}) * {} = {}",
        ode.dependent, ode.independent, p_x, ode.dependent, q_x
    ));

    // Compute integrating factor μ(x) = e^(∫P(x)dx)
    let p_integral = integrate(&p_x, &ode.independent)?;
    let mu = Expression::Function(Function::Exp, vec![p_integral.clone()]);

    steps.push(format!(
        "Integrating factor: μ({}) = e^(∫{} d{}) = e^({})",
        ode.independent, p_x, ode.independent, p_integral
    ));

    // Compute μ*Q
    let mu_times_q = Expression::Binary(BinaryOp::Mul, Box::new(mu.clone()), Box::new(q_x.clone()));

    // Integrate μ*Q
    let mu_q_integral = integrate(&mu_times_q.simplify(), &ode.independent)?;

    steps.push(format!(
        "Integrating: ∫μ({}) * ({}) d{} = {}",
        ode.independent, q_x, ode.independent, mu_q_integral
    ));

    // Solution: y = (1/μ)(∫μ*Q dx + C)
    let c = Expression::Variable(Variable::new("C"));
    let integral_plus_c = Expression::Binary(BinaryOp::Add, Box::new(mu_q_integral), Box::new(c));

    let solution = Expression::Binary(
        BinaryOp::Div,
        Box::new(integral_plus_c),
        Box::new(mu.clone()),
    )
    .simplify();

    steps.push(format!(
        "General solution: {} = (∫μQ d{} + C) / μ = {}",
        ode.dependent, ode.independent, solution
    ));

    Ok(ODESolution {
        general_solution: solution,
        method: "Integrating factor".to_string(),
        steps,
    })
}

/// Solve an initial value problem.
///
/// Given an ODE and initial condition y(x0) = y0, find the particular solution.
///
/// # Arguments
///
/// * `ode` - The first-order ODE
/// * `x0` - The initial x value
/// * `y0` - The initial y value y(x0)
///
/// # Returns
///
/// The particular solution satisfying the initial condition.
pub fn solve_ivp(
    ode: &FirstOrderODE,
    x0: &Expression,
    y0: &Expression,
) -> Result<ODESolution, ODEError> {
    // First, get the general solution
    let general = if ode.is_separable() {
        solve_separable(ode)?
    } else if ode.is_linear() {
        solve_linear(ode)?
    } else {
        return Err(ODEError::CannotSolve(
            "ODE is neither separable nor linear".to_string(),
        ));
    };

    let mut steps = general.steps.clone();
    steps.push(format!(
        "Applying initial condition: {}({}) = {}",
        ode.dependent, x0, y0
    ));

    // Substitute x = x0 and y = y0 into the general solution to find C
    let substituted = substitute_var(&general.general_solution, &ode.independent, x0);
    let equation = Expression::Binary(
        BinaryOp::Sub,
        Box::new(substituted),
        Box::new(y0.clone()),
    );

    // Try to solve for C
    if let Some(c_value) = solve_for_constant(&equation.simplify(), "C") {
        steps.push(format!("Solving for C: C = {}", c_value));

        // Substitute C back into the general solution
        let particular = substitute_var(&general.general_solution, "C", &c_value).simplify();
        steps.push(format!(
            "Particular solution: {} = {}",
            ode.dependent, particular
        ));

        Ok(ODESolution {
            general_solution: particular,
            method: format!("{} with initial condition", general.method),
            steps,
        })
    } else {
        Err(ODEError::InitialConditionError(
            "Could not solve for constant C".to_string(),
        ))
    }
}

/// Attempt to separate dy/dx = f(x,y) into g(x) * h(y).
///
/// Returns (g(x), h(y)) if separable, None otherwise.
fn try_separate(expr: &Expression, x_var: &str, y_var: &str) -> Option<(Expression, Expression)> {
    // Check if expression is already a product
    if let Expression::Binary(BinaryOp::Mul, left, right) = expr {
        let left_has_x = left.contains_variable(x_var);
        let left_has_y = left.contains_variable(y_var);
        let right_has_x = right.contains_variable(x_var);
        let right_has_y = right.contains_variable(y_var);

        // Case: g(x) * h(y)
        if left_has_x && !left_has_y && right_has_y && !right_has_x {
            return Some((left.as_ref().clone(), right.as_ref().clone()));
        }
        // Case: h(y) * g(x)
        if left_has_y && !left_has_x && right_has_x && !right_has_y {
            return Some((right.as_ref().clone(), left.as_ref().clone()));
        }
        // Case: purely x-dependent (h(y) = 1)
        if (left_has_x || right_has_x) && !left_has_y && !right_has_y {
            return Some((expr.clone(), Expression::Integer(1)));
        }
        // Case: purely y-dependent (g(x) = 1)
        if (left_has_y || right_has_y) && !left_has_x && !right_has_x {
            return Some((Expression::Integer(1), expr.clone()));
        }
    }

    // Check if expression is purely x-dependent or y-dependent
    let has_x = expr.contains_variable(x_var);
    let has_y = expr.contains_variable(y_var);

    if has_x && !has_y {
        // dy/dx = g(x) is separable with h(y) = 1
        return Some((expr.clone(), Expression::Integer(1)));
    }
    if has_y && !has_x {
        // dy/dx = h(y) is separable with g(x) = 1
        return Some((Expression::Integer(1), expr.clone()));
    }
    if !has_x && !has_y {
        // Constant: dy/dx = c, separable with g(x) = c, h(y) = 1
        return Some((expr.clone(), Expression::Integer(1)));
    }

    // Check for division that might be separable: g(x)/h(y) or h(y)/g(x)
    if let Expression::Binary(BinaryOp::Div, num, denom) = expr {
        let num_has_x = num.contains_variable(x_var);
        let num_has_y = num.contains_variable(y_var);
        let denom_has_x = denom.contains_variable(x_var);
        let denom_has_y = denom.contains_variable(y_var);

        // g(x) / k(y) = g(x) * (1/k(y))
        if num_has_x && !num_has_y && denom_has_y && !denom_has_x {
            let h_y = Expression::Binary(
                BinaryOp::Div,
                Box::new(Expression::Integer(1)),
                denom.clone(),
            );
            return Some((num.as_ref().clone(), h_y));
        }
    }

    None
}

/// Extract P(x) and Q(x) from a linear ODE in form dy/dx = -P(x)*y + Q(x).
///
/// Returns (P(x), Q(x)) if linear, None otherwise.
fn extract_linear_coefficients(
    rhs: &Expression,
    x_var: &str,
    y_var: &str,
) -> Option<(Expression, Expression)> {
    // The RHS should be of form: terms with y (linear in y) + terms without y
    // dy/dx = a*y + b  where a might depend on x, b might depend on x
    // Standard form: dy/dx + P*y = Q means rhs = -P*y + Q

    // Collect terms with y and without y
    let mut y_coefficient = Expression::Integer(0);
    let mut constant_terms = Expression::Integer(0);

    fn collect_terms(
        expr: &Expression,
        y_var: &str,
        y_coeff: &mut Expression,
        const_terms: &mut Expression,
    ) -> bool {
        match expr {
            // Simple y term
            Expression::Variable(v) if v.name == y_var => {
                *y_coeff = Expression::Binary(
                    BinaryOp::Add,
                    Box::new(y_coeff.clone()),
                    Box::new(Expression::Integer(1)),
                );
                true
            }
            // Sum: recurse into both sides
            Expression::Binary(BinaryOp::Add, left, right) => {
                collect_terms(left, y_var, y_coeff, const_terms)
                    && collect_terms(right, y_var, y_coeff, const_terms)
            }
            // Difference: handle subtraction
            Expression::Binary(BinaryOp::Sub, left, right) => {
                let mut neg_y_coeff = Expression::Integer(0);
                let mut neg_const = Expression::Integer(0);
                if !collect_terms(left, y_var, y_coeff, const_terms) {
                    return false;
                }
                if !collect_terms(right, y_var, &mut neg_y_coeff, &mut neg_const) {
                    return false;
                }
                // Subtract the right side contributions
                *y_coeff = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(y_coeff.clone()),
                    Box::new(neg_y_coeff),
                );
                *const_terms = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(const_terms.clone()),
                    Box::new(neg_const),
                );
                true
            }
            // Product: check if linear in y
            Expression::Binary(BinaryOp::Mul, left, right) => {
                let left_has_y = left.contains_variable(y_var);
                let right_has_y = right.contains_variable(y_var);

                if left_has_y && right_has_y {
                    // y^2 or similar - not linear
                    return false;
                }
                if !left_has_y && !right_has_y {
                    // No y - this is a constant term
                    *const_terms = Expression::Binary(
                        BinaryOp::Add,
                        Box::new(const_terms.clone()),
                        Box::new(expr.clone()),
                    );
                    return true;
                }
                // One factor is y (or contains y linearly), other is coefficient
                if left_has_y {
                    // left is y or y-term, right is coefficient
                    if matches!(left.as_ref(), Expression::Variable(v) if v.name == y_var) {
                        *y_coeff = Expression::Binary(
                            BinaryOp::Add,
                            Box::new(y_coeff.clone()),
                            right.clone(),
                        );
                        return true;
                    }
                } else {
                    // right is y or y-term, left is coefficient
                    if matches!(right.as_ref(), Expression::Variable(v) if v.name == y_var) {
                        *y_coeff = Expression::Binary(
                            BinaryOp::Add,
                            Box::new(y_coeff.clone()),
                            left.clone(),
                        );
                        return true;
                    }
                }
                false
            }
            // Negation
            Expression::Unary(UnaryOp::Neg, inner) => {
                let mut neg_y_coeff = Expression::Integer(0);
                let mut neg_const = Expression::Integer(0);
                if !collect_terms(inner, y_var, &mut neg_y_coeff, &mut neg_const) {
                    return false;
                }
                *y_coeff = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(y_coeff.clone()),
                    Box::new(neg_y_coeff),
                );
                *const_terms = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(const_terms.clone()),
                    Box::new(neg_const),
                );
                true
            }
            // Any expression without y is a constant term
            _ if !expr.contains_variable(y_var) => {
                *const_terms = Expression::Binary(
                    BinaryOp::Add,
                    Box::new(const_terms.clone()),
                    Box::new(expr.clone()),
                );
                true
            }
            // Anything else with y that doesn't fit above is not linear
            _ => false,
        }
    }

    if !collect_terms(rhs, y_var, &mut y_coefficient, &mut constant_terms) {
        return None;
    }

    // Simplify collected terms
    let y_coeff = y_coefficient.simplify();
    let q_x = constant_terms.simplify();

    // P(x) is the negative of the y coefficient (since rhs = -P*y + Q)
    let p_x = Expression::Unary(UnaryOp::Neg, Box::new(y_coeff)).simplify();

    // Check that P(x) doesn't contain y
    if p_x.contains_variable(y_var) {
        return None;
    }

    Some((p_x, q_x))
}

/// Substitute a variable with an expression.
fn substitute_var(expr: &Expression, var: &str, replacement: &Expression) -> Expression {
    match expr {
        Expression::Variable(v) if v.name == var => replacement.clone(),
        Expression::Variable(_) => expr.clone(),
        Expression::Integer(_) | Expression::Float(_) | Expression::Rational(_) => expr.clone(),
        Expression::Constant(_) => expr.clone(),
        Expression::Complex(_) => expr.clone(),
        Expression::Binary(op, left, right) => Expression::Binary(
            *op,
            Box::new(substitute_var(left, var, replacement)),
            Box::new(substitute_var(right, var, replacement)),
        ),
        Expression::Unary(op, operand) => {
            Expression::Unary(*op, Box::new(substitute_var(operand, var, replacement)))
        }
        Expression::Function(func, args) => Expression::Function(
            func.clone(),
            args.iter()
                .map(|arg| substitute_var(arg, var, replacement))
                .collect(),
        ),
        Expression::Power(base, exp) => Expression::Power(
            Box::new(substitute_var(base, var, replacement)),
            Box::new(substitute_var(exp, var, replacement)),
        ),
    }
}

/// Try to solve an implicit relation for y explicitly.
/// For example: ln(y) = x + C => y = e^(x + C)
fn try_solve_implicit_for_y(
    left: &Expression,
    right: &Expression,
    y_var: &str,
) -> Option<Expression> {
    // Simple case: left is just y
    if matches!(left, Expression::Variable(v) if v.name == y_var) {
        return Some(right.clone());
    }

    // Case: ln(y) = right => y = e^right
    if let Expression::Function(Function::Ln, args) = left {
        if args.len() == 1 {
            if matches!(&args[0], Expression::Variable(v) if v.name == y_var) {
                return Some(Expression::Function(Function::Exp, vec![right.clone()]));
            }
            // ln(|y|) case - handle absolute value
            if let Expression::Function(Function::Abs, inner_args) = &args[0] {
                if inner_args.len() == 1 {
                    if matches!(&inner_args[0], Expression::Variable(v) if v.name == y_var) {
                        // y = ±e^right, return positive branch
                        return Some(Expression::Function(Function::Exp, vec![right.clone()]));
                    }
                }
            }
        }
    }

    // Case: y^n = right => y = right^(1/n)
    if let Expression::Power(base, exp) = left {
        if matches!(base.as_ref(), Expression::Variable(v) if v.name == y_var) {
            if !exp.contains_variable(y_var) {
                let one_over_n = Expression::Binary(
                    BinaryOp::Div,
                    Box::new(Expression::Integer(1)),
                    exp.clone(),
                );
                return Some(Expression::Power(Box::new(right.clone()), Box::new(one_over_n)));
            }
        }
    }

    // Case: 1/y = right => y = 1/right
    if let Expression::Binary(BinaryOp::Div, num, denom) = left {
        if matches!(num.as_ref(), Expression::Integer(1)) {
            if matches!(denom.as_ref(), Expression::Variable(v) if v.name == y_var) {
                return Some(Expression::Binary(
                    BinaryOp::Div,
                    Box::new(Expression::Integer(1)),
                    Box::new(right.clone()),
                ));
            }
        }
    }

    None
}

/// Try to solve an equation for a constant (typically C).
fn solve_for_constant(equation: &Expression, const_name: &str) -> Option<Expression> {
    // Simple case: equation is of form C - value = 0 or value - C = 0
    // or C = value form

    match equation {
        // C - value = 0 => C = value
        Expression::Binary(BinaryOp::Sub, left, right) => {
            if matches!(left.as_ref(), Expression::Variable(v) if v.name == const_name) {
                return Some(right.as_ref().clone());
            }
            if matches!(right.as_ref(), Expression::Variable(v) if v.name == const_name) {
                return Some(left.as_ref().clone());
            }
        }
        // C + value = 0 => C = -value
        Expression::Binary(BinaryOp::Add, left, right) => {
            if matches!(left.as_ref(), Expression::Variable(v) if v.name == const_name) {
                return Some(Expression::Unary(UnaryOp::Neg, right.clone()));
            }
            if matches!(right.as_ref(), Expression::Variable(v) if v.name == const_name) {
                return Some(Expression::Unary(UnaryOp::Neg, left.clone()));
            }
        }
        _ => {}
    }

    // Try to isolate C from more complex equations
    // For now, try numerical evaluation if possible
    if let Some(c_value) = try_numerical_solve_for_c(equation, const_name) {
        return Some(c_value);
    }

    None
}

/// Try to numerically solve for C.
fn try_numerical_solve_for_c(equation: &Expression, const_name: &str) -> Option<Expression> {
    // If the equation doesn't contain C, we can't solve for it
    if !equation.contains_variable(const_name) {
        return None;
    }

    // If the equation is linear in C, we can solve analytically
    // equation = a*C + b = 0 => C = -b/a

    // Try to extract coefficient of C
    let mut c_coefficient = Expression::Integer(0);
    let mut constant_part = Expression::Integer(0);

    fn extract_c_terms(
        expr: &Expression,
        c_name: &str,
        c_coeff: &mut Expression,
        const_part: &mut Expression,
    ) -> bool {
        match expr {
            Expression::Variable(v) if v.name == c_name => {
                *c_coeff = Expression::Binary(
                    BinaryOp::Add,
                    Box::new(c_coeff.clone()),
                    Box::new(Expression::Integer(1)),
                );
                true
            }
            Expression::Binary(BinaryOp::Add, left, right) => {
                extract_c_terms(left, c_name, c_coeff, const_part)
                    && extract_c_terms(right, c_name, c_coeff, const_part)
            }
            Expression::Binary(BinaryOp::Sub, left, right) => {
                let mut neg_c = Expression::Integer(0);
                let mut neg_const = Expression::Integer(0);
                if !extract_c_terms(left, c_name, c_coeff, const_part) {
                    return false;
                }
                if !extract_c_terms(right, c_name, &mut neg_c, &mut neg_const) {
                    return false;
                }
                *c_coeff = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(c_coeff.clone()),
                    Box::new(neg_c),
                );
                *const_part = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(const_part.clone()),
                    Box::new(neg_const),
                );
                true
            }
            Expression::Binary(BinaryOp::Mul, left, right) => {
                let left_has_c = left.contains_variable(c_name);
                let right_has_c = right.contains_variable(c_name);
                if left_has_c && right_has_c {
                    return false; // Non-linear in C
                }
                if !left_has_c && !right_has_c {
                    *const_part = Expression::Binary(
                        BinaryOp::Add,
                        Box::new(const_part.clone()),
                        Box::new(expr.clone()),
                    );
                    return true;
                }
                // One side is C, other is coefficient
                if left_has_c {
                    if matches!(left.as_ref(), Expression::Variable(v) if v.name == c_name) {
                        *c_coeff = Expression::Binary(
                            BinaryOp::Add,
                            Box::new(c_coeff.clone()),
                            right.clone(),
                        );
                        return true;
                    }
                } else if matches!(right.as_ref(), Expression::Variable(v) if v.name == c_name) {
                    *c_coeff = Expression::Binary(
                        BinaryOp::Add,
                        Box::new(c_coeff.clone()),
                        left.clone(),
                    );
                    return true;
                }
                false
            }
            Expression::Unary(UnaryOp::Neg, inner) => {
                let mut neg_c = Expression::Integer(0);
                let mut neg_const = Expression::Integer(0);
                if !extract_c_terms(inner, c_name, &mut neg_c, &mut neg_const) {
                    return false;
                }
                *c_coeff = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(c_coeff.clone()),
                    Box::new(neg_c),
                );
                *const_part = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(const_part.clone()),
                    Box::new(neg_const),
                );
                true
            }
            _ if !expr.contains_variable(c_name) => {
                *const_part = Expression::Binary(
                    BinaryOp::Add,
                    Box::new(const_part.clone()),
                    Box::new(expr.clone()),
                );
                true
            }
            _ => false,
        }
    }

    if !extract_c_terms(equation, const_name, &mut c_coefficient, &mut constant_part) {
        return None;
    }

    let c_coeff = c_coefficient.simplify();
    let b = constant_part.simplify();

    // C = -b/a
    let neg_b = Expression::Unary(UnaryOp::Neg, Box::new(b));
    let c_value = Expression::Binary(BinaryOp::Div, Box::new(neg_b), Box::new(c_coeff)).simplify();

    Some(c_value)
}

// =============================================================================
// Second-Order ODE Support
// =============================================================================

/// Type of characteristic equation roots
#[derive(Debug, Clone, PartialEq)]
pub enum RootType {
    /// Two distinct real roots r₁ ≠ r₂
    TwoDistinctReal,
    /// One repeated real root r = r₁ = r₂
    RepeatedReal,
    /// Complex conjugate roots α ± βi
    ComplexConjugate,
}

/// Result of solving the characteristic equation
#[derive(Debug, Clone)]
pub struct CharacteristicRoots {
    /// First root (or real part for complex)
    pub r1: f64,
    /// Second root (or imaginary part for complex)
    pub r2: f64,
    /// Type of roots
    pub root_type: RootType,
}

/// Represents a second-order linear ODE with constant coefficients:
/// a*y'' + b*y' + c*y = f(x)
#[derive(Debug, Clone)]
pub struct SecondOrderODE {
    /// The dependent variable name (e.g., "y")
    pub dependent: String,
    /// The independent variable name (e.g., "x")
    pub independent: String,
    /// Coefficient of y'' (must be constant)
    pub a: f64,
    /// Coefficient of y' (must be constant)
    pub b: f64,
    /// Coefficient of y (must be constant)
    pub c: f64,
    /// The forcing function f(x), or zero for homogeneous
    pub forcing: Expression,
}

impl SecondOrderODE {
    /// Create a new second-order ODE: a*y'' + b*y' + c*y = f(x)
    pub fn new(
        dependent: &str,
        independent: &str,
        a: f64,
        b: f64,
        c: f64,
        forcing: Expression,
    ) -> Self {
        SecondOrderODE {
            dependent: dependent.to_string(),
            independent: independent.to_string(),
            a,
            b,
            c,
            forcing,
        }
    }

    /// Create a homogeneous ODE: a*y'' + b*y' + c*y = 0
    pub fn homogeneous(dependent: &str, independent: &str, a: f64, b: f64, c: f64) -> Self {
        Self::new(dependent, independent, a, b, c, Expression::Integer(0))
    }

    /// Check if this ODE is homogeneous (f(x) = 0)
    pub fn is_homogeneous(&self) -> bool {
        matches!(&self.forcing, Expression::Integer(0))
            || matches!(&self.forcing, Expression::Float(x) if x.abs() < 1e-15)
    }
}

/// Result of solving a second-order ODE
#[derive(Debug, Clone)]
pub struct SecondOrderSolution {
    /// The homogeneous solution (with C1, C2 constants)
    pub homogeneous_solution: Expression,
    /// The particular solution (if non-homogeneous)
    pub particular_solution: Option<Expression>,
    /// The general solution (homogeneous + particular)
    pub general_solution: Expression,
    /// Description of the solution method
    pub method: String,
    /// The characteristic roots
    pub roots: CharacteristicRoots,
    /// Solution steps for educational output
    pub steps: Vec<String>,
}

/// Solve the characteristic equation ar² + br + c = 0
#[must_use = "solving returns a result that should be used"]
pub fn solve_characteristic_equation(a: f64, b: f64, c: f64) -> Result<CharacteristicRoots, ODEError> {
    if a.abs() < 1e-15 {
        return Err(ODEError::CharacteristicEquationError(
            "Coefficient 'a' cannot be zero for second-order ODE".to_string(),
        ));
    }

    let discriminant = b * b - 4.0 * a * c;
    const EPSILON: f64 = 1e-10;

    if discriminant > EPSILON {
        // Two distinct real roots
        let sqrt_disc = discriminant.sqrt();
        let r1 = (-b + sqrt_disc) / (2.0 * a);
        let r2 = (-b - sqrt_disc) / (2.0 * a);
        Ok(CharacteristicRoots {
            r1,
            r2,
            root_type: RootType::TwoDistinctReal,
        })
    } else if discriminant < -EPSILON {
        // Complex conjugate roots α ± βi
        let alpha = -b / (2.0 * a);
        let beta = (-discriminant).sqrt() / (2.0 * a);
        Ok(CharacteristicRoots {
            r1: alpha,
            r2: beta,
            root_type: RootType::ComplexConjugate,
        })
    } else {
        // Repeated root
        let r = -b / (2.0 * a);
        Ok(CharacteristicRoots {
            r1: r,
            r2: r,
            root_type: RootType::RepeatedReal,
        })
    }
}

/// Build the homogeneous solution for two distinct real roots.
/// y = C1*e^(r1*x) + C2*e^(r2*x)
fn build_solution_distinct_real(r1: f64, r2: f64, x_var: &str) -> Expression {
    let x = Expression::Variable(Variable::new(x_var));
    let c1 = Expression::Variable(Variable::new("C1"));
    let c2 = Expression::Variable(Variable::new("C2"));

    // C1 * e^(r1*x)
    let exp1_arg = Expression::Binary(
        BinaryOp::Mul,
        Box::new(Expression::Float(r1)),
        Box::new(x.clone()),
    );
    let exp1 = Expression::Function(Function::Exp, vec![exp1_arg]);
    let term1 = Expression::Binary(BinaryOp::Mul, Box::new(c1), Box::new(exp1));

    // C2 * e^(r2*x)
    let exp2_arg = Expression::Binary(
        BinaryOp::Mul,
        Box::new(Expression::Float(r2)),
        Box::new(x),
    );
    let exp2 = Expression::Function(Function::Exp, vec![exp2_arg]);
    let term2 = Expression::Binary(BinaryOp::Mul, Box::new(c2), Box::new(exp2));

    // C1*e^(r1*x) + C2*e^(r2*x)
    Expression::Binary(BinaryOp::Add, Box::new(term1), Box::new(term2))
}

/// Build the homogeneous solution for a repeated root.
/// y = (C1 + C2*x) * e^(r*x)
fn build_solution_repeated(r: f64, x_var: &str) -> Expression {
    let x = Expression::Variable(Variable::new(x_var));
    let c1 = Expression::Variable(Variable::new("C1"));
    let c2 = Expression::Variable(Variable::new("C2"));

    // C1 + C2*x
    let c2_x = Expression::Binary(BinaryOp::Mul, Box::new(c2), Box::new(x.clone()));
    let linear = Expression::Binary(BinaryOp::Add, Box::new(c1), Box::new(c2_x));

    // e^(r*x)
    let exp_arg = Expression::Binary(
        BinaryOp::Mul,
        Box::new(Expression::Float(r)),
        Box::new(x),
    );
    let exp_term = Expression::Function(Function::Exp, vec![exp_arg]);

    // (C1 + C2*x) * e^(r*x)
    Expression::Binary(BinaryOp::Mul, Box::new(linear), Box::new(exp_term))
}

/// Build the homogeneous solution for complex conjugate roots α ± βi.
/// y = e^(αx) * (C1*cos(βx) + C2*sin(βx))
fn build_solution_complex(alpha: f64, beta: f64, x_var: &str) -> Expression {
    let x = Expression::Variable(Variable::new(x_var));
    let c1 = Expression::Variable(Variable::new("C1"));
    let c2 = Expression::Variable(Variable::new("C2"));

    // βx
    let beta_x = Expression::Binary(
        BinaryOp::Mul,
        Box::new(Expression::Float(beta)),
        Box::new(x.clone()),
    );

    // C1*cos(βx)
    let cos_term = Expression::Function(Function::Cos, vec![beta_x.clone()]);
    let term1 = Expression::Binary(BinaryOp::Mul, Box::new(c1), Box::new(cos_term));

    // C2*sin(βx)
    let sin_term = Expression::Function(Function::Sin, vec![beta_x]);
    let term2 = Expression::Binary(BinaryOp::Mul, Box::new(c2), Box::new(sin_term));

    // C1*cos(βx) + C2*sin(βx)
    let oscillatory = Expression::Binary(BinaryOp::Add, Box::new(term1), Box::new(term2));

    // If alpha is essentially zero, no damping envelope needed
    if alpha.abs() < 1e-10 {
        oscillatory
    } else {
        // e^(αx)
        let exp_arg = Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Float(alpha)),
            Box::new(x),
        );
        let exp_term = Expression::Function(Function::Exp, vec![exp_arg]);

        // e^(αx) * (C1*cos(βx) + C2*sin(βx))
        Expression::Binary(BinaryOp::Mul, Box::new(exp_term), Box::new(oscillatory))
    }
}

/// Solve a homogeneous second-order linear ODE with constant coefficients.
/// a*y'' + b*y' + c*y = 0
#[must_use = "solving returns a result that should be used"]
pub fn solve_second_order_homogeneous(ode: &SecondOrderODE) -> Result<SecondOrderSolution, ODEError> {
    let mut steps = Vec::new();
    steps.push(format!(
        "Given ODE: {}·{}'' + {}·{}' + {}·{} = 0",
        ode.a, ode.dependent, ode.b, ode.dependent, ode.c, ode.dependent
    ));

    // Form characteristic equation
    steps.push(format!(
        "Characteristic equation: {}·r² + {}·r + {} = 0",
        ode.a, ode.b, ode.c
    ));

    // Solve characteristic equation
    let roots = solve_characteristic_equation(ode.a, ode.b, ode.c)?;

    let (method, solution) = match roots.root_type {
        RootType::TwoDistinctReal => {
            steps.push(format!(
                "Discriminant Δ = {}² - 4·{}·{} = {} > 0",
                ode.b, ode.a, ode.c,
                ode.b * ode.b - 4.0 * ode.a * ode.c
            ));
            steps.push(format!("Two distinct real roots: r₁ = {:.4}, r₂ = {:.4}", roots.r1, roots.r2));
            steps.push(format!("General solution: y = C1·e^({:.4}·{}) + C2·e^({:.4}·{})",
                roots.r1, ode.independent, roots.r2, ode.independent));
            (
                "Characteristic equation - distinct real roots".to_string(),
                build_solution_distinct_real(roots.r1, roots.r2, &ode.independent),
            )
        }
        RootType::RepeatedReal => {
            steps.push(format!(
                "Discriminant Δ = {}² - 4·{}·{} = 0",
                ode.b, ode.a, ode.c
            ));
            steps.push(format!("Repeated root: r = {:.4}", roots.r1));
            steps.push(format!("General solution: y = (C1 + C2·{})·e^({:.4}·{})",
                ode.independent, roots.r1, ode.independent));
            (
                "Characteristic equation - repeated root".to_string(),
                build_solution_repeated(roots.r1, &ode.independent),
            )
        }
        RootType::ComplexConjugate => {
            steps.push(format!(
                "Discriminant Δ = {}² - 4·{}·{} = {} < 0",
                ode.b, ode.a, ode.c,
                ode.b * ode.b - 4.0 * ode.a * ode.c
            ));
            steps.push(format!("Complex conjugate roots: r = {:.4} ± {:.4}i", roots.r1, roots.r2));
            if roots.r1.abs() < 1e-10 {
                steps.push(format!("General solution: y = C1·cos({:.4}·{}) + C2·sin({:.4}·{})",
                    roots.r2, ode.independent, roots.r2, ode.independent));
            } else {
                steps.push(format!("General solution: y = e^({:.4}·{})·(C1·cos({:.4}·{}) + C2·sin({:.4}·{}))",
                    roots.r1, ode.independent, roots.r2, ode.independent, roots.r2, ode.independent));
            }
            (
                "Characteristic equation - complex conjugate roots".to_string(),
                build_solution_complex(roots.r1, roots.r2, &ode.independent),
            )
        }
    };

    Ok(SecondOrderSolution {
        homogeneous_solution: solution.clone(),
        particular_solution: None,
        general_solution: solution,
        method,
        roots,
        steps,
    })
}

/// Solve a second-order IVP: a*y'' + b*y' + c*y = 0 with y(x0) = y0, y'(x0) = yp0
pub fn solve_second_order_ivp(
    ode: &SecondOrderODE,
    x0: f64,
    y0: f64,
    yp0: f64,
) -> Result<Expression, ODEError> {
    if !ode.is_homogeneous() {
        return Err(ODEError::CannotSolve(
            "IVP solver currently only supports homogeneous equations".to_string(),
        ));
    }

    let solution = solve_second_order_homogeneous(ode)?;

    // Determine C1 and C2 from initial conditions
    let (c1, c2) = match solution.roots.root_type {
        RootType::TwoDistinctReal => {
            let r1 = solution.roots.r1;
            let r2 = solution.roots.r2;
            // y = C1*e^(r1*x) + C2*e^(r2*x)
            // y' = C1*r1*e^(r1*x) + C2*r2*e^(r2*x)
            // At x = x0:
            // y0 = C1*e^(r1*x0) + C2*e^(r2*x0)
            // yp0 = C1*r1*e^(r1*x0) + C2*r2*e^(r2*x0)
            let e1 = (r1 * x0).exp();
            let e2 = (r2 * x0).exp();
            // Solve 2x2 system:
            // [ e1, e2   ] [C1]   [y0]
            // [ r1*e1, r2*e2 ] [C2] = [yp0]
            let det = e1 * r2 * e2 - e2 * r1 * e1;
            if det.abs() < 1e-15 {
                return Err(ODEError::InitialConditionError(
                    "Cannot determine constants - singular system".to_string(),
                ));
            }
            let c1 = (y0 * r2 * e2 - yp0 * e2) / det;
            let c2 = (yp0 * e1 - y0 * r1 * e1) / det;
            (c1, c2)
        }
        RootType::RepeatedReal => {
            let r = solution.roots.r1;
            // y = (C1 + C2*x)*e^(r*x)
            // y' = (C2 + r*(C1 + C2*x))*e^(r*x) = (C2 + r*C1 + r*C2*x)*e^(r*x)
            // At x = x0:
            // y0 = (C1 + C2*x0)*e^(r*x0)
            // yp0 = (C2 + r*C1 + r*C2*x0)*e^(r*x0)
            let e = (r * x0).exp();
            // From first equation: C1 + C2*x0 = y0/e
            // From second: C2 + r*C1 + r*C2*x0 = yp0/e
            //              C2 + r*(C1 + C2*x0) = yp0/e
            //              C2 + r*y0/e = yp0/e
            //              C2 = (yp0 - r*y0)/e
            let y0_over_e = y0 / e;
            let c2 = (yp0 / e) - r * y0_over_e;
            let c1 = y0_over_e - c2 * x0;
            (c1, c2)
        }
        RootType::ComplexConjugate => {
            let alpha = solution.roots.r1;
            let beta = solution.roots.r2;
            // y = e^(α*x)*(C1*cos(β*x) + C2*sin(β*x))
            // y' = α*y + e^(α*x)*(-C1*β*sin(β*x) + C2*β*cos(β*x))
            // At x = x0:
            let e = (alpha * x0).exp();
            let cos_bx0 = (beta * x0).cos();
            let sin_bx0 = (beta * x0).sin();
            // y0 = e*(C1*cos + C2*sin)
            // yp0 = α*e*(C1*cos + C2*sin) + e*β*(-C1*sin + C2*cos)
            //     = e*(C1*(α*cos - β*sin) + C2*(α*sin + β*cos))
            // Matrix form:
            // [ e*cos, e*sin ] [C1]   [y0]
            // [ e*(α*cos-β*sin), e*(α*sin+β*cos) ] [C2] = [yp0]
            let a11 = e * cos_bx0;
            let a12 = e * sin_bx0;
            let a21 = e * (alpha * cos_bx0 - beta * sin_bx0);
            let a22 = e * (alpha * sin_bx0 + beta * cos_bx0);
            let det = a11 * a22 - a12 * a21;
            if det.abs() < 1e-15 {
                return Err(ODEError::InitialConditionError(
                    "Cannot determine constants - singular system".to_string(),
                ));
            }
            let c1 = (y0 * a22 - yp0 * a12) / det;
            let c2 = (yp0 * a11 - y0 * a21) / det;
            (c1, c2)
        }
    };

    // Substitute C1 and C2 into the general solution
    let general = solution.general_solution;
    let with_c1 = substitute_var(&general, "C1", &Expression::Float(c1));
    let with_c2 = substitute_var(&with_c1, "C2", &Expression::Float(c2));

    Ok(with_c2.simplify())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn mul(left: Expression, right: Expression) -> Expression {
        Expression::Binary(BinaryOp::Mul, Box::new(left), Box::new(right))
    }

    fn add(left: Expression, right: Expression) -> Expression {
        Expression::Binary(BinaryOp::Add, Box::new(left), Box::new(right))
    }

    fn div(left: Expression, right: Expression) -> Expression {
        Expression::Binary(BinaryOp::Div, Box::new(left), Box::new(right))
    }

    fn neg(expr: Expression) -> Expression {
        Expression::Unary(UnaryOp::Neg, Box::new(expr))
    }

    #[test]
    fn test_try_separate_simple_product() {
        // dy/dx = x * y
        let expr = mul(var("x"), var("y"));
        let result = try_separate(&expr, "x", "y");
        assert!(result.is_some());
        let (g_x, h_y) = result.unwrap();
        assert!(matches!(g_x, Expression::Variable(v) if v.name == "x"));
        assert!(matches!(h_y, Expression::Variable(v) if v.name == "y"));
    }

    #[test]
    fn test_try_separate_only_x() {
        // dy/dx = x^2
        let x = var("x");
        let expr = Expression::Power(Box::new(x), Box::new(int(2)));
        let result = try_separate(&expr, "x", "y");
        assert!(result.is_some());
        let (g_x, h_y) = result.unwrap();
        assert!(matches!(g_x, Expression::Power(_, _)));
        assert!(matches!(h_y, Expression::Integer(1)));
    }

    #[test]
    fn test_try_separate_only_y() {
        // dy/dx = y^2
        let y = var("y");
        let expr = Expression::Power(Box::new(y), Box::new(int(2)));
        let result = try_separate(&expr, "x", "y");
        assert!(result.is_some());
        let (g_x, h_y) = result.unwrap();
        assert!(matches!(g_x, Expression::Integer(1)));
        assert!(matches!(h_y, Expression::Power(_, _)));
    }

    #[test]
    fn test_try_separate_constant() {
        // dy/dx = 5
        let expr = int(5);
        let result = try_separate(&expr, "x", "y");
        assert!(result.is_some());
        let (g_x, h_y) = result.unwrap();
        assert!(matches!(g_x, Expression::Integer(5)));
        assert!(matches!(h_y, Expression::Integer(1)));
    }

    #[test]
    fn test_is_separable() {
        // dy/dx = x * y is separable
        let ode = FirstOrderODE::new("y", "x", mul(var("x"), var("y")));
        assert!(ode.is_separable());

        // dy/dx = x + y is NOT separable
        let ode2 = FirstOrderODE::new("y", "x", add(var("x"), var("y")));
        assert!(!ode2.is_separable());
    }

    #[test]
    fn test_is_linear() {
        // dy/dx = -y + x is linear (P(x) = 1, Q(x) = x)
        let ode = FirstOrderODE::new("y", "x", add(neg(var("y")), var("x")));
        assert!(ode.is_linear());

        // dy/dx = y^2 is NOT linear
        let y = var("y");
        let ode2 = FirstOrderODE::new(
            "y",
            "x",
            Expression::Power(Box::new(y), Box::new(int(2))),
        );
        assert!(!ode2.is_linear());
    }

    #[test]
    fn test_extract_linear_coefficients() {
        // dy/dx = -2*y + 3*x
        // Standard form: dy/dx + 2*y = 3*x
        // So P(x) = 2, Q(x) = 3*x
        let rhs = add(mul(int(-2), var("y")), mul(int(3), var("x")));
        let result = extract_linear_coefficients(&rhs, "x", "y");
        assert!(result.is_some());
    }

    #[test]
    fn test_solve_separable_simple() {
        // dy/dx = y
        // Solution: y = C * e^x
        let ode = FirstOrderODE::new("y", "x", var("y"));
        let result = solve_separable(&ode);
        assert!(result.is_ok());

        let solution = result.unwrap();
        assert_eq!(solution.method, "Separation of variables");
        assert!(!solution.steps.is_empty());
    }

    #[test]
    fn test_solve_separable_xy() {
        // dy/dx = x*y
        // Separating: (1/y) dy = x dx
        // Integrating: ln|y| = x^2/2 + C
        // Solution: y = A * e^(x^2/2) where A = e^C
        let ode = FirstOrderODE::new("y", "x", mul(var("x"), var("y")));
        let result = solve_separable(&ode);
        assert!(result.is_ok());
    }

    #[test]
    fn test_solve_linear_simple() {
        // dy/dx + y = 0
        // This is dy/dx = -y, so rhs = -y
        // P(x) = 1, Q(x) = 0
        // μ = e^x
        // Solution: y = C * e^(-x)
        let ode = FirstOrderODE::new("y", "x", neg(var("y")));
        let result = solve_linear(&ode);
        assert!(result.is_ok());

        let solution = result.unwrap();
        assert_eq!(solution.method, "Integrating factor");
    }

    #[test]
    fn test_solve_ivp() {
        // dy/dx = y, y(0) = 1
        // General solution: y = C * e^x
        // With y(0) = 1: C = 1
        // Particular solution: y = e^x
        let ode = FirstOrderODE::new("y", "x", var("y"));
        let result = solve_ivp(&ode, &int(0), &int(1));
        assert!(result.is_ok());
    }

    #[test]
    fn test_substitute_var() {
        let expr = add(var("x"), var("y"));
        let result = substitute_var(&expr, "x", &int(5));
        // Should get 5 + y
        assert!(matches!(
            result,
            Expression::Binary(BinaryOp::Add, left, _) if matches!(left.as_ref(), Expression::Integer(5))
        ));
    }

    #[test]
    fn test_try_solve_implicit_ln_y() {
        // ln(y) = x + C => y = e^(x + C)
        let left = Expression::Function(Function::Ln, vec![var("y")]);
        let right = add(var("x"), var("C"));
        let result = try_solve_implicit_for_y(&left, &right, "y");
        assert!(result.is_some());
        assert!(matches!(result.unwrap(), Expression::Function(Function::Exp, _)));
    }

    // =========================================================================
    // Second-Order ODE Tests
    // =========================================================================

    #[test]
    fn test_characteristic_equation_distinct_real() {
        // r² - 1 = 0 => r = ±1
        let roots = solve_characteristic_equation(1.0, 0.0, -1.0).unwrap();
        assert_eq!(roots.root_type, RootType::TwoDistinctReal);
        assert!((roots.r1 - 1.0).abs() < 1e-10);
        assert!((roots.r2 - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_characteristic_equation_complex() {
        // r² + 1 = 0 => r = ±i
        let roots = solve_characteristic_equation(1.0, 0.0, 1.0).unwrap();
        assert_eq!(roots.root_type, RootType::ComplexConjugate);
        assert!(roots.r1.abs() < 1e-10); // alpha = 0
        assert!((roots.r2 - 1.0).abs() < 1e-10); // beta = 1
    }

    #[test]
    fn test_characteristic_equation_repeated() {
        // r² - 2r + 1 = 0 => (r-1)² = 0 => r = 1 (double)
        let roots = solve_characteristic_equation(1.0, -2.0, 1.0).unwrap();
        assert_eq!(roots.root_type, RootType::RepeatedReal);
        assert!((roots.r1 - 1.0).abs() < 1e-10);
        assert!((roots.r2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_second_order_homogeneous_distinct_real() {
        // y'' - y = 0 => y = C1*e^x + C2*e^(-x)
        let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 0.0, -1.0);
        let solution = solve_second_order_homogeneous(&ode).unwrap();

        assert_eq!(solution.method, "Characteristic equation - distinct real roots");
        assert_eq!(solution.roots.root_type, RootType::TwoDistinctReal);
        assert!(!solution.steps.is_empty());
    }

    #[test]
    fn test_second_order_homogeneous_complex() {
        // y'' + y = 0 => y = C1*cos(x) + C2*sin(x)
        let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 0.0, 1.0);
        let solution = solve_second_order_homogeneous(&ode).unwrap();

        assert_eq!(solution.method, "Characteristic equation - complex conjugate roots");
        assert_eq!(solution.roots.root_type, RootType::ComplexConjugate);
    }

    #[test]
    fn test_second_order_homogeneous_repeated() {
        // y'' - 2y' + y = 0 => y = (C1 + C2*x)*e^x
        let ode = SecondOrderODE::homogeneous("y", "x", 1.0, -2.0, 1.0);
        let solution = solve_second_order_homogeneous(&ode).unwrap();

        assert_eq!(solution.method, "Characteristic equation - repeated root");
        assert_eq!(solution.roots.root_type, RootType::RepeatedReal);
    }

    #[test]
    fn test_second_order_ivp_complex() {
        // y'' + y = 0, y(0) = 1, y'(0) = 0 => y = cos(x)
        let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 0.0, 1.0);
        let solution = solve_second_order_ivp(&ode, 0.0, 1.0, 0.0).unwrap();

        // Evaluate at x = 0: should be 1
        let mut vars = std::collections::HashMap::new();
        vars.insert("x".to_string(), 0.0);
        let result = solution.evaluate(&vars).unwrap();
        assert!((result - 1.0).abs() < 1e-10);

        // Evaluate at x = π/2: should be 0
        vars.insert("x".to_string(), std::f64::consts::FRAC_PI_2);
        let result = solution.evaluate(&vars).unwrap();
        assert!(result.abs() < 1e-6);
    }

    #[test]
    fn test_second_order_ivp_distinct_real() {
        // y'' - y = 0, y(0) = 1, y'(0) = 0
        // General: y = C1*e^x + C2*e^(-x)
        // y(0) = C1 + C2 = 1
        // y'(0) = C1 - C2 = 0 => C1 = C2 = 0.5
        // y = 0.5*e^x + 0.5*e^(-x) = cosh(x)
        let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 0.0, -1.0);
        let solution = solve_second_order_ivp(&ode, 0.0, 1.0, 0.0).unwrap();

        // At x = 0, should be 1
        let mut vars = std::collections::HashMap::new();
        vars.insert("x".to_string(), 0.0);
        let result = solution.evaluate(&vars).unwrap();
        assert!((result - 1.0).abs() < 1e-10);

        // At x = 1, should be cosh(1) ≈ 1.543
        vars.insert("x".to_string(), 1.0);
        let result = solution.evaluate(&vars).unwrap();
        let expected = 1.0_f64.cosh();
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_second_order_ode_is_homogeneous() {
        let ode1 = SecondOrderODE::homogeneous("y", "x", 1.0, 2.0, 3.0);
        assert!(ode1.is_homogeneous());

        let ode2 = SecondOrderODE::new("y", "x", 1.0, 2.0, 3.0, var("x"));
        assert!(!ode2.is_homogeneous());
    }
}
