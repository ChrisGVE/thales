//! First-order ODE solvers (separable, linear, IVP).

use crate::ast::{BinaryOp, Expression, Function, UnaryOp, Variable};
use crate::integration::integrate;
use crate::resolution_path::{Operation, ResolutionPath, ResolutionPathBuilder};
use std::collections::HashMap;

use super::{FirstOrderODE, ODEError, ODESolution};

pub fn solve_separable(ode: &FirstOrderODE) -> Result<ODESolution, ODEError> {
    let mut steps = Vec::new();
    let rhs_expr = ode.rhs_expr();
    steps.push(format!(
        "Given ODE: d{}/d{} = {}",
        ode.dependent, ode.independent, rhs_expr
    ));

    // Try to separate the equation
    let (g_x, h_y) =
        try_separate(&rhs_expr, &ode.independent, &ode.dependent).ok_or(ODEError::NotSeparable)?;

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

    steps.push(format!(
        "General solution: {} = {}",
        ode.dependent, solution
    ));

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
    let rhs_expr = ode.rhs_expr();
    steps.push(format!(
        "Given ODE: d{}/d{} = {}",
        ode.dependent, ode.independent, rhs_expr
    ));

    // Extract P(x) and Q(x) from dy/dx = -P(x)*y + Q(x)
    // which is equivalent to dy/dx + P(x)*y = Q(x)
    let (p_x, q_x) = extract_linear_coefficients(&rhs_expr, &ode.independent, &ode.dependent)
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
    let equation = Expression::Binary(BinaryOp::Sub, Box::new(substituted), Box::new(y0.clone()));

    // Try to solve for C
    if let Some(c_value) = solve_for_constant(&equation.simplify(), "C") {
        steps.push(format!("Solving for C: C = {}", c_value));

        // Substitute C back into the general solution
        let particular = substitute_var(&general.general_solution, "C", &c_value).simplify();

        // Decision 2b invariant: the returned particular solution must be
        // explicit in the dependent variable. If `y` still appears, the
        // general solution was implicit (e.g. `try_solve_implicit_for_y`
        // fell through and we returned `left - right`) — propagate as an
        // error rather than returning an implicit form the caller cannot
        // evaluate.
        if particular.contains_variable(&ode.dependent) {
            return Err(ODEError::InitialConditionError(format!(
                "could not isolate '{}' in particular solution: {}",
                ode.dependent, particular
            )));
        }

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
pub(super) fn try_separate(
    expr: &Expression,
    x_var: &str,
    y_var: &str,
) -> Option<(Expression, Expression)> {
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
pub(super) fn extract_linear_coefficients(
    rhs: &Expression,
    _x_var: &str,
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
pub(super) fn substitute_var(expr: &Expression, var: &str, replacement: &Expression) -> Expression {
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
pub(super) fn try_solve_implicit_for_y(
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
                return Some(Expression::Power(
                    Box::new(right.clone()),
                    Box::new(one_over_n),
                ));
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
            // Exp(C) - value = 0 => C = Ln(value)  (isolating from an explicit
            // exponential form produced by ln-based antiderivatives).
            if let Expression::Function(Function::Exp, args) = left.as_ref() {
                if args.len() == 1 {
                    if matches!(&args[0], Expression::Variable(v) if v.name == const_name) {
                        return Some(Expression::Function(
                            Function::Ln,
                            vec![right.as_ref().clone()],
                        ));
                    }
                }
            }
            if let Expression::Function(Function::Exp, args) = right.as_ref() {
                if args.len() == 1 {
                    if matches!(&args[0], Expression::Variable(v) if v.name == const_name) {
                        return Some(Expression::Function(
                            Function::Ln,
                            vec![left.as_ref().clone()],
                        ));
                    }
                }
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
                *c_coeff =
                    Expression::Binary(BinaryOp::Sub, Box::new(c_coeff.clone()), Box::new(neg_c));
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
                    *c_coeff =
                        Expression::Binary(BinaryOp::Add, Box::new(c_coeff.clone()), left.clone());
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
                *c_coeff =
                    Expression::Binary(BinaryOp::Sub, Box::new(c_coeff.clone()), Box::new(neg_c));
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
