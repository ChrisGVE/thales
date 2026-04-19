//! First-order ODE solvers (separable, linear, IVP).
//!
//! Inspection walkers (`try_separate`, `extract_linear_coefficients`,
//! `substitute_var`, `try_solve_implicit_for_y`, `solve_for_constant`)
//! live in [`walkers`]; the public solver entry points re-export them.

mod walkers;

use walkers::solve_for_constant;
pub(super) use walkers::{
    extract_linear_coefficients, substitute_var, try_separate, try_solve_implicit_for_y,
};

use crate::ast::{BinaryOp, Expression, Function, Variable};
use crate::integration::integrate;
use crate::numeric::compile::{compile, decompile};
use crate::numeric::SymbolId;

use super::{FirstOrderODE, ODEError, ODESolution};

pub fn solve_separable(ode: &FirstOrderODE) -> Result<ODESolution, ODEError> {
    let mut steps = Vec::new();
    let rhs_arc = ode.rhs_arc();
    steps.push(format!(
        "Given ODE: d{}/d{} = {}",
        ode.dependent,
        ode.independent,
        decompile(&rhs_arc)
    ));

    // Try to separate the equation
    let x_id = SymbolId::intern(&ode.independent);
    let y_id = SymbolId::intern(&ode.dependent);
    let (g_arc, h_arc) = try_separate(&rhs_arc, x_id, y_id).ok_or(ODEError::NotSeparable)?;
    let g_x = decompile(&g_arc);
    let h_y = decompile(&h_arc);

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
    let left_arc = compile(&left_integral);
    let rhs_with_c_arc = compile(&rhs_with_c);
    let solution = try_solve_implicit_for_y(&left_arc, &rhs_with_c_arc, y_id)
        .map(|arc| decompile(&arc))
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
    let rhs_arc = ode.rhs_arc();
    steps.push(format!(
        "Given ODE: d{}/d{} = {}",
        ode.dependent,
        ode.independent,
        decompile(&rhs_arc)
    ));

    // Extract P(x) and Q(x) from dy/dx = -P(x)*y + Q(x)
    // which is equivalent to dy/dx + P(x)*y = Q(x)
    let y_id = SymbolId::intern(&ode.dependent);
    let (p_arc, q_arc) = extract_linear_coefficients(&rhs_arc, y_id).ok_or(ODEError::NotLinear)?;
    let p_x = decompile(&p_arc);
    let q_x = decompile(&q_arc);

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
    let c_id = SymbolId::intern("C");
    let x_id = SymbolId::intern(&ode.independent);
    let general_arc = compile(&general.general_solution);
    let x0_arc = compile(x0);
    let substituted_arc = substitute_var(&general_arc, x_id, &x0_arc);
    let substituted = decompile(&substituted_arc);
    let equation = Expression::Binary(BinaryOp::Sub, Box::new(substituted), Box::new(y0.clone()));

    // Try to solve for C
    let equation_arc = compile(&equation.simplify());
    if let Some(c_value_arc) = solve_for_constant(&equation_arc, c_id) {
        let c_value = decompile(&c_value_arc);
        steps.push(format!("Solving for C: C = {}", c_value));

        // Substitute C back into the general solution
        let particular_arc = substitute_var(&general_arc, c_id, &c_value_arc);
        let particular = decompile(&particular_arc).simplify();

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
