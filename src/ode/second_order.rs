//! Second-order ODE solvers (characteristic equation method).

use std::collections::HashMap;
use std::sync::Arc;

use crate::ast::{BinaryOp, Expression, Function, Variable};
use crate::numeric::compile::compile;
use crate::numeric::Expr;
use crate::resolution_path::{Operation, ResolutionPath, ResolutionPathBuilder};

use super::first_order::substitute_var;
use super::ODEError;

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

    /// Forcing function as a canonical `Arc<Expr>`.
    ///
    /// Migration bridge for the Expr-migration milestone: second-order solvers
    /// moving to `Arc<Expr>` internals call this accessor instead of compiling
    /// the `Expression`-typed field themselves. See also
    /// [`FirstOrderODE::rhs_arc`](crate::ode::FirstOrderODE::rhs_arc).
    #[must_use]
    pub fn forcing_arc(&self) -> Arc<Expr> {
        compile(&self.forcing)
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
pub fn solve_characteristic_equation(
    a: f64,
    b: f64,
    c: f64,
) -> Result<CharacteristicRoots, ODEError> {
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
    let exp2_arg = Expression::Binary(BinaryOp::Mul, Box::new(Expression::Float(r2)), Box::new(x));
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
    let exp_arg = Expression::Binary(BinaryOp::Mul, Box::new(Expression::Float(r)), Box::new(x));
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
pub fn solve_second_order_homogeneous(
    ode: &SecondOrderODE,
) -> Result<SecondOrderSolution, ODEError> {
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
                ode.b,
                ode.a,
                ode.c,
                ode.b * ode.b - 4.0 * ode.a * ode.c
            ));
            steps.push(format!(
                "Two distinct real roots: r₁ = {:.4}, r₂ = {:.4}",
                roots.r1, roots.r2
            ));
            steps.push(format!(
                "General solution: y = C1·e^({:.4}·{}) + C2·e^({:.4}·{})",
                roots.r1, ode.independent, roots.r2, ode.independent
            ));
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
            steps.push(format!(
                "General solution: y = (C1 + C2·{})·e^({:.4}·{})",
                ode.independent, roots.r1, ode.independent
            ));
            (
                "Characteristic equation - repeated root".to_string(),
                build_solution_repeated(roots.r1, &ode.independent),
            )
        }
        RootType::ComplexConjugate => {
            steps.push(format!(
                "Discriminant Δ = {}² - 4·{}·{} = {} < 0",
                ode.b,
                ode.a,
                ode.c,
                ode.b * ode.b - 4.0 * ode.a * ode.c
            ));
            steps.push(format!(
                "Complex conjugate roots: r = {:.4} ± {:.4}i",
                roots.r1, roots.r2
            ));
            if roots.r1.abs() < 1e-10 {
                steps.push(format!(
                    "General solution: y = C1·cos({:.4}·{}) + C2·sin({:.4}·{})",
                    roots.r2, ode.independent, roots.r2, ode.independent
                ));
            } else {
                steps.push(format!(
                    "General solution: y = e^({:.4}·{})·(C1·cos({:.4}·{}) + C2·sin({:.4}·{}))",
                    roots.r1, ode.independent, roots.r2, ode.independent, roots.r2, ode.independent
                ));
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
