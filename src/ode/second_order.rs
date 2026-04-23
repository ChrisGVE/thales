//! Second-order ODE solvers (characteristic equation method).

use std::collections::HashMap;
use std::sync::Arc;

use crate::ast::Expression;
use crate::numeric::compile::{compile, decompile};
use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::{normalize, SymbolId};

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
///
/// The forcing function is stored in canonical `Arc<Expr>` form. Use
/// [`SecondOrderODE::forcing_arc`] for engine-native access.
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
    forcing: Arc<Expr>,
}

impl SecondOrderODE {
    /// Create a new second-order ODE: a*y'' + b*y' + c*y = f(x)
    ///
    /// The forcing function is compiled to canonical `Arc<Expr>` form on
    /// construction.
    pub fn new(
        dependent: &str,
        independent: &str,
        a: f64,
        b: f64,
        c: f64,
        forcing: Expression,
    ) -> Self {
        Self::from_arc(dependent, independent, a, b, c, compile(&forcing))
    }

    /// Create a second-order ODE directly from a pre-compiled `Arc<Expr>`
    /// forcing function.
    ///
    /// Prefer this constructor when the caller already operates in canonical
    /// form — it avoids a redundant decompile/compile round-trip.
    #[must_use]
    pub fn from_arc(
        dependent: &str,
        independent: &str,
        a: f64,
        b: f64,
        c: f64,
        forcing: Arc<Expr>,
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
        Self::from_arc(dependent, independent, a, b, c, Expr::int(0))
    }

    /// Check if this ODE is homogeneous (f(x) = 0)
    pub fn is_homogeneous(&self) -> bool {
        use crate::numeric::ring::Ring;
        match self.forcing.as_ref() {
            Expr::Integer(n) => n.is_zero(),
            Expr::Float(x) => x.abs() < 1e-15,
            _ => false,
        }
    }

    /// Forcing function as a canonical `Arc<Expr>` (clone of the stored field).
    ///
    /// Engine-native accessor. Cheap `Arc::clone`.
    #[must_use]
    pub fn forcing_arc(&self) -> Arc<Expr> {
        Arc::clone(&self.forcing)
    }
}

/// Result of solving a second-order ODE
#[derive(Debug, Clone)]
pub struct SecondOrderSolution {
    /// The homogeneous solution (with C1, C2 constants), in canonical
    /// [`Arc<Expr>`] form.
    pub homogeneous_solution: Arc<Expr>,
    /// The particular solution (if non-homogeneous), in canonical
    /// [`Arc<Expr>`] form.
    pub particular_solution: Option<Arc<Expr>>,
    /// The general solution (homogeneous + particular), in canonical
    /// [`Arc<Expr>`] form.
    pub general_solution: Arc<Expr>,
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
fn build_solution_distinct_real(r1: f64, r2: f64, x_var: &str) -> Arc<Expr> {
    let x = Expr::symbol(x_var);
    let c1 = Expr::symbol("C1");
    let c2 = Expr::symbol("C2");

    // C1 * e^(r1*x)
    let exp1_arg = normalize::mul(Arc::new(Expr::Float(r1)), x.clone());
    let exp1 = Expr::func(FuncId::Exp, vec![exp1_arg]);
    let term1 = normalize::mul(c1, exp1);

    // C2 * e^(r2*x)
    let exp2_arg = normalize::mul(Arc::new(Expr::Float(r2)), x);
    let exp2 = Expr::func(FuncId::Exp, vec![exp2_arg]);
    let term2 = normalize::mul(c2, exp2);

    // C1*e^(r1*x) + C2*e^(r2*x)
    normalize::add(term1, term2)
}

/// Build the homogeneous solution for a repeated root.
/// y = (C1 + C2*x) * e^(r*x)
fn build_solution_repeated(r: f64, x_var: &str) -> Arc<Expr> {
    let x = Expr::symbol(x_var);
    let c1 = Expr::symbol("C1");
    let c2 = Expr::symbol("C2");

    // C1 + C2*x
    let c2_x = normalize::mul(c2, x.clone());
    let linear = normalize::add(c1, c2_x);

    // e^(r*x)
    let exp_arg = normalize::mul(Arc::new(Expr::Float(r)), x);
    let exp_term = Expr::func(FuncId::Exp, vec![exp_arg]);

    // (C1 + C2*x) * e^(r*x)
    normalize::mul(linear, exp_term)
}

/// Build the homogeneous solution for complex conjugate roots α ± βi.
/// y = e^(αx) * (C1*cos(βx) + C2*sin(βx))
fn build_solution_complex(alpha: f64, beta: f64, x_var: &str) -> Arc<Expr> {
    let x = Expr::symbol(x_var);
    let c1 = Expr::symbol("C1");
    let c2 = Expr::symbol("C2");

    // βx
    let beta_x = normalize::mul(Arc::new(Expr::Float(beta)), x.clone());

    // C1*cos(βx)
    let cos_term = Expr::func(FuncId::Cos, vec![beta_x.clone()]);
    let term1 = normalize::mul(c1, cos_term);

    // C2*sin(βx)
    let sin_term = Expr::func(FuncId::Sin, vec![beta_x]);
    let term2 = normalize::mul(c2, sin_term);

    // C1*cos(βx) + C2*sin(βx)
    let oscillatory = normalize::add(term1, term2);

    // If alpha is essentially zero, no damping envelope needed
    if alpha.abs() < 1e-10 {
        oscillatory
    } else {
        // e^(αx)
        let exp_arg = normalize::mul(Arc::new(Expr::Float(alpha)), x);
        let exp_term = Expr::func(FuncId::Exp, vec![exp_arg]);

        // e^(αx) * (C1*cos(βx) + C2*sin(βx))
        normalize::mul(exp_term, oscillatory)
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

    let (method, solution_arc) = match roots.root_type {
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
        homogeneous_solution: Arc::clone(&solution_arc),
        particular_solution: None,
        general_solution: solution_arc,
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
    let general_arc = Arc::clone(&solution.general_solution);
    let c1_id = SymbolId::intern("C1");
    let c2_id = SymbolId::intern("C2");
    let c1_arc = Expr::float(c1);
    let c2_arc = Expr::float(c2);
    let with_c1 = substitute_var(&general_arc, c1_id, &c1_arc);
    let with_c2 = substitute_var(&with_c1, c2_id, &c2_arc);

    // TODO(arc-migration): solve_second_order_ivp is re-exported from lib.rs — decompile at Rule 2 boundary.
    Ok(decompile(&with_c2).simplify())
}
