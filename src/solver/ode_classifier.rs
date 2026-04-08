//! ODE classifier: determines order, linearity, and type of differential equations.
//!
//! This module classifies [`FirstOrderODE`] and [`SecondOrderODE`] structs so that
//! callers can select the appropriate solver strategy without attempting each method
//! in turn.
//!
//! # Examples
//!
//! ```rust
//! use thales::ast::{BinaryOp, Expression, Variable};
//! use thales::ode::FirstOrderODE;
//! use thales::solver::ode_classifier::{classify_first_order, ODEType};
//!
//! // dy/dx = x * y  →  separable
//! let rhs = Expression::Binary(
//!     BinaryOp::Mul,
//!     Box::new(Expression::Variable(Variable::new("x"))),
//!     Box::new(Expression::Variable(Variable::new("y"))),
//! );
//! let ode = FirstOrderODE::new("y", "x", rhs);
//! let cls = classify_first_order(&ode);
//! assert_eq!(cls.ode_type, ODEType::Separable);
//! ```

use crate::ast::{BinaryOp, Expression, UnaryOp};
use crate::ode::{FirstOrderODE, SecondOrderODE};

// ---------------------------------------------------------------------------
// Public type definitions
// ---------------------------------------------------------------------------

/// Order of the differential equation.
#[derive(Debug, Clone, PartialEq)]
pub enum ODEOrder {
    /// dy/dx = f(x, y)
    First,
    /// a·y'' + b·y' + c·y = f(x)
    Second,
    /// Order > 2 (not yet handled by this library)
    Higher(usize),
}

/// Whether the ODE is linear in the dependent variable and its derivatives.
#[derive(Debug, Clone, PartialEq)]
pub enum ODELinearity {
    /// The dependent variable appears only to the first power and is not
    /// nested inside non-linear functions.
    Linear,
    /// Otherwise.
    Nonlinear,
}

/// Structural type of the ODE, used to route to the correct solver.
#[derive(Debug, Clone, PartialEq)]
pub enum ODEType {
    /// dy/dx = g(x) · h(y) — can be solved by separation of variables.
    Separable,
    /// dy/dx + P(x)·y = Q(x) — solved via integrating factor.
    Linear,
    /// dy/dx + P(x)·y = Q(x)·y^n, n ≠ 0,1 — solved by substitution v = y^(1-n).
    Bernoulli,
    /// M(x,y)dx + N(x,y)dy = 0 with ∂M/∂y = ∂N/∂x.
    Exact,
    /// a·y'' + b·y' + c·y = 0 — constant-coefficient second-order.
    ConstantCoefficient,
    /// a·y'' + b·y' + c·y = f(x) with f(x) ≠ 0.
    NonHomogeneousConstantCoefficient,
    /// Could not determine a specific type.
    Unknown,
}

/// Full classification of a differential equation.
#[derive(Debug, Clone, PartialEq)]
pub struct ODEClassification {
    /// Order of the equation.
    pub order: ODEOrder,
    /// Linearity in the dependent variable.
    pub linearity: ODELinearity,
    /// Structural type, used for solver routing.
    pub ode_type: ODEType,
    /// Whether all coefficients are constants (i.e. independent of x).
    pub has_constant_coefficients: bool,
    /// Name of the dependent variable (e.g. `"y"`).
    pub dependent_var: String,
    /// Name of the independent variable (e.g. `"x"`).
    pub independent_var: String,
}

// ---------------------------------------------------------------------------
// First-order classification
// ---------------------------------------------------------------------------

/// Classify a first-order ODE represented as dy/dx = rhs.
///
/// The classifier checks, in order:
/// 1. Separable: `try_separate` succeeds (uses `FirstOrderODE::is_separable`).
/// 2. Linear: `extract_linear_coefficients` succeeds (uses `FirstOrderODE::is_linear`).
/// 3. Bernoulli: rhs is of the form coefficient(x)·y^n, n ≠ 0,1.
/// 4. Fallback: Unknown.
///
/// # Examples
///
/// ```rust
/// use thales::ast::{BinaryOp, Expression, UnaryOp, Variable};
/// use thales::ode::FirstOrderODE;
/// use thales::solver::ode_classifier::{classify_first_order, ODEType, ODELinearity};
///
/// // dy/dx = -y + x  →  linear
/// let rhs = Expression::Binary(
///     BinaryOp::Add,
///     Box::new(Expression::Unary(
///         UnaryOp::Neg,
///         Box::new(Expression::Variable(Variable::new("y"))),
///     )),
///     Box::new(Expression::Variable(Variable::new("x"))),
/// );
/// let ode = FirstOrderODE::new("y", "x", rhs);
/// let cls = classify_first_order(&ode);
/// assert_eq!(cls.ode_type, ODEType::Linear);
/// assert_eq!(cls.linearity, ODELinearity::Linear);
/// ```
pub fn classify_first_order(ode: &FirstOrderODE) -> ODEClassification {
    // Determine linearity in y.
    let linearity = first_order_linearity(ode);

    // Determine structural type.
    let ode_type = if ode.is_separable() {
        ODEType::Separable
    } else if ode.is_linear() {
        ODEType::Linear
    } else if is_bernoulli(&ode.rhs, &ode.independent, &ode.dependent) {
        ODEType::Bernoulli
    } else {
        ODEType::Unknown
    };

    // First-order equations with constant coefficients means P(x) = const.
    let has_constant_coefficients = has_constant_coeff_first_order(ode);

    ODEClassification {
        order: ODEOrder::First,
        linearity,
        ode_type,
        has_constant_coefficients,
        dependent_var: ode.dependent.clone(),
        independent_var: ode.independent.clone(),
    }
}

// ---------------------------------------------------------------------------
// Second-order classification
// ---------------------------------------------------------------------------

/// Classify a second-order ODE with constant coefficients: a·y'' + b·y' + c·y = f(x).
///
/// [`SecondOrderODE`] always has constant coefficients (a, b, c are `f64`), so
/// `has_constant_coefficients` is always `true`. The `ode_type` distinguishes
/// homogeneous from non-homogeneous.
///
/// # Examples
///
/// ```rust
/// use thales::ode::SecondOrderODE;
/// use thales::solver::ode_classifier::{classify_second_order, ODEType};
///
/// // y'' + 2y' + y = 0  →  ConstantCoefficient (homogeneous)
/// let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 2.0, 1.0);
/// let cls = classify_second_order(&ode);
/// assert_eq!(cls.ode_type, ODEType::ConstantCoefficient);
/// assert!(cls.has_constant_coefficients);
/// ```
pub fn classify_second_order(ode: &SecondOrderODE) -> ODEClassification {
    let ode_type = if ode.is_homogeneous() {
        ODEType::ConstantCoefficient
    } else {
        ODEType::NonHomogeneousConstantCoefficient
    };

    ODEClassification {
        order: ODEOrder::Second,
        linearity: ODELinearity::Linear, // SecondOrderODE is always linear by construction
        ode_type,
        has_constant_coefficients: true, // a, b, c are always f64 constants
        dependent_var: ode.dependent.clone(),
        independent_var: ode.independent.clone(),
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Determine whether the first-order ODE is linear in the dependent variable.
///
/// A first-order ODE dy/dx = f(x,y) is linear when f is linear in y, i.e.
/// f = -P(x)·y + Q(x).  We reuse `FirstOrderODE::is_linear` for this test.
fn first_order_linearity(ode: &FirstOrderODE) -> ODELinearity {
    if ode.is_linear() {
        ODELinearity::Linear
    } else {
        ODELinearity::Nonlinear
    }
}

/// Determine whether the first-order ODE has constant coefficients.
///
/// For dy/dx = f(x,y), "constant coefficients" means f does not depend on x
/// at all (autonomous equation), e.g. dy/dx = ay.
fn has_constant_coeff_first_order(ode: &FirstOrderODE) -> bool {
    !ode.rhs.contains_variable(&ode.independent)
}

/// Heuristic test for Bernoulli form: dy/dx = g(x)·y^n, n ≠ 0, 1.
///
/// We look for a product `coeff * y^n` or `y^n * coeff` where coeff
/// contains x (or is constant) and n ≠ 0, 1.
fn is_bernoulli(rhs: &Expression, x_var: &str, y_var: &str) -> bool {
    extract_bernoulli_exponent(rhs, x_var, y_var)
        .map(|n| n != 0.0 && (n - 1.0).abs() > 1e-10)
        .unwrap_or(false)
}

/// Try to extract the exponent n from a Bernoulli-form expression g(x)·y^n.
///
/// Returns `None` if the expression does not match the Bernoulli pattern.
fn extract_bernoulli_exponent(expr: &Expression, x_var: &str, y_var: &str) -> Option<f64> {
    // Pattern: y^n or coeff * y^n or y^n * coeff
    match expr {
        // Plain y — exponent 1 (not Bernoulli but well-formed)
        Expression::Variable(v) if v.name == y_var => Some(1.0),

        // y^n
        Expression::Power(base, exp) => {
            let base_is_y = matches!(base.as_ref(), Expression::Variable(v) if v.name == y_var);
            if base_is_y && !exp.contains_variable(y_var) {
                exponent_to_f64(exp)
            } else {
                None
            }
        }

        // coeff(x) * y^n  or  y^n * coeff(x)
        Expression::Binary(BinaryOp::Mul, left, right) => {
            let left_has_y = left.contains_variable(y_var);
            let right_has_y = right.contains_variable(y_var);

            match (left_has_y, right_has_y) {
                (true, false) => extract_bernoulli_exponent(left, x_var, y_var),
                (false, true) => extract_bernoulli_exponent(right, x_var, y_var),
                _ => None,
            }
        }

        // Negation: -y^n is still Bernoulli
        Expression::Unary(UnaryOp::Neg, inner) => extract_bernoulli_exponent(inner, x_var, y_var),

        _ => None,
    }
}

/// Convert a constant expression to f64 for exponent comparison.
fn exponent_to_f64(expr: &Expression) -> Option<f64> {
    match expr {
        Expression::Integer(n) => Some(*n as f64),
        Expression::Float(f) => Some(*f),
        Expression::Rational(r) => {
            if *r.denom() == 0 {
                None
            } else {
                Some(*r.numer() as f64 / *r.denom() as f64)
            }
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Expression, UnaryOp, Variable};
    use crate::ode::{FirstOrderODE, SecondOrderODE};

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn mul(l: Expression, r: Expression) -> Expression {
        Expression::Binary(BinaryOp::Mul, Box::new(l), Box::new(r))
    }

    fn add(l: Expression, r: Expression) -> Expression {
        Expression::Binary(BinaryOp::Add, Box::new(l), Box::new(r))
    }

    fn neg(e: Expression) -> Expression {
        Expression::Unary(UnaryOp::Neg, Box::new(e))
    }

    fn pow(base: Expression, exp: Expression) -> Expression {
        Expression::Power(Box::new(base), Box::new(exp))
    }

    // ------------------------------------------------------------------
    // First-order: separable
    // ------------------------------------------------------------------

    #[test]
    fn test_classify_first_order_separable() {
        // dy/dx = x * y
        let ode = FirstOrderODE::new("y", "x", mul(var("x"), var("y")));
        let cls = classify_first_order(&ode);
        assert_eq!(cls.order, ODEOrder::First);
        assert_eq!(cls.ode_type, ODEType::Separable);
        assert_eq!(cls.linearity, ODELinearity::Nonlinear); // x*y is not -P*y+Q linear
        assert_eq!(cls.dependent_var, "y");
        assert_eq!(cls.independent_var, "x");
    }

    // ------------------------------------------------------------------
    // First-order: linear
    // ------------------------------------------------------------------

    #[test]
    fn test_classify_first_order_linear() {
        // dy/dx = -y + x  (standard linear: dy/dx + y = x)
        let rhs = add(neg(var("y")), var("x"));
        let ode = FirstOrderODE::new("y", "x", rhs);
        let cls = classify_first_order(&ode);
        assert_eq!(cls.order, ODEOrder::First);
        assert_eq!(cls.ode_type, ODEType::Linear);
        assert_eq!(cls.linearity, ODELinearity::Linear);
    }

    // ------------------------------------------------------------------
    // First-order: Bernoulli (y^2)
    // ------------------------------------------------------------------

    #[test]
    fn test_classify_first_order_bernoulli() {
        // dy/dx = x * y^2  →  Bernoulli with n=2
        let rhs = mul(var("x"), pow(var("y"), int(2)));
        let ode = FirstOrderODE::new("y", "x", rhs);
        let cls = classify_first_order(&ode);
        assert_eq!(cls.order, ODEOrder::First);
        assert_eq!(cls.ode_type, ODEType::Bernoulli);
        assert_eq!(cls.linearity, ODELinearity::Nonlinear);
    }

    // ------------------------------------------------------------------
    // First-order: constant coefficient (autonomous)
    // ------------------------------------------------------------------

    #[test]
    fn test_classify_first_order_constant_coeff() {
        // dy/dx = -3*y  (autonomous, no x in rhs)
        let rhs = mul(int(-3), var("y"));
        let ode = FirstOrderODE::new("y", "x", rhs);
        let cls = classify_first_order(&ode);
        assert!(cls.has_constant_coefficients);
    }

    #[test]
    fn test_classify_first_order_non_constant_coeff() {
        // dy/dx = x * y  (x appears → not constant-coefficient)
        let ode = FirstOrderODE::new("y", "x", mul(var("x"), var("y")));
        let cls = classify_first_order(&ode);
        assert!(!cls.has_constant_coefficients);
    }

    // ------------------------------------------------------------------
    // Second-order: homogeneous constant coefficient
    // ------------------------------------------------------------------

    #[test]
    fn test_classify_second_order_homogeneous() {
        // y'' + 2y' + y = 0
        let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 2.0, 1.0);
        let cls = classify_second_order(&ode);
        assert_eq!(cls.order, ODEOrder::Second);
        assert_eq!(cls.ode_type, ODEType::ConstantCoefficient);
        assert_eq!(cls.linearity, ODELinearity::Linear);
        assert!(cls.has_constant_coefficients);
        assert_eq!(cls.dependent_var, "y");
        assert_eq!(cls.independent_var, "x");
    }

    // ------------------------------------------------------------------
    // Second-order: non-homogeneous
    // ------------------------------------------------------------------

    #[test]
    fn test_classify_second_order_non_homogeneous() {
        // y'' + 2y' + y = x  (forced oscillator)
        let ode = SecondOrderODE::new("y", "x", 1.0, 2.0, 1.0, var("x"));
        let cls = classify_second_order(&ode);
        assert_eq!(cls.order, ODEOrder::Second);
        assert_eq!(cls.ode_type, ODEType::NonHomogeneousConstantCoefficient);
        assert!(cls.has_constant_coefficients);
        assert_eq!(cls.linearity, ODELinearity::Linear);
    }

    // ------------------------------------------------------------------
    // Second-order: zero forcing treated as homogeneous
    // ------------------------------------------------------------------

    #[test]
    fn test_classify_second_order_zero_forcing() {
        // SecondOrderODE::new with forcing = 0 is still homogeneous
        let ode = SecondOrderODE::new("y", "x", 1.0, 0.0, 1.0, Expression::Integer(0));
        let cls = classify_second_order(&ode);
        assert_eq!(cls.ode_type, ODEType::ConstantCoefficient);
    }
}
