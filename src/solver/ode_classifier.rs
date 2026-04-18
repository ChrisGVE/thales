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

use std::sync::Arc;

use crate::ast::Expression;
use crate::numeric::compile::compile;
use crate::numeric::{Expr, SymbolId};
use crate::ode::{FirstOrderODE, SecondOrderODE};

use super::helpers::contains_symbol;

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
/// 3. Bernoulli: rhs contains a y^n term (n ≠ 0,1) possibly with a linear term.
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
    let linearity = first_order_linearity(ode);

    let ode_type = if ode.is_separable() {
        ODEType::Separable
    } else if ode.is_linear() {
        ODEType::Linear
    } else if is_bernoulli(&ode.rhs, &ode.independent, &ode.dependent) {
        ODEType::Bernoulli
    } else {
        ODEType::Unknown
    };

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
        linearity: ODELinearity::Linear,
        ode_type,
        has_constant_coefficients: true,
        dependent_var: ode.dependent.clone(),
        independent_var: ode.independent.clone(),
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Determine whether the first-order ODE is linear in the dependent variable.
fn first_order_linearity(ode: &FirstOrderODE) -> ODELinearity {
    if ode.is_linear() {
        ODELinearity::Linear
    } else {
        ODELinearity::Nonlinear
    }
}

/// Determine whether the first-order ODE has constant coefficients.
///
/// "Constant coefficients" means the rhs does not depend on the independent
/// variable at all (autonomous equation), e.g. dy/dx = a*y.
///
/// Operates on a canonical `Arc<Expr>` form of the rhs using
/// [`contains_symbol`].
fn has_constant_coeff_first_order(ode: &FirstOrderODE) -> bool {
    let rhs_arc = compile(&ode.rhs);
    let x_id = SymbolId::intern(&ode.independent);
    !contains_symbol(&rhs_arc, x_id)
}

/// Heuristic test for Bernoulli form: dy/dx + P(x)·y = Q(x)·y^n, n ≠ 0, 1.
///
/// In terms of the RHS, rhs contains a y^n term (n ≠ 0, 1) possibly combined
/// with a linear-in-y term. We find the highest-power y-term in rhs.
fn is_bernoulli(rhs: &Expression, _x_var: &str, y_var: &str) -> bool {
    let rhs_arc = compile(rhs);
    let y_id = SymbolId::intern(y_var);
    find_max_y_exponent_expr(&rhs_arc, y_id)
        .map(|n| n != 0 && n != 1)
        .unwrap_or(false)
}

/// Find the maximum exponent of `y_var` in the canonical `expr`, walking
/// additive sums.
///
/// `y + x·y²` returns `2`; a pure product `x·y²` returns `2`.
///
/// Uses exact integer exponents — matches the canonical `Arc<Expr>` form
/// where `Pow` exponents are integer literals after normalization.
fn find_max_y_exponent_expr(expr: &Arc<Expr>, y_var: SymbolId) -> Option<i64> {
    match expr.as_ref() {
        Expr::Add(node) => {
            let mut best: Option<i64> = None;
            for (term, _coeff) in &node.terms {
                if let Some(n) = extract_y_exponent_expr(term, y_var) {
                    best = Some(match best {
                        Some(prev) if prev >= n => prev,
                        _ => n,
                    });
                }
            }
            best
        }
        _ => extract_y_exponent_expr(expr, y_var),
    }
}

/// Extract the exponent `n` from a single canonical term of the form
/// `coeff(x) · y^n`.
///
/// Returns `None` when the term does not involve `y_var` as a simple
/// factor, when the exponent is non-integer, or when `y_var` appears
/// inside a function or inside the exponent itself.
fn extract_y_exponent_expr(expr: &Arc<Expr>, y_var: SymbolId) -> Option<i64> {
    match expr.as_ref() {
        Expr::Symbol(s) if *s == y_var => Some(1),
        Expr::Pow(base, exp) => {
            let base_is_y = matches!(base.as_ref(), Expr::Symbol(s) if *s == y_var);
            if base_is_y && !contains_symbol(exp, y_var) {
                exponent_to_i64_expr(exp)
            } else {
                None
            }
        }
        Expr::Mul(node) => {
            // Canonical Mul carries factors as (base, exp) pairs. `y^2`
            // normalizes with `base = Pow(Symbol(y), 2)` and `exp = 1`,
            // so both direct `Symbol(y)` and `Pow(Symbol(y), k)` bases
            // are recognized here.
            for (base, exp) in &node.factors {
                if contains_symbol(exp, y_var) {
                    return None;
                }
                match base.as_ref() {
                    Expr::Symbol(s) if *s == y_var => {
                        return exponent_to_i64_expr(exp);
                    }
                    Expr::Pow(inner_base, inner_exp) => {
                        let inner_is_y = matches!(
                            inner_base.as_ref(),
                            Expr::Symbol(s) if *s == y_var
                        );
                        if inner_is_y && !contains_symbol(inner_exp, y_var) {
                            let outer = exponent_to_i64_expr(exp)?;
                            let inner = exponent_to_i64_expr(inner_exp)?;
                            return Some(outer * inner);
                        }
                        if contains_symbol(base, y_var) {
                            return None;
                        }
                    }
                    _ => {
                        if contains_symbol(base, y_var) {
                            return None;
                        }
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Convert a canonical `Arc<Expr>` constant to `i64` for exponent
/// comparison. Accepts integer literals and integer-valued rationals /
/// floats.
fn exponent_to_i64_expr(expr: &Arc<Expr>) -> Option<i64> {
    match expr.as_ref() {
        Expr::Integer(n) => n.to_i64(),
        Expr::Rational(r) => {
            if r.is_integer() {
                r.numer().to_i64()
            } else {
                None
            }
        }
        Expr::Float(f) => {
            if (*f - f.round()).abs() < 1e-10 {
                Some(f.round() as i64)
            } else {
                None
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
        // dy/dx = x * y is separable (g(x)=x, h(y)=y) and also satisfies
        // the linear form P(x)=-x, Q(x)=0. Separable takes priority in type
        // routing, and linearity is Linear because x*y fits -P(x)*y + Q(x).
        let ode = FirstOrderODE::new("y", "x", mul(var("x"), var("y")));
        let cls = classify_first_order(&ode);
        assert_eq!(cls.order, ODEOrder::First);
        assert_eq!(cls.ode_type, ODEType::Separable);
        assert_eq!(cls.linearity, ODELinearity::Linear);
        assert_eq!(cls.dependent_var, "y");
        assert_eq!(cls.independent_var, "x");
    }

    // ------------------------------------------------------------------
    // First-order: linear (non-separable)
    // ------------------------------------------------------------------

    #[test]
    fn test_classify_first_order_linear() {
        // dy/dx = -y + x  →  standard linear dy/dx + y = x
        // Not separable because rhs = (-y + x) mixes x and y additively.
        let rhs = add(neg(var("y")), var("x"));
        let ode = FirstOrderODE::new("y", "x", rhs);
        let cls = classify_first_order(&ode);
        assert_eq!(cls.order, ODEOrder::First);
        assert_eq!(cls.ode_type, ODEType::Linear);
        assert_eq!(cls.linearity, ODELinearity::Linear);
    }

    // ------------------------------------------------------------------
    // First-order: Bernoulli (non-separable, non-linear)
    // ------------------------------------------------------------------

    #[test]
    fn test_classify_first_order_bernoulli() {
        // dy/dx = y + x*y^2 — Bernoulli with n=2.
        // Not separable (additive mix of y and x*y^2 involving both vars).
        // Not linear (y^2 term present). Bernoulli is detected.
        let rhs = add(var("y"), mul(var("x"), pow(var("y"), int(2))));
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
        // dy/dx = -3*y  (autonomous: no x in rhs)
        let rhs = mul(int(-3), var("y"));
        let ode = FirstOrderODE::new("y", "x", rhs);
        let cls = classify_first_order(&ode);
        assert!(cls.has_constant_coefficients);
    }

    #[test]
    fn test_classify_first_order_non_constant_coeff() {
        // dy/dx = x * y  (x appears in rhs → not constant-coefficient)
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
