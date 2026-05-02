//! ODE extraction from mathlex equations containing `Derivative` nodes.

use crate::ast::{BinaryOp, Expression, UnaryOp, Variable};
use crate::ode::{FirstOrderODE, SecondOrderODE};

use super::convert::convert_expression;

/// The result of extracting an ODE from a mathlex equation.
#[derive(Debug)]
pub enum ExtractedODE {
    /// First-order ODE: dy/dx = f(x, y)
    First(FirstOrderODE),
    /// Second-order ODE: a·y'' + b·y' + c·y = f(x)
    Second(SecondOrderODE),
}

/// Attempt to extract an ODE from a mathlex equation containing `Derivative` nodes.
///
/// Recognizes two forms:
/// 1. **First-order**: `dy/dx = rhs` or `y' = rhs` — the LHS is a single first-order
///    derivative and the RHS is converted to a thales expression.
/// 2. **Second-order constant-coefficient**: `a·d²y/dx² + b·dy/dx + c·y = f(x)` — the
///    equation is a linear combination of the dependent variable and its first and second
///    derivatives, with constant (numeric) coefficients.
///
/// Returns `None` if the equation does not match either pattern.
///
/// # Examples
///
/// ```rust
/// use thales::mathlex_bridge::try_extract_ode;
///
/// // First-order: dy/dx = x*y
/// let ml = mathlex::parse("dy/dx = x*y").unwrap();
/// let ode = try_extract_ode(&ml).unwrap();
/// assert!(matches!(ode, thales::mathlex_bridge::ExtractedODE::First(_)));
/// ```
pub fn try_extract_ode(expr: &mathlex::Expression) -> Option<ExtractedODE> {
    let (left, right) = match &expr.kind {
        mathlex::ExprKind::Equation { left, right } => (left.as_ref(), right.as_ref()),
        _ => return None,
    };

    // Case 1: LHS is a single derivative — first-order ODE
    if let mathlex::ExprKind::Derivative { expr, var, order } = &left.kind {
        if *order == 1 {
            let dep_var = extract_variable_name(expr)?;
            let rhs = convert_expression(right).ok()?;
            return Some(ExtractedODE::First(FirstOrderODE::new(&dep_var, var, rhs)));
        }
    }

    // Case 2: collect derivative terms from both sides to form a·y'' + b·y' + c·y = f(x)
    try_extract_second_order(left, right)
}

/// Extract the variable name from a simple mathlex variable expression.
pub(super) fn extract_variable_name(expr: &mathlex::Expression) -> Option<String> {
    match &expr.kind {
        mathlex::ExprKind::Variable(name) => Some(name.clone()),
        _ => None,
    }
}

/// Accumulator for collecting constant-coefficient ODE terms.
#[derive(Default)]
struct ODETerms {
    /// Coefficient of y'' (second derivative)
    coeff_y2: f64,
    /// Coefficient of y' (first derivative)
    coeff_y1: f64,
    /// Coefficient of y (zeroth derivative)
    coeff_y0: f64,
    /// Remaining non-derivative, non-y terms (the forcing function)
    forcing_terms: Vec<Expression>,
    /// Dependent variable name (e.g. "y")
    dependent: Option<String>,
    /// Independent variable name (e.g. "x")
    independent: Option<String>,
}

impl ODETerms {
    /// Record a derivative term with its coefficient and sign.
    fn add_derivative_term(
        &mut self,
        coeff: f64,
        dep: &str,
        indep: &str,
        order: u32,
    ) -> Option<()> {
        // Verify consistent variable names
        if let Some(ref d) = self.dependent {
            if d != dep {
                return None;
            }
        } else {
            self.dependent = Some(dep.to_string());
        }
        if let Some(ref i) = self.independent {
            if i != indep {
                return None;
            }
        } else {
            self.independent = Some(indep.to_string());
        }

        match order {
            1 => self.coeff_y1 += coeff,
            2 => self.coeff_y2 += coeff,
            _ => return None, // only handle first and second order
        }
        Some(())
    }

    /// Record a dependent-variable term (y with no derivative) with its coefficient.
    fn add_y_term(&mut self, coeff: f64, dep: &str) -> Option<()> {
        if let Some(ref d) = self.dependent {
            if d != dep {
                return None;
            }
        } else {
            self.dependent = Some(dep.to_string());
        }
        self.coeff_y0 += coeff;
        Some(())
    }
}

/// Try to extract a second-order constant-coefficient ODE from a mathlex equation.
///
/// Collects terms from LHS - RHS = 0, identifying derivative terms with constant
/// coefficients. If the highest derivative order is 2, builds a `SecondOrderODE`.
/// If the highest is 1, builds a `FirstOrderODE` (for cases like `2*dy/dx + y = x`).
fn try_extract_second_order(
    left: &mathlex::Expression,
    right: &mathlex::Expression,
) -> Option<ExtractedODE> {
    let mut terms = ODETerms::default();

    // Collect terms from LHS with sign +1
    collect_ode_terms(left, 1.0, &mut terms)?;
    // Collect terms from RHS with sign -1 (moved to LHS)
    collect_ode_terms(right, -1.0, &mut terms)?;

    let dep = terms.dependent.as_ref()?;
    let indep = terms.independent.as_ref()?;

    // Build the forcing function: negate the accumulated forcing terms (they were
    // collected as LHS - RHS = 0, so forcing = -sum(non-derivative terms on LHS)
    // + sum(non-derivative terms on RHS))
    let forcing = if terms.forcing_terms.is_empty() {
        Expression::Integer(0)
    } else {
        let mut combined = terms.forcing_terms[0].clone();
        for term in &terms.forcing_terms[1..] {
            combined =
                Expression::Binary(BinaryOp::Add, Box::new(combined), Box::new(term.clone()));
        }
        // Negate: the forcing terms were collected as (LHS terms - RHS terms),
        // but the ODE form is a·y'' + b·y' + c·y = f(x), so f(x) = -forcing
        Expression::Unary(UnaryOp::Neg, Box::new(combined))
    };

    if terms.coeff_y2.abs() > 1e-15 {
        // Second-order ODE
        Some(ExtractedODE::Second(SecondOrderODE::new(
            dep,
            indep,
            terms.coeff_y2,
            terms.coeff_y1,
            terms.coeff_y0,
            forcing,
        )))
    } else if terms.coeff_y1.abs() > 1e-15 {
        // First-order linear ODE with constant coefficients: b·y' + c·y = f(x)
        // Normalize to dy/dx = (-c/b)·y + f(x)/b
        build_first_order_from_terms(dep, indep, &terms, forcing)
    } else {
        None
    }
}

/// Build a `FirstOrderODE` from accumulated terms where the highest order is 1.
fn build_first_order_from_terms(
    dep: &str,
    indep: &str,
    terms: &ODETerms,
    forcing: Expression,
) -> Option<ExtractedODE> {
    let b = terms.coeff_y1;
    let c = terms.coeff_y0;

    let mut rhs_parts: Vec<Expression> = Vec::new();

    // (-c/b) * y term
    if c.abs() > 1e-15 {
        let coeff = -c / b;
        rhs_parts.push(Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Float(coeff)),
            Box::new(Expression::Variable(Variable::new(dep))),
        ));
    }

    // f(x)/b term — forcing was already negated, so the RHS contribution is -forcing/b
    if !matches!(&forcing, Expression::Integer(0)) {
        if (b - 1.0).abs() < 1e-15 {
            // b = 1, just negate forcing
            rhs_parts.push(Expression::Unary(UnaryOp::Neg, Box::new(forcing)));
        } else {
            rhs_parts.push(Expression::Binary(
                BinaryOp::Div,
                Box::new(Expression::Unary(UnaryOp::Neg, Box::new(forcing))),
                Box::new(Expression::Float(b)),
            ));
        }
    }

    let rhs = if rhs_parts.is_empty() {
        Expression::Integer(0)
    } else {
        let mut combined = rhs_parts.remove(0);
        for part in rhs_parts {
            combined = Expression::Binary(BinaryOp::Add, Box::new(combined), Box::new(part));
        }
        combined
    };

    Some(ExtractedODE::First(FirstOrderODE::new(dep, indep, rhs)))
}

/// Recursively collect ODE terms from a mathlex expression, tracking sign.
///
/// Recognizes:
/// - `Derivative { expr: Variable(y), var: x, order: n }` → derivative term
/// - `Variable(y)` where y is the dependent variable → y term
/// - `Binary(Mul, const, derivative)` → scaled derivative term
/// - `Binary(Add/Sub, left, right)` → recurse into both
/// - `Unary(Neg, inner)` → recurse with flipped sign
/// - Anything else → forcing term (converted to thales expression)
fn collect_ode_terms(expr: &mathlex::Expression, sign: f64, terms: &mut ODETerms) -> Option<()> {
    match &expr.kind {
        mathlex::ExprKind::Binary {
            op: mathlex::BinaryOp::Add,
            left,
            right,
        } => {
            collect_ode_terms(left, sign, terms)?;
            collect_ode_terms(right, sign, terms)?;
        }

        mathlex::ExprKind::Binary {
            op: mathlex::BinaryOp::Sub,
            left,
            right,
        } => {
            collect_ode_terms(left, sign, terms)?;
            collect_ode_terms(right, -sign, terms)?;
        }

        mathlex::ExprKind::Unary {
            op: mathlex::UnaryOp::Neg,
            operand,
        } => {
            collect_ode_terms(operand, -sign, terms)?;
        }

        mathlex::ExprKind::Derivative { expr, var, order } => {
            let dep = extract_variable_name(expr)?;
            terms.add_derivative_term(sign, &dep, var, *order)?;
        }

        mathlex::ExprKind::Binary {
            op: mathlex::BinaryOp::Mul,
            left,
            right,
        } => {
            collect_mul_term(expr, left, right, sign, terms)?;
        }

        mathlex::ExprKind::Variable(name) => {
            collect_variable_term(expr, name, sign, terms)?;
        }

        mathlex::ExprKind::Integer(0) => {
            // Zero contributes nothing
        }

        _ => {
            // Any other expression is part of the forcing function
            push_forcing_term(expr, sign, terms)?;
        }
    }

    Some(())
}

/// Handle a `Binary(Mul, ...)` node in ODE term collection.
fn collect_mul_term(
    expr: &mathlex::Expression,
    left: &mathlex::Expression,
    right: &mathlex::Expression,
    sign: f64,
    terms: &mut ODETerms,
) -> Option<()> {
    // Try coeff * derivative or derivative * coeff
    if let Some(c) = mathlex_to_f64(left) {
        if let mathlex::ExprKind::Derivative { expr, var, order } = &right.kind {
            let dep = extract_variable_name(expr)?;
            terms.add_derivative_term(sign * c, &dep, var, *order)?;
            return Some(());
        }
        // Try coeff * y
        if let Some(dep) = extract_variable_name(right) {
            if terms.dependent.as_deref() == Some(&dep) || terms.dependent.is_none() {
                terms.add_y_term(sign * c, &dep)?;
                return Some(());
            }
        }
    }
    if let Some(c) = mathlex_to_f64(right) {
        if let mathlex::ExprKind::Derivative { expr, var, order } = &left.kind {
            let dep = extract_variable_name(expr)?;
            terms.add_derivative_term(sign * c, &dep, var, *order)?;
            return Some(());
        }
        // Try y * coeff
        if let Some(dep) = extract_variable_name(left) {
            if terms.dependent.as_deref() == Some(&dep) || terms.dependent.is_none() {
                terms.add_y_term(sign * c, &dep)?;
                return Some(());
            }
        }
    }
    // Not a recognized ODE term — treat as forcing
    push_forcing_term(expr, sign, terms)
}

/// Handle a `Variable` node in ODE term collection.
fn collect_variable_term(
    expr: &mathlex::Expression,
    name: &str,
    sign: f64,
    terms: &mut ODETerms,
) -> Option<()> {
    // Could be the dependent variable (y term with coeff 1)
    if terms.dependent.as_deref() == Some(name) || terms.dependent.is_none() {
        if terms.dependent.is_some() {
            terms.add_y_term(sign, name)?;
        } else {
            // Can't determine if this is the dependent variable yet;
            // treat as potential forcing term
            push_forcing_term(expr, sign, terms)?;
        }
    } else {
        // Different variable — part of forcing function
        push_forcing_term(expr, sign, terms)?;
    }
    Some(())
}

/// Convert an expression and push it (with sign) onto the forcing terms list.
fn push_forcing_term(expr: &mathlex::Expression, sign: f64, terms: &mut ODETerms) -> Option<()> {
    let converted = convert_expression(expr).ok()?;
    let signed = if sign < 0.0 {
        Expression::Unary(UnaryOp::Neg, Box::new(converted))
    } else {
        converted
    };
    terms.forcing_terms.push(signed);
    Some(())
}

/// Try to extract a constant f64 value from a mathlex expression.
pub(super) fn mathlex_to_f64(expr: &mathlex::Expression) -> Option<f64> {
    match &expr.kind {
        mathlex::ExprKind::Integer(n) => Some(*n as f64),
        mathlex::ExprKind::Float(f) => Some(f.value()),
        mathlex::ExprKind::Unary {
            op: mathlex::UnaryOp::Neg,
            operand,
        } => mathlex_to_f64(operand).map(|v| -v),
        _ => None,
    }
}
