//! Expression-AST walkers supporting the first-order ODE solvers.
//!
//! These helpers inspect or manipulate the legacy `Expression` tree at the
//! migration boundary — they are called after `FirstOrderODE::rhs_expr()` has
//! decompiled the canonical `Arc<Expr>` storage back to `Expression`. Porting
//! them to `Arc<Expr>` walkers is tracked as follow-up work under milestone
//! `Expr-migration` (task 10.4).

use crate::ast::{BinaryOp, Expression, Function, UnaryOp};
use crate::numeric::compile::{compile, decompile};
use crate::numeric::substitute::substitute as arc_substitute;
use crate::numeric::SymbolId;

/// Attempt to separate dy/dx = f(x,y) into g(x) * h(y).
///
/// Returns (g(x), h(y)) if separable, None otherwise.
pub(crate) fn try_separate(
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
pub(crate) fn extract_linear_coefficients(
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
///
/// Delegates to the `Arc<Expr>` walker [`crate::numeric::substitute::substitute`]
/// via compile/decompile at the boundary. Signature is preserved so callers
/// keep operating on `Expression` until they migrate.
pub(crate) fn substitute_var(expr: &Expression, var: &str, replacement: &Expression) -> Expression {
    let expr_arc = compile(expr);
    let replacement_arc = compile(replacement);
    let var_id = SymbolId::intern(var);
    let result = arc_substitute(&expr_arc, var_id, &replacement_arc);
    decompile(&result)
}

/// Try to solve an implicit relation for y explicitly.
/// For example: ln(y) = x + C => y = e^(x + C)
pub(crate) fn try_solve_implicit_for_y(
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
pub(crate) fn solve_for_constant(equation: &Expression, const_name: &str) -> Option<Expression> {
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
