//! Inequality solving algorithms.

use crate::ast::{BinaryOp, Expression, UnaryOp, Variable};
use std::collections::HashMap;

use super::types::{Bound, Inequality, InequalityError, InequalityResult, IntervalSolution};

pub fn solve_inequality(inequality: &Inequality, var: &str) -> InequalityResult {
    // Move everything to one side: f(x) <op> 0
    let (expr, is_strict, is_greater) = normalize_inequality(inequality);

    // Check if the variable appears in the expression
    if !expr.contains_variable(var) {
        // Constant inequality - evaluate
        return solve_constant_inequality(&expr, is_strict, is_greater);
    }

    // Try to determine polynomial degree
    let degree = estimate_polynomial_degree(&expr, var);

    match degree {
        Some(1) => solve_linear_inequality(&expr, var, is_strict, is_greater),
        Some(2) => solve_quadratic_inequality(&expr, var, is_strict, is_greater),
        Some(n) if n > 2 => {
            // Higher degree - try to factor or use numerical methods
            Err(InequalityError::CannotSolve(format!(
                "Polynomial degree {} is too high",
                n
            )))
        }
        _ => Err(InequalityError::NonPolynomial(
            "Cannot determine polynomial degree".to_string(),
        )),
    }
}

/// Normalize inequality to f(x) <op> 0 form.
/// Returns (expression, is_strict, is_greater_than_zero)
fn normalize_inequality(ineq: &Inequality) -> (Expression, bool, bool) {
    match ineq {
        // left < right => left - right < 0
        Inequality::LessThan(left, right) => {
            let diff = Expression::Binary(
                BinaryOp::Sub,
                Box::new(left.clone()),
                Box::new(right.clone()),
            );
            (diff, true, false)
        }
        // left ≤ right => left - right ≤ 0
        Inequality::LessEqual(left, right) => {
            let diff = Expression::Binary(
                BinaryOp::Sub,
                Box::new(left.clone()),
                Box::new(right.clone()),
            );
            (diff, false, false)
        }
        // left > right => left - right > 0
        Inequality::GreaterThan(left, right) => {
            let diff = Expression::Binary(
                BinaryOp::Sub,
                Box::new(left.clone()),
                Box::new(right.clone()),
            );
            (diff, true, true)
        }
        // left ≥ right => left - right ≥ 0
        Inequality::GreaterEqual(left, right) => {
            let diff = Expression::Binary(
                BinaryOp::Sub,
                Box::new(left.clone()),
                Box::new(right.clone()),
            );
            (diff, false, true)
        }
    }
}

/// Solve a constant inequality (no variables).
fn solve_constant_inequality(
    expr: &Expression,
    is_strict: bool,
    is_greater: bool,
) -> InequalityResult {
    // Try to evaluate the expression with no variables
    let vars: HashMap<String, f64> = HashMap::new();
    let val = expr.evaluate(&vars);

    match val {
        Some(v) => {
            let is_positive: bool = v > 0.0;
            let is_zero: bool = v.abs() < 1e-15;

            let satisfied = if is_greater {
                if is_strict {
                    is_positive && !is_zero
                } else {
                    is_positive || is_zero
                }
            } else {
                if is_strict {
                    !is_positive && !is_zero
                } else {
                    !is_positive || is_zero
                }
            };

            if satisfied {
                Ok(IntervalSolution::AllReals)
            } else {
                Ok(IntervalSolution::Empty)
            }
        }
        None => Err(InequalityError::CannotSolve(
            "Cannot evaluate constant expression".to_string(),
        )),
    }
}

/// Estimate the polynomial degree of an expression in a variable.
fn estimate_polynomial_degree(expr: &Expression, var: &str) -> Option<u32> {
    match expr {
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Rational(_)
        | Expression::Complex(_)
        | Expression::Constant(_) => Some(0),

        Expression::Variable(v) => {
            if v.name == var {
                Some(1)
            } else {
                Some(0)
            }
        }

        Expression::Unary(UnaryOp::Neg, inner) => estimate_polynomial_degree(inner, var),

        Expression::Binary(op, left, right) => {
            let left_deg = estimate_polynomial_degree(left, var)?;
            let right_deg = estimate_polynomial_degree(right, var)?;

            match op {
                BinaryOp::Add | BinaryOp::Sub => Some(left_deg.max(right_deg)),
                BinaryOp::Mul => Some(left_deg + right_deg),
                BinaryOp::Div => {
                    // f(x)/c is still polynomial if c doesn't contain x
                    if !right.contains_variable(var) {
                        Some(left_deg)
                    } else {
                        None // Rational function
                    }
                }
                BinaryOp::Mod => None,
            }
        }

        Expression::Power(base, exp) => {
            if !base.contains_variable(var) {
                Some(0)
            } else if !exp.contains_variable(var) {
                // x^n where n is constant
                let empty_vars: HashMap<String, f64> = HashMap::new();
                if let Some(n) = exp.evaluate(&empty_vars) {
                    if n >= 0.0 && (n - n.floor()).abs() < 1e-10 {
                        let base_deg = estimate_polynomial_degree(base, var)?;
                        Some(base_deg * (n as u32))
                    } else {
                        None // Non-integer exponent
                    }
                } else {
                    None
                }
            } else {
                None // Variable in exponent
            }
        }

        Expression::Function(_, _) => None, // Transcendental
        _ => None,
    }
}

/// Solve a linear inequality ax + b <op> 0.
fn solve_linear_inequality(
    expr: &Expression,
    var: &str,
    is_strict: bool,
    is_greater: bool,
) -> InequalityResult {
    // Extract coefficients a and b from ax + b
    let (a, b) = extract_linear_coefficients(expr, var)?;

    // ax + b <op> 0
    // x <op> -b/a (flipping if a < 0)

    let neg_b_over_a = if a.abs() < 1e-15 {
        // Degenerate case: b <op> 0
        if is_greater {
            if is_strict {
                return if b > 0.0 {
                    Ok(IntervalSolution::AllReals)
                } else {
                    Ok(IntervalSolution::Empty)
                };
            } else {
                return if b >= 0.0 {
                    Ok(IntervalSolution::AllReals)
                } else {
                    Ok(IntervalSolution::Empty)
                };
            }
        } else {
            if is_strict {
                return if b < 0.0 {
                    Ok(IntervalSolution::AllReals)
                } else {
                    Ok(IntervalSolution::Empty)
                };
            } else {
                return if b <= 0.0 {
                    Ok(IntervalSolution::AllReals)
                } else {
                    Ok(IntervalSolution::Empty)
                };
            }
        }
    } else {
        -b / a
    };

    let threshold = Expression::Float(neg_b_over_a);

    // Flip inequality direction if a < 0
    let flip = a < 0.0;
    let effective_greater = if flip { !is_greater } else { is_greater };

    if effective_greater {
        if is_strict {
            Ok(IntervalSolution::greater_than(threshold))
        } else {
            Ok(IntervalSolution::greater_equal(threshold))
        }
    } else {
        if is_strict {
            Ok(IntervalSolution::less_than(threshold))
        } else {
            Ok(IntervalSolution::less_equal(threshold))
        }
    }
}

/// Extract linear coefficients (a, b) from expression ax + b.
fn extract_linear_coefficients(
    expr: &Expression,
    var: &str,
) -> Result<(f64, f64), InequalityError> {
    // Simplified implementation - evaluate at x=0 to get b, at x=1 to get a+b
    let mut vars = HashMap::new();

    vars.insert(var.to_string(), 0.0);
    let b = expr
        .evaluate(&vars)
        .ok_or_else(|| InequalityError::CannotSolve("Cannot evaluate at x=0".to_string()))?;

    vars.insert(var.to_string(), 1.0);
    let a_plus_b = expr
        .evaluate(&vars)
        .ok_or_else(|| InequalityError::CannotSolve("Cannot evaluate at x=1".to_string()))?;

    Ok((a_plus_b - b, b))
}

/// Solve a quadratic inequality ax² + bx + c <op> 0.
fn solve_quadratic_inequality(
    expr: &Expression,
    var: &str,
    is_strict: bool,
    is_greater: bool,
) -> InequalityResult {
    // Extract coefficients
    let (a, b, c) = extract_quadratic_coefficients(expr, var)?;

    if a.abs() < 1e-15 {
        // Actually linear
        let linear_expr = Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Float(b)),
                Box::new(Expression::Variable(Variable::new(var))),
            )),
            Box::new(Expression::Float(c)),
        );
        return solve_linear_inequality(&linear_expr, var, is_strict, is_greater);
    }

    let discriminant = b * b - 4.0 * a * c;

    if discriminant < -1e-15 {
        // No real roots
        // ax² + bx + c has constant sign (same as a)
        let parabola_positive = a > 0.0;

        if is_greater {
            if parabola_positive {
                Ok(IntervalSolution::AllReals)
            } else {
                Ok(IntervalSolution::Empty)
            }
        } else {
            if parabola_positive {
                Ok(IntervalSolution::Empty)
            } else {
                Ok(IntervalSolution::AllReals)
            }
        }
    } else if discriminant.abs() < 1e-15 {
        // One double root
        let root = -b / (2.0 * a);
        let root_expr = Expression::Float(root);
        let parabola_positive = a > 0.0;

        // Parabola touches x-axis at root
        if is_greater {
            if is_strict {
                // > 0: all x except the root
                if parabola_positive {
                    Ok(IntervalSolution::Union(vec![
                        IntervalSolution::less_than(root_expr.clone()),
                        IntervalSolution::greater_than(root_expr),
                    ]))
                } else {
                    Ok(IntervalSolution::Empty)
                }
            } else {
                // >= 0
                if parabola_positive {
                    Ok(IntervalSolution::AllReals)
                } else {
                    // Only at the single point x = root
                    Ok(IntervalSolution::closed_interval(
                        root_expr.clone(),
                        root_expr,
                    ))
                }
            }
        } else {
            if is_strict {
                // < 0
                if parabola_positive {
                    Ok(IntervalSolution::Empty)
                } else {
                    Ok(IntervalSolution::Union(vec![
                        IntervalSolution::less_than(root_expr.clone()),
                        IntervalSolution::greater_than(root_expr),
                    ]))
                }
            } else {
                // <= 0
                if parabola_positive {
                    Ok(IntervalSolution::closed_interval(
                        root_expr.clone(),
                        root_expr,
                    ))
                } else {
                    Ok(IntervalSolution::AllReals)
                }
            }
        }
    } else {
        // Two distinct real roots
        let sqrt_disc = discriminant.sqrt();
        let r1 = (-b - sqrt_disc) / (2.0 * a);
        let r2 = (-b + sqrt_disc) / (2.0 * a);

        // Ensure r1 < r2
        let (root1, root2) = if r1 < r2 { (r1, r2) } else { (r2, r1) };
        let root1_expr = Expression::Float(root1);
        let root2_expr = Expression::Float(root2);

        let parabola_positive_outside = a > 0.0;

        // Test intervals: (-∞, r1), (r1, r2), (r2, +∞)
        // For a > 0: positive outside roots, negative between
        // For a < 0: negative outside roots, positive between

        if is_greater {
            if parabola_positive_outside {
                // > 0 outside roots
                if is_strict {
                    Ok(IntervalSolution::Union(vec![
                        IntervalSolution::less_than(root1_expr),
                        IntervalSolution::greater_than(root2_expr),
                    ]))
                } else {
                    Ok(IntervalSolution::Union(vec![
                        IntervalSolution::less_equal(root1_expr),
                        IntervalSolution::greater_equal(root2_expr),
                    ]))
                }
            } else {
                // > 0 between roots
                if is_strict {
                    Ok(IntervalSolution::open_interval(root1_expr, root2_expr))
                } else {
                    Ok(IntervalSolution::closed_interval(root1_expr, root2_expr))
                }
            }
        } else {
            if parabola_positive_outside {
                // < 0 between roots
                if is_strict {
                    Ok(IntervalSolution::open_interval(root1_expr, root2_expr))
                } else {
                    Ok(IntervalSolution::closed_interval(root1_expr, root2_expr))
                }
            } else {
                // < 0 outside roots
                if is_strict {
                    Ok(IntervalSolution::Union(vec![
                        IntervalSolution::less_than(root1_expr),
                        IntervalSolution::greater_than(root2_expr),
                    ]))
                } else {
                    Ok(IntervalSolution::Union(vec![
                        IntervalSolution::less_equal(root1_expr),
                        IntervalSolution::greater_equal(root2_expr),
                    ]))
                }
            }
        }
    }
}

/// Extract quadratic coefficients (a, b, c) from ax² + bx + c.
fn extract_quadratic_coefficients(
    expr: &Expression,
    var: &str,
) -> Result<(f64, f64, f64), InequalityError> {
    // Evaluate at three points to determine coefficients
    let mut vars = HashMap::new();

    vars.insert(var.to_string(), 0.0);
    let f0 = expr
        .evaluate(&vars)
        .ok_or_else(|| InequalityError::CannotSolve("Cannot evaluate at x=0".to_string()))?;

    vars.insert(var.to_string(), 1.0);
    let f1 = expr
        .evaluate(&vars)
        .ok_or_else(|| InequalityError::CannotSolve("Cannot evaluate at x=1".to_string()))?;

    vars.insert(var.to_string(), -1.0);
    let f_1 = expr
        .evaluate(&vars)
        .ok_or_else(|| InequalityError::CannotSolve("Cannot evaluate at x=-1".to_string()))?;

    // f(0) = c
    // f(1) = a + b + c
    // f(-1) = a - b + c
    // So: c = f(0)
    //     a + b = f(1) - c
    //     a - b = f(-1) - c
    //     a = ((f(1) - c) + (f(-1) - c)) / 2
    //     b = ((f(1) - c) - (f(-1) - c)) / 2

    let c = f0;
    let a = ((f1 - c) + (f_1 - c)) / 2.0;
    let b = ((f1 - c) - (f_1 - c)) / 2.0;

    Ok((a, b, c))
}
