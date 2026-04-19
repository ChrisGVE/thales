//! ODE solution verification helpers.
//!
//! Used by `solve_ivp`-style tests per the Expr-migration milestone
//! (decision 2b): tighten from lax `is_ok()` to:
//!
//! - **y-free**: the particular solution expression must not contain the
//!   dependent variable as a free Symbol. Any `Symbol(y)` in the tree is a
//!   bug — the solver failed to eliminate `y`.
//! - **IC satisfied**: substituting `x = x₀` into the particular solution
//!   must yield `y₀` (within ε for floating-point cases).

use std::collections::HashMap;

use crate::ast::Expression;

/// Assert that `expr` contains no free occurrence of the dependent variable
/// `y_var`.
///
/// Panics with a descriptive message if `y_var` appears as a `Symbol` in the
/// tree, which indicates the ODE solver failed to eliminate the dependent
/// variable when producing the explicit particular solution.
pub fn assert_y_free(expr: &Expression, y_var: &str) {
    assert!(
        !expr.contains_variable(y_var),
        "particular solution still contains dependent variable '{}': {}",
        y_var,
        expr
    );
}

/// Assert that the particular solution `expr` satisfies the initial condition
/// `y(x₀) = y₀` numerically within `eps`.
///
/// Panics if the substitution `x = x0` fails to evaluate, or if the result
/// differs from `y0` by more than `eps`.
pub fn assert_ic_satisfied(expr: &Expression, x_var: &str, x0: f64, y0: f64, eps: f64) {
    let mut env = HashMap::new();
    env.insert(x_var.to_string(), x0);
    let got = expr.evaluate(&env).unwrap_or_else(|| {
        panic!(
            "particular solution did not evaluate at {}={}: {}",
            x_var, x0, expr
        )
    });
    assert!(
        (got - y0).abs() < eps,
        "IC violated at {}={}: expected {}, got {}; expr={}",
        x_var,
        x0,
        y0,
        got,
        expr
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Function, Variable};

    fn var(n: &str) -> Expression {
        Expression::Variable(Variable::new(n))
    }

    #[test]
    fn y_free_passes_when_clean() {
        // e^x — no y
        let e = Expression::Function(Function::Exp, vec![var("x")]);
        assert_y_free(&e, "y");
    }

    #[test]
    #[should_panic(expected = "still contains dependent variable")]
    fn y_free_panics_when_y_present() {
        // y + 1
        let e = Expression::Binary(
            BinaryOp::Add,
            Box::new(var("y")),
            Box::new(Expression::Integer(1)),
        );
        assert_y_free(&e, "y");
    }

    #[test]
    fn ic_passes_for_e_x_at_zero() {
        // e^x evaluated at x=0 is 1
        let e = Expression::Function(Function::Exp, vec![var("x")]);
        assert_ic_satisfied(&e, "x", 0.0, 1.0, 1e-10);
    }

    #[test]
    #[should_panic(expected = "IC violated")]
    fn ic_panics_when_value_mismatches() {
        // e^x at x=0 is 1, not 2
        let e = Expression::Function(Function::Exp, vec![var("x")]);
        assert_ic_satisfied(&e, "x", 0.0, 2.0, 1e-10);
    }
}
