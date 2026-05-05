use super::*;
use crate::ast::{Expression, Variable};
use crate::numeric::trace::TechniqueTag;
use crate::ode::{FirstOrderODE, SecondOrderODE};

fn var(name: &str) -> Expression {
    Expression::Variable(Variable::new(name))
}

fn neg(expr: Expression) -> Expression {
    use crate::ast::UnaryOp;
    Expression::Unary(UnaryOp::Neg, Box::new(expr))
}

// ------------------------------------------------------------------
// OdeSolver trait tests
// ------------------------------------------------------------------

#[test]
fn can_solve_always_false() {
    use crate::ast::Equation;
    let solver = OdeSolver::new();
    let eq = Equation::new("test", Expression::Integer(0), Expression::Integer(0));
    assert!(!solver.can_solve(&eq));
}

#[test]
fn solve_trait_returns_unsupported() {
    use crate::ast::Equation;
    let solver = OdeSolver::new();
    let eq = Equation::new("test", Expression::Integer(0), Expression::Integer(0));
    let result = solver.solve(&eq, &Variable::new("x"));
    assert!(matches!(result, Err(SolverError::UnsupportedEquationType)));
}

// ------------------------------------------------------------------
// First-order ODE convenience function tests
// ------------------------------------------------------------------

#[test]
fn solve_first_order_separable_dy_equals_y() {
    // dy/dx = y  →  separable, general solution y = C·eˣ
    let ode = FirstOrderODE::new("y", "x", var("y"));
    let result = solve_ode_first_order(&ode);
    assert!(result.is_ok(), "Expected Ok, got {result:?}");
    let (solution, path) = result.unwrap();
    assert!(matches!(solution, Solution::Unique(_)));
    assert!(!path.steps().is_empty());
}

#[test]
fn solve_first_order_linear_dy_equals_minus_y() {
    // dy/dx = -y  →  linear (P = 1, Q = 0), solution y = C·e^(−x)
    let ode = FirstOrderODE::new("y", "x", neg(var("y")));
    let result = solve_ode_first_order(&ode);
    assert!(result.is_ok(), "Expected Ok, got {result:?}");
    let (solution, path) = result.unwrap();
    assert!(matches!(solution, Solution::Unique(_)));
    assert!(!path.steps().is_empty());
}

// ------------------------------------------------------------------
// Second-order ODE convenience function tests
// ------------------------------------------------------------------

#[test]
fn solve_second_order_homogeneous_y_pp_minus_y() {
    // y'' - y = 0  →  y = C₁·eˣ + C₂·e^(−x)
    let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 0.0, -1.0);
    let result = solve_ode_second_order(&ode);
    assert!(result.is_ok(), "Expected Ok, got {result:?}");
    let (solution, path) = result.unwrap();
    assert!(matches!(solution, Solution::Unique(_)));
    assert!(!path.steps().is_empty());
}

#[test]
fn solve_second_order_non_homogeneous_polynomial_forcing() {
    // y'' + y = x  →  particular solution y_p = x
    let ode = SecondOrderODE::new("y", "x", 1.0, 0.0, 1.0, var("x"));
    let result = solve_ode_second_order(&ode);
    assert!(result.is_ok(), "Expected Ok, got {result:?}");
    let (solution, path) = result.unwrap();
    assert!(matches!(solution, Solution::Unique(_)));
    assert!(!path.steps().is_empty());
}

#[test]
fn solve_second_order_non_homogeneous_exp_forcing() {
    // y'' - y = e^(2x)  →  particular solution y_p = (1/3)·e^(2x)
    use crate::ast::{BinaryOp, Expression, Function};
    let kx = Expression::Binary(
        BinaryOp::Mul,
        Box::new(Expression::Float(2.0)),
        Box::new(var("x")),
    );
    let forcing = Expression::Function(Function::Exp, vec![kx]);
    let ode = SecondOrderODE::new("y", "x", 1.0, 0.0, -1.0, forcing);
    let result = solve_ode_second_order(&ode);
    assert!(result.is_ok(), "Expected Ok, got {result:?}");
    let (solution, _path) = result.unwrap();
    assert!(matches!(solution, Solution::Unique(_)));
}

#[test]
fn solve_second_order_non_homogeneous_trig_resonance() {
    // y'' + y = sin(x)  →  resonant (k=1, char roots ±i)
    use crate::ast::{Expression, Function};
    let forcing = Expression::Function(Function::Sin, vec![var("x")]);
    let ode = SecondOrderODE::new("y", "x", 1.0, 0.0, 1.0, forcing);
    let result = solve_ode_second_order(&ode);
    assert!(result.is_ok(), "Expected Ok, got {result:?}");
    let (solution, _path) = result.unwrap();
    assert!(matches!(solution, Solution::Unique(_)));
}

#[test]
fn solve_second_order_non_homogeneous_unsupported_returns_error() {
    // y'' + y = tan(x)  →  unsupported forcing type
    use crate::ast::{Expression, Function};
    let forcing = Expression::Function(Function::Tan, vec![var("x")]);
    let ode = SecondOrderODE::new("y", "x", 1.0, 0.0, 1.0, forcing);
    let result = solve_ode_second_order(&ode);
    assert!(
        matches!(result, Err(SolverError::CannotSolve(_))),
        "Expected CannotSolve, got {result:?}"
    );
}

// ------------------------------------------------------------------
// Resolution path content tests
// ------------------------------------------------------------------

#[test]
fn first_order_path_starts_with_classify_step() {
    // dy/dx = y → separable; first step must classify as separable ODE
    let ode = FirstOrderODE::new("y", "x", var("y"));
    let (_solution, trace) = solve_ode_first_order(&ode).unwrap();
    let first = trace.steps().first().expect("trace must have steps");
    assert_eq!(first.tag, TechniqueTag::SeparationOfVariables);
    assert!(first.detail.contains("order=first"));
    assert!(first.detail.contains("ode_type=separable"));
}

#[test]
fn second_order_path_starts_with_classify_and_contains_solve_steps() {
    // y'' - y = 0 → second-order homogeneous constant-coefficient; first
    // step tags classification as a characteristic-equation technique.
    let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 0.0, -1.0);
    let (_solution, trace) = solve_ode_second_order(&ode).unwrap();
    let first = trace.steps().first().expect("trace must have steps");
    assert_eq!(first.tag, TechniqueTag::CharacteristicEquation);
    assert!(first.detail.contains("order=second"));
    // Subsequent solving steps stay at calculus tier.
    let all_calculus = trace
        .steps()
        .iter()
        .all(|s| s.tag.difficulty() == crate::numeric::trace::TechniqueDifficulty::Calculus);
    assert!(all_calculus, "Expected every trace step at Calculus tier");
}

// ------------------------------------------------------------------
// Text-based ODE solving (solve_ode_from_text)
// ------------------------------------------------------------------

#[test]
fn solve_from_text_first_order_separable() {
    // dy/dx = y → separable, y = C·eˣ
    let result = solve_ode_from_text("dy/dx = y");
    assert!(result.is_ok(), "Expected Ok, got {result:?}");
    let (solution, path) = result.unwrap();
    assert!(matches!(solution, Solution::Unique(_)));
    assert!(!path.steps().is_empty());
}

#[test]
fn solve_from_text_first_order_linear() {
    // dy/dx = -y → linear, y = C·e^(-x)
    let result = solve_ode_from_text("dy/dx = -y");
    assert!(result.is_ok(), "Expected Ok, got {result:?}");
    let (solution, _) = result.unwrap();
    assert!(matches!(solution, Solution::Unique(_)));
}

#[test]
fn solve_from_text_second_order_homogeneous() {
    // d2y/dx2 - y = 0 → y = C₁·eˣ + C₂·e^(-x)
    let result = solve_ode_from_text("d2y/dx2 - y = 0");
    assert!(result.is_ok(), "Expected Ok, got {result:?}");
    let (solution, path) = result.unwrap();
    assert!(matches!(solution, Solution::Unique(_)));
    assert!(!path.steps().is_empty());
}

#[test]
fn solve_from_text_second_order_with_first_deriv() {
    // d2y/dx2 + 2*dy/dx + y = 0 → repeated root, y = (C₁ + C₂·x)·e^(-x)
    let result = solve_ode_from_text("d2y/dx2 + 2*dy/dx + y = 0");
    assert!(result.is_ok(), "Expected Ok, got {result:?}");
    let (solution, _) = result.unwrap();
    assert!(matches!(solution, Solution::Unique(_)));
}

#[test]
fn solve_from_text_not_an_ode() {
    let result = solve_ode_from_text("x + y = 0");
    assert!(matches!(result, Err(SolverError::CannotSolve(_))));
}

#[test]
fn solve_from_text_invalid_input() {
    let result = solve_ode_from_text("not valid math @@@");
    assert!(matches!(result, Err(SolverError::CannotSolve(_))));
}

#[test]
fn solve_from_text_diff_notation() {
    // diff(y, x) = y → same as dy/dx = y
    let result = solve_ode_from_text("diff(y, x) = y");
    assert!(result.is_ok(), "Expected Ok, got {result:?}");
    let (solution, _) = result.unwrap();
    assert!(matches!(solution, Solution::Unique(_)));
}

// ------------------------------------------------------------------
// LaTeX-based ODE solving (solve_ode_from_latex)
// ------------------------------------------------------------------

#[test]
fn solve_from_latex_first_order_separable() {
    // \frac{d}{dx}(y) = y → separable, y = C·eˣ
    let result = solve_ode_from_latex(r#"\frac{d}{dx}(y) = y"#);
    assert!(result.is_ok(), "Expected Ok, got {result:?}");
    let (solution, path) = result.unwrap();
    assert!(matches!(solution, Solution::Unique(_)));
    assert!(!path.steps().is_empty());
}

#[test]
fn solve_from_latex_first_order_linear() {
    // \frac{d}{dx}(y) = -y → linear, y = C·e^(-x)
    let result = solve_ode_from_latex(r#"\frac{d}{dx}(y) = -y"#);
    assert!(result.is_ok(), "Expected Ok, got {result:?}");
    let (solution, _) = result.unwrap();
    assert!(matches!(solution, Solution::Unique(_)));
}

#[test]
fn solve_from_latex_second_order_homogeneous() {
    // \frac{d^2}{dx^2}(y) - y = 0
    let result = solve_ode_from_latex(r#"\frac{d^2}{dx^2}(y) - y = 0"#);
    assert!(result.is_ok(), "Expected Ok, got {result:?}");
    let (solution, path) = result.unwrap();
    assert!(matches!(solution, Solution::Unique(_)));
    assert!(!path.steps().is_empty());
}

#[test]
fn solve_from_latex_second_order_with_first_deriv() {
    // \frac{d^2}{dx^2}(y) + 2\frac{d}{dx}(y) + y = 0
    let result = solve_ode_from_latex(r#"\frac{d^2}{dx^2}(y) + 2\frac{d}{dx}(y) + y = 0"#);
    assert!(result.is_ok(), "Expected Ok, got {result:?}");
    let (solution, _) = result.unwrap();
    assert!(matches!(solution, Solution::Unique(_)));
}

#[test]
fn solve_from_latex_not_an_ode() {
    let result = solve_ode_from_latex(r#"x + y = 0"#);
    assert!(matches!(result, Err(SolverError::CannotSolve(_))));
}

#[test]
fn solve_from_latex_invalid_input() {
    let result = solve_ode_from_latex(r#"\invalid{bad"#);
    assert!(matches!(result, Err(SolverError::CannotSolve(_))));
}

// ------------------------------------------------------------------
// Text vs LaTeX equivalence
// ------------------------------------------------------------------

#[test]
fn text_and_latex_produce_same_first_order_solution() {
    let (text_sol, _) = solve_ode_from_text("dy/dx = y").unwrap();
    let (latex_sol, _) = solve_ode_from_latex(r#"\frac{d}{dx}(y) = y"#).unwrap();

    // Both should produce Unique solutions
    let text_expr = match text_sol {
        Solution::Unique(e) => e,
        _ => panic!("text: expected Unique"),
    };
    let latex_expr = match latex_sol {
        Solution::Unique(e) => e,
        _ => panic!("latex: expected Unique"),
    };

    // Evaluate both at x=1 — they should give the same result
    let mut env = std::collections::HashMap::new();
    env.insert("x".to_string(), 1.0);
    env.insert("C".to_string(), 1.0);
    let text_val = text_expr.evaluate(&env);
    let latex_val = latex_expr.evaluate(&env);
    assert_eq!(text_val, latex_val, "text and latex solutions diverge");
}

#[test]
fn text_and_latex_produce_same_second_order_solution() {
    let (text_sol, _) = solve_ode_from_text("d2y/dx2 - y = 0").unwrap();
    let (latex_sol, _) = solve_ode_from_latex(r#"\frac{d^2}{dx^2}(y) - y = 0"#).unwrap();

    let text_expr = match text_sol {
        Solution::Unique(e) => e,
        _ => panic!("text: expected Unique"),
    };
    let latex_expr = match latex_sol {
        Solution::Unique(e) => e,
        _ => panic!("latex: expected Unique"),
    };

    let mut env = std::collections::HashMap::new();
    env.insert("x".to_string(), 1.0);
    env.insert("C_1".to_string(), 1.0);
    env.insert("C_2".to_string(), 1.0);
    let text_val = text_expr.evaluate(&env);
    let latex_val = latex_expr.evaluate(&env);
    assert_eq!(text_val, latex_val, "text and latex solutions diverge");
}
