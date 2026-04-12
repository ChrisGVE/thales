//! Tests for the symbolic isolation engine.

use super::*;
use crate::ast::{BinaryOp, Expression, Function, SymbolicConstant, Variable};
use crate::resolution_path::ResolutionPathBuilder;
use std::collections::HashMap;

// ---- helpers ----

fn v(name: &str) -> Expression {
    Expression::Variable(Variable::new(name))
}

fn int(n: i64) -> Expression {
    Expression::Integer(n)
}

fn add(a: Expression, b: Expression) -> Expression {
    Expression::Binary(BinaryOp::Add, Box::new(a), Box::new(b))
}

#[allow(dead_code)]
fn sub(a: Expression, b: Expression) -> Expression {
    Expression::Binary(BinaryOp::Sub, Box::new(a), Box::new(b))
}

fn mul(a: Expression, b: Expression) -> Expression {
    Expression::Binary(BinaryOp::Mul, Box::new(a), Box::new(b))
}

fn div(a: Expression, b: Expression) -> Expression {
    Expression::Binary(BinaryOp::Div, Box::new(a), Box::new(b))
}

fn pow(base: Expression, exp: Expression) -> Expression {
    Expression::Power(Box::new(base), Box::new(exp))
}

fn func(f: Function, arg: Expression) -> Expression {
    Expression::Function(f, vec![arg])
}

fn pi() -> Expression {
    Expression::Constant(SymbolicConstant::Pi)
}

/// Run isolation and return the result expression.
fn isolate(lhs: Expression, rhs: Expression, var_name: &str) -> Expression {
    let variable = Variable::new(var_name);
    let path = ResolutionPathBuilder::new(lhs.clone());
    let (result, _) = symbolic_isolate(&lhs, &rhs, &variable, path)
        .unwrap_or_else(|e| panic!("Isolation failed for '{}': {:?}", var_name, e));
    result
}

/// Evaluate an expression with the given variable bindings.
fn eval(expr: &Expression, vals: &[(&str, f64)]) -> f64 {
    let map: HashMap<String, f64> = vals.iter().map(|(k, v)| (k.to_string(), *v)).collect();
    expr.evaluate(&map)
        .unwrap_or_else(|| panic!("Failed to evaluate: {:?}", expr))
}

// ---- tests ----

#[test]
fn basic_f_eq_m_mul_a_solve_a() {
    // F = m * a, solve for a -> a = F / m
    let result = isolate(v("F"), mul(v("m"), v("a")), "a");
    let val = eval(&result, &[("F", 10.0), ("m", 2.0)]);
    assert!((val - 5.0).abs() < 1e-10, "Expected 5.0, got {}", val);
}

#[test]
fn basic_f_eq_m_mul_a_solve_m() {
    // F = m * a, solve for m -> m = F / a
    let result = isolate(v("F"), mul(v("m"), v("a")), "m");
    let val = eval(&result, &[("F", 10.0), ("a", 2.0)]);
    assert!((val - 5.0).abs() < 1e-10, "Expected 5.0, got {}", val);
}

#[test]
fn basic_f_eq_m_mul_a_solve_f() {
    // F = m * a, solve for F -> F = m * a
    let result = isolate(v("F"), mul(v("m"), v("a")), "F");
    let val = eval(&result, &[("m", 3.0), ("a", 4.0)]);
    assert!((val - 12.0).abs() < 1e-10, "Expected 12.0, got {}", val);
}

#[test]
fn power_e_eq_m_c_squared_solve_m() {
    // E = m * c^2, solve for m -> m = E / c^2
    let rhs = mul(v("m"), pow(v("c"), int(2)));
    let result = isolate(v("E"), rhs, "m");
    // E=18, c=3 => m = 18/9 = 2
    let val = eval(&result, &[("E", 18.0), ("c", 3.0)]);
    assert!((val - 2.0).abs() < 1e-10, "Expected 2.0, got {}", val);
}

#[test]
fn power_e_eq_m_c_squared_solve_c() {
    // E = m * c^2, solve for c -> c = sqrt(E/m)
    let rhs = mul(v("m"), pow(v("c"), int(2)));
    let result = isolate(v("E"), rhs, "c");
    // E=18, m=2 => c = sqrt(9) = 3
    let val = eval(&result, &[("E", 18.0), ("m", 2.0)]);
    assert!((val - 3.0).abs() < 1e-10, "Expected 3.0, got {}", val);
}

#[test]
fn v_eq_v0_plus_a_mul_t_solve_t() {
    // v = v0 + a * t, solve for t -> t = (v - v0) / a
    let rhs = add(v("v0"), mul(v("a"), v("t")));
    let result = isolate(v("v"), rhs, "t");
    // v=20, v0=5, a=3 => t = 15/3 = 5
    let val = eval(&result, &[("v", 20.0), ("v0", 5.0), ("a", 3.0)]);
    assert!((val - 5.0).abs() < 1e-10, "Expected 5.0, got {}", val);
}

#[test]
fn v_eq_v0_plus_a_mul_t_solve_a() {
    // v = v0 + a * t, solve for a -> a = (v - v0) / t
    let rhs = add(v("v0"), mul(v("a"), v("t")));
    let result = isolate(v("v"), rhs, "a");
    let val = eval(&result, &[("v", 20.0), ("v0", 5.0), ("t", 3.0)]);
    assert!((val - 5.0).abs() < 1e-10, "Expected 5.0, got {}", val);
}

#[test]
fn v_eq_v0_plus_a_mul_t_solve_v0() {
    // v = v0 + a * t, solve for v0 -> v0 = v - a * t
    let rhs = add(v("v0"), mul(v("a"), v("t")));
    let result = isolate(v("v"), rhs, "v0");
    let val = eval(&result, &[("v", 20.0), ("a", 3.0), ("t", 5.0)]);
    assert!((val - 5.0).abs() < 1e-10, "Expected 5.0, got {}", val);
}

#[test]
fn kinematics_s_solve_for_a() {
    // s = u*t + (1/2)*a*t^2, solve for a
    // a = 2*(s - u*t) / t^2
    // Note: linear in a even though t^2 is present
    let half = Expression::Float(0.5);
    let rhs = add(
        mul(v("u"), v("t")),
        mul(mul(half, v("a")), pow(v("t"), int(2))),
    );
    let result = isolate(v("s"), rhs, "a");
    // s=30, u=5, t=2 => a = 2*(30-10)/4 = 2*20/4 = 10
    let val = eval(&result, &[("s", 30.0), ("u", 5.0), ("t", 2.0)]);
    assert!((val - 10.0).abs() < 1e-10, "Expected 10.0, got {}", val);
}

#[test]
fn gravity_solve_for_r() {
    // F = G*m1*m2/r^2, solve for r -> r = sqrt(G*m1*m2/F)
    let rhs = div(mul(mul(v("G"), v("m1")), v("m2")), pow(v("r"), int(2)));
    let result = isolate(v("F"), rhs, "r");
    // G*m1*m2 = 100, F=4 => r = sqrt(25) = 5
    let val = eval(
        &result,
        &[("F", 4.0), ("G", 10.0), ("m1", 5.0), ("m2", 2.0)],
    );
    assert!((val - 5.0).abs() < 1e-10, "Expected 5.0, got {}", val);
}

#[test]
fn sin_inversion() {
    // y = sin(x), solve for x -> x = arcsin(y)
    let result = isolate(v("y"), func(Function::Sin, v("x")), "x");
    // y = 0.5 => x = arcsin(0.5) ~ 0.5236
    let val = eval(&result, &[("y", 0.5)]);
    assert!(
        (val - 0.5_f64.asin()).abs() < 1e-10,
        "Expected arcsin(0.5), got {}",
        val
    );
}

#[test]
fn exp_inversion() {
    // y = exp(x), solve for x -> x = ln(y)
    let result = isolate(v("y"), func(Function::Exp, v("x")), "x");
    let val = eval(&result, &[("y", std::f64::consts::E)]);
    assert!((val - 1.0).abs() < 1e-10, "Expected 1.0, got {}", val);
}

#[test]
fn ln_inversion() {
    // y = ln(x), solve for x -> x = exp(y)
    let result = isolate(v("y"), func(Function::Ln, v("x")), "x");
    let val = eval(&result, &[("y", 1.0)]);
    assert!(
        (val - std::f64::consts::E).abs() < 1e-10,
        "Expected e, got {}",
        val
    );
}

#[test]
fn pendulum_period_solve_for_l() {
    // T = 2*pi*sqrt(L/g), solve for L -> L = g*(T/(2*pi))^2
    let rhs = mul(mul(int(2), pi()), func(Function::Sqrt, div(v("L"), v("g"))));
    let result = isolate(v("T"), rhs, "L");
    // For g=9.8, T=2*pi*sqrt(1/9.8)
    let g = 9.8_f64;
    let l_expected = 1.0;
    let t_val = 2.0 * std::f64::consts::PI * (l_expected / g).sqrt();
    let val = eval(&result, &[("T", t_val), ("g", g)]);
    assert!(
        (val - l_expected).abs() < 1e-8,
        "Expected {}, got {}",
        l_expected,
        val
    );
}

#[test]
fn simple_e_eq_h_mul_f_solve_f() {
    // E = h * f, solve for f -> f = E / h
    let result = isolate(v("E"), mul(v("h"), v("f")), "f");
    let val = eval(&result, &[("E", 12.0), ("h", 3.0)]);
    assert!((val - 4.0).abs() < 1e-10, "Expected 4.0, got {}", val);
}

#[test]
fn ideal_gas_solve_for_t() {
    // P*V = n*R*T, which we write as mul(P,V) = mul(mul(n,R),T)
    // Solve for T -> T = P*V / (n*R)
    let lhs = mul(v("P"), v("V"));
    let rhs = mul(mul(v("n"), v("R")), v("T"));
    let result = isolate(lhs, rhs, "T");
    // P=2, V=3, n=1, R=6 => T = 6/6 = 1
    let val = eval(&result, &[("P", 2.0), ("V", 3.0), ("n", 1.0), ("R", 6.0)]);
    assert!((val - 1.0).abs() < 1e-10, "Expected 1.0, got {}", val);
}

#[test]
fn parallel_resistance_solve_for_r1() {
    // R_par = R1*R2/(R1+R2), solve for R1
    // R1 appears in both numerator and denominator (cross-multiply case)
    let lhs = v("R_par");
    let rhs = div(mul(v("R1"), v("R2")), add(v("R1"), v("R2")));
    let result = isolate(lhs, rhs, "R1");
    // R_par=6, R2=12 => R1 = 6*12/(12-6) = 72/6 = 12... wait:
    // R_par = R1*R2/(R1+R2) => R_par*(R1+R2) = R1*R2
    // => R_par*R1 + R_par*R2 = R1*R2
    // => R1*(R_par - R2) = -R_par*R2
    // => R1 = R_par*R2/(R2 - R_par)
    // R_par=6, R2=12 => R1 = 6*12/(12-6) = 72/6 = 12
    let val = eval(&result, &[("R_par", 6.0), ("R2", 12.0)]);
    assert!((val - 12.0).abs() < 1e-10, "Expected 12.0, got {}", val);
}

#[test]
fn cross_multiply_simple_a_over_x_eq_b() {
    // a/x = b, solve for x → x = a/b
    let lhs = div(v("a"), v("x"));
    let rhs = v("b");
    let result = isolate(lhs, rhs, "x");
    let val = eval(&result, &[("a", 10.0), ("b", 2.0)]);
    assert!((val - 5.0).abs() < 1e-10, "Expected 5.0, got {}", val);
}

#[test]
fn cross_multiply_x_over_x_plus_1_eq_c() {
    // x/(x+1) = c, solve for x → x = c/(1-c)
    let lhs = div(v("x"), add(v("x"), int(1)));
    let rhs = v("c");
    let result = isolate(lhs, rhs, "x");
    // c=0.5 → x = 0.5/0.5 = 1
    let val = eval(&result, &[("c", 0.5)]);
    assert!((val - 1.0).abs() < 1e-10, "Expected 1.0, got {}", val);
}

#[test]
fn cross_multiply_with_different_values() {
    // R_par = R1*R2/(R1+R2), solve for R1 with different values
    let lhs = v("R_par");
    let rhs = div(mul(v("R1"), v("R2")), add(v("R1"), v("R2")));
    let result = isolate(lhs, rhs, "R1");
    // R_par=4, R2=12 → R1 = 4*12/(12-4) = 48/8 = 6
    let val = eval(&result, &[("R_par", 4.0), ("R2", 12.0)]);
    assert!((val - 6.0).abs() < 1e-10, "Expected 6.0, got {}", val);
}

#[test]
fn extraneous_solution_detected() {
    // x/(x-1) = 1/(x-1), solving for x
    // Cross-multiply: x = 1, but x=1 makes denominator (x-1) = 0
    // The solver should reject this as extraneous.
    let lhs = div(v("x"), sub(v("x"), int(1)));
    let rhs = div(int(1), sub(v("x"), int(1)));
    let variable = Variable::new("x");
    let path = ResolutionPathBuilder::new(lhs.clone());
    let result = symbolic_isolate(&lhs, &rhs, &variable, path);
    assert!(
        result.is_err(),
        "Expected error for extraneous solution, but isolation succeeded"
    );
}

#[test]
fn variable_not_found() {
    let variable = Variable::new("z");
    let path = ResolutionPathBuilder::new(v("x"));
    let result = symbolic_isolate(&v("x"), &v("y"), &variable, path);
    assert!(result.is_err());
}

#[test]
fn resolution_path_has_steps() {
    let lhs = v("F");
    let rhs = mul(v("m"), v("a"));
    let variable = Variable::new("a");
    let path = ResolutionPathBuilder::new(lhs.clone());
    let (_, final_path) = symbolic_isolate(&lhs, &rhs, &variable, path).unwrap();
    let resolution = final_path.finish(v("a"));
    assert!(
        resolution.step_count() > 0,
        "Resolution path should have at least one step"
    );
}
