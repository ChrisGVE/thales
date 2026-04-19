//! Probe: invert r(θ) = (2+cos(θ))·(1-cos(θ))^2/2 for θ(r).
//! Expected gap: cubic-in-cos(θ) with symbolic r coefficient is beyond
//! current solver (polynomial solver wants f64 coefficients).

use std::collections::HashMap;

use thales::ast::{BinaryOp, Equation, Expression, Variable};
use thales::parser::parse_expression;
use thales::solver::solve_for;

#[test]
fn probe_invert_r_of_theta() {
    // Build r = (2 + cos(theta)) * (1 - cos(theta))^2 / 2
    let rhs = parse_expression("(2 + cos(theta)) * (1 - cos(theta))^2 / 2").expect("parse rhs");
    let lhs = Expression::Variable(Variable::new("r"));
    let eq = Equation::new("polar_curve", lhs, rhs);

    let known: HashMap<String, f64> = HashMap::new();
    let res = solve_for(&eq, "theta", &known);
    eprintln!("solve_for(theta) = {:?}", res);

    match res {
        Ok(path) => eprintln!("Result: {}", path.result),
        Err(e) => eprintln!("Error: {:?}", e),
    }
}

#[test]
fn probe_invert_expanded_cubic_in_cos() {
    // Expanded: r = (cos(theta)^3 - 3*cos(theta) + 2) / 2
    let rhs = parse_expression("(cos(theta)^3 - 3*cos(theta) + 2) / 2").expect("parse rhs");
    let lhs = Expression::Variable(Variable::new("r"));
    let eq = Equation::new("polar_curve_expanded", lhs, rhs);

    let known: HashMap<String, f64> = HashMap::new();
    let res = solve_for(&eq, "theta", &known);
    eprintln!("solve_for(theta) expanded = {:?}", res);
}

#[test]
fn probe_invert_single_cos_term() {
    // Baseline: r = 2 + cos(theta). Should invert via arccos.
    let rhs = parse_expression("2 + cos(theta)").unwrap();
    let lhs = Expression::Variable(Variable::new("r"));
    let eq = Equation::new("single_cos", lhs, rhs);

    let known: HashMap<String, f64> = HashMap::new();
    let res = solve_for(&eq, "theta", &known);
    eprintln!("solve_for(theta) single cos = {:?}", res);
    assert!(res.is_ok(), "baseline arccos inversion should succeed");
}

#[test]
fn probe_invert_cubic_in_u_symbolic_r() {
    // Reduce to: u^3 - 3*u + (2 - 2*r) = 0, solve for u with symbolic r.
    // Tests cubic solver path with symbolic coefficient.
    let rhs = parse_expression("u^3 - 3*u + (2 - 2*r)").unwrap();
    let lhs = Expression::Integer(0);
    let eq = Equation::new("cubic_symbolic", lhs, rhs);

    let known: HashMap<String, f64> = HashMap::new();
    let res = solve_for(&eq, "u", &known);
    eprintln!("solve_for(u) cubic with symbolic r = {:?}", res);
    match res {
        Ok(path) => eprintln!("Result: {}", path.result),
        Err(e) => eprintln!("Error: {:?}", e),
    }
}
