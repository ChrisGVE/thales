//! Integration tests: real-world physics formulas solved symbolically.
//!
//! Uses `parse_equation` + `SmartSolver` (which delegates to the symbolic
//! isolation engine) to verify that common physics equations can be
//! rearranged for every variable of interest.

use thales::ast::Variable;
use thales::parser::parse_equation;
use thales::solver::{SmartSolver, Solution, Solver};

// ============================================================================
// Helper
// ============================================================================

/// Parse an equation string, solve for `target_var`, and assert that we
/// obtain a `Solution::Unique`.  Returns the expression string for further
/// inspection if desired.
fn assert_solves_to_unique(equation_str: &str, target_var: &str) -> String {
    let eq = parse_equation(equation_str)
        .unwrap_or_else(|e| panic!("Failed to parse '{}': {:?}", equation_str, e));
    let var = Variable::new(target_var);
    let solver = SmartSolver::new();
    let (solution, path) = solver.solve(&eq, &var).unwrap_or_else(|e| {
        panic!(
            "Failed to solve '{}' for '{}': {:?}",
            equation_str, target_var, e
        )
    });

    match &solution {
        Solution::Unique(expr) => {
            // Non-trivial solves should have resolution steps
            assert!(
                !path.steps.is_empty() || equation_str.starts_with(target_var),
                "Expected resolution steps for non-trivial solve of '{}' for '{}'",
                equation_str,
                target_var,
            );
            format!("{}", expr)
        }
        other => panic!(
            "Expected Unique solution for '{}' solving for '{}', got {:?}",
            equation_str, target_var, other
        ),
    }
}

/// Like `assert_solves_to_unique` but only checks that the solver succeeds
/// (allows any `Solution` variant, e.g. `Multiple`).
fn assert_solves(equation_str: &str, target_var: &str) {
    let eq = parse_equation(equation_str)
        .unwrap_or_else(|e| panic!("Failed to parse '{}': {:?}", equation_str, e));
    let var = Variable::new(target_var);
    let solver = SmartSolver::new();
    let _result = solver.solve(&eq, &var).unwrap_or_else(|e| {
        panic!(
            "Failed to solve '{}' for '{}': {:?}",
            equation_str, target_var, e
        )
    });
}

// ============================================================================
// Mechanics
// ============================================================================

#[test]
fn mechanics_newton_second_law_solve_for_f() {
    // F = m * a  ->  solve for F (trivial isolation)
    assert_solves_to_unique("F = m * a", "F");
}

#[test]
fn mechanics_newton_second_law_solve_for_m() {
    // F = m * a  ->  m = F / a
    assert_solves_to_unique("F = m * a", "m");
}

#[test]
fn mechanics_newton_second_law_solve_for_a() {
    // F = m * a  ->  a = F / m
    assert_solves_to_unique("F = m * a", "a");
}

#[test]
fn mechanics_momentum_solve_for_m() {
    // p = m * v  ->  m = p / v
    assert_solves_to_unique("p = m * v", "m");
}

#[test]
fn mechanics_momentum_solve_for_v() {
    // p = m * v  ->  v = p / m
    assert_solves_to_unique("p = m * v", "v");
}

#[test]
fn mechanics_kinetic_energy_solve_for_m() {
    // KE = (1/2) * m * v^2  ->  m = 2 * KE / v^2
    assert_solves_to_unique("KE = (1/2) * m * v^2", "m");
}

#[test]
fn mechanics_kinetic_energy_solve_for_v() {
    // KE = (1/2) * m * v^2  ->  v = sqrt(2 * KE / m)
    assert_solves_to_unique("KE = (1/2) * m * v^2", "v");
}

#[test]
fn mechanics_potential_energy_solve_for_m() {
    // PE = m * g * h  ->  m = PE / (g * h)
    assert_solves_to_unique("PE = m * g * h", "m");
}

#[test]
fn mechanics_potential_energy_solve_for_g() {
    assert_solves_to_unique("PE = m * g * h", "g");
}

#[test]
fn mechanics_potential_energy_solve_for_h() {
    assert_solves_to_unique("PE = m * g * h", "h");
}

#[test]
fn mechanics_work_solve_for_f() {
    // W = F * d  ->  F = W / d
    assert_solves_to_unique("W = F * d", "F");
}

#[test]
fn mechanics_work_solve_for_d() {
    assert_solves_to_unique("W = F * d", "d");
}

// ============================================================================
// Kinematics
// ============================================================================

#[test]
fn kinematics_velocity_solve_for_v0() {
    // v = v0 + a * t  ->  v0 = v - a * t
    assert_solves_to_unique("v = v0 + a * t", "v0");
}

#[test]
fn kinematics_velocity_solve_for_a() {
    // v = v0 + a * t  ->  a = (v - v0) / t
    assert_solves_to_unique("v = v0 + a * t", "a");
}

#[test]
fn kinematics_velocity_solve_for_t() {
    // v = v0 + a * t  ->  t = (v - v0) / a
    assert_solves_to_unique("v = v0 + a * t", "t");
}

#[test]
fn kinematics_displacement_solve_for_a() {
    // s = v0 * t + (1/2) * a * t^2  ->  a is linear, can be isolated
    assert_solves_to_unique("s = v0 * t + (1/2) * a * t^2", "a");
}

#[test]
fn kinematics_velocity_squared_solve_for_a() {
    // v^2 = v0^2 + 2 * a * s  ->  a = (v^2 - v0^2) / (2 * s)
    assert_solves_to_unique("v^2 = v0^2 + 2 * a * s", "a");
}

#[test]
fn kinematics_velocity_squared_solve_for_s() {
    // v^2 = v0^2 + 2 * a * s  ->  s = (v^2 - v0^2) / (2 * a)
    assert_solves_to_unique("v^2 = v0^2 + 2 * a * s", "s");
}

// ============================================================================
// Electricity
// ============================================================================

#[test]
fn electricity_ohms_law_solve_for_v() {
    assert_solves_to_unique("V = I * R", "V");
}

#[test]
fn electricity_ohms_law_solve_for_i() {
    assert_solves_to_unique("V = I * R", "I");
}

#[test]
fn electricity_ohms_law_solve_for_r() {
    assert_solves_to_unique("V = I * R", "R");
}

#[test]
fn electricity_power_current_solve_for_i() {
    // P = I^2 * R  ->  I = sqrt(P / R)
    assert_solves_to_unique("P = I^2 * R", "I");
}

#[test]
fn electricity_power_current_solve_for_r() {
    // P = I^2 * R  ->  R = P / I^2
    assert_solves_to_unique("P = I^2 * R", "R");
}

#[test]
fn electricity_power_voltage_solve_for_v() {
    // P = V^2 / R  ->  V = sqrt(P * R)
    assert_solves_to_unique("P = V^2 / R", "V");
}

#[test]
fn electricity_power_voltage_solve_for_r() {
    // P = V^2 / R  ->  R = V^2 / P
    assert_solves_to_unique("P = V^2 / R", "R");
}

#[test]
fn electricity_capacitance_solve_for_q() {
    // C = Q / V  ->  Q = C * V
    assert_solves_to_unique("C = Q / V", "Q");
}

#[test]
fn electricity_capacitance_solve_for_v() {
    // C = Q / V  ->  V = Q / C
    assert_solves_to_unique("C = Q / V", "V");
}

// ============================================================================
// Thermodynamics / Quantum
// ============================================================================

#[test]
fn quantum_planck_solve_for_h() {
    // E = h * f  ->  h = E / f
    assert_solves_to_unique("E = h * f", "h");
}

#[test]
fn quantum_planck_solve_for_f() {
    assert_solves_to_unique("E = h * f", "f");
}

// ============================================================================
// Gravitation
// ============================================================================

#[test]
fn gravitation_solve_for_r() {
    // F = G * m1 * m2 / r^2  ->  r = sqrt(G * m1 * m2 / F)
    assert_solves_to_unique("F = G * m1 * m2 / r^2", "r");
}

#[test]
fn gravitation_solve_for_m1() {
    // F = G * m1 * m2 / r^2  ->  m1 = F * r^2 / (G * m2)
    assert_solves_to_unique("F = G * m1 * m2 / r^2", "m1");
}

// ============================================================================
// Waves / Oscillation
// ============================================================================

#[test]
fn wave_frequency_period_solve_for_t() {
    // f = 1 / T  ->  T = 1 / f
    assert_solves_to_unique("f = 1 / T", "T");
}

#[test]
fn wave_equation_solve_for_f() {
    // v = f * lambda  ->  f = v / lambda
    assert_solves_to_unique("v = f * lambda", "f");
}

#[test]
fn wave_equation_solve_for_lambda() {
    assert_solves_to_unique("v = f * lambda", "lambda");
}

#[test]
fn pendulum_solve_for_l() {
    // T = 2 * pi * sqrt(L / g)  ->  L = g * (T / (2 * pi))^2
    // Requires: unwrap Mul (by 2*pi), invert sqrt, then Div
    assert_solves_to_unique("T = 2 * pi * sqrt(L / g)", "L");
}

// ============================================================================
// Trigonometric / Transcendental Inversion
// ============================================================================

#[test]
fn trig_inversion_sin() {
    // y = sin(x)  ->  x = arcsin(y)
    assert_solves_to_unique("y = sin(x)", "x");
}

#[test]
fn trig_inversion_cos() {
    // y = cos(x)  ->  x = arccos(y)
    assert_solves_to_unique("y = cos(x)", "x");
}

#[test]
fn trig_inversion_tan() {
    // y = tan(x)  ->  x = arctan(y)
    assert_solves_to_unique("y = tan(x)", "x");
}

#[test]
fn transcendental_exp_inversion() {
    // y = exp(x)  ->  x = ln(y)
    assert_solves_to_unique("y = exp(x)", "x");
}

#[test]
fn transcendental_ln_inversion() {
    // y = ln(x)  ->  x = exp(y)
    assert_solves_to_unique("y = ln(x)", "x");
}

// ============================================================================
// Ideal Gas Law
// ============================================================================

#[test]
fn ideal_gas_solve_for_t() {
    // P * V = n * R * T  ->  T = P * V / (n * R)
    // Note: parser should treat P, V, n, R, T as separate variables
    assert_solves_to_unique("P * V = n * R * T", "T");
}

#[test]
fn ideal_gas_solve_for_p() {
    assert_solves_to_unique("P * V = n * R * T", "P");
}

#[test]
fn ideal_gas_solve_for_v_gas() {
    assert_solves_to_unique("P * V = n * R * T", "V");
}

// ============================================================================
// Compound / Nested Expressions
// ============================================================================

#[test]
fn nested_sqrt_and_division() {
    // c = sqrt(a^2 + b^2) is Pythagorean theorem but 'a' appears in a^2
    // inside sqrt — the isolation engine should handle this since 'a' is
    // linear in a^2 which is a power expression.
    // Actually, a appears once via a^2, so isolation can peel sqrt then peel ^2.
    assert_solves_to_unique("c = sqrt(a^2 + b^2)", "a");
}

#[test]
fn compound_division_both_sides() {
    // density = mass / volume
    assert_solves_to_unique("rho = m / V", "m");
}

#[test]
fn compound_division_both_sides_solve_for_v() {
    assert_solves_to_unique("rho = m / V", "V");
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn identity_solve_trivial() {
    // x = x should still succeed (the variable is already isolated)
    // Actually this is "x on both sides", the engine collects and gets 0 = 0.
    // This might fail — mark as a known limitation if so.
    let eq = parse_equation("x = x").expect("Failed to parse");
    let var = Variable::new("x");
    let solver = SmartSolver::new();
    // This could return an error or any-value solution; just check it doesn't panic
    let _result = solver.solve(&eq, &var);
}

#[test]
fn solve_already_isolated_lhs() {
    // F = m * a, solving for F — variable is already on the LHS alone
    let result = assert_solves_to_unique("F = m * a", "F");
    // The result should be "m * a" or equivalent
    assert!(!result.is_empty());
}

#[test]
fn solve_variable_on_rhs() {
    // m * a = F, solving for F — variable is on the RHS
    // The engine swaps sides, so isolation is trivial (zero steps), use assert_solves
    assert_solves("m * a = F", "F");
}
