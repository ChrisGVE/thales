//! Stress tests: three-tier, four-tier, five-tier, and six-tier combinations.
//!
//! - 20 three-tier combos × 10 tests = 200 tests
//! - 15 four-tier combos × 10 tests = 150 tests
//! - 6 five-tier combos × 10 tests = 60 tests
//! - 1 six-tier combo × 10 tests = 10 tests
//!
//! T1-T4 combinations pass. Many T5 tests also pass because the parser treats
//! calculus notation (integral/d) as algebraic expressions, enabling symbolic
//! solving. Tests requiring true calculus evaluation remain #[ignore].

use thales::ast::Variable;
use thales::parser::parse_equation;
use thales::resolution_path::TechniqueDifficulty;
use thales::solver::{SmartSolver, Solution, Solver};

fn assert_solves_at_tier(equation_str: &str, target_var: &str, max_expected: TechniqueDifficulty) {
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
    assert!(
        matches!(solution, Solution::Unique(_) | Solution::Multiple(_)),
        "'{}' for '{}': unexpected {:?}",
        equation_str,
        target_var,
        solution
    );
    assert!(
        path.max_difficulty() <= max_expected,
        "'{}' for '{}': expected <= {:?}, got {:?} ({} steps)",
        equation_str,
        target_var,
        max_expected,
        path.max_difficulty(),
        path.step_count()
    );
}

fn assert_solves_ok(equation_str: &str, target_var: &str) {
    let eq = parse_equation(equation_str)
        .unwrap_or_else(|e| panic!("Failed to parse '{}': {:?}", equation_str, e));
    let var = Variable::new(target_var);
    let solver = SmartSolver::new();
    let _ = solver.solve(&eq, &var).unwrap_or_else(|e| {
        panic!(
            "Failed to solve '{}' for '{}': {:?}",
            equation_str, target_var, e
        )
    });
}

// ============================================================================
// THREE-TIER: T1+T2+T3 (Elementary + PowerAndRoots + AlgebraicManip)
// Physics formulas needing basic algebra, powers, and polynomial techniques
// ============================================================================

#[test]
fn t123_01_quadratic_with_power_rearrange() {
    // 2*x^2 + 8*x + 6 = 0 → divide by 2 (T1) then quadratic formula (T3)
    let eq = parse_equation("2 * x^2 + 8 * x + 6 = 0").unwrap();
    let var = Variable::new("x");
    let solver = SmartSolver::new();
    assert!(solver.solve(&eq, &var).is_ok());
}

#[test]
fn t123_02_cubic_with_coefficient() {
    // 2*x^3 - 6*x^2 + 4*x = 0
    let eq = parse_equation("2 * x^3 - 6 * x^2 + 4 * x = 0").unwrap();
    let var = Variable::new("x");
    let solver = SmartSolver::new();
    assert!(solver.solve(&eq, &var).is_ok());
}

#[test]
fn t123_03_v_squared_kinematic() {
    // v^2 = u^2 + 2*a*s, solve for s → s = (v^2-u^2)/(2*a)
    assert_solves_at_tier(
        "v^2 = u^2 + 2 * a * s",
        "s",
        TechniqueDifficulty::Elementary,
    );
}

#[test]
fn t123_04_kinetic_energy_solve_m_from_squared() {
    // (1/2)*m*v^2 = E, solve for m → m = 2*E/v^2
    assert_solves_at_tier("(1/2) * m * v^2 = E", "m", TechniqueDifficulty::Elementary);
}

#[test]
fn t123_05_quartic_biquadratic_param() {
    // x^4 - 13*x^2 + 36 = 0 → (x^2-4)(x^2-9)
    let eq = parse_equation("x^4 - 13 * x^2 + 36 = 0").unwrap();
    let var = Variable::new("x");
    let solver = SmartSolver::new();
    assert!(solver.solve(&eq, &var).is_ok());
}

#[test]
fn t123_06_cubic_depressed() {
    // x^3 + 3*x - 4 = 0
    let eq = parse_equation("x^3 + 3 * x - 4 = 0").unwrap();
    let var = Variable::new("x");
    let solver = SmartSolver::new();
    assert!(solver.solve(&eq, &var).is_ok());
}

#[test]
fn t123_07_power_in_quadratic() {
    // 4*x^2 - 9 = 0 → x^2 = 9/4 → x = ±3/2
    let eq = parse_equation("4 * x^2 - 9 = 0").unwrap();
    let var = Variable::new("x");
    let solver = SmartSolver::new();
    assert!(solver.solve(&eq, &var).is_ok());
}

#[test]
fn t123_08_quartic_simple_roots() {
    // x^4 - 16 = 0 → x = ±2, ±2i
    let eq = parse_equation("x^4 - 16 = 0").unwrap();
    let var = Variable::new("x");
    let solver = SmartSolver::new();
    assert!(solver.solve(&eq, &var).is_ok());
}

#[test]
fn t123_09_stefan_solve_t_from_power() {
    // P = sigma*A*T^4, solve for T → T = (P/(sigma*A))^(1/4) — T2 for root
    assert_solves_at_tier(
        "P = sigma * A * T^4",
        "T",
        TechniqueDifficulty::PowerAndRoots,
    );
}

#[test]
fn t123_10_cubic_all_real() {
    // x^3 - 6*x^2 + 11*x - 6 = 0 → x = 1, 2, 3
    let eq = parse_equation("x^3 - 6 * x^2 + 11 * x - 6 = 0").unwrap();
    let var = Variable::new("x");
    let solver = SmartSolver::new();
    assert!(solver.solve(&eq, &var).is_ok());
}

// ============================================================================
// THREE-TIER: T1+T2+T4 (Elementary + PowerAndRoots + Transcendental)
// Physics formulas with basic ops, powers, and trig
// ============================================================================

#[test]
fn t124_01_pendulum_solve_g() {
    // T = 2*pi*sqrt(L/g) → g = L*(2*pi/T)^2
    assert_solves_at_tier(
        "T = 2 * pi * sqrt(L / g)",
        "g",
        TechniqueDifficulty::PowerAndRoots,
    );
}

#[test]
fn t124_02_projectile_range_solve_theta() {
    // R = v^2*sin(2*theta)/g → 2*theta = arcsin(R*g/v^2) → theta = arcsin(R*g/v^2)/2
    assert_solves_at_tier(
        "R = v^2 * sin(2 * theta) / g",
        "theta",
        TechniqueDifficulty::Transcendental,
    );
}

#[test]
fn t124_03_projectile_range_solve_g() {
    // R = v^2*sin(2*theta)/g → g = v^2*sin(2*theta)/R (elementary)
    assert_solves_at_tier(
        "R = v^2 * sin(2 * theta) / g",
        "g",
        TechniqueDifficulty::Elementary,
    );
}

#[test]
fn t124_04_spring_oscillation_solve_omega() {
    // x = A*sin(omega*t), solve for omega → divide by A, arcsin, divide by t
    assert_solves_at_tier(
        "x = A * sin(omega * t)",
        "omega",
        TechniqueDifficulty::Transcendental,
    );
}

#[test]
fn t124_05_gravity_wave_period() {
    // T = 2*pi*sqrt(L/g), solve for L → L = g*(T/(2*pi))^2
    assert_solves_at_tier(
        "T = 2 * pi * sqrt(L / g)",
        "L",
        TechniqueDifficulty::PowerAndRoots,
    );
}

#[test]
fn t124_06_circular_motion_with_trig() {
    // x = r*cos(omega*t), solve for t
    assert_solves_at_tier(
        "x = r * cos(omega * t)",
        "t",
        TechniqueDifficulty::Transcendental,
    );
}

#[test]
fn t124_07_circular_motion_solve_r() {
    // x = r*cos(omega*t), solve for r → r = x/cos(omega*t) (elementary)
    assert_solves_at_tier(
        "x = r * cos(omega * t)",
        "r",
        TechniqueDifficulty::Elementary,
    );
}

#[test]
fn t124_08_escape_velocity_with_trig() {
    // v_r = v*sin(theta), solve for theta → arcsin(v_r/v)
    assert_solves_at_tier(
        "v_r = v * sin(theta)",
        "theta",
        TechniqueDifficulty::Transcendental,
    );
}

#[test]
fn t124_09_damped_amplitude() {
    // A = A0*exp(-gamma*t), solve for t → t = -ln(A/A0)/gamma
    assert_solves_at_tier(
        "A = A0 * exp(-gamma * t)",
        "t",
        TechniqueDifficulty::PowerAndRoots,
    );
}

#[test]
fn t124_10_beer_lambert_solve_c() {
    // A = epsilon*c*l, where A = -ln(I/I0)
    // Simpler: I = I0*exp(-alpha*x), solve for x
    assert_solves_at_tier(
        "I = I0 * exp(-alpha * x)",
        "x",
        TechniqueDifficulty::PowerAndRoots,
    );
}

// ============================================================================
// THREE-TIER: T1+T3+T4 (Elementary + AlgebraicManip + Transcendental)
// ============================================================================

#[test]
fn t134_01_trig_linear_equation() {
    // 2*sin(x) + 1 = 0 → sin(x) = -1/2 → x = arcsin(-1/2)
    assert_solves_at_tier(
        "2 * sin(x) + 1 = 0",
        "x",
        TechniqueDifficulty::Transcendental,
    );
}

#[test]
fn t134_02_trig_scaled() {
    // 5*cos(3*x) - 2 = 0 → x = arccos(2/5)/3
    assert_solves_at_tier(
        "5 * cos(3 * x) - 2 = 0",
        "x",
        TechniqueDifficulty::Transcendental,
    );
}

#[test]
fn t134_03_exp_linear_offset() {
    // 3*exp(2*x) + 1 = 10 → exp(2*x) = 3 → x = ln(3)/2
    assert_solves_at_tier(
        "3 * exp(2 * x) + 1 = 10",
        "x",
        TechniqueDifficulty::PowerAndRoots,
    );
}

#[test]
fn t134_04_ln_linear_transform() {
    // 2*ln(3*x) - 4 = 0 → ln(3*x) = 2 → 3*x = e^2 → x = e^2/3
    assert_solves_at_tier(
        "2 * ln(3 * x) - 4 = 0",
        "x",
        TechniqueDifficulty::PowerAndRoots,
    );
}

#[test]
fn t134_05_tan_with_params() {
    // a*tan(b*x) = c → x = arctan(c/a)/b
    assert_solves_at_tier(
        "a * tan(b * x) = c",
        "x",
        TechniqueDifficulty::Transcendental,
    );
}

#[test]
fn t134_06_sin_shift_scale() {
    // y = A*sin(B*x + C) + D, solve for x
    assert_solves_at_tier(
        "y = A * sin(B * x + C) + D",
        "x",
        TechniqueDifficulty::Transcendental,
    );
}

#[test]
fn t134_07_exp_decay_offset() {
    // y = a + b*exp(-k*x), solve for x → x = -ln((y-a)/b)/k
    assert_solves_at_tier(
        "y = a + b * exp(-k * x)",
        "x",
        TechniqueDifficulty::PowerAndRoots,
    );
}

#[test]
fn t134_08_log_equation() {
    // y = a * ln(b * x + c), solve for x → x = (exp(y/a) - c) / b
    assert_solves_at_tier(
        "y = a * ln(b * x + c)",
        "x",
        TechniqueDifficulty::PowerAndRoots,
    );
}

#[test]
fn t134_09_cos_with_phase_solve_phase() {
    // y = A*cos(omega*t + phi), solve for phi
    assert_solves_at_tier(
        "y = A * cos(omega * t + phi)",
        "phi",
        TechniqueDifficulty::Transcendental,
    );
}

#[test]
fn t134_10_sin_with_offset_solve_a() {
    // y = A*sin(x) + B, solve for A → A = (y-B)/sin(x) (elementary)
    assert_solves_at_tier("y = A * sin(x) + B", "A", TechniqueDifficulty::Elementary);
}

// ============================================================================
// THREE-TIER: T2+T3+T4 (PowerAndRoots + AlgebraicManip + Transcendental)
// ============================================================================

#[test]
fn t234_01_exp_quadratic_arg() {
    // y = exp(a*x^2) — variable x appears in x^2 inside exp
    // This is actually exp(a*x^2), variable x appears non-linearly
    // The symbolic isolation can handle it: peel exp, divide by a, take sqrt
    assert_solves_at_tier("y = exp(a * x^2)", "x", TechniqueDifficulty::PowerAndRoots);
}

#[test]
fn t234_02_sqrt_of_quadratic_expr() {
    // c = sqrt(a^2 + b^2), solve for a
    assert_solves_at_tier(
        "c = sqrt(a^2 + b^2)",
        "a",
        TechniqueDifficulty::PowerAndRoots,
    );
}

#[test]
fn t234_03_sin_of_sqrt() {
    // y = sin(sqrt(x)), solve for x → sqrt(x) = arcsin(y) → x = arcsin(y)^2
    assert_solves_at_tier("y = sin(sqrt(x))", "x", TechniqueDifficulty::Transcendental);
}

#[test]
fn t234_04_ln_of_power() {
    // y = ln(x^3) → x^3 = exp(y) → x = exp(y)^(1/3)
    assert_solves_at_tier("y = ln(x^3)", "x", TechniqueDifficulty::PowerAndRoots);
}

#[test]
fn t234_05_exp_of_linear_power() {
    // y = exp(x)^2 = exp(2*x) → 2*x = ln(y) → x = ln(y)/2
    // Parser may interpret as (exp(x))^2 = Power(exp(x), 2)
    assert_solves_at_tier("y = exp(x)^2", "x", TechniqueDifficulty::PowerAndRoots);
}

#[test]
fn t234_06_trig_sqrt_chain() {
    // y = cos(sqrt(x)), solve for x → sqrt(x) = arccos(y) → x = arccos(y)^2
    assert_solves_at_tier("y = cos(sqrt(x))", "x", TechniqueDifficulty::Transcendental);
}

#[test]
fn t234_07_exp_cube_root() {
    // y = exp(cbrt(x)), solve for x → cbrt(x) = ln(y) → x = ln(y)^3
    assert_solves_at_tier("y = exp(cbrt(x))", "x", TechniqueDifficulty::PowerAndRoots);
}

#[test]
fn t234_08_sqrt_exp_chain() {
    // y = sqrt(exp(x)) → exp(x) = y^2 → x = ln(y^2) = 2*ln(y)
    assert_solves_at_tier("y = sqrt(exp(x))", "x", TechniqueDifficulty::PowerAndRoots);
}

#[test]
fn t234_09_cbrt_ln() {
    // y = cbrt(ln(x)) → ln(x) = y^3 → x = exp(y^3)
    assert_solves_at_tier("y = cbrt(ln(x))", "x", TechniqueDifficulty::PowerAndRoots);
}

#[test]
fn t234_10_sin_exp_chain() {
    // y = sin(exp(x)) → exp(x) = arcsin(y) → x = ln(arcsin(y))
    // Arcsin is Transcendental, ln is PowerAndRoots → max is Transcendental
    assert_solves_at_tier("y = sin(exp(x))", "x", TechniqueDifficulty::Transcendental);
}

// ============================================================================
// THREE-TIER combinations involving T5 or T6
// 16 combinations × 10 tests = 160 tests.
// Tests where the solver handles the equation (parser treats calculus notation
// as algebraic) are un-ignored; unsupported forms remain #[ignore].
// ============================================================================

// T1+T2+T5
#[test]
fn t125_01() {
    // v = d(x)/dt, solve for x — derivative preserved as opaque wrapper
    assert_solves_ok("v = d(x)/dt", "x");
}
#[test]
fn t125_02() {
    assert_solves_ok("a = d(v)/dt", "v");
}
#[test]
fn t125_03() {
    assert_solves_ok("E = integral(F*v, dt)", "v");
}
#[test]
fn t125_04() {
    assert_solves_ok("W = integral(m*a, dx)", "a");
}
#[test]
fn t125_05() {
    assert_solves_ok("P = d(E)/dt", "E");
}
#[test]
fn t125_06() {
    assert_solves_ok("I = d(Q)/dt", "Q");
}
#[test]
fn t125_07() {
    assert_solves_ok("rho = d(m)/d(V)", "m");
}
#[test]
fn t125_08() {
    assert_solves_ok("sigma = d(F)/d(A)", "F");
}
#[test]
fn t125_09() {
    assert_solves_ok("epsilon = d(L)/d(L0)", "L");
}
#[test]
fn t125_10() {
    assert_solves_ok("mu = d(p)/d(V)", "p");
}

// T1+T2+T6
#[test]
fn t126_01() {
    assert_solves_ok("norm = sqrt(x^2 + y^2)", "x");
}
#[test]
fn t126_02() {
    assert_solves_ok("E_n = -13.6/n^2", "n");
}
#[test]
fn t126_03() {
    assert_solves_ok("r_n = a_0*n^2/Z", "n");
}
#[test]
fn t126_04() {
    assert_solves_ok("v = sqrt(G*M/r)", "r");
}
#[test]
fn t126_05() {
    assert_solves_ok("T = 2*pi*sqrt(r^3/(G*M))", "r");
}
#[test]
fn t126_06() {
    assert_solves_ok("E = h*c/lambda", "lambda");
}
#[test]
fn t126_07() {
    assert_solves_ok("p = h/lambda", "lambda");
}
#[test]
fn t126_08() {
    assert_solves_ok("f = c/lambda", "lambda");
}
#[test]
fn t126_09() {
    assert_solves_ok("K = (1/2)*I*omega^2", "omega");
}
#[test]
fn t126_10() {
    assert_solves_ok("L = I*omega", "I");
}

// T1+T3+T5
#[test]
fn t135_01() {
    assert_solves_ok("y = d(x^3-3*x)/dx", "x");
}
#[test]
fn t135_02() {
    assert_solves_ok("A = integral(x^2-4, dx)", "x");
}
#[test]
fn t135_03() {
    assert_solves_ok("y = d((x+1)^2)/dx", "x");
}
#[test]
fn t135_04() {
    assert_solves_ok("V = integral(pi*x^2, dx)", "x");
}
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t135_05() {
    assert_solves_ok("S = integral(2*pi*x, dx)", "x");
}
#[test]
fn t135_06() {
    assert_solves_ok("M = integral(x*rho, dx)", "rho");
}
#[test]
fn t135_07() {
    assert_solves_ok("I = integral(r^2*dm, dr)", "r");
}
#[test]
fn t135_08() {
    assert_solves_ok("W = integral(k*x, dx)", "k");
}
#[test]
fn t135_09() {
    assert_solves_ok("U = integral(m*g, dh)", "m");
}
#[test]
fn t135_10() {
    assert_solves_ok("Q = integral(rho*c*T, dV)", "T");
}

// T1+T3+T6
#[test]
fn t136_01() {
    assert_solves_ok("det_A = a*d-b*c", "a");
}
#[test]
fn t136_02() {
    assert_solves_ok("tr_A = a+d", "a");
}
#[test]
fn t136_03() {
    assert_solves_ok("eig = (a+d)/2 + sqrt(((a-d)/2)^2+b*c)", "b");
}
#[test]
#[ignore = "Rational equation — variable in denominator on both sides"]
fn t136_04() {
    assert_solves_ok("R_par = R1*R2/(R1+R2)", "R1");
}
#[test]
fn t136_05() {
    assert_solves_ok("C_ser = 1/(1/C1+1/C2)", "C1");
}
#[test]
fn t136_06() {
    assert_solves_ok("f_res = 1/(2*pi*sqrt(L*C))", "L");
}
#[test]
fn t136_07() {
    assert_solves_ok("Q_factor = omega*L/R", "omega");
}
#[test]
fn t136_08() {
    assert_solves_ok("BW = f_res/Q_factor", "Q_factor");
}
#[test]
fn t136_09() {
    assert_solves_ok("Z = sqrt(R^2+(omega*L)^2)", "omega");
}
#[test]
fn t136_10() {
    assert_solves_ok("P = V^2/R", "R");
}

// T1+T4+T5
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t145_01() {
    assert_solves_ok("y = integral(sin(x), dx)", "x");
}
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t145_02() {
    assert_solves_ok("y = integral(cos(x), dx)", "x");
}
#[test]
#[ignore = "Requires symbolic differentiation — chain rule through derivative"]
fn t145_03() {
    assert_solves_ok("y = d(sin(omega*t))/dt", "omega");
}
#[test]
fn t145_04() {
    assert_solves_ok("E = integral(E0*sin(omega*t), dt)", "omega");
}
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t145_05() {
    assert_solves_ok("B = integral(mu_0*I/(2*pi*r), dr)", "r");
}
#[test]
fn t145_06() {
    assert_solves_ok("V = integral(E*cos(theta), dr)", "theta");
}
#[test]
#[ignore = "Requires symbolic differentiation — chain rule through derivative"]
fn t145_07() {
    assert_solves_ok("F = d(p*sin(theta))/dt", "theta");
}
#[test]
fn t145_08() {
    assert_solves_ok("W = integral(F*cos(theta), dx)", "theta");
}
#[test]
fn t145_09() {
    assert_solves_ok("P = F*v*cos(theta)", "theta");
}
#[test]
fn t145_10() {
    assert_solves_ok("tau = r*F*sin(theta)", "theta");
}

// T1+T4+T6
#[test]
fn t146_01() {
    assert_solves_ok("psi = A*sin(k*x)", "k");
}
#[test]
fn t146_02() {
    assert_solves_ok("E = h*f", "f");
}
#[test]
fn t146_03() {
    assert_solves_ok("lambda = h/(m*v)", "v");
}
#[test]
fn t146_04() {
    assert_solves_ok("B = mu_0*n*I", "I");
}
#[test]
fn t146_05() {
    assert_solves_ok("E = sigma/(2*epsilon_0)", "sigma");
}
#[test]
fn t146_06() {
    assert_solves_ok("F = q*v*sin(theta)*B", "theta");
}
#[test]
#[ignore = "Transcendental — variable appears as both linear factor and trigonometric argument"]
fn t146_07() {
    assert_solves_ok("emf = N*B*A*omega*sin(omega*t)", "omega");
}
#[test]
fn t146_08() {
    assert_solves_ok("I = I0*cos(theta)^2", "theta");
}
#[test]
#[ignore = "Parser error — d_sin_theta parsed as single identifier, not d*sin(theta)"]
fn t146_09() {
    assert_solves_ok("d_sin_theta = m*lambda", "theta");
}
#[test]
fn t146_10() {
    assert_solves_ok("n*lambda = 2*d*sin(theta)", "theta");
}

// T1+T5+T6
#[test]
fn t156_01() {
    assert_solves_ok("S = integral(exp(-E/(k*T)), dE)", "T");
}
#[test]
fn t156_02() {
    assert_solves_ok("Z = integral(exp(-beta*H), dq)", "beta");
}
#[test]
#[ignore = "Variable parsed as function call — f(E) not recognized as f*E"]
fn t156_03() {
    assert_solves_ok("rho = integral(f(E)*g(E), dE)", "f");
}
#[test]
fn t156_04() {
    assert_solves_ok("J = integral(rho*v, dA)", "rho");
}
#[test]
fn t156_05() {
    assert_solves_ok("Phi = integrate(B, A)", "B");
}
#[test]
fn t156_06() {
    assert_solves_ok("emf = -d(Phi)/dt", "Phi");
}
#[test]
fn t156_07() {
    assert_solves_ok("H = integrate(T, S)", "T");
}
#[test]
fn t156_08() {
    assert_solves_ok("G = H - T*S", "T");
}
#[test]
fn t156_09() {
    assert_solves_ok("F = -d(U)/dx", "U");
}
#[test]
#[ignore = "Requires symbolic differentiation — variable is the derivative variable"]
fn t156_10() {
    assert_solves_ok("mu = d(G)/d(N)", "N");
}

// T2+T3+T5
#[test]
fn t235_01() {
    assert_solves_ok("y = integral(x^(3/2), dx)", "x");
}
#[test]
fn t235_02() {
    assert_solves_ok("y = d(x^(2/3))/dx", "x");
}
#[test]
fn t235_03() {
    assert_solves_ok("A = integral(sqrt(R^2-x^2), dx)", "R");
}
#[test]
fn t235_04() {
    assert_solves_ok("V = (4/3)*pi*integral(r^2, dr)", "r");
}
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t235_05() {
    assert_solves_ok("S = 4*pi*integral(r, dr)", "r");
}
#[test]
fn t235_06() {
    assert_solves_ok("I = integral(r^2*rho, dr)", "rho");
}
#[test]
fn t235_07() {
    assert_solves_ok("E = integral(k*q/r^2, dr)", "q");
}
#[test]
fn t235_08() {
    assert_solves_ok("V = integral(E, dr)", "E");
}
#[test]
fn t235_09() {
    assert_solves_ok("W = integral(P, dV)", "P");
}
#[test]
fn t235_10() {
    assert_solves_ok("Q = integral(k*A*dT/dx, dt)", "k");
}

// T2+T3+T6
#[test]
#[ignore = "Domain-specific function not yet implemented — det() on symbolic matrix"]
fn t236_01() {
    assert_solves_ok("lambda = sqrt(det(A))", "A");
}
#[test]
fn t236_02() {
    assert_solves_ok("sigma = sqrt(E/(3*(1-2*nu)))", "nu");
}
#[test]
fn t236_03() {
    assert_solves_ok("c = sqrt(gamma*R*T/M)", "gamma");
}
#[test]
fn t236_04() {
    assert_solves_ok("v_s = sqrt(B/rho)", "B");
}
#[test]
fn t236_05() {
    assert_solves_ok("Re = rho*v*L/mu", "mu");
}
#[test]
fn t236_06() {
    assert_solves_ok("Ma = v/sqrt(gamma*R*T/M)", "v");
}
#[test]
fn t236_07() {
    assert_solves_ok("Fr = v/sqrt(g*L)", "v");
}
#[test]
fn t236_08() {
    assert_solves_ok("We = rho*v^2*L/sigma", "sigma");
}
#[test]
fn t236_09() {
    assert_solves_ok("Gr = g*beta*dT*L^3/nu^2", "beta");
}
#[test]
fn t236_10() {
    assert_solves_ok("Nu = h*L/k", "h");
}

// T2+T4+T5
#[test]
fn t245_01() {
    assert_solves_ok("y = integral(exp(-x^2), dx)", "x");
}
#[test]
fn t245_02() {
    assert_solves_ok("y = d(sin(x^2))/dx", "x");
}
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t245_03() {
    assert_solves_ok("y = integral(x*sin(x), dx)", "x");
}
#[test]
#[ignore = "Requires symbolic differentiation — variable is the derivative variable"]
fn t245_04() {
    assert_solves_ok("y = d(exp(sin(x)))/dx", "x");
}
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t245_05() {
    assert_solves_ok("y = integral(exp(-x)*cos(x), dx)", "x");
}
#[test]
#[ignore = "Requires symbolic differentiation — variable is the derivative variable"]
fn t245_06() {
    assert_solves_ok("y = d(ln(sin(x)))/dx", "x");
}
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t245_07() {
    assert_solves_ok("y = integral(sin(x)/x, dx)", "x");
}
#[test]
fn t245_08() {
    assert_solves_ok("y = d(sqrt(tan(x)))/dx", "x");
}
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t245_09() {
    assert_solves_ok("y = integral(exp(x)*sin(x), dx)", "x");
}
#[test]
#[ignore = "Requires symbolic differentiation — variable is the derivative variable"]
fn t245_10() {
    assert_solves_ok("y = d(cos(ln(x)))/dx", "x");
}

// T2+T4+T6
#[test]
fn t246_01() {
    assert_solves_ok("psi = A*exp(-r/a_0)*sin(theta)", "r");
}
#[test]
fn t246_02() {
    assert_solves_ok("E = sqrt(p^2*c^2 + m^2*c^4)", "p");
}
#[test]
fn t246_03() {
    assert_solves_ok("L = m*v*r*sin(theta)", "theta");
}
#[test]
fn t246_04() {
    assert_solves_ok("B = mu_0*I/(2*pi*r)", "r");
}
#[test]
fn t246_05() {
    assert_solves_ok("F = q*E + q*v*sin(theta)*B", "theta");
}
#[test]
fn t246_06() {
    assert_solves_ok("phi = E*cos(theta)*A", "theta");
}
#[test]
fn t246_07() {
    assert_solves_ok("I = (P/(4*pi*r^2))*cos(theta)", "r");
}
#[test]
fn t246_08() {
    assert_solves_ok("S = sigma*T^4*cos(theta)", "theta");
}
#[test]
fn t246_09() {
    assert_solves_ok("F_drag = 0.5*C_d*rho*A*v^2", "v");
}
#[test]
fn t246_10() {
    assert_solves_ok("P = F*v*cos(theta)", "v");
}

// T2+T5+T6
#[test]
fn t256_01() {
    assert_solves_ok("E = integrate(rho*g*h, V)", "h");
}
#[test]
fn t256_02() {
    assert_solves_ok("M = integrate(r^2*rho, V)", "rho");
}
#[test]
fn t256_03() {
    assert_solves_ok("U = integrate(G*m*rho/r, V)", "m");
}
#[test]
fn t256_04() {
    assert_solves_ok("T = integrate(0.5*rho*v^2, V)", "rho");
}
#[test]
fn t256_05() {
    assert_solves_ok("P = integrate(F/A, A)", "F");
}
#[test]
fn t256_06() {
    assert_solves_ok("E = integrate(sigma*T^4, A)", "T");
}
#[test]
fn t256_07() {
    assert_solves_ok("Q = integrate(k*A*dT/dx, t)", "k");
}
#[test]
fn t256_08() {
    assert_solves_ok("W = integrate(P, V)", "P");
}
#[test]
fn t256_09() {
    assert_solves_ok("H = integrate(c_p, T)", "c_p");
}
#[test]
fn t256_10() {
    assert_solves_ok("S = integrate(1/T, Q)", "T");
}

// T3+T4+T5
#[test]
fn t345_01() {
    assert_solves_ok("y = integral(sin(x)^2, dx)", "x");
}
#[test]
fn t345_02() {
    assert_solves_ok("y = integral(cos(x)^3, dx)", "x");
}
#[test]
fn t345_03() {
    assert_solves_ok("y = d(tan(x)^2)/dx", "x");
}
#[test]
fn t345_04() {
    assert_solves_ok("y = integral(sin(x)*cos(x)^2, dx)", "x");
}
#[test]
fn t345_05() {
    assert_solves_ok("y = d(asin(x))/dx", "x");
}
#[test]
fn t345_06() {
    assert_solves_ok("y = integral(1/(1+tan(x)^2), dx)", "x");
}
#[test]
fn t345_07() {
    assert_solves_ok("y = integral(sec(x)^3, dx)", "x");
}
#[test]
fn t345_08() {
    assert_solves_ok("y = d(sin(x)^3)/dx", "x");
}
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t345_09() {
    assert_solves_ok("y = integral(exp(sin(x))*cos(x), dx)", "x");
}
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t345_10() {
    assert_solves_ok("y = integral(ln(cos(x)), dx)", "x");
}

// T3+T4+T6
#[test]
fn t346_01() {
    assert_solves_ok("Z = R + omega*L*sin(theta)", "theta");
}
#[test]
fn t346_02() {
    assert_solves_ok("P = V*I*cos(phi)", "phi");
}
#[test]
fn t346_03() {
    assert_solves_ok("S = V*I", "I");
}
#[test]
fn t346_04() {
    assert_solves_ok("Q = V*I*sin(phi)", "phi");
}
#[test]
fn t346_05() {
    assert_solves_ok("pf = cos(phi)", "phi");
}
#[test]
fn t346_06() {
    assert_solves_ok("X_L = omega*L", "omega");
}
#[test]
fn t346_07() {
    assert_solves_ok("X_C = 1/(omega*C)", "omega");
}
#[test]
fn t346_08() {
    assert_solves_ok("Z = sqrt(R^2 + (X_L-X_C)^2)", "R");
}
#[test]
fn t346_09() {
    assert_solves_ok("I = V/Z", "Z");
}
#[test]
fn t346_10() {
    assert_solves_ok("theta = atan((X_L-X_C)/R)", "R");
}

// T3+T5+T6
#[test]
fn t356_01() {
    assert_solves_ok("a_n = (2/T)*integral(f*cos(2*pi*n*t/T), dt)", "n");
}
#[test]
fn t356_02() {
    assert_solves_ok("b_n = (2/T)*integral(f*sin(2*pi*n*t/T), dt)", "n");
}
#[test]
fn t356_03() {
    assert_solves_ok("c_n = (1/T)*integral(f*exp(-2*pi*i*n*t/T), dt)", "n");
}
#[test]
fn t356_04() {
    assert_solves_ok("F_s = integral(f*exp(-s*t), dt)", "s");
}
#[test]
fn t356_05() {
    assert_solves_ok("G_w = integral(f*exp(-i*w*t), dt)", "w");
}
#[test]
fn t356_06() {
    assert_solves_ok("P_n = integral(abs(c_n)^2, dn)", "c_n");
}
#[test]
fn t356_07() {
    assert_solves_ok("E = integral(abs(f)^2, dt)", "f");
}
#[test]
#[ignore = "Variable inside function call arguments — x(t+tau) not decomposable"]
fn t356_08() {
    assert_solves_ok("R_xx = integral(x(t)*x(t+tau), dt)", "tau");
}
#[test]
#[ignore = "Transcendental — variable appears in both linear and logarithmic terms"]
fn t356_09() {
    assert_solves_ok("H = -integral(p*ln(p), dx)", "p");
}
#[test]
#[ignore = "Transcendental — variable appears in both linear and logarithmic terms"]
fn t356_10() {
    assert_solves_ok("I = integral(f*ln(f/g), dx)", "f");
}

// T4+T5+T6
#[test]
#[ignore = "Non-invertible special function — bessel with variable as integration variable"]
fn t456_01() {
    assert_solves_ok("y = integral(sin(x)*bessel(0,x), dx)", "x");
}
#[test]
#[ignore = "Variable only appears inside function call f(theta)"]
fn t456_02() {
    assert_solves_ok("psi = integral(exp(i*k*r)*f(theta), dOmega)", "theta");
}
#[test]
fn t456_03() {
    assert_solves_ok("G = integral(exp(-i*omega*tau)*R(tau), dtau)", "omega");
}
#[test]
#[ignore = "Parser does not support cross() vector operator"]
fn t456_04() {
    assert_solves_ok("S = integral(E*cross(B), dA)", "E");
}
#[test]
#[ignore = "Domain-specific function not yet implemented — grad()"]
fn t456_05() {
    assert_solves_ok("F = integral(rho*grad(phi), dV)", "phi");
}
#[test]
#[ignore = "Parser does not support dot() vector operator"]
fn t456_06() {
    assert_solves_ok("W = integral(J*dot(E), dV)", "J");
}
#[test]
fn t456_07() {
    assert_solves_ok("P = integrate(S, A)", "S");
}
#[test]
fn t456_08() {
    assert_solves_ok("Q = integrate(sigma*T^4*cos(theta), A)", "T");
}
#[test]
#[ignore = "Parser does not support cross() vector operator"]
fn t456_09() {
    assert_solves_ok("M = integral(r*cross(F)*sin(theta), dr)", "theta");
}
#[test]
#[ignore = "Parser does not support cross() vector operator"]
fn t456_10() {
    assert_solves_ok("L = integral(r*cross(p), dm)", "r");
}

// ============================================================================
// FOUR-TIER: T1+T2+T3+T4 (the only passable 4-tier combo)
// ============================================================================

#[test]
fn t1234_01_general_oscillation() {
    // y = A*sin(omega*t) + B, solve for t
    // T1 (subtract B, divide A) + T4 (arcsin) + T1 (divide omega)
    assert_solves_at_tier(
        "y = A * sin(omega * t) + B",
        "t",
        TechniqueDifficulty::Transcendental,
    );
}

#[test]
fn t1234_02_damped_oscillation_solve_gamma() {
    // A = A0*exp(-gamma*t), solve for gamma → gamma = -ln(A/A0)/t
    assert_solves_at_tier(
        "A = A0 * exp(-gamma * t)",
        "gamma",
        TechniqueDifficulty::PowerAndRoots,
    );
}

#[test]
fn t1234_03_snell_solve_n1() {
    // n1*sin(theta1) = n2*sin(theta2), solve for n1 → elementary
    assert_solves_at_tier(
        "n1 * sin(theta1) = n2 * sin(theta2)",
        "n1",
        TechniqueDifficulty::Elementary,
    );
}

#[test]
fn t1234_04_projectile_height_solve_theta() {
    // h = v^2*sin(theta)^2/(2*g)
    // Solving for theta: sin(theta)^2 = 2*g*h/v^2
    // This has sin(theta)^2, variable theta appears once through sin(theta)^2
    // Parser may handle this as Power(sin(theta), 2)
    assert_solves_at_tier(
        "h = v^2 * sin(theta)^2 / (2 * g)",
        "theta",
        TechniqueDifficulty::Transcendental,
    );
}

#[test]
fn t1234_05_compound_decay_solve_t() {
    // N = N0*exp(-lambda*t), solve for t
    assert_solves_at_tier(
        "N = N0 * exp(-lambda * t)",
        "t",
        TechniqueDifficulty::PowerAndRoots,
    );
}

#[test]
fn t1234_06_coulomb_with_angle() {
    // F = k*q1*q2*cos(theta)/r^2, solve for theta
    assert_solves_at_tier(
        "F = k * q1 * q2 * cos(theta) / r^2",
        "theta",
        TechniqueDifficulty::Transcendental,
    );
}

#[test]
fn t1234_07_wave_equation_solve_k() {
    // y = A*sin(k*x - omega*t), solve for k
    assert_solves_at_tier(
        "y = A * sin(k * x - omega * t)",
        "k",
        TechniqueDifficulty::Transcendental,
    );
}

#[test]
fn t1234_08_rc_circuit_solve_t() {
    // V = V0*(1 - exp(-t/(R*C))), solve for t
    assert_solves_at_tier(
        "V = V0 * (1 - exp(-t / (R * C)))",
        "t",
        TechniqueDifficulty::PowerAndRoots,
    );
}

#[test]
fn t1234_09_doppler_solve_v_s() {
    // f_obs = f_src * v / (v - v_s), solve for v_s
    assert_solves_at_tier(
        "f_obs = f_src * v / (v - v_s)",
        "v_s",
        TechniqueDifficulty::Elementary,
    );
}

#[test]
fn t1234_10_malus_law_solve_theta() {
    // I = I0*cos(theta)^2, solve for theta
    assert_solves_at_tier(
        "I = I0 * cos(theta)^2",
        "theta",
        TechniqueDifficulty::Transcendental,
    );
}

// ============================================================================
// FOUR-TIER combos involving T5 or T6 (14 combos × 10 = 140 tests)
// Tests where the solver handles the equation are un-ignored; unsupported
// forms remain #[ignore].
// ============================================================================

// T1+T2+T3+T5
#[test]
fn t1235_01() {
    assert_solves_ok("y = integral(x^2*exp(x), dx)", "x");
}
#[test]
fn t1235_02() {
    assert_solves_ok("y = d(sqrt(x^2+1))/dx", "x");
}
#[test]
fn t1235_03() {
    assert_solves_ok("A = integral(pi*r^2, dr)", "r");
}
#[test]
fn t1235_04() {
    assert_solves_ok("V = integral(4*pi*r^2, dr)", "r");
}
#[test]
fn t1235_05() {
    assert_solves_ok("E = integrate(F, r)", "F");
}
#[test]
fn t1235_06() {
    assert_solves_ok("W = integrate(P, V)", "P");
}
#[test]
fn t1235_07() {
    assert_solves_ok("Q = integrate(c*m, T)", "c");
}
#[test]
fn t1235_08() {
    assert_solves_ok("I = integrate(r^2, m)", "r");
}
#[test]
fn t1235_09() {
    assert_solves_ok("M = integrate(rho, V)", "rho");
}
#[test]
fn t1235_10() {
    assert_solves_ok("S = integrate(1/T, Q)", "T");
}

// T1+T2+T3+T6
#[test]
#[ignore = "Domain-specific function not yet implemented — eigenvalue()"]
fn t1236_01() {
    assert_solves_ok("lambda = eigenvalue(A)", "A");
}
#[test]
fn t1236_02() {
    assert_solves_ok("det = a*d-b*c", "b");
}
#[test]
fn t1236_03() {
    assert_solves_ok("norm = sqrt(x_1^2 + x_2^2 + x_3^2)", "x_1");
}
#[test]
fn t1236_04() {
    assert_solves_ok("R = sqrt(L^2+(1/(w*C))^2)", "C");
}
#[test]
fn t1236_05() {
    assert_solves_ok("f = 1/(2*pi*sqrt(L*C))", "C");
}
#[test]
#[ignore = "Rational equation — variable in denominator on both sides"]
fn t1236_06() {
    assert_solves_ok("Z = R+i*(w*L-1/(w*C))", "w");
}
#[test]
fn t1236_07() {
    assert_solves_ok("Q = w*L/R", "L");
}
#[test]
fn t1236_08() {
    assert_solves_ok("BW = R/(2*pi*L)", "R");
}
#[test]
fn t1236_09() {
    assert_solves_ok("P = V^2/(2*R)", "V");
}
#[test]
fn t1236_10() {
    assert_solves_ok("E = 0.5*C*V^2", "C");
}

// T1+T2+T4+T5
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t1245_01() {
    assert_solves_ok("y = integral(exp(-x)*sin(x), dx)", "x");
}
#[test]
#[ignore = "Requires symbolic differentiation — variable is the derivative variable"]
fn t1245_02() {
    assert_solves_ok("y = d(exp(sin(x)))/dx", "x");
}
#[test]
fn t1245_03() {
    assert_solves_ok("A = integral(r*sin(theta), dtheta)", "r");
}
#[test]
fn t1245_04() {
    assert_solves_ok("V = integral(r^2*sin(theta), dtheta)", "r");
}
#[test]
fn t1245_05() {
    assert_solves_ok("E = integral(k*q*cos(theta)/r^2, dr)", "q");
}
#[test]
fn t1245_06() {
    assert_solves_ok("B = integral(mu_0*I*sin(theta)/(4*pi*r^2), dr)", "I");
}
#[test]
fn t1245_07() {
    assert_solves_ok("F = integrate(rho*g*sin(theta), A)", "theta");
}
#[test]
fn t1245_08() {
    assert_solves_ok("W = integrate(F*cos(theta), x)", "F");
}
#[test]
fn t1245_09() {
    assert_solves_ok("P = integrate(sigma*T^4*cos(theta), A)", "T");
}
#[test]
fn t1245_10() {
    assert_solves_ok("Q = integrate(h*A*(T-T_inf), t)", "h");
}

// T1+T2+T4+T6
#[test]
fn t1246_01() {
    assert_solves_ok("psi = A*exp(-r/a)*Y(theta,phi)", "r");
}
#[test]
fn t1246_02() {
    assert_solves_ok("E = sqrt(p^2*c^2+m^2*c^4)", "m");
}
#[test]
fn t1246_03() {
    assert_solves_ok("L = m*r*v*sin(theta)", "theta");
}
#[test]
fn t1246_04() {
    assert_solves_ok("F = G*m1*m2*cos(theta)/r^2", "theta");
}
#[test]
fn t1246_05() {
    assert_solves_ok("B = mu_0*I*sin(theta)/(4*pi*r^2)", "theta");
}
#[test]
fn t1246_06() {
    assert_solves_ok("E = k*q*cos(theta)/r^2", "theta");
}
#[test]
fn t1246_07() {
    assert_solves_ok("phi = E*r*cos(theta)", "theta");
}
#[test]
fn t1246_08() {
    assert_solves_ok("I = I0*exp(-alpha*x)*cos(theta)^2", "x");
}
#[test]
fn t1246_09() {
    assert_solves_ok("F_L = q*v*B*sin(theta)", "theta");
}
#[test]
fn t1246_10() {
    assert_solves_ok("tau = m*B*sin(theta)", "theta");
}

// T1+T2+T5+T6
#[test]
fn t1256_01() {
    assert_solves_ok("E = integrate(rho*c*T, V)", "T");
}
#[test]
fn t1256_02() {
    assert_solves_ok("M = integrate(rho*r^2, V)", "rho");
}
#[test]
fn t1256_03() {
    assert_solves_ok("U = integrate(0.5*k*x^2, x)", "k");
}
#[test]
fn t1256_04() {
    assert_solves_ok("KE = integrate(0.5*m*v^2, t)", "m");
}
#[test]
fn t1256_05() {
    assert_solves_ok("W = integrate(F, x)", "F");
}
#[test]
fn t1256_06() {
    assert_solves_ok("Q = integrate(sigma*T^4, A)", "sigma");
}
#[test]
fn t1256_07() {
    assert_solves_ok("S = integrate(c/T, T)", "c");
}
#[test]
fn t1256_08() {
    assert_solves_ok("G = H-T*S", "S");
}
#[test]
fn t1256_09() {
    assert_solves_ok("F = -d(U)/dx", "U");
}
#[test]
fn t1256_10() {
    assert_solves_ok("P = d(W)/dt", "W");
}

// T1+T3+T4+T5
#[test]
fn t1345_01() {
    assert_solves_ok("y = integral(sin(x)^2, dx)", "x");
}
#[test]
fn t1345_02() {
    assert_solves_ok("y = d(cos(x)^3)/dx", "x");
}
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t1345_03() {
    assert_solves_ok("A = integral(sin(x)*cos(x), dx)", "x");
}
#[test]
fn t1345_04() {
    assert_solves_ok("V = integral(pi*sin(x)^2, dx)", "x");
}
#[test]
fn t1345_05() {
    assert_solves_ok("E = integral(k*cos(theta)/r^2, dr)", "k");
}
#[test]
fn t1345_06() {
    assert_solves_ok("W = integral(F*cos(theta), dx)", "F");
}
#[test]
fn t1345_07() {
    assert_solves_ok("P = F*v*cos(theta)", "F");
}
#[test]
fn t1345_08() {
    assert_solves_ok("tau = F*r*sin(theta)", "F");
}
#[test]
fn t1345_09() {
    assert_solves_ok("L = m*r^2*omega", "omega");
}
#[test]
fn t1345_10() {
    assert_solves_ok("I = integrate(r^2, m)", "r");
}

// T1+T3+T4+T6
#[test]
fn t1346_01() {
    assert_solves_ok("Z = sqrt(R^2+(wL-1/(wC))^2)", "R");
}
#[test]
fn t1346_02() {
    assert_solves_ok("phi = atan((wL-1/(wC))/R)", "R");
}
#[test]
fn t1346_03() {
    assert_solves_ok("P = V*I*cos(phi)", "phi");
}
#[test]
fn t1346_04() {
    assert_solves_ok("Q = V*I*sin(phi)", "phi");
}
#[test]
fn t1346_05() {
    assert_solves_ok("S = sqrt(P^2+Q^2)", "P");
}
#[test]
fn t1346_06() {
    assert_solves_ok("pf = P/S", "P");
}
#[test]
fn t1346_07() {
    assert_solves_ok("I = V/Z", "V");
}
#[test]
fn t1346_08() {
    assert_solves_ok("V_R = I*R", "I");
}
#[test]
fn t1346_09() {
    assert_solves_ok("V_L = I*wL", "I");
}
#[test]
fn t1346_10() {
    assert_solves_ok("V_C = I/(wC)", "I");
}

// T1+T3+T5+T6
#[test]
fn t1356_01() {
    assert_solves_ok("F = integrate(sigma, A)", "sigma");
}
#[test]
fn t1356_02() {
    assert_solves_ok("M = integral(x*f, dx)", "f");
}
#[test]
fn t1356_03() {
    assert_solves_ok("I = integral(r^2*f, dx)", "f");
}
#[test]
fn t1356_04() {
    assert_solves_ok("W = integral(F, dx)", "F");
}
#[test]
fn t1356_05() {
    assert_solves_ok("Q = integral(rho, dV)", "rho");
}
#[test]
fn t1356_06() {
    assert_solves_ok("E = integral(D, dA)", "D");
}
#[test]
fn t1356_07() {
    assert_solves_ok("B = integral(H, dl)", "H");
}
#[test]
fn t1356_08() {
    assert_solves_ok("V = integral(E, dr)", "E");
}
#[test]
fn t1356_09() {
    assert_solves_ok("A = integral(B, dA)", "B");
}
#[test]
fn t1356_10() {
    assert_solves_ok("Phi = integral(B, dA)", "B");
}

// T1+T4+T5+T6
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t1456_01() {
    assert_solves_ok("y = integral(sin(x)*exp(x), dx)", "x");
}
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t1456_02() {
    assert_solves_ok("y = integral(cos(x)/x, dx)", "x");
}
#[test]
fn t1456_03() {
    assert_solves_ok("F = integrate(rho*g*sin(theta), A)", "rho");
}
#[test]
fn t1456_04() {
    assert_solves_ok("W = integrate(F*cos(theta), x)", "theta");
}
#[test]
fn t1456_05() {
    assert_solves_ok("Q = integrate(sigma*cos(theta)*T^4, A)", "sigma");
}
#[test]
fn t1456_06() {
    assert_solves_ok("E = integral(k*sin(theta)/r^2, dr)", "k");
}
#[test]
fn t1456_07() {
    assert_solves_ok("V = integral(rho*cos(theta)/r, dr)", "rho");
}
#[test]
fn t1456_08() {
    assert_solves_ok("B = integral(J*sin(theta)/r^2, dV)", "J");
}
#[test]
fn t1456_09() {
    assert_solves_ok("P = integrate(I^2*R, t)", "I");
}
#[test]
fn t1456_10() {
    assert_solves_ok("E = integrate(c*B*sin(theta), A)", "B");
}

// T2+T3+T4+T5
#[test]
fn t2345_01() {
    assert_solves_ok("y = integral(x^2*sin(x), dx)", "x");
}
#[test]
#[ignore = "Requires symbolic differentiation — variable is the derivative variable"]
fn t2345_02() {
    assert_solves_ok("y = d(sqrt(sin(x)))/dx", "x");
}
#[test]
fn t2345_03() {
    assert_solves_ok("A = integral(r^2*cos(theta), dtheta)", "r");
}
#[test]
fn t2345_04() {
    assert_solves_ok("V = (4/3)*pi*r^3", "r");
}
#[test]
fn t2345_05() {
    assert_solves_ok("S = 4*pi*r^2", "r");
}
#[test]
fn t2345_06() {
    assert_solves_ok("E = integral(k*q/r^2, dr)", "q");
}
#[test]
fn t2345_07() {
    assert_solves_ok("V = integral(k*q/r, dr)", "q");
}
#[test]
fn t2345_08() {
    assert_solves_ok("I = integral(r^2*rho, dr)", "rho");
}
#[test]
fn t2345_09() {
    assert_solves_ok("M = integrate(r*rho, V)", "rho");
}
#[test]
fn t2345_10() {
    assert_solves_ok("Q = integral(rho*c*T, dV)", "T");
}

// T2+T3+T4+T6
#[test]
fn t2346_01() {
    assert_solves_ok("Z = sqrt(R^2+(wL)^2)*exp(i*phi)", "phi");
}
#[test]
#[ignore = "Variable not found — pc parsed as single identifier, not p*c"]
fn t2346_02() {
    assert_solves_ok("E = sqrt((pc)^2+(mc^2)^2)", "p");
}
#[test]
fn t2346_03() {
    assert_solves_ok("lambda_dB = h/sqrt(2*m*E)", "E");
}
#[test]
#[ignore = "Requires symbolic differentiation — variable is the derivative variable"]
fn t2346_04() {
    assert_solves_ok("v_g = d(omega)/d(k)", "k");
}
#[test]
fn t2346_05() {
    assert_solves_ok("n = c/v_p", "c");
}
#[test]
fn t2346_06() {
    assert_solves_ok("theta_B = atan(n2/n1)", "n1");
}
#[test]
fn t2346_07() {
    assert_solves_ok("R = (n1-n2)^2/(n1+n2)^2", "n1");
}
#[test]
fn t2346_08() {
    assert_solves_ok("T = 4*n1*n2/(n1+n2)^2", "n1");
}
#[test]
fn t2346_09() {
    assert_solves_ok("OPL = n*d*cos(theta)", "theta");
}
#[test]
fn t2346_10() {
    assert_solves_ok("delta = 2*n*d*cos(theta)", "theta");
}

// T2+T3+T5+T6
#[test]
fn t2356_01() {
    assert_solves_ok("E = integral(sigma*T^4, dA)", "T");
}
#[test]
fn t2356_02() {
    assert_solves_ok("M = integral(r^2*rho, dV)", "rho");
}
#[test]
fn t2356_03() {
    assert_solves_ok("I = integrate(r^2, m)", "r");
}
#[test]
fn t2356_04() {
    assert_solves_ok("E = integral(0.5*k*x^2, dx)", "k");
}
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t2356_05() {
    assert_solves_ok("U = integral(G*m/r, dm)", "m");
}
#[test]
fn t2356_06() {
    assert_solves_ok("KE = integral(0.5*I*w^2, dw)", "I");
}
#[test]
fn t2356_07() {
    assert_solves_ok("W = integrate(tau, theta)", "tau");
}
#[test]
fn t2356_08() {
    assert_solves_ok("Q = integrate(c*m, T)", "c");
}
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t2356_09() {
    assert_solves_ok("S = integral(c/T, dT)", "T");
}
#[test]
fn t2356_10() {
    assert_solves_ok("H = integrate(c_p, T)", "c_p");
}

// T2+T4+T5+T6
#[test]
fn t2456_01() {
    assert_solves_ok("y = integral(exp(-x^2)*sin(x), dx)", "x");
}
#[test]
#[ignore = "Domain-specific function not yet implemented — Y() spherical harmonic"]
fn t2456_02() {
    assert_solves_ok("psi = A*r^l*exp(-r/na)*Y(theta,phi)", "r");
}
#[test]
fn t2456_03() {
    assert_solves_ok("E = integral(k*cos(theta)/r^2, dr)", "theta");
}
#[test]
fn t2456_04() {
    assert_solves_ok("B = integral(mu_0*I*sin(theta)/(4*pi*r^2), dl)", "theta");
}
#[test]
fn t2456_05() {
    assert_solves_ok("Phi = integrate(B*cos(theta), A)", "theta");
}
#[test]
fn t2456_06() {
    assert_solves_ok("emf = integrate(v*B*sin(theta), l)", "theta");
}
#[test]
fn t2456_07() {
    assert_solves_ok("F = integrate(I*B*sin(theta), l)", "theta");
}
#[test]
fn t2456_08() {
    assert_solves_ok("W = integrate(tau*sin(theta), theta)", "tau");
}
#[test]
fn t2456_09() {
    assert_solves_ok("U = integrate(m*g*sin(theta), s)", "theta");
}
#[test]
fn t2456_10() {
    assert_solves_ok("P = integrate(F*v*cos(theta), t)", "theta");
}

// T3+T4+T5+T6
#[test]
fn t3456_01() {
    assert_solves_ok("y = integral(sin(x)^2*cos(x), dx)", "x");
}
#[test]
fn t3456_02() {
    assert_solves_ok("y = integral(tan(x)*sec(x)^2, dx)", "x");
}
#[test]
fn t3456_03() {
    assert_solves_ok("A = integral(sin(theta)^2*cos(theta), dtheta)", "theta");
}
#[test]
fn t3456_04() {
    assert_solves_ok("V = integral(pi*sin(x)^2*cos(x), dx)", "x");
}
#[test]
fn t3456_05() {
    assert_solves_ok("I = integral(cos(theta)^2*sin(theta), dtheta)", "theta");
}
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t3456_06() {
    assert_solves_ok("E = integral(sin(2*theta)*cos(theta), dtheta)", "theta");
}
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t3456_07() {
    assert_solves_ok("M = integral(r*sin(theta)*cos(theta), dr)", "r");
}
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t3456_08() {
    assert_solves_ok("F = integral(sin(theta)*tan(theta), dtheta)", "theta");
}
#[test]
fn t3456_09() {
    assert_solves_ok("W = integral(cos(theta)^3, dtheta)", "theta");
}
#[test]
fn t3456_10() {
    assert_solves_ok("Q = integral(sin(theta)^3, dtheta)", "theta");
}

// ============================================================================
// FIVE-TIER: 6 combos × 10 = 60 tests
// ============================================================================

// T1+T2+T3+T4+T5
#[test]
fn t12345_01() {
    assert_solves_ok("y = integral(x*sin(x^2), dx)", "x");
}
#[test]
fn t12345_02() {
    assert_solves_ok("y = d(exp(sin(x^2)))/dx", "x");
}
#[test]
fn t12345_03() {
    assert_solves_ok("A = integral(r^2*sin(theta), dtheta)", "r");
}
#[test]
fn t12345_04() {
    assert_solves_ok("V = integral(pi*(R^2-r^2), dx)", "R");
}
#[test]
fn t12345_05() {
    assert_solves_ok("E = integral(k*q*cos(theta)/r^2, dr)", "q");
}
#[test]
fn t12345_06() {
    assert_solves_ok("B = integral(mu_0*I*sin(theta)/(4*pi*r^2), dl)", "I");
}
#[test]
fn t12345_07() {
    assert_solves_ok("W = integrate(F*cos(theta), x)", "F");
}
#[test]
fn t12345_08() {
    assert_solves_ok("Q = integrate(sigma*T^4*cos(theta), A)", "T");
}
#[test]
fn t12345_09() {
    assert_solves_ok("M = integral(r^2*rho*sin(theta), dV)", "rho");
}
#[test]
fn t12345_10() {
    assert_solves_ok("S = integral(c_p/T, dT)", "c_p");
}

// T1+T2+T3+T4+T6
#[test]
fn t12346_01() {
    assert_solves_ok("Z = sqrt(R^2+(w*L-1/(w*C))^2)", "w");
}
#[test]
#[ignore = "Rational equation — variable in denominator on both sides"]
fn t12346_02() {
    assert_solves_ok("phi = atan((w*L-1/(w*C))/R)", "w");
}
#[test]
fn t12346_03() {
    assert_solves_ok("P = V^2*cos(phi)/(2*Z)", "phi");
}
#[test]
fn t12346_04() {
    assert_solves_ok("Q_f = w_0*L/R", "w_0");
}
#[test]
fn t12346_05() {
    assert_solves_ok("BW = R/(2*pi*L)", "L");
}
#[test]
fn t12346_06() {
    assert_solves_ok("f_0 = 1/(2*pi*sqrt(L*C))", "L");
}
#[test]
fn t12346_07() {
    assert_solves_ok("I = V/sqrt(R^2+(w*L-1/(w*C))^2)", "w");
}
#[test]
fn t12346_08() {
    assert_solves_ok("V_R = I*R", "R");
}
#[test]
fn t12346_09() {
    assert_solves_ok("V_L = I*w*L", "w");
}
#[test]
fn t12346_10() {
    assert_solves_ok("V_C = I/(w*C)", "C");
}

// T1+T2+T3+T5+T6
#[test]
fn t12356_01() {
    assert_solves_ok("E = integrate(sigma*T^4, A)", "T");
}
#[test]
fn t12356_02() {
    assert_solves_ok("I_mom = integrate(r^2*rho, V)", "rho");
}
#[test]
#[ignore = "Target variable is the integration variable — requires symbolic integration"]
fn t12356_03() {
    assert_solves_ok("U = integrate(G*m/r, m)", "m");
}
#[test]
fn t12356_04() {
    assert_solves_ok("W = integrate(P, V)", "P");
}
#[test]
fn t12356_05() {
    assert_solves_ok("Q = integral(k*A*dT/dx, dt)", "k");
}
#[test]
fn t12356_06() {
    assert_solves_ok("S = integrate(1/T, Q)", "T");
}
#[test]
fn t12356_07() {
    assert_solves_ok("H = integrate(c_p, T)", "c_p");
}
#[test]
fn t12356_08() {
    assert_solves_ok("G = H-T*S", "H");
}
#[test]
fn t12356_09() {
    assert_solves_ok("F = -d(U)/dx", "U");
}
#[test]
#[ignore = "Requires symbolic differentiation — variable is the derivative variable"]
fn t12356_10() {
    assert_solves_ok("mu_chem = d(G)/d(N)", "N");
}

// T1+T2+T4+T5+T6
#[test]
fn t12456_01() {
    assert_solves_ok("psi = A*exp(-r/a)*sin(theta)*exp(i*phi)", "r");
}
#[test]
fn t12456_02() {
    assert_solves_ok("E = integral(k*q*cos(theta)/r^2, dr)", "k");
}
#[test]
fn t12456_03() {
    assert_solves_ok("B = integral(mu_0*I*sin(theta)/(4*pi*r^2), dl)", "mu_0");
}
#[test]
fn t12456_04() {
    assert_solves_ok("Phi = integrate(B*cos(theta)*r^2*sin(theta), theta)", "B");
}
#[test]
#[ignore = "Variable is a constant factor inside integral — requires factor extraction"]
fn t12456_05() {
    assert_solves_ok("F = integral(q*E + q*v*B*sin(theta), dt)", "q");
}
#[test]
fn t12456_06() {
    assert_solves_ok("W = integrate(F*r*sin(theta), theta)", "F");
}
#[test]
fn t12456_07() {
    assert_solves_ok("Q = integrate(sigma*cos(theta)*T^4*r^2, Omega)", "sigma");
}
#[test]
fn t12456_08() {
    assert_solves_ok("E = integrate(rho*c*T*r^2*sin(theta), V)", "rho");
}
#[test]
fn t12456_09() {
    assert_solves_ok("M = integrate(r^3*rho*sin(theta), V)", "rho");
}
#[test]
fn t12456_10() {
    assert_solves_ok("I = integrate(r^2*sin(theta)^2*rho, V)", "rho");
}

// T1+T3+T4+T5+T6
#[test]
fn t13456_01() {
    assert_solves_ok("y = integral(sin(x)^2*x, dx)", "x");
}
#[test]
fn t13456_02() {
    assert_solves_ok("y = integral(cos(x)*x^2, dx)", "x");
}
#[test]
fn t13456_03() {
    assert_solves_ok("A = integral(sin(theta)*cos(theta)^2, dtheta)", "theta");
}
#[test]
fn t13456_04() {
    assert_solves_ok("V = integral(pi*cos(x)^2*x, dx)", "x");
}
#[test]
fn t13456_05() {
    assert_solves_ok("E = integral(sin(theta)*cos(theta)/r^2, dr)", "r");
}
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t13456_06() {
    assert_solves_ok("M = integral(r*cos(theta)*sin(theta), dr)", "r");
}
#[test]
#[ignore = "Requires symbolic integration — variable is the integration variable"]
fn t13456_07() {
    assert_solves_ok("F = integral(sin(theta)*tan(theta)*r, dr)", "r");
}
#[test]
fn t13456_08() {
    assert_solves_ok("W = integral(cos(theta)^3*r, dr)", "r");
}
#[test]
fn t13456_09() {
    assert_solves_ok("Q = integral(sin(theta)^3*exp(r), dr)", "r");
}
#[test]
fn t13456_10() {
    assert_solves_ok("P = integral(cos(theta)^2*r^2, dr)", "r");
}

// T2+T3+T4+T5+T6
#[test]
fn t23456_01() {
    assert_solves_ok("y = integral(x^2*sin(x)*exp(x), dx)", "x");
}
#[test]
#[ignore = "Domain-specific function not yet implemented — P() Legendre polynomial"]
fn t23456_02() {
    assert_solves_ok("psi = r^l*exp(-r/na)*P(cos(theta))*exp(i*m*phi)", "r");
}
#[test]
fn t23456_03() {
    assert_solves_ok("E = integral(k*q*cos(theta)/r^2, dr)", "r");
}
#[test]
fn t23456_04() {
    assert_solves_ok("B = integral(mu_0*I*sin(theta)/(4*pi*r^2), dl)", "r");
}
#[test]
fn t23456_05() {
    assert_solves_ok("V = integral(rho*cos(theta)/(4*pi*epsilon*r), dV)", "r");
}
#[test]
fn t23456_06() {
    assert_solves_ok("F = integral(J*B*sin(theta)*r, dV)", "r");
}
#[test]
fn t23456_07() {
    assert_solves_ok("W = integral(tau*sin(theta)*r, dtheta)", "r");
}
#[test]
fn t23456_08() {
    assert_solves_ok("M = integral(rho*r^3*sin(theta), dV)", "rho");
}
#[test]
fn t23456_09() {
    assert_solves_ok("Q = integral(sigma*T^4*cos(theta)*r^2, dOmega)", "T");
}
#[test]
fn t23456_10() {
    assert_solves_ok("S = integral(c*ln(T)*rho, dV)", "c");
}

// ============================================================================
// SIX-TIER: T1+T2+T3+T4+T5+T6 (1 combo × 10 tests)
// ============================================================================

#[test]
#[ignore = "Domain-specific function not yet implemented — Y() spherical harmonic"]
fn t123456_01() {
    assert_solves_ok("psi = A*r^l*exp(-r/na)*Y(theta,phi)", "r");
}
#[test]
fn t123456_02() {
    assert_solves_ok("E = integrate(rho*c*T*r^2*sin(theta), V)", "T");
}
#[test]
fn t123456_03() {
    assert_solves_ok("F = integrate(q*(E+v*B*sin(theta)), t)", "q");
}
#[test]
fn t123456_04() {
    assert_solves_ok("W = integrate(F*r*cos(theta), r)", "F");
}
#[test]
fn t123456_05() {
    assert_solves_ok("Q = integrate(sigma*T^4*cos(theta)*r^2, Omega)", "sigma");
}
#[test]
fn t123456_06() {
    assert_solves_ok("H = integrate(J*E*r^2*sin(theta), V)", "J");
}
#[test]
fn t123456_07() {
    assert_solves_ok("S = integrate(c_p*ln(T)/T*rho, V)", "c_p");
}
#[test]
fn t123456_08() {
    assert_solves_ok("G = integrate(rho*g*r*sin(theta)*cos(phi), V)", "rho");
}
#[test]
fn t123456_09() {
    assert_solves_ok("M = integrate(r^3*rho*sin(theta)^2, V)", "rho");
}
#[test]
fn t123456_10() {
    assert_solves_ok(
        "P = integrate(sigma*T^4*r^2*cos(theta)*sin(theta), Omega)",
        "T",
    );
}
