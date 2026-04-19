//! Method of undetermined coefficients for non-homogeneous second-order ODEs.
//!
//! Solves `a·y'' + b·y' + c·y = f(x)` for constant-coefficient linear ODEs
//! where the forcing function `f(x)` is a polynomial, exponential, or
//! trigonometric function (or a combination thereof).
//!
//! The general solution is `y = y_h + y_p` where `y_h` is the homogeneous
//! solution and `y_p` is a particular solution found by undetermined
//! coefficients.

use crate::ast::{BinaryOp, Expression, Function, Variable};

use super::{ODEError, SecondOrderODE};

// ---------------------------------------------------------------------------
// Forcing function classification
// ---------------------------------------------------------------------------

/// Classification of a forcing function `f(x)`.
///
/// Only types that the method of undetermined coefficients can handle
/// symbolically with constant-coefficient ODEs are represented.
#[derive(Debug, Clone, PartialEq)]
pub enum ForcingType {
    /// Polynomial `p_n·xⁿ + … + p₁·x + p₀`.
    ///
    /// Stores the non-negative integer degree.
    Polynomial { degree: u32 },

    /// Pure exponential `A·e^(k·x)`.
    ///
    /// Stores the exponent coefficient `k`.
    Exponential { k: f64 },

    /// Sine `A·sin(k·x)`, cosine `A·cos(k·x)`, or `A·sin(k·x) + B·cos(k·x)`.
    ///
    /// Stores the angular frequency `k`.
    Trigonometric { k: f64 },
}

/// Identify the type of a forcing function expression in variable `x_var`.
///
/// Returns `None` when the expression does not match any supported type.
///
/// # Supported patterns
///
/// | Pattern | Recognised as |
/// |---------|---------------|
/// | `k` (constant) | `Polynomial { degree: 0 }` |
/// | `x`, `k·x`, `x^n`, polynomial sum/difference | `Polynomial { degree: n }` |
/// | `exp(k·x)` or `e^(k·x)` | `Exponential { k }` |
/// | `sin(k·x)` or `cos(k·x)` | `Trigonometric { k }` |
pub fn identify_forcing_function(expr: &Expression, x_var: &str) -> Option<ForcingType> {
    classify_forcing(expr, x_var)
}

/// Recursive classifier — returns `None` if the expression is not a
/// supported forcing type or mixes incompatible types.
fn classify_forcing(expr: &Expression, x: &str) -> Option<ForcingType> {
    match expr {
        // Pure constants → degree-0 polynomial
        Expression::Integer(_) | Expression::Float(_) | Expression::Rational(_) => {
            Some(ForcingType::Polynomial { degree: 0 })
        }

        // Variable x → degree-1 polynomial
        Expression::Variable(v) if v.name == x => Some(ForcingType::Polynomial { degree: 1 }),

        // Other variables (constants from the user's perspective) → degree 0
        Expression::Variable(_) => Some(ForcingType::Polynomial { degree: 0 }),

        // Negation — same type as inner
        Expression::Unary(_, inner) => classify_forcing(inner, x),

        // Power: x^n → polynomial of degree n; e^(kx) → exponential
        Expression::Power(base, exp) => classify_power(base, exp, x),

        // Function call: exp/sin/cos
        Expression::Function(func, args) => classify_function(func.clone(), args, x),

        // Binary: addition/subtraction can merge same-type forcings
        // Multiplication scales by a constant — keeps the inner type
        Expression::Binary(op, left, right) => classify_binary(*op, left, right, x),

        _ => None,
    }
}

/// Classify `base^exp` expressions.
fn classify_power(base: &Expression, exp: &Expression, x: &str) -> Option<ForcingType> {
    use crate::ast::Function as F;
    // e^(k·x) via Power(e_const, k*x)
    let is_e = matches!(base, Expression::Constant(crate::ast::SymbolicConstant::E))
        || matches!(base, Expression::Function(F::Exp, _));

    if is_e {
        let k = extract_linear_coeff_of_x(exp, x)?;
        return Some(ForcingType::Exponential { k });
    }

    // x^n — polynomial
    if base.contains_variable(x) && !exp.contains_variable(x) {
        let degree = eval_as_nonneg_integer(exp)?;
        return Some(ForcingType::Polynomial { degree });
    }

    None
}

/// Classify `Function(f, args)` expressions.
fn classify_function(
    func: crate::ast::Function,
    args: &[Expression],
    x: &str,
) -> Option<ForcingType> {
    use crate::ast::Function as F;
    match func {
        F::Exp => {
            let arg = args.first()?;
            let k = extract_linear_coeff_of_x(arg, x)?;
            Some(ForcingType::Exponential { k })
        }
        F::Sin | F::Cos => {
            let arg = args.first()?;
            let k = extract_linear_coeff_of_x(arg, x)?;
            Some(ForcingType::Trigonometric { k })
        }
        _ => None,
    }
}

/// Classify `left OP right` expressions.
fn classify_binary(
    op: BinaryOp,
    left: &Expression,
    right: &Expression,
    x: &str,
) -> Option<ForcingType> {
    match op {
        BinaryOp::Add | BinaryOp::Sub => {
            let lt = classify_forcing(left, x)?;
            let rt = classify_forcing(right, x)?;
            merge_forcing_types(lt, rt)
        }
        BinaryOp::Mul => {
            // Scalar * f(x) — keep the non-constant type
            let lt = classify_forcing(left, x)?;
            let rt = classify_forcing(right, x)?;
            if matches!(lt, ForcingType::Polynomial { degree: 0 }) {
                Some(rt)
            } else if matches!(rt, ForcingType::Polynomial { degree: 0 }) {
                Some(lt)
            } else {
                // Product of two non-trivial terms — not supported
                None
            }
        }
        _ => None,
    }
}

/// Merge two `ForcingType` values from an addition/subtraction.
///
/// Same-type operands are merged (polynomials: take max degree; trig/exp:
/// must match k).  Mixed types are rejected.
fn merge_forcing_types(a: ForcingType, b: ForcingType) -> Option<ForcingType> {
    match (&a, &b) {
        (ForcingType::Polynomial { degree: d1 }, ForcingType::Polynomial { degree: d2 }) => {
            Some(ForcingType::Polynomial {
                degree: (*d1).max(*d2),
            })
        }
        (ForcingType::Exponential { k: k1 }, ForcingType::Exponential { k: k2 })
            if (k1 - k2).abs() < 1e-12 =>
        {
            Some(ForcingType::Exponential { k: *k1 })
        }
        (ForcingType::Trigonometric { k: k1 }, ForcingType::Trigonometric { k: k2 })
            if (k1 - k2).abs() < 1e-12 =>
        {
            Some(ForcingType::Trigonometric { k: *k1 })
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Helpers: coefficient extraction and evaluation
// ---------------------------------------------------------------------------

/// Extract `k` from a linear expression `k·x`, `x`, or constant `0`.
///
/// Returns `Some(k)` for expressions of the form `k*x`, `x`, `-x`, or `0`.
/// Returns `None` for non-linear expressions in `x`.
fn extract_linear_coeff_of_x(expr: &Expression, x: &str) -> Option<f64> {
    match expr {
        Expression::Integer(0) => Some(0.0),
        Expression::Variable(v) if v.name == x => Some(1.0),
        Expression::Float(k) => {
            if *k == 0.0 {
                Some(0.0)
            } else {
                None
            }
        }
        Expression::Unary(crate::ast::UnaryOp::Neg, inner) => {
            extract_linear_coeff_of_x(inner, x).map(|k| -k)
        }
        Expression::Binary(BinaryOp::Mul, left, right) => {
            // k * x  or  x * k
            let left_is_const = !left.contains_variable(x);
            let right_is_const = !right.contains_variable(x);

            if left_is_const && matches!(right.as_ref(), Expression::Variable(v) if v.name == x) {
                eval_constant(left)
            } else if right_is_const
                && matches!(left.as_ref(), Expression::Variable(v) if v.name == x)
            {
                eval_constant(right)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Evaluate a constant expression (no variables) to `f64`.
fn eval_constant(expr: &Expression) -> Option<f64> {
    if expr.contains_variable("") {
        // Bail out if anything variable-like slips through
        return None;
    }
    match expr {
        Expression::Integer(n) => Some(*n as f64),
        Expression::Float(f) => Some(*f),
        Expression::Rational(r) => Some(*r.numer() as f64 / *r.denom() as f64),
        Expression::Unary(crate::ast::UnaryOp::Neg, inner) => eval_constant(inner).map(|v| -v),
        _ => expr.evaluate(&std::collections::HashMap::new()),
    }
}

/// Evaluate an expression to a non-negative integer (for polynomial degrees).
fn eval_as_nonneg_integer(expr: &Expression) -> Option<u32> {
    let v = eval_constant(expr)?;
    if v >= 0.0 && v.fract() == 0.0 && v <= 20.0 {
        Some(v as u32)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Particular solution by undetermined coefficients
// ---------------------------------------------------------------------------

/// Find a particular solution `y_p` for a constant-coefficient second-order
/// non-homogeneous ODE using the method of undetermined coefficients.
///
/// The ODE has the form `a·y'' + b·y' + c·y = f(x)`.
///
/// # Errors
///
/// - [`ODEError::CannotSolve`] if `f(x)` is not a supported forcing type.
/// - [`ODEError::ResonanceDetected`] if simple resonance cannot be resolved
///   (double-resonance case).
pub fn particular_solution_undetermined(
    ode: &SecondOrderODE,
) -> Result<(Expression, Vec<String>), ODEError> {
    let forcing_expr = ode.forcing_expr();
    let forcing_type = identify_forcing_function(&forcing_expr, &ode.independent)
        .ok_or_else(|| ODEError::CannotSolve("forcing function type not supported".to_string()))?;

    let mut steps = Vec::new();
    steps.push(format!(
        "Non-homogeneous ODE: {}·{}'' + {}·{}' + {}·{} = {}",
        ode.a,
        ode.dependent,
        ode.b,
        ode.dependent,
        ode.c,
        ode.dependent,
        forcing_type_display(&forcing_type),
    ));
    steps.push("Apply method of undetermined coefficients".to_string());

    let result = match &forcing_type {
        ForcingType::Polynomial { degree } => particular_polynomial(ode, *degree, &mut steps)?,
        ForcingType::Exponential { k } => particular_exponential(ode, *k, &mut steps)?,
        ForcingType::Trigonometric { k } => particular_trig(ode, *k, &mut steps)?,
    };

    Ok((result, steps))
}

/// Human-readable label for a `ForcingType`.
fn forcing_type_display(ft: &ForcingType) -> &'static str {
    match ft {
        ForcingType::Polynomial { .. } => "polynomial",
        ForcingType::Exponential { .. } => "exponential",
        ForcingType::Trigonometric { .. } => "trigonometric",
    }
}

// ---------------------------------------------------------------------------
// Particular solution: polynomial forcing
// ---------------------------------------------------------------------------

/// Find `y_p` when `f(x)` is a polynomial of degree `d`.
///
/// The trial form is `A_d·x^d + … + A_0`.  If `c = 0` (zero is a
/// characteristic root), we multiply by `x` (or `x²` for double root).
fn particular_polynomial(
    ode: &SecondOrderODE,
    degree: u32,
    steps: &mut Vec<String>,
) -> Result<Expression, ODEError> {
    let x_var = &ode.independent;

    // Determine resonance multiplier
    let multiplier = resonance_multiplier_polynomial(ode)?;
    let trial_degree = degree + multiplier;

    steps.push(format!(
        "Trial form: y_p = polynomial of degree {} (multiplier x^{})",
        trial_degree, multiplier
    ));

    // Collect coefficients from the forcing polynomial
    let forcing_expr = ode.forcing_expr();
    let forcing_coeffs = extract_polynomial_coeffs(&forcing_expr, x_var, degree)?;

    // Solve for undetermined coefficients by matching powers of x
    let yp_coeffs = solve_polynomial_system(ode, &forcing_coeffs, multiplier)?;

    steps.push(format!("Particular solution coefficients: {:?}", yp_coeffs));

    Ok(build_polynomial_expr(&yp_coeffs, multiplier, x_var))
}

/// Return the power of `x` by which the trial must be multiplied.
///
/// Returns `0`, `1`, or `2` based on whether `0` is a simple/double or
/// no characteristic root.
fn resonance_multiplier_polynomial(ode: &SecondOrderODE) -> Result<u32, ODEError> {
    const EPS: f64 = 1e-12;
    // Characteristic roots are roots of ar² + br + c = 0.
    // Root = 0 iff c = 0 (and a ≠ 0).
    if ode.c.abs() > EPS {
        return Ok(0); // 0 is not a root
    }
    // c = 0: r = 0 is a root.  If b = 0 as well, double root at 0.
    if ode.b.abs() > EPS {
        Ok(1)
    } else {
        // b = 0 and c = 0: characteristic eq is a·r² = 0, double root at 0
        Ok(2)
    }
}

/// Extract polynomial coefficients `[c0, c1, …, cn]` (constant term first)
/// by symbolic evaluation of `f(x)` at `n+1` distinct points.
///
/// This is a Vandermonde interpolation — works for polynomials of degree ≤ 20.
fn extract_polynomial_coeffs(
    expr: &Expression,
    x_var: &str,
    degree: u32,
) -> Result<Vec<f64>, ODEError> {
    let n = (degree + 1) as usize;
    let points: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let values: Vec<f64> = points
        .iter()
        .map(|&xi| {
            let mut env = std::collections::HashMap::new();
            env.insert(x_var.to_string(), xi);
            expr.evaluate(&env).unwrap_or(0.0)
        })
        .collect();

    // Vandermonde solve (Gaussian elimination for small n)
    vandermonde_solve(&points, &values).ok_or_else(|| {
        ODEError::CannotSolve("failed to extract polynomial coefficients".to_string())
    })
}

/// Solve `y_p` coefficients from the ODE constraint for polynomial forcing.
///
/// Given `y_p = Σ a_j · x^(j+m)` where `m` = multiplier, substitute into
/// the ODE and match coefficients of each power of `x`.
fn solve_polynomial_system(
    ode: &SecondOrderODE,
    forcing: &[f64],
    multiplier: u32,
) -> Result<Vec<f64>, ODEError> {
    let n = forcing.len();
    let m = multiplier as usize;
    let total = n + m; // highest power in trial is (n-1) + m

    // Build coefficients for each power in the trial y_p = Σ A_j x^(j+m)
    // y_p'' contribution and y_p' contribution are computed analytically.
    let mut coeffs = vec![0.0f64; n];

    // Work from highest to lowest power to determine A_{n-1}, …, A_0
    // For power x^(j+m), the equation is:
    //   a·(j+m)(j+m-1)·A_j + b·(j+m)·A_{j} + c·A_j
    //   + contributions from higher A_k = f_j
    // We do back-substitution (highest degree first).
    let mut determined = vec![0.0f64; n];

    for j in (0..n).rev() {
        let power = (j + m) as f64;
        // Direct contribution of A_j to this power's equation:
        let coeff_a_j = ode.a * power * (power - 1.0) + ode.b * power + ode.c;

        // Contribution from A_{j+1} (one degree higher) to this power
        // via first derivative: b*(j+1+m)*A_{j+1}*x^(j+m) — already determined
        let mut rhs = forcing[j];
        if j + 1 < n {
            // a·(j+1+m)·(j+m)·A_{j+1}  (second deriv drops power by 2 → not relevant here)
            // b·(j+1+m)·A_{j+1}  (first deriv of x^(j+1+m) = (j+1+m)·x^(j+m))
            let p1 = (j + 1 + m) as f64;
            rhs -= ode.b * p1 * determined[j + 1];
        }
        if j + 2 < n {
            // a·(j+2+m)·(j+1+m)·A_{j+2}  (second deriv of x^(j+2+m) = (j+2+m)(j+1+m)·x^(j+m))
            let p2 = (j + 2 + m) as f64;
            let p2m1 = (j + 1 + m) as f64;
            rhs -= ode.a * p2 * p2m1 * determined[j + 2];
        }

        if coeff_a_j.abs() < 1e-12 {
            return Err(ODEError::ResonanceDetected(format!(
                "Undetermined coefficient for x^{} vanishes — higher-order resonance not handled",
                j + m
            )));
        }
        determined[j] = rhs / coeff_a_j;
        coeffs[j] = determined[j];
    }

    Ok(coeffs)
}

/// Build `Σ a_j · x^(j+m)` as an `Expression`.
fn build_polynomial_expr(coeffs: &[f64], multiplier: u32, x_var: &str) -> Expression {
    let m = multiplier as i64;
    let mut terms: Vec<Expression> = Vec::new();

    for (j, &a_j) in coeffs.iter().enumerate() {
        if a_j.abs() < 1e-15 {
            continue;
        }
        let power = j as i64 + m;
        let x_pow = build_x_power(x_var, power);
        let term = if (a_j - 1.0).abs() < 1e-15 {
            x_pow
        } else {
            Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Float(a_j)),
                Box::new(x_pow),
            )
        };
        terms.push(term);
    }

    terms_to_sum(terms)
}

/// Build `x^n` as an `Expression`.
fn build_x_power(x_var: &str, n: i64) -> Expression {
    let x = Expression::Variable(Variable::new(x_var));
    match n {
        0 => Expression::Integer(1),
        1 => x,
        _ => Expression::Power(Box::new(x), Box::new(Expression::Integer(n))),
    }
}

// ---------------------------------------------------------------------------
// Particular solution: exponential forcing
// ---------------------------------------------------------------------------

/// Find `y_p` when `f(x) = A·e^(k·x)`.
///
/// Trial: `y_p = B·e^(k·x)`.  If `k` is a simple characteristic root,
/// multiply by `x`; if double root, multiply by `x²`.
fn particular_exponential(
    ode: &SecondOrderODE,
    k: f64,
    steps: &mut Vec<String>,
) -> Result<Expression, ODEError> {
    let x_var = &ode.independent;
    let multiplier = resonance_multiplier_exp(ode, k)?;

    steps.push(format!(
        "Trial form: y_p = B·x^{}·e^({}·{})",
        multiplier, k, x_var
    ));

    // Evaluate forcing at one point to get amplitude A
    let amp = {
        let mut env = std::collections::HashMap::new();
        env.insert(x_var.to_string(), 1.0);
        ode.forcing_expr().evaluate(&env).unwrap_or(1.0) / (k * 1.0_f64).exp()
    };

    // Compute denominator: substitute y_p = B·x^m·e^(kx) into ODE, divide by e^(kx)
    // For m=0: a·k² + b·k + c
    // For m=1: 2a·k + b
    // For m=2: 2a
    let denom = match multiplier {
        0 => ode.a * k * k + ode.b * k + ode.c,
        1 => 2.0 * ode.a * k + ode.b,
        _ => 2.0 * ode.a,
    };

    if denom.abs() < 1e-12 {
        return Err(ODEError::ResonanceDetected(
            "Triple resonance in exponential case — not supported".to_string(),
        ));
    }

    let b_coeff = amp / denom;
    steps.push(format!("Coefficient B = {}", b_coeff));

    Ok(build_exp_particular(b_coeff, k, multiplier, x_var))
}

/// Determine resonance multiplier for exponential forcing `e^(k·x)`.
fn resonance_multiplier_exp(ode: &SecondOrderODE, k: f64) -> Result<u32, ODEError> {
    const EPS: f64 = 1e-10;
    use super::second_order::solve_characteristic_equation;

    let roots = solve_characteristic_equation(ode.a, ode.b, ode.c)
        .map_err(|e| ODEError::CannotSolve(e.to_string()))?;

    let matches_r1 = (roots.r1 - k).abs() < EPS;
    let matches_r2 = (roots.r2 - k).abs() < EPS;

    match (matches_r1, matches_r2) {
        (false, false) => Ok(0),
        (true, false) | (false, true) => Ok(1),
        (true, true) => Ok(2),
    }
}

/// Build `B · x^m · e^(k·x)` as an `Expression`.
fn build_exp_particular(b: f64, k: f64, multiplier: u32, x_var: &str) -> Expression {
    let exp_arg = Expression::Binary(
        BinaryOp::Mul,
        Box::new(Expression::Float(k)),
        Box::new(Expression::Variable(Variable::new(x_var))),
    );
    let exp_term = Expression::Function(Function::Exp, vec![exp_arg]);

    let base = if (b - 1.0).abs() < 1e-15 {
        exp_term
    } else {
        Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Float(b)),
            Box::new(exp_term),
        )
    };

    if multiplier == 0 {
        return base;
    }

    let x_pow = build_x_power(x_var, multiplier as i64);
    Expression::Binary(BinaryOp::Mul, Box::new(x_pow), Box::new(base))
}

// ---------------------------------------------------------------------------
// Particular solution: trigonometric forcing
// ---------------------------------------------------------------------------

/// Find `y_p` when `f(x) = A·sin(k·x) + B·cos(k·x)`.
///
/// Trial: `y_p = P·cos(k·x) + Q·sin(k·x)`.  If `±ki` are characteristic
/// roots (pure imaginary, which means `b=0` and `c/a = k²`), we multiply
/// by `x`.
fn particular_trig(
    ode: &SecondOrderODE,
    k: f64,
    steps: &mut Vec<String>,
) -> Result<Expression, ODEError> {
    let x_var = &ode.independent;
    let resonant = is_trig_resonant(ode, k);
    let multiplier = if resonant { 1_u32 } else { 0_u32 };

    steps.push(format!(
        "Trial form: y_p = x^{}·(P·cos({}·{}) + Q·sin({}·{}))",
        multiplier, k, x_var, k, x_var
    ));

    // Extract sin and cos amplitudes from forcing at two points
    let forcing_expr = ode.forcing_expr();
    let (f_sin, f_cos) = extract_trig_amplitudes(&forcing_expr, x_var, k)?;

    let (p, q) = solve_trig_system(ode, k, f_sin, f_cos, resonant)?;

    steps.push(format!(
        "Trig coefficients: P (cos) = {:.6}, Q (sin) = {:.6}",
        p, q
    ));

    Ok(build_trig_particular(p, q, k, multiplier, x_var))
}

/// Return `true` if `±ki` are characteristic roots of `a·r² + b·r + c = 0`.
///
/// This holds exactly when `b = 0` and `c = a·k²`.
fn is_trig_resonant(ode: &SecondOrderODE, k: f64) -> bool {
    const EPS: f64 = 1e-10;
    ode.b.abs() < EPS && (ode.c - ode.a * k * k).abs() < EPS
}

/// Extract sin-amplitude `f_s` and cos-amplitude `f_c` from `f(x)` such that
/// `f(x) ≈ f_s·sin(kx) + f_c·cos(kx)`.
fn extract_trig_amplitudes(
    forcing: &Expression,
    x_var: &str,
    k: f64,
) -> Result<(f64, f64), ODEError> {
    let pi = std::f64::consts::PI;
    // Evaluate at x = π/(2k) and x = 0 to separate sin and cos components
    let x1 = if k.abs() > 1e-12 { pi / (2.0 * k) } else { 1.0 };
    let x0 = 0.0_f64;

    let eval_at = |xi: f64| -> f64 {
        let mut env = std::collections::HashMap::new();
        env.insert(x_var.to_string(), xi);
        forcing.evaluate(&env).unwrap_or(0.0)
    };

    let f_at_0 = eval_at(x0); // = f_c·cos(0) + f_s·sin(0) = f_c
    let f_at_x1 = eval_at(x1); // = f_c·cos(π/2) + f_s·sin(π/2) = f_s

    Ok((f_at_x1, f_at_0))
}

/// Solve the 2×2 linear system for trig undetermined coefficients.
fn solve_trig_system(
    ode: &SecondOrderODE,
    k: f64,
    f_sin: f64,
    f_cos: f64,
    resonant: bool,
) -> Result<(f64, f64), ODEError> {
    // Non-resonant: y_p = P·cos(kx) + Q·sin(kx)
    // Substitute and collect:
    //   cos terms: (c - a·k²)·P + b·k·Q = f_cos
    //   sin terms: -b·k·P + (c - a·k²)·Q = f_sin
    if !resonant {
        let alpha = ode.c - ode.a * k * k;
        let beta = ode.b * k;

        let det = alpha * alpha + beta * beta;
        if det.abs() < 1e-12 {
            return Err(ODEError::ResonanceDetected(
                "Resonance in trig system — try resonant trial".to_string(),
            ));
        }
        let p = (alpha * f_cos + beta * f_sin) / det;
        let q = (-beta * f_cos + alpha * f_sin) / det;
        Ok((p, q))
    } else {
        // Resonant: y_p = x·(P·cos(kx) + Q·sin(kx))
        // After substitution, the 2×2 system becomes:
        //   2a·k·Q + b·(P·1 + …)  — for pure b=0 case:
        //   cos terms: 2a·k·Q = f_cos  →  Q = f_cos / (2ak)
        //   sin terms: -2a·k·P = f_sin → P = -f_sin / (2ak)
        let denom = 2.0 * ode.a * k;
        if denom.abs() < 1e-12 {
            return Err(ODEError::ResonanceDetected(
                "Cannot determine trig coefficients — degenerate resonance".to_string(),
            ));
        }
        let q = f_cos / denom;
        let p = -f_sin / denom;
        Ok((p, q))
    }
}

/// Build `x^m·(P·cos(k·x) + Q·sin(k·x))` as an `Expression`.
fn build_trig_particular(p: f64, q: f64, k: f64, multiplier: u32, x_var: &str) -> Expression {
    let kx = Expression::Binary(
        BinaryOp::Mul,
        Box::new(Expression::Float(k)),
        Box::new(Expression::Variable(Variable::new(x_var))),
    );
    let cos_term = Expression::Function(Function::Cos, vec![kx.clone()]);
    let sin_term = Expression::Function(Function::Sin, vec![kx]);

    let p_cos = if (p - 1.0).abs() < 1e-15 {
        cos_term
    } else {
        Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Float(p)),
            Box::new(cos_term),
        )
    };

    let q_sin = if (q - 1.0).abs() < 1e-15 {
        sin_term
    } else {
        Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Float(q)),
            Box::new(sin_term),
        )
    };

    let trig_sum = Expression::Binary(BinaryOp::Add, Box::new(p_cos), Box::new(q_sin));

    if multiplier == 0 {
        trig_sum
    } else {
        let x_pow = build_x_power(x_var, multiplier as i64);
        Expression::Binary(BinaryOp::Mul, Box::new(x_pow), Box::new(trig_sum))
    }
}

// ---------------------------------------------------------------------------
// Vandermonde solve (Lagrange interpolation for polynomial coefficients)
// ---------------------------------------------------------------------------

/// Solve the Vandermonde system `V·c = y` where `V[i][j] = x_i^j`.
///
/// Returns the coefficient vector `[c_0, c_1, …, c_{n-1}]` or `None` if
/// the system is singular.
fn vandermonde_solve(xs: &[f64], ys: &[f64]) -> Option<Vec<f64>> {
    let n = xs.len();
    if n == 0 {
        return None;
    }
    // Build augmented matrix [V | y]
    let mut mat: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row: Vec<f64> = (0..n).map(|j| xs[i].powi(j as i32)).collect();
            row.push(ys[i]);
            row
        })
        .collect();

    gaussian_eliminate(&mut mat, n)
}

/// Gaussian elimination with partial pivoting; returns solution vector.
fn gaussian_eliminate(mat: &mut Vec<Vec<f64>>, n: usize) -> Option<Vec<f64>> {
    const EPS: f64 = 1e-12;

    for col in 0..n {
        // Partial pivot
        let max_row =
            (col..n).max_by(|&a, &b| mat[a][col].abs().partial_cmp(&mat[b][col].abs()).unwrap())?;
        mat.swap(col, max_row);

        let pivot = mat[col][col];
        if pivot.abs() < EPS {
            return None;
        }

        for row in (col + 1)..n {
            let factor = mat[row][col] / pivot;
            for k in col..=n {
                let v = mat[col][k];
                mat[row][k] -= factor * v;
            }
        }
    }

    // Back substitution
    let mut result = vec![0.0; n];
    for row in (0..n).rev() {
        let mut val = mat[row][n];
        for k in (row + 1)..n {
            val -= mat[row][k] * result[k];
        }
        result[row] = val / mat[row][row];
    }
    Some(result)
}

// ---------------------------------------------------------------------------
// Utility: build sum from terms
// ---------------------------------------------------------------------------

/// Fold a `Vec<Expression>` into a left-associated sum.
/// Returns `Expression::Integer(0)` if empty.
fn terms_to_sum(terms: Vec<Expression>) -> Expression {
    if terms.is_empty() {
        return Expression::Integer(0);
    }
    terms
        .into_iter()
        .reduce(|acc, t| Expression::Binary(BinaryOp::Add, Box::new(acc), Box::new(t)))
        .expect("non-empty iterator always reduces")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ode::SecondOrderODE;

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn make_sin_x() -> Expression {
        Expression::Function(Function::Sin, vec![var("x")])
    }

    fn make_exp_kx(k: f64) -> Expression {
        let kx = Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Float(k)),
            Box::new(var("x")),
        );
        Expression::Function(Function::Exp, vec![kx])
    }

    // ------------------------------------------------------------------
    // identify_forcing_function
    // ------------------------------------------------------------------

    #[test]
    fn classify_constant_is_poly0() {
        let ft = identify_forcing_function(&Expression::Integer(3), "x");
        assert_eq!(ft, Some(ForcingType::Polynomial { degree: 0 }));
    }

    #[test]
    fn classify_x_is_poly1() {
        let ft = identify_forcing_function(&var("x"), "x");
        assert_eq!(ft, Some(ForcingType::Polynomial { degree: 1 }));
    }

    #[test]
    fn classify_x_squared_is_poly2() {
        let x2 = Expression::Power(Box::new(var("x")), Box::new(Expression::Integer(2)));
        let ft = identify_forcing_function(&x2, "x");
        assert_eq!(ft, Some(ForcingType::Polynomial { degree: 2 }));
    }

    #[test]
    fn classify_exp_is_exponential() {
        let ft = identify_forcing_function(&make_exp_kx(3.0), "x");
        assert_eq!(ft, Some(ForcingType::Exponential { k: 3.0 }));
    }

    #[test]
    fn classify_sin_is_trig() {
        let ft = identify_forcing_function(&make_sin_x(), "x");
        assert_eq!(ft, Some(ForcingType::Trigonometric { k: 1.0 }));
    }

    // ------------------------------------------------------------------
    // particular_solution_undetermined — polynomial forcing
    // ------------------------------------------------------------------

    #[test]
    fn particular_constant_forcing() {
        // y'' + y = 1  →  y_p = 1
        let ode = SecondOrderODE::new("y", "x", 1.0, 0.0, 1.0, Expression::Integer(1));
        let (yp, _steps) = particular_solution_undetermined(&ode).unwrap();

        let mut env = std::collections::HashMap::new();
        env.insert("x".to_string(), 0.0);
        let val = yp.evaluate(&env).unwrap_or(f64::NAN);
        assert!((val - 1.0).abs() < 1e-9, "y_p(0) should be 1, got {val}");
    }

    #[test]
    fn particular_linear_polynomial_forcing() {
        // y'' + y = x  →  y_p = x
        let ode = SecondOrderODE::new("y", "x", 1.0, 0.0, 1.0, var("x"));
        let (yp, _steps) = particular_solution_undetermined(&ode).unwrap();

        let eval_at = |xi: f64| {
            let mut env = std::collections::HashMap::new();
            env.insert("x".to_string(), xi);
            yp.evaluate(&env).unwrap_or(f64::NAN)
        };
        assert!((eval_at(2.0) - 2.0).abs() < 1e-9);
        assert!((eval_at(5.0) - 5.0).abs() < 1e-9);
    }

    // ------------------------------------------------------------------
    // particular_solution_undetermined — exponential forcing
    // ------------------------------------------------------------------

    #[test]
    fn particular_exponential_no_resonance() {
        // y'' - y = e^(2x)  →  y_p = (1/3)·e^(2x)
        // char roots: ±1, k=2 is not a root
        let ode = SecondOrderODE::new("y", "x", 1.0, 0.0, -1.0, make_exp_kx(2.0));
        let (yp, _steps) = particular_solution_undetermined(&ode).unwrap();

        let eval_at = |xi: f64| {
            let mut env = std::collections::HashMap::new();
            env.insert("x".to_string(), xi);
            yp.evaluate(&env).unwrap_or(f64::NAN)
        };
        let expected_at_1 = (1.0_f64 / 3.0) * 2.0_f64.exp();
        assert!((eval_at(1.0) - expected_at_1).abs() < 1e-9);
    }

    #[test]
    fn particular_exponential_resonance() {
        // y'' - y = e^x  →  k=1 is a characteristic root  →  y_p = (x/2)·e^x
        let ode = SecondOrderODE::new("y", "x", 1.0, 0.0, -1.0, make_exp_kx(1.0));
        let (yp, _steps) = particular_solution_undetermined(&ode).unwrap();

        // Verify by substituting into ODE numerically at x=1
        // y_p = (x/2)·e^x; y_p' = (1/2 + x/2)·e^x; y_p'' = (1 + x/2)·e^x
        // y_p'' - y_p = (1 + x/2)·e^x - (x/2)·e^x = e^x ✓
        let eval_at = |xi: f64| {
            let mut env = std::collections::HashMap::new();
            env.insert("x".to_string(), xi);
            yp.evaluate(&env).unwrap_or(f64::NAN)
        };
        let expected_at_1 = 0.5 * std::f64::consts::E;
        assert!((eval_at(1.0) - expected_at_1).abs() < 1e-9);
    }

    // ------------------------------------------------------------------
    // particular_solution_undetermined — trig forcing
    // ------------------------------------------------------------------

    #[test]
    fn particular_trig_no_resonance() {
        // y'' + 4y = sin(x)  →  y_p = (1/3)·sin(x)
        // Non-resonant: char roots ±2i, k=1 ≠ 2
        let ode = SecondOrderODE::new(
            "y",
            "x",
            1.0,
            0.0,
            4.0,
            Expression::Function(Function::Sin, vec![var("x")]),
        );
        let (yp, _steps) = particular_solution_undetermined(&ode).unwrap();

        let eval_at = |xi: f64| {
            let mut env = std::collections::HashMap::new();
            env.insert("x".to_string(), xi);
            yp.evaluate(&env).unwrap_or(f64::NAN)
        };
        // y_p(π/2) should be (1/3)·sin(π/2) = 1/3
        let expected = 1.0 / 3.0;
        assert!(
            (eval_at(std::f64::consts::FRAC_PI_2) - expected).abs() < 1e-9,
            "expected {expected}, got {}",
            eval_at(std::f64::consts::FRAC_PI_2)
        );
    }

    #[test]
    fn particular_trig_resonance() {
        // y'' + y = sin(x)  →  resonant (k=1, char roots ±i)
        // y_p = (-x/2)·cos(x)
        let ode = SecondOrderODE::new(
            "y",
            "x",
            1.0,
            0.0,
            1.0,
            Expression::Function(Function::Sin, vec![var("x")]),
        );
        let result = particular_solution_undetermined(&ode);
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
        let (yp, _steps) = result.unwrap();

        // Numerical check: y_p(1) = -0.5·cos(1)
        let expected = -0.5 * 1.0_f64.cos();
        let mut env = std::collections::HashMap::new();
        env.insert("x".to_string(), 1.0);
        let val = yp.evaluate(&env).unwrap_or(f64::NAN);
        assert!(
            (val - expected).abs() < 1e-9,
            "expected {expected}, got {val}"
        );
    }
}
