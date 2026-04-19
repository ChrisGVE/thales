//! Method of undetermined coefficients for non-homogeneous second-order ODEs.
//!
//! Solves `a·y'' + b·y' + c·y = f(x)` for constant-coefficient linear ODEs
//! where the forcing function `f(x)` is a polynomial, exponential, or
//! trigonometric function (or a combination thereof).
//!
//! The general solution is `y = y_h + y_p` where `y_h` is the homogeneous
//! solution and `y_p` is a particular solution found by undetermined
//! coefficients.

use std::sync::Arc;

use crate::ast::{Expression, SymbolicConstant};
use crate::numeric::compile::compile;
use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::SymbolId;
use crate::solver::helpers::contains_symbol;
use num::traits::Zero;

use super::{ODEError, SecondOrderODE};

mod exponential;
mod polynomial;
mod trigonometric;

use exponential::particular_exponential;
use polynomial::particular_polynomial;
use trigonometric::particular_trig;

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
    let arc = compile(expr);
    let x = SymbolId::intern(x_var);
    classify_forcing_expr(&arc, x)
}

/// Arc<Expr>-native forcing classifier. Walks the canonical `Expr` tree
/// directly; `Add` and `Mul` dispatch by merging the per-term
/// classifications (addition: same type only; multiplication: scalar
/// times a single non-trivial forcing).
fn classify_forcing_expr(expr: &Arc<Expr>, x: SymbolId) -> Option<ForcingType> {
    match expr.as_ref() {
        // Pure numeric leaves → degree-0 polynomial.
        Expr::Integer(_) | Expr::Float(_) | Expr::Rational(_) | Expr::Complex(_) => {
            Some(ForcingType::Polynomial { degree: 0 })
        }
        // E itself compiles as a Constant leaf — no forcing shape.
        Expr::Constant(_) => Some(ForcingType::Polynomial { degree: 0 }),

        // `x` → degree 1; other symbols → parameter, treated as constant.
        Expr::Symbol(s) if *s == x => Some(ForcingType::Polynomial { degree: 1 }),
        Expr::Symbol(_) => Some(ForcingType::Polynomial { degree: 0 }),

        // Power: x^n or e^(kx).
        Expr::Pow(base, exp) => classify_power_expr(base, exp, x),

        // Function call: exp / sin / cos.
        Expr::Func(id, args) => classify_function_expr(*id, args, x),

        // Sum — merge per-term classifications plus the scalar constant.
        Expr::Add(node) => {
            let mut acc = ForcingType::Polynomial { degree: 0 };
            for (term, _coeff) in &node.terms {
                let t = classify_forcing_expr(term, x)?;
                acc = merge_forcing_types(acc, t)?;
            }
            Some(acc)
        }

        // Product — scalar × single non-trivial factor.
        Expr::Mul(node) => {
            let mut acc = ForcingType::Polynomial { degree: 0 };
            for (base, exp) in &node.factors {
                let factor_arc: Arc<Expr> = if matches!(exp.as_ref(), Expr::Integer(n) if n.to_i64() == Some(1))
                {
                    base.clone()
                } else {
                    Arc::new(Expr::Pow(base.clone(), exp.clone()))
                };
                let t = classify_forcing_expr(&factor_arc, x)?;
                acc = merge_forcing_types_mul(acc, t)?;
            }
            Some(acc)
        }
    }
}

fn classify_power_expr(base: &Arc<Expr>, exp: &Arc<Expr>, x: SymbolId) -> Option<ForcingType> {
    let is_e = matches!(base.as_ref(), Expr::Constant(SymbolicConstant::E))
        || matches!(base.as_ref(), Expr::Func(FuncId::Exp, _));
    if is_e {
        let k = extract_linear_coeff_of_x_expr(exp, x)?;
        return Some(ForcingType::Exponential { k });
    }

    // x^n — polynomial.
    if contains_symbol(base, x) && !contains_symbol(exp, x) {
        let degree = eval_as_nonneg_integer_expr(exp)?;
        return Some(ForcingType::Polynomial { degree });
    }

    None
}

fn classify_function_expr(id: FuncId, args: &[Arc<Expr>], x: SymbolId) -> Option<ForcingType> {
    match id {
        FuncId::Exp => {
            let arg = args.first()?;
            let k = extract_linear_coeff_of_x_expr(arg, x)?;
            Some(ForcingType::Exponential { k })
        }
        FuncId::Sin | FuncId::Cos => {
            let arg = args.first()?;
            let k = extract_linear_coeff_of_x_expr(arg, x)?;
            Some(ForcingType::Trigonometric { k })
        }
        _ => None,
    }
}

/// Merge two forcing types under multiplication: scalar × f keeps the
/// non-trivial operand; product of two non-trivial terms is not supported.
fn merge_forcing_types_mul(a: ForcingType, b: ForcingType) -> Option<ForcingType> {
    if matches!(a, ForcingType::Polynomial { degree: 0 }) {
        return Some(b);
    }
    if matches!(b, ForcingType::Polynomial { degree: 0 }) {
        return Some(a);
    }
    None
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

/// Extract `k` from `k·x`, `x`, `-x`, or a zero constant (Arc<Expr>).
///
/// Returns `Some(k)` for supported linear shapes. In the canonical
/// `Expr` form `k·x` shows up as a `MulNode` with a single factor
/// `(Symbol(x), 1)` and a rational coefficient.
fn extract_linear_coeff_of_x_expr(expr: &Arc<Expr>, x: SymbolId) -> Option<f64> {
    match expr.as_ref() {
        Expr::Integer(n) if n.to_i64() == Some(0) => Some(0.0),
        Expr::Float(f) if *f == 0.0 => Some(0.0),
        Expr::Rational(r) if r.numer().is_zero() => Some(0.0),
        Expr::Symbol(s) if *s == x => Some(1.0),
        Expr::Mul(node) => {
            // Accept any product of the form `(rational · float · int … · x)`:
            // the coefficient field holds the rational part; any non-symbol
            // factor must be a numeric leaf and gets folded into `k`.
            let mut k = node.coeff.to_f64();
            let mut x_seen = false;
            for (base, exp) in &node.factors {
                if !matches!(exp.as_ref(), Expr::Integer(n) if n.to_i64() == Some(1)) {
                    return None;
                }
                match base.as_ref() {
                    Expr::Symbol(s) if *s == x => {
                        if x_seen {
                            return None;
                        }
                        x_seen = true;
                    }
                    Expr::Float(f) => k *= *f,
                    Expr::Integer(n) => k *= n.to_i64()? as f64,
                    Expr::Rational(r) => k *= r.to_f64(),
                    _ => return None,
                }
            }
            if x_seen {
                Some(k)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Evaluate a numeric leaf to a non-negative integer ≤ 20 (degree cap).
fn eval_as_nonneg_integer_expr(expr: &Arc<Expr>) -> Option<u32> {
    let v = match expr.as_ref() {
        Expr::Integer(n) => n.to_i64().map(|v| v as f64)?,
        Expr::Float(f) => *f,
        Expr::Rational(r) => r.to_f64(),
        _ => return None,
    };
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Function, Variable};
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
