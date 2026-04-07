//! Known series expansions for common functions.
//!
//! Provides pre-computed Maclaurin series for standard mathematical functions.

use crate::ast::{BinaryOp, Expression, Function, Variable};

use super::{factorial, RemainderTerm, Series, SeriesTerm};

/// Maclaurin series for e^x: sum(x^n / n!) for n = 0 to order.
pub fn exp_series(var: &Variable, order: u32) -> Series {
    let mut series = Series::new(var.clone(), Expression::Integer(0), order);

    for n in 0..=order {
        let coeff = if n == 0 {
            Expression::Integer(1)
        } else {
            let n_fact = factorial(n);
            Expression::Rational(num_rational::Ratio::new(1, n_fact as i64))
        };
        series.add_term(SeriesTerm::new(coeff, n));
    }

    series.set_remainder(RemainderTerm::BigO { order: order + 1 });
    series
}

/// Maclaurin series for sin(x): x - x³/3! + x⁵/5! - ...
pub fn sin_series(var: &Variable, order: u32) -> Series {
    let mut series = Series::new(var.clone(), Expression::Integer(0), order);

    let mut n = 0u32;
    while 2 * n + 1 <= order {
        let power = 2 * n + 1;
        let sign = if n % 2 == 0 { 1i64 } else { -1i64 };
        let fact = factorial(power) as i64;
        let coeff = Expression::Rational(num_rational::Ratio::new(sign, fact));
        series.add_term(SeriesTerm::new(coeff, power));
        n += 1;
    }

    series.set_remainder(RemainderTerm::BigO { order: order + 1 });
    series
}

/// Maclaurin series for cos(x): 1 - x²/2! + x⁴/4! - ...
pub fn cos_series(var: &Variable, order: u32) -> Series {
    let mut series = Series::new(var.clone(), Expression::Integer(0), order);

    let mut n = 0u32;
    while 2 * n <= order {
        let power = 2 * n;
        let sign = if n % 2 == 0 { 1i64 } else { -1i64 };
        let fact = if power == 0 {
            1
        } else {
            factorial(power) as i64
        };
        let coeff = Expression::Rational(num_rational::Ratio::new(sign, fact));
        series.add_term(SeriesTerm::new(coeff, power));
        n += 1;
    }

    series.set_remainder(RemainderTerm::BigO { order: order + 1 });
    series
}

/// Maclaurin series for ln(1+x): x - x²/2 + x³/3 - x⁴/4 + ...
pub fn ln_1_plus_x_series(var: &Variable, order: u32) -> Series {
    let mut series = Series::new(var.clone(), Expression::Integer(0), order);

    for n in 1..=order {
        let sign = if n % 2 == 1 { 1i64 } else { -1i64 };
        let coeff = Expression::Rational(num_rational::Ratio::new(sign, n as i64));
        series.add_term(SeriesTerm::new(coeff, n));
    }

    series.set_remainder(RemainderTerm::BigO { order: order + 1 });
    series
}

/// Maclaurin series for arctan(x): x - x³/3 + x⁵/5 - x⁷/7 + ...
pub fn arctan_series(var: &Variable, order: u32) -> Series {
    let mut series = Series::new(var.clone(), Expression::Integer(0), order);

    let mut n = 0u32;
    while 2 * n + 1 <= order {
        let power = 2 * n + 1;
        let sign = if n % 2 == 0 { 1i64 } else { -1i64 };
        let coeff = Expression::Rational(num_rational::Ratio::new(sign, power as i64));
        series.add_term(SeriesTerm::new(coeff, power));
        n += 1;
    }

    series.set_remainder(RemainderTerm::BigO { order: order + 1 });
    series
}

/// Binomial series for (1+x)^a: sum(C(a,n) * x^n).
/// Only works for symbolic or numeric exponents.
pub fn binomial_series(
    exponent: &Expression,
    var: &Variable,
    order: u32,
) -> super::SeriesResult<Series> {
    use std::collections::HashMap;

    let mut series = Series::new(var.clone(), Expression::Integer(0), order);

    // For now, only handle numeric exponents
    let a = match exponent.evaluate(&HashMap::new()) {
        Some(val) => val,
        None => {
            return Err(super::SeriesError::CannotExpand(
                "Binomial series requires numeric exponent".to_string(),
            ))
        }
    };

    // Compute binomial coefficients C(a, n) = a*(a-1)*...*(a-n+1) / n!
    let mut binom_coeff = 1.0;
    for n in 0..=order {
        if n > 0 {
            binom_coeff *= (a - (n as f64 - 1.0)) / (n as f64);
        }

        if binom_coeff.abs() > 1e-15 {
            let coeff = if (binom_coeff - binom_coeff.round()).abs() < 1e-10 {
                Expression::Integer(binom_coeff.round() as i64)
            } else {
                Expression::Float(binom_coeff)
            };
            series.add_term(SeriesTerm::new(coeff, n));
        }

        // For positive integer a, series terminates
        if a.fract() == 0.0 && a >= 0.0 && n as f64 >= a {
            break;
        }
    }

    // Only add remainder if series doesn't terminate
    if a.fract() != 0.0 || a < 0.0 {
        series.set_remainder(RemainderTerm::BigO { order: order + 1 });
    }

    Ok(series)
}

/// Try to match the expression to a known series for efficiency.
pub(crate) fn try_known_series(
    expr: &Expression,
    var: &Variable,
    center: &Expression,
    order: u32,
) -> Option<Series> {
    // Only handle Maclaurin (center = 0) for built-in series
    if !matches!(center, Expression::Integer(0)) {
        return None;
    }

    match expr {
        Expression::Function(Function::Exp, args) if args.len() == 1 => {
            if matches!(&args[0], Expression::Variable(v) if v.name == var.name) {
                return Some(exp_series(var, order));
            }
        }
        Expression::Function(Function::Sin, args) if args.len() == 1 => {
            if matches!(&args[0], Expression::Variable(v) if v.name == var.name) {
                return Some(sin_series(var, order));
            }
        }
        Expression::Function(Function::Cos, args) if args.len() == 1 => {
            if matches!(&args[0], Expression::Variable(v) if v.name == var.name) {
                return Some(cos_series(var, order));
            }
        }
        Expression::Function(Function::Ln, args) if args.len() == 1 => {
            // Check for ln(1 + x)
            if let Expression::Binary(BinaryOp::Add, left, right) = &args[0] {
                if matches!(**left, Expression::Integer(1))
                    && matches!(**right, Expression::Variable(ref v) if v.name == var.name)
                {
                    return Some(ln_1_plus_x_series(var, order));
                }
            }
        }
        Expression::Function(Function::Atan, args) if args.len() == 1 => {
            if matches!(&args[0], Expression::Variable(v) if v.name == var.name) {
                return Some(arctan_series(var, order));
            }
        }
        _ => {}
    }

    None
}
