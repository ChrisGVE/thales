//! Taylor and Maclaurin series expansion functions.

use crate::ast::{Expression, Variable};

use super::known_series::try_known_series;
use super::{
    compute_nth_derivative, evaluate_at, factorial, RemainderTerm, Series, SeriesResult, SeriesTerm,
};

/// Compute the Taylor series of an expression around a center point.
///
/// # Arguments
/// * `expr` - The expression to expand
/// * `var` - The variable of expansion
/// * `center` - The center point of expansion
/// * `order` - The number of terms to compute (0 through order)
///
/// # Returns
/// A `Series` containing the Taylor expansion terms and remainder.
pub fn taylor(
    expr: &Expression,
    var: &Variable,
    center: &Expression,
    order: u32,
) -> SeriesResult<Series> {
    // First, check if we can use a known series
    if let Some(series) = try_known_series(expr, var, center, order) {
        return Ok(series);
    }

    let mut series = Series::new(var.clone(), center.clone(), order);

    for n in 0..=order {
        // Compute nth derivative
        let nth_deriv = compute_nth_derivative(expr, var, n)?;

        // Evaluate at center
        let deriv_at_center = evaluate_at(&nth_deriv, var, center)?;

        // Compute coefficient: f^(n)(center) / n!
        let n_fact = factorial(n) as i64;
        let coefficient = if n_fact == 1 {
            deriv_at_center
        } else {
            Expression::Binary(
                crate::ast::BinaryOp::Div,
                Box::new(deriv_at_center),
                Box::new(Expression::Integer(n_fact)),
            )
            .simplify()
        };

        // Add term if non-zero
        let term = SeriesTerm::new(coefficient, n);
        series.add_term(term);
    }

    // Set remainder
    series.set_remainder(RemainderTerm::BigO { order: order + 1 });

    Ok(series)
}

/// Compute the Maclaurin series (Taylor series centered at 0).
pub fn maclaurin(expr: &Expression, var: &Variable, order: u32) -> SeriesResult<Series> {
    taylor(expr, var, &Expression::Integer(0), order)
}
