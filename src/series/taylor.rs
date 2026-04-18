//! Taylor and Maclaurin series expansion functions.
//!
//! Thin Expression-typed facade over the `Arc<Expr>`-native Taylor engine in
//! [`crate::numeric::series::taylor`]. The legacy [`Series`] shape (list of
//! non-zero [`SeriesTerm`]s with a `BigO` remainder) is preserved at the
//! public boundary.

use crate::ast::{Expression, Variable};
use crate::numeric::compile::{compile, decompile};
use crate::numeric::series::taylor as numeric_taylor;
use crate::numeric::SymbolId;
use num::traits::Zero;

use super::known_series::try_known_series;
use super::{RemainderTerm, Series, SeriesResult, SeriesTerm};

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
    // Prefer closed-form known-series recognition when the integrand matches.
    if let Some(series) = try_known_series(expr, var, center, order) {
        return Ok(series);
    }

    // Delegate coefficient computation to the Arc<Expr>-native engine.
    let var_id = SymbolId::intern(&var.name);
    let expr_arc = compile(expr);
    let center_arc = compile(center);
    let ts = numeric_taylor(&expr_arc, var_id, &center_arc, order as usize);

    let mut series = Series::new(var.clone(), center.clone(), order);
    for n in 0..=order {
        let coeff_arc = ts.coeff(n as usize);
        if coeff_arc.is_zero() {
            continue;
        }
        let coeff_expr = decompile(&coeff_arc).simplify();
        series.add_term(SeriesTerm::new(coeff_expr, n));
    }
    series.set_remainder(RemainderTerm::BigO { order: order + 1 });
    Ok(series)
}

/// Compute the Maclaurin series (Taylor series centered at 0).
pub fn maclaurin(expr: &Expression, var: &Variable, order: u32) -> SeriesResult<Series> {
    taylor(expr, var, &Expression::Integer(0), order)
}
