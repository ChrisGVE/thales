//! Taylor series fallback for limit evaluation.

use std::sync::Arc;

use num::traits::Zero;

use super::super::expr::Expr;
use super::super::SymbolId;
use super::helpers::SERIES_ORDER;
use super::LimitResult;

pub(crate) fn try_series_limit(
    expr: &Arc<Expr>,
    var: SymbolId,
    point: &Arc<Expr>,
) -> Option<LimitResult> {
    use super::super::series::taylor::taylor;

    let ts = taylor(expr, var, point, SERIES_ORDER);

    // Find the leading non-zero coefficient.
    for n in 0..=ts.order {
        let coeff = ts.coeff(n);
        if !coeff.is_zero() {
            if n == 0 {
                // Constant term is the limit.
                return Some(LimitResult::Value(coeff));
            } else {
                // Leading term is (x - center)^n * coeff.
                // As x → center, this → 0 for any finite non-zero coeff.
                // (The limit of x^n * coeff as x→0 is 0 for n≥1.)
                return Some(LimitResult::Value(Expr::int(0)));
            }
        }
    }

    // All coefficients are zero: limit is 0.
    Some(LimitResult::Value(Expr::int(0)))
}
