//! Limit computation for symbolic expressions.
//!
//! Implements limit evaluation using three strategies in order:
//!
//! 1. **Direct substitution** — substitute the point value and check for a
//!    well-defined numeric result.
//! 2. **L'Hôpital's rule** — for 0/0 or ∞/∞ indeterminate forms on `Div`
//!    expressions, differentiate numerator and denominator repeatedly (up to
//!    [`LHOPITAL_MAX_ITER`] times).
//! 3. **Taylor series** — expand the expression around the limit point, find
//!    the leading non-zero coefficient, and return its value.
//!
//! Limits at ±∞ are handled via the substitution `x = 1/t`, `t → 0⁺`.

use std::sync::Arc;

use super::expr::Expr;
use super::SymbolId;

mod classify;
mod evaluation;
mod helpers;
mod lhopital;
mod series;

// ── Public types ──────────────────────────────────────────────────────────────

/// The point at which a limit is evaluated.
#[derive(Clone, Debug)]
pub enum LimitPoint {
    /// Finite point: `x → a`.
    Value(Arc<Expr>),
    /// `x → +∞`.
    PosInfinity,
    /// `x → -∞`.
    NegInfinity,
    /// Left-hand limit: `x → a⁻`.
    FromLeft(Arc<Expr>),
    /// Right-hand limit: `x → a⁺`.
    FromRight(Arc<Expr>),
}

/// The outcome of a limit computation.
#[derive(Clone, Debug, PartialEq)]
pub enum LimitResult {
    /// The limit exists and equals a finite expression.
    Value(Arc<Expr>),
    /// The limit is +∞.
    PosInfinity,
    /// The limit is -∞.
    NegInfinity,
    /// The limit does not exist (e.g. left ≠ right, oscillates).
    DoesNotExist,
    /// The algorithm could not determine the limit.
    Indeterminate,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute `lim_{x → point} expr`.
///
/// # Arguments
///
/// * `expr`  — expression to take the limit of.
/// * `var`   — the variable that approaches the limit point.
/// * `point` — the value the variable approaches.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{Expr, FuncId, SymbolId};
/// use thales::numeric::limits::{limit, LimitPoint, LimitResult};
///
/// // lim_{x→0} sin(x)/x = 1
/// let x_id = SymbolId::intern("lim_sinc_x");
/// let x = Expr::symbol("lim_sinc_x");
/// let sin_x = Expr::func(FuncId::Sin, vec![x.clone()]);
/// let expr = thales::numeric::normalize::div(sin_x, x);
/// let result = limit(&expr, x_id, &LimitPoint::Value(Expr::int(0)));
/// assert_eq!(result, LimitResult::Value(Expr::int(1)));
/// ```
pub fn limit(expr: &Arc<Expr>, var: SymbolId, point: &LimitPoint) -> LimitResult {
    match point {
        LimitPoint::Value(a) | LimitPoint::FromLeft(a) | LimitPoint::FromRight(a) => {
            evaluation::limit_at_finite(expr, var, a, point)
        }
        LimitPoint::PosInfinity => evaluation::limit_at_infinity(expr, var, false),
        LimitPoint::NegInfinity => evaluation::limit_at_infinity(expr, var, true),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{normalize, Expr, FuncId, SymbolId};

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }

    fn lim(expr: &Arc<Expr>, name: &str, point: LimitPoint) -> LimitResult {
        let id = SymbolId::intern(name);
        limit(expr, id, &point)
    }

    // ── Direct substitution ───────────────────────────────────────────────

    #[test]
    fn test_limit_polynomial_finite() {
        // lim_{x→3} x^2 = 9
        let x = sym("lim_poly_x");
        let expr = normalize::pow(x, Expr::int(2));
        let result = lim(&expr, "lim_poly_x", LimitPoint::Value(Expr::int(3)));
        assert_eq!(result, LimitResult::Value(Expr::int(9)));
    }

    #[test]
    fn test_limit_constant() {
        // lim_{x→5} 7 = 7
        let expr = Expr::int(7);
        let result = lim(&expr, "lim_const_x", LimitPoint::Value(Expr::int(5)));
        assert_eq!(result, LimitResult::Value(Expr::int(7)));
    }

    // ── L'Hôpital 0/0 ────────────────────────────────────────────────────

    #[test]
    fn test_limit_lhopital_polynomial_ratio() {
        // lim_{x→1} (x^2 - 1) / (x - 1) = 2
        let x = sym("lim_lh_x");
        let x2_minus_1 = normalize::sub(normalize::pow(x.clone(), Expr::int(2)), Expr::int(1));
        let x_minus_1 = normalize::sub(x, Expr::int(1));
        let expr = normalize::div(x2_minus_1, x_minus_1);
        let result = lim(&expr, "lim_lh_x", LimitPoint::Value(Expr::int(1)));
        assert_eq!(result, LimitResult::Value(Expr::int(2)));
    }

    // ── Sinc limit via L'Hôpital or series ───────────────────────────────

    #[test]
    fn test_limit_sinc() {
        // lim_{x→0} sin(x)/x = 1
        let x = sym("lim_sinc_x");
        let sin_x = Expr::func(FuncId::Sin, vec![x.clone()]);
        let expr = normalize::div(sin_x, x);
        let result = lim(&expr, "lim_sinc_x", LimitPoint::Value(Expr::int(0)));
        assert_eq!(result, LimitResult::Value(Expr::int(1)));
    }

    // ── One-sided limits ──────────────────────────────────────────────────

    #[test]
    fn test_limit_one_over_x_right() {
        // lim_{x→0⁺} 1/x = +∞
        let x = sym("lim_1x_r");
        let expr = normalize::div(Expr::int(1), x);
        let result = lim(&expr, "lim_1x_r", LimitPoint::FromRight(Expr::int(0)));
        assert_eq!(result, LimitResult::PosInfinity);
    }

    #[test]
    fn test_limit_one_over_x_left() {
        // lim_{x→0⁻} 1/x = -∞
        let x = sym("lim_1x_l");
        let expr = normalize::div(Expr::int(1), x);
        let result = lim(&expr, "lim_1x_l", LimitPoint::FromLeft(Expr::int(0)));
        assert_eq!(result, LimitResult::NegInfinity);
    }

    #[test]
    fn test_limit_one_over_x_two_sided() {
        // lim_{x→0} 1/x does not exist (left ≠ right)
        let x = sym("lim_1x_ts");
        let expr = normalize::div(Expr::int(1), x);
        let result = lim(&expr, "lim_1x_ts", LimitPoint::Value(Expr::int(0)));
        assert_eq!(result, LimitResult::DoesNotExist);
    }

    // ── Infinity limits ───────────────────────────────────────────────────

    #[test]
    fn test_limit_exp_neg_inf() {
        // lim_{x→-∞} exp(x) = 0
        let x = sym("lim_exp_ni");
        let expr = Expr::func(FuncId::Exp, vec![x]);
        let result = lim(&expr, "lim_exp_ni", LimitPoint::NegInfinity);
        assert_eq!(result, LimitResult::Value(Expr::int(0)));
    }

    #[test]
    fn test_limit_one_over_x_at_pos_inf() {
        // lim_{x→+∞} 1/x = 0
        let x = sym("lim_1x_inf");
        let expr = normalize::div(Expr::int(1), x);
        let result = lim(&expr, "lim_1x_inf", LimitPoint::PosInfinity);
        assert_eq!(result, LimitResult::Value(Expr::int(0)));
    }
}
