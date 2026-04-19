//! Taylor and Laurent series expansion for symbolic expressions.
//!
//! # Overview
//!
//! This module provides:
//!
//! - [`TaylorSeries`] — truncated Taylor series around a center point.
//! - [`LaurentSeries`] — truncated Laurent series with a (possibly negative)
//!   leading power, suitable for expansions at poles.
//! - [`taylor`] — compute a Taylor series via repeated differentiation.
//! - [`convergence_radius`] — estimate the convergence radius via the ratio
//!   test on computed numerical coefficients.
//! - Series arithmetic: [`add`], [`mul`], [`truncate`].
//! - Known series constructors: [`sin_series`], [`cos_series`], [`exp_series`],
//!   [`ln_series`], [`atan_series`].
//!
//! # Example
//!
//! ```rust
//! use thales::numeric::{Expr, SymbolId};
//! use thales::numeric::series::{taylor, exp_series};
//!
//! let x_id = SymbolId::intern("series_ex");
//! let x    = Expr::symbol("series_ex");
//!
//! // exp(x) via repeated differentiation — should equal the known series
//! let expr = Expr::func(thales::numeric::FuncId::Exp, vec![x.clone()]);
//! let ts   = taylor(&expr, x_id, &Expr::int(0), 4);
//! assert_eq!(ts.order, 4);
//! assert_eq!(ts.coefficients.len(), 5); // a_0 … a_4
//! ```

use std::sync::Arc;

use super::expr::Expr;
use super::SymbolId;

pub mod arithmetic;
pub mod asymptotic;
pub mod composition;
pub mod convergence;
pub mod known;
pub mod singularity;
pub mod taylor;

pub use arithmetic::{add, mul, truncate};
pub use asymptotic::{
    asymptotic, limit_via_asymptotic, AsymptoticDirection, AsymptoticSeries, AsymptoticTerm, BigO,
};
pub use composition::{compose, revert};
pub use convergence::convergence_radius;
pub use known::{atan_series, cos_series, exp_series, ln_series, sin_series};
pub use singularity::{
    classify_singularity, find_singularities, pole_order, residue, Singularity, SingularityType,
};
pub use taylor::taylor;

// ── TaylorSeries ─────────────────────────────────────────────────────────────

/// A truncated Taylor series `Σ_{n=0}^{order} a_n · (x - center)^n`.
///
/// Coefficients are stored as `a_n` (the pure coefficient, not `a_n*(x-c)^n`).
/// The `n`-th coefficient equals `f^(n)(center) / n!`.
///
/// # Fields
///
/// * `center` — expansion point (e.g. `0` for Maclaurin series).
/// * `var` — the expansion variable.
/// * `coefficients` — `[a_0, a_1, …, a_order]`, length `order + 1`.
/// * `order` — truncation order (highest power retained).
#[derive(Clone, Debug)]
pub struct TaylorSeries {
    /// Expansion point.
    pub center: Arc<Expr>,
    /// Expansion variable.
    pub var: SymbolId,
    /// Coefficients `a_0, a_1, …, a_{order}`.
    pub coefficients: Vec<Arc<Expr>>,
    /// Truncation order (= `coefficients.len() - 1`).
    pub order: usize,
}

impl TaylorSeries {
    /// Create a `TaylorSeries` directly from pre-computed coefficients.
    ///
    /// # Panics
    ///
    /// Panics if `coefficients` is empty.
    pub fn from_coefficients(
        center: Arc<Expr>,
        var: SymbolId,
        coefficients: Vec<Arc<Expr>>,
    ) -> Self {
        assert!(
            !coefficients.is_empty(),
            "TaylorSeries must have ≥1 coefficient"
        );
        let order = coefficients.len() - 1;
        TaylorSeries {
            center,
            var,
            coefficients,
            order,
        }
    }

    /// Return coefficient `a_n`, or `0` if `n > order`.
    pub fn coeff(&self, n: usize) -> Arc<Expr> {
        self.coefficients
            .get(n)
            .cloned()
            .unwrap_or_else(|| Expr::int(0))
    }

    /// Reassemble the series as a single normalized `Arc<Expr>`:
    /// `Σ a_n · (x − center)^n`.
    #[must_use]
    pub fn to_expr(&self) -> Arc<Expr> {
        let var_expr: Arc<Expr> = Arc::new(Expr::Symbol(self.var));
        let shift = if self.center.is_zero() {
            var_expr
        } else {
            super::normalize::sub(var_expr, self.center.clone())
        };
        let mut acc: Arc<Expr> = Expr::int(0);
        for (n, c) in self.coefficients.iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            let power = super::normalize::pow(shift.clone(), Expr::int(n as i64));
            let term = super::normalize::mul(c.clone(), power);
            acc = super::normalize::add(acc, term);
        }
        acc
    }
}

// ── LaurentSeries ─────────────────────────────────────────────────────────────

/// A truncated Laurent series
/// `Σ_{n=leading_power}^{leading_power+order} a_n · (x - center)^n`.
///
/// `coefficients[i]` is the coefficient of `(x - center)^(leading_power + i)`.
/// `leading_power` may be negative (pole or essential singularity terms).
///
/// # Fields
///
/// * `center` — expansion point.
/// * `var` — expansion variable.
/// * `coefficients` — coefficients starting from the leading power term.
/// * `leading_power` — the lowest power present (may be negative).
/// * `order` — number of terms minus one (`coefficients.len() - 1`).
#[derive(Clone, Debug)]
pub struct LaurentSeries {
    /// Expansion point.
    pub center: Arc<Expr>,
    /// Expansion variable.
    pub var: SymbolId,
    /// Coefficients `[a_{leading}, a_{leading+1}, …]`.
    pub coefficients: Vec<Arc<Expr>>,
    /// Lowest power of `(x - center)` represented.
    pub leading_power: i32,
    /// `coefficients.len() - 1`.
    pub order: usize,
}

impl LaurentSeries {
    /// Create a `LaurentSeries` from coefficients and the leading power.
    ///
    /// # Panics
    ///
    /// Panics if `coefficients` is empty.
    pub fn from_coefficients(
        center: Arc<Expr>,
        var: SymbolId,
        coefficients: Vec<Arc<Expr>>,
        leading_power: i32,
    ) -> Self {
        assert!(
            !coefficients.is_empty(),
            "LaurentSeries must have ≥1 coefficient"
        );
        let order = coefficients.len() - 1;
        LaurentSeries {
            center,
            var,
            coefficients,
            leading_power,
            order,
        }
    }

    /// Return the coefficient of `(x - center)^power`, or `0` if out of range.
    pub fn coeff(&self, power: i32) -> Arc<Expr> {
        let idx = power - self.leading_power;
        if idx < 0 {
            return Expr::int(0);
        }
        self.coefficients
            .get(idx as usize)
            .cloned()
            .unwrap_or_else(|| Expr::int(0))
    }

    /// Convert to a [`TaylorSeries`] when `leading_power >= 0`.
    ///
    /// Returns `None` if the series has negative powers.
    pub fn to_taylor(&self) -> Option<TaylorSeries> {
        if self.leading_power < 0 {
            return None;
        }
        // Prepend zero coefficients for powers 0 .. leading_power
        let mut coeffs: Vec<Arc<Expr>> = (0..self.leading_power).map(|_| Expr::int(0)).collect();
        coeffs.extend(self.coefficients.iter().cloned());
        Some(TaylorSeries::from_coefficients(
            self.center.clone(),
            self.var,
            coeffs,
        ))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn zero() -> Arc<Expr> {
        Expr::int(0)
    }

    fn one() -> Arc<Expr> {
        Expr::int(1)
    }

    #[test]
    fn test_taylor_series_from_coefficients() {
        let var = SymbolId::intern("ts_mod_x");
        let coeffs = vec![one(), one(), one()];
        let ts = TaylorSeries::from_coefficients(zero(), var, coeffs);
        assert_eq!(ts.order, 2);
        assert_eq!(ts.coefficients.len(), 3);
    }

    #[test]
    fn test_taylor_coeff_in_range() {
        let var = SymbolId::intern("ts_mod_coeff");
        let coeffs = vec![Expr::int(3), Expr::int(7)];
        let ts = TaylorSeries::from_coefficients(zero(), var, coeffs);
        assert_eq!(
            *ts.coeff(0),
            Expr::Integer(super::super::SmallInt::from(3i64))
        );
        assert_eq!(
            *ts.coeff(1),
            Expr::Integer(super::super::SmallInt::from(7i64))
        );
    }

    #[test]
    fn test_taylor_coeff_out_of_range() {
        let var = SymbolId::intern("ts_mod_oor");
        let ts = TaylorSeries::from_coefficients(zero(), var, vec![one()]);
        assert!(ts.coeff(5).is_zero());
    }

    #[test]
    fn test_laurent_from_coefficients() {
        let var = SymbolId::intern("ls_mod_x");
        let coeffs = vec![one(), zero(), one()];
        let ls = LaurentSeries::from_coefficients(zero(), var, coeffs, -1);
        assert_eq!(ls.leading_power, -1);
        assert_eq!(ls.order, 2);
    }

    #[test]
    fn test_laurent_coeff_by_power() {
        let var = SymbolId::intern("ls_mod_cp");
        let coeffs = vec![Expr::int(5), Expr::int(6)];
        let ls = LaurentSeries::from_coefficients(zero(), var, coeffs, -1);
        assert_eq!(
            *ls.coeff(-1),
            Expr::Integer(super::super::SmallInt::from(5i64))
        );
        assert_eq!(
            *ls.coeff(0),
            Expr::Integer(super::super::SmallInt::from(6i64))
        );
        assert!(ls.coeff(1).is_zero());
        assert!(ls.coeff(-2).is_zero());
    }

    #[test]
    fn test_laurent_to_taylor_positive_leading() {
        let var = SymbolId::intern("ls_to_ts");
        let coeffs = vec![Expr::int(2), Expr::int(3)];
        let ls = LaurentSeries::from_coefficients(zero(), var, coeffs, 1);
        let ts = ls.to_taylor().expect("should convert");
        // leading_power=1, so a_0=0, a_1=2, a_2=3
        assert_eq!(ts.order, 2);
        assert!(ts.coeff(0).is_zero());
    }

    #[test]
    fn test_laurent_to_taylor_negative_leading() {
        let var = SymbolId::intern("ls_to_ts_neg");
        let coeffs = vec![one()];
        let ls = LaurentSeries::from_coefficients(zero(), var, coeffs, -1);
        assert!(ls.to_taylor().is_none());
    }
}
