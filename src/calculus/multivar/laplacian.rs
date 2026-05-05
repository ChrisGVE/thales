//! Laplacian of a scalar field.

use std::sync::Arc;

use crate::numeric::{normalize, Expr, SymbolId};

use super::gradient::partial;

// ── Laplacian ─────────────────────────────────────────────────────────────────

/// Compute the Laplacian ∇²f = Σᵢ ∂²f/∂xᵢ².
///
/// Applies the second partial derivative with respect to each variable in
/// `vars` and sums the results.  Each step goes through the normalising
/// [`partial`] helper (which delegates to
/// [`diff_arc`][crate::numeric::differentiation::diff_arc]), so
/// constant-folding and identity removal are applied automatically.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use thales::numeric::{Expr, SymbolId, normalize};
/// use thales::calculus::multivar::laplacian;
///
/// // f = x² + y²  →  ∇²f = 2 + 2 = 4
/// let xi = SymbolId::intern("lap_ex_x");
/// let yi = SymbolId::intern("lap_ex_y");
/// let x = Expr::symbol("lap_ex_x");
/// let y = Expr::symbol("lap_ex_y");
/// let f = normalize::add(
///     normalize::pow(x.clone(), Expr::int(2)),
///     normalize::pow(y.clone(), Expr::int(2)),
/// );
/// let result = laplacian(&f, &[xi, yi]);
/// assert_eq!(*result, Expr::Integer(thales::numeric::SmallInt::from(4i64)));
/// ```
pub fn laplacian(expr: &Arc<Expr>, vars: &[SymbolId]) -> Arc<Expr> {
    let mut acc = Expr::int(0);
    for v in vars {
        let d1 = partial(expr, v);
        let d2 = partial(&d1, v);
        acc = normalize::add(acc, d2);
    }
    acc
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{normalize, Expr, SmallInt, SymbolId};

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }
    fn id(name: &str) -> SymbolId {
        SymbolId::intern(name)
    }

    #[test]
    fn fast_laplacian_quadratic() {
        // f = x² + y²  →  ∇²f = 2 + 2 = 4
        let xi = id("lap_q_x");
        let yi = id("lap_q_y");
        let x = sym("lap_q_x");
        let y = sym("lap_q_y");
        let f = normalize::add(
            normalize::pow(x.clone(), Expr::int(2)),
            normalize::pow(y.clone(), Expr::int(2)),
        );
        let result = laplacian(&f, &[xi, yi]);
        assert_eq!(
            *result,
            Expr::Integer(SmallInt::from(4i64)),
            "∇²(x²+y²) should be 4"
        );
    }

    #[test]
    fn fast_laplacian_cubic() {
        // f = x³ + y³  →  ∇²f = 6x + 6y  (non-zero, not constant)
        let xi = id("lap_c_x");
        let yi = id("lap_c_y");
        let x = sym("lap_c_x");
        let y = sym("lap_c_y");
        let f = normalize::add(
            normalize::pow(x.clone(), Expr::int(3)),
            normalize::pow(y.clone(), Expr::int(3)),
        );
        let result = laplacian(&f, &[xi, yi]);
        // 6x + 6y ≠ 0 and not a plain integer constant
        assert!(!result.is_zero(), "∇²(x³+y³) should not be zero");
        assert!(
            !matches!(*result, Expr::Integer(_)),
            "∇²(x³+y³) should not be a constant"
        );
    }

    #[test]
    fn fast_laplacian_constant() {
        // f = 42  →  ∇²f = 0
        let xi = id("lap_k_x");
        let yi = id("lap_k_y");
        let result = laplacian(&Expr::int(42), &[xi, yi]);
        assert!(result.is_zero(), "∇²(42) should be 0");
    }

    #[test]
    fn fast_laplacian_linear() {
        // f = 3x + 2y  →  ∇²f = 0 (all second partials zero)
        let xi = id("lap_l_x");
        let yi = id("lap_l_y");
        let x = sym("lap_l_x");
        let y = sym("lap_l_y");
        let f = normalize::add(
            normalize::mul(Expr::int(3), x.clone()),
            normalize::mul(Expr::int(2), y.clone()),
        );
        let result = laplacian(&f, &[xi, yi]);
        assert!(result.is_zero(), "∇²(3x+2y) should be 0");
    }
}
