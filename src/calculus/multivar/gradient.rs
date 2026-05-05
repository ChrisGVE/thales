//! Partial derivative and gradient for scalar fields.

use std::sync::Arc;

use crate::numeric::{differentiation::diff_arc, Expr, SymbolId};

// ── Partial derivative ────────────────────────────────────────────────────────

/// Compute the partial derivative ∂`expr`/∂`var`.
///
/// All other symbols present in the expression are treated as constants,
/// which is exactly the behaviour of [`diff_arc`] for symbols that do not
/// match `var`.  The result is normalized (constant-folded, identities
/// removed) by the underlying smart constructors.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use thales::numeric::{Expr, SymbolId};
/// use thales::calculus::multivar::partial;
///
/// // ∂/∂x (x² + y) = 2x
/// let x_id = SymbolId::intern("prt_x");
/// let y_id = SymbolId::intern("prt_y");
/// let x = Expr::symbol("prt_x");
/// let y = Expr::symbol("prt_y");
/// let expr = thales::numeric::normalize::add(
///     thales::numeric::normalize::pow(x.clone(), Expr::int(2)),
///     y.clone(),
/// );
/// let result = partial(&expr, &x_id);
/// assert!(!result.is_zero()); // 2x ≠ 0
///
/// // ∂/∂y (x²) = 0  — treating x as constant
/// let result_y = partial(&thales::numeric::normalize::pow(x.clone(), Expr::int(2)), &y_id);
/// assert!(result_y.is_zero());
/// ```
pub fn partial(expr: &Arc<Expr>, var: &SymbolId) -> Arc<Expr> {
    diff_arc(expr, *var)
}

// ── Gradient ──────────────────────────────────────────────────────────────────

/// Compute the gradient ∇f = (∂f/∂x₁, …, ∂f/∂xₙ).
///
/// Returns one partial derivative per entry in `vars`, in the same order.
/// For a scalar field f(x₁, …, xₙ) the gradient is the vector of all
/// first-order partial derivatives.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use thales::numeric::{Expr, SymbolId, normalize};
/// use thales::calculus::multivar::gradient;
///
/// // f = x² + y²  →  ∇f = (2x, 2y)
/// let x_id = SymbolId::intern("grad_x");
/// let y_id = SymbolId::intern("grad_y");
/// let x = Expr::symbol("grad_x");
/// let y = Expr::symbol("grad_y");
/// let f = normalize::add(
///     normalize::pow(x.clone(), Expr::int(2)),
///     normalize::pow(y.clone(), Expr::int(2)),
/// );
/// let g = gradient(&f, &[x_id, y_id]);
/// assert_eq!(g.len(), 2);
/// assert!(!g[0].is_zero()); // ∂f/∂x = 2x
/// assert!(!g[1].is_zero()); // ∂f/∂y = 2y
///
/// // Gradient of a constant is all zeros
/// let gc = gradient(&Expr::int(5), &[x_id, y_id]);
/// assert!(gc.iter().all(|d| d.is_zero()));
/// ```
pub fn gradient(expr: &Arc<Expr>, vars: &[SymbolId]) -> Vec<Arc<Expr>> {
    vars.iter().map(|v| diff_arc(expr, *v)).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{normalize, Expr, FuncId, SmallInt, SymbolId};

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }
    fn id(name: &str) -> SymbolId {
        SymbolId::intern(name)
    }

    // ── partial ───────────────────────────────────────────────────────────

    #[test]
    fn test_partial_polynomial() {
        // ∂/∂x (x³ + y) = 3x²
        let xi = id("p_poly_x");
        let yi = id("p_poly_y");
        let x = sym("p_poly_x");
        let y = sym("p_poly_y");
        let f = normalize::add(normalize::pow(x.clone(), Expr::int(3)), y.clone());
        let result = partial(&f, &xi);
        // 3x² ≠ 0
        assert!(!result.is_zero(), "∂/∂x(x³+y) should not be zero");
        // wrt y: ∂/∂y (x³ + y) = 1
        let result_y = partial(&f, &yi);
        assert!(result_y.is_one(), "∂/∂y(x³+y) should be 1, got {result_y}");
    }

    #[test]
    fn test_partial_constant_is_zero() {
        // ∂/∂x (42) = 0
        let xi = id("p_const_x");
        assert!(partial(&Expr::int(42), &xi).is_zero());
    }

    #[test]
    fn test_partial_trig() {
        // ∂/∂x (sin(x) * cos(y)) = cos(x) * cos(y)
        let xi = id("p_trig_x");
        let x = sym("p_trig_x");
        let y = sym("p_trig_y");
        let sin_x = Expr::func(FuncId::Sin, vec![x.clone()]);
        let cos_y = Expr::func(FuncId::Cos, vec![y.clone()]);
        let f = normalize::mul(sin_x, cos_y);
        let result = partial(&f, &xi);
        // cos(x)*cos(y) ≠ 0
        assert!(!result.is_zero());
    }

    // ── gradient ──────────────────────────────────────────────────────────

    #[test]
    fn test_gradient_polynomial() {
        // f = x² + y²  →  ∇f = (2x, 2y)
        let xi = id("gr_x");
        let yi = id("gr_y");
        let x = sym("gr_x");
        let y = sym("gr_y");
        let f = normalize::add(
            normalize::pow(x.clone(), Expr::int(2)),
            normalize::pow(y.clone(), Expr::int(2)),
        );
        let g = gradient(&f, &[xi, yi]);
        assert_eq!(g.len(), 2);
        assert!(!g[0].is_zero(), "∂f/∂x should not be zero");
        assert!(!g[1].is_zero(), "∂f/∂y should not be zero");
    }

    #[test]
    fn test_gradient_constant_all_zero() {
        // ∇(5) = (0, 0)
        let xi = id("gr_cx");
        let yi = id("gr_cy");
        let g = gradient(&Expr::int(5), &[xi, yi]);
        assert!(
            g.iter().all(|d| d.is_zero()),
            "gradient of constant should be zero"
        );
    }

    #[test]
    fn test_gradient_linear() {
        // f = 3x + 2y  →  ∇f = (3, 2)
        let xi = id("gr_lin_x");
        let yi = id("gr_lin_y");
        let x = sym("gr_lin_x");
        let y = sym("gr_lin_y");
        let f = normalize::add(
            normalize::mul(Expr::int(3), x.clone()),
            normalize::mul(Expr::int(2), y.clone()),
        );
        let g = gradient(&f, &[xi, yi]);
        assert_eq!(
            *g[0],
            Expr::Integer(SmallInt::from(3i64)),
            "∂f/∂x should be 3"
        );
        assert_eq!(
            *g[1],
            Expr::Integer(SmallInt::from(2i64)),
            "∂f/∂y should be 2"
        );
    }
}
