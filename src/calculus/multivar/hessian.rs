//! Hessian matrix of second-order partial derivatives.

use std::sync::Arc;

use crate::numeric::{differentiation::diff_arc, Expr, SymbolId};

// ── Hessian matrix ────────────────────────────────────────────────────────────

/// Compute the Hessian matrix H where H[i][j] = ∂²f/∂xᵢ∂xⱼ.
///
/// The Hessian is an n × n symmetric matrix of second-order partial
/// derivatives of a scalar field f(x₁, …, xₙ).  Mixed partials are
/// computed by differentiating the i-th partial derivative with respect
/// to xⱼ.  By Clairaut's theorem the matrix is symmetric when f has
/// continuous second partials (no check is performed; the symbolic result
/// is always consistent).
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use thales::numeric::{Expr, SymbolId, normalize};
/// use thales::calculus::multivar::hessian;
///
/// // f = x² + x*y + y²  →  H = [[2, 1], [1, 2]]
/// let x_id = SymbolId::intern("hes_x");
/// let y_id = SymbolId::intern("hes_y");
/// let x = Expr::symbol("hes_x");
/// let y = Expr::symbol("hes_y");
/// let f = normalize::add(
///     normalize::add(
///         normalize::pow(x.clone(), Expr::int(2)),
///         normalize::mul(x.clone(), y.clone()),
///     ),
///     normalize::pow(y.clone(), Expr::int(2)),
/// );
/// let h = hessian(&f, &[x_id, y_id]);
/// assert_eq!(h.len(), 2);
/// assert_eq!(h[0].len(), 2);
/// // H[0][0] = ∂²f/∂x² = 2
/// assert_eq!(*h[0][0], Expr::Integer(thales::numeric::SmallInt::from(2i64)));
/// // H[0][1] = H[1][0] = ∂²f/∂x∂y = 1
/// assert!(h[0][1].is_one());
/// assert!(h[1][0].is_one());
///
/// // Hessian of a constant is all zeros
/// let hc = hessian(&Expr::int(7), &[x_id, y_id]);
/// assert!(hc.iter().flatten().all(|d| d.is_zero()));
/// ```
pub fn hessian(expr: &Arc<Expr>, vars: &[SymbolId]) -> Vec<Vec<Arc<Expr>>> {
    // Compute first partials once, then differentiate each again.
    let first: Vec<Arc<Expr>> = vars.iter().map(|v| diff_arc(expr, *v)).collect();
    first
        .iter()
        .map(|fi| vars.iter().map(|v| diff_arc(fi, *v)).collect())
        .collect()
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

    #[test]
    fn test_hessian_quadratic() {
        // f = x² + x*y + y²  →  H = [[2, 1], [1, 2]]
        let xi = id("hes_q_x");
        let yi = id("hes_q_y");
        let x = sym("hes_q_x");
        let y = sym("hes_q_y");
        let f = normalize::add(
            normalize::add(
                normalize::pow(x.clone(), Expr::int(2)),
                normalize::mul(x.clone(), y.clone()),
            ),
            normalize::pow(y.clone(), Expr::int(2)),
        );
        let h = hessian(&f, &[xi, yi]);
        assert_eq!(h.len(), 2);
        assert_eq!(h[0].len(), 2);
        assert_eq!(
            *h[0][0],
            Expr::Integer(SmallInt::from(2i64)),
            "H[0][0] should be 2"
        );
        assert!(h[0][1].is_one(), "H[0][1] should be 1");
        assert!(h[1][0].is_one(), "H[1][0] should be 1");
        assert_eq!(
            *h[1][1],
            Expr::Integer(SmallInt::from(2i64)),
            "H[1][1] should be 2"
        );
    }

    #[test]
    fn test_hessian_constant_all_zero() {
        // H(7) = [[0, 0], [0, 0]]
        let xi = id("hes_c_x");
        let yi = id("hes_c_y");
        let h = hessian(&Expr::int(7), &[xi, yi]);
        assert!(h.iter().flatten().all(|d| d.is_zero()));
    }

    #[test]
    fn test_hessian_symmetry() {
        // f = sin(x) * exp(y)  →  H[0][1] == H[1][0]  (symbolic equality)
        let xi = id("hes_sym_x");
        let yi = id("hes_sym_y");
        let x = sym("hes_sym_x");
        let y = sym("hes_sym_y");
        let f = normalize::mul(
            Expr::func(FuncId::Sin, vec![x.clone()]),
            Expr::func(FuncId::Exp, vec![y.clone()]),
        );
        let h = hessian(&f, &[xi, yi]);
        // Mixed partials should produce the same normalized form.
        assert_eq!(
            format!("{}", h[0][1]),
            format!("{}", h[1][0]),
            "H[0][1] and H[1][0] should be equal by Clairaut"
        );
    }
}
