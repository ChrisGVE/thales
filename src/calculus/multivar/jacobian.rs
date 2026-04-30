//! Jacobian matrix for vector-valued fields.

use std::sync::Arc;

use crate::numeric::{differentiation::diff_arc, Expr, SymbolId};

// ── Jacobian matrix ───────────────────────────────────────────────────────────

/// Compute the Jacobian matrix J where J[i][j] = ∂fᵢ/∂xⱼ.
///
/// Given a vector-valued function **f** = (f₁, …, fₘ) of variables
/// (x₁, …, xₙ) the Jacobian is the m × n matrix of partial derivatives.
/// Row `i` is the gradient of component `exprs[i]` with respect to all
/// variables in `vars`.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use thales::numeric::{Expr, SymbolId, normalize};
/// use thales::calculus::multivar::jacobian;
///
/// // f = (x*y, x + y)  →  J = [[y, x], [1, 1]]
/// let x_id = SymbolId::intern("jac_x");
/// let y_id = SymbolId::intern("jac_y");
/// let x = Expr::symbol("jac_x");
/// let y = Expr::symbol("jac_y");
/// let f0 = normalize::mul(x.clone(), y.clone()); // x*y
/// let f1 = normalize::add(x.clone(), y.clone()); // x + y
/// let j = jacobian(&[f0, f1], &[x_id, y_id]);
/// assert_eq!(j.len(), 2);        // 2 rows (components)
/// assert_eq!(j[0].len(), 2);     // 2 cols (variables)
/// // ∂(x*y)/∂x = y
/// assert!(!j[0][0].is_zero());
/// // ∂(x+y)/∂x = 1
/// assert!(j[1][0].is_one());
/// ```
pub fn jacobian(exprs: &[Arc<Expr>], vars: &[SymbolId]) -> Vec<Vec<Arc<Expr>>> {
    exprs
        .iter()
        .map(|f| vars.iter().map(|v| diff_arc(f, *v)).collect())
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{normalize, Expr, SymbolId};

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }
    fn id(name: &str) -> SymbolId {
        SymbolId::intern(name)
    }

    #[test]
    fn test_jacobian_shape() {
        // f = (x*y, x+y): 2×2 Jacobian
        let xi = id("jac_sh_x");
        let yi = id("jac_sh_y");
        let x = sym("jac_sh_x");
        let y = sym("jac_sh_y");
        let f0 = normalize::mul(x.clone(), y.clone());
        let f1 = normalize::add(x.clone(), y.clone());
        let j = jacobian(&[f0, f1], &[xi, yi]);
        assert_eq!(j.len(), 2);
        assert_eq!(j[0].len(), 2);
        assert_eq!(j[1].len(), 2);
    }

    #[test]
    fn test_jacobian_values() {
        // f = (x², xy)  →  J = [[2x, 0], [y, x]]
        let xi = id("jac_v_x");
        let yi = id("jac_v_y");
        let x = sym("jac_v_x");
        let y = sym("jac_v_y");
        let f0 = normalize::pow(x.clone(), Expr::int(2)); // x²
        let f1 = normalize::mul(x.clone(), y.clone()); // x*y
        let j = jacobian(&[f0, f1], &[xi, yi]);
        // ∂(x²)/∂y = 0
        assert!(j[0][1].is_zero(), "∂(x²)/∂y should be 0");
        // ∂(x*y)/∂x = y  (matches sym("jac_v_y"))
        assert_eq!(*j[1][0], *y, "∂(xy)/∂x should be y");
        // ∂(x*y)/∂y = x
        assert_eq!(*j[1][1], *x, "∂(xy)/∂y should be x");
    }

    #[test]
    fn test_jacobian_constants_all_zero() {
        // f = (1, 2)  →  J = [[0, 0], [0, 0]]
        let xi = id("jac_c_x");
        let yi = id("jac_c_y");
        let j = jacobian(&[Expr::int(1), Expr::int(2)], &[xi, yi]);
        assert!(j.iter().flatten().all(|d| d.is_zero()));
    }
}
