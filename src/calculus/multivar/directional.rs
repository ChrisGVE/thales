//! Directional derivative of a scalar field.

use std::sync::Arc;

use crate::numeric::{differentiation::diff_arc, normalize, Expr, SymbolId};

// ── Directional derivative ────────────────────────────────────────────────────

/// Compute the directional derivative Dᵥf = ∇f · dir.
///
/// Mathematically: Dᵥf = Σᵢ (∂f/∂xᵢ) · dirᵢ.
///
/// The direction vector `dir` need not be a unit vector; normalization is
/// the caller's responsibility when the unit-vector interpretation is
/// required.  The function computes the dot product of the gradient with
/// the supplied direction components symbolically.
///
/// `vars` and `dir` must have the same length; mismatched lengths are
/// silently truncated to `min(vars.len(), dir.len())`.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use thales::numeric::{Expr, SymbolId, normalize};
/// use thales::calculus::multivar::directional_derivative;
///
/// // f = x² + y,  dir = (1, 0)  →  Dᵥf = 2x
/// let x_id = SymbolId::intern("ddir_x");
/// let y_id = SymbolId::intern("ddir_y");
/// let x = Expr::symbol("ddir_x");
/// let y = Expr::symbol("ddir_y");
/// let f = normalize::add(normalize::pow(x.clone(), Expr::int(2)), y.clone());
/// let dir = vec![Expr::int(1), Expr::int(0)];
/// let result = directional_derivative(&f, &[x_id, y_id], &dir);
/// assert!(!result.is_zero()); // 2x ≠ 0
///
/// // Zero direction vector → result is 0
/// let zero_dir = vec![Expr::int(0), Expr::int(0)];
/// let zero_result = directional_derivative(&f, &[x_id, y_id], &zero_dir);
/// assert!(zero_result.is_zero());
/// ```
pub fn directional_derivative(expr: &Arc<Expr>, vars: &[SymbolId], dir: &[Arc<Expr>]) -> Arc<Expr> {
    let n = vars.len().min(dir.len());
    let mut acc = Expr::int(0);
    for i in 0..n {
        let grad_i = diff_arc(expr, vars[i]);
        let term = normalize::mul(grad_i, dir[i].clone());
        acc = normalize::add(acc, term);
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
    fn test_directional_axis_aligned() {
        // f = x² + y,  dir = (1, 0)  →  Dᵥf = 2x
        let xi = id("dd_ax_x");
        let yi = id("dd_ax_y");
        let x = sym("dd_ax_x");
        let y = sym("dd_ax_y");
        let f = normalize::add(normalize::pow(x.clone(), Expr::int(2)), y.clone());
        let dir = vec![Expr::int(1), Expr::int(0)];
        let result = directional_derivative(&f, &[xi, yi], &dir);
        // 2x  (y contribution zeroed by dir[1]=0)
        assert!(!result.is_zero());
    }

    #[test]
    fn test_directional_zero_direction() {
        // Any f,  dir = (0, 0)  →  0
        let xi = id("dd_z_x");
        let yi = id("dd_z_y");
        let x = sym("dd_z_x");
        let f = normalize::pow(x.clone(), Expr::int(3));
        let dir = vec![Expr::int(0), Expr::int(0)];
        let result = directional_derivative(&f, &[xi, yi], &dir);
        assert!(
            result.is_zero(),
            "directional derivative with zero dir should be 0"
        );
    }

    #[test]
    fn test_directional_diagonal() {
        // f = x + y,  dir = (1, 1)  →  Dᵥf = 1*1 + 1*1 = 2
        let xi = id("dd_diag_x");
        let yi = id("dd_diag_y");
        let x = sym("dd_diag_x");
        let y = sym("dd_diag_y");
        let f = normalize::add(x.clone(), y.clone());
        let dir = vec![Expr::int(1), Expr::int(1)];
        let result = directional_derivative(&f, &[xi, yi], &dir);
        assert_eq!(
            *result,
            Expr::Integer(SmallInt::from(2i64)),
            "D(1,1)(x+y) should be 2"
        );
    }
}
