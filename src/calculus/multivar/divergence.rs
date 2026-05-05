//! Divergence of a vector field.

use std::sync::Arc;

use crate::numeric::{normalize, Expr, SymbolId};

use super::gradient::partial;

// ── Divergence ────────────────────────────────────────────────────────────────

/// Compute the divergence ∇·F = Σᵢ ∂Fᵢ/∂xᵢ.
///
/// `field` and `vars` must have the same length.  If they differ, the
/// shorter one determines how many terms are summed (trailing components
/// of the longer slice are ignored).  Callers that need strict validation
/// should check lengths before calling.
///
/// Each component partial is computed via [`partial`] (which delegates to
/// the normalising [`diff_arc`][crate::numeric::differentiation::diff_arc]
/// engine), so constant-folding and identity removal are applied
/// automatically.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use thales::numeric::{Expr, SymbolId, normalize};
/// use thales::calculus::multivar::divergence;
///
/// // F = (x, y, z)  →  ∇·F = 3
/// let xi = SymbolId::intern("div_ex_x");
/// let yi = SymbolId::intern("div_ex_y");
/// let zi = SymbolId::intern("div_ex_z");
/// let x = Expr::symbol("div_ex_x");
/// let y = Expr::symbol("div_ex_y");
/// let z = Expr::symbol("div_ex_z");
/// let field = vec![x.clone(), y.clone(), z.clone()];
/// let result = divergence(&field, &[xi, yi, zi]);
/// assert_eq!(*result, Expr::Integer(thales::numeric::SmallInt::from(3i64)));
/// ```
pub fn divergence(field: &[Arc<Expr>], vars: &[SymbolId]) -> Arc<Expr> {
    let n = field.len().min(vars.len());
    let mut acc = Expr::int(0);
    for i in 0..n {
        let term = partial(&field[i], &vars[i]);
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
    fn fast_divergence_linear_field() {
        // F = (x, y, z)  →  ∇·F = 1 + 1 + 1 = 3
        let xi = id("dv_lin_x");
        let yi = id("dv_lin_y");
        let zi = id("dv_lin_z");
        let x = sym("dv_lin_x");
        let y = sym("dv_lin_y");
        let z = sym("dv_lin_z");
        let field = vec![x.clone(), y.clone(), z.clone()];
        let result = divergence(&field, &[xi, yi, zi]);
        assert_eq!(
            *result,
            Expr::Integer(SmallInt::from(3i64)),
            "∇·(x,y,z) should be 3"
        );
    }

    #[test]
    fn fast_divergence_quadratic_field() {
        // F = (x², y², z²)  →  ∇·F = 2x + 2y + 2z  (non-zero, not constant)
        let xi = id("dv_quad_x");
        let yi = id("dv_quad_y");
        let zi = id("dv_quad_z");
        let x = sym("dv_quad_x");
        let y = sym("dv_quad_y");
        let z = sym("dv_quad_z");
        let field = vec![
            normalize::pow(x.clone(), Expr::int(2)),
            normalize::pow(y.clone(), Expr::int(2)),
            normalize::pow(z.clone(), Expr::int(2)),
        ];
        let result = divergence(&field, &[xi, yi, zi]);
        // 2x + 2y + 2z ≠ 0 and not a constant integer
        assert!(!result.is_zero(), "∇·(x²,y²,z²) should not be zero");
        assert!(
            !matches!(*result, Expr::Integer(_)),
            "∇·(x²,y²,z²) should not be a constant"
        );
    }

    #[test]
    fn fast_divergence_mismatched_length_truncates() {
        // field has 2 components, vars has 3 — result uses min(2,3)=2 terms
        let xi = id("dv_mm_x");
        let yi = id("dv_mm_y");
        let zi = id("dv_mm_z");
        let x = sym("dv_mm_x");
        let y = sym("dv_mm_y");
        // F = (x, y) with vars (x, y, z) — should still work, giving 2
        let field = vec![x.clone(), y.clone()];
        let result = divergence(&field, &[xi, yi, zi]);
        assert_eq!(
            *result,
            Expr::Integer(SmallInt::from(2i64)),
            "truncated divergence of (x,y) w.r.t. (x,y,z) should be 2"
        );
    }

    #[test]
    fn fast_divergence_constant_field() {
        // F = (3, 5, 7)  →  ∇·F = 0 + 0 + 0 = 0
        let xi = id("dv_const_x");
        let yi = id("dv_const_y");
        let zi = id("dv_const_z");
        let field = vec![Expr::int(3), Expr::int(5), Expr::int(7)];
        let result = divergence(&field, &[xi, yi, zi]);
        assert!(result.is_zero(), "∇·(3,5,7) should be 0");
    }
}
