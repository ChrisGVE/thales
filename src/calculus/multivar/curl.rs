//! Curl of a 3-D vector field.

use std::sync::Arc;

use crate::numeric::{normalize, Expr, SymbolId};

use super::gradient::partial;

// ── Curl ──────────────────────────────────────────────────────────────────────

/// Compute the curl ∇×F for a 3-D vector field.
///
/// The result is the 3-component vector:
///
/// ```text
/// ∇×F = ( ∂F₃/∂y − ∂F₂/∂z,
///          ∂F₁/∂z − ∂F₃/∂x,
///          ∂F₂/∂x − ∂F₁/∂y )
/// ```
///
/// where `vars = [x, y, z]` and `field = [F₁, F₂, F₃]`.
///
/// Each partial is computed via [`partial`] which delegates to the
/// normalising [`diff_arc`][crate::numeric::differentiation::diff_arc]
/// engine, so constant-folding and identity removal are applied
/// automatically.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use thales::numeric::{Expr, SymbolId, normalize};
/// use thales::calculus::multivar::curl;
///
/// // F = (y, -x, 0)  →  ∇×F = (0, 0, -2)
/// let xi = SymbolId::intern("curl_ex_x");
/// let yi = SymbolId::intern("curl_ex_y");
/// let zi = SymbolId::intern("curl_ex_z");
/// let x = Expr::symbol("curl_ex_x");
/// let y = Expr::symbol("curl_ex_y");
/// let z = Expr::symbol("curl_ex_z");
/// let field = [y.clone(), normalize::neg(x.clone()), Expr::int(0)];
/// let [cx, cy, cz] = curl(&field, &[xi, yi, zi]);
/// assert!(cx.is_zero());
/// assert!(cy.is_zero());
/// ```
pub fn curl(field: &[Arc<Expr>; 3], vars: &[SymbolId; 3]) -> [Arc<Expr>; 3] {
    let [f1, f2, f3] = field;
    let [vx, vy, vz] = vars;

    // cx = ∂F₃/∂y − ∂F₂/∂z
    let cx = normalize::sub(partial(f3, vy), partial(f2, vz));
    // cy = ∂F₁/∂z − ∂F₃/∂x
    let cy = normalize::sub(partial(f1, vz), partial(f3, vx));
    // cz = ∂F₂/∂x − ∂F₁/∂y
    let cz = normalize::sub(partial(f2, vx), partial(f1, vy));

    [cx, cy, cz]
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

    fn ids(prefix: &str) -> [SymbolId; 3] {
        [
            id(&format!("{}_x", prefix)),
            id(&format!("{}_y", prefix)),
            id(&format!("{}_z", prefix)),
        ]
    }

    fn syms(prefix: &str) -> [Arc<Expr>; 3] {
        [
            sym(&format!("{}_x", prefix)),
            sym(&format!("{}_y", prefix)),
            sym(&format!("{}_z", prefix)),
        ]
    }

    #[test]
    fn fast_curl_rotational_field() {
        // F = (y, -x, 0)  →  ∇×F = (0, 0, -2)
        let vars = ids("cr_rot");
        let [x, y, _z] = syms("cr_rot");
        let field = [y.clone(), normalize::neg(x.clone()), Expr::int(0)];
        let [cx, cy, cz] = curl(&field, &vars);
        assert!(cx.is_zero(), "curl x-component should be 0, got {}", cx);
        assert!(cy.is_zero(), "curl y-component should be 0, got {}", cy);
        assert_eq!(
            *cz,
            Expr::Integer(SmallInt::from(-2i64)),
            "curl z-component should be -2, got {}",
            cz
        );
    }

    #[test]
    fn fast_curl_irrotational_field() {
        // F = (x, y, z)  →  ∇×F = (0, 0, 0)
        let vars = ids("cr_irr");
        let [x, y, z] = syms("cr_irr");
        let field = [x.clone(), y.clone(), z.clone()];
        let [cx, cy, cz] = curl(&field, &vars);
        assert!(cx.is_zero(), "curl x-component should be 0");
        assert!(cy.is_zero(), "curl y-component should be 0");
        assert!(cz.is_zero(), "curl z-component should be 0");
    }

    #[test]
    fn fast_curl_of_gradient_is_zero() {
        // For f = x² + y² + z², ∇f = (2x, 2y, 2z), curl(∇f) = 0
        let vars = ids("cr_grad");
        let [x, y, z] = syms("cr_grad");
        let f = normalize::add(
            normalize::add(
                normalize::pow(x.clone(), Expr::int(2)),
                normalize::pow(y.clone(), Expr::int(2)),
            ),
            normalize::pow(z.clone(), Expr::int(2)),
        );
        // gradient is (2x, 2y, 2z)
        let grad_x = partial(&f, &vars[0]);
        let grad_y = partial(&f, &vars[1]);
        let grad_z = partial(&f, &vars[2]);
        let field = [grad_x, grad_y, grad_z];
        let [cx, cy, cz] = curl(&field, &vars);
        assert!(cx.is_zero(), "curl(∇f) x-component should be 0");
        assert!(cy.is_zero(), "curl(∇f) y-component should be 0");
        assert!(cz.is_zero(), "curl(∇f) z-component should be 0");
    }
}
