//! Nabla (del) operator `∇` — unified interface for gradient, divergence,
//! curl, Laplacian, and the classical vector-calculus identities.
//!
//! `Nabla` owns a coordinate system (a list of [`SymbolId`] variables) and
//! delegates every operation to the pure engine functions in
//! [`crate::calculus::multivar`].  The identity methods (`div_of_curl`,
//! `curl_of_grad`, `div_of_grad`) exist to make the mathematical structure
//! explicit; they return the composed result that, for smooth fields, equals
//! zero (vector or scalar).
//!
//! # 3-D requirement
//!
//! `curl` and the two curl-containing identities require exactly 3 coordinate
//! variables.  Calling them with fewer panics at the array-indexing step inside
//! `multivar::curl`; callers must validate the dimension before constructing a
//! [`Nabla`] when curl is intended.

use std::sync::Arc;

use crate::calculus::multivar;
use crate::numeric::Expr;
use crate::numeric::SymbolId;

/// Unified del (`∇`) operator over a fixed coordinate system.
///
/// Construct via [`Nabla::new`] with the variable list that defines the
/// coordinate directions.  All methods delegate to the pure engine
/// functions in [`crate::calculus::multivar`] without allocating extra
/// intermediate representations.
pub struct Nabla {
    vars: Vec<SymbolId>,
}

impl Nabla {
    /// Create a new `Nabla` over the given coordinate variables.
    ///
    /// The variable order determines which partial derivative appears in
    /// which position in the output.  For 3-D operations (`curl`,
    /// `div_of_curl`, `curl_of_grad`) the first three variables are used
    /// as x, y, z respectively; the list must contain at least 3 entries.
    pub fn new(vars: Vec<SymbolId>) -> Self {
        Self { vars }
    }

    /// Gradient `∇f = (∂f/∂x₁, …, ∂f/∂xₙ)`.
    ///
    /// Returns one [`Arc<Expr>`] per coordinate variable, in the same
    /// order as the variable list supplied to [`Nabla::new`].
    pub fn grad(&self, f: &Arc<Expr>) -> Vec<Arc<Expr>> {
        multivar::gradient(f, &self.vars)
    }

    /// Divergence `∇·F = Σᵢ ∂Fᵢ/∂xᵢ`.
    ///
    /// `field` must have the same length as the variable list.
    pub fn div(&self, field: &[Arc<Expr>]) -> Arc<Expr> {
        multivar::divergence(field, &self.vars)
    }

    /// Curl `∇×F` for a 3-D vector field.
    ///
    /// Requires that the `Nabla` was constructed with at least 3 variables.
    /// Uses the first three as (x, y, z).
    pub fn curl(&self, field: &[Arc<Expr>; 3]) -> [Arc<Expr>; 3] {
        let vars: [SymbolId; 3] = [self.vars[0], self.vars[1], self.vars[2]];
        multivar::curl(field, &vars)
    }

    /// Laplacian `∇²f = Σᵢ ∂²f/∂xᵢ²`.
    pub fn laplacian(&self, f: &Arc<Expr>) -> Arc<Expr> {
        multivar::laplacian(f, &self.vars)
    }

    // ── Identity methods ──────────────────────────────────────────────────────

    /// `∇·(∇×F) = 0` — divergence of curl.
    ///
    /// For any smooth 3-D vector field this expression simplifies to zero.
    /// The method computes the composition exactly (no shortcut to the
    /// constant zero) so the engine's simplification can verify the identity
    /// holds for the concrete input.
    ///
    /// Requires the `Nabla` to have at least 3 variables.
    pub fn div_of_curl(&self, field: &[Arc<Expr>; 3]) -> Arc<Expr> {
        let curl_result = self.curl(field);
        // Divergence of curl uses all 3 coordinate directions.
        let vars3 = &self.vars[..3];
        multivar::divergence(&curl_result, vars3)
    }

    /// `∇×(∇f) = 0` — curl of gradient.
    ///
    /// For any smooth scalar field this vector expression simplifies to
    /// (0, 0, 0).  As with `div_of_curl`, the full composition is computed
    /// so the simplification can be verified.
    ///
    /// Requires the `Nabla` to have at least 3 variables.
    pub fn curl_of_grad(&self, f: &Arc<Expr>) -> [Arc<Expr>; 3] {
        let grad = self.grad(f);
        let grad_arr: [Arc<Expr>; 3] = [grad[0].clone(), grad[1].clone(), grad[2].clone()];
        self.curl(&grad_arr)
    }

    /// `∇·(∇f) = ∇²f` — divergence of gradient equals Laplacian.
    ///
    /// Computes the composition directly and returns the same result as
    /// [`Nabla::laplacian`].  Useful for verifying the identity in tests
    /// or exposing the two-step derivation via the trace.
    pub fn div_of_grad(&self, f: &Arc<Expr>) -> Arc<Expr> {
        let grad = self.grad(f);
        self.div(&grad)
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{normalize, Expr, SmallInt, SymbolId};

    fn id(name: &str) -> SymbolId {
        SymbolId::intern(name)
    }

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }

    fn nabla_3d(prefix: &str) -> Nabla {
        Nabla::new(vec![
            id(&format!("{}_x", prefix)),
            id(&format!("{}_y", prefix)),
            id(&format!("{}_z", prefix)),
        ])
    }

    fn syms_3d(prefix: &str) -> (Arc<Expr>, Arc<Expr>, Arc<Expr>) {
        (
            sym(&format!("{}_x", prefix)),
            sym(&format!("{}_y", prefix)),
            sym(&format!("{}_z", prefix)),
        )
    }

    // ── grad ──────────────────────────────────────────────────────────────

    #[test]
    fn fast_nabla_grad_polynomial() {
        // f = x² + y²  →  ∇f = (2x, 2y, 0)
        let nabla = nabla_3d("ng_poly");
        let (x, y, _z) = syms_3d("ng_poly");
        let f = normalize::add(
            normalize::pow(x.clone(), Expr::int(2)),
            normalize::pow(y.clone(), Expr::int(2)),
        );
        let g = nabla.grad(&f);
        assert_eq!(g.len(), 3);
        assert!(!g[0].is_zero(), "∂f/∂x should not be zero");
        assert!(!g[1].is_zero(), "∂f/∂y should not be zero");
        assert!(g[2].is_zero(), "∂f/∂z should be zero (f has no z)");
    }

    // ── div ───────────────────────────────────────────────────────────────

    #[test]
    fn fast_nabla_div_position_field() {
        // F = (x, y, z)  →  ∇·F = 3
        let nabla = nabla_3d("nd_pos");
        let (x, y, z) = syms_3d("nd_pos");
        let field = vec![x, y, z];
        let result = nabla.div(&field);
        assert_eq!(
            *result,
            Expr::Integer(SmallInt::from(3i64)),
            "div(x,y,z) should be 3, got {}",
            result
        );
    }

    // ── curl ──────────────────────────────────────────────────────────────

    #[test]
    fn fast_nabla_curl_rotation() {
        // F = (y, -x, 0)  →  ∇×F = (0, 0, -2)
        let nabla = nabla_3d("nc_rot");
        let (x, y, _z) = syms_3d("nc_rot");
        let field = [y.clone(), normalize::neg(x.clone()), Expr::int(0)];
        let [cx, cy, cz] = nabla.curl(&field);
        assert!(cx.is_zero(), "curl x-component should be 0, got {}", cx);
        assert!(cy.is_zero(), "curl y-component should be 0, got {}", cy);
        assert_eq!(
            *cz,
            Expr::Integer(SmallInt::from(-2i64)),
            "curl z-component should be -2, got {}",
            cz
        );
    }

    // ── laplacian ─────────────────────────────────────────────────────────

    #[test]
    fn fast_nabla_laplacian_quadratic() {
        // f = x² + y² + z²  →  ∇²f = 6
        let nabla = nabla_3d("nl_quad");
        let (x, y, z) = syms_3d("nl_quad");
        let f = normalize::add(
            normalize::add(
                normalize::pow(x, Expr::int(2)),
                normalize::pow(y, Expr::int(2)),
            ),
            normalize::pow(z, Expr::int(2)),
        );
        let result = nabla.laplacian(&f);
        assert_eq!(
            *result,
            Expr::Integer(SmallInt::from(6i64)),
            "laplacian(x²+y²+z²) should be 6, got {}",
            result
        );
    }

    // ── div_of_curl ───────────────────────────────────────────────────────

    #[test]
    fn fast_nabla_div_of_curl_is_zero() {
        // For any smooth F, div(curl(F)) = 0
        // Use F = (y*z, x*z, x*y)
        let nabla = nabla_3d("ndc");
        let (x, y, z) = syms_3d("ndc");
        let f1 = normalize::mul(y.clone(), z.clone());
        let f2 = normalize::mul(x.clone(), z.clone());
        let f3 = normalize::mul(x.clone(), y.clone());
        let field = [f1, f2, f3];
        let result = nabla.div_of_curl(&field);
        assert!(result.is_zero(), "div(curl(F)) should be 0, got {}", result);
    }

    // ── curl_of_grad ──────────────────────────────────────────────────────

    #[test]
    fn fast_nabla_curl_of_grad_is_zero() {
        // For any smooth f, curl(grad(f)) = (0, 0, 0)
        // Use f = x² + y² + z²
        let nabla = nabla_3d("ncg");
        let (x, y, z) = syms_3d("ncg");
        let f = normalize::add(
            normalize::add(
                normalize::pow(x, Expr::int(2)),
                normalize::pow(y, Expr::int(2)),
            ),
            normalize::pow(z, Expr::int(2)),
        );
        let [cx, cy, cz] = nabla.curl_of_grad(&f);
        assert!(cx.is_zero(), "curl(∇f) x-component should be 0, got {}", cx);
        assert!(cy.is_zero(), "curl(∇f) y-component should be 0, got {}", cy);
        assert!(cz.is_zero(), "curl(∇f) z-component should be 0, got {}", cz);
    }

    // ── div_of_grad ───────────────────────────────────────────────────────

    #[test]
    fn fast_nabla_div_of_grad_equals_laplacian() {
        // div(grad(f)) should equal laplacian(f)
        // Use f = x² + y² + z²
        let nabla = nabla_3d("ndg");
        let (x, y, z) = syms_3d("ndg");
        let f = normalize::add(
            normalize::add(
                normalize::pow(x, Expr::int(2)),
                normalize::pow(y, Expr::int(2)),
            ),
            normalize::pow(z, Expr::int(2)),
        );
        let div_grad = nabla.div_of_grad(&f);
        let lap = nabla.laplacian(&f);
        assert_eq!(
            *div_grad, *lap,
            "div(grad(f)) should equal laplacian(f): {} vs {}",
            div_grad, lap
        );
    }
}
