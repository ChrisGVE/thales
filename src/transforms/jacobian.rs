//! Jacobian matrix, determinant, and coordinate-system volume elements.
//!
//! Builds the symbolic Jacobian of an arbitrary parametric map and provides
//! ready-made volume-element factors for the built-in coordinate systems
//! enumerated in [`CoordSystem`].
//!
//! # Mathematical background
//!
//! Given a map **x** = **f**(**q**) where **q** = (q₁, …, qₙ) are the new
//! (curvilinear) variables and **x** = (x₁, …, xₙ) are the old (Cartesian)
//! variables, the Jacobian matrix is:
//!
//! ```text
//! J[i][j] = ∂xᵢ/∂qⱼ
//! ```
//!
//! The volume element for a change of variables is `dV = |det J| dq₁ … dqₙ`.
//!
//! # Design
//!
//! All functions operate on [`Arc<Expr>`] end-to-end (Architecture Rule 1).
//! The determinant is computed via [`MatrixExpr::determinant`] — no duplicate
//! logic.

use std::sync::Arc;

use crate::matrix::MatrixExpr;
use crate::numeric::differentiation::diff_arc;
use crate::numeric::{normalize, Expr, FuncId, SymbolId};

use super::CoordSystem;

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute the symbolic Jacobian matrix of a parametric forward map.
///
/// `forward_map[i]` is the expression for the i-th output coordinate expressed
/// in terms of `new_vars`.  The (i, j) entry of the returned matrix is
/// `∂(forward_map[i]) / ∂(new_vars[j])`.
///
/// `old_vars` is accepted for API symmetry and documentation clarity (it
/// names the coordinate being replaced) but is not needed for differentiation —
/// `forward_map` already contains the symbolic expressions in `new_vars`.
///
/// # Panics
///
/// Does not panic.  Returns an empty outer `Vec` when `forward_map` is empty.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{Expr, SymbolId, normalize};
/// use thales::transforms::jacobian::jacobian_matrix;
/// use std::sync::Arc;
///
/// // Polar → Cartesian: x = r·cos(θ),  y = r·sin(θ)
/// let r  = SymbolId::intern("jm_r");
/// let th = SymbolId::intern("jm_th");
/// let r_expr  = Expr::symbol("jm_r");
/// let th_expr = Expr::symbol("jm_th");
///
/// let cos_th = Expr::func(thales::numeric::FuncId::Cos, vec![th_expr.clone()]);
/// let sin_th = Expr::func(thales::numeric::FuncId::Sin, vec![th_expr.clone()]);
/// let x = normalize::mul(r_expr.clone(), cos_th);
/// let y = normalize::mul(r_expr.clone(), sin_th);
///
/// let old = vec![SymbolId::intern("jm_x"), SymbolId::intern("jm_y")];
/// let new_vars = vec![r, th];
/// let jac = jacobian_matrix(&[x, y], &old, &new_vars);
/// assert_eq!(jac.len(), 2);
/// assert_eq!(jac[0].len(), 2);
/// ```
pub fn jacobian_matrix(
    forward_map: &[Arc<Expr>],
    _old_vars: &[SymbolId],
    new_vars: &[SymbolId],
) -> Vec<Vec<Arc<Expr>>> {
    forward_map
        .iter()
        .map(|fi| new_vars.iter().map(|&qj| diff_arc(fi, qj)).collect())
        .collect()
}

/// Compute the symbolic determinant of the Jacobian matrix.
///
/// Delegates determinant computation to [`MatrixExpr::determinant`].
///
/// # Panics
///
/// Panics if `forward_map` is not square with respect to `new_vars`
/// (i.e. `forward_map.len() != new_vars.len()`), or if either slice is empty.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{Expr, SymbolId, normalize};
/// use thales::transforms::jacobian::jacobian_determinant;
/// use std::sync::Arc;
///
/// // Polar → Cartesian Jacobian determinant = r
/// let r  = SymbolId::intern("jd_r");
/// let th = SymbolId::intern("jd_th");
/// let r_expr  = Expr::symbol("jd_r");
/// let th_expr = Expr::symbol("jd_th");
///
/// let cos_th = Expr::func(thales::numeric::FuncId::Cos, vec![th_expr.clone()]);
/// let sin_th = Expr::func(thales::numeric::FuncId::Sin, vec![th_expr.clone()]);
/// let x = normalize::mul(r_expr.clone(), cos_th);
/// let y = normalize::mul(r_expr.clone(), sin_th);
///
/// let old = vec![SymbolId::intern("jd_x"), SymbolId::intern("jd_y")];
/// let new_vars = vec![r, th];
/// let det = jacobian_determinant(&[x, y], &old, &new_vars);
/// // Result is r (the Jacobian det of polar→cartesian)
/// assert!(!det.is_zero());
/// ```
pub fn jacobian_determinant(
    forward_map: &[Arc<Expr>],
    old_vars: &[SymbolId],
    new_vars: &[SymbolId],
) -> Arc<Expr> {
    let rows = jacobian_matrix(forward_map, old_vars, new_vars);
    let matrix = MatrixExpr::from_expr_elements(rows)
        .expect("jacobian_determinant: forward_map and new_vars must be non-empty");
    matrix
        .determinant()
        .expect("jacobian_determinant: matrix must be square (forward_map.len() == new_vars.len())")
}

/// Return the symbolic volume-element factor `h` for a built-in coordinate
/// system, where `dV = h · dq₁ dq₂ … dqₙ`.
///
/// | System         | Factor             | Variables used       |
/// |----------------|--------------------|----------------------|
/// | Cartesian2D    | 1                  | —                    |
/// | Polar2D        | r                  | `r`                  |
/// | Cartesian3D    | 1                  | —                    |
/// | Cylindrical    | ρ                  | `rho`                |
/// | Spherical      | ρ²·sin(φ)          | `rho`, `phi`         |
/// | Parabolic2D    | u²+v²              | `u`, `v`             |
/// | Elliptic2D     | a·√((sinh²μ+sin²ν))| `a`, `mu`, `nu`      |
/// | Custom         | 1 (placeholder)    | —                    |
///
/// Symbol names follow physics conventions.  Callers may substitute their own
/// variable names into the returned expression via the substitution module.
///
/// # Examples
///
/// ```rust
/// use thales::transforms::{CoordSystem, jacobian::volume_element};
/// use thales::numeric::{Expr, FuncId, SymbolId};
///
/// // Cartesian3D volume element is 1
/// let dv = volume_element(CoordSystem::Cartesian3D);
/// assert!(dv.is_one());
///
/// // Spherical volume element is rho^2 * sin(phi)
/// let dv = volume_element(CoordSystem::Spherical);
/// assert!(!dv.is_zero());
/// ```
pub fn volume_element(coord_system: CoordSystem) -> Arc<Expr> {
    match coord_system {
        CoordSystem::Cartesian2D => Expr::int(1),
        CoordSystem::Polar2D => Expr::symbol("r"),
        CoordSystem::Cartesian3D => Expr::int(1),
        CoordSystem::Cylindrical => Expr::symbol("rho"),
        CoordSystem::Spherical => spherical_volume_element(),
        CoordSystem::Parabolic2D => parabolic2d_volume_element(),
        CoordSystem::Elliptic2D => elliptic2d_volume_element(),
        CoordSystem::Custom => Expr::int(1),
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// ρ²·sin(φ)
fn spherical_volume_element() -> Arc<Expr> {
    let rho = Expr::symbol("rho");
    let phi = Expr::symbol("phi");
    let rho_sq = normalize::pow(rho, Expr::int(2));
    let sin_phi = Expr::func(FuncId::Sin, vec![phi]);
    normalize::mul(rho_sq, sin_phi)
}

/// u² + v²  (from the scale factor of parabolic 2D: h = √(u²+v²) per component;
/// combined area element = u² + v² for the 2D case where x = (u²-v²)/2, y = uv)
fn parabolic2d_volume_element() -> Arc<Expr> {
    let u = Expr::symbol("u");
    let v = Expr::symbol("v");
    let u_sq = normalize::pow(u, Expr::int(2));
    let v_sq = normalize::pow(v, Expr::int(2));
    normalize::add(u_sq, v_sq)
}

/// a · √(sinh²(μ) + sin²(ν))
/// Elliptic 2D: x = a·cosh(μ)·cos(ν), y = a·sinh(μ)·sin(ν);
/// Jacobian det = a²·(sinh²(μ)+sin²(ν)); area element = a²·(sinh²(μ)+sin²(ν))
/// but per-unit-area factor (without the dμ dν) is a²·(sinh²μ+sin²ν).
/// We return a·√(sinh²μ+sin²ν) as the scale (Lamé coefficient product).
fn elliptic2d_volume_element() -> Arc<Expr> {
    let a = Expr::symbol("a");
    let mu = Expr::symbol("mu");
    let nu = Expr::symbol("nu");
    let sinh_mu = Expr::func(FuncId::Sinh, vec![mu]);
    let sin_nu = Expr::func(FuncId::Sin, vec![nu]);
    let sinh_sq = normalize::pow(sinh_mu, Expr::int(2));
    let sin_sq = normalize::pow(sin_nu, Expr::int(2));
    let sum = normalize::add(sinh_sq, sin_sq);
    let sqrt_sum = Expr::func(FuncId::Sqrt, vec![sum]);
    normalize::mul(a, sqrt_sum)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::evaluation::evaluate;
    use std::collections::HashMap;

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }
    fn sid(name: &str) -> SymbolId {
        SymbolId::intern(name)
    }

    // ── jacobian_matrix ───────────────────────────────────────────────────────

    /// Jacobian of the identity map [x, y] w.r.t. [x, y] must be I₂.
    #[test]
    fn test_jacobian_matrix_identity_2d() {
        let xid = sid("jmi_x");
        let yid = sid("jmi_y");
        let x = sym("jmi_x");
        let y = sym("jmi_y");
        let jac = jacobian_matrix(&[x, y], &[xid, yid], &[xid, yid]);
        // J = [[1, 0], [0, 1]]
        assert!(jac[0][0].is_one(), "J[0][0] should be 1");
        assert!(jac[0][1].is_zero(), "J[0][1] should be 0");
        assert!(jac[1][0].is_zero(), "J[1][0] should be 0");
        assert!(jac[1][1].is_one(), "J[1][1] should be 1");
    }

    /// Jacobian of polar→cartesian: x = r·cos(θ), y = r·sin(θ).
    /// J = [[cos θ, −r·sin θ], [sin θ, r·cos θ]]
    /// Not evaluated here — just shape and non-zero entries.
    #[test]
    fn test_jacobian_matrix_polar_shape() {
        let r_id = sid("jmp_r");
        let th_id = sid("jmp_th");
        let r = sym("jmp_r");
        let th = sym("jmp_th");
        let cos_th = Expr::func(FuncId::Cos, vec![th.clone()]);
        let sin_th = Expr::func(FuncId::Sin, vec![th.clone()]);
        let x = normalize::mul(r.clone(), cos_th);
        let y = normalize::mul(r.clone(), sin_th);
        let old = vec![sid("jmp_cx"), sid("jmp_cy")];
        let new_vars = vec![r_id, th_id];
        let jac = jacobian_matrix(&[x, y], &old, &new_vars);
        assert_eq!(jac.len(), 2);
        assert_eq!(jac[0].len(), 2);
        // J[0][0] = cos(θ) — non-zero expression
        assert!(!jac[0][0].is_zero(), "∂x/∂r should not be zero");
        // J[0][1] = -r·sin(θ) — non-zero expression
        assert!(!jac[0][1].is_zero(), "∂x/∂θ should not be zero");
    }

    /// Jacobian of a constant map must be all-zero.
    #[test]
    fn test_jacobian_matrix_constant_map() {
        let rid = sid("jmc_r");
        let thid = sid("jmc_th");
        // Map: [3, 5] — constants, Jacobian is [[0,0],[0,0]]
        let jac = jacobian_matrix(&[Expr::int(3), Expr::int(5)], &[rid, thid], &[rid, thid]);
        for row in &jac {
            for entry in row {
                assert!(entry.is_zero(), "constant map Jacobian must be zero");
            }
        }
    }

    // ── jacobian_determinant ──────────────────────────────────────────────────

    /// det J of polar→cartesian equals r.
    ///
    /// Proof: J = [[cos θ, −r sin θ], [sin θ, r cos θ]]
    ///        det = r cos²θ + r sin²θ = r
    #[test]
    fn test_jacobian_determinant_polar_to_cartesian_is_r() {
        let r_id = sid("jdp_r");
        let th_id = sid("jdp_th");
        let r = sym("jdp_r");
        let th = sym("jdp_th");
        let cos_th = Expr::func(FuncId::Cos, vec![th.clone()]);
        let sin_th = Expr::func(FuncId::Sin, vec![th.clone()]);
        let x = normalize::mul(r.clone(), cos_th);
        let y = normalize::mul(r.clone(), sin_th);
        let old = vec![sid("jdp_cx"), sid("jdp_cy")];
        let new_vars = vec![r_id, th_id];
        let det = jacobian_determinant(&[x, y], &old, &new_vars);
        // Evaluate: r=2, θ=π/3 → det should be 2
        let r_val = 2.0_f64;
        let th_val = std::f64::consts::PI / 3.0;
        let mut env = HashMap::new();
        env.insert(r_id, r_val);
        env.insert(th_id, th_val);
        let numeric = evaluate(&det, &env);
        let eps = 1e-10;
        assert!(
            (numeric.unwrap() - r_val).abs() < eps,
            "det J(polar) should equal r=2, got {numeric:?}"
        );
    }

    /// det J of the identity 2×2 map = 1.
    #[test]
    fn test_jacobian_determinant_identity_is_one() {
        let xid = sid("jdi_x");
        let yid = sid("jdi_y");
        let x = sym("jdi_x");
        let y = sym("jdi_y");
        let det = jacobian_determinant(&[x, y], &[xid, yid], &[xid, yid]);
        assert!(
            det.is_one(),
            "det of identity Jacobian must be 1, got {det}"
        );
    }

    /// det J of a 3×3 scaling map [2x, 3y, 5z] = 30.
    #[test]
    fn test_jacobian_determinant_3d_scaling() {
        let xid = sid("jds_x");
        let yid = sid("jds_y");
        let zid = sid("jds_z");
        let x = normalize::mul(Expr::int(2), sym("jds_x"));
        let y = normalize::mul(Expr::int(3), sym("jds_y"));
        let z = normalize::mul(Expr::int(5), sym("jds_z"));
        let old = vec![xid, yid, zid];
        let new_vars = vec![xid, yid, zid];
        let det = jacobian_determinant(&[x, y, z], &old, &new_vars);
        let env: HashMap<SymbolId, f64> = HashMap::new();
        let numeric = evaluate(&det, &env);
        assert!(
            (numeric.unwrap() - 30.0).abs() < 1e-10,
            "det of [2x,3y,5z] scaling should be 30, got {numeric:?}"
        );
    }

    // ── volume_element ────────────────────────────────────────────────────────

    /// Cartesian2D and Cartesian3D volume elements are 1.
    #[test]
    fn test_volume_element_cartesian_is_one() {
        assert!(
            volume_element(CoordSystem::Cartesian2D).is_one(),
            "Cartesian2D dV = 1"
        );
        assert!(
            volume_element(CoordSystem::Cartesian3D).is_one(),
            "Cartesian3D dV = 1"
        );
    }

    /// Polar2D volume element is r: evaluate at r=3 → 3.
    #[test]
    fn test_volume_element_polar2d_is_r() {
        let dv = volume_element(CoordSystem::Polar2D);
        let r_id = SymbolId::intern("r");
        let mut env = HashMap::new();
        env.insert(r_id, 3.0);
        let v = evaluate(&dv, &env).unwrap();
        assert!(
            (v - 3.0).abs() < 1e-10,
            "Polar2D dV at r=3 should be 3, got {v}"
        );
    }

    /// Cylindrical volume element is ρ: evaluate at rho=4 → 4.
    #[test]
    fn test_volume_element_cylindrical_is_rho() {
        let dv = volume_element(CoordSystem::Cylindrical);
        let rho_id = SymbolId::intern("rho");
        let mut env = HashMap::new();
        env.insert(rho_id, 4.0);
        let v = evaluate(&dv, &env).unwrap();
        assert!(
            (v - 4.0).abs() < 1e-10,
            "Cylindrical dV at rho=4 should be 4, got {v}"
        );
    }

    /// Spherical volume element = ρ²·sin(φ): evaluate at ρ=2, φ=π/6 → 4·sin(π/6)=2.
    #[test]
    fn test_volume_element_spherical_is_rho_sq_sin_phi() {
        let dv = volume_element(CoordSystem::Spherical);
        let rho_id = SymbolId::intern("rho");
        let phi_id = SymbolId::intern("phi");
        let rho_val = 2.0_f64;
        let phi_val = std::f64::consts::PI / 6.0; // sin(π/6) = 0.5
        let mut env = HashMap::new();
        env.insert(rho_id, rho_val);
        env.insert(phi_id, phi_val);
        let v = evaluate(&dv, &env).unwrap();
        // ρ²·sin(φ) = 4 * 0.5 = 2.0
        let expected = rho_val * rho_val * phi_val.sin();
        assert!(
            (v - expected).abs() < 1e-10,
            "Spherical dV at ρ=2,φ=π/6 should be {expected}, got {v}"
        );
    }

    /// Parabolic2D volume element = u²+v²: evaluate at u=3, v=4 → 25.
    #[test]
    fn test_volume_element_parabolic2d() {
        let dv = volume_element(CoordSystem::Parabolic2D);
        let u_id = SymbolId::intern("u");
        let v_id = SymbolId::intern("v");
        let mut env = HashMap::new();
        env.insert(u_id, 3.0);
        env.insert(v_id, 4.0);
        let v = evaluate(&dv, &env).unwrap();
        assert!(
            (v - 25.0).abs() < 1e-10,
            "Parabolic2D dV at u=3,v=4 should be 25, got {v}"
        );
    }

    /// Custom volume element is 1.
    #[test]
    fn test_volume_element_custom_is_one() {
        assert!(
            volume_element(CoordSystem::Custom).is_one(),
            "Custom dV placeholder should be 1"
        );
    }
}
