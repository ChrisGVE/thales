//! General curvilinear coordinate machinery.
//!
//! Provides Lamé (scale-factor) coefficients and the curvilinear forms of
//! gradient, divergence, curl, and Laplacian for any orthogonal coordinate
//! system defined by a forward map **x** = **f**(**u**).
//!
//! # Design
//!
//! All internals use `Arc<Expr>` exclusively (Architecture Rule 1).
//! `Expression` never appears inside this module.
//!
//! The key abstraction is [`CurvilinearSystem`].  It stores:
//! - the coordinate names **u** = (u₁, …, uₙ)
//! - the Cartesian forward map **x** = (x₁(u), …, xₙ(u))
//! - the Jacobian column vectors (lazily computed on first call)
//! - the scale factors **h** (lazily derived from the Jacobian)
//!
//! All returned values are `Arc<Expr>` — symbolic, not numeric.

use std::sync::Arc;

use crate::numeric::{
    differentiation::diff_arc,
    expr::{Expr, FuncId},
    normalize, SymbolId,
};

// ── CurvilinearSystem ─────────────────────────────────────────────────────────

/// A general orthogonal curvilinear coordinate system in n dimensions.
///
/// Stores the forward map **x** = **f**(**u**) and provides lazy access to
/// the Jacobian columns, Lamé scale factors, and curvilinear differential
/// operators.
///
/// # Conventions
///
/// * `coords` — new coordinate variable names (u₁, …, uₙ).
/// * `forward_map[i]` — xᵢ expressed in terms of `coords`.
/// * `h[i] = √(Σⱼ (∂xⱼ/∂uᵢ)²)` — the i-th Lamé coefficient.
#[derive(Clone)]
pub struct CurvilinearSystem {
    /// Coordinate variable names (u₁, …, uₙ).
    pub coords: Vec<SymbolId>,
    /// Forward map: xᵢ expressed symbolically in `coords`.
    pub forward_map: Vec<Arc<Expr>>,
}

impl CurvilinearSystem {
    /// Construct a curvilinear system from coordinate names and forward map.
    ///
    /// `coords[i]` is the i-th curvilinear variable.
    /// `forward_map[i]` is xᵢ expressed in those variables.
    ///
    /// # Panics
    ///
    /// Does not panic.
    pub fn new(coords: Vec<SymbolId>, forward_map: Vec<Arc<Expr>>) -> Self {
        Self {
            coords,
            forward_map,
        }
    }

    /// Construct a standard 3-D cylindrical system (ρ, φ, z).
    ///
    /// Forward map: x = ρ·cos(φ), y = ρ·sin(φ), z = z.
    /// Scale factors: h_ρ = 1, h_φ = ρ, h_z = 1.
    pub fn cylindrical() -> Self {
        let rho = SymbolId::intern("rho");
        let phi = SymbolId::intern("phi");
        let z = SymbolId::intern("z");

        let rho_e = Expr::symbol("rho");
        let phi_e = Expr::symbol("phi");
        let z_e = Expr::symbol("z");

        let cos_phi = Expr::func(FuncId::Cos, vec![phi_e.clone()]);
        let sin_phi = Expr::func(FuncId::Sin, vec![phi_e]);

        let x = normalize::mul(rho_e.clone(), cos_phi);
        let y = normalize::mul(rho_e, sin_phi);

        Self::new(vec![rho, phi, z], vec![x, y, z_e])
    }

    /// Construct a standard 3-D spherical system (r, θ, φ).
    ///
    /// Physics convention: θ is the polar angle from the z-axis, φ is the
    /// azimuthal angle in the xy-plane.
    ///
    /// Forward map: x = r·sin(θ)·cos(φ), y = r·sin(θ)·sin(φ), z = r·cos(θ).
    /// Scale factors: h_r = 1, h_θ = r, h_φ = r·sin(θ).
    pub fn spherical() -> Self {
        let r = SymbolId::intern("r");
        let theta = SymbolId::intern("theta");
        let phi = SymbolId::intern("phi");

        let r_e = Expr::symbol("r");
        let theta_e = Expr::symbol("theta");
        let phi_e = Expr::symbol("phi");

        let sin_th = Expr::func(FuncId::Sin, vec![theta_e.clone()]);
        let cos_th = Expr::func(FuncId::Cos, vec![theta_e]);
        let cos_phi = Expr::func(FuncId::Cos, vec![phi_e.clone()]);
        let sin_phi = Expr::func(FuncId::Sin, vec![phi_e]);

        // x = r·sin(θ)·cos(φ)
        let x = normalize::mul(normalize::mul(r_e.clone(), sin_th.clone()), cos_phi);
        // y = r·sin(θ)·sin(φ)
        let y = normalize::mul(normalize::mul(r_e.clone(), sin_th), sin_phi);
        // z = r·cos(θ)
        let z = normalize::mul(r_e, cos_th);

        Self::new(vec![r, theta, phi], vec![x, y, z])
    }

    /// Compute the Jacobian columns: column j = [∂x₀/∂uⱼ, …, ∂xₙ/∂uⱼ].
    ///
    /// Returns a `Vec` of length n, where entry j is itself a `Vec<Arc<Expr>>`
    /// containing the partial derivatives of all Cartesian components with
    /// respect to the j-th curvilinear coordinate.
    pub fn jacobian_columns(&self) -> Vec<Vec<Arc<Expr>>> {
        let n = self.coords.len();
        (0..n)
            .map(|j| {
                self.forward_map
                    .iter()
                    .map(|xi| diff_arc(xi, self.coords[j]))
                    .collect()
            })
            .collect()
    }

    /// Compute the i-th Lamé scale factor: h_i = √(Σⱼ (∂xⱼ/∂uᵢ)²).
    ///
    /// Returns an `Arc<Expr>` that is symbolic (not numerically evaluated).
    pub fn scale_factor(&self, i: usize) -> Arc<Expr> {
        let col = self.jacobian_columns();
        scale_factor_from_column(&col[i])
    }

    /// Compute all n Lamé scale factors, returned in coordinate order.
    pub fn scale_factors(&self) -> Vec<Arc<Expr>> {
        let cols = self.jacobian_columns();
        cols.iter()
            .map(|col| scale_factor_from_column(col))
            .collect()
    }
}

// ── Scale factor helper ───────────────────────────────────────────────────────

/// Build h = √(Σⱼ dⱼ²) from a Jacobian column vector `d`.
fn scale_factor_from_column(column: &[Arc<Expr>]) -> Arc<Expr> {
    let sum_sq = column
        .iter()
        .map(|dij| normalize::pow(dij.clone(), Expr::int(2)))
        .fold(Expr::int(0), normalize::add);
    Expr::func(FuncId::Sqrt, vec![sum_sq])
}

// ── Curvilinear gradient ──────────────────────────────────────────────────────

/// Compute the curvilinear gradient of a scalar field `f`.
///
/// ∇f = Σᵢ (1/hᵢ) · (∂f/∂uᵢ) · êᵢ
///
/// Returns a `Vec<Arc<Expr>>` of length n, where the i-th element is
/// `(1/hᵢ) · (∂f/∂uᵢ)`.
pub fn curvilinear_gradient(f: &Arc<Expr>, sys: &CurvilinearSystem) -> Vec<Arc<Expr>> {
    let hs = sys.scale_factors();
    sys.coords
        .iter()
        .zip(hs.iter())
        .map(|(&ui, hi)| {
            let df_dui = diff_arc(f, ui);
            // (1/hᵢ) * ∂f/∂uᵢ = ∂f/∂uᵢ / hᵢ
            normalize::mul(df_dui, recip(hi.clone()))
        })
        .collect()
}

// ── Curvilinear divergence ────────────────────────────────────────────────────

/// Compute the curvilinear divergence of a vector field **F** = (F₀, F₁, F₂).
///
/// For a 3-D orthogonal system with scale factors (h₁, h₂, h₃):
///
/// ∇·F = (1/(h₁h₂h₃)) · Σᵢ ∂(hⱼhₖFᵢ)/∂uᵢ   (cyclic j, k)
///
/// For an n-D system the formula generalises: each term i is
/// ∂(Fᵢ · Πⱼ≠ᵢ hⱼ) / ∂uᵢ, divided by the product Πhⱼ.
///
/// `field[i]` is the i-th component Fᵢ (as `Arc<Expr>`).
///
/// # Panics
///
/// Panics if `field.len() != sys.coords.len()`.
pub fn curvilinear_divergence(field: &[Arc<Expr>], sys: &CurvilinearSystem) -> Arc<Expr> {
    let n = sys.coords.len();
    assert_eq!(field.len(), n, "field must have same dimension as system");

    let hs = sys.scale_factors();
    let h_product = product_of(&hs);

    // For each i: ∂/∂uᵢ [ Fᵢ · Π_{j≠i} hⱼ ]
    let sum = (0..n)
        .map(|i| {
            let h_others = product_without(i, &hs);
            let integrand = normalize::mul(field[i].clone(), h_others);
            diff_arc(&integrand, sys.coords[i])
        })
        .fold(Expr::int(0), normalize::add);

    normalize::mul(sum, recip(h_product))
}

// ── Curvilinear curl (3D only) ────────────────────────────────────────────────

/// Compute the curvilinear curl of a 3-D vector field **F**.
///
/// Standard formula (Arfken & Weber):
/// (∇×F)₀ = (1/(h₁h₂)) · [∂(h₂F₁)/∂u₀ − ∂(h₀F₀)/∂u₁]  … cyclic
///
/// More precisely, for an orthogonal system (h₀, h₁, h₂):
///
/// (∇×F)_i = (1/(hⱼhₖ)) · [∂(hₖFₖ)/∂uⱼ − ∂(hⱼFⱼ)/∂uₖ]
///
/// where (i, j, k) = cyclic permutation of (0, 1, 2).
///
/// Returns `[curl_0, curl_1, curl_2]`.
///
/// # Panics
///
/// Panics if the system is not exactly 3-D.
pub fn curvilinear_curl(field: &[Arc<Expr>; 3], sys: &CurvilinearSystem) -> [Arc<Expr>; 3] {
    assert_eq!(sys.coords.len(), 3, "curl is only defined for 3-D systems");
    let hs = sys.scale_factors();
    let cs = &sys.coords;

    let curl_component = |i: usize, j: usize, k: usize| -> Arc<Expr> {
        // (∇×F)_i = (1/(hⱼ·hₖ)) · [∂(hₖ·Fₖ)/∂uⱼ − ∂(hⱼ·Fⱼ)/∂uₖ]
        let hj_fj = normalize::mul(hs[j].clone(), field[j].clone());
        let hk_fk = normalize::mul(hs[k].clone(), field[k].clone());
        let term1 = diff_arc(&hk_fk, cs[j]);
        let term2 = diff_arc(&hj_fj, cs[k]);
        let diff = normalize::sub(term1, term2);
        let denom = normalize::mul(hs[j].clone(), hs[k].clone());
        normalize::mul(diff, recip(denom))
    };

    [
        curl_component(0, 1, 2),
        curl_component(1, 2, 0),
        curl_component(2, 0, 1),
    ]
}

// ── Curvilinear Laplacian ─────────────────────────────────────────────────────

/// Compute the curvilinear Laplacian of a scalar field `f`.
///
/// ∇²f = (1/Πhᵢ) · Σᵢ ∂/∂uᵢ [(Πⱼ≠ᵢ hⱼ / hᵢ) · ∂f/∂uᵢ]
///
/// Returns a single `Arc<Expr>`.
pub fn curvilinear_laplacian(f: &Arc<Expr>, sys: &CurvilinearSystem) -> Arc<Expr> {
    let n = sys.coords.len();
    let hs = sys.scale_factors();
    let h_product = product_of(&hs);

    // For each i: ∂/∂uᵢ [ (Πⱼ≠ᵢ hⱼ / hᵢ) · ∂f/∂uᵢ ]
    let sum = (0..n)
        .map(|i| {
            let df_dui = diff_arc(f, sys.coords[i]);
            // weight_i = Πⱼ≠ᵢ hⱼ / hᵢ
            let h_others = product_without(i, &hs);
            let weight = normalize::mul(h_others, recip(hs[i].clone()));
            let integrand = normalize::mul(weight, df_dui);
            diff_arc(&integrand, sys.coords[i])
        })
        .fold(Expr::int(0), normalize::add);

    normalize::mul(sum, recip(h_product))
}

// ── Private arithmetic helpers ────────────────────────────────────────────────

/// Build the symbolic product of all elements in `factors`.
/// Returns `1` (integer) for an empty slice.
fn product_of(factors: &[Arc<Expr>]) -> Arc<Expr> {
    factors.iter().cloned().fold(Expr::int(1), normalize::mul)
}

/// Build the symbolic product of all elements in `factors` except index `skip`.
/// Returns `1` for length-1 slices.
fn product_without(skip: usize, factors: &[Arc<Expr>]) -> Arc<Expr> {
    factors
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != skip)
        .map(|(_, h)| h.clone())
        .fold(Expr::int(1), normalize::mul)
}

/// Symbolic reciprocal: 1 / expr.
fn recip(expr: Arc<Expr>) -> Arc<Expr> {
    normalize::pow(expr, Expr::int(-1))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{evaluation::evaluate, SymbolId};
    use std::collections::HashMap;

    // Helper: evaluate an Arc<Expr> with a binding map.
    fn eval(e: &Arc<Expr>, env: &HashMap<SymbolId, f64>) -> f64 {
        evaluate(e, env).unwrap_or_else(|| panic!("evaluation failed for {e}"))
    }

    // ── Scale factors ─────────────────────────────────────────────────────────

    /// Cylindrical scale factors: (h_ρ, h_φ, h_z) = (1, ρ, 1).
    #[test]
    fn test_curv_cylindrical_scale_factors() {
        let sys = CurvilinearSystem::cylindrical();
        let hs = sys.scale_factors();
        assert_eq!(hs.len(), 3);

        let rho_id = SymbolId::intern("rho");
        let phi_id = SymbolId::intern("phi");
        let z_id = SymbolId::intern("z");

        let rho_val = 3.0_f64;
        let phi_val = 0.7_f64;
        let z_val = 2.0_f64;

        let mut env = HashMap::new();
        env.insert(rho_id, rho_val);
        env.insert(phi_id, phi_val);
        env.insert(z_id, z_val);

        let h0 = eval(&hs[0], &env);
        let h1 = eval(&hs[1], &env);
        let h2 = eval(&hs[2], &env);

        assert!((h0 - 1.0).abs() < 1e-10, "h_rho should be 1, got {h0}");
        assert!(
            (h1 - rho_val).abs() < 1e-10,
            "h_phi should be rho={rho_val}, got {h1}"
        );
        assert!((h2 - 1.0).abs() < 1e-10, "h_z should be 1, got {h2}");
    }

    /// Spherical scale factors: (h_r, h_θ, h_φ) = (1, r, r·sin(θ)).
    #[test]
    fn test_curv_spherical_scale_factors() {
        let sys = CurvilinearSystem::spherical();
        let hs = sys.scale_factors();
        assert_eq!(hs.len(), 3);

        let r_id = SymbolId::intern("r");
        let theta_id = SymbolId::intern("theta");
        let phi_id = SymbolId::intern("phi");

        let r_val = 4.0_f64;
        let theta_val = std::f64::consts::PI / 4.0;
        let phi_val = std::f64::consts::PI / 3.0;

        let mut env = HashMap::new();
        env.insert(r_id, r_val);
        env.insert(theta_id, theta_val);
        env.insert(phi_id, phi_val);

        let h0 = eval(&hs[0], &env);
        let h1 = eval(&hs[1], &env);
        let h2 = eval(&hs[2], &env);

        assert!((h0 - 1.0).abs() < 1e-10, "h_r should be 1, got {h0}");
        assert!(
            (h1 - r_val).abs() < 1e-10,
            "h_theta should be r={r_val}, got {h1}"
        );
        let expected_h2 = r_val * theta_val.sin();
        assert!(
            (h2 - expected_h2).abs() < 1e-10,
            "h_phi should be {expected_h2}, got {h2}"
        );
    }

    // ── Gradient ──────────────────────────────────────────────────────────────

    /// Gradient of f(r) = r² in spherical:
    /// ∂f/∂r = 2r, ∂f/∂θ = 0, ∂f/∂φ = 0
    /// Gradient components: (2r/h_r, 0/h_θ, 0/h_φ) = (2r, 0, 0).
    #[test]
    fn test_curv_spherical_gradient_r_squared() {
        let sys = CurvilinearSystem::spherical();
        let r_id = SymbolId::intern("r");
        let r_e = Expr::symbol("r");
        let f = normalize::pow(r_e, Expr::int(2));

        let grad = curvilinear_gradient(&f, &sys);
        assert_eq!(grad.len(), 3);

        let theta_id = SymbolId::intern("theta");
        let phi_id = SymbolId::intern("phi");
        let r_val = 3.0_f64;
        let theta_val = std::f64::consts::PI / 4.0;
        let phi_val = 0.5_f64;

        let mut env = HashMap::new();
        env.insert(r_id, r_val);
        env.insert(theta_id, theta_val);
        env.insert(phi_id, phi_val);

        let g0 = eval(&grad[0], &env);
        let g1 = eval(&grad[1], &env);
        let g2 = eval(&grad[2], &env);

        assert!(
            (g0 - 2.0 * r_val).abs() < 1e-10,
            "grad[0] should be 2r={}, got {g0}",
            2.0 * r_val
        );
        assert!(g1.abs() < 1e-10, "grad[1] should be 0, got {g1}");
        assert!(g2.abs() < 1e-10, "grad[2] should be 0, got {g2}");
    }

    // ── Divergence ────────────────────────────────────────────────────────────

    /// Divergence of radial field F = (r, 0, 0) in spherical.
    /// Known result: ∇·F = (1/r²) ∂(r²·r)/∂r = (1/r²)·3r² = 3.
    #[test]
    fn test_curv_spherical_divergence_radial_field() {
        let sys = CurvilinearSystem::spherical();
        let r_e = Expr::symbol("r");

        // F = (r, 0, 0)
        let field = vec![r_e, Expr::int(0), Expr::int(0)];
        let div = curvilinear_divergence(&field, &sys);

        let r_id = SymbolId::intern("r");
        let theta_id = SymbolId::intern("theta");
        let phi_id = SymbolId::intern("phi");

        // Test at multiple points — result should be 3 everywhere.
        for &(rv, tv, pv) in &[(2.0_f64, 1.0_f64, 0.5_f64), (5.0, 0.8, 1.2)] {
            let mut env = HashMap::new();
            env.insert(r_id, rv);
            env.insert(theta_id, tv);
            env.insert(phi_id, pv);
            let d = eval(&div, &env);
            assert!(
                (d - 3.0).abs() < 1e-8,
                "div(r,0,0) in spherical should be 3, got {d} at r={rv},θ={tv},φ={pv}"
            );
        }
    }

    // ── Curl ──────────────────────────────────────────────────────────────────

    /// Curl of F = (0, ρ, 0) in cylindrical should give (0, 0, 2).
    /// Standard result from cylindrical-coordinates curl.
    #[test]
    fn test_curv_cylindrical_curl_azimuthal_field() {
        let sys = CurvilinearSystem::cylindrical();
        let rho_e = Expr::symbol("rho");
        let field = [Expr::int(0), rho_e, Expr::int(0)];
        let curl = curvilinear_curl(&field, &sys);

        let rho_id = SymbolId::intern("rho");
        let phi_id = SymbolId::intern("phi");
        let z_id = SymbolId::intern("z");

        for &(rv, pv, zv) in &[(2.0_f64, 1.0_f64, 0.0_f64), (3.5, 0.3, 1.0)] {
            let mut env = HashMap::new();
            env.insert(rho_id, rv);
            env.insert(phi_id, pv);
            env.insert(z_id, zv);

            let c0 = eval(&curl[0], &env);
            let c1 = eval(&curl[1], &env);
            let c2 = eval(&curl[2], &env);

            assert!(c0.abs() < 1e-10, "curl[0] should be 0, got {c0}");
            assert!(c1.abs() < 1e-10, "curl[1] should be 0, got {c1}");
            assert!(
                (c2 - 2.0).abs() < 1e-10,
                "curl[2] should be 2, got {c2} at rho={rv}"
            );
        }
    }

    // ── Laplacian ─────────────────────────────────────────────────────────────

    /// Laplacian of f = 1/r in spherical should be 0 (away from origin).
    ///
    /// ∇²(1/r) = 0 is a classical result.
    #[test]
    fn test_curv_spherical_laplacian_one_over_r() {
        let sys = CurvilinearSystem::spherical();
        let r_e = Expr::symbol("r");
        let f = normalize::pow(r_e, Expr::int(-1)); // 1/r = r^(-1)

        let lap = curvilinear_laplacian(&f, &sys);

        let r_id = SymbolId::intern("r");
        let theta_id = SymbolId::intern("theta");
        let phi_id = SymbolId::intern("phi");

        for &(rv, tv, pv) in &[(2.0_f64, 0.5_f64, 0.3_f64), (5.0, 1.0, 1.2)] {
            let mut env = HashMap::new();
            env.insert(r_id, rv);
            env.insert(theta_id, tv);
            env.insert(phi_id, pv);
            let l = eval(&lap, &env);
            assert!(
                l.abs() < 1e-8,
                "Laplacian(1/r) in spherical should be 0, got {l} at r={rv}"
            );
        }
    }

    // ── General / parabolic cylindrical ──────────────────────────────────────

    /// Parabolic cylindrical system: x = uv, y = (u²-v²)/2, z = z.
    /// Scale factors: h_u = h_v = √(u²+v²), h_z = 1.
    /// Verify h_u (index 0) against the textbook formula.
    #[test]
    fn test_curv_parabolic_cylindrical_scale_factor() {
        let u = SymbolId::intern("pc_u");
        let v = SymbolId::intern("pc_v");
        let z = SymbolId::intern("pc_z");

        let u_e = Expr::symbol("pc_u");
        let v_e = Expr::symbol("pc_v");
        let z_e = Expr::symbol("pc_z");

        // x₁ = u·v, x₂ = (u²-v²)/2, x₃ = z
        let x1 = normalize::mul(u_e.clone(), v_e.clone());
        let u_sq = normalize::pow(u_e.clone(), Expr::int(2));
        let v_sq = normalize::pow(v_e.clone(), Expr::int(2));
        let x2 = normalize::mul(Expr::rational(1, 2), normalize::sub(u_sq, v_sq));

        let sys = CurvilinearSystem::new(vec![u, v, z], vec![x1, x2, z_e]);
        let hs = sys.scale_factors();

        // Evaluate at u=3, v=4 → expected h_u = √(9+16) = 5
        let mut env = HashMap::new();
        env.insert(u, 3.0_f64);
        env.insert(v, 4.0_f64);
        env.insert(z, 0.0_f64);

        let h0 = eval(&hs[0], &env);
        let expected = (3.0_f64 * 3.0 + 4.0 * 4.0_f64).sqrt();
        assert!(
            (h0 - expected).abs() < 1e-10,
            "Parabolic-cylindrical h_u at u=3,v=4 should be {expected}, got {h0}"
        );
    }

    // ── Symbolic free parameter ───────────────────────────────────────────────

    /// Coordinate system with a free symbolic parameter `a`.
    /// Elliptic cylindrical: x = a·cosh(u)·cos(v), y = a·sinh(u)·sin(v), z = z.
    /// Scale factors remain symbolic (in terms of a, u, v) — verify they are
    /// non-zero expressions and evaluate correctly at a concrete point.
    #[test]
    fn test_curv_elliptic_symbolic_parameter() {
        let u = SymbolId::intern("ec_u");
        let v = SymbolId::intern("ec_v");
        let z = SymbolId::intern("ec_z");
        let a = SymbolId::intern("ec_a");

        let u_e = Expr::symbol("ec_u");
        let v_e = Expr::symbol("ec_v");
        let z_e = Expr::symbol("ec_z");
        let a_e = Expr::symbol("ec_a");

        let cosh_u = Expr::func(FuncId::Cosh, vec![u_e.clone()]);
        let sinh_u = Expr::func(FuncId::Sinh, vec![u_e]);
        let cos_v = Expr::func(FuncId::Cos, vec![v_e.clone()]);
        let sin_v = Expr::func(FuncId::Sin, vec![v_e]);

        let x = normalize::mul(normalize::mul(a_e.clone(), cosh_u), cos_v);
        let y = normalize::mul(normalize::mul(a_e.clone(), sinh_u), sin_v);

        let sys = CurvilinearSystem::new(vec![u, v, z], vec![x, y, z_e]);
        let hs = sys.scale_factors();

        // h_u and h_v should be non-zero symbolic expressions.
        assert!(!hs[0].is_zero(), "h_u of elliptic system must not be zero");
        assert!(!hs[1].is_zero(), "h_v of elliptic system must not be zero");
        assert!(!hs[2].is_zero(), "h_z must not be zero");

        // Evaluate at a=2, u=0, v=0 → cosh(0)=1, sinh(0)=0, cos(0)=1, sin(0)=0.
        // h_u = √((a·sinh(u)·cos(v))² + (a·cosh(u)·sin(v))²)  = a·√(sinh²u+cos²v·... hmm)
        // actually h_u = a√(sinh²u + sin²v) — at u=0,v=0 → a·√(0+0) = 0, so pick u=1,v=1.
        let u_val = 1.0_f64;
        let v_val = 1.0_f64;
        let a_val = 2.0_f64;

        let mut env = HashMap::new();
        env.insert(u, u_val);
        env.insert(v, v_val);
        env.insert(z, 0.0_f64);
        env.insert(a, a_val);

        let h0 = eval(&hs[0], &env);
        // h_u = a·√(sinh²(u) + sin²(v))
        let expected_h0 = a_val * (u_val.sinh().powi(2) + v_val.sin().powi(2)).sqrt();
        assert!(
            (h0 - expected_h0).abs() < 1e-9,
            "elliptic-cylindrical h_u should be {expected_h0}, got {h0}"
        );
    }
}
