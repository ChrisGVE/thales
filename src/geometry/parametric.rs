//! Parametric curves and surfaces over [`Arc<Expr>`].
//!
//! All differential quantities are computed symbolically via the same
//! differentiation engine used elsewhere in thales. No numeric evaluation
//! occurs inside this module.
//!
//! # Conventions
//!
//! ## Normal vector
//!
//! **2D curve**: `r(t) = (x(t), y(t))`.
//! The tangent is `T = (x', y')`.  The principal normal is obtained by
//! rotating T by 90° counter-clockwise:
//!
//! ```text
//! N₂D = (-y', x')
//! ```
//!
//! This points to the left of the direction of travel (inward for a
//! counter-clockwise loop).  No normalization is applied; the caller
//! receives the un-normalized symbolic vector.
//!
//! **3D curve**: `r(t) = (x(t), y(t), z(t))`.
//! The osculating-plane normal is the component of `r''` that is
//! orthogonal to `r'`:
//!
//! ```text
//! N₃D = r'' - (r''·r' / r'·r') · r'
//! ```
//!
//! This is the standard Frenet–Serret principal normal direction (unnormalized).
//! For a rectilinear curve where `r'' ‖ r'` the result will be the zero vector.
//!
//! ## Curvature
//!
//! **2D**: `κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)` — the signed
//! determinant divided by the speed cubed.  The absolute-value wrapper is
//! applied symbolically via [`FuncId::Abs`].
//!
//! **3D**: `κ = |r' × r''| / |r'|³` where the cross-product magnitude is
//! computed as `sqrt((cross)·(cross))`.

use crate::numeric::{differentiation::diff_arc, expr::FuncId, normalize, Expr, SymbolId};
use std::sync::Arc;

// ── ParametricCurve ───────────────────────────────────────────────────────────

/// A parametric curve in 2-D or 3-D given as a vector of symbolic components.
///
/// - `components.len() == 2` → 2-D curve in the xy-plane.
/// - `components.len() == 3` → 3-D curve in xyz-space.
///
/// All components are [`Arc<Expr>`] expressions in the single parameter
/// [`param`](ParametricCurve::param).
///
/// # Examples
///
/// Unit circle parameterised by `t`:
///
/// ```rust
/// use std::sync::Arc;
/// use thales::numeric::{Expr, SymbolId, FuncId};
/// use thales::geometry::parametric::ParametricCurve;
///
/// let t_id = SymbolId::intern("pc_circle_t");
/// let t = Expr::symbol("pc_circle_t");
/// let cos_t = Expr::func(FuncId::Cos, vec![t.clone()]);
/// let sin_t = Expr::func(FuncId::Sin, vec![t.clone()]);
/// let circle = ParametricCurve { components: vec![cos_t, sin_t], param: t_id };
/// assert_eq!(circle.components.len(), 2);
/// ```
#[derive(Clone, Debug)]
pub struct ParametricCurve {
    /// Component expressions, one per spatial dimension.
    pub components: Vec<Arc<Expr>>,
    /// The parameter symbol with respect to which derivatives are taken.
    pub param: SymbolId,
}

impl ParametricCurve {
    /// Construct a `ParametricCurve` from component expressions and a parameter.
    pub fn new(components: Vec<Arc<Expr>>, param: SymbolId) -> Self {
        Self { components, param }
    }

    /// Compute the tangent vector `r'(t) = (dc₁/dt, dc₂/dt, …)`.
    ///
    /// Each component is differentiated with respect to [`param`](Self::param).
    /// The result is a vector of the same length as `components`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use thales::numeric::{Expr, SymbolId, FuncId};
    /// use thales::geometry::parametric::ParametricCurve;
    ///
    /// // r(t) = (cos t, sin t): r'(t) = (-sin t, cos t)
    /// let t_id = SymbolId::intern("tv_circ_t");
    /// let t = Expr::symbol("tv_circ_t");
    /// let c = ParametricCurve::new(
    ///     vec![
    ///         Expr::func(FuncId::Cos, vec![t.clone()]),
    ///         Expr::func(FuncId::Sin, vec![t.clone()]),
    ///     ],
    ///     t_id,
    /// );
    /// let tv = c.tangent_vector();
    /// assert_eq!(tv.len(), 2);
    /// assert!(!tv[0].is_zero()); // -sin t
    /// assert!(!tv[1].is_zero()); //  cos t
    /// ```
    pub fn tangent_vector(&self) -> Vec<Arc<Expr>> {
        self.components
            .iter()
            .map(|c| diff_arc(c, self.param))
            .collect()
    }

    /// Compute the (unnormalized) normal vector.
    ///
    /// See the [module-level conventions](self) for the precise mathematical
    /// definition used for 2-D and 3-D curves.
    ///
    /// Panics if `components.len()` is neither 2 nor 3.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use thales::numeric::{Expr, SymbolId, FuncId};
    /// use thales::geometry::parametric::ParametricCurve;
    ///
    /// // Unit circle: N₂D = (sin t, cos t) — (−(−sin t), cos t)
    /// let t_id = SymbolId::intern("nv_circ_t");
    /// let t = Expr::symbol("nv_circ_t");
    /// let c = ParametricCurve::new(
    ///     vec![
    ///         Expr::func(FuncId::Cos, vec![t.clone()]),
    ///         Expr::func(FuncId::Sin, vec![t.clone()]),
    ///     ],
    ///     t_id,
    /// );
    /// let nv = c.normal_vector();
    /// assert_eq!(nv.len(), 2);
    /// // -T_y = -cos(t) ≠ 0
    /// assert!(!nv[0].is_zero());
    /// // T_x = -sin(t) ≠ 0
    /// assert!(!nv[1].is_zero());
    /// ```
    pub fn normal_vector(&self) -> Vec<Arc<Expr>> {
        let n = self.components.len();
        assert!(
            n == 2 || n == 3,
            "normal_vector requires a 2-D or 3-D curve (got {} components)",
            n
        );
        let tangent = self.tangent_vector();
        if n == 2 {
            // N₂D = (-T_y, T_x)
            vec![normalize::neg(tangent[1].clone()), tangent[0].clone()]
        } else {
            // N₃D = r'' − (r''·r' / r'·r') · r'
            let r_pp = second_derivatives(&self.components, self.param);
            let r_p = &tangent;

            let dot_rpp_rp = dot3(&r_pp, r_p);
            let dot_rp_rp = dot3(r_p, r_p);
            let proj_scalar = normalize::div(dot_rpp_rp, dot_rp_rp);

            (0..3)
                .map(|i| {
                    let proj_i = normalize::mul(proj_scalar.clone(), r_p[i].clone());
                    normalize::sub(r_pp[i].clone(), proj_i)
                })
                .collect()
        }
    }

    /// Compute the symbolic curvature `κ`.
    ///
    /// See the [module-level conventions](self) for the formula used.
    ///
    /// Panics if `components.len()` is neither 2 nor 3.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use thales::numeric::{Expr, SymbolId, FuncId};
    /// use thales::geometry::parametric::ParametricCurve;
    ///
    /// // Unit circle has curvature 1 (after simplification).
    /// let t_id = SymbolId::intern("curv_circ_t");
    /// let t = Expr::symbol("curv_circ_t");
    /// let c = ParametricCurve::new(
    ///     vec![
    ///         Expr::func(FuncId::Cos, vec![t.clone()]),
    ///         Expr::func(FuncId::Sin, vec![t.clone()]),
    ///     ],
    ///     t_id,
    /// );
    /// let kappa = c.curvature();
    /// assert!(!kappa.is_zero());
    /// ```
    pub fn curvature(&self) -> Arc<Expr> {
        let n = self.components.len();
        assert!(
            n == 2 || n == 3,
            "curvature requires a 2-D or 3-D curve (got {} components)",
            n
        );
        let r_p = self.tangent_vector();
        let r_pp = second_derivatives(&self.components, self.param);

        if n == 2 {
            // κ = |x'y'' − y'x''| / (x'² + y'²)^(3/2)
            let det = normalize::sub(
                normalize::mul(r_p[0].clone(), r_pp[1].clone()),
                normalize::mul(r_p[1].clone(), r_pp[0].clone()),
            );
            // Avoid Func(Abs, [0]) — normalize::div(0, …) already returns 0.
            let abs_det = if det.is_zero() {
                det
            } else {
                Expr::func(FuncId::Abs, vec![det])
            };
            let speed_sq = normalize::add(
                normalize::pow(r_p[0].clone(), Expr::int(2)),
                normalize::pow(r_p[1].clone(), Expr::int(2)),
            );
            let speed_cubed = normalize::pow(speed_sq, Expr::rational(3, 2));
            normalize::div(abs_det, speed_cubed)
        } else {
            // κ = |r' × r''| / |r'|³
            let cross = cross3(&r_p, &r_pp);
            let cross_mag = vec_magnitude(&cross);
            let speed_sq = dot3(&r_p, &r_p);
            let speed_cubed = normalize::pow(speed_sq, Expr::rational(3, 2));
            normalize::div(cross_mag, speed_cubed)
        }
    }

    /// Compute the arc-length integrand `ds/dt = sqrt(Σ (dcᵢ/dt)²)`.
    ///
    /// The arc length of the curve from `a` to `b` is `∫ₐᵇ arc_length_integrand() dt`.
    /// This method returns the integrand as a symbolic `Func(Sqrt, [inner])` expression.
    /// No trig identity folding (e.g. `sin²t + cos²t → 1`) is applied at this level;
    /// the result is the raw symbolic sum-of-squares under a `Sqrt` node.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use thales::numeric::{Expr, SymbolId, FuncId};
    /// use thales::geometry::parametric::ParametricCurve;
    ///
    /// // Unit circle: arc-length integrand = sqrt(sin²t + cos²t).
    /// let t_id = SymbolId::intern("ali_circ_t");
    /// let t = Expr::symbol("ali_circ_t");
    /// let c = ParametricCurve::new(
    ///     vec![
    ///         Expr::func(FuncId::Cos, vec![t.clone()]),
    ///         Expr::func(FuncId::Sin, vec![t.clone()]),
    ///     ],
    ///     t_id,
    /// );
    /// let ds = c.arc_length_integrand();
    /// assert!(!ds.is_zero());
    /// ```
    pub fn arc_length_integrand(&self) -> Arc<Expr> {
        let tangent = self.tangent_vector();
        let sum_sq = sum_of_squares(&tangent);
        Expr::func(FuncId::Sqrt, vec![sum_sq])
    }
}

// ── ParametricSurface ─────────────────────────────────────────────────────────

/// A parametric surface in 3-D given as a vector of 3 symbolic components.
///
/// The surface is parameterised by two [`SymbolId`]s stored in `params`:
/// `params[0]` = u, `params[1]` = v.
///
/// All components are [`Arc<Expr>`] expressions in `u` and `v`.
///
/// # Examples
///
/// Parametrised plane `r(u, v) = (u, v, 0)`:
///
/// ```rust
/// use std::sync::Arc;
/// use thales::numeric::{Expr, SymbolId};
/// use thales::geometry::parametric::ParametricSurface;
///
/// let u_id = SymbolId::intern("ps_plane_u");
/// let v_id = SymbolId::intern("ps_plane_v");
/// let u = Expr::symbol("ps_plane_u");
/// let v = Expr::symbol("ps_plane_v");
/// let plane = ParametricSurface {
///     components: vec![u, v, Expr::int(0)],
///     params: [u_id, v_id],
/// };
/// assert_eq!(plane.components.len(), 3);
/// ```
#[derive(Clone, Debug)]
pub struct ParametricSurface {
    /// Component expressions `(r_x, r_y, r_z)`.
    pub components: Vec<Arc<Expr>>,
    /// Parameter symbols `[u, v]`.
    pub params: [SymbolId; 2],
}

impl ParametricSurface {
    /// Construct a `ParametricSurface` from 3 component expressions and two parameters.
    ///
    /// Panics if `components.len() != 3`.
    pub fn new(components: Vec<Arc<Expr>>, params: [SymbolId; 2]) -> Self {
        assert_eq!(
            components.len(),
            3,
            "ParametricSurface requires exactly 3 components (got {})",
            components.len()
        );
        Self { components, params }
    }

    /// Compute the surface-area integrand `|∂r/∂u × ∂r/∂v|`.
    ///
    /// The surface area of the patch is `∬ surface_area_integrand() du dv`.
    /// This method returns the integrand (the magnitude of the cross product
    /// of the two partial-derivative vectors); numerical integration is the
    /// caller's responsibility.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use thales::numeric::{Expr, SymbolId};
    /// use thales::geometry::parametric::ParametricSurface;
    ///
    /// // Plane r = (u, v, 0): ∂r/∂u = (1,0,0), ∂r/∂v = (0,1,0)
    /// // cross = (0,0,1), magnitude = 1.
    /// let u_id = SymbolId::intern("sai_plane_u");
    /// let v_id = SymbolId::intern("sai_plane_v");
    /// let u = Expr::symbol("sai_plane_u");
    /// let v = Expr::symbol("sai_plane_v");
    /// let plane = ParametricSurface::new(vec![u, v, Expr::int(0)], [u_id, v_id]);
    /// let ds = plane.surface_area_integrand();
    /// assert!(!ds.is_zero());
    /// ```
    pub fn surface_area_integrand(&self) -> Arc<Expr> {
        debug_assert_eq!(self.components.len(), 3, "surface must be 3-D");
        let [u_id, v_id] = self.params;
        let dr_du: Vec<Arc<Expr>> = self.components.iter().map(|c| diff_arc(c, u_id)).collect();
        let dr_dv: Vec<Arc<Expr>> = self.components.iter().map(|c| diff_arc(c, v_id)).collect();
        let cross = cross3(&dr_du, &dr_dv);
        vec_magnitude(&cross)
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Differentiate each component of a vector twice w.r.t. `var`.
fn second_derivatives(components: &[Arc<Expr>], var: SymbolId) -> Vec<Arc<Expr>> {
    components
        .iter()
        .map(|c| {
            let first = diff_arc(c, var);
            diff_arc(&first, var)
        })
        .collect()
}

/// Symbolic sum of squares of a slice of expressions.
fn sum_of_squares(v: &[Arc<Expr>]) -> Arc<Expr> {
    v.iter().fold(Expr::int(0), |acc, c| {
        normalize::add(acc, normalize::pow(c.clone(), Expr::int(2)))
    })
}

/// Symbolic dot product of two 3-D vectors (also works for 2-D if lengths match).
fn dot3(a: &[Arc<Expr>], b: &[Arc<Expr>]) -> Arc<Expr> {
    debug_assert_eq!(a.len(), b.len(), "dot3: vectors must have equal length");
    a.iter().zip(b.iter()).fold(Expr::int(0), |acc, (ai, bi)| {
        normalize::add(acc, normalize::mul(ai.clone(), bi.clone()))
    })
}

/// Symbolic cross product of two 3-D vectors.
///
/// Returns `[a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]`.
fn cross3(a: &[Arc<Expr>], b: &[Arc<Expr>]) -> Vec<Arc<Expr>> {
    debug_assert_eq!(a.len(), 3);
    debug_assert_eq!(b.len(), 3);
    vec![
        normalize::sub(
            normalize::mul(a[1].clone(), b[2].clone()),
            normalize::mul(a[2].clone(), b[1].clone()),
        ),
        normalize::sub(
            normalize::mul(a[2].clone(), b[0].clone()),
            normalize::mul(a[0].clone(), b[2].clone()),
        ),
        normalize::sub(
            normalize::mul(a[0].clone(), b[1].clone()),
            normalize::mul(a[1].clone(), b[0].clone()),
        ),
    ]
}

/// Magnitude of a symbolic vector: `sqrt(Σ vᵢ²)`.
///
/// When the sum of squares reduces to an exact non-negative integer constant
/// that is a perfect square, the result is folded to that integer:
/// e.g. `sqrt(36) → 6`, `sqrt(1) → 1`.  For non-integer or non-perfect-square
/// arguments the result stays as `Func(Sqrt, [inner])`.
fn vec_magnitude(v: &[Arc<Expr>]) -> Arc<Expr> {
    let sq = sum_of_squares(v);
    // Constant-fold integer perfect squares at construction time.
    if let Expr::Integer(n) = sq.as_ref() {
        if let Some(val) = n.to_i64() {
            if val >= 0 {
                let isqrt = integer_sqrt(val);
                if isqrt * isqrt == val {
                    return Expr::int(isqrt);
                }
            }
        }
    }
    Expr::func(FuncId::Sqrt, vec![sq])
}

/// Integer square root (floor) via Newton's method.
fn integer_sqrt(n: i64) -> i64 {
    if n <= 0 {
        return 0;
    }
    let mut x = (n as f64).sqrt() as i64;
    // Ensure we land on the exact root (floating-point may drift by ±1).
    while x * x > n {
        x -= 1;
    }
    while (x + 1) * (x + 1) <= n {
        x += 1;
    }
    x
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

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build the unit-circle curve (cos t, sin t) with the given parameter name.
    fn unit_circle(t_name: &str) -> ParametricCurve {
        let t_id = id(t_name);
        let t = sym(t_name);
        ParametricCurve::new(
            vec![
                Expr::func(FuncId::Cos, vec![t.clone()]),
                Expr::func(FuncId::Sin, vec![t.clone()]),
            ],
            t_id,
        )
    }

    /// Build the helix (cos t, sin t, t) with the given parameter name.
    fn helix(t_name: &str) -> ParametricCurve {
        let t_id = id(t_name);
        let t = sym(t_name);
        ParametricCurve::new(
            vec![
                Expr::func(FuncId::Cos, vec![t.clone()]),
                Expr::func(FuncId::Sin, vec![t.clone()]),
                t.clone(),
            ],
            t_id,
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ParametricCurve::tangent_vector
    // ─────────────────────────────────────────────────────────────────────────

    /// Unit circle: tangent is (-sin t, cos t).
    /// Verify both components are non-zero and have correct function structure.
    #[test]
    fn test_tangent_vector_circle_2d() {
        let c = unit_circle("tv2d_t");
        let tv = c.tangent_vector();
        assert_eq!(tv.len(), 2, "tangent has 2 components for 2-D curve");
        // d/dt cos(t) = -sin(t)  →  Mul(-1, sin(t))  — non-zero
        assert!(!tv[0].is_zero(), "tx should be -sin(t)");
        // d/dt sin(t) = cos(t)  — non-zero
        assert!(!tv[1].is_zero(), "ty should be cos(t)");
        // cos(t) is Func(Cos, …) — check the positive component
        match tv[1].as_ref() {
            Expr::Func(FuncId::Cos, _) => {}
            other => panic!("tv[1] should be cos(t), got {other}"),
        }
    }

    /// 3D helix: tangent is (-sin t, cos t, 1).
    #[test]
    fn test_tangent_vector_helix_3d() {
        let c = helix("tv3d_t");
        let tv = c.tangent_vector();
        assert_eq!(tv.len(), 3, "tangent has 3 components for 3-D curve");
        // d/dt cos(t) = -sin(t)
        assert!(!tv[0].is_zero(), "tx should be -sin(t)");
        // d/dt sin(t) = cos(t)
        assert!(!tv[1].is_zero(), "ty should be cos(t)");
        // d/dt t = 1
        assert!(tv[2].is_one(), "tz should be 1 for the helix");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ParametricCurve::normal_vector
    // ─────────────────────────────────────────────────────────────────────────

    /// 2D unit circle: normal is (sin t, -sin t) … actually N = (-T_y, T_x).
    /// T = (-sin t, cos t), so N = (-cos t, -sin t).  Both non-zero.
    #[test]
    fn test_normal_vector_circle_2d() {
        let c = unit_circle("nv2d_t");
        let nv = c.normal_vector();
        assert_eq!(nv.len(), 2);
        // N[0] = -T_y = -cos(t) ≠ 0
        assert!(!nv[0].is_zero(), "N[0] should be -cos(t)");
        // N[1] = T_x = -sin(t) ≠ 0
        assert!(!nv[1].is_zero(), "N[1] should be -sin(t)");
    }

    /// Straight line r(t) = (t, 0): tangent = (1, 0), normal = (0, 1).
    #[test]
    fn test_normal_vector_line_2d() {
        let t_id = id("nv2d_line_t");
        let t = sym("nv2d_line_t");
        let line = ParametricCurve::new(vec![t.clone(), Expr::int(0)], t_id);
        let nv = line.normal_vector();
        // T = (1, 0), N₂D = (-0, 1) = (0, 1)
        assert!(nv[0].is_zero(), "N[0] for horizontal line should be 0");
        assert!(nv[1].is_one(), "N[1] for horizontal line should be 1");
    }

    /// 3D helix normal is non-zero (it lies in the xy-plane by construction).
    #[test]
    fn test_normal_vector_helix_3d() {
        let c = helix("nv3d_t");
        let nv = c.normal_vector();
        assert_eq!(nv.len(), 3);
        // The helix normal is (-cos t, -sin t, 0) — both first components non-zero.
        assert!(!nv[0].is_zero(), "N[0] for helix should be non-zero");
        assert!(!nv[1].is_zero(), "N[1] for helix should be non-zero");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ParametricCurve::curvature
    // ─────────────────────────────────────────────────────────────────────────

    /// Unit circle curvature expression is non-zero (it evaluates to 1).
    #[test]
    fn test_curvature_circle_2d_nonzero() {
        let c = unit_circle("curv2d_t");
        let kappa = c.curvature();
        assert!(
            !kappa.is_zero(),
            "curvature of unit circle should be non-zero"
        );
    }

    /// Straight line curvature is zero: det = 1*0 - 0*0 = 0.
    #[test]
    fn test_curvature_straight_line_2d_zero() {
        let t_id = id("curv2d_line_t");
        let t = sym("curv2d_line_t");
        // r(t) = (t, t): tangent (1,1), second deriv (0,0) → det = 1*0 - 1*0 = 0
        let line = ParametricCurve::new(vec![t.clone(), t.clone()], t_id);
        let kappa = line.curvature();
        // numerator = |x'y'' - y'x''| = |1*0 - 1*0| = 0 → κ = 0 / denom = 0
        assert!(kappa.is_zero(), "straight line has zero curvature");
    }

    /// Helix curvature is non-zero (it evaluates to 1/2).
    #[test]
    fn test_curvature_helix_3d_nonzero() {
        let c = helix("curv3d_t");
        let kappa = c.curvature();
        assert!(!kappa.is_zero(), "helix curvature should be non-zero");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ParametricCurve::arc_length_integrand
    // ─────────────────────────────────────────────────────────────────────────

    /// Unit circle arc-length integrand is `sqrt(sin²t + cos²t)`.
    ///
    /// The normalizer does not apply the Pythagorean identity, so the result
    /// is a `Func(Sqrt, [inner])` where `inner` is an `Add` of trig squares.
    /// We verify the outer Sqrt wrapper and that the inner is non-zero.
    #[test]
    fn test_arc_length_integrand_circle_2d_is_sqrt_of_sum() {
        let c = unit_circle("ali2d_t");
        let ds = c.arc_length_integrand();
        // Result is sqrt(cos²t + sin²t) — structural check
        match ds.as_ref() {
            Expr::Func(FuncId::Sqrt, args) => {
                assert!(
                    !args[0].is_zero(),
                    "inner of arc-length sqrt should be non-zero, got {}",
                    args[0]
                );
            }
            other => panic!("expected Func(Sqrt, …), got {other}"),
        }
    }

    /// Helix arc-length integrand is `sqrt(1 + sin²t + cos²t)`.
    ///
    /// The normalizer does not reduce `sin²t + cos²t → 1`, so the inner
    /// argument is an `Add` node — not the integer 2.  We verify the `Sqrt`
    /// wrapper and that the inner expression is non-zero.
    #[test]
    fn test_arc_length_integrand_helix_is_sqrt_of_sum() {
        let c = helix("ali3d_t");
        let ds = c.arc_length_integrand();
        // Result is sqrt(1 + cos²t + sin²t) — structural check
        match ds.as_ref() {
            Expr::Func(FuncId::Sqrt, args) => {
                assert!(
                    !args[0].is_zero(),
                    "inner of helix arc-length sqrt should be non-zero, got {}",
                    args[0]
                );
            }
            other => panic!("expected Func(Sqrt, …), got {other}"),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ParametricSurface::surface_area_integrand
    // ─────────────────────────────────────────────────────────────────────────

    /// Plane r(u,v) = (u, v, 0): ∂r/∂u = (1,0,0), ∂r/∂v = (0,1,0).
    /// Cross product = (0,0,1), magnitude = sqrt(1) = 1.
    #[test]
    fn test_surface_area_integrand_plane_is_one() {
        let u_id = id("sai_plane_u");
        let v_id = id("sai_plane_v");
        let u = sym("sai_plane_u");
        let v = sym("sai_plane_v");
        let plane = ParametricSurface::new(vec![u, v, Expr::int(0)], [u_id, v_id]);
        let ds = plane.surface_area_integrand();
        assert!(
            ds.is_one(),
            "plane surface-area integrand should be 1, got {ds}"
        );
    }

    /// Sphere patch r(u,v) = (sin u cos v, sin u sin v, cos u): surface-area
    /// integrand is |sin u| (the standard result). We only verify non-zero.
    #[test]
    fn test_surface_area_integrand_sphere_nonzero() {
        let u_id = id("sai_sph_u");
        let v_id = id("sai_sph_v");
        let u = sym("sai_sph_u");
        let v = sym("sai_sph_v");
        // r = (sin(u)*cos(v), sin(u)*sin(v), cos(u))
        let sin_u = Expr::func(FuncId::Sin, vec![u.clone()]);
        let cos_u = Expr::func(FuncId::Cos, vec![u.clone()]);
        let cos_v = Expr::func(FuncId::Cos, vec![v.clone()]);
        let sin_v = Expr::func(FuncId::Sin, vec![v.clone()]);
        let rx = normalize::mul(sin_u.clone(), cos_v);
        let ry = normalize::mul(sin_u.clone(), sin_v);
        let rz = cos_u;
        let sphere = ParametricSurface::new(vec![rx, ry, rz], [u_id, v_id]);
        let ds = sphere.surface_area_integrand();
        assert!(
            !ds.is_zero(),
            "sphere surface-area integrand should be non-zero"
        );
    }

    /// Scaled plane r(u,v) = (2u, 3v, 0): cross = (0,0,6), magnitude = 6.
    #[test]
    fn test_surface_area_integrand_scaled_plane() {
        let u_id = id("sai_sc_u");
        let v_id = id("sai_sc_v");
        let u = sym("sai_sc_u");
        let v = sym("sai_sc_v");
        // r = (2u, 3v, 0)
        let rx = normalize::mul(Expr::int(2), u);
        let ry = normalize::mul(Expr::int(3), v);
        let scaled = ParametricSurface::new(vec![rx, ry, Expr::int(0)], [u_id, v_id]);
        let ds = scaled.surface_area_integrand();
        // ∂r/∂u = (2,0,0), ∂r/∂v = (0,3,0), cross = (0,0,6), |cross| = 6
        assert_eq!(
            *ds,
            Expr::Integer(SmallInt::from(6i64)),
            "scaled-plane integrand should be 6, got {ds}"
        );
    }
}
