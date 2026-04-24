//! Symbolic 3-D affine/linear transformations.
//!
//! All functions return [`MatrixExpr`] matrices whose entries are [`Arc<Expr>`]
//! values. No numeric evaluation occurs here; callers use
//! [`crate::numeric::simplify`] or [`crate::numeric::evaluation::evaluate`] to
//! reduce results to numbers.
//!
//! # Transform types
//!
//! | Function | Matrix | Formula |
//! |---|---|---|
//! | [`rotation_3d`] | 3×3 | Rodrigues' rotation formula |
//! | [`reflection_3d`] | 3×3 | Householder: I − 2·n·nᵀ/(n·n) |
//! | [`scale_3d`] | 3×3 | Diagonal scale matrix |
//! | [`translation_3d`] | offset `Point3D` | Translation is a separate affine offset |
//! | [`apply_3d`] | — | Applies a 3×3 `MatrixExpr` to a `Point3D` |
//! | [`compose_3d`] | 3×3 | Matrix product `outer * inner` |
//!
//! # Translation note
//!
//! The existing `src/transforms/transform2d.rs` uses `nalgebra::Matrix3<f64>`
//! (numeric only), not symbolic `MatrixExpr`. It therefore does NOT establish a
//! symbolic homogeneous-4×4 convention. This module keeps 3×3 symbolic matrices
//! for linear transforms and returns a `Point3D` for translation, consistent
//! with the symbolic-only contract. Callers combine a rotation `R` (3×3) with a
//! translation offset `t` (Point3D) as `R·p + t` using `apply_3d` then manual
//! coordinate-wise addition.

use crate::matrix::MatrixExpr;
use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::normalize;
use std::sync::Arc;

use super::primitives::Point3D;

// ── rotation_3d ───────────────────────────────────────────────────────────────

/// 3-D rotation matrix about an arbitrary axis via Rodrigues' formula.
///
/// # Formula
///
/// Given an axis vector **k** = `(ax, ay, az)` (not necessarily unit length)
/// and angle `θ`, the unit axis is:
/// ```text
/// n = k / ‖k‖,   ‖k‖ = sqrt(ax² + ay² + az²)
/// ```
/// Let `c = cos(θ)`, `s = sin(θ)`, `K` be the cross-product matrix of `n`:
/// ```text
/// K = ┌             ┐
///     │  0   -nz   ny│
///     │  nz   0   -nx│
///     │ -ny   nx   0 │
///     └             ┘
/// ```
/// Rodrigues gives:
/// ```text
/// R = I + s·K + (1 − c)·K²
/// ```
/// which expands to:
/// ```text
/// R = ┌─────────────────────────────────────────────────────────────┐
///     │  c + nx²(1-c)         nx·ny(1-c) − nz·s    nx·nz(1-c) + ny·s │
///     │  ny·nx(1-c) + nz·s    c + ny²(1-c)          ny·nz(1-c) − nx·s │
///     │  nz·nx(1-c) − ny·s    nz·ny(1-c) + nx·s    c + nz²(1-c)      │
///     └─────────────────────────────────────────────────────────────┘
/// ```
///
/// # Arguments
///
/// * `axis` - Rotation axis (any non-zero length; normalised symbolically).
/// * `angle` - Rotation angle in radians as a symbolic expression.
///
/// # Examples
///
/// ```rust
/// use thales::geometry::{Point3D, transforms3d::rotation_3d};
/// use thales::numeric::expr::Expr;
/// use std::f64::consts::PI;
///
/// let z_axis = Point3D::new(Expr::int(0), Expr::int(0), Expr::int(1));
/// let angle  = Expr::float(PI / 2.0);
/// let r = rotation_3d(&z_axis, angle);
/// assert_eq!(r.rows(), 3);
/// assert_eq!(r.cols(), 3);
/// ```
pub fn rotation_3d(axis: &Point3D, angle: Arc<Expr>) -> MatrixExpr {
    let ax = axis.x.clone();
    let ay = axis.y.clone();
    let az = axis.z.clone();

    // ‖k‖ = sqrt(ax² + ay² + az²)
    let ax2 = normalize::pow(ax.clone(), Expr::int(2));
    let ay2 = normalize::pow(ay.clone(), Expr::int(2));
    let az2 = normalize::pow(az.clone(), Expr::int(2));
    let norm_sq = normalize::add(normalize::add(ax2.clone(), ay2.clone()), az2.clone());
    let norm = Expr::func(FuncId::Sqrt, vec![norm_sq]);

    // Unit axis components
    let nx = normalize::div(ax.clone(), norm.clone());
    let ny = normalize::div(ay.clone(), norm.clone());
    let nz = normalize::div(az.clone(), norm.clone());

    let c = Expr::func(FuncId::Cos, vec![angle.clone()]);
    let s = Expr::func(FuncId::Sin, vec![angle]);
    let one_minus_c = normalize::sub(Expr::int(1), c.clone());

    // nx², ny², nz²
    let nx2 = normalize::pow(nx.clone(), Expr::int(2));
    let ny2 = normalize::pow(ny.clone(), Expr::int(2));
    let nz2 = normalize::pow(nz.clone(), Expr::int(2));

    // products
    let nxny = normalize::mul(nx.clone(), ny.clone());
    let nxnz = normalize::mul(nx.clone(), nz.clone());
    let nynz = normalize::mul(ny.clone(), nz.clone());

    // R[0][0] = c + nx²(1-c)
    let r00 = normalize::add(c.clone(), normalize::mul(nx2, one_minus_c.clone()));
    // R[0][1] = nx·ny(1-c) − nz·s
    let r01 = normalize::sub(
        normalize::mul(nxny.clone(), one_minus_c.clone()),
        normalize::mul(nz.clone(), s.clone()),
    );
    // R[0][2] = nx·nz(1-c) + ny·s
    let r02 = normalize::add(
        normalize::mul(nxnz.clone(), one_minus_c.clone()),
        normalize::mul(ny.clone(), s.clone()),
    );
    // R[1][0] = ny·nx(1-c) + nz·s
    let r10 = normalize::add(
        normalize::mul(nxny, one_minus_c.clone()),
        normalize::mul(nz.clone(), s.clone()),
    );
    // R[1][1] = c + ny²(1-c)
    let r11 = normalize::add(c.clone(), normalize::mul(ny2, one_minus_c.clone()));
    // R[1][2] = ny·nz(1-c) − nx·s
    let r12 = normalize::sub(
        normalize::mul(nynz.clone(), one_minus_c.clone()),
        normalize::mul(nx.clone(), s.clone()),
    );
    // R[2][0] = nz·nx(1-c) − ny·s
    let r20 = normalize::sub(
        normalize::mul(nxnz, one_minus_c.clone()),
        normalize::mul(ny.clone(), s.clone()),
    );
    // R[2][1] = nz·ny(1-c) + nx·s
    let r21 = normalize::add(
        normalize::mul(nynz, one_minus_c.clone()),
        normalize::mul(nx.clone(), s.clone()),
    );
    // R[2][2] = c + nz²(1-c)
    let r22 = normalize::add(c, normalize::mul(nz2, one_minus_c));

    MatrixExpr::from_expr_elements(vec![
        vec![r00, r01, r02],
        vec![r10, r11, r12],
        vec![r20, r21, r22],
    ])
    .expect("3x3 rotation_3d matrix is always valid")
}

// ── reflection_3d ─────────────────────────────────────────────────────────────

/// 3-D Householder reflection in the plane with the given normal.
///
/// # Formula
///
/// For a plane with (possibly un-normalised) normal **n** = `(nx, ny, nz)`:
/// ```text
/// R = I − 2·(n·nᵀ) / (nᵀ·n)
/// ```
/// which gives:
/// ```text
/// R[i][j] = δᵢⱼ − 2·nᵢ·nⱼ / ‖n‖²
/// ```
///
/// # Arguments
///
/// * `plane_normal` - Normal vector of the reflection plane (any length).
///
/// # Examples
///
/// ```rust
/// use thales::geometry::{Point3D, transforms3d::{reflection_3d, apply_3d}};
/// use thales::numeric::{expr::Expr, simplify};
///
/// // Reflect in xy-plane (normal = (0,0,1)): z flips sign.
/// let n = Point3D::new(Expr::int(0), Expr::int(0), Expr::int(1));
/// let r = reflection_3d(&n);
/// let p = Point3D::new(Expr::int(1), Expr::int(2), Expr::int(3));
/// let reflected = apply_3d(&r, &p);
/// let rx = simplify(&reflected.x);
/// let ry = simplify(&reflected.y);
/// let rz = simplify(&reflected.z);
/// assert_eq!(rx, Expr::int(1));
/// assert_eq!(ry, Expr::int(2));
/// assert_eq!(rz, Expr::int(-3));
/// ```
pub fn reflection_3d(plane_normal: &Point3D) -> MatrixExpr {
    let nx = plane_normal.x.clone();
    let ny = plane_normal.y.clone();
    let nz = plane_normal.z.clone();

    // ‖n‖² = nx² + ny² + nz²
    let nx2 = normalize::pow(nx.clone(), Expr::int(2));
    let ny2 = normalize::pow(ny.clone(), Expr::int(2));
    let nz2 = normalize::pow(nz.clone(), Expr::int(2));
    let norm_sq = normalize::add(normalize::add(nx2.clone(), ny2.clone()), nz2.clone());

    // 2 * nᵢ * nⱼ / ‖n‖²
    let two = Expr::int(2);

    let factor = |ni: Arc<Expr>, nj: Arc<Expr>| -> Arc<Expr> {
        normalize::div(
            normalize::mul(normalize::mul(two.clone(), ni), nj),
            norm_sq.clone(),
        )
    };

    // Diagonal: 1 − 2·nᵢ² / ‖n‖²
    let r00 = normalize::sub(Expr::int(1), factor(nx.clone(), nx.clone()));
    let r11 = normalize::sub(Expr::int(1), factor(ny.clone(), ny.clone()));
    let r22 = normalize::sub(Expr::int(1), factor(nz.clone(), nz.clone()));

    // Off-diagonal: − 2·nᵢ·nⱼ / ‖n‖²
    let r01 = normalize::neg(factor(nx.clone(), ny.clone()));
    let r02 = normalize::neg(factor(nx.clone(), nz.clone()));
    let r10 = normalize::neg(factor(ny.clone(), nx.clone()));
    let r12 = normalize::neg(factor(ny.clone(), nz.clone()));
    let r20 = normalize::neg(factor(nz.clone(), nx.clone()));
    let r21 = normalize::neg(factor(nz.clone(), ny.clone()));

    MatrixExpr::from_expr_elements(vec![
        vec![r00, r01, r02],
        vec![r10, r11, r12],
        vec![r20, r21, r22],
    ])
    .expect("3x3 reflection_3d matrix is always valid")
}

// ── scale_3d ──────────────────────────────────────────────────────────────────

/// Diagonal 3×3 scale matrix.
///
/// # Formula
///
/// ```text
/// S = ┌         ┐
///     │ sx  0  0│
///     │ 0  sy  0│
///     │ 0   0 sz│
///     └         ┘
/// ```
///
/// Applying `S` to `(x, y, z)` yields `(sx·x, sy·y, sz·z)`.
///
/// # Arguments
///
/// * `sx` - Scale factor for x.
/// * `sy` - Scale factor for y.
/// * `sz` - Scale factor for z.
///
/// # Examples
///
/// ```rust
/// use thales::geometry::{Point3D, transforms3d::{scale_3d, apply_3d}};
/// use thales::numeric::{expr::Expr, simplify};
///
/// let s = scale_3d(Expr::int(2), Expr::int(3), Expr::int(4));
/// let p = Point3D::new(Expr::int(1), Expr::int(1), Expr::int(1));
/// let sp = apply_3d(&s, &p);
/// assert_eq!(simplify(&sp.x), Expr::int(2));
/// assert_eq!(simplify(&sp.y), Expr::int(3));
/// assert_eq!(simplify(&sp.z), Expr::int(4));
/// ```
pub fn scale_3d(sx: Arc<Expr>, sy: Arc<Expr>, sz: Arc<Expr>) -> MatrixExpr {
    MatrixExpr::diagonal(vec![sx, sy, sz])
}

// ── translation_3d ────────────────────────────────────────────────────────────

/// Translation offset as a symbolic 3-D point.
///
/// # Design note
///
/// This module uses **3×3 symbolic matrices** for linear transforms. The
/// existing numeric `transform2d.rs` uses `nalgebra::Matrix3<f64>` (not
/// `MatrixExpr`), so there is no prior symbolic 4×4 homogeneous convention to
/// follow. Translation is therefore a separate affine offset expressed as a
/// `Point3D`. To apply a full affine transform `T = (R, t)` to a point `p`:
/// ```text
/// p' = apply_3d(&R, &p) + t
/// ```
/// where `+` is coordinate-wise `normalize::add` on each component.
///
/// # Arguments
///
/// * `tx` - Translation in x.
/// * `ty` - Translation in y.
/// * `tz` - Translation in z.
pub fn translation_3d(tx: Arc<Expr>, ty: Arc<Expr>, tz: Arc<Expr>) -> Point3D {
    Point3D::new(tx, ty, tz)
}

// ── apply_3d ──────────────────────────────────────────────────────────────────

/// Apply a 3×3 symbolic matrix to a `Point3D`.
///
/// Computes `M · p` as standard matrix-vector product:
/// ```text
/// result.x = M[0][0]·p.x + M[0][1]·p.y + M[0][2]·p.z
/// result.y = M[1][0]·p.x + M[1][1]·p.y + M[1][2]·p.z
/// result.z = M[2][0]·p.x + M[2][1]·p.y + M[2][2]·p.z
/// ```
///
/// # Panics
///
/// Panics if `transform` is not a 3×3 matrix.
///
/// # Arguments
///
/// * `transform` - A 3×3 [`MatrixExpr`].
/// * `point` - The point to transform.
///
/// # Examples
///
/// ```rust
/// use thales::geometry::{Point3D, transforms3d::{scale_3d, apply_3d}};
/// use thales::numeric::{expr::Expr, simplify};
///
/// let s = scale_3d(Expr::int(2), Expr::int(3), Expr::int(4));
/// let p = Point3D::new(Expr::int(1), Expr::int(1), Expr::int(1));
/// let q = apply_3d(&s, &p);
/// assert_eq!(simplify(&q.x), Expr::int(2));
/// ```
pub fn apply_3d(transform: &MatrixExpr, point: &Point3D) -> Point3D {
    assert_eq!(
        transform.dimensions(),
        (3, 3),
        "apply_3d requires a 3x3 matrix"
    );

    let coords = [point.x.clone(), point.y.clone(), point.z.clone()];
    let result: Vec<Arc<Expr>> = (0..3)
        .map(|row| {
            let terms: Vec<Arc<Expr>> = (0..3)
                .map(|col| {
                    normalize::mul(
                        transform.get(row, col).expect("3x3 index valid").clone(),
                        coords[col].clone(),
                    )
                })
                .collect();
            // Sum the three terms
            normalize::add(
                normalize::add(terms[0].clone(), terms[1].clone()),
                terms[2].clone(),
            )
        })
        .collect();

    Point3D::new(result[0].clone(), result[1].clone(), result[2].clone())
}

// ── compose_3d ────────────────────────────────────────────────────────────────

/// Compose two 3×3 transforms: `outer ∘ inner`.
///
/// Applies `inner` first, then `outer`. Mathematically this is the matrix
/// product `outer * inner`. Because matrix multiplication is non-commutative,
/// `compose_3d(R, S) ≠ compose_3d(S, R)` in general.
///
/// # Panics
///
/// Panics if either argument is not a 3×3 matrix.
///
/// # Arguments
///
/// * `outer` - The transform applied second.
/// * `inner` - The transform applied first.
///
/// # Examples
///
/// ```rust
/// use thales::geometry::transforms3d::{rotation_3d, scale_3d, compose_3d};
/// use thales::geometry::Point3D;
/// use thales::numeric::expr::Expr;
/// use std::f64::consts::PI;
///
/// let z = Point3D::new(Expr::int(0), Expr::int(0), Expr::int(1));
/// let r = rotation_3d(&z, Expr::float(PI / 2.0));
/// let s = scale_3d(Expr::int(2), Expr::int(2), Expr::int(2));
/// let rs = compose_3d(&r, &s); // rotate after scale
/// let sr = compose_3d(&s, &r); // scale after rotate
/// assert_ne!(rs, sr);          // non-commutative in general
/// ```
pub fn compose_3d(outer: &MatrixExpr, inner: &MatrixExpr) -> MatrixExpr {
    assert_eq!(outer.dimensions(), (3, 3), "compose_3d: outer must be 3x3");
    assert_eq!(inner.dimensions(), (3, 3), "compose_3d: inner must be 3x3");
    outer
        .mul(inner)
        .expect("3x3 * 3x3 multiplication is always compatible")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::evaluation::evaluate;
    use crate::numeric::simplify;
    use std::collections::HashMap;
    use std::f64::consts::PI;

    // Helper: evaluate a symbolic Arc<Expr> with no variable bindings.
    fn eval(e: &Arc<Expr>) -> f64 {
        let env: HashMap<_, f64> = HashMap::new();
        let simplified = simplify(e);
        evaluate(&simplified, &env).expect("expression must be fully numeric")
    }

    // ── rotation_3d ──────────────────────────────────────────────────────────

    /// Rotation about z-axis by π/2 applied to (1,0,0) gives (0,1,0).
    ///
    /// R_z(π/2) = [[0,-1,0],[1,0,0],[0,0,1]]
    /// R_z(π/2) · (1,0,0) = (0·1+(-1)·0+0·0, 1·1+0·0+0·0, 0) = (0, 1, 0)
    #[test]
    fn test_rotation_3d_z_axis_quarter_turn() {
        let z_axis = Point3D::new(Expr::int(0), Expr::int(0), Expr::int(1));
        let angle = Expr::float(PI / 2.0);
        let r = rotation_3d(&z_axis, angle);
        let p = Point3D::new(Expr::int(1), Expr::int(0), Expr::int(0));
        let q = apply_3d(&r, &p);

        let qx = eval(&q.x);
        let qy = eval(&q.y);
        let qz = eval(&q.z);

        assert!((qx - 0.0).abs() < 1e-10, "x should be 0, got {qx}");
        assert!((qy - 1.0).abs() < 1e-10, "y should be 1, got {qy}");
        assert!((qz - 0.0).abs() < 1e-10, "z should be 0, got {qz}");
    }

    /// Rotation about x-axis by π/2 applied to (0,1,0) gives (0,0,1).
    #[test]
    fn test_rotation_3d_x_axis_quarter_turn() {
        let x_axis = Point3D::new(Expr::int(1), Expr::int(0), Expr::int(0));
        let angle = Expr::float(PI / 2.0);
        let r = rotation_3d(&x_axis, angle);
        let p = Point3D::new(Expr::int(0), Expr::int(1), Expr::int(0));
        let q = apply_3d(&r, &p);

        let qx = eval(&q.x);
        let qy = eval(&q.y);
        let qz = eval(&q.z);

        assert!((qx - 0.0).abs() < 1e-10, "x should be 0, got {qx}");
        assert!((qy - 0.0).abs() < 1e-10, "y should be 0, got {qy}");
        assert!((qz - 1.0).abs() < 1e-10, "z should be 1, got {qz}");
    }

    // ── reflection_3d ────────────────────────────────────────────────────────

    /// Reflection in xy-plane (normal=(0,0,1)) applied to (1,2,3) gives (1,2,-3).
    #[test]
    fn test_reflection_3d_xy_plane() {
        let n = Point3D::new(Expr::int(0), Expr::int(0), Expr::int(1));
        let r = reflection_3d(&n);
        let p = Point3D::new(Expr::int(1), Expr::int(2), Expr::int(3));
        let q = apply_3d(&r, &p);

        let qx = eval(&q.x);
        let qy = eval(&q.y);
        let qz = eval(&q.z);

        assert!((qx - 1.0).abs() < 1e-10, "x should be 1, got {qx}");
        assert!((qy - 2.0).abs() < 1e-10, "y should be 2, got {qy}");
        assert!((qz - (-3.0)).abs() < 1e-10, "z should be -3, got {qz}");
    }

    /// Reflection in yz-plane (normal=(1,0,0)) applied to (3,4,5) gives (-3,4,5).
    #[test]
    fn test_reflection_3d_yz_plane() {
        let n = Point3D::new(Expr::int(1), Expr::int(0), Expr::int(0));
        let r = reflection_3d(&n);
        let p = Point3D::new(Expr::int(3), Expr::int(4), Expr::int(5));
        let q = apply_3d(&r, &p);

        let qx = eval(&q.x);
        let qy = eval(&q.y);
        let qz = eval(&q.z);

        assert!((qx - (-3.0)).abs() < 1e-10, "x should be -3, got {qx}");
        assert!((qy - 4.0).abs() < 1e-10, "y should be 4, got {qy}");
        assert!((qz - 5.0).abs() < 1e-10, "z should be 5, got {qz}");
    }

    // ── scale_3d ─────────────────────────────────────────────────────────────

    /// scale_3d(2,3,4) applied to (1,1,1) gives (2,3,4).
    #[test]
    fn test_scale_3d_unit_point() {
        let s = scale_3d(Expr::int(2), Expr::int(3), Expr::int(4));
        let p = Point3D::new(Expr::int(1), Expr::int(1), Expr::int(1));
        let q = apply_3d(&s, &p);

        assert_eq!(simplify(&q.x), Expr::int(2));
        assert_eq!(simplify(&q.y), Expr::int(3));
        assert_eq!(simplify(&q.z), Expr::int(4));
    }

    /// scale_3d(a, b, c) applied to (x, y, z) gives (a·x, b·y, c·z) symbolically.
    #[test]
    fn test_scale_3d_symbolic() {
        let s = scale_3d(Expr::symbol("a"), Expr::symbol("b"), Expr::symbol("c"));
        let p = Point3D::new(Expr::symbol("x"), Expr::symbol("y"), Expr::symbol("z"));
        let q = apply_3d(&s, &p);

        // Each coordinate must simplify to symbol * symbol product (non-zero).
        assert!(
            !q.x.is_zero() && !q.y.is_zero() && !q.z.is_zero(),
            "symbolic scale output must not collapse to zero"
        );
    }

    // ── compose_3d ───────────────────────────────────────────────────────────

    /// compose_3d(R, S) ≠ compose_3d(S, R) for rotation and non-uniform scale.
    ///
    /// Uses rotation about z by π/4 and scale(2, 3, 1). These do not commute
    /// because the scale is anisotropic in xy.
    #[test]
    fn test_compose_3d_noncommutative() {
        let z = Point3D::new(Expr::int(0), Expr::int(0), Expr::int(1));
        let r = rotation_3d(&z, Expr::float(PI / 4.0));
        let s = scale_3d(Expr::int(2), Expr::int(3), Expr::int(1));
        let rs = compose_3d(&r, &s);
        let sr = compose_3d(&s, &r);
        // Apply both to (1, 0, 0) and compare.
        let p = Point3D::new(Expr::int(1), Expr::int(0), Expr::int(0));
        let q_rs = apply_3d(&rs, &p);
        let q_sr = apply_3d(&sr, &p);
        let y_rs = eval(&q_rs.y);
        let y_sr = eval(&q_sr.y);
        // For R·S: scale first → (2,0,0), then rotate → (2·cos π/4, 2·sin π/4, 0)
        // For S·R: rotate first → (cos π/4, sin π/4, 0), then scale → (2·cos π/4, 3·sin π/4, 0)
        // y_rs = 2·sin(π/4) ≈ 1.414, y_sr = 3·sin(π/4) ≈ 2.121
        assert!(
            (y_rs - y_sr).abs() > 1e-6,
            "R∘S and S∘R must differ: y_rs={y_rs}, y_sr={y_sr}"
        );
    }

    /// compose_3d of identity with any matrix returns that matrix (numeric check).
    #[test]
    fn test_compose_3d_with_identity() {
        let s = scale_3d(Expr::int(5), Expr::int(7), Expr::int(11));
        let id = MatrixExpr::identity(3);
        let composed = compose_3d(&id, &s);
        let p = Point3D::new(Expr::int(1), Expr::int(1), Expr::int(1));
        let q_composed = apply_3d(&composed, &p);
        let q_direct = apply_3d(&s, &p);
        assert_eq!(
            simplify(&q_composed.x),
            simplify(&q_direct.x),
            "I∘S should equal S"
        );
        assert_eq!(
            simplify(&q_composed.y),
            simplify(&q_direct.y),
            "I∘S should equal S"
        );
        assert_eq!(
            simplify(&q_composed.z),
            simplify(&q_direct.z),
            "I∘S should equal S"
        );
    }
}
