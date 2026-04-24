//! Symbolic geometric relations between primitive shapes.
//!
//! Provides distance, intersection, and tangent/normal-line computations
//! between the types defined in [`super::primitives`] and
//! [`super::parametric`].  All coordinates and parameters are
//! [`Arc<Expr>`] values; no numeric evaluation occurs inside this module.
//!
//! # Distance functions
//!
//! | Function | From → To |
//! |----------|-----------|
//! | [`dist_point2d_point2d`] | Point2D ↔ Point2D |
//! | [`dist_point3d_point3d`] | Point3D ↔ Point3D |
//! | [`dist_point2d_line2d`] | Point2D ↔ Line2D |
//! | [`dist_point3d_line3d`] | Point3D ↔ Line3D (represented as `Line3D`) |
//! | [`dist_point3d_plane3d`] | Point3D ↔ Plane3D |
//! | [`dist_point2d_circle2d`] | Point2D ↔ Circle2D (min surface distance) |
//! | [`dist_point3d_sphere3d`] | Point3D ↔ Sphere3D (min surface distance) |
//! | [`dist_line2d_line2d`] | Line2D ↔ Line2D (parallel case) |
//! | [`dist_line3d_line3d`] | Line3D ↔ Line3D (including skew) |
//! | [`dist_line3d_plane3d`] | Line3D ↔ Plane3D (parallel case) |
//! | [`dist_plane3d_plane3d`] | Plane3D ↔ Plane3D (parallel case) |
//!
//! # Intersection functions
//!
//! | Function | Result type |
//! |----------|-------------|
//! | [`intersect_line2d_line2d`] | [`LineLineIntersection2D`] |
//! | [`intersect_line3d_line3d`] | [`LineLineIntersection3D`] |
//! | [`intersect_line2d_circle2d`] | [`Vec<Point2D>`] |
//! | [`intersect_line3d_plane3d`] | [`LinePlaneIntersection`] |
//! | [`intersect_plane3d_plane3d`] | [`PlanePlaneIntersection`] |
//! | [`intersect_circle2d_circle2d`] | [`Vec<Point2D>`] |
//!
//! # Tangent / normal lines
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`tangent_to_circle2d`] | Tangent to circle at a surface point |
//! | [`normal_to_circle2d`] | Normal to circle at a surface point |
//! | [`tangent_to_parametric_curve`] | Tangent at parameter value `t₀` |
//! | [`normal_to_parametric_curve`] | Normal at parameter value `t₀` |

use crate::numeric::{expr::FuncId, normalize, substitute::substitute, Expr, SymbolId};
use std::sync::Arc;

use super::parametric::ParametricCurve;
use super::primitives::{Circle2D, Line2D, Plane3D, Point2D, Point3D, Sphere3D};

// ── 3-D line type ─────────────────────────────────────────────────────────────

/// A symbolic line in 3-D defined by a point and a direction vector.
///
/// Analogous to [`Line2D`] but in three dimensions.
#[derive(Clone, Debug, PartialEq)]
pub struct Line3D {
    /// A point that lies on the line.
    pub point: Point3D,
    /// Direction vector `(Δx, Δy, Δz)`.
    pub direction: Point3D,
}

impl Line3D {
    /// Construct a [`Line3D`] from a base point and a direction vector.
    pub fn new(point: Point3D, direction: Point3D) -> Self {
        Self { point, direction }
    }

    /// Construct a [`Line3D`] from two distinct points.
    pub fn from_two_points(p1: Point3D, p2: Point3D) -> Self {
        let dx = normalize::sub(p2.x.clone(), p1.x.clone());
        let dy = normalize::sub(p2.y.clone(), p1.y.clone());
        let dz = normalize::sub(p2.z.clone(), p1.z.clone());
        Self {
            point: p1,
            direction: Point3D::new(dx, dy, dz),
        }
    }
}

// ── Intersection result types ─────────────────────────────────────────────────

/// Result of a 2-D line–line intersection query.
#[derive(Clone, Debug, PartialEq)]
pub enum LineLineIntersection2D {
    /// Lines intersect at exactly one point.
    Point(Point2D),
    /// Lines are parallel and distinct — no intersection.
    Parallel,
    /// Lines are coincident — every point is an intersection.
    Coincident,
}

/// Result of a 3-D line–line intersection query.
#[derive(Clone, Debug, PartialEq)]
pub enum LineLineIntersection3D {
    /// Lines intersect at exactly one point.
    Point(Point3D),
    /// Lines are parallel and distinct — no intersection.
    Parallel,
    /// Lines are coincident — every point is an intersection.
    Coincident,
    /// Lines are skew (not coplanar, no common point).
    Skew,
}

/// Result of a line–plane intersection query.
#[derive(Clone, Debug, PartialEq)]
pub enum LinePlaneIntersection {
    /// Line intersects the plane at exactly one point.
    Point(Point3D),
    /// Line is parallel to (and not on) the plane — no intersection.
    Parallel,
    /// Line lies entirely in the plane.
    InPlane,
}

/// Result of a plane–plane intersection query.
#[derive(Clone, Debug, PartialEq)]
pub enum PlanePlaneIntersection {
    /// Planes intersect along a line.
    Line(Line3D),
    /// Planes are parallel and distinct — no intersection.
    Parallel,
    /// Planes are coincident.
    Coincident,
}

// ── Distance helpers (internal) ───────────────────────────────────────────────

/// Symbolic Euclidean distance: `sqrt(Σ (aᵢ − bᵢ)²)`.
fn euclidean_distance(diffs: &[Arc<Expr>]) -> Arc<Expr> {
    let sum_sq = diffs.iter().fold(Expr::int(0), |acc, d| {
        normalize::add(acc, normalize::pow(d.clone(), Expr::int(2)))
    });
    // Constant-fold perfect integer squares.
    if let Expr::Integer(n) = sum_sq.as_ref() {
        if let Some(val) = n.to_i64() {
            if val >= 0 {
                let isqrt = integer_sqrt(val);
                if isqrt * isqrt == val {
                    return Expr::int(isqrt);
                }
            }
        }
    }
    Expr::func(FuncId::Sqrt, vec![sum_sq])
}

/// Integer square-root (floor), with upward correction.
fn integer_sqrt(n: i64) -> i64 {
    if n <= 0 {
        return 0;
    }
    let mut x = (n as f64).sqrt() as i64;
    while x * x > n {
        x -= 1;
    }
    while (x + 1) * (x + 1) <= n {
        x += 1;
    }
    x
}

/// Dot product of two parallel slices of expressions (arbitrary length).
fn dot(a: &[Arc<Expr>], b: &[Arc<Expr>]) -> Arc<Expr> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).fold(Expr::int(0), |acc, (ai, bi)| {
        normalize::add(acc, normalize::mul(ai.clone(), bi.clone()))
    })
}

/// Symbolic cross product of two 3-D vectors.
fn cross3(a: &[Arc<Expr>], b: &[Arc<Expr>]) -> [Arc<Expr>; 3] {
    debug_assert_eq!(a.len(), 3);
    debug_assert_eq!(b.len(), 3);
    [
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

/// Symbolic magnitude of a 3-D vector: `sqrt(Σ vᵢ²)`.
fn magnitude3(v: &[Arc<Expr>; 3]) -> Arc<Expr> {
    euclidean_distance(&[v[0].clone(), v[1].clone(), v[2].clone()])
}

/// Symbolic magnitude of a 2-D vector: `sqrt(dx² + dy²)`.
fn magnitude2(dx: Arc<Expr>, dy: Arc<Expr>) -> Arc<Expr> {
    euclidean_distance(&[dx, dy])
}

// ── Distance: point ↔ point ───────────────────────────────────────────────────

/// Symbolic Euclidean distance between two 2-D points.
///
/// `d = sqrt((x₂ − x₁)² + (y₂ − y₁)²)`
pub fn dist_point2d_point2d(a: &Point2D, b: &Point2D) -> Arc<Expr> {
    let dx = normalize::sub(b.x.clone(), a.x.clone());
    let dy = normalize::sub(b.y.clone(), a.y.clone());
    euclidean_distance(&[dx, dy])
}

/// Symbolic Euclidean distance between two 3-D points.
///
/// `d = sqrt((x₂ − x₁)² + (y₂ − y₁)² + (z₂ − z₁)²)`
pub fn dist_point3d_point3d(a: &Point3D, b: &Point3D) -> Arc<Expr> {
    let dx = normalize::sub(b.x.clone(), a.x.clone());
    let dy = normalize::sub(b.y.clone(), a.y.clone());
    let dz = normalize::sub(b.z.clone(), a.z.clone());
    euclidean_distance(&[dx, dy, dz])
}

// ── Distance: point ↔ line ────────────────────────────────────────────────────

/// Symbolic distance from a 2-D point to a 2-D line.
///
/// Uses the formula `|d × (p − q)| / |d|` where `d` is the direction vector
/// and `q` is any point on the line.
///
/// For a line with direction `(dx, dy)` through `(qx, qy)` and point
/// `(px, py)`:
///
/// ```text
/// distance = |dx·(py − qy) − dy·(px − qx)| / sqrt(dx² + dy²)
/// ```
pub fn dist_point2d_line2d(p: &Point2D, line: &Line2D) -> Arc<Expr> {
    let dx = &line.direction.x;
    let dy = &line.direction.y;
    let qx = &line.point.x;
    let qy = &line.point.y;

    // cross magnitude: |dx*(py - qy) - dy*(px - qx)|
    let py_minus_qy = normalize::sub(p.y.clone(), qy.clone());
    let px_minus_qx = normalize::sub(p.x.clone(), qx.clone());
    let cross = normalize::sub(
        normalize::mul(dx.clone(), py_minus_qy),
        normalize::mul(dy.clone(), px_minus_qx),
    );
    let abs_cross = if cross.is_zero() {
        cross
    } else {
        Expr::func(FuncId::Abs, vec![cross])
    };
    let dir_mag = magnitude2(dx.clone(), dy.clone());
    normalize::div(abs_cross, dir_mag)
}

/// Symbolic distance from a 3-D point to a 3-D line.
///
/// Uses the formula `|d × (p − q)| / |d|` where `d` is the direction vector
/// and `q` is any point on the line, with `×` the 3-D cross product.
pub fn dist_point3d_line3d(p: &Point3D, line: &Line3D) -> Arc<Expr> {
    let d = [
        line.direction.x.clone(),
        line.direction.y.clone(),
        line.direction.z.clone(),
    ];
    let diff = [
        normalize::sub(p.x.clone(), line.point.x.clone()),
        normalize::sub(p.y.clone(), line.point.y.clone()),
        normalize::sub(p.z.clone(), line.point.z.clone()),
    ];
    let cross = cross3(&d, &diff);
    let cross_mag = magnitude3(&cross);
    let dir_mag = magnitude3(&d);
    normalize::div(cross_mag, dir_mag)
}

// ── Distance: point ↔ plane ───────────────────────────────────────────────────

/// Symbolic distance from a 3-D point to a plane.
///
/// For plane with normal `(A, B, C)` through `(x₀, y₀, z₀)` and point
/// `(px, py, pz)`:
///
/// ```text
/// distance = |A(px − x₀) + B(py − y₀) + C(pz − z₀)| / sqrt(A² + B² + C²)
/// ```
pub fn dist_point3d_plane3d(p: &Point3D, plane: &Plane3D) -> Arc<Expr> {
    let n = &plane.normal;
    let q = &plane.point;

    let ax = normalize::mul(n.x.clone(), normalize::sub(p.x.clone(), q.x.clone()));
    let by = normalize::mul(n.y.clone(), normalize::sub(p.y.clone(), q.y.clone()));
    let cz = normalize::mul(n.z.clone(), normalize::sub(p.z.clone(), q.z.clone()));
    let numerator = normalize::add(normalize::add(ax, by), cz);
    let abs_num = if numerator.is_zero() {
        numerator
    } else {
        Expr::func(FuncId::Abs, vec![numerator])
    };
    let normal_mag = magnitude3(&[n.x.clone(), n.y.clone(), n.z.clone()]);
    normalize::div(abs_num, normal_mag)
}

// ── Distance: point ↔ circle / sphere ─────────────────────────────────────────

/// Symbolic minimum distance from a 2-D point to the circle boundary.
///
/// `d = |dist(p, center) − radius|`
pub fn dist_point2d_circle2d(p: &Point2D, circle: &Circle2D) -> Arc<Expr> {
    let center_dist = dist_point2d_point2d(p, &circle.center);
    let diff = normalize::sub(center_dist, circle.radius.clone());
    Expr::func(FuncId::Abs, vec![diff])
}

/// Symbolic minimum distance from a 3-D point to the sphere surface.
///
/// `d = |dist(p, center) − radius|`
pub fn dist_point3d_sphere3d(p: &Point3D, sphere: &Sphere3D) -> Arc<Expr> {
    let center_dist = dist_point3d_point3d(p, &sphere.center);
    let diff = normalize::sub(center_dist, sphere.radius.clone());
    Expr::func(FuncId::Abs, vec![diff])
}

// ── Distance: line ↔ line ─────────────────────────────────────────────────────

/// Symbolic distance between two 2-D lines (parallel case only).
///
/// If the lines are not parallel, returns `None` (distance is 0 at
/// intersection; the caller should call [`intersect_line2d_line2d`] first).
///
/// Uses the perpendicular-distance formula: pick any point from one line
/// and compute its distance to the other line.
pub fn dist_line2d_line2d(a: &Line2D, b: &Line2D) -> Option<Arc<Expr>> {
    // Check parallel: cross of directions is zero
    let cross = normalize::sub(
        normalize::mul(a.direction.x.clone(), b.direction.y.clone()),
        normalize::mul(a.direction.y.clone(), b.direction.x.clone()),
    );
    if cross.is_zero() {
        // Parallel — distance = dist from b.point to line a
        Some(dist_point2d_line2d(&b.point, a))
    } else {
        None
    }
}

/// Symbolic distance between two 3-D lines (parallel or skew).
///
/// - **Parallel lines**: distance = perpendicular distance from a point on
///   one line to the other.
/// - **Skew lines**: uses the formula `|( b.point − a.point ) · (d_a × d_b)| / |d_a × d_b|`.
/// - **Intersecting / coincident lines**: distance = 0.
pub fn dist_line3d_line3d(a: &Line3D, b: &Line3D) -> Arc<Expr> {
    let da = [
        a.direction.x.clone(),
        a.direction.y.clone(),
        a.direction.z.clone(),
    ];
    let db = [
        b.direction.x.clone(),
        b.direction.y.clone(),
        b.direction.z.clone(),
    ];

    let cross = cross3(&da, &db);
    let cross_mag = magnitude3(&cross);

    if cross_mag.is_zero() {
        // Parallel or coincident — use point-to-line distance
        return dist_point3d_line3d(&b.point, a);
    }

    // Skew / intersecting: |diff · cross| / |cross|
    let diff = [
        normalize::sub(b.point.x.clone(), a.point.x.clone()),
        normalize::sub(b.point.y.clone(), a.point.y.clone()),
        normalize::sub(b.point.z.clone(), a.point.z.clone()),
    ];
    let dot_val = dot(&diff, &cross);
    let abs_dot = if dot_val.is_zero() {
        dot_val
    } else {
        Expr::func(FuncId::Abs, vec![dot_val])
    };
    normalize::div(abs_dot, cross_mag)
}

// ── Distance: line ↔ plane ────────────────────────────────────────────────────

/// Symbolic distance from a 3-D line to a plane (parallel case only).
///
/// Returns `None` when the line is not parallel to the plane (the line
/// intersects the plane, making distance 0).  When parallel, returns the
/// distance from `line.point` to the plane.
pub fn dist_line3d_plane3d(line: &Line3D, plane: &Plane3D) -> Option<Arc<Expr>> {
    let n = &plane.normal;
    let d = &line.direction;
    // n · d = 0 means parallel
    let n_dot_d = normalize::add(
        normalize::add(
            normalize::mul(n.x.clone(), d.x.clone()),
            normalize::mul(n.y.clone(), d.y.clone()),
        ),
        normalize::mul(n.z.clone(), d.z.clone()),
    );
    if n_dot_d.is_zero() {
        Some(dist_point3d_plane3d(&line.point, plane))
    } else {
        None
    }
}

// ── Distance: plane ↔ plane ───────────────────────────────────────────────────

/// Symbolic distance between two 3-D planes (parallel case only).
///
/// Returns `None` when the planes are not parallel (they intersect, making
/// distance 0).  When parallel, returns the distance from `b.point` to
/// plane `a`.
pub fn dist_plane3d_plane3d(a: &Plane3D, b: &Plane3D) -> Option<Arc<Expr>> {
    // Two planes are parallel when their normals are proportional.
    // Check: n_a × n_b == 0
    let na = [a.normal.x.clone(), a.normal.y.clone(), a.normal.z.clone()];
    let nb = [b.normal.x.clone(), b.normal.y.clone(), b.normal.z.clone()];
    let cross = cross3(&na, &nb);
    let cross_mag = magnitude3(&cross);
    if cross_mag.is_zero() {
        Some(dist_point3d_plane3d(&b.point, a))
    } else {
        None
    }
}

// ── Intersection: line ↔ line (2-D) ──────────────────────────────────────────

/// Symbolic intersection of two 2-D lines.
///
/// Lines are given in parametric form `p + t·d`.  The system
/// `p₁ + t·d₁ = p₂ + s·d₂` is solved for `t` (and `s` implicitly).
///
/// Returns:
/// - [`LineLineIntersection2D::Point`] — unique intersection point.
/// - [`LineLineIntersection2D::Parallel`] — direction cross is zero, but offset
///   cross is non-zero.
/// - [`LineLineIntersection2D::Coincident`] — both crosses are zero.
pub fn intersect_line2d_line2d(a: &Line2D, b: &Line2D) -> LineLineIntersection2D {
    // Direction cross: d₁ × d₂ = d₁x·d₂y − d₁y·d₂x
    let denom = normalize::sub(
        normalize::mul(a.direction.x.clone(), b.direction.y.clone()),
        normalize::mul(a.direction.y.clone(), b.direction.x.clone()),
    );

    if denom.is_zero() {
        // Check if points are on the same line (coincident)
        // Vector from a.point to b.point
        let dpx = normalize::sub(b.point.x.clone(), a.point.x.clone());
        let dpy = normalize::sub(b.point.y.clone(), a.point.y.clone());
        // Cross of (b.point - a.point) with direction a
        let offset_cross = normalize::sub(
            normalize::mul(dpx, a.direction.y.clone()),
            normalize::mul(dpy, a.direction.x.clone()),
        );
        if offset_cross.is_zero() {
            LineLineIntersection2D::Coincident
        } else {
            LineLineIntersection2D::Parallel
        }
    } else {
        // t = ((b.point - a.point) × d₂) / (d₁ × d₂)
        // where × is the 2-D "cross" (scalar)
        let dpx = normalize::sub(b.point.x.clone(), a.point.x.clone());
        let dpy = normalize::sub(b.point.y.clone(), a.point.y.clone());
        let cross_dp_d2 = normalize::sub(
            normalize::mul(dpx, b.direction.y.clone()),
            normalize::mul(dpy, b.direction.x.clone()),
        );
        let t = normalize::div(cross_dp_d2, denom);

        // Intersection point = a.point + t * a.direction
        let ix = normalize::add(
            a.point.x.clone(),
            normalize::mul(t.clone(), a.direction.x.clone()),
        );
        let iy = normalize::add(a.point.y.clone(), normalize::mul(t, a.direction.y.clone()));
        LineLineIntersection2D::Point(Point2D::new(ix, iy))
    }
}

// ── Intersection: line ↔ line (3-D) ──────────────────────────────────────────

/// Symbolic intersection of two 3-D lines.
///
/// Solves the system in the x and y equations for parameters `t` and `s`,
/// then verifies consistency in z.
///
/// Returns [`LineLineIntersection3D`] with the four possible outcomes.
pub fn intersect_line3d_line3d(a: &Line3D, b: &Line3D) -> LineLineIntersection3D {
    let da = [
        a.direction.x.clone(),
        a.direction.y.clone(),
        a.direction.z.clone(),
    ];
    let db = [
        b.direction.x.clone(),
        b.direction.y.clone(),
        b.direction.z.clone(),
    ];

    let cross_ab = cross3(&da, &db);
    let cross_mag = magnitude3(&cross_ab);

    if cross_mag.is_zero() {
        // Parallel or coincident
        let diff = [
            normalize::sub(b.point.x.clone(), a.point.x.clone()),
            normalize::sub(b.point.y.clone(), a.point.y.clone()),
            normalize::sub(b.point.z.clone(), a.point.z.clone()),
        ];
        let diff_cross = cross3(&diff, &da);
        let diff_cross_mag = magnitude3(&diff_cross);
        if diff_cross_mag.is_zero() {
            LineLineIntersection3D::Coincident
        } else {
            LineLineIntersection3D::Parallel
        }
    } else {
        // Could intersect or be skew
        // Compute t = | diff · (da × db) | / | da × db |²  (scalar triple product)
        // Actually use 2-equation system: solve for t in x and y coordinates.
        // denom_xy = da.x * db.y - da.y * db.x
        let denom_xy = normalize::sub(
            normalize::mul(da[0].clone(), db[1].clone()),
            normalize::mul(da[1].clone(), db[0].clone()),
        );

        let diff_x = normalize::sub(b.point.x.clone(), a.point.x.clone());
        let diff_y = normalize::sub(b.point.y.clone(), a.point.y.clone());
        let diff_z = normalize::sub(b.point.z.clone(), a.point.z.clone());

        if !denom_xy.is_zero() {
            // Solve from x-y equations
            // t = (diff_x * db.y - diff_y * db.x) / denom_xy
            let t_num = normalize::sub(
                normalize::mul(diff_x.clone(), db[1].clone()),
                normalize::mul(diff_y.clone(), db[0].clone()),
            );
            let t = normalize::div(t_num, denom_xy.clone());

            // s = (diff_x * da.y - diff_y * da.x) / denom_xy
            let s_num = normalize::sub(
                normalize::mul(diff_x, da[1].clone()),
                normalize::mul(diff_y, da[0].clone()),
            );
            let s = normalize::div(s_num, denom_xy);

            // Check z consistency: a.z + t*da.z == b.z + s*db.z
            let lhs_z = normalize::add(a.point.z.clone(), normalize::mul(t.clone(), da[2].clone()));
            let rhs_z = normalize::add(b.point.z.clone(), normalize::mul(s, db[2].clone()));
            let z_residual = normalize::sub(lhs_z, rhs_z);

            if z_residual.is_zero() {
                let ix =
                    normalize::add(a.point.x.clone(), normalize::mul(t.clone(), da[0].clone()));
                let iy =
                    normalize::add(a.point.y.clone(), normalize::mul(t.clone(), da[1].clone()));
                let iz = normalize::add(a.point.z.clone(), normalize::mul(t, da[2].clone()));
                LineLineIntersection3D::Point(Point3D::new(ix, iy, iz))
            } else {
                LineLineIntersection3D::Skew
            }
        } else {
            // x-y system degenerate; try x-z
            let denom_xz = normalize::sub(
                normalize::mul(da[0].clone(), db[2].clone()),
                normalize::mul(da[2].clone(), db[0].clone()),
            );
            if !denom_xz.is_zero() {
                let t_num = normalize::sub(
                    normalize::mul(diff_x.clone(), db[2].clone()),
                    normalize::mul(diff_z.clone(), db[0].clone()),
                );
                let t = normalize::div(t_num, denom_xz.clone());
                let s_num = normalize::sub(
                    normalize::mul(diff_x, da[2].clone()),
                    normalize::mul(diff_z, da[0].clone()),
                );
                let s = normalize::div(s_num, denom_xz);

                let lhs_y =
                    normalize::add(a.point.y.clone(), normalize::mul(t.clone(), da[1].clone()));
                let rhs_y = normalize::add(b.point.y.clone(), normalize::mul(s, db[1].clone()));
                let y_residual = normalize::sub(lhs_y, rhs_y);

                if y_residual.is_zero() {
                    let ix =
                        normalize::add(a.point.x.clone(), normalize::mul(t.clone(), da[0].clone()));
                    let iy =
                        normalize::add(a.point.y.clone(), normalize::mul(t.clone(), da[1].clone()));
                    let iz = normalize::add(a.point.z.clone(), normalize::mul(t, da[2].clone()));
                    LineLineIntersection3D::Point(Point3D::new(ix, iy, iz))
                } else {
                    LineLineIntersection3D::Skew
                }
            } else {
                // All 2×2 systems degenerate — lines are parallel (already
                // caught above when cross_mag.is_zero()); this branch is
                // unreachable for well-defined inputs.
                LineLineIntersection3D::Skew
            }
        }
    }
}

// ── Intersection: line ↔ circle (2-D) ────────────────────────────────────────

/// Symbolic intersection of a 2-D line with a 2-D circle.
///
/// Substitutes the parametric line `(qx + t·dx, qy + t·dy)` into the
/// circle equation `(x − cx)² + (y − cy)² = r²`, yielding a quadratic
/// in `t`.  Solves the quadratic symbolically via the discriminant formula.
///
/// Returns a `Vec<Point2D>` with:
/// - 0 elements — no real intersection (line misses circle).
/// - 1 element  — tangent (discriminant = 0).
/// - 2 elements — two distinct intersection points.
pub fn intersect_line2d_circle2d(line: &Line2D, circle: &Circle2D) -> Vec<Point2D> {
    // Substituting x = qx + t*dx, y = qy + t*dy into
    //   (x - cx)^2 + (y - cy)^2 = r^2
    // gives A*t^2 + B*t + C = 0 where:
    //   A = dx^2 + dy^2
    //   B = 2*(dx*(qx-cx) + dy*(qy-cy))
    //   C = (qx-cx)^2 + (qy-cy)^2 - r^2

    let dx = &line.direction.x;
    let dy = &line.direction.y;
    let qx = &line.point.x;
    let qy = &line.point.y;
    let cx = &circle.center.x;
    let cy = &circle.center.y;
    let r = &circle.radius;

    let dxcx = normalize::sub(qx.clone(), cx.clone()); // qx - cx
    let dycy = normalize::sub(qy.clone(), cy.clone()); // qy - cy

    let a = normalize::add(
        normalize::pow(dx.clone(), Expr::int(2)),
        normalize::pow(dy.clone(), Expr::int(2)),
    );
    let b = normalize::mul(
        Expr::int(2),
        normalize::add(
            normalize::mul(dx.clone(), dxcx.clone()),
            normalize::mul(dy.clone(), dycy.clone()),
        ),
    );
    let c = normalize::sub(
        normalize::add(
            normalize::pow(dxcx, Expr::int(2)),
            normalize::pow(dycy, Expr::int(2)),
        ),
        normalize::pow(r.clone(), Expr::int(2)),
    );

    // discriminant = B^2 - 4*A*C
    let discriminant = normalize::sub(
        normalize::pow(b.clone(), Expr::int(2)),
        normalize::mul(normalize::mul(Expr::int(4), a.clone()), c),
    );

    // We need to evaluate the discriminant sign; only possible for
    // numeric (integer/rational) discriminants.
    let disc_sign = numeric_sign(&discriminant);

    match disc_sign {
        Some(s) if s < 0 => {
            // No real roots
            vec![]
        }
        Some(0) => {
            // Tangent: t = -B / (2A)
            let t = normalize::div(normalize::neg(b), normalize::mul(Expr::int(2), a.clone()));
            let px = normalize::add(qx.clone(), normalize::mul(t.clone(), dx.clone()));
            let py = normalize::add(qy.clone(), normalize::mul(t, dy.clone()));
            vec![Point2D::new(px, py)]
        }
        _ => {
            // Two roots: t = (-B ± sqrt(disc)) / (2A)
            let sqrt_disc = Expr::func(FuncId::Sqrt, vec![discriminant]);
            let two_a = normalize::mul(Expr::int(2), a);
            let neg_b = normalize::neg(b);

            let t1 = normalize::div(
                normalize::add(neg_b.clone(), sqrt_disc.clone()),
                two_a.clone(),
            );
            let t2 = normalize::div(normalize::sub(neg_b, sqrt_disc), two_a);

            let p1x = normalize::add(qx.clone(), normalize::mul(t1.clone(), dx.clone()));
            let p1y = normalize::add(qy.clone(), normalize::mul(t1, dy.clone()));

            let p2x = normalize::add(qx.clone(), normalize::mul(t2.clone(), dx.clone()));
            let p2y = normalize::add(qy.clone(), normalize::mul(t2, dy.clone()));

            vec![Point2D::new(p1x, p1y), Point2D::new(p2x, p2y)]
        }
    }
}

/// Return the sign of a numeric `Arc<Expr>`: `Some(-1/0/+1)` for integer or
/// rational constants, `None` for symbolic expressions.
fn numeric_sign(expr: &Arc<Expr>) -> Option<i32> {
    match expr.as_ref() {
        Expr::Integer(n) => {
            if let Some(v) = n.to_i64() {
                if v < 0 {
                    Some(-1)
                } else if v == 0 {
                    Some(0)
                } else {
                    Some(1)
                }
            } else {
                None
            }
        }
        Expr::Rational(r) => {
            use num::traits::Zero;
            if r.is_zero() {
                Some(0)
            } else if *r < num::traits::Zero::zero() {
                Some(-1)
            } else {
                Some(1)
            }
        }
        Expr::Float(f) => {
            if *f < 0.0 {
                Some(-1)
            } else if *f == 0.0 {
                Some(0)
            } else {
                Some(1)
            }
        }
        _ => None,
    }
}

// ── Intersection: line ↔ plane (3-D) ─────────────────────────────────────────

/// Symbolic intersection of a 3-D line with a plane.
///
/// Parametrizes the line as `p + t·d` and substitutes into the plane
/// equation `n · (x − q) = 0` to solve for `t`.
pub fn intersect_line3d_plane3d(line: &Line3D, plane: &Plane3D) -> LinePlaneIntersection {
    let n = &plane.normal;
    let q = &plane.point;
    let d = &line.direction;
    let p = &line.point;

    // n · d
    let n_dot_d = normalize::add(
        normalize::add(
            normalize::mul(n.x.clone(), d.x.clone()),
            normalize::mul(n.y.clone(), d.y.clone()),
        ),
        normalize::mul(n.z.clone(), d.z.clone()),
    );

    if n_dot_d.is_zero() {
        // Line is parallel to plane; check if it lies in the plane
        let pq = [
            normalize::sub(p.x.clone(), q.x.clone()),
            normalize::sub(p.y.clone(), q.y.clone()),
            normalize::sub(p.z.clone(), q.z.clone()),
        ];
        let n_arr = [n.x.clone(), n.y.clone(), n.z.clone()];
        let n_dot_pq = dot(&n_arr, &pq);
        if n_dot_pq.is_zero() {
            LinePlaneIntersection::InPlane
        } else {
            LinePlaneIntersection::Parallel
        }
    } else {
        // t = n · (q − p) / (n · d)
        let qp = [
            normalize::sub(q.x.clone(), p.x.clone()),
            normalize::sub(q.y.clone(), p.y.clone()),
            normalize::sub(q.z.clone(), p.z.clone()),
        ];
        let n_arr = [n.x.clone(), n.y.clone(), n.z.clone()];
        let n_dot_qp = dot(&n_arr, &qp);
        let t = normalize::div(n_dot_qp, n_dot_d);

        let ix = normalize::add(p.x.clone(), normalize::mul(t.clone(), d.x.clone()));
        let iy = normalize::add(p.y.clone(), normalize::mul(t.clone(), d.y.clone()));
        let iz = normalize::add(p.z.clone(), normalize::mul(t, d.z.clone()));
        LinePlaneIntersection::Point(Point3D::new(ix, iy, iz))
    }
}

// ── Intersection: plane ↔ plane (3-D) ────────────────────────────────────────

/// Symbolic intersection of two 3-D planes.
///
/// Returns:
/// - [`PlanePlaneIntersection::Line`] — a `Line3D` with direction `n₁ × n₂`
///   and a base point found by solving the two plane equations (treating z = 0
///   as a free parameter when the system is non-degenerate in x-y).
/// - [`PlanePlaneIntersection::Parallel`] — normals are proportional but planes differ.
/// - [`PlanePlaneIntersection::Coincident`] — same plane.
pub fn intersect_plane3d_plane3d(a: &Plane3D, b: &Plane3D) -> PlanePlaneIntersection {
    let na = [a.normal.x.clone(), a.normal.y.clone(), a.normal.z.clone()];
    let nb = [b.normal.x.clone(), b.normal.y.clone(), b.normal.z.clone()];
    let dir = cross3(&na, &nb);
    let dir_mag = magnitude3(&dir);

    if dir_mag.is_zero() {
        // Parallel or coincident
        let diff = [
            normalize::sub(b.point.x.clone(), a.point.x.clone()),
            normalize::sub(b.point.y.clone(), a.point.y.clone()),
            normalize::sub(b.point.z.clone(), a.point.z.clone()),
        ];
        let n_dot_diff = dot(&na, &diff);
        if n_dot_diff.is_zero() {
            PlanePlaneIntersection::Coincident
        } else {
            PlanePlaneIntersection::Parallel
        }
    } else {
        // Find a point on both planes: set z = 0, solve the 2×2 system
        //   na.x * x + na.y * y = na · a.point
        //   nb.x * x + nb.y * y = nb · b.point
        let da = dot(
            &na,
            &[a.point.x.clone(), a.point.y.clone(), a.point.z.clone()],
        );
        let db = dot(
            &nb,
            &[b.point.x.clone(), b.point.y.clone(), b.point.z.clone()],
        );

        let denom_xy = normalize::sub(
            normalize::mul(na[0].clone(), nb[1].clone()),
            normalize::mul(na[1].clone(), nb[0].clone()),
        );

        let (px, py, pz) = if !denom_xy.is_zero() {
            // x = (da * nb.y - db * na.y) / denom_xy
            let x = normalize::div(
                normalize::sub(
                    normalize::mul(da.clone(), nb[1].clone()),
                    normalize::mul(db.clone(), na[1].clone()),
                ),
                denom_xy.clone(),
            );
            // y = (na.x * db - nb.x * da) / denom_xy
            let y = normalize::div(
                normalize::sub(
                    normalize::mul(na[0].clone(), db),
                    normalize::mul(nb[0].clone(), da),
                ),
                denom_xy,
            );
            (x, y, Expr::int(0))
        } else {
            // x-y system degenerate; try setting y = 0, solve x-z
            let denom_xz = normalize::sub(
                normalize::mul(na[0].clone(), nb[2].clone()),
                normalize::mul(na[2].clone(), nb[0].clone()),
            );
            // x = (da * nb.z - db * na.z) / denom_xz
            let x = normalize::div(
                normalize::sub(
                    normalize::mul(da.clone(), nb[2].clone()),
                    normalize::mul(db.clone(), na[2].clone()),
                ),
                denom_xz.clone(),
            );
            // z = (na.x * db - nb.x * da) / denom_xz
            let z = normalize::div(
                normalize::sub(
                    normalize::mul(na[0].clone(), db),
                    normalize::mul(nb[0].clone(), da),
                ),
                denom_xz,
            );
            (x, Expr::int(0), z)
        };

        let line_point = Point3D::new(px, py, pz);
        let line_dir = Point3D::new(dir[0].clone(), dir[1].clone(), dir[2].clone());
        PlanePlaneIntersection::Line(Line3D::new(line_point, line_dir))
    }
}

// ── Intersection: circle ↔ circle (2-D) ──────────────────────────────────────

/// Symbolic intersection of two 2-D circles.
///
/// Subtracts the two circle equations to get a linear equation (the
/// radical axis), then intersects that line with the first circle using
/// [`intersect_line2d_circle2d`].
///
/// Returns a `Vec<Point2D>` with 0, 1, or 2 points.
pub fn intersect_circle2d_circle2d(c1: &Circle2D, c2: &Circle2D) -> Vec<Point2D> {
    // (x-cx1)² + (y-cy1)² = r1²
    // (x-cx2)² + (y-cy2)² = r2²
    //
    // Subtract: expand and simplify to get the radical axis:
    // 2*(cx2-cx1)*x + 2*(cy2-cy1)*y = r1²-r2² + cx2²-cx1² + cy2²-cy1²
    //
    // Represent as Line2D through the radical axis.

    let cx1 = &c1.center.x;
    let cy1 = &c1.center.y;
    let cx2 = &c2.center.x;
    let cy2 = &c2.center.y;
    let r1 = &c1.radius;
    let r2 = &c2.radius;

    // Direction of radical axis is perpendicular to the line of centers
    // d = (cy1 - cy2, cx2 - cx1)  (perpendicular to (cx2-cx1, cy2-cy1))
    let rad_dx = normalize::sub(cy1.clone(), cy2.clone()); // -(cy2-cy1)
    let rad_dy = normalize::sub(cx2.clone(), cx1.clone()); // (cx2-cx1)

    // A point on the radical axis: solve for a specific x (or y).
    // The radical axis equation:
    //   2*(cx2-cx1)*x + 2*(cy2-cy1)*y
    //     = r1² - r2² + cx2² - cx1² + cy2² - cy1²
    //
    // Coefficient of x: A = 2*(cx2 - cx1)
    // Coefficient of y: B = 2*(cy2 - cy1)
    // RHS: R = r1² - r2² + cx2² - cx1² + cy2² - cy1²

    let diff_cx = normalize::sub(cx2.clone(), cx1.clone());
    let diff_cy = normalize::sub(cy2.clone(), cy1.clone());
    let aa = normalize::mul(Expr::int(2), diff_cx.clone());
    let bb = normalize::mul(Expr::int(2), diff_cy.clone());

    let rhs = normalize::add(
        normalize::sub(
            normalize::pow(r1.clone(), Expr::int(2)),
            normalize::pow(r2.clone(), Expr::int(2)),
        ),
        normalize::sub(
            normalize::add(
                normalize::pow(cx2.clone(), Expr::int(2)),
                normalize::pow(cy2.clone(), Expr::int(2)),
            ),
            normalize::add(
                normalize::pow(cx1.clone(), Expr::int(2)),
                normalize::pow(cy1.clone(), Expr::int(2)),
            ),
        ),
    );

    // Find a base point on the radical axis.
    // If A ≠ 0: set y = 0, x = R/A
    // else if B ≠ 0: set x = 0, y = R/B
    // else: coincident circles, no unique axis (handle separately)
    let base_point = if !aa.is_zero() {
        let bpx = normalize::div(rhs.clone(), aa.clone());
        Point2D::new(bpx, Expr::int(0))
    } else if !bb.is_zero() {
        let bpy = normalize::div(rhs.clone(), bb.clone());
        Point2D::new(Expr::int(0), bpy)
    } else {
        // Coincident or point-degenerate: circles are concentric.
        // Intersection depends entirely on radii.
        return vec![];
    };

    let radical_axis = Line2D {
        point: base_point,
        direction: Point2D::new(rad_dx, rad_dy),
    };

    intersect_line2d_circle2d(&radical_axis, c1)
}

// ── Tangent / normal lines ────────────────────────────────────────────────────

/// Tangent line to a circle at a point on the circle.
///
/// The tangent at `p` on circle `c` is perpendicular to the radius
/// `(p − center)`.  Direction = `(-(py - cy), px - cx)` (90° rotation of radius).
///
/// Returns `None` if `p` is not on the circle (checked numerically when
/// coordinates are numeric constants; symbolic inputs always return `Some`).
pub fn tangent_to_circle2d(circle: &Circle2D, p: &Point2D) -> Option<Line2D> {
    let cx = &circle.center.x;
    let cy = &circle.center.y;

    // Radius vector from center to p
    let rx = normalize::sub(p.x.clone(), cx.clone());
    let ry = normalize::sub(p.y.clone(), cy.clone());

    // For numeric inputs, verify the point is on the circle.
    if let (Some(rxv), Some(ryv), Some(rv)) = (
        expr_to_f64(&rx),
        expr_to_f64(&ry),
        expr_to_f64(&circle.radius),
    ) {
        let dist_sq = rxv * rxv + ryv * ryv;
        let r_sq = rv * rv;
        if (dist_sq - r_sq).abs() > 1e-9 * r_sq.max(1.0) {
            return None;
        }
    }

    // Tangent direction: perpendicular to radius = (-ry, rx)
    let tan_dx = normalize::neg(ry);
    let tan_dy = rx;

    Some(Line2D {
        point: p.clone(),
        direction: Point2D::new(tan_dx, tan_dy),
    })
}

/// Normal line to a circle at a point on the circle.
///
/// The normal is the line through the center and `p`; direction = radius vector
/// `(px − cx, py − cy)`.
///
/// Returns `None` if `p` is not on the circle (same check as
/// [`tangent_to_circle2d`]).
pub fn normal_to_circle2d(circle: &Circle2D, p: &Point2D) -> Option<Line2D> {
    let cx = &circle.center.x;
    let cy = &circle.center.y;

    let rx = normalize::sub(p.x.clone(), cx.clone());
    let ry = normalize::sub(p.y.clone(), cy.clone());

    // Numeric on-circle check.
    if let (Some(rxv), Some(ryv), Some(rv)) = (
        expr_to_f64(&rx),
        expr_to_f64(&ry),
        expr_to_f64(&circle.radius),
    ) {
        let dist_sq = rxv * rxv + ryv * ryv;
        let r_sq = rv * rv;
        if (dist_sq - r_sq).abs() > 1e-9 * r_sq.max(1.0) {
            return None;
        }
    }

    Some(Line2D {
        point: p.clone(),
        direction: Point2D::new(rx, ry),
    })
}

/// Tangent line to a parametric curve at parameter value `t₀`.
///
/// The tangent direction is the derivative vector `r'(t₀)`.  Each
/// component of `r'(t)` is evaluated at `t₀` by symbolic substitution.
///
/// Returns a [`Line2D`] for 2-D curves and a [`Line3D`] for 3-D curves,
/// wrapped in [`TangentLine`].
pub fn tangent_to_parametric_curve(curve: &ParametricCurve, t0: &Arc<Expr>) -> TangentLine {
    let tangent = curve.tangent_vector();
    let t_id = curve.param;

    // Substitute t₀ into each component of the tangent vector.
    let tangent_at_t0: Vec<Arc<Expr>> = tangent.iter().map(|c| substitute(c, t_id, t0)).collect();

    // Also substitute t₀ into each component of the position to get the base point.
    let point_at_t0: Vec<Arc<Expr>> = curve
        .components
        .iter()
        .map(|c| substitute(c, t_id, t0))
        .collect();

    match tangent_at_t0.len() {
        2 => TangentLine::Line2D(Line2D {
            point: Point2D::new(point_at_t0[0].clone(), point_at_t0[1].clone()),
            direction: Point2D::new(tangent_at_t0[0].clone(), tangent_at_t0[1].clone()),
        }),
        3 => TangentLine::Line3D(Line3D {
            point: Point3D::new(
                point_at_t0[0].clone(),
                point_at_t0[1].clone(),
                point_at_t0[2].clone(),
            ),
            direction: Point3D::new(
                tangent_at_t0[0].clone(),
                tangent_at_t0[1].clone(),
                tangent_at_t0[2].clone(),
            ),
        }),
        n => panic!(
            "tangent_to_parametric_curve: unsupported dimension {} (must be 2 or 3)",
            n
        ),
    }
}

/// Normal line to a parametric curve at parameter value `t₀`.
///
/// For a 2-D curve the normal direction is the 90°-rotated tangent
/// `(-T_y, T_x)`.  For a 3-D curve the normal direction is the
/// component of `r''(t₀)` orthogonal to `r'(t₀)` (Frenet normal).
pub fn normal_to_parametric_curve(curve: &ParametricCurve, t0: &Arc<Expr>) -> TangentLine {
    let normal = curve.normal_vector();
    let t_id = curve.param;

    let normal_at_t0: Vec<Arc<Expr>> = normal.iter().map(|c| substitute(c, t_id, t0)).collect();

    let point_at_t0: Vec<Arc<Expr>> = curve
        .components
        .iter()
        .map(|c| substitute(c, t_id, t0))
        .collect();

    match normal_at_t0.len() {
        2 => TangentLine::Line2D(Line2D {
            point: Point2D::new(point_at_t0[0].clone(), point_at_t0[1].clone()),
            direction: Point2D::new(normal_at_t0[0].clone(), normal_at_t0[1].clone()),
        }),
        3 => TangentLine::Line3D(Line3D {
            point: Point3D::new(
                point_at_t0[0].clone(),
                point_at_t0[1].clone(),
                point_at_t0[2].clone(),
            ),
            direction: Point3D::new(
                normal_at_t0[0].clone(),
                normal_at_t0[1].clone(),
                normal_at_t0[2].clone(),
            ),
        }),
        n => panic!(
            "normal_to_parametric_curve: unsupported dimension {} (must be 2 or 3)",
            n
        ),
    }
}

/// Return type for tangent/normal lines that can be 2-D or 3-D.
#[derive(Clone, Debug, PartialEq)]
pub enum TangentLine {
    /// 2-D tangent or normal line.
    Line2D(Line2D),
    /// 3-D tangent or normal line.
    Line3D(Line3D),
}

// ── Internal: numeric expression extractor ────────────────────────────────────

/// Try to extract an `f64` value from a purely numeric `Arc<Expr>`.
fn expr_to_f64(e: &Arc<Expr>) -> Option<f64> {
    match e.as_ref() {
        Expr::Integer(n) => n.to_i64().map(|v| v as f64),
        Expr::Rational(r) => Some(r.to_f64()),
        Expr::Float(f) => Some(*f),
        _ => None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{
        evaluation::evaluate, expr::FuncId, normalize, Expr, FuncId as NF, SymbolId,
    };
    use std::collections::HashMap;

    // ── Helper builders ───────────────────────────────────────────────────────

    fn pt2(x: i64, y: i64) -> Point2D {
        Point2D::new(Expr::int(x), Expr::int(y))
    }

    fn pt3(x: i64, y: i64, z: i64) -> Point3D {
        Point3D::new(Expr::int(x), Expr::int(y), Expr::int(z))
    }

    fn line2d_through(p: Point2D, d: Point2D) -> Line2D {
        Line2D {
            point: p,
            direction: d,
        }
    }

    fn circle_at(cx: i64, cy: i64, r: i64) -> Circle2D {
        Circle2D::new(pt2(cx, cy), Expr::int(r))
    }

    /// Numerically evaluate a ground (symbol-free) `Arc<Expr>`, panicking if not evaluable.
    fn to_f64(e: &Arc<Expr>) -> f64 {
        let empty: HashMap<SymbolId, f64> = HashMap::new();
        evaluate(e.as_ref(), &empty).unwrap_or_else(|| panic!("to_f64: cannot evaluate {e}"))
    }

    // ── Distance: point ↔ point ───────────────────────────────────────────────

    /// Distance from (3, 4) to origin = 5.
    #[test]
    fn test_dist_point2d_point2d_345() {
        let a = pt2(0, 0);
        let b = pt2(3, 4);
        let d = dist_point2d_point2d(&a, &b);
        assert_eq!(*d, Expr::Integer(crate::numeric::SmallInt::from(5i64)));
    }

    /// Distance from (0,0,0) to (1,2,2) = sqrt(9) = 3.
    #[test]
    fn test_dist_point3d_point3d_unit() {
        let a = pt3(0, 0, 0);
        let b = pt3(1, 2, 2);
        let d = dist_point3d_point3d(&a, &b);
        assert_eq!(*d, Expr::Integer(crate::numeric::SmallInt::from(3i64)));
    }

    // ── Distance: point ↔ line (2-D) ──────────────────────────────────────────

    /// Distance from (1, 0) to the line y = x (through origin, direction (1,1)).
    ///
    /// Formula: |dx*(py-qy) - dy*(px-qx)| / |d|
    ///        = |1*(0-0) - 1*(1-0)| / sqrt(2)
    ///        = 1 / sqrt(2) = sqrt(2)/2 ≈ 0.7071
    #[test]
    fn test_dist_point2d_line2d_diagonal() {
        let p = pt2(1, 0);
        let line = Line2D::from_two_points(pt2(0, 0), pt2(1, 1));
        let d = dist_point2d_line2d(&p, &line);
        let val = to_f64(&d);
        let expected = 1.0_f64 / 2.0_f64.sqrt();
        assert!(
            (val - expected).abs() < 1e-9,
            "expected {expected}, got {val}"
        );
    }

    /// Point on the line → distance = 0.
    #[test]
    fn test_dist_point2d_line2d_on_line() {
        let p = pt2(2, 2);
        let line = Line2D::from_two_points(pt2(0, 0), pt2(1, 1));
        let d = dist_point2d_line2d(&p, &line);
        assert!(d.is_zero(), "point on line should have distance 0");
    }

    // ── Distance: point ↔ plane (3-D) ─────────────────────────────────────────

    /// Distance from (0,0,0) to plane 2x + 3y + 6z = 7.
    ///
    /// Plane through (7/2, 0, 0) with normal (2, 3, 6).
    /// dist = |2*0 + 3*0 + 6*0 - 7| / sqrt(4+9+36) = 7/7 = 1.
    ///
    /// We build the plane as: point=(7/2,0,0), normal=(2,3,6).
    #[test]
    fn test_dist_point3d_plane3d_gives_one() {
        let origin = pt3(0, 0, 0);
        // Plane 2x + 3y + 6z = 7: normal=(2,3,6), point where only x: (7/2, 0, 0)
        let plane_pt = Point3D::new(Expr::rational(7, 2), Expr::int(0), Expr::int(0));
        let plane = Plane3D {
            point: plane_pt,
            normal: pt3(2, 3, 6),
        };
        let d = dist_point3d_plane3d(&origin, &plane);
        let val = to_f64(&d);
        assert!(
            (val - 1.0).abs() < 1e-9,
            "distance from origin to plane 2x+3y+6z=7 should be 1, got {val}"
        );
    }

    // ── Distance: skew lines (3-D) ─────────────────────────────────────────────

    /// Canonical skew-line example.
    ///
    /// Line a: point (0,0,0), direction (1,0,0) — the x-axis.
    /// Line b: point (0,1,0), direction (0,0,1) — shifted along y, direction z.
    ///
    /// d_a × d_b = (1,0,0) × (0,0,1) = (0*1-0*0, 0*0-1*1, 1*0-0*0) = (0,-1,0)
    /// diff = b.point - a.point = (0,1,0)
    /// dist = |diff · (da×db)| / |da×db| = |(0,1,0)·(0,-1,0)| / 1 = 1
    #[test]
    fn test_dist_line3d_line3d_skew() {
        let a = Line3D::new(pt3(0, 0, 0), pt3(1, 0, 0));
        let b = Line3D::new(pt3(0, 1, 0), pt3(0, 0, 1));
        let d = dist_line3d_line3d(&a, &b);
        let val = to_f64(&d);
        assert!(
            (val - 1.0).abs() < 1e-9,
            "skew-line distance should be 1, got {val}"
        );
    }

    // ── Intersection: line ↔ line (2-D) ───────────────────────────────────────

    /// Lines x-axis and y-axis intersect at origin.
    #[test]
    fn test_intersect_line2d_line2d_at_origin() {
        let x_axis = Line2D::from_two_points(pt2(0, 0), pt2(1, 0));
        let y_axis = Line2D::from_two_points(pt2(0, 0), pt2(0, 1));
        match intersect_line2d_line2d(&x_axis, &y_axis) {
            LineLineIntersection2D::Point(p) => {
                assert!(p.x.is_zero(), "ix should be 0");
                assert!(p.y.is_zero(), "iy should be 0");
            }
            other => panic!("expected Point, got {other:?}"),
        }
    }

    /// Lines y=0 and y=1 are parallel.
    #[test]
    fn test_intersect_line2d_parallel() {
        let l1 = Line2D::from_two_points(pt2(0, 0), pt2(1, 0)); // y = 0
        let l2 = Line2D::from_two_points(pt2(0, 1), pt2(1, 1)); // y = 1
        assert_eq!(
            intersect_line2d_line2d(&l1, &l2),
            LineLineIntersection2D::Parallel
        );
    }

    /// Coincident lines (same line, two representations).
    #[test]
    fn test_intersect_line2d_coincident() {
        let l1 = Line2D::from_two_points(pt2(0, 0), pt2(2, 2));
        let l2 = Line2D::from_two_points(pt2(1, 1), pt2(3, 3));
        assert_eq!(
            intersect_line2d_line2d(&l1, &l2),
            LineLineIntersection2D::Coincident
        );
    }

    /// Diagonal lines y=x and y=-x+2 intersect at (1,1).
    #[test]
    fn test_intersect_line2d_specific_point() {
        // y = x : through (0,0) direction (1,1)
        let l1 = Line2D::from_two_points(pt2(0, 0), pt2(1, 1));
        // y = -x + 2 : through (2,0) direction (-1,1)
        let l2 = Line2D::from_two_points(pt2(2, 0), pt2(1, 1));
        match intersect_line2d_line2d(&l1, &l2) {
            LineLineIntersection2D::Point(p) => {
                assert_eq!(
                    *p.x,
                    Expr::Integer(crate::numeric::SmallInt::from(1i64)),
                    "x"
                );
                assert_eq!(
                    *p.y,
                    Expr::Integer(crate::numeric::SmallInt::from(1i64)),
                    "y"
                );
            }
            other => panic!("expected Point(1,1), got {other:?}"),
        }
    }

    // ── Intersection: line ↔ circle (2-D) ─────────────────────────────────────

    /// x = 1 is tangent to the unit circle — one intersection at (1, 0).
    #[test]
    fn test_intersect_line2d_circle2d_tangent() {
        // Line x = 1: point (1, 0), direction (0, 1)
        let line = line2d_through(pt2(1, 0), Point2D::new(Expr::int(0), Expr::int(1)));
        let circle = circle_at(0, 0, 1);
        let pts = intersect_line2d_circle2d(&line, &circle);
        assert_eq!(pts.len(), 1, "x=1 should be tangent to unit circle");
        let p = &pts[0];
        assert_eq!(
            *p.x,
            Expr::Integer(crate::numeric::SmallInt::from(1i64)),
            "tangent x"
        );
        assert!(p.y.is_zero(), "tangent y should be 0");
    }

    /// Line x = 2 misses the unit circle — zero intersections.
    #[test]
    fn test_intersect_line2d_circle2d_no_intersection() {
        let line = line2d_through(pt2(2, 0), Point2D::new(Expr::int(0), Expr::int(1)));
        let circle = circle_at(0, 0, 1);
        let pts = intersect_line2d_circle2d(&line, &circle);
        assert_eq!(pts.len(), 0, "x=2 should miss unit circle");
    }

    /// y = 0 (x-axis) intersects the unit circle at (-1, 0) and (1, 0).
    #[test]
    fn test_intersect_line2d_circle2d_two_points() {
        let line = Line2D::from_two_points(pt2(-2, 0), pt2(2, 0)); // y = 0
        let circle = circle_at(0, 0, 1);
        let pts = intersect_line2d_circle2d(&line, &circle);
        assert_eq!(pts.len(), 2, "x-axis intersects unit circle at 2 points");
        // Check y-coordinates are zero
        for p in &pts {
            assert!(p.y.is_zero(), "intersection y should be 0 on x-axis");
        }
        // x values should be ±1
        let xs: Vec<f64> = pts.iter().map(|p| to_f64(&p.x)).collect();
        let mut xs_sorted = xs.clone();
        xs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((xs_sorted[0] - (-1.0)).abs() < 1e-9, "x0 should be -1");
        assert!((xs_sorted[1] - 1.0).abs() < 1e-9, "x1 should be 1");
    }

    // ── Intersection: circle ↔ circle (2-D) ───────────────────────────────────

    /// Unit circle at origin and circle of radius 1 at (1, 0) intersect
    /// at two points: (1/2, ±sqrt(3)/2).
    #[test]
    fn test_intersect_circle2d_circle2d_two_points() {
        let c1 = circle_at(0, 0, 1);
        let c2 = circle_at(1, 0, 1);
        let pts = intersect_circle2d_circle2d(&c1, &c2);
        assert_eq!(
            pts.len(),
            2,
            "two unit circles should intersect at 2 points"
        );
        // Both intersection points should have x = 1/2 and y = ±sqrt(3)/2
        for p in &pts {
            let xv = to_f64(&p.x);
            assert!((xv - 0.5).abs() < 1e-9, "x should be 0.5, got {xv}");
            let yv = to_f64(&p.y).abs();
            let expected_y = (3.0_f64 / 4.0_f64).sqrt();
            assert!(
                (yv - expected_y).abs() < 1e-9,
                "y abs should be sqrt(3)/2 ≈ {expected_y}, got {yv}"
            );
        }
    }

    // ── Intersection: line ↔ plane (3-D) ──────────────────────────────────────

    /// z-axis hits the xy-plane (z=0) at origin.
    #[test]
    fn test_intersect_line3d_plane3d_hits_xy_plane() {
        // z-axis: through (0,0,5), direction (0,0,-1)
        let line = Line3D::new(
            pt3(0, 0, 5),
            Point3D::new(Expr::int(0), Expr::int(0), Expr::int(-1)),
        );
        // xy-plane: through origin, normal (0,0,1)
        let plane = Plane3D {
            point: pt3(0, 0, 0),
            normal: pt3(0, 0, 1),
        };
        match intersect_line3d_plane3d(&line, &plane) {
            LinePlaneIntersection::Point(p) => {
                assert!(p.x.is_zero(), "ix should be 0");
                assert!(p.y.is_zero(), "iy should be 0");
                assert!(p.z.is_zero(), "iz should be 0");
            }
            other => panic!("expected Point, got {other:?}"),
        }
    }

    /// Line parallel to plane — no intersection.
    #[test]
    fn test_intersect_line3d_plane3d_parallel() {
        // Line y = 1 in xz plane: through (0,1,0), direction (1,0,0)
        let line = Line3D::new(pt3(0, 1, 0), pt3(1, 0, 0));
        // xy-plane: normal (0,0,1), point origin
        let plane = Plane3D {
            point: pt3(0, 0, 0),
            normal: pt3(0, 1, 0),
        };
        // n = (0,1,0), d = (1,0,0), n·d = 0 → parallel
        // Check: p = (0,1,0), q = (0,0,0), n·(p-q) = (0,1,0)·(0,1,0) = 1 ≠ 0
        assert_eq!(
            intersect_line3d_plane3d(&line, &plane),
            LinePlaneIntersection::Parallel
        );
    }

    // ── Intersection: plane ↔ plane (3-D) ─────────────────────────────────────

    /// xy-plane (z=0) and xz-plane (y=0) intersect along the x-axis.
    #[test]
    fn test_intersect_plane3d_plane3d_x_axis() {
        // xy-plane: normal (0,0,1), through origin
        let pxy = Plane3D {
            point: pt3(0, 0, 0),
            normal: pt3(0, 0, 1),
        };
        // xz-plane: normal (0,1,0), through origin
        let pxz = Plane3D {
            point: pt3(0, 0, 0),
            normal: pt3(0, 1, 0),
        };
        match intersect_plane3d_plane3d(&pxy, &pxz) {
            PlanePlaneIntersection::Line(line) => {
                // Direction should be n_xy × n_xz = (0,0,1) × (0,1,0) = (-1,0,0) or (1,0,0)
                let d = &line.direction;
                assert!(
                    d.y.is_zero() && d.z.is_zero(),
                    "intersection of xy/xz planes should be x-axis direction, got {:?}",
                    d
                );
            }
            other => panic!("expected Line, got {other:?}"),
        }
    }

    /// Parallel planes — no intersection.
    #[test]
    fn test_intersect_plane3d_plane3d_parallel() {
        let p1 = Plane3D {
            point: pt3(0, 0, 0),
            normal: pt3(0, 0, 1),
        };
        let p2 = Plane3D {
            point: pt3(0, 0, 1),
            normal: pt3(0, 0, 1),
        };
        assert_eq!(
            intersect_plane3d_plane3d(&p1, &p2),
            PlanePlaneIntersection::Parallel
        );
    }

    // ── Tangent / Normal to circle ─────────────────────────────────────────────

    /// Tangent to unit circle at (1, 0): direction should be (0, 1) [or (-0, 1)].
    #[test]
    fn test_tangent_to_circle2d_at_1_0() {
        let circle = circle_at(0, 0, 1);
        let p = pt2(1, 0);
        let tan = tangent_to_circle2d(&circle, &p).expect("should be on circle");
        // Tangent direction = (-ry, rx) = (-(0-0), (1-0)) = (0, 1)
        assert!(
            tan.direction.x.is_zero(),
            "tangent dx should be 0, got {:?}",
            tan.direction.x
        );
        assert!(
            tan.direction.y.is_one(),
            "tangent dy should be 1, got {:?}",
            tan.direction.y
        );
    }

    /// Normal to unit circle at (0, 1): direction should be (0, 1).
    #[test]
    fn test_normal_to_circle2d_at_0_1() {
        let circle = circle_at(0, 0, 1);
        let p = pt2(0, 1);
        let norm = normal_to_circle2d(&circle, &p).expect("should be on circle");
        assert!(norm.direction.x.is_zero(), "normal dx should be 0");
        assert!(norm.direction.y.is_one(), "normal dy should be 1");
    }

    /// Point off circle → `None`.
    #[test]
    fn test_tangent_to_circle2d_off_circle_returns_none() {
        let circle = circle_at(0, 0, 1);
        let p = pt2(2, 0); // outside
        assert!(tangent_to_circle2d(&circle, &p).is_none());
    }

    // ── Tangent / Normal to parametric curve ───────────────────────────────────

    /// Tangent to (cos t, sin t) at t = 0: direction (-sin 0, cos 0) = (0, 1).
    #[test]
    fn test_tangent_to_parametric_circle_at_t0() {
        let t_name = "rel_tan_t0";
        let t_id = SymbolId::intern(t_name);
        let t = Expr::symbol(t_name);
        let curve = ParametricCurve::new(
            vec![
                Expr::func(crate::numeric::FuncId::Cos, vec![t.clone()]),
                Expr::func(crate::numeric::FuncId::Sin, vec![t.clone()]),
            ],
            t_id,
        );
        let t0 = Expr::int(0);
        match tangent_to_parametric_curve(&curve, &t0) {
            TangentLine::Line2D(line) => {
                // direction at t=0: (-sin(0), cos(0)) = (0, 1)
                assert!(
                    line.direction.x.is_zero(),
                    "tangent dx at t=0 should be 0, got {:?}",
                    line.direction.x
                );
                assert!(
                    line.direction.y.is_one(),
                    "tangent dy at t=0 should be 1, got {:?}",
                    line.direction.y
                );
            }
            other => panic!("expected Line2D, got {other:?}"),
        }
    }

    /// Normal to (cos t, sin t) at t = 0: direction = N₂D = (-cos 0, -sin 0) = (-1, 0).
    #[test]
    fn test_normal_to_parametric_circle_at_t0() {
        let t_name = "rel_norm_t0";
        let t_id = SymbolId::intern(t_name);
        let t = Expr::symbol(t_name);
        let curve = ParametricCurve::new(
            vec![
                Expr::func(crate::numeric::FuncId::Cos, vec![t.clone()]),
                Expr::func(crate::numeric::FuncId::Sin, vec![t.clone()]),
            ],
            t_id,
        );
        let t0 = Expr::int(0);
        match normal_to_parametric_curve(&curve, &t0) {
            TangentLine::Line2D(line) => {
                // Normal at t=0: N₂D = (-T_y, T_x) = (-cos(0), -sin(0)) = (-1, 0)
                let nx = to_f64(&line.direction.x);
                let ny = to_f64(&line.direction.y);
                assert!(
                    (nx + 1.0).abs() < 1e-9,
                    "normal dx at t=0 should be -1, got {nx}"
                );
                assert!(ny.abs() < 1e-9, "normal dy at t=0 should be 0, got {ny}");
            }
            other => panic!("expected Line2D, got {other:?}"),
        }
    }

    // ── Parallel-line distance ────────────────────────────────────────────────

    /// Parallel lines y=0 and y=3 have distance 3.
    #[test]
    fn test_dist_parallel_lines_2d() {
        let l1 = Line2D::from_two_points(pt2(0, 0), pt2(1, 0)); // y = 0
        let l2 = Line2D::from_two_points(pt2(0, 3), pt2(1, 3)); // y = 3
        let d = dist_line2d_line2d(&l1, &l2).expect("parallel lines should have distance");
        let val = to_f64(&d);
        assert!(
            (val - 3.0).abs() < 1e-9,
            "parallel lines y=0 and y=3 should have distance 3, got {val}"
        );
    }

    /// Non-parallel lines → `None`.
    #[test]
    fn test_dist_non_parallel_lines_2d_is_none() {
        let l1 = Line2D::from_two_points(pt2(0, 0), pt2(1, 0));
        let l2 = Line2D::from_two_points(pt2(0, 0), pt2(0, 1));
        assert!(dist_line2d_line2d(&l1, &l2).is_none());
    }
}
