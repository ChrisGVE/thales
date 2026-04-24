//! Symbolic primitive geometry types.
//!
//! All coordinates and parameters are [`Arc<Expr>`] values, keeping every
//! quantity fully symbolic. No numeric evaluation occurs here.
//!
//! # Types
//!
//! | Type | Description |
//! |------|-------------|
//! | [`Point2D`] | 2-D point `(x, y)` |
//! | [`Point3D`] | 3-D point `(x, y, z)` |
//! | [`PointND`] | N-D point with arbitrary coordinate count |
//! | [`Line2D`] | Line in 2-D (point + direction vector) |
//! | [`Plane3D`] | Plane in 3-D (point + normal vector) |
//! | [`Circle2D`] | Circle in 2-D (center + radius) |
//! | [`Sphere3D`] | Sphere in 3-D (center + radius) |
//! | [`Ellipse2D`] | Ellipse in 2-D (center, semi-axes, rotation) |

use crate::numeric::{normalize, Expr};
use std::sync::Arc;

// ── Point2D ───────────────────────────────────────────────────────────────────

/// A symbolic point in the 2-D plane.
///
/// Both coordinates are [`Arc<Expr>`] so they can be numeric constants,
/// symbolic variables, or compound symbolic expressions.
///
/// # Examples
///
/// ```rust
/// use thales::geometry::Point2D;
/// use thales::numeric::Expr;
///
/// let p = Point2D { x: Expr::int(3), y: Expr::int(4) };
/// assert_eq!(p.x, Expr::int(3));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Point2D {
    /// Horizontal coordinate.
    pub x: Arc<Expr>,
    /// Vertical coordinate.
    pub y: Arc<Expr>,
}

impl Point2D {
    /// Construct a [`Point2D`] from two [`Arc<Expr>`] coordinates.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use thales::geometry::Point2D;
    /// use thales::numeric::Expr;
    ///
    /// let p = Point2D::new(Expr::symbol("a"), Expr::symbol("b"));
    /// ```
    pub fn new(x: Arc<Expr>, y: Arc<Expr>) -> Self {
        Self { x, y }
    }
}

// ── Point3D ───────────────────────────────────────────────────────────────────

/// A symbolic point in 3-D space.
///
/// # Examples
///
/// ```rust
/// use thales::geometry::Point3D;
/// use thales::numeric::Expr;
///
/// let p = Point3D::new(Expr::int(1), Expr::int(2), Expr::int(3));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Point3D {
    /// First coordinate (x / i component).
    pub x: Arc<Expr>,
    /// Second coordinate (y / j component).
    pub y: Arc<Expr>,
    /// Third coordinate (z / k component).
    pub z: Arc<Expr>,
}

impl Point3D {
    /// Construct a [`Point3D`] from three [`Arc<Expr>`] coordinates.
    pub fn new(x: Arc<Expr>, y: Arc<Expr>, z: Arc<Expr>) -> Self {
        Self { x, y, z }
    }
}

// ── PointND ───────────────────────────────────────────────────────────────────

/// A symbolic point in N-dimensional space.
///
/// The number of dimensions is determined by the length of `coords` at
/// construction time. There is no static dimension check; callers are
/// responsible for consistency when mixing `PointND` values.
///
/// # Examples
///
/// ```rust
/// use thales::geometry::PointND;
/// use thales::numeric::Expr;
///
/// let p = PointND::new(vec![Expr::int(1), Expr::int(2), Expr::int(3), Expr::int(4)]);
/// assert_eq!(p.dimension(), 4);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct PointND {
    /// Coordinate expressions, one per dimension.
    pub coords: Vec<Arc<Expr>>,
}

impl PointND {
    /// Construct a [`PointND`] from a `Vec` of coordinate expressions.
    pub fn new(coords: Vec<Arc<Expr>>) -> Self {
        Self { coords }
    }

    /// Return the number of dimensions (i.e. `coords.len()`).
    pub fn dimension(&self) -> usize {
        self.coords.len()
    }
}

// ── Line2D ────────────────────────────────────────────────────────────────────

/// A symbolic line in 2-D defined by a point and a direction vector.
///
/// The direction vector is stored as a [`Point2D`] whose `x` and `y` fields
/// hold the Δx and Δy components respectively.
///
/// # Examples
///
/// ```rust
/// use thales::geometry::{Line2D, Point2D};
/// use thales::numeric::Expr;
///
/// let p1 = Point2D::new(Expr::int(0), Expr::int(0));
/// let p2 = Point2D::new(Expr::int(1), Expr::int(1));
/// let line = Line2D::from_two_points(p1, p2);
/// // direction = (1 - 0, 1 - 0) = (1, 1)
/// assert_eq!(line.direction.x, Expr::int(1));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Line2D {
    /// A point that lies on the line.
    pub point: Point2D,
    /// Direction vector `(Δx, Δy)`.
    pub direction: Point2D,
}

impl Line2D {
    /// Construct a [`Line2D`] from two distinct points.
    ///
    /// The stored point is `p1`; the direction vector is
    /// `(p2.x − p1.x, p2.y − p1.y)` computed symbolically via
    /// [`normalize::sub`].
    pub fn from_two_points(p1: Point2D, p2: Point2D) -> Self {
        let dx = normalize::sub(p2.x.clone(), p1.x.clone());
        let dy = normalize::sub(p2.y.clone(), p1.y.clone());
        Self {
            point: p1,
            direction: Point2D::new(dx, dy),
        }
    }
}

// ── Plane3D ───────────────────────────────────────────────────────────────────

/// A symbolic plane in 3-D defined by a point and a normal vector.
///
/// The normal is stored as a [`Point3D`] whose fields hold the `(A, B, C)`
/// components of the plane equation `A(x−x₀) + B(y−y₀) + C(z−z₀) = 0`.
///
/// # Examples
///
/// ```rust
/// use thales::geometry::{Plane3D, Point3D};
/// use thales::numeric::Expr;
///
/// // xy-plane: z = 0
/// let o  = Point3D::new(Expr::int(0), Expr::int(0), Expr::int(0));
/// let px = Point3D::new(Expr::int(1), Expr::int(0), Expr::int(0));
/// let py = Point3D::new(Expr::int(0), Expr::int(1), Expr::int(0));
/// let plane = Plane3D::from_three_points(o, px, py);
/// // normal = (0, 0, 1) — z-axis direction
/// assert_eq!(plane.normal.z, Expr::int(1));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Plane3D {
    /// A point that lies on the plane.
    pub point: Point3D,
    /// Normal vector `(A, B, C)`.
    pub normal: Point3D,
}

impl Plane3D {
    /// Construct a [`Plane3D`] from three non-collinear points.
    ///
    /// Computes edge vectors `u = p2 − p1` and `v = p3 − p1` symbolically,
    /// then forms the normal via the symbolic cross product `u × v`:
    ///
    /// ```text
    /// n_x = u_y * v_z − u_z * v_y
    /// n_y = u_z * v_x − u_x * v_z
    /// n_z = u_x * v_y − u_y * v_x
    /// ```
    ///
    /// No normalization is applied; the normal is the raw cross product.
    pub fn from_three_points(p1: Point3D, p2: Point3D, p3: Point3D) -> Self {
        // Edge vectors
        let ux = normalize::sub(p2.x.clone(), p1.x.clone());
        let uy = normalize::sub(p2.y.clone(), p1.y.clone());
        let uz = normalize::sub(p2.z.clone(), p1.z.clone());

        let vx = normalize::sub(p3.x.clone(), p1.x.clone());
        let vy = normalize::sub(p3.y.clone(), p1.y.clone());
        let vz = normalize::sub(p3.z.clone(), p1.z.clone());

        // Symbolic cross product  u × v
        let nx = normalize::sub(
            normalize::mul(uy.clone(), vz.clone()),
            normalize::mul(uz.clone(), vy.clone()),
        );
        let ny = normalize::sub(
            normalize::mul(uz.clone(), vx.clone()),
            normalize::mul(ux.clone(), vz.clone()),
        );
        let nz = normalize::sub(
            normalize::mul(ux.clone(), vy.clone()),
            normalize::mul(uy.clone(), vx.clone()),
        );

        Self {
            point: p1,
            normal: Point3D::new(nx, ny, nz),
        }
    }
}

// ── Circle2D ──────────────────────────────────────────────────────────────────

/// A symbolic circle in the 2-D plane.
///
/// Defined by its center [`Point2D`] and a symbolic radius expression.
///
/// # Examples
///
/// ```rust
/// use thales::geometry::{Circle2D, Point2D};
/// use thales::numeric::Expr;
///
/// let c = Circle2D::new(
///     Point2D::new(Expr::int(0), Expr::int(0)),
///     Expr::symbol("r"),
/// );
/// assert_eq!(c.radius, Expr::symbol("r"));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Circle2D {
    /// Center point.
    pub center: Point2D,
    /// Radius (must be positive for a geometric circle, but no constraint
    /// is enforced symbolically).
    pub radius: Arc<Expr>,
}

impl Circle2D {
    /// Construct a [`Circle2D`] with the given center and radius.
    pub fn new(center: Point2D, radius: Arc<Expr>) -> Self {
        Self { center, radius }
    }
}

// ── Sphere3D ──────────────────────────────────────────────────────────────────

/// A symbolic sphere in 3-D space.
///
/// Defined by its center [`Point3D`] and a symbolic radius expression.
///
/// # Examples
///
/// ```rust
/// use thales::geometry::{Sphere3D, Point3D};
/// use thales::numeric::Expr;
///
/// let s = Sphere3D::new(
///     Point3D::new(Expr::int(0), Expr::int(0), Expr::int(0)),
///     Expr::symbol("R"),
/// );
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Sphere3D {
    /// Center point.
    pub center: Point3D,
    /// Radius.
    pub radius: Arc<Expr>,
}

impl Sphere3D {
    /// Construct a [`Sphere3D`] with the given center and radius.
    pub fn new(center: Point3D, radius: Arc<Expr>) -> Self {
        Self { center, radius }
    }
}

// ── Ellipse2D ─────────────────────────────────────────────────────────────────

/// A symbolic ellipse in the 2-D plane.
///
/// Defined by its center, semi-major axis `a`, semi-minor axis `b`, and a
/// rotation angle `θ` (in radians) that describes the tilt of the major axis
/// relative to the positive x-axis. All parameters are [`Arc<Expr>`].
///
/// # Examples
///
/// ```rust
/// use thales::geometry::{Ellipse2D, Point2D};
/// use thales::numeric::Expr;
///
/// let e = Ellipse2D::new(
///     Point2D::new(Expr::int(0), Expr::int(0)),
///     Expr::symbol("a"),
///     Expr::symbol("b"),
///     Expr::int(0),   // axis-aligned
/// );
/// assert_eq!(e.semi_major, Expr::symbol("a"));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Ellipse2D {
    /// Center of the ellipse.
    pub center: Point2D,
    /// Length of the semi-major axis.
    pub semi_major: Arc<Expr>,
    /// Length of the semi-minor axis.
    pub semi_minor: Arc<Expr>,
    /// Rotation of the major axis (radians, counter-clockwise from x-axis).
    pub rotation: Arc<Expr>,
}

impl Ellipse2D {
    /// Construct an [`Ellipse2D`] with the given parameters.
    pub fn new(
        center: Point2D,
        semi_major: Arc<Expr>,
        semi_minor: Arc<Expr>,
        rotation: Arc<Expr>,
    ) -> Self {
        Self {
            center,
            semi_major,
            semi_minor,
            rotation,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Point2D ──────────────────────────────────────────────────────────────

    /// Constructor stores x and y fields verbatim.
    #[test]
    fn test_point2d_new_stores_fields() {
        let x = Expr::int(3);
        let y = Expr::int(4);
        let p = Point2D::new(x.clone(), y.clone());
        assert_eq!(p.x, x);
        assert_eq!(p.y, y);
    }

    /// Struct literal and `new` constructor produce the same value.
    #[test]
    fn test_point2d_literal_matches_new() {
        let p1 = Point2D {
            x: Expr::int(1),
            y: Expr::int(2),
        };
        let p2 = Point2D::new(Expr::int(1), Expr::int(2));
        assert_eq!(p1, p2);
    }

    // ── Point3D ──────────────────────────────────────────────────────────────

    /// Constructor stores x, y and z fields verbatim.
    #[test]
    fn test_point3d_new_stores_fields() {
        let p = Point3D::new(Expr::int(1), Expr::int(2), Expr::int(3));
        assert_eq!(p.x, Expr::int(1));
        assert_eq!(p.y, Expr::int(2));
        assert_eq!(p.z, Expr::int(3));
    }

    /// Symbolic coordinates are preserved without evaluation.
    #[test]
    fn test_point3d_symbolic_coords() {
        let p = Point3D::new(Expr::symbol("a"), Expr::symbol("b"), Expr::symbol("c"));
        assert_eq!(p.x, Expr::symbol("a"));
        assert_eq!(p.y, Expr::symbol("b"));
        assert_eq!(p.z, Expr::symbol("c"));
    }

    // ── PointND ──────────────────────────────────────────────────────────────

    /// `dimension()` returns the length of `coords`.
    #[test]
    fn test_pointnd_dimension() {
        let p = PointND::new(vec![Expr::int(1), Expr::int(2), Expr::int(3), Expr::int(4)]);
        assert_eq!(p.dimension(), 4);
    }

    /// Zero-dimensional point has dimension 0.
    #[test]
    fn test_pointnd_empty_dimension() {
        let p = PointND::new(vec![]);
        assert_eq!(p.dimension(), 0);
    }

    // ── Line2D ───────────────────────────────────────────────────────────────

    /// `from_two_points` stores `p1` as the base point.
    #[test]
    fn test_line2d_base_point_is_p1() {
        let p1 = Point2D::new(Expr::int(2), Expr::int(3));
        let p2 = Point2D::new(Expr::int(5), Expr::int(7));
        let line = Line2D::from_two_points(p1.clone(), p2);
        assert_eq!(line.point.x, Expr::int(2));
        assert_eq!(line.point.y, Expr::int(3));
    }

    /// `from_two_points` computes direction as `p2 − p1` symbolically.
    ///
    /// With integer inputs the normalize layer reduces `5 - 2 → 3` and
    /// `7 - 3 → 4` at construction time.
    #[test]
    fn test_line2d_direction_from_two_integer_points() {
        let p1 = Point2D::new(Expr::int(2), Expr::int(3));
        let p2 = Point2D::new(Expr::int(5), Expr::int(7));
        let line = Line2D::from_two_points(p1, p2);
        assert_eq!(line.direction.x, Expr::int(3));
        assert_eq!(line.direction.y, Expr::int(4));
    }

    // ── Plane3D ──────────────────────────────────────────────────────────────

    /// `from_three_points` stores `p1` as the base point.
    #[test]
    fn test_plane3d_base_point_is_p1() {
        let o = Point3D::new(Expr::int(1), Expr::int(2), Expr::int(3));
        let px = Point3D::new(Expr::int(2), Expr::int(2), Expr::int(3));
        let py = Point3D::new(Expr::int(1), Expr::int(3), Expr::int(3));
        let plane = Plane3D::from_three_points(o.clone(), px, py);
        assert_eq!(plane.point.x, Expr::int(1));
        assert_eq!(plane.point.y, Expr::int(2));
        assert_eq!(plane.point.z, Expr::int(3));
    }

    /// Normal of the xy-plane (z = 0) is `(0, 0, 1)`.
    ///
    /// Three points in the xy-plane: O=(0,0,0), A=(1,0,0), B=(0,1,0).
    /// Edge vectors u=(1,0,0), v=(0,1,0). Cross product = (0,0,1).
    #[test]
    fn test_plane3d_xy_plane_normal_is_z_unit() {
        let o = Point3D::new(Expr::int(0), Expr::int(0), Expr::int(0));
        let a = Point3D::new(Expr::int(1), Expr::int(0), Expr::int(0));
        let b = Point3D::new(Expr::int(0), Expr::int(1), Expr::int(0));
        let plane = Plane3D::from_three_points(o, a, b);
        assert_eq!(plane.normal.x, Expr::int(0), "n_x should be 0");
        assert_eq!(plane.normal.y, Expr::int(0), "n_y should be 0");
        assert_eq!(plane.normal.z, Expr::int(1), "n_z should be 1");
    }

    /// Normal of the xz-plane (y = 0) is `(0, -1, 0)` (or `(0, 1, 0)` depending
    /// on orientation). We verify the two edge dot products with the normal are 0.
    ///
    /// Points in xz-plane: O=(0,0,0), A=(1,0,0), B=(0,0,1).
    /// u=(1,0,0), v=(0,0,1). u×v = (0*1−0*0, 0*0−1*1, 1*0−0*0) = (0,−1,0).
    #[test]
    fn test_plane3d_xz_plane_normal_orthogonal_to_edges() {
        let o = Point3D::new(Expr::int(0), Expr::int(0), Expr::int(0));
        let a = Point3D::new(Expr::int(1), Expr::int(0), Expr::int(0));
        let b = Point3D::new(Expr::int(0), Expr::int(0), Expr::int(1));
        let plane = Plane3D::from_three_points(o, a, b);

        // n = u × v; edge u = (1,0,0); edge v = (0,0,1)
        // n · u = n_x*1 + n_y*0 + n_z*0 = n_x
        // n · v = n_x*0 + n_y*0 + n_z*1 = n_z
        // Both must be 0.
        let n = &plane.normal;

        // Compute dot(n, u) = n.x * 1 + n.y * 0 + n.z * 0 = n.x
        let dot_nu = normalize::add(
            normalize::add(
                normalize::mul(n.x.clone(), Expr::int(1)),
                normalize::mul(n.y.clone(), Expr::int(0)),
            ),
            normalize::mul(n.z.clone(), Expr::int(0)),
        );

        // Compute dot(n, v) = n.x * 0 + n.y * 0 + n.z * 1 = n.z
        let dot_nv = normalize::add(
            normalize::add(
                normalize::mul(n.x.clone(), Expr::int(0)),
                normalize::mul(n.y.clone(), Expr::int(0)),
            ),
            normalize::mul(n.z.clone(), Expr::int(1)),
        );

        assert_eq!(dot_nu, Expr::int(0), "normal must be orthogonal to edge u");
        assert_eq!(dot_nv, Expr::int(0), "normal must be orthogonal to edge v");
    }

    // ── Circle2D ─────────────────────────────────────────────────────────────

    /// Constructor stores center and radius verbatim.
    #[test]
    fn test_circle2d_stores_fields() {
        let c = Point2D::new(Expr::int(1), Expr::int(2));
        let r = Expr::symbol("r");
        let circle = Circle2D::new(c.clone(), r.clone());
        assert_eq!(circle.center, c);
        assert_eq!(circle.radius, r);
    }

    /// Numeric center and radius are preserved without simplification.
    #[test]
    fn test_circle2d_numeric_radius() {
        let c = Point2D::new(Expr::int(0), Expr::int(0));
        let r = Expr::int(5);
        let circle = Circle2D::new(c, r.clone());
        assert_eq!(circle.radius, r);
    }

    // ── Sphere3D ─────────────────────────────────────────────────────────────

    /// Constructor stores center and radius verbatim.
    #[test]
    fn test_sphere3d_stores_fields() {
        let c = Point3D::new(Expr::int(0), Expr::int(0), Expr::int(0));
        let r = Expr::symbol("R");
        let sphere = Sphere3D::new(c.clone(), r.clone());
        assert_eq!(sphere.center, c);
        assert_eq!(sphere.radius, r);
    }

    /// Symbolic center coordinates are preserved.
    #[test]
    fn test_sphere3d_symbolic_center() {
        let cx = Expr::symbol("cx");
        let cy = Expr::symbol("cy");
        let cz = Expr::symbol("cz");
        let sphere = Sphere3D::new(
            Point3D::new(cx.clone(), cy.clone(), cz.clone()),
            Expr::int(1),
        );
        assert_eq!(sphere.center.x, cx);
        assert_eq!(sphere.center.y, cy);
        assert_eq!(sphere.center.z, cz);
    }

    // ── Ellipse2D ────────────────────────────────────────────────────────────

    /// Constructor stores all four fields verbatim.
    #[test]
    fn test_ellipse2d_stores_fields() {
        let center = Point2D::new(Expr::int(0), Expr::int(0));
        let a = Expr::symbol("a");
        let b = Expr::symbol("b");
        let theta = Expr::int(0);
        let ellipse = Ellipse2D::new(center.clone(), a.clone(), b.clone(), theta.clone());
        assert_eq!(ellipse.center, center);
        assert_eq!(ellipse.semi_major, a);
        assert_eq!(ellipse.semi_minor, b);
        assert_eq!(ellipse.rotation, theta);
    }

    /// An axis-aligned unit ellipse (a=1, b=1, θ=0) has both semi-axes equal.
    #[test]
    fn test_ellipse2d_unit_circle_special_case() {
        let e = Ellipse2D::new(
            Point2D::new(Expr::int(0), Expr::int(0)),
            Expr::int(1),
            Expr::int(1),
            Expr::int(0),
        );
        assert_eq!(
            e.semi_major, e.semi_minor,
            "unit circle: semi_major == semi_minor"
        );
    }
}
