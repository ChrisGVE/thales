//! Geometry module — symbolic primitive types, parametric curves/surfaces, and
//! 3-D affine/linear transformations.
//!
//! Provides point types, primitive shapes, parametric curve/surface types, and
//! symbolic 3-D transform functions. All coordinates and parameters are
//! [`Arc<Expr>`] values; no numeric evaluation occurs inside this module.
//!
//! # Modules
//!
//! - [`primitives`] — point types (`Point2D`, `Point3D`, `PointND`) and shape
//!   types (`Line2D`, `Plane3D`, `Circle2D`, `Sphere3D`, `Ellipse2D`).
//! - [`parametric`] — parametric curve (`ParametricCurve`) and surface
//!   (`ParametricSurface`) types with tangent, normal, curvature, and
//!   arc-length/surface-area integrand methods.
//! - [`transforms3d`] — 3-D rotation, reflection, scale, translation, and
//!   transform composition (`apply_3d`, `compose_3d`).

pub mod parametric;
pub mod primitives;
pub mod relations;
pub mod transforms3d;

pub use parametric::{ParametricCurve, ParametricSurface};
pub use primitives::{Circle2D, Ellipse2D, Line2D, Plane3D, Point2D, Point3D, PointND, Sphere3D};
pub use relations::{
    dist_line2d_line2d, dist_line3d_line3d, dist_line3d_plane3d, dist_plane3d_plane3d,
    dist_point2d_circle2d, dist_point2d_line2d, dist_point2d_point2d, dist_point3d_line3d,
    dist_point3d_plane3d, dist_point3d_point3d, dist_point3d_sphere3d, intersect_circle2d_circle2d,
    intersect_line2d_circle2d, intersect_line2d_line2d, intersect_line3d_line3d,
    intersect_line3d_plane3d, intersect_plane3d_plane3d, normal_to_circle2d,
    normal_to_parametric_curve, tangent_to_circle2d, tangent_to_parametric_curve, Line3D,
    LineLineIntersection2D, LineLineIntersection3D, LinePlaneIntersection, PlanePlaneIntersection,
    TangentLine,
};
pub use transforms3d::{
    apply_3d, compose_3d, reflection_3d, rotation_3d, scale_3d, translation_3d,
};
