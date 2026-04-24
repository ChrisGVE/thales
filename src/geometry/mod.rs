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
pub mod transforms3d;

pub use parametric::{ParametricCurve, ParametricSurface};
pub use primitives::{Circle2D, Ellipse2D, Line2D, Plane3D, Point2D, Point3D, PointND, Sphere3D};
pub use transforms3d::{
    apply_3d, compose_3d, reflection_3d, rotation_3d, scale_3d, translation_3d,
};
