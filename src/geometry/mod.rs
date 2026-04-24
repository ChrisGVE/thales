//! Geometry module — symbolic primitive types and parametric curves/surfaces.
//!
//! Provides point types, primitive shapes, and parametric curve/surface types
//! whose coordinates and parameters are all [`Arc<Expr>`] values. All
//! computation remains symbolic; no numeric evaluation occurs inside this
//! module.
//!
//! # Modules
//!
//! - [`primitives`] — point types (`Point2D`, `Point3D`, `PointND`) and shape
//!   types (`Line2D`, `Plane3D`, `Circle2D`, `Sphere3D`, `Ellipse2D`).
//! - [`parametric`] — parametric curve (`ParametricCurve`) and surface
//!   (`ParametricSurface`) types with tangent, normal, curvature, and
//!   arc-length/surface-area integrand methods.

pub mod parametric;
pub mod primitives;

pub use parametric::{ParametricCurve, ParametricSurface};
pub use primitives::{Circle2D, Ellipse2D, Line2D, Plane3D, Point2D, Point3D, PointND, Sphere3D};
