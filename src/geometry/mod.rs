//! Geometry module — symbolic primitive types.
//!
//! Provides point types and primitive shapes whose coordinates and parameters
//! are all [`Arc<Expr>`] values. All computation remains symbolic; no numeric
//! evaluation occurs inside this module.
//!
//! # Modules
//!
//! - [`primitives`] — point types (`Point2D`, `Point3D`, `PointND`) and shape
//!   types (`Line2D`, `Plane3D`, `Circle2D`, `Sphere3D`, `Ellipse2D`).

pub mod primitives;

pub use primitives::{Circle2D, Ellipse2D, Line2D, Plane3D, Point2D, Point3D, PointND, Sphere3D};
