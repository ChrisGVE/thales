//! 2D and 3D Cartesian coordinate types and conversions.

use nalgebra::{Vector2, Vector3};
use num_complex::Complex64;
use std::f64::consts::PI;

use super::{Cylindrical, Polar, Spherical};

/// 2D Cartesian coordinates (x, y).
///
/// Represents a point in the 2D Cartesian plane with x and y coordinates.
/// Provides conversions to polar coordinates, complex numbers, and nalgebra vectors.
///
/// # Mathematical Representation
///
/// A point P in 2D Cartesian coordinates is represented as:
/// ```text
/// P = (x, y)
/// ```
///
/// # Examples
///
/// ```
/// use thales::transforms::Cartesian2D;
///
/// // Create a point at (3, 4)
/// let point = Cartesian2D::new(3.0, 4.0);
/// assert_eq!(point.x, 3.0);
/// assert_eq!(point.y, 4.0);
///
/// // Calculate distance from origin
/// let magnitude = point.magnitude();
/// assert!((magnitude - 5.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cartesian2D {
    /// x-coordinate (horizontal axis)
    pub x: f64,
    /// y-coordinate (vertical axis)
    pub y: f64,
}

impl Cartesian2D {
    /// Create new 2D Cartesian coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::transforms::Cartesian2D;
    ///
    /// let point = Cartesian2D::new(3.0, 4.0);
    /// assert_eq!(point.x, 3.0);
    /// assert_eq!(point.y, 4.0);
    /// ```
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Convert to polar coordinates.
    ///
    /// Converts Cartesian coordinates (x, y) to polar coordinates (r, θ) using:
    /// ```text
    /// r = √(x² + y²)
    /// θ = atan2(y, x)
    /// ```
    ///
    /// The angle θ is in radians, measured counterclockwise from the positive x-axis,
    /// and ranges from -π to π.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::transforms::Cartesian2D;
    /// use std::f64::consts::PI;
    ///
    /// // Point at (1, 1) should be at 45 degrees (π/4 radians)
    /// let point = Cartesian2D::new(1.0, 1.0);
    /// let polar = point.to_polar();
    /// assert!((polar.r - std::f64::consts::SQRT_2).abs() < 1e-10);
    /// assert!((polar.theta - PI / 4.0).abs() < 1e-10);
    ///
    /// // Point on negative x-axis
    /// let point = Cartesian2D::new(-2.0, 0.0);
    /// let polar = point.to_polar();
    /// assert!((polar.r - 2.0).abs() < 1e-10);
    /// assert!((polar.theta - PI).abs() < 1e-10);
    /// ```
    pub fn to_polar(&self) -> Polar {
        let r = (self.x * self.x + self.y * self.y).sqrt();
        let theta = self.y.atan2(self.x);
        Polar::new(r, theta)
    }

    /// Convert to complex number.
    ///
    /// Represents the Cartesian point (x, y) as the complex number x + yi.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::transforms::Cartesian2D;
    ///
    /// let point = Cartesian2D::new(3.0, 4.0);
    /// let complex = point.to_complex();
    /// assert_eq!(complex.re, 3.0);
    /// assert_eq!(complex.im, 4.0);
    /// ```
    pub fn to_complex(&self) -> Complex64 {
        Complex64::new(self.x, self.y)
    }

    /// Convert to nalgebra vector.
    ///
    /// Returns a 2D column vector `[x, y]ᵀ` for use with nalgebra linear algebra operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::transforms::Cartesian2D;
    ///
    /// let point = Cartesian2D::new(3.0, 4.0);
    /// let vec = point.to_vector();
    /// assert_eq!(vec[0], 3.0);
    /// assert_eq!(vec[1], 4.0);
    /// ```
    pub fn to_vector(&self) -> Vector2<f64> {
        Vector2::new(self.x, self.y)
    }

    /// Distance from origin.
    ///
    /// Calculates the Euclidean distance from the origin (0, 0) to the point (x, y):
    /// ```text
    /// |P| = √(x² + y²)
    /// ```
    ///
    /// This is equivalent to the radius r in polar coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::transforms::Cartesian2D;
    ///
    /// // 3-4-5 right triangle
    /// let point = Cartesian2D::new(3.0, 4.0);
    /// assert!((point.magnitude() - 5.0).abs() < 1e-10);
    ///
    /// // Point at origin
    /// let origin = Cartesian2D::new(0.0, 0.0);
    /// assert_eq!(origin.magnitude(), 0.0);
    /// ```
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}
/// 3D Cartesian coordinates (x, y, z).
///
/// Represents a point in 3D space using Cartesian coordinates with x, y, and z axes.
/// Provides conversions to spherical and cylindrical coordinate systems, and integration
/// with nalgebra's Vector3.
///
/// # Mathematical Representation
///
/// A point P in 3D Cartesian coordinates is represented as:
/// ```text
/// P = (x, y, z)
/// ```
///
/// # Coordinate System Diagram
///
/// ```text
///        z
///        ↑
///        |
///        |    P(x,y,z)
///        |   /
///        |  /
///        | /
///        |/________→ y
///       /
///      /
///     /
///    ↓
///    x
/// ```
///
/// # Examples
///
/// ```
/// use thales::transforms::Cartesian3D;
///
/// // Create a point at (3, 4, 5)
/// let point = Cartesian3D::new(3.0, 4.0, 5.0);
/// assert_eq!(point.x, 3.0);
/// assert_eq!(point.y, 4.0);
/// assert_eq!(point.z, 5.0);
///
/// // Calculate distance from origin
/// let magnitude = point.magnitude();
/// let expected = (3.0*3.0 + 4.0*4.0 + 5.0*5.0_f64).sqrt();
/// assert!((magnitude - expected).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cartesian3D {
    /// x-coordinate (horizontal axis)
    pub x: f64,
    /// y-coordinate (horizontal axis perpendicular to x)
    pub y: f64,
    /// z-coordinate (vertical axis)
    pub z: f64,
}

impl Cartesian3D {
    /// Create new 3D Cartesian coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::transforms::Cartesian3D;
    ///
    /// let point = Cartesian3D::new(1.0, 2.0, 3.0);
    /// assert_eq!(point.x, 1.0);
    /// assert_eq!(point.y, 2.0);
    /// assert_eq!(point.z, 3.0);
    /// ```
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Convert to spherical coordinates.
    ///
    /// Converts Cartesian coordinates (x, y, z) to spherical coordinates (r, θ, φ) using:
    /// ```text
    /// r = √(x² + y² + z²)
    /// θ = atan2(y, x)           [azimuthal angle in xy-plane]
    /// φ = acos(z / r)           [polar angle from z-axis]
    /// ```
    ///
    /// Uses the **physics convention** where:
    /// - r ≥ 0: radius (distance from origin)
    /// - θ ∈ [0, 2π): azimuthal angle in xy-plane from x-axis
    /// - φ ∈ [0, π]: polar angle from positive z-axis
    ///
    /// # Coordinate System Diagram
    ///
    /// ```text
    ///        z
    ///        ↑
    ///        |    P
    ///        |   /|
    ///        |  / |
    ///        | /  | r·cos(φ)
    ///      φ |/   |
    ///        O----●----------→ y
    ///       /     |
    ///      /    r·sin(φ)
    ///     /    θ
    ///    ↓
    ///    x
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::transforms::Cartesian3D;
    /// use std::f64::consts::PI;
    ///
    /// // Point on positive z-axis
    /// let point = Cartesian3D::new(0.0, 0.0, 5.0);
    /// let spherical = point.to_spherical();
    /// assert!((spherical.r - 5.0).abs() < 1e-10);
    /// assert!((spherical.phi - 0.0).abs() < 1e-10);  // φ = 0 on z-axis
    ///
    /// // Point in xy-plane at 45 degrees
    /// let point = Cartesian3D::new(1.0, 1.0, 0.0);
    /// let spherical = point.to_spherical();
    /// assert!((spherical.r - std::f64::consts::SQRT_2).abs() < 1e-10);
    /// assert!((spherical.theta - PI / 4.0).abs() < 1e-10);
    /// assert!((spherical.phi - PI / 2.0).abs() < 1e-10);  // φ = π/2 in xy-plane
    /// ```
    pub fn to_spherical(&self) -> Spherical {
        let r = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        let theta = (self.y.atan2(self.x)).rem_euclid(2.0 * PI); // azimuthal angle
        let phi = if r == 0.0 {
            0.0
        } else {
            (self.z / r).acos() // polar angle from z-axis
        };
        Spherical::new(r, theta, phi)
    }

    /// Convert to cylindrical coordinates.
    ///
    /// Converts Cartesian coordinates (x, y, z) to cylindrical coordinates (ρ, φ, z) using:
    /// ```text
    /// ρ = √(x² + y²)
    /// φ = atan2(y, x)
    /// z = z
    /// ```
    ///
    /// # Coordinate System Diagram
    ///
    /// ```text
    ///        z
    ///        ↑
    ///        |     P(x,y,z)
    ///        |    /|
    ///        |   / |
    ///        |  /  | z
    ///        | /   |
    ///        |/____●----------→ y
    ///       /      |
    ///      /     ρ (radial distance from z-axis)
    ///     /    φ
    ///    ↓
    ///    x
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::transforms::Cartesian3D;
    /// use std::f64::consts::PI;
    ///
    /// // Point at (3, 4, 5)
    /// let point = Cartesian3D::new(3.0, 4.0, 5.0);
    /// let cylindrical = point.to_cylindrical();
    /// assert!((cylindrical.rho - 5.0).abs() < 1e-10);  // √(3² + 4²) = 5
    /// assert!((cylindrical.z - 5.0).abs() < 1e-10);
    ///
    /// // Point on positive y-axis
    /// let point = Cartesian3D::new(0.0, 2.0, 3.0);
    /// let cylindrical = point.to_cylindrical();
    /// assert!((cylindrical.rho - 2.0).abs() < 1e-10);
    /// assert!((cylindrical.phi - PI / 2.0).abs() < 1e-10);
    /// assert!((cylindrical.z - 3.0).abs() < 1e-10);
    /// ```
    pub fn to_cylindrical(&self) -> Cylindrical {
        let rho = (self.x * self.x + self.y * self.y).sqrt();
        let phi = self.y.atan2(self.x);
        Cylindrical::new(rho, phi, self.z)
    }

    /// Convert to nalgebra vector.
    ///
    /// Returns a 3D column vector `[x, y, z]ᵀ` for use with nalgebra linear algebra operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::transforms::Cartesian3D;
    ///
    /// let point = Cartesian3D::new(1.0, 2.0, 3.0);
    /// let vec = point.to_vector();
    /// assert_eq!(vec[0], 1.0);
    /// assert_eq!(vec[1], 2.0);
    /// assert_eq!(vec[2], 3.0);
    /// ```
    pub fn to_vector(&self) -> Vector3<f64> {
        Vector3::new(self.x, self.y, self.z)
    }

    /// Distance from origin.
    ///
    /// Calculates the Euclidean distance from the origin (0, 0, 0) to the point (x, y, z):
    /// ```text
    /// |P| = √(x² + y² + z²)
    /// ```
    ///
    /// This is equivalent to the radius r in spherical coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::transforms::Cartesian3D;
    ///
    /// // 3-4-5 right triangle in 3D space
    /// let point = Cartesian3D::new(0.0, 3.0, 4.0);
    /// assert!((point.magnitude() - 5.0).abs() < 1e-10);
    ///
    /// // Point at origin
    /// let origin = Cartesian3D::new(0.0, 0.0, 0.0);
    /// assert_eq!(origin.magnitude(), 0.0);
    ///
    /// // 3D Pythagorean quintuple (1, 2, 2) → √9 = 3
    /// let point = Cartesian3D::new(1.0, 2.0, 2.0);
    /// assert!((point.magnitude() - 3.0).abs() < 1e-10);
    /// ```
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}
