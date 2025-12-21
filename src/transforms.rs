//! Coordinate system transformations.
//!
//! Provides transformations between different coordinate systems
//! (Cartesian, Polar, Spherical, Cylindrical) and complex number operations.

use nalgebra::{Matrix3, Vector2, Vector3};
use num_complex::Complex64;
use std::f64::consts::PI;

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
/// use mathsolver_core::transforms::Cartesian2D;
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
    /// use mathsolver_core::transforms::Cartesian2D;
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
    /// use mathsolver_core::transforms::Cartesian2D;
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
    /// use mathsolver_core::transforms::Cartesian2D;
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
    /// use mathsolver_core::transforms::Cartesian2D;
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
    /// use mathsolver_core::transforms::Cartesian2D;
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

/// Polar coordinates (r, θ).
///
/// Represents a point in the 2D plane using polar coordinates with radius r
/// and angle θ (theta) measured counterclockwise from the positive x-axis.
///
/// # Mathematical Representation
///
/// A point P in polar coordinates is represented as:
/// ```text
/// P = (r, θ)
/// where:
///   r ≥ 0 is the distance from the origin
///   θ is the angle in radians from the positive x-axis
/// ```
///
/// # Conversion Formulas
///
/// From Cartesian (x, y) to Polar (r, θ):
/// ```text
/// r = √(x² + y²)
/// θ = atan2(y, x)
/// ```
///
/// From Polar (r, θ) to Cartesian (x, y):
/// ```text
/// x = r cos(θ)
/// y = r sin(θ)
/// ```
///
/// # Examples
///
/// ```
/// use mathsolver_core::transforms::{Polar, Cartesian2D};
/// use std::f64::consts::PI;
///
/// // Point at radius 5, angle 45 degrees (π/4 radians)
/// let polar = Polar::new(5.0, PI / 4.0);
/// assert_eq!(polar.r, 5.0);
/// assert!((polar.theta - PI / 4.0).abs() < 1e-10);
///
/// // Convert to Cartesian
/// let cartesian = polar.to_cartesian();
/// assert!((cartesian.x - 5.0 * (PI / 4.0).cos()).abs() < 1e-10);
/// assert!((cartesian.y - 5.0 * (PI / 4.0).sin()).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Polar {
    /// Radius (distance from origin)
    pub r: f64,
    /// Angle in radians (counterclockwise from positive x-axis)
    pub theta: f64,
}

impl Polar {
    /// Create new polar coordinates.
    ///
    /// # Arguments
    ///
    /// * `r` - Radius (distance from origin), typically r ≥ 0
    /// * `theta` - Angle in radians, measured counterclockwise from positive x-axis
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::transforms::Polar;
    /// use std::f64::consts::PI;
    ///
    /// // Point at distance 3 from origin, 60 degrees (π/3 radians)
    /// let polar = Polar::new(3.0, PI / 3.0);
    /// assert_eq!(polar.r, 3.0);
    /// assert!((polar.theta - PI / 3.0).abs() < 1e-10);
    /// ```
    pub fn new(r: f64, theta: f64) -> Self {
        Self { r, theta }
    }

    /// Normalize angle to [0, 2π).
    ///
    /// Adjusts the angle θ to be in the standard range [0, 2π) radians.
    /// This is useful for comparing angles and for canonical representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::transforms::Polar;
    /// use std::f64::consts::PI;
    ///
    /// // Angle greater than 2π
    /// let mut polar = Polar::new(5.0, 3.0 * PI);
    /// polar.normalize_angle();
    /// assert!((polar.theta - PI).abs() < 1e-10);
    ///
    /// // Negative angle
    /// let mut polar = Polar::new(5.0, -PI / 2.0);
    /// polar.normalize_angle();
    /// assert!((polar.theta - 3.0 * PI / 2.0).abs() < 1e-10);
    /// ```
    pub fn normalize_angle(&mut self) {
        self.theta = self.theta.rem_euclid(2.0 * PI);
    }

    /// Convert to Cartesian coordinates.
    ///
    /// Converts polar coordinates (r, θ) to Cartesian coordinates (x, y) using:
    /// ```text
    /// x = r cos(θ)
    /// y = r sin(θ)
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::transforms::Polar;
    /// use std::f64::consts::PI;
    ///
    /// // Point at radius 2, angle 0 (positive x-axis)
    /// let polar = Polar::new(2.0, 0.0);
    /// let cartesian = polar.to_cartesian();
    /// assert!((cartesian.x - 2.0).abs() < 1e-10);
    /// assert!((cartesian.y - 0.0).abs() < 1e-10);
    ///
    /// // Point at radius 1, angle π/2 (positive y-axis)
    /// let polar = Polar::new(1.0, PI / 2.0);
    /// let cartesian = polar.to_cartesian();
    /// assert!((cartesian.x - 0.0).abs() < 1e-10);
    /// assert!((cartesian.y - 1.0).abs() < 1e-10);
    ///
    /// // Round-trip conversion
    /// let original = Polar::new(5.0, PI / 4.0);
    /// let cartesian = original.to_cartesian();
    /// let polar = cartesian.to_polar();
    /// assert!((polar.r - original.r).abs() < 1e-10);
    /// assert!((polar.theta - original.theta).abs() < 1e-10);
    /// ```
    pub fn to_cartesian(&self) -> Cartesian2D {
        let x = self.r * self.theta.cos();
        let y = self.r * self.theta.sin();
        Cartesian2D::new(x, y)
    }

    /// Convert to complex number (polar form).
    ///
    /// Represents the polar coordinates as a complex number in polar form:
    /// ```text
    /// z = r e^(iθ) = r(cos(θ) + i sin(θ))
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::transforms::Polar;
    /// use std::f64::consts::PI;
    ///
    /// let polar = Polar::new(5.0, PI / 6.0);
    /// let complex = polar.to_complex();
    /// assert!((complex.norm() - 5.0).abs() < 1e-10);
    /// assert!((complex.arg() - PI / 6.0).abs() < 1e-10);
    /// ```
    pub fn to_complex(&self) -> Complex64 {
        Complex64::from_polar(self.r, self.theta)
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
/// use mathsolver_core::transforms::Cartesian3D;
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
    /// use mathsolver_core::transforms::Cartesian3D;
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
    /// use mathsolver_core::transforms::Cartesian3D;
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
    /// use mathsolver_core::transforms::Cartesian3D;
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
    /// use mathsolver_core::transforms::Cartesian3D;
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
    /// use mathsolver_core::transforms::Cartesian3D;
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

/// Spherical coordinates (r, θ, φ).
///
/// Represents a point in 3D space using spherical coordinates with radius r,
/// azimuthal angle θ, and polar angle φ.
///
/// # Physics Convention
///
/// This implementation uses the **physics convention** (ISO 31-11), NOT the mathematics convention.
///
/// **Physics Convention** (used here):
/// - r ≥ 0: radial distance from origin
/// - θ ∈ [0, 2π): azimuthal angle in xy-plane from positive x-axis
/// - φ ∈ [0, π]: polar angle (inclination) from positive z-axis
///
/// **Mathematics Convention** (NOT used):
/// - r ≥ 0: radial distance
/// - θ ∈ [0, π]: polar angle from positive z-axis (equivalent to our φ)
/// - φ ∈ [0, 2π): azimuthal angle (equivalent to our θ)
///
/// # Coordinate System Diagram
///
/// ```text
///        z
///        ↑
///        |    P
///        |   /|
///        |  / |
///        | /  |
///      φ |/)r |
///        O----●-------→ y
///       /  θ  |
///      /      ρ (projection onto xy-plane)
///     /
///    ↓
///    x
///
/// where:
///   r = radius (distance OP)
///   θ = azimuthal angle (counterclockwise from x-axis in xy-plane)
///   φ = polar angle (angle from positive z-axis)
///   ρ = r·sin(φ) (projection of r onto xy-plane)
/// ```
///
/// # Conversion Formulas
///
/// From Cartesian (x, y, z) to Spherical (r, θ, φ):
/// ```text
/// r = √(x² + y² + z²)
/// θ = atan2(y, x)
/// φ = acos(z / r)
/// ```
///
/// From Spherical (r, θ, φ) to Cartesian (x, y, z):
/// ```text
/// x = r sin(φ) cos(θ)
/// y = r sin(φ) sin(θ)
/// z = r cos(φ)
/// ```
///
/// # Examples
///
/// ```
/// use mathsolver_core::transforms::{Spherical, Cartesian3D};
/// use std::f64::consts::PI;
///
/// // Point on positive z-axis at distance 5
/// let spherical = Spherical::new(5.0, 0.0, 0.0);  // φ = 0 points along +z
/// let cartesian = spherical.to_cartesian();
/// assert!((cartesian.x - 0.0).abs() < 1e-10);
/// assert!((cartesian.y - 0.0).abs() < 1e-10);
/// assert!((cartesian.z - 5.0).abs() < 1e-10);
///
/// // Point in xy-plane at 45 degrees from x-axis
/// let spherical = Spherical::new(2.0, PI / 4.0, PI / 2.0);  // φ = π/2 is xy-plane
/// let cartesian = spherical.to_cartesian();
/// assert!((cartesian.x - std::f64::consts::SQRT_2).abs() < 1e-10);
/// assert!((cartesian.y - std::f64::consts::SQRT_2).abs() < 1e-10);
/// assert!((cartesian.z - 0.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spherical {
    /// Radius (distance from origin), r ≥ 0
    pub r: f64,
    /// Azimuthal angle θ in radians (angle in xy-plane from x-axis), θ ∈ [0, 2π)
    pub theta: f64,
    /// Polar angle φ in radians (angle from positive z-axis), φ ∈ [0, π]
    pub phi: f64,
}

impl Spherical {
    /// Create new spherical coordinates.
    ///
    /// # Arguments
    ///
    /// * `r` - Radius (distance from origin), typically r ≥ 0
    /// * `theta` - Azimuthal angle in radians (angle in xy-plane from x-axis)
    /// * `phi` - Polar angle in radians (angle from positive z-axis)
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::transforms::Spherical;
    /// use std::f64::consts::PI;
    ///
    /// // Point at radius 5, 30° azimuthal, 45° polar
    /// let spherical = Spherical::new(5.0, PI / 6.0, PI / 4.0);
    /// assert_eq!(spherical.r, 5.0);
    /// assert!((spherical.theta - PI / 6.0).abs() < 1e-10);
    /// assert!((spherical.phi - PI / 4.0).abs() < 1e-10);
    /// ```
    pub fn new(r: f64, theta: f64, phi: f64) -> Self {
        Self { r, theta, phi }
    }

    /// Convert to Cartesian coordinates.
    ///
    /// Converts spherical coordinates (r, θ, φ) to Cartesian coordinates (x, y, z) using:
    /// ```text
    /// x = r sin(φ) cos(θ)
    /// y = r sin(φ) sin(θ)
    /// z = r cos(φ)
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::transforms::Spherical;
    /// use std::f64::consts::PI;
    ///
    /// // Point at radius 1 on positive x-axis
    /// let spherical = Spherical::new(1.0, 0.0, PI / 2.0);
    /// let cartesian = spherical.to_cartesian();
    /// assert!((cartesian.x - 1.0).abs() < 1e-10);
    /// assert!((cartesian.y - 0.0).abs() < 1e-10);
    /// assert!((cartesian.z - 0.0).abs() < 1e-10);
    ///
    /// // Point at radius 1 on positive z-axis
    /// let spherical = Spherical::new(1.0, 0.0, 0.0);
    /// let cartesian = spherical.to_cartesian();
    /// assert!((cartesian.x - 0.0).abs() < 1e-10);
    /// assert!((cartesian.y - 0.0).abs() < 1e-10);
    /// assert!((cartesian.z - 1.0).abs() < 1e-10);
    ///
    /// // Round-trip conversion
    /// let original = Spherical::new(3.0, PI / 3.0, PI / 6.0);
    /// let cartesian = original.to_cartesian();
    /// let spherical = cartesian.to_spherical();
    /// assert!((spherical.r - original.r).abs() < 1e-10);
    /// assert!((spherical.theta - original.theta).abs() < 1e-10);
    /// assert!((spherical.phi - original.phi).abs() < 1e-10);
    /// ```
    pub fn to_cartesian(&self) -> Cartesian3D {
        let x = self.r * self.phi.sin() * self.theta.cos();
        let y = self.r * self.phi.sin() * self.theta.sin();
        let z = self.r * self.phi.cos();
        Cartesian3D::new(x, y, z)
    }

    /// Convert to cylindrical coordinates.
    ///
    /// Converts spherical coordinates (r, θ, φ) to cylindrical coordinates (ρ, φ_cyl, z) using:
    /// ```text
    /// ρ = r sin(φ)
    /// φ_cyl = θ
    /// z = r cos(φ)
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::transforms::Spherical;
    /// use std::f64::consts::PI;
    ///
    /// // Point at radius 2, θ=30°, φ=60°
    /// let spherical = Spherical::new(2.0, PI / 6.0, PI / 3.0);
    /// let cylindrical = spherical.to_cylindrical();
    /// assert!((cylindrical.rho - 2.0 * (PI / 3.0).sin()).abs() < 1e-10);
    /// assert!((cylindrical.phi - PI / 6.0).abs() < 1e-10);
    /// assert!((cylindrical.z - 2.0 * (PI / 3.0).cos()).abs() < 1e-10);
    ///
    /// // Point in xy-plane (φ = π/2)
    /// let spherical = Spherical::new(5.0, PI / 4.0, PI / 2.0);
    /// let cylindrical = spherical.to_cylindrical();
    /// assert!((cylindrical.rho - 5.0).abs() < 1e-10);
    /// assert!((cylindrical.z - 0.0).abs() < 1e-10);
    /// ```
    pub fn to_cylindrical(&self) -> Cylindrical {
        let rho = self.r * self.phi.sin();
        Cylindrical::new(rho, self.theta, self.r * self.phi.cos())
    }
}

/// Cylindrical coordinates (ρ, φ, z).
///
/// Represents a point in 3D space using cylindrical coordinates with radial distance ρ
/// from the z-axis, azimuthal angle φ, and height z along the z-axis.
///
/// # Mathematical Representation
///
/// A point P in cylindrical coordinates is represented as:
/// ```text
/// P = (ρ, φ, z)
/// where:
///   ρ ≥ 0 is the radial distance from the z-axis (radius in xy-plane)
///   φ is the azimuthal angle in radians from the positive x-axis
///   z is the height along the z-axis
/// ```
///
/// # Coordinate System Diagram
///
/// ```text
///        z
///        ↑
///        |     P(ρ,φ,z)
///        |    /|
///        |   / |
///        |  /  | z (height)
///        | /   |
///        |/____|_____→ y
///       /      ●
///      /       |
///     /        ρ (radial distance from z-axis)
///    ↓       φ
///    x
///
/// Top view (looking down z-axis):
///
///      y
///      ↑
///      |    P
///      |   /
///      |  /
///      | / ρ
///    φ |/)
///      |/________→ x
/// ```
///
/// # Conversion Formulas
///
/// From Cartesian (x, y, z) to Cylindrical (ρ, φ, z):
/// ```text
/// ρ = √(x² + y²)
/// φ = atan2(y, x)
/// z = z
/// ```
///
/// From Cylindrical (ρ, φ, z) to Cartesian (x, y, z):
/// ```text
/// x = ρ cos(φ)
/// y = ρ sin(φ)
/// z = z
/// ```
///
/// From Cylindrical (ρ, φ, z) to Spherical (r, θ, φ_sph):
/// ```text
/// r = √(ρ² + z²)
/// θ = φ
/// φ_sph = acos(z / r)
/// ```
///
/// # Examples
///
/// ```
/// use mathsolver_core::transforms::{Cylindrical, Cartesian3D};
/// use std::f64::consts::PI;
///
/// // Point at radius 3 from z-axis, angle 60°, height 4
/// let cylindrical = Cylindrical::new(3.0, PI / 3.0, 4.0);
/// assert_eq!(cylindrical.rho, 3.0);
/// assert!((cylindrical.phi - PI / 3.0).abs() < 1e-10);
/// assert_eq!(cylindrical.z, 4.0);
///
/// // Convert to Cartesian
/// let cartesian = cylindrical.to_cartesian();
/// assert!((cartesian.x - 3.0 * (PI / 3.0).cos()).abs() < 1e-10);
/// assert!((cartesian.y - 3.0 * (PI / 3.0).sin()).abs() < 1e-10);
/// assert!((cartesian.z - 4.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cylindrical {
    /// Radial distance from z-axis (radius in xy-plane), ρ ≥ 0
    pub rho: f64,
    /// Azimuthal angle in radians (angle in xy-plane from x-axis)
    pub phi: f64,
    /// Height along z-axis
    pub z: f64,
}

impl Cylindrical {
    /// Create new cylindrical coordinates.
    ///
    /// # Arguments
    ///
    /// * `rho` - Radial distance from z-axis, typically ρ ≥ 0
    /// * `phi` - Azimuthal angle in radians (angle in xy-plane from x-axis)
    /// * `z` - Height along z-axis
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::transforms::Cylindrical;
    /// use std::f64::consts::PI;
    ///
    /// // Point at radius 2 from z-axis, 45°, height 3
    /// let cylindrical = Cylindrical::new(2.0, PI / 4.0, 3.0);
    /// assert_eq!(cylindrical.rho, 2.0);
    /// assert!((cylindrical.phi - PI / 4.0).abs() < 1e-10);
    /// assert_eq!(cylindrical.z, 3.0);
    /// ```
    pub fn new(rho: f64, phi: f64, z: f64) -> Self {
        Self { rho, phi, z }
    }

    /// Convert to Cartesian coordinates.
    ///
    /// Converts cylindrical coordinates (ρ, φ, z) to Cartesian coordinates (x, y, z) using:
    /// ```text
    /// x = ρ cos(φ)
    /// y = ρ sin(φ)
    /// z = z
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::transforms::Cylindrical;
    /// use std::f64::consts::PI;
    ///
    /// // Point at radius 5 on positive x-axis, height 2
    /// let cylindrical = Cylindrical::new(5.0, 0.0, 2.0);
    /// let cartesian = cylindrical.to_cartesian();
    /// assert!((cartesian.x - 5.0).abs() < 1e-10);
    /// assert!((cartesian.y - 0.0).abs() < 1e-10);
    /// assert!((cartesian.z - 2.0).abs() < 1e-10);
    ///
    /// // Point at radius 2 on positive y-axis, height 3
    /// let cylindrical = Cylindrical::new(2.0, PI / 2.0, 3.0);
    /// let cartesian = cylindrical.to_cartesian();
    /// assert!((cartesian.x - 0.0).abs() < 1e-10);
    /// assert!((cartesian.y - 2.0).abs() < 1e-10);
    /// assert!((cartesian.z - 3.0).abs() < 1e-10);
    ///
    /// // Round-trip conversion
    /// let original = Cylindrical::new(4.0, PI / 6.0, 5.0);
    /// let cartesian = original.to_cartesian();
    /// let cylindrical = cartesian.to_cylindrical();
    /// assert!((cylindrical.rho - original.rho).abs() < 1e-10);
    /// assert!((cylindrical.phi - original.phi).abs() < 1e-10);
    /// assert!((cylindrical.z - original.z).abs() < 1e-10);
    /// ```
    pub fn to_cartesian(&self) -> Cartesian3D {
        let x = self.rho * self.phi.cos();
        let y = self.rho * self.phi.sin();
        Cartesian3D::new(x, y, self.z)
    }

    /// Convert to spherical coordinates.
    ///
    /// Converts cylindrical coordinates (ρ, φ, z) to spherical coordinates (r, θ, φ_sph) using:
    /// ```text
    /// r = √(ρ² + z²)
    /// θ = φ
    /// φ_sph = acos(z / r)
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::transforms::Cylindrical;
    /// use std::f64::consts::PI;
    ///
    /// // Point at radius 3, angle 30°, height 4
    /// let cylindrical = Cylindrical::new(3.0, PI / 6.0, 4.0);
    /// let spherical = cylindrical.to_spherical();
    /// assert!((spherical.r - 5.0).abs() < 1e-10);  // √(3² + 4²) = 5
    /// assert!((spherical.theta - PI / 6.0).abs() < 1e-10);
    ///
    /// // Point in xy-plane (z = 0)
    /// let cylindrical = Cylindrical::new(2.0, PI / 4.0, 0.0);
    /// let spherical = cylindrical.to_spherical();
    /// assert!((spherical.r - 2.0).abs() < 1e-10);
    /// assert!((spherical.phi - PI / 2.0).abs() < 1e-10);  // φ = π/2 for z=0
    /// ```
    pub fn to_spherical(&self) -> Spherical {
        let r = (self.rho * self.rho + self.z * self.z).sqrt();
        let phi = if r == 0.0 { 0.0 } else { (self.z / r).acos() };
        Spherical::new(r, self.phi, phi)
    }
}

/// Complex number operations and transformations.
///
/// Provides utilities for working with complex numbers including conversions between
/// Cartesian (x + yi) and polar (r∠θ) forms, as well as operations leveraging polar
/// representation such as De Moivre's theorem for computing powers and roots.
///
/// # Complex Number Representations
///
/// **Cartesian form**: z = x + yi
/// - Real part: x = Re(z)
/// - Imaginary part: y = Im(z)
/// - From num_complex::Complex64
///
/// **Polar form**: z = r∠θ = r e^(iθ)
/// - Magnitude: r = |z| = √(x² + y²)
/// - Argument: θ = arg(z) = atan2(y, x)
/// - Euler's formula: e^(iθ) = cos(θ) + i sin(θ)
///
/// # Integration with num_complex
///
/// This struct provides convenience methods that integrate with the `num_complex::Complex64`
/// type and the [`Polar`] coordinate struct, enabling seamless conversions and operations
/// across both representations.
///
/// # Applications
///
/// Complex number operations in polar form are essential for:
/// - **Signal processing**: Frequency domain analysis, Fourier transforms
/// - **Electrical engineering**: AC circuit analysis (impedance, phasors)
/// - **Control systems**: Transfer functions, stability analysis
/// - **Quantum mechanics**: Wave functions, probability amplitudes
/// - **Computer graphics**: Rotations and transformations in 2D
///
/// # Examples
///
/// ```
/// use mathsolver_core::transforms::ComplexOps;
/// use num_complex::Complex64;
/// use std::f64::consts::PI;
///
/// // Convert complex number to polar form
/// let z = Complex64::new(3.0, 4.0);
/// let polar = ComplexOps::to_polar(z);
/// assert!((polar.r - 5.0).abs() < 1e-10);  // magnitude = √(3² + 4²) = 5
/// assert!((polar.theta - (4.0_f64).atan2(3.0)).abs() < 1e-10);
///
/// // Convert back from polar to Cartesian
/// let z_back = ComplexOps::from_polar(polar);
/// assert!((z_back.re - 3.0).abs() < 1e-10);
/// assert!((z_back.im - 4.0).abs() < 1e-10);
///
/// // Compute power using De Moivre's theorem
/// let z = Complex64::new(1.0, 1.0);  // 1 + i
/// let z_cubed = ComplexOps::de_moivre(z, 3.0);
/// // (1+i)³ = -2 + 2i
/// assert!((z_cubed.re - -2.0).abs() < 1e-10);
/// assert!((z_cubed.im - 2.0).abs() < 1e-10);
/// ```
pub struct ComplexOps;

impl ComplexOps {
    /// Convert complex number to polar form (r, θ).
    ///
    /// Converts a complex number from Cartesian form (x + yi) to polar form (r∠θ).
    ///
    /// # Conversion Formulas
    ///
    /// ```text
    /// r = |z| = √(x² + y²)    (magnitude)
    /// θ = arg(z) = atan2(y, x) (argument)
    /// ```
    ///
    /// The argument θ is in radians, measured counterclockwise from the positive real axis,
    /// and ranges from -π to π.
    ///
    /// # Arguments
    ///
    /// * `c` - Complex number in Cartesian form (num_complex::Complex64)
    ///
    /// # Returns
    ///
    /// [`Polar`] coordinates (r, θ) representing the same complex number
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::transforms::ComplexOps;
    /// use num_complex::Complex64;
    /// use std::f64::consts::PI;
    ///
    /// // Convert 1 + i to polar form
    /// let z = Complex64::new(1.0, 1.0);
    /// let polar = ComplexOps::to_polar(z);
    /// assert!((polar.r - std::f64::consts::SQRT_2).abs() < 1e-10);
    /// assert!((polar.theta - PI / 4.0).abs() < 1e-10);  // 45 degrees
    ///
    /// // Pure real number
    /// let z = Complex64::new(5.0, 0.0);
    /// let polar = ComplexOps::to_polar(z);
    /// assert!((polar.r - 5.0).abs() < 1e-10);
    /// assert!((polar.theta - 0.0).abs() < 1e-10);
    ///
    /// // Pure imaginary number
    /// let z = Complex64::new(0.0, 3.0);
    /// let polar = ComplexOps::to_polar(z);
    /// assert!((polar.r - 3.0).abs() < 1e-10);
    /// assert!((polar.theta - PI / 2.0).abs() < 1e-10);  // 90 degrees
    ///
    /// // Negative real number (angle = π)
    /// let z = Complex64::new(-2.0, 0.0);
    /// let polar = ComplexOps::to_polar(z);
    /// assert!((polar.r - 2.0).abs() < 1e-10);
    /// assert!((polar.theta - PI).abs() < 1e-10);  // 180 degrees
    /// ```
    pub fn to_polar(c: Complex64) -> Polar {
        Polar::new(c.norm(), c.arg())
    }

    /// Convert polar form to complex number.
    ///
    /// Converts polar coordinates (r∠θ) to Cartesian form (x + yi).
    ///
    /// # Conversion Formulas
    ///
    /// Using Euler's formula:
    /// ```text
    /// z = r e^(iθ) = r(cos(θ) + i sin(θ))
    /// x = r cos(θ)  (real part)
    /// y = r sin(θ)  (imaginary part)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `p` - [`Polar`] coordinates (r, θ)
    ///
    /// # Returns
    ///
    /// Complex number in Cartesian form (num_complex::Complex64)
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::transforms::{ComplexOps, Polar};
    /// use std::f64::consts::PI;
    ///
    /// // Convert polar form to Cartesian
    /// let polar = Polar::new(5.0, PI / 3.0);  // r=5, θ=60°
    /// let z = ComplexOps::from_polar(polar);
    /// assert!((z.re - 5.0 * (PI / 3.0).cos()).abs() < 1e-10);
    /// assert!((z.im - 5.0 * (PI / 3.0).sin()).abs() < 1e-10);
    ///
    /// // Round-trip conversion
    /// let original = Polar::new(3.0, PI / 4.0);
    /// let complex = ComplexOps::from_polar(original);
    /// let polar = ComplexOps::to_polar(complex);
    /// assert!((polar.r - original.r).abs() < 1e-10);
    /// assert!((polar.theta - original.theta).abs() < 1e-10);
    ///
    /// // Unit circle point at 90 degrees
    /// let polar = Polar::new(1.0, PI / 2.0);
    /// let z = ComplexOps::from_polar(polar);
    /// assert!((z.re - 0.0).abs() < 1e-10);
    /// assert!((z.im - 1.0).abs() < 1e-10);
    /// ```
    pub fn from_polar(p: Polar) -> Complex64 {
        Complex64::from_polar(p.r, p.theta)
    }

    /// De Moivre's theorem: (r∠θ)^n = r^n∠(nθ).
    ///
    /// Computes powers (or fractional powers for roots) of complex numbers using
    /// De Moivre's theorem, which states that raising a complex number to a power
    /// in polar form multiplies the magnitude by the power and the angle by the power.
    ///
    /// # Mathematical Background
    ///
    /// **De Moivre's Theorem**: For a complex number z = r∠θ and any real number n:
    /// ```text
    /// z^n = (r∠θ)^n = r^n∠(nθ)
    /// ```
    ///
    /// In Cartesian form, this is equivalent to:
    /// ```text
    /// (r(cos(θ) + i sin(θ)))^n = r^n(cos(nθ) + i sin(nθ))
    /// ```
    ///
    /// **Special Cases**:
    /// - Integer powers: (1+i)² = 2i, (1+i)³ = -2+2i
    /// - Fractional powers (roots): z^(1/n) gives the principal nth root
    /// - Negative powers: z^(-1) = 1/z (multiplicative inverse)
    ///
    /// # Applications
    ///
    /// - **Trigonometric identities**: Deriving formulas like cos(3θ) = 4cos³(θ) - 3cos(θ)
    /// - **Signal processing**: Computing harmonics and frequency components
    /// - **Electrical engineering**: AC circuit analysis with complex impedances
    /// - **Quantum mechanics**: Time evolution of wave functions
    ///
    /// # Arguments
    ///
    /// * `c` - Complex number to raise to power (num_complex::Complex64)
    /// * `n` - Exponent (can be integer, fractional, or negative)
    ///
    /// # Returns
    ///
    /// Result of c^n in Cartesian form
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::transforms::ComplexOps;
    /// use num_complex::Complex64;
    /// use std::f64::consts::PI;
    ///
    /// // Square of 1+i: (1+i)² = 2i
    /// let z = Complex64::new(1.0, 1.0);
    /// let z_squared = ComplexOps::de_moivre(z, 2.0);
    /// assert!((z_squared.re - 0.0).abs() < 1e-10);
    /// assert!((z_squared.im - 2.0).abs() < 1e-10);
    ///
    /// // Cube of 1+i: (1+i)³ = -2+2i
    /// let z_cubed = ComplexOps::de_moivre(z, 3.0);
    /// assert!((z_cubed.re - -2.0).abs() < 1e-10);
    /// assert!((z_cubed.im - 2.0).abs() < 1e-10);
    ///
    /// // Square root (principal): (1+i)^(1/2)
    /// let z_sqrt = ComplexOps::de_moivre(z, 0.5);
    /// // Verify by squaring: z_sqrt² should equal z
    /// let z_back = ComplexOps::de_moivre(z_sqrt, 2.0);
    /// assert!((z_back.re - z.re).abs() < 1e-10);
    /// assert!((z_back.im - z.im).abs() < 1e-10);
    ///
    /// // Fourth power of unit circle point at 45°
    /// let z = Complex64::from_polar(1.0, PI / 4.0);  // e^(iπ/4)
    /// let z_fourth = ComplexOps::de_moivre(z, 4.0);   // e^(iπ) = -1
    /// assert!((z_fourth.re - -1.0).abs() < 1e-10);
    /// assert!((z_fourth.im - 0.0).abs() < 1e-10);
    ///
    /// // Negative power (reciprocal): (2+0i)^(-1) = 0.5
    /// let z = Complex64::new(2.0, 0.0);
    /// let z_inv = ComplexOps::de_moivre(z, -1.0);
    /// assert!((z_inv.re - 0.5).abs() < 1e-10);
    /// assert!((z_inv.im - 0.0).abs() < 1e-10);
    /// ```
    pub fn de_moivre(c: Complex64, n: f64) -> Complex64 {
        let polar = Self::to_polar(c);
        let r_n = polar.r.powf(n);
        let theta_n = polar.theta * n;
        Complex64::from_polar(r_n, theta_n)
    }

    /// Compute all nth roots of a complex number.
    ///
    /// **TODO**: Not yet implemented. Returns empty vector as placeholder.
    ///
    /// When implemented, will find all n distinct nth roots of a complex number.
    /// For any complex number z and positive integer n, there are exactly n
    /// distinct nth roots equally spaced around a circle in the complex plane.
    ///
    /// # Mathematical Background
    ///
    /// For z = r∠θ, the n distinct nth roots are:
    /// ```text
    /// z_k = r^(1/n) ∠ ((θ + 2πk) / n)  for k = 0, 1, 2, ..., n-1
    /// ```
    ///
    /// These roots are evenly distributed at angles 2π/n radians apart on a circle
    /// of radius r^(1/n) centered at the origin.
    ///
    /// # Examples (when implemented)
    ///
    /// ```ignore
    /// use mathsolver_core::transforms::ComplexOps;
    /// use num_complex::Complex64;
    ///
    /// // Find all cube roots of 8 (real number)
    /// let z = Complex64::new(8.0, 0.0);
    /// let roots = ComplexOps::nth_root(z, 3);
    /// assert_eq!(roots.len(), 3);
    /// // Roots: 2, -1+√3i, -1-√3i
    ///
    /// // Find all square roots of i
    /// let z = Complex64::new(0.0, 1.0);
    /// let roots = ComplexOps::nth_root(z, 2);
    /// assert_eq!(roots.len(), 2);
    /// // Roots: (1+i)/√2, -(1+i)/√2
    /// ```
    ///
    /// # Arguments
    ///
    /// * `_c` - Complex number to find roots of
    /// * `_n` - Root degree (positive integer)
    ///
    /// # Returns
    ///
    /// Vector of all n distinct nth roots (empty until implemented)
    pub fn nth_root(_c: Complex64, _n: i32) -> Vec<Complex64> {
        // TODO: Implement all n roots
        // Returns n equally spaced roots around the circle
        vec![]
    }
}

/// Homogeneous transformation matrix for 2D transformations.
///
/// Represents 2D transformations (translation, rotation, scaling) using
/// homogeneous coordinates and a 3×3 transformation matrix.
///
/// # Mathematical Representation
///
/// A point (x, y) in homogeneous coordinates is represented as [x, y, 1]ᵀ.
/// Transformations are represented as 3×3 matrices:
///
/// ```text
/// ┌         ┐   ┌   ┐   ┌    ┐
/// │ a  b  tx│   │ x │   │ x' │
/// │ c  d  ty│ × │ y │ = │ y' │
/// │ 0  0  1 │   │ 1 │   │ 1  │
/// └         ┘   └   ┘   └    ┘
/// ```
///
/// # Common Transformations
///
/// **Translation** by (dx, dy):
/// ```text
/// ┌         ┐
/// │ 1  0  dx│
/// │ 0  1  dy│
/// │ 0  0  1 │
/// └         ┘
/// ```
///
/// **Rotation** by θ radians counterclockwise:
/// ```text
/// ┌                    ┐
/// │ cos(θ)  -sin(θ)  0 │
/// │ sin(θ)   cos(θ)  0 │
/// │   0        0     1 │
/// └                    ┘
/// ```
///
/// **Scaling** by (sx, sy):
/// ```text
/// ┌         ┐
/// │ sx  0  0│
/// │ 0  sy  0│
/// │ 0   0  1│
/// └         ┘
/// ```
///
/// # Integration with nalgebra
///
/// This struct wraps nalgebra's `Matrix3<f64>` for efficient linear algebra
/// operations. Transformations can be composed through matrix multiplication.
///
/// # Examples
///
/// ```
/// use mathsolver_core::transforms::{Transform2D, Cartesian2D};
///
/// // Identity transformation leaves points unchanged
/// let identity = Transform2D::identity();
/// let point = Cartesian2D::new(3.0, 4.0);
/// let transformed = identity.apply(point);
/// assert_eq!(transformed.x, point.x);
/// assert_eq!(transformed.y, point.y);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform2D {
    matrix: Matrix3<f64>,
}

impl Transform2D {
    /// Identity transformation.
    ///
    /// Creates the identity transformation that leaves all points unchanged.
    /// The identity matrix is:
    /// ```text
    /// ┌       ┐
    /// │ 1 0 0 │
    /// │ 0 1 0 │
    /// │ 0 0 1 │
    /// └       ┘
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::transforms::{Transform2D, Cartesian2D};
    ///
    /// let identity = Transform2D::identity();
    /// let point = Cartesian2D::new(5.0, -3.0);
    /// let result = identity.apply(point);
    /// assert_eq!(result.x, 5.0);
    /// assert_eq!(result.y, -3.0);
    /// ```
    pub fn identity() -> Self {
        Self {
            matrix: Matrix3::identity(),
        }
    }

    /// Translation transformation.
    ///
    /// **TODO**: Not yet implemented. Returns identity transformation as placeholder.
    ///
    /// When implemented, will create a transformation that translates points by (dx, dy):
    /// ```text
    /// (x, y) → (x + dx, y + dy)
    /// ```
    ///
    /// Matrix form:
    /// ```text
    /// ┌         ┐
    /// │ 1  0  dx│
    /// │ 0  1  dy│
    /// │ 0  0  1 │
    /// └         ┘
    /// ```
    ///
    /// # Arguments
    ///
    /// * `_dx` - Translation in x direction
    /// * `_dy` - Translation in y direction
    pub fn translation(_dx: f64, _dy: f64) -> Self {
        // TODO: Implement 2D translation matrix
        Self::identity()
    }

    /// Rotation transformation (angle in radians).
    ///
    /// **TODO**: Not yet implemented. Returns identity transformation as placeholder.
    ///
    /// When implemented, will create a transformation that rotates points
    /// counterclockwise by θ radians around the origin:
    /// ```text
    /// x' = x cos(θ) - y sin(θ)
    /// y' = x sin(θ) + y cos(θ)
    /// ```
    ///
    /// Matrix form:
    /// ```text
    /// ┌                    ┐
    /// │ cos(θ)  -sin(θ)  0 │
    /// │ sin(θ)   cos(θ)  0 │
    /// │   0        0     1 │
    /// └                    ┘
    /// ```
    ///
    /// # Arguments
    ///
    /// * `_theta` - Rotation angle in radians (counterclockwise)
    pub fn rotation(_theta: f64) -> Self {
        // TODO: Implement 2D rotation matrix
        Self::identity()
    }

    /// Scaling transformation.
    ///
    /// **TODO**: Not yet implemented. Returns identity transformation as placeholder.
    ///
    /// When implemented, will create a transformation that scales points
    /// by (sx, sy) in the x and y directions:
    /// ```text
    /// (x, y) → (sx·x, sy·y)
    /// ```
    ///
    /// Matrix form:
    /// ```text
    /// ┌         ┐
    /// │ sx  0  0│
    /// │ 0  sy  0│
    /// │ 0   0  1│
    /// └         ┘
    /// ```
    ///
    /// # Arguments
    ///
    /// * `_sx` - Scale factor in x direction
    /// * `_sy` - Scale factor in y direction
    pub fn scaling(_sx: f64, _sy: f64) -> Self {
        // TODO: Implement 2D scaling matrix
        Self::identity()
    }

    /// Apply transformation to point.
    ///
    /// **TODO**: Not yet implemented. Returns point unchanged as placeholder.
    ///
    /// When implemented, will transform a point using homogeneous coordinates:
    /// ```text
    /// ┌         ┐   ┌   ┐   ┌    ┐
    /// │ a  b  tx│   │ x │   │ x' │
    /// │ c  d  ty│ × │ y │ = │ y' │
    /// │ 0  0  1 │   │ 1 │   │ 1  │
    /// └         ┘   └   ┘   └    ┘
    /// ```
    ///
    /// The result is (x', y') after dividing by the homogeneous coordinate.
    ///
    /// # Arguments
    ///
    /// * `point` - The point to transform
    ///
    /// # Returns
    ///
    /// The transformed point
    pub fn apply(&self, point: Cartesian2D) -> Cartesian2D {
        // TODO: Implement homogeneous coordinate transformation
        point
    }

    /// Compose with another transformation.
    ///
    /// **TODO**: Not yet implemented. Returns self unchanged as placeholder.
    ///
    /// When implemented, will compose two transformations through matrix multiplication.
    /// The result transformation T = self × other applies other first, then self:
    /// ```text
    /// T(p) = self(other(p))
    /// ```
    ///
    /// Note: Matrix multiplication is not commutative, so order matters.
    ///
    /// # Arguments
    ///
    /// * `_other` - The transformation to compose with
    ///
    /// # Returns
    ///
    /// The composed transformation
    pub fn compose(&self, _other: &Transform2D) -> Transform2D {
        // TODO: Implement matrix multiplication
        *self
    }
}

/// Rotation matrices for 3D transformations.
///
/// Provides methods to create 3×3 rotation matrices for rotating points in 3D space
/// around the coordinate axes or arbitrary axes. Returns nalgebra Matrix3 for integration
/// with linear algebra operations.
///
/// # Mathematical Background
///
/// Rotation matrices are orthogonal matrices that preserve distances and angles.
/// For any rotation matrix R:
/// - R^T R = I (orthogonal)
/// - det(R) = 1 (proper rotation, not reflection)
///
/// # Integration with nalgebra
///
/// All methods return `nalgebra::Matrix3<f64>` which can be:
/// - Multiplied with Vector3 to rotate points
/// - Composed through matrix multiplication
/// - Inverted by transposition (R^(-1) = R^T)
///
/// # Examples
///
/// ```
/// use mathsolver_core::transforms::{Rotation3D, Cartesian3D};
/// use std::f64::consts::PI;
///
/// // Rotate point around z-axis by 90 degrees
/// let rot = Rotation3D::around_z(PI / 2.0);
/// let point = Cartesian3D::new(1.0, 0.0, 0.0);
/// let vec = point.to_vector();
/// let rotated = rot * vec;
/// assert!((rotated[0] - 0.0).abs() < 1e-10);
/// assert!((rotated[1] - 1.0).abs() < 1e-10);
/// assert!((rotated[2] - 0.0).abs() < 1e-10);
/// ```
pub struct Rotation3D;

impl Rotation3D {
    /// Rotation around x-axis.
    ///
    /// Creates a rotation matrix for rotating points counterclockwise around the x-axis
    /// by angle θ when looking from positive x towards the origin (right-hand rule).
    ///
    /// # Matrix Form
    ///
    /// ```text
    /// Rx(θ) = ┌                   ┐
    ///         │ 1     0        0  │
    ///         │ 0  cos(θ)  -sin(θ)│
    ///         │ 0  sin(θ)   cos(θ)│
    ///         └                   ┘
    /// ```
    ///
    /// # Coordinate Transformation
    ///
    /// ```text
    /// x' = x
    /// y' = y cos(θ) - z sin(θ)
    /// z' = y sin(θ) + z cos(θ)
    /// ```
    ///
    /// # Coordinate System Diagram
    ///
    /// ```text
    ///   Looking down the x-axis (from +x towards origin):
    ///
    ///         z
    ///         ↑
    ///         |     P'
    ///         |    /
    ///       θ |   /
    ///         |  /
    ///         | /___P
    ///         |/________→ y
    ///
    ///   Positive rotation is counterclockwise (right-hand rule)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `theta` - Rotation angle in radians (counterclockwise when looking from +x)
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::transforms::Rotation3D;
    /// use nalgebra::Vector3;
    /// use std::f64::consts::PI;
    ///
    /// // Rotate 90° around x-axis: (0,1,0) → (0,0,1)
    /// let rot = Rotation3D::around_x(PI / 2.0);
    /// let vec = Vector3::new(0.0, 1.0, 0.0);
    /// let result = rot * vec;
    /// assert!((result[0] - 0.0).abs() < 1e-10);
    /// assert!((result[1] - 0.0).abs() < 1e-10);
    /// assert!((result[2] - 1.0).abs() < 1e-10);
    /// ```
    pub fn around_x(theta: f64) -> Matrix3<f64> {
        Matrix3::new(
            1.0,
            0.0,
            0.0,
            0.0,
            theta.cos(),
            -theta.sin(),
            0.0,
            theta.sin(),
            theta.cos(),
        )
    }

    /// Rotation around y-axis.
    ///
    /// Creates a rotation matrix for rotating points counterclockwise around the y-axis
    /// by angle θ when looking from positive y towards the origin (right-hand rule).
    ///
    /// # Matrix Form
    ///
    /// ```text
    /// Ry(θ) = ┌                   ┐
    ///         │  cos(θ)  0  sin(θ)│
    ///         │    0     1    0   │
    ///         │ -sin(θ)  0  cos(θ)│
    ///         └                   ┘
    /// ```
    ///
    /// # Coordinate Transformation
    ///
    /// ```text
    /// x' = x cos(θ) + z sin(θ)
    /// y' = y
    /// z' = -x sin(θ) + z cos(θ)
    /// ```
    ///
    /// # Coordinate System Diagram
    ///
    /// ```text
    ///   Looking down the y-axis (from +y towards origin):
    ///
    ///         z
    ///         ↑
    ///         |     P
    ///         |    /
    ///       θ |   /
    ///         |  /
    ///         | /___P'
    ///         |/________→ x
    ///
    ///   Positive rotation is counterclockwise (right-hand rule)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `theta` - Rotation angle in radians (counterclockwise when looking from +y)
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::transforms::Rotation3D;
    /// use nalgebra::Vector3;
    /// use std::f64::consts::PI;
    ///
    /// // Rotate 90° around y-axis: (1,0,0) → (0,0,-1)
    /// let rot = Rotation3D::around_y(PI / 2.0);
    /// let vec = Vector3::new(1.0, 0.0, 0.0);
    /// let result = rot * vec;
    /// assert!((result[0] - 0.0).abs() < 1e-10);
    /// assert!((result[1] - 0.0).abs() < 1e-10);
    /// assert!((result[2] - -1.0).abs() < 1e-10);
    /// ```
    pub fn around_y(theta: f64) -> Matrix3<f64> {
        Matrix3::new(
            theta.cos(),
            0.0,
            theta.sin(),
            0.0,
            1.0,
            0.0,
            -theta.sin(),
            0.0,
            theta.cos(),
        )
    }

    /// Rotation around z-axis.
    ///
    /// Creates a rotation matrix for rotating points counterclockwise around the z-axis
    /// by angle θ when looking from positive z towards the origin (right-hand rule).
    ///
    /// # Matrix Form
    ///
    /// ```text
    /// Rz(θ) = ┌                   ┐
    ///         │ cos(θ)  -sin(θ)  0│
    ///         │ sin(θ)   cos(θ)  0│
    ///         │   0        0     1│
    ///         └                   ┘
    /// ```
    ///
    /// # Coordinate Transformation
    ///
    /// ```text
    /// x' = x cos(θ) - y sin(θ)
    /// y' = x sin(θ) + y cos(θ)
    /// z' = z
    /// ```
    ///
    /// # Coordinate System Diagram
    ///
    /// ```text
    ///   Looking down the z-axis (from +z towards origin):
    ///
    ///         y
    ///         ↑
    ///         |     P'
    ///         |    /
    ///       θ |   /
    ///         |  /
    ///         | /___P
    ///         |/________→ x
    ///
    ///   Positive rotation is counterclockwise (right-hand rule)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `theta` - Rotation angle in radians (counterclockwise when looking from +z)
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::transforms::Rotation3D;
    /// use nalgebra::Vector3;
    /// use std::f64::consts::PI;
    ///
    /// // Rotate 90° around z-axis: (1,0,0) → (0,1,0)
    /// let rot = Rotation3D::around_z(PI / 2.0);
    /// let vec = Vector3::new(1.0, 0.0, 0.0);
    /// let result = rot * vec;
    /// assert!((result[0] - 0.0).abs() < 1e-10);
    /// assert!((result[1] - 1.0).abs() < 1e-10);
    /// assert!((result[2] - 0.0).abs() < 1e-10);
    ///
    /// // Compose rotations: 90° + 90° = 180°
    /// let rot1 = Rotation3D::around_z(PI / 2.0);
    /// let rot2 = Rotation3D::around_z(PI / 2.0);
    /// let combined = rot2 * rot1;
    /// let vec = Vector3::new(1.0, 0.0, 0.0);
    /// let result = combined * vec;
    /// assert!((result[0] - -1.0).abs() < 1e-10);
    /// assert!((result[1] - 0.0).abs() < 1e-10);
    /// ```
    pub fn around_z(theta: f64) -> Matrix3<f64> {
        Matrix3::new(
            theta.cos(),
            -theta.sin(),
            0.0,
            theta.sin(),
            theta.cos(),
            0.0,
            0.0,
            0.0,
            1.0,
        )
    }

    /// Rotation around arbitrary axis (Rodrigues' rotation formula).
    ///
    /// **TODO**: Not yet implemented. Returns identity matrix as placeholder.
    ///
    /// When implemented, will create a rotation matrix for rotating points around
    /// an arbitrary unit axis vector by angle θ using Rodrigues' rotation formula.
    ///
    /// # Rodrigues' Rotation Formula
    ///
    /// For a unit vector **k** = (kx, ky, kz) and angle θ:
    /// ```text
    /// R = I + sin(θ)K + (1 - cos(θ))K²
    ///
    /// where K is the cross-product matrix:
    /// K = ┌              ┐
    ///     │  0   -kz   ky│
    ///     │  kz   0   -kx│
    ///     │ -ky   kx   0 │
    ///     └              ┘
    /// ```
    ///
    /// # Arguments
    ///
    /// * `_axis` - Unit vector defining rotation axis (should be normalized)
    /// * `_theta` - Rotation angle in radians (right-hand rule)
    pub fn around_axis(_axis: Vector3<f64>, _theta: f64) -> Matrix3<f64> {
        // TODO: Implement Rodrigues' formula
        Matrix3::identity()
    }
}

// TODO: Add quaternion representations for 3D rotations
// TODO: Add homogeneous 3D transformations
// TODO: Add projection transformations (orthographic, perspective)
// TODO: Add coordinate frame conversions
// TODO: Add support for reference frame transformations
// TODO: Add geodetic coordinate systems (lat/lon/alt)
// TODO: Add support for non-Euclidean geometries
