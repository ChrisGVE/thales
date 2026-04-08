//! Polar, spherical, and cylindrical coordinate types and conversions.

use num_complex::Complex64;
use std::f64::consts::PI;

use super::{Cartesian2D, Cartesian3D};

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
/// use thales::transforms::{Polar, Cartesian2D};
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
    /// use thales::transforms::Polar;
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
    /// use thales::transforms::Polar;
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
    /// use thales::transforms::Polar;
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
    /// use thales::transforms::Polar;
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
/// use thales::transforms::{Spherical, Cartesian3D};
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
    /// use thales::transforms::Spherical;
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
    /// use thales::transforms::Spherical;
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
    /// use thales::transforms::Spherical;
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
/// use thales::transforms::{Cylindrical, Cartesian3D};
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
    /// use thales::transforms::Cylindrical;
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
    /// use thales::transforms::Cylindrical;
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
    /// use thales::transforms::Cylindrical;
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
