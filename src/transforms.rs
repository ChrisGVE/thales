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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cartesian3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Cartesian3D {
    /// Create new 3D Cartesian coordinates.
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Convert to spherical coordinates.
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
    pub fn to_cylindrical(&self) -> Cylindrical {
        let rho = (self.x * self.x + self.y * self.y).sqrt();
        let phi = self.y.atan2(self.x);
        Cylindrical::new(rho, phi, self.z)
    }

    /// Convert to nalgebra vector.
    pub fn to_vector(&self) -> Vector3<f64> {
        Vector3::new(self.x, self.y, self.z)
    }

    /// Distance from origin.
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

/// Spherical coordinates (r, θ, φ).
/// Uses physics convention: r (radius), θ (azimuthal), φ (polar from z-axis).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spherical {
    /// Radius (distance from origin)
    pub r: f64,
    /// Azimuthal angle θ in radians (angle in xy-plane from x-axis)
    pub theta: f64,
    /// Polar angle φ in radians (angle from positive z-axis)
    pub phi: f64,
}

impl Spherical {
    /// Create new spherical coordinates.
    pub fn new(r: f64, theta: f64, phi: f64) -> Self {
        Self { r, theta, phi }
    }

    /// Convert to Cartesian coordinates.
    pub fn to_cartesian(&self) -> Cartesian3D {
        let x = self.r * self.phi.sin() * self.theta.cos();
        let y = self.r * self.phi.sin() * self.theta.sin();
        let z = self.r * self.phi.cos();
        Cartesian3D::new(x, y, z)
    }

    /// Convert to cylindrical coordinates.
    pub fn to_cylindrical(&self) -> Cylindrical {
        let rho = self.r * self.phi.sin();
        Cylindrical::new(rho, self.theta, self.r * self.phi.cos())
    }
}

/// Cylindrical coordinates (ρ, φ, z).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cylindrical {
    /// Radial distance from z-axis
    pub rho: f64,
    /// Azimuthal angle in radians
    pub phi: f64,
    /// Height along z-axis
    pub z: f64,
}

impl Cylindrical {
    /// Create new cylindrical coordinates.
    pub fn new(rho: f64, phi: f64, z: f64) -> Self {
        Self { rho, phi, z }
    }

    /// Convert to Cartesian coordinates.
    pub fn to_cartesian(&self) -> Cartesian3D {
        let x = self.rho * self.phi.cos();
        let y = self.rho * self.phi.sin();
        Cartesian3D::new(x, y, self.z)
    }

    /// Convert to spherical coordinates.
    pub fn to_spherical(&self) -> Spherical {
        let r = (self.rho * self.rho + self.z * self.z).sqrt();
        let phi = if r == 0.0 {
            0.0
        } else {
            (self.z / r).acos()
        };
        Spherical::new(r, self.phi, phi)
    }
}

/// Complex number operations and transformations.
pub struct ComplexOps;

impl ComplexOps {
    /// Convert complex number to polar form (r, θ).
    pub fn to_polar(c: Complex64) -> Polar {
        Polar::new(c.norm(), c.arg())
    }

    /// Convert polar form to complex number.
    pub fn from_polar(p: Polar) -> Complex64 {
        Complex64::from_polar(p.r, p.theta)
    }

    /// De Moivre's theorem: (r∠θ)^n = r^n∠(nθ).
    pub fn de_moivre(c: Complex64, n: f64) -> Complex64 {
        let polar = Self::to_polar(c);
        let r_n = polar.r.powf(n);
        let theta_n = polar.theta * n;
        Complex64::from_polar(r_n, theta_n)
    }

    /// nth root of complex number.
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
pub struct Rotation3D;

impl Rotation3D {
    /// Rotation around x-axis.
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
