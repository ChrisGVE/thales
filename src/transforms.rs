//! Coordinate system transformations.
//!
//! Provides transformations between different coordinate systems
//! (Cartesian, Polar, Spherical, Cylindrical) and complex number operations.

use nalgebra::{Matrix3, Vector2, Vector3};
use num_complex::Complex64;
use std::f64::consts::PI;

/// 2D Cartesian coordinates (x, y).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cartesian2D {
    pub x: f64,
    pub y: f64,
}

impl Cartesian2D {
    /// Create new 2D Cartesian coordinates.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Convert to polar coordinates.
    pub fn to_polar(&self) -> Polar {
        let r = (self.x * self.x + self.y * self.y).sqrt();
        let theta = self.y.atan2(self.x);
        Polar::new(r, theta)
    }

    /// Convert to complex number.
    pub fn to_complex(&self) -> Complex64 {
        Complex64::new(self.x, self.y)
    }

    /// Convert to nalgebra vector.
    pub fn to_vector(&self) -> Vector2<f64> {
        Vector2::new(self.x, self.y)
    }

    /// Distance from origin.
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

/// Polar coordinates (r, θ).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Polar {
    /// Radius (distance from origin)
    pub r: f64,
    /// Angle in radians (counterclockwise from positive x-axis)
    pub theta: f64,
}

impl Polar {
    /// Create new polar coordinates.
    pub fn new(r: f64, theta: f64) -> Self {
        Self { r, theta }
    }

    /// Normalize angle to [0, 2π).
    pub fn normalize_angle(&mut self) {
        self.theta = self.theta.rem_euclid(2.0 * PI);
    }

    /// Convert to Cartesian coordinates.
    pub fn to_cartesian(&self) -> Cartesian2D {
        let x = self.r * self.theta.cos();
        let y = self.r * self.theta.sin();
        Cartesian2D::new(x, y)
    }

    /// Convert to complex number (polar form).
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform2D {
    matrix: Matrix3<f64>,
}

impl Transform2D {
    /// Identity transformation.
    pub fn identity() -> Self {
        Self {
            matrix: Matrix3::identity(),
        }
    }

    /// Translation transformation.
    pub fn translation(_dx: f64, _dy: f64) -> Self {
        // TODO: Implement 2D translation matrix
        Self::identity()
    }

    /// Rotation transformation (angle in radians).
    pub fn rotation(_theta: f64) -> Self {
        // TODO: Implement 2D rotation matrix
        Self::identity()
    }

    /// Scaling transformation.
    pub fn scaling(_sx: f64, _sy: f64) -> Self {
        // TODO: Implement 2D scaling matrix
        Self::identity()
    }

    /// Apply transformation to point.
    pub fn apply(&self, point: Cartesian2D) -> Cartesian2D {
        // TODO: Implement homogeneous coordinate transformation
        point
    }

    /// Compose with another transformation.
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
