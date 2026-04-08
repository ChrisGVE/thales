//! 2D transformations and 3D rotation utilities.

use nalgebra::{Matrix3, Vector3};

use super::Cartesian2D;

/// Homogeneous transformation matrix for 2D transformations.
///
/// Represents 2D transformations (translation, rotation, scaling) using
/// homogeneous coordinates and a 3x3 transformation matrix.
///
/// # Mathematical Representation
///
/// A point (x, y) in homogeneous coordinates is represented as [x, y, 1]^T.
/// Transformations are represented as 3x3 matrices:
///
/// ```text
/// | a  b  tx |   | x |   | x' |
/// | c  d  ty | x | y | = | y' |
/// | 0  0  1  |   | 1 |   | 1  |
/// ```
///
/// # Examples
///
/// ```
/// use thales::transforms::{Transform2D, Cartesian2D};
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
    /// use thales::transforms::{Transform2D, Cartesian2D};
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
    /// Creates a transformation that translates points by (dx, dy):
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
    /// * `dx` - Translation in x direction
    /// * `dy` - Translation in y direction
    pub fn translation(dx: f64, dy: f64) -> Self {
        Self {
            matrix: Matrix3::new(1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0),
        }
    }

    /// Rotation transformation (angle in radians).
    ///
    /// Creates a transformation that rotates points counterclockwise by θ radians
    /// around the origin:
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
    /// * `theta` - Rotation angle in radians (counterclockwise)
    pub fn rotation(theta: f64) -> Self {
        let (s, c) = theta.sin_cos();
        Self {
            matrix: Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0),
        }
    }

    /// Scaling transformation.
    ///
    /// Creates a transformation that scales points by (sx, sy) in the x and y directions:
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
    /// * `sx` - Scale factor in x direction
    /// * `sy` - Scale factor in y direction
    pub fn scaling(sx: f64, sy: f64) -> Self {
        Self {
            matrix: Matrix3::new(sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0),
        }
    }

    /// Apply transformation to point.
    ///
    /// Transforms a point using homogeneous coordinates:
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
        use nalgebra::Vector3;
        let h = Vector3::new(point.x, point.y, 1.0);
        let result = self.matrix * h;
        Cartesian2D::new(result[0] / result[2], result[1] / result[2])
    }

    /// Compose with another transformation.
    ///
    /// Composes two transformations through matrix multiplication.
    /// The result transformation T = self × other applies `other` first, then `self`:
    /// ```text
    /// T(p) = self(other(p))
    /// ```
    ///
    /// Note: Matrix multiplication is not commutative, so order matters.
    ///
    /// # Arguments
    ///
    /// * `other` - The transformation to compose with
    ///
    /// # Returns
    ///
    /// The composed transformation
    pub fn compose(&self, other: &Transform2D) -> Transform2D {
        Transform2D {
            matrix: self.matrix * other.matrix,
        }
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
/// use thales::transforms::{Rotation3D, Cartesian3D};
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
    /// use thales::transforms::Rotation3D;
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
    /// use thales::transforms::Rotation3D;
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
    /// use thales::transforms::Rotation3D;
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
