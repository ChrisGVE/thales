//! FFI implementation functions for coordinate transforms, complex numbers, and units.

use crate::transforms::{Cartesian2D, Cartesian3D, ComplexOps, Polar, Spherical};
use num_complex::Complex64;

/// Convert Cartesian to polar coordinates.
pub(super) fn cartesian_to_polar_ffi(x: f64, y: f64) -> super::ffi::PolarCoords {
    let cart = Cartesian2D::new(x, y);
    let polar = cart.to_polar();
    super::ffi::PolarCoords {
        r: polar.r,
        theta: polar.theta,
    }
}

/// Convert polar to Cartesian coordinates.
pub(super) fn polar_to_cartesian_ffi(r: f64, theta: f64) -> super::ffi::CartesianCoords2D {
    let polar = Polar::new(r, theta);
    let cart = polar.to_cartesian();
    super::ffi::CartesianCoords2D {
        x: cart.x,
        y: cart.y,
    }
}

/// Convert 3D Cartesian to spherical coordinates.
pub(super) fn cartesian_to_spherical_ffi(x: f64, y: f64, z: f64) -> super::ffi::SphericalCoords {
    let cart = Cartesian3D::new(x, y, z);
    let spherical = cart.to_spherical();
    super::ffi::SphericalCoords {
        r: spherical.r,
        theta: spherical.theta,
        phi: spherical.phi,
    }
}

/// Convert spherical to 3D Cartesian coordinates.
pub(super) fn spherical_to_cartesian_ffi(
    r: f64,
    theta: f64,
    phi: f64,
) -> super::ffi::CartesianCoords3D {
    let spherical = Spherical::new(r, theta, phi);
    let cart = spherical.to_cartesian();
    super::ffi::CartesianCoords3D {
        x: cart.x,
        y: cart.y,
        z: cart.z,
    }
}

/// Add two complex numbers.
pub(super) fn complex_add_ffi(
    a_re: f64,
    a_im: f64,
    b_re: f64,
    b_im: f64,
) -> super::ffi::ComplexNumber {
    let a = Complex64::new(a_re, a_im);
    let b = Complex64::new(b_re, b_im);
    let result = a + b;
    super::ffi::ComplexNumber {
        real: result.re,
        imaginary: result.im,
    }
}

/// Multiply two complex numbers.
pub(super) fn complex_multiply_ffi(
    a_re: f64,
    a_im: f64,
    b_re: f64,
    b_im: f64,
) -> super::ffi::ComplexNumber {
    let a = Complex64::new(a_re, a_im);
    let b = Complex64::new(b_re, b_im);
    let result = a * b;
    super::ffi::ComplexNumber {
        real: result.re,
        imaginary: result.im,
    }
}

/// Convert complex number to polar form.
pub(super) fn complex_to_polar_ffi(re: f64, im: f64) -> super::ffi::PolarCoords {
    let c = Complex64::new(re, im);
    let polar = ComplexOps::to_polar(c);
    super::ffi::PolarCoords {
        r: polar.r,
        theta: polar.theta,
    }
}

/// Raise complex number to a power using De Moivre's theorem.
pub(super) fn complex_power_ffi(re: f64, im: f64, n: f64) -> super::ffi::ComplexNumber {
    let c = Complex64::new(re, im);
    let result = ComplexOps::de_moivre(c, n);
    super::ffi::ComplexNumber {
        real: result.re,
        imaginary: result.im,
    }
}

/// Apply a 2D translation to a point.
pub(super) fn translate_2d_ffi(x: f64, y: f64, dx: f64, dy: f64) -> super::ffi::CartesianCoords2D {
    use crate::transforms::Transform2D;
    let t = Transform2D::translation(dx, dy);
    let result = t.apply(Cartesian2D::new(x, y));
    super::ffi::CartesianCoords2D {
        x: result.x,
        y: result.y,
    }
}

/// Rotate a 2D point around the origin by the given angle (radians).
pub(super) fn rotate_2d_ffi(x: f64, y: f64, theta: f64) -> super::ffi::CartesianCoords2D {
    use crate::transforms::Transform2D;
    let t = Transform2D::rotation(theta);
    let result = t.apply(Cartesian2D::new(x, y));
    super::ffi::CartesianCoords2D {
        x: result.x,
        y: result.y,
    }
}

/// Scale a 2D point relative to the origin.
pub(super) fn scale_2d_ffi(x: f64, y: f64, sx: f64, sy: f64) -> super::ffi::CartesianCoords2D {
    use crate::transforms::Transform2D;
    let t = Transform2D::scaling(sx, sy);
    let result = t.apply(Cartesian2D::new(x, y));
    super::ffi::CartesianCoords2D {
        x: result.x,
        y: result.y,
    }
}

/// Compute all n distinct nth roots of a complex number.
///
/// Returns a JSON array of `[re, im]` pairs.
pub(super) fn complex_nth_roots_ffi(re: f64, im: f64, n: i32) -> Result<String, String> {
    if n <= 0 {
        return Err("n must be positive".to_string());
    }
    let c = Complex64::new(re, im);
    let roots = ComplexOps::nth_root(c, n);
    let pairs: Vec<[f64; 2]> = roots.iter().map(|r| [r.re, r.im]).collect();
    serde_json::to_string(&pairs).map_err(|e| format!("Serialization error: {}", e))
}

/// Convert a value from one unit to another.
///
/// Uses the built-in unit system with common SI and derived units.
pub(super) fn convert_units_ffi(value: f64, from_unit: &str, to_unit: &str) -> Result<f64, String> {
    use crate::dimensions::UnitRegistry;
    let registry = UnitRegistry::with_common_units();
    registry.convert(value, from_unit, to_unit)
}
