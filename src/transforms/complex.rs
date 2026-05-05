//! Complex number operations and transformations.

use std::sync::Arc;

use num_complex::Complex64;

use crate::numeric::expr::{Expr, FuncId};

use super::Polar;

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
/// use thales::transforms::ComplexOps;
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
    /// use thales::transforms::ComplexOps;
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
    /// use thales::transforms::{ComplexOps, Polar};
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
    /// use thales::transforms::ComplexOps;
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

    /// Compute all nth roots of a complex number using De Moivre's theorem.
    ///
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
    /// of radius r^(1/n) centred at the origin.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::transforms::ComplexOps;
    /// use num_complex::Complex64;
    ///
    /// // Find all cube roots of 8 (real number): 2, -1+√3i, -1-√3i
    /// let z = Complex64::new(8.0, 0.0);
    /// let roots = ComplexOps::nth_root(z, 3);
    /// assert_eq!(roots.len(), 3);
    ///
    /// // Find all square roots of -1: i and -i
    /// let z = Complex64::new(-1.0, 0.0);
    /// let roots = ComplexOps::nth_root(z, 2);
    /// assert_eq!(roots.len(), 2);
    /// ```
    ///
    /// # Arguments
    ///
    /// * `c` - Complex number to find roots of
    /// * `n` - Root degree (must be a positive integer; returns empty vec for n ≤ 0)
    ///
    /// # Returns
    ///
    /// Vector of all n distinct nth roots ordered by increasing angle.
    pub fn nth_root(c: Complex64, n: i32) -> Vec<Complex64> {
        if n <= 0 {
            return vec![];
        }
        let polar = Self::to_polar(c);
        let r_n = polar.r.powf(1.0 / f64::from(n));
        let n_usize = n as usize;
        (0..n_usize)
            .map(|k| {
                let theta_k = (polar.theta + 2.0 * std::f64::consts::PI * k as f64) / f64::from(n);
                Complex64::from_polar(r_n, theta_k)
            })
            .collect()
    }
}

/// Extract the symbolic real part of an [`Arc<Expr>`] expression.
///
/// Wraps the expression in a `Re(...)` node, allowing the simplifier to
/// reduce it when the structure is known (e.g. `Re(n) = n` for real `n`).
///
/// # Examples
///
/// ```
/// use thales::transforms::arc_real_part;
/// use thales::numeric::expr::Expr;
///
/// let x = Expr::symbol("z");
/// let re_z = arc_real_part(&x);
/// assert_eq!(re_z.to_string(), "Re(z)");
/// ```
pub fn arc_real_part(expr: &Arc<Expr>) -> Arc<Expr> {
    Expr::func(FuncId::Re, vec![expr.clone()])
}

/// Extract the symbolic imaginary part of an [`Arc<Expr>`] expression.
///
/// Wraps the expression in an `Im(...)` node.
///
/// # Examples
///
/// ```
/// use thales::transforms::arc_imag_part;
/// use thales::numeric::expr::Expr;
///
/// let x = Expr::symbol("z");
/// let im_z = arc_imag_part(&x);
/// assert_eq!(im_z.to_string(), "Im(z)");
/// ```
pub fn arc_imag_part(expr: &Arc<Expr>) -> Arc<Expr> {
    Expr::func(FuncId::Im, vec![expr.clone()])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::expr::{Expr, FuncId};

    // --- arc_real_part / arc_imag_part ---

    #[test]
    fn arc_real_part_wraps_in_re_node() {
        let x = Expr::symbol("arc_re_x");
        let result = arc_real_part(&x);
        match result.as_ref() {
            Expr::Func(FuncId::Re, args) if args.len() == 1 => {
                assert_eq!(*args[0], *x);
            }
            _ => panic!("expected Re(x), got {:?}", result),
        }
    }

    #[test]
    fn arc_imag_part_wraps_in_im_node() {
        let x = Expr::symbol("arc_im_x");
        let result = arc_imag_part(&x);
        match result.as_ref() {
            Expr::Func(FuncId::Im, args) if args.len() == 1 => {
                assert_eq!(*args[0], *x);
            }
            _ => panic!("expected Im(x), got {:?}", result),
        }
    }

    #[test]
    fn arc_real_part_of_real_literal_simplifies() {
        use crate::numeric::simplify::simplify;
        let n = Expr::int(7);
        let re_n = arc_real_part(&n);
        let simplified = simplify(&re_n);
        assert_eq!(*simplified, *n, "Re(7) should simplify to 7");
    }

    #[test]
    fn arc_imag_part_of_real_literal_simplifies_to_zero() {
        use crate::numeric::simplify::simplify;
        let n = Expr::int(7);
        let im_n = arc_imag_part(&n);
        let simplified = simplify(&im_n);
        assert!(simplified.is_zero(), "Im(7) should simplify to 0");
    }
}
