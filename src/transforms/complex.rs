//! Complex number operations and transformations.

use num_complex::Complex64;

use crate::ast::{BinaryOp, Equation, Expression, Variable};

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

/// Decompose an expression into its real and imaginary parts.
///
/// Given an expression representing a complex quantity z = x + iy, returns a pair
/// `(real_part, imag_part)` as symbolic [`Expression`] values.
///
/// The `complex_vars` slice maps each complex variable name to its real and imaginary
/// component names: `(z, z_re, z_im)`.  For expressions that are not recognised as
/// complex (e.g. a plain real float), the imaginary part is [`Expression::Float`]`(0.0)`.
///
/// # Handled cases
///
/// | Input                      | Real part            | Imaginary part       |
/// |----------------------------|----------------------|----------------------|
/// | `Complex(a + bi)`          | `Float(a)`           | `Float(b)`           |
/// | `Variable("z")` (mapped)   | `Variable("z_re")`   | `Variable("z_im")`   |
/// | `Variable("x")` (unmapped) | `Variable("x")`      | `Float(0.0)`         |
/// | `Binary(Add, e1, e2)`      | `Add(re1, re2)`      | `Add(im1, im2)`      |
/// | `Binary(Sub, e1, e2)`      | `Sub(re1, re2)`      | `Sub(im1, im2)`      |
/// | `Binary(Mul, e1, e2)`      | `re1*re2 - im1*im2`  | `re1*im2 + im1*re2`  |
/// | `Integer(n)`               | `Integer(n)`         | `Float(0.0)`         |
/// | `Float(f)`                 | `Float(f)`           | `Float(0.0)`         |
///
/// # Arguments
///
/// * `expr` - The expression to decompose.
/// * `complex_vars` - Triples `(z_name, re_name, im_name)` naming each complex variable
///   and its real/imaginary counterparts.
///
/// # Examples
///
/// ```
/// use thales::transforms::separate_real_imag;
/// use thales::ast::{Expression, Variable};
/// use num_complex::Complex64;
///
/// // Constant complex number
/// let z = Expression::Complex(Complex64::new(3.0, 4.0));
/// let (re, im) = separate_real_imag(&z, &[]);
/// assert_eq!(re, Expression::Float(3.0));
/// assert_eq!(im, Expression::Float(4.0));
///
/// // Variable mapped to real/imaginary parts
/// let z = Expression::Variable(Variable::new("z"));
/// let mapping = [("z".to_string(), "z_re".to_string(), "z_im".to_string())];
/// let (re, im) = separate_real_imag(&z, &mapping);
/// assert_eq!(re, Expression::Variable(Variable::new("z_re")));
/// assert_eq!(im, Expression::Variable(Variable::new("z_im")));
/// ```
pub fn separate_real_imag(
    expr: &Expression,
    complex_vars: &[(String, String, String)],
) -> (Expression, Expression) {
    match expr {
        Expression::Complex(c) => (Expression::Float(c.re), Expression::Float(c.im)),
        Expression::Integer(n) => (Expression::Integer(*n), Expression::Float(0.0)),
        Expression::Float(f) => (Expression::Float(*f), Expression::Float(0.0)),
        Expression::Variable(v) => {
            if let Some((_, re_name, im_name)) = complex_vars.iter().find(|(z, _, _)| z == &v.name)
            {
                (
                    Expression::Variable(Variable::new(re_name.clone())),
                    Expression::Variable(Variable::new(im_name.clone())),
                )
            } else {
                (Expression::Variable(v.clone()), Expression::Float(0.0))
            }
        }
        Expression::Binary(op, lhs, rhs) => separate_binary(*op, lhs, rhs, complex_vars),
        // All other variants are treated as purely real.
        other => (other.clone(), Expression::Float(0.0)),
    }
}

/// Internal helper: decompose a binary expression into real and imaginary parts.
fn separate_binary(
    op: BinaryOp,
    lhs: &Expression,
    rhs: &Expression,
    vars: &[(String, String, String)],
) -> (Expression, Expression) {
    let (re1, im1) = separate_real_imag(lhs, vars);
    let (re2, im2) = separate_real_imag(rhs, vars);
    match op {
        BinaryOp::Add | BinaryOp::Sub => (
            Expression::Binary(op, Box::new(re1), Box::new(re2)),
            Expression::Binary(op, Box::new(im1), Box::new(im2)),
        ),
        BinaryOp::Mul => {
            // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            let ac =
                Expression::Binary(BinaryOp::Mul, Box::new(re1.clone()), Box::new(re2.clone()));
            let bd =
                Expression::Binary(BinaryOp::Mul, Box::new(im1.clone()), Box::new(im2.clone()));
            let ad = Expression::Binary(BinaryOp::Mul, Box::new(re1), Box::new(im2));
            let bc = Expression::Binary(BinaryOp::Mul, Box::new(im1), Box::new(re2));
            (
                Expression::Binary(BinaryOp::Sub, Box::new(ac), Box::new(bd)),
                Expression::Binary(BinaryOp::Add, Box::new(ad), Box::new(bc)),
            )
        }
        // For Div and Mod fall back to treating the whole expression as real.
        _ => {
            let full = Expression::Binary(op, Box::new(lhs.clone()), Box::new(rhs.clone()));
            (full, Expression::Float(0.0))
        }
    }
}

/// Decompose a complex equation into two real equations.
///
/// Given a complex equation `lhs = rhs` in terms of complex variable(s), produces two
/// real equations by separating real and imaginary parts:
///
/// - The **first** returned equation captures the real part: `Re(lhs) = Re(rhs)`.
/// - The **second** returned equation captures the imaginary part: `Im(lhs) = Im(rhs)`.
///
/// The IDs of the returned equations are derived from the original by appending `_re`
/// and `_im` respectively.
///
/// # Arguments
///
/// * `equation` - The complex equation to decompose.
/// * `complex_vars` - Triples `(z_name, re_name, im_name)` naming each complex variable
///   and its real/imaginary counterparts (forwarded to [`separate_real_imag`]).
///
/// # Examples
///
/// ```
/// use thales::transforms::decompose_complex_equation;
/// use thales::ast::{Equation, Expression, Variable, BinaryOp};
/// use num_complex::Complex64;
///
/// // z + 1 = 2i  (with z = x + iy)
/// let z = Expression::Variable(Variable::new("z"));
/// let one = Expression::Integer(1);
/// let two_i = Expression::Complex(Complex64::new(0.0, 2.0));
/// let lhs = Expression::Binary(BinaryOp::Add, Box::new(z), Box::new(one));
/// let eq = Equation::new("test", lhs, two_i);
///
/// let mapping = [("z".to_string(), "x".to_string(), "y".to_string())];
/// let (real_eq, imag_eq) = decompose_complex_equation(&eq, &mapping);
///
/// // Real part: x + 1 = 0.0
/// assert_eq!(real_eq.id, "test_re");
/// // Imaginary part: y + 0.0 = 2.0
/// assert_eq!(imag_eq.id, "test_im");
/// ```
pub fn decompose_complex_equation(
    equation: &Equation,
    complex_vars: &[(String, String, String)],
) -> (Equation, Equation) {
    let (lhs_re, lhs_im) = separate_real_imag(&equation.left, complex_vars);
    let (rhs_re, rhs_im) = separate_real_imag(&equation.right, complex_vars);
    let real_eq = Equation::new(format!("{}_re", equation.id), lhs_re, rhs_re);
    let imag_eq = Equation::new(format!("{}_im", equation.id), lhs_im, rhs_im);
    (real_eq, imag_eq)
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use crate::ast::{BinaryOp, Equation, Expression, Variable};

    use super::{decompose_complex_equation, separate_real_imag};

    fn mapping() -> Vec<(String, String, String)> {
        vec![("z".to_string(), "x".to_string(), "y".to_string())]
    }

    // --- separate_real_imag ---

    #[test]
    fn separate_constant_complex() {
        let z = Expression::Complex(Complex64::new(3.0, 4.0));
        let (re, im) = separate_real_imag(&z, &[]);
        assert_eq!(re, Expression::Float(3.0));
        assert_eq!(im, Expression::Float(4.0));
    }

    #[test]
    fn separate_variable_with_mapping() {
        let z = Expression::Variable(Variable::new("z"));
        let (re, im) = separate_real_imag(&z, &mapping());
        assert_eq!(re, Expression::Variable(Variable::new("x")));
        assert_eq!(im, Expression::Variable(Variable::new("y")));
    }

    #[test]
    fn separate_variable_without_mapping_is_real() {
        let v = Expression::Variable(Variable::new("a"));
        let (re, im) = separate_real_imag(&v, &[]);
        assert_eq!(re, Expression::Variable(Variable::new("a")));
        assert_eq!(im, Expression::Float(0.0));
    }

    #[test]
    fn separate_add_distributes() {
        // (z + 1) -> (x + 1, y + 0.0)
        let z = Expression::Variable(Variable::new("z"));
        let one = Expression::Integer(1);
        let expr = Expression::Binary(BinaryOp::Add, Box::new(z), Box::new(one));
        let (re, im) = separate_real_imag(&expr, &mapping());
        assert_eq!(
            re,
            Expression::Binary(
                BinaryOp::Add,
                Box::new(Expression::Variable(Variable::new("x"))),
                Box::new(Expression::Integer(1))
            )
        );
        assert_eq!(
            im,
            Expression::Binary(
                BinaryOp::Add,
                Box::new(Expression::Variable(Variable::new("y"))),
                Box::new(Expression::Float(0.0))
            )
        );
    }

    #[test]
    fn separate_mul_uses_complex_product_rule() {
        // z * z = (x*x - y*y) + (x*y + y*x)i
        let z1 = Expression::Variable(Variable::new("z"));
        let z2 = Expression::Variable(Variable::new("z"));
        let expr = Expression::Binary(BinaryOp::Mul, Box::new(z1), Box::new(z2));
        let (re, im) = separate_real_imag(&expr, &mapping());
        // Re: x*x - y*y
        assert_eq!(
            re,
            Expression::Binary(
                BinaryOp::Sub,
                Box::new(Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(Expression::Variable(Variable::new("x"))),
                    Box::new(Expression::Variable(Variable::new("x")))
                )),
                Box::new(Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(Expression::Variable(Variable::new("y"))),
                    Box::new(Expression::Variable(Variable::new("y")))
                ))
            )
        );
        // Im: x*y + y*x
        assert_eq!(
            im,
            Expression::Binary(
                BinaryOp::Add,
                Box::new(Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(Expression::Variable(Variable::new("x"))),
                    Box::new(Expression::Variable(Variable::new("y")))
                )),
                Box::new(Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(Expression::Variable(Variable::new("y"))),
                    Box::new(Expression::Variable(Variable::new("x")))
                ))
            )
        );
    }

    // --- decompose_complex_equation ---

    #[test]
    fn decompose_z_plus_one_eq_two_i() {
        // z + 1 = 2i  =>  real: x + 1 = 0.0,  imag: y + 0.0 = 2.0
        let z = Expression::Variable(Variable::new("z"));
        let one = Expression::Integer(1);
        let two_i = Expression::Complex(Complex64::new(0.0, 2.0));
        let lhs = Expression::Binary(BinaryOp::Add, Box::new(z), Box::new(one));
        let eq = Equation::new("eq1", lhs, two_i);
        let (real_eq, imag_eq) = decompose_complex_equation(&eq, &mapping());
        assert_eq!(real_eq.id, "eq1_re");
        assert_eq!(imag_eq.id, "eq1_im");
        // Real equation LHS: x + 1
        assert_eq!(
            real_eq.left,
            Expression::Binary(
                BinaryOp::Add,
                Box::new(Expression::Variable(Variable::new("x"))),
                Box::new(Expression::Integer(1))
            )
        );
        assert_eq!(real_eq.right, Expression::Float(0.0));
        // Imaginary equation LHS: y + 0.0
        assert_eq!(
            imag_eq.left,
            Expression::Binary(
                BinaryOp::Add,
                Box::new(Expression::Variable(Variable::new("y"))),
                Box::new(Expression::Float(0.0))
            )
        );
        assert_eq!(imag_eq.right, Expression::Float(2.0));
    }

    #[test]
    fn decompose_z_squared_eq_minus_one() {
        // z² = -1  (represented as z*z = Complex(-1+0i))
        // Real part:  x*x - y*y = -1.0
        // Imag part:  x*y + y*x = 0.0
        let z1 = Expression::Variable(Variable::new("z"));
        let z2 = Expression::Variable(Variable::new("z"));
        let z_sq = Expression::Binary(BinaryOp::Mul, Box::new(z1), Box::new(z2));
        let minus_one = Expression::Complex(Complex64::new(-1.0, 0.0));
        let eq = Equation::new("z_sq", z_sq, minus_one);
        let (real_eq, imag_eq) = decompose_complex_equation(&eq, &mapping());
        assert_eq!(real_eq.id, "z_sq_re");
        assert_eq!(imag_eq.id, "z_sq_im");
        assert_eq!(real_eq.right, Expression::Float(-1.0));
        assert_eq!(imag_eq.right, Expression::Float(0.0));
    }
}
