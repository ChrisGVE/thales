//! Eigenvalue and eigenvector computation for matrices.

use num_complex::Complex64;

use super::{MatrixError, MatrixExpr, MatrixResult};

impl MatrixExpr {
    /// Compute eigenvalues of the matrix numerically.
    ///
    /// Returns complex eigenvalues. For matrices with purely real eigenvalues
    /// the imaginary part is zero.
    ///
    /// For 2x2 matrices, uses the quadratic formula.
    /// For larger matrices, uses numerical methods (QR iteration).
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::numeric::expr::Expr;
    ///
    /// let m = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(2), Expr::int(1)],
    ///     vec![Expr::int(1), Expr::int(2)],
    /// ]).unwrap();
    ///
    /// let eigenvalues = m.eigenvalues_numeric().unwrap();
    /// // Eigenvalues should be 1 and 3 (real)
    /// ```
    pub fn eigenvalues_numeric(&self) -> MatrixResult<Vec<Complex64>> {
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "Eigenvalues require a square matrix".to_string(),
            ));
        }

        let empty = std::collections::HashMap::new();
        let elements = self.evaluate(&empty).ok_or_else(|| {
            MatrixError::InvalidOperation("Cannot evaluate matrix numerically".to_string())
        })?;

        #[cfg(feature = "lapack")]
        {
            crate::lapack::eigenvalues(&elements).map_err(MatrixError::InvalidOperation)
        }

        #[cfg(not(feature = "lapack"))]
        {
            match self.rows {
                1 => Ok(vec![Complex64::new(elements[0][0], 0.0)]),
                2 => eigenvalues_2x2(&elements),
                3 => eigenvalues_3x3(&elements),
                _ => eigenvalues_qr(&elements),
            }
        }
    }

    /// Compute eigenvector for a given eigenvalue numerically.
    ///
    /// Returns the eigenvector as a column of real values. For complex
    /// eigenvalues only the real part is used in the power iteration; the
    /// limitation is documented on the caller side.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square.
    pub fn eigenvector_numeric(&self, eigenvalue: f64) -> MatrixResult<Vec<f64>> {
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "Eigenvector requires a square matrix".to_string(),
            ));
        }

        let empty = std::collections::HashMap::new();
        let elements = self.evaluate(&empty).ok_or_else(|| {
            MatrixError::InvalidOperation("Cannot evaluate matrix numerically".to_string())
        })?;

        #[cfg(feature = "lapack")]
        {
            crate::lapack::eigenvector(&elements, eigenvalue).map_err(MatrixError::InvalidOperation)
        }

        #[cfg(not(feature = "lapack"))]
        {
            eigenvector_inverse_iteration(&elements, eigenvalue)
        }
    }

    /// Compute all eigenpairs (eigenvalue, eigenvector) numerically.
    ///
    /// The eigenvalue component is `Complex64`. For complex eigenvalues the
    /// eigenvector is computed using the real part of the eigenvalue, which
    /// gives an approximation; a full complex eigenvector solver is not yet
    /// implemented.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square.
    pub fn eigenpairs_numeric(&self) -> MatrixResult<Vec<(Complex64, Vec<f64>)>> {
        #[cfg(feature = "lapack")]
        {
            if !self.is_square() {
                return Err(MatrixError::InvalidOperation(
                    "Eigenpairs require a square matrix".to_string(),
                ));
            }

            let empty = std::collections::HashMap::new();
            let elements = self.evaluate(&empty).ok_or_else(|| {
                MatrixError::InvalidOperation("Cannot evaluate matrix numerically".to_string())
            })?;

            crate::lapack::eigenpairs(&elements).map_err(MatrixError::InvalidOperation)
        }

        #[cfg(not(feature = "lapack"))]
        {
            let eigenvalues = self.eigenvalues_numeric()?;
            let mut pairs = Vec::with_capacity(eigenvalues.len());

            for eigenvalue in eigenvalues {
                // eigenvector_numeric operates on a real matrix with a real shift;
                // for complex eigenvalues we pass the real part as the best available
                // approximation until a full complex eigenvector path is added.
                let eigenvector = self.eigenvector_numeric(eigenvalue.re)?;
                pairs.push((eigenvalue, eigenvector));
            }

            Ok(pairs)
        }
    }

    /// Check if the matrix is diagonalizable.
    ///
    /// A matrix is diagonalizable if and only if for every eigenvalue its
    /// geometric multiplicity equals its algebraic multiplicity. This
    /// implementation checks that directly: it groups eigenvalues by value
    /// (within tolerance), and for any repeated eigenvalue it computes the
    /// rank of (A − λI) to determine the geometric multiplicity.
    pub fn is_diagonalizable(&self) -> MatrixResult<bool> {
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "Diagonalizability check requires a square matrix".to_string(),
            ));
        }

        // Symmetric matrices are always diagonalizable (spectral theorem).
        let transpose = self.transpose();
        let empty = std::collections::HashMap::new();

        if let (Some(a), Some(at)) = (self.evaluate(&empty), transpose.evaluate(&empty)) {
            let is_symmetric = a.iter().zip(at.iter()).all(|(row_a, row_at)| {
                row_a
                    .iter()
                    .zip(row_at.iter())
                    .all(|(x, y)| (x - y).abs() < 1e-10)
            });

            if is_symmetric {
                return Ok(true);
            }
        }

        let elements = self.evaluate(&empty).ok_or_else(|| {
            MatrixError::InvalidOperation("Cannot evaluate matrix numerically".to_string())
        })?;
        let n = self.rows;

        let eigenvalues = self.eigenvalues_numeric()?;

        // Group eigenvalues — only the real part matters for the shift because
        // eigenvector_numeric already works on a real system. We group by the
        // full complex value so that conjugate pairs are not accidentally merged.
        // Two eigenvalues are considered equal if both real and imaginary parts
        // agree within 1e-8.
        let mut processed: Vec<Complex64> = Vec::new();

        for &ev in &eigenvalues {
            if processed
                .iter()
                .any(|&p| (ev.re - p.re).abs() < 1e-8 && (ev.im - p.im).abs() < 1e-8)
            {
                continue;
            }

            // Count algebraic multiplicity (number of eigenvalues equal to ev).
            let alg_mult = eigenvalues
                .iter()
                .filter(|&&e| (e.re - ev.re).abs() < 1e-8 && (e.im - ev.im).abs() < 1e-8)
                .count();

            if alg_mult > 1 {
                // Geometric multiplicity = n − rank(A − λI).
                let shift = ev.re; // real shift; adequate for real matrices
                let mut a_minus_lambda = elements.clone();
                for i in 0..n {
                    a_minus_lambda[i][i] -= shift;
                }
                let rank = compute_rank(&a_minus_lambda);
                let geo_mult = n - rank;

                if geo_mult < alg_mult {
                    return Ok(false);
                }
            }

            processed.push(ev);
        }

        Ok(true)
    }
}

// =============================================================================
// Helper functions for eigenvalue computation (fallback when LAPACK is not available)
// =============================================================================

/// Compute eigenvalues for a 2x2 matrix using the quadratic formula.
#[cfg(not(feature = "lapack"))]
fn eigenvalues_2x2(elements: &[Vec<f64>]) -> MatrixResult<Vec<Complex64>> {
    let a = elements[0][0];
    let b = elements[0][1];
    let c = elements[1][0];
    let d = elements[1][1];

    // Characteristic equation: λ² - (a+d)λ + (ad - bc) = 0
    // Using quadratic formula: λ = ((a+d) ± sqrt((a+d)² - 4(ad-bc))) / 2
    let trace = a + d;
    let det = a * d - b * c;
    let discriminant = trace * trace - 4.0 * det;

    if discriminant < 0.0 {
        // Complex conjugate pair.
        let real_part = trace / 2.0;
        let imag_part = (-discriminant).sqrt() / 2.0;
        Ok(vec![
            Complex64::new(real_part, imag_part),
            Complex64::new(real_part, -imag_part),
        ])
    } else {
        let sqrt_disc = discriminant.sqrt();
        let lambda1 = (trace + sqrt_disc) / 2.0;
        let lambda2 = (trace - sqrt_disc) / 2.0;
        Ok(vec![
            Complex64::new(lambda1, 0.0),
            Complex64::new(lambda2, 0.0),
        ])
    }
}

/// Compute eigenvalues for a 3x3 matrix using Cardano's formula.
#[cfg(not(feature = "lapack"))]
fn eigenvalues_3x3(elements: &[Vec<f64>]) -> MatrixResult<Vec<Complex64>> {
    let a11 = elements[0][0];
    let a12 = elements[0][1];
    let a13 = elements[0][2];
    let a21 = elements[1][0];
    let a22 = elements[1][1];
    let a23 = elements[1][2];
    let a31 = elements[2][0];
    let a32 = elements[2][1];
    let a33 = elements[2][2];

    // Coefficients of λ³ + p*λ² + q*λ + r = 0
    let trace = a11 + a22 + a33;
    let p = -trace;

    // Sum of 2x2 principal minors
    let minor12 = a11 * a22 - a12 * a21;
    let minor13 = a11 * a33 - a13 * a31;
    let minor23 = a22 * a33 - a23 * a32;
    let q = minor12 + minor13 + minor23;

    // Determinant
    let det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31);
    let r = -det;

    solve_cubic(p, q, r)
}

/// Compute eigenvalues using QR algorithm for larger matrices.
#[cfg(not(feature = "lapack"))]
fn eigenvalues_qr(elements: &[Vec<f64>]) -> MatrixResult<Vec<Complex64>> {
    let n = elements.len();
    let mut a = elements.to_vec();

    const MAX_ITER: usize = 100;
    const TOL: f64 = 1e-10;

    for _ in 0..MAX_ITER {
        let (q, r) = qr_decomposition(&a);
        a = matrix_multiply(&r, &q);

        let mut converged = true;
        'outer: for i in 0..n {
            for j in 0..i {
                if a[i][j].abs() > TOL {
                    converged = false;
                    break 'outer;
                }
            }
        }

        if converged {
            break;
        }
    }

    Ok((0..n).map(|i| Complex64::new(a[i][i], 0.0)).collect())
}

/// Find eigenvector using inverse iteration.
#[cfg(not(feature = "lapack"))]
fn eigenvector_inverse_iteration(elements: &[Vec<f64>], eigenvalue: f64) -> MatrixResult<Vec<f64>> {
    let n = elements.len();

    // Compute A - λI
    let mut a_minus_lambda: Vec<Vec<f64>> = elements.to_vec();
    for i in 0..n {
        a_minus_lambda[i][i] -= eigenvalue;
    }

    // Start with a normalized vector [1, 2, ..., n] / norm
    let mut v: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v {
        *x /= norm;
    }

    const MAX_ITER: usize = 50;
    const TOL: f64 = 1e-8;

    for _ in 0..MAX_ITER {
        // Solve (A - λI + εI)w = v using Gaussian elimination
        let mut augmented = a_minus_lambda.clone();
        for i in 0..n {
            augmented[i][i] += 1e-10; // Small perturbation for near-singular matrix
        }

        let w = solve_linear_system(&augmented, &v);

        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-14 {
            break;
        }

        let w_normalized: Vec<f64> = w.iter().map(|x| x / norm).collect();

        let diff: f64 = v
            .iter()
            .zip(w_normalized.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        v = w_normalized;

        if diff < TOL {
            break;
        }
    }

    Ok(v)
}

/// Solve cubic equation x³ + p*x² + q*x + r = 0 using Cardano's formula.
///
/// Returns all three roots as `Complex64`. When the discriminant indicates
/// one real root and two complex conjugates, the imaginary parts are
/// preserved rather than discarded.
#[cfg(not(feature = "lapack"))]
fn solve_cubic(p: f64, q: f64, r: f64) -> MatrixResult<Vec<Complex64>> {
    // Depress the cubic: substitute x = t - p/3
    // t³ + at + b = 0 where a = q - p²/3, b = r - pq/3 + 2p³/27
    let a = q - p * p / 3.0;
    let b = r - p * q / 3.0 + 2.0 * p * p * p / 27.0;

    let discriminant = -4.0 * a * a * a - 27.0 * b * b;
    let offset = -p / 3.0;

    if discriminant > 0.0 {
        // Three distinct real roots.
        let acos_arg = (-b / 2.0 / ((-a / 3.0).powi(3).sqrt())).clamp(-1.0, 1.0);
        let theta = acos_arg.acos();
        let r_cubed = (-a / 3.0).sqrt();

        let t1 = 2.0 * r_cubed * (theta / 3.0).cos();
        let t2 = 2.0 * r_cubed * ((theta + 2.0 * std::f64::consts::PI) / 3.0).cos();
        let t3 = 2.0 * r_cubed * ((theta + 4.0 * std::f64::consts::PI) / 3.0).cos();

        Ok(vec![
            Complex64::new(t1 + offset, 0.0),
            Complex64::new(t2 + offset, 0.0),
            Complex64::new(t3 + offset, 0.0),
        ])
    } else if discriminant.abs() < 1e-10 {
        // Multiple roots (all real).
        if b.abs() < 1e-10 {
            Ok(vec![
                Complex64::new(offset, 0.0),
                Complex64::new(offset, 0.0),
                Complex64::new(offset, 0.0),
            ])
        } else {
            let double_root = 3.0 * b / a;
            let simple_root = -3.0 * b / (2.0 * a);
            Ok(vec![
                Complex64::new(double_root + offset, 0.0),
                Complex64::new(simple_root + offset, 0.0),
                Complex64::new(simple_root + offset, 0.0),
            ])
        }
    } else {
        // One real root and two complex conjugates.
        let sqrt_disc = (b * b / 4.0 + a * a * a / 27.0).sqrt();
        let u = (-b / 2.0 + sqrt_disc).cbrt();
        let v = (-b / 2.0 - sqrt_disc).cbrt();
        let real_root = u + v + offset;
        // Complex pair: -(u+v)/2 ± i*(u-v)*sqrt(3)/2
        let complex_re = -(u + v) / 2.0 + offset;
        let complex_im = (u - v) * (3.0_f64).sqrt() / 2.0;
        Ok(vec![
            Complex64::new(real_root, 0.0),
            Complex64::new(complex_re, complex_im),
            Complex64::new(complex_re, -complex_im),
        ])
    }
}

/// QR decomposition using Gram-Schmidt process.
#[cfg(not(feature = "lapack"))]
fn qr_decomposition(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut q = vec![vec![0.0; n]; n];
    let mut r = vec![vec![0.0; n]; n];

    for j in 0..n {
        let mut v: Vec<f64> = (0..n).map(|i| a[i][j]).collect();

        for i in 0..j {
            let q_i: Vec<f64> = (0..n).map(|k| q[k][i]).collect();
            r[i][j] = dot_product(&q_i, &v);
            for k in 0..n {
                v[k] -= r[i][j] * q_i[k];
            }
        }

        r[j][j] = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if r[j][j] > 1e-14 {
            for k in 0..n {
                q[k][j] = v[k] / r[j][j];
            }
        }
    }

    (q, r)
}

/// Dot product of two vectors.
#[cfg(not(feature = "lapack"))]
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Matrix multiplication for f64 matrices.
#[cfg(not(feature = "lapack"))]
fn matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut result = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    result
}

/// Solve linear system Ax = b using Gaussian elimination with partial pivoting.
#[cfg(not(feature = "lapack"))]
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut aug: Vec<Vec<f64>> = a.to_vec();
    let mut rhs = b.to_vec();

    for k in 0..n {
        let mut max_row = k;
        let mut max_val = aug[k][k].abs();
        for i in (k + 1)..n {
            if aug[i][k].abs() > max_val {
                max_val = aug[i][k].abs();
                max_row = i;
            }
        }

        if max_row != k {
            aug.swap(k, max_row);
            rhs.swap(k, max_row);
        }

        if aug[k][k].abs() > 1e-14 {
            for i in (k + 1)..n {
                let factor = aug[i][k] / aug[k][k];
                for j in k..n {
                    aug[i][j] -= factor * aug[k][j];
                }
                rhs[i] -= factor * rhs[k];
            }
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        if aug[i][i].abs() > 1e-14 {
            x[i] = rhs[i];
            for j in (i + 1)..n {
                x[i] -= aug[i][j] * x[j];
            }
            x[i] /= aug[i][i];
        }
    }

    x
}

/// Compute the rank of a matrix using Gaussian elimination with partial pivoting.
///
/// Used by `is_diagonalizable` to determine geometric multiplicity of eigenvalues.
fn compute_rank(a: &[Vec<f64>]) -> usize {
    let m = a.len();
    if m == 0 {
        return 0;
    }
    let n = a[0].len();
    let mut mat = a.to_vec();
    let mut rank = 0;
    let mut row = 0;

    for col in 0..n {
        // Find pivot in this column from current row downward.
        let pivot_row =
            (row..m).max_by(|&i, &j| mat[i][col].abs().partial_cmp(&mat[j][col].abs()).unwrap());

        let pivot_row = match pivot_row {
            Some(r) if mat[r][col].abs() > 1e-10 => r,
            _ => continue,
        };

        mat.swap(row, pivot_row);
        rank += 1;

        let pivot = mat[row][col];
        for j in col..n {
            mat[row][j] /= pivot;
        }

        for i in 0..m {
            if i != row && mat[i][col].abs() > 1e-14 {
                let factor = mat[i][col];
                for j in col..n {
                    mat[i][j] -= factor * mat[row][j];
                }
            }
        }

        row += 1;
        if row == m {
            break;
        }
    }

    rank
}
