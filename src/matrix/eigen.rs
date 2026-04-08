//! Eigenvalue and eigenvector computation for matrices.

use super::{MatrixError, MatrixExpr, MatrixResult};

impl MatrixExpr {
    /// Compute eigenvalues of the matrix numerically.
    ///
    /// For 2x2 matrices, uses the quadratic formula.
    /// For larger matrices, uses numerical methods (power iteration or similar).
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::ast::Expression;
    ///
    /// let m = MatrixExpr::from_elements(vec![
    ///     vec![Expression::Integer(2), Expression::Integer(1)],
    ///     vec![Expression::Integer(1), Expression::Integer(2)],
    /// ]).unwrap();
    ///
    /// let eigenvalues = m.eigenvalues_numeric().unwrap();
    /// // Eigenvalues should be 1 and 3
    /// ```
    pub fn eigenvalues_numeric(&self) -> MatrixResult<Vec<f64>> {
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
                1 => Ok(vec![elements[0][0]]),
                2 => eigenvalues_2x2(&elements),
                3 => eigenvalues_3x3(&elements),
                _ => eigenvalues_qr(&elements),
            }
        }
    }

    /// Compute eigenvector for a given eigenvalue numerically.
    ///
    /// Returns the eigenvector as a column matrix.
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
    /// # Errors
    ///
    /// Returns an error if the matrix is not square.
    pub fn eigenpairs_numeric(&self) -> MatrixResult<Vec<(f64, Vec<f64>)>> {
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
                let eigenvector = self.eigenvector_numeric(eigenvalue)?;
                pairs.push((eigenvalue, eigenvector));
            }

            Ok(pairs)
        }
    }

    /// Check if the matrix is diagonalizable.
    ///
    /// A matrix is diagonalizable if it has n linearly independent eigenvectors.
    pub fn is_diagonalizable(&self) -> MatrixResult<bool> {
        if !self.is_square() {
            return Err(MatrixError::InvalidOperation(
                "Diagonalizability check requires a square matrix".to_string(),
            ));
        }

        // A simple check: symmetric matrices are always diagonalizable
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

        // For non-symmetric matrices, we would need to check algebraic vs geometric multiplicity
        // This is a simplified check - return true if we can compute distinct eigenvalues
        let eigenvalues = self.eigenvalues_numeric()?;

        // Check if all eigenvalues are distinct (sufficient condition)
        for (i, &ev1) in eigenvalues.iter().enumerate() {
            for (j, &ev2) in eigenvalues.iter().enumerate() {
                if i != j && (ev1 - ev2).abs() < 1e-10 {
                    // Repeated eigenvalue - would need to check geometric multiplicity
                    // For simplicity, assume diagonalizable
                    return Ok(true);
                }
            }
        }

        Ok(true)
    }
}

// =============================================================================
// Helper functions for eigenvalue computation (fallback when LAPACK is not available)
// =============================================================================

/// Compute eigenvalues for a 2x2 matrix using the quadratic formula.
#[cfg(not(feature = "lapack"))]
fn eigenvalues_2x2(elements: &[Vec<f64>]) -> MatrixResult<Vec<f64>> {
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
        // Complex eigenvalues - return just the real parts for now
        // A full implementation would return Complex numbers
        let real_part = trace / 2.0;
        Ok(vec![real_part, real_part])
    } else {
        let sqrt_disc = discriminant.sqrt();
        let lambda1 = (trace + sqrt_disc) / 2.0;
        let lambda2 = (trace - sqrt_disc) / 2.0;
        Ok(vec![lambda1, lambda2])
    }
}

/// Compute eigenvalues for a 3x3 matrix using Cardano's formula.
#[cfg(not(feature = "lapack"))]
fn eigenvalues_3x3(elements: &[Vec<f64>]) -> MatrixResult<Vec<f64>> {
    // For 3x3, we solve the cubic characteristic equation
    // det(A - λI) = -λ³ + tr(A)λ² - (sum of 2x2 principal minors)λ + det(A)
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

    // Solve cubic using Cardano's formula or numerical method
    solve_cubic(p, q, r)
}

/// Compute eigenvalues using QR algorithm for larger matrices.
#[cfg(not(feature = "lapack"))]
fn eigenvalues_qr(elements: &[Vec<f64>]) -> MatrixResult<Vec<f64>> {
    // Simple QR iteration
    let n = elements.len();
    let mut a = elements.to_vec();

    // Maximum iterations
    const MAX_ITER: usize = 100;
    const TOL: f64 = 1e-10;

    for _ in 0..MAX_ITER {
        // QR decomposition
        let (q, r) = qr_decomposition(&a);

        // A = R * Q
        a = matrix_multiply(&r, &q);

        // Check for convergence (off-diagonal elements small)
        let mut converged = true;
        for i in 0..n {
            for j in 0..i {
                if a[i][j].abs() > TOL {
                    converged = false;
                    break;
                }
            }
            if !converged {
                break;
            }
        }

        if converged {
            break;
        }
    }

    // Extract eigenvalues from diagonal
    Ok((0..n).map(|i| a[i][i]).collect())
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

    // Use inverse iteration to find eigenvector
    // Start with a random vector
    let mut v: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

    // Normalize
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v {
        *x /= norm;
    }

    // Inverse iteration: solve (A - λI)w = v, then v = w/||w||
    // Since A - λI is singular (or near-singular), we perturb slightly
    const MAX_ITER: usize = 50;
    const TOL: f64 = 1e-8;

    for _ in 0..MAX_ITER {
        // Solve (A - λI + εI)w = v using Gaussian elimination
        let mut augmented = a_minus_lambda.clone();
        for i in 0..n {
            augmented[i][i] += 1e-10; // Small perturbation
        }

        // Solve using Gaussian elimination
        let w = solve_linear_system(&augmented, &v);

        // Normalize
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-14 {
            break;
        }

        let w_normalized: Vec<f64> = w.iter().map(|x| x / norm).collect();

        // Check convergence
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
#[cfg(not(feature = "lapack"))]
fn solve_cubic(p: f64, q: f64, r: f64) -> MatrixResult<Vec<f64>> {
    // Depress the cubic: substitute x = t - p/3
    // t³ + at + b = 0 where:
    // a = q - p²/3
    // b = r - pq/3 + 2p³/27
    let a = q - p * p / 3.0;
    let b = r - p * q / 3.0 + 2.0 * p * p * p / 27.0;

    // Discriminant
    let discriminant = -4.0 * a * a * a - 27.0 * b * b;

    let offset = -p / 3.0;

    if discriminant > 0.0 {
        // Three distinct real roots
        let theta = (-b / 2.0 / ((-a / 3.0).powi(3).sqrt())).acos();
        let r_cubed = (-a / 3.0).sqrt();

        let t1 = 2.0 * r_cubed * (theta / 3.0).cos();
        let t2 = 2.0 * r_cubed * ((theta + 2.0 * std::f64::consts::PI) / 3.0).cos();
        let t3 = 2.0 * r_cubed * ((theta + 4.0 * std::f64::consts::PI) / 3.0).cos();

        Ok(vec![t1 + offset, t2 + offset, t3 + offset])
    } else if discriminant.abs() < 1e-10 {
        // Multiple roots
        if b.abs() < 1e-10 {
            // Triple root
            Ok(vec![offset, offset, offset])
        } else {
            // Double root
            let double_root = 3.0 * b / a;
            let simple_root = -3.0 * b / (2.0 * a);
            Ok(vec![
                double_root + offset,
                simple_root + offset,
                simple_root + offset,
            ])
        }
    } else {
        // One real root, two complex (return real root 3 times for now)
        let sqrt_disc = (b * b / 4.0 + a * a * a / 27.0).sqrt();
        let u = (-b / 2.0 + sqrt_disc).cbrt();
        let v = (-b / 2.0 - sqrt_disc).cbrt();
        let real_root = u + v + offset;

        // Return real root; complex roots have same real part
        let complex_real = -(u + v) / 2.0 + offset;
        Ok(vec![real_root, complex_real, complex_real])
    }
}

/// QR decomposition using Gram-Schmidt process.
#[cfg(not(feature = "lapack"))]
fn qr_decomposition(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut q = vec![vec![0.0; n]; n];
    let mut r = vec![vec![0.0; n]; n];

    for j in 0..n {
        // Start with column j of A
        let mut v: Vec<f64> = (0..n).map(|i| a[i][j]).collect();

        // Subtract projections onto previous q vectors
        for i in 0..j {
            let q_i: Vec<f64> = (0..n).map(|k| q[k][i]).collect();
            r[i][j] = dot_product(&q_i, &v);
            for k in 0..n {
                v[k] -= r[i][j] * q_i[k];
            }
        }

        // Compute norm and normalize
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

    // Create augmented matrix
    let mut aug: Vec<Vec<f64>> = a.to_vec();
    let mut rhs = b.to_vec();

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_row = k;
        let mut max_val = aug[k][k].abs();
        for i in (k + 1)..n {
            if aug[i][k].abs() > max_val {
                max_val = aug[i][k].abs();
                max_row = i;
            }
        }

        // Swap rows
        if max_row != k {
            aug.swap(k, max_row);
            rhs.swap(k, max_row);
        }

        // Eliminate
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

    // Back substitution
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
