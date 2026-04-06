//! LAPACK-accelerated numerical linear algebra operations.
//!
//! This module provides high-performance implementations of numerical matrix
//! operations using LAPACK via nalgebra-lapack. On macOS/iOS, this links
//! against Apple's Accelerate framework for hardware-optimized routines.
//!
//! Enable with the `lapack` feature flag:
//! ```toml
//! [dependencies]
//! thales = { version = "0.3", features = ["lapack"] }
//! ```

use nalgebra::DMatrix;
use nalgebra_lapack::{Eigen, LU, QR};

/// Compute eigenvalues of a real square matrix using LAPACK.
///
/// Returns a vector of real eigenvalues. For matrices with complex eigenvalues,
/// only the real parts are returned (matching the fallback behavior).
pub fn eigenvalues(elements: &[Vec<f64>]) -> Result<Vec<f64>, String> {
    let n = elements.len();
    if n == 0 {
        return Ok(vec![]);
    }

    let flat: Vec<f64> = (0..n)
        .flat_map(|col| (0..n).map(move |row| elements[row][col]))
        .collect();

    let matrix = DMatrix::from_vec(n, n, flat);

    let eigen = Eigen::new(matrix, false, false)
        .ok_or_else(|| "LAPACK eigenvalue decomposition failed".to_string())?;

    let real_eigenvalues = eigen.eigenvalues_re;
    Ok(real_eigenvalues.as_slice().to_vec())
}

/// Compute eigenvector for a given eigenvalue using LAPACK.
///
/// Returns the eigenvector corresponding to the eigenvalue closest
/// to the given value.
pub fn eigenvector(elements: &[Vec<f64>], eigenvalue: f64) -> Result<Vec<f64>, String> {
    let n = elements.len();
    if n == 0 {
        return Err("Empty matrix".to_string());
    }

    let flat: Vec<f64> = (0..n)
        .flat_map(|col| (0..n).map(move |row| elements[row][col]))
        .collect();

    let matrix = DMatrix::from_vec(n, n, flat);

    let eigen = Eigen::new(matrix, false, true)
        .ok_or_else(|| "LAPACK eigenvalue decomposition failed".to_string())?;

    let eigenvectors = eigen
        .eigenvectors
        .ok_or_else(|| "LAPACK did not compute eigenvectors".to_string())?;

    // Find the eigenvalue closest to the requested one
    let eigenvalues = eigen.eigenvalues_re.as_slice();
    let mut best_idx = 0;
    let mut best_diff = f64::INFINITY;
    for (i, &ev) in eigenvalues.iter().enumerate() {
        let diff = (ev - eigenvalue).abs();
        if diff < best_diff {
            best_diff = diff;
            best_idx = i;
        }
    }

    Ok(eigenvectors.column(best_idx).iter().copied().collect())
}

/// Compute all eigenpairs (eigenvalue, eigenvector) using LAPACK.
pub fn eigenpairs(elements: &[Vec<f64>]) -> Result<Vec<(f64, Vec<f64>)>, String> {
    let n = elements.len();
    if n == 0 {
        return Ok(vec![]);
    }

    let flat: Vec<f64> = (0..n)
        .flat_map(|col| (0..n).map(move |row| elements[row][col]))
        .collect();

    let matrix = DMatrix::from_vec(n, n, flat);

    let eigen = Eigen::new(matrix, false, true)
        .ok_or_else(|| "LAPACK eigenvalue decomposition failed".to_string())?;

    let eigenvectors = eigen
        .eigenvectors
        .ok_or_else(|| "LAPACK did not compute eigenvectors".to_string())?;

    let eigenvalues = eigen.eigenvalues_re.as_slice();
    let mut pairs = Vec::with_capacity(n);
    for (i, &ev) in eigenvalues.iter().enumerate() {
        let vec: Vec<f64> = eigenvectors.column(i).iter().copied().collect();
        pairs.push((ev, vec));
    }

    Ok(pairs)
}

/// QR decomposition using LAPACK.
///
/// Returns (Q, R) as row-major Vec<Vec<f64>> matrices.
pub fn qr_decomposition(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = a.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    let flat: Vec<f64> = (0..n)
        .flat_map(|col| (0..n).map(move |row| a[row][col]))
        .collect();

    let matrix = DMatrix::from_vec(n, n, flat);
    let qr = QR::new(matrix);

    let q_mat = qr.q();
    let r_mat = qr.r();

    let q: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| q_mat[(i, j)]).collect())
        .collect();

    let r: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| r_mat[(i, j)]).collect())
        .collect();

    (q, r)
}

/// Solve a linear system Ax = b using LAPACK LU decomposition.
pub fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }

    let flat: Vec<f64> = (0..n)
        .flat_map(|col| (0..n).map(move |row| a[row][col]))
        .collect();

    let matrix = DMatrix::from_vec(n, n, flat);
    let b_vec = nalgebra::DVector::from_vec(b.to_vec());

    let lu = LU::new(matrix);
    match lu.solve(&b_vec) {
        Some(x) => x.as_slice().to_vec(),
        None => {
            // Singular matrix — fall back to zero vector
            vec![0.0; n]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_eigenvalues_2x2() {
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3
        let m = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let mut evs = eigenvalues(&m).unwrap();
        evs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(approx_eq(evs[0], 1.0, 1e-10));
        assert!(approx_eq(evs[1], 3.0, 1e-10));
    }

    #[test]
    fn test_eigenvalues_3x3_diagonal() {
        let m = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0],
            vec![0.0, 0.0, 3.0],
        ];
        let mut evs = eigenvalues(&m).unwrap();
        evs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(approx_eq(evs[0], 1.0, 1e-10));
        assert!(approx_eq(evs[1], 2.0, 1e-10));
        assert!(approx_eq(evs[2], 3.0, 1e-10));
    }

    #[test]
    fn test_eigenvector() {
        // [[2, 1], [1, 2]] — eigenvector for eigenvalue 3 is [1, 1] (normalized)
        let m = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let ev = eigenvector(&m, 3.0).unwrap();
        // Eigenvector should be proportional to [1, 1]
        let ratio = ev[0] / ev[1];
        assert!(approx_eq(ratio.abs(), 1.0, 1e-10));
    }

    #[test]
    fn test_eigenpairs() {
        let m = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let pairs = eigenpairs(&m).unwrap();
        assert_eq!(pairs.len(), 2);
        // Each eigenvector should satisfy Av = λv
        for (eigenvalue, eigenvector) in &pairs {
            for i in 0..2 {
                let av: f64 = (0..2).map(|j| m[i][j] * eigenvector[j]).sum();
                let lv = eigenvalue * eigenvector[i];
                assert!(approx_eq(av, lv, 1e-10));
            }
        }
    }

    #[test]
    fn test_qr_decomposition() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let (q, r) = qr_decomposition(&a);

        // Q should be orthogonal: Q^T * Q ≈ I
        let n = q.len();
        for i in 0..n {
            for j in 0..n {
                let dot: f64 = (0..n).map(|k| q[k][i] * q[k][j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(dot, expected, 1e-10));
            }
        }

        // Q * R should equal A
        for i in 0..n {
            for j in 0..n {
                let qr_ij: f64 = (0..n).map(|k| q[i][k] * r[k][j]).sum();
                assert!(approx_eq(qr_ij, a[i][j], 1e-10));
            }
        }
    }

    #[test]
    fn test_solve_linear_system() {
        // [[2, 1], [1, 3]] * [x, y] = [5, 10]
        // Solution: x = 1, y = 3
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 10.0];
        let x = solve_linear_system(&a, &b);
        assert!(approx_eq(x[0], 1.0, 1e-10));
        assert!(approx_eq(x[1], 3.0, 1e-10));
    }

    #[test]
    fn test_large_eigenvalues() {
        // 5x5 symmetric matrix
        let m = vec![
            vec![5.0, 1.0, 0.0, 0.0, 0.0],
            vec![1.0, 4.0, 1.0, 0.0, 0.0],
            vec![0.0, 1.0, 3.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0, 2.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0, 1.0],
        ];
        let evs = eigenvalues(&m).unwrap();
        assert_eq!(evs.len(), 5);
        // All eigenvalues should be real and positive for this matrix
        for ev in &evs {
            assert!(*ev > 0.0);
        }
    }
}
