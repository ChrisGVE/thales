//! Fourth-Order Runge-Kutta (RK4) Numerical ODE Solver
//!
//! Provides RK4 integration for first-order ODEs and systems. Use this as a
//! numerical fallback when symbolic methods are unavailable.
//!
//! # Examples
//!
//! ```rust
//! use thales::runge_kutta::{rk4_solve, Rk4Config};
//!
//! // Solve y' = y, y(0) = 1  =>  y(x) = e^x
//! let config = Rk4Config::new(0.0, 1.0, 2.0, 200);
//! let result = rk4_solve(|_x, y| y, config).unwrap();
//! // y(2.0) ≈ e² ≈ 7.389
//! assert!((result.y_final - std::f64::consts::E.powi(2)).abs() < 1e-4);
//! ```

use crate::ode::ODEError;

/// Configuration for an RK4 integration run.
#[derive(Debug, Clone)]
pub struct Rk4Config {
    /// Initial value of the independent variable.
    pub x0: f64,
    /// Initial value of the dependent variable y(x0).
    pub y0: f64,
    /// Final value of the independent variable.
    pub x_end: f64,
    /// Number of integration steps (must be ≥ 1).
    pub steps: usize,
}

impl Rk4Config {
    /// Create a new [`Rk4Config`].
    ///
    /// # Arguments
    ///
    /// * `x0`    – initial x
    /// * `y0`    – initial y(x0)
    /// * `x_end` – target x value
    /// * `steps` – number of RK4 steps (more → more accurate)
    #[must_use]
    pub fn new(x0: f64, y0: f64, x_end: f64, steps: usize) -> Self {
        Self {
            x0,
            y0,
            x_end,
            steps,
        }
    }
}

/// Output of a completed RK4 integration.
#[derive(Debug, Clone)]
pub struct Rk4Solution {
    /// Final value of the independent variable (equals `config.x_end`).
    pub x_final: f64,
    /// Approximated value of y at `x_final`.
    pub y_final: f64,
    /// All (x, y) pairs collected during integration, including the
    /// initial point and every step endpoint.
    pub trajectory: Vec<(f64, f64)>,
}

/// Integrate a scalar first-order ODE y' = f(x, y) using RK4.
///
/// Returns [`Rk4Solution`] containing the trajectory and final value.
///
/// # Arguments
///
/// * `f`      – the right-hand side function f(x, y)
/// * `config` – integration parameters
///
/// # Errors
///
/// Returns [`ODEError::CannotSolve`] when `steps` is zero.
pub fn rk4_solve<F>(f: F, config: Rk4Config) -> Result<Rk4Solution, ODEError>
where
    F: Fn(f64, f64) -> f64,
{
    if config.steps == 0 {
        return Err(ODEError::CannotSolve(
            "RK4 requires at least one step".to_string(),
        ));
    }

    let h = (config.x_end - config.x0) / config.steps as f64;
    let mut x = config.x0;
    let mut y = config.y0;
    let mut trajectory = Vec::with_capacity(config.steps + 1);
    trajectory.push((x, y));

    for _ in 0..config.steps {
        let k1 = f(x, y);
        let k2 = f(x + 0.5 * h, y + 0.5 * h * k1);
        let k3 = f(x + 0.5 * h, y + 0.5 * h * k2);
        let k4 = f(x + h, y + h * k3);

        y += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        x += h;
        trajectory.push((x, y));
    }

    Ok(Rk4Solution {
        x_final: x,
        y_final: y,
        trajectory,
    })
}

/// Output of a completed RK4 system integration.
#[derive(Debug, Clone)]
pub struct Rk4SystemSolution {
    /// Final value of the independent variable.
    pub x_final: f64,
    /// Approximated values of y_i at x_final.
    pub y_final: Vec<f64>,
    /// All (x, [y_1, ..., y_n]) pairs collected during integration.
    pub trajectory: Vec<(f64, Vec<f64>)>,
}

/// Integrate a system of first-order ODEs y' = F(x, y) using RK4.
///
/// The system is supplied as a closure mapping `(x, &[y]) -> Vec<f64>`.
///
/// # Arguments
///
/// * `f`      – system function F(x, y_vec) → dy/dx vector
/// * `x0`     – initial x
/// * `y0`     – initial condition vector
/// * `x_end`  – final x
/// * `steps`  – number of integration steps
///
/// # Errors
///
/// Returns [`ODEError::CannotSolve`] when `steps` is zero or the system
/// function returns an inconsistently-sized vector.
pub fn rk4_system_solve<F>(
    f: F,
    x0: f64,
    y0: Vec<f64>,
    x_end: f64,
    steps: usize,
) -> Result<Rk4SystemSolution, ODEError>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    if steps == 0 {
        return Err(ODEError::CannotSolve("steps must be ≥ 1".into()));
    }

    let n = y0.len();
    let h = (x_end - x0) / steps as f64;
    let mut x = x0;
    let mut y = y0.clone();
    let mut trajectory = Vec::with_capacity(steps + 1);
    trajectory.push((x, y.clone()));

    for _ in 0..steps {
        let k1 = f(x, &y);
        validate_system_size(&k1, n)?;

        let y_mid1: Vec<f64> = y
            .iter()
            .zip(&k1)
            .map(|(yi, ki)| yi + 0.5 * h * ki)
            .collect();
        let k2 = f(x + 0.5 * h, &y_mid1);
        validate_system_size(&k2, n)?;

        let y_mid2: Vec<f64> = y
            .iter()
            .zip(&k2)
            .map(|(yi, ki)| yi + 0.5 * h * ki)
            .collect();
        let k3 = f(x + 0.5 * h, &y_mid2);
        validate_system_size(&k3, n)?;

        let y_end: Vec<f64> = y.iter().zip(&k3).map(|(yi, ki)| yi + h * ki).collect();
        let k4 = f(x + h, &y_end);
        validate_system_size(&k4, n)?;

        y = y
            .iter()
            .enumerate()
            .map(|(i, yi)| yi + h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0)
            .collect();
        x += h;
        trajectory.push((x, y.clone()));
    }

    Ok(Rk4SystemSolution {
        x_final: x,
        y_final: y,
        trajectory,
    })
}

/// Check that a system output has the expected dimension.
fn validate_system_size(output: &[f64], expected: usize) -> Result<(), ODEError> {
    if output.len() != expected {
        return Err(ODEError::CannotSolve(format!(
            "System function returned {} values, expected {}",
            output.len(),
            expected
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rk4_exponential_growth() {
        // y' = y, y(0) = 1  =>  y(x) = e^x
        let config = Rk4Config::new(0.0, 1.0, 1.0, 1000);
        let sol = rk4_solve(|_x, y| y, config).unwrap();
        assert!((sol.y_final - std::f64::consts::E).abs() < 1e-6);
    }

    #[test]
    fn test_rk4_trajectory_length() {
        let config = Rk4Config::new(0.0, 1.0, 1.0, 10);
        let sol = rk4_solve(|_x, y| y, config).unwrap();
        // trajectory has initial point + 10 steps = 11 entries
        assert_eq!(sol.trajectory.len(), 11);
    }

    #[test]
    fn test_rk4_zero_steps_error() {
        let config = Rk4Config::new(0.0, 1.0, 1.0, 0);
        let result = rk4_solve(|_x, y| y, config);
        assert!(matches!(result, Err(ODEError::CannotSolve(_))));
    }

    #[test]
    fn test_rk4_simple_linear() {
        // y' = 2, y(0) = 0  =>  y(x) = 2x
        let config = Rk4Config::new(0.0, 0.0, 3.0, 300);
        let sol = rk4_solve(|_x, _y| 2.0, config).unwrap();
        assert!((sol.y_final - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_rk4_system_solve() {
        // Convert y'' + y = 0 to system:
        //   u0' = u1,  u1' = -u0
        // Initial: u0(0) = 1, u1(0) = 0  =>  u0 = cos(x)
        let sol = rk4_system_solve(
            |_x, u| vec![u[1], -u[0]],
            0.0,
            vec![1.0, 0.0],
            std::f64::consts::PI,
            10_000,
        )
        .unwrap();
        // cos(π) = -1
        assert!((sol.y_final[0] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_rk4_system_harmonic_oscillator() {
        // y' = [y2, -y1], y(0) = [1, 0]  =>  y1 = cos(t), y2 = -sin(t)
        // At t = π: y1 = cos(π) = -1, y2 = -sin(π) ≈ 0
        let sol = rk4_system_solve(
            |_t, y| vec![y[1], -y[0]],
            0.0,
            vec![1.0, 0.0],
            std::f64::consts::PI,
            10_000,
        )
        .unwrap();
        assert!((sol.y_final[0] - (-1.0)).abs() < 1e-6, "y1(π) ≈ -1");
        assert!(sol.y_final[1].abs() < 1e-6, "y2(π) ≈ 0");
        assert!((sol.x_final - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn test_rk4_system_trajectory_length() {
        let steps = 50;
        let sol =
            rk4_system_solve(|_t, y| vec![y[1], -y[0]], 0.0, vec![1.0, 0.0], 1.0, steps).unwrap();
        assert_eq!(sol.trajectory.len(), steps + 1);
    }

    #[test]
    fn test_rk4_system_zero_steps_error() {
        let result = rk4_system_solve(|_x, u| vec![u[0]], 0.0, vec![1.0], 1.0, 0);
        assert!(matches!(result, Err(ODEError::CannotSolve(_))));
    }
}
