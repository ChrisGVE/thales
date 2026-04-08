//! Brent's method (hybrid root finder, very robust).

use crate::ast::{Equation, Expression, Variable};
use crate::resolution_path::{Operation, ResolutionPath, ResolutionPathBuilder};
use std::collections::HashMap;

use super::{NumericalConfig, NumericalError, NumericalResult, NumericalSolution};

/// Brent's method (hybrid root finder, very robust).
///
/// Implements Brent's method, a root-finding algorithm that combines the
/// reliability of bisection with the speed of the secant method and inverse
/// quadratic interpolation.
///
/// At each step the algorithm picks the fastest method whose step falls within
/// the current bracket and is smaller than half the previous step; otherwise it
/// falls back to bisection.  This guarantees convergence whenever the root is
/// bracketed while achieving super-linear convergence in practice.
///
/// # Example
///
/// ```
/// use thales::numerical::{BrentsMethod, NumericalConfig};
/// use thales::ast::{Equation, Expression, Variable};
///
/// // Solve x² = 2  (find √2)
/// let equation = Equation::new(
///     "sqrt2",
///     Expression::Power(
///         Box::new(Expression::Variable(Variable::new("x"))),
///         Box::new(Expression::Integer(2)),
///     ),
///     Expression::Integer(2),
/// );
///
/// let solver = BrentsMethod::with_default_config();
/// let (solution, _path) = solver.solve(&equation, &Variable::new("x"), (1.0, 2.0)).unwrap();
///
/// assert!((solution.value - std::f64::consts::SQRT_2).abs() < 1e-10);
/// assert!(solution.converged);
/// ```
#[derive(Debug)]
pub struct BrentsMethod {
    config: NumericalConfig,
}

impl BrentsMethod {
    /// Creates a new Brent's method solver with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Numerical configuration (tolerance, iterations, etc.)
    pub fn new(config: NumericalConfig) -> Self {
        Self { config }
    }

    /// Creates a new Brent's method solver with default configuration.
    pub fn with_default_config() -> Self {
        Self {
            config: NumericalConfig::default(),
        }
    }

    /// Find a root using Brent's method.
    ///
    /// Requires a bracketing interval `(a, b)` such that `f(a)` and `f(b)` have
    /// opposite signs.  Returns an error if the interval does not bracket a root.
    ///
    /// # Errors
    ///
    /// * `NumericalError::Other` – interval does not bracket a root
    /// * `NumericalError::NoConvergence` – max iterations reached
    /// * `NumericalError::EvaluationFailed` – function evaluation failed
    pub fn solve(
        &self,
        equation: &Equation,
        variable: &Variable,
        interval: (f64, f64),
    ) -> NumericalResult<(NumericalSolution, ResolutionPath)> {
        let (f, eval) = brent_make_eval(equation, variable);
        let (mut a, mut b, mut fa, mut fb) = brent_init_bracket(interval, &eval)?;

        let mut path = ResolutionPathBuilder::new(f);
        path = path.step(
            Operation::NumericalApproximation,
            format!("Starting Brent on [{a}, {b}]: f(a)={fa:.6e}, f(b)={fb:.6e}"),
            Expression::Float(b),
        );

        let mut st = BrentState {
            c: a,
            fc: fa,
            d: b - a,
            e: b - a,
            mflag: true,
        };
        let mut iterations = 0;
        let mut converged = false;

        for i in 0..self.config.max_iterations {
            iterations = i + 1;
            if fb.abs() < self.config.tolerance {
                converged = true;
                break;
            }

            let (s, bisected) = brent_next_point(a, b, fa, fb, &st, self.config.tolerance);
            st.mflag = bisected;
            let fs = eval(s)?;

            if i % 10 == 0 {
                let method = if bisected { "bisect" } else { "interpolate" };
                path = path.step(
                    Operation::NumericalApproximation,
                    format!("Iter {iterations}: x={s:.10}, f(x)={fs:.6e} [{method}]"),
                    Expression::Float(s),
                );
            }

            brent_update_bracket(&mut a, &mut b, &mut st, &mut fa, &mut fb, s, fs);
            if (b - a).abs() < self.config.tolerance {
                converged = true;
                break;
            }
        }

        if !converged {
            return Err(NumericalError::NoConvergence);
        }

        path = path.step(
            Operation::NumericalApproximation,
            format!("Converged: x={b:.15}, |f(x)|={:.6e}", fb.abs()),
            Expression::Float(b),
        );
        let sol = NumericalSolution {
            value: b,
            iterations,
            residual: fb.abs(),
            converged,
        };
        Ok((sol, path.finish(Expression::Float(b))))
    }
}

// ============================================================================
// Brent's method helpers
// ============================================================================

/// Auxiliary state carried between iterations of Brent's method.
struct BrentState {
    /// Previous best estimate (before last swap)
    c: f64,
    /// f(c)
    fc: f64,
    /// Step from the iteration before last
    d: f64,
    /// Step from last iteration
    e: f64,
    /// Was the last accepted step a bisection?
    mflag: bool,
}

/// Build the residual expression `f(x) = lhs - rhs` and a point-evaluator closure.
///
/// The returned closure owns all data it needs (cloned from the inputs).
fn brent_make_eval(
    equation: &Equation,
    variable: &Variable,
) -> (Expression, impl Fn(f64) -> NumericalResult<f64>) {
    let f = Expression::Binary(
        crate::ast::BinaryOp::Sub,
        Box::new(equation.left.clone()),
        Box::new(equation.right.clone()),
    );
    let f_owned = f.clone();
    let var_name = variable.name.clone();
    let eval = move |xv: f64| -> NumericalResult<f64> {
        let mut vars = HashMap::new();
        vars.insert(var_name.clone(), xv);
        f_owned.evaluate(&vars).ok_or_else(|| {
            NumericalError::EvaluationFailed(format!("Failed to evaluate at x = {xv}"))
        })
    };
    (f, eval)
}

/// Validate and orient the initial bracket for Brent's method.
///
/// Returns `(a, b, fa, fb)` with the guarantee that `|f(b)| <= |f(a)|`.
/// Errors if `f(a)` and `f(b)` have the same sign (no root bracketed).
fn brent_init_bracket(
    interval: (f64, f64),
    eval: &impl Fn(f64) -> NumericalResult<f64>,
) -> NumericalResult<(f64, f64, f64, f64)> {
    let (mut a, mut b) = interval;
    let mut fa = eval(a)?;
    let mut fb = eval(b)?;
    if fa * fb > 0.0 {
        return Err(NumericalError::Other(format!(
            "Brent's method requires f(a) and f(b) to have opposite signs. \
             f({}) = {}, f({}) = {}",
            a, fa, b, fb
        )));
    }
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }
    Ok((a, b, fa, fb))
}

/// Compute and validate the next Brent step, falling back to bisection when needed.
///
/// Returns `(step_value, used_bisection)`.
fn brent_next_point(a: f64, b: f64, fa: f64, fb: f64, st: &BrentState, tol: f64) -> (f64, bool) {
    let s = brent_interpolation_step(a, b, st.c, fa, fb, st.fc, tol);
    let bisect_mid = (a + b) / 2.0;
    let lo = (3.0 * a + b) / 4.0;
    let hi = b;
    let in_bracket = if lo <= hi {
        lo <= s && s <= hi
    } else {
        hi <= s && s <= lo
    };
    let reject = !in_bracket
        || (st.mflag && (s - b).abs() >= (b - st.c).abs() / 2.0)
        || (!st.mflag && (s - b).abs() >= st.e.abs() / 2.0)
        || (st.mflag && (b - st.c).abs() < tol)
        || (!st.mflag && st.e.abs() < tol);
    if reject {
        (bisect_mid, true)
    } else {
        (s, false)
    }
}

/// Update the bracket and `BrentState` after evaluating the accepted step.
///
/// Maintains `f(a) * f(b) <= 0` and `|f(b)| <= |f(a)|`.
fn brent_update_bracket(
    a: &mut f64,
    b: &mut f64,
    st: &mut BrentState,
    fa: &mut f64,
    fb: &mut f64,
    s_final: f64,
    fs: f64,
) {
    st.d = st.e;
    st.e = *b - *a;
    st.c = *b;
    st.fc = *fb;
    if *fa * fs < 0.0 {
        *b = s_final;
        *fb = fs;
    } else {
        *a = s_final;
        *fa = fs;
    }
    if fa.abs() < fb.abs() {
        std::mem::swap(a, b);
        std::mem::swap(fa, fb);
    }
}

/// Compute the raw interpolation candidate (IQI or secant) for Brent's method.
///
/// Returns a value that may lie outside the bracket; the caller (`brent_next_point`)
/// decides whether to use it or fall back to bisection.
fn brent_interpolation_step(a: f64, b: f64, c: f64, fa: f64, fb: f64, fc: f64, tol: f64) -> f64 {
    if (fa - fc).abs() > tol && (fb - fc).abs() > tol {
        let r = fb / fc;
        let s = fb / fa;
        let t = fa / fc;
        let p = s * (t * (r - t) * (c - b) - (1.0 - r) * (b - a));
        let q = (t - 1.0) * (r - 1.0) * (s - 1.0);
        if q.abs() > tol {
            return b + p / q;
        }
    }
    let denom = fb - fa;
    if denom.abs() > tol {
        b - fb * (b - a) / denom
    } else {
        b + 2.0 * (b - a) // signal: outside bracket, force bisection
    }
}
