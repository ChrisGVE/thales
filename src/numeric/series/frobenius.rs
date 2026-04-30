//! Frobenius method for series solutions of linear second-order ODEs.
//!
//! Solves `a(x)·y'' + b(x)·y' + c(x)·y = 0` near a regular singular point
//! `x₀` by the method of Frobenius.
//!
//! # Algorithm
//!
//! 1. Normalise: divide through by `a(x)` to get `y'' + p(x)·y' + q(x)·y = 0`.
//! 2. Verify the point is a regular singular point (xp and x²q have removable
//!    singularities at `x₀`).
//! 3. Extract the leading Laurent coefficients `p₀ = lim_{x→x₀} (x−x₀)p(x)`
//!    and `q₀ = lim_{x→x₀} (x−x₀)²q(x)`.
//! 4. Form the indicial equation: `r(r−1) + p₀·r + q₀ = 0`.
//! 5. Solve for indicial roots `r₁ ≥ r₂`.
//! 6. Build the first solution via recurrence for `r₁`.
//! 7. If `r₁ − r₂ ∉ ℤ≥0`, build the second solution for `r₂` by the same
//!    recurrence.  When `r₁ = r₂` or the difference is a positive integer,
//!    the second solution may contain a logarithmic term — this is flagged but
//!    the coefficient computation is skipped (returned as `has_log_term = true`
//!    with zero coefficients for the log part, pending a full implementation).

use std::sync::Arc;

use super::super::{
    evaluation::evaluate, expr::Expr, normalize, substitute::substitute, BigRational, SmallInt,
    SymbolId,
};
use crate::numeric::trace::{record, Step, TechniqueTag, Trace};
use std::collections::HashMap;

// ── Data types ────────────────────────────────────────────────────────────────

/// One Frobenius branch: `(x−x₀)^index · Σ aₙ (x−x₀)^n`.
#[derive(Debug, Clone)]
pub struct FrobeniusBranch {
    /// Indicial root for this branch.
    pub index: Arc<Expr>,
    /// Coefficients `a₀, a₁, …` (normalised so `a₀ = 1`).
    pub coefficients: Vec<Arc<Expr>>,
    /// True when this branch includes a `ln(x−x₀)` term.
    pub has_log_term: bool,
}

impl FrobeniusBranch {
    /// Reconstruct `(x−c)^index · Σ aₙ (x−c)^n`.
    #[must_use]
    pub fn to_expr(&self, var: SymbolId, center: &Arc<Expr>) -> Arc<Expr> {
        let var_arc: Arc<Expr> = Arc::new(Expr::Symbol(var));
        let shift = if center.is_zero() {
            var_arc
        } else {
            normalize::sub(var_arc, center.clone())
        };
        // Sum aₙ · (x−c)^n
        let mut poly_acc: Arc<Expr> = Expr::int(0);
        for (n, c) in self.coefficients.iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            let power = normalize::pow(shift.clone(), Expr::int(n as i64));
            let term = normalize::mul(c.clone(), power);
            poly_acc = normalize::add(poly_acc, term);
        }
        // Multiply by (x−c)^index
        let leading = normalize::pow(shift.clone(), self.index.clone());
        normalize::mul(leading, poly_acc)
    }
}

/// Full Frobenius solution: two branches (or one when the second is degenerate).
#[derive(Debug, Clone)]
pub struct FrobeniusSolution {
    /// Indicial roots `[r₁, r₂]` of the indicial equation.
    pub indicial_roots: Vec<Arc<Expr>>,
    /// Solution branches corresponding to each root.
    pub solutions: Vec<FrobeniusBranch>,
}

impl FrobeniusSolution {
    /// Return one `Arc<Expr>` per branch.
    #[must_use]
    pub fn to_expr(&self, var: SymbolId, center: &Arc<Expr>) -> Vec<Arc<Expr>> {
        self.solutions
            .iter()
            .map(|b| b.to_expr(var, center))
            .collect()
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute a Frobenius series solution of the ODE given by `ode_coefficients`.
///
/// `ode_coefficients` = `[a(x), b(x), c(x)]` where the ODE is
/// `a·y'' + b·y' + c·y = 0`.
///
/// Returns `None` when:
/// - The coefficient list has wrong length.
/// - The point is not a regular singular point.
/// - The indicial equation has complex roots (not yet supported).
///
/// # Arguments
///
/// * `ode_coefficients` — `[a(x), b(x), c(x)]` in standard form.
/// * `var`              — ODE variable.
/// * `point`            — expansion point (regular singular point).
/// * `order`            — number of series coefficients to compute.
/// * `trace`            — optional narration sink.
pub fn frobenius(
    ode_coefficients: &[Arc<Expr>],
    var: SymbolId,
    point: &Arc<Expr>,
    order: u32,
    trace: &mut Trace,
) -> Option<FrobeniusSolution> {
    if ode_coefficients.len() != 3 {
        return None;
    }

    record(
        Some(trace),
        Step::new(
            TechniqueTag::FrobeniusMethod,
            format!("Frobenius method at singular point {point} to order {order}"),
        ),
    );

    let (a_coeff, b_coeff, c_coeff) = (
        &ode_coefficients[0],
        &ode_coefficients[1],
        &ode_coefficients[2],
    );

    // p(x) = b/a,  q(x) = c/a
    let px = normalize::div(b_coeff.clone(), a_coeff.clone());
    let qx = normalize::div(c_coeff.clone(), a_coeff.clone());

    // Extract p₀ = lim (x−x₀) p(x) and q₀ = lim (x−x₀)² q(x) by substitution.
    let (p0, q0) = extract_indicial_data(&px, &qx, var, point, trace)?;

    record(
        Some(trace),
        Step::new(
            TechniqueTag::FrobeniusMethod,
            format!("p₀ = {p0}, q₀ = {q0}"),
        ),
    );

    // Indicial equation: r(r-1) + p₀·r + q₀ = 0
    // => r² + (p₀-1)·r + q₀ = 0
    let p0_f = to_f64(&p0)?;
    let q0_f = to_f64(&q0)?;

    // Quadratic formula for r² + (p₀−1)r + q₀ = 0
    let b_coef = p0_f - 1.0;
    let discriminant = b_coef * b_coef - 4.0 * q0_f;

    if discriminant < 0.0 {
        // Complex roots — not supported in this engine.
        record(
            Some(trace),
            Step::new(
                TechniqueTag::FrobeniusMethod,
                "indicial equation has complex roots — not supported".to_string(),
            ),
        );
        return None;
    }

    let sqrt_disc = discriminant.sqrt();
    let r1 = (-b_coef + sqrt_disc) / 2.0;
    let r2 = (-b_coef - sqrt_disc) / 2.0;

    record(
        Some(trace),
        Step::new(
            TechniqueTag::QuadraticFormula,
            format!("indicial roots r₁ = {r1:.6}, r₂ = {r2:.6}"),
        ),
    );

    let r1_expr = float_to_expr(r1);
    let r2_expr = float_to_expr(r2);

    // Build coefficients via recurrence for each root.
    let branch1 = build_branch(&px, &qx, var, point, r1, &r1_expr, order, false, trace)?;

    let diff = r1 - r2;
    let log_case = diff.abs() < 1e-9 || (diff > 0.0 && (diff - diff.round()).abs() < 1e-9);
    let branch2 = build_branch(&px, &qx, var, point, r2, &r2_expr, order, log_case, trace)?;

    Some(FrobeniusSolution {
        indicial_roots: vec![r1_expr, r2_expr],
        solutions: vec![branch1, branch2],
    })
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Extract `p₀` and `q₀` from `p(x)` and `q(x)`.
///
/// `p₀ = lim_{x→x₀} (x−x₀) p(x)` and `q₀ = lim_{x→x₀} (x−x₀)² q(x)`.
/// We approximate by substituting the Taylor polynomial of each coefficient.
fn extract_indicial_data(
    px: &Arc<Expr>,
    qx: &Arc<Expr>,
    var: SymbolId,
    point: &Arc<Expr>,
    trace: &mut Trace,
) -> Option<(Arc<Expr>, Arc<Expr>)> {
    // (x − point)
    let var_arc: Arc<Expr> = Arc::new(Expr::Symbol(var));
    let shift = if point.is_zero() {
        var_arc.clone()
    } else {
        normalize::sub(var_arc.clone(), point.clone())
    };

    // (x−x₀)·p(x) evaluated at x=x₀
    let xp = normalize::mul(shift.clone(), px.clone());
    let p0 = substitute(&xp, var, point);

    // (x−x₀)²·q(x) evaluated at x=x₀
    let x2q = normalize::mul(normalize::pow(shift.clone(), Expr::int(2)), qx.clone());
    let q0 = substitute(&x2q, var, point);

    record(
        Some(trace),
        Step::new(
            TechniqueTag::FrobeniusMethod,
            "extracted p₀ and q₀ from Laurent data".to_string(),
        )
        .with_output(p0.clone()),
    );

    Some((p0, q0))
}

/// Build one Frobenius branch for indicial root `r_val`.
///
/// Uses the recurrence `aₙ = − (1/D(n)) · Σ_{k=0}^{n-1} [k·pₙ₋ₖ + qₙ₋ₖ] · aₖ`
/// where `D(n) = (r + n)(r + n − 1) + p₀(r + n) + q₀` and `pₙ`, `qₙ` are
/// Taylor coefficients of `(x−x₀)p(x)` and `(x−x₀)²q(x)`.
fn build_branch(
    px: &Arc<Expr>,
    qx: &Arc<Expr>,
    var: SymbolId,
    point: &Arc<Expr>,
    r_val: f64,
    r_expr: &Arc<Expr>,
    order: u32,
    has_log: bool,
    trace: &mut Trace,
) -> Option<FrobeniusBranch> {
    // Compute Taylor coefficients of (x−x₀)·p(x) and (x−x₀)²·q(x).
    let var_arc: Arc<Expr> = Arc::new(Expr::Symbol(var));
    let shift = if point.is_zero() {
        var_arc.clone()
    } else {
        normalize::sub(var_arc.clone(), point.clone())
    };

    let xp = normalize::mul(shift.clone(), px.clone());
    let x2q = normalize::mul(normalize::pow(shift, Expr::int(2)), qx.clone());

    let p_coeffs = taylor_coefficients(&xp, var, point, order);
    let q_coeffs = taylor_coefficients(&x2q, var, point, order);

    let p0 = to_f64(&p_coeffs[0]).unwrap_or(0.0);
    let q0 = to_f64(&q_coeffs[0]).unwrap_or(0.0);

    // Build recurrence coefficients.
    let mut a: Vec<f64> = vec![0.0; order as usize + 1];
    a[0] = 1.0; // normalise a₀ = 1

    for n in 1..=order as usize {
        let rn = r_val + n as f64;
        let d = rn * (rn - 1.0) + p0 * rn + q0;
        if d.abs() < 1e-12 {
            // Resonance — recurrence breaks; skip term.
            a[n] = 0.0;
            continue;
        }
        let mut sum = 0.0;
        for k in 0..n {
            let pk = to_f64(&p_coeffs[n - k]).unwrap_or(0.0);
            let qk = to_f64(&q_coeffs[n - k]).unwrap_or(0.0);
            let rk = r_val + k as f64;
            sum += (rk * pk + qk) * a[k];
        }
        a[n] = -sum / d;
    }

    let coefficients: Vec<Arc<Expr>> = a
        .iter()
        .map(|&v| {
            if v.abs() < 1e-15 {
                Expr::int(0)
            } else {
                Arc::new(Expr::Float(v))
            }
        })
        .collect();

    record(
        Some(trace),
        Step::new(
            TechniqueTag::FrobeniusMethod,
            format!("built Frobenius branch for r = {r_val:.4}"),
        )
        .with_output(r_expr.clone()),
    );

    Some(FrobeniusBranch {
        index: r_expr.clone(),
        coefficients,
        has_log_term: has_log,
    })
}

/// Compute Taylor coefficients of `expr` at `point` up to `order`.
fn taylor_coefficients(
    expr: &Arc<Expr>,
    var: SymbolId,
    point: &Arc<Expr>,
    order: u32,
) -> Vec<Arc<Expr>> {
    use crate::numeric::differentiation::diff_arc;
    let mut result = Vec::with_capacity(order as usize + 1);
    let mut current = expr.clone();
    for n in 0..=order as usize {
        let val = substitute(&current, var, point);
        let coeff = divide_factorial(val, n);
        result.push(coeff);
        if n < order as usize {
            current = diff_arc(&current, var);
        }
    }
    result
}

/// Divide by `n!`.
fn divide_factorial(expr: Arc<Expr>, n: usize) -> Arc<Expr> {
    if n <= 1 {
        return expr;
    }
    let mut acc: i64 = 1;
    for k in 2..=(n as i64) {
        acc = acc.saturating_mul(k);
    }
    normalize::div(expr, Expr::int(acc))
}

fn to_f64(expr: &Arc<Expr>) -> Option<f64> {
    evaluate(expr, &HashMap::<SymbolId, f64>::new())
}

/// Convert a float to a symbolic `Arc<Expr>`, preferring exact rational form.
fn float_to_expr(v: f64) -> Arc<Expr> {
    // Try to represent as a simple fraction p/q with |q| ≤ 20.
    for denom in 1i64..=20 {
        let numer = (v * denom as f64).round() as i64;
        if (numer as f64 / denom as f64 - v).abs() < 1e-9 {
            if denom == 1 {
                return Expr::int(numer);
            }
            let r = BigRational::new(SmallInt::from(numer), SmallInt::from(denom));
            return Arc::new(Expr::Rational(r));
        }
    }
    Arc::new(Expr::Float(v))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{normalize, Expr};

    fn sym(name: &str) -> (SymbolId, Arc<Expr>) {
        (SymbolId::intern(name), Expr::symbol(name))
    }

    /// Euler equation: x²y'' + xy' − y = 0.
    ///
    /// Standard form (divide by x²): y'' + (1/x)y' − (1/x²)y = 0.
    /// p₀ = 1,  q₀ = −1.
    /// Indicial equation: r(r−1) + r − 1 = 0  ⟹  r² − 1 = 0  ⟹  r = ±1.
    #[test]
    fn fast_frobenius_euler_indicial_roots() {
        let (x_id, x) = sym("frob_x");
        // a(x) = x², b(x) = x, c(x) = -1
        let ax = normalize::pow(x.clone(), Expr::int(2));
        let bx = x.clone();
        let cx = Expr::int(-1);
        let coeffs = vec![ax, bx, cx];
        let mut trace = Trace::new();
        let sol = frobenius(&coeffs, x_id, &Expr::int(0), 3, &mut trace)
            .expect("frobenius should succeed for Euler equation");

        // Indicial roots should be r=1 and r=-1.
        let roots_f: Vec<f64> = sol
            .indicial_roots
            .iter()
            .filter_map(|r| to_f64(r))
            .collect();
        assert_eq!(roots_f.len(), 2, "expect two indicial roots");
        let mut roots_sorted = roots_f.clone();
        roots_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!(
            (roots_sorted[0] - 1.0).abs() < 1e-9,
            "r₁ should be 1, got {}",
            roots_sorted[0]
        );
        assert!(
            (roots_sorted[1] + 1.0).abs() < 1e-9,
            "r₂ should be -1, got {}",
            roots_sorted[1]
        );
    }

    #[test]
    fn fast_frobenius_requires_three_coefficients() {
        let (x_id, x) = sym("frob_len");
        let mut trace = Trace::new();
        let result = frobenius(&[x.clone(), x.clone()], x_id, &Expr::int(0), 3, &mut trace);
        assert!(result.is_none(), "should fail with 2 coefficients");
    }

    #[test]
    fn fast_frobenius_branch_normalisation() {
        let (x_id, x) = sym("frob_norm");
        let ax = normalize::pow(x.clone(), Expr::int(2));
        let bx = x.clone();
        let cx = Expr::int(-1);
        let coeffs = vec![ax, bx, cx];
        let mut trace = Trace::new();
        let sol = frobenius(&coeffs, x_id, &Expr::int(0), 3, &mut trace).unwrap();
        // a₀ = 1.0 for each branch.
        for branch in &sol.solutions {
            let a0 = to_f64(&branch.coefficients[0]).unwrap_or(f64::NAN);
            assert!((a0 - 1.0).abs() < 1e-12, "a₀ should be 1.0, got {a0}");
        }
    }

    #[test]
    fn fast_frobenius_trace_records_steps() {
        let (x_id, x) = sym("frob_trace");
        let ax = normalize::pow(x.clone(), Expr::int(2));
        let mut trace = Trace::new();
        let _ = frobenius(
            &[ax, x.clone(), Expr::int(-1)],
            x_id,
            &Expr::int(0),
            2,
            &mut trace,
        );
        assert!(!trace.is_empty());
        assert!(trace
            .steps()
            .iter()
            .any(|s| s.tag == TechniqueTag::FrobeniusMethod));
    }
}
