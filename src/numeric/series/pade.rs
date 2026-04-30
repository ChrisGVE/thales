//! Padé approximant construction from a Taylor series.
//!
//! A `[m/n]` Padé approximant of `f(x)` at `center` is the rational function
//! `P(x)/Q(x)` where `deg P ≤ m`, `deg Q ≤ n`, `Q(center) = 1`, and
//! `P/Q − f = O((x − center)^{m+n+1})`.
//!
//! # Algorithm
//!
//! 1. Compute the Taylor series `f(x) = Σ cₖ (x−center)^k` to order `m+n`.
//! 2. Write `P = Σ pᵢ t^i` (deg ≤ m) and `Q = 1 + Σ qⱼ t^j` (deg ≤ n),
//!    where `t = x − center`.
//! 3. Matching coefficients `[P = Q·(Σ cₖ tᵏ)] mod t^{m+n+1}` gives a linear
//!    system for `q₁, …, qₙ`:
//!    `Σ_{j=1}^{n} cₘ₊ₖ₋ⱼ qⱼ = −cₘ₊ₖ`   for `k = 1, …, n`.
//! 4. Solve that system (Gaussian elimination on `f64` data).
//! 5. Back-substitute to obtain `p₀, …, pₘ`.

use std::sync::Arc;

use super::super::{expr::Expr, normalize, SymbolId};
use super::taylor::taylor;
use crate::numeric::trace::{record, Step, TechniqueTag, Trace};

// ── Data type ─────────────────────────────────────────────────────────────────

/// A `[m/n]` Padé approximant `P(x)/Q(x)`.
///
/// Coefficients are stored as `[a₀, a₁, …]` for the polynomial
/// `a₀ + a₁(x−center) + a₂(x−center)² + …`.
/// `Q[0]` is always `1` (normalisation).
#[derive(Debug, Clone)]
pub struct PadeApproximant {
    /// Numerator coefficients `p₀, …, pₘ`.
    pub numerator: Vec<Arc<Expr>>,
    /// Denominator coefficients `q₀=1, q₁, …, qₙ`.
    pub denominator: Vec<Arc<Expr>>,
    /// Expansion variable.
    pub var: SymbolId,
    /// Expansion centre.
    pub center: Arc<Expr>,
    /// Numerator degree.
    pub m: u32,
    /// Denominator degree.
    pub n: u32,
}

impl PadeApproximant {
    /// Reconstruct the rational expression `P(x) / Q(x)`.
    #[must_use]
    pub fn to_expr(&self) -> Arc<Expr> {
        let var_arc: Arc<Expr> = Arc::new(Expr::Symbol(self.var));
        let t = if self.center.is_zero() {
            var_arc
        } else {
            normalize::sub(var_arc, self.center.clone())
        };

        let p = poly_eval(&self.numerator, &t);
        let q = poly_eval(&self.denominator, &t);
        normalize::div(p, q)
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute a `[m/n]` Padé approximant of `expr` at `center`.
///
/// Returns `None` when the linear system is singular (denominator
/// polynomial degenerates).
///
/// # Arguments
///
/// * `expr`   — expression to approximate.
/// * `var`    — the expansion variable.
/// * `center` — expansion point.
/// * `m`      — numerator degree.
/// * `n`      — denominator degree.
/// * `trace`  — optional narration sink.
pub fn pade(
    expr: &Arc<Expr>,
    var: SymbolId,
    center: &Arc<Expr>,
    m: u32,
    n: u32,
    trace: &mut Trace,
) -> Option<PadeApproximant> {
    record(
        Some(trace),
        Step::new(
            TechniqueTag::PadeApproximant,
            format!("[{m}/{n}] Padé approximant at {center}"),
        )
        .with_input(expr.clone()),
    );

    let total_order = (m + n) as usize;
    let ts = taylor(expr, var, center, total_order);

    // Extract f64 Taylor coefficients c[0..=m+n].
    let c: Vec<f64> = (0..=total_order)
        .map(|k| expr_to_f64(&ts.coeff(k)))
        .collect();

    record(
        Some(trace),
        Step::new(
            TechniqueTag::PadeApproximant,
            format!("Taylor coefficients computed to order {total_order}"),
        ),
    );

    // Solve for denominator coefficients q[1..=n].
    let q_coeffs = solve_denominator(&c, m as usize, n as usize)?;

    record(
        Some(trace),
        Step::new(
            TechniqueTag::PadeApproximant,
            "solved linear system for denominator coefficients".to_string(),
        ),
    );

    // Compute numerator coefficients p[0..=m].
    let p_coeffs = compute_numerator(&c, &q_coeffs, m as usize);

    // Convert to Arc<Expr>.
    let numerator: Vec<Arc<Expr>> = p_coeffs.iter().map(|&v| float_to_expr(v)).collect();
    let mut denominator: Vec<Arc<Expr>> = vec![Expr::int(1)];
    denominator.extend(q_coeffs.iter().map(|&v| float_to_expr(v)));

    record(
        Some(trace),
        Step::new(
            TechniqueTag::PadeApproximant,
            format!("numerator deg {m}, denominator deg {n}"),
        ),
    );

    Some(PadeApproximant {
        numerator,
        denominator,
        var,
        center: center.clone(),
        m,
        n,
    })
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Solve the `n×n` linear system for denominator coefficients.
///
/// The system is: for k = 1, …, n:
///   `Σ_{j=1}^{n} c[m+k-j] · q[j] = −c[m+k]`
fn solve_denominator(c: &[f64], m: usize, n: usize) -> Option<Vec<f64>> {
    if n == 0 {
        return Some(vec![]);
    }

    // Build augmented matrix [A | b] where A[i][j] = c[m+i+1-j-1] = c[m+i-j].
    let mut mat: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row: Vec<f64> = (0..n)
                .map(|j| {
                    let idx = (m + i).saturating_sub(j);
                    *c.get(idx).unwrap_or(&0.0)
                })
                .collect();
            // RHS: −c[m+1+i]
            let rhs_idx = m + 1 + i;
            row.push(-(*c.get(rhs_idx).unwrap_or(&0.0)));
            row
        })
        .collect();

    gaussian_elimination(&mut mat)
}

/// Gaussian elimination with partial pivoting on an `n × (n+1)` augmented
/// matrix. Returns the solution vector of length `n`.
fn gaussian_elimination(mat: &mut Vec<Vec<f64>>) -> Option<Vec<f64>> {
    let n = mat.len();
    for col in 0..n {
        // Find pivot.
        let pivot_row = (col..n).max_by(|&a, &b| {
            mat[a][col]
                .abs()
                .partial_cmp(&mat[b][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        mat.swap(col, pivot_row);

        let pivot = mat[col][col];
        if pivot.abs() < 1e-12 {
            return None; // Singular system.
        }

        // Normalise pivot row.
        for j in col..=n {
            mat[col][j] /= pivot;
        }

        // Eliminate.
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = mat[row][col];
            for j in col..=n {
                let val = mat[col][j] * factor;
                mat[row][j] -= val;
            }
        }
    }

    Some((0..n).map(|i| mat[i][n]).collect())
}

/// Compute numerator coefficients from Taylor coefficients and `q`.
///
/// `p[k] = Σ_{j=0}^{min(k,n)} c[k-j] · q[j]`  for `k = 0, …, m`.
/// (With `q[0] = 1`.)
fn compute_numerator(c: &[f64], q: &[f64], m: usize) -> Vec<f64> {
    let n = q.len();
    (0..=m)
        .map(|k| {
            let mut sum = c[k]; // j = 0 term, q[0] = 1
            for j in 1..=n.min(k) {
                sum += c[k - j] * q[j - 1];
            }
            sum
        })
        .collect()
}

/// Evaluate a polynomial given as coefficient vector `[a₀, a₁, …]` at `t`.
fn poly_eval(coeffs: &[Arc<Expr>], t: &Arc<Expr>) -> Arc<Expr> {
    if coeffs.is_empty() {
        return Expr::int(0);
    }
    let mut acc: Arc<Expr> = Expr::int(0);
    for (k, c) in coeffs.iter().enumerate() {
        if c.is_zero() {
            continue;
        }
        let power = normalize::pow(t.clone(), Expr::int(k as i64));
        let term = normalize::mul(c.clone(), power);
        acc = normalize::add(acc, term);
    }
    acc
}

/// Convert `Arc<Expr>` to `f64` via `Display`-based numeric matching.
fn expr_to_f64(expr: &Arc<Expr>) -> f64 {
    use crate::numeric::evaluation::evaluate;
    use std::collections::HashMap;
    evaluate(expr, &HashMap::<SymbolId, f64>::new()).unwrap_or(0.0)
}

/// Convert `f64` to `Arc<Expr>`, preferring exact rational form for small fractions.
fn float_to_expr(v: f64) -> Arc<Expr> {
    if v.abs() < 1e-15 {
        return Expr::int(0);
    }
    // Try p/q with |q| ≤ 20.
    for denom in 1i64..=20 {
        let numer = (v * denom as f64).round() as i64;
        if (numer as f64 / denom as f64 - v).abs() < 1e-9 {
            if denom == 1 {
                return Expr::int(numer);
            }
            let r = crate::numeric::BigRational::new(
                crate::numeric::SmallInt::from(numer),
                crate::numeric::SmallInt::from(denom),
            );
            return Arc::new(Expr::Rational(r));
        }
    }
    Arc::new(Expr::Float(v))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{FuncId, SymbolId};

    fn sym(name: &str) -> (SymbolId, Arc<Expr>) {
        (SymbolId::intern(name), Expr::symbol(name))
    }

    /// [1/1] Padé of exp(x) at 0 = (1 + x/2) / (1 − x/2).
    #[test]
    fn fast_pade_exp_1_1() {
        let (x_id, x) = sym("pade_exp11_x");
        let expr = Expr::func(FuncId::Exp, vec![x]);
        let mut trace = Trace::new();
        let pa =
            pade(&expr, x_id, &Expr::int(0), 1, 1, &mut trace).expect("pade [1/1] should succeed");

        // p₀ = 1, p₁ = 1/2
        let p0 = expr_to_f64(&pa.numerator[0]);
        let p1 = expr_to_f64(&pa.numerator[1]);
        assert!((p0 - 1.0).abs() < 1e-9, "p₀ = 1, got {p0}");
        assert!((p1 - 0.5).abs() < 1e-9, "p₁ = 1/2, got {p1}");

        // q₀ = 1, q₁ = -1/2
        let q1 = expr_to_f64(&pa.denominator[1]);
        assert!((q1 + 0.5).abs() < 1e-9, "q₁ = -1/2, got {q1}");
    }

    /// [2/2] Padé of exp(x) at 0 — known coefficients.
    #[test]
    fn fast_pade_exp_2_2() {
        let (x_id, x) = sym("pade_exp22_x");
        let expr = Expr::func(FuncId::Exp, vec![x]);
        let mut trace = Trace::new();
        let pa =
            pade(&expr, x_id, &Expr::int(0), 2, 2, &mut trace).expect("pade [2/2] should succeed");

        // The [2/2] Padé of exp is (1 + x/2 + x²/12) / (1 − x/2 + x²/12).
        let p0 = expr_to_f64(&pa.numerator[0]);
        let p1 = expr_to_f64(&pa.numerator[1]);
        let p2 = expr_to_f64(&pa.numerator[2]);
        assert!((p0 - 1.0).abs() < 1e-9, "p₀ = 1");
        assert!((p1 - 0.5).abs() < 1e-9, "p₁ = 1/2");
        assert!((p2 - 1.0 / 12.0).abs() < 1e-9, "p₂ = 1/12, got {p2}");

        let q1 = expr_to_f64(&pa.denominator[1]);
        let q2 = expr_to_f64(&pa.denominator[2]);
        assert!((q1 + 0.5).abs() < 1e-9, "q₁ = -1/2, got {q1}");
        assert!((q2 - 1.0 / 12.0).abs() < 1e-9, "q₂ = 1/12, got {q2}");
    }

    /// [0/0] degenerates to the constant term of the Taylor series.
    #[test]
    fn fast_pade_zero_zero() {
        let (x_id, x) = sym("pade_00_x");
        let expr = Expr::func(FuncId::Exp, vec![x]);
        let mut trace = Trace::new();
        let pa =
            pade(&expr, x_id, &Expr::int(0), 0, 0, &mut trace).expect("pade [0/0] should succeed");
        let p0 = expr_to_f64(&pa.numerator[0]);
        assert!((p0 - 1.0).abs() < 1e-9, "p₀ = 1 for exp at 0");
    }

    #[test]
    fn fast_pade_to_expr_no_panic() {
        let (x_id, x) = sym("pade_te_x");
        let expr = Expr::func(FuncId::Exp, vec![x]);
        let mut trace = Trace::new();
        let pa = pade(&expr, x_id, &Expr::int(0), 1, 1, &mut trace).unwrap();
        let e = pa.to_expr();
        assert!(!e.is_zero());
    }

    #[test]
    fn fast_pade_trace_records_steps() {
        let (x_id, x) = sym("pade_trace_x");
        let expr = Expr::func(FuncId::Exp, vec![x]);
        let mut trace = Trace::new();
        let _ = pade(&expr, x_id, &Expr::int(0), 1, 1, &mut trace);
        assert!(!trace.is_empty());
        assert!(trace
            .steps()
            .iter()
            .any(|s| s.tag == TechniqueTag::PadeApproximant));
    }
}
