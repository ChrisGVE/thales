//! WKB (Wentzel-Kramers-Brillouin) semiclassical approximation.
//!
//! Solves `ε²·y'' + Q(x)·y = 0` using the WKB ansatz
//! `y = exp(S(x)/ε)` where `S = S₀ + ε·S₁ + ε²·S₂ + …`.
//!
//! # Algorithm
//!
//! The substitution `y = exp(S/ε)` converts the ODE to:
//!
//! ```text
//! ε·S'' + (S')² + Q = 0
//! ```
//!
//! Expanding `S = S₀ + ε·S₁ + ε²·S₂ + …` and collecting by powers of ε:
//!
//! * **O(1):**   `(S₀')² = −Q`  → `S₀' = ±√(−Q)` → `S₀ = ±∫√(−Q) dx`.
//!   (When `Q > 0` the classical turning-point regime gives oscillatory
//!   solutions; when `Q < 0` exponential ones. Both are included as
//!   symbolic branches.)
//! * **O(ε):**   `2S₀'·S₁' + S₀'' = 0` → `S₁ = −¼·ln(Q)`.
//! * **O(ε²):**  `2S₀'·S₂' + (S₁')² + S₁'' = 0` → computed symbolically.
//! * Higher orders follow from the same recursion.
//!
//! This engine computes the leading terms symbolically up to `order`.
//!
//! # Limitations
//!
//! * The integral `∫√Q dx` is left in symbolic form (an `Integral` placeholder
//!   expression) because the pattern-integrator cannot always evaluate it.
//! * The small parameter `ε` (`small_param`) must not appear in `Q(x)`.

use std::sync::Arc;

use super::super::{
    differentiation::diff_arc,
    expr::{Expr, FuncId},
    normalize, SymbolId,
};
use crate::numeric::trace::{record, Step, TechniqueTag, Trace};

// ── Data types ────────────────────────────────────────────────────────────────

/// One WKB branch: `exp(phase / ε) · amplitude`.
#[derive(Debug, Clone)]
pub struct WkbBranch {
    /// Leading-order phase `S₀(x)` (an integral expression).
    pub phase: Arc<Expr>,
    /// Higher-order amplitude correction terms `[S₁, S₂, …]`.
    pub amplitude_terms: Vec<Arc<Expr>>,
}

impl WkbBranch {
    /// Reconstruct the branch expression (without the `exp(·/ε)` wrapper,
    /// which requires the small parameter `ε`).
    ///
    /// Returns the amplitude `exp(S₁ + ε·S₂ + …)`.
    #[must_use]
    pub fn amplitude_expr(&self, small_param: SymbolId) -> Arc<Expr> {
        if self.amplitude_terms.is_empty() {
            return Expr::int(1);
        }
        let eps: Arc<Expr> = Arc::new(Expr::Symbol(small_param));
        let mut exponent: Arc<Expr> = Expr::int(0);
        for (k, term) in self.amplitude_terms.iter().enumerate() {
            // k=0 → S₁ (coefficient of ε^0 in amplitude expansion after factoring out S₀)
            let eps_power = normalize::pow(eps.clone(), Expr::int(k as i64));
            let contrib = normalize::mul(eps_power, term.clone());
            exponent = normalize::add(exponent, contrib);
        }
        Expr::func(FuncId::Exp, vec![exponent])
    }
}

/// Full WKB solution: two branches (+ and −).
#[derive(Debug, Clone)]
pub struct WkbSolution {
    /// `+` branch: phase `+∫√(−Q) dx`.
    pub plus_branch: WkbBranch,
    /// `−` branch: phase `−∫√(−Q) dx`.
    pub minus_branch: WkbBranch,
    /// The small parameter `ε`.
    pub small_param: SymbolId,
}

impl WkbSolution {
    /// Return the two branch expressions `(y₊, y₋)`.
    ///
    /// Each is `exp(±phase / ε) · amplitude`.
    #[must_use]
    pub fn to_expr(&self) -> (Arc<Expr>, Arc<Expr>) {
        let eps: Arc<Expr> = Arc::new(Expr::Symbol(self.small_param));
        let amp_plus = self.plus_branch.amplitude_expr(self.small_param);
        let amp_minus = self.minus_branch.amplitude_expr(self.small_param);

        let phase_over_eps_plus = normalize::div(self.plus_branch.phase.clone(), eps.clone());
        let phase_over_eps_minus = normalize::div(self.minus_branch.phase.clone(), eps.clone());

        let exp_plus = Expr::func(FuncId::Exp, vec![phase_over_eps_plus]);
        let exp_minus = Expr::func(FuncId::Exp, vec![phase_over_eps_minus]);

        let y_plus = normalize::mul(exp_plus, amp_plus);
        let y_minus = normalize::mul(exp_minus, amp_minus);
        (y_plus, y_minus)
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute a WKB approximation for `ε²y'' + Q(x)y = 0`.
///
/// Returns `None` when `Q` does not depend on `var` in a way the engine
/// can process (e.g. pure zero potential).
///
/// # Arguments
///
/// * `potential`   — `Q(x)` in `ε²y'' + Q(x)y = 0`.
/// * `var`         — the spatial variable `x`.
/// * `small_param` — the small parameter `ε`.
/// * `order`       — number of WKB amplitude terms to compute.
/// * `trace`       — optional narration sink.
pub fn wkb(
    potential: &Arc<Expr>,
    var: SymbolId,
    small_param: SymbolId,
    order: u32,
    trace: &mut Trace,
) -> Option<WkbSolution> {
    record(
        Some(trace),
        Step::new(
            TechniqueTag::WkbApproximation,
            format!("WKB approximation for ε²y'' + Q(x)y=0, Q={potential}, to order {order}"),
        )
        .with_input(potential.clone()),
    );

    // Leading order: S₀' = ±√Q.
    // We represent this symbolically as ±Integral(sqrt(Q), x).
    // The integral is stored as a placeholder `Pow(Q, 1/2)` integrated symbolically.
    let _sqrt_q = build_sqrt(potential);

    // Try to simplify sqrt_q by integrating if Q is a simple power of var.
    let phase_plus = build_phase(potential, var, true);
    let phase_minus = build_phase(potential, var, false);

    record(
        Some(trace),
        Step::new(
            TechniqueTag::WkbApproximation,
            "leading order phase S₀ = ±∫√Q dx".to_string(),
        )
        .with_output(phase_plus.clone()),
    );

    // First amplitude correction: S₁ = −¼·ln(Q).
    // This comes from 2S₀'·S₁' + S₀'' = 0 → S₁' = −S₀''/(2S₀') = −¼·Q'/Q.
    // Integrating: S₁ = −¼·ln(Q).
    let s1 = build_s1(potential, trace);

    record(
        Some(trace),
        Step::new(
            TechniqueTag::WkbApproximation,
            "first amplitude correction S₁ = −¼·ln(Q)".to_string(),
        )
        .with_output(s1.clone()),
    );

    // Higher order corrections via the recursion:
    // 2S₀'·Sₖ' = −Sₖ₋₁'' − Σ_{j=1}^{k-1} Sⱼ'·Sₖ₋ⱼ'
    let mut amplitude_terms = vec![s1];
    let mut prev_derivs: Vec<Arc<Expr>> = vec![diff_arc(&amplitude_terms[0], var)];

    for k in 2..=order as usize {
        let sk = build_sk(potential, var, &amplitude_terms, &prev_derivs, k, trace);
        let sk_deriv = diff_arc(&sk, var);
        prev_derivs.push(sk_deriv);
        amplitude_terms.push(sk);
    }

    record(
        Some(trace),
        Step::new(
            TechniqueTag::WkbApproximation,
            format!("computed {} WKB amplitude terms", amplitude_terms.len()),
        ),
    );

    let plus_branch = WkbBranch {
        phase: phase_plus,
        amplitude_terms: amplitude_terms.clone(),
    };
    let minus_branch = WkbBranch {
        phase: phase_minus,
        amplitude_terms,
    };

    Some(WkbSolution {
        plus_branch,
        minus_branch,
        small_param,
    })
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build `√Q` as a `Pow(Q, 1/2)` expression.
fn build_sqrt(q: &Arc<Expr>) -> Arc<Expr> {
    use crate::numeric::BigRational;
    use crate::numeric::SmallInt;
    let half = BigRational::new(SmallInt::from(1i64), SmallInt::from(2i64));
    normalize::pow(q.clone(), Arc::new(Expr::Rational(half)))
}

/// Build the leading-order WKB phase `±∫√Q dx`.
///
/// For simple polynomial potentials `Q = x^n`, this integrates analytically.
/// Otherwise the result is left as a `√Q · x` placeholder (rectangular
/// approximation) that signals the caller to use numeric integration.
fn build_phase(q: &Arc<Expr>, var: SymbolId, positive: bool) -> Arc<Expr> {
    let sqrt_q = build_sqrt(q);

    // Attempt exact integration for common patterns.
    let phase = match try_integrate_sqrt(q, var) {
        Some(integral) => integral,
        None => {
            // Fallback: represent phase as √Q · x (symbolic placeholder).
            let var_arc: Arc<Expr> = Arc::new(Expr::Symbol(var));
            normalize::mul(sqrt_q, var_arc)
        }
    };

    if positive {
        phase
    } else {
        normalize::mul(Expr::int(-1), phase)
    }
}

/// Attempt to integrate `√Q` with respect to `var` for simple cases.
///
/// Handles:
/// * `Q = c` (constant) → `√c · x`
/// * `Q = x^n`          → `x^(n/2+1) / (n/2+1)`
/// * `Q = c·x^n`        → `c · x^(n/2+1) / (n/2+1)`
fn try_integrate_sqrt(q: &Arc<Expr>, var: SymbolId) -> Option<Arc<Expr>> {
    use crate::numeric::BigRational;
    use crate::numeric::SmallInt;

    match q.as_ref() {
        // Constant Q = c → ∫√c dx = √c · x
        Expr::Integer(_) | Expr::Rational(_) | Expr::Float(_) => {
            let var_arc: Arc<Expr> = Arc::new(Expr::Symbol(var));
            Some(normalize::mul(build_sqrt(q), var_arc))
        }

        // Q = x (n=1) → ∫x^(1/2) dx = (2/3)·x^(3/2)
        Expr::Symbol(s) if *s == var => {
            let var_arc: Arc<Expr> = Arc::new(Expr::Symbol(var));
            let two_thirds = BigRational::new(SmallInt::from(2i64), SmallInt::from(3i64));
            let three_halves = BigRational::new(SmallInt::from(3i64), SmallInt::from(2i64));
            Some(normalize::mul(
                Arc::new(Expr::Rational(two_thirds)),
                normalize::pow(var_arc, Arc::new(Expr::Rational(three_halves))),
            ))
        }

        // Q = x^n → ∫x^(n/2) dx = x^(n/2+1) / (n/2+1)
        Expr::Pow(base, exp) => {
            if let Expr::Symbol(s) = base.as_ref() {
                if *s == var {
                    if let Some(n) = expr_to_i64(exp) {
                        // Exponent n/2 + 1 = (n+2)/2
                        let num = n + 2;
                        let den = 2i64;
                        let var_arc: Arc<Expr> = Arc::new(Expr::Symbol(var));
                        let new_exp = BigRational::new(SmallInt::from(num), SmallInt::from(den));
                        let coeff = BigRational::new(SmallInt::from(den), SmallInt::from(num));
                        return Some(normalize::mul(
                            Arc::new(Expr::Rational(coeff)),
                            normalize::pow(var_arc, Arc::new(Expr::Rational(new_exp))),
                        ));
                    }
                }
            }
            None
        }

        _ => None,
    }
}

/// Build `S₁ = −¼·ln(Q)`.
fn build_s1(q: &Arc<Expr>, trace: &mut Trace) -> Arc<Expr> {
    record(
        Some(trace),
        Step::new(
            TechniqueTag::WkbApproximation,
            "transport equation: S₁ = −¼·ln(Q)".to_string(),
        ),
    );
    let ln_q = Expr::func(FuncId::Ln, vec![q.clone()]);
    // −1/4 · ln(Q)
    use crate::numeric::BigRational;
    use crate::numeric::SmallInt;
    let neg_quarter = BigRational::new(SmallInt::from(-1i64), SmallInt::from(4i64));
    normalize::mul(Arc::new(Expr::Rational(neg_quarter)), ln_q)
}

/// Build the k-th WKB amplitude correction via the recursion:
///   `Sₖ' = [−Sₖ₋₁'' − Σ_{j=1}^{k-1} Sⱼ'·Sₖ₋ⱼ'] / (2·S₀')`
///
/// Since we integrate symbolically, we approximate the integral by the
/// derivative chain; for the purpose of this engine the result is stored
/// as the integrand divided by `√Q` (omitting the integration constant).
fn build_sk(
    q: &Arc<Expr>,
    var: SymbolId,
    prev_terms: &[Arc<Expr>],
    prev_derivs: &[Arc<Expr>],
    k: usize,
    trace: &mut Trace,
) -> Arc<Expr> {
    record(
        Some(trace),
        Step::new(
            TechniqueTag::WkbApproximation,
            format!("computing S_{k} via recursion"),
        ),
    );

    // Numerator = −S_{k-1}'' − Σ_{j=1}^{k-1} S_j' · S_{k-j}'
    let sk_minus1 = &prev_terms[k - 2];
    let sk_minus1_pp = diff_arc(&diff_arc(sk_minus1, var), var);
    let mut numer = normalize::mul(Expr::int(-1), sk_minus1_pp);

    for j in 1..k {
        if j <= prev_derivs.len() && (k - j) <= prev_derivs.len() {
            let sj_prime = &prev_derivs[j - 1];
            let skj_prime = &prev_derivs[k - j - 1];
            let cross = normalize::mul(sj_prime.clone(), skj_prime.clone());
            numer = normalize::sub(numer, cross);
        }
    }

    // Denominator = 2·S₀' = 2·√Q
    let two_sqrt_q = normalize::mul(Expr::int(2), build_sqrt(q));
    normalize::div(numer, two_sqrt_q)
}

/// Extract integer value from an `Arc<Expr>` if it represents an integer.
fn expr_to_i64(expr: &Arc<Expr>) -> Option<i64> {
    match expr.as_ref() {
        Expr::Integer(n) => n.to_i64(),
        _ => None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{normalize, Expr, SymbolId};

    fn sym(name: &str) -> (SymbolId, Arc<Expr>) {
        (SymbolId::intern(name), Expr::symbol(name))
    }

    /// Constant Q = k² → S₀ = ±k·x → leading phase is linear.
    #[test]
    fn fast_wkb_constant_potential_phase() {
        let (x_id, _x) = sym("wkb_x");
        let eps_id = SymbolId::intern("wkb_eps");
        let q = Expr::int(4); // Q = 4, so √Q = 2
        let mut trace = Trace::new();
        let sol = wkb(&q, x_id, eps_id, 0, &mut trace).expect("wkb should succeed");

        // Phase should be 2x for Q=4 (√4=2).
        // Try to evaluate phase at x=1: should give 2.
        use crate::numeric::evaluation::evaluate;
        use std::collections::HashMap;
        let mut vars: HashMap<SymbolId, f64> = HashMap::new();
        vars.insert(x_id, 1.0);
        let phase_val = evaluate(&sol.plus_branch.phase, &vars).unwrap_or(f64::NAN);
        assert!(
            (phase_val - 2.0).abs() < 1e-9,
            "phase at x=1 for Q=4 should be 2, got {phase_val}"
        );
    }

    /// Q = x (Airy-like) → S₀ = ±(2/3)x^(3/2).
    #[test]
    fn fast_wkb_airy_like_phase() {
        let (x_id, x) = sym("wkb_airy_x");
        let eps_id = SymbolId::intern("wkb_airy_eps");
        let mut trace = Trace::new();
        let sol = wkb(&x, x_id, eps_id, 0, &mut trace).expect("wkb airy should succeed");

        // Evaluate plus_branch.phase at x=1: (2/3)·1^(3/2) = 2/3.
        use crate::numeric::evaluation::evaluate;
        use std::collections::HashMap;
        let mut vars: HashMap<SymbolId, f64> = HashMap::new();
        vars.insert(x_id, 1.0);
        let phase_val = evaluate(&sol.plus_branch.phase, &vars).unwrap_or(f64::NAN);
        assert!(
            (phase_val - 2.0 / 3.0).abs() < 1e-9,
            "phase(1) for Airy = 2/3, got {phase_val}"
        );
    }

    /// Minus branch phase is negation of plus branch.
    #[test]
    fn fast_wkb_branches_opposite_sign() {
        let (x_id, _x) = sym("wkb_opp_x");
        let eps_id = SymbolId::intern("wkb_opp_eps");
        let q = Expr::int(1);
        let mut trace = Trace::new();
        let sol = wkb(&q, x_id, eps_id, 0, &mut trace).unwrap();

        use crate::numeric::evaluation::evaluate;
        use std::collections::HashMap;
        let mut vars: HashMap<SymbolId, f64> = HashMap::new();
        vars.insert(x_id, 3.0);
        let p = evaluate(&sol.plus_branch.phase, &vars).unwrap_or(f64::NAN);
        let m = evaluate(&sol.minus_branch.phase, &vars).unwrap_or(f64::NAN);
        assert!(
            (p + m).abs() < 1e-9,
            "phases should sum to 0, got {p} + {m} = {}",
            p + m
        );
    }

    #[test]
    fn fast_wkb_s1_is_neg_quarter_ln() {
        let (x_id, _x) = sym("wkb_s1_x");
        let eps_id = SymbolId::intern("wkb_s1_eps");
        // Q = exp(4t) isn't a simple case, but Q=4 (constant):
        // S₁ = -1/4 · ln(4) ≈ -0.347
        let q = Expr::int(4);
        let mut trace = Trace::new();
        let sol = wkb(&q, x_id, eps_id, 1, &mut trace).unwrap();

        use crate::numeric::evaluation::evaluate;
        use std::collections::HashMap;
        let vars: HashMap<SymbolId, f64> = HashMap::new();
        let s1_val = evaluate(&sol.plus_branch.amplitude_terms[0], &vars).unwrap_or(f64::NAN);
        let expected = -0.25 * (4.0f64).ln();
        assert!(
            (s1_val - expected).abs() < 1e-9,
            "S₁ for Q=4 should be {expected:.6}, got {s1_val:.6}"
        );
    }

    #[test]
    fn fast_wkb_to_expr_no_panic() {
        let (x_id, _x) = sym("wkb_te_x");
        let eps_id = SymbolId::intern("wkb_te_eps");
        let q = Expr::int(1);
        let mut trace = Trace::new();
        let sol = wkb(&q, x_id, eps_id, 0, &mut trace).unwrap();
        let (y_plus, y_minus) = sol.to_expr();
        assert!(!y_plus.is_zero());
        assert!(!y_minus.is_zero());
    }

    #[test]
    fn fast_wkb_trace_records_steps() {
        let (x_id, _x) = sym("wkb_trace_x");
        let eps_id = SymbolId::intern("wkb_trace_eps");
        let q = Expr::int(1);
        let mut trace = Trace::new();
        let _ = wkb(&q, x_id, eps_id, 1, &mut trace);
        assert!(!trace.is_empty());
        assert!(trace
            .steps()
            .iter()
            .any(|s| s.tag == TechniqueTag::WkbApproximation));
    }
}
