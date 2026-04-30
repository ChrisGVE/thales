//! Puiseux series expansion engine.
//!
//! A Puiseux series is a Laurent series with **fractional** exponents.  It
//! arises at branch points where the function has a ramification of index
//! `r` — i.e. where the local coordinate is `w = (x − c)^(1/r)`.
//!
//! # Algorithm
//!
//! 1. Detect the ramification index `r` by scanning the expression for
//!    fractional powers of the expansion variable.
//! 2. Substitute `w = (x − center)^(1/r)` so the expression becomes a
//!    Laurent polynomial in `w`.
//! 3. Compute a Taylor series of the substituted expression in `w` at 0.
//! 4. Re-express each term back in `(x − center)` with exponent `n/r`.
//!
//! For expressions with no fractional power (r = 1) the engine falls back
//! to an ordinary Taylor series.

use std::sync::Arc;

use super::super::small_int::SmallInt;
use super::super::{
    differentiation::diff_arc, expr::Expr, normalize, substitute::substitute, BigRational, SymbolId,
};
use super::TaylorSeries;
use crate::numeric::trace::{record, Step, TechniqueTag, Trace};

// ── Data types ────────────────────────────────────────────────────────────────

/// A single term of a Puiseux series: `coefficient · (x − center)^(p/q)`.
#[derive(Debug, Clone)]
pub struct PuiseuxTerm {
    /// Rational exponent `p/q` (stored as `(numerator, denominator)`).
    pub exponent_num: i64,
    pub exponent_den: u64,
    /// Coefficient expression.
    pub coefficient: Arc<Expr>,
}

/// A truncated Puiseux series `Σ cₙ · (x − center)^(numer/denom)`.
///
/// Terms are stored sorted by exponent in ascending order.
#[derive(Debug, Clone)]
pub struct PuiseuxSeries {
    /// Expansion centre.
    pub center: Arc<Expr>,
    /// Expansion variable.
    pub var: SymbolId,
    /// Terms: `(exponent_numerator, exponent_denominator, coefficient)`.
    pub terms: Vec<PuiseuxTerm>,
    /// Ramification index (denominator of all exponents).
    pub ramification: u64,
}

impl PuiseuxSeries {
    /// Reassemble the series as a single normalized `Arc<Expr>`:
    /// `Σ cₙ · (x − center)^(num/den)`.
    #[must_use]
    pub fn to_expr(&self) -> Arc<Expr> {
        let var_expr: Arc<Expr> = Arc::new(Expr::Symbol(self.var));
        let shift = if self.center.is_zero() {
            var_expr
        } else {
            normalize::sub(var_expr, self.center.clone())
        };
        let mut acc: Arc<Expr> = Expr::int(0);
        for term in &self.terms {
            if term.coefficient.is_zero() {
                continue;
            }
            let exp = rational_expr(term.exponent_num, term.exponent_den);
            let power = normalize::pow(shift.clone(), exp);
            let t = normalize::mul(term.coefficient.clone(), power);
            acc = normalize::add(acc, t);
        }
        acc
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute a Puiseux series of `expr` around `center` in variable `var`.
///
/// Returns `None` only when Taylor-coefficient extraction fails (e.g. the
/// function is not differentiable at the centre in the substituted variable).
///
/// # Arguments
///
/// * `expr`   — expression to expand.
/// * `var`    — the expansion variable.
/// * `center` — point around which to expand.
/// * `order`  — number of terms to compute.
/// * `trace`  — optional narration sink.
pub fn puiseux(
    expr: &Arc<Expr>,
    var: SymbolId,
    center: &Arc<Expr>,
    order: u32,
    trace: &mut Trace,
) -> Option<PuiseuxSeries> {
    record(
        Some(trace),
        Step::new(
            TechniqueTag::PuiseuxExpansion,
            format!(
                "Puiseux series in {} around {} to order {}",
                var, center, order
            ),
        )
        .with_input(expr.clone()),
    );

    let r = detect_ramification(expr, var);

    record(
        Some(trace),
        Step::new(
            TechniqueTag::PuiseuxExpansion,
            format!("ramification index r = {r}"),
        ),
    );

    // Substitute w = (x - center)^(1/r)  =>  x - center = w^r.
    // We introduce a fresh symbol w and replace (x-center) with w^r.
    let w_name = format!("__puiseux_w_{}", var);
    let w_id = SymbolId::intern(&w_name);
    let w_expr: Arc<Expr> = Arc::new(Expr::Symbol(w_id));

    // Build x = center + w^r, then substitute var -> center + w^r.
    let w_r = normalize::pow(w_expr.clone(), Expr::int(r as i64));
    let x_val = normalize::add(center.clone(), w_r);
    let substituted = substitute(expr, var, &x_val);

    record(
        Some(trace),
        Step::new(
            TechniqueTag::Substitution,
            format!("substituted x → center + w^{r}"),
        )
        .with_input(substituted.clone()),
    );

    // Taylor series of substituted expression in w at 0.
    let ts = taylor_in_w(&substituted, w_id, order);

    // Lift back: coefficient of w^n  corresponds to exponent n/r.
    let mut terms: Vec<PuiseuxTerm> = ts
        .coefficients
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.is_zero())
        .map(|(n, c)| PuiseuxTerm {
            exponent_num: n as i64,
            exponent_den: r,
            coefficient: c.clone(),
        })
        .collect();

    // Sort by exponent (ascending).
    terms.sort_by(|a, b| {
        let ea = a.exponent_num as f64 / a.exponent_den as f64;
        let eb = b.exponent_num as f64 / b.exponent_den as f64;
        ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
    });

    record(
        Some(trace),
        Step::new(
            TechniqueTag::PuiseuxExpansion,
            format!("computed {} Puiseux terms", terms.len()),
        ),
    );

    Some(PuiseuxSeries {
        center: center.clone(),
        var,
        terms,
        ramification: r,
    })
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Detect the smallest ramification index by scanning for fractional exponents
/// of `var` in the expression tree.
///
/// Returns `1` (ordinary Taylor) if no fractional powers are found.
fn detect_ramification(expr: &Arc<Expr>, var: SymbolId) -> u64 {
    let mut r: u64 = 1;
    collect_ramification(expr, var, &mut r);
    r
}

fn collect_ramification(expr: &Arc<Expr>, var: SymbolId, r: &mut u64) {
    use super::super::expr::FuncId;
    match expr.as_ref() {
        Expr::Pow(base, exp) => {
            if contains_var(base, var) {
                if let Some(den) = rational_denominator(exp) {
                    *r = lcm(*r, den);
                }
            }
            collect_ramification(base, var, r);
            collect_ramification(exp, var, r);
        }
        Expr::Add(node) => {
            for (term, _) in &node.terms {
                collect_ramification(term, var, r);
            }
        }
        Expr::Mul(node) => {
            for (base, exp) in &node.factors {
                collect_ramification(base, var, r);
                collect_ramification(exp, var, r);
            }
        }
        // sqrt(f(var)) introduces ramification index 2.
        Expr::Func(FuncId::Sqrt, args) => {
            for a in args {
                if contains_var(a, var) {
                    *r = lcm(*r, 2);
                }
                collect_ramification(a, var, r);
            }
        }
        // cbrt(f(var)) introduces ramification index 3.
        Expr::Func(FuncId::Cbrt, args) => {
            for a in args {
                if contains_var(a, var) {
                    *r = lcm(*r, 3);
                }
                collect_ramification(a, var, r);
            }
        }
        Expr::Func(_, args) => {
            for a in args {
                collect_ramification(a, var, r);
            }
        }
        _ => {}
    }
}

/// Return the denominator of a rational `Expr`, or `None`.
fn rational_denominator(expr: &Arc<Expr>) -> Option<u64> {
    match expr.as_ref() {
        Expr::Rational(r) => {
            let d = r.denom();
            d.to_i64().map(|x| x.unsigned_abs())
        }
        _ => None,
    }
}

/// True if `expr` contains `var` as a symbol anywhere.
fn contains_var(expr: &Arc<Expr>, var: SymbolId) -> bool {
    match expr.as_ref() {
        Expr::Symbol(s) => *s == var,
        Expr::Integer(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Complex(_)
        | Expr::Constant(_) => false,
        Expr::Add(node) => node.terms.keys().any(|t| contains_var(t, var)),
        Expr::Mul(node) => node
            .factors
            .iter()
            .any(|(b, e)| contains_var(b, var) || contains_var(e, var)),
        Expr::Pow(b, e) => contains_var(b, var) || contains_var(e, var),
        Expr::Func(_, args) => args.iter().any(|a| contains_var(a, var)),
    }
}

/// Compute GCD of two u64 values via Euclidean algorithm.
fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Compute LCM of two u64 values.
fn lcm(a: u64, b: u64) -> u64 {
    a / gcd(a, b) * b
}

/// Build the rational `Expr` for the fraction `num / den`.
fn rational_expr(num: i64, den: u64) -> Arc<Expr> {
    if den == 1 {
        return Expr::int(num);
    }
    let g = gcd(num.unsigned_abs(), den);
    let n = num / g as i64;
    let d = den / g;
    if d == 1 {
        return Expr::int(n);
    }
    let r = BigRational::new(SmallInt::from(n), SmallInt::from(d as i64));
    Arc::new(Expr::Rational(r))
}

/// Compute a Taylor series of `expr` in `w` at 0 up to `order` terms,
/// using repeated symbolic differentiation.
fn taylor_in_w(expr: &Arc<Expr>, w: SymbolId, order: u32) -> TaylorSeries {
    let zero = Expr::int(0);
    let mut coefficients = Vec::with_capacity(order as usize + 1);
    let mut current = expr.clone();

    for n in 0..=order as usize {
        let at_zero = substitute(&current, w, &zero);
        let coeff = divide_by_factorial(at_zero, n);
        coefficients.push(coeff);
        if n < order as usize {
            current = diff_arc(&current, w);
        }
    }

    TaylorSeries::from_coefficients(Expr::int(0), w, coefficients)
}

/// Divide `expr` by `n!`.
fn divide_by_factorial(expr: Arc<Expr>, n: usize) -> Arc<Expr> {
    if n <= 1 {
        return expr;
    }
    let mut acc: i64 = 1;
    for k in 2..=(n as i64) {
        acc = acc.saturating_mul(k);
    }
    normalize::div(expr, Expr::int(acc))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{normalize, FuncId};

    fn sym(name: &str) -> (SymbolId, Arc<Expr>) {
        (SymbolId::intern(name), Expr::symbol(name))
    }

    /// sqrt(1 + x) at x = 0 should give 1 + x/2 - x²/8 + ...
    #[test]
    fn fast_puiseux_sqrt_1_plus_x() {
        let (x_id, x) = sym("psx_x");
        let one_plus_x = normalize::add(Expr::int(1), x);
        let expr = Expr::func(FuncId::Sqrt, vec![one_plus_x]);
        let mut trace = Trace::new();
        let series =
            puiseux(&expr, x_id, &Expr::int(0), 3, &mut trace).expect("puiseux should succeed");

        // For sqrt(1+x), ramification = 1 (no fractional power of x directly).
        // Coefficient of x^0 = 1, coefficient of x^1 = 1/2.
        let coeff_0 = series
            .terms
            .iter()
            .find(|t| t.exponent_num == 0)
            .map(|t| t.coefficient.clone());
        assert!(
            coeff_0.map(|c| c.is_one()).unwrap_or(false),
            "a_0 of sqrt(1+x) should be 1"
        );

        // With r=2, exponent 1 = 2/2, so exponent_num=2, exponent_den=2.
        let coeff_1 = series
            .terms
            .iter()
            .find(|t| t.exponent_num as f64 / t.exponent_den as f64 - 1.0 < 1e-9
                   && (t.exponent_num as f64 / t.exponent_den as f64 - 1.0).abs() < 1e-9);
        assert!(coeff_1.is_some(), "should have x^1 term (as exponent 1.0)");
        let c1 = &coeff_1.unwrap().coefficient;
        match c1.as_ref() {
            Expr::Rational(r) => {
                let v = r.to_f64();
                assert!(
                    (v - 0.5).abs() < 1e-12,
                    "a_1 of sqrt(1+x) should be 1/2, got {v}"
                );
            }
            _ => panic!("expected rational for a_1, got {c1}"),
        }
    }

    /// sqrt(x) at x = 0 has ramification index 2.
    #[test]
    fn fast_puiseux_sqrt_x_ramification() {
        let (x_id, x) = sym("psx_rx");
        let expr = Expr::func(FuncId::Sqrt, vec![x.clone()]);
        let r = detect_ramification(&expr, x_id);
        assert_eq!(r, 2, "sqrt(x) should have ramification index 2");
    }

    /// to_expr round-trip: build a series manually and reassemble.
    #[test]
    fn fast_puiseux_to_expr_roundtrip() {
        let (x_id, _) = sym("psx_rt");
        let series = PuiseuxSeries {
            center: Expr::int(0),
            var: x_id,
            terms: vec![
                PuiseuxTerm {
                    exponent_num: 1,
                    exponent_den: 2,
                    coefficient: Expr::int(1),
                },
                PuiseuxTerm {
                    exponent_num: 3,
                    exponent_den: 2,
                    coefficient: Expr::int(1),
                },
            ],
            ramification: 2,
        };
        let expr = series.to_expr();
        // Just check it doesn't panic and returns non-zero.
        assert!(!expr.is_zero());
    }

    #[test]
    fn fast_puiseux_trace_records_steps() {
        let (x_id, x) = sym("psx_trace");
        let mut trace = Trace::new();
        let _ = puiseux(&x, x_id, &Expr::int(0), 2, &mut trace);
        assert!(!trace.is_empty());
        assert!(trace
            .steps()
            .iter()
            .any(|s| s.tag == TechniqueTag::PuiseuxExpansion));
    }
}
