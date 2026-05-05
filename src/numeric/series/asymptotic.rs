//! Asymptotic expansion engine for `Arc<Expr>`.
//!
//! Handles Poincaré-style asymptotic series at `±∞` and at `0`, returning a
//! sorted sequence of dominant power-form terms. The engine walks the canonical
//! [`Expr`] shape — `Add`/`Mul`/`Pow`/`Symbol` — and extracts monomials of the
//! expansion variable.
//!
//! # Trace
//!
//! When the caller passes `Some(&mut Trace)`, the engine records a
//! [`TechniqueTag::AsymptoticExpansion`] step at each decision point. Callers
//! that only need the result pass `None` and pay no allocation cost.
//!
//! # Limitations
//!
//! The engine expects an already-expanded polynomial / Laurent-polynomial
//! shape in `var`. Expressions involving `var` wrapped in a non-power
//! transcendental context (e.g. `sin(x)`, `e^x`) are rejected with `None` —
//! series composition / Taylor-at-infinity paths belong to separate engines.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use num::traits::{One, Zero};

use super::super::{
    evaluation::evaluate,
    expr::Expr,
    limits::LimitResult,
    normalize,
    trace::{record, Step, TechniqueTag, Trace},
    BigRational, SymbolId,
};

// ── Direction ────────────────────────────────────────────────────────────────

/// Direction in which the asymptotic expansion is taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AsymptoticDirection {
    /// `x → +∞`.
    PosInfinity,
    /// `x → -∞`.
    NegInfinity,
    /// `x → 0`.
    Zero,
}

impl fmt::Display for AsymptoticDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AsymptoticDirection::PosInfinity => write!(f, "x→+∞"),
            AsymptoticDirection::NegInfinity => write!(f, "x→-∞"),
            AsymptoticDirection::Zero => write!(f, "x→0"),
        }
    }
}

// ── Term ─────────────────────────────────────────────────────────────────────

/// A single asymptotic-series term `coefficient · x^exponent`.
///
/// Both components are `Arc<Expr>` so symbolic and rational powers round-trip
/// through the engine without loss.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsymptoticTerm {
    /// Scalar multiplier in front of the power of `var`.
    pub coefficient: Arc<Expr>,
    /// Exponent attached to `var` for this term.
    pub exponent: Arc<Expr>,
}

impl AsymptoticTerm {
    /// Build a term from its coefficient and exponent.
    #[must_use]
    pub fn new(coefficient: Arc<Expr>, exponent: Arc<Expr>) -> Self {
        AsymptoticTerm {
            coefficient,
            exponent,
        }
    }

    /// True when the coefficient is a numeric zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.coefficient.is_zero()
    }
}

impl fmt::Display for AsymptoticTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.exponent.is_zero() {
            return write!(f, "{}", self.coefficient);
        }
        if self.exponent.is_one() {
            return write!(f, "{}·x", self.coefficient);
        }
        write!(f, "{}·x^({})", self.coefficient, self.exponent)
    }
}

// ── BigO ─────────────────────────────────────────────────────────────────────

/// Big-O notation describing the residual after truncation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BigO {
    /// Order expression, e.g. `x^n`.
    pub order: Arc<Expr>,
    /// Variable of the expansion.
    pub var: SymbolId,
}

impl BigO {
    /// Construct a Big-O term.
    #[must_use]
    pub fn new(order: Arc<Expr>, var: SymbolId) -> Self {
        BigO { order, var }
    }
}

impl fmt::Display for BigO {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "O({})", self.order)
    }
}

// ── Series ───────────────────────────────────────────────────────────────────

/// An asymptotic series: ordered terms sorted by dominance for `direction`.
#[derive(Debug, Clone)]
pub struct AsymptoticSeries {
    /// Dominant-first list of terms.
    pub terms: Vec<AsymptoticTerm>,
    /// Expansion variable.
    pub var: SymbolId,
    /// Expansion direction.
    pub direction: AsymptoticDirection,
}

impl AsymptoticSeries {
    /// Empty series.
    #[must_use]
    pub fn new(var: SymbolId, direction: AsymptoticDirection) -> Self {
        AsymptoticSeries {
            terms: Vec::new(),
            var,
            direction,
        }
    }

    /// Append a non-zero term. Zero terms are silently dropped.
    pub fn add_term(&mut self, term: AsymptoticTerm) {
        if !term.is_zero() {
            self.terms.push(term);
        }
    }

    /// First (most dominant) term, if any.
    #[must_use]
    pub fn dominant_term(&self) -> Option<&AsymptoticTerm> {
        self.terms.first()
    }

    /// Exponent of the dominant term.
    #[must_use]
    pub fn order_of_magnitude(&self) -> Option<Arc<Expr>> {
        self.dominant_term().map(|t| t.exponent.clone())
    }

    /// Bundle the series with the next-order error term `O(x^{e±1})`.
    #[must_use]
    pub fn with_error_term(&self) -> (Self, BigO) {
        let series = self.clone();
        let error_order = match self.terms.last() {
            Some(last) => match self.direction {
                AsymptoticDirection::PosInfinity | AsymptoticDirection::NegInfinity => {
                    normalize::sub(last.exponent.clone(), Expr::int(1))
                }
                AsymptoticDirection::Zero => normalize::add(last.exponent.clone(), Expr::int(1)),
            },
            None => Expr::int(0),
        };
        let var_expr = Arc::new(Expr::Symbol(self.var));
        let order = normalize::pow(var_expr, error_order);
        (series, BigO::new(order, self.var))
    }

    /// Reassemble the series as a single normalized `Arc<Expr>`.
    #[must_use]
    pub fn to_expr(&self) -> Arc<Expr> {
        if self.terms.is_empty() {
            return Expr::int(0);
        }
        let var_expr: Arc<Expr> = Arc::new(Expr::Symbol(self.var));
        let mut acc: Arc<Expr> = Expr::int(0);
        for term in &self.terms {
            let power = normalize::pow(var_expr.clone(), term.exponent.clone());
            let contrib = normalize::mul(term.coefficient.clone(), power);
            acc = normalize::add(acc, contrib);
        }
        acc
    }
}

impl fmt::Display for AsymptoticSeries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }
        write!(f, "As {}: ", self.direction)?;
        for (i, term) in self.terms.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            write!(f, "{term}")?;
        }
        Ok(())
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Compute an asymptotic expansion of `expr` in `var` under `direction`.
///
/// The output is an [`AsymptoticSeries`] sorted so that `terms[0]` is the
/// leading (dominant) term. `order` bounds the number of retained terms.
///
/// Returns `None` when `expr` does not reduce to a Laurent-polynomial in
/// `var` — e.g. contains `sin(x)`, `ln(x)`, or any transcendental wrapper
/// around `var`.
pub fn asymptotic(
    expr: &Arc<Expr>,
    var: SymbolId,
    direction: AsymptoticDirection,
    order: usize,
    mut trace: Option<&mut Trace>,
) -> Option<AsymptoticSeries> {
    record(
        trace.as_deref_mut(),
        Step::new(
            TechniqueTag::AsymptoticExpansion,
            format!("{direction} order {order}"),
        )
        .with_input(expr.clone()),
    );

    let mut series = AsymptoticSeries::new(var, direction);
    extract_terms(expr, var, &mut series)?;
    sort_by_dominance(&mut series.terms, direction);
    if order > 0 && series.terms.len() > order {
        series.terms.truncate(order);
    }
    Some(series)
}

/// Resolve a limit via asymptotic expansion.
///
/// Uses the dominant term's exponent and coefficient sign to classify the
/// limit as a finite value or signed infinity. Returns
/// [`LimitResult::Indeterminate`] when the expansion fails or the dominant
/// term's components are not numerically resolvable.
pub fn limit_via_asymptotic(
    expr: &Arc<Expr>,
    var: SymbolId,
    direction: AsymptoticDirection,
    mut trace: Option<&mut Trace>,
) -> LimitResult {
    let series = match asymptotic(expr, var, direction, 5, trace.as_deref_mut()) {
        Some(s) => s,
        None => return LimitResult::Indeterminate,
    };

    let Some(dominant) = series.dominant_term() else {
        return LimitResult::Value(Expr::int(0));
    };

    let exp = match to_f64(&dominant.exponent) {
        Some(e) => e,
        None => return LimitResult::Indeterminate,
    };
    let coeff_sign = to_f64(&dominant.coefficient).map(f64::signum);

    match direction {
        AsymptoticDirection::PosInfinity | AsymptoticDirection::NegInfinity => {
            if exp > 0.0 {
                match coeff_sign {
                    Some(s) if s > 0.0 => LimitResult::PosInfinity,
                    Some(s) if s < 0.0 => LimitResult::NegInfinity,
                    _ => LimitResult::Indeterminate,
                }
            } else if exp < 0.0 {
                LimitResult::Value(Expr::int(0))
            } else {
                LimitResult::Value(dominant.coefficient.clone())
            }
        }
        AsymptoticDirection::Zero => {
            if exp > 0.0 {
                LimitResult::Value(Expr::int(0))
            } else if exp < 0.0 {
                match coeff_sign {
                    Some(s) if s > 0.0 => LimitResult::PosInfinity,
                    Some(s) if s < 0.0 => LimitResult::NegInfinity,
                    _ => LimitResult::Indeterminate,
                }
            } else {
                LimitResult::Value(dominant.coefficient.clone())
            }
        }
    }
}

// ── Term extraction ──────────────────────────────────────────────────────────

fn extract_terms(expr: &Arc<Expr>, var: SymbolId, series: &mut AsymptoticSeries) -> Option<()> {
    match expr.as_ref() {
        Expr::Add(node) => {
            if !node.constant.is_zero() {
                series.add_term(AsymptoticTerm::new(
                    rational_to_arc(&node.constant),
                    Expr::int(0),
                ));
            }
            for (term, coeff) in &node.terms {
                let (mono_coeff, exponent) = extract_monomial(term, var)?;
                let full_coeff = normalize::mul(rational_to_arc(coeff), mono_coeff);
                series.add_term(AsymptoticTerm::new(full_coeff, exponent));
            }
            Some(())
        }
        _ => {
            let (coeff, exponent) = extract_monomial(expr, var)?;
            series.add_term(AsymptoticTerm::new(coeff, exponent));
            Some(())
        }
    }
}

fn extract_monomial(expr: &Arc<Expr>, var: SymbolId) -> Option<(Arc<Expr>, Arc<Expr>)> {
    match expr.as_ref() {
        Expr::Integer(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Complex(_)
        | Expr::Constant(_) => Some((expr.clone(), Expr::int(0))),

        Expr::Symbol(s) => {
            if *s == var {
                Some((Expr::int(1), Expr::int(1)))
            } else {
                Some((expr.clone(), Expr::int(0)))
            }
        }

        Expr::Pow(base, exp) => {
            if let Some(v_exp) = var_power_exponent(base, exp, var) {
                return Some((Expr::int(1), v_exp));
            }
            if !contains_symbol(expr, var) {
                Some((expr.clone(), Expr::int(0)))
            } else {
                None
            }
        }

        Expr::Mul(node) => {
            let mut total_exp: Arc<Expr> = Expr::int(0);
            let mut coeff: Arc<Expr> = rational_to_arc(&node.coeff);
            for (base, factor_exp) in &node.factors {
                if let Some(v_exp) = var_power_exponent(base, factor_exp, var) {
                    total_exp = normalize::add(total_exp, v_exp);
                    continue;
                }
                if contains_symbol(base, var) || contains_symbol(factor_exp, var) {
                    return None;
                }
                let factor = normalize::pow(base.clone(), factor_exp.clone());
                coeff = normalize::mul(coeff, factor);
            }
            Some((coeff, total_exp))
        }

        Expr::Func(_, _) | Expr::Add(_) => {
            if !contains_symbol(expr, var) {
                Some((expr.clone(), Expr::int(0)))
            } else {
                None
            }
        }
    }
}

/// If `base^factor_exp` is a power of `var`, return the effective exponent.
///
/// Canonical forms produced by `normalize::mul` store `x^k` as
/// `(base=Symbol(x), exp=k)` in the bare case and `(base=Pow(x,k), exp=1)`
/// when `x^k` lands inside a product through the `_` arm of `mul_into_node`.
/// Both shapes map to `k * factor_exp` here.
fn var_power_exponent(
    base: &Arc<Expr>,
    factor_exp: &Arc<Expr>,
    var: SymbolId,
) -> Option<Arc<Expr>> {
    match base.as_ref() {
        Expr::Symbol(s) if *s == var => Some(factor_exp.clone()),
        Expr::Pow(inner_base, inner_exp) => {
            if let Expr::Symbol(s) = inner_base.as_ref() {
                if *s == var {
                    return Some(normalize::mul(inner_exp.clone(), factor_exp.clone()));
                }
            }
            None
        }
        _ => None,
    }
}

// ── Dominance sorting ────────────────────────────────────────────────────────

/// Sort terms so the most dominant under `direction` sits at index 0.
pub(crate) fn sort_by_dominance(terms: &mut [AsymptoticTerm], direction: AsymptoticDirection) {
    terms.sort_by(|a, b| {
        let ea = to_f64(&a.exponent).unwrap_or(0.0);
        let eb = to_f64(&b.exponent).unwrap_or(0.0);
        match direction {
            AsymptoticDirection::PosInfinity | AsymptoticDirection::NegInfinity => {
                eb.partial_cmp(&ea).unwrap_or(std::cmp::Ordering::Equal)
            }
            AsymptoticDirection::Zero => ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal),
        }
    });
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn contains_symbol(expr: &Arc<Expr>, var: SymbolId) -> bool {
    match expr.as_ref() {
        Expr::Symbol(s) => *s == var,
        Expr::Integer(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Complex(_)
        | Expr::Constant(_) => false,
        Expr::Add(node) => node.terms.keys().any(|t| contains_symbol(t, var)),
        Expr::Mul(node) => node
            .factors
            .iter()
            .any(|(b, e)| contains_symbol(b, var) || contains_symbol(e, var)),
        Expr::Pow(b, e) => contains_symbol(b, var) || contains_symbol(e, var),
        Expr::Func(_, args) => args.iter().any(|a| contains_symbol(a, var)),
    }
}

fn rational_to_arc(r: &BigRational) -> Arc<Expr> {
    if r.is_zero() {
        return Expr::int(0);
    }
    if r.denom().is_one() {
        if let Some(n) = r.numer().to_i64() {
            return Expr::int(n);
        }
    }
    Arc::new(Expr::Rational(r.clone()))
}

fn to_f64(expr: &Arc<Expr>) -> Option<f64> {
    evaluate(expr, &HashMap::<SymbolId, f64>::new())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{FuncId, SymbolId};

    fn sym(name: &str) -> (SymbolId, Arc<Expr>) {
        (SymbolId::intern(name), Expr::symbol(name))
    }

    // ── extract_monomial ──

    #[test]
    fn monomial_integer() {
        let (x_id, _) = sym("asy_int");
        let (c, e) = extract_monomial(&Expr::int(5), x_id).unwrap();
        assert!(c.is_one() || matches!(c.as_ref(), Expr::Integer(n) if n.to_i64() == Some(5)));
        assert!(e.is_zero());
    }

    #[test]
    fn monomial_var() {
        let (x_id, x) = sym("asy_var");
        let (c, e) = extract_monomial(&x, x_id).unwrap();
        assert!(c.is_one());
        assert!(e.is_one());
    }

    #[test]
    fn monomial_other_var() {
        let (x_id, _) = sym("asy_otx");
        let (_, y) = sym("asy_oty");
        let (c, e) = extract_monomial(&y, x_id).unwrap();
        assert!(e.is_zero());
        assert!(matches!(c.as_ref(), Expr::Symbol(_)));
    }

    #[test]
    fn monomial_x_squared() {
        let (x_id, x) = sym("asy_xsq");
        let expr = normalize::pow(x, Expr::int(2));
        let (c, e) = extract_monomial(&expr, x_id).unwrap();
        assert!(c.is_one());
        assert!(matches!(e.as_ref(), Expr::Integer(n) if n.to_i64() == Some(2)));
    }

    #[test]
    fn monomial_x_neg_two() {
        let (x_id, x) = sym("asy_xneg");
        let expr = normalize::pow(x, Expr::int(-2));
        let (c, e) = extract_monomial(&expr, x_id).unwrap();
        assert!(c.is_one());
        assert!(matches!(e.as_ref(), Expr::Integer(n) if n.to_i64() == Some(-2)));
    }

    #[test]
    fn monomial_three_x() {
        let (x_id, x) = sym("asy_3x");
        let expr = normalize::mul(Expr::int(3), x);
        let (c, e) = extract_monomial(&expr, x_id).unwrap();
        assert!(matches!(c.as_ref(), Expr::Integer(n) if n.to_i64() == Some(3)));
        assert!(e.is_one());
    }

    #[test]
    fn monomial_rejects_sin_x() {
        let (x_id, x) = sym("asy_sinx");
        let expr = Expr::func(FuncId::Sin, vec![x]);
        assert!(extract_monomial(&expr, x_id).is_none());
    }

    // ── asymptotic ──

    #[test]
    fn asymptotic_of_x_at_inf() {
        let (x_id, x) = sym("asy_xinf");
        let series =
            asymptotic(&x, x_id, AsymptoticDirection::PosInfinity, 4, None).expect("series");
        assert_eq!(series.terms.len(), 1);
        assert!(series.terms[0].coefficient.is_one());
        assert!(series.terms[0].exponent.is_one());
    }

    #[test]
    fn asymptotic_polynomial_order_at_pos_infty() {
        // 2x^3 + x^2 + 1  →  dominant is 2·x^3
        let (x_id, x) = sym("asy_poly_pi");
        let two_x3 = normalize::mul(Expr::int(2), normalize::pow(x.clone(), Expr::int(3)));
        let x2 = normalize::pow(x.clone(), Expr::int(2));
        let expr = normalize::add(normalize::add(two_x3, x2), Expr::int(1));

        let series = asymptotic(&expr, x_id, AsymptoticDirection::PosInfinity, 5, None)
            .expect("polynomial expansion");
        assert!(series.terms.len() >= 1);
        let dom = &series.terms[0];
        assert_eq!(
            to_f64(&dom.exponent).unwrap() as i32,
            3,
            "dominant exponent at +∞"
        );
        assert_eq!(to_f64(&dom.coefficient).unwrap() as i32, 2);
    }

    #[test]
    fn asymptotic_polynomial_at_zero() {
        // 2x^3 + x^2 + 1  →  at x→0, constant 1 dominates
        let (x_id, x) = sym("asy_poly_z");
        let two_x3 = normalize::mul(Expr::int(2), normalize::pow(x.clone(), Expr::int(3)));
        let x2 = normalize::pow(x.clone(), Expr::int(2));
        let expr = normalize::add(normalize::add(two_x3, x2), Expr::int(1));

        let series =
            asymptotic(&expr, x_id, AsymptoticDirection::Zero, 5, None).expect("expansion");
        let dom = &series.terms[0];
        assert_eq!(to_f64(&dom.exponent).unwrap() as i32, 0);
        assert_eq!(to_f64(&dom.coefficient).unwrap() as i32, 1);
    }

    #[test]
    fn asymptotic_one_over_x() {
        // 1/x  →  single term with exponent -1
        let (x_id, x) = sym("asy_1overx");
        let expr = normalize::pow(x, Expr::int(-1));
        let series =
            asymptotic(&expr, x_id, AsymptoticDirection::PosInfinity, 3, None).expect("expansion");
        assert_eq!(series.terms.len(), 1);
        let dom = &series.terms[0];
        assert_eq!(to_f64(&dom.exponent).unwrap() as i32, -1);
        assert!(dom.coefficient.is_one());
    }

    #[test]
    fn asymptotic_one_plus_one_over_x_ordering() {
        // 1 + 1/x  at  +∞  →  [1 · x^0, 1 · x^-1]
        let (x_id, x) = sym("asy_1p1overx");
        let expr = normalize::add(Expr::int(1), normalize::pow(x, Expr::int(-1)));
        let series =
            asymptotic(&expr, x_id, AsymptoticDirection::PosInfinity, 3, None).expect("expansion");
        assert_eq!(series.terms.len(), 2);
        assert_eq!(to_f64(&series.terms[0].exponent).unwrap() as i32, 0);
        assert_eq!(to_f64(&series.terms[1].exponent).unwrap() as i32, -1);
    }

    #[test]
    fn asymptotic_rejects_transcendental() {
        let (x_id, x) = sym("asy_sinx_reject");
        let expr = Expr::func(FuncId::Sin, vec![x]);
        assert!(asymptotic(&expr, x_id, AsymptoticDirection::PosInfinity, 3, None).is_none());
    }

    // ── limit_via_asymptotic ──

    #[test]
    fn limit_x_to_inf() {
        // 2x + 1 → +∞
        let (x_id, x) = sym("asy_lim_pinf");
        let expr = normalize::add(normalize::mul(Expr::int(2), x), Expr::int(1));
        assert_eq!(
            limit_via_asymptotic(&expr, x_id, AsymptoticDirection::PosInfinity, None),
            LimitResult::PosInfinity
        );
    }

    #[test]
    fn limit_neg_x_to_inf() {
        // -3x → -∞
        let (x_id, x) = sym("asy_lim_ninf");
        let expr = normalize::mul(Expr::int(-3), x);
        assert_eq!(
            limit_via_asymptotic(&expr, x_id, AsymptoticDirection::PosInfinity, None),
            LimitResult::NegInfinity
        );
    }

    #[test]
    fn limit_one_over_x_to_inf() {
        // 1/x → 0
        let (x_id, x) = sym("asy_lim_1overx");
        let expr = normalize::pow(x, Expr::int(-1));
        assert_eq!(
            limit_via_asymptotic(&expr, x_id, AsymptoticDirection::PosInfinity, None),
            LimitResult::Value(Expr::int(0))
        );
    }

    #[test]
    fn limit_one_over_x_at_zero() {
        // 1/x as x→0 → +∞ (coefficient > 0)
        let (x_id, x) = sym("asy_lim_1overx_z");
        let expr = normalize::pow(x, Expr::int(-1));
        assert_eq!(
            limit_via_asymptotic(&expr, x_id, AsymptoticDirection::Zero, None),
            LimitResult::PosInfinity
        );
    }

    #[test]
    fn limit_constant_at_inf() {
        // 5 + 1/x at +∞ → 5
        let (x_id, x) = sym("asy_lim_const");
        let expr = normalize::add(Expr::int(5), normalize::pow(x, Expr::int(-1)));
        let result = limit_via_asymptotic(&expr, x_id, AsymptoticDirection::PosInfinity, None);
        match result {
            LimitResult::Value(v) => {
                assert_eq!(to_f64(&v).unwrap() as i32, 5);
            }
            other => panic!("expected Value(5), got {other:?}"),
        }
    }

    // ── trace plumbing ──

    #[test]
    fn trace_records_expansion_step() {
        let (x_id, x) = sym("asy_trace");
        let expr = normalize::pow(x, Expr::int(2));
        let mut trace = Trace::new();
        let _ = asymptotic(
            &expr,
            x_id,
            AsymptoticDirection::PosInfinity,
            3,
            Some(&mut trace),
        );
        assert!(!trace.is_empty());
        assert_eq!(trace.steps()[0].tag, TechniqueTag::AsymptoticExpansion);
    }

    #[test]
    fn trace_none_is_cheap() {
        let (x_id, x) = sym("asy_trace_none");
        let expr = normalize::pow(x, Expr::int(2));
        // Should compile and run without allocating a trace.
        let _ = asymptotic(&expr, x_id, AsymptoticDirection::PosInfinity, 3, None).expect("series");
    }

    // ── to_expr round-trip ──

    #[test]
    fn series_to_expr_roundtrip() {
        // 2x^2 + 3 → series → reassemble
        let (x_id, x) = sym("asy_roundtrip");
        let two_x2 = normalize::mul(Expr::int(2), normalize::pow(x.clone(), Expr::int(2)));
        let expr = normalize::add(two_x2, Expr::int(3));
        let series =
            asymptotic(&expr, x_id, AsymptoticDirection::PosInfinity, 5, None).expect("expansion");
        let rebuilt = series.to_expr();
        // Structural equality after canonicalization.
        assert_eq!(rebuilt, expr);
    }

    #[test]
    fn with_error_term_builds_big_o() {
        let (x_id, x) = sym("asy_err");
        let expr = normalize::pow(x, Expr::int(2));
        let series =
            asymptotic(&expr, x_id, AsymptoticDirection::PosInfinity, 3, None).expect("expansion");
        let (_, big_o) = series.with_error_term();
        // O(x^1) since last term was x^2 and direction is +∞ → e-1
        match big_o.order.as_ref() {
            Expr::Symbol(s) => assert_eq!(*s, x_id),
            Expr::Pow(_, e) => {
                assert_eq!(to_f64(e).unwrap() as i32, 1);
            }
            other => panic!("unexpected order expr: {other:?}"),
        }
    }
}
