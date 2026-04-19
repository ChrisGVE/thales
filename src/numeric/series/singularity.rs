//! Pole classification, residue, and singularity enumeration.
//!
//! Operates on `Arc<Expr>` directly, relying on:
//!
//! - [`numeric::limits::limit`] for probing the value of
//!   `(x − c)^k · f(x)` at `x = c`; a finite non-zero limit identifies the
//!   pole order.
//! - [`numeric::differentiation::diff_arc`] for the
//!   `residue = g^{(k−1)}(c) / (k−1)!` formula at a pole of order `k`,
//!   where `g(x) = (x − c)^k · f(x)`.
//!
//! # Trace
//!
//! Each public entry point emits one step under the relevant
//! [`TechniqueTag`] when a `&mut Trace` is supplied. Callers that only need
//! the result pass `None` and pay no allocation cost.
//!
//! # Scope
//!
//! - Handles isolated finite poles reachable through limit + derivative.
//! - Does not attempt to classify branch points, essential singularities of
//!   transcendental functions, or non-isolated singularities — those return
//!   [`SingularityType::Essential`] as a catch-all when pole detection fails
//!   while the function is not regular at the point.
//! - `find_singularities` walks rational-function denominators for obvious
//!   zeros. Non-rational inputs return an empty vector.

use std::fmt;
use std::sync::Arc;

use num::traits::{One, Zero};

use super::super::{
    differentiation::diff_arc,
    expr::Expr,
    limits::{limit, LimitPoint, LimitResult},
    normalize,
    trace::{record, Step, TechniqueTag, Trace},
    BigRational, MulNode, SymbolId,
};
use super::taylor::substitute;

/// Upper bound on the pole order the engine attempts to detect via the
/// `(x − c)^k · f(x)` limit sweep. Larger orders are classified as essential.
pub const MAX_POLE_ORDER: u32 = 10;

// ── Types ────────────────────────────────────────────────────────────────────

/// Category of an isolated singularity at a point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SingularityType {
    /// The limit exists and is finite; the singularity is removable.
    Removable,
    /// A pole of the given positive order.
    Pole(u32),
    /// Essential or unresolved singularity (no finite pole order detected).
    Essential,
}

impl fmt::Display for SingularityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SingularityType::Removable => write!(f, "removable singularity"),
            SingularityType::Pole(k) => write!(f, "pole of order {k}"),
            SingularityType::Essential => write!(f, "essential singularity"),
        }
    }
}

/// An isolated singularity and its type.
#[derive(Debug, Clone)]
pub struct Singularity {
    /// Location of the singularity.
    pub location: Arc<Expr>,
    /// Classification.
    pub kind: SingularityType,
}

impl Singularity {
    /// Construct.
    #[must_use]
    pub fn new(location: Arc<Expr>, kind: SingularityType) -> Self {
        Singularity { location, kind }
    }

    /// True when the singularity is a pole.
    #[must_use]
    pub fn is_pole(&self) -> bool {
        matches!(self.kind, SingularityType::Pole(_))
    }

    /// Pole order if applicable.
    #[must_use]
    pub fn pole_order(&self) -> Option<u32> {
        match self.kind {
            SingularityType::Pole(k) => Some(k),
            _ => None,
        }
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Detect the pole order of `expr` at `point`.
///
/// Returns `0` when the function is regular (or has a removable singularity)
/// at the point, a positive integer for detected pole orders, and `None`
/// when the behaviour exceeds [`MAX_POLE_ORDER`] without resolving.
pub fn pole_order(
    expr: &Arc<Expr>,
    var: SymbolId,
    point: &Arc<Expr>,
    mut trace: Option<&mut Trace>,
) -> Option<u32> {
    record(
        trace.as_deref_mut(),
        Step::new(
            TechniqueTag::PoleClassification,
            format!("probe at {point}"),
        )
        .with_input(expr.clone()),
    );

    // Regular point: direct limit is finite.
    if let LimitResult::Value(_) = limit(expr, var, &LimitPoint::Value(point.clone())) {
        return Some(0);
    }

    for k in 1..=MAX_POLE_ORDER {
        let factor = normalize::pow(shift_expr(var, point), Expr::int(k as i64));
        let probe = normalize::mul(factor, expr.clone());
        let probe_limit = limit(&probe, var, &LimitPoint::Value(point.clone()));
        if let LimitResult::Value(v) = probe_limit {
            if !v.is_zero() {
                return Some(k);
            }
            // Zero limit means higher order of cancellation — keep searching,
            // but treat regular behaviour after a zero match as a fixed point.
        }
    }
    None
}

/// Classify the singularity of `expr` at `point`.
pub fn classify_singularity(
    expr: &Arc<Expr>,
    var: SymbolId,
    point: &Arc<Expr>,
    trace: Option<&mut Trace>,
) -> SingularityType {
    match pole_order(expr, var, point, trace) {
        Some(0) => SingularityType::Removable,
        Some(k) => SingularityType::Pole(k),
        None => SingularityType::Essential,
    }
}

/// Compute the residue of `expr` at `pole`.
///
/// Uses `res = g^{(k−1)}(c) / (k−1)!` where `g(x) = (x − c)^k · f(x)` and
/// `k = pole_order(f, c)`. Returns `None` for essential / unresolved
/// singularities, and `Some(0)` for regular points.
pub fn residue(
    expr: &Arc<Expr>,
    var: SymbolId,
    pole: &Arc<Expr>,
    mut trace: Option<&mut Trace>,
) -> Option<Arc<Expr>> {
    let order = pole_order(expr, var, pole, trace.as_deref_mut())?;
    if order == 0 {
        return Some(Expr::int(0));
    }

    record(
        trace.as_deref_mut(),
        Step::new(
            TechniqueTag::ResidueTheorem,
            format!("pole order {order} at {pole}"),
        ),
    );

    let factor = normalize::pow(shift_expr(var, pole), Expr::int(order as i64));
    let mut g = mul_expanding_pow(factor, expr.clone());

    for _ in 1..order {
        g = diff_arc(&g, var);
    }

    let at_pole = substitute(&g, var, pole);
    let factorial = factorial_i64(order.saturating_sub(1));
    Some(normalize::div(at_pole, Expr::int(factorial)))
}

/// Enumerate obvious singularities of a rational-looking expression.
///
/// Walks the expression for `(x − c)^{-k}` factors and records each
/// distinct center `c` as a pole of the summed order. Non-rational shapes
/// return an empty vector.
pub fn find_singularities(expr: &Arc<Expr>, var: SymbolId) -> Vec<Singularity> {
    let mut poles: Vec<(Arc<Expr>, u32)> = Vec::new();
    collect_pole_factors(expr, var, &mut poles);
    poles
        .into_iter()
        .map(|(loc, k)| Singularity::new(loc, SingularityType::Pole(k)))
        .collect()
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn shift_expr(var: SymbolId, center: &Arc<Expr>) -> Arc<Expr> {
    if center.is_zero() {
        Arc::new(Expr::Symbol(var))
    } else {
        normalize::sub(Arc::new(Expr::Symbol(var)), center.clone())
    }
}

/// Multiply `a` and `b`, expanding top-level `Pow(base, exp)` factors through
/// `MulNode::add_factor` so that `x^k · x^{-k}` cancels via exponent addition.
///
/// `normalize::mul` treats `Pow` as an opaque factor and stores it at
/// exponent 1, which prevents cancellation between a bare `x` and a
/// `Pow(x, -1)` sibling. The residue formula depends on that cancellation,
/// so this helper rebuilds the product with explicit `add_factor` calls.
fn mul_expanding_pow(a: Arc<Expr>, b: Arc<Expr>) -> Arc<Expr> {
    let mut node = MulNode::one();
    absorb_into_mul(&mut node, &a);
    absorb_into_mul(&mut node, &b);
    finish_mul(node)
}

fn absorb_into_mul(node: &mut MulNode, expr: &Arc<Expr>) {
    match expr.as_ref() {
        Expr::Integer(n) => {
            let r = BigRational::from_integer(n.clone());
            let new_coeff = &node.coeff * &r;
            node.coeff = new_coeff;
        }
        Expr::Rational(r) => {
            let new_coeff = &node.coeff * r;
            node.coeff = new_coeff;
        }
        Expr::Mul(inner) => {
            let scaled = &node.coeff * &inner.coeff;
            node.coeff = scaled;
            for (base, exp) in &inner.factors {
                absorb_pair(node, base.clone(), exp.clone());
            }
        }
        Expr::Pow(base, exp) => {
            absorb_pair(node, base.clone(), exp.clone());
        }
        _ => {
            node.add_factor(expr.clone(), Expr::int(1));
        }
    }
}

fn absorb_pair(node: &mut MulNode, base: Arc<Expr>, exp: Arc<Expr>) {
    // Recursively unfold nested Pow on the base: Pow(Pow(x, a), b) → x^(a*b).
    match base.as_ref() {
        Expr::Pow(inner_base, inner_exp) => {
            let new_exp = normalize::mul(inner_exp.clone(), exp);
            absorb_pair(node, inner_base.clone(), new_exp);
        }
        _ => node.add_factor(base, exp),
    }
}

fn finish_mul(node: MulNode) -> Arc<Expr> {
    if node.coeff.is_zero() {
        return Expr::int(0);
    }
    if node.factors.is_empty() {
        return rational_to_expr(node.coeff);
    }
    if node.coeff.is_one() && node.factors.len() == 1 {
        let (base, exp) = node.factors.into_iter().next().unwrap();
        if exp.is_one() {
            return base;
        }
        return Arc::new(Expr::Pow(base, exp));
    }
    Arc::new(Expr::Mul(node))
}

fn rational_to_expr(r: BigRational) -> Arc<Expr> {
    if r.is_zero() {
        return Expr::int(0);
    }
    if r.denom().is_one() {
        if let Some(n) = r.numer().to_i64() {
            return Expr::int(n);
        }
    }
    Arc::new(Expr::Rational(r))
}

fn factorial_i64(n: u32) -> i64 {
    let mut acc: i64 = 1;
    for k in 2..=(n as i64) {
        acc = acc.saturating_mul(k);
    }
    acc
}

/// Collect `(x − c)^{-k}` factors observable via structural inspection.
///
/// Handles the canonical `Mul` case (where a `Pow(.., -k)` factor appears)
/// and the bare `Pow(.., -k)` case. Arbitrary `Add` denominators are beyond
/// this structural walk; use [`pole_order`] directly for those.
fn collect_pole_factors(expr: &Arc<Expr>, var: SymbolId, out: &mut Vec<(Arc<Expr>, u32)>) {
    match expr.as_ref() {
        Expr::Pow(base, exp) => {
            if let Some(k) = neg_integer(exp) {
                if let Some(center) = center_of_shift(base, var) {
                    record_pole(out, center, k);
                }
            }
        }
        Expr::Mul(node) => {
            for (base, factor_exp) in &node.factors {
                if let Some(k) = neg_integer(factor_exp) {
                    if let Some(center) = center_of_shift(base, var) {
                        record_pole(out, center, k);
                    }
                } else if let Expr::Pow(inner_base, inner_exp) = base.as_ref() {
                    if let Some(k) = neg_integer(inner_exp) {
                        if let Some(center) = center_of_shift(inner_base, var) {
                            record_pole(out, center, k);
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

fn record_pole(out: &mut Vec<(Arc<Expr>, u32)>, center: Arc<Expr>, k: u32) {
    if let Some(entry) = out.iter_mut().find(|(c, _)| *c == center) {
        entry.1 += k;
    } else {
        out.push((center, k));
    }
}

/// If `exp` is a negative integer, return its absolute value.
fn neg_integer(exp: &Arc<Expr>) -> Option<u32> {
    match exp.as_ref() {
        Expr::Integer(n) => n
            .to_i64()
            .and_then(|v| if v < 0 { u32::try_from(-v).ok() } else { None }),
        _ => None,
    }
}

/// If `base` is `x` or `x - c`, return the center `c` (defaulting to 0).
fn center_of_shift(base: &Arc<Expr>, var: SymbolId) -> Option<Arc<Expr>> {
    match base.as_ref() {
        Expr::Symbol(s) if *s == var => Some(Expr::int(0)),
        Expr::Add(node) => {
            // (x - c) normalizes to Add{constant = -c, terms = {x: 1}}.
            if node.terms.len() != 1 {
                return None;
            }
            let (term, coeff) = node.terms.iter().next()?;
            if !coeff.is_one() {
                return None;
            }
            if let Expr::Symbol(s) = term.as_ref() {
                if *s == var {
                    // center = -node.constant
                    let neg = -&node.constant;
                    return Some(Arc::new(Expr::Rational(neg)));
                }
            }
            None
        }
        _ => None,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(name: &str) -> (SymbolId, Arc<Expr>) {
        (SymbolId::intern(name), Expr::symbol(name))
    }

    // ── pole_order ──

    #[test]
    fn regular_point_has_order_zero() {
        // f(x) = x  at  x = 2 is regular.
        let (x_id, x) = sym("sing_reg");
        assert_eq!(pole_order(&x, x_id, &Expr::int(2), None), Some(0));
    }

    #[test]
    fn simple_pole_at_zero() {
        // 1/x  at  x = 0 is a simple pole.
        let (x_id, x) = sym("sing_simple");
        let f = normalize::pow(x, Expr::int(-1));
        assert_eq!(pole_order(&f, x_id, &Expr::int(0), None), Some(1));
    }

    #[test]
    fn double_pole_at_zero() {
        // 1/x^2 → order 2.
        let (x_id, x) = sym("sing_double");
        let f = normalize::pow(x, Expr::int(-2));
        assert_eq!(pole_order(&f, x_id, &Expr::int(0), None), Some(2));
    }

    #[test]
    fn pole_at_nonzero_center() {
        // 1/(x - 3)  at  x = 3.
        let (x_id, x) = sym("sing_shift");
        let shifted = normalize::sub(x, Expr::int(3));
        let f = normalize::pow(shifted, Expr::int(-1));
        assert_eq!(pole_order(&f, x_id, &Expr::int(3), None), Some(1));
    }

    // ── residue ──

    #[test]
    fn residue_simple_pole_one_over_x() {
        // Res(1/x; 0) = 1.
        let (x_id, x) = sym("res_1overx");
        let f = normalize::pow(x, Expr::int(-1));
        let r = residue(&f, x_id, &Expr::int(0), None).expect("residue");
        assert!(r.is_one(), "residue = {r}");
    }

    #[test]
    fn residue_double_pole() {
        // Res(1/x^2; 0) = 0  (no x^{-1} term).
        let (x_id, x) = sym("res_1overx2");
        let f = normalize::pow(x, Expr::int(-2));
        let r = residue(&f, x_id, &Expr::int(0), None).expect("residue");
        assert!(r.is_zero(), "residue = {r}");
    }

    #[test]
    fn residue_one_over_shifted() {
        // Res(1/(x - 2); 2) = 1.
        let (x_id, x) = sym("res_shift");
        let shifted = normalize::sub(x, Expr::int(2));
        let f = normalize::pow(shifted, Expr::int(-1));
        let r = residue(&f, x_id, &Expr::int(2), None).expect("residue");
        assert!(r.is_one(), "residue = {r}");
    }

    #[test]
    fn residue_of_k_over_x() {
        // Res(5/x; 0) = 5.
        let (x_id, x) = sym("res_5overx");
        let f = normalize::mul(Expr::int(5), normalize::pow(x, Expr::int(-1)));
        let r = residue(&f, x_id, &Expr::int(0), None).expect("residue");
        // Evaluate numerically.
        match r.as_ref() {
            Expr::Integer(n) => assert_eq!(n.to_i64(), Some(5)),
            other => panic!("expected 5, got {other}"),
        }
    }

    // ── classify ──

    #[test]
    fn classify_regular() {
        let (x_id, x) = sym("cls_reg");
        assert_eq!(
            classify_singularity(&x, x_id, &Expr::int(1), None),
            SingularityType::Removable
        );
    }

    #[test]
    fn classify_pole() {
        let (x_id, x) = sym("cls_pole");
        let f = normalize::pow(x, Expr::int(-3));
        assert_eq!(
            classify_singularity(&f, x_id, &Expr::int(0), None),
            SingularityType::Pole(3)
        );
    }

    // ── find_singularities ──

    #[test]
    fn find_singularities_one_over_x() {
        let (x_id, x) = sym("find_1overx");
        let f = normalize::pow(x, Expr::int(-1));
        let sings = find_singularities(&f, x_id);
        assert_eq!(sings.len(), 1);
        assert!(sings[0].is_pole());
        assert_eq!(sings[0].pole_order(), Some(1));
        assert!(sings[0].location.is_zero());
    }

    #[test]
    fn find_singularities_shift() {
        // 1/(x - 4)
        let (x_id, x) = sym("find_shift");
        let f = normalize::pow(normalize::sub(x, Expr::int(4)), Expr::int(-1));
        let sings = find_singularities(&f, x_id);
        assert_eq!(sings.len(), 1);
        let loc = &sings[0].location;
        match loc.as_ref() {
            Expr::Integer(n) => assert_eq!(n.to_i64(), Some(4)),
            Expr::Rational(r) => assert!((r.to_f64() - 4.0).abs() < 1e-15),
            other => panic!("expected 4, got {other}"),
        }
        assert_eq!(sings[0].pole_order(), Some(1));
    }

    #[test]
    fn find_singularities_non_rational_empty() {
        let (x_id, x) = sym("find_nonrat");
        let sings = find_singularities(&x, x_id);
        assert!(sings.is_empty());
    }

    // ── Trace plumbing ──

    #[test]
    fn pole_order_records_trace() {
        let (x_id, x) = sym("trace_pole");
        let f = normalize::pow(x, Expr::int(-1));
        let mut trace = Trace::new();
        let _ = pole_order(&f, x_id, &Expr::int(0), Some(&mut trace));
        assert_eq!(trace.steps()[0].tag, TechniqueTag::PoleClassification);
    }

    #[test]
    fn residue_records_trace() {
        let (x_id, x) = sym("trace_res");
        let f = normalize::pow(x, Expr::int(-1));
        let mut trace = Trace::new();
        let _ = residue(&f, x_id, &Expr::int(0), Some(&mut trace));
        let tags: Vec<_> = trace.steps().iter().map(|s| s.tag).collect();
        assert!(tags.contains(&TechniqueTag::PoleClassification));
        assert!(tags.contains(&TechniqueTag::ResidueTheorem));
    }
}
