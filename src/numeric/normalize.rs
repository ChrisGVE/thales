//! Construction-time normalization for expressions.
//!
//! Smart constructors that apply canonical form rules during expression
//! construction, eliminating the need for a separate simplification pass.
//!
//! # Rules applied
//!
//! 1. Flatten nested Add/Mul
//! 2. Sort by term order (inherent in BTreeMap)
//! 3. Combine like terms (inherent in AddNode/MulNode)
//! 4. Evaluate constant arithmetic
//! 5. Identity removal (0+x, 1*x)
//! 6. Zero annihilation (0*x)
//! 7. Normalize negative coefficients
//! 8. Unwrap single-element nodes
//! 9. GCD coefficient extraction for Add

use super::expr::Expr;
use super::{AddNode, BigRational, MulNode, SmallInt};
use num::traits::{One, Signed, Zero};
use std::sync::Arc;

// ── Addition ────────────────────────────────────────────────────────────────

/// Build a normalized sum of two expressions.
pub fn add(lhs: Arc<Expr>, rhs: Arc<Expr>) -> Arc<Expr> {
    // Rule 4: constant + constant → constant
    if let Some(result) = try_add_numeric(&lhs, &rhs) {
        return result;
    }

    // Rule 5: 0 + x → x, x + 0 → x
    if lhs.is_zero() {
        return rhs;
    }
    if rhs.is_zero() {
        return lhs;
    }

    // Build AddNode, flattening nested Adds (rule 1)
    let mut node = AddNode::zero();
    add_into_node(&mut node, &lhs, &BigRational::one());
    add_into_node(&mut node, &rhs, &BigRational::one());

    finish_add_node(node)
}

/// Build a normalized sum from multiple terms.
pub fn add_many(terms: impl IntoIterator<Item = Arc<Expr>>) -> Arc<Expr> {
    let mut node = AddNode::zero();
    let one = BigRational::one();
    for term in terms {
        add_into_node(&mut node, &term, &one);
    }
    finish_add_node(node)
}

/// Flatten an expression into an AddNode with a scaling factor.
fn add_into_node(node: &mut AddNode, expr: &Arc<Expr>, scale: &BigRational) {
    match expr.as_ref() {
        // Rule 1: flatten nested Add
        Expr::Add(inner) => {
            let scaled_const = &inner.constant * scale;
            node.add_constant(scaled_const);
            for (term, coeff) in &inner.terms {
                node.add_term(term.clone(), coeff * scale);
            }
        }
        // Rule 4: absorb constants
        Expr::Integer(n) => {
            node.add_constant(&BigRational::from_integer(n.clone()) * scale);
        }
        Expr::Rational(r) => {
            node.add_constant(r * scale);
        }
        // Decompose MulNode: coeff * single_base^1 → treat as (coeff*scale) * base
        Expr::Mul(m) if m.factor_count() == 1 => {
            let (base, exp) = m.factors.iter().next().unwrap();
            if exp.is_one() {
                let effective_coeff = &m.coeff * scale;
                node.add_term(base.clone(), effective_coeff);
            } else {
                // base^exp with non-1 exponent: treat whole Mul as opaque term
                let effective_coeff = scale.clone();
                node.add_term(expr.clone(), effective_coeff);
            }
        }
        // Mul with coeff != 1 and multiple factors: extract coeff
        Expr::Mul(m) if !m.coeff.is_one() => {
            let effective_coeff = &m.coeff * scale;
            // Rebuild a Mul with coeff=1 as the term key
            let mut unit_mul = MulNode::one();
            for (base, exp) in &m.factors {
                unit_mul.add_factor(base.clone(), exp.clone());
            }
            let key = finish_mul_node(unit_mul);
            node.add_term(key, effective_coeff);
        }
        // Everything else is a term with the given coefficient
        _ => {
            node.add_term(expr.clone(), scale.clone());
        }
    }
}

/// Finalize an AddNode into an Arc<Expr>, applying unwrap and GCD rules.
fn finish_add_node(mut node: AddNode) -> Arc<Expr> {
    // Rule 9: GCD coefficient extraction
    let gcd_factor = extract_gcd(&mut node);

    // Rule 8: unwrap single-element or constant-only
    let inner = if node.terms.is_empty() {
        rational_to_expr(node.constant)
    } else if node.constant.is_zero() && node.terms.len() == 1 {
        let (term, coeff) = node.terms.into_iter().next().unwrap();
        if coeff.is_one() {
            term
        } else {
            make_coeff_times_expr(coeff, term)
        }
    } else {
        Arc::new(Expr::Add(node))
    };

    // Wrap with GCD factor if extracted
    match gcd_factor {
        Some(g) => make_coeff_times_expr(g, inner),
        None => inner,
    }
}

// ── Subtraction ─────────────────────────────────────────────────────────────

/// Build a normalized difference of two expressions.
pub fn sub(lhs: Arc<Expr>, rhs: Arc<Expr>) -> Arc<Expr> {
    let mut node = AddNode::zero();
    let one = BigRational::one();
    let neg_one = BigRational::from(-1i64);
    add_into_node(&mut node, &lhs, &one);
    add_into_node(&mut node, &rhs, &neg_one);
    finish_add_node(node)
}

/// Negate an expression.
pub fn neg(expr: Arc<Expr>) -> Arc<Expr> {
    match expr.as_ref() {
        Expr::Integer(n) => Arc::new(Expr::Integer(-n)),
        Expr::Rational(r) => Arc::new(Expr::Rational(-r)),
        Expr::Float(f) => Arc::new(Expr::Float(-f)),
        Expr::Add(a) => Arc::new(Expr::Add(a.negate())),
        Expr::Mul(m) => {
            let mut negated = m.clone();
            negated.coeff = -&negated.coeff;
            Arc::new(Expr::Mul(negated))
        }
        _ => {
            let mut node = MulNode::from_coeff(BigRational::from(-1i64));
            node.add_factor(expr.clone(), Expr::int(1));
            finish_mul_node(node)
        }
    }
}

// ── Multiplication ──────────────────────────────────────────────────────────

/// Build a normalized product of two expressions.
pub fn mul(lhs: Arc<Expr>, rhs: Arc<Expr>) -> Arc<Expr> {
    // Rule 6: 0 * x → 0, x * 0 → 0 — but ONLY when the other operand is not
    // itself a singular form (`Pow` with a non-positive exponent over a base
    // that could vanish, i.e. `0^(-k)`, `0^0`, ...). Collapsing `0 * 0^(-k)`
    // to `0` would hide indeterminate `0/0` forms from limit analysis.
    if (lhs.is_zero() && !is_singular_pow(&rhs)) || (rhs.is_zero() && !is_singular_pow(&lhs)) {
        return Expr::int(0);
    }

    // Rule 5: 1 * x → x, x * 1 → x
    if lhs.is_one() {
        return rhs;
    }
    if rhs.is_one() {
        return lhs;
    }

    // Rule 4: constant * constant → constant
    if let Some(result) = try_mul_numeric(&lhs, &rhs) {
        return result;
    }

    // Build MulNode, flattening nested Muls (rule 1)
    let mut node = MulNode::one();
    mul_into_node(&mut node, &lhs);
    mul_into_node(&mut node, &rhs);

    finish_mul_node(node)
}

/// Flatten an expression into a MulNode.
fn mul_into_node(node: &mut MulNode, expr: &Arc<Expr>) {
    match expr.as_ref() {
        // Rule 1: flatten nested Mul
        Expr::Mul(inner) => {
            node.scale(&inner.coeff);
            for (base, exp) in &inner.factors {
                node.add_factor(base.clone(), exp.clone());
            }
        }
        // Merge `Pow(base, exp)` directly as a `(base, exp)` factor so
        // that `x · y^(-1)` canonicalizes to a MulNode with factor
        // `y^(-1)` keyed on `y`, not on the opaque `Pow` node. This lets
        // downstream decompile recognise `y^(-1)` and emit `Div`, and
        // lets `add_factor` combine like-base powers.
        Expr::Pow(base, exp) => {
            node.add_factor(base.clone(), exp.clone());
        }
        // Rule 4: absorb numeric coefficients
        Expr::Integer(n) => {
            node.scale(&BigRational::from_integer(n.clone()));
        }
        Expr::Rational(r) => {
            node.scale(r);
        }
        // Everything else is base^1
        _ => {
            node.add_factor(expr.clone(), Expr::int(1));
        }
    }
}

/// Finalize a MulNode into an Arc<Expr>, applying unwrap rules.
fn finish_mul_node(node: MulNode) -> Arc<Expr> {
    // Rule 6: zero coefficient — collapse to 0 only when no factor is a
    // singular `0^(-k)` / `0^0` form. Collapsing otherwise would hide an
    // indeterminate `0/0` from downstream limit / simplification analysis.
    if node.coeff.is_zero() {
        let has_singular = node.factors.iter().any(|(base, exp)| {
            (base.is_zero() && !is_positive_numeric(exp)) || is_singular_pow(base)
        });
        if !has_singular {
            return Expr::int(0);
        }
    }

    // Rule 8: unwrap
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

    // Rule 7: -1 * single factor → negate cleanly if possible
    // (kept as MulNode with coeff -1, which is canonical)

    Arc::new(Expr::Mul(node))
}

// ── Power ───────────────────────────────────────────────────────────────────

/// Build a normalized power expression.
pub fn pow(base: Arc<Expr>, exp: Arc<Expr>) -> Arc<Expr> {
    // Rule: x^0 → 1 (for non-zero x; 0^0 defined as 1 here)
    if exp.is_zero() {
        return Expr::int(1);
    }

    // Rule: x^1 → x
    if exp.is_one() {
        return base;
    }

    // Rule: 0^n → 0 (only when exp is definitely positive; negative or unknown
    // exponents keep the singular form so downstream analyses — limits,
    // differentiation, etc. — can see the 0^(-k) singularity instead of a
    // collapsed 0).
    if base.is_zero() {
        if is_positive_numeric(&exp) {
            return Expr::int(0);
        }
        return Arc::new(Expr::Pow(base, exp));
    }

    // Rule: 1^n → 1
    if base.is_one() {
        return Expr::int(1);
    }

    // Rule 4: integer^integer → evaluate if exp fits i64
    if let (Expr::Integer(b), Expr::Integer(e)) = (base.as_ref(), exp.as_ref()) {
        if let Some(e_val) = e.to_i64() {
            if e_val >= 0 && e_val <= 64 {
                return Arc::new(Expr::Integer(b.pow(e_val as u32)));
            }
            // Negative integer exponent → rational
            if e_val < 0 && e_val >= -64 {
                let result = b.pow((-e_val) as u32);
                if !result.is_zero() {
                    return Arc::new(Expr::Rational(BigRational::new(
                        SmallInt::from(1i64),
                        result,
                    )));
                }
            }
        }
    }

    // Rule 4b: float^integer or float^float → evaluate numerically
    let base_f = match base.as_ref() {
        Expr::Float(f) => Some(*f),
        Expr::Integer(n) => n.to_i64().map(|v| v as f64),
        Expr::Rational(r) => Some(r.to_f64()),
        _ => None,
    };
    let exp_f = match exp.as_ref() {
        Expr::Float(f) => Some(*f),
        Expr::Integer(n) => n.to_i64().map(|v| v as f64),
        Expr::Rational(r) => Some(r.to_f64()),
        _ => None,
    };
    if let (Some(b), Some(e)) = (base_f, exp_f) {
        // Only fold when at least one operand is Float (avoid shadowing exact
        // integer^integer cases handled above, and avoid promoting exact
        // integer results to lossy Float).
        let base_is_float = matches!(base.as_ref(), Expr::Float(_));
        let exp_is_float = matches!(exp.as_ref(), Expr::Float(_));
        if base_is_float || exp_is_float {
            return Arc::new(Expr::Float(b.powf(e)));
        }
    }

    // Rule: (a^b)^c → a^(b*c) when b is integer
    if let Expr::Pow(inner_base, inner_exp) = base.as_ref() {
        if is_integer_expr(inner_exp) {
            let new_exp = mul(inner_exp.clone(), exp);
            return pow(inner_base.clone(), new_exp);
        }
    }

    // Rule: (coeff * Π base_i^e_i)^n when n is integer →
    //        coeff^n * Π base_i^(e_i*n)
    if let Expr::Mul(m) = base.as_ref() {
        if is_integer_expr(&exp) {
            let mut result = MulNode::one();
            // pow the coefficient
            if let Some(coeff_powered) = try_rational_pow(&m.coeff, &exp) {
                result.coeff = coeff_powered;
            } else {
                // Can't evaluate coefficient power; don't distribute
                return Arc::new(Expr::Pow(base, exp));
            }
            for (b, e) in &m.factors {
                let new_exp = mul(e.clone(), exp.clone());
                result.add_factor(b.clone(), new_exp);
            }
            return finish_mul_node(result);
        }
    }

    Arc::new(Expr::Pow(base, exp))
}

// ── Division ────────────────────────────────────────────────────────────────

/// Build a normalized quotient `lhs / rhs`.
pub fn div(lhs: Arc<Expr>, rhs: Arc<Expr>) -> Arc<Expr> {
    // x / 1 → x
    if rhs.is_one() {
        return lhs;
    }

    if rhs.is_zero() {
        return Arc::new(Expr::Float(f64::NAN));
    }

    // 0 / x → 0 (x ≠ 0, confirmed above)
    if lhs.is_zero() {
        return Expr::int(0);
    }

    // constant / constant
    if let Some(result) = try_div_numeric(&lhs, &rhs) {
        return result;
    }

    // x / y → x * y^(-1)
    let inv = pow(rhs, Expr::int(-1));
    mul(lhs, inv)
}

// ── Numeric helpers ─────────────────────────────────────────────────────────

/// Try to add two numeric expressions.
fn try_add_numeric(a: &Expr, b: &Expr) -> Option<Arc<Expr>> {
    match (a, b) {
        (Expr::Integer(x), Expr::Integer(y)) => Some(Arc::new(Expr::Integer(x + y))),
        (Expr::Rational(x), Expr::Rational(y)) => Some(rational_to_expr(x + y)),
        (Expr::Integer(x), Expr::Rational(y)) | (Expr::Rational(y), Expr::Integer(x)) => {
            Some(rational_to_expr(&BigRational::from_integer(x.clone()) + y))
        }
        (Expr::Float(x), Expr::Float(y)) => Some(Arc::new(Expr::Float(x + y))),
        (Expr::Float(x), Expr::Integer(y)) | (Expr::Integer(y), Expr::Float(x)) => {
            if let Some(yi) = y.to_i64() {
                Some(Arc::new(Expr::Float(x + yi as f64)))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Try to multiply two numeric expressions.
fn try_mul_numeric(a: &Expr, b: &Expr) -> Option<Arc<Expr>> {
    match (a, b) {
        (Expr::Integer(x), Expr::Integer(y)) => Some(Arc::new(Expr::Integer(x * y))),
        (Expr::Rational(x), Expr::Rational(y)) => Some(rational_to_expr(x * y)),
        (Expr::Integer(x), Expr::Rational(y)) | (Expr::Rational(y), Expr::Integer(x)) => {
            Some(rational_to_expr(&BigRational::from_integer(x.clone()) * y))
        }
        (Expr::Float(x), Expr::Float(y)) => Some(Arc::new(Expr::Float(x * y))),
        (Expr::Float(x), Expr::Integer(y)) | (Expr::Integer(y), Expr::Float(x)) => {
            if let Some(yi) = y.to_i64() {
                Some(Arc::new(Expr::Float(x * yi as f64)))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Try to divide two numeric expressions.
fn try_div_numeric(a: &Expr, b: &Expr) -> Option<Arc<Expr>> {
    match (a, b) {
        (Expr::Integer(x), Expr::Integer(y)) if !y.is_zero() => {
            Some(rational_to_expr(BigRational::new(x.clone(), y.clone())))
        }
        (Expr::Rational(x), Expr::Rational(y)) if !y.is_zero() => {
            Some(rational_to_expr(x * &y.recip()))
        }
        (Expr::Integer(x), Expr::Rational(y)) if !y.is_zero() => Some(rational_to_expr(
            &BigRational::from_integer(x.clone()) * &y.recip(),
        )),
        (Expr::Rational(x), Expr::Integer(y)) if !y.is_zero() => Some(rational_to_expr(
            x * &BigRational::new(SmallInt::from(1i64), y.clone()),
        )),
        _ => None,
    }
}

/// Convert a BigRational to the simplest Expr representation.
fn rational_to_expr(r: BigRational) -> Arc<Expr> {
    if r.is_integer() {
        Arc::new(Expr::Integer(r.numer().clone()))
    } else {
        Arc::new(Expr::Rational(r))
    }
}

/// Check if an expression is a known integer.
fn is_integer_expr(e: &Expr) -> bool {
    matches!(e, Expr::Integer(_))
}

/// Check if an expression is a known strictly-positive numeric literal.
/// Used to guard the `0^n → 0` rule so that 0 raised to zero, a negative,
/// or an unknown exponent keeps its singular form rather than collapsing.
fn is_positive_numeric(e: &Expr) -> bool {
    match e {
        Expr::Integer(n) => !n.is_zero() && !n.is_negative(),
        Expr::Rational(r) => !r.is_zero() && !r.is_negative(),
        Expr::Float(f) => *f > 0.0,
        _ => false,
    }
}

/// Returns true if `expr` is a singular power form — `base^exp` where the
/// base is a literal zero and the exponent is not strictly positive. Such
/// forms encode undefined / indeterminate values (e.g. `0^(-1)` = `1/0`) and
/// must not be collapsed through `0 * …` multiplication.
fn is_singular_pow(expr: &Expr) -> bool {
    if let Expr::Pow(base, exp) = expr {
        if base.is_zero() && !is_positive_numeric(exp) {
            return true;
        }
    }
    false
}

/// Try to raise a BigRational to an integer power from an Expr.
fn try_rational_pow(r: &BigRational, exp: &Expr) -> Option<BigRational> {
    if let Expr::Integer(e) = exp {
        if let Some(ev) = e.to_i64() {
            if ev >= 0 && ev <= 64 {
                return Some(r.pow_u32(ev as u32));
            }
            if ev < 0 && ev >= -64 && !r.is_zero() {
                return Some(r.recip().pow_u32((-ev) as u32));
            }
        }
    }
    None
}

/// GCD extraction: if all coefficients (including constant, when nonzero)
/// share a common integer factor > 1, wrap the AddNode in a MulNode with
/// that factor as the coefficient.
///
/// Returns `Some(gcd)` if extraction happened, `None` otherwise.
/// The caller is responsible for wrapping the result.
fn extract_gcd(node: &mut AddNode) -> Option<BigRational> {
    if node.terms.len() < 2 {
        return None; // Need at least 2 terms for meaningful extraction
    }

    // Only extract when all coefficients are integers
    if !node.constant.is_zero() && !node.constant.denom().is_one() {
        return None;
    }
    for coeff in node.terms.values() {
        if !coeff.denom().is_one() {
            return None;
        }
    }

    // Compute GCD across all term coefficients (skip zero constant)
    let mut iter = node.terms.values();
    let first = iter.next()?;
    let mut g = first.numer().abs();
    for coeff in iter {
        g = g.gcd(&coeff.numer().abs());
        if g.is_one() {
            return None;
        }
    }
    // Include constant if nonzero
    if !node.constant.is_zero() {
        g = g.gcd(&node.constant.numer().abs());
    }

    if g.is_zero() || g.is_one() {
        return None;
    }

    // Divide all coefficients by g
    let divisor = BigRational::from_integer(g.clone());
    node.constant = &node.constant / &divisor;
    let entries: Vec<_> = node
        .terms
        .iter()
        .map(|(k, v)| (k.clone(), v / &divisor))
        .collect();
    node.terms.clear();
    for (k, v) in entries {
        node.terms.insert(k, v);
    }

    Some(BigRational::from_integer(g))
}

/// Build `coeff * expr` as a MulNode or simpler form.
fn make_coeff_times_expr(coeff: BigRational, expr: Arc<Expr>) -> Arc<Expr> {
    if coeff.is_one() {
        return expr;
    }
    if coeff.is_zero() {
        return Expr::int(0);
    }

    // If expr is already a MulNode, absorb the coefficient
    if let Expr::Mul(m) = expr.as_ref() {
        let mut node = m.clone();
        node.scale(&coeff);
        return finish_mul_node(node);
    }

    let mut node = MulNode::from_coeff(coeff);
    node.add_factor(expr, Expr::int(1));
    finish_mul_node(node)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }

    // ── Addition ────────────────────────────────────────────────────────

    #[test]
    fn test_add_constants() {
        let result = add(Expr::int(2), Expr::int(3));
        assert_eq!(*result, Expr::Integer(SmallInt::from(5i64)));
    }

    #[test]
    fn test_add_zero_identity() {
        let x = sym("norm_add_zero_x");
        let r1 = add(Expr::int(0), x.clone());
        assert!(Arc::ptr_eq(&r1, &x));
        let r2 = add(x.clone(), Expr::int(0));
        assert!(Arc::ptr_eq(&r2, &x));
    }

    #[test]
    fn test_add_like_terms() {
        // x + x → 2*x
        let x = sym("norm_alt_x");
        let result = add(x.clone(), x.clone());
        match result.as_ref() {
            Expr::Mul(m) => {
                assert_eq!(m.coeff, BigRational::from(2i64));
                assert_eq!(m.factor_count(), 1);
            }
            _ => panic!("expected MulNode, got {result}"),
        }
    }

    #[test]
    fn test_add_cancellation() {
        // x + (-x) → 0
        let x = sym("norm_cancel_x");
        let neg_x = neg(x.clone());
        let result = add(x, neg_x);
        assert!(result.is_zero());
    }

    #[test]
    fn test_add_flatten_nested() {
        // (x + y) + z → flat 3-term Add
        let x = sym("norm_flat_x");
        let y = sym("norm_flat_y");
        let z = sym("norm_flat_z");
        let xy = add(x.clone(), y.clone());
        let result = add(xy, z.clone());
        match result.as_ref() {
            Expr::Add(node) => {
                assert_eq!(node.term_count(), 3);
                assert!(node.constant.is_zero());
            }
            _ => panic!("expected AddNode, got {result}"),
        }
    }

    #[test]
    fn test_add_constant_folding_mixed() {
        // (x + 2) + 3 → x + 5
        let x = sym("norm_cf_x");
        let x_plus_2 = add(x.clone(), Expr::int(2));
        let result = add(x_plus_2, Expr::int(3));
        match result.as_ref() {
            Expr::Add(node) => {
                assert_eq!(node.constant, BigRational::from(5i64));
                assert_eq!(node.term_count(), 1);
            }
            _ => panic!("expected AddNode, got {result}"),
        }
    }

    #[test]
    fn test_add_gcd_extraction() {
        // 6x + 4y → 2*(3x + 2y)
        let x = sym("norm_gcd_x");
        let y = sym("norm_gcd_y");
        let six_x = mul(Expr::int(6), x.clone());
        let four_y = mul(Expr::int(4), y.clone());
        let result = add(six_x, four_y);
        match result.as_ref() {
            Expr::Mul(m) => {
                assert_eq!(m.coeff, BigRational::from(2i64));
                assert_eq!(m.factor_count(), 1);
                // The inner factor should be an AddNode with 3x + 2y
                let (inner_base, inner_exp) = m.factors.iter().next().unwrap();
                assert!(inner_exp.is_one());
                match inner_base.as_ref() {
                    Expr::Add(node) => {
                        assert_eq!(node.terms[&x], BigRational::from(3i64));
                        assert_eq!(node.terms[&y], BigRational::from(2i64));
                    }
                    _ => panic!("expected inner AddNode"),
                }
            }
            _ => panic!("expected MulNode wrapping AddNode, got {result}"),
        }
    }

    #[test]
    fn test_add_rationals() {
        // 1/2 + 1/3 → 5/6
        let result = add(Expr::rational(1, 2), Expr::rational(1, 3));
        assert_eq!(*result, Expr::Rational(BigRational::from_i64(5, 6)));
    }

    #[test]
    fn test_add_integer_becomes_integer() {
        // 1/2 + 1/2 → 1
        let result = add(Expr::rational(1, 2), Expr::rational(1, 2));
        assert_eq!(*result, Expr::Integer(SmallInt::from(1i64)));
    }

    // ── Subtraction ─────────────────────────────────────────────────────

    #[test]
    fn test_sub_same() {
        let x = sym("norm_sub_x");
        let result = sub(x.clone(), x.clone());
        assert!(result.is_zero());
    }

    #[test]
    fn test_sub_constants() {
        let result = sub(Expr::int(5), Expr::int(3));
        assert_eq!(*result, Expr::Integer(SmallInt::from(2i64)));
    }

    // ── Negation ────────────────────────────────────────────────────────

    #[test]
    fn test_neg_integer() {
        let result = neg(Expr::int(5));
        assert_eq!(*result, Expr::Integer(SmallInt::from(-5i64)));
    }

    #[test]
    fn test_neg_rational() {
        let result = neg(Expr::rational(1, 2));
        assert_eq!(*result, Expr::Rational(BigRational::from_i64(-1, 2)));
    }

    // ── Multiplication ──────────────────────────────────────────────────

    #[test]
    fn test_mul_constants() {
        let result = mul(Expr::int(3), Expr::int(4));
        assert_eq!(*result, Expr::Integer(SmallInt::from(12i64)));
    }

    #[test]
    fn test_mul_zero_annihilation() {
        let x = sym("norm_mza_x");
        assert!(mul(Expr::int(0), x.clone()).is_zero());
        assert!(mul(x, Expr::int(0)).is_zero());
    }

    #[test]
    fn test_mul_identity() {
        let x = sym("norm_mi_x");
        let r1 = mul(Expr::int(1), x.clone());
        assert!(Arc::ptr_eq(&r1, &x));
        let r2 = mul(x.clone(), Expr::int(1));
        assert!(Arc::ptr_eq(&r2, &x));
    }

    #[test]
    fn test_mul_same_base() {
        // x * x → x^2
        let x = sym("norm_msb_x");
        let result = mul(x.clone(), x.clone());
        match result.as_ref() {
            Expr::Pow(base, exp) => {
                assert_eq!(**base, *x);
                assert_eq!(**exp, Expr::Integer(SmallInt::from(2i64)));
            }
            _ => panic!("expected Pow, got {result}"),
        }
    }

    #[test]
    fn test_mul_flatten_nested() {
        // (2x) * (3y) → 6*x*y
        let x = sym("norm_mfn_x");
        let y = sym("norm_mfn_y");
        let two_x = mul(Expr::int(2), x.clone());
        let three_y = mul(Expr::int(3), y.clone());
        let result = mul(two_x, three_y);
        match result.as_ref() {
            Expr::Mul(m) => {
                assert_eq!(m.coeff, BigRational::from(6i64));
                assert_eq!(m.factor_count(), 2);
            }
            _ => panic!("expected MulNode, got {result}"),
        }
    }

    #[test]
    fn test_mul_coefficient_absorb() {
        // 3 * (2*x) → 6*x
        let x = sym("norm_mca_x");
        let two_x = mul(Expr::int(2), x.clone());
        let result = mul(Expr::int(3), two_x);
        match result.as_ref() {
            Expr::Mul(m) => {
                assert_eq!(m.coeff, BigRational::from(6i64));
                assert_eq!(m.factor_count(), 1);
            }
            _ => panic!("expected MulNode, got {result}"),
        }
    }

    // ── Power ───────────────────────────────────────────────────────────

    #[test]
    fn test_pow_zero_exp() {
        let x = sym("norm_pze_x");
        let result = pow(x, Expr::int(0));
        assert_eq!(*result, Expr::Integer(SmallInt::from(1i64)));
    }

    #[test]
    fn test_pow_one_exp() {
        let x = sym("norm_poe_x");
        let result = pow(x.clone(), Expr::int(1));
        assert!(Arc::ptr_eq(&result, &x));
    }

    #[test]
    fn test_pow_zero_base() {
        let result = pow(Expr::int(0), Expr::int(5));
        assert!(result.is_zero());
    }

    #[test]
    fn test_pow_one_base() {
        let result = pow(Expr::int(1), Expr::int(100));
        assert!(result.is_one());
    }

    #[test]
    fn test_pow_integer_eval() {
        let result = pow(Expr::int(2), Expr::int(10));
        assert_eq!(*result, Expr::Integer(SmallInt::from(1024i64)));
    }

    #[test]
    fn test_pow_negative_exp() {
        // 2^(-1) → 1/2
        let result = pow(Expr::int(2), Expr::int(-1));
        assert_eq!(*result, Expr::Rational(BigRational::from_i64(1, 2)));
    }

    #[test]
    fn test_pow_of_pow_integer_exp() {
        // (x^2)^3 → x^6
        let x = sym("norm_pop_x");
        let x_sq = pow(x.clone(), Expr::int(2));
        let result = pow(x_sq, Expr::int(3));
        match result.as_ref() {
            Expr::Pow(base, exp) => {
                assert_eq!(**base, *x);
                assert_eq!(**exp, Expr::Integer(SmallInt::from(6i64)));
            }
            _ => panic!("expected Pow, got {result}"),
        }
    }

    #[test]
    fn test_pow_distributes_over_mul() {
        // (2x)^3 → 8*x^3
        let x = sym("norm_pdom_x");
        let two_x = mul(Expr::int(2), x.clone());
        let result = pow(two_x, Expr::int(3));
        match result.as_ref() {
            Expr::Mul(m) => {
                assert_eq!(m.coeff, BigRational::from(8i64));
                assert_eq!(m.factor_count(), 1);
            }
            _ => panic!("expected MulNode, got {result}"),
        }
    }

    // ── Division ────────────────────────────────────────────────────────

    #[test]
    fn test_div_by_one() {
        let x = sym("norm_dbo_x");
        let result = div(x.clone(), Expr::int(1));
        assert!(Arc::ptr_eq(&result, &x));
    }

    #[test]
    fn test_div_zero_numerator() {
        let x = sym("norm_dzn_x");
        let result = div(Expr::int(0), x);
        assert!(result.is_zero());
    }

    #[test]
    fn test_div_integers() {
        // 6 / 4 → 3/2
        let result = div(Expr::int(6), Expr::int(4));
        assert_eq!(*result, Expr::Rational(BigRational::from_i64(3, 2)));
    }

    #[test]
    fn test_div_integers_exact() {
        // 6 / 3 → 2
        let result = div(Expr::int(6), Expr::int(3));
        assert_eq!(*result, Expr::Integer(SmallInt::from(2i64)));
    }

    // ── Combined ────────────────────────────────────────────────────────

    #[test]
    fn test_construction_canonical() {
        // x + (y + z) should construct as flat 3-term Add
        let x = sym("norm_cc_x");
        let y = sym("norm_cc_y");
        let z = sym("norm_cc_z");
        let yz = add(y.clone(), z.clone());
        let result = add(x.clone(), yz);
        match result.as_ref() {
            Expr::Add(node) => assert_eq!(node.term_count(), 3),
            _ => panic!("expected flat AddNode"),
        }
    }

    #[test]
    fn test_constant_arithmetic_no_simplify_call() {
        // 2 + 3 constructs as 5, not Add(2, 3)
        let result = add(Expr::int(2), Expr::int(3));
        assert_eq!(*result, Expr::Integer(SmallInt::from(5i64)));
    }

    #[test]
    fn test_identity_no_simplify_call() {
        // x + 0 constructs as x
        let x = sym("norm_insc_x");
        let result = add(x.clone(), Expr::int(0));
        assert!(Arc::ptr_eq(&result, &x));
    }

    #[test]
    fn test_add_many_empty() {
        let result = add_many(std::iter::empty());
        assert!(result.is_zero());
    }

    #[test]
    fn test_add_many_single() {
        let x = sym("norm_ams_x");
        let result = add_many(std::iter::once(x.clone()));
        assert!(Arc::ptr_eq(&result, &x));
    }

    #[test]
    fn test_add_many_multiple() {
        let x = sym("norm_amm_x");
        let y = sym("norm_amm_y");
        let z = sym("norm_amm_z");
        let result = add_many(vec![x.clone(), y.clone(), z.clone()]);
        match result.as_ref() {
            Expr::Add(node) => assert_eq!(node.term_count(), 3),
            _ => panic!("expected AddNode"),
        }
    }

    #[test]
    fn div_zero_by_zero_is_nan() {
        let zero = Expr::int(0);
        let result = div(zero.clone(), zero);
        match result.as_ref() {
            Expr::Float(f) => assert!(f.is_nan(), "0/0 should be NaN, got {}", f),
            other => panic!("0/0 should be Float(NaN), got {:?}", other),
        }
    }

    #[test]
    fn div_nonzero_by_zero_is_nan() {
        let five = Expr::int(5);
        let zero = Expr::int(0);
        let result = div(five, zero);
        match result.as_ref() {
            Expr::Float(f) => assert!(f.is_nan(), "5/0 should be NaN, got {}", f),
            other => panic!("5/0 should be Float(NaN), got {:?}", other),
        }
    }

    #[test]
    fn div_zero_by_nonzero_is_zero() {
        let zero = Expr::int(0);
        let five = Expr::int(5);
        let result = div(zero, five);
        assert!(result.is_zero(), "0/5 should be 0");
    }
}
