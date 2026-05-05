//! Trigonometric simplification strategies.
//!
//! Provides rewrite rules and composed strategies for working with trig
//! expressions in the numeric AST. Rules are structured as:
//!
//! - **Constant evaluation**: `sin(0) → 0`, `cos(0) → 1`
//! - **Pythagorean identity**: `sin²(x) + cos²(x) → 1`
//! - **Double-angle expansion**: `sin(2x) → 2·sin(x)·cos(x)`,
//!   `cos(2x) → cos²(x) - sin²(x)`
//! - **Double-angle contraction**: `2·sin(x)·cos(x) → sin(2x)`
//! - **Tangent expansion**: `tan(x) → sin(x)/cos(x)`
//!
//! # Strategies
//! - [`trig_simplify`] — Pythagorean identity + constant folding
//! - [`trig_expand`] — expand double angles into products/sums
//! - [`trig_contract`] — contract products of sin·cos into single functions
use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::normalize;
use crate::numeric::rewrite::{choice, innermost, try_rule, Strategy};
use num::traits::{One, Zero};
use std::sync::Arc;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Return the single argument of a unary `Func` node, or `None`.
fn unary_arg(id: FuncId, e: &Arc<Expr>) -> Option<Arc<Expr>> {
    match e.as_ref() {
        Expr::Func(fid, args) if *fid == id && args.len() == 1 => Some(args[0].clone()),
        _ => None,
    }
}

/// Build `sin(x)` as `Arc<Expr>`.
fn sin(x: Arc<Expr>) -> Arc<Expr> {
    Expr::func(FuncId::Sin, vec![x])
}

/// Build `cos(x)` as `Arc<Expr>`.
fn cos(x: Arc<Expr>) -> Arc<Expr> {
    Expr::func(FuncId::Cos, vec![x])
}

// ── Constant evaluation rules ─────────────────────────────────────────────────

/// `sin(0) → 0`
pub fn rule_sin_zero() -> Strategy {
    try_rule(|e| {
        let arg = unary_arg(FuncId::Sin, e)?;
        if arg.is_zero() {
            Some(Expr::int(0))
        } else {
            None
        }
    })
}

/// `cos(0) → 1`
pub fn rule_cos_zero() -> Strategy {
    try_rule(|e| {
        let arg = unary_arg(FuncId::Cos, e)?;
        if arg.is_zero() {
            Some(Expr::int(1))
        } else {
            None
        }
    })
}

// ── Pythagorean identity ──────────────────────────────────────────────────────

/// `sin²(x) + cos²(x) → 1`
///
/// Matches an `AddNode` that contains both a `sin(x)^2` term with coefficient 1
/// and a `cos(x)^2` term with coefficient 1 for the same argument `x`. Any
/// remaining terms are preserved.
pub fn rule_sin_sq_plus_cos_sq() -> Strategy {
    use num::traits::One as _;
    try_rule(|e| {
        let node = match e.as_ref() {
            Expr::Add(n) => n,
            _ => return None,
        };

        // Find sin²(x) and cos²(x) terms, both with coefficient 1
        let (sin_arg, sin_coeff) = find_trig_sq_direct(node, FuncId::Sin)?;
        let (cos_arg, cos_coeff) = find_trig_sq_direct(node, FuncId::Cos)?;

        if sin_arg != cos_arg {
            return None;
        }
        if !sin_coeff.is_one() || !cos_coeff.is_one() {
            return None;
        }

        let mut terms: Vec<Arc<Expr>> = vec![Expr::int(1)];
        if !node.constant.is_zero() {
            terms.push(Arc::new(Expr::Rational(node.constant.clone())));
        }
        for (term, coeff) in &node.terms {
            let skip =
                is_trig_sq(term, FuncId::Sin, sin_arg) || is_trig_sq(term, FuncId::Cos, cos_arg);
            if !skip {
                let scaled = normalize::mul(Arc::new(Expr::Rational(coeff.clone())), term.clone());
                terms.push(scaled);
            }
        }
        Some(normalize::add_many(terms))
    })
}

/// Find a `func(x)^2` term in an `AddNode` where the result is exactly
/// `(arg_ref, coeff)`. Uses direct `Pow` matching.
fn find_trig_sq_direct<'a>(
    node: &'a crate::numeric::AddNode,
    id: FuncId,
) -> Option<(&'a Arc<Expr>, &'a crate::numeric::BigRational)> {
    use crate::numeric::SmallInt;
    for (term, coeff) in &node.terms {
        if let Expr::Pow(base, exp) = term.as_ref() {
            if matches!(exp.as_ref(), Expr::Integer(n) if *n == SmallInt::from(2i64)) {
                if let Expr::Func(fid, args) = base.as_ref() {
                    if *fid == id && args.len() == 1 {
                        return Some((&args[0], coeff));
                    }
                }
            }
        }
    }
    None
}

/// Returns `true` if `term` is `id(expected_arg)^2`.
fn is_trig_sq(term: &Arc<Expr>, id: FuncId, expected_arg: &Arc<Expr>) -> bool {
    use crate::numeric::SmallInt;
    if let Expr::Pow(base, exp) = term.as_ref() {
        if matches!(exp.as_ref(), Expr::Integer(n) if *n == SmallInt::from(2i64)) {
            if let Expr::Func(fid, args) = base.as_ref() {
                return *fid == id && args.len() == 1 && args[0] == *expected_arg;
            }
        }
    }
    false
}

// ── Double-angle expansion rules ──────────────────────────────────────────────

/// `sin(2x) → 2·sin(x)·cos(x)`
///
/// Matches `sin(2·x)` where `2·x` is a `Mul` node with integer coefficient 2
/// and a single symbolic factor, or `2·arg` represented as such.
pub fn rule_double_angle_sin() -> Strategy {
    try_rule(|e| {
        let arg = unary_arg(FuncId::Sin, e)?;
        let (coeff, inner) = extract_int_coeff_2(&arg)?;
        if coeff != 2 {
            return None;
        }
        // sin(2x) → 2·sin(x)·cos(x)
        let two = Expr::int(2);
        Some(normalize::mul(
            two,
            normalize::mul(sin(inner.clone()), cos(inner)),
        ))
    })
}

/// `cos(2x) → cos²(x) - sin²(x)`
pub fn rule_double_angle_cos() -> Strategy {
    try_rule(|e| {
        let arg = unary_arg(FuncId::Cos, e)?;
        let (coeff, inner) = extract_int_coeff_2(&arg)?;
        if coeff != 2 {
            return None;
        }
        // cos(2x) → cos²(x) - sin²(x)
        let two = Expr::int(2);
        let cos_sq = normalize::pow(cos(inner.clone()), two.clone());
        let sin_sq = normalize::pow(sin(inner), two);
        Some(normalize::sub(cos_sq, sin_sq))
    })
}

/// Extract an integer coefficient from `n·x` (Mul with single factor, integer
/// coeff). Returns `(n, x)` when `n` is exactly representable as i64.
fn extract_int_coeff_2(expr: &Arc<Expr>) -> Option<(i64, Arc<Expr>)> {
    match expr.as_ref() {
        Expr::Mul(m) if m.factor_count() == 1 => {
            let n = m.coeff.numer();
            let d = m.coeff.denom();
            if !d.is_one() {
                return None;
            }
            let iv = n.to_i64()?;
            let (base, exp) = m.factors.iter().next()?;
            if !exp.is_one() {
                return None;
            }
            Some((iv, base.clone()))
        }
        _ => None,
    }
}

// ── Contraction rule ──────────────────────────────────────────────────────────

/// `2·sin(x)·cos(x) → sin(2x)`
///
/// Matches a `MulNode` with integer coefficient 2 and exactly two factors
/// `sin(x)^1` and `cos(x)^1` for the same argument `x`.
pub fn rule_contract_sin_cos() -> Strategy {
    try_rule(|e| {
        let m = match e.as_ref() {
            Expr::Mul(m) => m,
            _ => return None,
        };

        // Coefficient must be exactly 2
        let n = m.coeff.numer();
        let d = m.coeff.denom();
        if !d.is_one() || n.to_i64()? != 2 {
            return None;
        }

        // Exactly two factors, both with exponent 1
        if m.factor_count() != 2 {
            return None;
        }

        let factors: Vec<_> = m.factors.iter().collect();
        let (b0, e0) = factors[0];
        let (b1, e1) = factors[1];
        if !e0.is_one() || !e1.is_one() {
            return None;
        }

        // Identify which is sin and which is cos
        let (sin_base, cos_base) = match (b0.as_ref(), b1.as_ref()) {
            (Expr::Func(FuncId::Sin, sa), Expr::Func(FuncId::Cos, ca))
                if sa.len() == 1 && ca.len() == 1 =>
            {
                (&sa[0], &ca[0])
            }
            (Expr::Func(FuncId::Cos, ca), Expr::Func(FuncId::Sin, sa))
                if sa.len() == 1 && ca.len() == 1 =>
            {
                (&sa[0], &ca[0])
            }
            _ => return None,
        };

        // Arguments must match
        if sin_base != cos_base {
            return None;
        }

        // sin(2x)
        let two_x = normalize::mul(Expr::int(2), sin_base.clone());
        Some(sin(two_x))
    })
}

// ── Tan expansion rule ────────────────────────────────────────────────────────

/// `tan(x) → sin(x) / cos(x)`
pub fn rule_tan_to_sin_cos() -> Strategy {
    try_rule(|e| {
        let arg = unary_arg(FuncId::Tan, e)?;
        Some(normalize::div(sin(arg.clone()), cos(arg)))
    })
}

// ── Composed strategies ───────────────────────────────────────────────────────

/// Expand trig functions: double-angle identities and tan definition.
///
/// Applies `sin(2x) → 2·sin(x)·cos(x)`, `cos(2x) → cos²(x) - sin²(x)`,
/// and `tan(x) → sin(x)/cos(x)` bottom-up until fixpoint.
pub fn trig_expand() -> Strategy {
    let rule = choice(
        rule_double_angle_sin(),
        choice(rule_double_angle_cos(), rule_tan_to_sin_cos()),
    );
    innermost(rule, 32)
}

fn rule_numeric_fold() -> Strategy {
    try_rule(|e| {
        let node = match e.as_ref() {
            Expr::Add(n) => n,
            _ => return None,
        };
        let mut terms: Vec<Arc<Expr>> = Vec::new();
        if !node.constant.is_zero() {
            terms.push(Arc::new(Expr::Rational(node.constant.clone())));
        }
        for (term, coeff) in &node.terms {
            terms.push(normalize::mul(
                Arc::new(Expr::Rational(coeff.clone())),
                term.clone(),
            ));
        }
        Some(normalize::add_many(terms))
    })
}

/// Simplify trig expressions: Pythagorean identity and constant folding.
///
/// Applies `sin²(x) + cos²(x) → 1`, `sin(0) → 0`, and `cos(0) → 1`
/// bottom-up until fixpoint.
pub fn trig_simplify() -> Strategy {
    let rule = choice(
        rule_sin_sq_plus_cos_sq(),
        choice(
            rule_sin_zero(),
            choice(rule_cos_zero(), rule_numeric_fold()),
        ),
    );
    innermost(rule, 32)
}

/// Contract trig products: `2·sin(x)·cos(x) → sin(2x)`.
///
/// Bottom-up fixpoint application of the contraction rule.
pub fn trig_contract() -> Strategy {
    innermost(rule_contract_sin_cos(), 32)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::normalize;

    fn x() -> Arc<Expr> {
        Expr::symbol("trig_x")
    }

    fn sin_x() -> Arc<Expr> {
        sin(x())
    }

    fn cos_x() -> Arc<Expr> {
        cos(x())
    }

    // ── sin(0) → 0 ────────────────────────────────────────────────────────

    #[test]
    fn test_sin_zero() {
        let e = Expr::func(FuncId::Sin, vec![Expr::int(0)]);
        let result = rule_sin_zero()(&e).unwrap();
        assert!(result.is_zero(), "sin(0) should be 0");
    }

    #[test]
    fn test_sin_nonzero_no_apply() {
        let e = Expr::func(FuncId::Sin, vec![x()]);
        assert!(rule_sin_zero()(&e).is_none(), "sin(x) should not reduce");
    }

    // ── cos(0) → 1 ────────────────────────────────────────────────────────

    #[test]
    fn test_cos_zero() {
        let e = Expr::func(FuncId::Cos, vec![Expr::int(0)]);
        let result = rule_cos_zero()(&e).unwrap();
        assert!(result.is_one(), "cos(0) should be 1");
    }

    #[test]
    fn test_cos_nonzero_no_apply() {
        let e = Expr::func(FuncId::Cos, vec![x()]);
        assert!(rule_cos_zero()(&e).is_none(), "cos(x) should not reduce");
    }

    // ── sin²(x) + cos²(x) → 1 ────────────────────────────────────────────

    #[test]
    fn test_pythagorean_identity_basic() {
        let two = Expr::int(2);
        let sin_sq = normalize::pow(sin_x(), two.clone());
        let cos_sq = normalize::pow(cos_x(), two);
        let expr = normalize::add(sin_sq, cos_sq);
        let result = trig_simplify()(&expr).unwrap();
        assert!(
            result.is_one(),
            "sin²(x) + cos²(x) should be 1, got {result}"
        );
    }

    #[test]
    fn test_pythagorean_with_extra_term() {
        let two = Expr::int(2);
        let sin_sq = normalize::pow(sin_x(), two.clone());
        let cos_sq = normalize::pow(cos_x(), two);
        let expr = normalize::add(normalize::add(sin_sq, cos_sq), Expr::symbol("trig_y"));
        let result = trig_simplify()(&expr).unwrap();
        // Should be 1 + y
        match result.as_ref() {
            Expr::Add(node) => {
                assert!(
                    node.constant.is_one(),
                    "constant should be 1, got {}",
                    node.constant
                );
                assert_eq!(node.term_count(), 1, "should have one remaining term");
            }
            _ => panic!("expected Add node, got {result}"),
        }
    }

    #[test]
    fn test_pythagorean_different_args_no_apply() {
        let two = Expr::int(2);
        let sin_sq = normalize::pow(sin(Expr::symbol("trig_px")), two.clone());
        let cos_sq = normalize::pow(cos(Expr::symbol("trig_py")), two);
        let expr = normalize::add(sin_sq, cos_sq);
        // Different arguments: should NOT reduce
        let result = trig_simplify()(&expr).unwrap();
        assert!(!result.is_one(), "sin²(x) + cos²(y) should NOT reduce to 1");
    }

    // ── sin(2x) → 2·sin(x)·cos(x) ────────────────────────────────────────

    #[test]
    fn test_double_angle_sin_expand() {
        let two_x = normalize::mul(Expr::int(2), x());
        let sin_2x = Expr::func(FuncId::Sin, vec![two_x]);
        let result = rule_double_angle_sin()(&sin_2x).unwrap();
        // Should be 2·sin(x)·cos(x) — a MulNode with coeff 2
        match result.as_ref() {
            Expr::Mul(m) => {
                use crate::numeric::BigRational;
                assert_eq!(m.coeff, BigRational::from(2i64), "coefficient should be 2");
                assert_eq!(m.factor_count(), 2, "should have sin and cos factors");
            }
            _ => panic!("expected Mul node, got {result}"),
        }
    }

    #[test]
    fn test_double_angle_sin_no_apply_for_nonmultiple() {
        // sin(3x) should not apply the double-angle rule
        let three_x = normalize::mul(Expr::int(3), x());
        let sin_3x = Expr::func(FuncId::Sin, vec![three_x]);
        assert!(rule_double_angle_sin()(&sin_3x).is_none());
    }

    // ── cos(2x) → cos²(x) - sin²(x) ─────────────────────────────────────

    #[test]
    fn test_double_angle_cos_expand() {
        let two_x = normalize::mul(Expr::int(2), x());
        let cos_2x = Expr::func(FuncId::Cos, vec![two_x]);
        let result = rule_double_angle_cos()(&cos_2x).unwrap();
        // Should be cos²(x) - sin²(x) — an Add or Mul node
        // Structure: AddNode containing cos^2 and -sin^2
        match result.as_ref() {
            Expr::Add(_) => {} // expected
            Expr::Mul(_) => {} // 1 term may collapse
            _ => panic!("expected Add or Mul, got {result}"),
        }
        let s = result.to_string();
        assert!(
            s.contains("cos") && s.contains("sin"),
            "result should contain cos and sin: {s}"
        );
    }

    // ── 2·sin(x)·cos(x) → sin(2x) ────────────────────────────────────────

    #[test]
    fn test_contract_sin_cos() {
        let two_sin_x_cos_x = normalize::mul(Expr::int(2), normalize::mul(sin_x(), cos_x()));
        let result = rule_contract_sin_cos()(&two_sin_x_cos_x).unwrap();
        // Should be sin(2x)
        let arg = unary_arg(FuncId::Sin, &result).expect("result should be sin(...)");
        // Argument should be 2*x
        match arg.as_ref() {
            Expr::Mul(m) => {
                use crate::numeric::BigRational;
                assert_eq!(m.coeff, BigRational::from(2i64));
                assert_eq!(m.factor_count(), 1);
            }
            _ => panic!("expected Mul(2, x) as sin arg, got {arg}"),
        }
    }

    #[test]
    fn test_contract_no_apply_wrong_coeff() {
        let three_sin_cos = normalize::mul(Expr::int(3), normalize::mul(sin_x(), cos_x()));
        assert!(
            rule_contract_sin_cos()(&three_sin_cos).is_none(),
            "coefficient 3 should not trigger contraction"
        );
    }

    // ── tan(x) → sin(x)/cos(x) ───────────────────────────────────────────

    #[test]
    fn test_tan_expansion() {
        let tan_x = Expr::func(FuncId::Tan, vec![x()]);
        let result = rule_tan_to_sin_cos()(&tan_x).unwrap();
        // Result is sin(x) * cos(x)^(-1) — either Mul or Pow
        let s = result.to_string();
        assert!(
            s.contains("sin") && s.contains("cos"),
            "should contain sin and cos: {s}"
        );
    }

    // ── Composed strategies ───────────────────────────────────────────────

    #[test]
    fn test_trig_simplify_constants() {
        // sin(0) + cos(0) should simplify to 0 + 1 = 1
        let e = normalize::add(
            Expr::func(FuncId::Sin, vec![Expr::int(0)]),
            Expr::func(FuncId::Cos, vec![Expr::int(0)]),
        );
        let result = trig_simplify()(&e).unwrap();
        assert!(result.is_one(), "sin(0)+cos(0) should be 1, got {result}");
    }

    #[test]
    fn test_trig_expand_sin_2x() {
        let two_x = normalize::mul(Expr::int(2), x());
        let sin_2x = Expr::func(FuncId::Sin, vec![two_x]);
        let result = trig_expand()(&sin_2x).unwrap();
        let s = result.to_string();
        assert!(
            s.contains("sin") && s.contains("cos"),
            "expanded form should have sin and cos: {s}"
        );
    }

    #[test]
    fn test_trig_contract_basic() {
        let two_sin_cos = normalize::mul(Expr::int(2), normalize::mul(sin_x(), cos_x()));
        let result = trig_contract()(&two_sin_cos).unwrap();
        // Should produce sin(2x)
        assert!(
            unary_arg(FuncId::Sin, &result).is_some(),
            "contracted form should be sin(...), got {result}"
        );
    }
}
