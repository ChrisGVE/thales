//! Logarithm and exponential simplification strategies.
//!
//! Provides rewrite rules and composed strategies for `ln` and `exp` in the
//! numeric AST. Rules are structured as:
//!
//! - **Constant evaluation**: `ln(1) → 0`, `exp(0) → 1`
//! - **Cancellation**: `exp(ln(x)) → x`, `ln(exp(x)) → x`
//! - **Expansion** (`expand_log` / `exp_expand`):
//!   - `ln(a·b) → ln(a) + ln(b)`
//!   - `ln(a^b) → b·ln(a)`
//!   - `exp(a+b) → exp(a)·exp(b)`
//! - **Contraction** (`contract_log` / `exp_contract`):
//!   - `ln(a) + ln(b) → ln(a·b)`
//!   - `b·ln(a) → ln(a^b)`
//!   - `exp(a)·exp(b) → exp(a+b)`
//!
//! # Strategies
//! - [`log_exp_cancel`]  — cancellation identities and constant folding
//! - [`expand_log`]      — expand `ln` over products and powers
//! - [`contract_log`]    — contract sums/scaled logs back into single `ln`
//! - [`exp_expand`]      — expand `exp` over sums
//! - [`exp_contract`]    — contract products of `exp` into a single `exp`

use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::normalize;
use crate::numeric::rewrite::{choice, innermost, try_rule, Strategy};
use crate::numeric::BigRational;
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

/// Build `ln(x)` as `Arc<Expr>`.
fn ln(x: Arc<Expr>) -> Arc<Expr> {
    Expr::func(FuncId::Ln, vec![x])
}

/// Build `exp(x)` as `Arc<Expr>`.
fn exp(x: Arc<Expr>) -> Arc<Expr> {
    Expr::func(FuncId::Exp, vec![x])
}

// ── Constant evaluation rules ─────────────────────────────────────────────────

/// `ln(1) → 0`
///
/// # Examples
///
/// ```rust
/// use thales::numeric::expr::Expr;
/// use thales::numeric::log_exp_rules::rule_log_one;
///
/// let e = Expr::func(thales::numeric::FuncId::Ln, vec![Expr::int(1)]);
/// let result = rule_log_one()(&e).unwrap();
/// assert!(result.is_zero());
/// ```
pub fn rule_log_one() -> Strategy {
    try_rule(|e| {
        let arg = unary_arg(FuncId::Ln, e)?;
        if arg.is_one() {
            Some(Expr::int(0))
        } else {
            None
        }
    })
}

/// `exp(0) → 1`
///
/// # Examples
///
/// ```rust
/// use thales::numeric::expr::Expr;
/// use thales::numeric::log_exp_rules::rule_exp_zero;
///
/// let e = Expr::func(thales::numeric::FuncId::Exp, vec![Expr::int(0)]);
/// let result = rule_exp_zero()(&e).unwrap();
/// assert!(result.is_one());
/// ```
pub fn rule_exp_zero() -> Strategy {
    try_rule(|e| {
        let arg = unary_arg(FuncId::Exp, e)?;
        if arg.is_zero() {
            Some(Expr::int(1))
        } else {
            None
        }
    })
}

// ── Cancellation rules ────────────────────────────────────────────────────────

/// `exp(ln(x)) → x`
///
/// Domain consideration: valid for all `x > 0`. The rule applies structurally;
/// callers are responsible for ensuring positivity when needed.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::expr::Expr;
/// use thales::numeric::log_exp_rules::rule_exp_log_cancel;
/// use thales::numeric::FuncId;
///
/// let x = Expr::symbol("x");
/// let e = Expr::func(FuncId::Exp, vec![Expr::func(FuncId::Ln, vec![x.clone()])]);
/// let result = rule_exp_log_cancel()(&e).unwrap();
/// assert_eq!(*result, *x);
/// ```
pub fn rule_exp_log_cancel() -> Strategy {
    try_rule(|e| {
        let outer = unary_arg(FuncId::Exp, e)?;
        let inner = unary_arg(FuncId::Ln, &outer)?;
        Some(inner)
    })
}

/// `ln(exp(x)) → x`
///
/// # Examples
///
/// ```rust
/// use thales::numeric::expr::Expr;
/// use thales::numeric::log_exp_rules::rule_log_exp_cancel;
/// use thales::numeric::FuncId;
///
/// let x = Expr::symbol("x");
/// let e = Expr::func(FuncId::Ln, vec![Expr::func(FuncId::Exp, vec![x.clone()])]);
/// let result = rule_log_exp_cancel()(&e).unwrap();
/// assert_eq!(*result, *x);
/// ```
pub fn rule_log_exp_cancel() -> Strategy {
    try_rule(|e| {
        let outer = unary_arg(FuncId::Ln, e)?;
        let inner = unary_arg(FuncId::Exp, &outer)?;
        Some(inner)
    })
}

// ── Log expansion rules ───────────────────────────────────────────────────────

/// `ln(a·b) → ln(a) + ln(b)`
///
/// Matches `ln` applied to a `MulNode` with two or more symbolic factors.
/// The coefficient of the `MulNode` is handled separately: if the coefficient
/// is not 1 it is left in place to avoid introducing `ln(coeff)` terms.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{normalize, log_exp_rules::rule_log_of_product, FuncId};
/// use thales::numeric::expr::Expr;
///
/// let a = Expr::symbol("a");
/// let b = Expr::symbol("b");
/// let prod = normalize::mul(a.clone(), b.clone());
/// let e = Expr::func(FuncId::Ln, vec![prod]);
/// let result = rule_log_of_product()(&e).unwrap();
/// let s = result.to_string();
/// assert!(s.contains("ln(a)") && s.contains("ln(b)"), "got: {s}");
/// ```
pub fn rule_log_of_product() -> Strategy {
    try_rule(|e| {
        let arg = unary_arg(FuncId::Ln, e)?;
        let m = match arg.as_ref() {
            Expr::Mul(m) => m,
            _ => return None,
        };
        // Require at least two symbolic factors (exponent 1) to expand
        let unit_factors: Vec<Arc<Expr>> = m
            .factors
            .iter()
            .filter(|(_, exp)| exp.is_one())
            .map(|(base, _)| base.clone())
            .collect();
        if unit_factors.len() < 2 {
            return None;
        }
        // Build sum of ln(factor) for each factor
        let terms: Vec<Arc<Expr>> = unit_factors.into_iter().map(ln).collect();
        Some(normalize::add_many(terms))
    })
}

/// `ln(a^b) → b·ln(a)`
///
/// Matches `ln` applied to a `Pow` node.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{normalize, log_exp_rules::rule_log_of_power, FuncId};
/// use thales::numeric::expr::Expr;
///
/// let a = Expr::symbol("a");
/// let b = Expr::symbol("b");
/// let pow = normalize::pow(a.clone(), b.clone());
/// let e = Expr::func(FuncId::Ln, vec![pow]);
/// let result = rule_log_of_power()(&e).unwrap();
/// let s = result.to_string();
/// assert!(s.contains("ln(a)") && s.contains('b'), "got: {s}");
/// ```
pub fn rule_log_of_power() -> Strategy {
    try_rule(|e| {
        let arg = unary_arg(FuncId::Ln, e)?;
        let (base, exp) = match arg.as_ref() {
            Expr::Pow(b, ex) => (b.clone(), ex.clone()),
            _ => return None,
        };
        // ln(a^b) = b * ln(a)
        Some(normalize::mul(exp, ln(base)))
    })
}

// ── Exp expansion rule ────────────────────────────────────────────────────────

/// `exp(a+b) → exp(a)·exp(b)`
///
/// Matches `exp` applied to an `AddNode` with two or more terms.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{normalize, log_exp_rules::rule_exp_of_sum, FuncId};
/// use thales::numeric::expr::Expr;
///
/// let a = Expr::symbol("a");
/// let b = Expr::symbol("b");
/// let sum = normalize::add(a.clone(), b.clone());
/// let e = Expr::func(FuncId::Exp, vec![sum]);
/// let result = rule_exp_of_sum()(&e).unwrap();
/// let s = result.to_string();
/// assert!(s.contains("exp(a)") && s.contains("exp(b)"), "got: {s}");
/// ```
pub fn rule_exp_of_sum() -> Strategy {
    try_rule(|e| {
        let arg = unary_arg(FuncId::Exp, e)?;
        let node = match arg.as_ref() {
            Expr::Add(n) => n,
            _ => return None,
        };
        // Require at least two symbolic terms
        if node.term_count() < 2 {
            return None;
        }
        // Build product of exp(coeff*term) for each term, ignoring constant here
        // (constant part handled separately below)
        let mut factors: Vec<Arc<Expr>> = node
            .terms
            .iter()
            .map(|(term, coeff)| {
                let scaled = normalize::mul(Arc::new(Expr::Rational(coeff.clone())), term.clone());
                exp(scaled)
            })
            .collect();
        // If there is a nonzero constant, include exp(constant)
        if !node.constant.is_zero() {
            factors.push(exp(Arc::new(Expr::Rational(node.constant.clone()))));
        }
        if factors.len() < 2 {
            return None;
        }
        let mut result = factors.remove(0);
        for f in factors {
            result = normalize::mul(result, f);
        }
        Some(result)
    })
}

// ── Log contraction rules ─────────────────────────────────────────────────────

/// `ln(a) + ln(b) → ln(a·b)`
///
/// Matches an `AddNode` containing at least two `ln(...)` terms, each with
/// coefficient 1. Any remaining terms are preserved.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{normalize, log_exp_rules::rule_log_sum_to_product, FuncId};
/// use thales::numeric::expr::Expr;
///
/// let a = Expr::symbol("a");
/// let b = Expr::symbol("b");
/// let e = normalize::add(
///     Expr::func(FuncId::Ln, vec![a.clone()]),
///     Expr::func(FuncId::Ln, vec![b.clone()]),
/// );
/// let result = rule_log_sum_to_product()(&e).unwrap();
/// let s = result.to_string();
/// assert!(s.contains("ln(") && s.contains('a') && s.contains('b'), "got: {s}");
/// ```
pub fn rule_log_sum_to_product() -> Strategy {
    try_rule(|e| {
        let node = match e.as_ref() {
            Expr::Add(n) => n,
            _ => return None,
        };
        // Collect all ln(x) terms with coefficient 1
        let mut ln_args: Vec<Arc<Expr>> = Vec::new();
        for (term, coeff) in &node.terms {
            if *coeff == BigRational::one() {
                if let Some(arg) = unary_arg(FuncId::Ln, term) {
                    ln_args.push(arg);
                }
            }
        }
        if ln_args.len() < 2 {
            return None;
        }
        // Build ln(a*b*...) from the collected arguments
        let product = ln_args
            .into_iter()
            .reduce(|acc, x| normalize::mul(acc, x))
            .expect("at least 2 elements");
        let contracted = ln(product);
        // Re-add non-ln terms and remaining terms
        let mut remaining: Vec<Arc<Expr>> = vec![contracted];
        if !node.constant.is_zero() {
            remaining.push(Arc::new(Expr::Rational(node.constant.clone())));
        }
        for (term, coeff) in &node.terms {
            if unary_arg(FuncId::Ln, term).is_some() && *coeff == BigRational::one() {
                continue; // already consumed
            }
            remaining.push(normalize::mul(
                Arc::new(Expr::Rational(coeff.clone())),
                term.clone(),
            ));
        }
        Some(normalize::add_many(remaining))
    })
}

/// `b·ln(a) → ln(a^b)`
///
/// Matches a `MulNode` whose single factor is `ln(x)^1` and whose rational
/// coefficient `b` can be used as an exponent.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{normalize, log_exp_rules::rule_coeff_log_to_power, FuncId};
/// use thales::numeric::expr::Expr;
///
/// let a = Expr::symbol("a");
/// let ln_a = Expr::func(FuncId::Ln, vec![a.clone()]);
/// // 3 * ln(a)
/// let e = normalize::mul(Expr::int(3), ln_a);
/// let result = rule_coeff_log_to_power()(&e).unwrap();
/// let s = result.to_string();
/// assert!(s.contains("ln(") && s.contains('3') && s.contains('a'), "got: {s}");
/// ```
pub fn rule_coeff_log_to_power() -> Strategy {
    try_rule(|e| {
        let m = match e.as_ref() {
            Expr::Mul(m) => m,
            _ => return None,
        };
        // Exactly one factor with exponent 1, which must be ln(x)
        if m.factor_count() != 1 {
            return None;
        }
        let (base, exp_e) = m.factors.iter().next()?;
        if !exp_e.is_one() {
            return None;
        }
        let ln_arg = unary_arg(FuncId::Ln, base)?;
        // b*ln(a) → ln(a^b)
        let b = Arc::new(Expr::Rational(m.coeff.clone()));
        Some(ln(normalize::pow(ln_arg, b)))
    })
}

// ── Exp contraction rule ──────────────────────────────────────────────────────

/// `exp(a)·exp(b) → exp(a+b)`
///
/// Matches a `MulNode` with coefficient 1 and exactly two factors `exp(x)^1`
/// and `exp(y)^1`. Any additional non-exp factors are left outside.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{normalize, log_exp_rules::rule_exp_product_to_sum, FuncId};
/// use thales::numeric::expr::Expr;
///
/// let a = Expr::symbol("a");
/// let b = Expr::symbol("b");
/// let e = normalize::mul(
///     Expr::func(FuncId::Exp, vec![a.clone()]),
///     Expr::func(FuncId::Exp, vec![b.clone()]),
/// );
/// let result = rule_exp_product_to_sum()(&e).unwrap();
/// let s = result.to_string();
/// assert!(s.contains("exp(") && s.contains('a') && s.contains('b'), "got: {s}");
/// ```
pub fn rule_exp_product_to_sum() -> Strategy {
    try_rule(|e| {
        let m = match e.as_ref() {
            Expr::Mul(m) => m,
            _ => return None,
        };
        // Collect all exp(x)^1 factors
        let exp_args: Vec<Arc<Expr>> = m
            .factors
            .iter()
            .filter(|(_, exp_e)| exp_e.is_one())
            .filter_map(|(base, _)| unary_arg(FuncId::Exp, base))
            .collect();
        if exp_args.len() < 2 {
            return None;
        }
        // Build exp(a+b+...) from all collected arguments
        let sum = exp_args
            .into_iter()
            .reduce(|acc, x| normalize::add(acc, x))
            .expect("at least 2 elements");
        Some(exp(sum))
    })
}

// ── Composed strategies ───────────────────────────────────────────────────────

/// Cancel inverse pairs: `exp(ln(x)) → x`, `ln(exp(x)) → x`,
/// `ln(1) → 0`, `exp(0) → 1`.
///
/// Applies bottom-up until fixpoint.
///
/// # Examples
///
/// ```rust
/// use std::sync::Arc;
/// use thales::numeric::{normalize, log_exp_rules::log_exp_cancel, FuncId};
/// use thales::numeric::expr::Expr;
///
/// let x = Expr::symbol("x");
/// let e = Expr::func(FuncId::Exp, vec![Expr::func(FuncId::Ln, vec![x.clone()])]);
/// let result = log_exp_cancel()(&e).unwrap();
/// assert_eq!(*result, *x);
/// ```
pub fn log_exp_cancel() -> Strategy {
    let rule = choice(
        rule_exp_log_cancel(),
        choice(
            rule_log_exp_cancel(),
            choice(rule_log_one(), rule_exp_zero()),
        ),
    );
    innermost(rule, 32)
}

/// Expand `ln` over products and powers, and `exp` over sums.
///
/// Applies `ln(a·b) → ln(a) + ln(b)`, `ln(a^b) → b·ln(a)`,
/// and `exp(a+b) → exp(a)·exp(b)` bottom-up until fixpoint.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{normalize, log_exp_rules::expand_log, FuncId};
/// use thales::numeric::expr::Expr;
///
/// let a = Expr::symbol("a");
/// let b = Expr::symbol("b");
/// let prod = normalize::mul(a.clone(), b.clone());
/// let e = Expr::func(FuncId::Ln, vec![prod]);
/// let result = expand_log()(&e).unwrap();
/// let s = result.to_string();
/// assert!(s.contains("ln(a)") && s.contains("ln(b)"), "got: {s}");
/// ```
pub fn expand_log() -> Strategy {
    let rule = choice(
        rule_log_of_product(),
        choice(rule_log_of_power(), rule_exp_of_sum()),
    );
    innermost(rule, 32)
}

/// Contract sums/scaled logs back into single `ln`, and products of `exp`
/// into a single `exp`.
///
/// Applies `ln(a) + ln(b) → ln(a·b)`, `b·ln(a) → ln(a^b)`,
/// and `exp(a)·exp(b) → exp(a+b)` bottom-up until fixpoint.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{normalize, log_exp_rules::contract_log, FuncId};
/// use thales::numeric::expr::Expr;
///
/// let a = Expr::symbol("a");
/// let b = Expr::symbol("b");
/// let e = normalize::add(
///     Expr::func(FuncId::Ln, vec![a.clone()]),
///     Expr::func(FuncId::Ln, vec![b.clone()]),
/// );
/// let result = contract_log()(&e).unwrap();
/// let s = result.to_string();
/// assert!(s.contains("ln(") && s.contains('a') && s.contains('b'), "got: {s}");
/// ```
pub fn contract_log() -> Strategy {
    let rule = choice(
        rule_log_sum_to_product(),
        choice(rule_coeff_log_to_power(), rule_exp_product_to_sum()),
    );
    innermost(rule, 32)
}

/// Expand `exp` over sums (alias targeting `exp` expansion only).
///
/// Applies `exp(a+b) → exp(a)·exp(b)` bottom-up until fixpoint.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{normalize, log_exp_rules::exp_expand, FuncId};
/// use thales::numeric::expr::Expr;
///
/// let a = Expr::symbol("a");
/// let b = Expr::symbol("b");
/// let sum = normalize::add(a.clone(), b.clone());
/// let e = Expr::func(FuncId::Exp, vec![sum]);
/// let result = exp_expand()(&e).unwrap();
/// let s = result.to_string();
/// assert!(s.contains("exp(a)") && s.contains("exp(b)"), "got: {s}");
/// ```
pub fn exp_expand() -> Strategy {
    innermost(rule_exp_of_sum(), 32)
}

/// Contract products of `exp` into a single `exp` (alias targeting `exp`
/// contraction only).
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{normalize, log_exp_rules::exp_contract, FuncId};
/// use thales::numeric::expr::Expr;
///
/// let a = Expr::symbol("a");
/// let b = Expr::symbol("b");
/// let e = normalize::mul(
///     Expr::func(FuncId::Exp, vec![a.clone()]),
///     Expr::func(FuncId::Exp, vec![b.clone()]),
/// );
/// let result = exp_contract()(&e).unwrap();
/// let s = result.to_string();
/// assert!(s.contains("exp(") && s.contains('a') && s.contains('b'), "got: {s}");
/// ```
pub fn exp_contract() -> Strategy {
    innermost(rule_exp_product_to_sum(), 32)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::normalize;

    fn x() -> Arc<Expr> {
        Expr::symbol("log_x")
    }

    fn a() -> Arc<Expr> {
        Expr::symbol("log_a")
    }

    fn b() -> Arc<Expr> {
        Expr::symbol("log_b")
    }

    // ── ln(1) → 0 ─────────────────────────────────────────────────────────

    #[test]
    fn test_log_one() {
        let e = Expr::func(FuncId::Ln, vec![Expr::int(1)]);
        let result = rule_log_one()(&e).unwrap();
        assert!(result.is_zero(), "ln(1) should be 0");
    }

    #[test]
    fn test_log_one_no_apply_for_nonone() {
        let e = Expr::func(FuncId::Ln, vec![x()]);
        assert!(rule_log_one()(&e).is_none());
    }

    // ── exp(0) → 1 ────────────────────────────────────────────────────────

    #[test]
    fn test_exp_zero() {
        let e = Expr::func(FuncId::Exp, vec![Expr::int(0)]);
        let result = rule_exp_zero()(&e).unwrap();
        assert!(result.is_one(), "exp(0) should be 1");
    }

    #[test]
    fn test_exp_zero_no_apply_for_nonzero() {
        let e = Expr::func(FuncId::Exp, vec![x()]);
        assert!(rule_exp_zero()(&e).is_none());
    }

    // ── exp(ln(x)) → x ────────────────────────────────────────────────────

    #[test]
    fn test_exp_log_cancel() {
        let e = Expr::func(FuncId::Exp, vec![Expr::func(FuncId::Ln, vec![x()])]);
        let result = rule_exp_log_cancel()(&e).unwrap();
        assert_eq!(*result, *x());
    }

    #[test]
    fn test_exp_log_cancel_no_apply_for_plain_exp() {
        let e = Expr::func(FuncId::Exp, vec![x()]);
        assert!(rule_exp_log_cancel()(&e).is_none());
    }

    // ── ln(exp(x)) → x ────────────────────────────────────────────────────

    #[test]
    fn test_log_exp_cancel() {
        let e = Expr::func(FuncId::Ln, vec![Expr::func(FuncId::Exp, vec![x()])]);
        let result = rule_log_exp_cancel()(&e).unwrap();
        assert_eq!(*result, *x());
    }

    #[test]
    fn test_log_exp_cancel_no_apply_for_plain_ln() {
        let e = Expr::func(FuncId::Ln, vec![x()]);
        assert!(rule_log_exp_cancel()(&e).is_none());
    }

    // ── ln(a*b) → ln(a) + ln(b) ───────────────────────────────────────────

    #[test]
    fn test_log_of_product() {
        let prod = normalize::mul(a(), b());
        let e = Expr::func(FuncId::Ln, vec![prod]);
        let result = rule_log_of_product()(&e).unwrap();
        let s = result.to_string();
        assert!(
            s.contains("ln(log_a)") && s.contains("ln(log_b)"),
            "expected ln(a)+ln(b), got: {s}"
        );
    }

    #[test]
    fn test_log_of_product_no_apply_for_single_factor() {
        let e = Expr::func(FuncId::Ln, vec![x()]);
        assert!(rule_log_of_product()(&e).is_none());
    }

    // ── ln(a^b) → b*ln(a) ─────────────────────────────────────────────────

    #[test]
    fn test_log_of_power() {
        let pow = normalize::pow(a(), b());
        let e = Expr::func(FuncId::Ln, vec![pow]);
        let result = rule_log_of_power()(&e).unwrap();
        let s = result.to_string();
        assert!(
            s.contains("ln(log_a)") && s.contains("log_b"),
            "expected b*ln(a), got: {s}"
        );
    }

    #[test]
    fn test_log_of_power_integer_exp() {
        // ln(x^3) → 3*ln(x)
        let pow = normalize::pow(x(), Expr::int(3));
        let e = Expr::func(FuncId::Ln, vec![pow]);
        let result = rule_log_of_power()(&e).unwrap();
        let s = result.to_string();
        assert!(
            s.contains("ln(log_x)") && s.contains('3'),
            "expected 3*ln(x), got: {s}"
        );
    }

    // ── exp(a+b) → exp(a)*exp(b) ──────────────────────────────────────────

    #[test]
    fn test_exp_of_sum() {
        let sum = normalize::add(a(), b());
        let e = Expr::func(FuncId::Exp, vec![sum]);
        let result = rule_exp_of_sum()(&e).unwrap();
        let s = result.to_string();
        assert!(
            s.contains("exp(log_a)") && s.contains("exp(log_b)"),
            "expected exp(a)*exp(b), got: {s}"
        );
    }

    #[test]
    fn test_exp_of_sum_single_term_no_apply() {
        // exp(x) alone — no sum, rule should not apply
        let e = Expr::func(FuncId::Exp, vec![x()]);
        assert!(rule_exp_of_sum()(&e).is_none());
    }

    // ── ln(a) + ln(b) → ln(a*b) ───────────────────────────────────────────

    #[test]
    fn test_log_sum_to_product() {
        let e = normalize::add(
            Expr::func(FuncId::Ln, vec![a()]),
            Expr::func(FuncId::Ln, vec![b()]),
        );
        let result = rule_log_sum_to_product()(&e).unwrap();
        let s = result.to_string();
        assert!(
            s.contains("ln(") && s.contains("log_a") && s.contains("log_b"),
            "expected ln(a*b), got: {s}"
        );
    }

    #[test]
    fn test_log_sum_single_ln_no_apply() {
        // Only one ln term — should not contract
        let e = normalize::add(Expr::func(FuncId::Ln, vec![a()]), x());
        assert!(rule_log_sum_to_product()(&e).is_none());
    }

    // ── b*ln(a) → ln(a^b) ─────────────────────────────────────────────────

    #[test]
    fn test_coeff_log_to_power() {
        let ln_a = Expr::func(FuncId::Ln, vec![a()]);
        let e = normalize::mul(Expr::int(3), ln_a);
        let result = rule_coeff_log_to_power()(&e).unwrap();
        let s = result.to_string();
        assert!(
            s.contains("ln(") && s.contains('3') && s.contains("log_a"),
            "expected ln(a^3), got: {s}"
        );
    }

    #[test]
    fn test_coeff_log_to_power_no_apply_for_non_ln() {
        // 3*sin(a) — should not apply
        let sin_a = Expr::func(FuncId::Sin, vec![a()]);
        let e = normalize::mul(Expr::int(3), sin_a);
        assert!(rule_coeff_log_to_power()(&e).is_none());
    }

    // ── exp(a)*exp(b) → exp(a+b) ──────────────────────────────────────────

    #[test]
    fn test_exp_product_to_sum() {
        let e = normalize::mul(
            Expr::func(FuncId::Exp, vec![a()]),
            Expr::func(FuncId::Exp, vec![b()]),
        );
        let result = rule_exp_product_to_sum()(&e).unwrap();
        let s = result.to_string();
        assert!(
            s.contains("exp(") && s.contains("log_a") && s.contains("log_b"),
            "expected exp(a+b), got: {s}"
        );
    }

    #[test]
    fn test_exp_product_single_no_apply() {
        // Single exp — should not apply
        let e = Expr::func(FuncId::Exp, vec![a()]);
        assert!(rule_exp_product_to_sum()(&e).is_none());
    }

    // ── Composed: log_exp_cancel ──────────────────────────────────────────

    #[test]
    fn test_composed_cancel_exp_ln() {
        let e = Expr::func(FuncId::Exp, vec![Expr::func(FuncId::Ln, vec![x()])]);
        let result = log_exp_cancel()(&e).unwrap();
        assert_eq!(*result, *x(), "exp(ln(x)) should cancel to x");
    }

    #[test]
    fn test_composed_cancel_ln_exp() {
        let e = Expr::func(FuncId::Ln, vec![Expr::func(FuncId::Exp, vec![x()])]);
        let result = log_exp_cancel()(&e).unwrap();
        assert_eq!(*result, *x(), "ln(exp(x)) should cancel to x");
    }

    #[test]
    fn test_composed_cancel_ln_one() {
        let e = Expr::func(FuncId::Ln, vec![Expr::int(1)]);
        let result = log_exp_cancel()(&e).unwrap();
        assert!(result.is_zero(), "ln(1) should be 0");
    }

    #[test]
    fn test_composed_cancel_exp_zero() {
        let e = Expr::func(FuncId::Exp, vec![Expr::int(0)]);
        let result = log_exp_cancel()(&e).unwrap();
        assert!(result.is_one(), "exp(0) should be 1");
    }

    // ── Composed: expand_log ──────────────────────────────────────────────

    #[test]
    fn test_expand_log_product() {
        let prod = normalize::mul(a(), b());
        let e = Expr::func(FuncId::Ln, vec![prod]);
        let result = expand_log()(&e).unwrap();
        let s = result.to_string();
        assert!(
            s.contains("ln(log_a)") && s.contains("ln(log_b)"),
            "expand_log should expand ln(a*b): {s}"
        );
    }

    #[test]
    fn test_expand_log_power() {
        let pow = normalize::pow(a(), Expr::int(2));
        let e = Expr::func(FuncId::Ln, vec![pow]);
        let result = expand_log()(&e).unwrap();
        let s = result.to_string();
        assert!(
            s.contains("ln(log_a)") && s.contains('2'),
            "expand_log should give 2*ln(a): {s}"
        );
    }

    #[test]
    fn test_expand_exp_sum() {
        let sum = normalize::add(a(), b());
        let e = Expr::func(FuncId::Exp, vec![sum]);
        let result = expand_log()(&e).unwrap();
        let s = result.to_string();
        assert!(
            s.contains("exp(log_a)") && s.contains("exp(log_b)"),
            "expand_log should expand exp(a+b): {s}"
        );
    }

    // ── Composed: contract_log ────────────────────────────────────────────

    #[test]
    fn test_contract_log_sum() {
        let e = normalize::add(
            Expr::func(FuncId::Ln, vec![a()]),
            Expr::func(FuncId::Ln, vec![b()]),
        );
        let result = contract_log()(&e).unwrap();
        let s = result.to_string();
        assert!(
            s.contains("ln(") && s.contains("log_a") && s.contains("log_b"),
            "contract_log should produce ln(a*b): {s}"
        );
    }

    #[test]
    fn test_contract_exp_product() {
        let e = normalize::mul(
            Expr::func(FuncId::Exp, vec![a()]),
            Expr::func(FuncId::Exp, vec![b()]),
        );
        let result = contract_log()(&e).unwrap();
        let s = result.to_string();
        assert!(
            s.contains("exp(") && s.contains("log_a") && s.contains("log_b"),
            "contract_log should produce exp(a+b): {s}"
        );
    }

    // ── Composed: exp_expand ──────────────────────────────────────────────

    #[test]
    fn test_exp_expand_strategy() {
        let sum = normalize::add(a(), b());
        let e = Expr::func(FuncId::Exp, vec![sum]);
        let result = exp_expand()(&e).unwrap();
        let s = result.to_string();
        assert!(
            s.contains("exp(log_a)") && s.contains("exp(log_b)"),
            "exp_expand should expand exp(a+b): {s}"
        );
    }

    // ── Composed: exp_contract ────────────────────────────────────────────

    #[test]
    fn test_exp_contract_strategy() {
        let e = normalize::mul(
            Expr::func(FuncId::Exp, vec![a()]),
            Expr::func(FuncId::Exp, vec![b()]),
        );
        let result = exp_contract()(&e).unwrap();
        let s = result.to_string();
        assert!(
            s.contains("exp(") && s.contains("log_a") && s.contains("log_b"),
            "exp_contract should produce exp(a+b): {s}"
        );
    }
}
