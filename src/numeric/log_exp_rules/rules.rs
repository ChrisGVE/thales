//! Individual rewrite rules for logarithm and exponential simplification.

use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::normalize;
use crate::numeric::rewrite::{try_rule, Strategy};
use crate::numeric::BigRational;
use num::traits::{One, Zero};
use std::sync::Arc;

use super::{exp, ln, unary_arg};

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
        let (base, exp_e) = match arg.as_ref() {
            Expr::Pow(b, ex) => (b.clone(), ex.clone()),
            _ => return None,
        };
        // ln(a^b) = b * ln(a)
        Some(normalize::mul(exp_e, ln(base)))
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
