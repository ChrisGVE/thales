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
use crate::numeric::rewrite::{choice, innermost, Strategy};
use std::sync::Arc;

mod rules;
#[cfg(test)]
mod tests;

pub use rules::{
    rule_coeff_log_to_power, rule_exp_log_cancel, rule_exp_of_sum, rule_exp_product_to_sum,
    rule_exp_zero, rule_log_exp_cancel, rule_log_of_power, rule_log_of_product, rule_log_one,
    rule_log_sum_to_product,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Return the single argument of a unary `Func` node, or `None`.
pub(crate) fn unary_arg(id: FuncId, e: &Arc<Expr>) -> Option<Arc<Expr>> {
    match e.as_ref() {
        Expr::Func(fid, args) if *fid == id && args.len() == 1 => Some(args[0].clone()),
        _ => None,
    }
}

/// Build `ln(x)` as `Arc<Expr>`.
pub(crate) fn ln(x: Arc<Expr>) -> Arc<Expr> {
    Expr::func(FuncId::Ln, vec![x])
}

/// Build `exp(x)` as `Arc<Expr>`.
pub(crate) fn exp(x: Arc<Expr>) -> Arc<Expr> {
    Expr::func(FuncId::Exp, vec![x])
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
