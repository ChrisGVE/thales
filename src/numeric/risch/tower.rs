//! Transcendental tower identification for the Risch algorithm.
//!
//! Analyzes an [`Expr`] to determine the tower of transcendental extensions
//! needed to integrate it. The tower is the ordered list of generators:
//!
//! - Level 0: the base field `Q(x)` (rational functions in `x`).
//! - Level k: a new transcendental `θₖ` which is either `ln(uₖ)` or `exp(uₖ)`
//!   for some expression `uₖ` already in the lower tower.
//!
//! # Example towers
//!
//! | Integrand        | Tower                         |
//! |------------------|-------------------------------|
//! | `1/x`            | `[x]` — rational only         |
//! | `ln(x)`          | `[x, ln(x)]`                  |
//! | `exp(x)`         | `[x, exp(x)]`                 |
//! | `x·exp(x²)`      | `[x, exp(x²)]`                |
//! | `exp(x)·ln(x)`   | `[x, ln(x), exp(x)]`          |

use crate::numeric::expr::{Expr, FuncId};
use crate::numeric::SymbolId;
use std::sync::Arc;

// ── TowerLevel ───────────────────────────────────────────────────────────────

/// A single level in the transcendental tower.
#[derive(Clone, Debug, PartialEq)]
pub enum TowerLevel {
    /// The base rational variable `x`.
    Base(SymbolId),
    /// A logarithmic extension `ln(arg)`.
    Log(Arc<Expr>),
    /// An exponential extension `exp(arg)`.
    Exp(Arc<Expr>),
}

impl TowerLevel {
    /// Returns `true` if this level is a logarithmic extension.
    pub fn is_log(&self) -> bool {
        matches!(self, TowerLevel::Log(_))
    }

    /// Returns `true` if this level is an exponential extension.
    pub fn is_exp(&self) -> bool {
        matches!(self, TowerLevel::Exp(_))
    }
}

// ── TowerKind ────────────────────────────────────────────────────────────────

/// High-level classification of an integrand's tower structure.
#[derive(Clone, Debug, PartialEq)]
pub enum TowerKind {
    /// No transcendental extensions — pure rational function.
    Rational,
    /// Extensions are logarithms only.
    Logarithmic,
    /// Extensions are exponentials only.
    Exponential,
    /// Both logarithmic and exponential extensions present.
    Mixed,
}

// ── Tower ────────────────────────────────────────────────────────────────────

/// The full transcendental tower for an integrand.
///
/// The `levels` field lists generator extensions in dependency order:
/// `levels[0]` is always the base variable; each subsequent level depends
/// only on previous ones.
#[derive(Clone, Debug)]
pub struct Tower {
    /// Ordered list of tower levels, starting with the base variable.
    pub levels: Vec<TowerLevel>,
    /// High-level classification of the tower.
    pub kind: TowerKind,
}

impl Tower {
    /// Construct a tower containing only the base rational variable.
    pub fn rational(var: SymbolId) -> Self {
        Tower {
            levels: vec![TowerLevel::Base(var)],
            kind: TowerKind::Rational,
        }
    }

    /// Number of transcendental extension levels (excluding the base).
    pub fn extension_depth(&self) -> usize {
        self.levels.len().saturating_sub(1)
    }

    /// Returns `true` if the tower contains any logarithmic extension.
    pub fn has_log(&self) -> bool {
        self.levels.iter().any(|l| l.is_log())
    }

    /// Returns `true` if the tower contains any exponential extension.
    pub fn has_exp(&self) -> bool {
        self.levels.iter().any(|l| l.is_exp())
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Analyze `expr` and build the minimal transcendental tower required to
/// represent (and potentially integrate) it with respect to `var`.
///
/// The algorithm walks the expression tree, collecting every `ln(·)` and
/// `exp(·)` sub-expression. Duplicate arguments are collapsed so each
/// unique transcendental appears at most once in the tower.
///
/// # Examples
///
/// ```rust
/// use thales::numeric::{Expr, FuncId, SymbolId};
/// use thales::numeric::risch::tower::{build_tower, TowerKind};
///
/// let x_id = SymbolId::intern("tw_x");
/// let x = Expr::symbol("tw_x");
/// let ln_x = Expr::func(FuncId::Ln, vec![x]);
/// let tower = build_tower(&ln_x, x_id);
/// assert_eq!(tower.kind, TowerKind::Logarithmic);
/// assert_eq!(tower.extension_depth(), 1);
/// ```
pub fn build_tower(expr: &Arc<Expr>, var: SymbolId) -> Tower {
    let mut logs: Vec<Arc<Expr>> = Vec::new();
    let mut exps: Vec<Arc<Expr>> = Vec::new();
    collect_extensions(expr, &mut logs, &mut exps);

    let kind = match (logs.is_empty(), exps.is_empty()) {
        (true, true) => TowerKind::Rational,
        (false, true) => TowerKind::Logarithmic,
        (true, false) => TowerKind::Exponential,
        (false, false) => TowerKind::Mixed,
    };

    let mut levels = vec![TowerLevel::Base(var)];
    for arg in logs {
        levels.push(TowerLevel::Log(arg));
    }
    for arg in exps {
        levels.push(TowerLevel::Exp(arg));
    }

    Tower { levels, kind }
}

/// Recursively collect `ln(arg)` and `exp(arg)` sub-expressions.
///
/// Deduplicates by argument structural equality so each unique
/// transcendental generator appears at most once.
fn collect_extensions(expr: &Arc<Expr>, logs: &mut Vec<Arc<Expr>>, exps: &mut Vec<Arc<Expr>>) {
    match expr.as_ref() {
        Expr::Integer(_) | Expr::Rational(_) | Expr::Float(_) | Expr::Symbol(_) => {}

        Expr::Add(node) => {
            for (term, _) in &node.terms {
                collect_extensions(term, logs, exps);
            }
        }

        Expr::Mul(node) => {
            for (base, exp) in &node.factors {
                collect_extensions(base, logs, exps);
                collect_extensions(exp, logs, exps);
            }
        }

        Expr::Pow(base, exp) => {
            collect_extensions(base, logs, exps);
            collect_extensions(exp, logs, exps);
        }

        Expr::Func(FuncId::Ln, args) if args.len() == 1 => {
            let arg = &args[0];
            if !logs.iter().any(|a| a == arg) {
                logs.push(arg.clone());
            }
            // Descend into argument for nested transcendentals
            collect_extensions(arg, logs, exps);
        }

        Expr::Func(FuncId::Exp, args) if args.len() == 1 => {
            let arg = &args[0];
            if !exps.iter().any(|a| a == arg) {
                exps.push(arg.clone());
            }
            collect_extensions(arg, logs, exps);
        }

        Expr::Func(_, args) => {
            for arg in args {
                collect_extensions(arg, logs, exps);
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{normalize, Expr, FuncId, SymbolId};

    fn x_sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }

    fn x_id(name: &str) -> SymbolId {
        SymbolId::intern(name)
    }

    #[test]
    fn test_tower_pure_rational() {
        let x = x_sym("tw_rat_x");
        // 1/(x^2 + 1) — no transcendentals
        let denom = normalize::add(normalize::pow(x.clone(), Expr::int(2)), Expr::int(1));
        let expr = normalize::div(Expr::int(1), denom);
        let tower = build_tower(&expr, x_id("tw_rat_x"));
        assert_eq!(tower.kind, TowerKind::Rational);
        assert_eq!(tower.extension_depth(), 0);
    }

    #[test]
    fn test_tower_single_log() {
        let x = x_sym("tw_log_x");
        let ln_x = Expr::func(FuncId::Ln, vec![x]);
        let tower = build_tower(&ln_x, x_id("tw_log_x"));
        assert_eq!(tower.kind, TowerKind::Logarithmic);
        assert_eq!(tower.extension_depth(), 1);
        assert!(tower.has_log());
        assert!(!tower.has_exp());
    }

    #[test]
    fn test_tower_single_exp() {
        let x = x_sym("tw_exp_x");
        let exp_x = Expr::func(FuncId::Exp, vec![x]);
        let tower = build_tower(&exp_x, x_id("tw_exp_x"));
        assert_eq!(tower.kind, TowerKind::Exponential);
        assert_eq!(tower.extension_depth(), 1);
        assert!(!tower.has_log());
        assert!(tower.has_exp());
    }

    #[test]
    fn test_tower_mixed() {
        let x = x_sym("tw_mixed_x");
        let ln_x = Expr::func(FuncId::Ln, vec![x.clone()]);
        let exp_x = Expr::func(FuncId::Exp, vec![x.clone()]);
        let expr = normalize::mul(ln_x, exp_x);
        let tower = build_tower(&expr, x_id("tw_mixed_x"));
        assert_eq!(tower.kind, TowerKind::Mixed);
        assert!(tower.has_log());
        assert!(tower.has_exp());
    }

    #[test]
    fn test_tower_deduplicates_log() {
        // ln(x) + ln(x) — only one generator
        let x = x_sym("tw_dedup_x");
        let ln_x = Expr::func(FuncId::Ln, vec![x.clone()]);
        let expr = normalize::add(ln_x.clone(), ln_x);
        let tower = build_tower(&expr, x_id("tw_dedup_x"));
        assert_eq!(tower.extension_depth(), 1);
    }

    #[test]
    fn test_tower_nested_exp_in_arg() {
        // exp(exp(x)) — two exp extensions with different arguments
        let x = x_sym("tw_nest_x");
        let exp_x = Expr::func(FuncId::Exp, vec![x]);
        let exp_exp_x = Expr::func(FuncId::Exp, vec![exp_x]);
        let tower = build_tower(&exp_exp_x, x_id("tw_nest_x"));
        assert_eq!(tower.kind, TowerKind::Exponential);
        assert!(tower.extension_depth() >= 1);
    }

    #[test]
    fn test_tower_levels_start_with_base() {
        let x = x_sym("tw_base_x");
        let xid = x_id("tw_base_x");
        let expr = Expr::func(FuncId::Ln, vec![x]);
        let tower = build_tower(&expr, xid);
        assert!(matches!(tower.levels[0], TowerLevel::Base(id) if id == xid));
    }

    #[test]
    fn test_tower_level_predicates() {
        let x = x_sym("tw_pred_x");
        assert!(!TowerLevel::Base(x_id("tw_pred_x")).is_log());
        assert!(!TowerLevel::Base(x_id("tw_pred_x")).is_exp());
        assert!(TowerLevel::Log(x.clone()).is_log());
        assert!(!TowerLevel::Log(x.clone()).is_exp());
        assert!(TowerLevel::Exp(x.clone()).is_exp());
        assert!(!TowerLevel::Exp(x).is_log());
    }
}
