//! Configuration for numerical fallback behaviour.
//!
//! [`FallbackConfig`] controls whether and how the engine falls back to
//! numerical evaluation when symbolic strategies are exhausted.
//!
//! # Free function
//!
//! [`node_count`] performs an iterative node-count over an `Arc<Expr>` tree
//! without recursion, suitable for cheap complexity gating.

use std::sync::Arc;

use crate::numeric::expr::Expr;

// ── FallbackConfig ────────────────────────────────────────────────────────────

/// Controls numerical-fallback behaviour for a D0 engine run.
///
/// Build one with [`FallbackConfig::disabled`] (the default) or
/// [`FallbackConfig::enabled`] / [`FallbackConfig::enabled_silent`].
/// Pass it into [`crate::engine::context::SolveContext`] via
/// [`crate::engine::context::SolveContext::with_fallback`].
#[derive(Debug, Clone, PartialEq)]
pub struct FallbackConfig {
    /// Allow numerical evaluation as a last resort.
    ///
    /// Default: `false`.
    pub numerical: bool,
    /// Emit narrative steps that describe numerical computation.
    ///
    /// Only consulted when `numerical` is `true`. Default: `true`.
    pub numerical_narrative: bool,
    /// If set, only fall back when the expression tree has at least this
    /// many nodes. `None` means no complexity gate (always attempt when
    /// `numerical` is `true`).
    ///
    /// Default: `None`.
    pub complexity_threshold: Option<usize>,
}

impl FallbackConfig {
    /// Build a config that disables numerical fallback entirely (the default).
    #[must_use]
    pub fn disabled() -> Self {
        FallbackConfig {
            numerical: false,
            numerical_narrative: true,
            complexity_threshold: None,
        }
    }

    /// Build a config that enables numerical fallback with narrative output.
    #[must_use]
    pub fn enabled() -> Self {
        FallbackConfig {
            numerical: true,
            numerical_narrative: true,
            complexity_threshold: None,
        }
    }

    /// Build a config that enables numerical fallback but suppresses narrative.
    #[must_use]
    pub fn enabled_silent() -> Self {
        FallbackConfig {
            numerical: true,
            numerical_narrative: false,
            complexity_threshold: None,
        }
    }
}

impl Default for FallbackConfig {
    fn default() -> Self {
        FallbackConfig::disabled()
    }
}

// ── node_count ────────────────────────────────────────────────────────────────

/// Count the total number of [`Expr`] nodes in the tree rooted at `expr`.
///
/// The traversal is iterative (stack-based), so it is safe for deeply nested
/// expressions that would otherwise overflow the call stack.
/// Every [`Expr`] node, including leaf nodes, contributes 1 to the total.
#[must_use]
pub fn node_count(expr: &Arc<Expr>) -> usize {
    let mut count: usize = 0;
    let mut stack: Vec<Arc<Expr>> = vec![Arc::clone(expr)];
    while let Some(current) = stack.pop() {
        count += 1;
        match current.as_ref() {
            Expr::Func(_, args) => {
                for arg in args {
                    stack.push(Arc::clone(arg));
                }
            }
            Expr::Pow(base, exp) => {
                stack.push(Arc::clone(base));
                stack.push(Arc::clone(exp));
            }
            Expr::Add(add_node) => {
                for child in add_node.terms.keys() {
                    stack.push(Arc::clone(child));
                }
            }
            Expr::Mul(mul_node) => {
                for (base, exp) in &mul_node.factors {
                    stack.push(Arc::clone(base));
                    stack.push(Arc::clone(exp));
                }
            }
            // Leaf nodes: Integer, Rational, Float, Complex, Constant, Symbol
            Expr::Integer(_)
            | Expr::Rational(_)
            | Expr::Float(_)
            | Expr::Complex(_)
            | Expr::Constant(_)
            | Expr::Symbol(_) => {}
        }
    }
    count
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{Expr, SmallInt};

    fn int_expr(n: i64) -> Arc<Expr> {
        Arc::new(Expr::Integer(SmallInt::from(n)))
    }

    // ── FallbackConfig tests ──────────────────────────────────────────────────

    #[test]
    fn fast_fallback_config_default_is_disabled() {
        let cfg = FallbackConfig::default();
        assert!(!cfg.numerical);
        assert!(cfg.numerical_narrative);
        assert!(cfg.complexity_threshold.is_none());
    }

    #[test]
    fn fast_fallback_config_disabled_constructor() {
        let cfg = FallbackConfig::disabled();
        assert!(!cfg.numerical);
        assert!(cfg.numerical_narrative);
        assert!(cfg.complexity_threshold.is_none());
    }

    #[test]
    fn fast_fallback_config_enabled_constructor() {
        let cfg = FallbackConfig::enabled();
        assert!(cfg.numerical);
        assert!(cfg.numerical_narrative);
        assert!(cfg.complexity_threshold.is_none());
    }

    #[test]
    fn fast_fallback_config_enabled_silent_suppresses_narrative() {
        let cfg = FallbackConfig::enabled_silent();
        assert!(cfg.numerical);
        assert!(!cfg.numerical_narrative);
        assert!(cfg.complexity_threshold.is_none());
    }

    #[test]
    fn fast_fallback_config_partialeq() {
        assert_eq!(FallbackConfig::disabled(), FallbackConfig::disabled());
        assert_ne!(FallbackConfig::disabled(), FallbackConfig::enabled());
        assert_ne!(FallbackConfig::enabled(), FallbackConfig::enabled_silent());
    }

    #[test]
    fn fast_fallback_config_clone() {
        let cfg = FallbackConfig::enabled();
        let cloned = cfg.clone();
        assert_eq!(cfg, cloned);
    }

    #[test]
    fn fast_fallback_config_threshold_field() {
        let cfg = FallbackConfig {
            complexity_threshold: Some(100),
            ..FallbackConfig::enabled()
        };
        assert_eq!(cfg.complexity_threshold, Some(100));
    }

    // ── node_count tests ──────────────────────────────────────────────────────

    #[test]
    fn fast_node_count_leaf_is_one() {
        let expr = int_expr(42);
        assert_eq!(node_count(&expr), 1);
    }

    #[test]
    fn fast_node_count_pow_two_children() {
        let base = int_expr(2);
        let exp = int_expr(3);
        let pow = Arc::new(Expr::Pow(base, exp));
        // Pow node (1) + base (1) + exp (1) = 3
        assert_eq!(node_count(&pow), 3);
    }

    #[test]
    fn fast_node_count_func_single_arg() {
        use crate::numeric::expr::FuncId;
        let arg = int_expr(1);
        let func = Arc::new(Expr::Func(FuncId::Sin, vec![arg]));
        // Func (1) + arg (1) = 2
        assert_eq!(node_count(&func), 2);
    }

    #[test]
    fn fast_node_count_func_multiple_args() {
        use crate::numeric::expr::FuncId;
        let a = int_expr(1);
        let b = int_expr(2);
        let c = int_expr(3);
        let func = Arc::new(Expr::Func(FuncId::Min, vec![a, b, c]));
        // Func (1) + 3 leaf args = 4
        assert_eq!(node_count(&func), 4);
    }

    #[test]
    fn fast_node_count_func_no_args() {
        use crate::numeric::expr::FuncId;
        let func = Arc::new(Expr::Func(FuncId::Sin, vec![]));
        // Func node only
        assert_eq!(node_count(&func), 1);
    }

    #[test]
    fn fast_node_count_nested_pow() {
        // ((2^3)^4): outer Pow → inner Pow → 2, 3, and outer exp 4 = 5 nodes
        let two = int_expr(2);
        let three = int_expr(3);
        let inner = Arc::new(Expr::Pow(two, three));
        let four = int_expr(4);
        let outer = Arc::new(Expr::Pow(inner, four));
        assert_eq!(node_count(&outer), 5);
    }
}
