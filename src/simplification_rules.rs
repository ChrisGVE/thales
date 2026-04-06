//! Standard algebraic simplification rules for the pattern-matching pipeline.
//!
//! This module provides rule sets used by [`Expression::simplify`] as a
//! pattern-matching pass that runs after the structural simplification phase.
//!
//! # Rule Sets
//!
//! - [`arithmetic_rules`] — identity, annihilator, cancellation rules
//! - [`power_rules`] — power/exponent simplification
//! - [`log_exp_rules`] — logarithm/exponential inverse rules
//! - [`trig_pythagorean_rule`] — sin²(x)+cos²(x)→1
//! - [`all_simplification_rules`] — all rules combined in precedence order

use crate::ast::{Expression, Function, SymbolicConstant};
use crate::pattern::{Pattern, Rule};

// ── Arithmetic rules ──────────────────────────────────────────────────────────

/// Rules that handle cancellation under subtraction and division.
///
/// - x - x → 0
/// - x / x → 1  (no domain guard; the caller must not pass a zero dividend)
pub fn arithmetic_rules() -> Vec<Rule> {
    vec![sub_self_rule(), div_self_rule()]
}

/// x - x → 0
fn sub_self_rule() -> Rule {
    use crate::ast::BinaryOp;
    Rule::new(
        Pattern::binary(
            BinaryOp::Sub,
            Pattern::wildcard("x"),
            Pattern::wildcard("x"),
        ),
        Pattern::exact(Expression::Integer(0)),
    )
    .named("sub_self")
}

/// x / x → 1
fn div_self_rule() -> Rule {
    use crate::ast::BinaryOp;
    Rule::new(
        Pattern::binary(
            BinaryOp::Div,
            Pattern::wildcard("x"),
            Pattern::wildcard("x"),
        ),
        Pattern::exact(Expression::Integer(1)),
    )
    .named("div_self")
}

// ── Power rules ───────────────────────────────────────────────────────────────

/// Rules for power/exponent simplification (x^0, x^1).
///
/// These duplicate those in `pattern::common_rules` so that the unified
/// `all_simplification_rules()` list is self-contained.
pub fn power_rules() -> Vec<Rule> {
    use crate::pattern::common_rules;
    vec![common_rules::power_zero(), common_rules::power_one()]
}

// ── Log / Exp inverse rules ───────────────────────────────────────────────────

/// Rules relating logarithm and exponential as inverses.
///
/// - ln(e^x) → x
/// - e^(ln(x)) → x
pub fn log_exp_rules() -> Vec<Rule> {
    vec![ln_of_exp_rule(), exp_of_ln_rule()]
}

/// ln(e^x) → x
///
/// Matches `Function::Ln` applied to `e^x` (where `e` is
/// `Expression::Constant(SymbolicConstant::E)`).
fn ln_of_exp_rule() -> Rule {
    Rule::new(
        Pattern::function(
            Function::Ln,
            vec![Pattern::power(
                Pattern::exact(Expression::Constant(SymbolicConstant::E)),
                Pattern::wildcard("x"),
            )],
        ),
        Pattern::wildcard("x"),
    )
    .named("ln_of_exp")
}

/// e^(ln(x)) → x
fn exp_of_ln_rule() -> Rule {
    Rule::new(
        Pattern::power(
            Pattern::exact(Expression::Constant(SymbolicConstant::E)),
            Pattern::function(Function::Ln, vec![Pattern::wildcard("x")]),
        ),
        Pattern::wildcard("x"),
    )
    .named("exp_of_ln")
}

// ── Trigonometric Pythagorean identity ────────────────────────────────────────

/// sin²(x) + cos²(x) → 1
///
/// Returns a single-element `Vec` for a consistent calling convention
/// with the other rule-set functions.
pub fn trig_pythagorean_rule() -> Vec<Rule> {
    vec![sin_sq_plus_cos_sq_rule()]
}

/// Builds the sin²(x) + cos²(x) → 1 pattern rule.
fn sin_sq_plus_cos_sq_rule() -> Rule {
    use crate::ast::BinaryOp;
    let sin_sq = Pattern::power(
        Pattern::function(Function::Sin, vec![Pattern::wildcard("x")]),
        Pattern::exact(Expression::Integer(2)),
    );
    let cos_sq = Pattern::power(
        Pattern::function(Function::Cos, vec![Pattern::wildcard("x")]),
        Pattern::exact(Expression::Integer(2)),
    );
    Rule::new(
        Pattern::binary(BinaryOp::Add, sin_sq, cos_sq),
        Pattern::exact(Expression::Integer(1)),
    )
    .named("sin_sq_plus_cos_sq")
}

// ── Combined rule set ─────────────────────────────────────────────────────────

/// Return all simplification rules in application-priority order.
///
/// The order matters: more specific rules (e.g., `ln_of_exp`) should come
/// before broader ones so that `apply_rules_to_fixpoint` does not mask them.
///
/// Order:
/// 1. Pythagorean identity (most specific — prevents later confusion)
/// 2. Log/exp inverses
/// 3. Arithmetic cancellation (sub_self, div_self)
/// 4. Common algebraic rules (additive/multiplicative identity/zero, double-neg)
/// 5. Power rules (x^0, x^1)
pub fn all_simplification_rules() -> Vec<Rule> {
    use crate::pattern::common_rules;

    let mut rules = Vec::new();
    rules.extend(trig_pythagorean_rule());
    rules.extend(log_exp_rules());
    rules.extend(arithmetic_rules());
    rules.extend(common_rules::all());
    rules
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Variable};
    use crate::pattern::{apply_rule, apply_rules_to_fixpoint};

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    // ── sub_self ──────────────────────────────────────────────────────────────

    #[test]
    fn test_sub_self_variable() {
        let rule = sub_self_rule();
        let expr = Expression::Binary(BinaryOp::Sub, Box::new(var("x")), Box::new(var("x")));
        assert_eq!(apply_rule(&expr, &rule), Some(int(0)));
    }

    #[test]
    fn test_sub_self_does_not_match_different() {
        let rule = sub_self_rule();
        let expr = Expression::Binary(BinaryOp::Sub, Box::new(var("x")), Box::new(var("y")));
        assert_eq!(apply_rule(&expr, &rule), None);
    }

    // ── div_self ──────────────────────────────────────────────────────────────

    #[test]
    fn test_div_self_variable() {
        let rule = div_self_rule();
        let expr = Expression::Binary(BinaryOp::Div, Box::new(var("x")), Box::new(var("x")));
        assert_eq!(apply_rule(&expr, &rule), Some(int(1)));
    }

    #[test]
    fn test_div_self_complex_expr() {
        let rule = div_self_rule();
        let subexpr = Expression::Binary(BinaryOp::Add, Box::new(var("x")), Box::new(int(1)));
        let expr = Expression::Binary(BinaryOp::Div, Box::new(subexpr.clone()), Box::new(subexpr));
        assert_eq!(apply_rule(&expr, &rule), Some(int(1)));
    }

    // ── ln_of_exp ─────────────────────────────────────────────────────────────

    #[test]
    fn test_ln_of_exp() {
        let rule = ln_of_exp_rule();
        let e = Expression::Constant(SymbolicConstant::E);
        let expr = Expression::Function(
            Function::Ln,
            vec![Expression::Power(Box::new(e), Box::new(var("x")))],
        );
        assert_eq!(apply_rule(&expr, &rule), Some(var("x")));
    }

    #[test]
    fn test_ln_of_exp_does_not_match_wrong_base() {
        let rule = ln_of_exp_rule();
        // ln(2^x) should not match
        let expr = Expression::Function(
            Function::Ln,
            vec![Expression::Power(Box::new(int(2)), Box::new(var("x")))],
        );
        assert_eq!(apply_rule(&expr, &rule), None);
    }

    // ── exp_of_ln ─────────────────────────────────────────────────────────────

    #[test]
    fn test_exp_of_ln() {
        let rule = exp_of_ln_rule();
        let e = Expression::Constant(SymbolicConstant::E);
        let expr = Expression::Power(
            Box::new(e),
            Box::new(Expression::Function(Function::Ln, vec![var("x")])),
        );
        assert_eq!(apply_rule(&expr, &rule), Some(var("x")));
    }

    #[test]
    fn test_exp_of_ln_does_not_match_wrong_base() {
        let rule = exp_of_ln_rule();
        // 2^(ln(x)) should not match
        let expr = Expression::Power(
            Box::new(int(2)),
            Box::new(Expression::Function(Function::Ln, vec![var("x")])),
        );
        assert_eq!(apply_rule(&expr, &rule), None);
    }

    // ── sin²+cos² ─────────────────────────────────────────────────────────────

    #[test]
    fn test_sin_sq_plus_cos_sq() {
        let rule = sin_sq_plus_cos_sq_rule();
        let sin_sq = Expression::Power(
            Box::new(Expression::Function(Function::Sin, vec![var("x")])),
            Box::new(int(2)),
        );
        let cos_sq = Expression::Power(
            Box::new(Expression::Function(Function::Cos, vec![var("x")])),
            Box::new(int(2)),
        );
        let expr = Expression::Binary(BinaryOp::Add, Box::new(sin_sq), Box::new(cos_sq));
        assert_eq!(apply_rule(&expr, &rule), Some(int(1)));
    }

    #[test]
    fn test_sin_sq_plus_cos_sq_commutative() {
        // cos²(x) + sin²(x) should also match (commutative)
        let rule = sin_sq_plus_cos_sq_rule();
        let sin_sq = Expression::Power(
            Box::new(Expression::Function(Function::Sin, vec![var("x")])),
            Box::new(int(2)),
        );
        let cos_sq = Expression::Power(
            Box::new(Expression::Function(Function::Cos, vec![var("x")])),
            Box::new(int(2)),
        );
        // reversed order
        let expr = Expression::Binary(BinaryOp::Add, Box::new(cos_sq), Box::new(sin_sq));
        assert_eq!(apply_rule(&expr, &rule), Some(int(1)));
    }

    #[test]
    fn test_sin_sq_plus_cos_sq_different_args_no_match() {
        // sin²(x) + cos²(y) — different arguments, must not match
        let rule = sin_sq_plus_cos_sq_rule();
        let sin_sq = Expression::Power(
            Box::new(Expression::Function(Function::Sin, vec![var("x")])),
            Box::new(int(2)),
        );
        let cos_sq = Expression::Power(
            Box::new(Expression::Function(Function::Cos, vec![var("y")])),
            Box::new(int(2)),
        );
        let expr = Expression::Binary(BinaryOp::Add, Box::new(sin_sq), Box::new(cos_sq));
        assert_eq!(apply_rule(&expr, &rule), None);
    }

    // ── all_simplification_rules via fixpoint ─────────────────────────────────

    #[test]
    fn test_fixpoint_x_plus_0() {
        let rules = all_simplification_rules();
        let expr = Expression::Binary(BinaryOp::Add, Box::new(var("x")), Box::new(int(0)));
        assert_eq!(apply_rules_to_fixpoint(&expr, &rules, 20), var("x"));
    }

    #[test]
    fn test_fixpoint_x_times_1() {
        let rules = all_simplification_rules();
        let expr = Expression::Binary(BinaryOp::Mul, Box::new(var("x")), Box::new(int(1)));
        assert_eq!(apply_rules_to_fixpoint(&expr, &rules, 20), var("x"));
    }

    #[test]
    fn test_fixpoint_x_times_0() {
        let rules = all_simplification_rules();
        let expr = Expression::Binary(BinaryOp::Mul, Box::new(var("x")), Box::new(int(0)));
        assert_eq!(apply_rules_to_fixpoint(&expr, &rules, 20), int(0));
    }

    #[test]
    fn test_fixpoint_double_neg() {
        use crate::ast::UnaryOp;
        let rules = all_simplification_rules();
        let expr = Expression::Unary(
            UnaryOp::Neg,
            Box::new(Expression::Unary(UnaryOp::Neg, Box::new(var("x")))),
        );
        assert_eq!(apply_rules_to_fixpoint(&expr, &rules, 20), var("x"));
    }

    #[test]
    fn test_fixpoint_x_pow_1() {
        let rules = all_simplification_rules();
        let expr = Expression::Power(Box::new(var("x")), Box::new(int(1)));
        assert_eq!(apply_rules_to_fixpoint(&expr, &rules, 20), var("x"));
    }

    #[test]
    fn test_fixpoint_x_pow_0() {
        let rules = all_simplification_rules();
        let expr = Expression::Power(Box::new(var("x")), Box::new(int(0)));
        assert_eq!(apply_rules_to_fixpoint(&expr, &rules, 20), int(1));
    }
}
