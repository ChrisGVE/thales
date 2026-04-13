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
//! - [`euler_rules`] — Euler's formula: e^(i·x)→cos(x)+i·sin(x), |e^(i·x)|→1
//! - [`all_simplification_rules`] — all rules combined in precedence order

use crate::ast::{BinaryOp, Expression, Function, SymbolicConstant, UnaryOp};
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
///
/// Only applies when `x` is a real numeric constant. For symbolic `x`,
/// the identity requires knowing `x` is real (not complex), which needs
/// domain assumptions not yet available.
fn ln_of_exp_rule() -> Rule {
    Rule::with_condition(
        Pattern::function(
            Function::Ln,
            vec![Pattern::power(
                Pattern::exact(Expression::Constant(SymbolicConstant::E)),
                Pattern::wildcard("x"),
            )],
        ),
        Pattern::wildcard("x"),
        |bindings| {
            bindings
                .get("x")
                .is_some_and(Expression::is_numeric_constant)
        },
    )
    .named("ln_of_exp")
}

/// e^(ln(x)) → x
///
/// Only applies when `x` is a positive numeric constant. For symbolic `x`,
/// requires knowing `x > 0`, which needs domain assumptions.
fn exp_of_ln_rule() -> Rule {
    Rule::with_condition(
        Pattern::power(
            Pattern::exact(Expression::Constant(SymbolicConstant::E)),
            Pattern::function(Function::Ln, vec![Pattern::wildcard("x")]),
        ),
        Pattern::wildcard("x"),
        |bindings| {
            bindings
                .get("x")
                .is_some_and(|x| Expression::extract_numeric_value(x).is_some_and(|v| v > 0.0))
        },
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

// ── Euler's formula rules ─────────────────────────────────────────────────────

/// Rules derived from Euler's formula: e^(i·x) = cos(x) + i·sin(x).
///
/// - e^(i·x) → cos(x) + i·sin(x)
/// - |e^(i·x)| → 1
pub fn euler_rules() -> Vec<Rule> {
    vec![euler_magnitude_rule(), euler_expansion_rule()]
}

/// e^(i·x) → cos(x) + i·sin(x)
///
/// Matches `e^(i*x)` where `e` is `Expression::Constant(SymbolicConstant::E)`
/// and `i` is `Expression::Constant(SymbolicConstant::I)`.  The multiplication
/// `i*x` is matched commutatively, so `x*i` also triggers the rule.
fn euler_expansion_rule() -> Rule {
    let i = Expression::Constant(SymbolicConstant::I);
    let e = Expression::Constant(SymbolicConstant::E);

    // LHS: e^(i * x)
    let lhs = Pattern::power(
        Pattern::exact(e),
        Pattern::binary(
            BinaryOp::Mul,
            Pattern::exact(i.clone()),
            Pattern::wildcard("x"),
        ),
    );

    // RHS: cos(x) + i * sin(x)
    let cos_x = Pattern::function(Function::Cos, vec![Pattern::wildcard("x")]);
    let sin_x = Pattern::function(Function::Sin, vec![Pattern::wildcard("x")]);
    let i_sin_x = Pattern::binary(BinaryOp::Mul, Pattern::exact(i), sin_x);
    let rhs = Pattern::binary(BinaryOp::Add, cos_x, i_sin_x);

    Rule::new(lhs, rhs).named("euler_expansion")
}

/// |e^(i·x)| → 1
///
/// The magnitude of any point on the complex unit circle is 1.
fn euler_magnitude_rule() -> Rule {
    let i = Expression::Constant(SymbolicConstant::I);
    let e = Expression::Constant(SymbolicConstant::E);

    // LHS: |e^(i * x)|
    let inner = Pattern::power(
        Pattern::exact(e),
        Pattern::binary(BinaryOp::Mul, Pattern::exact(i), Pattern::wildcard("x")),
    );
    let lhs = Pattern::unary(UnaryOp::Abs, inner);

    Rule::new(lhs, Pattern::exact(Expression::Integer(1))).named("euler_magnitude")
}

// ── Combined rule set ─────────────────────────────────────────────────────────

/// Return all simplification rules in application-priority order.
///
/// The order matters: more specific rules (e.g., `ln_of_exp`) should come
/// before broader ones so that `apply_rules_to_fixpoint` does not mask them.
///
/// Order:
/// 1. Pythagorean identity (most specific — prevents later confusion)
/// 2. Euler's formula (before log/exp to avoid base confusion)
/// 3. Log/exp inverses
/// 4. Arithmetic cancellation (sub_self, div_self)
/// 5. Common algebraic rules (additive/multiplicative identity/zero, double-neg)
/// 6. Power rules (x^0, x^1)
pub fn all_simplification_rules() -> Vec<Rule> {
    use crate::pattern::common_rules;

    let mut rules = Vec::new();
    rules.extend(trig_pythagorean_rule());
    rules.extend(euler_rules());
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
    fn test_ln_of_exp_numeric() {
        // ln(e^3) → 3 — numeric exponent, safe to simplify
        let rule = ln_of_exp_rule();
        let e = Expression::Constant(SymbolicConstant::E);
        let expr = Expression::Function(
            Function::Ln,
            vec![Expression::Power(Box::new(e), Box::new(int(3)))],
        );
        assert_eq!(apply_rule(&expr, &rule), Some(int(3)));
    }

    #[test]
    fn test_ln_of_exp_symbolic_blocked() {
        // ln(e^x) must NOT simplify when x is symbolic (could be complex)
        let rule = ln_of_exp_rule();
        let e = Expression::Constant(SymbolicConstant::E);
        let expr = Expression::Function(
            Function::Ln,
            vec![Expression::Power(Box::new(e), Box::new(var("x")))],
        );
        assert_eq!(apply_rule(&expr, &rule), None);
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
    fn test_exp_of_ln_positive_numeric() {
        // e^(ln(5)) → 5 — positive constant, safe to simplify
        let rule = exp_of_ln_rule();
        let e = Expression::Constant(SymbolicConstant::E);
        let expr = Expression::Power(
            Box::new(e),
            Box::new(Expression::Function(Function::Ln, vec![int(5)])),
        );
        assert_eq!(apply_rule(&expr, &rule), Some(int(5)));
    }

    #[test]
    fn test_exp_of_ln_symbolic_blocked() {
        // e^(ln(x)) must NOT simplify when x is symbolic (could be ≤ 0)
        let rule = exp_of_ln_rule();
        let e = Expression::Constant(SymbolicConstant::E);
        let expr = Expression::Power(
            Box::new(e),
            Box::new(Expression::Function(Function::Ln, vec![var("x")])),
        );
        assert_eq!(apply_rule(&expr, &rule), None);
    }

    #[test]
    fn test_exp_of_ln_negative_blocked() {
        // e^(ln(-3)) must NOT simplify — ln(-3) is undefined for reals
        let rule = exp_of_ln_rule();
        let e = Expression::Constant(SymbolicConstant::E);
        let expr = Expression::Power(
            Box::new(e),
            Box::new(Expression::Function(
                Function::Ln,
                vec![Expression::Integer(-3)],
            )),
        );
        assert_eq!(apply_rule(&expr, &rule), None);
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

    // ── euler_expansion ───────────────────────────────────────────────────────

    fn e_pow_ix(x: Expression) -> Expression {
        let e = Expression::Constant(SymbolicConstant::E);
        let i = Expression::Constant(SymbolicConstant::I);
        let i_times_x = Expression::Binary(BinaryOp::Mul, Box::new(i), Box::new(x));
        Expression::Power(Box::new(e), Box::new(i_times_x))
    }

    fn euler_expected(x: Expression) -> Expression {
        let i = Expression::Constant(SymbolicConstant::I);
        let cos_x = Expression::Function(Function::Cos, vec![x.clone()]);
        let sin_x = Expression::Function(Function::Sin, vec![x]);
        let i_sin_x = Expression::Binary(BinaryOp::Mul, Box::new(i), Box::new(sin_x));
        Expression::Binary(BinaryOp::Add, Box::new(cos_x), Box::new(i_sin_x))
    }

    #[test]
    fn test_euler_expansion_variable() {
        let rule = euler_expansion_rule();
        let expr = e_pow_ix(var("x"));
        assert_eq!(apply_rule(&expr, &rule), Some(euler_expected(var("x"))));
    }

    #[test]
    fn test_euler_expansion_commutative_mul() {
        // x*i (reversed) should also match due to commutative Mul matching
        let rule = euler_expansion_rule();
        let e = Expression::Constant(SymbolicConstant::E);
        let i = Expression::Constant(SymbolicConstant::I);
        let x_times_i = Expression::Binary(BinaryOp::Mul, Box::new(var("x")), Box::new(i));
        let expr = Expression::Power(Box::new(e), Box::new(x_times_i));
        assert_eq!(apply_rule(&expr, &rule), Some(euler_expected(var("x"))));
    }

    #[test]
    fn test_euler_expansion_does_not_match_real_exp() {
        // e^x (no imaginary unit) must not match
        let rule = euler_expansion_rule();
        let e = Expression::Constant(SymbolicConstant::E);
        let expr = Expression::Power(Box::new(e), Box::new(var("x")));
        assert_eq!(apply_rule(&expr, &rule), None);
    }

    #[test]
    fn test_euler_expansion_does_not_match_wrong_base() {
        // 2^(i*x) must not match — base must be e
        let rule = euler_expansion_rule();
        let i = Expression::Constant(SymbolicConstant::I);
        let i_times_x = Expression::Binary(BinaryOp::Mul, Box::new(i), Box::new(var("x")));
        let expr = Expression::Power(Box::new(int(2)), Box::new(i_times_x));
        assert_eq!(apply_rule(&expr, &rule), None);
    }

    // ── euler_magnitude ───────────────────────────────────────────────────────

    #[test]
    fn test_euler_magnitude_variable() {
        use crate::ast::UnaryOp;
        let rule = euler_magnitude_rule();
        let expr = Expression::Unary(UnaryOp::Abs, Box::new(e_pow_ix(var("x"))));
        assert_eq!(apply_rule(&expr, &rule), Some(int(1)));
    }

    #[test]
    fn test_euler_magnitude_does_not_match_real_exp() {
        use crate::ast::UnaryOp;
        let rule = euler_magnitude_rule();
        let e = Expression::Constant(SymbolicConstant::E);
        let real_exp = Expression::Power(Box::new(e), Box::new(var("x")));
        let expr = Expression::Unary(UnaryOp::Abs, Box::new(real_exp));
        assert_eq!(apply_rule(&expr, &rule), None);
    }

    // ── euler rules via fixpoint ──────────────────────────────────────────────

    #[test]
    fn test_fixpoint_euler_expansion() {
        let rules = all_simplification_rules();
        let expr = e_pow_ix(var("t"));
        assert_eq!(
            apply_rules_to_fixpoint(&expr, &rules, 20),
            euler_expected(var("t"))
        );
    }

    #[test]
    fn test_fixpoint_euler_magnitude() {
        use crate::ast::UnaryOp;
        let rules = all_simplification_rules();
        let expr = Expression::Unary(UnaryOp::Abs, Box::new(e_pow_ix(var("t"))));
        assert_eq!(apply_rules_to_fixpoint(&expr, &rules, 20), int(1));
    }
}
