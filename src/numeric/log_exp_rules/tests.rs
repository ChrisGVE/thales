#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::numeric::expr::{Expr, FuncId};
    use crate::numeric::log_exp_rules::{
        contract_log, exp_contract, exp_expand, expand_log, log_exp_cancel,
        rule_coeff_log_to_power, rule_exp_log_cancel, rule_exp_of_sum, rule_exp_product_to_sum,
        rule_exp_zero, rule_log_exp_cancel, rule_log_of_power, rule_log_of_product, rule_log_one,
        rule_log_sum_to_product,
    };
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
