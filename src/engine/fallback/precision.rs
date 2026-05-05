//! Precision levels and attempt outcomes for numerical fallback.
//!
//! The engine tries precision levels in [`CHAIN`] order, escalating
//! to higher precision only when a lower level proves insufficient.
//!
//! [`PrecisionAttemptOutcome`] captures the result of one attempt at a
//! particular level; [`NumericalResult`] carries a successful evaluation.

use std::sync::Arc;

use crate::engine::trace_tree::StrategyId;
use crate::numeric::Expr;

// ── PrecisionLevel ────────────────────────────────────────────────────────────

/// The numeric precision tier to use when evaluating an expression.
///
/// Variants are ordered from lowest to highest precision; the `Ord`
/// implementation reflects this ordering so callers can compare levels directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrecisionLevel {
    /// IEEE 754 double (≈15–17 significant decimal digits).
    F64 = 0,
    /// IEEE 754 double with interval/bounded arithmetic for error tracking.
    F64Bounded = 1,
    /// 128-bit decimal floating point (≈38 significant digits).
    BigDecimal128 = 2,
    /// 256-bit decimal floating point (≈77 significant digits).
    BigDecimal256 = 3,
    /// 512-bit decimal floating point (≈154 significant digits).
    BigDecimal512 = 4,
}

/// Ordered chain of precision levels tried during numerical fallback.
///
/// The engine iterates this slice from left to right, stopping as soon
/// as one level yields a result of sufficient accuracy.
pub const CHAIN: &[PrecisionLevel] = &[
    PrecisionLevel::F64,
    PrecisionLevel::F64Bounded,
    PrecisionLevel::BigDecimal128,
    PrecisionLevel::BigDecimal256,
    PrecisionLevel::BigDecimal512,
];

impl PrecisionLevel {
    /// Approximate number of significant decimal digits this level provides.
    #[must_use]
    pub fn significant_digits(self) -> u32 {
        match self {
            PrecisionLevel::F64 => 15,
            PrecisionLevel::F64Bounded => 15,
            PrecisionLevel::BigDecimal128 => 38,
            PrecisionLevel::BigDecimal256 => 77,
            PrecisionLevel::BigDecimal512 => 154,
        }
    }

    /// Human-readable label for this precision level.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            PrecisionLevel::F64 => "f64",
            PrecisionLevel::F64Bounded => "f64-bounded",
            PrecisionLevel::BigDecimal128 => "big-decimal-128",
            PrecisionLevel::BigDecimal256 => "big-decimal-256",
            PrecisionLevel::BigDecimal512 => "big-decimal-512",
        }
    }

    /// Returns `true` when this level cannot satisfy `digits_required`.
    ///
    /// Used to skip levels that are known to be insufficient before even
    /// attempting evaluation, saving unnecessary work.
    #[must_use]
    pub fn should_skip_for_precision(self, digits_required: u32) -> bool {
        self.significant_digits() < digits_required
    }
}

// ── NumericalResult ───────────────────────────────────────────────────────────

/// A successful numerical evaluation at a specific precision level.
#[derive(Debug, Clone)]
pub struct NumericalResult {
    /// The computed numerical value as an expression (typically `Expr::Float`
    /// or `Expr::Rational` for exact rational results).
    pub value: Arc<Expr>,
    /// The precision level used to obtain this result.
    pub precision: PrecisionLevel,
    /// Digits of precision actually achieved (may be ≤ `precision.significant_digits()`
    /// for ill-conditioned expressions).
    pub digits_achieved: u32,
    /// Optional error bound: half-width of the certified interval around `value`.
    /// `None` when the evaluator does not provide certified error bounds.
    pub error_bound: Option<Arc<Expr>>,
    /// `true` when the result is approximate rather than exact.
    pub approximate: bool,
    /// `true` when significant precision was lost during evaluation (e.g.
    /// catastrophic cancellation). Consumers should escalate to a higher level.
    pub precision_loss: bool,
    /// Which strategy (evaluator) produced this result.
    pub evaluator_id: StrategyId,
}

// ── PrecisionAttemptOutcome ───────────────────────────────────────────────────

/// Outcome of one numerical evaluation attempt at a specific precision level.
#[derive(Debug)]
pub enum PrecisionAttemptOutcome {
    /// Evaluation succeeded; the result is returned.
    Success(NumericalResult),
    /// The level was not precise enough; partial progress is available.
    InsufficientPrecision {
        /// Best partial result available at this level (may be inaccurate).
        partial: Arc<Expr>,
        /// Digits of precision actually achieved before loss.
        digits_achieved: u32,
        /// Minimum digits required for this problem.
        digits_required: u32,
    },
    /// Evaluation failed for a non-precision reason (e.g. domain error).
    Failed(String),
    /// The resource budget was exhausted before the attempt could complete.
    BudgetExhausted,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::SmallInt;

    fn int_val(n: i64) -> Arc<Expr> {
        Arc::new(Expr::Integer(SmallInt::from(n)))
    }

    fn dummy_result(level: PrecisionLevel) -> NumericalResult {
        NumericalResult {
            value: int_val(0),
            precision: level,
            digits_achieved: level.significant_digits(),
            error_bound: None,
            approximate: true,
            precision_loss: false,
            evaluator_id: StrategyId("test"),
        }
    }

    // ── PrecisionLevel ordering ───────────────────────────────────────────────

    #[test]
    fn fast_precision_chain_order_ascending() {
        let mut prev = CHAIN[0];
        for &level in &CHAIN[1..] {
            assert!(
                level > prev,
                "{:?} should be greater than {:?}",
                level,
                prev
            );
            prev = level;
        }
    }

    #[test]
    fn fast_precision_chain_contains_all_five_levels() {
        assert_eq!(CHAIN.len(), 5);
        assert_eq!(CHAIN[0], PrecisionLevel::F64);
        assert_eq!(CHAIN[4], PrecisionLevel::BigDecimal512);
    }

    #[test]
    fn fast_precision_significant_digits_ascending() {
        let digits: Vec<u32> = CHAIN.iter().map(|l| l.significant_digits()).collect();
        for i in 0..digits.len() - 1 {
            assert!(
                digits[i] <= digits[i + 1],
                "digits[{}]={} > digits[{}]={}",
                i,
                digits[i],
                i + 1,
                digits[i + 1]
            );
        }
    }

    #[test]
    fn fast_precision_significant_digits_values() {
        assert_eq!(PrecisionLevel::F64.significant_digits(), 15);
        assert_eq!(PrecisionLevel::F64Bounded.significant_digits(), 15);
        assert_eq!(PrecisionLevel::BigDecimal128.significant_digits(), 38);
        assert_eq!(PrecisionLevel::BigDecimal256.significant_digits(), 77);
        assert_eq!(PrecisionLevel::BigDecimal512.significant_digits(), 154);
    }

    #[test]
    fn fast_precision_label_unique_and_nonempty() {
        let labels: Vec<&str> = CHAIN.iter().map(|l| l.label()).collect();
        for label in &labels {
            assert!(!label.is_empty());
        }
        // All labels distinct
        let mut sorted = labels.clone();
        sorted.dedup();
        assert_eq!(sorted.len(), labels.len());
    }

    #[test]
    fn fast_precision_should_skip_low_requirement() {
        // F64 provides 15 digits; should not skip for 10-digit requirement
        assert!(!PrecisionLevel::F64.should_skip_for_precision(10));
    }

    #[test]
    fn fast_precision_should_skip_high_requirement() {
        // F64 provides 15 digits; should skip for 50-digit requirement
        assert!(PrecisionLevel::F64.should_skip_for_precision(50));
    }

    #[test]
    fn fast_precision_should_skip_exact_boundary() {
        // Exactly at the digit count: sufficient, do not skip
        let digits = PrecisionLevel::BigDecimal128.significant_digits();
        assert!(!PrecisionLevel::BigDecimal128.should_skip_for_precision(digits));
        // One over: insufficient, skip
        assert!(PrecisionLevel::BigDecimal128.should_skip_for_precision(digits + 1));
    }

    #[test]
    fn fast_precision_ord_cmp() {
        assert!(PrecisionLevel::F64 < PrecisionLevel::BigDecimal512);
        assert!(PrecisionLevel::BigDecimal256 > PrecisionLevel::BigDecimal128);
        assert_eq!(PrecisionLevel::F64Bounded, PrecisionLevel::F64Bounded);
    }

    // ── NumericalResult ───────────────────────────────────────────────────────

    #[test]
    fn fast_precision_numerical_result_fields() {
        let r = dummy_result(PrecisionLevel::F64);
        assert_eq!(r.precision, PrecisionLevel::F64);
        assert_eq!(r.digits_achieved, 15);
        assert!(!r.precision_loss);
        assert!(r.approximate);
        assert!(r.error_bound.is_none());
    }

    #[test]
    fn fast_precision_numerical_result_with_error_bound() {
        let bound = int_val(1);
        let r = NumericalResult {
            value: int_val(42),
            precision: PrecisionLevel::BigDecimal256,
            digits_achieved: 77,
            error_bound: Some(Arc::clone(&bound)),
            approximate: true,
            precision_loss: false,
            evaluator_id: StrategyId("test"),
        };
        assert!(r.error_bound.is_some());
        assert_eq!(r.precision, PrecisionLevel::BigDecimal256);
    }

    // ── PrecisionAttemptOutcome ───────────────────────────────────────────────

    #[test]
    fn fast_precision_outcome_success_variant() {
        let outcome = PrecisionAttemptOutcome::Success(dummy_result(PrecisionLevel::F64));
        assert!(matches!(outcome, PrecisionAttemptOutcome::Success(_)));
    }

    #[test]
    fn fast_precision_outcome_insufficient_precision_variant() {
        let outcome = PrecisionAttemptOutcome::InsufficientPrecision {
            partial: int_val(0),
            digits_achieved: 10,
            digits_required: 50,
        };
        assert!(matches!(
            outcome,
            PrecisionAttemptOutcome::InsufficientPrecision { .. }
        ));
        if let PrecisionAttemptOutcome::InsufficientPrecision {
            digits_achieved,
            digits_required,
            ..
        } = outcome
        {
            assert!(digits_achieved < digits_required);
        }
    }

    #[test]
    fn fast_precision_outcome_failed_variant() {
        let outcome = PrecisionAttemptOutcome::Failed("domain error".to_string());
        assert!(matches!(outcome, PrecisionAttemptOutcome::Failed(_)));
        if let PrecisionAttemptOutcome::Failed(msg) = outcome {
            assert!(!msg.is_empty());
        }
    }

    #[test]
    fn fast_precision_outcome_budget_exhausted_variant() {
        let outcome = PrecisionAttemptOutcome::BudgetExhausted;
        assert!(matches!(outcome, PrecisionAttemptOutcome::BudgetExhausted));
    }
}
