//! Small angle and scaled form approximations for mathematical expressions.
//!
//! This module provides approximation techniques useful for manual calculation and
//! numerical optimization, particularly for slide rule applications and scaled computations.
//!
//! # Key Features
//!
//! - **Small angle approximations**: sin(θ) ≈ θ, cos(θ) ≈ 1 - θ²/2, tan(θ) ≈ θ
//! - **Scaled exponential forms**: e^(-kx) for different scaling factors
//! - **Error bound computation**: Taylor remainder theorem for conservative bounds
//! - **Validity checking**: Ensures approximations are used only within valid ranges
//!
//! # Examples
//!
//! ```rust
//! use thales::approximations::{apply_small_angle_approx, ApproxType, is_approximation_valid};
//! use thales::ast::{Expression, Variable, Function};
//!
//! // Small angle approximation for sin(x) with threshold 0.1
//! let x = Variable::new("x");
//! let sin_x = Expression::Function(Function::Sin, vec![Expression::Variable(x.clone())]);
//!
//! let result = apply_small_angle_approx(&sin_x, &x, 0.1);
//! if let Some(approx) = result {
//!     // Error bound is threshold³/6 ≈ 1.67e-4 for threshold=0.1
//!     assert!(approx.error_bound < 2e-4);
//! }
//!
//! // Check if approximation is valid
//! assert!(is_approximation_valid(&ApproxType::SmallAngleSin, 0.05));
//! assert!(!is_approximation_valid(&ApproxType::SmallAngleSin, 1.0));
//! ```

use crate::ast::{BinaryOp, Expression, Function, UnaryOp, Variable};
use crate::resolution_path::{Operation, ResolutionStep};
use std::collections::HashMap;

/// Result of an approximation with error bounds and validity information.
#[derive(Debug, Clone, PartialEq)]
pub struct ApproxResult {
    /// The approximation expression.
    pub approximation: Expression,

    /// Upper bound on the approximation error (conservative estimate).
    pub error_bound: f64,

    /// Valid range for the approximation (min, max).
    pub valid_range: (f64, f64),

    /// Human-readable description of the approximation formula used.
    pub formula_used: String,
}

/// Types of approximations supported.
#[derive(Debug, Clone, PartialEq)]
pub enum ApproxType {
    /// sin(θ) ≈ θ for small θ
    SmallAngleSin,

    /// cos(θ) ≈ 1 - θ²/2 for small θ
    SmallAngleCos,

    /// tan(θ) ≈ θ for small θ
    SmallAngleTan,

    /// 1 - cos(θ) ≈ θ²/2 for small θ
    SmallAngle1MinusCos,

    /// e^(-kx) with custom scaling factor
    ScaledExp(f64),

    /// sqrt(1-x²) for |x| << 1
    PythagoreanSmall,
}

/// Scaled exponential form selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScaledExpForm {
    /// e^x (standard form)
    Standard,

    /// e^(-0.1x) (scaled by 0.1)
    Scaled01,

    /// e^(-0.01x) (scaled by 0.01)
    Scaled001,

    /// e^(-kx) with custom scaling factor k
    Custom(f64),
}

impl ScaledExpForm {
    /// Get the scaling factor for this form.
    pub fn scaling_factor(&self) -> f64 {
        match self {
            ScaledExpForm::Standard => 1.0,
            ScaledExpForm::Scaled01 => 0.1,
            ScaledExpForm::Scaled001 => 0.01,
            ScaledExpForm::Custom(k) => *k,
        }
    }
}

/// Apply small angle approximation to an expression.
///
/// Detects trigonometric functions with small arguments and replaces them
/// with their Taylor series approximations, providing error bounds.
///
/// # Arguments
///
/// * `expr` - The expression to approximate
/// * `var` - The variable representing the angle
/// * `threshold` - Maximum angle magnitude (in radians) for approximation (default ~0.1)
///
/// # Returns
///
/// `Some(ApproxResult)` if a small angle approximation was applied, `None` otherwise.
///
/// # Examples
///
/// ```rust
/// use thales::approximations::apply_small_angle_approx;
/// use thales::ast::{Expression, Variable, Function};
///
/// let x = Variable::new("x");
/// let sin_x = Expression::Function(Function::Sin, vec![Expression::Variable(x.clone())]);
///
/// let result = apply_small_angle_approx(&sin_x, &x, 0.1);
/// assert!(result.is_some());
/// ```
pub fn apply_small_angle_approx(
    expr: &Expression,
    var: &Variable,
    threshold: f64,
) -> Option<ApproxResult> {
    match expr {
        Expression::Function(func, args) if args.len() == 1 => {
            let arg = &args[0];

            // Check if the argument is just the variable or a simple multiple
            if !is_small_angle_candidate(arg, var) {
                return None;
            }

            match func {
                Function::Sin => {
                    // sin(θ) ≈ θ for small θ
                    // Error bound: |θ³/6| using Taylor remainder
                    Some(ApproxResult {
                        approximation: arg.clone(),
                        error_bound: threshold.powi(3) / 6.0,
                        valid_range: (-threshold, threshold),
                        formula_used: "sin(θ) ≈ θ".to_string(),
                    })
                }
                Function::Cos => {
                    // cos(θ) ≈ 1 - θ²/2 for small θ
                    // Error bound: |θ⁴/24|
                    let theta_squared =
                        Expression::Power(Box::new(arg.clone()), Box::new(Expression::Integer(2)));
                    let term = Expression::Binary(
                        BinaryOp::Div,
                        Box::new(theta_squared),
                        Box::new(Expression::Integer(2)),
                    );
                    let approximation = Expression::Binary(
                        BinaryOp::Sub,
                        Box::new(Expression::Integer(1)),
                        Box::new(term),
                    );

                    Some(ApproxResult {
                        approximation,
                        error_bound: threshold.powi(4) / 24.0,
                        valid_range: (-threshold, threshold),
                        formula_used: "cos(θ) ≈ 1 - θ²/2".to_string(),
                    })
                }
                Function::Tan => {
                    // tan(θ) ≈ θ for small θ
                    // Error bound: |θ³/3| (slightly less accurate than sin)
                    Some(ApproxResult {
                        approximation: arg.clone(),
                        error_bound: threshold.powi(3) / 3.0,
                        valid_range: (-threshold, threshold),
                        formula_used: "tan(θ) ≈ θ".to_string(),
                    })
                }
                _ => None,
            }
        }
        // Handle 1 - cos(θ) ≈ θ²/2
        Expression::Binary(BinaryOp::Sub, left, right) => {
            if let (Expression::Integer(1), Expression::Function(Function::Cos, args)) =
                (left.as_ref(), right.as_ref())
            {
                if args.len() == 1 && is_small_angle_candidate(&args[0], var) {
                    let arg = &args[0];
                    let theta_squared =
                        Expression::Power(Box::new(arg.clone()), Box::new(Expression::Integer(2)));
                    let approximation = Expression::Binary(
                        BinaryOp::Div,
                        Box::new(theta_squared),
                        Box::new(Expression::Integer(2)),
                    );

                    return Some(ApproxResult {
                        approximation,
                        error_bound: threshold.powi(4) / 12.0,
                        valid_range: (-threshold, threshold),
                        formula_used: "1 - cos(θ) ≈ θ²/2".to_string(),
                    });
                }
            }
            None
        }
        _ => None,
    }
}

/// Check if an expression is a candidate for small angle approximation.
fn is_small_angle_candidate(expr: &Expression, var: &Variable) -> bool {
    match expr {
        Expression::Variable(v) => v == var,
        Expression::Unary(UnaryOp::Neg, inner) => is_small_angle_candidate(inner, var),
        Expression::Binary(BinaryOp::Mul, left, right) => {
            // Allow constant * variable
            matches!(left.as_ref(), Expression::Integer(_) | Expression::Float(_))
                && matches!(right.as_ref(), Expression::Variable(v) if v == var)
                || matches!(
                    right.as_ref(),
                    Expression::Integer(_) | Expression::Float(_)
                ) && matches!(left.as_ref(), Expression::Variable(v) if v == var)
        }
        _ => false,
    }
}

/// Compute the approximation error at a specific value.
///
/// Evaluates both the exact expression and the approximation at the given
/// variable value, returning the absolute difference.
///
/// # Arguments
///
/// * `exact` - The exact expression
/// * `approx` - The approximation expression
/// * `var` - The variable to substitute
/// * `value` - The value at which to evaluate
///
/// # Returns
///
/// The absolute error |exact - approx| at the given value.
///
/// # Examples
///
/// ```rust
/// use thales::approximations::compute_approximation_error;
/// use thales::ast::{Expression, Variable, Function};
///
/// let x = Variable::new("x");
/// let exact = Expression::Function(Function::Sin, vec![Expression::Variable(x.clone())]);
/// let approx = Expression::Variable(x.clone());
///
/// // Error for sin(0.05) ≈ 0.05 is about 2.08e-5
/// let error = compute_approximation_error(&exact, &approx, &x, 0.05);
/// assert!(error < 3e-5);
/// ```
pub fn compute_approximation_error(
    exact: &Expression,
    approx: &Expression,
    var: &Variable,
    value: f64,
) -> f64 {
    let mut vars = HashMap::new();
    vars.insert(var.name.clone(), value);

    let exact_val = exact.evaluate(&vars).unwrap_or(0.0);
    let approx_val = approx.evaluate(&vars).unwrap_or(0.0);

    (exact_val - approx_val).abs()
}

/// Select appropriate scaled exponential form based on argument range.
///
/// Chooses the scaling factor that provides best precision for the given
/// argument range, useful for slide rule and logarithm table calculations.
///
/// # Arguments
///
/// * `argument_range` - (min, max) range of the exponential argument
///
/// # Returns
///
/// The recommended `ScaledExpForm` for the given range.
///
/// # Examples
///
/// ```rust
/// use thales::approximations::{select_exp_scaling, ScaledExpForm};
///
/// // For arguments in [0, 10], use standard form
/// let form = select_exp_scaling((0.0, 10.0));
/// assert_eq!(form, ScaledExpForm::Standard);
///
/// // For arguments in [0, 100], use scaled form
/// let form = select_exp_scaling((0.0, 100.0));
/// assert_eq!(form, ScaledExpForm::Scaled01);
/// ```
pub fn select_exp_scaling(argument_range: (f64, f64)) -> ScaledExpForm {
    let (min, max) = argument_range;
    let range_magnitude = (max - min).abs();

    // Select scaling based on range magnitude
    if range_magnitude <= 20.0 {
        ScaledExpForm::Standard
    } else if range_magnitude <= 200.0 {
        ScaledExpForm::Scaled01
    } else {
        ScaledExpForm::Scaled001
    }
}

/// Check if an approximation is valid for a given variable value.
///
/// Determines whether the approximation can be safely used at the specified
/// value, based on the approximation type's validity criteria.
///
/// # Arguments
///
/// * `approx_type` - The type of approximation
/// * `variable_value` - The value at which to check validity
///
/// # Returns
///
/// `true` if the approximation is valid at this value, `false` otherwise.
///
/// # Examples
///
/// ```rust
/// use thales::approximations::{is_approximation_valid, ApproxType};
///
/// // sin(0.05) ≈ 0.05 is valid
/// assert!(is_approximation_valid(&ApproxType::SmallAngleSin, 0.05));
///
/// // sin(1.0) ≈ 1.0 is not valid (angle too large)
/// assert!(!is_approximation_valid(&ApproxType::SmallAngleSin, 1.0));
/// ```
pub fn is_approximation_valid(approx_type: &ApproxType, variable_value: f64) -> bool {
    let abs_value = variable_value.abs();

    match approx_type {
        ApproxType::SmallAngleSin
        | ApproxType::SmallAngleCos
        | ApproxType::SmallAngleTan
        | ApproxType::SmallAngle1MinusCos => {
            // Small angle approximations valid for |θ| < 0.2 radians (~11.5°)
            // Using conservative threshold of 0.2 for safety
            abs_value < 0.2
        }
        ApproxType::ScaledExp(_) => {
            // Scaled exponentials generally valid for reasonable ranges
            // Avoid extreme values that would overflow
            abs_value < 700.0 // e^700 is near float max
        }
        ApproxType::PythagoreanSmall => {
            // sqrt(1-x²) approximation valid for |x| << 1
            abs_value < 0.1
        }
    }
}

/// Generate a resolution step for an approximation.
///
/// Creates a `ResolutionStep` documenting the approximation applied, including
/// the error bound and validity range.
///
/// # Arguments
///
/// * `original` - The original exact expression
/// * `approximation` - The approximation expression
/// * `error_bound` - Conservative upper bound on approximation error
/// * `formula_used` - Description of the approximation formula
///
/// # Returns
///
/// A `ResolutionStep` documenting the approximation.
///
/// # Examples
///
/// ```rust
/// use thales::approximations::generate_approximation_step;
/// use thales::ast::{Expression, Variable, Function};
///
/// let x = Variable::new("x");
/// let original = Expression::Function(Function::Sin, vec![Expression::Variable(x.clone())]);
/// let approx = Expression::Variable(x.clone());
///
/// let step = generate_approximation_step(
///     &original,
///     &approx,
///     1e-5,
///     "sin(θ) ≈ θ".to_string(),
/// );
/// ```
pub fn generate_approximation_step(
    original: &Expression,
    approximation: &Expression,
    error_bound: f64,
    formula_used: String,
) -> ResolutionStep {
    let explanation = format!(
        "Apply approximation: {}. Error bound: {:.2e}",
        formula_used, error_bound
    );

    ResolutionStep::new(
        Operation::ApproximationSubstitution {
            original: original.clone(),
            approximation: approximation.clone(),
            error_bound,
        },
        explanation,
        approximation.clone(),
    )
}

/// Optimize Pythagorean form sqrt(1-x²) based on x magnitude.
///
/// Selects the most numerically stable form:
/// - For |x| << 1: Use series expansion
/// - For x close to 1: Use sqrt(1-x)*sqrt(1+x)
///
/// # Arguments
///
/// * `expr` - Expression of form sqrt(1-x²)
///
/// # Returns
///
/// Optimized form if applicable, or None.
pub fn optimize_pythagorean(expr: &Expression) -> Option<Expression> {
    // Check if expression matches sqrt(1 - x²)
    if let Expression::Function(Function::Sqrt, args) = expr {
        if args.len() != 1 {
            return None;
        }

        if let Expression::Binary(BinaryOp::Sub, left, right) = &args[0] {
            if matches!(left.as_ref(), Expression::Integer(1)) {
                if let Expression::Power(base, exp) = right.as_ref() {
                    if matches!(exp.as_ref(), Expression::Integer(2)) {
                        // Found sqrt(1 - x²)
                        // Rewrite as sqrt(1-x) * sqrt(1+x) for better precision near x=1
                        let one = Expression::Integer(1);
                        let one_minus_x = Expression::Binary(
                            BinaryOp::Sub,
                            Box::new(one.clone()),
                            Box::new(base.as_ref().clone()),
                        );
                        let one_plus_x = Expression::Binary(
                            BinaryOp::Add,
                            Box::new(one),
                            Box::new(base.as_ref().clone()),
                        );

                        let optimized = Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(Expression::Function(Function::Sqrt, vec![one_minus_x])),
                            Box::new(Expression::Function(Function::Sqrt, vec![one_plus_x])),
                        );

                        return Some(optimized);
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_angle_sin_approximation() {
        let x = Variable::new("x");
        let sin_x = Expression::Function(Function::Sin, vec![Expression::Variable(x.clone())]);

        let result = apply_small_angle_approx(&sin_x, &x, 0.1);
        assert!(result.is_some());

        let approx = result.unwrap();
        assert_eq!(approx.approximation, Expression::Variable(x.clone()));
        assert!(approx.error_bound < 2e-4); // 0.1³/6 ≈ 1.67e-4
        assert_eq!(approx.valid_range, (-0.1, 0.1));
        assert_eq!(approx.formula_used, "sin(θ) ≈ θ");
    }

    #[test]
    fn test_small_angle_cos_approximation() {
        let x = Variable::new("x");
        let cos_x = Expression::Function(Function::Cos, vec![Expression::Variable(x.clone())]);

        let result = apply_small_angle_approx(&cos_x, &x, 0.1);
        assert!(result.is_some());

        let approx = result.unwrap();
        // cos(θ) ≈ 1 - θ²/2
        assert!(approx.error_bound < 5e-5); // 0.1⁴/24 ≈ 4.17e-6
        assert_eq!(approx.formula_used, "cos(θ) ≈ 1 - θ²/2");
    }

    #[test]
    fn test_small_angle_tan_approximation() {
        let x = Variable::new("x");
        let tan_x = Expression::Function(Function::Tan, vec![Expression::Variable(x.clone())]);

        let result = apply_small_angle_approx(&tan_x, &x, 0.1);
        assert!(result.is_some());

        let approx = result.unwrap();
        assert_eq!(approx.approximation, Expression::Variable(x));
        assert!(approx.error_bound < 4e-4); // 0.1³/3 ≈ 3.33e-4
        assert_eq!(approx.formula_used, "tan(θ) ≈ θ");
    }

    #[test]
    fn test_one_minus_cos_approximation() {
        let x = Variable::new("x");
        let cos_x = Expression::Function(Function::Cos, vec![Expression::Variable(x.clone())]);
        let one_minus_cos = Expression::Binary(
            BinaryOp::Sub,
            Box::new(Expression::Integer(1)),
            Box::new(cos_x),
        );

        let result = apply_small_angle_approx(&one_minus_cos, &x, 0.1);
        assert!(result.is_some());

        let approx = result.unwrap();
        assert_eq!(approx.formula_used, "1 - cos(θ) ≈ θ²/2");
        assert!(approx.error_bound < 1e-4);
    }

    #[test]
    fn test_error_computation() {
        let x = Variable::new("x");
        let exact = Expression::Function(Function::Sin, vec![Expression::Variable(x.clone())]);
        let approx = Expression::Variable(x.clone());

        // sin(0.05) = 0.0499791693... ≈ 0.05, error ≈ 2.08e-5
        let error = compute_approximation_error(&exact, &approx, &x, 0.05);
        assert!(error < 2.1e-5);

        // sin(0.1) = 0.0998334166... ≈ 0.1, error ≈ 1.67e-4
        let error = compute_approximation_error(&exact, &approx, &x, 0.1);
        assert!(error < 2e-4);
    }

    #[test]
    fn test_approximation_validity() {
        // Small angles should be valid
        assert!(is_approximation_valid(&ApproxType::SmallAngleSin, 0.05));
        assert!(is_approximation_valid(&ApproxType::SmallAngleSin, 0.1));

        // Large angles should not be valid
        assert!(!is_approximation_valid(&ApproxType::SmallAngleSin, 1.0));
        assert!(!is_approximation_valid(&ApproxType::SmallAngleCos, 0.5));

        // Pythagorean approximation
        assert!(is_approximation_valid(&ApproxType::PythagoreanSmall, 0.01));
        assert!(!is_approximation_valid(&ApproxType::PythagoreanSmall, 0.5));
    }

    #[test]
    fn test_exp_scaling_selection() {
        // Small range: use standard form
        let form = select_exp_scaling((0.0, 10.0));
        assert_eq!(form, ScaledExpForm::Standard);

        // Medium range: use 0.1 scaling
        let form = select_exp_scaling((0.0, 100.0));
        assert_eq!(form, ScaledExpForm::Scaled01);

        // Large range: use 0.01 scaling
        let form = select_exp_scaling((0.0, 500.0));
        assert_eq!(form, ScaledExpForm::Scaled001);
    }

    #[test]
    fn test_scaled_exp_form_factor() {
        assert_eq!(ScaledExpForm::Standard.scaling_factor(), 1.0);
        assert_eq!(ScaledExpForm::Scaled01.scaling_factor(), 0.1);
        assert_eq!(ScaledExpForm::Scaled001.scaling_factor(), 0.01);
        assert_eq!(ScaledExpForm::Custom(0.5).scaling_factor(), 0.5);
    }

    #[test]
    fn test_pythagorean_optimization() {
        // Create sqrt(1 - x²)
        let x = Expression::Variable(Variable::new("x"));
        let x_squared = Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(2)));
        let one_minus_x_squared = Expression::Binary(
            BinaryOp::Sub,
            Box::new(Expression::Integer(1)),
            Box::new(x_squared),
        );
        let sqrt_expr = Expression::Function(Function::Sqrt, vec![one_minus_x_squared]);

        let optimized = optimize_pythagorean(&sqrt_expr);
        assert!(optimized.is_some());

        // Should be sqrt(1-x) * sqrt(1+x)
        if let Some(Expression::Binary(BinaryOp::Mul, left, right)) = optimized {
            assert!(matches!(
                left.as_ref(),
                Expression::Function(Function::Sqrt, _)
            ));
            assert!(matches!(
                right.as_ref(),
                Expression::Function(Function::Sqrt, _)
            ));
        } else {
            panic!("Expected multiplication of two square roots");
        }
    }

    #[test]
    fn test_approximation_error_bounds_are_conservative() {
        // Verify that stated error bounds are actually conservative
        let x = Variable::new("x");

        // Test sin approximation
        let sin_x = Expression::Function(Function::Sin, vec![Expression::Variable(x.clone())]);
        let result = apply_small_angle_approx(&sin_x, &x, 0.1).unwrap();

        // Check multiple points within range
        for value in [0.01, 0.05, 0.09] {
            let actual_error =
                compute_approximation_error(&sin_x, &result.approximation, &x, value);
            assert!(
                actual_error <= result.error_bound,
                "Actual error {} exceeds stated bound {} at value {}",
                actual_error,
                result.error_bound,
                value
            );
        }
    }

    #[test]
    fn test_approximation_step_generation() {
        let x = Variable::new("x");
        let original = Expression::Function(Function::Sin, vec![Expression::Variable(x.clone())]);
        let approx = Expression::Variable(x);

        let step = generate_approximation_step(&original, &approx, 1e-5, "sin(θ) ≈ θ".to_string());

        assert!(step.explanation.contains("sin(θ) ≈ θ"));
        assert!(step.explanation.contains("Error bound"));
        assert_eq!(step.result, approx);
    }
}
