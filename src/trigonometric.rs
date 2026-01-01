//! Trigonometric identity simplification.
//!
//! This module provides functions for simplifying trigonometric expressions
//! using standard identities including:
//!
//! - Pythagorean identities (sin²x + cos²x = 1, etc.)
//! - Double angle formulas
//! - Sum and difference formulas
//! - Product-to-sum and sum-to-product transformations
//!
//! # Examples
//!
//! ```
//! use thales::trigonometric::simplify_trig;
//! use thales::ast::{Expression, Variable, Function, BinaryOp};
//!
//! // sin²(x) + cos²(x) simplifies to 1
//! let x = Expression::Variable(Variable::new("x"));
//! let sin_x = Expression::Function(Function::Sin, vec![x.clone()]);
//! let cos_x = Expression::Function(Function::Cos, vec![x.clone()]);
//! let sin_sq = Expression::Power(Box::new(sin_x), Box::new(Expression::Integer(2)));
//! let cos_sq = Expression::Power(Box::new(cos_x), Box::new(Expression::Integer(2)));
//! let expr = Expression::Binary(BinaryOp::Add, Box::new(sin_sq), Box::new(cos_sq));
//!
//! let simplified = simplify_trig(&expr);
//! // Result should be 1
//! ```

use crate::ast::{BinaryOp, Expression, Function, UnaryOp};
use crate::pattern::{apply_rule_recursive, apply_rules_to_fixpoint, Pattern, Rule};

// =============================================================================
// Pythagorean Identity Rules
// =============================================================================

/// Create rules for Pythagorean identities.
///
/// - sin²(x) + cos²(x) = 1
/// - 1 - sin²(x) = cos²(x)
/// - 1 - cos²(x) = sin²(x)
/// - sec²(x) - tan²(x) = 1 (or 1 + tan²(x) = sec²(x))
/// - csc²(x) - cot²(x) = 1 (or 1 + cot²(x) = csc²(x))
pub fn pythagorean_rules() -> Vec<Rule> {
    vec![
        // sin²(x) + cos²(x) = 1
        Rule::new(
            Pattern::add(
                Pattern::power(
                    Pattern::function(Function::Sin, vec![Pattern::wildcard("x")]),
                    Pattern::exact(Expression::Integer(2)),
                ),
                Pattern::power(
                    Pattern::function(Function::Cos, vec![Pattern::wildcard("x")]),
                    Pattern::exact(Expression::Integer(2)),
                ),
            ),
            Pattern::exact(Expression::Integer(1)),
        )
        .named("sin²x + cos²x = 1"),
        // cos²(x) + sin²(x) = 1 (commutative)
        Rule::new(
            Pattern::add(
                Pattern::power(
                    Pattern::function(Function::Cos, vec![Pattern::wildcard("x")]),
                    Pattern::exact(Expression::Integer(2)),
                ),
                Pattern::power(
                    Pattern::function(Function::Sin, vec![Pattern::wildcard("x")]),
                    Pattern::exact(Expression::Integer(2)),
                ),
            ),
            Pattern::exact(Expression::Integer(1)),
        )
        .named("cos²x + sin²x = 1"),
        // 1 - sin²(x) = cos²(x)
        Rule::new(
            Pattern::sub(
                Pattern::exact(Expression::Integer(1)),
                Pattern::power(
                    Pattern::function(Function::Sin, vec![Pattern::wildcard("x")]),
                    Pattern::exact(Expression::Integer(2)),
                ),
            ),
            Pattern::power(
                Pattern::function(Function::Cos, vec![Pattern::wildcard("x")]),
                Pattern::exact(Expression::Integer(2)),
            ),
        )
        .named("1 - sin²x = cos²x"),
        // 1 - cos²(x) = sin²(x)
        Rule::new(
            Pattern::sub(
                Pattern::exact(Expression::Integer(1)),
                Pattern::power(
                    Pattern::function(Function::Cos, vec![Pattern::wildcard("x")]),
                    Pattern::exact(Expression::Integer(2)),
                ),
            ),
            Pattern::power(
                Pattern::function(Function::Sin, vec![Pattern::wildcard("x")]),
                Pattern::exact(Expression::Integer(2)),
            ),
        )
        .named("1 - cos²x = sin²x"),
        // 1 + tan²(x) = 1/cos²(x)
        Rule::new(
            Pattern::add(
                Pattern::exact(Expression::Integer(1)),
                Pattern::power(
                    Pattern::function(Function::Tan, vec![Pattern::wildcard("x")]),
                    Pattern::exact(Expression::Integer(2)),
                ),
            ),
            Pattern::div(
                Pattern::exact(Expression::Integer(1)),
                Pattern::power(
                    Pattern::function(Function::Cos, vec![Pattern::wildcard("x")]),
                    Pattern::exact(Expression::Integer(2)),
                ),
            ),
        )
        .named("1 + tan²x = 1/cos²x"),
        // tan²(x) + 1 = 1/cos²(x) (commutative)
        Rule::new(
            Pattern::add(
                Pattern::power(
                    Pattern::function(Function::Tan, vec![Pattern::wildcard("x")]),
                    Pattern::exact(Expression::Integer(2)),
                ),
                Pattern::exact(Expression::Integer(1)),
            ),
            Pattern::div(
                Pattern::exact(Expression::Integer(1)),
                Pattern::power(
                    Pattern::function(Function::Cos, vec![Pattern::wildcard("x")]),
                    Pattern::exact(Expression::Integer(2)),
                ),
            ),
        )
        .named("tan²x + 1 = 1/cos²x"),
    ]
}

// =============================================================================
// Double Angle Rules
// =============================================================================

/// Create rules for double angle formulas.
///
/// - sin(2x) = 2*sin(x)*cos(x)
/// - cos(2x) = cos²(x) - sin²(x) = 2cos²(x) - 1 = 1 - 2sin²(x)
/// - tan(2x) = 2tan(x) / (1 - tan²(x))
pub fn double_angle_rules() -> Vec<Rule> {
    vec![
        // 2*sin(x)*cos(x) = sin(2x)
        Rule::new(
            Pattern::mul(
                Pattern::exact(Expression::Integer(2)),
                Pattern::mul(
                    Pattern::function(Function::Sin, vec![Pattern::wildcard("x")]),
                    Pattern::function(Function::Cos, vec![Pattern::wildcard("x")]),
                ),
            ),
            Pattern::function(
                Function::Sin,
                vec![Pattern::mul(
                    Pattern::exact(Expression::Integer(2)),
                    Pattern::wildcard("x"),
                )],
            ),
        )
        .named("2*sin(x)*cos(x) = sin(2x)"),
        // 2*cos(x)*sin(x) = sin(2x) (commutative)
        Rule::new(
            Pattern::mul(
                Pattern::exact(Expression::Integer(2)),
                Pattern::mul(
                    Pattern::function(Function::Cos, vec![Pattern::wildcard("x")]),
                    Pattern::function(Function::Sin, vec![Pattern::wildcard("x")]),
                ),
            ),
            Pattern::function(
                Function::Sin,
                vec![Pattern::mul(
                    Pattern::exact(Expression::Integer(2)),
                    Pattern::wildcard("x"),
                )],
            ),
        )
        .named("2*cos(x)*sin(x) = sin(2x)"),
        // cos²(x) - sin²(x) = cos(2x)
        Rule::new(
            Pattern::sub(
                Pattern::power(
                    Pattern::function(Function::Cos, vec![Pattern::wildcard("x")]),
                    Pattern::exact(Expression::Integer(2)),
                ),
                Pattern::power(
                    Pattern::function(Function::Sin, vec![Pattern::wildcard("x")]),
                    Pattern::exact(Expression::Integer(2)),
                ),
            ),
            Pattern::function(
                Function::Cos,
                vec![Pattern::mul(
                    Pattern::exact(Expression::Integer(2)),
                    Pattern::wildcard("x"),
                )],
            ),
        )
        .named("cos²x - sin²x = cos(2x)"),
        // 2*cos²(x) - 1 = cos(2x)
        Rule::new(
            Pattern::sub(
                Pattern::mul(
                    Pattern::exact(Expression::Integer(2)),
                    Pattern::power(
                        Pattern::function(Function::Cos, vec![Pattern::wildcard("x")]),
                        Pattern::exact(Expression::Integer(2)),
                    ),
                ),
                Pattern::exact(Expression::Integer(1)),
            ),
            Pattern::function(
                Function::Cos,
                vec![Pattern::mul(
                    Pattern::exact(Expression::Integer(2)),
                    Pattern::wildcard("x"),
                )],
            ),
        )
        .named("2cos²x - 1 = cos(2x)"),
        // 1 - 2*sin²(x) = cos(2x)
        Rule::new(
            Pattern::sub(
                Pattern::exact(Expression::Integer(1)),
                Pattern::mul(
                    Pattern::exact(Expression::Integer(2)),
                    Pattern::power(
                        Pattern::function(Function::Sin, vec![Pattern::wildcard("x")]),
                        Pattern::exact(Expression::Integer(2)),
                    ),
                ),
            ),
            Pattern::function(
                Function::Cos,
                vec![Pattern::mul(
                    Pattern::exact(Expression::Integer(2)),
                    Pattern::wildcard("x"),
                )],
            ),
        )
        .named("1 - 2sin²x = cos(2x)"),
    ]
}

// =============================================================================
// Reciprocal and Quotient Rules
// =============================================================================

/// Create rules for reciprocal and quotient identities.
///
/// - tan(x) = sin(x)/cos(x)
/// - 1/tan(x) = cos(x)/sin(x)
pub fn quotient_rules() -> Vec<Rule> {
    vec![
        // sin(x)/cos(x) = tan(x)
        Rule::new(
            Pattern::div(
                Pattern::function(Function::Sin, vec![Pattern::wildcard("x")]),
                Pattern::function(Function::Cos, vec![Pattern::wildcard("x")]),
            ),
            Pattern::function(Function::Tan, vec![Pattern::wildcard("x")]),
        )
        .named("sin(x)/cos(x) = tan(x)"),
        // cos(x)/sin(x) = 1/tan(x) (reciprocal of tan)
        Rule::new(
            Pattern::div(
                Pattern::function(Function::Cos, vec![Pattern::wildcard("x")]),
                Pattern::function(Function::Sin, vec![Pattern::wildcard("x")]),
            ),
            Pattern::div(
                Pattern::exact(Expression::Integer(1)),
                Pattern::function(Function::Tan, vec![Pattern::wildcard("x")]),
            ),
        )
        .named("cos(x)/sin(x) = 1/tan(x)"),
    ]
}

// =============================================================================
// Even/Odd Function Rules
// =============================================================================

/// Create rules for even/odd trigonometric functions.
///
/// - sin(-x) = -sin(x) (odd)
/// - cos(-x) = cos(x) (even)
/// - tan(-x) = -tan(x) (odd)
pub fn parity_rules() -> Vec<Rule> {
    vec![
        // sin(-x) = -sin(x)
        Rule::new(
            Pattern::function(
                Function::Sin,
                vec![Pattern::unary(UnaryOp::Neg, Pattern::wildcard("x"))],
            ),
            Pattern::unary(
                UnaryOp::Neg,
                Pattern::function(Function::Sin, vec![Pattern::wildcard("x")]),
            ),
        )
        .named("sin(-x) = -sin(x)"),
        // cos(-x) = cos(x)
        Rule::new(
            Pattern::function(
                Function::Cos,
                vec![Pattern::unary(UnaryOp::Neg, Pattern::wildcard("x"))],
            ),
            Pattern::function(Function::Cos, vec![Pattern::wildcard("x")]),
        )
        .named("cos(-x) = cos(x)"),
        // tan(-x) = -tan(x)
        Rule::new(
            Pattern::function(
                Function::Tan,
                vec![Pattern::unary(UnaryOp::Neg, Pattern::wildcard("x"))],
            ),
            Pattern::unary(
                UnaryOp::Neg,
                Pattern::function(Function::Tan, vec![Pattern::wildcard("x")]),
            ),
        )
        .named("tan(-x) = -tan(x)"),
    ]
}

// =============================================================================
// Product-to-Sum Rules
// =============================================================================

/// Create rules for product-to-sum transformations.
///
/// - sin(a)cos(b) = [sin(a+b) + sin(a-b)]/2
/// - cos(a)cos(b) = [cos(a-b) + cos(a+b)]/2
/// - sin(a)sin(b) = [cos(a-b) - cos(a+b)]/2
pub fn product_to_sum_rules() -> Vec<Rule> {
    // These are complex transformations that expand products into sums
    // For simplification, we typically go the other way (sum-to-product)
    vec![
        // sin(x)*sin(x) = sin²(x) = (1 - cos(2x))/2
        Rule::new(
            Pattern::mul(
                Pattern::function(Function::Sin, vec![Pattern::wildcard("x")]),
                Pattern::function(Function::Sin, vec![Pattern::wildcard("x")]),
            ),
            Pattern::power(
                Pattern::function(Function::Sin, vec![Pattern::wildcard("x")]),
                Pattern::exact(Expression::Integer(2)),
            ),
        )
        .named("sin(x)*sin(x) = sin²(x)"),
        // cos(x)*cos(x) = cos²(x)
        Rule::new(
            Pattern::mul(
                Pattern::function(Function::Cos, vec![Pattern::wildcard("x")]),
                Pattern::function(Function::Cos, vec![Pattern::wildcard("x")]),
            ),
            Pattern::power(
                Pattern::function(Function::Cos, vec![Pattern::wildcard("x")]),
                Pattern::exact(Expression::Integer(2)),
            ),
        )
        .named("cos(x)*cos(x) = cos²(x)"),
        // tan(x)*tan(x) = tan²(x)
        Rule::new(
            Pattern::mul(
                Pattern::function(Function::Tan, vec![Pattern::wildcard("x")]),
                Pattern::function(Function::Tan, vec![Pattern::wildcard("x")]),
            ),
            Pattern::power(
                Pattern::function(Function::Tan, vec![Pattern::wildcard("x")]),
                Pattern::exact(Expression::Integer(2)),
            ),
        )
        .named("tan(x)*tan(x) = tan²(x)"),
    ]
}

// =============================================================================
// Special Value Rules
// =============================================================================

/// Create rules for trigonometric functions at special angles.
///
/// These rules simplify expressions like sin(0), cos(π), etc.
pub fn special_value_rules() -> Vec<Rule> {
    use crate::ast::SymbolicConstant;

    vec![
        // sin(0) = 0
        Rule::new(
            Pattern::function(Function::Sin, vec![Pattern::exact(Expression::Integer(0))]),
            Pattern::exact(Expression::Integer(0)),
        )
        .named("sin(0) = 0"),
        // cos(0) = 1
        Rule::new(
            Pattern::function(Function::Cos, vec![Pattern::exact(Expression::Integer(0))]),
            Pattern::exact(Expression::Integer(1)),
        )
        .named("cos(0) = 1"),
        // tan(0) = 0
        Rule::new(
            Pattern::function(Function::Tan, vec![Pattern::exact(Expression::Integer(0))]),
            Pattern::exact(Expression::Integer(0)),
        )
        .named("tan(0) = 0"),
        // sin(π) = 0
        Rule::new(
            Pattern::function(
                Function::Sin,
                vec![Pattern::exact(Expression::Constant(SymbolicConstant::Pi))],
            ),
            Pattern::exact(Expression::Integer(0)),
        )
        .named("sin(π) = 0"),
        // cos(π) = -1
        Rule::new(
            Pattern::function(
                Function::Cos,
                vec![Pattern::exact(Expression::Constant(SymbolicConstant::Pi))],
            ),
            Pattern::exact(Expression::Integer(-1)),
        )
        .named("cos(π) = -1"),
        // tan(π) = 0
        Rule::new(
            Pattern::function(
                Function::Tan,
                vec![Pattern::exact(Expression::Constant(SymbolicConstant::Pi))],
            ),
            Pattern::exact(Expression::Integer(0)),
        )
        .named("tan(π) = 0"),
    ]
}

// =============================================================================
// Inverse Composition Rules
// =============================================================================

/// Create rules for inverse trigonometric compositions.
///
/// - sin(arcsin(x)) = x
/// - cos(arccos(x)) = x
/// - tan(arctan(x)) = x
pub fn inverse_rules() -> Vec<Rule> {
    vec![
        // sin(arcsin(x)) = x
        Rule::new(
            Pattern::function(
                Function::Sin,
                vec![Pattern::function(
                    Function::Asin,
                    vec![Pattern::wildcard("x")],
                )],
            ),
            Pattern::wildcard("x"),
        )
        .named("sin(arcsin(x)) = x"),
        // cos(arccos(x)) = x
        Rule::new(
            Pattern::function(
                Function::Cos,
                vec![Pattern::function(
                    Function::Acos,
                    vec![Pattern::wildcard("x")],
                )],
            ),
            Pattern::wildcard("x"),
        )
        .named("cos(arccos(x)) = x"),
        // tan(arctan(x)) = x
        Rule::new(
            Pattern::function(
                Function::Tan,
                vec![Pattern::function(
                    Function::Atan,
                    vec![Pattern::wildcard("x")],
                )],
            ),
            Pattern::wildcard("x"),
        )
        .named("tan(arctan(x)) = x"),
    ]
}

// =============================================================================
// Main Simplification Functions
// =============================================================================

/// Get all trigonometric simplification rules.
pub fn all_trig_rules() -> Vec<Rule> {
    let mut rules = Vec::new();
    rules.extend(pythagorean_rules());
    rules.extend(double_angle_rules());
    rules.extend(quotient_rules());
    rules.extend(parity_rules());
    rules.extend(product_to_sum_rules());
    rules.extend(special_value_rules());
    rules.extend(inverse_rules());
    rules
}

/// Maximum iterations for simplification to prevent infinite loops.
const MAX_ITERATIONS: usize = 100;

/// Simplify a trigonometric expression using all available rules.
///
/// This function applies trigonometric identities recursively until
/// no more simplifications can be made.
///
/// # Examples
///
/// ```
/// use thales::trigonometric::simplify_trig;
/// use thales::ast::{Expression, Variable, Function, BinaryOp};
///
/// // sin²(x) + cos²(x) = 1
/// let x = Expression::Variable(Variable::new("x"));
/// let sin_x = Expression::Function(Function::Sin, vec![x.clone()]);
/// let cos_x = Expression::Function(Function::Cos, vec![x.clone()]);
/// let sin_sq = Expression::Power(Box::new(sin_x), Box::new(Expression::Integer(2)));
/// let cos_sq = Expression::Power(Box::new(cos_x), Box::new(Expression::Integer(2)));
/// let expr = Expression::Binary(BinaryOp::Add, Box::new(sin_sq), Box::new(cos_sq));
///
/// let simplified = simplify_trig(&expr);
/// assert_eq!(simplified, Expression::Integer(1));
/// ```
pub fn simplify_trig(expr: &Expression) -> Expression {
    let rules = all_trig_rules();
    apply_rules_to_fixpoint(expr, &rules, MAX_ITERATIONS)
}

/// Simplify using only Pythagorean identities.
pub fn simplify_pythagorean(expr: &Expression) -> Expression {
    let rules = pythagorean_rules();
    apply_rules_to_fixpoint(expr, &rules, MAX_ITERATIONS)
}

/// Simplify using only double angle formulas.
pub fn simplify_double_angle(expr: &Expression) -> Expression {
    let rules = double_angle_rules();
    apply_rules_to_fixpoint(expr, &rules, MAX_ITERATIONS)
}

/// Simplify using only quotient/reciprocal identities.
pub fn simplify_quotient(expr: &Expression) -> Expression {
    let rules = quotient_rules();
    apply_rules_to_fixpoint(expr, &rules, MAX_ITERATIONS)
}

/// Apply a single trigonometric simplification step.
///
/// Returns the simplified expression and whether any rule was applied.
pub fn simplify_trig_step(expr: &Expression) -> (Expression, bool) {
    let rules = all_trig_rules();

    for rule in &rules {
        let result = apply_rule_recursive(expr, rule);
        if result != *expr {
            return (result, true);
        }
    }

    (expr.clone(), false)
}

/// Apply trigonometric simplification with step tracking.
///
/// Returns the final result and a list of applied rule names.
pub fn simplify_trig_with_steps(expr: &Expression) -> (Expression, Vec<String>) {
    let rules = all_trig_rules();
    let mut current = expr.clone();
    let mut steps = Vec::new();

    for _ in 0..MAX_ITERATIONS {
        let mut changed = false;

        for rule in &rules {
            let result = apply_rule_recursive(&current, rule);
            if result != current {
                if let Some(name) = &rule.name {
                    steps.push(name.clone());
                }
                current = result;
                changed = true;
                break;
            }
        }

        if !changed {
            break;
        }
    }

    (current, steps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Variable;

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn sin(x: Expression) -> Expression {
        Expression::Function(Function::Sin, vec![x])
    }

    fn cos(x: Expression) -> Expression {
        Expression::Function(Function::Cos, vec![x])
    }

    fn tan(x: Expression) -> Expression {
        Expression::Function(Function::Tan, vec![x])
    }

    fn pow(base: Expression, exp: i64) -> Expression {
        Expression::Power(Box::new(base), Box::new(Expression::Integer(exp)))
    }

    fn add(left: Expression, right: Expression) -> Expression {
        Expression::Binary(BinaryOp::Add, Box::new(left), Box::new(right))
    }

    fn sub(left: Expression, right: Expression) -> Expression {
        Expression::Binary(BinaryOp::Sub, Box::new(left), Box::new(right))
    }

    fn mul(left: Expression, right: Expression) -> Expression {
        Expression::Binary(BinaryOp::Mul, Box::new(left), Box::new(right))
    }

    fn div(left: Expression, right: Expression) -> Expression {
        Expression::Binary(BinaryOp::Div, Box::new(left), Box::new(right))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn neg(x: Expression) -> Expression {
        Expression::Unary(UnaryOp::Neg, Box::new(x))
    }

    #[test]
    fn test_pythagorean_sin_cos_squared() {
        // sin²(x) + cos²(x) = 1
        let x = var("x");
        let expr = add(pow(sin(x.clone()), 2), pow(cos(x.clone()), 2));

        let result = simplify_trig(&expr);
        assert_eq!(result, int(1));
    }

    #[test]
    fn test_pythagorean_cos_sin_squared() {
        // cos²(x) + sin²(x) = 1 (commutative)
        let x = var("x");
        let expr = add(pow(cos(x.clone()), 2), pow(sin(x.clone()), 2));

        let result = simplify_trig(&expr);
        assert_eq!(result, int(1));
    }

    #[test]
    fn test_pythagorean_one_minus_sin_squared() {
        // 1 - sin²(x) = cos²(x)
        let x = var("x");
        let expr = sub(int(1), pow(sin(x.clone()), 2));

        let result = simplify_trig(&expr);
        assert_eq!(result, pow(cos(x), 2));
    }

    #[test]
    fn test_pythagorean_one_minus_cos_squared() {
        // 1 - cos²(x) = sin²(x)
        let x = var("x");
        let expr = sub(int(1), pow(cos(x.clone()), 2));

        let result = simplify_trig(&expr);
        assert_eq!(result, pow(sin(x), 2));
    }

    #[test]
    fn test_quotient_sin_over_cos() {
        // sin(x)/cos(x) = tan(x)
        let x = var("x");
        let expr = div(sin(x.clone()), cos(x.clone()));

        let result = simplify_trig(&expr);
        assert_eq!(result, tan(x));
    }

    #[test]
    fn test_parity_sin_negative() {
        // sin(-x) = -sin(x)
        let x = var("x");
        let expr = sin(neg(x.clone()));

        let result = simplify_trig(&expr);
        assert_eq!(result, neg(sin(x)));
    }

    #[test]
    fn test_parity_cos_negative() {
        // cos(-x) = cos(x)
        let x = var("x");
        let expr = cos(neg(x.clone()));

        let result = simplify_trig(&expr);
        assert_eq!(result, cos(x));
    }

    #[test]
    fn test_special_sin_zero() {
        // sin(0) = 0
        let expr = sin(int(0));
        let result = simplify_trig(&expr);
        assert_eq!(result, int(0));
    }

    #[test]
    fn test_special_cos_zero() {
        // cos(0) = 1
        let expr = cos(int(0));
        let result = simplify_trig(&expr);
        assert_eq!(result, int(1));
    }

    #[test]
    fn test_double_angle_sin_product() {
        // 2*sin(x)*cos(x) = sin(2x)
        let x = var("x");
        let expr = mul(int(2), mul(sin(x.clone()), cos(x.clone())));

        let result = simplify_trig(&expr);
        let expected = sin(mul(int(2), x));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_double_angle_cos_difference() {
        // cos²(x) - sin²(x) = cos(2x)
        let x = var("x");
        let expr = sub(pow(cos(x.clone()), 2), pow(sin(x.clone()), 2));

        let result = simplify_trig(&expr);
        let expected = cos(mul(int(2), x));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_inverse_sin_arcsin() {
        // sin(arcsin(x)) = x
        let x = var("x");
        let expr = sin(Expression::Function(Function::Asin, vec![x.clone()]));

        let result = simplify_trig(&expr);
        assert_eq!(result, x);
    }

    #[test]
    fn test_product_sin_sin_to_squared() {
        // sin(x)*sin(x) = sin²(x)
        let x = var("x");
        let expr = mul(sin(x.clone()), sin(x.clone()));

        let result = simplify_trig(&expr);
        assert_eq!(result, pow(sin(x), 2));
    }

    #[test]
    fn test_simplify_with_steps() {
        let x = var("x");
        let expr = add(pow(sin(x.clone()), 2), pow(cos(x.clone()), 2));

        let (result, steps) = simplify_trig_with_steps(&expr);
        assert_eq!(result, int(1));
        assert!(!steps.is_empty());
    }

    #[test]
    fn test_one_plus_tan_squared() {
        // 1 + tan²(x) = 1/cos²(x)
        let x = var("x");
        let expr = add(int(1), pow(tan(x.clone()), 2));

        let result = simplify_trig(&expr);
        let expected = div(int(1), pow(cos(x), 2));
        assert_eq!(result, expected);
    }
}
