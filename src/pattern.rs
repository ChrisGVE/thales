//! Pattern matching for expression transformations.
//!
//! This module provides a pattern matching system that enables rule-based
//! expression transformations using wildcards and pattern templates.
//!
//! # Overview
//!
//! The pattern matching system allows you to:
//! - Define patterns that match expressions with wildcards
//! - Bind matched subexpressions to named variables
//! - Apply transformation rules based on pattern matches
//! - Build rule-based simplification and rewriting systems
//!
//! # Examples
//!
//! ## Simple Pattern Matching
//!
//! ```
//! use mathsolver_core::pattern::{Pattern, match_pattern};
//! use mathsolver_core::ast::{Expression, BinaryOp, Variable};
//!
//! // Pattern: a + b (wildcards for any expressions)
//! let pattern = Pattern::Binary(
//!     BinaryOp::Add,
//!     Box::new(Pattern::Wildcard("a".to_string())),
//!     Box::new(Pattern::Wildcard("b".to_string()))
//! );
//!
//! // Expression: x + y
//! let x = Expression::Variable(Variable::new("x"));
//! let y = Expression::Variable(Variable::new("y"));
//! let expr = Expression::Binary(BinaryOp::Add, Box::new(x.clone()), Box::new(y.clone()));
//!
//! // Match and get bindings
//! let bindings = match_pattern(&expr, &pattern).unwrap();
//! assert_eq!(bindings.get("a"), Some(&x));
//! assert_eq!(bindings.get("b"), Some(&y));
//! ```
//!
//! ## Transformation Rules
//!
//! ```
//! use mathsolver_core::pattern::{Pattern, Rule, apply_rule};
//! use mathsolver_core::ast::{Expression, BinaryOp};
//!
//! // Rule: x + 0 = x (additive identity)
//! let rule = Rule::new(
//!     Pattern::Binary(
//!         BinaryOp::Add,
//!         Box::new(Pattern::Wildcard("x".to_string())),
//!         Box::new(Pattern::Exact(Expression::Integer(0)))
//!     ),
//!     Pattern::Wildcard("x".to_string())
//! );
//!
//! // Apply rule to: y + 0
//! let y = Expression::Variable(mathsolver_core::ast::Variable::new("y"));
//! let expr = Expression::Binary(BinaryOp::Add, Box::new(y.clone()), Box::new(Expression::Integer(0)));
//!
//! let result = apply_rule(&expr, &rule);
//! assert_eq!(result, Some(y));
//! ```

use crate::ast::{BinaryOp, Expression, Function, UnaryOp};
use std::collections::HashMap;

/// A pattern for matching expressions.
///
/// Patterns can include wildcards that match any expression and bind
/// the matched expression to a name for later use in transformations.
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    /// Matches any expression and binds it to the given name.
    ///
    /// # Example
    ///
    /// `Wildcard("x")` matches any expression and binds it to "x".
    Wildcard(String),

    /// Matches any expression without binding.
    ///
    /// Use this when you need to match something but don't care about its value.
    Any,

    /// Matches an exact expression.
    ///
    /// The expression must be structurally equal to match.
    Exact(Expression),

    /// Matches a binary operation with the given operator and sub-patterns.
    Binary(BinaryOp, Box<Pattern>, Box<Pattern>),

    /// Matches a unary operation with the given operator and sub-pattern.
    Unary(UnaryOp, Box<Pattern>),

    /// Matches a function call with the given function and argument patterns.
    Function(Function, Vec<Pattern>),

    /// Matches a power expression with base and exponent patterns.
    Power(Box<Pattern>, Box<Pattern>),

    /// Matches an integer with a specific value.
    Integer(i64),

    /// Matches any integer and optionally binds it.
    AnyInteger(Option<String>),

    /// Matches any variable and optionally binds its name.
    AnyVariable(Option<String>),
}

impl Pattern {
    /// Create a wildcard pattern that binds to the given name.
    pub fn wildcard(name: &str) -> Self {
        Pattern::Wildcard(name.to_string())
    }

    /// Create an exact pattern from an expression.
    pub fn exact(expr: Expression) -> Self {
        Pattern::Exact(expr)
    }

    /// Create a binary operation pattern.
    pub fn binary(op: BinaryOp, left: Pattern, right: Pattern) -> Self {
        Pattern::Binary(op, Box::new(left), Box::new(right))
    }

    /// Create a unary operation pattern.
    pub fn unary(op: UnaryOp, operand: Pattern) -> Self {
        Pattern::Unary(op, Box::new(operand))
    }

    /// Create a function pattern.
    pub fn function(func: Function, args: Vec<Pattern>) -> Self {
        Pattern::Function(func, args)
    }

    /// Create a power pattern.
    pub fn power(base: Pattern, exp: Pattern) -> Self {
        Pattern::Power(Box::new(base), Box::new(exp))
    }

    /// Create an addition pattern: a + b
    pub fn add(left: Pattern, right: Pattern) -> Self {
        Pattern::binary(BinaryOp::Add, left, right)
    }

    /// Create a subtraction pattern: a - b
    pub fn sub(left: Pattern, right: Pattern) -> Self {
        Pattern::binary(BinaryOp::Sub, left, right)
    }

    /// Create a multiplication pattern: a * b
    pub fn mul(left: Pattern, right: Pattern) -> Self {
        Pattern::binary(BinaryOp::Mul, left, right)
    }

    /// Create a division pattern: a / b
    pub fn div(left: Pattern, right: Pattern) -> Self {
        Pattern::binary(BinaryOp::Div, left, right)
    }
}

/// Match an expression against a pattern.
///
/// Returns `Some(bindings)` if the expression matches the pattern,
/// where `bindings` is a map from wildcard names to matched expressions.
/// Returns `None` if the expression doesn't match.
///
/// # Arguments
///
/// * `expr` - The expression to match against
/// * `pattern` - The pattern to match
///
/// # Returns
///
/// * `Some(HashMap<String, Expression>)` - Bindings from wildcards to expressions
/// * `None` - If the expression doesn't match the pattern
///
/// # Examples
///
/// ```
/// use mathsolver_core::pattern::{Pattern, match_pattern};
/// use mathsolver_core::ast::{Expression, BinaryOp, Variable};
///
/// // Match x + y against a + b pattern
/// let pattern = Pattern::add(
///     Pattern::wildcard("a"),
///     Pattern::wildcard("b")
/// );
///
/// let x = Expression::Variable(Variable::new("x"));
/// let y = Expression::Variable(Variable::new("y"));
/// let expr = Expression::Binary(BinaryOp::Add, Box::new(x.clone()), Box::new(y.clone()));
///
/// let bindings = match_pattern(&expr, &pattern).unwrap();
/// assert_eq!(bindings.get("a"), Some(&x));
/// assert_eq!(bindings.get("b"), Some(&y));
/// ```
pub fn match_pattern(expr: &Expression, pattern: &Pattern) -> Option<HashMap<String, Expression>> {
    let mut bindings = HashMap::new();
    if match_pattern_internal(expr, pattern, &mut bindings) {
        Some(bindings)
    } else {
        None
    }
}

/// Internal pattern matching with bindings accumulation.
fn match_pattern_internal(
    expr: &Expression,
    pattern: &Pattern,
    bindings: &mut HashMap<String, Expression>,
) -> bool {
    match pattern {
        Pattern::Wildcard(name) => {
            // Check if this wildcard was already bound
            if let Some(existing) = bindings.get(name) {
                // Must match the same expression
                expr == existing
            } else {
                // Bind this wildcard
                bindings.insert(name.clone(), expr.clone());
                true
            }
        }

        Pattern::Any => true,

        Pattern::Exact(target) => expr == target,

        Pattern::Integer(n) => matches!(expr, Expression::Integer(m) if m == n),

        Pattern::AnyInteger(opt_name) => {
            if let Expression::Integer(n) = expr {
                if let Some(name) = opt_name {
                    bindings.insert(name.clone(), Expression::Integer(*n));
                }
                true
            } else {
                false
            }
        }

        Pattern::AnyVariable(opt_name) => {
            if let Expression::Variable(v) = expr {
                if let Some(name) = opt_name {
                    bindings.insert(name.clone(), expr.clone());
                }
                true
            } else {
                false
            }
        }

        Pattern::Binary(op, left_pat, right_pat) => {
            if let Expression::Binary(expr_op, left_expr, right_expr) = expr {
                if op == expr_op {
                    // Try direct match
                    if match_pattern_internal(left_expr, left_pat, bindings)
                        && match_pattern_internal(right_expr, right_pat, bindings)
                    {
                        return true;
                    }

                    // Try commutative match for + and *
                    if matches!(op, BinaryOp::Add | BinaryOp::Mul) {
                        let mut comm_bindings = bindings.clone();
                        if match_pattern_internal(left_expr, right_pat, &mut comm_bindings)
                            && match_pattern_internal(right_expr, left_pat, &mut comm_bindings)
                        {
                            *bindings = comm_bindings;
                            return true;
                        }
                    }
                }
            }
            false
        }

        Pattern::Unary(op, operand_pat) => {
            if let Expression::Unary(expr_op, operand) = expr {
                op == expr_op && match_pattern_internal(operand, operand_pat, bindings)
            } else {
                false
            }
        }

        Pattern::Function(func, arg_pats) => {
            if let Expression::Function(expr_func, args) = expr {
                if func == expr_func && arg_pats.len() == args.len() {
                    for (arg_pat, arg) in arg_pats.iter().zip(args.iter()) {
                        if !match_pattern_internal(arg, arg_pat, bindings) {
                            return false;
                        }
                    }
                    true
                } else {
                    false
                }
            } else {
                false
            }
        }

        Pattern::Power(base_pat, exp_pat) => {
            if let Expression::Power(base, exp) = expr {
                match_pattern_internal(base, base_pat, bindings)
                    && match_pattern_internal(exp, exp_pat, bindings)
            } else {
                false
            }
        }
    }
}

/// Apply bindings to a pattern to produce an expression.
///
/// Substitutes wildcards in the pattern with their bound expressions.
///
/// # Arguments
///
/// * `bindings` - Map from wildcard names to expressions
/// * `pattern` - The pattern to instantiate
///
/// # Returns
///
/// The expression with wildcards replaced by their bindings.
///
/// # Panics
///
/// Panics if a wildcard in the pattern is not found in bindings.
///
/// # Examples
///
/// ```
/// use mathsolver_core::pattern::{Pattern, apply_pattern};
/// use mathsolver_core::ast::{Expression, Variable};
/// use std::collections::HashMap;
///
/// let mut bindings = HashMap::new();
/// bindings.insert("x".to_string(), Expression::Variable(Variable::new("y")));
///
/// let pattern = Pattern::wildcard("x");
/// let result = apply_pattern(&bindings, &pattern);
/// assert_eq!(result, Expression::Variable(Variable::new("y")));
/// ```
pub fn apply_pattern(bindings: &HashMap<String, Expression>, pattern: &Pattern) -> Expression {
    match pattern {
        Pattern::Wildcard(name) => bindings
            .get(name)
            .cloned()
            .unwrap_or_else(|| panic!("Unbound wildcard: {}", name)),

        Pattern::Any => panic!("Cannot apply pattern with Any - use Wildcard instead"),

        Pattern::Exact(expr) => expr.clone(),

        Pattern::Integer(n) => Expression::Integer(*n),

        Pattern::AnyInteger(opt_name) => {
            if let Some(name) = opt_name {
                bindings
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| panic!("Unbound integer wildcard: {}", name))
            } else {
                panic!("Cannot apply pattern with unnamed AnyInteger")
            }
        }

        Pattern::AnyVariable(opt_name) => {
            if let Some(name) = opt_name {
                bindings
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| panic!("Unbound variable wildcard: {}", name))
            } else {
                panic!("Cannot apply pattern with unnamed AnyVariable")
            }
        }

        Pattern::Binary(op, left, right) => Expression::Binary(
            *op,
            Box::new(apply_pattern(bindings, left)),
            Box::new(apply_pattern(bindings, right)),
        ),

        Pattern::Unary(op, operand) => {
            Expression::Unary(*op, Box::new(apply_pattern(bindings, operand)))
        }

        Pattern::Function(func, args) => Expression::Function(
            func.clone(),
            args.iter().map(|a| apply_pattern(bindings, a)).collect(),
        ),

        Pattern::Power(base, exp) => Expression::Power(
            Box::new(apply_pattern(bindings, base)),
            Box::new(apply_pattern(bindings, exp)),
        ),
    }
}

/// A transformation rule with pattern and replacement.
///
/// Rules can optionally have a condition function that must return true
/// for the rule to apply.
#[derive(Clone)]
pub struct Rule {
    /// The pattern to match against.
    pub pattern: Pattern,
    /// The replacement pattern to apply.
    pub replacement: Pattern,
    /// Optional condition that must be satisfied for the rule to apply.
    pub condition: Option<fn(&HashMap<String, Expression>) -> bool>,
    /// Optional name for debugging/logging.
    pub name: Option<String>,
}

impl Rule {
    /// Create a new rule without a condition.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::pattern::{Pattern, Rule};
    /// use mathsolver_core::ast::{Expression, BinaryOp};
    ///
    /// // Rule: x + 0 = x
    /// let rule = Rule::new(
    ///     Pattern::add(Pattern::wildcard("x"), Pattern::exact(Expression::Integer(0))),
    ///     Pattern::wildcard("x")
    /// );
    /// ```
    pub fn new(pattern: Pattern, replacement: Pattern) -> Self {
        Rule {
            pattern,
            replacement,
            condition: None,
            name: None,
        }
    }

    /// Create a rule with a condition.
    pub fn with_condition(
        pattern: Pattern,
        replacement: Pattern,
        condition: fn(&HashMap<String, Expression>) -> bool,
    ) -> Self {
        Rule {
            pattern,
            replacement,
            condition: Some(condition),
            name: None,
        }
    }

    /// Add a name to the rule for debugging.
    pub fn named(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }
}

/// Apply a transformation rule to an expression.
///
/// Returns `Some(transformed)` if the rule matches and any condition is satisfied.
/// Returns `None` if the rule doesn't apply.
///
/// # Examples
///
/// ```
/// use mathsolver_core::pattern::{Pattern, Rule, apply_rule};
/// use mathsolver_core::ast::{Expression, BinaryOp, Variable};
///
/// // Rule: x * 1 = x
/// let rule = Rule::new(
///     Pattern::mul(Pattern::wildcard("x"), Pattern::exact(Expression::Integer(1))),
///     Pattern::wildcard("x")
/// );
///
/// let y = Expression::Variable(Variable::new("y"));
/// let expr = Expression::Binary(BinaryOp::Mul, Box::new(y.clone()), Box::new(Expression::Integer(1)));
///
/// assert_eq!(apply_rule(&expr, &rule), Some(y));
/// ```
pub fn apply_rule(expr: &Expression, rule: &Rule) -> Option<Expression> {
    let bindings = match_pattern(expr, &rule.pattern)?;

    // Check condition if present
    if let Some(condition) = rule.condition {
        if !condition(&bindings) {
            return None;
        }
    }

    Some(apply_pattern(&bindings, &rule.replacement))
}

/// Apply a rule recursively to all subexpressions.
///
/// Transforms the expression by applying the rule at every level,
/// starting from the leaves and working up.
pub fn apply_rule_recursive(expr: &Expression, rule: &Rule) -> Expression {
    // First, recursively transform children
    let transformed = match expr {
        Expression::Binary(op, left, right) => Expression::Binary(
            *op,
            Box::new(apply_rule_recursive(left, rule)),
            Box::new(apply_rule_recursive(right, rule)),
        ),
        Expression::Unary(op, operand) => {
            Expression::Unary(*op, Box::new(apply_rule_recursive(operand, rule)))
        }
        Expression::Function(func, args) => Expression::Function(
            func.clone(),
            args.iter().map(|a| apply_rule_recursive(a, rule)).collect(),
        ),
        Expression::Power(base, exp) => Expression::Power(
            Box::new(apply_rule_recursive(base, rule)),
            Box::new(apply_rule_recursive(exp, rule)),
        ),
        // Leaf nodes
        _ => expr.clone(),
    };

    // Then try to apply rule at this level
    apply_rule(&transformed, rule).unwrap_or(transformed)
}

/// Apply multiple rules repeatedly until no more changes occur.
///
/// This applies rules in order, and repeats until the expression
/// reaches a fixed point (no rules apply).
///
/// # Arguments
///
/// * `expr` - The expression to transform
/// * `rules` - The rules to apply
/// * `max_iterations` - Maximum number of iteration passes
///
/// # Returns
///
/// The transformed expression after applying rules to convergence.
pub fn apply_rules_to_fixpoint(expr: &Expression, rules: &[Rule], max_iterations: usize) -> Expression {
    let mut current = expr.clone();

    for _ in 0..max_iterations {
        let mut changed = false;

        for rule in rules {
            let new_expr = apply_rule_recursive(&current, rule);
            if new_expr != current {
                current = new_expr;
                changed = true;
                break; // Restart from first rule
            }
        }

        if !changed {
            break;
        }
    }

    current
}

/// Common algebraic rules for simplification.
pub mod common_rules {
    use super::*;

    /// Create the additive identity rule: x + 0 = x
    pub fn additive_identity() -> Rule {
        Rule::new(
            Pattern::add(Pattern::wildcard("x"), Pattern::exact(Expression::Integer(0))),
            Pattern::wildcard("x"),
        )
        .named("additive_identity")
    }

    /// Create the multiplicative identity rule: x * 1 = x
    pub fn multiplicative_identity() -> Rule {
        Rule::new(
            Pattern::mul(Pattern::wildcard("x"), Pattern::exact(Expression::Integer(1))),
            Pattern::wildcard("x"),
        )
        .named("multiplicative_identity")
    }

    /// Create the multiplicative zero rule: x * 0 = 0
    pub fn multiplicative_zero() -> Rule {
        Rule::new(
            Pattern::mul(Pattern::wildcard("x"), Pattern::exact(Expression::Integer(0))),
            Pattern::exact(Expression::Integer(0)),
        )
        .named("multiplicative_zero")
    }

    /// Create the double negation rule: --x = x
    pub fn double_negation() -> Rule {
        Rule::new(
            Pattern::unary(UnaryOp::Neg, Pattern::unary(UnaryOp::Neg, Pattern::wildcard("x"))),
            Pattern::wildcard("x"),
        )
        .named("double_negation")
    }

    /// Create the power of zero rule: x^0 = 1 (for x != 0)
    pub fn power_zero() -> Rule {
        Rule::new(
            Pattern::power(Pattern::wildcard("x"), Pattern::exact(Expression::Integer(0))),
            Pattern::exact(Expression::Integer(1)),
        )
        .named("power_zero")
    }

    /// Create the power of one rule: x^1 = x
    pub fn power_one() -> Rule {
        Rule::new(
            Pattern::power(Pattern::wildcard("x"), Pattern::exact(Expression::Integer(1))),
            Pattern::wildcard("x"),
        )
        .named("power_one")
    }

    /// Get all common algebraic simplification rules.
    pub fn all() -> Vec<Rule> {
        vec![
            additive_identity(),
            multiplicative_identity(),
            multiplicative_zero(),
            double_negation(),
            power_zero(),
            power_one(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Variable;

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn add(left: Expression, right: Expression) -> Expression {
        Expression::Binary(BinaryOp::Add, Box::new(left), Box::new(right))
    }

    fn mul(left: Expression, right: Expression) -> Expression {
        Expression::Binary(BinaryOp::Mul, Box::new(left), Box::new(right))
    }

    fn power(base: Expression, exp: Expression) -> Expression {
        Expression::Power(Box::new(base), Box::new(exp))
    }

    #[test]
    fn test_wildcard_matching() {
        let pattern = Pattern::add(Pattern::wildcard("a"), Pattern::wildcard("b"));

        let expr = add(var("x"), var("y"));
        let bindings = match_pattern(&expr, &pattern).unwrap();

        assert_eq!(bindings.get("a"), Some(&var("x")));
        assert_eq!(bindings.get("b"), Some(&var("y")));
    }

    #[test]
    fn test_exact_matching() {
        let pattern = Pattern::add(Pattern::wildcard("x"), Pattern::exact(int(0)));

        let expr1 = add(var("y"), int(0));
        assert!(match_pattern(&expr1, &pattern).is_some());

        let expr2 = add(var("y"), int(1));
        assert!(match_pattern(&expr2, &pattern).is_none());
    }

    #[test]
    fn test_commutativity() {
        // Pattern: a + b
        let pattern = Pattern::add(Pattern::wildcard("a"), Pattern::wildcard("b"));

        // Should match y + x with a=y, b=x
        let expr = add(var("y"), var("x"));
        let bindings = match_pattern(&expr, &pattern).unwrap();
        assert!(bindings.contains_key("a"));
        assert!(bindings.contains_key("b"));
    }

    #[test]
    fn test_same_wildcard_must_match_same_expr() {
        // Pattern: a + a (same variable on both sides)
        let pattern = Pattern::add(Pattern::wildcard("a"), Pattern::wildcard("a"));

        // x + x should match
        let expr1 = add(var("x"), var("x"));
        assert!(match_pattern(&expr1, &pattern).is_some());

        // x + y should not match
        let expr2 = add(var("x"), var("y"));
        assert!(match_pattern(&expr2, &pattern).is_none());
    }

    #[test]
    fn test_apply_pattern() {
        let mut bindings = HashMap::new();
        bindings.insert("x".to_string(), var("y"));

        let pattern = Pattern::add(Pattern::wildcard("x"), Pattern::exact(int(1)));
        let result = apply_pattern(&bindings, &pattern);

        assert_eq!(result, add(var("y"), int(1)));
    }

    #[test]
    fn test_additive_identity_rule() {
        let rule = common_rules::additive_identity();

        let expr = add(var("x"), int(0));
        let result = apply_rule(&expr, &rule);

        assert_eq!(result, Some(var("x")));
    }

    #[test]
    fn test_multiplicative_zero_rule() {
        let rule = common_rules::multiplicative_zero();

        let expr = mul(add(var("x"), var("y")), int(0));
        let result = apply_rule(&expr, &rule);

        assert_eq!(result, Some(int(0)));
    }

    #[test]
    fn test_power_rules() {
        let zero_rule = common_rules::power_zero();
        let one_rule = common_rules::power_one();

        let expr1 = power(var("x"), int(0));
        assert_eq!(apply_rule(&expr1, &zero_rule), Some(int(1)));

        let expr2 = power(var("x"), int(1));
        assert_eq!(apply_rule(&expr2, &one_rule), Some(var("x")));
    }

    #[test]
    fn test_nested_matching() {
        // Pattern: f(g(a))
        let pattern = Pattern::function(
            Function::Sin,
            vec![Pattern::function(Function::Cos, vec![Pattern::wildcard("a")])],
        );

        // Expression: sin(cos(x))
        let expr = Expression::Function(
            Function::Sin,
            vec![Expression::Function(Function::Cos, vec![var("x")])],
        );

        let bindings = match_pattern(&expr, &pattern).unwrap();
        assert_eq!(bindings.get("a"), Some(&var("x")));
    }

    #[test]
    fn test_recursive_rule_application() {
        let rule = common_rules::additive_identity();

        // (x + 0) + (y + 0)
        let expr = add(add(var("x"), int(0)), add(var("y"), int(0)));

        // Should simplify to x + y
        let result = apply_rule_recursive(&expr, &rule);
        assert_eq!(result, add(var("x"), var("y")));
    }

    #[test]
    fn test_fixpoint_simplification() {
        let rules = common_rules::all();

        // (x * 1) + 0
        let expr = add(mul(var("x"), int(1)), int(0));

        // Should simplify to x
        let result = apply_rules_to_fixpoint(&expr, &rules, 10);
        assert_eq!(result, var("x"));
    }
}
