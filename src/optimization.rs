//! Operation ordering optimizer for manual calculation.
//!
//! This module provides tools for optimizing the order of mathematical operations
//! to facilitate manual calculation, such as with slide rules or logarithm tables.
//!
//! # Key Features
//!
//! - **Multiplicative chain detection**: Identifies sequences of multiplications/divisions
//!   that can be efficiently computed on a slide rule
//! - **Additive operation minimization**: Reorders operations to reduce additions (which
//!   require numerical work on slide rules)
//! - **Precision tracking**: Estimates precision loss through computation steps
//! - **Small angle approximations**: Uses sin(x) ≈ x for small angles
//! - **Scaled exponential forms**: Optimizes exponential computations
//!
//! # Example
//!
//! ```ignore
//! use thales::optimization::{OperationConfig, optimize_computation_order};
//! use thales::ast::Expression;
//!
//! let config = OperationConfig::default();
//! let expr = Expression::parse("a * b * c / d").unwrap();
//! let steps = analyze_expression(&expr);
//! let optimized = optimize_computation_order(&steps, &config);
//! ```

use crate::ast::{BinaryOp, Expression, Function, Variable};
use std::fmt;

/// Configuration for operation optimization.
#[derive(Debug, Clone)]
pub struct OperationConfig {
    /// Target precision in significant digits.
    pub target_precision: u32,
    /// Whether to use small angle approximations.
    pub use_small_angle_approx: bool,
    /// Threshold for small angle approximation in radians (default ~0.1).
    pub small_angle_threshold: f64,
    /// Weight for preferring multiplication over addition (higher = prefer more).
    pub prefer_multiplication_weight: f64,
    /// Weight for avoiding addition operations (higher = avoid more).
    pub avoid_addition_weight: f64,
    /// Scaling levels for exponential computations (e.g., [1.0, 0.1, 0.01]).
    pub exp_scaling_levels: Vec<f64>,
}

impl Default for OperationConfig {
    fn default() -> Self {
        OperationConfig {
            target_precision: 4, // 4 significant digits (typical slide rule precision)
            use_small_angle_approx: true,
            small_angle_threshold: 0.1, // ~5.7 degrees
            prefer_multiplication_weight: 1.0,
            avoid_addition_weight: 1.5,
            exp_scaling_levels: vec![1.0, 0.1, 0.01, 0.001],
        }
    }
}

/// Types of mathematical operations.
#[derive(Debug, Clone, PartialEq)]
pub enum OperationType {
    /// Addition: a + b
    Addition,
    /// Subtraction: a - b
    Subtraction,
    /// Multiplication: a * b
    Multiplication,
    /// Division: a / b
    Division,
    /// Reciprocal: 1/a
    Reciprocal,
    /// Square: a^2
    Square,
    /// Cube: a^3
    Cube,
    /// Square root: sqrt(a)
    SquareRoot,
    /// Cube root: cbrt(a)
    CubeRoot,
    /// Sine (argument in radians)
    Sin,
    /// Cosine (argument in radians)
    Cos,
    /// Tangent (argument in radians)
    Tan,
    /// Common logarithm (base 10)
    Log,
    /// Natural logarithm (base e)
    Ln,
    /// Exponential: e^x
    Exp,
    /// Small angle approximation: sin(x) ≈ x, tan(x) ≈ x
    SmallAngleApprox,
    /// Scaled exponential: e^(-kx) for better precision
    ScaledExp(f64),
    /// Power: a^b
    Power,
    /// Constant value (no computation needed)
    Constant,
    /// Variable lookup (no computation needed)
    Variable,
}

impl fmt::Display for OperationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OperationType::Addition => write!(f, "+"),
            OperationType::Subtraction => write!(f, "-"),
            OperationType::Multiplication => write!(f, "×"),
            OperationType::Division => write!(f, "÷"),
            OperationType::Reciprocal => write!(f, "1/x"),
            OperationType::Square => write!(f, "x²"),
            OperationType::Cube => write!(f, "x³"),
            OperationType::SquareRoot => write!(f, "√x"),
            OperationType::CubeRoot => write!(f, "∛x"),
            OperationType::Sin => write!(f, "sin"),
            OperationType::Cos => write!(f, "cos"),
            OperationType::Tan => write!(f, "tan"),
            OperationType::Log => write!(f, "log₁₀"),
            OperationType::Ln => write!(f, "ln"),
            OperationType::Exp => write!(f, "e^x"),
            OperationType::SmallAngleApprox => write!(f, "≈x"),
            OperationType::ScaledExp(k) => write!(f, "e^(-{}x)", k),
            OperationType::Power => write!(f, "x^y"),
            OperationType::Constant => write!(f, "const"),
            OperationType::Variable => write!(f, "var"),
        }
    }
}

/// A single computation step.
#[derive(Debug, Clone)]
pub struct ComputationStep {
    /// The type of operation.
    pub operation: OperationType,
    /// The operands (expressions or references to previous steps).
    pub operands: Vec<StepOperand>,
    /// The resulting expression.
    pub result: Expression,
    /// Estimated precision loss (in significant digits).
    pub precision_loss: f64,
    /// Relative manual computation effort (higher = more effort).
    pub manual_effort: u32,
    /// Description of the step for human reading.
    pub description: String,
}

/// Operand for a computation step.
#[derive(Debug, Clone)]
pub enum StepOperand {
    /// A constant value.
    Constant(f64),
    /// A variable reference.
    Variable(Variable),
    /// Reference to a previous step's result.
    StepRef(usize),
    /// An expression.
    Expr(Expression),
}

impl ComputationStep {
    /// Create a new computation step.
    pub fn new(
        operation: OperationType,
        operands: Vec<StepOperand>,
        result: Expression,
    ) -> Self {
        let (precision_loss, manual_effort) = estimate_operation_cost(&operation);
        let description = format_step_description(&operation, &operands);

        ComputationStep {
            operation,
            operands,
            result,
            precision_loss,
            manual_effort,
            description,
        }
    }
}

/// Estimate the cost of an operation in terms of precision loss and manual effort.
fn estimate_operation_cost(op: &OperationType) -> (f64, u32) {
    match op {
        OperationType::Addition => (0.1, 3), // Additions require numerical work
        OperationType::Subtraction => (0.2, 3), // Subtractions can cause cancellation
        OperationType::Multiplication => (0.01, 1), // Easy on slide rule
        OperationType::Division => (0.01, 1), // Easy on slide rule
        OperationType::Reciprocal => (0.01, 1), // Just flip the slide
        OperationType::Square => (0.02, 1), // A scale on slide rule
        OperationType::Cube => (0.03, 2), // K scale or two multiplications
        OperationType::SquareRoot => (0.02, 1), // A scale on slide rule
        OperationType::CubeRoot => (0.03, 2), // K scale or iterative
        OperationType::Sin => (0.05, 2), // S scale
        OperationType::Cos => (0.05, 2), // S scale (using sin(90-x))
        OperationType::Tan => (0.05, 2), // T scale
        OperationType::Log => (0.03, 1), // L scale
        OperationType::Ln => (0.04, 2), // L scale + conversion factor
        OperationType::Exp => (0.04, 2), // Inverse of ln
        OperationType::SmallAngleApprox => (0.0, 0), // No computation needed
        OperationType::ScaledExp(_) => (0.03, 2), // Better precision than raw exp
        OperationType::Power => (0.05, 3), // log + multiply + antilog
        OperationType::Constant => (0.0, 0), // No computation
        OperationType::Variable => (0.0, 0), // No computation
    }
}

/// Format a human-readable description of a computation step.
fn format_step_description(op: &OperationType, operands: &[StepOperand]) -> String {
    let operand_strs: Vec<String> = operands
        .iter()
        .map(|o| match o {
            StepOperand::Constant(c) => format!("{:.4}", c),
            StepOperand::Variable(v) => v.name.clone(),
            StepOperand::StepRef(i) => format!("[step {}]", i + 1),
            StepOperand::Expr(e) => format!("{}", e),
        })
        .collect();

    match op {
        OperationType::Addition => format!("{} + {}", operand_strs.get(0).unwrap_or(&"?".into()), operand_strs.get(1).unwrap_or(&"?".into())),
        OperationType::Subtraction => format!("{} - {}", operand_strs.get(0).unwrap_or(&"?".into()), operand_strs.get(1).unwrap_or(&"?".into())),
        OperationType::Multiplication => format!("{} × {}", operand_strs.get(0).unwrap_or(&"?".into()), operand_strs.get(1).unwrap_or(&"?".into())),
        OperationType::Division => format!("{} ÷ {}", operand_strs.get(0).unwrap_or(&"?".into()), operand_strs.get(1).unwrap_or(&"?".into())),
        OperationType::Reciprocal => format!("1 ÷ {}", operand_strs.get(0).unwrap_or(&"?".into())),
        OperationType::Square => format!("({})²", operand_strs.get(0).unwrap_or(&"?".into())),
        OperationType::Cube => format!("({})³", operand_strs.get(0).unwrap_or(&"?".into())),
        OperationType::SquareRoot => format!("√{}", operand_strs.get(0).unwrap_or(&"?".into())),
        OperationType::CubeRoot => format!("∛{}", operand_strs.get(0).unwrap_or(&"?".into())),
        OperationType::Sin => format!("sin({})", operand_strs.get(0).unwrap_or(&"?".into())),
        OperationType::Cos => format!("cos({})", operand_strs.get(0).unwrap_or(&"?".into())),
        OperationType::Tan => format!("tan({})", operand_strs.get(0).unwrap_or(&"?".into())),
        OperationType::Log => format!("log₁₀({})", operand_strs.get(0).unwrap_or(&"?".into())),
        OperationType::Ln => format!("ln({})", operand_strs.get(0).unwrap_or(&"?".into())),
        OperationType::Exp => format!("e^{}", operand_strs.get(0).unwrap_or(&"?".into())),
        OperationType::SmallAngleApprox => format!("{} (small angle)", operand_strs.get(0).unwrap_or(&"?".into())),
        OperationType::ScaledExp(k) => format!("e^(-{} × {})", k, operand_strs.get(0).unwrap_or(&"?".into())),
        OperationType::Power => format!("{}^{}", operand_strs.get(0).unwrap_or(&"?".into()), operand_strs.get(1).unwrap_or(&"?".into())),
        OperationType::Constant => operand_strs.get(0).cloned().unwrap_or("?".into()),
        OperationType::Variable => operand_strs.get(0).cloned().unwrap_or("?".into()),
    }
}

/// A chain of multiplicative operations.
#[derive(Debug, Clone)]
pub struct MultiplicativeChain {
    /// Factors in the numerator.
    pub numerator_factors: Vec<Expression>,
    /// Factors in the denominator.
    pub denominator_factors: Vec<Expression>,
}

impl MultiplicativeChain {
    /// Create a new empty chain.
    pub fn new() -> Self {
        MultiplicativeChain {
            numerator_factors: Vec::new(),
            denominator_factors: Vec::new(),
        }
    }

    /// Check if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.numerator_factors.is_empty() && self.denominator_factors.is_empty()
    }

    /// Get the total number of factors.
    pub fn len(&self) -> usize {
        self.numerator_factors.len() + self.denominator_factors.len()
    }

    /// Convert to a single expression.
    pub fn to_expression(&self) -> Expression {
        if self.numerator_factors.is_empty() && self.denominator_factors.is_empty() {
            return Expression::Integer(1);
        }

        let numerator = if self.numerator_factors.is_empty() {
            Expression::Integer(1)
        } else {
            self.numerator_factors.iter().skip(1).fold(
                self.numerator_factors[0].clone(),
                |acc, f| Expression::Binary(BinaryOp::Mul, Box::new(acc), Box::new(f.clone())),
            )
        };

        if self.denominator_factors.is_empty() {
            numerator
        } else {
            let denominator = self.denominator_factors.iter().skip(1).fold(
                self.denominator_factors[0].clone(),
                |acc, f| Expression::Binary(BinaryOp::Mul, Box::new(acc), Box::new(f.clone())),
            );
            Expression::Binary(BinaryOp::Div, Box::new(numerator), Box::new(denominator))
        }
    }
}

impl Default for MultiplicativeChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Find multiplicative chains in an expression.
/// Identifies sequences like a*b*c/d/e that can be done efficiently on a slide rule.
pub fn find_multiplicative_chains(expr: &Expression) -> Vec<MultiplicativeChain> {
    let mut chains = Vec::new();
    find_chains_recursive(expr, &mut chains, true);
    chains
}

fn find_chains_recursive(expr: &Expression, chains: &mut Vec<MultiplicativeChain>, _is_numerator: bool) {
    match expr {
        Expression::Binary(BinaryOp::Mul, _left, _right) => {
            // Start or continue a chain
            let mut chain = MultiplicativeChain::new();
            collect_multiplicative_factors(expr, &mut chain.numerator_factors, &mut chain.denominator_factors, true);
            if chain.len() >= 2 {
                chains.push(chain);
            }
        }
        Expression::Binary(BinaryOp::Div, _left, _right) => {
            // Start or continue a chain
            let mut chain = MultiplicativeChain::new();
            collect_multiplicative_factors(expr, &mut chain.numerator_factors, &mut chain.denominator_factors, true);
            if chain.len() >= 2 {
                chains.push(chain);
            }
        }
        Expression::Binary(BinaryOp::Add, left, right) |
        Expression::Binary(BinaryOp::Sub, left, right) => {
            // Recurse into sub-expressions
            find_chains_recursive(left, chains, true);
            find_chains_recursive(right, chains, true);
        }
        Expression::Function(_, args) => {
            for arg in args {
                find_chains_recursive(arg, chains, true);
            }
        }
        Expression::Power(base, exp) => {
            find_chains_recursive(base, chains, true);
            find_chains_recursive(exp, chains, true);
        }
        _ => {}
    }
}

fn collect_multiplicative_factors(
    expr: &Expression,
    numerator: &mut Vec<Expression>,
    denominator: &mut Vec<Expression>,
    in_numerator: bool,
) {
    match expr {
        Expression::Binary(BinaryOp::Mul, left, right) => {
            collect_multiplicative_factors(left, numerator, denominator, in_numerator);
            collect_multiplicative_factors(right, numerator, denominator, in_numerator);
        }
        Expression::Binary(BinaryOp::Div, left, right) => {
            collect_multiplicative_factors(left, numerator, denominator, in_numerator);
            collect_multiplicative_factors(right, numerator, denominator, !in_numerator);
        }
        _ => {
            if in_numerator {
                numerator.push(expr.clone());
            } else {
                denominator.push(expr.clone());
            }
        }
    }
}

/// Report on precision throughout a computation.
#[derive(Debug, Clone)]
pub struct PrecisionReport {
    /// Initial precision in significant digits.
    pub initial_precision: u32,
    /// Final precision after all operations.
    pub final_precision: u32,
    /// Precision after each step.
    pub precision_per_step: Vec<f64>,
    /// The step that caused the most precision loss (if any).
    pub bottleneck_step: Option<usize>,
    /// Warnings about precision issues.
    pub warnings: Vec<String>,
}

impl PrecisionReport {
    /// Create a new precision report.
    pub fn new(initial_precision: u32) -> Self {
        PrecisionReport {
            initial_precision,
            final_precision: initial_precision,
            precision_per_step: Vec::new(),
            bottleneck_step: None,
            warnings: Vec::new(),
        }
    }
}

/// Track precision through a sequence of computation steps.
pub fn track_precision(steps: &[ComputationStep], config: &OperationConfig) -> PrecisionReport {
    let mut report = PrecisionReport::new(config.target_precision);
    let mut current_precision = config.target_precision as f64;
    let mut max_loss = 0.0;
    let mut max_loss_step = None;

    for (i, step) in steps.iter().enumerate() {
        current_precision -= step.precision_loss;
        report.precision_per_step.push(current_precision);

        if step.precision_loss > max_loss {
            max_loss = step.precision_loss;
            max_loss_step = Some(i);
        }

        if current_precision < 1.0 {
            report.warnings.push(format!(
                "Step {}: Precision dropped below 1 significant digit",
                i + 1
            ));
        }
    }

    report.final_precision = current_precision.max(0.0) as u32;
    report.bottleneck_step = max_loss_step;

    if report.final_precision < config.target_precision / 2 {
        report.warnings.push(format!(
            "Final precision ({} digits) is less than half target ({} digits)",
            report.final_precision, config.target_precision
        ));
    }

    report
}

/// Optimize the order of computation steps.
/// Goals:
/// - Minimize additive operations (must be done numerically)
/// - Maximize multiplicative chains (can be accumulated on slide rule)
/// - Use scaled exponential forms for better precision
/// - Apply small angle approximations when valid
pub fn optimize_computation_order(
    steps: &[ComputationStep],
    _config: &OperationConfig,
) -> Vec<ComputationStep> {
    // Simple optimization: group multiplicative operations together
    let mut multiplicative: Vec<ComputationStep> = Vec::new();
    let mut additive: Vec<ComputationStep> = Vec::new();
    let mut other: Vec<ComputationStep> = Vec::new();

    for step in steps {
        match step.operation {
            OperationType::Multiplication
            | OperationType::Division
            | OperationType::Reciprocal
            | OperationType::Square
            | OperationType::SquareRoot => {
                multiplicative.push(step.clone());
            }
            OperationType::Addition | OperationType::Subtraction => {
                additive.push(step.clone());
            }
            _ => {
                other.push(step.clone());
            }
        }
    }

    // Reorder: constants/variables, then multiplicative, then other, then additive
    let mut result = Vec::new();

    // First: constants and variables (no computation)
    for step in &other {
        if matches!(step.operation, OperationType::Constant | OperationType::Variable) {
            result.push(step.clone());
        }
    }

    // Then: multiplicative operations (efficient on slide rule)
    result.extend(multiplicative);

    // Then: transcendental functions and powers
    for step in &other {
        if !matches!(step.operation, OperationType::Constant | OperationType::Variable) {
            result.push(step.clone());
        }
    }

    // Finally: additive operations (least efficient)
    result.extend(additive);

    result
}

/// A human-readable step for manual calculation.
#[derive(Debug, Clone)]
pub struct ManualStep {
    /// Human-readable instruction.
    pub instruction: String,
    /// Intermediate result (if known).
    pub intermediate_result: Option<f64>,
    /// Estimated precision at this step.
    pub precision: u32,
}

impl fmt::Display for ManualStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(result) = self.intermediate_result {
            write!(f, "{} → {:.4} ({}dp)", self.instruction, result, self.precision)
        } else {
            write!(f, "{} ({}dp)", self.instruction, self.precision)
        }
    }
}

/// Convert computation steps to human-readable manual steps.
pub fn to_manual_steps(steps: &[ComputationStep], config: &OperationConfig) -> Vec<ManualStep> {
    let mut manual_steps = Vec::new();
    let mut current_precision = config.target_precision;

    for step in steps {
        let instruction = match &step.operation {
            OperationType::Multiplication => {
                format!("Multiply: {}", step.description)
            }
            OperationType::Division => {
                format!("Divide: {}", step.description)
            }
            OperationType::Addition => {
                format!("Add (numerically): {}", step.description)
            }
            OperationType::Subtraction => {
                format!("Subtract (numerically): {}", step.description)
            }
            OperationType::Square => {
                format!("Square (use A scale): {}", step.description)
            }
            OperationType::SquareRoot => {
                format!("Square root (use A scale): {}", step.description)
            }
            OperationType::Log => {
                format!("Log base 10 (use L scale): {}", step.description)
            }
            OperationType::Ln => {
                format!("Natural log (L scale × 2.303): {}", step.description)
            }
            OperationType::Sin => {
                format!("Sine (use S scale): {}", step.description)
            }
            OperationType::Cos => {
                format!("Cosine (use S scale, 90° - x): {}", step.description)
            }
            OperationType::Tan => {
                format!("Tangent (use T scale): {}", step.description)
            }
            _ => step.description.clone(),
        };

        current_precision = (current_precision as f64 - step.precision_loss).max(0.0) as u32;

        manual_steps.push(ManualStep {
            instruction,
            intermediate_result: None, // Would need actual evaluation
            precision: current_precision,
        });
    }

    manual_steps
}

/// Analyze an expression and extract computation steps.
pub fn analyze_expression(expr: &Expression) -> Vec<ComputationStep> {
    let mut steps = Vec::new();
    analyze_recursive(expr, &mut steps);
    steps
}

fn analyze_recursive(expr: &Expression, steps: &mut Vec<ComputationStep>) {
    match expr {
        Expression::Integer(n) => {
            steps.push(ComputationStep::new(
                OperationType::Constant,
                vec![StepOperand::Constant(*n as f64)],
                expr.clone(),
            ));
        }
        Expression::Float(f) => {
            steps.push(ComputationStep::new(
                OperationType::Constant,
                vec![StepOperand::Constant(*f)],
                expr.clone(),
            ));
        }
        Expression::Variable(v) => {
            steps.push(ComputationStep::new(
                OperationType::Variable,
                vec![StepOperand::Variable(v.clone())],
                expr.clone(),
            ));
        }
        Expression::Binary(op, left, right) => {
            analyze_recursive(left, steps);
            let left_ref = steps.len() - 1;
            analyze_recursive(right, steps);
            let right_ref = steps.len() - 1;

            let op_type = match op {
                BinaryOp::Add => OperationType::Addition,
                BinaryOp::Sub => OperationType::Subtraction,
                BinaryOp::Mul => OperationType::Multiplication,
                BinaryOp::Div => OperationType::Division,
                _ => OperationType::Addition, // fallback
            };

            steps.push(ComputationStep::new(
                op_type,
                vec![StepOperand::StepRef(left_ref), StepOperand::StepRef(right_ref)],
                expr.clone(),
            ));
        }
        Expression::Power(base, exp) => {
            // Check for special cases
            if let Expression::Integer(2) = **exp {
                analyze_recursive(base, steps);
                let base_ref = steps.len() - 1;
                steps.push(ComputationStep::new(
                    OperationType::Square,
                    vec![StepOperand::StepRef(base_ref)],
                    expr.clone(),
                ));
            } else if let Expression::Float(e) = **exp {
                if (e - 0.5).abs() < 1e-10 {
                    analyze_recursive(base, steps);
                    let base_ref = steps.len() - 1;
                    steps.push(ComputationStep::new(
                        OperationType::SquareRoot,
                        vec![StepOperand::StepRef(base_ref)],
                        expr.clone(),
                    ));
                } else {
                    analyze_recursive(base, steps);
                    let base_ref = steps.len() - 1;
                    analyze_recursive(exp, steps);
                    let exp_ref = steps.len() - 1;
                    steps.push(ComputationStep::new(
                        OperationType::Power,
                        vec![StepOperand::StepRef(base_ref), StepOperand::StepRef(exp_ref)],
                        expr.clone(),
                    ));
                }
            } else {
                analyze_recursive(base, steps);
                let base_ref = steps.len() - 1;
                analyze_recursive(exp, steps);
                let exp_ref = steps.len() - 1;
                steps.push(ComputationStep::new(
                    OperationType::Power,
                    vec![StepOperand::StepRef(base_ref), StepOperand::StepRef(exp_ref)],
                    expr.clone(),
                ));
            }
        }
        Expression::Function(func, args) => {
            let mut arg_refs = Vec::new();
            for arg in args {
                analyze_recursive(arg, steps);
                arg_refs.push(steps.len() - 1);
            }

            let op_type = match func {
                Function::Sin => OperationType::Sin,
                Function::Cos => OperationType::Cos,
                Function::Tan => OperationType::Tan,
                Function::Ln => OperationType::Ln,
                Function::Log10 => OperationType::Log,
                Function::Exp => OperationType::Exp,
                Function::Sqrt => OperationType::SquareRoot,
                Function::Cbrt => OperationType::CubeRoot,
                _ => OperationType::Constant, // fallback for unsupported functions
            };

            let operands: Vec<StepOperand> = arg_refs.iter().map(|&r| StepOperand::StepRef(r)).collect();
            steps.push(ComputationStep::new(op_type, operands, expr.clone()));
        }
        _ => {
            // For other expression types, just mark as constant
            steps.push(ComputationStep::new(
                OperationType::Constant,
                vec![StepOperand::Expr(expr.clone())],
                expr.clone(),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Expression, Variable};

    #[test]
    fn test_operation_config_default() {
        let config = OperationConfig::default();
        assert_eq!(config.target_precision, 4);
        assert!(config.use_small_angle_approx);
        assert!((config.small_angle_threshold - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_find_multiplicative_chain() {
        let a = Expression::Variable(Variable::new("a"));
        let b = Expression::Variable(Variable::new("b"));
        let c = Expression::Variable(Variable::new("c"));

        // a * b * c
        let expr = Expression::Binary(
            BinaryOp::Mul,
            Box::new(Expression::Binary(
                BinaryOp::Mul,
                Box::new(a.clone()),
                Box::new(b.clone()),
            )),
            Box::new(c.clone()),
        );

        let chains = find_multiplicative_chains(&expr);
        assert!(!chains.is_empty());
        assert_eq!(chains[0].numerator_factors.len(), 3);
    }

    #[test]
    fn test_multiplicative_chain_with_division() {
        let a = Expression::Variable(Variable::new("a"));
        let b = Expression::Variable(Variable::new("b"));
        let c = Expression::Variable(Variable::new("c"));

        // a / b / c = a * (1/b) * (1/c)
        let expr = Expression::Binary(
            BinaryOp::Div,
            Box::new(Expression::Binary(
                BinaryOp::Div,
                Box::new(a.clone()),
                Box::new(b.clone()),
            )),
            Box::new(c.clone()),
        );

        let chains = find_multiplicative_chains(&expr);
        assert!(!chains.is_empty());
        assert_eq!(chains[0].numerator_factors.len(), 1);
        assert_eq!(chains[0].denominator_factors.len(), 2);
    }

    #[test]
    fn test_analyze_expression() {
        let x = Expression::Variable(Variable::new("x"));
        let two = Expression::Integer(2);

        // x * 2
        let expr = Expression::Binary(BinaryOp::Mul, Box::new(x), Box::new(two));

        let steps = analyze_expression(&expr);
        assert!(steps.len() >= 3); // x, 2, x*2
        assert!(matches!(steps.last().unwrap().operation, OperationType::Multiplication));
    }

    #[test]
    fn test_precision_tracking() {
        let config = OperationConfig::default();

        let steps = vec![
            ComputationStep::new(
                OperationType::Multiplication,
                vec![],
                Expression::Integer(1),
            ),
            ComputationStep::new(
                OperationType::Addition,
                vec![],
                Expression::Integer(1),
            ),
            ComputationStep::new(
                OperationType::Subtraction,
                vec![],
                Expression::Integer(1),
            ),
        ];

        let report = track_precision(&steps, &config);
        assert!(report.final_precision < config.target_precision);
        assert!(!report.precision_per_step.is_empty());
    }

    #[test]
    fn test_optimize_computation_order() {
        let config = OperationConfig::default();

        let steps = vec![
            ComputationStep::new(OperationType::Addition, vec![], Expression::Integer(1)),
            ComputationStep::new(OperationType::Multiplication, vec![], Expression::Integer(1)),
            ComputationStep::new(OperationType::Division, vec![], Expression::Integer(1)),
            ComputationStep::new(OperationType::Subtraction, vec![], Expression::Integer(1)),
        ];

        let optimized = optimize_computation_order(&steps, &config);

        // Multiplicative operations should come before additive
        let mul_pos = optimized.iter().position(|s| matches!(s.operation, OperationType::Multiplication)).unwrap();
        let add_pos = optimized.iter().position(|s| matches!(s.operation, OperationType::Addition)).unwrap();
        assert!(mul_pos < add_pos);
    }

    #[test]
    fn test_to_manual_steps() {
        let config = OperationConfig::default();

        let steps = vec![
            ComputationStep::new(OperationType::Multiplication, vec![], Expression::Integer(1)),
            ComputationStep::new(OperationType::Sin, vec![], Expression::Integer(1)),
        ];

        let manual = to_manual_steps(&steps, &config);
        assert_eq!(manual.len(), 2);
        assert!(manual[0].instruction.contains("Multiply"));
        assert!(manual[1].instruction.contains("Sine"));
    }

    #[test]
    fn test_multiplicative_chain_to_expression() {
        let a = Expression::Variable(Variable::new("a"));
        let b = Expression::Variable(Variable::new("b"));

        let mut chain = MultiplicativeChain::new();
        chain.numerator_factors.push(a.clone());
        chain.numerator_factors.push(b.clone());

        let expr = chain.to_expression();
        // Should be a * b
        if let Expression::Binary(BinaryOp::Mul, left, right) = expr {
            assert!(matches!(*left, Expression::Variable(_)));
            assert!(matches!(*right, Expression::Variable(_)));
        } else {
            panic!("Expected multiplication expression");
        }
    }

    #[test]
    fn test_operation_type_display() {
        assert_eq!(format!("{}", OperationType::Addition), "+");
        assert_eq!(format!("{}", OperationType::Multiplication), "×");
        assert_eq!(format!("{}", OperationType::Sin), "sin");
        assert_eq!(format!("{}", OperationType::ScaledExp(0.5)), "e^(-0.5x)");
    }
}
