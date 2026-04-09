//! Conversion layer between mathlex AST types and thales AST types.
//!
//! mathlex provides the parsing infrastructure; thales retains its own Expression
//! type for computation (evaluation, differentiation, simplification, solving).
//! This module bridges the two representations.

use crate::ast::{BinaryOp, Equation, Expression, Function, SymbolicConstant, UnaryOp, Variable};
use crate::ode::{FirstOrderODE, SecondOrderODE};

/// Convert a mathlex expression into a thales expression.
///
/// mathlex has ~60 expression variants covering calculus notation, set theory,
/// tensors, etc. Thales only handles the computational subset (arithmetic,
/// functions, variables, constants). Unsupported mathlex variants are mapped
/// to a `Function(Custom("..."), args)` or returned as an error.
pub fn convert_expression(expr: &mathlex::Expression) -> Result<Expression, String> {
    match expr {
        mathlex::Expression::Integer(n) => Ok(Expression::Integer(*n)),

        mathlex::Expression::Float(f) => Ok(Expression::Float(f.value())),

        mathlex::Expression::Variable(name) => {
            Ok(Expression::Variable(Variable::new(name.as_str())))
        }

        mathlex::Expression::Constant(c) => match c {
            mathlex::MathConstant::Pi => Ok(Expression::Constant(SymbolicConstant::Pi)),
            mathlex::MathConstant::E => Ok(Expression::Constant(SymbolicConstant::E)),
            mathlex::MathConstant::I => Ok(Expression::Constant(SymbolicConstant::I)),
            other => Err(format!("unsupported constant: {:?}", other)),
        },

        mathlex::Expression::Unary { op, operand } => {
            let inner = convert_expression(operand)?;
            match op {
                mathlex::UnaryOp::Neg => Ok(Expression::Unary(UnaryOp::Neg, Box::new(inner))),
                mathlex::UnaryOp::Pos => Ok(inner),
                mathlex::UnaryOp::Factorial => Ok(Expression::Function(
                    Function::Custom("factorial".to_string()),
                    vec![inner],
                )),
                mathlex::UnaryOp::Transpose => Ok(Expression::Function(
                    Function::Custom("transpose".to_string()),
                    vec![inner],
                )),
            }
        }

        mathlex::Expression::Binary { op, left, right } => {
            let l = convert_expression(left)?;
            let r = convert_expression(right)?;
            match op {
                mathlex::BinaryOp::Add => {
                    Ok(Expression::Binary(BinaryOp::Add, Box::new(l), Box::new(r)))
                }
                mathlex::BinaryOp::Sub => {
                    Ok(Expression::Binary(BinaryOp::Sub, Box::new(l), Box::new(r)))
                }
                mathlex::BinaryOp::Mul => {
                    Ok(Expression::Binary(BinaryOp::Mul, Box::new(l), Box::new(r)))
                }
                mathlex::BinaryOp::Div => {
                    Ok(Expression::Binary(BinaryOp::Div, Box::new(l), Box::new(r)))
                }
                mathlex::BinaryOp::Pow => Ok(Expression::Power(Box::new(l), Box::new(r))),
                mathlex::BinaryOp::Mod => {
                    Ok(Expression::Binary(BinaryOp::Mod, Box::new(l), Box::new(r)))
                }
                mathlex::BinaryOp::PlusMinus | mathlex::BinaryOp::MinusPlus => {
                    Err(format!("unsupported binary operator: {:?}", op))
                }
            }
        }

        mathlex::Expression::Function { name, args } => {
            let converted_args: Result<Vec<Expression>, String> =
                args.iter().map(convert_expression).collect();
            let converted_args = converted_args?;
            let func = match_function_name(name);
            Ok(Expression::Function(func, converted_args))
        }

        mathlex::Expression::Equation { left, right } => {
            // Equations are not expressions in thales; this path is used
            // when an equation appears inside a larger expression context.
            // We represent it as left - right (implicit "= 0" form).
            let l = convert_expression(left)?;
            let r = convert_expression(right)?;
            Ok(Expression::Binary(BinaryOp::Sub, Box::new(l), Box::new(r)))
        }

        mathlex::Expression::Rational {
            numerator,
            denominator,
        } => {
            let n = convert_expression(numerator)?;
            let d = convert_expression(denominator)?;
            Ok(Expression::Binary(BinaryOp::Div, Box::new(n), Box::new(d)))
        }

        mathlex::Expression::Complex { real, imaginary } => {
            let r = convert_expression(real)?;
            let im = convert_expression(imaginary)?;
            // Represent as real + imaginary * i
            let i_times_im = Expression::Binary(
                BinaryOp::Mul,
                Box::new(im),
                Box::new(Expression::Constant(SymbolicConstant::I)),
            );
            Ok(Expression::Binary(
                BinaryOp::Add,
                Box::new(r),
                Box::new(i_times_im),
            ))
        }

        // Cross product / dot product — in scalar contexts, treat as multiplication
        mathlex::Expression::CrossProduct { left, right }
        | mathlex::Expression::DotProduct { left, right } => {
            let l = convert_expression(left)?;
            let r = convert_expression(right)?;
            Ok(Expression::Binary(BinaryOp::Mul, Box::new(l), Box::new(r)))
        }

        // Outer product
        mathlex::Expression::OuterProduct { left, right } => {
            let l = convert_expression(left)?;
            let r = convert_expression(right)?;
            Ok(Expression::Function(
                Function::Custom("outer_product".to_string()),
                vec![l, r],
            ))
        }

        // Vector and Matrix — map to Custom functions for now
        mathlex::Expression::Vector(elems) => {
            let converted: Result<Vec<Expression>, String> =
                elems.iter().map(convert_expression).collect();
            Ok(Expression::Function(
                Function::Custom("vector".to_string()),
                converted?,
            ))
        }

        mathlex::Expression::Matrix(rows) => {
            // Flatten rows into a single args list with row separators
            let mut args = Vec::new();
            for row in rows {
                for elem in row {
                    args.push(convert_expression(elem)?);
                }
            }
            Ok(Expression::Function(
                Function::Custom("matrix".to_string()),
                args,
            ))
        }

        // Calculus notation — thales handles these internally, not via parsing
        mathlex::Expression::Derivative { expr, var, order } => {
            let inner = convert_expression(expr)?;
            // Apply differentiation order times
            let mut result = inner;
            for _ in 0..*order {
                result = result.differentiate(var);
            }
            Ok(result)
        }

        mathlex::Expression::PartialDerivative { expr, var, order } => {
            let inner = convert_expression(expr)?;
            // Partial derivatives use the same differentiation engine as ordinary
            // derivatives — the caller is responsible for holding other variables
            // constant (which happens automatically in symbolic differentiation).
            let mut result = inner;
            for _ in 0..*order {
                result = result.differentiate(var);
            }
            Ok(result)
        }

        mathlex::Expression::Gradient { expr } => {
            // Gradient is a vector calculus operation — not directly representable
            // as a scalar expression. Map to a custom function placeholder.
            let inner = convert_expression(expr)?;
            Ok(Expression::Function(
                Function::Custom("gradient".to_string()),
                vec![inner],
            ))
        }

        mathlex::Expression::Integral { integrand, var, .. } => {
            let inner = convert_expression(integrand)?;
            Ok(Expression::Function(
                Function::Custom("integral".to_string()),
                vec![inner, Expression::Variable(Variable::new(var.as_str()))],
            ))
        }

        mathlex::Expression::Sum {
            body,
            index,
            lower,
            upper,
        } => {
            let body_expr = convert_expression(body)?;
            let lower_expr = convert_expression(lower)?;
            let upper_expr = convert_expression(upper)?;
            Ok(Expression::Function(
                Function::Custom("sum".to_string()),
                vec![
                    body_expr,
                    Expression::Variable(Variable::new(index.as_str())),
                    lower_expr,
                    upper_expr,
                ],
            ))
        }

        mathlex::Expression::Product {
            body,
            index,
            lower,
            upper,
        } => {
            let body_expr = convert_expression(body)?;
            let lower_expr = convert_expression(lower)?;
            let upper_expr = convert_expression(upper)?;
            Ok(Expression::Function(
                Function::Custom("product".to_string()),
                vec![
                    body_expr,
                    Expression::Variable(Variable::new(index.as_str())),
                    lower_expr,
                    upper_expr,
                ],
            ))
        }

        mathlex::Expression::Limit { expr, var, to, .. } => {
            let inner = convert_expression(expr)?;
            let to_expr = convert_expression(to)?;
            Ok(Expression::Function(
                Function::Custom("limit".to_string()),
                vec![
                    inner,
                    Expression::Variable(Variable::new(var.as_str())),
                    to_expr,
                ],
            ))
        }

        // Catch-all for unsupported mathlex variants
        other => Err(format!(
            "unsupported mathlex expression type: {}",
            variant_name(other)
        )),
    }
}

/// Extract an equation (left, right) from a mathlex Expression::Equation.
pub fn convert_equation(expr: &mathlex::Expression) -> Result<Equation, String> {
    match expr {
        mathlex::Expression::Equation { left, right } => {
            let l = convert_expression(left)?;
            let r = convert_expression(right)?;
            Ok(Equation::new("", l, r))
        }
        _ => Err(format!("expected Equation, got: {}", variant_name(expr))),
    }
}

/// The result of extracting an ODE from a mathlex equation.
#[derive(Debug)]
pub enum ExtractedODE {
    /// First-order ODE: dy/dx = f(x, y)
    First(FirstOrderODE),
    /// Second-order ODE: a·y'' + b·y' + c·y = f(x)
    Second(SecondOrderODE),
}

/// Attempt to extract an ODE from a mathlex equation containing `Derivative` nodes.
///
/// Recognizes two forms:
/// 1. **First-order**: `dy/dx = rhs` or `y' = rhs` — the LHS is a single first-order
///    derivative and the RHS is converted to a thales expression.
/// 2. **Second-order constant-coefficient**: `a·d²y/dx² + b·dy/dx + c·y = f(x)` — the
///    equation is a linear combination of the dependent variable and its first and second
///    derivatives, with constant (numeric) coefficients.
///
/// Returns `None` if the equation does not match either pattern.
///
/// # Examples
///
/// ```rust
/// use thales::mathlex_bridge::try_extract_ode;
///
/// // First-order: dy/dx = x*y
/// let ml = mathlex::parse("dy/dx = x*y").unwrap();
/// let ode = try_extract_ode(&ml).unwrap();
/// assert!(matches!(ode, thales::mathlex_bridge::ExtractedODE::First(_)));
/// ```
pub fn try_extract_ode(expr: &mathlex::Expression) -> Option<ExtractedODE> {
    let (left, right) = match expr {
        mathlex::Expression::Equation { left, right } => (left.as_ref(), right.as_ref()),
        _ => return None,
    };

    // Case 1: LHS is a single derivative — first-order ODE
    if let mathlex::Expression::Derivative { expr, var, order } = left {
        if *order == 1 {
            let dep_var = extract_variable_name(expr)?;
            let rhs = convert_expression(right).ok()?;
            return Some(ExtractedODE::First(FirstOrderODE::new(&dep_var, var, rhs)));
        }
    }

    // Case 2: collect derivative terms from both sides to form a·y'' + b·y' + c·y = f(x)
    try_extract_second_order(left, right)
}

/// Extract the variable name from a simple mathlex variable expression.
fn extract_variable_name(expr: &mathlex::Expression) -> Option<String> {
    match expr {
        mathlex::Expression::Variable(name) => Some(name.clone()),
        _ => None,
    }
}

/// Accumulator for collecting constant-coefficient ODE terms.
#[derive(Default)]
struct ODETerms {
    /// Coefficient of y'' (second derivative)
    coeff_y2: f64,
    /// Coefficient of y' (first derivative)
    coeff_y1: f64,
    /// Coefficient of y (zeroth derivative)
    coeff_y0: f64,
    /// Remaining non-derivative, non-y terms (the forcing function)
    forcing_terms: Vec<Expression>,
    /// Dependent variable name (e.g. "y")
    dependent: Option<String>,
    /// Independent variable name (e.g. "x")
    independent: Option<String>,
}

impl ODETerms {
    /// Record a derivative term with its coefficient and sign.
    fn add_derivative_term(
        &mut self,
        coeff: f64,
        dep: &str,
        indep: &str,
        order: u32,
    ) -> Option<()> {
        // Verify consistent variable names
        if let Some(ref d) = self.dependent {
            if d != dep {
                return None;
            }
        } else {
            self.dependent = Some(dep.to_string());
        }
        if let Some(ref i) = self.independent {
            if i != indep {
                return None;
            }
        } else {
            self.independent = Some(indep.to_string());
        }

        match order {
            1 => self.coeff_y1 += coeff,
            2 => self.coeff_y2 += coeff,
            _ => return None, // only handle first and second order
        }
        Some(())
    }

    /// Record a dependent-variable term (y with no derivative) with its coefficient.
    fn add_y_term(&mut self, coeff: f64, dep: &str) -> Option<()> {
        if let Some(ref d) = self.dependent {
            if d != dep {
                return None;
            }
        } else {
            self.dependent = Some(dep.to_string());
        }
        self.coeff_y0 += coeff;
        Some(())
    }
}

/// Try to extract a second-order constant-coefficient ODE from a mathlex equation.
///
/// Collects terms from LHS - RHS = 0, identifying derivative terms with constant
/// coefficients. If the highest derivative order is 2, builds a `SecondOrderODE`.
/// If the highest is 1, builds a `FirstOrderODE` (for cases like `2*dy/dx + y = x`).
fn try_extract_second_order(
    left: &mathlex::Expression,
    right: &mathlex::Expression,
) -> Option<ExtractedODE> {
    let mut terms = ODETerms::default();

    // Collect terms from LHS with sign +1
    collect_ode_terms(left, 1.0, &mut terms)?;
    // Collect terms from RHS with sign -1 (moved to LHS)
    collect_ode_terms(right, -1.0, &mut terms)?;

    let dep = terms.dependent.as_ref()?;
    let indep = terms.independent.as_ref()?;

    // Build the forcing function: negate the accumulated forcing terms (they were
    // collected as LHS - RHS = 0, so forcing = -sum(non-derivative terms on LHS)
    // + sum(non-derivative terms on RHS))
    let forcing = if terms.forcing_terms.is_empty() {
        Expression::Integer(0)
    } else {
        let mut combined = terms.forcing_terms[0].clone();
        for term in &terms.forcing_terms[1..] {
            combined =
                Expression::Binary(BinaryOp::Add, Box::new(combined), Box::new(term.clone()));
        }
        // Negate: the forcing terms were collected as (LHS terms - RHS terms),
        // but the ODE form is a·y'' + b·y' + c·y = f(x), so f(x) = -forcing
        Expression::Unary(UnaryOp::Neg, Box::new(combined))
    };

    if terms.coeff_y2.abs() > 1e-15 {
        // Second-order ODE
        Some(ExtractedODE::Second(SecondOrderODE::new(
            dep,
            indep,
            terms.coeff_y2,
            terms.coeff_y1,
            terms.coeff_y0,
            forcing,
        )))
    } else if terms.coeff_y1.abs() > 1e-15 {
        // First-order linear ODE with constant coefficients: b·y' + c·y = f(x)
        // Normalize to dy/dx = (-c/b)·y + f(x)/b
        let b = terms.coeff_y1;
        let c = terms.coeff_y0;

        let mut rhs_parts: Vec<Expression> = Vec::new();

        // (-c/b) * y term
        if c.abs() > 1e-15 {
            let coeff = -c / b;
            rhs_parts.push(Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Float(coeff)),
                Box::new(Expression::Variable(Variable::new(dep))),
            ));
        }

        // f(x)/b term — forcing was already negated, so the RHS contribution is -forcing/b
        if !matches!(&forcing, Expression::Integer(0)) {
            if (b - 1.0).abs() < 1e-15 {
                // b = 1, just negate forcing
                rhs_parts.push(Expression::Unary(UnaryOp::Neg, Box::new(forcing)));
            } else {
                rhs_parts.push(Expression::Binary(
                    BinaryOp::Div,
                    Box::new(Expression::Unary(UnaryOp::Neg, Box::new(forcing))),
                    Box::new(Expression::Float(b)),
                ));
            }
        }

        let rhs = if rhs_parts.is_empty() {
            Expression::Integer(0)
        } else {
            let mut combined = rhs_parts.remove(0);
            for part in rhs_parts {
                combined = Expression::Binary(BinaryOp::Add, Box::new(combined), Box::new(part));
            }
            combined
        };

        Some(ExtractedODE::First(FirstOrderODE::new(dep, indep, rhs)))
    } else {
        None
    }
}

/// Recursively collect ODE terms from a mathlex expression, tracking sign.
///
/// Recognizes:
/// - `Derivative { expr: Variable(y), var: x, order: n }` → derivative term
/// - `Variable(y)` where y is the dependent variable → y term
/// - `Binary(Mul, const, derivative)` → scaled derivative term
/// - `Binary(Add/Sub, left, right)` → recurse into both
/// - `Unary(Neg, inner)` → recurse with flipped sign
/// - Anything else → forcing term (converted to thales expression)
fn collect_ode_terms(expr: &mathlex::Expression, sign: f64, terms: &mut ODETerms) -> Option<()> {
    match expr {
        mathlex::Expression::Binary {
            op: mathlex::BinaryOp::Add,
            left,
            right,
        } => {
            collect_ode_terms(left, sign, terms)?;
            collect_ode_terms(right, sign, terms)?;
        }

        mathlex::Expression::Binary {
            op: mathlex::BinaryOp::Sub,
            left,
            right,
        } => {
            collect_ode_terms(left, sign, terms)?;
            collect_ode_terms(right, -sign, terms)?;
        }

        mathlex::Expression::Unary {
            op: mathlex::UnaryOp::Neg,
            operand,
        } => {
            collect_ode_terms(operand, -sign, terms)?;
        }

        mathlex::Expression::Derivative { expr, var, order } => {
            let dep = extract_variable_name(expr)?;
            terms.add_derivative_term(sign, &dep, var, *order)?;
        }

        mathlex::Expression::Binary {
            op: mathlex::BinaryOp::Mul,
            left,
            right,
        } => {
            // Try coeff * derivative or derivative * coeff
            if let Some(c) = mathlex_to_f64(left) {
                if let mathlex::Expression::Derivative { expr, var, order } = right.as_ref() {
                    let dep = extract_variable_name(expr)?;
                    terms.add_derivative_term(sign * c, &dep, var, *order)?;
                    return Some(());
                }
                // Try coeff * y
                if let Some(dep) = extract_variable_name(right) {
                    if terms.dependent.as_deref() == Some(&dep) || terms.dependent.is_none() {
                        terms.add_y_term(sign * c, &dep)?;
                        return Some(());
                    }
                }
            }
            if let Some(c) = mathlex_to_f64(right) {
                if let mathlex::Expression::Derivative { expr, var, order } = left.as_ref() {
                    let dep = extract_variable_name(expr)?;
                    terms.add_derivative_term(sign * c, &dep, var, *order)?;
                    return Some(());
                }
                // Try y * coeff
                if let Some(dep) = extract_variable_name(left) {
                    if terms.dependent.as_deref() == Some(&dep) || terms.dependent.is_none() {
                        terms.add_y_term(sign * c, &dep)?;
                        return Some(());
                    }
                }
            }
            // Not a recognized ODE term — treat as forcing
            let converted = convert_expression(expr).ok()?;
            let signed = if sign < 0.0 {
                Expression::Unary(UnaryOp::Neg, Box::new(converted))
            } else {
                converted
            };
            terms.forcing_terms.push(signed);
        }

        mathlex::Expression::Variable(name) => {
            // Could be the dependent variable (y term with coeff 1)
            if terms.dependent.as_deref() == Some(name.as_str()) || terms.dependent.is_none() {
                // Only treat as y-term if we've already seen derivatives of this variable,
                // or if we haven't seen any dependent variable yet (will be validated later)
                if terms.dependent.is_some() {
                    terms.add_y_term(sign, name)?;
                } else {
                    // Can't determine if this is the dependent variable yet;
                    // treat as potential forcing term
                    let converted = convert_expression(expr).ok()?;
                    let signed = if sign < 0.0 {
                        Expression::Unary(UnaryOp::Neg, Box::new(converted))
                    } else {
                        converted
                    };
                    terms.forcing_terms.push(signed);
                }
            } else {
                // Different variable — part of forcing function
                let converted = convert_expression(expr).ok()?;
                let signed = if sign < 0.0 {
                    Expression::Unary(UnaryOp::Neg, Box::new(converted))
                } else {
                    converted
                };
                terms.forcing_terms.push(signed);
            }
        }

        mathlex::Expression::Integer(0) => {
            // Zero contributes nothing
        }

        _ => {
            // Any other expression is part of the forcing function
            let converted = convert_expression(expr).ok()?;
            let signed = if sign < 0.0 {
                Expression::Unary(UnaryOp::Neg, Box::new(converted))
            } else {
                converted
            };
            terms.forcing_terms.push(signed);
        }
    }

    Some(())
}

/// Try to extract a constant f64 value from a mathlex expression.
fn mathlex_to_f64(expr: &mathlex::Expression) -> Option<f64> {
    match expr {
        mathlex::Expression::Integer(n) => Some(*n as f64),
        mathlex::Expression::Float(f) => Some(f.value()),
        mathlex::Expression::Unary {
            op: mathlex::UnaryOp::Neg,
            operand,
        } => mathlex_to_f64(operand).map(|v| -v),
        _ => None,
    }
}

/// Map a mathlex function name string to a thales Function enum variant.
fn match_function_name(name: &str) -> Function {
    match name {
        // Trigonometric
        "sin" => Function::Sin,
        "cos" => Function::Cos,
        "tan" => Function::Tan,
        "arcsin" | "asin" => Function::Asin,
        "arccos" | "acos" => Function::Acos,
        "arctan" | "atan" => Function::Atan,
        "atan2" => Function::Atan2,

        // Hyperbolic
        "sinh" => Function::Sinh,
        "cosh" => Function::Cosh,
        "tanh" => Function::Tanh,

        // Exponential & Logarithmic
        "exp" => Function::Exp,
        "ln" => Function::Ln,
        "log" => Function::Log,
        "log2" | "lg" => Function::Log2,
        "log10" => Function::Log10,

        // Roots & Power
        "sqrt" => Function::Sqrt,
        "cbrt" => Function::Cbrt,
        "pow" => Function::Pow,

        // Rounding
        "floor" => Function::Floor,
        "ceil" => Function::Ceil,
        "round" => Function::Round,

        // Utility
        "abs" => Function::Abs,
        "sgn" | "sign" => Function::Sign,
        "min" => Function::Min,
        "max" => Function::Max,

        // Everything else → Custom
        other => Function::Custom(other.to_string()),
    }
}

/// Get a human-readable name for a mathlex expression variant (for error messages).
fn variant_name(expr: &mathlex::Expression) -> &'static str {
    match expr {
        mathlex::Expression::Integer(_) => "Integer",
        mathlex::Expression::Float(_) => "Float",
        mathlex::Expression::Variable(_) => "Variable",
        mathlex::Expression::Constant(_) => "Constant",
        mathlex::Expression::Unary { .. } => "Unary",
        mathlex::Expression::Binary { .. } => "Binary",
        mathlex::Expression::Function { .. } => "Function",
        mathlex::Expression::Equation { .. } => "Equation",
        mathlex::Expression::Rational { .. } => "Rational",
        mathlex::Expression::Complex { .. } => "Complex",
        mathlex::Expression::Vector(_) => "Vector",
        mathlex::Expression::Matrix(_) => "Matrix",
        mathlex::Expression::Derivative { .. } => "Derivative",
        mathlex::Expression::PartialDerivative { .. } => "PartialDerivative",
        mathlex::Expression::Gradient { .. } => "Gradient",
        mathlex::Expression::Integral { .. } => "Integral",
        mathlex::Expression::Sum { .. } => "Sum",
        mathlex::Expression::Product { .. } => "Product",
        mathlex::Expression::Limit { .. } => "Limit",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_integer() {
        let ml = mathlex::Expression::Integer(42);
        let result = convert_expression(&ml).unwrap();
        assert_eq!(result, Expression::Integer(42));
    }

    #[test]
    fn test_convert_variable() {
        let ml = mathlex::Expression::Variable("x".to_string());
        let result = convert_expression(&ml).unwrap();
        assert_eq!(result, Expression::Variable(Variable::new("x")));
    }

    #[test]
    fn test_convert_pi() {
        let ml = mathlex::Expression::Constant(mathlex::MathConstant::Pi);
        let result = convert_expression(&ml).unwrap();
        assert_eq!(result, Expression::Constant(SymbolicConstant::Pi));
    }

    #[test]
    fn test_convert_addition() {
        let ml = mathlex::Expression::Binary {
            op: mathlex::BinaryOp::Add,
            left: Box::new(mathlex::Expression::Integer(1)),
            right: Box::new(mathlex::Expression::Integer(2)),
        };
        let result = convert_expression(&ml).unwrap();
        assert_eq!(
            result,
            Expression::Binary(
                BinaryOp::Add,
                Box::new(Expression::Integer(1)),
                Box::new(Expression::Integer(2))
            )
        );
    }

    #[test]
    fn test_convert_power() {
        let ml = mathlex::Expression::Binary {
            op: mathlex::BinaryOp::Pow,
            left: Box::new(mathlex::Expression::Variable("x".to_string())),
            right: Box::new(mathlex::Expression::Integer(2)),
        };
        let result = convert_expression(&ml).unwrap();
        assert_eq!(
            result,
            Expression::Power(
                Box::new(Expression::Variable(Variable::new("x"))),
                Box::new(Expression::Integer(2))
            )
        );
    }

    #[test]
    fn test_convert_sin() {
        let ml = mathlex::Expression::Function {
            name: "sin".to_string(),
            args: vec![mathlex::Expression::Variable("x".to_string())],
        };
        let result = convert_expression(&ml).unwrap();
        assert_eq!(
            result,
            Expression::Function(
                Function::Sin,
                vec![Expression::Variable(Variable::new("x"))]
            )
        );
    }

    #[test]
    fn test_convert_negation() {
        let ml = mathlex::Expression::Unary {
            op: mathlex::UnaryOp::Neg,
            operand: Box::new(mathlex::Expression::Integer(5)),
        };
        let result = convert_expression(&ml).unwrap();
        assert_eq!(
            result,
            Expression::Unary(UnaryOp::Neg, Box::new(Expression::Integer(5)))
        );
    }

    #[test]
    fn test_function_name_aliases() {
        assert_eq!(match_function_name("arcsin"), Function::Asin);
        assert_eq!(match_function_name("asin"), Function::Asin);
        assert_eq!(match_function_name("sgn"), Function::Sign);
        assert_eq!(match_function_name("sign"), Function::Sign);
        assert_eq!(match_function_name("lg"), Function::Log2);
        assert_eq!(match_function_name("cbrt"), Function::Cbrt);
        assert_eq!(match_function_name("round"), Function::Round);
    }

    #[test]
    fn test_convert_equation() {
        let ml = mathlex::Expression::Equation {
            left: Box::new(mathlex::Expression::Variable("x".to_string())),
            right: Box::new(mathlex::Expression::Integer(5)),
        };
        let eq = convert_equation(&ml).unwrap();
        assert_eq!(eq.left, Expression::Variable(Variable::new("x")));
        assert_eq!(eq.right, Expression::Integer(5));
    }

    // ------------------------------------------------------------------
    // Derivative conversion
    // ------------------------------------------------------------------

    #[test]
    fn test_convert_derivative_first_order() {
        // d/dx(x^2) should evaluate to 2x
        let ml = mathlex::Expression::Derivative {
            expr: Box::new(mathlex::Expression::Binary {
                op: mathlex::BinaryOp::Pow,
                left: Box::new(mathlex::Expression::Variable("x".to_string())),
                right: Box::new(mathlex::Expression::Integer(2)),
            }),
            var: "x".to_string(),
            order: 1,
        };
        let result = convert_expression(&ml).unwrap();
        // d/dx(x^2) = 2x — check it evaluates to 2 at x=1
        let mut env = std::collections::HashMap::new();
        env.insert("x".to_string(), 1.0);
        assert_eq!(result.evaluate(&env), Some(2.0));
    }

    #[test]
    fn test_convert_derivative_second_order() {
        // d²/dx²(x³) should evaluate to 6x
        let ml = mathlex::Expression::Derivative {
            expr: Box::new(mathlex::Expression::Binary {
                op: mathlex::BinaryOp::Pow,
                left: Box::new(mathlex::Expression::Variable("x".to_string())),
                right: Box::new(mathlex::Expression::Integer(3)),
            }),
            var: "x".to_string(),
            order: 2,
        };
        let result = convert_expression(&ml).unwrap();
        let mut env = std::collections::HashMap::new();
        env.insert("x".to_string(), 2.0);
        // d²/dx²(x³) = 6x, at x=2: 12.0
        assert_eq!(result.evaluate(&env), Some(12.0));
    }

    #[test]
    fn test_convert_partial_derivative() {
        // ∂/∂x(x²·y) should evaluate to 2xy
        let ml = mathlex::Expression::PartialDerivative {
            expr: Box::new(mathlex::Expression::Binary {
                op: mathlex::BinaryOp::Mul,
                left: Box::new(mathlex::Expression::Binary {
                    op: mathlex::BinaryOp::Pow,
                    left: Box::new(mathlex::Expression::Variable("x".to_string())),
                    right: Box::new(mathlex::Expression::Integer(2)),
                }),
                right: Box::new(mathlex::Expression::Variable("y".to_string())),
            }),
            var: "x".to_string(),
            order: 1,
        };
        let result = convert_expression(&ml).unwrap();
        let mut env = std::collections::HashMap::new();
        env.insert("x".to_string(), 3.0);
        env.insert("y".to_string(), 2.0);
        // ∂/∂x(x²y) = 2xy, at x=3, y=2: 12.0
        assert_eq!(result.evaluate(&env), Some(12.0));
    }

    #[test]
    fn test_convert_gradient() {
        let ml = mathlex::Expression::Gradient {
            expr: Box::new(mathlex::Expression::Variable("f".to_string())),
        };
        let result = convert_expression(&ml).unwrap();
        assert!(matches!(
            result,
            Expression::Function(Function::Custom(ref name), _) if name == "gradient"
        ));
    }

    // ------------------------------------------------------------------
    // Derivative conversion via mathlex parser
    // ------------------------------------------------------------------

    #[test]
    fn test_parse_and_convert_leibniz_derivative() {
        let ml = mathlex::parse("dy/dx").unwrap();
        // Should be Derivative { expr: Variable("y"), var: "x", order: 1 }
        let result = convert_expression(&ml).unwrap();
        // d/dx(y) — y is treated as constant w.r.t. x, so derivative is 0
        // This is correct: without knowing y = f(x), symbolic diff gives 0
        assert_eq!(
            result.evaluate(&std::collections::HashMap::new()),
            Some(0.0)
        );
    }

    #[test]
    fn test_parse_and_convert_prime_derivative() {
        // y' has var="" (implicit independent variable)
        let ml = mathlex::parse("y'").unwrap();
        assert!(matches!(ml, mathlex::Expression::Derivative { .. }));
    }

    // ------------------------------------------------------------------
    // ODE extraction
    // ------------------------------------------------------------------

    #[test]
    fn test_extract_first_order_ode_leibniz() {
        let ml = mathlex::parse("dy/dx = x*y").unwrap();
        let ode = try_extract_ode(&ml).unwrap();
        match ode {
            ExtractedODE::First(fo) => {
                assert_eq!(fo.dependent, "y");
                assert_eq!(fo.independent, "x");
                // rhs should be x*y
                let mut env = std::collections::HashMap::new();
                env.insert("x".to_string(), 2.0);
                env.insert("y".to_string(), 3.0);
                assert_eq!(fo.rhs.evaluate(&env), Some(6.0));
            }
            _ => panic!("expected FirstOrderODE"),
        }
    }

    #[test]
    fn test_extract_first_order_ode_simple() {
        // dy/dx = y
        let ml = mathlex::parse("dy/dx = y").unwrap();
        let ode = try_extract_ode(&ml).unwrap();
        match ode {
            ExtractedODE::First(fo) => {
                assert_eq!(fo.dependent, "y");
                assert_eq!(fo.independent, "x");
            }
            _ => panic!("expected FirstOrderODE"),
        }
    }

    #[test]
    fn test_extract_second_order_ode_homogeneous() {
        // d2y/dx2 + 3*dy/dx + 2*y = 0
        let ml = mathlex::parse("d2y/dx2 + 3*dy/dx + 2*y = 0").unwrap();
        let ode = try_extract_ode(&ml).unwrap();
        match ode {
            ExtractedODE::Second(so) => {
                assert_eq!(so.dependent, "y");
                assert_eq!(so.independent, "x");
                assert!((so.a - 1.0).abs() < 1e-10);
                assert!((so.b - 3.0).abs() < 1e-10);
                assert!((so.c - 2.0).abs() < 1e-10);
                assert!(so.is_homogeneous());
            }
            _ => panic!("expected SecondOrderODE"),
        }
    }

    #[test]
    fn test_extract_second_order_ode_forced() {
        // d2y/dx2 + 2*dy/dx + y = x
        let ml = mathlex::parse("d2y/dx2 + 2*dy/dx + y = x").unwrap();
        let ode = try_extract_ode(&ml).unwrap();
        match ode {
            ExtractedODE::Second(so) => {
                assert_eq!(so.dependent, "y");
                assert_eq!(so.independent, "x");
                assert!((so.a - 1.0).abs() < 1e-10);
                assert!((so.b - 2.0).abs() < 1e-10);
                assert!((so.c - 1.0).abs() < 1e-10);
                assert!(!so.is_homogeneous());
            }
            _ => panic!("expected SecondOrderODE"),
        }
    }

    #[test]
    fn test_extract_ode_not_an_equation() {
        let ml = mathlex::parse("dy/dx").unwrap();
        assert!(try_extract_ode(&ml).is_none());
    }

    #[test]
    fn test_extract_ode_no_derivatives() {
        let ml = mathlex::parse("x + y = 0").unwrap();
        assert!(try_extract_ode(&ml).is_none());
    }
}
