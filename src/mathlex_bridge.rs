//! Conversion layer between mathlex AST types and thales AST types.
//!
//! mathlex provides the parsing infrastructure; thales retains its own Expression
//! type for computation (evaluation, differentiation, simplification, solving).
//! This module bridges the two representations.

use crate::ast::{BinaryOp, Equation, Expression, Function, SymbolicConstant, UnaryOp, Variable};

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
}
