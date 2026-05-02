//! Conversion from mathlex `Expression` / `Equation` into thales AST types.

use crate::ast::{BinaryOp, Equation, Expression, SymbolicConstant, UnaryOp, Variable};

use super::helpers::{match_function_name, variant_name, variant_name_from_kind};

/// Check if an expression is zero (integer 0 or float 0.0).
pub(super) fn is_zero_expr(expr: &Expression) -> bool {
    match expr {
        Expression::Integer(0) => true,
        Expression::Float(f) => *f == 0.0,
        _ => false,
    }
}

/// Convert a mathlex expression into a thales expression.
///
/// mathlex has ~60 expression variants covering calculus notation, set theory,
/// tensors, etc. Thales only handles the computational subset (arithmetic,
/// functions, variables, constants). Unsupported mathlex variants are mapped
/// to a `Function(Custom("..."), args)` or returned as an error.
pub fn convert_expression(expr: &mathlex::Expression) -> Result<Expression, String> {
    match &expr.kind {
        mathlex::ExprKind::Integer(n) => Ok(Expression::Integer(*n)),

        mathlex::ExprKind::Float(f) => Ok(Expression::Float(f.value())),

        mathlex::ExprKind::Variable(name) => Ok(Expression::Variable(Variable::new(name.as_str()))),

        mathlex::ExprKind::Constant(c) => match c {
            mathlex::MathConstant::Pi => Ok(Expression::Constant(SymbolicConstant::Pi)),
            mathlex::MathConstant::E => Ok(Expression::Constant(SymbolicConstant::E)),
            mathlex::MathConstant::I => Ok(Expression::Constant(SymbolicConstant::I)),
            other => Err(format!("unsupported constant: {:?}", other)),
        },

        mathlex::ExprKind::Unary { op, operand } => {
            let inner = convert_expression(operand)?;
            match op {
                mathlex::UnaryOp::Neg => Ok(Expression::Unary(UnaryOp::Neg, Box::new(inner))),
                mathlex::UnaryOp::Pos => Ok(inner),
                mathlex::UnaryOp::Factorial => Ok(Expression::Function(
                    crate::ast::Function::Custom("factorial".to_string()),
                    vec![inner],
                )),
                mathlex::UnaryOp::Transpose => Ok(Expression::Function(
                    crate::ast::Function::Custom("transpose".to_string()),
                    vec![inner],
                )),
            }
        }

        mathlex::ExprKind::Binary { op, left, right } => {
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

        mathlex::ExprKind::Function { name, args } => {
            let converted_args: Result<Vec<Expression>, String> =
                args.iter().map(convert_expression).collect();
            let converted_args = converted_args?;
            let func = match_function_name(name);
            Ok(Expression::Function(func, converted_args))
        }

        mathlex::ExprKind::Equation { left, right } => {
            // Equations are not expressions in thales; this path is used
            // when an equation appears inside a larger expression context.
            // We represent it as left - right (implicit "= 0" form).
            let l = convert_expression(left)?;
            let r = convert_expression(right)?;
            Ok(Expression::Binary(BinaryOp::Sub, Box::new(l), Box::new(r)))
        }

        mathlex::ExprKind::Rational {
            numerator,
            denominator,
        } => {
            let n = convert_expression(numerator)?;
            let d = convert_expression(denominator)?;
            Ok(Expression::Binary(BinaryOp::Div, Box::new(n), Box::new(d)))
        }

        mathlex::ExprKind::Complex { real, imaginary } => {
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

        mathlex::ExprKind::CrossProduct { left, right } => {
            let l = convert_expression(left)?;
            let r = convert_expression(right)?;
            Ok(Expression::Function(
                crate::ast::Function::Custom("cross_product".to_string()),
                vec![l, r],
            ))
        }
        mathlex::ExprKind::DotProduct { left, right } => {
            let l = convert_expression(left)?;
            let r = convert_expression(right)?;
            Ok(Expression::Function(
                crate::ast::Function::Custom("dot_product".to_string()),
                vec![l, r],
            ))
        }

        // Outer product
        mathlex::ExprKind::OuterProduct { left, right } => {
            let l = convert_expression(left)?;
            let r = convert_expression(right)?;
            Ok(Expression::Function(
                crate::ast::Function::Custom("outer_product".to_string()),
                vec![l, r],
            ))
        }

        // Vector and Matrix — map to Custom functions for now
        mathlex::ExprKind::Vector(elems) => {
            let converted: Result<Vec<Expression>, String> =
                elems.iter().map(convert_expression).collect();
            Ok(Expression::Function(
                crate::ast::Function::Custom("vector".to_string()),
                converted?,
            ))
        }

        mathlex::ExprKind::Matrix(rows) => {
            let nrows = rows.len();
            let ncols = rows.first().map_or(0, |r| r.len());
            let mut args = Vec::with_capacity(2 + nrows * ncols);
            args.push(Expression::Integer(nrows as i64));
            args.push(Expression::Integer(ncols as i64));
            for row in rows {
                for elem in row {
                    args.push(convert_expression(elem)?);
                }
            }
            Ok(Expression::Function(
                crate::ast::Function::Custom("matrix".to_string()),
                args,
            ))
        }

        // Calculus notation — thales handles these internally, not via parsing
        mathlex::ExprKind::Derivative { expr, var, order } => {
            let inner = convert_expression(expr)?;
            // Try symbolic differentiation first
            let mut result = inner.clone();
            for _ in 0..*order {
                result = result.differentiate(var);
            }
            // If differentiation collapses to zero/constant but the original had
            // variables (e.g., d(V)/dt where V is treated as independent of t),
            // preserve as opaque derivative wrapper so the solver can still find
            // variables inside.
            let original_vars = inner.variables();
            let result_vars = result.variables();
            if !original_vars.is_empty() && result_vars.is_empty() && is_zero_expr(&result) {
                // Preserve as opaque derivative
                Ok(Expression::Function(
                    crate::ast::Function::Custom("derivative".to_string()),
                    vec![
                        inner,
                        Expression::Variable(Variable::new(var)),
                        Expression::Integer(*order as i64),
                    ],
                ))
            } else {
                Ok(result)
            }
        }

        mathlex::ExprKind::PartialDerivative { expr, var, order } => {
            let inner = convert_expression(expr)?;
            // Try symbolic differentiation first
            let mut result = inner.clone();
            for _ in 0..*order {
                result = result.differentiate(var);
            }
            // Same opaque-preservation logic as ordinary derivatives
            let original_vars = inner.variables();
            let result_vars = result.variables();
            if !original_vars.is_empty() && result_vars.is_empty() && is_zero_expr(&result) {
                Ok(Expression::Function(
                    crate::ast::Function::Custom("derivative".to_string()),
                    vec![
                        inner,
                        Expression::Variable(Variable::new(var)),
                        Expression::Integer(*order as i64),
                    ],
                ))
            } else {
                Ok(result)
            }
        }

        mathlex::ExprKind::Gradient { expr } => {
            // Gradient is a vector calculus operation — not directly representable
            // as a scalar expression. Map to a custom function placeholder.
            let inner = convert_expression(expr)?;
            Ok(Expression::Function(
                crate::ast::Function::Custom("gradient".to_string()),
                vec![inner],
            ))
        }

        mathlex::ExprKind::Integral { integrand, var, .. } => {
            let inner = convert_expression(integrand)?;
            Ok(Expression::Function(
                crate::ast::Function::Custom("integral".to_string()),
                vec![inner, Expression::Variable(Variable::new(var.as_str()))],
            ))
        }

        mathlex::ExprKind::Sum {
            body,
            index,
            lower,
            upper,
        } => {
            let body_expr = convert_expression(body)?;
            let lower_expr = convert_expression(lower)?;
            let upper_expr = convert_expression(upper)?;
            Ok(Expression::Function(
                crate::ast::Function::Custom("sum".to_string()),
                vec![
                    body_expr,
                    Expression::Variable(Variable::new(index.as_str())),
                    lower_expr,
                    upper_expr,
                ],
            ))
        }

        mathlex::ExprKind::Product {
            body,
            index,
            lower,
            upper,
        } => {
            let body_expr = convert_expression(body)?;
            let lower_expr = convert_expression(lower)?;
            let upper_expr = convert_expression(upper)?;
            Ok(Expression::Function(
                crate::ast::Function::Custom("product".to_string()),
                vec![
                    body_expr,
                    Expression::Variable(Variable::new(index.as_str())),
                    lower_expr,
                    upper_expr,
                ],
            ))
        }

        mathlex::ExprKind::Limit { expr, var, to, .. } => {
            let inner = convert_expression(expr)?;
            let to_expr = convert_expression(to)?;
            Ok(Expression::Function(
                crate::ast::Function::Custom("limit".to_string()),
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
            variant_name_from_kind(other)
        )),
    }
}

/// Extract an equation (left, right) from a mathlex Expression::Equation.
pub fn convert_equation(expr: &mathlex::Expression) -> Result<Equation, String> {
    match &expr.kind {
        mathlex::ExprKind::Equation { left, right } => {
            let l = convert_expression(left)?;
            let r = convert_expression(right)?;
            Ok(Equation::new("", l, r))
        }
        _ => Err(format!("expected Equation, got: {}", variant_name(expr))),
    }
}
