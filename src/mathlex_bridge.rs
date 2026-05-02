//! Conversion layer between mathlex AST types and thales AST types.
//!
//! mathlex provides the parsing infrastructure; thales retains its own Expression
//! type for computation (evaluation, differentiation, simplification, solving).
//! This module bridges the two representations.

use crate::ast::{BinaryOp, Equation, Expression, Function, SymbolicConstant, UnaryOp, Variable};
use crate::ode::{FirstOrderODE, SecondOrderODE};

/// Check if an expression is zero (integer 0 or float 0.0).
fn is_zero_expr(expr: &Expression) -> bool {
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
                    Function::Custom("factorial".to_string()),
                    vec![inner],
                )),
                mathlex::UnaryOp::Transpose => Ok(Expression::Function(
                    Function::Custom("transpose".to_string()),
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
                Function::Custom("cross_product".to_string()),
                vec![l, r],
            ))
        }
        mathlex::ExprKind::DotProduct { left, right } => {
            let l = convert_expression(left)?;
            let r = convert_expression(right)?;
            Ok(Expression::Function(
                Function::Custom("dot_product".to_string()),
                vec![l, r],
            ))
        }

        // Outer product
        mathlex::ExprKind::OuterProduct { left, right } => {
            let l = convert_expression(left)?;
            let r = convert_expression(right)?;
            Ok(Expression::Function(
                Function::Custom("outer_product".to_string()),
                vec![l, r],
            ))
        }

        // Vector and Matrix — map to Custom functions for now
        mathlex::ExprKind::Vector(elems) => {
            let converted: Result<Vec<Expression>, String> =
                elems.iter().map(convert_expression).collect();
            Ok(Expression::Function(
                Function::Custom("vector".to_string()),
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
                Function::Custom("matrix".to_string()),
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
                    Function::Custom("derivative".to_string()),
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
                    Function::Custom("derivative".to_string()),
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
                Function::Custom("gradient".to_string()),
                vec![inner],
            ))
        }

        mathlex::ExprKind::Integral { integrand, var, .. } => {
            let inner = convert_expression(integrand)?;
            Ok(Expression::Function(
                Function::Custom("integral".to_string()),
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
                Function::Custom("sum".to_string()),
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
                Function::Custom("product".to_string()),
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
    let (left, right) = match &expr.kind {
        mathlex::ExprKind::Equation { left, right } => (left.as_ref(), right.as_ref()),
        _ => return None,
    };

    // Case 1: LHS is a single derivative — first-order ODE
    if let mathlex::ExprKind::Derivative { expr, var, order } = &left.kind {
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
    match &expr.kind {
        mathlex::ExprKind::Variable(name) => Some(name.clone()),
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
    match &expr.kind {
        mathlex::ExprKind::Binary {
            op: mathlex::BinaryOp::Add,
            left,
            right,
        } => {
            collect_ode_terms(left, sign, terms)?;
            collect_ode_terms(right, sign, terms)?;
        }

        mathlex::ExprKind::Binary {
            op: mathlex::BinaryOp::Sub,
            left,
            right,
        } => {
            collect_ode_terms(left, sign, terms)?;
            collect_ode_terms(right, -sign, terms)?;
        }

        mathlex::ExprKind::Unary {
            op: mathlex::UnaryOp::Neg,
            operand,
        } => {
            collect_ode_terms(operand, -sign, terms)?;
        }

        mathlex::ExprKind::Derivative { expr, var, order } => {
            let dep = extract_variable_name(expr)?;
            terms.add_derivative_term(sign, &dep, var, *order)?;
        }

        mathlex::ExprKind::Binary {
            op: mathlex::BinaryOp::Mul,
            left,
            right,
        } => {
            // Try coeff * derivative or derivative * coeff
            if let Some(c) = mathlex_to_f64(left) {
                if let mathlex::ExprKind::Derivative { expr, var, order } = &right.kind {
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
                if let mathlex::ExprKind::Derivative { expr, var, order } = &left.kind {
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

        mathlex::ExprKind::Variable(name) => {
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

        mathlex::ExprKind::Integer(0) => {
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
    match &expr.kind {
        mathlex::ExprKind::Integer(n) => Some(*n as f64),
        mathlex::ExprKind::Float(f) => Some(f.value()),
        mathlex::ExprKind::Unary {
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

        // Complex projections
        "re" | "Re" | "RE" => Function::Re,
        "im" | "Im" | "IM" => Function::Im,
        "conj" | "Conj" | "CONJ" => Function::Conj,

        // Everything else → Custom
        other => Function::Custom(other.to_string()),
    }
}

/// Get a human-readable name for a mathlex expression variant (for error messages).
fn variant_name(expr: &mathlex::Expression) -> &'static str {
    variant_name_from_kind(&expr.kind)
}

/// Get a human-readable name for a mathlex ExprKind variant (for error messages).
fn variant_name_from_kind(kind: &mathlex::ExprKind) -> &'static str {
    match kind {
        mathlex::ExprKind::Integer(_) => "Integer",
        mathlex::ExprKind::Float(_) => "Float",
        mathlex::ExprKind::Variable(_) => "Variable",
        mathlex::ExprKind::Constant(_) => "Constant",
        mathlex::ExprKind::Unary { .. } => "Unary",
        mathlex::ExprKind::Binary { .. } => "Binary",
        mathlex::ExprKind::Function { .. } => "Function",
        mathlex::ExprKind::Equation { .. } => "Equation",
        mathlex::ExprKind::Rational { .. } => "Rational",
        mathlex::ExprKind::Complex { .. } => "Complex",
        mathlex::ExprKind::Vector(_) => "Vector",
        mathlex::ExprKind::Matrix(_) => "Matrix",
        mathlex::ExprKind::Derivative { .. } => "Derivative",
        mathlex::ExprKind::PartialDerivative { .. } => "PartialDerivative",
        mathlex::ExprKind::Gradient { .. } => "Gradient",
        mathlex::ExprKind::Integral { .. } => "Integral",
        mathlex::ExprKind::Sum { .. } => "Sum",
        mathlex::ExprKind::Product { .. } => "Product",
        mathlex::ExprKind::Limit { .. } => "Limit",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_integer() {
        let ml = mathlex::Expression::integer(42);
        let result = convert_expression(&ml).unwrap();
        assert_eq!(result, Expression::Integer(42));
    }

    #[test]
    fn test_convert_variable() {
        let ml = mathlex::Expression::variable("x");
        let result = convert_expression(&ml).unwrap();
        assert_eq!(result, Expression::Variable(Variable::new("x")));
    }

    #[test]
    fn test_convert_pi() {
        let ml = mathlex::Expression::constant(mathlex::MathConstant::Pi);
        let result = convert_expression(&ml).unwrap();
        assert_eq!(result, Expression::Constant(SymbolicConstant::Pi));
    }

    #[test]
    fn test_convert_addition() {
        let ml = mathlex::ExprKind::Binary {
            op: mathlex::BinaryOp::Add,
            left: Box::new(mathlex::Expression::integer(1)),
            right: Box::new(mathlex::Expression::integer(2)),
        }
        .into();
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
        let ml = mathlex::ExprKind::Binary {
            op: mathlex::BinaryOp::Pow,
            left: Box::new(mathlex::Expression::variable("x")),
            right: Box::new(mathlex::Expression::integer(2)),
        }
        .into();
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
        let ml = mathlex::ExprKind::Function {
            name: "sin".to_string(),
            args: vec![mathlex::Expression::variable("x")],
        }
        .into();
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
        let ml = mathlex::ExprKind::Unary {
            op: mathlex::UnaryOp::Neg,
            operand: Box::new(mathlex::Expression::integer(5)),
        }
        .into();
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
        let ml = mathlex::ExprKind::Equation {
            left: Box::new(mathlex::Expression::variable("x")),
            right: Box::new(mathlex::Expression::integer(5)),
        }
        .into();
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
        let ml = mathlex::ExprKind::Derivative {
            expr: Box::new(
                mathlex::ExprKind::Binary {
                    op: mathlex::BinaryOp::Pow,
                    left: Box::new(mathlex::Expression::variable("x")),
                    right: Box::new(mathlex::Expression::integer(2)),
                }
                .into(),
            ),
            var: "x".to_string(),
            order: 1,
        }
        .into();
        let result = convert_expression(&ml).unwrap();
        // d/dx(x^2) = 2x — check it evaluates to 2 at x=1
        let mut env = std::collections::HashMap::new();
        env.insert("x".to_string(), 1.0);
        assert_eq!(result.evaluate(&env), Some(2.0));
    }

    #[test]
    fn test_convert_derivative_second_order() {
        // d²/dx²(x³) should evaluate to 6x
        let ml = mathlex::ExprKind::Derivative {
            expr: Box::new(
                mathlex::ExprKind::Binary {
                    op: mathlex::BinaryOp::Pow,
                    left: Box::new(mathlex::Expression::variable("x")),
                    right: Box::new(mathlex::Expression::integer(3)),
                }
                .into(),
            ),
            var: "x".to_string(),
            order: 2,
        }
        .into();
        let result = convert_expression(&ml).unwrap();
        let mut env = std::collections::HashMap::new();
        env.insert("x".to_string(), 2.0);
        // d²/dx²(x³) = 6x, at x=2: 12.0
        assert_eq!(result.evaluate(&env), Some(12.0));
    }

    #[test]
    fn test_convert_partial_derivative() {
        // ∂/∂x(x²·y) should evaluate to 2xy
        let ml = mathlex::ExprKind::PartialDerivative {
            expr: Box::new(
                mathlex::ExprKind::Binary {
                    op: mathlex::BinaryOp::Mul,
                    left: Box::new(
                        mathlex::ExprKind::Binary {
                            op: mathlex::BinaryOp::Pow,
                            left: Box::new(mathlex::Expression::variable("x")),
                            right: Box::new(mathlex::Expression::integer(2)),
                        }
                        .into(),
                    ),
                    right: Box::new(mathlex::Expression::variable("y")),
                }
                .into(),
            ),
            var: "x".to_string(),
            order: 1,
        }
        .into();
        let result = convert_expression(&ml).unwrap();
        let mut env = std::collections::HashMap::new();
        env.insert("x".to_string(), 3.0);
        env.insert("y".to_string(), 2.0);
        // ∂/∂x(x²y) = 2xy, at x=3, y=2: 12.0
        assert_eq!(result.evaluate(&env), Some(12.0));
    }

    #[test]
    fn test_convert_gradient() {
        let ml = mathlex::ExprKind::Gradient {
            expr: Box::new(mathlex::Expression::variable("f")),
        }
        .into();
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
        // d/dx(y) — y is independent of x, so symbolic diff would give 0.
        // But since that would lose variable "y", the bridge preserves
        // it as an opaque derivative wrapper so the solver can find y.
        assert!(result.contains_variable("y"));
        assert!(matches!(
            result,
            Expression::Function(Function::Custom(_), _)
        ));
    }

    #[test]
    fn test_parse_and_convert_prime_derivative() {
        // y' has var="" (implicit independent variable)
        let ml = mathlex::parse("y'").unwrap();
        assert!(matches!(&ml.kind, mathlex::ExprKind::Derivative { .. }));
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
                env.insert(crate::numeric::SymbolId::intern("x"), 2.0);
                env.insert(crate::numeric::SymbolId::intern("y"), 3.0);
                assert_eq!(
                    crate::numeric::evaluation::evaluate(&fo.rhs_arc(), &env),
                    Some(6.0)
                );
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

    // ==================================================================
    // LaTeX variants — each test above has a LaTeX counterpart ensuring
    // both parsers produce equivalent results through the bridge.
    // ==================================================================

    #[test]
    fn test_convert_derivative_first_order_latex() {
        // \frac{d}{dx}(x^2) should evaluate to 2x
        let ml = mathlex::parse_latex(r#"\frac{d}{dx}(x^2)"#).unwrap();
        let result = convert_expression(&ml).unwrap();
        let mut env = std::collections::HashMap::new();
        env.insert("x".to_string(), 1.0);
        assert_eq!(result.evaluate(&env), Some(2.0));
    }

    #[test]
    fn test_convert_derivative_second_order_latex() {
        // \frac{d^2}{dx^2}(x^3) should evaluate to 6x
        let ml = mathlex::parse_latex(r#"\frac{d^2}{dx^2}(x^3)"#).unwrap();
        let result = convert_expression(&ml).unwrap();
        let mut env = std::collections::HashMap::new();
        env.insert("x".to_string(), 2.0);
        assert_eq!(result.evaluate(&env), Some(12.0));
    }

    #[test]
    fn test_convert_partial_derivative_latex() {
        // \frac{\partial}{\partial x}(x^2 \cdot y) should evaluate to 2xy
        let ml = mathlex::parse_latex(r#"\frac{\partial}{\partial x}(x^2 \cdot y)"#).unwrap();
        let result = convert_expression(&ml).unwrap();
        let mut env = std::collections::HashMap::new();
        env.insert("x".to_string(), 3.0);
        env.insert("y".to_string(), 2.0);
        assert_eq!(result.evaluate(&env), Some(12.0));
    }

    #[test]
    fn test_convert_gradient_latex() {
        let ml = mathlex::parse_latex(r#"\nabla f"#).unwrap();
        let result = convert_expression(&ml).unwrap();
        assert!(matches!(
            result,
            Expression::Function(Function::Custom(ref name), _) if name == "gradient"
        ));
    }

    #[test]
    fn test_parse_and_convert_derivative_latex() {
        // \frac{d}{dx}(y) — y is independent of x; preserved as opaque derivative
        let ml = mathlex::parse_latex(r#"\frac{d}{dx}(y)"#).unwrap();
        let result = convert_expression(&ml).unwrap();
        assert!(result.contains_variable("y"));
        assert!(matches!(
            result,
            Expression::Function(Function::Custom(_), _)
        ));
    }

    #[test]
    fn test_extract_first_order_ode_latex() {
        // \frac{d}{dx}(y) = x \cdot y
        let ml = mathlex::parse_latex(r#"\frac{d}{dx}(y) = x \cdot y"#).unwrap();
        let ode = try_extract_ode(&ml).unwrap();
        match ode {
            ExtractedODE::First(fo) => {
                assert_eq!(fo.dependent, "y");
                assert_eq!(fo.independent, "x");
                let mut env = std::collections::HashMap::new();
                env.insert(crate::numeric::SymbolId::intern("x"), 2.0);
                env.insert(crate::numeric::SymbolId::intern("y"), 3.0);
                assert_eq!(
                    crate::numeric::evaluation::evaluate(&fo.rhs_arc(), &env),
                    Some(6.0)
                );
            }
            _ => panic!("expected FirstOrderODE"),
        }
    }

    #[test]
    fn test_extract_first_order_ode_simple_latex() {
        // \frac{d}{dx}(y) = y
        let ml = mathlex::parse_latex(r#"\frac{d}{dx}(y) = y"#).unwrap();
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
    fn test_extract_second_order_ode_homogeneous_latex() {
        // \frac{d^2}{dx^2}(y) + 3\frac{d}{dx}(y) + 2y = 0
        let ml =
            mathlex::parse_latex(r#"\frac{d^2}{dx^2}(y) + 3\frac{d}{dx}(y) + 2y = 0"#).unwrap();
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
    fn test_extract_second_order_ode_forced_latex() {
        // \frac{d^2}{dx^2}(y) + 2\frac{d}{dx}(y) + y = x
        let ml = mathlex::parse_latex(r#"\frac{d^2}{dx^2}(y) + 2\frac{d}{dx}(y) + y = x"#).unwrap();
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
    fn test_extract_ode_not_an_equation_latex() {
        let ml = mathlex::parse_latex(r#"\frac{d}{dx}(y)"#).unwrap();
        assert!(try_extract_ode(&ml).is_none());
    }

    #[test]
    fn test_extract_ode_no_derivatives_latex() {
        let ml = mathlex::parse_latex(r#"x + y = 0"#).unwrap();
        assert!(try_extract_ode(&ml).is_none());
    }

    // ── match_function_name for Re/Im/Conj ───────────────────────────────

    #[test]
    fn test_match_function_name_re_lowercase() {
        use crate::ast::Function;
        assert!(matches!(match_function_name("re"), Function::Re));
    }

    #[test]
    fn test_match_function_name_re_capitalized() {
        use crate::ast::Function;
        assert!(matches!(match_function_name("Re"), Function::Re));
    }

    #[test]
    fn test_match_function_name_im_lowercase() {
        use crate::ast::Function;
        assert!(matches!(match_function_name("im"), Function::Im));
    }

    #[test]
    fn test_match_function_name_im_capitalized() {
        use crate::ast::Function;
        assert!(matches!(match_function_name("Im"), Function::Im));
    }

    #[test]
    fn test_match_function_name_conj() {
        use crate::ast::Function;
        assert!(matches!(match_function_name("conj"), Function::Conj));
        assert!(matches!(match_function_name("Conj"), Function::Conj));
    }

    #[test]
    fn test_match_function_name_conj_uppercase() {
        use crate::ast::Function;
        assert!(matches!(match_function_name("CONJ"), Function::Conj));
    }
}
