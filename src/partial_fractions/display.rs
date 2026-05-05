//! Display and LaTeX rendering for partial fraction types.

use crate::ast::{BinaryOp, Expression, Function, Variable};

use super::decompose::float_to_expression;
use super::types::{PartialFractionResult, PartialFractionTerm};

impl PartialFractionTerm {
    /// Convert this term to an Expression.
    pub fn to_expression(&self, var: &str) -> Expression {
        match self {
            PartialFractionTerm::Linear {
                coefficient,
                root,
                power,
            } => {
                let x_minus_a = if *root >= 0.0 {
                    Expression::Binary(
                        BinaryOp::Sub,
                        Box::new(Expression::Variable(Variable::new(var))),
                        Box::new(float_to_expression(*root)),
                    )
                } else {
                    Expression::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::Variable(Variable::new(var))),
                        Box::new(float_to_expression(-*root)),
                    )
                };

                let denom = if *power == 1 {
                    x_minus_a
                } else {
                    Expression::Power(
                        Box::new(x_minus_a),
                        Box::new(Expression::Integer(*power as i64)),
                    )
                };

                Expression::Binary(
                    BinaryOp::Div,
                    Box::new(float_to_expression(*coefficient)),
                    Box::new(denom),
                )
            }
            PartialFractionTerm::Quadratic {
                a_coeff,
                b_coeff,
                p,
                q,
                power,
            } => {
                // (Ax + B) / (x² + px + q)^n
                let x = Expression::Variable(Variable::new(var));

                // Numerator: Ax + B
                let ax = Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(float_to_expression(*a_coeff)),
                    Box::new(x.clone()),
                );
                let numerator = Expression::Binary(
                    BinaryOp::Add,
                    Box::new(ax),
                    Box::new(float_to_expression(*b_coeff)),
                );

                // Denominator: x² + px + q
                let x_squared =
                    Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(2)));
                let px = Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(float_to_expression(*p)),
                    Box::new(x.clone()),
                );
                let quad = Expression::Binary(
                    BinaryOp::Add,
                    Box::new(Expression::Binary(
                        BinaryOp::Add,
                        Box::new(x_squared),
                        Box::new(px),
                    )),
                    Box::new(float_to_expression(*q)),
                );

                let denom = if *power == 1 {
                    quad
                } else {
                    Expression::Power(Box::new(quad), Box::new(Expression::Integer(*power as i64)))
                };

                Expression::Binary(BinaryOp::Div, Box::new(numerator), Box::new(denom))
            }
            PartialFractionTerm::Polynomial(expr) => expr.clone(),
        }
    }

    /// Integrate this partial fraction term.
    pub fn integrate(&self, var: &str) -> Expression {
        match self {
            PartialFractionTerm::Linear {
                coefficient,
                root,
                power,
            } => {
                if *power == 1 {
                    // ∫ A/(x-a) dx = A * ln|x-a|
                    let x_minus_a = if *root >= 0.0 {
                        Expression::Binary(
                            BinaryOp::Sub,
                            Box::new(Expression::Variable(Variable::new(var))),
                            Box::new(float_to_expression(*root)),
                        )
                    } else {
                        Expression::Binary(
                            BinaryOp::Add,
                            Box::new(Expression::Variable(Variable::new(var))),
                            Box::new(float_to_expression(-*root)),
                        )
                    };

                    let ln_term = Expression::Function(
                        Function::Ln,
                        vec![Expression::Function(Function::Abs, vec![x_minus_a])],
                    );

                    if (*coefficient - 1.0).abs() < 1e-12 {
                        ln_term
                    } else {
                        Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(float_to_expression(*coefficient)),
                            Box::new(ln_term),
                        )
                    }
                } else {
                    // ∫ A/(x-a)^n dx = -A/((n-1)(x-a)^(n-1)) for n > 1
                    let x_minus_a = if *root >= 0.0 {
                        Expression::Binary(
                            BinaryOp::Sub,
                            Box::new(Expression::Variable(Variable::new(var))),
                            Box::new(float_to_expression(*root)),
                        )
                    } else {
                        Expression::Binary(
                            BinaryOp::Add,
                            Box::new(Expression::Variable(Variable::new(var))),
                            Box::new(float_to_expression(-*root)),
                        )
                    };

                    let new_coeff = -*coefficient / ((*power - 1) as f64);
                    let new_power = *power - 1;

                    let denom = if new_power == 1 {
                        x_minus_a
                    } else {
                        Expression::Power(
                            Box::new(x_minus_a),
                            Box::new(Expression::Integer(new_power as i64)),
                        )
                    };

                    Expression::Binary(
                        BinaryOp::Div,
                        Box::new(float_to_expression(new_coeff)),
                        Box::new(denom),
                    )
                }
            }
            PartialFractionTerm::Quadratic {
                a_coeff,
                b_coeff,
                p,
                q,
                power,
            } => {
                if *power == 1 {
                    // Split (Ax+B)/(x²+px+q) into A/2 * 2x/(x²+px+q) + (B - Ap/2) * 1/(x²+px+q)
                    // First part: A/2 * ln|x²+px+q|
                    // Second part: (B - Ap/2) * arctan form

                    let x = Expression::Variable(Variable::new(var));

                    // First part: (A/2) * ln|x² + px + q|
                    let x_squared =
                        Expression::Power(Box::new(x.clone()), Box::new(Expression::Integer(2)));
                    let px = Expression::Binary(
                        BinaryOp::Mul,
                        Box::new(float_to_expression(*p)),
                        Box::new(x.clone()),
                    );
                    let quad = Expression::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::Binary(
                            BinaryOp::Add,
                            Box::new(x_squared),
                            Box::new(px),
                        )),
                        Box::new(float_to_expression(*q)),
                    );

                    let ln_part = Expression::Binary(
                        BinaryOp::Mul,
                        Box::new(float_to_expression(*a_coeff / 2.0)),
                        Box::new(Expression::Function(
                            Function::Ln,
                            vec![Expression::Function(Function::Abs, vec![quad])],
                        )),
                    );

                    // Second part: arctan
                    // Complete the square: x² + px + q = (x + p/2)² + (q - p²/4)
                    let h = p / 2.0;
                    let k_squared = q - p * p / 4.0;

                    if k_squared > 0.0 {
                        let k = k_squared.sqrt();
                        let c = b_coeff - a_coeff * h;

                        // c/k * arctan((x + h)/k)
                        let x_plus_h = Expression::Binary(
                            BinaryOp::Add,
                            Box::new(x.clone()),
                            Box::new(float_to_expression(h)),
                        );
                        let arg = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(x_plus_h),
                            Box::new(float_to_expression(k)),
                        );
                        let arctan_part = Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(float_to_expression(c / k)),
                            Box::new(Expression::Function(Function::Atan, vec![arg])),
                        );

                        Expression::Binary(BinaryOp::Add, Box::new(ln_part), Box::new(arctan_part))
                    } else {
                        // This shouldn't happen for irreducible quadratic
                        ln_part
                    }
                } else {
                    // Higher powers are more complex; return the term as-is for now
                    self.to_expression(var)
                }
            }
            PartialFractionTerm::Polynomial(expr) => {
                // Integrate the polynomial term by term
                // This is a simplified implementation
                crate::integration::integrate(expr, var).unwrap_or_else(|_| expr.clone())
            }
        }
    }
}

impl PartialFractionResult {
    /// Convert the decomposition back to an expression (sum of terms).
    pub fn to_expression(&self) -> Expression {
        if self.terms.is_empty() {
            return Expression::Integer(0);
        }

        let mut result = self.terms[0].to_expression(&self.variable);
        for term in self.terms.iter().skip(1) {
            result = Expression::Binary(
                BinaryOp::Add,
                Box::new(result),
                Box::new(term.to_expression(&self.variable)),
            );
        }
        result
    }

    /// Integrate all terms and return the combined result.
    pub fn integrate(&self) -> Expression {
        if self.terms.is_empty() {
            return Expression::Integer(0);
        }

        let mut result = self.terms[0].integrate(&self.variable);
        for term in self.terms.iter().skip(1) {
            result = Expression::Binary(
                BinaryOp::Add,
                Box::new(result),
                Box::new(term.integrate(&self.variable)),
            );
        }
        result
    }
}
