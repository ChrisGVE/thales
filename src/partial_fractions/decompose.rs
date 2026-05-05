//! Partial fraction decomposition algorithm.

use crate::ast::{BinaryOp, Expression, UnaryOp, Variable};
use std::collections::HashMap;

use super::polynomial::{
    evaluate_polynomial, extract_coefficients, find_polynomial_roots, get_polynomial_degree,
    is_irreducible_quadratic, is_polynomial,
};
use super::types::{DecomposeError, PartialFractionResult, PartialFractionTerm};

pub fn decompose(
    numerator: &Expression,
    denominator: &Expression,
    var: &Variable,
) -> Result<PartialFractionResult, DecomposeError> {
    let var_name = &var.name;
    let mut steps = Vec::new();

    // Check that both are polynomials
    if !is_polynomial(numerator, var_name) {
        return Err(DecomposeError::NotRational(
            "Numerator is not a polynomial".to_string(),
        ));
    }
    if !is_polynomial(denominator, var_name) {
        return Err(DecomposeError::NotRational(
            "Denominator is not a polynomial".to_string(),
        ));
    }

    steps.push("Verified expression is a rational function".to_string());

    // Get degrees
    let num_degree = get_polynomial_degree(numerator, var_name).unwrap_or(0);
    let denom_degree = get_polynomial_degree(denominator, var_name).unwrap_or(0);

    // Extract coefficients
    let num_coeffs = extract_coefficients(numerator, var_name).ok_or_else(|| {
        DecomposeError::NotRational("Cannot extract numerator coefficients".to_string())
    })?;
    let denom_coeffs = extract_coefficients(denominator, var_name).ok_or_else(|| {
        DecomposeError::NotRational("Cannot extract denominator coefficients".to_string())
    })?;

    steps.push(format!(
        "Numerator degree: {}, Denominator degree: {}",
        num_degree, denom_degree
    ));

    // Handle improper fractions (numerator degree >= denominator degree)
    let mut terms = Vec::new();
    let working_num_coeffs;
    if num_degree >= denom_degree {
        steps.push("Improper fraction: performing polynomial division".to_string());
        let (quotient, remainder) = polynomial_division(&num_coeffs, &denom_coeffs);

        // Add polynomial term
        if !quotient.is_empty() {
            let poly_expr = coefficients_to_expression(&quotient, var_name);
            terms.push(PartialFractionTerm::Polynomial(poly_expr));
            steps.push(format!("Polynomial quotient extracted"));
        }

        working_num_coeffs = remainder;
    } else {
        working_num_coeffs = num_coeffs.clone();
    }

    // Find roots of denominator
    let roots = find_polynomial_roots(&denom_coeffs);
    steps.push(format!("Found {} real roots in denominator", roots.len()));

    if roots.is_empty() && denom_degree > 0 {
        // Denominator might be irreducible quadratic or higher
        if denom_degree == 2 {
            let a = *denom_coeffs.get(&2).unwrap_or(&1.0);
            let b = *denom_coeffs.get(&1).unwrap_or(&0.0);
            let c = *denom_coeffs.get(&0).unwrap_or(&0.0);

            // Normalize: x² + (b/a)x + (c/a)
            let p = b / a;
            let q = c / a;

            if is_irreducible_quadratic(p, q) {
                steps.push("Denominator is an irreducible quadratic".to_string());

                // For Ax+B over irreducible quadratic, solve for A and B
                // From the numerator coefficients
                let num_a = *working_num_coeffs.get(&1).unwrap_or(&0.0) / a;
                let num_b = *working_num_coeffs.get(&0).unwrap_or(&0.0) / a;

                terms.push(PartialFractionTerm::Quadratic {
                    a_coeff: num_a,
                    b_coeff: num_b,
                    p,
                    q,
                    power: 1,
                });

                return Ok(PartialFractionResult {
                    terms,
                    variable: var_name.clone(),
                    steps,
                });
            }
        }

        return Err(DecomposeError::CannotFactor(
            "Cannot factor denominator into linear/quadratic factors".to_string(),
        ));
    }

    // Use cover-up method for simple linear factors
    for (root, multiplicity) in &roots {
        steps.push(format!(
            "Processing root {} with multiplicity {}",
            root, multiplicity
        ));

        for power in 1..=*multiplicity {
            // Cover-up method: substitute x = root into numerator / (remaining factors)
            let coeff = compute_coefficient_cover_up(
                &working_num_coeffs,
                &denom_coeffs,
                *root,
                power,
                &roots,
            );

            terms.push(PartialFractionTerm::Linear {
                coefficient: coeff,
                root: *root,
                power,
            });

            steps.push(format!(
                "Coefficient for 1/(x-{})^{}: {}",
                root, power, coeff
            ));
        }
    }

    Ok(PartialFractionResult {
        terms,
        variable: var_name.clone(),
        steps,
    })
}

/// Perform polynomial long division.
fn polynomial_division(
    num: &HashMap<i32, f64>,
    denom: &HashMap<i32, f64>,
) -> (HashMap<i32, f64>, HashMap<i32, f64>) {
    let mut quotient = HashMap::new();
    let mut remainder = num.clone();

    let denom_degree = denom.keys().copied().max().unwrap_or(0);
    let denom_leading = *denom.get(&denom_degree).unwrap_or(&1.0);

    loop {
        let rem_degree = remainder.keys().copied().max().unwrap_or(-1);
        if rem_degree < denom_degree || rem_degree < 0 {
            break;
        }

        let rem_leading = *remainder.get(&rem_degree).unwrap_or(&0.0);
        let factor = rem_leading / denom_leading;
        let power_diff = rem_degree - denom_degree;

        *quotient.entry(power_diff).or_insert(0.0) += factor;

        // Subtract factor * denom from remainder
        for (pow, coeff) in denom.iter() {
            let new_pow = pow + power_diff;
            *remainder.entry(new_pow).or_insert(0.0) -= factor * coeff;
        }

        // Clean up near-zero coefficients
        remainder.retain(|_, v| v.abs() > 1e-12);
    }

    (quotient, remainder)
}

/// Convert coefficient map back to expression.
pub(super) fn coefficients_to_expression(coeffs: &HashMap<i32, f64>, var: &str) -> Expression {
    if coeffs.is_empty() {
        return Expression::Integer(0);
    }

    let mut terms: Vec<Expression> = Vec::new();

    let mut powers: Vec<i32> = coeffs.keys().copied().collect();
    powers.sort_by(|a, b| b.cmp(a)); // Descending order

    for pow in powers {
        let coeff = *coeffs.get(&pow).unwrap_or(&0.0);
        if coeff.abs() < 1e-12 {
            continue;
        }

        let term = if pow == 0 {
            float_to_expression(coeff)
        } else if pow == 1 {
            if (coeff - 1.0).abs() < 1e-12 {
                Expression::Variable(Variable::new(var))
            } else if (coeff + 1.0).abs() < 1e-12 {
                Expression::Unary(
                    UnaryOp::Neg,
                    Box::new(Expression::Variable(Variable::new(var))),
                )
            } else {
                Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(float_to_expression(coeff)),
                    Box::new(Expression::Variable(Variable::new(var))),
                )
            }
        } else {
            let var_power = Expression::Power(
                Box::new(Expression::Variable(Variable::new(var))),
                Box::new(Expression::Integer(pow as i64)),
            );
            if (coeff - 1.0).abs() < 1e-12 {
                var_power
            } else if (coeff + 1.0).abs() < 1e-12 {
                Expression::Unary(UnaryOp::Neg, Box::new(var_power))
            } else {
                Expression::Binary(
                    BinaryOp::Mul,
                    Box::new(float_to_expression(coeff)),
                    Box::new(var_power),
                )
            }
        };

        terms.push(term);
    }

    if terms.is_empty() {
        return Expression::Integer(0);
    }

    let mut result = terms.remove(0);
    for term in terms {
        // Check if term is negative
        if let Expression::Unary(UnaryOp::Neg, inner) = &term {
            result = Expression::Binary(BinaryOp::Sub, Box::new(result), inner.clone());
        } else {
            result = Expression::Binary(BinaryOp::Add, Box::new(result), Box::new(term));
        }
    }

    result
}

/// Convert a float to an expression.
pub(super) fn float_to_expression(f: f64) -> Expression {
    if f < 0.0 {
        Expression::Unary(UnaryOp::Neg, Box::new(float_to_expression(-f)))
    } else if (f.round() - f).abs() < 1e-12 {
        Expression::Integer(f.round() as i64)
    } else {
        Expression::Float(f)
    }
}

/// Compute coefficient using cover-up method.
fn compute_coefficient_cover_up(
    num_coeffs: &HashMap<i32, f64>,
    denom_coeffs: &HashMap<i32, f64>,
    root: f64,
    power: u32,
    all_roots: &[(f64, u32)],
) -> f64 {
    // For simple case (power = 1), use direct cover-up:
    // A = f(root) where f(x) = num(x) / (denom(x) / (x-root))

    let num_val = evaluate_polynomial(num_coeffs, root);

    // Compute denom / (x - root)^power
    // By evaluating the derivative if needed
    if power == 1 {
        // Simple cover-up
        let mut denom_without_root = 1.0;
        for (r, mult) in all_roots {
            if (*r - root).abs() > 1e-12 {
                denom_without_root *= (root - r).powi(*mult as i32);
            }
        }

        // Also account for leading coefficient
        let denom_degree = denom_coeffs.keys().copied().max().unwrap_or(0);
        let leading = *denom_coeffs.get(&denom_degree).unwrap_or(&1.0);
        denom_without_root *= leading;

        if denom_without_root.abs() < 1e-15 {
            0.0
        } else {
            num_val / denom_without_root
        }
    } else {
        // For repeated roots, need to use differentiation approach
        // This is more complex; for now, use numerical approximation
        let h = 1e-6;
        let coeff = (evaluate_polynomial(num_coeffs, root + h)
            / evaluate_polynomial_without_root(denom_coeffs, root + h, root, power))
        .abs();
        coeff
    }
}

/// Evaluate polynomial with (x-root)^power factored out.
fn evaluate_polynomial_without_root(
    coeffs: &HashMap<i32, f64>,
    x: f64,
    root: f64,
    power: u32,
) -> f64 {
    let full = evaluate_polynomial(coeffs, x);
    let factor = (x - root).powi(power as i32);
    if factor.abs() < 1e-15 {
        // Use L'Hôpital-like approach
        1.0
    } else {
        full / factor
    }
}
