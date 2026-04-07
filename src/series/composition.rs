//! Series composition, reversion, and arithmetic operations.

use crate::ast::{BinaryOp, Expression};
use std::collections::HashMap;
use std::ops::{Add, Div, Mul, Sub};

use super::{Series, SeriesError, SeriesResult, SeriesTerm};

// Series arithmetic operations using std::ops traits

impl Add for Series {
    type Output = SeriesResult<Series>;

    fn add(self, rhs: Series) -> SeriesResult<Series> {
        // Check that centers match
        if self.center != rhs.center {
            return Err(SeriesError::InvalidCenter(
                "Cannot add series with different centers".into(),
            ));
        }
        if self.variable.name != rhs.variable.name {
            return Err(SeriesError::CannotExpand(
                "Cannot add series with different variables".into(),
            ));
        }

        let min_order = self.order.min(rhs.order);
        let mut result = Series::new(self.variable.clone(), self.center.clone(), min_order);

        // Collect coefficients by power
        let mut coeffs: HashMap<u32, Expression> = HashMap::new();

        for term in &self.terms {
            if term.power <= min_order {
                coeffs.insert(term.power, term.coefficient.clone());
            }
        }

        for term in &rhs.terms {
            if term.power <= min_order {
                let coeff = coeffs.entry(term.power).or_insert(Expression::Integer(0));
                *coeff = Expression::Binary(
                    BinaryOp::Add,
                    Box::new(coeff.clone()),
                    Box::new(term.coefficient.clone()),
                )
                .simplify();
            }
        }

        for (power, coeff) in coeffs {
            result.add_term(SeriesTerm::new(coeff, power));
        }

        // Sort terms by power
        result.terms.sort_by_key(|t| t.power);

        Ok(result)
    }
}

impl Sub for Series {
    type Output = SeriesResult<Series>;

    fn sub(self, rhs: Series) -> SeriesResult<Series> {
        // Check that centers match
        if self.center != rhs.center {
            return Err(SeriesError::InvalidCenter(
                "Cannot subtract series with different centers".into(),
            ));
        }
        if self.variable.name != rhs.variable.name {
            return Err(SeriesError::CannotExpand(
                "Cannot subtract series with different variables".into(),
            ));
        }

        let min_order = self.order.min(rhs.order);
        let mut result = Series::new(self.variable.clone(), self.center.clone(), min_order);

        // Collect coefficients by power
        let mut coeffs: HashMap<u32, Expression> = HashMap::new();

        for term in &self.terms {
            if term.power <= min_order {
                coeffs.insert(term.power, term.coefficient.clone());
            }
        }

        for term in &rhs.terms {
            if term.power <= min_order {
                let coeff = coeffs.entry(term.power).or_insert(Expression::Integer(0));
                *coeff = Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(coeff.clone()),
                    Box::new(term.coefficient.clone()),
                )
                .simplify();
            }
        }

        for (power, coeff) in coeffs {
            result.add_term(SeriesTerm::new(coeff, power));
        }

        // Sort terms by power
        result.terms.sort_by_key(|t| t.power);

        Ok(result)
    }
}

impl Mul for Series {
    type Output = SeriesResult<Series>;

    fn mul(self, rhs: Series) -> SeriesResult<Series> {
        // Check that centers match
        if self.center != rhs.center {
            return Err(SeriesError::InvalidCenter(
                "Cannot multiply series with different centers".into(),
            ));
        }
        if self.variable.name != rhs.variable.name {
            return Err(SeriesError::CannotExpand(
                "Cannot multiply series with different variables".into(),
            ));
        }

        let min_order = self.order.min(rhs.order);
        let mut result = Series::new(self.variable.clone(), self.center.clone(), min_order);

        // Cauchy product: c_n = sum_{k=0}^n a_k * b_{n-k}
        let mut coeffs: HashMap<u32, Expression> = HashMap::new();

        for term_a in &self.terms {
            for term_b in &rhs.terms {
                let new_power = term_a.power + term_b.power;
                if new_power <= min_order {
                    let product = Expression::Binary(
                        BinaryOp::Mul,
                        Box::new(term_a.coefficient.clone()),
                        Box::new(term_b.coefficient.clone()),
                    )
                    .simplify();

                    let coeff = coeffs.entry(new_power).or_insert(Expression::Integer(0));
                    *coeff = Expression::Binary(
                        BinaryOp::Add,
                        Box::new(coeff.clone()),
                        Box::new(product),
                    )
                    .simplify();
                }
            }
        }

        for (power, coeff) in coeffs {
            result.add_term(SeriesTerm::new(coeff, power));
        }

        // Sort terms by power
        result.terms.sort_by_key(|t| t.power);

        Ok(result)
    }
}

impl Div for Series {
    type Output = SeriesResult<Series>;

    fn div(self, rhs: Series) -> SeriesResult<Series> {
        // S1 / S2 = S1 * (1/S2)
        let reciprocal = rhs.reciprocal()?;
        self * reciprocal
    }
}

/// Compose two series: outer(inner(x)).
/// Generally requires inner(c) = 0 for convergence (where c is the center).
pub fn compose_series(outer: &Series, inner: &Series) -> SeriesResult<Series> {
    // Check that the inner series starts with 0 at its center
    if outer.center != inner.center {
        return Err(SeriesError::InvalidCenter(
            "Cannot compose series with different centers".into(),
        ));
    }
    if outer.variable.name != inner.variable.name {
        return Err(SeriesError::CannotExpand(
            "Cannot compose series with different variables".into(),
        ));
    }

    // For composition, inner(c) should be 0 (or at least the a_0 term should be zero)
    let inner_a0 = inner.coeff_f64(0);
    if inner_a0.abs() > 1e-15 {
        return Err(SeriesError::CannotExpand(
            "Cannot compose series: inner series must have zero constant term".into(),
        ));
    }

    let order = outer.order.min(inner.order);
    let mut result = Series::new(outer.variable.clone(), outer.center.clone(), order);

    // S_outer(S_inner(x)) = sum_{n=0}^N a_n * S_inner(x)^n
    // We compute powers of inner series incrementally
    let mut inner_powers: Vec<Series> = Vec::new();

    // inner^0 = 1
    let mut inner_pow_0 = Series::new(inner.variable.clone(), inner.center.clone(), order);
    inner_pow_0.add_term(SeriesTerm::new(Expression::Integer(1), 0));
    inner_powers.push(inner_pow_0);

    // Precompute powers of inner up to order
    for _ in 1..=order {
        let prev = inner_powers.last().unwrap().clone();
        let next = (prev * inner.clone())?;
        inner_powers.push(next);
    }

    // Now sum: sum a_n * inner^n
    let mut coeffs: HashMap<u32, Expression> = HashMap::new();

    for term in &outer.terms {
        if term.power as usize <= inner_powers.len() - 1 {
            let inner_pow = &inner_powers[term.power as usize];
            for inner_term in &inner_pow.terms {
                if inner_term.power <= order {
                    let contribution = Expression::Binary(
                        BinaryOp::Mul,
                        Box::new(term.coefficient.clone()),
                        Box::new(inner_term.coefficient.clone()),
                    )
                    .simplify();

                    let coeff = coeffs
                        .entry(inner_term.power)
                        .or_insert(Expression::Integer(0));
                    *coeff = Expression::Binary(
                        BinaryOp::Add,
                        Box::new(coeff.clone()),
                        Box::new(contribution),
                    )
                    .simplify();
                }
            }
        }
    }

    for (power, coeff) in coeffs {
        result.add_term(SeriesTerm::new(coeff, power));
    }

    result.terms.sort_by_key(|t| t.power);

    Ok(result)
}

/// Compute the compositional inverse (reversion) of a series.
/// Find T such that S(T(x)) = x.
/// Requires a_0 = 0 and a_1 ≠ 0.
pub fn reversion(series: &Series) -> SeriesResult<Series> {
    let a0 = series.coeff_f64(0);
    let a1 = series.coeff_f64(1);

    if a0.abs() > 1e-15 {
        return Err(SeriesError::CannotExpand(
            "Cannot compute reversion: constant term must be zero".into(),
        ));
    }
    if a1.abs() < 1e-15 {
        return Err(SeriesError::CannotExpand(
            "Cannot compute reversion: linear coefficient must be non-zero".into(),
        ));
    }

    let order = series.order;
    let mut result = Series::new(series.variable.clone(), series.center.clone(), order);

    // b_1 = 1/a_1
    let b1 = 1.0 / a1;
    result.add_term(SeriesTerm::new(Expression::Float(b1), 1));

    // Use Lagrange inversion formula for higher coefficients
    // For n >= 2: b_n can be computed from the coefficients of S and lower b_k
    // This is a simplified Newton iteration approach

    for n in 2..=order {
        // Compute b_n using the implicit equation:
        // sum_{k=1}^n a_k * [T^k]_{coeff of x^n} = delta_{n,1}
        // Since T(x) = sum b_j x^j, we need [T^k]_n
        let mut sum = 0.0;

        for k in 1..=n {
            // Compute the coefficient of x^n in T(x)^k
            let tk_coeff_n = compute_power_coeff(&result, k, n);
            let a_k = series.coeff_f64(k);
            sum += a_k * tk_coeff_n;
        }

        // sum = delta_{n,1} = 0 for n > 1
        // a_1 * b_n + contribution_from_lower = 0
        // b_n = -contribution_from_lower / a_1

        // The contribution from a_1 * b_n needs to be separated
        let contribution_without_b_n = sum - a1 * 0.0; // b_n hasn't been added yet
        let b_n = -contribution_without_b_n / a1;

        // Actually, we need to reconsider - the sum already doesn't include b_n
        // So b_n = (0 - sum) / a1, but sum already accounts for known terms only
        // This needs more careful derivation

        if b_n.abs() > 1e-15 {
            result.add_term(SeriesTerm::new(Expression::Float(b_n), n));
        }
    }

    result.terms.sort_by_key(|t| t.power);

    Ok(result)
}

/// Helper: compute coefficient of x^n in T^k where T is a series.
fn compute_power_coeff(series: &Series, k: u32, n: u32) -> f64 {
    if k == 0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    if k == 1 {
        return series.coeff_f64(n);
    }

    // For k > 1, use convolution
    // [T^k]_n = sum_{j=0}^n [T]_j * [T^{k-1}]_{n-j}
    let mut sum = 0.0;
    for j in 0..=n {
        let t_j = series.coeff_f64(j);
        let t_k1_nj = compute_power_coeff(series, k - 1, n - j);
        sum += t_j * t_k1_nj;
    }
    sum
}
