//! Rational-function integration via partial-fraction decomposition.
//!
//! This module detects rational-function integrands `N(x)/D(x)` and integrates
//! them in closed form by decomposing into partial fractions. Each component
//! term is integrated directly:
//!
//! - `A/(x−a)`        → `A·ln|x−a|`
//! - `A/(x−a)^n, n≥2` → `−A/((n−1)(x−a)^{n−1})`
//! - `(Bx+C)/(x²+px+q)` → `·ln(x²+…) + ·atan(…)`
//!
//! All computation operates on [`Arc<Expr>`] internally (Rule 1). The public
//! function accepts and returns [`Arc<Expr>`]; callers at the Rule-2 boundary
//! (the `integrate` public API) compile/decompile around this call.

use std::sync::Arc;

use crate::ast::Variable;
use crate::numeric::compile::{compile, decompile};
use crate::numeric::expr::Expr;
use crate::numeric::SymbolId;
use crate::partial_fractions::{decompose, is_polynomial};

/// Attempt to integrate a compiled `Arc<Expr>` expression as a rational function.
///
/// Returns `Some(antiderivative)` if the expression is a genuine fraction
/// (numerator and denominator both polynomials in `var`) and decomposition
/// succeeds. Returns `None` otherwise so the caller can fall through to the
/// next integration strategy.
///
/// Internally: extract num/denom from Arc<Expr> → decompile each →
/// decompose → integrate terms → compile result.
///
/// # Recursion safety
///
/// Only expressions whose `Arc<Expr>` form has a non-trivial denominator are
/// intercepted. Pure polynomials (denominator = 1) are left to the pattern
/// integrator. This prevents the infinite recursion that would arise from the
/// `Polynomial` partial-fraction term calling back into `integrate`.
pub(super) fn try_integrate_rational(expr: &Arc<Expr>, var: SymbolId) -> Option<Arc<Expr>> {
    let var_name_owned = var.as_str();
    let var_name: &str = &var_name_owned;

    // Extract numerator and denominator directly from the Arc<Expr> form.
    // After compile+normalize, a fraction N/D is represented as:
    //   - Mul(coeff, Pow(D, -1))          for 1/D or c/D
    //   - Mul(N, Pow(D, -1))              for N/D
    //   - Pow(D, -1)                      for 1/D when N=1
    // We must NOT intercept bare polynomials (which are also rational).
    let (num_arc, den_arc) = extract_rational_parts(expr)?;

    // Decompile num/denom to Expression for the polynomial check and decompose.
    let num_expr = decompile(&num_arc);
    let den_expr = decompile(&den_arc);

    // Both sides must be polynomials in var.
    if !is_polynomial(&num_expr, var_name) || !is_polynomial(&den_expr, var_name) {
        return None;
    }

    let var_ast = Variable::new(var_name);
    let result = decompose(&num_expr, &den_expr, &var_ast).ok()?;

    // Integrate all partial-fraction terms at the Expression level,
    // then compile the result to Arc<Expr> to stay in the internal representation.
    let antideriv_expr = result.integrate();
    Some(compile(&antideriv_expr))
}

/// Extract `(numerator, denominator)` as `Arc<Expr>` from a normalized expression.
///
/// Handles the canonical forms produced by `normalize::div`:
/// - `Pow(D, -1)`        → numerator = 1, denominator = D
/// - `Mul(N, Pow(D,-1))` → numerator = N (reconstructed from Mul node minus
///                           the negative-exponent factors), denominator = D
///
/// Returns `None` when the expression has no negative-exponent factor (i.e., it
/// is a pure polynomial or a transcendental — both should fall through).
fn extract_rational_parts(expr: &Arc<Expr>) -> Option<(Arc<Expr>, Arc<Expr>)> {
    match expr.as_ref() {
        // 1/D encoded as Pow(D, -1)
        Expr::Pow(base, exp) => {
            if is_neg_one_int(exp) {
                Some((Expr::int(1), base.clone()))
            } else {
                None
            }
        }

        // N/D encoded as Mul node with some factors having negative exponents
        Expr::Mul(node) => {
            let mut den_factors: Vec<(Arc<Expr>, Arc<Expr>)> = Vec::new();
            let mut num_factors: Vec<(Arc<Expr>, Arc<Expr>)> = Vec::new();

            // Collect denominator factors: base^k where k < 0
            for (base, exp) in &node.factors {
                if is_negative_int_exp(exp) {
                    // Denominator: base^|k|
                    let pos_exp = negate_exp(exp);
                    den_factors.push((base.clone(), pos_exp));
                } else {
                    num_factors.push((base.clone(), exp.clone()));
                }
            }

            // Must have at least one denominator factor to be a fraction.
            if den_factors.is_empty() {
                return None;
            }

            // Reconstruct denominator as product of base^|exp| factors.
            let den = build_product(&node.coeff, &den_factors, false);

            // Reconstruct numerator from remaining factors (positive exponents)
            // and the scalar coefficient is in the numerator side.
            let num = build_product_num(&node.coeff, &num_factors);

            Some((num, den))
        }

        _ => None,
    }
}

/// Returns true if `exp` is an integer Arc<Expr> equal to -1.
fn is_neg_one_int(exp: &Arc<Expr>) -> bool {
    matches!(exp.as_ref(), Expr::Integer(n) if n.to_i64().map(|v| v == -1).unwrap_or(false))
}

/// Returns true if `exp` is a negative integer Arc<Expr>.
fn is_negative_int_exp(exp: &Arc<Expr>) -> bool {
    matches!(exp.as_ref(), Expr::Integer(n) if n.to_i64().map(|v| v < 0).unwrap_or(false))
}

/// Negate an exponent: returns the Arc<Expr> for -exp.
fn negate_exp(exp: &Arc<Expr>) -> Arc<Expr> {
    match exp.as_ref() {
        Expr::Integer(n) => {
            let v = n.to_i64().unwrap_or(0);
            Expr::int(-v)
        }
        _ => Expr::int(1), // fallback; shouldn't happen given is_negative_int_exp guard
    }
}

/// Build a power-product expression from a list of (base, positive_exp) pairs.
/// The scalar `coeff` of the MulNode is ignored here (numerator side handles it).
fn build_product(
    _coeff: &crate::numeric::BigRational,
    factors: &[(Arc<Expr>, Arc<Expr>)],
    _include_coeff: bool,
) -> Arc<Expr> {
    use crate::numeric::normalize;
    factors.iter().fold(Expr::int(1), |acc, (base, exp)| {
        let term = normalize::pow(base.clone(), exp.clone());
        normalize::mul(acc, term)
    })
}

/// Reconstruct the numerator from (base, positive_exp) factors and the MulNode
/// scalar coefficient. The coefficient belongs on the numerator side.
fn build_product_num(
    coeff: &crate::numeric::BigRational,
    factors: &[(Arc<Expr>, Arc<Expr>)],
) -> Arc<Expr> {
    use crate::numeric::normalize;
    use num::traits::One;

    let base = factors.iter().fold(Expr::int(1), |acc, (b, e)| {
        let term = normalize::pow(b.clone(), e.clone());
        normalize::mul(acc, term)
    });

    // Multiply by the scalar coefficient if it is not 1.
    if coeff.is_one() {
        base
    } else {
        let c = big_rational_to_arc(coeff);
        normalize::mul(c, base)
    }
}

/// Convert a `BigRational` to an `Arc<Expr>`.
fn big_rational_to_arc(r: &crate::numeric::BigRational) -> Arc<Expr> {
    use num::traits::One;
    if r.denom().is_one() {
        Expr::int(r.numer().to_i64().unwrap_or_else(|| r.to_f64() as i64))
    } else {
        Arc::new(Expr::Rational(r.clone()))
    }
}
