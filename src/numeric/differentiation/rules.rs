//! Basic differentiation rules: Add, Mul, Pow, and numeric helpers.

use std::sync::Arc;

use super::super::{expr::Expr, normalize, AddNode, BigRational, MulNode, SymbolId};
use super::diff_arc;

// ── Differentiation of Add ────────────────────────────────────────────────────

pub(crate) fn diff_add(node: &AddNode, var: SymbolId) -> Arc<Expr> {
    // Constant part disappears; differentiate each term
    // AddNode: constant + Σ coeff_i * term_i
    // d/dx = Σ coeff_i * diff(term_i)
    let mut result = Expr::int(0);
    for (term, coeff) in &node.terms {
        let dt = diff_arc(term, var);
        let scaled = normalize::mul(rational_to_arc(coeff), dt);
        result = normalize::add(result, scaled);
    }
    result
}

/// Convert a &BigRational to an Arc<Expr>.
pub(crate) fn rational_to_arc(r: &BigRational) -> Arc<Expr> {
    if r.is_integer() {
        if let Some(n) = r.numer().to_i64() {
            Expr::int(n)
        } else {
            Arc::new(Expr::Rational(r.clone()))
        }
    } else {
        Arc::new(Expr::Rational(r.clone()))
    }
}

// ── Differentiation of Mul ───────────────────────────────────────────────────

pub(crate) fn diff_mul_node(node: &MulNode, _full: &Arc<Expr>, var: SymbolId) -> Arc<Expr> {
    // MulNode: coeff * Π base_i^exp_i
    // The coefficient is a constant → its derivative contributes 0.
    // We treat each factor base_i^exp_i as a sub-expression and apply
    // the product rule across all factors.
    //
    // d/dx [ coeff * f1 * f2 * ... * fn ]
    //   = coeff * Σ_i [ (d/dx fi) * Π_{j≠i} fj ]
    //
    // Each fi = base_i^exp_i is differentiated via diff_pow.

    let factors: Vec<Arc<Expr>> = node
        .factors
        .iter()
        .map(|(base, exp)| normalize::pow(base.clone(), exp.clone()))
        .collect();

    let coeff_expr: Arc<Expr> = rational_to_arc(&node.coeff);

    // Build product-rule sum
    let mut sum = Expr::int(0);
    for i in 0..factors.len() {
        let fi = &factors[i];
        let dfi = diff_arc(fi, var);
        if dfi.is_zero() {
            continue;
        }
        // Build product of all factors except i
        let mut others = Expr::int(1) as Arc<Expr>;
        for (j, fj) in factors.iter().enumerate() {
            if j != i {
                others = normalize::mul(others, fj.clone());
            }
        }
        let term = normalize::mul(dfi, others);
        sum = normalize::add(sum, term);
    }

    normalize::mul(coeff_expr, sum)
}

// ── Differentiation of Pow ───────────────────────────────────────────────────

pub(crate) fn diff_pow(base: &Arc<Expr>, exp: &Arc<Expr>, var: SymbolId) -> Arc<Expr> {
    // Check if exponent is a numeric constant
    let exp_is_const = is_numeric_const(exp);
    let base_du = diff_arc(base, var);
    let exp_du = diff_arc(exp, var);

    if exp_is_const {
        // Power rule: d/dx(u^n) = n * u^(n-1) * u'
        if base_du.is_zero() {
            return Expr::int(0);
        }
        let n_minus_1 = normalize::sub(exp.clone(), Expr::int(1));
        let u_pow = normalize::pow(base.clone(), n_minus_1);
        let n_u_pow = normalize::mul(exp.clone(), u_pow);
        normalize::mul(n_u_pow, base_du)
    } else {
        // General: d/dx(u^v) = u^v * (v'*ln(u) + v*u'/u)
        use super::super::expr::FuncId;
        let u_pow_v = normalize::pow(base.clone(), exp.clone());
        let ln_u = Expr::func(FuncId::Ln, vec![base.clone()]);
        let v_prime_ln_u = normalize::mul(exp_du, ln_u);
        let u_prime_over_u = normalize::div(base_du, base.clone());
        let v_u_prime_over_u = normalize::mul(exp.clone(), u_prime_over_u);
        let bracket = normalize::add(v_prime_ln_u, v_u_prime_over_u);
        normalize::mul(u_pow_v, bracket)
    }
}

/// Returns true if `expr` is a numeric constant (no free variables).
pub(crate) fn is_numeric_const(expr: &Arc<Expr>) -> bool {
    matches!(
        expr.as_ref(),
        Expr::Integer(_) | Expr::Rational(_) | Expr::Float(_)
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::diff;
    use crate::numeric::{Expr, SymbolId};
    use std::sync::Arc;

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }
    fn x_id(name: &str) -> SymbolId {
        SymbolId::intern(name)
    }

    #[test]
    fn test_diff_product_rule() {
        // d/dx(x * y) = y  (y is treated as constant)
        let xid = x_id("dpr_x");
        let x = sym("dpr_x");
        let y = sym("dpr_y");
        use crate::numeric::normalize;
        let e = normalize::mul(x.clone(), y.clone());
        let result = diff(&e, xid);
        // normalized result should equal y
        assert_eq!(*result, *y, "expected y, got {result}");
    }

    #[test]
    fn test_diff_x_squared() {
        // d/dx(x^2) via mul: x*x → 2*x
        let xid = x_id("dxs_x");
        let x = sym("dxs_x");
        use crate::numeric::normalize;
        let e = normalize::mul(x.clone(), x.clone());
        let result = diff(&e, xid);
        // 2*x as a Mul node
        assert!(!result.is_zero(), "got 0 for d/dx(x^2)");
    }

    #[test]
    fn test_diff_pow_integer_exp() {
        // d/dx(x^3) = 3*x^2
        let xid = x_id("dpow_x");
        let x = sym("dpow_x");
        let e = Expr::pow(x.clone(), Expr::int(3));
        let result = diff(&e, xid);
        assert!(!result.is_zero(), "got 0 for d/dx(x^3)");
    }

    #[test]
    fn test_diff_pow_constant_base() {
        // d/dx(2^x) = 2^x * ln(2)  (exp_is_const=false)
        let xid = x_id("dpow_cx");
        let e = Expr::pow(Expr::int(2), sym("dpow_cx"));
        let result = diff(&e, xid);
        assert!(!result.is_zero(), "got 0 for d/dx(2^x)");
    }

    #[test]
    fn test_diff_pow_no_var() {
        // d/dx(y^2) = 0
        let xid = x_id("dpow_nv_x");
        let y = sym("dpow_nv_y");
        let e = Expr::pow(y, Expr::int(2));
        let result = diff(&e, xid);
        assert!(result.is_zero(), "expected 0 for d/dx(y^2)");
    }
}
