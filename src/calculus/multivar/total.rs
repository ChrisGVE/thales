//! Total derivative via the multivariate chain rule.

use std::sync::Arc;

use crate::numeric::{differentiation::diff_arc, normalize, Expr, SymbolId};

// ── Total derivative ──────────────────────────────────────────────────────────

/// Compute the total derivative df using the multivariate chain rule.
///
/// Given a scalar expression f(x₁, …, xₙ, y₁, …, yₖ) where the yⱼ are
/// themselves functions of the independents (xᵢ), the total derivative
/// with respect to the independents is:
///
/// ```text
/// df = Σᵢ (∂f/∂xᵢ) + Σⱼ (∂f/∂yⱼ) · (substituted chain contribution)
/// ```
///
/// Concretely, `dependents` is a slice of `(yⱼ_id, dyⱼ_expr)` pairs where
/// `dyⱼ_expr` is the expression for dyⱼ/dt (or dyⱼ in differential form).
/// The function returns:
///
/// ```text
/// Σᵢ ∂f/∂xᵢ  +  Σⱼ (∂f/∂yⱼ) · dyⱼ_expr
/// ```
///
/// This is the symbolic total differential treating `independents` as the
/// free variables and accumulating chain-rule contributions from the
/// `dependents`.
///
/// # Arguments
///
/// * `expr` — scalar expression f
/// * `independents` — variable IDs that appear directly in f as free variables
/// * `dependents` — `(yⱼ_id, dyⱼ)` pairs: symbol ID of a dependent variable
///   and its differential expression (e.g. derivative w.r.t. an independent)
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use thales::numeric::{Expr, SymbolId, normalize};
/// use thales::calculus::multivar::total_derivative;
///
/// // f = x + y,  x independent,  y = x (so dy/dx = 1)
/// // df/dx = ∂f/∂x + ∂f/∂y * dy/dx = 1 + 1*1 = 2
/// let x_id = SymbolId::intern("td_x");
/// let y_id = SymbolId::intern("td_y");
/// let x = Expr::symbol("td_x");
/// let y = Expr::symbol("td_y");
/// let f = normalize::add(x.clone(), y.clone());
/// let dy = Expr::int(1); // dy/dx = 1
/// let result = total_derivative(&f, &[x_id], &[(y_id, dy)]);
/// // ∂f/∂x = 1, ∂f/∂y * 1 = 1, total = 2
/// assert_eq!(*result, Expr::Integer(thales::numeric::SmallInt::from(2i64)));
///
/// // f = x²,  no dependents → total derivative = ∂f/∂x = 2x
/// let f2 = normalize::pow(x.clone(), Expr::int(2));
/// let result2 = total_derivative(&f2, &[x_id], &[]);
/// assert!(!result2.is_zero());
/// ```
pub fn total_derivative(
    expr: &Arc<Expr>,
    independents: &[SymbolId],
    dependents: &[(SymbolId, Arc<Expr>)],
) -> Arc<Expr> {
    let mut acc = Expr::int(0);

    // Direct partial contributions from independent variables.
    for &xi in independents {
        let df_dxi = diff_arc(expr, xi);
        acc = normalize::add(acc, df_dxi);
    }

    // Chain-rule contributions from dependent variables.
    for (yj, dyj) in dependents {
        let df_dyj = diff_arc(expr, *yj);
        let chain = normalize::mul(df_dyj, dyj.clone());
        acc = normalize::add(acc, chain);
    }

    acc
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{normalize, Expr, SmallInt, SymbolId};

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }
    fn id(name: &str) -> SymbolId {
        SymbolId::intern(name)
    }

    #[test]
    fn test_total_derivative_chain() {
        // f = x + y,  x independent,  dy/dx = 1
        // df/dx = ∂f/∂x + ∂f/∂y * dy/dx = 1 + 1*1 = 2
        let xi = id("td_chain_x");
        let yi = id("td_chain_y");
        let x = sym("td_chain_x");
        let y = sym("td_chain_y");
        let f = normalize::add(x.clone(), y.clone());
        let dy = Expr::int(1);
        let result = total_derivative(&f, &[xi], &[(yi, dy)]);
        assert_eq!(*result, Expr::Integer(SmallInt::from(2i64)));
    }

    #[test]
    fn test_total_derivative_no_dependents() {
        // f = x²,  x independent,  no dependents  →  2x
        let xi = id("td_nodep_x");
        let x = sym("td_nodep_x");
        let f = normalize::pow(x.clone(), Expr::int(2));
        let result = total_derivative(&f, &[xi], &[]);
        assert!(
            !result.is_zero(),
            "total derivative of x² should not be zero"
        );
    }

    #[test]
    fn test_total_derivative_constant() {
        // f = 5,  any independents/dependents  →  0
        let xi = id("td_const_x");
        let yi = id("td_const_y");
        let dy = Expr::int(3);
        let result = total_derivative(&Expr::int(5), &[xi], &[(yi, dy)]);
        assert!(result.is_zero(), "total derivative of constant should be 0");
    }
}
