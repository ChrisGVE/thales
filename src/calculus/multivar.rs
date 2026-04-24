//! Multivariate differential calculus on [`Arc<Expr>`] trees.
//!
//! All functions in this module delegate to the single-variable
//! [`diff_arc`][crate::numeric::differentiation::diff_arc] engine and compose
//! its results to form higher-level multivariate objects.  No symbolic logic
//! is duplicated: every differentiation step goes through the same chain-rule
//! and constant-folding machinery that powers the univariate case.
//!
//! # Functions
//!
//! | Function | Math object |
//! |---|---|
//! | [`partial`] | ∂f/∂xᵢ |
//! | [`gradient`] | ∇f = (∂f/∂x₁, …, ∂f/∂xₙ) |
//! | [`jacobian`] | J[i][j] = ∂fᵢ/∂xⱼ |
//! | [`hessian`] | H[i][j] = ∂²f/∂xᵢ∂xⱼ |
//! | [`directional_derivative`] | Dᵥf = ∇f · v̂ |
//! | [`total_derivative`] | df = Σ (∂f/∂xᵢ) dxᵢ  (chain rule with substitution) |

use std::sync::Arc;

use crate::numeric::{differentiation::diff_arc, normalize, Expr, SymbolId};

// ── Partial derivative ────────────────────────────────────────────────────────

/// Compute the partial derivative ∂`expr`/∂`var`.
///
/// All other symbols present in the expression are treated as constants,
/// which is exactly the behaviour of [`diff_arc`] for symbols that do not
/// match `var`.  The result is normalized (constant-folded, identities
/// removed) by the underlying smart constructors.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use thales::numeric::{Expr, SymbolId};
/// use thales::calculus::multivar::partial;
///
/// // ∂/∂x (x² + y) = 2x
/// let x_id = SymbolId::intern("prt_x");
/// let y_id = SymbolId::intern("prt_y");
/// let x = Expr::symbol("prt_x");
/// let y = Expr::symbol("prt_y");
/// let expr = thales::numeric::normalize::add(
///     thales::numeric::normalize::pow(x.clone(), Expr::int(2)),
///     y.clone(),
/// );
/// let result = partial(&expr, &x_id);
/// assert!(!result.is_zero()); // 2x ≠ 0
///
/// // ∂/∂y (x²) = 0  — treating x as constant
/// let result_y = partial(&thales::numeric::normalize::pow(x.clone(), Expr::int(2)), &y_id);
/// assert!(result_y.is_zero());
/// ```
pub fn partial(expr: &Arc<Expr>, var: &SymbolId) -> Arc<Expr> {
    diff_arc(expr, *var)
}

// ── Gradient ──────────────────────────────────────────────────────────────────

/// Compute the gradient ∇f = (∂f/∂x₁, …, ∂f/∂xₙ).
///
/// Returns one partial derivative per entry in `vars`, in the same order.
/// For a scalar field f(x₁, …, xₙ) the gradient is the vector of all
/// first-order partial derivatives.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use thales::numeric::{Expr, SymbolId, normalize};
/// use thales::calculus::multivar::gradient;
///
/// // f = x² + y²  →  ∇f = (2x, 2y)
/// let x_id = SymbolId::intern("grad_x");
/// let y_id = SymbolId::intern("grad_y");
/// let x = Expr::symbol("grad_x");
/// let y = Expr::symbol("grad_y");
/// let f = normalize::add(
///     normalize::pow(x.clone(), Expr::int(2)),
///     normalize::pow(y.clone(), Expr::int(2)),
/// );
/// let g = gradient(&f, &[x_id, y_id]);
/// assert_eq!(g.len(), 2);
/// assert!(!g[0].is_zero()); // ∂f/∂x = 2x
/// assert!(!g[1].is_zero()); // ∂f/∂y = 2y
///
/// // Gradient of a constant is all zeros
/// let gc = gradient(&Expr::int(5), &[x_id, y_id]);
/// assert!(gc.iter().all(|d| d.is_zero()));
/// ```
pub fn gradient(expr: &Arc<Expr>, vars: &[SymbolId]) -> Vec<Arc<Expr>> {
    vars.iter().map(|v| diff_arc(expr, *v)).collect()
}

// ── Jacobian matrix ───────────────────────────────────────────────────────────

/// Compute the Jacobian matrix J where J[i][j] = ∂fᵢ/∂xⱼ.
///
/// Given a vector-valued function **f** = (f₁, …, fₘ) of variables
/// (x₁, …, xₙ) the Jacobian is the m × n matrix of partial derivatives.
/// Row `i` is the gradient of component `exprs[i]` with respect to all
/// variables in `vars`.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use thales::numeric::{Expr, SymbolId, normalize};
/// use thales::calculus::multivar::jacobian;
///
/// // f = (x*y, x + y)  →  J = [[y, x], [1, 1]]
/// let x_id = SymbolId::intern("jac_x");
/// let y_id = SymbolId::intern("jac_y");
/// let x = Expr::symbol("jac_x");
/// let y = Expr::symbol("jac_y");
/// let f0 = normalize::mul(x.clone(), y.clone()); // x*y
/// let f1 = normalize::add(x.clone(), y.clone()); // x + y
/// let j = jacobian(&[f0, f1], &[x_id, y_id]);
/// assert_eq!(j.len(), 2);        // 2 rows (components)
/// assert_eq!(j[0].len(), 2);     // 2 cols (variables)
/// // ∂(x*y)/∂x = y
/// assert!(!j[0][0].is_zero());
/// // ∂(x+y)/∂x = 1
/// assert!(j[1][0].is_one());
/// ```
pub fn jacobian(exprs: &[Arc<Expr>], vars: &[SymbolId]) -> Vec<Vec<Arc<Expr>>> {
    exprs
        .iter()
        .map(|f| vars.iter().map(|v| diff_arc(f, *v)).collect())
        .collect()
}

// ── Hessian matrix ────────────────────────────────────────────────────────────

/// Compute the Hessian matrix H where H[i][j] = ∂²f/∂xᵢ∂xⱼ.
///
/// The Hessian is an n × n symmetric matrix of second-order partial
/// derivatives of a scalar field f(x₁, …, xₙ).  Mixed partials are
/// computed by differentiating the i-th partial derivative with respect
/// to xⱼ.  By Clairaut's theorem the matrix is symmetric when f has
/// continuous second partials (no check is performed; the symbolic result
/// is always consistent).
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use thales::numeric::{Expr, SymbolId, normalize};
/// use thales::calculus::multivar::hessian;
///
/// // f = x² + x*y + y²  →  H = [[2, 1], [1, 2]]
/// let x_id = SymbolId::intern("hes_x");
/// let y_id = SymbolId::intern("hes_y");
/// let x = Expr::symbol("hes_x");
/// let y = Expr::symbol("hes_y");
/// let f = normalize::add(
///     normalize::add(
///         normalize::pow(x.clone(), Expr::int(2)),
///         normalize::mul(x.clone(), y.clone()),
///     ),
///     normalize::pow(y.clone(), Expr::int(2)),
/// );
/// let h = hessian(&f, &[x_id, y_id]);
/// assert_eq!(h.len(), 2);
/// assert_eq!(h[0].len(), 2);
/// // H[0][0] = ∂²f/∂x² = 2
/// assert_eq!(*h[0][0], Expr::Integer(thales::numeric::SmallInt::from(2i64)));
/// // H[0][1] = H[1][0] = ∂²f/∂x∂y = 1
/// assert!(h[0][1].is_one());
/// assert!(h[1][0].is_one());
///
/// // Hessian of a constant is all zeros
/// let hc = hessian(&Expr::int(7), &[x_id, y_id]);
/// assert!(hc.iter().flatten().all(|d| d.is_zero()));
/// ```
pub fn hessian(expr: &Arc<Expr>, vars: &[SymbolId]) -> Vec<Vec<Arc<Expr>>> {
    // Compute first partials once, then differentiate each again.
    let first: Vec<Arc<Expr>> = vars.iter().map(|v| diff_arc(expr, *v)).collect();
    first
        .iter()
        .map(|fi| vars.iter().map(|v| diff_arc(fi, *v)).collect())
        .collect()
}

// ── Directional derivative ────────────────────────────────────────────────────

/// Compute the directional derivative Dᵥf = ∇f · dir.
///
/// Mathematically: Dᵥf = Σᵢ (∂f/∂xᵢ) · dirᵢ.
///
/// The direction vector `dir` need not be a unit vector; normalization is
/// the caller's responsibility when the unit-vector interpretation is
/// required.  The function computes the dot product of the gradient with
/// the supplied direction components symbolically.
///
/// `vars` and `dir` must have the same length; mismatched lengths are
/// silently truncated to `min(vars.len(), dir.len())`.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use thales::numeric::{Expr, SymbolId, normalize};
/// use thales::calculus::multivar::directional_derivative;
///
/// // f = x² + y,  dir = (1, 0)  →  Dᵥf = 2x
/// let x_id = SymbolId::intern("ddir_x");
/// let y_id = SymbolId::intern("ddir_y");
/// let x = Expr::symbol("ddir_x");
/// let y = Expr::symbol("ddir_y");
/// let f = normalize::add(normalize::pow(x.clone(), Expr::int(2)), y.clone());
/// let dir = vec![Expr::int(1), Expr::int(0)];
/// let result = directional_derivative(&f, &[x_id, y_id], &dir);
/// assert!(!result.is_zero()); // 2x ≠ 0
///
/// // Zero direction vector → result is 0
/// let zero_dir = vec![Expr::int(0), Expr::int(0)];
/// let zero_result = directional_derivative(&f, &[x_id, y_id], &zero_dir);
/// assert!(zero_result.is_zero());
/// ```
pub fn directional_derivative(expr: &Arc<Expr>, vars: &[SymbolId], dir: &[Arc<Expr>]) -> Arc<Expr> {
    let n = vars.len().min(dir.len());
    let mut acc = Expr::int(0);
    for i in 0..n {
        let grad_i = diff_arc(expr, vars[i]);
        let term = normalize::mul(grad_i, dir[i].clone());
        acc = normalize::add(acc, term);
    }
    acc
}

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
    use crate::numeric::{normalize, Expr, FuncId, SmallInt, SymbolId};

    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }
    fn id(name: &str) -> SymbolId {
        SymbolId::intern(name)
    }

    // ── partial ───────────────────────────────────────────────────────────

    #[test]
    fn test_partial_polynomial() {
        // ∂/∂x (x³ + y) = 3x²
        let xi = id("p_poly_x");
        let yi = id("p_poly_y");
        let x = sym("p_poly_x");
        let y = sym("p_poly_y");
        let f = normalize::add(normalize::pow(x.clone(), Expr::int(3)), y.clone());
        let result = partial(&f, &xi);
        // 3x² ≠ 0
        assert!(!result.is_zero(), "∂/∂x(x³+y) should not be zero");
        // wrt y: ∂/∂y (x³ + y) = 1
        let result_y = partial(&f, &yi);
        assert!(result_y.is_one(), "∂/∂y(x³+y) should be 1, got {result_y}");
    }

    #[test]
    fn test_partial_constant_is_zero() {
        // ∂/∂x (42) = 0
        let xi = id("p_const_x");
        assert!(partial(&Expr::int(42), &xi).is_zero());
    }

    #[test]
    fn test_partial_trig() {
        // ∂/∂x (sin(x) * cos(y)) = cos(x) * cos(y)
        let xi = id("p_trig_x");
        let x = sym("p_trig_x");
        let y = sym("p_trig_y");
        let sin_x = Expr::func(FuncId::Sin, vec![x.clone()]);
        let cos_y = Expr::func(FuncId::Cos, vec![y.clone()]);
        let f = normalize::mul(sin_x, cos_y);
        let result = partial(&f, &xi);
        // cos(x)*cos(y) ≠ 0
        assert!(!result.is_zero());
    }

    // ── gradient ──────────────────────────────────────────────────────────

    #[test]
    fn test_gradient_polynomial() {
        // f = x² + y²  →  ∇f = (2x, 2y)
        let xi = id("gr_x");
        let yi = id("gr_y");
        let x = sym("gr_x");
        let y = sym("gr_y");
        let f = normalize::add(
            normalize::pow(x.clone(), Expr::int(2)),
            normalize::pow(y.clone(), Expr::int(2)),
        );
        let g = gradient(&f, &[xi, yi]);
        assert_eq!(g.len(), 2);
        assert!(!g[0].is_zero(), "∂f/∂x should not be zero");
        assert!(!g[1].is_zero(), "∂f/∂y should not be zero");
    }

    #[test]
    fn test_gradient_constant_all_zero() {
        // ∇(5) = (0, 0)
        let xi = id("gr_cx");
        let yi = id("gr_cy");
        let g = gradient(&Expr::int(5), &[xi, yi]);
        assert!(
            g.iter().all(|d| d.is_zero()),
            "gradient of constant should be zero"
        );
    }

    #[test]
    fn test_gradient_linear() {
        // f = 3x + 2y  →  ∇f = (3, 2)
        let xi = id("gr_lin_x");
        let yi = id("gr_lin_y");
        let x = sym("gr_lin_x");
        let y = sym("gr_lin_y");
        let f = normalize::add(
            normalize::mul(Expr::int(3), x.clone()),
            normalize::mul(Expr::int(2), y.clone()),
        );
        let g = gradient(&f, &[xi, yi]);
        assert_eq!(
            *g[0],
            Expr::Integer(SmallInt::from(3i64)),
            "∂f/∂x should be 3"
        );
        assert_eq!(
            *g[1],
            Expr::Integer(SmallInt::from(2i64)),
            "∂f/∂y should be 2"
        );
    }

    // ── jacobian ──────────────────────────────────────────────────────────

    #[test]
    fn test_jacobian_shape() {
        // f = (x*y, x+y): 2×2 Jacobian
        let xi = id("jac_sh_x");
        let yi = id("jac_sh_y");
        let x = sym("jac_sh_x");
        let y = sym("jac_sh_y");
        let f0 = normalize::mul(x.clone(), y.clone());
        let f1 = normalize::add(x.clone(), y.clone());
        let j = jacobian(&[f0, f1], &[xi, yi]);
        assert_eq!(j.len(), 2);
        assert_eq!(j[0].len(), 2);
        assert_eq!(j[1].len(), 2);
    }

    #[test]
    fn test_jacobian_values() {
        // f = (x², xy)  →  J = [[2x, 0], [y, x]]
        let xi = id("jac_v_x");
        let yi = id("jac_v_y");
        let x = sym("jac_v_x");
        let y = sym("jac_v_y");
        let f0 = normalize::pow(x.clone(), Expr::int(2)); // x²
        let f1 = normalize::mul(x.clone(), y.clone()); // x*y
        let j = jacobian(&[f0, f1], &[xi, yi]);
        // ∂(x²)/∂y = 0
        assert!(j[0][1].is_zero(), "∂(x²)/∂y should be 0");
        // ∂(x*y)/∂x = y  (matches sym("jac_v_y"))
        assert_eq!(*j[1][0], *y, "∂(xy)/∂x should be y");
        // ∂(x*y)/∂y = x
        assert_eq!(*j[1][1], *x, "∂(xy)/∂y should be x");
    }

    #[test]
    fn test_jacobian_constants_all_zero() {
        // f = (1, 2)  →  J = [[0, 0], [0, 0]]
        let xi = id("jac_c_x");
        let yi = id("jac_c_y");
        let j = jacobian(&[Expr::int(1), Expr::int(2)], &[xi, yi]);
        assert!(j.iter().flatten().all(|d| d.is_zero()));
    }

    // ── hessian ───────────────────────────────────────────────────────────

    #[test]
    fn test_hessian_quadratic() {
        // f = x² + x*y + y²  →  H = [[2, 1], [1, 2]]
        let xi = id("hes_q_x");
        let yi = id("hes_q_y");
        let x = sym("hes_q_x");
        let y = sym("hes_q_y");
        let f = normalize::add(
            normalize::add(
                normalize::pow(x.clone(), Expr::int(2)),
                normalize::mul(x.clone(), y.clone()),
            ),
            normalize::pow(y.clone(), Expr::int(2)),
        );
        let h = hessian(&f, &[xi, yi]);
        assert_eq!(h.len(), 2);
        assert_eq!(h[0].len(), 2);
        assert_eq!(
            *h[0][0],
            Expr::Integer(SmallInt::from(2i64)),
            "H[0][0] should be 2"
        );
        assert!(h[0][1].is_one(), "H[0][1] should be 1");
        assert!(h[1][0].is_one(), "H[1][0] should be 1");
        assert_eq!(
            *h[1][1],
            Expr::Integer(SmallInt::from(2i64)),
            "H[1][1] should be 2"
        );
    }

    #[test]
    fn test_hessian_constant_all_zero() {
        // H(7) = [[0, 0], [0, 0]]
        let xi = id("hes_c_x");
        let yi = id("hes_c_y");
        let h = hessian(&Expr::int(7), &[xi, yi]);
        assert!(h.iter().flatten().all(|d| d.is_zero()));
    }

    #[test]
    fn test_hessian_symmetry() {
        // f = sin(x) * exp(y)  →  H[0][1] == H[1][0]  (symbolic equality)
        let xi = id("hes_sym_x");
        let yi = id("hes_sym_y");
        let x = sym("hes_sym_x");
        let y = sym("hes_sym_y");
        let f = normalize::mul(
            Expr::func(FuncId::Sin, vec![x.clone()]),
            Expr::func(FuncId::Exp, vec![y.clone()]),
        );
        let h = hessian(&f, &[xi, yi]);
        // Mixed partials should produce the same normalized form.
        assert_eq!(
            format!("{}", h[0][1]),
            format!("{}", h[1][0]),
            "H[0][1] and H[1][0] should be equal by Clairaut"
        );
    }

    // ── directional_derivative ────────────────────────────────────────────

    #[test]
    fn test_directional_axis_aligned() {
        // f = x² + y,  dir = (1, 0)  →  Dᵥf = 2x
        let xi = id("dd_ax_x");
        let yi = id("dd_ax_y");
        let x = sym("dd_ax_x");
        let y = sym("dd_ax_y");
        let f = normalize::add(normalize::pow(x.clone(), Expr::int(2)), y.clone());
        let dir = vec![Expr::int(1), Expr::int(0)];
        let result = directional_derivative(&f, &[xi, yi], &dir);
        // 2x  (y contribution zeroed by dir[1]=0)
        assert!(!result.is_zero());
    }

    #[test]
    fn test_directional_zero_direction() {
        // Any f,  dir = (0, 0)  →  0
        let xi = id("dd_z_x");
        let yi = id("dd_z_y");
        let x = sym("dd_z_x");
        let f = normalize::pow(x.clone(), Expr::int(3));
        let dir = vec![Expr::int(0), Expr::int(0)];
        let result = directional_derivative(&f, &[xi, yi], &dir);
        assert!(
            result.is_zero(),
            "directional derivative with zero dir should be 0"
        );
    }

    #[test]
    fn test_directional_diagonal() {
        // f = x + y,  dir = (1, 1)  →  Dᵥf = 1*1 + 1*1 = 2
        let xi = id("dd_diag_x");
        let yi = id("dd_diag_y");
        let x = sym("dd_diag_x");
        let y = sym("dd_diag_y");
        let f = normalize::add(x.clone(), y.clone());
        let dir = vec![Expr::int(1), Expr::int(1)];
        let result = directional_derivative(&f, &[xi, yi], &dir);
        assert_eq!(
            *result,
            Expr::Integer(SmallInt::from(2i64)),
            "D(1,1)(x+y) should be 2"
        );
    }

    // ── total_derivative ──────────────────────────────────────────────────

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
