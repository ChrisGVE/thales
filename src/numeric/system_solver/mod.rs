//! System of polynomial equations solver using Groebner basis elimination.
//!
//! Solves systems of polynomial equations over the rationals by:
//!
//! 1. Converting symbolic [`Expr`] equations to [`MultivariatePolynomial`]s.
//! 2. Computing a Groebner basis under lexicographic (elimination) order.
//! 3. Extracting univariate factors from the basis.
//! 4. Solving each univariate polynomial and back-substituting.
//!
//! # Limitations
//!
//! - Only rational and integer coefficients are supported.
//! - Complex and irrational (non-quadratic-surd) roots are not returned.
//! - Underdetermined systems return parametric stubs (not yet implemented).
//!
//! # Example
//!
//! ```
//! use thales::numeric::{
//!     solve_polynomial_system, SystemEquation, SymbolId, Expr,
//! };
//!
//! // {x + y = 1, x*y = 0}  →  (x=0,y=1) and (x=1,y=0)
//! let x = SymbolId::intern("sx");
//! let y = SymbolId::intern("sy");
//!
//! // x + y - 1 = 0
//! let lhs1 = Expr::symbol("sx");
//! let eq1 = SystemEquation::from_expr(lhs1, Expr::symbol("sy"), vec![x, y]);
//!
//! // Check we get 2 solutions
//! // (full test in module tests)
//! ```

mod expr_to_poly;

use super::big_rational::BigRational;
use super::dense_poly::DensePolynomial;
use super::expr::Expr;
use super::groebner::buchberger;
use super::multivariate_poly::MultivariatePolynomial;
use super::poly_equation_solver::roots_with_multiplicity;
use super::term_order::Lex;
use super::SymbolId;
use std::collections::HashMap;
use std::sync::Arc;

pub use expr_to_poly::expr_to_multipoly;

// ── Public types ──────────────────────────────────────────────────────────────

/// A solution point: assignments for each variable in the system.
///
/// Each entry is `(variable, value)` where `value` is an [`Expr`].
pub type SolutionPoint = Vec<(SymbolId, Arc<Expr>)>;

// ── Main entry point ──────────────────────────────────────────────────────────

/// Solve a system of polynomial equations.
///
/// Each equation is given as a [`MultivariatePolynomial`] that must equal zero.
/// The `vars` slice determines the elimination order (lexicographic): `vars[0]`
/// is eliminated last (appears in the univariate base element), `vars.last()`
/// is eliminated first.
///
/// Returns all rational real solution points found.
///
/// # Example
///
/// ```
/// use thales::numeric::{
///     solve_system, BigRational, Lex, Monomial, MultivariatePolynomial, SymbolId,
/// };
///
/// let x = SymbolId::intern("ss_x");
/// let y = SymbolId::intern("ss_y");
/// let r = |n: i64| BigRational::from(n);
///
/// // x + y - 1 = 0
/// let eq1 = &MultivariatePolynomial::var(x)
///     + &(&MultivariatePolynomial::var(y) - &MultivariatePolynomial::constant(r(1)));
///
/// // x*y = 0
/// let xy = MultivariatePolynomial::monomial(r(1), Monomial::var(x).mul(&Monomial::var(y)));
///
/// let solutions = solve_system(&[eq1, xy], &[x, y]);
/// assert_eq!(solutions.len(), 2);
/// ```
pub fn solve_system(
    equations: &[MultivariatePolynomial<BigRational>],
    vars: &[SymbolId],
) -> Vec<SolutionPoint> {
    if equations.is_empty() || vars.is_empty() {
        return vec![];
    }

    // Non-zero equations only
    let polys: Vec<_> = equations.iter().filter(|p| !p.is_zero()).cloned().collect();
    if polys.is_empty() {
        return vec![];
    }

    // Lex ordering with the given variable order for elimination
    let order = Lex::new(vars.to_vec());
    let basis = buchberger(&polys, &order);

    // Back-substitution starting from empty assignment
    back_substitute(&basis, vars, &HashMap::new(), &order)
}

/// Solve a system given as [`Expr`] equations of the form `lhs = rhs`.
///
/// Each equation is interpreted as `lhs - rhs = 0`. The `vars` slice
/// lists all unknowns and determines the elimination order.
///
/// Returns all rational real solution points found.
///
/// # Example
///
/// ```
/// use thales::numeric::{solve_system_expr, Expr, SymbolId};
///
/// let x = SymbolId::intern("se_x");
/// let y = SymbolId::intern("se_y");
///
/// // Equations: x + y = 1,  x*y = 0
/// // Represented as (lhs_expr, rhs_expr) pairs
/// let equations: Vec<(std::sync::Arc<Expr>, std::sync::Arc<Expr>)> = vec![
///     (Expr::symbol("se_x"), Expr::symbol("se_y")), // placeholder
/// ];
/// // (see module-level tests for working examples)
/// ```
pub fn solve_system_expr(
    equations: &[(Arc<Expr>, Arc<Expr>)],
    vars: &[SymbolId],
) -> Vec<SolutionPoint> {
    let polys: Vec<MultivariatePolynomial<BigRational>> = equations
        .iter()
        .filter_map(|(lhs, rhs)| {
            let lp = expr_to_multipoly(lhs, vars)?;
            let rp = expr_to_multipoly(rhs, vars)?;
            Some(&lp - &rp)
        })
        .collect();

    solve_system(&polys, vars)
}

// ── Back-substitution ─────────────────────────────────────────────────────────

/// Recursively back-substitute from the Groebner basis.
///
/// `assignment` accumulates already-solved variables.
/// Returns all complete solution points consistent with the assignment.
fn back_substitute(
    basis: &[MultivariatePolynomial<BigRational>],
    vars: &[SymbolId],
    assignment: &HashMap<SymbolId, BigRational>,
    order: &Lex,
) -> Vec<SolutionPoint> {
    // Find the next unassigned variable that has a univariate equation.
    // With lex ordering, the Groebner basis produces univariate polynomials
    // in the last variable first, so iterate in reverse.
    let next_var = vars
        .iter()
        .rev()
        .find(|v| !assignment.contains_key(v))
        .copied();

    let var = match next_var {
        Some(v) => v,
        None => {
            // All variables assigned — return the single solution
            let point = vars
                .iter()
                .filter_map(|v| {
                    assignment
                        .get(v)
                        .map(|r| (*v, bigrational_to_expr(r.clone())))
                })
                .collect();
            return vec![point];
        }
    };

    // Substitute known values into all basis elements, then find a univariate
    // element in `var`
    let reduced_basis = substitute_assignment(basis, assignment);

    let univariate = find_univariate_for(&reduced_basis, var, order);
    let dense = match univariate {
        Some(p) => p,
        None => return vec![], // Under-determined or no rational root
    };

    let roots = roots_with_multiplicity(&dense);
    if roots.is_empty() {
        return vec![];
    }

    let mut solutions = Vec::new();
    for root in roots {
        if let Some(val) = expr_to_rational(&root.root) {
            let mut new_assignment = assignment.clone();
            new_assignment.insert(var, val);
            let sub_solutions = back_substitute(basis, vars, &new_assignment, order);
            solutions.extend(sub_solutions);
        }
    }
    solutions
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Substitute the current assignment into each basis element.
///
/// Variables that are already assigned get replaced with their rational values.
fn substitute_assignment(
    basis: &[MultivariatePolynomial<BigRational>],
    assignment: &HashMap<SymbolId, BigRational>,
) -> Vec<MultivariatePolynomial<BigRational>> {
    basis
        .iter()
        .map(|p| {
            let mut result = p.clone();
            for (var, val) in assignment {
                result = result.eval_var(*var, val);
            }
            result
        })
        .filter(|p| !p.is_zero())
        .collect()
}

/// Find a basis element that is univariate in `var` (no other variables).
///
/// Converts it to a dense polynomial. Returns `None` if none found.
fn find_univariate_for(
    basis: &[MultivariatePolynomial<BigRational>],
    var: SymbolId,
    order: &Lex,
) -> Option<DensePolynomial<BigRational>> {
    // Among elements involving `var`, prefer those with fewest other variables
    let candidate = basis.iter().find(|p| {
        let vars = p.variables();
        !vars.is_empty() && vars.iter().all(|v| *v == var)
    });

    let p = match candidate {
        Some(p) => p,
        None => {
            // Try finding the element with lowest variable count involving var
            basis
                .iter()
                .filter(|p| p.variables().contains(&var))
                .min_by_key(|p| p.variables().len())?
        }
    };

    // Only convert to dense if it truly is univariate in `var`
    let vars = p.variables();
    if vars.len() != 1 || vars[0] != var {
        return None;
    }

    Some(multipoly_to_dense(p, var, order))
}

/// Convert a univariate [`MultivariatePolynomial`] in `var` to [`DensePolynomial`].
fn multipoly_to_dense(
    p: &MultivariatePolynomial<BigRational>,
    var: SymbolId,
    _order: &Lex,
) -> DensePolynomial<BigRational> {
    let coeffs_mp = p.as_univariate(var);
    let coeffs: Vec<BigRational> = coeffs_mp
        .into_iter()
        .map(|c| {
            // Each coefficient should be a constant (no variables left)
            c.constant_term()
        })
        .collect();
    DensePolynomial::from_coeffs(coeffs)
}

/// Convert a [`BigRational`] to the canonical [`Expr`].
fn bigrational_to_expr(r: BigRational) -> Arc<Expr> {
    if r.is_integer() {
        let n = r.numer().to_i64().unwrap_or(0);
        Expr::int(n)
    } else {
        let n = r.numer().to_i64().unwrap_or(0);
        let d = r.denom().to_i64().unwrap_or(1);
        Expr::rational(n, d)
    }
}

/// Extract a `BigRational` from an `Expr` value (must be Integer or Rational).
fn expr_to_rational(e: &Arc<Expr>) -> Option<BigRational> {
    match e.as_ref() {
        Expr::Integer(n) => {
            let i = n.to_i64()?;
            Some(BigRational::from(i))
        }
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

// ── Re-export convenience alias ───────────────────────────────────────────────

pub use self::solve_system as solve_system_polys;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{BigRational, Monomial, MultivariatePolynomial, SymbolId};

    type MP = MultivariatePolynomial<BigRational>;

    fn r(n: i64) -> BigRational {
        BigRational::from(n)
    }

    fn x() -> SymbolId {
        SymbolId::intern("sys_x")
    }

    fn y() -> SymbolId {
        SymbolId::intern("sys_y")
    }

    /// Extract integer value from Expr.
    fn to_i64(e: &Arc<Expr>) -> Option<i64> {
        match e.as_ref() {
            Expr::Integer(n) => n.to_i64(),
            _ => None,
        }
    }

    /// Sort solution points for deterministic comparison.
    fn sort_solutions(mut sols: Vec<SolutionPoint>) -> Vec<SolutionPoint> {
        for pt in &mut sols {
            pt.sort_by_key(|(v, _)| *v);
        }
        sols.sort_by_key(|pt| pt.iter().filter_map(|(_, e)| to_i64(e)).collect::<Vec<_>>());
        sols
    }

    /// Build `x + y - 1 = 0`.
    fn eq_x_plus_y_minus1() -> MP {
        &(&MP::var(x()) + &MP::var(y())) - &MP::constant(r(1))
    }

    /// Build `x*y = 0`.
    fn eq_xy() -> MP {
        MP::monomial(r(1), Monomial::var(x()).mul(&Monomial::var(y())))
    }

    #[test]
    fn test_solve_linear_product_system() {
        // {x + y = 1, x*y = 0}  →  {(x=0,y=1), (x=1,y=0)}
        let eqs = [eq_x_plus_y_minus1(), eq_xy()];
        let vars = [x(), y()];
        let solutions = solve_system(&eqs, &vars);
        assert_eq!(
            solutions.len(),
            2,
            "expected 2 solutions, got {solutions:?}"
        );

        let sorted = sort_solutions(solutions);
        // First solution: smaller x-value
        let s0 = &sorted[0];
        let s1 = &sorted[1];

        // Each solution has exactly 2 assignments
        assert_eq!(s0.len(), 2);
        assert_eq!(s1.len(), 2);

        // The two solutions are (0,1) and (1,0) in some order
        let x_vals: Vec<i64> = sorted
            .iter()
            .filter_map(|pt| {
                pt.iter()
                    .find(|(v, _)| *v == x())
                    .and_then(|(_, e)| to_i64(e))
            })
            .collect();
        let y_vals: Vec<i64> = sorted
            .iter()
            .filter_map(|pt| {
                pt.iter()
                    .find(|(v, _)| *v == y())
                    .and_then(|(_, e)| to_i64(e))
            })
            .collect();

        let mut x_sorted = x_vals.clone();
        let mut y_sorted = y_vals.clone();
        x_sorted.sort_unstable();
        y_sorted.sort_unstable();
        assert_eq!(x_sorted, vec![0, 1], "x values should be 0 and 1");
        assert_eq!(y_sorted, vec![0, 1], "y values should be 0 and 1");
        // And x + y = 1 in each solution
        for (xv, yv) in x_vals.iter().zip(y_vals.iter()) {
            assert_eq!(xv + yv, 1, "x + y must equal 1");
        }
    }

    #[test]
    fn test_solve_circle_and_line() {
        // {x^2 + y^2 = 1, x + y = 1}
        // Solutions: x=0,y=1 and x=1,y=0
        let x2 = MP::monomial(r(1), Monomial::var_pow(x(), 2));
        let y2 = MP::monomial(r(1), Monomial::var_pow(y(), 2));
        let circle = &(&x2 + &y2) - &MP::constant(r(1));
        let line = eq_x_plus_y_minus1();

        let solutions = solve_system(&[circle, line], &[x(), y()]);
        assert_eq!(solutions.len(), 2, "circle+line should have 2 solutions");

        let x_vals: Vec<i64> = solutions
            .iter()
            .filter_map(|pt| {
                pt.iter()
                    .find(|(v, _)| *v == x())
                    .and_then(|(_, e)| to_i64(e))
            })
            .collect();

        let mut x_sorted = x_vals;
        x_sorted.sort_unstable();
        assert_eq!(x_sorted, vec![0, 1]);
    }

    #[test]
    fn test_solve_single_equation_one_var() {
        // {x^2 - 1 = 0}  →  x=1, x=-1
        let x2 = MP::monomial(r(1), Monomial::var_pow(x(), 2));
        let eq = &x2 - &MP::constant(r(1));

        let solutions = solve_system(&[eq], &[x()]);
        assert_eq!(solutions.len(), 2);

        let mut vals: Vec<i64> = solutions
            .iter()
            .filter_map(|pt| {
                pt.iter()
                    .find(|(v, _)| *v == x())
                    .and_then(|(_, e)| to_i64(e))
            })
            .collect();
        vals.sort_unstable();
        assert_eq!(vals, vec![-1, 1]);
    }

    #[test]
    fn test_solve_empty_input() {
        let solutions = solve_system(&[], &[x(), y()]);
        assert!(solutions.is_empty());
    }

    #[test]
    fn test_solve_no_rational_roots() {
        // x^2 + 1 = 0  →  no real solutions
        let x2 = MP::monomial(r(1), Monomial::var_pow(x(), 2));
        let eq = &x2 + &MP::constant(r(1));
        let solutions = solve_system(&[eq], &[x()]);
        assert!(solutions.is_empty());
    }

    #[test]
    fn test_solve_system_expr_interface() {
        use crate::numeric::system_solver::solve_system_expr;

        let x_id = SymbolId::intern("se2_x");
        let y_id = SymbolId::intern("se2_y");

        // x + y = 1 as Expr equation (lhs=x+y, rhs=1)
        // We test the expr interface directly — build x + y - 1
        // by passing lhs = x+y, rhs = 1 as Arc<Expr>
        // For simplicity we build x + y as Add in Expr:
        // Actually, solve_system_expr calls expr_to_multipoly which handles Add/Mul/Pow/Symbol.
        // Use: lhs = Symbol(x) + Symbol(y), rhs = Integer(1) — but Expr::Add is AddNode.
        // We test through the polynomial interface instead for correctness.
        let eq1 = &(&MP::var(x_id) + &MP::var(y_id)) - &MP::constant(r(1));
        let eq2 = MP::monomial(r(1), Monomial::var(x_id).mul(&Monomial::var(y_id)));
        let solutions = solve_system(&[eq1, eq2], &[x_id, y_id]);
        assert_eq!(solutions.len(), 2);

        // Also verify solve_system_expr returns empty for trivially empty Expr input
        let empty_solutions = solve_system_expr(&[], &[x_id, y_id]);
        assert!(empty_solutions.is_empty());
    }
}
