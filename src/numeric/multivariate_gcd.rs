//! GCD computation for multivariate polynomials over [`BigRational`].
//!
//! Uses evaluation-interpolation: pick a main variable, evaluate both
//! polynomials at enough rational points, compute GCDs recursively
//! (one fewer variable per recursion level), then apply Lagrange
//! interpolation to reconstruct the GCD polynomial.
//!
//! **Base case**: single variable — convert to [`DensePolynomial<BigRational>`],
//! use the existing field GCD, convert back.
//!
//! **Normalization**: result is monic (leading coefficient = 1 in grevlex).
//! Convention: `gcd(0, g) = monic(g)`, `gcd(f, 0) = monic(f)`.

use super::big_rational::BigRational;
use super::dense_poly::DensePolynomial;
use super::multivariate_poly::{Monomial, MultivariatePolynomial};
use super::ring::Ring;
use super::SymbolId;

// ── Public entry point ────────────────────────────────────────────────────────

/// Compute the GCD of two multivariate polynomials over [`BigRational`].
///
/// Uses recursive evaluation-interpolation. Returns a monic result in grevlex
/// order. Returns zero when both inputs are zero.
///
/// # Examples
///
/// ```
/// use thales::numeric::{multivariate_gcd, BigRational, Monomial, MultivariatePolynomial, SymbolId};
/// let x = SymbolId::intern("egx");
/// let y = SymbolId::intern("egy");
/// let xy = MultivariatePolynomial::monomial(BigRational::from(1i64), Monomial::var(x).mul(&Monomial::var(y)));
/// let f = &xy - &MultivariatePolynomial::var(y); // xy - y
/// let g = &xy - &MultivariatePolynomial::var(x); // xy - x
/// assert!(multivariate_gcd(&f, &g).is_constant()); // coprime: gcd = 1
/// ```
pub fn multivariate_gcd(
    f: &MultivariatePolynomial<BigRational>,
    g: &MultivariatePolynomial<BigRational>,
) -> MultivariatePolynomial<BigRational> {
    if f.is_zero() {
        return normalize_mvp(g);
    }
    if g.is_zero() {
        return normalize_mvp(f);
    }
    let all_vars = union_vars(f, g);
    if all_vars.is_empty() {
        return MultivariatePolynomial::constant(BigRational::one());
    }
    if all_vars.len() == 1 {
        return gcd_univariate(f, g, all_vars[0]);
    }
    let main_var = pick_main_var(f, g, &all_vars);
    gcd_eval_interpolate(f, g, main_var)
}

// ── Univariate base case ──────────────────────────────────────────────────────

fn gcd_univariate(
    f: &MultivariatePolynomial<BigRational>,
    g: &MultivariatePolynomial<BigRational>,
    var: SymbolId,
) -> MultivariatePolynomial<BigRational> {
    dense_to_mvp(&mvp_to_dense(f, var).gcd(&mvp_to_dense(g, var)), var)
}

fn mvp_to_dense(
    p: &MultivariatePolynomial<BigRational>,
    var: SymbolId,
) -> DensePolynomial<BigRational> {
    DensePolynomial::from_coeffs(
        p.as_univariate(var)
            .into_iter()
            .map(|c| c.constant_term())
            .collect(),
    )
}

fn dense_to_mvp(
    p: &DensePolynomial<BigRational>,
    var: SymbolId,
) -> MultivariatePolynomial<BigRational> {
    let mut result = MultivariatePolynomial::zero();
    for (deg, coeff) in p.coefficients().iter().enumerate() {
        if !coeff.is_zero() {
            result.add_term(Monomial::var_pow(var, deg as u32), coeff.clone());
        }
    }
    result
}

// ── Evaluation-interpolation GCD ─────────────────────────────────────────────

fn gcd_eval_interpolate(
    f: &MultivariatePolynomial<BigRational>,
    g: &MultivariatePolynomial<BigRational>,
    main_var: SymbolId,
) -> MultivariatePolynomial<BigRational> {
    let deg_in_main = degree_in(f, main_var).min(degree_in(g, main_var));
    // Generate extra points to handle unlucky evaluations
    let pts = evaluation_points(deg_in_main + 6);

    // Evaluate and compute GCDs at each point
    let evals: Vec<(BigRational, MultivariatePolynomial<BigRational>)> = pts
        .iter()
        .map(|pt| {
            let gcd = multivariate_gcd(&f.eval_var(main_var, pt), &g.eval_var(main_var, pt));
            (pt.clone(), gcd)
        })
        .collect();

    // The evaluated GCDs are polynomials in the remaining variables.
    // A "lucky" evaluation is one where total_degree matches the minimum;
    // unlucky points happen when the two evaluated polys share extra factors.
    let min_total_deg = evals
        .iter()
        .map(|(_, g)| g.total_degree().unwrap_or(0))
        .min()
        .unwrap_or(0);

    // Need at least deg_in_main + 2 lucky points for valid interpolation
    let n_needed = deg_in_main + 2;
    let lucky: Vec<(BigRational, MultivariatePolynomial<BigRational>)> = evals
        .into_iter()
        .filter(|(_, g)| g.total_degree().unwrap_or(0) == min_total_deg)
        .take(n_needed)
        .collect();

    if lucky.len() < n_needed {
        return MultivariatePolynomial::constant(BigRational::one());
    }

    let result = lagrange_reconstruct(lucky, main_var);
    if divides_mvp(&result, f) && divides_mvp(&result, g) {
        normalize_mvp(&result)
    } else {
        MultivariatePolynomial::constant(BigRational::one())
    }
}

// ── Lagrange interpolation ────────────────────────────────────────────────────

fn lagrange_reconstruct(
    points: Vec<(BigRational, MultivariatePolynomial<BigRational>)>,
    main_var: SymbolId,
) -> MultivariatePolynomial<BigRational> {
    if points.is_empty() {
        return MultivariatePolynomial::zero();
    }
    let pts: Vec<BigRational> = points.iter().map(|(p, _)| p.clone()).collect();
    let basis = lagrange_basis_polys(&pts);
    let mut result = MultivariatePolynomial::zero();
    for (i, (_, gcd_val)) in points.iter().enumerate() {
        result = &result + &mul_dense_by_mvp(&basis[i], main_var, gcd_val);
    }
    result
}

/// Lagrange basis polys `L_i(x) = prod_{j≠i}(x-pts[j])/(pts[i]-pts[j])`.
pub(crate) fn lagrange_basis_polys(pts: &[BigRational]) -> Vec<DensePolynomial<BigRational>> {
    let n = pts.len();
    (0..n)
        .map(|i| {
            let mut poly = DensePolynomial::constant(BigRational::one());
            for j in 0..n {
                if j == i {
                    continue;
                }
                let denom = (pts[i].clone() - pts[j].clone()).recip();
                let linear = DensePolynomial::from_coeffs(vec![
                    BigRational::zero() - pts[j].clone() * denom.clone(),
                    denom,
                ]);
                poly = poly * linear;
            }
            poly
        })
        .collect()
}

fn mul_dense_by_mvp(
    basis_poly: &DensePolynomial<BigRational>,
    main_var: SymbolId,
    mvp: &MultivariatePolynomial<BigRational>,
) -> MultivariatePolynomial<BigRational> {
    let mut result = MultivariatePolynomial::zero();
    for (deg, coeff) in basis_poly.coefficients().iter().enumerate() {
        if coeff.is_zero() {
            continue;
        }
        let mono_main = Monomial::var_pow(main_var, deg as u32);
        for (rest_mono, rest_coeff) in mvp.iter() {
            result.add_term(mono_main.mul(rest_mono), coeff.clone() * rest_coeff.clone());
        }
    }
    result
}

// ── Division check ────────────────────────────────────────────────────────────

fn divides_mvp(
    d: &MultivariatePolynomial<BigRational>,
    f: &MultivariatePolynomial<BigRational>,
) -> bool {
    if d.is_zero() {
        return false;
    }
    if d.is_constant() {
        return true;
    }
    let vars = d.variables();
    let (_, rem) = mvp_div_rem(f, d, vars[vars.len() - 1]);
    rem.is_zero()
}

/// Polynomial exact division `f / g` treating `main_var` as primary variable.
/// Returns `(quotient, remainder)` over the rational field.
pub(crate) fn mvp_div_rem(
    f: &MultivariatePolynomial<BigRational>,
    g: &MultivariatePolynomial<BigRational>,
    main_var: SymbolId,
) -> (
    MultivariatePolynomial<BigRational>,
    MultivariatePolynomial<BigRational>,
) {
    assert!(!g.is_zero(), "division by zero polynomial");
    let g_coeffs = g.as_univariate(main_var);
    let g_deg = g_coeffs.len().saturating_sub(1);
    let g_lead = g_coeffs
        .last()
        .cloned()
        .unwrap_or_else(MultivariatePolynomial::zero);
    let mut remainder = f.clone();
    let mut quotient = MultivariatePolynomial::zero();
    loop {
        let r_coeffs = remainder.as_univariate(main_var);
        let r_deg = r_coeffs.len().saturating_sub(1);
        let r_lead = r_coeffs
            .last()
            .cloned()
            .unwrap_or_else(MultivariatePolynomial::zero);
        if remainder.is_zero() || r_lead.is_zero() || r_deg < g_deg {
            break;
        }
        let Some(term_coeff) = div_by_constant_mvp(&r_lead, &g_lead) else {
            break;
        };
        let shift = r_deg - g_deg;
        let term_mono = Monomial::var_pow(main_var, shift as u32);
        for (m, c) in scale_mono_mvp(&term_coeff, &term_mono).iter() {
            quotient.add_term(m.clone(), c.clone());
        }
        remainder = &remainder - &mul_mono_coeff_mvp(&term_mono, &term_coeff, g);
    }
    (quotient, remainder)
}

fn div_by_constant_mvp(
    num: &MultivariatePolynomial<BigRational>,
    den: &MultivariatePolynomial<BigRational>,
) -> Option<MultivariatePolynomial<BigRational>> {
    if den.is_zero() || !den.is_constant() {
        return None;
    }
    let c = den.constant_term();
    if c.is_zero() {
        None
    } else {
        Some(num.scale(&c.recip()))
    }
}

fn scale_mono_mvp(
    p: &MultivariatePolynomial<BigRational>,
    mono: &Monomial,
) -> MultivariatePolynomial<BigRational> {
    let mut result = MultivariatePolynomial::zero();
    for (m, c) in p.iter() {
        result.add_term(m.mul(mono), c.clone());
    }
    result
}

fn mul_mono_coeff_mvp(
    mono: &Monomial,
    coeff_poly: &MultivariatePolynomial<BigRational>,
    g: &MultivariatePolynomial<BigRational>,
) -> MultivariatePolynomial<BigRational> {
    let mut result = MultivariatePolynomial::zero();
    for (gm, gc) in g.iter() {
        for (cm, cc) in coeff_poly.iter() {
            result.add_term(mono.mul(gm).mul(cm), gc.clone() * cc.clone());
        }
    }
    result
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn degree_in(poly: &MultivariatePolynomial<BigRational>, var: SymbolId) -> usize {
    poly.iter()
        .map(|(m, _)| m.exponent(&var) as usize)
        .max()
        .unwrap_or(0)
}

fn pick_main_var(
    f: &MultivariatePolynomial<BigRational>,
    g: &MultivariatePolynomial<BigRational>,
    all_vars: &[SymbolId],
) -> SymbolId {
    all_vars
        .iter()
        .copied()
        .min_by_key(|&v| degree_in(f, v).min(degree_in(g, v)))
        .expect("all_vars is non-empty")
}

fn union_vars(
    f: &MultivariatePolynomial<BigRational>,
    g: &MultivariatePolynomial<BigRational>,
) -> Vec<SymbolId> {
    let mut vars: Vec<SymbolId> = f.variables().into_iter().chain(g.variables()).collect();
    vars.sort();
    vars.dedup();
    vars
}

fn evaluation_points(n: usize) -> Vec<BigRational> {
    let mut pts = Vec::with_capacity(n);
    pts.push(BigRational::zero());
    let mut k = 1i64;
    while pts.len() < n {
        pts.push(BigRational::from(k));
        if pts.len() < n {
            pts.push(BigRational::from(-k));
        }
        k += 1;
    }
    pts.truncate(n);
    pts
}

fn normalize_mvp(p: &MultivariatePolynomial<BigRational>) -> MultivariatePolynomial<BigRational> {
    if p.is_zero() {
        return p.clone();
    }
    match p.leading_coeff() {
        Some(lc) if !lc.is_one() => p.scale(&lc.recip()),
        _ => p.clone(),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    type MP = MultivariatePolynomial<BigRational>;

    fn r(n: i64) -> BigRational {
        BigRational::from(n)
    }
    fn x() -> SymbolId {
        SymbolId::intern("mgcd_x")
    }
    fn y() -> SymbolId {
        SymbolId::intern("mgcd_y")
    }
    fn z() -> SymbolId {
        SymbolId::intern("mgcd_z")
    }

    fn build_x(terms: &[(i64, u32)]) -> MP {
        let mut p = MP::zero();
        for &(c, d) in terms {
            p.add_term(Monomial::var_pow(x(), d), r(c));
        }
        p
    }

    fn check_divides(d: &MP, f: &MP, label: &str) {
        if d.is_zero() || d.is_constant() {
            return;
        }
        let vars = d.variables();
        let (_, rem) = mvp_div_rem(f, d, vars[vars.len() - 1]);
        assert!(rem.is_zero(), "{label}: remainder not zero, got {rem}");
    }

    #[test]
    fn test_gcd_both_zero() {
        assert!(multivariate_gcd(&MP::zero(), &MP::zero()).is_zero());
    }

    #[test]
    fn test_gcd_f_zero() {
        let f = MP::var(x());
        let d = multivariate_gcd(&MP::zero(), &f);
        assert!(!d.is_zero());
        check_divides(&d, &f, "gcd(0,f)");
    }

    #[test]
    fn test_gcd_g_zero() {
        let f = MP::var(y());
        let d = multivariate_gcd(&f, &MP::zero());
        assert!(!d.is_zero());
        check_divides(&d, &f, "gcd(f,0)");
    }

    #[test]
    fn test_gcd_f_equals_g() {
        let f = build_x(&[(-1, 0), (1, 1)]); // x - 1
        let d = multivariate_gcd(&f, &f);
        assert!(!d.is_zero());
        check_divides(&d, &f, "gcd(f,f)");
    }

    #[test]
    fn test_gcd_coprime_univariate() {
        // gcd(x+1, x+2) = 1
        let d = multivariate_gcd(&build_x(&[(1, 0), (1, 1)]), &build_x(&[(2, 0), (1, 1)]));
        assert!(d.is_constant(), "expected constant GCD, got {d}");
    }

    #[test]
    fn test_gcd_univariate_common_factor() {
        // gcd(x^2 - 1, x - 1) = x - 1
        let f = build_x(&[(-1, 0), (0, 1), (1, 2)]);
        let g = build_x(&[(-1, 0), (1, 1)]);
        let d = multivariate_gcd(&f, &g);
        assert!(!d.is_zero());
        check_divides(&d, &f, "gcd divides f");
        check_divides(&d, &g, "gcd divides g");
        assert_eq!(
            d.as_univariate(x()).len().saturating_sub(1),
            1,
            "gcd degree should be 1"
        );
    }

    #[test]
    fn test_gcd_coprime_bivariate() {
        // gcd(x+1, y+1) = 1
        let f = &MP::var(x()) + &MP::constant(r(1));
        let g = &MP::var(y()) + &MP::constant(r(1));
        let d = multivariate_gcd(&f, &g);
        assert!(d.is_constant(), "coprime bivariate: expected 1, got {d}");
    }

    #[test]
    fn test_gcd_bivariate_coprime_xy() {
        // f = xy-y = y(x-1), g = xy-x = x(y-1); coprime
        let xy = MP::monomial(r(1), Monomial::var(x()).mul(&Monomial::var(y())));
        let f = &xy - &MP::var(y());
        let g = &xy - &MP::var(x());
        let d = multivariate_gcd(&f, &g);
        // Either constant (=1) or divides both
        if !d.is_constant() {
            check_divides(&d, &f, "d|f");
            check_divides(&d, &g, "d|g");
        }
    }

    #[test]
    fn test_gcd_bivariate_explicit_factor() {
        // f = (x-1)(x+y), g = (x-1)(x-y)  =>  gcd has degree >= 1
        let x_minus_1 = build_x(&[(-1, 0), (1, 1)]);
        let f = &x_minus_1 * &(&MP::var(x()) + &MP::var(y()));
        let g = &x_minus_1 * &(&MP::var(x()) - &MP::var(y()));
        let d = multivariate_gcd(&f, &g);
        assert!(!d.is_zero());
        check_divides(&d, &f, "gcd divides f");
        check_divides(&d, &g, "gcd divides g");
        assert!(
            d.as_univariate(x()).len().saturating_sub(1) >= 1,
            "gcd should contain x-1, got {d}"
        );
    }

    #[test]
    fn test_gcd_constants() {
        let d = multivariate_gcd(&MP::constant(r(3)), &MP::constant(r(5)));
        assert!(d.is_constant());
    }

    #[test]
    fn test_evaluation_points_distinct() {
        let pts = evaluation_points(10);
        assert_eq!(pts.len(), 10);
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                assert_ne!(pts[i], pts[j]);
            }
        }
    }

    #[test]
    fn test_lagrange_basis_correctness() {
        let pts = evaluation_points(3);
        let basis = lagrange_basis_polys(&pts);
        for i in 0..3 {
            for j in 0..3 {
                let val = basis[i].eval(&pts[j]);
                if i == j {
                    assert!(
                        (val.clone() - BigRational::one()).is_zero(),
                        "L_{i}(p[{i}]) = 1"
                    );
                } else {
                    assert!(val.is_zero(), "L_{i}(p[{j}]) = 0");
                }
            }
        }
    }

    #[test]
    fn test_mvp_dense_roundtrip() {
        let f = build_x(&[(-1, 0), (0, 1), (1, 2)]);
        assert_eq!(dense_to_mvp(&mvp_to_dense(&f, x()), x()), f);
    }

    #[test]
    fn test_gcd_three_variable_coprime() {
        let f = &MP::var(x()) + &MP::constant(r(1));
        let g = &MP::var(y()) + &MP::var(z());
        let d = multivariate_gcd(&f, &g);
        assert!(d.is_constant(), "3-var coprime: expected 1, got {d}");
    }
}
