//! Coordinate change engine — substitutes Cartesian variables with
//! curvilinear coordinates and returns the corresponding Jacobian determinant.
//!
//! # Design
//!
//! All internal computation operates on `Arc<Expr>` (Architecture Rule 1).
//! `Expression` inputs are compiled at entry and the results are decompiled
//! back to `Expression` at exit (Architecture Rule 2).
//!
//! The caller receives `(substituted_expr, jacobian_det)` as a pair of
//! `Expression` values and multiplies them according to the integration context.
//!
//! # Supported systems
//!
//! | System      | Map                                               | |J|            |
//! |-------------|---------------------------------------------------|--------------------|
//! | Polar2D     | x = r·cos(θ), y = r·sin(θ)                       | r                  |
//! | Cylindrical | x = ρ·cos(φ), y = ρ·sin(φ), z = z               | ρ                  |
//! | Spherical   | x = r·sin(θ)·cos(φ), y = r·sin(θ)·sin(φ), z = r·cos(θ) | r²·sin(θ)  |
//! | Cartesian2D | identity                                          | 1                  |
//! | Cartesian3D | identity                                          | 1                  |
//! | Custom      | identity (caller supplies own Jacobian)           | 1                  |

#![allow(dead_code)]

use std::sync::Arc;

use crate::ast::Expression;
use crate::numeric::compile::{compile, decompile};
use crate::numeric::normalize;
use crate::numeric::substitute::substitute;
use crate::numeric::{Expr, FuncId, SymbolId};
use crate::transforms::jacobian::jacobian_determinant;
use crate::transforms::CoordSystem;

// ── Public API ────────────────────────────────────────────────────────────────

/// Error type for coordinate-change operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeCoordError {
    /// The number of `from_vars` does not match the dimensionality expected by
    /// `system`.
    DimensionMismatch {
        /// System name for the error message.
        system: &'static str,
        /// Expected number of variables.
        expected: usize,
        /// Actual number of variables supplied.
        got: usize,
    },
    /// The number of `from_vars` and `to_vars` differ.
    VarCountMismatch { from_count: usize, to_count: usize },
}

impl std::fmt::Display for ChangeCoordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch {
                system,
                expected,
                got,
            } => write!(
                f,
                "coordinate system `{system}` requires {expected} variables, got {got}"
            ),
            Self::VarCountMismatch {
                from_count,
                to_count,
            } => write!(
                f,
                "from_vars ({from_count}) and to_vars ({to_count}) must have the same length"
            ),
        }
    }
}

/// Change coordinates in `expr`, returning `(substituted_expr, jacobian_det)`.
///
/// `from_vars` names the Cartesian variables currently appearing in `expr`
/// (e.g. `["x", "y"]`).  `to_vars` names the new coordinate variables in
/// the target system (e.g. `["r", "theta"]`).  `system` selects the built-in
/// coordinate map that defines the substitution and its Jacobian.
///
/// The Jacobian determinant is returned as a separate `Expression` so that
/// the caller can choose how to combine it (e.g. insert it into an integrand
/// or display it independently).
///
/// # Errors
///
/// Returns [`ChangeCoordError::VarCountMismatch`] when `from_vars.len() !=
/// to_vars.len()`, or [`ChangeCoordError::DimensionMismatch`] when the
/// variable count is incompatible with `system`.
pub fn change_coords(
    expr: &Expression,
    from_vars: &[String],
    to_vars: &[String],
    system: CoordSystem,
) -> Result<(Expression, Expression), ChangeCoordError> {
    if from_vars.len() != to_vars.len() {
        return Err(ChangeCoordError::VarCountMismatch {
            from_count: from_vars.len(),
            to_count: to_vars.len(),
        });
    }

    // Build the forward map (new-coord expressions for each old Cartesian var)
    // and intern the new-coord variable IDs.
    let (forward_map, old_ids, new_ids) = build_map(from_vars, to_vars, system)?;

    // Compile the integrand once.
    let mut arc_expr = compile(expr);

    // Apply each substitution: replace from_vars[i] with forward_map[i].
    for (from_id, replacement) in old_ids.iter().zip(forward_map.iter()) {
        arc_expr = substitute(&arc_expr, *from_id, replacement);
    }

    // Jacobian determinant of the forward map w.r.t. new_vars.
    let jac = jacobian_determinant(&forward_map, &old_ids, &new_ids);

    Ok((decompile(&arc_expr), decompile(&jac)))
}

// ── Coordinate maps ───────────────────────────────────────────────────────────

/// Build `(forward_map, old_symbol_ids, new_symbol_ids)` for the requested
/// system.
///
/// `forward_map[i]` is the `Arc<Expr>` expression for `from_vars[i]` in terms
/// of `to_vars`.
fn build_map(
    from_vars: &[String],
    to_vars: &[String],
    system: CoordSystem,
) -> Result<(Vec<Arc<Expr>>, Vec<SymbolId>, Vec<SymbolId>), ChangeCoordError> {
    let n = from_vars.len();

    let old_ids: Vec<SymbolId> = from_vars.iter().map(|v| SymbolId::intern(v)).collect();
    let new_ids: Vec<SymbolId> = to_vars.iter().map(|v| SymbolId::intern(v)).collect();

    let map = match system {
        CoordSystem::Polar2D => {
            if n != 2 {
                return Err(ChangeCoordError::DimensionMismatch {
                    system: "Polar2D",
                    expected: 2,
                    got: n,
                });
            }
            polar2d_map(&to_vars[0], &to_vars[1])
        }

        CoordSystem::Cylindrical => {
            if n != 3 {
                return Err(ChangeCoordError::DimensionMismatch {
                    system: "Cylindrical",
                    expected: 3,
                    got: n,
                });
            }
            cylindrical_map(&to_vars[0], &to_vars[1], &to_vars[2])
        }

        CoordSystem::Spherical => {
            if n != 3 {
                return Err(ChangeCoordError::DimensionMismatch {
                    system: "Spherical",
                    expected: 3,
                    got: n,
                });
            }
            spherical_map(&to_vars[0], &to_vars[1], &to_vars[2])
        }

        // Identity maps — Cartesian and Custom pass through unchanged.
        CoordSystem::Cartesian2D => {
            if n != 2 {
                return Err(ChangeCoordError::DimensionMismatch {
                    system: "Cartesian2D",
                    expected: 2,
                    got: n,
                });
            }
            identity_map(to_vars)
        }

        CoordSystem::Cartesian3D => {
            if n != 3 {
                return Err(ChangeCoordError::DimensionMismatch {
                    system: "Cartesian3D",
                    expected: 3,
                    got: n,
                });
            }
            identity_map(to_vars)
        }

        CoordSystem::Parabolic2D => {
            if n != 2 {
                return Err(ChangeCoordError::DimensionMismatch {
                    system: "Parabolic2D",
                    expected: 2,
                    got: n,
                });
            }
            parabolic2d_map(&to_vars[0], &to_vars[1])
        }

        CoordSystem::Elliptic2D => {
            if n != 2 {
                return Err(ChangeCoordError::DimensionMismatch {
                    system: "Elliptic2D",
                    expected: 2,
                    got: n,
                });
            }
            elliptic2d_map(&to_vars[0], &to_vars[1])
        }

        CoordSystem::Custom => {
            // Custom: treat as identity so callers get a sensible fall-through.
            identity_map(to_vars)
        }
    };

    Ok((map, old_ids, new_ids))
}

// ── Map constructors ──────────────────────────────────────────────────────────

/// Polar 2D: x = r·cos(θ),  y = r·sin(θ).
fn polar2d_map(r_name: &str, theta_name: &str) -> Vec<Arc<Expr>> {
    let r = Expr::symbol(r_name);
    let theta = Expr::symbol(theta_name);
    let cos_theta = Expr::func(FuncId::Cos, vec![theta.clone()]);
    let sin_theta = Expr::func(FuncId::Sin, vec![theta]);
    let x = normalize::mul(r.clone(), cos_theta);
    let y = normalize::mul(r, sin_theta);
    vec![x, y]
}

/// Cylindrical: x = ρ·cos(φ),  y = ρ·sin(φ),  z = z.
fn cylindrical_map(rho_name: &str, phi_name: &str, z_name: &str) -> Vec<Arc<Expr>> {
    let rho = Expr::symbol(rho_name);
    let phi = Expr::symbol(phi_name);
    let z = Expr::symbol(z_name);
    let cos_phi = Expr::func(FuncId::Cos, vec![phi.clone()]);
    let sin_phi = Expr::func(FuncId::Sin, vec![phi]);
    let x = normalize::mul(rho.clone(), cos_phi);
    let y = normalize::mul(rho, sin_phi);
    vec![x, y, z]
}

/// Spherical: x = r·sin(θ)·cos(φ),  y = r·sin(θ)·sin(φ),  z = r·cos(θ).
///
/// Convention: θ is the polar angle from the z-axis (physics / ISO 80000),
/// φ is the azimuthal angle.
fn spherical_map(r_name: &str, theta_name: &str, phi_name: &str) -> Vec<Arc<Expr>> {
    let r = Expr::symbol(r_name);
    let theta = Expr::symbol(theta_name);
    let phi = Expr::symbol(phi_name);
    let sin_theta = Expr::func(FuncId::Sin, vec![theta.clone()]);
    let cos_theta = Expr::func(FuncId::Cos, vec![theta]);
    let cos_phi = Expr::func(FuncId::Cos, vec![phi.clone()]);
    let sin_phi = Expr::func(FuncId::Sin, vec![phi]);
    // x = r·sin(θ)·cos(φ)
    let x = normalize::mul(normalize::mul(r.clone(), sin_theta.clone()), cos_phi);
    // y = r·sin(θ)·sin(φ)
    let y = normalize::mul(normalize::mul(r.clone(), sin_theta), sin_phi);
    // z = r·cos(θ)
    let z = normalize::mul(r, cos_theta);
    vec![x, y, z]
}

/// Parabolic 2D: x = (u² - v²)/2,  y = u·v.
fn parabolic2d_map(u_name: &str, v_name: &str) -> Vec<Arc<Expr>> {
    let u = Expr::symbol(u_name);
    let v = Expr::symbol(v_name);
    let u_sq = normalize::pow(u.clone(), Expr::int(2));
    let v_sq = normalize::pow(v.clone(), Expr::int(2));
    let diff = normalize::add(u_sq, normalize::mul(Expr::int(-1), v_sq));
    let x = normalize::mul(
        Arc::new(Expr::Rational(crate::numeric::BigRational::from_i64(1, 2))),
        diff,
    );
    let y = normalize::mul(u, v);
    vec![x, y]
}

/// Elliptic 2D: x = a·cosh(μ)·cos(ν),  y = a·sinh(μ)·sin(ν).
/// The focal parameter `a` is treated as a free symbol named `"a"`.
fn elliptic2d_map(mu_name: &str, nu_name: &str) -> Vec<Arc<Expr>> {
    let a = Expr::symbol("a");
    let mu = Expr::symbol(mu_name);
    let nu = Expr::symbol(nu_name);
    let cosh_mu = Expr::func(FuncId::Cosh, vec![mu.clone()]);
    let sinh_mu = Expr::func(FuncId::Sinh, vec![mu]);
    let cos_nu = Expr::func(FuncId::Cos, vec![nu.clone()]);
    let sin_nu = Expr::func(FuncId::Sin, vec![nu]);
    let x = normalize::mul(normalize::mul(a.clone(), cosh_mu), cos_nu);
    let y = normalize::mul(normalize::mul(a, sinh_mu), sin_nu);
    vec![x, y]
}

/// Identity map: each new var maps to itself.
fn identity_map(to_vars: &[String]) -> Vec<Arc<Expr>> {
    to_vars.iter().map(|v| Expr::symbol(v)).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Expression, Variable};
    use crate::numeric::evaluation::evaluate;
    use crate::numeric::SymbolId;
    use std::collections::HashMap;

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn add(a: Expression, b: Expression) -> Expression {
        Expression::Binary(BinaryOp::Add, Box::new(a), Box::new(b))
    }

    fn mul(a: Expression, b: Expression) -> Expression {
        Expression::Binary(BinaryOp::Mul, Box::new(a), Box::new(b))
    }

    fn pow(base: Expression, exp: Expression) -> Expression {
        Expression::Power(Box::new(base), Box::new(exp))
    }

    // ── Polar2D ───────────────────────────────────────────────────────────────

    /// x² + y² in polar → r² (after substitution). Jacobian = r.
    #[test]
    fn fast_polar_x_sq_plus_y_sq_becomes_r_sq() {
        let expr = add(pow(var("x"), int(2)), pow(var("y"), int(2)));
        let (subst, jac) = change_coords(
            &expr,
            &["x".to_string(), "y".to_string()],
            &["r".to_string(), "theta".to_string()],
            CoordSystem::Polar2D,
        )
        .unwrap();

        // Evaluate substituted expr at r=3, theta=pi/4 → should equal 9 (r²)
        let r_id = SymbolId::intern("r");
        let theta_id = SymbolId::intern("theta");
        let subst_arc = compile(&subst);
        let mut env = HashMap::new();
        env.insert(r_id, 3.0_f64);
        env.insert(theta_id, std::f64::consts::PI / 4.0);
        let val = evaluate(&subst_arc, &env).unwrap();
        assert!(
            (val - 9.0).abs() < 1e-10,
            "x²+y² in polar at r=3 should be 9, got {val}"
        );

        // Jacobian at r=3 → should be 3.
        let jac_arc = compile(&jac);
        let mut env2 = HashMap::new();
        env2.insert(r_id, 3.0_f64);
        env2.insert(theta_id, 1.0_f64); // arbitrary angle
        let jac_val = evaluate(&jac_arc, &env2).unwrap();
        assert!(
            (jac_val - 3.0).abs() < 1e-10,
            "polar Jacobian at r=3 should be 3, got {jac_val}"
        );
    }

    /// Constant 1 in polar — substituted expr is still 1, Jacobian is r.
    #[test]
    fn fast_polar_constant_integrand() {
        let expr = int(1);
        let (subst, jac) = change_coords(
            &expr,
            &["x".to_string(), "y".to_string()],
            &["r".to_string(), "theta".to_string()],
            CoordSystem::Polar2D,
        )
        .unwrap();

        let subst_arc = compile(&subst);
        let empty: HashMap<SymbolId, f64> = HashMap::new();
        assert_eq!(evaluate(&subst_arc, &empty).unwrap(), 1.0);

        let r_id = SymbolId::intern("r");
        let theta_id = SymbolId::intern("theta");
        let jac_arc = compile(&jac);
        let mut env = HashMap::new();
        env.insert(r_id, 5.0_f64);
        // The Jacobian det is r·cos²(θ)+r·sin²(θ) = r before trig simplification,
        // so theta must be bound for numeric evaluation.  Any value is correct
        // (sin²+cos²=1), but we must provide it so evaluate() does not return None.
        env.insert(theta_id, 0.0_f64);
        assert!(
            (evaluate(&jac_arc, &env).unwrap() - 5.0).abs() < 1e-10,
            "polar Jacobian should equal r"
        );
    }

    // ── Spherical ─────────────────────────────────────────────────────────────

    /// Constant 1 in spherical — Jacobian should equal r²·sin(θ).
    #[test]
    fn fast_spherical_jacobian_is_r_sq_sin_theta() {
        let expr = int(1);
        let (_, jac) = change_coords(
            &expr,
            &["x".to_string(), "y".to_string(), "z".to_string()],
            &["r".to_string(), "theta".to_string(), "phi".to_string()],
            CoordSystem::Spherical,
        )
        .unwrap();

        // Evaluate at r=2, theta=pi/6, phi=pi/4 → 4 * sin(pi/6) = 4*0.5 = 2
        let r_id = SymbolId::intern("r");
        let theta_id = SymbolId::intern("theta");
        let phi_id = SymbolId::intern("phi");
        let r_val = 2.0_f64;
        let theta_val = std::f64::consts::PI / 6.0;
        let phi_val = std::f64::consts::PI / 4.0;
        let mut env = HashMap::new();
        env.insert(r_id, r_val);
        env.insert(theta_id, theta_val);
        env.insert(phi_id, phi_val);

        let jac_arc = compile(&jac);
        let val = evaluate(&jac_arc, &env).unwrap();
        let expected = r_val * r_val * theta_val.sin();
        assert!(
            (val - expected).abs() < 1e-10,
            "spherical Jacobian at r=2,θ=π/6 should be {expected}, got {val}"
        );
    }

    // ── Cylindrical ───────────────────────────────────────────────────────────

    /// Jacobian of cylindrical coords = ρ.
    #[test]
    fn fast_cylindrical_jacobian_is_rho() {
        let expr = int(1);
        let (_, jac) = change_coords(
            &expr,
            &["x".to_string(), "y".to_string(), "z".to_string()],
            &["rho".to_string(), "phi".to_string(), "z".to_string()],
            CoordSystem::Cylindrical,
        )
        .unwrap();

        let rho_id = SymbolId::intern("rho");
        let phi_id = SymbolId::intern("phi");
        let z_id = SymbolId::intern("z");
        let mut env = HashMap::new();
        env.insert(rho_id, 4.0_f64);
        env.insert(phi_id, 1.0_f64);
        env.insert(z_id, 0.0_f64);

        let jac_arc = compile(&jac);
        let val = evaluate(&jac_arc, &env).unwrap();
        assert!(
            (val - 4.0).abs() < 1e-10,
            "cylindrical Jacobian at rho=4 should be 4, got {val}"
        );
    }

    // ── Cartesian identity ────────────────────────────────────────────────────

    /// Cartesian2D identity: Jacobian = 1.
    #[test]
    fn fast_cartesian2d_jacobian_is_one() {
        let expr = add(var("x"), var("y"));
        let (_, jac) = change_coords(
            &expr,
            &["x".to_string(), "y".to_string()],
            &["u".to_string(), "v".to_string()],
            CoordSystem::Cartesian2D,
        )
        .unwrap();
        let jac_arc = compile(&jac);
        let empty: HashMap<SymbolId, f64> = HashMap::new();
        assert_eq!(evaluate(&jac_arc, &empty).unwrap(), 1.0);
    }

    // ── Error cases ───────────────────────────────────────────────────────────

    #[test]
    fn fast_dimension_mismatch_polar_rejected() {
        let expr = int(1);
        let err = change_coords(
            &expr,
            &["x".to_string()],
            &["r".to_string()],
            CoordSystem::Polar2D,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ChangeCoordError::DimensionMismatch {
                system: "Polar2D",
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn fast_var_count_mismatch_rejected() {
        let expr = int(1);
        let err = change_coords(
            &expr,
            &["x".to_string(), "y".to_string()],
            &["r".to_string()],
            CoordSystem::Polar2D,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ChangeCoordError::VarCountMismatch {
                from_count: 2,
                to_count: 1
            }
        ));
    }
}
