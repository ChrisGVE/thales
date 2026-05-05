//! ODE system types — systems of first-order ODEs.

use std::collections::HashMap;
use std::sync::Arc;

use crate::ast::Expression;
use crate::numeric::compile::compile;
use crate::numeric::evaluation::evaluate;
use crate::numeric::expr::Expr;
use crate::numeric::SymbolId;
use crate::solver::helpers::contains_symbol;

use super::types::ODEError;

/// A system of first-order ODEs: y'_i = f_i(t, y_1, ..., y_n).
///
/// Equations are stored in canonical `Arc<Expr>` form. Use constructors
/// to build from `Expression` (I/O boundary) or `Arc<Expr>` (engine).
#[derive(Debug, Clone)]
pub struct OdeSystem {
    equations: Vec<Arc<Expr>>,
    /// Names of the unknown functions y_1, …, y_n (e.g. `["y1", "y2"]`).
    pub fn_names: Vec<String>,
    /// Name of the independent variable (e.g. `"t"`).
    pub var: String,
    /// Number of equations (= `fn_names.len()`).
    pub n: usize,
}

impl OdeSystem {
    /// Create from `Expression`-typed RHS expressions.
    ///
    /// Each expression `equations[i]` is the right-hand side of
    /// `fn_names[i]' = f_i(var, fn_names[0], ..., fn_names[n-1])`.
    /// The number of equations must equal the number of function names.
    pub fn new(
        equations: Vec<Expression>,
        fn_names: Vec<String>,
        var: String,
    ) -> Result<Self, ODEError> {
        if equations.len() != fn_names.len() {
            return Err(ODEError::SystemDimensionMismatch(format!(
                "got {} equations but {} function names",
                equations.len(),
                fn_names.len()
            )));
        }
        let n = equations.len();
        let compiled: Vec<Arc<Expr>> = equations.iter().map(compile).collect();
        Ok(Self {
            equations: compiled,
            fn_names,
            var,
            n,
        })
    }

    /// Create from pre-compiled `Arc<Expr>` RHS expressions.
    ///
    /// Prefer this constructor when the caller already operates in canonical
    /// form — it avoids a redundant decompile/compile round-trip.
    pub fn from_arc(
        equations: Vec<Arc<Expr>>,
        fn_names: Vec<String>,
        var: String,
    ) -> Result<Self, ODEError> {
        if equations.len() != fn_names.len() {
            return Err(ODEError::SystemDimensionMismatch(format!(
                "got {} equations but {} function names",
                equations.len(),
                fn_names.len()
            )));
        }
        let n = equations.len();
        Ok(Self {
            equations,
            fn_names,
            var,
            n,
        })
    }

    /// Get the compiled RHS expressions.
    pub fn equations(&self) -> &[Arc<Expr>] {
        &self.equations
    }
}

/// Solution of an ODE system.
#[derive(Debug, Clone)]
pub struct OdeSystemSolution {
    /// One solution expression per unknown function, in `Arc<Expr>` form.
    pub components: Vec<Arc<Expr>>,
    /// Description of the solution method.
    pub method: String,
    /// Narrated solution steps.
    pub steps: Vec<String>,
}

// ── Public solver functions ───────────────────────────────────────────────────

/// Extract the constant-coefficient matrix A from y' = Ay.
///
/// Returns an n×n matrix where A[i][j] is the coefficient of y_j in equation i.
/// Fails if any equation is non-linear or has variable-coefficient terms.
pub fn extract_linear_system_matrix(system: &OdeSystem) -> Result<Vec<Vec<f64>>, ODEError> {
    let n = system.n;
    let sym_ids: Vec<SymbolId> = system
        .fn_names
        .iter()
        .map(|name| SymbolId::intern(name))
        .collect();
    let var_id = SymbolId::intern(&system.var);
    let mut matrix = vec![vec![0.0f64; n]; n];
    for (i, eq) in system.equations().iter().enumerate() {
        let row = extract_coefficients(eq, &sym_ids, var_id, n)?;
        matrix[i] = row;
    }
    Ok(matrix)
}

/// Solve a linear constant-coefficient ODE system symbolically (2×2 only).
///
/// Delegates to the eigenvalue solver in [`super::linear_system`].
pub fn solve_linear_system(system: &OdeSystem) -> Result<OdeSystemSolution, ODEError> {
    super::linear_system::solve_linear_system(system)
}

/// Solve an ODE system numerically using RK4.
///
/// Evaluates each equation expression at each step, binding the independent
/// variable and all state variables from `fn_names`.
pub fn solve_system_numeric(
    system: &OdeSystem,
    t0: f64,
    y0: Vec<f64>,
    t_end: f64,
    steps: usize,
) -> Result<crate::runge_kutta::Rk4SystemSolution, ODEError> {
    if y0.len() != system.n {
        return Err(ODEError::SystemDimensionMismatch(format!(
            "y0 length {} ≠ system dimension {}",
            y0.len(),
            system.n
        )));
    }
    let var_id = SymbolId::intern(&system.var);
    let sym_ids: Vec<SymbolId> = system
        .fn_names
        .iter()
        .map(|name| SymbolId::intern(name))
        .collect();
    let equations: Vec<Arc<Expr>> = system.equations().to_vec();
    let n = system.n;
    let system_fn = move |t: f64, y: &[f64]| -> Vec<f64> {
        let mut bindings = HashMap::with_capacity(n + 1);
        bindings.insert(var_id, t);
        for (i, &yi) in y.iter().enumerate() {
            bindings.insert(sym_ids[i], yi);
        }
        equations
            .iter()
            .map(|eq| evaluate(eq, &bindings).unwrap_or(0.0))
            .collect()
    };
    crate::runge_kutta::rk4_system_solve(system_fn, t0, y0, t_end, steps)
}

// ── Coefficient extraction helpers ───────────────────────────────────────────

/// Extract the coefficient of each state variable from a linear expression.
fn extract_coefficients(
    expr: &Arc<Expr>,
    sym_ids: &[SymbolId],
    var_id: SymbolId,
    n: usize,
) -> Result<Vec<f64>, ODEError> {
    let mut coeffs = vec![0.0f64; n];
    accumulate_coefficients(expr, sym_ids, var_id, n, 1.0, &mut coeffs)?;
    Ok(coeffs)
}

/// Accumulate coefficients from `expr` scaled by `scale` into `coeffs`.
///
/// Rejects non-linear terms, independent-variable appearances, and
/// constant terms (homogeneous systems only).
fn accumulate_coefficients(
    expr: &Arc<Expr>,
    sym_ids: &[SymbolId],
    var_id: SymbolId,
    n: usize,
    scale: f64,
    coeffs: &mut Vec<f64>,
) -> Result<(), ODEError> {
    match expr.as_ref() {
        Expr::Symbol(s) => {
            if *s == var_id {
                return Err(ODEError::NotLinearConstantCoefficient(format!(
                    "independent variable '{s}' appears in RHS"
                )));
            }
            if let Some(j) = sym_ids.iter().position(|id| id == s) {
                coeffs[j] += scale;
            } else {
                return Err(ODEError::NotLinearConstantCoefficient(format!(
                    "unknown symbol '{s}' in linear system"
                )));
            }
        }
        Expr::Integer(_) | Expr::Rational(_) | Expr::Float(_) => {
            let bindings: HashMap<SymbolId, f64> = HashMap::new();
            let v = evaluate(expr, &bindings).unwrap_or(0.0);
            if v.abs() > 1e-14 {
                return Err(ODEError::NotLinearConstantCoefficient(
                    "constant term in RHS — not a homogeneous system".into(),
                ));
            }
        }
        Expr::Add(node) => {
            if node.constant.to_f64().abs() > 1e-14 {
                return Err(ODEError::NotLinearConstantCoefficient(
                    "constant term in RHS — not a homogeneous system".into(),
                ));
            }
            for (term, coeff) in &node.terms {
                let term_scale = scale * coeff.to_f64();
                accumulate_coefficients(term, sym_ids, var_id, n, term_scale, coeffs)?;
            }
        }
        Expr::Mul(node) => {
            accumulate_mul_coefficients(node, expr, sym_ids, var_id, scale, coeffs)?;
        }
        _ => {
            if sym_ids.iter().any(|id| contains_symbol(expr, *id)) {
                return Err(ODEError::NotLinearConstantCoefficient(
                    "non-linear term involving state variable".into(),
                ));
            }
        }
    }
    Ok(())
}

/// Handle the `Mul` case of [`accumulate_coefficients`].
fn accumulate_mul_coefficients(
    node: &crate::numeric::MulNode,
    expr: &Arc<Expr>,
    sym_ids: &[SymbolId],
    var_id: SymbolId,
    scale: f64,
    coeffs: &mut Vec<f64>,
) -> Result<(), ODEError> {
    let num_scale = scale * node.coeff.to_f64();
    let mut state_idx: Option<usize> = None;
    for (base, exp) in &node.factors {
        if let Expr::Symbol(s) = base.as_ref() {
            if *s == var_id {
                return Err(ODEError::NotLinearConstantCoefficient(
                    "independent variable in product term".into(),
                ));
            }
            if let Some(j) = sym_ids.iter().position(|id| id == s) {
                let exp_is_one = matches!(exp.as_ref(), Expr::Integer(n) if n.to_i64() == Some(1));
                if !exp_is_one {
                    return Err(ODEError::NotLinearConstantCoefficient(
                        "state variable raised to power ≠ 1".into(),
                    ));
                }
                if state_idx.is_some() {
                    return Err(ODEError::NotLinearConstantCoefficient(
                        "product of two state variables".into(),
                    ));
                }
                state_idx = Some(j);
                continue;
            }
        }
        if sym_ids.iter().any(|id| contains_symbol(base, *id)) || contains_symbol(exp, var_id) {
            return Err(ODEError::NotLinearConstantCoefficient(
                "non-constant factor in coefficient".into(),
            ));
        }
    }
    match state_idx {
        Some(j) => coeffs[j] += num_scale,
        None => {
            // Check that the whole expr is numerically zero before rejecting.
            let bindings: HashMap<SymbolId, f64> = HashMap::new();
            let v = evaluate(expr, &bindings).unwrap_or(0.0);
            if v.abs() > 1e-14 {
                return Err(ODEError::NotLinearConstantCoefficient(
                    "constant term in RHS".into(),
                ));
            }
        }
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Expression, Variable};

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    #[test]
    fn test_ode_system_new_valid() {
        let eqs = vec![var("y2"), var("y1")];
        let names = vec!["y1".into(), "y2".into()];
        let sys = OdeSystem::new(eqs, names, "t".into());
        assert!(sys.is_ok());
        let sys = sys.unwrap();
        assert_eq!(sys.n, 2);
        assert_eq!(sys.var, "t");
        assert_eq!(sys.fn_names, vec!["y1", "y2"]);
        assert_eq!(sys.equations().len(), 2);
    }

    #[test]
    fn test_ode_system_dimension_mismatch() {
        let eqs = vec![var("y2")];
        let names = vec!["y1".into(), "y2".into()];
        let sys = OdeSystem::new(eqs, names, "t".into());
        assert!(matches!(sys, Err(ODEError::SystemDimensionMismatch(_))));
    }

    #[test]
    fn test_ode_system_from_arc_valid() {
        use crate::numeric::compile::compile;
        let eqs = vec![compile(&var("y2")), compile(&var("y1"))];
        let names = vec!["y1".into(), "y2".into()];
        let sys = OdeSystem::from_arc(eqs, names, "t".into());
        assert!(sys.is_ok());
        let sys = sys.unwrap();
        assert_eq!(sys.n, 2);
    }

    #[test]
    fn test_ode_system_from_arc_dimension_mismatch() {
        use crate::numeric::compile::compile;
        let eqs = vec![compile(&var("y2"))];
        let names = vec!["y1".into(), "y2".into()];
        let sys = OdeSystem::from_arc(eqs, names, "t".into());
        assert!(matches!(sys, Err(ODEError::SystemDimensionMismatch(_))));
    }

    #[test]
    fn test_ode_system_solution_fields() {
        use crate::numeric::compile::compile;
        let sol = OdeSystemSolution {
            components: vec![compile(&var("C1")), compile(&var("C2"))],
            method: "eigenvalue".into(),
            steps: vec!["step 1".into(), "step 2".into()],
        };
        assert_eq!(sol.components.len(), 2);
        assert_eq!(sol.method, "eigenvalue");
        assert_eq!(sol.steps.len(), 2);
    }

    #[test]
    fn test_extract_linear_system_matrix_identity() {
        // y1' = y1, y2' = y2 → A = [[1,0],[0,1]]
        let eqs = vec![var("y1"), var("y2")];
        let names = vec!["y1".into(), "y2".into()];
        let sys = OdeSystem::new(eqs, names, "t".into()).unwrap();
        let mat = extract_linear_system_matrix(&sys).unwrap();
        assert!((mat[0][0] - 1.0).abs() < 1e-14);
        assert!(mat[0][1].abs() < 1e-14);
        assert!(mat[1][0].abs() < 1e-14);
        assert!((mat[1][1] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_extract_linear_system_matrix_coupling() {
        // y1' = y2, y2' = y1 → A = [[0,1],[1,0]]
        let eqs = vec![var("y2"), var("y1")];
        let names = vec!["y1".into(), "y2".into()];
        let sys = OdeSystem::new(eqs, names, "t".into()).unwrap();
        let mat = extract_linear_system_matrix(&sys).unwrap();
        assert!(mat[0][0].abs() < 1e-14);
        assert!((mat[0][1] - 1.0).abs() < 1e-14);
        assert!((mat[1][0] - 1.0).abs() < 1e-14);
        assert!(mat[1][1].abs() < 1e-14);
    }

    #[test]
    fn test_solve_linear_system_harmonic() {
        // y1' = y2, y2' = -y1 → eigenvalues ±i (pure imaginary)
        let eqs = vec![var("y2"), {
            use crate::ast::{BinaryOp, Expression};
            Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Float(-1.0)),
                Box::new(var("y1")),
            )
        }];
        let names = vec!["y1".into(), "y2".into()];
        let sys = OdeSystem::new(eqs, names, "t".into()).unwrap();
        let sol = solve_linear_system(&sys).unwrap();
        assert_eq!(sol.components.len(), 2);
        assert!(sol.method.contains("eigenvalue"));
        assert!(!sol.steps.is_empty());
    }

    #[test]
    fn test_solve_system_numeric_harmonic() {
        // y1' = y2, y2' = -y1; y1(0)=1, y2(0)=0 → y1(π) ≈ -1
        let eqs = vec![var("y2"), {
            use crate::ast::{BinaryOp, Expression};
            Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Float(-1.0)),
                Box::new(var("y1")),
            )
        }];
        let names = vec!["y1".into(), "y2".into()];
        let sys = OdeSystem::new(eqs, names, "t".into()).unwrap();
        let sol =
            solve_system_numeric(&sys, 0.0, vec![1.0, 0.0], std::f64::consts::PI, 10_000).unwrap();
        assert!((sol.y_final[0] - (-1.0)).abs() < 1e-5, "y1(π) ≈ -1");
        assert!(sol.y_final[1].abs() < 1e-5, "y2(π) ≈ 0");
    }

    #[test]
    fn test_solve_system_numeric_dimension_mismatch() {
        let eqs = vec![var("y1")];
        let names = vec!["y1".into()];
        let sys = OdeSystem::new(eqs, names, "t".into()).unwrap();
        let result = solve_system_numeric(&sys, 0.0, vec![1.0, 2.0], 1.0, 10);
        assert!(matches!(result, Err(ODEError::SystemDimensionMismatch(_))));
    }
}
