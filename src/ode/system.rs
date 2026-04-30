//! ODE system types — systems of first-order ODEs.

use std::sync::Arc;

use crate::ast::Expression;
use crate::numeric::compile::compile;
use crate::numeric::Expr;

use super::types::ODEError;

/// A system of first-order ODEs: y'_i = f_i(t, y_1, ..., y_n).
///
/// Equations are stored in canonical `Arc<Expr>` form. Use constructors
/// to build from `Expression` (I/O boundary) or `Arc<Expr>` (engine).
#[derive(Debug, Clone)]
pub struct OdeSystem {
    equations: Vec<Arc<Expr>>,
    pub fn_names: Vec<String>,
    pub var: String,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Expression, Variable};

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    #[test]
    fn test_ode_system_new_valid() {
        // y1' = y2, y2' = y1
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
}
