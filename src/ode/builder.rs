//! Fluent builder API for constructing ODE problems programmatically.
//!
//! This module provides [`ODEBuilder`] and convenience functions for constructing
//! first-order and second-order ODEs without manually populating struct fields.
//!
//! # Examples
//!
//! ```rust
//! use thales::ode::builder::{ODEBuilder, first_order_ode, second_order_homogeneous};
//! use thales::ast::{BinaryOp, Expression, Variable};
//!
//! // Build dy/dx = x*y via the fluent builder
//! let x = Expression::Variable(Variable::new("x"));
//! let y = Expression::Variable(Variable::new("y"));
//! let rhs = Expression::Binary(BinaryOp::Mul, Box::new(x), Box::new(y));
//!
//! let ode = ODEBuilder::new()
//!     .dependent("y")
//!     .independent("x")
//!     .build_first_order(rhs)
//!     .expect("valid ODE");
//!
//! assert_eq!(ode.dependent, "y");
//! assert_eq!(ode.independent, "x");
//! ```

use crate::ast::Expression;

use super::types::{FirstOrderODE, ODEError};
use super::SecondOrderODE;

// ─── Builder ────────────────────────────────────────────────────────────────

/// Fluent builder for constructing ODE problems.
///
/// Call [`ODEBuilder::new`], chain `.dependent()` and `.independent()` to
/// specify variable names, then finalise with one of the `build_*` methods.
///
/// # Errors
///
/// All `build_*` methods return [`ODEError::NotInExpectedForm`] when a required
/// variable name has not been set.
#[derive(Debug, Default, Clone)]
pub struct ODEBuilder {
    dependent: Option<String>,
    independent: Option<String>,
}

impl ODEBuilder {
    /// Create a new builder with no variables set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the name of the dependent variable (e.g. `"y"`).
    #[must_use]
    pub fn dependent(mut self, var: &str) -> Self {
        self.dependent = Some(var.to_string());
        self
    }

    /// Set the name of the independent variable (e.g. `"x"`).
    #[must_use]
    pub fn independent(mut self, var: &str) -> Self {
        self.independent = Some(var.to_string());
        self
    }

    /// Consume the builder and produce a [`FirstOrderODE`]: `dy/dx = rhs`.
    ///
    /// # Errors
    ///
    /// Returns [`ODEError::NotInExpectedForm`] if the dependent or independent
    /// variable name is missing.
    pub fn build_first_order(self, rhs: Expression) -> Result<FirstOrderODE, ODEError> {
        let (dep, ind) = self.require_vars()?;
        Ok(FirstOrderODE::new(&dep, &ind, rhs))
    }

    /// Consume the builder and produce a homogeneous [`SecondOrderODE`]:
    /// `a·y'' + b·y' + c·y = 0`.
    ///
    /// # Errors
    ///
    /// Returns [`ODEError::NotInExpectedForm`] if the dependent or independent
    /// variable name is missing.
    pub fn build_second_order_homogeneous(
        self,
        a: f64,
        b: f64,
        c: f64,
    ) -> Result<SecondOrderODE, ODEError> {
        let (dep, ind) = self.require_vars()?;
        Ok(SecondOrderODE::homogeneous(&dep, &ind, a, b, c))
    }

    /// Consume the builder and produce a non-homogeneous [`SecondOrderODE`]:
    /// `a·y'' + b·y' + c·y = forcing`.
    ///
    /// # Errors
    ///
    /// Returns [`ODEError::NotInExpectedForm`] if the dependent or independent
    /// variable name is missing.
    pub fn build_second_order(
        self,
        a: f64,
        b: f64,
        c: f64,
        forcing: Expression,
    ) -> Result<SecondOrderODE, ODEError> {
        let (dep, ind) = self.require_vars()?;
        Ok(SecondOrderODE::new(&dep, &ind, a, b, c, forcing))
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn require_vars(self) -> Result<(String, String), ODEError> {
        let dep = self
            .dependent
            .ok_or_else(|| ODEError::NotInExpectedForm("dependent variable not set".to_string()))?;
        let ind = self.independent.ok_or_else(|| {
            ODEError::NotInExpectedForm("independent variable not set".to_string())
        })?;
        Ok((dep, ind))
    }
}

// ─── Convenience functions ───────────────────────────────────────────────────

/// Build a [`FirstOrderODE`] (`dy/dx = rhs`) in a single call.
///
/// This is a short-hand for constructing [`ODEBuilder`] with both variables
/// set and calling [`ODEBuilder::build_first_order`].
///
/// # Examples
///
/// ```rust
/// use thales::ode::builder::first_order_ode;
/// use thales::ast::{Expression, Variable};
///
/// let rhs = Expression::Variable(Variable::new("y"));
/// let ode = first_order_ode("y", "x", rhs);
/// assert_eq!(ode.dependent, "y");
/// assert_eq!(ode.independent, "x");
/// ```
#[must_use]
pub fn first_order_ode(y: &str, x: &str, rhs: Expression) -> FirstOrderODE {
    FirstOrderODE::new(y, x, rhs)
}

/// Build a homogeneous [`SecondOrderODE`] (`a·y'' + b·y' + c·y = 0`) in a
/// single call.
///
/// This is a short-hand for constructing [`ODEBuilder`] with both variables
/// set and calling [`ODEBuilder::build_second_order_homogeneous`].
///
/// # Examples
///
/// ```rust
/// use thales::ode::builder::second_order_homogeneous;
///
/// // y'' - y = 0
/// let ode = second_order_homogeneous("y", "x", 1.0, 0.0, -1.0);
/// assert!((ode.a - 1.0).abs() < f64::EPSILON);
/// assert!((ode.c - (-1.0)).abs() < f64::EPSILON);
/// ```
#[must_use]
pub fn second_order_homogeneous(y: &str, x: &str, a: f64, b: f64, c: f64) -> SecondOrderODE {
    SecondOrderODE::homogeneous(y, x, a, b, c)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Expression, Variable};
    use crate::ode::{solve_second_order_homogeneous, solve_separable, RootType};

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn mul(l: Expression, r: Expression) -> Expression {
        Expression::Binary(BinaryOp::Mul, Box::new(l), Box::new(r))
    }

    // ── ODEBuilder ────────────────────────────────────────────────────────────

    #[test]
    fn test_builder_first_order_fields() {
        // dy/dx = x*y
        let rhs = mul(var("x"), var("y"));
        let ode = ODEBuilder::new()
            .dependent("y")
            .independent("x")
            .build_first_order(rhs.clone())
            .expect("build should succeed");

        assert_eq!(ode.dependent, "y");
        assert_eq!(ode.independent, "x");
        // rhs should mirror what we passed in
        assert!(matches!(ode.rhs, Expression::Binary(BinaryOp::Mul, _, _)));
    }

    #[test]
    fn test_builder_second_order_homogeneous_coefficients() {
        // y'' - y = 0  =>  a=1, b=0, c=-1
        let ode = ODEBuilder::new()
            .dependent("y")
            .independent("x")
            .build_second_order_homogeneous(1.0, 0.0, -1.0)
            .expect("build should succeed");

        assert_eq!(ode.dependent, "y");
        assert_eq!(ode.independent, "x");
        assert!((ode.a - 1.0).abs() < f64::EPSILON);
        assert!((ode.b - 0.0).abs() < f64::EPSILON);
        assert!((ode.c - (-1.0)).abs() < f64::EPSILON);
        assert!(ode.is_homogeneous());
    }

    #[test]
    fn test_builder_second_order_nonhomogeneous() {
        let forcing = var("x");
        let ode = ODEBuilder::new()
            .dependent("y")
            .independent("x")
            .build_second_order(1.0, 0.0, -1.0, forcing)
            .expect("build should succeed");

        assert!(!ode.is_homogeneous());
    }

    #[test]
    fn test_builder_error_missing_dependent() {
        let rhs = var("y");
        let result = ODEBuilder::new().independent("x").build_first_order(rhs);

        assert!(matches!(result, Err(ODEError::NotInExpectedForm(_))));
    }

    #[test]
    fn test_builder_error_missing_independent() {
        let rhs = var("y");
        let result = ODEBuilder::new().dependent("y").build_first_order(rhs);

        assert!(matches!(result, Err(ODEError::NotInExpectedForm(_))));
    }

    #[test]
    fn test_builder_error_both_missing() {
        let result = ODEBuilder::new().build_second_order_homogeneous(1.0, 0.0, -1.0);

        assert!(matches!(result, Err(ODEError::NotInExpectedForm(_))));
    }

    // ── Convenience functions ─────────────────────────────────────────────────

    #[test]
    fn test_convenience_first_order_matches_builder() {
        let rhs = mul(var("x"), var("y"));

        let via_builder = ODEBuilder::new()
            .dependent("y")
            .independent("x")
            .build_first_order(rhs.clone())
            .expect("build");

        let via_fn = first_order_ode("y", "x", rhs);

        assert_eq!(via_builder.dependent, via_fn.dependent);
        assert_eq!(via_builder.independent, via_fn.independent);
    }

    #[test]
    fn test_convenience_second_order_matches_builder() {
        let via_builder = ODEBuilder::new()
            .dependent("y")
            .independent("x")
            .build_second_order_homogeneous(2.0, -3.0, 1.0)
            .expect("build");

        let via_fn = second_order_homogeneous("y", "x", 2.0, -3.0, 1.0);

        assert_eq!(via_builder.dependent, via_fn.dependent);
        assert_eq!(via_builder.independent, via_fn.independent);
        assert!((via_builder.a - via_fn.a).abs() < f64::EPSILON);
        assert!((via_builder.b - via_fn.b).abs() < f64::EPSILON);
        assert!((via_builder.c - via_fn.c).abs() < f64::EPSILON);
    }

    // ── Integration test: builder → solve ─────────────────────────────────────

    #[test]
    fn test_builder_to_solve_separable() {
        // dy/dx = x*y  →  separable, solution y = A·e^(x²/2)
        let rhs = mul(var("x"), var("y"));
        let ode = ODEBuilder::new()
            .dependent("y")
            .independent("x")
            .build_first_order(rhs)
            .expect("build");

        let solution = solve_separable(&ode).expect("should solve");
        assert_eq!(solution.method, "Separation of variables");
        assert!(!solution.steps.is_empty());
    }

    #[test]
    fn test_builder_to_solve_second_order_homogeneous() {
        // y'' - y = 0  →  distinct real roots ±1
        let ode = ODEBuilder::new()
            .dependent("y")
            .independent("x")
            .build_second_order_homogeneous(1.0, 0.0, -1.0)
            .expect("build");

        let solution = solve_second_order_homogeneous(&ode).expect("should solve");
        assert_eq!(solution.roots.root_type, RootType::TwoDistinctReal);
        assert!(!solution.steps.is_empty());
    }
}
