//! Multi-Equation System Solver
//!
//! This module provides a general-purpose equation system solver that can:
//! - Accept multiple equations of any supported type (algebraic, ODE, integral, differential, etc.)
//! - Accept known values and target variable(s)
//! - Build a dependency graph to determine solving order
//! - Chain solutions through multiple equations
//! - Track all steps in a unified resolution path
//!
//! # Example
//!
//! ```rust
//! use thales::equation_system::{EquationSystem, SystemContext, MultiEquationSolver};
//! use thales::parse_equation;
//!
//! // Create a system of equations
//! let mut system = EquationSystem::new();
//! system.add_equation("eq1", parse_equation("F = m * a").unwrap());
//! system.add_equation("eq2", parse_equation("v = u + a * t").unwrap());
//!
//! // Set up the context with known values and targets
//! let context = SystemContext::new()
//!     .with_known_value("F", 100.0)
//!     .with_known_value("m", 20.0)
//!     .with_known_value("u", 0.0)
//!     .with_known_value("t", 5.0)
//!     .with_target("a")
//!     .with_target("v");
//!
//! // Solve the system
//! let solver = MultiEquationSolver::new();
//! let solution = solver.solve(&system, &context).unwrap();
//!
//! // Get results: a = 5.0, v = 25.0
//! assert!((solution.get_numeric("a").unwrap() - 5.0).abs() < 1e-10);
//! assert!((solution.get_numeric("v").unwrap() - 25.0).abs() < 1e-10);
//! ```

mod multi_solver;
mod nonlinear;
pub mod types;

pub use multi_solver::{MultiEquationSolver, SolverConfig};
pub use nonlinear::{
    broyden_system, fixed_point_system, newton_raphson_system, residual_norm,
    solve_linear_system_lu, validate_jacobian, BroydenSolver, ConvergenceBehavior,
    ConvergenceDiagnostics, FixedPointSolver, NewtonRaphsonSolver, NonlinearSystem,
    NonlinearSystemConfig, NonlinearSystemSolver, NonlinearSystemSolverError,
    NonlinearSystemSolverResult, SmartNonlinearSystemSolver,
};
pub use types::{
    Constraint, DependencyGraph, EquationSystem, EquationType, IntegralInfo, MultiEquationSolution,
    NamedEquation, ODEInfo, SolutionStrategy, SolutionValue, SolveMethod, SolveStep, StepResult,
    SystemContext, SystemError, SystemOperation, SystemResolutionPath, SystemStep,
};
