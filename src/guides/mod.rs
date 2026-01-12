//! # User Guides
//!
//! Tutorial modules for common workflows in the thales library.
//!
//! These guides provide step-by-step instructions for accomplishing
//! common tasks, from basic equation solving to advanced calculus operations.
//!
//! ## Available Guides
//!
//! | Guide | Description |
//! |-------|-------------|
//! | [`solving_equations`] | Linear, quadratic, polynomial, and system solving |
//! | [`calculus_operations`] | Derivatives, integrals, limits, and ODEs |
//! | [`series_expansions`] | Taylor, Maclaurin, Laurent, and asymptotic series |
//! | [`coordinate_systems`] | 2D/3D transformations and complex numbers |
//! | [`numerical_methods`] | Root-finding when symbolic methods fail |
//! | [`working_with_units`] | Dimensional analysis and unit conversion |
//! | [`error_handling`] | Working with `ThalesError` and module errors |
//!
//! ## Quick Navigation
//!
//! **New to thales?** Start with [`solving_equations`] for the basics.
//!
//! **Working with calculus?** See [`calculus_operations`] for derivatives,
//! integrals, and differential equations.
//!
//! **Need numerical solutions?** Check [`numerical_methods`] for when
//! symbolic solving isn't possible.

pub mod solving_equations;
pub mod calculus_operations;
pub mod series_expansions;
pub mod coordinate_systems;
pub mod numerical_methods;
pub mod working_with_units;
pub mod error_handling;
