//! Numerical fallback configuration and precision types.
//!
//! This module provides the infrastructure for D0.4 numerical fallback:
//!
//! - [`FallbackConfig`] — opt-in configuration for numerical evaluation.
//! - [`node_count`] — cheap iterative complexity metric on `Arc<Expr>` trees.
//! - [`PrecisionLevel`] and [`CHAIN`] — ordered precision tiers.
//! - [`NumericalResult`] — a successful numerical evaluation.
//! - [`PrecisionAttemptOutcome`] — outcome of one evaluation attempt.

pub mod config;
pub mod precision;
pub mod trigger;

pub use config::{node_count, FallbackConfig};
pub use precision::{NumericalResult, PrecisionAttemptOutcome, PrecisionLevel, CHAIN};
pub use trigger::{FallbackTrigger, ImpossibilityClass};
