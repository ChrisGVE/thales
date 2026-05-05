//! Numerical fallback configuration and precision types.
//!
//! This module provides the infrastructure for D0.4 numerical fallback:
//!
//! - [`FallbackConfig`] — opt-in configuration for numerical evaluation.
//! - [`node_count`] — cheap iterative complexity metric on `Arc<Expr>` trees.
//! - [`PrecisionLevel`] and [`CHAIN`] — ordered precision tiers.
//! - [`NumericalResult`] — a successful numerical evaluation.
//! - [`PrecisionAttemptOutcome`] — outcome of one evaluation attempt.
//! - [`NumericalEvaluator`] — plug-in trait for numerical backends.
//! - [`NumericalEvaluatorRegistry`] / [`global_registry`] — evaluator registry.
//! - [`FallbackRunner`] — drives the precision-escalation loop.

pub mod config;
pub mod evaluator;
pub mod precision;
pub mod registry;
pub mod runner;
pub mod trigger;

#[cfg(test)]
pub mod testutils;

pub use config::{node_count, FallbackConfig};
pub use evaluator::NumericalEvaluator;
pub use precision::{NumericalResult, PrecisionAttemptOutcome, PrecisionLevel, CHAIN};
pub use registry::{global_registry, NumericalEvaluatorRegistry};
pub use runner::FallbackRunner;
pub use trigger::{FallbackTrigger, ImpossibilityClass};
