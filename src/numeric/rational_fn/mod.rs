//! Rational function type: numerator/denominator polynomial fraction.
//!
//! [`RationalFunction<R>`] represents `p(x)/q(x)` where `p` and `q` are
//! dense univariate polynomials over a field `R`. Automatically reduces
//! to lowest terms via GCD on construction.

pub(super) mod advanced;
pub(super) mod core;

pub use core::{PartialFractionTerm, RationalFunction};

#[cfg(test)]
mod tests_advanced;
#[cfg(test)]
mod tests_core;
