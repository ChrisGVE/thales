//! Optional domain extensions for Thales.
//! Enable features in Cargo.toml to activate.

#[cfg(feature = "vector-calc")]
pub mod vector_calc;

#[cfg(feature = "number-theory")]
pub mod number_theory;
