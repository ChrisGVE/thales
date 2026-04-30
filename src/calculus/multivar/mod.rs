//! Multivariate differential calculus on [`Arc<Expr>`] trees.
//!
//! All functions in this module delegate to the single-variable
//! [`diff_arc`][crate::numeric::differentiation::diff_arc] engine and compose
//! its results to form higher-level multivariate objects.  No symbolic logic
//! is duplicated: every differentiation step goes through the same chain-rule
//! and constant-folding machinery that powers the univariate case.
//!
//! # Functions
//!
//! | Function | Math object |
//! |---|---|
//! | [`partial`] | ∂f/∂xᵢ |
//! | [`gradient`] | ∇f = (∂f/∂x₁, …, ∂f/∂xₙ) |
//! | [`jacobian`] | J[i][j] = ∂fᵢ/∂xⱼ |
//! | [`hessian`] | H[i][j] = ∂²f/∂xᵢ∂xⱼ |
//! | [`directional_derivative`] | Dᵥf = ∇f · v̂ |
//! | [`total_derivative`] | df = Σ (∂f/∂xᵢ) dxᵢ  (chain rule with substitution) |

mod directional;
mod gradient;
mod hessian;
mod jacobian;
mod total;

pub use directional::directional_derivative;
pub use gradient::{gradient, partial};
pub use hessian::hessian;
pub use jacobian::jacobian;
pub use total::total_derivative;
