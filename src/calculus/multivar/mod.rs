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
//! | [`divergence`] | ∇·F = Σᵢ ∂Fᵢ/∂xᵢ |
//! | [`curl`] | ∇×F = (∂F₃/∂y − ∂F₂/∂z, …) |
//! | [`laplacian`] | ∇²f = Σᵢ ∂²f/∂xᵢ² |
//! | [`jacobian`] | J[i][j] = ∂fᵢ/∂xⱼ |
//! | [`hessian`] | H[i][j] = ∂²f/∂xᵢ∂xⱼ |
//! | [`directional_derivative`] | Dᵥf = ∇f · v̂ |
//! | [`total_derivative`] | df = Σ (∂f/∂xᵢ) dxᵢ  (chain rule with substitution) |

mod curl;
mod directional;
mod divergence;
mod gradient;
mod hessian;
mod jacobian;
mod laplacian;
mod total;

pub use curl::curl;
pub use directional::directional_derivative;
pub use divergence::divergence;
pub use gradient::{gradient, partial};
pub use hessian::hessian;
pub use jacobian::jacobian;
pub use laplacian::laplacian;
pub use total::total_derivative;
