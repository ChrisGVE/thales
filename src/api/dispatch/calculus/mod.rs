//! Calculus command dispatchers (Diff, PartialDiff, Gradient, TotalDiff,
//! Divergence, Curl, Laplacian, Jacobian, Hessian, DirectionalDiff,
//! Integrate, DefIntegrate).

mod diff;
mod gradient;
mod integrate;
mod vector_ops;

pub(super) use diff::{diff_cmd, partial_diff_cmd};
pub(super) use gradient::{gradient_cmd, total_diff_cmd};
pub(super) use integrate::{def_integrate_cmd, integrate_cmd, multi_integrate_cmd};
pub(super) use vector_ops::{
    curl_cmd, directional_diff_cmd, divergence_cmd, hessian_cmd, jacobian_cmd, laplacian_cmd,
};
