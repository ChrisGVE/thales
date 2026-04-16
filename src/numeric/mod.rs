//! Exact numeric types for computer algebra.
//!
//! Provides overflow-safe integer and rational types used throughout
//! the expression tree and polynomial arithmetic.
//!
//! - [`SmallInt`] — tagged union: inline `i64` or heap-allocated `BigInt`
//! - [`BigRational`] — exact rational with `SmallInt` components
//! - [`SymbolId`] — interned variable name (4-byte `Copy` handle)
//! - [`Expr`] — Arc-based expression with structural sharing
//! - [`AddNode`] — canonical n-ary sum
//! - [`MulNode`] — canonical n-ary product
//! - [`ExprPool`] — hash-consing pool for common sub-expression elimination

mod big_rational;
pub mod compile;
pub mod expr;
pub mod ring;
mod small_int;
mod symbol;

mod add_node;
pub mod algebraic_rules;
mod compute_ctx;
mod dense_poly;
pub mod differentiation;
pub mod eigenvalue;
pub mod evaluation;
mod f4;
mod finite_field_factor;
mod groebner;
mod hensel;
mod hermite;
pub mod inequality_solver;
pub mod limits;
pub mod log_exp_rules;
mod matrix;
mod modular_gcd;
mod mul_node;
mod multivariate_gcd;
mod multivariate_poly;
pub mod normalize;
mod number_theory;
pub mod pattern_integrate;
mod poly_equation_solver;
mod poly_factoring;
mod poly_ops;
mod primality;
mod rational_equation_solver;
mod rational_fn;
pub mod rewrite;
pub mod risch;
mod rothstein_trager;
pub mod series;
pub mod simplify;
mod solution_set;
mod sparse_poly;
pub mod system_solver;
mod term_order;
pub mod trig_rules;
mod zassenhaus;

pub use add_node::AddNode;
pub use big_rational::BigRational;
pub use compute_ctx::{CancelHandle, ComputeContext, ComputeError, ComputeResult, FeatureFlags};
pub use dense_poly::DensePolynomial;
pub use differentiation::{diff, implicit_diff};
pub use eigenvalue::{
    characteristic_polynomial, eigenvalues, EigenError, EigenvalueResult, ExprMatrix,
};
pub use expr::{Expr, ExprPool, FuncId};
pub use f4::f4;
pub use finite_field_factor::factor_over_gfp;
pub use groebner::buchberger;
pub use hensel::{hensel_lift, HenselLift};
pub use hermite::{hermite_reduce, HermiteReduction};
pub use inequality_solver::{solve_inequality, InequalityType};
pub use limits::{limit, LimitPoint, LimitResult};
pub use matrix::{Matrix, MatrixError};
pub use modular_gcd::modular_gcd;
pub use mul_node::MulNode;
pub use multivariate_gcd::multivariate_gcd;
pub use multivariate_poly::{Monomial, MultivariatePolynomial};
pub use number_theory::{crt, ext_gcd, mod_inverse, mod_pow, ExtGcdResult};
pub use pattern_integrate::pattern_integrate;
pub use poly_equation_solver::{roots_with_multiplicity, solve_polynomial, RootWithMultiplicity};
pub use poly_factoring::SqfFactor;
pub use primality::{factor, is_prime, PrimeFactor};
pub use rational_equation_solver::{solve_rational, solve_rational_equation};
pub use rational_fn::RationalFunction;
pub use risch::{risch_integrate, IntegrationResult};
pub use rothstein_trager::{
    integrate_rational_log, rothstein_trager, rothstein_trager_from_hermite, LogIntegral, LogTerm,
};
pub use series::{taylor, TaylorSeries};
pub use simplify::simplify;
pub use small_int::SmallInt;
pub use solution_set::{Constraint, IntervalBound, SolutionSet};
pub use sparse_poly::SparsePolynomial;
pub use symbol::SymbolId;
pub use system_solver::{expr_to_multipoly, solve_system, solve_system_expr, SolutionPoint};
pub use term_order::{DegLex, GrevLex, Lex, MonomialOrder, OrderedMonomial};
pub use zassenhaus::factor_integer_poly;
