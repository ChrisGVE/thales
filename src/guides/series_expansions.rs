//! # Series Expansions Guide
//!
//! Series-expansion engines live in [`crate::numeric::series`]. The engines
//! operate on `Arc<Expr>`; callers that start from the public [`Expression`]
//! type compile at the boundary and decompile the result.
//!
//! [`Expression`]: crate::ast::Expression
//!
//! ## Engines
//!
//! | Engine | Function | Return type |
//! | ------ | -------- | ----------- |
//! | Taylor / Maclaurin | [`numeric::series::taylor`] | [`numeric::series::TaylorSeries`] |
//! | Laurent | [`numeric::series::laurent_expand`] | [`numeric::series::LaurentSeries`] |
//! | Asymptotic (`x→±∞`, `x→0`) | [`numeric::series::asymptotic`] | [`numeric::series::AsymptoticSeries`] |
//! | Composition `g(f(x))` | [`numeric::series::compose`] | `TaylorSeries` |
//! | Lagrange reversion `f⁻¹` | [`numeric::series::revert`] | `TaylorSeries` |
//! | Singularity / residue | [`numeric::series::residue`], [`numeric::series::classify_singularity`], [`numeric::series::find_singularities`] | — |
//! | Known standard series | [`numeric::series::sin_series`], [`numeric::series::cos_series`], [`numeric::series::exp_series`], [`numeric::series::ln_series`], [`numeric::series::atan_series`] | `TaylorSeries` |
//!
//! [`numeric::series`]: crate::numeric::series
//! [`numeric::series::taylor`]: crate::numeric::series::taylor
//! [`numeric::series::laurent_expand`]: crate::numeric::series::laurent_expand
//! [`numeric::series::asymptotic`]: crate::numeric::series::asymptotic
//! [`numeric::series::compose`]: crate::numeric::series::compose
//! [`numeric::series::revert`]: crate::numeric::series::revert
//! [`numeric::series::residue`]: crate::numeric::series::residue
//! [`numeric::series::classify_singularity`]: crate::numeric::series::classify_singularity
//! [`numeric::series::find_singularities`]: crate::numeric::series::find_singularities
//! [`numeric::series::TaylorSeries`]: crate::numeric::series::TaylorSeries
//! [`numeric::series::LaurentSeries`]: crate::numeric::series::LaurentSeries
//! [`numeric::series::AsymptoticSeries`]: crate::numeric::series::AsymptoticSeries
//! [`numeric::series::sin_series`]: crate::numeric::series::sin_series
//! [`numeric::series::cos_series`]: crate::numeric::series::cos_series
//! [`numeric::series::exp_series`]: crate::numeric::series::exp_series
//! [`numeric::series::ln_series`]: crate::numeric::series::ln_series
//! [`numeric::series::atan_series`]: crate::numeric::series::atan_series
//!
//! ## Calling pattern
//!
//! All engines share the same shape — supply an `Arc<Expr>`, a [`SymbolId`]
//! for the expansion variable, and engine-specific parameters (center,
//! order, direction). When starting from an `Expression`, compile at the
//! entry and decompile for display:
//!
//! [`SymbolId`]: crate::numeric::SymbolId
//!
//! ```rust,ignore
//! use thales::numeric::compile::{compile, decompile};
//! use thales::numeric::expr::Expr;
//! use thales::numeric::series::taylor;
//! use thales::numeric::SymbolId;
//! use thales::parser::parse_expression;
//!
//! let parsed = parse_expression("exp(x)").unwrap();
//! let arc_expr = compile(&parsed);
//! let var_id = SymbolId::intern("x");
//! let ts = taylor(&arc_expr, var_id, &Expr::int(0), 5);
//! let series_expr = decompile(&ts.to_expr());
//! println!("{}", series_expr);
//! ```
//!
//! ## Narrated expansion
//!
//! All engines accept an optional `&mut Trace` sink. When supplied, each
//! decision point emits a [`numeric::trace::Step`] tagged with the technique
//! applied. Callers that only want the computed result pass `None` and
//! pay no allocation cost.
//!
//! [`numeric::trace::Step`]: crate::numeric::trace::Step
//!
//! ## Related modules
//!
//! - [`crate::numeric::limits`] — uses series for L'Hôpital / asymptotic limits.
//! - [`crate::numeric::differentiation`] — symbolic derivatives underpinning Taylor.
//! - [`crate::numeric::pattern_integrate`] — pattern-based antiderivatives.
