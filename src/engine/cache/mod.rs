//! D0 memoization cache layer.
//!
//! Two cache tiers:
//!
//! - [`SolveCache`] — per-solve transient cache; entries optionally promoted
//!   to [`KnowledgeCache`] at solve completion.
//! - [`KnowledgeCache`] — persistent cross-solve cache; shared across all
//!   invocations of the engine.
//!
//! Entry types ([`PositiveCacheEntry`], [`NegativeCacheEntry`], [`CacheEntry`])
//! and the source discriminant ([`CacheSource`]) live in [`entry`].
//! Statistics ([`CacheStats`]) live in [`stats`].
//! Lookup results ([`CacheLookup`]) are re-exported from [`knowledge`].

pub mod entry;
pub mod knowledge;
pub mod solve;
pub mod stats;

pub use entry::{CacheEntry, CacheSource, NegativeCacheEntry, PositiveCacheEntry};
pub use knowledge::{CacheLookup, KnowledgeCache};
pub use solve::{Promotable, SolveCache};
pub use stats::CacheStats;
