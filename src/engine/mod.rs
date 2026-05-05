//! D0 search strategy engine.
//!
//! This module contains the foundational types for the D0 search strategy
//! engine: trace trees, resource budgets, assumption sets, property sets,
//! reason types, execution modes, canonical pattern representation,
//! solve context, strategy trait, and assumption normalization/entailment.

pub mod assumption;
pub mod assumption_key;
pub mod cache;
pub mod canonical_pattern;
pub mod context;
pub mod fallback;
pub mod legacy;
pub mod mode;
pub mod narrate;
pub mod property;
pub mod reason;
pub mod resource;
pub mod runner;
pub mod strategy;
pub mod trace_tree;

mod assumption_entailment;

pub use assumption::{AssumptionGuard, AssumptionSet};
pub use assumption_key::{
    entails, normalize_assumption, sign_with_varmap, AssumptionConstraint, AssumptionSignature,
    Domain, NormalizedAssumption,
};
pub use cache::{
    CacheEntry, CacheLookup, CacheSource, CacheStats, KnowledgeCache, NegativeCacheEntry,
    PositiveCacheEntry, Promotable, SolveCache,
};
pub use canonical_pattern::{
    canonicalize, pattern_hash, structural_hash, CanonicalPattern, PatternHash, SlotId, VarMap,
};
pub use context::SolveContext;
pub use fallback::{
    global_registry, node_count, FallbackConfig, FallbackRunner, FallbackTrigger,
    ImpossibilityClass, NumericalEvaluator, NumericalEvaluatorRegistry, NumericalResult,
    PrecisionAttemptOutcome, PrecisionLevel, CHAIN,
};
pub use legacy::{LegacyEngine, LegacyResult};
pub use mode::{ExecutionMode, TreeComparison};
pub use property::{Property, PropertyConstraint, PropertySet};
pub use reason::{FailureReason, ImpossibilityProof, PartialReason, ResourceRequest};
#[cfg(feature = "rayon")]
pub use resource::{BranchGuard, ResourceGate};
pub use resource::{ResourceBudget, ResourceStatus};
#[cfg(feature = "rayon")]
pub use runner::RayonRunner;
pub use runner::{SequentialRunner, StrategyRunner};
pub use strategy::{
    MergerFn, Strategy, StrategyCandidate, StrategyResult, StrategyStatus, SubProblem,
};
pub use trace_tree::{
    BranchHandle, BranchOutcome, BranchReason, JoinHandle, JoinReason, PruneReason, StrategyId,
    TraceBranch, TraceNode, TraceTree,
};
