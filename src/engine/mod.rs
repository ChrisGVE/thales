//! D0 search strategy engine.
//!
//! This module contains the foundational types for the D0 search strategy
//! engine: trace trees, resource budgets, assumption sets, property sets,
//! reason types, execution modes, canonical pattern representation,
//! solve context, and strategy trait + result types.

pub mod assumption;
pub mod canonical_pattern;
pub mod context;
pub mod mode;
pub mod property;
pub mod reason;
pub mod resource;
pub mod strategy;
pub mod trace_tree;

pub use assumption::{AssumptionGuard, AssumptionSet};
pub use canonical_pattern::{
    canonicalize, pattern_hash, structural_hash, CanonicalPattern, PatternHash, SlotId, VarMap,
};
pub use context::SolveContext;
pub use mode::{ExecutionMode, TreeComparison};
pub use property::{Property, PropertyConstraint, PropertySet};
pub use reason::{FailureReason, ImpossibilityProof, PartialReason, ResourceRequest};
pub use resource::{ResourceBudget, ResourceStatus};
pub use strategy::{
    MergerFn, Strategy, StrategyCandidate, StrategyResult, StrategyStatus, SubProblem,
};
pub use trace_tree::{
    BranchHandle, BranchOutcome, BranchReason, JoinHandle, JoinReason, PruneReason, StrategyId,
    TraceBranch, TraceNode, TraceTree,
};
