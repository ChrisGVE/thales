//! D0 search strategy engine.
//!
//! This module contains the foundational types for the D0 search strategy
//! engine: trace trees, resource budgets, assumption sets, property sets,
//! reason types, execution modes, and canonical pattern representation.

pub mod assumption;
pub mod canonical_pattern;
pub mod mode;
pub mod property;
pub mod reason;
pub mod resource;
pub mod trace_tree;

pub use assumption::{AssumptionGuard, AssumptionSet};
pub use canonical_pattern::{
    CanonicalPattern, PatternHash, SlotId, VarMap, canonicalize, pattern_hash, structural_hash,
};
pub use mode::{ExecutionMode, TreeComparison};
pub use property::{Property, PropertyConstraint, PropertySet};
pub use reason::{FailureReason, ImpossibilityProof, PartialReason, ResourceRequest};
pub use resource::{ResourceBudget, ResourceStatus};
pub use trace_tree::{
    BranchHandle, BranchOutcome, BranchReason, JoinHandle, JoinReason, PruneReason, StrategyId,
    TraceBranch, TraceNode, TraceTree,
};
