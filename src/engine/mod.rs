//! D0 search strategy engine.
//!
//! This module contains the foundational types for the D0 search strategy
//! engine: trace trees, resource budgets, assumption sets, property sets,
//! reason types, execution modes, canonical pattern representation, and
//! assumption normalization / entailment for D0.12 pattern recognition.

pub mod assumption;
pub mod assumption_key;
pub mod canonical_pattern;
pub mod mode;
pub mod property;
pub mod reason;
pub mod resource;
pub mod trace_tree;

mod assumption_entailment;

pub use assumption::{AssumptionGuard, AssumptionSet};
pub use assumption_key::{
    entails, normalize_assumption, sign_with_varmap, AssumptionConstraint, AssumptionSignature,
    Domain, NormalizedAssumption,
};
pub use canonical_pattern::{
    canonicalize, pattern_hash, structural_hash, CanonicalPattern, PatternHash, SlotId, VarMap,
};
pub use mode::{ExecutionMode, TreeComparison};
pub use property::{Property, PropertyConstraint, PropertySet};
pub use reason::{FailureReason, ImpossibilityProof, PartialReason, ResourceRequest};
pub use resource::{ResourceBudget, ResourceStatus};
pub use trace_tree::{
    BranchHandle, BranchOutcome, BranchReason, JoinHandle, JoinReason, PruneReason, StrategyId,
    TraceBranch, TraceNode, TraceTree,
};
