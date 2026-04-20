//! [`Condition`] — predicate under which a [`super::ResultEntry`] applies.
//!
//! Attached to [`super::ResultKey::Branch`] to partition the solution space
//! across multiple entries in a [`super::Response`].

use crate::Expression;

use super::Domain;

/// Predicate under which a result is valid.
///
/// Use cases:
///
/// - **Multi-root equations**: one `Branch(Eq(var, root))` entry per root.
/// - **Parametric solutions**: `Parametric` with an integer parameter `n ∈ ℤ`
///   captures `x = π/6 + 2πn`.
/// - **Inequalities**: `Interval` per disjoint interval in the solution set.
/// - **Piecewise / one-sided limits**: `Case("from the left")`, etc.
/// - **Conditional convergence**: `Case("converges")` + `Parametric` guard on
///   the parameter.
/// - **ODE / matrix**: `Case("general")`, `Case("particular")`,
///   `Case("homogeneous")`, `Case("unique")`, …
#[derive(Debug, Clone, PartialEq)]
pub enum Condition {
    /// Variable equals a specific expression. e.g. `x = 2`.
    Eq(String, Expression),
    /// Variable lies in a closed / open interval.
    Interval(String, Bound, Bound),
    /// Logical combination of conditions.
    Compound(Box<CompoundCondition>),
    /// Main condition parametrised over a free variable in a domain.
    /// e.g. `x = π/6 + 2πn, n ∈ ℤ` → `Parametric { main: Eq("x", ...), parameter: "n", domain: Integer }`.
    Parametric {
        /// Primary condition, referencing `parameter` in its expression.
        main: Box<Condition>,
        /// Name of the free parameter.
        parameter: String,
        /// Domain over which the parameter ranges.
        domain: Domain,
    },
    /// Stable free-form label. Use for enumerable cases that do not reduce to
    /// the structured variants above ("general", "particular", "homogeneous",
    /// "converges", "unique", "from the left", …).
    Case(String),
    /// Variable restricted to a sub-domain.
    InDomain(String, Domain),
}

/// Logical combinations of [`Condition`]s.
#[derive(Debug, Clone, PartialEq)]
pub enum CompoundCondition {
    /// All children must hold.
    And(Vec<Condition>),
    /// At least one child must hold.
    Or(Vec<Condition>),
    /// Negation of the wrapped condition.
    Not(Condition),
}

/// Interval endpoint.
#[derive(Debug, Clone, PartialEq)]
pub enum Bound {
    /// Open endpoint at the given expression (excluded).
    Open(Expression),
    /// Closed endpoint at the given expression (included).
    Closed(Expression),
    /// Negative infinity (always open).
    NegInf,
    /// Positive infinity (always open).
    PosInf,
}
