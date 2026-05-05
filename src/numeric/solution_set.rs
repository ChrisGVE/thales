//! Comprehensive solution set type for equation solving.
//!
//! [`SolutionSet`] replaces the simple `Solution` enum with a rich type
//! supporting all solution kinds: finite sets, intervals, unions,
//! complements, parametric families, and conditional solutions.

use super::expr::Expr;
use std::fmt;
use std::sync::Arc;

/// Represents the complete solution set of an equation or system.
///
/// Supports all standard mathematical solution types including finite
/// sets, intervals, unions, complements, and parametric families.
#[derive(Clone, Debug, PartialEq)]
pub enum SolutionSet {
    /// No solution exists (empty set, ∅).
    Empty,

    /// A finite set of explicit solutions.
    Finite(Vec<Arc<Expr>>),

    /// A continuous interval `(a, b)`, `[a, b]`, `[a, b)`, or `(a, b]`.
    Interval {
        /// Lower bound of the interval.
        low: IntervalBound,
        /// Upper bound of the interval.
        high: IntervalBound,
    },

    /// Union of multiple solution sets: S₁ ∪ S₂ ∪ ... ∪ Sₙ.
    Union(Vec<SolutionSet>),

    /// Complement of a set relative to the reals: ℝ \ S.
    Complement(Box<SolutionSet>),

    /// Parametric solution family with free variables and constraints.
    Parametric {
        /// Free variables (parameters).
        free_vars: Vec<Arc<Expr>>,
        /// Solution expressions (one per solve variable).
        expressions: Vec<Arc<Expr>>,
        /// Constraints on the free variables.
        constraints: Vec<Constraint>,
    },

    /// A conditional solution: valid only when a condition holds.
    Conditional {
        /// Boolean condition that must be satisfied.
        condition: Arc<Expr>,
        /// The solution set, valid when `condition` holds.
        set: Box<SolutionSet>,
    },
}

/// A bound of an interval, possibly infinite.
#[derive(Clone, Debug, PartialEq)]
pub enum IntervalBound {
    /// Negative infinity (always open).
    NegInfinity,
    /// Positive infinity (always open).
    PosInfinity,
    /// A finite bound with open/closed flag.
    Finite {
        /// The boundary value.
        value: Arc<Expr>,
        /// Whether the bound is inclusive (`[` or `]`) vs exclusive (`(` or `)`).
        inclusive: bool,
    },
}

/// A constraint on a solution (e.g., x > 0, x ≠ 1).
#[derive(Clone, Debug, PartialEq)]
pub enum Constraint {
    /// Variable not equal to a value.
    NotEqual(Arc<Expr>, Arc<Expr>),
    /// Expression must be positive.
    Positive(Arc<Expr>),
    /// Expression must be non-negative.
    NonNegative(Arc<Expr>),
    /// General boolean expression that must hold.
    Predicate(Arc<Expr>),
}

// ── Constructors ────────────────────────────────────────────────────────────

impl SolutionSet {
    /// The empty set (no solutions).
    pub fn empty() -> Self {
        SolutionSet::Empty
    }

    /// All real numbers (universal set).
    pub fn all_reals() -> Self {
        SolutionSet::Interval {
            low: IntervalBound::NegInfinity,
            high: IntervalBound::PosInfinity,
        }
    }

    /// A single solution.
    pub fn singleton(value: Arc<Expr>) -> Self {
        SolutionSet::Finite(vec![value])
    }

    /// A finite set from an iterator.
    pub fn from_values(values: impl IntoIterator<Item = Arc<Expr>>) -> Self {
        let v: Vec<_> = values.into_iter().collect();
        if v.is_empty() {
            SolutionSet::Empty
        } else {
            SolutionSet::Finite(v)
        }
    }

    /// Open interval `(a, b)`.
    pub fn open(a: Arc<Expr>, b: Arc<Expr>) -> Self {
        SolutionSet::Interval {
            low: IntervalBound::Finite {
                value: a,
                inclusive: false,
            },
            high: IntervalBound::Finite {
                value: b,
                inclusive: false,
            },
        }
    }

    /// Closed interval `[a, b]`.
    pub fn closed(a: Arc<Expr>, b: Arc<Expr>) -> Self {
        SolutionSet::Interval {
            low: IntervalBound::Finite {
                value: a,
                inclusive: true,
            },
            high: IntervalBound::Finite {
                value: b,
                inclusive: true,
            },
        }
    }

    /// Half-open interval `[a, b)`.
    pub fn closed_open(a: Arc<Expr>, b: Arc<Expr>) -> Self {
        SolutionSet::Interval {
            low: IntervalBound::Finite {
                value: a,
                inclusive: true,
            },
            high: IntervalBound::Finite {
                value: b,
                inclusive: false,
            },
        }
    }

    /// Half-open interval `(a, b]`.
    pub fn open_closed(a: Arc<Expr>, b: Arc<Expr>) -> Self {
        SolutionSet::Interval {
            low: IntervalBound::Finite {
                value: a,
                inclusive: false,
            },
            high: IntervalBound::Finite {
                value: b,
                inclusive: true,
            },
        }
    }

    /// `(-∞, b)` or `(-∞, b]`.
    pub fn less_than(b: Arc<Expr>, inclusive: bool) -> Self {
        SolutionSet::Interval {
            low: IntervalBound::NegInfinity,
            high: IntervalBound::Finite {
                value: b,
                inclusive,
            },
        }
    }

    /// `(a, ∞)` or `[a, ∞)`.
    pub fn greater_than(a: Arc<Expr>, inclusive: bool) -> Self {
        SolutionSet::Interval {
            low: IntervalBound::Finite {
                value: a,
                inclusive,
            },
            high: IntervalBound::PosInfinity,
        }
    }

    /// ℝ \ {excluded values}.
    pub fn all_except(excluded: Vec<Arc<Expr>>) -> Self {
        if excluded.is_empty() {
            return SolutionSet::all_reals();
        }
        SolutionSet::Complement(Box::new(SolutionSet::Finite(excluded)))
    }
}

// ── Set operations ──────────────────────────────────────────────────────────

impl SolutionSet {
    /// Union of two solution sets, with basic simplification.
    pub fn union(self, other: SolutionSet) -> SolutionSet {
        match (&self, &other) {
            (SolutionSet::Empty, _) => other,
            (_, SolutionSet::Empty) => self,
            _ => {
                let mut parts = Vec::new();
                // Flatten nested unions
                flatten_union(self, &mut parts);
                flatten_union(other, &mut parts);
                if parts.len() == 1 {
                    parts.into_iter().next().unwrap()
                } else {
                    SolutionSet::Union(parts)
                }
            }
        }
    }

    /// Complement relative to ℝ.
    pub fn complement(self) -> SolutionSet {
        match self {
            SolutionSet::Empty => SolutionSet::all_reals(),
            SolutionSet::Complement(inner) => *inner,
            other => SolutionSet::Complement(Box::new(other)),
        }
    }

    /// Returns `true` if this is the empty set.
    pub fn is_empty(&self) -> bool {
        matches!(self, SolutionSet::Empty)
    }

    /// Returns `true` if this is a finite set.
    pub fn is_finite(&self) -> bool {
        matches!(self, SolutionSet::Finite(_))
    }

    /// Returns the number of solutions, if finite.
    pub fn cardinality(&self) -> Option<usize> {
        match self {
            SolutionSet::Empty => Some(0),
            SolutionSet::Finite(v) => Some(v.len()),
            _ => None,
        }
    }

    /// Extract finite solutions as a slice, if this is a finite set.
    pub fn as_finite(&self) -> Option<&[Arc<Expr>]> {
        match self {
            SolutionSet::Finite(v) => Some(v),
            _ => None,
        }
    }
}

/// Flatten nested Union variants into a single Vec.
fn flatten_union(set: SolutionSet, out: &mut Vec<SolutionSet>) {
    match set {
        SolutionSet::Union(parts) => {
            for part in parts {
                flatten_union(part, out);
            }
        }
        SolutionSet::Empty => {} // Skip empties
        other => out.push(other),
    }
}

// ── Display ─────────────────────────────────────────────────────────────────

impl fmt::Display for SolutionSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolutionSet::Empty => write!(f, "∅"),
            SolutionSet::Finite(vals) => {
                write!(f, "{{")?;
                for (i, v) in vals.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "}}")
            }
            SolutionSet::Interval { low, high } => {
                write!(f, "{}, {}", low.display_low(), high.display_high())
            }
            SolutionSet::Union(parts) => {
                for (i, part) in parts.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ∪ ")?;
                    }
                    write!(f, "{part}")?;
                }
                Ok(())
            }
            SolutionSet::Complement(inner) => write!(f, "ℝ \\ {inner}"),
            SolutionSet::Parametric {
                free_vars,
                expressions,
                constraints,
            } => {
                write!(f, "{{")?;
                for (i, expr) in expressions.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{expr}")?;
                }
                write!(f, " | ")?;
                for (i, var) in free_vars.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{var} ∈ ℝ")?;
                }
                for c in constraints {
                    write!(f, ", {c}")?;
                }
                write!(f, "}}")
            }
            SolutionSet::Conditional { condition, set } => {
                write!(f, "{set} if {condition}")
            }
        }
    }
}

impl IntervalBound {
    /// Format as the low (left) bound of an interval.
    fn display_low(&self) -> String {
        match self {
            IntervalBound::NegInfinity => "(-∞".to_string(),
            IntervalBound::PosInfinity => "∞)".to_string(),
            IntervalBound::Finite { value, inclusive } => {
                if *inclusive {
                    format!("[{value}")
                } else {
                    format!("({value}")
                }
            }
        }
    }

    /// Format as the high (right) bound of an interval.
    fn display_high(&self) -> String {
        match self {
            IntervalBound::NegInfinity => "-∞)".to_string(),
            IntervalBound::PosInfinity => "∞)".to_string(),
            IntervalBound::Finite { value, inclusive } => {
                if *inclusive {
                    format!("{value}]")
                } else {
                    format!("{value})")
                }
            }
        }
    }
}

impl fmt::Display for IntervalBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_low())
    }
}

impl fmt::Display for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constraint::NotEqual(a, b) => write!(f, "{a} ≠ {b}"),
            Constraint::Positive(e) => write!(f, "{e} > 0"),
            Constraint::NonNegative(e) => write!(f, "{e} ≥ 0"),
            Constraint::Predicate(e) => write!(f, "{e}"),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    fn sym(name: &str) -> Arc<Expr> {
        Expr::symbol(name)
    }

    #[test]
    fn test_empty() {
        let s = SolutionSet::empty();
        assert!(s.is_empty());
        assert_eq!(s.cardinality(), Some(0));
        assert_eq!(s.to_string(), "∅");
    }

    #[test]
    fn test_singleton() {
        let s = SolutionSet::singleton(Expr::int(42));
        assert!(s.is_finite());
        assert_eq!(s.cardinality(), Some(1));
        assert_eq!(s.to_string(), "{42}");
    }

    #[test]
    fn test_finite_multiple() {
        let s = SolutionSet::from_values(vec![Expr::int(1), Expr::int(-1)]);
        assert_eq!(s.cardinality(), Some(2));
        assert_eq!(s.to_string(), "{1, -1}");
    }

    #[test]
    fn test_from_values_empty() {
        let s = SolutionSet::from_values(Vec::<Arc<Expr>>::new());
        assert!(s.is_empty());
    }

    #[test]
    fn test_open_interval() {
        let s = SolutionSet::open(Expr::int(-1), Expr::int(1));
        assert_eq!(s.cardinality(), None);
        assert_eq!(s.to_string(), "(-1, 1)");
    }

    #[test]
    fn test_closed_interval() {
        let s = SolutionSet::closed(Expr::int(0), Expr::int(10));
        assert_eq!(s.to_string(), "[0, 10]");
    }

    #[test]
    fn test_half_open_intervals() {
        let co = SolutionSet::closed_open(Expr::int(0), Expr::int(1));
        assert_eq!(co.to_string(), "[0, 1)");

        let oc = SolutionSet::open_closed(Expr::int(0), Expr::int(1));
        assert_eq!(oc.to_string(), "(0, 1]");
    }

    #[test]
    fn test_less_than() {
        let s = SolutionSet::less_than(Expr::int(5), false);
        assert_eq!(s.to_string(), "(-∞, 5)");
    }

    #[test]
    fn test_greater_than_inclusive() {
        let s = SolutionSet::greater_than(Expr::int(0), true);
        assert_eq!(s.to_string(), "[0, ∞)");
    }

    #[test]
    fn test_all_reals() {
        let s = SolutionSet::all_reals();
        assert_eq!(s.to_string(), "(-∞, ∞)");
    }

    #[test]
    fn test_complement_empty_is_reals() {
        let s = SolutionSet::empty().complement();
        assert_eq!(s, SolutionSet::all_reals());
    }

    #[test]
    fn test_complement_of_complement() {
        let inner = SolutionSet::singleton(Expr::int(0));
        let double = inner.clone().complement().complement();
        assert_eq!(double, inner);
    }

    #[test]
    fn test_all_except() {
        let s = SolutionSet::all_except(vec![Expr::int(0), Expr::int(1)]);
        assert_eq!(s.to_string(), "ℝ \\ {0, 1}");
    }

    #[test]
    fn test_all_except_empty_is_reals() {
        let s = SolutionSet::all_except(vec![]);
        assert_eq!(s, SolutionSet::all_reals());
    }

    #[test]
    fn test_union_basic() {
        let a = SolutionSet::less_than(Expr::int(-1), false);
        let b = SolutionSet::greater_than(Expr::int(1), false);
        let u = a.union(b);
        match &u {
            SolutionSet::Union(parts) => assert_eq!(parts.len(), 2),
            _ => panic!("expected Union"),
        }
    }

    #[test]
    fn test_union_with_empty() {
        let s = SolutionSet::singleton(Expr::int(5));
        let u = SolutionSet::empty().union(s.clone());
        assert_eq!(u, s);
    }

    #[test]
    fn test_union_flattens() {
        let a = SolutionSet::singleton(Expr::int(1));
        let b = SolutionSet::singleton(Expr::int(2));
        let c = SolutionSet::singleton(Expr::int(3));
        let u = a.union(b).union(c);
        match &u {
            SolutionSet::Union(parts) => assert_eq!(parts.len(), 3),
            _ => panic!("expected flat Union with 3 parts"),
        }
    }

    #[test]
    fn test_x_squared_eq_one() {
        // x² = 1 → {1, -1}
        let s = SolutionSet::from_values(vec![Expr::int(1), Expr::int(-1)]);
        assert_eq!(s.cardinality(), Some(2));
    }

    #[test]
    fn test_x_squared_ge_zero() {
        // x² ≥ 0 → ℝ (universal)
        let s = SolutionSet::all_reals();
        assert_eq!(s.cardinality(), None);
    }

    #[test]
    fn test_x_squared_lt_zero() {
        // x² < 0 → ∅
        let s = SolutionSet::empty();
        assert!(s.is_empty());
    }

    #[test]
    fn test_abs_x_lt_one() {
        // |x| < 1 → (-1, 1)
        let s = SolutionSet::open(Expr::int(-1), Expr::int(1));
        match &s {
            SolutionSet::Interval { low, high } => {
                match low {
                    IntervalBound::Finite { inclusive, .. } => assert!(!inclusive),
                    _ => panic!("expected finite low"),
                }
                match high {
                    IntervalBound::Finite { inclusive, .. } => assert!(!inclusive),
                    _ => panic!("expected finite high"),
                }
            }
            _ => panic!("expected Interval"),
        }
    }

    #[test]
    fn test_parametric() {
        let x = sym("ss_param_x");
        let y = sym("ss_param_y");
        let s = SolutionSet::Parametric {
            free_vars: vec![y.clone()],
            expressions: vec![x.clone()],
            constraints: vec![Constraint::Positive(y)],
        };
        assert_eq!(s.cardinality(), None);
    }

    #[test]
    fn test_conditional() {
        let cond = sym("ss_cond_c");
        let inner = SolutionSet::singleton(Expr::int(1));
        let s = SolutionSet::Conditional {
            condition: cond,
            set: Box::new(inner),
        };
        assert_eq!(s.cardinality(), None);
    }

    #[test]
    fn test_as_finite() {
        let s = SolutionSet::from_values(vec![Expr::int(1), Expr::int(2)]);
        assert_eq!(s.as_finite().unwrap().len(), 2);

        let s2 = SolutionSet::all_reals();
        assert!(s2.as_finite().is_none());
    }

    #[test]
    fn test_display_interval_high_bound() {
        let s = SolutionSet::Interval {
            low: IntervalBound::Finite {
                value: Expr::int(0),
                inclusive: true,
            },
            high: IntervalBound::Finite {
                value: Expr::int(5),
                inclusive: false,
            },
        };
        assert_eq!(s.to_string(), "[0, 5)");
    }

    #[test]
    fn test_constraint_display() {
        let x = sym("ss_cd_x");
        assert_eq!(
            Constraint::NotEqual(x.clone(), Expr::int(0)).to_string(),
            "ss_cd_x ≠ 0"
        );
        assert_eq!(Constraint::Positive(x.clone()).to_string(), "ss_cd_x > 0");
        assert_eq!(
            Constraint::NonNegative(x.clone()).to_string(),
            "ss_cd_x ≥ 0"
        );
    }
}
