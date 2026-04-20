//! [`ExprPath`] — positional address into an [`Expression`] tree.
//!
//! A narrated step can point at the specific subexpression it manipulated.
//! Enables narratives like "factor the numerator of `x² − 4`" rather than
//! the ambiguous "factor".
//!
//! # Segments
//!
//! Path segments match the structural positions the thales `Expr` tree can
//! occupy. A path is an ordered sequence of segments interpreted root-to-leaf.

/// Path from an expression root to one of its subexpressions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExprPath(pub Vec<PathSegment>);

impl ExprPath {
    /// Empty path — refers to the root expression itself.
    #[must_use]
    pub const fn root() -> Self {
        Self(Vec::new())
    }

    /// Is this path the root?
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.0.is_empty()
    }

    /// Depth of the path.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.0.len()
    }

    /// Append a segment.
    pub fn push(&mut self, segment: PathSegment) {
        self.0.push(segment);
    }
}

/// One step of an [`ExprPath`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathSegment {
    /// n-th operand of an `Add` node.
    AddOperand(usize),
    /// n-th factor of a `Mul` node.
    MulFactor(usize),
    /// Base of a `Pow` node.
    PowBase,
    /// Exponent of a `Pow` node.
    PowExp,
    /// n-th argument of a `Func` node.
    FuncArg(usize),
    /// Numerator of a rational expression (sugar over `MulFactor(0)` with
    /// positive exponents).
    NumeratorOf,
    /// Denominator of a rational expression.
    DenominatorOf,
    /// Left-hand side of a binary relation (`=`, `<`, `≤`, …).
    LhsOf,
    /// Right-hand side of a binary relation.
    RhsOf,
    /// Cell `(row, col)` of a matrix expression.
    MatrixCell(usize, usize),
    /// Component index of a vector expression.
    VectorComponent(usize),
}
