//! [`ExprPath`] — positional address into an [`Expression`] tree.
//!
//! A narrated step can point at the specific subexpression it manipulated.
//! Enables narratives like "factor the numerator of `x² − 4`" rather than
//! the ambiguous "factor".
//!
//! # Segments
//!
//! Path segments describe the structural positions an `Expression` tree can
//! occupy. A path is an ordered sequence of segments interpreted
//! root-to-leaf. [`resolve`] walks a path against an `Expression` and
//! returns the subexpression at that position, or `None` if the path does
//! not address a valid subtree.

use crate::ast::{BinaryOp, Expression};

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
    /// n-th operand of a left-associative `Add` chain. Flattening walks
    /// `Binary(Add, l, r)` recursively so `a + b + c + d` exposes four
    /// operands indexed 0..3. `Subtract` is treated as `Add(negated rhs)`:
    /// `a − b − c` exposes three operands.
    AddOperand(usize),
    /// n-th factor of a left-associative `Mul` chain. Same flattening as
    /// `AddOperand`. `Divide` is treated as `Mul(reciprocal rhs)`.
    MulFactor(usize),
    /// Base of a `Power` node.
    PowBase,
    /// Exponent of a `Power` node.
    PowExp,
    /// n-th argument of a `Function` call.
    FuncArg(usize),
    /// Numerator of a `Binary(Divide, n, d)` or left operand of a rational
    /// expression.
    NumeratorOf,
    /// Denominator of a `Binary(Divide, n, d)`.
    DenominatorOf,
    /// Left operand of any `Binary` node (including equations when the
    /// `BinaryOp` encodes a relation).
    LhsOf,
    /// Right operand of any `Binary` node.
    RhsOf,
    /// Cell `(row, col)` of a matrix expression. Reserved — `Expression`
    /// does not currently carry matrix variants; tracked for future
    /// `Matrix` integration.
    MatrixCell(usize, usize),
    /// Component index of a vector expression. Reserved (see
    /// `MatrixCell`).
    VectorComponent(usize),
}

/// Walk `path` against `expr`, returning the subexpression at the addressed
/// position.
///
/// Returns `None` if any segment does not match the structure encountered at
/// that depth (e.g. `PowBase` on a leaf, `AddOperand(5)` when only three
/// operands exist, `MatrixCell` — reserved for future matrix variants).
#[must_use]
pub fn resolve<'a>(expr: &'a Expression, path: &ExprPath) -> Option<&'a Expression> {
    let mut current = expr;
    for seg in &path.0 {
        current = step(current, *seg)?;
    }
    Some(current)
}

fn step(expr: &Expression, seg: PathSegment) -> Option<&Expression> {
    match (expr, seg) {
        // Power: base / exponent.
        (Expression::Power(b, _), PathSegment::PowBase) => Some(b),
        (Expression::Power(_, e), PathSegment::PowExp) => Some(e),

        // Function argument.
        (Expression::Function(_, args), PathSegment::FuncArg(n)) => args.get(n),

        // Binary left / right generic.
        (Expression::Binary(_, l, _), PathSegment::LhsOf) => Some(l),
        (Expression::Binary(_, _, r), PathSegment::RhsOf) => Some(r),

        // Divide numerator / denominator.
        (Expression::Binary(BinaryOp::Div, n, _), PathSegment::NumeratorOf) => Some(n),
        (Expression::Binary(BinaryOp::Div, _, d), PathSegment::DenominatorOf) => Some(d),

        // Add chain operand: flatten Binary(Add / Sub, …) chain left-to-right.
        (Expression::Binary(BinaryOp::Add, _, _), PathSegment::AddOperand(n))
        | (Expression::Binary(BinaryOp::Sub, _, _), PathSegment::AddOperand(n)) => {
            collect_add_chain(expr).into_iter().nth(n)
        }

        // Mul chain factor: flatten Binary(Mul / Div, …) chain left-to-right.
        (Expression::Binary(BinaryOp::Mul, _, _), PathSegment::MulFactor(n))
        | (Expression::Binary(BinaryOp::Div, _, _), PathSegment::MulFactor(n)) => {
            collect_mul_chain(expr).into_iter().nth(n)
        }

        // Everything else: unmatched.
        _ => None,
    }
}

fn collect_add_chain(expr: &Expression) -> Vec<&Expression> {
    let mut out = Vec::new();
    push_add_chain(expr, &mut out);
    out
}

fn push_add_chain<'a>(expr: &'a Expression, out: &mut Vec<&'a Expression>) {
    match expr {
        Expression::Binary(BinaryOp::Add, l, r) => {
            push_add_chain(l, out);
            push_add_chain(r, out);
        }
        Expression::Binary(BinaryOp::Sub, l, r) => {
            push_add_chain(l, out);
            // Right-hand side of Subtract is logically negated; for
            // path addressing we expose it as the raw subtree so the
            // caller can see `b` in `a − b`.
            push_add_chain(r, out);
        }
        other => out.push(other),
    }
}

fn collect_mul_chain(expr: &Expression) -> Vec<&Expression> {
    let mut out = Vec::new();
    push_mul_chain(expr, &mut out);
    out
}

fn push_mul_chain<'a>(expr: &'a Expression, out: &mut Vec<&'a Expression>) {
    match expr {
        Expression::Binary(BinaryOp::Mul, l, r) => {
            push_mul_chain(l, out);
            push_mul_chain(r, out);
        }
        Expression::Binary(BinaryOp::Div, l, r) => {
            push_mul_chain(l, out);
            // Denominator exposed as raw subtree.
            push_mul_chain(r, out);
        }
        other => out.push(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Expression, Function};

    fn var(name: &str) -> Expression {
        Expression::Variable(crate::ast::Variable::new(name))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn bin(op: BinaryOp, l: Expression, r: Expression) -> Expression {
        Expression::Binary(op, Box::new(l), Box::new(r))
    }

    #[test]
    fn root_resolves_to_self() {
        let e = var("x");
        let path = ExprPath::root();
        assert_eq!(resolve(&e, &path), Some(&e));
    }

    #[test]
    fn power_base_and_exp() {
        let e = Expression::Power(Box::new(var("x")), Box::new(int(2)));
        let base = resolve(&e, &ExprPath(vec![PathSegment::PowBase])).unwrap();
        let exp = resolve(&e, &ExprPath(vec![PathSegment::PowExp])).unwrap();
        assert_eq!(base, &var("x"));
        assert_eq!(exp, &int(2));
    }

    #[test]
    fn func_arg() {
        let e = Expression::Function(Function::Sin, vec![var("x")]);
        let arg = resolve(&e, &ExprPath(vec![PathSegment::FuncArg(0)])).unwrap();
        assert_eq!(arg, &var("x"));
        assert!(resolve(&e, &ExprPath(vec![PathSegment::FuncArg(1)])).is_none());
    }

    #[test]
    fn binary_lhs_rhs() {
        let e = bin(BinaryOp::Add, var("x"), int(1));
        assert_eq!(
            resolve(&e, &ExprPath(vec![PathSegment::LhsOf])),
            Some(&var("x"))
        );
        assert_eq!(
            resolve(&e, &ExprPath(vec![PathSegment::RhsOf])),
            Some(&int(1))
        );
    }

    #[test]
    fn divide_numerator_denominator() {
        let e = bin(BinaryOp::Div, var("x"), int(2));
        assert_eq!(
            resolve(&e, &ExprPath(vec![PathSegment::NumeratorOf])),
            Some(&var("x"))
        );
        assert_eq!(
            resolve(&e, &ExprPath(vec![PathSegment::DenominatorOf])),
            Some(&int(2))
        );
    }

    #[test]
    fn add_chain_flattened() {
        // ((x + y) + z)
        let e = bin(
            BinaryOp::Add,
            bin(BinaryOp::Add, var("x"), var("y")),
            var("z"),
        );
        assert_eq!(
            resolve(&e, &ExprPath(vec![PathSegment::AddOperand(0)])),
            Some(&var("x"))
        );
        assert_eq!(
            resolve(&e, &ExprPath(vec![PathSegment::AddOperand(1)])),
            Some(&var("y"))
        );
        assert_eq!(
            resolve(&e, &ExprPath(vec![PathSegment::AddOperand(2)])),
            Some(&var("z"))
        );
        assert!(resolve(&e, &ExprPath(vec![PathSegment::AddOperand(3)])).is_none());
    }

    #[test]
    fn mul_chain_flattened() {
        // ((x * y) * z)
        let e = bin(
            BinaryOp::Mul,
            bin(BinaryOp::Mul, var("x"), var("y")),
            var("z"),
        );
        assert_eq!(
            resolve(&e, &ExprPath(vec![PathSegment::MulFactor(0)])),
            Some(&var("x"))
        );
        assert_eq!(
            resolve(&e, &ExprPath(vec![PathSegment::MulFactor(2)])),
            Some(&var("z"))
        );
    }

    #[test]
    fn nested_path() {
        // sin(x + 1) → FuncArg(0) → AddOperand(1) → 1
        let e = Expression::Function(Function::Sin, vec![bin(BinaryOp::Add, var("x"), int(1))]);
        let path = ExprPath(vec![PathSegment::FuncArg(0), PathSegment::AddOperand(1)]);
        assert_eq!(resolve(&e, &path), Some(&int(1)));
    }

    #[test]
    fn invalid_segment_returns_none() {
        let e = var("x");
        assert!(resolve(&e, &ExprPath(vec![PathSegment::PowBase])).is_none());
    }

    #[test]
    fn matrix_segment_reserved() {
        let e = var("x");
        assert!(resolve(&e, &ExprPath(vec![PathSegment::MatrixCell(0, 0)])).is_none());
        assert!(resolve(&e, &ExprPath(vec![PathSegment::VectorComponent(0)])).is_none());
    }
}
