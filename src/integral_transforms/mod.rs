//! Symbolic integral transforms via table lookup and algebraic properties.
//!
//! Each transform (Laplace, Fourier, etc.) maintains a table of known
//! transform pairs and applies algebraic properties (linearity, shift,
//! derivative) to extend coverage beyond direct table hits.

use std::sync::Arc;

use crate::numeric::{Expr, SymbolId};
use num::Zero;

// Submodules — declared here, implemented in later subtasks
pub mod fourier_transform;
pub mod inverse_fourier;
pub mod inverse_laplace;
pub mod laplace;

// ── Error type ────────────────────────────────────────────────────────────────

/// Error type for integral transforms.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TransformError {
    /// No matching entry in the transform table.
    NoTableEntry(String),
    /// The input is not an elementary function for this transform.
    NonElementary(String),
    /// Invalid input (wrong form, missing variables, etc.).
    InvalidInput(String),
}

impl std::fmt::Display for TransformError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformError::NoTableEntry(msg) => write!(f, "no transform table entry: {msg}"),
            TransformError::NonElementary(msg) => {
                write!(f, "non-elementary for transform: {msg}")
            }
            TransformError::InvalidInput(msg) => write!(f, "invalid transform input: {msg}"),
        }
    }
}

impl std::error::Error for TransformError {}

// ── Result type ───────────────────────────────────────────────────────────────

/// Result of an integral transform computation.
#[derive(Debug, Clone)]
pub struct TransformResult {
    /// The transformed expression in `Arc<Expr>` canonical form.
    pub expr: Arc<Expr>,
    /// The new domain variable (e.g., `"s"` for Laplace, `"ω"` for Fourier).
    pub domain_var: String,
    /// Region of convergence / validity condition (optional).
    pub convergence: Option<String>,
    /// Narrated steps.
    pub steps: Vec<String>,
}

// ── Shared utilities ──────────────────────────────────────────────────────────

/// Split an expression into a list of `(coefficient, term)` pairs.
///
/// Given `a*f + b*g + c`, returns `[(a, f), (b, g), (c, 1)]`.
/// Operates on normalized `Arc<Expr>` trees.  For a non-additive
/// expression the result is a single pair `(1.0, expr)`.
///
/// Coefficients are extracted from the canonical `AddNode`/`MulNode`
/// representation, so the returned `f64` values are always exact for
/// rational coefficients that fit in `f64`.
pub fn split_linear_terms(expr: &Arc<Expr>, _var: SymbolId) -> Vec<(f64, Arc<Expr>)> {
    match expr.as_ref() {
        Expr::Add(node) => {
            let mut terms: Vec<(f64, Arc<Expr>)> = Vec::new();

            // Constant part of the sum (if non-zero)
            if !node.constant.is_zero() {
                let c = node.constant.to_f64();
                terms.push((
                    c,
                    Arc::new(Expr::Integer(crate::numeric::SmallInt::from(1i64))),
                ));
            }

            // Symbolic terms: each entry is (term, coeff)
            for (term, coeff) in &node.terms {
                terms.push((coeff.to_f64(), Arc::clone(term)));
            }

            if terms.is_empty() {
                // Zero expression
                terms.push((
                    0.0,
                    Arc::new(Expr::Integer(crate::numeric::SmallInt::from(0i64))),
                ));
            }

            terms
        }
        _ => {
            // For a single (possibly scaled) term, extract coefficient via MulNode
            let (coeff, term) = extract_coefficient(expr);
            vec![(coeff, term)]
        }
    }
}

/// Extract a leading numeric coefficient from an expression.
///
/// - `Mul(node)` with a rational coefficient → returns `(coeff, rest)` where
///   `rest` is the same `MulNode` expression with coefficient set to 1, or
///   the single factor if there is exactly one factor with exponent 1.
/// - A pure numeric expression → returns `(value, Integer(1))`.
/// - Anything else → returns `(1.0, expr)`.
fn extract_coefficient(expr: &Arc<Expr>) -> (f64, Arc<Expr>) {
    match expr.as_ref() {
        Expr::Mul(node) => {
            let coeff = node.coeff.to_f64();
            if node.factors.is_empty() {
                // Pure numeric Mul — treat as constant
                return (
                    coeff,
                    Arc::new(Expr::Integer(crate::numeric::SmallInt::from(1i64))),
                );
            }
            // Return the coefficient and the expression itself as the term,
            // but with a coefficient of 1.0 to avoid double-counting.
            // Callers use the (coeff, term) pair where term is the non-numeric part.
            // We reconstruct a unit-coefficient version by cloning the node.
            use crate::numeric::MulNode;
            let mut unit_node = MulNode::one();
            unit_node.factors = node.factors.clone();
            (coeff, Arc::new(Expr::Mul(unit_node)))
        }
        Expr::Integer(n) => {
            let c = n.to_i64().map(|v| v as f64).unwrap_or(f64::INFINITY);
            (
                c,
                Arc::new(Expr::Integer(crate::numeric::SmallInt::from(1i64))),
            )
        }
        Expr::Float(f) => (
            *f,
            Arc::new(Expr::Integer(crate::numeric::SmallInt::from(1i64))),
        ),
        Expr::Rational(r) => (
            r.to_f64(),
            Arc::new(Expr::Integer(crate::numeric::SmallInt::from(1i64))),
        ),
        _ => (1.0, Arc::clone(expr)),
    }
}

/// Try to extract a numeric constant from an expression.
///
/// Returns `Some(f64)` for `Integer`, `Float`, and `Rational` leaves;
/// `None` for all other variants.
pub fn as_constant(expr: &Arc<Expr>) -> Option<f64> {
    match expr.as_ref() {
        Expr::Integer(n) => n.to_i64().map(|v| v as f64),
        Expr::Float(f) => Some(*f),
        Expr::Rational(r) => Some(r.to_f64()),
        _ => None,
    }
}

/// Return `true` if `expr` contains the given variable anywhere in its tree.
pub fn contains_var(expr: &Arc<Expr>, var: SymbolId) -> bool {
    match expr.as_ref() {
        Expr::Symbol(s) => *s == var,
        Expr::Integer(_) | Expr::Float(_) | Expr::Rational(_) | Expr::Complex(_) => false,
        Expr::Constant(_) => false,
        Expr::Add(node) => node.terms.keys().any(|t| contains_var(t, var)),
        Expr::Mul(node) => node.factors.keys().any(|base| contains_var(base, var)),
        Expr::Pow(base, exp) => contains_var(base, var) || contains_var(exp, var),
        Expr::Func(_, args) => args.iter().any(|a| contains_var(a, var)),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Expression, Variable};
    use crate::numeric::compile::compile;

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    #[test]
    fn test_split_linear_terms_single() {
        let e = compile(&var("t"));
        let terms = split_linear_terms(&e, SymbolId::intern("t"));
        assert_eq!(terms.len(), 1);
    }

    #[test]
    fn test_split_linear_terms_sum() {
        // 2*t + 3
        let e = compile(&Expression::Binary(
            BinaryOp::Add,
            Box::new(Expression::Binary(
                BinaryOp::Mul,
                Box::new(int(2)),
                Box::new(var("t")),
            )),
            Box::new(int(3)),
        ));
        let terms = split_linear_terms(&e, SymbolId::intern("t"));
        assert!(terms.len() >= 2);
    }

    #[test]
    fn test_contains_var() {
        let e = compile(&var("t"));
        assert!(contains_var(&e, SymbolId::intern("t")));
        assert!(!contains_var(&e, SymbolId::intern("s")));
    }

    #[test]
    fn test_as_constant_integer() {
        let e = compile(&int(42));
        assert_eq!(as_constant(&e), Some(42.0));
    }

    #[test]
    fn test_as_constant_float() {
        let e = compile(&Expression::Float(3.14));
        assert_eq!(as_constant(&e), Some(3.14));
    }

    #[test]
    fn test_as_constant_symbol_is_none() {
        let e = compile(&var("x"));
        assert_eq!(as_constant(&e), None);
    }

    #[test]
    fn test_contains_var_in_sum() {
        // t + 5 — must contain t but not s
        let e = compile(&Expression::Binary(
            BinaryOp::Add,
            Box::new(var("t")),
            Box::new(int(5)),
        ));
        assert!(contains_var(&e, SymbolId::intern("t")));
        assert!(!contains_var(&e, SymbolId::intern("s")));
    }
}
