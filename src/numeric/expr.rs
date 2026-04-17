//! Arc-based expression type with structural sharing.
//!
//! [`Expr`] is the new expression representation for the CAS. All compound
//! expressions hold children via `Arc<Expr>`, enabling O(1) clone and
//! structural sharing of common sub-expressions.
//!
//! Use [`ExprPool`] for automatic common sub-expression elimination
//! during construction: identical sub-expressions return the same `Arc`.

use super::{AddNode, BigRational, MulNode, SmallInt, SymbolId};
use crate::ast::SymbolicConstant;
use num_complex::Complex64;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Weak};

// ── FuncId ────────────────────────────────────────────────────────────────────

/// Identifier for built-in and user-defined functions.
///
/// Covers the standard transcendental functions used throughout the CAS.
/// [`FuncId::Other`] holds a [`SymbolId`] for user-defined or less common
/// functions, keeping the enum open for extension without breaking changes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FuncId {
    // ── Trigonometric ────────────────────────────────────────────────────────
    /// Sine function.
    Sin,
    /// Cosine function.
    Cos,
    /// Tangent function.
    Tan,
    // ── Inverse trigonometric ────────────────────────────────────────────────
    /// Arcsine function.
    Asin,
    /// Arccosine function.
    Acos,
    /// Arctangent function (single argument).
    Atan,
    /// Two-argument arctangent: atan2(y, x).
    Atan2,
    // ── Hyperbolic ───────────────────────────────────────────────────────────
    /// Hyperbolic sine.
    Sinh,
    /// Hyperbolic cosine.
    Cosh,
    /// Hyperbolic tangent.
    Tanh,
    // ── Logarithmic ─────────────────────────────────────────────────────────
    /// Natural logarithm.
    Ln,
    /// Natural exponential.
    Exp,
    /// Logarithm with arbitrary base: log(base, x).
    Log,
    /// Base-2 logarithm.
    Log2,
    /// Base-10 logarithm.
    Log10,
    // ── Power / root ─────────────────────────────────────────────────────────
    /// Square root.
    Sqrt,
    /// Cube root.
    Cbrt,
    // ── Rounding ─────────────────────────────────────────────────────────────
    /// Floor function.
    Floor,
    /// Ceiling function.
    Ceil,
    /// Round to nearest integer.
    Round,
    // ── Utility ──────────────────────────────────────────────────────────────
    /// Absolute value.
    Abs,
    /// Sign (signum) function.
    Sign,
    /// Minimum of two arguments.
    Min,
    /// Maximum of two arguments.
    Max,
    /// User-defined or extension function identified by a [`SymbolId`].
    Other(SymbolId),
}

impl PartialOrd for FuncId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FuncId {
    fn cmp(&self, other: &Self) -> Ordering {
        func_id_rank(self)
            .cmp(&func_id_rank(other))
            .then_with(|| match (self, other) {
                (FuncId::Other(a), FuncId::Other(b)) => a.cmp(b),
                _ => Ordering::Equal,
            })
    }
}

fn func_id_rank(f: &FuncId) -> u8 {
    match f {
        FuncId::Sin => 0,
        FuncId::Cos => 1,
        FuncId::Tan => 2,
        FuncId::Asin => 3,
        FuncId::Acos => 4,
        FuncId::Atan => 5,
        FuncId::Atan2 => 6,
        FuncId::Sinh => 7,
        FuncId::Cosh => 8,
        FuncId::Tanh => 9,
        FuncId::Ln => 10,
        FuncId::Exp => 11,
        FuncId::Log => 12,
        FuncId::Log2 => 13,
        FuncId::Log10 => 14,
        FuncId::Sqrt => 15,
        FuncId::Cbrt => 16,
        FuncId::Floor => 17,
        FuncId::Ceil => 18,
        FuncId::Round => 19,
        FuncId::Abs => 20,
        FuncId::Sign => 21,
        FuncId::Min => 22,
        FuncId::Max => 23,
        FuncId::Other(_) => 24,
    }
}

impl fmt::Display for FuncId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FuncId::Sin => write!(f, "sin"),
            FuncId::Cos => write!(f, "cos"),
            FuncId::Tan => write!(f, "tan"),
            FuncId::Asin => write!(f, "asin"),
            FuncId::Acos => write!(f, "acos"),
            FuncId::Atan => write!(f, "atan"),
            FuncId::Atan2 => write!(f, "atan2"),
            FuncId::Sinh => write!(f, "sinh"),
            FuncId::Cosh => write!(f, "cosh"),
            FuncId::Tanh => write!(f, "tanh"),
            FuncId::Ln => write!(f, "ln"),
            FuncId::Exp => write!(f, "exp"),
            FuncId::Log => write!(f, "log"),
            FuncId::Log2 => write!(f, "log2"),
            FuncId::Log10 => write!(f, "log10"),
            FuncId::Sqrt => write!(f, "sqrt"),
            FuncId::Cbrt => write!(f, "cbrt"),
            FuncId::Floor => write!(f, "floor"),
            FuncId::Ceil => write!(f, "ceil"),
            FuncId::Round => write!(f, "round"),
            FuncId::Abs => write!(f, "abs"),
            FuncId::Sign => write!(f, "sign"),
            FuncId::Min => write!(f, "min"),
            FuncId::Max => write!(f, "max"),
            FuncId::Other(s) => write!(f, "{s}"),
        }
    }
}

// ── Expr ──────────────────────────────────────────────────────────────────────

/// Core expression type for the computer algebra system.
///
/// All compound expressions hold children via `Arc<Expr>`, enabling
/// O(1) clone and structural sharing.
///
/// # Ordering
///
/// Expressions have a total structural ordering used by `BTreeMap`
/// keys in [`AddNode`] and [`MulNode`]. Variant order:
/// Integer < Rational < Float < Complex < Constant < Symbol < Add < Mul < Pow < Func.
#[derive(Clone, Debug)]
pub enum Expr {
    /// Exact integer.
    Integer(SmallInt),
    /// Exact rational.
    Rational(BigRational),
    /// Floating-point approximation.
    Float(f64),
    /// Complex number (a + bi).
    Complex(Complex64),
    /// Symbolic constant (Pi, E, I).
    Constant(SymbolicConstant),
    /// Named variable (interned).
    Symbol(SymbolId),
    /// Sum: `constant + Σ(coeff · term)`.
    Add(AddNode),
    /// Product: `coeff · Π(base ^ exp)`.
    Mul(MulNode),
    /// Power: `base ^ exponent`.
    Pow(Arc<Expr>, Arc<Expr>),
    /// Function application: `f(arg1, arg2, ...)`.
    Func(FuncId, Vec<Arc<Expr>>),
}

/// Returns a numeric rank for deterministic variant ordering.
fn variant_rank(e: &Expr) -> u8 {
    match e {
        Expr::Integer(_) => 0,
        Expr::Rational(_) => 1,
        Expr::Float(_) => 2,
        Expr::Complex(_) => 3,
        Expr::Constant(_) => 4,
        Expr::Symbol(_) => 5,
        Expr::Add(_) => 6,
        Expr::Mul(_) => 7,
        Expr::Pow(_, _) => 8,
        Expr::Func(_, _) => 9,
    }
}

// ── Equality ─────────────────────────────────────────────────────────────────

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Expr::Integer(a), Expr::Integer(b)) => a == b,
            (Expr::Rational(a), Expr::Rational(b)) => a == b,
            (Expr::Float(a), Expr::Float(b)) => a.to_bits() == b.to_bits(),
            (Expr::Complex(a), Expr::Complex(b)) => {
                a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits()
            }
            (Expr::Constant(a), Expr::Constant(b)) => a == b,
            (Expr::Symbol(a), Expr::Symbol(b)) => a == b,
            (Expr::Add(a), Expr::Add(b)) => a == b,
            (Expr::Mul(a), Expr::Mul(b)) => a == b,
            (Expr::Pow(ab, ae), Expr::Pow(bb, be)) => ab == bb && ae == be,
            (Expr::Func(fa, aa), Expr::Func(fb, ab)) => fa == fb && aa == ab,
            _ => false,
        }
    }
}

impl Eq for Expr {}

// ── Hashing ──────────────────────────────────────────────────────────────────

impl Hash for Expr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        variant_rank(self).hash(state);
        match self {
            Expr::Integer(n) => n.hash(state),
            Expr::Rational(r) => r.hash(state),
            Expr::Float(f) => f.to_bits().hash(state),
            Expr::Complex(c) => {
                c.re.to_bits().hash(state);
                c.im.to_bits().hash(state);
            }
            Expr::Constant(c) => c.hash(state),
            Expr::Symbol(s) => s.hash(state),
            Expr::Add(a) => a.hash(state),
            Expr::Mul(m) => m.hash(state),
            Expr::Pow(b, e) => {
                b.hash(state);
                e.hash(state);
            }
            Expr::Func(id, args) => {
                id.hash(state);
                for arg in args {
                    arg.hash(state);
                }
            }
        }
    }
}

// ── Ordering ─────────────────────────────────────────────────────────────────

impl PartialOrd for Expr {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Expr {
    fn cmp(&self, other: &Self) -> Ordering {
        variant_rank(self)
            .cmp(&variant_rank(other))
            .then_with(|| match (self, other) {
                (Expr::Integer(a), Expr::Integer(b)) => a.cmp(b),
                (Expr::Rational(a), Expr::Rational(b)) => a.cmp(b),
                (Expr::Float(a), Expr::Float(b)) => a.total_cmp(b),
                (Expr::Complex(a), Expr::Complex(b)) => {
                    a.re.total_cmp(&b.re).then_with(|| a.im.total_cmp(&b.im))
                }
                (Expr::Constant(a), Expr::Constant(b)) => {
                    fn const_rank(c: &SymbolicConstant) -> u8 {
                        match c {
                            SymbolicConstant::Pi => 0,
                            SymbolicConstant::E => 1,
                            SymbolicConstant::I => 2,
                        }
                    }
                    const_rank(a).cmp(&const_rank(b))
                }
                (Expr::Symbol(a), Expr::Symbol(b)) => a.cmp(b),
                (Expr::Add(a), Expr::Add(b)) => a.cmp(b),
                (Expr::Mul(a), Expr::Mul(b)) => a.cmp(b),
                (Expr::Pow(ab, ae), Expr::Pow(bb, be)) => ab.cmp(bb).then_with(|| ae.cmp(be)),
                (Expr::Func(fa, aa), Expr::Func(fb, ab)) => {
                    fa.cmp(fb).then_with(|| aa.iter().cmp(ab.iter()))
                }
                // Unreachable: variant_rank equality implies same variant.
                _ => Ordering::Equal,
            })
    }
}

// ── Constructors ─────────────────────────────────────────────────────────────

impl Expr {
    /// Wrap in `Arc` (no pooling).
    pub fn arc(self) -> Arc<Expr> {
        Arc::new(self)
    }

    /// Integer expression wrapped in `Arc`.
    pub fn int(n: i64) -> Arc<Expr> {
        Arc::new(Expr::Integer(SmallInt::from(n)))
    }

    /// Rational expression wrapped in `Arc`.
    pub fn rational(n: i64, d: i64) -> Arc<Expr> {
        Arc::new(Expr::Rational(BigRational::from_i64(n, d)))
    }

    /// Float expression wrapped in `Arc`.
    pub fn float(v: f64) -> Arc<Expr> {
        Arc::new(Expr::Float(v))
    }

    /// Symbol expression wrapped in `Arc`.
    pub fn symbol(name: &str) -> Arc<Expr> {
        Arc::new(Expr::Symbol(SymbolId::intern(name)))
    }

    /// Power expression wrapped in `Arc`.
    pub fn pow(base: Arc<Expr>, exp: Arc<Expr>) -> Arc<Expr> {
        Arc::new(Expr::Pow(base, exp))
    }

    /// Function-application expression wrapped in `Arc`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use thales::numeric::expr::{Expr, FuncId};
    ///
    /// let x = Expr::symbol("x");
    /// let sin_x = Expr::func(FuncId::Sin, vec![x]);
    /// ```
    pub fn func(id: FuncId, args: Vec<Arc<Expr>>) -> Arc<Expr> {
        Arc::new(Expr::Func(id, args))
    }

    /// Complex number expression wrapped in `Arc`.
    pub fn complex(re: f64, im: f64) -> Arc<Expr> {
        Arc::new(Expr::Complex(Complex64::new(re, im)))
    }

    /// Symbolic constant expression wrapped in `Arc`.
    pub fn constant(c: SymbolicConstant) -> Arc<Expr> {
        Arc::new(Expr::Constant(c))
    }

    /// Pi (π) constant wrapped in `Arc`.
    pub fn pi() -> Arc<Expr> {
        Arc::new(Expr::Constant(SymbolicConstant::Pi))
    }

    /// Euler's number (e) constant wrapped in `Arc`.
    pub fn e() -> Arc<Expr> {
        Arc::new(Expr::Constant(SymbolicConstant::E))
    }

    /// Imaginary unit (i) constant wrapped in `Arc`.
    pub fn i_unit() -> Arc<Expr> {
        Arc::new(Expr::Constant(SymbolicConstant::I))
    }

    /// Returns `true` if this is a numeric zero.
    pub fn is_zero(&self) -> bool {
        match self {
            Expr::Integer(n) => n.is_zero(),
            Expr::Rational(r) => r.is_zero(),
            Expr::Float(f) => *f == 0.0,
            Expr::Complex(c) => c.re == 0.0 && c.im == 0.0,
            _ => false,
        }
    }

    /// Returns `true` if this is a numeric one.
    pub fn is_one(&self) -> bool {
        match self {
            Expr::Integer(n) => n.is_one(),
            Expr::Rational(r) => r.is_one(),
            Expr::Float(f) => *f == 1.0,
            Expr::Complex(c) => c.re == 1.0 && c.im == 0.0,
            _ => false,
        }
    }
}

use num::traits::{One, Zero};

// ── Display helpers ──────────────────────────────────────────────────────────

/// Returns `true` if this expression needs parentheses when used as the base
/// of a power expression (i.e. it is a sum or product).
pub(crate) fn needs_parens_as_pow_base(e: &Expr) -> bool {
    matches!(e, Expr::Add(_) | Expr::Mul(_))
}

/// Returns `true` if this expression needs parentheses when used as a factor
/// inside a product (i.e. it is a sum).
pub(crate) fn needs_parens_as_mul_factor(e: &Expr) -> bool {
    matches!(e, Expr::Add(_))
}

/// Write `expr` surrounded by parentheses if `wrap` is true, otherwise plain.
pub(crate) fn fmt_maybe_paren(expr: &Expr, wrap: bool, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if wrap {
        write!(f, "({expr})")
    } else {
        write!(f, "{expr}")
    }
}

// ── Display ──────────────────────────────────────────────────────────────────

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Integer(n) => write!(f, "{n}"),
            Expr::Rational(r) => write!(f, "{r}"),
            Expr::Float(v) => write!(f, "{v}"),
            Expr::Complex(c) => {
                if c.im == 0.0 {
                    write!(f, "{}", c.re)
                } else if c.re == 0.0 {
                    write!(f, "{}i", c.im)
                } else if c.im < 0.0 {
                    write!(f, "{}{}i", c.re, c.im)
                } else {
                    write!(f, "{}+{}i", c.re, c.im)
                }
            }
            Expr::Constant(c) => write!(f, "{c}"),
            Expr::Symbol(s) => write!(f, "{s}"),
            // Add and Mul render themselves; no outer parens at top level.
            Expr::Add(a) => write!(f, "{a}"),
            Expr::Mul(m) => write!(f, "{m}"),
            Expr::Pow(b, e) => {
                // Wrap base only if it is itself a sum or product.
                fmt_maybe_paren(b, needs_parens_as_pow_base(b), f)?;
                write!(f, "^")?;
                // Wrap exponent only if it contains operators.
                let wrap_exp = matches!(e.as_ref(), Expr::Add(_) | Expr::Mul(_) | Expr::Pow(_, _));
                fmt_maybe_paren(e, wrap_exp, f)
            }
            Expr::Func(id, args) => {
                write!(f, "{id}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
        }
    }
}

// ── Exponent arithmetic helpers ──────────────────────────────────────────────

/// Add two exponent expressions. Evaluates numeric cases exactly;
/// constructs an `Add` node for symbolic cases.
pub(crate) fn add_exponents(a: &Arc<Expr>, b: &Arc<Expr>) -> Arc<Expr> {
    match (a.as_ref(), b.as_ref()) {
        (Expr::Integer(x), Expr::Integer(y)) => Arc::new(Expr::Integer(x + y)),
        (Expr::Rational(x), Expr::Rational(y)) => Arc::new(Expr::Rational(x + y)),
        (Expr::Integer(x), Expr::Rational(y)) | (Expr::Rational(y), Expr::Integer(x)) => {
            Arc::new(Expr::Rational(&BigRational::from_integer(x.clone()) + y))
        }
        _ => {
            let mut node = AddNode::zero();
            node.add_term(a.clone(), BigRational::one());
            node.add_term(b.clone(), BigRational::one());
            Arc::new(Expr::Add(node))
        }
    }
}

/// Negate an exponent expression. Evaluates numeric cases exactly;
/// constructs a scaled term for symbolic cases.
pub(crate) fn negate_exponent(e: &Arc<Expr>) -> Arc<Expr> {
    match e.as_ref() {
        Expr::Integer(n) => Arc::new(Expr::Integer(-n)),
        Expr::Rational(r) => Arc::new(Expr::Rational(-r)),
        _ => {
            let mut node = MulNode::one();
            node.coeff = BigRational::from(-1i64);
            node.add_factor(e.clone(), Expr::int(1));
            Arc::new(Expr::Mul(node))
        }
    }
}

// ── ExprPool (hash-consing) ──────────────────────────────────────────────────

/// Expression pool for common sub-expression elimination.
///
/// Interning expressions through a pool ensures that structurally equal
/// sub-expressions share the same `Arc` allocation (pointer equality).
pub struct ExprPool {
    cache: HashMap<u64, Vec<Weak<Expr>>>,
}

impl ExprPool {
    /// Create a new empty pool.
    pub fn new() -> Self {
        ExprPool {
            cache: HashMap::new(),
        }
    }

    /// Intern an expression, returning a shared `Arc`.
    ///
    /// If a structurally equal expression already lives in the pool,
    /// returns a clone of the existing `Arc` (same allocation).
    pub fn intern(&mut self, expr: Expr) -> Arc<Expr> {
        let hash = compute_hash(&expr);

        if let Some(entries) = self.cache.get(&hash) {
            for weak in entries {
                if let Some(arc) = weak.upgrade() {
                    if *arc == expr {
                        return arc;
                    }
                }
            }
        }

        let arc = Arc::new(expr);
        self.cache
            .entry(hash)
            .or_default()
            .push(Arc::downgrade(&arc));
        arc
    }

    /// Remove expired (no longer referenced) entries.
    pub fn gc(&mut self) {
        self.cache.retain(|_, entries| {
            entries.retain(|w| w.upgrade().is_some());
            !entries.is_empty()
        });
    }

    /// Number of cached entry slots (including stale weak refs).
    pub fn len(&self) -> usize {
        self.cache.values().map(|v| v.len()).sum()
    }

    /// Whether the pool has no entries.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

impl Default for ExprPool {
    fn default() -> Self {
        Self::new()
    }
}

fn compute_hash(expr: &Expr) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    expr.hash(&mut h);
    h.finish()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_equality() {
        let a = Expr::Integer(SmallInt::from(42i64));
        let b = Expr::Integer(SmallInt::from(42i64));
        assert_eq!(a, b);
    }

    #[test]
    fn test_different_variants_not_equal() {
        let i = Expr::Integer(SmallInt::from(1i64));
        let r = Expr::Rational(BigRational::from(1i64));
        assert_ne!(i, r);
    }

    #[test]
    fn test_float_nan_equality() {
        let a = Expr::Float(f64::NAN);
        let b = Expr::Float(f64::NAN);
        // NaN bits are equal -> structural equality holds
        assert_eq!(a, b);
    }

    #[test]
    fn test_float_neg_zero() {
        let a = Expr::Float(0.0);
        let b = Expr::Float(-0.0);
        // 0.0 and -0.0 have different bits -> structurally different
        assert_ne!(a, b);
    }

    #[test]
    fn test_symbol_equality() {
        let a = Expr::Symbol(SymbolId::intern("x"));
        let b = Expr::Symbol(SymbolId::intern("x"));
        assert_eq!(a, b);
    }

    #[test]
    fn test_symbol_inequality() {
        let a = Expr::Symbol(SymbolId::intern("var_a"));
        let b = Expr::Symbol(SymbolId::intern("var_b"));
        assert_ne!(a, b);
    }

    #[test]
    fn test_ordering_across_variants() {
        let int = Expr::Integer(SmallInt::from(100i64));
        let sym = Expr::Symbol(SymbolId::intern("z"));
        // Integer (rank 0) < Symbol (rank 3)
        assert!(int < sym);
    }

    #[test]
    fn test_ordering_within_integers() {
        let a = Expr::Integer(SmallInt::from(3i64));
        let b = Expr::Integer(SmallInt::from(7i64));
        assert!(a < b);
    }

    #[test]
    fn test_ordering_within_symbols() {
        let a = Expr::Symbol(SymbolId::intern("ord_a"));
        let b = Expr::Symbol(SymbolId::intern("ord_b"));
        // SymbolId ordering is by intern order
        assert!(a < b);
    }

    #[test]
    fn test_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        let a = Expr::Integer(SmallInt::from(42i64));
        let b = Expr::Integer(SmallInt::from(42i64));

        let ha = {
            let mut h = DefaultHasher::new();
            a.hash(&mut h);
            h.finish()
        };
        let hb = {
            let mut h = DefaultHasher::new();
            b.hash(&mut h);
            h.finish()
        };
        assert_eq!(ha, hb);
    }

    #[test]
    fn test_pow_equality() {
        let x = Expr::symbol("pow_x");
        let two = Expr::int(2);
        let a = Expr::Pow(x.clone(), two.clone());
        let b = Expr::Pow(x, two);
        assert_eq!(a, b);
    }

    #[test]
    fn test_pow_ordering() {
        let x = Expr::symbol("pow_ord_x");
        let a = Expr::Pow(x.clone(), Expr::int(2));
        let b = Expr::Pow(x, Expr::int(3));
        assert!(a < b);
    }

    #[test]
    fn test_arc_clone_is_cheap() {
        let x = Expr::symbol("arc_cheap");
        let y = x.clone();
        assert!(Arc::ptr_eq(&x, &y));
    }

    #[test]
    fn test_arc_modification_creates_new() {
        let x = Expr::symbol("arc_mod");
        let y = x.clone();
        // Modifying through a new Arc doesn't affect original
        let z = Expr::int(99);
        assert!(!Arc::ptr_eq(&x, &z));
        assert!(Arc::ptr_eq(&x, &y));
    }

    #[test]
    fn test_display_integer() {
        assert_eq!(Expr::Integer(SmallInt::from(42i64)).to_string(), "42");
    }

    #[test]
    fn test_display_symbol() {
        assert_eq!(
            Expr::Symbol(SymbolId::intern("disp_x")).to_string(),
            "disp_x"
        );
    }

    #[test]
    fn test_display_pow() {
        let e = Expr::Pow(Expr::symbol("disp_pow_x"), Expr::int(2));
        assert_eq!(e.to_string(), "disp_pow_x^2");
    }

    #[test]
    fn test_is_zero() {
        assert!(Expr::Integer(SmallInt::from(0i64)).is_zero());
        assert!(Expr::Rational(BigRational::zero()).is_zero());
        assert!(Expr::Float(0.0).is_zero());
        assert!(!Expr::Integer(SmallInt::from(1i64)).is_zero());
        assert!(!Expr::Symbol(SymbolId::intern("zero_test")).is_zero());
    }

    #[test]
    fn test_is_one() {
        assert!(Expr::Integer(SmallInt::from(1i64)).is_one());
        assert!(Expr::Rational(BigRational::one()).is_one());
        assert!(Expr::Float(1.0).is_one());
        assert!(!Expr::Integer(SmallInt::from(0i64)).is_one());
    }

    // ── ExprPool tests ───────────────────────────────────────────────────────

    #[test]
    fn test_pool_intern_same_returns_same_arc() {
        let mut pool = ExprPool::new();
        let x1 = pool.intern(Expr::Symbol(SymbolId::intern("pool_x")));
        let x2 = pool.intern(Expr::Symbol(SymbolId::intern("pool_x")));
        assert!(Arc::ptr_eq(&x1, &x2));
    }

    #[test]
    fn test_pool_different_exprs_different_arcs() {
        let mut pool = ExprPool::new();
        let x = pool.intern(Expr::Symbol(SymbolId::intern("pool_diff_x")));
        let y = pool.intern(Expr::Symbol(SymbolId::intern("pool_diff_y")));
        assert!(!Arc::ptr_eq(&x, &y));
    }

    #[test]
    fn test_pool_integers() {
        let mut pool = ExprPool::new();
        let a = pool.intern(Expr::Integer(SmallInt::from(7i64)));
        let b = pool.intern(Expr::Integer(SmallInt::from(7i64)));
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn test_pool_gc_removes_stale() {
        let mut pool = ExprPool::new();
        {
            let _x = pool.intern(Expr::Symbol(SymbolId::intern("pool_gc_tmp")));
            assert_eq!(pool.len(), 1);
            // _x dropped here
        }
        pool.gc();
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn test_pool_gc_keeps_live() {
        let mut pool = ExprPool::new();
        let x = pool.intern(Expr::Symbol(SymbolId::intern("pool_gc_live")));
        pool.gc();
        assert_eq!(pool.len(), 1);
        drop(x);
    }

    #[test]
    fn test_pool_complex_expression() {
        let mut pool = ExprPool::new();
        let x = pool.intern(Expr::Symbol(SymbolId::intern("pool_cx")));
        let two = pool.intern(Expr::Integer(SmallInt::from(2i64)));
        let pow1 = pool.intern(Expr::Pow(x.clone(), two.clone()));
        let pow2 = pool.intern(Expr::Pow(x, two));
        assert!(Arc::ptr_eq(&pow1, &pow2));
    }

    // ── AddNode in Expr tests ────────────────────────────────────────────────

    #[test]
    fn test_add_node_in_expr() {
        let x = Expr::symbol("add_in_x");
        let mut node = AddNode::zero();
        node.add_term(x, BigRational::from(2i64));
        node.add_constant(BigRational::from(3i64));

        let e1 = Expr::Add(node.clone());
        let e2 = Expr::Add(node);
        assert_eq!(e1, e2);
    }

    // ── MulNode in Expr tests ────────────────────────────────────────────────

    #[test]
    fn test_mul_node_in_expr() {
        let x = Expr::symbol("mul_in_x");
        let mut node = MulNode::one();
        node.add_factor(x, Expr::int(2));

        let e1 = Expr::Mul(node.clone());
        let e2 = Expr::Mul(node);
        assert_eq!(e1, e2);
    }

    // ── Exponent helpers ─────────────────────────────────────────────────────

    #[test]
    fn test_add_integer_exponents() {
        let a = Expr::int(2);
        let b = Expr::int(3);
        let result = add_exponents(&a, &b);
        assert_eq!(*result, Expr::Integer(SmallInt::from(5i64)));
    }

    #[test]
    fn test_add_rational_exponents() {
        let a = Expr::rational(1, 2);
        let b = Expr::rational(1, 3);
        let result = add_exponents(&a, &b);
        assert_eq!(*result, Expr::Rational(BigRational::from_i64(5, 6)));
    }

    #[test]
    fn test_negate_integer_exponent() {
        let e = Expr::int(3);
        let result = negate_exponent(&e);
        assert_eq!(*result, Expr::Integer(SmallInt::from(-3i64)));
    }

    #[test]
    fn test_negate_rational_exponent() {
        let e = Expr::rational(1, 2);
        let result = negate_exponent(&e);
        assert_eq!(*result, Expr::Rational(BigRational::from_i64(-1, 2)));
    }

    // ── FuncId tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_func_id_equality() {
        assert_eq!(FuncId::Sin, FuncId::Sin);
        assert_ne!(FuncId::Sin, FuncId::Cos);
    }

    #[test]
    fn test_func_id_ordering() {
        assert!(FuncId::Sin < FuncId::Cos);
        assert!(FuncId::Cos < FuncId::Tan);
        assert!(FuncId::Tan < FuncId::Ln);
    }

    #[test]
    fn test_func_id_new_variants_equality() {
        assert_eq!(FuncId::Asin, FuncId::Asin);
        assert_ne!(FuncId::Asin, FuncId::Acos);
        assert_eq!(FuncId::Sinh, FuncId::Sinh);
        assert_ne!(FuncId::Sinh, FuncId::Cosh);
        assert_eq!(FuncId::Log, FuncId::Log);
        assert_eq!(FuncId::Log2, FuncId::Log2);
        assert_eq!(FuncId::Log10, FuncId::Log10);
        assert_eq!(FuncId::Cbrt, FuncId::Cbrt);
        assert_eq!(FuncId::Floor, FuncId::Floor);
        assert_eq!(FuncId::Ceil, FuncId::Ceil);
        assert_eq!(FuncId::Round, FuncId::Round);
        assert_eq!(FuncId::Sign, FuncId::Sign);
        assert_eq!(FuncId::Min, FuncId::Min);
        assert_eq!(FuncId::Max, FuncId::Max);
    }

    #[test]
    fn test_func_id_full_ordering() {
        // Verify the complete rank ordering is monotonically increasing
        assert!(FuncId::Sin < FuncId::Cos);
        assert!(FuncId::Cos < FuncId::Tan);
        assert!(FuncId::Tan < FuncId::Asin);
        assert!(FuncId::Asin < FuncId::Acos);
        assert!(FuncId::Acos < FuncId::Atan);
        assert!(FuncId::Atan < FuncId::Atan2);
        assert!(FuncId::Atan2 < FuncId::Sinh);
        assert!(FuncId::Sinh < FuncId::Cosh);
        assert!(FuncId::Cosh < FuncId::Tanh);
        assert!(FuncId::Tanh < FuncId::Ln);
        assert!(FuncId::Ln < FuncId::Exp);
        assert!(FuncId::Exp < FuncId::Log);
        assert!(FuncId::Log < FuncId::Log2);
        assert!(FuncId::Log2 < FuncId::Log10);
        assert!(FuncId::Log10 < FuncId::Sqrt);
        assert!(FuncId::Sqrt < FuncId::Cbrt);
        assert!(FuncId::Cbrt < FuncId::Floor);
        assert!(FuncId::Floor < FuncId::Ceil);
        assert!(FuncId::Ceil < FuncId::Round);
        assert!(FuncId::Round < FuncId::Abs);
        assert!(FuncId::Abs < FuncId::Sign);
        assert!(FuncId::Sign < FuncId::Min);
        assert!(FuncId::Min < FuncId::Max);
        assert!(FuncId::Max < FuncId::Other(SymbolId::intern("ord_z")));
    }

    #[test]
    fn test_func_id_display() {
        assert_eq!(FuncId::Sin.to_string(), "sin");
        assert_eq!(FuncId::Cos.to_string(), "cos");
        assert_eq!(FuncId::Tan.to_string(), "tan");
        assert_eq!(FuncId::Ln.to_string(), "ln");
        assert_eq!(FuncId::Exp.to_string(), "exp");
        assert_eq!(FuncId::Sqrt.to_string(), "sqrt");
        assert_eq!(FuncId::Abs.to_string(), "abs");
    }

    #[test]
    fn test_func_id_new_variants_display() {
        assert_eq!(FuncId::Asin.to_string(), "asin");
        assert_eq!(FuncId::Acos.to_string(), "acos");
        assert_eq!(FuncId::Atan.to_string(), "atan");
        assert_eq!(FuncId::Atan2.to_string(), "atan2");
        assert_eq!(FuncId::Sinh.to_string(), "sinh");
        assert_eq!(FuncId::Cosh.to_string(), "cosh");
        assert_eq!(FuncId::Tanh.to_string(), "tanh");
        assert_eq!(FuncId::Log.to_string(), "log");
        assert_eq!(FuncId::Log2.to_string(), "log2");
        assert_eq!(FuncId::Log10.to_string(), "log10");
        assert_eq!(FuncId::Cbrt.to_string(), "cbrt");
        assert_eq!(FuncId::Floor.to_string(), "floor");
        assert_eq!(FuncId::Ceil.to_string(), "ceil");
        assert_eq!(FuncId::Round.to_string(), "round");
        assert_eq!(FuncId::Sign.to_string(), "sign");
        assert_eq!(FuncId::Min.to_string(), "min");
        assert_eq!(FuncId::Max.to_string(), "max");
    }

    #[test]
    fn test_func_id_other_display() {
        let id = FuncId::Other(SymbolId::intern("my_func"));
        assert_eq!(id.to_string(), "my_func");
    }

    #[test]
    fn test_func_constructor() {
        let x = Expr::symbol("fx");
        let sin_x = Expr::func(FuncId::Sin, vec![x.clone()]);
        match sin_x.as_ref() {
            Expr::Func(FuncId::Sin, args) => {
                assert_eq!(args.len(), 1);
                assert_eq!(*args[0], *x);
            }
            _ => panic!("expected Func(Sin, ...)"),
        }
    }

    #[test]
    fn test_func_display() {
        let x = Expr::symbol("disp_fx");
        let sin_x = Expr::func(FuncId::Sin, vec![x]);
        assert_eq!(sin_x.to_string(), "sin(disp_fx)");
    }

    #[test]
    fn test_func_display_two_args() {
        let x = Expr::symbol("disp_fx2");
        let y = Expr::symbol("disp_fy2");
        let f = Expr::func(FuncId::Other(SymbolId::intern("pow2")), vec![x, y]);
        assert_eq!(f.to_string(), "pow2(disp_fx2, disp_fy2)");
    }

    #[test]
    fn test_func_equality() {
        let x = Expr::symbol("feq_x");
        let a = Expr::func(FuncId::Sin, vec![x.clone()]);
        let b = Expr::func(FuncId::Sin, vec![x.clone()]);
        let c = Expr::func(FuncId::Cos, vec![x]);
        assert_eq!(*a, *b);
        assert_ne!(*a, *c);
    }

    #[test]
    fn test_func_ordering_by_id() {
        let x = Expr::symbol("ford_x");
        let sin_x = Expr::Func(FuncId::Sin, vec![x.clone()]);
        let cos_x = Expr::Func(FuncId::Cos, vec![x]);
        // Sin (rank 0) < Cos (rank 1)
        assert!(sin_x < cos_x);
    }

    #[test]
    fn test_func_ordering_above_pow() {
        let x = Expr::symbol("ford2_x");
        let pow_x = Expr::Pow(x.clone(), Expr::int(2));
        let sin_x = Expr::Func(FuncId::Sin, vec![x]);
        // Pow (rank 6) < Func (rank 7)
        assert!(pow_x < sin_x);
    }

    #[test]
    fn test_func_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        let x = Expr::symbol("fhash_x");
        let a = Expr::Func(FuncId::Sin, vec![x.clone()]);
        let b = Expr::Func(FuncId::Sin, vec![x]);
        let ha = {
            let mut h = DefaultHasher::new();
            a.hash(&mut h);
            h.finish()
        };
        let hb = {
            let mut h = DefaultHasher::new();
            b.hash(&mut h);
            h.finish()
        };
        assert_eq!(ha, hb);
    }

    #[test]
    fn test_func_is_not_zero_or_one() {
        let x = Expr::symbol("fnz_x");
        let f = Expr::Func(FuncId::Sin, vec![x]);
        assert!(!f.is_zero());
        assert!(!f.is_one());
    }

    // ── Complex tests ────────────────────────────────────────────────────────

    #[test]
    fn test_complex_equality() {
        let a = Expr::Complex(Complex64::new(1.0, 2.0));
        let b = Expr::Complex(Complex64::new(1.0, 2.0));
        assert_eq!(a, b);
    }

    #[test]
    fn test_complex_inequality() {
        let a = Expr::Complex(Complex64::new(1.0, 2.0));
        let b = Expr::Complex(Complex64::new(1.0, 3.0));
        assert_ne!(a, b);
    }

    #[test]
    fn test_complex_nan_equality() {
        let a = Expr::Complex(Complex64::new(f64::NAN, 0.0));
        let b = Expr::Complex(Complex64::new(f64::NAN, 0.0));
        // NaN bits are identical -> structural equality holds
        assert_eq!(a, b);
    }

    #[test]
    fn test_complex_neg_zero() {
        let a = Expr::Complex(Complex64::new(0.0, 0.0));
        let b = Expr::Complex(Complex64::new(-0.0, 0.0));
        // 0.0 and -0.0 have different bits -> structurally different
        assert_ne!(a, b);
    }

    #[test]
    fn test_complex_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        let a = Expr::Complex(Complex64::new(3.0, 4.0));
        let b = Expr::Complex(Complex64::new(3.0, 4.0));
        let ha = {
            let mut h = DefaultHasher::new();
            a.hash(&mut h);
            h.finish()
        };
        let hb = {
            let mut h = DefaultHasher::new();
            b.hash(&mut h);
            h.finish()
        };
        assert_eq!(ha, hb);
    }

    #[test]
    fn test_complex_ordering_by_real() {
        let a = Expr::Complex(Complex64::new(1.0, 5.0));
        let b = Expr::Complex(Complex64::new(2.0, 0.0));
        assert!(a < b);
    }

    #[test]
    fn test_complex_ordering_by_imag_when_real_equal() {
        let a = Expr::Complex(Complex64::new(1.0, 2.0));
        let b = Expr::Complex(Complex64::new(1.0, 3.0));
        assert!(a < b);
    }

    #[test]
    fn test_complex_display_full() {
        let e = Expr::Complex(Complex64::new(3.0, 4.0));
        assert_eq!(e.to_string(), "3+4i");
    }

    #[test]
    fn test_complex_display_negative_imag() {
        let e = Expr::Complex(Complex64::new(3.0, -4.0));
        assert_eq!(e.to_string(), "3-4i");
    }

    #[test]
    fn test_complex_display_pure_real() {
        let e = Expr::Complex(Complex64::new(5.0, 0.0));
        assert_eq!(e.to_string(), "5");
    }

    #[test]
    fn test_complex_display_pure_imaginary() {
        let e = Expr::Complex(Complex64::new(0.0, 2.0));
        assert_eq!(e.to_string(), "2i");
    }

    #[test]
    fn test_complex_is_zero() {
        assert!(Expr::Complex(Complex64::new(0.0, 0.0)).is_zero());
        assert!(!Expr::Complex(Complex64::new(0.0, 1.0)).is_zero());
        assert!(!Expr::Complex(Complex64::new(1.0, 0.0)).is_zero());
    }

    #[test]
    fn test_complex_is_one() {
        assert!(Expr::Complex(Complex64::new(1.0, 0.0)).is_one());
        assert!(!Expr::Complex(Complex64::new(1.0, 1.0)).is_one());
        assert!(!Expr::Complex(Complex64::new(0.0, 0.0)).is_one());
    }

    #[test]
    fn test_complex_constructor() {
        let c = Expr::complex(3.0, -1.0);
        match c.as_ref() {
            Expr::Complex(v) => {
                assert_eq!(v.re, 3.0);
                assert_eq!(v.im, -1.0);
            }
            _ => panic!("expected Complex"),
        }
    }

    // ── Constant tests ───────────────────────────────────────────────────────

    #[test]
    fn test_constant_equality() {
        let a = Expr::Constant(SymbolicConstant::Pi);
        let b = Expr::Constant(SymbolicConstant::Pi);
        assert_eq!(a, b);
        let c = Expr::Constant(SymbolicConstant::E);
        assert_ne!(a, c);
    }

    #[test]
    fn test_constant_ordering() {
        let pi = Expr::Constant(SymbolicConstant::Pi);
        let e = Expr::Constant(SymbolicConstant::E);
        let i = Expr::Constant(SymbolicConstant::I);
        // Pi (0) < E (1) < I (2)
        assert!(pi < e);
        assert!(e < i);
        assert!(pi < i);
    }

    #[test]
    fn test_constant_display() {
        assert_eq!(Expr::Constant(SymbolicConstant::Pi).to_string(), "π");
        assert_eq!(Expr::Constant(SymbolicConstant::E).to_string(), "e");
        assert_eq!(Expr::Constant(SymbolicConstant::I).to_string(), "i");
    }

    #[test]
    fn test_constant_constructors() {
        match Expr::pi().as_ref() {
            Expr::Constant(SymbolicConstant::Pi) => {}
            _ => panic!("expected Constant(Pi)"),
        }
        match Expr::e().as_ref() {
            Expr::Constant(SymbolicConstant::E) => {}
            _ => panic!("expected Constant(E)"),
        }
        match Expr::i_unit().as_ref() {
            Expr::Constant(SymbolicConstant::I) => {}
            _ => panic!("expected Constant(I)"),
        }
    }

    #[test]
    fn test_constant_is_not_zero_or_one() {
        assert!(!Expr::pi().is_zero());
        assert!(!Expr::pi().is_one());
        assert!(!Expr::e().is_zero());
        assert!(!Expr::e().is_one());
    }

    // ── ExprPool interning for new variants ──────────────────────────────────

    #[test]
    fn test_pool_complex_interning() {
        let mut pool = ExprPool::new();
        let a = pool.intern(Expr::Complex(Complex64::new(1.0, 2.0)));
        let b = pool.intern(Expr::Complex(Complex64::new(1.0, 2.0)));
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn test_pool_constant_interning() {
        let mut pool = ExprPool::new();
        let a = pool.intern(Expr::Constant(SymbolicConstant::Pi));
        let b = pool.intern(Expr::Constant(SymbolicConstant::Pi));
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn test_pool_different_constants_different_arcs() {
        let mut pool = ExprPool::new();
        let pi = pool.intern(Expr::Constant(SymbolicConstant::Pi));
        let e = pool.intern(Expr::Constant(SymbolicConstant::E));
        assert!(!Arc::ptr_eq(&pi, &e));
    }

    // ── Cross-variant ordering: Float < Complex < Constant < Symbol ──────────

    #[test]
    fn test_cross_variant_float_lt_complex() {
        let f = Expr::Float(999.0);
        let c = Expr::Complex(Complex64::new(0.0, 0.0));
        assert!(f < c);
    }

    #[test]
    fn test_cross_variant_complex_lt_constant() {
        let c = Expr::Complex(Complex64::new(100.0, 100.0));
        let k = Expr::Constant(SymbolicConstant::Pi);
        assert!(c < k);
    }

    #[test]
    fn test_cross_variant_constant_lt_symbol() {
        let k = Expr::Constant(SymbolicConstant::I);
        let s = Expr::Symbol(SymbolId::intern("cross_sym"));
        assert!(k < s);
    }

    // ── Display improvement tests ────────────────────────────────────────────

    /// `x^2` — no unnecessary parens on symbol base or integer exponent.
    #[test]
    fn test_display_pow_symbol_int_exp() {
        let e = Expr::Pow(Expr::symbol("disp_imp_x"), Expr::int(2));
        assert_eq!(e.to_string(), "disp_imp_x^2");
    }

    /// `(x + y)^2` — Add base must be wrapped in parens.
    #[test]
    fn test_display_pow_add_base_needs_parens() {
        let x = Expr::symbol("disp_add_base_x");
        let y = Expr::symbol("disp_add_base_y");
        let mut node = AddNode::zero();
        node.add_term(x, BigRational::from(1i64));
        node.add_term(y, BigRational::from(1i64));
        let sum = Arc::new(Expr::Add(node));
        let e = Expr::Pow(sum, Expr::int(2));
        let s = e.to_string();
        assert!(s.starts_with('('), "expected leading '(' for Add base: {s}");
        assert!(s.contains(")^2"), "expected ')^2' after Add base: {s}");
    }

    /// `x + y` — no outer parens when Add is the top-level expression.
    #[test]
    fn test_display_add_no_outer_parens() {
        let x = Expr::symbol("disp_top_x");
        let y = Expr::symbol("disp_top_y");
        let mut node = AddNode::zero();
        node.add_term(x, BigRational::from(1i64));
        node.add_term(y, BigRational::from(1i64));
        let s = Expr::Add(node).to_string();
        assert!(!s.starts_with('('), "unexpected outer parens: {s}");
    }

    /// `x - y` — AddNode with negative coeff uses ` - ` not ` + (-1)*`.
    #[test]
    fn test_display_add_subtraction() {
        let x = Expr::symbol("disp_sub_x");
        let y = Expr::symbol("disp_sub_y");
        let mut node = AddNode::zero();
        node.add_term(x, BigRational::from(1i64));
        node.add_term(y, BigRational::from(-1i64));
        let s = Expr::Add(node).to_string();
        assert!(s.contains(" - "), "expected ' - ' for subtraction: {s}");
        assert!(!s.contains("(-1)"), "should not show '(-1)' coeff: {s}");
    }

    /// `2x` — MulNode with coeff 2, no `*` between coeff and variable.
    #[test]
    fn test_display_mul_coeff_no_star() {
        let x = Expr::symbol("x");
        let mut node = MulNode::one();
        node.coeff = BigRational::from(2i64);
        node.add_factor(x, Expr::int(1));
        let s = Expr::Mul(node).to_string();
        assert_eq!(s, "2x");
    }

    /// `-x` — MulNode with coeff -1 shows `-` not `-1*`.
    #[test]
    fn test_display_mul_neg_one_coeff() {
        let x = Expr::symbol("x");
        let mut node = MulNode::one();
        node.coeff = BigRational::from(-1i64);
        node.add_factor(x, Expr::int(1));
        let s = Expr::Mul(node).to_string();
        assert_eq!(s, "-x");
    }

    /// `x/y` — MulNode with y^(-1) renders as `x/y`.
    #[test]
    fn test_display_mul_division() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let mut node = MulNode::one();
        node.add_factor(x, Expr::int(1));
        node.add_factor(y, Expr::int(-1));
        let s = Expr::Mul(node).to_string();
        assert_eq!(s, "x/y");
    }
}
