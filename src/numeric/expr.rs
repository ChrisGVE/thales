//! Arc-based expression type with structural sharing.
//!
//! [`Expr`] is the new expression representation for the CAS. All compound
//! expressions hold children via `Arc<Expr>`, enabling O(1) clone and
//! structural sharing of common sub-expressions.
//!
//! Use [`ExprPool`] for automatic common sub-expression elimination
//! during construction: identical sub-expressions return the same `Arc`.

use super::{AddNode, BigRational, MulNode, SmallInt, SymbolId};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Weak};

/// Core expression type for the computer algebra system.
///
/// All compound expressions hold children via `Arc<Expr>`, enabling
/// O(1) clone and structural sharing.
///
/// # Ordering
///
/// Expressions have a total structural ordering used by `BTreeMap`
/// keys in [`AddNode`] and [`MulNode`]. Variant order:
/// Integer < Rational < Float < Symbol < Add < Mul < Pow.
#[derive(Clone, Debug)]
pub enum Expr {
    /// Exact integer.
    Integer(SmallInt),
    /// Exact rational.
    Rational(BigRational),
    /// Floating-point approximation.
    Float(f64),
    /// Named variable (interned).
    Symbol(SymbolId),
    /// Sum: `constant + Σ(coeff · term)`.
    Add(AddNode),
    /// Product: `coeff · Π(base ^ exp)`.
    Mul(MulNode),
    /// Power: `base ^ exponent`.
    Pow(Arc<Expr>, Arc<Expr>),
}

/// Returns a numeric rank for deterministic variant ordering.
fn variant_rank(e: &Expr) -> u8 {
    match e {
        Expr::Integer(_) => 0,
        Expr::Rational(_) => 1,
        Expr::Float(_) => 2,
        Expr::Symbol(_) => 3,
        Expr::Add(_) => 4,
        Expr::Mul(_) => 5,
        Expr::Pow(_, _) => 6,
    }
}

// ── Equality ─────────────────────────────────────────────────────────────────

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Expr::Integer(a), Expr::Integer(b)) => a == b,
            (Expr::Rational(a), Expr::Rational(b)) => a == b,
            (Expr::Float(a), Expr::Float(b)) => a.to_bits() == b.to_bits(),
            (Expr::Symbol(a), Expr::Symbol(b)) => a == b,
            (Expr::Add(a), Expr::Add(b)) => a == b,
            (Expr::Mul(a), Expr::Mul(b)) => a == b,
            (Expr::Pow(ab, ae), Expr::Pow(bb, be)) => ab == bb && ae == be,
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
            Expr::Symbol(s) => s.hash(state),
            Expr::Add(a) => a.hash(state),
            Expr::Mul(m) => m.hash(state),
            Expr::Pow(b, e) => {
                b.hash(state);
                e.hash(state);
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
                (Expr::Symbol(a), Expr::Symbol(b)) => a.cmp(b),
                (Expr::Add(a), Expr::Add(b)) => a.cmp(b),
                (Expr::Mul(a), Expr::Mul(b)) => a.cmp(b),
                (Expr::Pow(ab, ae), Expr::Pow(bb, be)) => ab.cmp(bb).then_with(|| ae.cmp(be)),
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

    /// Returns `true` if this is a numeric zero.
    pub fn is_zero(&self) -> bool {
        match self {
            Expr::Integer(n) => n.is_zero(),
            Expr::Rational(r) => r.is_zero(),
            Expr::Float(f) => *f == 0.0,
            _ => false,
        }
    }

    /// Returns `true` if this is a numeric one.
    pub fn is_one(&self) -> bool {
        match self {
            Expr::Integer(n) => n.is_one(),
            Expr::Rational(r) => r.is_one(),
            Expr::Float(f) => *f == 1.0,
            _ => false,
        }
    }
}

use num::traits::{One, Zero};

// ── Display ──────────────────────────────────────────────────────────────────

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Integer(n) => write!(f, "{n}"),
            Expr::Rational(r) => write!(f, "{r}"),
            Expr::Float(v) => write!(f, "{v}"),
            Expr::Symbol(s) => write!(f, "{s}"),
            Expr::Add(a) => write!(f, "({a})"),
            Expr::Mul(m) => write!(f, "({m})"),
            Expr::Pow(b, e) => write!(f, "({b})^({e})"),
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
        // NaN bits are equal → structural equality holds
        assert_eq!(a, b);
    }

    #[test]
    fn test_float_neg_zero() {
        let a = Expr::Float(0.0);
        let b = Expr::Float(-0.0);
        // 0.0 and -0.0 have different bits → structurally different
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
        assert_eq!(e.to_string(), "(disp_pow_x)^(2)");
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
}
