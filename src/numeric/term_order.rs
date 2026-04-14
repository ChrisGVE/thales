//! Monomial term orderings for polynomial canonical form.
//!
//! Defines configurable orderings used by multivariate polynomial operations.
//! Default is graded reverse lexicographic (grevlex). Others available for
//! specialized algorithms like Groebner basis computation.

use super::multivariate_poly::Monomial;
use super::SymbolId;
use std::cmp::Ordering;

/// A monomial ordering strategy.
///
/// Implementations define a total order on monomials. The ordering determines
/// the canonical form of polynomials and affects algorithm behavior
/// (especially Groebner basis computation).
pub trait MonomialOrder: Clone + std::fmt::Debug {
    /// Compare two monomials under this ordering.
    fn cmp_monomials(&self, a: &Monomial, b: &Monomial) -> Ordering;
}

/// Graded reverse lexicographic ordering (grevlex).
///
/// This is the default ordering. Monomials are compared by:
/// 1. Total degree (higher degree > lower degree)
/// 2. For equal total degree: compare exponents of variables in reverse
///    order. The monomial with a smaller exponent on the last differing
///    variable is greater.
///
/// This ordering is efficient for Groebner basis computation.
#[derive(Clone, Debug, Default)]
pub struct GrevLex {
    /// Variable ordering (if empty, uses natural SymbolId order).
    var_order: Vec<SymbolId>,
}

impl GrevLex {
    /// Create grevlex with default (SymbolId) variable ordering.
    pub fn new() -> Self {
        GrevLex {
            var_order: Vec::new(),
        }
    }

    /// Create grevlex with a specific variable ordering.
    ///
    /// Variables listed first are considered "smaller" (innermost in grevlex).
    pub fn with_var_order(vars: Vec<SymbolId>) -> Self {
        GrevLex { var_order: vars }
    }

    /// Get ordered variables for comparison.
    fn ordered_vars(&self, a: &Monomial, b: &Monomial) -> Vec<SymbolId> {
        if !self.var_order.is_empty() {
            return self.var_order.clone();
        }
        // Collect all variables from both monomials
        let mut vars: Vec<_> = a.iter().chain(b.iter()).map(|(id, _)| *id).collect();
        vars.sort();
        vars.dedup();
        vars
    }
}

impl MonomialOrder for GrevLex {
    fn cmp_monomials(&self, a: &Monomial, b: &Monomial) -> Ordering {
        match a.total_degree().cmp(&b.total_degree()) {
            Ordering::Equal => {
                let vars = self.ordered_vars(a, b);
                // Reverse iterate for grevlex
                for &v in vars.iter().rev() {
                    let ea = a.exponent(&v);
                    let eb = b.exponent(&v);
                    match ea.cmp(&eb) {
                        Ordering::Equal => continue,
                        Ordering::Less => return Ordering::Greater,
                        Ordering::Greater => return Ordering::Less,
                    }
                }
                Ordering::Equal
            }
            ord => ord,
        }
    }
}

/// Pure lexicographic ordering (lex).
///
/// Monomials are compared by the exponent of the first variable, then
/// the second, and so on. Used for elimination orderings in Groebner
/// basis computation.
#[derive(Clone, Debug)]
pub struct Lex {
    /// Variable ordering (required for lex).
    var_order: Vec<SymbolId>,
}

impl Lex {
    /// Create lex ordering with specified variable order.
    ///
    /// Variables listed first have highest priority.
    pub fn new(vars: Vec<SymbolId>) -> Self {
        Lex { var_order: vars }
    }
}

impl MonomialOrder for Lex {
    fn cmp_monomials(&self, a: &Monomial, b: &Monomial) -> Ordering {
        for &v in &self.var_order {
            let ea = a.exponent(&v);
            let eb = b.exponent(&v);
            match ea.cmp(&eb) {
                Ordering::Equal => continue,
                ord => return ord,
            }
        }
        Ordering::Equal
    }
}

/// Graded lexicographic ordering (deglex).
///
/// First by total degree, then lexicographic.
#[derive(Clone, Debug)]
pub struct DegLex {
    /// Variable ordering.
    var_order: Vec<SymbolId>,
}

impl DegLex {
    /// Create deglex ordering with specified variable order.
    pub fn new(vars: Vec<SymbolId>) -> Self {
        DegLex { var_order: vars }
    }
}

impl MonomialOrder for DegLex {
    fn cmp_monomials(&self, a: &Monomial, b: &Monomial) -> Ordering {
        match a.total_degree().cmp(&b.total_degree()) {
            Ordering::Equal => {
                for &v in &self.var_order {
                    let ea = a.exponent(&v);
                    let eb = b.exponent(&v);
                    match ea.cmp(&eb) {
                        Ordering::Equal => continue,
                        ord => return ord,
                    }
                }
                Ordering::Equal
            }
            ord => ord,
        }
    }
}

/// Wrapper for comparing monomials with a specific ordering.
///
/// Useful for BTreeMap keys when you need a non-default ordering.
#[derive(Clone, Debug)]
pub struct OrderedMonomial<O: MonomialOrder> {
    /// The underlying monomial.
    pub monomial: Monomial,
    /// The ordering to use.
    order: O,
}

impl<O: MonomialOrder> OrderedMonomial<O> {
    /// Wrap a monomial with an ordering.
    pub fn new(monomial: Monomial, order: O) -> Self {
        OrderedMonomial { monomial, order }
    }
}

impl<O: MonomialOrder> PartialEq for OrderedMonomial<O> {
    fn eq(&self, other: &Self) -> bool {
        self.monomial == other.monomial
    }
}

impl<O: MonomialOrder> Eq for OrderedMonomial<O> {}

impl<O: MonomialOrder> PartialOrd for OrderedMonomial<O> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<O: MonomialOrder> Ord for OrderedMonomial<O> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.order.cmp_monomials(&self.monomial, &other.monomial)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn x() -> SymbolId {
        SymbolId::intern("to_x")
    }

    fn y() -> SymbolId {
        SymbolId::intern("to_y")
    }

    fn z() -> SymbolId {
        SymbolId::intern("to_z")
    }

    #[test]
    fn test_grevlex_total_degree() {
        let order = GrevLex::new();
        let x2 = Monomial::var_pow(x(), 2);
        let xy = Monomial::var(x()).mul(&Monomial::var(y()));
        let xm = Monomial::var(x());

        // Degree 2 > degree 1
        assert_eq!(order.cmp_monomials(&x2, &xm), Ordering::Greater);
        assert_eq!(order.cmp_monomials(&xy, &xm), Ordering::Greater);
    }

    #[test]
    fn test_grevlex_same_degree() {
        let order = GrevLex::with_var_order(vec![x(), y()]);
        // x^2 vs xy vs y^2 — all degree 2
        let x2 = Monomial::var_pow(x(), 2);
        let xy = Monomial::var(x()).mul(&Monomial::var(y()));
        let y2 = Monomial::var_pow(y(), 2);

        // In grevlex with var_order [x, y]:
        // Last var is y. x^2 has y^0, xy has y^1, y^2 has y^2
        // Smaller exponent on last var = greater → x^2 > xy > y^2
        assert_eq!(order.cmp_monomials(&x2, &xy), Ordering::Greater);
        assert_eq!(order.cmp_monomials(&xy, &y2), Ordering::Greater);
        assert_eq!(order.cmp_monomials(&x2, &y2), Ordering::Greater);
    }

    #[test]
    fn test_grevlex_expected_chain() {
        let order = GrevLex::with_var_order(vec![x(), y()]);
        // x^2 > xy > y^2 > x > y > 1
        let monomials = vec![
            Monomial::var_pow(x(), 2),
            Monomial::var(x()).mul(&Monomial::var(y())),
            Monomial::var_pow(y(), 2),
            Monomial::var(x()),
            Monomial::var(y()),
            Monomial::one(),
        ];

        for i in 0..monomials.len() {
            for j in (i + 1)..monomials.len() {
                assert_eq!(
                    order.cmp_monomials(&monomials[i], &monomials[j]),
                    Ordering::Greater,
                    "{} should be > {}",
                    monomials[i],
                    monomials[j]
                );
            }
        }
    }

    #[test]
    fn test_lex_ordering() {
        let order = Lex::new(vec![x(), y(), z()]);

        // In lex: x > y > z, x^2 > x*y > x*z > y^2 > ...
        let x2 = Monomial::var_pow(x(), 2);
        let xy = Monomial::var(x()).mul(&Monomial::var(y()));
        let y2 = Monomial::var_pow(y(), 2);
        let xm = Monomial::var(x());

        // x^2 > xy (same first exponent is 2 vs 1? No: x^2 has x-exp=2, xy has x-exp=1)
        assert_eq!(order.cmp_monomials(&x2, &xy), Ordering::Greater);
        // xy > y^2 (x-exp: 1 > 0)
        assert_eq!(order.cmp_monomials(&xy, &y2), Ordering::Greater);
        // x > y^2 in lex (x-exp: 1 > 0)
        assert_eq!(order.cmp_monomials(&xm, &y2), Ordering::Greater);
    }

    #[test]
    fn test_deglex_ordering() {
        let order = DegLex::new(vec![x(), y()]);

        let x2 = Monomial::var_pow(x(), 2);
        let xy = Monomial::var(x()).mul(&Monomial::var(y()));
        let y2 = Monomial::var_pow(y(), 2);

        // All degree 2, then lex: x^2 > xy > y^2
        assert_eq!(order.cmp_monomials(&x2, &xy), Ordering::Greater);
        assert_eq!(order.cmp_monomials(&xy, &y2), Ordering::Greater);
    }

    #[test]
    fn test_ordered_monomial_btreemap() {
        use std::collections::BTreeMap;
        let order = GrevLex::with_var_order(vec![x(), y()]);

        let mut map = BTreeMap::new();
        map.insert(OrderedMonomial::new(Monomial::var(x()), order.clone()), 1);
        map.insert(
            OrderedMonomial::new(Monomial::var_pow(x(), 2), order.clone()),
            2,
        );
        map.insert(OrderedMonomial::new(Monomial::one(), order.clone()), 3);

        // Iteration should be in ascending order: 1, x, x^2
        let keys: Vec<_> = map.keys().map(|k| k.monomial.total_degree()).collect();
        assert_eq!(keys, vec![0, 1, 2]);
    }

    #[test]
    fn test_grevlex_equal_monomials() {
        let order = GrevLex::new();
        let a = Monomial::var(x());
        let b = Monomial::var(x());
        assert_eq!(order.cmp_monomials(&a, &b), Ordering::Equal);
    }

    #[test]
    fn test_one_vs_one() {
        let order = GrevLex::new();
        assert_eq!(
            order.cmp_monomials(&Monomial::one(), &Monomial::one()),
            Ordering::Equal
        );
    }
}
