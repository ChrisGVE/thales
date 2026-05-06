//! Global symbol interning for O(1) variable name comparison.
//!
//! [`SymbolId`] is a `Copy` type (4 bytes) that represents an interned string.
//! Two symbols with the same string always have the same ID, enabling
//! equality checks via `u32` comparison instead of string comparison.

use std::collections::HashMap;
use std::fmt;
use std::sync::{LazyLock, RwLock};

/// Sentinel base for strategy-introduced variables in the D0 cache.
///
/// `SymbolId` indices at or above this value are sentinels created by
/// `fresh_id_gen` during rehydration; they are never interned in the global
/// symbol table. Their [`Display`] and [`Debug`] representations are
/// `$slot_N` where `N = u32::MAX - index`.
pub(crate) const SENTINEL_BASE: u32 = u32::MAX - 65535;

/// A compact, `Copy` handle to an interned string.
///
/// Created via [`SymbolId::intern`]. Two `SymbolId` values are equal if and
/// only if they refer to the same string. Comparison is O(1).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymbolId(u32);

impl SymbolId {
    /// Intern a string, returning its unique ID.
    ///
    /// If the string was previously interned, returns the same ID.
    /// Thread-safe.
    pub fn intern(name: &str) -> Self {
        // Fast path: check if already interned (read lock only)
        {
            let table = INTERNER.read().expect("symbol interner poisoned");
            if let Some(&id) = table.str_to_id.get(name) {
                return id;
            }
        }
        // Slow path: acquire write lock and insert
        let mut table = INTERNER.write().expect("symbol interner poisoned");
        // Double-check after acquiring write lock
        if let Some(&id) = table.str_to_id.get(name) {
            return id;
        }
        let id = SymbolId(table.id_to_str.len() as u32);
        let owned = name.to_string();
        table.str_to_id.insert(owned.clone(), id);
        table.id_to_str.push(owned);
        id
    }

    /// Recover the original string for this symbol.
    pub fn as_str(&self) -> String {
        let table = INTERNER.read().expect("symbol interner poisoned");
        table.id_to_str[self.0 as usize].clone()
    }

    /// Returns the raw `u32` index.
    #[inline]
    pub fn index(self) -> u32 {
        self.0
    }
}

impl fmt::Display for SymbolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 >= SENTINEL_BASE {
            let slot_num = u32::MAX - self.0;
            return write!(f, "$slot_{}", slot_num);
        }
        let table = INTERNER.read().expect("symbol interner poisoned");
        write!(f, "{}", table.id_to_str[self.0 as usize])
    }
}

impl fmt::Debug for SymbolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 >= SENTINEL_BASE {
            let slot_num = u32::MAX - self.0;
            return write!(f, "SymbolId({}, \"$slot_{}\")", self.0, slot_num);
        }
        let table = INTERNER.read().expect("symbol interner poisoned");
        write!(
            f,
            "SymbolId({}, \"{}\")",
            self.0, table.id_to_str[self.0 as usize]
        )
    }
}

// ── Global interner ───────────────────────────────────────────────────────────

struct SymbolTable {
    str_to_id: HashMap<String, SymbolId>,
    id_to_str: Vec<String>,
}

static INTERNER: LazyLock<RwLock<SymbolTable>> = LazyLock::new(|| {
    RwLock::new(SymbolTable {
        str_to_id: HashMap::new(),
        id_to_str: Vec::new(),
    })
});

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intern_same_string_same_id() {
        let a = SymbolId::intern("x");
        let b = SymbolId::intern("x");
        assert_eq!(a, b);
    }

    #[test]
    fn test_different_strings_different_ids() {
        let a = SymbolId::intern("alpha");
        let b = SymbolId::intern("beta");
        assert_ne!(a, b);
    }

    #[test]
    fn test_display_recovers_string() {
        let id = SymbolId::intern("theta");
        assert_eq!(id.to_string(), "theta");
    }

    #[test]
    fn test_as_str() {
        let id = SymbolId::intern("gamma");
        assert_eq!(id.as_str(), "gamma");
    }

    #[test]
    fn test_copy_semantics() {
        let a = SymbolId::intern("delta");
        let b = a; // Copy
        assert_eq!(a, b);
        assert_eq!(std::mem::size_of_val(&a), 4);
    }

    #[test]
    fn test_ordering() {
        // Ordering is by intern order (index), not lexicographic
        let first = SymbolId::intern("zzz_first");
        let second = SymbolId::intern("aaa_second");
        assert!(first < second);
    }

    #[test]
    fn test_hash_usable_in_collections() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SymbolId::intern("a_sym"));
        set.insert(SymbolId::intern("b_sym"));
        set.insert(SymbolId::intern("a_sym")); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_concurrent_interning() {
        use std::sync::Arc;
        use std::thread;

        let results: Vec<_> = (0..8)
            .map(|i| {
                let name = Arc::new(format!("concurrent_{i}"));
                let name2 = name.clone();
                thread::spawn(move || {
                    let a = SymbolId::intern(&name);
                    let b = SymbolId::intern(&name2);
                    assert_eq!(a, b);
                    a
                })
            })
            .collect();

        let ids: Vec<_> = results.into_iter().map(|h| h.join().unwrap()).collect();
        // All different names should have different IDs
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), 8);
    }
}
