//! Unit registry with predefined units.

use super::unit::Unit;
use std::collections::HashMap;

pub struct UnitRegistry {
    units: HashMap<String, Unit>,
}

impl UnitRegistry {
    /// Create a new empty registry.
    ///
    /// Use this when you want to build a custom registry with only specific units.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::dimensions::UnitRegistry;
    ///
    /// let registry = UnitRegistry::new();
    /// // Registry is empty, ready for custom units
    /// ```
    pub fn new() -> Self {
        Self {
            units: HashMap::new(),
        }
    }

    /// Create a registry pre-populated with SI base units.
    ///
    /// # SI Base Units
    ///
    /// The following units will be added (when implemented):
    /// - meter (m) - length
    /// - kilogram (kg) - mass
    /// - second (s) - time
    /// - ampere (A) - electric current
    /// - kelvin (K) - temperature
    /// - mole (mol) - amount of substance
    /// - candela (cd) - luminous intensity
    ///
    /// # TODO
    ///
    /// Currently returns an empty registry. Implementation pending.
    pub fn with_si_base() -> Self {
        let registry = Self::new();
        // TODO: Add SI base units
        // meter, kilogram, second, ampere, kelvin, mole, candela
        registry
    }

    /// Create a registry with common derived units.
    ///
    /// Includes SI base units plus common derived units such as:
    /// - newton (N) - force
    /// - joule (J) - energy
    /// - watt (W) - power
    /// - pascal (Pa) - pressure
    /// - volt (V) - electric potential
    /// - ohm (Ω) - electric resistance
    /// - hertz (Hz) - frequency
    ///
    /// # TODO
    ///
    /// Currently returns a registry with only base units. Derived units pending.
    pub fn with_common_units() -> Self {
        let registry = Self::with_si_base();
        // TODO: Add common derived units
        // newton, joule, watt, pascal, etc.
        registry
    }

    /// Add a unit to the registry.
    ///
    /// The unit is indexed by its symbol. If a unit with the same symbol already
    /// exists, it will be replaced.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::dimensions::{BaseDimension, Dimension, Unit, UnitRegistry};
    ///
    /// let mut registry = UnitRegistry::new();
    /// let length_dim = Dimension::from_base(BaseDimension::Length, 1);
    /// let meter = Unit::new("meter", "m", length_dim, 1.0);
    ///
    /// registry.add_unit(meter);
    /// assert!(registry.get("m").is_some());
    /// ```
    pub fn add_unit(&mut self, unit: Unit) {
        self.units.insert(unit.symbol.clone(), unit);
    }

    /// Get a unit by its symbol.
    ///
    /// # Arguments
    ///
    /// * `symbol` - The unit symbol to look up (e.g., "m", "kg", "s")
    ///
    /// # Returns
    ///
    /// * `Some(&Unit)` - Reference to the unit if found
    /// * `None` - If no unit with that symbol exists
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::dimensions::{BaseDimension, Dimension, Unit, UnitRegistry};
    ///
    /// let mut registry = UnitRegistry::new();
    /// let length_dim = Dimension::from_base(BaseDimension::Length, 1);
    /// registry.add_unit(Unit::new("meter", "m", length_dim, 1.0));
    ///
    /// let meter = registry.get("m");
    /// assert!(meter.is_some());
    /// assert_eq!(meter.unwrap().name, "meter");
    ///
    /// let unknown = registry.get("xyz");
    /// assert!(unknown.is_none());
    /// ```
    pub fn get(&self, symbol: &str) -> Option<&Unit> {
        self.units.get(symbol)
    }

    /// Convert a value between two units using their symbols.
    ///
    /// This is a convenience method that looks up both units and performs the conversion.
    ///
    /// # Arguments
    ///
    /// * `value` - Value to convert
    /// * `from` - Symbol of the source unit
    /// * `to` - Symbol of the target unit
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - Converted value if both units exist and are compatible
    /// * `Err(String)` - Error if units don't exist or are incompatible
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::dimensions::{BaseDimension, Dimension, Unit, UnitRegistry};
    ///
    /// let mut registry = UnitRegistry::new();
    /// let length_dim = Dimension::from_base(BaseDimension::Length, 1);
    /// registry.add_unit(Unit::new("meter", "m", length_dim.clone(), 1.0));
    /// registry.add_unit(Unit::new("kilometer", "km", length_dim.clone(), 1000.0));
    ///
    /// let meters = registry.convert(5.0, "km", "m").unwrap();
    /// assert_eq!(meters, 5000.0);
    ///
    /// // Error cases
    /// assert!(registry.convert(5.0, "unknown", "m").is_err());
    /// ```
    pub fn convert(&self, value: f64, from: &str, to: &str) -> Result<f64, String> {
        let from_unit = self
            .get(from)
            .ok_or_else(|| format!("Unknown unit: {}", from))?;
        let to_unit = self
            .get(to)
            .ok_or_else(|| format!("Unknown unit: {}", to))?;
        from_unit.convert_to(value, to_unit)
    }
}

impl Default for UnitRegistry {
    fn default() -> Self {
        Self::with_common_units()
    }
}
