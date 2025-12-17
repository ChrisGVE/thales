//! Unit and dimension handling for physical quantities.
//!
//! Provides dimensional analysis and unit conversion capabilities
//! for equations involving physical quantities.

use std::collections::HashMap;
use std::fmt;

/// SI base dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BaseDimension {
    /// Length (meter)
    Length,
    /// Mass (kilogram)
    Mass,
    /// Time (second)
    Time,
    /// Electric current (ampere)
    Current,
    /// Temperature (kelvin)
    Temperature,
    /// Amount of substance (mole)
    Amount,
    /// Luminous intensity (candela)
    Luminosity,
}

/// Dimension as a product of base dimensions with exponents.
///
/// For example: velocity = Length^1 * Time^-1
#[derive(Debug, Clone, PartialEq)]
pub struct Dimension {
    exponents: HashMap<BaseDimension, i32>,
}

impl Dimension {
    /// Create a dimensionless quantity.
    pub fn dimensionless() -> Self {
        Self {
            exponents: HashMap::new(),
        }
    }

    /// Create a dimension from base dimension.
    pub fn from_base(base: BaseDimension, exponent: i32) -> Self {
        let mut exponents = HashMap::new();
        if exponent != 0 {
            exponents.insert(base, exponent);
        }
        Self { exponents }
    }

    /// Check if dimension is dimensionless.
    pub fn is_dimensionless(&self) -> bool {
        self.exponents.is_empty()
    }

    /// Multiply two dimensions.
    pub fn multiply(&self, _other: &Dimension) -> Dimension {
        // TODO: Implement dimension multiplication
        // Add exponents for matching base dimensions
        self.clone()
    }

    /// Divide two dimensions.
    pub fn divide(&self, _other: &Dimension) -> Dimension {
        // TODO: Implement dimension division
        // Subtract exponents for matching base dimensions
        self.clone()
    }

    /// Raise dimension to a power.
    pub fn power(&self, _exponent: i32) -> Dimension {
        // TODO: Implement dimension exponentiation
        // Multiply all exponents by the power
        self.clone()
    }

    /// Check if two dimensions are compatible.
    pub fn is_compatible(&self, other: &Dimension) -> bool {
        self == other
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: Format dimension nicely (e.g., "m/s^2")
        write!(f, "[dimension]")
    }
}

/// Unit of measurement with dimension and conversion factor.
#[derive(Debug, Clone, PartialEq)]
pub struct Unit {
    /// Unit name (e.g., "meter", "kilogram")
    pub name: String,
    /// Unit symbol (e.g., "m", "kg")
    pub symbol: String,
    /// Physical dimension
    pub dimension: Dimension,
    /// Conversion factor to SI base unit
    pub to_si_factor: f64,
    /// Offset for affine units (e.g., Celsius, Fahrenheit)
    pub offset: f64,
}

impl Unit {
    /// Create a new unit.
    pub fn new(
        name: impl Into<String>,
        symbol: impl Into<String>,
        dimension: Dimension,
        to_si_factor: f64,
    ) -> Self {
        Self {
            name: name.into(),
            symbol: symbol.into(),
            dimension,
            to_si_factor,
            offset: 0.0,
        }
    }

    /// Create a unit with offset (for affine units).
    pub fn with_offset(
        name: impl Into<String>,
        symbol: impl Into<String>,
        dimension: Dimension,
        to_si_factor: f64,
        offset: f64,
    ) -> Self {
        Self {
            name: name.into(),
            symbol: symbol.into(),
            dimension,
            to_si_factor,
            offset,
        }
    }

    /// Convert value from this unit to SI base unit.
    pub fn to_si(&self, value: f64) -> f64 {
        value * self.to_si_factor + self.offset
    }

    /// Convert value from SI base unit to this unit.
    pub fn from_si(&self, value: f64) -> f64 {
        (value - self.offset) / self.to_si_factor
    }

    /// Convert value from this unit to another unit.
    pub fn convert_to(&self, value: f64, target: &Unit) -> Result<f64, String> {
        if !self.dimension.is_compatible(&target.dimension) {
            return Err(format!(
                "Incompatible dimensions: {} vs {}",
                self.dimension, target.dimension
            ));
        }
        let si_value = self.to_si(value);
        Ok(target.from_si(si_value))
    }
}

/// Registry of standard units.
#[derive(Debug, Clone)]
pub struct UnitRegistry {
    units: HashMap<String, Unit>,
}

impl UnitRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            units: HashMap::new(),
        }
    }

    /// Create a registry with SI base units.
    pub fn with_si_base() -> Self {
        let registry = Self::new();
        // TODO: Add SI base units
        // meter, kilogram, second, ampere, kelvin, mole, candela
        registry
    }

    /// Create a registry with common derived units.
    pub fn with_common_units() -> Self {
        let registry = Self::with_si_base();
        // TODO: Add common derived units
        // newton, joule, watt, pascal, etc.
        registry
    }

    /// Add a unit to the registry.
    pub fn add_unit(&mut self, unit: Unit) {
        self.units.insert(unit.symbol.clone(), unit);
    }

    /// Get a unit by symbol.
    pub fn get(&self, symbol: &str) -> Option<&Unit> {
        self.units.get(symbol)
    }

    /// Convert between two units by symbol.
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

/// Physical quantity with value and unit.
#[derive(Debug, Clone, PartialEq)]
pub struct Quantity {
    pub value: f64,
    pub unit: Unit,
}

impl Quantity {
    /// Create a new quantity.
    pub fn new(value: f64, unit: Unit) -> Self {
        Self { value, unit }
    }

    /// Convert to another unit.
    pub fn convert_to(&self, target_unit: &Unit) -> Result<Quantity, String> {
        let converted_value = self.unit.convert_to(self.value, target_unit)?;
        Ok(Quantity::new(converted_value, target_unit.clone()))
    }

    /// Get the SI base value.
    pub fn to_si(&self) -> f64 {
        self.unit.to_si(self.value)
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.value, self.unit.symbol)
    }
}

// TODO: Add support for compound units (m/s, kg*m/s^2)
// TODO: Add unit parsing from strings ("5.2 m/s")
// TODO: Add unit system conversions (SI, Imperial, CGS)
// TODO: Add unit prefix support (kilo, mega, milli, micro)
// TODO: Add dimensional analysis validation for equations
// TODO: Add automatic unit inference
// TODO: Add support for currency units with exchange rates
// TODO: Add temperature conversion with proper handling of absolute vs relative
