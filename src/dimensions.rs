//! Unit and dimension handling for physical quantities.
//!
//! This module provides comprehensive dimensional analysis and unit conversion capabilities
//! for equations involving physical quantities. It implements a type-safe system based on
//! the International System of Units (SI) with support for derived units and unit conversions.
//!
//! # Dimensional Analysis
//!
//! Dimensional analysis is a method to check the consistency of equations and perform
//! unit conversions by tracking the fundamental physical dimensions (length, mass, time, etc.)
//! through calculations. Each physical quantity has an associated dimension that can be
//! expressed as a product of powers of base dimensions.
//!
//! # Core Concepts
//!
//! - **BaseDimension**: The seven SI base dimensions (length, mass, time, current, temperature,
//!   amount of substance, luminous intensity)
//! - **Dimension**: A composite dimension expressed as a product of base dimensions with exponents
//!   (e.g., velocity = Length¹ × Time⁻¹)
//! - **Unit**: A specific measurement standard with a dimension and conversion factor to SI base units
//! - **Quantity**: A numeric value paired with its unit of measurement
//! - **UnitRegistry**: A collection of units for lookup and conversion
//!
//! # Examples
//!
//! ## Creating and Using Units
//!
//! ```
//! use mathsolver_core::dimensions::{BaseDimension, Dimension, Unit, Quantity};
//!
//! // Create a velocity dimension: Length / Time
//! let length_dim = Dimension::from_base(BaseDimension::Length, 1);
//! let time_dim = Dimension::from_base(BaseDimension::Time, -1);
//! let velocity_dim = length_dim.multiply(&time_dim);
//!
//! // Create meter per second unit
//! let mps = Unit::new("meter per second", "m/s", velocity_dim.clone(), 1.0);
//!
//! // Create a quantity
//! let speed = Quantity::new(10.0, mps.clone());
//! println!("{}", speed); // "10 m/s"
//! ```
//!
//! ## Unit Conversion
//!
//! ```
//! use mathsolver_core::dimensions::{BaseDimension, Dimension, Unit};
//!
//! // Create length dimension and units
//! let length_dim = Dimension::from_base(BaseDimension::Length, 1);
//! let meter = Unit::new("meter", "m", length_dim.clone(), 1.0);
//! let kilometer = Unit::new("kilometer", "km", length_dim.clone(), 1000.0);
//!
//! // Convert 5 km to meters
//! let meters = kilometer.convert_to(5.0, &meter).unwrap();
//! assert_eq!(meters, 5000.0);
//! ```
//!
//! ## Temperature with Offset (Affine Units)
//!
//! ```
//! use mathsolver_core::dimensions::{BaseDimension, Dimension, Unit};
//!
//! // Create temperature dimension
//! let temp_dim = Dimension::from_base(BaseDimension::Temperature, 1);
//!
//! // Kelvin (SI base unit)
//! let kelvin = Unit::new("kelvin", "K", temp_dim.clone(), 1.0);
//!
//! // Celsius with offset: K = °C + 273.15
//! let celsius = Unit::with_offset("celsius", "°C", temp_dim.clone(), 1.0, 273.15);
//!
//! // Convert 0°C to Kelvin
//! let k = celsius.convert_to(0.0, &kelvin).unwrap();
//! assert_eq!(k, 273.15);
//! ```
//!
//! ## Derived Units (Force)
//!
//! ```
//! use mathsolver_core::dimensions::{BaseDimension, Dimension, Unit};
//!
//! // Force dimension: Mass × Length × Time⁻²
//! let mass_dim = Dimension::from_base(BaseDimension::Mass, 1);
//! let length_dim = Dimension::from_base(BaseDimension::Length, 1);
//! let time_dim = Dimension::from_base(BaseDimension::Time, -2);
//! let force_dim = mass_dim.multiply(&length_dim).multiply(&time_dim);
//!
//! // Newton: kg⋅m/s²
//! let newton = Unit::new("newton", "N", force_dim.clone(), 1.0);
//!
//! // Dyne: g⋅cm/s² = 10⁻⁵ N
//! let dyne = Unit::new("dyne", "dyn", force_dim.clone(), 1e-5);
//!
//! let force_in_dynes = newton.convert_to(1.0, &dyne).unwrap();
//! assert!((force_in_dynes - 100000.0).abs() < 0.01);
//! ```
//!
//! ## Using UnitRegistry
//!
//! ```
//! use mathsolver_core::dimensions::{BaseDimension, Dimension, Unit, UnitRegistry};
//!
//! let mut registry = UnitRegistry::new();
//!
//! // Add length units
//! let length_dim = Dimension::from_base(BaseDimension::Length, 1);
//! registry.add_unit(Unit::new("meter", "m", length_dim.clone(), 1.0));
//! registry.add_unit(Unit::new("foot", "ft", length_dim.clone(), 0.3048));
//!
//! // Convert using registry
//! let meters = registry.convert(100.0, "ft", "m").unwrap();
//! assert!((meters - 30.48).abs() < 0.001);
//! ```
//!
//! # TODO Items
//!
//! The following features are planned for future implementation:
//!
//! - Compound unit parsing and formatting (e.g., "m/s", "kg⋅m/s²")
//! - Unit prefix support (kilo-, mega-, milli-, micro-, etc.)
//! - Parsing quantities from strings (e.g., "5.2 m/s")
//! - Unit system conversions (SI, Imperial, CGS)
//! - Automatic dimensional analysis validation for equations
//! - Currency units with exchange rates
//! - Enhanced temperature conversion handling (absolute vs relative)

use std::collections::HashMap;
use std::fmt;

/// The seven fundamental SI base dimensions.
///
/// These dimensions form the foundation of the International System of Units (SI).
/// All other physical quantities can be expressed as combinations of these base dimensions.
///
/// # SI Base Dimensions
///
/// | Dimension | SI Base Unit | Symbol |
/// |-----------|--------------|--------|
/// | Length | meter | m |
/// | Mass | kilogram | kg |
/// | Time | second | s |
/// | Current | ampere | A |
/// | Temperature | kelvin | K |
/// | Amount | mole | mol |
/// | Luminosity | candela | cd |
///
/// # Examples
///
/// ```
/// use mathsolver_core::dimensions::BaseDimension;
///
/// // Each base dimension represents a fundamental physical property
/// let length = BaseDimension::Length;
/// let mass = BaseDimension::Mass;
/// let time = BaseDimension::Time;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BaseDimension {
    /// Length dimension - fundamental measure of spatial extent.
    ///
    /// SI base unit: meter (m)
    Length,

    /// Mass dimension - fundamental measure of matter quantity.
    ///
    /// SI base unit: kilogram (kg)
    Mass,

    /// Time dimension - fundamental measure of temporal duration.
    ///
    /// SI base unit: second (s)
    Time,

    /// Electric current dimension - fundamental measure of charge flow.
    ///
    /// SI base unit: ampere (A)
    Current,

    /// Thermodynamic temperature dimension - fundamental measure of thermal energy.
    ///
    /// SI base unit: kelvin (K)
    Temperature,

    /// Amount of substance dimension - fundamental measure of particle count.
    ///
    /// SI base unit: mole (mol)
    Amount,

    /// Luminous intensity dimension - fundamental measure of perceived light power.
    ///
    /// SI base unit: candela (cd)
    Luminosity,
}

/// Composite dimension expressed as a product of base dimensions with integer exponents.
///
/// A `Dimension` represents the dimensional formula of a physical quantity using the
/// exponent-based representation: L^a × M^b × T^c × ... where L=Length, M=Mass, T=Time, etc.
///
/// # Mathematical Representation
///
/// Each dimension is stored as a mapping from base dimensions to their exponents:
/// - Velocity: Length¹ × Time⁻¹ → {Length: 1, Time: -1}
/// - Force: Mass¹ × Length¹ × Time⁻² → {Mass: 1, Length: 1, Time: -2}
/// - Energy: Mass¹ × Length² × Time⁻² → {Mass: 1, Length: 2, Time: -2}
///
/// # Examples
///
/// ```
/// use mathsolver_core::dimensions::{BaseDimension, Dimension};
///
/// // Create velocity dimension: Length / Time = Length¹ × Time⁻¹
/// let length_dim = Dimension::from_base(BaseDimension::Length, 1);
/// let time_dim = Dimension::from_base(BaseDimension::Time, -1);
/// let velocity_dim = length_dim.multiply(&time_dim);
///
/// // Create dimensionless quantity (pure number)
/// let dimensionless = Dimension::dimensionless();
/// assert!(dimensionless.is_dimensionless());
/// ```
///
/// ## Derived Dimensions
///
/// Common derived dimensions:
/// - **Velocity**: L¹T⁻¹ (meters per second)
/// - **Acceleration**: L¹T⁻² (meters per second squared)
/// - **Force**: M¹L¹T⁻² (newtons = kg⋅m/s²)
/// - **Energy**: M¹L²T⁻² (joules = kg⋅m²/s²)
/// - **Power**: M¹L²T⁻³ (watts = kg⋅m²/s³)
/// - **Pressure**: M¹L⁻¹T⁻² (pascals = kg/(m⋅s²))
/// - **Frequency**: T⁻¹ (hertz = 1/s)
#[derive(Debug, Clone, PartialEq)]
pub struct Dimension {
    exponents: HashMap<BaseDimension, i32>,
}

impl Dimension {
    /// Create a dimensionless quantity.
    ///
    /// Dimensionless quantities are pure numbers with no associated physical dimension.
    /// Examples include ratios, angles (in radians), and mathematical constants.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::dimensions::Dimension;
    ///
    /// let dimensionless = Dimension::dimensionless();
    /// assert!(dimensionless.is_dimensionless());
    /// ```
    pub fn dimensionless() -> Self {
        Self {
            exponents: HashMap::new(),
        }
    }

    /// Create a dimension from a single base dimension with an exponent.
    ///
    /// # Arguments
    ///
    /// * `base` - The base dimension (Length, Mass, Time, etc.)
    /// * `exponent` - The power to which the base dimension is raised
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::dimensions::{BaseDimension, Dimension};
    ///
    /// // Create area dimension: Length²
    /// let area = Dimension::from_base(BaseDimension::Length, 2);
    ///
    /// // Create frequency dimension: Time⁻¹
    /// let frequency = Dimension::from_base(BaseDimension::Time, -1);
    ///
    /// // Zero exponent creates dimensionless
    /// let none = Dimension::from_base(BaseDimension::Mass, 0);
    /// assert!(none.is_dimensionless());
    /// ```
    pub fn from_base(base: BaseDimension, exponent: i32) -> Self {
        let mut exponents = HashMap::new();
        if exponent != 0 {
            exponents.insert(base, exponent);
        }
        Self { exponents }
    }

    /// Check if this dimension is dimensionless.
    ///
    /// Returns `true` if the dimension has no base dimension components
    /// (i.e., it represents a pure number).
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::dimensions::{BaseDimension, Dimension};
    ///
    /// let dimensionless = Dimension::dimensionless();
    /// assert!(dimensionless.is_dimensionless());
    ///
    /// let length = Dimension::from_base(BaseDimension::Length, 1);
    /// assert!(!length.is_dimensionless());
    /// ```
    pub fn is_dimensionless(&self) -> bool {
        self.exponents.is_empty()
    }

    /// Multiply two dimensions by adding their exponents.
    ///
    /// Dimension multiplication corresponds to physical quantity multiplication:
    /// - Distance × Distance = Area (L¹ × L¹ = L²)
    /// - Force × Distance = Energy (ML¹T⁻² × L¹ = ML²T⁻²)
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::dimensions::{BaseDimension, Dimension};
    ///
    /// // Velocity × Time = Distance
    /// let velocity = Dimension::from_base(BaseDimension::Length, 1)
    ///     .multiply(&Dimension::from_base(BaseDimension::Time, -1));
    /// let time = Dimension::from_base(BaseDimension::Time, 1);
    /// let distance = velocity.multiply(&time);
    /// // Result: Length¹ × Time⁻¹ × Time¹ = Length¹
    /// ```
    ///
    /// # TODO
    ///
    /// Currently returns a clone of self. Full implementation pending.
    pub fn multiply(&self, _other: &Dimension) -> Dimension {
        // TODO: Implement dimension multiplication
        // Add exponents for matching base dimensions
        self.clone()
    }

    /// Divide two dimensions by subtracting their exponents.
    ///
    /// Dimension division corresponds to physical quantity division:
    /// - Distance / Time = Velocity (L¹ / T¹ = L¹T⁻¹)
    /// - Force / Mass = Acceleration (ML¹T⁻² / M¹ = L¹T⁻²)
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::dimensions::{BaseDimension, Dimension};
    ///
    /// // Distance / Time = Velocity
    /// let distance = Dimension::from_base(BaseDimension::Length, 1);
    /// let time = Dimension::from_base(BaseDimension::Time, 1);
    /// let velocity = distance.divide(&time);
    /// // Result: Length¹T⁻¹
    /// ```
    ///
    /// # TODO
    ///
    /// Currently returns a clone of self. Full implementation pending.
    pub fn divide(&self, _other: &Dimension) -> Dimension {
        // TODO: Implement dimension division
        // Subtract exponents for matching base dimensions
        self.clone()
    }

    /// Raise a dimension to a power by multiplying all exponents.
    ///
    /// Dimension exponentiation corresponds to physical quantity exponentiation:
    /// - (Length)² = Area (L¹ → L²)
    /// - (Velocity)² = (L¹T⁻¹)² = L²T⁻²
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::dimensions::{BaseDimension, Dimension};
    ///
    /// // Square a velocity to get (m/s)²
    /// let velocity = Dimension::from_base(BaseDimension::Length, 1)
    ///     .multiply(&Dimension::from_base(BaseDimension::Time, -1));
    /// let velocity_squared = velocity.power(2);
    /// // Result: Length²T⁻²
    /// ```
    ///
    /// # TODO
    ///
    /// Currently returns a clone of self. Full implementation pending.
    pub fn power(&self, _exponent: i32) -> Dimension {
        // TODO: Implement dimension exponentiation
        // Multiply all exponents by the power
        self.clone()
    }

    /// Check if two dimensions are compatible (can be converted between).
    ///
    /// Two dimensions are compatible if they have identical exponents for all
    /// base dimensions. Only compatible dimensions can be added, subtracted,
    /// or converted between.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::dimensions::{BaseDimension, Dimension};
    ///
    /// let length1 = Dimension::from_base(BaseDimension::Length, 1);
    /// let length2 = Dimension::from_base(BaseDimension::Length, 1);
    /// let area = Dimension::from_base(BaseDimension::Length, 2);
    ///
    /// assert!(length1.is_compatible(&length2)); // meters ↔ feet
    /// assert!(!length1.is_compatible(&area));   // meters ↮ square meters
    /// ```
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
///
/// A `Unit` represents a specific measurement standard for a physical quantity.
/// It includes the unit's name, symbol, dimension, and conversion factors to the
/// SI base unit system.
///
/// # Conversion Formula
///
/// Conversion to SI base units uses the formula:
/// ```text
/// SI_value = value * to_si_factor + offset
/// ```
///
/// For most units, `offset = 0` (linear/multiplicative conversion).
/// For affine units like temperature, `offset ≠ 0` (affine/additive conversion).
///
/// # Examples
///
/// ## Linear Units (Length)
///
/// ```
/// use mathsolver_core::dimensions::{BaseDimension, Dimension, Unit};
///
/// let length_dim = Dimension::from_base(BaseDimension::Length, 1);
///
/// // Meter (SI base unit)
/// let meter = Unit::new("meter", "m", length_dim.clone(), 1.0);
///
/// // Kilometer (1 km = 1000 m)
/// let kilometer = Unit::new("kilometer", "km", length_dim.clone(), 1000.0);
///
/// // Inch (1 in = 0.0254 m)
/// let inch = Unit::new("inch", "in", length_dim.clone(), 0.0254);
/// ```
///
/// ## Affine Units (Temperature)
///
/// Temperature units require an offset for absolute temperature scales:
///
/// ```
/// use mathsolver_core::dimensions::{BaseDimension, Dimension, Unit};
///
/// let temp_dim = Dimension::from_base(BaseDimension::Temperature, 1);
///
/// // Kelvin (SI base unit)
/// let kelvin = Unit::new("kelvin", "K", temp_dim.clone(), 1.0);
///
/// // Celsius: K = °C + 273.15
/// let celsius = Unit::with_offset("celsius", "°C", temp_dim.clone(), 1.0, 273.15);
///
/// // Fahrenheit: K = (°F + 459.67) * 5/9
/// let fahrenheit = Unit::with_offset(
///     "fahrenheit", "°F", temp_dim.clone(), 5.0/9.0, 459.67 * 5.0/9.0
/// );
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Unit {
    /// Unit name (e.g., "meter", "kilogram").
    ///
    /// Full descriptive name of the unit in singular form.
    pub name: String,

    /// Unit symbol (e.g., "m", "kg").
    ///
    /// Standard abbreviated symbol for the unit, following international conventions.
    pub symbol: String,

    /// Physical dimension of this unit.
    ///
    /// Defines what physical quantity this unit measures (length, mass, etc.)
    pub dimension: Dimension,

    /// Conversion factor to SI base unit.
    ///
    /// Multiplier used in conversion: `SI_value = value * to_si_factor + offset`
    pub to_si_factor: f64,

    /// Offset for affine units (e.g., Celsius, Fahrenheit).
    ///
    /// Additive offset used in conversion: `SI_value = value * to_si_factor + offset`
    /// - For linear units (meters, kilograms): `offset = 0`
    /// - For affine units (Celsius, Fahrenheit): `offset ≠ 0`
    pub offset: f64,
}

impl Unit {
    /// Create a new linear unit (without offset).
    ///
    /// Use this for units with purely multiplicative conversion to SI base units.
    /// Most units are linear: length, mass, velocity, force, energy, etc.
    ///
    /// # Arguments
    ///
    /// * `name` - Full name of the unit (e.g., "meter", "kilogram")
    /// * `symbol` - Standard symbol (e.g., "m", "kg")
    /// * `dimension` - Physical dimension of the unit
    /// * `to_si_factor` - Multiplier to convert to SI base unit
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::dimensions::{BaseDimension, Dimension, Unit};
    ///
    /// let length_dim = Dimension::from_base(BaseDimension::Length, 1);
    ///
    /// // SI base unit
    /// let meter = Unit::new("meter", "m", length_dim.clone(), 1.0);
    ///
    /// // 1 mile = 1609.344 meters
    /// let mile = Unit::new("mile", "mi", length_dim.clone(), 1609.344);
    ///
    /// let meters = mile.to_si(5.0);
    /// assert_eq!(meters, 8046.72);
    /// ```
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

    /// Create an affine unit with offset.
    ///
    /// Use this for units with both multiplicative and additive conversion to SI base units.
    /// This is primarily needed for temperature scales (Celsius, Fahrenheit).
    ///
    /// # Arguments
    ///
    /// * `name` - Full name of the unit
    /// * `symbol` - Standard symbol
    /// * `dimension` - Physical dimension
    /// * `to_si_factor` - Multiplier in conversion formula
    /// * `offset` - Additive offset in conversion formula
    ///
    /// # Conversion Formula
    ///
    /// `SI_value = value * to_si_factor + offset`
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::dimensions::{BaseDimension, Dimension, Unit};
    ///
    /// let temp_dim = Dimension::from_base(BaseDimension::Temperature, 1);
    ///
    /// // Celsius: K = °C + 273.15
    /// let celsius = Unit::with_offset("celsius", "°C", temp_dim.clone(), 1.0, 273.15);
    ///
    /// let kelvin_value = celsius.to_si(0.0);
    /// assert_eq!(kelvin_value, 273.15); // 0°C = 273.15 K
    ///
    /// let celsius_value = celsius.from_si(373.15);
    /// assert_eq!(celsius_value, 100.0); // 373.15 K = 100°C
    /// ```
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

    /// Convert a value from this unit to SI base unit.
    ///
    /// Applies the conversion formula: `SI_value = value * to_si_factor + offset`
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::dimensions::{BaseDimension, Dimension, Unit};
    ///
    /// let length_dim = Dimension::from_base(BaseDimension::Length, 1);
    /// let kilometer = Unit::new("kilometer", "km", length_dim, 1000.0);
    ///
    /// let meters = kilometer.to_si(5.0);
    /// assert_eq!(meters, 5000.0);
    /// ```
    pub fn to_si(&self, value: f64) -> f64 {
        value * self.to_si_factor + self.offset
    }

    /// Convert a value from SI base unit to this unit.
    ///
    /// Applies the inverse conversion formula: `value = (SI_value - offset) / to_si_factor`
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::dimensions::{BaseDimension, Dimension, Unit};
    ///
    /// let length_dim = Dimension::from_base(BaseDimension::Length, 1);
    /// let foot = Unit::new("foot", "ft", length_dim, 0.3048);
    ///
    /// let feet = foot.from_si(100.0);
    /// assert!((feet - 328.084).abs() < 0.001);
    /// ```
    pub fn from_si(&self, value: f64) -> f64 {
        (value - self.offset) / self.to_si_factor
    }

    /// Convert a value from this unit to another unit.
    ///
    /// Performs two-step conversion: source → SI base → target unit.
    /// Returns an error if the dimensions are incompatible.
    ///
    /// # Arguments
    ///
    /// * `value` - Value in the source unit
    /// * `target` - Target unit to convert to
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - Converted value if dimensions are compatible
    /// * `Err(String)` - Error message if dimensions are incompatible
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::dimensions::{BaseDimension, Dimension, Unit};
    ///
    /// let length_dim = Dimension::from_base(BaseDimension::Length, 1);
    /// let mile = Unit::new("mile", "mi", length_dim.clone(), 1609.344);
    /// let kilometer = Unit::new("kilometer", "km", length_dim.clone(), 1000.0);
    ///
    /// let km = mile.convert_to(10.0, &kilometer).unwrap();
    /// assert!((km - 16.09344).abs() < 0.0001);
    ///
    /// // Incompatible dimensions produce error
    /// let time_dim = Dimension::from_base(BaseDimension::Time, 1);
    /// let second = Unit::new("second", "s", time_dim, 1.0);
    /// assert!(mile.convert_to(10.0, &second).is_err());
    /// ```
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

/// Registry of standard units for lookup and conversion.
///
/// `UnitRegistry` provides a centralized collection of units that can be accessed by
/// their symbols. It simplifies unit conversions by managing unit instances and providing
/// string-based conversion methods.
///
/// # Examples
///
/// ```
/// use mathsolver_core::dimensions::{BaseDimension, Dimension, Unit, UnitRegistry};
///
/// let mut registry = UnitRegistry::new();
///
/// // Add length units
/// let length_dim = Dimension::from_base(BaseDimension::Length, 1);
/// registry.add_unit(Unit::new("meter", "m", length_dim.clone(), 1.0));
/// registry.add_unit(Unit::new("kilometer", "km", length_dim.clone(), 1000.0));
/// registry.add_unit(Unit::new("mile", "mi", length_dim.clone(), 1609.344));
///
/// // Convert using symbols
/// let km = registry.convert(10.0, "mi", "km").unwrap();
/// assert!((km - 16.09344).abs() < 0.0001);
/// ```
#[derive(Debug, Clone)]
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
    /// use mathsolver_core::dimensions::UnitRegistry;
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
    /// use mathsolver_core::dimensions::{BaseDimension, Dimension, Unit, UnitRegistry};
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
    /// use mathsolver_core::dimensions::{BaseDimension, Dimension, Unit, UnitRegistry};
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
    /// use mathsolver_core::dimensions::{BaseDimension, Dimension, Unit, UnitRegistry};
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

/// Physical quantity combining a numeric value with its unit of measurement.
///
/// A `Quantity` represents a measured or calculated physical value with explicit units,
/// enabling type-safe dimensional analysis and unit conversions.
///
/// # Examples
///
/// ## Creating Quantities
///
/// ```
/// use mathsolver_core::dimensions::{BaseDimension, Dimension, Unit, Quantity};
///
/// let length_dim = Dimension::from_base(BaseDimension::Length, 1);
/// let meter = Unit::new("meter", "m", length_dim, 1.0);
///
/// let distance = Quantity::new(100.0, meter);
/// println!("{}", distance); // "100 m"
/// ```
///
/// ## Converting Quantities
///
/// ```
/// use mathsolver_core::dimensions::{BaseDimension, Dimension, Unit, Quantity};
///
/// let length_dim = Dimension::from_base(BaseDimension::Length, 1);
/// let meter = Unit::new("meter", "m", length_dim.clone(), 1.0);
/// let kilometer = Unit::new("kilometer", "km", length_dim.clone(), 1000.0);
///
/// let distance_m = Quantity::new(5000.0, meter);
/// let distance_km = distance_m.convert_to(&kilometer).unwrap();
///
/// assert_eq!(distance_km.value, 5.0);
/// assert_eq!(distance_km.unit.symbol, "km");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Quantity {
    /// The numeric value of the quantity.
    pub value: f64,

    /// The unit of measurement for this quantity.
    pub unit: Unit,
}

impl Quantity {
    /// Create a new quantity from a value and unit.
    ///
    /// # Arguments
    ///
    /// * `value` - The numeric magnitude
    /// * `unit` - The unit of measurement
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::dimensions::{BaseDimension, Dimension, Unit, Quantity};
    ///
    /// let mass_dim = Dimension::from_base(BaseDimension::Mass, 1);
    /// let kilogram = Unit::new("kilogram", "kg", mass_dim, 1.0);
    ///
    /// let mass = Quantity::new(75.5, kilogram);
    /// assert_eq!(mass.value, 75.5);
    /// ```
    pub fn new(value: f64, unit: Unit) -> Self {
        Self { value, unit }
    }

    /// Convert this quantity to another unit.
    ///
    /// Creates a new `Quantity` with the value converted to the target unit.
    /// Returns an error if the units have incompatible dimensions.
    ///
    /// # Arguments
    ///
    /// * `target_unit` - The unit to convert to
    ///
    /// # Returns
    ///
    /// * `Ok(Quantity)` - New quantity in target units if compatible
    /// * `Err(String)` - Error message if dimensions are incompatible
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::dimensions::{BaseDimension, Dimension, Unit, Quantity};
    ///
    /// let time_dim = Dimension::from_base(BaseDimension::Time, 1);
    /// let second = Unit::new("second", "s", time_dim.clone(), 1.0);
    /// let minute = Unit::new("minute", "min", time_dim.clone(), 60.0);
    ///
    /// let duration_s = Quantity::new(300.0, second);
    /// let duration_min = duration_s.convert_to(&minute).unwrap();
    ///
    /// assert_eq!(duration_min.value, 5.0);
    /// assert_eq!(duration_min.unit.symbol, "min");
    /// ```
    pub fn convert_to(&self, target_unit: &Unit) -> Result<Quantity, String> {
        let converted_value = self.unit.convert_to(self.value, target_unit)?;
        Ok(Quantity::new(converted_value, target_unit.clone()))
    }

    /// Get the value in SI base units.
    ///
    /// Converts this quantity to its equivalent value in the SI base unit system.
    /// This is useful for calculations that need to be performed in a consistent
    /// unit system.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathsolver_core::dimensions::{BaseDimension, Dimension, Unit, Quantity};
    ///
    /// let length_dim = Dimension::from_base(BaseDimension::Length, 1);
    /// let kilometer = Unit::new("kilometer", "km", length_dim, 1000.0);
    ///
    /// let distance = Quantity::new(5.0, kilometer);
    /// let si_value = distance.to_si();
    ///
    /// assert_eq!(si_value, 5000.0); // 5 km = 5000 m
    /// ```
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
