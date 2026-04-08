//! Unit type and operations.

use super::types::Dimension;

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
/// use thales::dimensions::{BaseDimension, Dimension, Unit};
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
/// use thales::dimensions::{BaseDimension, Dimension, Unit};
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
    /// use thales::dimensions::{BaseDimension, Dimension, Unit};
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
    /// use thales::dimensions::{BaseDimension, Dimension, Unit};
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
    /// use thales::dimensions::{BaseDimension, Dimension, Unit};
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
    /// use thales::dimensions::{BaseDimension, Dimension, Unit};
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
    /// use thales::dimensions::{BaseDimension, Dimension, Unit};
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
