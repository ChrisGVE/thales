//! Physical quantities with units.

use super::unit::Unit;
use std::fmt;

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
/// use thales::dimensions::{BaseDimension, Dimension, Unit, Quantity};
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
/// use thales::dimensions::{BaseDimension, Dimension, Unit, Quantity};
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
    /// use thales::dimensions::{BaseDimension, Dimension, Unit, Quantity};
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
    /// use thales::dimensions::{BaseDimension, Dimension, Unit, Quantity};
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
    /// use thales::dimensions::{BaseDimension, Dimension, Unit, Quantity};
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
