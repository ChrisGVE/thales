//! Fundamental types for dimensional analysis.

use std::collections::HashMap;
use std::fmt;

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
/// use thales::dimensions::{BaseDimension, Dimension};
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
    pub(crate) exponents: HashMap<BaseDimension, i32>,
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
    /// use thales::dimensions::Dimension;
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
    /// use thales::dimensions::{BaseDimension, Dimension};
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
    /// use thales::dimensions::{BaseDimension, Dimension};
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
    /// use thales::dimensions::{BaseDimension, Dimension};
    ///
    /// // Velocity × Time = Distance
    /// let velocity = Dimension::from_base(BaseDimension::Length, 1)
    ///     .multiply(&Dimension::from_base(BaseDimension::Time, -1));
    /// let time = Dimension::from_base(BaseDimension::Time, 1);
    /// let distance = velocity.multiply(&time);
    /// // Result: Length¹ × Time⁻¹ × Time¹ = Length¹
    /// ```
    ///
    pub fn multiply(&self, other: &Dimension) -> Dimension {
        let mut exponents = self.exponents.clone();
        for (&base, &exp) in &other.exponents {
            let entry = exponents.entry(base).or_insert(0);
            *entry += exp;
            if *entry == 0 {
                exponents.remove(&base);
            }
        }
        Self { exponents }
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
    /// use thales::dimensions::{BaseDimension, Dimension};
    ///
    /// // Distance / Time = Velocity
    /// let distance = Dimension::from_base(BaseDimension::Length, 1);
    /// let time = Dimension::from_base(BaseDimension::Time, 1);
    /// let velocity = distance.divide(&time);
    /// // Result: Length¹T⁻¹
    /// ```
    ///
    pub fn divide(&self, other: &Dimension) -> Dimension {
        let mut exponents = self.exponents.clone();
        for (&base, &exp) in &other.exponents {
            let entry = exponents.entry(base).or_insert(0);
            *entry -= exp;
            if *entry == 0 {
                exponents.remove(&base);
            }
        }
        Self { exponents }
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
    /// use thales::dimensions::{BaseDimension, Dimension};
    ///
    /// // Square a velocity to get (m/s)²
    /// let velocity = Dimension::from_base(BaseDimension::Length, 1)
    ///     .multiply(&Dimension::from_base(BaseDimension::Time, -1));
    /// let velocity_squared = velocity.power(2);
    /// // Result: Length²T⁻²
    /// ```
    ///
    pub fn power(&self, exponent: i32) -> Dimension {
        if exponent == 0 {
            return Self::dimensionless();
        }
        let exponents = self
            .exponents
            .iter()
            .map(|(&base, &exp)| (base, exp * exponent))
            .collect();
        Self { exponents }
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
    /// use thales::dimensions::{BaseDimension, Dimension};
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

/// Return the SI symbol string for a base dimension.
pub(super) fn base_dimension_symbol(base: BaseDimension) -> &'static str {
    match base {
        BaseDimension::Length => "m",
        BaseDimension::Mass => "kg",
        BaseDimension::Time => "s",
        BaseDimension::Current => "A",
        BaseDimension::Temperature => "K",
        BaseDimension::Amount => "mol",
        BaseDimension::Luminosity => "cd",
    }
}

/// Format an integer exponent as Unicode superscript characters.
///
/// Returns an empty string for exponent 1 (conventional omission).
pub(super) fn format_superscript(exp: i32) -> String {
    if exp == 1 {
        return String::new();
    }
    let digits: &[char] = &['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    let minus = '⁻';
    let (sign, abs_exp) = if exp < 0 {
        (true, (-exp) as u32)
    } else {
        (false, exp as u32)
    };
    let mut result = String::new();
    if sign {
        result.push(minus);
    }
    for ch in abs_exp.to_string().chars() {
        let digit = ch.to_digit(10).expect("digit") as usize;
        result.push(digits[digit]);
    }
    result
}

/// Canonical ordering for display: Length, Mass, Time, Current, Temperature, Amount, Luminosity.
pub(super) fn base_dimension_order(base: &BaseDimension) -> u8 {
    match base {
        BaseDimension::Length => 0,
        BaseDimension::Mass => 1,
        BaseDimension::Time => 2,
        BaseDimension::Current => 3,
        BaseDimension::Temperature => 4,
        BaseDimension::Amount => 5,
        BaseDimension::Luminosity => 6,
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.exponents.is_empty() {
            return write!(f, "1");
        }
        let mut pairs: Vec<(BaseDimension, i32)> =
            self.exponents.iter().map(|(&b, &e)| (b, e)).collect();
        pairs.sort_by_key(|(b, _)| base_dimension_order(b));
        let mut parts = pairs.iter().map(|(base, exp)| {
            format!(
                "{}{}",
                base_dimension_symbol(*base),
                format_superscript(*exp)
            )
        });
        write!(f, "{}", parts.next().unwrap_or_default())?;
        for part in parts {
            write!(f, "·{}", part)?;
        }
        Ok(())
    }
}
