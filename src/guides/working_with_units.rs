//! # Working with Units and Dimensional Analysis
//!
//! This guide covers the unit and dimension system in thales, enabling type-safe
//! physical calculations with automatic unit conversions and dimensional consistency
//! checking. Learn how to create quantities with units, perform conversions, and
//! leverage dimensional analysis to catch errors at runtime.
//!
//! ## Overview of Dimensional Analysis
//!
//! Dimensional analysis is a fundamental technique in physics and engineering that
//! tracks the physical dimensions (length, mass, time, etc.) through calculations.
//! The thales library provides a robust dimensional analysis system based on the
//! International System of Units (SI).
//!
//! **Key Benefits:**
//! - Catch dimensional inconsistencies before they cause errors
//! - Perform automatic unit conversions between compatible units
//! - Support both linear units (meters, kilograms) and affine units (Celsius, Fahrenheit)
//! - Build custom unit registries for specific domains
//!
//! **Core Concepts:**
//! - **[`BaseDimension`]**: Seven SI base dimensions (length, mass, time, current, temperature, amount, luminosity)
//! - **[`Dimension`]**: Composite dimensions formed from base dimensions with exponents
//! - **[`Unit`]**: Specific measurement standards with conversion factors to SI base units
//! - **[`Quantity`]**: Numeric values paired with their units
//! - **[`UnitRegistry`]**: Collections of units for lookup and conversion
//!
//! [`BaseDimension`]: crate::dimensions::BaseDimension
//! [`Dimension`]: crate::dimensions::Dimension
//! [`Unit`]: crate::dimensions::Unit
//! [`Quantity`]: crate::dimensions::Quantity
//! [`UnitRegistry`]: crate::dimensions::UnitRegistry
//!
//! ## Creating Quantities with Units
//!
//! ### Basic Length Measurement
//!
//! Start by creating a simple length quantity:
//!
//! ```rust,ignore
//! use thales::{Dimension, Unit, Quantity};
//! use thales::dimensions::BaseDimension;
//!
//! // Create length dimension: L¹
//! let length_dim = Dimension::from_base(BaseDimension::Length, 1);
//!
//! // Create meter unit (SI base unit for length)
//! let meter = Unit::new("meter", "m", length_dim, 1.0);
//!
//! // Create a quantity: 5.5 meters
//! let distance = Quantity::new(5.5, meter);
//! println!("{}", distance);  // "5.5 m"
//! ```
//!
//! ### Multiple Units for the Same Dimension
//!
//! Different units can share the same dimension:
//!
//! ```rust,ignore
//! use thales::{Dimension, Unit, Quantity};
//! use thales::dimensions::BaseDimension;
//!
//! let length_dim = Dimension::from_base(BaseDimension::Length, 1);
//!
//! // SI base unit: meter
//! let meter = Unit::new("meter", "m", length_dim.clone(), 1.0);
//!
//! // Derived units: kilometer and mile
//! let kilometer = Unit::new("kilometer", "km", length_dim.clone(), 1000.0);
//! let mile = Unit::new("mile", "mi", length_dim.clone(), 1609.344);
//!
//! // Create quantities with different units
//! let dist_m = Quantity::new(100.0, meter);      // 100 m
//! let dist_km = Quantity::new(5.0, kilometer);   // 5 km
//! let dist_mi = Quantity::new(3.0, mile);        // 3 mi
//! ```
//!
//! ## Unit Conversions
//!
//! ### Converting Between Compatible Units
//!
//! Units with the same dimension can be converted between each other:
//!
//! ```rust,ignore
//! use thales::{Dimension, Unit};
//! use thales::dimensions::BaseDimension;
//!
//! let length_dim = Dimension::from_base(BaseDimension::Length, 1);
//! let kilometer = Unit::new("kilometer", "km", length_dim.clone(), 1000.0);
//! let meter = Unit::new("meter", "m", length_dim.clone(), 1.0);
//!
//! // Convert 5 kilometers to meters
//! let meters = kilometer.convert_to(5.0, &meter).unwrap();
//! assert_eq!(meters, 5000.0);
//! ```
//!
//! ### Quantity Conversions
//!
//! Convert entire quantities to different units:
//!
//! ```rust,ignore
//! use thales::{Dimension, Unit, Quantity};
//! use thales::dimensions::BaseDimension;
//!
//! let time_dim = Dimension::from_base(BaseDimension::Time, 1);
//! let second = Unit::new("second", "s", time_dim.clone(), 1.0);
//! let minute = Unit::new("minute", "min", time_dim.clone(), 60.0);
//!
//! // Create quantity in seconds
//! let duration_s = Quantity::new(300.0, second);
//!
//! // Convert to minutes
//! let duration_min = duration_s.convert_to(&minute).unwrap();
//! assert_eq!(duration_min.value, 5.0);
//! assert_eq!(duration_min.unit.symbol, "min");
//! ```
//!
//! ### Affine Unit Conversions (Temperature)
//!
//! Temperature scales require special handling due to offset conversions:
//!
//! ```rust,ignore
//! use thales::{Dimension, Unit};
//! use thales::dimensions::BaseDimension;
//!
//! let temp_dim = Dimension::from_base(BaseDimension::Temperature, 1);
//!
//! // Kelvin (SI base unit, no offset)
//! let kelvin = Unit::new("kelvin", "K", temp_dim.clone(), 1.0);
//!
//! // Celsius: K = °C + 273.15
//! let celsius = Unit::with_offset("celsius", "°C", temp_dim.clone(), 1.0, 273.15);
//!
//! // Fahrenheit: K = (°F + 459.67) × 5/9
//! let fahrenheit = Unit::with_offset(
//!     "fahrenheit", "°F", temp_dim.clone(), 5.0/9.0, 459.67 * 5.0/9.0
//! );
//!
//! // Convert 0°C to Kelvin
//! let k = celsius.convert_to(0.0, &kelvin).unwrap();
//! assert_eq!(k, 273.15);
//!
//! // Convert 32°F to Celsius
//! let c = fahrenheit.convert_to(32.0, &celsius).unwrap();
//! assert!((c - 0.0).abs() < 0.01);  // 32°F = 0°C
//! ```
//!
//! ## Dimensional Consistency Checking
//!
//! ### Preventing Incompatible Conversions
//!
//! The type system prevents converting between incompatible dimensions:
//!
//! ```rust,ignore
//! use thales::{Dimension, Unit};
//! use thales::dimensions::BaseDimension;
//!
//! let length_dim = Dimension::from_base(BaseDimension::Length, 1);
//! let time_dim = Dimension::from_base(BaseDimension::Time, 1);
//!
//! let meter = Unit::new("meter", "m", length_dim, 1.0);
//! let second = Unit::new("second", "s", time_dim, 1.0);
//!
//! // This will fail: can't convert length to time
//! let result = meter.convert_to(5.0, &second);
//! assert!(result.is_err());
//! ```
//!
//! ### Checking Dimension Compatibility
//!
//! Verify dimension compatibility before attempting operations:
//!
//! ```rust,ignore
//! use thales::{Dimension};
//! use thales::dimensions::BaseDimension;
//!
//! let length1 = Dimension::from_base(BaseDimension::Length, 1);
//! let length2 = Dimension::from_base(BaseDimension::Length, 1);
//! let area = Dimension::from_base(BaseDimension::Length, 2);
//!
//! assert!(length1.is_compatible(&length2));  // Both are L¹
//! assert!(!length1.is_compatible(&area));    // L¹ ≠ L²
//! ```
//!
//! ## The UnitRegistry
//!
//! ### Creating a Custom Registry
//!
//! Build a registry with units relevant to your domain:
//!
//! ```rust,ignore
//! use thales::{Dimension, Unit, UnitRegistry};
//! use thales::dimensions::BaseDimension;
//!
//! let mut registry = UnitRegistry::new();
//!
//! // Add length units
//! let length_dim = Dimension::from_base(BaseDimension::Length, 1);
//! registry.add_unit(Unit::new("meter", "m", length_dim.clone(), 1.0));
//! registry.add_unit(Unit::new("kilometer", "km", length_dim.clone(), 1000.0));
//! registry.add_unit(Unit::new("foot", "ft", length_dim.clone(), 0.3048));
//!
//! // Add mass units
//! let mass_dim = Dimension::from_base(BaseDimension::Mass, 1);
//! registry.add_unit(Unit::new("kilogram", "kg", mass_dim.clone(), 1.0));
//! registry.add_unit(Unit::new("pound", "lb", mass_dim.clone(), 0.453592));
//!
//! // Convert using symbol lookup
//! let feet = registry.convert(100.0, "m", "ft").unwrap();
//! assert!((feet - 328.084).abs() < 0.001);
//! ```
//!
//! ### Looking Up Units
//!
//! Retrieve units by their symbols:
//!
//! ```rust,ignore
//! use thales::{Dimension, Unit, UnitRegistry};
//! use thales::dimensions::BaseDimension;
//!
//! let mut registry = UnitRegistry::new();
//! let length_dim = Dimension::from_base(BaseDimension::Length, 1);
//! registry.add_unit(Unit::new("meter", "m", length_dim, 1.0));
//!
//! // Look up the meter unit
//! let meter = registry.get("m");
//! assert!(meter.is_some());
//! assert_eq!(meter.unwrap().name, "meter");
//!
//! // Non-existent units return None
//! assert!(registry.get("xyz").is_none());
//! ```
//!
//! ## Common Physical Units
//!
//! ### Derived Units: Velocity
//!
//! Velocity has dimension L¹T⁻¹ (length per time):
//!
//! ```rust,ignore
//! use thales::{Dimension, Unit};
//! use thales::dimensions::BaseDimension;
//!
//! // Create velocity dimension: Length / Time
//! let length_dim = Dimension::from_base(BaseDimension::Length, 1);
//! let time_dim = Dimension::from_base(BaseDimension::Time, -1);
//! let velocity_dim = length_dim.multiply(&time_dim);
//!
//! // Meters per second (m/s)
//! let mps = Unit::new("meter per second", "m/s", velocity_dim.clone(), 1.0);
//!
//! // Kilometers per hour (km/h): 1 km/h = 1000/3600 m/s
//! let kph = Unit::new("kilometer per hour", "km/h", velocity_dim.clone(), 1000.0/3600.0);
//!
//! // Convert 100 km/h to m/s
//! let speed_mps = kph.convert_to(100.0, &mps).unwrap();
//! assert!((speed_mps - 27.778).abs() < 0.001);
//! ```
//!
//! ### Derived Units: Force
//!
//! Force has dimension M¹L¹T⁻² (mass × acceleration):
//!
//! ```rust,ignore
//! use thales::{Dimension, Unit};
//! use thales::dimensions::BaseDimension;
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
//! // Dyne (CGS): g⋅cm/s² = 10⁻⁵ N
//! let dyne = Unit::new("dyne", "dyn", force_dim.clone(), 1e-5);
//!
//! // Convert 1 N to dynes
//! let force_dynes = newton.convert_to(1.0, &dyne).unwrap();
//! assert!((force_dynes - 100000.0).abs() < 0.01);
//! ```
//!
//! ### Derived Units: Energy
//!
//! Energy has dimension M¹L²T⁻² (force × distance):
//!
//! ```rust,ignore
//! use thales::{Dimension, Unit};
//! use thales::dimensions::BaseDimension;
//!
//! // Energy dimension: Mass × Length² × Time⁻²
//! let mass_dim = Dimension::from_base(BaseDimension::Mass, 1);
//! let length_dim = Dimension::from_base(BaseDimension::Length, 2);
//! let time_dim = Dimension::from_base(BaseDimension::Time, -2);
//! let energy_dim = mass_dim.multiply(&length_dim).multiply(&time_dim);
//!
//! // Joule: kg⋅m²/s²
//! let joule = Unit::new("joule", "J", energy_dim.clone(), 1.0);
//!
//! // Calorie: 1 cal = 4.184 J
//! let calorie = Unit::new("calorie", "cal", energy_dim.clone(), 4.184);
//!
//! // Convert 100 calories to joules
//! let energy_j = calorie.convert_to(100.0, &joule).unwrap();
//! assert!((energy_j - 418.4).abs() < 0.001);
//! ```
//!
//! ## Custom Unit Definitions
//!
//! ### Creating Domain-Specific Units
//!
//! Define units specific to your application domain:
//!
//! ```rust,ignore
//! use thales::{Dimension, Unit, UnitRegistry};
//! use thales::dimensions::BaseDimension;
//!
//! let mut physics_registry = UnitRegistry::new();
//!
//! // Astronomy: light-year (distance)
//! let length_dim = Dimension::from_base(BaseDimension::Length, 1);
//! let light_year = Unit::new(
//!     "light-year",
//!     "ly",
//!     length_dim.clone(),
//!     9.461e15  // meters per light-year
//! );
//! physics_registry.add_unit(light_year);
//!
//! // Particle physics: electronvolt (energy)
//! let mass_dim = Dimension::from_base(BaseDimension::Mass, 1);
//! let length_dim_sq = Dimension::from_base(BaseDimension::Length, 2);
//! let time_dim_neg2 = Dimension::from_base(BaseDimension::Time, -2);
//! let energy_dim = mass_dim.multiply(&length_dim_sq).multiply(&time_dim_neg2);
//!
//! let electronvolt = Unit::new(
//!     "electronvolt",
//!     "eV",
//!     energy_dim,
//!     1.602176634e-19  // joules per eV
//! );
//! physics_registry.add_unit(electronvolt);
//! ```
//!
//! ### Imperial and US Customary Units
//!
//! ```rust,ignore
//! use thales::{Dimension, Unit, UnitRegistry};
//! use thales::dimensions::BaseDimension;
//!
//! let mut imperial_registry = UnitRegistry::new();
//!
//! let length_dim = Dimension::from_base(BaseDimension::Length, 1);
//!
//! // Imperial length units
//! imperial_registry.add_unit(Unit::new("inch", "in", length_dim.clone(), 0.0254));
//! imperial_registry.add_unit(Unit::new("foot", "ft", length_dim.clone(), 0.3048));
//! imperial_registry.add_unit(Unit::new("yard", "yd", length_dim.clone(), 0.9144));
//! imperial_registry.add_unit(Unit::new("mile", "mi", length_dim.clone(), 1609.344));
//!
//! // Convert 1 mile to feet
//! let feet = imperial_registry.convert(1.0, "mi", "ft").unwrap();
//! assert_eq!(feet, 5280.0);
//! ```
//!
//! ## Working with SI Base Units
//!
//! The seven SI base units form the foundation of the unit system:
//!
//! ```rust,ignore
//! use thales::{Dimension, Unit, UnitRegistry};
//! use thales::dimensions::BaseDimension;
//!
//! let mut si_registry = UnitRegistry::new();
//!
//! // Length: meter (m)
//! let meter = Unit::new(
//!     "meter", "m",
//!     Dimension::from_base(BaseDimension::Length, 1),
//!     1.0
//! );
//!
//! // Mass: kilogram (kg)
//! let kilogram = Unit::new(
//!     "kilogram", "kg",
//!     Dimension::from_base(BaseDimension::Mass, 1),
//!     1.0
//! );
//!
//! // Time: second (s)
//! let second = Unit::new(
//!     "second", "s",
//!     Dimension::from_base(BaseDimension::Time, 1),
//!     1.0
//! );
//!
//! // Electric current: ampere (A)
//! let ampere = Unit::new(
//!     "ampere", "A",
//!     Dimension::from_base(BaseDimension::Current, 1),
//!     1.0
//! );
//!
//! // Temperature: kelvin (K)
//! let kelvin = Unit::new(
//!     "kelvin", "K",
//!     Dimension::from_base(BaseDimension::Temperature, 1),
//!     1.0
//! );
//!
//! // Amount of substance: mole (mol)
//! let mole = Unit::new(
//!     "mole", "mol",
//!     Dimension::from_base(BaseDimension::Amount, 1),
//!     1.0
//! );
//!
//! // Luminous intensity: candela (cd)
//! let candela = Unit::new(
//!     "candela", "cd",
//!     Dimension::from_base(BaseDimension::Luminosity, 1),
//!     1.0
//! );
//!
//! // Add all to registry
//! si_registry.add_unit(meter);
//! si_registry.add_unit(kilogram);
//! si_registry.add_unit(second);
//! si_registry.add_unit(ampere);
//! si_registry.add_unit(kelvin);
//! si_registry.add_unit(mole);
//! si_registry.add_unit(candela);
//! ```
//!
//! ## Best Practices
//!
//! ### Always Use Type-Safe Quantities
//!
//! Prefer [`Quantity`] over raw numbers to maintain dimensional safety:
//!
//! ```rust,ignore
//! // Good: Type-safe with units
//! let distance = Quantity::new(100.0, meter);
//!
//! // Risky: Raw number with implicit units
//! let distance = 100.0;  // Is this meters? feet? miles?
//! ```
//!
//! ### Check Compatibility Before Operations
//!
//! Validate dimensional compatibility before calculations:
//!
//! ```rust,ignore
//! use thales::{Dimension, Unit};
//! use thales::dimensions::BaseDimension;
//!
//! let length = Dimension::from_base(BaseDimension::Length, 1);
//! let time = Dimension::from_base(BaseDimension::Time, 1);
//!
//! // Check before attempting conversion
//! if !length.is_compatible(&time) {
//!     println!("Cannot convert between length and time");
//! }
//! ```
//!
//! ### Use SI Base Units for Calculations
//!
//! Convert to SI base units before performing calculations:
//!
//! ```rust,ignore
//! use thales::{Dimension, Unit, Quantity};
//! use thales::dimensions::BaseDimension;
//!
//! let length_dim = Dimension::from_base(BaseDimension::Length, 1);
//! let kilometer = Unit::new("kilometer", "km", length_dim, 1000.0);
//!
//! let distance = Quantity::new(5.0, kilometer);
//!
//! // Convert to SI base unit (meters) for calculation
//! let si_value = distance.to_si();
//! assert_eq!(si_value, 5000.0);  // 5 km = 5000 m
//! ```
//!
//! ## See Also
//!
//! - [`crate::dimensions`] - Complete API documentation for the dimensions module
//! - [`solving_equations`](crate::guides::solving_equations) - Using units in equation solving
//! - [`calculus_operations`](crate::guides::calculus_operations) - Dimensional analysis in calculus
