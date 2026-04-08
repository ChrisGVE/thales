//! Units and dimensional analysis.
//!
//! Provides dimensional analysis capabilities with SI base units,
//! unit conversions, and physical quantity arithmetic.

mod quantity;
mod registry;
mod types;
mod unit;

pub use quantity::Quantity;
pub use registry::UnitRegistry;
pub use types::{BaseDimension, Dimension};
pub use unit::Unit;

#[cfg(test)]
mod tests {
    use super::*;

    fn length() -> Dimension {
        Dimension::from_base(BaseDimension::Length, 1)
    }

    fn time() -> Dimension {
        Dimension::from_base(BaseDimension::Time, 1)
    }

    fn mass() -> Dimension {
        Dimension::from_base(BaseDimension::Mass, 1)
    }

    #[test]
    fn multiply_adds_exponents() {
        // Length × Length = Length²
        let area = length().multiply(&length());
        assert_eq!(area.exponents[&BaseDimension::Length], 2);
        assert_eq!(area.exponents.len(), 1);
    }

    #[test]
    fn multiply_velocity_is_length_per_time() {
        // velocity = Length / Time = Length × Time⁻¹
        let inv_time = Dimension::from_base(BaseDimension::Time, -1);
        let velocity = length().multiply(&inv_time);
        assert_eq!(velocity.exponents[&BaseDimension::Length], 1);
        assert_eq!(velocity.exponents[&BaseDimension::Time], -1);
    }

    #[test]
    fn divide_velocity() {
        // velocity = length / time
        let velocity = length().divide(&time());
        assert_eq!(velocity.exponents[&BaseDimension::Length], 1);
        assert_eq!(velocity.exponents[&BaseDimension::Time], -1);
    }

    #[test]
    fn divide_acceleration() {
        // acceleration = velocity / time = length / time²
        let velocity = length().divide(&time());
        let acceleration = velocity.divide(&time());
        assert_eq!(acceleration.exponents[&BaseDimension::Length], 1);
        assert_eq!(acceleration.exponents[&BaseDimension::Time], -2);
    }

    #[test]
    fn divide_dimensionless_cancellation() {
        // length / length = dimensionless
        let result = length().divide(&length());
        assert!(result.is_dimensionless());
    }

    #[test]
    fn power_velocity_squared() {
        // velocity² = Length²·Time⁻²
        let velocity = length().divide(&time());
        let v2 = velocity.power(2);
        assert_eq!(v2.exponents[&BaseDimension::Length], 2);
        assert_eq!(v2.exponents[&BaseDimension::Time], -2);
    }

    #[test]
    fn power_zero_is_dimensionless() {
        assert!(length().power(0).is_dimensionless());
    }

    #[test]
    fn energy_is_mass_times_velocity_squared() {
        // E = m·v² = Mass × Length²·Time⁻²
        let velocity = length().divide(&time());
        let energy = mass().multiply(&velocity.power(2));
        assert_eq!(energy.exponents[&BaseDimension::Mass], 1);
        assert_eq!(energy.exponents[&BaseDimension::Length], 2);
        assert_eq!(energy.exponents[&BaseDimension::Time], -2);
    }

    #[test]
    fn display_dimensionless() {
        assert_eq!(Dimension::dimensionless().to_string(), "1");
    }

    #[test]
    fn display_length() {
        assert_eq!(length().to_string(), "m");
    }

    #[test]
    fn display_velocity() {
        let velocity = length().divide(&time());
        assert_eq!(velocity.to_string(), "m·s⁻¹");
    }

    #[test]
    fn display_acceleration() {
        let accel = length().divide(&time().multiply(&time()));
        assert_eq!(accel.to_string(), "m·s⁻²");
    }

    #[test]
    fn display_energy() {
        let velocity = length().divide(&time());
        let energy = mass().multiply(&velocity.power(2));
        assert_eq!(energy.to_string(), "m²·kg·s⁻²");
    }
}
