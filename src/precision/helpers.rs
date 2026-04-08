//! Rounding and precision helper functions.

use super::types::RoundingMode;

/// Round a number to a specific number of decimal places.
pub(super) fn round_to_decimal(value: f64, places: u32, mode: RoundingMode) -> f64 {
    let factor = 10_f64.powi(places as i32);
    let scaled = value * factor;
    let rounded = apply_rounding(scaled, mode);
    rounded / factor
}

/// Round a number to a specific number of significant figures.
pub(super) fn round_to_sig_figs(value: f64, figures: u32, mode: RoundingMode) -> f64 {
    if value == 0.0 {
        return 0.0;
    }
    let magnitude = value.abs().log10().floor() as i32;
    let scale = 10_f64.powi(figures as i32 - 1 - magnitude);
    let scaled = value * scale;
    let rounded = apply_rounding(scaled, mode);
    rounded / scale
}

/// Apply rounding mode to a value.
pub(super) fn apply_rounding(value: f64, mode: RoundingMode) -> f64 {
    match mode {
        RoundingMode::HalfUp => {
            let floor = value.floor();
            if value - floor >= 0.5 {
                floor + 1.0
            } else {
                floor
            }
        }
        RoundingMode::HalfEven => {
            let floor = value.floor();
            let frac = value - floor;
            if frac > 0.5 || (frac == 0.5 && floor as i64 % 2 != 0) {
                floor + 1.0
            } else {
                floor
            }
        }
        RoundingMode::Truncate => value.trunc(),
        RoundingMode::Ceiling => value.ceil(),
        RoundingMode::Floor => value.floor(),
    }
}
